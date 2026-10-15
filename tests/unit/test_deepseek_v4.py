"""deepseek_v4 (deepseek-ai/DeepSeek-V4-Flash-0731): lossless FP8 / MXFP4 source dequantization, the adapter on the REAL
checkpoint layout (all 48 safetensors headers, fixture), the MLX model against an independent numpy port of the
reference prefill (tests/unit/ds4_numpy_ref.py) and its own cached decode, and `suc compress` end to end with the
per-expert Fisher allocation under a total-size budget -> SUC1 -> Python runtime from the codes. No torch: the torch
check against the official inference/model.py is scripts/deepseek_v4_reference_check.py (Mac Studio)."""
import gzip
import json
import math
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import yaml
from mlx.utils import tree_flatten

from suc.io import quantized_source as qs
from suc.io.safetensors_store import SafetensorsStore, TensorInfo
from suc.models import mlx_deepseek_v4 as ds4
from suc.models.adapters import DeepseekV4Adapter, get_adapter
from suc.models.streaming import load_lazy
from tiny_models import tiny_deepseek_v4_config, write_tiny_deepseek_v4

FIX = Path(__file__).parent / "fixtures"
REAL_CFG = json.loads((FIX / "deepseek_v4_flash_config.json").read_text())
REAL = json.loads(gzip.open(FIX / "deepseek_v4_flash_tensors.json.gz").read())
_ITEM = {"BF16": 2, "I64": 8, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}
EXPERT_ROLES = ("expert_gate", "expert_up", "expert_down")


def _fake_store(tensors: dict) -> SafetensorsStore:
    infos, off = {}, 0
    for n, (dt, shape) in sorted(tensors.items()):
        nb = math.prod(shape) * _ITEM[dt]
        infos[n] = TensorInfo(n, dt, tuple(shape), Path("/nonexistent"), off, nb)
        off += nb
    return SafetensorsStore(infos)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return write_tiny_deepseek_v4(tmp_path_factory.mktemp("ds4") / "model")


def _f32_model(d, sims: bool = False):
    model, _ = load_lazy(d)
    model.set_dtype(mx.float32)
    for layer in model.layers:
        for m in layer.modules():
            for k in ("_sim_kv", "_sim_fp4"):
                if hasattr(m, k):
                    setattr(m, k, sims)
    return model


# ------------------------------------------------------------------ source dequantization (D8 exception)

def test_dequant_is_bit_exact_vs_numpy_reference():
    rng = np.random.default_rng(0)
    w8 = rng.integers(0, 256, (300, 260), dtype=np.uint8)
    w8[(w8 & 0x7F) == 0x7F] = 0x11                                   # no NaN codes
    s8 = rng.integers(100, 140, (3, 3), dtype=np.uint8)              # ceil(300/128) x ceil(260/128): partial blocks
    got = np.array(qs.dequant_fp8_block(mx.array(w8), mx.array(s8)).astype(mx.float32))
    assert np.array_equal(got.astype(np.float64), qs.np_dequant_fp8_block(w8, s8))
    w4 = rng.integers(-128, 128, (64, 96), dtype=np.int8)
    s4 = rng.integers(110, 140, (64, 6), dtype=np.uint8)
    got = np.array(qs.dequant_mxfp4(mx.array(w4), mx.array(s4)).astype(mx.float32))
    assert np.array_equal(got.astype(np.float64), qs.np_dequant_mxfp4(w4, s4))
    # every finite E4M3 byte decodes exactly (MLX from_fp8 vs the bit-field table)
    t = qs.e4m3_table()
    fin = np.isfinite(t)
    dec = np.array(mx.from_fp8(mx.array(np.arange(256, dtype=np.uint8)), dtype=mx.float32)).astype(np.float64)
    assert np.array_equal(dec[fin], t[fin])
    # MLX's own mxfp4 layout is byte-compatible (doc §4): same values as the explicit unpack
    q = mx.array(w4).view(mx.uint8).reshape(64, -1).view(mx.uint32)
    ref = mx.dequantize(q, mx.array(s4), group_size=32, bits=4, mode="mxfp4").astype(mx.float32)
    assert np.array_equal(np.array(ref), got)


def test_scale_range_guards_losslessness():
    ok = mx.array(np.array([[10, 245]], dtype=np.uint8))
    assert qs.check_scales("fp8_block", ok) == (10, 245)
    for bad in (9, 246, 255):
        with pytest.raises(qs.NotLossless):
            qs.check_scales("fp8_block", mx.array(np.array([[bad]], dtype=np.uint8)), "w")
    with pytest.raises(qs.NotLossless):
        qs.check_scales("mxfp4", mx.array(np.array([[255]], dtype=np.uint8)))


def test_store_dequantizes_the_tiny_checkpoint_exactly(tiny_dir):
    hf = SafetensorsStore.from_dir(tiny_dir)
    ad = get_adapter(json.loads((tiny_dir / "config.json").read_text()))
    ad.validate(hf.names())
    st = ad.source_store(hf)
    ad.validate(st.names())                                          # the SUC namespace validates too
    assert not [n for n in st.names() if n.startswith("mtp.") or (n.endswith(".scale") and ".hc_" not in n
                                                                   and not n.startswith("model.hc_head."))]
    for suc_name, ref_name in (("model.layers.1.self_attn.wq_b.weight", "layers.1.attn.wq_b"),
                               ("model.layers.2.mlp.experts.5.down_proj.weight", "layers.2.ffn.experts.5.w2"),
                               ("model.layers.0.mlp.shared_experts.up_proj.weight", "layers.0.ffn.shared_experts.w3")):
        w = np.array(hf.load(ref_name + ".weight"))
        s = np.array(hf.load(ref_name + ".scale"))
        f = qs.np_dequant_mxfp4 if hf.info(ref_name + ".weight").dtype == "I8" else qs.np_dequant_fp8_block
        got = st.load(suc_name)
        assert got.dtype == mx.bfloat16 and st.info(suc_name).dtype == "BF16"
        assert tuple(got.shape) == st.info(suc_name).shape
        assert np.array_equal(np.array(got.astype(mx.float32)).astype(np.float64), f(w, s))
    ranges = st.check_quant_scales()
    assert set(ranges) == {"fp8_block", "mxfp4"} and ranges["mxfp4"]["tensors"] == 4 * 8 * 3
    # the lazily loaded MLX model (sanitize) holds exactly the store's dequantized tensors
    model, _ = load_lazy(tiny_dir)
    params = dict(tree_flatten(model.parameters()))
    for n in st.names():
        if (m := re.match(r"^(.*)\.experts\.(\d+)\.(\w+)\.weight$", n)) is not None:
            p = params[f"{m.group(1)}.switch_mlp.{m.group(3)}.weight"][int(m.group(2))]
        else:
            p = params[n]
        a = st.load(n)
        assert np.array_equal(np.array(p.astype(mx.float32)), np.array(a.astype(mx.float32))), n


# ------------------------------------------------------------------ adapter on the real checkpoint layout

def test_adapter_on_the_real_checkpoint():
    names = list(REAL["tensors"])
    assert len(names) == 72317
    assert sum(math.prod(s) * _ITEM[d] for d, s in REAL["tensors"].values()) == REAL["index_total_size"]
    ad = get_adapter(REAL_CFG)
    assert isinstance(ad, DeepseekV4Adapter) and ad.num_layers == 43 and ad.num_experts == 256
    ad.validate(names)
    st = ad.source_store(_fake_store(REAL["tensors"]))
    ad.validate(st.names())
    assert not [n for n in st.names() if n.startswith("mtp.") or not (n.startswith("model.") or n == "lm_head.weight")]
    assert len([n for n in st.names() if n.endswith(".scale")]) == 2 * 43 + 1     # hc_{attn,ffn}.scale, hc_head.scale
    assert len(st.quant_names()) == 43 * (5 + 3 + 3 * 256) + 21
    specs = [s for s in ad.linears(st.names()) if s.eligible]
    roles = {}
    for s in specs:
        roles[s.role] = roles.get(s.role, 0) + 1
    assert roles["expert_gate"] == roles["expert_up"] == roles["expert_down"] == 43 * 256
    assert roles["attn_o_a"] == 43 and "indexer_q_b" not in roles and "compressor_kv" not in roles
    elig = sum(math.prod(st.info(s.name).shape) for s in specs)
    experts = sum(math.prod(st.info(s.name).shape) for s in specs if s.role in EXPERT_ROLES)
    assert experts == 277_025_390_592 and elig == 282_706_575_360    # docs/arch/deepseek_v4.md §5 / §6
    e = st.info("model.layers.7.mlp.experts.200.gate_proj.weight")
    assert e.shape == (2048, 4096) and e.dtype == "BF16" and st.quant_source(e.name).kind == "mxfp4"
    assert st.info("model.layers.7.mlp.experts.200.down_proj.weight").shape == (4096, 2048)
    assert st.quant_source("model.layers.3.self_attn.wo_a.weight").kind == "fp8_block"
    assert st.info("model.layers.1.mlp.gate.tid2eid").dtype == "I64"
    with pytest.raises(ValueError, match="unclassified"):
        ad.validate(names + ["layers.0.attn.mystery"])
    broken = [n for n in names if n != "layers.4.ffn.gate.bias"]
    with pytest.raises(ValueError, match="e_score_correction_bias"):
        ad.validate(broken)
    assert ad.kv_bytes_per_token() == 6880                           # doc §9.8: ~6.9 KB per token


def test_mlx_parameters_map_one_to_one_onto_the_real_tensors():
    """Every MLX parameter of the real config (lazy, never materialized) has exactly the shape of its SUC-namespace
    tensor (expert stacks = the stacked per-expert tensors), and every SUC tensor is a parameter."""
    ad = get_adapter(REAL_CFG)
    st = ad.source_store(_fake_store(REAL["tensors"]))
    model = ds4.Model(ds4.ModelArgs.from_dict(REAL_CFG))
    params = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    want = {}
    for n in st.names():
        if m := re.match(r"^(.*)\.experts\.(\d+)\.(\w+)\.weight$", n):
            want[f"{m.group(1)}.switch_mlp.{m.group(3)}.weight"] = (ad.num_experts, *st.info(n).shape)
        else:
            want[n] = st.info(n).shape
    assert params == want
    assert ad.residual_width == 4 * 4096 + 3


# ------------------------------------------------------------------ MLX model numerics

def test_mlx_matches_the_numpy_reference_port(tiny_dir):
    from ds4_numpy_ref import Ref, load_reference_weights

    cfg = json.loads((tiny_dir / "config.json").read_text())
    ref = Ref(cfg, load_reference_weights(tiny_dir))
    tok = np.random.default_rng(0).integers(2, 300, 61)              # CSA: 15 entries > top-4; HCA (16): 3 entries
    outs, logits = ref.forward(tok)
    model = _f32_model(tiny_dir)
    h = model.embed_stream(mx.array(tok[None]))
    for i, layer in enumerate(model.layers):
        h = layer(h)
        got = np.array(h[0, :, : 4 * 256]).reshape(len(tok), 4, 256)
        assert np.linalg.norm(got - outs[i]) / np.linalg.norm(outs[i]) < 1e-5, i
    lg = np.array(model.head_stream(h)[0])
    assert np.abs(lg - logits).max() < 1e-4
    # the indexer selection matters at this length: without it the logits change
    for layer in model.layers:
        if "indexer" in layer.self_attn:
            layer.self_attn.indexer.topk = 10 ** 6
    assert np.abs(np.array(model(mx.array(tok[None]))[0]) - logits).max() > 1e-2


def test_cached_decode_equals_prefill(tiny_dir):
    model = _f32_model(tiny_dir)
    mx.random.seed(3)
    x = mx.random.randint(2, 300, (2, 70))
    full = model(x)
    for split in (1, 13):                                            # window 8, ratios 4 / 16 across chunk borders
        cache = model.make_cache()
        steps = [model(x[:, :split], cache=cache)] + [model(x[:, t:t + 1], cache=cache) for t in range(split, 70)]
        assert mx.abs(mx.concatenate(steps, axis=1) - full).max().item() < 1e-4
    from suc.models.streaming import stream_blocks
    r = stream_blocks(model, x, return_logits=True, release=False, micro_batch=2)
    assert mx.array_equal(r["logits"], full)                         # identical ops and shapes: bit-exact
    sim = _f32_model(tiny_dir, sims=True)                            # with the QAT round trips: rounding flips only
    cache = sim.make_cache()
    inc = mx.concatenate([sim(x[:, :9], cache=cache)] + [sim(x[:, t:t + 1], cache=cache) for t in range(9, 70)], 1)
    ref = sim(x)
    p = mx.softmax(ref, -1)
    assert (p * (mx.log(p) - mx.log(mx.softmax(inc, -1)))).sum(-1).max().item() < 1e-3


def _np_pow2_ceil(v):
    return np.ldexp(1.0, np.ceil(np.log2(v.astype(np.float32).astype(np.float64))).astype(np.int64))


def test_round_trips_follow_the_kernel_semantics():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((5, 256)) * np.exp(rng.standard_normal((5, 1)) * 3)).astype(np.float32)
    x[0, :64] = 0.0                                                   # amax floor 1e-4
    got = np.array(ds4.fp8_round_trip(mx.array(x), 64))
    t = qs.e4m3_table()
    vals = np.unique(np.abs(t[np.isfinite(t)]))
    xb = x.reshape(5, 4, 64).astype(np.float64)
    s = _np_pow2_ceil(np.maximum(np.abs(xb).max(-1, keepdims=True), 1e-4).astype(np.float32) * np.float32(1 / 448))
    y = np.clip(xb / s, -448, 448)
    idx = np.abs(np.abs(y)[..., None] - vals).argmin(-1)              # nearest (no exact ties in random data)
    ref = (np.sign(y) * vals[idx] * s).reshape(5, 256)
    assert np.array_equal(got.astype(np.float64), ref)
    # FP4: nearest E2M1 magnitude, ties to the even code (0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 2.5 -> 2, 5 -> 4)
    y = np.array([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] + [0.1] * 24], dtype=np.float32)
    got = np.array(ds4.fp4_round_trip(mx.array(y), 32))
    assert got[0, :8].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]
    # straight-through: identity gradient
    g = mx.grad(lambda a: ds4.fp8_round_trip(a, 64).sum())(mx.array(x))
    assert mx.all(g == 1).item()


def test_rope_frequencies_match_the_reference_yarn():
    from ds4_numpy_ref import _freqs

    f = ds4.rope_inv_freq(64, 160000, {"factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32,
                                        "beta_slow": 1})
    ref = np.angle(_freqs(64, 2, 65536, 160000, 16, 32, 1)[1])
    assert np.allclose(f, ref, rtol=1e-6)


def test_hash_routing_reads_token_ids_from_the_stream(tiny_dir):
    model, _ = load_lazy(tiny_dir)
    tok = mx.array([[0, 1, 63, 64, 299, 4095 % 300]])
    h = model.embed_stream(tok)
    assert h.shape[-1] == 4 * 256 + 3
    assert mx.array_equal(ds4.tokens_from_stream(h[..., 4 * 256:]), tok.astype(mx.int32))
    big = ds4.ModelArgs.from_dict(tiny_deepseek_v4_config(vocab_size=262143))
    assert big.token_channels == 3
    with pytest.raises(NotImplementedError):
        ds4.ModelArgs.from_dict(tiny_deepseek_v4_config(vocab_size=300_000))


def test_gate_scores_follow_the_reference_score_func():
    """Gate = inference/model.py Gate at the pinned revision (7872f01b): the checkpoint's score_func is sqrtsoftplus
    (config.json scoring_func, inference/config.json score_func), scores = F.softplus(logits).sqrt() in float32 (torch's
    softplus returns x above 20), selection on scores + bias, weights = the unbiased scores of the chosen experts
    normalised and x route_scale. Checked against a numpy port of that formula; the sigmoid / softmax branches too."""
    assert REAL_CFG["scoring_func"] == "sqrtsoftplus" == ds4.ModelArgs.scoring_func
    E, K, D = 16, 4, 32
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, D)).astype(np.float32) * 4   # logits up to ~|40|: both sides of torch's threshold
    w = rng.normal(size=(E, D)).astype(np.float32)
    bias = rng.normal(size=(E,)).astype(np.float32) * 0.5
    for func in ("sqrtsoftplus", "sigmoid", "softmax"):
        a = ds4.ModelArgs.from_dict(tiny_deepseek_v4_config(n_routed_experts=E, num_experts_per_tok=K, hidden_size=D,
                                                            scoring_func=func, num_hash_layers=0))
        g = ds4.Gate(a, layer=0)
        g.weight = mx.array(w)
        g.e_score_correction_bias = mx.array(bias)
        inds, wts = g(mx.array(x), None)
        lg = (x.astype(np.float64) @ w.T.astype(np.float64))
        if func == "sqrtsoftplus":
            s = np.sqrt(np.where(lg > 20, lg, np.log1p(np.exp(np.minimum(lg, 20)))))
        elif func == "sigmoid":
            s = 1 / (1 + np.exp(-lg))
        else:
            s = np.exp(lg - lg.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
        ref_i = np.argsort(-(s + bias), axis=-1, kind="stable")[:, :K]
        got_i = np.array(inds)
        assert (np.sort(got_i, -1) == np.sort(ref_i, -1)).all(), func
        ref_w = np.take_along_axis(s, got_i.astype(np.int64), -1)
        if func != "softmax":
            ref_w = ref_w / ref_w.sum(-1, keepdims=True)
        np.testing.assert_allclose(np.array(wts), ref_w * a.routed_scaling_factor, rtol=2e-5, atol=1e-6)


# ------------------------------------------------------------------ allocation building blocks

def test_per_member_candidates_and_subset_stack_encoding():
    from suc.allocation.greedy import build_candidates
    from suc.quant.codecs import make_codec
    from suc.quant.stacks import encode_stack, set_codebook_scope

    mx.random.seed(0)
    W = {f"model.layers.0.mlp.experts.{e}.up_proj.weight": mx.random.normal((32, 64)) for e in range(4)}
    stack = "model.layers.0.mlp.switch_mlp.up_proj.weight"
    codec = set_codebook_scope(make_codec({"kind": "vq", "k": 16, "block_size": 4, "kmeans_iters": 3}),
                               "expert_stack")
    loss = lambda n, w, wh: float(mx.sum((w - wh) ** 2).item())     # noqa: E731
    units = {stack: list(W)}
    per = build_candidates(W, [stack], {"c": codec}, loss, seed=1, stacks=units, per_member=True)
    tot = build_candidates(W, [stack], {"c": codec}, loss, seed=1, stacks=units)
    assert set(per) == set(W)
    assert sum(per[n][0].bits for n in W) == tot[stack][0].bits
    assert math.isclose(sum(per[n][0].loss for n in W), tot[stack][0].loss, rel_tol=1e-6)
    # encoding a SUBSET against the whole-stack fit reproduces the scored encodings exactly
    full = {n: e for n, _, e in encode_stack(codec, W.__getitem__, list(W), seed=1, stack=stack)}
    sub = list(W)[2:]
    part = {n: e for n, _, e in encode_stack(codec, W.__getitem__, sub, seed=1, stack=stack, fit_names=list(W))}
    for n in sub:
        assert mx.array_equal(part[n].arrays["indices"], full[n].arrays["indices"])
        assert mx.array_equal(codec.decode(part[n]), codec.decode(full[n]))


def test_mixed_switch_matches_per_expert_linears():
    from suc.kernels.suc_switch import SucSwitchMixed, switch_from_encodings
    from suc.quant.codecs import make_codec

    mx.random.seed(1)
    E, out, inn = 6, 32, 64
    ws = [mx.random.normal((out, inn)) * 0.1 for _ in range(E)]
    c4 = make_codec({"kind": "vq", "k": 16, "block_size": 4, "kmeans_iters": 2, "rotate_out": False})
    c2 = make_codec({"kind": "vq", "k": 16, "block_size": 2, "kmeans_iters": 2, "rotate_out": False})
    enc = [(c4 if e % 3 else c2).encode(w, seed=e) for e, w in enumerate(ws)]
    m = switch_from_encodings(enc)
    assert isinstance(m, SucSwitchMixed) and m.members == [[0, 3], [1, 2, 4, 5]]
    x = mx.random.normal((5, 1, 1, inn))
    idx = mx.array([[0, 1], [2, 3], [4, 5], [5, 0], [3, 3]]).astype(mx.uint32)
    y = m(x, idx)
    for b in range(5):
        for j in range(2):
            e = int(idx[b, j].item())
            ref = x[b, 0, 0] @ (c4 if e % 3 else c2).decode(enc[e]).T
            assert mx.abs(y[b, j, 0] - ref).max().item() < 1e-3


# ------------------------------------------------------------------ suc compress -> SUC1 -> runtime

def _cands():
    def vq(g, xs=True):
        c = {"kind": "vq", "k": 16, "block_size": g, "rotate_out": False, "kmeans_iters": 3,
             "residual_fraction": 0.02, "residual_structure": "columns"}
        return dict(c, codebook_scope="expert_stack") if xs else c
    from suc.config import CodecConfig
    c = [vq(4), vq(2), vq(2, False), {"kind": "bf16"}]
    names = [CodecConfig(**x).name for x in c]
    nonexp = ["attn_q_a", "attn_q_b", "attn_kv", "attn_o_a", "attn_o_b", "shared_expert_gate", "shared_expert_up",
              "shared_expert_down"]
    roles = {names[0]: list(EXPERT_ROLES), names[1]: list(EXPERT_ROLES), names[2]: nonexp, names[3]: nonexp}
    return c, names, roles


@pytest.fixture(scope="module")
def compressed(tmp_path_factory):
    import suc.pipeline.compress as comp
    from test_calibrate_run import FakeTokenizer, _docs

    tmp = tmp_path_factory.mktemp("ds4c")
    mx.random.seed(0)
    tokens = mx.random.randint(2, 300, (2048,), dtype=mx.int32)
    saved = comp.load_wikitext2, comp.tokenize
    comp.load_wikitext2 = lambda split: ("unused", "wikihash")
    comp.tokenize = lambda tok, text: tokens
    try:
        docs, ood = _docs()
        model_dir = write_tiny_deepseek_v4(tmp / "model", seed=2)
        cands, names, roles = _cands()
        budget = 4_050_000
        cfg = {"name": "c_ds4", "seed": 7, "output_dir": str(tmp / "results"), "run_dir": str(tmp / "run"),
               "model": {"id": "tiny/deepseek_v4", "local_path": str(model_dir)},
               "calibration": {"seq_len": 32, "n_calibration": 16, "n_validation": 8, "n_ood": 4, "micro_batch": 4},
               "plan": {"mode": "greedy", "budget_bytes": budget, "allocation_unit": "expert", "candidates": cands,
                        "candidate_roles": roles, "embedding_codec": {"kind": "affine", "bits": 4},
                        "fisher_sequences": 8, "fisher_micro_batch": 2},
               "eval": {"seq_len": 64, "max_tokens": 1024, "kl_sequences": 4, "chunk": 2},
               "finetune": None, "rotation_groups": True, "verify_tensors": 8,
               "preflight": {"runtime_budget_bytes": 2.5e7, "runtime_context": 4096,
                             "runtime_activation_bytes": 1e6}}
        p = tmp / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg))
        r = json.loads(comp.run_compress(p, tokenizer=FakeTokenizer(), docs=docs, ood_docs=ood).read_text())
        # the native-MXFP4 variant: same config + one candidate, run on top of this run's calibrate / reference /
        # candidates artifacts (copied into a new run dir), as on the Studio
        import shutil

        nat = dict(cfg, name="c_ds4_native", run_dir=str(tmp / "run_native"),
                   plan=dict(cfg["plan"], candidates=cands + [{"kind": "mxfp4"}],
                             candidate_roles=dict(roles, mxfp4_native=list(EXPERT_ROLES))))
        for st in ("calibrate", "reference", "candidates"):
            shutil.copytree(tmp / "run" / st, tmp / "run_native" / st)
        pn = tmp / "cfg_native.yaml"
        pn.write_text(yaml.safe_dump(nat))
        rn = json.loads(comp.run_compress(pn, tokenizer=FakeTokenizer(), docs=docs, ood_docs=ood).read_text())
    finally:
        comp.load_wikitext2, comp.tokenize = saved
    return r, model_dir, budget, names, rn


def test_compress_budget_and_per_expert_plan(compressed):
    r, _, budget, names, _ = compressed
    p = r["plan"]
    assert r["architecture"]["family"] == "deepseek_v4" and p["allocation_unit"] == "expert"
    assert p["predicted_total_bytes"] <= budget and r["container"]["bytes"] <= budget
    assert r["preflight"]["source_precision"]["lossless"] is True
    assert r["preflight"]["runtime"]["kv_bytes"] == 4096 * get_adapter(tiny_deepseek_v4_config()).kv_bytes_per_token() \
        + 4 * 8 * 128 * 2
    plan = json.loads((Path(r["container"]["path"]).parent.parent / "plan" / "plan.json").read_text())["tensors"]
    for n, c in plan.items():
        role = "expert" if ".mlp.experts." in n else ("head" if n in ("model.embed_tokens.weight", "lm_head.weight")
                                                      else "other")
        if role == "expert":
            assert c in names[:2], (n, c)                             # only the expert codecs (bf16 not allowed)
        elif role == "other":
            assert c in names[2:], (n, c)
    stacks = {}
    for n, c in plan.items():
        if m := re.match(r"^(.*)\.experts\.\d+\.(\w+)\.weight$", n):
            stacks.setdefault((m.group(1), m.group(2)), set()).add(c)
    assert any(len(v) > 1 for v in stacks.values())                   # some stack mixes the two expert codecs
    assert np.isfinite(r["quality"]["kl"]["kl_mean"])


def test_suc1_runtime_from_codes(compressed):
    from suc.analysis.sensitivity import set_weight
    from suc.io.suc1 import SucModel
    from suc.runtime.suc1_model import greedy, load_suc1_model

    r, model_dir, _, _, _ = compressed
    sm = SucModel.load(r["container"]["path"], verify=True)
    assert not [n for n in sm.names() if n.startswith(("mtp.", "layers."))]
    assert sm.descriptor("model.layers.0.hc_attn.fn")["representation_type"] == "raw"
    model, _, rep = load_suc1_model(r["container"]["path"], group=False)
    kinds = {c["kind"] for c in rep["converted"]}
    assert "SucSwitchMixed" in kinds and "SucResidualLinear" in kinds
    assert model.layers[1].hc_ffn.fn.dtype == mx.float32 and model.layers[0].mlp.gate.tid2eid.dtype == mx.int32
    ref, _ = load_lazy(model_dir)
    for n in sm.names():
        if sm.descriptor(n)["representation_type"] != "raw":
            set_weight(ref, n, sm.decode_tensor(n))
    x = mx.random.randint(2, 300, (2, 40))
    lr, lq = ref(x).astype(mx.float32), model(x).astype(mx.float32)
    p = mx.softmax(lr, axis=-1)
    kl = (p * (mx.log(p + 1e-9) - mx.log(mx.softmax(lq, axis=-1) + 1e-9))).sum(-1).mean().item()
    assert kl < 2e-2, kl
    out = greedy(model, [5, 6, 7, 8], 4)
    assert len(out) == 4 and all(0 <= t < 300 for t in out)


# ------------------------------------------------------------------ native MXFP4 expert pass-through

def test_native_mxfp4_codec_and_suc1_are_bit_exact(tmp_path):
    from suc.io.suc1 import SucModel, write_suc1
    from suc.quant.codecs import METADATA_BITS, MXFP4Native, make_codec

    rng = np.random.default_rng(3)
    w = rng.standard_normal((96, 256)).astype(np.float32) * 0.03
    w[5, :32] = 0.0                                                   # an all-zero group
    packed, scale = qs.np_quantize_mxfp4(w)
    packed.view(np.uint8)[7, 3] = 0x88                                # two -0 codes (decode to +0 like the source)
    ref = qs.dequant_mxfp4(mx.array(packed), mx.array(scale))         # the D8 reference (exact BF16)
    c = make_codec({"kind": "mxfp4"})
    e = c.encode_source(mx.array(packed), mx.array(scale))
    assert mx.array_equal(c.decode(e), ref.astype(mx.float32)).item()
    assert np.array_equal(np.array(c.decode(e)), qs.np_dequant_mxfp4(packed, scale).astype(np.float32))
    assert e.bits()["total"] == MXFP4Native.bits_for(w.shape) == 96 * 256 * 4.25 + METADATA_BITS
    # re-quantization from the reference: equal in value (canonical bytes may differ from the source)
    assert mx.array_equal(c.decode(c.encode(ref, seed=0)), c.decode(e)).item()
    with pytest.raises(ValueError, match="not exactly representable"):
        c.encode(mx.array(w), seed=0)
    write_suc1(tmp_path / "n.suc1", architecture={"model_type": "x"}, tokenizer_id="t", tensors={"a.weight": e},
               metadata={})
    sm = SucModel.load(tmp_path / "n.suc1", verify=True)
    d = sm.descriptor("a.weight")
    assert d["representation_type"] == "mxfp4" and d["layout"] == "mxfp4_u8_nibbles" and d["index_bits"] == 4
    assert np.array_equal(sm.payload("a.weight", "codes"), packed.view(np.uint8))          # verbatim bytes
    assert np.array_equal(sm.payload("a.weight", "scales"), scale)
    assert {r["key"]: r["part"] for r in d["arrays"]} == {"codes": "indices", "scales": "scales"}
    assert mx.array_equal(sm.decode_tensor("a.weight"), ref.astype(mx.float32)).item()   # reference decoder
    back = sm.encoded("a.weight")
    assert back.params == e.params and all(mx.array_equal(back.arrays[k], e.arrays[k]).item() for k in e.arrays)


def test_mixed_switch_with_native_experts_runs_the_reference():
    from suc.kernels.mxfp4 import MXFP4SwitchLinear
    from suc.kernels.suc_switch import SucSwitchMixed, switch_from_encodings
    from suc.quant.codecs import make_codec

    rng = np.random.default_rng(4)
    E, out, inn = 6, 64, 128
    nat, vq = make_codec({"kind": "mxfp4"}), make_codec({"kind": "vq", "k": 16, "block_size": 4, "kmeans_iters": 2,
                                                         "rotate_out": False})
    src = [qs.np_quantize_mxfp4(rng.standard_normal((out, inn)).astype(np.float32) * 0.05) for _ in range(E)]
    refs = [qs.dequant_mxfp4(mx.array(p), mx.array(s)).astype(mx.float32) for p, s in src]
    enc = [nat.encode_source(mx.array(src[e][0]), mx.array(src[e][1])) if e % 2 == 0 else vq.encode(refs[e], seed=e)
           for e in range(E)]
    m = switch_from_encodings(enc)
    assert isinstance(m, SucSwitchMixed) and m.members == [[0, 2, 4], [1, 3, 5]]
    assert isinstance(m.subs[0], MXFP4SwitchLinear)
    assert isinstance(switch_from_encodings(enc[0::2]), MXFP4SwitchLinear)
    x = mx.array(rng.standard_normal((5, 1, 1, inn)).astype(np.float32))
    idx = mx.array([[0, 1], [2, 3], [4, 5], [5, 0], [2, 4]]).astype(mx.uint32)
    y = m(x, idx)
    for b in range(5):
        for j in range(2):
            e = int(idx[b, j].item())
            ref = x[b, 0, 0] @ (refs[e] if e % 2 == 0 else vq.decode(enc[e])).T
            tol = 1e-5 if e % 2 == 0 else 1e-3                        # native: float32 run of the exact reference
            assert mx.abs(y[b, j, 0] - ref).max().item() < tol * max(1.0, mx.abs(ref).max().item()), (b, j)


def test_native_candidate_in_the_greedy_allocation():
    from suc.allocation.greedy import Candidate, greedy_allocate

    # per expert: (VQ G=4, VQ G=2, native); the sensitive expert "b" goes native, the budget is respected
    cands = {n: [Candidate("g4", 220, l4), Candidate("g2", 420, l2), Candidate("mxfp4_native", 425, 0.0)]
             for n, l4, l2 in (("a", 1.0, 0.2), ("b", 50.0, 9.0), ("c", 2.0, 0.5))}
    plan = greedy_allocate(cands, 220 * 2 + 425)
    assert plan["b"].codec == "mxfp4_native" and sum(c.bits for c in plan.values()) <= 220 * 2 + 425
    assert {plan["a"].codec, plan["c"].codec} == {"g4"}


def test_native_variant_reuses_fisher_and_fits_the_budget(compressed):
    r, model_dir, budget, _, rn = compressed
    assert {k: rn["stages"][k]["status"] for k in ("calibrate", "reference", "candidates")} == \
        dict.fromkeys(("calibrate", "reference", "candidates"), "skipped")       # no recalibration / Fisher
    p = rn["plan"]
    assert p["predicted_total_bytes"] <= budget and rn["container"]["bytes"] <= budget
    assert p["native_candidates_offered"]["mxfp4_native"] == 4 * 8 * 3                # every routed expert tensor
    plan = json.loads((Path(rn["container"]["path"]).parent.parent / "plan" / "plan.json").read_text())["tensors"]
    native = [n for n, c in plan.items() if c == "mxfp4_native"]
    assert native and all(".mlp.experts." in n for n in native)
    assert p["predicted_loss"] <= r["plan"]["predicted_loss"] + 1e-12         # one more candidate never hurts
    hf = SafetensorsStore.from_dir(model_dir)
    st = get_adapter(json.loads((model_dir / "config.json").read_text())).source_store(hf)
    from suc.io.suc1 import SucModel

    sm = SucModel.load(rn["container"]["path"], verify=True)
    for n in native:                                                  # bit-exact vs the checkpoint bytes
        kind, w8, s8 = st.load_source(n)
        assert kind == "mxfp4" and sm.descriptor(n)["representation_type"] == "mxfp4"
        assert np.array_equal(sm.payload(n, "codes"), np.array(w8).view(np.uint8))
        assert np.array_equal(sm.payload(n, "scales"), np.array(s8))
        assert mx.array_equal(sm.decode_tensor(n), st.load(n).astype(mx.float32)).item()
    enc = {t["name"]: t for t in json.loads((Path(rn["container"]["path"]).parent.parent / "encode" /
                                             "encode.json").read_text())["tensors"]}
    assert all(enc[n]["rel_mse"] == 0.0 for n in native)


def test_native_experts_run_from_codes(compressed):
    from suc.kernels.mxfp4 import MXFP4SwitchLinear
    from suc.analysis.sensitivity import set_weight
    from suc.io.suc1 import SucModel
    from suc.runtime.suc1_model import load_suc1_model

    _, model_dir, _, _, rn = compressed
    path = rn["container"]["path"]
    sm = SucModel.load(path, verify=False)
    model, _, rep = load_suc1_model(path, group=False)
    found = 0
    for i, layer in enumerate(model.layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            m = getattr(layer.mlp.switch_mlp, proj)
            subs = getattr(m, "subs", [m])
            members = getattr(m, "members", [list(range(m.num_experts))])
            for sub, mem in zip(subs, members):
                if not isinstance(sub, MXFP4SwitchLinear):
                    continue
                x = mx.random.normal((len(mem), 1, 1, sub.in_features))
                y = m(x, mx.array(mem).astype(mx.uint32).reshape(-1, 1))
                for k, e in enumerate(mem):                           # native experts = the D8 reference
                    w = sm.decode_tensor(f"model.layers.{i}.mlp.experts.{e}.{proj}.weight")
                    ref = x[k, 0, 0] @ w.T
                    assert mx.abs(y[k, 0, 0] - ref).max().item() < 1e-5 * max(1.0, mx.abs(ref).max().item())
                    found += 1
    assert found and any(c["kind"] in ("SucSwitchMixed", "MXFP4SwitchLinear") for c in rep["converted"])
    ref, _ = load_lazy(model_dir)
    for n in sm.names():
        if sm.descriptor(n)["representation_type"] != "raw":
            set_weight(ref, n, sm.decode_tensor(n))
    x = mx.random.randint(2, 300, (2, 40))
    lr, lq = ref(x).astype(mx.float32), model(x).astype(mx.float32)
    p = mx.softmax(lr, axis=-1)
    kl = (p * (mx.log(p + 1e-9) - mx.log(mx.softmax(lq, axis=-1) + 1e-9))).sum(-1).mean().item()
    assert kl < 2e-2, kl


def test_native_experts_stream_bit_identical(compressed):
    """Routed experts with native MXFP4 records (next to SUC-coded ones) streamed through a small ExpertCache (pread,
    evictions on every step) give the resident runtime's logits bit for bit."""
    from suc.io.suc1 import SucModel
    from suc.runtime.expert_cache import ExpertLayout
    from suc.runtime.expert_streaming import load_streamed_suc1_model
    from suc.runtime.suc1_model import load_suc1_model, run_tokens

    _, _, _, _, rn = compressed
    path = rn["container"]["path"]
    sm = SucModel.load(path, verify=False)
    lay = ExpertLayout(sm)
    native = [n for n in sm.names() if ".mlp.experts." in n and sm.descriptor(n)["representation_type"] == "mxfp4"]
    assert native
    model, _, _ = load_suc1_model(path, group=False)
    tokens = [int(t) for t in np.random.default_rng(3).integers(2, 300, 12)]
    resident = run_tokens(model, tokens, hidden=False)[0]
    k = model.args.num_experts_per_tok if hasattr(model, "args") else 2
    one = max(lay.entry(li, e).nbytes for li in range(len(lay.layers)) for e in range(lay.num_experts[li]))
    for budget in (10 ** 9, (k + 1) * one):
        smodel, _, cache, rep = load_streamed_suc1_model(path, budget, hot_experts=None, io="pread", nocache=False)
        got = run_tokens(smodel, tokens, hidden=False)[0]
        assert np.array_equal(got, resident), budget
        assert cache.stats.resident_bytes <= budget
        if budget < 10 ** 9:
            assert cache.stats.misses > 0 and cache.stats.evictions > 0
        cache.close()


def test_native_variant_config_reuses_the_base_run():
    import suc.pipeline.compress as comp
    from suc.config import load_compress_config

    repo = Path(__file__).resolve().parents[2]
    a = _config(repo, "compress_deepseek_v4_flash.yaml")
    b = _config(repo, "compress_deepseek_v4_flash_native.yaml")
    # identical calibrate / reference / candidates stage inputs -> the base run's artifacts are reused
    assert comp.scored_plan_delta(b.plan) == comp.config_delta(a.plan) == comp.scored_plan_delta(a.plan)
    for k in ("calibration", "eval"):
        assert comp.config_delta(getattr(a, k)) == comp.config_delta(getattr(b, k))
    assert (a.seed, a.model, a.plan.budget_bytes) == (b.seed, b.model, b.plan.budget_bytes)
    assert comp.native_candidate_names(b.plan) == {"mxfp4_native"}
    ad = get_adapter(REAL_CFG)
    st = ad.source_store(_fake_store(REAL["tensors"]))
    from suc.quant.run import select_specs

    names = st.names()
    specs = select_specs(ad, names, None, include_embeddings=False)
    ctx = comp.Ctx(cfg=b, model_dir=Path("/nonexistent"), store=st, hf_config=REAL_CFG, adapter=ad, specs=specs,
                   head_specs=[], model_hash="x", revision=b.model.revision)
    assert comp.codec_shape_errors(ctx) == [] and comp.native_candidate_errors(ctx) == []
    experts = [s for s in specs if s.role in EXPERT_ROLES]
    assert all(st.quant_source(s.name).kind == "mxfp4" for s in experts)
    assert all("mxfp4_native" not in comp._allowed_candidates(ctx, s.role) for s in specs if s.role not in EXPERT_ROLES)


def test_final_config_fits_the_real_checkpoint():
    import suc.pipeline.compress as comp
    from suc.config import load_compress_config
    from suc.quant.run import select_specs

    repo = Path(__file__).resolve().parents[2]
    cfg = _config(repo, "compress_deepseek_v4_flash.yaml")
    assert cfg.model.revision == "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    assert cfg.plan.mode == "greedy" and cfg.plan.allocation_unit == "expert"
    for c in cfg.plan.candidates:
        assert c.kind not in ("vq", "additive") or c.k * c.block_size * 2 <= 16 * 1024      # D4
    ad = get_adapter(REAL_CFG)
    st = ad.source_store(_fake_store(REAL["tensors"]))
    names = st.names()
    specs = select_specs(ad, names, None, include_embeddings=False)
    head = [s for s in select_specs(ad, names, None, include_embeddings=True) if s.role in ("embed", "lm_head")]
    ctx = comp.Ctx(cfg=cfg, model_dir=Path("/nonexistent"), store=st, hf_config=REAL_CFG, adapter=ad, specs=specs,
                   head_specs=head, model_hash="x", revision=cfg.model.revision)
    assert comp.codec_shape_errors(ctx) == []
    low = [c.name for c in cfg.plan.candidates if c.kind == "vq" and c.block_size == 4]
    for s in specs:                                     # sub-3-bit codecs only for routed experts
        allowed = comp._allowed_candidates(ctx, s.role)
        assert bool(set(low) & set(allowed)) == (s.role in EXPERT_ROLES)
    fx = comp.fixed_bits(ctx, {s.name for s in specs})
    alloc = sum(math.prod(st.info(s.name).shape) for s in specs)
    bpw = (cfg.plan.budget_bytes * 8 - fx["total"]) / alloc
    assert 2.9 < bpw < 3.05                                           # ~3 bpw average: the known-good 108 GB point
    kv = ad.kv_bytes_per_token() * cfg.preflight.runtime_context + ad.kv_fixed_bytes()
    assert cfg.plan.budget_bytes + kv + cfg.preflight.runtime_activation_bytes <= cfg.preflight.runtime_budget_bytes
    est = comp.estimate_resources(ctx)
    assert est["memory"]["peak_estimate_bytes"] < 0.85 * 128 * 2 ** 30



@pytest.mark.parametrize("variant,mode", [("native_max", "resident"), ("flat_max", "resident"),
                                          ("native_stream", "expert_streaming"), ("s97", "resident"),
                                          ("s94", "resident")])
def test_replan_variant_configs_reuse_the_base_run(variant, mode):
    """docs/arch/deepseek_v4.md §12: the re-plan variants keep the candidates-stage inputs of the base run (only the
    allocation-only plan.allocation differs), pass the runtime check, and the greedy honours allocation.exclude and
    allocation.budget_bytes (synthetic candidates on the real tensor list: no weights needed)."""
    import suc.pipeline.compress as comp
    from suc.allocation.greedy import Candidate
    from suc.config import load_compress_config
    from suc.quant.codecs import MXFP4Native
    from suc.quant.run import select_specs

    repo = Path(__file__).resolve().parents[2]
    a = _config(repo, "compress_deepseek_v4_flash.yaml")
    v = _config(repo, f"compress_deepseek_v4_flash_{variant}.yaml")
    assert comp.scored_plan_delta(v.plan) == comp.scored_plan_delta(a.plan)
    for k in ("calibration", "eval"):
        assert comp.config_delta(getattr(a, k)) == comp.config_delta(getattr(v, k))
    assert (a.seed, a.model, a.plan.budget_bytes) == (v.seed, v.model, v.plan.budget_bytes)
    assert comp.config_delta(v.plan) != comp.config_delta(a.plan)               # the plan stage itself re-runs
    ad = get_adapter(REAL_CFG)
    st = ad.source_store(_fake_store(REAL["tensors"]))
    names = st.names()
    specs = select_specs(ad, names, None, include_embeddings=False)
    head = [s for s in select_specs(ad, names, None, include_embeddings=True) if s.role in ("embed", "lm_head")]
    ctx = comp.Ctx(cfg=v, model_dir=Path("/nonexistent"), store=st, hf_config=REAL_CFG, adapter=ad, specs=specs,
                   head_specs=head, model_hash="x", revision=v.model.revision)
    assert comp.codec_shape_errors(ctx) == [] and comp.native_candidate_errors(ctx) == []
    rec, errors, _ = comp.runtime_memory_check(ctx)
    assert errors == [] and rec["mode"] == mode
    # synthetic scores: every scored candidate of a role, bits from its nominal bpw, loss decreasing with bits
    bpw = {c.name: comp._est_bpw(c, 4096, 2048 * 4096) for c in v.plan.candidates if c.kind != "mxfp4"}
    cands = {}
    for s in specs:
        n = math.prod(st.info(s.name).shape)
        cands[s.name] = [Candidate(c, int(bpw[c] * n), (1.0 + hash(s.name) % 7) * 2.0 ** -bpw[c])
                         for c in comp._allowed_candidates(ctx, s.role, scored=True)]
    plan, summary = comp.allocate_plan(ctx, cands)
    assert summary["predicted_total_bytes"] <= v.plan.allocation_budget_bytes() == summary["budget_bytes"]
    for s in specs:
        assert not v.plan.excluded(plan[s.name], s.role), (s.name, plan[s.name])
    experts = {plan[s.name] for s in specs if s.role in EXPERT_ROLES}
    if variant == "native_stream":
        assert experts == {"mxfp4_native"} and summary["predicted_total_bytes"] > rec["runtime_budget_bytes"]
        assert MXFP4Native.bits_for((2048, 4096)) * 33024 / 8 < summary["predicted_total_bytes"]
    elif variant == "flat_max":
        assert experts <= {c.name for c in v.plan.candidates if c.kind in ("vq", "additive")} and \
            "mxfp4_native" not in experts
    else:
        assert "mxfp4_native" in experts

def test_reference_check_script_mlx_side(tiny_dir, tmp_path):
    """The MLX half of scripts/deepseek_v4_reference_check.py against a reference made by the MLX model itself: zero
    error in float32. The torch half runs on the Mac Studio (verified there on this tiny checkpoint: float32 block
    outputs within ~1e-6 relative of the official inference/model.py with the plain-torch kernels)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ds4ref", Path(__file__).resolve().parents[2]
                                                  / "scripts/deepseek_v4_reference_check.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    tokens = np.random.default_rng(0).integers(2, 300, (1, 40)).astype(np.int64)
    outs = ref.mlx_block_outputs(tiny_dir, tokens, 4, mx.float32)
    assert set(outs) == {"block_00", "block_01", "block_02", "block_03", "logits"}
    npz = tmp_path / "ref.npz"
    np.savez(npz, tokens=tokens, dtype="float32", blocks=4, seq=40, seed=0, **outs)
    p = ref.mlx_side(ref.argparse.Namespace(model_dir=str(tiny_dir), ref=str(npz), revision="tiny",
                                            output_dir=str(tmp_path / "results")))
    r = json.loads(p.read_text())["mlx_vs_torch"]
    assert max(v["max_abs"] for v in r.values()) == 0.0
    args = ref.reference_args(REAL_CFG)
    assert args["dim"] == 4096 and args["n_hash_layers"] == 3 and args["original_seq_len"] == 65536


def _config(repo, name):
    """Load experiments/configs/<name>; skip when the file is not in this checkout (the public snapshot omits the
    configs of discarded variants)."""
    from suc.config import load_compress_config
    path = repo / "experiments/configs" / name
    if not path.exists():
        pytest.skip(f"{name} is not in this checkout")
    return load_compress_config(path)
