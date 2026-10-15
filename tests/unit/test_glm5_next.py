"""glm5_next (zai-org/GLM-5.3-Flash): FP8 (E4M3 x F32 block multiplier) source dequantization, the adapter on the REAL
checkpoint layout (all 62 safetensors headers, fixture), the MLX model against transformers' glm5_next (a torch
reference of the tiny checkpoint, fixture made with scripts/glm5_next_reference_check.py on the Mac Studio) and its
own cached decode, and `suc compress` end to end with an expert role budget in expert-streaming mode -> SUC1 -> the
Python runtime from the codes, with the routed experts resident or streamed through an ExpertCache (identical)."""
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
from suc.models import mlx_glm5_next as glm
from suc.models.adapters import Glm5NextAdapter, get_adapter
from suc.models.streaming import load_lazy
from tiny_models import tiny_glm5_next_config, write_tiny_glm5_next

FIX = Path(__file__).parent / "fixtures"
REAL_CFG = json.loads((FIX / "glm5.3_flash_config.json").read_text())
REAL = json.loads(gzip.open(FIX / "glm5.3_flash_tensors.json.gz").read())
_ITEM = {"BF16": 2, "F32": 4, "F8_E4M3": 1}
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
    return write_tiny_glm5_next(tmp_path_factory.mktemp("glm") / "model")


def _f32_model(d):
    model, _ = load_lazy(d)
    model.set_dtype(mx.float32)
    return model


# ------------------------------------------------------------------ source dequantization

def test_fp8_f32_scale_dequant_matches_the_reference_rounding():
    rng = np.random.default_rng(0)
    w8 = rng.integers(0, 256, (300, 260), dtype=np.uint8)
    w8[(w8 & 0x7F) == 0x7F] = 0x11                                     # no NaN codes
    s = (rng.random((3, 3)) * 1e-3 + 1e-5).astype(np.float32)          # partial blocks, non power-of-two scales
    got = qs.dequant_fp8_block_f32(mx.array(w8), mx.array(s))
    assert got.dtype == mx.bfloat16
    ref = qs.np_dequant_fp8_block_f32(w8, s)                            # float32 product (transformers)
    want = np.array(mx.array(ref).astype(mx.bfloat16).astype(mx.float32))
    assert np.array_equal(np.array(got.astype(mx.float32)), want)
    f32 = np.array(qs.dequant_fp8_block_f32(mx.array(w8), mx.array(s), dtype=mx.float32))
    assert np.array_equal(f32, ref)                                     # the same IEEE float32 multiply
    assert qs.check_scales("fp8_block_f32", mx.array(s)) == (float(s.min()), float(s.max()))
    for bad in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(qs.NotLossless):
            qs.check_scales("fp8_block_f32", mx.array(np.array([[1.0, bad]], dtype=np.float32)))


def test_store_dequantizes_the_tiny_checkpoint(tiny_dir):
    hf = SafetensorsStore.from_dir(tiny_dir)
    ad = get_adapter(json.loads((tiny_dir / "config.json").read_text()))
    ad.validate(hf.names())
    st = ad.source_store(hf)
    ad.validate(st.names())                                             # the SUC namespace validates too
    assert not [n for n in st.names() if n.startswith(("model.visual.", "model.language_model."))
                or n.endswith("_scale_inv") or n.startswith("model.layers.4.")]
    for suc in ("model.layers.3.self_attn.q_b_proj.weight", "model.layers.2.mlp.experts.5.down_proj.weight",
                "model.layers.0.mlp.up_proj.weight"):
        raw = "model.language_model." + suc[len("model."):]
        ref = qs.np_dequant_fp8_block_f32(np.array(hf.load(raw)), np.array(hf.load(raw + "_scale_inv")))
        got = st.load(suc)
        assert got.dtype == mx.bfloat16 and st.info(suc).dtype == "BF16" and tuple(got.shape) == st.info(suc).shape
        assert np.array_equal(np.array(got.astype(mx.float32)),
                              np.array(mx.array(ref).astype(mx.bfloat16).astype(mx.float32)))
    r = st.check_quant_scales()
    assert set(r) == {"fp8_block_f32"} and r["fp8_block_f32"]["tensors"] == 3 + 4 + 3 * (8 * 3 + 3)
    model, _ = load_lazy(tiny_dir)                                      # sanitize == the store, tensor for tensor
    params = dict(tree_flatten(model.parameters()))
    for n in st.names():
        if (m := re.match(r"^(.*)\.experts\.(\d+)\.(\w+)\.weight$", n)) is not None:
            p = params[f"{m.group(1)}.switch_mlp.{m.group(3)}.weight"][int(m.group(2))]
        else:
            p = params[n]
        assert np.array_equal(np.array(p.astype(mx.float32)), np.array(st.load(n).astype(mx.float32))), n
    assert len(params) == len({re.sub(r"\.experts\.\d+\.", ".switch_mlp.", n) for n in st.names()})


# ------------------------------------------------------------------ adapter on the real checkpoint layout

def test_adapter_on_the_real_checkpoint():
    names = list(REAL["tensors"])
    assert len(names) == 76108
    assert sum(math.prod(s) * _ITEM[d] for d, s in REAL["tensors"].values()) == REAL["index_total_size"]
    ad = get_adapter(REAL_CFG)
    assert isinstance(ad, Glm5NextAdapter) and ad.num_layers == 45 and ad.num_experts == 288
    assert [i for i in range(45) if ad.layer_type(i) == "dsa"] == list(range(3, 45, 4))
    assert [i for i in range(45) if not ad.is_sparse_layer(i)] == [0, 1, 2]
    ad.validate(names)
    st = ad.source_store(_fake_store(REAL["tensors"]))
    ad.validate(st.names())
    assert all(n.startswith("model.layers.") or n in ("model.embed_tokens.weight", "model.norm.weight",
                                                      "lm_head.weight") for n in st.names())
    assert not [n for n in st.names() if n.startswith("model.layers.45.")]
    assert len(st.quant_names()) == 42 * (3 * 288 + 3) + 3 * 3 + 11 * 4    # experts, shared, dense MLP, MLA
    specs = [s for s in ad.linears(st.names()) if s.eligible]
    roles = {}
    for s in specs:
        roles[s.role] = roles.get(s.role, 0) + 1
    assert roles["expert_gate"] == 42 * 288 and roles["kda_o"] == 34 and roles["attn_o"] == 11
    assert not {"attn_kv_b", "indexer_q_b", "indexer_k", "kda_f_a", "kda_beta", "router"} & set(roles)
    elig = sum(math.prod(st.info(s.name).shape) for s in specs)
    experts = sum(math.prod(st.info(s.name).shape) for s in specs if s.role in EXPERT_ROLES)
    assert experts == 304_405_807_104                                   # docs/arch/glm5_next.md §5
    assert elig == 311_586_455_552                                      # §6 minus kv_b (kept BF16)
    e = st.info("model.layers.7.mlp.experts.200.gate_proj.weight")
    assert e.shape == (2048, 4096) and e.dtype == "BF16" and st.quant_source(e.name).kind == "fp8_block_f32"
    assert st.info("model.layers.4.self_attn.q_proj.weight").dtype == "BF16"
    assert st.quant_source("model.layers.4.self_attn.q_proj.weight") is None
    with pytest.raises(ValueError, match="unclassified"):
        ad.validate(names + ["model.language_model.layers.0.self_attn.mystery"])
    broken = [n for n in names if n != "model.language_model.layers.9.self_attn.A_log"]
    with pytest.raises(ValueError, match="A_log"):
        ad.validate(broken)
    assert ad.kv_bytes_per_token() == 11 * (512 + 2 * 128) * 2          # 16.9 KB per token
    assert ad.kv_fixed_bytes() == 34 * (64 * 128 * 128 * 4 + 3 * 8192 * 3 * 2)
    assert ad.residual_width == 4 * 4096


def test_mlx_parameters_map_one_to_one_onto_the_real_tensors():
    """Every MLX parameter of the real config (lazy, never materialized) has exactly the shape of its SUC-namespace
    tensor (expert stacks = the stacked per-expert tensors), and every SUC tensor is a parameter."""
    ad = get_adapter(REAL_CFG)
    st = ad.source_store(_fake_store(REAL["tensors"]))
    model = glm.Model(glm.ModelArgs.from_dict(REAL_CFG))
    params = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    want = {}
    for n in st.names():
        if m := re.match(r"^(.*)\.experts\.(\d+)\.(\w+)\.weight$", n):
            want[f"{m.group(1)}.switch_mlp.{m.group(3)}.weight"] = (ad.num_experts, *st.info(n).shape)
        else:
            want[n] = st.info(n).shape
    assert params == want


# ------------------------------------------------------------------ MLX model numerics

def test_mlx_matches_the_transformers_reference(tiny_dir):
    """Block outputs and logits of transformers 5.18 Glm5NextTextDecoderLayer (float32, KDA chunked form, the DSA
    indexer selecting 2 of up to 10 k-pools) on the tiny checkpoint: fixture written by
    `scripts/glm5_next_reference_check.py torch --model-dir <tiny> --blocks 4 --seq 40 --dtype float32` on the Mac
    Studio (torch venv). Measured there: every block <= 1.4e-6 relative, logits KL 9e-13."""
    z = np.load(FIX / "glm5_next_tiny_torch_ref.npz")
    model = _f32_model(tiny_dir)
    h = model.embed_stream(mx.array(z["tokens"]))
    for i, layer in enumerate(model.layers):
        h = layer(h)
        if f"block_{i:02d}" in z.files:
            got = np.array(h).reshape(z[f"block_{i:02d}"].shape)
            ref = z[f"block_{i:02d}"]
            assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-5, i
    lg = np.array(model.head_stream(h))
    assert np.abs(lg - z["logits"]).max() < 1e-4
    # the selection matters at this length: with every pool selected the logits change
    model.layers[3].self_attn.indexer.topk_pools = 10 ** 6
    assert np.abs(np.array(model(mx.array(z["tokens"]))) - z["logits"]).max() > 1e-3


def test_cached_decode_equals_prefill(tiny_dir):
    model = _f32_model(tiny_dir)
    mx.random.seed(3)
    x = mx.random.randint(2, 250, (2, 45))
    full = model(x)
    for split in (1, 6, 13):                                            # pool borders, tail, selection active
        cache = model.make_cache()
        steps = [model(x[:, :split], cache=cache)] + [model(x[:, t:t + 1], cache=cache) for t in range(split, 45)]
        assert mx.abs(mx.concatenate(steps, axis=1) - full).max().item() < 1e-4
    from suc.models.streaming import stream_blocks
    r = stream_blocks(model, x, return_logits=True, release=False, micro_batch=2)
    assert mx.array_equal(r["logits"], full)                            # identical ops and shapes: bit-exact
    assert np.isfinite(np.array(_bf16(tiny_dir)(x).astype(mx.float32))).all()


def _bf16(d):
    model, _ = load_lazy(d)
    return model


def test_config_parsing_variants():
    cfg = tiny_glm5_next_config()
    a = glm.ModelArgs.from_dict(cfg).text
    assert a.layer_kinds == ["kda", "kda", "kda", "dsa"] and a.sparse == [False, True, True, True]
    t = dict(cfg["text_config"])
    del t["layer_types"], t["mlp_layer_types"]
    a2 = glm.TextArgs.from_dict(t)                                      # kda_layers + first_k_dense_replace
    assert a2.layer_kinds == a.layer_kinds and a2.sparse == a.sparse
    with pytest.raises(NotImplementedError):
        glm.TextArgs.from_dict({**t, "indexer_types": ["full", "full", "full", "shared"]})
    m = glm.Model(glm.ModelArgs.from_dict(cfg))
    m.set_dtype(mx.bfloat16)
    p = dict(tree_flatten(m.parameters()))
    assert p["model.layers.0.hc_attn_fn"].dtype == mx.float32 and p["model.layers.0.self_attn.A_log"].dtype == mx.float32
    assert p["model.layers.1.self_attn.q_conv1d.weight"].dtype == mx.float32
    assert p["model.layers.1.mlp.gate.e_score_correction_bias"].dtype == mx.float32
    assert p["model.layers.3.self_attn.q_b_proj.weight"].dtype == mx.bfloat16


# ------------------------------------------------------------------ suc compress -> SUC1 -> runtime

def _cands():
    def vq(g, xs=True):
        c = {"kind": "vq", "k": 16, "block_size": g, "rotate_out": False, "kmeans_iters": 3,
             "residual_fraction": 0.02, "residual_structure": "columns"}
        return dict(c, codebook_scope="expert_stack") if xs else c
    from suc.config import CodecConfig
    c = [vq(4), vq(2), vq(2, False), {"kind": "bf16"}]
    names = [CodecConfig(**x).name for x in c]
    nonexp = ["kda_q", "kda_k", "kda_v", "kda_o", "attn_q_a", "attn_q_b", "attn_kv_a", "attn_o", "mlp_gate", "mlp_up",
              "mlp_down", "shared_expert_gate", "shared_expert_up", "shared_expert_down"]
    roles = {names[0]: list(EXPERT_ROLES), names[1]: list(EXPERT_ROLES), names[2]: nonexp, names[3]: nonexp}
    return c, names, roles


EXPERT_BPW = 2.0


@pytest.fixture(scope="module")
def compressed(tmp_path_factory):
    import suc.pipeline.compress as comp
    from test_calibrate_run import FakeTokenizer, _docs

    tmp = tmp_path_factory.mktemp("glmc")
    mx.random.seed(0)
    tokens = mx.random.randint(2, 250, (2048,), dtype=mx.int32)
    saved = comp.load_wikitext2, comp.tokenize
    comp.load_wikitext2 = lambda split: ("unused", "wikihash")
    comp.tokenize = lambda tok, text: tokens
    try:
        docs, ood = _docs()
        model_dir = write_tiny_glm5_next(tmp / "model", seed=2)
        cands, names, roles = _cands()
        cfg = {"name": "c_glm", "seed": 7, "output_dir": str(tmp / "results"), "run_dir": str(tmp / "run"),
               "model": {"id": "tiny/glm5_next", "local_path": str(model_dir)},
               "calibration": {"seq_len": 32, "n_calibration": 16, "n_validation": 8, "n_ood": 4, "micro_batch": 4},
               "plan": {"mode": "greedy", "budget_bpw": 6.0, "allocation_unit": "expert", "candidates": cands,
                        "role_budgets": [{"roles": list(EXPERT_ROLES), "budget_bpw": EXPERT_BPW}],
                        "candidate_roles": roles, "embedding_codec": {"kind": "affine", "bits": 4},
                        "fisher_sequences": 8, "fisher_micro_batch": 2},
               "eval": {"seq_len": 64, "max_tokens": 1024, "kl_sequences": 4, "chunk": 2},
               "finetune": None, "rotation_groups": True, "verify_tensors": 8,
               "preflight": {"runtime_budget_bytes": 2.2e6, "runtime_context": 1024,
                             "runtime_activation_bytes": 1e5, "expert_streaming": True}}
        p = tmp / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg))
        r = json.loads(comp.run_compress(p, tokenizer=FakeTokenizer(), docs=docs, ood_docs=ood).read_text())
    finally:
        comp.load_wikitext2, comp.tokenize = saved
    return r, model_dir, names, cfg, p


def test_compress_role_budget_and_streaming_preflight(compressed):
    r, _, names, _, _ = compressed
    p = r["plan"]
    assert r["architecture"]["family"] == "glm5_next" and p["allocation_unit"] == "expert"
    assert r["preflight"]["source_precision"]["lossless"] is True
    rb = p["role_budgets"][0]
    assert rb["roles"] == list(EXPERT_ROLES) and rb["params"] == 3 * 8 * 3 * 128 * 128
    assert rb["predicted_bits"] <= rb["budget_bits"] and rb["predicted_bpw"] > 0.9 * EXPERT_BPW
    enc = {t["name"]: t for t in r["encode"]["tensors"]}
    ex = [t for n, t in enc.items() if ".mlp.experts." in n]
    assert sum(t["bits"]["total"] for t in ex) / sum(math.prod(t["shape"]) for t in ex) <= EXPERT_BPW
    rt = r["preflight"]["runtime"]
    assert rt["mode"] == "expert_streaming" and rt["need_bytes"] > rt["runtime_budget_bytes"]
    assert 0 < rt["expert_cache_bytes"] < rt["routed_expert_bytes"]
    assert any("expert-streaming" in w for w in r["preflight"].get("warnings", []))
    plan = json.loads((Path(r["container"]["path"]).parent.parent / "plan" / "plan.json").read_text())["tensors"]
    for n, c in plan.items():
        if ".mlp.experts." in n:
            assert c in names[:2], (n, c)
        elif n not in ("model.embed_tokens.weight", "lm_head.weight"):
            assert c in names[2:], (n, c)
    hot = json.loads((Path(r["container"]["path"]).parent / "hot_experts.json").read_text())
    assert sorted(hot["layers"]) == [f"model.layers.{i}.mlp" for i in (1, 2, 3)]
    assert all(sum(v) == 16 * 32 * 2 for v in hot["layers"].values())   # n_calibration x seq_len x top-k
    assert np.isfinite(r["quality"]["kl"]["kl_mean"])


def test_preflight_rejects_an_oversized_plan_without_streaming(compressed):
    import suc.pipeline.compress as comp
    from suc.config import compress_config_from_dict
    from suc.quant.run import select_specs

    _, model_dir, _, cfg, _ = compressed
    hf = json.loads((model_dir / "config.json").read_text())
    ad = get_adapter(hf)
    st = ad.source_store(SafetensorsStore.from_dir(model_dir))
    names = st.names()
    specs = select_specs(ad, names, None, include_embeddings=False)
    head = [s for s in select_specs(ad, names, None, include_embeddings=True) if s.role in ("embed", "lm_head")]
    for streaming, budget, ok in ((False, 2.2e6, False), (True, 2.2e6, True), (True, 1.75e6, False)):
        c = json.loads(json.dumps(cfg))
        c["preflight"].update(expert_streaming=streaming, runtime_budget_bytes=budget)
        ctx = comp.Ctx(cfg=compress_config_from_dict(c), model_dir=model_dir, store=st, hf_config=hf, adapter=ad,
                       specs=specs, head_specs=head, model_hash="x", revision="tiny")
        rec, errors, _ = comp.runtime_memory_check(ctx)
        assert (not errors) == ok, (streaming, budget, errors)


def _logits(model, tokens):
    from suc.runtime.suc1_model import run_tokens

    return run_tokens(model, tokens, hidden=False)[0]


def test_runtime_resident_and_streamed_are_identical(compressed):
    from suc.io.suc1 import SucModel
    from suc.runtime.expert_cache import ExpertLayout
    from suc.runtime.expert_streaming import load_streamed_suc1_model
    from suc.runtime.suc1_model import load_suc1_model

    r, model_dir, _, _, _ = compressed
    path = r["container"]["path"]
    sm = SucModel.load(path, verify=True)
    assert not [n for n in sm.names() if n.startswith(("model.visual.", "model.language_model.", "model.layers.4."))]
    model, _, rep = load_suc1_model(path, group=False)
    kinds = {c["kind"] for c in rep["converted"]}
    assert "SucSwitchMixed" in kinds or "SucSwitchLinear" in kinds
    tokens = [int(t) for t in np.random.default_rng(1).integers(2, 250, 24)]
    resident = _logits(model, tokens)
    lay = ExpertLayout(sm)
    one = max(lay.entry(li, e).nbytes for li in range(len(lay.layers)) for e in range(8))
    for budget, hot in ((10 ** 9, "auto"), (2 * 2 * one, None), (3 * 2 * one, "auto")):   # all / 2 / 3 layers' top-2
        sm_model, _, cache, srep = load_streamed_suc1_model(path, budget, hot_experts=hot, pin_fraction=0.5,
                                                            io="pread", nocache=False)
        got = _logits(sm_model, tokens)
        assert np.array_equal(got, resident), budget
        st = cache.stats
        assert st.resident_bytes <= budget and st.requests == len(tokens) * 3 * 2
        if budget < 10 ** 9:
            assert st.misses > 0 and st.evictions > 0
        if hot == "auto":
            assert srep["expert_streaming"]["pinned_experts"] > 0
            assert st.pinned_bytes <= 0.5 * budget
        cache.close()
    # prefill path (sorted expert rows, several tokens per call) streamed == resident
    x = mx.array([tokens])
    sm_model, _, cache, _ = load_streamed_suc1_model(path, 10 ** 9, hot_experts=None, nocache=False)
    ref = model(x)
    assert mx.array_equal(sm_model(x), ref)
    cache.close()
    # a prefill needing more experts than the cache holds runs in expert groups (same terms, other summation order)
    sm_model, _, cache, _ = load_streamed_suc1_model(path, 3 * one, hot_experts=None, nocache=False)
    got = sm_model(x)
    assert cache.stats.resident_bytes <= 3 * one and cache.stats.evictions > 0
    assert mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item() < 0.05 * mx.abs(ref).max().item()
    cache.close()
    from suc.cli.generate import run as generate

    a = generate(path, tokens="5,6,7,8", steps=4, quiet=True)
    b = generate(path, tokens="5,6,7,8", steps=4, quiet=True, expert_budget=2 * 2 * one)
    assert a["generated"] == b["generated"] and b["expert_streaming"]["stats"]["misses"] > 0


def test_final_config_fits_the_real_checkpoint():
    import suc.pipeline.compress as comp
    from suc.config import load_compress_config
    from suc.quant.run import select_specs

    repo = Path(__file__).resolve().parents[2]
    cfg = _config(repo, "compress_glm5.3_flash.yaml")
    assert cfg.model.revision == "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
    assert cfg.plan.mode == "greedy" and cfg.plan.allocation_unit == "expert" and cfg.preflight.expert_streaming
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
        assert bool(set(low) & set(comp._allowed_candidates(ctx, s.role))) == (s.role in EXPERT_ROLES)
    w = comp.predicted_weight_bytes(ctx)
    assert w["routed_expert_bpw"] == 3.0 and abs(w["routed_experts"] - 304_405_807_104 * 3 / 8) < 1
    assert 118e9 < w["total"] < 122e9                                   # the owner's ~118-120 GB file
    rec, errors, warnings = comp.runtime_memory_check(ctx)
    assert not errors and rec["mode"] == "expert_streaming" and warnings
    assert rec["need_bytes"] > cfg.preflight.runtime_budget_bytes
    assert 0.85 < rec["expert_cache_fraction"] < 0.95 and rec["resident_bytes"] < 7e9
    est = comp.estimate_resources(ctx)
    assert est["memory"]["peak_estimate_bytes"] < 0.85 * 128 * 2 ** 30


@pytest.mark.parametrize("variant,bpw", [("r245", 2.45), ("r235", 2.35)])
def test_resident_size_variants_reuse_the_base_run(variant, bpw):
    """compress_glm5.3_flash_{r245,r235}: plan.allocation.role_budgets (allocation-only) lowers the routed-expert bpw,
    the candidates-stage inputs stay those of the base run, and the plan passes the RESIDENT runtime check (no expert
    streaming); the greedy honours the override (synthetic candidates on the real tensor list)."""
    import suc.pipeline.compress as comp
    from suc.allocation.greedy import Candidate
    from suc.config import compress_config_from_dict, load_compress_config
    from suc.quant.run import select_specs

    repo = Path(__file__).resolve().parents[2]
    a = _config(repo, "compress_glm5.3_flash.yaml")
    v = _config(repo, f"compress_glm5.3_flash_{variant}.yaml")
    assert comp.scored_plan_delta(v.plan) == comp.scored_plan_delta(a.plan)
    for k in ("calibration", "eval"):
        assert comp.config_delta(getattr(a, k)) == comp.config_delta(getattr(v, k))
    assert v.plan.role_budgets == a.plan.role_budgets and v.plan.allocation_role_budgets()[0]["budget_bpw"] == bpw
    assert not v.preflight.expert_streaming
    ad = get_adapter(REAL_CFG)
    st = ad.source_store(_fake_store(REAL["tensors"]))
    names = st.names()
    specs = select_specs(ad, names, None, include_embeddings=False)
    head = [s for s in select_specs(ad, names, None, include_embeddings=True) if s.role in ("embed", "lm_head")]
    ctx = comp.Ctx(cfg=v, model_dir=Path("/nonexistent"), store=st, hf_config=REAL_CFG, adapter=ad, specs=specs,
                   head_specs=head, model_hash="x", revision=v.model.revision)
    w = comp.predicted_weight_bytes(ctx)
    assert w["routed_expert_bpw"] == bpw and w["total"] < 101e9
    rec, errors, _ = comp.runtime_memory_check(ctx)
    assert errors == [] and rec["mode"] == "resident"
    bits = {c.name: comp._est_bpw(c, 4096, 2048 * 4096) for c in v.plan.candidates}
    cands = {}
    for s in specs:
        n = math.prod(st.info(s.name).shape)
        cands[s.name] = [Candidate(c, int(bits[c] * n), (1.0 + hash(s.name) % 7) * 2.0 ** -bits[c])
                         for c in comp._allowed_candidates(ctx, s.role, scored=True)]
    _, summary = comp.allocate_plan(ctx, cands)
    g = summary["role_budgets"][0]
    assert g["budget_bpw"] == bpw and g["predicted_bpw"] <= bpw
    # the override is validated like plan.role_budgets; budget_bytes still needs a budget_bytes plan
    d = {"name": "x", "model": {"id": "tiny/x", "local_path": "/tmp"},
         "plan": {"mode": "greedy", "budget_bpw": 4.0, "candidates": [{"kind": "affine", "bits": 4}]}}
    for bad in ({"role_budgets": []}, {"role_budgets": [{"roles": ["nope"], "budget_bpw": 2}]},
                {"budget_bytes": 1000}, {"role_budgets": [{"roles": ["expert_up"], "budget_bpw": 0}]}):
        with pytest.raises(ValueError):
            compress_config_from_dict({**d, "plan": {**d["plan"], "allocation": bad}})
    ok = compress_config_from_dict({**d, "plan": {**d["plan"], "allocation": {
        "role_budgets": [{"roles": ["expert_up"], "budget_bpw": 2.5}]}}})
    assert ok.plan.allocation_role_budgets() == ({"roles": ["expert_up"], "budget_bpw": 2.5},)


def test_reference_check_script_mlx_side(tiny_dir, tmp_path):
    """The MLX half of scripts/glm5_next_reference_check.py against a reference made by the MLX model itself: zero
    error in float32 (the torch half ran on the Mac Studio: test_mlx_matches_the_transformers_reference)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("glmref", Path(__file__).resolve().parents[2]
                                                  / "scripts/glm5_next_reference_check.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    tokens = np.random.default_rng(0).integers(2, 250, (1, 30)).astype(np.int64)
    outs = ref.mlx_block_outputs(tiny_dir, tokens, 4, mx.float32)
    assert set(outs) == {"block_00", "block_01", "block_02", "block_03", "logits"}
    npz = tmp_path / "ref.npz"
    np.savez(npz, tokens=tokens, dtype="float32", blocks=4, seq=30, seed=0, **outs)
    p = ref.mlx_side(ref.argparse.Namespace(model_dir=str(tiny_dir), ref=str(npz), revision="tiny",
                                            output_dir=str(tmp_path / "results")))
    r = json.loads(p.read_text())["mlx_vs_torch"]
    assert max(v["max_abs"] for v in r.values()) == 0.0
    t = ref.text_config_dict(REAL_CFG)
    assert t["layer_types"].count("indexed_attention") == 11 and "deepseek_sparse_attention" not in t["layer_types"]


def _config(repo, name):
    """Load experiments/configs/<name>; skip when the file is not in this checkout (the public snapshot omits the
    configs of discarded variants)."""
    from suc.config import load_compress_config
    path = repo / "experiments/configs" / name
    if not path.exists():
        pytest.skip(f"{name} is not in this checkout")
    return load_compress_config(path)
