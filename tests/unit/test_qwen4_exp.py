"""qwen4_exp (Qwen/Qwen3.8-Flash-Next): adapter on the REAL checkpoint layout (all 131 safetensors headers, fixture),
hash constants vs the real I64 buffers, MLX model self-consistency on a tiny random config built from the real
config's structure (full vs streamed vs cached decode, indexer vs a port of the HF selection loop), and `suc compress`
end to end -> SUC1 -> Python runtime (SucLinear / SucSwitchLinear). No torch: the torch reference check of the real
block 0..3 is scripts/qwen4_exp_reference_check.py (Mac Studio)."""
import json
import math
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import yaml
from mlx.utils import tree_flatten

from suc.analysis.sensitivity import set_weight
from suc.io.safetensors_store import SafetensorsStore, TensorInfo
from suc.models import mlx_qwen4_exp as q4
from suc.models.adapters import Qwen4ExpAdapter, get_adapter
from suc.models.streaming import load_lazy, stream_blocks
from tiny_models import tiny_qwen4_exp_config, write_tiny_qwen4_exp

FIX = Path(__file__).parent / "fixtures"
REAL_CFG = json.loads((FIX / "qwen3.8_flash_next_config.json").read_text())
REAL = json.loads((FIX / "qwen3.8_flash_next_tensors.json").read_text())
_ITEM = {"BF16": 2, "I64": 8, "F32": 4}


def _fake_store(tensors: dict) -> SafetensorsStore:
    """Header-only store of the real checkpoint (one fake shard, tensors laid out back to back)."""
    infos, off = {}, 0
    for n, (dt, shape) in sorted(tensors.items()):
        nb = math.prod(shape) * _ITEM[dt]
        infos[n] = TensorInfo(n, dt, tuple(shape), Path("/nonexistent"), off, nb)
        off += nb
    return SafetensorsStore(infos)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return write_tiny_qwen4_exp(tmp_path_factory.mktemp("q4") / "model")


# ------------------------------------------------------------------ real checkpoint layout (no weights)

def test_hash_constants_equal_the_real_checkpoint_buffers():
    a = q4.TextArgs.from_dict(REAL_CFG["text_config"])
    buf = REAL["i64_buffers"]
    p = "model.language_model.layers.1.ple.ple_embedding."
    assert q4.layer_multipliers(a.vocab_size, a.ngram_size, 0, a.seed) == buf[p + "layer_multipliers"]
    sizes, offsets, rows = q4.ngram_head_sizes(a, 0)
    assert sizes == buf[p + "ngram_heads_vocab_sizes"] and offsets == buf[p + "ngram_heads_offsets"]
    shard = REAL["tensors"][p + "ngram_embedding.shard_0.weight"][1]
    assert rows == a.split_ngram_parts * shard[0] == 320_001_536 and shard[1] * a.ngram_heads == a.ple_embed_dim


def test_adapter_on_the_real_checkpoint():
    names = list(REAL["tensors"])
    assert len(names) == 1658
    total = sum(math.prod(s) * _ITEM[d] for d, s in REAL["tensors"].values())
    assert total == REAL["index_total_size"]
    ad = get_adapter(REAL_CFG)
    assert isinstance(ad, Qwen4ExpAdapter) and ad.num_layers == 48 and ad.num_experts == 512
    ad.validate(names)
    hf = _fake_store(REAL["tensors"])
    st = ad.source_store(hf)
    ad.validate(st.names())                                   # the SUC namespace validates too
    assert not [n for n in st.names() if n.startswith(("model.visual.", "mtp.", "model.language_model."))]
    dropped = sum(i.nbytes for n in hf.names() if ad.is_dropped(n) for i in [hf.info(n)])
    assert st.total_bytes() == hf.total_bytes() - dropped       # every kept byte maps exactly once
    specs = [s for s in ad.linears(st.names()) if s.eligible]
    elig = sum(math.prod(st.info(s.name).shape) for s in specs)
    assert elig == 123_738_521_600                            # docs/arch/qwen4_exp.md §6 / §10
    roles = {}
    for s in specs:
        roles[s.role] = roles.get(s.role, 0) + 1
    assert roles["expert_gate"] == roles["expert_up"] == roles["expert_down"] == 48 * 512
    assert roles["linattn_qkv"] == 36 and roles["attn_q"] == 12 and roles["ple_key"] == 1
    assert len(ad.table_tensors(st.names())) == 128
    # expert slices are disjoint, contiguous runs of the fused stacks
    g = st.info("model.layers.0.mlp.experts.7.gate_proj.weight")
    u = st.info("model.layers.0.mlp.experts.7.up_proj.weight")
    f = hf.info("model.language_model.layers.0.mlp.experts.gate_up_proj")
    assert g.shape == u.shape == (640, 2560) and g.offset == f.offset + 7 * 6_553_600 and u.offset == g.offset + g.nbytes
    with pytest.raises(ValueError, match="unclassified"):
        ad.validate(names + ["model.language_model.layers.0.mystery.weight"])


def test_mlx_parameters_map_one_to_one_onto_the_real_tensors():
    """Every MLX parameter of the real config (lazy, never materialized) has exactly the shape of its SUC-namespace
    tensor (expert stacks = the stacked per-expert slices), and every non-table tensor is a parameter."""
    ad = get_adapter(REAL_CFG)
    st = ad.source_store(_fake_store(REAL["tensors"]))
    model = q4.Model(q4.ModelArgs.from_dict(REAL_CFG))
    params = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    want = {}
    for n in st.names():
        if model.is_table_tensor(n):
            continue
        if m := re.match(r"^(.*)\.experts\.(\d+)\.(\w+)\.weight$", n):
            key = f"{m.group(1)}.switch_mlp.{m.group(3)}.weight"
            want[key] = (ad.num_experts, *st.info(n).shape)
        else:
            want[n] = st.info(n).shape
    assert params == want


# ------------------------------------------------------------------ tiny model: self-consistency

def test_virtual_store_slices_equal_the_fused_stacks(tiny_dir):
    hf = SafetensorsStore.from_dir(tiny_dir)
    ad = get_adapter(json.loads((tiny_dir / "config.json").read_text()))
    ad.validate(hf.names())
    st = ad.source_store(hf)
    gu = np.array(hf.load("model.language_model.layers.2.mlp.experts.gate_up_proj"))
    dn = np.array(hf.load("model.language_model.layers.2.mlp.experts.down_proj"))
    for e in range(8):
        p = f"model.layers.2.mlp.experts.{e}."
        assert np.array_equal(np.array(st.load(p + "gate_proj.weight")), gu[e, :32])
        assert np.array_equal(np.array(st.load(p + "up_proj.weight")), gu[e, 32:])
        assert np.array_equal(np.array(st.load(p + "down_proj.weight")), dn[e])


def test_streamed_full_and_cached_forward_agree(tiny_dir):
    model, _ = load_lazy(tiny_dir)
    mx.random.seed(3)
    x = mx.random.randint(2, 128, (2, 40))
    x[:, 17] = 1                                              # an EOS inside: n-gram context resets
    full = model(x)
    r = stream_blocks(model, x, return_logits=True, release=False, micro_batch=1)
    assert mx.array_equal(r["logits"], full)                  # identical ops: bit-exact
    cache = model.make_cache()
    steps = [model(x[:, :13], cache=cache)] + [model(x[:, t:t + 1], cache=cache) for t in range(13, 40)]
    inc = mx.concatenate(steps, axis=1)
    assert mx.abs(inc - full).max().item() < 1e-4            # float32 checkpoint; QSA sparse from 12 tokens on
    # the indexer is active at this length: a dense-attention model gives different logits
    for layer in model.layers:
        if "self_attn" in layer:
            layer.self_attn.indexer.topk = 10 ** 6
    assert mx.abs(model(x) - full).max().item() > 1e-3


def test_bf16_checkpoint_streamed_bit_exact(tmp_path):
    d = write_tiny_qwen4_exp(tmp_path / "m", seed=1, dtype=mx.bfloat16)
    model, _ = load_lazy(d)
    x = mx.random.randint(2, 128, (3, 20))
    full = model(x)
    r = stream_blocks(model, x, return_logits=True, micro_batch=2)   # blocks released after use
    assert full.dtype == mx.bfloat16 and mx.array_equal(r["logits"], full)


def _hf_selection(q: np.ndarray, bk: np.ndarray, ratio: int, topk: int, t_kv: int):
    """Port of the HF QSA loop (modular_qwen4_exp.py Qwen4ExpTextQSAIndexer.forward) for one sequence starting at
    position 0. q (L, nh, d), bk (n_blocks, d) block keys. Returns (mask (L, T) bool, rows with a tie at the top-k
    boundary)."""
    L = q.shape[0]
    mask = np.zeros((L, t_kv), dtype=bool)
    ties = set()
    for t in range(L):
        visible = np.arange(t + 1)
        nb = visible.size // ratio
        sel = []
        if nb > 0:
            blocks = visible[: nb * ratio].reshape(nb, ratio)
            scores = np.maximum(q[t] @ bk[:nb].T, 0).sum(0) / math.sqrt(q.shape[-1])
            k = min(topk, nb)
            order = np.argsort(-scores, kind="stable")
            if k < nb and scores[order[k - 1]] == scores[order[k]]:
                ties.add(t)
            sel = list(blocks[order[:k]].reshape(-1))
        mask[t, sel + list(visible[nb * ratio:])] = True
    return mask, ties


def test_indexer_matches_the_hf_selection_loop(tiny_dir):
    model, _ = load_lazy(tiny_dir)
    ix = model.layers[3].self_attn.indexer
    mx.random.seed(5)
    L = 64
    x = mx.random.normal((1, L, 64))
    sel = ix(x, 0)
    assert sel is not None and sel.shape == (1, 1, L, L)
    qk = ix.index_qk_proj(x)
    d = ix.head_dim
    q = ix.q_layernorm(qk[..., : ix.n_heads * d].reshape(1, L, ix.n_heads, d)).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, ix.rot, traditional=False, base=ix.base, scale=1.0, offset=0)[0].transpose(1, 0, 2)
    nb = L // ix.ratio
    pooled = qk[0, : nb * ix.ratio, ix.n_heads * d:].reshape(nb, ix.ratio, d).mean(axis=1)
    bk = q4._rope_at(ix.k_layernorm(pooled), mx.arange(nb) * ix.ratio, ix.rot, ix.base)
    ref, ties = _hf_selection(np.array(q), np.array(bk), ix.ratio, ix.topk, L)
    got = np.array(sel[0, 0])
    rows = [t for t in range(L) if t not in ties]
    assert len(rows) > L // 2
    assert np.array_equal(got[rows], ref[rows])
    assert got.sum(-1).max() <= ix.topk * ix.ratio + ix.ratio - 1       # budget + incomplete tail
    assert ix(x[:, :11], 0) is None                                     # <= budget + ratio - 1 tokens: dense


def test_ngram_ids_with_context_and_eos():
    a = q4.TextArgs.from_dict(tiny_qwen4_exp_config()["text_config"])
    rng = np.random.default_rng(0)
    tok = rng.integers(2, 128, (2, 30))
    tok[0, 9] = a.eos
    eos_ctx = np.full((2, 2), a.eos)
    full = q4.ngram_ids(tok, eos_ctx, a, 0)
    part = q4.ngram_ids(tok[:, 20:], tok[:, 18:20], a, 0)
    assert np.array_equal(full[:, 20:], part)                            # 2 cached ids reproduce the context
    fresh = q4.ngram_ids(tok[:1, 10:], eos_ctx[:1], a, 0)
    assert np.array_equal(full[:1, 10:], fresh)                          # context resets after an EOS
    sizes, offsets, rows = q4.ngram_head_sizes(a, 0)
    assert full.min() >= 0 and full.max() < rows
    assert all(offsets[h] <= full[..., h].min() and full[..., h].max() < offsets[h] + sizes[h] for h in range(4))


def test_gated_delta_custom_vjp_equals_the_ops_gradient():
    from mlx_lm.models.gated_delta import gated_delta_ops

    mx.random.seed(0)
    B, T, Hk, Hv, Dk, Dv = 1, 6, 2, 4, 32, 16
    q, k = mx.random.normal((B, T, Hk, Dk)) * 0.2, mx.random.normal((B, T, Hk, Dk)) * 0.2
    v = mx.random.normal((B, T, Hv, Dv))
    g, beta = mx.random.uniform(0.5, 1.0, (B, T, Hv)), mx.random.uniform(0, 1, (B, T, Hv))
    s0 = mx.zeros((B, Hv, Dv, Dk))
    f1 = lambda q_, v_: q4._gated_delta(q_, k, v_, g, beta, s0)[0].sum()      # noqa: E731
    f2 = lambda q_, v_: gated_delta_ops(q_, k, v_, g, beta, s0)[0].sum()      # noqa: E731
    assert mx.allclose(q4._gated_delta(q, k, v, g, beta, s0)[0], gated_delta_ops(q, k, v, g, beta, s0)[0], atol=1e-5)
    g1, g2 = mx.grad(f1, argnums=(0, 1))(q, v), mx.grad(f2, argnums=(0, 1))(q, v)
    assert all(mx.allclose(a, b, atol=1e-6) for a, b in zip(g1, g2))



def test_gated_delta_qk_normalization_is_the_hf_l2norm(monkeypatch):
    """q, k reach the recurrence as HF/FLA l2norm(x) = x * rsqrt(sum(x^2) + 1e-6) (q also / sqrt(Dk)): eps on the
    SUM, not on the mean (the mlx-lm rms_norm(x, 1e-6) shortcut is off by 1e-6 * Dk, visible on small-norm rows)."""
    a = q4.ModelArgs.from_dict(tiny_qwen4_exp_config()).text
    net = q4.GatedDeltaNet(a)
    seen = {}

    def spy(q, k, v, g, beta, state):
        seen.update(q=q, k=k)
        return mx.zeros(v.shape, dtype=v.dtype), state

    monkeypatch.setattr(q4, "_gated_delta", spy)
    mx.random.seed(1)
    B, S, Dk = 1, 5, net.head_k_dim
    x = mx.random.normal((B, S, a.hidden_size))
    net.conv1d.weight = mx.random.normal(net.conv1d.weight.shape) * 1e-3     # small-norm q / k rows
    net(x)
    conv_in = mx.concatenate([mx.zeros((B, net.conv_kernel_size - 1, net.conv_dim)), net.in_proj_qkv(x)], axis=1)
    conv = mx.sigmoid(c := q4.depthwise_causal_conv(conv_in, net.conv1d.weight, S)) * c
    q = conv[..., :net.key_dim].reshape(B, S, -1, Dk)
    k = conv[..., net.key_dim:2 * net.key_dim].reshape(B, S, -1, Dk)
    l2 = lambda t: t * mx.rsqrt((t * t).sum(-1, keepdims=True) + 1e-6)      # noqa: E731
    assert float(mx.abs(seen["q"] - l2(q) / math.sqrt(Dk)).max()) <= 1e-6 * float(mx.abs(seen["q"]).max())
    assert float(mx.abs(seen["k"] - l2(k)).max()) <= 1e-6 * float(mx.abs(seen["k"]).max())

# ------------------------------------------------------------------ suc compress -> SUC1 -> runtime

def _q4_cfg(tmp_path: Path, model_dir: Path, plan: dict) -> dict:
    return {"name": "c_q4", "seed": 7, "output_dir": str(tmp_path / "results"), "run_dir": str(tmp_path / "run"),
            "model": {"id": "tiny/qwen4_exp", "local_path": str(model_dir)},
            "calibration": {"seq_len": 32, "n_calibration": 16, "n_validation": 8, "n_ood": 4, "micro_batch": 4},
            "plan": plan, "eval": {"seq_len": 64, "max_tokens": 1024, "kl_sequences": 4, "chunk": 2},
            "finetune": None, "rotation_groups": True, "verify_tensors": 8}


XS = {"kind": "vq", "k": 16, "block_size": 2, "rotate_out": False, "kmeans_iters": 3, "residual_fraction": 0.02,
      "residual_structure": "columns", "codebook_scope": "expert_stack"}
XS_NAME = "vq_k16_g2_rot1s_xs+res0.02_activation_cols"


@pytest.fixture
def q4_env(tmp_path, monkeypatch):
    import suc.pipeline.compress as comp
    from test_calibrate_run import FakeTokenizer, _docs

    mx.random.seed(0)
    tokens = mx.random.randint(2, 128, (2048,), dtype=mx.int32)
    monkeypatch.setattr(comp, "load_wikitext2", lambda split: ("unused", "wikihash"))
    monkeypatch.setattr(comp, "tokenize", lambda tok, text: tokens)
    docs, ood = _docs()
    model_dir = write_tiny_qwen4_exp(tmp_path / "model", seed=2, dtype=mx.bfloat16)

    def run(plan: dict):
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(_q4_cfg(tmp_path, model_dir, plan)))
        return json.loads(comp.run_compress(p, tokenizer=FakeTokenizer(), docs=docs, ood_docs=ood).read_text())

    return tmp_path, model_dir, run


def test_compress_uniform_expert_stacks_then_python_runtime(q4_env):
    from suc.io.suc1 import SucModel
    from suc.runtime.suc1_model import greedy, load_suc1_model

    tmp_path, model_dir, run = q4_env
    r = run({"mode": "uniform", "codec": XS_NAME, "allocation_unit": "expert_stack", "candidates": [XS],
             "embedding_codec": {"kind": "affine", "bits": 4}})
    assert r["architecture"]["family"] == "qwen4_exp" and np.isfinite(r["quality"]["perplexity"]["ppl"])
    n_elig = 3 * 3 + 4 + 2 + 4 * (3 + 3 * 8)    # linattn (3 layers), attn q/k/v/o, PLE, per layer shared + experts
    assert r["plan"]["by_codec"] == {XS_NAME: n_elig}
    sm = SucModel.load(r["container"]["path"], verify=True)
    names = set(sm.names())
    assert not [n for n in names if n.startswith(("model.visual", "mtp.", "model.language_model."))]
    assert {q4.ngram_shard_name(1, j) for j in range(2)} <= names               # tables kept raw (mmap)
    assert sm.descriptor("model.layers.1.ple.key_proj.weight")["representation_type"] == "vq"
    assert sm.descriptor("model.layers.0.mlp.gate.weight")["representation_type"] == "raw"
    ids = {tuple(sm.descriptor(f"model.layers.2.mlp.experts.{e}.down_proj.weight")["codebook_ids"]) for e in range(8)}
    assert len(ids) == 1                                                        # one codebook per expert stack
    assert r["preflight"]["memory"]["mmap_table_bytes"] == 2 * 2032 * 16 * 2
    # Python runtime from the SUC1 file alone (SucLinear / SucSwitchLinear on the codes) vs the decoded model
    model, _, rep = load_suc1_model(r["container"]["path"], group=False)
    kinds = {c["kind"] for c in rep["converted"]}
    assert {"SucSwitchLinear"} <= kinds and any(k.startswith("Suc") and k != "SucSwitchLinear" for k in kinds)
    ref, _ = load_lazy(model_dir)                    # + the reference decode of every encoded tensor
    for n in sm.names():
        if sm.descriptor(n)["representation_type"] != "raw":
            set_weight(ref, n, sm.decode_tensor(n))
    x = mx.random.randint(2, 128, (2, 24))
    lr, lq = ref(x).astype(mx.float32), model(x).astype(mx.float32)
    p = mx.softmax(lr, axis=-1)
    kl = (p * (mx.log(p + 1e-9) - mx.log(mx.softmax(lq, axis=-1) + 1e-9))).sum(-1).mean().item()
    assert kl < 2e-2, kl
    out = greedy(model, [5, 6, 7, 8], 4)
    assert len(out) == 4 and all(0 <= t < 128 for t in out)
    model2, _, _ = load_suc1_model(r["container"]["path"], group=True)
    assert mx.abs(model2(x).astype(mx.float32) - lq).max().item() < 0.25


NON_EXPERT_ROLES = ["attn_q", "attn_k", "attn_v", "attn_o", "linattn_qkv", "linattn_z", "linattn_out",
                    "shared_expert_gate", "shared_expert_up", "shared_expert_down", "ple_key", "ple_value"]


def test_compress_uniform_role_codecs(q4_env):
    """plan.role_codecs: affine 8-bit non-expert linears, VQ expert stacks, BF16 embedding, affine 8-bit LM head
    (a fixed per-role plan without the Fisher pass), through encode -> SUC1 -> Python runtime."""
    from suc.io.suc1 import SucModel
    from suc.runtime.suc1_model import load_suc1_model

    _, _, run = q4_env
    a8 = {"kind": "affine", "bits": 8, "group_size": 32}
    r = run({"mode": "uniform", "codec": XS_NAME, "allocation_unit": "expert_stack",
             "candidates": [XS, a8, {"kind": "bf16"}],
             "role_codecs": {"affine_b8_g32": NON_EXPERT_ROLES, "bf16": ["embed"]},
             "embedding_codec": a8})
    assert r["plan"]["by_codec"] == {XS_NAME: 4 * 3 * 8, "affine_b8_g32": 3 * 3 + 4 + 2 + 4 * 3}
    sm = SucModel.load(r["container"]["path"], verify=True)
    rep = lambda n: sm.descriptor(n)["representation_type"]  # noqa: E731
    assert rep("model.layers.1.ple.key_proj.weight") == "affine"
    assert rep("model.layers.2.mlp.experts.0.down_proj.weight") == "vq"
    assert rep("lm_head.weight") == "affine" and rep("model.embed_tokens.weight") == "bf16"
    assert r["quality"]["kl"]["kl_mean"] < 0.05, r["quality"]["kl"]
    model, _, _ = load_suc1_model(r["container"]["path"], group=False)
    assert np.isfinite(model(mx.array([[5, 6, 7, 8]])).astype(mx.float32)).all().item()


def test_role_codecs_validation():
    from suc.config import compress_config_from_dict

    a8 = {"kind": "affine", "bits": 8}
    base = {"name": "x", "model": {"id": "tiny/x", "local_path": "/tmp"},
            "plan": {"mode": "uniform", "codec": XS_NAME, "candidates": [XS, a8], "allocation_unit": "expert_stack",
                     "embedding_codec": a8}}
    ok = compress_config_from_dict({**base, "plan": {**base["plan"], "role_codecs": {"affine_b8_g64": ["attn_q"]}}})
    assert ok.plan.uniform_codec("attn_q") == "affine_b8_g64" and ok.plan.uniform_codec("expert_up") == XS_NAME
    for bad in ({"nope": ["attn_q"]}, {"affine_b8_g64": ["not_a_role"]}, {"affine_b8_g64": []},
                {"affine_b8_g64": ["attn_q"], XS_NAME: ["attn_q"]}):
        with pytest.raises(ValueError):
            compress_config_from_dict({**base, "plan": {**base["plan"], "role_codecs": bad}})


def test_compress_greedy_fisher_through_the_stream(q4_env):
    """Greedy plan: the streamed Fisher differentiates every block (gated-delta custom VJP, PLE, indexer)."""
    _, _, run = q4_env
    r = run({"mode": "greedy", "budget_bpw": 4.0, "allocation_unit": "expert_stack",
             "candidates": [{"kind": "affine", "bits": 4, "group_size": 32}, {"kind": "affine", "bits": 2, "group_size": 32}, XS],
             "embedding_codec": {"kind": "affine", "bits": 4}, "fisher_sequences": 8, "fisher_micro_batch": 2})
    assert r["plan"]["allocated_bpw"] <= 4.0 + 1e-9 and np.isfinite(r["quality"]["kl"]["kl_mean"])


def test_final_config_loads_and_fits_the_real_shapes():
    from suc.config import load_compress_config
    from suc.pipeline.compress import _codec_shape_error

    repo = Path(__file__).resolve().parents[2]
    cfg = _config(repo, "compress_qwen3.8_flash_next.yaml")
    assert cfg.model.revision == "de4b8e4d43b917e7706784d8bb445c9af86a3540"
    assert cfg.plan.allocation_unit == "expert_stack" and cfg.plan.mode == "uniform"
    assert (cfg.ngram_codec.kind, cfg.ngram_codec.bits, cfg.ngram_codec.scale_dtype) == ("row_affine", 4, "f16")
    c = cfg.plan.codec_configs()[cfg.plan.codec]
    assert c.codebook_scope == "expert_stack" and c.k * c.block_size * 2 <= 16 * 1024
    ad = get_adapter(REAL_CFG)
    st = ad.source_store(_fake_store(REAL["tensors"]))
    shapes = {st.info(s.name).shape for s in ad.linears(st.names()) if s.eligible}
    assert not [s for s in shapes if _codec_shape_error(c, s)]
    for n in ("model.embed_tokens.weight", "lm_head.weight"):
        assert _codec_shape_error(cfg.plan.embedding_codec, st.info(n).shape) is None


def test_ngram_codec_config_validation():
    from suc.config import compress_config_from_dict
    from suc.pipeline.compress import config_delta

    base = {"name": "x", "model": {"id": "tiny/x", "local_path": "/tmp"},
            "plan": {"mode": "uniform", "codec": XS_NAME, "candidates": [XS], "allocation_unit": "expert_stack"}}
    cfg = compress_config_from_dict(base)
    assert cfg.ngram_codec.kind == "raw" and "ngram_codec" not in config_delta(cfg)   # default: fingerprints unchanged
    assert compress_config_from_dict({**base, "ngram_codec": {"kind": "row_affine", "bits": 4}}).ngram_codec.bits == 4
    for bad in ({"kind": "vq"}, {"kind": "row_affine", "bits": 3}, {"kind": "row_affine", "group_size": 64}):
        with pytest.raises(ValueError):
            compress_config_from_dict({**base, "ngram_codec": bad})


def test_reference_check_script_mlx_side(tiny_dir, tmp_path):
    """The MLX half of scripts/qwen4_exp_reference_check.py against a reference made by the MLX model itself
    (float32): zero error in float32, small in bfloat16. The torch half runs on the Mac Studio."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("q4ref", Path(__file__).resolve().parents[2]
                                                  / "scripts/qwen4_exp_reference_check.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    tokens = np.random.default_rng(0).integers(2, 128, (1, 32)).astype(np.int64)
    tokens[0, 16] = 1
    outs = ref.mlx_block_outputs(tiny_dir, tokens, 4, mx.float32)
    assert set(outs) == {"block_00", "block_01", "block_02", "block_03", "logits"}
    npz = tmp_path / "ref.npz"
    np.savez(npz, tokens=tokens, dtype="float32", blocks=4, seq=32, seed=0, **outs)
    p = ref.mlx_side(ref.argparse.Namespace(model_dir=str(tiny_dir), ref=str(npz), revision="tiny",
                                            output_dir=str(tmp_path / "results")))
    r = json.loads(p.read_text())["mlx_vs_torch"]
    assert max(v["max_abs"] for v in r["float32"].values()) == 0.0
    assert r["bfloat16"]["logits"]["top1"] > 0.8 and r["bfloat16"]["block_03"]["rel_fro"] < 5e-2


def _config(repo, name):
    """Load experiments/configs/<name>; skip when the file is not in this checkout (the public snapshot omits the
    configs of discarded variants)."""
    from suc.config import load_compress_config
    path = repo / "experiments/configs" / name
    if not path.exists():
        pytest.skip(f"{name} is not in this checkout")
    return load_compress_config(path)
