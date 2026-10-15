"""Transformer integration (spec §22, §34): a Qwen3 whose linears are SucLinear modules == the model with decoded
weights. Oracle = the same model with every converted weight replaced by codec.decode(encoding)."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from suc.analysis.sensitivity import set_weight
from suc.evaluation.baselines import logits_kl
from suc.evaluation.speed import measure_generation
from suc.kernels.suc_linear import SucLinear
from suc.quant.codecs import make_codec
from suc.quant.residual import ResidualCodec
from suc.runtime.compressed_model import (SucResidualLinear, convert_to_compressed, merge_reports,
                                          module_from_encoded, resident_weight_bytes)
from tiny_models import tiny_qwen3

PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
VQ = {"kind": "vq", "k": 16, "block_size": 4, "rotate": True, "kmeans_iters": 5}
ADD2 = {"kind": "additive", "k": 16, "block_size": 4, "num_codebooks": 2, "rotate": True, "kmeans_iters": 5}


def names(n_layers, layers=None):
    return [f"model.layers.{i}.{p}.weight" for i in (layers if layers is not None else range(n_layers)) for p in PROJ]


def codec_of(cfg, residual):
    base = make_codec(cfg)
    return base if residual is None else ResidualCodec(base, residual[0], value_dtype=residual[1])


def get(model, name):
    *path, attr = name.split(".")
    cur = model
    for p in path:
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    return cur[attr]


def encode_all(model, tensor_names, codec, seed=0):
    return {n: codec.encode(get(model, n), seed=seed + i) for i, n in enumerate(tensor_names)}


def reference_and_compressed(cfg, residual, dtype, layers=None):
    ref, comp = tiny_qwen3(seed=0), tiny_qwen3(seed=0)
    if dtype != mx.float32:
        ref.set_dtype(dtype)
        comp.set_dtype(dtype)
    codec = codec_of(cfg, residual)
    enc = encode_all(comp, names(2, layers), codec)
    for n, e in enc.items():
        set_weight(ref, n, codec.decode(e))
    mx.eval(ref.parameters())
    report = convert_to_compressed(comp, enc)
    return ref, comp, report


def tokens(b=3, t=12, seed=0):
    return mx.array(np.random.default_rng(seed).integers(0, 128, size=(b, t)).astype(np.int32))


# ------------------------------------------------------------------------------------- module level

@pytest.mark.parametrize("cfg", [VQ, ADD2])
@pytest.mark.parametrize("residual", [None, (0.02, "fp16"), (0.02, "int8"), (0.0, "fp16")])
@pytest.mark.parametrize("b", [1, 4, 16])
def test_module_matches_decoded_weight(cfg, residual, b):
    rng = np.random.default_rng(1)
    w = mx.array(rng.normal(size=(96, 128)).astype(np.float32)) / np.sqrt(128)
    codec = codec_of(cfg, residual)
    e = codec.encode(w, seed=3)
    m = module_from_encoded(e)
    assert isinstance(m, SucResidualLinear) == (residual is not None)
    x = mx.random.normal((b, 128), key=mx.random.key(2))
    ref = x @ codec.decode(e).T
    y = m(x)
    rel = float(mx.linalg.norm(y - ref) / mx.linalg.norm(ref))
    assert y.shape == (b, 96) and rel < 1e-5, rel


@pytest.mark.parametrize("value_dtype", ["fp16", "int8"])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize("b", [1, 5, 33])
def test_residual_kernel_is_exact_sparse_matmul(value_dtype, dtype, b):
    w = mx.random.normal((80, 64), key=mx.random.key(0))
    codec = ResidualCodec(make_codec(VQ), 0.05, value_dtype=value_dtype)
    e = codec.encode(w, seed=0)
    m = module_from_encoded(e)
    v = e.arrays["res_val"].astype(mx.float32) * (e.arrays["res_scale"].astype(mx.float32)
                                                   if "res_scale" in e.arrays else 1.0)
    r = mx.zeros(80 * 64).at[e.arrays["res_pos"].astype(mx.int32)].add(v).reshape(80, 64)
    x = mx.random.normal((b, 64), key=mx.random.key(1)).astype(dtype)
    y0 = mx.random.normal((b, 80), key=mx.random.key(2))
    ref = np.array(y0) + np.array(x.astype(mx.float32)) @ np.array(r).T
    got = m.residual(x, y0)
    m.residual_path = "mlx"
    alt = m.residual(x, y0)
    assert got.dtype == mx.float32
    assert np.allclose(np.array(got), ref, atol=1e-5) and np.allclose(np.array(alt), ref, atol=1e-5)
    assert m.res_cols.dtype == mx.uint16 and m.res_row_ptr.dtype == mx.uint32 and m.n_residual == 256
    assert int(m.res_row_ptr[-1]) == 256


def test_residual_layout_rejects_unsorted_positions():
    w = mx.random.normal((16, 32), key=mx.random.key(0))
    e = ResidualCodec(make_codec(dict(VQ, k=8)), 0.1).encode(w, seed=0)
    e.arrays["res_pos"] = e.arrays["res_pos"][::-1]
    with pytest.raises(ValueError, match="sorted"):
        module_from_encoded(e)


def test_module_keeps_bias_and_leading_dims():
    w = mx.random.normal((32, 64), key=mx.random.key(0))
    e = make_codec(VQ).encode(w, seed=0)
    bias = mx.arange(32).astype(mx.float32)
    m = module_from_encoded(e, bias=bias)
    x = mx.random.normal((2, 3, 64), key=mx.random.key(1))
    ref = x @ make_codec(VQ).decode(e).T + bias
    assert np.allclose(np.array(m(x)), np.array(ref), atol=1e-4)


# ------------------------------------------------------------------------------------- whole model

@pytest.mark.parametrize("cfg,residual", [(VQ, None), (ADD2, None), (VQ, (0.01, "int8")), (ADD2, (0.01, "fp16"))])
@pytest.mark.parametrize("dtype,tol,kl_tol", [(mx.float32, 1e-4, 1e-8), (mx.bfloat16, 3e-2, 3e-4)])
def test_compressed_logits_match_decoded_model(cfg, residual, dtype, tol, kl_tol):
    ref, comp, report = reference_and_compressed(cfg, residual, dtype)
    assert report["tensors_converted"] == 14
    kind = SucResidualLinear if residual else SucLinear
    assert all(type(get(comp, n[:-7])) is kind for n in names(2))
    toks = tokens()
    lr, lc = ref(toks).astype(mx.float32), comp(toks).astype(mx.float32)
    rel = float(mx.linalg.norm(lc - lr) / mx.linalg.norm(lr))
    kl = logits_kl(lr, lc)
    assert rel < tol, rel
    assert kl["kl_mean"] < kl_tol and kl["top1_agreement"] > 0.95, kl


def test_mixed_representations_in_one_model():
    """§22: layer 0 SUC, layer 1 stays dense except an MLX-affine gate_proj; embedding untouched (tied)."""
    ref, comp = tiny_qwen3(seed=0), tiny_qwen3(seed=0)
    for m in (ref, comp):
        nn.quantize(m.model.layers[1].mlp, group_size=64, bits=4, class_predicate=lambda p, mod: p == "gate_proj")
    codec = make_codec(ADD2)
    enc = encode_all(comp, names(2, [0]), codec)
    for n, e in enc.items():
        set_weight(ref, n, codec.decode(e))
    convert_to_compressed(comp, enc)
    assert type(comp.model.layers[0].mlp.down_proj) is SucLinear
    assert type(comp.model.layers[1].mlp.gate_proj) is nn.QuantizedLinear
    assert type(comp.model.layers[1].mlp.up_proj) is nn.Linear
    assert type(comp.model.embed_tokens) is nn.Embedding
    toks = tokens()
    lr, lc = ref(toks), comp(toks)
    assert float(mx.linalg.norm(lc - lr) / mx.linalg.norm(lr)) < 1e-4
    with pytest.raises(TypeError, match="QuantizedLinear"):
        convert_to_compressed(comp, {"model.layers.1.mlp.gate_proj.weight": enc[names(2, [0])[4]]})


def test_report_and_dense_weights_dropped():
    ref, comp, report = reference_and_compressed(VQ, (0.01, "int8"), mx.bfloat16)
    dense = sum(get(ref, n).nbytes for n in names(2))
    assert report["bytes_before"] == dense
    assert 0 < report["bytes_after"] < dense
    assert report["model_bytes_before"] - report["model_bytes_after"] == dense - report["bytes_after"]
    assert resident_weight_bytes(comp) == report["model_bytes_after"]
    for n in names(2):
        assert "weight" not in get(comp, n[:-7])
    row = report["converted"][0]
    assert {"name", "codec", "kind", "shape", "bytes_before", "bytes_after", "n_residual"} <= set(row)
    assert row["n_residual"] > 0 and row["kind"] == "SucResidualLinear"
    merged = merge_reports([report, report])
    assert merged["tensors_converted"] == 28 and merged["bytes_before"] == 2 * dense


def test_conversion_releases_dense_memory():
    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2048, 2048, bias=False)

    h = Holder()
    mx.eval(h.parameters())
    e = make_codec({"kind": "vq", "k": 256, "block_size": 8, "rotate": True, "kmeans_iters": 2,
                    "fit_blocks": 4096}).encode(h.proj.weight, seed=0)
    mx.eval(e.arrays)
    rep = convert_to_compressed(h, {"proj.weight": e})
    assert type(h.proj) is SucLinear
    freed = rep["active_memory_before"] - rep["active_memory_after"]
    assert freed >= 0.9 * (2048 * 2048 * 4 - rep["bytes_after"]), rep


def test_rejects_bad_targets():
    comp = tiny_qwen3(seed=0)
    e = make_codec(VQ).encode(comp.model.layers[0].self_attn.q_proj.weight, seed=0)
    with pytest.raises(ValueError, match="shape"):
        convert_to_compressed(comp, {"model.layers.0.self_attn.k_proj.weight": e})
    with pytest.raises(KeyError):
        convert_to_compressed(comp, {"model.layers.0.self_attn.nope.weight": e})
    with pytest.raises(ValueError, match="expert"):
        convert_to_compressed(comp, {"model.layers.0.mlp.experts.3.gate_proj.weight": e})
    bf = make_codec({"kind": "bf16"}).encode(comp.model.layers[0].self_attn.q_proj.weight, seed=0)
    with pytest.raises(ValueError, match="vq or additive"):
        convert_to_compressed(comp, {"model.layers.0.self_attn.q_proj.weight": bf})


def test_compressed_model_generates():
    ref, comp, _ = reference_and_compressed(ADD2, (0.01, "int8"), mx.float32)
    r = measure_generation(comp, prompt_len=8, decode_tokens=4, batch_size=2, warmup=0, repeats=1, seed=1)
    q = measure_generation(ref, prompt_len=8, decode_tokens=4, batch_size=2, warmup=0, repeats=1, seed=1)
    assert r["decode_tok_s"] > 0
    assert r["repeats"][0]["tokens"] == q["repeats"][0]["tokens"]


# ------------------------------------------------------------------------------------- real model

TEXT = ("The history of the printing press begins in the fifteenth century, when movable metal type made it possible "
        "to reproduce books quickly. def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"
        "Solve for x: 3x + 7 = 22, so x = 5. Please summarise the main argument of the essay in two sentences.")


@pytest.mark.model
@pytest.mark.parametrize("cfg,residual", [({"kind": "vq", "k": 256, "block_size": 4, "rotate": True},
                                           (0.005, "int8")),
                                          ({"kind": "additive", "k": 256, "block_size": 8, "num_codebooks": 3,
                                            "rotate": True}, None)])
def test_qwen3_0p6b_compressed_matches_decoded(cfg, residual):
    """float32: compressed == decoded (exact up to rounding). bfloat16 (the deployed dtype): the compressed model is
    no farther from the float32 decoded model than the bf16 decoded model is (bf16 noise floor)."""
    from mlx_lm import load
    rev = "c1899de289a04d12100db370d81485cdf75e47ca"
    ref, tok = load("Qwen/Qwen3-0.6B", revision=rev)
    comp, _ = load("Qwen/Qwen3-0.6B", revision=rev)
    codec = codec_of(dict(cfg, kmeans_iters=4, fit_blocks=32768), residual)
    layers = [0, 2, 13, 27]
    enc = encode_all(comp, names(28, layers), codec)
    dec = {n: codec.decode(e) for n, e in enc.items()}
    ids = tok.encode(TEXT)
    toks = mx.array([ids[i:i + 24] for i in range(0, 72, 24)])
    for n, w in dec.items():
        set_weight(ref, n, w)
    l_ref16 = ref(toks)
    ref.set_dtype(mx.float32)
    for n, w in dec.items():
        set_weight(ref, n, w)
    l_ref32 = ref(toks)
    rep = convert_to_compressed(comp, enc)
    assert rep["tensors_converted"] == 28
    l_c16 = comp(toks)
    comp.set_dtype(mx.float32)
    l_c32 = comp(toks)
    exact = logits_kl(l_ref32, l_c32)
    assert exact["kl_mean"] < 1e-5 and exact["top1_agreement"] == 1.0, exact
    floor, got = logits_kl(l_ref32, l_ref16), logits_kl(l_ref32, l_c16)
    assert got["kl_mean"] <= 1.5 * floor["kl_mean"] + 1e-3, (got, floor)


# ------------------------------------------------------------------------------------- shared-input groups
from suc.evaluation.reconstruct import rotation_seeds  # noqa: E402
from suc.kernels.suc_linear import SucFusedLinear  # noqa: E402
from suc.runtime.compressed_model import SucGroupMember, SucInputGroup, group_projections  # noqa: E402


def encode_grouped(model, tensor_names, codec, seed=0, mixed=None):
    """Like encode_all, but q/k/v and gate/up share seed_in (suc.evaluation.reconstruct.rotation_seeds)."""
    out = {}
    for i, n in enumerate(tensor_names):
        c = mixed(n) if mixed is not None else codec
        out[n] = c.encode(get(model, n), seed=seed + i, **rotation_seeds(n, seed))
    return out


def grouped_reference_and_compressed(cfg, residual, dtype, *, fuse=True, mixed=None, incremental=False):
    ref, comp = tiny_qwen3(seed=0), tiny_qwen3(seed=0)
    if dtype != mx.float32:
        ref.set_dtype(dtype)
        comp.set_dtype(dtype)
    codec = codec_of(cfg, residual)
    pick = (lambda n: mixed(n, codec)) if mixed is not None else None
    enc = encode_grouped(comp, names(2), codec, mixed=pick)
    for n, e in enc.items():
        set_weight(ref, n, (pick(n) if pick else codec).decode(e))
    mx.eval(ref.parameters())
    if incremental:
        reps = [convert_to_compressed(comp, {n: e}, group=False) for n, e in enc.items()]
        report = merge_reports(reps)
        report["groups"] = group_projections(comp, fuse=fuse)
    else:
        report = convert_to_compressed(comp, enc, fuse=fuse)
    return ref, comp, report


@pytest.mark.parametrize("cfg,residual", [(VQ, None), (ADD2, None), (VQ, (0.01, "int8"))])
@pytest.mark.parametrize("fuse", [True, False])
@pytest.mark.parametrize("dtype,tol,kl_tol", [(mx.float32, 1e-4, 1e-7), (mx.bfloat16, 3e-2, 3e-4)])
def test_grouped_model_matches_decoded(cfg, residual, fuse, dtype, tol, kl_tol):
    ref, comp, report = grouped_reference_and_compressed(cfg, residual, dtype, fuse=fuse)
    assert len(report["groups"]) == 4              # 2 layers × (qkv, gate_up)
    for L in range(2):
        lay = comp.model.layers[L]
        assert isinstance(lay.self_attn.suc_qkv, SucInputGroup) and isinstance(lay.mlp.suc_gate_up, SucInputGroup)
        assert all(type(lay.self_attn[p]) is SucGroupMember for p in ("q_proj", "k_proj", "v_proj"))
        assert type(lay.self_attn.o_proj) is (SucResidualLinear if residual else SucLinear)
        fused = fuse          # residual members are fused too (SucFusedResidual, compressed_model.FUSE_RESIDUAL)
        assert lay.self_attn.suc_qkv.fused_parts == int(fused)
        assert len(lay.self_attn.suc_qkv.parts) == (1 if fused else 3)
    toks = tokens()
    lr, lc = ref(toks).astype(mx.float32), comp(toks).astype(mx.float32)
    rel = float(mx.linalg.norm(lc - lr) / mx.linalg.norm(lr))
    kl = logits_kl(lr, lc)
    assert rel < tol, rel
    assert kl["kl_mean"] < kl_tol and kl["top1_agreement"] > 0.95, kl


def test_grouping_keeps_memory_and_matches_ungrouped():
    ref, comp, report = grouped_reference_and_compressed(VQ, None, mx.float32, incremental=True)
    _, flat, _ = grouped_reference_and_compressed(VQ, None, mx.float32, fuse=False, incremental=True)
    assert len(report["groups"]) == 4 and all(g["fused_parts"] == 1 for g in report["groups"])
    # the dropped member arrays are not kept alive: fused (concatenated) == unfused bytes, minus shared signs_in
    signs = 2 * (2 * 64 + 1 * 64) * 1      # per layer: 2 extra signs_in in qkv, 1 in gate_up (in = 64), int8
    assert resident_weight_bytes(flat) - resident_weight_bytes(comp) == signs
    assert report["model_bytes_after"] - resident_weight_bytes(comp) == signs
    toks = tokens()
    a, b = comp(toks), flat(toks)
    assert float(mx.linalg.norm(a - b) / mx.linalg.norm(b)) < 1e-5


def test_mixed_codecs_in_one_group():
    """v_proj with a residual and k in a different layout: q and v (same code layout) are fused into one
    SucFusedResidual, k runs alone on the shared V x."""
    other = make_codec(dict(VQ, block_size=8))
    res = ResidualCodec(make_codec(VQ), 0.02)
    pick = lambda n, base: res if "v_proj" in n else other if "k_proj" in n else base
    ref, comp, report = grouped_reference_and_compressed(VQ, None, mx.float32, mixed=pick)
    g = comp.model.layers[0].self_attn.suc_qkv
    assert g.fused_parts == 1 and len(g.parts) == 2
    assert comp.model.layers[0].mlp.suc_gate_up.fused_parts == 1
    toks = tokens()
    lr, lc = ref(toks), comp(toks)
    assert float(mx.linalg.norm(lc - lr) / mx.linalg.norm(lr)) < 1e-4


def test_group_cache_is_identity_keyed_and_single_use():
    ref, comp, _ = grouped_reference_and_compressed(VQ, None, mx.float32)
    att_c, att_r = comp.model.layers[0].self_attn, ref.model.layers[0].self_attn
    x = mx.random.normal((2, 5, 64), key=mx.random.key(0))
    y = mx.random.normal((2, 5, 64), key=mx.random.key(1))
    close = lambda a, b: float(mx.linalg.norm(a - b) / mx.linalg.norm(b)) < 1e-5
    # interleaved inputs, repeated calls, out-of-order members: always the member's own result for that input
    assert close(att_c.k_proj(x), att_r.k_proj(x))
    assert close(att_c.q_proj(y), att_r.q_proj(y))
    assert close(att_c.v_proj(x), att_r.v_proj(x))
    assert close(att_c.q_proj(y), att_r.q_proj(y))
    assert close(att_c.q_proj(y), att_r.q_proj(y))
    xc = mx.array(x)                          # equal values, different object: recomputed, still correct
    assert close(att_c.v_proj(xc), att_r.v_proj(x))
    g = att_c.suc_qkv
    att_c.q_proj(x), att_c.k_proj(x), att_c.v_proj(x)
    assert g._cache.x is None and g._cache.ys is None   # entry dropped once every member took its output


def test_groups_need_shared_seed():
    """Per-tensor seeds (the old encodings): nothing is grouped, every projection stays a SucLinear."""
    ref, comp, report = reference_and_compressed(VQ, None, mx.float32)
    assert report["groups"] == []
    assert type(comp.model.layers[0].self_attn.q_proj) is SucLinear


def test_grouped_model_generates_like_reference():
    ref, comp, _ = grouped_reference_and_compressed(VQ, None, mx.float32)
    r = measure_generation(comp, prompt_len=8, decode_tokens=4, batch_size=2, warmup=0, repeats=1, seed=1)
    q = measure_generation(ref, prompt_len=8, decode_tokens=4, batch_size=2, warmup=0, repeats=1, seed=1)
    assert r["repeats"][0]["tokens"] == q["repeats"][0]["tokens"]


@pytest.mark.model
def test_qwen3_0p6b_grouped_fused_matches_decoded():
    """Shared seed_in + fused q|k|v and gate|up launches on the real model: float32 compressed == decoded."""
    from mlx_lm import load
    rev = "c1899de289a04d12100db370d81485cdf75e47ca"
    ref, tok = load("Qwen/Qwen3-0.6B", revision=rev)
    comp, _ = load("Qwen/Qwen3-0.6B", revision=rev)
    codec = make_codec({"kind": "vq", "k": 256, "block_size": 4, "rotate": True, "kmeans_iters": 3,
                        "fit_blocks": 32768})
    enc = encode_grouped(comp, names(28, [0, 27]), codec)
    ref.set_dtype(mx.float32)
    for n, e in enc.items():
        set_weight(ref, n, codec.decode(e))
    rep = convert_to_compressed(comp, enc)
    assert len(rep["groups"]) == 4 and all(g["fused_parts"] == 1 for g in rep["groups"])
    comp.set_dtype(mx.float32)
    ids = tok.encode(TEXT)
    toks = mx.array([ids[i:i + 24] for i in range(0, 72, 24)])
    exact = logits_kl(ref(toks), comp(toks))
    assert exact["kl_mean"] < 1e-5 and exact["top1_agreement"] == 1.0, exact


def test_conversion_with_autotune_enabled_on_tiny_sizes(tmp_path, monkeypatch):
    """Regression: with SUC_AUTOTUNE=1, ensure_tuned measured the tiny model's rotation sizes (< 1024) and crashed
    in convert_to_compressed. Rotation and fusion tuning run for real (short chains); the size-independent
    codebook / MMA sections are stubbed (they stream 100+ MB of weights)."""
    import warnings
    import suc.kernels.autotune as at
    monkeypatch.setenv("SUC_AUTOTUNE", "1")
    monkeypatch.setenv("SUC_TUNING_CACHE", str(tmp_path / "tuning.json"))
    at.clear_memory_cache()
    monkeypatch.setattr(at, "autotune", lambda **kw: None)
    monkeypatch.setattr(at, "tune_mma_min_batch", lambda **kw: None)
    rot, fuse = at.measure_rotation, at.measure_fuse_elementwise
    monkeypatch.setattr(at, "measure_rotation", lambda n: rot(n, links=2, reps=1, warmup_s=0.0))
    monkeypatch.setattr(at, "measure_fuse_elementwise", lambda h, i: fuse(h, i, links=2, reps=1, warmup_s=0.0))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)       # no tuning section may fail
            ref, comp, _ = grouped_reference_and_compressed(VQ, None, mx.float32)
        entry = at._load()[at.device_key()]
        assert {"64", "128"} <= set(entry["rotation"]["choice"]) and "fuse_elementwise" in entry
        x = tokens()
        assert np.allclose(np.array(comp(x)), np.array(ref(x)), atol=1e-3)
    finally:
        at.clear_memory_cache()
