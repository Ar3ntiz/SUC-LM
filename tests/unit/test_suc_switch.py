"""MoE experts executed from SUC codes (SucSwitchLinear) and whole-model conversion of a tiny Qwen3-MoE.

Oracles: the Python reference decoder (codec.decode of each expert's Encoded) applied row by row in float32, and
the same model with every expert row of the stacked SwitchLinear weights replaced by its decode (reconstructed
model, suc.evaluation.reconstruct semantics)."""
import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear

import suc.kernels.suc_switch as sw
from suc.analysis.sensitivity import get_weight, set_weight
from suc.evaluation.baselines import logits_kl
from suc.evaluation.speed import measure_generation
from suc.kernels.suc_switch import SucSwitchLinear
from suc.quant.codecs import make_codec
from suc.quant.residual import ResidualCodec
from suc.runtime.compressed_model import convert_to_compressed, resident_weight_bytes
from tiny_models import tiny_qwen3_moe

VQ = {"kind": "vq", "k": 16, "block_size": 4, "rotate": True, "kmeans_iters": 5}
ADD2 = {"kind": "additive", "k": 16, "block_size": 4, "num_codebooks": 2, "rotate": True, "kmeans_iters": 5}
VQ_NOROT = {"kind": "vq", "k": 16, "block_size": 4, "rotate": False, "kmeans_iters": 5}
ATTN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
EXPERT_PROJ = ("gate_proj", "up_proj", "down_proj")


def codec_of(cfg, residual):
    """residual: None, (fraction, value_dtype) for CSR, or (fraction, value_dtype, "columns", column_share)."""
    base = make_codec(cfg)
    if residual is None:
        return base
    if len(residual) == 2:
        return ResidualCodec(base, residual[0], value_dtype=residual[1])
    return ResidualCodec(base, residual[0], value_dtype=residual[1], structure=residual[2], column_share=residual[3])


def encode_experts(n_exp, out, inn, codec, seed=0):
    rng = np.random.default_rng(seed)
    ws = [mx.array(rng.normal(size=(out, inn)).astype(np.float32)) / np.sqrt(inn) for _ in range(n_exp)]
    return ws, [codec.encode(w, seed=seed + 10 * e) for e, w in enumerate(ws)]


def errors(y, ref):
    d = (y.astype(mx.float32) - ref).reshape(-1)
    return {"max_abs": float(mx.max(mx.abs(d))), "mean_abs": float(mx.mean(mx.abs(d))),
            "rel": float(mx.linalg.norm(d) / mx.linalg.norm(ref.reshape(-1)))}


def row_reference(x2, eidx, decoded):
    """Python reference decoder, row by row in float32: y_n = decode(expert e_n) x_n."""
    e = np.array(eidx).tolist()
    return mx.stack([x2[n].astype(mx.float32) @ decoded[k].T for n, k in enumerate(e)])


# ------------------------------------------------------------------------------------------- module level

@pytest.mark.parametrize("cfg", [VQ, ADD2, VQ_NOROT])
@pytest.mark.parametrize("residual", [None, (0.03, "int8"), (0.03, "fp16")])
@pytest.mark.parametrize("path", ["gather", "tiled", "grouped"])
@pytest.mark.parametrize("n", [1, 8, 40])
def test_rows_match_reference_decoder(cfg, residual, path, n):
    codec = codec_of(cfg, residual)
    _, encs = encode_experts(6, 32, 64, codec, seed=1)
    m = SucSwitchLinear.from_encodings(encs)
    assert m.num_experts == 6 and m.input_dims == 64 and m.output_dims == 32
    assert (m.n_residual > 0) == (residual is not None and residual[0] > 0)
    decoded = [codec.decode(e).astype(mx.float32) for e in encs]
    x = mx.random.normal((n, 64), key=mx.random.key(n))
    eidx = mx.array(np.random.default_rng(n).integers(0, 6, size=n).astype(np.uint32))
    y = m.rows(x, eidx, path=path)
    err = errors(y, row_reference(x, eidx, decoded))
    assert y.shape == (n, 32) and err["rel"] < 1e-5, err


COLS = [(0.03, "int8", "columns", 1.0), (0.03, "fp16", "columns", 1.0), (0.03, "int8", "columns", 0.5),
        (0.03, "fp16", "columns", 0.5)]


@pytest.mark.parametrize("cfg", [VQ, ADD2, VQ_NOROT])
@pytest.mark.parametrize("residual", COLS)
@pytest.mark.parametrize("path", ["gather", "tiled", "grouped"])
@pytest.mark.parametrize("n", [1, 8, 40])
def test_column_residual_rows_match_reference_decoder(cfg, residual, path, n):
    """Column-structured residuals (int8 + per-column scales / fp16; column_share < 1 adds a CSR tail): one launch
    for all selected experts, exact vs the reference decoder in float32."""
    codec = codec_of(cfg, residual)
    _, encs = encode_experts(6, 32, 64, codec, seed=1)
    m = SucSwitchLinear.from_encodings(encs)
    assert m.n_columns == sum(int(e.params["columns"]) for e in encs) > 0
    assert (m.n_residual > 0) == (residual[3] < 1.0)
    decoded = [codec.decode(e).astype(mx.float32) for e in encs]
    x = mx.random.normal((n, 64), key=mx.random.key(n))
    eidx = mx.array(np.random.default_rng(n).integers(0, 6, size=n).astype(np.uint32))
    y = m.rows(x, eidx, path=path)
    err = errors(y, row_reference(x, eidx, decoded))
    assert y.shape == (n, 32) and err["rel"] < 2e-6, err


@pytest.mark.parametrize("value_dtype", ["int8", "fp16"])
@pytest.mark.parametrize("out,inn", [(32, 64), (30, 64)])
def test_column_residual_ragged_experts(value_dtype, out, inn):
    """Experts with different column counts (different fractions), with a CSR tail or without any residual, and an
    output size that is not a multiple of 4 (scalar row loads)."""
    base = make_codec(VQ)
    codecs = [ResidualCodec(base, 0.03, value_dtype, structure="columns"),
              ResidualCodec(base, 0.08, value_dtype, structure="columns", column_share=0.5), base,
              ResidualCodec(base, 0.015, value_dtype, structure="columns"), base]
    rng = np.random.default_rng(4)
    ws = [mx.array(rng.normal(size=(out, inn)).astype(np.float32)) / np.sqrt(inn) for _ in codecs]
    encs = [c.encode(w, seed=3 + e) for e, (c, w) in enumerate(zip(codecs, ws))]
    counts = [int(e.params.get("columns", 0)) for e in encs]
    assert len(set(counts)) > 2 and min(counts) == 0
    m = SucSwitchLinear.from_encodings(encs)
    assert m.n_columns == sum(counts) and m.n_residual == int(encs[1].params["n"]) > 0
    decoded = [c.decode(e).astype(mx.float32) for c, e in zip(codecs, encs)]
    x = mx.random.normal((40, inn), key=mx.random.key(5))
    eidx = mx.array(np.arange(40, dtype=np.uint32) % len(codecs))
    for path in ("gather", "tiled", "grouped"):
        err = errors(m.rows(x, eidx, path=path), row_reference(x, eidx, decoded))
        assert err["rel"] < 2e-6, (path, err)


def test_column_residual_bias_dtypes_and_compile():
    codec = codec_of(VQ, (0.03, "int8", "columns", 0.5))
    _, encs = encode_experts(4, 32, 64, codec, seed=6)
    decoded = [codec.decode(e).astype(mx.float32) for e in encs]
    bias = mx.random.normal((4, 32), key=mx.random.key(7))
    e = mx.array([3, 0, 0, 2, 1], dtype=mx.uint32)
    x = mx.random.normal((5, 64), key=mx.random.key(8))
    m = SucSwitchLinear.from_encodings(encs, bias=bias)
    assert errors(m.rows(x, e), row_reference(x, e, decoded) + bias[e])["rel"] < 2e-6
    m = SucSwitchLinear.from_encodings(encs)
    import suc.kernels.suc_linear as sl
    old = sl.COMPILE
    try:
        outs = []
        for c in (True, False):
            sl.COMPILE = c
            outs.append(m.rows(x.astype(mx.bfloat16), e, path="gather"))
    finally:
        sl.COMPILE = old
    assert outs[0].dtype == mx.bfloat16
    assert np.array_equal(np.array(outs[0].astype(mx.float32)), np.array(outs[1].astype(mx.float32)))
    assert errors(outs[0], row_reference(x.astype(mx.bfloat16), e, decoded))["rel"] < 2e-2


def test_column_residual_rejects_mixed_value_dtypes():
    base = make_codec(VQ)
    w = mx.random.normal((32, 64), key=mx.random.key(0))
    a = ResidualCodec(base, 0.03, "int8", structure="columns").encode(w, seed=0)
    b = ResidualCodec(base, 0.03, "fp16", structure="columns").encode(w, seed=0)
    with pytest.raises(ValueError, match="dtype"):
        SucSwitchLinear.from_encodings([a, b])


@pytest.mark.parametrize("dtype,tol", [(mx.float32, 1e-5), (mx.float16, 3e-3), (mx.bfloat16, 2e-2)])
def test_activation_dtypes(dtype, tol):
    codec = codec_of(ADD2, (0.02, "int8"))
    _, encs = encode_experts(4, 64, 32, codec, seed=2)
    m = SucSwitchLinear.from_encodings(encs)
    decoded = [codec.decode(e).astype(mx.float32) for e in encs]
    x = mx.random.normal((24, 32), key=mx.random.key(3)).astype(dtype)
    eidx = mx.array(np.random.default_rng(3).integers(0, 4, size=24).astype(np.uint32))
    for path in ("gather", "tiled"):
        y = m.rows(x, eidx, path=path)
        assert y.dtype == dtype
        err = errors(y, row_reference(x, eidx, decoded))
        assert err["rel"] < tol, (path, err)


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("residual", [None, (0.03, "int8")])
def test_call_matches_switch_linear_and_glu(sort, residual):
    """Same call signature / output shape as mlx-lm SwitchLinear (decode and sorted-prefill layouts) and inside a
    SwitchGLU (x expanded to (..., 1, 1, in), indices (..., k))."""
    codec = codec_of(VQ, residual)
    E, H, I = 8, 64, 32
    ws_g, enc_g = encode_experts(E, I, H, codec, seed=4)
    ws_u, enc_u = encode_experts(E, I, H, codec, seed=5)
    ws_d, enc_d = encode_experts(E, H, I, codec, seed=6)
    glu = SwitchGLU(H, I, E)
    for proj, encs in (("gate_proj", enc_g), ("up_proj", enc_u), ("down_proj", enc_d)):
        glu[proj].weight = mx.stack([codec.decode(e) for e in encs])
    lin = glu.gate_proj
    sm = SucSwitchLinear.from_encodings(enc_g)
    rng = np.random.default_rng(7)
    ntok = 40 if sort else 3                      # SwitchGLU sorts when indices.size >= 64
    x = mx.random.normal((1, ntok, H), key=mx.random.key(8))
    inds = mx.array(np.stack([rng.choice(E, 2, replace=False) for _ in range(ntok)])[None].astype(np.uint32))
    xe = mx.expand_dims(x, (-2, -3))
    ref, y = lin(xe, inds), sm(xe, inds)
    assert y.shape == ref.shape
    assert errors(y, ref)["rel"] < 1e-5
    ref_glu = glu(x, inds)
    glu.gate_proj, glu.up_proj, glu.down_proj = sm, SucSwitchLinear.from_encodings(enc_u), \
        SucSwitchLinear.from_encodings(enc_d)
    y_glu = glu(x, inds)
    assert y_glu.shape == ref_glu.shape and errors(y_glu, ref_glu)["rel"] < 1e-5


def test_bias_and_compile_paths_agree():
    codec = codec_of(VQ, None)
    _, encs = encode_experts(4, 32, 64, codec, seed=9)
    bias = mx.random.normal((4, 32), key=mx.random.key(1))
    m = SucSwitchLinear.from_encodings(encs, bias=bias)
    x = mx.random.normal((5, 64), key=mx.random.key(2))
    e = mx.array([3, 0, 0, 2, 1], dtype=mx.uint32)
    ref = row_reference(x, e, [codec.decode(c) for c in encs]) + bias[e]
    import suc.kernels.suc_linear as sl
    old = sl.COMPILE
    try:
        outs = []
        for c in (True, False):
            sl.COMPILE = c
            outs.append(m.rows(x, e, path="gather"))
    finally:
        sl.COMPILE = old
    for y in outs:
        assert errors(y, ref)["rel"] < 1e-5
    assert np.array_equal(np.array(outs[0]), np.array(outs[1]))


def test_rejects_mixed_layouts():
    a = make_codec(VQ).encode(mx.random.normal((32, 64)), seed=0)
    b = make_codec(ADD2).encode(mx.random.normal((32, 64)), seed=0)
    with pytest.raises(ValueError, match="layout"):
        SucSwitchLinear.from_encodings([a, b])
    c = make_codec(VQ).encode(mx.random.normal((16, 64)), seed=0)
    with pytest.raises(ValueError, match="shape"):
        SucSwitchLinear.from_encodings([a, c])


def test_auto_path_switches_at_threshold(monkeypatch):
    _, encs = encode_experts(3, 32, 64, make_codec(VQ), seed=3)
    m = SucSwitchLinear.from_encodings(encs)
    seen = []
    orig = SucSwitchLinear.rows

    def spy(self, x2, eidx, path="auto"):
        seen.append(self._choose(x2.shape[0], path))
        return orig(self, x2, eidx, path)
    monkeypatch.setattr(SucSwitchLinear, "rows", spy)
    for n in (1, sw.GATHER_MAX_ROWS, sw.GATHER_MAX_ROWS + 1):
        m(mx.random.normal((n, 1, 64)), mx.zeros((n,), mx.uint32))
    assert seen == ["gather", "gather", "tiled"]


# ------------------------------------------------------------------------------------------- whole model

def expert_names(model, layers=(0, 1)):
    n_exp = model.args.num_experts
    return [f"model.layers.{i}.mlp.experts.{e}.{p}.weight" for i in layers for e in range(n_exp) for p in EXPERT_PROJ
            if f"model.layers.{i}.mlp.switch_mlp.{p}.weight" in dict(tree_flatten(model.parameters()))]


def attn_names(layers=(0, 1)):
    return [f"model.layers.{i}.{p}.weight" for i in layers for p in ATTN]


def reference_and_compressed(cfg, residual, dtype, mlp_only=(), attn_residual="same"):
    """attn_residual: residual of the attention projections ("same" = `residual`)."""
    ref, comp = tiny_qwen3_moe(seed=0, mlp_only_layers=mlp_only), tiny_qwen3_moe(seed=0, mlp_only_layers=mlp_only)
    if dtype != mx.float32:
        ref.set_dtype(dtype)
        comp.set_dtype(dtype)
    codec = codec_of(cfg, residual)
    acodec = codec if attn_residual == "same" else codec_of(cfg, attn_residual)
    names = attn_names() + expert_names(comp)
    if mlp_only:
        names += [f"model.layers.{i}.mlp.{p}.weight" for i in mlp_only for p in EXPERT_PROJ]
    codecs = {n: acodec if "self_attn" in n else codec for n in names}
    enc = {n: codecs[n].encode(get_weight(comp, n).astype(mx.float32), seed=i) for i, n in enumerate(names)}
    for n, e in enc.items():
        set_weight(ref, n, codecs[n].decode(e))
    mx.eval(ref.parameters())
    report = convert_to_compressed(comp, enc)
    return ref, comp, report, enc


def tokens(b=3, t=12, seed=0):
    return mx.array(np.random.default_rng(seed).integers(0, 128, size=(b, t)).astype(np.int32))


@pytest.mark.parametrize("cfg,residual", [(VQ, None), (ADD2, (0.02, "int8")), (VQ, (0.02, "fp16")),
                                          (VQ, (0.02, "int8", "columns", 1.0)), (ADD2, (0.02, "fp16", "columns", 0.5))])
@pytest.mark.parametrize("dtype,tol,kl_tol", [(mx.float32, 1e-5, 1e-6), (mx.bfloat16, 5e-2, 1e-3)])
@pytest.mark.parametrize("t", [1, 12, 40])
def test_compressed_moe_logits_match_reconstructed(cfg, residual, dtype, tol, kl_tol, t):
    """t = 1 / 12: decode-sized gather path; t = 40 (3 x 40 x 2 = 240 pairs): sorted SwitchGLU + tiled path.
    float32: logits rel. error ~4e-7 (rounding only); the float32 KL estimate itself is noisy at ~1e-7 (it can come
    out negative), hence |KL| < 1e-6. Column residuals in bf16: the attention projections keep a CSR residual of the
    same fraction, so the bf16 tolerance measures the expert stacks (the bf16 noise of the dense column path on the
    attention projections of this tiny model alone reaches rel ~0.05; float32 checks every projection)."""
    cols = residual is not None and len(residual) > 2
    attn = residual[:2] if cols and dtype != mx.float32 else "same"
    ref, comp, report, _ = reference_and_compressed(cfg, residual, dtype, attn_residual=attn)
    assert sum(r["kind"] == "SucSwitchLinear" for r in report["converted"]) == 6
    for i in range(2):
        for p in EXPERT_PROJ:
            assert type(comp.model.layers[i].mlp.switch_mlp[p]) is SucSwitchLinear
    toks = tokens(t=t)
    lr, lc = ref(toks).astype(mx.float32), comp(toks).astype(mx.float32)
    rel = float(mx.linalg.norm(lc - lr) / mx.linalg.norm(lr))
    kl = logits_kl(lr, lc)
    assert rel < tol, rel
    assert abs(kl["kl_mean"]) < kl_tol and kl["top1_agreement"] > 0.95, kl


def test_moe_with_dense_layer_and_router_untouched():
    ref, comp, report, enc = reference_and_compressed(VQ, None, mx.float32, mlp_only=(1,))
    assert type(comp.model.layers[0].mlp.gate).__name__ == "Linear"           # router stays as is
    assert "weight" in comp.model.layers[0].mlp.gate
    kinds = {r["name"]: r["kind"] for r in report["converted"]}
    assert kinds["model.layers.0.mlp.switch_mlp.up_proj.weight"] == "SucSwitchLinear"
    assert kinds["model.layers.1.mlp.up_proj.weight"] in ("SucLinear", "SucFusedLinear")
    toks = tokens()
    lr, lc = ref(toks), comp(toks)
    assert float(mx.linalg.norm(lc - lr) / mx.linalg.norm(lr)) < 1e-5
    assert abs(logits_kl(lr, lc)["kl_mean"]) < 1e-6


def test_report_and_resident_bytes():
    ref, comp, report, enc = reference_and_compressed(ADD2, (0.02, "int8"), mx.bfloat16)
    rows = [r for r in report["converted"] if r["kind"] == "SucSwitchLinear"]
    stack_bytes = sum(ref.model.layers[i].mlp.switch_mlp[p].weight.nbytes for i in range(2) for p in EXPERT_PROJ)
    assert sum(r["bytes_before"] for r in rows) == stack_bytes
    for r in rows:
        assert r["experts"] == 8 and len(r["shape"]) == 3 and 0 < r["bytes_after"] < r["bytes_before"]
        assert r["n_residual"] > 0
    assert report["tensors_converted"] == len(enc)
    assert resident_weight_bytes(comp) == report["model_bytes_after"] < report["model_bytes_before"]
    for i in range(2):
        for p in EXPERT_PROJ:
            assert "weight" not in comp.model.layers[i].mlp.switch_mlp[p]


def test_rejects_incomplete_expert_set():
    comp = tiny_qwen3_moe(seed=0)
    codec = make_codec(VQ)
    names = [f"model.layers.0.mlp.experts.{e}.up_proj.weight" for e in range(7)]
    enc = {n: codec.encode(get_weight(comp, n), seed=0) for n in names}
    with pytest.raises(ValueError, match="experts"):
        convert_to_compressed(comp, enc)


def test_compressed_moe_generates_like_reconstructed():
    ref, comp, _, _ = reference_and_compressed(ADD2, (0.02, "int8"), mx.float32)
    r = measure_generation(comp, prompt_len=8, decode_tokens=4, batch_size=2, warmup=0, repeats=1, seed=1)
    q = measure_generation(ref, prompt_len=8, decode_tokens=4, batch_size=2, warmup=0, repeats=1, seed=1)
    assert r["decode_tok_s"] > 0
    assert r["repeats"][0]["tokens"] == q["repeats"][0]["tokens"]


@pytest.mark.parametrize("rotation", ["fused", "mlx"])
@pytest.mark.parametrize("shape", [(64, 32), (48, 96)])     # 48 / 96 = 12·2^k: MLX fallback sizes for fused
def test_rotation_modes_and_sizes(rotation, shape, monkeypatch):
    import suc.kernels.suc_linear as sl
    monkeypatch.setattr(sl, "ROTATION", rotation)
    out, inn = shape
    codec = codec_of(VQ, (0.02, "int8"))
    _, encs = encode_experts(5, out, inn, codec, seed=11)
    m = SucSwitchLinear.from_encodings(encs)
    decoded = [codec.decode(e).astype(mx.float32) for e in encs]
    x = mx.random.normal((40, inn), key=mx.random.key(4))
    eidx = mx.array(np.random.default_rng(4).integers(0, 5, size=40).astype(np.uint32))
    for path in ("gather", "tiled"):
        err = errors(m.rows(x, eidx, path=path), row_reference(x, eidx, decoded))
        assert err["rel"] < 1e-5, (path, err)


def test_tile_plan_round_trip():
    e = np.array([3, 0, 3, 3, 1, 0, 3, 3, 3], np.uint32)
    src, dst, tile_e, tile = sw.tile_plan(e, 5)
    assert tile == 4 and list(tile_e) == [0, 1, 3, 3]
    assert np.array_equal(src[dst], np.arange(e.size))                 # every row lands where it is read back
    assert all(e[src[dst[n]]] == tile_e[dst[n] // tile] for n in range(e.size))
