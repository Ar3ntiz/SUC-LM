"""Column-structured residuals (ResidualCodec structure="columns"): dense int8 residual on c input columns
(+ optional sparse tail), equal bit budget as the CSR residual of the same fraction."""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from suc.quant.codecs import VectorCodebook
from suc.quant.residual import ResidualCodec, residual_from_config

BASE = VectorCodebook(16, block_size=8, rotate=False, kmeans_iters=5)


def _w(shape=(64, 256), seed=1) -> mx.array:
    return mx.array((0.02 * np.random.default_rng(seed).standard_normal(shape)).astype(np.float32))


def _np(x: mx.array) -> np.ndarray:
    return np.array(x.astype(mx.float32), dtype=np.float64)


def _imp(shape, hot=(3, 77, 200), seed=0) -> mx.array:
    ex2 = np.full(shape[1], 1e-2)
    ex2[list(hot)] = 100.0
    return mx.broadcast_to(mx.array(ex2.astype(np.float32))[None], shape)


W = _w()
IMP = _imp(W.shape)


def test_default_structure_is_csr_and_params_unchanged() -> None:
    a = ResidualCodec(BASE, 0.02, "int8", "importance")
    assert a.structure == "csr" and a.name == "vq+res0.02_int8_importance"
    e = a.encode(W, seed=3, importance=IMP)
    assert "structure" not in e.params and "res_cols" not in e.arrays


def test_columns_layout_and_budget() -> None:
    csr = ResidualCodec(BASE, 0.02, "int8", "importance").encode(W, seed=3, importance=IMP)
    c = ResidualCodec(BASE, 0.02, "int8", "importance", structure="columns")
    assert c.name == "vq+res0.02_int8_importance_cols"
    e = c.encode(W, seed=3, importance=IMP)
    p, a = e.params, e.arrays
    out, inn = W.shape
    col_cost = out * 8 + math.ceil(math.log2(inn)) + 16
    assert p["columns"] == csr.params["residual_bits"] // col_cost and p["columns"] >= 3
    assert p["n"] == 0 and a["res_pos"].size == 0
    assert a["res_cols"].dtype == mx.uint32 and a["res_cols"].shape == (p["columns"],)
    assert a["res_colval"].dtype == mx.int8 and a["res_colval"].shape == (p["columns"], out)
    assert a["res_colscale"].dtype == mx.float16 and a["res_colscale"].shape == (p["columns"],)
    cols = np.array(a["res_cols"]).astype(np.int64)
    assert np.all(np.diff(cols) > 0) and {3, 77, 200} <= set(cols.tolist())
    assert e.bits()["residual"] == p["residual_bits"] == p["columns"] * col_cost
    assert e.bits()["residual"] <= csr.bits()["residual"]


def test_columns_decode_int8_step_bound() -> None:
    c = ResidualCodec(BASE, 0.02, "int8", "importance", structure="columns")
    e = c.encode(W, seed=3, importance=IMP)
    wb = _np(BASE.decode(e.base()))
    r = _np(W) - wb
    wh = _np(c.decode(e))
    cols = np.array(e.arrays["res_cols"]).astype(np.int64)
    s = _np(e.arrays["res_colscale"])
    err = np.abs(_np(W) - wh)
    assert np.all(err[:, cols] <= 0.5 * s[None] * (1 + 1e-3) + 1e-7)
    other = np.setdiff1d(np.arange(W.shape[1]), cols)
    np.testing.assert_allclose(err[:, other], np.abs(r[:, other]), atol=1e-7)


def test_columns_beat_csr_on_column_concentrated_importance() -> None:
    w = _w((256, 256), 5)
    imp = _imp(w.shape)        # 3 hot columns x 256 rows = 768 entries > the 328 CSR entries of 0.5%

    def werr(structure: str) -> float:
        c = ResidualCodec(BASE, 0.005, "int8", "importance", structure=structure)
        wh = _np(c.decode(c.encode(w, seed=3, importance=imp)))
        return float((_np(imp) * (_np(w) - wh) ** 2).sum())
    assert werr("columns") < werr("csr")


def test_hybrid_has_tail_and_stays_in_budget() -> None:
    csr = ResidualCodec(BASE, 0.02, "int8", "importance").encode(W, seed=3, importance=IMP)
    c = ResidualCodec(BASE, 0.02, "int8", "importance", structure="columns", column_share=0.5)
    assert c.name == "vq+res0.02_int8_importance_cols0.5"
    e = c.encode(W, seed=3, importance=IMP)
    assert e.params["n"] > 0 and e.params["columns"] > 0
    assert e.bits()["residual"] <= csr.bits()["residual"]
    # tail positions are sorted / unique and decode adds both parts
    pos = np.array(e.arrays["res_pos"]).astype(np.int64)
    assert np.all(np.diff(pos) > 0)
    werr = lambda wh: float((_np(IMP) * (_np(W) - _np(wh)) ** 2).sum())  # noqa: E731
    assert werr(c.decode(e)) < werr(BASE.decode(e.base()))


def test_columns_fp16_and_magnitude() -> None:
    c = ResidualCodec(BASE, 0.05, "fp16", "magnitude", structure="columns")
    e = c.encode(W, seed=3)
    assert "res_colscale" not in e.arrays and e.arrays["res_colval"].dtype == mx.float16
    out, inn = W.shape
    assert e.params["residual_bits"] == e.params["columns"] * (out * 16 + math.ceil(math.log2(inn)))


def test_columns_from_config_and_validation() -> None:
    c = residual_from_config({"base": {"kind": "bf16"}, "fraction": 0.01, "value_dtype": "int8",
                              "structure": "columns", "column_share": 0.75})
    assert c.structure == "columns" and c.column_share == 0.75
    for bad in ({"structure": "rows"}, {"structure": "columns", "column_share": 0.0},
                {"structure": "columns", "column_share": 1.5}, {"structure": "csr", "column_share": 0.5}):
        with pytest.raises(ValueError):
            residual_from_config({"base": {"kind": "bf16"}, "fraction": 0.01, **bad})


def test_columns_zero_fraction() -> None:
    c = ResidualCodec(BASE, 0.0, "int8", "importance", structure="columns")
    e = c.encode(W, seed=3, importance=IMP)
    assert e.params["columns"] == 0 and e.bits()["residual"] == 0
    np.testing.assert_array_equal(_np(c.decode(e)), _np(BASE.decode(e.base())))


def test_codec_config_residual_structure() -> None:
    from suc.config import quantize_config_from_dict

    def cfg(**kw):
        return quantize_config_from_dict({"name": "q", "model": {"id": "Qwen/Qwen3-0.6B"},
                                          "codecs": [{"kind": "vq", "residual_fraction": 0.005, **kw}]}).codecs[0]
    assert cfg().residual_structure == "csr" and cfg().name == "vq_k256_g8_rot+res0.005_activation"
    c = cfg(residual_structure="columns")
    assert c.name == "vq_k256_g8_rot+res0.005_activation_cols" and "residual_structure" not in c.codec_dict()
    assert cfg(residual_structure="columns", residual_column_share=0.75).name.endswith("_cols0.75")
    for bad in ({"residual_structure": "rows"}, {"residual_column_share": 0.5},
                {"residual_structure": "columns", "residual_column_share": 0.0}):
        with pytest.raises(ValueError):
            cfg(**bad)


# ------------------------------------------------------------------------------------- runtime (Metal)
import suc.runtime.compressed_model as cm  # noqa: E402
from suc.evaluation.baselines import logits_kl  # noqa: E402
from suc.analysis.sensitivity import set_weight  # noqa: E402
from suc.evaluation.reconstruct import rotation_seeds  # noqa: E402
from suc.quant.codecs import make_codec  # noqa: E402
from tiny_models import tiny_qwen3  # noqa: E402

VQ = {"kind": "vq", "k": 16, "block_size": 4, "rotate": True, "kmeans_iters": 5}
PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _col_codec(share=1.0, dt="int8", frac=0.03):
    return ResidualCodec(make_codec(VQ), frac, dt, "magnitude", structure="columns", column_share=share)


@pytest.mark.parametrize("share", [1.0, 0.5])
@pytest.mark.parametrize("dt", ["int8", "fp16"])
@pytest.mark.parametrize("b", [1, 4, 16])
def test_column_module_matches_decoded_weight(share, dt, b) -> None:
    w = mx.random.normal((96, 128), key=mx.random.key(0)) / np.sqrt(128)
    codec = _col_codec(share, dt)
    e = codec.encode(w, seed=3)
    m = cm.module_from_encoded(e)
    assert isinstance(m, cm.SucResidualLinear) and m.n_columns == e.params["columns"] > 0
    assert (m.n_residual > 0) == (share < 1)
    x = mx.random.normal((b, 128), key=mx.random.key(2))
    ref = x @ codec.decode(e).T
    y = m(x)
    rel = float(mx.linalg.norm(y - ref) / mx.linalg.norm(ref))
    assert y.shape == (b, 96) and rel < 1e-5, rel


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_structured_kernel_is_exact_matmul(dtype) -> None:
    w = mx.random.normal((80, 64), key=mx.random.key(0))
    codec = _col_codec(0.5, "int8", 0.1)
    e = codec.encode(w, seed=0)
    m = cm.module_from_encoded(e)
    r = codec.decode(e) - codec.base.decode(e.base())                 # R = columns + tail, float32
    x = mx.random.normal((5, 64), key=mx.random.key(1)).astype(dtype)
    y0 = mx.random.normal((5, 80), key=mx.random.key(2))
    ref = np.array(y0) + np.array(x.astype(mx.float32)) @ np.array(r).T
    got = m.residual(x, y0)
    assert got.dtype == mx.float32 and np.allclose(np.array(got), ref, atol=1e-5)


def _struct_case(sizes, c, n_csr, inn, vdt, rng):
    """Random row-concatenated residual (members `sizes`, c columns and n_csr CSR entries per member) and its
    dense float64 reference R (Σ sizes, inn)."""
    S, out = len(sizes), sum(sizes)
    R = np.zeros((out, inn))
    cols, vals, cscale, counts, rcols, rvals = [], [], [], [], [], []
    rscale = rng.uniform(0.5, 2, S).astype(np.float32)
    r0 = 0
    for m, o in enumerate(sizes):
        cm_ = np.sort(rng.choice(inn, c, replace=False))
        v = rng.integers(-127, 128, (c, o)).astype(np.int8 if vdt == "int8" else np.float16)
        sc = rng.uniform(0.01, 0.1, c).astype(np.float32)
        R[r0:r0 + o, cm_] += (v.astype(np.float64) * sc[:, None]).T
        cols.append(cm_), vals.append(v.reshape(-1)), cscale.append(sc)
        pos = np.sort(rng.choice(o * inn, n_csr, replace=False))
        rv = rng.integers(-127, 128, n_csr).astype(np.int8 if vdt == "int8" else np.float16)
        np.add.at(R, (r0 + pos // inn, pos % inn), rv.astype(np.float64) * rscale[m])
        counts.append(np.bincount(pos // inn, minlength=o)), rcols.append(pos % inn), rvals.append(rv)
        r0 += o
    row_start = np.concatenate([[0], np.cumsum(sizes)])
    col_ptr = np.arange(S + 1) * c
    col_off = np.concatenate([[0], np.cumsum([c * o for o in sizes])[:-1]])
    meta = mx.array(np.concatenate([row_start, col_ptr, col_off]).astype(np.uint32))
    arrs = dict(meta=meta, cols=mx.array(np.concatenate(cols).astype(np.uint32)),
                cval=mx.array(np.concatenate(vals)), cscale=mx.array(np.concatenate(cscale)),
                row_ptr=mx.array(np.concatenate([[0], np.cumsum(np.concatenate(counts))]).astype(np.uint32)),
                rcols=mx.array(np.concatenate(rcols).astype(np.uint32)), rvals=mx.array(np.concatenate(rvals)),
                rscale=mx.array(rscale))
    return arrs, R


@pytest.mark.parametrize("sizes", [(96,), (37,), (64, 16, 16), (24, 13, 40)])
@pytest.mark.parametrize("b", [1, 3, 9, 16])
@pytest.mark.parametrize("c,n_csr", [(0, 50), (5, 0), (70, 0), (300, 40)])
@pytest.mark.parametrize("vdt", ["int8", "fp16"])
def test_structured_kernel_layouts(sizes, b, c, n_csr, vdt) -> None:
    """Row quads / single rows, column split over simdgroups (JS) + reduction, batch tiles, CSR tail."""
    rng = np.random.default_rng(len(sizes) * 1000 + b * 10 + c)
    inn = 384
    a, R = _struct_case(sizes, c, n_csr, inn, vdt, rng)
    x = mx.array(rng.standard_normal((b, inn)).astype(np.float32))
    y0 = mx.array(rng.standard_normal((b, sum(sizes))).astype(np.float32))
    quads = all(o % 4 == 0 for o in sizes) if len(sizes) > 1 else None
    got = cm.structured_residual_metal(x, y0, a["meta"], a["cols"] if c else None, a["cval"] if c else None,
                                       a["cscale"] if c else None, a["row_ptr"] if n_csr else None,
                                       a["rcols"] if n_csr else None, a["rvals"] if n_csr else None, a["rscale"],
                                       len(sizes), row_quads=quads)
    ref = np.array(y0, np.float64) + np.array(x, np.float64) @ R.T
    np.testing.assert_allclose(np.array(got), ref, rtol=1e-5, atol=1e-4 * (1 + np.abs(ref).max()))


def _names(layers=2):
    return [f"model.layers.{i}.{p}.weight" for i in range(layers) for p in PROJ]


def _get(model, name):
    cur = model
    for p in name.split("."):
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    return cur


def _grouped(codec_for, dtype=mx.float32, fuse=True):
    ref, comp = tiny_qwen3(seed=0), tiny_qwen3(seed=0)
    if dtype != mx.float32:
        ref.set_dtype(dtype)
        comp.set_dtype(dtype)
    enc = {}
    for i, n in enumerate(_names()):
        enc[n] = codec_for(n).encode(_get(comp, n), seed=i, **rotation_seeds(n, 0))
        set_weight(ref, n, codec_for(n).decode(enc[n]))
    mx.eval(ref.parameters())
    rep = cm.convert_to_compressed(comp, enc, fuse=fuse)
    return ref, comp, rep


def _toks():
    return mx.array(np.random.default_rng(0).integers(0, 128, size=(3, 12)).astype(np.int32))


@pytest.mark.parametrize("kind", ["columns", "hybrid", "csr", "mixed"])
@pytest.mark.parametrize("dtype,tol,kl_tol", [(mx.float32, 1e-4, 1e-7), (mx.bfloat16, 3e-2, 3e-4)])
def test_grouped_residual_members_are_fused_and_exact(kind, dtype, tol, kl_tol) -> None:
    codecs = {"columns": _col_codec(1.0), "hybrid": _col_codec(0.5),
              "csr": ResidualCodec(make_codec(VQ), 0.03, "int8")}
    if kind == "mixed":   # residual and plain members in one group
        pick = lambda n: make_codec(VQ) if ("k_proj" in n or "up_proj" in n) else codecs["columns"]  # noqa: E731
    else:
        pick = lambda n: codecs[kind]  # noqa: E731
    ref, comp, rep = _grouped(pick, dtype)
    assert len(rep["groups"]) == 4 and all(g["fused_parts"] == 1 and g["parts"] == 1 for g in rep["groups"])
    assert isinstance(comp.model.layers[0].self_attn.suc_qkv.parts[0], cm.SucFusedResidual)
    lr, lc = ref(_toks()).astype(mx.float32), comp(_toks()).astype(mx.float32)
    rel = float(mx.linalg.norm(lc - lr) / mx.linalg.norm(lr))
    kl = logits_kl(lr, lc)
    assert rel < tol, rel
    assert kl["kl_mean"] < kl_tol and kl["top1_agreement"] > 0.95, kl


def test_fuse_residual_switch_restores_per_member_path(monkeypatch) -> None:
    monkeypatch.setattr(cm, "FUSE_RESIDUAL", False)
    ref, comp, rep = _grouped(lambda n: ResidualCodec(make_codec(VQ), 0.03, "int8"))
    assert all(g["fused_parts"] == 0 and g["parts"] in (2, 3) for g in rep["groups"])
    lr, lc = ref(_toks()), comp(_toks())
    assert float(mx.linalg.norm(lc - lr) / mx.linalg.norm(lr)) < 1e-4


def test_fused_residual_memory_equals_members() -> None:
    _, fused, _ = _grouped(lambda n: _col_codec(0.5))
    _, flat, _ = _grouped(lambda n: _col_codec(0.5), fuse=False)
    a, b = cm.resident_weight_bytes(fused), cm.resident_weight_bytes(flat)
    # fused drops the duplicated input signs (2 in qkv, 1 in gate_up per layer, in = 64, int8 since R5); the
    # concatenated residual adds per group: meta (3S+2 u32) and S - 1 extra CSR scale floats ... at most 128 B
    assert abs((b - a) - 2 * (2 * 64 + 64) * 1) <= 4 * 128, (a, b)


# ------------------------------------------------------------------------------------- SUC1 container
@pytest.mark.parametrize("share,dt", [(1.0, "int8"), (0.5, "int8"), (0.5, "fp16")])
def test_suc1_round_trip_and_reference_decoder(tmp_path, share, dt) -> None:
    from suc.io.suc1 import SucModel, write_suc1

    codec = ResidualCodec(make_codec(dict(VQ, block_size=8)), 0.03, dt, "importance", structure="columns",
                          column_share=share)
    w = _w((64, 128), 9)
    e = codec.encode(w, seed=1, importance=_imp(w.shape, hot=(5, 60, 100)))
    p = tmp_path / "c.suc"
    write_suc1(p, architecture={"model_type": "qwen3"}, tokenizer_id="t", tensors={"w": e})
    m = SucModel.load(p)
    d = m.descriptor("w")
    assert d["residual_type"] == f"columns_{dt}"
    cols = next(r for r in d["arrays"] if r["key"] == "res_cols")
    assert cols["packed"]["bits"] == e.params["column_index_bits"] == 7
    back = m.encoded("w")
    assert back.params == e.params and sorted(back.arrays) == sorted(e.arrays)
    for k in e.arrays:
        assert np.array_equal(np.array(back.arrays[k]), np.array(e.arrays[k])), k
    np.testing.assert_array_equal(_np(m.decode_tensor("w")), _np(codec.decode(e)))   # bit-exact
    assert m.bits()["residual"] == 8 * sum(r["nbytes"] for r in d["arrays"] if r["part"] == "residual")
