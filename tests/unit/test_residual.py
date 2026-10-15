"""Tests for the sparse residual / outlier codec (spec §13, README §2)."""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from suc.quant.codecs import AffineRTN, Encoded, VectorCodebook
from suc.quant.residual import ResidualCodec, residual_from_config


def _randn(shape: tuple[int, int], seed: int = 0, scale: float = 0.02) -> mx.array:
    return mx.array((scale * np.random.default_rng(seed).standard_normal(shape)).astype(np.float32))


def _np(x: mx.array) -> np.ndarray:
    return np.array(x.astype(mx.float32), dtype=np.float64)


BASE = VectorCodebook(16, block_size=8, rotate=False, kmeans_iters=5)
W = _randn((64, 128), 1)


def _base_resid(base=BASE, w=W) -> tuple[np.ndarray, np.ndarray]:
    wb = _np(base.decode(base.encode(w, seed=3)))
    return wb, _np(w) - wb


def test_fraction_zero_equals_base() -> None:
    c = ResidualCodec(BASE, 0.0)
    e = c.encode(W, seed=3)
    assert isinstance(e, Encoded) and e.codec == c.name == "vq+res0_fp16_magnitude"
    wb, _ = _base_resid()
    assert np.array_equal(_np(c.decode(e)), wb)
    b, bb = e.bits(), BASE.encode(W, seed=3).bits()
    assert b["residual"] == 0 and b == bb


def test_fraction_one_fp16_reproduces_w() -> None:
    c = ResidualCodec(BASE, 1.0)
    wh = _np(c.decode(c.encode(W, seed=3)))
    _, r = _base_resid()
    # fp16 rounding of R (normal: 2^-11 relative; subnormal: 2^-25 absolute) + one float32 add
    assert np.all(np.abs(_np(W) - wh) <= np.abs(r) * 2.0 ** -11 + 2.0 ** -25 + np.abs(_np(W)) * 2.0 ** -23)


def test_error_monotone_in_fraction() -> None:
    errs = []
    for f in (0.0, 0.01, 0.05, 0.2, 0.5, 1.0):
        for dt in ("fp16", "int8"):
            c = ResidualCodec(BASE, f, dt)
            errs.append((f, dt, float(((_np(W) - _np(c.decode(c.encode(W, seed=3)))) ** 2).sum())))
    for dt in ("fp16", "int8"):
        e = [x for _, d, x in errs if d == dt]
        assert all(a > b for a, b in zip(e, e[1:])), (dt, e)


def test_magnitude_picks_largest_abs_residual() -> None:
    c = ResidualCodec(BASE, 0.03)
    e = c.encode(W, seed=3)
    _, r = _base_resid()
    n = round(0.03 * W.size)
    pos = np.array(e.arrays["res_pos"])
    assert e.arrays["res_pos"].dtype == mx.uint32 and pos.size == n == e.params["n"]
    assert np.all(np.diff(pos.astype(np.int64)) > 0)  # sorted, unique
    assert set(pos.tolist()) == set(np.argsort(-np.abs(r).ravel())[:n].tolist())
    np.testing.assert_array_equal(np.array(e.arrays["res_val"]), r.ravel()[pos].astype(np.float16))


def test_importance_ones_equals_magnitude() -> None:
    a = ResidualCodec(BASE, 0.05).encode(W, seed=3)
    b = ResidualCodec(BASE, 0.05, selection="importance").encode(W, seed=3, importance=mx.ones(W.shape))
    assert np.array_equal(np.array(a.arrays["res_pos"]), np.array(b.arrays["res_pos"]))


def test_importance_required_for_importance_mode() -> None:
    with pytest.raises(ValueError):
        ResidualCodec(BASE, 0.05, selection="importance").encode(W, seed=3)
    with pytest.raises(ValueError):
        ResidualCodec(BASE, 0.05, selection="importance").encode(W, seed=3, importance=mx.ones((3, 3)))


def test_importance_selection_beats_magnitude_on_weighted_error() -> None:
    rng = np.random.default_rng(7)
    f = np.full(W.shape, 1e-3)
    hot = rng.choice(W.size, size=W.size // 20, replace=False)  # 5% of entries carry nearly all importance
    f.ravel()[hot] = 1.0
    imp = mx.array(f.astype(np.float32))

    def werr(sel: str) -> float:
        c = ResidualCodec(BASE, 0.02, selection=sel)
        wh = _np(c.decode(c.encode(W, seed=3, importance=imp)))
        return float((f * (_np(W) - wh) ** 2).sum())
    assert werr("importance") < 0.7 * werr("magnitude")


def test_int8_error_bounded_by_step() -> None:
    c = ResidualCodec(BASE, 0.1, "int8")
    e = c.encode(W, seed=3)
    assert e.arrays["res_val"].dtype == mx.int8 and e.arrays["res_scale"].dtype == mx.float16
    s = float(e.arrays["res_scale"].astype(mx.float32).item())
    _, r = _base_resid()
    pos = np.array(e.arrays["res_pos"])
    assert s == pytest.approx(np.abs(r.ravel()[pos]).max() / 127, rel=1e-3)
    err = np.abs(_np(W) - _np(c.decode(e))).ravel()
    assert err[pos].max() <= 0.5 * s * (1 + 1e-4)
    mask = np.ones(W.size, bool)
    mask[pos] = False
    np.testing.assert_allclose(err[mask], np.abs(r.ravel()[mask]), atol=1e-9)


@pytest.mark.parametrize("dt,vbits,extra", [("fp16", 16, 0), ("int8", 8, 16)])
def test_bits_accounting(dt: str, vbits: int, extra: int) -> None:
    w = _randn((64, 128), 2)
    base = AffineRTN(4, 64)
    e = ResidualCodec(base, 0.01, dt).encode(w, seed=0)
    bb = base.encode(w, seed=0).bits()
    n, N = round(0.01 * 8192), 8192  # 82 entries, positions need ceil(log2 8192) = 13 bits
    b = e.bits()
    assert b["residual"] == n * (13 + vbits) + extra == 82 * (13 + vbits) + extra
    assert {k: b[k] for k in ("indices", "codebooks", "scales", "metadata")} == \
        {k: bb[k] for k in ("indices", "codebooks", "scales", "metadata")}
    assert b["total"] == bb["total"] + b["residual"]
    assert e.bpw() == b["total"] / N
    p = e.params
    assert p["position_bits"] == 13 and p["value_bits"] == vbits
    assert p["position_bits_entropy"] == pytest.approx(math.log2(N / n) + 1.44)
    assert p["residual_bits_entropy"] == pytest.approx(n * (p["position_bits_entropy"] + vbits) + extra)


@pytest.mark.parametrize("shape", [(32, 256), (256, 64), (768, 128)])
def test_shapes(shape: tuple[int, int]) -> None:
    w = _randn(shape, 4)
    c = ResidualCodec(VectorCodebook(16, block_size=8, kmeans_iters=3), 0.02, "int8")
    wh = c.decode(c.encode(w, seed=1))
    assert wh.shape == shape and wh.dtype == mx.float32


def test_from_config_and_validation() -> None:
    c = residual_from_config({"base": {"kind": "affine", "bits": 2, "group_size": 64}, "fraction": 0.005,
                              "value_dtype": "int8", "selection": "importance"})
    assert isinstance(c, ResidualCodec) and c.name == "affine+res0.005_int8_importance"
    assert residual_from_config({"base": {"kind": "bf16"}, "fraction": 0.1}).name == "bf16+res0.1_fp16_magnitude"
    for bad in ({"fraction": -0.1}, {"fraction": 1.5}, {"fraction": 0.1, "value_dtype": "int4"},
                {"fraction": 0.1, "selection": "random"}):
        with pytest.raises(ValueError):
            residual_from_config({"base": {"kind": "bf16"}, **bad})
    e = c.encode(W, seed=0, importance=mx.ones(W.shape))
    with pytest.raises(ValueError):
        ResidualCodec(BASE, 0.005).decode(e)
