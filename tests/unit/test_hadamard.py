"""Tests for randomized Hadamard incoherence processing (spec §0 D3, README §2)."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from suc.quant.hadamard import (hadamard, random_signs, rht, rht_inverse, rotate_weight,
                                supported_size, unrotate_weight)

# Real Qwen3 dims: 0.6B hidden/intermediate, 8B hidden/intermediate, 30B-A3B hidden/moe_intermediate.
QWEN3_SIZES = (1024, 3072, 4096, 12288, 2048, 768, 6144, 128)


def _np(x: mx.array) -> np.ndarray:
    return np.array(x.astype(mx.float32))


def _randn(shape: tuple[int, ...], seed: int = 0) -> mx.array:
    return mx.array(np.random.default_rng(seed).standard_normal(shape).astype(np.float32))


def test_supported_size() -> None:
    for n in (*QWEN3_SIZES, 1, 2, 12, 20, 28, 40, 56, 16384, 32768, 20 * 4096, 17, 17 * 1024):
        assert supported_size(n), n
    for n in (0, -4, 3, 6, 9728, 25600, 19 * 64):
        assert not supported_size(n), n


@pytest.mark.parametrize("n", [1, 2, 8, 12, 20, 24, 28, 40, 56, 64, 96])
def test_orthonormal(n: int) -> None:
    h = _np(hadamard(mx.eye(n)))  # rows = H e_i -> H^T
    np.testing.assert_allclose(h @ h.T, np.eye(n), atol=1e-5)
    hu = _np(hadamard(mx.eye(n), scale=False))
    np.testing.assert_allclose(np.abs(hu), 1.0, atol=1e-5)  # +-1 entries when unscaled


def test_hartley_factor_17() -> None:
    """m = 17 (Qwen3.8-27B MLP width 17408 = 17 * 2^10): the float32 Hartley table is symmetric, orthonormal to float32
    rounding and equals its float64 definition rounded once."""
    from suc.quant.hadamard import _h_m

    h = np.array(_h_m(17))
    j = np.arange(17)
    a = 2 * np.pi * np.outer(j, j) / 17
    assert np.array_equal(h, ((np.cos(a) + np.sin(a)) / np.sqrt(17)).astype(np.float32))
    assert np.array_equal(h, h.T)
    np.testing.assert_allclose(h.astype(np.float64) @ h.T.astype(np.float64), np.eye(17), atol=5e-8)
    full = _np(hadamard(mx.eye(17 * 32)))
    np.testing.assert_allclose(full @ full.T, np.eye(17 * 32), atol=1e-5)


@pytest.mark.parametrize("n", (*QWEN3_SIZES, 16384, 28 * 512, 17 * 1024, 17 * 64))
def test_norm_preservation_and_roundtrip(n: int) -> None:
    x = _randn((3, 5, n), seed=n)
    y = rht(x, seed=7)
    assert y.shape == x.shape and y.dtype == mx.float32
    np.testing.assert_allclose(np.linalg.norm(_np(y), axis=-1), np.linalg.norm(_np(x), axis=-1),
                               rtol=1e-5)
    err = float(mx.abs(rht_inverse(y, seed=7) - x).max())
    assert err < 1e-5, err


def test_unsupported_raises() -> None:
    with pytest.raises(ValueError):
        hadamard(mx.zeros((2, 9728)))
    with pytest.raises(ValueError):
        rht(mx.zeros((6,)), seed=0)
    with pytest.raises(ValueError):
        rotate_weight(mx.zeros((4, 9728)), 0, 1)


def test_random_signs_deterministic() -> None:
    a, b, c = random_signs(1024, 3), random_signs(1024, 3), random_signs(1024, 4)
    assert a.dtype == mx.float32 and a.shape == (1024,)
    assert set(np.unique(_np(a)).tolist()) == {-1.0, 1.0}
    assert mx.array_equal(a, b) and not mx.array_equal(a, c)
    x = _randn((4, 256))
    assert mx.array_equal(rht(x, 11), rht(x, 11))
    assert not mx.allclose(rht(x, 11), rht(x, 12))


@pytest.mark.parametrize("shape", [(768, 2048), (3072, 1024), (128, 12288), (64, 17408)])
def test_weight_identity(shape: tuple[int, int]) -> None:
    out_f, in_f = shape
    w, x = _randn(shape, 1), _randn((in_f, 3), 2)
    wr = rotate_weight(w, seed_in=5, seed_out=9)
    # W x == U^T (W' (V x)); rht acts on the last axis, so move the feature axis last.
    y = rht_inverse((wr @ rht(x.T, 5).T).T, 9).T
    ref = w @ x
    rel = float(mx.abs(y - ref).max() / mx.abs(ref).max())
    assert rel < 1e-5, rel
    err = float(mx.abs(unrotate_weight(wr, 5, 9) - w).max())
    assert err < 1e-5, err
    np.testing.assert_allclose(np.linalg.norm(_np(wr)), np.linalg.norm(_np(w)), rtol=1e-5)


def test_bfloat16_input_computed_in_float32() -> None:
    w = _randn((256, 1024)).astype(mx.bfloat16)
    wr = rotate_weight(w, 1, 2)
    assert wr.dtype == mx.float32
    back = unrotate_weight(wr, 1, 2)
    err = float(mx.abs(back - w.astype(mx.float32)).max())
    assert err < 1e-5, err  # the bf16 values are exactly representable in float32


def test_outlier_suppression() -> None:
    rng = np.random.default_rng(0)
    w = rng.standard_normal((1024, 3072)).astype(np.float32) * 0.02
    idx = rng.choice(w.size, size=8, replace=False)
    w.flat[idx] = 5.0 * np.sign(rng.standard_normal(8))  # few huge outliers (~250 sigma)
    ratio = lambda a: float(np.abs(a).max() / a.std())  # noqa: E731
    kurt = lambda a: float(((a - a.mean()) ** 4).mean() / a.var() ** 2)  # noqa: E731
    wr = _np(rotate_weight(mx.array(w), 3, 4))
    print(f"\nmax|w|/std before={ratio(w):.1f} after={ratio(wr):.1f}; "
          f"kurtosis before={kurt(w):.1f} after={kurt(wr):.2f}")
    assert ratio(wr) < ratio(w) / 10
    assert kurt(wr) < 3.5  # close to Gaussian (3.0)


# m = 20 / 28 sizes: 5120 = 20 * 256 (Qwen3-14B/32B hidden), 7168 = 28 * 256; plus small and split (> 8192) ones.
DETERMINISM_SIZES = (5120, 7168, 20, 56, 640, 448, 3072, 12288, 20 * 16384)


@pytest.mark.parametrize("n", DETERMINISM_SIZES)
@pytest.mark.parametrize("transpose", [False, True])
def test_cpu_gpu_bit_exact(n: int, transpose: bool) -> None:
    """The H_m step is a sequential fma chain: MLX GPU (Metal kernel) and CPU (numpy) agree bit for bit."""
    if not mx.metal.is_available():
        pytest.skip("no Metal device")
    from suc.quant.hadamard import _transform
    x = _randn((3 if n > 100000 else 16, n), seed=n)
    g = _np(_transform(x, transpose))
    with mx.stream(mx.cpu):
        c = _np(_transform(x, transpose))
    assert np.array_equal(g.view(np.uint32), c.view(np.uint32))


@pytest.mark.parametrize("m", [12, 20, 28])
def test_mix_is_sequential_fma_chain(m: int) -> None:
    """_mix equals o = fma(H[i, k], y[k, c], o) for k = 0..m-1 (C++ decoder order), computed exactly here with
    Python floats (float64 holds the exact product; the sum is rounded to odd before the float32 rounding)."""
    from suc.quant.hadamard import _fma_f32, _h_m_np, _mix
    rng = np.random.default_rng(m)
    y = rng.standard_normal((2, m, 64)).astype(np.float32) * np.float32(10.0) ** rng.integers(-3, 3, (2, m, 64))
    y = y.astype(np.float32)
    for transpose in (False, True):
        h = _h_m_np(m, transpose)
        ref = np.zeros_like(y)
        for k in range(m):
            ref = _fma_f32(np.broadcast_to(h[:, k, None], y.shape), np.broadcast_to(y[:, k:k + 1, :], y.shape), ref)
        got = _np(_mix(mx.array(y), m, transpose))
        assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))


def test_fma_f32_correctly_rounded() -> None:
    """Round-to-odd emulation vs an exact rational reference on cases where naive float64 double rounding fails."""
    from fractions import Fraction
    from suc.quant.hadamard import _fma_f32
    rng = np.random.default_rng(0)
    a = rng.standard_normal(4000).astype(np.float32)
    b = rng.standard_normal(4000).astype(np.float32)
    # c close to -a*b so the exact sum needs many bits, plus tie-adjacent constructions
    c = (-(a.astype(np.float64) * b)).astype(np.float32) + rng.standard_normal(4000).astype(np.float32) * 1e-7
    c = c.astype(np.float32)
    a[:1000] = np.float32(1 + 2 ** -23)
    b[:1000] = np.float32(1 + 2 ** -23)
    c[:1000] = np.float32(2 ** -48) * rng.choice([-1, 1], 1000).astype(np.float32)
    got = _fma_f32(a, b, c)
    for i in range(0, 4000, 7):
        exact = Fraction(float(a[i])) * Fraction(float(b[i])) + Fraction(float(c[i]))
        # correctly rounded float32 of the exact value: compare against both neighbours of got
        g = got[i]
        lo, hi = np.nextafter(g, np.float32(-np.inf)), np.nextafter(g, np.float32(np.inf))
        dg = abs(Fraction(float(g)) - exact)
        assert dg <= abs(Fraction(float(lo)) - exact) and dg <= abs(Fraction(float(hi)) - exact), i


@pytest.mark.parametrize("n", [768, 3072, 6144, 12288])
def test_m12_unchanged_vs_matmul(n: int) -> None:
    """The fma chain reproduces the former MLX matmul H_12 step bit for bit on the Qwen3 m = 12 sizes."""
    from suc.quant.hadamard import _factor, _h_m, _pow2, _transform
    m, p = _factor(n)
    x = _randn((8, n), seed=n + 1)
    for transpose in (False, True):
        h = _h_m(m)
        old = ((h.T if transpose else h) @ _pow2(x.reshape(8, m, p))).reshape(8, n)
        assert mx.array_equal(old, _transform(x, transpose))
