"""Randomized Hadamard incoherence processing (spec §0 D3, README §2).

W' = U W V^T with U = H diag(s_out), V = H diag(s_in), H orthonormal and s random +-1 signs,
so W x = U^T W' (V x). Everything acts on the last axis and is computed in float32.

H of size n = m * 2^k is defined as H_m (x) H_{2^k} (Kronecker), m in {1, 12, 20, 28, 17}: the
2^k factor is Sylvester and uses the MLX built-in `mx.hadamard_transform` (split further by
Kronecker above its float32 limit of 8192); the H_m factor is MLX's own m x m matrix for m = 12 / 20 / 28. H_20 and
H_28 are not symmetric, hence the explicit transpose for inverses. No Hadamard matrix of order 17 exists, so m = 17
(17408 = 17 * 2^10: the MLP width of Qwen3.5 / Qwen3.8-27B) uses the orthonormal discrete Hartley matrix
H_17[j, k] = (cos + sin)(2 pi j k / 17) / sqrt(17), symmetric, stored as the float32 table _H17_HEX (rounded once
from float64; |H Hᵀ - I| <= 3.4e-8), the same table in C++ (cpp/src/hadamard.cpp) and the Metal kernels
(suc.kernels.rotation._hm_tables). It is orthogonal, not Hadamard: its entries are not all ±1/sqrt(17) (0.016 …
0.343), so it spreads a coordinate less evenly over the 17 blocks than a Hadamard factor would; the 2^10 Sylvester
factor and the random signs do the bulk of the incoherence. Sizes such as 9728 = 19 * 2^9 (Qwen3-4B intermediate)
are unsupported and raise ValueError.

The H_m step is DEFINED as a sequential float32 fma chain (`_mix`), identical to the C++ decoder
(cpp/src/hadamard.cpp):  o = 0;  for k = 0 .. m-1:  o = fma(H[i, k], y[k, c], o).
An MLX matmul is not used because its accumulation order depends on the device and shape (for
m = 20 / 28 GPU, CPU and the fma chain all differed by up to ~2e-7 relative; m = 12 with 2^k >= 64
already matched the chain, so those results are unchanged). On the GPU the chain is a small Metal
kernel (Metal `fma` is correctly rounded); elsewhere it is computed in numpy, each fma as float64
round-to-odd followed by rounding to float32, which equals the correctly rounded float32 fma
(53 >= 24 + 2, Boldo & Melquiond 2008). Both are bit-identical to each other and to C++ (float32
subnormals aside: Metal may flush them). Gradients (vjp) use the transposed chain (the step is linear); jvp/vmap are not supported.
"""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

_M_FACTORS = (1, 12, 20, 28, 17)
_MLX_M = (1, 12, 20, 28)   # orders mx.hadamard_transform handles natively
# H_17 (discrete Hartley, row-major, float32 hex): see the module docstring
_H17_HEX = (
    "0x1.f0b6840000000p-3", "0x1.f0b6840000000p-3", "0x1.f0b6840000000p-3", "0x1.f0b6840000000p-3",
    "0x1.f0b6840000000p-3", "0x1.f0b6840000000p-3", "0x1.f0b6840000000p-3", "0x1.f0b6840000000p-3",
    "0x1.f0b6840000000p-3", "0x1.f0b6840000000p-3", "0x1.f0b6840000000p-3", "0x1.f0b6840000000p-3",
    "0x1.f0b6840000000p-3", "0x1.f0b6840000000p-3", "0x1.f0b6840000000p-3", "0x1.f0b6840000000p-3",
    "0x1.f0b6840000000p-3", "0x1.f0b6840000000p-3", "0x1.414d5e0000000p-2", "0x1.5edab60000000p-2",
    "0x1.4d05880000000p-2", "0x1.0e36660000000p-2", "0x1.55d1be0000000p-3", "0x1.84323e0000000p-5",
    "-0x1.41a8080000000p-4", "-0x1.8cfc160000000p-3", "-0x1.21c35c0000000p-2", "-0x1.55e6580000000p-2",
    "-0x1.5bdc680000000p-2", "-0x1.32d7740000000p-2", "-0x1.c0c3600000000p-3", "-0x1.be786a0000000p-4",
    "0x1.03891c0000000p-6", "0x1.1bbcd40000000p-3", "0x1.f0b6840000000p-3", "0x1.5edab60000000p-2",
    "0x1.0e36660000000p-2", "0x1.84323e0000000p-5", "-0x1.8cfc160000000p-3", "-0x1.55e6580000000p-2",
    "-0x1.32d7740000000p-2", "-0x1.be786a0000000p-4", "0x1.1bbcd40000000p-3", "0x1.414d5e0000000p-2",
    "0x1.4d05880000000p-2", "0x1.55d1be0000000p-3", "-0x1.41a8080000000p-4", "-0x1.21c35c0000000p-2",
    "-0x1.5bdc680000000p-2", "-0x1.c0c3600000000p-3", "0x1.03891c0000000p-6", "0x1.f0b6840000000p-3",
    "0x1.4d05880000000p-2", "0x1.84323e0000000p-5", "-0x1.21c35c0000000p-2", "-0x1.32d7740000000p-2",
    "0x1.03891c0000000p-6", "0x1.414d5e0000000p-2", "0x1.0e36660000000p-2", "-0x1.41a8080000000p-4",
    "-0x1.55e6580000000p-2", "-0x1.c0c3600000000p-3", "0x1.1bbcd40000000p-3", "0x1.5edab60000000p-2",
    "0x1.55d1be0000000p-3", "-0x1.8cfc160000000p-3", "-0x1.5bdc680000000p-2", "-0x1.be786a0000000p-4",
    "0x1.f0b6840000000p-3", "0x1.0e36660000000p-2", "-0x1.8cfc160000000p-3", "-0x1.32d7740000000p-2",
    "0x1.1bbcd40000000p-3", "0x1.4d05880000000p-2", "-0x1.41a8080000000p-4", "-0x1.5bdc680000000p-2",
    "0x1.03891c0000000p-6", "0x1.5edab60000000p-2", "0x1.84323e0000000p-5", "-0x1.55e6580000000p-2",
    "-0x1.be786a0000000p-4", "0x1.414d5e0000000p-2", "0x1.55d1be0000000p-3", "-0x1.21c35c0000000p-2",
    "-0x1.c0c3600000000p-3", "0x1.f0b6840000000p-3", "0x1.55d1be0000000p-3", "-0x1.55e6580000000p-2",
    "0x1.03891c0000000p-6", "0x1.4d05880000000p-2", "-0x1.8cfc160000000p-3", "-0x1.c0c3600000000p-3",
    "0x1.414d5e0000000p-2", "0x1.84323e0000000p-5", "-0x1.5bdc680000000p-2", "0x1.1bbcd40000000p-3",
    "0x1.0e36660000000p-2", "-0x1.21c35c0000000p-2", "-0x1.be786a0000000p-4", "0x1.5edab60000000p-2",
    "-0x1.41a8080000000p-4", "-0x1.32d7740000000p-2", "0x1.f0b6840000000p-3", "0x1.84323e0000000p-5",
    "-0x1.32d7740000000p-2", "0x1.414d5e0000000p-2", "-0x1.41a8080000000p-4", "-0x1.c0c3600000000p-3",
    "0x1.5edab60000000p-2", "-0x1.8cfc160000000p-3", "-0x1.be786a0000000p-4", "0x1.4d05880000000p-2",
    "-0x1.21c35c0000000p-2", "0x1.03891c0000000p-6", "0x1.0e36660000000p-2", "-0x1.55e6580000000p-2",
    "0x1.1bbcd40000000p-3", "0x1.55d1be0000000p-3", "-0x1.5bdc680000000p-2", "0x1.f0b6840000000p-3",
    "-0x1.41a8080000000p-4", "-0x1.be786a0000000p-4", "0x1.0e36660000000p-2", "-0x1.5bdc680000000p-2",
    "0x1.414d5e0000000p-2", "-0x1.8cfc160000000p-3", "0x1.03891c0000000p-6", "0x1.55d1be0000000p-3",
    "-0x1.32d7740000000p-2", "0x1.5edab60000000p-2", "-0x1.21c35c0000000p-2", "0x1.1bbcd40000000p-3",
    "0x1.84323e0000000p-5", "-0x1.c0c3600000000p-3", "0x1.4d05880000000p-2", "-0x1.55e6580000000p-2",
    "0x1.f0b6840000000p-3", "-0x1.8cfc160000000p-3", "0x1.1bbcd40000000p-3", "-0x1.41a8080000000p-4",
    "0x1.03891c0000000p-6", "0x1.84323e0000000p-5", "-0x1.be786a0000000p-4", "0x1.55d1be0000000p-3",
    "-0x1.c0c3600000000p-3", "0x1.0e36660000000p-2", "-0x1.32d7740000000p-2", "0x1.4d05880000000p-2",
    "-0x1.5bdc680000000p-2", "0x1.5edab60000000p-2", "-0x1.55e6580000000p-2", "0x1.414d5e0000000p-2",
    "-0x1.21c35c0000000p-2", "0x1.f0b6840000000p-3", "-0x1.21c35c0000000p-2", "0x1.414d5e0000000p-2",
    "-0x1.55e6580000000p-2", "0x1.5edab60000000p-2", "-0x1.5bdc680000000p-2", "0x1.4d05880000000p-2",
    "-0x1.32d7740000000p-2", "0x1.0e36660000000p-2", "-0x1.c0c3600000000p-3", "0x1.55d1be0000000p-3",
    "-0x1.be786a0000000p-4", "0x1.84323e0000000p-5", "0x1.03891c0000000p-6", "-0x1.41a8080000000p-4",
    "0x1.1bbcd40000000p-3", "-0x1.8cfc160000000p-3", "0x1.f0b6840000000p-3", "-0x1.55e6580000000p-2",
    "0x1.4d05880000000p-2", "-0x1.c0c3600000000p-3", "0x1.84323e0000000p-5", "0x1.1bbcd40000000p-3",
    "-0x1.21c35c0000000p-2", "0x1.5edab60000000p-2", "-0x1.32d7740000000p-2", "0x1.55d1be0000000p-3",
    "0x1.03891c0000000p-6", "-0x1.8cfc160000000p-3", "0x1.414d5e0000000p-2", "-0x1.5bdc680000000p-2",
    "0x1.0e36660000000p-2", "-0x1.be786a0000000p-4", "-0x1.41a8080000000p-4", "0x1.f0b6840000000p-3",
    "-0x1.5bdc680000000p-2", "0x1.55d1be0000000p-3", "0x1.1bbcd40000000p-3", "-0x1.55e6580000000p-2",
    "0x1.0e36660000000p-2", "0x1.03891c0000000p-6", "-0x1.21c35c0000000p-2", "0x1.4d05880000000p-2",
    "-0x1.be786a0000000p-4", "-0x1.8cfc160000000p-3", "0x1.5edab60000000p-2", "-0x1.c0c3600000000p-3",
    "-0x1.41a8080000000p-4", "0x1.414d5e0000000p-2", "-0x1.32d7740000000p-2", "0x1.84323e0000000p-5",
    "0x1.f0b6840000000p-3", "-0x1.32d7740000000p-2", "-0x1.41a8080000000p-4", "0x1.5edab60000000p-2",
    "-0x1.be786a0000000p-4", "-0x1.21c35c0000000p-2", "0x1.0e36660000000p-2", "0x1.1bbcd40000000p-3",
    "-0x1.5bdc680000000p-2", "0x1.84323e0000000p-5", "0x1.414d5e0000000p-2", "-0x1.c0c3600000000p-3",
    "-0x1.8cfc160000000p-3", "0x1.4d05880000000p-2", "0x1.03891c0000000p-6", "-0x1.55e6580000000p-2",
    "0x1.55d1be0000000p-3", "0x1.f0b6840000000p-3", "-0x1.c0c3600000000p-3", "-0x1.21c35c0000000p-2",
    "0x1.55d1be0000000p-3", "0x1.414d5e0000000p-2", "-0x1.be786a0000000p-4", "-0x1.55e6580000000p-2",
    "0x1.84323e0000000p-5", "0x1.5edab60000000p-2", "0x1.03891c0000000p-6", "-0x1.5bdc680000000p-2",
    "-0x1.41a8080000000p-4", "0x1.4d05880000000p-2", "0x1.1bbcd40000000p-3", "-0x1.32d7740000000p-2",
    "-0x1.8cfc160000000p-3", "0x1.0e36660000000p-2", "0x1.f0b6840000000p-3", "-0x1.be786a0000000p-4",
    "-0x1.5bdc680000000p-2", "-0x1.8cfc160000000p-3", "0x1.55d1be0000000p-3", "0x1.5edab60000000p-2",
    "0x1.1bbcd40000000p-3", "-0x1.c0c3600000000p-3", "-0x1.55e6580000000p-2", "-0x1.41a8080000000p-4",
    "0x1.0e36660000000p-2", "0x1.414d5e0000000p-2", "0x1.03891c0000000p-6", "-0x1.32d7740000000p-2",
    "-0x1.21c35c0000000p-2", "0x1.84323e0000000p-5", "0x1.4d05880000000p-2", "0x1.f0b6840000000p-3",
    "0x1.03891c0000000p-6", "-0x1.c0c3600000000p-3", "-0x1.5bdc680000000p-2", "-0x1.21c35c0000000p-2",
    "-0x1.41a8080000000p-4", "0x1.55d1be0000000p-3", "0x1.4d05880000000p-2", "0x1.414d5e0000000p-2",
    "0x1.1bbcd40000000p-3", "-0x1.be786a0000000p-4", "-0x1.32d7740000000p-2", "-0x1.55e6580000000p-2",
    "-0x1.8cfc160000000p-3", "0x1.84323e0000000p-5", "0x1.0e36660000000p-2", "0x1.5edab60000000p-2",
    "0x1.f0b6840000000p-3", "0x1.1bbcd40000000p-3", "0x1.03891c0000000p-6", "-0x1.be786a0000000p-4",
    "-0x1.c0c3600000000p-3", "-0x1.32d7740000000p-2", "-0x1.5bdc680000000p-2", "-0x1.55e6580000000p-2",
    "-0x1.21c35c0000000p-2", "-0x1.8cfc160000000p-3", "-0x1.41a8080000000p-4", "0x1.84323e0000000p-5",
    "0x1.55d1be0000000p-3", "0x1.0e36660000000p-2", "0x1.4d05880000000p-2", "0x1.5edab60000000p-2",
    "0x1.414d5e0000000p-2"
)
_MAX_POW2 = 8192  # mx.hadamard_transform float32 limit on the 2^k factor


def _factor(n: int) -> tuple[int, int] | None:
    for m in _M_FACTORS:
        if n > 0 and n % m == 0 and (p := n // m) & (p - 1) == 0:
            return m, p
    return None


def supported_size(n: int) -> bool:
    return _factor(n) is not None


@lru_cache(maxsize=None)
def _h_m(m: int) -> mx.array:
    """Orthonormal m x m factor: MLX's Hadamard matrix (hadamard_transform(I) = H^T) for m = 12 / 20 / 28, the float32
    Hartley table for m = 17."""
    if m == 17:
        return mx.array(np.array([float.fromhex(h) for h in _H17_HEX], dtype=np.float32).reshape(17, 17))
    return mx.hadamard_transform(mx.eye(m, dtype=mx.float32)).T


@lru_cache(maxsize=None)
def _h_m_np(m: int, transpose: bool) -> np.ndarray:
    h = np.array(_h_m(m), dtype=np.float32)
    return np.ascontiguousarray(h.T if transpose else h)


_MIX_SOURCE = """
    uint c = thread_position_in_grid.x;
    uint ri = thread_position_in_grid.y;
    if (c >= P) return;
    uint i = ri % M;
    const device float* yr = y + ulong(ri - i) * P + c;
    const device float* hr = h + i * M;
    float o = 0.0f;
    for (uint k = 0; k < M; ++k) o = fma(hr[k], yr[ulong(k) * P], o);
    out[ulong(ri) * P + c] = o;
"""


@lru_cache(maxsize=None)
def _mix_kernel():
    return mx.fast.metal_kernel(name="suc_hadamard_mix_fma", input_names=["y", "h"], output_names=["out"],
                                source=_MIX_SOURCE)


def _mix_metal(y: mx.array, m: int, transpose: bool) -> mx.array:
    p = y.shape[-1]
    rows = y.size // p  # = batch * m
    h = mx.array(_h_m_np(m, transpose).reshape(-1))
    return _mix_kernel()(inputs=[y, h], template=[("M", m), ("P", p)], grid=(p, rows, 1),
                         threadgroup=(min(256, p), 1, 1), output_shapes=[y.shape],
                         output_dtypes=[mx.float32])[0]


def _fma_f32(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Correctly rounded float32 fma(a, b, c) (elementwise, float32 inputs) via float64 round-to-odd."""
    prod = a.astype(np.float64) * b  # exact: 24 + 24 bits < 53
    c64 = c.astype(np.float64)
    s = prod + c64
    bb = s - prod
    err = (prod - (s - bb)) + (c64 - bb)  # TwoSum: s + err == prod + c exactly
    fix = (err != 0) & ((s.view(np.int64) & 1) == 0) & np.isfinite(s)
    if fix.any():
        s[fix] = np.nextafter(s[fix], np.where(err[fix] > 0, np.inf, -np.inf))
    return s.astype(np.float32)


def _mix_numpy(y: np.ndarray, m: int, transpose: bool, chunk: int = 1 << 22) -> np.ndarray:
    """out[..., i, c] = fma chain over k of h[i, k] * y[..., k, c]; y is float32 (..., m, p)."""
    h = _h_m_np(m, transpose)
    p = y.shape[-1]
    y3 = np.ascontiguousarray(y, dtype=np.float32).reshape(-1, m, p)
    out = np.empty_like(y3)
    step = max(1, chunk // (m * p))
    for r0 in range(0, y3.shape[0], step):
        yb = y3[r0:r0 + step]
        o = np.zeros_like(yb)
        for k in range(m):
            o = _fma_f32(h[None, :, k, None], yb[:, k:k + 1, :], o)
        out[r0:r0 + step] = o
    return out.reshape(y.shape)


def _mix_impl(y: mx.array, m: int, transpose: bool) -> mx.array:
    if mx.default_device() == mx.gpu and mx.metal.is_available():
        return _mix_metal(y, m, transpose)
    return mx.array(_mix_numpy(np.array(y), m, transpose))


@lru_cache(maxsize=None)
def _mix_fn(m: int, transpose: bool):
    @mx.custom_function
    def mix(y):
        return _mix_impl(y, m, transpose)

    @mix.vjp
    def mix_vjp(primals, cotangent, output):
        return _mix_impl(cotangent.astype(mx.float32), m, not transpose)

    return mix


def _mix(y: mx.array, m: int, transpose: bool) -> mx.array:
    """H_m (or H_m^T) applied along axis -2 of float32 y (..., m, p) as a sequential fma chain."""
    return _mix_fn(m, transpose)(y)


def _pow2(x: mx.array) -> mx.array:
    p = x.shape[-1]
    if p <= _MAX_POW2:
        return mx.hadamard_transform(x)
    lead = x.shape[:-1]
    y = mx.hadamard_transform(x.reshape(*lead, p // _MAX_POW2, _MAX_POW2))
    y = _pow2(mx.swapaxes(y, -1, -2))
    return mx.swapaxes(y, -1, -2).reshape(*lead, p)


def _transform(x: mx.array, transpose: bool) -> mx.array:
    n = x.shape[-1]
    if (f := _factor(n)) is None:
        raise ValueError(f"unsupported Hadamard size {n}: need n = m * 2^k with m in {_M_FACTORS}")
    m, p = f
    x = x.astype(mx.float32)
    if m == 1:
        return _pow2(x)
    lead = x.shape[:-1]
    y = _pow2(x.reshape(*lead, m, p))
    return _mix(y, m, transpose).reshape(*lead, n)


def hadamard(x: mx.array, scale: bool = True) -> mx.array:
    """Walsh-Hadamard transform H x on the last axis; orthonormal if scale, else +-1 entries."""
    y = _transform(x, transpose=False)
    return y if scale else y * float(np.sqrt(x.shape[-1]))


def random_signs(n: int, seed: int) -> mx.array:
    """Deterministic +-1 float32 vector from (seed, n) (numpy PCG64, stable across versions)."""
    bits = np.random.default_rng(seed).integers(0, 2, size=n)
    return mx.array((2 * bits - 1).astype(np.float32))


def rht(x: mx.array, seed: int) -> mx.array:
    """Randomized Hadamard V x = H (s * x) on the last axis."""
    n = x.shape[-1]
    if not supported_size(n):
        raise ValueError(f"unsupported Hadamard size {n}")
    return _transform(x.astype(mx.float32) * random_signs(n, seed), transpose=False)


def rht_inverse(y: mx.array, seed: int) -> mx.array:
    """Exact inverse V^T y = s * (H^T y) on the last axis."""
    return _transform(y, transpose=True) * random_signs(y.shape[-1], seed)


def rotate_weight(w: mx.array, seed_in: int, seed_out: int) -> mx.array:
    """W' = U W V^T for W of shape (out, in): V on the in dim, U on the out dim. Returns float32."""
    return rht(rht(w, seed_in).T, seed_out).T


def unrotate_weight(wr: mx.array, seed_in: int, seed_out: int) -> mx.array:
    """Inverse of rotate_weight: W = U^T W' V."""
    return rht_inverse(rht_inverse(wr, seed_in).T, seed_out).T
