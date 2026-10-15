"""Row-wise affine quantization of lookup tables (qwen4_exp n-gram embedding shards, docs/arch/qwen4_exp.md §6).

One group per table row (group_size = row width, e.g. 160): q = round((x - b) / s) in [0, 2^bits - 1] with
b = min(row), s = (max(row) - min(row)) / (2^bits - 1) (min-max RTN, the MLX affine rule), s and b stored in f16
(or bf16, one dtype per file: the directory precedes the payloads, so it is fixed before any shard is read; a range
that overflows f16 is an error). Decode: q * s + b in float32 (= the SUC1 reference decoder of an `affine` tensor).
Storage cost: bits + 2 * 16 / row_width bits per value (4-bit, 160 wide: 4.2 bpp).

Packed layout = suc.quant.packing (rows of little-endian u32 words, code i at bits [i*bits, (i+1)*bits) of the row's
bit stream); with bits in {2, 4, 8} a row of width n needs n * bits / 32 words (n * bits must be a multiple of 32).

Everything here is numpy on the CPU: a table never goes to the GPU whole, and a shard is processed in row chunks.
"""
from __future__ import annotations

import numpy as np

ROW_BITS = (2, 4, 8)


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bf16 bit pattern (uint16); NaN stays NaN."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + np.uint32(0x7FFF)
    out = ((u + r) >> 16).astype(np.uint16)
    return np.where(np.isnan(x), np.uint16(0x7FC0), out)


def words_per_row(n: int, bits: int) -> int:
    if bits not in ROW_BITS:
        raise ValueError(f"row-affine bits must be one of {ROW_BITS}, got {bits}")
    if (n * bits) % 32:
        raise ValueError(f"row width {n} x {bits} bits is not a multiple of 32")
    return n * bits // 32


def pack_rows(q: np.ndarray, bits: int) -> np.ndarray:
    """(rows, n) codes < 2^bits -> (rows, n*bits/32) uint32 in the suc.quant.packing layout."""
    rows, n = q.shape
    per = 32 // bits
    w = words_per_row(n, bits)
    q32 = q.astype(np.uint32).reshape(rows, w, per)
    out = np.zeros((rows, w), np.uint32)
    for j in range(per):
        out |= q32[:, :, j] << np.uint32(j * bits)
    return out


def unpack_rows(words: np.ndarray, bits: int, n: int) -> np.ndarray:
    """Inverse of pack_rows: (rows, n*bits/32) uint32 -> (rows, n) uint8."""
    rows = words.shape[0]
    per = 32 // bits
    if words.shape[1] != words_per_row(n, bits):
        raise ValueError(f"{words.shape[1]} words per row do not hold {n} codes of {bits} bits")
    shifts = (np.arange(per, dtype=np.uint32) * bits)
    q = (words[:, :, None] >> shifts) & np.uint32((1 << bits) - 1)
    return q.reshape(rows, n).astype(np.uint8)


def _store(v: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        return f32_to_bf16(v)
    with np.errstate(over="ignore"):         # overflow -> inf, rejected by quantize_rows
        return v.astype("<f2")


def _load(v: np.ndarray, dtype: str) -> np.ndarray:
    return v.astype(np.float32) if dtype == "f16" else bf16_to_f32(v)


def quantize_rows(x: np.ndarray, bits: int = 4, scale_dtype: str = "f16") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """float32 (rows, n) -> (packed words (rows, n*bits/32) uint32, scales (rows, 1), biases (rows, 1)) with scales /
    biases in their storage form (f16 or bf16 bit patterns). Codes are computed against the STORED (rounded) scale
    and bias, so decode error is the RTN error of the stored grid only."""
    if scale_dtype not in ("f16", "bf16"):
        raise ValueError(f"scale_dtype must be f16 or bf16, got {scale_dtype!r}")
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("table rows contain non-finite values")
    qmax = (1 << bits) - 1
    lo, hi = x.min(axis=1), x.max(axis=1)
    b_st = _store(lo, scale_dtype)
    b = _load(b_st, scale_dtype)
    s_st = _store((hi - b) / qmax, scale_dtype)
    s = _load(s_st, scale_dtype)
    if not (np.isfinite(s).all() and np.isfinite(b).all()):
        raise ValueError(f"row range does not fit {scale_dtype} scales / biases (use bf16)")
    safe = np.where(s > 0, s, 1.0).astype(np.float32)
    q = np.clip(np.rint((x - b[:, None]) / safe[:, None]), 0, qmax).astype(np.uint8)
    return pack_rows(q, bits), s_st[:, None], b_st[:, None]


def dequantize_rows(words: np.ndarray, scales: np.ndarray, biases: np.ndarray, bits: int, n: int,
                    scale_dtype: str = "f16") -> np.ndarray:
    """float32 (rows, n) = q * s + b (the SUC1 reference decode of an affine tensor with group_size n)."""
    q = unpack_rows(words, bits, n).astype(np.float32)
    s = _load(np.asarray(scales).reshape(-1), scale_dtype)
    b = _load(np.asarray(biases).reshape(-1), scale_dtype)
    return q * s[:, None] + b[:, None]


__all__ = ["ROW_BITS", "bf16_to_f32", "f32_to_bf16", "words_per_row", "pack_rows", "unpack_rows", "quantize_rows",
           "dequantize_rows"]
