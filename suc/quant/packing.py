"""Bit-packed index layout — the execution layout of spec §37 (the research layout is a plain (rows, n) array).

Each row is an independent little-endian bit stream: element j occupies stream bits [j*bits, (j+1)*bits),
stream bit p lives in word p // 32 at bit p % 32 (LSB first). Elements may straddle a word boundary
(bits = 3, 5, 6, 10, 12 ...). Each row is zero-padded to ceil(n*bits/32) uint32 words so a kernel can
address row r at word offset r * words_per_row. Conversion runs in numpy (not in the inference path).
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np


def words_per_row(n: int, bits: int) -> int:
    return -(-n * bits // 32)


def _check_bits(bits: int) -> None:
    if not 1 <= bits <= 32:
        raise ValueError(f"bits must be in [1, 32], got {bits}")


def pack_indices(idx: mx.array, bits: int) -> mx.array:
    """(rows, n) non-negative ints < 2**bits → (rows, ceil(n*bits/32)) uint32."""
    _check_bits(bits)
    a = np.asarray(idx).astype(np.int64)
    if a.ndim != 2:
        raise ValueError(f"idx must be 2-D (rows, n), got shape {a.shape}")
    if a.size and (a.min() < 0 or a.max() >= 1 << bits):
        raise ValueError(f"indices must lie in [0, 2**{bits})")
    rows, n = a.shape
    w = words_per_row(n, bits)
    stream = ((a[:, :, None] >> np.arange(bits)) & 1).astype(np.uint8).reshape(rows, n * bits)
    stream = np.pad(stream, ((0, 0), (0, w * 32 - n * bits))).reshape(rows, w, 32)
    packed = (stream.astype(np.uint64) << np.arange(32, dtype=np.uint64)).sum(-1, dtype=np.uint64)
    return mx.array(packed.astype(np.uint32))


def unpack_indices(packed: mx.array, bits: int, n: int) -> mx.array:
    """Exact inverse of pack_indices → (rows, n) uint32."""
    _check_bits(bits)
    p = np.asarray(packed).astype(np.uint32)
    rows, w = p.shape
    if w != words_per_row(n, bits):
        raise ValueError(f"packed has {w} words per row, expected {words_per_row(n, bits)} for n={n}, bits={bits}")
    stream = ((p[:, :, None] >> np.arange(32, dtype=np.uint32)) & 1).reshape(rows, w * 32)[:, :n * bits]
    vals = (stream.reshape(rows, n, bits).astype(np.uint64) << np.arange(bits, dtype=np.uint64)).sum(-1, dtype=np.uint64)
    return mx.array(vals.astype(np.uint32))
