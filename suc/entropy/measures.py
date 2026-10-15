"""Empirical entropy estimators for code indices and residual metadata (spec §15).

Plug-in estimates are biased low when the number of histogram bins is comparable to the sample size (order-1
with K = 256 has 65536 bins), so every conditional estimate also has a two-fold held-out cross-entropy: counts
fitted on one half, code length measured on the other with add-alpha smoothing. The held-out number is an
achievable rate for a coder that ships the fitted table; the plug-in number is not.
"""
from __future__ import annotations

import math

import numpy as np


def entropy_bits(counts: np.ndarray) -> float:
    """Plug-in Shannon entropy (bits/symbol) of a histogram."""
    c = np.asarray(counts, dtype=np.float64).reshape(-1)
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def order0(sym: np.ndarray, k: int) -> float:
    return entropy_bits(np.bincount(np.asarray(sym).reshape(-1).astype(np.int64), minlength=k))


def conditional_plugin(ctx: np.ndarray, sym: np.ndarray, k: int) -> float:
    """Plug-in H(sym | ctx) = H(ctx, sym) - H(ctx) in bits/symbol."""
    ctx, sym = np.asarray(ctx).reshape(-1).astype(np.int64), np.asarray(sym).reshape(-1).astype(np.int64)
    joint = np.bincount(ctx * k + sym, minlength=k * k)
    return entropy_bits(joint) - entropy_bits(joint.reshape(k, k).sum(axis=1))


def heldout_cross_entropy(ctx_a, sym_a, ctx_b, sym_b, k: int, alpha: float = 0.5) -> float:
    """Bits/symbol of coding (ctx_b, sym_b) with the add-alpha conditional model fitted on (ctx_a, sym_a).
    Pass ctx = zeros for an order-0 model."""
    ca, sa = np.asarray(ctx_a).reshape(-1).astype(np.int64), np.asarray(sym_a).reshape(-1).astype(np.int64)
    cb, sb = np.asarray(ctx_b).reshape(-1).astype(np.int64), np.asarray(sym_b).reshape(-1).astype(np.int64)
    nctx = int(max(ca.max(initial=0), cb.max(initial=0))) + 1
    joint = np.bincount(ca * k + sa, minlength=nctx * k).reshape(nctx, k).astype(np.float64)
    p = (joint + alpha) / (joint.sum(axis=1, keepdims=True) + alpha * k)
    return float(-np.log2(p[cb, sb]).mean()) if cb.size else 0.0


def two_fold(ctx: np.ndarray, sym: np.ndarray, fold: np.ndarray, k: int, alpha: float = 0.5) -> float:
    """Average held-out cross-entropy over the two folds given by the boolean mask `fold`."""
    ctx, sym, fold = (np.asarray(a).reshape(-1) for a in (ctx, sym, fold))
    a, b = fold, ~fold
    h_ab = heldout_cross_entropy(ctx[a], sym[a], ctx[b], sym[b], k, alpha)
    h_ba = heldout_cross_entropy(ctx[b], sym[b], ctx[a], sym[a], k, alpha)
    return float((h_ab * b.sum() + h_ba * a.sum()) / max(1, fold.size))


def log2_binomial(n: int, k: int) -> float:
    """log2 C(n, k): bits to name a k-subset of n positions (bound for sorted unique positions)."""
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def rice_bits(gaps: np.ndarray) -> tuple[int, int]:
    """(best Rice parameter, exact total bits) for coding positive integers `gaps` (>= 1) as gaps - 1."""
    g = np.asarray(gaps, dtype=np.int64).reshape(-1) - 1
    if g.size == 0:
        return 0, 0
    if g.min() < 0:
        raise ValueError("gaps must be >= 1")
    return min(((r, rice_bits_at(gaps, r)) for r in range(32)), key=lambda t: t[1])


def rice_bits_at(gaps: np.ndarray, r: int) -> int:
    """Exact bits of the Rice code with parameter r for gaps (>= 1) coded as gaps - 1."""
    g = np.asarray(gaps, dtype=np.int64).reshape(-1) - 1
    return int((g >> r).sum()) + g.size * (1 + r)


def rice_encode(values: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """Rice code of non-negative integers split into two bit planes (same total bits as interleaved Rice):
    `unary` = for each value q = v >> r one-bits followed by a terminating zero; `low` = the r low bits of every
    value, fixed width. Both as packed uint8 (np.packbits, big-endian bit order). The split makes the decoder
    vectorizable: the k-th zero of the unary plane terminates value k."""
    v = np.asarray(values, dtype=np.int64).reshape(-1)
    if v.size and v.min() < 0:
        raise ValueError("values must be >= 0")
    q = v >> r
    ends = np.cumsum(q + 1) - 1                      # index of each terminating zero
    unary = np.ones(int(ends[-1]) + 1 if v.size else 0, dtype=np.uint8)
    unary[ends] = 0
    low = ((v[:, None] >> np.arange(r - 1, -1, -1)) & 1).astype(np.uint8) if r else np.zeros((v.size, 0), np.uint8)
    return np.packbits(unary), np.packbits(low.reshape(-1))


def rice_decode(unary: np.ndarray, low: np.ndarray, r: int, n: int) -> np.ndarray:
    bits = np.unpackbits(unary)
    ends = np.flatnonzero(bits == 0)[:n]
    if ends.size != n:
        raise ValueError("truncated unary plane")
    q = np.diff(np.concatenate([[-1], ends])) - 1
    lo = np.unpackbits(low)[:n * r].reshape(n, r).astype(np.int64) if r else np.zeros((n, 0), np.int64)
    lv = (lo << np.arange(r - 1, -1, -1)).sum(axis=1) if r else np.zeros(n, np.int64)
    return (q << r) | lv
