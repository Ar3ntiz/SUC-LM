"""Vector k-means in MLX (float32, GPU), chunked over points so memory stays O(chunk * K).

Distance: d(w, c) = sum_j h_j (w_j - c_j)^2 with optional per-coordinate weights h, either shared
(G,) or per point (N, G) (activation-weighted distance, README §3). With h=None it is plain squared
Euclidean. The weighted centroid minimising the cluster cost is c_j = sum_n h_nj x_nj / sum_n h_nj.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class KMeansResult:
    codebook: mx.array      # (K, G) float32
    assignments: mx.array   # (N,) uint32, nearest code of every point under `codebook`
    inertia: float          # sum of (weighted) squared distances for (codebook, assignments)
    iterations: int         # Lloyd updates performed
    converged: bool


def _check_weights(x: mx.array, weights: mx.array | None) -> mx.array | None:
    if weights is None:
        return None
    w = weights.astype(mx.float32)
    if w.shape not in ((x.shape[1],), tuple(x.shape)):
        raise ValueError(f"weights shape {w.shape} must be (G,) or (N, G) for x {x.shape}")
    return w


def nearest(x: mx.array, codebook: mx.array, weights: mx.array | None = None,
            chunk: int = 65536) -> tuple[mx.array, mx.array]:
    """Nearest code per row of x (N, G): (assignments uint32 (N,), squared distances float32 (N,)).

    argmin uses the expanded form -2 (h*x)·c + h·c^2 (the per-row constant drops out); the returned
    distance is recomputed exactly as sum_j h_j (x_j - c_j)^2 for the chosen code (no cancellation).
    """
    x, c = x.astype(mx.float32), codebook.astype(mx.float32)
    w = _check_weights(x, weights)
    c2 = c * c
    # h·c^2 per code: shared across rows unless weights are per point (then computed per chunk)
    csq = c2.sum(1) if w is None else (c2 @ w if w.ndim == 1 else None)
    out_a, out_d = [], []
    for s in range(0, x.shape[0], chunk):
        xc = x[s:s + chunk]
        wc = None if w is None else (w if w.ndim == 1 else w[s:s + chunk])
        xw = xc if wc is None else xc * wc
        score = -2.0 * (xw @ c.T) + (csq[None] if csq is not None else wc @ c2.T)
        a = mx.argmin(score, axis=1).astype(mx.uint32)
        diff = xc - c[a]
        d = (diff * diff if wc is None else wc * diff * diff).sum(1)
        mx.eval(a, d)
        out_a.append(a)
        out_d.append(d)
    if not out_a:
        return mx.zeros((0,), mx.uint32), mx.zeros((0,), mx.float32)
    return mx.concatenate(out_a), mx.concatenate(out_d)


def _point_dist(x: mx.array, c: mx.array, w: mx.array | None) -> mx.array:
    diff = x - c[None]
    return (diff * diff if w is None else w * diff * diff).sum(1)


def _kmeanspp(s: mx.array, ws: mx.array | None, k: int, key: mx.array) -> mx.array:
    """D^2 sampling on the subsample s; falls back to uniform when all distances are zero."""
    keys = mx.random.split(key, k)
    first = mx.random.randint(0, s.shape[0], (), key=keys[0])
    centers = [s[first]]
    dmin = _point_dist(s, centers[0], ws)
    mx.eval(dmin)
    for i in range(1, k):
        p = mx.where(dmin.sum() > 0, dmin, mx.ones_like(dmin))
        idx = mx.random.categorical(mx.log(p), key=keys[i])
        centers.append(s[idx])
        dmin = mx.minimum(dmin, _point_dist(s, centers[-1], ws))
        mx.eval(dmin)
    return mx.stack(centers)


def _update(x: mx.array, w: mx.array | None, a: mx.array, d: mx.array, c: mx.array,
            chunk_elems: int = 1 << 24) -> mx.array:
    """Weighted centroid step; empty clusters get the points with the largest current distance.

    Sums use chunked one-hot GEMMs (onehot^T @ x) accumulated in a fixed order: unlike a float
    scatter-add (GPU atomics) this is bit-reproducible, which the seed-determinism contract needs.
    """
    k, g = c.shape
    per_point = w is not None and w.ndim == 2  # a shared (G,) weight does not change the minimiser
    codes = mx.arange(k, dtype=mx.uint32)
    step = max(1, chunk_elems // k)
    num = mx.zeros((k, g), mx.float32)
    den = mx.zeros((k, g if per_point else 1), mx.float32)
    for s in range(0, x.shape[0], step):
        oh = (a[s:s + step, None] == codes[None]).astype(mx.float32)
        xc = x[s:s + step]
        if per_point:
            wc = w[s:s + step]
            num, den = num + oh.T @ (xc * wc), den + oh.T @ wc
        else:
            num, den = num + oh.T @ xc, den + oh.sum(0)[:, None]
        mx.eval(num, den)
    occupied = (den > 0).any(1) if per_point else den[:, 0] > 0
    new = mx.where(den > 0, num / mx.maximum(den, 1e-30), c)
    empty = np.flatnonzero(~np.array(occupied))
    if empty.size:
        far = mx.argsort(-d)[: empty.size]  # deterministic: stable sort of distances
        new[mx.array(empty.astype(np.int32))] = x[far]
    mx.eval(new)
    return new


def kmeans(x: mx.array, k: int, iters: int = 25, seed: int = 0,
           init: Literal["kmeans++", "random"] = "kmeans++", weights: mx.array | None = None,
           tol: float = 1e-4, sample_init: int = 65536) -> KMeansResult:
    """Lloyd k-means on x (N, G). Deterministic given seed; converged when the relative inertia
    change between consecutive iterations is < tol."""
    x = x.astype(mx.float32)
    n = x.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= N, got k={k}, N={n}")
    w = _check_weights(x, weights)
    k_perm, k_init = mx.random.split(mx.random.key(seed))
    if init == "random":
        c = x[mx.random.permutation(n, key=k_perm)[:k]]
    elif init == "kmeans++":
        idx = mx.random.permutation(n, key=k_perm)[: min(n, sample_init)]
        c = _kmeanspp(x[idx], None if w is None else w if w.ndim == 1 else w[idx], k, k_init)
    else:
        raise ValueError(f"unknown init: {init!r}")
    mx.eval(c)
    a, d = nearest(x, c, w)
    inertia, it, converged = float(d.sum()), 0, False
    while it < iters:
        it += 1
        c = _update(x, w, a, d, c)
        a, d = nearest(x, c, w)
        cur = float(d.sum())
        done = abs(inertia - cur) <= tol * max(inertia, 1e-30)
        inertia = cur
        if done:
            converged = True
            break
    return KMeansResult(codebook=c, assignments=a, inertia=inertia, iterations=it, converged=converged)
