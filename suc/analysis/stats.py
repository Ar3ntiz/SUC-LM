"""Per-tensor scalar statistics and per-block statistics (spec §8 T1.1, T1.2).

Everything operates on ONE tensor at a time (D8): the caller streams tensors from the
safetensors store. Heavy math (moments, block reductions, Gram matrices) runs in MLX in
float32; numpy is used only for quantiles/histograms of the resulting float32 arrays.
All returned values are plain Python floats/ints/lists (JSON-serializable).
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

QUANTILES = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
SUMMARY_PCTS = {"p05": 5.0, "p50": 50.0, "p95": 95.0, "p99": 99.0}
DIRECTION_SAMPLE = 4096


def scalar_stats(w: mx.array, hist_bins: int = 64, outlier_sigmas: tuple[float, ...] = (3.0, 6.0)) -> dict:
    """Scalar statistics of a tensor (T1.1), computed in float32.

    kurtosis is the excess (Fisher) kurtosis E[(x-mu)^4]/sigma^4 - 3 (0 for a Gaussian, 3 for a
    Laplace); it is reported as 0.0 for a constant tensor. outlier_rate[str(k)] is the fraction
    of elements with |x - mu| > k*sigma. The histogram is symmetric on [-abs_max, abs_max].
    """
    x = w.astype(mx.float32).reshape(-1)
    mu = mx.mean(x)
    d = x - mu
    var = mx.mean(d * d)
    m4 = mx.mean(d * d * d * d)
    ax = mx.abs(x)
    sd = mx.sqrt(var)
    rates = [mx.mean((mx.abs(d) > k * sd).astype(mx.float32)) for k in outlier_sigmas]
    mx.eval(mu, var, m4, sd, *rates)
    varf = var.item()
    a = np.array(x)
    abs_max = float(np.abs(a).max())
    lim = abs_max if abs_max > 0 else 1.0
    counts, edges = np.histogram(a, bins=hist_bins, range=(-lim, lim))
    qs = np.quantile(a, QUANTILES)
    return {
        "numel": int(x.size),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": mu.item(),
        "std": sd.item(),
        "abs_mean": mx.mean(ax).item(),
        "abs_max": abs_max,
        "quantiles": {f"q{q}": float(v) for q, v in zip(QUANTILES, qs)},
        "kurtosis": m4.item() / (varf * varf) - 3.0 if varf > 0 else 0.0,
        "outlier_rate": {str(float(k)): r.item() for k, r in zip(outlier_sigmas, rates)},
        "histogram": {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]},
    }


def to_blocks(w: mx.array, block_size: int) -> mx.array:
    """(out, in) -> (out*in//G, G): contiguous blocks along the input dim, row-major."""
    if w.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(w.shape)}")
    if w.shape[1] % block_size != 0:
        raise ValueError(f"in_features={w.shape[1]} not divisible by block_size={block_size}")
    return w.reshape(-1, block_size)


def row_rms_normalize(w: mx.array, eps: float = 1e-8) -> tuple[mx.array, mx.array]:
    """Return (w / s_r[:, None], s_r) in float32, s_r = RMS of row r (per-row scale, README §2)."""
    x = w.astype(mx.float32)
    rms = mx.sqrt(mx.mean(x * x, axis=1))
    return x / mx.maximum(rms, eps)[:, None], rms


def _summary(v: mx.array) -> dict:
    a = np.array(v)
    out = {"mean": float(a.mean()), "std": float(a.std())}
    out.update({k: float(p) for k, p in zip(SUMMARY_PCTS, np.percentile(a, list(SUMMARY_PCTS.values())))})
    return out


def isotropic_abs_cosine(g: int) -> float:
    """E|cos| between two independent uniform directions in R^g: Gamma(g/2)/(sqrt(pi) Gamma((g+1)/2))."""
    return math.exp(math.lgamma(g / 2) - math.lgamma((g + 1) / 2)) / math.sqrt(math.pi)


def block_stats(w: mx.array, block_size: int, normalize_rows: bool = True, max_blocks: int | None = None,
                seed: int = 0) -> dict:
    """Per-block statistics (T1.2) summarized over blocks.

    Optionally divides each row by its RMS first (the per-row scale the codebook representation
    factors out), then optionally keeps a deterministic uniform subsample of ``max_blocks`` blocks.
    Per-block quantities: l1, l2, variance (population, over the G elements), max_abs, mean,
    sparsity_exact (fraction == 0) and sparsity_1e-3 (fraction |x| < 1e-3 * abs_max of the
    (normalized) tensor). Each is summarized as {mean, std, p05, p50, p95, p99}.

    direction_concentration: mean |cosine| over all distinct pairs of unit block directions
    b/||b|| on a random sample of up to 4096 non-zero blocks. It is NOT 0 for unstructured
    data: for isotropic random directions it equals E|cos| = direction_concentration_isotropic
    (~sqrt(2/(pi G)), 0.28 at G=8, 0.07 at G=128). Values above that reference indicate shared
    directional structure (what a vector codebook can exploit); 1.0 means all blocks collinear.
    """
    x = row_rms_normalize(w)[0] if normalize_rows else w.astype(mx.float32)
    b = to_blocks(x, block_size)
    n = b.shape[0]
    k_sub, k_dir = mx.random.split(mx.random.key(seed))
    if max_blocks is not None and max_blocks < n:
        b = b[mx.sort(mx.random.permutation(n, key=k_sub)[:max_blocks])]
    ab = mx.abs(b)
    thr = 1e-3 * mx.max(ab)
    mean = mx.mean(b, axis=1)
    per_block = {
        "l1": mx.sum(ab, axis=1),
        "l2": mx.sqrt(mx.sum(b * b, axis=1)),
        "variance": mx.mean(mx.square(b - mean[:, None]), axis=1),
        "max_abs": mx.max(ab, axis=1),
        "mean": mean,
        "sparsity_exact": mx.mean((b == 0).astype(mx.float32), axis=1),
        "sparsity_1e-3": mx.mean((ab < thr).astype(mx.float32), axis=1),
    }
    mx.eval(per_block)
    out: dict = {"block_size": block_size, "num_blocks": n, "num_blocks_used": b.shape[0]}
    out.update({k: _summary(v) for k, v in per_block.items()})
    out["direction_concentration"] = _direction_concentration(b, per_block["l2"], k_dir)
    out["direction_concentration_isotropic"] = isotropic_abs_cosine(block_size)
    return out


def _direction_concentration(b: mx.array, l2: mx.array, key: mx.array) -> float:
    nz = np.flatnonzero(np.array(l2) > 0)
    if nz.size < 2:
        return 0.0
    if nz.size > DIRECTION_SAMPLE:
        nz = nz[np.array(mx.sort(mx.random.permutation(nz.size, key=key)[:DIRECTION_SAMPLE]))]
    idx = mx.array(nz)
    u = b[idx] / l2[idx][:, None]
    m = u.shape[0]
    total = mx.sum(mx.abs(u @ u.T)) - mx.sum(mx.abs(mx.sum(u * u, axis=1)))  # drop the diagonal
    return (total / (m * (m - 1))).item()
