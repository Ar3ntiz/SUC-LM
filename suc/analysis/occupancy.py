"""Codebook occupancy and fit diagnostics for Phase 1 (spec §8 T1.3/T1.4, D4 ceiling)."""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from suc.quant.kmeans import kmeans, nearest

CODEBOOK_CEILING_BYTES = 16384  # D4: K * G * 2 bytes (FP16) per codebook resident in a threadgroup


def occupancy_metrics(assignments: mx.array, k: int) -> dict:
    """Usage statistics of k codes. gini is over all k codes (unused codes count as zero usage):
    0 = uniform, 1 - 1/k = all mass on one code. top10pct_share uses the ceil(0.1 k) most-used codes."""
    counts = np.bincount(np.array(assignments).astype(np.int64), minlength=k)[:k].astype(np.float64)
    total = counts.sum()
    p = counts / total if total > 0 else counts
    nz = p[p > 0]
    h = float(-(nz * np.log2(nz)).sum()) if nz.size else 0.0
    srt = np.sort(p)  # ascending
    gini = float(((2 * np.arange(1, k + 1) - k - 1) * srt).sum() / k) if total > 0 else 0.0
    occupied = int((counts > 0).sum())
    return {
        "k": k,
        "occupied": occupied,
        "occupied_frac": occupied / k,
        "entropy_bits": h,
        "entropy_frac": h / math.log2(k) if k > 1 else 0.0,
        "top1_share": float(srt[-1]) if k else 0.0,
        "top10pct_share": float(srt[::-1][: max(1, math.ceil(0.1 * k))].sum()),
        "gini": gini,
    }


def codebook_fit_report(blocks: mx.array, k: int, seed: int = 0, iters: int = 25, holdout_frac: float = 0.2,
                        max_fit_blocks: int | None = 262144) -> dict:
    """Fit a K-entry vector codebook on blocks (N, G) and measure it on held-out blocks.

    Split: a seeded permutation; the first round(N * holdout_frac) blocks are held out, the rest form
    the fit part, truncated to max_fit_blocks (a random subsample, since the order is permuted).
    rel_mse_* = sum ||b - C[i]||^2 / sum ||b||^2 over that part.
    dist_* are held-out squared distances ||b - C[i]||^2 divided by the held-out mean ||b||^2
    (so dist_mean == rel_mse_heldout; p95/p99 describe the tail on the same scale).
    """
    b = blocks.astype(mx.float32)
    n, g = b.shape
    n_held = int(round(n * holdout_frac))
    if n_held < 1:
        raise ValueError("holdout_frac leaves no held-out blocks")
    perm = mx.random.permutation(n, key=mx.random.key(seed + 1_000_003))
    held, fit_idx = b[perm[:n_held]], perm[n_held:]
    if max_fit_blocks is not None:
        fit_idx = fit_idx[:max_fit_blocks]
    fit = b[fit_idx]
    res = kmeans(fit, k, iters=iters, seed=seed)
    a_h, d_h = nearest(held, res.codebook)
    fit_energy = float((fit * fit).sum())
    held_sq = (held * held).sum(1)
    held_mean_sq = float(held_sq.mean())
    d = np.array(d_h, dtype=np.float64) / held_mean_sq
    cb_bytes = k * g * 2
    return {
        "k": k,
        "block_size": g,
        "num_blocks": n,
        "fit_blocks": int(fit.shape[0]),
        "heldout_blocks": n_held,
        "nominal_index_bpw": math.log2(k) / g,
        "codebook_bytes_fp16": cb_bytes,
        "execution_viable": cb_bytes <= CODEBOOK_CEILING_BYTES,
        "rel_mse_fit": res.inertia / fit_energy,
        "rel_mse_heldout": float(d_h.sum()) / float(held_sq.sum()),
        "dist_mean": float(d.mean()),
        "dist_p95": float(np.quantile(d, 0.95)),
        "dist_p99": float(np.quantile(d, 0.99)),
        "occupancy": occupancy_metrics(a_h, k),
        "iterations": res.iterations,
        "converged": res.converged,
    }
