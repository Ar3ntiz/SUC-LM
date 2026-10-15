"""Research-layout weight codecs of the quality track (spec §10-§12, README §2).

Each codec encodes ONE 2-D weight W (out, in) into an `Encoded` (plain arrays + JSON-able params)
and decodes it back to float32. `Encoded.bits()` is the ACTUAL storage (spec §25), never nominal:

  indices   ceil(log2 K) bits per stored index (fixed-width execution layout; raw 16 bits for bf16)
  codebooks every codebook entry at its stored dtype (fp16)
  scales    row scales / affine group scales+biases at their stored dtype (fp16)
  residual  0 here (Phase 6)
  metadata  a constant 64 bytes per tensor (descriptor: shape, codec, seeds, offsets)

Codecs:
  BF16Codec        identity baseline.
  AffineRTN        MLX affine round-to-nearest (per-group asymmetric min/max, fp16 scale+bias per group
                   along the input dim); encode calls mx.quantize so values and storage are MLX's exactly.
  ScalarCodebook   Q1: row-RMS normalisation + 1-D k-means codebook per tensor or per row-group.
  VectorCodebook   Q2/Q3: optional randomized Hadamard rotation + row-RMS normalisation + vector k-means.
  AdditiveCodebook Q4: sum of M vector codebooks, greedy residual fit + optional joint refinement.
  MXFP4Native      checkpoint MXFP4 codes + E8M0 scales stored verbatim (quantized-at-source experts, 4.25 bpw).

Codebooks are rounded to fp16 BEFORE assignment, so the stored indices are optimal for the stored
codebook and decode(encode(w)) is exactly what a kernel reading the fp16 codebook would compute.

Shared codebooks (vq / additive, suc.quant.stacks): sample_blocks(w, ...) returns a subsample of the prepared
(rotated, row-normalised) blocks of one tensor, fit(x, h, seed=) fits the (M, K, G) codebooks on pooled samples of
several tensors, and encode(w, ..., codebooks=c, codebook_owner=b) only assigns indices (+ row scales) against the
given codebooks; params then record codebook_shared / codebook_owner and bits() counts the codebooks only for the
owner. Without codebooks= encode is unchanged bit for bit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol

import mlx.core as mx
import numpy as np

from suc.analysis.stats import to_blocks
from suc.quant.hadamard import rht, rht_inverse, rotate_weight, supported_size, unrotate_weight
from suc.quant.kmeans import _update, kmeans, nearest

METADATA_BITS = 512  # 64 bytes per tensor descriptor
_SCALE_KEYS = ("row_scales", "scales", "biases")


@dataclass
class Encoded:
    codec: str
    shape: tuple[int, int]
    params: dict
    arrays: dict[str, mx.array] = field(default_factory=dict)

    def bits(self) -> dict:
        a, p = self.arrays, self.params
        if "weight" in a:  # raw payload (bf16 identity)
            idx = a["weight"].size * a["weight"].itemsize * 8
        elif "codes" in a:  # native MXFP4: packed E2M1 bytes, two codes per byte
            idx = a["codes"].size * a["codes"].itemsize * 8
        elif "stream_bits" in p:  # trellis (suc.quant.trellis): the packed tail-biting bitstream, not the containers
            idx = p["stream_bits"]
        else:
            idx = p["index_bits"] * a["indices"].size
        # a codebook shared by an expert stack (encode(..., codebooks=, codebook_owner=)) is counted once: only
        # by its owner (expert 0); the other experts store a reference to it (suc.quant.stacks)
        cb = a["codebooks"].size * a["codebooks"].itemsize * 8 \
            if "codebooks" in a and p.get("codebook_owner", True) else 0
        if "lut" in a:    # trellis hyb lookup table
            cb += a["lut"].size * a["lut"].itemsize * 8
        sc = sum(a[k].size * a[k].itemsize * 8 for k in _SCALE_KEYS if k in a)
        out = {"indices": int(idx), "codebooks": int(cb), "scales": int(sc), "residual": 0,
               "metadata": METADATA_BITS}
        out["total"] = sum(out.values())
        return out

    def bpw(self) -> float:
        return self.bits()["total"] / (self.shape[0] * self.shape[1])


class Codec(Protocol):
    name: str

    def encode(self, w: mx.array, *, seed: int, col_weights: mx.array | None = None) -> Encoded: ...

    def decode(self, e: Encoded) -> mx.array: ...


def codebook_bytes(e: Encoded) -> int:
    """Bytes of ONE codebook (K * G * 2 for fp16), the quantity bounded by the D4 ceiling.
    An additive tile needs params["M"] of them resident at once. 0 for codebook-free codecs."""
    if "codebooks" not in e.arrays:
        return 0
    cb = e.arrays["codebooks"]
    return int(cb[0].size * cb.itemsize)


# ---------------------------------------------------------------------------------------- helpers

def _check_2d(w: mx.array) -> tuple[int, int]:
    if w.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(w.shape)}")
    return int(w.shape[0]), int(w.shape[1])


def _check_decode(e: Encoded, name: str) -> None:
    if e.codec != name:
        raise ValueError(f"cannot decode {e.codec!r} with codec {name!r}")


def _index_bits(k: int) -> int:
    return max(1, math.ceil(math.log2(k)))


def _index_dtype(k: int) -> mx.Dtype:
    return mx.uint8 if k <= 256 else mx.uint16 if k <= 65536 else mx.uint32


def _row_normalize(x: mx.array) -> tuple[mx.array, mx.array]:
    """x / s_r with s_r the row RMS rounded to fp16 (the stored scale); zero rows stay zero."""
    s16 = mx.sqrt(mx.mean(x * x, axis=1)).astype(mx.float16)
    s = s16.astype(mx.float32)
    return x / mx.where(s > 0, s, 1.0)[:, None], s16


def _subsample(n: int, m: int, seed: int) -> mx.array | None:
    """Sorted deterministic uniform subsample of m of n indices (None = use all)."""
    if m >= n:
        return None
    return mx.sort(mx.random.permutation(n, key=mx.random.key(seed))[:m])


def _derive_seeds(seed: int, n: int) -> list[int]:
    return [int(s) for s in np.random.default_rng(seed).integers(0, 2 ** 31 - 1, size=n)]


def _fit(x: mx.array, h: mx.array | None, k: int, iters: int, seed: int, fit_n: int) -> mx.array:
    """k-means on a subsample of x (N, G), weights h (N, G) or None; returns the fp16-rounded codebook
    as float32 (K, G)."""
    sub = _subsample(x.shape[0], fit_n, seed)
    xs, hs = (x, h) if sub is None else (x[sub], None if h is None else h[sub])
    if k > xs.shape[0]:
        raise ValueError(f"k={k} exceeds the number of fitting points {xs.shape[0]}")
    return kmeans(xs, k, iters=iters, seed=seed, weights=hs).codebook.astype(mx.float16).astype(mx.float32)


# ----------------------------------------------------------------------------------------- codecs

class BF16Codec:
    name = "bf16"

    def encode(self, w: mx.array, *, seed: int, col_weights: mx.array | None = None) -> Encoded:
        shape = _check_2d(w)
        return Encoded(self.name, shape, {}, {"weight": w.astype(mx.bfloat16)})

    def decode(self, e: Encoded) -> mx.array:
        _check_decode(e, self.name)
        return e.arrays["weight"].astype(mx.float32)


class AffineRTN:
    """MLX affine RTN: bpw = bits + 32/group_size (+ metadata).

    Encoding calls mx.quantize on the fp16 cast of w (bf16 -> fp16 is exact for normal values) so
    codes, scales and biases are bit-identical to MLX; an independent float32 re-implementation of
    the same formulas disagrees with the Metal kernel on ~0.02% of elements at exact rounding ties.
    The packed codes are unpacked into one uint8 code per weight (research layout)."""
    name = "affine"

    def __init__(self, bits: int, group_size: int = 64) -> None:
        if bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"MLX affine supports bits in 2,3,4,5,6,8, got {bits}")
        if group_size not in (32, 64, 128):
            raise ValueError(f"MLX affine supports group_size in 32,64,128, got {group_size}")
        self.bits, self.group_size = bits, group_size

    def encode(self, w: mx.array, *, seed: int, col_weights: mx.array | None = None) -> Encoded:
        out, inn = _check_2d(w)
        if inn % self.group_size:
            raise ValueError(f"in_features={inn} not divisible by group_size={self.group_size}")
        wq, s, b = mx.quantize(w.astype(mx.float16), group_size=self.group_size, bits=self.bits, mode="affine")
        codes = _unpack_bitstream(np.array(wq), self.bits, inn)
        params = {"bits": self.bits, "group_size": self.group_size, "index_bits": self.bits}
        return Encoded(self.name, (out, inn), params, {"indices": mx.array(codes), "scales": s, "biases": b})

    def decode(self, e: Encoded) -> mx.array:
        _check_decode(e, self.name)
        out, inn = e.shape
        gs = e.params["group_size"]
        q = e.arrays["indices"].astype(mx.float32).reshape(out, inn // gs, gs)
        s, b = (e.arrays[k].astype(mx.float32)[..., None] for k in ("scales", "biases"))
        return (q * s + b).reshape(out, inn)


def _unpack_bitstream(packed: np.ndarray, bits: int, n: int, rows_per_chunk: int = 4096) -> np.ndarray:
    """MLX packs each row as a little-endian bit stream of `bits`-bit codes; returns (out, n) uint8."""
    out = np.empty((packed.shape[0], n), np.uint8)
    weights = (1 << np.arange(bits)).astype(np.uint8)
    for r in range(0, packed.shape[0], rows_per_chunk):
        by = np.ascontiguousarray(packed[r:r + rows_per_chunk]).view(np.uint8)
        bitsarr = np.unpackbits(by, axis=1, bitorder="little")[:, : n * bits]
        out[r:r + rows_per_chunk] = (bitsarr.reshape(by.shape[0], n, bits) * weights).sum(-1, dtype=np.uint8)
    return out


class ScalarCodebook:
    """Q1: W ~ s_r * C_g[i] per weight; one 1-D codebook per tensor or per group of `row_group` rows.

    col_weights (in,) make the k-means distance activation-weighted: every weight in column j counts
    with weight h_j (README §3). k-means is fitted on <= fit_points weights per codebook."""
    name = "scalar"

    def __init__(self, k: int, scope: Literal["tensor", "row_group"] = "tensor", row_group: int = 64,
                 kmeans_iters: int = 25, fit_points: int = 1 << 20) -> None:
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        if scope not in ("tensor", "row_group"):
            raise ValueError(f"unknown scope {scope!r}")
        self.k, self.scope, self.row_group = k, scope, row_group
        self.kmeans_iters, self.fit_points = kmeans_iters, fit_points

    def encode(self, w: mx.array, *, seed: int, col_weights: mx.array | None = None) -> Encoded:
        out, inn = _check_2d(w)
        xn, s16 = _row_normalize(w.astype(mx.float32))
        rg = out if self.scope == "tensor" else self.row_group
        cbs, idx = [], []
        for g, r0 in enumerate(range(0, out, rg)):
            x = xn[r0:r0 + rg].reshape(-1, 1)
            h = None if col_weights is None else mx.broadcast_to(
                col_weights.astype(mx.float32)[None], (min(rg, out - r0), inn)).reshape(-1, 1)
            c = _fit(x, h, self.k, self.kmeans_iters, seed + g, self.fit_points)
            a, _ = nearest(x, c, h)
            cbs.append(c)
            idx.append(a.reshape(-1, inn))
        params = {"k": self.k, "scope": self.scope, "row_group": rg, "index_bits": _index_bits(self.k), "seed": seed}
        arrays = {"codebooks": mx.stack(cbs).astype(mx.float16),                     # (n_groups, K, 1)
                  "indices": mx.concatenate(idx).astype(_index_dtype(self.k)),         # (out, in)
                  "row_scales": s16}
        return Encoded(self.name, (out, inn), params, arrays)

    def decode(self, e: Encoded) -> mx.array:
        _check_decode(e, self.name)
        out, inn = e.shape
        cb = e.arrays["codebooks"].astype(mx.float32)[..., 0]                        # (n_groups, K)
        grp = mx.arange(out) // e.params["row_group"]
        flat = grp[:, None] * cb.shape[1] + e.arrays["indices"].astype(mx.int32)
        return cb.reshape(-1)[flat] * e.arrays["row_scales"].astype(mx.float32)[:, None]


class _BlockCodebookBase:
    """Shared rotate -> row-normalise -> blocks pipeline of the vector and additive codecs.

    col_weights (in,) become per-coordinate weights of the block distance when rotate=False. When
    rotate=True the error metric diag(h) in the input space becomes V diag(h) V^T in the rotated
    space, whose diagonal is the constant mean(h) for a Hadamard V (|V_ij|^2 = 1/n): the diagonal
    weighting is uniform and col_weights are ignored (a non-diagonal metric is out of scope here).

    Rotation seeds: seed_in / seed_out default to _derive_seeds(seed, 2); encode(..., seed_in=, seed_out=) overrides
    either one (e.g. q/k/v of a block sharing V so the runtime rotates their common input once). Both are stored
    in params, so decoding never re-derives them. Ignored when rotate=False (and seed_out when the output size
    has no Hadamard).

    Sides: rotate_in=False drops V (the input is already incoherent, e.g. a residual-stream rotation folded into the
    weights); rotate_out=False drops U (one-sided, input rotation only). params["rotate_in"] / ["rotate_out"] record
    what was applied and the unused seed is None; col_weights apply whenever V is not applied."""
    name = ""

    def __init__(self, k: int, block_size: int, rotate: bool, kmeans_iters: int, fit_blocks: int,
                 rotate_in: bool = True, rotate_out: bool = True) -> None:
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        self.k, self.block_size, self.rotate = k, block_size, rotate
        self.rotate_in, self.rotate_out = bool(rotate_in), bool(rotate_out)
        self.kmeans_iters, self.fit_blocks = kmeans_iters, fit_blocks

    def _prepare(self, w: mx.array, seed: int, col_weights: mx.array | None, seed_in: int | None = None,
                 seed_out: int | None = None) -> tuple[tuple[int, int], dict, mx.array, mx.array, mx.array | None]:
        out, inn = _check_2d(w)
        g = self.block_size
        if inn % g:
            raise ValueError(f"in_features={inn} not divisible by block_size={g}")
        x = w.astype(mx.float32)
        rin = self.rotate and self.rotate_in
        rout = self.rotate and self.rotate_out and supported_size(out)
        params = {"k": self.k, "block_size": g, "rotate": bool(rin or rout), "index_bits": _index_bits(self.k),
                  "seed": seed}
        if rin or rout:
            # V (input side) is required when rotate_in; U (output side) is skipped when rotate_out=False (one-sided)
            # or when the output size has no Hadamard — e.g. embeddings / LM heads with a vocabulary of 151936, where
            # rotating across tokens would not be usable at inference anyway. W' = W Vᵀ is then rht() on every row.
            # rotate_in=False: the input is already incoherent (e.g. the rotated residual stream of a folded model,
            # suc.models.rotation_folding), so no online input rotation is needed: W' = U W.
            if rin and not supported_size(inn):
                raise ValueError(f"Hadamard rotation unsupported for input size {inn} (shape {(out, inn)})")
            d_in, d_out = _derive_seeds(seed, 2)
            # explicit overrides (rotation groups: tensors reading the same input share V, README §5)
            params["seed_in"] = (d_in if seed_in is None else int(seed_in)) if rin else None
            params["seed_out"] = (d_out if seed_out is None else int(seed_out)) if rout else None
            params["rotate_in"], params["rotate_out"] = bool(rin), bool(rout)
            if rin and rout:
                x = rotate_weight(x, params["seed_in"], params["seed_out"])
            elif rin:
                x = rht(x, params["seed_in"])
            else:
                x = rht(x.T, params["seed_out"]).T
        xn, s16 = _row_normalize(x)
        blocks = to_blocks(xn, g)
        h = None
        if col_weights is not None and not rin:   # an output-side rotation keeps the per-column metric diagonal
            cw = col_weights.astype(mx.float32).reshape(inn // g, g)
            h = mx.broadcast_to(cw[None], (out, inn // g, g)).reshape(-1, g)
        return (out, inn), params, blocks, s16, h

    def sample_blocks(self, w: mx.array, *, seed: int, n: int, col_weights: mx.array | None = None,
                      seed_in: int | None = None, seed_out: int | None = None) -> tuple[mx.array, mx.array | None]:
        """Deterministic subsample of n prepared blocks of w (rotated and row-normalised exactly as encode with the
        same seeds would) and their distance weights (None without col_weights): the fitting sample of w."""
        _, _, blocks, _, h = self._prepare(w, seed, col_weights, seed_in, seed_out)
        sub = _subsample(blocks.shape[0], n, seed)
        if sub is None:
            return blocks, h
        return blocks[sub], None if h is None else h[sub]

    def _shared(self, codebooks: mx.array, m: int) -> mx.array:
        c = mx.array(codebooks).astype(mx.float16).astype(mx.float32)
        if tuple(c.shape) != (m, self.k, self.block_size):
            raise ValueError(f"codebooks must have shape {(m, self.k, self.block_size)}, got {tuple(c.shape)}")
        return c

    @staticmethod
    def _mark_shared(params: dict, owner: bool) -> None:
        params["codebook_shared"], params["codebook_owner"] = True, bool(owner)

    def _finish(self, e: Encoded, blocks_hat: mx.array) -> mx.array:
        out, inn = e.shape
        x = blocks_hat.reshape(out, inn) * e.arrays["row_scales"].astype(mx.float32)[:, None]
        p = e.params
        if not p["rotate"]:
            return x
        rin, rout = p.get("rotate_in", True), p.get("rotate_out", True)
        if rin and rout:
            return unrotate_weight(x, p["seed_in"], p["seed_out"])
        if rin:
            return rht_inverse(x, p["seed_in"])
        return rht_inverse(x.T, p["seed_out"]).T

    def _sum_codes(self, e: Encoded) -> mx.array:
        cb = e.arrays["codebooks"].astype(mx.float32)                                # (M, K, G)
        idx = e.arrays["indices"].astype(mx.int32)                                   # (M, out, in//G)
        return sum(cb[m][idx[m].reshape(-1)] for m in range(cb.shape[0]))


class VectorCodebook(_BlockCodebookBase):
    """Q2/Q3: w_b ~ s_r * C[i_b] (in the rotated basis when rotate=True). M = 1 additive special case."""
    name = "vq"

    def __init__(self, k: int, block_size: int = 8, rotate: bool = True, kmeans_iters: int = 25,
                 fit_blocks: int = 262144, rotate_in: bool = True, rotate_out: bool = True) -> None:
        super().__init__(k, block_size, rotate, kmeans_iters, fit_blocks, rotate_in, rotate_out)

    def fit(self, x: mx.array, h: mx.array | None = None, *, seed: int) -> mx.array:
        """(1, K, G) fp16-rounded codebook (float32) fitted on prepared blocks x (N, G) (<= fit_blocks used)."""
        return _fit(x, h, self.k, self.kmeans_iters, seed, self.fit_blocks)[None]

    def encode(self, w: mx.array, *, seed: int, col_weights: mx.array | None = None, seed_in: int | None = None,
               seed_out: int | None = None, codebooks: mx.array | None = None, codebook_owner: bool = True) -> Encoded:
        shape, params, blocks, s16, h = self._prepare(w, seed, col_weights, seed_in, seed_out)
        if codebooks is None:
            c = _fit(blocks, h, self.k, self.kmeans_iters, seed, self.fit_blocks)
        else:
            c = self._shared(codebooks, 1)[0]
            self._mark_shared(params, codebook_owner)
        a, _ = nearest(blocks, c, h)
        arrays = {"codebooks": c[None].astype(mx.float16),
                  "indices": a.reshape(1, shape[0], -1).astype(_index_dtype(self.k)), "row_scales": s16}
        return Encoded(self.name, shape, params, arrays)

    def decode(self, e: Encoded) -> mx.array:
        _check_decode(e, self.name)
        return self._finish(e, self._sum_codes(e))


class AdditiveCodebook(_BlockCodebookBase):
    """Q4: w_b ~ s_r * sum_m C_m[i_{b,m}].

    Greedy: codebook m is k-means-fitted (on <= fit_blocks blocks) to the residual left by codebooks
    1..m-1, then every block is assigned. Joint refinement (refine_iters rounds): for each m, target =
    blocks - sum_{m' != m} C_m'[i_m']; one weighted centroid update of C_m on ALL blocks with the current
    assignments, fp16 rounding, re-assignment. Each step is non-increasing in the total (weighted)
    squared error up to fp16 rounding of the codebook."""
    name = "additive"

    def __init__(self, k: int, block_size: int = 8, num_codebooks: int = 2, rotate: bool = True,
                 refine_iters: int = 0, kmeans_iters: int = 25, fit_blocks: int = 262144, rotate_in: bool = True,
                 rotate_out: bool = True) -> None:
        super().__init__(k, block_size, rotate, kmeans_iters, fit_blocks, rotate_in, rotate_out)
        if num_codebooks < 1:
            raise ValueError(f"num_codebooks must be >= 1, got {num_codebooks}")
        self.num_codebooks, self.refine_iters = num_codebooks, refine_iters

    def _greedy_fit(self, blocks: mx.array, h: mx.array | None, seed: int) -> tuple[list, list, mx.array]:
        """Greedy residual fit + joint refinement (class docstring) on `blocks`: (codebooks, assignments, residual)."""
        seeds = _derive_seeds(seed + 1, self.num_codebooks)
        cbs, assign, resid = [], [], blocks
        for m in range(self.num_codebooks):
            c = _fit(resid, h, self.k, self.kmeans_iters, seeds[m], self.fit_blocks)
            a, _ = nearest(resid, c, h)
            resid = resid - c[a]
            mx.eval(resid)
            cbs.append(c)
            assign.append(a)
        for _ in range(self.refine_iters):
            for m in range(self.num_codebooks):
                target = resid + cbs[m][assign[m]]
                d = _dist(target, cbs[m][assign[m]], h)
                c = _update(target, h, assign[m], d, cbs[m]).astype(mx.float16).astype(mx.float32)
                a, _ = nearest(target, c, h)
                resid = target - c[a]
                mx.eval(resid)
                cbs[m], assign[m] = c, a
        return cbs, assign, resid

    def fit(self, x: mx.array, h: mx.array | None = None, *, seed: int) -> mx.array:
        """(M, K, G) fp16-rounded codebooks (float32): the greedy fit + joint refinement of encode, run on the
        prepared blocks x (N, G) only (e.g. pooled samples of every expert of a stack)."""
        return mx.stack(self._greedy_fit(x.astype(mx.float32), h, seed)[0])

    def _assign_fixed(self, blocks: mx.array, h: mx.array | None, cbs: mx.array) -> list:
        """Assignments against FIXED codebooks: greedy (codebook m on the residual of 1..m-1), then refine_iters
        coordinate-descent passes that re-assign each m given the others (each pass is non-increasing)."""
        assign, resid = [], blocks
        for m in range(self.num_codebooks):
            a, _ = nearest(resid, cbs[m], h)
            resid = resid - cbs[m][a]
            mx.eval(resid)
            assign.append(a)
        for _ in range(self.refine_iters):
            for m in range(self.num_codebooks):
                target = resid + cbs[m][assign[m]]
                a, _ = nearest(target, cbs[m], h)
                resid = target - cbs[m][a]
                mx.eval(resid)
                assign[m] = a
        return assign

    def encode(self, w: mx.array, *, seed: int, col_weights: mx.array | None = None, seed_in: int | None = None,
               seed_out: int | None = None, codebooks: mx.array | None = None, codebook_owner: bool = True) -> Encoded:
        shape, params, blocks, s16, h = self._prepare(w, seed, col_weights, seed_in, seed_out)
        if codebooks is None:
            cbs, assign, _ = self._greedy_fit(blocks, h, seed)
        else:
            c = self._shared(codebooks, self.num_codebooks)
            cbs = [c[m] for m in range(self.num_codebooks)]
            assign = self._assign_fixed(blocks, h, cbs)
            self._mark_shared(params, codebook_owner)
        params.update(num_codebooks=self.num_codebooks, M=self.num_codebooks, refine_iters=self.refine_iters)
        arrays = {"codebooks": mx.stack(cbs).astype(mx.float16),
                  "indices": mx.stack(assign).reshape(self.num_codebooks, shape[0], -1).astype(_index_dtype(self.k)),
                  "row_scales": s16}
        return Encoded(self.name, shape, params, arrays)

    def decode(self, e: Encoded) -> mx.array:
        _check_decode(e, self.name)
        return self._finish(e, self._sum_codes(e))


def _dist(x: mx.array, c: mx.array, h: mx.array | None) -> mx.array:
    d = (x - c) ** 2
    return (d if h is None else h * d).sum(1)


# ------------------------------------------------------------------------------ native MXFP4 pass-through

MXFP4_GROUP = 32
_E2M1_MAGS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# E2M1 code -> value, code 8 (negative zero) decodes to +0.0 as in the source dequantization (suc.io.quantized_source)
_E2M1 = np.concatenate([_E2M1_MAGS, -_E2M1_MAGS]).astype(np.float32)
_E2M1[8] = 0.0


class MXFP4Native:
    """Native MXFP4 pass-through for weights that are MXFP4 at source (DeepSeek-V4 routed experts): the checkpoint's
    packed E2M1 codes (uint8 [out, in/2], element 2c in the low nibble) and E8M0 group scales (uint8 [out, in/32], one
    per 32 inputs of a row) are stored VERBATIM (encode_source), so the representation is bit-exact vs the source and
    its error against the D8 reference (the exact BF16 dequantization) is zero; 4 + 8/32 = 4.25 bpw + metadata.

    W[i, k] = E2M1(code(i, k)) * 2^(S[i, k // 32] - 127)

    encode(w) re-derives codes from a weight that is EXACTLY representable in MXFP4 (smallest scale per group, +0 for
    zeros; ValueError otherwise), so a re-quantization equals the source in value but not necessarily in bytes (a group
    may have used a larger scale, -0 codes): the pipeline uses encode_source on the checkpoint bytes."""
    name = "mxfp4"
    lossless = True

    def __init__(self, group_size: int = MXFP4_GROUP) -> None:
        if group_size != MXFP4_GROUP:
            raise ValueError(f"MXFP4 uses groups of {MXFP4_GROUP} inputs, got {group_size}")
        self.group_size = group_size

    @staticmethod
    def bits_for(shape) -> int:
        """Encoded.bits()["total"] of a native encoding of `shape` (out, in): codes + scales + metadata."""
        out, inn = int(shape[0]), int(shape[1])
        return out * inn * 4 + out * (inn // MXFP4_GROUP) * 8 + METADATA_BITS

    def _params(self, source: str) -> dict:
        return {"format": "mxfp4", "group_size": self.group_size, "index_bits": 4, "element": "e2m1",
                "scale": "e8m0", "source": source}

    def encode_source(self, packed: mx.array, scale: mx.array) -> Encoded:
        """Verbatim checkpoint bytes: packed int8 / uint8 [out, in/2] + E8M0 uint8 [out, in/32]."""
        from suc.io.quantized_source import check_scales

        out, half = int(packed.shape[0]), int(packed.shape[1])
        inn = 2 * half
        if tuple(scale.shape) != (out, inn // self.group_size) or inn % self.group_size:
            raise ValueError(f"mxfp4 scale shape {tuple(scale.shape)} does not match codes {tuple(packed.shape)}")
        check_scales("mxfp4", scale)          # the BF16 reference of these bytes is exact (NotLossless otherwise)
        codes = packed.view(mx.uint8) if packed.dtype == mx.int8 else packed.astype(mx.uint8)
        return Encoded(self.name, (out, inn), self._params("verbatim"), {"codes": codes, "scales": scale.astype(mx.uint8)})

    def encode(self, w: mx.array, *, seed: int, col_weights: mx.array | None = None) -> Encoded:
        out, inn = _check_2d(w)
        g = self.group_size
        if inn % g:
            raise ValueError(f"in_features={inn} not divisible by the MXFP4 group size {g}")
        x = np.array(w.astype(mx.float32)).astype(np.float64).reshape(out, inn // g, g)
        amax = np.abs(x).max(-1)
        e = np.ceil(np.log2(np.maximum(amax, 1e-300) / 6.0)).astype(np.int64)
        e = np.where(amax > 6.0 * np.ldexp(1.0, e), e + 1, e)                 # exact: smallest e with amax <= 6 * 2^e
        e = np.where(amax <= 6.0 * np.ldexp(1.0, e - 1), e - 1, e)
        e = np.where(amax > 0, e, 0)
        s = e + 127
        if s.min() < 1 or s.max() > 254:
            raise ValueError("weight outside the E8M0 scale range: not representable in MXFP4")
        y = x / np.ldexp(1.0, e)[..., None]
        mag = np.abs(y)
        k = np.clip(np.searchsorted(_E2M1_MAGS, mag), 0, 7)
        if not np.array_equal(_E2M1_MAGS[k], mag):
            raise ValueError("weight is not exactly representable in MXFP4 (use encode_source on the checkpoint bytes)")
        code = (k + 8 * ((y < 0) & (k > 0))).astype(np.uint8).reshape(out, inn)
        packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
        return Encoded(self.name, (out, inn), self._params("requantized"),
                       {"codes": mx.array(packed), "scales": mx.array(s.astype(np.uint8))})

    def decode(self, e: Encoded) -> mx.array:
        _check_decode(e, self.name)
        out, inn = e.shape
        b = e.arrays["codes"]
        codes = mx.stack([b & 0x0F, b >> 4], axis=-1).reshape(out, inn).astype(mx.uint32)
        vals = mx.take(mx.array(_E2M1), codes)
        s = (e.arrays["scales"].astype(mx.uint32) << 23).view(mx.float32)          # 2^(S - 127), exact
        return (vals.reshape(out, inn // self.group_size, self.group_size) * s[..., None]).reshape(out, inn)


# ---------------------------------------------------------------------------------------- factory

_KINDS = {"bf16": BF16Codec, "affine": AffineRTN, "scalar": ScalarCodebook, "vq": VectorCodebook,
          "additive": AdditiveCodebook, "mxfp4": MXFP4Native}


def make_codec(cfg: dict) -> Codec:
    """Build a codec from {"kind": "bf16"|"affine"|"scalar"|"vq"|"additive"|"mxfp4"|"tcq", **constructor params}."""
    cfg = dict(cfg)
    kind = cfg.pop("kind", None)
    if kind == "tcq":    # lazy: suc.quant.trellis imports this module
        from suc.quant.trellis import TrellisCodec
        return TrellisCodec(**cfg)
    if kind not in _KINDS:
        raise ValueError(f"unknown codec kind {kind!r}; expected one of {sorted(_KINDS)}")
    try:
        return _KINDS[kind](**cfg)
    except TypeError as ex:
        raise ValueError(f"bad parameters for codec {kind!r}: {ex}") from None
