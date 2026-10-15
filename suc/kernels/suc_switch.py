"""SucSwitchLinear: MoE expert projections (mlx-lm SwitchLinear) executed directly from SUC codes.

An mlx-lm MoE block holds each expert projection as one stacked weight (E, out, in) and calls
SwitchLinear(x, indices) = gather_mm (row n of x times expert indices[n]). SUC encodes every expert of the
checkpoint ("…experts.{e}.<proj>.weight") as its own tensor (suc.evaluation.reconstruct): own codebooks, row scales,
rotation seeds (V_e, U_e) and optional sparse residual R_e. SucSwitchLinear keeps exactly that information,
concatenated over the E experts of one projection so ONE launch serves all selected experts:

    codebooks       (E, M, K, G) fp16          (expert e's M codebooks; D4 ceiling per expert: M·K·G·2 ≤ 16 KB)
    packed_indices  (E, M, out, words) uint32  (suc.quant.packing layout per expert)
    row_scales      (E, out) float32
    signs_in        (E, in)  / signs_out (E, out) int8 ±1 (per-expert randomized Hadamard signs)
  Shared expert-stack layout (suc.quant.stacks, codebook scope "expert_stack"): when every expert carries the same
  codebooks, codebooks is stored ONCE as (1, M, K, G) (the kernel prologue then does not offset the codebook pointer:
  one resident copy instead of E, staying hot in cache); when every expert has the same rotation seeds, signs_in /
  signs_out are (1, in) / (1, out) and the rotations run as plain (row-independent) launches. Per-expert layouts
  (earlier SUC1 files) are kept as they are.
    residual (opt.) CSR over the E·out rows: res_row_ptr (E·out + 1,) uint32, res_cols (n,) uint16/32, res_val (n,)
                    int8 or fp16 as stored, res_scale (E,) float32 (int8 scale of each expert, 1 for fp16)
    column residual (opt., ResidualCodec structure="columns"): every expert keeps its own column set; the sets are
                    concatenated in expert order with a column-pointer table (column counts may differ per expert,
                    experts without one have an empty range): colres_ptr (E + 1,) uint32, colres_cols (C,) uint16/32,
                    colres_val (C, out) int8 or fp16 as stored (column-major, as in the dense layout),
                    colres_scale (C,) float32 (per-column int8 scale, 1 for fp16); an optional CSR tail uses the
                    residual arrays above
    bias (opt.)     (E, out)

Per row n (a token-expert pair, e = expert of the row):  y_n = U_eᵀ (W'_e (V_e x_n)) + R_e x_n (+ b_e)
  1. x̃ = V_e x  — one fused rotation launch with a per-row sign table (suc.kernels.rotation, rows=)
  2. W'_e x̃     — the SUC lookup kernels of suc.kernels.vq_matmul, unchanged except for a prologue that offsets
                   the codebook / index / scale / x / y pointers of each threadgroup row (grid.y = tile) to the
                   tile's expert. A "tile" is BT consecutive rows of x that all use the same expert.
  3. U_eᵀ ỹ     — one fused inverse rotation launch with per-row signs
  4. + R_e x    — one CSR launch, row pointers offset by e·out (float32, before the single rounding); with column
                   residuals ONE structured launch (switch_structured_residual_metal: columns of expert e + its CSR
                   tail, rounded to the activation dtype inside the kernel when there is no bias)

Launch strategy (rows(x2, eidx, path)):
  "gather" (default for N ≤ GATHER_MAX_ROWS rows, i.e. decode: N = batch · top_k): BT = 1, one tile per row, so
      eidx is used as is: no host synchronisation, the whole call goes through mx.compile like SucLinear. Every
      row re-reads its expert's codes; with distinct experts per row (the decode case) that is the minimum anyway.
  "tiled" (prefill): the rows are grouped by expert on the host (one sync for the plan), each expert's rows are cut
      into tiles of BT ≤ 16 rows (padding rows compute garbage that is dropped), and ONE launch of the vec kernel
      (BT < MMA_MIN_BATCH) or the simdgroup-matrix kernel (BT ≥ MMA_MIN_BATCH) serves all tiles, so each expert's
      codes are read once per tile instead of once per row.
  "grouped": per-expert calls of the dense suc.kernels.vq_matmul.additive_matmul_metal on that expert's rows
      (one launch per active expert; an independent cross-check of the tile prologue, not used by "auto").
Rotations and residual are the same launches for every path.

Precision mirrors SucLinear / SucResidualLinear: rotations in float32, x̃ rounded to the activation dtype, kernel
output float32 when an output rotation or a residual follows, one rounding to the activation dtype at the end.
"""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import suc.kernels.suc_linear as suc_linear
import suc.kernels.vq_matmul as _vqm
from suc.kernels import rotation as _rot
from suc.kernels.suc_linear import SucLinear, compiled_call
from suc.quant.codecs import Encoded
from suc.quant.hadamard import _transform, hadamard

GATHER_MAX_ROWS = 32      # "auto": rows (token-expert pairs) up to which the sync-free BT = 1 gather path is used
TILE_SIZES = (1, 2, 4, 8, 16)   # BT candidates of the tiled path (≤ vq_matmul.MAX_BATCH_TILE)

# Prologue inserted before the vq_matmul kernel bodies: tile t = grid.y uses expert eidx[t] and rows
# x[t·BT : (t+1)·BT]. The bodies index codebook / packed / scales / x / y relative to these pointers.
_PROLOGUE = """
    {
        uint suc_t = threadgroup_position_in_grid.y;
        ulong suc_e = ulong(eidx[suc_t]);
        codebook += suc_e * ulong(CBE * M * K * G);
        packed += suc_e * ulong(M * OUT * WORDS);
        scales += suc_e * ulong(OUT);
        x += ulong(suc_t) * ulong(BT * IN);
        y += ulong(suc_t) * ulong(BT * OUT);
    }
"""
_SRC = {"vec": lambda: _vqm._LOOKUP_SRC, "mma": lambda: _vqm._MMA_REG_SRC}


@lru_cache(maxsize=None)
def _gather_kernel(kind: str):
    return mx.fast.metal_kernel(name=f"suc_switch_matmul_{kind}",
                                input_names=["x", "codebook", "packed", "scales", "eidx"], output_names=["y"],
                                source=_PROLOGUE + _SRC[kind](), header=_vqm._HEADER)


def gather_matmul_metal(x: mx.array, codebooks: mx.array, packed_indices: mx.array, row_scales: mx.array,
                        tile_experts: mx.array, *, bits: int, in_features: int, tile: int = 1,
                        out_dtype: mx.Dtype | None = None, path: str = "auto") -> mx.array:
    """y = per-tile expert matmul: x (T·tile, in), tile t = rows [t·tile, (t+1)·tile) uses expert tile_experts[t];
    codebooks (E, M, K, G) or (1, M, K, G) (one codebook set shared by every expert), packed_indices
    (E, M, out, words), row_scales (E, out) → (T·tile, out) in out_dtype (default x.dtype). path: "auto" (vec below
    vq_matmul.MMA_MIN_BATCH rows per tile, simdgroup-matrix from it), "vec", "mma"."""
    if codebooks.ndim != 4 or packed_indices.ndim != 4 or row_scales.ndim != 2:
        raise ValueError(f"expected codebooks (E, M, K, G), packed (E, M, out, words), scales (E, out); got "
                         f"{codebooks.shape}, {packed_indices.shape}, {row_scales.shape}")
    if codebooks.shape[0] not in (1, packed_indices.shape[0]):
        raise ValueError(f"codebooks {codebooks.shape} must hold 1 or {packed_indices.shape[0]} codebook sets")
    if not 1 <= tile <= _vqm.MAX_BATCH_TILE:
        raise ValueError(f"tile must be in 1..{_vqm.MAX_BATCH_TILE}, got {tile}")
    if path not in ("auto", "vec", "mma"):
        raise ValueError(f"path must be 'auto', 'vec' or 'mma', got {path!r}")
    n_t = tile_experts.shape[0]
    if x.shape[0] != n_t * tile:
        raise ValueError(f"x has {x.shape[0]} rows, expected {n_t} tiles x {tile}")
    m, k, g, out, words = _vqm._check(x, codebooks[0], packed_indices[0], row_scales[0], bits, in_features)
    if m * k * g * 2 > _vqm.CODEBOOK_TG_LIMIT:
        raise ValueError(f"{m} codebooks of K*G*2 = {k * g * 2} bytes exceed the 16 KB ceiling (D4)")
    out_dtype = out_dtype or x.dtype
    tmpl = [("T", x.dtype), ("BT", tile), ("OUT", out), ("IN", in_features), ("NG", in_features // g), ("G", g),
            ("K", k), ("BITS", bits), ("WORDS", words), ("CBE", 0 if codebooks.shape[0] == 1 else 1)]
    if path == "mma" or (path == "auto" and tile >= _vqm.mma_min_batch(m)):
        kind, sgs, rows = "mma", _vqm.MMA_SG, 8 * _vqm.MMA_RT
        tmpl += [("SG", sgs), ("M", m), ("RT", _vqm.MMA_RT), ("NB", -(-tile // 8)), ("KCG", _vqm._mma_kcg(g, bits)),
                 ("TT", _vqm._mma_tile(x.dtype, m)), ("CBTG", 0), ("TO", out_dtype)]
    else:
        kind, sgs, rows = "vec", _vqm.LOOKUP_SG, _vqm._lookup_rows(tile)
        tmpl += [("SG", sgs), ("M", m), ("R", rows), ("VEC", _vqm.LOOKUP_VEC), ("CBTG", 0), ("HM", 0),
                 ("TO", out_dtype)]
    n_tg = -(-out // (sgs * rows))
    return _gather_kernel(kind)(
        inputs=[x, codebooks.astype(mx.float16), packed_indices.astype(mx.uint32), row_scales.astype(mx.float32),
                tile_experts.astype(mx.uint32)],
        template=tmpl, grid=(n_tg * sgs * 32, n_t, 1), threadgroup=(sgs * 32, 1, 1),
        output_shapes=[(n_t * tile, out)], output_dtypes=[out_dtype])[0]


_RESIDUAL_SRC = """
    uint r = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    if (r >= OUT) return;
    uint e = eidx[b];
    ulong rr = ulong(e) * OUT + r;
    float acc = 0.0f;
    for (uint j = row_ptr[rr]; j < row_ptr[rr + 1]; ++j)
        acc += float(x[ulong(b) * IN + cols[j]]) * float(vals[j]);
    y[ulong(b) * OUT + r] = y_in[ulong(b) * OUT + r] + scale[e] * acc;
"""


@lru_cache(maxsize=None)
def _residual_kernel():
    return mx.fast.metal_kernel(name="suc_switch_sparse_residual",
                                input_names=["x", "y_in", "row_ptr", "cols", "vals", "scale", "eidx"],
                                output_names=["y"], source=_RESIDUAL_SRC)


def gather_residual_metal(x2: mx.array, y32: mx.array, eidx: mx.array, row_ptr: mx.array, cols: mx.array,
                          vals: mx.array, scale: mx.array) -> mx.array:
    """y32 (N, out) float32 + R_{eidx[n]} x2[n] per row; R = CSR over the E·out stacked rows, scale (E,)."""
    b, inn = x2.shape
    out = y32.shape[1]
    tx = min(256, out)
    return _residual_kernel()(inputs=[x2, y32, row_ptr, cols, vals, scale, eidx.astype(mx.uint32)],
                              template=[("IN", inn), ("OUT", out)], grid=(-(-out // tx) * tx, b, 1),
                              threadgroup=(tx, 1, 1), output_shapes=[(b, out)], output_dtypes=[mx.float32])[0]


# Column-structured residual of expert stacks (the per-expert analogue of compressed_model.structured_residual_metal).
# One threadgroup per (row b, tile of TR = 32·RQ output rows); e = eidx[b]. The threadgroup first gathers
# x[b, cols_j] · scale_j of the CMAX-bounded column range [cptr[e], cptr[e+1]) of its expert into threadgroup memory;
# lane l owns RQ consecutive output rows (one 4-element vector load per column from the column-major values when
# out % 4 == 0), simdgroup js takes columns j ≡ js (mod JS) and CSR tail entries k ≡ js (mod JS) of rows e·OUT + r;
# the JS partial sums meet in threadgroup memory. float32 throughout, one rounding to OT at the end. Rows are
# independent, so the gather (decode) and tiled (prefill) paths run the same launch on the rows in input order.
_COLRES_SRC = """
    constexpr uint TR = 32 * RQ;
    uint tid = thread_position_in_threadgroup.x;
    uint tn = threads_per_threadgroup.x;
    uint lane = tid % 32, js = tid / 32;
    uint b = threadgroup_position_in_grid.y;
    uint e = eidx[b];
    uint rt0 = threadgroup_position_in_grid.x * TR;
    uint r0 = rt0 + lane * RQ;
    ulong xb = ulong(b) * IN;
    threadgroup float xg[CMAX];
    threadgroup float red[RED];
    uint j0 = cptr[e];
    uint nc = cptr[e + 1] - j0;
    for (uint t = tid; t < nc; t += tn) xg[t] = float(x[xb + ccols[j0 + t]]) * cscale[j0 + t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[RQ];
    for (uint q = 0; q < RQ; ++q) acc[q] = 0.0f;
    if (r0 < OUT) {
        for (uint j = js; j < nc; j += JS) {
            ulong qb = ulong(j0 + j) * OUT + r0;
            float w[RQ];
            if (RQ == 4) {
                float4 v = suc_ld4(cval + qb);
                w[0] = v.x; w[RQ > 1 ? 1 : 0] = v.y; w[RQ > 2 ? 2 : 0] = v.z; w[RQ > 3 ? 3 : 0] = v.w;
            } else {
                for (uint q = 0; q < RQ; ++q) w[q] = float(cval[qb + q]);
            }
            float xv = xg[j];
            for (uint q = 0; q < RQ; ++q) acc[q] += w[q] * xv;
        }
        if (HAS_CSR) {
            float sc = rscale[e];
            for (uint q = 0; q < RQ; ++q) {
                uint r = r0 + q;
                if (r >= OUT) break;
                ulong rr = ulong(e) * OUT + r;
                uint k1 = row_ptr[rr + 1];
                for (uint k = row_ptr[rr] + js; k < k1; k += JS) {
                    float wv = sc * float(rvals[k]);
                    acc[q] += wv * float(x[xb + rcols[k]]);
                }
            }
        }
    }
    ulong yb = ulong(b) * OUT;
    if (JS > 1) {
        for (uint q = 0; q < RQ; ++q) red[js * TR + lane * RQ + q] = acc[q];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = tid; t < TR; t += tn) {
            uint r = rt0 + t;
            if (r >= OUT) continue;
            float v = 0.0f;
            for (uint s = 0; s < JS; ++s) v += red[s * TR + t];
            y[yb + r] = OT(y_in[yb + r] + v);
        }
    } else {
        for (uint q = 0; q < RQ; ++q) {
            uint r = r0 + q;
            if (r >= OUT) break;
            y[yb + r] = OT(y_in[yb + r] + acc[q]);
        }
    }
"""
COLRES_MAX_COLUMNS = 4096   # per expert (threadgroup gather buffer: 16 KiB)


@lru_cache(maxsize=None)
def _colres_kernel():
    from suc.runtime.compressed_model import _STRUCT_HEADER   # suc_ld4 (lazy: compressed_model imports this module)
    return mx.fast.metal_kernel(name="suc_switch_structured_residual",
                                input_names=["x", "y_in", "eidx", "cptr", "ccols", "cval", "cscale", "row_ptr",
                                             "rcols", "rvals", "rscale"],
                                output_names=["y"], source=_COLRES_SRC, header=_STRUCT_HEADER)


def colres_launch(out: int, cmax: int) -> tuple[int, int, int]:
    """(RQ, JS, RED) of switch_structured_residual_metal (also computed by the C++ host, cpp/src/metal/switch.cpp):
    4-row vector loads when out % 4 == 0; JS = largest power of two ≤ min(8, cmax) simdgroups split the columns."""
    rq = 4 if out % 4 == 0 else 1
    js = 8
    while js > 1 and js > cmax:
        js //= 2
    return rq, js, (js * 32 * rq if js > 1 else 1)


def switch_structured_residual_metal(x2: mx.array, y32: mx.array, eidx: mx.array, cptr: mx.array, cols: mx.array,
                                     cval: mx.array, cscale: mx.array, cmax: int, row_ptr: mx.array | None,
                                     rcols: mx.array | None, rvals: mx.array | None, rscale: mx.array,
                                     out_dtype: mx.Dtype = mx.float32) -> mx.array:
    """y32 (N, out) float32 + R_{eidx[n]} x2[n] per row, rounded once to out_dtype: column part of expert e =
    columns [cptr[e], cptr[e+1]) of cols / cval (C, out) / cscale (C,), cmax ≥ the largest per-expert count; optional
    CSR tail (row_ptr over the E·out stacked rows, rcols, rvals · rscale[e]; None without one)."""
    from suc.runtime.compressed_model import _dummy
    n, inn = x2.shape
    out = y32.shape[1]
    if not 1 <= cmax <= COLRES_MAX_COLUMNS:
        raise ValueError(f"per-expert residual columns must be in 1..{COLRES_MAX_COLUMNS}, got {cmax}")
    has_csr = row_ptr is not None
    rq, js, red = colres_launch(out, cmax)
    inputs = [x2, y32, eidx.astype(mx.uint32), cptr, cols, cval, cscale,
              row_ptr if has_csr else _dummy(mx.uint32), rcols if has_csr else _dummy(mx.uint32),
              rvals if has_csr else _dummy(mx.int8), rscale]
    n_tg = -(-out // (32 * rq))
    return _colres_kernel()(inputs=inputs, template=[("IN", inn), ("OUT", out), ("CMAX", cmax), ("RQ", rq), ("JS", js),
                                                     ("RED", red), ("HAS_CSR", has_csr), ("OT", out_dtype)],
                            grid=(n_tg * 32 * js, n, 1), threadgroup=(32 * js, 1, 1), output_shapes=[(n, out)],
                            output_dtypes=[out_dtype])[0]


# ------------------------------------------------------------------------------------------------ rotations

def rotate_rows(x2: mx.array, signs: mx.array, eidx: mx.array, *, inverse: bool,
                out_dtype: mx.Dtype) -> mx.array:
    """Row n: H (signs[e] ⊙ x_n) (inverse=False) or signs[e] ⊙ (Hᵀ x_n) (inverse=True), e = eidx[n].
    signs (1, n) (one rotation shared by every expert) uses the row-independent launch."""
    n = x2.shape[-1]
    shared = signs.shape[0] == 1
    if suc_linear.ROTATION == "fused" and _rot.supported(n):
        if shared:
            return _rot.rotate(x2, signs[0], inverse=inverse, out_dtype=out_dtype)
        return _rot.rotate(x2, signs, inverse=inverse, out_dtype=out_dtype, rows=eidx)
    s = signs[0] if shared else signs[eidx]
    if not inverse:
        xs = x2.astype(mx.float32) * s
        y = mx.hadamard_transform(xs, scale=float(1.0 / np.sqrt(n))) if suc_linear._native(n, False) else hadamard(xs)
    elif suc_linear._native(n, True):
        y = mx.hadamard_transform(x2.astype(mx.float32), scale=float(1.0 / np.sqrt(n))) * s
    else:
        y = _transform(x2.astype(mx.float32), transpose=True) * s
    return y.astype(out_dtype)


# ------------------------------------------------------------------------------------------------ module

def _index_dtype(n: int) -> mx.Dtype:
    return mx.uint16 if n <= 1 << 16 else mx.uint32


def _layout(e: Encoded) -> tuple:
    p = e.params
    rot = p.get("rotate", False)
    if e.codec == "tcq":       # trellis code layout (no codebooks); kernel constants depend on code / L / V / gain
        return (e.codec, int(p["kv"]), (float(p["k"]), int(p["L"]), int(p["V"]), int(p["T"]), p["code"],
                                        float(p["gain"]), tuple(p.get("code_mu", ())), tuple(p.get("code_sd", ()))),
                bool(rot and p.get("rotate_in", True)), bool(rot and p.get("rotate_out", False)))
    return (e.codec, p["index_bits"], tuple(e.arrays["codebooks"].shape), bool(rot and p.get("rotate_in", True)),
            bool(rot and p.get("rotate_out", True)))


class _Meta:
    """Plain attributes outside the parameter tree."""
    __slots__ = ("seeds_in", "seeds_out", "colres_cmax")

    def __init__(self, seeds_in, seeds_out, colres_cmax=0):
        self.seeds_in, self.seeds_out, self.colres_cmax = seeds_in, seeds_out, colres_cmax


class SucSwitchLinear(nn.Module):
    """Drop-in replacement for mlx-lm SwitchLinear running on the SUC codes of its E experts (module docstring)."""

    def __init__(self, codebooks: mx.array | None, packed_indices: mx.array, row_scales: mx.array, *, bits: int,
                 in_features: int, seeds_in: tuple[int, ...] | None = None, seeds_out: tuple[int, ...] | None = None,
                 signs_in: mx.array | None = None, signs_out: mx.array | None = None, bias: mx.array | None = None,
                 residual: dict | None = None, tcq: dict | None = None):
        """tcq (trellis-coded experts, suc.quant.trellis): the experts' common params; packed_indices are then the
        streams (E, out, in/T · NW) uint32 and codebooks is None (suc.kernels.tcq_matmul.tcq_gather_matmul_metal)."""
        super().__init__()
        if tcq is not None:
            from suc.kernels.tcq_matmul import kernel_consts

            if packed_indices.ndim != 3 or row_scales.shape != packed_indices.shape[:2]:
                raise ValueError(f"inconsistent tcq stacks: words {packed_indices.shape}, scales {row_scales.shape}")
            self._tcq = suc_linear.TcqMeta(tcq)
            self.tcq_cst = mx.array(kernel_consts(tcq))
            self.packed_indices = packed_indices.astype(mx.uint32)
            self.row_scales = row_scales.astype(mx.float32)
            self.bits, self.in_features, self.out_features = bits, in_features, packed_indices.shape[1]
        else:
            E, m = packed_indices.shape[:2]
            if codebooks.shape[:2] not in ((E, m), (1, m)) or row_scales.shape[0] != E:
                raise ValueError(f"inconsistent stacks: codebooks {codebooks.shape}, packed {packed_indices.shape}, "
                                 f"scales {row_scales.shape}")
            self.codebooks = codebooks.astype(mx.float16)
            self.packed_indices = packed_indices.astype(mx.uint32)
            self.row_scales = row_scales.astype(mx.float32)
            self.bits, self.in_features, self.out_features = bits, in_features, packed_indices.shape[2]
        self._meta = _Meta(seeds_in, seeds_out)
        if signs_in is not None:
            self.signs_in = signs_in.astype(suc_linear.SIGN_DTYPE)
        if signs_out is not None:
            self.signs_out = signs_out.astype(suc_linear.SIGN_DTYPE)
        if bias is not None:
            self.bias = bias
        self.n_residual = self.n_columns = 0
        if residual is not None and residual["n"] > 0:
            self.res_row_ptr, self.res_cols = residual["row_ptr"], residual["cols"]
            self.res_val, self.res_scale = residual["val"], residual["scale"]
            self.n_residual = int(residual["n"])
        if residual is not None and residual.get("n_columns", 0) > 0:
            self.colres_ptr, self.colres_cols = residual["col_ptr"], residual["col_cols"]
            self.colres_val, self.colres_scale = residual["col_val"], residual["col_scale"]
            if "res_scale" not in self:
                self.res_scale = residual["scale"]
            self.n_columns = int(residual["n_columns"])
            self._meta.colres_cmax = int(residual["col_max"])

    # -- construction ------------------------------------------------------------------------------
    @classmethod
    def from_encodings(cls, encodings: list[Encoded], bias: mx.array | None = None) -> "SucSwitchLinear":
        """From the E per-expert Encoded of one projection (vq / additive, optionally ResidualCodec-wrapped), in
        expert order. All experts must share shape and code layout (codec, index bits, M, K, G, rotation sides)."""
        if not encodings:
            raise ValueError("no expert encodings")
        shapes = {tuple(e.shape) for e in encodings}
        if len(shapes) != 1:
            raise ValueError(f"experts differ in shape: {sorted(shapes)}")
        out, inn = shapes.pop()
        bases = [e.base() if _is_residual(e) else e for e in encodings]
        layouts = {_layout(b) for b in bases}
        if len(layouts) != 1:
            raise ValueError(f"experts differ in code layout (codec, bits, codebooks, rotate_in, rotate_out): "
                             f"{sorted(layouts, key=str)}")
        lins = [SucLinear.from_encoded(b) for b in bases]
        f = lins[0]
        rot_in, rot_out = f.seed_in is not None, f.seed_out is not None
        res = cls._residual([e for e in encodings], out, inn)
        tcq = bases[0].params if f.tcq is not None else None
        return cls(None if tcq is not None else _stack_or_shared([l.codebooks for l in lins]),
                   mx.stack([l.packed_indices for l in lins]),
                   mx.stack([l.row_scales for l in lins]), bits=f.bits, in_features=inn,
                   seeds_in=tuple(l.seed_in for l in lins) if rot_in else None,
                   seeds_out=tuple(l.seed_out for l in lins) if rot_out else None,
                   signs_in=_stack_or_shared([l.signs_in for l in lins]) if rot_in else None,
                   signs_out=_stack_or_shared([l.signs_out for l in lins]) if rot_out else None, bias=bias,
                   residual=res, tcq=tcq)

    @staticmethod
    def _residual(encodings: list[Encoded], out: int, inn: int) -> dict | None:
        """Concatenated CSR of the experts' sparse residuals (experts without one contribute empty rows)."""
        if not any(_is_residual(e) for e in encodings):
            return None
        cres = SucSwitchLinear._column_residual(encodings, out, inn)
        dtypes = {e.arrays["res_val"].dtype for e in encodings if _is_residual(e) and e.arrays["res_pos"].size}
        if len(dtypes) > 1:
            raise ValueError(f"experts store residual values in different dtypes: {dtypes}")
        vdt = dtypes.pop() if dtypes else mx.float16
        counts, cols, vals, scales = [], [], [], []
        for e in encodings:
            if _is_residual(e) and e.arrays["res_pos"].size:
                pos = np.asarray(e.arrays["res_pos"].astype(mx.uint32)).astype(np.int64)
                if np.any(np.diff(pos) <= 0) or pos[-1] >= out * inn:
                    raise ValueError("res_pos must be sorted, unique and inside the weight")
                counts.append(np.bincount(pos // inn, minlength=out))
                cols.append(pos % inn)
                vals.append(e.arrays["res_val"])
                sc = e.arrays.get("res_scale")
                scales.append(float(sc.astype(mx.float32).item()) if sc is not None else 1.0)
            else:
                counts.append(np.zeros(out, np.int64))
                scales.append(1.0)
        n = int(sum(c.sum() for c in counts))
        row_ptr = np.concatenate([[0], np.cumsum(np.concatenate(counts))])
        if row_ptr[-1] >= 1 << 32:
            raise ValueError("residual too large for uint32 row pointers")
        return {"n": n, "row_ptr": mx.array(row_ptr.astype(np.uint32)),
                "cols": mx.array(np.concatenate(cols) if cols else np.zeros(0, np.int64)).astype(_index_dtype(inn)),
                "val": mx.concatenate(vals).astype(vdt) if vals else mx.zeros((0,), vdt),
                "scale": mx.array(np.array(scales, np.float32)), **cres}

    @staticmethod
    def _column_residual(encodings: list[Encoded], out: int, inn: int) -> dict:
        """Concatenated per-expert column residuals with the column-pointer table ({} when no expert has one)."""
        parts = []
        for e in encodings:
            a = e.arrays if _is_residual(e) else {}
            if "res_cols" not in a or a["res_cols"].size == 0:
                parts.append(None)
                continue
            c = np.asarray(a["res_cols"].astype(mx.uint32)).astype(np.int64)
            if np.any(np.diff(c) <= 0) or c[-1] >= inn:
                raise ValueError("res_cols must be sorted, unique and inside the input dimension")
            val = a["res_colval"]
            if tuple(val.shape) != (c.size, out):
                raise ValueError(f"res_colval {tuple(val.shape)} does not match ({c.size}, {out})")
            sc = a["res_colscale"].astype(mx.float32).reshape(-1) if "res_colscale" in a \
                else mx.ones((c.size,), mx.float32)
            parts.append((c, val, sc))
        have = [p for p in parts if p is not None]
        if not have:
            return {}
        dtypes = {p[1].dtype for p in have}
        if len(dtypes) > 1:
            raise ValueError(f"experts store column residual values in different dtypes: {dtypes}")
        counts = np.array([0 if p is None else p[0].size for p in parts], np.int64)
        return {"n_columns": int(counts.sum()), "col_max": int(counts.max()),
                "col_ptr": mx.array(np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)),
                "col_cols": mx.array(np.concatenate([p[0] for p in have])).astype(_index_dtype(inn)),
                "col_val": mx.concatenate([p[1] for p in have], axis=0),
                "col_scale": mx.concatenate([p[2] for p in have])}

    # -- SwitchLinear interface --------------------------------------------------------------------
    @property
    def num_experts(self) -> int:
        return self.packed_indices.shape[0]

    @property
    def tcq(self):
        """suc.kernels.suc_linear.TcqMeta of a trellis-coded stack, None for the codebook kinds."""
        return self.__dict__.get("_tcq")

    @property
    def shared_codebooks(self) -> bool:
        """True when one codebook set serves every expert (shared expert-stack layout)."""
        return self.tcq is None and self.codebooks.shape[0] == 1 and self.num_experts > 1

    @property
    def input_dims(self) -> int:
        return self.in_features

    @property
    def output_dims(self) -> int:
        return self.out_features

    @property
    def seeds_in(self):
        return self._meta.seeds_in

    @property
    def seeds_out(self):
        return self._meta.seeds_out

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        """Same contract as SwitchLinear / mx.gather_mm: x (*bx, R, in), indices (*bi) with bx, bi broadcastable
        → (*broadcast(bx, bi), R, out); row block n uses expert indices[n]. sorted_indices is only a hint."""
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected (..., {self.in_features}) input, got {x.shape}")
        r = x.shape[-2]
        batch = tuple(np.broadcast_shapes(tuple(x.shape[:-2]), tuple(indices.shape)))
        x2 = mx.broadcast_to(x, (*batch, r, self.in_features)).reshape(-1, self.in_features)
        e = mx.broadcast_to(indices.astype(mx.uint32).reshape(*indices.shape, 1), (*batch, r)).reshape(-1)
        return self.rows(x2, e).reshape(*batch, r, self.out_features)

    # -- row computation ---------------------------------------------------------------------------
    def _choose(self, n: int, path: str) -> str:
        if path not in ("auto", "gather", "tiled", "grouped"):
            raise ValueError(f"path must be 'auto', 'gather', 'tiled' or 'grouped', got {path!r}")
        return path if path != "auto" else ("gather" if n <= GATHER_MAX_ROWS else "tiled")

    def rows(self, x2: mx.array, eidx: mx.array, path: str = "auto") -> mx.array:
        """y (N, out) for x2 (N, in), row n through expert eidx[n] (N,)."""
        eidx = eidx.astype(mx.uint32)
        path = self._choose(x2.shape[0], path)
        if path == "gather":
            if suc_linear.COMPILE:
                return compiled_call(self, self._gather_rows, ("gather", suc_linear.ROTATION), x2, eidx)
            return self._gather_rows(x2, eidx)
        return self._forward(x2, eidx, path)

    def _gather_rows(self, x2: mx.array, eidx: mx.array) -> mx.array:
        return self._forward(x2, eidx, "gather")

    def _forward(self, x2: mx.array, eidx: mx.array, path: str) -> mx.array:
        dtype = x2.dtype
        xr = rotate_rows(x2, self.signs_in, eidx, inverse=False, out_dtype=dtype) if "signs_in" in self else x2
        f32 = "signs_out" in self or self.n_residual > 0 or self.n_columns > 0
        kdt = mx.float32 if f32 else dtype
        if path == "gather":
            y = self._kernel(xr, eidx, 1, kdt)
        elif path == "tiled":
            y = self._tiled(xr, eidx, kdt)
        else:
            y = self._grouped(xr, eidx, kdt)
        res = self.n_residual > 0 or self.n_columns > 0
        if "signs_out" in self:
            y = rotate_rows(y, self.signs_out, eidx, inverse=True, out_dtype=mx.float32 if res else dtype)
        if self.n_columns:
            csr = self.n_residual > 0
            bias = "bias" in self
            y = switch_structured_residual_metal(x2, y, eidx, self.colres_ptr, self.colres_cols, self.colres_val,
                                                 self.colres_scale, self._meta.colres_cmax,
                                                 self.res_row_ptr if csr else None, self.res_cols if csr else None,
                                                 self.res_val if csr else None, self.res_scale,
                                                 mx.float32 if bias else dtype)
            if bias:
                y = (y + self.bias[eidx].astype(mx.float32)).astype(dtype)
            return y
        if self.n_residual:
            y = gather_residual_metal(x2, y, eidx, self.res_row_ptr, self.res_cols, self.res_val, self.res_scale)
            if "bias" in self:
                y = y + self.bias[eidx].astype(mx.float32)
            return y.astype(dtype)
        y = y.astype(dtype)
        if "bias" in self:
            y = y + self.bias[eidx].astype(dtype)
        return y

    def _kernel(self, xt: mx.array, tile_e: mx.array, tile: int, out_dtype: mx.Dtype) -> mx.array:
        t = self.tcq
        if t is not None:
            from suc.kernels.tcq_matmul import tcq_gather_matmul_metal

            return tcq_gather_matmul_metal(xt, self.packed_indices, self.row_scales, tile_e, k=t.k, L=t.L, V=t.V,
                                           T=t.T, code=t.code, cst=self.tcq_cst, tile=tile, out_dtype=out_dtype)
        return gather_matmul_metal(xt, self.codebooks, self.packed_indices, self.row_scales, tile_e, bits=self.bits,
                                   in_features=self.in_features, tile=tile, out_dtype=out_dtype)

    def _tiled(self, xr: mx.array, eidx: mx.array, out_dtype: mx.Dtype) -> mx.array:
        src, dst, tile_e, tile = tile_plan(np.asarray(eidx), self.num_experts)
        y = self._kernel(xr[mx.array(src)], mx.array(tile_e), tile, out_dtype)
        return y[mx.array(dst)]

    def _grouped(self, xr: mx.array, eidx: mx.array, out_dtype: mx.Dtype) -> mx.array:
        e = np.asarray(eidx).astype(np.int64)
        order = np.argsort(e, kind="stable")
        ys = []
        t = self.tcq
        for ex in np.unique(e):
            rows = order[e[order] == ex]
            if t is not None:
                from suc.kernels.tcq_matmul import tcq_matmul_metal

                ys.append(tcq_matmul_metal(xr[mx.array(rows)], self.packed_indices[int(ex)], self.row_scales[int(ex)],
                                           k=t.k, L=t.L, V=t.V, T=t.T, code=t.code, cst=self.tcq_cst,
                                           out_dtype=out_dtype))
                continue
            ys.append(_vqm.additive_matmul_metal(xr[mx.array(rows)], self.codebooks[0 if self.shared_codebooks
                                                                                   else int(ex)],
                                                 self.packed_indices[int(ex)], self.row_scales[int(ex)],
                                                 bits=self.bits, in_features=self.in_features, out_dtype=out_dtype))
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return mx.concatenate(ys, axis=0)[mx.array(inv)]


def tile_plan(eidx: np.ndarray, num_experts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Group rows by expert into tiles of BT rows: (src (T·BT,) row of x per padded row (padding → row 0),
    dst (N,) padded row of each input row, tile_experts (T,), BT). BT = smallest TILE_SIZES entry ≥ the mean rows
    per active expert (capped at the largest)."""
    e = eidx.astype(np.int64).reshape(-1)
    n = e.size
    counts = np.bincount(e, minlength=num_experts)
    active = counts[counts > 0]
    mean = int(np.ceil(active.mean())) if active.size else 1
    tile = next((t for t in TILE_SIZES if t >= mean), TILE_SIZES[-1])
    order = np.argsort(e, kind="stable")
    es = e[order]
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    ntiles = -(-counts // tile)
    pad_start = np.concatenate([[0], np.cumsum(ntiles * tile)])[:-1]
    pos_sorted = pad_start[es] + (np.arange(n) - starts[es])
    src = np.zeros(int(ntiles.sum()) * tile, np.int64)
    src[pos_sorted] = order
    dst = np.empty(n, np.int64)
    dst[order] = pos_sorted
    tile_e = np.repeat(np.arange(num_experts), ntiles).astype(np.uint32)
    return src.astype(np.uint32), dst.astype(np.uint32), tile_e, tile


def _stack_or_shared(arrays: list[mx.array]) -> mx.array:
    """(1, ...) when every array is bit-identical (one copy serves all experts), else the (E, ...) stack."""
    first = np.asarray(arrays[0])
    if len(arrays) > 1 and all(a.dtype == arrays[0].dtype and np.array_equal(np.asarray(a), first)
                               for a in arrays[1:]):
        return arrays[0][None]
    return mx.stack(arrays)


def _is_residual(e: Encoded) -> bool:
    return "res_pos" in e.arrays and hasattr(e, "base")


# ------------------------------------------------------------------------------------------- mixed-codec stacks

class SucSwitchMixed(nn.Module):
    """MoE expert stack whose experts use DIFFERENT codecs (allocation unit "expert" of suc compress): one
    SucSwitchLinear (native MXFP4 experts: MXFP4SwitchLinear) per codec over its experts (shared codebooks per codec stay shared). Every sub-stack computes every
    routed row with its local expert index (rows routed to another sub-stack use local index 0 and are masked out),
    so the result is row for row the result of the sub-stack holding the expert; the cost is one launch (and one pass
    over the selected rows) per codec present in the stack."""

    def __init__(self, subs: list[SucSwitchLinear], members: list[list[int]], num_experts: int):
        super().__init__()
        if len(subs) != len(members) or sorted(e for m in members for e in m) != list(range(num_experts)):
            raise ValueError("SucSwitchMixed: the sub-stacks must partition the experts")
        self.subs = subs
        local = np.full((len(subs), num_experts), -1, dtype=np.int32)
        for s, m in enumerate(members):
            local[s, m] = np.arange(len(m))
        self.local = mx.array(local)
        self.members = [list(m) for m in members]
        self.in_features, self.out_features = subs[0].in_features, subs[0].out_features
        self._n = num_experts
        self.freeze()

    @property
    def num_experts(self) -> int:
        return self._n

    @property
    def input_dims(self) -> int:
        return self.in_features

    @property
    def output_dims(self) -> int:
        return self.out_features

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        y = None
        for s, sub in enumerate(self.subs):
            loc = self.local[s][indices.astype(mx.uint32)]
            ys = sub(x, mx.maximum(loc, 0).astype(mx.uint32))
            m = (loc >= 0).reshape(*loc.shape, 1, 1)
            y = mx.where(m, ys, mx.zeros_like(ys)) if y is None else mx.where(m, ys, y)
        return y


def switch_from_encodings(encodings: list[Encoded], bias: mx.array | None = None):
    """SucSwitchLinear when every expert shares one code layout (as from_encodings requires; residuals may differ),
    else a SucSwitchMixed (one sub-stack per layout, experts in order within each). Native MXFP4 experts
    (suc.quant.codecs.MXFP4Native) form one suc.kernels.mxfp4.MXFP4SwitchLinear (sub-)stack."""
    from suc.kernels.mxfp4 import MXFP4SwitchLinear

    def build(encs: list[Encoded], bias=None):
        if encs[0].codec == "mxfp4":
            return MXFP4SwitchLinear.from_encodings(encs, bias=bias)
        return SucSwitchLinear.from_encodings(encs, bias=bias)

    groups: dict[tuple, list[int]] = {}
    for i, e in enumerate(encodings):
        key = ("mxfp4",) if e.codec == "mxfp4" else _layout(e.base() if _is_residual(e) else e)
        groups.setdefault(key, []).append(i)
    if len(groups) == 1:
        return build(encodings, bias=bias)
    if bias is not None:
        raise NotImplementedError("mixed-codec expert stacks with a bias")
    members = sorted(groups.values())
    subs = [build([encodings[i] for i in m]) for m in members]
    return SucSwitchMixed(subs, members, len(encodings))
