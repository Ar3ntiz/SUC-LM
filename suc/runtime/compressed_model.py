"""Full-transformer integration (spec §22 Phase 15, §46; README §5): an mlx-lm model whose linear layers run
directly on SUC codes.

convert_to_compressed(model, encodings) replaces, in place, every nn.Linear whose weight has an Encoded of kind
vq / additive (optionally wrapped by suc.quant.residual.ResidualCodec) by a SucLinear, or by a SucResidualLinear
when the Encoded carries a sparse residual. Every other module is left untouched, so one model mixes BF16
nn.Linear, MLX QuantizedLinear / QuantizedEmbedding and SUC modules (mixed representations per tensor, §22).

Embedding / LM head: never converted here. They stay as loaded (BF16) or as quantized by the caller with MLX
affine (suc.evaluation.baselines.apply_representation or nn.quantize). The tied embedding of small Qwen3 models
is also the LM head (embed_tokens.as_linear), which a SucLinear cannot serve (gather of rows from codes is a
different kernel); an untied lm_head is not an eligible role of the adapter (suc.models.adapters._NOT_ELIGIBLE).

Tensor names are in the checkpoint namespace ("model.layers.3.mlp.down_proj.weight"), which for dense Qwen3 is
also the mlx-lm parameter path. MoE experts: the checkpoint tensors "<p>.experts.{e}.<proj>.weight" of one projection
are the rows of the stacked mlx-lm SwitchLinear "<p>.switch_mlp.<proj>" (the mapping of suc.analysis.sensitivity /
suc.evaluation.reconstruct); all E experts of a stack must be given together and replace it by ONE
suc.kernels.suc_switch.SucSwitchLinear (codes of all experts concatenated, one launch for all selected experts).
The router ("<p>.gate", not an eligible role) is never converted.

Sparse residual (W = W_base + R, R stored at n positions; suc.quant.residual). R lives in the ORIGINAL (unrotated)
weight space, so R x uses the unrotated input:
    y = Uᵀ(W' V x) + R x,   (R x)_r = Σ_{c ∈ supp R_r} R_rc x_c
Execution layout: CSR (positions are sorted row-major) — row_ptr (out+1,) uint32, cols (n,) uint16 (uint32 when
in > 65536), values as stored (int8 + one float32 scale, or fp16): 3 or 4 bytes per entry + 4 bytes per row.
The Metal kernel sparse_residual_metal runs one thread per (batch row, output row), sums its entries in float32 and
adds the result to the float32 output of the SUC kernel / Uᵀ before the single rounding to the activation dtype.
The first version used MLX gather + scatter-add (.at[].add, kept as residual_path="mlx" for A/B): 14.3 ms of a
27.5 ms Qwen3-0.6B decode step at 0.5% residual (M3), i.e. slower than all the SUC matmul kernels together.

Column-structured residual (ResidualCodec structure="columns", docs/benchmarks.md Q5b): R = R_c (dense on c input
columns) + an optional CSR tail. Execution layout: colres_cols (c,) uint16/uint32, colres_val (c, out) as stored
(int8 or fp16; column-major, so the threads of consecutive output rows read consecutive bytes), colres_scale (c,)
float32. structured_residual_metal runs one thread per (batch row, output row): Σ_j colres_val[j, r] · scale_j ·
x[cols_j] + the CSR tail of that row, in float32, added before the single rounding. The same kernel serves a
row-concatenation of S ≤ 4 members (SucFusedResidual: q|k|v, gate|up with residuals), each member with its own
columns, CSR rows and int8 scale, so residual members keep the fused SUC / output-rotation launches and the whole
group needs ONE residual launch (FUSE_RESIDUAL; False restores the earlier per-member CSR path for A/B).

Input rotations fused into element-wise launches (K6, FUSE_ELEMENTWISE): the shared q|k|v and gate|up input
rotations run inside the input / post-attention RMSNorm launch (RotatedRMSNorm), down_proj's inside the SwiGLU launch
(SucMLP); see fuse_elementwise_rotations. With one-sided encodings (rotate_out False) a block then has no rotation
launch except o_proj's input rotation.

Memory: the replaced nn.Linear (and its dense weight) is dropped from the module tree; with no other reference
the weight buffer is released and mx.clear_cache() returns it, so mx.get_active_memory() reflects the compressed
form. The report counts bytes of the dense weights removed and of every array the new modules hold
(codebooks, packed indices, row scales, rotation signs, residual arrays).
"""
from __future__ import annotations

import gc
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

import suc.kernels.rotation as _rot
from suc.kernels import autotune
import suc.kernels.suc_linear as suc_linear
from suc.kernels.suc_linear import SucFusedLinear, SucLinear, compiled_call, fusable_key
from suc.kernels.suc_switch import SucSwitchLinear, switch_from_encodings
from suc.quant.codecs import Encoded

_EXPERT = re.compile(r"^(.*)\.experts\.(\d+)\.(\w+)\.weight$")


def _index_dtype(n: int) -> mx.Dtype:
    return mx.uint16 if n <= 1 << 16 else mx.uint32


_RESIDUAL_SRC = """
    uint r = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    if (r >= OUT) return;
    float acc = 0.0f;
    for (uint j = row_ptr[r]; j < row_ptr[r + 1]; ++j)
        acc += float(x[b * IN + cols[j]]) * float(vals[j]);
    y[b * OUT + r] = y_in[b * OUT + r] + scale[0] * acc;
"""
_RESIDUAL_KERNEL = None


def sparse_residual_metal(x2: mx.array, y32: mx.array, row_ptr: mx.array, cols: mx.array, vals: mx.array,
                          scale: mx.array) -> mx.array:
    """y32 (B, out) float32 + R x2, R in CSR form (row_ptr (out+1,), cols (n,), vals (n,) · scale (1,))."""
    global _RESIDUAL_KERNEL
    if _RESIDUAL_KERNEL is None:
        _RESIDUAL_KERNEL = mx.fast.metal_kernel(name="suc_sparse_residual",
                                                input_names=["x", "y_in", "row_ptr", "cols", "vals", "scale"],
                                                output_names=["y"], source=_RESIDUAL_SRC)
    b, inn = x2.shape
    out = y32.shape[1]
    tx = min(256, out)
    return _RESIDUAL_KERNEL(inputs=[x2, y32, row_ptr, cols, vals, scale], template=[("IN", inn), ("OUT", out)],
                            grid=(-(-out // tx) * tx, b, 1), threadgroup=(tx, 1, 1), output_shapes=[(b, out)],
                            output_dtypes=[mx.float32])[0]


# True: residual members are fused into SucFusedResidual group parts and every residual (also CSR-only) runs through
# structured_residual_metal with the rounding in the kernel; False: per-member CSR launch + cast, as before (A/B).
FUSE_RESIDUAL = True

# Latency-oriented layout (a decode step runs these launches one after another, so the time of ONE launch matters,
# not the throughput of many independent ones — the first version, one thread per output row looping over all c
# columns, cost 40-70 µs per dependent launch on 8B shapes although its bytes take a few µs):
#   threadgroup = 32 lanes × JS simdgroups; lane l owns RQ = 4 consecutive output rows (one 4-element vector load
#   per column from the column-major values: a simdgroup reads 128 consecutive bytes), simdgroup js takes columns
#   j ≡ js (mod JS) and CSR tail entries e ≡ js (mod JS); the JS partial sums meet in threadgroup memory.
#   The threadgroup first gathers x[b, cols_j] · scale_j of ALL CTOT columns into threadgroup memory (one pass).
_STRUCT_SRC = """
    const uint B = x_shape[0];
    constexpr uint TR = 32 * RQ;                      // output rows per threadgroup
    uint tid = thread_position_in_threadgroup.x;
    uint tn = threads_per_threadgroup.x;              // 32 * JS
    uint lane = tid % 32, js = tid / 32;
    uint b0 = threadgroup_position_in_grid.y * BT;
    uint rt0 = threadgroup_position_in_grid.x * TR;
    uint r0 = rt0 + lane * RQ;
    threadgroup float xg[XG];
    threadgroup float red[RED];
    if (HAS_COLS) {
        for (uint t = tid; t < BT * CTOT; t += tn) {
            uint bb = t / CTOT, j = t - bb * CTOT;
            xg[t] = (b0 + bb < B) ? float(x[(b0 + bb) * IN + ccols[j]]) * cscale[j] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float acc[RQ][BT];
    for (uint q = 0; q < RQ; ++q) for (uint bb = 0; bb < BT; ++bb) acc[q][bb] = 0.0f;
    if (r0 < OUT) {
        uint m = 0;
        for (uint s = 1; s < S; ++s) m += uint(r0 >= meta[s]);
        if (HAS_COLS) {
            uint om = meta[m + 1] - meta[m];
            uint j0 = meta[S + 1 + m], j1 = meta[S + 2 + m];
            uint qb = meta[2 * S + 2 + m] + (r0 - meta[m]);
            for (uint j = j0 + js; j < j1; j += JS) {
                float w[RQ];
                if (RQ == 4) {
                    float4 v = suc_ld4(cval + qb + (j - j0) * om);
                    w[0] = v.x; w[RQ > 1 ? 1 : 0] = v.y; w[RQ > 2 ? 2 : 0] = v.z; w[RQ > 3 ? 3 : 0] = v.w;
                } else {
                    for (uint q = 0; q < RQ; ++q) w[q] = float(cval[qb + (j - j0) * om + q]);
                }
                for (uint bb = 0; bb < BT; ++bb) {
                    float xv = xg[bb * CTOT + j];
                    for (uint q = 0; q < RQ; ++q) acc[q][bb] += w[q] * xv;
                }
            }
        }
        if (HAS_CSR) {
            float sc = rscale[m];
            for (uint q = 0; q < RQ; ++q) {
                uint r = r0 + q;
                if (r >= OUT) break;
                uint e1 = row_ptr[r + 1];
                for (uint e = row_ptr[r] + js; e < e1; e += JS) {
                    float wv = sc * float(rvals[e]);
                    uint c = rcols[e];
                    for (uint bb = 0; bb < BT; ++bb)
                        if (b0 + bb < B) acc[q][bb] += wv * float(x[(b0 + bb) * IN + c]);
                }
            }
        }
    }
    if (JS > 1) {
        for (uint bb = 0; bb < BT; ++bb)
            for (uint q = 0; q < RQ; ++q) red[(js * BT + bb) * TR + lane * RQ + q] = acc[q][bb];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = tid; t < BT * TR; t += tn) {
            uint bb = t / TR, rr = t - bb * TR;
            uint r = rt0 + rr, b = b0 + bb;
            if (r >= OUT || b >= B) continue;
            float v = 0.0f;
            for (uint s = 0; s < JS; ++s) v += red[(s * BT + bb) * TR + rr];
            y[b * OUT + r] = OT(y_in[b * OUT + r] + v);
        }
    } else {
        for (uint q = 0; q < RQ; ++q) {
            uint r = r0 + q;
            if (r >= OUT) break;
            for (uint bb = 0; bb < BT; ++bb) {
                uint b = b0 + bb;
                if (b < B) y[b * OUT + r] = OT(y_in[b * OUT + r] + acc[q][bb]);
            }
        }
    }
"""
_STRUCT_HEADER = """
// 4 consecutive values as float4: one vector load from device memory (16/8/4-byte aligned: q % 4 == 0);
// MLX passes small inputs (the 1-element placeholders) in the constant address space
template <typename V> inline float4 suc_ld4(const device V* p) { return float4(*((const device vec<V, 4>*)p)); }
template <typename V> inline float4 suc_ld4(const constant V* p) { return float4(p[0], p[1], p[2], p[3]); }
"""
_STRUCT_KERNEL = None
_XG_FLOATS = 6144     # threadgroup gather buffer (24 KiB): BT * CTOT must fit
_TG_BYTES = 32 * 1024  # threadgroup memory per threadgroup (gather + reduction)
_DUMMY: dict = {}


def _dummy(dtype: mx.Dtype) -> mx.array:
    """A materialized 1-element placeholder input (built from numpy: no Full launch, also inside mx.compile)."""
    if dtype not in _DUMMY:
        _DUMMY[dtype] = mx.array(np.zeros((1,), {mx.uint32: np.uint32, mx.int8: np.int8,
                                                  mx.float32: np.float32}[dtype]))
    return _DUMMY[dtype]


def structured_residual_metal(x2: mx.array, y32: mx.array, meta: mx.array, cols: mx.array | None,
                              cval: mx.array | None, cscale: mx.array | None, row_ptr: mx.array | None,
                              rcols: mx.array | None, rvals: mx.array | None, rscale: mx.array, segments: int,
                              out_dtype: mx.Dtype = mx.float32, row_quads: bool | None = None) -> mx.array:
    """y32 (B, out) float32 + R x2 for a row-concatenation of `segments` members (meta: row starts (S+1), column
    pointers (S+1), column-block offsets (S), uint32). Column part (cols / cval (Σ c_m·out_m) / cscale) and CSR
    part (row_ptr / rcols / rvals · rscale[member]) are each optional (None). The sum is rounded once to
    out_dtype inside the kernel (saves the separate cast launch). row_quads: every member's row count is a
    multiple of 4 (4-row vector loads); None infers it for one segment (out % 4 == 0), False for several."""
    global _STRUCT_KERNEL
    if _STRUCT_KERNEL is None:
        _STRUCT_KERNEL = mx.fast.metal_kernel(
            name="suc_structured_residual",
            input_names=["x", "y_in", "meta", "ccols", "cval", "cscale", "row_ptr", "rcols", "rvals", "rscale"],
            output_names=["y"], source=_STRUCT_SRC, header=_STRUCT_HEADER)
    b, inn = x2.shape
    out = y32.shape[1]
    has_cols, has_csr = cols is not None, row_ptr is not None
    ctot = int(cols.shape[0]) if has_cols else 0
    if ctot > _XG_FLOATS:
        raise ValueError(f"{ctot} residual columns exceed the gather buffer ({_XG_FLOATS})")
    if row_quads is None:
        row_quads = segments == 1 and out % 4 == 0
    rq = 4 if row_quads else 1
    bt = 1 if b == 1 else 4 if b <= 4 else 8
    while bt > 1 and bt * ctot > _XG_FLOATS:
        bt //= 2
    js = 8                       # simdgroups splitting the columns: gather + reduction buffers within 32 KiB
    while js > 1 and 4 * (bt * ctot + js * 32 * rq * bt) > _TG_BYTES:
        js //= 2
    red = js * 32 * rq * bt if js > 1 else 1
    inputs = [x2, y32, meta,
              cols if has_cols else _dummy(mx.uint32), cval if has_cols else _dummy(mx.int8),
              cscale if has_cols else _dummy(mx.float32), row_ptr if has_csr else _dummy(mx.uint32),
              rcols if has_csr else _dummy(mx.uint32), rvals if has_csr else _dummy(mx.int8), rscale]
    n_tg = -(-out // (32 * rq))
    return _STRUCT_KERNEL(inputs=inputs, template=[("IN", inn), ("OUT", out), ("S", segments), ("CTOT", max(ctot, 1)),
                                                   ("BT", bt), ("XG", max(bt * ctot, 1)), ("RQ", rq), ("JS", js),
                                                   ("RED", red),
                                                   ("HAS_COLS", has_cols), ("HAS_CSR", has_csr), ("OT", out_dtype)],
                          grid=(n_tg * 32 * js, -(-b // bt), 1), threadgroup=(32 * js, 1, 1),
                          output_shapes=[(b, out)], output_dtypes=[out_dtype])[0]


def _column_parts(cols: mx.array | None, val: mx.array | None, scale: mx.array | None, out: int, inn: int):
    """Validated (cols int64 (c,), values (c, out) as stored, scale float32 (c,)) of a column residual."""
    if cols is None or cols.size == 0:
        return None
    c = np.asarray(cols.astype(mx.uint32)).astype(np.int64)
    if np.any(np.diff(c) <= 0) or c[-1] >= inn:
        raise ValueError("res_cols must be sorted, unique and inside the input dimension")
    if tuple(val.shape) != (c.size, out):
        raise ValueError(f"res_colval {tuple(val.shape)} does not match ({c.size}, {out})")
    sc = scale.astype(mx.float32).reshape(-1) if scale is not None else mx.ones((c.size,), mx.float32)
    return c, val, sc


class SucResidualLinear(SucLinear):
    """SucLinear + sparse residual R (original weight space): y = SucLinear(x) + R x, summed in float32.

    residual_path: "metal" (CSR kernel, default) or "mlx" (gather + scatter-add, the slower first version).
    With a column-structured residual (res_cols / res_colval / res_colscale) the residual runs through
    structured_residual_metal (columns + CSR tail in one launch)."""

    def __init__(self, *args, res_pos: mx.array, res_val: mx.array, res_scale: mx.array | None = None,
                 residual_path: str = "metal", res_cols: mx.array | None = None, res_colval: mx.array | None = None,
                 res_colscale: mx.array | None = None, **kw):
        super().__init__(*args, **kw)
        if residual_path not in ("metal", "mlx"):
            raise ValueError(f"residual_path must be 'metal' or 'mlx', got {residual_path!r}")
        pos = np.asarray(res_pos.astype(mx.uint32)).astype(np.int64)
        if pos.size and (np.any(np.diff(pos) <= 0) or pos[-1] >= self.in_features * self.out_features):
            raise ValueError("res_pos must be sorted, unique and inside the weight")
        inn, out = self.in_features, self.out_features
        rows = pos // inn
        self.res_row_ptr = mx.array(np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=out))]).astype(np.uint32))
        self.res_cols = mx.array(pos % inn).astype(_index_dtype(inn))
        self.res_val = res_val            # int8 (with res_scale) or fp16, as stored
        self.res_scale = (res_scale.astype(mx.float32) if res_scale is not None else mx.array(1.0)).reshape(1)
        self.n_residual = int(pos.size)
        self.residual_path = residual_path
        cp = _column_parts(res_cols, res_colval, res_colscale, out, inn)
        self.n_columns = 0
        if cp is not None:
            c, val, sc = cp
            self.colres_cols = mx.array(c).astype(_index_dtype(inn))
            self.colres_val, self.colres_scale = val, sc
            self.n_columns = int(c.size)
        self.colres_meta = mx.array(np.array([0, out, 0, self.n_columns, 0], np.uint32))

    def compilable(self) -> bool:
        return self.residual_path == "metal"   # the "mlx" reference path reads arrays on the host

    def _structured(self) -> bool:
        """Column residual, or CSR through the structured kernel (FUSE_RESIDUAL, metal path)."""
        return self.n_columns > 0 or (FUSE_RESIDUAL and self.residual_path == "metal" and self.n_residual > 0)

    def residual(self, x2: mx.array, y32: mx.array, out_dtype: mx.Dtype = mx.float32) -> mx.array:
        """y32 (B, out) float32 + R x2 for x2 (B, in) in the original (unrotated) basis (float32; the column path
        can round to out_dtype in the kernel)."""
        if self._structured():
            if self.residual_path != "metal":
                raise ValueError("column-structured residuals have only the metal path")
            csr = self.n_residual > 0
            cols = self.n_columns > 0
            return structured_residual_metal(x2, y32, self.colres_meta, self.colres_cols if cols else None,
                                             self.colres_val if cols else None, self.colres_scale if cols else None,
                                             self.res_row_ptr if csr else None,
                                             self.res_cols if csr else None, self.res_val if csr else None,
                                             self.res_scale, 1, out_dtype)
        if self.n_residual == 0:
            return y32
        if self.residual_path == "metal":
            return sparse_residual_metal(x2, y32, self.res_row_ptr, self.res_cols, self.res_val, self.res_scale)
        # reference path: row ids rebuilt on the host per call (A/B and tests only)
        rows = mx.array(np.repeat(np.arange(self.out_features, dtype=np.int32), np.diff(np.array(self.res_row_ptr))))
        contrib = x2[:, self.res_cols].astype(mx.float32) * (self.res_val.astype(mx.float32) * self.res_scale)
        return y32.at[:, rows].add(contrib)

    def from_rotated(self, xr: mx.array, x2: mx.array | None = None) -> mx.array:
        """y = Uᵀ(W' x̃) + R x2 (+ bias), summed in float32 and rounded once to xr.dtype. x2 is required (R acts on
        the unrotated input)."""
        if x2 is None:
            raise ValueError("SucResidualLinear.from_rotated needs the unrotated input x2")
        y32 = self.kernel(xr, mx.float32)
        if self.seed_out is not None:
            y32 = self.rotate_out(y32, mx.float32)
        if self._structured() and "bias" not in self:
            return self.residual(x2, y32, xr.dtype)     # one rounding, inside the residual kernel
        y32 = self.residual(x2, y32)
        if "bias" in self:
            y32 = y32 + self.bias.astype(mx.float32)
        return y32.astype(xr.dtype)


def _is_residual(e: Encoded) -> bool:
    return "res_pos" in e.arrays and hasattr(e, "base")


def module_from_encoded(e: Encoded, bias: mx.array | None = None) -> SucLinear:
    """SucLinear (vq / additive / tcq) or SucResidualLinear (ResidualCodec over them) for one Encoded; a native
    MXFP4 pass-through (codec "mxfp4") runs from its stored codes as suc.kernels.mxfp4.MXFP4Linear."""
    if e.codec == "mxfp4":
        from suc.kernels.mxfp4 import MXFP4Linear

        return MXFP4Linear.from_encoded(e, bias=bias)
    if _is_residual(e):
        base = e.base()
        lin = SucLinear.from_encoded(base)   # validates the base kind, packs the indices
        a = e.arrays
        m = SucResidualLinear(None if lin.tcq is not None else lin.codebooks, lin.packed_indices, lin.row_scales,
                              bits=lin.bits, in_features=lin.in_features, seed_in=lin.seed_in, seed_out=lin.seed_out,
                              bias=bias, tcq=base.params if lin.tcq is not None else None,
                              res_pos=a["res_pos"], res_val=a["res_val"], res_scale=a.get("res_scale"),
                              res_cols=a.get("res_cols"), res_colval=a.get("res_colval"),
                              res_colscale=a.get("res_colscale"))
        return m
    m = SucLinear.from_encoded(e)
    if bias is not None:
        m.bias = bias
    return m


def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    """(parent module, attribute) of the Linear holding checkpoint tensor `name` ("….weight")."""
    if not name.endswith(".weight"):
        raise ValueError(f"{name!r} is not a weight tensor name")
    *path, attr = name[: -len(".weight")].split(".")
    cur = model
    try:
        for p in path:
            cur = cur[int(p)] if isinstance(cur, list) else cur[p]
        mod = cur[attr]
    except (KeyError, IndexError, ValueError, TypeError):
        raise KeyError(f"tensor {name!r} not found in model") from None
    if not isinstance(mod, nn.Module):
        raise KeyError(f"tensor {name!r} does not belong to a module")
    return cur, attr


def expert_stack(name: str) -> tuple[str, int] | None:
    """("<p>.switch_mlp.<proj>.weight", e) for an expert tensor "<p>.experts.{e}.<proj>.weight", else None."""
    if (m := _EXPERT.match(name)) is None:
        return None
    return f"{m.group(1)}.switch_mlp.{m.group(3)}.weight", int(m.group(2))


def _expert_targets(model: nn.Module, encodings: dict[str, Encoded]) -> list[tuple]:
    """[(stack name, [expert names in order], parent, attr)] for the expert tensors of `encodings`."""
    from mlx_lm.models.switch_layers import SwitchLinear
    stacks: dict[str, dict[int, str]] = {}
    for name in encodings:
        if (st := expert_stack(name)) is not None:
            stacks.setdefault(st[0], {})[st[1]] = name
    out = []
    for stack, rows in stacks.items():
        try:
            parent, attr = _parent(model, stack)
        except KeyError:
            raise ValueError(f"{stack!r}: expert tensors given but the model has no MoE expert stack there") from None
        old = parent[attr]
        if type(old) is not SwitchLinear:
            raise TypeError(f"{stack!r}: expected mlx-lm SwitchLinear, found {type(old).__name__}")
        n_exp, out_f, in_f = old.weight.shape
        if sorted(rows) != list(range(n_exp)):
            missing = sorted(set(range(n_exp)) - set(rows))
            raise ValueError(f"{stack!r}: all {n_exp} experts must be converted together; missing experts "
                             f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
        for e, n in rows.items():
            if tuple(encodings[n].shape) != (out_f, in_f):
                raise ValueError(f"{n!r}: Encoded shape {tuple(encodings[n].shape)} != expert weight {(out_f, in_f)}")
        out.append((stack, [rows[e] for e in range(n_exp)], parent, attr))
    return out


def _nbytes(m: nn.Module) -> int:
    return sum(v.nbytes for _, v in tree_flatten(m.parameters()))


def resident_weight_bytes(model: nn.Module) -> int:
    """Bytes of every array in the module tree (parameters incl. packed codes, scales, signs, residuals)."""
    return _nbytes(model)


def _plain(m: SucLinear) -> SucLinear:
    """A plain SucLinear sharing m's code arrays (no residual, no bias), for fusable_key / SucFusedLinear."""
    return SucLinear(m.codebooks, m.packed_indices, m.row_scales, bits=m.bits, in_features=m.in_features,
                     seed_in=m.seed_in, seed_out=m.seed_out)


def group_fusable_key(m: SucLinear) -> tuple | None:
    """fusable_key, extended to SucResidualLinear members (fused into a SucFusedResidual) when FUSE_RESIDUAL."""
    if m.tcq is not None:
        return None                     # trellis-coded projections run their own launch (no fused tcq kernel)
    if type(m) is SucResidualLinear and FUSE_RESIDUAL and "bias" not in m and m.residual_path == "metal":
        return fusable_key(_plain(m))   # residual and plain members fuse together (SucFusedResidual)
    return fusable_key(m)


class SucFusedResidual(SucFusedLinear):
    """SucFusedLinear whose members (SucResidualLinear or plain SucLinear) carry residuals: one SUC launch, one
    output-rotation launch (float32, unsplit) and ONE structured residual launch for all members, then a single
    rounding and the split. Member residual arrays are concatenated (memory unchanged, + a few bytes of meta)."""

    def __init__(self, members: list[SucLinear]):
        super().__init__([_plain(m) for m in members])
        inn, sizes = self.in_features, self.sizes
        S = len(members)
        res = [m for m in members if isinstance(m, SucResidualLinear)]
        vdt = {m.res_val.dtype for m in res if m.n_residual}
        cdt = {m.colres_val.dtype for m in res if m.n_columns}
        if len(vdt) > 1 or len(cdt) > 1:
            raise ValueError(f"members store residual values in different dtypes: {vdt | cdt}")
        row_start = np.concatenate([[0], np.cumsum(sizes)]).astype(np.uint32)
        nc = [m.n_columns if isinstance(m, SucResidualLinear) else 0 for m in members]
        col_ptr = np.concatenate([[0], np.cumsum(nc)]).astype(np.uint32)
        col_off = np.concatenate([[0], np.cumsum([c * o for c, o in zip(nc, sizes)])[:-1]]).astype(np.uint32)
        self.colres_meta = mx.array(np.concatenate([row_start, col_ptr, col_off]).astype(np.uint32))
        self.n_columns = int(col_ptr[-1])
        if self.n_columns:
            with_cols = [m for m in members if isinstance(m, SucResidualLinear) and m.n_columns]
            self.colres_cols = mx.concatenate([m.colres_cols.astype(mx.uint32) for m in with_cols]).astype(_index_dtype(inn))
            self.colres_val = mx.concatenate([m.colres_val.reshape(-1) for m in with_cols])
            self.colres_scale = mx.concatenate([m.colres_scale for m in with_cols])
        self.n_residual = sum(m.n_residual for m in res)
        if self.n_residual:
            counts, cols, vals = [], [], []
            for m, o in zip(members, sizes):
                if isinstance(m, SucResidualLinear) and m.n_residual:
                    counts.append(np.diff(np.array(m.res_row_ptr)).astype(np.int64))
                    cols.append(m.res_cols.astype(mx.uint32))
                    vals.append(m.res_val)
                else:
                    counts.append(np.zeros(o, np.int64))
            self.res_row_ptr = mx.array(np.concatenate([[0], np.cumsum(np.concatenate(counts))]).astype(np.uint32))
            self.res_cols = mx.concatenate(cols).astype(_index_dtype(inn))
            self.res_val = mx.concatenate(vals)
        self.res_scale = mx.concatenate([m.res_scale.reshape(1) if isinstance(m, SucResidualLinear)
                                         else mx.ones((1,)) for m in members]).astype(mx.float32)
        self._segments = _Ref(S)

    def residual(self, x2: mx.array, y32: mx.array, out_dtype: mx.Dtype = mx.float32) -> mx.array:
        if not self.n_columns and not self.n_residual:
            return y32.astype(out_dtype)
        cols = self.n_columns > 0
        csr = self.n_residual > 0
        return structured_residual_metal(x2, y32, self.colres_meta, self.colres_cols if cols else None,
                                         self.colres_val if cols else None, self.colres_scale if cols else None,
                                         self.res_row_ptr if csr else None, self.res_cols if csr else None,
                                         self.res_val if csr else None, self.res_scale, self._segments.obj,
                                         out_dtype, row_quads=all(n % 4 == 0 for n in self.sizes))

    def from_rotated_split(self, xr: mx.array, x2: mx.array | None = None, residual: bool = True) -> list[mx.array]:
        """One (B, out_s) output per member in xr.dtype; x2 (the unrotated input) is required with residual."""
        if not residual:
            return super().from_rotated_split(xr)
        if x2 is None:
            raise ValueError("SucFusedResidual.from_rotated_split needs the unrotated input x2")
        y32 = self.kernel(xr, mx.float32)
        if self.seed_out is not None:
            y32 = _rot.rht_inverse_fused(y32, self.signs_out, self.sizes, out_dtype=mx.float32)
        y = self.residual(x2, y32, xr.dtype)
        return [y[:, o - n:o] for o, n in zip(np.cumsum(self.sizes), self.sizes)]

    def from_rotated(self, xr: mx.array, x2: mx.array | None = None) -> mx.array:
        return mx.concatenate(self.from_rotated_split(xr, x2), axis=-1)


# ------------------------------------------------------------------------------------ shared-input projections
# Projections of one parent module that read the same input (mlx-lm Qwen3 / Llama-style blocks):
#   self_attn: q = q_proj(x), k = k_proj(x), v = v_proj(x);   mlp: down_proj(silu(gate_proj(x)) * up_proj(x)).
INPUT_GROUPS: dict[str, tuple[str, ...]] = {"qkv": ("q_proj", "k_proj", "v_proj"), "gate_up": ("gate_proj", "up_proj")}


class _Ref:
    """Plain (non-Module) reference, so a member view does not re-register its group in the module tree."""
    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj


class _Cache:
    __slots__ = ("x", "ys", "pending")

    def __init__(self):
        self.x, self.ys, self.pending = None, None, set()


class SucInputGroup(nn.Module):
    """SUC projections that read the same input x with the same input rotation V (README §5).

    One call computes x̃ = V x ONCE and the outputs of all members: members with the same code layout are
    row-concatenated into one SucFusedLinear (one SUC launch + one segmented Uᵀ launch for all of them); the others
    (residual / biased / different layout) run from_rotated(x̃, x) on the shared x̃.

    The original attributes (q_proj, …) become SucGroupMember views; the first view called with an input x
    computes every member's output for that x, the other views take theirs from a one-entry cache. The cache is
    keyed on the identity of x (`is`, holding a strong reference so the id cannot be reused) and each output is
    handed out once: the entry is dropped when all members have taken theirs, and a view called with another
    input — or twice with the same one — recomputes. A cache hit therefore always returns exactly what the member
    would compute itself; misuse costs time, never correctness. This was preferred over rewriting the attention /
    MLP __call__ of each mlx-lm architecture (fragile across model files and mlx-lm versions)."""

    def __init__(self, members: dict[str, SucLinear], fuse: bool = True):
        super().__init__()
        mods = list(members.values())
        if len(mods) < 2:
            raise ValueError("a group needs at least two members")
        if len({(m.seed_in, m.in_features) for m in mods}) != 1:
            raise ValueError("group members must share in_features and seed_in")
        clusters: dict[tuple, list[int]] = {}
        for i, m in enumerate(mods):
            k = group_fusable_key(m) if fuse else None
            clusters.setdefault(("fused", k) if k is not None else ("single", i), []).append(i)
        parts, layout = [], []          # layout[j] = member indices served by parts[j]
        for (kind, _), idx in clusters.items():
            if kind == "fused" and len(idx) >= 2:
                ms = [mods[i] for i in idx]
                parts.append(SucFusedResidual(ms) if any(isinstance(m, SucResidualLinear) for m in ms)
                             else SucFusedLinear(ms))
                layout.append(tuple(idx))
            else:
                parts.extend(mods[i] for i in idx)
                layout.extend((i,) for i in idx)
        self.parts = parts
        # metadata outside the parameter tree (plain attributes of a non-Module holder)
        self._meta = _Ref({"names": tuple(members), "in_features": mods[0].in_features,
                           "out_sizes": tuple(m.out_features for m in mods), "layout": tuple(layout)})
        self.mode = "full"          # "full" | "kernel_only" | "no_residual" | "null" (benchmarks/e2e_generate.py profile)
        self._cache = _Cache()
        self._pre = _Ref(None)      # (x, x̃) offered by a RotatedRMSNorm

    @property
    def names(self) -> tuple[str, ...]:
        return self._meta.obj["names"]

    @property
    def in_features(self) -> int:
        return self._meta.obj["in_features"]

    @property
    def out_sizes(self) -> tuple[int, ...]:
        return self._meta.obj["out_sizes"]

    @property
    def fused_parts(self) -> int:
        return sum(isinstance(p, SucFusedLinear) for p in self.parts)

    def outputs(self, x: mx.array | None, xr: mx.array | None = None) -> list[mx.array]:
        """Every member's output for x (..., in), in member order. xr: x̃ = V x already computed by the producer of
        x (RotatedRMSNorm); x may then be None when no part reads the unrotated input (needs_input())."""
        lead = (x if x is not None else xr).shape[:-1]
        x2 = x.reshape(-1, self.in_features) if x is not None else None
        if xr is not None:
            xr = xr.reshape(-1, self.in_features)
        elif self.mode != "kernel_only":
            xr = self.parts[0].rotate_in(x2)    # all parts share V
        else:
            xr = x2
        ys: list = [None] * len(self.names)
        for part, idx in zip(self.parts, self._meta.obj["layout"]):
            if self.mode == "kernel_only":
                y = part.kernel(xr)
                outs = [y[:, o - n:o] for o, n in zip(np.cumsum([self.out_sizes[i] for i in idx]),
                                                       [self.out_sizes[i] for i in idx])]
            elif isinstance(part, SucFusedResidual):
                outs = part.from_rotated_split(xr, x2, residual=self.mode != "no_residual")
            elif isinstance(part, SucFusedLinear):
                outs = part.from_rotated_split(xr)
            elif self.mode == "no_residual" and isinstance(part, SucResidualLinear):
                outs = [SucLinear.from_rotated(part, xr, x2)]
            else:
                outs = [part.from_rotated(xr, x2)]
            for i, y in zip(idx, outs):
                ys[i] = y.reshape(*lead, self.out_sizes[i])
        return ys

    # ------------------------------------------------------------ pre-rotated input (RotatedRMSNorm, K6)
    def prerotatable(self) -> bool:
        """The shared input rotation can be computed by the producer of x (the mode rotates and V exists)."""
        p = self.parts[0]
        return self.mode in ("full", "no_residual") and p.seed_in is not None and _rot.supported(self.in_features)

    def needs_input(self) -> bool:
        """Some part reads the unrotated input x (residuals, in the current mode)."""
        return self.mode == "full" and any(isinstance(p, (SucResidualLinear, SucFusedResidual)) for p in self.parts)

    @property
    def signs_in(self) -> mx.array:
        return self.parts[0].signs_in

    def offer(self, x: mx.array, xr: mx.array) -> None:
        """x̃ = V x for the next call with exactly this x (identity); consumed by that call."""
        self._pre.obj = (x, xr)

    def _take(self, x: mx.array) -> mx.array | None:
        pre = self._pre.obj
        if pre is not None and pre[0] is x:
            self._pre.obj = None
            return pre[1]
        return None

    def _outputs_full(self, x: mx.array) -> list[mx.array]:
        xr = self._take(x)
        compile_ = suc_linear.COMPILE and all(p.compilable() for p in self.parts)
        key = ("outputs", suc_linear.ROTATION, self.mode)
        if xr is not None:
            if self.needs_input():
                f, args, key = self.outputs, (x, xr), key + ("pre_x",)
            else:
                f, args, key = (lambda r: self.outputs(None, r)), (xr,), key + ("pre",)
            return compiled_call(self, f, key, *args) if compile_ else f(*args)
        if compile_:
            return compiled_call(self, self.outputs, key, x)
        return self.outputs(x)

    def output(self, x: mx.array, i: int) -> mx.array:
        c = self._cache
        if c.x is x and i in c.pending:
            c.pending.discard(i)
            y = c.ys[i]
            if not c.pending:
                c.x, c.ys = None, None
            return y
        ys = self._outputs_full(x)
        c.x, c.ys, c.pending = x, ys, set(range(len(ys))) - {i}
        return ys[i]

    def __call__(self, x: mx.array) -> list[mx.array]:
        return self._outputs_full(x)


class SucGroupMember(nn.Module):
    """Stand-in for one projection of a SucInputGroup (same call signature as the nn.Linear it replaced)."""

    def __init__(self, group: SucInputGroup, index: int):
        super().__init__()
        self._group = _Ref(group)
        self.index = index
        self.in_features, self.out_features = group.in_features, group.out_sizes[index]

    @property
    def group(self) -> SucInputGroup:
        return self._group.obj

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected (..., {self.in_features}) input, got {x.shape}")
        return self.group.output(x, self.index)


def group_projections(model: nn.Module, *, fuse: bool = True, elementwise: bool = True) -> list[dict]:
    """Wrap every INPUT_GROUPS set of SucLinear siblings that share (in_features, seed_in) — e.g. encoded with
    suc.evaluation.reconstruct.rotation_seeds — into a SucInputGroup stored as parent.suc_<group> (q_proj, … become
    SucGroupMember views). Members with other seeds / types are left alone; ≥ 2 members form a group (so v_proj
    alone in BF16 still lets q/k share). With elementwise (and FUSE_ELEMENTWISE), the input rotations are then
    fused into the norms / SwiGLU (fuse_elementwise_rotations). Returns [{parent, group, members, fused_parts,
    parts}]."""
    out = []
    for path, parent in list(model.named_modules()):
        for gname, attrs in INPUT_GROUPS.items():
            if f"suc_{gname}" in parent:
                continue
            cand = {a: parent[a] for a in attrs if a in parent and isinstance(parent[a], SucLinear)}
            by_key: dict[tuple, dict[str, SucLinear]] = {}
            for a, m in cand.items():
                by_key.setdefault((m.in_features, m.seed_in), {})[a] = m
            members = max(by_key.values(), key=len, default={})
            if len(members) < 2:
                continue
            g = SucInputGroup(members, fuse=fuse)
            mx.eval(g.parameters())     # materialize the concatenated codes (else the first call builds them)
            parent[f"suc_{gname}"] = g
            for i, a in enumerate(members):
                parent[a] = SucGroupMember(g, i)
            out.append({"parent": path, "group": gname, "members": list(members), "fused_parts": g.fused_parts,
                        "parts": len(g.parts)})
    if elementwise and fuse_elementwise():
        fuse_elementwise_rotations(model)
    return out



# ----------------------------------------------------------------- input rotations fused into element-wise ops (K6)
# The online input rotation of a projection is moved into the launch of the element-wise op that produces its input
# (suc.kernels.rotation.rms_norm_rht / swiglu_rht; no change in math: the op's T-valued output is rotated in float32
# and rounded to T, exactly as before):
#   input_layernorm → q|k|v,  post_attention_layernorm → gate|up   (RotatedRMSNorm: x̃ handed to the SucInputGroup)
#   silu(gate)·up → down_proj                                        (SucMLP)
# The unrotated op output is written by the same launch only when a residual reads it (column / CSR residuals act on
# the original basis). Every path falls back to the unfused computation when the hand-over does not happen (other
# input, profile modes), so a miss costs time, never correctness.
# None (default): per device from suc.kernels.autotune ("fuse_elementwise", measured in decode context), else off —
# on the M1 Ultra (Qwen3-8B, K6) the fused norm/SwiGLU + rotation kernels of the single-threadgroup layout were
# slower than separate launches (VQ G=4 B=1: 43.4 vs 47.3 tok/s; rotation 8.3 vs 6.8 ms/step). True / False force it.
FUSE_ELEMENTWISE: bool | None = None


def fuse_elementwise(op: str | None = None) -> bool:
    """Whether the input rotation is fused into op ("norm" | "swiglu"; None: into either) — FUSE_ELEMENTWISE, or
    per device and op from the autotune cache when it is None (default off)."""
    if FUSE_ELEMENTWISE is not None:
        return bool(FUSE_ELEMENTWISE)
    ops = ("norm", "swiglu") if op is None else (op,)
    return any(bool(autotune.fuse_elementwise_choice(o)) for o in ops)

_SWIGLU_CALL = "returnself.down_proj(swiglu(self.gate_proj(x),self.up_proj(x)))"


class RotatedRMSNorm(nn.Module):
    """nn.RMSNorm (same weight / eps, same parameter path) whose output x feeds a SucInputGroup: one launch computes
    rms_norm(x) and x̃ = V rms_norm(x) and offers x̃ to the group (identity-keyed, consumed by its next call)."""

    def __init__(self, norm: nn.Module, group: SucInputGroup):
        super().__init__()
        self.weight = norm.weight
        self.eps = float(norm.eps)
        self._group = _Ref(group)

    def __call__(self, x: mx.array) -> mx.array:
        g = self._group.obj
        if not (fuse_elementwise("norm") and g.prerotatable() and x.shape[-1] == g.in_features):
            return mx.fast.rms_norm(x, self.weight, self.eps)
        need = g.needs_input()

        def f(x_):
            xr_, xn_ = _rot.rms_norm_rht(x_, self.weight, self.eps, g.signs_in, with_input=need)
            return [xr_, xn_] if need else [xr_]
        outs = compiled_call(self, f, ("norm_rht", need), x) if suc_linear.COMPILE else f(x)
        xr = outs[0]
        xn = outs[1] if need else mx.fast.rms_norm(x, self.weight, self.eps)   # lazy: evaluated only if used
        g.offer(xn, xr)
        return xn


def _is_swiglu_mlp(m: nn.Module) -> bool:
    """An mlx-lm MLP whose __call__ is exactly down_proj(swiglu(gate_proj(x), up_proj(x))) (Qwen3 / Llama / Qwen2)."""
    import inspect
    try:
        src = inspect.getsource(type(m).__call__)
    except (OSError, TypeError):
        return False
    body = "".join(src.split("\n", 1)[1].split()) if "\n" in src else ""
    return body == _SWIGLU_CALL


class SucMLP(nn.Module):
    """Stand-in for a SwiGLU MLP (same children, same parameter paths) whose down_proj runs on SUC codes: silu(g)·u
    and down_proj's input rotation in one launch. mode: "full" | "no_residual" | "kernel_only" (profile)."""

    def __init__(self, mlp: nn.Module):
        super().__init__()
        for k, v in mlp.items():
            self[k] = v
        self.mode = "full"

    def _fusable(self, d) -> bool:
        return (fuse_elementwise("swiglu") and isinstance(d, SucLinear) and d.seed_in is not None
                and _rot.supported(d.in_features) and "bias" not in d and d.compilable())

    def __call__(self, x: mx.array) -> mx.array:
        from mlx_lm.models.activations import swiglu
        gate, up = self.gate_proj(x), self.up_proj(x)
        d = self.down_proj
        if self.mode == "kernel_only" and isinstance(d, SucLinear):
            h = swiglu(gate, up)
            y = d.kernel(h.reshape(-1, d.in_features))
            return y.reshape(*h.shape[:-1], d.out_features)
        if not self._fusable(d) or self.mode not in ("full", "no_residual"):
            return d(swiglu(gate, up))
        need = isinstance(d, SucResidualLinear) and self.mode == "full"
        lead = gate.shape[:-1]

        def f(g_, u_):
            hr, h = _rot.swiglu_rht(g_.reshape(-1, d.in_features), u_.reshape(-1, d.in_features), d.signs_in,
                                    with_input=need)
            return [d.from_rotated(hr, h) if need else SucLinear.from_rotated(d, hr)]
        y = compiled_call(d, f, ("swiglu_rht", need, suc_linear.ROTATION), gate, up)[0] if suc_linear.COMPILE \
            else f(gate, up)[0]
        return y.reshape(*lead, d.out_features)


def fuse_elementwise_rotations(model: nn.Module) -> dict:
    """Install RotatedRMSNorm on the norms feeding q|k|v and gate|up groups and SucMLP on SwiGLU MLPs whose down_proj
    is a rotated SucLinear (call after group_projections). Returns {"norms": n, "mlps": n}."""
    norms = mlps = 0
    for _, layer in list(model.named_modules()):
        for norm_attr, child_attr, gname in (("input_layernorm", "self_attn", "suc_qkv"),
                                             ("post_attention_layernorm", "mlp", "suc_gate_up")):
            if norm_attr not in layer or child_attr not in layer:
                continue
            norm, child = layer[norm_attr], layer[child_attr]
            if type(norm) is not nn.RMSNorm or not isinstance(child, nn.Module) or gname not in child:
                continue
            g = child[gname]
            if isinstance(g, SucInputGroup) and g.parts[0].seed_in is not None and _rot.supported(g.in_features):
                layer[norm_attr] = RotatedRMSNorm(norm, g)
                norms += 1
        if "mlp" in layer and isinstance(layer["mlp"], nn.Module) and type(layer["mlp"]) is not SucMLP:
            mlp = layer["mlp"]
            d = mlp.get("down_proj") if "down_proj" in mlp else None
            if isinstance(d, SucLinear) and _is_swiglu_mlp(mlp):
                layer["mlp"] = SucMLP(mlp)
                mlps += 1
    return {"norms": norms, "mlps": mlps}

def _tuning_shapes(model: nn.Module, encodings: dict[str, Encoded]) -> dict:
    """Sizes the per-device kernel tuning measures for this model: online rotation sizes (inputs of rotated
    tensors, outputs of two-sided ones) and hidden / intermediate sizes (RMSNorm / SwiGLU fusion)."""
    sizes = set()
    for e in encodings.values():
        p = e.params if isinstance(e.params, dict) else {}
        p = p.get("base", p) if isinstance(p.get("base"), dict) else p
        if p.get("rotate", False):
            if p.get("rotate_in", True):
                sizes.add(int(e.shape[-1]))
            if p.get("rotate_out", True):
                sizes.add(int(e.shape[-2]))
    args = getattr(model, "args", None)
    hidden, inter = getattr(args, "hidden_size", None), getattr(args, "intermediate_size", None)
    return {"rotation_sizes": tuple(sorted(sizes)), "hidden": hidden, "intermediate": inter}


def convert_to_compressed(model: nn.Module, encodings: dict[str, Encoded], *, clear_cache: bool = True,
                          group: bool = True, fuse: bool = True) -> dict:
    """Replace each nn.Linear named in `encodings` (checkpoint names "….weight") by a SUC module, in place.

    Only nn.Linear modules are replaced (QuantizedLinear / Embedding are refused); the shape of the Encoded must
    match the weight. MoE expert tensors ("…experts.{e}.<proj>.weight", every expert of a stack) replace their
    mlx-lm SwitchLinear by a SucSwitchLinear (one report row per stack, with "experts"). The original weights are
    dropped. With group=True, projections that read the same input
    with the same seed_in are then wrapped by group_projections (shared rotation; fuse=True also row-concatenates
    members with equal code layout into one launch). Call group_projections(model) once at the end instead when
    converting tensor by tensor (group=False per call). Returns
    {"converted": [{name, codec, kind, shape, bytes_before, bytes_after, n_residual}], "tensors_converted",
     "bytes_before", "bytes_after" (of the replaced modules only), "model_bytes_before", "model_bytes_after",
     "active_memory_before", "active_memory_after" (mx.get_active_memory, after mx.clear_cache)}.
    """
    autotune.ensure_tuned(**_tuning_shapes(model, encodings))   # once per device / size, outside compiled graphs
    targets = []
    switch_targets = _expert_targets(model, encodings)
    for name, e in encodings.items():
        if expert_stack(name) is not None:
            continue
        parent, attr = _parent(model, name)
        old = parent[attr]
        if type(old) is not nn.Linear:
            raise TypeError(f"{name!r}: expected nn.Linear, found {type(old).__name__}")
        if tuple(old.weight.shape) != tuple(e.shape):
            raise ValueError(f"{name!r}: Encoded shape {tuple(e.shape)} != weight {tuple(old.weight.shape)}")
        targets.append((name, e, parent, attr))
    model_before = _nbytes(model)
    if clear_cache:
        gc.collect()
        mx.clear_cache()
    act_before = mx.get_active_memory()
    rows, before, after = [], 0, 0
    for name, e, parent, attr in targets:
        old = parent[attr]
        b0 = _nbytes(old)
        new = module_from_encoded(e, bias=old.bias if "bias" in old else None)
        mx.eval(new.parameters())
        parent[attr] = new
        del old
        b1 = _nbytes(new)
        before, after = before + b0, after + b1
        rows.append({"name": name, "codec": e.codec, "kind": type(new).__name__, "shape": list(e.shape),
                     "bytes_before": b0, "bytes_after": b1,
                     "n_residual": getattr(new, "n_residual", 0)})
    for stack, enames, parent, attr in switch_targets:
        old = parent[attr]
        b0 = _nbytes(old)
        new = switch_from_encodings([encodings[n] for n in enames], bias=old.get("bias"))
        mx.eval(new.parameters())
        parent[attr] = new
        del old
        b1 = _nbytes(new)
        before, after = before + b0, after + b1
        e0 = encodings[enames[0]]
        rows.append({"name": stack, "codec": e0.codec, "kind": type(new).__name__,
                     "shape": [len(enames), *e0.shape], "experts": len(enames),
                     "bytes_before": b0, "bytes_after": b1,
                     "n_residual": sum(m.n_residual for m in getattr(new, "subs", [new]))})
    groups = group_projections(model, fuse=fuse) if group else []
    if clear_cache:
        gc.collect()
        mx.clear_cache()
    return {"converted": rows, "groups": groups, "tensors_converted": sum(r.get("experts", 1) for r in rows),
            "bytes_before": before, "bytes_after": after,
            "model_bytes_before": model_before, "model_bytes_after": _nbytes(model),
            "active_memory_before": act_before, "active_memory_after": mx.get_active_memory()}


def merge_reports(reports: list[dict]) -> dict:
    """Combine per-call reports of convert_to_compressed (incremental, tensor-by-tensor conversion)."""
    if not reports:
        return {"converted": [], "tensors_converted": 0, "bytes_before": 0, "bytes_after": 0}
    return {"converted": [r for rep in reports for r in rep["converted"]],
            "tensors_converted": sum(r["tensors_converted"] for r in reports),
            "bytes_before": sum(r["bytes_before"] for r in reports),
            "bytes_after": sum(r["bytes_after"] for r in reports),
            "model_bytes_before": reports[0]["model_bytes_before"],
            "model_bytes_after": reports[-1]["model_bytes_after"],
            "active_memory_before": reports[0]["active_memory_before"],
            "active_memory_after": reports[-1]["active_memory_after"]}
