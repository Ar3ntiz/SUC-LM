"""Sparse residual / outlier codec (spec §13, README §2): W = W_base + R, R kept on a sparse support.

After any base codec, R = w - decode(encode(w)) (float32). n = round(fraction * N) entries of R are
stored exactly enough to matter, chosen by
  magnitude   largest |R|
  importance  largest F ⊙ R², F = per-weight importance (empirical Fisher, same shape as W): the
              entries whose correction gives the largest predicted loss reduction ½ F R².
Stored arrays: "res_pos" flat row-major positions (uint32, sorted, unique) and "res_val" (fp16, or
int8 symmetric with ONE fp16 scale per tensor in "res_scale", scale = max|R_sel| / 127). The base
Encoded is kept with prefixed array keys ("base.<key>") and params["base"] / params["base_codec"].

Bit accounting (bits()["residual"], the ACTUAL fixed-width layout):
  n * (ceil(log2 N) + value_bits) (+ 16 for the int8 scale when n > 0); all other parts are the base's.
params also reports an entropy-coded estimate (NOT counted in bits()): delta-coded sorted positions
with a geometric gap distribution cost ≈ log2(N/n) + 1.44 bits per entry (Golomb/Elias-style
bound), giving params["residual_bits_entropy"]. That is a storage figure; the execution layout of a
kernel would use the fixed-width positions (spec §15).

Column-structured residuals (structure="columns", docs/benchmarks.md Q5b). Activation-selected residual entries
concentrate on the few input columns j with a huge E[x_j²] (massive-activation channels), and the residual lives in
the ORIGINAL (unrotated) weight space, so they can be stored as a dense block instead of a CSR scatter:
  B        the bit budget of the CSR residual of the same fraction (n * (position_bits + value_bits) + 16)
  c        floor(column_share * B / column_cost) input columns, column_cost = out * value_bits
           + ceil(log2 in) (column index) + 16 (one fp16 scale per column, int8 only)
  columns  the c columns with the largest Σ_r score_rj (score as for the CSR selection), sorted ascending:
           "res_cols" (c,) uint32, "res_colval" (c, out) int8 (column-major: column j's residual R[:, cols[j]])
           with "res_colscale" (c,) fp16 = max|R[:, j]| / 127 (or fp16 values without a scale)
  tail     with column_share < 1, the rest of the budget is a CSR tail (res_pos / res_val / res_scale as above)
           selected on R minus the dequantized column part; with column_share = 1 there is no tail (n = 0).
The total never exceeds B. Runtime: y += R_c · x[cols] (a small dense GEMV on gathered inputs) + the CSR tail.
The default structure="csr" leaves names, params and arrays exactly as before.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from suc.quant.codecs import Codec, Encoded, make_codec

_VALUE_BITS = {"fp16": 16, "int8": 8}
_PREFIX = "base."


def _int8_sym(v: mx.array, axis: int | None = None) -> tuple[mx.array, mx.array]:
    """Symmetric int8 codes and fp16 scale(s) max|v| / 127 (one per tensor, or one per slice along `axis`)."""
    m = mx.max(mx.abs(v)) if axis is None else mx.max(mx.abs(v), axis=axis)
    s16 = (m / 127).astype(mx.float16)
    s = s16.astype(mx.float32)
    if axis is not None:
        s = mx.expand_dims(s, axis)
    return mx.clip(mx.round(v / mx.where(s > 0, s, 1.0)), -127, 127).astype(mx.int8), s16


def column_residual(arrays: dict, out: int) -> tuple[mx.array, mx.array] | None:
    """(cols (c,) int32, dequantized values (c, out) float32) of a column-structured residual, None without one."""
    if "res_cols" not in arrays or arrays["res_cols"].size == 0:
        return None
    v = arrays["res_colval"].astype(mx.float32).reshape(-1, out)
    if "res_colscale" in arrays:
        v = v * arrays["res_colscale"].astype(mx.float32)[:, None]
    return arrays["res_cols"].astype(mx.int32), v


@dataclass
class _ResidualEncoded(Encoded):
    def base(self) -> Encoded:
        arrays = {k[len(_PREFIX):]: v for k, v in self.arrays.items() if k.startswith(_PREFIX)}
        return Encoded(self.params["base_codec"], self.shape, self.params["base"], arrays)

    def bits(self) -> dict:
        out = self.base().bits()
        out["residual"] = int(out["residual"] + self.params["residual_bits"])
        out["total"] = sum(v for k, v in out.items() if k != "total")
        return out


class ResidualCodec:
    def __init__(self, base: Codec, fraction: float, value_dtype: Literal["fp16", "int8"] = "fp16",
                 selection: Literal["magnitude", "importance"] = "magnitude",
                 structure: Literal["csr", "columns"] = "csr", column_share: float = 1.0) -> None:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        if value_dtype not in _VALUE_BITS:
            raise ValueError(f"unknown value_dtype {value_dtype!r}")
        if selection not in ("magnitude", "importance"):
            raise ValueError(f"unknown selection {selection!r}")
        if structure not in ("csr", "columns"):
            raise ValueError(f"unknown residual structure {structure!r}")
        if not 0.0 < column_share <= 1.0:
            raise ValueError(f"column_share must be in (0, 1], got {column_share}")
        if structure == "csr" and column_share != 1.0:
            raise ValueError("column_share is only meaningful with structure='columns'")
        self.base, self.fraction, self.value_dtype, self.selection = base, fraction, value_dtype, selection
        self.structure, self.column_share = structure, column_share
        self.name = f"{base.name}+res{fraction:g}_{value_dtype}_{selection}"
        if structure == "columns":
            self.name += "_cols" + (f"{column_share:g}" if column_share != 1.0 else "")

    def encode(self, w: mx.array, *, seed: int, col_weights: mx.array | None = None,
               importance: mx.array | None = None, hessian: mx.array | None = None,
               base_encoded: Encoded | None = None, **rotation_seeds) -> Encoded:
        """rotation_seeds (seed_in= / seed_out=) are forwarded to the base codec's encode when given, and so are
        shared codebooks (codebooks= / codebook_owner=, suc.quant.stacks; vq / additive bases only). base_encoded:
        an encoding of w by the base codec computed elsewhere (e.g. rows of a batched trellis encode) is used
        instead of encoding w again."""
        if self.selection == "importance":
            if importance is None:
                raise ValueError("selection='importance' requires an importance array")
            if tuple(importance.shape) != tuple(w.shape):
                raise ValueError(f"importance shape {tuple(importance.shape)} != weight shape {tuple(w.shape)}")
        shared = {k: rotation_seeds.pop(k) for k in ("codebooks", "codebook_owner") if k in rotation_seeds}
        if set(rotation_seeds) - {"seed_in", "seed_out"}:
            raise TypeError(f"unexpected keyword arguments {sorted(set(rotation_seeds) - {'seed_in', 'seed_out'})}")
        rs = {k: v for k, v in rotation_seeds.items() if v is not None}
        if rs and "seed_in" not in inspect.signature(self.base.encode).parameters:
            rs = {}   # base without rotation (bf16 / affine / scalar): nothing to share
        if shared.get("codebooks") is not None:
            if "codebooks" not in inspect.signature(self.base.encode).parameters:
                raise TypeError(f"base codec {self.base.name!r} does not take shared codebooks")
            rs.update(shared)
        if hessian is not None:     # full input second moment (LDLQ bases only, suc.quant.trellis)
            if "hessian" not in inspect.signature(self.base.encode).parameters:
                raise TypeError(f"base codec {self.base.name!r} does not take a hessian")
            rs["hessian"] = hessian
        be = base_encoded if base_encoded is not None else self.base.encode(w, seed=seed, col_weights=col_weights, **rs)
        r2d = w.astype(mx.float32) - self.base.decode(be)
        out_f, in_f = r2d.shape
        big_n = r2d.size
        n = round(self.fraction * big_n)
        vbits = _VALUE_BITS[self.value_dtype]
        pbits = max(1, math.ceil(math.log2(big_n)))
        imp = None if self.selection == "magnitude" else importance.astype(mx.float32).reshape(out_f, in_f)
        arrays = {_PREFIX + k: v for k, v in be.arrays.items()}
        col_params, col_bits = {}, 0
        if self.structure == "columns":
            budget = n * (pbits + vbits) + (16 if self.value_dtype == "int8" and n else 0)
            cbits = max(1, math.ceil(math.log2(in_f)))
            col_cost = out_f * vbits + cbits + (16 if self.value_dtype == "int8" else 0)
            c = min(in_f, int(self.column_share * budget) // col_cost)
            col_score = mx.sum(r2d * r2d if imp is None else imp * r2d * r2d, axis=0)
            cols = mx.sort(mx.argpartition(-col_score, c - 1)[:c] if 0 < c < in_f else mx.arange(c)).astype(mx.uint32)
            block = r2d[:, cols.astype(mx.int32)].T                     # (c, out) column-major
            if self.value_dtype == "fp16":
                arrays["res_colval"] = block.astype(mx.float16)
                deq = arrays["res_colval"].astype(mx.float32)
            else:
                q, s16 = _int8_sym(block, axis=1)
                arrays["res_colval"], arrays["res_colscale"] = q, s16
                deq = q.astype(mx.float32) * s16.astype(mx.float32)[:, None]
            arrays["res_cols"] = cols
            if c:
                r2d = r2d.at[:, cols.astype(mx.int32)].add(-deq.T)     # what the columns leave for the tail
            col_bits = c * col_cost
            left = budget - col_bits
            n = max(0, (left - (16 if self.value_dtype == "int8" else 0)) // (pbits + vbits)) \
                if self.column_share < 1.0 else 0
            col_params = {"structure": "columns", "column_share": self.column_share, "columns": c,
                          "column_index_bits": cbits, "column_bits": col_bits, "budget_bits": budget}
        r = r2d.reshape(-1)
        score = mx.abs(r) if imp is None else imp.reshape(-1) * r * r
        pos = mx.sort(mx.argpartition(-score, n - 1)[:n] if 0 < n < big_n else mx.arange(n)).astype(mx.uint32)
        vals = r[pos]
        arrays["res_pos"] = pos
        extra = 0
        if self.value_dtype == "fp16":
            arrays["res_val"] = vals.astype(mx.float16)
        else:
            if n:
                q, s16 = _int8_sym(vals)
            else:
                q, s16 = vals.astype(mx.int8), mx.array(0.0).astype(mx.float16)
            arrays["res_val"], arrays["res_scale"] = q, s16
            extra = 16 if n else 0
        pbits_ent = math.log2(big_n / n) + 1.44 if n else 0.0
        params = {"base_codec": be.codec, "base": be.params, "fraction": self.fraction, "n": n,
                  "value_dtype": self.value_dtype, "selection": self.selection, "seed": seed,
                  "position_bits": pbits, "value_bits": vbits, "residual_bits": n * (pbits + vbits) + extra + col_bits,
                  "position_bits_entropy": pbits_ent,
                  "residual_bits_entropy": n * (pbits_ent + vbits) + extra + col_bits, **col_params}
        return _ResidualEncoded(self.name, be.shape, params, arrays)

    def residual_bits_for(self, shape) -> int:
        """params["residual_bits"] of an encoding of `shape` (out, in): it depends only on the shape (encode's
        budget arithmetic, which entries / columns are kept does not change the count)."""
        out_f, in_f = int(shape[0]), int(shape[1])
        big_n = out_f * in_f
        n = round(self.fraction * big_n)
        vbits = _VALUE_BITS[self.value_dtype]
        pbits = max(1, math.ceil(math.log2(big_n)))
        i8 = self.value_dtype == "int8"
        if self.structure != "columns":
            return n * (pbits + vbits) + (16 if i8 and n else 0)
        budget = n * (pbits + vbits) + (16 if i8 and n else 0)
        col_cost = out_f * vbits + max(1, math.ceil(math.log2(in_f))) + (16 if i8 else 0)
        c = min(in_f, int(self.column_share * budget) // col_cost)
        col_bits = c * col_cost
        tail = max(0, (budget - col_bits - (16 if i8 else 0)) // (pbits + vbits)) if self.column_share < 1.0 else 0
        return tail * (pbits + vbits) + (16 if i8 and tail else 0) + col_bits

    def bits_for(self, shape) -> int:
        """Encoded.bits()["total"] of an encoding of `shape` (base codecs with a data-independent bits_for)."""
        return int(self.base.bits_for(shape)) + self.residual_bits_for(shape)

    def decode(self, e: Encoded) -> mx.array:
        if e.codec != self.name:
            raise ValueError(f"cannot decode {e.codec!r} with codec {self.name!r}")
        if not isinstance(e, _ResidualEncoded):
            raise ValueError("not a residual Encoded")
        wb = self.base.decode(e.base())
        if (cr := column_residual(e.arrays, wb.shape[0])) is not None:
            wb = wb.at[:, cr[0]].add(cr[1].T)
        if e.params["n"] == 0:
            return wb
        v = e.arrays["res_val"].astype(mx.float32)
        if "res_scale" in e.arrays:
            v = v * e.arrays["res_scale"].astype(mx.float32)
        flat = wb.reshape(-1).at[e.arrays["res_pos"].astype(mx.int32)].add(v)
        return flat.reshape(wb.shape)


def residual_from_config(cfg: dict) -> ResidualCodec:
    """{"base": {make_codec cfg}, "fraction": float, "value_dtype"?: "fp16"|"int8", "selection"?: ...,
    "structure"?: "csr"|"columns", "column_share"?: float}."""
    cfg = dict(cfg)
    try:
        base = make_codec(cfg.pop("base"))
        return ResidualCodec(base, **cfg)
    except (KeyError, TypeError) as ex:
        raise ValueError(f"bad residual codec config: {ex}") from None
