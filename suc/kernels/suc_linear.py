"""SucLinear: drop-in nn.Linear running directly on SUC codes (README §5, K2/K4 of the kernel track).

Representation (suc.quant.codecs vq / additive): M fp16 codebooks C_m (K, G), bit-packed indices
(M, out, words) in the suc.quant.packing layout, row scales s (out,), optional rotation seeds. The stored
weight is W' = U W Vᵀ (suc.quant.hadamard.rotate_weight) with Ŵ'[r, g] = s_r · Σ_m C_m[i_{m,r,g}], so

    y = W x = Uᵀ (W' (V x)):   x̃ = V x = H (σ_in ⊙ x),   ỹ = W' x̃ (Metal kernel),   y = Uᵀ ỹ = σ_out ⊙ (Hᵀ ỹ).

Each side is optional: seed_in = None skips V, seed_out = None skips Uᵀ (codecs params rotate_in / rotate_out:
an output size without a Hadamard, e.g. a 151936-row LM head, is stored as W' = W Vᵀ only).

Rotations run in float32 with mx.hadamard_transform, which natively handles n = m · 2^k, m ∈ {1, 12, 20, 28}
and equals suc.quant.hadamard.hadamard up to float32 rounding (≤ ~1e-6 abs for unit inputs; exactly for powers
of two). Hᵀ = H only for m ∈ {1, 12} (Sylvester and MLX's H_12 are symmetric); for m ∈ {20, 28}, or when the
power-of-two factor exceeds MLX's float32 limit, the inverse falls back to suc.quant.hadamard.rht_inverse.
(The fallback is the reference transform, whose H_m step is the deterministic fma chain; the native
mx.hadamard_transform path differs from it by float32 rounding, ≤ ~2.3e-7 relative for 5120 / 7168 on the M3.)
Precision: the rotations are computed in float32; x̃ is then rounded to the activation dtype (like any
activation entering a dense layer) because a float32 x̃ makes the kernel 1.3x / 2x / 3x slower at B = 1 / 4 / 16
(twice the x bytes re-read per row; M3, 12288x4096); ỹ is written in float32 and rounded once, after Uᵀ.

Launches per call (ROTATION = "fused", default): rotate_in (1 Metal launch, suc.kernels.rotation: cast + signs +
H + cast), the SUC kernel, rotate_out (1 launch: Hᵀ + signs + cast) — 3 instead of the 8 of ROTATION = "mlx"
(cast·sign, hadamard_transform, cast on each side; kept for A/B and for sizes the fused kernel does not take).
Callers that share x̃ = V x between projections (q/k/v, gate/up: suc.runtime.compressed_model) call rotate_in once
and from_rotated(x̃) per projection.

SucFusedLinear concatenates several SucLinear with the same input, rotation V and code layout (bits, M, K, G)
along the output dimension: ONE SUC kernel launch (each row reads its own tensor's codebooks: per-segment codebook
offsets, see fused_matmul_metal) and ONE output-rotation launch whose segments are the members' own Uᵀ.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import suc.kernels.vq_matmul as _vqm
from suc.kernels import rotation as _rot
from suc.kernels.vq_matmul import additive_matmul_metal
from suc.quant.hadamard import _MLX_M, _factor, _transform, hadamard, random_signs
from suc.quant.packing import pack_indices, words_per_row

_MAX_POW2 = 8192  # mx.hadamard_transform float32 limit on the power-of-two factor
ROTATION = "fused"  # "fused" (suc.kernels.rotation, 1 launch per side) | "mlx" (fast_rht / fast_rht_inverse)
# mx.compile the per-call graph (module arrays as implicit inputs, so set_dtype / update are seen). Building the
# lazy graph of one SucLinear call costs ~18 µs of Python (M3), about as much as its GPU time at batch 1; a compiled
# call ~3.5 µs. Retraced per input shape / dtype and per ROTATION value.
COMPILE = True
# dtype of the stored rotation signs (±1, exact in any dtype): int8 = 1 byte per element — every blocks-layout
# rotation threadgroup reads the whole sign vector, so float32 signs were 2/3 of its loads (R5)
SIGN_DTYPE = mx.int8
# Fused (row-concatenated) kernels stage each threadgroup's segment codebooks in threadgroup memory when CODEBOOK_IN
# resolves to "threadgroup" and the segment boundaries allow it (False: always device memory, the earlier path; A/B).
FUSED_STAGING = True


class _Slot:
    """Plain attribute holder outside the module tree (compiled functions)."""
    __slots__ = ("fns",)

    def __init__(self):
        self.fns = {}


def compiled_call(module: nn.Module, fn, key, *args):
    """fn(*args) through an mx.compile cached on the module under key (module.state as implicit inputs)."""
    slot = module.__dict__.get("_suc_slot")
    if slot is None:
        slot = _Slot()
        object.__setattr__(module, "_suc_slot", slot)
    f = slot.fns.get(key)
    if f is None:
        f = slot.fns[key] = mx.compile(fn, inputs=module.state)
    return f(*args)


def _native(n: int, inverse: bool) -> bool:
    f = _factor(n)
    if f is None:
        raise ValueError(f"unsupported Hadamard size {n}: need n = m * 2^k with m in (1, 12, 20, 28, 17)")
    m, p = f
    # m = 17 (Hartley factor) is not an MLX Hadamard size: always the suc.quant.hadamard path
    return p <= _MAX_POW2 and m in _MLX_M and (not inverse or m in (1, 12))


def fast_rht(x: mx.array, signs: mx.array) -> mx.array:
    """V x = H (signs ⊙ x) on the last axis, float32 (== suc.quant.hadamard.rht with the same signs)."""
    n = x.shape[-1]
    xs = x.astype(mx.float32) * signs
    if _native(n, inverse=False):
        return mx.hadamard_transform(xs, scale=float(1.0 / np.sqrt(n)))
    return hadamard(xs)


def fast_rht_inverse(y: mx.array, signs: mx.array) -> mx.array:
    """Vᵀ y = signs ⊙ (Hᵀ y) on the last axis, float32 (== suc.quant.hadamard.rht_inverse)."""
    n = y.shape[-1]
    if _native(n, inverse=True):
        return mx.hadamard_transform(y.astype(mx.float32), scale=float(1.0 / np.sqrt(n))) * signs
    return _transform(y, transpose=True) * signs


class TcqMeta:
    """Code layout of a trellis-coded (tcq) SucLinear / SucSwitchLinear, kept outside the parameter tree."""
    __slots__ = ("k", "L", "V", "T", "code", "gain")

    def __init__(self, p: dict):
        self.k, self.L, self.V, self.T = float(p["k"]), int(p["L"]), int(p["V"]), int(p["T"])
        self.code, self.gain = str(p["code"]), float(p["gain"])

    def key(self) -> tuple:
        return (self.k, self.L, self.V, self.T, self.code, self.gain)


class SucLinear(nn.Module):
    """Drop-in replacement for nn.Linear running directly on SUC codes (README §5).

    tcq (trellis-coded tensors, suc.quant.trellis): the tensor's params; packed_indices are then the packed streams
    (out, in/T · NW) uint32 of suc.kernels.tcq_matmul, codebooks is None and `bits` = k·V, the kernel is
    tcq_matmul_metal (rotations, residual and bias exactly as for the codebook kinds)."""

    def __init__(self, codebooks: mx.array | None, packed_indices: mx.array, row_scales: mx.array, *, bits: int,
                 in_features: int, seed_in: int | None = None, seed_out: int | None = None,
                 bias: mx.array | None = None, tcq: dict | None = None):
        super().__init__()
        if tcq is not None:
            from suc.kernels.tcq_matmul import kernel_consts

            self._tcq = TcqMeta(tcq)
            self.packed_indices = packed_indices.astype(mx.uint32)
            self.row_scales = row_scales.astype(mx.float32)
            self.tcq_cst = mx.array(kernel_consts(tcq))
            self.bits, self.in_features, self.out_features = bits, in_features, packed_indices.shape[0]
            self._init_rotation(seed_in, seed_out, bias)
            return
        if codebooks.ndim == 2:
            codebooks = codebooks[None]
        if packed_indices.ndim == 2:
            packed_indices = packed_indices[None]
        m, k, g = codebooks.shape
        if packed_indices.shape[0] != m:
            raise ValueError(f"{m} codebooks but {packed_indices.shape[0]} index streams")
        out = packed_indices.shape[1]
        if packed_indices.shape[2] != words_per_row(in_features // g, bits):
            raise ValueError(f"packed_indices {packed_indices.shape} do not match in_features={in_features}, "
                             f"G={g}, bits={bits}")
        self.codebooks = codebooks.astype(mx.float16)
        self.packed_indices = packed_indices.astype(mx.uint32)
        self.row_scales = row_scales.astype(mx.float32)
        self.bits, self.in_features, self.out_features = bits, in_features, out
        self._init_rotation(seed_in, seed_out, bias)

    @property
    def tcq(self) -> TcqMeta | None:
        """Code layout of a trellis-coded tensor, None for the codebook kinds."""
        return self.__dict__.get("_tcq")

    def _init_rotation(self, seed_in: int | None, seed_out: int | None, bias: mx.array | None) -> None:
        in_features, out = self.in_features, self.out_features
        self.seed_in, self.seed_out = seed_in, seed_out
        if seed_in is not None:
            self.signs_in = random_signs(in_features, seed_in).astype(SIGN_DTYPE)
        if seed_out is not None:
            self.signs_out = random_signs(out, seed_out).astype(SIGN_DTYPE)
        if bias is not None:
            self.bias = bias

    @classmethod
    def from_encoded(cls, e) -> "SucLinear":
        if e.codec == "tcq":
            return cls._from_tcq(e)
        return cls._from_codebooks(e)

    @classmethod
    def _from_tcq(cls, e) -> "SucLinear":
        """From a tcq Encoded: research layout (chunks) packed into the streams, or "packed_words" (out, in/T · NW)
        uint32 as stored in SUC1 (SucModel.encoded_packed)."""
        from suc.kernels.tcq_matmul import pack_encoded

        p, a = e.params, e.arrays
        if p["code"] not in ("1mad", "3inst", "pair3"):
            raise ValueError(f"tcq code {p['code']!r} has no kernel")
        words = a["packed_words"] if "packed_words" in a else pack_encoded(e)
        rot = p.get("rotate", False)
        rot_in, rot_out = rot and p.get("rotate_in", True), rot and p.get("rotate_out", False)
        return cls(None, words, a["row_scales"], bits=int(p["kv"]), in_features=e.shape[1],
                   seed_in=p["seed_in"] if rot_in else None, seed_out=p["seed_out"] if rot_out else None, tcq=p)

    @classmethod
    def _from_codebooks(cls, e) -> "SucLinear":
        """From a suc.quant.codecs Encoded of kind vq or additive (indices (M, out, in//G) → packed layout). An Encoded
        that already carries "packed_indices" (M, out, words) uint32 in the packed layout (SucModel.encoded_packed:
        the SUC1 storage layout) is used as is."""
        if e.codec not in ("vq", "additive"):
            raise ValueError(f"SucLinear needs an Encoded of kind vq or additive, got {e.codec!r}")
        p, a = e.params, e.arrays
        bits = p["index_bits"]
        if "packed_indices" in a:
            packed = a["packed_indices"]
        else:
            packed = mx.stack([pack_indices(a["indices"][m], bits) for m in range(a["indices"].shape[0])])
        rot = p.get("rotate", False)
        rot_in, rot_out = rot and p.get("rotate_in", True), rot and p.get("rotate_out", True)
        return cls(a["codebooks"], packed, a["row_scales"], bits=bits, in_features=e.shape[1],
                   seed_in=p["seed_in"] if rot_in else None, seed_out=p["seed_out"] if rot_out else None)

    def kernel(self, x2: mx.array, out_dtype: mx.Dtype | None = None, path: str = "auto") -> mx.array:
        """W' x2 on (B, in) — the Metal kernel only (no rotation, no bias). path: "auto" (vec kernel for batch
        tiles < MMA_MIN_BATCH, simdgroup-matrix kernel from it), "vec" or "mma" (suc.kernels.vq_matmul)."""
        t = self.tcq
        if t is not None:
            from suc.kernels.tcq_matmul import tcq_matmul_metal

            return tcq_matmul_metal(x2, self.packed_indices, self.row_scales, k=t.k, L=t.L, V=t.V, T=t.T, code=t.code,
                                    cst=self.tcq_cst, path=path, out_dtype=out_dtype)
        return additive_matmul_metal(x2, self.codebooks, self.packed_indices, self.row_scales, bits=self.bits,
                                     in_features=self.in_features, out_dtype=out_dtype, path=path)

    def rotate_in(self, x2: mx.array) -> mx.array:
        """x̃ = V x2 rounded to x2.dtype (x2 unchanged when seed_in is None)."""
        if self.seed_in is None:
            return x2
        if ROTATION == "fused" and _rot.supported(self.in_features):
            return _rot.rht_fused(x2, self.signs_in)
        return fast_rht(x2, self.signs_in).astype(x2.dtype)

    def rotate_out(self, y32: mx.array, dtype: mx.Dtype) -> mx.array:
        """y = Uᵀ ỹ in dtype from the float32 kernel output (seed_out is not None)."""
        if ROTATION == "fused" and _rot.supported(self.out_features):
            return _rot.rht_inverse_fused(y32, self.signs_out, out_dtype=dtype)
        return fast_rht_inverse(y32, self.signs_out).astype(dtype)

    def from_rotated(self, xr: mx.array, x2: mx.array | None = None) -> mx.array:
        """y (B, out) in xr.dtype from x̃ = rotate_in(x2) (B, in). x2 (the unrotated input) is only read by
        subclasses with an unrotated-space term (SucResidualLinear)."""
        dtype = xr.dtype
        if self.seed_out is not None:
            y = self.rotate_out(self.kernel(xr, mx.float32), dtype)
        else:
            y = self.kernel(xr)
        if "bias" in self:
            y = y + self.bias.astype(dtype)
        return y

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected (..., {self.in_features}) input, got {x.shape}")
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        if COMPILE and self.compilable():
            y = compiled_call(self, self._forward, ("fwd", ROTATION), x2)
        else:
            y = self._forward(x2)
        return y.reshape(*lead, self.out_features)

    def _forward(self, x2: mx.array) -> mx.array:
        return self.from_rotated(self.rotate_in(x2), x2)

    def compilable(self) -> bool:
        return True


# --------------------------------------------------------------------------- fused projections (q|k|v, gate|up)
# The lookup kernels of suc.kernels.vq_matmul read codebook entry idx of stream m at codebook + (m·K + idx)·G. For a
# row-concatenation of S tensors the codebooks are stacked (S·M, K, G) and row r reads segment SUC_SEG(r) — the
# number of segment boundaries SB1 < SB2 < SB3 it has passed (template constants, unused ones = OUT, so no extra
# memory and no per-row table). Staged codebooks (CBTG = 1, CODEBOOK_IN resolving to "threadgroup": the M1 Ultra,
# where staging is 1.2-3x faster, K3) are supported when no threadgroup straddles a segment boundary (every
# boundary a multiple of the rows per threadgroup): each threadgroup stages only its own segment's M codebooks, so
# the D4 16 KB ceiling holds per segment. Otherwise the device-memory path (CBTG = 0). HM = 0, MMA "reg" only.
# (Before CBTG was supported, fused additive M=3 groups ran ~1.6x slower than the unfused members on the M1 Ultra:
# 8B kernel time 19.6 -> 31.5 ms per decode step, docs/benchmarks.md Q5b runtime.)
MAX_SEGMENTS = 4
_SEG_DEF = "\n#define SUC_SEG(r) (uint((r) >= SB1) + uint((r) >= SB2) + uint((r) >= SB3))\n"
_PATCHES = {
    "vec": (lambda: _vqm._LOOKUP_SRC, [
        ("for (uint i = thread_position_in_threadgroup.x; i < M * K * G; i += SG * 32) cbt[i] = codebook[i];",
         "for (uint i = thread_position_in_threadgroup.x; i < M * K * G; i += SG * 32) cbt[i] = codebook[SUC_SEG(threadgroup_position_in_grid.x * SG * R) * (M * K * G) + i];"),
        ("cb_add<G>(codebook + (m * K + idx) * G, c[rr] + sj * G);",
         "cb_add<G>(codebook + ((SUC_SEG(rows[rr]) * M + m) * K + idx) * G, c[rr] + sj * G);")]),
    "mma": (lambda: _vqm._MMA_REG_SRC, [
        ("for (uint i = thread_position_in_threadgroup.x; i < M * K * G; i += SG * 32) cbt[i] = codebook[i];",
         "for (uint i = thread_position_in_threadgroup.x; i < M * K * G; i += SG * 32) cbt[i] = codebook[SUC_SEG(threadgroup_position_in_grid.x * SG * 8 * RT) * (M * K * G) + i];"),
        ("cb_add<G>(codebook + (m * K + idx) * G, c[rt] + t * G);",
         "cb_add<G>(codebook + ((SUC_SEG(rows[rt]) * M + m) * K + idx) * G, c[rt] + t * G);")]),
}
_FUSED_KERNELS: dict[str, object] = {}


def _fused_kernel(kind: str):
    if kind not in _FUSED_KERNELS:
        get_src, subs = _PATCHES[kind]
        src = get_src()
        for old, new in subs:
            if src.count(old) != 1:
                raise RuntimeError(f"suc.kernels.vq_matmul {kind} source changed: cannot patch {old!r}")
            src = src.replace(old, new)
        _FUSED_KERNELS[kind] = mx.fast.metal_kernel(
            name=f"suc_fused_matmul_{kind}", input_names=["x", "codebook", "packed", "scales"],
            output_names=["y"], source=src, header=_vqm._HEADER + _SEG_DEF)
    return _FUSED_KERNELS[kind]


def fused_matmul_metal(x: mx.array, codebooks: mx.array, packed_indices: mx.array, row_scales: mx.array, *,
                       bits: int, in_features: int, seg_rows: tuple[int, ...], out_dtype: mx.Dtype | None = None,
                       path: str = "auto", codebook_in: str | None = None) -> mx.array:
    """y = x Ŵᵀ for a row-concatenation of S = len(seg_rows) tensors: codebooks (S, M, K, G) (segment s's own), packed
    indices (M, Σ seg_rows, words), row scales (Σ seg_rows,). Same kernels, tiling and dispatch (vec below
    MMA_MIN_BATCH, simdgroup-matrix "reg" from it) as suc.kernels.vq_matmul.additive_matmul_metal; codebook_in
    overrides vq_matmul.CODEBOOK_IN (staging needs segment boundaries aligned to the threadgroup rows, else device)."""
    if codebooks.ndim != 4 or codebooks.shape[0] != len(seg_rows):
        raise ValueError(f"codebooks must be ({len(seg_rows)}, M, K, G), got {codebooks.shape}")
    if not 1 <= len(seg_rows) <= MAX_SEGMENTS:
        raise ValueError(f"1..{MAX_SEGMENTS} segments supported, got {len(seg_rows)}")
    if path not in ("auto", "vec", "mma"):
        raise ValueError(f"path must be 'auto', 'vec' or 'mma', got {path!r}")
    S, m, k, g = codebooks.shape
    _, _, _, out, words = _vqm._check(x, codebooks[0], packed_indices, row_scales, bits, in_features)
    if sum(seg_rows) != out:
        raise ValueError(f"seg_rows {seg_rows} do not sum to {out} rows")
    if m * k * g * 2 > _vqm.CODEBOOK_TG_LIMIT:
        raise ValueError(f"{m} codebooks of K*G*2 = {k * g * 2} bytes exceed the 16 KB ceiling (D4)")
    bounds = list(np.cumsum(seg_rows)[:-1]) + [out] * (MAX_SEGMENTS - len(seg_rows))
    cb = codebooks.astype(mx.float16).reshape(S * m, k, g)
    packed, s = packed_indices.astype(mx.uint32), row_scales.astype(mx.float32)
    out_dtype = out_dtype or x.dtype
    ng = in_features // g
    staged = (FUSED_STAGING or codebook_in is not None) and \
        _vqm.resolve_codebook_in(codebook_in or _vqm.CODEBOOK_IN, m=m) == "threadgroup"
    ys = []
    for b0 in range(0, x.shape[0], _vqm.MAX_BATCH_TILE):
        xb = x[b0:b0 + _vqm.MAX_BATCH_TILE]
        bt = xb.shape[0]
        tmpl = [("T", x.dtype), ("BT", bt), ("OUT", out), ("IN", in_features), ("NG", ng), ("G", g), ("K", k),
                ("BITS", bits), ("WORDS", words), ("SB1", int(bounds[0])), ("SB2", int(bounds[1])),
                ("SB3", int(bounds[2]))]
        mma = path == "mma" or (path == "auto" and bt >= _vqm.mma_min_batch(m))
        sgs, rows = (_vqm.MMA_SG, 8 * _vqm.MMA_RT) if mma else (_vqm.LOOKUP_SG, _vqm._lookup_rows(bt))
        cbtg = int(staged and all(int(sb) % (sgs * rows) == 0 for sb in bounds[:len(seg_rows) - 1]))
        if mma:
            kind = "mma"
            tmpl += [("SG", sgs), ("M", m), ("RT", _vqm.MMA_RT), ("NB", -(-bt // 8)), ("KCG", _vqm._mma_kcg(g, bits)),
                     ("TT", _vqm._mma_tile(x.dtype, m)), ("CBTG", cbtg), ("TO", out_dtype)]
        else:
            kind = "vec"
            tmpl += [("SG", sgs), ("M", m), ("R", rows), ("VEC", _vqm.LOOKUP_VEC), ("CBTG", cbtg), ("HM", 0),
                     ("TO", out_dtype)]
        n_tg = -(-out // (sgs * rows))
        ys.append(_fused_kernel(kind)(inputs=[xb, cb, packed, s], template=tmpl, grid=(n_tg * sgs * 32, 1, 1),
                                      threadgroup=(sgs * 32, 1, 1), output_shapes=[(bt, out)],
                                      output_dtypes=[out_dtype])[0])
    return ys[0] if len(ys) == 1 else mx.concatenate(ys, axis=0)


def fusable_key(m: "SucLinear") -> tuple | None:
    """Members with equal keys can be row-concatenated into one SucFusedLinear (None: not fusable)."""
    if type(m) is not SucLinear or "bias" in m or m.tcq is not None:
        return None
    rot_out = m.seed_out is not None
    if rot_out and not _rot.supported(m.out_features):
        return None
    if m.seed_in is not None and not _rot.supported(m.in_features):
        return None
    return (m.in_features, m.bits, tuple(m.codebooks.shape), m.seed_in, rot_out)


class SucFusedLinear(SucLinear):
    """Row-concatenation of S ≥ 1 SucLinear sharing (in_features, bits, M, K, G, seed_in, rotate_out-ness):
    one SUC launch and one segmented output-rotation launch for all members. Codebooks (S, M, K, G) stay per
    member; row scales, packed indices and output signs are concatenated, so memory is unchanged."""

    def __init__(self, members: list[SucLinear]):
        keys = {fusable_key(m) for m in members}
        if len(keys) != 1 or None in keys:
            raise ValueError(f"members are not fusable: {keys}")
        if len(members) > MAX_SEGMENTS:
            raise ValueError(f"at most {MAX_SEGMENTS} members, got {len(members)}")
        nn.Module.__init__(self)
        f = members[0]
        self.codebooks = mx.stack([m.codebooks for m in members])                     # (S, M, K, G)
        self.packed_indices = mx.concatenate([m.packed_indices for m in members], axis=1)
        self.row_scales = mx.concatenate([m.row_scales for m in members])
        self.bits, self.in_features = f.bits, f.in_features
        self.sizes = tuple(m.out_features for m in members)
        self.out_features = sum(self.sizes)
        self.seed_in = f.seed_in
        self.seed_out = tuple(m.seed_out for m in members) if f.seed_out is not None else None
        if self.seed_in is not None:
            self.signs_in = f.signs_in
        if self.seed_out is not None:
            self.signs_out = mx.concatenate([m.signs_out for m in members])

    def kernel(self, x2: mx.array, out_dtype: mx.Dtype | None = None, path: str = "auto") -> mx.array:
        return fused_matmul_metal(x2, self.codebooks, self.packed_indices, self.row_scales, bits=self.bits,
                                  in_features=self.in_features, seg_rows=self.sizes, out_dtype=out_dtype, path=path)

    def from_rotated_split(self, xr: mx.array) -> list[mx.array]:
        """One (B, out_s) output per member, in xr.dtype."""
        dtype = xr.dtype
        if self.seed_out is not None:
            return _rot.rht_inverse_fused(self.kernel(xr, mx.float32), self.signs_out, self.sizes, out_dtype=dtype,
                                          split=True)
        y = self.kernel(xr)
        return [y[:, o - n:o] for o, n in zip(np.cumsum(self.sizes), self.sizes)]

    def from_rotated(self, xr: mx.array, x2: mx.array | None = None) -> mx.array:
        return mx.concatenate(self.from_rotated_split(xr), axis=-1)
