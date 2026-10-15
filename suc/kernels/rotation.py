"""Fused randomized-Hadamard rotations for SucLinear (README §5): one Metal launch per side.

    input side   x̃ = V x  = H (σ_in ⊙ x)      cast(x) → float32 → sign multiply → H → cast to out dtype
    output side  y = Uᵀ ỹ = σ_out ⊙ (Hᵀ ỹ)     float32 ỹ → Hᵀ → sign multiply → cast to out dtype

replacing the 3 + 3 MLX launches of suc.kernels.suc_linear.fast_rht / fast_rht_inverse (cast·sign, hadamard_transform,
cast). H = H_m ⊗ H_p (n = m·p, p = 2^k, m ∈ {1, 12, 20, 28, 17}; m is the major index, exactly as
suc.quant.hadamard._transform): Sylvester butterflies over p, then the m×m mix with MLX's own H_m (passed as an
input, transposed for the inverse). All arithmetic is float32; the result equals suc.quant.hadamard.rht /
rht_inverse up to float32 rounding (the layouts differ only in the float32 summation order; bf16 outputs agree
with the pre-R5 kernel except for rare rounding-boundary cases, ≤ 1 ulp). The reference H_m step is a sequential
fma chain (suc.quant.hadamard._mix, bit-identical on GPU / CPU / C++). Measured on the M3 for m = 12 / 20 / 28
(3072, 5120, 7168, 12288): the "row" layout happens to reproduce the reference bit for bit; "blocks" (signed
block sums in a different order) differs by ≤ 2.3e-7 relative to the largest output. Runtime kernels are only
required to match within tolerance (tests/unit/test_rotation_kernel.py: rel. L2 < 2e-6, max abs < 1e-5).

Segments: the output side of a fused projection (q|k|v or gate|up concatenated along the output dim,
suc.runtime.compressed_model) holds several independent rotations side by side; `sizes` lists them and every
(batch row, segment) pair is one threadgroup, so a whole fused projection is rotated by ONE launch.

Per-row signs (MoE experts, suc.kernels.suc_switch): with `rows` (B,) uint32 the signs are a table (E, Σ sizes) and
batch row b uses signs[rows[b]] — every row of one launch can belong to a different expert (its own V_e / U_e).

Kernel layouts (LAYOUT; "auto" = per size from the per-device autotune cache, else "blocks"):

  "blocks" (R5, default): one threadgroup per Q-element OUTPUT block (Q = 128 … 512, block_config), so a batch-1
      rotation of n elements runs on n/Q GPU cores at once. Every block reads the whole input row (it is a few KB
      and in cache): with H_P = H_{P/Q} ⊗ H_Q, block (a, c1) is y = H_Q z, z = Σ_(b, a1) H_m[a, b] (−1)^popcount(c1 & a1)
      σx_(b, a1) — a signed sum of the M·P/Q input blocks (split over the simdgroups, partial sums combined in
      threadgroup memory) followed by an in-simdgroup H_Q (in-register stages + simd_shuffle_xor). No grid-wide step,
      one launch, a short critical path per threadgroup. 12288 = 12 · 1024 is handled in the same kernel (the H_12
      factor enters the coefficients; constant table, no loads).
  "row" (K4): one threadgroup of T threads per (row, segment), T = min(256, p / VEC): thread t owns VEC contiguous
      columns per chunk (vector loads; VEC = 1 is the original strided layout). Butterfly strides inside a chunk or
      ≥ T·VEC are in-register, the next five are simd_shuffle_xor, the rest exchange through a threadgroup buffer
      (16 KB, chunked over registers); then the m × m mix (H_m as compile-time constants with HM_CONST).
      The whole n log n transform serializes on ONE GPU core: in a decode step (input just written by a many-core
      projection kernel, caches cold) this cost 13–22 µs per rotation on the M3 vs ~2–3 µs for an empty launch
      (benchmarks/rotation_latency.py), the blocks layout 4–9 µs.

Epilogue fusion into the SUC matmul kernel was considered and rejected: Hᵀ mixes ALL output rows, while each SUC
threadgroup owns only 32–128 of them, so it would need a grid-wide barrier (or a second pass, i.e. this launch).
"""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from suc.quant.hadamard import _factor, _h_m

THREADS_MAX = 256
BUF_FLOATS = 4096          # threadgroup exchange buffer (16 KB)
VEC = 4                    # contiguous elements per thread and vector load (C; 1 = the strided K4 layout)
HM_CONST = True            # H_m (m = 12, 20, 28, 17) as compile-time constants (False: read from the hm input)
# R5 kernel layout: "blocks" = one threadgroup per Q-element output block (many cores, short critical path);
# "row" = one threadgroup per row and segment (the K4 kernel, with VEC / HM_CONST); "auto" = per size n the
# per-device autotune choice (suc.kernels.autotune, "rotation"), else "blocks".
LAYOUT = "auto"
# blocks layout shape: output elements per threadgroup Q (32 lanes × Q/32 contiguous) and simdgroups per
# threadgroup (the input blocks are split over them). None = per size n: the per-device autotune choice
# (suc.kernels.autotune, "rotation"), else default_block_config(n).
BLOCK_Q: int | None = None
BLOCK_SG: int | None = None
TG_FLOATS = 8192 - 64      # threadgroup memory budget for the partial sums (32 KB minus the prologue's reductions)

# Launch policy (audit F09). A compiled Metal pipeline accepts at most maxTotalThreadsPerThreadgroup threads, and that
# limit shrinks with the kernel's register use; MLX's Python API does not expose it, and exceeding it is not a Python
# exception: MLX throws std::invalid_argument from native code and the interpreter aborts. Measured on the M1 Ultra
# (2026-10-10, subprocess probe, row layout): T = 512 aborts for m ∈ {17, 20, 28} at p = 1024 with VEC ≤ 2 (limit
# 384 for 17408) and for every m at n ≥ 34816 whatever VEC; T = 256 runs every probed size up to n = 114688 = 28·4096
# (VEC 1/2/4). The C++ runtime always launches 256. So every launch of this module uses at most
# LAUNCH_MAX_THREADS threads: the row layout clips THREADS_MAX to it (T stays a power of two dividing every 2^k
# factor — the butterflies need that, so T is never clamped to a non-power-of-two such as 384), the blocks layout
# clips its simdgroups to LAUNCH_MAX_THREADS / 32, and _check_launch rejects anything else before dispatch.
# LAUNCH_POLICY is part of the autotune cache identity (suc.kernels.autotune.rotation_choice).
LAUNCH_MAX_THREADS = 256
LAUNCH_POLICY = "tg256-pow2-v1"


def _is_pow2(v: int) -> bool:
    return v > 0 and v & (v - 1) == 0


def row_threads_max() -> int:
    """THREADS_MAX clipped to the launch policy (ValueError unless THREADS_MAX is a power of two ≥ 32)."""
    if not _is_pow2(int(THREADS_MAX)) or THREADS_MAX < 32:
        raise ValueError(f"rotation THREADS_MAX={THREADS_MAX} must be a power of two >= 32")
    return min(int(THREADS_MAX), LAUNCH_MAX_THREADS)


def _check_launch(t: int, layout: str) -> None:
    """Reject a threadgroup size outside the launch policy BEFORE dispatch (an over-limit launch aborts the
    process instead of raising)."""
    if not (32 <= t <= LAUNCH_MAX_THREADS and t % 32 == 0 and (layout == "blocks" or _is_pow2(t))):
        raise ValueError(f"rotation {layout} launch with {t} threads violates the launch policy {LAUNCH_POLICY} "
                         f"(<= {LAUNCH_MAX_THREADS}, a multiple of 32{', a power of two' if layout == 'row' else ''})")

_HEADER = """
#define SUC_UNROLL _Pragma("clang loop unroll(full)")
// Unscaled H = H_m ⊗ H_p on the registers of thread t (E = M·P/T values). Layout (C | P/T, C a power of two):
// v[a·PT + h·C + l] holds element a·P + h·T·C + t·C + l (l < C contiguous → vector loads; C = 1 is the old strided
// layout a·P + t + h·T). The butterflies only need every register-index bit below PT to be an element-index bit
// (l bits: strides 1 … C/2, h bits: strides T·C …), so the core is the same for every C.
// HC = 1: the H_m entries come from the compile-time tables suc_hm_const (folded into immediates by the unrolled
// mix, no loads); HC = 0: from the input h (already transposed for the inverse). TR: transpose the table.
template <int M, int P, int T, int BUFN, int HC, int TR, typename PH>
inline void suc_rht_core(thread float* v, PH h, threadgroup float* buf, uint t) {
    constexpr int PT = P / T;
    constexpr int E = M * PT;
    constexpr int CH = (BUFN / T) < E ? (BUFN / T) : E;
    // register strides s = 1 … PT/2 (element strides 1 … C/2 and T·C … P/2): partner in the same thread
    SUC_UNROLL
    for (int s = 1; s < PT; s <<= 1)
        SUC_UNROLL
        for (int a = 0; a < M; ++a)
            SUC_UNROLL
            for (int j = 0; j < PT; ++j)
                if ((j & s) == 0) {
                    float u = v[a * PT + j], w = v[a * PT + (j | s)];
                    v[a * PT + j] = u + w;
                    v[a * PT + (j | s)] = u - w;
                }
    // element strides C·s, s = 1 … 16: partner lane t ^ s of the same simdgroup
    SUC_UNROLL
    for (uint s = 1; s < (T < 32 ? T : 32); s <<= 1) {
        bool hi = (t & s) != 0;
        SUC_UNROLL
        for (int e = 0; e < E; ++e) {
            float w = simd_shuffle_xor(v[e], ushort(s));
            v[e] = hi ? (w - v[e]) : (v[e] + w);
        }
    }
    // element strides C·s, s = 32 … T/2: through threadgroup memory, CH registers at a time
    SUC_UNROLL
    for (uint s = 32; s < T; s <<= 1) {
        bool hi = (t & s) != 0;
        SUC_UNROLL
        for (int e0 = 0; e0 < E; e0 += CH) {
            SUC_UNROLL
            for (int e = 0; e < CH && e0 + e < E; ++e) buf[e * T + t] = v[e0 + e];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            SUC_UNROLL
            for (int e = 0; e < CH && e0 + e < E; ++e) {
                float w = buf[e * T + (t ^ s)];
                v[e0 + e] = hi ? (w - v[e0 + e]) : (v[e0 + e] + w);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    // H_m mix across the m sub-vectors (h already transposed for the inverse)
    if (M > 1) {
        SUC_UNROLL
        for (int j = 0; j < PT; ++j) {
            float u[M];
            SUC_UNROLL
            for (int a = 0; a < M; ++a) {
                float acc = 0.0f;
                SUC_UNROLL
                for (int b = 0; b < M; ++b)
                    acc += (HC ? suc_hm_const<M>(TR ? b : a, TR ? a : b) : float(h[a * M + b])) * v[b * PT + j];
                u[a] = acc;
            }
            SUC_UNROLL
            for (int a = 0; a < M; ++a) v[a * PT + j] = u[a];
        }
    }
}

// C contiguous values p[0 … C) ↔ float v[0 … C): one vector load / store of the raw bits (device memory);
// constant address space and C = 1: scalar accesses
template <int S> struct suc_ubits { typedef uint type; };
template <> struct suc_ubits<2> { typedef ushort type; };
template <> struct suc_ubits<1> { typedef uchar type; };
template <int C, typename TV> struct suc_io {
    typedef typename suc_ubits<sizeof(TV)>::type U;
    static inline void ld(const device TV* p, thread float* v) {
        vec<U, C> q = *reinterpret_cast<const device vec<U, C>*>(p);
        SUC_UNROLL
        for (int l = 0; l < C; ++l) v[l] = float(as_type<TV>(q[l]));
    }
    static inline void st(device TV* p, thread const float* v) {
        vec<U, C> q;
        SUC_UNROLL
        for (int l = 0; l < C; ++l) q[l] = as_type<U>(static_cast<TV>(v[l]));
        *reinterpret_cast<device vec<U, C>*>(p) = q;
    }
};
template <typename TV> struct suc_io<1, TV> {
    static inline void ld(const device TV* p, thread float* v) { v[0] = float(p[0]); }
    static inline void st(device TV* p, thread const float* v) { p[0] = static_cast<TV>(v[0]); }
};
template <int C, typename TV> inline void suc_ld(const device TV* p, thread float* v) { suc_io<C, TV>::ld(p, v); }
template <int C, typename TV> inline void suc_ld(constant TV* p, thread float* v) {
    SUC_UNROLL
    for (int l = 0; l < C; ++l) v[l] = float(p[l]);
}
template <int C, typename TV> inline void suc_st(device TV* p, thread const float* v) { suc_io<C, TV>::st(p, v); }

// Element of register e of thread t (layout above): a·P + h·T·C + t·C + l, e = a·PT + h·C + l.
template <int P, int T, int C>
inline uint suc_idx(int e, uint t) {
    constexpr int PT = P / T;
    return uint((e / PT) * P + ((e % PT) / C) * T * C + (e % C)) + t * C;
}

// PX / PS / PH / PY: pointer types (MLX passes small inputs in the constant address space)
template <int M, int P, int T, int C, int INV, int BUFN, int HC, typename PX, typename PS, typename PH, typename PY>
inline void suc_rht_row(PX x, PS signs, PH hmat, PY y, threadgroup float* buf, uint t) {
    constexpr int PT = P / T;
    constexpr int E = M * PT;
    float v[E];
    SUC_UNROLL
    for (int e = 0; e < E; e += C) {
        uint i = suc_idx<P, T, C>(e, t);
        suc_ld<C>(x + i, v + e);
        if (!INV) {
            float sg[C];
            suc_ld<C>(signs + i, sg);
            SUC_UNROLL
            for (int l = 0; l < C; ++l) v[e + l] *= sg[l];
        }
    }
    suc_rht_core<M, P, T, BUFN, HC, INV>(v, hmat, buf, t);
    // required: without it bf16 n = 17408 (M = 17, C = 1, HC = 1, forward) stored ~0 for sub-vectors a = 0, 1 on
    // the M3 (registers v[0 … 7] lost between the H_m mix and the store); every other dtype / C / M was correct
    simdgroup_barrier(mem_flags::mem_none);
    const float sc = rsqrt(float(P));
    SUC_UNROLL
    for (int e = 0; e < E; e += C) {
        uint i = suc_idx<P, T, C>(e, t);
        float o[C];
        SUC_UNROLL
        for (int l = 0; l < C; ++l) o[l] = v[e + l] * sc;
        if (INV) {
            float sg[C];
            suc_ld<C>(signs + i, sg);
            SUC_UNROLL
            for (int l = 0; l < C; ++l) o[l] *= sg[l];
        }
        suc_st<C>(y + i, o);
    }
}
"""


_BLOCKS_HEADER = """
// ---- "blocks" layout (R5): one threadgroup per Q-element output block, every block reads the whole input.
// Element (a, c1, q) = a·P + c1·Q + q of a segment (a < M, c1 < P/Q, q < Q). Sylvester H_P = H_{P/Q} ⊗ H_Q, so output
// block g = (a, c1) is   y_g = H_Q z,  z = Σ_r coef(g, r) · σx_r   over the R = M·P/Q input blocks r = (b, a1), with
// coef = H_m[a, b] · (-1)^popcount(c1 & a1)  (H_m[b, a] for the inverse; no H_m factor for m = 1). Simdgroup sg sums
// rows r ≡ sg (mod NSG), lane L owns block elements L·C … L·C + C − 1 (one vector load per row); the NSG partial sums
// meet in threadgroup memory, simdgroup 0 adds them in order and applies H_Q (C-element in-register stages, then
// simd_shuffle_xor). No grid-wide step: ONE launch of G = n/Q threadgroups, each with a short critical path (the
// single-threadgroup "row" layout serializes the whole n log n butterfly on one GPU core).
// Sum of squares of the N values in buf (filled by the caller, no barrier yet) in exactly the order of MLX's
// rms_single_row (N ≤ 4096): a partial sum per 4 contiguous elements, simd_sum per 128-element block, simd_sum over
// the zero-padded 32 block sums. red: 32 floats of threadgroup memory. Every thread gets the result.
template <int N, int T>
inline float suc_mlx_sumsq(threadgroup float* buf, threadgroup float* red, uint t) {
    if (t < 32) red[t] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint lane = t % 32;
    for (uint k = t / 32; k < (N + 127) / 128; k += T / 32) {
        float acc = 0.0f;
        if (k * 128 + lane * 4 + 4 <= N)
            for (int l = 0; l < 4; ++l) {
                float xi = buf[k * 128 + lane * 4 + l];
                acc += xi * xi;
            }
        acc = simd_sum(acc);
        if (lane == 0) red[k] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = simd_sum(red[lane]);
    // required: without it bf16 n = 3072 (row layout, T = 64, C = 4) came out wrong on the M3 when xn is not stored
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return ss;
}

// Threadgroup sum of the per-thread values ss (any N; not MLX's order).
template <int T>
inline float suc_tg_sum(float ss, threadgroup float* red, uint t) {
    ss = simd_sum(ss);
    if (T > 32) {
        if (t % 32 == 0) red[t / 32] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ss = 0.0f;
        for (uint k = 0; k < T / 32; ++k) ss += red[k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return ss;
}

template <int M> inline float suc_hm_at(uint a, uint b) {
    return M == 12 ? SUC_H12[a * 12 + b] : M == 20 ? SUC_H20[a * 20 + b] : M == 28 ? SUC_H28[a * 28 + b]
         : M == 17 ? SUC_H17[a * 17 + b] : 1.0f;
}

// v[k·C + l] (k < K): σ-free values of row r = sg + k·NSG (r < R) at column lane·C + l. Accumulate, reduce, H_Q,
// scale, (inverse: signs), store. Returns after storing (only simdgroup 0 stores).
template <int M, int P, int Q, int T, int C, int INV, typename PS, typename PY>
inline void suc_blocks_finish(thread float* v, PS signs, PY y, uint g, threadgroup float* buf, uint t) {
    constexpr int NSG = T / 32;
    constexpr int PQ = P / Q;
    constexpr int R = M * PQ;
    constexpr int K = (R + NSG - 1) / NSG;
    const uint sg = t / 32, lane = t % 32;
    const uint a_out = g / PQ, c1 = g % PQ;
    float acc[C];
    SUC_UNROLL
    for (int l = 0; l < C; ++l) acc[l] = 0.0f;
    SUC_UNROLL
    for (int k = 0; k < K; ++k) {
        uint r = sg + k * NSG;
        if (R % NSG == 0 || r < R) {
            uint bm = r / PQ, a1 = r % PQ;
            float coef = (popcount(c1 & a1) & 1) ? -1.0f : 1.0f;
            if (M > 1) coef *= suc_hm_at<M>(INV ? bm : a_out, INV ? a_out : bm);
            SUC_UNROLL
            for (int l = 0; l < C; ++l) acc[l] += coef * v[k * C + l];
        }
    }
    if (NSG > 1) {
        SUC_UNROLL
        for (int l = 0; l < C; ++l) buf[sg * Q + lane * C + l] = acc[l];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg != 0) return;
        SUC_UNROLL
        for (int l = 0; l < C; ++l) {
            float a = 0.0f;
            SUC_UNROLL
            for (int k = 0; k < NSG; ++k) a += buf[k * Q + lane * C + l];
            acc[l] = a;
        }
    }
    SUC_UNROLL
    for (int s = 1; s < C; s <<= 1)
        SUC_UNROLL
        for (int l = 0; l < C; ++l)
            if ((l & s) == 0) {
                float u = acc[l], w = acc[l | s];
                acc[l] = u + w;
                acc[l | s] = u - w;
            }
    SUC_UNROLL
    for (uint s = 1; s < 32; s <<= 1) {
        bool hi = (lane & s) != 0;
        SUC_UNROLL
        for (int l = 0; l < C; ++l) {
            float w = simd_shuffle_xor(acc[l], ushort(s));
            acc[l] = hi ? (w - acc[l]) : (acc[l] + w);
        }
    }
    const float sc = rsqrt(float(P));
    const uint i = g * Q + lane * C;
    float o[C];
    SUC_UNROLL
    for (int l = 0; l < C; ++l) o[l] = acc[l] * sc;
    if (INV) {
        float sgn[C];
        suc_ld<C>(signs + i, sgn);
        SUC_UNROLL
        for (int l = 0; l < C; ++l) o[l] *= sgn[l];
    }
    suc_st<C>(y + i, o);
}

// Plain rotation of output block g of one segment (x, signs, y already offset to the segment).
template <int M, int P, int Q, int T, int C, int INV, typename PX, typename PS, typename PY>
inline void suc_blocks_row(PX x, PS signs, PY y, uint g, threadgroup float* buf, uint t) {
    constexpr int NSG = T / 32;
    constexpr int R = M * (P / Q);
    constexpr int K = (R + NSG - 1) / NSG;
    const uint sg = t / 32, lane = t % 32;
    float v[K * C];
    SUC_UNROLL
    for (int k = 0; k < K; ++k) {
        uint r = sg + k * NSG;
        if (R % NSG == 0 || r < R) {
            uint i = r * Q + lane * C;
            suc_ld<C>(x + i, v + k * C);
            if (!INV) {
                float sgn[C];
                suc_ld<C>(signs + i, sgn);
                SUC_UNROLL
                for (int l = 0; l < C; ++l) v[k * C + l] *= sgn[l];
            }
        }
    }
    suc_blocks_finish<M, P, Q, T, C, INV>(v, signs, y, g, buf, t);
}
"""


def _hm_tables() -> str:
    """Metal source of suc_hm_const<M>(a, b) = H_m[a, b] (float32, exact hex literals): MLX's Hadamard matrices for
    m = 12, 20, 28 and the Hartley factor for m = 17 (suc.quant.hadamard)."""
    import numpy as np
    out = []
    for m in (12, 20, 28, 17):
        h = np.array(_h_m(m), dtype=np.float32).reshape(-1)
        vals = ", ".join(float(v).hex() + "f" for v in h)
        out.append(f"constant constexpr float SUC_H{m}[{m * m}] = {{{vals}}};")
    out.append("template <int M> inline float suc_hm_const(int a, int b) {\n"
               "    return M == 12 ? SUC_H12[a * 12 + b] : M == 20 ? SUC_H20[a * 20 + b] : M == 28 ? SUC_H28[a * 28 + b]"
               " : M == 17 ? SUC_H17[a * 17 + b] : 1.0f;\n}")
    return "\n".join(out) + "\n"


_HEADER = _hm_tables() + _HEADER + _BLOCKS_HEADER


def supported(n: int) -> bool:
    """n = m · 2^k with m ∈ {1, 12, 20, 28, 17} and 2^k ≥ 32 (smaller sizes use the MLX fallback)."""
    f = _factor(n)
    return f is not None and f[1] >= 32


@lru_cache(maxsize=None)
def _kernel(nseg: int, split: bool, row_signs: bool = False):
    """split=False: one output (B, Σ sizes); split=True: one output y{s} (B, size_s) per segment.
    row_signs=True: extra input sidx (B,), batch row b reads signs + sidx[b] * NTOT."""
    calls = []
    sbase = "signs + ulong(sidx[b]) * NTOT + " if row_signs else "signs + "
    for s in range(nseg):
        dst = f"y{s} + b * P{s} * M{s}" if split else f"y + b * NTOT + OFF{s}"
        calls.append(f"    if (seg == {s}) suc_rht_row<M{s}, P{s}, T, C{s}, INV, BUFN, HC>(x + b * NTOT + OFF{s}, "
                     f"{sbase}OFF{s}, hm + HO{s}, {dst}, buf, t);")
    src = "\n".join(["    uint t = thread_position_in_threadgroup.x;",
                     "    uint seg = threadgroup_position_in_grid.y;",
                     "    uint b = threadgroup_position_in_grid.z;",
                     "    threadgroup float buf[BUFN];", *calls])
    outs = [f"y{s}" for s in range(nseg)] if split else ["y"]
    return mx.fast.metal_kernel(name=f"suc_rht_fused_{nseg}{'_split' if split else ''}{'_rows' if row_signs else ''}",
                                input_names=["x", "signs", "hm"] + (["sidx"] if row_signs else []),
                                output_names=outs, source=src, header=_HEADER)


@lru_cache(maxsize=None)
def threads_and_vec(ps: tuple[int, ...], tmax: int, vec: int) -> tuple[int, tuple[int, ...]]:
    """(T, C per segment) for the 2^k factors ps: T = min(tmax, max(32, min(ps) / vec)) threads, C = min(vec, p / T).
    tmax and vec must be powers of two (tmax ≥ 32), so T is a power of two dividing every p and C divides p / T
    (the butterfly layout needs both); callers pass row_threads_max(), never more than LAUNCH_MAX_THREADS."""
    if not _is_pow2(tmax) or tmax < 32 or not _is_pow2(vec):
        raise ValueError(f"rotation threads {tmax} / VEC {vec}: need powers of two, threads >= 32")
    t = min(tmax, max(32, min(ps) // vec))
    cs = tuple(min(vec, p // t) for p in ps)
    if not all(_is_pow2(p) and p % t == 0 and (p // t) % c == 0 for p, c in zip(ps, cs)):
        raise ValueError(f"rotation threads {t} / VEC {vec} do not divide the 2^k factors {ps}")
    return t, cs


@lru_cache(maxsize=None)
def _plan(sizes: tuple[int, ...], inverse: bool, split: bool, tmax: int, bufn: int, row_signs: bool = False,
          vec: int = 1):
    """(kernel, template, H_m matrices, threads) for one segment layout (cached: the per-call Python cost of a
    decode step is part of what this module removes)."""
    if not all(supported(n) for n in sizes):
        raise ValueError(f"unsupported rotation sizes {sizes}: need m·2^k, m ∈ (1, 12, 20, 28, 17), 2^k ≥ 32")
    fs = [_factor(n) for n in sizes]
    ms, ps = tuple(f[0] for f in fs), tuple(f[1] for f in fs)
    t, cs = threads_and_vec(ps, tmax, vec)
    mats, tmpl, off, ho = [], [("T", t), ("INV", int(inverse)), ("BUFN", bufn), ("HC", int(HM_CONST)),
                               ("NTOT", sum(sizes))], 0, 0
    for s, (m, p) in enumerate(zip(ms, ps)):
        h = _h_m(m) if m > 1 else mx.ones((1, 1), dtype=mx.float32)
        mats.append((h.T if inverse else h).reshape(-1))
        tmpl += [(f"M{s}", m), (f"P{s}", p), (f"C{s}", cs[s]), (f"OFF{s}", off), (f"HO{s}", ho)]
        off, ho = off + m * p, ho + m * m
    hm = mx.concatenate(mats).astype(mx.float32)
    mx.eval(hm)
    return _kernel(len(sizes), split, row_signs), tmpl, hm, t


@lru_cache(maxsize=None)
def _blocks_kernel(nseg: int, split: bool, row_signs: bool = False):
    calls = []
    sbase = "signs + ulong(sidx[b]) * NTOT + " if row_signs else "signs + "
    for s in range(nseg):
        dst = f"y{s} + b * P{s} * M{s}" if split else f"y + b * NTOT + OFF{s}"
        calls.append(f"    {'else ' if s else ''}if (gy < GE{s}) suc_blocks_row<M{s}, P{s}, Q, T, C, INV>("
                     f"x + b * NTOT + OFF{s}, {sbase}OFF{s}, {dst}, gy - GS{s}, buf, t);")
    src = "\n".join(["    uint t = thread_position_in_threadgroup.x;",
                     "    uint gy = threadgroup_position_in_grid.y;",
                     "    uint b = threadgroup_position_in_grid.z;",
                     "    threadgroup float buf[BUFQ];", *calls])
    outs = [f"y{s}" for s in range(nseg)] if split else ["y"]
    return mx.fast.metal_kernel(name=f"suc_rht_blocks_{nseg}{'_split' if split else ''}{'_rows' if row_signs else ''}",
                                input_names=["x", "signs"] + (["sidx"] if row_signs else []),
                                output_names=outs, source=src, header=_HEADER)


def default_block_config(n: int) -> tuple[int, int]:
    """(Q, simdgroups) of the blocks layout without a tuned entry: Q = the power of two ≤ n/16 in [128, 512]
    (M3, decode-step context: 4096 → 256, 12288 → 512; Q = 128 at 12288 was 2× slower — 96 threadgroups each
    re-reading the 72 KB of input and signs), 8 simdgroups."""
    q = 128
    while q * 2 <= min(512, n // 16):
        q *= 2
    return q, 8


def layout_for(n: int) -> str:
    """"blocks" | "row" for a rotation of size n (LAYOUT, resolved per device when "auto")."""
    if LAYOUT != "auto":
        return LAYOUT
    from suc.kernels import autotune
    tuned = autotune.rotation_choice(n)
    return tuned["layout"] if tuned else "blocks"


def block_config(n: int) -> tuple[int, int]:
    """(Q, simdgroups) for a rotation of size n: BLOCK_Q / BLOCK_SG overrides, else the tuned choice, else the
    default (the shape is further clipped to the size: blocks_shape)."""
    from suc.kernels import autotune
    tuned = autotune.rotation_choice(n)
    q, sg = (tuned["q"], tuned["sg"]) if tuned and tuned.get("layout") == "blocks" else default_block_config(n)
    return BLOCK_Q or q, BLOCK_SG or sg


def blocks_shape(ps: tuple[int, ...], ms: tuple[int, ...], q_max: int, sg_max: int) -> tuple[int, int]:
    """(Q, T) of the blocks layout: Q = min(q_max, min p) ≥ 32, T = 32 · min(sg_max, fewest input blocks)."""
    q = min(q_max, min(ps))
    # partial sums fit in 32 KB; at most LAUNCH_MAX_THREADS threads (fewer simdgroups only split the input blocks
    # differently, the result is the same up to float32 summation order)
    sg = min(sg_max, min(m * p // q for m, p in zip(ms, ps)), TG_FLOATS // q, LAUNCH_MAX_THREADS // 32)
    return q, 32 * max(1, sg)


@lru_cache(maxsize=None)
def _blocks_plan(sizes: tuple[int, ...], inverse: bool, split: bool, q_max: int, sg_max: int,
                 row_signs: bool = False):
    """(kernel, template, threads, threadgroups per row) of the blocks layout for one segment layout."""
    if not all(supported(n) for n in sizes):
        raise ValueError(f"unsupported rotation sizes {sizes}: need m·2^k, m ∈ (1, 12, 20, 28, 17), 2^k ≥ 32")
    fs = [_factor(n) for n in sizes]
    ms, ps = tuple(f[0] for f in fs), tuple(f[1] for f in fs)
    q, t = blocks_shape(ps, ms, q_max, sg_max)
    tmpl, off, gs = [("Q", q), ("T", t), ("C", q // 32), ("INV", int(inverse)), ("NTOT", sum(sizes)),
                     ("BUFQ", (t // 32) * q)], 0, 0
    for s, (m, p) in enumerate(zip(ms, ps)):
        tmpl += [(f"M{s}", m), (f"P{s}", p), (f"OFF{s}", off), (f"GS{s}", gs), (f"GE{s}", gs + m * p // q)]
        off, gs = off + m * p, gs + m * p // q
    return _blocks_kernel(len(sizes), split, row_signs), tmpl, t, gs


SIGN_DTYPES = (mx.float32, mx.int8)   # read by the kernels as stored (int8: 1 byte per element instead of 4)


def _signs_arg(signs: mx.array) -> mx.array:
    return signs if signs.dtype in SIGN_DTYPES else signs.astype(mx.float32)


def rotate(x: mx.array, signs: mx.array, sizes: tuple[int, ...] | None = None, *, inverse: bool = False,
           out_dtype: mx.Dtype | None = None, split: bool = False,
           rows: mx.array | None = None) -> mx.array | list[mx.array]:
    """Segment-wise randomized Hadamard on the last axis of x (B, Σ sizes), one launch.

    inverse=False: H (σ ⊙ x) per segment (V x);  inverse=True: σ ⊙ (Hᵀ x) per segment (Vᵀ x / Uᵀ ỹ).
    signs: (Σ sizes,) ±1 float32 or int8 (each segment's own signs, concatenated; int8 — what SucLinear stores —
    is read as is, other dtypes are cast to float32). out_dtype defaults to x.dtype.
    split=True returns one array (..., size_s) per segment (written directly, no slicing copies).
    rows (B,) uint32 (B = number of rows of x): per-row sign table, signs (E, Σ sizes), row b uses signs[rows[b]]."""
    n_tot = x.shape[-1]
    sizes = (n_tot,) if sizes is None else tuple(int(s) for s in sizes)
    if sum(sizes) != n_tot or signs.shape[-1] != n_tot or signs.ndim != (1 if rows is None else 2):
        raise ValueError(f"sizes {sizes} / signs {signs.shape} do not match the last axis {n_tot}")
    out_dtype = out_dtype or x.dtype
    lead = x.shape[:-1]
    x2 = x.reshape(-1, n_tot)
    b = x2.shape[0]
    shapes = [(b, n) for n in sizes] if split else [(b, n_tot)]
    if rows is not None and rows.shape != (b,):
        raise ValueError(f"rows must be ({b},), got {rows.shape}")
    extra = [rows.astype(mx.uint32)] if rows is not None else []
    lay = layout_for(max(sizes))
    if lay == "blocks":
        kern, tmpl, t, g = _blocks_plan(sizes, inverse, split, *block_config(max(sizes)), rows is not None)
        inputs, grid = [x2, _signs_arg(signs), *extra], (t, g, b)
    else:
        kern, tmpl, hm, t = _plan(sizes, inverse, split, row_threads_max(), BUF_FLOATS, rows is not None, VEC)
        inputs, grid = [x2, _signs_arg(signs), hm, *extra], (t, len(sizes), b)
    _check_launch(t, lay)
    ys = kern(inputs=inputs, template=tmpl, grid=grid, threadgroup=(t, 1, 1),
              output_shapes=shapes, output_dtypes=[out_dtype] * len(shapes))
    if split:
        return [y.reshape(*lead, n) for y, n in zip(ys, sizes)]
    return ys[0].reshape(*lead, n_tot)



# ------------------------------------------------------------------------- element-wise prologues (K6)
# The input rotation of a projection fused into the element-wise op that produces its input, one launch instead
# of two (the op's own launch + the rotation's):
#   PRO = 1  RMSNorm   xn = T(w · T(x · rsqrt(mean(x²) + eps)))   (the rounding of mx.fast.rms_norm)
#   PRO = 2  SwiGLU    xn = T(silu(g) · u)                         (float32 math, one rounding)
# then x̃ = H (σ ⊙ float(xn)) in float32, rounded to T — exactly what rotating the op's T-valued output does.
# with_input also writes xn (the unrotated input that column / CSR residuals read); otherwise it is not stored.
# One threadgroup per row (T = min(256, p) threads; the RMSNorm sum of squares is a simd_sum + threadgroup sum).
_PRO_SRC = """
    constexpr int PT = P / T;
    constexpr int E = M * PT;
    constexpr uint N = M * P;
    uint t = thread_position_in_threadgroup.x;
    uint b = threadgroup_position_in_grid.z;
    threadgroup float buf[BUFN];
    threadgroup float red[32];
    float v[E];
    if (PRO == 1) {
        SUC_UNROLL
        for (int e = 0; e < E; e += C) suc_ld<C>(x + b * N + suc_idx<P, T, C>(e, t), v + e);
        float ss = 0.0f;
        if (N <= 4096 && N <= BUFN) {        // MLX's order: the normed output is bit-identical to mx.fast.rms_norm
            SUC_UNROLL
            for (int e = 0; e < E; ++e) buf[suc_idx<P, T, C>(e, t)] = v[e];
            ss = suc_mlx_sumsq<N, T>(buf, red, t);
        } else {
            SUC_UNROLL
            for (int e = 0; e < E; ++e) ss += v[e] * v[e];
            ss = suc_tg_sum<T>(ss, red, t);
        }
        float r = metal::precise::rsqrt(ss / float(N) + eps[0]);
        SUC_UNROLL
        for (int e = 0; e < E; e += C) {
            float wv[C];
            suc_ld<C>(w + suc_idx<P, T, C>(e, t), wv);
            SUC_UNROLL
            for (int l = 0; l < C; ++l) v[e + l] = float(TX(wv[l] * float(TX(v[e + l] * r))));
        }
    } else {
        SUC_UNROLL
        for (int e = 0; e < E; e += C) {
            uint i = b * N + suc_idx<P, T, C>(e, t);
            float g[C], u[C];
            suc_ld<C>(x + i, g);
            suc_ld<C>(w + i, u);
            SUC_UNROLL
            for (int l = 0; l < C; ++l) v[e + l] = float(TX(g[l] / (1.0f + metal::exp(-g[l])) * u[l]));
        }
    }
    SUC_UNROLL
    for (int e = 0; e < E; e += C) {
        uint i = suc_idx<P, T, C>(e, t);
        suc_st<C>(xn + b * N + i, v + e);
        float sg[C];
        suc_ld<C>(signs + i, sg);
        SUC_UNROLL
        for (int l = 0; l < C; ++l) v[e + l] *= sg[l];
    }
    suc_rht_core<M, P, T, BUFN, HC, 0>(v, hm, buf, t);
    simdgroup_barrier(mem_flags::mem_none);   // see suc_rht_row
    const float sc = rsqrt(float(P));
    SUC_UNROLL
    for (int e = 0; e < E; e += C) {
        float o[C];
        SUC_UNROLL
        for (int l = 0; l < C; ++l) o[l] = v[e + l] * sc;
        suc_st<C>(y + b * N + suc_idx<P, T, C>(e, t), o);
    }
"""


_XN_STORE = "        suc_st<C>(xn + b * N + i, v + e);\n"


# Blocks layout (R5): every threadgroup (output block g) loads the whole row anyway, so it computes the RMSNorm sum
# of squares itself (same values, same order in every threadgroup: identical r everywhere) — no grid-wide step.
# xn (with_input) is written by the threadgroup whose own block it is.
_PRO_BLOCKS_SRC = """
    constexpr int NSG = T / 32;
    constexpr int R = M * (P / Q);
    constexpr int K = (R + NSG - 1) / NSG;
    constexpr uint N = M * P;
    uint t = thread_position_in_threadgroup.x;
    uint g = threadgroup_position_in_grid.y;
    uint b = threadgroup_position_in_grid.z;
    const uint sg = t / 32, lane = t % 32;
    threadgroup float buf[BUFN];
    threadgroup float red[32];
    float v[K * C];
    if (PRO == 1) {
        SUC_UNROLL
        for (int k = 0; k < K; ++k) {
            uint r = sg + k * NSG;
            if (R % NSG == 0 || r < R) suc_ld<C>(x + b * N + r * Q + lane * C, v + k * C);
        }
        float ss = 0.0f;
        if (N <= 4096 && N <= BUFN) {        // MLX's order: the normed output is bit-identical to mx.fast.rms_norm
            SUC_UNROLL
            for (int k = 0; k < K; ++k) {
                uint r = sg + k * NSG;
                if (R % NSG == 0 || r < R)
                    SUC_UNROLL
                    for (int l = 0; l < C; ++l) buf[r * Q + lane * C + l] = v[k * C + l];
            }
            ss = suc_mlx_sumsq<N, T>(buf, red, t);
        } else {
            SUC_UNROLL
            for (int k = 0; k < K; ++k)
                if (R % NSG == 0 || sg + k * NSG < R)
                    SUC_UNROLL
                    for (int l = 0; l < C; ++l) ss += v[k * C + l] * v[k * C + l];
            ss = suc_tg_sum<T>(ss, red, t);
        }
        float rs = metal::precise::rsqrt(ss / float(N) + eps[0]);
        SUC_UNROLL
        for (int k = 0; k < K; ++k) {
            uint r = sg + k * NSG;
            if (R % NSG == 0 || r < R) {
                float wv[C];
                suc_ld<C>(w + r * Q + lane * C, wv);
                SUC_UNROLL
                for (int l = 0; l < C; ++l) v[k * C + l] = float(TX(wv[l] * float(TX(v[k * C + l] * rs))));
            }
        }
    } else {
        SUC_UNROLL
        for (int k = 0; k < K; ++k) {
            uint r = sg + k * NSG;
            if (R % NSG == 0 || r < R) {
                uint i = b * N + r * Q + lane * C;
                float gv[C], uv[C];
                suc_ld<C>(x + i, gv);
                suc_ld<C>(w + i, uv);
                SUC_UNROLL
                for (int l = 0; l < C; ++l) v[k * C + l] = float(TX(gv[l] / (1.0f + metal::exp(-gv[l])) * uv[l]));
            }
        }
    }
    SUC_UNROLL
    for (int k = 0; k < K; ++k) {
        uint r = sg + k * NSG;
        if (R % NSG == 0 || r < R) {
        if (r == g) suc_st<C>(xn + b * N + r * Q + lane * C, v + k * C);
            float sgn[C];
            suc_ld<C>(signs + r * Q + lane * C, sgn);
            SUC_UNROLL
            for (int l = 0; l < C; ++l) v[k * C + l] *= sgn[l];
        }
    }
    suc_blocks_finish<M, P, Q, T, C, 0>(v, signs, y + b * N, g, buf, t);
"""
_XN_BLOCKS_STORE = "        if (r == g) suc_st<C>(xn + b * N + r * Q + lane * C, v + k * C);\n"


@lru_cache(maxsize=None)
def _pro_blocks_kernel(outn: bool):
    src = _PRO_BLOCKS_SRC if outn else _PRO_BLOCKS_SRC.replace(_XN_BLOCKS_STORE, "")
    return mx.fast.metal_kernel(name=f"suc_rht_prologue_blocks{'_n' if outn else ''}",
                                input_names=["x", "w", "signs", "eps"],
                                output_names=["y", "xn"] if outn else ["y"], source=src, header=_HEADER)


@lru_cache(maxsize=None)
def _pro_kernel(outn: bool):
    return mx.fast.metal_kernel(name=f"suc_rht_prologue{'_n' if outn else ''}",
                                input_names=["x", "w", "signs", "hm", "eps"],
                                output_names=["y", "xn"] if outn else ["y"],
                                source=_PRO_SRC if outn else _PRO_SRC.replace(_XN_STORE, ""), header=_HEADER)


@lru_cache(maxsize=None)
def _eps_array(eps: float) -> mx.array:
    a = mx.array([eps], dtype=mx.float32)
    mx.eval(a)
    return a


def _prologue(pro: int, x: mx.array, w: mx.array, signs: mx.array, eps: float, with_input: bool):
    n = x.shape[-1]
    if not supported(n):
        raise ValueError(f"unsupported rotation size {n}")
    m, p = _factor(n)
    lead = x.shape[:-1]
    x2, w2 = x.reshape(-1, n), (w if pro == 1 else w.reshape(-1, n))
    b = x2.shape[0]
    nout = 2 if with_input else 1
    if layout_for(n) == "blocks":
        q, t = blocks_shape((p,), (m,), *block_config(n))
        _check_launch(t, "blocks")
        outs = _pro_blocks_kernel(with_input)(
            inputs=[x2, w2, _signs_arg(signs), _eps_array(float(eps))],
            template=[("T", t), ("Q", q), ("C", q // 32), ("BUFN", max(min(n, 4096), (t // 32) * q)), ("M", m), ("P", p),
                      ("PRO", pro), ("TX", x.dtype)],
            grid=(t, n // q, b), threadgroup=(t, 1, 1), output_shapes=[(b, n)] * nout,
            output_dtypes=[x.dtype] * nout)
        outs = [o.reshape(*lead, n) for o in outs]
        return (outs[0], outs[1]) if with_input else (outs[0], None)
    tmax = row_threads_max()
    _, tmpl, hm, t = _plan((n,), False, False, tmax, BUF_FLOATS, False, VEC)
    c = threads_and_vec((p,), tmax, VEC)[1][0]
    _check_launch(t, "row")
    outs = _pro_kernel(with_input)(
        inputs=[x2, w2, _signs_arg(signs), hm, _eps_array(float(eps))],
        template=[("T", t), ("BUFN", BUF_FLOATS), ("HC", int(HM_CONST)), ("M", m), ("P", p), ("C", c), ("PRO", pro),
                  ("TX", x.dtype)],
        grid=(t, 1, b), threadgroup=(t, 1, 1), output_shapes=[(b, n)] * (2 if with_input else 1),
        output_dtypes=[x.dtype] * (2 if with_input else 1))
    outs = [o.reshape(*lead, n) for o in outs]
    return (outs[0], outs[1]) if with_input else (outs[0], None)


def rms_norm_rht(x: mx.array, weight: mx.array, eps: float, signs: mx.array, *, with_input: bool = False):
    """(x̃ = H (σ ⊙ rms_norm(x)), rms_norm(x) or None) in x.dtype, one launch (== rht_fused(mx.fast.rms_norm(x,
    weight, eps), signs) up to the float32 summation order of the mean square)."""
    return _prologue(1, x, weight.astype(x.dtype), signs, eps, with_input)


def swiglu_rht(gate: mx.array, up: mx.array, signs: mx.array, *, with_input: bool = False):
    """(x̃ = H (σ ⊙ h), h or None), h = silu(gate) · up rounded once to gate.dtype, one launch."""
    if gate.shape != up.shape or gate.dtype != up.dtype:
        raise ValueError(f"gate {gate.shape} {gate.dtype} and up {up.shape} {up.dtype} must match")
    return _prologue(2, gate, up, signs, 0.0, with_input)

def rht_fused(x: mx.array, signs: mx.array, out_dtype: mx.Dtype | None = None) -> mx.array:
    """V x = H (σ ⊙ x) in one launch (== suc.quant.hadamard.rht with the same signs, float32 math)."""
    return rotate(x, signs, inverse=False, out_dtype=out_dtype)


def rht_inverse_fused(y: mx.array, signs: mx.array, sizes: tuple[int, ...] | None = None,
                      out_dtype: mx.Dtype | None = None, split: bool = False) -> mx.array | list[mx.array]:
    """σ ⊙ (Hᵀ y) per segment in one launch (== suc.quant.hadamard.rht_inverse per segment)."""
    return rotate(y, signs, sizes, inverse=True, out_dtype=out_dtype, split=split)
