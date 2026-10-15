"""Direct VQ / additive-codebook matmul y = x Ŵᵀ from the compressed form (K1/K2/K4a, spec §19–§21, README §5).

Representation: M codebooks C_m (K, G) fp16, indices i (M, out, in/G) < K = 2**bits, row scales s (out,), with
    Ŵ[r, gG:(g+1)G] = s_r · Σ_m C_m[i_{m,r,g}]          (M = 1: plain vector codebook)
No rotation, no residual (see suc.kernels.suc_linear for the rotated module). Ŵ is never materialized.

Strategies of vq_matmul_metal (M = 1) — additive_matmul_metal always uses "lookup" (path= selects vec / mma):
  "lookup" — dispatch per batch tile: "lookup_vec" for tiles < MMA_MIN_BATCH rows, "lookup_mma" from it.
  "lookup_mma" (K3, Kernel B with matrix units) — fused decode + simdgroup_matrix 8x8 MMA, float32 accumulators.
      MMA_KIND = "reg" (default): each lane decodes exactly the two Ŵ elements it owns in the A fragment of
      Yᵀ = Ŵ·Xᵀ (a lane owns PG whole groups per K block: whole-entry codebook reads, one index fetch per PG
      groups; PG = 2 for odd G, up to 4 for G = 2, see "Low G"), x fragments are read straight
      from device memory and shared by MMA_RT row tiles; no threadgroup memory and no barriers.
      MMA_KIND = "tg": Ŵ tile (8·MMA_RT rows × ~MMA_KC cols) decoded per simdgroup into threadgroup memory, x
      tile staged per threadgroup, two barriers per K chunk (the textbook layout; kept for A/B).
      Tile element type MMA_TILE: half when x is float16 and M = 1 (Ŵ exact in half), else float32.
  "lookup_vec" (K2, Kernels B/C of §19, strategies B/C of §20) — one Metal kernel. Each simdgroup owns R output
      rows processed TOGETHER; its 32 lanes stride over whole chunks of uint32 index words (lcm(bits, 32) bits:
      one word of 32/bits groups when bits divides 32, e.g. 3 words / 16 groups for 6 bits; unaligned widths at
      larger batch tiles: one group per step with on-the-fly unpacking, see "Low G"). Per group a lane reads x_g once into
      registers (half4/float4 vector loads) and reuses it for all R rows; the M codebook entries are read as
      half4 vectors (from device memory through the GPU caches by default, CODEBOOK_IN) and summed before the
      dot product. float32 accumulators acc[R][BT], simd_sum at the end. M·K·G·2 ≤ 16 KB is enforced (D4).
  "lookup_k1" — the K1 kernel kept as the measured baseline (one row at a time, one index per lane step,
      scalar codebook/x reads, codebook staged in threadgroup memory). M = 1 only.
  "table" (Strategy D, §20) — MLX + Metal. MLX matmul builds T[g, k, b] = <C[k], x[b, gG:(g+1)G]> in float32;
      a Metal kernel then computes y[b, r] = s_r · Σ_g T[g, i_{r,g}, b] with gathers and additions only.
Batches larger than MAX_BATCH_TILE are processed in tiles of MAX_BATCH_TILE rows (per-thread accumulators).

K2 measurements, interleaved A/B runs on an Apple M3 whose GPU was shared with other jobs (indicative only;
min / median µs, 12288x4096, B = 1 unless noted; scratch A/B runs + benchmarks/k2_suc_linear.py):
  + R rows per simdgroup together, x_g loaded once into registers per lane step and shared by the R rows, whole
    uint32 index words, half4 codebook/x loads: ~2x over K1 overall (G=8: K1 ~1050-1350 vs ~520-740 with the
    threadgroup codebook in the same contended runs). R = 4 is best at B = 1 (R = 1/2/8: 550/480/690 vs 310),
    R = 2 at B = 4, R = 1 at B = 16 (register pressure of acc[R][BT]) -> LOOKUP_ROWS.
  + codebook read from device memory instead of staged in threadgroup memory: G=8 M=1 ~310 vs ~400-470,
    M=2 ~350-570 vs ~760-1040, M=3 ~590 vs ~1040 (staging = 4-12 KB threadgroup memory per threadgroup lowers
    occupancy, plus random-entry bank conflicts) -> "device" on the M3; the M1 Ultra prefers "threadgroup" (K3 A/B),
    hence CODEBOOK_IN = "auto".
  ~ codebook replicated 2x/4x in threadgroup memory (lane picks copy lane % n, or lane / 8; with or without
    16-byte skew between copies): better than a single staged copy for M = 1 (rep4 ~165-305 vs ~250-470) but
    not better than device reads, and impossible for M ≥ 2 within 16 KB -> dropped.
  - codebook pre-converted to float32 in threadgroup memory (8 KB for K=256 G=8): slower (~540-1050).
  - B = 16 stays ~2x slower than MLX q4 (compute/L1-bound on x re-reads, no simdgroup-matrix use) -> K3.

K3 measurements, Apple M3 laptop, GPU otherwise idle, warm interleaved A/B (8 copies cycled; indicative only;
median µs, 12288x4096, M = 1 G = 8 unless noted; scratch A/B runs):
  + "lookup_mma", MMA_KIND = "reg" vs "tg": reg is faster everywhere (B=4: 350 vs 545, B=16: 575 vs 921) — the tg
    variant's two barriers per K chunk and 10-37 KB of threadgroup memory serialize decode and MMA.
  + reg with the K permutation (one whole codebook entry per lane, half4 reads, one index fetch per 4·G columns):
    B=4 350 -> 306 (= vec 305), B=8 289 (vec 621, q4 809), B=16 643 (vec 2054, q4 1192). G=4 B=16 563 vs q4 1155;
    M=2 B=16 691 vs q4 1160. The reg time is ~flat for B=1..8 (≈ the FMA cost of one 8-wide batch tile, ~225 µs on
    the M3 at peak): the mma path is compute-bound, the vec path wins below ~5 rows.
  ~ crossover vec/mma on the M3: B=4 tie for G=8 and M=2, vec 17% faster for G=4 (278 vs 334); B=5/6: mma faster
    (G=4: 354/335 vs 415/455) -> MMA_MIN_BATCH = 5; A/B 4 / 5 / 8 on the M1 Ultra (k3_cold_cache.py times
    the other path as <suc>_vec / <suc>_mma at every B ≥ 4).
  ~ MMA_RT / MMA_SG: RT=4 best (RT=1: +50%, RT=2: +10-20%, RT=8 spills registers); SG 2/4/8 equal. MMA_TILE
    float instead of half for M = 1: +5-15%.
  ~ batch-1 vec switches (LOOKUP_VEC 2/4 = uint2/uint4 index loads, LOOKUP_SG 4/2, R = 2/8, LOOKUP_HALF_MATH):
    none faster on the M3 (VEC=4: 215 vs 146 for G=8; others within noise); kept off. They target the M1 Ultra,
    where the vec kernel is ALU/issue-bound (213 GB/s at 2 bpw) rather than bandwidth-bound.

Low G (Q5c: single codebooks G = 2, K = 256 / 64 / 16 and G = 3, K = 512 / 64 — 8-, 6-, 4-, 9-, 6-bit indices).
Apple M3 laptop, cold cache, interleaved A/B against the K3 kernels (scratch runs; GPU partly shared with other jobs,
so ±10% noise; median µs, 12288x4096 and 4096x12288; G = 3 only on 4096x12288 since 4096 % 3 != 0):
  + vec, word chunks: a lane step reads the CW = BITS / gcd(BITS, 32) words of lcm(BITS, 32) bits and unpacks its
    32 / gcd groups at compile-time offsets (before: unaligned widths unpacked one group per step with suc_unpack,
    i.e. 1-2 loads + variable shifts per group). 6-bit G=2 B=1: 467-575 -> 220-250 (-50%), B=4: 650-1070 -> 360-600
    (-40%); 6-bit G=3 B=1: 330-420 -> 200-250 (-40%). Kept for even G up to BT = 6 and for odd G at BT ≤ 2 with
    CW ≤ 3: beyond, the unrolled chunk (and the lane-strided x reads of odd G) cost more than the unpacking saves
    (6-bit G=2 B=8/16: +40% / +130%; 9-bit G=3 B=4: +180%) -> the per-group path there.
  - vec, 32 consecutive groups per simdgroup step with the BITS words loaded once and handed to the lanes by
    simd_shuffle (x reads coalesced): slower than the per-group path everywhere (+20-40%) -> dropped.
  + vec, x read in half4 spans of 2 groups for G = 2 (XS): 6-bit B=2 -30% vs one half2 per group; within noise
    elsewhere. 4-group spans for G = 3 (three half4): slower at B ≤ 2 (+60% vs 1 group) -> G = 3 reads per group.
  + half2 x / codebook reads for G % 4 == 2 (before: scalar half loads): within noise on the bandwidth-bound M3 at
    B = 1 (G=2 K=256 ~300 µs at 4 bpw ≈ the M3's ~85 GB/s, as q4), meant for the ALU-bound M1 Ultra.
  + mma "reg", PG groups per lane and block (a whole PG·G-column span per lane, indices of the PG groups from one
    word pair): odd G (PG = 2) replaces the per-element natural-order path: G=3 B=4/8 -50%, B=16 -25..-35%
    (9-bit 1234 -> 835, 6-bit 1081 -> 812). G = 2: PG = 4 up to NB = 1 (B ≤ 8: -15% 8-bit, -35% 6-bit, -8% 4-bit),
    PG = 1 at NB = 2 (PG 2 / 4: +5..+14% at B = 16, registers) -> PGG2.
  - "table" (Strategy D) for small K, unfused (MLX builds T[g, k, b], Metal gathers): 2-20x slower than lookup at
    every B for K = 16 / 64 (G=2 K=16 B=1 364 vs 154 µs; T is written to and re-read from memory). A fused table
    built in threadgroup memory per K chunk costs NG·K·BT·G MACs per threadgroup, the same as the lookup MACs of
    K rows, so it can only win with ≫ K rows per threadgroup; not pursued.
"""
from __future__ import annotations

from typing import Literal

import mlx.core as mx

from suc.quant.packing import words_per_row

CODEBOOK_TG_LIMIT = 16 * 1024   # bytes, D4 (all codebooks a threadgroup stages, together)
MAX_BATCH_TILE = 16             # batch rows per kernel launch (register accumulators per thread)
SIMDGROUPS_PER_TG = 8           # 256 threads per threadgroup
ROWS_PER_SIMDGROUP = 4          # K1 kernels: output rows handled sequentially by one simdgroup
# K2 lookup: rows processed together per simdgroup, by batch tile size (register budget R·(BT + G) floats)
LOOKUP_ROWS = {1: 4, 2: 4, 4: 2, 8: 2, 16: 1}
# K3 lookup ("vec" path, B < MMA_MIN_BATCH) batch-1 A/B switches (see module docstring, "K3"):
LOOKUP_SG = 8                   # simdgroups per threadgroup of the vec kernel
LOOKUP_VEC = 1                  # consecutive uint32 index words per lane step (1, 2, 4; 4 = one uint4 load)
LOOKUP_HALF_MATH = False        # fp16 x, M = 1, G % 4 == 0: half4 products + dot(), float accumulation
# K3 simdgroup-matrix ("mma") path: fused decode + 8x8 simdgroup MMA, used for batch tiles >= MMA_MIN_BATCH
MMA_MIN_BATCH_DEFAULT = 5       # M3 cold-cache: vec ≤ mma at B = 4 (decode block 1301 vs 1818 µs, G=4); A/B on M1 Ultra
# None: per device and M from suc.kernels.autotune ("mma_min_batch"), else MMA_MIN_BATCH_DEFAULT; an int forces it
MMA_MIN_BATCH: int | None = None
MMA_SG = 4                      # simdgroups per threadgroup
MMA_RT = 4                      # 8-row output tiles per simdgroup (rows per threadgroup = 8 * MMA_RT * MMA_SG)
MMA_KC = 64                     # target input columns per K chunk (rounded to whole index words / 8 columns)
# Ŵ/x tile element type: "auto" = half when x is float16 and M = 1 (Ŵ tile exact), else float32
MMA_TILE: Literal["auto", "half", "float"] = "auto"
# "reg": Ŵ decoded straight into simdgroup_matrix fragments (no threadgroup memory / barriers);
# "tg": Ŵ tile decoded into threadgroup memory, x tile staged per threadgroup
MMA_KIND: Literal["reg", "tg"] = "reg"
# K2 lookup: where lanes read codebook entries ("device": straight from device memory through the GPU caches;
# "threadgroup": staged per threadgroup; "auto": per GPU generation, see resolve_codebook_in). The D4 16 KB
# ceiling bounds the codebook bytes either way.
CODEBOOK_IN: Literal["auto", "device", "threadgroup"] = "auto"
# GPU architecture prefixes (mx.metal.device_info()["architecture"]) where staging wins. M1 Ultra (g13d), K3
# cold-cache A/B: threadgroup codebook is 1.2-3x faster per 8B block at B = 1/4 (M=3: 1049 -> 353 µs), while on the
# M3 (g15) device reads were faster (K2 A/B above).
CODEBOOK_TG_ARCHS: tuple[str, ...] = ("applegpu_g13",)


def resolve_codebook_in(codebook_in: str, architecture: str | None = None, *, key: str | None = None,
                        m: int = 1) -> str:
    """Resolve "auto" to "device" | "threadgroup".

    Order: the per-device measurement cached by suc.kernels.autotune (key = "<architecture>|<device name>",
    nearest measured M), else the built-in per-architecture default (CODEBOOK_TG_ARCHS)."""
    if codebook_in not in ("auto", "device", "threadgroup"):
        raise ValueError(f"codebook_in must be 'auto', 'device' or 'threadgroup', got {codebook_in!r}")
    if codebook_in != "auto":
        return codebook_in
    from suc.kernels import autotune

    if key is None and architecture is None:
        key = autotune.device_key()
    if key is not None:
        cached = autotune.lookup(key, m)
        if cached is not None:
            return cached
        architecture = architecture or key.split("|")[0]
    return "threadgroup" if architecture.startswith(CODEBOOK_TG_ARCHS) else "device"


def mma_min_batch(m: int = 1) -> int:
    """Batch tile from which the simdgroup-matrix kernel is used (MMA_MIN_BATCH, else the tuned crossover for M
    codebooks, else MMA_MIN_BATCH_DEFAULT)."""
    if MMA_MIN_BATCH is not None:
        return MMA_MIN_BATCH
    from suc.kernels import autotune
    return autotune.mma_min_batch_choice(m) or MMA_MIN_BATCH_DEFAULT


def _mma_kcg(g: int, bits: int) -> int:
    """Groups per K chunk: a multiple of the groups per index word (aligned bits) with KCG·G % 8 == 0, ≈ MMA_KC."""
    import math
    gpu = 32 // bits if 32 % bits == 0 else 1
    step = math.lcm(gpu, 8 // math.gcd(g, 8))
    return max(step, -(-(MMA_KC // g) // step) * step)


def _lookup_rows(bt: int) -> int:
    return next(r for b, r in sorted(LOOKUP_ROWS.items()) if bt <= b)


def vq_decode(codebook: mx.array, indices: mx.array, row_scales: mx.array) -> mx.array:
    """Reference Ŵ (out, in) float32 from unpacked indices (out, in/G)."""
    out, ng = indices.shape
    k, g = codebook.shape
    w = codebook.astype(mx.float32)[indices.reshape(-1).astype(mx.uint32)].reshape(out, ng * g)
    return row_scales.astype(mx.float32)[:, None] * w


def vq_matmul_reference(x: mx.array, codebook: mx.array, indices: mx.array, row_scales: mx.array) -> mx.array:
    """Oracle: x (B, in) @ vq_decode(...)ᵀ in float32 → (B, out) float32."""
    return x.astype(mx.float32) @ vq_decode(codebook, indices, row_scales).T


def additive_decode(codebooks: mx.array, indices: mx.array, row_scales: mx.array) -> mx.array:
    """Reference Ŵ (out, in) float32 from M codebooks (M, K, G) and unpacked indices (M, out, in/G)."""
    m, out, ng = indices.shape
    cb = codebooks.astype(mx.float32)
    w = sum(cb[j][indices[j].reshape(-1).astype(mx.uint32)] for j in range(m)).reshape(out, ng * cb.shape[2])
    return row_scales.astype(mx.float32)[:, None] * w


def additive_matmul_reference(x: mx.array, codebooks: mx.array, indices: mx.array, row_scales: mx.array) -> mx.array:
    """Oracle: x (B, in) @ additive_decode(...)ᵀ in float32 → (B, out) float32."""
    return x.astype(mx.float32) @ additive_decode(codebooks, indices, row_scales).T


_HEADER = """
// element g of a little-endian packed row (suc.quant.packing); P is any uint32 pointer type
template <int BITS, typename P>
inline uint suc_unpack(P prow, uint g) {
    uint p = g * BITS, w = p >> 5, off = p & 31;
    uint v = prow[w] >> off;
    if (off + BITS > 32) v |= prow[w + 1] << (32 - off);
    return BITS == 32 ? v : (v & ((1u << BITS) - 1u));
}

// bits [off, off + BITS) of the word pair w[i], w[i + 1] (unmasked); i / off are compile-time in unrolled loops
template <int BITS>
inline uint suc_bits(thread const uint* w, uint i, uint off) {
    uint v = w[i] >> off;
    if (BITS < 32 && off + BITS > 32) v |= w[i + 1] << (32 - off);
    return v;
}

// float4 / float2 vector loads of half / float x from device or constant memory (MLX binds small inputs to the
// constant address space; overloads instead of casts, so a dead vector branch never casts across address spaces)
#define SUC_XLD(SPACE, TY)                                                                                   \
inline float4 suc_ld4(const SPACE TY* p) { return float4(*(const SPACE TY##4*)p); }                         \
inline float2 suc_ld2(const SPACE TY* p) { return float2(*(const SPACE TY##2*)p); }                         \
inline half4 suc_ldh4(const SPACE TY* p) { return half4(*(const SPACE TY##4*)p); }
template <typename P> inline float4 suc_ld4(P p) { return float4(float(p[0]), float(p[1]), float(p[2]), float(p[3])); }
template <typename P> inline float2 suc_ld2(P p) { return float2(float(p[0]), float(p[1])); }
template <typename P> inline half4 suc_ldh4(P p) { return half4(suc_ld4(p)); }
SUC_XLD(device, half)
SUC_XLD(constant, half)
SUC_XLD(device, float)
SUC_XLD(constant, float)

// o[0:G] = float(p[0:G]); vector loads when the element type has a native 4-vector and G % 4 == 0 (G % 2 == 0)
template <typename T, int G> struct XLoad {
    template <typename P> static inline void run(P p, thread float* o) {
        for (int j = 0; j < G; ++j) o[j] = float(p[j]);
    }
};
#define SUC_XLOAD(TY)                                                                                        \
template <int G> struct XLoad<TY, G> {                                                                       \
    template <typename P> static inline void run(P p, thread float* o) {                                     \
        if (G % 4 == 0) {                                                                                    \
            for (int j = 0; j < G / 4; ++j) {                                                                \
                float4 v = suc_ld4(p + 4 * j);                                                               \
                o[4 * j] = v.x; o[4 * j + 1] = v.y; o[4 * j + 2] = v.z; o[4 * j + 3] = v.w;                  \
            }                                                                                                \
        } else if (G % 2 == 0) {                                                                             \
            for (int j = 0; j < G / 2; ++j) {                                                                \
                float2 v = suc_ld2(p + 2 * j);                                                               \
                o[2 * j] = v.x; o[2 * j + 1] = v.y;                                                          \
            }                                                                                                \
        } else {                                                                                             \
            for (int j = 0; j < G; ++j) o[j] = float(p[j]);                                                  \
        }                                                                                                    \
    }                                                                                                        \
};
SUC_XLOAD(half)
SUC_XLOAD(float)

// c[0:G] += codebook entry e[0:G]; half4 / half2 vector reads when G % 4 / G % 2 == 0 (any address space)
#define SUC_CB_ADD(SPACE)                                                                     \
template <int G>                                                                              \
inline void cb_add(const SPACE half* e, thread float* c) {                                    \
    if (G % 4 == 0) {                                                                         \
        for (int j = 0; j < G / 4; ++j) {                                                     \
            float4 v = float4(((const SPACE half4*)e)[j]);                                    \
            c[4 * j] += v.x; c[4 * j + 1] += v.y; c[4 * j + 2] += v.z; c[4 * j + 3] += v.w;   \
        }                                                                                     \
    } else if (G % 2 == 0) {                                                                  \
        for (int j = 0; j < G / 2; ++j) {                                                     \
            float2 v = float2(((const SPACE half2*)e)[j]);                                    \
            c[2 * j] += v.x; c[2 * j + 1] += v.y;                                             \
        }                                                                                     \
    } else {                                                                                  \
        for (int j = 0; j < G; ++j) c[j] += float(e[j]);                                      \
    }                                                                                         \
}
SUC_CB_ADD(threadgroup)
SUC_CB_ADD(device)
SUC_CB_ADD(constant)
"""

# ------------------------------------------------------------------------------------------- K2 lookup
# Grid: threadgroup → SG*R consecutive rows, simdgroup → R rows processed together, lanes → index units.
# A unit is one uint32 word (ALIGNED: bits divides 32, GPU = 32/bits groups) or one group (unaligned bits).
_LOOKUP_SRC = """
    // Index units. CHUNK (aligned widths, BITS | 32, and the unaligned widths where it measured faster): a lane step
    // reads VEC chunks of CW consecutive uint32 words holding GPU whole groups (CW·32 = lcm(BITS, 32) bits: 6 bits →
    // 3 words / 16 groups, 9 bits → 9 words / 32 groups); group j of a chunk sits at a compile-time bit offset once
    // the j loop is unrolled, so every word is loaded once and unpacking is a constant shift / or / mask. Groups are
    // consumed XS at a time so that x is read in 4-element vectors (XS·G % 4 == 0: G = 2 → 2 groups = one half4,
    // G = 3 → 4 groups = three half4). Otherwise one group per lane step (lanes on consecutive groups, suc_unpack).
    constexpr uint GCD = (BITS & (~BITS + 1u)) < 32u ? (BITS & (~BITS + 1u)) : 32u;   // gcd(BITS, 32)
    constexpr bool ALIGNED = GCD == BITS;
    constexpr bool CHUNK = ALIGNED || (G % 2 == 0 && BT <= 6) || (BT <= 2 && BITS / GCD <= 3);
    constexpr uint CW = BITS / GCD;
    constexpr uint GPU = 32 / GCD;
    constexpr uint NU = (NG + GPU - 1) / GPU;
    constexpr uint MASK = BITS >= 32 ? 0xffffffffu : ((1u << BITS) - 1u);
    constexpr uint XS0 = G % 2 == 0 && G % 4 != 0 ? 2 : 1;   // odd G: 4-group spans measured slower (registers)
    constexpr uint XS = (!HM && IN % 4 == 0 && GPU % XS0 == 0) ? XS0 : 1;
    threadgroup half cbt[CBTG ? M * K * G : 2];
    if (CBTG) {
        for (uint i = thread_position_in_threadgroup.x; i < M * K * G; i += SG * 32) cbt[i] = codebook[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint lane = thread_index_in_simdgroup;
    uint row0 = (threadgroup_position_in_grid.x * SG + simdgroup_index_in_threadgroup) * R;
    if (row0 >= OUT) return;
    uint rows[R];
    for (int rr = 0; rr < R; ++rr) rows[rr] = min(row0 + rr, uint(OUT - 1));   // tail rows: computed, not stored
    float acc[R][BT];
    for (int rr = 0; rr < R; ++rr) for (int b = 0; b < BT; ++b) acc[rr][b] = 0.0f;

    auto prow_of = [&](int m, int rr) { return packed + (m * OUT + rows[rr]) * WORDS; };   // packed row of (m, rr)
    // acc[rr][b] += <Σ_m C_m[ix[m][s][rr]], x[b, (g + s)G : (g + s + 1)G]> over the S groups g .. g + S - 1
    auto groups_dot = [&](uint g, auto S, thread const uint (&ix)[M][XS][R]) {
        constexpr uint NS = decltype(S)::value;
        if (HM) {
            // half math (fp16 x, M = 1, G % 4 == 0): half4 products summed by dot(), float accumulation
            constexpr int G4 = G / 4 > 0 ? G / 4 : 1;
            half4 e[R][G4];
            for (int rr = 0; rr < R; ++rr) {
                const device half4* ep = (const device half4*)(codebook + ix[0][0][rr] * G);
                for (int q = 0; q < G4; ++q) e[rr][q] = ep[q];
            }
            for (int b = 0; b < BT; ++b) {
                half4 xv[G4];
                for (int q = 0; q < G4; ++q) xv[q] = suc_ldh4(x + b * IN + g * G + 4 * q);
                for (int rr = 0; rr < R; ++rr) {
                    float d = 0.0f;
                    for (int q = 0; q < G4; ++q) d += float(dot(e[rr][q], xv[q]));
                    acc[rr][b] += d;
                }
            }
            return;
        }
        float c[R][NS * G];
        for (int rr = 0; rr < R; ++rr) {
            for (uint q = 0; q < NS * G; ++q) c[rr][q] = 0.0f;
            for (uint sj = 0; sj < NS; ++sj)
                for (int m = 0; m < M; ++m) {
                    uint idx = ix[m][sj][rr];
                    if (CBTG) cb_add<G>(cbt + (m * K + idx) * G, c[rr] + sj * G);
                    else cb_add<G>(codebook + (m * K + idx) * G, c[rr] + sj * G);
                }
        }
        for (int b = 0; b < BT; ++b) {
            float xv[NS * G];
            XLoad<T, NS * G>::run(x + b * IN + g * G, xv);
            for (int rr = 0; rr < R; ++rr) {
                float d = 0.0f;
                for (uint q = 0; q < NS * G; ++q) d += c[rr][q] * xv[q];
                acc[rr][b] += d;
            }
        }
    };

    if (CHUNK) {
        // VEC consecutive units per lane step; VEC = 4 with one-word units and 16-byte aligned rows → one uint4 load
        constexpr bool V4 = CW == 1 && VEC == 4 && (WORDS % 4) == 0;
        for (uint u0 = lane * VEC; u0 < NU; u0 += 32 * VEC) {
            uint wd[M][R][VEC * CW];
            for (int m = 0; m < M; ++m)
                for (int rr = 0; rr < R; ++rr) {
                    auto prow = prow_of(m, rr);
                    if (V4) {
                        uint4 q = ((const device uint4*)prow)[u0 / 4];
                        wd[m][rr][0] = q.x; wd[m][rr][1] = q.y; wd[m][rr][2] = q.z; wd[m][rr][3] = q.w;
                    } else {
                        #pragma clang loop unroll(full)
                        for (uint v = 0; v < VEC * CW; ++v) wd[m][rr][v] = prow[min(u0 * CW + v, uint(WORDS - 1))];
                    }
                }
            #pragma clang loop unroll(full)
            for (uint v = 0; v < VEC; ++v) {
                uint u = u0 + v;
                if (VEC > 1 && u >= NU) break;
                #pragma clang loop unroll(full)
                for (uint j0 = 0; j0 < GPU; j0 += XS) {
                    uint g = u * GPU + j0;
                    if (NG % GPU != 0 && g >= NG) break;
                    // bit offsets of groups j0 .. j0 + XS - 1 inside the chunk (compile-time after unrolling)
                    uint ix[M][XS][R];
                    for (uint sj = 0; sj < XS; ++sj) {
                        const uint jw = v * CW + ((j0 + sj) * BITS) / 32, joff = ((j0 + sj) * BITS) % 32;
                        for (int m = 0; m < M; ++m)
                            for (int rr = 0; rr < R; ++rr) ix[m][sj][rr] = suc_bits<BITS>(wd[m][rr], jw, joff) & MASK;
                    }
                    if (XS == 1 || NG % XS == 0 || g + XS <= NG) {
                        groups_dot(g, integral_constant<uint, XS>(), ix);
                    } else {                       // partial last span (NG % XS != 0): one group at a time
                        for (uint sj = 0; sj < XS && g + sj < NG; ++sj) {
                            uint ix1[M][XS][R];
                            for (int m = 0; m < M; ++m)
                                for (int rr = 0; rr < R; ++rr) ix1[m][0][rr] = ix[m][sj][rr];
                            groups_dot(g + sj, integral_constant<uint, 1>(), ix1);
                        }
                    }
                }
            }
        }
    } else {
        for (uint g = lane; g < NG; g += 32) {
            uint ix[M][XS][R];
            for (int m = 0; m < M; ++m)
                for (int rr = 0; rr < R; ++rr) ix[m][0][rr] = suc_unpack<BITS>(prow_of(m, rr), g);
            groups_dot(g, integral_constant<uint, 1>(), ix);
        }
    }
    for (int rr = 0; rr < R; ++rr) {
        uint row = row0 + rr;
        float s = scales[rows[rr]];
        for (int b = 0; b < BT; ++b) {
            float t = simd_sum(acc[rr][b]);
            if (lane == 0 && row < OUT) y[b * OUT + row] = static_cast<TO>(t * s);
        }
    }
"""

# ------------------------------------------------------------------------------------------- K3 mma
# Grid: threadgroup → SG·8·RT consecutive rows, simdgroup → RT 8-row tiles × NB 8-row batch tiles (accumulators
# simdgroup_matrix<float, 8, 8> acc[RT][NB]). Per K chunk of KC = KCG·G input columns: all threads stage the
# x tile (BP × KC, zero-padded batch rows / columns) in threadgroup memory, each simdgroup decodes its own
# (8·RT) × KC tile of Ŵ / s (codes → Σ_m C_m entries; one uint32 index word per lane step for aligned bits),
# then 8x8 MMAs: Y[b, r] += X[b, k] · Ŵᵀ[k, r] (Ŵ tile loaded transposed). Row scales are applied at the store.
_MMA_SRC = """
    constexpr bool ALIGNED = (32 % BITS) == 0;
    constexpr uint GPU = ALIGNED ? 32 / BITS : 1;
    constexpr uint MASK = BITS >= 32 ? 0xffffffffu : ((1u << BITS) - 1u);
    constexpr uint KC = KCG * G;
    constexpr uint RW = 8 * RT;
    constexpr uint BP = 8 * NB;
    threadgroup TT xs[BP * KC];
    threadgroup TT ws[SG * RW * KC];
    threadgroup float st[SG * 64];
    threadgroup half cbt[CBTG ? M * K * G : 2];
    uint tid = thread_position_in_threadgroup.x;
    if (CBTG) {
        for (uint i = tid; i < M * K * G; i += SG * 32) cbt[i] = codebook[i];
    }
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint row0 = (threadgroup_position_in_grid.x * SG + sg) * RW;
    threadgroup TT* wt = ws + sg * RW * KC;

    simdgroup_matrix<float, 8, 8> acc[RT][NB];
    for (int rt = 0; rt < RT; ++rt)
        for (int nb = 0; nb < NB; ++nb) acc[rt][nb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint g0 = 0; g0 < NG; g0 += KCG) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint c0 = g0 * G;
        for (uint i = tid; i < BP * KC; i += SG * 32) {
            uint b = i / KC, c = i % KC;
            xs[i] = (b < BT && c0 + c < IN) ? TT(x[b * IN + c0 + c]) : TT(0);
        }
        if (ALIGNED) {
            constexpr uint WPC = KCG / GPU;
            for (uint i = lane; i < RW * WPC; i += 32) {
                uint rr = i / WPC, w = i % WPC, row = row0 + rr, gw = g0 / GPU + w;
                threadgroup TT* dst = wt + rr * KC + w * GPU * G;
                bool live = row < OUT && gw < WORDS;
                uint wd[M];
                for (int m = 0; m < M; ++m) wd[m] = live ? packed[(m * OUT + row) * WORDS + gw] : 0u;
                for (uint j = 0; j < GPU; ++j) {
                    float c[G];
                    for (int q = 0; q < G; ++q) c[q] = 0.0f;
                    if (live && gw * GPU + j < NG) {
                        for (int m = 0; m < M; ++m) {
                            uint idx = (wd[m] >> (j * BITS)) & MASK;
                            if (CBTG) cb_add<G>(cbt + (m * K + idx) * G, c);
                            else cb_add<G>(codebook + (m * K + idx) * G, c);
                        }
                    }
                    for (int q = 0; q < G; ++q) dst[j * G + q] = TT(c[q]);
                }
            }
        } else {
            for (uint i = lane; i < RW * KCG; i += 32) {
                uint rr = i / KCG, j = i % KCG, row = row0 + rr, g = g0 + j;
                float c[G];
                for (int q = 0; q < G; ++q) c[q] = 0.0f;
                if (row < OUT && g < NG) {
                    for (int m = 0; m < M; ++m) {
                        uint idx = suc_unpack<BITS>(packed + (m * OUT + row) * WORDS, g);
                        if (CBTG) cb_add<G>(cbt + (m * K + idx) * G, c);
                        else cb_add<G>(codebook + (m * K + idx) * G, c);
                    }
                }
                threadgroup TT* dst = wt + rr * KC + j * G;
                for (int q = 0; q < G; ++q) dst[q] = TT(c[q]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < KC; kk += 8) {
            simdgroup_matrix<TT, 8, 8> a[NB], bm[RT];
            for (int nb = 0; nb < NB; ++nb) simdgroup_load(a[nb], xs + nb * 8 * KC + kk, KC);
            for (int rt = 0; rt < RT; ++rt) simdgroup_load(bm[rt], wt + rt * 8 * KC + kk, KC, ulong2(0, 0), true);
            for (int rt = 0; rt < RT; ++rt)
                for (int nb = 0; nb < NB; ++nb) simdgroup_multiply_accumulate(acc[rt][nb], a[nb], bm[rt], acc[rt][nb]);
        }
    }
    threadgroup float* sp = st + sg * 64;
    for (int rt = 0; rt < RT; ++rt)
        for (int nb = 0; nb < NB; ++nb) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc[rt][nb], sp, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                uint b = nb * 8 + e / 8, row = row0 + rt * 8 + e % 8;
                if (b < BT && row < OUT) y[b * OUT + row] = static_cast<TO>(sp[e] * scales[row]);
            }
        }
"""

# K3 mma, register-fragment variant ("reg"): no threadgroup memory, no barriers. Computes Yᵀ = Ŵ · Xᵀ per
# simdgroup with A = Ŵ tile (8 rows × 8 cols), B = Xᵀ tile (8 cols × 8 batch), D = Yᵀ (8 rows × 8 batch).
# Lane ℓ owns fragment elements (fm, fn), (fm, fn + 1) with qid = ℓ/4, fm = (qid & 4) + (ℓ/2) % 4,
# fn = (qid & 2)·2 + (ℓ % 2)·2 (the simdgroup_matrix<T, 8, 8> thread_elements() layout used by MLX steel):
# it decodes Ŵ[r0 + fm, k0 + fn : k0 + fn + 2] straight from its index and codebook entry (one half2 read per
# codebook when G is even), reads x[b0 + fn (+1), k0 + fm], and stores y[b, r0 + fm] from D. The x fragments
# are shared by the RT row tiles of the simdgroup; the index word is reloaded per k-step (L1 hit).
_MMA_REG_SRC = """
    // groups per lane and block for G = 2: 4 (one 8-column word span per lane) up to 8 batch rows; at NB ≥ 2 the
    // c[RT][8] registers cost more than the saved index fetches (M3 A/B 1 / 2 / 4, module docstring "Low G")
    constexpr uint PGG2 = NB >= 2 ? 1 : 4;
    constexpr bool ALIGNED = (32 % BITS) == 0;
    constexpr uint GPU = ALIGNED ? 32 / BITS : 1;
    constexpr uint MASK = BITS >= 32 ? 0xffffffffu : ((1u << BITS) - 1u);
    constexpr uint RW = 8 * RT;
    threadgroup half cbt[CBTG ? M * K * G : 2];
    if (CBTG) {
        for (uint i = thread_position_in_threadgroup.x; i < M * K * G; i += SG * 32) cbt[i] = codebook[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint row0 = (threadgroup_position_in_grid.x * SG + sg) * RW;
    if (row0 >= OUT) return;
    uint qid = lane / 4;
    uint fm = (qid & 4) + (lane / 2) % 4;
    uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    uint rows[RT];
    bool rlive[RT];
    for (int rt = 0; rt < RT; ++rt) {
        uint r = row0 + rt * 8 + fm;
        rlive[rt] = r < OUT;
        rows[rt] = min(r, uint(OUT - 1));
    }
    simdgroup_matrix<float, 8, 8> acc[RT][NB];
    for (int rt = 0; rt < RT; ++rt)
        for (int nb = 0; nb < NB; ++nb) acc[rt][nb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    auto prow_of = [&](uint m, uint row) { return packed + (m * OUT + row) * WORDS; };   // packed row of (m, row)
    auto index_of = [&](uint m, uint row, uint gi) -> uint {
        auto prow = prow_of(m, row);
        return ALIGNED ? (prow[gi / GPU] >> ((gi % GPU) * BITS)) & MASK : suc_unpack<BITS>(prow, gi);
    };
    // K permuted inside each block of 4·PG groups (4·PG·G columns, PG·G/2 k-steps): lane (fm, fn = 2f) owns the PG
    // consecutive groups f·PG .. f·PG + PG - 1 of the block (PG·G contiguous columns); in k-step p its fragment
    // columns fn + e are physical columns blk0 + f·PG·G + 2p + e. One index fetch per PG groups (PG > 1: the PG
    // indices come from one word pair) and whole-entry codebook reads (G/4 half4 or G/2 half2). The same
    // permutation on the x fragment (row fm ↦ blk0 + (fm/2)·PG·G + 2p + fm%2) leaves the MMA sum unchanged.
    // PG = 1 for even G ≥ 4 (the K3 layout), 2 for odd G (a pair of columns never straddles an owned span), PGG2
    // for G = 2.
    constexpr uint PG = G % 2 != 0 ? 2 : (G == 2 ? PGG2 : 1);
    constexpr uint SPAN = PG * G;
    // indices of groups gi .. gi + PG - 1 of one row (PG·BITS ≤ 33 bits: within one word pair)
    auto indices_of = [&](uint m, uint row, uint gi, thread uint* ix) {
        auto prow = prow_of(m, row);
        if (PG == 1 || PG * BITS > 33) {
            for (uint t = 0; t < PG; ++t) ix[t] = gi + t < NG ? index_of(m, row, gi + t) : 0u;
            return;
        }
        uint p0 = gi * BITS, wi = p0 / 32, off0 = p0 % 32;
        uint lo = prow[wi], hi = wi + 1 < WORDS ? prow[wi + 1] : 0u;
        for (uint t = 0; t < PG; ++t) {
            uint off = off0 + t * BITS, v;
            if (off >= 32) v = hi >> (off - 32);
            else v = (lo >> off) | (off > 0 && off + BITS > 32 ? hi << (32 - off) : 0u);
            ix[t] = v & MASK;
        }
    };
    for (uint blk0 = 0; blk0 < IN; blk0 += 4 * SPAN) {
        uint gi = blk0 / G + (fn / 2) * PG;
        float c[RT][SPAN];
        for (int rt = 0; rt < RT; ++rt) {
            for (uint q = 0; q < SPAN; ++q) c[rt][q] = 0.0f;
            if (gi < NG) {
                for (int m = 0; m < M; ++m) {
                    uint ixs[PG];
                    indices_of(m, rows[rt], gi, ixs);
                    for (uint t = 0; t < PG; ++t) {
                        if (PG > 1 && gi + t >= NG) break;
                        uint idx = ixs[t];
                        if (CBTG) cb_add<G>(cbt + (m * K + idx) * G, c[rt] + t * G);
                        else cb_add<G>(codebook + (m * K + idx) * G, c[rt] + t * G);
                    }
                }
            }
        }
        for (uint p = 0; p < SPAN / 2; ++p) {
            simdgroup_matrix<TT, 8, 8> bx[NB];
            uint pc = blk0 + (fm / 2) * SPAN + 2 * p + (fm % 2);
            bool kin = pc < IN;
            for (int nb = 0; nb < NB; ++nb) {
                uint bb = nb * 8 + fn;
                bx[nb].thread_elements()[0] = (kin && bb < BT) ? TT(x[bb * IN + pc]) : TT(0);
                bx[nb].thread_elements()[1] = (kin && bb + 1 < BT) ? TT(x[(bb + 1) * IN + pc]) : TT(0);
            }
            for (int rt = 0; rt < RT; ++rt) {
                simdgroup_matrix<TT, 8, 8> aw;
                aw.thread_elements()[0] = TT(c[rt][2 * p]);
                aw.thread_elements()[1] = TT(c[rt][2 * p + 1]);
                for (int nb = 0; nb < NB; ++nb)
                    simdgroup_multiply_accumulate(acc[rt][nb], aw, bx[nb], acc[rt][nb]);
            }
        }
    }
    for (int rt = 0; rt < RT; ++rt) {
        if (!rlive[rt]) continue;
        uint r = rows[rt];
        float s = scales[r];
        for (int nb = 0; nb < NB; ++nb) {
            uint b = nb * 8 + fn;
            float2 d = float2(acc[rt][nb].thread_elements()[0], acc[rt][nb].thread_elements()[1]);
            if (b < BT) y[b * OUT + r] = static_cast<TO>(d.x * s);
            if (b + 1 < BT) y[(b + 1) * OUT + r] = static_cast<TO>(d.y * s);
        }
    }
"""

# ------------------------------------------------------------------------------------------- K1 kernels
# Row mapping: threadgroup → SG*RPS consecutive output rows, simdgroup → RPS consecutive rows (one after another),
# lanes → input groups (stride 32). Negative result (M3, K1): staging x chunks in threadgroup memory was slower.
_ROW_LOOP_OPEN = """
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint row0 = (threadgroup_position_in_grid.x * SG + sg) * RPS;
    for (uint rr = 0; rr < RPS; ++rr) {
        uint row = row0 + rr;
        if (row >= OUT) break;
        const device uint32_t* prow = packed + row * WORDS;
        float acc[BT];
        for (int b = 0; b < BT; ++b) acc[b] = 0.0f;
"""

_ROW_LOOP_CLOSE = """
        float s = scales[row];
        for (int b = 0; b < BT; ++b) {
            float t = simd_sum(acc[b]);
            if (lane == 0) y[b * OUT + row] = static_cast<T>(t * s);
        }
    }
"""

_LOOKUP_K1_SRC = """
    threadgroup half cb[K * G];
    for (uint i = thread_position_in_threadgroup.x; i < K * G; i += SG * 32) cb[i] = codebook[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
""" + _ROW_LOOP_OPEN + """
        for (uint g = lane; g < NG; g += 32) {
            uint idx = suc_unpack<BITS>(prow, g);
            float c[G];
            for (int j = 0; j < G; ++j) c[j] = float(cb[idx * G + j]);
            for (int b = 0; b < BT; ++b) {
                const device T* xp = x + b * IN + g * G;
                float d = 0.0f;
                for (int j = 0; j < G; ++j) d += c[j] * float(xp[j]);
                acc[b] += d;
            }
        }
""" + _ROW_LOOP_CLOSE

_TABLE_SRC = _ROW_LOOP_OPEN + """
        for (uint g = lane; g < NG; g += 32) {
            const device float* tg = table + (g * K + suc_unpack<BITS>(prow, g)) * BT;
            for (int b = 0; b < BT; ++b) acc[b] += tg[b];
        }
""" + _ROW_LOOP_CLOSE

_SOURCES = {"lookup": (_LOOKUP_SRC, ["x", "codebook", "packed", "scales"]),
            "mma": (_MMA_SRC, ["x", "codebook", "packed", "scales"]),
            "mma_reg": (_MMA_REG_SRC, ["x", "codebook", "packed", "scales"]),
            "lookup_k1": (_LOOKUP_K1_SRC, ["x", "codebook", "packed", "scales"]),
            "table": (_TABLE_SRC, ["table", "packed", "scales"])}
_KERNELS: dict[str, object] = {}


def _kernel(strategy: str):
    if strategy not in _KERNELS:
        src, inputs = _SOURCES[strategy]
        _KERNELS[strategy] = mx.fast.metal_kernel(name=f"suc_vq_matmul_{strategy}", input_names=inputs,
                                                  output_names=["y"], source=src, header=_HEADER)
    return _KERNELS[strategy]


def _check(x: mx.array, codebooks: mx.array, packed: mx.array, row_scales: mx.array, bits: int,
           in_features: int) -> tuple[int, int, int, int, int]:
    """Validate (B, in) x, (M, K, G) codebooks, (M, out, words) packed indices; returns (M, K, G, out, words)."""
    if x.ndim != 2 or x.shape[1] != in_features:
        raise ValueError(f"x must be (B, {in_features}), got {x.shape}")
    m, k, g = codebooks.shape
    if in_features % g:
        raise ValueError(f"in_features={in_features} is not a multiple of G={g}")
    if k > 1 << bits:
        raise ValueError(f"codebook has K={k} entries but indices have only {bits} bits")
    if packed.ndim != 3 or packed.shape[0] != m:
        raise ValueError(f"packed_indices must be ({m}, out, words), got {packed.shape}")
    _, out, words = packed.shape
    if words != words_per_row(in_features // g, bits):
        raise ValueError(f"packed_indices has {words} words per row, expected {words_per_row(in_features // g, bits)}")
    if row_scales.shape != (out,):
        raise ValueError(f"row_scales must be ({out},), got {row_scales.shape}")
    return m, k, g, out, words


TG_MEMORY_LIMIT = 32 * 1024     # bytes of threadgroup memory per threadgroup on Apple GPUs


def _mma_tile(x_dtype: mx.Dtype, m: int) -> mx.Dtype:
    if MMA_TILE == "auto":
        return mx.float16 if (x_dtype == mx.float16 and m == 1) else mx.float32
    return mx.float16 if MMA_TILE == "half" else mx.float32


def _launch(strategy: str, x: mx.array, cb: mx.array, packed: mx.array, s: mx.array, *, m: int, k: int, g: int,
            out: int, words: int, bits: int, in_features: int, codebook_in: str | None = None,
            out_dtype: mx.Dtype | None = None, path: str = "auto") -> mx.array:
    out_dtype = out_dtype or x.dtype
    codebook_in = resolve_codebook_in(codebook_in or CODEBOOK_IN, m=m)
    if path not in ("auto", "vec", "mma"):
        raise ValueError(f"path must be 'auto', 'vec' or 'mma', got {path!r}")
    ng = in_features // g
    ys = []
    for b0 in range(0, x.shape[0], MAX_BATCH_TILE):
        xb = x[b0:b0 + MAX_BATCH_TILE]
        bt = xb.shape[0]
        tmpl = [("T", x.dtype), ("BT", bt), ("OUT", out), ("IN", in_features), ("NG", ng), ("G", g), ("K", k),
                ("BITS", bits), ("WORDS", words)]
        kname = strategy
        if strategy == "lookup":
            use_mma = path == "mma" or (path == "auto" and bt >= mma_min_batch(m))
            inputs = [xb, cb, packed, s]
            cbtg = int(codebook_in == "threadgroup")
            if use_mma:
                kname, sgs, rows = ("mma" if MMA_KIND == "tg" else "mma_reg"), MMA_SG, 8 * MMA_RT
                tt, kcg, nb = _mma_tile(x.dtype, m), _mma_kcg(g, bits), -(-bt // 8)
                tsz = 2 if tt == mx.float16 else 4
                tg_bytes = cbtg * m * k * g * 2
                if MMA_KIND == "tg":   # shrink the K chunk (whole index words) until the tiles fit in 32 KB
                    tile = lambda c: tsz * c * g * 8 * (nb + MMA_SG * MMA_RT) + 4 * 64 * MMA_SG
                    while tg_bytes + tile(kcg) > TG_MEMORY_LIMIT and kcg % 2 == 0 and \
                            (kcg // 2) * g % 8 == 0 and (32 % bits or (kcg // 2) % (32 // bits) == 0):
                        kcg //= 2
                    tg_bytes += tile(kcg)
                if tg_bytes > TG_MEMORY_LIMIT:
                    raise ValueError(f"mma path needs {tg_bytes} B of threadgroup memory > 32 KB; lower MMA_KC / "
                                     f"MMA_RT / MMA_SG or use codebook_in='device'")
                tmpl = tmpl + [("SG", sgs), ("M", m), ("RT", MMA_RT), ("NB", nb), ("KCG", kcg), ("TT", tt),
                               ("CBTG", cbtg), ("TO", out_dtype)]
            else:
                sgs, rows = LOOKUP_SG, _lookup_rows(bt)
                hm = int(LOOKUP_HALF_MATH and x.dtype == mx.float16 and m == 1 and g % 4 == 0 and not cbtg)
                tmpl = tmpl + [("SG", sgs), ("M", m), ("R", rows), ("VEC", LOOKUP_VEC), ("CBTG", cbtg),
                               ("HM", hm), ("TO", out_dtype)]
        else:
            sgs, rows = SIMDGROUPS_PER_TG, ROWS_PER_SIMDGROUP
            tmpl = tmpl + [("SG", sgs), ("RPS", rows)]
            if strategy == "lookup_k1":
                inputs = [xb, cb, packed, s]
            else:
                # T[g, k, b] = <C[k], x[b, gG:(g+1)G]>, batch innermost so one gather reads BT contiguous floats
                xg = xb.astype(mx.float32).reshape(bt, ng, g).transpose(1, 2, 0)
                table = cb.astype(mx.float32).reshape(k, g) @ xg
                inputs = [table, packed, s]
        n_tg = -(-out // (sgs * rows))
        ys.append(_kernel(kname)(inputs=inputs, template=tmpl, grid=(n_tg * sgs * 32, 1, 1),
                                 threadgroup=(sgs * 32, 1, 1), output_shapes=[(bt, out)],
                                 output_dtypes=[out_dtype])[0])
    return ys[0] if len(ys) == 1 else mx.concatenate(ys, axis=0)


def additive_matmul_metal(x: mx.array, codebooks: mx.array, packed_indices: mx.array, row_scales: mx.array, *,
                          bits: int, in_features: int, codebook_in: str | None = None,
                          out_dtype: mx.Dtype | None = None, path: str = "auto") -> mx.array:
    """y = x Ŵᵀ, Ŵ = s ⊙ Σ_m C_m[i_m] (Kernel C, §19) from codebooks (M, K, G) and packed indices (M, out, words)
    → (B, out) in out_dtype (default x.dtype). All M codebooks together must satisfy M·K·G·2 ≤ 16 KB (D4),
    else ValueError.
    codebook_in overrides CODEBOOK_IN ("auto" | "device" | "threadgroup", see module docstring).
    path: "auto" (vec kernel for batch tiles < MMA_MIN_BATCH, simdgroup-matrix kernel from it), "vec", "mma"."""
    m, k, g, out, words = _check(x, codebooks, packed_indices, row_scales, bits, in_features)
    if m * k * g * 2 > CODEBOOK_TG_LIMIT:
        raise ValueError(f"{m} codebooks of K*G*2 = {k * g * 2} bytes ({m * k * g * 2} together) exceed the "
                         f"16 KB threadgroup ceiling (D4)")
    return _launch("lookup", x, codebooks.astype(mx.float16), packed_indices.astype(mx.uint32),
                   row_scales.astype(mx.float32), m=m, k=k, g=g, out=out, words=words, bits=bits,
                   in_features=in_features, codebook_in=codebook_in, out_dtype=out_dtype, path=path)


def vq_matmul_metal(x: mx.array, codebook: mx.array, packed_indices: mx.array, row_scales: mx.array, *,
                    bits: int, in_features: int,
                    strategy: Literal["lookup", "lookup_vec", "lookup_mma", "lookup_k1", "table"] = "lookup"
                    ) -> mx.array:
    """y = x Ŵᵀ from (codebook (K, G), bit-packed indices (out, words), row scales) → (B, out) in x.dtype.

    "lookup" dispatches by batch tile (vec kernel below MMA_MIN_BATCH, simdgroup-matrix kernel from it);
    "lookup_vec" / "lookup_mma" force one path.
    The codebook is cast to float16 (its storage dtype), scales to float32; accumulation is float32.
    """
    path = {"lookup_vec": "vec", "lookup_mma": "mma"}.get(strategy, "auto")
    strategy = "lookup" if strategy in ("lookup_vec", "lookup_mma") else strategy
    if strategy not in _SOURCES or strategy.startswith("mma"):
        raise ValueError(f"unknown strategy {strategy!r}")
    if codebook.ndim != 2 or packed_indices.ndim != 2:
        raise ValueError(f"codebook must be (K, G) and packed_indices (out, words), got {codebook.shape}, "
                         f"{packed_indices.shape}")
    m, k, g, out, words = _check(x, codebook[None], packed_indices[None], row_scales, bits, in_features)
    if strategy != "table" and k * g * 2 > CODEBOOK_TG_LIMIT:
        raise ValueError(f"codebook K*G*2 = {k * g * 2} bytes exceeds the 16 KB threadgroup ceiling (D4)")
    return _launch(strategy, x, codebook.astype(mx.float16), packed_indices.astype(mx.uint32),
                   row_scales.astype(mx.float32), m=1, k=k, g=g, out=out, words=words, bits=bits,
                   in_features=in_features, path=path)
