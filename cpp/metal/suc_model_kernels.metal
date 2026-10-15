// Model-graph kernels of the C++ runtime (milestone 3): the operations the Python runtime takes from MLX
// (mlx-lm Qwen3 on top of suc.runtime.compressed_model), written to follow MLX's operation order and rounding so
// that whole decode steps can be compared bit for bit with the Python/MLX runtime:
//
//   suc_embed_affine / suc_embed_dense   embedding row gather; affine codes are dequantized as the reference decoder
//                                        does (q·s, + b: two float32 roundings), then rounded to T, i.e. exactly the
//                                        decoded weight cast to the model dtype (= the Python reference model)
//   suc_matvec_affine / _dense          logits = x Wᵀ (W = the same per-element values), float32 accumulation,
//                                        rounded to T (MLX gemv rounds its float32 result once; the summation order
//                                        here is NOT MLX's, so logits agree to accumulation rounding only)
//   suc_rms_norm                         = mx.fast.rms_norm (rms_single_row order for N <= 4096, rms_looped above)
//   suc_rope                             = mx.fast.rope (rope_single: traditional = False, one position per call)
//   suc_sdpa_vector                      = MLX sdpa_vector (single pass), verbatim minus mask / sinks
//   suc_swiglu                           = mlx_lm.models.activations.swiglu (silu(g) · u, compiled, ops in T)
//   suc_add                              = residual add in T
//   suc_argmax                           = mx.argmax (first index of the maximum)
//   suc_copy_cols                        strided row copy (outputs of fused projection groups)
//
// Instantiated at run time by the C++ host (cpp/src/metal/model.cpp) with explicit `template [[host_name]]`
// lines, like cpp/metal/suc_kernels.metal (which this file is appended to at build time).
//
// suc_sdpa_vector is adapted from mlx/backend/metal/kernels/sdpa_vector.h (MLX, Copyright © 2024 Apple Inc.,
// MIT License): the function-constant branches for masks, sinks and transposed queries are removed (decode with
// one query row, no mask, as mlx-lm calls it); the arithmetic is unchanged.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// element j of a little-endian packed row (rows_u32le_bitstream)
template <int BITS>
inline uint suc_m_unpack(const device uint32_t* row, uint j) {
    uint p = j * BITS, w = p >> 5, off = p & 31;
    uint v = row[w] >> off;
    if (off + BITS > 32) v |= row[w + 1] << (32 - off);
    return v & ((1u << BITS) - 1u);
}

// affine weight element as the reference decoder computes it (format.md §10: q·s + b, two roundings), in float32
template <int BITS, int GS, typename TS>
inline float suc_m_affine(const device uint32_t* row, const device TS* s, const device TS* b, uint j) {
    float q = float(suc_m_unpack<BITS>(row, j));
    float t = q * float(s[j / GS]);
    t = t + float(b[j / GS]);
    return t;
}

// ---------------------------------------------------------------------------------------------- embedding
// y[r, :] = T(W[tok[r], :]) for r < grid.y. Grid (IN, rows).
template <typename T, int BITS, int GS, int IN, int WORDS, typename TS>
[[kernel]] void suc_embed_affine(
    const device uint32_t* tok [[buffer(0)]],
    const device uint32_t* packed [[buffer(1)]],
    const device TS* scales [[buffer(2)]],
    const device TS* biases [[buffer(3)]],
    device T* y [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]) {
    const uint r = tok[pos.y];
    const uint j = pos.x;
    if (j >= IN) return;
    const device uint32_t* row = packed + size_t(r) * WORDS;
    y[size_t(pos.y) * IN + j] = static_cast<T>(suc_m_affine<BITS, GS, TS>(row, scales + size_t(r) * (IN / GS),
                                                                          biases + size_t(r) * (IN / GS), j));
}

template <typename T, int IN, typename TW>
[[kernel]] void suc_embed_dense(
    const device uint32_t* tok [[buffer(0)]],
    const device TW* w [[buffer(1)]],
    device T* y [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]) {
    const uint r = tok[pos.y];
    if (pos.x >= IN) return;
    y[size_t(pos.y) * IN + pos.x] = static_cast<T>(float(w[size_t(r) * IN + pos.x]));
}

// ---------------------------------------------------------------------------------------------- matvec (LM head)
// y[o] = T(Σ_j float(x[j]) · float(T(W[o, j]))), one simdgroup per RPS rows, 2 simdgroups per threadgroup.
// Lane l handles elements 8c .. 8c+7 for c ≡ l (mod 32); per-lane partials in j order, then simd_sum.
template <typename T, int BITS, int GS, int IN, int OUT, int WORDS, typename TS, int RPS>
[[kernel]] void suc_matvec_affine(
    const device T* x [[buffer(0)]],
    const device uint32_t* packed [[buffer(1)]],
    const device TS* scales [[buffer(2)]],
    const device TS* biases [[buffer(3)]],
    device T* y [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint o0 = (tg * 2 + sg) * RPS;
    float acc[RPS];
    for (int r = 0; r < RPS; ++r) acc[r] = 0.0f;
    for (uint c = lane; c < IN / 8; c += 32) {
        float xv[8];
        for (int e = 0; e < 8; ++e) xv[e] = float(x[c * 8 + e]);
        for (int r = 0; r < RPS; ++r) {
            const uint o = o0 + r;
            if (o >= OUT) break;
            const device uint32_t* row = packed + size_t(o) * WORDS;
            const device TS* s = scales + size_t(o) * (IN / GS);
            const device TS* b = biases + size_t(o) * (IN / GS);
            for (int e = 0; e < 8; ++e) {
                float w = float(static_cast<T>(suc_m_affine<BITS, GS, TS>(row, s, b, c * 8 + e)));
                acc[r] += xv[e] * w;
            }
        }
    }
    for (int r = 0; r < RPS; ++r) {
        float v = simd_sum(acc[r]);
        if (lane == 0 && o0 + r < OUT) y[o0 + r] = static_cast<T>(v);
    }
}

template <typename T, int IN, int OUT, typename TW, int RPS>
[[kernel]] void suc_matvec_dense(
    const device T* x [[buffer(0)]],
    const device TW* w [[buffer(1)]],
    device T* y [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint o0 = (tg * 2 + sg) * RPS;
    float acc[RPS];
    for (int r = 0; r < RPS; ++r) acc[r] = 0.0f;
    for (uint c = lane; c < IN / 8; c += 32) {
        float xv[8];
        for (int e = 0; e < 8; ++e) xv[e] = float(x[c * 8 + e]);
        for (int r = 0; r < RPS; ++r) {
            const uint o = o0 + r;
            if (o >= OUT) break;
            for (int e = 0; e < 8; ++e) acc[r] += xv[e] * float(static_cast<T>(float(w[size_t(o) * IN + c * 8 + e])));
        }
    }
    for (int r = 0; r < RPS; ++r) {
        float v = simd_sum(acc[r]);
        if (lane == 0 && o0 + r < OUT) y[o0 + r] = static_cast<T>(v);
    }
}

// = MLX gemv (y = x Wᵀ, the matrix rows contiguous) of the DECODED weight T(q·s + b): the accumulation order of
// suc_moe_gemv (MLX GEMVKernel, tiling (SN, TN, BN) chosen as MLX's host code does), with the affine element decoded
// on the fly as the reference decoder does. Bit-identical to the Python runtime's LM head / nn.Linear of the decoded
// weight for one row (milestone 4; the earlier suc_matvec_* summed in another order).
template <typename T, int BITS, int GS, int IN, int OUT, int WORDS, typename TS, int SN, int TN, int BN>
[[kernel]] void suc_gemv_affine(
    const device T* x [[buffer(0)]],
    const device uint32_t* packed [[buffer(1)]],
    const device TS* scales [[buffer(2)]],
    const device TS* biases [[buffer(3)]],
    device T* y [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint tpg_sg [[simdgroups_per_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int SM = 32 / SN;
    constexpr int BLK = BN * SN * TN;
    threadgroup float tg[BN * SM];
    const int thrM = simd_lid / SN, thrN = simd_lid % SN;
    // BN = 1: a threadgroup may hold several simdgroups, each on its own SM rows (the host's choice; one row's
    // arithmetic does not depend on it). BN > 1: the BN simdgroups split one row block's columns (MLX).
    const int sgN = BN > 1 ? int(simd_gid) : 0;
    const int row = (BN > 1 ? int(tid.x) : int(tid.x * tpg_sg + simd_gid)) * SM + thrM;
    const bool live = row < OUT;
    const device T* xr = x + ulong(tid.y) * IN;
    const ulong o = ulong(live ? row : 0);
    const device uint32_t* wr = packed + o * WORDS;
    const device TS* sr = scales + o * (IN / GS);
    const device TS* br = biases + o * (IN / GS);
    float acc = 0.0f;
    if constexpr (TN == 4 && IN % 4 == 0 && GS % 4 == 0 && 32 % BITS == 0) {
        // Word-at-a-time decode: the thread's 4 consecutive elements j..j+3 (j % 4 == 0) lie in one uint32 and one
        // scale group, so one word, one scale and one bias load serve all four. Per element the arithmetic and the
        // accumulation order are those of the generic path below (q·s, + b, rounded to T, acc += m · x in j order),
        // so the result is bit-identical; only the loads change. U steps are loaded before they are accumulated.
        constexpr int EPW = 32 / BITS, NIT = (IN + BLK - 1) / BLK, U = 4;
        constexpr uint MASK = (1u << BITS) - 1u;
        const int jb = (sgN * SN + thrN) * TN;
        for (int it = 0; it < NIT; it += U) {
            uint wv[U];
            float4 xv[U];
            float sv[U], bv[U];
            for (int u = 0; u < U; ++u) {
                const int j = (it + u) * BLK + jb;
                if (it + u < NIT && j < IN) {
                    wv[u] = wr[j / EPW] >> uint((j % EPW) * BITS);
                    xv[u] = float4(*(const device vec<T, 4>*)(xr + j));
                    sv[u] = float(sr[j / GS]);
                    bv[u] = float(br[j / GS]);
                }
            }
            for (int u = 0; u < U; ++u) {
                const int j = (it + u) * BLK + jb;
                if (it + u < NIT && j < IN) {
                    for (int tn = 0; tn < 4; ++tn) {
                        float t = float((wv[u] >> uint(tn * BITS)) & MASK) * sv[u];
                        t = t + bv[u];
                        float m = float(static_cast<T>(t));
                        acc += m * xv[u][tn];
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < IN; i += BLK) {
            const int b = i + (sgN * SN + thrN) * TN;
            for (int tn = 0; tn < TN; ++tn) {
                const int j = b + tn;
                if (j < IN) {
                    float v = float(xr[j]);
                    float m = float(static_cast<T>(suc_m_affine<BITS, GS, TS>(wr, sr, br, uint(j))));
                    acc += m * v;
                }
            }
        }
    }
    for (ushort sn = SN / 2; sn >= 1; sn >>= 1) acc += simd_shuffle_down(acc, sn);
    if (BN > 1) {
        if (thrN == 0) tg[sgN * SM + thrM] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgN == 0 && thrN == 0)
            for (int s = 1; s < BN; ++s) acc += tg[s * SM + thrM];
    }
    if (sgN == 0 && thrN == 0 && live) y[ulong(tid.y) * OUT + row] = static_cast<T>(acc);
}

// suc_gemv_affine over NB input rows at once (batched prefill): every weight element is decoded once and applied to
// the NB rows; row b's arithmetic and accumulation order are exactly those of suc_gemv_affine on that row alone (the
// same per-element decode, the same j order, the same shuffle tree and simdgroup sum), so each output row is
// bit-identical to a one-row launch. Rows r = tid.y · NB + b < nrows; x row r at x + r · xs, y row r at y + r · ys
// (strides in elements). Grid of threadgroups (blocks, ceil(nrows / NB)).
template <typename T, int BITS, int GS, int IN, int OUT, int WORDS, typename TS, int SN, int TN, int BN, int NB>
[[kernel]] void suc_gemv_affine_rows(
    const device T* x [[buffer(0)]],
    const device uint32_t* packed [[buffer(1)]],
    const device TS* scales [[buffer(2)]],
    const device TS* biases [[buffer(3)]],
    device T* y [[buffer(4)]],
    constant uint& nrows [[buffer(5)]],
    constant uint& xs [[buffer(6)]],
    constant uint& ys [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint tpg_sg [[simdgroups_per_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int SM = 32 / SN;
    constexpr int BLK = BN * SN * TN;
    threadgroup float tg[BN * SM * NB];
    const int thrM = simd_lid / SN, thrN = simd_lid % SN;
    const int sgN = BN > 1 ? int(simd_gid) : 0;
    const int row = (BN > 1 ? int(tid.x) : int(tid.x * tpg_sg + simd_gid)) * SM + thrM;
    const bool live = row < OUT;
    const uint r0 = tid.y * NB;
    const int nb = int(min(uint(NB), nrows - r0));
    const ulong o = ulong(live ? row : 0);
    const device uint32_t* wr = packed + o * WORDS;
    const device TS* sr = scales + o * (IN / GS);
    const device TS* br = biases + o * (IN / GS);
    float acc[NB];
    for (int b = 0; b < NB; ++b) acc[b] = 0.0f;
    if constexpr (TN == 4 && IN % 4 == 0 && GS % 4 == 0 && 32 % BITS == 0) {
        constexpr int EPW = 32 / BITS, NIT = (IN + BLK - 1) / BLK, U = 4;
        constexpr uint MASK = (1u << BITS) - 1u;
        const int jb = (sgN * SN + thrN) * TN;
        for (int it = 0; it < NIT; it += U) {
            uint wv[U];
            float sv[U], bv[U];
            for (int u = 0; u < U; ++u) {
                const int j = (it + u) * BLK + jb;
                if (it + u < NIT && j < IN) {
                    wv[u] = wr[j / EPW] >> uint((j % EPW) * BITS);
                    sv[u] = float(sr[j / GS]);
                    bv[u] = float(br[j / GS]);
                }
            }
            for (int u = 0; u < U; ++u) {
                const int j = (it + u) * BLK + jb;
                if (it + u < NIT && j < IN) {
                    float m[4];
                    for (int tn = 0; tn < 4; ++tn) {
                        float t = float((wv[u] >> uint(tn * BITS)) & MASK) * sv[u];
                        t = t + bv[u];
                        m[tn] = float(static_cast<T>(t));
                    }
                    for (int b = 0; b < NB; ++b) {
                        if (b >= nb) break;
                        const float4 xv = float4(*(const device vec<T, 4>*)(x + ulong(r0 + b) * xs + j));
                        for (int tn = 0; tn < 4; ++tn) acc[b] += m[tn] * xv[tn];
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < IN; i += BLK) {
            const int bb = i + (sgN * SN + thrN) * TN;
            for (int tn = 0; tn < TN; ++tn) {
                const int j = bb + tn;
                if (j < IN) {
                    float m = float(static_cast<T>(suc_m_affine<BITS, GS, TS>(wr, sr, br, uint(j))));
                    for (int b = 0; b < NB; ++b) {
                        if (b >= nb) break;
                        float v = float(x[ulong(r0 + b) * xs + j]);
                        acc[b] += m * v;
                    }
                }
            }
        }
    }
    for (int b = 0; b < NB; ++b)
        for (ushort sn = SN / 2; sn >= 1; sn >>= 1) acc[b] += simd_shuffle_down(acc[b], sn);
    if (BN > 1) {
        if (thrN == 0)
            for (int b = 0; b < NB; ++b) tg[(b * BN + sgN) * SM + thrM] = acc[b];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgN == 0 && thrN == 0)
            for (int b = 0; b < NB; ++b)
                for (int s = 1; s < BN; ++s) acc[b] += tg[(b * BN + s) * SM + thrM];
    }
    if (sgN == 0 && thrN == 0 && live)
        for (int b = 0; b < NB; ++b)
            if (b < nb) y[ulong(r0 + b) * ys + row] = static_cast<T>(acc[b]);
}

// Row gather / scatter of 32-bit words (batched MoE tiles): gather dst row j = src row map[j]; scatter dst row map[j] =
// src row j, skipping map[j] == ~0u (padding rows of a tile). Rows of `words` uint32 words. Grid (words, rows).
template <typename U>
[[kernel]] void suc_gather_words(
    const device U* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    const device uint32_t* map [[buffer(2)]],
    constant uint& words [[buffer(3)]],
    uint2 i [[thread_position_in_grid]]) {
    if (i.x >= words) return;
    dst[ulong(i.y) * words + i.x] = src[ulong(map[i.y]) * words + i.x];
}
template <typename U>
[[kernel]] void suc_scatter_words(
    const device U* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    const device uint32_t* map [[buffer(2)]],
    constant uint& words [[buffer(3)]],
    uint2 i [[thread_position_in_grid]]) {
    if (i.x >= words) return;
    const uint r = map[i.y];
    if (r == ~0u) return;
    dst[ulong(r) * words + i.x] = src[ulong(i.y) * words + i.x];
}


// ---------------------------------------------------------------------------------------------- RMSNorm
// = mx.fast.rms_norm(x, w, eps) on rows of N. One threadgroup per row.
// N <= 4096 (MLX rms_single_row): TPG = 32 · ceil(ceil(N / 4) / 32) threads, 4 contiguous elements per thread.
// N > 4096 (MLX rms_looped): TPG = 1024 threads, 4 contiguous elements per thread per step of 4 · TPG.
// out = T(w · T(x · rsqrt(mean(x²) + eps))) (w · … in T).
template <typename T, int N, int TPG>
[[kernel]] void suc_rms_norm(
    const device T* x [[buffer(0)]],
    const device T* w [[buffer(1)]],
    device T* y [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int R = 4;
    threadgroup float local_inv_mean[32];
    x += size_t(row) * N;
    y += size_t(row) * N;
    float acc = 0;
    if (N <= 4096) {
        if (lid * R + R <= N) {
            for (int i = 0; i < R; i++) {
                float xi = x[lid * R + i];
                acc += xi * xi;
            }
        } else {
            for (int i = 0; i < R; i++) {
                if ((lid * R + i) < N) {
                    float xi = x[lid * R + i];
                    acc += xi * xi;
                }
            }
        }
    } else {
        for (uint r = 0; r < uint(N); r += TPG * R) {
            if (r + lid * R + R <= N) {
                for (int i = 0; i < R; i++) {
                    float xi = x[r + lid * R + i];
                    acc += xi * xi;
                }
            } else {
                for (int i = 0; i < R; i++) {
                    if ((r + lid * R + i) < N) {
                        float xi = x[r + lid * R + i];
                        acc += xi * xi;
                    }
                }
            }
        }
    }
    acc = simd_sum(acc);
    if (sg == 0) local_inv_mean[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_inv_mean[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_inv_mean[lane]);
        if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / N + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = local_inv_mean[0];
    for (uint r = 0; r < uint(N); r += TPG * R) {
        for (int i = 0; i < R; i++) {
            uint j = r + lid * R + i;
            if (j < uint(N)) y[j] = w[j] * static_cast<T>(x[j] * inv);
        }
        if (N <= 4096) break;
    }
}

// ---------------------------------------------------------------------------------------------- RoPE
// = mx.fast.rope(x, D, traditional=False, base, scale=1, offset): rows of D (heads), in place allowed. Grid
// (D/2, rows); row r is at position offset + r / rows_per_pos (several tokens of a prefill chunk, rows_per_pos
// heads each). base_log2 = log2(base) (MLX passes std::log2(base)). Output row r goes to y + r * D (the caller
// offsets y, e.g. into the KV cache).
template <typename T, int D>
[[kernel]] void suc_rope(
    const device T* x [[buffer(0)]],
    device T* y [[buffer(1)]],
    constant int& offset [[buffer(2)]],
    constant float& base_log2 [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& rows_per_pos [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]) {
    constexpr uint HALF = D / 2;
    float d = static_cast<float>(pos.x) / static_cast<float>(HALF);
    float inv_freq = metal::exp2(-d * base_log2);
    float L = scale * static_cast<float>(offset + int(pos.y / rows_per_pos));
    float theta = L * inv_freq;
    float costheta = metal::fast::cos(theta);
    float sintheta = metal::fast::sin(theta);
    uint i1 = pos.x + pos.y * D;
    uint i2 = i1 + HALF;
    float x1 = static_cast<float>(x[i1]);
    float x2 = static_cast<float>(x[i2]);
    float rx1 = x1 * costheta - x2 * sintheta;
    float rx2 = x1 * sintheta + x2 * costheta;
    y[i1] = static_cast<T>(rx1);
    y[i2] = static_cast<T>(rx2);
}

// ---------------------------------------------------------------------------------------------- attention
// MLX sdpa_vector (single pass), one threadgroup of 32 simdgroups per (query head, query row): grid (Hq, n, 1)
// threadgroups of 1024 threads. Queries / output rows (n, Hq, D) (MLX's query_transposed layout); keys / values at
// kv_head * head_stride + i * seq_stride. Causal: query row t attends to keys 0 .. N - n + t (MLX do_causal; the
// skipped keys never touch the accumulators, so each row equals a single-row call with N - n + t + 1 keys).
template <typename T, int D, int V>
[[kernel]] void suc_sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread = V / BD;
    int inner_k_stride = BN * int(k_seq_stride);
    int inner_v_stride = BN * int(v_seq_stride);

    typedef float U;

    thread U q[qk_per_thread];
    thread U k[qk_per_thread];
    thread U o[v_per_thread];

    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];

    const int q_batch_head_idx = tid.x;
    const int q_seq_idx = tid.y;
    const int kv_head_idx = q_batch_head_idx / gqa_factor;
    const int o_offset = tpg.x * q_seq_idx + q_batch_head_idx;
    const int q_offset = o_offset;
    const int n_keys = N - int(tpg.y) + q_seq_idx + 1;
    queries += q_offset * D + simd_lid * qk_per_thread;
    keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * qk_per_thread;
    values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride + simd_lid * v_per_thread;

    out += o_offset * V + simd_gid * v_per_thread;

    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = static_cast<U>(scale) * queries[i];
    }
    for (int i = 0; i < v_per_thread; i++) {
        o[i] = 0;
    }

    U max_score = -FLT_MAX;   // Limits<float>::finite_min
    U sum_exp_score = 0;

    for (int i = simd_gid; i < n_keys; i += BN) {
        for (int j = 0; j < qk_per_thread; j++) {
            k[j] = keys[j];
        }
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * k[j];
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * values[j];
        }

        keys += inner_k_stride;
        values += inner_v_stride;
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (int i = 0; i < v_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; i++) {
            out[i] = static_cast<T>(o[i]);
        }
    }
}

// ---------------------------------------------------------------------------------------------- element-wise
// MLX Sigmoid (unary_ops.h) with T arithmetic: y = 1 / (1 + exp(|x|)), x < 0 ? y : 1 - y. MLX's bfloat math
// overloads (bf16_math.h) evaluate exp in float32 and round to bfloat; half and float use the native functions.
inline bfloat suc_m_exp(bfloat x) { return static_cast<bfloat>(metal::precise::exp(static_cast<float>(x))); }
inline half suc_m_exp(half x) { return metal::precise::exp(x); }
inline float suc_m_exp(float x) { return metal::precise::exp(x); }
inline bfloat suc_m_abs(bfloat x) { return static_cast<bfloat>(metal::abs(static_cast<float>(x))); }
inline half suc_m_abs(half x) { return metal::abs(x); }
inline float suc_m_abs(float x) { return metal::abs(x); }

// = mlx_lm swiglu(g, u) = silu(g) · u (bit-identical for bfloat16 and float32 on the M3; float16 is not).
template <typename T>
[[kernel]] void suc_swiglu(
    const device T* g [[buffer(0)]],
    const device T* u [[buffer(1)]],
    device T* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    T x = g[i];
    T e = suc_m_exp(suc_m_abs(x));
    T s1 = static_cast<T>(1) / (static_cast<T>(1) + e);
    T s = (x < static_cast<T>(0)) ? s1 : static_cast<T>(1) - s1;
    T t = x * s;
    y[i] = t * u[i];
}

template <typename T>
[[kernel]] void suc_add(
    const device T* a [[buffer(0)]],
    const device T* b [[buffer(1)]],
    device T* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------------------------- copy
// y[r * ys + c] = x[r * xs + c] for r < rows, c < cols (exact copy): distributes the (B, Σ out) output of a fused
// projection group to its members' buffers (= the slices y[:, o - n:o] the Python runtime takes; milestone 4).
template <typename T>
[[kernel]] void suc_copy_cols(
    const device T* x [[buffer(0)]],
    device T* y [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& xs [[buffer(3)]],
    constant uint& ys [[buffer(4)]],
    uint2 i [[thread_position_in_grid]]) {
    if (i.x >= cols) return;
    y[ulong(i.y) * ys + i.x] = x[ulong(i.y) * xs + i.x];
}

// ---------------------------------------------------------------------------------------------- argmax
// out[0] = first index of the maximum of x[0:n] (= mx.argmax; NaN is not handled). One threadgroup of TPG threads.
template <typename T, int TPG>
[[kernel]] void suc_argmax(
    const device T* x [[buffer(0)]],
    device uint32_t* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup uint bi[32];
    float best = -INFINITY;
    uint idx = 0xFFFFFFFFu;
    for (uint i = lid; i < n; i += TPG) {
        float v = float(x[i]);
        if (v > best || idx == 0xFFFFFFFFu) {
            best = v;
            idx = i;
        }
    }
    for (ushort d = 16; d >= 1; d /= 2) {
        float ov = simd_shuffle_down(best, d);
        uint oi = simd_shuffle_down(idx, d);
        if (lane + d < 32 && (ov > best || (ov == best && oi < idx))) {
            best = ov;
            idx = oi;
        }
    }
    if (lane == 0) {
        bv[sg] = best;
        bi[sg] = idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        for (uint s = 1; s < TPG / 32; ++s)
            if (bv[s] > best || (bv[s] == best && bi[s] < idx)) {
                best = bv[s];
                idx = bi[s];
            }
        out[0] = idx;
    }
}

// ================================================================================================ MoE (milestone 4)
// Router, routing and combine of mlx-lm Qwen3MoeSparseMoeBlock (cpp/src/metal/switch.cpp, MoeBlock), in MLX's
// operation order and rounding:
//   suc_moe_gemv     gates = x Wᵀ (nn.Linear, MLX gemv): float32 products / sums in the GEMVKernel order for the
//                    (SN, TN, BN) tiling given as template constants, one rounding to T
//   suc_moe_route    softmax(gates, precise=True) (MLX softmax_single_row, N_READS = 4), top-k with
//                    argpartition(-k)[..., -k:] semantics (= the last k of a stable ascending argsort: ascending gate
//                    value, ties by expert index), take_along_axis, optional norm_topk_prob (sum: sequential in T from
//                    0, as MLX's row reduce; then a T division)
//   suc_moe_combine  (y * scores[..., None]).sum(axis=-2): products rounded to T, sequential sum in T from 0 (MLX's
//                    column reduce over k)
//   suc_cast         y = TO(x) (astype, round to nearest even)

// y[r, o] = T(Σ_j W[o, j] · x[r, j]). Threadgroup = BN simdgroups of 32 lanes; a simdgroup holds SM = 32 / SN rows of
// SN lanes; lane (thrN, sgN) sums the TN-element chunks j = i·BLK + (sgN·SN + thrN)·TN + tn (BLK = BN·SN·TN) in
// order, then simd_shuffle_down tree over the SN lanes, then simdgroup 0 adds the other simdgroups' sums in order.
// Grid: (ceil(OUT / SM) threadgroups, rows).
template <typename T, typename TW, int IN, int OUT, int SN, int TN, int BN>
[[kernel]] void suc_moe_gemv(
    const device T* x [[buffer(0)]],
    const device TW* w [[buffer(1)]],
    device T* y [[buffer(2)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int SM = 32 / SN;
    constexpr int BLK = BN * SN * TN;
    threadgroup float tg[BN * SM];
    const int thrM = simd_lid / SN, thrN = simd_lid % SN;
    const int sgN = simd_gid;
    const int row = int(tid.x) * SM + thrM;
    const bool live = row < OUT;
    const device T* xr = x + ulong(tid.y) * IN;
    const device TW* wr = w + ulong(live ? row : 0) * IN;
    float acc = 0.0f;
    for (int i = 0; i < IN; i += BLK) {
        const int b = i + (sgN * SN + thrN) * TN;
        for (int tn = 0; tn < TN; ++tn) {
            const int j = b + tn;
            if (j < IN) {
                float v = float(xr[j]);
                float m = float(T(wr[j]));
                acc += m * v;
            }
        }
    }
    for (ushort sn = SN / 2; sn >= 1; sn >>= 1) acc += simd_shuffle_down(acc, sn);
    if (BN > 1) {
        if (thrN == 0) tg[sgN * SM + thrM] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgN == 0 && thrN == 0)
            for (int s = 1; s < BN; ++s) acc += tg[s * SM + thrM];
    }
    if (sgN == 0 && thrN == 0 && live) y[ulong(tid.y) * OUT + row] = static_cast<T>(acc);
}

// suc_moe_gemv over NB input rows at once (batched prefill): every weight element is loaded once and applied to the NB
// rows; row b's products, j order, shuffle tree and simdgroup sum are those of suc_moe_gemv on that row alone, so each
// output row is bit-identical to a one-row launch. Rows r = tid.y · NB + b < nrows; x row r at x + r · xs, y row r at
// y + r · ys (strides in elements). Grid of threadgroups (ceil(OUT / SM), ceil(nrows / NB)).
template <typename T, typename TW, int IN, int OUT, int SN, int TN, int BN, int NB>
[[kernel]] void suc_moe_gemv_rows(
    const device T* x [[buffer(0)]],
    const device TW* w [[buffer(1)]],
    device T* y [[buffer(2)]],
    constant uint& nrows [[buffer(3)]],
    constant uint& xs [[buffer(4)]],
    constant uint& ys [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int SM = 32 / SN;
    constexpr int BLK = BN * SN * TN;
    threadgroup float tg[BN * SM * NB];
    const int thrM = simd_lid / SN, thrN = simd_lid % SN;
    const int sgN = simd_gid;
    const int row = int(tid.x) * SM + thrM;
    const bool live = row < OUT;
    const uint r0 = tid.y * NB;
    const int nb = int(min(uint(NB), nrows - r0));
    const device TW* wr = w + ulong(live ? row : 0) * IN;
    float acc[NB];
    for (int b = 0; b < NB; ++b) acc[b] = 0.0f;
    for (int i = 0; i < IN; i += BLK) {
        const int bb = i + (sgN * SN + thrN) * TN;
        for (int tn = 0; tn < TN; ++tn) {
            const int j = bb + tn;
            if (j < IN) {
                float m = float(T(wr[j]));
                for (int b = 0; b < NB; ++b) {
                    if (b >= nb) break;
                    float v = float(x[ulong(r0 + b) * xs + j]);
                    acc[b] += m * v;
                }
            }
        }
    }
    for (int b = 0; b < NB; ++b)
        for (ushort sn = SN / 2; sn >= 1; sn >>= 1) acc[b] += simd_shuffle_down(acc[b], sn);
    if (BN > 1) {
        if (thrN == 0)
            for (int b = 0; b < NB; ++b) tg[(b * BN + sgN) * SM + thrM] = acc[b];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgN == 0 && thrN == 0)
            for (int b = 0; b < NB; ++b)
                for (int s = 1; s < BN; ++s) acc[b] += tg[(b * BN + s) * SM + thrM];
    }
    if (sgN == 0 && thrN == 0 && live)
        for (int b = 0; b < NB; ++b)
            if (b < nb) y[ulong(r0 + b) * ys + row] = static_cast<T>(acc[b]);
}

// One threadgroup of TPG = 32 · ceil(ceil(E / 4) / 32) threads per row (MLX's softmax threadgroup size).
// gates (rows, E) T -> eidx (rows, K) uint32, scores (rows, K) T.
template <typename T, int E, int K, int NORM, int TPG>
[[kernel]] void suc_moe_route(
    const device T* gates [[buffer(0)]],
    device uint32_t* eidx [[buffer(1)]],
    device T* scores [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
    // ---- softmax: MLX softmax_single_row<T, float, 4>, verbatim arithmetic
    constexpr int N_READS = 4;
    constexpr int SIMD_SIZE = 32;
    const int lid = _lid;
    const int axis_size = E;
    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_normalizer[SIMD_SIZE];
    threadgroup T prob[E];
    threadgroup uint sel[K];
    threadgroup T sc[K];
    float ld[N_READS];
    const device T* in = gates + gid * size_t(axis_size) + lid * N_READS;
    if (lid * N_READS + N_READS <= axis_size) {
        for (int i = 0; i < N_READS; i++) ld[i] = float(in[i]);
    } else {
        for (int i = 0; i < N_READS; i++)
            ld[i] = ((lid * N_READS + i) < axis_size) ? float(in[i]) : -metal::numeric_limits<float>::infinity();
    }
    if (simd_group_id == 0) {
        local_max[simd_lane_id] = -metal::numeric_limits<float>::infinity();
        local_normalizer[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maxval = -metal::numeric_limits<float>::max();
    for (int i = 0; i < N_READS; i++) maxval = (maxval < ld[i]) ? ld[i] : maxval;
    maxval = simd_max(maxval);
    if (simd_lane_id == 0) local_max[simd_group_id] = maxval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        maxval = simd_max(local_max[simd_lane_id]);
        if (simd_lane_id == 0) local_max[0] = maxval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = local_max[0];
    float normalizer = 0;
    for (int i = 0; i < N_READS; i++) {
        float exp_x = fast::exp(ld[i] - maxval);
        ld[i] = exp_x;
        normalizer += exp_x;
    }
    normalizer = simd_sum(normalizer);
    if (simd_lane_id == 0) local_normalizer[simd_group_id] = normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        normalizer = simd_sum(local_normalizer[simd_lane_id]);
        if (simd_lane_id == 0) local_normalizer[0] = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = 1 / local_normalizer[0];
    for (int i = 0; i < N_READS; i++)
        if (lid * N_READS + i < axis_size) prob[lid * N_READS + i] = T(ld[i] * normalizer);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- top-k: rank of expert i in a stable ascending sort; the last K ranks are selected, in rank order
    for (int i = lid; i < E; i += TPG) {
        const float v = float(prob[i]);
        int rank = 0;
        for (int j = 0; j < E; ++j) {
            const float u = float(prob[j]);
            rank += (u < v || (u == v && j < i)) ? 1 : 0;
        }
        if (rank >= E - K) {
            sel[rank - (E - K)] = uint(i);
            sc[rank - (E - K)] = prob[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        T s = T(0);
        if (NORM)
            for (int j = 0; j < K; ++j) s = T(float(s) + float(sc[j]));
        for (int j = 0; j < K; ++j) {
            eidx[gid * K + j] = sel[j];
            scores[gid * K + j] = NORM ? T(float(sc[j]) / float(s)) : sc[j];
        }
    }
}

// out[r, h] = Σ_j T(y[r, j, h] · scores[r, j]) (sequential in T from 0). Grid (H, rows).
template <typename T, int K, int H>
[[kernel]] void suc_moe_combine(
    const device T* y [[buffer(0)]],
    const device T* scores [[buffer(1)]],
    device T* out [[buffer(2)]],
    uint2 i [[thread_position_in_grid]]) {
    if (i.x >= H) return;
    T acc = T(0);
    for (int j = 0; j < K; ++j) {
        T p = T(float(y[(ulong(i.y) * K + j) * H + i.x]) * float(scores[ulong(i.y) * K + j]));
        acc = T(float(acc) + float(p));
    }
    out[ulong(i.y) * H + i.x] = acc;
}

template <typename TI, typename TO>
[[kernel]] void suc_cast(
    const device TI* x [[buffer(0)]],
    device TO* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = static_cast<TO>(x[i]);
}

// y[r, c] = x[r / REP, c] (the token rows broadcast to the k expert rows of SwitchGLU). Grid (cols, rows of y).
template <typename T>
[[kernel]] void suc_repeat_rows(
    const device T* x [[buffer(0)]],
    device T* y [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& rep [[buffer(3)]],
    uint2 i [[thread_position_in_grid]]) {
    if (i.x >= cols) return;
    y[ulong(i.y) * cols + i.x] = x[ulong(i.y / rep) * cols + i.x];
}

// ---------------------------------------------------------------------------------------------- two-pass attention
// MLX's sdpa_vector_2pass_1 / _1_gqa / _2 (mlx/backend/metal/kernels/sdpa_vector.h, MLX, Copyright © 2024 Apple Inc.,
// MIT License) for one query row per head (decode): masks, sinks and transposed queries removed, `blocks` a template
// constant (MLX: function constant 26 / the grid depth), arithmetic unchanged. Pass 1 writes per-block partial outputs
// (in T), sums and maxima; pass 2 combines them. Milestone 4 (deviation 2 of milestone 3).
template <typename T, int D, int V, int BLOCKS>
[[kernel]] void suc_sdpa_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int blocks = BLOCKS;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int q_offset = o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride + simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride + simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;
  for (int i = 0; i < qk_per_thread; i++) q[i] = static_cast<U>(scale) * queries[i];
  U max_score = -FLT_MAX;   // Limits<float>::finite_min
  U sum_exp_score = 0;
  for (int i = block_idx; i < N; i += blocks) {
    U score = 0;
    for (int i = 0; i < qk_per_thread; i++) score += q[i] * keys[i];
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int i = 0; i < v_per_thread; i++) o[i] = o[i] * factor + exp_score * values[i];
    keys += blocks * int(k_seq_stride);
    values += blocks * int(v_seq_stride);
  }
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (int i = 0; i < v_per_thread; i++) out[i] = static_cast<T>(o[i]);
}

template <typename T, int D, int V, int G, int HPT>
[[kernel]] void suc_sdpa_2pass_1_gqa(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr int NT = G / HPT;
  typedef float U;
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int blocks = tpg.z;
  const int g = tidtg.y;
  const int cchunk = g / NT;
  const int h0 = (g % NT) * HPT;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * G;
  const int base_head = batch_idx * num_q_heads + kv_head_idx * G;
  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  const int chunk = (N + blocks - 1) / blocks;
  const int kstart = block_idx * chunk;
  const int kend = min(N, kstart + chunk);
  const int sub = (chunk + HPT - 1) / HPT;
  const int s0 = kstart + cchunk * sub;
  const int s1 = min(kend, s0 + sub);
  const device T* kp = keys + kv_batch_head_idx * k_head_stride + s0 * k_seq_stride + simd_lid * qk_per_thread;
  const device T* vp = values + kv_batch_head_idx * v_head_stride + s0 * v_seq_stride + simd_lid * v_per_thread;
  U q[HPT][qk_per_thread];
  for (int j = 0; j < HPT; j++) {
    const device T* qp = queries + (base_head + h0 + j) * D + simd_lid * qk_per_thread;
    for (int i = 0; i < qk_per_thread; i++) q[j][i] = static_cast<U>(scale) * qp[i];
  }
  U max_score[HPT];
  U sum_exp_score[HPT];
  U o[HPT][v_per_thread];
  for (int j = 0; j < HPT; j++) {
    max_score[j] = -FLT_MAX;
    sum_exp_score[j] = 0;
    for (int i = 0; i < v_per_thread; i++) o[j][i] = 0;
  }
  for (int t = s0; t < s1; t++) {
    U kr[qk_per_thread];
    U vr[v_per_thread];
    for (int i = 0; i < qk_per_thread; i++) kr[i] = kp[i];
    for (int i = 0; i < v_per_thread; i++) vr[i] = vp[i];
    kp += k_seq_stride;
    vp += v_seq_stride;
    for (int j = 0; j < HPT; j++) {
      U score = 0;
      for (int i = 0; i < qk_per_thread; i++) score += q[j][i] * kr[i];
      score = simd_sum(score);
      U new_max = max(max_score[j], score);
      U factor = fast::exp(max_score[j] - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score[j] = new_max;
      sum_exp_score[j] = sum_exp_score[j] * factor + exp_score;
      for (int i = 0; i < v_per_thread; i++) o[j][i] = o[j][i] * factor + exp_score * vr[i];
    }
  }
  threadgroup U o_sh[G * HPT * V];
  threadgroup U se_sh[G * HPT];
  threadgroup U mx_sh[G * HPT];
  for (int j = 0; j < HPT; j++) {
    int slot = (h0 + j) * HPT + cchunk;
    U inv = sum_exp_score[j] > 0 ? 1 / sum_exp_score[j] : 0;
    for (int i = 0; i < v_per_thread; i++) o_sh[slot * V + simd_lid * v_per_thread + i] = o[j][i] * inv;
    if (simd_lid == 0) {
      se_sh[slot] = sum_exp_score[j];
      mx_sh[slot] = max_score[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  U gmax = -FLT_MAX;
  for (int s = 0; s < HPT; s++) gmax = max(gmax, mx_sh[g * HPT + s]);
  U denom = 0;
  U acc[v_per_thread] = {0};
  for (int s = 0; s < HPT; s++) {
    U w = se_sh[g * HPT + s] * fast::exp(mx_sh[g * HPT + s] - gmax);
    denom += w;
    for (int i = 0; i < v_per_thread; i++) acc[i] += w * o_sh[(g * HPT + s) * V + simd_lid * v_per_thread + i];
  }
  const int o_offset = base_head + g;
  device T* op = out + o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  for (int i = 0; i < v_per_thread; i++) op[i] = static_cast<T>(acc[i]);
  if (simd_lid == 0) {
    sums[o_offset * blocks + block_idx] = denom;
    maxs[o_offset * blocks + block_idx] = gmax;
  }
}

template <typename T, int D>
[[kernel]] void suc_sdpa_2pass_2(
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& blocks [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  typedef float U;
  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * blocks;
  maxs += q_offset * blocks;
  out += q_offset * D + simd_gid * elem_per_thread;
  U sum_exp_score = 0.0;
  U max_score = -FLT_MAX;
  for (int b = 0; b < blocks / BN; ++b) max_score = max(max_score, maxs[simd_lid + BN * b]);
  max_score = simd_max(max_score);
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);
    for (int i = 0; i < elem_per_thread; i++) o[i] += factor * static_cast<U>(partials[i]);
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) out[i] = static_cast<T>(o[i]);
  }
}

// ================================================================================== batched prefill GEMM (suc_mm_*)
// Y (n, OUT) = X (n, IN) · Ŵᵀ for the batched prefill of the frontier graphs (not row-exact: the default since
// 2026-10-10; --prefill exact keeps the row-exact gemvs): 8 x 8 simdgroup MMAs with float32 accumulation and one
// rounding to the output dtype. Threadgroup tile: BM rows x 64 outputs, k steps of 32; 4 simdgroups, each all BM rows x
// 16 outputs. The weight tile (64 x 32) is decoded into threadgroup memory as float by a loader (the decoded element
// is the one the decode-step gemv uses: affine T(q·s + b), dense T(w)); x is staged as float. IN % 32 == 0 (host).
// Grid (ceil(n / BM), ceil(OUT / 64)) threadgroups of 128 (M tiles of one weight tile adjacent: weight reuse in cache).
constant constexpr int SUC_MM_BN = 64, SUC_MM_BK = 32, SUC_MM_LD = SUC_MM_BK + 4;

template <typename T, int BM>
METAL_FUNC void suc_mm_stage_x(const device T* x, uint xs, uint n, uint r0, int k0, threadgroup float* As, uint lid) {
    for (uint e = lid; e < uint(BM * 8); e += 128) {
        const uint r = e / 8, c = (e % 8) * 4;
        float4 v = float4(0.0f);
        if (r0 + r < n) v = float4(*(const device vec<T, 4>*)(x + ulong(r0 + r) * xs + uint(k0) + c));
        *(threadgroup float4*)(As + r * SUC_MM_LD + c) = v;
    }
}

// acc (BM x 64 tile) -> y rows r0 .., columns o0 .. (bounds n / OUT), rounded to TY. Cs: 4 x 64 floats.
template <typename TY, int BM>
METAL_FUNC void suc_mm_store(thread simdgroup_float8x8 (&acc)[BM / 8][2], device TY* y, uint ys, uint n, int OUT,
                             uint r0, int o0, threadgroup float* Cs, uint sg, uint lane) {
    threadgroup float* cs = Cs + sg * 64;
    for (int i = 0; i < BM / 8; ++i)
        for (int j = 0; j < 2; ++j) {
            simdgroup_store(acc[i][j], cs, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                const uint r = r0 + uint(i * 8) + e / 8;
                const int o = o0 + int(sg) * 16 + j * 8 + int(e % 8);
                if (r < n && o < OUT) y[ulong(r) * ys + o] = static_cast<TY>(cs[e]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
}

// One k step: As (BM x 32) times Bs (64 outputs x 32)ᵀ into acc.
template <int BM>
METAL_FUNC void suc_mm_step(thread simdgroup_float8x8 (&acc)[BM / 8][2], const threadgroup float* As,
                            const threadgroup float* Bs, uint sg) {
    for (int kk = 0; kk < SUC_MM_BK; kk += 8) {
        simdgroup_float8x8 b0, b1;
        simdgroup_load(b0, Bs + (sg * 16) * SUC_MM_LD + kk, SUC_MM_LD, ulong2(0, 0), true);
        simdgroup_load(b1, Bs + (sg * 16 + 8) * SUC_MM_LD + kk, SUC_MM_LD, ulong2(0, 0), true);
        for (int i = 0; i < BM / 8; ++i) {
            simdgroup_float8x8 a;
            simdgroup_load(a, As + (i * 8) * SUC_MM_LD + kk, SUC_MM_LD);
            simdgroup_multiply_accumulate(acc[i][0], a, b0, acc[i][0]);
            simdgroup_multiply_accumulate(acc[i][1], a, b1, acc[i][1]);
        }
    }
}

// Affine weights (packed (OUT, WORDS) u32 rows, scales / biases (OUT, IN / GS) TS): element T(q·s + b).
template <typename T, int BM, int BITS, int GS, int IN, int OUT, int WORDS, typename TS, typename TY>
[[kernel]] void suc_mm_affine(
    const device T* x [[buffer(0)]],
    const device uint32_t* packed [[buffer(1)]],
    const device TS* scales [[buffer(2)]],
    const device TS* biases [[buffer(3)]],
    device TY* y [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& xs [[buffer(6)]],
    constant uint& ys [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float As[BM * SUC_MM_LD];
    threadgroup float Bs[SUC_MM_BN * SUC_MM_LD];
    threadgroup float Cs[4 * 64];
    const uint r0 = tg.x * BM;
    const int o0 = int(tg.y) * SUC_MM_BN;
    simdgroup_float8x8 acc[BM / 8][2];
    for (int i = 0; i < BM / 8; ++i) acc[i][0] = acc[i][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < IN; k0 += SUC_MM_BK) {
        suc_mm_stage_x<T, BM>(x, xs, n, r0, k0, As, lid);
        for (uint e = lid; e < uint(SUC_MM_BN * 8); e += 128) {
            const int o = o0 + int(e / 8), c = int(e % 8) * 4;
            float4 v = float4(0.0f);
            if (o < OUT) {
                const device uint32_t* wr = packed + ulong(o) * WORDS;
                const int j = k0 + c;
                const float s = float(scales[ulong(o) * (IN / GS) + j / GS]);
                const float b = float(biases[ulong(o) * (IN / GS) + j / GS]);
                for (int t = 0; t < 4; ++t) {
                    float q = float(suc_m_unpack<BITS>(wr, uint(j + t)));
                    float w = q * s;
                    w = w + b;
                    v[t] = float(static_cast<T>(w));
                }
            }
            *(threadgroup float4*)(Bs + (e / 8) * SUC_MM_LD + uint(c)) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        suc_mm_step<BM>(acc, As, Bs, sg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    suc_mm_store<TY, BM>(acc, y, ys, n, OUT, r0, o0, Cs, sg, lane);
}

// Dense weights W (OUT, IN) in TW: element T(w).
template <typename T, int BM, typename TW, int IN, int OUT, typename TY>
[[kernel]] void suc_mm_dense(
    const device T* x [[buffer(0)]],
    const device TW* w [[buffer(1)]],
    device TY* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& xs [[buffer(4)]],
    constant uint& ys [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float As[BM * SUC_MM_LD];
    threadgroup float Bs[SUC_MM_BN * SUC_MM_LD];
    threadgroup float Cs[4 * 64];
    const uint r0 = tg.x * BM;
    const int o0 = int(tg.y) * SUC_MM_BN;
    simdgroup_float8x8 acc[BM / 8][2];
    for (int i = 0; i < BM / 8; ++i) acc[i][0] = acc[i][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < IN; k0 += SUC_MM_BK) {
        suc_mm_stage_x<T, BM>(x, xs, n, r0, k0, As, lid);
        for (uint e = lid; e < uint(SUC_MM_BN * 8); e += 128) {
            const int o = o0 + int(e / 8), c = int(e % 8) * 4;
            float4 v = float4(0.0f);
            if (o < OUT) {
                const vec<TW, 4> wv = *(const device vec<TW, 4>*)(w + ulong(o) * IN + uint(k0 + c));
                // float output (router / compressor gemvs on float32 inputs): the stored weight; else rounded to T
                for (int t = 0; t < 4; ++t)
                    v[t] = metal::is_same_v<TY, float> ? float(wv[t]) : float(static_cast<T>(wv[t]));
            }
            *(threadgroup float4*)(Bs + (e / 8) * SUC_MM_LD + uint(c)) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        suc_mm_step<BM>(acc, As, Bs, sg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    suc_mm_store<TY, BM>(acc, y, ys, n, OUT, r0, o0, Cs, sg, lane);
}

// Trellis-coded (tcq) weights on the suc_mm skeleton (batched prefill, not row-exact): element (o, j) decoded at
// random access from its L-bit window of the row's sequence j / T (= the windows suc_tcq_mma reads), times the row
// scale (applied per weight here, per output in suc_tcq_*). MODE 0: one tensor, x / y rows r < n (grid.x = row tiles of
// BM); MODE 1 / 2: expert tiles (grid.x = tile t, BT = n rows per tile, expert teidx[t]) over stacked (E, OUT, NSEQ,
// NW) / (E, OUT) arrays (1) or the offset table eoff (E, 3) (2; ~0 = empty slot, skipped).
// RUNS = 1 (tile modes): grid.x = run index; run r = runs[r] (first tile, tiles) of consecutive tiles of ONE expert
// (rows first · n .. (first + tiles) · n - 1, tiles · n <= BM): the expert's weights are decoded once per run.
template <typename T, int BM, typename TSC, int L, int V, int KV, int TT, int NW, int IN, int OUT, int CODE, int MODE,
          typename TO, int RUNS>
[[kernel]] void suc_tcq_mm(
    const device T* x [[buffer(0)]],
    const device uint32_t* words [[buffer(1)]],
    const device TSC* scales [[buffer(2)]],
    const device float* cst [[buffer(3)]],
    const device uint32_t* teidx [[buffer(4)]],
    device TO* y [[buffer(5)]],
    const device ulong* eoff [[buffer(6)]],
    constant uint& n [[buffer(7)]],
    const device uint2* runs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float As[BM * SUC_MM_LD];
    threadgroup float Bs[SUC_MM_BN * SUC_MM_LD];
    threadgroup float Cs[4 * 64];
    constexpr uint MASKL = (1u << L) - 1u;
    uint t0 = tg.x, nr = n;
    if (RUNS) {
        const uint2 rr = runs[tg.x];
        t0 = rr.x;
        nr = rr.y * n;
    }
    constexpr uint NB = NW * 32;
    constexpr uint NSEQ = IN / TT;
    uint r0 = 0;
    if (MODE == 0) {
        r0 = tg.x * BM;
    } else {
        const ulong e = ulong(teidx[t0]);
        if (MODE == 2) {
            if (eoff[3 * e + 1] == ~0ul) return;
            words += eoff[3 * e + 1];
            scales += eoff[3 * e + 2];
        } else {
            words += e * ulong(OUT) * NSEQ * NW;
            scales += e * ulong(OUT);
        }
        x += ulong(t0) * n * IN;
        y += ulong(t0) * n * OUT;
    }
    float mu[V], cc[V];
    for (int j = 0; j < V; ++j) mu[j] = cst[2 * j], cc[j] = cst[2 * j + 1];
    const int o0 = int(tg.y) * SUC_MM_BN;
    simdgroup_float8x8 acc[BM / 8][2];
    for (int i = 0; i < BM / 8; ++i) acc[i][0] = acc[i][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < IN; k0 += SUC_MM_BK) {
        suc_mm_stage_x<T, BM>(x, uint(IN), nr, r0, k0, As, lid);
        for (uint e = lid; e < uint(SUC_MM_BN * 8); e += 128) {
            const int o = o0 + int(e / 8), c = int(e % 8) * 4;
            float4 v = float4(0.0f);
            if (o < OUT) {
                const float s = float(scales[o]);
                for (int t = 0; t < 4; ++t) {
                    const uint j = uint(k0 + c + t);
                    const uint q = j / TT, st = (j % TT) / V, jj = (j % TT) % V;
                    const uint p = uint(int((st + 1) * KV) - L + int(NB)) % NB;
                    const device uint* sw = words + (ulong(o) * NSEQ + q) * NW;
                    const uint w0 = p >> 5, off = p & 31u;
                    const uint u = off == 0 ? sw[w0] : ((sw[w0] >> off) | (sw[(w0 + 1) % NW] << (32u - off)));
                    float raw[V];
                    suc_tcq_raw<CODE, V>(u & MASKL, raw);
                    v[t] = (raw[jj] - mu[jj]) * cc[jj] * s;
                }
            }
            *(threadgroup float4*)(Bs + (e / 8) * SUC_MM_LD + uint(c)) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        suc_mm_step<BM>(acc, As, Bs, sg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    suc_mm_store<TO, BM>(acc, y, uint(OUT), nr, OUT, r0, o0, Cs, sg, lane);
}
