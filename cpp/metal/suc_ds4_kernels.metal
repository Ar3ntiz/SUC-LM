// deepseek_v4 (DeepSeek-V4-Flash) model-graph kernels of the C++ runtime (cpp/src/metal/deepseek_v4.cpp) and the native
// MXFP4 / mixed-codec expert-stack kernels (cpp/src/metal/expert_mixed.cpp). Reference semantics:
// suc/models/mlx_deepseek_v4.py on the Python SUC1 runtime (suc.runtime.suc1_model.load_suc1_model), one decode row.
// Appended after cpp/metal/suc_qwen4_kernels.metal at build time (uses suc_model_kernels / qwen4 helpers: q4_key,
// q4_index_select).
//
// MLX's operation order was measured against MLX 0.32.3 (probes in the milestone-6 notes of cpp/README.md):
//   * every MLX element-wise op rounds its result: kernels that chain several such ops start with
//     `#pragma clang fp contract(off)` (the Metal compiler otherwise fuses a*b + c into an fma);
//   * transcendental ops are MLX's: exp / sigmoid / log / sqrt / rsqrt / cos / sin precise, softmax and attention
//     fast::exp;
//   * row sums (mean of squares) follow MLX's row reduce: TPG threads, 4 contiguous elements per thread per step,
//     simd_sum, then across simdgroups; TPG = 32 for N <= 512, 128 for 1024, 512 for 2048, 1024 from 4096 on;
//   * column sums (reductions over a non-contiguous axis: hyper-connection streams, compressor rows, indexer heads,
//     routed experts) follow MLX's column reduce: up to 16 rows as 8 interleaved sequential partials summed in order,
//     from 32 rows on 32 interleaved partials combined by simd_sum;
//   * the 4 x 4 hyper-connection matmul (comb^T @ streams) is an fma chain over the 4 streams (MLX steel gemm, f32);
//   * mx.softmax over a non-last axis is max, precise exp, column sum, division; over the last axis it is
//     softmax_single_row (fast::exp, sequential per-thread sums, simd_sum, multiply by the reciprocal);
//   * mx.hadamard_transform on bf16: hadamard_n with max radix 16, intermediates rounded to T after each radix pass.
//
// ds4_mxfp4_qmv is adapted from mlx/backend/metal/kernels/fp_quantized.h (fp_qmv_impl, fp_qmv_fast_impl, qdot,
// fp4_e2m1, fp8_e8m0) and ds4_sdpa_vector from sdpa_vector.h, ds4_hadamard from hadamard.h, ds4_fb_pv from gemv.h
// (GEMVTKernel), ds4_softmax_row / _looped from softmax.h, the fp8 conversion from fp8.h (MLX, Copyright © 2023-2025
// Apple Inc., MIT License); the arithmetic of the ported parts is unchanged.

// ---------------------------------------------------------------------------------------------- helpers
inline float ds4_sigmoid(float x) {   // MLX Sigmoid (float)
    float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
    return x < 0.0f ? y : 1.0f - y;
}
inline float ds4_log1p(float x) {     // MLX utils.h log1p
    float xp1 = 1.0f + x;
    if (xp1 == metal::numeric_limits<float>::max()) return metal::numeric_limits<float>::max();
    if (xp1 == 1.0f) return x;
    return x * (metal::precise::log(xp1) / (xp1 - 1.0f));
}
inline float ds4_logaddexp0(float x) {   // MLX LogAddExp(x, 0)
    float maxval = metal::max(x, 0.0f), minval = metal::min(x, 0.0f);
    return (minval == -INFINITY || maxval == INFINITY) ? maxval
                                                       : (maxval + ds4_log1p(metal::precise::exp(minval - maxval)));
}

// E4M3 conversion of MLX fp8.h (PyTorch's saturating round-to-nearest-even) and back through half.
inline uint8_t ds4_to_e4m3(float f) {
    uint32_t fp8_max = 543 << 21;
    uint32_t denorm_mask = 141 << 23;
    uint32_t f_bits = as_type<uint32_t>(f);
    uint32_t sign = f_bits & 0x80000000;
    uint8_t bits;
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
        bits = 0x7E;
    } else {
        if (f_bits < (121 << 23)) {
            f_bits = as_type<uint32_t>(as_type<float>(f_bits) + as_type<float>(denorm_mask));
            bits = static_cast<uint8_t>(f_bits - denorm_mask);
        } else {
            uint8_t mant_odd = (f_bits >> 20) & 1;
            f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
            f_bits += mant_odd;
            bits = static_cast<uint8_t>(f_bits >> 20);
        }
    }
    bits |= static_cast<uint8_t>(sign >> 24);
    return bits;
}
inline float ds4_from_e4m3(uint8_t bits) {
    uint16_t v = bits & 127;
    uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;
    uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
    half converted = as_type<half>(u);
    return static_cast<float>(converted * 256.0h);
}
// 2^ceil(log2(amax * inv_max)) by bit manipulation (mlx_deepseek_v4._pow2_ceil_scale), float32
inline float ds4_pow2_ceil(float amax, float inv_max) {
    uint v = as_type<uint>(amax * inv_max);
    int e = int((v >> 23) & 0xFF) - 127 + ((v & 0x7FFFFF) != 0 ? 1 : 0);
    e = metal::clamp(e, -126, 127);
    return as_type<float>(uint(e + 127) << 23);
}

// Column sum of R values v(r) (one column of a non-contiguous reduction) in MLX's column-reduce order. For R <= 16 a
// single thread (8 interleaved partials summed in order); the caller passes lane 0's value only.
template <typename F>
inline float ds4_colsum_small(F v, int R) {
    float part[8];
    for (int y = 0; y < 8; ++y) part[y] = 0.0f;
    for (int y = 0; y < 8; ++y)
        for (int r = y; r < R; r += 8) part[y] = v(r) + part[y];
    float t = part[0];
    for (int y = 1; y < 8; ++y) t = part[y] + t;
    return t;
}

// ================================================================================================ MXFP4 experts
// mx.gather_qmm(x, w, scales, rhs_indices, transpose=True, group_size=32, bits=4, mode="mxfp4") for one row per
// expert (decode): MLX's fp_gather_qmv (fp_qmv_impl) / fp_gather_qmv_fast (fp_qmv_fast_impl, K % 512 == 0 and
// N % 8 == 0), group size 32, E2M1 codes, E8M0 scales, no global scale. Row n reads expert eoff[eidx[n]] (byte offsets
// of its codes / scales in the bound views); an offset of ~0 marks an empty slot (the row is skipped).
// Threadgroup (32, 2, 1); grid of threadgroups (1, ceil(out / 8), N).
inline float ds4_fp4_value(uint8_t bits) {   // fp4_e2m1 -> float16 -> float (MLX fp4.h)
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0h;
    return static_cast<float>((bits & 8) ? -converted : converted);
}
inline float ds4_e8m0(uint8_t s) {
    uint32_t out = (s == 0 ? 0x400000u : (static_cast<uint32_t>(s) << 23));
    return as_type<float>(out);
}
template <typename T, int values_per_thread>
inline void ds4_load_vector(const device T* x, thread float* x_thread) {
    for (int i = 0; i < values_per_thread; i++) x_thread[i] = x[i];
}
template <typename T, int values_per_thread>
inline void ds4_load_vector_safe(const device T* x, thread float* x_thread, int N) {
    for (int i = 0; i < N; i++) x_thread[i] = x[i];
    for (int i = N; i < values_per_thread; i++) x_thread[i] = 0;
}
template <int values_per_thread>
inline float ds4_qdot(const device uint8_t* w, const thread float* x_thread, float scale) {
    float accum = 0;
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
        accum += (x_thread[4 * i] * ds4_fp4_value(uint8_t(ws[i])) +
                  x_thread[4 * i + 1] * ds4_fp4_value(uint8_t(ws[i] >> 4)) +
                  x_thread[4 * i + 2] * ds4_fp4_value(uint8_t(ws[i] >> 8)) +
                  x_thread[4 * i + 3] * ds4_fp4_value(uint8_t(ws[i] >> 12)));
    }
    return scale * accum;
}
template <int values_per_thread>
inline float ds4_qdot_safe(const device uint8_t* w, const thread float* x_thread, float scale, int N) {
    float accum = 0;
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
        accum += (x_thread[4 * i] * ds4_fp4_value(uint8_t(ws[i])) +
                  x_thread[4 * i + 1] * ds4_fp4_value(uint8_t(ws[i] >> 4)) +
                  x_thread[4 * i + 2] * ds4_fp4_value(uint8_t(ws[i] >> 8)) +
                  x_thread[4 * i + 3] * ds4_fp4_value(uint8_t(ws[i] >> 12)));
    }
    return scale * accum;
}

template <typename T>
METAL_FUNC void ds4_fp_qmv_fast_impl(const device uint32_t* w, const device uint8_t* scales, const device T* x,
                                     device T* y, const int in_vec_size, const int out_vec_size, uint3 tid,
                                     uint simd_gid, uint simd_lid) {
    constexpr int group_size = 32;
    constexpr int results_per_simdgroup = 4;
    constexpr int packs_per_thread = 2;
    constexpr int num_simdgroups = 2;
    constexpr int pack_factor = 8;
    constexpr int bytes_per_pack = 4;
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * 32;
    constexpr int scale_step_per_thread = group_size / values_per_thread;
    const device uint8_t* ws = (const device uint8_t*)w;
    thread float x_thread[values_per_thread];
    thread float result[results_per_simdgroup] = {0};
    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / group_size;
    const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
    ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + out_row;
    for (int k = 0; k < in_vec_size; k += block_size) {
        ds4_load_vector<T, values_per_thread>(x, x_thread);
        for (int row = 0; row < results_per_simdgroup; row++) {
            auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
            const device auto* sl = scales + row * in_vec_size_g;
            float s = ds4_e8m0(sl[0]);
            result[row] += ds4_qdot<values_per_thread>(wl, x_thread, s);
        }
        ws += block_size * bytes_per_pack / pack_factor;
        scales += block_size / group_size;
        x += block_size;
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0) y[row] = static_cast<T>(result[row]);
    }
}

template <typename T>
METAL_FUNC void ds4_fp_qmv_impl(const device uint32_t* w, const device uint8_t* scales, const device T* x,
                                device T* y, const int in_vec_size, const int out_vec_size, uint3 tid, uint simd_gid,
                                uint simd_lid) {
    constexpr int group_size = 32;
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int packs_per_thread = 1;
    constexpr int pack_factor = 8;
    constexpr int bytes_per_pack = 4;
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * 32;
    constexpr int scale_step_per_thread = group_size / values_per_thread;
    const device uint8_t* ws = (const device uint8_t*)w;
    thread float x_thread[values_per_thread];
    thread float result[results_per_simdgroup] = {0};
    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / group_size;
    const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
    const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);
    if (out_row >= out_vec_size) return;
    if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
        ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        x += tid.x * in_vec_size + simd_lid * values_per_thread;
        y += tid.x * out_vec_size + out_row;
        int k = 0;
        for (; k < in_vec_size - block_size; k += block_size) {
            ds4_load_vector<T, values_per_thread>(x, x_thread);
            for (int row = 0; row < results_per_simdgroup && out_row + row < out_vec_size; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                const device auto* sl = scales + row * in_vec_size_g;
                float s = ds4_e8m0(sl[0]);
                result[row] += ds4_qdot<values_per_thread>(wl, x_thread, s);
            }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / group_size;
            x += block_size;
        }
        const int remaining = clamp(static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0,
                                    values_per_thread);
        if (remaining > 0) {
            ds4_load_vector_safe<T, values_per_thread>(x, x_thread, remaining);
            for (int row = 0; row < results_per_simdgroup && out_row + row < out_vec_size; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                const device auto* sl = scales + row * in_vec_size_g;
                float s = ds4_e8m0(sl[0]);
                result[row] += ds4_qdot<values_per_thread>(wl, x_thread, s);
            }
        }
        for (int row = 0; row < results_per_simdgroup && out_row + row < out_vec_size; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) y[row] = static_cast<T>(result[row]);
        }
    } else {
        ws += used_out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        x += tid.x * in_vec_size + simd_lid * values_per_thread;
        y += tid.x * out_vec_size + used_out_row;
        int k = 0;
        for (; k < in_vec_size - block_size; k += block_size) {
            ds4_load_vector<T, values_per_thread>(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                const device auto* sl = scales + row * in_vec_size_g;
                float s = ds4_e8m0(sl[0]);
                result[row] += ds4_qdot<values_per_thread>(wl, x_thread, s);
            }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / group_size;
            x += block_size;
        }
        const int remaining = clamp(static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0,
                                    values_per_thread);
        if (remaining > 0) {
            ds4_load_vector_safe<T, values_per_thread>(x, x_thread, remaining);
            for (int row = 0; row < results_per_simdgroup; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                const device auto* sl = scales + row * in_vec_size_g;
                float s = ds4_e8m0(sl[0]);
                result[row] += ds4_qdot_safe<values_per_thread>(wl, x_thread, s, remaining);
            }
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) y[row] = static_cast<T>(result[row]);
        }
    }
}

template <typename T, int FAST>
[[kernel]] void ds4_mxfp4_qmv(
    const device uint8_t* wbase [[buffer(0)]],
    const device uint8_t* sbase [[buffer(1)]],
    const device ulong* eoff [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* eidx [[buffer(4)]],
    device T* y [[buffer(5)]],
    constant int& in_vec_size [[buffer(6)]],
    constant int& out_vec_size [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    const uint n = tid.z;
    const ulong e = ulong(eidx[n]);
    const ulong co = eoff[2 * e];
    if (co == ~0ul) return;
    const device uint32_t* w = (const device uint32_t*)(wbase + co);
    const device uint8_t* sc = sbase + eoff[2 * e + 1];
    const device T* xr = x + ulong(n) * ulong(in_vec_size);
    device T* yr = y + ulong(n) * ulong(out_vec_size);
    const uint3 t2 = uint3(0, tid.y, 0);
    if (FAST) ds4_fp_qmv_fast_impl<T>(w, sc, xr, yr, in_vec_size, out_vec_size, t2, simd_gid, simd_lid);
    else ds4_fp_qmv_impl<T>(w, sc, xr, yr, in_vec_size, out_vec_size, t2, simd_gid, simd_lid);
}

// Batched prefill: ds4_fp_qmv_fast_impl / ds4_fp_qmv_impl over BT rows that share one expert (a tile). The weights
// and scales are read once per tile; row b keeps its own x_thread / result registers and gets exactly the arithmetic
// and accumulation order of the one-row kernel (same ds4_qdot per block, same simd_sum), so every row is bit-identical
// to ds4_mxfp4_qmv. x rows b at x + b · in_vec_size, y rows at y + b · out_vec_size.
template <typename T, int BT>
METAL_FUNC void ds4_fp_qmv_fast_rows(const device uint32_t* w, const device uint8_t* scales, const device T* x,
                                     device T* y, const int in_vec_size, const int out_vec_size, uint3 tid,
                                     uint simd_gid, uint simd_lid) {
    constexpr int group_size = 32;
    constexpr int results_per_simdgroup = 4;
    constexpr int packs_per_thread = 2;
    constexpr int num_simdgroups = 2;
    constexpr int pack_factor = 8;
    constexpr int bytes_per_pack = 4;
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * 32;
    constexpr int scale_step_per_thread = group_size / values_per_thread;
    const device uint8_t* ws = (const device uint8_t*)w;
    thread float x_thread[BT][values_per_thread];
    thread float result[BT][results_per_simdgroup];
    for (int b = 0; b < BT; ++b)
        for (int r = 0; r < results_per_simdgroup; ++r) result[b][r] = 0;
    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / group_size;
    const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
    ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += simd_lid * values_per_thread;
    y += out_row;
    for (int k = 0; k < in_vec_size; k += block_size) {
        for (int b = 0; b < BT; ++b) ds4_load_vector<T, values_per_thread>(x + b * in_vec_size, x_thread[b]);
        for (int row = 0; row < results_per_simdgroup; row++) {
            auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
            const device auto* sl = scales + row * in_vec_size_g;
            float s = ds4_e8m0(sl[0]);
            for (int b = 0; b < BT; ++b) result[b][row] += ds4_qdot<values_per_thread>(wl, x_thread[b], s);
        }
        ws += block_size * bytes_per_pack / pack_factor;
        scales += block_size / group_size;
        x += block_size;
    }
    for (int b = 0; b < BT; ++b)
        for (int row = 0; row < results_per_simdgroup; row++) {
            result[b][row] = simd_sum(result[b][row]);
            if (simd_lid == 0) y[b * out_vec_size + row] = static_cast<T>(result[b][row]);
        }
}

template <typename T, int BT>
METAL_FUNC void ds4_fp_qmv_rows(const device uint32_t* w, const device uint8_t* scales, const device T* x,
                                device T* y, const int in_vec_size, const int out_vec_size, uint3 tid, uint simd_gid,
                                uint simd_lid) {
    constexpr int group_size = 32;
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int packs_per_thread = 1;
    constexpr int pack_factor = 8;
    constexpr int bytes_per_pack = 4;
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * 32;
    constexpr int scale_step_per_thread = group_size / values_per_thread;
    const device uint8_t* ws = (const device uint8_t*)w;
    thread float x_thread[BT][values_per_thread];
    thread float result[BT][results_per_simdgroup];
    for (int b = 0; b < BT; ++b)
        for (int r = 0; r < results_per_simdgroup; ++r) result[b][r] = 0;
    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / group_size;
    const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
    const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);
    if (out_row >= out_vec_size) return;
    const bool small = out_vec_size < (num_simdgroups * results_per_simdgroup);
    const int base_row = small ? out_row : used_out_row;
    const int nrows = small ? min(results_per_simdgroup, out_vec_size - out_row) : results_per_simdgroup;
    ws += base_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += base_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += simd_lid * values_per_thread;
    y += base_row;
    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
        for (int b = 0; b < BT; ++b) ds4_load_vector<T, values_per_thread>(x + b * in_vec_size, x_thread[b]);
        for (int row = 0; row < nrows; row++) {
            auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
            const device auto* sl = scales + row * in_vec_size_g;
            float s = ds4_e8m0(sl[0]);
            for (int b = 0; b < BT; ++b) result[b][row] += ds4_qdot<values_per_thread>(wl, x_thread[b], s);
        }
        ws += block_size * bytes_per_pack / pack_factor;
        scales += block_size / group_size;
        x += block_size;
    }
    const int remaining = clamp(static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0, values_per_thread);
    if (remaining > 0) {
        for (int b = 0; b < BT; ++b)
            ds4_load_vector_safe<T, values_per_thread>(x + b * in_vec_size, x_thread[b], remaining);
        for (int row = 0; row < nrows; row++) {
            auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
            const device auto* sl = scales + row * in_vec_size_g;
            float s = ds4_e8m0(sl[0]);
            for (int b = 0; b < BT; ++b) {
                if (small) result[b][row] += ds4_qdot<values_per_thread>(wl, x_thread[b], s);
                else result[b][row] += ds4_qdot_safe<values_per_thread>(wl, x_thread[b], s, remaining);
            }
        }
    }
    for (int b = 0; b < BT; ++b)
        for (int row = 0; row < nrows; row++) {
            result[b][row] = simd_sum(result[b][row]);
            if (simd_lid == 0) y[b * out_vec_size + row] = static_cast<T>(result[b][row]);
        }
}

// Tile t = tid.z: rows t·BT .. t·BT + BT - 1 of x / y, all through expert teidx[t] (empty slot: skipped).
template <typename T, int FAST, int BT>
[[kernel]] void ds4_mxfp4_qmv_tile(
    const device uint8_t* wbase [[buffer(0)]],
    const device uint8_t* sbase [[buffer(1)]],
    const device ulong* eoff [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* teidx [[buffer(4)]],
    device T* y [[buffer(5)]],
    constant int& in_vec_size [[buffer(6)]],
    constant int& out_vec_size [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    const uint t = tid.z;
    const ulong e = ulong(teidx[t]);
    const ulong co = eoff[2 * e];
    if (co == ~0ul) return;
    const device uint32_t* w = (const device uint32_t*)(wbase + co);
    const device uint8_t* sc = sbase + eoff[2 * e + 1];
    const device T* xr = x + ulong(t) * BT * ulong(in_vec_size);
    device T* yr = y + ulong(t) * BT * ulong(out_vec_size);
    const uint3 t2 = uint3(0, tid.y, 0);
    if (FAST) ds4_fp_qmv_fast_rows<T, BT>(w, sc, xr, yr, in_vec_size, out_vec_size, t2, simd_gid, simd_lid);
    else ds4_fp_qmv_rows<T, BT>(w, sc, xr, yr, in_vec_size, out_vec_size, t2, simd_gid, simd_lid);
}

// Batched-prefill MMA variant of ds4_mxfp4_qmv_tile (not row-exact; suc_mm_* skeleton of suc_model_kernels.metal):
// tile t = tg.x holds rows t·BT .. t·BT + BT - 1 (BT <= BM), all through expert teidx[t] (empty slot: skipped); the
// 64 x 32 weight tile is decoded to float (fp4_e2m1 code · e8m0 scale, the value the qdot of the gemv multiplies).
// Grid (count, ceil(out / 64)) threadgroups of 128.
// RUNS = 1: grid.x = run index, runs[r] = (first tile, tiles) of consecutive tiles of one expert (tiles · BT <= BM).
template <typename T, int BM, int BT, int RUNS>
[[kernel]] void ds4_mxfp4_mm_tile(
    const device uint8_t* wbase [[buffer(0)]],
    const device uint8_t* sbase [[buffer(1)]],
    const device ulong* eoff [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* teidx [[buffer(4)]],
    device T* y [[buffer(5)]],
    constant int& in_vec_size [[buffer(6)]],
    constant int& out_vec_size [[buffer(7)]],
    const device uint2* runs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float As[BM * SUC_MM_LD];
    threadgroup float Bs[SUC_MM_BN * SUC_MM_LD];
    threadgroup float Cs[4 * 64];
    uint t = tg.x, nr = BT;
    if (RUNS) {
        const uint2 rr = runs[tg.x];
        t = rr.x;
        nr = rr.y * BT;
    }
    const ulong e = ulong(teidx[t]);
    const ulong co = eoff[2 * e];
    if (co == ~0ul) return;
    const int IN = in_vec_size, OUT = out_vec_size;
    const device uint8_t* w = wbase + co;
    const device uint8_t* sc = sbase + eoff[2 * e + 1];
    const device T* xr = x + ulong(t) * BT * ulong(IN);
    device T* yr = y + ulong(t) * BT * ulong(OUT);
    const int o0 = int(tg.y) * SUC_MM_BN;
    simdgroup_float8x8 acc[BM / 8][2];
    for (int i = 0; i < BM / 8; ++i) acc[i][0] = acc[i][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < IN; k0 += SUC_MM_BK) {
        suc_mm_stage_x<T, BM>(xr, uint(IN), nr, 0, k0, As, lid);
        for (uint i = lid; i < uint(SUC_MM_BN * 8); i += 128) {
            const int o = o0 + int(i / 8), c = int(i % 8) * 4;
            float4 v = float4(0.0f);
            if (o < OUT) {
                const int j = k0 + c;
                const ushort q = *(const device ushort*)(w + ulong(o) * ulong(IN / 2) + ulong(j / 2));
                const float s = ds4_e8m0(sc[ulong(o) * ulong(IN / 32) + ulong(j / 32)]);
                v[0] = s * ds4_fp4_value(uint8_t(q));
                v[1] = s * ds4_fp4_value(uint8_t(q >> 4));
                v[2] = s * ds4_fp4_value(uint8_t(q >> 8));
                v[3] = s * ds4_fp4_value(uint8_t(q >> 12));
            }
            *(threadgroup float4*)(Bs + (i / 8) * SUC_MM_LD + uint(c)) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        suc_mm_step<BM>(acc, As, Bs, sg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    suc_mm_store<T, BM>(acc, yr, uint(OUT), nr, OUT, 0, o0, Cs, sg, lane);
}

// Mixed-codec stack: y[n, :] = ysub[sel[eidx[n]], n, :] (the sub-stack holding row n's expert). Grid (OUT, N).
template <typename T>
[[kernel]] void ds4_mixed_select(
    const device T* ysub [[buffer(0)]],
    const device uint32_t* eidx [[buffer(1)]],
    const device uint32_t* sel [[buffer(2)]],
    device T* y [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& OUT [[buffer(5)]],
    uint2 g [[thread_position_in_grid]]) {
    if (g.x >= OUT) return;
    const ulong s = sel[eidx[g.y]];
    y[ulong(g.y) * OUT + g.x] = ysub[(s * N + g.y) * OUT + g.x];
}

// ================================================================================================ hyper-connections
// Input of a mixer: xf = float(x) (W values) and ss = Σ xf² in MLX's row-reduce order (one threadgroup of TPG).
template <typename T, int W, int TPG>
[[kernel]] void ds4_hc_in(
    const device T* x [[buffer(0)]],
    device float* xf [[buffer(1)]],
    device float* ss [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    threadgroup float sh[32];
    for (uint i = lid; i < uint(W); i += TPG) xf[i] = float(x[i]);
    const device T* in = x + lid * 4;
    float total = 0.0f;
    constexpr int blocks = W / (TPG * 4);
    constexpr int extra = W - blocks * TPG * 4;
    for (int i = 0; i < blocks; i++) {
        for (int j = 0; j < 4; j++) {
            const float v = float(in[j]);
            const float sq = v * v;
            total = sq + total;
        }
        in += TPG * 4;
    }
    const int index = int(lid) * 4;
    if (index + 4 <= extra) {
        for (int j = 0; j < 4; j++) {
            const float v = float(in[j]);
            const float sq = v * v;
            total = sq + total;
        }
    } else {
        for (int j = 0; index + j < extra; j++) {
            const float v = float(in[j]);
            const float sq = v * v;
            total = sq + total;
        }
    }
    total = simd_sum(total);
    constexpr uint ns = (TPG + 31) / 32;
    if (ns > 1) {
        if (lane == 0) sh[sgid] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float v = lid < ns ? sh[lid] : 0.0f;
        total = simd_sum(v);
    }
    if (lid == 0) ss[0] = total;
}

// Mixer coefficients (one thread): rs = rsqrt(ss / W + norm_eps), mixes = raw · rs;
//   HEAD = 0 (HyperConnection.pre): pre = σ(mixes[:hc] s0 + b) + eps, post = 2 σ(mixes[hc:2hc] s1 + b),
//            comb = sinkhorn(softmax_rows(mixes[2hc:] s2 + b) + eps) (ITERS normalisations);
//   HEAD = 1 (HCHead): pre = σ(raw · rs · s0 + b) + eps.
// coef: pre (HC) | post (HC) | comb (HC, HC), float32.
template <typename TP, int HC, int ITERS, int HEAD>
[[kernel]] void ds4_hc_coef(
    const device float* raw [[buffer(0)]],
    const device float* ss [[buffer(1)]],
    const device TP* base [[buffer(2)]],
    const device TP* scale [[buffer(3)]],
    device float* coef [[buffer(4)]],
    constant float& inv_w [[buffer(5)]],
    constant float& norm_eps [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (tid != 0) return;
    const float mean = ss[0] * inv_w;
    const float rs = metal::precise::rsqrt(mean + norm_eps);
    if (HEAD) {
        const float s0 = float(scale[0]);
        for (int i = 0; i < HC; ++i) {
            const float a = raw[i] * rs;
            const float b = a * s0;
            const float c = b + float(base[i]);
            coef[i] = ds4_sigmoid(c) + eps;
        }
        return;
    }
    const float s0 = float(scale[0]), s1 = float(scale[1]), s2 = float(scale[2]);
    for (int i = 0; i < HC; ++i) {
        const float m = raw[i] * rs;
        const float a = m * s0;
        coef[i] = ds4_sigmoid(a + float(base[i])) + eps;
        const float m2 = raw[HC + i] * rs;
        const float a2 = m2 * s1;
        coef[HC + i] = 2.0f * ds4_sigmoid(a2 + float(base[HC + i]));
    }
    float c[HC][HC];
    for (int i = 0; i < HC; ++i) {
        for (int j = 0; j < HC; ++j) {
            const int k = 2 * HC + i * HC + j;
            const float m = raw[k] * rs;
            const float a = m * s2;
            c[i][j] = a + float(base[k]);
        }
        // softmax_single_row (one thread holds the row): max, fast::exp, sequential sum, reciprocal multiply
        float mx = -metal::numeric_limits<float>::max();
        for (int j = 0; j < HC; ++j) mx = (mx < c[i][j]) ? c[i][j] : mx;
        float nrm = 0.0f;
        for (int j = 0; j < HC; ++j) {
            const float e = fast::exp(c[i][j] - mx);
            c[i][j] = e;
            nrm += e;
        }
        nrm = 1.0f / nrm;
        for (int j = 0; j < HC; ++j) c[i][j] = c[i][j] * nrm;
        for (int j = 0; j < HC; ++j) c[i][j] = c[i][j] + eps;
    }
    // column normalisation, then ITERS - 1 x (row, column)
    for (int j = 0; j < HC; ++j) {
        float s = 0.0f;
        for (int i = 0; i < HC; ++i) s = c[i][j] + s;
        const float d = s + eps;
        for (int i = 0; i < HC; ++i) c[i][j] = c[i][j] / d;
    }
    for (int it = 0; it < ITERS - 1; ++it) {
        for (int i = 0; i < HC; ++i) {
            float s = 0.0f;
            for (int j = 0; j < HC; ++j) s = c[i][j] + s;
            const float d = s + eps;
            for (int j = 0; j < HC; ++j) c[i][j] = c[i][j] / d;
        }
        for (int j = 0; j < HC; ++j) {
            float s = 0.0f;
            for (int i = 0; i < HC; ++i) s = c[i][j] + s;
            const float d = s + eps;
            for (int i = 0; i < HC; ++i) c[i][j] = c[i][j] / d;
        }
    }
    for (int i = 0; i < HC; ++i)
        for (int j = 0; j < HC; ++j) coef[2 * HC + i * HC + j] = c[i][j];
}

// Parallel form of the Sinkhorn coefficients (one simdgroup instead of one thread). Lane l < HC·HC holds element
// (i, j) = (l / HC, l % HC) of the comb matrix; the other lanes mirror lane 0 and write nothing. Every row / column sum
// is taken by each lane over simd_shuffle copies of the same values in the same order as the one-thread loop (sequential
// from 0.0f), and every element-wise op is the same, so each lane ends with exactly the value the sequential kernel
// computes for its element.
template <int HC, int ITERS>
inline float ds4_sinkhorn_lane(float c, uint l, float eps) {
#pragma clang fp contract(off)
    const ushort i = ushort(l / HC), j = ushort(l % HC), rb = ushort(i * HC);
    float mx = -metal::numeric_limits<float>::max();
    for (ushort k = 0; k < HC; ++k) {
        const float v = simd_shuffle(c, ushort(rb + k));
        mx = (mx < v) ? v : mx;
    }
    const float e = fast::exp(c - mx);
    float nrm = 0.0f;
    for (ushort k = 0; k < HC; ++k) nrm += simd_shuffle(e, ushort(rb + k));
    nrm = 1.0f / nrm;
    c = e * nrm;
    c = c + eps;
    {
        float s = 0.0f;
        for (ushort k = 0; k < HC; ++k) s = simd_shuffle(c, ushort(k * HC + j)) + s;
        const float d = s + eps;
        c = c / d;
    }
    for (int it = 0; it < ITERS - 1; ++it) {
        float s = 0.0f;
        for (ushort k = 0; k < HC; ++k) s = simd_shuffle(c, ushort(rb + k)) + s;
        float d = s + eps;
        c = c / d;
        s = 0.0f;
        for (ushort k = 0; k < HC; ++k) s = simd_shuffle(c, ushort(k * HC + j)) + s;
        d = s + eps;
        c = c / d;
    }
    return c;
}

// ds4_hc_coef on one simdgroup (threadgroup of 32, one threadgroup): the same coefficients bit for bit, the Sinkhorn
// normalisations element-parallel (ds4_sinkhorn_lane), pre / post on lanes 0 .. 2·HC - 1. HC · HC <= 32.
template <typename TP, int HC, int ITERS, int HEAD>
[[kernel]] void ds4_hc_coef_par(
    const device float* raw [[buffer(0)]],
    const device float* ss [[buffer(1)]],
    const device TP* base [[buffer(2)]],
    const device TP* scale [[buffer(3)]],
    device float* coef [[buffer(4)]],
    constant float& inv_w [[buffer(5)]],
    constant float& norm_eps [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    static_assert(HC * HC <= 32, "one simdgroup holds the comb matrix");
    // row-batched launches (grid = rows): row r reads raw / coef row r ((2 + HC) · HC floats) and ss[r]; one-row
    // launches (grid 1) are unchanged
    constexpr uint MIXR = (2 + HC) * HC;
    raw += row * MIXR;
    ss += row;
    coef += row * MIXR;
    const float mean = ss[0] * inv_w;
    const float rs = metal::precise::rsqrt(mean + norm_eps);
    if (HEAD) {
        if (lane < uint(HC)) {
            const float s0 = float(scale[0]);
            const float a = raw[lane] * rs;
            const float b = a * s0;
            const float c = b + float(base[lane]);
            coef[lane] = ds4_sigmoid(c) + eps;
        }
        return;
    }
    if (lane < uint(HC)) {
        const float m = raw[lane] * rs;
        const float a = m * float(scale[0]);
        coef[lane] = ds4_sigmoid(a + float(base[lane])) + eps;
    } else if (lane < uint(2 * HC)) {
        const float m2 = raw[lane] * rs;
        const float a2 = m2 * float(scale[1]);
        coef[lane] = 2.0f * ds4_sigmoid(a2 + float(base[lane]));
    }
    const uint l = lane < uint(HC * HC) ? lane : 0u;
    const int k = 2 * HC + int(l);
    const float m = raw[k] * rs;
    const float a = m * float(scale[2]);
    const float c = ds4_sinkhorn_lane<HC, ITERS>(a + float(base[k]), l, eps);
    if (lane < uint(HC * HC)) coef[2 * HC + lane] = c;
}

// ds4_hc_collapse with the pre coefficients computed in place from the mixer logits (the expressions of ds4_hc_coef /
// HEAD), so that it does not wait for the coefficient kernel: y[d] = T(Σ_i pre_i · T'(x[i, d])) where x is the stream
// in T (= xf, its float32 copy). Grid (D).
template <typename T, typename TP, int D, int HC, int HEAD>
[[kernel]] void ds4_hc_collapse_pre(
    const device T* x [[buffer(0)]],
    const device float* raw [[buffer(1)]],
    const device float* ss [[buffer(2)]],
    const device TP* base [[buffer(3)]],
    const device TP* scale [[buffer(4)]],
    device T* y [[buffer(5)]],
    constant float& inv_w [[buffer(6)]],
    constant float& norm_eps [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint d [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (d >= uint(D)) return;
    const float mean = ss[0] * inv_w;
    const float rs = metal::precise::rsqrt(mean + norm_eps);
    const float s0 = float(scale[0]);
    float acc = 0.0f;
    for (int i = 0; i < HC; ++i) {
        float pre;
        if (HEAD) {
            const float a = raw[i] * rs;
            const float b = a * s0;
            const float c = b + float(base[i]);
            pre = ds4_sigmoid(c) + eps;
        } else {
            const float m = raw[i] * rs;
            const float a = m * s0;
            pre = ds4_sigmoid(a + float(base[i])) + eps;
        }
        const float p = pre * float(x[i * D + d]);
        acc = p + acc;
    }
    y[d] = static_cast<T>(acc);
}

// Element-wise helpers without fma contraction (MLX rounds every op).
inline float ds4_rs_of(float ss, float inv_w, float eps) {
#pragma clang fp contract(off)
    const float mean = ss * inv_w;
    return metal::precise::rsqrt(mean + eps);
}
inline float ds4_mul_nc(float a, float b) {
#pragma clang fp contract(off)
    return a * b;
}
inline float ds4_add_nc(float a, float b) {
#pragma clang fp contract(off)
    return a + b;
}
// The pre coefficient of stream i as the coefficient kernels compute it. MODE 0: ds4 HyperConnection.pre, 1: ds4
// HCHead, 2: GLM mHC (logits not rescaled).
template <typename TP, int MODE>
inline float ds4_hc_pre_coef(const device float* raw, const device TP* base, float s0, float rs, float eps, int i) {
#pragma clang fp contract(off)
    if (MODE == 1) {
        const float a = raw[i] * rs;
        const float b = a * s0;
        const float c = b + float(base[i]);
        return ds4_sigmoid(c) + eps;
    }
    if (MODE == 0) {
        const float m = raw[i] * rs;
        const float a = m * s0;
        return ds4_sigmoid(a + float(base[i])) + eps;
    }
    const float a = raw[i] * s0;
    return ds4_sigmoid(a + float(base[i])) + eps;
}

// Collapse + RMSNorm in one threadgroup (D <= 4096, TPG = 32 · ceil(D / 4 / 32), MLX rms_single_row): thread lid
// collapses its 4 elements j = 4·lid + e exactly as ds4_hc_collapse_pre (c_j = T(Σ_i pre_i · x[i, j])), keeps them in
// registers and runs suc_rms_norm's body on them (same sums, same rounding): yn = w · T(c · rsqrt(mean(c²) + eps)).
// Replaces collapse -> barrier -> suc_rms_norm; the collapsed row itself is not stored.
template <typename T, typename TP, int D, int HC, int MODE, int TPG>
[[kernel]] void ds4_hc_collapse_rms(
    const device T* x [[buffer(0)]],
    const device float* raw [[buffer(1)]],
    const device float* ss [[buffer(2)]],
    const device TP* base [[buffer(3)]],
    const device TP* scale [[buffer(4)]],
    const device T* w [[buffer(5)]],
    device T* yn [[buffer(6)]],
    constant float& inv_w [[buffer(7)]],
    constant float& norm_eps [[buffer(8)]],
    constant float& hc_eps [[buffer(9)]],
    constant float& rms_eps [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    static_assert(D <= 4096, "rms_single_row");
    constexpr int R = 4;
    // row-batched launches (grid = rows, MODE 0 / 2): stream row r (HC · D), mixer row r, ss[r], output row r
    x += ulong(row) * HC * D;
    raw += row * ((2 + HC) * HC);
    ss += row;
    yn += ulong(row) * D;
    threadgroup float local_inv_mean[32];
    float pre[HC];
    {
        const float rs = MODE == 2 ? 1.0f : ds4_rs_of(ss[0], inv_w, norm_eps);
        const float s0 = float(scale[0]);
        for (int i = 0; i < HC; ++i) pre[i] = ds4_hc_pre_coef<TP, MODE>(raw, base, s0, rs, hc_eps, i);
    }
    T c[R];
    for (int e = 0; e < R; ++e) {
        const uint j = lid * R + e;
        float acc = 0.0f;
        if (j < uint(D))
            for (int i = 0; i < HC; ++i) acc = ds4_add_nc(ds4_mul_nc(pre[i], float(x[i * D + j])), acc);
        c[e] = static_cast<T>(acc);
    }
    float acc = 0;
    if (lid * R + R <= uint(D)) {
        for (int i = 0; i < R; i++) {
            float xi = c[i];
            acc += xi * xi;
        }
    } else {
        for (int i = 0; i < R; i++) {
            if ((lid * R + i) < uint(D)) {
                float xi = c[i];
                acc += xi * xi;
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
        if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / D + rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = local_inv_mean[0];
    for (int i = 0; i < R; i++) {
        uint j = lid * R + i;
        if (j < uint(D)) yn[j] = w[j] * static_cast<T>(c[i] * inv);
    }
}

// suc_moe_gemv (MLX GEMVKernel order, (SN, TN, BN) tiling) with the input read in TX and converted to float32 per
// element, float32 weights and a float32 output: the hyper-connection mixer gemv straight on the stream in T, whose
// float32 copy xf it would otherwise wait for. float(x[j]) is the value xf[j] holds, so the result is that of the gemv
// over xf.
// SCALED = 1 (GLM mHC: the mixer input normalized before fn, glm_hc_scale): the input element is
// float(x[j]) · rsqrt(ss / W + eps), computed with glm_hc_scale's expressions.
template <typename TX, typename TW, int IN, int OUT, int SN, int TN, int BN, int SCALED>
[[kernel]] void ds4_gemv_xf(
    const device TX* x [[buffer(0)]],
    const device TW* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    const device float* ss [[buffer(3)]],
    constant float& inv_w [[buffer(4)]],
    constant float& eps [[buffer(5)]],
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
    const device TX* xr = x + ulong(tid.y) * IN;
    const device TW* wr = w + ulong(live ? row : 0) * IN;
    const float rs = SCALED ? ds4_rs_of(ss[0], inv_w, eps) : 1.0f;
    float acc = 0.0f;
    for (int i = 0; i < IN; i += BLK) {
        const int b = i + (sgN * SN + thrN) * TN;
        for (int tn = 0; tn < TN; ++tn) {
            const int j = b + tn;
            if (j < IN) {
                float v = SCALED ? ds4_mul_nc(float(xr[j]), rs) : float(xr[j]);
                float m = float(float(wr[j]));
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
    if (sgN == 0 && thrN == 0 && live) y[ulong(tid.y) * OUT + row] = acc;
}

// y[d] = T(Σ_i pre_i · xf[i, d]) (products rounded, column sum over the HC streams). Grid (D).
template <typename T, int D, int HC>
[[kernel]] void ds4_hc_collapse(
    const device float* xf [[buffer(0)]],
    const device float* coef [[buffer(1)]],
    device T* y [[buffer(2)]],
    uint d [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (d >= uint(D)) return;
    float acc = 0.0f;
    for (int i = 0; i < HC; ++i) {
        const float p = coef[i] * xf[i * D + d];
        acc = p + acc;
    }
    y[d] = static_cast<T>(acc);
}

// y[j, d] = T(post_j · f[d] + Σ_i comb[i, j] · res[i, d]) (the matmul an fma chain over i from 0). Grid (D, HC).
template <typename T, int D, int HC>
[[kernel]] void ds4_hc_post(
    const device T* f [[buffer(0)]],
    const device T* res [[buffer(1)]],
    const device float* coef [[buffer(2)]],
    device T* y [[buffer(3)]],
    uint3 g [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    const uint d = g.x, j = g.y;
    if (d >= uint(D)) return;
    // row-batched launches (grid z = rows): f row r (D), res / y rows r (HC · D), coef row r ((2 + HC) · HC)
    f += ulong(g.z) * D;
    res += ulong(g.z) * HC * D;
    y += ulong(g.z) * HC * D;
    coef += g.z * ((2 + HC) * HC);
    float acc = 0.0f;
    for (int i = 0; i < HC; ++i) acc = metal::fma(coef[2 * HC + i * HC + j], float(res[i * D + d]), acc);
    const float p = coef[HC + j] * float(f[d]);
    y[j * D + d] = static_cast<T>(p + acc);
}

// ================================================================================================ attention pieces
// Per-head RMS norm of q in float32 (the unweighted norm of Attention.__call__, MLX row-reduce order with TPG threads),
// rounded to T, then RoPE on the last RD channels (interleaved pairs, float32, rounded to T). One threadgroup per head.
template <typename T, int HD, int RD, int TPG>
[[kernel]] void ds4_qnorm_rope(
    device T* q [[buffer(0)]],
    const device float* inv_freq [[buffer(1)]],
    constant uint& pos [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint h [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    threadgroup float sh[32];
    threadgroup float inv_sh;
    threadgroup T rbuf[RD];
    device T* x = q + ulong(h) * HD;
    const device T* in = x + lid * 4;
    float total = 0.0f;
    constexpr int blocks = HD / (TPG * 4);
    constexpr int extra = HD - blocks * TPG * 4;
    for (int i = 0; i < blocks; i++) {
        for (int j = 0; j < 4; j++) {
            const float v = float(in[j]);
            const float sq = v * v;
            total = sq + total;
        }
        in += TPG * 4;
    }
    const int index = int(lid) * 4;
    if (index + 4 <= extra) {
        for (int j = 0; j < 4; j++) {
            const float v = float(in[j]);
            const float sq = v * v;
            total = sq + total;
        }
    } else {
        for (int j = 0; index + j < extra; j++) {
            const float v = float(in[j]);
            const float sq = v * v;
            total = sq + total;
        }
    }
    total = simd_sum(total);
    constexpr uint ns = (TPG + 31) / 32;
    if (ns > 1) {
        if (lane == 0) sh[sgid] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float v = lid < ns ? sh[lid] : 0.0f;
        total = simd_sum(v);
    }
    if (lid == 0) inv_sh = metal::precise::rsqrt(total * (1.0f / float(HD)) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    const float inv = inv_sh;
    // rounded normalised values (every thread reads before any write)
    constexpr int PER = (HD + TPG - 1) / TPG;
    T vals[PER];
    for (int k = 0; k < PER; ++k) {
        const int i = int(lid) + k * TPG;
        if (i < HD) vals[k] = static_cast<T>(float(x[i]) * inv);
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (int k = 0; k < PER; ++k) {
        const int i = int(lid) + k * TPG;
        if (i < HD - RD) x[i] = vals[k];
    }
    // pairs of the RoPE part: thread t < RD / 2 owns pair t (channels HD - RD + 2t, + 1)
    for (int k = 0; k < PER; ++k) {
        const int i = int(lid) + k * TPG;
        if (i >= HD - RD && i < HD) rbuf[i - (HD - RD)] = vals[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = lid; t < uint(RD / 2); t += TPG) {
        const float ang = float(pos) * inv_freq[t];
        const float c = metal::precise::cos(ang), s = metal::precise::sin(ang);
        const float a = float(rbuf[2 * t]), b = float(rbuf[2 * t + 1]);
        const float ac = a * c, bs = b * s, as = a * s, bc = b * c;
        x[HD - RD + 2 * t] = static_cast<T>(ac - bs);
        x[HD - RD + 2 * t + 1] = static_cast<T>(as + bc);
    }
}

// RoPE (Rope.apply) on the last RD channels of rows of D, in place, all rows at one position; INV: inverse (sin
// negated). Grid (RD / 2, rows).
template <typename T, int D, int RD, int INV>
[[kernel]] void ds4_rope(
    device T* x [[buffer(0)]],
    const device float* inv_freq [[buffer(1)]],
    constant uint& pos [[buffer(2)]],
    uint2 g [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    const uint t = g.x;
    if (t >= uint(RD / 2)) return;
    device T* r = x + ulong(g.y) * D + (D - RD);
    const float ang = float(pos) * inv_freq[t];
    const float c = metal::precise::cos(ang);
    float s = metal::precise::sin(ang);
    if (INV) s = -s;
    const float a = float(r[2 * t]), b = float(r[2 * t + 1]);
    const float ac = a * c, bs = b * s, as = a * s, bc = b * c;
    r[2 * t] = static_cast<T>(ac - bs);
    r[2 * t + 1] = static_cast<T>(as + bc);
}

// KV store with the FP8 simulation of the non-RoPE channels (fp8_round_trip(x[:NQ], 64), straight-through value =
// the quantized one): y[i] = x[i] for i >= NQ (or SIM = 0). One threadgroup of 64 threads per block of 64 (grid
// (D / 64 threadgroups)); NQ % 64 == 0.
template <typename T, int D, int NQ, int SIM>
[[kernel]] void ds4_kv_store(
    const device T* x [[buffer(0)]],
    device T* y [[buffer(1)]],
    uint blk [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]) {
#pragma clang fp contract(off)
    threadgroup float red[2];
    const uint i = blk * 64 + lid;
    const float v = i < uint(D) ? float(x[i]) : 0.0f;
    if (!SIM || blk * 64 >= uint(NQ)) {
        if (i < uint(D)) y[i] = x[i];
        return;
    }
    float a = simd_max(metal::abs(v));
    if ((lid & 31) == 0) red[sgid] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float amax = metal::max(metal::max(red[0], red[1]), 1e-4f);
    const float s = ds4_pow2_ceil(amax, 1.0f / 448.0f);
    const float q = metal::clamp(v / s, -448.0f, 448.0f);
    const float r = ds4_from_e4m3(ds4_to_e4m3(q)) * s;
    y[i] = static_cast<T>(r);
}

// FP4 simulation (fp4_round_trip(x, 32)) in place on rows of D: one simdgroup per block of 32. Grid (D / 32 * 32, rows).
template <typename T, int D>
[[kernel]] void ds4_fp4_sim(
    device T* x [[buffer(0)]],
    uint2 g [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    device T* r = x + ulong(g.y) * D;
    const uint i = g.x;
    const float v = float(r[i]);
    const float amax = metal::max(simd_max(metal::abs(v)), 6.0f * 1.1754943508222875e-38f);
    const float s = ds4_pow2_ceil(amax, 1.0f / 6.0f);
    const float y = metal::clamp(v / s, -6.0f, 6.0f);
    const float a = metal::abs(y);
    const uint code = uint(a > 0.25f) + uint(a >= 0.75f) + uint(a > 1.25f) + uint(a >= 1.75f) + uint(a > 2.5f) +
                      uint(a >= 3.5f) + uint(a > 5.0f);
    const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float sg = y < 0.0f ? -1.0f : 1.0f;
    const float m = mags[code] * sg;
    const float q = m * s;
    simdgroup_barrier(mem_flags::mem_device);
    r[i] = static_cast<T>(q);
    (void)lane;
}

// mx.hadamard_transform(x, scale) on rows of N = 2^k (MLX hadamard_n, max radix 16, read width 4), in place allowed.
// Grid (N / 16, rows), threadgroup (N / 16, 1).
template <short R>
METAL_FUNC void ds4_radix(thread float* x) {
    constexpr short logR = __builtin_ctz(R);
    short h = 1;
    for (short s = 0; s < logR; s++) {
        for (short i = 0; i < R / 2; i++) {
            short k = i & (h - 1);
            short j = ((i - k) << 1) + k;
            float a = x[j];
            float b = x[j + h];
            x[j] = a + b;
            x[j + h] = a - b;
        }
        h <<= 1;
    }
}
template <typename T, int N>
[[kernel]] void ds4_hadamard(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    uint3 elem [[thread_position_in_grid]]) {
    constexpr short max_radix = 16;
    constexpr short read_width = 4;
    constexpr short num_threads = N / max_radix;
    constexpr short logN = __builtin_ctz(N);
    constexpr short logR = __builtin_ctz(max_radix);
    constexpr short num_steps = logN / logR;
    constexpr short logFinal = logN % logR;
    constexpr short final_radix = 1 << (logFinal);
    int batch_idx = elem.y * N;
    short i = elem.x;
    threadgroup T buf[N];
    for (short j = 0; j < max_radix / read_width; j++) {
        short index = j * read_width * num_threads + i * read_width;
        for (short r = 0; r < read_width; r++) buf[index + r] = in[batch_idx + index + r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float x[max_radix];
    short h = 1;
    for (short s = 0; s < num_steps; s++) {
        short k = i & (h - 1);
        short j = ((i - k) << logR) + k;
        for (short r = 0; r < max_radix; r++) x[r] = buf[j + h * r];
        ds4_radix<max_radix>(x);
        for (short r = 0; r < max_radix; r++) buf[j + h * r] = T(x[r]);
        h <<= logR;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (final_radix > 1) {
        for (int t = 0; t < max_radix / final_radix; t++) {
            short index = i + t * num_threads;
            short k = index & (h - 1);
            short j = ((index - k) << logFinal) + k;
            for (short r = 0; r < final_radix; r++) x[r] = buf[j + h * r];
            ds4_radix<final_radix>(x);
            for (short r = 0; r < final_radix; r++) buf[j + h * r] = T(x[r]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (short j = 0; j < max_radix / read_width; j++) {
        short index = j * read_width * num_threads + i * read_width;
        for (short r = 0; r < read_width; r++) out[batch_idx + index + r] = T(buf[index + r] * scale);
    }
}

// Compressor pooling of one complete group (Compressor.__call__): rows NR = R (no overlap) or 2R (overlap: the first R
// rows are the previous group's first D channels (zeros / -inf before the first group), the last R the current
// group's channels D .. 2D - 1); gs = sc + ape; p = softmax over the rows (max, precise exp, column sum, division);
// pooled[c] = T(Σ_r kv_r · p_r) (column sums in MLX's order). One threadgroup of 32 per channel c < D.
// kv / sc: group buffers (R, CD) float32; ape (R, CD).
template <typename T, typename TA, int D, int R, int OV>
[[kernel]] void ds4_comp_pool(
    const device float* kv_cur [[buffer(0)]],
    const device float* sc_cur [[buffer(1)]],
    const device float* kv_prev [[buffer(2)]],
    const device float* sc_prev [[buffer(3)]],
    const device TA* ape [[buffer(4)]],
    device T* out [[buffer(5)]],
    constant uint& has_prev [[buffer(6)]],
    uint c [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    constexpr int CD = OV ? 2 * D : D;
    constexpr int NR = OV ? 2 * R : R;
    auto kvv = [&](int r) -> float {
        if (OV) {
            if (r < R) return has_prev ? kv_prev[r * CD + c] : 0.0f;
            return kv_cur[(r - R) * CD + D + c];
        }
        return kv_cur[r * CD + c];
    };
    auto scv = [&](int r) -> float {
        if (OV) {
            if (r < R) return has_prev ? (sc_prev[r * CD + c] + float(ape[r * CD + c])) : -INFINITY;
            return sc_cur[(r - R) * CD + D + c] + float(ape[(r - R) * CD + D + c]);
        }
        return sc_cur[r * CD + c] + float(ape[r * CD + c]);
    };
    if (NR <= 16) {
        if (lane != 0) return;
        float mx = -INFINITY;
        for (int r = 0; r < NR; ++r) mx = metal::max(mx, scv(r));
        float e[NR];
        for (int r = 0; r < NR; ++r) e[r] = metal::precise::exp(scv(r) - mx);
        const float s = ds4_colsum_small([&](int r) { return e[r]; }, NR);
        const float pooled = ds4_colsum_small([&](int r) { const float p = e[r] / s; return kvv(r) * p; }, NR);
        out[c] = static_cast<T>(pooled);
        return;
    }
    float mx = -INFINITY;
    for (int r = int(lane); r < NR; r += 32) mx = metal::max(mx, scv(r));
    mx = simd_max(mx);
    float s = 0.0f;
    for (int r = int(lane); r < NR; r += 32) {
        const float e = metal::precise::exp(scv(r) - mx);
        s = e + s;
    }
    s = simd_sum(s);
    float t = 0.0f;
    for (int r = int(lane); r < NR; r += 32) {
        const float e = metal::precise::exp(scv(r) - mx);
        const float p = e / s;
        const float v = kvv(r) * p;
        t = v + t;
    }
    t = simd_sum(t);
    if (lane == 0) out[c] = static_cast<T>(t);
}


// ------------------------------------------------------------------------------- batched prefill (not row-exact)
// Row-batched variants of the per-token kernels above for the chunk attention of the default batched prefill:
// ds4_qnorm_rope_rows (head h at position pos + h / hpp), ds4_rope_rows (row r at pos + (r / rpp) · pstep), ds4_kv_store_rows
// (row r of x -> row r of y with stride ys; grid (D / 64, rows)), ds4_comp_pool_rows (group gi = grid.y: cur rows at
// kv_cur + gi·R·CD, prev = the previous group's rows, the first group's prev from kv_prev / has_prev).
template <typename T, int HD, int RD, int TPG>
[[kernel]] void ds4_qnorm_rope_rows(
    device T* q [[buffer(0)]],
    const device float* inv_freq [[buffer(1)]],
    constant uint& pos [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    constant uint& hpp [[buffer(4)]],
    uint h [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    threadgroup float sh[32];
    threadgroup float inv_sh;
    threadgroup T rbuf[RD];
    device T* x = q + ulong(h) * HD;
    const device T* in = x + lid * 4;
    float total = 0.0f;
    constexpr int blocks = HD / (TPG * 4);
    constexpr int extra = HD - blocks * TPG * 4;
    for (int i = 0; i < blocks; i++) {
        for (int j = 0; j < 4; j++) {
            const float v = float(in[j]);
            const float sq = v * v;
            total = sq + total;
        }
        in += TPG * 4;
    }
    const int index = int(lid) * 4;
    if (index + 4 <= extra) {
        for (int j = 0; j < 4; j++) {
            const float v = float(in[j]);
            const float sq = v * v;
            total = sq + total;
        }
    } else {
        for (int j = 0; index + j < extra; j++) {
            const float v = float(in[j]);
            const float sq = v * v;
            total = sq + total;
        }
    }
    total = simd_sum(total);
    constexpr uint ns = (TPG + 31) / 32;
    if (ns > 1) {
        if (lane == 0) sh[sgid] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float v = lid < ns ? sh[lid] : 0.0f;
        total = simd_sum(v);
    }
    if (lid == 0) inv_sh = metal::precise::rsqrt(total * (1.0f / float(HD)) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    const float inv = inv_sh;
    // rounded normalised values (every thread reads before any write)
    constexpr int PER = (HD + TPG - 1) / TPG;
    T vals[PER];
    for (int k = 0; k < PER; ++k) {
        const int i = int(lid) + k * TPG;
        if (i < HD) vals[k] = static_cast<T>(float(x[i]) * inv);
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (int k = 0; k < PER; ++k) {
        const int i = int(lid) + k * TPG;
        if (i < HD - RD) x[i] = vals[k];
    }
    // pairs of the RoPE part: thread t < RD / 2 owns pair t (channels HD - RD + 2t, + 1)
    for (int k = 0; k < PER; ++k) {
        const int i = int(lid) + k * TPG;
        if (i >= HD - RD && i < HD) rbuf[i - (HD - RD)] = vals[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = lid; t < uint(RD / 2); t += TPG) {
        const float ang = float(pos + h / hpp) * inv_freq[t];
        const float c = metal::precise::cos(ang), s = metal::precise::sin(ang);
        const float a = float(rbuf[2 * t]), b = float(rbuf[2 * t + 1]);
        const float ac = a * c, bs = b * s, as = a * s, bc = b * c;
        x[HD - RD + 2 * t] = static_cast<T>(ac - bs);
        x[HD - RD + 2 * t + 1] = static_cast<T>(as + bc);
    }
}

template <typename T, int D, int RD, int INV>
[[kernel]] void ds4_rope_rows(
    device T* x [[buffer(0)]],
    const device float* inv_freq [[buffer(1)]],
    constant uint& pos [[buffer(2)]],
    constant uint& rpp [[buffer(3)]],
    constant uint& pstep [[buffer(4)]],
    uint2 g [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    const uint t = g.x;
    if (t >= uint(RD / 2)) return;
    device T* r = x + ulong(g.y) * D + (D - RD);
    const float ang = float(pos + (g.y / rpp) * pstep) * inv_freq[t];
    const float c = metal::precise::cos(ang);
    float s = metal::precise::sin(ang);
    if (INV) s = -s;
    const float a = float(r[2 * t]), b = float(r[2 * t + 1]);
    const float ac = a * c, bs = b * s, as = a * s, bc = b * c;
    r[2 * t] = static_cast<T>(ac - bs);
    r[2 * t + 1] = static_cast<T>(as + bc);
}

template <typename T, int D, int NQ, int SIM>
[[kernel]] void ds4_kv_store_rows(
    const device T* x [[buffer(0)]],
    device T* y [[buffer(1)]],
    constant uint& ys [[buffer(2)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]) {
#pragma clang fp contract(off)
    threadgroup float red[2];
    const uint blk = tgp.x;
    x += ulong(tgp.y) * D;
    y += ulong(tgp.y) * ys;
    const uint i = blk * 64 + lid;
    const float v = i < uint(D) ? float(x[i]) : 0.0f;
    if (!SIM || blk * 64 >= uint(NQ)) {
        if (i < uint(D)) y[i] = x[i];
        return;
    }
    float a = simd_max(metal::abs(v));
    if ((lid & 31) == 0) red[sgid] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float amax = metal::max(metal::max(red[0], red[1]), 1e-4f);
    const float s = ds4_pow2_ceil(amax, 1.0f / 448.0f);
    const float q = metal::clamp(v / s, -448.0f, 448.0f);
    const float r = ds4_from_e4m3(ds4_to_e4m3(q)) * s;
    y[i] = static_cast<T>(r);
}

template <typename T, typename TA, int D, int R, int OV>
[[kernel]] void ds4_comp_pool_rows(
    const device float* kv_cur_ [[buffer(0)]],
    const device float* sc_cur_ [[buffer(1)]],
    const device float* kv_prev_ [[buffer(2)]],
    const device float* sc_prev_ [[buffer(3)]],
    const device TA* ape [[buffer(4)]],
    device T* out_ [[buffer(5)]],
    constant uint& has_prev [[buffer(6)]],
    uint2 cg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    constexpr int CD = OV ? 2 * D : D;
    constexpr int NR = OV ? 2 * R : R;
    const uint c = cg.x, gi = cg.y;
    const device float* kv_cur = kv_cur_;
    const device float* sc_cur = sc_cur_;
    const device float* kv_prev = kv_prev_;
    const device float* sc_prev = sc_prev_;
    device T* out = out_;
    if (gi > 0) {
        kv_prev = kv_cur + ulong(gi - 1) * R * CD;
        sc_prev = sc_cur + ulong(gi - 1) * R * CD;
    }
    kv_cur += ulong(gi) * R * CD;
    sc_cur += ulong(gi) * R * CD;
    out += ulong(gi) * D;
    const uint hp = gi > 0 ? 1u : has_prev;
    auto kvv = [&](int r) -> float {
        if (OV) {
            if (r < R) return hp ? kv_prev[r * CD + c] : 0.0f;
            return kv_cur[(r - R) * CD + D + c];
        }
        return kv_cur[r * CD + c];
    };
    auto scv = [&](int r) -> float {
        if (OV) {
            if (r < R) return hp ? (sc_prev[r * CD + c] + float(ape[r * CD + c])) : -INFINITY;
            return sc_cur[(r - R) * CD + D + c] + float(ape[(r - R) * CD + D + c]);
        }
        return sc_cur[r * CD + c] + float(ape[r * CD + c]);
    };
    if (NR <= 16) {
        if (lane != 0) return;
        float mx = -INFINITY;
        for (int r = 0; r < NR; ++r) mx = metal::max(mx, scv(r));
        float e[NR];
        for (int r = 0; r < NR; ++r) e[r] = metal::precise::exp(scv(r) - mx);
        const float s = ds4_colsum_small([&](int r) { return e[r]; }, NR);
        const float pooled = ds4_colsum_small([&](int r) { const float p = e[r] / s; return kvv(r) * p; }, NR);
        out[c] = static_cast<T>(pooled);
        return;
    }
    float mx = -INFINITY;
    for (int r = int(lane); r < NR; r += 32) mx = metal::max(mx, scv(r));
    mx = simd_max(mx);
    float s = 0.0f;
    for (int r = int(lane); r < NR; r += 32) {
        const float e = metal::precise::exp(scv(r) - mx);
        s = e + s;
    }
    s = simd_sum(s);
    float t = 0.0f;
    for (int r = int(lane); r < NR; r += 32) {
        const float e = metal::precise::exp(scv(r) - mx);
        const float p = e / s;
        const float v = kvv(r) * p;
        t = v + t;
    }
    t = simd_sum(t);
    if (lane == 0) out[c] = static_cast<T>(t);
}

// Ring <-> linear key rows: DIR 0 gather lin[r] = ring[(start + r) % W], DIR 1 scatter ring[(start + r) % W] = lin[r].
// Grid (D, rows).
template <typename T, int DIR>
[[kernel]] void ds4_ring_copy(
    device T* lin [[buffer(0)]],
    device T* ring [[buffer(1)]],
    constant uint& start [[buffer(2)]],
    constant uint& W [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    uint2 g [[thread_position_in_grid]]) {
    if (g.x >= D) return;
    const ulong rr = ulong((start + g.y) % W) * D + g.x, lr = ulong(g.y) * D + g.x;
    if (DIR == 0) lin[lr] = ring[rr];
    else ring[rr] = lin[lr];
}

// Chunk attention (batched prefill, float32 online softmax): row i (position p0 + i) of q (n, H, D) attends to its
// window keys = rows [i + WP - wn_i + 1, i + WP] of the linear key buffer klin (WP = keys before the chunk in klin,
// wn_i = min(p0 + i + 1, window)) and the first ne[i] compressed entries, plus the head's sink logit T(sink). One
// simdgroup per (row, head), 4 heads per threadgroup. Grid (H / 4, n) threadgroups of 128. Output o (n, H, D) in T.
// RND = 1: the roundings of the decode step's unfused (matmul) attention, q' = T(T(scale) · q), scores in T and the
// probabilities rounded to T before p·V (two passes over the keys); RND = 2: the same in one online-softmax pass
// without the probability rounding (faster); RND = 0: the vector kernel's roundings (float32 q · scale and scores).
template <typename T, int D, int RND>
[[kernel]] void ds4_attn_chunk(
    const device T* q [[buffer(0)]],
    const device T* klin [[buffer(1)]],
    const device T* ent [[buffer(2)]],
    const device uint32_t* ne [[buffer(3)]],
    const device float* sinks [[buffer(4)]],
    device T* o [[buffer(5)]],
    constant uint& H [[buffer(6)]],
    constant uint& p0 [[buffer(7)]],
    constant uint& WP [[buffer(8)]],
    constant uint& window [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    const device uint32_t* sidx [[buffer(11)]],
    const device uint32_t* scnt [[buffer(12)]],
    constant uint& topk [[buffer(13)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int PT = D / 32;
    const uint i = tg.y, h = tg.x * 4 + sg;
    if (h >= H) return;
    const device T* qr = q + (ulong(i) * H + h) * D + lane * PT;
    float qv[PT], ov[PT];
    const T ts = static_cast<T>(scale);
    for (int k = 0; k < PT; ++k) {
        qv[k] = RND ? float(static_cast<T>(float(ts) * float(qr[k]))) : scale * float(qr[k]);
        ov[k] = 0.0f;
    }
    const float sink = float(static_cast<T>(sinks[h]));
    const uint p = p0 + i, wn = min(p + 1, window);
    const ulong k0 = ulong(i) + WP + 1 - wn;
    // topk > 0 (indexer layers past topk entries): the row's selected entries sidx[i · topk ..] (ascending, scnt[i])
    const bool sel = topk > 0 && ne[i] > topk;
    const uint nei = sel ? scnt[i] : ne[i];
    const device uint32_t* si = sidx + ulong(i) * topk;
    auto key = [&](uint j) {
        return (j < wn ? klin + (k0 + j) * D : ent + ulong(sel ? si[j - wn] : j - wn) * D) + lane * PT;
    };
    auto score = [&](uint j) {
        const device T* kr = key(j);
        float d = 0.0f;
        for (int k = 0; k < PT; ++k) d = metal::fma(qv[k], float(kr[k]), d);
        d = simd_sum(d);
        return RND ? float(static_cast<T>(d)) : d;
    };
    const uint N = wn + nei;
    device T* orow = o + (ulong(i) * H + h) * D + lane * PT;
    if (RND == 1) {   // two passes: probabilities normalised and rounded to T before p·V, as softmax + matmul
        // up to SCAP keys: every score (a T value) computed once and kept in threadgroup memory for the sum and p·V
        // passes (the same values as recomputing them, so the same bits)
        constexpr uint SCAP = sizeof(T) == 2 ? 2048 : 1024;   // 16 KB
        threadgroup T scs[4][SCAP];
        threadgroup T* sc = scs[sg];
        float m = sink;
        if (N <= SCAP) {
            for (uint j = 0; j < N; ++j) {
                const float sj = score(j);
                if (lane == 0) sc[j] = static_cast<T>(sj);
                m = metal::max(m, sj);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            float l = metal::precise::exp(sink - m);
            for (uint j = 0; j < N; ++j) l += metal::precise::exp(float(sc[j]) - m);
            for (uint j = 0; j < N; ++j) {
                const float pj = float(static_cast<T>(metal::precise::exp(float(sc[j]) - m) / l));
                const device T* vr = key(j);
                for (int k = 0; k < PT; ++k) ov[k] = metal::fma(pj, float(vr[k]), ov[k]);
            }
            for (int k = 0; k < PT; ++k) orow[k] = static_cast<T>(ov[k]);
            return;
        }
        for (uint j = 0; j < N; ++j) m = metal::max(m, score(j));
        float l = metal::precise::exp(sink - m);
        for (uint j = 0; j < N; ++j) l += metal::precise::exp(score(j) - m);
        for (uint j = 0; j < N; ++j) {
            const float pj = float(static_cast<T>(metal::precise::exp(score(j) - m) / l));
            const device T* vr = key(j);
            for (int k = 0; k < PT; ++k) ov[k] = metal::fma(pj, float(vr[k]), ov[k]);
        }
        for (int k = 0; k < PT; ++k) orow[k] = static_cast<T>(ov[k]);
        return;
    }
    float m = sink, l = 1.0f;   // the sink: exp(sink - m) = 1 with no value
    for (uint j = 0; j < N; ++j) {
        const float d = score(j);
        const device T* kr = key(j);
        const float mn = metal::max(m, d);
        const float a = metal::precise::exp(m - mn), b = metal::precise::exp(d - mn);
        l = l * a + b;
        for (int k = 0; k < PT; ++k) ov[k] = ov[k] * a + b * float(kr[k]);
        m = mn;
    }
    const float inv = 1.0f / l;
    for (int k = 0; k < PT; ++k) orow[k] = static_cast<T>(ov[k] * inv);
}

// ds4_attn_chunk (RND = 1) with HPS heads per simdgroup (4 simdgroups: 4 · HPS heads per threadgroup, grid
// (H / (4 · HPS), rows)): every key row is loaded once for the HPS heads of a simdgroup; each head's arithmetic (the
// score's fma chain and simd_sum, the T roundings, the two-pass softmax, p·V) is that of ds4_attn_chunk, so the same
// bits. Scores kept in threadgroup memory up to SCAP keys per head (else recomputed).
template <typename T, int D, int HPS, int KEEP>
[[kernel]] void ds4_attn_chunk_h(
    const device T* q [[buffer(0)]],
    const device T* klin [[buffer(1)]],
    const device T* ent [[buffer(2)]],
    const device uint32_t* ne [[buffer(3)]],
    const device float* sinks [[buffer(4)]],
    device T* o [[buffer(5)]],
    constant uint& H [[buffer(6)]],
    constant uint& p0 [[buffer(7)]],
    constant uint& WP [[buffer(8)]],
    constant uint& window [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    const device uint32_t* sidx [[buffer(11)]],
    const device uint32_t* scnt [[buffer(12)]],
    constant uint& topk [[buffer(13)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int PT = D / 32;
    constexpr uint SCAP = KEEP ? (sizeof(T) == 2 ? 2048 : 1024) / HPS : 1;   // 16 KB per threadgroup (KEEP)
    threadgroup T scs[4][HPS][SCAP];
    const uint i = tg.y, h0 = (tg.x * 4 + sg) * HPS;
    if (h0 >= H) return;
    float qv[HPS][PT], ov[HPS][PT], sink[HPS], m[HPS], l[HPS];
    const T ts = static_cast<T>(scale);
    for (int e = 0; e < HPS; ++e) {
        const device T* qr = q + (ulong(i) * H + h0 + e) * D + lane * PT;
        for (int k = 0; k < PT; ++k) {
            qv[e][k] = float(static_cast<T>(float(ts) * float(qr[k])));
            ov[e][k] = 0.0f;
        }
        sink[e] = float(static_cast<T>(sinks[h0 + e]));
        m[e] = sink[e];
    }
    const uint p = p0 + i, wn = min(p + 1, window);
    const ulong k0 = ulong(i) + WP + 1 - wn;
    const bool sel = topk > 0 && ne[i] > topk;
    const uint nei = sel ? scnt[i] : ne[i];
    const device uint32_t* si = sidx + ulong(i) * topk;
    auto key = [&](uint j) {
        return (j < wn ? klin + (k0 + j) * D : ent + ulong(sel ? si[j - wn] : j - wn) * D) + lane * PT;
    };
    const uint N = wn + nei;
    auto scores = [&](uint j, thread float* sv) {
        const device T* kr = key(j);
        float kv[PT];
        for (int k = 0; k < PT; ++k) kv[k] = float(kr[k]);
        for (int e = 0; e < HPS; ++e) {
            float d = 0.0f;
            for (int k = 0; k < PT; ++k) d = metal::fma(qv[e][k], kv[k], d);
            d = simd_sum(d);
            sv[e] = float(static_cast<T>(d));
        }
    };
    const bool keep = KEEP && N <= SCAP;
    for (uint j = 0; j < N; ++j) {
        float sv[HPS];
        scores(j, sv);
        for (int e = 0; e < HPS; ++e) {
            if (keep && lane == 0) scs[sg][e][j] = static_cast<T>(sv[e]);
            m[e] = metal::max(m[e], sv[e]);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = 0; e < HPS; ++e) l[e] = metal::precise::exp(sink[e] - m[e]);
    if (keep) {
        // kept scores: the exponentials of 32 keys at a time, one per lane, then the sums in key order (shuffles) and
        // p · V with each lane's p broadcast: the values and the order of every operation are those of the loops
        // below, so the same bits, with 1 / 32 of the exponentials
        for (uint c0 = 0; c0 < N; c0 += 32) {
            const uint j = c0 + lane;
            float ex[HPS];
            for (int e = 0; e < HPS; ++e) ex[e] = j < N ? metal::precise::exp(float(scs[sg][e][j]) - m[e]) : 0.0f;
            const uint cn = min(32u, N - c0);
            for (uint t = 0; t < cn; ++t)
                for (int e = 0; e < HPS; ++e) l[e] += simd_shuffle(ex[e], ushort(t));
        }
        for (uint c0 = 0; c0 < N; c0 += 32) {
            const uint j = c0 + lane;
            float pl[HPS];
            for (int e = 0; e < HPS; ++e)
                pl[e] = j < N ? float(static_cast<T>(metal::precise::exp(float(scs[sg][e][j]) - m[e]) / l[e])) : 0.0f;
            const uint cn = min(32u, N - c0);
            for (uint t = 0; t < cn; ++t) {
                const device T* vr = key(c0 + t);
                float vv[PT];
                for (int k = 0; k < PT; ++k) vv[k] = float(vr[k]);
                for (int e = 0; e < HPS; ++e) {
                    const float pj = simd_shuffle(pl[e], ushort(t));
                    for (int k = 0; k < PT; ++k) ov[e][k] = metal::fma(pj, vv[k], ov[e][k]);
                }
            }
        }
        for (int e = 0; e < HPS; ++e) {
            device T* orow = o + (ulong(i) * H + h0 + e) * D + lane * PT;
            for (int k = 0; k < PT; ++k) orow[k] = static_cast<T>(ov[e][k]);
        }
        return;
    }
    for (uint j = 0; j < N; ++j) {
        float sv[HPS];
        if (keep) {
            for (int e = 0; e < HPS; ++e) sv[e] = float(scs[sg][e][j]);
        } else {
            scores(j, sv);
        }
        for (int e = 0; e < HPS; ++e) l[e] += metal::precise::exp(sv[e] - m[e]);
    }
    for (uint j = 0; j < N; ++j) {
        float sv[HPS];
        if (keep) {
            for (int e = 0; e < HPS; ++e) sv[e] = float(scs[sg][e][j]);
        } else {
            scores(j, sv);
        }
        const device T* vr = key(j);
        float vv[PT];
        for (int k = 0; k < PT; ++k) vv[k] = float(vr[k]);
        for (int e = 0; e < HPS; ++e) {
            const float pj = float(static_cast<T>(metal::precise::exp(sv[e] - m[e]) / l[e]));
            for (int k = 0; k < PT; ++k) ov[e][k] = metal::fma(pj, vv[k], ov[e][k]);
        }
    }
    for (int e = 0; e < HPS; ++e) {
        device T* orow = o + (ulong(i) * H + h0 + e) * D + lane * PT;
        for (int k = 0; k < PT; ++k) orow[k] = static_cast<T>(ov[e][k]);
    }
}

// Lightning-indexer scores: s[j] = Σ_h max(float(T(q_h · k_j)), 0) · w_h with w_h = float(wp[h]) · c (float32), the
// sum over heads in MLX's column order. q (NH, IDD) and k (entries, IDD) in T. One threadgroup of 32 per entry j.
template <typename T, int NH, int IDD>
[[kernel]] void ds4_index_scores(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* wp [[buffer(2)]],
    device float* score [[buffer(3)]],
    constant float& cw [[buffer(4)]],
    uint j [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    const device T* kr = k + ulong(j) * IDD;
    auto head = [&](int h) -> float {
        float acc = 0.0f;
        for (int d = 0; d < IDD; ++d) acc = metal::fma(float(q[h * IDD + d]), float(kr[d]), acc);
        const float s = metal::max(float(static_cast<T>(acc)), 0.0f);
        const float w = float(wp[h]) * cw;
        return s * w;
    };
    if (NH <= 16) {
        if (lane != 0) return;
        score[j] = ds4_colsum_small(head, NH);
        return;
    }
    float t = 0.0f;
    for (int h = int(lane); h < NH; h += 32) {
        const float v = head(h);
        t = v + t;
    }
    t = simd_sum(t);
    if (lane == 0) score[j] = t;
}

// Lightning-indexer scores of a chunk's query rows (fast batched prefill): row i (q rows (NH, IDD) at i · NH · IDD,
// weights wp at i · NH) against entry j < ne[i] (the entries visible at its position), score[i · ps + j] with the
// arithmetic of ds4_index_scores. Grid (entries, rows) threadgroups of 32.
template <typename T, int NH, int IDD>
[[kernel]] void ds4_index_scores_rows(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* wp [[buffer(2)]],
    device float* score [[buffer(3)]],
    constant float& cw [[buffer(4)]],
    const device uint32_t* ne [[buffer(5)]],
    constant uint& ps [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    const uint j = tg.x, i = tg.y;
    if (j >= ne[i]) return;
    const device T* qi = q + ulong(i) * NH * IDD;
    const device T* wi = wp + ulong(i) * NH;
    const device T* kr = k + ulong(j) * IDD;
    auto head = [&](int h) -> float {
        float acc = 0.0f;
        for (int d = 0; d < IDD; ++d) acc = metal::fma(float(qi[h * IDD + d]), float(kr[d]), acc);
        const float s = metal::max(float(static_cast<T>(acc)), 0.0f);
        const float w = float(wi[h]) * cw;
        return s * w;
    };
    if (NH <= 16) {
        if (lane != 0) return;
        score[ulong(i) * ps + j] = ds4_colsum_small(head, NH);
        return;
    }
    float t = 0.0f;
    for (int h = int(lane); h < NH; h += 32) {
        const float v = head(h);
        t = v + t;
    }
    t = simd_sum(t);
    if (lane == 0) score[ulong(i) * ps + j] = t;
}

// Indexer scores of chunk rows by simdgroup-matrix products (fast batched prefill; float32 tiles, another
// accumulation order than the decode kernels): threadgroup (32 entries, row i) of NH / 16 simdgroups, S (NH, 32) =
// q_i (NH, IDD) · keys (32, IDD)ᵀ in k steps of 32, then score[i · ps + j] = Σ_h w_h · relu(s_hj) with
//   MODE 0 (DeepSeek lightning indexer): s = float(T(dot)), w_h = float(wp[i, h]) · c1
//   MODE 1 (GLM DSA pools):              s = dot · c2,       w_h = float(wp[i, h]) · c1
// for j < ne[i]. Grid (ceil(entries / 32), rows).
template <typename T, int NH, int IDD, int MODE>
[[kernel]] void ds4_index_scores_mm(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* wp [[buffer(2)]],
    device float* score [[buffer(3)]],
    constant float& c1 [[buffer(4)]],
    const device uint32_t* ne [[buffer(5)]],
    constant uint& ps [[buffer(6)]],
    constant float& c2 [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint lid = lid2.x;
    static_assert(NH % 16 == 0 && NH <= 64 && IDD % 32 == 0, "tile shape");
    constexpr int NT = NH / 16 * 32;   // threads
    constexpr int KS = 32, LD = KS + 4;
    threadgroup float Qs[NH * LD];
    threadgroup float Ks[32 * LD];
    const uint i = tg.y, j0 = tg.x * 32, nei = ne[i];
    if (j0 >= nei) return;
    const device T* qi = q + ulong(i) * NH * IDD;
    simdgroup_float8x8 acc[2][4];
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 4; ++b) acc[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < IDD; k0 += KS) {
        for (uint e = lid; e < uint(NH * KS); e += NT) {
            const uint h = e / KS, c = e % KS;
            Qs[h * LD + c] = float(qi[h * IDD + k0 + c]);
        }
        for (uint e = lid; e < uint(32 * KS); e += NT) {
            const uint r = e / KS, c = e % KS, j = j0 + r;
            Ks[r * LD + c] = j < nei ? float(k[ulong(j) * IDD + k0 + c]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < KS; kk += 8) {
            simdgroup_float8x8 b[4];
            for (int t = 0; t < 4; ++t) simdgroup_load(b[t], Ks + (t * 8) * LD + kk, LD, ulong2(0, 0), true);
            for (int a2 = 0; a2 < 2; ++a2) {
                simdgroup_float8x8 am;
                simdgroup_load(am, Qs + (sg * 16 + a2 * 8) * LD + kk, LD);
                for (int t = 0; t < 4; ++t) simdgroup_multiply_accumulate(acc[a2][t], am, b[t], acc[a2][t]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // S (NH, 32) -> Qs (row stride LD >= 32)
    for (int a2 = 0; a2 < 2; ++a2)
        for (int t = 0; t < 4; ++t) simdgroup_store(acc[a2][t], Qs + (sg * 16 + a2 * 8) * LD + t * 8, LD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32 && j0 + lid < nei) {
        const device T* wi = wp + ulong(i) * NH;
        float tot = 0.0f;
        for (int h = 0; h < NH; ++h) {
            const float d = Qs[h * LD + lid];
            const float sv = MODE == 0 ? metal::max(float(static_cast<T>(d)), 0.0f) : metal::max(d * c2, 0.0f);
            tot += sv * (float(wi[h]) * c1);
        }
        score[ulong(i) * ps + j0 + lid] = tot;
    }
}

// Top-K entries of each chunk row (q4_index_select's exact threshold search and tie rule: the K largest scores, ties
// by lowest index) as an ascending index list idx[i · K ..] of cnt[i] = min(ne[i], K) entries; rows with ne[i] <= K
// list every visible entry. One threadgroup of TPG per row.
template <int TPG>
[[kernel]] void ds4_index_select_rows(
    const device float* score [[buffer(0)]],
    device uint32_t* idx [[buffer(1)]],
    device uint32_t* cnt [[buffer(2)]],
    const device uint32_t* ne [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& ps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint red[32];
    threadgroup uint base_sh[TPG / 32 + 1];
    const uint nb = ne[row];
    score += ulong(row) * ps;
    idx += ulong(row) * K;
    uint ans = 0, lo = nb;
    const bool all = nb <= K;
    if (!all) {
        for (int bit = 31; bit >= 0; --bit) {
            const uint cand = ans | (1u << bit);
            if (q4_count<TPG>(score, nb, cand, 0, 0, red, lid, sgid, lane) >= K) ans = cand;
        }
        const uint ge = q4_count<TPG>(score, nb, ans, 0, 0, red, lid, sgid, lane);
        const uint eq = q4_count<TPG>(score, nb, ans, 1, 0xFFFFFFFFu, red, lid, sgid, lane);
        const uint need = K - (ge - eq);
        uint l = 1, h = nb;
        while (l < h) {
            const uint mid = (l + h) / 2;
            if (q4_count<TPG>(score, nb, ans, 1, mid, red, lid, sgid, lane) >= need) h = mid;
            else l = mid + 1;
        }
        lo = l;
    }
    // ordered compaction, TPG entries per pass
    uint out = 0;
    for (uint c0 = 0; c0 < nb; c0 += TPG) {
        const uint i = c0 + lid;
        uint f = 0;
        if (i < nb) {
            if (all) f = 1;
            else {
                const uint k = q4_key(score[i]);
                f = (k > ans || (k == ans && i < lo)) ? 1 : 0;
            }
        }
        const uint pre = simd_prefix_exclusive_sum(f);
        const uint tot = simd_sum(f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) base_sh[sgid] = tot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint b = out, all_t = 0;
        for (uint s = 0; s < uint(TPG / 32); ++s) {
            const uint v = base_sh[s];
            if (s < sgid) b += v;
            all_t += v;
        }
        if (f) idx[b + pre] = i;
        out += all_t;
    }
    if (lid == 0) cnt[row] = out;
}

// ---------------------------------------------------------------------------------------------- attention
// Keys / values (K = V): key i < wn is window key (ws + i) % wsz of the ring `win`, key i >= wn is compressed entry
// i - wn of `ent`; a compressed entry is used when nsel == 0 or sel[i - wn] != 0 (the indexer selection).
//
// ds4_sdpa_vector: MLX sdpa_vector (single pass, one query row) with the bool-mask and sinks branches (max_score =
// T(sink), sum_exp = 1 for simdgroup 0). Grid (H threadgroups of 1024).
template <typename T, int D>
[[kernel]] void ds4_sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* win [[buffer(1)]],
    const device T* ent [[buffer(2)]],
    const device uint8_t* sel [[buffer(3)]],
    const device float* sinks [[buffer(4)]],
    device T* out [[buffer(5)]],
    constant uint& wn [[buffer(6)]],
    constant uint& ws [[buffer(7)]],
    constant uint& wsz [[buffer(8)]],
    constant uint& nent [[buffer(9)]],
    constant uint& nsel [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread = D / BD;
    typedef float U;
    thread U q[qk_per_thread];
    thread U k[qk_per_thread];
    thread U o[v_per_thread];
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];
    const int head = tid.x;
    const int N = int(wn + nent);
    queries += head * D + simd_lid * qk_per_thread;
    out += head * D + simd_gid * v_per_thread;
    for (int i = 0; i < qk_per_thread; i++) q[i] = static_cast<U>(scale) * queries[i];
    for (int i = 0; i < v_per_thread; i++) o[i] = 0;
    U max_score = -FLT_MAX;
    U sum_exp_score = 0;
    if (simd_gid == 0) {
        max_score = static_cast<U>(static_cast<T>(sinks[head]));
        sum_exp_score = 1;
    }
    for (int i = simd_gid; i < N; i += BN) {
        const bool is_win = i < int(wn);
        bool use_key = is_win || nsel == 0 || sel[i - int(wn)] != 0;
        if (use_key) {
            const device T* row = is_win ? win + ulong((ws + uint(i)) % wsz) * D : ent + ulong(i - int(wn)) * D;
            const device T* keys = row + simd_lid * qk_per_thread;
            const device T* values = row + simd_lid * v_per_thread;
            for (int j = 0; j < qk_per_thread; j++) k[j] = keys[j];
            U score = 0;
            for (int j = 0; j < qk_per_thread; j++) score += q[j] * k[j];
            score = simd_sum(score);
            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor + exp_score * values[j];
        }
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
        for (int i = 0; i < v_per_thread; i++) out[i] = static_cast<T>(o[i]);
    }
}

// MLX's unfused attention (fast.cpp fallback; head dims other than 64 / 96 / 128 / 256, or > 32 query heads per KV
// head): s = [T(sink) | T(T(T(scale) · q) · k_i) or -inf], p = softmax(s, precise), out = p[1:] @ V.
// ds4_fb_scores: the scores in MLX's gemv order (GEMVKernel, SN = 32, TN = 4, BN = 8 when D >= 16 · N, else 1).
// s (H, N + 1) in T. Grid (ceil(N / SM) threadgroups of 32 · BN, H).
template <typename T, int D, int SN, int TN, int BN>
[[kernel]] void ds4_fb_scores(
    const device T* q [[buffer(0)]],
    const device T* win [[buffer(1)]],
    const device T* ent [[buffer(2)]],
    const device uint8_t* sel [[buffer(3)]],
    const device float* sinks [[buffer(4)]],
    device T* s_out [[buffer(5)]],
    constant uint& wn [[buffer(6)]],
    constant uint& ws [[buffer(7)]],
    constant uint& wsz [[buffer(8)]],
    constant uint& nent [[buffer(9)]],
    constant uint& nsel [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int SM = 32 / SN;
    constexpr int BLK = BN * SN * TN;
    threadgroup float tg[BN * SM];
    const int thrM = simd_lid / SN, thrN = simd_lid % SN;
    const int sgN = simd_gid;
    const int NN = int(wn + nent);
    const int row = int(tid.x) * SM + thrM;
    const bool live = row < NN;
    const int h = tid.y;
    const int rr = live ? row : 0;
    const bool is_win = rr < int(wn);
    const device T* kr = is_win ? win + ulong((ws + uint(rr)) % wsz) * D : ent + ulong(rr - int(wn)) * D;
    const T ts = static_cast<T>(scale);
    float acc = 0.0f;
    for (int i = 0; i < D; i += BLK) {
        const int b = i + (sgN * SN + thrN) * TN;
        for (int tn = 0; tn < TN; ++tn) {
            const int j = b + tn;
            if (j < D) {
                float v = float(static_cast<T>(float(ts) * float(q[h * D + j])));
                float m = float(kr[j]);
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
    if (sgN == 0 && thrN == 0 && live) {
        const bool use = row < int(wn) || nsel == 0 || sel[row - int(wn)] != 0;
        s_out[ulong(h) * (NN + 1) + 1 + row] = use ? static_cast<T>(acc) : static_cast<T>(-INFINITY);
    }
    if (sgN == 0 && thrN == 0 && row == 0) s_out[ulong(h) * (NN + 1)] = static_cast<T>(sinks[h]);
}

// MLX folds the fallback's matmuls over the H query heads of the one KV head (fast.cpp: q unflattened to
// (B, 1, H, L, D), K / V broadcast over the heads) into ONE matmul with M = H rows (matmul.cpp "Collapse batches into
// M"; p[1:] is copied to a contiguous array first), so for H > 1 the per-head gemv orders above do not apply:
//   q' Kᵀ   gemv_wide (Apple GPU generation >= 15, H <= 15 heads, bf16 / f16, two or more keys; ds4_fb_scores_wide)
//           else a steel GEMM (ds4_mma, NT); one key: MLX gemv over the H rows of q' (ds4_fb_scores with BN by H);
//   p V     a steel GEMM (ds4_mma), split-k from 512 keys on (K = keys >= max(H, D)).
// Keys are read in place: row r < wn is window ring slot (ws + r) % wsz, else compressed entry r - wn.
template <typename T, int D>
inline const device T* ds4_key_row(const device T* win, const device T* ent, int r, uint wn, uint ws, uint wsz) {
    return r < int(wn) ? win + ulong((ws + uint(r)) % wsz) * D : ent + ulong(r - int(wn)) * D;
}

// ds4_fb_scores_wide: MLX GemvWide (gemv.h) for out[h, n] = q'_h · k_n: KL lanes reduce one key row (float4 dot
// products; KL * 8-vector main blocks summed into a zeroed partial first, then the tail vectors, then the halving
// shuffles), 32 / KL keys per simdgroup, KL / 8 simdgroups = 4 keys per threadgroup; every head in turn (MLX's tiling
// of the heads over threadgroups does not change any one output). q' = T(T(scale) · q) as fast.cpp. Output as
// ds4_fb_scores. Grid (ceil(N / 4)) threadgroups of 32 · KL / 8.
template <typename T, int D, int KL>
[[kernel]] void ds4_fb_scores_wide(
    const device T* q [[buffer(0)]],
    const device T* win [[buffer(1)]],
    const device T* ent [[buffer(2)]],
    const device uint8_t* sel [[buffer(3)]],
    const device float* sinks [[buffer(4)]],
    device T* s_out [[buffer(5)]],
    constant uint& wn [[buffer(6)]],
    constant uint& ws [[buffer(7)]],
    constant uint& wsz [[buffer(8)]],
    constant uint& nent [[buffer(9)]],
    constant uint& nsel [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    constant uint& H [[buffer(12)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int UNROLL = 8;
    constexpr int RPS = 32 / KL;                 // rows per simdgroup
    constexpr int RPT = RPS * (KL / 8);          // rows per threadgroup (4)
    const int NN = int(wn + nent);
    const int k_lane = int(simd_lid) % KL, sg_row = int(simd_lid) / KL;
    const int out_row = int(tgid) * RPT + RPS * int(simd_gid) + sg_row;
    const int row = min(out_row, NN - 1);
    const device T* wrow = ds4_key_row<T, D>(win, ent, row, wn, ws, wsz);
    const T ts = static_cast<T>(scale);
    constexpr int NV4 = D / 4;
    constexpr int NMAIN = NV4 - NV4 % (KL * UNROLL);
    for (uint h = 0; h < H; ++h) {
        const device T* qh = q + ulong(h) * D;
        auto x4 = [&](int i) {
            return float4(float(static_cast<T>(float(ts) * float(qh[4 * i]))),
                          float(static_cast<T>(float(ts) * float(qh[4 * i + 1]))),
                          float(static_cast<T>(float(ts) * float(qh[4 * i + 2]))),
                          float(static_cast<T>(float(ts) * float(qh[4 * i + 3]))));
        };
        auto w4 = [&](int i) {
            return float4(float(wrow[4 * i]), float(wrow[4 * i + 1]), float(wrow[4 * i + 2]), float(wrow[4 * i + 3]));
        };
        float result = 0.0f;
        for (int base = 0; base < NMAIN; base += KL * UNROLL) {
            float4 wf[UNROLL];
            for (int i = 0; i < UNROLL; i++) wf[i] = w4(base + i * KL + k_lane);
            float acc = 0.0f;
            for (int i = 0; i < UNROLL; i++) acc += dot(wf[i], x4(base + i * KL + k_lane));
            result += acc;
        }
        for (int idx = NMAIN + k_lane; idx < NV4; idx += KL) result += dot(w4(idx), x4(idx));
        for (ushort off = KL / 2; off >= 1; off >>= 1) result += simd_shuffle_down(result, off);
        if (k_lane == 0 && out_row < NN) {
            const bool use = out_row < int(wn) || nsel == 0 || sel[out_row - int(wn)] != 0;
            s_out[ulong(h) * (NN + 1) + 1 + out_row] = use ? static_cast<T>(result) : static_cast<T>(-INFINITY);
        }
        if (k_lane == 0 && out_row == 0) s_out[ulong(h) * (NN + 1)] = static_cast<T>(sinks[h]);
    }
}

// ds4_mma: MLX's steel GEMM order (fused: the K % BK tail first, then the BK blocks; split-k: partitions of PSIZE
// values, the last taking the rest) as glm_mma (cpp/metal/suc_glm5_kernels.metal: one simdgroup per 8 x 8 output tile
// and partition, 8 x 8 x 8 simdgroup MMAs in k order), with the keys read in place:
//   NT = 1  scores: A = q (M = H, K = D) with A = T(T(ascale) · q); B row n = key n (N = keys);
//   NT = 0  p V:    A = probs (M = H rows of lda, starting at element aoff; K = keys); B row k = key k (N = D).
// C_part (parts, M, N) float32. Grid (ceil(N / 8), ceil(M / 8), parts) threadgroups of 32.
template <typename T, int D, int NT>
[[kernel]] void ds4_mma(
    const device T* A [[buffer(0)]],
    const device T* win [[buffer(1)]],
    const device T* ent [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& lda [[buffer(7)]],
    constant uint& aoff [[buffer(8)]],
    constant uint& psize [[buffer(9)]],
    constant uint& tail_bk [[buffer(10)]],
    constant float& ascale [[buffer(11)]],
    constant uint& wn [[buffer(12)]],
    constant uint& ws [[buffer(13)]],
    constant uint& wsz [[buffer(14)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tgs [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float As[64];
    threadgroup float Bs[64];
    threadgroup float Cs[64];
    const int m0 = int(tg.y) * 8, n0 = int(tg.x) * 8;
    const int part = int(tg.z), parts = int(tgs.z);
    const T ts = static_cast<T>(ascale);
    simdgroup_float8x8 c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    auto run = [&](int k0, int k1) {
        for (int kc = k0; kc < k1; kc += 8) {
            for (uint e = lane; e < 64; e += 32) {
                const int r = int(e) / 8, kk = int(e) % 8;
                const int m = m0 + r, k = kc + kk;
                float av = 0.0f;
                if (m < int(M) && k < k1) {
                    const T a = A[ulong(aoff) + ulong(m) * lda + k];
                    av = NT ? float(static_cast<T>(float(ts) * float(a))) : float(a);
                }
                As[e] = av;
                const int n = n0 + int(e) % 8, k2 = kc + int(e) / 8;
                float bv = 0.0f;
                if (n < int(N) && k2 < k1)
                    bv = NT ? float(ds4_key_row<T, D>(win, ent, n, wn, ws, wsz)[k2])
                            : float(ds4_key_row<T, D>(win, ent, k2, wn, ws, wsz)[n]);
                Bs[e] = bv;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 a, b;
            simdgroup_load(a, As, 8);
            simdgroup_load(b, Bs, 8);
            simdgroup_multiply_accumulate(c, a, b, c);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    };
    if (psize == 0) {
        const int al = (int(K) / int(tail_bk)) * int(tail_bk);
        if (al < int(K)) run(al, int(K));
        run(0, al);
    } else {
        const int k0 = part * int(psize);
        run(k0, part + 1 == parts ? int(K) : k0 + int(psize));
    }
    simdgroup_store(c, Cs, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 64; e += 32) {
        const int m = m0 + int(e) / 8, n = n0 + int(e) % 8;
        if (m < int(M) && n < int(N)) C[(ulong(part) * M + m) * N + n] = Cs[e];
    }
}

// Partials -> T (Σ_p C[p] sequentially from 0, MLX gemm_splitk_accum; one partition: T(C)).
//   SCORES = 1: s (H, keys + 1) = [T(sink_h) | T(score) or -inf (selection)] (N = keys);
//   SCORES = 0: out (M, N) row-major. Grid (N, M).
template <typename T, int SCORES>
[[kernel]] void ds4_mma_out(
    const device float* C [[buffer(0)]],
    device T* out [[buffer(1)]],
    const device uint8_t* sel [[buffer(2)]],
    const device float* sinks [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& parts [[buffer(6)]],
    constant uint& wn [[buffer(7)]],
    constant uint& nsel [[buffer(8)]],
    uint2 g [[thread_position_in_grid]]) {
    if (g.x >= N || g.y >= M) return;
    const ulong i = ulong(g.y) * N + g.x;
    float acc = 0.0f;
    for (uint p = 0; p < parts; ++p) acc += C[ulong(p) * M * N + i];
    if (!SCORES) {
        out[i] = static_cast<T>(acc);
        return;
    }
    const int n = int(g.x);
    const bool use = n < int(wn) || nsel == 0 || sel[n - int(wn)] != 0;
    out[ulong(g.y) * (N + 1) + 1 + n] = use ? static_cast<T>(acc) : static_cast<T>(-INFINITY);
    if (n == 0) out[ulong(g.y) * (N + 1)] = static_cast<T>(sinks[g.y]);
}

// softmax(x, precise=True) over rows of A (MLX softmax_single_row, AccT = float, N_READS = 4; TPG threads).
template <typename T>
[[kernel]] void ds4_softmax_row(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant int& axis_size [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
    constexpr int N_READS = 4;
    constexpr int SIMD_SIZE = 32;
    typedef float AccT;
    int lid = _lid;
    threadgroup AccT local_max[SIMD_SIZE];
    threadgroup AccT local_normalizer[SIMD_SIZE];
    AccT ld[N_READS];
    in += gid * size_t(axis_size) + lid * N_READS;
    if (lid * N_READS + N_READS <= axis_size) {
        for (int i = 0; i < N_READS; i++) ld[i] = AccT(in[i]);
    } else {
        for (int i = 0; i < N_READS; i++) ld[i] = ((lid * N_READS + i) < axis_size) ? AccT(in[i]) : -INFINITY;
    }
    if (simd_group_id == 0) {
        local_max[simd_lane_id] = -INFINITY;
        local_normalizer[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    AccT maxval = -FLT_MAX;
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
    AccT normalizer = 0;
    for (int i = 0; i < N_READS; i++) {
        AccT exp_x = fast::exp(ld[i] - maxval);
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
    out += gid * size_t(axis_size) + lid * N_READS;
    if (lid * N_READS + N_READS <= axis_size) {
        for (int i = 0; i < N_READS; i++) out[i] = T(ld[i] * normalizer);
    } else {
        for (int i = 0; i < N_READS; i++)
            if ((lid * N_READS + i) < axis_size) out[i] = T(ld[i] * normalizer);
    }
}

// MLX softmax_looped (rows longer than 4096; AccT = float, N_READS = 4).
template <typename T>
[[kernel]] void ds4_softmax_looped(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant int& axis_size [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
    constexpr int N_READS = 4;
    constexpr int SIMD_SIZE = 32;
    typedef float AccT;
    in += gid * size_t(axis_size);
    threadgroup AccT local_max[SIMD_SIZE];
    threadgroup AccT local_normalizer[SIMD_SIZE];
    AccT prevmax;
    AccT maxval = -FLT_MAX;
    AccT normalizer = 0;
    for (int r = 0; r < static_cast<int>(((axis_size + int(N_READS * lsize) - 1) / int(N_READS * lsize))); r++) {
        int offset = r * lsize * N_READS + lid * N_READS;
        AccT vals[N_READS];
        if (offset + N_READS <= axis_size) {
            for (int i = 0; i < N_READS; i++) vals[i] = AccT(in[offset + i]);
        } else {
            for (int i = 0; i < N_READS; i++) vals[i] = (offset + i < axis_size) ? AccT(in[offset + i]) : -INFINITY;
        }
        prevmax = maxval;
        for (int i = 0; i < N_READS; i++) maxval = (maxval < vals[i]) ? vals[i] : maxval;
        normalizer *= fast::exp(prevmax - maxval);
        for (int i = 0; i < N_READS; i++) normalizer += fast::exp(vals[i] - maxval);
    }
    prevmax = maxval;
    maxval = simd_max(maxval);
    normalizer *= fast::exp(prevmax - maxval);
    normalizer = simd_sum(normalizer);
    prevmax = maxval;
    if (simd_lane_id == 0) local_max[simd_group_id] = maxval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = simd_max(local_max[simd_lane_id]);
    normalizer *= fast::exp(prevmax - maxval);
    if (simd_lane_id == 0) local_normalizer[simd_group_id] = normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = simd_sum(local_normalizer[simd_lane_id]);
    normalizer = 1 / normalizer;
    out += gid * size_t(axis_size);
    for (int r = 0; r < static_cast<int>(((axis_size + int(N_READS * lsize) - 1) / int(N_READS * lsize))); r++) {
        int offset = r * lsize * N_READS + lid * N_READS;
        if (offset + N_READS <= axis_size) {
            for (int i = 0; i < N_READS; i++) out[offset + i] = T(fast::exp(in[offset + i] - maxval) * normalizer);
        } else {
            for (int i = 0; i < N_READS; i++)
                if (offset + i < axis_size) out[offset + i] = T(fast::exp(in[offset + i] - maxval) * normalizer);
        }
    }
}

// out[h, :] = p[h, 1:] @ V (MLX GEMVTKernel with BM = 1 (BM > 1 not used), SM, SN, BN, TM, TN; float accumulation,
// one rounding). V rows = the keys (window ring + entries). Grid (ceil(D / blockN) threadgroups of 32 · BN, H).
template <typename T, int D, int SM, int SN, int BN, int TM, int TN>
[[kernel]] void ds4_fb_pv(
    const device T* p [[buffer(0)]],
    const device T* win [[buffer(1)]],
    const device T* ent [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant uint& wn [[buffer(4)]],
    constant uint& ws [[buffer(5)]],
    constant uint& wsz [[buffer(6)]],
    constant uint& nent [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int BM = 1;
    constexpr int threadsM = BM * SM;
    constexpr int threadsN = BN * SN;
    constexpr int blockM = threadsM * TM;
    constexpr int blockN = threadsN * TN;
    const int in_vec_size = int(wn + nent);
    const int out_vec_size = D;
    const int h = tid.y;
    const device T* in_vec = p + ulong(h) * (in_vec_size + 1) + 1;
    auto vrow = [&](int i) -> const device T* {
        return i < int(wn) ? win + ulong((ws + uint(i)) % wsz) * D : ent + ulong(i - int(wn)) * D;
    };
    float result[TN] = {0};
    T inter[TN];
    float v_coeff[TM];
    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);
    const int sgM = BN != 1 ? (simd_gid / BN) : int(simd_gid);
    const int sgN = BN != 1 ? (simd_gid % BN) : 0;
    const int simdM = SM * sgM;
    const int simdN = SN * sgN;
    int cm = (simdM + thrM);
    int cn = (simdN + thrN);
    int bm = cm * TM;
    int bn = cn * TN;
    int out_col = tid.x * blockN + bn;
    const int n_iter = in_vec_size / blockM;
    const int last_iter = blockM * n_iter;
    const int leftover = in_vec_size - last_iter;
    if (out_col < out_vec_size) {
        out_col = out_col + TN < out_vec_size ? out_col : out_vec_size - TN;
        for (int i = 0; i < n_iter; ++i) {
            threadgroup_barrier(mem_flags::mem_none);
            for (int tm = 0; tm < TM; tm++) v_coeff[tm] = static_cast<float>(in_vec[bm + tm]);
            for (int tm = 0; tm < TM; tm++) {
                auto vc = static_cast<float>(v_coeff[tm]);
                const device T* vr = vrow(bm + tm);
                for (int tn = 0; tn < TN; tn++) inter[tn] = vr[out_col + tn];
                for (int tn = 0; tn < TN; tn++) result[tn] += vc * inter[tn];
            }
            bm += blockM;
        }
        if (leftover > 0) {
            for (int tm = 0; tm < TM && bm + tm < in_vec_size; tm++) {
                v_coeff[tm] = static_cast<float>(in_vec[bm + tm]);
                const device T* vr = vrow(bm + tm);
                for (int tn = 0; tn < TN; tn++) inter[tn] = vr[out_col + tn];
                for (int tn = 0; tn < TN; tn++) result[tn] += v_coeff[tm] * inter[tn];
            }
        }
    }
    for (int tn = 0; tn < TN; tn++)
        for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) result[tn] += simd_shuffle_down(result[tn], SN * sm);
    if (cm == 0 && out_col < out_vec_size)
        for (int j = 0; j < TN; j++) out[ulong(h) * D + out_col + j] = static_cast<T>(result[j]);
}

// wo_a through a SUC module (every row group on every input group): z[g, r] = y[g, g · R + r]. Grid (R, G).
template <typename T>
[[kernel]] void ds4_diag(
    const device T* y [[buffer(0)]],
    device T* z [[buffer(1)]],
    constant uint& G [[buffer(2)]],
    constant uint& R [[buffer(3)]],
    uint2 g [[thread_position_in_grid]]) {
    if (g.x >= R) return;
    z[g.y * R + g.x] = y[ulong(g.y) * G * R + g.y * R + g.x];
}

// ================================================================================================ MoE
// Parallel top-K of b[0..E) (threadgroup memory): K rounds of a threadgroup argmax over the untaken entries, order
// "larger value first, ties by the lower index" (simd shuffles, then the simdgroup winners in index order). For non-NaN
// values this selects exactly what the sequential scan `if (best < 0 || b > bv)` over e = 0..E-1 selects in each round
// (that scan keeps the first maximum), so the result is bit-identical to it; it replaced a one-thread scan that cost
// ~0.3 ms per MoE layer on an M1 Ultra (6 x 256 dependent device loads). Every thread of the threadgroup must call it.
inline bool ds4_topk_better(float av, int ai, float bv, int bi) {
    if (ai < 0) return false;
    if (bi < 0) return true;
    if (av > bv) return true;
    if (bv > av) return false;
    return ai < bi;
}
template <int E, int K>
inline void ds4_topk_parallel(threadgroup const float* b, threadgroup bool* taken, threadgroup float* wv,
                              threadgroup int* wi, threadgroup uint* sel, uint lid, uint sl, uint sg, uint nsg) {
    if (lid < uint(E)) taken[lid] = false;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < K; ++j) {
        const bool ok = lid < uint(E) && !taken[lid];
        float v = ok ? b[lid] : 0.0f;
        int i = ok ? int(lid) : -1;
        for (ushort off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(v, off);
            const int oi = simd_shuffle_down(i, off);
            if (sl + off < 32 && ds4_topk_better(ov, oi, v, i)) {
                v = ov;
                i = oi;
            }
        }
        if (sl == 0) {
            wv[sg] = v;
            wi[sg] = i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float bv = wv[0];
            int bi = wi[0];
            for (uint g = 1; g < nsg; ++g)
                if (ds4_topk_better(wv[g], wi[g], bv, bi)) {
                    bv = wv[g];
                    bi = wi[g];
                }
            sel[j] = uint(bi);
            taken[bi] = true;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Gate: scores from the float32 router logits (FUNC 0: sqrt(softplus), 1: sigmoid; one thread per expert), experts
// from tid2eid[token] (HASH) or the top K of scores + bias (argpartition order: descending, ties by lower index;
// ds4_topk_parallel), weights = the chosen scores, / their sum (sequential), · route_scale. eidx (K) uint32, w (K) float32.
// FUNC 2 (softmax scoring) runs in thread 0 (fast::exp, sequential sum: not MLX's softmax order).
// One threadgroup of TPG >= E threads (TPG <= 1024, a multiple of 32).
template <int E, int K, int FUNC, int HASH, typename TI, typename TB>
[[kernel]] void ds4_route(
    const device float* logits [[buffer(0)]],
    const device TB* bias [[buffer(1)]],
    const device TI* tid2eid [[buffer(2)]],
    const device uint32_t* tok [[buffer(3)]],
    device uint32_t* eidx [[buffer(4)]],
    device float* w [[buffer(5)]],
    constant float& route_scale [[buffer(6)]],
    constant uint& tok_stride [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]) {
#pragma clang fp contract(off)
    // row-batched launches (grid = rows): logits row r (E), tok[r · tok_stride], eidx / w rows r (K)
    logits += row * E;
    tok += row * tok_stride;
    eidx += row * K;
    w += row * K;
    threadgroup float s[E];
    threadgroup float bs[E];
    threadgroup bool taken[E];
    threadgroup float wvs[32];
    threadgroup int wis[32];
    threadgroup uint sel[K];
    if (FUNC != 2 && lid < uint(E))
        s[lid] = FUNC == 1 ? ds4_sigmoid(logits[lid]) : metal::precise::sqrt(ds4_logaddexp0(logits[lid]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (FUNC == 2) {
        if (lid == 0) {
            float mx = -INFINITY;
            for (int e = 0; e < E; ++e) mx = metal::max(mx, logits[e]);
            float z = 0.0f;
            for (int e = 0; e < E; ++e) {
                s[e] = fast::exp(logits[e] - mx);
                z += s[e];
            }
            const float r = 1.0f / z;
            for (int e = 0; e < E; ++e) s[e] = s[e] * r;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint idx[K];
    if (HASH) {
        if (lid != 0) return;
        const ulong t = tok[0];
        for (int j = 0; j < K; ++j) idx[j] = uint(tid2eid[t * K + j]);
    } else {
        if (lid < uint(E)) bs[lid] = s[lid] + float(bias[lid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ds4_topk_parallel<E, K>(bs, taken, wvs, wis, sel, lid, sl, sg, nsg);
        if (lid != 0) return;
        for (int j = 0; j < K; ++j) idx[j] = sel[j];
    }
    float wv[K];
    for (int j = 0; j < K; ++j) wv[j] = s[idx[j]];
    if (FUNC != 2) {
        float t = 0.0f;
        for (int j = 0; j < K; ++j) t = wv[j] + t;
        for (int j = 0; j < K; ++j) wv[j] = wv[j] / t;
    }
    for (int j = 0; j < K; ++j) {
        eidx[j] = idx[j];
        w[j] = wv[j] * route_scale;
    }
}

// h[n, i] = T(silu(min(g, l)) · clip(u, -l, l) [· w[n]]) (float32, each product rounded; LIM: l > 0). Grid (F, rows).
template <typename T, int WEIGHT, int LIM>
[[kernel]] void ds4_swiglu(
    const device T* g [[buffer(0)]],
    const device T* u [[buffer(1)]],
    const device float* w [[buffer(2)]],
    device T* h [[buffer(3)]],
    constant uint& F [[buffer(4)]],
    constant float& limit [[buffer(5)]],
    uint2 t [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (t.x >= F) return;
    const ulong i = ulong(t.y) * F + t.x;
    float gv = float(g[i]), uv = float(u[i]);
    if (LIM) {
        uv = metal::clamp(uv, -limit, limit);
        gv = metal::min(gv, limit);
    }
    const float silu = gv * ds4_sigmoid(gv);
    float v = silu * uv;
    if (WEIGHT) v = v * w[t.y];
    h[i] = static_cast<T>(v);
}

// MoE output: y[d] = T(Σ_k float(yk[k, d]) + float(sh[d])) (column sum over the K expert rows in MLX's order).
// Grid (D).
template <typename T, int K>
[[kernel]] void ds4_moe_out(
    const device T* yk [[buffer(0)]],
    const device T* sh [[buffer(1)]],
    device T* y [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    const uint d = gid.x;
    if (d >= D) return;
    // row-batched launches (grid y = rows): yk row r (K · D), sh / y rows r (D)
    yk += ulong(gid.y) * K * D;
    sh += ulong(gid.y) * D;
    y += ulong(gid.y) * D;
    float t;
    if (K <= 16) {
        t = ds4_colsum_small([&](int k) { return float(yk[ulong(k) * D + d]); }, K);
    } else {
        // more than 16 experts per token: 32 interleaved partials would need a simdgroup; sequential partials here
        float part[32];
        for (int l = 0; l < 32; ++l) part[l] = 0.0f;
        for (int l = 0; l < 32; ++l)
            for (int k = l; k < K; k += 32) part[l] = float(yk[ulong(k) * D + d]) + part[l];
        t = 0.0f;
        for (int l = 0; l < 32; ++l) t = part[l] + t;
    }
    y[d] = static_cast<T>(t + float(sh[d]));
}

// Token-id broadcast: tok[i] = id for the embedding gather of hc identical streams (i < n).
[[kernel]] void ds4_fill_u32(
    device uint32_t* y [[buffer(0)]],
    constant uint& v [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    y[i] = v;
}

// ================================================================================================ mixed expert stacks
// ds4_switch_colres_masked = suc_switch_structured_residual (generated from suc.kernels.suc_switch._COLRES_SRC, body
// unchanged) plus the sub-stack's offset table: rows whose expert is an empty slot of this sub-stack (routed to another
// sub-stack of a mixed stack) return at once, so every sub-stack writes its own rows straight into the stack output and
// the select launch disappears. The rows it writes get exactly the bits of the unmasked kernel.
template <typename TX, typename TCC, typename TCV, typename TRC, typename TRV, int IN, int OUT, int CMAX, int RQ, int JS,
          int RED, bool HAS_CSR, typename OT>
[[kernel]] void ds4_switch_colres_masked(
    const device TX* x [[buffer(0)]],
    const device float* y_in [[buffer(1)]],
    const device uint32_t* eidx [[buffer(2)]],
    const device uint32_t* cptr [[buffer(3)]],
    const device TCC* ccols [[buffer(4)]],
    const device TCV* cval [[buffer(5)]],
    const device float* cscale [[buffer(6)]],
    const device uint32_t* row_ptr [[buffer(7)]],
    const device TRC* rcols [[buffer(8)]],
    const device TRV* rvals [[buffer(9)]],
    const device float* rscale [[buffer(10)]],
    device OT* y [[buffer(11)]],
    const device ulong* eoff [[buffer(12)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {

    constexpr uint TR = 32 * RQ;
    uint tid = thread_position_in_threadgroup.x;
    uint tn = threads_per_threadgroup.x;
    uint lane = tid % 32, js = tid / 32;
    uint b = threadgroup_position_in_grid.y;
    uint e = eidx[b];
    if (eoff[3 * ulong(e) + 1] == ~0ul) return;   // row routed to another sub-stack: leave y untouched
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
}
