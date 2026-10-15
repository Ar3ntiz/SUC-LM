// glm5_next (GLM-5.3-Flash) model-graph kernels of the C++ runtime (cpp/src/metal/glm5.cpp). Reference semantics:
// suc/models/mlx_glm5_next.py on the Python SUC1 runtime (suc.runtime.suc1_model.load_suc1_model), one decode row.
// Appended after cpp/metal/suc_ds4_kernels.metal at build time; reuses its helpers (ds4_sigmoid, ds4_colsum_small)
// and the qwen4 helper q4_sigmoid (MLX sigmoid in T arithmetic).
//
// Operation order: the MLX orders measured for milestone 6 (see the head of suc_ds4_kernels.metal): every element-wise
// op rounds (kernels chaining such ops disable fma contraction), precise exp / sqrt / rsqrt, row sums of <= 64 values
// sequential in one thread (MLX row_reduce_small), longer rows in MLX's row_reduce_simple order (TPG threads, 4
// contiguous values per thread per step, simd_sum), column sums of <= 16 rows as 8 interleaved partials
// (ds4_colsum_small). Kernels ported from MLX (gemv_t, sdpa_vector, layer_norm, the gated-delta step of mlx-lm) keep
// MLX's expressions unchanged (MLX / mlx-lm, Copyright © 2023-2025 Apple Inc., MIT License).
//
//   glm_hc_scale      xs = xf · rsqrt(mean(xf²) + eps)          (mHC: the mixer input is normalized BEFORE fn)
//   glm_hc_coef       pre / post / Sinkhorn comb from the 24 mixer logits (one thread)
//   glm_hc_post       x_j = T(T(T(post_j) · f) + T(Σ_i T(comb_ij) res_i))   (TF:1337, in the model dtype)
//   glm_kda_conv      SiLU(causal depthwise conv4) in float32 over the rolling bf16 state, rounded to T
//   glm_l2norm        q / k: x / sqrt(Σ x² + 1e-6) (float32; q times Dk^-0.5)
//   glm_kda_gate      decay = exp(-5 σ(exp(A_log) (f + dt_bias))) per channel, beta = σ_T(b)
//   glm_gated_delta   mlx-lm gated_delta kernel, vectorized gate (g: (Hv, Dk)), float inputs, state in place
//   glm_sig_mul       y = T(n · σ(float(gate)))
//   glm_layer_norm    mx.fast.layer_norm (MLX layer_norm_single_row), weight + bias in T
//   glm_route         sigmoid router: top-k of scores + bias (descending, ties by index), / (Σ + 1e-20) · scale
//   glm_moe_out       T(T(Σ_k float(y_k) · w_k) + shared)
//   glm_stream_mean   head: T(Σ_i x_i · (1 / hc)) (float32 column sum)
//   glm_gemv_t        MLX gemv_t (x @ W, W row-major (K, N)), batched (per-head matrices)
//   glm_gemv_b        MLX gemv (x @ Wᵀ), batched (per-head matrices)
//   glm_fb_scores     MLX's unfused attention fallback for one key: scores T(T(T(scale) q) · k) (gemv order)
//   glm_sdpa_vector   MLX sdpa_vector (one query row per head, K = V = the cached latents, bool mask = selection)
//   glm_mma / glm_mma_out / glm_index_reduce   MLX's steel GEMM order (fused / split-k) for the folded fallback
//                     attention matmuls and the indexer pool scores Σ_h w_h · relu(q_h · K_p · d^-0.5)

// ---------------------------------------------------------------------------------------------- mHC
template <int W>
[[kernel]] void glm_hc_scale(
    const device float* xf [[buffer(0)]],
    const device float* ss [[buffer(1)]],
    device float* xs [[buffer(2)]],
    constant float& inv_w [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (i >= uint(W)) return;
    const float mean = ss[0] * inv_w;
    const float rs = metal::precise::rsqrt(mean + eps);
    xs[i] = xf[i] * rs;
}

// coef: pre (HC) | post (HC) | comb (HC, HC), float32 (Glm5NextTextHyperConnection.forward on the mixer logits m).
template <typename TP, int HC, int ITERS>
[[kernel]] void glm_hc_coef(
    const device float* raw [[buffer(0)]],
    const device TP* base [[buffer(1)]],
    const device TP* scale [[buffer(2)]],
    device float* coef [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (tid != 0) return;
    const float s0 = float(scale[0]), s1 = float(scale[1]), s2 = float(scale[2]);
    for (int i = 0; i < HC; ++i) {
        const float a = raw[i] * s0;
        coef[i] = ds4_sigmoid(a + float(base[i])) + eps;
        const float a2 = raw[HC + i] * s1;
        coef[HC + i] = 2.0f * ds4_sigmoid(a2 + float(base[HC + i]));
    }
    float c[HC][HC];
    for (int i = 0; i < HC; ++i) {
        for (int j = 0; j < HC; ++j) {
            const int k = 2 * HC + i * HC + j;
            const float a = raw[k] * s2;
            c[i][j] = a + float(base[k]);
        }
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

// glm_hc_coef on one simdgroup (threadgroup of 32): bit for bit the same coefficients, the Sinkhorn normalisations
// element-parallel (ds4_sinkhorn_lane), pre / post on lanes 0 .. 2·HC - 1. HC · HC <= 32.
template <typename TP, int HC, int ITERS>
[[kernel]] void glm_hc_coef_par(
    const device float* raw [[buffer(0)]],
    const device TP* base [[buffer(1)]],
    const device TP* scale [[buffer(2)]],
    device float* coef [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    static_assert(HC * HC <= 32, "one simdgroup holds the comb matrix");
    if (lane < uint(HC)) {
        const float a = raw[lane] * float(scale[0]);
        coef[lane] = ds4_sigmoid(a + float(base[lane])) + eps;
    } else if (lane < uint(2 * HC)) {
        const float a2 = raw[lane] * float(scale[1]);
        coef[lane] = 2.0f * ds4_sigmoid(a2 + float(base[lane]));
    }
    const uint l = lane < uint(HC * HC) ? lane : 0u;
    const int k = 2 * HC + int(l);
    const float a = raw[k] * float(scale[2]);
    const float c = ds4_sinkhorn_lane<HC, ITERS>(a + float(base[k]), l, eps);
    if (lane < uint(HC * HC)) coef[2 * HC + lane] = c;
}

// ds4_hc_collapse with glm_hc_coef's pre coefficients computed in place (so that it does not wait for the coefficient
// kernel): y[d] = T(Σ_i pre_i · float(x[i, d])). Grid (D).
template <typename T, typename TP, int D, int HC>
[[kernel]] void glm_hc_collapse_pre(
    const device T* x [[buffer(0)]],
    const device float* raw [[buffer(1)]],
    const device TP* base [[buffer(2)]],
    const device TP* scale [[buffer(3)]],
    device T* y [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint d [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (d >= uint(D)) return;
    const float s0 = float(scale[0]);
    float acc = 0.0f;
    for (int i = 0; i < HC; ++i) {
        const float a = raw[i] * s0;
        const float pre = ds4_sigmoid(a + float(base[i])) + eps;
        const float p = pre * float(x[i * D + d]);
        acc = p + acc;
    }
    y[d] = static_cast<T>(acc);
}

// y[j, d] = T(T(T(post_j) · f[d]) + T(Σ_i T(comb[i, j]) · res[i, d])) — the bf16 matmul as a float32 fma chain over i
// (MLX steel gemm), one rounding. Grid (D, HC).
template <typename T, int D, int HC>
[[kernel]] void glm_hc_post(
    const device T* f [[buffer(0)]],
    const device T* res [[buffer(1)]],
    const device float* coef [[buffer(2)]],
    device T* y [[buffer(3)]],
    uint2 g [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    const uint d = g.x, j = g.y;
    if (d >= uint(D)) return;
    float acc = 0.0f;
    for (int i = 0; i < HC; ++i)
        acc = metal::fma(float(static_cast<T>(coef[2 * HC + i * HC + j])), float(res[i * D + d]), acc);
    const T mm = static_cast<T>(acc);
    const float pj = float(static_cast<T>(coef[HC + j]));
    const T p = static_cast<T>(pj * float(f[d]));
    y[j * D + d] = static_cast<T>(float(p) + float(mm));
}

// head_stream: y[d] = T(Σ_i float(x[i, d]) · inv_hc) (column sum over the HC streams, MLX order). Grid (D).
template <typename T, int D, int HC>
[[kernel]] void glm_stream_mean(
    const device T* x [[buffer(0)]],
    device T* y [[buffer(1)]],
    constant float& inv_hc [[buffer(2)]],
    uint2 g [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    const uint d = g.x;
    if (d >= uint(D)) return;
    x += ulong(g.y) * HC * D;
    y += ulong(g.y) * D;
    const float s = ds4_colsum_small([&](int i) { return float(x[i * D + d]); }, HC);
    y[d] = static_cast<T>(s * inv_hc);
}

// ---------------------------------------------------------------------------------------------- KDA
// Channel c of C = 3 · H · Dk: conv_in = [state rows 0 .. K-2, xnew]; acc = ((t_0 + t_1) + t_2) + ..., t_i =
// float(conv_in[i, c]) · w[c, i] (float32); y[c] = T(acc · σ(acc)); the state shifts by one row. Grid (C).
template <typename T, int C, int K>
[[kernel]] void glm_kda_conv(
    const device T* xnew [[buffer(0)]],
    device T* state [[buffer(1)]],
    const device float* w [[buffer(2)]],
    device T* y [[buffer(3)]],
    uint c [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (c >= uint(C)) return;
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        const float v = i < K - 1 ? float(state[ulong(i) * C + c]) : float(xnew[c]);
        const float t = v * w[ulong(c) * K + i];
        acc = i == 0 ? t : acc + t;
    }
    const float s = acc * ds4_sigmoid(acc);
    y[c] = static_cast<T>(s);
    for (int i = 0; i + 1 < K - 1; ++i) state[ulong(i) * C + c] = state[ulong(i + 1) * C + c];
    state[ulong(K - 2) * C + c] = xnew[c];
}

// Row r < 2H of the conv output x (q heads, then k heads; Dk = N values each, T): out = x / sqrt(Σ x² + 1e-6)
// (float32), q rows also · qscale. Σ x²: N <= 64 sequential in one thread (row_reduce_small), else MLX's
// row_reduce_simple (TPG threads). One threadgroup of TPG per row; out (2H, N) float32.
template <typename T, int N, int TPG>
[[kernel]] void glm_l2norm(
    const device T* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& qrows [[buffer(2)]],
    constant float& qscale [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    threadgroup float sh[33];
    const device T* xr = x + ulong(row) * N;
    float total = 0.0f;
    if (N <= 64) {
        if (lid == 0) {
            for (int j = 0; j < N; ++j) {
                const float v = float(xr[j]);
                const float sq = v * v;
                total = sq + total;
            }
            sh[32] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = sh[32];
    } else {
        const device T* in = xr + lid * 4;
        constexpr int blocks = N / (TPG * 4);
        constexpr int extra = N - blocks * TPG * 4;
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
            if (lid == 0) sh[32] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            total = sh[32];
        }
    }
    const float den = metal::precise::sqrt(total + 1e-6f);
    for (uint j = lid; j < uint(N); j += TPG) {
        float v = float(xr[j]) / den;
        if (row < qrows) v = v * qscale;
        out[ulong(row) * N + j] = v;
    }
}

// Channel c < H · Dk: decay[c] = exp(lb · σ(exp(A_log[h]) · (float(f[c]) + dt_bias[c]))), h = c / Dk;
// c < H also: beta[c] = float(σ_T(b[c])). Grid (H · Dk).
template <typename T, typename TA, typename TB, int DK>
[[kernel]] void glm_kda_gate(
    const device T* f [[buffer(0)]],
    const device TB* dt_bias [[buffer(1)]],
    const device TA* a_log [[buffer(2)]],
    const device T* b [[buffer(3)]],
    device float* decay [[buffer(4)]],
    device float* beta [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant float& lb [[buffer(8)]],
    uint c [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (c >= n) return;
    const uint h = c / uint(DK);
    const float fv = float(f[c]) + float(dt_bias[c]);
    const float ea = metal::precise::exp(float(a_log[h]));
    const float a = ea * fv;
    const float s = ds4_sigmoid(a);
    const float gl = lb * s;
    decay[c] = metal::precise::exp(gl);
    if (c < heads) beta[c] = float(q4_sigmoid(b[c]));
}

// mlx_lm.models.gated_delta._make_gated_delta_kernel(has_mask=False, vectorized=True) for T = 1 (mlx-lm, MIT License),
// InT = float32: q, k (Hk, Dk), v (Hv, Dv), g (Hv, Dk), beta (Hv) float; state (Hv, Dv, Dk) float32 in place;
// y (Hv, Dv) float = float(TO(out)) (the Python model casts the output to the model dtype and back).
// Grid (32, Dv, Hv), threadgroup (32, 4, 1).
template <typename TO, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void glm_gated_delta(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    const device float* g [[buffer(3)]],
    const device float* beta [[buffer(4)]],
    device float* st [[buffer(5)]],
    device float* y [[buffer(6)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
    auto n = thread_position_in_grid.z;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    auto q_ = q + hk_idx * Dk;
    auto k_ = k + hk_idx * Dk;
    auto v_ = v + hv_idx * Dv;
    y += hv_idx * Dv;
    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    auto i_state = st + (n * Dv + dv_idx) * Dk;
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = static_cast<float>(i_state[s_idx]);
    }
    auto g_ = g + hv_idx * Dk;
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[s_idx];
        kv_mem += state[i] * k_[s_idx];
    }
    kv_mem = simd_sum(kv_mem);
    auto delta = (v_[dv_idx] - kv_mem) * beta[hv_idx];
    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) y[dv_idx] = float(static_cast<TO>(static_cast<float>(out)));
    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        i_state[s_idx] = state[i];
    }
}

// y[i] = T(x[i] · σ(float(gate[i]))) (float32 x). Grid (n).
template <typename T>
[[kernel]] void glm_sig_mul(
    const device float* x [[buffer(0)]],
    const device T* gate [[buffer(1)]],
    device T* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (i >= n) return;
    const float s = ds4_sigmoid(float(gate[i]));
    y[i] = static_cast<T>(x[i] * s);
}

// y = T(x) as float (bf16 rows -> float rows of the k-pool buffers). Grid (n).
template <typename T>
[[kernel]] void glm_to_f32(
    const device T* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = float(x[i]);
}

// ---------------------------------------------------------------------------------------------- DSA indexer
// mx.fast.layer_norm(x, w, b, eps) for one row of N <= 4096 (MLX layer_norm_single_row, N_READS = 4, TPG threads):
// mean and variance in float32 (per-thread sequential, simd_sum), out = w · T(x̂) + b in T.
template <typename T, int N, int TPG>
[[kernel]] void glm_layer_norm(
    const device T* x [[buffer(0)]],
    const device T* w [[buffer(1)]],
    const device T* b [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
    constexpr int N_READS = 4;
    constexpr uint ns = (TPG + 31) / 32;
    float thread_x[N_READS] = {0};
    threadgroup float local_sumx[32];
    x += ulong(row) * N;
    out += ulong(row) * N;
    x += lid * N_READS;
    w += lid * N_READS;
    b += lid * N_READS;
    out += lid * N_READS;
    const bool safe = int(lid) * N_READS + N_READS <= N;
    const int n = N - int(lid) * N_READS;
    if (safe) {
        for (int i = 0; i < N_READS; i++) thread_x[i] = x[i];
    } else {
        for (int i = 0; i < n; i++) thread_x[i] = x[i];
    }
    float mean = 0;
    for (int i = 0; i < N_READS; i++) mean += thread_x[i];
    mean = simd_sum(mean);
    if (ns > 1) {
        if (simd_lane_id == 0) local_sumx[simd_group_id] = mean;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mean = simd_sum(simd_lane_id < ns ? local_sumx[simd_lane_id] : 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mean /= N;
    float normalizer = 0;
    if (!safe) {
        for (int i = max(n, 0); i < N_READS; i++) thread_x[i] = mean;
    }
    for (int i = 0; i < N_READS; i++) {
        thread_x[i] -= mean;
        normalizer += thread_x[i] * thread_x[i];
    }
    normalizer = simd_sum(normalizer);
    if (ns > 1) {
        if (simd_lane_id == 0) local_sumx[simd_group_id] = normalizer;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        normalizer = simd_sum(simd_lane_id < ns ? local_sumx[simd_lane_id] : 0.0f);
    }
    normalizer = metal::precise::rsqrt(normalizer / N + eps);
    if (safe) {
        for (int i = 0; i < N_READS; i++) {
            thread_x[i] *= normalizer;
            out[i] = w[i] * static_cast<T>(thread_x[i]) + b[i];
        }
    } else {
        for (int i = 0; i < n; i++) {
            thread_x[i] *= normalizer;
            out[i] = w[i] * static_cast<T>(thread_x[i]) + b[i];
        }
    }
}

// ---------------------------------------------------------------------------------------------- MoE
// Router (Glm5NextTextTopkRouter): s = σ(logits) (float32), experts = top K of s + bias (descending, ties by lower
// index: MLX's argpartition is a stable sort on the GPU), w = s[idx] / (Σ s[idx] + 1e-20) · scale (sequential sum).
// One threadgroup of TPG >= E threads.
template <int E, int K, typename TB>
[[kernel]] void glm_route(
    const device float* logits [[buffer(0)]],
    const device TB* bias [[buffer(1)]],
    device uint32_t* eidx [[buffer(2)]],
    device float* w [[buffer(3)]],
    constant float& route_scale [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]) {
#pragma clang fp contract(off)
    threadgroup float s[E];
    threadgroup float bs[E];
    threadgroup bool taken[E];
    threadgroup float wvs[32];
    threadgroup int wis[32];
    threadgroup uint sel[K];
    logits += ulong(row) * E;
    eidx += ulong(row) * K;
    w += ulong(row) * K;
    if (lid < uint(E)) {
        s[lid] = ds4_sigmoid(logits[lid]);
        bs[lid] = s[lid] + float(bias[lid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ds4_topk_parallel<E, K>(bs, taken, wvs, wis, sel, lid, sl, sg, nsg);   // = the sequential first-maximum scan
    if (lid != 0) return;
    uint idx[K];
    for (int j = 0; j < K; ++j) idx[j] = sel[j];
    float wv[K];
    for (int j = 0; j < K; ++j) wv[j] = s[idx[j]];
    float t = 0.0f;
    for (int j = 0; j < K; ++j) t = wv[j] + t;
    const float den = t + 1e-20f;
    for (int j = 0; j < K; ++j) {
        eidx[j] = idx[j];
        const float q = wv[j] / den;
        w[j] = q * route_scale;
    }
}

// y[d] = T(float(T(Σ_k float(yk[k, d]) · w[k])) + float(sh[d])) (products rounded, column sum over K in MLX's order).
// Grid (D).
template <typename T, int K>
[[kernel]] void glm_moe_out(
    const device T* yk [[buffer(0)]],
    const device float* w [[buffer(1)]],
    const device T* sh [[buffer(2)]],
    device T* y [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    uint2 g [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    const uint d = g.x;
    if (d >= D) return;
    yk += ulong(g.y) * K * D;
    w += ulong(g.y) * K;
    sh += ulong(g.y) * D;
    y += ulong(g.y) * D;
    auto term = [&](int k) {
        const float p = float(yk[ulong(k) * D + d]) * w[k];
        return p;
    };
    float t;
    if (K <= 16) {
        t = ds4_colsum_small(term, K);
    } else {
        float part[32];
        for (int l = 0; l < 32; ++l) part[l] = 0.0f;
        for (int l = 0; l < 32; ++l)
            for (int k = l; k < K; k += 32) part[l] = term(k) + part[l];
        t = 0.0f;
        for (int l = 0; l < 32; ++l) t = part[l] + t;
    }
    const T r = static_cast<T>(t);
    y[d] = static_cast<T>(float(r) + float(sh[d]));
}

// ---------------------------------------------------------------------------------------------- MLA (absorbed)
// y[h, :] = x[h, 0:K] @ W_h (W_h = w + h · w_hs elements, row-major (K, N)): MLX GEMVTKernel (BM = 1, SM, SN, BN, TM,
// TN; float accumulation, one rounding). x row h at x + h · x_hs. Grid (ceil(N / blockN) threadgroups of 32 · BN, H).
template <typename T, typename TW, int N, int SM, int SN, int BN, int TM, int TN>
[[kernel]] void glm_gemv_t(
    const device T* x [[buffer(0)]],
    const device TW* w [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& x_hs [[buffer(4)]],
    constant ulong& w_hs [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int BM = 1;
    constexpr int threadsM = BM * SM;
    constexpr int threadsN = BN * SN;
    constexpr int blockM = threadsM * TM;
    constexpr int blockN = threadsN * TN;
    const int in_vec_size = int(K);
    const int out_vec_size = N;
    const int h = tid.y;
    const device T* in_vec = x + ulong(h) * x_hs;
    const device TW* mat = w + ulong(h) * w_hs;
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
                const device TW* vr = mat + ulong(bm + tm) * N;
                for (int tn = 0; tn < TN; tn++) inter[tn] = static_cast<T>(vr[out_col + tn]);
                for (int tn = 0; tn < TN; tn++) result[tn] += vc * inter[tn];
            }
            bm += blockM;
        }
        if (leftover > 0) {
            for (int tm = 0; tm < TM && bm + tm < in_vec_size; tm++) {
                v_coeff[tm] = static_cast<float>(in_vec[bm + tm]);
                const device TW* vr = mat + ulong(bm + tm) * N;
                for (int tn = 0; tn < TN; tn++) inter[tn] = static_cast<T>(vr[out_col + tn]);
                for (int tn = 0; tn < TN; tn++) result[tn] += v_coeff[tm] * inter[tn];
            }
        }
    }
    for (int tn = 0; tn < TN; tn++)
        for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) result[tn] += simd_shuffle_down(result[tn], SN * sm);
    if (cm == 0 && out_col < out_vec_size)
        for (int j = 0; j < TN; j++) out[ulong(h) * N + out_col + j] = static_cast<T>(result[j]);
}

// y[h, o] = T(Σ_j W_h[o, j] · x[h, j]) with W_h = w + h · w_hs (rows of IN): suc_moe_gemv with per-batch matrices
// (MLX gemv order). Grid (ceil(OUT / SM) threadgroups, H).
template <typename T, typename TW, int IN, int OUT, int SN, int TN, int BN>
[[kernel]] void glm_gemv_b(
    const device T* x [[buffer(0)]],
    const device TW* w [[buffer(1)]],
    device T* y [[buffer(2)]],
    constant ulong& w_hs [[buffer(3)]],
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
    const device TW* wr = w + ulong(tid.y) * w_hs + ulong(live ? row : 0) * IN;
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

// Key i (token i < N) is used when nsel == 0, or i >= ncomp · kpool (the incomplete tail), or sel[i / kpool] != 0.
inline bool glm_use_key(int i, constant uint& nsel, const device uint8_t* sel, uint kpool, uint ncomp) {
    return nsel == 0 || uint(i) >= ncomp * kpool || sel[uint(i) / kpool] != 0;
}

// MLX's unfused attention fallback: s[h, i] = T(T(T(scale) · q_h) · k_i) in gemv order (SN = 32 or 4, TN = 4, BN),
// -inf for unselected keys. q (H, D), keys (N, D), s (H, N) in T. Grid (ceil(N / SM) threadgroups of 32 · BN, H).
template <typename T, int D, int SN, int TN, int BN>
[[kernel]] void glm_fb_scores(
    const device T* q [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device uint8_t* sel [[buffer(2)]],
    device T* s_out [[buffer(3)]],
    constant uint& NN [[buffer(4)]],
    constant uint& nsel [[buffer(5)]],
    constant uint& kpool [[buffer(6)]],
    constant uint& ncomp [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int SM = 32 / SN;
    constexpr int BLK = BN * SN * TN;
    threadgroup float tg[BN * SM];
    const int thrM = simd_lid / SN, thrN = simd_lid % SN;
    const int sgN = simd_gid;
    const int row = int(tid.x) * SM + thrM;
    const bool live = row < int(NN);
    const int h = tid.y;
    const device T* kr = keys + ulong(live ? row : 0) * D;
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
        const bool use = glm_use_key(row, nsel, sel, kpool, ncomp);
        s_out[ulong(h) * NN + row] = use ? static_cast<T>(acc) : static_cast<T>(-INFINITY);
    }
}

// MLX sdpa_vector (single pass, one query row per head, every head on the one latent KV head) with the bool-mask
// branch fed by the k-pool selection. K = V = keys (N, D). Grid (H threadgroups of 1024).
template <typename T, int D>
[[kernel]] void glm_sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device uint8_t* sel [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant uint& NN [[buffer(4)]],
    constant uint& nsel [[buffer(5)]],
    constant uint& kpool [[buffer(6)]],
    constant uint& ncomp [[buffer(7)]],
    constant float& scale [[buffer(8)]],
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
    const int N = int(NN);
    queries += head * D + simd_lid * qk_per_thread;
    out += head * D + simd_gid * v_per_thread;
    for (int i = 0; i < qk_per_thread; i++) q[i] = static_cast<U>(scale) * queries[i];
    for (int i = 0; i < v_per_thread; i++) o[i] = 0;
    U max_score = -FLT_MAX;
    U sum_exp_score = 0;
    for (int i = simd_gid; i < N; i += BN) {
        if (glm_use_key(i, nsel, sel, kpool, ncomp)) {
            const device T* row = keys + ulong(i) * D;
            const device T* kk = row + simd_lid * qk_per_thread;
            const device T* values = row + simd_lid * v_per_thread;
            for (int j = 0; j < qk_per_thread; j++) k[j] = kk[j];
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

// ---------------------------------------------------------------------------------------------- MLX steel GEMM order
// MLX folds a matmul whose B operand is broadcast over the batch into one GEMM (M = batch · rows): the fallback
// attention (H query heads on the one latent KV head: scores = q' Kᵀ, out = P V) and the indexer pool scores run as
// steel GEMMs, not per-head gemvs. Their float32 accumulation order is reproduced here (measured against MLX 0.32.3
// with float32 operands, bit for bit):
//   steel_gemm_fused   one accumulator per output, 8 x 8 x 8 simdgroup MMAs in k order, the unaligned K tail
//                      (K % BK) FIRST, then the aligned BK blocks (steel_gemm_fused.h: "Do unaligned K iterations first");
//   steel_gemm_splitk  when K >= max(M, N) and K / 16 >= 8: P = min(max(2, next_pow2((K / 16) / (ceil(M / 32) ·
//                      ceil(N / 32)))), 32) partitions of ((K / 16) / P) · 16 values (the last one takes the rest,
//                      its tail at the end), each from a zero accumulator, then out = ((0 + C_0) + C_1) + ... .
// glm_mma: one simdgroup per 8 x 8 output tile and partition; C_part (parts, M, N) float32. A (M, K) row-major (lda);
// B (N, K) when NT else (K, N) (ldb). SCALE_A: A element = float(TA(TA(ascale) · a)) (the fallback's q · scale in T).
// Grid (ceil(N / 8), ceil(M / 8), parts) threadgroups of 32.
template <typename TA, typename TB, int NT, int SCALE_A>
[[kernel]] void glm_mma(
    const device TA* A [[buffer(0)]],
    const device TB* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& lda [[buffer(6)]],
    constant uint& ldb [[buffer(7)]],
    constant uint& psize [[buffer(8)]],    // partition size (split-k); 0 = one fused partition
    constant uint& tail_bk [[buffer(9)]],  // fused: BK of the tail-first rule
    constant float& ascale [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tgs [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float As[64];
    threadgroup float Bs[64];
    threadgroup float Cs[64];
    const int m0 = int(tg.y) * 8, n0 = int(tg.x) * 8;
    const int part = int(tg.z), parts = int(tgs.z);
    const TA ts = static_cast<TA>(ascale);
    simdgroup_float8x8 c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    auto run = [&](int k0, int k1) {
        for (int kc = k0; kc < k1; kc += 8) {
            for (uint e = lane; e < 64; e += 32) {
                const int r = int(e) / 8, kk = int(e) % 8;
                const int m = m0 + r, k = kc + kk;
                float av = 0.0f;
                if (m < int(M) && k < k1) {
                    const TA a = A[ulong(m) * lda + k];
                    av = SCALE_A ? float(static_cast<TA>(float(ts) * float(a))) : float(a);
                }
                As[e] = av;
                const int n = n0 + int(e) % 8, k2 = kc + int(e) / 8;
                float bv = 0.0f;
                if (n < int(N) && k2 < k1) bv = NT ? float(B[ulong(n) * ldb + k2]) : float(B[ulong(k2) * ldb + n]);
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

// out[m, n] = T(Σ_p C[p, m, n]) (sequential from 0, MLX gemm_splitk_accum; one partition: T(C)); MASK: -inf for keys
// the selection drops (n = key index). Grid (N, M).
template <typename T, int MASK>
[[kernel]] void glm_mma_out(
    const device float* C [[buffer(0)]],
    device T* out [[buffer(1)]],
    const device uint8_t* sel [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& parts [[buffer(5)]],
    constant uint& nsel [[buffer(6)]],
    constant uint& kpool [[buffer(7)]],
    constant uint& ncomp [[buffer(8)]],
    uint2 g [[thread_position_in_grid]]) {
    if (g.x >= N || g.y >= M) return;
    const ulong i = ulong(g.y) * N + g.x;
    float acc = 0.0f;
    for (uint p = 0; p < parts; ++p) acc += C[ulong(p) * M * N + i];
    if (MASK && !glm_use_key(int(g.x), nsel, sel, kpool, ncomp)) {
        out[i] = static_cast<T>(-INFINITY);
        return;
    }
    out[i] = static_cast<T>(acc);
}

// Indexer pool scores from the GEMM partials C (parts, NH, P) of q · Kᵀ (float32): score[p] = Σ_h w_h ·
// max(s_hp · dscale, 0), s_hp = Σ_parts C, w_h = float(wp[h]) · hscale; the head sum in MLX's column order.
// One threadgroup of 32 per pool.
template <typename T, int NH>
[[kernel]] void glm_index_reduce(
    const device float* C [[buffer(0)]],
    const device T* wp [[buffer(1)]],
    device float* score [[buffer(2)]],
    constant uint& P [[buffer(3)]],
    constant uint& parts [[buffer(4)]],
    constant float& dscale [[buffer(5)]],
    constant float& hscale [[buffer(6)]],
    uint j [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    auto head = [&](int h) -> float {
        float acc = 0.0f;
        for (uint p = 0; p < parts; ++p) acc += C[(ulong(p) * NH + h) * P + j];
        const float s = metal::max(acc * dscale, 0.0f);
        const float w = float(wp[h]) * hscale;
        return w * s;
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

// ------------------------------------------------------------------------------- batched prefill (not row-exact)
// Row-batched variants of the decode kernels above for the default batched prefill (glm5.cpp encode_chunk_fast): the
// same per-row arithmetic (so every row gets its decode step's bits for these ops); only the projections (GEMMs) and
// the chunk attention (glm_attn_chunk: another accumulation order than MLX's matmuls) differ from decode steps.
//   glm_hc_ss_rows          Σ x² of each stream row (ds4_hc_in without the float copy); grid (rows) threadgroups
//   glm_hc_mix_rows         ds4_gemv_xf<SCALED = 1> with the row's own Σ x²; grid (blocks, rows)
//   glm_hc_coef_rows        glm_hc_coef_par per row; grid (rows) threadgroups of 32
//   glm_hc_collapse_rms_rows ds4_hc_collapse_rms<MODE = 2> per row; grid (rows) threadgroups of TPG
//   glm_hc_post_rows        glm_hc_post per row; grid (D, HC, rows)
//   glm_kda_conv_rows       glm_kda_conv over n tokens in order (window in registers); grid (C)
//   glm_l2norm_rows         glm_l2norm per token; grid (2H, n) threadgroups
//   glm_cast_rows           y[r, c] = float(x[r, c]) with row strides; grid (cols, rows)
//   glm_kda_gate_rows       glm_kda_gate per token; grid (H · Dk, n)
//   glm_gated_delta_rows    glm_gated_delta over n tokens in order, the state in registers; grid (32, Dv, Hv)
//   glm_attn_chunk          absorbed MLA attention of n query rows over the cached latents 0 .. p0 + i (with the
//                           rows' DSA pool selection); one simdgroup per (row, head)
//   glm_index_scores_rows   DSA indexer pool scores of the chunk rows (q · K in float32 lanes, not MLX's GEMM order)
//   glm_copy_f32            y[i] = x[i]
template <typename T, int W, int TPG>
[[kernel]] void glm_hc_ss_rows(
    const device T* x [[buffer(0)]],
    device float* ss [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    threadgroup float sh[32];
    const device T* in = x + ulong(row) * W + lid * 4;
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
    if (lid == 0) ss[row] = total;
}

template <typename TX, typename TW, int IN, int OUT, int SN, int TN, int BN>
[[kernel]] void glm_hc_mix_rows(
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
    const float rs = ds4_rs_of(ss[tid.y], inv_w, eps);
    float acc = 0.0f;
    for (int i = 0; i < IN; i += BLK) {
        const int b = i + (sgN * SN + thrN) * TN;
        for (int tn = 0; tn < TN; ++tn) {
            const int j = b + tn;
            if (j < IN) {
                float v = ds4_mul_nc(float(xr[j]), rs);
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

template <typename TP, int HC, int ITERS>
[[kernel]] void glm_hc_coef_rows(
    const device float* raw [[buffer(0)]],
    const device TP* base [[buffer(1)]],
    const device TP* scale [[buffer(2)]],
    device float* coef [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    static_assert(HC * HC <= 32, "one simdgroup holds the comb matrix");
    constexpr int MIX = (2 + HC) * HC;
    raw += ulong(row) * MIX;
    coef += ulong(row) * MIX;
    if (lane < uint(HC)) {
        const float a = raw[lane] * float(scale[0]);
        coef[lane] = ds4_sigmoid(a + float(base[lane])) + eps;
    } else if (lane < uint(2 * HC)) {
        const float a2 = raw[lane] * float(scale[1]);
        coef[lane] = 2.0f * ds4_sigmoid(a2 + float(base[lane]));
    }
    const uint l = lane < uint(HC * HC) ? lane : 0u;
    const int k = 2 * HC + int(l);
    const float a = raw[k] * float(scale[2]);
    const float c = ds4_sinkhorn_lane<HC, ITERS>(a + float(base[k]), l, eps);
    if (lane < uint(HC * HC)) coef[2 * HC + lane] = c;
}

template <typename T, typename TP, int D, int HC, int TPG>
[[kernel]] void glm_hc_collapse_rms_rows(
    const device T* x [[buffer(0)]],
    const device float* raw [[buffer(1)]],
    const device TP* base [[buffer(3)]],
    const device TP* scale [[buffer(4)]],
    const device T* w [[buffer(5)]],
    device T* yn [[buffer(6)]],
    constant float& hc_eps [[buffer(9)]],
    constant float& rms_eps [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    static_assert(D <= 4096, "rms_single_row");
    constexpr int R = 4;
    constexpr int MIX = (2 + HC) * HC;
    x += ulong(row) * HC * D;
    raw += ulong(row) * MIX;
    yn += ulong(row) * D;
    threadgroup float local_inv_mean[32];
    float pre[HC];
    {
        const float s0 = float(scale[0]);
        for (int i = 0; i < HC; ++i) pre[i] = ds4_hc_pre_coef<TP, 2>(raw, base, s0, 1.0f, hc_eps, i);
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

template <typename T, int D, int HC>
[[kernel]] void glm_hc_post_rows(
    const device T* f [[buffer(0)]],
    const device T* res [[buffer(1)]],
    const device float* coef [[buffer(2)]],
    device T* y [[buffer(3)]],
    uint3 g [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    constexpr int MIX = (2 + HC) * HC;
    const uint d = g.x, j = g.y;
    if (d >= uint(D)) return;
    f += ulong(g.z) * D;
    res += ulong(g.z) * HC * D;
    coef += ulong(g.z) * MIX;
    y += ulong(g.z) * HC * D;
    float acc = 0.0f;
    for (int i = 0; i < HC; ++i)
        acc = metal::fma(float(static_cast<T>(coef[2 * HC + i * HC + j])), float(res[i * D + d]), acc);
    const T mm = static_cast<T>(acc);
    const float pj = float(static_cast<T>(coef[HC + j]));
    const T p = static_cast<T>(pj * float(f[d]));
    y[j * D + d] = static_cast<T>(float(p) + float(mm));
}

// x: the q, k, v projections of the n tokens as three planar (n, C / 3) blocks; y (n, C) row-interleaved [q | k | v].
template <typename T, int C, int K>
[[kernel]] void glm_kda_conv_rows(
    const device T* x [[buffer(0)]],
    device T* state [[buffer(1)]],
    const device float* w [[buffer(2)]],
    device T* y [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint c [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (c >= uint(C)) return;
    constexpr int S = C / 3;
    const device T* xs = x + ulong(c / uint(S)) * n * S + (c % uint(S));
    T win[K];
    float wk[K];
    for (int i = 0; i < K - 1; ++i) win[i] = state[ulong(i) * C + c];
    for (int i = 0; i < K; ++i) wk[i] = w[ulong(c) * K + i];
    for (uint t = 0; t < n; ++t) {
        win[K - 1] = xs[ulong(t) * S];
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
            const float tt = float(win[i]) * wk[i];
            acc = i == 0 ? tt : acc + tt;
        }
        const float s = acc * ds4_sigmoid(acc);
        y[ulong(t) * C + c] = static_cast<T>(s);
        for (int i = 0; i < K - 1; ++i) win[i] = win[i + 1];
    }
    for (int i = 0; i < K - 1; ++i) state[ulong(i) * C + c] = win[i];
}

template <typename T, int N, int TPG>
[[kernel]] void glm_l2norm_rows(
    const device T* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& qrows [[buffer(2)]],
    constant float& qscale [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    threadgroup float sh[33];
    const uint lid = lid2.x;
    const uint row = tg.x;
    const device T* xr = x + ulong(tg.y) * 3 * qrows * N + ulong(row) * N;
    out += ulong(tg.y) * 2 * qrows * N;
    float total = 0.0f;
    if (N <= 64) {
        if (lid == 0) {
            for (int j = 0; j < N; ++j) {
                const float v = float(xr[j]);
                const float sq = v * v;
                total = sq + total;
            }
            sh[32] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = sh[32];
    } else {
        const device T* in = xr + lid * 4;
        constexpr int blocks = N / (TPG * 4);
        constexpr int extra = N - blocks * TPG * 4;
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
            if (lid == 0) sh[32] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            total = sh[32];
        }
    }
    const float den = metal::precise::sqrt(total + 1e-6f);
    for (uint j = lid; j < uint(N); j += TPG) {
        float v = float(xr[j]) / den;
        if (row < qrows) v = v * qscale;
        out[ulong(row) * N + j] = v;
    }
}

template <typename T>
[[kernel]] void glm_cast_rows(
    const device T* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& xs [[buffer(3)]],
    constant uint& ys [[buffer(4)]],
    uint2 g [[thread_position_in_grid]]) {
    if (g.x >= cols) return;
    y[ulong(g.y) * ys + g.x] = float(x[ulong(g.y) * xs + g.x]);
}

template <typename T, typename TA, typename TB, int DK>
[[kernel]] void glm_kda_gate_rows(
    const device T* f [[buffer(0)]],
    const device TB* dt_bias [[buffer(1)]],
    const device TA* a_log [[buffer(2)]],
    const device T* b [[buffer(3)]],
    device float* decay [[buffer(4)]],
    device float* beta [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant float& lb [[buffer(8)]],
    uint2 g [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    const uint c = g.x;
    if (c >= n) return;
    f += ulong(g.y) * n;
    decay += ulong(g.y) * n;
    b += ulong(g.y) * heads;
    beta += ulong(g.y) * heads;
    const uint h = c / uint(DK);
    const float fv = float(f[c]) + float(dt_bias[c]);
    const float ea = metal::precise::exp(float(a_log[h]));
    const float a = ea * fv;
    const float s = ds4_sigmoid(a);
    const float gl = lb * s;
    decay[c] = metal::precise::exp(gl);
    if (c < heads) beta[c] = float(q4_sigmoid(b[c]));
}

// qk: (n, 2 · Hk · Dk) float (q heads, then k heads per token); v, y (n, Hv · Dv); g (n, Hv · Dk); beta (n, Hv).
// DVP state rows (dv) per simdgroup (they share the q / k / g loads); the next token's inputs are loaded before the
// current token's update (software pipelining: the recurrence is latency-bound). Same arithmetic per (dv, token) as
// glm_gated_delta. Grid (32, Dv / DVP, Hv), threadgroup (32, 4, 1).
template <typename TO, int Dk, int Dv, int Hk, int Hv, int DVP>
[[kernel]] void glm_gated_delta_rows(
    const device float* qk [[buffer(0)]],
    const device float* v [[buffer(2)]],
    const device float* g [[buffer(3)]],
    const device float* beta [[buffer(4)]],
    device float* st [[buffer(5)]],
    device float* y [[buffer(6)]],
    constant uint& ntok [[buffer(7)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
    const uint n = thread_position_in_grid.z;
    const uint hv_idx = n % Hv;
    const uint hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    const uint dk_idx = thread_position_in_threadgroup.x;
    const uint dv0 = thread_position_in_grid.y * DVP;
    float state[DVP][n_per_t];
    for (int r = 0; r < DVP; ++r) {
        const device float* i_state = st + (ulong(n) * Dv + dv0 + r) * Dk;
        for (int i = 0; i < n_per_t; ++i) state[r][i] = i_state[n_per_t * dk_idx + i];
    }
    float qn[n_per_t], kn[n_per_t], gn[n_per_t], vn[DVP], bn = 0.0f;
    auto load = [&](uint t) {
        const device float* q_ = qk + ulong(t) * 2 * Hk * Dk + hk_idx * Dk + n_per_t * dk_idx;
        const device float* k_ = qk + ulong(t) * 2 * Hk * Dk + Hk * Dk + hk_idx * Dk + n_per_t * dk_idx;
        const device float* g_ = g + ulong(t) * Hv * Dk + hv_idx * Dk + n_per_t * dk_idx;
        for (int i = 0; i < n_per_t; ++i) qn[i] = q_[i], kn[i] = k_[i], gn[i] = g_[i];
        for (int r = 0; r < DVP; ++r) vn[r] = v[ulong(t) * Hv * Dv + hv_idx * Dv + dv0 + r];
        bn = beta[ulong(t) * Hv + hv_idx];
    };
    if (ntok) load(0);
    for (uint t = 0; t < ntok; ++t) {
        float qc[n_per_t], kc[n_per_t], gc[n_per_t], vc[DVP];
        for (int i = 0; i < n_per_t; ++i) qc[i] = qn[i], kc[i] = kn[i], gc[i] = gn[i];
        for (int r = 0; r < DVP; ++r) vc[r] = vn[r];
        const float bc = bn;
        if (t + 1 < ntok) load(t + 1);
        for (int r = 0; r < DVP; ++r) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                state[r][i] = state[r][i] * gc[i];
                kv_mem += state[r][i] * kc[i];
            }
            kv_mem = simd_sum(kv_mem);
            const float delta = (vc[r] - kv_mem) * bc;
            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                state[r][i] = state[r][i] + kc[i] * delta;
                out += state[r][i] * qc[i];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0)
                y[ulong(t) * Hv * Dv + hv_idx * Dv + dv0 + r] = float(static_cast<TO>(static_cast<float>(out)));
        }
    }
    for (int r = 0; r < DVP; ++r) {
        device float* i_state = st + (ulong(n) * Dv + dv0 + r) * Dk;
        for (int i = 0; i < n_per_t; ++i) i_state[n_per_t * dk_idx + i] = state[r][i];
    }
}

// Absorbed MLA attention of the chunk rows: row i (position p0 + i) of q (n, H, D) over the latents 0 .. p0 + i (K = V).
// DSA selection: row i has np_i = (p0 + i + 1) / kp complete pools; when np_i > topk its key j is used only if it
// lies past the complete pools or its pool is selected (sel row i, stride ps): glm_use_key. One simdgroup per (row,
// head), 4 heads per threadgroup: grid (ceil(H / 4), n) threadgroups of 128. RND = 1: the roundings of the decode
// step's folded matmul attention (q' = T(T(scale) · q), scores and probabilities in T, precise exp); RND = 0: those of
// its sdpa_vector (float32 scale · q and scores; online softmax). Unused keys are skipped (decode: score -inf, p = 0).
template <typename T, int D, int RND>
[[kernel]] void glm_attn_chunk(
    const device T* q [[buffer(0)]],
    const device T* lat [[buffer(1)]],
    device T* o [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    constant uint& p0 [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    const device uint8_t* sel [[buffer(6)]],
    constant uint& ps [[buffer(7)]],
    constant uint& kp [[buffer(8)]],
    constant uint& topk [[buffer(9)]],
    const device uint32_t* sidx [[buffer(10)]],
    const device uint32_t* scnt [[buffer(11)]],
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
    const uint N0 = p0 + i + 1, np = N0 / kp;
    const bool use_sel = np > topk;
    // selected rows: the keys of the row's selected pools (ascending list sidx[i · topk ..], scnt[i] pools), then the
    // incomplete tail pool; the same keys in the same order as the mask over 0 .. N0 - 1, without visiting the others
    const device uint32_t* si = sidx + ulong(i) * topk;
    const uint nsel = use_sel ? scnt[i] * kp : 0u;
    const uint N = use_sel ? nsel + (N0 - np * kp) : N0;
    auto key_of = [&](uint t) -> uint { return !use_sel ? t : (t < nsel ? si[t / kp] * kp + t % kp : np * kp + (t - nsel)); };
    auto used = [&](uint) { return true; };
    (void)sel;
    (void)ps;
    auto score = [&](uint j) {
        const device T* kr = lat + ulong(key_of(j)) * D + lane * PT;
        float d = 0.0f;
        for (int k = 0; k < PT; ++k) d = metal::fma(qv[k], float(kr[k]), d);
        d = simd_sum(d);
        return RND ? float(static_cast<T>(d)) : d;
    };
    device T* orow = o + (ulong(i) * H + h) * D + lane * PT;
    if (RND) {   // pass 1: max and Σ exp(s - max) online (rescaled; decode: max first); pass 2: T(p) · V
        // up to SCAP keys: the scores (T values) of pass 1 kept in threadgroup memory for pass 2 (same bits)
        constexpr uint SCAP = sizeof(T) == 2 ? 2560 : 1280;   // 20 KB
        threadgroup T scs[4][SCAP];
        threadgroup T* sc = scs[sg];
        if (N <= SCAP) {
            float m = -INFINITY, l = 0.0f;
            for (uint j = 0; j < N; ++j) {
                const float sj = score(j);
                if (lane == 0) sc[j] = static_cast<T>(sj);
                if (sj > m) {
                    l = l * metal::precise::exp(m - sj) + 1.0f;
                    m = sj;
                } else {
                    l += metal::precise::exp(sj - m);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = 0; j < N; ++j) {
                const float pj = float(static_cast<T>(metal::precise::exp(float(sc[j]) - m) / l));
                const device T* vr = lat + ulong(key_of(j)) * D + lane * PT;
                for (int k = 0; k < PT; ++k) ov[k] = metal::fma(pj, float(vr[k]), ov[k]);
            }
            for (int k = 0; k < PT; ++k) orow[k] = static_cast<T>(ov[k]);
            return;
        }
        float m = -INFINITY, l = 0.0f;
        for (uint j = 0; j < N; ++j) {
            if (!used(j)) continue;
            const float sj = score(j);
            if (sj > m) {
                l = l * metal::precise::exp(m - sj) + 1.0f;
                m = sj;
            } else {
                l += metal::precise::exp(sj - m);
            }
        }
        for (uint j = 0; j < N; ++j) {
            if (!used(j)) continue;
            const float pj = float(static_cast<T>(metal::precise::exp(score(j) - m) / l));
            const device T* vr = lat + ulong(key_of(j)) * D + lane * PT;
            for (int k = 0; k < PT; ++k) ov[k] = metal::fma(pj, float(vr[k]), ov[k]);
        }
        for (int k = 0; k < PT; ++k) orow[k] = static_cast<T>(ov[k]);
        return;
    }
    float m = -INFINITY, l = 0.0f;
    for (uint j = 0; j < N; ++j) {
        if (!used(j)) continue;
        const float d = score(j);
        const device T* kr = lat + ulong(key_of(j)) * D + lane * PT;
        const float mn = metal::max(m, d);
        const float a = metal::precise::exp(m - mn), b = metal::precise::exp(d - mn);
        l = l * a + b;
        for (int k = 0; k < PT; ++k) ov[k] = ov[k] * a + b * float(kr[k]);
        m = mn;
    }
    const float inv = 1.0f / l;
    for (int k = 0; k < PT; ++k) orow[k] = static_cast<T>(ov[k] * inv);
}

// glm_attn_chunk (RND = 1) with HPS heads per simdgroup and 2 simdgroups per threadgroup (grid (H / (2 · HPS), rows),
// 64 threads): each latent row is loaded once for the HPS heads of a simdgroup; every head's arithmetic is that of
// glm_attn_chunk (same bits). Scores kept in threadgroup memory up to SCAP keys per head (else recomputed).
template <typename T, int D, int HPS, int KEEP>
[[kernel]] void glm_attn_chunk_h(
    const device T* q [[buffer(0)]],
    const device T* lat [[buffer(1)]],
    device T* o [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    constant uint& p0 [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    const device uint8_t* sel [[buffer(6)]],
    constant uint& ps [[buffer(7)]],
    constant uint& kp [[buffer(8)]],
    constant uint& topk [[buffer(9)]],
    const device uint32_t* sidx [[buffer(10)]],
    const device uint32_t* scnt [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int PT = D / 32;
    constexpr uint SCAP = KEEP ? (sizeof(T) == 2 ? 5120 : 2560) / HPS : 1;   // 20 KB per threadgroup (KEEP)
    threadgroup T scs[2][HPS][SCAP];
    (void)sel;
    (void)ps;
    const uint i = tg.y, h0 = (tg.x * 2 + sg) * HPS;
    if (h0 >= H) return;
    float qv[HPS][PT], ov[HPS][PT], m[HPS], l[HPS];
    const T ts = static_cast<T>(scale);
    for (int e = 0; e < HPS; ++e) {
        const device T* qr = q + (ulong(i) * H + h0 + e) * D + lane * PT;
        for (int k = 0; k < PT; ++k) {
            qv[e][k] = float(static_cast<T>(float(ts) * float(qr[k])));
            ov[e][k] = 0.0f;
        }
        m[e] = -INFINITY;
        l[e] = 0.0f;
    }
    const uint N0 = p0 + i + 1, np = N0 / kp;
    const bool use_sel = np > topk;
    const device uint32_t* si = sidx + ulong(i) * topk;
    const uint nsel = use_sel ? scnt[i] * kp : 0u;
    const uint N = use_sel ? nsel + (N0 - np * kp) : N0;
    auto key_of = [&](uint t) -> uint { return !use_sel ? t : (t < nsel ? si[t / kp] * kp + t % kp : np * kp + (t - nsel)); };
    auto scores = [&](uint j, thread float* sv) {
        const device T* kr = lat + ulong(key_of(j)) * D + lane * PT;
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
    {
        // 32 keys at a time: the scores (one simd_sum each) kept by lane t for key c0 + t, the running max before
        // each key by a max scan (exact), the exponentials one per lane, then the online recurrence in key order over
        // shuffles; pass 2 likewise with each lane's p broadcast. Every value and every operation's order are those of
        // the sequential loops below (same bits), with 1 / 32 of the exponentials.
        for (uint c0 = 0; c0 < N; c0 += 32) {
            const uint cn = min(32u, N - c0);
            float mine[HPS];
            for (int e = 0; e < HPS; ++e) mine[e] = -INFINITY;
            for (uint t = 0; t < cn; ++t) {
                float sv[HPS];
                scores(c0 + t, sv);
                if (lane == t)
                    for (int e = 0; e < HPS; ++e) mine[e] = sv[e];
            }
            if (keep && lane < cn)
                for (int e = 0; e < HPS; ++e) scs[sg][e][c0 + lane] = static_cast<T>(mine[e]);
            for (int e = 0; e < HPS; ++e) {
                float pm = mine[e];   // inclusive prefix max over lanes, then exclusive with the running max
                for (ushort d = 1; d < 32; d <<= 1) {
                    const float o2 = simd_shuffle_up(pm, d);
                    if (lane >= d) pm = metal::max(pm, o2);
                }
                float ex = simd_shuffle_up(pm, ushort(1));
                const float mp = lane == 0 ? m[e] : metal::max(m[e], ex);
                const bool gt = mine[e] > mp;
                const float f = lane < cn ? (gt ? metal::precise::exp(mp - mine[e]) : metal::precise::exp(mine[e] - mp))
                                          : 0.0f;
                for (uint t = 0; t < cn; ++t) {
                    const bool g = simd_shuffle(uint(gt), ushort(t)) != 0u;
                    const float fv = simd_shuffle(f, ushort(t));
                    if (g) {
                        l[e] = l[e] * fv + 1.0f;
                        m[e] = simd_shuffle(mine[e], ushort(t));
                    } else {
                        l[e] += fv;
                    }
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c0 = 0; c0 < N; c0 += 32) {
            const uint cn = min(32u, N - c0);
            float pl[HPS];
            for (int e = 0; e < HPS; ++e) pl[e] = 0.0f;
            if (keep) {
                if (lane < cn)
                    for (int e = 0; e < HPS; ++e)
                        pl[e] = float(static_cast<T>(metal::precise::exp(float(scs[sg][e][c0 + lane]) - m[e]) / l[e]));
            } else {
                for (uint t = 0; t < cn; ++t) {
                    float sv[HPS];
                    scores(c0 + t, sv);
                    if (lane == t)
                        for (int e = 0; e < HPS; ++e)
                            pl[e] = float(static_cast<T>(metal::precise::exp(sv[e] - m[e]) / l[e]));
                }
            }
            for (uint t = 0; t < cn; ++t) {
                const device T* vr = lat + ulong(key_of(c0 + t)) * D + lane * PT;
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
    }
}

// DSA indexer pool scores of the chunk rows: score[i, p] = Σ_h w_h · relu(dscale · (q_h · K_p)) (w_h = float(wp[i, h])
// · hscale; the head sum in glm_index_reduce's order) for the pools p < (p0 + i + 1) / kp; q (n, NH, IDD), pools
// (P, IDD), wp (n, NH) in T; score (n, ps) float32. The dot products in float32 (lanes over IDD; decode: MLX's GEMM
// order). One simdgroup per (pool, row): grid (P, n) threadgroups of 32.
template <typename T, int NH, int IDD>
[[kernel]] void glm_index_scores_rows(
    const device T* q [[buffer(0)]],
    const device T* pools [[buffer(1)]],
    const device T* wp [[buffer(2)]],
    device float* score [[buffer(3)]],
    constant uint& ps [[buffer(4)]],
    constant uint& p0 [[buffer(5)]],
    constant uint& kp [[buffer(6)]],
    constant float& dscale [[buffer(7)]],
    constant float& hscale [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    const uint p = tg.x, i = tg.y;
    if (p >= (p0 + i + 1) / kp) return;
    const device T* kr = pools + ulong(p) * IDD;
    const device T* qr = q + ulong(i) * NH * IDD;
    float vals[NH];
    for (int h = 0; h < NH; ++h) {
        float acc = 0.0f;
        for (int d = int(lane); d < IDD; d += 32) acc = metal::fma(float(qr[h * IDD + d]), float(kr[d]), acc);
        acc = simd_sum(acc);
        const float s = metal::max(acc * dscale, 0.0f);
        const float w = float(wp[ulong(i) * NH + h]) * hscale;
        vals[h] = w * s;
    }
    if (NH <= 16) {
        if (lane == 0) score[ulong(i) * ps + p] = ds4_colsum_small([&](int h) { return vals[h]; }, NH);
        return;
    }
    float t = 0.0f;
    for (int h = int(lane); h < NH; h += 32) {
        const float v = vals[h];
        t = v + t;
    }
    t = simd_sum(t);
    if (lane == 0) score[ulong(i) * ps + p] = t;
}

template <typename U>
[[kernel]] void glm_copy_f32(
    const device U* x [[buffer(0)]],
    device U* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = x[i];
}
