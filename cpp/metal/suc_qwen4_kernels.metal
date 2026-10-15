// qwen4_exp (Qwen3.8-Flash-Next) model-graph kernels of the C++ runtime (cpp/src/metal/qwen4.cpp). Reference semantics:
// suc/models/mlx_qwen4_exp.py on the Python SUC1 runtime (suc.runtime.suc1_model.load_suc1_model), one decode row.
// Appended after cpp/metal/suc_model_kernels.metal at build time (uses its helpers suc_m_exp / suc_m_abs / suc_m_unpack).
//
// These kernels follow the dtype flow of the Python model (every tensor the Python model materializes in the model
// dtype T is rounded to T here at the same point; Python float scalars are rounded to T first, as MLX's weak scalar
// typing does; column sums of T arrays accumulate in T, as MLX's column reduce). They are NOT bit-exact ports of every
// MLX kernel: float32 summation orders (norms, softplus, attention fallback) may differ, so results agree with the
// Python runtime to rounding (checked by tests/unit/test_cpp_qwen4.py: hidden states, logits KL, greedy tokens).
//
//   q4_rms           RMSNorm variants: (1 + w) RMSNorm in float32 (Qwen4ExpTextRMSNorm, per group of N), the weightless
//                    mx.fast.rms_norm times a scalar (Gated DeltaNet q / k), RMSNormGated (plain w, sigmoid / silu gate)
//   q4_conv          depthwise causal conv over a rolling state (Gated DeltaNet k = 4; PLE k = 4, dilation 3 + residual)
//   q4_ple_gate      PLE gate: signed_sqrt(<k_i, q_i> / sqrt(H)) per stream, sigmoid(gate) * v (MLX bf16 row reduce)
//   q4_hc_mix        hyper-connection mixer output u = mean_i(sigmoid(up)_i * xn_i)
//   q4_silu_div      silu(d / hc)
//   q4_inject        x_i + f * (2 sigmoid(inj_i / hc)) for every stream i
//   q4_delta_prep    beta = sigmoid(b), g = exp(-exp(A_log) * softplus(a + dt_bias)) (float32)
//   q4_gated_delta   mlx_lm.models.gated_delta kernel (one step), state updated in place
//   q4_rope          mx.fast.rope on the first ROT of D channels (non-traditional), the rest copied
//   q4_sigmoid_mul   attention output gate o * sigmoid(gate)
//   q4_sdpa_vector_sel / q4_sdpa_2pass_1_sel   MLX sdpa_vector / sdpa_vector_2pass_1 with the bool-mask branch fed by the
//                    QSA block selection (head dims 64 / 96 / 128 / 256; pass 2 = suc_sdpa_2pass_2)
//   q4_attn          decode attention for other head dims: FB = 1 emulates MLX's unfused fallback (bf16 scores /
//                    probabilities), FB = 0 is plain float32
//   q4_block_key     QSA block key: RoPE(k_layernorm(mean of the block's raw keys)) at the block start
//   q4_index_scores  QSA block scores sum_h relu(q_h . k_b) / sqrt(d)
//   q4_index_select  top-k blocks (exact threshold search; ties by lowest block index)
//   q4_route         softmax router in float32, top-k (argpartition order), renormalized, scores rounded to T
//   q4_shared_add    y + sigmoid(shared_expert_gate) * shared_expert

template <typename T>
inline T q4_sigmoid(T x) {   // MLX Sigmoid in T arithmetic (as suc_swiglu)
    T e = suc_m_exp(suc_m_abs(x));
    T s1 = static_cast<T>(1) / (static_cast<T>(1) + e);
    return (x < static_cast<T>(0)) ? s1 : static_cast<T>(1) - s1;
}
template <typename T>
inline T q4_silu(T x) { return x * q4_sigmoid(x); }
inline float q4_sigmoid_f(float x) {
    float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
    return x < 0.0f ? y : 1.0f - y;
}
inline float q4_log1p(float x) {   // MLX utils.h log1p
    float xp1 = 1.0f + x;
    if (xp1 == metal::numeric_limits<float>::max()) return metal::numeric_limits<float>::max();
    if (xp1 == 1.0f) return x;
    return x * (metal::precise::log(xp1) / (xp1 - 1.0f));
}
inline float q4_softplus(float x) {   // nn.softplus = logaddexp(x, 0) (MLX LogAddExp)
    float maxval = metal::max(x, 0.0f), minval = metal::min(x, 0.0f);
    return (minval == -INFINITY || maxval == INFINITY) ? maxval
                                                       : maxval + q4_log1p(metal::precise::exp(minval - maxval));
}

// ---------------------------------------------------------------------------------------------- RMSNorm variants
// Row r: x + r * xs, y + r * ys (elements); weight w + (r % wmod) * N. inv = rsqrt(mean(x²) + eps) with the sum in
// MLX rms_single_row order (TPG = 32 · ceil(ceil(N / 4) / 32) threads, 4 contiguous elements each; N <= 4096) or
// rms_looped (TPG = 1024).
//   MODE 0: y = T(float(x) · inv · (1 + float(w)))                         Qwen4ExpTextRMSNorm (float32, one rounding)
//   MODE 1: y = T(float(T(float(x) · inv)) · scale)                        mx.fast.rms_norm(x, None) · T scalar
//   MODE 2: h = T(float(w) · float(T(float(x) · inv))), y = T(float(h) · act(float(z)))   RMSNormGated (z + r * N)
// GATE (MODE 2): 0 sigmoid, 1 silu (float32).
template <typename T, typename TW, int N, int TPG, int MODE, int GATE>
[[kernel]] void q4_rms(
    const device T* x [[buffer(0)]],
    const device TW* w [[buffer(1)]],
    device T* y [[buffer(2)]],
    const device T* z [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    constant uint& xs [[buffer(6)]],
    constant uint& ys [[buffer(7)]],
    constant uint& wmod [[buffer(8)]],
    constant uint& rpt [[buffer(9)]],
    constant uint& ts [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int R = 4;
    threadgroup float local_inv_mean[32];
    // rpt > 0 (rows of several tokens): input row r at (r / rpt) · ts + (r % rpt) · xs
    x += rpt ? size_t(row / rpt) * ts + size_t(row % rpt) * xs : size_t(row) * xs;
    y += size_t(row) * ys;
    const device TW* wr = w + size_t(row % wmod) * N;
    float acc = 0;
    for (uint r = 0; r < uint(N); r += TPG * R) {
        for (int i = 0; i < R; i++) {
            if ((r + lid * R + i) < uint(N)) {
                float xi = x[r + lid * R + i];
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
        if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / N + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = local_inv_mean[0];
    for (uint j = lid; j < uint(N); j += TPG) {
        const float xv = float(x[j]);
        if (MODE == 0) {
            y[j] = static_cast<T>(xv * inv * (1.0f + float(wr[j])));
        } else if (MODE == 1) {
            y[j] = static_cast<T>(float(static_cast<T>(xv * inv)) * scale);
        } else {
            const float h = float(static_cast<T>(float(wr[j]) * float(static_cast<T>(xv * inv))));
            const float zv = float(z[size_t(row) * N + j]);
            const float gv = GATE == 0 ? q4_sigmoid_f(zv) : zv * q4_sigmoid_f(zv);
            y[j] = static_cast<T>(h * gv);
        }
    }
}

// ---------------------------------------------------------------------------------------------- causal conv
// Channel c: conv_in = [state rows 0 .. S-1 (S = (K-1)·DIL), xnew]; acc = Σ_k float(conv_in[k·DIL, c]) · float(w[c, k])
// (float32, k ascending), s = silu_T(T(acc)); the state shifts by one row (each thread owns its channel).
//   MODE 0: y[c] = s                                  Gated DeltaNet conv1d + SiLU
//   MODE 1: y[c] = T(xin[c] + T(g[c] + s))            PLE: x + (g + silu(conv(norm_conv(g))))
template <typename T, typename TW, int K, int DIL, int MODE>
[[kernel]] void q4_conv(
    const device T* xnew [[buffer(0)]],
    device T* state [[buffer(1)]],
    const device TW* w [[buffer(2)]],
    device T* y [[buffer(3)]],
    const device T* g [[buffer(4)]],
    const device T* xin [[buffer(5)]],
    constant uint& C [[buffer(6)]],
    uint c [[thread_position_in_grid]]) {
    if (c >= C) return;
    constexpr int S = (K - 1) * DIL;
    T col[S + 1];
    for (int r = 0; r < S; ++r) col[r] = state[size_t(r) * C + c];
    col[S] = xnew[c];
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float term = float(col[k * DIL]) * float(w[size_t(c) * K + k]);
        acc = k == 0 ? term : acc + term;
    }
    const T s = q4_silu(static_cast<T>(acc));
    if (MODE == 0) {
        y[c] = s;
    } else {
        const T inner = static_cast<T>(float(g[c]) + float(s));
        y[c] = static_cast<T>(float(xin[c]) + float(inner));
    }
    for (int r = 0; r < S; ++r) state[size_t(r) * C + c] = col[r + 1];
}

// ---------------------------------------------------------------------------------------------- PLE gate
// One threadgroup per stream i: p_j = T(k[i, j] · q[i, j]); s = Σ_j p_j in MLX's T row-reduce order (measured against
// MLX 0.32 for (4, N) rows: N <= 64 sequential in T from 0; larger N: TPG threads (N <= 512: 32, <= 1024: 128, else
// 1024) with T partials over 4 contiguous elements per step of 4·TPG, float simd sums rounded to T, then the simdgroup
// partials summed the same way);
// gate = T(s / rsh) (rsh = T(sqrt(H))); gate = T(T(sqrt(T(max(T(|gate|), fl)))) · sign(gate)) (fl = T(1e-6));
// g[i, j] = T(sigmoid_T(gate) · v[j]).
template <typename T, int N, int TPG>
[[kernel]] void q4_ple_gate(
    const device T* k [[buffer(0)]],
    const device T* q [[buffer(1)]],
    const device T* v [[buffer(2)]],
    device T* g [[buffer(3)]],
    constant float& rsh [[buffer(4)]],
    constant float& fl [[buffer(5)]],
    uint i [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    threadgroup float gate_tg;
    const device T* kr = k + size_t(i) * N;
    const device T* qr = q + size_t(i) * N;
    if (N <= 64) {
        if (lid == 0) {
            T s = T(0);
            for (int j = 0; j < N; ++j) s = static_cast<T>(float(s) + float(static_cast<T>(float(kr[j]) * float(qr[j]))));
            gate_tg = float(s);
        }
    } else {
        T tot = T(0);
        for (uint r = 0; r < uint(N); r += TPG * 4)
            for (int e = 0; e < 4; ++e) {
                const uint j = r + lid * 4 + e;
                if (j < uint(N)) tot = static_cast<T>(float(static_cast<T>(float(kr[j]) * float(qr[j]))) + float(tot));
            }
        const T sgs = static_cast<T>(simd_sum(float(tot)));
        if (sgid == 0) part[lane] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) part[sgid] = float(sgs);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0) {
            const T s = static_cast<T>(simd_sum(part[lane]));
            if (lane == 0) gate_tg = float(s);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        T gt = static_cast<T>(gate_tg / rsh);
        const T a = static_cast<T>(metal::max(float(suc_m_abs(gt)), fl));
        const T r = static_cast<T>(metal::precise::sqrt(float(a)));
        const float sgn = float((float(gt) > 0.0f) - (float(gt) < 0.0f));
        gt = static_cast<T>(float(r) * sgn);
        gate_tg = float(q4_sigmoid(gt));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sig = gate_tg;
    for (uint j = lid; j < uint(N); j += TPG) g[size_t(i) * N + j] = static_cast<T>(sig * float(v[j]));
}

// ---------------------------------------------------------------------------------------------- hyper-connections
// u[j] = T(T(Σ_i T(sigmoid_T(up[i·H + j]) · xn[i·H + j])) · T(1 / HC)) (sequential T sum from 0 = MLX's column reduce;
// mean = sum · (1 / n)). Grid (H).
template <typename T, int H, int HC>
[[kernel]] void q4_hc_mix(
    const device T* up [[buffer(0)]],
    const device T* xn [[buffer(1)]],
    device T* u [[buffer(2)]],
    uint2 g [[thread_position_in_grid]]) {
    const uint j = g.x;
    if (j >= uint(H)) return;
    // row-batched launches (grid y = rows): up / xn rows of HC · H, u rows of H
    up += ulong(g.y) * HC * H;
    xn += ulong(g.y) * HC * H;
    u += ulong(g.y) * H;
    T acc = T(0);
    for (int i = 0; i < HC; ++i) {
        const T m = q4_sigmoid(up[i * H + j]);
        const T p = static_cast<T>(float(m) * float(xn[i * H + j]));
        acc = static_cast<T>(float(acc) + float(p));
    }
    u[j] = static_cast<T>(float(acc) * float(static_cast<T>(1.0f / HC)));
}

// y[j] = silu_T(T(d[j] / T(HC))). Grid (n).
template <typename T, int HC>
[[kernel]] void q4_silu_div(
    const device T* d [[buffer(0)]],
    device T* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint j [[thread_position_in_grid]]) {
    if (j >= n) return;
    y[j] = q4_silu(static_cast<T>(float(d[j]) / float(static_cast<T>(HC))));
}

// y[i·H + j] = T(x[i·H + j] + T(f[j] · c_i)), c_i = T(2 · sigmoid_T(T(inj[i] / T(HC)))). Grid (H, HC). In place allowed.
template <typename T, int H, int HC>
[[kernel]] void q4_inject(
    const device T* x [[buffer(0)]],
    const device T* f [[buffer(1)]],
    const device T* inj [[buffer(2)]],
    device T* y [[buffer(3)]],
    uint3 p [[thread_position_in_grid]]) {
    if (p.x >= uint(H)) return;
    // row-batched launches (grid z = rows): x / y rows of HC · H, f rows of H, inj rows of HC
    x += ulong(p.z) * HC * H;
    y += ulong(p.z) * HC * H;
    f += ulong(p.z) * H;
    inj += ulong(p.z) * HC;
    const T c = static_cast<T>(2.0f * float(q4_sigmoid(static_cast<T>(float(inj[p.y]) / float(static_cast<T>(HC))))));
    const size_t o = size_t(p.y) * H + p.x;
    y[o] = static_cast<T>(float(x[o]) + float(static_cast<T>(float(f[p.x]) * float(c))));
}

// ---------------------------------------------------------------------------------------------- Gated DeltaNet
// beta[h] = sigmoid_T(b[h]); g[h] = exp(-exp(float(A_log[h])) · softplus(float(a[h]) + float(dt_bias[h]))). Grid (Hv).
template <typename T, typename TW>
[[kernel]] void q4_delta_prep(
    const device T* a [[buffer(0)]],
    const device T* b [[buffer(1)]],
    const device TW* a_log [[buffer(2)]],
    const device TW* dt_bias [[buffer(3)]],
    device float* g [[buffer(4)]],
    device T* beta [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint h [[thread_position_in_grid]]) {
    if (h >= n) return;
    beta[h] = q4_sigmoid(b[h]);
    const float na = -metal::precise::exp(float(a_log[h]));
    g[h] = metal::precise::exp(na * q4_softplus(float(a[h]) + float(dt_bias[h])));
}

// mlx_lm.models.gated_delta._make_gated_delta_kernel(has_mask=False, vectorized=False) for T = 1 (MLX, MIT License),
// state float32 updated in place (each thread reads and writes only its own state elements). q, k: (Hk, Dk); v, y:
// (Hv, Dv); g, beta: (Hv); state (Hv, Dv, Dk). Grid (32, Dv, Hv), threadgroup (32, 4, 1).
template <typename T, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void q4_gated_delta(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device float* g [[buffer(3)]],
    const device T* beta [[buffer(4)]],
    device float* st [[buffer(5)]],
    device T* y [[buffer(6)]],
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
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g[hv_idx];
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
    if (thread_index_in_simdgroup == 0) y[dv_idx] = static_cast<T>(out);
    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        i_state[s_idx] = state[i];
    }
}

// ---------------------------------------------------------------------------------------------- chunk variants
// Batched prefill of the Gated DeltaNet layers: the three sequential steps of n consecutive tokens in ONE launch each,
// every token's arithmetic exactly that of the one-token kernels above (same expressions in the same order, the
// state carried in registers between tokens instead of a device round trip of the same float32 / T values).

// q4_conv MODE 0 over n tokens: xnew row t at xnew + t · C, y row t at y + t · C; the window shifts once per token.
template <typename T, typename TW, int K, int DIL>
[[kernel]] void q4_conv_rows(
    const device T* xnew [[buffer(0)]],
    device T* state [[buffer(1)]],
    const device TW* w [[buffer(2)]],
    device T* y [[buffer(3)]],
    constant uint& C [[buffer(6)]],
    constant uint& n [[buffer(7)]],
    uint c [[thread_position_in_grid]]) {
    if (c >= C) return;
    constexpr int S = (K - 1) * DIL;
    T col[S + 1];
    for (int r = 0; r < S; ++r) col[r] = state[size_t(r) * C + c];
    for (uint t = 0; t < n; ++t) {
        col[S] = xnew[size_t(t) * C + c];
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float term = float(col[k * DIL]) * float(w[size_t(c) * K + k]);
            acc = k == 0 ? term : acc + term;
        }
        y[size_t(t) * C + c] = q4_silu(static_cast<T>(acc));
        for (int r = 0; r < S; ++r) col[r] = col[r + 1];
    }
    for (int r = 0; r < S; ++r) state[size_t(r) * C + c] = col[r];
}

// q4_delta_prep over n tokens: rows of Hv (a, b, g, beta at + t · Hv). Grid (Hv, n).
template <typename T, typename TW>
[[kernel]] void q4_delta_prep_rows(
    const device T* a [[buffer(0)]],
    const device T* b [[buffer(1)]],
    const device TW* a_log [[buffer(2)]],
    const device TW* dt_bias [[buffer(3)]],
    device float* g [[buffer(4)]],
    device T* beta [[buffer(5)]],
    constant uint& nh [[buffer(6)]],
    uint2 p [[thread_position_in_grid]]) {
    const uint h = p.x;
    if (h >= nh) return;
    const size_t o = size_t(p.y) * nh + h;
    beta[o] = q4_sigmoid(b[o]);
    const float na = -metal::precise::exp(float(a_log[h]));
    g[o] = metal::precise::exp(na * q4_softplus(float(a[o]) + float(dt_bias[h])));
}

// q4_gated_delta over n tokens: q, k rows of Hk · Dk; v row t at v + t · VS (inside the conv output rows); g, beta rows
// of Hv; y rows of Hv · Dv. Grid (32, Dv, Hv), threadgroup (32, 4, 1).
template <typename T, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void q4_gated_delta_rows(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device float* g [[buffer(3)]],
    const device T* beta [[buffer(4)]],
    device float* st [[buffer(5)]],
    device T* y [[buffer(6)]],
    constant uint& ntok [[buffer(7)]],
    constant uint& VS [[buffer(8)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
    auto n = thread_position_in_grid.z;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    auto i_state = st + (n * Dv + dv_idx) * Dk;
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = static_cast<float>(i_state[s_idx]);
    }
    for (uint t = 0; t < ntok; ++t) {
        auto q_ = q + size_t(t) * (Hk * Dk) + hk_idx * Dk;
        auto k_ = k + size_t(t) * (Hk * Dk) + hk_idx * Dk;
        auto v_ = v + size_t(t) * VS + hv_idx * Dv;
        auto g_ = g + size_t(t) * Hv;
        auto beta_ = beta + size_t(t) * Hv;
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
        }
        kv_mem = simd_sum(kv_mem);
        auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) y[size_t(t) * (Hv * Dv) + hv_idx * Dv + dv_idx] = static_cast<T>(out);
    }
    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        i_state[s_idx] = state[i];
    }
}

// ---------------------------------------------------------------------------------------------- attention
// = mx.fast.rope(x, ROT, traditional=False, base, scale=1, offset) on rows of D (rows at position `offset`): pairs
// (j, j + ROT/2) for j < ROT/2 with the arithmetic of suc_rope; channels ROT .. D-1 copied. Grid (D/2, rows).
template <typename T, int D, int ROT>
[[kernel]] void q4_rope(
    const device T* x [[buffer(0)]],
    device T* y [[buffer(1)]],
    constant int& offset [[buffer(2)]],
    constant float& base_log2 [[buffer(3)]],
    constant uint& xs [[buffer(4)]],
    constant uint& ys [[buffer(5)]],
    constant uint& rpp [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]]) {
    constexpr uint HALF = ROT / 2;
    // rpp > 0 (chunk rows): row r at position offset + r / rpp
    const int position = rpp ? offset + int(pos.y / rpp) : offset;
    const device T* xr = x + size_t(pos.y) * xs;
    device T* yr = y + size_t(pos.y) * ys;
    if (pos.x < HALF) {
        float d = static_cast<float>(pos.x) / static_cast<float>(HALF);
        float inv_freq = metal::exp2(-d * base_log2);
        float theta = static_cast<float>(position) * inv_freq;
        float costheta = metal::fast::cos(theta);
        float sintheta = metal::fast::sin(theta);
        uint i1 = pos.x, i2 = pos.x + HALF;
        float x1 = static_cast<float>(xr[i1]);
        float x2 = static_cast<float>(xr[i2]);
        yr[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
        yr[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
    } else {
        const uint j = ROT + 2 * (pos.x - HALF);
        if (j < uint(D)) {
            yr[j] = xr[j];
            yr[j + 1] = xr[j + 1];
        }
    }
}

// o[h·D + j] = T(o · sigmoid_T(qg[h·2D + D + j])). Grid (D, Hq).
template <typename T, int D>
[[kernel]] void q4_sigmoid_mul(
    device T* o [[buffer(0)]],
    const device T* qg [[buffer(1)]],
    uint2 p [[thread_position_in_grid]]) {
    if (p.x >= uint(D)) return;
    const size_t i = size_t(p.y) * D + p.x;
    o[i] = static_cast<T>(float(o[i]) * float(q4_sigmoid(qg[size_t(p.y) * 2 * D + D + p.x])));
}

// Decode attention, one threadgroup of TPG threads per query head h (kv head h / (Hq / Hkv)); keys / values at rows
// i · seq_stride + kv · D of the caches, n keys. Selection (QSA): key i is used when i >= nb · ratio (incomplete tail)
// or sel[i / ratio] != 0; nb = 0 selects every key. sc: float scratch (Hq, ≥ n).
//   FB = 0: float32 (q scaled by `scale` in float, as MLX sdpa_vector), out = T(Σ p_i v_i / Σ p_i)
//   FB = 1: MLX's unfused fallback: q' = T(T(scale) · q), s_i = T(q' · k_i), p_i = T(softmax_f32(s)_i), out = T(Σ p_i v_i)
template <typename T, int D, int TPG, int FB>
[[kernel]] void q4_attn(
    const device T* q [[buffer(0)]],
    const device T* kc [[buffer(1)]],
    const device T* vc [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sc [[buffer(4)]],
    const device uint8_t* sel [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& gqa [[buffer(7)]],
    constant uint& seq_stride [[buffer(8)]],
    constant uint& nb [[buffer(9)]],
    constant uint& ratio [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    constant uint& sc_stride [[buffer(12)]],
    uint h [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    threadgroup float qs[D];
    const uint kvh = h / gqa;
    device float* s = sc + size_t(h) * sc_stride;
    for (uint d = lid; d < uint(D); d += TPG) {
        const float qv = float(q[size_t(h) * D + d]);
        qs[d] = FB ? float(static_cast<T>(float(static_cast<T>(scale)) * qv)) : scale * qv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY;
    for (uint i = lid; i < n; i += TPG) {
        const bool use = nb == 0 || i >= nb * ratio || sel[i / ratio] != 0;
        float v = -INFINITY;
        if (use) {
            const device T* kr = kc + size_t(i) * seq_stride + kvh * D;
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) acc += qs[d] * float(kr[d]);
            v = FB ? float(static_cast<T>(acc)) : acc;
        }
        s[i] = v;
        mx = metal::max(mx, v);
    }
    mx = simd_max(mx);
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float m = lane < TPG / 32 ? red[lane] : -INFINITY;
        m = simd_max(m);
        if (lane == 0) red[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint i = lid; i < n; i += TPG) {
        const float e = s[i] == -INFINITY ? 0.0f : metal::precise::exp(s[i] - mx);
        s[i] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    if (lane == 0) red[sgid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float t = lane < TPG / 32 ? red[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) red[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    const float inv = 1.0f / red[0];
    for (uint d = lid; d < uint(D); d += TPG) {
        float acc = 0.0f;
        for (uint i = 0; i < n; ++i) {
            const float p = FB ? float(static_cast<T>(s[i] * inv)) : s[i];
            if (p != 0.0f) acc += p * float(vc[size_t(i) * seq_stride + kvh * D + d]);
        }
        out[size_t(h) * D + d] = static_cast<T>(FB ? acc : acc * inv);
    }
}

// ---------------------------------------------------------------------------------------------- attention with QSA selection
// suc_sdpa_vector / suc_sdpa_2pass_1 (MLX sdpa_vector.h, MIT License) with MLX's bool-mask branch: key i is skipped
// unless use_key = (nb == 0 || i >= nb · ratio || sel[i / ratio]) (the QSA block selection; the incomplete tail is always
// visible). nb = 0 is the unmasked kernel. Pass 2 of the two-pass variant is suc_sdpa_2pass_2.
template <typename T, int D, int V>
[[kernel]] void q4_sdpa_vector_sel(
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
    const device uint8_t* sel [[buffer(11)]],
    const constant uint& nb [[buffer(12)]],
    const constant uint& ratio [[buffer(13)]],
    const constant uint& sel_stride [[buffer(14)]],
    const constant uint& sel_k [[buffer(15)]],
    const device uint32_t* sidx [[buffer(16)]],
    const device uint32_t* scnt [[buffer(17)]],
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
    // chunk rows (sel_stride > 0): row q_seq_idx selects when its complete blocks n_keys / ratio exceed sel_k, from its
    // own mask row
    uint nbr = nb;
    // list mode (sel_stride == 0 and sel_k > 0, chunk rows): a row past sel_k blocks attends its selected blocks
    // sidx[q_seq_idx · sel_k ..] (ascending, scnt[q_seq_idx] of them), then the incomplete tail, key t of that
    // sequence on simdgroup t % BN
    bool list = false;
    uint nsel = 0, ntail = 0, tstart = 0;
    const device uint32_t* sl = sidx;
    if (sel_stride) {
        nbr = uint(n_keys) / ratio;
        if (nbr <= sel_k) nbr = 0;
        sel += ulong(q_seq_idx) * sel_stride;
    } else if (sel_k) {
        nbr = uint(n_keys) / ratio;
        list = nbr > sel_k;
        if (list) {
            sl = sidx + ulong(q_seq_idx) * sel_k;
            nsel = scnt[q_seq_idx] * ratio;
            tstart = nbr * ratio;
            ntail = uint(n_keys) - tstart;
        }
        nbr = 0;
    }
    queries += q_offset * D + simd_lid * qk_per_thread;
    const device T* kbase = keys + kv_head_idx * k_head_stride + simd_lid * qk_per_thread;
    const device T* vbase = values + kv_head_idx * v_head_stride + simd_lid * v_per_thread;
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

    if (list) {
        for (uint t = simd_gid; t < nsel + ntail; t += BN) {
            const uint ki = t < nsel ? sl[t / ratio] * ratio + t % ratio : tstart + (t - nsel);
            const device T* kr = kbase + ulong(ki) * k_seq_stride;
            const device T* vr = vbase + ulong(ki) * v_seq_stride;
            U score = 0;
            for (int j = 0; j < qk_per_thread; j++) score += q[j] * U(kr[j]);
            score = simd_sum(score);
            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor + exp_score * vr[j];
        }
    }
    for (int i = list ? n_keys : simd_gid; i < n_keys; i += BN) {
      const bool use_key = nbr == 0 || uint(i) >= nbr * ratio || sel[uint(i) / ratio] != 0;
      if (use_key) {
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

template <typename T, int D, int V, int BLOCKS>
[[kernel]] void q4_sdpa_2pass_1_sel(
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
    const device uint8_t* sel [[buffer(13)]],
    const constant uint& nb [[buffer(14)]],
    const constant uint& ratio [[buffer(15)]],
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
   const bool use_key = nb == 0 || uint(i) >= nb * ratio || sel[uint(i) / ratio] != 0;
   if (use_key) {
    U score = 0;
    for (int i = 0; i < qk_per_thread; i++) score += q[i] * keys[i];
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int i = 0; i < v_per_thread; i++) o[i] = o[i] * factor + exp_score * values[i];
   }
    keys += blocks * int(k_seq_stride);
    values += blocks * int(v_seq_stride);
  }
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (int i = 0; i < v_per_thread; i++) out[i] = static_cast<T>(o[i]);
}

// ---------------------------------------------------------------------------------------------- QSA indexer
// Block key of block b (one threadgroup of D threads): pooled = T(mean_f32 of keys rows b·ratio .. b·ratio + ratio - 1),
// kn = T(float(pooled) · inv · (1 + w)) ((1 + w) RMSNorm), then indexer._rope_at at position p = b · ratio (float32:
// inv_f = 1 / base^(2i / ROT), a = p · inv_f, y1 = x1 cos a + (-x2) sin a, y2 = x2 cos a + x1 sin a) -> bk[b] in T.
template <typename T, typename TW, int D, int ROT>
[[kernel]] void q4_block_key(
    const device T* keys [[buffer(0)]],
    const device TW* w [[buffer(1)]],
    device T* bk [[buffer(2)]],
    constant uint& b0 [[buffer(3)]],
    constant uint& ratio [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant float& base [[buffer(6)]],
    uint tgi [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    threadgroup float kn[D];
    const uint b = b0 + tgi;   // grid of several threadgroups: consecutive blocks b0, b0 + 1, ...
    float acc = 0.0f;
    for (uint r = 0; r < ratio; ++r) acc = r == 0 ? float(keys[size_t(b * ratio) * D + lid]) : acc + float(keys[size_t(b * ratio + r) * D + lid]);
    const float pooled = float(static_cast<T>(acc / float(ratio)));
    float sq = simd_sum(pooled * pooled);
    if (sgid == 0) red[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sgid] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float t = simd_sum(red[lane]);
        if (lane == 0) red[0] = metal::precise::rsqrt(t / D + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kn[lid] = float(static_cast<T>(pooled * red[0] * (1.0f + float(w[lid]))));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint HALF = ROT / 2;
    float yv = kn[lid];
    if (lid < uint(ROT)) {
        const uint i = lid % HALF;
        const float inv_f = 1.0f / metal::precise::pow(base, float(2 * i) / float(ROT));
        const float a = float(b * ratio) * inv_f;
        const float c = metal::precise::cos(a), s = metal::precise::sin(a);
        yv = lid < HALF ? kn[lid] * c + (-kn[lid + HALF]) * s : kn[lid] * c + kn[lid - HALF] * s;
    }
    bk[size_t(b) * D + lid] = static_cast<T>(yv);
}

// score[b] = (Σ_h max(Σ_d float(q[h, d]) · float(bk[b, d]), 0)) / sqrt_d for b < nb. Grid (nb).
template <typename T, int D, int NH>
[[kernel]] void q4_index_scores(
    const device T* q [[buffer(0)]],
    const device T* bk [[buffer(1)]],
    device float* score [[buffer(2)]],
    constant uint& nb [[buffer(3)]],
    constant float& sqrt_d [[buffer(4)]],
    uint b [[thread_position_in_grid]]) {
    if (b >= nb) return;
    float s = 0.0f;
    for (int h = 0; h < NH; ++h) {
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) acc += float(q[h * D + d]) * float(bk[size_t(b) * D + d]);
        s = h == 0 ? metal::max(acc, 0.0f) : s + metal::max(acc, 0.0f);
    }
    score[b] = s / sqrt_d;
}

// Chunk rows (fast batched prefill): row i at position p0 + i scores its complete blocks nb_i = (p0 + i + 1) / ratio
// with q4_index_scores' arithmetic when nb_i > K (q rows (NH, D) at i · qs, scores at i · ps). Grid (blocks, rows).
template <typename T, int D, int NH>
[[kernel]] void q4_index_scores_rows(
    const device T* q [[buffer(0)]],
    const device T* bk [[buffer(1)]],
    device float* score [[buffer(2)]],
    constant uint& p0 [[buffer(3)]],
    constant float& sqrt_d [[buffer(4)]],
    constant uint& ratio [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& qs [[buffer(7)]],
    constant uint& ps [[buffer(8)]],
    uint2 g [[thread_position_in_grid]]) {
    const uint b = g.x, i = g.y, nb = (p0 + i + 1) / ratio;
    if (nb <= K || b >= nb) return;
    q += ulong(i) * qs;
    float s = 0.0f;
    for (int h = 0; h < NH; ++h) {
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) acc += float(q[h * D + d]) * float(bk[size_t(b) * D + d]);
        s = h == 0 ? metal::max(acc, 0.0f) : s + metal::max(acc, 0.0f);
    }
    score[ulong(i) * ps + b] = s / sqrt_d;
}

// sel[b] = 1 for the K largest scores of nb (exact: radix search of the K-th largest value on order-preserving keys,
// then ties at that value by lowest block index), else 0. One threadgroup of TPG threads.
inline uint q4_key(float f) {
    uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// threadgroup-wide count of blocks with key >= thr (mode 0) or key == thr and index < jmax (mode 1)
template <int TPG>
inline uint q4_count(const device float* score, uint nb, uint thr, int mode, uint jmax, threadgroup uint* red,
                     uint lid, uint sgid, uint lane) {
    uint c = 0;
    for (uint i = lid; i < nb; i += TPG) {
        const uint k = q4_key(score[i]);
        c += mode == 0 ? uint(k >= thr) : uint(k == thr && i < jmax);
    }
    c = simd_sum(c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sgid] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint t = 0;
    for (uint s = 0; s < uint(TPG / 32); ++s) t += red[s];
    return t;
}
template <int TPG>
[[kernel]] void q4_index_select(
    const device float* score [[buffer(0)]],
    device uint8_t* sel [[buffer(1)]],
    constant uint& nb [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint red[32];
    uint ans = 0;   // the largest key with at least K blocks >= it (= the key of the K-th largest score)
    for (int bit = 31; bit >= 0; --bit) {
        const uint cand = ans | (1u << bit);
        if (q4_count<TPG>(score, nb, cand, 0, 0, red, lid, sgid, lane) >= K) ans = cand;
    }
    const uint ge = q4_count<TPG>(score, nb, ans, 0, 0, red, lid, sgid, lane);
    const uint eq = q4_count<TPG>(score, nb, ans, 1, 0xFFFFFFFFu, red, lid, sgid, lane);
    const uint need = K - (ge - eq);   // ties at the threshold to take, lowest block indices first (>= 1)
    uint lo = 1, hi = nb;              // smallest jmax with count(== ans, index < jmax) >= need
    while (lo < hi) {
        const uint mid = (lo + hi) / 2;
        if (q4_count<TPG>(score, nb, ans, 1, mid, red, lid, sgid, lane) >= need) hi = mid;
        else lo = mid + 1;
    }
    for (uint i = lid; i < nb; i += TPG) {
        const uint k = q4_key(score[i]);
        sel[i] = (k > ans || (k == ans && i < lo)) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------------------------- MoE
// SparseMoeBlock routing: probs = softmax_f32(float(gates)) (MLX softmax_single_row, float), top-k with argpartition
// semantics (the last K of a stable ascending argsort), scores = probs / Σ probs (float32, sequential) when NORM, then
// rounded to T (the astype before the combine). gates (rows, E) T -> eidx (rows, K), scores (rows, K) T.
template <typename T, int E, int K, int NORM, int TPG>
[[kernel]] void q4_route(
    const device T* gates [[buffer(0)]],
    device uint32_t* eidx [[buffer(1)]],
    device T* scores [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
    constexpr int N_READS = 4;
    const int lid = _lid;
    threadgroup float local_max[32];
    threadgroup float local_normalizer[32];
    threadgroup float prob[E];
    threadgroup uint selx[K];
    threadgroup float sc[K];
    float ld[N_READS];
    const device T* in = gates + gid * size_t(E);
    for (int i = 0; i < N_READS; i++) {
        const int j = lid * N_READS + i;
        ld[i] = j < E ? float(in[j]) : -INFINITY;
    }
    if (simd_group_id == 0) {
        local_max[simd_lane_id] = -INFINITY;
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
        if (lid * N_READS + i < E) prob[lid * N_READS + i] = ld[i] * normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = lid; i < E; i += TPG) {
        const float v = prob[i];
        int rank = 0;
        for (int j = 0; j < E; ++j) {
            const float u = prob[j];
            rank += (u < v || (u == v && j < i)) ? 1 : 0;
        }
        if (rank >= E - K) {
            selx[rank - (E - K)] = uint(i);
            sc[rank - (E - K)] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float s = 0.0f;
        if (NORM)
            for (int j = 0; j < K; ++j) s = j == 0 ? sc[0] : s + sc[j];
        for (int j = 0; j < K; ++j) {
            eidx[gid * K + j] = selx[j];
            scores[gid * K + j] = static_cast<T>(NORM ? sc[j] / s : sc[j]);
        }
    }
}

// y[j] = T(y[j] + T(sigmoid_T(sg[0]) · sh[j])). Grid (n).
template <typename T>
[[kernel]] void q4_shared_add(
    device T* y [[buffer(0)]],
    const device T* sg [[buffer(1)]],
    const device T* sh [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint2 g [[thread_position_in_grid]]) {
    const uint j = g.x;
    if (j >= n) return;
    // row-batched launches (grid y = rows of n)
    y += ulong(g.y) * n;
    sh += ulong(g.y) * n;
    sg += g.y;
    const T s = q4_sigmoid(sg[0]);
    y[j] = static_cast<T>(float(y[j]) + float(static_cast<T>(float(s) * float(sh[j]))));
}
