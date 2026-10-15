// Unit tests of the Metal backend (no framework, needs a Metal GPU): the lookup kernels ("vec" and
// "mma" paths, aligned and unaligned index widths, M = 1..4, G = 2..8, batch tiling), the rotation kernels
// (blocks layout incl. 12 * 2^k and 2^14, dense fallback for 2^k < 32, forward and inverse) and the structured
// residual kernel (CSR int8 / fp16, columns int8 / fp16, both) against float64 CPU references built from the
// same host arrays. Layers read from SUC1 files are cross-checked from Python (tests/unit/test_cpp_metal.py).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "suc/bits.hpp"
#include "suc/hadamard.hpp"
#include "suc/metal.hpp"
#include "suc/numpy_random.hpp"

using namespace suc;
using namespace suc::metal;

static int g_fail = 0, g_checks = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_fail;                                                              \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
    }                                                                        \
  } while (0)

static std::vector<uint32_t> pack_row(const std::vector<uint32_t>& v, unsigned bits) {
  std::vector<uint32_t> w(words_per_row(v.size(), bits), 0);
  for (std::size_t j = 0; j < v.size(); ++j)
    for (unsigned b = 0; b < bits; ++b)
      if ((v[j] >> b) & 1u) {
        const std::size_t p = j * bits + b;
        w[p / 32] |= 1u << (p % 32);
      }
  return w;
}

static float round_act(float v, Act a) {
  float o;
  uint16_t h[2];
  f32_to_act(&v, a == Act::F32 ? static_cast<void*>(&o) : static_cast<void*>(h), 1, a);
  if (a != Act::F32) act_to_f32(h, &o, 1, a);
  return o;
}

struct Spec {
  std::size_t out, in, M, G, K;
  unsigned bits;
  bool rot_in = false, rot_out = false;
  double csr = 0.0;      // fraction of entries in the CSR residual
  bool csr_i8 = true;
  std::size_t cols = 0;  // column residual
  bool col_i8 = true;
};

struct Ref {
  HostLayer h;
  std::vector<double> w;   // W' (out, in) in the rotated basis, float64
  std::vector<std::vector<uint32_t>> idx;
};

static Ref make_layer(const Spec& s, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  Ref r;
  HostLayer& h = r.h;
  h.out = s.out, h.in = s.in, h.M = s.M, h.G = s.G, h.K = s.K, h.bits = s.bits;
  h.codebooks.resize(s.M * s.K * s.G);
  for (auto& c : h.codebooks) c = f32_to_f16_bits(nd(rng));
  const std::size_t ng = s.in / s.G;
  std::uniform_int_distribution<uint32_t> ui(0, uint32_t(s.K - 1));
  for (std::size_t m = 0; m < s.M; ++m)
    for (std::size_t o = 0; o < s.out; ++o) {
      std::vector<uint32_t> v(ng);
      for (auto& x : v) x = ui(rng);
      auto w = pack_row(v, s.bits);
      h.packed.insert(h.packed.end(), w.begin(), w.end());
      r.idx.push_back(v);
    }
  std::uniform_real_distribution<float> us(0.9f, 1.1f);
  h.row_scales.resize(s.out);
  for (auto& x : h.row_scales) x = us(rng) / std::sqrt(float(s.in * s.M));
  r.w.assign(s.out * s.in, 0.0);
  for (std::size_t o = 0; o < s.out; ++o)
    for (std::size_t g = 0; g < ng; ++g)
      for (std::size_t q = 0; q < s.G; ++q) {
        double acc = 0;
        for (std::size_t m = 0; m < s.M; ++m)
          acc += f16_to_f32(h.codebooks[(m * s.K + r.idx[m * s.out + o][g]) * s.G + q]);
        r.w[o * s.in + g * s.G + q] = double(h.row_scales[o]) * acc;
      }
  if (s.rot_in) h.seed_in = seed * 7 + 1;
  if (s.rot_out) h.seed_out = seed * 7 + 2;
  if (s.csr > 0) {
    std::bernoulli_distribution keep(s.csr);
    h.res_row_ptr.assign(s.out + 1, 0);
    h.res_scale = s.csr_i8 ? 0.002f : 1.0f;
    for (std::size_t o = 0; o < s.out; ++o) {
      for (std::size_t c = 0; c < s.in; ++c)
        if (keep(rng)) {
          h.res_cols.push_back(uint32_t(c));
          if (s.csr_i8) h.res_val_i8.push_back(int8_t(int(rng() % 255) - 127));
          else h.res_val_f16.push_back(f32_to_f16_bits(0.05f * nd(rng)));
        }
      h.res_row_ptr[o + 1] = uint32_t(h.res_cols.size());
    }
  }
  if (s.cols) {
    std::vector<uint32_t> all(s.in);
    for (std::size_t c = 0; c < s.in; ++c) all[c] = uint32_t(c);
    std::shuffle(all.begin(), all.end(), rng);
    h.col_cols.assign(all.begin(), all.begin() + long(s.cols));
    std::sort(h.col_cols.begin(), h.col_cols.end());
    for (std::size_t j = 0; j < s.cols * s.out; ++j) {
      if (s.col_i8) h.col_val_i8.push_back(int8_t(int(rng() % 255) - 127));
      else h.col_val_f16.push_back(f32_to_f16_bits(0.05f * nd(rng)));
    }
    if (s.col_i8)
      for (std::size_t j = 0; j < s.cols; ++j) h.col_scale.push_back(0.001f * (1.0f + float(j % 5)));
  }
  return r;
}

// float64 reference of the full layer with the GPU's roundings of x and x̃ to the activation dtype.
static std::vector<double> reference(const Ref& r, const std::vector<float>& xa, std::size_t B, Act act) {
  const HostLayer& h = r.h;
  std::vector<double> y(B * h.out, 0.0);
  std::vector<float> scratch(2 * std::max(h.in, h.out));
  for (std::size_t b = 0; b < B; ++b) {
    std::vector<float> xr(xa.begin() + long(b * h.in), xa.begin() + long((b + 1) * h.in));
    if (h.seed_in) {
      auto sg = random_signs(h.in, *h.seed_in);
      for (std::size_t j = 0; j < h.in; ++j) xr[j] = xr[j] * sg[j];
      hadamard_transform(xr.data(), h.in, false, scratch.data());
      for (auto& v : xr) v = round_act(v, act);
    }
    std::vector<double> yt(h.out, 0.0);
    for (std::size_t o = 0; o < h.out; ++o) {
      double acc = 0;
      for (std::size_t j = 0; j < h.in; ++j) acc += r.w[o * h.in + j] * double(xr[j]);
      yt[o] = acc;
    }
    if (h.seed_out) {
      std::vector<float> t(yt.begin(), yt.end());
      hadamard_transform(t.data(), h.out, true, scratch.data());
      auto sg = random_signs(h.out, *h.seed_out);
      for (std::size_t o = 0; o < h.out; ++o) yt[o] = double(t[o]) * sg[o];
    }
    const float* x = xa.data() + b * h.in;
    for (std::size_t o = 0; o < h.out && !h.res_cols.empty(); ++o)
      for (uint32_t e = h.res_row_ptr[o]; e < h.res_row_ptr[o + 1]; ++e) {
        const double v = h.res_val_i8.empty() ? f16_to_f32(h.res_val_f16[e]) : double(h.res_val_i8[e]) * h.res_scale;
        yt[o] += v * x[h.res_cols[e]];
      }
    for (std::size_t j = 0; j < h.col_cols.size(); ++j)
      for (std::size_t o = 0; o < h.out; ++o) {
        const double v = h.col_val_i8.empty() ? f16_to_f32(h.col_val_f16[j * h.out + o])
                                              : double(h.col_val_i8[j * h.out + o]) * h.col_scale[j];
        yt[o] += v * x[h.col_cols[j]];
      }
    for (std::size_t o = 0; o < h.out; ++o) y[b * h.out + o] = yt[o];
  }
  return y;
}

static double tol_for(Act a) { return a == Act::F32 ? 2e-5 : a == Act::F16 ? 2.5e-3 : 1.6e-2; }

static double max_rel(const std::vector<float>& got, const std::vector<double>& ref) {
  double mx = 0, ma = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    mx = std::max(mx, std::fabs(double(got[i]) - ref[i]));
    ma = std::max(ma, std::fabs(ref[i]));
  }
  if (!std::isfinite(mx)) return INFINITY;
  return ma > 0 ? mx / ma : mx;
}

static void test_layers(Context& ctx) {
  const std::vector<Spec> specs = {
      {200, 384, 1, 8, 256, 8},                      // aligned 8-bit, tail rows (200 % 32 != 0)
      {256, 512, 1, 4, 16, 4},                       // 4-bit
      {96, 256, 2, 8, 64, 6},                        // unaligned 6-bit, M = 2
      {64, 192, 3, 4, 16, 4},                        // M = 3
      {128, 256, 4, 8, 256, 8},                      // M = 4 K = 256 G = 8 (16 KB, the dev container's layout)
      {64, 120, 1, 3, 32, 5},                        // odd G (natural-order mma path), unaligned bits
      {64, 128, 1, 2, 1024, 10},                     // G = 2, 10-bit
      {48, 256, 1, 8, 2, 1},                         // 1-bit
      {64, 128, 1, 8, 16, 4, true, true},            // rotations in + out (2^k)
      {96, 3072, 1, 8, 16, 4, true, false},          // 12 * 2^8 input rotation
      {40, 448, 1, 8, 16, 4, true, true},            // dense fallback both sides (28 * 16, 20 * 2)
      {128, 256, 2, 8, 16, 4, false, false, 0.02, true},        // CSR int8
      {128, 256, 1, 8, 16, 4, true, true, 0.02, false},         // CSR fp16 + rotations
      {128, 256, 1, 8, 16, 4, false, false, 0.0, true, 7, true},   // columns int8
      {130, 256, 1, 8, 16, 4, true, false, 0.01, true, 5, false},  // columns fp16 + CSR tail, out % 4 != 0
  };
  const std::vector<std::size_t> batches = {1, 3, 4, 5, 8, 16, 17, 33};
  uint64_t seed = 1;
  for (const Spec& s : specs) {
    const Ref r = make_layer(s, seed++);
    const Linear L = Linear::from_host(ctx, r.h);
    for (Act act : {Act::F32, Act::F16, Act::BF16})
      for (std::size_t B : batches)
        for (Path path : {Path::Auto, Path::Vec, Path::Mma}) {
          if (path != Path::Auto && act != Act::F16 && B != 5) continue;   // keep the run short
          std::mt19937_64 rng(seed * 1000 + B);
          std::normal_distribution<float> nd(0.0f, 1.0f);
          std::vector<float> x(B * s.in);
          for (auto& v : x) v = round_act(nd(rng), act);
          const auto y = L.forward(ctx, x, B, act, path);
          const auto ref = reference(r, x, B, act);
          const double e = max_rel(y, ref);
          const bool ok = e <= tol_for(act);
          CHECK(ok);
          if (!ok)
            std::fprintf(stderr, "  layer %zux%zu M=%zu G=%zu K=%zu bits=%u rot=%d%d csr=%g cols=%zu act=%s B=%zu "
                                 "path=%d: max_rel=%.3g\n", s.out, s.in, s.M, s.G, s.K, s.bits, s.rot_in, s.rot_out,
                         s.csr, s.cols, std::string(act_name(act)).c_str(), B, int(path), e);
        }
  }
}

static void test_rotations(Context& ctx) {
  for (std::size_t n : {64u, 128u, 1024u, 3072u, 12288u, 640u, 16384u, 448u, 40u, 24u, 2u}) {
    const std::size_t B = 3;
    std::mt19937_64 rng(n);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(B * n);
    for (auto& v : x) v = nd(rng);
    const auto sg = random_signs(n, 77);
    const auto hm = hadamard_hm(hadamard_factor(n).m);
    BufPtr xb = ctx.upload(x.data(), x.size() * 4), yb = ctx.new_buffer(x.size() * 4);
    BufPtr sb = ctx.upload(sg.data(), sg.size() * 4), hb = ctx.upload(hm.data(), hm.size() * 4);
    for (bool inv : {false, true}) {
      NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
      MTL::CommandBuffer* cb = ctx.queue()->commandBuffer();
      auto* enc = cb->computeCommandEncoder();
      encode_rht(ctx, enc, {xb.get(), 0}, {yb.get(), 0}, B, n, {sb.get(), 0}, {hb.get(), 0}, inv, Act::F32, Act::F32);
      enc->endEncoding();
      cb->commit();
      cb->waitUntilCompleted();
      pool->release();
      const float* y = static_cast<const float*>(yb->contents());
      std::vector<float> scratch(2 * n);
      double err = 0, mx = 0;
      for (std::size_t b = 0; b < B; ++b) {
        std::vector<float> v(x.begin() + long(b * n), x.begin() + long((b + 1) * n));
        if (!inv) for (std::size_t j = 0; j < n; ++j) v[j] *= sg[j];
        hadamard_transform(v.data(), n, inv, scratch.data());
        if (inv) for (std::size_t j = 0; j < n; ++j) v[j] *= sg[j];
        for (std::size_t j = 0; j < n; ++j) {
          err = std::max(err, double(std::fabs(v[j] - y[b * n + j])));
          mx = std::max(mx, double(std::fabs(v[j])));
        }
      }
      const bool ok = err <= 1e-5 * mx;
      CHECK(ok);
      if (!ok) std::fprintf(stderr, "  rotation n=%zu inverse=%d: max_abs=%.3g (max %.3g)\n", n, inv, err, mx);
    }
  }
}

static void test_conversions() {
  CHECK(parse_act("f16") == Act::F16 && parse_act("bf16") == Act::BF16 && parse_act("f32") == Act::F32);
  for (float v : {0.0f, -0.0f, 1.0f, -2.5f, 65504.0f, 6.103515625e-05f, 5.9604645e-08f})
    CHECK(f16_to_f32(f32_to_f16_bits(v)) == v);
  CHECK(f32_to_f16_bits(1.0f + 2.0f / 4096) == 0x3C00);          // ties to even
  CHECK(f32_to_f16_bits(1.0f + 6.0f / 4096) == 0x3C02);
  CHECK(f32_to_f16_bits(1.0f + 3.0f / 4096) == 0x3C01);
  CHECK(f32_to_bf16_bits(1.0f + 1.0f / 256) == 0x3F80);           // ties to even
  CHECK(f32_to_bf16_bits(1.0f + 3.0f / 256) == 0x3F82);
  CHECK(bf16_to_f32(f32_to_bf16_bits(-3.0f)) == -3.0f);
}

// ---- MoE model-graph kernels (cpp/metal/suc_model_kernels.metal; bit-exactness vs MLX: tests/unit/test_cpp_moe.py)
static void run_sync(Context& ctx, const std::function<void(MTL::ComputeCommandEncoder*)>& body) {
  MTL::CommandBuffer* cb = ctx.queue()->commandBuffer();
  MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  body(enc);
  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();
}

static float bf(float v) { return round_act(v, Act::BF16); }

// suc_moe_gemv (GEMVKernel order for (SN, TN, BN)) and suc_moe_combine (bf16 products, sequential bf16 sums) against
// exact CPU emulations; suc_moe_route: top-k invariants (distinct ids, ascending scores, ties by ascending id,
// normalized sum) on gates with deliberate ties.
static void test_moe_kernels(Context& ctx) {
  std::mt19937 rng(5);
  std::normal_distribution<float> nd;
  for (auto [H, E, sn, bn] : {std::tuple<int, int, int, int>{128, 8, 32, 8}, {2048, 128, 32, 8}, {128, 64, 32, 1},
                              {64, 8, 4, 1}, {200, 12, 32, 1}}) {
    const int rows = 3, tn = 4, sm = 32 / sn, blk = bn * sn * tn;
    std::vector<float> x(rows * H), w(E * H);
    for (auto& v : x) v = bf(nd(rng));
    for (auto& v : w) v = bf(0.05f * nd(rng));
    std::vector<uint16_t> xb(x.size()), wb(w.size());
    f32_to_act(x.data(), xb.data(), x.size(), Act::BF16);
    f32_to_act(w.data(), wb.data(), w.size(), Act::BF16);
    BufPtr X = ctx.upload(xb.data(), xb.size() * 2), W = ctx.upload(wb.data(), wb.size() * 2);
    BufPtr Y = ctx.new_buffer(rows * E * 2);
    run_sync(ctx, [&](MTL::ComputeCommandEncoder* enc) {
      enc->setComputePipelineState(ctx.pipeline("suc_moe_gemv", "bfloat, bfloat, " + std::to_string(H) + ", " +
                                                std::to_string(E) + ", " + std::to_string(sn) + ", 4, " +
                                                std::to_string(bn)));
      enc->setBuffer(X.get(), 0, 0);
      enc->setBuffer(W.get(), 0, 1);
      enc->setBuffer(Y.get(), 0, 2);
      enc->dispatchThreadgroups(MTL::Size((E + sm - 1) / sm, rows, 1), MTL::Size(32 * bn, 1, 1));
    });
    std::vector<float> y(rows * E);
    act_to_f32(Y->contents(), y.data(), y.size(), Act::BF16);
    bool ok = true;
    for (int r = 0; r < rows; ++r)
      for (int o = 0; o < E; ++o) {
        std::vector<float> lane(bn * sn, 0.0f);
        for (int i = 0; i < H; i += blk)
          for (int l = 0; l < bn * sn; ++l)
            for (int t = 0; t < tn; ++t) {
              const int j = i + l * tn + t;
              if (j < H) lane[l] = lane[l] + w[o * H + j] * x[r * H + j];
            }
        float tot = 0.0f;
        for (int s = 0; s < bn; ++s) {
          std::vector<float> v(lane.begin() + s * sn, lane.begin() + (s + 1) * sn);
          for (int d = sn / 2; d >= 1; d /= 2)
            for (int l = 0; l + d < sn; ++l) v[l] = v[l] + v[l + d];
          tot = s ? tot + v[0] : v[0];
        }
        ok = ok && bf(tot) == y[r * E + o];
      }
    CHECK(ok);
  }
  // combine
  for (int K : {2, 8}) {
    const int H = 300, n = 2;
    std::vector<float> yk(n * K * H), sc(n * K);
    for (auto& v : yk) v = bf(nd(rng));
    for (auto& v : sc) v = bf(std::fabs(nd(rng)) * 0.3f);
    std::vector<uint16_t> a(yk.size()), b(sc.size());
    f32_to_act(yk.data(), a.data(), a.size(), Act::BF16);
    f32_to_act(sc.data(), b.data(), b.size(), Act::BF16);
    BufPtr A = ctx.upload(a.data(), a.size() * 2), B = ctx.upload(b.data(), b.size() * 2), O = ctx.new_buffer(n * H * 2);
    run_sync(ctx, [&](MTL::ComputeCommandEncoder* enc) {
      enc->setComputePipelineState(ctx.pipeline("suc_moe_combine", "bfloat, " + std::to_string(K) + ", " + std::to_string(H)));
      enc->setBuffer(A.get(), 0, 0);
      enc->setBuffer(B.get(), 0, 1);
      enc->setBuffer(O.get(), 0, 2);
      enc->dispatchThreads(MTL::Size(H, n, 1), MTL::Size(256, 1, 1));
    });
    std::vector<float> o(n * H);
    act_to_f32(O->contents(), o.data(), o.size(), Act::BF16);
    bool ok = true;
    for (int r = 0; r < n; ++r)
      for (int h = 0; h < H; ++h) {
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) acc = bf(acc + bf(yk[(r * K + j) * H + h] * sc[r * K + j]));
        ok = ok && acc == o[r * H + h];
      }
    CHECK(ok);
  }
  // route: E = 16 with pairwise equal gates (tied probabilities), K = 4 / 8, norm on / off
  for (auto [K, norm] : {std::pair<int, int>{4, 1}, {8, 0}}) {
    const int E = 16, n = 4, tpg = 32;
    std::vector<float> g(n * E);
    for (int r = 0; r < n; ++r)
      for (int e = 0; e < E; e += 2) g[r * E + e] = g[r * E + e + 1] = bf(nd(rng));
    std::vector<uint16_t> gb(g.size());
    f32_to_act(g.data(), gb.data(), g.size(), Act::BF16);
    BufPtr G = ctx.upload(gb.data(), gb.size() * 2), I = ctx.new_buffer(n * K * 4), S = ctx.new_buffer(n * K * 2);
    run_sync(ctx, [&](MTL::ComputeCommandEncoder* enc) {
      enc->setComputePipelineState(ctx.pipeline("suc_moe_route", "bfloat, " + std::to_string(E) + ", " +
                                                std::to_string(K) + ", " + std::to_string(norm) + ", " +
                                                std::to_string(tpg)));
      enc->setBuffer(G.get(), 0, 0);
      enc->setBuffer(I.get(), 0, 1);
      enc->setBuffer(S.get(), 0, 2);
      enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(tpg, 1, 1));
    });
    const auto* ids = static_cast<const uint32_t*>(I->contents());
    std::vector<float> s(n * K);
    act_to_f32(S->contents(), s.data(), s.size(), Act::BF16);
    bool ok = true;
    for (int r = 0; r < n; ++r) {
      std::vector<int> order(E);
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return g[r * E + a] < g[r * E + b]; });
      float sum = 0.0f;
      for (int j = 0; j < K; ++j) {
        ok = ok && ids[r * K + j] < uint32_t(E);
        if (j) ok = ok && (s[r * K + j - 1] < s[r * K + j] || (s[r * K + j - 1] == s[r * K + j] && ids[r * K + j - 1] < ids[r * K + j]));
        sum += s[r * K + j];
      }
      // tied gates come in pairs: the top pair is selected as (2i, 2i + 1)
      ok = ok && ids[r * K + K - 1] == uint32_t(order[E - 1]) && ids[r * K + K - 2] == uint32_t(order[E - 2]);
      if (norm) ok = ok && std::fabs(sum - 1.0f) < 0.02f;
    }
    CHECK(ok);
  }
}

int main() {
  try {
    test_conversions();
    Context ctx;
    std::printf("device %s (%s), tuning cache %s, codebook_in(M=1) %s, mma_min_batch %d\n", ctx.device_name().c_str(),
                ctx.architecture().c_str(), ctx.tuning_cache_found() ? "found" : "absent",
                ctx.codebook_in(1).c_str(), ctx.mma_min_batch(1));
    test_rotations(ctx);
    test_layers(ctx);
    test_moe_kernels(ctx);
    std::printf("%d pipelines compiled in %.2f s\n", int(ctx.pipelines_compiled()), ctx.compile_seconds());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
  std::printf("%d checks, %d failures\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
