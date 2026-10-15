// suc-mm-bench: throughput of the batched-prefill GEMM kernels (suc_mm_affine / suc_mm_dense, suc/metal.hpp) on
// random weights, for kernel tuning. Synthetic data only (no SUC1 file).
//
//   suc-mm-bench [--rows N] [--in K] [--out M] [--bits 8|4|0] [--group 64] [--iters 20]
//     bits 0 = dense bf16 weights. Prints GPU ms per launch and TFLOPS (2 · N · K · M / time).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "suc/metal.hpp"

using namespace suc::metal;

int main(int argc, char** argv) {
  std::size_t n = 1024, in = 4096, out = 4096, group = 64, iters = 20;
  unsigned bits = 8;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const std::size_t v = std::strtoull(argv[i + 1], nullptr, 10);
    if (a == "--rows") n = v;
    else if (a == "--in") in = v;
    else if (a == "--out") out = v;
    else if (a == "--bits") bits = unsigned(v);
    else if (a == "--group") group = v;
    else if (a == "--iters") iters = v;
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  Context ctx;
  std::mt19937 rng(1);
  const std::size_t words = bits ? (in * bits + 31) / 32 : 0;
  BufPtr x = ctx.new_buffer(n * in * 2), y = ctx.new_buffer(n * out * 2);
  {
    auto* p = static_cast<uint16_t*>(x->contents());
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n * in; ++i) p[i] = f32_to_bf16_bits(d(rng));
  }
  BufPtr w, s, b;
  if (bits) {
    w = ctx.new_buffer(out * words * 4);
    auto* pw = static_cast<uint32_t*>(w->contents());
    for (std::size_t i = 0; i < out * words; ++i) pw[i] = rng();
    s = ctx.new_buffer(out * (in / group) * 2);
    b = ctx.new_buffer(out * (in / group) * 2);
    auto* ps = static_cast<uint16_t*>(s->contents());
    auto* pb = static_cast<uint16_t*>(b->contents());
    for (std::size_t i = 0; i < out * (in / group); ++i) ps[i] = f32_to_f16_bits(0.01f), pb[i] = f32_to_f16_bits(-0.5f);
  } else {
    w = ctx.new_buffer(out * in * 2);
    auto* pw = static_cast<uint16_t*>(w->contents());
    std::uniform_real_distribution<float> d(-0.05f, 0.05f);
    for (std::size_t i = 0; i < out * in; ++i) pw[i] = f32_to_bf16_bits(d(rng));
  }
  auto run = [&]() {
    MTL::CommandBuffer* cb = ctx.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
    for (std::size_t it = 0; it < iters; ++it) {
      const bool ok = bits ? encode_mm_affine(ctx, enc, {x.get(), 0}, {w.get(), 0}, {s.get(), 0}, {b.get(), 0},
                                              {y.get(), 0}, n, in, out, bits, group, in, out, words, "half", Act::BF16)
                           : encode_mm_dense(ctx, enc, {x.get(), 0}, {w.get(), 0}, {y.get(), 0}, n, in, out, in, out,
                                             "bfloat", Act::BF16);
      if (!ok) { std::fprintf(stderr, "unsupported shape\n"); std::exit(1); }
      enc->memoryBarrier(MTL::BarrierScopeBuffers);
    }
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    return cb->GPUEndTime() - cb->GPUStartTime();
  };
  run();   // compile + warm
  const double g = run();
  const double ms = 1e3 * g / double(iters);
  const double tflops = 2.0 * double(n) * double(in) * double(out) / (ms * 1e-3) / 1e12;
  std::printf("rows %zu in %zu out %zu bits %u: %.3f ms / launch, %.2f TFLOPS\n", n, in, out, bits, ms, tflops);
  return 0;
}
