// suc-expert-bench: SSD read micro-benchmark of the expert streaming engine (suc/expert_stream.hpp).
//
// Registers synthetic "experts" over the bytes of an existing (large) SUC1 file — the contents do not matter, only the
// read pattern: LAYERS layers of E experts of EXPERT_MB each (E = as many as fit the file), every expert two reads
// (97 % "indices" + 3 % "scales" in a different region of the file, as a SUC1 expert's sections). Then replays TOKENS
// tokens of uniform random top-k routing through ExpertStream::fetch, layer after layer with no compute in between
// (pure IO, the decode-time critical path), for each IO thread count. Reports the stream statistics, the device
// bandwidth during fetches (bytes read / time blocked) and per-layer miss latency percentiles.
//
// --gpu adds the decode protocol's GPU side: per layer, after the fetch, one command buffer whose kernel reads every
// byte of the routed experts' slots, committed and waited for (the host sync of a streamed MoE layer). ms/token then
// includes the per-split cost of the pools (residency: SUC_RESIDENCY_SET=0 shows the run without a residency set).
// --prefetch: after each fetch, the next layer's routing (drawn one layer ahead: a perfect prediction) is prefetched.
// --experts E sets the experts per layer (the ranges wrap around the file), so pools can exceed the file size.
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "suc/expert_stream.hpp"
#include "suc/metal.hpp"
#include "suc/model.hpp"

using namespace suc;
using namespace suc::metal;

static std::vector<unsigned> parse_list(const std::string& s) {
  std::vector<unsigned> v;
  std::size_t p = 0;
  while (p < s.size()) {
    const std::size_t q = s.find(',', p);
    v.push_back(unsigned(std::stoul(s.substr(p, q - p))));
    if (q == std::string::npos) break;
    p = q + 1;
  }
  return v;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: suc-expert-bench FILE.suc1 [--expert-mb 2.58] [--layers 48] [--top-k 10] [--tokens 32]\n"
                 "                        [--budget-frac 0.25] [--threads 1,4,8] [--policy lru|slru] [--cached-io]\n"
                 "                        [--seed 0] [--json OUT.json] [--gpu] [--experts E] [--prefetch]\n");
    return 2;
  }
  const std::string path = argv[1];
  double expert_mb = 2.58, budget_frac = 0.25;
  std::size_t layers = 48, top_k = 10, tokens = 32;
  std::vector<unsigned> threads = {1, 4, 8};
  bool slru = true, nocache = true, gpu = false, prefetch = false;
  std::size_t experts = 0;
  uint64_t seed = 0;
  std::string json_out;
  try {
    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (a == "--expert-mb") expert_mb = std::stod(next());
      else if (a == "--layers") layers = std::stoul(next());
      else if (a == "--top-k") top_k = std::stoul(next());
      else if (a == "--tokens") tokens = std::stoul(next());
      else if (a == "--budget-frac") budget_frac = std::stod(next());
      else if (a == "--threads") threads = parse_list(next());
      else if (a == "--policy") slru = next() == "slru";
      else if (a == "--cached-io") nocache = false;
      else if (a == "--seed") seed = std::stoull(next());
      else if (a == "--json") json_out = next();
      else if (a == "--gpu") gpu = true;
      else if (a == "--prefetch") prefetch = true;
      else if (a == "--experts") experts = std::stoul(next());
      else throw std::invalid_argument("unknown option " + a);
    }
    OpenOptions oo;
    oo.verify_checksum = false;
    Suc1File f = Suc1File::open(path, oo);
    const uint64_t size = f.file_size();
    const uint64_t eb = uint64_t(expert_mb * 1e6) / 64 * 64;
    const uint64_t idx = eb * 97 / 100 / 64 * 64, scl = eb - idx;
    const std::size_t E = experts ? experts : std::size_t(size / (layers * eb));
    if (E < top_k + 1) throw std::invalid_argument("file too small for this many layers / experts");
    // indices of every expert in the first 97 % of the file, scales after them (wrapping around with --experts)
    const uint64_t span = std::min<uint64_t>(uint64_t(layers * E), size / eb);
    const uint64_t scl_base = span * idx;
    std::printf("file %s: %.2f GB, page-cache resident %.1f %% before the run\n", path.c_str(), size / 1e9,
                100.0 * double(resident_bytes(f.bytes(0, size).data(), size)) / double(size));
    std::printf("synthetic experts: %zu layers x %zu experts x %.3f MB (reads %.3f + %.3f MB), top-%zu, %zu tokens, "
                "budget %.0f %% of %.2f GB, %s, %s\n",
                layers, E, eb / 1e6, idx / 1e6, scl / 1e6, top_k, tokens, 100 * budget_frac,
                double(layers * E * eb) / 1e9, slru ? "slru" : "lru", nocache ? "pread + F_NOCACHE" : "pread (page cache)");
    Context ctx;
    MTL::ComputePipelineState* touch = nullptr;
    MTL::Buffer* sink = nullptr;
    if (gpu) {   // one simdgroup-wide strided sum over each routed slot's bytes
      static const char* src = R"(
#include <metal_stdlib>
using namespace metal;
kernel void touch_slots(device const uint4* pool [[buffer(0)]], constant ulong* at [[buffer(1)]],
                        constant ulong& words [[buffer(2)]], device uint* out [[buffer(3)]],
                        uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                        uint2 ng [[threadgroups_per_grid]]) {
  device const uint4* p = pool + at[g.y] / 16;
  uint4 s = 0;
  for (ulong i = g.x * 256 + t; i < words; i += ng.x * 256) s += p[i];
  if (s.x + s.y + s.z + s.w == 0xdeadbeefu) out[0] = 1;
})";
      NS::Error* err = nullptr;
      MTL::Library* lib = ctx.device()->newLibrary(NS::String::string(src, NS::UTF8StringEncoding), nullptr, &err);
      if (!lib) throw std::runtime_error("touch kernel does not compile");
      MTL::Function* fn = lib->newFunction(NS::String::string("touch_slots", NS::UTF8StringEncoding));
      touch = ctx.device()->newComputePipelineState(fn, &err);
      fn->release();
      lib->release();
      sink = ctx.device()->newBuffer(64, MTL::ResourceStorageModeShared);
      std::printf("gpu: one command buffer per layer reading the routed slots; residency set %s\n",
                  ctx.residency_set() ? "on" : "off");
    }
    std::ofstream jo;
    if (!json_out.empty()) {
      jo.open(json_out);
      jo << "{\"file\": \"" << path << "\", \"device\": \"" << ctx.device_name() << "\", \"expert_bytes\": " << eb
         << ", \"layers\": " << layers << ", \"experts\": " << E << ", \"top_k\": " << top_k << ", \"tokens\": " << tokens
         << ", \"budget_frac\": " << budget_frac << ", \"nocache\": " << (nocache ? "true" : "false") << ", \"runs\": [";
    }
    for (std::size_t ti = 0; ti < threads.size(); ++ti) {
      ExpertStreamOptions xo;
      xo.budget_bytes = uint64_t(budget_frac * double(layers * E * eb));
      xo.io_threads = threads[ti];
      xo.nocache = nocache;
      xo.policy = slru ? ExpertStreamOptions::Policy::SLRU : ExpertStreamOptions::Policy::LRU;
      ExpertStream xs(ctx, f, xo);
      std::vector<std::vector<uint64_t>> slot_at(layers, std::vector<uint64_t>(E, 0));
      std::vector<MTL::Buffer*> pools(layers, nullptr);
      for (std::size_t l = 0; l < layers; ++l) {
        std::vector<std::vector<ExpertStream::Range>> r(E);
        for (std::size_t e = 0; e < E; ++e) {
          const uint64_t k = (l * E + e) % span;
          r[e] = {{k * idx, idx}, {scl_base + k * scl, scl}};
        }
        xs.add_layer("layer." + std::to_string(l), std::move(r), top_k,
                     [&slot_at, &pools, l](uint32_t e, MTL::Buffer* pool, const uint64_t* at) {
                       slot_at[l][e] = at[0];
                       pools[l] = pool;
                     });
      }
      xs.finalize();
      if (ti == 0) std::printf("pools %.2f GB, residency set %.2f GB (%.2f GB left out)\n", xs.pool_bytes() / 1e9,
                               ctx.residency_set_bytes() / 1e9, ctx.residency_skipped_bytes() / 1e9);
      double gpu_s = 0.0;
      auto run_layer = [&](std::size_t l, const uint32_t* ids) {
        if (!gpu) return;
        std::vector<uint64_t> at(top_k);
        for (std::size_t k = 0; k < top_k; ++k) at[k] = slot_at[l][ids[k]];
        const uint64_t words = idx / 16;
        MTL::CommandBuffer* cb = ctx.queue()->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(touch);
        enc->setBuffer(pools[l], 0, 0);
        enc->setBytes(at.data(), at.size() * 8, 1);
        enc->setBytes(&words, 8, 2);
        enc->setBuffer(sink, 0, 3);
        enc->dispatchThreadgroups(MTL::Size(64, top_k, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        gpu_s += cb->GPUEndTime() - cb->GPUStartTime();
      };
      // a different region of the routing space per thread count (seed + ti) keeps earlier runs' reads cold
      std::mt19937_64 rng(seed + 1000003 * ti);
      std::vector<double> lat_ms;
      std::vector<uint32_t> ids(E);
      for (std::size_t e = 0; e < E; ++e) ids[e] = uint32_t(e);
      const auto t0 = std::chrono::steady_clock::now();
      std::vector<uint32_t> nids = ids;
      std::shuffle(nids.begin(), nids.end(), rng);
      for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t l = 0; l < layers; ++l) {
          ids.swap(nids);   // this layer's routing, drawn one layer ahead
          nids = ids;
          std::shuffle(nids.begin(), nids.end(), rng);
          const uint64_t m0 = xs.stats().misses;
          const auto a = std::chrono::steady_clock::now();
          xs.fetch(int(l), ids.data(), top_k);
          if (xs.stats().misses > m0)
            lat_ms.push_back(1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
          if (prefetch) xs.prefetch(int((l + 1) % layers), nids.data(), top_k);   // a perfect prediction
          run_layer(l, ids.data());
        }
      const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::sort(lat_ms.begin(), lat_ms.end());
      auto pct = [&](double q) { return lat_ms.empty() ? 0.0 : lat_ms[std::size_t(q * double(lat_ms.size() - 1))]; };
      const ExpertStreamStats& s = xs.stats();
      const double gbps = s.wait_seconds > 0 ? double(s.bytes_read) / s.wait_seconds / 1e9 : 0.0;
      std::printf("threads %2u: %.2f ms/token IO, hit rate %.3f, %.1f MB/token read, %.2f GB/s while blocked; "
                  "layer miss latency p50 %.2f / p90 %.2f / max %.2f ms (%zu layers with misses)\n",
                  threads[ti], 1e3 * wall / double(tokens), s.hit_rate(), double(s.bytes_read) / 1e6 / double(tokens),
                  gbps, pct(0.5), pct(0.9), pct(1.0), lat_ms.size());
      if (gpu)
        std::printf("            GPU %.2f ms/token, host blocked in reads %.2f ms/token, rest (commit / wait / CPU) "
                    "%.2f ms/token\n", 1e3 * gpu_s / double(tokens), 1e3 * s.wait_seconds / double(tokens),
                    1e3 * (wall - gpu_s - s.wait_seconds) / double(tokens));
      if (jo) {
        jo << (ti ? ", " : "") << "{\"io_threads\": " << threads[ti] << ", \"ms_per_token\": " << 1e3 * wall / double(tokens)
           << ", \"gbps_blocked\": " << gbps << ", \"layer_miss_ms_p50\": " << pct(0.5)
           << ", \"layer_miss_ms_p90\": " << pct(0.9) << ", \"gpu\": " << (gpu ? "true" : "false")
           << ", \"gpu_ms_per_token\": " << 1e3 * gpu_s / double(tokens)
           << ", \"residency_set_bytes\": " << ctx.residency_set_bytes() << ", \"stream\": " << xs.stats_json() << "}";
      }
    }
    if (jo) jo << "]}\n";
  } catch (const std::exception& e) {
    std::fprintf(stderr, "suc-expert-bench: %s\n", e.what());
    return 1;
  }
  return 0;
}
