// suc-linear-bench: cold-cache microbenchmark of the C++ Metal SUC linear layer, the C++ counterpart of the
// "layers" mode of benchmarks/k3_cold_cache.py (same methods, random representations, copy counts and timing rule),
// so the C++ port can be compared with the Python/MLX kernels on the same machine.
//
// Per (method, batch): N distinct random weight copies, N = max(ceil(gb·1e9 / FP16 bytes of one layer),
// ceil(min_mb·1e6 / bytes of one copy)), so every cycle streams the weights from DRAM rather than the SLC. One sample
// = one command buffer (concurrent compute encoder) holding the N independent layer calls on the same x, timed from
// the start of encoding to completion (encode + GPU, like one mx.eval over N outputs); µs/call = sample / N; median and
// min over --samples cycles after one warm-up cycle. Also reports the GPU-only time (GPUEndTime - GPUStartTime).
//
// Methods: <suc>_kernel (lookup kernel only, auto path: vec below mma_min_batch, simdgroup-matrix from it), at every
// B >= 4 also the path auto does not take (<suc>_vec or <suc>_mma), and <suc>_full (incl. the Hadamard rotations
// V x and Uᵀ ỹ, seeds 1 / 2). <suc> in vq_G8_K256, vq_G4_K256, add2_G8_K256, add3_G8_K256 (k3_cold_cache.SUC).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "suc/bits.hpp"
#include "suc/metal.hpp"

using namespace suc;
using namespace suc::metal;

namespace {

struct SucMethod {
  std::size_t M, G, K;
};
const std::map<std::string, SucMethod> kSuc = {{"vq_G8_K256", {1, 8, 256}}, {"vq_G4_K256", {1, 4, 256}},
                                                {"add2_G8_K256", {2, 8, 256}}, {"add3_G8_K256", {3, 8, 256}}};
const char* kOrder[] = {"vq_G8_K256", "vq_G4_K256", "add2_G8_K256", "add3_G8_K256"};

void usage() {
  std::fprintf(stderr,
               "usage: suc-linear-bench [--shape 12288x4096] [--methods SUBSTR,...] [--batches 1,4,16]\n"
               "                        [--gb 2] [--min-mb 512] [--samples 5] [--dtype f16|bf16|f32]\n"
               "                        [--no-vec-compare] [--json OUT.json] [--codebook-in auto|device|threadgroup]\n"
               "                        [--mma-min-batch N] [--max-batch-tile N] [--tuning-cache PATH|-]\n");
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  std::size_t a = 0;
  while (a <= s.size()) {
    std::size_t b = s.find(',', a);
    if (b == std::string::npos) b = s.size();
    if (b > a) v.push_back(s.substr(a, b - a));
    a = b + 1;
  }
  return v;
}

uint64_t splitmix(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

HostLayer random_layer(std::size_t out, std::size_t in, const SucMethod& sm, uint64_t seed) {
  HostLayer h;
  h.out = out, h.in = in, h.M = sm.M, h.G = sm.G, h.K = sm.K;
  h.bits = unsigned(std::lround(std::log2(double(sm.K))));
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  h.codebooks.resize(sm.M * sm.K * sm.G);
  for (auto& c : h.codebooks) c = f32_to_f16_bits(nd(rng));
  const std::size_t words = words_per_row(in / sm.G, h.bits);
  h.packed.resize(sm.M * out * words);   // random uint32 words: valid indices for K = 2^bits
  uint64_t s = seed * 0x2545F4914F6CDD1Dull + 1;
  for (std::size_t i = 0; i < h.packed.size(); i += 2) {
    const uint64_t v = splitmix(s);
    h.packed[i] = uint32_t(v);
    if (i + 1 < h.packed.size()) h.packed[i + 1] = uint32_t(v >> 32);
  }
  std::uniform_real_distribution<float> us(0.9f, 1.1f);
  const float sd = 1.0f / std::sqrt(float(in)) / std::sqrt(float(sm.M));   // ≈ norm-preserving (k3_cold_cache)
  h.row_scales.resize(out);
  for (auto& x : h.row_scales) x = us(rng) * sd;
  h.seed_in = 1;
  h.seed_out = 2;
  return h;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

std::string git_commit() {
  std::string s;
  if (FILE* p = popen("git rev-parse HEAD 2>/dev/null", "r")) {
    char buf[128];
    if (std::fgets(buf, sizeof buf, p)) s = buf;
    pclose(p);
  }
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t out = 12288, in = 4096;
  std::vector<std::string> select;
  std::vector<std::size_t> batches = {1, 4, 16};
  double gb = 2.0, min_mb = 512.0;
  int samples = 5;
  bool vec_compare = true;
  std::string json_path;
  Act act = Act::F16;
  Tuning tu;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* flag) { return !std::strcmp(argv[i], flag) && i + 1 < argc; };
    if (arg("--shape")) {
      if (std::sscanf(argv[++i], "%zux%zu", &out, &in) != 2) { usage(); return 2; }
    } else if (arg("--methods")) select = split(argv[++i]);
    else if (arg("--batches")) {
      batches.clear();
      for (const auto& b : split(argv[++i])) batches.push_back(std::stoul(b));
    } else if (arg("--gb")) gb = std::atof(argv[++i]);
    else if (arg("--min-mb")) min_mb = std::atof(argv[++i]);
    else if (arg("--samples")) samples = std::max(1, std::atoi(argv[++i]));
    else if (arg("--dtype")) act = parse_act(argv[++i]);
    else if (arg("--json")) json_path = argv[++i];
    else if (arg("--codebook-in")) tu.codebook_in = argv[++i];
    else if (arg("--mma-min-batch")) tu.mma_min_batch = std::atoi(argv[++i]);
    else if (arg("--max-batch-tile")) tu.max_batch_tile = std::atoi(argv[++i]);
    else if (arg("--tuning-cache")) tu.tuning_cache = argv[++i];
    else if (!std::strcmp(argv[i], "--no-vec-compare")) vec_compare = false;
    else { usage(); return 2; }
  }
  try {
    Context ctx(tu);
    char ts[64];
    std::time_t now = std::time(nullptr);
    std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    std::printf("# device=%s (%s) shape=%zux%zu dtype=%s gb=%g min_mb=%g samples=%d tuning_cache=%s "
                "codebook_in(M=1..3)=%s/%s/%s mma_min_batch(M=1)=%d\n", ctx.device_name().c_str(),
                ctx.architecture().c_str(), out, in, std::string(act_name(act)).c_str(), gb, min_mb, samples,
                ctx.tuning_cache_found() ? "found" : "absent", ctx.codebook_in(1).c_str(), ctx.codebook_in(2).c_str(),
                ctx.codebook_in(3).c_str(), ctx.mma_min_batch(1));
    std::printf("%12s %3s %-24s %5s %10s %9s %10s %7s %7s\n", "layer", "B", "method", "N", "median_us", "min_us",
                "gpu_med_us", "bpw", "GB/s");
    std::string rows;
    const std::size_t fp16_bytes = out * in * 2;
    for (const char* name : kOrder) {
      const SucMethod& sm = kSuc.at(name);
      // methods of this representation over all batches (as k3_cold_cache.methods)
      std::vector<std::pair<std::string, std::size_t>> todo;
      for (std::size_t b : batches) {
        auto add = [&](const std::string& v) {
          const std::string m = std::string(name) + "_" + v;
          bool keep = select.empty();
          for (const auto& s : select) keep = keep || m.find(s) != std::string::npos;
          if (keep) todo.push_back({v, b});
        };
        add("kernel");
        if (vec_compare && b >= 4) add(int(b) >= ctx.mma_min_batch(int(sm.M)) ? "vec" : "mma");
        add("full");
      }
      if (todo.empty()) continue;
      const HostLayer proto = random_layer(out, in, sm, 12345);
      const std::size_t per_copy = proto.codebooks.size() * 2 + proto.packed.size() * 4 + proto.row_scales.size() * 4;
      const std::size_t n = std::max<std::size_t>(
          {std::size_t(std::ceil(gb * 1e9 / double(fp16_bytes))), std::size_t(std::ceil(min_mb * 1e6 / double(per_copy))), 1});
      std::vector<Linear> copies;
      copies.reserve(n);
      for (std::size_t i = 0; i < n; ++i) copies.push_back(Linear::from_host(ctx, random_layer(out, in, sm, 12345 + 1000 * i + 7)));
      std::sort(todo.begin(), todo.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
      for (const auto& [variant, B] : todo) {
        std::mt19937_64 rng(B);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> xh(B * in);
        for (auto& v : xh) v = nd(rng);
        BufPtr x = ctx.new_buffer(B * in * act_size(act));
        f32_to_act(xh.data(), x->contents(), xh.size(), act);
        std::vector<BufPtr> ys(n);
        std::vector<Workspace> wss(n);
        for (auto& y : ys) y = ctx.new_buffer(B * out * act_size(act));
        const Path path = variant == "vec" ? Path::Vec : variant == "mma" ? Path::Mma : Path::Auto;
        std::vector<double> wall, gpu;
        for (int s = -1; s < samples; ++s) {
          NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
          const auto t0 = std::chrono::steady_clock::now();
          MTL::CommandBuffer* cb = ctx.queue()->commandBuffer();
          MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
          for (std::size_t i = 0; i < n; ++i) {
            if (variant == "full") copies[i].encode(ctx, enc, {x.get(), 0}, {ys[i].get(), 0}, B, act, wss[i]);
            else copies[i].encode_kernel(ctx, enc, {x.get(), 0}, {ys[i].get(), 0}, B, act, act, path);
          }
          enc->endEncoding();
          cb->commit();
          cb->waitUntilCompleted();
          const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          if (cb->status() != MTL::CommandBufferStatusCompleted) throw std::runtime_error("command buffer failed");
          if (s >= 0) {
            wall.push_back(dt / double(n));
            gpu.push_back((cb->GPUEndTime() - cb->GPUStartTime()) / double(n));
          }
          pool->release();
        }
        const double med = median(wall), mn = *std::min_element(wall.begin(), wall.end()), gmed = median(gpu);
        const std::string method = std::string(name) + "_" + variant;
        const double bpw = 8.0 * double(per_copy) / double(out * in), gbps = double(per_copy) / med / 1e9;
        std::printf("%6zux%-5zu %3zu %-24s %5zu %10.1f %9.1f %10.1f %7.3f %7.1f\n", out, in, B, method.c_str(), n,
                    med * 1e6, mn * 1e6, gmed * 1e6, bpw, gbps);
        std::fflush(stdout);
        char buf[1024];
        std::snprintf(buf, sizeof buf,
                      "%s{\"mode\":\"layers\",\"out\":%zu,\"inp\":%zu,\"batch\":%zu,\"method\":\"%s\",\"path\":\"%s\","
                      "\"copies\":%zu,\"time_us\":%.3f,\"min_us\":%.3f,\"gpu_time_us\":%.3f,\"weight_bytes\":%zu,"
                      "\"bpw\":%.5f,\"gbps\":%.3f,\"samples_us\":[",
                      rows.empty() ? "" : ",\n  ", out, in, B, method.c_str(),
                      copies[0].path_for(ctx, B, path).c_str(), n, med * 1e6, mn * 1e6, gmed * 1e6, per_copy, bpw,
                      gbps);
        rows += buf;
        for (std::size_t k = 0; k < wall.size(); ++k) {
          std::snprintf(buf, sizeof buf, "%s%.3f", k ? "," : "", wall[k] * 1e6);
          rows += buf;
        }
        rows += "]}";
      }
    }
    if (!json_path.empty()) {
      FILE* fp = std::fopen(json_path.c_str(), "w");
      if (!fp) throw std::runtime_error("cannot write " + json_path);
      std::fprintf(fp,
                   "{\"benchmark\":\"cpp_suc_linear_bench\",\"device\":\"%s\",\"architecture\":\"%s\","
                   "\"git_commit\":\"%s\",\"timestamp\":\"%s\",\"seed\":12345,\"runtime\":\"suc cpp metal (milestone 2)\","
                   "\"args\":{\"shape\":[%zu,%zu],\"dtype\":\"%s\",\"gb\":%g,\"min_mb\":%g,\"samples\":%d},"
                   "\"kernel_config\":{\"codebook_in\":[\"%s\",\"%s\",\"%s\"],\"mma_min_batch\":%d,"
                   "\"tuning_cache\":%s},\"rows\":[\n  %s]}\n",
                   ctx.device_name().c_str(), ctx.architecture().c_str(), git_commit().c_str(), ts, out, in,
                   std::string(act_name(act)).c_str(), gb, min_mb, samples, ctx.codebook_in(1).c_str(),
                   ctx.codebook_in(2).c_str(), ctx.codebook_in(3).c_str(), ctx.mma_min_batch(1),
                   ctx.tuning_cache_found() ? "true" : "false", rows.c_str());
      std::fclose(fp);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
