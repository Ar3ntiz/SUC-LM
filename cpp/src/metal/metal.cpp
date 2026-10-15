// Metal backend (see suc/metal.hpp). The only translation unit with the metal-cpp private implementation.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "suc/metal.hpp"

#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <unordered_map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "suc/decode.hpp"
#include "suc/hadamard.hpp"
#include "suc/json.hpp"
#include "suc/numpy_random.hpp"

namespace suc::metal {

extern const char* const kEmbeddedKernelSource;   // generated at build time from cpp/metal/suc_kernels.metal
extern const char* const kEmbeddedModelKernelSource;   // cpp/metal/suc_model_kernels.metal (model-graph ops)
extern const char* const kEmbeddedQwen4KernelSource;   // cpp/metal/suc_qwen4_kernels.metal (qwen4_exp ops)
extern const char* const kEmbeddedDs4KernelSource;     // cpp/metal/suc_ds4_kernels.metal (deepseek_v4 ops, MXFP4)
extern const char* const kEmbeddedGlm5KernelSource;    // cpp/metal/suc_glm5_kernels.metal (glm5_next ops)

namespace {

// Scoped autorelease pool (metal-cpp returns autoreleased objects from non-new/alloc calls).
struct Pool {
  NS::AutoreleasePool* p = NS::AutoreleasePool::alloc()->init();
  ~Pool() { p->release(); }
};

NS::String* nsstr(const std::string& s) { return NS::String::string(s.c_str(), NS::UTF8StringEncoding); }
std::string str(const NS::String* s) { return s ? std::string(s->utf8String()) : std::string(); }

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("metal: " + msg); }

std::size_t page_size() { return std::size_t(::getpagesize()); }
std::size_t round_up(std::size_t n, std::size_t a) { return (n + a - 1) / a * a; }

// Constants of the Python kernels that are not configurable (suc.kernels.vq_matmul / rotation / compressed_model).
constexpr std::size_t kCodebookLimit = 16 * 1024;   // D4: all codebooks of one tensor together
constexpr std::size_t kTgMemoryLimit = 32 * 1024;
constexpr int kTgFloats = 8192 - 64;                // rotation.TG_FLOATS
constexpr std::size_t kXgFloats = 6144;             // compressed_model._XG_FLOATS
constexpr std::size_t kStructTgBytes = 32 * 1024;   // compressed_model._TG_BYTES
const int kLookupRows[][2] = {{1, 4}, {2, 4}, {4, 2}, {8, 2}, {16, 1}};   // vq_matmul.LOOKUP_ROWS

int lookup_rows(std::size_t bt) {
  for (const auto& e : kLookupRows)
    if (bt <= std::size_t(e[0])) return e[1];
  return kLookupRows[4][1];
}

// vq_matmul._mma_kcg: groups per K chunk (template constant of the mma kernel; unused by the "reg" body).
int mma_kcg(int g, int bits, int kc) {
  const int gpu = 32 % bits == 0 ? 32 / bits : 1;
  const int step = std::lcm(gpu, 8 / std::gcd(g, 8));
  const int want = (kc / g + step - 1) / step * step;
  return std::max(step, want);
}

// rotation.default_block_config
std::pair<int, int> default_block_config(std::size_t n) {
  int q = 128;
  while (std::size_t(q) * 2 <= std::min<std::size_t>(512, n / 16)) q *= 2;
  return {q, 8};
}

// suc.kernels.autotune._nearest: the entry whose integer key is nearest to m (ties: the larger key).
const Json* nearest(const Json* choices, int m) {
  if (!choices || !choices->is_object() || choices->as_object().empty()) return nullptr;
  const Json* best = nullptr;
  long bd = 0, bk = 0;
  for (const auto& [k, v] : choices->as_object()) {
    long key = std::strtol(k.c_str(), nullptr, 10), d = std::labs(key - m);
    if (!best || d < bd || (d == bd && key > bk)) best = &v, bd = d, bk = key;
  }
  return best;
}

std::string cache_path(const std::string& configured) {
  if (!configured.empty()) return configured;
  if (const char* e = std::getenv("SUC_TUNING_CACHE"); e && *e) return e;
  const char* home = std::getenv("HOME");
  return std::string(home ? home : "") + "/.cache/suc/kernel_tuning.json";
}

}  // namespace

// ------------------------------------------------------------------------------------------ activations

Act parse_act(std::string_view n) {
  if (n == "f16" || n == "float16") return Act::F16;
  if (n == "bf16" || n == "bfloat16") return Act::BF16;
  if (n == "f32" || n == "float32") return Act::F32;
  throw std::invalid_argument("unknown activation dtype '" + std::string(n) + "' (f16 | bf16 | f32)");
}
std::string_view act_name(Act a) { return a == Act::F16 ? "f16" : a == Act::BF16 ? "bf16" : "f32"; }
std::size_t act_size(Act a) { return a == Act::F32 ? 4 : 2; }
const char* act_msl(Act a) { return a == Act::F16 ? "half" : a == Act::BF16 ? "bfloat" : "float"; }

uint16_t f32_to_f16_bits(float f) { return std::bit_cast<uint16_t>(static_cast<_Float16>(f)); }

uint16_t f32_to_bf16_bits(float f) {
  uint32_t u = std::bit_cast<uint32_t>(f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return uint16_t((u >> 16) | 0x40);   // quiet NaN
  u += 0x7FFFu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}

void f32_to_act(const float* src, void* dst, std::size_t n, Act a) {
  if (a == Act::F32) {
    std::memcpy(dst, src, n * 4);
    return;
  }
  auto* d = static_cast<uint16_t*>(dst);
  for (std::size_t i = 0; i < n; ++i) d[i] = a == Act::F16 ? f32_to_f16_bits(src[i]) : f32_to_bf16_bits(src[i]);
}

void act_to_f32(const void* src, float* dst, std::size_t n, Act a) {
  if (a == Act::F32) {
    std::memcpy(dst, src, n * 4);
    return;
  }
  const auto* s = static_cast<const uint16_t*>(src);
  for (std::size_t i = 0; i < n; ++i) dst[i] = a == Act::F16 ? f16_to_f32(s[i]) : bf16_to_f32(s[i]);
}

std::string kernel_source() {
  // The SUC projection kernels (generated) followed by the model-graph kernels; $SUC_METAL_SOURCE replaces the first.
  std::string src = kEmbeddedKernelSource;
  if (const char* p = std::getenv("SUC_METAL_SOURCE"); p && *p) {
    std::ifstream in(p);
    if (!in) fail(std::string("cannot read SUC_METAL_SOURCE=") + p);
    std::stringstream ss;
    ss << in.rdbuf();
    src = ss.str();
  }
  return src + "\n" + kEmbeddedModelKernelSource + "\n" + kEmbeddedQwen4KernelSource + "\n" + kEmbeddedDs4KernelSource +
         "\n" + kEmbeddedGlm5KernelSource;
}

// ------------------------------------------------------------------------------------------ Context

struct Context::Impl {
  Tuning tuning;
  MTL::Device* device = nullptr;
  MTL::CommandQueue* queue = nullptr;
  std::string name, arch, source;
  std::unique_ptr<Json> cache;     // the device's entry of kernel_tuning.json (null when absent)
  std::map<std::string, MTL::ComputePipelineState*> pipelines;
  struct KeyHash {
    std::size_t operator()(const PsoKey& k) const {
      uint64_t h = 1469598103934665603ull ^ k.site;
      for (int64_t x : k.v) h = (h ^ uint64_t(x)) * 1099511628211ull;
      return std::size_t(h);
    }
  };
  std::unordered_map<PsoKey, MTL::ComputePipelineState*, KeyHash> keyed;   // owned by `pipelines`
  // memoized tuning answers (the cache JSON is immutable after construction; the lookups ran on every dispatch)
  std::map<int, std::string> cbin_memo;
  std::map<int, int> mmb_memo;
  std::map<std::size_t, std::pair<int, int>> rbc_memo;
  std::mutex mu;
  double compile_s = 0.0;
  std::size_t counter = 0;
  BufPtr dummy;
  // residency set (see Context::make_resident)
  MTL::ResidencySet* rset = nullptr;
  std::mutex rmu;
  std::vector<MTL::Allocation*> rmembers;
  uint64_t rbytes = 0, rskipped = 0, rcap = 0;
  std::atomic<bool> rdirty{false};
  std::atomic<std::size_t> rcalls{0};
  // drops the members only the set still holds (their owners released them); caller holds rmu
  bool prune_residency() {
    bool changed = false;
    std::size_t keep = 0;
    for (MTL::Allocation* a : rmembers) {
      if (a->retainCount() <= 1) {
        rbytes -= a->allocatedSize();
        rset->removeAllocation(a);
        changed = true;
      } else {
        rmembers[keep++] = a;
      }
    }
    rmembers.resize(keep);
    return changed;
  }

  const Json* section(const char* key) const { return cache ? cache->find(key) : nullptr; }
  const Json* rotation_choice(std::size_t n) const {
    const Json* r = section("rotation");
    const Json* c = r ? r->find("choice") : nullptr;
    return c ? c->find(std::to_string(n)) : nullptr;
  }
};

Context::Context(const Tuning& t) : p_(std::make_unique<Impl>()) {
  Pool pool;
  p_->tuning = t;
  p_->device = MTL::CreateSystemDefaultDevice();
  if (!p_->device) fail("no Metal device");
  p_->queue = p_->device->newCommandQueue();
  if (!p_->queue) fail("cannot create a command queue");
  p_->name = str(p_->device->name());
  if (auto* a = p_->device->architecture()) p_->arch = str(a->name());
  p_->source = kernel_source();
  if (t.tuning_cache != "-") {
    std::ifstream in(cache_path(t.tuning_cache));
    if (in) {
      std::stringstream ss;
      ss << in.rdbuf();
      try {
        Json root = Json::parse(ss.str());
        if (const Json* e = root.find(device_key())) p_->cache = std::make_unique<Json>(*e);
      } catch (const JsonError&) {
        // unreadable cache: built-in defaults (as suc.kernels.autotune._load)
      }
    }
  }
  {   // residency set
    const char* env = std::getenv("SUC_RESIDENCY_SET");
    const bool want = !(env && std::string(env) == "0");
    if (want && NS::ProcessInfo::processInfo()->isOperatingSystemAtLeastVersion({15, 0, 0})) {
      MTL::ResidencySetDescriptor* d = MTL::ResidencySetDescriptor::alloc()->init();
      d->setInitialCapacity(4096);
      NS::Error* err = nullptr;
      p_->rset = p_->device->newResidencySet(d, &err);
      d->release();
      if (p_->rset) {
        p_->queue->addResidencySet(p_->rset);
        uint64_t cap = p_->device->recommendedMaxWorkingSetSize();
        int mb = 0;
        std::size_t len = sizeof(mb);
        if (::sysctlbyname("iogpu.wired_limit_mb", &mb, &len, nullptr, 0) == 0 && mb > 0)
          cap = std::max<uint64_t>(cap, uint64_t(mb) << 20);
        p_->rcap = cap;
      }
    }
  }
  p_->dummy = new_buffer(64);
}

Context::~Context() {
  if (p_->rset) {
    p_->queue->removeResidencySet(p_->rset);
    p_->rset->removeAllAllocations();
    p_->rset->commit();
    p_->rset->release();
  }
  for (auto& [k, ps] : p_->pipelines) ps->release();
  if (p_->queue) p_->queue->release();
  if (p_->device) p_->device->release();
}

MTL::Device* Context::device() const { return p_->device; }
MTL::CommandQueue* Context::queue() const {
  // pending residency changes, and every 1024 command buffers a sweep for members nobody else holds
  if (p_->rset && (p_->rdirty.load(std::memory_order_relaxed) || (++p_->rcalls & 1023) == 0)) flush_residency();
  return p_->queue;
}

void Context::make_resident(MTL::Allocation* a) const {
  if (!p_->rset || !a) return;
  std::lock_guard<std::mutex> lock(p_->rmu);
  const uint64_t n = a->allocatedSize();
  if (p_->rbytes + n > p_->rcap && p_->prune_residency()) p_->rdirty = true;
  if (p_->rbytes + n > p_->rcap) {
    p_->rskipped += n;
    return;
  }
  p_->rset->addAllocation(a);   // retained by the set
  p_->rmembers.push_back(a);
  p_->rbytes += n;
  p_->rdirty = true;
}

void Context::flush_residency() const {
  if (!p_->rset) return;
  std::lock_guard<std::mutex> lock(p_->rmu);
  bool changed = p_->rdirty.exchange(false);
  changed = p_->prune_residency() || changed;
  if (!changed) return;
  p_->rset->commit();
  p_->rset->requestResidency();
}

bool Context::residency_set() const { return p_->rset != nullptr; }
uint64_t Context::residency_set_bytes() const {
  std::lock_guard<std::mutex> lock(p_->rmu);
  return p_->rbytes;
}
uint64_t Context::residency_skipped_bytes() const {
  std::lock_guard<std::mutex> lock(p_->rmu);
  return p_->rskipped;
}
const std::string& Context::device_name() const { return p_->name; }
const std::string& Context::architecture() const { return p_->arch; }
const Tuning& Context::tuning() const { return p_->tuning; }
bool Context::tuning_cache_found() const { return p_->cache != nullptr; }
double Context::compile_seconds() const { return p_->compile_s; }
std::size_t Context::pipelines_compiled() const { return p_->pipelines.size(); }
BufRef Context::dummy() const { return {p_->dummy.get(), 0}; }

std::string Context::codebook_in(int m) const {
  {
    std::lock_guard<std::mutex> lock(p_->mu);
    if (auto it = p_->cbin_memo.find(m); it != p_->cbin_memo.end()) return it->second;
  }
  std::string r = codebook_in_uncached(m);
  std::lock_guard<std::mutex> lock(p_->mu);
  p_->cbin_memo[m] = r;
  return r;
}

std::string Context::codebook_in_uncached(int m) const {
  const std::string& c = p_->tuning.codebook_in;
  if (c == "device" || c == "threadgroup") return c;
  if (c != "auto") fail("codebook_in must be auto, device or threadgroup, got " + c);
  if (const Json* v = nearest(p_->section("codebook_in"), m); v && v->is_string()) return v->as_string();
  return p_->arch.rfind("applegpu_g13", 0) == 0 ? "threadgroup" : "device";   // vq_matmul.CODEBOOK_TG_ARCHS
}

int Context::mma_min_batch(int m) const {
  {
    std::lock_guard<std::mutex> lock(p_->mu);
    if (auto it = p_->mmb_memo.find(m); it != p_->mmb_memo.end()) return it->second;
  }
  const int r = mma_min_batch_uncached(m);
  std::lock_guard<std::mutex> lock(p_->mu);
  p_->mmb_memo[m] = r;
  return r;
}

int Context::mma_min_batch_uncached(int m) const {
  if (p_->tuning.mma_min_batch > 0) return p_->tuning.mma_min_batch;
  // $SUC_MMA_MIN_BATCH: experiments (e.g. a large value keeps every batch on the vec lookup)
  if (const char* e = std::getenv("SUC_MMA_MIN_BATCH"); e && std::atoi(e) > 0) return std::atoi(e);
  const Json* s = p_->section("mma_min_batch");
  if (const Json* v = nearest(s ? s->find("choice") : nullptr, m); v && v->is_int()) return int(v->as_int());
  return 5;
}

std::pair<int, int> Context::rotation_block_config(std::size_t n) const {
  {
    std::lock_guard<std::mutex> lock(p_->mu);
    if (auto it = p_->rbc_memo.find(n); it != p_->rbc_memo.end()) return it->second;
  }
  const std::pair<int, int> r = rotation_block_config_uncached(n);
  std::lock_guard<std::mutex> lock(p_->mu);
  p_->rbc_memo[n] = r;
  return r;
}

std::pair<int, int> Context::rotation_block_config_uncached(std::size_t n) const {
  std::pair<int, int> qs = default_block_config(n);
  if (const Json* c = p_->rotation_choice(n); c && c->find("layout") && (*c)["layout"].as_string() == "blocks")
    qs = {int((*c)["q"].as_int()), int((*c)["sg"].as_int())};
  if (p_->tuning.block_q) qs.first = p_->tuning.block_q;
  if (p_->tuning.block_sg) qs.second = p_->tuning.block_sg;
  return qs;
}

bool Context::fuse_elementwise(const std::string& op) const {
  const Json* f = p_->section("fuse_elementwise");
  const Json* c = f ? f->find("choice") : nullptr;
  const Json* v = c ? c->find(op) : nullptr;
  return v && v->is_bool() && v->as_bool();
}

bool Context::rotation_layout_overridden(std::size_t n) const {
  const Json* c = p_->rotation_choice(n);
  if (!c || !c->find("layout")) return false;
  const std::string l = (*c)["layout"].as_string();
  return l != "blocks" && l != "row";
}

bool Context::rotation_row_layout(std::size_t n) const {
  const Json* c = p_->rotation_choice(n);
  return c && c->find("layout") && (*c)["layout"].as_string() == "row";
}

MTL::ComputePipelineState* Context::pipeline(const std::string& kernel, const std::string& targs) {
  const std::string key = kernel + "<" + targs + ">";
  std::lock_guard<std::mutex> lock(p_->mu);
  if (auto it = p_->pipelines.find(key); it != p_->pipelines.end()) return it->second;
  Pool pool;
  const auto t0 = std::chrono::steady_clock::now();
  const std::string host = "suc_k" + std::to_string(p_->counter++);
  const std::string src = p_->source + "\ntemplate [[host_name(\"" + host + "\")]] [[kernel]] decltype(" + key + ") " +
                          key + ";\n";
  MTL::CompileOptions* opt = MTL::CompileOptions::alloc()->init();
  opt->setFastMathEnabled(p_->tuning.fast_math);
  // language version: the OS default (the newest), as MLX on current systems; the MMA body uses a lambda
  NS::Error* err = nullptr;
  MTL::Library* lib = p_->device->newLibrary(nsstr(src), opt, &err);
  opt->release();
  if (!lib) {   // also on stderr: an exception thrown while a command encoder is open aborts before it is printed
    const std::string msg = "compiling " + key + " failed:\n" + (err ? str(err->localizedDescription()) : std::string("?"));
    std::fprintf(stderr, "%s\n", msg.c_str());
    fail(msg);
  }
  MTL::Function* fn = lib->newFunction(nsstr(host));
  lib->release();
  if (!fn) fail("function " + host + " not found");
  MTL::ComputePipelineState* ps = p_->device->newComputePipelineState(fn, &err);
  fn->release();
  if (!ps) fail("pipeline " + key + ": " + (err ? str(err->localizedDescription()) : std::string("?")));
  p_->compile_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  p_->pipelines.emplace(key, ps);
  return ps;
}

MTL::ComputePipelineState* Context::find_pipeline(const PsoKey& key) const {
  std::lock_guard<std::mutex> lock(p_->mu);
  auto it = p_->keyed.find(key);
  return it == p_->keyed.end() ? nullptr : it->second;
}

void Context::store_pipeline(const PsoKey& key, MTL::ComputePipelineState* ps) {
  std::lock_guard<std::mutex> lock(p_->mu);
  p_->keyed[key] = ps;
}

BufPtr Context::new_buffer(std::size_t bytes) const {
  MTL::Buffer* b = p_->device->newBuffer(std::max<std::size_t>(bytes, 16), MTL::ResourceStorageModeShared);
  if (!b) fail("cannot allocate " + std::to_string(bytes) + " bytes");
  std::memset(b->contents(), 0, b->length());
  if (bytes >= (64u << 10)) make_resident(b);
  return NS::TransferPtr(b);
}

BufPtr Context::upload(const void* data, std::size_t bytes) const {
  BufPtr b = new_buffer(bytes);
  if (bytes) std::memcpy(b->contents(), data, bytes);
  return b;
}

// ------------------------------------------------------------------------------------------ MappedFile

MappedFile::MappedFile(Context& ctx, const Suc1File& f, uint64_t max_view_bytes)
    : MappedFile(ctx, f, std::vector<std::pair<uint64_t, uint64_t>>{}, max_view_bytes) {}

MappedFile::MappedFile(Context& ctx, const Suc1File& f, const std::vector<std::pair<uint64_t, uint64_t>>& exclude,
                       uint64_t max_view_bytes)
    : file_(&f) {
  Pool pool;
  const std::size_t page = page_size();
  const uint8_t* base = f.bytes(0, f.file_size()).data();
  if (reinterpret_cast<uintptr_t>(base) % page) fail("the SUC1 mapping is not page-aligned");
  // mmap maps whole pages: the tail of the last page is addressable (zero), so the length can be rounded up.
  const uint64_t total = round_up(f.file_size(), page);
  // excluded whole pages (a page shared with other data stays mapped), merged
  std::vector<std::pair<uint64_t, uint64_t>> ex;
  for (auto [b, e] : exclude) {
    const uint64_t pb = round_up(b, page), pe = std::min<uint64_t>(e / page * page, total);
    if (pb < pe) ex.push_back({pb, pe});
  }
  std::sort(ex.begin(), ex.end());
  std::vector<std::pair<uint64_t, uint64_t>> merged;
  for (auto r : ex) {
    if (!merged.empty() && r.first <= merged.back().second) merged.back().second = std::max(merged.back().second, r.second);
    else merged.push_back(r);
  }
  uint64_t cur = 0;
  for (auto [b, e] : merged) {
    if (b > cur) add_views(ctx, cur, b, max_view_bytes);
    excluded_ += e - b;
    cur = e;
  }
  if (cur < total) add_views(ctx, cur, total, max_view_bytes);
  // A command buffer binding views beyond the GPU working set waits for residency forever (no error is reported).
  const uint64_t ws = ctx.device()->recommendedMaxWorkingSetSize();
  if (view_bytes() > ws)
    std::fprintf(stderr,
                 "warning: the GPU views of %s total %.2f GB > recommendedMaxWorkingSetSize %.2f GB; a command buffer "
                 "that binds all of them may never be scheduled (raise iogpu.wired_limit_mb or stream the experts)\n",
                 f.path().c_str(), view_bytes() / 1e9, ws / 1e9);
}

namespace {

// Tensor group of a name: the prefix through "layers.N." (all tensors of one transformer layer), else the name.
std::string_view tensor_group(std::string_view name) {
  for (std::size_t p = name.find("layers."); p != std::string_view::npos; p = name.find("layers.", p + 1)) {
    if (p && name[p - 1] != '.') continue;
    std::size_t q = p + 7;
    while (q < name.size() && name[q] >= '0' && name[q] <= '9') ++q;
    if (q > p + 7 && q < name.size() && name[q] == '.') return name.substr(0, q + 1);
  }
  return name;
}

void merge_ranges(std::vector<std::pair<uint64_t, uint64_t>>& r) {
  std::sort(r.begin(), r.end());
  std::vector<std::pair<uint64_t, uint64_t>> out;
  for (auto x : r) {
    if (!out.empty() && x.first < out.back().second) out.back().second = std::max(out.back().second, x.second);
    else out.push_back(x);
  }
  r.swap(out);
}

}  // namespace

std::vector<std::pair<uint64_t, uint64_t>> mapping_units(const Suc1File& f) {
  struct Seg {
    uint64_t b, e;
    std::size_t g;
  };
  std::unordered_map<std::string_view, std::size_t> gid;
  std::vector<Seg> segs;
  std::vector<std::pair<uint64_t, uint64_t>> cb_span;   // per group: [lowest, highest end) of its codebooks
  const auto& cbs = f.codebooks();
  for (const TensorDesc& t : f.tensors()) {
    const std::size_t g = gid.emplace(tensor_group(t.name), gid.size()).first->second;
    if (cb_span.size() <= g) cb_span.resize(g + 1, {~uint64_t(0), 0});
    for (const ArrayRecord& a : t.arrays)
      if (a.nbytes) segs.push_back({a.offset, a.offset + a.nbytes, g});
    for (uint64_t id : t.codebook_ids) {
      const CodebookDesc& c = cbs.at(id);
      cb_span[g].first = std::min(cb_span[g].first, c.offset);
      cb_span[g].second = std::max(cb_span[g].second, c.offset + c.nbytes);
    }
  }
  std::sort(segs.begin(), segs.end(), [](const Seg& a, const Seg& b) { return a.b < b.b; });
  // runs of consecutive segments of one group (a section holds the groups one after the other)
  std::vector<std::pair<uint64_t, uint64_t>> units;
  for (std::size_t i = 0; i < segs.size();) {
    std::size_t j = i;
    uint64_t e = segs[i].e;
    while (j + 1 < segs.size() && segs[j + 1].g == segs[i].g) e = std::max(e, segs[++j].e);
    units.push_back({segs[i].b, e});
    i = j + 1;
  }
  for (auto r : cb_span)
    if (r.first < r.second) units.push_back(r);
  merge_ranges(units);
  return units;
}

uint64_t MappedFile::view_bytes() const {
  uint64_t n = 0;
  for (const View& v : views_) n += v.length;
  return n;
}

void MappedFile::add_views(Context& ctx, uint64_t begin, uint64_t end, uint64_t max_view_bytes) {
  const std::size_t page = page_size();
  const uint8_t* base = file_->bytes(0, file_->file_size()).data();
  uint64_t max_len = ctx.device()->maxBufferLength();
  if (max_view_bytes) max_len = std::min(max_len, max_view_bytes);
  max_len = max_len / page * page;
  if (max_len < page) fail("max view size below one page");
  auto add = [&](uint64_t s, uint64_t e) {
    MTL::Buffer* b = ctx.device()->newBuffer(base + s, e - s, MTL::ResourceStorageModeShared, nullptr);
    if (!b) fail("newBufferWithBytesNoCopy failed for " + std::to_string(e - s) + " bytes at " + std::to_string(s));
    ctx.make_resident(b);
    views_.push_back({NS::TransferPtr(b), s, e - s});
  };
  mapped_ += end - begin;
  if (end - begin <= max_len) {
    add(begin, end);
    return;
  }
  // Greedy: extend the current view [start, ...) unit by unit while it fits in max_len; cut before the first unit that
  // does not fit (the end rounded up / the next start rounded down to pages: consecutive views share <= 1 page).
  if (units_.empty()) units_ = mapping_units(*file_);
  uint64_t start = begin, cut = begin;   // cut: end of the last unit that fits (page-rounded)
  for (auto [ub, ue] : units_) {
    ub = std::max(ub, begin), ue = std::min(ue, end);
    if (ub >= ue) continue;
    if (round_up(ue, page) - start <= max_len) {
      cut = round_up(ue, page);
      continue;
    }
    if (cut > start) {
      add(start, cut);
      start = std::min<uint64_t>(ub / page * page, cut);   // >= cut - page: units are disjoint and sorted
    }
    if (round_up(ue, page) - start > max_len)
      fail("a mapping unit of " + std::to_string(ue - ub) + " bytes at " + std::to_string(ub) +
           " does not fit in one GPU view of " + std::to_string(max_len) + " bytes (maxBufferLength)");
    cut = round_up(ue, page);
  }
  // the tail (bytes after the last unit: footer, padding; no unit there, so any cut is fine)
  if (end - start > max_len) {
    if (cut > start) add(start, cut), start = cut;
    for (; end - start > max_len; start += max_len) add(start, start + max_len);
  }
  add(start, end);
}

bool MappedFile::overlaps(uint64_t begin, uint64_t end) const {
  for (const View& v : views_)
    if (begin < v.start + v.length && v.start < end) return true;
  return false;
}

BufRef MappedFile::at(uint64_t off, uint64_t n) const {
  for (const View& v : views_)
    if (off >= v.start && off + n <= v.start + v.length) return {v.buf.get(), std::size_t(off - v.start)};
  fail("no mapped view holds bytes [" + std::to_string(off) + ", " + std::to_string(off + n) + ")");
}

// ------------------------------------------------------------------------------------------ Linear

namespace {

bool param_bool(const Json& p, std::string_view key, bool dflt) {
  const Json* v = p.find(key);
  return (v && !v->is_null()) ? v->as_bool() : dflt;
}

std::optional<uint64_t> seed_of(const Json& p, std::string_view key) {
  const Json* v = p.find(key);
  if (!v || v->is_null()) throw FormatError(std::string("rotation needs params.") + std::string(key));
  return v->as_uint();
}

// Rotation sizes this backend can run: n = m * 2^k (blocks layout for 2^k >= 32, dense fallback below).
bool rotation_ok(std::size_t n) { return hadamard_supported(n); }

void barrier(MTL::ComputeCommandEncoder* enc) { enc->memoryBarrier(MTL::BarrierScopeBuffers); }

std::string join(std::initializer_list<std::string> xs) {
  std::string s;
  for (const auto& x : xs) s += (s.empty() ? "" : ", ") + x;
  return s;
}
std::string I(long v) { return std::to_string(v); }

}  // namespace

// ------------------------------------------------------------------------------------------ tcq kernels

TcqLayout TcqLayout::from_params(const Json& p, std::size_t in, const std::string& name) {
  auto bad = [&](const std::string& m) -> TcqLayout { throw FormatError(name + ": " + m); };
  TcqLayout tl;
  const std::string& code = p["code"].as_string();
  if (code == "1mad") tl.code = 0;
  else if (code == "3inst") tl.code = 1;
  else if (code == "pair3") tl.code = 2;
  else return bad("tcq code '" + code + "' has no kernel (1mad / 3inst / pair3)");
  tl.V = int(p["V"].as_uint());
  tl.L = int(p["L"].as_uint());
  tl.T = int(p["T"].as_uint());
  const double k = p["k"].as_double(), gain = p["gain"].as_double();
  if (tl.V != 1 && tl.V != 2) return bad("tcq kernels support V in (1, 2)");
  if (tl.code == 2 && tl.V != 2) return bad("tcq code pair3 needs V = 2");
  tl.KV = int(std::lround(k * tl.V));
  const long tk = std::lround(double(tl.T) * k);
  tl.NW = int(tk / 32);
  if (const Json* kv = p.find("kv"); kv && int(kv->as_uint()) != tl.KV) return bad("tcq params kv != k * V");
  if (tl.KV < 1 || tl.L < tl.KV || tl.L > 24) return bad("tcq L / kv outside the kernels' range");
  if (tl.T <= 0 || in % std::size_t(tl.T) || tl.T % (32 * tl.V) || (tl.T / 32) % 4 || tk % 32 || !tl.NW)
    return bad("tcq shapes unsupported by the kernels (in % T, T % (32 V), (T / 32) % 4, T k % 32)");
  const Json* mu = p.find("code_mu");
  const Json* sd = p.find("code_sd");
  if (!mu || !sd || mu->size() != std::size_t(tl.V) || sd->size() != std::size_t(tl.V))
    return bad("tcq params need code_mu / code_sd (V entries)");
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.17g|%d|%d|%d|%d|%.17g", k, tl.L, tl.V, tl.T, tl.code, gain);
  tl.key = buf;
  for (int j = 0; j < tl.V; ++j) {
    const double m = mu->at(std::size_t(j)).as_double(), s = sd->at(std::size_t(j)).as_double();
    tl.cst[2 * j] = float(m);
    tl.cst[2 * j + 1] = float(gain / s);
    std::snprintf(buf, sizeof buf, "|%.17g,%.17g", m, s);
    tl.key += buf;
  }
  return tl;
}

bool tcq_mma(std::size_t bt, Path path) { return path == Path::Mma || (path == Path::Auto && bt >= kTcqMmaMinBatch); }

void encode_tcq(Context& ctx, MTL::ComputeCommandEncoder* enc, const TcqLayout& tl, int mode, BufRef x, BufRef words,
                BufRef scales, bool scales_f16, BufRef cst, BufRef eidx, BufRef eoff, BufRef y, std::size_t bt,
                std::size_t tiles, std::size_t out, std::size_t in, Act xa, Act ya, Path path) {
  if (!bt || bt > kTcqMaxBatchTile) fail("tcq: batch tiles must have 1.." + std::to_string(kTcqMaxBatchTile) + " rows");
  if (mode < 0 || mode > 2 || (mode == 0 && tiles != 1)) fail("tcq: bad launch mode");
  if (!tiles) return;
  // batched prefill: the mma tiles multiply float32 matrices (tcq_matmul rounds the decoded weights and x to the
  // activation dtype: on the tiny glm5 tcq file per-position KL up to 0.245 against token prefill)
  const bool f32mm = path == Path::Prefill;
  if (f32mm) path = Path::Auto;
  const bool mma = tcq_mma(bt, path);
  const int sgs = mma ? 4 : 8;                       // MMA_SG / VEC_SG
  const int rows = mma ? 8 * 4 : (bt <= 2 ? 4 : 2);  // 8 * MMA_RT / VEC_ROWS (halved above 2 rows)
  const PsoKey key{7001, {int64_t(out) | int64_t(in) << 32,
                          int64_t(tl.L) | int64_t(tl.V) << 8 | int64_t(tl.KV) << 16 | int64_t(tl.code) << 24 |
                              int64_t(tl.NW) << 32,
                          int64_t(tl.T), int64_t(bt),
                          int64_t(mma) | int64_t(mode) << 1 | int64_t(scales_f16) << 3 | int64_t(f32mm && mma) << 4,
                          int64_t(xa), int64_t(ya)}};
  enc->setComputePipelineState(ctx.pipeline(key, [&]() {
    const std::string name = std::string(mode ? "suc_tcq_switch_" : "suc_tcq_") + (mma ? "mma" : "vec") +
                             (mode == 2 ? "_tab" : "");
    // TXX, TSC, L, V, KV, T, NW, IN, OUT, BT, CODE (tcq_matmul_metal template order, after the input dtypes)
    std::string targs = join({act_msl(xa), scales_f16 ? "half" : "float", I(tl.L), I(tl.V), I(tl.KV), I(tl.T),
                              I(tl.NW), I(long(in)), I(long(out)), I(long(bt)), I(tl.code)});
    if (mma) targs += ", " + join({I(sgs), "4", I(long((bt + 7) / 8)), f32mm ? "float" : act_msl(xa), act_msl(ya)});
    else targs += ", " + join({I(sgs), I(rows), act_msl(ya)});
    return std::pair<std::string, std::string>{name, targs};
  }));
  enc->setBuffer(x.buffer, x.offset, 0);
  enc->setBuffer(words.buffer, words.offset, 1);
  enc->setBuffer(scales.buffer, scales.offset, 2);
  enc->setBuffer(cst.buffer, cst.offset, 3);
  if (mode == 0) {
    enc->setBuffer(y.buffer, y.offset, 4);
  } else {
    enc->setBuffer(eidx.buffer, eidx.offset, 4);
    enc->setBuffer(y.buffer, y.offset, 5);
    if (mode == 2) enc->setBuffer(eoff.buffer, eoff.offset, 6);
  }
  const std::size_t n_tg = (out + std::size_t(rows * sgs) - 1) / std::size_t(rows * sgs);
  enc->dispatchThreads(MTL::Size(n_tg * sgs * 32, tiles, 1), MTL::Size(sgs * 32, 1, 1));
}

void encode_tcq_mm(Context& ctx, MTL::ComputeCommandEncoder* enc, const TcqLayout& tl, int mode, BufRef x,
                   BufRef words, BufRef scales, bool scales_f16, BufRef cst, BufRef teidx, BufRef eoff, BufRef y,
                   std::size_t n, std::size_t tiles, std::size_t out, std::size_t in, Act act, Act ya,
                   BufRef runs, std::size_t nruns, std::size_t run_rows) {
  if (in % 32) fail("tcq mm: in_features not a multiple of 32");
  if (!n || (mode && !tiles)) return;
  const bool rn = runs && mode;
  const std::size_t tr = rn ? std::max(run_rows, n) : n;   // rows per threadgroup row
  const int bm = mode ? (tr <= 8 ? 8 : tr <= 16 ? 16 : tr <= 32 ? 32 : 64) : (n >= 64 ? 64 : n >= 32 ? 32 : n >= 16 ? 16 : 8);
  if (mode && tr > 64) fail("tcq mm: tiles / runs of more than 64 rows");
  const PsoKey key{7002, {int64_t(out) | int64_t(in) << 32,
                          int64_t(tl.L) | int64_t(tl.V) << 8 | int64_t(tl.KV) << 16 | int64_t(tl.code) << 24 |
                              int64_t(tl.NW) << 32,
                          int64_t(tl.T), int64_t(bm),
                          int64_t(mode) | int64_t(scales_f16) << 3 | int64_t(rn) << 4, int64_t(act), int64_t(ya)}};
  enc->setComputePipelineState(ctx.pipeline(key, [&]() {
    return std::pair<std::string, std::string>{
        "suc_tcq_mm", join({act_msl(act), I(bm), scales_f16 ? "half" : "float", I(tl.L), I(tl.V), I(tl.KV), I(tl.T),
                            I(tl.NW), I(long(in)), I(long(out)), I(tl.code), I(mode), act_msl(ya), I(rn)})};
  }));
  enc->setBuffer(x.buffer, x.offset, 0);
  enc->setBuffer(words.buffer, words.offset, 1);
  enc->setBuffer(scales.buffer, scales.offset, 2);
  enc->setBuffer(cst.buffer, cst.offset, 3);
  BufRef d = ctx.dummy();
  BufRef te = mode ? teidx : d, eo = mode == 2 ? eoff : d;
  enc->setBuffer(te.buffer, te.offset, 4);
  enc->setBuffer(y.buffer, y.offset, 5);
  enc->setBuffer(eo.buffer, eo.offset, 6);
  const uint32_t n32 = uint32_t(n);
  enc->setBytes(&n32, 4, 7);
  BufRef ru = rn ? runs : d;
  enc->setBuffer(ru.buffer, ru.offset, 8);
  const std::size_t gx = rn ? nruns : mode ? tiles : (n + bm - 1) / bm;
  enc->dispatchThreadgroups(MTL::Size(gx, (out + 63) / 64, 1), MTL::Size(128, 1, 1));
}

bool Linear::supported(const TensorDesc& t, std::string* why) {
  auto no = [&](const std::string& w) {
    if (why) *why = w;
    return false;
  };
  const bool tcq = t.representation_type == "tcq";
  if (t.representation_type != "vq" && t.representation_type != "additive" && !tcq)
    return no("representation " + t.representation_type + " has no GPU kernel (vq / additive / tcq only)");
  if (t.shape.size() != 2) return no("not 2-D");
  const std::string& rt = t.residual_type;
  if (rt != "none" && rt != "sparse_fp16" && rt != "sparse_int8" && rt != "columns_fp16" && rt != "columns_int8")
    return no("residual type " + rt);
  if (!tcq && (t.codebook_ids.empty() || !t.index_bits || *t.index_bits < 1 || *t.index_bits > 32))
    return no("no codebooks / bits");
  if (!t.params) return no("no params");
  const Json& p = t.has_residual() ? (*t.params)["base"] : *t.params;
  if (tcq) {
    try {
      TcqLayout::from_params(p, t.shape[1], t.name);
    } catch (const std::exception& e) {
      return no(e.what());
    }
  }
  if (param_bool(p, "rotate", false)) {
    if (param_bool(p, "rotate_in", true) && !rotation_ok(t.shape[1])) return no("input rotation size unsupported");
    // tcq tensors are stored with rotate_out = false (and default to it, as SucLinear._from_tcq)
    if (param_bool(p, "rotate_out", !tcq) && !rotation_ok(t.shape[0])) return no("output rotation size unsupported");
  }
  return true;
}

Linear Linear::from_file(Context& ctx, const MappedFile& mf, const TensorDesc& t) {
  std::string why;
  if (!supported(t, &why)) throw FormatError(t.name + ": " + why);
  const Suc1File& f = mf.file();
  Linear L;
  L.name_ = t.name;
  L.out_ = t.shape[0];
  L.in_ = t.shape[1];
  const bool residual = t.has_residual();
  const Json& p = residual ? (*t.params)["base"] : *t.params;
  const std::string pre = residual ? "base." : "";
  if (t.representation_type == "tcq") {
    // streams (out · in/T, NW) packed at kv bits: viewed as (out, in/T · NW) u32 they are the kernel layout (zero-copy)
    L.tcq_ = true;
    L.tcq_l_ = TcqLayout::from_params(p, L.in_, t.name);
    L.bits_ = unsigned(L.tcq_l_.KV);
    L.words_ = L.tcq_l_.words(L.in_);
    const ArrayRecord& ch = t.array(pre + "chunks");
    const std::size_t nseq = L.in_ / std::size_t(L.tcq_l_.T);
    if (!ch.packed || ch.bits != L.bits_ || ch.shape != std::vector<uint64_t>{L.out_ * nseq, uint64_t(L.tcq_l_.NW)} ||
        ch.nbytes != L.out_ * L.words_ * 4)
      throw FormatError(t.name + ": stored tcq streams do not match the codec params");
    L.packed_ = mf.at(ch);
    const ArrayRecord& rs = t.array(pre + "row_scales");
    if (rs.dtype != DType::F16 || rs.shape != std::vector<uint64_t>{L.out_}) throw FormatError(t.name + ": row_scales");
    L.scales_ = mf.at(rs);
    L.scales_f16_ = true;
    L.tcq_cst_ = ctx.upload(L.tcq_l_.cst, sizeof L.tcq_l_.cst);
    L.copied_bytes_ += sizeof L.tcq_l_.cst;
    L.zero_copy_bytes_ += ch.nbytes + rs.nbytes;
    L.weight_bytes_ = ch.nbytes + rs.nbytes;
  } else {
  L.M_ = t.codebook_ids.size();
  const CodebookDesc& c0 = f.codebooks().at(t.codebook_ids[0]);
  L.K_ = c0.entries;
  L.G_ = c0.vector_length;
  L.bits_ = unsigned(*t.index_bits);
  if (L.G_ == 0 || L.in_ % L.G_) throw FormatError(t.name + ": in_features not divisible by G");
  if (L.K_ > (uint64_t(1) << L.bits_)) throw FormatError(t.name + ": K exceeds 2^index_bits");
  const std::size_t cb_bytes = L.M_ * L.K_ * L.G_ * 2;
  if (cb_bytes > kCodebookLimit) throw FormatError(t.name + ": codebooks exceed the 16 KB ceiling (D4)");

  // codebooks: zero-copy when the M segments are adjacent f16 arrays in id order, else packed into one buffer
  bool adjacent = true;
  for (std::size_t m = 0; m < L.M_; ++m) {
    const CodebookDesc& c = f.codebooks().at(t.codebook_ids[m]);
    if (c.entries != L.K_ || c.vector_length != L.G_) throw FormatError(t.name + ": codebooks of different shapes");
    if (c.dtype != DType::F16) throw FormatError(t.name + ": codebooks must be f16");
    adjacent = adjacent && c.offset == c0.offset + m * L.K_ * L.G_ * 2;
  }
  if (adjacent) {
    L.codebooks_ = mf.at(c0.offset, cb_bytes);
    L.codebooks_zero_copy_ = true;
    L.zero_copy_bytes_ += cb_bytes;
  } else {
    std::vector<uint8_t> cb(cb_bytes);
    for (std::size_t m = 0; m < L.M_; ++m) {
      auto s = f.payload(f.codebooks().at(t.codebook_ids[m]));
      std::memcpy(cb.data() + m * L.K_ * L.G_ * 2, s.data(), s.size());
    }
    L.owned_.push_back(ctx.upload(cb.data(), cb.size()));
    L.codebooks_ = {L.owned_.back().get(), 0};
    L.copied_bytes_ += cb_bytes;
  }
  // packed indices (M·out rows of words) and f16 row scales: zero-copy
  const ArrayRecord& qi = t.array(pre + "indices");
  const std::size_t ng = L.in_ / L.G_;
  L.words_ = words_per_row(ng, L.bits_);
  if (!qi.packed || qi.bits != L.bits_ || qi.shape != std::vector<uint64_t>{L.M_ * L.out_, L.words_})
    throw FormatError(t.name + ": unexpected packed index layout");
  L.packed_ = mf.at(qi);
  const ArrayRecord& rs = t.array(pre + "row_scales");
  if (rs.dtype != DType::F16 || rs.shape != std::vector<uint64_t>{L.out_}) throw FormatError(t.name + ": row_scales");
  L.scales_ = mf.at(rs);
  L.scales_f16_ = true;
  L.zero_copy_bytes_ += qi.nbytes + rs.nbytes;
  L.weight_bytes_ = cb_bytes + qi.nbytes + rs.nbytes;
  }

  // rotations (signs re-derived from the seeds, as SucLinear does; tcq: rotate_out defaults to false)
  if (param_bool(p, "rotate", false)) {
    if (param_bool(p, "rotate_in", true)) L.seed_in_ = seed_of(p, "seed_in");
    if (param_bool(p, "rotate_out", !L.tcq_)) L.seed_out_ = seed_of(p, "seed_out");
  }
  auto signs = [&](std::size_t n, uint64_t seed, BufPtr& s, BufPtr& hm) {
    const std::vector<float> v = random_signs(n, seed);
    s = ctx.upload(v.data(), v.size() * 4);
    const auto hf = hadamard_factor(n);
    const std::vector<float> h = hadamard_hm(hf.m);
    hm = ctx.upload(h.data(), h.size() * 4);
    L.copied_bytes_ += v.size() * 4;
  };
  if (L.seed_in_) signs(L.in_, *L.seed_in_, L.signs_in_, L.hm_in_);
  if (L.seed_out_) signs(L.out_, *L.seed_out_, L.signs_out_, L.hm_out_);

  if (!residual) return L;
  // residual, execution layout of SucResidualLinear: CSR rows from the sorted flat positions, cols u16 (in <= 65536)
  const Json& tp = *t.params;
  const bool u16 = L.in_ <= 65536;
  L.rcols_u16_ = L.ccols_u16_ = u16;
  float rscale = 1.0f;
  if (tp["n"].as_uint() > 0) {
    const auto pos = array_as_u32(f, t.array("res_pos"));
    const ArrayRecord& rv = t.array("res_val");
    if (rv.logical_numel() != pos.size()) throw FormatError(t.name + ": res_pos / res_val length mismatch");
    if (rv.dtype != DType::I8 && rv.dtype != DType::F16) throw FormatError(t.name + ": res_val dtype");
    L.rvals_i8_ = rv.dtype == DType::I8;
    std::vector<uint32_t> row_ptr(L.out_ + 1, 0);
    std::vector<uint32_t> c32(pos.size());
    std::vector<uint16_t> c16(u16 ? pos.size() : 0);
    for (std::size_t i = 0; i < pos.size(); ++i) {
      const uint64_t r = pos[i] / L.in_, c = pos[i] % L.in_;
      if (r >= L.out_ || (i && pos[i] <= pos[i - 1])) throw FormatError(t.name + ": res_pos not sorted / in range");
      ++row_ptr[r + 1];
      c32[i] = uint32_t(c);
      if (u16) c16[i] = uint16_t(c);
    }
    for (std::size_t r = 0; r < L.out_; ++r) row_ptr[r + 1] += row_ptr[r];
    L.owned_.push_back(ctx.upload(row_ptr.data(), row_ptr.size() * 4));
    L.row_ptr_ = {L.owned_.back().get(), 0};
    if (u16) L.owned_.push_back(ctx.upload(c16.data(), c16.size() * 2));
    else L.owned_.push_back(ctx.upload(c32.data(), c32.size() * 4));
    L.rcols_ = {L.owned_.back().get(), 0};
    L.rvals_ = mf.at(rv);
    if (const ArrayRecord* sc = t.find_array("res_scale")) rscale = array_as_f32(f, *sc).at(0);
    L.n_csr_ = pos.size();
    const std::size_t cb = row_ptr.size() * 4 + pos.size() * (u16 ? 2 : 4);
    L.copied_bytes_ += cb;
    L.zero_copy_bytes_ += rv.nbytes;
    L.weight_bytes_ += cb + rv.nbytes;
  }
  if (const ArrayRecord* rc = t.find_array("res_cols"); rc && rc->logical_numel()) {
    const auto cols = array_as_u32(f, *rc);
    for (std::size_t j = 0; j < cols.size(); ++j)
      if (cols[j] >= L.in_ || (j && cols[j] <= cols[j - 1])) throw FormatError(t.name + ": res_cols not sorted / in range");
    if (cols.size() > kXgFloats) throw FormatError(t.name + ": more residual columns than the gather buffer (6144)");
    const ArrayRecord& cv = t.array("res_colval");
    if (cv.shape != std::vector<uint64_t>{cols.size(), L.out_}) throw FormatError(t.name + ": res_colval shape");
    if (cv.dtype != DType::I8 && cv.dtype != DType::F16) throw FormatError(t.name + ": res_colval dtype");
    L.cval_i8_ = cv.dtype == DType::I8;
    if (u16) {
      std::vector<uint16_t> c16(cols.begin(), cols.end());
      L.owned_.push_back(ctx.upload(c16.data(), c16.size() * 2));
    } else {
      L.owned_.push_back(ctx.upload(cols.data(), cols.size() * 4));
    }
    L.ccols_ = {L.owned_.back().get(), 0};
    L.cval_ = mf.at(cv);
    std::vector<float> cs(cols.size(), 1.0f);
    if (const ArrayRecord* sc = t.find_array("res_colscale")) cs = array_as_f32(f, *sc);
    if (cs.size() != cols.size()) throw FormatError(t.name + ": res_colscale size");
    L.owned_.push_back(ctx.upload(cs.data(), cs.size() * 4));
    L.cscale_ = {L.owned_.back().get(), 0};
    L.n_cols_ = cols.size();
    const std::size_t cb = cols.size() * ((u16 ? 2 : 4) + 4);
    L.copied_bytes_ += cb;
    L.zero_copy_bytes_ += cv.nbytes;
    L.weight_bytes_ += cb + cv.nbytes;
  }
  L.rscale_ = ctx.upload(&rscale, 4);
  const uint32_t meta[5] = {0, uint32_t(L.out_), 0, uint32_t(L.n_cols_), 0};
  L.meta_ = ctx.upload(meta, sizeof meta);
  return L;
}

Linear Linear::from_host(Context& ctx, const HostLayer& h) {
  Linear L;
  L.name_ = "host";
  L.out_ = h.out;
  L.in_ = h.in;
  L.M_ = h.M;
  L.K_ = h.K;
  L.G_ = h.G;
  L.bits_ = h.bits;
  if (!h.G || h.in % h.G || !h.bits || h.bits > 32 || h.K > (uint64_t(1) << h.bits)) fail("invalid host layer shape");
  L.words_ = words_per_row(h.in / h.G, h.bits);
  if (h.codebooks.size() != h.M * h.K * h.G || h.packed.size() != h.M * h.out * L.words_ || h.row_scales.size() != h.out)
    fail("host layer arrays do not match its shape");
  if (h.codebooks.size() * 2 > kCodebookLimit) fail("codebooks exceed the 16 KB ceiling (D4)");
  auto own = [&](const void* d, std::size_t n) {
    L.owned_.push_back(ctx.upload(d, n));
    L.copied_bytes_ += n;
    return BufRef{L.owned_.back().get(), 0};
  };
  L.codebooks_ = own(h.codebooks.data(), h.codebooks.size() * 2);
  L.packed_ = own(h.packed.data(), h.packed.size() * 4);
  L.scales_ = own(h.row_scales.data(), h.row_scales.size() * 4);
  L.scales_f16_ = false;
  L.weight_bytes_ = L.copied_bytes_;
  L.seed_in_ = h.seed_in;
  L.seed_out_ = h.seed_out;
  auto signs = [&](std::size_t n, uint64_t seed, BufPtr& s, BufPtr& hm) {
    if (!rotation_ok(n)) fail("unsupported rotation size " + std::to_string(n));
    const std::vector<float> v = random_signs(n, seed);
    s = ctx.upload(v.data(), v.size() * 4);
    const std::vector<float> hh = hadamard_hm(hadamard_factor(n).m);
    hm = ctx.upload(hh.data(), hh.size() * 4);
  };
  if (L.seed_in_) signs(L.in_, *L.seed_in_, L.signs_in_, L.hm_in_);
  if (L.seed_out_) signs(L.out_, *L.seed_out_, L.signs_out_, L.hm_out_);
  const bool u16 = h.in <= 65536;
  L.rcols_u16_ = L.ccols_u16_ = u16;
  if (!h.res_cols.empty()) {
    if (h.res_row_ptr.size() != h.out + 1) fail("res_row_ptr size");
    L.n_csr_ = h.res_cols.size();
    L.row_ptr_ = own(h.res_row_ptr.data(), h.res_row_ptr.size() * 4);
    if (u16) {
      std::vector<uint16_t> c(h.res_cols.begin(), h.res_cols.end());
      L.rcols_ = own(c.data(), c.size() * 2);
    } else {
      L.rcols_ = own(h.res_cols.data(), h.res_cols.size() * 4);
    }
    L.rvals_i8_ = !h.res_val_i8.empty();
    if (L.rvals_i8_) L.rvals_ = own(h.res_val_i8.data(), h.res_val_i8.size());
    else L.rvals_ = own(h.res_val_f16.data(), h.res_val_f16.size() * 2);
  }
  if (!h.col_cols.empty()) {
    L.n_cols_ = h.col_cols.size();
    if (u16) {
      std::vector<uint16_t> c(h.col_cols.begin(), h.col_cols.end());
      L.ccols_ = own(c.data(), c.size() * 2);
    } else {
      L.ccols_ = own(h.col_cols.data(), h.col_cols.size() * 4);
    }
    L.cval_i8_ = !h.col_val_i8.empty();
    if (L.cval_i8_) L.cval_ = own(h.col_val_i8.data(), h.col_val_i8.size());
    else L.cval_ = own(h.col_val_f16.data(), h.col_val_f16.size() * 2);
    std::vector<float> cs = h.col_scale.empty() ? std::vector<float>(L.n_cols_, 1.0f) : h.col_scale;
    L.cscale_ = own(cs.data(), cs.size() * 4);
  }
  L.weight_bytes_ = L.copied_bytes_;
  L.rscale_ = ctx.upload(&h.res_scale, 4);
  const uint32_t meta[5] = {0, uint32_t(L.out_), 0, uint32_t(L.n_cols_), 0};
  L.meta_ = ctx.upload(meta, sizeof meta);
  return L;
}

std::string Linear::path_for(const Context& ctx, std::size_t B, Path path) const {
  if (path == Path::Prefill) path = Path::Auto;
  std::string s;
  const std::size_t tile = tcq_ ? kTcqMaxBatchTile : std::size_t(ctx.tuning().max_batch_tile);
  for (std::size_t b0 = 0; b0 < B; b0 += tile) {
    const std::size_t bt = std::min(tile, B - b0);
    const bool mma = tcq_ ? tcq_mma(bt, path)
                          : path == Path::Mma || (path == Path::Auto && int(bt) >= ctx.mma_min_batch(int(M_)));
    s += (s.empty() ? "" : "+") + std::string(mma ? "mma" : "vec");
  }
  return s;
}

void Linear::encode_kernel(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t B, Act xa,
                           Act ya, Path path) const {
  const bool prefill = path == Path::Prefill;
  if (prefill) path = Path::Auto;
  if (tcq_) {   // = tcq_matmul_metal: batch tiles of at most MAX_BATCH_TILE rows, one launch each
    // $SUC_TCQ_TILE: smaller tiles (diagnostics; the 16-row mma tile is slow on the M1 Ultra for some layouts)
    static const std::size_t tt = [] { const char* e = std::getenv("SUC_TCQ_TILE"); const int v = e ? std::atoi(e) : 0; return v >= 1 && v <= 16 ? std::size_t(v) : kTcqMaxBatchTile; }();
    // batched prefill with more rows than one tile: the GEMM kernel over all rows ($SUC_TCQ_MM=0: the tiles)
    static const bool mm = [] { const char* e = std::getenv("SUC_TCQ_MM"); return !(e && e[0] == '0'); }();
    if (prefill && mm && B > kTcqMaxBatchTile && in_ % 32 == 0 && xa != Act::F32) {
      encode_tcq_mm(ctx, enc, tcq_l_, 0, x, packed_, scales_, scales_f16_, {tcq_cst_.get(), 0}, {}, {}, y, B, 1, out_, in_,
                    xa, ya);
      return;
    }
    for (std::size_t b0 = 0; b0 < B; b0 += tt) {
      const std::size_t bt = std::min(tt, B - b0);
      encode_tcq(ctx, enc, tcq_l_, 0, x.at(b0 * in_ * act_size(xa)), packed_, scales_, scales_f16_,
                 {tcq_cst_.get(), 0}, {}, {}, y.at(b0 * out_ * act_size(ya)), bt, 1, out_, in_, xa, ya,
                 prefill ? Path::Prefill : path);
    }
    return;
  }
  const Tuning& tu = ctx.tuning();
  const std::size_t tile = std::size_t(tu.max_batch_tile);
  const int m = int(M_);
  const bool cbtg = ctx.codebook_in(m) == "threadgroup";
  const char* ts = scales_f16_ ? "half" : "float";
  for (std::size_t b0 = 0; b0 < B; b0 += tile) {
    const std::size_t bt = std::min(tile, B - b0);
    const bool mma = path == Path::Mma || (path == Path::Auto && int(bt) >= ctx.mma_min_batch(m));
    const int sgs = mma ? tu.mma_sg : tu.lookup_sg;
    const int rows = mma ? 8 * tu.mma_rt : lookup_rows(bt);
    if (mma && cbtg && M_ * K_ * G_ * 2 > kTgMemoryLimit) fail("mma path: codebook staging exceeds 32 KB");
    // every template argument is a function of these (no object pointers: objects die and addresses are reused)
    const PsoKey key{1, {int64_t(out_), int64_t(in_),
                         int64_t(G_) | int64_t(K_) << 8 | int64_t(bits_) << 32 | int64_t(M_) << 40 |
                             int64_t(scales_f16_) << 48,
                         int64_t(bt), int64_t(mma), int64_t(xa), int64_t(ya)}};
    MTL::ComputePipelineState* ps = ctx.pipeline(key, [&]() {
      // T, BT, OUT, IN, NG, G, K, BITS, WORDS (vq_matmul._launch template order)
      const std::string head = join({act_msl(xa), I(long(bt)), I(long(out_)), I(long(in_)), I(long(in_ / G_)),
                                     I(long(G_)), I(long(K_)), I(long(bits_)), I(long(words_))});
      if (mma) {
        const char* tt = (xa == Act::F16 && M_ == 1) ? "half" : "float";
        return std::pair<std::string, std::string>{
            "suc_lookup_mma", head + ", " + join({I(sgs), I(m), I(tu.mma_rt), I(long((bt + 7) / 8)),
                                                  I(mma_kcg(int(G_), int(bits_), tu.mma_kc)), tt, I(cbtg),
                                                  act_msl(ya), ts})};
      }
      return std::pair<std::string, std::string>{
          "suc_lookup_vec",
          head + ", " + join({I(sgs), I(m), I(rows), I(tu.lookup_vec), I(cbtg), "0", act_msl(ya), ts})};
    });
    enc->setComputePipelineState(ps);
    enc->setBuffer(x.buffer, x.offset + b0 * in_ * act_size(xa), 0);
    enc->setBuffer(codebooks_.buffer, codebooks_.offset, 1);
    enc->setBuffer(packed_.buffer, packed_.offset, 2);
    enc->setBuffer(scales_.buffer, scales_.offset, 3);
    enc->setBuffer(y.buffer, y.offset + b0 * out_ * act_size(ya), 4);
    const std::size_t n_tg = (out_ + std::size_t(sgs * rows) - 1) / std::size_t(sgs * rows);
    enc->dispatchThreads(MTL::Size(n_tg * sgs * 32, 1, 1), MTL::Size(sgs * 32, 1, 1));
  }
}

void encode_rht(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t B, std::size_t n,
                BufRef signs, BufRef hm, bool inverse, Act xa, Act ya) {
  const HadamardFactor f = hadamard_factor(n);
  if (!f.m) fail("unsupported rotation size " + std::to_string(n));
  const int m = int(f.m), p = int(f.p);
  if (p >= 32 && ctx.rotation_row_layout(n)) {
    encode_rot_row(ctx, enc, x, signs, {y}, {n}, B, inverse, xa, ya);
    return;
  }
  if (p >= 32) {   // rotation.supported: the blocks layout (rotation._blocks_plan / blocks_shape)
    const auto [qmax, sgmax] = ctx.rotation_block_config(n);
    const int q = std::min(qmax, p);
    const int sg = std::min({sgmax, m * p / q, kTgFloats / q});
    const int t = 32 * std::max(1, sg);
    const int ge = m * p / q;
    const PsoKey key{2, {int64_t(n), int64_t(inverse), int64_t(xa), int64_t(ya), q, t}};
    enc->setComputePipelineState(ctx.pipeline(key, [&]() {
      return std::pair<std::string, std::string>{
          "suc_rot_blocks", join({act_msl(xa), act_msl(ya), I(q), I(t), I(q / 32), I(inverse), I(long(n)),
                                  I((t / 32) * q), I(m), I(p), "0", "0", I(ge)})};
    }));
    enc->setBuffer(x.buffer, x.offset, 0);
    enc->setBuffer(signs.buffer, signs.offset, 1);
    enc->setBuffer(y.buffer, y.offset, 2);
    enc->dispatchThreads(MTL::Size(t, ge, B), MTL::Size(t, 1, 1));
    return;
  }
  // small 2^k factor (Python: MLX hadamard_transform fallback): dense per-element kernel
  const std::string targs = join({act_msl(xa), act_msl(ya), I(m), I(p), I(inverse)});
  enc->setComputePipelineState(ctx.pipeline("suc_rot_dense", targs));
  enc->setBuffer(x.buffer, x.offset, 0);
  if (!hm) fail("dense rotation fallback needs H_m");
  enc->setBuffer(signs.buffer, signs.offset, 1);
  enc->setBuffer(y.buffer, y.offset, 2);
  enc->setBuffer(hm.buffer, hm.offset, 3);
  const std::size_t tx = std::min<std::size_t>(256, n);
  enc->dispatchThreads(MTL::Size(n, B, 1), MTL::Size(tx, 1, 1));
}

void encode_rot_row(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef signs,
                    const std::vector<BufRef>& outs, const std::vector<std::size_t>& sizes, std::size_t B, bool inverse,
                    Act xa, Act ya, BufRef sidx) {
  const std::size_t S = sizes.size();
  if (S < 1 || S > 3) fail("row-layout rotation: 1 to 3 segments");
  const bool split = outs.size() > 1, rows = sidx.buffer != nullptr;
  if (split && outs.size() != S) fail("row-layout rotation: one output per segment expected");
  if (rows && S != 1) fail("row-layout rotation: per-row signs need one segment");
  std::vector<HadamardFactor> fs;
  std::size_t pmin = 0, ntot = 0;
  for (std::size_t n : sizes) {
    const HadamardFactor f = hadamard_factor(n);
    if (!f.m || f.p < 32) fail("row-layout rotation: unsupported size " + std::to_string(n));
    fs.push_back(f);
    pmin = pmin ? std::min(pmin, f.p) : f.p;
    ntot += n;
  }
  // rotation.threads_and_vec(ps, THREADS_MAX = 256, VEC = 4)
  const std::size_t t = std::min<std::size_t>(256, std::max<std::size_t>(32, pmin / 4));
  std::string targs = join({act_msl(xa), act_msl(ya), I(long(t)), I(inverse), "4096", "1", I(long(ntot))});
  std::size_t off = 0, ho = 0;
  for (const auto& f : fs) {
    targs += ", " + join({I(long(f.m)), I(long(f.p)), I(long(std::min<std::size_t>(4, f.p / t))), I(long(off)),
                          I(long(ho))});
    off += f.m * f.p;
    ho += f.m * f.m;
  }
  const std::string kname = "suc_rot_row" + (S > 1 ? std::to_string(S) : std::string()) + (split ? "_split" : "") +
                            (rows ? "_rows" : "");
  enc->setComputePipelineState(ctx.pipeline(kname, targs));
  const BufRef d = ctx.dummy();
  enc->setBuffer(x.buffer, x.offset, 0);
  enc->setBuffer(signs.buffer, signs.offset, 1);
  enc->setBuffer(d.buffer, d.offset, 2);   // hm: unused with HM_CONST (H_m from the compile-time tables)
  int o = 3;
  if (rows) enc->setBuffer(sidx.buffer, sidx.offset, o++);
  for (const BufRef& y : outs) enc->setBuffer(y.buffer, y.offset, o++);
  enc->dispatchThreads(MTL::Size(t, S, B), MTL::Size(t, 1, 1));
}

// = compressed_model.structured_residual_metal(x2, y32, meta, ..., segments=1, out_dtype)
// $SUC_RESIDUAL_EXACT=1 (experiments): every structured-residual launch keeps the one-row reduction split JS.
bool residual_exact_env() {
  static const bool v = [] { const char* e = std::getenv("SUC_RESIDUAL_EXACT"); return e && e[0] == '1'; }();
  return v;
}

void Linear::encode_residual(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y32, BufRef y,
                             std::size_t B, Act act, bool exact) const {
  const bool has_cols = n_cols_ > 0, has_csr = n_csr_ > 0;
  const std::size_t ctot = has_cols ? n_cols_ : 0;
  const int rq = out_ % 4 == 0 ? 4 : 1;
  auto fits = [&](std::size_t bt, std::size_t js) { return 4 * (bt * ctot + js * 32 * rq * bt) <= kStructTgBytes; };
  std::size_t bt = B == 1 ? 1 : B <= 4 ? 4 : 8;
  while (bt > 1 && bt * ctot > kXgFloats) bt /= 2;
  std::size_t js = 8;
  if (exact || residual_exact_env()) {
    // A row's sum is split over JS lanes (entries j ≡ js mod JS) and the JS partial sums are added in order: the
    // one-row launch's JS (bt = 1) is kept and the rows per threadgroup shrink until it fits instead.
    while (js > 1 && !fits(1, js)) js /= 2;
    while (bt > 1 && !fits(bt, js)) bt /= 2;
  } else {
    while (js > 1 && !fits(bt, js)) js /= 2;
  }
  const std::size_t red = js > 1 ? js * 32 * rq * bt : 1;
  const char* tcc = has_cols ? (ccols_u16_ ? "uint16_t" : "uint32_t") : "uint32_t";
  const char* tcv = has_cols ? (cval_i8_ ? "int8_t" : "half") : "int8_t";
  const char* trc = has_csr ? (rcols_u16_ ? "uint16_t" : "uint32_t") : "uint32_t";
  const char* trv = has_csr ? (rvals_i8_ ? "int8_t" : "half") : "int8_t";
  const PsoKey key{3, {int64_t(in_), int64_t(out_), int64_t(ctot), int64_t(bt),
                       int64_t(has_cols) | int64_t(has_csr) << 1 | int64_t(ccols_u16_) << 2 | int64_t(cval_i8_) << 3 |
                           int64_t(rcols_u16_) << 4 | int64_t(rvals_i8_) << 5,
                       int64_t(act)}};
  enc->setComputePipelineState(ctx.pipeline(key, [&]() {
    return std::pair<std::string, std::string>{
        "suc_structured_residual",
        join({act_msl(act), tcc, tcv, trc, trv, I(long(in_)), I(long(out_)), "1", I(long(std::max<std::size_t>(ctot, 1))),
              I(long(bt)), I(long(std::max<std::size_t>(bt * ctot, 1))), I(rq), I(long(js)), I(long(red)),
              has_cols ? "true" : "false", has_csr ? "true" : "false", act_msl(act)})};
  }));
  const BufRef d = ctx.dummy();
  auto set = [&](BufRef r, int i) { enc->setBuffer(r.buffer ? r.buffer : d.buffer, r.buffer ? r.offset : 0, i); };
  set(x, 0);
  set(y32, 1);
  set({meta_.get(), 0}, 2);
  set(has_cols ? ccols_ : BufRef{}, 3);
  set(has_cols ? cval_ : BufRef{}, 4);
  set(has_cols ? cscale_ : BufRef{}, 5);
  set(has_csr ? row_ptr_ : BufRef{}, 6);
  set(has_csr ? rcols_ : BufRef{}, 7);
  set(has_csr ? rvals_ : BufRef{}, 8);
  set({rscale_.get(), 0}, 9);
  set(y, 10);
  const int shape[2] = {int(B), int(in_)};
  enc->setBytes(shape, sizeof shape, 11);
  const std::size_t n_tg = (out_ + 32 * rq - 1) / (32 * rq);
  enc->dispatchThreads(MTL::Size(n_tg * 32 * js, (B + bt - 1) / bt, 1), MTL::Size(32 * js, 1, 1));
}

namespace {
void ensure(Context& ctx, BufPtr& b, std::size_t& cap, std::size_t bytes) {
  if (cap < bytes) {
    b = ctx.new_buffer(bytes);
    cap = bytes;
  }
}
}  // namespace

void Linear::encode(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t B, Act act,
                    Workspace& ws, Path path) const {
  BufRef xr = x;
  if (seed_in_) {
    ensure(ctx, ws.xr, ws.xr_bytes, B * in_ * act_size(act));
    xr = {ws.xr.get(), 0};
    encode_rht(ctx, enc, x, xr, B, in_, {signs_in_.get(), 0}, {hm_in_.get(), 0}, false, act, act);
    barrier(enc);
  }
  const bool res = has_residual();
  if (!seed_out_ && !res) {
    encode_kernel(ctx, enc, xr, y, B, act, act, path);
    return;
  }
  ensure(ctx, ws.y32, ws.y32_bytes, B * out_ * 4);
  BufRef y32{ws.y32.get(), 0};
  encode_kernel(ctx, enc, xr, y32, B, act, Act::F32, path);
  barrier(enc);
  if (seed_out_) {
    if (!res) {
      encode_rht(ctx, enc, y32, y, B, out_, {signs_out_.get(), 0}, {hm_out_.get(), 0}, true, Act::F32, act);
      return;
    }
    ensure(ctx, ws.y32b, ws.y32b_bytes, B * out_ * 4);
    BufRef y32b{ws.y32b.get(), 0};
    encode_rht(ctx, enc, y32, y32b, B, out_, {signs_out_.get(), 0}, {hm_out_.get(), 0}, true, Act::F32, Act::F32);
    barrier(enc);
    y32 = y32b;
  }
  encode_residual(ctx, enc, x, y32, y, B, act);
}

std::size_t Linear::max_exact_rows(const Context& ctx) const {
  if (tcq_) return std::min<std::size_t>(kTcqExactRows, kTcqMaxBatchTile);   // vec tiles: rows independent
  if (1 >= ctx.mma_min_batch(int(M_))) return 1;   // decode runs the mma lookup: no row-exact batched variant
  // suc_lookup_vec's CHUNK switch (the per-lane group assignment) as a function of BT (= SwitchLinear::max_tile_rows)
  const unsigned low = bits_ & (~bits_ + 1u), gcd = low < 32u ? low : 32u;
  auto chunk = [&](std::size_t bt) {
    return gcd == bits_ || (G_ % 2 == 0 && bt <= 6) || (bt <= 2 && bits_ / gcd <= 3);
  };
  // Default 2: the vec lookup is ALU-bound beyond ~2 rows per weight read (M1 Ultra, Qwen3.8-27B MLP: 6.3 s GPU for 407
  // prompt tokens at 2 rows, 6.4-7.0 at 4, 9.7 at 8, 18.2 at 16; 7.3 at 1). $SUC_EXACT_TILE overrides (bits unchanged).
  static const std::size_t env_cap = [] { const char* e = std::getenv("SUC_EXACT_TILE"); return e ? std::size_t(std::atoi(e)) : 2; }();
  const std::size_t cap = std::min<std::size_t>(std::max<std::size_t>(env_cap, 1), std::size_t(std::max(1, ctx.tuning().max_batch_tile)));
  std::size_t b = 1;
  while (b < cap && chunk(b + 1) == chunk(1)) ++b;
  return b;
}

void Linear::encode_kernel_exact(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t B,
                                 Act xa, Act ya) const {
  if (tcq_) {   // tiles of up to kTcqExactRows rows on the vec kernel (the decode step's kernel at B = 1)
    for (std::size_t b0 = 0; b0 < B; b0 += kTcqExactRows) {
      const std::size_t bt = std::min(kTcqExactRows, B - b0);
      encode_tcq(ctx, enc, tcq_l_, 0, x.at(b0 * in_ * act_size(xa)), packed_, scales_, scales_f16_,
                 {tcq_cst_.get(), 0}, {}, {}, y.at(b0 * out_ * act_size(ya)), bt, 1, out_, in_, xa, ya, Path::Vec);
    }
    return;
  }
  const std::size_t tile = max_exact_rows(ctx);
  const Tuning& tu = ctx.tuning();
  const int m = int(M_);
  const bool cbtg = ctx.codebook_in(m) == "threadgroup";
  const char* ts = scales_f16_ ? "half" : "float";
  static const int env_rows = [] { const char* e = std::getenv("SUC_EXACT_ROWS"); return e ? std::atoi(e) : 0; }();
  static const int env_sg = [] { const char* e = std::getenv("SUC_EXACT_SG"); return e ? std::atoi(e) : 0; }();
  for (std::size_t b0 = 0; b0 < B; b0 += tile) {
    const std::size_t bt = std::min(tile, B - b0);
    const BufRef xt = x.at(b0 * in_ * act_size(xa)), yt = y.at(b0 * out_ * act_size(ya));
    if (bt == 1) {   // the decode step's own launch
      encode_kernel(ctx, enc, xt, yt, 1, xa, ya, Path::Auto);
      continue;
    }
    // the vec lookup (CHUNK as for BT = 1 up to max_exact_rows); the rows per simdgroup R and simdgroups per
    // threadgroup SG only block the output rows (no row's arithmetic depends on them). $SUC_EXACT_ROWS / $SUC_EXACT_SG
    // override them (tuning experiments; the M1 Ultra sweep on Qwen3.8-27B found lookup_rows(bt) / LOOKUP_SG best).
    const int rows = env_rows > 0 ? env_rows : lookup_rows(bt);
    const int sgs = env_sg > 0 ? env_sg : tu.lookup_sg;
    const PsoKey key{4, {int64_t(out_), int64_t(in_),
                         int64_t(G_) | int64_t(K_) << 8 | int64_t(bits_) << 32 | int64_t(M_) << 40 |
                             int64_t(scales_f16_) << 48,
                         int64_t(bt), int64_t(rows) | int64_t(sgs) << 16, int64_t(xa), int64_t(ya)}};
    enc->setComputePipelineState(ctx.pipeline(key, [&]() {
      const std::string head = join({act_msl(xa), I(long(bt)), I(long(out_)), I(long(in_)), I(long(in_ / G_)),
                                     I(long(G_)), I(long(K_)), I(long(bits_)), I(long(words_))});
      return std::pair<std::string, std::string>{
          "suc_lookup_vec",
          head + ", " + join({I(sgs), I(m), I(rows), I(tu.lookup_vec), I(cbtg), "0", act_msl(ya), ts})};
    }));
    enc->setBuffer(xt.buffer, xt.offset, 0);
    enc->setBuffer(codebooks_.buffer, codebooks_.offset, 1);
    enc->setBuffer(packed_.buffer, packed_.offset, 2);
    enc->setBuffer(scales_.buffer, scales_.offset, 3);
    enc->setBuffer(yt.buffer, yt.offset, 4);
    const std::size_t n_tg = (out_ + std::size_t(sgs * rows) - 1) / std::size_t(sgs * rows);
    enc->dispatchThreads(MTL::Size(n_tg * sgs * 32, 1, 1), MTL::Size(sgs * 32, 1, 1));
  }
}

void Linear::encode_rows_exact(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t B,
                               Act act, Workspace& ws) const {
  BufRef xr = x;
  if (seed_in_) {
    ensure(ctx, ws.xr, ws.xr_bytes, B * in_ * act_size(act));
    xr = {ws.xr.get(), 0};
    encode_rht(ctx, enc, x, xr, B, in_, {signs_in_.get(), 0}, {hm_in_.get(), 0}, false, act, act);
    barrier(enc);
  }
  const bool res = has_residual();
  if (!seed_out_ && !res) {
    encode_kernel_exact(ctx, enc, xr, y, B, act, act);
    return;
  }
  ensure(ctx, ws.y32, ws.y32_bytes, B * out_ * 4);
  BufRef y32{ws.y32.get(), 0};
  encode_kernel_exact(ctx, enc, xr, y32, B, act, Act::F32);
  barrier(enc);
  if (seed_out_) {
    if (!res) {
      encode_rht(ctx, enc, y32, y, B, out_, {signs_out_.get(), 0}, {hm_out_.get(), 0}, true, Act::F32, act);
      return;
    }
    ensure(ctx, ws.y32b, ws.y32b_bytes, B * out_ * 4);
    BufRef y32b{ws.y32b.get(), 0};
    encode_rht(ctx, enc, y32, y32b, B, out_, {signs_out_.get(), 0}, {hm_out_.get(), 0}, true, Act::F32, Act::F32);
    barrier(enc);
    y32 = y32b;
  }
  encode_residual(ctx, enc, x, y32, y, B, act, true);
}

void Linear::encode_rotate_in(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef xr, std::size_t B,
                              Act act) const {
  if (!seed_in_) fail(name_ + ": encode_rotate_in without an input rotation");
  encode_rht(ctx, enc, x, xr, B, in_, {signs_in_.get(), 0}, {hm_in_.get(), 0}, false, act, act);
}

void Linear::encode_stage(Context& ctx, MTL::ComputeCommandEncoder* enc, int stage, BufRef x, BufRef xr, BufRef y,
                          std::size_t B, Act act, Workspace& ws, Path path) const {
  const bool res = has_residual();
  if (stage < 0 || stage >= stages()) fail(name_ + ": no stage " + std::to_string(stage));
  if (stage == 0) {
    if (!seed_out_ && !res) {
      encode_kernel(ctx, enc, xr, y, B, act, act, path);
      return;
    }
    ensure(ctx, ws.y32, ws.y32_bytes, B * out_ * 4);
    encode_kernel(ctx, enc, xr, {ws.y32.get(), 0}, B, act, Act::F32, path);
    return;
  }
  if (stage == 1 && seed_out_) {
    if (!res) {
      encode_rht(ctx, enc, {ws.y32.get(), 0}, y, B, out_, {signs_out_.get(), 0}, {hm_out_.get(), 0}, true, Act::F32,
                 act);
      return;
    }
    ensure(ctx, ws.y32b, ws.y32b_bytes, B * out_ * 4);
    encode_rht(ctx, enc, {ws.y32.get(), 0}, {ws.y32b.get(), 0}, B, out_, {signs_out_.get(), 0}, {hm_out_.get(), 0},
               true, Act::F32, Act::F32);
    return;
  }
  encode_residual(ctx, enc, x, {seed_out_ ? ws.y32b.get() : ws.y32.get(), 0}, y, B, act);
}

// ------------------------------------------------------------------------------------------ FusedLinear

namespace {

// rotation.supported: n = m * 2^k with 2^k >= 32 (the blocks layout; the Python fusion requires it)
bool blocks_supported(std::size_t n) {
  const HadamardFactor f = hadamard_factor(n);
  return f.m && f.p >= 32;
}

const uint8_t* host_ptr(BufRef r) { return static_cast<const uint8_t*>(r.buffer->contents()) + r.offset; }

}  // namespace

std::string FusedLinear::key(const Linear& l) {
  if (l.tcq_) return "";   // trellis-coded projections run their own launch (compressed_model.group_fusable_key)
  if (l.seed_out_ && !blocks_supported(l.out_)) return "";
  if (l.seed_in_ && !blocks_supported(l.in_)) return "";
  return join({I(long(l.in_)), I(long(l.bits_)), I(long(l.M_)), I(long(l.K_)), I(long(l.G_)),
               l.seed_in_ ? std::to_string(*l.seed_in_) : std::string("none"), l.seed_out_ ? "out" : "-",
               l.scales_f16_ ? "h" : "f"});
}

FusedLinear FusedLinear::build(Context& ctx, const std::vector<const Linear*>& ms, bool in_place) {
  if (ms.size() < 2 || ms.size() > 4) fail("FusedLinear: 2..4 members (suc_linear.MAX_SEGMENTS)");
  const std::string k0 = key(*ms[0]);
  for (const Linear* m : ms)
    if (key(*m).empty() || key(*m) != k0) fail("FusedLinear: members are not fusable (" + m->name_ + ")");
  const Linear& f = *ms[0];
  FusedLinear F;
  F.in_ = f.in_;
  F.M_ = f.M_;
  F.K_ = f.K_;
  F.G_ = f.G_;
  F.bits_ = f.bits_;
  F.words_ = f.words_;
  F.rot_out_ = f.seed_out_.has_value();
  F.scales_f16_ = f.scales_f16_;
  for (const Linear* m : ms) F.sizes_.push_back(m->out_), F.out_ += m->out_;
  const std::size_t S = ms.size(), cb1 = f.M_ * f.K_ * f.G_ * 2;
  auto own = [&](const void* d, std::size_t n) {
    F.copied_bytes_ += n;
    return ctx.upload(d, n);
  };
  // codebooks (S, M, K, G)
  std::vector<uint8_t> cb(S * cb1);
  for (std::size_t s = 0; s < S; ++s) std::memcpy(cb.data() + s * cb1, host_ptr(ms[s]->codebooks_), cb1);
  F.codebooks_ = own(cb.data(), cb.size());
  const std::size_t wb = F.words_ * 4;
  const std::size_t ss = f.scales_f16_ ? 2 : 4;
  std::size_t at = 0;
  // in place: every member's packed indices and f16 scales in ONE buffer (the mapped view), 4- / 2-byte aligned
  bool tab = in_place && f.scales_f16_;
  for (const Linear* l : ms)
    tab = tab && l->packed_.buffer == f.packed_.buffer && l->scales_.buffer == f.packed_.buffer &&
          l->packed_.offset % 4 == 0 && l->scales_.offset % 2 == 0;
  if (tab) {
    std::vector<uint64_t> t(ms.size() * (F.M_ + 1));
    for (std::size_t s = 0; s < ms.size(); ++s) {
      const Linear* l = ms[s];
      for (std::size_t m = 0; m < F.M_; ++m) t[s * (F.M_ + 1) + m] = l->packed_.offset / 4 + m * l->out_ * F.words_;
      t[s * (F.M_ + 1) + F.M_] = l->scales_.offset / 2;
    }
    F.table_ = own(t.data(), t.size() * 8);
    F.packed_ref_ = F.scales_ref_ = {f.packed_.buffer, 0};
    F.weight_bytes_ = cb.size();
    for (const Linear* l : ms) F.weight_bytes_ += l->M_ * l->out_ * wb + l->out_ * ss;
  } else {
    // packed indices (M, Σ out, words): stream m of every member, member after member
    std::vector<uint8_t> pk(F.M_ * F.out_ * wb);
    for (std::size_t m = 0; m < F.M_; ++m)
      for (const Linear* l : ms) {
        std::memcpy(pk.data() + at, host_ptr(l->packed_) + m * l->out_ * wb, l->out_ * wb);
        at += l->out_ * wb;
      }
    F.packed_ = own(pk.data(), pk.size());
    // row scales (f16 as stored, or float32 for host layers)
    std::vector<uint8_t> sc(F.out_ * ss);
    at = 0;
    for (const Linear* l : ms) std::memcpy(sc.data() + at, host_ptr(l->scales_), l->out_ * ss), at += l->out_ * ss;
    F.scales_ = own(sc.data(), sc.size());
    F.packed_ref_ = {F.packed_.get(), 0};
    F.scales_ref_ = {F.scales_.get(), 0};
    F.weight_bytes_ = cb.size() + pk.size() + sc.size();
  }
  if (F.rot_out_) {
    std::vector<float> sg(F.out_);
    at = 0;
    for (const Linear* l : ms) std::memcpy(sg.data() + at, l->signs_out_->contents(), l->out_ * 4), at += l->out_;
    F.signs_out_ = own(sg.data(), sg.size() * 4);
  }
  // residuals (SucFusedResidual): CSR rows concatenated, column blocks concatenated, one int8 scale per member
  for (const Linear* l : ms) F.n_csr_ += l->n_csr_, F.n_cols_ += l->n_cols_;
  if (!F.has_residual()) return F;
  const bool u16 = F.in_ <= 65536;
  F.rcols_u16_ = F.ccols_u16_ = u16;
  const std::size_t ci = u16 ? 2 : 4;
  std::vector<float> rscale;
  for (const Linear* l : ms) rscale.push_back(l->rscale_ ? *static_cast<const float*>(l->rscale_->contents()) : 1.0f);
  F.rscale_ = own(rscale.data(), rscale.size() * 4);
  if (F.n_csr_) {
    std::optional<bool> i8;
    for (const Linear* l : ms)
      if (l->n_csr_) {
        if (i8 && *i8 != l->rvals_i8_) fail("FusedLinear: members store residual values in different dtypes");
        i8 = l->rvals_i8_;
      }
    F.rvals_i8_ = *i8;
    const std::size_t vb = F.rvals_i8_ ? 1 : 2;
    std::vector<uint32_t> rp(F.out_ + 1, 0);
    std::vector<uint8_t> cols(F.n_csr_ * ci), vals(F.n_csr_ * vb);
    std::size_t r0 = 0, e0 = 0;
    for (const Linear* l : ms) {
      if (l->n_csr_) {
        const auto* lp = reinterpret_cast<const uint32_t*>(host_ptr(l->row_ptr_));
        for (std::size_t r = 0; r < l->out_; ++r) rp[r0 + r + 1] = uint32_t(e0 + lp[r + 1]);
        std::memcpy(cols.data() + e0 * ci, host_ptr(l->rcols_), l->n_csr_ * ci);
        std::memcpy(vals.data() + e0 * vb, host_ptr(l->rvals_), l->n_csr_ * vb);
        e0 += l->n_csr_;
      } else {
        for (std::size_t r = 0; r < l->out_; ++r) rp[r0 + r + 1] = uint32_t(e0);
      }
      r0 += l->out_;
    }
    F.row_ptr_ = own(rp.data(), rp.size() * 4);
    F.rcols_ = own(cols.data(), cols.size());
    F.rvals_ = own(vals.data(), vals.size());
    F.weight_bytes_ += rp.size() * 4 + cols.size() + vals.size();
  }
  std::vector<uint32_t> meta;
  {
    uint32_t a = 0;
    meta.push_back(0);
    for (const Linear* l : ms) meta.push_back(a += uint32_t(l->out_));
    a = 0;
    meta.push_back(0);
    for (const Linear* l : ms) meta.push_back(a += uint32_t(l->n_cols_));
    a = 0;
    for (const Linear* l : ms) meta.push_back(a), a += uint32_t(l->n_cols_ * l->out_);
  }
  F.meta_ = own(meta.data(), meta.size() * 4);
  if (F.n_cols_) {
    if (F.n_cols_ > kXgFloats) fail("FusedLinear: more residual columns than the gather buffer (6144)");
    std::optional<bool> i8;
    for (const Linear* l : ms)
      if (l->n_cols_) {
        if (i8 && *i8 != l->cval_i8_) fail("FusedLinear: members store column residuals in different dtypes");
        i8 = l->cval_i8_;
      }
    F.cval_i8_ = *i8;
    const std::size_t vb = F.cval_i8_ ? 1 : 2;
    std::vector<uint8_t> cols, vals;
    std::vector<float> cs;
    for (const Linear* l : ms) {
      if (!l->n_cols_) continue;
      const uint8_t* c = host_ptr(l->ccols_);
      cols.insert(cols.end(), c, c + l->n_cols_ * ci);
      const uint8_t* v = host_ptr(l->cval_);
      vals.insert(vals.end(), v, v + l->n_cols_ * l->out_ * vb);
      const float* x = reinterpret_cast<const float*>(host_ptr(l->cscale_));
      cs.insert(cs.end(), x, x + l->n_cols_);
    }
    F.ccols_ = own(cols.data(), cols.size());
    F.cval_ = own(vals.data(), vals.size());
    F.cscale_ = own(cs.data(), cs.size() * 4);
    F.weight_bytes_ += cols.size() + vals.size() + cs.size() * 4;
  }
  return F;
}

// = suc_linear.fused_matmul_metal
void FusedLinear::encode_kernel(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t B,
                                Act xa, Act ya, Path path) const {
  const Tuning& tu = ctx.tuning();
  const std::size_t tile = std::size_t(tu.max_batch_tile);
  const int m = int(M_);
  const bool staged = ctx.codebook_in(m) == "threadgroup";   // FUSED_STAGING
  const char* ts = scales_f16_ ? "half" : "float";
  std::vector<std::size_t> bounds;
  {
    std::size_t a = 0;
    for (std::size_t s = 0; s + 1 < sizes_.size(); ++s) bounds.push_back(a += sizes_[s]);
  }
  std::vector<std::size_t> sb = bounds;
  while (sb.size() < 3) sb.push_back(out_);
  if (path == Path::Prefill) path = Path::Auto;
  for (std::size_t b0 = 0; b0 < B; b0 += tile) {
    const std::size_t bt = std::min(tile, B - b0);
    const bool mma = path == Path::Mma || (path == Path::Auto && int(bt) >= ctx.mma_min_batch(m));
    const std::string head = join({act_msl(xa), I(long(bt)), I(long(out_)), I(long(in_)), I(long(in_ / G_)),
                                   I(long(G_)), I(long(K_)), I(long(bits_)), I(long(words_)), I(long(sb[0])),
                                   I(long(sb[1])), I(long(sb[2]))});
    const int sgs = mma ? tu.mma_sg : tu.lookup_sg;
    const int rows = mma ? 8 * tu.mma_rt : lookup_rows(bt);
    bool cbtg = staged;
    for (std::size_t b : bounds) cbtg = cbtg && b % std::size_t(sgs * rows) == 0;
    std::string kernel, targs;
    if (mma) {
      const char* tt = (xa == Act::F16 && M_ == 1) ? "half" : "float";
      kernel = table_.get() ? "suc_fused_lookup_mma_tab" : "suc_fused_lookup_mma";
      targs = head + ", " + join({I(sgs), I(m), I(tu.mma_rt), I(long((bt + 7) / 8)),
                                  I(mma_kcg(int(G_), int(bits_), tu.mma_kc)), tt, I(cbtg), act_msl(ya), ts});
    } else {
      kernel = table_.get() ? "suc_fused_lookup_vec_tab" : "suc_fused_lookup_vec";
      targs = head + ", " + join({I(sgs), I(m), I(rows), I(tu.lookup_vec), I(cbtg), "0", act_msl(ya), ts});
    }
    enc->setComputePipelineState(ctx.pipeline(kernel, targs));
    enc->setBuffer(x.buffer, x.offset + b0 * in_ * act_size(xa), 0);
    enc->setBuffer(codebooks_.get(), 0, 1);
    enc->setBuffer(packed_ref_.buffer, packed_ref_.offset, 2);
    enc->setBuffer(scales_ref_.buffer, scales_ref_.offset, 3);
    enc->setBuffer(y.buffer, y.offset + b0 * out_ * act_size(ya), 4);
    if (table_.get()) enc->setBuffer(table_.get(), 0, 5);
    const std::size_t n_tg = (out_ + std::size_t(sgs * rows) - 1) / std::size_t(sgs * rows);
    enc->dispatchThreads(MTL::Size(n_tg * sgs * 32, 1, 1), MTL::Size(sgs * 32, 1, 1));
  }
}

void FusedLinear::encode_stage(Context& ctx, MTL::ComputeCommandEncoder* enc, int stage, BufRef x, BufRef xr,
                               const std::vector<BufRef>& ys, BufRef yf, std::size_t B, Act act, Workspace& ws,
                               Path path) const {
  const bool res = has_residual();
  if (stage < 0 || stage >= stages()) fail("FusedLinear: no stage " + std::to_string(stage));
  if (stage == 0) {
    if (!rot_out_ && !res) {
      encode_kernel(ctx, enc, xr, yf, B, act, act, path);
      return;
    }
    ensure(ctx, ws.y32, ws.y32_bytes, B * out_ * 4);
    encode_kernel(ctx, enc, xr, {ws.y32.get(), 0}, B, act, Act::F32, path);
    return;
  }
  if (stage == 1 && rot_out_) {
    // = rotation.rotate(y32, signs_out, sizes, inverse=True, split=not res): rotation._blocks_plan
    const std::size_t S = sizes_.size();
    if (S < 2 || S > 3) fail("FusedLinear: the segmented rotation is generated for 2 or 3 segments");
    if (!res && ys.size() != S) fail("FusedLinear: one output per member expected");
    const std::size_t nmax = *std::max_element(sizes_.begin(), sizes_.end());
    if (ctx.rotation_row_layout(nmax)) {   // rotation._plan (layout_for(max(sizes)) == "row")
      std::vector<BufRef> outs;
      if (res) {
        ensure(ctx, ws.y32b, ws.y32b_bytes, B * out_ * 4);
        outs.push_back({ws.y32b.get(), 0});
      } else {
        outs = ys;
      }
      encode_rot_row(ctx, enc, {ws.y32.get(), 0}, {signs_out_.get(), 0}, outs, sizes_, B, true, Act::F32,
                     res ? Act::F32 : act);
      return;
    }
    const auto [qmax, sgmax] = ctx.rotation_block_config(nmax);
    int q = qmax, sg = std::min(sgmax, kTgFloats / qmax);
    std::vector<HadamardFactor> fs;
    for (std::size_t n : sizes_) fs.push_back(hadamard_factor(n));
    for (const auto& h : fs) q = std::min<int>(q, int(h.p));
    sg = std::min(sgmax, kTgFloats / q);
    for (const auto& h : fs) sg = std::min<int>(sg, int(h.m * h.p) / q);
    const int t = 32 * std::max(1, sg);
    std::string targs = join({"float", act_msl(res ? Act::F32 : act), I(q), I(t), I(q / 32), "1", I(long(out_)),
                              I((t / 32) * q)});
    std::size_t off = 0, gs = 0;
    for (const auto& h : fs) {
      const std::size_t ge = gs + h.m * h.p / std::size_t(q);
      targs += ", " + join({I(long(h.m)), I(long(h.p)), I(long(off)), I(long(gs)), I(long(ge))});
      off += h.m * h.p;
      gs = ge;
    }
    const std::string kname = "suc_rot_blocks" + std::to_string(S) + (res ? "" : "_split");
    enc->setComputePipelineState(ctx.pipeline(kname, targs));
    enc->setBuffer(ws.y32.get(), 0, 0);
    enc->setBuffer(signs_out_.get(), 0, 1);
    if (res) {
      ensure(ctx, ws.y32b, ws.y32b_bytes, B * out_ * 4);
      enc->setBuffer(ws.y32b.get(), 0, 2);
    } else {
      for (std::size_t s = 0; s < S; ++s) enc->setBuffer(ys[s].buffer, ys[s].offset, 2 + s);
    }
    enc->dispatchThreads(MTL::Size(t, gs, B), MTL::Size(t, 1, 1));
    return;
  }
  // = compressed_model.structured_residual_metal(..., segments=S, row_quads=all(n % 4 == 0))
  const BufRef y32 = {rot_out_ ? ws.y32b.get() : ws.y32.get(), 0};
  const bool has_cols = n_cols_ > 0, has_csr = n_csr_ > 0;
  const std::size_t ctot = n_cols_;
  bool quads = true;
  for (std::size_t n : sizes_) quads = quads && n % 4 == 0;
  const int rq = quads ? 4 : 1;
  std::size_t bt = B == 1 ? 1 : B <= 4 ? 4 : 8;
  while (bt > 1 && bt * ctot > kXgFloats) bt /= 2;
  std::size_t js = 8;
  if (residual_exact_env()) {   // experiments: the one-row reduction split JS for any B (see Linear::encode_residual)
    while (js > 1 && 4 * (ctot + js * 32 * rq) > kStructTgBytes) js /= 2;
    while (bt > 1 && 4 * (bt * ctot + js * 32 * rq * bt) > kStructTgBytes) bt /= 2;
  } else {
    while (js > 1 && 4 * (bt * ctot + js * 32 * rq * bt) > kStructTgBytes) js /= 2;
  }
  const std::size_t red = js > 1 ? js * 32 * rq * bt : 1;
  const char* tcc = has_cols ? (ccols_u16_ ? "uint16_t" : "uint32_t") : "uint32_t";
  const char* tcv = has_cols ? (cval_i8_ ? "int8_t" : "half") : "int8_t";
  const char* trc = has_csr ? (rcols_u16_ ? "uint16_t" : "uint32_t") : "uint32_t";
  const char* trv = has_csr ? (rvals_i8_ ? "int8_t" : "half") : "int8_t";
  const std::string targs =
      join({act_msl(act), tcc, tcv, trc, trv, I(long(in_)), I(long(out_)), I(long(sizes_.size())),
            I(long(std::max<std::size_t>(ctot, 1))), I(long(bt)), I(long(std::max<std::size_t>(bt * ctot, 1))), I(rq),
            I(long(js)), I(long(red)), has_cols ? "true" : "false", has_csr ? "true" : "false", act_msl(act)});
  enc->setComputePipelineState(ctx.pipeline("suc_structured_residual", targs));
  const BufRef d = ctx.dummy();
  auto set = [&](MTL::Buffer* b, std::size_t o, int i) { enc->setBuffer(b ? b : d.buffer, b ? o : 0, i); };
  set(x.buffer, x.offset, 0);
  set(y32.buffer, 0, 1);
  set(meta_.get(), 0, 2);
  set(has_cols ? ccols_.get() : nullptr, 0, 3);
  set(has_cols ? cval_.get() : nullptr, 0, 4);
  set(has_cols ? cscale_.get() : nullptr, 0, 5);
  set(has_csr ? row_ptr_.get() : nullptr, 0, 6);
  set(has_csr ? rcols_.get() : nullptr, 0, 7);
  set(has_csr ? rvals_.get() : nullptr, 0, 8);
  set(rscale_.get(), 0, 9);
  set(yf.buffer, yf.offset, 10);
  const int shape[2] = {int(B), int(in_)};
  enc->setBytes(shape, sizeof shape, 11);
  const std::size_t n_tg = (out_ + 32 * rq - 1) / (32 * rq);
  enc->dispatchThreads(MTL::Size(n_tg * 32 * js, (B + bt - 1) / bt, 1), MTL::Size(32 * js, 1, 1));
}

void encode_copy_cols(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t rows,
                      std::size_t cols, std::size_t xs, std::size_t ys, Act act) {
  enc->setComputePipelineState(ctx.pipeline("suc_copy_cols", act_msl(act)));
  enc->setBuffer(x.buffer, x.offset, 0);
  enc->setBuffer(y.buffer, y.offset, 1);
  const uint32_t c = uint32_t(cols), a = uint32_t(xs), b = uint32_t(ys);
  enc->setBytes(&c, 4, 2);
  enc->setBytes(&a, 4, 3);
  enc->setBytes(&b, 4, 4);
  enc->dispatchThreads(MTL::Size(cols, rows, 1), MTL::Size(std::min<std::size_t>(256, cols), 1, 1));
}

bool prefill_exact_env() {
  static const bool v = [] { const char* e = std::getenv("SUC_PREFILL_EXACT"); return e && e[0] == '1'; }();
  return v;
}

namespace {
// Rows per threadgroup of the suc_mm_* kernels: 64 for large chunks, 32 / 16 / 8 for small ones ($SUC_MM_BM forces).
int mm_bm(std::size_t n) {
  static const int env = [] { const char* e = std::getenv("SUC_MM_BM"); return e ? std::atoi(e) : 0; }();
  if (env == 8 || env == 16 || env == 32 || env == 64) return env;
  return n >= 64 ? 64 : n >= 32 ? 32 : n >= 16 ? 16 : 8;
}
}  // namespace

bool encode_mm_affine(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef packed, BufRef scales,
                      BufRef biases, BufRef y, std::size_t n, std::size_t xs, std::size_t ys, unsigned bits,
                      std::size_t group, std::size_t in, std::size_t out, std::size_t words, const char* sdtype, Act act,
                      const char* y_msl) {
  if (!n || in % 32 || group % 4 || xs % 4) return false;
  const int bm = mm_bm(n);
  const std::string ty = y_msl ? y_msl : act_msl(act);
  enc->setComputePipelineState(ctx.pipeline(
      "suc_mm_affine", join({act_msl(act), I(bm), I(long(bits)), I(long(group)), I(long(in)), I(long(out)),
                             I(long(words)), sdtype, ty})));
  enc->setBuffer(x.buffer, x.offset, 0);
  enc->setBuffer(packed.buffer, packed.offset, 1);
  enc->setBuffer(scales.buffer, scales.offset, 2);
  enc->setBuffer(biases.buffer, biases.offset, 3);
  enc->setBuffer(y.buffer, y.offset, 4);
  const uint32_t n32 = uint32_t(n), a = uint32_t(xs), b = uint32_t(ys);
  enc->setBytes(&n32, 4, 5);
  enc->setBytes(&a, 4, 6);
  enc->setBytes(&b, 4, 7);
  enc->dispatchThreadgroups(MTL::Size((n + bm - 1) / bm, (out + 63) / 64, 1), MTL::Size(128, 1, 1));
  return true;
}

bool encode_mm_dense(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef w, BufRef y, std::size_t n,
                     std::size_t xs, std::size_t ys, std::size_t in, std::size_t out, const char* wdtype, Act act,
                     const char* y_msl) {
  if (!n || in % 32 || xs % 4) return false;
  const int bm = mm_bm(n);
  const std::string ty = y_msl ? y_msl : act_msl(act);
  enc->setComputePipelineState(ctx.pipeline(
      "suc_mm_dense", join({act_msl(act), I(bm), wdtype, I(long(in)), I(long(out)), ty})));
  enc->setBuffer(x.buffer, x.offset, 0);
  enc->setBuffer(w.buffer, w.offset, 1);
  enc->setBuffer(y.buffer, y.offset, 2);
  const uint32_t n32 = uint32_t(n), a = uint32_t(xs), b = uint32_t(ys);
  enc->setBytes(&n32, 4, 3);
  enc->setBytes(&a, 4, 4);
  enc->setBytes(&b, 4, 5);
  enc->dispatchThreadgroups(MTL::Size((n + bm - 1) / bm, (out + 63) / 64, 1), MTL::Size(128, 1, 1));
  return true;
}

std::vector<float> Linear::forward(Context& ctx, const std::vector<float>& x, std::size_t B, Act act, Path path,
                                   bool rows_exact) const {
  if (x.size() != B * in_) fail("forward: x has " + std::to_string(x.size()) + " values, expected B * in");
  Pool pool;
  BufPtr xb = ctx.new_buffer(B * in_ * act_size(act));
  f32_to_act(x.data(), xb->contents(), x.size(), act);
  BufPtr yb = ctx.new_buffer(B * out_ * act_size(act));
  Workspace ws;
  MTL::CommandBuffer* cb = ctx.queue()->commandBuffer();
  MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  if (rows_exact) encode_rows_exact(ctx, enc, {xb.get(), 0}, {yb.get(), 0}, B, act, ws);
  else encode(ctx, enc, {xb.get(), 0}, {yb.get(), 0}, B, act, ws, path);
  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();
  if (cb->status() != MTL::CommandBufferStatusCompleted)
    fail("command buffer failed: " + (cb->error() ? str(cb->error()->localizedDescription()) : std::string("?")));
  std::vector<float> y(B * out_);
  act_to_f32(yb->contents(), y.data(), y.size(), act);
  return y;
}

}  // namespace suc::metal
