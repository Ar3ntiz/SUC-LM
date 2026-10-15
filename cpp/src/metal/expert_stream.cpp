// SSD expert streaming: slot pools, pread + F_NOCACHE IO threads, LRU / SLRU, pinned hot set (suc/expert_stream.hpp).
#include "suc/expert_stream.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <list>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

#include "suc/json.hpp"

namespace suc::metal {

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("expert stream: " + msg); }

uint64_t align64(uint64_t v) { return (v + 63) / 64 * 64; }

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// "<prefix>.experts.<digits>.<rest>"
bool routed_expert_name(const std::string& n) {
  std::size_t p = 0;
  while ((p = n.find(".experts.", p)) != std::string::npos) {
    std::size_t q = p + 9, d = q;
    while (d < n.size() && n[d] >= '0' && n[d] <= '9') ++d;
    if (d > q && d < n.size() && n[d] == '.') return true;
    p = q;
  }
  return false;
}

// Completion of a group of reads.
struct Batch {
  std::mutex mu;
  std::condition_variable cv;
  std::size_t remaining = 0;
  std::exception_ptr error;
  uint64_t calls = 0, bytes = 0;
  double read_s = 0.0;
  void done(uint64_t c, uint64_t b, double s, std::exception_ptr e) {
    std::lock_guard<std::mutex> l(mu);
    calls += c;
    bytes += b;
    read_s += s;
    if (e && !error) error = e;
    if (--remaining == 0) cv.notify_all();
  }
  void wait() {
    std::unique_lock<std::mutex> l(mu);
    cv.wait(l, [&] { return remaining == 0; });
  }
};

struct ReadTask {
  uint8_t* dst = nullptr;
  uint64_t off = 0, bytes = 0;
  std::shared_ptr<Batch> batch;
};

// Fixed pool of IO threads reading with pread from one descriptor.
class IoPool {
 public:
  IoPool(int fd, unsigned n) : fd_(fd) {
    for (unsigned i = 0; i < std::max(1u, n); ++i) th_.emplace_back([this] { run(); });
  }
  ~IoPool() {
    {
      std::lock_guard<std::mutex> l(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : th_) t.join();
  }
  void submit(std::vector<ReadTask>& tasks) {
    if (tasks.empty()) return;
    {
      std::lock_guard<std::mutex> l(mu_);
      for (auto& t : tasks) q_.push_back(std::move(t));
    }
    tasks.clear();
    cv_.notify_all();
  }

 private:
  void run() {
    for (;;) {
      ReadTask t;
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [&] { return stop_ || !q_.empty(); });
        if (q_.empty()) return;
        t = std::move(q_.front());
        q_.pop_front();
      }
      const auto t0 = Clock::now();
      uint64_t done = 0, calls = 0;
      std::exception_ptr err;
      while (done < t.bytes) {
        const ssize_t r = ::pread(fd_, t.dst + done, std::size_t(t.bytes - done), off_t(t.off + done));
        ++calls;
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) {
          err = std::make_exception_ptr(std::runtime_error(
              "expert stream: read of " + std::to_string(t.bytes) + " bytes at " + std::to_string(t.off) +
              (r < 0 ? std::string(" failed: ") + std::strerror(errno) : std::string(" hit the end of the file"))));
          break;
        }
        done += uint64_t(r);
      }
      t.batch->done(calls, done, since(t0), err);
    }
  }
  int fd_;
  std::vector<std::thread> th_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<ReadTask> q_;
  bool stop_ = false;
};

}  // namespace

std::vector<std::pair<uint64_t, uint64_t>> routed_expert_ranges(const Suc1File& f) {
  std::vector<std::pair<uint64_t, uint64_t>> r;
  for (const TensorDesc& t : f.tensors()) {
    if (!routed_expert_name(t.name)) continue;
    for (const ArrayRecord& a : t.arrays)
      if (a.nbytes) r.push_back({a.offset, a.offset + a.nbytes});
  }
  std::sort(r.begin(), r.end());
  std::vector<std::pair<uint64_t, uint64_t>> m;
  for (auto x : r) {
    if (!m.empty() && x.first <= m.back().second + 63) m.back().second = std::max(m.back().second, x.second);
    else m.push_back(x);
  }
  return m;
}

void split_command_buffer(Context& ctx, MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc, double* gpu_s) {
  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();
  if (cb->status() != MTL::CommandBufferStatusCompleted) {
    auto* e = cb->error();
    fail("command buffer failed: " + std::string(e ? e->localizedDescription()->utf8String() : "?"));
  }
  if (gpu_s) *gpu_s += cb->GPUEndTime() - cb->GPUStartTime();
  cb = ctx.queue()->commandBuffer();
  enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
}

// ------------------------------------------------------------------------------------------ ExpertStream

struct ExpertStream::Impl {
  struct Run {
    uint64_t off, bytes, at;   // file offset, bytes, offset inside the slot
  };
  struct Layer {
    std::string name;
    std::size_t E = 0, nranges = 0, min_slots = 0;
    std::vector<std::vector<Run>> runs;          // per expert
    std::vector<std::vector<uint64_t>> at;       // per expert: slot offset of each registered range
    uint64_t slot_bytes = 0, expert_bytes = 0;   // slot size; Σ registered bytes over experts
    Bind bind;
    std::size_t slots = 0, n_pinned = 0;
    BufPtr pool;
    std::vector<int32_t> slot_of;                // expert -> slot (-1)
    std::vector<uint8_t> pinned, prefetched, prot;
    std::vector<uint32_t> free_slots;
    std::list<uint32_t> probation, protected_;   // front = most recent
    std::vector<std::list<uint32_t>::iterator> pos;
    std::size_t protected_cap = 0;
    std::vector<std::shared_ptr<Batch>> inflight;
  };

  Context* ctx = nullptr;
  const Suc1File* file = nullptr;
  ExpertStreamOptions opt;
  std::vector<Layer> layers;
  bool final = false;
  int fd = -1;
  std::unique_ptr<IoPool> io;
  ExpertStreamStats st;
  uint64_t pool_bytes = 0, pinned_bytes = 0, expert_bytes = 0;

  // ---- policy (unpinned residents only)
  void touch(Layer& L, uint32_t e) {
    if (opt.policy == ExpertStreamOptions::Policy::LRU) {
      L.probation.splice(L.probation.begin(), L.probation, L.pos[e]);
      return;
    }
    if (L.prot[e]) {
      L.protected_.splice(L.protected_.begin(), L.protected_, L.pos[e]);
      return;
    }
    L.protected_.splice(L.protected_.begin(), L.probation, L.pos[e]);   // promote
    L.prot[e] = 1;
    while (L.protected_.size() > L.protected_cap && !L.protected_.empty()) {   // demote the oldest protected
      auto it = std::prev(L.protected_.end());
      const uint32_t d = *it;
      L.probation.splice(L.probation.begin(), L.protected_, it);
      L.prot[d] = 0;
    }
  }
  void insert(Layer& L, uint32_t e) {
    L.probation.push_front(e);
    L.pos[e] = L.probation.begin();
    L.prot[e] = 0;
  }
  void remove(Layer& L, uint32_t e) {
    (L.prot[e] ? L.protected_ : L.probation).erase(L.pos[e]);
    L.prot[e] = 0;
  }
  // least recent unpinned resident not in keep (probation first)
  int64_t victim(const Layer& L, const std::unordered_set<uint32_t>& keep) const {
    for (const auto* lst : {&L.probation, &L.protected_})
      for (auto it = lst->rbegin(); it != lst->rend(); ++it)
        if (!keep.count(*it)) return *it;
    return -1;
  }

  // slot for expert e (free or evicted), binds it and appends its reads
  void place(Layer& L, uint32_t e, const std::unordered_set<uint32_t>& keep, std::vector<ReadTask>& tasks,
             const std::shared_ptr<Batch>& batch) {
    uint32_t slot;
    if (!L.free_slots.empty()) {
      slot = L.free_slots.back();
      L.free_slots.pop_back();
    } else {
      const int64_t v = victim(L, keep);
      if (v < 0) fail(L.name + ": no evictable slot (all slots pinned or in use by this call)");
      remove(L, uint32_t(v));
      slot = uint32_t(L.slot_of[v]);
      L.slot_of[v] = -1;
      if (L.prefetched[v]) ++st.prefetch_wasted;
      L.prefetched[v] = 0;
      ++st.evictions;
    }
    L.slot_of[e] = int32_t(slot);
    const uint64_t base = uint64_t(slot) * L.slot_bytes;
    std::vector<uint64_t> at(L.nranges);
    for (std::size_t j = 0; j < L.nranges; ++j) at[j] = base + L.at[e][j];
    L.bind(e, L.pool.get(), at.data());
    auto* dst = static_cast<uint8_t*>(L.pool->contents()) + base;
    for (const Run& r : L.runs[e])
      for (uint64_t o = 0; o < r.bytes; o += opt.max_read) {
        const uint64_t n = std::min(opt.max_read, r.bytes - o);
        tasks.push_back({dst + r.at + o, r.off + o, n, batch});
        ++batch->remaining;
      }
  }

  void wait_inflight(Layer& L) {
    if (L.inflight.empty()) return;
    const auto t0 = Clock::now();
    for (auto& b : L.inflight) {
      b->wait();
      account(*b, false);
    }
    st.wait_seconds += since(t0);
    L.inflight.clear();
  }
  void account(Batch& b, bool sync) {
    if (b.error) std::rethrow_exception(b.error);
    st.read_calls += b.calls;
    st.bytes_read += b.bytes;
    if (sync) st.sync_bytes_read += b.bytes;
    st.read_seconds += b.read_s;
  }

  void allocate();
  void pin_hot_set();
};

ExpertStream::ExpertStream(Context& ctx, const Suc1File& f, const ExpertStreamOptions& opt)
    : p_(std::make_unique<Impl>()) {
  p_->ctx = &ctx;
  p_->file = &f;
  p_->opt = opt;
  if (opt.budget_bytes == 0) fail("budget_bytes must be > 0");
  if (opt.pin_fraction < 0.0 || opt.pin_fraction > 1.0) fail("pin_fraction must be in [0, 1]");
  if (opt.max_read < 4096) p_->opt.max_read = 4096;
  p_->fd = ::open(f.path().c_str(), O_RDONLY);
  if (p_->fd < 0) fail("cannot open " + f.path() + ": " + std::strerror(errno));
#ifdef F_NOCACHE
  if (opt.nocache && ::fcntl(p_->fd, F_NOCACHE, 1) != 0) fail(std::string("F_NOCACHE failed: ") + std::strerror(errno));
#endif
  p_->io = std::make_unique<IoPool>(p_->fd, opt.io_threads);
}

ExpertStream::~ExpertStream() {
  if (!p_) return;
  for (auto& L : p_->layers)
    for (auto& b : L.inflight) b->wait();
  p_->io.reset();
  if (p_->fd >= 0) ::close(p_->fd);
}

int ExpertStream::add_layer(const std::string& name, std::vector<std::vector<Range>> ranges, std::size_t min_slots,
                            Bind bind) {
  Impl& m = *p_;
  if (m.final) fail("add_layer after finalize");
  if (ranges.empty()) fail(name + ": no experts");
  Impl::Layer L;
  L.name = name;
  L.E = ranges.size();
  L.nranges = ranges[0].size();
  L.min_slots = std::max<std::size_t>(1, std::min(min_slots, L.E));
  L.bind = std::move(bind);
  L.runs.resize(L.E);
  L.at.resize(L.E);
  for (std::size_t e = 0; e < L.E; ++e) {
    const auto& rs = ranges[e];
    if (rs.size() != L.nranges) fail(name + ": experts register different numbers of ranges");
    std::vector<std::size_t> order(rs.size());
    for (std::size_t j = 0; j < order.size(); ++j) order[j] = j;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return rs[a].first < rs[b].first; });
    // runs: ranges merged when the gap is <= merge_gap; each run keeps the file layout, starts 64-aligned in the slot
    std::vector<Impl::Run>& runs = L.runs[e];
    std::vector<uint64_t>& at = L.at[e];
    at.assign(rs.size(), 0);
    uint64_t slot = 0;
    for (std::size_t j : order) {
      const auto [off, n] = rs[j];
      if (off + n > m.file->file_size()) fail(name + ": range beyond the end of the file");
      L.expert_bytes += n;
      if (!runs.empty() && off >= runs.back().off && off <= runs.back().off + runs.back().bytes + m.opt.merge_gap) {
        Impl::Run& r = runs.back();
        r.bytes = std::max(r.bytes, off + n - r.off);
      } else {
        if (!runs.empty()) slot = align64(runs.back().at + runs.back().bytes);
        runs.push_back({off, n, slot});
      }
      at[j] = runs.back().at + (off - runs.back().off);
    }
    const uint64_t used = runs.empty() ? 0 : align64(runs.back().at + runs.back().bytes);
    L.slot_bytes = std::max(L.slot_bytes, used);
  }
  L.slot_bytes = std::max<uint64_t>(64, L.slot_bytes);
  m.expert_bytes += L.expert_bytes;
  m.layers.push_back(std::move(L));
  return int(m.layers.size() - 1);
}

void ExpertStream::Impl::allocate() {
  // same capacity fraction for every layer, at least its minimum; leftover budget handed out slot by slot
  uint64_t all = 0, need = 0;
  for (const Layer& L : layers) {
    all += L.E * L.slot_bytes;
    need += L.min_slots * L.slot_bytes;
  }
  if (need > opt.budget_bytes)
    fail("budget " + std::to_string(opt.budget_bytes) + " B cannot hold the minimum slots of every layer (" +
         std::to_string(need) + " B)");
  const double frac = std::min(1.0, double(opt.budget_bytes) / double(all));
  uint64_t used = 0;
  for (Layer& L : layers) {
    L.slots = std::clamp<std::size_t>(std::size_t(frac * double(L.E)), L.min_slots, L.E);
    used += L.slots * L.slot_bytes;
  }
  for (bool shrink = used > opt.budget_bytes; shrink;) {   // rounding up to the minimum may overshoot
    shrink = false;
    for (Layer& L : layers)
      if (used > opt.budget_bytes && L.slots > L.min_slots) {
        --L.slots;
        used -= L.slot_bytes;
        shrink = true;
      }
    shrink = shrink && used > opt.budget_bytes;
  }
  for (bool grew = true; grew;) {
    grew = false;
    for (Layer& L : layers)
      if (L.slots < L.E && used + L.slot_bytes <= opt.budget_bytes) {
        ++L.slots;
        used += L.slot_bytes;
        grew = true;
      }
  }
  pool_bytes = used;
  for (Layer& L : layers) {
    L.pool = ctx->new_buffer(L.slots * L.slot_bytes);
    L.slot_of.assign(L.E, -1);
    L.pinned.assign(L.E, 0);
    L.prefetched.assign(L.E, 0);
    L.prot.assign(L.E, 0);
    L.pos.resize(L.E);
    L.free_slots.clear();
    for (std::size_t s = L.slots; s-- > 0;) L.free_slots.push_back(uint32_t(s));
  }
}

void ExpertStream::Impl::pin_hot_set() {
  if (opt.hot_experts.empty() || opt.pin_fraction <= 0.0) return;
  std::ifstream in(opt.hot_experts, std::ios::binary);
  if (!in) fail("cannot read " + opt.hot_experts);
  std::stringstream ss;
  ss << in.rdbuf();
  const Json j = Json::parse(ss.str());
  const Json* lj = j.find("layers");
  if (!lj || !lj->is_object()) fail(opt.hot_experts + ": no \"layers\" object");
  struct Cand {
    uint64_t count;
    std::size_t layer;
    uint32_t e;
  };
  std::vector<Cand> c;
  for (std::size_t li = 0; li < layers.size(); ++li) {
    const Json* cj = lj->find(layers[li].name);
    if (!cj) continue;
    const auto& a = cj->as_array();
    for (std::size_t e = 0; e < std::min(a.size(), layers[li].E); ++e)
      if (const uint64_t n = a[e].as_uint(); n > 0) c.push_back({n, li, uint32_t(e)});
  }
  std::sort(c.begin(), c.end(), [](const Cand& a, const Cand& b) {
    return a.count != b.count ? a.count > b.count : a.layer != b.layer ? a.layer < b.layer : a.e < b.e;
  });
  const uint64_t cap = uint64_t(opt.pin_fraction * double(opt.budget_bytes));
  auto batch = std::make_shared<Batch>();
  std::vector<ReadTask> tasks;
  for (const Cand& k : c) {
    Layer& L = layers[k.layer];
    if (pinned_bytes + L.slot_bytes > cap) continue;
    // keep room for one call's experts (unless every expert has a slot: then the unpinned ones always fit)
    if (L.slots < L.E && L.slots - L.n_pinned - 1 < L.min_slots) continue;
    place(L, k.e, {}, tasks, batch);
    L.pinned[k.e] = 1;
    ++L.n_pinned;
    pinned_bytes += L.slot_bytes;
  }
  if (!batch->remaining) return;
  io->submit(tasks);
  const auto t0 = Clock::now();
  batch->wait();
  st.wait_seconds += since(t0);
  account(*batch, true);
}

void ExpertStream::finalize() {
  Impl& m = *p_;
  if (m.final) return;
  if (m.layers.empty()) fail("no layers registered");
  m.allocate();
  m.pin_hot_set();
  for (Impl::Layer& L : m.layers)
    L.protected_cap = std::size_t(m.opt.protected_fraction * double(L.slots - L.n_pinned));
  m.final = true;
}

bool ExpertStream::finalized() const { return p_->final; }

void ExpertStream::fetch(int layer, const uint32_t* ids, std::size_t count) {
  Impl& m = *p_;
  if (!m.final) finalize();
  Impl::Layer& L = m.layers.at(std::size_t(layer));
  m.wait_inflight(L);
  ++m.st.fetch_calls;
  std::vector<uint32_t> distinct;
  std::unordered_set<uint32_t> keep;
  for (std::size_t i = 0; i < count; ++i) {
    const uint32_t e = ids[i];
    if (e >= L.E) fail(L.name + ": expert id " + std::to_string(e) + " out of range");
    if (keep.insert(e).second) distinct.push_back(e);
  }
  std::size_t unpinned = 0;
  for (uint32_t e : distinct) unpinned += !L.pinned[e];
  if (unpinned > L.slots - L.n_pinned)
    fail(L.name + ": one call routes " + std::to_string(unpinned) + " distinct unpinned experts, the pool has " +
         std::to_string(L.slots - L.n_pinned) + " unpinned slots (smaller chunks or a larger budget)");
  std::vector<uint32_t> miss;
  for (uint32_t e : distinct) {
    ++m.st.requests;
    if (L.slot_of[e] < 0) {
      miss.push_back(e);
      continue;
    }
    ++m.st.hits;
    if (L.pinned[e]) continue;
    if (L.prefetched[e]) {
      ++m.st.prefetch_hits;
      L.prefetched[e] = 0;
    }
    m.touch(L, e);
  }
  if (miss.empty()) return;
  m.st.misses += miss.size();
  auto batch = std::make_shared<Batch>();
  std::vector<ReadTask> tasks;
  for (uint32_t e : miss) {
    m.place(L, e, keep, tasks, batch);
    m.insert(L, e);
  }
  m.io->submit(tasks);
  const auto t0 = Clock::now();
  batch->wait();
  m.st.wait_seconds += since(t0);
  m.account(*batch, true);
}

std::size_t ExpertStream::prefetch(int layer, const uint32_t* ids, std::size_t count) {
  Impl& m = *p_;
  if (!m.final) finalize();
  Impl::Layer& L = m.layers.at(std::size_t(layer));
  m.wait_inflight(L);   // normally empty: a layer's prefetch follows its previous fetch
  std::unordered_set<uint32_t> keep;
  std::vector<uint32_t> want;
  for (std::size_t i = 0; i < count; ++i) {
    const uint32_t e = ids[i];
    if (e >= L.E) fail(L.name + ": predicted expert id out of range");
    if (keep.insert(e).second && L.slot_of[e] < 0) want.push_back(e);
  }
  const std::size_t room = L.slots - L.n_pinned;
  if (want.size() > room) want.resize(room);
  if (want.empty()) return 0;
  auto batch = std::make_shared<Batch>();
  std::vector<ReadTask> tasks;
  for (uint32_t e : want) {
    m.place(L, e, keep, tasks, batch);
    m.insert(L, e);
    L.prefetched[e] = 1;
  }
  m.st.prefetch_issued += want.size();
  m.io->submit(tasks);
  L.inflight.push_back(batch);
  return want.size();
}

const ExpertStreamOptions& ExpertStream::options() const { return p_->opt; }
std::size_t ExpertStream::layers() const { return p_->layers.size(); }
const std::string& ExpertStream::layer_name(int l) const { return p_->layers.at(std::size_t(l)).name; }
std::size_t ExpertStream::num_experts(int l) const { return p_->layers.at(std::size_t(l)).E; }
std::size_t ExpertStream::slots(int l) const { return p_->layers.at(std::size_t(l)).slots; }
std::size_t ExpertStream::pinned(int l) const { return p_->layers.at(std::size_t(l)).n_pinned; }
uint64_t ExpertStream::slot_bytes(int l) const { return p_->layers.at(std::size_t(l)).slot_bytes; }
std::size_t ExpertStream::max_distinct(int l) const {
  const auto& L = p_->layers.at(std::size_t(l));
  return L.slots - L.n_pinned;
}
std::size_t ExpertStream::min_max_distinct() const {
  std::size_t v = SIZE_MAX;
  for (std::size_t l = 0; l < p_->layers.size(); ++l) v = std::min(v, max_distinct(int(l)));
  return v;
}
std::vector<uint32_t> ExpertStream::resident(int l) const {
  const auto& L = p_->layers.at(std::size_t(l));
  std::vector<uint32_t> r;
  for (std::size_t e = 0; e < L.slot_of.size(); ++e)
    if (L.slot_of[e] >= 0) r.push_back(uint32_t(e));
  return r;
}
uint64_t ExpertStream::pool_bytes() const { return p_->pool_bytes; }
uint64_t ExpertStream::pinned_bytes() const { return p_->pinned_bytes; }
uint64_t ExpertStream::expert_bytes_total() const { return p_->expert_bytes; }
const ExpertStreamStats& ExpertStream::stats() const { return p_->st; }
void ExpertStream::reset_stats() { p_->st = {}; }

std::string ExpertStream::stats_json() const {
  const Impl& m = *p_;
  const ExpertStreamStats& s = m.st;
  std::ostringstream o;
  std::size_t slots = 0, pinned = 0;
  for (const auto& L : m.layers) {
    slots += L.slots;
    pinned += L.n_pinned;
  }
  o << "{\"budget_bytes\": " << m.opt.budget_bytes << ", \"pool_bytes\": " << m.pool_bytes
    << ", \"expert_bytes_total\": " << m.expert_bytes << ", \"layers\": " << m.layers.size()
    << ", \"slots\": " << slots << ", \"slot_bytes_max\": "
    << (m.layers.empty() ? 0 : std::max_element(m.layers.begin(), m.layers.end(), [](const auto& a, const auto& b) {
          return a.slot_bytes < b.slot_bytes;
        })->slot_bytes)
    << ", \"pinned_experts\": " << pinned << ", \"pinned_bytes\": " << m.pinned_bytes << ", \"policy\": \""
    << (m.opt.policy == ExpertStreamOptions::Policy::LRU ? "lru" : "slru") << "\", \"io_threads\": " << m.opt.io_threads
    << ", \"nocache\": " << (m.opt.nocache ? "true" : "false") << ", \"prefetch\": " << (m.opt.prefetch ? "true" : "false")
    << ", \"hot_experts\": \"" << m.opt.hot_experts << "\", \"fetch_calls\": " << s.fetch_calls
    << ", \"requests\": " << s.requests << ", \"hits\": " << s.hits << ", \"hit_rate\": " << s.hit_rate()
    << ", \"prefetch_hits\": " << s.prefetch_hits << ", \"misses\": " << s.misses << ", \"evictions\": " << s.evictions
    << ", \"prefetch_issued\": " << s.prefetch_issued << ", \"prefetch_wasted\": " << s.prefetch_wasted
    << ", \"read_calls\": " << s.read_calls << ", \"bytes_read\": " << s.bytes_read
    << ", \"sync_bytes_read\": " << s.sync_bytes_read << ", \"read_ms\": " << 1e3 * s.read_seconds
    << ", \"wait_ms\": " << 1e3 * s.wait_seconds << "}";
  return o.str();
}

}  // namespace suc::metal
