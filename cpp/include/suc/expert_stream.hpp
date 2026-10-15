// SSD expert streaming for the Metal runtime (docs/expert_streaming.md): routed MoE experts are read on demand from the
// SUC1 file into a fixed pool of GPU slots per MoE layer, under one byte budget for all pools.
//
//   pools     one shared-storage MTLBuffer per layer of S_l slots; S_l from the budget (same capacity fraction per
//             layer, at least the layer's minimum, typically top-k), slot = the largest expert's streamed bytes.
//   reads     pread on a descriptor with F_NOCACHE (no page-cache copy: the slot is the only copy), on a pool of IO
//             threads; ALL misses of a layer are issued together (the SSD needs concurrent reads), each run of adjacent
//             arrays as one read (split into <= 1 MiB pieces). The stored bytes are the kernel's operands (packed u32
//             index words, f16 row scales, mxfp4 codes / E8M0 scales, ...): no conversion.
//   policy    per layer LRU or segmented LRU (probation + protected 80 %); a pinned hot set (the most-routed experts
//             of the calibration pass, hot_experts.json of `suc compress`) is read once and never evicted.
//   prefetch  optional: prefetch(layer, ids) reads predicted experts asynchronously (e.g. the next layer's router on
//             the current hidden state); fetch() waits for a layer's pending prefetch reads before touching its slots.
//
// Codec-agnostic hook. A consumer (an expert stack of any codec) registers, per expert, the file byte ranges its kernel
// reads, and a Bind callback; when an expert is installed in a slot the stream calls bind(expert, pool, at) with the
// pool buffer and the byte offset of each registered range inside it, and the consumer points its per-expert offset
// table there (metal::SwitchLinear's StackLayout::Streamed; see MoeBlock::attach_stream). The graph's protocol per
// streamed MoE layer: encode the router, end the command buffer and wait (split_command_buffer), fetch(layer, routed
// ids) on the host, then encode the experts. Command buffers that read a layer's slots must have completed before the
// next fetch of that layer (the graphs run streamed decode with one command buffer in flight).
//
// The routed-expert payloads must stay OUT of the GPU views of the file mapping (MappedFile(ctx, f,
// routed_expert_ranges(f))): Metal makes every page of a no-copy buffer resident when a command buffer uses it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "suc/metal.hpp"

namespace suc::metal {

struct ExpertStreamOptions {
  uint64_t budget_bytes = 0;      // all slot pools together
  std::string hot_experts;        // hot_experts.json ({"layers": {"<prefix>.mlp": [counts]}}); "" = no pinned set
  double pin_fraction = 0.5;      // at most this share of the budget pinned (greedy by routing count over all layers)
  enum class Policy { LRU, SLRU } policy = Policy::SLRU;
  double protected_fraction = 0.8;   // SLRU: protected segment share of a layer's unpinned slots
  unsigned io_threads = 8;
  bool nocache = true;            // F_NOCACHE on the read descriptor
  bool prefetch = false;          // graphs: predict the next MoE layer's experts and prefetch them
  uint64_t merge_gap = 4096;      // registered ranges closer than this in the file are read as one run
  uint64_t max_read = 1u << 20;   // runs are split into reads of at most this many bytes
};

struct ExpertStreamStats {
  uint64_t fetch_calls = 0, requests = 0, hits = 0, prefetch_hits = 0, misses = 0, evictions = 0;
  uint64_t prefetch_issued = 0, prefetch_wasted = 0;
  uint64_t read_calls = 0, bytes_read = 0, sync_bytes_read = 0;
  double read_seconds = 0.0;      // sum over reads of their wall time (IO threads)
  double wait_seconds = 0.0;      // host time blocked in fetch() on reads (sync misses + pending prefetches)
  double hit_rate() const { return requests ? double(hits) / double(requests) : 0.0; }
};

// File byte ranges [begin, end) of every array of every routed-expert tensor ("<prefix>.experts.<e>.<rest>"), merged
// across alignment padding only (gaps < 64 bytes). For MappedFile's exclude list.
std::vector<std::pair<uint64_t, uint64_t>> routed_expert_ranges(const Suc1File& f);

// Ends `enc`, commits `cb` and waits for it (throws when it failed), then opens a new command buffer and concurrent
// encoder on the context's queue: the host-sync point of a streamed MoE layer. *gpu_s accumulates the GPU time.
void split_command_buffer(Context& ctx, MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc,
                          double* gpu_s = nullptr);

class ExpertStream {
 public:
  using Range = std::pair<uint64_t, uint64_t>;   // (file offset, bytes)
  using Bind = std::function<void(uint32_t expert, MTL::Buffer* pool, const uint64_t* at)>;

  ExpertStream(Context& ctx, const Suc1File& f, const ExpertStreamOptions& opt);
  ~ExpertStream();
  ExpertStream(const ExpertStream&) = delete;
  ExpertStream& operator=(const ExpertStream&) = delete;

  // Registration (before finalize): `ranges[e]` = the byte ranges expert e's kernels read (same count for every
  // expert; sizes may differ); min_slots = the most distinct experts one call may need (top-k for one token). The
  // name keys the hot set ("model.layers.N.mlp"). Returns the layer id.
  int add_layer(const std::string& name, std::vector<std::vector<Range>> ranges, std::size_t min_slots, Bind bind);
  // Splits the budget into the per-layer pools (throws when it cannot hold every layer's minimum), allocates them and
  // reads the pinned hot set. Called by the graph after its layers are attached; fetch() calls it if needed.
  void finalize();
  bool finalized() const;

  // Makes the experts ids[0..count) (repeats allowed) of `layer` resident: hits are touched, all misses are read
  // together (blocking). Distinct unpinned ids must fit the layer's unpinned slots (max_distinct).
  void fetch(int layer, const uint32_t* ids, std::size_t count);
  // Starts asynchronous reads of the non-resident ids of `layer` (a prediction); returns the number issued.
  std::size_t prefetch(int layer, const uint32_t* ids, std::size_t count);

  const ExpertStreamOptions& options() const;
  std::size_t layers() const;
  const std::string& layer_name(int layer) const;
  std::size_t num_experts(int layer) const;
  std::size_t slots(int layer) const;
  std::size_t pinned(int layer) const;
  uint64_t slot_bytes(int layer) const;
  std::size_t max_distinct(int layer) const;      // unpinned slots
  std::size_t min_max_distinct() const;           // min over layers
  std::vector<uint32_t> resident(int layer) const;   // experts in slots (pinned included), ascending
  uint64_t pool_bytes() const;                    // Σ slots · slot bytes (<= budget)
  uint64_t pinned_bytes() const;
  uint64_t expert_bytes_total() const;            // Σ over layers and experts of the registered bytes
  const ExpertStreamStats& stats() const;
  void reset_stats();
  std::string stats_json() const;                 // one JSON object (options, pools, stats)

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace suc::metal
