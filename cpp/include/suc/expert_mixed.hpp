// Expert stacks with native MXFP4 experts and mixed codecs (DeepSeek-V4 files of `suc compress` with the per-expert
// allocation unit and the native candidate), on the Metal backend.
//
//   Mxfp4Switch  native MXFP4 experts (SUC1 representation "mxfp4", docs/format.md §5.4): the verbatim E2M1 codes and
//                E8M0 scales read in place from the mapping (one per-expert offset table), run by a port of MLX's
//                fp_gather_qmv / fp_gather_qmv_fast (mlx/backend/metal/kernels/fp_quantized.h; the kernels of
//                mx.gather_qmm(..., mode="mxfp4") in suc.kernels.mxfp4.MXFP4SwitchLinear): row n of x through expert
//                eidx[n], float32 accumulation in MLX's order, one rounding to the activation dtype.
//   ExpertStack  one projection (gate / up / down) of a layer's routed experts = suc.kernels.suc_switch
//                .switch_from_encodings: a SwitchLinear when every expert has one code layout, an Mxfp4Switch when all
//                are native, else a mixed stack (SucSwitchMixed): one sub-stack per layout (native experts form one),
//                each addressed by GLOBAL expert ids over slots (SwitchLinear::from_file_slots), and a select launch
//                that takes row n from the sub-stack holding eidx[n].
//                  Rows::Routed (default)  rows routed to another sub-stack's experts are skipped by each sub-stack's
//                                          lookup / gather kernel (no weight reads for them): every routed row reads
//                                          its expert's codes once.
//                  Rows::All               every sub-stack runs every routed row (rows of other sub-stacks through its
//                                          first expert, the Python SucSwitchMixed semantics: local index 0, masked).
//                Both give the same bits (each row's result comes from its own sub-stack, computed by the same kernels).
// No host synchronisation: expert ids are produced and consumed on the GPU.
//
// SSD expert streaming (suc/expert_stream.hpp): built with streamed = true, no expert payload is mapped for the GPU;
// per expert the stack exposes the two file ranges its kernel reads (SUC: packed indices, row scales; native: codes,
// scales) and bind_expert() points the owning sub-stack's offset table at the expert's slot in the stream's pool.
// Rows::All is not available when streamed (it reads a sub-stack's first expert for foreign rows, which need not be
// resident).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "suc/metal.hpp"
#include "suc/switch.hpp"

namespace suc::metal {

class Mxfp4Switch {
 public:
  // slots[s] = expert id of the file read by slot s (row expert ids are slot ids), or -1 (empty: rows routed there are
  // skipped, their output rows are left untouched). Every member must be an mxfp4 tensor of the same shape.
  // streamed: slots must be s or -1; the codes / scales bases become the stream's pool on the first bind_expert.
  static Mxfp4Switch from_file(Context& ctx, const MappedFile& mf, const std::string& prefix, const std::string& proj,
                               const std::vector<int64_t>& slots, bool streamed = false);
  std::size_t in_features() const { return in_; }
  std::size_t out_features() const { return out_; }
  std::size_t members() const { return members_; }
  std::size_t weight_bytes() const { return weight_bytes_; }
  std::size_t copied_bytes() const { return copied_bytes_; }
  bool fast() const { return fast_; }   // MLX fp_gather_qmv_fast (in % 512 == 0, out % 8 == 0)
  bool streamed() const { return streamed_; }
  // Streamed: file ranges (offset, bytes) of slot e's codes and scales; binding of slot e to its stream slot.
  std::pair<std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>> expert_ranges(std::size_t e) const;
  void bind_expert(std::size_t e, MTL::Buffer* pool, uint64_t codes_at, uint64_t scales_at);
  // y (N, out) = x (N, in) through slot eidx[n] (act dtype in and out). One dispatch.
  // tiles: x / y in the tile-major layout of suc/switch.hpp (N = count · bt rows), one ds4_mxfp4_qmv_tile launch
  // tile per tile (rows bit-identical to the one-row kernel).
  void encode(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef eidx, BufRef y, std::size_t N,
              Act act, const Tiles* tiles = nullptr) const;

 private:
  std::size_t in_ = 0, out_ = 0, members_ = 0;
  bool fast_ = false, streamed_ = false;
  std::vector<uint64_t> c_off_, s_off_;   // streamed: file offsets per slot (~0 = empty)
  BufRef codes_, scales_;   // views holding every member's arrays (offset table relative to them)
  BufPtr eoff_;             // (slots, 2) ulong byte offsets of codes / scales; ~0 = empty slot
  std::size_t weight_bytes_ = 0, copied_bytes_ = 0;
};

struct ExpertScratch {
  std::vector<Workspace> ws;   // one per sub-stack
  BufPtr ysub;                 // (sub-stacks, N, out) in act
  std::size_t ysub_b = 0;
  std::size_t bytes() const;
};

class ExpertStack {
 public:
  enum class Rows { Routed, All };
  // The E experts "<prefix>.experts.{e}.<proj>.weight" (E = 0: count them).
  static ExpertStack from_file(Context& ctx, const MappedFile& mf, const std::string& prefix, const std::string& proj,
                               std::size_t num_experts = 0, Rows rows = Rows::Routed, bool streamed = false);

  std::size_t num_experts() const { return E_; }
  std::size_t in_features() const { return in_; }
  std::size_t out_features() const { return out_; }
  bool mixed() const { return sub_suc_.size() + sub_nat_.size() > 1; }
  std::size_t sub_stacks() const { return sub_suc_.size() + sub_nat_.size(); }
  std::size_t native_experts() const { return native_; }
  bool native_only() const { return native_ == E_; }
  Rows rows() const { return rows_; }
  std::size_t weight_bytes() const;
  std::size_t copied_bytes() const;
  // Expert e's sub-stack: (true, index into the native sub-stacks) or (false, index into the SUC sub-stacks).
  std::pair<bool, std::size_t> group_of(std::size_t e) const { return {nat_of_[e], grp_[e]}; }
  // Sub-stack slot of expert e: SUC sub-stacks 0 .. n_suc - 1, then the native ones (0 for a uniform stack).
  std::size_t sub_slot(std::size_t e) const { return nat_of_[e] ? sub_suc_.size() + grp_[e] : grp_[e]; }
  bool streamed() const { return streamed_; }
  // Streamed: the two file ranges (offset, bytes) expert e's kernel reads, and its binding to a stream slot (byte
  // offsets of those ranges in `pool`), dispatched to the sub-stack holding e.
  std::pair<std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>> expert_ranges(std::size_t e) const;
  void bind_expert(std::size_t e, MTL::Buffer* pool, uint64_t a0, uint64_t a1);

  // y (N, out) = rows of x (N, in) through experts eidx (N,) uint32, x / y in act. Memory barriers inside; scratch
  // must outlive the command buffer.
  void encode(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef eidx, BufRef y, std::size_t N, Act act,
              ExpertScratch& s) const;

  // Several stacks over the same routed rows (e.g. gate and up of one layer) encoded phase by phase: the stages of all
  // their sub-stacks (SwitchLinear::encode_stage, the native gather in the lookup stage) share one memory barrier per
  // stage instead of one per sub-stack and stage. A mixed Rows::Routed stack whose SUC sub-stacks end in a column
  // residual over an offset table writes each sub-stack's rows straight into y (masked residual; native rows are
  // skipped by the gather), so it needs no ysub and no select launch; otherwise ysub + select as encode(). The bits of
  // every output row are those of encode(): the same kernels on the same inputs, only the barriers and the row copy
  // differ. A barrier follows the last launch.
  struct Job {
    const ExpertStack* stack;
    BufRef x, y;
    ExpertScratch* scratch;
  };
  // tiles: rows in the tile-major layout of suc/switch.hpp (eidx = per-row expert ids, N = count · bt), lookup /
  // gather launches per tile; every row's bits are those of the untiled call. With tiles->host_teidx, each sub-stack
  // of a mixed stack runs only over the tile range of its own experts (rotations, lookup, residual over those rows;
  // other rows of the range are computed by the same kernels and discarded as before), not over all count tiles.
  // $SUC_MIXED_ALL_TILES=1: all tiles (the previous behaviour, A/B).
  static void encode_group(Context& ctx, MTL::ComputeCommandEncoder* enc, const std::vector<Job>& jobs, BufRef eidx,
                           std::size_t N, Act act, const Tiles* tiles = nullptr);
  // Largest row-exact tile height over the sub-stacks (SwitchLinear::max_tile_rows; native gathers: 16).
  std::size_t max_tile_rows(const Context& ctx) const;
  // Largest tile height of the batched (not row-exact) prefill: 16 (the vq / additive "mma" lookup tile, the native
  // GEMM tile; tcq sub-stacks run each 16-row tile as two 8-row tiles, see SwitchLinear). $SUC_FAST_TILE_ROWS overrides.
  std::size_t fast_tile_rows() const;
  bool routed_writes() const;   // encode_group writes y directly (no select) for this stack

 private:
  std::size_t E_ = 0, in_ = 0, out_ = 0, native_ = 0;
  Rows rows_ = Rows::Routed;
  bool streamed_ = false;
  std::vector<SwitchLinear> sub_suc_;
  std::vector<Mxfp4Switch> sub_nat_;
  std::vector<uint8_t> nat_of_;
  std::vector<std::size_t> grp_;
  BufPtr sel_;   // (E,) uint32: sub-stack slot of expert e in the ysub layout (SUC sub-stacks first, then native)
};

}  // namespace suc::metal
