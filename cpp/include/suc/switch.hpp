// MoE on the Metal backend (milestone 4): expert stacks and the Qwen3-MoE sparse block, executed from a mapped SUC1
// file with the kernels of the Python/MLX runtime.
//
//   SwitchLinear  the E expert tensors "<prefix>.experts.{e}.<proj>.weight" of one projection as ONE launch plan,
//                 = suc.kernels.suc_switch.SucSwitchLinear, "gather" path (BT = 1, one tile per row, row n through
//                 expert eidx[n]):
//                   x̃ = V_e x      rotation (per-row sign table, or the plain launch when every expert has the same
//                                  seed_in), rounded to the activation dtype
//                   ỹ = W'_e x̃     suc_switch_lookup_* (vq_matmul bodies behind the tile -> expert prologue); float32
//                                  when a later stage follows
//                   y = U_eᵀ ỹ     inverse rotation (per-row or shared signs)
//                   + R_e x        stacked CSR residual (float32), then one rounding to the activation dtype
//   MoeBlock      router + routing + SwitchGLU + combine of one layer = mlx-lm Qwen3MoeSparseMoeBlock with
//                 SucSwitchLinear experts (suc.runtime.suc1_model.load_suc1_model): gates = x Wᵀ (MLX gemv order),
//                 softmax (precise), top-k (argpartition order), norm_topk_prob, gate / up, swiglu, down,
//                 (y · scores).sum(-2) (bf16 products, sequential bf16 sum). No host synchronisation: the expert ids
//                 are produced and consumed on the GPU.
//
// Layouts of the expert arrays (packed indices, row scales, codebooks):
//   Stacked  the verbatim Python kernel (dense (E, M, out, words) / (E, out) / (E, M, K, G) stacks): zero-copy when the
//            experts' arrays are consecutive in the file in expert order, else built (copied) at load.
//   Table    C++-only prologue variant (suc_switch_lookup_*_tab): one base pointer per array plus a per-expert offset
//            table, so the experts are read in place from the mapping wherever the writer put them. Same kernel bodies,
//            same numerics (only the pointer arithmetic differs).
//   Auto     Stacked when it is zero-copy, else Table (never copies the expert payloads).
//   Streamed Table kernels whose packed-index / row-scale bases are the slot pool of an ExpertStream
//            (suc/expert_stream.hpp): the file payloads are never mapped for the GPU; bind_expert() points expert e's
//            offset-table entries at its slot. Codebooks, signs and residuals stay resident as in Table.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "suc/metal.hpp"

namespace suc::metal {

enum class StackLayout { Auto, Stacked, Table, Streamed };

// Expert tiles of a batched prefill: the routed rows sorted by expert into `count` tiles of `bt` rows each (one expert
// per tile, padding rows repeat a row of their tile). With Tiles, the row arguments of the expert-stack encoders are in
// this tile-major layout (N = count · bt rows, eidx = the per-row expert ids) and the lookup / gather kernels run one
// launch tile per tile (BT = bt rows sharing the expert's weight reads, expert teidx[t]); every other stage stays per
// row. Each row's arithmetic is that of the one-row kernels (see max_tile_rows), so the outputs are bit-identical to the
// untiled launch of the same rows.
struct Tiles {
  BufRef teidx;            // (count,) uint32 expert of each tile
  std::size_t count = 0;
  std::size_t bt = 1;
  // Optional host copy of teidx (count entries, alive while encoding): a mixed expert stack (suc/expert_mixed.hpp)
  // then runs each sub-stack only over the tile range [first, last] of the tiles of its experts instead of all tiles
  // (plan the tiles with plan_tiles(..., rank) so that each sub-stack's tiles are contiguous).
  const uint32_t* host_teidx = nullptr;
  // false (batched prefill, not row-exact): tiles up to 16 rows on the simdgroup-matrix kernels (lookup "mma" from the
  // tuned mma_min_batch rows, ds4_mxfp4_mm_tile from 8), no row-exactness check of bt.
  bool exact = true;
};

// Host plan of the expert tiles of routed rows (row id, expert): rows grouped by expert (ascending ids, rows in the
// given order), cut into tiles of bt rows; a partial tile is padded with copies of its first row (dst = ~0u: never
// written back). bt is the power of two <= max_bt that minimises the estimated tile cost Σ_tiles (1 + α (bt − 1))
// (α = $SUC_TILE_ALPHA, default 0.3: the measured extra cost of a vec lookup row relative to its weight reads);
// $SUC_TILE_BT forces bt (clipped to max_bt). Any grouping gives the same bits (rows are independent).
// rank (optional, indexed by expert id): experts are ordered by (rank, id) instead of id, e.g. by sub-stack of a mixed
// expert stack so that each sub-stack's tiles are contiguous.
struct TilePlan {
  std::size_t bt = 1;
  std::vector<uint32_t> teidx;   // (count,) expert of each tile
  std::vector<uint32_t> src;     // (count · bt,) source row id of each tile row
  std::vector<uint32_t> dst;     // (count · bt,) destination row id, ~0u for padding
  std::vector<uint32_t> eidx;    // (count · bt,) expert of each tile row
  std::size_t count() const { return teidx.size(); }
  std::size_t rows() const { return src.size(); }
};
// alpha >= 0 replaces the default 0.3 (batched prefill on the simdgroup-matrix tiles: a row costs less); $SUC_TILE_ALPHA
// overrides both.
TilePlan plan_tiles(const std::vector<std::pair<uint32_t, uint32_t>>& rows, std::size_t max_bt,
                    const std::vector<uint32_t>* rank = nullptr, double alpha = -1.0);
// Runs of consecutive tiles of one expert (host teidx of `count` tiles), each of at most max_tiles tiles: (first tile,
// tiles) pairs, for the run-merging GEMM tile kernels of the batched prefill (an expert's weights decoded once per run).
std::vector<std::pair<uint32_t, uint32_t>> tile_runs(const uint32_t* teidx, std::size_t count, std::size_t max_tiles);
// Upper bound of plan_tiles(...).rows() for n routed rows over E experts.
inline std::size_t max_tile_plan_rows(std::size_t n, std::size_t E, std::size_t max_bt) {
  return n + std::min(n, E) * (max_bt - 1);
}

class ExpertStream;

class SwitchLinear {
 public:
  // Expert tensor name "<prefix>.experts.{e}.<proj>.weight".
  static std::string expert_name(const std::string& prefix, std::size_t e, const std::string& proj);
  // Number of consecutive experts 0, 1, ... present in the file for (prefix, proj).
  static std::size_t count_experts(const Suc1File& f, const std::string& prefix, const std::string& proj);
  // num_experts = 0: count them. Every expert must be a vq / additive / tcq tensor with the same shape and code layout
  // (codec, index bits, M, K, G, rotation sides; tcq: the TcqLayout key) and a residual or none: CSR (sparse_*), or column-structured
  // (columns_*: per-expert column sets with an optional CSR tail, as SucSwitchLinear._residual / _column_residual).
  static SwitchLinear from_file(Context& ctx, const MappedFile& mf, const std::string& prefix, const std::string& proj,
                                std::size_t num_experts = 0, StackLayout layout = StackLayout::Auto);
  // Sub-stack over expert SLOTS (mixed-codec stacks, suc/expert_mixed.hpp): slot s (= the global expert id a routed
  // row carries) reads expert slots[s] of the file, or is empty (-1): the lookup kernel skips rows routed to an empty
  // slot (no weight reads; their output rows are undefined and must not be used). Always the Table layout; num_experts()
  // is slots.size(). streamed: StackLayout::Streamed instead (slots must then be s or -1; only the member slots are
  // bound / registered with an ExpertStream).
  static SwitchLinear from_file_slots(Context& ctx, const MappedFile& mf, const std::string& prefix,
                                      const std::string& proj, const std::vector<int64_t>& slots,
                                      bool streamed = false);
  bool empty_slot(std::size_t e) const { return e < empty_.size() && empty_[e]; }

  const std::string& name() const { return name_; }
  std::size_t num_experts() const { return E_; }
  std::size_t in_features() const { return in_; }
  std::size_t out_features() const { return out_; }
  std::size_t num_codebooks() const { return M_; }
  unsigned bits() const { return bits_; }
  // Trellis-coded experts (representation tcq, = SucSwitchLinear with tcq params): one TcqLayout for the stack, the
  // kernels suc_tcq_switch_* (Stacked) / suc_tcq_switch_*_tab (Table, Streamed); no codebooks.
  bool tcq() const { return tcq_; }
  bool shared_codebooks() const { return !cbe_; }        // CBE = 0: one codebook set serves every expert
  bool rotate_in() const { return rot_in_; }
  bool rotate_out() const { return rot_out_; }
  bool shared_signs_in() const { return rot_in_ && !rows_in_; }
  bool shared_signs_out() const { return rot_out_ && !rows_out_; }
  bool has_residual() const { return n_res_ > 0 || n_cols_ > 0; }
  std::size_t residual_entries() const { return n_res_; }
  std::size_t residual_columns() const { return n_cols_; }   // Σ over experts of their residual columns
  StackLayout layout() const { return layout_; }          // Stacked, Table or Streamed (resolved)
  // Streamed: file ranges (offset, bytes) of expert e's packed indices and row scales (the arrays the kernel reads
  // from the slot pool), and the binding of expert e to its slot (byte offsets of those arrays in `pool`; one pool
  // buffer per stack).
  std::pair<std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>> expert_ranges(std::size_t e) const;
  void bind_expert(std::size_t e, MTL::Buffer* pool, uint64_t packed_at, uint64_t scales_at);
  std::size_t weight_bytes() const { return weight_bytes_; }
  std::size_t copied_bytes() const { return copied_bytes_; }
  std::size_t zero_copy_bytes() const { return zero_copy_bytes_; }

  // y (N, out) = rows of x (N, in) through experts eidx (N,) uint32 (GPU buffer), x / y in `act`. Encodes 1-4
  // dispatches separated by memory barriers; ws (xr: N·in·act, y32 / y32b: N·out·4 bytes) grows on demand and must
  // outlive the command buffer. Rows are independent (any order, repeated experts allowed).
  void encode(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef eidx, BufRef y, std::size_t N, Act act,
              Workspace& ws, const Tiles* tiles = nullptr) const;
  // Largest tile height whose lookup launch is row-for-row bit-identical to the one-row launch of decode: the vec
  // lookup keeps its per-lane group assignment (CHUNK) for every BT up to this value; 1 when decode runs the mma path.
  std::size_t max_tile_rows(const Context& ctx) const;
  // encode() split at its memory barriers, so that several stacks can share them (ExpertStack::encode_group): stage
  // st in [0, kStages) encodes the launches encode() places between its st-th and (st+1)-th barrier and returns false
  // when that stage is empty for this stack. Running the stages in order with a barrier after every non-empty one
  // encodes exactly encode() (same kernels, buffers and order). prepare() must run before the first stage.
  // masked: the last stage writes only the rows whose expert is a member of this (slot) sub-stack and leaves the other
  // rows of y untouched; requires routed_writes().
  static constexpr int kStages = 5;
  void prepare(Context& ctx, Workspace& ws, std::size_t N, Act act) const;
  bool encode_stage(int st, Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef eidx, BufRef y,
                    std::size_t N, Act act, Workspace& ws, bool masked = false, const Tiles* tiles = nullptr) const;
  // The last stage is the column residual over an offset table with empty slots marked (masked writes possible).
  bool routed_writes() const { return n_cols_ > 0 && eoff_.get() != nullptr; }
  // Synchronous convenience: x (N·in floats, rounded to act first), eidx (N) -> y (N·out floats, values of act).
  std::vector<float> forward(Context& ctx, const std::vector<float>& x, const std::vector<uint32_t>& eidx, Act act) const;

 private:
  static SwitchLinear build(Context& ctx, const MappedFile& mf, const std::string& prefix, const std::string& proj,
                            const std::vector<int64_t>& slots, StackLayout layout, bool subset);
  void encode_rotation(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef eidx, BufRef y, std::size_t N,
                       bool inverse, Act xa, Act ya) const;
  void encode_kernel(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef eidx, BufRef y, std::size_t N,
                     Act xa, Act ya, const Tiles* tiles = nullptr) const;
  void encode_column_residual(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef eidx, BufRef y32,
                              BufRef y, std::size_t N, Act act, bool masked = false) const;

  std::string name_;
  std::size_t E_ = 0, out_ = 0, in_ = 0, M_ = 1, K_ = 0, G_ = 0, words_ = 0;
  unsigned bits_ = 0;
  bool cbe_ = true;
  bool tcq_ = false;
  TcqLayout tcq_l_;
  BufPtr tcq_cst_;
  StackLayout layout_ = StackLayout::Stacked;
  BufRef codebooks_, packed_, scales_;
  BufPtr eoff_;                                   // Table: (E, 3) ulong element offsets (codebook, packed, scales)
  std::vector<uint64_t> pk_off_, sc_off_;         // Streamed: file offsets of each expert's indices / scales
  uint64_t pk_bytes_ = 0, sc_bytes_ = 0;
  bool rot_in_ = false, rot_out_ = false, rows_in_ = false, rows_out_ = false;
  BufPtr signs_in_, signs_out_, hm_in_, hm_out_;  // float32 (1, n) shared or (E, n) per-row tables
  std::size_t n_res_ = 0;
  bool rcols_u16_ = true, rvals_i8_ = true;
  BufPtr row_ptr_, rcols_, rvals_, rscale_;
  // column residual: cptr (E + 1) uint32, ccols (C) uint16/32, cval (C, out) int8/fp16, cscale (C) float32
  std::size_t n_cols_ = 0, cmax_ = 0;
  bool ccols_u16_ = true, cval_i8_ = true;
  BufPtr cptr_, ccols_, cval_, cscale_;
  std::vector<BufPtr> owned_;
  std::vector<uint8_t> empty_;                    // slot sub-stacks: 1 = empty slot
  std::size_t weight_bytes_ = 0, copied_bytes_ = 0, zero_copy_bytes_ = 0;
};

// Shape of one sparse MoE block (fields of the HF config.json / ModelConfig).
struct MoeConfig {
  std::size_t hidden = 0;            // hidden_size
  std::size_t num_experts = 0;       // num_experts
  std::size_t top_k = 0;             // num_experts_per_tok
  std::size_t intermediate = 0;      // moe_intermediate_size
  bool norm_topk_prob = true;
};

// Per-call scratch of MoeBlock::encode (grows on demand; must outlive the command buffer).
struct MoeScratch {
  BufPtr gates, eidx, scores, xk, g, u, h, yk, ykr;
  std::size_t gates_b = 0, eidx_b = 0, scores_b = 0, xk_b = 0, g_b = 0, u_b = 0, h_b = 0, yk_b = 0, ykr_b = 0;
  Workspace ws_gate, ws_up, ws_down;
  std::size_t bytes() const;
};

class MoeBlock {
 public:
  // layer_prefix "model.layers.{L}": router "<p>.mlp.gate.weight" (E, hidden) and the expert stacks
  // "<p>.mlp.experts.{e}.{gate,up,down}_proj.weight". The router weight is read in place when stored in `act`
  // (raw), else converted once to `act` (the Python reference casts it to the model dtype too).
  static MoeBlock from_file(Context& ctx, const MappedFile& mf, const std::string& layer_prefix, const MoeConfig& cfg,
                            Act act, StackLayout layout = StackLayout::Auto);

  const MoeConfig& config() const { return cfg_; }
  Act act() const { return act_; }
  const SwitchLinear& gate_proj() const { return gate_; }
  const SwitchLinear& up_proj() const { return up_; }
  const SwitchLinear& down_proj() const { return down_; }
  std::size_t copied_bytes() const;
  std::size_t weight_bytes() const;

  // SSD expert streaming (suc/expert_stream.hpp). Requires the stacks loaded with StackLayout::Streamed and a fixed
  // address of this block (held by pointer) from here on. Registers the layer "<layer_prefix>.mlp" (the hot-set key)
  // with per expert the ranges {gate, up, down} x {packed indices, row scales} and min_slots = top_k.
  void attach_stream(ExpertStream& s);
  bool streamed() const { return stream_ != nullptr; }
  ExpertStream* stream() const { return stream_; }
  int stream_layer() const { return stream_layer_; }
  // Host side of a streamed layer, after the routing ran on the GPU and before encode_experts: make the routed
  // experts (eidx: count host-visible uint32) resident (blocking reads of the misses); prefetch: asynchronous reads
  // of predicted ids.
  void stream_in(const uint32_t* eidx, std::size_t count) const;
  void stream_prefetch(const uint32_t* eidx, std::size_t count) const;

  // y (n, hidden) = block(xn (n, hidden)), both in act() (the caller adds the residual stream). Router -> routing ->
  // experts -> combine, memory barriers between dependent launches, no host sync. y must not alias xn.
  void encode(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef xn, BufRef y, std::size_t n, MoeScratch& s) const;
  // Router + routing only: eidx (n, top_k) uint32 and scores (n, top_k) in act().
  void encode_route(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef xn, BufRef eidx, BufRef scores,
                    std::size_t n, MoeScratch& s) const;
  // Experts + combine with a given routing (eidx (n, top_k) uint32, scores (n, top_k) in act()).
  void encode_experts(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef xn, BufRef eidx, BufRef scores, BufRef y,
                      std::size_t n, MoeScratch& s) const;
  // encode_experts in expert tiles (batched prefill): tile row j computes token tok[j]'s row through expert reid[j]
  // (tile-major layout of suc/switch.hpp, tiles.count · tiles.bt rows) and is written back to routed row dst[j]
  // (n · top_k + k; ~0u = padding); then the same combine. Bit-identical to encode_experts. Row ids and tiles must
  // cover every routed row of the n tokens exactly once.
  // combine = false: stop after the write-back (several tile groups of one chunk, e.g. streamed fetch groups), then
  // encode_combine once.
  void encode_experts_tiled(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef xn, BufRef tok, BufRef dst,
                            BufRef reid, const Tiles& tiles, BufRef scores, BufRef y, std::size_t n,
                            MoeScratch& s, bool combine = true) const;
  void encode_combine(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef scores, BufRef y, std::size_t n,
                      MoeScratch& s) const;
  std::size_t max_tile_rows(const Context& ctx) const;
  // Largest tile height of the batched (not row-exact) prefill: 16, 8 with tcq experts (see ExpertStack::fast_tile_rows).
  std::size_t fast_tile_rows() const;

 private:
  MoeConfig cfg_;
  Act act_ = Act::BF16;
  std::string prefix_;
  ExpertStream* stream_ = nullptr;
  int stream_layer_ = -1;
  BufRef router_;
  BufPtr router_owned_;
  SwitchLinear gate_, up_, down_;
};

}  // namespace suc::metal
