// Metal backend (milestone 2): one SUC linear layer y = x Ŵᵀ executed on the GPU straight from a mapped
// SUC1 file, with the kernels of the Python/MLX runtime (cpp/metal/suc_kernels.metal is generated from
// suc.kernels.vq_matmul, suc.kernels.rotation and suc.runtime.compressed_model; see cpp/README.md).
//
//   Context     device, command queue, run-time compiled kernel library (one pipeline per template
//               instantiation, cached), and the per-device tuning choices (same cache file and defaults as
//               suc.kernels.autotune / vq_matmul / rotation).
//   MappedFile  zero-copy MTLBuffers over the whole mmapped SUC1 file (page-aligned base, length rounded up
//               to the page size, split into views of at most maxBufferLength when the file is larger; the views
//               are cut between mapping units so that they overlap by at most one page, see MappedFile).
//               Payload segments are 64-byte aligned, so every array is addressed as (view, byte offset).
//   Linear      execution plan of one vq / additive tensor (optionally with rotations and residuals),
//               = suc.kernels.suc_linear.SucLinear / suc.runtime.compressed_model.SucResidualLinear:
//                 x̃ = V x (rotation kernel, rounded to the activation dtype)
//                 ỹ = W' x̃ (lookup kernel: "vec" for batch tiles < mma_min_batch, simdgroup-matrix "mma" from it)
//                 y = Uᵀ ỹ (inverse rotation) [+ R x (structured residual kernel: columns + CSR tail)]
//               float32 between stages, one rounding to the activation dtype at the end.
//
// metal-cpp types appear in this header; include paths: -isystem cpp/third_party/metal-cpp.
#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "suc/suc1.hpp"

namespace suc::metal {

// Activation dtypes (x and y of a layer).
enum class Act : int { F16, BF16, F32 };
Act parse_act(std::string_view name);   // "f16" | "bf16" | "f32"
std::string_view act_name(Act a);
std::size_t act_size(Act a);
const char* act_msl(Act a);             // "half" | "bfloat" | "float"
// Host conversions (round to nearest even, as numpy / MLX astype).
void f32_to_act(const float* src, void* dst, std::size_t n, Act a);
void act_to_f32(const void* src, float* dst, std::size_t n, Act a);
uint16_t f32_to_f16_bits(float f);
uint16_t f32_to_bf16_bits(float f);

using BufPtr = NS::SharedPtr<MTL::Buffer>;

// A byte range inside a GPU buffer.
struct BufRef {
  MTL::Buffer* buffer = nullptr;
  std::size_t offset = 0;
  BufRef at(std::size_t extra) const { return {buffer, offset + extra}; }
  explicit operator bool() const { return buffer != nullptr; }
};

// Kernel configuration; defaults are the module constants of the Python kernels.
struct Tuning {
  std::string codebook_in = "auto";  // "auto" | "device" | "threadgroup" (vq_matmul.CODEBOOK_IN)
  int mma_min_batch = 0;             // 0: tuned choice, else 5 (vq_matmul.MMA_MIN_BATCH_DEFAULT)
  int lookup_sg = 8;                 // vq_matmul.LOOKUP_SG
  int lookup_vec = 1;                // vq_matmul.LOOKUP_VEC
  int mma_sg = 4, mma_rt = 4, mma_kc = 64;  // vq_matmul.MMA_SG / MMA_RT / MMA_KC (MMA_KIND = "reg")
  int max_batch_tile = 16;           // vq_matmul.MAX_BATCH_TILE
  int block_q = 0, block_sg = 0;     // rotation.BLOCK_Q / BLOCK_SG (0 = tuned choice, else default_block_config)
  bool fast_math = false;            // MLX compiles mx.fast.metal_kernel sources with fast math disabled
  // kernel_tuning.json of suc.kernels.autotune: "" = $SUC_TUNING_CACHE or ~/.cache/suc/kernel_tuning.json,
  // "-" = do not read any cache (built-in defaults only).
  std::string tuning_cache;
};

// Key of a pipeline lookup that avoids formatting the template-argument string on the hot path: a call-site id and up
// to 7 integers that determine (kernel, template arguments) at that site (object pointers, sizes, dtypes).
struct PsoKey {
  uint32_t site = 0;
  int64_t v[7] = {0, 0, 0, 0, 0, 0, 0};
  bool operator==(const PsoKey& o) const {
    if (site != o.site) return false;
    for (int i = 0; i < 7; ++i)
      if (v[i] != o.v[i]) return false;
    return true;
  }
};

class Context {
 public:
  explicit Context(const Tuning& t = {});
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  MTL::Device* device() const;
  MTL::CommandQueue* queue() const;
  const std::string& device_name() const;
  const std::string& architecture() const;   // e.g. "applegpu_g15g"
  std::string device_key() const { return architecture() + "|" + device_name(); }
  const Tuning& tuning() const;
  bool tuning_cache_found() const;

  // Resolved choices (= vq_matmul.resolve_codebook_in / mma_min_batch, rotation.block_config).
  std::string codebook_in(int m) const;                // "device" | "threadgroup"
  int mma_min_batch(int m) const;
  std::pair<int, int> rotation_block_config(std::size_t n) const;   // (Q max, simdgroups max)
  // Tuned choice of suc.kernels.autotune.fuse_elementwise_choice(op) ("norm" | "swiglu"): fuse the input rotation
  // into that element-wise launch. false when the cache has no entry (Python: FUSE_ELEMENTWISE None -> off).
  bool fuse_elementwise(const std::string& op) const;
  // Rotation layout of size n (= rotation.layout_for with LAYOUT = "auto"): the tuned choice "row" (one threadgroup
  // per row: rotation._kernel / _plan; e.g. n = 1024 on the M1 Ultra), else "blocks".
  bool rotation_row_layout(std::size_t n) const;
  // The tuning cache asks for a rotation layout this port does not have (neither "blocks" nor "row").
  bool rotation_layout_overridden(std::size_t n) const;

  // (memoized; the *_uncached variants read the tuning cache)
  std::string codebook_in_uncached(int m) const;
  int mma_min_batch_uncached(int m) const;
  std::pair<int, int> rotation_block_config_uncached(std::size_t n) const;

  // Pipeline of `kernel<targs>` (compiled from the embedded source on first use, then cached).
  MTL::ComputePipelineState* pipeline(const std::string& kernel, const std::string& targs);
  // Cached lookup by key: `make()` returns {kernel, targs} and runs only on the first use of the key.
  template <class F>
  MTL::ComputePipelineState* pipeline(const PsoKey& key, F&& make) {
    if (MTL::ComputePipelineState* ps = find_pipeline(key)) return ps;
    const std::pair<std::string, std::string> kt = make();
    MTL::ComputePipelineState* ps = pipeline(kt.first, kt.second);
    store_pipeline(key, ps);
    return ps;
  }
  MTL::ComputePipelineState* find_pipeline(const PsoKey& key) const;
  void store_pipeline(const PsoKey& key, MTL::ComputePipelineState* ps);
  double compile_seconds() const;
  std::size_t pipelines_compiled() const;

  BufPtr new_buffer(std::size_t bytes) const;                    // shared storage, zero-filled
  BufPtr upload(const void* data, std::size_t bytes) const;      // shared storage copy
  BufRef dummy() const;                                          // 64 zero bytes (absent kernel inputs)

  // Residency set (macOS 15+; SUC_RESIDENCY_SET=0 disables it). Without one, every command buffer makes the buffers it
  // uses resident again, and once the GPU allocations approach the wired limit the driver unwires idle ones between
  // command buffers: measured on the M3 (16 GB), 8 x 1.2 GB shared buffers, one per command buffer: commit + wait p50
  // 23 ms / p90 335 ms, 0.2 ms with the buffers in a residency set. A streamed MoE decode commits one command buffer per
  // MoE layer, each using a multi-GB slot pool (GLM-5.3-Flash on the Studio: ~55 ms per split, 2.58 s/token).
  // new_buffer() (>= 64 KiB), the file views and the expert pools join the queue's set while the set stays within
  // max(recommendedMaxWorkingSetSize, iogpu.wired_limit_mb); the set retains its members, members nobody else holds
  // any more are dropped at the next flush. Changes are committed lazily (flush_residency(), called by queue()).
  void make_resident(MTL::Allocation* a) const;
  void flush_residency() const;
  bool residency_set() const;              // a set is in use
  uint64_t residency_set_bytes() const;    // bytes in the set
  uint64_t residency_skipped_bytes() const;   // bytes left out (cap reached)

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

// The kernel source the Context compiles: the SUC projection kernels (embedded at build time; $SUC_METAL_SOURCE
// overrides them with a file) followed by the model-graph kernels (cpp/metal/suc_model_kernels.metal).
std::string kernel_source();

class MappedFile {
 public:
  // max_view_bytes: largest buffer per view (0 = the device's maxBufferLength; smaller values exercise the
  // multi-view path in tests).
  //
  // Splitting: a command buffer that binds a no-copy view makes the WHOLE view resident, and the residency limit
  // (recommendedMaxWorkingSetSize, ~ iogpu.wired_limit_mb) is charged per buffer, also for pages two views share.
  // The former split into half-overlapping views charged ~1.4x the file (DeepSeek-V4-Flash, 107.4 GB on the M1 Ultra
  // with maxBufferLength 86.6 GB: two views of 86.6 + 64.1 GB > the 115.3 GB working set), so the first command
  // buffer never got scheduled. The views are now cut only between mapping units (mapping_units()), so they overlap by
  // at most one page and charge ~ the file size.
  MappedFile(Context& ctx, const Suc1File& f, uint64_t max_view_bytes = 0);
  // Views that leave out the byte ranges `exclude` ([begin, end) file offsets; the whole pages inside each range):
  // a command buffer that uses a no-copy buffer makes ALL of its pages resident (measured on the M3: one float read
  // per 128 MB of an 8 GB mapped file faulted in all 8 GB), so data that only the CPU reads row by row (qwen4_exp n-gram
  // tables) must not lie inside any view. Each remaining segment is covered as above.
  MappedFile(Context& ctx, const Suc1File& f, const std::vector<std::pair<uint64_t, uint64_t>>& exclude,
             uint64_t max_view_bytes = 0);
  const Suc1File& file() const { return *file_; }
  // Buffer and offset holding [file_offset, file_offset + nbytes); throws if no view contains the range.
  BufRef at(uint64_t file_offset, uint64_t nbytes) const;
  BufRef at(const ArrayRecord& r) const { return at(r.offset, r.nbytes); }
  std::size_t views() const { return views_.size(); }
  uint64_t mapped_bytes() const { return mapped_; }        // bytes covered by the views (union)
  uint64_t view_bytes() const;                              // sum of the view lengths (what residency accounts)
  uint64_t excluded_bytes() const { return excluded_; }     // bytes left out of every view
  bool overlaps(uint64_t begin, uint64_t end) const;        // some view holds a byte of [begin, end)

 private:
  struct View {
    BufPtr buf;
    uint64_t start = 0, length = 0;
  };
  const Suc1File* file_;
  void add_views(Context& ctx, uint64_t begin, uint64_t end, uint64_t max_view_bytes);
  std::vector<View> views_;
  std::vector<std::pair<uint64_t, uint64_t>> units_;   // mapping_units(file), computed when a split is needed
  uint64_t mapped_ = 0, excluded_ = 0;
};

// Byte ranges [begin, end) that a loader may address as ONE buffer range, sorted and disjoint: per tensor group
// (the tensors of one transformer layer, "...layers.N.*", or a single tensor outside the layers) and payload section,
// the run of consecutive segments of that group (expert stacks and fused projections address the arrays of several
// tensors of one layer as one range), plus, per group, the span of the codebooks its tensors use; overlapping ranges
// merged. MappedFile cuts views only between units.
std::vector<std::pair<uint64_t, uint64_t>> mapping_units(const Suc1File& f);

// Prefill: Auto, plus the GEMM kernels where the batched (not row-exact) prefill allows them (tcq: suc_tcq_mm over all
// rows beyond one 16-row tile instead of tcq_matmul's tiles).
enum class Path { Auto, Vec, Mma, Prefill };

// Code layout of a trellis-coded tensor (representation "tcq", docs/format.md §5.5; suc.kernels.tcq_matmul): the
// kernels' template constants and their constants input cst = float32 [mu_0, c_0, mu_1, c_1] with mu_j =
// float(code_mu[j]), c_j = float(gain / code_sd[j]) (division in double: tcq_matmul.kernel_consts).
struct TcqLayout {
  int L = 0, V = 0, KV = 0, T = 0, NW = 0, code = 0;   // code: 0 = 1mad, 1 = 3inst, 2 = pair3
  float cst[4] = {0, 0, 0, 0};
  // = suc_switch._layout of a tcq tensor without the rotation sides (k, L, V, T, code, gain, code_mu, code_sd)
  std::string key;
  std::size_t words(std::size_t in) const { return in / std::size_t(T) * std::size_t(NW); }
  // From the (base) params of a tcq tensor of shape (out, in); throws FormatError when the kernels cannot run it.
  static TcqLayout from_params(const Json& p, std::size_t in, const std::string& name);
};
inline constexpr std::size_t kTcqMaxBatchTile = 16;   // tcq_matmul.MAX_BATCH_TILE
inline constexpr std::size_t kTcqMmaMinBatch = 5;     // tcq_matmul.TCQ_MMA_MIN_BATCH
// Largest batch tile whose "vec" launch is row-for-row bit-identical to the one-row launch (no row's arithmetic depends
// on the other rows of the tile; the auto rule keeps tiles below kTcqMmaMinBatch on the vec kernel).
inline constexpr std::size_t kTcqExactRows = 4;
// One launch of the tcq kernels over `tiles` batch tiles of bt <= 16 rows each, kind from
// suc.kernels.tcq_matmul.launch_plan(bt, path): "vec" below 5 rows (SG = 8, R = 4 for bt <= 2 else 2), "mma" from 5
// (SG = 4, RT = 4, NB = ceil(bt / 8)); threadgroups = ceil(out / (rows per simdgroup · SG)).
//   mode 0: dense tensor (tiles = 1): suc_tcq_{vec,mma};
//   mode 1: expert stack, dense (E, out, words) / (E, out) arrays, tile t through expert eidx[t] (suc_tcq_switch_*);
//   mode 2: expert stack through the offset table eoff (E, 3) ulong (suc_tcq_switch_*_tab; ~0 = empty slot).
// x (tiles · bt, in) in xa (rotated basis), y (tiles · bt, out) in ya; scales f16 (file) or float32.
void encode_tcq(Context& ctx, MTL::ComputeCommandEncoder* enc, const TcqLayout& tl, int mode, BufRef x, BufRef words,
                BufRef scales, bool scales_f16, BufRef cst, BufRef eidx, BufRef eoff, BufRef y, std::size_t bt,
                std::size_t tiles, std::size_t out, std::size_t in, Act xa, Act ya, Path path = Path::Auto);
bool tcq_mma(std::size_t bt, Path path);   // launch_plan(bt, path) chooses "mma"
// GEMM variant (suc_tcq_mm, batched prefill, not bit-identical to encode_tcq): mode 0 = one tensor over n rows; mode 1 / 2
// = expert tiles of n <= 64 rows (stacked / offset table, as encode_tcq). x in act, y in ya; in % 32 == 0.
// runs (optional, tile modes): `nruns` (first tile, tiles) uint32 pairs of consecutive tiles of one expert, at most
// run_rows rows each (<= 64): one threadgroup row per run instead of per tile.
void encode_tcq_mm(Context& ctx, MTL::ComputeCommandEncoder* enc, const TcqLayout& tl, int mode, BufRef x,
                   BufRef words, BufRef scales, bool scales_f16, BufRef cst, BufRef teidx, BufRef eoff, BufRef y,
                   std::size_t n, std::size_t tiles, std::size_t out, std::size_t in, Act act, Act ya,
                   BufRef runs = {}, std::size_t nruns = 0, std::size_t run_rows = 0);

// Reusable per-call scratch (x̃, float32 intermediates); grows on demand.
struct Workspace {
  BufPtr xr, y32, y32b;
  std::size_t xr_bytes = 0, y32_bytes = 0, y32b_bytes = 0;
};

// Host arrays of one layer (tests, synthetic benchmarks); uploaded (copied) by Linear::from_host.
struct HostLayer {
  std::size_t out = 0, in = 0, M = 1, K = 0, G = 0;
  unsigned bits = 0;
  std::vector<uint16_t> codebooks;   // (M, K, G) f16 bits
  std::vector<uint32_t> packed;      // (M, out, words) rows_u32le_bitstream
  std::vector<float> row_scales;     // (out,) float32 (as SucLinear holds them)
  std::optional<uint64_t> seed_in, seed_out;
  // residual (original weight space): CSR (row_ptr (out+1), cols, values (int8 * scale or fp16))
  std::vector<uint32_t> res_row_ptr, res_cols;
  std::vector<int8_t> res_val_i8;
  std::vector<uint16_t> res_val_f16;
  float res_scale = 1.0f;
  // column part: cols (c,), values (c, out) column-major, scale (c,) (1 for fp16)
  std::vector<uint32_t> col_cols;
  std::vector<int8_t> col_val_i8;
  std::vector<uint16_t> col_val_f16;
  std::vector<float> col_scale;
};

// Randomized Hadamard rotation of B rows of length n, one launch (= suc.kernels.rotation.rotate, one segment):
// inverse = false: y = H (signs ⊙ x) (V x);  inverse = true: y = signs ⊙ (Hᵀ x) (Uᵀ ỹ). float32 math.
// n = m * 2^k: for 2^k >= 32 the tuned layout ("blocks", or "row" when Context::rotation_row_layout(n)), else a dense
// per-element fallback (hm = H_m, m*m float32).
void encode_rht(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t B, std::size_t n,
                BufRef signs, BufRef hm, bool inverse, Act x_act, Act y_act);
// The "row" layout of suc.kernels.rotation.rotate (rotation._plan / _kernel; VEC = 4, THREADS_MAX = 256, BUF_FLOATS
// = 4096, HM_CONST): 1-3 segments `sizes` of every row of x (B, Σ sizes); one output (outs.size() == 1) or one per
// segment (split); sidx (one segment only): per-row sign table, row b uses signs + sidx[b] · n.
void encode_rot_row(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef signs,
                    const std::vector<BufRef>& outs, const std::vector<std::size_t>& sizes, std::size_t B, bool inverse,
                    Act x_act, Act y_act, BufRef sidx = {});

class Linear {
 public:
  // Whether a tensor can run on this backend (vq / additive / tcq base, residual optional); `why` gets the reason.
  // tcq (trellis-coded, docs/format.md §5.5): the streams are read in place (zero-copy) by the tcq kernels; rotations
  // and residual are the same stages as for vq.
  static bool supported(const TensorDesc& t, std::string* why = nullptr);
  static Linear from_file(Context& ctx, const MappedFile& mf, const TensorDesc& t);
  static Linear from_host(Context& ctx, const HostLayer& h);

  const std::string& name() const { return name_; }
  std::size_t in_features() const { return in_; }
  std::size_t out_features() const { return out_; }
  std::size_t num_codebooks() const { return M_; }
  unsigned bits() const { return bits_; }
  bool tcq() const { return tcq_; }
  const TcqLayout& tcq_layout() const { return tcq_l_; }
  bool rotate_in() const { return seed_in_.has_value(); }
  bool rotate_out() const { return seed_out_.has_value(); }
  bool has_residual() const { return n_csr_ > 0 || n_cols_ > 0; }
  std::size_t residual_entries() const { return n_csr_; }
  std::size_t residual_columns() const { return n_cols_; }
  // Bytes of the representation the kernels read (codebooks + indices + scales + residual arrays).
  std::size_t weight_bytes() const { return weight_bytes_; }
  // Bytes this object copied to the GPU (codebook packing, CSR conversion, signs) vs read zero-copy from the file.
  std::size_t copied_bytes() const { return copied_bytes_; }
  std::size_t zero_copy_bytes() const { return zero_copy_bytes_; }
  bool codebooks_zero_copy() const { return codebooks_zero_copy_; }

  // y (B, out) = layer(x (B, in)), both in `act`. Encodes 1-4 dispatches separated by memory barriers (the
  // encoder may be concurrent). ws must outlive the command buffer.
  void encode(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t B, Act act,
              Workspace& ws, Path path = Path::Auto) const;
  // Kernel only: y (B, out) in y_act = W' x (no rotation, no residual), x in x_act.
  void encode_kernel(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t B, Act x_act,
                     Act y_act, Path path = Path::Auto) const;
  // encode() split into stages, for callers that share one input rotation between projections with the same
  // seed_in (q|k|v, gate|up) and interleave several projections stage by stage (one barrier per stage instead of
  // one per stage and projection). Same dispatches and numerics as encode():
  //   encode_rotate_in: x̃ = V x (only if rotate_in(); any Linear with the same seed_in and in_features may serve);
  //   stage s < stages(): 0 = lookup kernel from x̃ (= x without rotate_in), 1.. = inverse rotation and / or
  //   residual (which reads the unrotated x). The caller puts a memory barrier between consecutive stages.
  void encode_rotate_in(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef xr, std::size_t B,
                        Act act) const;
  std::optional<uint64_t> seed_in() const { return seed_in_; }
  BufRef signs_in() const { return {signs_in_.get(), 0}; }   // float32 (in,), empty without rotate_in
  int stages() const { return 1 + int(seed_out_.has_value()) + int(has_residual()); }
  void encode_stage(Context& ctx, MTL::ComputeCommandEncoder* enc, int stage, BufRef x, BufRef xr, BufRef y,
                    std::size_t B, Act act, Workspace& ws, Path path = Path::Auto) const;
  // Synchronous convenience: x (B·in floats, rounded to act first) -> y (B·out floats, values of act).
  // rows_exact: encode_rows_exact instead of encode (path ignored).
  std::vector<float> forward(Context& ctx, const std::vector<float>& x, std::size_t B, Act act,
                             Path path = Path::Auto, bool rows_exact = false) const;
  // Last path taken per batch tile by encode_kernel ("vec" / "mma"), for reports.
  std::string path_for(const Context& ctx, std::size_t B, Path path = Path::Auto) const;
  // Row-exact batch (batched prefill of graphs that promise decode-step bits): y (B, out) = layer(x (B, in)) with every
  // output row bit-identical to encode() of that row alone (B = 1, the decode step). The rotations and the residual
  // are row-independent launches over the B rows (the residual keeps the one-row reduction split JS); the lookup runs
  // in tiles of at most max_exact_rows() rows on the vec path, whose per-lane group assignment (CHUNK) is then that of
  // the one-row launch. Same workspace use and barriers as encode().
  void encode_rows_exact(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t B, Act act,
                         Workspace& ws) const;
  // Largest lookup tile of encode_rows_exact (1 when decode runs the mma lookup; at most 16 and max_batch_tile).
  std::size_t max_exact_rows(const Context& ctx) const;

 private:
  friend class FusedLinear;
  // exact: the reduction split (JS) of the one-row launch for any B (row-exact batches), else the batch-tuned one.
  void encode_residual(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y32, BufRef y,
                       std::size_t B, Act act, bool exact = false) const;
  void encode_kernel_exact(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t B, Act x_act,
                           Act y_act) const;

  std::string name_;
  std::size_t out_ = 0, in_ = 0, M_ = 1, K_ = 0, G_ = 0, words_ = 0;
  unsigned bits_ = 0;
  BufRef codebooks_, packed_, scales_;
  bool scales_f16_ = true;
  // tcq: packed_ = the streams (out, words_) u32, codebooks_ unused, tcq_cst_ = the kernel constants
  bool tcq_ = false;
  TcqLayout tcq_l_;
  BufPtr tcq_cst_;
  std::optional<uint64_t> seed_in_, seed_out_;
  BufPtr signs_in_, signs_out_, hm_in_, hm_out_;
  // residual
  std::size_t n_csr_ = 0, n_cols_ = 0;
  BufRef row_ptr_, rcols_, rvals_, ccols_, cval_, cscale_;
  bool rcols_u16_ = true, rvals_i8_ = true, ccols_u16_ = true, cval_i8_ = true;
  BufPtr rscale_, meta_;
  std::vector<BufPtr> owned_;
  std::size_t weight_bytes_ = 0, copied_bytes_ = 0, zero_copy_bytes_ = 0;
  bool codebooks_zero_copy_ = false;
};

// Row-concatenation of S = 2..4 Linear layers that read the same input with the same input rotation and code layout
// (q|k|v, gate|up), = suc.kernels.suc_linear.SucFusedLinear and, when a member has a residual,
// suc.runtime.compressed_model.SucFusedResidual (the default Python runtime: convert_to_compressed(group=True,
// fuse=True)):
//   stage 0: ONE lookup launch for all members (suc_fused_lookup_*: row r reads codebook set SUC_SEG(r));
//   stage 1: ONE segmented inverse rotation (each member's own Uᵀ; only with rotate_out);
//   stage 2: ONE structured residual launch over the S members (only with a residual), rounding inside the kernel.
// Index / row-scale layouts:
//   in place (default when every member's indices and f16 scales lie in one mapped view): the C++-only kernel variant
//     suc_fused_lookup_*_tab reads each member's own arrays from the mapping through a per-(segment, codebook stream)
//     offset table (same bodies and numerics, only the row addressing differs); nothing large is copied;
//   copied (fallback, or `in_place = false`): the concatenated (M, Σ out, words) indices and (Σ out,) scales of
//     suc_linear.SucFusedLinear, built at load, for the verbatim Python kernel.
// Always built at load (small): codebooks (S, M, K, G), output signs, concatenated CSR / column residuals.
class FusedLinear {
 public:
  // The fusion key of suc.runtime.compressed_model.group_fusable_key (in, bits, (M, K, G), seed_in, rotate_out and
  // blocks-layout rotation sizes); "" when the layer cannot be fused.
  static std::string key(const Linear& l);
  static FusedLinear build(Context& ctx, const std::vector<const Linear*>& members, bool in_place = true);
  bool in_place() const { return table_.get() != nullptr; }

  std::size_t segments() const { return sizes_.size(); }
  const std::vector<std::size_t>& sizes() const { return sizes_; }
  std::size_t in_features() const { return in_; }
  std::size_t out_features() const { return out_; }
  bool rotate_out() const { return rot_out_; }
  bool has_residual() const { return n_csr_ > 0 || n_cols_ > 0; }
  int stages() const { return 1 + int(rot_out_) + int(has_residual()); }
  // true: the result is written to the fused (B, Σ out) buffer yf (member s at column Σ_{t<s} out_t); false: the
  // segmented output rotation writes member s straight to ys[s] ((B, out_s) contiguous).
  bool fused_output() const { return has_residual() || !rot_out_; }
  std::size_t copied_bytes() const { return copied_bytes_; }
  std::size_t weight_bytes() const { return weight_bytes_; }

  // xr = V x (shared, computed by the caller), x = the unrotated input (read by the residual). The caller puts a
  // memory barrier between consecutive stages. ws.y32 / ws.y32b hold the float32 intermediates (B·Σ out).
  void encode_stage(Context& ctx, MTL::ComputeCommandEncoder* enc, int stage, BufRef x, BufRef xr,
                    const std::vector<BufRef>& ys, BufRef yf, std::size_t B, Act act, Workspace& ws,
                    Path path = Path::Auto) const;
  void encode_kernel(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t B, Act x_act,
                     Act y_act, Path path = Path::Auto) const;

 private:
  std::vector<std::size_t> sizes_;
  std::size_t in_ = 0, out_ = 0, M_ = 1, K_ = 0, G_ = 0, words_ = 0;
  unsigned bits_ = 0;
  bool rot_out_ = false, scales_f16_ = true;
  BufPtr codebooks_, packed_, scales_, signs_out_;
  BufRef packed_ref_, scales_ref_;   // in place: the mapped view holding every member's arrays (offset 0)
  BufPtr table_;                     // in place: (S, M + 1) ulong element offsets (M index streams, scales)
  std::size_t n_csr_ = 0, n_cols_ = 0;
  BufPtr row_ptr_, rcols_, rvals_, ccols_, cval_, cscale_, rscale_, meta_;
  bool rcols_u16_ = true, rvals_i8_ = true, ccols_u16_ = true, cval_i8_ = true;
  std::size_t copied_bytes_ = 0, weight_bytes_ = 0;
};

// Strided row copy y[r, 0:cols] = x[r, 0:cols] (row strides xs / ys elements) in `act` (suc_copy_cols).
void encode_copy_cols(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t rows,
                      std::size_t cols, std::size_t xs, std::size_t ys, Act act);

// ------------------------------------------------------------------------------- batched prefill (not row-exact)
// The frontier graphs' batched prefill (deepseek_v4, glm5_next, qwen4_exp / qwen3_5) runs, by default since 2026-10-10,
// simdgroup-matrix GEMMs over the chunk rows (suc_mm_* kernels, float32 accumulation, one rounding): faster, but not
// bit-identical to decode steps (the difference is measured: scripts/prefill_batch_vs_token.py). The row-exact path
// (every row with its decode step's arithmetic) is kept behind `--prefill exact` / $SUC_PREFILL_EXACT=1.
bool prefill_exact_env();   // $SUC_PREFILL_EXACT=1
// Smallest chunk row count for which the GEMM kernels replace the row gemvs (below: the row-exact gemvs, fast anyway).
inline constexpr std::size_t kMmMinRows = 8;
// y rows r (stride ys) = x rows r (stride xs) · Ŵᵀ for r < n, W affine (packed (out, words) u32, scales / biases
// (out, in / group) in `sdtype` ("half" | "bfloat" | "float")); y in y_msl (default: the activation dtype).
// false (nothing encoded) when the shape is not supported (in % 32 != 0).
bool encode_mm_affine(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef packed, BufRef scales,
                      BufRef biases, BufRef y, std::size_t n, std::size_t xs, std::size_t ys, unsigned bits,
                      std::size_t group, std::size_t in, std::size_t out, std::size_t words, const char* sdtype, Act act,
                      const char* y_msl = nullptr);
// Same for a dense weight (out, in) in `wdtype`.
bool encode_mm_dense(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef w, BufRef y, std::size_t n,
                     std::size_t xs, std::size_t ys, std::size_t in, std::size_t out, const char* wdtype, Act act,
                     const char* y_msl = nullptr);

}  // namespace suc::metal
