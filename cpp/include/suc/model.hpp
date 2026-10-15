// Model graph (milestone 3): a Qwen3 dense decoder executed on the Metal backend straight from a SUC1 file.
//
//   ModelConfig   from directory.architecture (the HF config.json of the checkpoint)
//   LayerNames    per-family tensor-name table (checkpoint names, as suc.models.adapters)
//   Model         weights zero-copy from the mapping (SUC projections = metal::Linear; affine / raw embedding,
//                 LM head and norms read in place), a preallocated activation arena and KV cache, and one
//                 command buffer per decode step.
//
// Operation order of one step = mlx-lm Qwen3 with the projections of suc.runtime.compressed_model (one token,
// mask None, KV cache):
//   h0 = embed(tok)
//   per layer: xn = rms(h, in_norm); q, k, v = proj(xn); q = rope(rms_heads(q, q_norm)); k = rope(rms_heads(k,
//   k_norm)); o = sdpa(q, K[0:pos+1], V[0:pos+1]); h' = h + o_proj(o); xn = rms(h', post_norm);
//   h = h' + down(swiglu(gate(xn), up(xn)))
//   logits = lm_head(rms(h, norm)); next = argmax(logits)
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "suc/metal.hpp"
#include "suc/suc1.hpp"

namespace suc::metal {

class ExpertStream;

struct ModelConfig {
  std::string model_type;
  std::size_t hidden = 0, intermediate = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, vocab = 0;
  double rms_norm_eps = 1e-6, rope_theta = 10000.0;
  bool tie_word_embeddings = false;
  std::string torch_dtype;
  // qwen3_moe (mlx-lm Qwen3MoeSparseMoeBlock): a layer is sparse when it is not in mlp_only_layers, num_experts > 0
  // and (layer + 1) % decoder_sparse_step == 0; its MLP is the router + top-k SwitchGLU (moe_intermediate_size).
  std::size_t num_experts = 0, top_k = 0, moe_intermediate = 0, decoder_sparse_step = 1;
  std::vector<std::size_t> mlp_only_layers;
  bool norm_topk_prob = false;
  bool sparse_layer(std::size_t l) const;
  // Throws FormatError for unsupported families / settings (MoE, rope_scaling, attention bias, sliding window).
  static ModelConfig from_architecture(const Json& arch);
};

struct LayerNames {
  std::string q, k, v, o, gate, up, down, input_norm, post_norm, q_norm, k_norm;
};

// Per-family tensor-name table (checkpoint namespace). Currently: qwen3 (dense).
struct NameTable {
  std::string embed, final_norm, lm_head;
  LayerNames layer(std::size_t i) const;
  static NameTable for_family(const std::string& model_type);
  std::string prefix;
};

struct ModelOptions {
  Act act = Act::BF16;        // activation dtype (the model dtype of the Python runtime)
  std::size_t max_context = 4096;
  std::size_t chunk = 64;     // most tokens per command buffer in forward() (prefill); sizes the activation arena
  // q|k|v and gate|up with one common seed_in share one input rotation and run stage by stage (same numerics);
  // false: every projection runs its own Linear::encode (A/B).
  bool share_rotations = true;
  // with share_rotations: the RMSNorm before q|k|v and gate|up runs fused with their shared input rotation in one
  // launch (suc.kernels.rotation K6 prologue; bit-identical for hidden <= 4096 and a blocks-layout rotation size).
  // -1 = the per-device tuned choice (Context::fuse_elementwise("norm"), as the Python runtime), 0 = off, 1 = on.
  int fuse_norm = -1;
  // with share_rotations: q|k|v and gate|up members with the same code layout run as ONE fused launch per stage
  // (FusedLinear) = the default Python runtime (convert_to_compressed(group=True, fuse=True)); false = one Linear per
  // projection = the Python plan runtime (group=False). Fusion changes the rotation block shapes, so the two plans
  // agree to rounding only.
  bool fuse_projections = true;
  // fused groups read the members' indices / scales in place from the mapping (FusedLinear in-place layout); false:
  // the concatenated copy of the Python layout (A/B; costs a copy of every fused member's indices at load).
  bool fused_in_place = true;
  // LM head and non-SUC projections: the MLX gemv accumulation order (bit-identical logits for one row); false: the
  // milestone-3 matvec (another float32 summation order; A/B).
  bool mlx_gemv = true;
  // decode attention: MLX's two-pass kernels where MLX dispatches them (long contexts; see Model::Impl::sdpa2_rule):
  // -1 = MLX's rule, 0 = always the single-pass kernel (the milestone-3 behaviour).
  int sdpa_2pass = -1;
  // SwiGLU and the down projection's input rotation in one launch (SucMLP): -1 = the tuned choice
  // (Context::fuse_elementwise("swiglu")) with fuse_projections, else off (the Python plan runtime never fuses it).
  int fuse_swiglu = -1;
  // SSD expert streaming (suc/expert_stream.hpp; the MappedFile must leave routed_expert_ranges(file) out): every
  // sparse layer's experts come from this stream's slot pools. Each sparse layer then splits the command buffer after
  // its router (host fetch of the routed experts), forward() runs chunks of at most stream.min_max_distinct() / top_k
  // tokens, and generate() keeps one command buffer in flight. With options().prefetch the next sparse layer's router
  // is applied to the current layer's input and its top-k experts are read ahead. Same numerics as resident.
  ExpertStream* expert_stream = nullptr;
};

// Memory of the runtime (bytes).
struct MemoryReport {
  uint64_t mapped_file = 0;      // the SUC1 mapping (= resident weights when touched)
  uint64_t copied_weights = 0;   // execution-layout arrays built at load (signs, CSR, packed codebooks)
  uint64_t kv_cache = 0, arena = 0, workspace = 0;
};

class Model {
 public:
  Model(Context& ctx, const MappedFile& mf, const ModelOptions& opt = {});
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  const ModelConfig& config() const;
  std::size_t position() const;   // tokens in the KV cache
  void reset();                   // empty the KV cache

  enum class Logits { None, Last, All };
  // Run tokens at the current position (causal attention inside the chunk) and advance it; inputs longer than
  // ModelOptions::chunk are split into several command buffers. One token = one decode step. Returns argmax(logits)
  // (first index of the maximum) of the rows with logits: none, the last token, or every token.
  // A chunk of n > 1 tokens runs the projections at batch n (the "mma" path from mma_min_batch rows), so its numerics
  // differ from n single-token steps by kernel rounding (attention, norms and element-wise ops are row-wise identical).
  std::vector<uint32_t> forward(const std::vector<uint32_t>& tokens, Logits which = Logits::Last);
  uint32_t step(uint32_t token, bool with_logits = true);   // forward({token}, Last / None); 0 without logits
  // Greedy decode of `steps` tokens starting with `first` at the current position, with up to `depth` command
  // buffers in flight: each step's argmax writes the next token into the GPU token buffer, which the next step's
  // embedding gather reads, so the host encodes step t + 1 while the GPU runs step t (no host round trip). Same
  // kernels and numerics as step(); returns the `steps` argmax tokens (or up to and including `stop`). Logits /
  // hidden states of the steps are not kept. on_token (optional) sees each returned token as soon as its step has
  // completed, while the next step(s) already run on the GPU (streaming); returning false stops like `stop` (that
  // token is the last one returned). An exception thrown by on_token is rethrown after the in-flight steps drain.
  struct GenerateStats {
    std::vector<double> step_ms, gpu_ms, encode_ms;   // per step: completion-to-completion wall time, GPU, encode
    double total_s = 0.0;
  };
  std::vector<uint32_t> generate(uint32_t first, std::size_t steps, std::size_t depth = 2,
                                 GenerateStats* stats = nullptr, std::optional<uint32_t> stop = std::nullopt,
                                 const std::function<bool(uint32_t)>& on_token = nullptr);

  // Results of the last command buffer (host copies, converted to float32 from the activation dtype).
  std::vector<float> logits() const;                 // (rows with logits, vocab)
  std::vector<float> hidden_states() const;          // (n, layers + 1, hidden): h0 = embedding, h_l after layer l
  std::size_t last_rows() const;                     // n of the last command buffer
  double last_gpu_seconds() const;                   // GPU time of the last command buffer
  double last_encode_seconds() const;                // host time to encode the last command buffer

  MemoryReport memory() const;
  std::size_t suc_projections() const;               // projections running on SUC kernels
  std::size_t other_projections() const;             // projections running on the affine / dense matvec
  bool norm_fused() const;                           // the resolved ModelOptions::fuse_norm
  bool swiglu_fused() const;                         // the resolved ModelOptions::fuse_swiglu
  std::size_t fused_groups() const;                  // FusedLinear parts (row-concatenated projection groups)
  std::size_t moe_layers() const;                    // sparse (router + expert stacks) layers

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

// Resident memory of this process (task_info): physical footprint and resident size, bytes.
struct ProcessMemory {
  uint64_t phys_footprint = 0, resident = 0, resident_peak = 0;
};
ProcessMemory process_memory();
// Bytes of [addr, addr + len) resident in memory (mincore; for the file mapping this counts the page-cache pages the
// GPU has touched, which the process resident size does not show).
uint64_t resident_bytes(const void* addr, std::size_t len);

}  // namespace suc::metal
