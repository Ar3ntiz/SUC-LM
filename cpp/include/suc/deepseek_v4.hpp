// deepseek_v4 (deepseek-ai/DeepSeek-V4-Flash, docs/arch/deepseek_v4.md) on the Metal backend, straight from a SUC1 file.
//
// Reference semantics: suc/models/mlx_deepseek_v4.py as the Python SUC1 runtime builds it
// (suc.runtime.suc1_model.load_suc1_model, group=False), one token per decode step (Model.step_hidden):
//   x = hc identical streams of embed(token)
//   per layer: y, post, comb = hc_attn.pre(x)  (RMS-scaled mixer, sigmoid gates, Sinkhorn-normalised 4 x 4 mixing)
//              x = hc_attn.post(attention(attn_norm(y)), x, post, comb)
//              y, post, comb = hc_ffn.pre(x);   x = hc_ffn.post(moe(ffn_norm(y), token), x, post, comb)
//   logits = lm_head(norm(hc_head(x)))
//   attention = low-rank q (wq_a, q_norm, wq_b, per-head float32 RMS norm), one K = V head (wkv, kv_norm), RoPE on the
//               last qk_rope_head_dim channels (YaRN + compress_rope_theta on compressed layers), FP8 simulation of the
//               non-RoPE KV channels, sliding window, the layer's compressed entries (ratio 4: overlapping gated
//               pooling + the lightning indexer's top-k selection with its own ratio-4 compressor (Hadamard + FP4
//               simulation); ratio 128: every entry), attention sinks, inverse RoPE on the output, grouped low-rank
//               output (wo_a per row group, wo_b);
//   moe = gate (sqrt(softplus) scores; hash layers: tid2eid[token]; others top-k of scores + bias) -> routed experts
//         (ExpertStack: SUC codes, native MXFP4 or mixed; clamped SwiGLU, routing weight before w2, float32 sum) +
//         shared expert.
// Weights in place from the mapping: SUC projections (metal::Linear), expert stacks (ExpertStack), raw tensors; dense
// projections run MLX's gemv order (suc_moe_gemv), affine embedding / LM head as in qwen4.
//
// Numerics: see cpp/metal/suc_ds4_kernels.metal (MLX's orders measured against MLX 0.32.3); the deviations that remain
// are listed in cpp/README.md (milestone 6).
//
// Not supported (fail loudly): tied embeddings, n_shared_experts != 1, simulate_act_fp8, vision / MTP (ignored),
// batch prefill (the prompt runs as decode steps, several per command buffer).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "suc/expert_mixed.hpp"
#include "suc/metal.hpp"
#include "suc/suc1.hpp"

namespace suc::metal {

struct Ds4Config {
  std::size_t hidden = 0, layers = 0, heads = 0, head_dim = 0, rope_dim = 0, vocab = 0;
  std::size_t q_lora = 0, o_groups = 0, o_lora = 0, window = 0;
  std::vector<std::size_t> ratios;   // per layer (0, 4 or another ratio)
  double rope_theta = 10000.0, compress_rope_theta = 160000.0;
  bool yarn = false;
  double yarn_factor = 1.0, yarn_beta_fast = 32.0, yarn_beta_slow = 1.0;
  std::size_t yarn_orig = 0;
  std::size_t index_heads = 0, index_dim = 0, index_topk = 0;
  std::size_t experts = 0, top_k = 0, moe_inter = 0, hash_layers = 0;
  int scoring = 0;   // 0 sqrtsoftplus, 1 sigmoid, 2 softmax
  double route_scale = 1.0, swiglu_limit = 0.0;
  std::size_t hc = 4, sinkhorn_iters = 20;
  double hc_eps = 1e-6, rms_eps = 1e-6;
  bool sim_kv_fp8 = true, sim_index_fp4 = true;
  uint32_t eos = 1, bos = 0;
  std::string torch_dtype;

  std::size_t stream() const { return hc * hidden; }
  std::size_t token_channels() const { return hash_layers ? 3 : 0; }
  static bool is_ds4(const Json& arch);
  static Ds4Config from_architecture(const Json& arch);
};

class ExpertStream;

struct Ds4Options {
  Act act = Act::BF16;
  std::size_t max_context = 4096;
  std::size_t chunk = 16;   // decode steps encoded per command buffer in forward() (prompt prefill)
  // Batched (layer-major) prefill: forward() runs every chunk of up to `chunk` tokens layer by layer. The projections
  // run once per chunk over all its rows (affine: suc_gemv_affine_rows, every row's arithmetic that of the one-row
  // gemv), the routed experts once per layer per chunk in expert tiles (rows sharing an expert share its weight reads,
  // row-exact kernels), and the order-dependent parts (compressors, KV window, indexer, attention) token by token in
  // position order. One host sync per MoE layer per chunk (the tile plan; streamed: the expert fetch). Same bits as
  // decode steps (tested). chunk = rows per chunk.
  bool layer_prefill = false;
  // Row-exact batched prefill (the description above; `--prefill exact`, $SUC_PREFILL_EXACT=1). Default false: the
  // chunk runs simdgroup-matrix GEMMs (affine / dense / SUC "mma" lookup), 16-row MMA expert tiles, row-batched
  // compressors (one GEMM per compressor and chunk, pooled groups in one launch) and one chunk attention launch per
  // layer (float32 online softmax over each row's window keys and compressed entries); not bit-identical to decode
  // steps (measured: scripts/prefill_batch_vs_token.py). Layers whose indexer would select (entries > index_topk) keep
  // the sequential per-token attention.
  bool exact_prefill = false;
  ExpertStack::Rows mixed_rows = ExpertStack::Rows::Routed;
  // SSD expert streaming (suc/expert_stream.hpp): the routed experts of every layer ("model.layers.N.mlp") are read
  // into the stream's slot pools; per layer the router runs, the command buffer is split (host sync), the routed ids
  // are fetched, then the experts run. Prefetch (stream option): the next layer's gate (router + bias, or tid2eid of
  // the token on hash layers) on this layer's MoE input. Requires mixed_rows == Routed. Must outlive the model.
  ExpertStream* expert_stream = nullptr;
};

struct Ds4Memory {
  uint64_t mapped_file = 0;      // the SUC1 mapping (GPU views)
  uint64_t copied_weights = 0;   // execution-layout arrays built at load (offset tables, signs, residual tables)
  uint64_t kv_cache = 0;         // window keys, compressed entries, indexer entries, compressor group buffers
  uint64_t arena = 0, workspace = 0;
};

struct Ds4StackStats {
  std::size_t stacks = 0, mixed = 0, native_only = 0, native_experts = 0, sub_stacks = 0;
};

// One stage of the SUC_DS4_PROFILE=1 diagnostic profile: accumulated GPU seconds and command buffers.
struct Ds4ProfileRow {
  std::string stage;
  double gpu_s = 0.0;
  std::size_t command_buffers = 0;
};

class Ds4Model {
 public:
  Ds4Model(Context& ctx, const MappedFile& mf, const Ds4Options& opt = {});
  ~Ds4Model();
  Ds4Model(const Ds4Model&) = delete;
  Ds4Model& operator=(const Ds4Model&) = delete;

  const Ds4Config& config() const;
  std::size_t position() const;
  void reset();

  // Runs tokens at the current position (decode steps, up to Ds4Options::chunk per command buffer; layer_prefill: the
  // batched prefill in chunks of `chunk` tokens) and returns argmax(logits) of the last token (0 when with_logits is
  // false). hidden_states() / logits() are those of the last token either way.
  uint32_t forward(const std::vector<uint32_t>& tokens, bool with_logits = true);
  uint32_t step(uint32_t token, bool with_logits = true) { return forward({token}, with_logits); }
  // forward() that also returns the logits of every token (tokens.size() x vocab floats): batched prefill chunks compute
  // the head for every chunk row (measurement of the batched vs token prefill), decode steps (chunk 1) per step.
  std::vector<float> forward_rows(const std::vector<uint32_t>& tokens);

  uint64_t chunk_fetches() const;   // streamed expert fetches of batched-prefill chunks
  std::vector<float> logits() const;          // (vocab) of the last step with logits
  // (layers + 1, hc · hidden + token channels) streams of the last step: the Python stream tensor (with the 3 base-64
  // token-id digit channels behind the streams when the model has hash-routed layers).
  std::vector<float> hidden_states() const;
  double last_gpu_seconds() const;
  double last_encode_seconds() const;
  std::vector<Ds4ProfileRow> profile() const;   // SUC_DS4_PROFILE=1 only (empty otherwise)
  void reset_profile();

  Ds4Memory memory() const;
  std::size_t suc_projections() const;
  Ds4StackStats expert_stacks() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace suc::metal
