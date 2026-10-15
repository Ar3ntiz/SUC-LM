// qwen4_exp (Qwen/Qwen3.8-Flash-Next, docs/arch/qwen4_exp.md) on the Metal backend, straight from a SUC1 file.
//
// Reference semantics: suc/models/mlx_qwen4_exp.py as the Python SUC1 runtime builds it
// (suc.runtime.suc1_model.load_suc1_model), one token per decode step (step_hidden):
//   x = repeat(embed(tok), hc)                                        stream of hc · hidden channels
//   per layer: [PLE layers] x += PLE(x, n-gram rows of the token history)
//              u, c = attn_hyper_connection(x); x += c ⊗ (linear_attn(u) | self_attn(u))   (gated residual)
//              u, c = mlp_hyper_connection(x);  x += c ⊗ moe(u)
//   logits = lm_head(hyper_connection_mixer(x))
//   linear_attn = Gated DeltaNet (conv1d + SiLU, q / k L2 norm, gated delta rule with a float32 recurrent state,
//                 RMSNormGated with a sigmoid gate); self_attn = gated GQA with q / k (1 + w) norms, partial RoPE,
//                 output gate and the QSA block indexer (block keys cached, exact top-k selection from
//                 indexer_budget / indexer_compress_ratio blocks on); moe = softmax router (float32) top-k +
//                 SucSwitchLinear experts (MoeBlock::encode_experts) + sigmoid-gated shared expert.
//
// Weights: SUC projections (metal::Linear, metal::SwitchLinear) and raw / affine tensors are read in place from the
// mapping (the BF16 hyper-connection mixers, routers, in_proj_a / b, indexer projection, norms, conv weights, A_log,
// dt_bias). The n-gram tables (raw BF16 or the row-affine 4-bit records of `suc repack`) are never made resident: per
// token the host hashes the token history (HF Qwen4ExpTextNGramEmbedding: xor of products with the per-layer
// multipliers, mod the per-head primes, EOS-aware context) and reads the ngram_heads rows from the mapping (row-affine
// rows are decoded q · s + b on the CPU), then uploads them (ngram_heads · dim values per token).
//
// Dense qwen3_5 (Qwen/Qwen3.8-27B, suc/models/mlx_qwen3_5.py) runs on the same graph with `dense` set: one stream
// (hc = 1), pre-norm blocks x += mixer((1 + w) RMSNorm(x)); x += SwiGLU MLP((1 + w) RMSNorm(x)); logits =
// lm_head((1 + w) RMSNorm(x)); same Gated DeltaNet / gated attention kernels, no hyper-connections, MoE, PLE or indexer
// (output_gate_type "swish" = silu).
//
// Not supported (fail loudly): tied embeddings, attention bias, layer types other than linear / full attention, PLE on a
// full-attention layer, indexer_kv_heads != 1, MTP (ignored), vision (ignored), batch prefill (the prompt runs as decode
// steps, several per command buffer).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "suc/metal.hpp"
#include "suc/model.hpp"
#include "suc/suc1.hpp"

namespace suc::metal {

struct Qwen4Config {
  std::size_t hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, vocab = 0;
  double rms_norm_eps = 1e-6, rope_theta = 1e7;
  std::size_t rotary_dim = 0;
  std::vector<bool> full_attention;   // per layer
  std::size_t lin_v_heads = 0, lin_k_heads = 0, lin_k_dim = 0, lin_v_dim = 0, conv_kernel = 4;
  std::size_t num_experts = 0, top_k = 0, moe_intermediate = 0, shared_intermediate = 0;
  bool norm_topk_prob = true;
  std::size_t hc = 4, hc_lowrank = 0;
  std::vector<std::size_t> ple_layer_ids;   // one-indexed, sorted
  std::size_t ple_dim = 0, ple_conv_kernel = 4, ngram_size = 3, heads_per_ngram = 8, split_ngram_parts = 0;
  uint64_t seed = 1234, ngram_vocab_size_base = 20000000, ngram_divisible_by = 128;
  bool indexer = false;
  std::size_t idx_heads = 0, idx_dim = 0, idx_budget = 0, idx_ratio = 0;
  bool gate_sigmoid = true;           // RMSNormGated activation (output_gate_type, else hidden_act)
  bool dense = false;                 // qwen3_5: pre-norm blocks with a dense SwiGLU MLP (no hc / MoE / PLE / indexer)
  std::size_t intermediate = 0;       // dense MLP width
  uint32_t eos = 0;
  std::string torch_dtype;

  std::size_t stream() const { return hc * hidden; }
  std::size_t ngram_heads() const { return (ngram_size - 1) * heads_per_ngram; }
  int ple_index(std::size_t layer) const;   // index into ple_layer_ids of 0-indexed `layer`, or -1
  static bool is_qwen4(const Json& arch);   // model_type qwen4_exp or qwen3_5 (dense)
  static Qwen4Config from_architecture(const Json& arch);
};

class ExpertStream;

struct Qwen4Options {
  Act act = Act::BF16;
  std::size_t max_context = 4096;
  std::size_t chunk = 16;   // decode steps encoded per command buffer in forward() (prompt prefill)
  // Batched (layer-major) prefill: forward() runs every chunk of up to `chunk` tokens layer by layer: projections over
  // all chunk rows (affine: suc_gemv_affine_rows, every row's arithmetic that of the one-row gemv), routed experts in
  // expert tiles per layer per chunk (row-exact kernels), the recurrent / cached parts (PLE conv, hyper-connection
  // mixers, Gated DeltaNet, KV cache, indexer, attention) token by token in position order. One host sync per MoE
  // layer per chunk. Same bits as decode steps (tested).
  bool layer_prefill = false;
  // Row-exact batched prefill (the description above; `--prefill exact`, $SUC_PREFILL_EXACT=1). Default false: the
  // chunk projections run simdgroup-matrix GEMMs (affine / dense / SUC "mma" lookup, tcq GEMM), routed experts in
  // 16-row MMA tiles (8 with tcq experts); not bit-identical to decode steps (measured: scripts/prefill_batch_vs_token.py).
  bool exact_prefill = false;
  // SSD expert streaming (suc/expert_stream.hpp; the MappedFile must also leave routed_expert_ranges(file) out): the
  // routed experts come from the stream's slot pools; every layer splits the command buffer after its router (host
  // fetch of the routed experts). With options().prefetch the next layer's router runs on this layer's MoE input
  // and its top-k experts are read ahead. Same numerics as resident.
  ExpertStream* expert_stream = nullptr;
};

struct Qwen4Memory {
  uint64_t mapped_file = 0;      // the SUC1 mapping
  uint64_t ngram_tables = 0;     // bytes of the n-gram tables in the file (read row by row, never resident)
  uint64_t copied_weights = 0;   // execution-layout arrays built at load
  uint64_t kv_cache = 0;         // K / V caches + indexer keys and block keys
  uint64_t states = 0;           // conv / recurrent / PLE conv states
  uint64_t arena = 0, workspace = 0;
};

// File byte ranges of the n-gram tables (every array of the `.ngram_embedding.shard_` tensors). The MappedFile of a
// Qwen4Model must leave them out of its GPU views (MappedFile(ctx, f, qwen4_table_ranges(f))): Metal makes every page of a
// no-copy buffer resident when a command buffer uses it, and the tables (102 GB raw / 27 GB row-affine on the real model)
// are only read on the CPU, ngram_heads rows per token.
std::vector<std::pair<uint64_t, uint64_t>> qwen4_table_ranges(const Suc1File& f);

class Qwen4Model {
 public:
  // Throws when a view of `mf` overlaps an n-gram table (see qwen4_table_ranges).
  Qwen4Model(Context& ctx, const MappedFile& mf, const Qwen4Options& opt = {});
  ~Qwen4Model();
  Qwen4Model(const Qwen4Model&) = delete;
  Qwen4Model& operator=(const Qwen4Model&) = delete;

  const Qwen4Config& config() const;
  std::size_t position() const;
  void reset();   // empty the caches and states, n-gram context back to EOS

  // Runs tokens as decode steps at the current position (up to Qwen4Options::chunk steps per command buffer) and
  // returns argmax(logits) of the last token (0 when with_logits is false).
  uint32_t forward(const std::vector<uint32_t>& tokens, bool with_logits = true);
  uint32_t step(uint32_t token, bool with_logits = true) { return forward({token}, with_logits); }
  // forward() that also returns the logits of every token (tokens.size() x vocab floats): batched prefill chunks compute
  // the LM head for every chunk row (measurement of the batched vs token prefill), decode steps (chunk 1) per step.
  std::vector<float> forward_rows(const std::vector<uint32_t>& tokens);

  std::vector<float> logits() const;          // (vocab) of the last step with logits
  std::vector<float> hidden_states() const;   // (layers + 1, hc · hidden) streams of the last step
  uint64_t chunk_fetches() const;              // streamed expert fetches of batched-prefill chunks
  double last_gpu_seconds() const;
  double last_encode_seconds() const;
  double last_ngram_seconds() const;          // host time of the n-gram hashing + row gathers of the last call

  Qwen4Memory memory() const;
  std::size_t suc_projections() const;        // SUC Linear projections (expert stacks counted once per stack)
  std::size_t expert_stacks() const;
  bool ngram_quantized() const;               // row-affine (suc repack) n-gram tables

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace suc::metal
