// glm5_next (zai-org/GLM-5.3-Flash, docs/arch/glm5_next.md) on the Metal backend, straight from a SUC1 file.
//
// Reference semantics: suc/models/mlx_glm5_next.py as the Python SUC1 runtime builds it
// (suc.runtime.suc1_model.load_suc1_model, group=False), one token per decode step (Model.step_hidden):
//   x = hc identical streams of embed(token)
//   per layer: post, comb, y = mHC_attn(x)  (unweighted RMS norm of the 4 · hidden stream, then the 24 mixer logits,
//                                           sigmoid gates, softmax + Sinkhorn 4 x 4 mixing, stream collapse)
//              x = post ⊗ attn(input_layernorm(y)) + combᵀ x       (in the model dtype, TF:1337)
//              post, comb, y = mHC_ffn(x);  x = post ⊗ mlp(post_attention_layernorm(y)) + combᵀ x
//   logits = lm_head(norm(mean of the streams))
//   attn = KDA (34 of 45 layers): q / k / v projections, SiLU(causal depthwise conv4) in float32, q / k L2-normalized,
//          the safe-gate per-channel decay exp(-5 σ(exp(A_log) (f_b f_a x + dt_bias))), beta = σ(b x), the gated delta
//          rule (float32 state), o_norm (float32 RMSNorm) · σ(g_b g_a x), o_proj;
//        or MLA + DSA (11 layers): NoPE MLA with kv_b absorbed per head (q'_h = W_k,hᵀ q_h against the cached latents,
//          o_h = W_v,h (P c)), restricted to the DSA indexer's token selection: per token the indexer key
//          LayerNorm(wk x) and gate score (kpool_gate x); every index_kpool tokens a pooled key (softmax over the pool
//          of gate + ape, per channel); from index_topk / index_kpool complete pools on, the top pools by
//          Σ_h w_h relu(q_h · K_p / √d) plus the incomplete tail;
//   mlp = dense SwiGLU (layers < first_k_dense_replace) or MoE: sigmoid router (float32) + correction bias top-k,
//         normalized, · routed_scaling_factor; routed experts (ExpertStack: SUC codes, mixed codecs allowed; clamped
//         SwiGLU, routing weight after down_proj, float32 sum) + shared expert.
// Weights in place from the mapping: SUC projections (metal::Linear), expert stacks (ExpertStack), raw tensors; kept
// dense projections (KDA gate factors, indexer, kv_b, router, mHC mixers) run MLX's gemv orders; affine embedding /
// LM head as in qwen4 / deepseek_v4.
//
// SSD expert streaming (Glm5Options::expert_stream, suc/expert_stream.hpp): the routed experts come from the stream's
// slot pools; every MoE layer splits the command buffer after its router (host fetch of the routed experts); with
// options().prefetch the next MoE layer's router runs on this layer's MoE input and its top-k are read ahead. Same
// numerics as resident.
//
// Not supported (fail loudly): RoPE (qk_rope_head_dim > 0), shared (cross-layer) indexer top-k, routers other than
// sigmoid with one group, n_shared_experts != 1, attention bias, tied embeddings, vision / MTP (ignored).
//
// Prompt prefill (Glm5Options): decode steps (several per command buffer), the row-exact layer-major chunks of the
// streamed experts (decode-step bits), or, with layer_prefill (the default of the tools since 2026-10-10), the batched
// (not row-exact) layer-major chunks: every projection over the chunk rows as a simdgroup-matrix GEMM
// (Linear::encode(Path::Prefill), encode_mm_affine / encode_mm_dense; the router as a float32 GEMM), the routed experts
// in MMA tiles (Tiles::exact = false), the mHC / KDA element-wise ops, causal conv and gated-delta recurrence as
// row-batched launches with the decode kernels' per-row arithmetic (the recurrence loops over the chunk tokens inside
// one launch per layer), MLA as a chunk attention kernel over the cached latents with the rows' DSA pool selection
// (indexer scores of all rows in one launch, top pools per row). Not bit-identical to decode steps (GEMM vs gemv accumulation orders; measured by
// scripts/prefill_batch_vs_token.py); exact_prefill / $SUC_PREFILL_EXACT=1 keep the previous paths.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "suc/expert_mixed.hpp"
#include "suc/metal.hpp"
#include "suc/suc1.hpp"

namespace suc::metal {

class ExpertStream;

struct Glm5Config {
  std::size_t hidden = 0, layers = 0, vocab = 0, intermediate = 0;
  std::vector<bool> dsa;      // per layer: MLA + DSA (else KDA)
  std::vector<bool> sparse;   // per layer: MoE (else dense MLP)
  // KDA
  std::size_t lin_heads = 0, lin_dim = 0, conv_kernel = 4;
  double lower_bound = -5.0;
  // MLA
  std::size_t heads = 0, q_lora = 0, kv_lora = 0, nope = 0, v_dim = 0;
  // DSA indexer
  std::size_t index_heads = 0, index_dim = 0, index_topk = 0, kpool = 4;
  bool select_tail = true;
  // MoE
  std::size_t experts = 0, top_k = 0, moe_inter = 0;
  bool norm_topk_prob = true;
  double route_scale = 1.0, swiglu_limit = 0.0;
  // mHC
  std::size_t hc = 4, sinkhorn_iters = 20;
  double hc_eps = 1e-6, rms_eps = 1e-5;
  std::vector<uint32_t> eos;
  std::string torch_dtype;

  std::size_t stream() const { return hc * hidden; }
  std::size_t topk_pools() const { return index_topk / kpool; }
  static bool is_glm5(const Json& arch);
  static Glm5Config from_architecture(const Json& arch);
};

struct Glm5Options {
  Act act = Act::BF16;
  std::size_t max_context = 4096;
  // tokens per command buffer in forward() (prompt prefill). Resident experts: decode steps. Streamed experts and
  // chunk > 1: layer-major chunks (each layer runs all chunk tokens up to the router, then one fetch of the union of
  // their routed experts), same bits as decode steps.
  std::size_t chunk = 16;
  // batched (not row-exact) layer-major chunks of `chunk` tokens, resident or streamed (see the head of the file);
  // ignored with exact_prefill or $SUC_PREFILL_EXACT=1 (then the paths above)
  bool layer_prefill = false;
  bool exact_prefill = false;
  ExpertStack::Rows mixed_rows = ExpertStack::Rows::Routed;
  ExpertStream* expert_stream = nullptr;   // the MappedFile must then leave routed_expert_ranges(file) out
};

struct Glm5Memory {
  uint64_t mapped_file = 0;      // the SUC1 mapping (GPU views)
  uint64_t copied_weights = 0;   // execution-layout arrays built at load (conv weights, offset tables, signs, residuals)
  uint64_t kv_cache = 0;         // MLA latents, indexer pool rows and pooled keys
  uint64_t states = 0;           // KDA conv tails and float32 recurrent states
  uint64_t arena = 0, workspace = 0;
};

struct Glm5StackStats {
  std::size_t stacks = 0, mixed = 0, sub_stacks = 0;
};

class Glm5Model {
 public:
  Glm5Model(Context& ctx, const MappedFile& mf, const Glm5Options& opt = {});
  ~Glm5Model();
  Glm5Model(const Glm5Model&) = delete;
  Glm5Model& operator=(const Glm5Model&) = delete;

  const Glm5Config& config() const;
  std::size_t position() const;
  void reset();
  uint64_t chunk_fetches() const;   // expert fetches of layer-major prefill chunks (streamed experts), cumulative
  // $SUC_GLM_PROFILE=1: GPU seconds per prefill stage of the batched chunks (each stage its own command buffer)
  std::vector<std::pair<std::string, double>> profile() const;
  void reset_profile();

  // Runs tokens as decode steps at the current position (up to Glm5Options::chunk steps per command buffer) and
  // returns argmax(logits) of the last token (0 when with_logits is false).
  uint32_t forward(const std::vector<uint32_t>& tokens, bool with_logits = true);
  uint32_t step(uint32_t token, bool with_logits = true) { return forward({token}, with_logits); }
  // forward(tokens, true) that also returns the logits of every token ((tokens, vocab) float32): the batched chunks run
  // the head on every chunk row, decode steps record theirs (measurement runs: scripts/prefill_batch_vs_token.py).
  std::vector<float> forward_rows(const std::vector<uint32_t>& tokens);
  bool batched_prefill() const;   // forward() runs the batched (not row-exact) layer-major chunks

  std::vector<float> logits() const;          // (vocab) of the last step with logits
  std::vector<float> hidden_states() const;   // (layers + 1, hc · hidden) streams of the last step
  double last_gpu_seconds() const;
  double last_encode_seconds() const;

  Glm5Memory memory() const;
  std::size_t suc_projections() const;
  Glm5StackStats expert_stacks() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace suc::metal
