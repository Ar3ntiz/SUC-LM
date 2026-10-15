# deepseek_v4 — deepseek-ai/DeepSeek-V4-Flash-0731

Architecture and checkpoint analysis for running SUC on DeepSeek-V4-Flash-0731 (304 B total parameters, about
13 B active). It is written for the post-project scaling work (CLAUDE.md "Current state"). Nothing here has been
run on the model. All numbers come from `config.json`, the safetensors index and five fetched shard headers.

## 1. Identity

| Item | Value |
|---|---|
| Repo | `deepseek-ai/DeepSeek-V4-Flash-0731` (MIT licence) |
| Pinned revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| `model_type` / `architectures` | `deepseek_v4` / `["DeepseekV4ForCausalLM"]` |
| `transformers_version` (config.json) | `4.57.1` (generation_config: `4.46.3`) |
| Index `total_size` | 166,878,536,440 B (166.88 GB, 155.4 GiB) |
| Shards / tensors | 48 shards, 72,317 tensors |
| README claim | The model "has the same model structure as DeepSeek-V4-Flash-DSpark": the checkpoint includes a speculative-decoding module (DSpark) under `mtp.*`. |

Sources consulted:
- `config.json`, `generation_config.json`, `tokenizer_config.json`, `model.safetensors.index.json` and `README.md` at the
  pinned revision, plus the repo's reference implementation `inference/model.py`, `inference/kernel.py`,
  `inference/convert.py` and `inference/config.json`. In this document, "model.py:N" and the like refer to those files.
- huggingface/transformers main: `modeling_deepseek_v4.py`, `modular_deepseek_v4.py`, `configuration_deepseek_v4.py`
  (cited as "tf-modeling:N" / "tf-config:N").
- ml-explore/mlx-lm main: `mlx_lm/models/deepseek_v41.py` (cited as "v41:N"). Neither mlx-lm main nor the installed
  mlx-lm 0.31.3 has a `deepseek_v4` module.
- Safetensors headers were fetched with HTTP Range requests (no weight bytes) for shards **00001** (embed), **00004**
  (layer 2, a CSA layer), **00005** (layer 3, an HCA layer), **00045** (head, final norm, hc_head) and **00048**
  (`mtp.2`, including the markov and confidence heads).
  - Measured from these headers: every pattern except `mtp.0.main_proj.{weight,scale}` and `mtp.0.main_norm.weight`.
  - Derived from model.py:832-833: those three. Layers other than 2 and 3, and mtp stages 0 and 1, are assumed to
    have the same shapes as the measured layer with the same compress ratio.
  - Cross-check: summing count × shape × dtype size over all 72,317 index entries reproduces `total_size`
    **to the byte** (difference 0). So the shapes and dtypes below hold for the whole checkpoint.

Config discrepancy: `config.json` has `num_nextn_predict_layers: 1`, but the checkpoint holds `mtp.0`, `mtp.1` and
`mtp.2`. `inference/config.json` has `n_mtp_layers: 3`, and `compress_ratios` has 46 entries (43 layers plus 3 MTP
stages, all of them 0).

## 2. Architecture summary

| Key | Value | Key | Value |
|---|---|---|---|
| hidden_size | 4096 | num_hidden_layers | 43 (+3 DSpark/MTP stages) |
| vocab_size | 129280 (untied head) | num_attention_heads | 64, head_dim 512, qk_rope_head_dim 64 |
| num_key_value_heads | 1 (single shared K=V vector) | q_lora_rank | 1024 |
| o_groups / o_lora_rank | 8 / 1024 | sliding_window | 128 (every layer) |
| compress_ratios | `[0,0,4,128,4,128,…,4]` | index_n_heads / index_head_dim / index_topk | 64 / 128 / 512 |
| n_routed_experts / top-k | 256 / 6 | n_shared_experts | 1 |
| moe_intermediate_size | 2048 (routed and shared) | num_hash_layers | 3 |
| scoring_func / topk_method | `sqrtsoftplus` / `noaux_tc` | routed_scaling_factor | 1.5, norm_topk_prob true |
| swiglu_limit | 10.0 | hc_mult / sinkhorn iters / hc_eps | 4 / 20 / 1e-6 |
| rope_theta / compress_rope_theta | 10000 / 160000 | rope_scaling | YaRN factor 16, orig 65536, β_fast 32, β_slow 1 |
| rms_norm_eps | 1e-6 | max_position_embeddings | 1,048,576 |
| dspark | block 5, noise token 128799, target layers [40,41,42], markov rank 256 | quantization_config | fp8 e4m3, ue8m0 scales, block [128,128]; `expert_dtype: fp4` |

**Per-layer schedule.** All 43 layers are MoE; there are no dense FFN layers.

| Layers | Attention | MoE routing |
|---|---|---|
| 0, 1 | sliding window only (ratio 0), RoPE θ=10000, no YaRN (model.py:483-485) | hash (`tid2eid`) |
| 2 | CSA: ratio 4 with overlap + indexer top-512 | hash |
| 4, 6, …, 42 (20 layers) | CSA: ratio 4 with overlap + indexer | score (`sqrtsoftplus` + bias) |
| 3, 5, …, 41 (20 layers) | HCA: ratio 128, no overlap, no indexer, attends every causal compressed entry (model.py:519, 275-282) | score |
| mtp.0-2 | sliding window only (DSparkAttention asserts ratio 0, model.py:753) | score (each stage has its own 256 experts) |

In total, 21 layers are CSA (they have `indexer.*`) and 20 are HCA, which gives 41 `attn.compressor.*`. This matches
the index counts. Transformers names the three types `sliding_attention`, `compressed_sparse_attention` and
`heavily_compressed_attention` (tf-config:28-32).

**Attention** (model.py:490-548). It is MQA with a low-rank query:
- `qr = RMSNorm_q(W_qa x)` (4096→1024).
- `q = W_qb qr` is reshaped to [64, 512], then each head gets an **unweighted** RMS normalisation,
  `q_h ← q_h · rsqrt(mean(q_h²)+eps)` (model.py:504; tf `q_b_norm`, tf-modeling:789).
- RoPE is applied to the **last** 64 dims (interleaved pairs, model.py:241, 505).
- `kv = RMSNorm_kv(W_kv x)` (4096→512) is used as both key and value. RoPE goes on its last 64 dims, and the first
  448 dims are FP8 round-tripped with block 64 (model.py:508-512, a QAT simulation).
- Key set per query: the last 128 window tokens, plus compressed entries (CSA: indexer top-512; HCA: all causal
  entries).
- `o_h = Σ_j p_j kv_j` with `p = exp(s_j − m) / (Σ exp(s − m) + exp(sink_h − m))` and `s = q_h·kv / √512`. The sink
  is a learnable per-head logit (kernel.py:321-348). No YaRN mscale is applied (the scale is `head_dim**-0.5`,
  model.py:470).
- Inverse RoPE on the last 64 dims of `o` (model.py:539).
- **Grouped low-rank output**: `o` [64×512] is split into 8 groups of 4096 (8 heads each); `u_g = W_oa[g] o_g`
  (4096→1024 per group); `y = W_ob · concat(u_g)` (8192→4096) (model.py:542-547). `wo_a` is stored as one
  [8192, 4096] tensor, which is block-diagonal by output-row groups.
- Compressed layers use YaRN RoPE with θ=160000 for both window and compressed keys (model.py:481-488).

**Compressor** (model.py:285-383). Ratio `r`, `coff = 2` if `r == 4` (overlap) else 1.
- `kv = W_ckv x` and `g = W_cg x + ape[t mod r]` (out `coff·d`), computed in **fp32** (model.py:328-330).
- Each entry is `c = Σ_t softmax_t(g_t) ⊙ kv_t`, a per-channel softmax over the tokens of the group.
- With overlap, entry *i* pools the first-half channels of the previous 4 tokens and the second-half channels of
  the current 4 tokens (overlap_transform, model.py:313-320). The receptive field is therefore 8 tokens.
- After pooling: RMSNorm, then RoPE at position `i·r` (model.py:368-373), then FP8 round-trip (main) or Hadamard
  rotation + FP4 round-trip (indexer) (model.py:374-378).

**Indexer** (CSA layers, model.py:386-439).
- `q_I = W_Iqb qr` → [64, 128]; RoPE; Hadamard rotation; FP4 simulation.
- Keys come from the indexer's **own** compressor (ratio 4, head 128, `rotate=True`).
- `w = W_w x · 128^-½ · 64^-½`; `score_t = Σ_h w_h · relu(q_I,h · k_t)`; then causal top-512.
- The Hadamard rotation is applied to both `q` and `k`, so it cancels in the dot product. It only matters for the
  FP4 simulation.

**MoE** (model.py:551-649).
- `scores = sqrt(softplus(W_g x))` in fp32.
- Score layers: top-6 of `scores + bias`, where the bias affects selection only (noaux_tc).
- Hash layers 0-2: the experts are `tid2eid[input_ids]` (a fixed lookup, no bias).
- In both cases the weights are the unbiased scores of the chosen experts, normalised to sum 1, × 1.5
  (model.py:585-588).
- There is one always-on shared expert. Experts are SwiGLU with `up = clamp(up, ±10)` and `gate = min(gate, 10)`,
  computed in fp32 (model.py:601-611). The same clamp applies to the shared expert (model.py:632).
- **Hash routing makes the FFN depend on token ids, not only on hidden states.**

**Residual: manifold hyper-connections (mHC)** (model.py:652-716, kernel.py:372-427). The residual is 4 streams
`X ∈ R^{4×4096}`, initialised to 4 copies of the embedding (model.py:916). For each sublayer:
- `mix = (flatten(X) / rms) · fnᵀ` (24 values).
- `pre = σ(mix₀:₄·s₀ + b) + eps` and `post = 2σ(mix₄:₈·s₁ + b)`.
- `comb = Sinkhorn(softmax(mix₈:₂₄·s₂ + b as 4×4) + eps)`: column-normalise, then 19 × (row, column).
- `h = Σ_i pre_i X_i`, then `F(norm(h))`, then `X'_j = post_j · F + Σ_i comb_ij X_i` (model.py:692).
- The final collapse is a separate **hc_head**, `pre = σ(mix·s + b) + eps` with its own `hc_head_fn [4,16384]`,
  followed by `norm` and `head` (model.py:709-716, 922-923).

**DSpark speculative head** (`mtp.*`, model.py:743-936). Three full Blocks (window attention, 256 experts, HC).
- Stage 0 has `main_proj` (12288→4096) over the concatenated stream means of target layers 40-42, plus `main_norm`.
- The draft block is 5 tokens: the last token followed by 4 noise tokens (model.py:851-858). Its attention sees the
  main-model window KV recomputed through its own `wkv`.
- The last stage holds `hc_head`, `norm`, a **markov head** (bigram low-rank logit bias: `markov_w1` embedding
  [V,256] and `markov_w2` head [V,256], applied autoregressively across the block) and a **confidence head**
  (`Linear(4096+256 → 1)`).
- It reuses the main `embed` and `head` (model.py:903-904).
- Not needed for language-model quality; SUC drops it (§6).

There are no multimodal parts, no n-gram/engram tables and no linear attention.

## 3. Checkpoint tensor inventory

L = 43 layers, CSA = 21, HCA = 20, E = 256. "M" = shape measured in a header; "D" = derived. The scale companions
`*.scale` are listed with their weight.

| pattern | count | dtype | shape | role | SUC treatment |
|---|---|---|---|---|---|
| `embed.weight` | 1 | BF16 | [129280,4096] M | embed | affine 4-bit g64 |
| `head.weight` | 1 | BF16 | [129280,4096] M | lm_head (untied) | affine 4-bit g64 |
| `norm.weight` | 1 | BF16 | [4096] M | final norm | keep |
| `hc_head_{fn,base,scale}` | 1 each | F32 | [4,16384] / [4] / [1] M | final HC collapse | keep F32 |
| `layers.#.hc_{attn,ffn}_fn` | 43 each | F32 | [24,16384] M | HC mixer | keep F32 |
| `layers.#.hc_{attn,ffn}_{base,scale}` | 43 each | F32 | [24] / [3] M | HC mixer | keep F32 |
| `layers.#.attn_norm.weight`, `ffn_norm.weight` | 43 each | BF16 | [4096] M | norms | keep (fold candidates) |
| `layers.#.attn.wq_a.weight` + `.scale` | 43 | F8_E4M3 + F8_E8M0 | [1024,4096] + [8,32] M | attn_q_a | SUC |
| `layers.#.attn.q_norm.weight` | 43 | BF16 | [1024] M | q low-rank norm | keep |
| `layers.#.attn.wq_b.weight` + `.scale` | 43 | F8_E4M3 + E8M0 | [32768,1024] + [256,8] M | attn_q_b | SUC |
| `layers.#.attn.wkv.weight` + `.scale` | 43 | F8_E4M3 + E8M0 | [512,4096] + [4,32] M | attn_kv (K=V) | SUC |
| `layers.#.attn.kv_norm.weight` | 43 | BF16 | [512] M | kv norm | keep |
| `layers.#.attn.wo_a.weight` + `.scale` | 43 | F8_E4M3 + E8M0 | [8192,4096] + [64,32] M | attn_o_a (8 groups) | SUC (row-group aware) |
| `layers.#.attn.wo_b.weight` + `.scale` | 43 | F8_E4M3 + E8M0 | [4096,8192] + [32,64] M | attn_o_b | SUC |
| `layers.#.attn.attn_sink` | 43 | F32 | [64] M | sink logits | keep |
| `layers.#.attn.compressor.wkv/wgate.weight` (r=4) | 21 each | BF16 | [1024,4096] M | compressor_kv/gate | keep BF16 (ablation: SUC) |
| `layers.#.attn.compressor.wkv/wgate.weight` (r=128) | 20 each | BF16 | [512,4096] M | compressor_kv/gate | keep BF16 |
| `layers.#.attn.compressor.ape` | 21 / 20 | F32 | [4,1024] / [128,512] M | positional bias | keep |
| `layers.#.attn.compressor.norm.weight` | 41 | BF16 | [512] M | norm | keep |
| `layers.#.attn.indexer.wq_b.weight` + `.scale` | 21 | F8_E4M3 + E8M0 | [8192,1024] + [64,8] M | indexer_q_b | SUC |
| `layers.#.attn.indexer.weights_proj.weight` | 21 | BF16 | [64,4096] M | indexer head weights | keep BF16 |
| `layers.#.attn.indexer.compressor.wkv/wgate.weight` | 21 each | BF16 | [256,4096] M | indexer compressor | keep BF16 |
| `layers.#.attn.indexer.compressor.ape` / `.norm.weight` | 21 each | F32 / BF16 | [4,256] / [128] M | indexer compressor | keep |
| `layers.#.ffn.gate.weight` | 43 | BF16 | [256,4096] M | router | keep BF16 |
| `layers.#.ffn.gate.bias` | 40 (layers 3-42) | F32 | [256] M | noaux_tc bias | keep |
| `layers.#.ffn.gate.tid2eid` | 3 (layers 0-2) | I64 | [129280,6] M | hash routing table | keep (lossless → uint8) |
| `layers.#.ffn.experts.#.w1/w3.weight` | 11008 each | I8 (packed FP4) | [2048,2048] M (logical [2048,4096]) | expert_gate / expert_up | native MXFP4 or SUC transcode (§6) |
| `layers.#.ffn.experts.#.w1/w3.scale` | 11008 each | F8_E8M0 | [2048,128] M | MXFP4 scales | consumed by dequant |
| `layers.#.ffn.experts.#.w2.weight` + `.scale` | 11008 | I8 + E8M0 | [4096,1024] + [4096,64] M (logical [4096,2048]) | expert_down | as above |
| `layers.#.ffn.shared_experts.w1/w3.weight` + `.scale` | 43 each | F8_E4M3 + E8M0 | [2048,4096] + [16,32] M | shared_expert_gate/up | SUC |
| `layers.#.ffn.shared_experts.w2.weight` + `.scale` | 43 | F8_E4M3 + E8M0 | [4096,2048] + [32,16] M | shared_expert_down | SUC |
| `mtp.#.*` (attn/ffn/hc/norms, same shapes as a window layer) | 3 stages | as main | M (mtp.2) | DSpark block | drop |
| `mtp.#.ffn.experts.#.w{1,2,3}.{weight,scale}` | 768 × 6 | I8 / E8M0 | as main M | DSpark experts | drop |
| `mtp.0.main_proj.weight` + `.scale`, `mtp.0.main_norm.weight` | 1 each | F8_E4M3 + E8M0 / BF16 | [4096,12288] + [32,96] / [4096] D | DSpark input | drop |
| `mtp.2.markov_head.markov_w{1,2}.weight` | 1 each | BF16 | [129280,256] M | markov bigram head | drop |
| `mtp.2.confidence_head.proj.weight` | 1 | BF16 | [1,4352] M | confidence head | drop |
| `mtp.2.norm.weight`, `mtp.2.hc_head_{fn,base,scale}` | 1 each | BF16 / F32 | [4096] / [4,16384] / [4] / [1] M | DSpark head | drop |

## 4. Stored dtypes and quantization

The README badge ("INT8 + FP8") is misleading. The header dtype `I8` is **packed FP4 (E2M1, two per byte)**, not
int8 values (model.py:137-143, convert.py:21-33). The checkpoint has three numeric families:

1. **FP8 E4M3 with 128×128 block scales in UE8M0**. Header dtypes `F8_E4M3` and `F8_E8M0` are used for all
   attention linears, the indexer `wq_b`, the shared experts and `mtp.0.main_proj`.
   - Scale shape is `[ceil(out/128), ceil(in/128)]` (model.py:144-148), e.g. `wq_b` [32768,1024] → [256,8].
   - UE8M0 means 8 exponent bits, no sign or mantissa, bias 127, and 0xFF is NaN.
   - Dequantisation:
     `W[i,j] = e4m3(W8[i,j]) · 2^(S[i//128, j//128] − 127)`.
     convert.py:126 does exactly this for `wo_a`: `weight.unflatten(0,(-1,128)).unflatten(-1,(-1,128)).float() * scale[:,None,:,None].float()`, converting it to BF16 because model.py:468 builds `wo_a` in BF16.
   - In the reference GEMM, the per-block scale multiplies the partial FP32 accumulator (kernel.py:241-249).
2. **MXFP4 for routed experts** (`expert_dtype: fp4`). The weight is I8 `[out, in/2]` and the scale is F8_E8M0
   `[out, in/32]`: one power-of-two scale per 32 consecutive input elements of a row (model.py:139-143).
   - Packing: byte `b` at column `c` holds element `2c` in the **low nibble** and `2c+1` in the high nibble
     (convert.py:30-33: `low = x & 0x0F; high = (x >> 4) & 0x0F; stack([low, high], -1)`).
   - A nibble is sign bit 3 plus magnitude `[0, .5, 1, 1.5, 2, 3, 4, 6]` (FP4_TABLE, convert.py:11-14).
   - Dequantisation: `W[i,k] = FP4_TABLE[nib(i,k)] · 2^(S[i, k//32] − 127)`.
   - Effective storage: 4 + 8/32 = **4.25 bpw**.
   - convert.py:17-52 can also re-express the FP4 experts losslessly as FP8 with 128×128 scales (`--expert-dtype fp8`).
3. **BF16 / F32** for the rest. Compressors and `weights_proj` are BF16 but computed in fp32 (model.py:301-304,
   400); HC parameters, sinks, `ape` and the gate bias are F32; `tid2eid` is I64 (cast to int32 in model.py:564).

**Exactness, verified locally with MLX 0.32.3.**
- All 254 finite E4M3 values are exactly representable in BF16, and E2M1 values are too. Multiplying by a power of
  two is exact while the result stays in the BF16 normal range.
- So per-tensor dequantisation of the checkpoint to BF16 is **lossless**. The only possible exception is extreme
  scale exponents, which is checked by asserting `min(S) ≥ 2` and `max(S) ≤ 252` on load. Scale ranges have not
  been inspected yet.
- `mx.dequantize(mode="mxfp4", group_size=32)` on bytes packed as above (viewed as uint32) reproduces the reference
  formula with max error 0. That is, the MLX MXFP4 layout is byte-compatible with the checkpoint.

**MLX sanitize in v41 cannot be reused as is for FP8.**
- `_repack` (v41:1049-1056) converts the E8M0 scales to a uint8 exponent and views the weight as uint32, and keeps
  everything packed (MLX `mxfp8` / `mxfp4` group 32).
- For FP8 it assumes **32×32** scale tiles: it repeats the scale 32× along the rows only. That matches V4.1, whose
  `wq_a` [1280,5120] has scale [40,160] (header of V4.1 shard 17). It does not match V4-Flash's 128×128 tiles.
- V4-Flash needs `repeat(S, 128, axis=-2)` and `repeat(S, 4, axis=-1)` to reach MLX's one-scale-per-32-inputs-per-row
  layout. This is lossless: same block value. Expert MXFP4 repacking works unchanged.

**D8 conflict.**
- D8 requires a full-precision BF16/FP16 source. DeepSeek publishes only this FP8/FP4 checkpoint, which is a QAT
  product: the scales and grids are part of the trained model.
- Treatment: a documented D8 exception. SUC dequantises each tensor losslessly to BF16 on the fly in the store, and
  the "full-precision reference" for every KL/PPL measurement is the dequantised checkpoint.
- Re-encoding the routed experts, which are already at 4.25 bpw, at ~4.2 bpp would save about 1% of their bytes and
  add a second quantisation error on top of the FP4 grid. **The default is therefore to keep the experts in native
  MXFP4.** SUC is applied to the FP8 tensors, which have a 2× size win. Expert transcoding is a separate research
  tier at ≤3 bpw, with native MXFP4 as the baseline to beat (§6, risk 1).

## 5. Parameter accounting

Computed per pattern from the shapes above. FP4 tensors count 2 parameters per stored byte; scales and `tid2eid`
are counted as bytes, not parameters. The total stored size equals the index `total_size` exactly.

| Category | Params | Stored (incl. scales) | BF16-equivalent |
|---|---|---|---|
| Routed experts (43×256×3×2048×4096) | 277.025 B | 147.17 GB (MXFP4) | 554.05 GB |
| Shared experts (43×3×8.39 M) | 1.082 B | 1.082 GB (FP8) | 2.16 GB |
| Attention linears wq_a/wq_b/wkv/wo_a/wo_b (0.180+1.443+0.090+1.443+1.443) | 4.599 B | 4.600 GB (FP8) | 9.20 GB |
| Indexer wq_b (21×8192×1024) | 0.176 B | 0.176 GB (FP8) | 0.35 GB |
| Compressors + indexer compressors (BF16) | 0.304 B | 0.608 GB | 0.61 GB |
| Indexer weights_proj (BF16) | 0.0055 B | 0.011 GB | 0.011 GB |
| Embed + head (2×129280×4096) | 1.059 B | 2.118 GB | 2.12 GB |
| Router weights (43×256×4096) | 0.0451 B | 0.090 GB | 0.090 GB |
| HC mixers + hc_head (F32) | 0.0339 B | 0.136 GB | 0.068 GB |
| Norms, sinks, ape, gate bias, tid2eid | ~1.9 M | 0.026 GB | — |
| **Main model** | **284.332 B** | **156.02 GB** | 568.7 GB |
| DSpark/MTP (768 experts 19.33 B, markov 0.066 B, rest 0.452 B) | 19.846 B | 10.86 GB | 39.7 GB |
| **Total** | **304.178 B** | **166.88 GB** | 608.4 GB |

Active parameters per token, main model and excluding the embedding lookup:
- 43 × (6+1) experts × 25.17 M = 7.577 B
- plus attention 4.599 B
- plus indexer `wq_b` 0.176 B
- plus compressors and `weights_proj` 0.310 B
- plus router 0.045 B
- plus HC 0.034 B
- plus head 0.530 B
- **≈ 13.27 B**

## 6. SUC encoding plan

**Eligible**, main model only, 282.88 B params. Proposed adapter roles:

| Role (new unless noted) | Tensor | Shape (out×in) | Notes |
|---|---|---|---|
| `attn_q_a` | `attn.wq_a` | 1024×4096 | reads attn_norm output |
| `attn_q_b` | `attn.wq_b` | 32768×1024 | reads q_norm output |
| `attn_kv` | `attn.wkv` | 512×4096 | single K=V head; feeds every attention score, so it is sensitive |
| `attn_o_a` | `attn.wo_a` | 8 groups × (1024×4096) | grouped: row-group g reads input slice g; base representation per row-group (D6), rotation applied per group slice |
| `attn_o_b` | `attn.wo_b` | 4096×8192 | |
| `indexer_q_b` | `attn.indexer.wq_b` | 8192×1024 | only feeds the top-k selection; errors change which entries are selected, not the values |
| `shared_expert_{gate,up,down}` (existing) | `ffn.shared_experts.w1/w3/w2` | 2048×4096 / 4096×2048 | w1=gate, w3=up, w2=down |
| `expert_{gate,up,down}` (existing) | `ffn.experts.E.w1/w3/w2` | same | **default: keep native MXFP4** (eligible only for the ≤3 bpw transcode tier) |

**Kept** (1.45 B params, about 2.99 GB at stored precision):
- `router` (BF16), `gate.bias`, `tid2eid` (stored as uint8: expert ids < 256, lossless, 2.3 MB).
- Compressor and indexer-compressor `wkv`/`wgate` and `weights_proj` (BF16, computed in fp32). These are new roles
  `compressor_kv`, `compressor_gate`, `indexer_weights`, with `eligible=False` by default and an ablation row for
  SUC on them later.
- All norms, `attn_sink`, `ape`, every `hc_*` (F32).
- `embed` and `head`: affine 4-bit g64 per the SUC1 codec, 4.5 bpp, 0.60 GB.
- DSpark/MTP (`mtp.*`, 19.85 B) is **dropped**, as transformers does (`_keys_to_ignore_on_load_unexpected`,
  tf-modeling:1256) and v41 does (`_DROP_PREFIXES`, v41:1014). It could be added back later as a speculative-decoding
  option.

**Size estimates** (main model; "rest" = attention, shared and indexer_q_b at 4.2 bpp = 3.08 GB; kept non-embedding
tensors = 0.87 GB):

| Plan | Experts | Total (embed/head 4-bit) | Total (embed/head BF16) |
|---|---|---|---|
| SUC1 everywhere at 4.2 bpp | 145.4 GB | **149.98 GB** | 151.50 GB |
| Native MXFP4 experts + SUC1 rest (recommended baseline) | 147.17 GB | **151.71 GB** | 153.23 GB |
| Experts transcoded to 3.0 bpw | 103.9 GB | 108.4 GB | 110.0 GB |
| Experts 2.5 bpw | 86.6 GB | 91.1 GB | 92.6 GB |
| Experts 2.2 bpw | 76.2 GB | 80.7 GB | 82.2 GB |
| Experts 2.0 bpw | 69.3 GB | 73.8 GB | 75.3 GB |

**Layer-streaming memory**, D8, per block:
- One MoE block holds 6.44 B expert params: 3.42 GB stored, **12.9 GB dequantised to BF16**. Attention adds about
  0.21 GB in BF16, and compressors/HC about 0.1 GB.
- With the expert-stack codebook scope (one codebook per layer×projection over 256 experts), the unit of work is
  256×2048×4096 = 2.15 B params: 4.3 GB BF16, 8.6 GB fp32.
- Calibration activations are **4× wider** because of the HC streams. 512 × 512 tokens × 4 × 4096 × 2 B = **8.6 GB
  per stream buffer** in BF16, and ~17 GB for input plus output, or 34 GB in fp32.
- Estimated peak for one block: ~35-50 GB. That is fine on the 128 GB Studio and impossible on the 16 GB laptop.

## 7. Hadamard compatibility

Distinct `in_features` (per group for `wo_a`):

| in_features | m·2^k | Tensors | OK |
|---|---|---|---|
| 4096 | 1·2^12 | wq_a, wkv, wo_a (per group), shared w1/w3, expert w1/w3, compressors, weights_proj, router | yes |
| 1024 | 1·2^10 | wq_b, indexer.wq_b | yes |
| 8192 | 1·2^13 | wo_b | yes |
| 2048 | 1·2^11 | shared w2, expert w2 | yes |
| 12288 | 12·2^10 | mtp.0.main_proj (dropped) | yes |

Every eligible linear is compatible and no fallback is needed.

Notes:
- `wo_a` must be rotated **per group**: the same H applied to each 4096-wide input slice. Applying one 32768-wide
  rotation would mix groups and break the block-diagonal structure.
- `wq_b` and `indexer.wq_b` both read `qr`. If they share a rotation seed, one `H·qr` serves both.
- The model's own `rotate_activation` (indexer q/k, model.py:253-257) is an activation transform unrelated to SUC
  weight rotation.

## 8. MLX implementation

**What exists.** mlx-lm has no `deepseek_v4` module. The closest relative is `deepseek_v41.py`, for V4.1 on mlx-lm
main. Its sanitize (v41:1105-1165):
- maps the reference names to HF-style MLX names via `_rename` (v41:1019-1046): `attn→self_attn`, `wq_a→q_a_proj`,
  `wo_a→o_a_proj`, `hc_attn_*→attn_hc.*`, `attn_sink→sinks`, `gate.bias→e_score_correction_bias`, `w1/w3/w2 →
  gate/up/down_proj`;
- stacks the experts into `switch_mlp`, keeps FP8/FP4 packed (`_repack`), splits `wo_a` into its 8 groups, and drops
  `mtp.*`.

**Reusable unchanged:**
- `HyperConnection` math and `_sinkhorn`, including its Metal kernel, which uses the same order as kernel.py:401-423.
- `_collapse` / `_expand`; HC parameters stay F32 via `cast_predicate`.
- `_sparse_attention` (gather plus `mx.fast.scaled_dot_product_attention` with `sinks`).
- `_window_indices`, `_apply_rope` (interleaved, `traditional=True`), the YaRN setup (`mscale=0`; window-only layers
  use θ=10000).
- Grouped `o_a_proj` via `SwitchLinear`, `Router` scoring (sqrtsoftplus, bias selection, normalisation × 1.5), the
  clamped SwiGLU, `SwitchGLU` experts, `SharedMLP`.
- The `_fake_quant_*` helpers, and expert MXFP4 repacking.

**Must differ / be written:**
1. **Hash routing**: a `HashRouter` with `tid2eid` (no bias; weights still come from `W_g` scores). `input_ids`
   must be threaded through `TextModel`→`DecoderLayer`→`MoE`. v41 `Router` has no `tid2eid`, and v41 `MoE` gets no
   token ids.
2. **Compressor with `ape` and overlap**:
   - ratio 4 uses `coff=2` (projections [1024,4096]) with the overlap transform and its two-window decode state
     (model.py:313-365);
   - ratio 128 is the plain pooled path with `ape` [128,512].
   - v41's `Compressor` has neither `ape` nor overlap (v41:318-351).
3. **Indexer with its own compressor** (head 128, Hadamard + FP4 simulation). v41 instead derives index keys from
   the shared latent via `k_proj`/`k_norm`, plus cross-layer `kv_source`/`index_source`/candidate-block sharing
   (v41:403-480), none of which exists in V4. V4 has no cross-layer sharing: every compressed layer owns its KV.
4. **Per-head unweighted q RMS norm** after `q_b_proj` (model.py:504). It is absent in v41 (v41:549-552).
5. **HC wiring**:
   - V4 collapses each sublayer input with its *own* `pre` (model.py:696-706), whereas v41 shifts `pre` by one
     sublayer: attention uses the previous layer's `ffn_pre` (v41:757-767).
   - V4 ends with `hc_head` (sigmoid pre with its own `hc_head_fn/base/scale`, model.py:709-716); v41 ends with
     `_collapse` using the last `pre_mix` (v41:1007).
6. **FP8 128×128 repack** (§4): fix `_repack` for 128-row × 128-column tiles.
7. **QAT simulation details** differ from v41:
   - window KV: FP8 round-trip on the 448 non-RoPE dims with block 64 (model.py:512), whereas v41 uses block 32 on
     all dims (v41:556);
   - compressed main KV: FP8 block 64, whereas v41 uses FP4 block 16 with E4M3 scales (v41:608);
   - activations are round-tripped to FP8 before every FP8/FP4 GEMM (model.py:119-124). v41 skips this (v41:100).
   - The default should follow v41 (skip the activation simulation) and measure the KL of doing so as an ablation.
8. `rms_norm_eps` 1e-6 (v41 default 1e-20); `tid2eid` load cast int64→int32/uint8.
9. Optional and deferred: DSpark speculative decoding (block drafting with markov and confidence heads).

**Adapter notes.** The checkpoint names are the reference-implementation names, without the `model.` prefix and not
HF-style (`layers.N.attn.wq_a.weight`, `layers.N.ffn.experts.E.w1.weight`). The SUC adapter must classify these
**original** names; mlx-lm's post-sanitize names are not used for classification.
- Name patterns: `_L = r"layers\.(?P<layer>\d+)"` and `_E = r"ffn\.experts\.(?P<expert>\d+)"`.
- `NORMS = {"attn": "layers.{layer}.attn_norm.weight", "mlp": "layers.{layer}.ffn_norm.weight", "final":
  "norm.weight"}`. If rotation folding folds `attn_norm`, every reader must be included: wq_a, wkv, compressor
  wkv/wgate (×2 for CSA), indexer weights_proj. Likewise q_norm feeds wq_b and indexer.wq_b, and ffn_norm feeds the
  router, the experts and the shared expert. The BF16 kept readers then get folded weights, which is still lossless.
- Scale companions (`*.scale`) are claimed by their `.weight` (a new "quant_scale" nonlinear class) and consumed by
  the store's dequantisation.
- `mtp\..*` gets an explicit "dropped" class so that `validate()` does not fail on 72k "unknown" tensors and does
  not expect them.
- `expected_roles(layer)`:
  - CSA layers additionally require the indexer and compressor;
  - HCA layers require the compressor only;
  - layers 0-1 require neither;
  - layers 0-2 require `tid2eid` and no `gate.bias`.
- `SafetensorsStore._DTYPES` lacks `F8_E4M3` and `F8_E8M0`. Add them as raw `u1` reads, then decode:
  `mx.from_fp8` for E4M3 and `2^(u8−127)` for E8M0.
- Add a `load_dequantized(name)` that pairs weight and scale (including I8 → FP4 unpacking) and returns BF16 one
  tensor at a time (D8).
- The calibration pipeline must pass token ids to each block (hash layers) and handle the [B,S,4,4096] HC residual
  as the block state.
- The transformers class (tf-modeling) uses HF-style names (`self_attn.q_a_proj`, `attn_hc`, fused expert
  `gate_up_proj`, tf-modeling:786-797, 1036). How it maps the reference checkpoint names was not inspected, because
  the conversion mapping lives in a file that was not fetched.

**Test plan.**
1. A tiny random V4 in `tests/unit/tiny_models.py`: hidden 256, 4 layers with ratios [0,4,128,4], 1 hash layer,
   8 experts top-2, hc_mult 4, o_groups 2, plus one `mtp.0` stage to exercise dropping. Write it in the reference
   naming, with FP8 128×128 and MXFP4 tensors generated by a reference quantiser.
2. Adapter: `validate()` passes; role and eligibility counts; every index pattern is classified.
3. Store: FP8/E8M0/FP4 dequantisation is bit-exact against a numpy port of convert.py:11-52 and 126.
4. MLX model numerical identity:
   - against a CPU port of `inference/model.py` with the tilelang kernels replaced by reference torch/numpy code
     (sparse_attn, sinkhorn, act_quant), on the tiny model, for prefill and step-by-step decode across compressor
     boundaries (positions 3/4/7/8 and 127/128);
   - targets: max abs logit error < 1e-4 in fp32 and KL < 1e-6.
5. Real model on the Studio: compare a 2-layer slice (layers 2-3, through `hc_head`/`head` skipped) against the
   reference, then full-model wikitext2 PPL against the published numbers if any exist.

## 9. Risks / open questions

1. **It does not fit at 4.2 bpp.** The main model is ~150 GB at SUC1 or with native MXFP4, against 128 GB of unified
   memory. The default GPU wired limit is about 75%, roughly 96 GB, and can be raised with
   `sysctl iogpu.wired_limit_mb`.
   - Running needs experts at ≤ ~3.0 bpw (108 GB, wired-limit raise required, tight) or ~2.5 bpw (91 GB, fits).
   - Whether VQ at 2-3 bpw beats native MXFP4 per byte is the core open question. The baseline is the QAT FP4
     weights, which are much stronger than post-training 4-bit.
2. **The D8 exception** (FP8/FP4-only source) must be recorded in the result metadata (`source_precision: fp8+mxfp4`).
   Also confirm that the scale exponent ranges keep BF16 dequantisation exact.
3. **Second-generation quantisation error on QAT weights.** The FP4 grid plus E8M0 scales are what the model was
   trained against. VQ on the dequantised values may lose more than on a BF16 model at equal bpp. The ablation is
   to transcode the experts at 3.0 bpw and compare against native and against MLX `mxfp4` repack.
4. **Hash layers.** Experts in layers 0-2 are selected by token id. Calibration coverage per expert depends on the
   token distribution, not on hidden states, so rare-token experts may get few calibration tokens. Report the
   minimum tokens per expert separately for hash layers.
5. **Indexer sensitivity.** Errors in wq_a/q_norm/indexer.wq_b change the top-512 selection (a discrete decision).
   KL may be dominated by selection flips at long context. Evaluate at seq_len ≥ 2048 (≥ 512 compressed entries
   need ≥ 2048 tokens).
6. **Long-context evaluation.** CSA/HCA only matter beyond 128 tokens. The ratio-128 path needs ≥ 128 tokens per
   entry, so PPL at 2048 exercises at most 16 HCA entries. Long-context KL (≥ 16k) is needed for a fair claim.
7. **Streaming memory** (§6): the HC 4× residual makes calibration buffers 4× larger than for a 4096-hidden plain
   model. The 128 GB Studio is fine, with a ~35-50 GB peak; the laptop is excluded. Disk: 167 GB download plus
   output.
8. **Runtime KV.** Per token: window 128×512 (fixed), CSA 21 × (512+128)/4 × 2 B, HCA 20 × 512/128 × 2 B ≈ 6.9 KB.
   That is about 7 GB at 1 M tokens, which is small. The decode cost is the indexer top-k over all compressed
   entries, and `argpartition` over 256 k entries per query is slow in MLX.
9. **Unsupported or unverified ops.**
   - Per-channel softmax pooling and overlap state: plain MLX, fine.
   - Hadamard 128 for the indexer: `mx.hadamard_transform`, available.
   - FP4 fake-quant rounding mode vs tilelang `T.Cast(FP4)` (round-to-nearest-even?): unverified.
   - `fast_round_scale` power-of-two ceil (kernel.py:22-37) vs v41 `ceil(log2)`: unverified on ties.
10. **Config inconsistencies.** `num_nextn_predict_layers: 1` vs 3 stages present. `convert.py:97` tests for
    `"tie2eid"`, a typo for `tid2eid` that is harmless because the key falls back to `gate`. Both are trivial, but
    the adapter should trust the index and `inference/config.json`, not `num_nextn_predict_layers`.
11. **Rotation folding across HC.** The norm weights can be folded, but the HC `pre` collapse happens before the
    norm, so no rotation of the residual stream itself is possible: there are 4 streams with learned mixing. Only
    one-sided input rotations, already the SUC1 design, are valid.

## 10. Summary numbers

```yaml
model: deepseek-ai/DeepSeek-V4-Flash-0731
revision: 7872f01b1d1fe23eabc4c98b48bffcef5a386062
total_params: 304.18e9          # main 284.33e9 + DSpark/MTP 19.85e9 (FP4 counted as 2 params/byte)
bf16_equiv_gb: 608.4            # main model only: 568.7
stored_gb: 166.88               # = index total_size; experts MXFP4 4.25 bpw, attention/shared FP8 128x128, rest BF16/F32
eligible_params: 282.88e9       # experts 277.03e9 + attention 4.60e9 + shared 1.08e9 + indexer wq_b 0.18e9
kept_params: 1.45e9             # embed+head 1.06e9, compressors 0.30e9, router 0.045e9, HC 0.034e9, misc
active_params_per_token: 13.27e9
est_suc1_gb_at_4.2bpp: 151.5    # all eligible 4.2 bpp, kept at stored precision (embed/head BF16)
est_suc1_gb_at_4.2bpp_emb4bit: 150.0
est_native_mxfp4_experts_suc_rest_gb: 151.7   # embed/head 4-bit
est_experts_3.0bpw_gb: 108.4
est_experts_2.5bpw_gb: 91.1
est_experts_2.0bpw_gb: 73.8
fits_128gb_runtime: no          # at 4.2 bpp / native; yes only with experts <= ~2.5-3.0 bpw (wired-limit raise for 3.0)
effort_estimate_person_days: 11 # adapter+FP8/FP4 store 2, MLX model from deepseek_v41 (hash router, overlap compressor, own indexer, q-norm, HC wiring, hc_head, 128x128 repack) 6, tests 3
mlx_lm_support: none (deepseek_v41 is the closest base; ~60% reusable)
```

## 11. Implementation status (2026-10-03, laptop; nothing run on the real weights yet)

| Piece | Where | Verified (tiny random checkpoint in the original FP8 / MXFP4 layout, `tests/unit/test_deepseek_v4.py`) |
|---|---|---|
| Exact FP8 (128x128 E8M0) / MXFP4 (per-32 E8M0) dequantization, one tensor / expert at a time | `suc/io/quantized_source.py`, `SafetensorsStore` (`QuantSource`) | bit-exact vs a numpy port of convert.py; scale-range guard (`NotLossless`) run by preflight over every scale tensor (`run_dir/source_scales.json`) |
| Adapter `deepseek_v4` (checkpoint ↔ SUC namespace, roles, validate) | `suc/models/adapters.py` | the real 72,317-tensor header list (fixture `tests/unit/fixtures/deepseek_v4_flash_tensors.json.gz`): eligible 282,706,575,360 params (experts 277.03 B), every MLX parameter shape = its SUC tensor |
| MLX model (HC + Sinkhorn + hc_head, CSA / HCA compressors with overlap, indexer top-k, q-norm, sinks, YaRN, hash routing via 3 token-id channels in the stream, clamped SwiGLU, shared expert; cached decode) | `suc/models/mlx_deepseek_v4.py` | float32 vs an independent numpy port of model.py: rel. error ~1e-6 per block; cached decode = prefill; on the Mac Studio vs the OFFICIAL inference/model.py with plain-torch kernels (`scripts/deepseek_v4_reference_check.py`): float32 rel. error ≤ 1e-6, logit KL 5e-13 |
| Per-expert Fisher allocation under a total-size budget | `plan.allocation_unit: expert`, `plan.budget_bytes`, `plan.candidate_roles` | budget respected by the plan and the SUC1 file; experts of one stack mix codecs, one pooled codebook fit per (stack, codec) |
| Runtime of mixed-codec stacks | `suc.kernels.suc_switch.SucSwitchMixed` | `load_suc1_model` / `suc generate` from the codes vs the decoded reference |
| Native MXFP4 expert pass-through (user decision 2026-10-03): SUC1 representation `mxfp4` = the checkpoint's E2M1 codes + E8M0 scales verbatim (4.25 bpp, docs/format.md §5.4); greedy candidate `{kind: mxfp4}` (loss 0, exact bits, never Fisher-scored, so a re-plan reuses calibrate / reference / candidates); runtime through MLX `gather_qmm(mode="mxfp4")`, alone or inside `SucSwitchMixed` | `suc.quant.codecs.MXFP4Native`, `suc.pipeline.compress.add_native_candidates` / `scored_plan_delta`, `suc.kernels.mxfp4`; config `compress_deepseek_v4_flash_native.yaml` | SUC1 payload bytes = checkpoint bytes, reference decoder = exact BF16 dequantization; variant run on top of the base run's artifacts skips calibrate / reference / candidates and stays under budget; native experts from the codes = x · W_refᵀ (float32, rel. 1e-5) |

Deviations from the plan above: indexer `wq_b` is NOT eligible (kept BF16: it only drives the top-k selection, which the
Fisher proxy cannot see, and calibration at 512 tokens never selects); `wo_a` stays one [8192, 4096] tensor (the dense
path uses the grouped einsum, a SUC module computes all row groups for every input group and keeps the diagonal blocks);
the activation FP8 round trip before FP8 / FP4 GEMMs is off by default (`simulate_act_fp8`), the KV FP8 and indexer FP4
round trips are on. Config: `experiments/configs/compress_deepseek_v4_flash.yaml` (107.5 GB total, runtime check
≤ 111 GB at 32k context).

## 12. Why the KL is high (analysis 2026-10-05, M3, from the Studio run artifacts; nothing re-run)

Measured (Mac Studio, `compress_deepseek_v4_flash{,_native}`, reference = exact BF16 dequantization of the FP8/MXFP4
checkpoint): base 3.021 bpp, KL 0.250 (p95 1.03, top-1 agreement 82.1 %), PPL 5.43; native 3.021 bpp, KL 0.248
(p95 1.07, top-1 81.9 %), PPL 5.39; reference PPL 4.779. For scale: Qwen3.8-Flash-Next 4.155 bpw KL 0.151,
Qwen3-30B-A3B greedy 4.0 bpp KL 0.064.

**Where the bits go** (`plan/plan.json` + `candidates/candidates.json`; 33,024 routed-expert tensors = 43 × 256 × 3):

| Run | G=4 2.16 bpp (rel. MSE 0.096) | M=3 3.16 bpp (0.0357) | G=2 4.16 bpp (0.0078) | native 4.25 bpp (0) | expert bpw |
|---|---|---|---|---|---|
| base | 18,417 (56 %) 41.7 GB | 5,559 (17 %) 18.4 GB | 9,048 (27 %) 39.5 GB | — | 2.875 |
| native | 19,362 (59 %) 43.8 GB | 4,422 (13 %) 14.7 GB | 0 | 9,240 (28 %) 41.2 GB | 2.878 |

- Attention / shared experts: the greedy already protects them — affine 8-bit on 332–337 of 344 tensors, BF16 on 3
  `wkv`, VQ G=2 on 4 (base) / 9 (native) `wq_b`; ~5.9 GB, Fisher share < 2 %. Embedding / LM head affine 4-bit
  (rel. MSE 0.008), kept tensors 1.22 GB raw. Nothing outside the routed experts is a plausible main KL source.
- Per layer: layers 0–7 (incl. the hash-routed 0–2) get the most G=4 (L0: 683 of 768 tensors), layers 19 and 21–35 the
  most G=2 / native (Fisher share 3–4 % each vs 1.4 % for L0). No per-layer measured error exists (evaluate only gives
  end-to-end PPL / KL), so whether early layers are under-protected is not measured.
- **Fisher-predicted vs measured**: replacing G=2 by native halved the predicted loss (1.22e-4 → 6.37e-5, −48 %) but
  the KL moved −0.7 % (ΔNLL vs reference −6 %). KL / predicted loss is 2,050 (base) and 3,900 (native) vs 650 on
  Qwen3-30B-A3B: the diagonal-Fisher proxy (32 × 512 tokens ≈ 384 routed tokens per expert; 229 expert tensors never
  routed in that sample, Spearman 0.65 with the 262k-token calibration routing counts) is extremely heavy-tailed (1 %
  of the expert tensors hold 39 % of the G=4 loss) and under-weights the large, many small errors of the G=4 tier.
  Two crude proxies track the measured change much better: Σ rel. MSE over expert tensors (−1.0 %) and Σ routing count
  × rel. MSE (−3.9 %). Working hypothesis: the KL is driven by the ~56–59 % of experts at rel. MSE 0.096 (the
  second-order additive model is not valid at that error size); the Fisher barbell (G=4 + ≥ 4.16 bpp) is the wrong
  shape at ~2.9 bpw.
- **FP4-trained experts vs requantization**: every SUC expert codec rotates the input (`rot1s`), which turns the E2M1 ×
  E8M0 lattice into a near-Gaussian source: rel. MSE is identical for all 33,024 tensors to 3 digits (G=4 0.0959,
  M=3 0.0357, G=2 0.0078), about 1.5–2.3× the Gaussian rate–distortion bound of the index rate (0.0625 / 0.0156 /
  0.0039). The codecs gain nothing from the checkpoint being FP4 already, and G=2 (4.16 bpp, error > 0) is strictly
  dominated by native MXFP4 (4.25 bpp, error 0) — the native plan took it 1:1. A lattice-aware sub-4-bit codec (e.g.
  E2M1 → 3-bit sub-grid with the E8M0 scale kept) is the obvious next codec; not implemented, not measured.

**Variants** (plan-only re-plans of the base run's candidates with `suc.allocation.greedy.greedy_allocate`, same fixed
part 1.958 GB; they reproduce both measured plans exactly). The runtime limit for weights is 111 GB − KV 0.231 GB
(32k) − activations 2 GB = 108.77 GB → 108.75 GB. "Σ rel. MSE" / "count-weighted" are relative to the base run.

| Variant (config `compress_deepseek_v4_flash_<v>.yaml`) | File | Experts G4 / M3 / native | Attention / shared | Fisher pred. | Σ rel. MSE | count-weighted |
|---|---|---|---|---|---|---|
| base (measured KL 0.250) | 107.50 GB | 18,417 / 5,559 / — (+9,048 G=2) | 337 A8, 4 G2, 3 BF16 | 1.22e-4 | 1.000 | 1.000 |
| native (measured KL 0.248) | 107.50 GB | 19,362 / 4,422 / 9,240 | 332 A8, 9 G2, 3 BF16 | 6.37e-5 | 0.990 | 0.961 |
| V1 `native_max`: native, budget 108.75 GB, G=2 excluded | 108.75 GB | 18,822 / 4,412 / 9,790 | 335 A8, 6 G2, 3 BF16 | 5.82e-5 | 0.964 | 0.924 |
| V2 `flat_max`: experts G=4 / M=3 only, 108.75 GB | 108.75 GB | 8,490 / 24,534 / — | 309 A8, 35 BF16 | 3.22e-4 | 0.830 | 0.921 |
| V3 `native_stream`: all experts native + SSD expert streaming | 155.17 GB | 0 / 0 / 33,024 | 344 A8 | 9.1e-7 | 0 | 0 |

- V1 is the low-risk step (+1.25 GB, every proxy improves a little); expect at most a few % on KL.
- V2 is the discriminating experiment: Fisher predicts 5× worse than native, the rel.-MSE proxies predict better.
  Its measured KL decides which proxy drives the remaining frontier models (and whether the greedy needs a
  routing-count / error-size-aware loss instead of the raw diagonal Fisher).
- V3 keeps the runtime at ≤ 111 GB through `preflight.expert_streaming` (as GLM-5.3): resident 8.0 GB non-expert +
  KV + activations, expert cache 100.8 GB = 68 % of the routed experts, ~89 % of the routings by calibration counts
  (worst layer 80 %); the rest is read from SSD per token (decode-speed cost not measured). It measures the KL floor
  that does not come from the routed experts (head 4-bit, attention 8-bit) and is the quality ceiling at 111 GB.
- An all-M=3 expert set (109.4 GB of experts alone) does not fit; G=4 cannot be dropped entirely at ≤ 111 GB resident.

The variants reuse calibrate / reference / candidates: new `plan.allocation` field (`suc.config.CompressPlanConfig`:
`budget_bytes` override and `exclude: {candidate: [roles]}`), applied by `allocate_plan`, `predicted_weight_bytes` and
the encode memory estimate, and dropped from the candidates-stage fingerprint (`scored_plan_delta`); existing
fingerprints are unchanged (field default `None`). Test: `tests/unit/test_deepseek_v4.py::
test_replan_variant_configs_reuse_the_base_run` (identical candidates-stage inputs, runtime check, exclusions and
budget honoured on the real tensor list). Launch commands are in each config header (copy the base run's calibrate /
reference / candidates / source_scales.json into the variant's run dir first). Not run yet.
