# glm5_next — zai-org/GLM-5.3-Flash

Architecture note for SUC (adapter, MLX model, encoding plan). Facts come from the pinned checkpoint files; formulas
cite the code that defines them. Shapes marked **(H)** were read from a safetensors header, **(D)** are derived from
`config.json`. Where something could not be verified it is stated explicitly.

Citation shorthands:
- `TF:` = transformers main `src/transformers/models/glm5_next/modeling_glm5_next.py` (copy used: scratchpad
  `hf/tf/modeling_glm5_next.py`; the installed transformers 5.18.0 copy differs only in a deprecated `device` kwarg of
  the vision RoPE, checked with `diff`). `TFC:` = `configuration_glm5_next.py` (same source).
- `TFI:` = installed transformers 5.18.0 (`~/.venvs/suc-lm/lib/python3.12/site-packages/transformers/`).
- `MLX:` = mlx-lm main `mlx_lm/models/*.py` (scratchpad `hf/mlx/`); `MLXI:` = installed mlx-lm 0.31.3.

## 1. Identity

| Item | Value |
|---|---|
| Repo / revision | `zai-org/GLM-5.3-Flash` @ `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` |
| `model_type` / architectures | `glm5_next` (text sub-config `glm5_next_text`, vision `glm5_next_vision`) / `Glm5NextForConditionalGeneration` |
| transformers version at export | `5.16.0` (`config.json`, `generation_config.json`) |
| Index `total_size` | 328 326 771 576 B (328.33 GB) — reproduced **exactly** (diff 0 B) by the accounting in §5 |
| Shards / tensors | 62 shards (`model-000NN-of-00062.safetensors`), 76 108 tensors, 93 name patterns (+1 shape variant) |
| Modeling code in repo | none (README points to transformers / SGLang / vLLM); reference = transformers `glm5_next` |
| License / modality | MIT; image-text-to-text (README: "320B total parameters and just 18B active") |
| Sources consulted | `config.json`, `generation_config.json`, `tokenizer_config.json`, `model.safetensors.index.json`, `README.md`; headers of shards **00001** (embed, lm_head, MTP `eh_proj`/`enorm`/`hnorm`), **00002** (MTP layer 45 + dense/KDA layers 0–) and **00062** (vision tower + last layers); TF/TFC above; TFI `conversion_mapping.py`, `integrations/finegrained_fp8.py`, `quantizers/quantizers_utils.py`; MLX `glm_moe_dsa.py`, `deepseek_v32.py`, `kimi_linear.py`, `deepseek_v41.py`; MLXI `gated_delta.py`; SUC `suc/models/adapters.py`, `suc/io/safetensors_store.py`, `suc/quant/hadamard.py`, `docs/spec.md` D7/D8 |

**Every** name pattern in the index has a header-measured example (`cover.py`: 61 patterns from shard 00002, 26 from
00062, 5 from 00001, plus the two `o_proj` shape variants). All shapes in §3 are therefore **(H)**; the per-layer
repetition (which layer has which pattern) comes from the index and `config.json`.

## 2. Architecture summary

| Config key (`text_config`) | Value |
|---|---|
| `num_hidden_layers` | 45 decoder layers (0–44) + 1 MTP layer (index 45, `num_nextn_predict_layers` 1) |
| `hidden_size` / `vocab_size` | 4096 / 154 880; `tie_word_embeddings` false |
| `layer_types` | 34 × `linear_attention` (KDA) + 11 × `deepseek_sparse_attention` (MLA + DSA indexer) |
| KDA (`linear_attn_config`) | 64 heads × head_dim 128 (q/k/v 8192), `short_conv_kernel_size` 4, `gate_lower_bound` −5.0 |
| MLA | 64 heads (`num_key_value_heads` 64), `q_lora_rank` 1536, `kv_lora_rank` 512, `qk_nope_head_dim` 256, `qk_rope_head_dim` **0**, `v_head_dim` 256, `mla_use_nope` true |
| DSA indexer | `index_n_heads` 32, `index_head_dim` 128, `index_topk` 2048, `index_kpool` 4 (`index_kpool_compress` true, `always_select_tail` true), `indexer_types` all `full` |
| mHC | `mhc` true, `hc_mult` 4, `hc_sinkhorn_iters` 20, `hc_eps` 1e-6 |
| MLP | `first_k_dense_replace` 3 → layers 0–2 dense (`intermediate_size` 12288); layers 3–44 MoE |
| MoE | 288 routed experts (`moe_intermediate_size` 2048), top-8, 1 shared expert (2048), sigmoid scoring, `noaux_tc`, `n_group` 1 / `topk_group` 1, `norm_topk_prob` true, `routed_scaling_factor` 2.5, `moe_router_dtype` float32 |
| Activation | SiLU-GLU with `swiglu_limit` 10.0 |
| Norm | RMSNorm eps 1e-5 |
| Positional | **NoPE** in the text model (no RoPE anywhere); `max_position_embeddings` 1 048 576 |
| Vision | 24-block ViT, hidden 1024, 16 heads, patch 14, temporal patch 2, spatial merge 2, out 4096 |

**Per-layer schedule.** KDA layers = `linear_attn_config.kda_layers` = {0,1,2, 4,5,6, 8,9,10, …, 40,41,42, 44}
(**0-indexed**, i.e. every `i % 4 != 3`, plus 44); DSA layers = `full_attn_layers` = {3,7,11,15,19,23,27,31,35,39,43}.
Layers 0–2: KDA + dense MLP. Layers 3–44: MoE (42 layers). MTP layer 45: DSA attention + MoE (index: 38 patterns
under `layers.45.`). Note: the TFC default schedule (`TFC:164-173`) produces the same split, but the shipped
`config.json` names the DSA type `deepseek_sparse_attention`, which TFC only remaps from `full_attention`
(`TFC:171-173`) and which is not a key of the mask mapping (`TF:1489-1492`, looked up at `TF:1500`) — see risk 6.

### 2.1 Hyper-connections (mHC) — residual scheme
The residual stream is `hc_mult`=4 copies: `H0 = embed(x)` broadcast to `[B,S,4,4096]` (`TF:1494`). Each decoder layer
has two HC modules (`attn_hc`, `ffn_hc`; checkpoint `hc_attn_*`, `hc_ffn_*`, renamed by `TFI:conversion_mapping.py:614-619`).
For streams `X ∈ R^{4×4096}` (`TF:286-330`):
- `m = fn · RMSNorm_unweighted(vec(X))` with `fn ∈ R^{24×16384}`, computed in F32 (`TF:301-304`);
- `pre = σ(m_pre·scale0 + base_pre) + eps`, `post = 2·σ(m_post·scale1 + base_post)`,
  `comb = softmax(m_comb·scale2 + base_comb) + eps` (4×4), then Sinkhorn: column-normalise once, then 19 × (row, column)
  normalisation (`TF:314-325`);
- block input `x = Σ_i pre_i X_i` (`TF:329`); output `X' = post ⊗ F(x) + combᵀ X` (`TF:1336-1338`, same for FFN `TF:1345-1347`).

`input_layernorm` / `post_attention_layernorm` are applied **after** the collapse (`TF:1316`, `TF:1343`). Final head:
**unweighted mean** of the 4 streams, then `norm` (`TF:333-337`, `TF:1510`) — no `hc_head` parameters exist.

### 2.2 KDA linear attention (34 layers, `TF:621-770`)
- `q,k,v = SiLU(causal depthwise conv4(W_{q,k,v} x))` (separate `q/k/v_conv1d` [8192,1,4] in the checkpoint, concatenated
  into one conv by `TFI:conversion_mapping.py:637-643`; conv `TF:430-449`, activation = `hidden_act` silu, `TF:711-717`).
- Gate (log-decay per channel, `TF:340-370`): `g = −5 · σ(exp(A_log_h) · (f_b(f_a x) + dt_bias))`, `A_log` [64] per head,
  `dt_bias` [8192] per channel; `g ∈ (−5, 0)` because `gate_lower_bound` −5 selects the "safe gate" branch (`TF:363-364`;
  `TFC:194-198`). This differs from Kimi-Linear's `−exp(A_log)·softplus(·)`.
- `β = σ(b_proj x)` per head (`TF:734`).
- Delta rule with per-key-channel decay (`TF:464-514`): q,k L2-normalised (eps 1e-6), `q ← q/√128`;
  `S ← S·diag(exp g_t)` (decay along the key dim), `δ = β (v − Sᵀk)`, `S ← S + k δᵀ`, `o = Sᵀ q`. Chunked form `TF:518-615`.
  State per layer: 64 × 128 × 128 F32 (`TF:763`).
- Output: `o_proj( RMSNorm_w(o) · σ(g_b(g_a x)) )` per head (`TF:766-768`, gated norm in F32 `TF:375-394`).

### 2.3 MLA with DSA indexer and k-pool compression (11 layers + MTP, `TF:1101-1276`, `TF:773-1061`)
- Query: `q_resid = RMSNorm(q_a x)` (1536), `q = q_b(q_resid)` → 64 × 256 (`TF:1187-1188`).
- KV: `c = RMSNorm(kv_a x)` (512; the rope slice is empty since `qk_rope_head_dim`=0, `TF:1190-1193`), cached as latent;
  `[k_h ‖ v_h] = kv_b(c)` → 64 × (256+256) (`TF:1156-1173`). Scale `256^-0.5` (`TF:1148`). No RoPE (NoPE enforced at
  `TFC:224-227`; `position_embeddings=None` at `TF:1503`).
- Indexer (per query token t): `q_I = wq_b(q_resid)` (32 × 128), `k_I = LayerNorm(wk x)` (1 × 128, shared by heads),
  `γ = W_gate x` (`index_kpool_compress_gate` [128,4096]) (`TF:834-837`). Keys are grouped in pools of 4 consecutive
  tokens starting at the first non-pad token; pool key `K_p = Σ_{j∈p} softmax_j(γ_j + ape[j mod 4]) ⊙ k_I,j`
  (per-channel softmax over the pool, `TF:936-1009`). Score `I(t,p) = Σ_h w_h(t)·32^-0.5 · ReLU(q_I,h·K_p·128^-0.5)`
  with `w = weights_proj x` (`TF:862-867`). Top `2048/4 = 512` complete pools are selected, expanded to raw tokens,
  and the current incomplete tail pool (≤3 tokens) is always appended (`TF:884-914`, `TF:1011-1061`) → ≤2051 keys.
- Attention is ordinary softmax attention restricted to those keys via a mask (`TF:1238-1276`). `indexer_types` are all
  `full`, so no cross-layer top-k sharing is used in this checkpoint (`TF:1150-1154`).

### 2.4 MoE and dense MLP
- Router (`TF:158-183`): `logits = x·Wᵀ` in F32; `s = σ(logits)`; choice score `s + e_score_correction_bias`; group
  limit is a no-op (1 group); top-8 by choice score; weights = `s` (without bias) of the chosen experts, normalised to
  sum 1 (+1e-20), × 2.5.
- Experts: `down( SiLU(min(gate, 10)) · clip(up, −10, 10) )` (`TF:137-142`); shared expert identical form, added
  unweighted (`TF:200-207`); dense MLP same clamps (`TF:98-104`). HF fuses per-expert tensors into `gate_up_proj`
  [288, 4096, 4096] / `down_proj` (`TFI:conversion_mapping.py:620-636`); the checkpoint stores them per expert.

### 2.5 MTP, vision, other
- MTP layer 45: `eh_proj` [4096, 8192] **(H)** (= concat of `enorm(embed)` and `hnorm(hidden)` → 4096, DeepSeek-style),
  a full DSA attention + MoE block **without** HC parameters, and `shared_head.norm` (head weights shared with `lm_head`).
  transformers does **not** implement it: `_keys_to_ignore_on_load_unexpected = [layers.45., shared_head.]` (`TF:1379`).
  Its exact forward (single stream? how it interacts with mHC) is therefore unverified. `index_share_for_mtp_iteration`
  true hints at indexer reuse during MTP. SUC: drop.
- Vision tower (`model.visual.*`, 0.56 B params, BF16): ignore for the text-only pipeline (keep optional, BF16 or
  4-bit later).
- No n-gram / engram embedding tables.

## 3. Checkpoint tensor inventory

All shapes (H). Prefix `L.` = `model.language_model.layers.#.`; counts from the index. Treatment column: **SUC** =
eligible linear (SUC1 codec), **keep** = stored as is (dequantised FP8 → BF16 where applicable), **drop**.

| pattern | count | dtype | shape | role | SUC treatment |
|---|---:|---|---|---|---|
| `lm_head.weight` | 1 | BF16 | [154880, 4096] | lm_head | keep BF16 / affine 4-bit g64 |
| `model.language_model.embed_tokens.weight` | 1 | BF16 | [154880, 4096] | embed | keep BF16 / affine 4-bit g64 |
| `model.language_model.norm.weight` | 1 | BF16 | [4096] | final norm | keep |
| `L.input_layernorm.weight`, `L.post_attention_layernorm.weight` | 46 + 46 | BF16 | [4096] | norms (0–45) | keep |
| `L.hc_{attn,ffn}_fn` | 45 + 45 | BF16 | [24, 16384] | mHC projection | keep |
| `L.hc_{attn,ffn}_base` | 45 + 45 | F32 | [24] | mHC bias | keep |
| `L.hc_{attn,ffn}_scale` | 45 + 45 | F32 | [3] | mHC scales | keep |
| `L.self_attn.{q,k,v}_proj.weight` (KDA) | 34 × 3 | BF16 | [8192, 4096] | kda_q/k/v | SUC |
| `L.self_attn.o_proj.weight` (KDA layers) | 34 | BF16 | [4096, 8192] | kda_o | SUC |
| `L.self_attn.{q,k,v}_conv1d.weight` | 34 × 3 | BF16 | [8192, 1, 4] | short conv | keep (F32 at runtime, `TF:1378`) |
| `L.self_attn.f_a_proj.weight`, `g_a_proj.weight` | 34 + 34 | BF16 | [128, 4096] | KDA gate low-rank A | keep |
| `L.self_attn.f_b_proj.weight`, `g_b_proj.weight` | 34 + 34 | BF16 | [8192, 128] | KDA gate low-rank B | keep |
| `L.self_attn.b_proj.weight` | 34 | BF16 | [64, 4096] | KDA beta | keep |
| `L.self_attn.A_log` | 34 | F32 | [64] | KDA decay | keep F32 |
| `L.self_attn.dt_bias` | 34 | F32 | [8192] | KDA gate bias | keep F32 |
| `L.self_attn.o_norm.weight` | 34 | BF16 | [128] | gated RMSNorm | keep |
| `L.self_attn.q_a_proj.weight` (+`_scale_inv`) | 12 | F8_E4M3 (+F32) | [1536, 4096] (+[12, 32]) | mla_q_a | SUC |
| `L.self_attn.q_a_layernorm.weight` | 12 | BF16 | [1536] | norm | keep |
| `L.self_attn.q_b_proj.weight` (+`_scale_inv`) | 12 | F8_E4M3 (+F32) | [16384, 1536] (+[128, 12]) | mla_q_b | SUC |
| `L.self_attn.kv_a_proj_with_mqa.weight` (+`_scale_inv`) | 12 | F8_E4M3 (+F32) | [512, 4096] (+[4, 32]) | mla_kv_a | SUC |
| `L.self_attn.kv_a_layernorm.weight` | 12 | BF16 | [512] | norm | keep |
| `L.self_attn.kv_b_proj.weight` | 12 | BF16 | [32768, 512] | mla_kv_b | SUC (see §6) |
| `L.self_attn.o_proj.weight` (+`_scale_inv`) (DSA layers) | 12 | F8_E4M3 (+F32) | [4096, 16384] (+[32, 128]) | mla_o | SUC |
| `L.self_attn.indexer.wq_b.weight` | 12 | BF16 | [4096, 1536] | indexer_q | keep |
| `L.self_attn.indexer.wk.weight` | 12 | BF16 | [128, 4096] | indexer_k | keep |
| `L.self_attn.indexer.k_norm.{weight,bias}` | 12 + 12 | BF16 | [128] | LayerNorm | keep |
| `L.self_attn.indexer.weights_proj.weight` | 12 | BF16 | [32, 4096] | indexer head weights | keep |
| `L.self_attn.indexer.index_kpool_compress_gate` | 12 | BF16 | [128, 4096] | k-pool gate | keep |
| `L.self_attn.indexer.index_kpool_compress_ape` | 12 | BF16 | [4, 128] | k-pool pos. bias | keep |
| `L.mlp.{gate,up}_proj.weight` (+`_scale_inv`) (layers 0–2) | 3 + 3 | F8_E4M3 (+F32) | [12288, 4096] (+[96, 32]) | mlp_gate/up | SUC |
| `L.mlp.down_proj.weight` (+`_scale_inv`) (layers 0–2) | 3 | F8_E4M3 (+F32) | [4096, 12288] (+[32, 96]) | mlp_down | SUC |
| `L.mlp.gate.weight` | 43 | BF16 | [288, 4096] | router | keep BF16 |
| `L.mlp.gate.e_score_correction_bias` | 43 | F32 | [288] | router bias | keep F32 |
| `L.mlp.experts.#.{gate,up}_proj.weight` (+`_scale_inv`) | 12384 + 12384 | F8_E4M3 (+F32) | [2048, 4096] (+[16, 32]) | expert_gate/up | SUC |
| `L.mlp.experts.#.down_proj.weight` (+`_scale_inv`) | 12384 | F8_E4M3 (+F32) | [4096, 2048] (+[32, 16]) | expert_down | SUC |
| `L.mlp.shared_experts.{gate,up}_proj.weight` (+`_scale_inv`) | 43 + 43 | F8_E4M3 (+F32) | [2048, 4096] (+[16, 32]) | shared_expert_gate/up | SUC |
| `L.mlp.shared_experts.down_proj.weight` (+`_scale_inv`) | 43 | F8_E4M3 (+F32) | [4096, 2048] (+[32, 16]) | shared_expert_down | SUC |
| `layers.45.eh_proj.weight` | 1 | BF16 | [4096, 8192] | MTP | drop |
| `layers.45.{enorm,hnorm,shared_head.norm}.weight` | 3 | BF16 | [4096] | MTP | drop |
| `model.visual.blocks.#.attn.qkv.{weight,bias}` | 24 + 24 | BF16 | [3072, 1024] / [3072] | vision | drop (optional keep) |
| `model.visual.blocks.#.attn.proj.{weight,bias}` | 24 + 24 | BF16 | [1024, 1024] / [1024] | vision | drop |
| `model.visual.blocks.#.attn.{q,k}_norm.weight` | 24 + 24 | BF16 | [64] | vision | drop |
| `model.visual.blocks.#.mlp.{gate,up}_proj.{weight,bias}` | 48 + 48 | BF16 | [4096, 1024] / [4096] | vision | drop |
| `model.visual.blocks.#.mlp.down_proj.{weight,bias}` | 24 + 24 | BF16 | [1024, 4096] / [1024] | vision | drop |
| `model.visual.blocks.#.norm{1,2}.weight` | 24 + 24 | BF16 | [1024] | vision | drop |
| `model.visual.patch_embed.proj.{weight,bias}` | 1 + 1 | BF16 | [1024, 3, 2, 14, 14] / [1024] | vision | drop |
| `model.visual.post_layernorm.weight` | 1 | BF16 | [1024] | vision | drop |
| `model.visual.downsample.{weight,bias}` | 1 + 1 | BF16 | [4096, 1024, 2, 2] / [4096] | vision | drop |
| `model.visual.merger.{gate,up}_proj.weight` | 1 + 1 | BF16 | [10240, 4096] | vision | drop |
| `model.visual.merger.down_proj.weight` | 1 | BF16 | [4096, 10240] | vision | drop |
| `model.visual.merger.proj.weight` | 1 | BF16 | [4096, 4096] | vision | drop |
| `model.visual.merger.post_projection_norm.{weight,bias}` | 1 + 1 | BF16 | [4096] | vision | drop |

The 12 MLA/indexer counts are layers {3,7,…,43} + 45; the 43 MoE counts are layers 3–44 + 45; 12 384 = 43 × 288.
The MTP layer's attention/MoE tensors are part of the rows above (they share patterns) but are dropped with layer 45.

## 4. Stored dtypes and quantization

**Format** (`quantization_config`): `quant_method: fp8`, `fmt: e4m3`, `weight_block_size: [128, 128]`,
`activation_scheme: dynamic`. Each FP8 weight `W ∈ F8_E4M3^{M×N}` has a sibling `weight_scale_inv` in **F32** with
shape `[⌈M/128⌉, ⌈N/128⌉]` (verified on every FP8 pattern in the headers; all M, N are multiples of 128, so no padding
occurs). Despite the name, the scale is a **multiplier**:

```
W_deq[i, j] = float(W_fp8[i, j]) * weight_scale_inv[i // 128, j // 128]
```

- `TFI:integrations/finegrained_fp8.py:1063-1083` (`Fp8Dequantize._dequantize_one`: block size inferred from the scale
  grid, F32 math, `q * s`, cast to the target dtype); E8M0/`uint8` scale branch not used here (scales are F32).
- `MLX:deepseek_v32.py:496-519` (`mx.from_fp8(weight, bfloat16)`, pad to 128, `weight * scale_inv[:, None, :, None]`,
  cast BF16) — the same formula; this is the sanitize mlx-lm applies to FP8 DeepSeek/GLM-DSA checkpoints.
- Activations: `dynamic` = per-token, per-128-channel FP8 activation quantization inside FP8 kernels (`TFI:finegrained_fp8.py:179-205`);
  irrelevant for SUC (BF16 activations).

**Which tensors are FP8** (index: a `.weight` has a `weight_scale_inv` sibling): 37 338 weights — routed experts, shared
experts, dense MLP, and MLA `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `o_proj` (DSA layers only). Header dtype is
`F8_E4M3` for all of them.

**`modules_to_not_convert` (1509 entries) summary.** Entries use the *pre-wrapper* naming (`model.layers.N…`,
`visual.…`, without `language_model.`), plus generic names. Content:
- generic: `attn_mha`, `attn_mqa`, `dt_bias`, `hyper_connection`, `lm_head`, `mapping_proj`, `router`, `weights_proj`,
  `model.embed_tokens`, `model.norm`, `model.visual`, `visual`;
- per layer 0–44: `hc_{attn,ffn}_{fn,base,scale}` (6 × 45), `input_layernorm`/`post_attention_layernorm` (×46 incl. 45);
- 34 KDA layers: `q/k/v_proj`, `o_proj`, `q/k/v_conv1d`, `f_a/f_b/g_a/g_b_proj`, `b_proj`, `A_log`, `dt_bias`, `o_norm`,
  plus fused names absent from the checkpoint (`qkv_proj`, `fused_qkvbfg_a_proj` — an inference-engine fusion);
- 12 DSA layers (incl. 45): `kv_b_proj`, `q_a_layernorm`, `kv_a_layernorm`, all 7 indexer tensors;
- 43 MoE layers: `mlp.gate`, `mlp.gate.e_score_correction_bias`; MTP: `eh_proj`, `enorm`, `hnorm`, `shared_head.norm`;
- vision: all 24 blocks' tensors and the merger/patch-embed/downsample/post-layernorm.

**Verification against headers** (script over index + 3 headers): after normalising `model.language_model.` →
`model.` and `model.visual` → `visual`, **all 892 non-FP8 `.weight` tensors are covered by an exact entry**, and **none
of the 37 338 FP8 weights is covered**. Header dtypes agree: every non-FP8 tensor is BF16, except F32 for
`hc_*_base`, `hc_*_scale`, `A_log`, `dt_bias`, `e_score_correction_bias` (and the scale tensors). So the non-FP8 set is:
embeddings, LM head, **all KDA attention**, MLA `kv_b_proj`, the whole indexer, routers, mHC, norms, MTP glue, vision.
The vendor kept all linear-attention projections and `kv_b_proj` in BF16 — a sensitivity hint for SUC allocation.

**D8 conflict.** D8 (`docs/spec.md` §0) requires a full-precision BF16/FP16 source and lists FP8 artifacts as
comparison-only. This repo ships no BF16 copy (whether a separate BF16 repo exists was not checked); ~98 % of the parameters (all experts) are
FP8 only. Proposed treatment:
1. Treat the **dequantised FP8 checkpoint as the full-precision reference** (state it in every result JSON:
   `source_format: fp8_e4m3_block128`). Dequantise per tensor on the fly with the formula above (D8 layer streaming is
   unaffected: one tensor at a time).
2. Dequantise into **F32** for SUC fitting. Into BF16 it is exact only if the scales are powers of two: E4M3 has a
   4-bit significand, an arbitrary F32 scale up to 24 bits, so the product generally needs more than BF16's 8 bits.
   Open check on the Studio: `log2(scale_inv)` integral for all blocks? If yes, BF16 dequant is lossless.
3. Transcoding is worthwhile: the source is 8 bpw (+0.002 bpp scales), so SUC1 at ~4.2 bpp halves it. This is not
   the "re-quantising a 4-bit tensor" case. Native FP8 cannot run efficiently on M1 (no FP8 matmul units) anyway.
4. Baseline comparisons: the FP8 checkpoint itself is the quality reference (KL 0); MLX affine 4-bit of the
   dequantised weights is the strong low-bit baseline.

`suc/io/safetensors_store.py:18-31` has no `F8_E4M3` entry (load would raise `NotImplementedError`, line 102-103); add
`"F8_E4M3": (np.dtype("u1"), None)` + `mx.from_fp8` view, and a `load_dequantized(name)` that pairs `weight_scale_inv`.
(`suc/orchestrate/hub.py:16` already knows the byte size of F8 types.)

## 5. Parameter accounting

Computed from header shapes × index counts (scratchpad `glm53_acct.py`); stored bytes sum to the index `total_size`
exactly. Scale tensors (18.8 M F32 values) are counted in stored bytes, not in params.

| Category | Params | BF16-equiv GB | Stored GB | Arithmetic |
|---|---:|---:|---:|---|
| Routed experts (L3–44) | 304.41 B | 608.81 | 304.48 | 42 × 288 × 3 × 2048 × 4096 |
| Shared experts (L3–44) | 1.057 B | 2.11 | 1.06 | 42 × 3 × 2048 × 4096 |
| Dense MLP (L0–2) | 0.453 B | 0.91 | 0.45 | 3 × 3 × 12288 × 4096 |
| KDA attention (34 L) | 4.683 B | 9.37 | 9.37 | per layer 137.7 M: q/k/v/o 4 × 33.55 M + f/g low-rank 2.10 M + b 0.26 M + conv 0.10 M |
| MLA (11 L) | 1.292 B | 2.58 | 1.48 | per layer 117.4 M: q_a 6.29 + q_b 25.17 + kv_a 2.10 + kv_b 16.78 + o 67.11 M |
| DSA indexer (11 L) | 0.082 B | 0.16 | 0.16 | per layer 7.47 M (wq_b 6.29 M) |
| Router (42 L) | 0.050 B | 0.10 | 0.10 | 42 × 288 × 4096 (+ F32 bias) |
| mHC (45 L) | 0.035 B | 0.07 | 0.07 | 45 × 2 × (24 × 16384 + 27) |
| Embed + LM head | 1.269 B | 2.54 | 2.54 | 2 × 154880 × 4096 |
| Norms / other | 0.0004 B | 0.00 | 0.00 | |
| MTP layer 45 (all) | 7.433 B | 14.87 | 7.49 | 288 experts 7.25 B + shared, MLA, indexer, router, `eh_proj` 33.6 M |
| Vision | 0.564 B | 1.13 | 1.13 | |
| **Total** | **321.32 B** | **642.65** | **328.33** | index total_size 328 326 771 576 B ✓ |

Main text model (no MTP, no vision): **313.33 B params**; routed experts are 97.2 % of it.

- **Active per token** (main model): `313.33 − 304.41 × (1 − 8/288)` = **17.38 B** (16.74 B excluding the embedding
  lookup); README says "18B active".
- Largest single tensors: embed / lm_head 634 M each; largest linear: DSA `o_proj` 67.1 M.
- Per decoder block: MoE block 7.40–7.41 B params (14.8 GB BF16, ~7.5 GB stored FP8); dense block 0.29 B.

## 6. SUC encoding plan

**Eligible** (311.77 B params, all in layers 0–44):

| Family | Params | Proposed adapter role | in_features |
|---|---:|---|---:|
| routed experts gate/up/down | 304.41 B | `expert_gate/up/down` (expert_stack codebooks, 288 per stack) | 4096 / 2048 |
| shared experts | 1.057 B | `shared_expert_gate/up/down` | 4096 / 2048 |
| dense MLP L0–2 | 0.453 B | `mlp_gate/up/down` | 4096 / 12288 |
| KDA q/k/v | 3.42 B | **new** `kda_q`, `kda_k`, `kda_v` (or reuse `attn_q/k/v` with a layer-kind tag) | 4096 |
| KDA o | 1.14 B | **new** `kda_o` | 8192 |
| MLA q_a, q_b, kv_a, kv_b, o | 1.29 B | **new** `mla_q_a`, `mla_q_b`, `mla_kv_a`, `mla_kv_b`, reuse `attn_o` | 4096 / 1536 / 4096 / 512 / 16384 |

**Keep** (main model, 1.556 B params, 3.11 GB stored): embed + lm_head (2.54 GB BF16), indexer (`wq_b`, `wk`,
`weights_proj`, kpool gate/ape, k_norm — 82 M; selection is discrete, errors flip top-k), router + bias (BF16/F32,
D8 default), KDA gate low-rank `f_a/f_b/g_a/g_b` (107 M) and `b_proj` (9 M) — these drive the decay/β/output gate and
are tiny, `A_log`, `dt_bias`, `q/k/v_conv1d`, `o_norm`, mHC `fn/base/scale` (35 M; their output feeds a Sinkhorn
normaliser), all norms. New non-eligible roles for the adapter: `kda_gate_a/b`, `kda_out_gate_a/b`, `kda_beta`,
`indexer_q/k/weights/kpool_gate`, `hc_fn`.

Notes:
- `kv_b_proj` is BF16 in the source (vendor-protected). MLX runs it absorbed (split per head into `embed_q`
  [64, 256, 512]ᵀ and `unembed_out` [64, 256, 512], `MLX:deepseek_v32.py:530-571`). Encode it either as a 2D tensor and
  decode before the split (research layout), or encode the two absorbed halves (in_features 256 and 512, both 2^k).
  First pass: keep BF16 (0.37 GB) and treat as a low-priority ablation.
- KDA projections are BF16-only in the source and sit on the recurrent path (errors persist in state S); give them
  priority in the Fisher allocator rather than a fixed 4 bpw.
- Already-quantized source: transcode FP8 → (F32) → SUC1, per tensor (§4). MTP and vision: dropped (MTP could be
  added later for speculative decoding; 7.43 B params, ~3.9 GB at 4.2 bpp).

**Size estimates** (main text model):
- eligible at 4.2 bpp: 311.77 B × 4.2 / 8 = **163.68 GB**; kept at stored precision: **3.11 GB** → **166.79 GB**.
- variant with embed + lm_head at affine 4-bit g64 (4.5 bpp): 163.68 + (3.11 − 2.54 + 0.71) = **164.97 GB**.
- Bits per eligible weight that would fit a ~110 GB weight budget on the 128 GB Studio: (110 − 2) / 311.77 B × 8 ≈
  **2.8 bpp** (≤ 3 bpw regime → D3 requires AQLM/QuIP# comparison).

**Layer-streaming memory** (D8): one MoE block = 7.41 B params → 14.8 GB BF16 (29.6 GB if held in F32; better: keep
FP8 resident, 7.5 GB, and dequantise one expert tensor (8.4 M → 34 MB F32) at a time). Calibration activations are
**4× larger than usual** because of mHC: 512 seqs × 512 tok × 4 streams × 4096 × 2 B = **8.6 GB per copy** (BF16), so
input + output streams ≈ 17 GB (34 GB in F32). Peak per block ≈ 7.5 (FP8 block) + 17–34 (activations) + expert-routing
buffers → 30–50 GB: fits 128 GB, does not fit the 16 GB laptop (dev work must use a tiny random model).

## 7. Hadamard compatibility

Distinct `in_features` of eligible linears (`suc/quant/hadamard.py:29`, m ∈ {1, 12, 20, 28}):

| in_features | factorization | tensors | compatible |
|---:|---|---|---|
| 512 | 1 · 2^9 | `kv_b_proj` | yes |
| 1536 | 12 · 2^7 | `q_b_proj` | yes |
| 2048 | 1 · 2^11 | expert / shared `down_proj` | yes |
| 4096 | 1 · 2^12 | experts/shared/dense gate+up, KDA q/k/v, `q_a_proj`, `kv_a_proj_with_mqa` | yes |
| 8192 | 1 · 2^13 | KDA `o_proj` | yes |
| 12288 | 12 · 2^10 | dense `down_proj` | yes |
| 16384 | 1 · 2^14 | MLA `o_proj` | yes |

All eligible tensors are compatible; no fallback needed. Kept-but-possible later: 128 (`f_b/g_b`, 2^7), 1536 (indexer
`wq_b`), 256 (absorbed `embed_q`), 8192 (`eh_proj`). Vision (dropped): 1024, 4096, 10240 = 20 · 2^9 all compatible;
only `patch_embed` (1176 = 3·2·14·14, conv) is not, and it is not a linear anyway.

Residual-stream rotation folding (`suc/models/rotation_folding.py`) is not part of SUC1, but note: a per-stream
orthogonal R on the 4096 dim commutes with the HC stream mixing (pre/post/comb act on the stream index), and the
unweighted RMSNorm over the flattened 16384 vector is invariant under `I_4 ⊗ R`, so folding would additionally require
`fn' = fn · (I_4 ⊗ Rᵀ)` for both HC modules per layer. Unverified; not needed for one-sided input rotation.

## 8. MLX implementation

**What exists.** mlx-lm (main and 0.31.3) has **no `glm5_next` module**. Reusable pieces:

| Piece | Source | Reuse |
|---|---|---|
| FP8 block dequant in `sanitize` | `MLX:deepseek_v32.py:496-519` | as is (SUC does its own dequant in the store) |
| MoE gate (sigmoid + bias, noaux_tc, groups) + SwitchGLU experts + stacking `experts.#` → `switch_mlp` | `MLX:deepseek_v32.py:295-390, 526-533` | yes; add the `swiglu_limit` clamps (upper-only on gate, symmetric on up) — not present there |
| MLA absorbed attention, `kv_b` split into `embed_q`/`unembed_out` (MultiLinear) | `MLX:deepseek_v32.py:124-270, 534-571` | yes, but with `qk_rope_head_dim = 0` (remove the RoPE branch; head dim 256 instead of 128+64) |
| DSA indexer | `MLX:deepseek_v32.py:57-122` (`glm_moe_dsa.py` only subclasses it) | partly: projections/ReLU/weighting match (`weights·n_h^-0.5·softmax_scale`); **missing**: NoPE (it applies RoPE), k-pool compression (gate + ape softmax pooling), pool top-512 → token expansion, tail append, "first valid key" pool alignment |
| KDA layer: q/k/v `ShortConv1d` with SiLU + state, `f_a/f_b`, `b_proj`, `g_a/g_b`, `o_norm·σ(gate)`, conv-weight `moveaxis(2,1)` sanitize | `MLX:kimi_linear.py:236-385, 540-555` | yes, with a new gate function: `decay = exp(−5·σ(exp(A_log)·(a + dt_bias)))` instead of `compute_g` = `exp(−exp(A_log)·softplus(a + dt_bias))` (`MLXI:gated_delta.py:9-10`); the vectorized gated-delta Metal kernel (`MLXI:gated_delta.py:13-125, 262-280`) takes the decay as input and is reusable. **Layer indexing differs**: Kimi's `kda_layers` are 1-based (`MLX:kimi_linear.py:390-391`), GLM's are 0-based |
| mHC `HyperConnection` + Sinkhorn Metal kernel + collapse/expand | `MLX:deepseek_v41.py:150-264` | yes: same formula as `TF:286-330` (pre σ+eps, post 2σ, softmax+eps, column-first Sinkhorn); replace DSV4's weighted `hc_head` by an unweighted mean |
| Hybrid cache (ArraysCache for conv + recurrent state, KVCache for MLA latent) | `MLX:kimi_linear.py:482-490` | adapt; DSA cache must also hold the indexer packed state (k 128 + gate 128 per token) |

**What must be written.** `suc/models/mlx/glm5_next.py` (or an mlx-lm-style module): ModelArgs reading `text_config`,
KDA attention with the safe gate, NoPE MLA, k-pool indexer, mHC decoder layer, MoE with clamps, final stream mean,
sanitize (drop `layers.45.*` and `model.visual.*`, strip `model.language_model.` prefix, FP8 dequant, stack experts,
split kv_b, rename `q/k/v_conv1d`). A sparse-attention gather kernel is optional: for ≤2051 selected keys a dense
masked SDPA is acceptable at short contexts (as in `TF:1238-1276`).

**Adapter** (`suc/models/adapters.py`, new `Glm5NextAdapter`, `_REGISTRY["glm5_next"]`). The adapter classifies
**original HF names**, which here carry the `model.language_model.` prefix — the base `_L = r"model\.layers\.…"` and
`model\.embed_tokens` patterns do not match, so the adapter overrides them: `_L = r"model\.language_model\.layers\.(?P<layer>\d+)"`,
`model\.language_model\.embed_tokens\.weight`, `model\.language_model\.norm\.weight`. It must (a) choose the role of
`self_attn.o_proj` by layer kind (KDA vs DSA, from `config.text_config.layer_types`), (b) accept `.weight_scale_inv`
siblings as non-linear companions, (c) declare F32 params (`A_log`, `dt_bias`, `hc_*`, `e_score_correction_bias`,
`index_kpool_compress_*`) as known non-linear tensors, (d) exclude `layers.45` (`num_hidden_layers` = 45 → MTP index
is out of range) and `model.visual.*` from validation, (e) expect 288 experts on layers 3–44 and 0 on 0–2, (f) map
norms: `attn` = `input_layernorm` (read by KDA q/k/v/f_a/g_a/b, MLA q_a/kv_a, indexer wk/weights_proj/kpool_gate),
`mlp` = `post_attention_layernorm`. `config.json` nests the text keys under `text_config` — the adapter must read there.

**Test plan.**
1. Tiny random config (hidden 64, 4 layers = KDA, KDA, KDA, DSA, `first_k_dense_replace` 1, 8 experts top-2, hc_mult 4,
   kpool 4, index_topk 8, linear heads 2 × 16) saved as safetensors with the **original naming**, FP8 tensors for the
   same subset as the real checkpoint (`F8_E4M3` + F32 128-block scales, using small blocks only if the tiny dims
   require; better use dims ≥ 128 for FP8 tensors).
2. Adapter `validate()` on the real index (names only, laptop OK) and on the tiny model.
3. Dequant identity: store `load_dequantized` vs a NumPy reference of the formula (bit-exact in F32).
4. Numerical identity of the MLX model vs the transformers 5.18 `Glm5NextForConditionalGeneration` reference on the
   tiny model (requires torch, which is **not** installed in `suc-lm` — use a separate env, CPU): logits max-abs and
   KL, per-module checks for KDA (chunked vs recurrent), indexer top-k sets (exact equality) and mHC Sinkhorn output.
5. Prefill vs step-by-step decode equality (conv/recurrent state, indexer cache, tail pool) over 3× `index_kpool`
   boundary crossings.

## 9. Risks / open questions

1. **Does not fit 128 GB at SUC1 4.2 bpp** (166.8 GB). Running needs ≈ 2.8 bpp on experts (or expert offloading
   from SSD: only 8/288 experts per token, ~8.3 GB of active eligible weights at 4.2 bpp, but random-access reads of
   8 experts × 13.2 MB per layer, up to ~4.4 GB per token over 42 layers on cache misses, would dominate decode time). Also the macOS default GPU wired limit
   (~75 % of RAM) must be raised (`iogpu.wired_limit_mb`), which is a system setting the owner must change.
2. **D8 violation by source format**: FP8-only release; the "full precision" reference is the dequantised FP8
   checkpoint (§4). Must be recorded in result metadata and in the final report.
3. BF16 dequantisation is exact only for power-of-two scales — check on the Studio before choosing BF16 vs F32 fitting.
4. mHC quadruples calibration activation memory (8.6 GB per 512 × 512 copy); reduce `n_calibration` × `seq_len` or
   stream activations to disk. Also 288 experts × 42 layers: rarely-routed experts need enough calibration tokens
   (report min tokens per expert, D8).
5. MTP layer semantics are not implemented in transformers (`TF:1379`) — unverified; dropped.
6. The shipped config uses layer type `deepseek_sparse_attention`; TF's mask mapping only has `indexed_attention` /
   `linear_attention` (`TF:1489-1492`, `TF:1500`), and TFC remaps only `full_attention` (`TFC:171-173`). Running the HF
   reference with this `config.json` may raise `KeyError` — unverified (no torch here); the MLX port and the tiny test
   must map it explicitly.
7. `modules_to_not_convert` uses names without `language_model.`; TFI's matcher is `re.match` from the start or
   `endswith` (`TFI:quantizers/quantizers_utils.py:26-42`), so these entries may not match HF module names when the
   reference is loaded with FP8 kernels. Irrelevant for SUC (dtypes come from headers), relevant for an HF reference run
   (prefer `dequantize=True` or a pre-dequantised BF16 copy for the reference).
8. KDA errors accumulate in the recurrent state over long contexts; quality must be checked at long context, not only
   wikitext windows of 2048.
9. Indexer top-k is discrete: weight error in q_b/q_a (shared `q_resid` path) also perturbs selection; indexer kept
   BF16, but `q_a_proj` is eligible and feeds both — consider protecting it (6.3 M params per layer, cheap).
10. Unsupported/missing MLX ops: none fundamental; needed custom work is the k-pool pooling/selection (argpartition +
    gather), the safe-gate decay function, and NoPE MLA. No FP8 matmul on M1 (irrelevant after transcoding).
11. KV/state memory at runtime is small: 11 DSA layers × (512 latent + 257 indexer) × 2 B ≈ 17 KB/token (2.2 GB at
    128 K), KDA state 34 × 4 MB = 136 MB.
12. Vision tower dropped → text-only evaluation; the model is natively multimodal and text-only quality is what SUC
    measures.

## 10. Summary numbers

```yaml
model: zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
total_params: 321.32e9            # incl. MTP 7.43e9 and vision 0.56e9; main text model 313.33e9
bf16_equiv_gb: 642.65
stored_gb: 328.33                 # FP8 E4M3 block-128 (F32 scale_inv) for experts/MLP/MLA, BF16 rest; = index total_size
eligible_params: 311.77e9         # experts 304.41e9, shared 1.06e9, dense MLP 0.45e9, KDA q/k/v/o 4.56e9, MLA 1.29e9
kept_params: 1.556e9              # embed+head 1.27e9, indexer, router, KDA gates, mHC, convs, norms (MTP/vision dropped)
active_params_per_token: 17.38e9  # 16.74e9 without the embedding lookup
est_suc1_gb_at_4.2bpp: 166.79     # 163.68 eligible + 3.11 kept at stored precision
est_suc1_gb_at_4.2bpp_embed4bit: 164.97
fits_128gb_runtime: no            # needs ~2.8 bpp on eligible weights (or expert offloading)
hadamard_incompatible_in_features: []
effort_estimate_person_days: 14   # adapter+FP8 store 2.5, MLX model (KDA gate, NoPE MLA, k-pool indexer, mHC, cache) 7, tests 4.5
mlx_lm_support: none              # reusable: deepseek_v32 (MoE, MLA, FP8 sanitize), kimi_linear (KDA), deepseek_v41 (mHC)
```

## 11. Implementation (2026-10-03)

- Source: `suc.io.quantized_source` kind `fp8_block_f32` (E4M3 x F32 block multiplier, float32 product rounded once to
  BF16 = transformers' / mlx-lm's own dequantization; not lossless w.r.t. the infinite-precision product, risk 3);
  preflight only rejects non-finite / non-positive scales.
- Adapter `Glm5NextAdapter` (`suc/models/adapters.py`): SUC namespace = checkpoint names with `model.language_model.`
  -> `model.`; MTP layer 45, vision tower and `weight_scale_inv` companions dropped. New roles `kda_q/k/v/o` (eligible),
  `kda_f_a/f_b/g_a/g_b/beta`, `attn_kv_a` (eligible), `attn_kv_b`, `indexer_k` (kept). Eligible 311.59 B params
  (experts 304.41 B); kv_b stays BF16 (run absorbed per head).
- MLX model `suc/models/mlx_glm5_next.py` (mHC, KDA with the vectorized gated-delta kernel, absorbed NoPE MLA, k-pool
  DSA indexer, clamped SwiGLU MoE); checked against transformers 5.18 on a tiny checkpoint with the indexer selection
  active (`scripts/glm5_next_reference_check.py`, Studio torch venv): float32 block outputs <= 1.4e-6 relative, logits
  KL 9e-13; bfloat16 4.4e-2 relative on the last block / KL 1e-3 (rounding order, chaotic tiny random model). Risk 6
  confirmed and handled: the script maps `deepseek_sparse_attention` -> `indexed_attention` for transformers. A torch
  check on the real blocks is pending (needs the download).
- `suc compress` (`experiments/configs/compress_glm5.3_flash.yaml`): `plan.role_budgets` gives the routed experts their
  own greedy pass at exactly 3.0 bpw on average (114.2 GB), the other allocated tensors 4.5 bpw (4.0 GB), fixed part
  1.8 GB -> predicted file 120.0 GB > 111 GB runtime budget. `preflight.expert_streaming: true` accepts it: non-expert
  weights 5.85 GB + KV 0.70 GB (32k tokens + KDA state) + 2 GB activations leave a 102.4 GB expert cache (89.7 % of the
  routed experts). The container stage writes `hot_experts.json` (calibration routing counts).
- Runtime: `suc.runtime.expert_streaming.load_streamed_suc1_model` / `suc generate --expert-budget` (ExpertCache,
  hot experts pinned up to half the budget, segmented LRU for the rest); bit-identical to the resident model on the
  tiny checkpoint; calls whose experts exceed the cache are split into expert groups.
- C++ runtime (2026-10-04): `suc1-generate` runs glm5_next files (cpp/README.md milestone 7: `suc/glm5.hpp`,
  `src/metal/glm5.cpp`), resident or with SSD expert streaming (`--expert-budget`, `--hot-experts auto`); streams and
  logits bit-identical to the Python plan runtime on tiny files (bf16 and float32, 64 MLA heads over 564 positions),
  `tests/unit/test_cpp_glm5.py`. MLX runs the fallback attention's `q Kᵀ` / `P V` and the indexer's `q poolsᵀ` as single
  GEMMs (heads folded into M), whose split-k / fused order the C++ reproduces (`glm_mma`).
