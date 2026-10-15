# qwen4_exp — Qwen/Qwen3.8-Flash-Next

Architecture note for SUC (adapter, MLX model, encoding plan). Facts come from the pinned checkpoint files; formulas
cite the code that defines them. Shapes marked **(H)** were read from a safetensors header, **(D)** are derived from
`config.json` (and, where stated, from the index `total_size`).

Code references: `modeling:N` / `modular:N` / `configuration:N` = transformers main
`src/transformers/models/qwen4_exp/{modeling,modular,configuration}_qwen4_exp.py` (copy fetched 2026-10-03; the repo
itself ships **no** modeling code). `mlx:qwen3_5.py:N` etc. = ml-explore/mlx-lm main `mlx_lm/models/`. mlx-lm (main
and installed 0.31.3) has **no** `qwen4_exp` module.

## 1. Identity

| Item | Value |
|---|---|
| Repo / revision | `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| `model_type` / architectures | `qwen4_exp` (text sub-config `qwen4_exp_text`) / `Qwen4ExpForConditionalGeneration` |
| transformers version at export | `5.8.0.dev0` |
| Index `total_size` | 359 999 963 128 B (360.0 GB, 335.3 GiB) — reproduced **exactly** by the accounting in §5 |
| Shards / tensors | 131 shards (`model-00001..00131-of-00131.safetensors`), 1 658 tensors |
| Sources consulted | `config.json`, `generation_config.json`, `tokenizer_config.json`, `model.safetensors.index.json`, `README.md`; transformers main modeling/modular/configuration files above; mlx-lm main `qwen3_next.py`, `qwen3_5.py`, `qwen3_5_moe.py` (gated-delta linear attention and Qwen3.5-MoE sanitize) |
| Headers fetched (H) | shards 00001 (global mixer, layer 0 non-expert, all `model.visual.*`), 00002 (`layers.0.mlp.experts.gate_up_proj`), 00005 (PLE buffers + n-gram shards 0–1), 00037 (PLE `conv1d`/`key_proj`/`value_proj`/norms, layer 10 linear attention, n-gram tail), 00041 (layer 11 full attention + indexer + MoE), 00124 (`mtp.fc_hidden`) |
| Derived (D) | `embed_tokens`, `lm_head` ([248320, 2560]); `mtp.fc_embedding` ([2560, 2560]) and the two `mtp.pre_fc_norm_*` (sum of sizes 12 800) are fixed by the `total_size` residual (26 240 000 B left after all other tensors), see §5 |

README (`README.md:45`): "125B with 6B activated, plus 51B n-gram embedding and 4B MTP". Our counts (§5): 125.7 B
main-model weights without n-gram tables, 6.04 B active (excluding the LM head), 51.20 B n-gram, 2.61 B MTP-specific
(3.9 B if the shared embedding + LM head are counted, which matches "4B").

## 2. Architecture summary

| Config key (`text_config`) | Value |
|---|---|
| `num_hidden_layers` | 48 |
| `hidden_size` | 2560; residual stream is **4 × 2560 = 10240** wide (`hc_count` 4) |
| `layer_types` | `[lin, lin, lin, full] × 12`; full attention at layers 3, 7, …, 47 (12 layers), Gated DeltaNet at the other 36 |
| full attention | 24 Q heads, 2 KV heads (GQA 12:1), `head_dim` 256, output gate (q_proj doubled), q/k RMSNorm, QSA sparse indexer |
| linear attention | Gated DeltaNet: 16 QK heads × 128, 48 V heads × 128, conv kernel 4, sigmoid output gate (`output_gate_type`) |
| MoE (every layer) | 512 routed experts, top-10, `moe_intermediate_size` 640; 1 shared expert (640) with sigmoid gate; softmax router, `norm_topk_prob` True (default, `modular:171`); no dense layers |
| hyper-connections | `hc_count` 4 streams, `hc_lowrank` 320 |
| PLE / n-gram | `ple_layer_ids` [2] (one-indexed → layer index **1**), `ngram_size` 3, `heads_per_ngram` 8 → 16 hash heads × 160 dims, ≈20 M rows per head, `split_ngram_parts` 128 |
| norms | RMSNorm eps 1e-6, **(1 + w)** convention; no final `model.norm` (the final hyper-connection mixer replaces it) |
| RoPE | MRoPE interleaved, sections [11, 11, 10], θ 1e7, `partial_rotary_factor` 0.25 → 64 rotary dims of 256 |
| activation | SiLU (SwiGLU MLPs); no clamp |
| vocab | 248 320, untied embedding / LM head |
| context | `max_position_embeddings` 262 144 |
| MTP | 1 layer (full attention + MoE), `mtp_use_dedicated_embeddings` false |
| vision | 27-block ViT, hidden 1152, MLP 4304, patch 16, temporal patch 2, merger → 2560 |

**Hyper-connections (gated residual).** The embedding is replicated into 4 streams, `h = repeat(e, 4)` (`modeling:1476`).
Each decoder layer has two `GatedResidual` modules (`modular:531-559`, `modeling:999-1030`). For stream input
`x ∈ R^{4·2560}`: `x̂ = hc_norm(x)` (RMSNorm per 2560-group, weight 1 + w, `modular:299-310`, `modeling:148-168`);
`m = σ(W_up · silu(W_down x̂ / 4))` (`W_down` [320, 10240], `W_up` [10240, 320]); the sub-layer input is
`u = mean_i(m_i ⊙ x̂_i)` ∈ R^2560 (`modular:553-555`); injection weights `c = 2·σ(W_inj x̂ / 4)` ∈ R^4
(`modular:558`). After the sub-layer `f`: `x ← x + c_i · f(u)` for every stream i (`modular:830-837`). Note the
skip path uses the **un-normalized** `x`. The final `hyper_connection_mixer` (no `W_inj`, `modular:906`) returns `u`,
which feeds `lm_head` directly (`modeling:1489`, `:1660`) — there is no other final norm.

**Decoder layer** (`modular:801-838`): `x += PLE(x, ids)` (layer 1 only) → attn gated residual → linear or full
attention → inject → MLP gated residual → MoE → inject.

**Gated DeltaNet** (`modeling:461-621`, same as Qwen3.5, mlx `qwen3_5.py:87-205`): `qkv = in_proj_qkv(u)` [10240 =
2·2048 + 6144], causal depthwise conv1d (k=4) + SiLU, split q/k (16 heads × 128), v (48 × 128); q, k L2-normalized
(`use_qk_l2norm_in_kernel=True`, `modeling:592,605`), q/k repeated 3× to 48 heads (`:579-581`); `β = σ(in_proj_b u)`,
`g = −exp(A_log) · softplus(in_proj_a u + dt_bias)` (`:575-577`); gated delta rule (chunked / recurrent,
`:259-460`); `o = RMSNormGated(o, z)` with `z = in_proj_z(u)` and gate **σ(z)** (`output_gate_type: sigmoid`,
`modular:313-327`; `RMSNormGated` uses plain `w`, init ones, `modeling:176-192`); `out_proj` [2560, 6144].

**Full attention with QSA indexer** (`modeling:815-897`, `:669-776`). `q_proj` [12288, 2560] is split per head into
query and gate (`:863-866`); q/k RMSNorm on head_dim, partial RoPE on the first 64 dims (`modular:330-365`), SDPA with
scale 256^−0.5, output `o ⊙ σ(gate)` (`:894`), `o_proj`. The indexer (`modular:368-474`): `index_qk_proj` [640, 2560]
→ 4 query heads + 1 key head of 128; q gets `q_layernorm` + RoPE; for each query the visible keys are grouped into
complete blocks of 4 consecutive tokens, block key = RoPE(k_layernorm(mean of the 4 raw keys)) at the block's first
position (`:436-445`); score_b = Σ_h relu(q_h · k_b) / √128 (`:447-450`); top `2048/4 = 512` blocks are selected,
plus the incomplete tail (< 4 tokens) (`:452-458`); the selection becomes an additional mask on the causal mask
(`:461-474`, `:492-497`). **Consequence:** with ≤ 512 complete blocks every block is selected, so for any query that
sees ≤ 2051 tokens QSA is exactly dense causal attention. All SUC calibration (512) and PPL (2048) windows are in
that regime.

**MoE** (`modeling:917-997`): router `softmax(W_g u)` in fp32, top-10, renormalized (`:966-974`), no bias correction,
no group limit, no routed scaling factor. Expert e: `gate, up = chunk(gate_up_proj[e] · u, 2)` — gate rows
0..639, up rows 640..1279 (`:947`); `down_proj[e] · (silu(gate) ⊙ up)`. Shared expert: SwiGLU MLP 640, scaled by
`σ(shared_expert_gate · u)` (`:992`).

**PLE / n-gram embedding** (layer 1 only; `modular:562-783`, `modeling:1076-1252`). Token history (with 2 cached
previous ids, EOS-padded; n-gram context resets after EOS, `modular:643-657`) is hashed for bigrams and trigrams:
`mix = Σ⊕_p (t_{−p} · mult_p)` (xor of products, `modular:691-696`), per head `id = mix mod P_h + offset_h`
(`:699-700`); `P_h` = 16 consecutive primes above 19 999 999 (20 000 003 … 20 000 171, total 320 001 446 rows,
padded to 320 001 536 = 128 × 2 500 012); `mult_p` from a splitmix64 seeded by `seed` 1234 (`:569-585`; `seed` is
not in config.json, so the default applies; products stay < 2^63 by construction, `:577-578`). The 16 looked-up
160-dim rows are concatenated to e ∈ R^2560 (`:708`). Then `k = norm_key(key_proj e)` [10240], `v = value_proj e`
[2560], `q = norm_query(x)` per stream, `gate_i = signed_sqrt(⟨k_i, q_i⟩/√2560)`, `g_i = σ(gate_i) · v`
(`:771-776`); output `g + silu(conv1d_dilated(norm_conv(g)))` with depthwise conv k=4, dilation 3 over 10240 channels
(`:744-761`, `:782`). The 128 `shard_#` tensors are concatenated into one embedding at load (`modular:108-110`).
The 51.2 B-parameter table costs 16 row reads per token and no matmul.

**MTP.** Ignored by transformers (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`, `modeling:1318`) and dropped
by mlx-lm Qwen3.5 sanitize (`mlx:qwen3_5.py:350`). Tensors: `fc_embedding`, `fc_hidden` [2560, 2560], two pre-fc
norms, one full-attention+indexer+MoE layer with two gated residuals and a final mixer. No reference code: the exact
dataflow (and which pre-fc norm is 2560 vs 10240) is **uncertain**.

**Vision.** Qwen3.5-MoE vision tower (`modeling:1688-2015`), 0.449 B params, 27 blocks, `deepstack_visual_indexes` [].

## 3. Checkpoint tensor inventory

Prefix `T = model.language_model`, `L = T.layers.#`. All dtypes BF16 except the three I64 buffers. Count = tensors.

| pattern | count | dtype | shape | role | SUC treatment |
|---|---|---|---|---|---|
| `T.embed_tokens.weight` | 1 | BF16 | [248320, 2560] (D) | embed | affine 4-bit g64 |
| `lm_head.weight` | 1 | BF16 | [248320, 2560] (D) | lm_head (untied) | affine 4-bit g64 |
| `T.hyper_connection_mixer.hc_norm.weight` | 1 | BF16 | [10240] (H) | final mixer norm | keep |
| `T.hyper_connection_mixer.input_mix_weight_{down,up}.weight` | 2 | BF16 | [320,10240] / [10240,320] (H) | final mixer | keep BF16 |
| `L.{attn,mlp}_hyper_connection.hc_norm.weight` | 96 | BF16 | [10240] (H) | hc norm | keep |
| `L.{attn,mlp}_hyper_connection.input_mix_weight_down.weight` | 96 | BF16 | [320, 10240] (H) | hc mixer down | keep BF16 (ablation: eligible) |
| `L.{attn,mlp}_hyper_connection.input_mix_weight_up.weight` | 96 | BF16 | [10240, 320] (H) | hc mixer up | keep BF16 (ablation: eligible) |
| `L.{attn,mlp}_hyper_connection.block_inject_weight.weight` | 96 | BF16 | [4, 10240] (H) | hc injection | keep |
| `L.linear_attn.in_proj_qkv.weight` | 36 | BF16 | [10240, 2560] (H) | linattn_qkv | **eligible** |
| `L.linear_attn.in_proj_z.weight` | 36 | BF16 | [6144, 2560] (H) | linattn_z | **eligible** |
| `L.linear_attn.out_proj.weight` | 36 | BF16 | [2560, 6144] (H) | linattn_out | **eligible** |
| `L.linear_attn.in_proj_{a,b}.weight` | 72 | BF16 | [48, 2560] (H) | linattn_a / linattn_b (decay, β) | keep |
| `L.linear_attn.conv1d.weight` | 36 | BF16 | [10240, 1, 4] (H) | depthwise conv | keep |
| `L.linear_attn.{A_log,dt_bias}` | 72 | BF16 | [48] (H) | SSM params | keep |
| `L.linear_attn.norm.weight` | 36 | BF16 | [128] (H) | gated norm (plain w) | keep |
| `L.self_attn.q_proj.weight` | 12 | BF16 | [12288, 2560] (H) | attn_q (+ output gate) | **eligible** |
| `L.self_attn.{k,v}_proj.weight` | 24 | BF16 | [512, 2560] (H) | attn_k / attn_v | **eligible** |
| `L.self_attn.o_proj.weight` | 12 | BF16 | [2560, 6144] (H) | attn_o | **eligible** |
| `L.self_attn.{q,k}_norm.weight` | 24 | BF16 | [256] (H) | head norm | keep |
| `L.self_attn.indexer.index_qk_proj.weight` | 12 | BF16 | [640, 2560] (H) | indexer_qk | keep BF16 (ablation: eligible) |
| `L.self_attn.indexer.{q,k}_layernorm.weight` | 24 | BF16 | [128] (H) | indexer norm | keep |
| `L.mlp.experts.gate_up_proj` | 48 | BF16 | [512, 1280, 2560] (H) | expert_gate (rows 0..639) + expert_up (640..1279), fused stack | **eligible** (split per expert) |
| `L.mlp.experts.down_proj` | 48 | BF16 | [512, 2560, 640] (H) | expert_down stack | **eligible** |
| `L.mlp.gate.weight` | 48 | BF16 | [512, 2560] (H) | router | keep BF16 |
| `L.mlp.shared_expert.{gate,up}_proj.weight` | 96 | BF16 | [640, 2560] (H) | shared_expert_gate / _up | **eligible** |
| `L.mlp.shared_expert.down_proj.weight` | 48 | BF16 | [2560, 640] (H) | shared_expert_down | **eligible** |
| `L.mlp.shared_expert_gate.weight` | 48 | BF16 | [1, 2560] (H) | shared-expert scalar gate | keep |
| `L.ple.key_proj.weight` | 1 | BF16 | [10240, 2560] (H) | ple_key | **eligible** |
| `L.ple.value_proj.weight` | 1 | BF16 | [2560, 2560] (H) | ple_value | **eligible** |
| `L.ple.conv1d.weight` | 1 | BF16 | [10240, 1, 4] (H) | dilated depthwise conv | keep |
| `L.ple.norm_{key,query,conv}.weight` | 3 | BF16 | [10240] (H) | PLE norms (1 + w) | keep |
| `L.ple.ple_embedding.ngram_embedding.shard_#.weight` | 128 | BF16 | [2500012, 160] (H) | ngram_embed (row-split table) | affine 4-bit (see §6), stored separately |
| `L.ple.ple_embedding.{layer_multipliers, ngram_heads_offsets, ngram_heads_vocab_sizes}` | 3 | I64 | [3] / [16] / [16] (H) | hash constants | keep verbatim (verify vs recomputation) |
| `model.visual.*` (27 blocks × 12 + merger 6 + patch_embed 2 + pos_embed 1) | 333 | BF16 | e.g. qkv [3456,1152], fc1 [4304,1152], merger fc1 [4608,4608], patch [1152,3,2,16,16] (H) | vision | drop (text-only) |
| `mtp.fc_hidden.weight` | 1 | BF16 | [2560, 2560] (H) | MTP | drop |
| `mtp.fc_embedding.weight` | 1 | BF16 | [2560, 2560] (D, residual) | MTP | drop |
| `mtp.pre_fc_norm_{embedding,hidden}.weight` | 2 | BF16 | sizes sum 12 800 → {2560, 10240} (D, assignment uncertain) | MTP | drop |
| `mtp.hyper_connection_mixer.*`, `mtp.layers.0.*` (same patterns as a main full-attention layer) | 22 | BF16 | as main layer (H for mixer down/norm) | MTP | drop |

Layers carrying PLE: index 1 only (`ple_layer_ids` is one-indexed, `modular:91-92`, `:796`). The fused stack is
contiguous per expert on dim 0: expert e of `gate_up_proj` is one 6 553 600 B run, of `down_proj` a 3 276 800 B run,
so `SafetensorsStore.load_rows(name, e, e+1)` (`suc/io/safetensors_store.py:113`) loads one expert without the 3.36 GB stack.

## 4. Stored dtypes and quantization

Every floating-point tensor in the checkpoint is **BF16** (verified in all six fetched headers; no `quantization_config`
in `config.json`, no scale tensors in the index). The only non-float tensors are the three I64 hash buffers of the
PLE layer. D8 ("full-precision source only") is therefore satisfied literally; no dequantization step exists.
`SafetensorsStore` already supports BF16 and I64 (`suc/io/safetensors_store.py:18-31`).

Reference conventions a port must reproduce exactly:
- `Qwen4ExpTextRMSNorm`: `y = x·rsqrt(mean(x²)+eps)·(1 + w)` in fp32 (`modeling:163-168`), optionally per 2560-group
  (`:156-160`) — applies to hc_norm, PLE norms, q/k_norm, indexer norms. mlx-lm's Qwen3.5 sanitize adds 1.0 to a fixed
  list of norm keys only when it detects an un-sanitized conv1d (`mlx:qwen3_5.py:346-368`); that list does not cover
  `hc_norm`, `norm_key/query/conv`, `q/k_layernorm`.
- `RMSNormGated` (linear attention): plain `w`, gate activation sigmoid (`modeling:176-192`, `modular:313-327`);
  mlx-lm's `Qwen3NextRMSNormGated` hard-codes SiLU via `precise_swiglu` (`mlx:qwen3_next.py:58-71`).
- conv1d weights are stored [C, 1, K]; MLX `nn.Conv1d` wants [C, K, 1] (`mlx:qwen3_5.py:363-364` moveaxis).

## 5. Parameter accounting

Arithmetic (H = 2560, HC = 10240):

| Category | Formula | Params | BF16 GB |
|---|---|---|---|
| Routed experts (48 layers) | 48 × 512 × (1280·2560 + 2560·640) = 48 × 2 516 582 400 | 120.796 B | 241.59 |
| Shared experts (48) | 48 × 3 × 640·2560 | 0.236 B | 0.47 |
| Linear-attention linears (36) | 36 × (10240·2560 + 6144·2560 + 2560·6144 + 2·48·2560) | 2.085 B | 4.17 |
| Full-attention linears incl. indexer (12) | 12 × (12288·2560 + 2·512·2560 + 2560·6144 + 640·2560) | 0.617 B | 1.23 |
| Hyper-connections (96 + final mixer) | 96 × (2·320·10240 + 4·10240 + 10240) + (2·320·10240 + 10240) | 0.640 B | 1.28 |
| Router + shared gate (48) | 48 × (512·2560 + 2560) | 0.063 B | 0.13 |
| Embedding + LM head | 2 × 248320·2560 | 1.271 B | 2.54 |
| N-gram tables | 128 × 2 500 012 × 160 | **51.200 B** | **102.40** |
| PLE linears/conv/norms | 10240·2560 + 2560² + 10240·4 + 3·10240 | 0.033 B | 0.07 |
| Small (conv1d, A_log, dt_bias, norms) | — | 0.002 B | 0.003 |
| MTP (1 layer + fc + mixer) | 2.594 B (H-pattern) + 13.12 M (fc_embedding, pre-fc norms; residual) | 2.607 B | 5.21 |
| Vision | 27 × 11.95 M + merger + patch + pos | 0.449 B | 0.90 |
| I64 buffers | 35 values × 8 B | — | 280 B |
| **Total** | | **179 999 981 424** | **359.999 963 128** = index `total_size` exactly |

Stored bytes = BF16-equivalent bytes = 360.0 GB (no quantized tensors). The exact match pins the derived shapes
(`embed_tokens`, `lm_head`, `fc_embedding` [2560, 2560], pre-fc norms summing to 12 800); among the plausible MTP
layouts only `fc_embedding` = `fc_hidden` = [2560, 2560] with one 2560 and one 10240 norm fits.

Active parameters per token (main model): attention/linear-attention 2.70 B (all) + 10 routed experts × 4.92 M × 48 =
2.36 B + shared 0.236 B + hyper-connections 0.640 B + routers 0.063 B + PLE linears 0.033 B + 16 n-gram rows
(2 560 values) = **6.04 B** without the LM head (README: 6 B), **6.67 B** with it. BF16 bytes touched per token:
≈ 13.3 GB; at SUC1 ≈ 2.8 GB of eligible weights + ≈ 1.5 GB BF16 kept tensors (hc mixers dominate) + head.

## 6. SUC encoding plan

**Eligible (SUC1: VQ G=2 K=256 + one-sided RHT input rotation + 0.5 % int8 column residual, ≈ 4.2 bpp):**

| Proposed role | Tensors | in_features | Params |
|---|---|---|---|
| `expert_gate`, `expert_up` (slices of fused `experts.gate_up_proj`), `expert_down` | 48 × 512 × 3 | 2560 / 640 | 120.796 B |
| `shared_expert_gate/up/down` (existing roles) | 48 × 3 | 2560 / 640 | 0.236 B |
| `linattn_qkv`, `linattn_z`, `linattn_out` (new) | 36 × 3 | 2560 / 6144 | 2.076 B |
| `attn_q` (incl. output-gate half), `attn_k`, `attn_v`, `attn_o` | 12 × 4 | 2560 / 6144 | 0.597 B |
| `ple_key`, `ple_value` (new) | 1 + 1 | 2560 | 0.033 B |
| **Total eligible** | | | **123.739 B** |

Expert stacks: use `codebook_scope: expert_stack` (`suc/quant/stacks.py:1-30`) with stack keys derived from the fused
tensor: one stack each for gate, up (row slices 0..639 / 640..1279 of `gate_up_proj`, the split mlx-lm also does,
`mlx:qwen3_5_moe.py:36-50`) and down per layer; gate and up share `seed_in`. Allocation unit = expert stack, as in
`experiments/configs/compress_qwen3_30b_a3b_final.yaml`.

**Kept at BF16 (0.734 B params, 1.47 GB):** routers, `shared_expert_gate`, `in_proj_a/b`, `A_log`, `dt_bias`, conv1d
weights, all norms, `block_inject_weight`, all hyper-connection `input_mix_weight_{down,up}` (0.640 B — 87 % of the kept
budget; they are read on every token by a sigmoid gate, and quantizing them is an ablation, not a default), and the
indexer `index_qk_proj` (19.7 M; kept so that the block selection at long context is not perturbed by quantization;
ablation). I64 buffers verbatim.

**Embeddings:** `embed_tokens` and `lm_head` affine 4-bit g64 (4.5 bpp, 0.72 GB), as for Qwen3.
**N-gram tables (51.2 B):** not a linear; a row-gather table. Options:
- keep BF16 (102.4 GB) memory-mapped, never resident: 16 random rows × 320 B per token;
- affine 4-bit: `mx.quantize` group sizes are 32/64/128 and the row length 160 is not a multiple of 64, so g32
  (5.0 bpp, 32.0 GB) via MLX, or a custom per-row g160 / two-group layout (≈ 4.2 bpp, 26.9 GB) with a hand-written
  gather-dequant (trivial: no matmul). Recommended default: per-row 4-bit (scale + bias per 160-row) stored as a
  separate memory-mapped table; 3-bit or VQ on rows is an ablation (sensitivity is unknown; each row is hit rarely).
  Encode one `shard_#` tensor at a time (800 MB each, D8); never concatenate the 128 shards.

**Implemented (2026-10-03):** `suc compress` stores the shards raw (BF16); `suc repack IN OUT` rewrites a finished SUC1
file with every raw n-gram shard as a row-wise 4-bit affine record (`affine`, `group_size` 160, f16 scale + bias per
row, 4.2 bpp, `params.scheme = "row_minmax"`, docs/format.md §5), copying every other tensor byte for byte, one shard
at a time, straight to the output (disk needed = output size). The runtime (`suc.runtime.suc1_model.load_suc1_model`,
`suc generate`) attaches raw or row-affine shards as mmap views and decodes only the 16 gathered rows per token on the
CPU (`NgramTable` / `RowAffineShard`). Quality effect of the 4-bit tables on the real model: not measured yet.
Since 2026-10-04 `suc compress` writes that same record directly with `ngram_codec: {kind: row_affine, bits: 4}`
(top-level config key; default `raw` keeps earlier fingerprints): the container stage quantizes one shard at a time
from the checkpoint mmap in row chunks (byte-identical table records to compress raw + `suc repack`, unit-tested on
the tiny model) and the evaluate stage attaches the tables from the SUC1 file, so PPL / KL are measured with the tables
as stored. The final config uses it.

**Dropped:** MTP (2.61 B; could be re-added later for speculative decoding), vision (0.449 B; text-only target).

**Estimated SUC1 size (text model, MTP and vision dropped):**

| Component | GB |
|---|---|
| Eligible 123.739 B × 4.2 / 8 | 64.96 |
| Kept BF16 0.734 B × 2 | 1.47 |
| Embedding + LM head, 4-bit g64 | 0.72 |
| N-gram tables BF16 / 4-bit g32 / 4-bit per-row g160 | 102.40 / 32.00 / 26.88 |
| **Total** with n-gram BF16 / g32 / g160 | **169.5 / 99.1 / 94.0** |

GPU-resident working set (n-gram table on CPU / mmap): ≈ 67.2 GB.

**Memory per block during layer streaming (D8):** BF16 weights of one block ≈ 5.19 GB (routed stacks 5.03 GB), but
experts are loaded one at a time (6.6 + 3.3 MB), so weight residency per step is small. The dominant term is
activations: the residual stream is 10 240 wide, so 512 × 512 calibration tokens hold 262 144 × 10 240 × 2 B =
5.37 GB per BF16 copy (input + output of a block ≈ 10.7 GB, fp32 ×2); sub-layer inputs `u` (2560) add 1.3 GB each.
Estimated peak per block ≈ 20–30 GB, comfortably below 128 GB; with `ActivationSpill` (suc/models/streaming.py:133)
it can be reduced further. Layer 1 additionally needs the n-gram lookups for the calibration tokens:
262 144 × 16 row gathers from the mmap (≈ 84 MB of data), not the 102 GB table.

## 7. Hadamard compatibility

`supported_size(n)` requires n = m · 2^k, m ∈ {1, 12, 20, 28} (`suc/quant/hadamard.py:36-45`).

| in_features | used by | factorization | OK |
|---|---|---|---|
| 2560 | expert gate/up, shared gate/up, linattn qkv/z, attn q/k/v, PLE key/value (also the kept indexer, router, in_proj_a/b) | 20 · 2^7 | yes |
| 640 | expert down, shared down | 20 · 2^5 | yes |
| 6144 | linattn out_proj, attn o_proj | 12 · 2^9 | yes |
| 10240 | hc `input_mix_weight_down` (kept; only if the ablation makes it eligible) | 20 · 2^9 | yes |
| 320 | hc `input_mix_weight_up` (kept; ablation) | 20 · 2^4 | yes |
| 1152, 4304, 4608 | vision (dropped) | 9·2^7, 16·269, 9·2^9 | no — irrelevant while vision is dropped |

Every eligible text linear is Hadamard-compatible; no fallback is needed. Note that residual-stream rotation folding
(`suc/models/rotation_folding.py`, `rotation_groups: true`) does **not** apply: the sub-layer input is
`mean_i(m_i ⊙ x̂_i)` with a data-dependent elementwise gate, which is not equivariant under a rotation of the stream,
and there is no plain RMSNorm whose weight could be folded (`NORMS` in `suc/models/adapters.py` has no counterpart).
Use the per-tensor one-sided input rotation only (online RHT of the activation), with `rotation_groups: false`.

## 8. MLX implementation

**What exists.** No mlx-lm module for `qwen4_exp`. Reusable pieces: `gated_delta.py` (`gated_delta_update`,
`normalize_qk`, Metal kernel) and the `GatedDeltaNet` of `mlx:qwen3_5.py:87-205` (same tensor names `in_proj_qkv/z/a/b`,
same GQA 16→48 handling); `SwitchGLU` + Qwen3-Next MoE block with softmax/top-k/renorm and sigmoid shared gate
(`mlx:qwen3_next.py:299-341`); the Qwen3.5-MoE sanitize that renames `model.language_model` → `language_model.model`,
drops `model.visual` and splits `gate_up_proj` into `switch_mlp.{gate,up}_proj` (`mlx:qwen3_5_moe.py:23-52`) and drops
`mtp.` (`mlx:qwen3_5.py:350`).

**What must be written (`qwen4_exp.py`, local module registered for `mlx_lm.utils.load_model`):**
1. `RMSNorm` with (1 + w) and optional per-2560 grouping (fold `+1` in sanitize for every such norm, not only the
   Qwen3.5 key list); `RMSNormGated` with sigmoid gate.
2. `GatedResidual` (hc mixer + injection) and the 4-stream residual; embedding repeat; final mixer → `lm_head`.
3. Full attention: Qwen3-Next gated attention + partial RoPE (64 dims; text-only MRoPE with equal T/H/W positions
   reduces to plain RoPE on 64 dims — to be verified numerically against `modeling:110-140`), plus the QSA indexer.
   v1: dense attention is exact for ≤ 2051 visible tokens (§2) — implement the indexer as a vectorized block-scoring
   + top-512 mask (needed for long-context evaluation), with an indexer key cache (raw keys, 128 per token per layer).
4. PLE: n-gram hashing on CPU/int64 (numpy, wrap-free by construction), EOS-segment shifting, 2-token id cache;
   lookup over 128 separate shard arrays (`row // 2 500 012`, never concatenated — HF's own note: "~95 GiB",
   `modeling:1323-1326`); key/value projections, signed-sqrt gate, dilated conv (k=4, dilation 3) with a 9-step state.
5. Cache: per linear layer conv + recurrent state; layer 1 extra PLE conv state and id context; KV + indexer keys for
   the 12 attention layers.

**Adapter notes (`Qwen4ExpAdapter`, classifies ORIGINAL HF names).** Read structure from `config["text_config"]`
(the top-level config has no `num_hidden_layers`). Layer regex prefix `model\.language_model\.layers\.(?P<layer>\d+)`;
global names `model\.language_model\.embed_tokens\.weight`, `lm_head\.weight`. Expert tensors are 3-D fused stacks
with no `experts.{e}` in the name: the adapter must emit virtual per-expert specs `(name, expert e, row range)` —
`gate_up_proj` → `expert_gate` rows [0, 640), `expert_up` rows [640, 1280); `down_proj` → `expert_down` — which
requires extending `LinearSpec` (slice info) and `validate()` (per-layer expert count = `num_experts` from the stack
shape). New roles: `linattn_qkv`, `linattn_z`, `linattn_out` (eligible), `linattn_a`, `linattn_b`, `indexer_qk`,
`hc_down`, `hc_up`, `hc_inject`, `shared_expert_router` (shared_expert_gate) (not eligible by default), `ple_key`,
`ple_value` (eligible), `ngram_embed` (embedding-like). `expected_roles(layer)` depends on `layer_types[layer]` and on
`layer + 1 ∈ ple_layer_ids`. `NONLINEAR_PATTERNS` must include hc_norm, all PLE/indexer norms, `A_log`, `dt_bias`
(no `.weight` suffix), conv1d, I64 buffers. `model.visual.*` and `mtp.*` are recognised and marked dropped, so
`validate()` still fails on genuinely unknown names. `NORMS` is undefined for this family (rotation folding off).

**Streaming changes.** `suc/models/streaming.py` assumes hidden width = `hidden_size` and passes only hidden states to
a block; here blocks consume and produce 10 240-wide stream tensors and layer 1 also needs token ids (PLE). `embed()`
must repeat ×4 and `head()` must apply the final mixer.

**Test plan.**
1. Tiny random config (hidden 64, hc_count 4, hc_lowrank 8, 4 layers `[lin, lin, lin, full]`, `ple_layer_ids` [2],
   `ngram_vocab_size_base` 1000, `split_ngram_parts` 2, 8 experts top-2, indexer budget 8 / compress 4) built in
   transformers 5.8 (torch, CPU), saved with `save_pretrained`; load in MLX; logits max/mean abs error and KL vs
   torch at seq 16 (dense regime) and seq 64 (indexer selecting), plus incremental decode with cache vs full forward.
2. Hash unit test: recomputed `layer_multipliers`, `ngram_heads_offsets`, `ngram_heads_vocab_sizes` equal the
   checkpoint I64 buffers (read on the Studio from the full checkpoint), and n-gram ids match torch for sequences
   containing EOS.
3. Adapter test on the real index (names only): every tensor classified, counts as §3, expert slices cover the stacks.
4. Identity conversion on the real model through the adapter (bit-exact BF16 round trip per tensor), as for Qwen3.

## 9. Risks / open questions

1. **Size and disk.** 360 GB BF16 download; the Studio needs ≥ 360 GB free plus the SUC output (≈ 95–170 GB) and
   activation spills. Download/hash time is many hours.
2. **N-gram table (102 GB BF16).** Any code path that materializes or concatenates it (HF loader semantics, mlx-lm
   lazy load + sanitize, `tree_map` copies) OOMs or pushes the system into swap. It must stay as 128 mmap-backed
   arrays and be gathered on CPU. Its quantization sensitivity is unknown; it cannot be validated by MSE alone (§48).
3. **128 GB runtime.** Eligible + kept + head ≈ 67 GB GPU-resident fits; adding the n-gram table at 4-bit (27–32 GB)
   gives 94–99 GB, above the default Metal wired limit (~75 % of RAM ≈ 96 GB) — keep the table on the CPU side/mmap.
   KV cache is small (12 layers × 2 KV heads × 256 × 2 × 2 B = 24.6 KB/token + 3 KB indexer keys; 262k tokens ≈
   7.2 GB) and the linear-attention state is constant (≈ 113 MB per sequence in fp32).
4. **End goal (8–32 GB Macs) is out of reach at 4.2 bpp:** 65 GB of experts. Only expert offloading (SSD streaming
   of routed experts, 10 of 512 per layer per token) or ≈ 2 bpp experts (~31 GB) plus an SSD-resident n-gram table
   would approach 32 GB; this is beyond the current SUC1 design.
5. **Hyper-connections.** 4× wider activations in calibration (memory, spill I/O) and no residual-stream rotation
   folding (§7). The hc mixers (0.64 B) are kept BF16 — they cost 1.28 GB, read on every token: ≈ 28 % of the ≈ 4.6 GB of
   weights touched per token at SUC1 (eligible active 2.8 + kept 1.5 + 4-bit head 0.36). Quantizing them is the first
   ablation if decode bandwidth matters.
6. **QSA is never exercised at calibration/eval lengths ≤ 2051 tokens** (exactly dense). Quantization of q/k and the
   indexer could change block selection only at long context: a long-context evaluation (≥ 8k, e.g. PPL on long
   documents) with an MLX indexer is needed before any long-context claim. The HF reference indexer is a Python
   double loop (`modular:424-459`), too slow as a long-context reference beyond a few thousand tokens.
7. **Rarely-routed experts.** 512 experts, top-10: 262 144 calibration tokens give 5 120 tokens per expert on
   average; the minimum may be near zero. Report min tokens per expert; stack-shared codebooks mitigate.
8. **MTP layout uncertain** (no reference code; pre-fc norm assignment inferred from the byte residual). Irrelevant
   while MTP is dropped.
9. **MRoPE.** Text-only use with equal position ids should reduce to plain partial RoPE; the interleaved
   recomposition (`modeling:136-145`) must still be ported exactly if vision is ever enabled.
10. **Norm conventions.** Mixed (1 + w) RMSNorm vs plain-w gated norm; a missed `+1` is silent and catastrophic —
    covered by the tiny-model identity test.
11. **transformers 5.8.0.dev0** is required for the torch reference (new cache API with multiple conv states);
    pin a commit for the reference tests.
12. **`seed` default.** `config.json` has no `seed`; the hash multipliers depend on the default 1234
    (`configuration:156`). Verify against the stored `layer_multipliers` buffer before trusting recomputation.

## 10. Summary numbers

```yaml
model: Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540   # model_type qwen4_exp
total_params: 179999981424           # 180.0 B incl. 51.2 B n-gram, 2.61 B MTP, 0.45 B vision
bf16_equiv_gb: 360.0                 # = index total_size 359 999 963 128 B
stored_gb: 360.0                     # all BF16 (+280 B I64 buffers); no quantized tensors
eligible_params: 123738521600        # experts 120.80 B, shared 0.24 B, linear-attn 2.08 B, attn 0.60 B, PLE 0.03 B
kept_params: 52.29e9                 # 0.734 B BF16 (routers, hc mixers, indexer, small) + 1.27 B embed/head (4-bit) + 51.2 B n-gram; MTP + vision dropped
active_params_per_token: 6.04e9      # 6.67e9 incl. LM head
est_suc1_gb_at_4.2bpp: 169.5         # eligible 64.96 + kept BF16 1.47 + embed/head 4-bit 0.72 + n-gram BF16 102.4
est_suc1_gb_ngram_4bit: 94.0         # n-gram per-row 4-bit g160 (26.9 GB); 99.1 with MLX g32 (32.0 GB)
fits_128gb_runtime: yes              # ~67 GB GPU-resident; n-gram table on CPU/mmap (4-bit or BF16 from SSD)
effort_estimate_person_days: 12      # adapter + fused-stack slices 2, MLX model (hc, PLE/n-gram, QSA, gated norm) 6, streaming changes 2, tests 2
mlx_lm_support: none                 # reuse qwen3_5 GatedDeltaNet, gated_delta kernel, qwen3_next MoE, qwen3_5_moe sanitize
```

## 11. Implementation status (SUC pipeline)

| Piece | Where | Status |
|---|---|---|
| MLX model | `suc/models/mlx_qwen4_exp.py` (registered in `suc.models.streaming.LOCAL_MODELS`) | Gated DeltaNet (mlx-lm kernel, custom VJP through the ops reference), QSA attention + vectorized indexer, gated residual mixers, PLE with CPU n-gram hashing over memory-mapped shards, SwitchGLU MoE + gated shared expert, caches for decode |
| Adapter | `Qwen4ExpAdapter` (`suc/models/adapters.py`) | validates the HF names of all 1 658 real tensors; `source_store` = SUC namespace (`model.` prefix, per-expert views of the fused stacks as byte ranges, vision / MTP dropped) |
| Streaming | `embed_stream` / `head_stream` | blocks are shape-preserving on `[4 streams (10240) | n-gram rows (2560)]` = 12 800 channels; the n-gram rows are gathered from the token ids at embedding time |
| Runtime | `suc.runtime.suc1_model.load_suc1_model` | SucLinear / SucSwitchLinear from the SUC1 codes; n-gram tables attached as zero-copy SUC1 payload views |
| C++ runtime | `cpp/src/metal/qwen4.cpp` (`suc1-generate` on a qwen4_exp file; cpp/README.md milestone 5) | decode graph from the SUC1 codes, bit-identical to the Python plan runtime on tiny files (raw and row-affine tables); n-gram tables outside the GPU views, rows read on the CPU; Qwen3.5 tokenizer regex + Qwen3.8 chat template |
| Config | `experiments/configs/compress_qwen3.8_flash_next.yaml` | uniform VQ G=2 K=256 `_xs` + 0.5 % column residual; embedding / head affine-4; n-gram tables row-affine 4-bit (`ngram_codec`, mmap) |

Validated without torch (`tests/unit/test_qwen4_exp.py`): adapter on the real header layout (eligible 123 738 521 600
params; every MLX parameter of the real config has exactly the shape of its tensor), hash constants equal the real I64
buffers, tiny model (real structure) streamed = full forward bit-exact (fp32 and bf16), cached decode = full forward
(QSA sparse), indexer = port of the HF selection loop, `suc compress` uniform and greedy end to end, SUC1 -> runtime.

**Preflight estimate** (`estimate_resources` on the real layout, final config): peak memory 37.4 GiB (encode; calibrate
20.6, evaluate 22.8, reference 15.7), largest block 4.89 GiB BF16, n-gram tables 95.4 GiB memory-mapped (not counted
as resident). Disk: ≈ 401 GiB new in the run directory (container 315 GiB transient: file + section files of the
streamed writer; encode shards 61 GiB; reference logits 12 GiB), so ≥ 501 GiB free with the 1.25 margin, plus the
335 GiB snapshot. Estimated SUC1 file 169 GB (157 GiB), of which 102.4 GB are the raw n-gram tables. These figures
predate `ngram_codec: row_affine`; with 4-bit tables (26.9 GB) the file is ≈ 94 GB by arithmetic (not re-run).

**Torch reference check (Mac Studio, after the full snapshot is downloaded)** — `scripts/qwen4_exp_reference_check.py`:
the torch half builds the HF `Qwen4ExpTextDecoderLayer` of blocks 0–3 one at a time (meta device, n-gram rows read on
demand, never the 102 GB table) in float32 on 64 seeded tokens with an EOS in the middle; the MLX half runs the same
tokens in float32 (port correctness, expected ≈ 1e-5 relative) and bfloat16 and writes a result JSON with per-block
errors. Blocks 0–3 cover Gated DeltaNet, PLE (layer 1) and QSA attention (layer 3); `--blocks all` adds the logits,
`--seq 4096` the sparse indexer regime.

```bash
cd ~/Projects/SUC-LM
M=$(HF_HUB_OFFLINE=1 .venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.8-Flash-Next', revision='de4b8e4d43b917e7706784d8bb445c9af86a3540'))")
~/suc-local/torch-venv/bin/python scripts/qwen4_exp_reference_check.py torch --model-dir "$M" --blocks 4 --seq 64 --out logs/q4_ref_b4.npz
HF_HUB_OFFLINE=1 .venv/bin/python scripts/qwen4_exp_reference_check.py mlx --model-dir "$M" --ref logs/q4_ref_b4.npz
```

Open: n-gram table quantization (per-row 4-bit needs a gather-dequant path), hc-mixer / indexer quantization
ablations, long-context (≥ 8k) evaluation of the QSA indexer, MTP head for speculative decoding.
