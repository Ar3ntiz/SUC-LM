# Prompt prefill: SUC C++ runtime vs ds4

Runtime 1.4.7 update: [combined SUC before/after measurements](optimization/COMBINED_RESULTS.md) and [prefill policy](optimization/INFERENCE_1.4.7.md). Comparator engines in this document were not rerun during this refactoring; retain the historical paired protocols below. New SUC kernel measurements must not be substituted into an old cross-runtime ratio with different prompts/chunks.

Question: why does ds4 prefill long prompts several times faster than the SUC C++ runtime, while decode is at
parity? What would it take to bring SUC to the same level?

**Starting point.** Enlarged hard suite, Mac Studio M1 Ultra, per-case prefill tok/s from
R:`20261011T055014Z_eval_hard20_ds_tcq` and its siblings, medians:

| Model | Prompts | ds4 GGUF | SUC1 (`1d16b17`) |
|---|---|---|---|
| DeepSeek-V4-Flash | > 4k tokens | IQ2_XXS 224 tok/s | tcq 50 tok/s |
| DeepSeek-V4-Flash | all prompts | 135 tok/s | 88 tok/s |
| GLM-5.3-Flash | > 4k tokens | Q2 117 tok/s | tcq 80 tok/s |
| Qwen3.8-Flash-Next | > 4k tokens | Q4 558 tok/s | side8 150 tok/s |

Decode is at parity: about 24 tok/s in both runtimes.

Per-length detail for the SUC baseline:

| Model | 8k | 16k | 31k |
|---|---|---|---|
| DeepSeek tcq | 67 tok/s | 50 tok/s | 38 tok/s |
| GLM tcq | 94 tok/s | 80 tok/s | 64 tok/s |
| Qwen3.8 side8 | ~150 tok/s | ~149 tok/s | ~138 tok/s |

ds4 is flat across the same lengths: 217–243, 117 and 554–563 tok/s.

Shorter prompts from the same results, prefill tok/s by prompt length, median (ds4 → SUC):

| Model | < 200 tokens | 200–1,000 tokens | ~1.3k tokens (1 case) |
|---|---|---|---|
| DeepSeek | 108 → 86 | 180 → 119 | 237 → 148 |
| GLM | 56 → 65 | 83 → 113 | 118 → 123 |
| Qwen3.8 | 191 → 143 | 337 → 220 | 529 → 214 |

Below 2k tokens GLM is at parity or ahead. DeepSeek trails ds4 by about 1.6x and Qwen3.8 by about 2.5x.

Falling throughput means a cost per token that grows with the context. In SUC that cost is attention. A gap that
stays flat at every length is a cost per token per layer: projections, experts, launches.

## Summary

Mac Studio M1 Ultra; base = `1d16b17`, new = this work. Prefill tok/s on the shared long prompt:

| Model | 8k: base → final | 32k: base → final | ds4 (hard suite) |
|---|---|---|---|
| DeepSeek-V4-Flash tcq | 65.7 → 163.2 (2.48x) | 37.3 → 137.8 (3.69x) | ~220 |
| GLM-5.3-Flash tcq | 94.2 → 105.1 (1.12x) | 63.4 → 93.8 (1.48x) | ~117 |
| Qwen3.8-Flash side8 | 139.7 → 323.4 (2.31x) | 144.2 → 322.9 (2.24x) | ~558 |

Base comes from the profile runs (§3); final comes from the final queue's runs (§5, build `a7d2b0f`).

1. **The main cause was work that still ran token by token on long prompts.**
   - DeepSeek's CSA layers past 2,048 tokens: 7.7 ms per token at 8k, 19.7 ms at 32k.
   - Qwen3.8's full-attention layers: 1.8 ms per token at 8k, 3.2 ms at 32k.
   - GLM's DSA attention, which walked every position.

   ds4 batches all of these over the chunk. They are batched now (§4).
2. **The remaining gap is GPU arithmetic in two places, at roughly ds4's FLOPs.**
   - Routed experts. Our tcq / VQ tiles are 8–16 rows; ds4 uses 32-token dequantized tiles. This costs DeepSeek
     3.1 ms, GLM 3.7 ms and Qwen3.8 0.9 ms per token.
   - The scalar chunk attention, which is the same kernel class as ds4's.

   Host syncs are not a cause: wall minus GPU time is 1–3 %. The exception was Qwen3.8's serial n-gram gathers on a
   cold page cache, up to 1.9 ms per token; these are now parallel.
3. **Rounding.** Batching keeps the decode step's arithmetic where it was cheap; the row-batched launches are
   bit-identical. The new paths are measured against token prefill: on the tiny files, KL ≤ 1e-5 on the M3 and one
   near-tie position at 0.026 on the M1 Ultra. At 2,048 tokens, DeepSeek / GLM batched-vs-token KL is unchanged; Qwen3.8 changed its attention path and continuation agreement (§5).

## 1. How ds4 prefills

Source: the built Studio checkout `~/Projects/ds4` (commit `9139e2a`). An older laptop copy
of ds4 has no Qwen3.8 or GLM-5.3 paths. File references below are
`ds4.c`, `ds4_metal.m` and `metal/*.metal` in that commit.

### Common to all three families

- **Big chunks, layer-major.**
  - Chunk sizes:
    - DeepSeek-V4-Flash: whole prompt up to 4,096 tokens, then 4,096-token chunks (`ds4_prefill_cap_for_prompt`).
    - GLM-5.3: 2,048-token chunks (`glm_graph_indexed_prefill_chunk_tokens`).
    - Qwen3.8: 8,192-token chunks (`qwen4_prefill_chunk_tokens`).
  - Each chunk runs every token through layer 0, then layer 1, and so on.
- **Few dispatches, no per-token work.**
  - Every kernel runs over all chunk rows.
  - There are about 40–75 dispatches per layer per chunk for DeepSeek and about 4 per MoE block.
  - Nothing is launched per token.
- **Host syncs are not the trick.** ds4 commits and waits after every layer on long DeepSeek prompts (one command
  buffer per layer, `metal_graph_prefill_layer_major`). GLM syncs once per chunk, or every 16 layers when it
  flushes. Qwen3.8 syncs once per chunk. The CPU encodes only a few dozen dispatches per layer, so these syncs are
  cheap.
- **f32 activations, half inside the tiles, float accumulation.**
  - Activations between kernels are f32.
  - Matmul and attention tiles convert their operands to half in threadgroup memory and accumulate in float
    (`simdgroup_half8x8` × `simdgroup_float8x8`).
  - KV and latent caches are f16 on Apple.
- **MoE with grouping on the GPU.**
  - A map kernel (`kernel_mul_mm_id_map0`, Qwen: `kernel_qwen4_moe_build_lists`) builds per-expert token lists and a
    compact work list.
  - Expert GEMMs (`kernel_mul_mm_id`: 64 rows × 32 tokens × K 32) dequantize each weight tile into threadgroup
    memory once, then reuse it for 32 tokens. Larger chunks mean more tokens per expert.
  - Gate and up share one activation tile, with the SwiGLU fused into the epilogue (`*_pair_swiglu*`). The mid is f16.
  - The routed MoE takes about 4 dispatches, plus about 6 for the router.
- **No resident-path host sync for routing.** GEMM grids are padded to a worst case, and empty tiles exit early.

### DeepSeek-V4-Flash (CSA / HCA + lightning indexer)

- **Sliding-window, HCA and short CSA layers.** These run a tiled flash attention over all chunk rows:
  `kernel_flash_attn_ext_f16_dk512_dv512` with 8 queries × 64 keys per tile, half Q/K/V and float softmax/O.
  A block map skips fully masked 64-key blocks, so the 128-token window costs O(n · window). HCA and CSA with
  ≤ 512 compressed rows attend densely over the window plus every visible compressed row. This is exact, because
  top-512 of at most 512 rows selects them all.
- **CSA past 512 compressed rows.**
  - Batched indexer scores: a TensorOps or tiled GEMM over (16 tokens × 32 rows), half inputs, then ReLU and the
    head-weighted sum (`kernel_dsv4_indexer_scores_*`).
  - Per-row top-k: an argsort plus merge.
  - The indices are sorted ascending.
  - `kernel_dsv4_indexed_mixed_attention_heads16`: one threadgroup per token and 16 heads, keys staged as half4 in
    threadgroup memory, online softmax.
- **Compressors.** Batched over the chunk: two f16 GEMMs and about 10 pooling, norm, rope and quantize kernels.
- **Hyper-connections.** Fused, row-batched kernels (`kernel_dsv4_hc_split_weighted_sum_norm4`,
  `kernel_dsv4_hc_expand4`).

### GLM-5.3 (KDA + MLA / DSA)

- **MLA stays absorbed on prefill**, as on decode. There is no per-head K/V expansion.
- **Attention runs in latent space** over the f16 latent cache, in slices of 2,048 tokens:
  `kernel_glm_attention_indexed_batch_lora_group8_vec_causal` for dense slices, the `indexed` variant for sparse
  slices.
- **DSA scores**: a tiled GEMM over the pooled keys (`kernel_glm_indexer_scores_tiled`), then a per-row argsort top-k.
  The selected pools are expanded to token ids.
- **KDA is a sequential recurrence**, the same as SUC: conv4 + gates in one kernel, the gated-delta recurrence with
  the state row in registers. It is not a chunked WY form.

### Qwen3.8-Flash-Next (Gated DeltaNet + gated QSA attention)

- **Gated DeltaNet is a sequential scan** (`kernel_qwen4_gdn_scan_r4`, state in registers), as in SUC.
- **Full-attention layers** are batched over all chunk rows:
  - q/k norm + mRoPE in one kernel;
  - block keys over the chunk;
  - block scores by simdgroup MMA (`kernel_qwen4_idx_score_mm`) and a radix top-k per row;
  - `kernel_qwen4_attn_mm`, a flash attention with half Q/K/V and float O, with the sigmoid gate fused.
- **MoE**:
  - `kernel_qwen4_moe_build_lists` (atomics);
  - `kernel_qwen4_moe_mm_mid`: 32 rows × 8·NT tokens, gate and up dequantized to half in threadgroup memory, SwiGLU fused;
  - `..._mm_down`;
  - `kernel_qwen4_moe_reduce`, which folds in the router weights, the shared expert and the hyper-connection residual.

## 2. What SUC did at `1d16b17` (the baseline)

The "fast batched prefill" (report §1c) already had:
- GEMM / MMA projections;
- MMA expert tiles (8–16 rows);
- DeepSeek compressors and attention batched per chunk;
- one host sync per MoE layer (expert tile plan on the host).

Chunks were 1,024 tokens for DeepSeek and GLM, 512 for Qwen3.8.

Differences from ds4 found by reading the code (cpp/src/metal):

1. **DeepSeek: CSA layers fall back to token-by-token attention past 2,048 tokens.**
   - `attn_chunk` used the batched `attn_chunk_fast` only while no row of the chunk had more than `index_topk` (512)
     compressed entries, i.e. up to 4 · 512 = 2,048 tokens.
   - Past that, every CSA layer (21 of 43) ran the whole attention per token: compressor rows, emit, window store,
     indexer projections, rope, Hadamard, FP4, scores, top-k, attention, inverse rope. That is about 15 dispatches
     and barriers per token per layer.
   - This is why SUC DeepSeek throughput falls from 67 to 38 tok/s between 8k and 31k.
2. **Qwen3.8: the full-attention layers ran token by token at every length.**
   - Each token went through copy-in, q/k norm, rope, indexer, block key, scores, top-k, attention, gate and copy-out.
   - That is about 12 dispatches with 4–5 barriers per token in each of the 10 attention layers.
3. **Per-row launches of row-local kernels.**
   - Hyper-connection pre (4 launches per row) and post, routing, MoE output, embedding.
   - Qwen3.8 hyper-mix rms / silu / mix, inject, routing, shared-expert add.
   - For a 1,024-token DeepSeek chunk that is about 12 × 1,024 single-threadgroup launches per layer, about 0.5 M per
     chunk.
   - They were encoded on the CPU while the GPU waited at each per-layer host sync.
4. **GLM: chunk attention and selection scale with the context, not the selection.**
   - `glm_attn_chunk` walked all `p + 1` positions of every row, testing the pool mask twice.
   - The per-row top-k ran as one `q4_index_select` launch per row.
   - The pool scores used a lane-per-head scalar kernel.
5. **Attention arithmetic.**
   - DeepSeek's and GLM's chunk attention (one simdgroup per row and head, `RND = 1` to keep the decode step's
     roundings) recomputed each score in every pass: three times for DeepSeek, twice for GLM.
   - The indexer scores were scalar.

The ranked, measured attribution follows in §3 (Studio profile at 2k / 8k / 32k).

## 3. Measured attribution (Mac Studio M1 Ultra, 2026-10-11/12)

**Protocol.** `scripts/studio_prefill_profile.sh`, sequential runs after `studio_wait_idle.sh`, with LM Studio
models unloaded. Each run is `frontier_measure.py generate --long-prompt N --steps 2` on the shared long speed
prompt. Results are `prof_<model>_<build>_<N>[_stages|_c<chunk>]`; the tables come from
`scripts/prefill_profile_table.py`.

- *base* = commit `1d16b17`, the fast batched prefill before this work.
- *new* = `9547468`: the changes of §4 up to the row-batched launches and the 2-heads-per-simdgroup attention, but
  not yet the lane-parallel exponentials or the 4,096-token chunk defaults.
- Wall tok/s comes from the unprofiled runs.
- The stage profiles (`SUC_*_PROFILE=1`) commit and wait at every stage boundary, so their GPU sums are ≈ 3 % above
  the wall time.
- The base build's DeepSeek stage sums also include its 1,024-token warm-up chunk. They are scaled by N / (N + 1,024)
  below. The new tool resets the profile after the warm-up.

### DeepSeek-V4-Flash tcq

**Prefill tok/s, wall:**

| Build | 2k | 8k | 32k |
|---|---|---|---|
| base | 137.7 ¹ | 65.7 | 37.3 |
| new (chunk 1,024) | 147.5 | 134.5 | 126.2 |
| ds4 IQ2_XXS (hard suite) | ~237 | 217–243 | 218–222 |

¹ From the base stage run; the unprofiled §1c record is 41 tok/s at an older commit.

At 8k the speed-up is 2.05x, at 32k 3.4x. **Chunk sweep at 8k** (new build): 134.5 / 142.8 / 147.9 tok/s at chunks
1,024 / 2,048 / 4,096. The default is now 4,096 (2.25x vs base).

**GPU ms per prompt token by stage, 8,192 tokens:**

| Stage | base | new |
|---|---|---|
| CSA attention token by token (`chunk_attn_seq`, base only) | **7.67** | — |
| routed experts | 3.08 | 3.07 |
| attention block (projections, compressors, indexer, chunk attention, `wo`) | 2.46 | 3.54 |
| … of which chunk attention kernel | — | 1.66 |
| … of which `wo_a` / `wo_b` | 0.35 | 1.09 |
| … of which batched indexer | — | 0.14 |
| shared expert + router | 0.34 | 0.34 |
| hyper-connections | 0.36 | 0.35 |
| **total** | 13.93 | 7.31 |

The base profile has coarser stage labels: its `chunk_attn` covers the batched parts of all layers. The new build
moves the CSA layers into the batched path, which is why its attention block grew while the sequential stage
disappeared.

**At 32k (new build):** experts 3.08, chunk attention kernel 1.98, `wo` 1.00, projections and compressors 0.65,
indexer 0.42 ms/token. The total is 7.83 ms, against 7.31 at 8k: the indexer scores and selection over up to 8k
entries add 0.28 ms, the attention kernel 0.32 ms.

**Ranking of what was left after the fixes, 8k, new build, 7.3 ms/token:**
1. **Routed experts: 3.07 ms (42 %).** These are mixed tcq2 / tcq3 / G4 / native stacks in 8–16-row tiles; the
   trellis decode is paid per tile. At about 4.2 TFLOPS effective this is ~20 % of the M1 Ultra's fp32 peak. ds4's
   IQ2_XXS / Q2_K tiles dequantize once per 32 tokens.
   - Larger chunks fill the tiles better: 4,096 gives +10 % on the whole prefill.
   - The rest is the codec's decode cost. A 16-row tcq tile was 3.5x slower before (cpp/README.md), so this needs a
     new tcq GEMM.
2. **Chunk attention: 1.66 ms (23 %), growing with context.**
   - Already halved with the same bits: two heads per simdgroup, scores kept, and since then the exponentials one per
     lane (M3: 670 → 343 ms on the 64-head dev file). This part is not in the `new` column.
   - A simdgroup-matrix version was slower (§4, negative results).
3. **`wo_a` / `wo_b`: 1.09 ms (15 %).** These are affine 8-bit GEMMs at about 5 TFLOPS. The kernel reaches ~60 % of
   peak on the M3 (`suc-mm-bench`, 4096², 1,024 rows: 2.1 TFLOPS), so little is left here.
4. Everything else: projections and compressors 0.65, shared + router 0.34, hyper-connections 0.35, indexer 0.14 ms.
   Host time is ≈ 3 % (wall vs GPU).

### GLM-5.3-Flash tcq

**Prefill tok/s, wall:**

| Build | 2k | 8k | 32k |
|---|---|---|---|
| base | 122.2 ¹ | 94.2 | 63.4 |
| new (chunk 1,024) | 118.0 | 99.8 | 94.2 |
| ds4 Q2 (hard suite, SSD streaming) | ~118 | 117 | 117 |

¹ From the stage run.

At 32k the speed-up is 1.49x. At 2k the two builds are equal within run-to-run noise.

**Chunk sweep at 8k:** 99.8 / 99.7 / 105.9 tok/s at chunks 1,024 / 2,048 / 4,096. The default is now 4,096.

**Stages, new build, GPU ms/token, 2k / 8k / 32k:**
- routed experts: 3.65 / 3.66 / 3.66
- MLA + DSA attention kernel: 1.33 / 2.67 / 3.01
- KDA projections: 1.17
- shared + router: 0.42
- KDA output norm + o: 0.40
- value / o projection: 0.32
- KDA recurrence: 0.13
- DSA index select: 0.04 at 8k

Base at 8k, for comparison: attention 3.07 ms (one simdgroup per head, walking every position up to `p` with a mask
test) and index select 0.17 ms (one launch per row).

The remaining gap at 8k and beyond is the attention kernel. Every row past 2,048 tokens attends 2,051 latent keys
× 64 heads × 512 dims, scored twice. This is the same scalar kernel class as ds4's `group8_vec`.

### Qwen3.8-Flash-Next side8

**Prefill tok/s, wall:**

| Build | 2k | 8k | 32k |
|---|---|---|---|
| base | — | 139.7 | 144.2 |
| new (chunk 512) | 171.0 ¹ | 206.7 | 239.2 |
| new, chunk 1,024 (8k sweep) | — | 343.7 | — |
| ds4 Q4 (hard suite) | ~529 | 557–563 | 554–556 |

¹ First run after loading the file.

**The wall times hide a host cost.** The GPU time per prompt token of the unprofiled runs is:
- base: 4.36 ms at 8k, 5.72 ms at 32k (stage runs);
- new: 2.82 / 3.03 / 3.12 ms at 2k / 8k / 32k.

The wall time of the new build swings between 2.95 and 5.85 ms per token on identical prompts. The difference is the
n-gram embedding rows, gathered on the host token by token from the 69 GB file. On a cold page cache these are random
SSD reads: up to ~1.9 ms per prompt token at 8k, and 3 ms in the first run.

These gathers now run in parallel across the chunk's tokens (same bits). This change is not in the `new` column; the
final queue below measures it. With a warm cache, chunk 1,024 reaches 343.7 tok/s (2.95 ms wall, 2.80 ms GPU).

**Stages, GPU ms/token, base → new:**

| Stage | 2k | 8k | 32k |
|---|---|---|---|
| full attention (QSA) | 0.79 → 0.28 | 1.80 → 0.50 | 3.16 → 0.59 |
| routed experts | 0.87 | 0.89 | 0.89 |
| linear-attention projections (in / out) | 0.60 | 0.59 | 0.59 |
| hyper-connections (attention + MLP mixers, inject) | 0.50 | 0.50 | 0.50 |
| Gated DeltaNet core | 0.14 | 0.14 | 0.14 |
| attention projections | 0.19 | 0.19 | 0.19 |
| router + shared expert | 0.11 | 0.10 | 0.10 |
| PLE | 0.10 | 0.10 | 0.10 |
| **total** | 3.34 → 2.80 | 4.36 → 3.04 | 5.72 → 3.13 |

Rows that do not change between base and new show a single value.

- Batching the attention removes 1.3 ms at 8k and 2.6 ms at 32k.
- The row-batched hyper-connection launches did not change the GPU time: the per-row launches were concurrent
  dispatches. Their saving is host encode time.
- **What separates the new build from ds4 (≈ 1.8 ms/token):**
  - Routed experts: 0.89 ms, mma lookup tiles of 16 rows.
  - The hyper-connection mixers: 0.50 ms. ds4 folds the HC combine into `kernel_qwen4_moe_reduce`.
  - The linear-attention projections: 0.60 ms.
  - Each of these runs at roughly ds4's FLOPs. The SUC lookup GEMM measured 2.5 TFLOPS on the M3 (4096 × 2560, 512
    rows, `suc-linear-bench`), close to that GPU's peak, so the remaining factor is mostly kernel count and fusion,
    not the codec.

### Ranked causes (all three models)

| # | Cause | Model / length | Measured cost before → after |
|---|---|---|---|
| 1 | CSA layers token by token past 2,048 tokens (no batched indexer) | DeepSeek 8k / 32k | 7.7 / 19.7 → 0.14 / 0.42 ms per token (indexer), attention batched |
| 2 | QSA full attention token by token at every length | Qwen3.8 8k / 32k | 1.80 / 3.16 → 0.50 / 0.59 ms |
| 3 | DSA attention walking every position, per-row top-k launches | GLM 32k | 7.70 + 0.61 → 3.01 + 0.14 ms |
| 4 | Serial n-gram gathers on the host (cold page cache) | Qwen3.8 | up to 1.9 ms → parallel (final queue) |
| 5 | Chunk 1,024 / 512 vs ds4's 4,096 / 8,192 (expert tiles, weight reuse) | all | DeepSeek +10 %, GLM +6 %, Qwen3.8 GPU −8 % |
| 6 | Chunk attention recomputing scores and exponentials | DeepSeek / GLM | M3: 670 → 343 ms, same bits (final queue) |
| — | **Remaining: routed experts** (tcq / VQ tiles of 8–16 rows) | DeepSeek 3.07, GLM 3.66, Qwen3.8 0.89 ms | not changed (codec GEMM work) |
| — | Remaining: scalar chunk attention | DeepSeek 1.7–2.0, GLM 2.7–3.0, Qwen3.8 0.5–0.6 ms | ds4 uses the same kernel class |

**Host-sync hypothesis rejected.** SUC syncs once per MoE layer for the host tile plan. ds4 syncs once per layer on
long DeepSeek prompts. In SUC, wall minus GPU time is 1–3 % for DeepSeek and GLM. The only large host cost was
Qwen3.8's n-gram gather.

## 5. Final measurements (completed 2026-10-12)

`scripts/studio_prefill_final.sh` runs on the Studio with the final build (`a7d2b0f`, the new chunk defaults).

For each model it records:
- the frontier prompt (§1b decode record);
- the short, 2,048-, 8,192- and 32,768-token prompts, each with 256 greedy tokens (prefill tok/s, TTFT, and decode at
  that context);
- the batched-vs-token KL / top-1 at 2,048 tokens (`scripts/prefill_batch_vs_token.py`).

The models are DeepSeek-V4-Flash tcq, GLM-5.3-Flash tcq, Qwen3.8-Flash side8, Qwen3.8-27B side8, and Qwen3-8B / 30B-A3B
final. The script then regenerates report §1b / §1c (with a new long-prompt table) and the README prefill column
(`scripts/cpp_speed_rows.py`) and commits them on the Studio. Its log is `logs/prefill_final.log`.

**Results for the three models compared with ds4** (build `a7d2b0f`, 2026-10-12, `gen_cpp_<model>_long{2k,8k,32k}`):

| Model | 2k tok/s | 8k tok/s | 32k tok/s | vs base (8k / 32k) | ds4 (8k / 32k) | decode ms/tok at 2k / 8k / 32k |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash tcq | 171.4 | 163.2 | 137.8 | 2.48x / 3.69x | 217–243 / 218–222 | 46.7 / 54.3 / 73.1 |
| GLM-5.3-Flash tcq | 116.3 | 105.1 | 93.8 | 1.12x / 1.48x | 117 / 117 | 49.5 / 55.1 / 76.1 |
| Qwen3.8-Flash side8 | 341.2 | 323.4 | 322.9 | 2.31x / 2.24x | 557–563 / 554–556 | 33.8 / 33.0 / 34.2 |

Base is `1d16b17` at the same lengths (§3): DeepSeek 65.7 / 37.3, GLM 94.2 / 63.4, Qwen3.8 139.7 / 144.2 tok/s.

- **Share of ds4's throughput.** SUC now reaches about 70 % / 62 % of ds4 on DeepSeek (8k / 32k), 90 % / 80 % on GLM,
  and 58 % on Qwen3.8.
- **Qwen3.8 n-gram time.** The rows are now gathered in parallel: 0.21 ms per prompt token at 8k, against up to
  1.9 ms before.
- **Batched vs token prefill at 2,048 tokens** (`prefill_batch_vs_token_*`). For DeepSeek and GLM the values are unchanged from the
  2026-10-10 records, because at 2,048 tokens the new paths are bit-identical to the earlier fast prefill: the
  indexer does not select yet, and the row-batched launches and the attention variants give the same bits.
  - DeepSeek: prompt KL mean 0.026, top-1 94.4 %; free-running greedy first differs at step 22 of 256.
  - GLM: KL mean 0.012, top-1 93.8 %.
  - **Qwen3.8 caveat** (its attention path did change at 2k: single pass instead of decode's two passes).
    - Prompt KL mean 0.031 vs 0.034 before; top-1 91.5 %.
    - The last prompt token's top-1 flipped (KL 0.038), so the greedy continuation differs from step 0.
    - Continuation top-1 agreement is 80.6 % (6 mismatches out of 31) vs 90.3 % before (R:`20261012T051524Z_prefill_batch_vs_token_qwen3.8_flash_next_side8_ngram4`
      vs R:`20261010T233323Z_...`).
    - To check next: whether `--prefill exact` attention (two-pass for rows past 1,024 keys) removes the flip.
- **Qwen3.8-27B side8.** 93.0 / 84.8 / 58.1 tok/s at 2k / 8k / 32k (57.1 at 2k before). KL 6e-4, top-1 98.6 %.


## 4. Changes made (cpp/README.md, "Prefill after the ds4 comparison")

All of these stay within SUC's constraints:
- the weights still run from codes, with no full dequantization;
- the layer-major chunks keep D8 streaming of experts;
- the row-batched launches are bit-identical to their per-row form, checked by tests;
- the other changes are measured against token prefill (KL / top-1, report §1c).

1. **DeepSeek: batched lightning indexer.** This applies to rows past `index_topk` entries.
   - Query and weight projections as GEMMs.
   - Rope, Hadamard and FP4 over all rows.
   - Simdgroup-matrix scores (`ds4_index_scores_mm`).
   - Exact per-row top-k as an ascending entry list (`ds4_index_select_rows`, `q4_index_select`'s rule).
   - The existing chunk attention reads that list.

   No context length falls back to token-by-token attention any more. On the tiny files, the batched selection is as
   close to token prefill as the sequential path: KL 1e-5, top-1 100 %.
2. **Qwen3.8: batched gated QSA attention** (`full_attn_core_rows`).
   - Norms and RoPE with per-row positions.
   - Block keys of the chunk in one launch.
   - Block scores and top-k block lists per row.
   - One `sdpa_vector` launch over (heads, rows). Rows past the budget visit only their selected blocks plus the tail.
   - The gate runs over all rows.
3. **Row-batched launches** for every row-local kernel of the DeepSeek and Qwen3.8 chunks. Same bits; the kernels take
   the row from the grid.
4. **GLM: DSA over the selected pools only.**
   - The attention walks each row's pool list instead of every position. Same bits.
   - One top-k launch per chunk instead of one per row.
   - Simdgroup-matrix pool scores.
5. **Chunk attention arithmetic, same bits.**
   - DeepSeek keeps each score in threadgroup memory instead of recomputing it in three passes.
   - DeepSeek and GLM process two heads per simdgroup, so every key / latent row is loaded once for both heads
     (`ds4_attn_chunk_h`, `glm_attn_chunk_h`).
   - The softmax exponentials are computed one per lane over 32-key blocks and combined in key order through shuffles.
   - Dev measurement on the M3 (DeepSeek widths, 64 heads, 4,096 tokens, 2 layers): attention kernel 1,034 → 483 →
     343 ms, logits byte-identical. On the GLM file (64 heads, top-k 2,048) the gain was small: about 2,025 → 1,690
     ms, noisy.
6. **Qwen3.8 host work.**
   - The n-gram row gathers of a chunk run in parallel threads (same bits).
   - The Gated DeltaNet q / k norms of all chunk tokens run in one launch each.
7. **Chunk defaults from the Studio sweep.**
   - 4,096 tokens for DeepSeek and GLM, and 1,024 beyond 8,192-token prompts.
   - 1,024 for qwen4_exp / qwen3_5.
   - Every default is at most the prompt, which also keeps the scratch small for short prompts.

The profile's `new` build (`9547468`) contains items 1–5 without the exponential change. Items 5's exponentials, 6 and
7 are in the final queue's build (`a7d2b0f`).

### Tried and dropped (negative results)

- **Chunks of 4,096 on a 32k prompt.** On the M1 Ultra the 97 GB DeepSeek tcq file ran out of GPU memory
  (`kIOGPUCommandBufferCallbackErrorOutOfMemory`; the record failed). Prompts beyond 8,192 tokens therefore keep
  chunks of 1,024, which ran fine at 32k in the profile.

- **Simdgroup-matrix chunk attention for DeepSeek** (8 heads × 16-key blocks per threadgroup, half Q / K / V, float
  accumulation, two passes with the decode roundings).
  - On the M3 dev file it was **slower**: 1,241 ms (16-key blocks) and 956 ms (8-key blocks), against 478 ms for the
    scalar two-heads-per-simdgroup kernel.
  - It was also less exact vs token prefill: KL max 0.13, top-1 98.3 %.
  - The cause is that with one shared K = V row and 64 heads, the per-block threadgroup barriers and the 2 × 64-step
    k loop dominate. ds4 does not use MMA here either: its indexed CSA kernel is a scalar `simd_sum` kernel with half
    keys.
  - The code was removed.
- **Keeping GLM's chunk-attention scores in threadgroup memory.** This needs 20 KB per threadgroup for 2,051 keys.
  It was no faster than recomputing them (2,208 vs 1,849 ms, same bits), because occupancy drops to one threadgroup
  per core. It stays off by default (`$SUC_GLM_ATTN_KEEP=1`).

## Appendix: profile tables (`scripts/prefill_profile_table.py --top 10`)

The DeepSeek *base* stage columns include that build's 1,024-token warm-up chunk. Scale them by N / (N + 1,024) to
get per prompt token; the text above does this. GLM and Qwen3.8 are unaffected.

| Model | build | 2k tok/s | 8k tok/s | 32k tok/s | results |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash tcq | base | — | 65.7 | 37.3 | R:`20261011T221839Z_prof_ds_base_8192`, R:`20261011T223529Z_prof_ds_base_32768` |
| DeepSeek-V4-Flash tcq | new | 147.5 | 134.5 | 126.2 | R:`20261011T214443Z_prof_ds_new_2048`, R:`20261011T215020Z_prof_ds_new_8192`, R:`20261011T220004Z_prof_ds_new_32768` |
| GLM-5.3-Flash tcq | base | — | 94.2 | 63.4 | R:`20261011T234108Z_prof_glm_base_8192`, R:`20261011T235200Z_prof_glm_base_32768` |
| GLM-5.3-Flash tcq | new | 118.0 | 99.8 | 94.2 | R:`20261011T230246Z_prof_glm_new_2048`, R:`20261011T230856Z_prof_glm_new_8192`, R:`20261011T232036Z_prof_glm_new_32768` |
| Qwen3.8-Flash-Next side8 | base | — | 139.7 | 144.2 | R:`20261012T003335Z_prof_q38_base_8192`, R:`20261012T003852Z_prof_q38_base_32768` |
| Qwen3.8-Flash-Next side8 | new | 171.0 | 206.7 | 239.2 | R:`20261012T001056Z_prof_q38_new_2048`, R:`20261012T001434Z_prof_q38_new_8192`, R:`20261012T002026Z_prof_q38_new_32768` |

Chunk sweep (this build, 8,192 tokens):

- DeepSeek-V4-Flash tcq: default 134.5 tok/s; chunk 2048: 142.8, chunk 4096: 147.9
- GLM-5.3-Flash tcq: default 99.8 tok/s; chunk 2048: 99.7, chunk 4096: 105.9
- Qwen3.8-Flash-Next side8: default 206.7 tok/s; chunk 1024: 343.7, chunk 2048: 200.1, chunk 4096: 338.1

Stage profile, DeepSeek-V4-Flash tcq, 2,048 tokens (GPU ms per token; stage-synced runs):

| stage | base | new |
|---|---|---|
| chunk_attn | 4.638 | 0.000 |
| chunk_experts | 4.629 | 3.068 |
| attn_kernel | 0.000 | 1.159 |
| attn_wo | 0.000 | 1.078 |
| attn_proj_comp | 0.000 | 0.647 |
| chunk_shared_router | 0.507 | 0.335 |
| chunk_hc_post_pre | 0.281 | 0.177 |
| chunk_hc_pre | 0.264 | 0.168 |
| chunk_moe_out_hc_post | 0.022 | 0.013 |
| experts_gate_up | 0.005 | 0.000 |
| total | 10.366 | 6.646 |

Stage profile, DeepSeek-V4-Flash tcq, 8,192 tokens (GPU ms per token; stage-synced runs):

| stage | base | new |
|---|---|---|
| chunk_attn_seq | 8.629 | 0.000 |
| chunk_experts | 3.460 | 3.074 |
| chunk_attn | 2.375 | 0.000 |
| attn_kernel | 0.000 | 1.660 |
| attn_wo | 0.000 | 1.089 |
| attn_proj_comp | 0.000 | 0.647 |
| chunk_attn_wo | 0.397 | 0.000 |
| chunk_shared_router | 0.382 | 0.337 |
| chunk_hc_post_pre | 0.210 | 0.177 |
| chunk_hc_pre | 0.198 | 0.168 |
| total | 15.674 | 7.310 |

Stage profile, DeepSeek-V4-Flash tcq, 32,768 tokens (GPU ms per token; stage-synced runs):

| stage | base | new |
|---|---|---|
| chunk_attn_seq | 20.323 | 0.000 |
| chunk_experts | 3.185 | 3.083 |
| chunk_attn | 2.032 | 0.000 |
| attn_kernel | 0.000 | 1.983 |
| attn_wo | 0.000 | 0.999 |
| attn_proj_comp | 0.000 | 0.647 |
| chunk_attn_wo | 0.458 | 0.000 |
| attn_indexer | 0.000 | 0.421 |
| chunk_shared_router | 0.351 | 0.336 |
| chunk_hc_post_pre | 0.193 | 0.177 |
| total | 26.740 | 7.826 |

Stage profile, GLM-5.3-Flash tcq, 2,048 tokens (GPU ms per token; stage-synced runs):

| stage | base | new |
|---|---|---|
| moe_routed_experts | 3.654 | 3.654 |
| mla_attention | 1.226 | 1.327 |
| kda_proj | 1.167 | 1.166 |
| moe_shared_router | 0.419 | 0.419 |
| kda_onorm_o | 0.401 | 0.401 |
| mla_value_o | 0.320 | 0.319 |
| mlp | 0.158 | 0.158 |
| mla_proj_pools | 0.134 | 0.135 |
| hc_post_pre | 0.129 | 0.129 |
| kda_recurrence | 0.128 | 0.128 |
| total | 7.952 | 8.051 |

Stage profile, GLM-5.3-Flash tcq, 8,192 tokens (GPU ms per token; stage-synced runs):

| stage | base | new |
|---|---|---|
| moe_routed_experts | 3.665 | 3.663 |
| mla_attention | 3.070 | 2.671 |
| kda_proj | 1.166 | 1.166 |
| moe_shared_router | 0.418 | 0.418 |
| kda_onorm_o | 0.401 | 0.401 |
| mla_value_o | 0.320 | 0.320 |
| mla_index_select | 0.174 | 0.035 |
| mlp | 0.157 | 0.157 |
| mla_proj_pools | 0.149 | 0.149 |
| hc_post_pre | 0.129 | 0.129 |
| total | 9.994 | 9.454 |

Stage profile, GLM-5.3-Flash tcq, 32,768 tokens (GPU ms per token; stage-synced runs):

| stage | base | new |
|---|---|---|
| mla_attention | 7.704 | 3.011 |
| moe_routed_experts | 3.659 | 3.661 |
| kda_proj | 1.166 | 1.166 |
| mla_index_select | 0.611 | 0.141 |
| moe_shared_router | 0.418 | 0.418 |
| kda_onorm_o | 0.402 | 0.402 |
| mla_value_o | 0.327 | 0.327 |
| mlp | 0.157 | 0.157 |
| mla_proj_pools | 0.152 | 0.152 |
| hc_post_pre | 0.129 | 0.129 |
| total | 15.074 | 9.914 |

Stage profile, Qwen3.8-Flash-Next side8, 2,048 tokens (GPU ms per token; stage-synced runs):

| stage | base | new |
|---|---|---|
| moe_experts | 0.871 | 0.871 |
| attn_core | 0.790 | 0.278 |
| lin_in_proj | 0.416 | 0.409 |
| inject_hyper_mlp | 0.249 | 0.247 |
| hyper_attn | 0.243 | 0.235 |
| lin_out_proj | 0.181 | 0.180 |
| lin_core | 0.143 | 0.136 |
| attn_in_proj | 0.127 | 0.126 |
| moe_router_shared | 0.110 | 0.103 |
| ple | 0.098 | 0.100 |
| total | 3.335 | 2.795 |

Stage profile, Qwen3.8-Flash-Next side8, 8,192 tokens (GPU ms per token; stage-synced runs):

| stage | base | new |
|---|---|---|
| attn_core | 1.799 | 0.501 |
| moe_experts | 0.886 | 0.886 |
| lin_in_proj | 0.414 | 0.409 |
| inject_hyper_mlp | 0.249 | 0.247 |
| hyper_attn | 0.243 | 0.236 |
| lin_out_proj | 0.180 | 0.180 |
| lin_core | 0.143 | 0.137 |
| attn_in_proj | 0.126 | 0.127 |
| moe_router_shared | 0.110 | 0.103 |
| ple | 0.097 | 0.100 |
| total | 4.356 | 3.036 |

Stage profile, Qwen3.8-Flash-Next side8, 32,768 tokens (GPU ms per token; stage-synced runs):

| stage | base | new |
|---|---|---|
| attn_core | 3.158 | 0.592 |
| moe_experts | 0.886 | 0.886 |
| lin_in_proj | 0.414 | 0.411 |
| inject_hyper_mlp | 0.249 | 0.248 |
| hyper_attn | 0.242 | 0.237 |
| lin_out_proj | 0.180 | 0.180 |
| lin_core | 0.142 | 0.137 |
| attn_in_proj | 0.127 | 0.127 |
| moe_router_shared | 0.110 | 0.103 |
| ple | 0.099 | 0.100 |
| total | 5.717 | 3.132 |

