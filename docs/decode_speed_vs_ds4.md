# Batch-1 decode speed: SUC C++ runtime vs ds4

Question: why does ds4 (`~/Projects/ds4` on the Studio, commit `9139e2a`, Metal) decode DeepSeek-V4-Flash and
GLM-5.3-Flash 2–3x faster than the SUC C++ runtime (`suc1-generate`)? The starting numbers are from
[compression_report.md §5](compression_report.md): DeepSeek IQ2_XXS GGUF 40.9 ms/token vs SUC1 V1 `native_max`
81.7 ms/token; GLM Q2 GGUF 41.1 ms/token vs SUC1 135.5 ms/token.

Everything below was measured on the Mac Studio M1 Ultra (128 GB) on 2026-10-07. Runs were sequential with the
machine otherwise idle. The protocol is the frontier short prompt: the same sentence, 256 greedy tokens, batch 1,
decode median of steps 2…n.

## Summary

- **This is a GPU-time problem, not host overhead.** SUC V1 spent 79.7 ms of GPU time per token and 2.5 ms on the
  host. ds4 spends 36.5 ms of GPU time and about 4.6 ms on the host. Both runtimes keep only one token in flight.
- **Both read about the same bytes per token.** SUC reads ≈ 10.8 GB per token, ds4 ≈ 9.5–9.8 GB. The gap comes from
  how fast those bytes are read: SUC achieved 136 GB/s before the fix and 155 GB/s after it; ds4 achieves 260–270 GB/s.
- **The single largest cause was routing.** The `ds4_route` top-k (6 of 256 experts) ran in one thread and cost
  12.6 ms per token (0.29 ms × 43 layers).
  - It is now a parallel threadgroup top-k with the same selection rule (fixed in this pass).
  - The same serial scan existed in `glm_route` and was fixed the same way.
- **Results after the fix:**
  - DeepSeek V1: **82.3 → 72.0 ms/token** (GPU 79.7 → 69.4 ms).
  - GLM streamed: **154.9 → 132.6 ms/token** (GPU 90.4 → 67.5 ms).
  - Logits are bit-identical to before (byte-identical `--logits-out` over 75 positions on the real V1 file). The
    tiny-model cross-checks against the Python runtime pass (54 passed, 7 skipped).
- **Second pass (§7): DeepSeek V1 71.2 → 51.4 ms/token** (GPU 68.6 → 49.0 ms), bit-identical (V1 and GLM logits
  byte-identical, tiny-model cross-checks pass). Fixes 1–4 below plus the hyper-connection / MoE phase merges; fix 5
  (fused attention) is not done. The rest of this summary describes the state before §7.
- **Defaults (§8): DeepSeek V1 49.2 → 39.1 ms/token** (GPU 47.0 → 36.8 ms), GLM streamed 127.2 → 114.6 ms/token
  (GPU 60.6 → 46.9 ms): expert codebooks in threadgroup memory per the device's tuned `codebook_in`, 2 rows per
  simdgroup in the expert vec lookup, 4 row simdgroups in the affine gemv. Logits byte-identical on all seven C++
  model files checked (prompt + 256 greedy steps).
- **Remaining DeepSeek gap (before §7): ≈ 33 ms of GPU time.** It is ranked in the [Ranked causes](#ranked-causes-deepseek-v1-after-the-route-fix)
  section. In short:
  - mixed expert stacks running at 110 GB/s (≈ 14 ms recoverable);
  - the element-wise affine 8-bit gemv at 150–180 GB/s (≈ 13 ms);
  - `wo_a` decoded to bf16, which doubles its bytes (≈ 3 ms);
  - hyper-connection kernels that are latency-bound (≈ 7 ms);
  - unfused attention (≈ 3 ms).

  The estimates assume about 300–350 GB/s for the bandwidth-bound parts. That is what our own dense bf16 gemv already
  reaches (371 GB/s on `wo_a`), and ds4 reaches 260–270 GB/s across its whole decode step.
- **GLM is not a like-for-like comparison.** ds4's GLM Q2 file (96.5 GB) fits in memory: ds4 maps 89.9 GiB of it plus
  a 16 GB expert cache, so its "streamed" run is effectively resident. The SUC1 GLM file (119.9 GB) has to stream, and
  at a 91.1 % hit rate it waits 53.5 ms per token on SSD reads.

## 1. Per-token breakdown, DeepSeek-V4-Flash

| | SUC1 V1 `native_max` (before) | SUC1 V1 (after route fix) | ds4, IQ2_XXS GGUF (resident) |
|---|---|---|---|
| decode wall, median | 82.26 ms | 71.98 ms (record run: 72.16) | 40.86 ms |
| GPU time per token | 79.69 ms | 69.42 ms | 36.5 ms (34.8–37.9 per window of 64 CBs) |
| host: encode / commit, wait / read | 1.58 ms encode, ≈ 1 ms wait + argmax readback | same | 6.8 ms encode (layers 0–3 already run on the GPU meanwhile), 0.04 ms read; ≈ 4.6 ms host share overall |
| command buffers per token | 1 (one concurrent-dispatch encoder, memory barriers) | 1 | 2 (3 once compressed entries exist): a split after layer 4 overlaps encoding and execution |
| host syncs per token | 1 | 1 | 1 (after the last CB; CPU argmax over the full logits) |
| dispatches per token (static count) | ≈ 3,200 (≈ 74 per layer) | ≈ 3,200 | ≈ 1,200–1,400 (≈ 26–37 per layer) |
| weight bytes read per token | ≈ 10.8 GB | ≈ 10.8 GB | ≈ 9.5 GB (9.8 GB incl. the indexer `q_b`) |
| effective bandwidth (bytes / GPU time) | 136 GB/s | 155 GB/s | 260–270 GB/s |
| tokens to tokens | — | identical to before | — |

Notes on the table:

- **Bytes per token.**
  - SUC counts every non-expert tensor as stored, except `wo_a`, which is counted at its decoded bf16 size (see
    below).
  - The 6 routed experts per layer are counted at the mean expert size of the file. The file has ≈ 30 % native MXFP4
    experts at 4.25 bpw, plus VQ (2.16 bpw) and additive (3.16 bpw) experts, each with int8 column residuals and an
    input rotation; the mean is 9.16 MB per expert across gate, up and down.
  - Embedding rows and `tid2eid` are excluded.
  - ds4: every non-expert tensor plus 6/256 of each expert stack, from the GGUF tensor table. The indexer `q_b`
    (352 MB) is listed separately because neither runtime needs it before the context exceeds `index_topk`.
- **Dispatch counts** are static. SUC's comes from the encode path in `src/metal/deepseek_v4.cpp` (see §3). ds4's is
  from reading its decode graph.
- **ds4 GPU time** is `DS4_METAL_GPU_BUSY_PROFILE` (Σ GPUEndTime − GPUStartTime), mapped to tokens through the
  interleaved `DS4_METAL_GRAPH_TOKEN_PROFILE` lines. Logs: `logs/speed/ds4_token.stderr` on the Studio.

Per token, the gap is therefore:

- ≈ 33 ms of GPU execution;
- −2 ms of host time: SUC's host is cheaper, because its encode is 1.6 ms against ds4's 6.8 ms;
- +1.3 GB of extra reads.

Overlapping several tokens is not what makes ds4 fast. It never has more than one token in flight. It does overlap
encoding with execution inside a token, but SUC's encode is already short.

## 2. SUC GPU time per stage (DeepSeek V1)

### How the profile was taken

`SUC_DS4_PROFILE=1 suc1-generate …` is a diagnostic mode added in this pass. At every stage boundary it commits the
command buffer and waits, and it reports the summed `GPUEndTime − GPUStartTime` of each stage. It runs the same kernels
in the same order: generated tokens are identical to the normal run.

Because stages no longer overlap, the profile adds about 1,000 command-buffer boundaries per token. Each one costs
≈ 8 µs of GPU time at minimum. The profiled total is therefore 75.7 ms against 69.4 ms unprofiled, and the per-stage
numbers are slightly high for the tiny stages.

### Results after the route fix

Run: `logs/speed/route_check.log`. MB = weight bytes read per token by the stage.

| stage | GPU ms/token | % | MB/token | GB/s | what runs |
|---|---|---|---|---|---|
| experts gate + up | 14.03 | 18.5 | ≈ 1,576 | 112 | 2 mixed stacks: per VQ / additive sub-stack input rotation + lookup kernel + int8 column residual, native MXFP4 sub-stack, select |
| `wo_b` (affine 8-bit) | 10.44 | 13.8 | 1,533 | 147 | `suc_gemv_affine` |
| `wo_a` (decoded bf16, 8 group gemvs) | 7.77 | 10.3 | 2,886 | 371 | dense gemv on the bf16 copy of an affine 8-bit tensor (1,533 MB as stored) |
| experts down | 7.25 | 9.6 | ≈ 788 | 109 | as gate / up |
| wq_a + wkv + compressor rows | 5.32 | 7.0 | 901 | 169 | affine gemv + float32 gemv over bf16 compressor weights |
| wq_b + rope | 5.02 | 6.6 | 1,424 | 284 | affine 8-bit (37 layers), VQ (6) |
| hc: Sinkhorn coefficients | 4.48 | 5.9 | 0 | latency | one thread, 20 Sinkhorn iterations, 87 calls |
| shared expert gate + up | 4.31 | 5.7 | 767 | 178 | 2 affine 8-bit gemvs |
| attention core | 3.92 | 5.2 | small | latency | scores (steel GEMM + split-k), softmax, p·V (GEMM), inverse rope: 6 launches |
| shared expert down | 2.51 | 3.3 | 383 | 153 | affine 8-bit gemv |
| head (hc + norm + LM head + argmax) | 1.92 | 2.5 | 298 | 155 | affine 4-bit gemv |
| hc: mix gemv (24 × 16384, float32) | 1.75 | 2.3 | 135 | 77 | MLX gemv order, 87 calls |
| hc: input reduce | 1.30 | 1.7 | — | latency | one threadgroup over 16,384 values |
| compressor emit + kv store + q norm/rope | 1.31 | 1.7 | — | latency | |
| hc: collapse + RMS norm | 1.07 | 1.4 | — | latency | |
| **router top-k (`ds4_route`)** | **0.91** (was **12.62**) | 1.2 | — | latency | parallel top-k (was one thread) |
| moe out + hc post, hc post (attn) | 1.00 | 1.3 | — | | |
| router gemv + shared SwiGLU | 0.59 | 0.8 | 90 | | |
| q / kv norm, expert SwiGLU | 0.74 | 1.0 | — | | |
| **total** | **75.66** (unprofiled 69.42) | | ≈ 10,800 | | |

Before the fix the same profile (`logs/speed/suc_prof2.log`) was identical except for router top-k at 12.62 ms per
token. Total GPU time was 87.4 ms profiled and 79.7 ms unprofiled.

### ds4's own stage profiler

`DS4_METAL_DECODE_STAGE_PROFILE` is not comparable with the table above:

- it reports wall time including encode and a sync at every stage;
- it turns off ds4's fused paths.

It sums to 164 ms per token against a 41 ms real decode. The log is kept as `logs/speed/ds4_stage.stderr`, but it is
not used for any number in this document.

## 3. Hot kernels: ds4 vs SUC

This part is static: it comes from reading the code (ds4 source as on the laptop; the kernel and flag names were
checked to exist in the Studio commit).

| | ds4 (IQ2_XXS / Q2_K / Q8_0 GGUF) | SUC C++ (V1 `native_max`) |
|---|---|---|
| expert gate + up | **one launch** for all 6 experts. `kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` computes gate and up, SiLU, clamp and the route weight. Grid (2048/8, 1, 6), 64 threads, 8 rows per threadgroup; the IQ2 grid is copied to threadgroup memory (2.1 KB) and dequantized in registers; `simd_sum` reduction | per projection **8 launches** separated by barriers. Each of the 2 VQ / additive sub-stacks runs an input Hadamard rotation (per-expert signs), a lookup gemv into float32 and an int8 column-residual add. The native MXFP4 sub-stack is one launch, and a select launch picks each row's sub-stack. Every sub-stack's grid covers all 6 rows; rows routed elsewhere exit at the `_tab` prologue. SwiGLU is a separate launch. |
| expert down | **one launch** (`kernel_mul_mv_id_q2_K_sum6_f32`) that sums the 6 experts | 8 launches as above, then `ds4_moe_out` (sum + shared) |
| shared expert | `shared_gate_up_swiglu_q8_0` (1 launch); down fused with the routed sum and HC post (`shared_down_hc_expand_q8_0`) | 3 affine gemvs + SwiGLU + sum: 5 launches |
| 8-bit dense matvec | Q8_0 `kernel_mul_mv_q8_0_f32`: blocks of 32 int8 + one f16 scale, vector loads | `suc_gemv_affine`. It decodes **each element separately** (`suc_m_affine` per index: bit extraction, a scale and a bias load per weight), rounds to the activation dtype, then runs MLX's dense-gemv accumulation order. This is how the C++ stays bit-identical to the Python reference, which dequantizes affine modules before a dense gemv. 147–178 GB/s. |
| `wo_a` (grouped low-rank output) | `attention_output_low_q8` on the Q8 weight | decoded at load to **bf16** (+2.9 GB of copied memory), then 8 per-group dense gemvs |
| hyper-connections | 3 launches. The fused `hc_split_weighted_sum_norm` does Sinkhorn, the weighted sum and the RMSNorm; the hc fn is f16 | 4 launches + RMSNorm: one-threadgroup reduce, float32 24 × 16384 gemv, **one-thread** Sinkhorn, collapse. ≈ 100 µs per call, 87 calls per token. |
| router | `matmul_plain` + `router_select` on the GPU, no readback | gemv + `ds4_route`: was one thread (12.6 ms per token), now a parallel top-k (0.9 ms) |
| attention (head_dim 512) | one fused kernel for the indexed layers (`kernel_dsv4_indexed_mixed_attention_heads8_rb16`: half4 K / Q in threadgroup memory, f32 softmax); flash-attn vec f16 for the raw layers | MLX's unfused fallback (`ds4_mma` scores with split-k, `ds4_mma_out`, softmax, `ds4_mma` p·V, `ds4_mma_out`) + inverse rope |
| activations | float32 | bf16 (float32 for the router, hc and compressors) |
| encoder | serial (default) encoder, no explicit barriers | `MTLDispatchTypeConcurrent` + `memoryBarrier` between dependent launches |
| `simdgroup_matrix` at batch 1 | not used | not used in the decode gemvs (the lookup kernel has an MMA variant for batches) |
| residency | `MTLResidencySet` over the mapped views, `madvise(WILLNEED)` | GPU views of the mapping in an `MTLResidencySet` on the queue (default, `SUC_RESIDENCY_SET=0` disables): same |

## Ranked causes (DeepSeek V1, after the route fix)

The GPU gap is 69.4 vs 36.5 ms. Gains are estimates of GPU ms per token.

1. **Mixed expert stacks: 21.6 ms for ≈ 2.36 GB (110 GB/s).**
   - Fix:
     - one launch per projection over the 6 routed rows, with per-row codec dispatch inside the kernel. This removes
       the empty threadgroups, the select pass and 2 of 3 grids;
     - one input rotation per routed row instead of one per sub-stack;
     - the column residual folded into the lookup kernel's epilogue;
     - gate + up + SwiGLU (× route weight) in one launch, and down + the 6-row sum in one launch (ds4's pair_swiglu and
       sum6 structure).
   - The elementwise residual, select and sum orders can be kept, so the result stays bit-identical.
   - Expected at 300 GB/s: 7.9 ms, a gain of **≈ 13–14 ms**.
2. **Element-wise affine 8-bit gemv: ≈ 26 ms at 150–180 GB/s.** This covers `wo_b`, `wq_b`, `wq_a`, `wkv`, the shared
   expert and the head.
   - Fix: decode one `uint32` (4 weights at 8 bits) per load, and load scale and bias once per group. Keep the thread's
     4-element step and MLX's dense-gemv accumulation order, which keeps the result bit-identical. Fuse shared
     gate + up + SwiGLU.
   - Expected at 350 GB/s: 13 ms, a gain of **≈ 13 ms**.
   - Alternative: MLX's qmv order. It is faster still, but the Python reference would then need the same change.
3. **Hyper-connections: 10.2 ms of mostly latency-bound launches** (Sinkhorn 4.5, mix gemv 1.75, reduce 1.3,
   collapse + norm 1.1, post 1.0).
   - Fix:
     - run Sinkhorn on 16 threads, one per matrix element, with the same per-element operation order (bit-identical);
     - fuse the reduce, mix gemv and coefficients into one launch per call;
     - fuse the collapse into the RMSNorm.
   - Expected: **≈ 6–7 ms**.
4. **`wo_a` read as decoded bf16: 2.89 GB instead of 1.53 GB.**
   - Fix: a grouped (block-diagonal) affine kernel built on fix 2.
   - Expected: 7.8 → ≈ 4.4 ms, a gain of **≈ 3.4 ms**. It also frees 2.9 GB of copied execution memory.
5. **Attention core + small attention ops: ≈ 5 ms in ≈ 15 launches per layer.**
   - Fix: a fused head_dim-512 decode attention over the window ring and the compressed entries (ds4: one kernel).
   - Expected: **≈ 3 ms**.
6. **Float32 compressor gemvs: 608 MB at ≈ 170 GB/s.**
   - Fix: a bf16-weight gemv tuned like `wo_a`'s dense path.
   - Expected: **≈ 1.5 ms**.

Fixes 1–6 add up to ≈ 40 ms. That is an upper bound, because the stages overlap a little in the unprofiled run. It
would bring V1 to ds4's ≈ 36 ms of GPU time.

### Ruled out

- **Host overhead:** SUC 2.5 ms vs ds4 4.6 ms per token.
- **Multi-token pipelining:** neither runtime does it.
- **File size:** the GGUF reads 9–12 % fewer bytes, worth ≈ 4 ms at ds4's bandwidth.

### Launch count

≈ 3,200 vs ≈ 1,300 launches per token. This is not a separate item: it is the mechanism behind causes 1, 3 and 5,
where the concurrent encoder serialises at a barrier between almost every pair of launches.

## 4. Fix done in this pass: parallel router top-k

`ds4_route` (DeepSeek, 256 experts, top 6) and `glm_route` (GLM, top-k of `σ(logits) + bias`) used to compute the top k
in thread 0. That thread ran K rounds over all E experts, with a nested "already taken" test and a device load of the
bias in every iteration. On the M1 Ultra this cost 0.29 ms per MoE layer.

`ds4_topk_parallel` (in `cpp/metal/suc_ds4_kernels.metal`) replaces it:

- the score + bias values go into threadgroup memory;
- K rounds of a threadgroup argmax run over the untaken entries;
- the argmax is a simd shuffle tree, then the simdgroup winners are combined in index order;
- the comparison order is "larger value first, ties by the lower index".

For non-NaN scores this is exactly what the sequential `if (best < 0 || b > bv)` scan selects, because that scan keeps
the first maximum. Hash-routed layers and the softmax scoring path are unchanged.

### Verification

- **Tiny-model cross-checks:** `tests/unit/test_cpp_deepseek_v4.py` and `tests/unit/test_cpp_glm5.py` pass on the
  Studio: 54 passed, 7 skipped. They compare the C++ streams and logits bit for bit against the Python plan runtime,
  resident and streamed.
- **Real V1 file:** `--logits-out` over 11 prompt + 64 generated positions, written by the binary before the fix
  (`~/suc-local/suc1-generate.oldroute`) and after it: `cmp` reports the two files byte-identical (38.3 MB each). The
  256 generated tokens are identical as well.

### Speed

| | before | after |
|---|---|---|
| DeepSeek V1 resident, decode ms/token | 82.26 (record: 81.75) | 71.98 (record: 72.16) |
| DeepSeek V1, GPU ms/token | 79.7 | 69.4 |
| DeepSeek V1, router top-k GPU ms/token | 12.62 | 0.91 |
| GLM streamed, decode ms/token | 154.93 | 132.64 |
| GLM streamed, GPU ms/token | 90.4 | 67.5 |

The GLM rows use the `gen_cpp_glm5.3_flash_streamed` protocol (`--expert-budget 95G --hot-experts auto`). They are not
the report's `suc perf` ctx-128 number (135.5 ms).

Record JSONs:

- R:`20261007T080648Z_gen_cpp_deepseek_v4_flash_native_max` (commit `2ac7ab6`)
- R:`20261007T080807Z_gen_cpp_glm5.3_flash_streamed`
- before: R:`20261006T212714Z_gen_cpp_deepseek_v4_flash_native_max`
- before: R:`20261007T004109Z_gen_cpp_glm5.3_flash_streamed`

## 5. GLM-5.3-Flash

After the fix, SUC's GLM decode is 132.6 ms/token, of which 67.5 ms is GPU time. The rest is host time (129 ms in the
"encode" column):

- **42 host syncs per token.** The command buffer is split at every MoE layer, so the routed expert ids can be read
  back and fetched.
- **SSD waits:** 13.7 s over 256 tokens, i.e. 53.5 ms per token, at a 91.1 % expert hit rate.
- **SSD reads:** 275 MB per token.

ds4's GLM run is "SSD streaming" in name only:

- the Q2 file is 96.5 GB, so ds4 maps 89.87 GiB of it globally (page cache) with a 16 GB expert cache
  (`logs/gguf/glm_q2/serve_perf.log`);
- in practice the experts are memory-resident.

The SUC1 GLM file is 119.9 GB, above the 111 GB budget, so it has to stream. The same-footing comparison would need a
SUC1 GLM file of ≤ ≈ 100 GB, run resident. On the GPU side, the DeepSeek fixes 1–3 apply unchanged to the GLM graph:
it uses the same mixed stacks, the same affine gemv and the same hyper-connection kernels.

## 6. Method and reproduction

1. **Build.** On the Studio, `cpp/build.sh ~/suc-local/cpp-build` (commits `c72773d`, `6d1c083`, `2ac7ab6`).
2. **SUC.** Use the real V1 file
   `experiments/artifacts/compress/compress_deepseek_v4_flash_native_max/container/model.suc1`:

   ```bash
   suc1-generate FILE --prompt "The history of the printing press begins in the fifteenth century" --steps 256
   ```

   - Add `SUC_DS4_PROFILE=1` for the stage profile.
   - The per-token GPU and encode medians are the `GPU` / `encode` fields of the normal stderr line.
   - Scripts: `logs/speed/suc_prof.sh`, `suc_prof2.sh`, `route_check.sh`.
3. **ds4.** Use `ds4_score serve` (the `gguf_compare.py` short-prompt runner, the same HF prompt ids, 256 tokens,
   1 warm-up + 1 run) with:
   - `DS4_METAL_GRAPH_TOKEN_PROFILE=1 DS4_METAL_GPU_BUSY_PROFILE=1` for encode / execute / read per token and the GPU
     busy time per command buffer;
   - `DS4_METAL_DECODE_STAGE_PROFILE=1` for the stage wall times (not used, see §2);
   - no profiling flags for the plain run: 40.86 ms/token, the same as the report.

   Script: `logs/speed/ds4_prof.sh` / `ds4_prof.py`. ds4 itself is not modified.
4. **Bytes per token.** SUC: `suc1-inspect --tensors` (`logs/speed/v1_tensors.txt`). GGUF: the tensor table of the
   file (offsets → bytes), as in `scripts/gguf_compare.py gguf_inventory`.

### Caveats

- Single runs.
- The Metal System Trace (xctrace) is not available: the Studio has only the Command Line Tools. There are therefore no
  per-dispatch GPU timestamps for either runtime, and dispatch counts are static.
- Expected gains are estimates from bytes and the bandwidths noted above, not measurements.

## 7. Second pass: fixes 1–4 and the low-occupancy audit

Dev build `~/suc-dev/cpp-build` on the Studio (branch commit "cpp: DeepSeek-V4 / GLM decode speed, bit-identical …"),
same protocol as above (V1 resident, the frontier prompt, 256 tokens; the stage profile with `SUC_DS4_PROFILE=1`).
Single runs, machine idle (a Qwen3.8 compress / eval queue ran before and after; runs were placed between).

| | before (route fix only) | after |
|---|---|---|
| DeepSeek V1 decode, median ms/token | 71.16 | **51.36** |
| GPU ms/token (unprofiled) | 68.63 | 48.96 |
| profiled GPU total | 75.22 | 55.75 |
| copied execution arrays | 6,645 MB | 3,759 MB (`wo_a` no longer decoded to bf16) |
| V1 logits (`--logits-out`, 11 prompt + 64 generated positions) | reference | byte-identical (`cmp`) |
| GLM streamed logits (32 steps, `--expert-budget 95G --hot-experts auto`) | reference | byte-identical |
| tiny-model C++ ↔ Python cross-checks (`test_cpp_deepseek_v4.py`, `test_cpp_glm5.py`) | 54 passed, 7 skipped | 54 passed, 7 skipped |

The record runs of this build (`scripts/frontier_measure.py generate`, V1 and GLM streamed) are queued on the Studio
behind the Qwen3.8 evaluation; their JSONs are committed by the queue (see the commit that adds them).

### Per fix (profiled GPU ms/token, before → after)

The profile commits at every stage boundary, so it measures each stage alone; merged phases inside a stage show their
saving, overlap across stages does not (the unprofiled total gains more than the profile).

| fix | stages | before | after | Δ |
|---|---|---|---|---|
| 2. affine gemv, one `uint32` (4 × 8-bit weights), one scale and one bias load per 4 elements, 4 steps of loads issued before the accumulation (same per-element ops and order) | `wo_b`, `wq_b`, `wq_a + wkv + comp`, shared gate / up / down, head | 10.40 + 4.99 + 5.29 + 4.27 + 2.53 + 1.92 = 29.40 | 5.80 + 4.02 + 3.25 + 2.98 + 1.70 + 1.31 = 19.06 | −10.3 |
| 4. `wo_a` per group on the affine codes in place (fix-2 kernel on row slices) | `attn_wo_a` | 7.74 | 4.41 | −3.3 |
| 1. mixed expert stacks phase-grouped: the stages (input rotation, lookup + native MXFP4 gather, column residual) of all sub-stacks of gate **and** up share one barrier each (12 → 3 serial phases for gate + up, 6 → 3 for down); each SUC sub-stack's column residual skips rows of other sub-stacks (`ds4_switch_colres_masked`) and writes straight into the output, so the select launch and the `ysub` copy are gone | experts gate + up, down | 13.92 + 7.19 = 21.11 | 12.58 + 6.16 = 18.74 | −2.4 |
| 3. hyper-connections: Sinkhorn on one simdgroup (one lane per comb element, row / column sums over `simd_shuffle` in the sequential order), the mixer gemv reads the stream directly (concurrent with the sum of squares), the collapse computes its own pre coefficients and is fused with the RMSNorm (concurrent with the coefficients): 5 → 2 phases per call, 87 calls | `hc_*` | 4.48 + 1.05 + 1.30 + 1.74 = 8.57 | 1.58 + 1.12 + 1.25 + 1.71 = 5.66 | −2.9 (+ overlap) |
| MoE / attention phase merges: router gemv and compressor rows straight from `xn` (no float32 copy + barrier), routing beside the shared SwiGLU, shared down concurrent with the routed experts' first stage (GLM: same) | router, compressor, shared down | — | — | in the rows above / overlap |
| 5. fused attention | `attn_core` | 3.89 | 3.63 | not done |

Remaining per-stage leaders: experts 18.7 ms (≈ 126 GB/s: the lookup kernels are gather-bound, not bandwidth-bound),
`wo_b` 5.8 ms (264 GB/s), `wo_a` 4.4, `wq_b` 4.0, attention core 3.6.

Why fix 1 gained less than estimated: the phases were not the main cost. The VQ / additive lookup kernels themselves run
at ≈ 120 GB/s of codes at batch 1 (codebook gathers per 2–4 weights); one launch per projection would need a new lookup
kernel with per-row codec dispatch. Two bit-identical knobs of the existing kernel were added for measurement (the
codebook source and rows / simdgroups per threadgroup do not change any row's arithmetic; tiny-model cross-checks pass
with them set): `SUC_SWITCH_CBTG=1` (expert codebooks staged in threadgroup memory, as the dense lookup does on this
GPU), `SUC_SWITCH_ROWS`, `SUC_SWITCH_SG`, and `SUC_AFFINE_SG` (row simdgroups per threadgroup of the affine gemv). The
sweep over them and the new defaults are in §8.

### Bit-identity argument per kernel

- `suc_gemv_affine` fast path: per element `t = q · s`, `t = t + b`, `m = float(T(t))`, `acc += m · x[j]`, j in the
  same order per thread; only the loads differ. The generic path is kept for other shapes.
- `ds4_switch_colres_masked`: the generated `suc_switch_structured_residual` body plus an early return for rows whose
  expert is an empty slot of the sub-stack (the offset-table marker the `_tab` lookup already uses).
- `ds4_sinkhorn_lane` / `ds4_hc_coef_par` / `glm_hc_coef_par`: each lane forms every sum it needs from shuffled copies
  of the same values, sequentially from 0.0f, under `fp contract(off)`.
- `ds4_gemv_xf`: `suc_moe_gemv` with `float(x[j])` read from the stream (the value its float32 copy holds); the GLM
  variant multiplies by the same `rsqrt(ss / W + eps)` as `glm_hc_scale`, with the scale computed without contraction.
- `ds4_hc_collapse_rms`: the collapse expressions, then `suc_rms_norm`'s body (D ≤ 4096, one threadgroup) on the
  register values.

### Single-threadgroup / low-occupancy audit (DeepSeek and GLM graphs, after this pass)

No launch runs on a single thread any more. Launches that still use **one threadgroup**:

| kernel | graph | threads | why it stays |
|---|---|---|---|
| `ds4_hc_in` (sum of squares of the 4 × 4096 stream) | both | 1024 | MLX row-reduce order (TPG = 1024); now concurrent with the mixer gemv |
| `ds4_hc_coef_par` / `glm_hc_coef_par` | both | 32 (one simdgroup) | 4 × 4 Sinkhorn; one lane per element; concurrent with the collapse |
| `ds4_hc_collapse_rms` (and `suc_rms_norm` with rows = 1: q / kv norms, head) | both | 1024 (D = 4096) | MLX `rms_single_row` order for one row |
| `ds4_route` / `glm_route` | both | E rounded to 32 (256 / 288) | parallel top-k since the first pass; the softmax-scored path (`FUNC 2`, not used by these files) and the K = 6 weight normalisation stay in thread 0 |
| `q4_index_select` | both (indexer) | 1024 | exact threshold search; only when the context exceeds `index_topk` |
| `glm_layer_norm` (indexer key) | GLM | ≤ 1024 | MLX `layer_norm_single_row` |
| `suc_argmax` | both (head) | 1024 | ≈ 129k logits once per token (strided, coalesced) |
| `ds4_gemv_xf` (hc mixer gemv, 24 rows) | both | 24 threadgroups × 256 | MLX gemv order fixes the per-row split; low occupancy but not one threadgroup |

## 8. Lookup / gemv knob defaults (bit-identical)

The four knobs of §7 only change where the expert codebooks are read from and how rows are blocked into simdgroups and
threadgroups; every output row is computed by one simdgroup with the same loads, products and summation order. Sweep
on V1 (`logs/speed/speed2_sweep.log`, `~/suc-local` build of 046c06d, 128 steps, median ms/token, single runs; SG =
simdgroups per threadgroup, rows = rows per simdgroup of `suc_switch_lookup_vec`):

| expert codebooks | rows 4, SG 8 | rows 4, SG 4 | rows 2, SG 8 | rows 2, SG 4 | rows 1, SG 8 | rows 1, SG 4 |
|---|---|---|---|---|---|---|
| device (old default; rows 4, SG 8) | 49.11 | 47.20 | 47.17 | 45.77 | 47.77 | 47.34 |
| threadgroup (`CBTG=1`) | 45.62 | 44.52 | **41.88** | 42.77 | 42.87 | 44.47 |

Affine gemv row simdgroups per threadgroup (`SUC_AFFINE_SG`, rest default): 1 (old default) 49.16, 2 46.81,
**4 45.63**, 8 46.37.

**New defaults** (`cpp/src/metal/switch.cpp`, `deepseek_v4.cpp`, `glm5.cpp`): expert codebooks staged in threadgroup
memory when the device's tuned `codebook_in` for that M is "threadgroup" (the per-device choice the dense lookup
already uses, from `kernel_tuning.json` written by `suc.kernels.autotune`; on the M1 Ultra every M is "threadgroup",
on the M3 "device" was faster for the dense lookup) and they fit in 32 KB; 2 rows per simdgroup (rows 2 beat rows 4 with
either codebook source); SG 8 (= `LOOKUP_SG`, unchanged); affine gemv 4 row simdgroups. Env overrides are kept:
`SUC_SWITCH_CBTG=0|1`, `SUC_SWITCH_ROWS`, `SUC_SWITCH_SG`, `SUC_AFFINE_SG`.

**Identity** (`~/suc-dev/logs/ident.sh` on the Studio, dev build of this commit vs the `~/suc-local` build of 046c06d;
same prompt as the records, `--logits-out` over the prompt + 256 greedy steps, `cmp`): byte-identical for every file
and combination below, so the greedy tokens are identical too. Tiny-model C++ ↔ Python cross-checks
(`test_cpp_moe`, `test_cpp_qwen4`, `test_cpp_deepseek_v4`, `test_cpp_glm5`, `test_cpp_expert_stream`,
`test_cpp_generate`, `test_cpp_metal`): 142 passed, 7 skipped with the new defaults; 96 passed, 7 skipped with
`SUC_SWITCH_CBTG=1` forced.

| file | old default (ms/token, GPU) | switch knobs only (CBTG 1, rows 2, SG 8) | affine SG 4 only | new defaults (both) | logits vs old |
|---|---|---|---|---|---|
| DeepSeek V1 native_max | 49.23 (46.98) | 42.44 (40.20) | 45.93 (43.57) | **39.05 (36.76)** | identical (all 4) |
| GLM-5.3-Flash streamed, 95 GB budget | 127.21 (60.64) | 116.55 (50.41) | 124.07 (59.40) | **114.56 (46.86)** | identical (all 4) |
| Qwen3.8-Flash-Next side8 (ngram4) | 35.03 (26.70) | — | — | 27.66 (25.93) | identical |
| Qwen3.8-Flash-Next iso (ngram4) | 35.41 (27.05) | — | — | 28.27 (26.53) | identical |
| Qwen3-8B final (dense: no knob applies) | 14.05 (13.01) | — | — | 13.99 (12.94) | identical |
| Qwen3-30B-A3B final | 14.93 (13.36) | — | — | 14.60 (13.04) | identical |

Single runs with `--logits-out` (not records). The Qwen3.8 wall-clock drop is mostly page-cache order, not the knobs:
the first (old-build) run read the 4-bit n-gram tables from SSD (n-gram host 6.3 ms/token), the second found them
cached (0.07 ms); the knob effect there is the GPU column (−0.8 / −0.5 ms). The new build with the old values forced by
env (`CBTG=0 ROWS=4 AFFINE_SG=1`) reproduces the old build's logits and speed on V1 (49.44 ms) and GLM. The DeepSeek
V3 file (native experts only, no SUC expert stacks) is covered by the affine knob, whose identity V1 and GLM show.
Records of all C++ models with these defaults: `scripts/studio_cpp_speed_defaults.sh` (report §1b,
`scripts/cpp_speed_rows.py`).
