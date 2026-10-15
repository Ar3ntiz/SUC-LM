# SUC — Final report

> **Current numbers** (2026-10-14): [compression_report.md §1](compression_report.md#1-summary). Speeds in this draft
> predate the tcq re-encodes, the fast batched prefill and the decode passes of 2026-10-07/12.

Semantic Universal Compression: codebook compression of LLM weights with sensitivity-aware allocation,
activation-selected residuals, and Metal kernels that run inference directly on the compressed representation.

Status of this document: **draft, third pass**, with the speed results replaced by the matched-harness `suc perf` runs
(§3.7a). All earlier placeholders are filled (final Qwen3-30B-A3B files and their C++ runs, low-G VQ decode speed,
Qwen3-0.6B on the M3, QA side-by-side). What is still missing is listed in [§8](#8-remaining-gaps).

**Corrected finding.** The earlier C++-vs-MLX speed comparison was wrong: MLX was timed with synchronous per-token
decode (the e2e harness), SUC C++ with pipelined decode. The resulting claims ("22% faster" on Qwen3-30B-A3B,
"parity" on Qwen3-8B, "within 3.4%" on Qwen3-0.6B) are withdrawn. In the matched harness (`suc perf`, same tool,
prompt and contexts, one session, MLX pipelined like mlx-lm) SUC C++ decodes 6% faster than MLX 4-bit only on the
uniform Qwen3-30B-A3B file at batch 1, reaches 75% of MLX 4-bit on Qwen3-8B, and is slower in prefill everywhere (§3.7a).

**How numbers are cited.** `R:<id>` means `experiments/results/<id>.json` (and the `.md` next to it when there is one);
`B:<section>` means the section of that name in [benchmarks.md](benchmarks.md). Every reported timing and memory figure
was measured on the **Mac Studio M1 Ultra, 128 GB**, unless it is marked *laptop* (Apple M3, 16 GB, indicative only).
Quality protocol, unless stated otherwise: WikiText-2 **test** split in full (299,078 tokens, 145 non-overlapping
windows of 2048), KL against BF16 on 16 held-out validation sequences × 512 tokens of the Phase 2 corpus. Calibration
uses a disjoint 512 × 512-token mix (natural / code / math / instruction). bpp is the **actual** whole-model bpp: indices, codebooks, scales,
residual values and positions, metadata, embedding and LM head included. BF16 PPL: Qwen3-8B 9.717, Qwen3-30B-A3B
8.697, Qwen3-0.6B 19.96 in the Phase 0 harness and 20.973 in the `suc compress` / `suc quantize` protocol used in §3.12
(each SUC row is compared with the BF16 value of its own protocol).

---

## Contents

1. [Summary](#1-summary)
2. [Method as built](#2-method-as-built)
3. [Results](#3-results)
4. [Answers to the spec §50 questions](#4-answers-to-the-spec-50-questions)
5. [Success level reached, and what was not done](#5-success-level-reached-and-what-was-not-done)
6. [How to use](#6-how-to-use)
7. [Future work](#7-future-work)
8. [Remaining gaps](#8-remaining-gaps)

---

## 1. Summary

SUC replaces every large linear layer of a decoder-only transformer with codebook indices (one or several small
codebooks, ≤ 16 KB each), per-row scales and a small residual of exact int8 entries selected by activation
statistics. Weights are rotated by a randomized Hadamard transform before coding. The best configuration found uses
**one low-dimensional codebook per tensor** (VQ with 2-dimensional groups from ≈ 2.95 to 4.2 bpp, 4-dimensional at
≈ 2.5 bpp), not the additive multi-codebooks the project started with. A Fisher-guided greedy allocator
can assign a different codec per tensor (or per MoE expert stack). The result is stored in a memory-mappable SUC1
file. Two runtimes execute it directly on the GPU without rebuilding the BF16 weights: the Python/MLX runtime and a
standalone C++ runtime. The pipeline (`suc compress`) runs unattended from a full-precision Hugging Face checkpoint.

### 1.1 Headline results (Qwen3-8B, dense)

| What | Result | Baseline | Source |
|---|---|---|---|
| **Final configuration, end to end** (`suc compress`, every stage layer-streamed) | **VQ G=2 K=256** + 0.5% activation column residual, one-sided rotation, embedding/LM head affine 4-bit: **4.221 bpp, SUC1 file 4,322.2 MB (4122 MiB), PPL 9.994 (+2.9%), KL 0.0494**; 1361 s on the Mac Studio | MLX affine 4-bit g64: 4.50 bpp, 4,607.7 MB, PPL 10.082 (+3.8%); RTN 4-bit KL 0.092 | R:20261002T191301Z_compress_qwen3_8b_final, R:20260930T223113Z_phase0_target_qwen3_8b, R:20261001T033522Z_q1q4_codecs_qwen3_8b |
| **Decode of that file, B = 1, ctx 128 (matched harness `suc perf`)** | **C++ `suc1-generate`: 13.89 ms/token (72.0 tok/s), 75% of MLX 4-bit** (separate projections 13.40 ms, 78%); Python runtime (pipelined) 13.41 ms; prefill C++ 340 tok/s | MLX 4-bit **10.44 ms/token (95.8 tok/s)**, prefill 588 tok/s; BF16 27.75 ms (36.0 tok/s) | R:20261003T015734Z_perf_qwen3_8b_final |
| ≈ 3.4 bpp | VQ G=2 K=64 + res, 3.37 bpp: PPL 10.618 (+9.3%), KL 0.124 | additive M=3 + res at 3.38: 10.692 / 0.153; greedy 3.38: 10.78 / 0.141; RTN 3-bit at 3.50: 13.36 / 0.370 | R:20261002T175941Z_q5c_lowdim_vq_qwen3_8b, R:20261002T115608Z_pareto_allocate_qwen3_8b, R:20261001T033522Z_q1q4_codecs_qwen3_8b |
| ≈ 3.0 bpp | VQ G=2 K=32 + res, 2.95 bpp: PPL 11.560 (+19.0%), KL 0.234 | greedy 3.0 at 3.00 bpp: 11.71 / 0.244; no 3-bit baseline at ≤ 3.0 bpp | same |
| ≈ 2.5 bpp | VQ G=4 K=256 + res, 2.53 bpp: PPL 14.43 (+48.5%), KL 0.430 (Q5c); 14.15 / 0.429 in the Pareto run | RTN 2-bit at 2.50 bpp: PPL 1631 (collapse) | R:20261002T175941Z_q5c_lowdim_vq_qwen3_8b, R:20261002T115608Z_pareto_allocate_qwen3_8b, R:20261001T033522Z_q1q4_codecs_qwen3_8b |
| Previous default (additive) | additive M=4 K=256 G=8 + 0.5% column residual: 4.225 bpp, PPL 10.121 / 10.081 (seeds 1 / 2), KL 0.068 / 0.071 | same | R:20261002T075009Z_abl_base_qwen3_8b, R:20261002T133705Z_abl_repeat_qwen3_8b |
| Weights in memory (decode) | final file: 4.32 GB mapped, process resident peak 0.28 GB beyond the mapping (C++). Python runtime: VQ G=4 2.28 GiB; M=4 + residual 4.03 GiB | MLX 4-bit 4.29 GiB, BF16 15.26 GiB | R:20261002T191814Z_cpp_generate_check, R:20261002T050049Z_e2e_r5_after_qwen3_8b, R:20261002T133705Z_e2e_abl_default_qwen3_8b |
| Decode tok/s, B = 1 / 4 / 16 (Python runtime, e2e harness) | VQ G=4 (2.0 bpw): **65.0 / 139.2 / 262.3**; VQ G=4 + col. residual: 55.7 / 110.0 / 233.3 | MLX 4-bit: 74.4 / 137.6 / 193.0 | R:20261002T050049Z_e2e_r5_after_qwen3_8b |
| | additive M=4 + col. residual (previous default): 34.6 / 85.9 / 141.4 | MLX 4-bit: 74.9 / 137.7 / 194.4 | R:20261002T133705Z_e2e_abl_default_qwen3_8b |
| | VQ G=2 K=256 + col. residual (final codec): 55.2 / 113.1 / 165.5; VQ G=2 K=64 + col. residual (≈ 3.2 bpw): 49.4 / 85.1 / 107.2 | MLX 4-bit, same run: 74.4 / 137.8 / 194.3 | R:20261002T212251Z_e2e_lowg_qwen3_8b |

Caveats on the table:

- **Quality rows are single-seed.** Seed-to-seed noise of the additive configuration is ≈ ±3% in KL and ±1% in PPL
  (B:Mandatory ablations). The final compress run reproduces the ablation's VQ G=2 numbers exactly (PPL 9.994, KL
  0.0494 in both R:20261002T085834Z_abl_codebook_qwen3_8b and R:20261002T191301Z_compress_qwen3_8b_final), but with
  the same seed: this checks determinism and the SUC1 round trip, not seed repeatability. The gain of VQ G=2 over
  additive M=4 (−27% KL) and of G=2 K=64 over M=3 (−19% KL) are well outside the noise. The gain over MLX 4-bit in
  ΔPPL (2.9% vs 3.8%) is close to the PPL noise; the KL gain (0.049 vs 0.092 for RTN 4-bit) is not.
- The two VQ G=4 + res rows at 2.53 bpp differ by 2% in PPL (14.43 Q5c vs 14.15 Pareto run), more than the ±1%
  seed noise of the additive codec. The cause was not investigated; the conclusion at 2.5 bpp (G=4 beats G=2 K=16 and
  RTN 2-bit) does not depend on it.
- **The C++ runtime is 25% slower than MLX 4-bit at B = 1 on the dense 8B** (matched harness, §3.7a): 0.75× at
  ctx 128, 0.76–0.77× at 512–2048, 0.93× at 8192; prefill 0.34–0.58× of MLX 4-bit. The earlier "parity" (13.3 vs
  13.3 ms/token) compared pipelined C++ with synchronous MLX decode and is withdrawn. The C++ runtime decodes batch 1 only.
- The Python runtime is slower than MLX 4-bit at B = 1 in every configuration. Matched harness, final file: 0.78× at
  B = 1, 0.90× at B = 4, 0.97× at B = 16 (ctx 128; §3.7a). Older figures (synchronous decode for both): 19.9 ms/token
  on the final file (cpp check harness), 74% of MLX 4-bit's rate for the same codec in the e2e harness (55.2 vs 74.4 tok/s, same run);
  best 98% for VQ G=2 K=16 and 96% for VQ G=4 without residuals (codecs that are unusable without residuals). At
  B = 16 single-codebook VQ G=4 is faster than MLX 4-bit (+35%); the final codec VQ G=2 K=256 + residual is slower at
  every batch (74% / 82% / 85% of MLX 4-bit at B = 1 / 4 / 16; R:20261002T212251Z_e2e_lowg_qwen3_8b).
- e2e decode speeds in the Python harness use timing-oriented codes (fast k-means, magnitude-selected residuals of the
  same size). Kernel cost depends only on shapes and counts, so they apply to the quality-track encodings of the same
  codec. Each speed run checked exactness against the same weights decoded to BF16 (logit KL 0.001–0.004). The C++
  timing above runs the actual final file.

### 1.2 Headline results (Qwen3-30B-A3B, MoE)

| What | Result | Baseline | Source |
|---|---|---|---|
| **Final file, uniform** (`suc compress`, VQ G=2 K=256 shared per expert stack + 0.5% act. column residual, one-sided rotation, embedding/LM head affine 4-bit, every stage streamed) | **4.164 bpp (file 4.174), 15,892.7 MB model / 15,193.9 MiB file, PPL 9.141 (+5.1%), KL 0.0644**; 1994 s | MLX affine 4-bit: 4.50 bpp, 17.18 GB, PPL 9.062 (+4.2%, Phase 0); RTN 4-bit (same quality protocol): 4.51 bpp, PPL 9.071 (+4.3%), KL 0.081 | R:20261002T212215Z_compress_qwen3_30b_a3b_final, R:20261001T005202Z_phase0_target_qwen3_30b_a3b, R:20261002T070150Z_q7_allocate_qwen3_30b_a3b |
| **Final file, Fisher-greedy 4.0** (336 units: one per attention tensor or expert stack; 110 tensors kept BF16, all attention since an expert stack is one 128-tensor unit) | **4.015 bpp (file 4.025), 15,323.7 MB model / 14,649.7 MiB file, PPL 9.105 (+4.7%), KL 0.0640**; 5443 s (3497 s of them candidate scoring) | same | R:20261002T204858Z_compress_qwen3_30b_a3b_final_greedy |
| **Decode of those files, B = 1 (matched harness `suc perf`)** | **C++ uniform: 12.60 ms/token (79.4 tok/s) at ctx 128, 14.14 ms at ctx 2048 — 6% / 4% faster than MLX 4-bit**; C++ greedy 17.11 ms (58.5 tok/s, slower); Python runtime (greedy, pipelined) 20.75 ms. **Prefill C++ uniform 376 / 325 tok/s (ctx 128 / 2048): 2–4× slower** | MLX 4-bit **13.38 / 14.80 ms/token** (74.8 / 67.6 tok/s), prefill 752 / 1341 tok/s; BF16 19.40 ms (51.6 tok/s) | R:20261003T031427Z_perf_qwen3_30b_a3b_final |
| Memory during C++ decode | mapped file 15.93 / 15.36 GB + process resident peak 1.63 / 1.47 GB (uniform / greedy) | MLX 4-bit decode peak 16,609 MiB (17.4 GB); BF16 58,381 MiB | same |
| Fisher-greedy allocation over expert stacks, 3.0 bpp | greedy 3.0: 3.04 bpp, PPL 9.871 (+13.5%), KL 0.129 | uniform M=3 + res at 3.19 bpp: KL 0.161 (greedy −20%) | R:20261002T070150Z_q7_allocate_qwen3_30b_a3b |
| 4.0 bpp | greedy 4.0: 4.02 bpp, PPL 9.248 (+6.3%), KL 0.078 | RTN 4-bit 4.51 bpp: PPL 9.071 (+4.3%), KL 0.081 | same |
| 2.5 bpp | greedy 2.5: 2.55 bpp, PPL 10.536 (+21.1%), KL 0.177 | — | same |
| Shared codebooks per expert stack | 25× faster encode (648 s vs 16,342 s for M=3), +4% KL, −0.06 bpp | per-expert codebooks | R:20261002T043700Z_q5x_expert_stacks_qwen3_30b_a3b |
| Capacity | runs from codes in **8.2 GB** (VQ G=4, 2.09 bpw linears) at 40.5 tok/s B=1 | MLX 4-bit 17.2 GB, 62.7 tok/s; BF16 61.1 GB | R:20261001T163505Z_moe_e2e_check_Qwen3-30B-A3B |
| Compressed vs reconstructed BF16 (routing check) | KL 0.036, top-1 0.922 — a BF16 rounding effect, not a SUC defect | control without any SUC code (same decodes, stochastic BF16 rounding): KL 0.042, top-1 0.914 | R:20261002T185011Z_moe_routing_check_Qwen3-30B-A3B |

Caveats: the 8.2 GB MoE run used timing codes and predates the K6/R5 runtime fixes that took the dense 8B from 46 to
65 tok/s. Its compressed-vs-reconstructed logit KL (0.032 there, 0.036 in the routing check) is ten times the dense
check; the routing check shows that a perturbation of the same size without any SUC code gives the same expert-set
flips and slightly more KL (§3.8), so the gap is inherent to top-k routing under BF16. At ≈ 4 bpp both final files
beat RTN 4-bit in KL (0.064 vs 0.081, −21%) but not in PPL (+4.7% / +5.1% vs +4.3%); quality rows are single-seed.
Speed (matched harness, §3.7a): the uniform file decodes 6% faster than MLX 4-bit at ctx 128 and 4% at ctx 2048,
but 13% slower at ctx 8192 (19.07 vs 16.63 ms), and its prefill is 2–4× slower (up to 5× at ctx 8192). The greedy
file is slower than MLX 4-bit at every context (17.11 ms at ctx 128; its BF16 attention units and mixed codecs are the
likely cause, not profiled). The earlier "−18% time per token" (12.9 vs 15.75 ms) compared pipelined C++ with
synchronous MLX decode and is withdrawn. At ≈ 4 bpp the compressed 30B-A3B needs ≈ 17 GB (file + process), about as much as MLX 4-bit: it does not fit
a 16 GB Mac; the 8.2 GB capacity point is VQ G=4 (≈ 2.1 bpw), whose quality is far lower (expert-stack VQ G=4: KL 0.35,
R:20261002T043700Z_q5x_expert_stacks_qwen3_30b_a3b).

### 1.3 Where SUC loses (negative results, in order of importance)

| Result | Evidence |
|---|---|
| **≤ 2.5 bpw: far behind QuIP#/QTIP/PV-Tuning on the same model.** Published Qwen3-8B at ≈ 2 bpw (linears only, FP16 head): QuIP# +27%, PV-Tuning +9.6%, QTIP +6.9%. SUC: +91% (VQ G=4 + res, 2.18 bpp, no FT), +70% with block FT, +32% at 2.5 linear bpw (greedy, 4-bit head). | [comparison_published.md](comparison_published.md) §3.1, B:Q5, B:Q6, B:Q7 |
| **Fisher-greedy allocation is worse than a uniform codec below ≈ 3 bpp** on the dense 8B (2.5 bpp: PPL 18.43 vs 14.15 for uniform VQ G=4 at 2.53). At 3.38 and 4.22 bpp greedy equals uniform within noise. ≤ 1.75 bpp is broken with every candidate set. | R:20261002T115608Z_pareto_allocate_qwen3_8b |
| **Block-wise fine-tuning transfers poorly to 8B**: −32% / −42% KL on the 0.6B, −8% / −18% on the 8B, −9% KL but +0.4% PPL on the previous default (additive M=4 + CSR). | R:20261001T081616Z_q6_finetune_qwen3_0.6b, R:20261001T095733Z_q6_finetune_qwen3_8b, R:20261002T130457Z_abl_finetune_qwen3_8b |
| **Residual-stream rotation folding (QuaRot-style) breaks quality**: exact in float32, but encoded it takes PPL 34.5 → 192 (M=3, 0.6B). The cause is the massive-activation channel, which folding moves into the encoded weights. | R:20261001T130811Z_r3_rotation_folding_qwen3_0.6b |
| **SUC decode is slower than MLX 4-bit on the dense models, and prefill is slower everywhere** (matched harness `suc perf`, §3.7a). Qwen3-8B, B = 1: C++ 0.75×, Python runtime 0.78× (0.90× / 0.97× at B = 4 / 16); 0.6B on the M3 0.68× (indicative). Prefill: 8B 0.34–0.58×, 30B-A3B uniform 0.21–0.50×. Only the uniform 30B-A3B file decodes faster than MLX 4-bit (+6% at ctx 128, +4% at 2048; −13% at 8192); the greedy 30B-A3B file is 0.74–0.81×. The earlier claims of 8B parity and 30B-A3B "22% faster" came from a mismatched harness (synchronous MLX) and are withdrawn. | R:20261003T015734Z_perf_qwen3_8b_final, R:20261003T031427Z_perf_qwen3_30b_a3b_final |
| **The ≈ 3-bpp codecs are not faster than 4-bit ones.** VQ G=2 K=64 (6-bit indices) + residual decodes at 49.4 / 85.1 / 107.2 tok/s (B = 1 / 4 / 16) vs 55.2 / 113.1 / 165.5 for K=256 (8-bit indices) in the same run: index unpacking costs more than the saved bytes, worst at B = 16. On the M3 the 3.5-bpp 0.6B file decodes at 7.07 vs 6.95 ms/token for the 4.25-bpp one. | R:20261002T212251Z_e2e_lowg_qwen3_8b, B:Final Qwen3-0.6B SUC1 files on an Apple M3 (16 GB) |
| **On the 30B-A3B, ≈ 4-bpp SUC is slightly worse than RTN 4-bit in PPL** (+4.7% / +5.1% vs +4.3%) although better in KL, and it saves only 7–11% of the bytes; at that rate it needs ≈ 17 GB, like MLX 4-bit. | R:20261002T204858Z_compress_qwen3_30b_a3b_final_greedy, R:20261002T212215Z_compress_qwen3_30b_a3b_final |
| **Entropy coding does not pay at runtime** and saves 4.3% of the 8B file. | R:20261002T142930Z_entropy_coding_qwen3_8b |
| **The original "lookup table of scalars" idea loses to RTN** (scalar K=16 unrotated, no residual: PPL 76 at 4 bpp). The data-free additive codebooks also lose to RTN until activation-selected residuals are added. | B:Q1–Q4 (both models) |

---

## 2. Method as built

| Component | As built | Evidence that it is needed / the choice |
|---|---|---|
| **Incoherence processing** | Randomized Hadamard rotation, **one-sided** (W Vᵀ: input side only) by default (`rotate_out: false`). The input rotation is shared by q/k/v and by gate/up (`rotation_groups`). Signs are re-derived from seeds, not stored. | Rotation removes all > 6σ outliers (kurtosis up to 32 → < 0.06). Removing it costs +41% KL at 4.2 bpp (ablation). One-sided vs two-sided is equal in quality within noise, and removes 4 of 8 online rotations per block (+21% B=1 speed). |
| **Base codebooks** | Per-tensor k-means codebooks (k-means++, deterministic seeds) on RMS-normalised rotated rows, with per-row fp16 scales. Single-codebook VQ (M=1) or additive (M codebooks summed, greedy residual fitting). D4 ceiling: K·G·2 bytes ≤ 16 KB. | **Ablation: at 4 bpw, VQ G=2 K=256 (1 KB codebook) beats additive M=4 G=8 (4 × 4 KB) by 27% in KL; even a rotated scalar K=16 does.** At 2 bpp VQ G=4 beats additive M=2 (PPL 162 vs 218). **Q5c: across the range, VQ G=2 beats additive M=3 at 3.37 bpp (−19% KL) and the greedy mixture at ≈ 3 bpp (2.95 vs 3.00 bpp, −4% KL); at 2.53 bpp G=4 K=256 beats G=2 K=16 (KL 0.430 vs 0.524).** Low-dimensional VQ is the better representation *and* the cheaper kernel (one lookup per group). **Default: VQ G=2 K=256 (1 KB codebook, 8-bit indices).** G=3 does not divide the Qwen3 hidden sizes. |
| **Residuals** | 0.5% of entries per tensor, int8 values, selected by E[x_j²]·R²_ij (Phase 2 activation statistics), in the original (unrotated) basis. **Column structure** by default: whole input columns stored dense int8 with one fp16 scale per column (q/k/v and gate/up have ~99% of their selected entries in 32 columns). CSR remains available. | Removing residuals: +113% KL (4.08 bpp). Magnitude selection at equal bits: +107% KL. Columns vs CSR: −2…−10% KL on the 8B and 0.6B; within noise in the ablation. The column pass costs 1.1 ms/step vs 3.4–40 ms for CSR. |
| **Embedding / LM head** | Affine 4-bit g64 (fixed `embedding_codec`), or allocated from `head_candidates`. | Encoding the LM head with the low-bit codebook was the largest single error source (Q7: KL 0.248 → 0.162 at the same block codec). Allocated, it never takes more than 4-bit (Q7b). |
| **Sensitivity** | Empirical Fisher ½ Σ F ΔW², exact and layer-streamed. Diagonal-H proxy as fallback. | Spearman vs measured KL (n = 24 sampled perturbations): Fisher 0.75 / 0.82 / 0.85 (0.6B / 8B / 30B), diag-H 0.45 / 0.49 / 0.18, weight MSE ≈ 0. |
| **Allocation** | Greedy marginal-value allocator (spec §39) over per-tensor candidates, or over **expert stacks** (one unit per MoE layer × projection). Budget = average bpw over the allocated tensors. | Ranks whole-model plans correctly (Spearman 0.93–0.98 dense, 0.82 MoE). The gain depends on the sensitivity spread: on the MoE −20% KL at 3 bpp; on the dense 8B within noise at ≥ 3.4 bpp, and worse below 3 bpp. |
| **Expert-stack codebooks (MoE)** | `codebook_scope: expert_stack`: one codebook set per (layer, projection), fitted on a pooled sample of all 128 experts; per-expert indices, scales and residuals. Stored once in SUC1 (content dedup). | 25× faster encode, +4% KL. It is the only way the ~180B MoE targets are practical; per-expert codebooks would need ≈ 10–15 h per codec there (projection). |
| **Block-wise fine-tuning** | Indices fixed; codebooks, row scales and residual values tuned with Adam to match each BF16 block's output on the already-quantized prefix. The straight-through estimator keeps the stored precision. 100 steps per block, 64 sequences. Measured with CSR residuals only (the current code also accepts column residuals; not measured). Not supported with expert stacks. | Helps on the 0.6B, little on the 8B (see §1.3). |
| **SUC1 format** | 64-byte header, canonical JSON directory, 64-byte-aligned little-endian payload sections (codebooks, indices, scales, residual, raw) and a SHA-256 footer. Deterministic. Indices bit-packed at their exact width. Codebooks deduplicated. Streamed writer (one tensor resident). See [format.md](format.md). | The reference decoder and the C++ decoder reproduce every codec bit for bit. Final 8B file: 4,322.2 MB, 4.2216 file bpp vs 4.2210 model bpp (header, directory and padding cost 0.0006 bpp); 16 sampled encodings bit-exact on reload, reference-decoder max rel. error 0 (R:20261002T191301Z_compress_qwen3_8b_final). |
| **Kernels** | `mx.fast.metal_kernel`: VQ/additive lookup kernel (vec path) and simdgroup-matrix path (MMA) from a tuned batch; codebooks staged in threadgroup memory where faster; fused q\|k\|v and gate\|up launches with per-segment codebook staging; column-residual kernel fused into the group launch; "blocks"-layout online Hadamard (one launch over many cores); optional RMSNorm/SwiGLU + rotation fusion. The weight matrix is never materialised. | Each kernel checked against the Python reference decoder (max/mean/relative error, logit KL). Exactness of the full model: f32 logit KL ≈ 1e-8 (0.6B); bf16 KL 0.001–0.007 (8B). |
| **Per-device autotuning** | `suc.kernels.autotune`: codebook placement per M, rotation layout and block shape per size, element-wise fusion per op, MMA crossover batch. Measured once per device (cold-cache, decode context) and cached in `~/.cache/suc/kernel_tuning.json`. | Choices flip between GPU generations: threadgroup codebooks are 1.1–3× faster on the M1 Ultra and slower on the M3; norm fusion is slower on the M1 Ultra. |
| **C++ runtime** | `cpp/`: SUC1 reader + CPU reference decoder, Metal backend running the same kernels from the mapped file (zero copy), Qwen3 dense and Qwen3-MoE graph, Qwen BPE tokenizer and chat template, sampling, pipelined decode (`suc1-generate`). | Hidden states and logits bit-identical to the Python runtime on both 8B files (3 prompts × 64 greedy tokens each; R:20261002T053015Z_cpp_generate_check, R:20261002T191814Z_cpp_generate_check). On the final file (cpp check harness, synchronous Python decode): 13.3 ms/token at B = 1 vs 19.9 ms for the Python runtime; matched-harness speed vs MLX 4-bit in §3.7a. Qwen3-MoE: bit-identical on both final 30B-A3B files (3 prompts × 64 greedy tokens), 12.9 / 16.5 ms/token (R:20261002T232117Z_cpp_check_qwen3_30b_a3b_final, R:20261002T230743Z_cpp_check_qwen3_30b_a3b_final_greedy). Qwen3-0.6B on the M3: bit-identical, 6.95 ms/token (B:Final Qwen3-0.6B SUC1 files on an Apple M3 (16 GB)). |
| **D8 streaming pipeline** | `suc compress`: preflight → calibrate → reference → [candidates] → plan → encode → [finetune] → container → evaluate → report. Every stage resumable. In the current code every stage is layer-streamed (one block of weights resident; large hidden states are spilled to disk). | **Final 8B run: every stage streamed** (the result lists no stage that needs the whole BF16 model or holds every encoding), 1361 s total; peak MLX memory per stage 6.53 / 7.56 / 15.22 / 3.20 / 8.13 GiB (calibrate / reference / encode / container / evaluate), against 16.4 GB of BF16 weights (R:20261002T191301Z_compress_qwen3_8b_final; §6.1). **Final 30B-A3B runs (61 GB BF16): every stage streamed**, uniform 1994 s / greedy 5443 s, peak MLX memory ≤ 8.05 GiB in every stage (calibrate 5.11, reference 5.23, candidates 8.05, encode 7.80, container 1.79, evaluate 6.00–6.14 GiB; R:20261002T212215Z_compress_qwen3_30b_a3b_final, R:20261002T204858Z_compress_qwen3_30b_a3b_final_greedy). |

Not built: the multiplicative-weights / adaptive allocator (§29, §40), output-loss index updates (PV-Tuning style),
end-to-end KD fine-tuning on the 8B, fine-tuning of shared expert stacks, the partial-product table kernel (K1 found
it slower in its first form), llama.cpp/GGUF interoperability.

### 2.1 What "Semantic" and "Universal" mean in the system as built

- **Semantic** (spec: bound the distortion of the model's *outputs*, D_semantic(M, M_q) ≤ ε, not the weight error).
  As built: KL/PPL against the reference are the acceptance metrics; sensitivity is the empirical Fisher of the model
  loss (Spearman 0.75–0.85 vs measured KL, weight MSE ≈ 0); residual entries are selected from real activation
  statistics. Weaker than the name suggests: codebooks are fitted to rotated weights (not to the output), block-wise
  fine-tuning is not in the final recipe, and on dense models the final plan is uniform because Fisher-greedy allocation
  was within noise (§3.5). The semantic part is decisive on MoE and on the frontier models, where Fisher decides which
  experts drop below 3 bits and which stay in their native format.
- **Universal** (spec: an allocation/universalization principle inspired by Cover's Universal Portfolio, one framework
  for any model). As built: one pipeline, one file format (SUC1) and one runtime family for Qwen3 dense/MoE, qwen4_exp
  (Qwen3.8-Flash-Next), DeepSeek-V4-Flash and GLM-5.3-Flash, with an allocator choosing per tensor / expert among
  heterogeneous representations (VQ G=2/G=4, additive, affine; native MXFP4 pass-through, §3.14). Not built: an adaptive
  allocator in the Cover sense (no prior knowledge of the data distribution); the allocator is a budgeted greedy over a
  Fisher estimate.
- In short: SUC as built is sensitivity-guided low-dimensional vector quantization in a single executable format, run
  directly from codes. The name holds with the reading above.

---

## 3. Results

### 3.1 Phase 0 — baselines (Mac Studio)

| Model | Representation | bpp | Size (GB) | PPL | ΔPPL | KL (512 tok) | Decode B=1 / B=16 (tok/s) | Prefill p2048 (tok/s) |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Qwen3-8B | BF16 | 16.00 | 16.38 | 9.717 | — | — | 33.4 / 177.4 | 865 |
| | MLX affine 8-bit g64 | 8.50 | 8.70 | 9.726 | +0.1% | 0.0018 | 55.7 / 195.0 | 720 |
| | MLX affine 4-bit g64 | 4.50 | 4.61 | 10.082 | +3.8% | 0.109 | 64.9 / 189.2 | 703 |
| Qwen3-30B-A3B | BF16 | 16.00 | 61.06 | 8.697 | — | — | 46.9 / 182.8 | 1712 |
| | MLX affine 8-bit | 8.50 | 32.44 | 8.718 | +0.2% | 0.0038 | 59.6 / 240.7 | 1364 |
| | MLX affine 4-bit (router 8-bit) | 4.50 | 17.18 | 9.062 | +4.2% | 0.069 | 56.6 / 227.3 | 1327 |

Sources: R:20260930T223113Z_phase0_target_qwen3_8b (provenance caveat in B:Phase 0 — Qwen3-8B),
R:20261001T005202Z_phase0_target_qwen3_30b_a3b. The Phase 0 harness decodes 128 tokens with a synchronous eval per token.
The e2e harness used for the SUC speed rows (§3.7) decodes 32 tokens and measures MLX 4-bit at 74–75 tok/s. Speeds are
compared only within one harness. Finding: at 4 bits the runtime, not bandwidth, limits batch-1 decode (INT4 reaches
~37% of peak bandwidth, BF16 ~68%). For the MoE, fewer bytes barely speed up decode; capacity is the win.

### 3.2 Phase 1–2 — weight structure and sensitivity

| Finding | 0.6B | 8B | 30B-A3B | Source |
|---|---|---|---|---|
| Max excess kurtosis before → after rotation | 15.9 → < 0.02 | 8.7 → 0.012 | 32.4 → 0.057 | B:Phase 1 (three sections) |
| Rel. MSE, VQ G=8 K=256 (1 bpw), rotated / scalar Lloyd–Max / R(D) bound | 0.324 / 0.363 / 0.250 | 0.324 / 0.363 / 0.250 | 0.324 | same |
| Rel. MSE at 2 bpw (VQ G=4 K=256) / scalar | — | 0.098 / 0.118 | 0.098 | same |
| Gap to the Shannon bound, G=4, 1 → 2.5 bpw | — | 1.4 → 2.1 dB | — | same |
| Shared directional structure between blocks | none (isotropic) | — | — | B:Phase 1 — dev |
| Index entropy / log₂K | ≥ 99% | ≥ 98% | ≥ 98% | same |
| Spearman vs measured KL: Fisher / diag-H / weight MSE | 0.75 / 0.45 / −0.25 | 0.82 / 0.49 / 0.39 | 0.85 / 0.18 / −0.13 | R:20260930T233540Z_phase2_calib_qwen3_0.6b, R:20260930T235623Z_phase2_calib_qwen3_8b, R:20261001T002456Z_phase2_calib_qwen3_30b_a3b |
| Sensitivity spread (KL for the same 1-bpw error) | 0.016–0.141 | 0.002–1.20 (`layers.6.mlp.down_proj`) | experts ≤ 0.0005, attention up to 0.076 | same |

### 3.3 Quality track on Qwen3-8B — codecs, residuals, rotation (Q1–Q5b, R1)

Q5 protocol: the embedding and LM head are encoded with the row's codec.

| Codec | Model bpp | PPL | ΔPPL | KL | Source |
|---|---:|---:|---:|---:|---|
| RTN affine 4-bit g64 | 4.50 | 10.088 | +3.8% | 0.092 | R:20261001T033522Z_q1q4_codecs_qwen3_8b |
| Additive M=4, no residual | 4.01 | 10.557 | +8.6% | 0.165 | same |
| Additive M=4 + 0.5% act. residual (CSR) | 4.18 | 10.038 | +3.3% | 0.084 | R:20261001T053826Z_q5_residual_qwen3_8b |
| Same, columns | 4.18 | 10.073 | +3.7% | 0.082 | R:20261001T180551Z_q5b_residual_structure_qwen3_8b |
| Same, CSR, one-sided rotation | 4.18 | 10.130 | +4.3% | 0.077 | R:20261001T192950Z_r1_rotation_sides_qwen3_8b |
| Scalar codebook K=16 (unrotated, no residual) | 4.00 | 76.1 | — | 1.80 | R:20261001T033522Z_q1q4_codecs_qwen3_8b |
| RTN affine 3-bit | 3.50 | 13.364 | +37.5% | 0.370 | same |
| Additive M=3, no residual | 3.01 | 13.407 | +38.0% | 0.473 | same |
| Additive M=3 + 0.5% act. residual (CSR) | 3.18 | 11.316 | +16.5% | 0.248 | R:20261001T053826Z_q5_residual_qwen3_8b |
| Same, one-sided | 3.18 | 11.076 | +14.0% | 0.246 | R:20261001T192950Z_r1_rotation_sides_qwen3_8b |
| Additive M=3 + 0.5% **magnitude** residual | 3.18 | 13.043 | +34.2% | 0.454 | R:20261001T053826Z_q5_residual_qwen3_8b |
| RTN affine 2-bit | 2.50 | 1631 | — | 5.09 | R:20261001T033522Z_q1q4_codecs_qwen3_8b |
| VQ G=4 K=256, no residual | 2.00 | 162.5 | — | 2.58 | same |
| Additive M=2, no residual | 2.01 | 217.7 | — | 2.97 | same |
| VQ G=4 + 0.5% act. residual (columns) | 2.18 | 18.536 | +90.8% | 0.736 | R:20261001T180551Z_q5b_residual_structure_qwen3_8b |
| VQ G=8 (≈ 1 bpp), with or without rotation | 1.01 | > 10⁷ | — | > 14 | R:20261001T033522Z_q1q4_codecs_qwen3_8b |

**Q5c — low-dimensional single-codebook VQ across the bit range** (config
`experiments/configs/q5c_lowdim_vq_qwen3_8b.yaml`; protocol A as in the ablations: one-sided rotation, 0.5% activation
column residual, embedding/LM head affine 4-bit; R:20261002T175941Z_q5c_lowdim_vq_qwen3_8b, plus the ablation row
VQ G=2 K=256 from R:20261002T085834Z_abl_codebook_qwen3_8b; B:Q5c):

| Codec (+ 0.5% act. column residual) | Model bpp | PPL | ΔPPL | KL | Top-1 | Best alternative at equal bits |
|---|---:|---:|---:|---:|---:|---|
| **VQ G=2 K=256** (1 KB codebook) | 4.221 | **9.994** | +2.9% | **0.0494** | — | additive M=4: 10.121 / 0.0678 (R:20261002T075009Z_abl_base_qwen3_8b) |
| **VQ G=2 K=64** | 3.373 | **10.618** | +9.3% | **0.1237** | 0.857 | additive M=3: 10.692 / 0.1526 (same run); greedy 3.38: 10.78 / 0.141 |
| additive M=3 K=256 G=8 (reference) | 3.376 | 10.692 | +10.0% | 0.1526 | 0.844 | |
| **VQ G=2 K=32** | 2.949 | **11.560** | +19.0% | **0.2335** | 0.807 | greedy 3.0 at 3.00 bpp: 11.71 / 0.244 |
| **VQ G=4 K=256** | 2.525 | **14.432** | +48.5% | **0.4304** | 0.750 | VQ G=2 K=16 below |
| VQ G=2 K=16 | 2.525 | 15.616 | +60.7% | 0.5240 | 0.726 | |

Conclusion: **from ≈ 2.95 to 4.2 bpp a single 2-dimensional codebook is the best SUC codec measured on Qwen3-8B**:
at equal bits it beats the additive multi-codebooks (KL −27% at 4.22, −19% at 3.37) and the Fisher-greedy mixtures
(−12% at 3.37–3.38, −4% at 2.95–3.00). At ≈ 2.5 bpp the 4-dimensional codebook wins (G=2 K=16 is too coarse). G=3
is not usable on Qwen3 (hidden sizes not divisible by 3). Encode is also 2.4–3.6× cheaper than additive M=3
(92–138 s vs 329 s, `Encode (s)` column of the result). All rows are single-seed (noise ≈ ±3% KL, ±1% PPL);
the −4% at ≈ 3 bpp is at the edge of it, the others are not. Batch-1 speed (Python runtime, §3.7): G=2 K=64 + residual 49.4 tok/s vs 55.2
for K=256 + residual and 74.4 for MLX 4-bit in the same run (R:20261002T212251Z_e2e_lowg_qwen3_8b); 6-bit indices are
slower than 8-bit ones, so the ≈ 3.4-bpp point saves memory but not time. G=2 K=32 (5-bit indices, the 2.95-bpp point)
was not in the speed run.

### 3.4 Fine-tuning (Q6)

| Model / codec | bpp | PPL untuned → tuned | KL untuned → tuned | Time / peak | Source |
|---|---:|---|---|---|---|
| 0.6B, additive M=3 + 0.5% res | 3.20 | 35.77 → 29.83 | 0.593 → 0.405 (−32%) | 214 s | R:20261001T081616Z_q6_finetune_qwen3_0.6b |
| 0.6B, VQ G=4 + 0.5% res | 2.18 | 565 → 107.5 | 2.66 → 1.54 (−42%) | 177 s | same |
| 8B, additive M=3 + 0.5% res | 3.18 | 11.316 → 11.223 | 0.248 → 0.228 (−8%) | 43 min / 34 GiB | R:20261001T095733Z_q6_finetune_qwen3_8b |
| 8B, VQ G=4 + 0.5% res | 2.18 | 18.70 → 16.49 | 0.742 → 0.610 (−18%) | 33 min / 30 GiB | same |
| 8B, M=4 + 0.5% CSR res, one-sided, head affine 4 | 4.225 | 10.185 → 10.225 | 0.0691 → 0.0629 (−9%) | — | R:20261002T083514Z_abl_residual_qwen3_8b, R:20261002T130457Z_abl_finetune_qwen3_8b |

Every block's output error falls by 40–45% on the 8B, yet whole-model KL falls only 8–18%. A per-block target does
not capture error accumulation across blocks. End-to-end (logit) fine-tuning was not run on the 8B.

### 3.5 Allocation (Q7, Q7b, Pareto)

**Qwen3-8B Pareto frontier** (greedy Fisher allocation, embedding and LM head allocated too; R:20261002T115608Z_pareto_allocate_qwen3_8b;
Spearman(predicted, measured KL) across plans 0.982):

| Plan | Model bpp | PPL | ΔPPL | KL |
|---|---:|---:|---:|---:|
| greedy 1.05 | 1.05 | 42,922 | broken | 8.47 |
| greedy 1.5 | 1.50 | 915.7 | broken | 4.43 |
| greedy 1.75 | 1.75 | 145.2 | broken | 2.60 |
| greedy 2.0 | 2.00 | 39.53 | +307% | 1.405 |
| greedy 2.25 | 2.25 | 28.45 | +193% | 1.071 |
| greedy 2.5 | 2.50 | 18.43 | +90% | 0.725 |
| **uniform VQ G=4 + res** | 2.53 | **14.15** | +46% | **0.429** |
| *uniform VQ G=4 K=256 + res (Q5c)* ¶ | 2.53 | 14.43 | +48.5% | 0.430 |
| *uniform VQ G=2 K=32 + res (Q5c)* ¶ | 2.95 | **11.56** | +19.0% | **0.234** |
| greedy 3.0 | 3.00 | 11.71 | +20.5% | 0.244 |
| *uniform VQ G=2 K=64 + res (Q5c)* ¶ | 3.37 | **10.62** | +9.3% | **0.124** |
| uniform M=3 + res | 3.38 | 10.65 | +9.6% | 0.150 |
| greedy 3.38 | 3.38 | 10.78 | +10.9% | 0.141 |
| greedy 3.5 | 3.50 | 10.65 | +9.6% | 0.115 |
| greedy 4.0 | 4.00 | 10.26 | +5.6% | 0.078 |
| *uniform VQ G=2 K=256 + res (final)* ¶ | 4.22 | **9.994** | +2.9% | **0.049** |
| greedy 4.22 | 4.22 | 10.27 | +5.7% | 0.073 |
| uniform M=4 + res | 4.22 | 10.19 | +4.9% | 0.069 |
| *MLX affine 4-bit (Phase 0)* | 4.50 | 10.082 | +3.8% | *0.109 (512-token protocol)* |
| *RTN affine 4-bit (Q protocol)* | 4.50 | 10.088 | +3.8% | 0.092 |

¶ Uniform low-dimensional VQ rows from Q5c (R:20261002T175941Z_q5c_lowdim_vq_qwen3_8b) and the final run
(R:20261002T191301Z_compress_qwen3_8b_final), protocol A (head fixed at affine 4-bit); they were not allocator
candidates in the Pareto run. With them, a uniform VQ point has lower KL than the greedy plan at about the same bits
(within 0.05 bpp) at 2.5, 3.0, 3.38 and 4.22 bpp; greedy 3.5 and 4.0 have no uniform VQ point at equal bits (they lie
between G=2 K=64 and G=2 K=256). A greedy run over
low-G VQ candidates was not done.

Earlier 8B allocation runs (head fixed at affine 4-bit, CSR residual, two-sided rotation): greedy 2.5 / 3.0 / 3.5 /
4.0 at 2.80 / 3.23 / 3.65 / 4.08 bpp give PPL 12.82 / 11.00 / 10.55 / 10.19, KL 0.308 / 0.176 / 0.124 / 0.084
(R:20261001T072625Z_q7_allocate_qwen3_8b). With the head allocated too (Q7b), the allocator never buys more than 4-bit
for it (R:20261001T154452Z_q7b_allocate_head_qwen3_8b).

**Qwen3-30B-A3B** (allocation over 336 units, one per attention tensor or expert stack;
R:20261002T070150Z_q7_allocate_qwen3_30b_a3b; Spearman 0.821):

| Plan | Model bpp | PPL | ΔPPL | KL | Top-1 |
|---|---:|---:|---:|---:|---:|
| greedy 2.5 | 2.55 | 10.536 | +21.1% | 0.177 | 0.850 |
| greedy 3.0 | 3.04 | 9.871 | +13.5% | 0.129 | 0.867 |
| uniform M=3 + res | 3.19 | 9.961 | +14.5% | 0.161 | 0.854 |
| greedy 3.5 | 3.53 | 9.584 | +10.2% | 0.098 | 0.877 |
| greedy 4.0 | 4.02 | 9.248 | +6.3% | 0.078 | 0.889 |
| uniform M=4 + res | 4.17 | 9.235 | +6.2% | 0.083 | 0.885 |
| RTN 4-bit | 4.51 | 9.071 | +4.3% | 0.081 | 0.887 |

At every budget the greedy plans keep 87–133 of the 336 units in BF16. All of them are attention projections: v_proj
47–48 of 48, k_proj 37–47, plus a few o_proj and q_proj (counted from the plan rows). Attention is about 3% of the
parameters. The experts go to M=3/M=4/VQ codecs. This matches the Phase 2 finding that a
single expert is ~100× less sensitive than attention. The runtime speed of these mixed plans was not measured.

### 3.6 Mandatory ablations (spec §26) — Qwen3-8B, 4.22 bpp

Protocol A (`suc quantize`, head affine 4-bit, shared q/k/v and gate/up rotation). Base = additive M=4 K=256 G=8,
one-sided rotation, 0.5% activation column residual. Sources: R:20261002T075009Z_abl_base_qwen3_8b,
R:20261002T083514Z_abl_residual_qwen3_8b, R:20261002T085834Z_abl_codebook_qwen3_8b,
R:20261002T130457Z_abl_finetune_qwen3_8b, R:20261002T133705Z_abl_repeat_qwen3_8b, R:20261002T115608Z_pareto_allocate_qwen3_8b,
R:20261002T142930Z_entropy_coding_qwen3_8b, R:20261002T1*_e2e_abl_*_qwen3_8b.

| Row (spec §26) | bpp | PPL | ΔPPL | KL | Δ KL vs base | Verdict |
|---|---:|---:|---:|---:|---:|---|
| SUC (base) | 4.225 | 10.121 | +4.2% | 0.0678 | — | |
| SUC, seed 2 (repeatability) | 4.225 | 10.081 | +3.7% | 0.0707 | +4% | noise ≈ ±3% KL, ±1% PPL |
| − incoherence (no rotation) | 4.225 | 10.455 | +7.6% | 0.0954 | **+41%** | required |
| two-sided instead of one-sided | 4.225 | 10.113 / 10.290 (seed 2) | +4.1% / +5.9% | 0.0725 / 0.0708 | +7% / 0% | quality-neutral; one-sided is faster |
| − residuals (4.08 bpp) | 4.083 | 10.632 | +9.4% | 0.1443 | **+113%** | required |
| − activation selection (magnitude columns) | 4.225 | 10.649 | +9.6% | 0.1403 | **+107%** | required (the "semantic" part of the residual) |
| − column structure (CSR) | 4.225 | 10.185 | +4.8% | 0.0691 | +2% | within noise; columns are faster |
| + block fine-tuning (CSR) | 4.225 | 10.225 | +5.2% | 0.0629 | −9% vs CSR | small KL gain, PPL slightly worse |
| − multi-codebook: VQ G=2 K=256 | 4.221 | **9.994** | **+2.9%** | **0.0494** | **−27%** | **additive is not the right choice at 4 bpw** |
| − vector quantization: scalar K=16 G=1, rotated | 4.221 | 10.074 | +3.7% | 0.0554 | −18% | rotated scalar also beats additive M=4 |
| − semantic allocation: greedy vs uniform at 4.22 / 3.38 bpp (protocol B) | 4.22 / 3.38 | 10.27 vs 10.19 / 10.78 vs 10.65 | | 0.073 vs 0.069 / 0.141 vs 0.150 | within noise | no gain on the dense model at ≥ 3.4 bpp (MoE: −20% at 3 bpp, §3.5) |
| − adaptive allocation (§29/§40) | — | — | — | — | — | **not done** (not implemented) |
| − entropy coding | 4.226 → 4.043 (file) | unchanged | | unchanged | | storage-only, −4.3% file |
| − fused kernels (decode tok/s B=1 / B=16) | | all fusions 34.6 / 141.4; no fusion 33.9 / 126.9; **no fused codebook staging 20.5 / 109.6**; residual unfused 33.8 / 151.8; two-sided 34.0 / 136.3 | | | | staged codebooks are the one fusion that matters at B=1 |

What the ablation says about where the improvement comes from: **rotation** and **activation-selected residuals**
create the quality. The residuals are what lets codebooks beat RTN at all. The choice of a **low-dimensional
codebook** matters more than the additive structure; Q5c (§3.3) confirms this from 2.95 to 4.2 bpp, so the
ablation's "− multi-codebook" row became the final configuration. Allocation matters only when the sensitivity
spread is large (MoE). Fine-tuning and entropy coding add little. The ablation base (additive M=4) is no longer the
default; the ablation rows were not re-run around the VQ G=2 base.

### 3.7 Kernel track and direct execution

**§43 single-layer gate (cold cache, Qwen3-8B block = 7 linears, µs, codebook in threadgroup memory;
R:20261001T072637Z_k3_m1ultra_*):**

| B | FP16 | MLX q4 (4.5) | MLX q3 (3.5) | MLX q2 (2.5) | VQ G=8 (1.0) | VQ G=4 (2.0) | add. M=2 (2.0) | add. M=3 (3.0) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 685 | 215 | 196 | **167** | 174 | 188 | 248 | 353 |
| 4 | 1718 | 490 | 561 | 460 | 814 | **445** | 782 | 894 |
| 16 | 1772 | 1262 | 1345 | **1230** | 1510 | 1486 | 1682 | 1906 |

Single-codebook VQ beats MLX 4-bit at B = 1 and 4 but not MLX 2-bit. Additive codebooks cost ~M× a single codebook.
The C++ Metal backend reproduces these single-layer costs (e.g. 12288×4096 VQ G=4, B=1: 30.3 µs C++ vs 27.0 µs Python
kernel; R:20261002T051929Z_cpp_metal_linear_12288x4096, R:20261002T052055Z_k3_cold_cache_suc_8b).

**End-to-end decode on Qwen3-8B, Python runtime, progression of the runtime work (tok/s, B = 1 / 4 / 16):**

| Step | VQ G=4 | VQ G=4 + residual | additive M=3 + residual | MLX 4-bit | Source |
|---|---|---|---|---|---|
| K5 first integration (two-sided, CSR) | 38.7 / 108.8 / 225.9 | 26.6 / 67.9 / 124.9 | 23.4 / 59.7 / 106.8 | 74.9 / 137.8 / 193.3 | R:20261001T074850Z_e2e_generate_qwen3_8b |
| R1 one-sided rotation | 46.1 / 124.3 / 242.4 | — | — | 75.3 / 137.9 / 193.1 | R:20261001T181337Z_e2e_generate_qwen3_8b_rot_onesided |
| K6 column residual + fused staging (fusion off) | 47.3 / 126.2 / 252.9 | 45.4 / 103.2 / 224.6 | 35.0 / 91.9 / 147.9 | 75.0 / 137.5 / 192.0 | R:20261001T202042Z_e2e_k6_after_qwen3_8b, R:20261001T203018Z_e2e_k6_nofuse_elementwise_qwen3_8b |
| **R5 blocks-layout rotation + autotuning** | **65.0 / 139.2 / 262.3** | **55.7 / 110.0 / 233.3** | **42.8 / 91.9 / 146.9** | 74.4 / 137.6 / 193.0 | R:20261002T050049Z_e2e_r5_after_qwen3_8b |

**Final runtime matrix on Qwen3-8B (Python runtime, prompt 128, 32 decode tokens; R:20261002T050049Z_e2e_r5_after_qwen3_8b,
R:20261002T133705Z_e2e_abl_default_qwen3_8b):**

| Variant | Linear bpw | Resident weights (GiB) | Decode B=1 / 4 / 16 (tok/s) | % of MLX 4-bit at B=1 | TTFT B=1 (ms) | Prefill B=1 (tok/s) | Peak B=1 (GiB) |
|---|---:|---:|---|---:|---:|---:|---:|
| BF16 | 16 | 15.26 | 34.0 / 52.1 / 188.8 | 45% | 219 | 584 | 15.42 |
| MLX affine 4-bit | 4.5 | 4.29 | 74.9 / 137.7 / 194.4 | 100% | 209 | 612 | 4.60 |
| SUC VQ G=4 | 2.00 | 2.28 | 65.0 / 139.2 / 262.3 | 87% | 244 | 525 | 2.56 |
| SUC VQ G=4 + 0.5% col. res | 2.17 | 2.41 | 55.7 / 110.0 / 233.3 | 75% | 265 | 483 | 2.87 |
| SUC additive M=3 + 0.5% col. res | 3.17 | 3.22 | 42.8 / 91.9 / 146.9 | 58% | 349 | 367 | 3.68 |
| SUC additive M=4 | 4.01 | 3.90 | 37.6 / 95.3 / 155.0 | 50% | 386 | 331 | 4.18 |
| SUC additive M=4 + 0.5% col. res | 4.17 | 4.03 | 34.6 / 85.9 / 141.4 | 46% | 405 | 316 | 4.49 |

**Low-dimensional VQ decode speed (Python runtime, same harness as the matrix; R:20261002T212251Z_e2e_lowg_qwen3_8b,
B:Low-G VQ decode speed (Python runtime) and same-session baselines).** Timing codes (fast k-means, column residuals of
the protocol size), one-sided rotation, embedding/LM head 4-bit; each variant checked against the same weights decoded
to BF16 (logit KL 0.0009–0.009). MLX 4-bit was measured in the same run.

| Variant | Linear bpw | Resident (MiB) | Decode B=1 / 4 / 16 (tok/s) | ms/token B=1 | % of MLX 4-bit B=1 / 4 / 16 | TTFT B=1 (ms) | Prefill B=1 (tok/s) |
|---|---:|---:|---|---:|---|---:|---:|
| MLX affine 4-bit | 4.5 | 4,394 | 74.4 / 137.8 / 194.3 | 13.4 | 100 / 100 / 100 | 208 | 617 |
| VQ G=2 K=256 | 4.01 | 3,987 | 64.9 / 126.2 / 186.4 | 15.4 | 87 / 92 / 96 | 357 | 358 |
| **VQ G=2 K=256 + col. residual (final codec)** | 4.18 | 4,126 | **55.2 / 113.1 / 165.5** | 18.1 | 74 / 82 / 85 | 382 | 335 |
| VQ G=2 K=64 | 3.01 | 3,159 | 60.0 / 105.8 / 114.2 | 16.7 | 81 / 77 / 59 | 488 | 263 |
| VQ G=2 K=64 + col. residual | 3.18 | 3,298 | 49.4 / 85.1 / 107.2 | 20.2 | 66 / 62 / 55 | 488 | 262 |
| VQ G=2 K=16 | 2.01 | 2,331 | 72.6 / 123.8 / 233.9 | 13.8 | 98 / 90 / 120 | 352 | 364 |
| VQ G=2 K=16 + col. residual | 2.18 | 2,470 | 59.6 / 98.8 / 166.2 | 16.8 | 80 / 72 / 86 | 383 | 334 |
| VQ G=4 K=256 | 2.01 | 2,331 | 71.2 / 119.9 / 261.7 | 14.0 | 96 / 87 / 135 | 240 | 533 |
| VQ G=4 K=256 + col. residual | 2.18 | 2,470 | 60.9 / 99.0 / 187.9 | 16.4 | 82 / 72 / 97 | 269 | 476 |

Findings: (1) In the Python runtime the final codec decodes at 74–85% of MLX 4-bit at every batch; G=2 is not faster
than MLX 4-bit at any batch, unlike G=4 at B = 16. (2) 6-bit indices (K=64) are slower than 8-bit ones at every batch,
worst at B = 16 (114 vs 186 tok/s): index unpacking, not bytes, sets the cost, so the ≈ 3.4-bpp point saves memory
(−828 MiB) but not time. 4-bit indices (K=16) are cheap again. This is a kernel gap, not a property of the codec.
(3) Prefill with G=2 is 42–58% of MLX 4-bit's rate. (4) Run-to-run spread: VQ G=4 + residual gave 55.7 tok/s at B = 1
in the R5 run and 60.9 here (+9%), so cross-run comparisons in this harness carry ≈ ±10%. G=2 K=32 (5-bit indices) was
not in the run. Batch > 1 is measured only in Python; the C++ runtime decodes batch 1.

On the actual final file, the cpp check measures 13.3 ms/token in C++ and 19.9 ms/token in the Python runtime at B = 1
(next table; the Python figure is synchronous decode — pipelined, in §3.7a, the two runtimes are equal).

**C++ runtime on the final Qwen3-8B SUC1 file** (`compress_qwen3_8b_final`, 4.221 bpp; R:20261002T191814Z_cpp_generate_check;
64 greedy steps per prompt, contexts of 68–79 positions, median ms/token):

| | C++ separate, pipelined | C++ separate, sync | C++ fused, sync | Python default runtime | Python per-projection plan |
|---|---:|---:|---:|---:|---:|
| Decode B=1 (ms/token) | **13.3** (75 tok/s) | 14.3 | 14.9 | 19.9 | 19.5 |
| Batch prefill, 256 tokens (ms/token) | 3.0 | | | 2.9 | 3.1 |
| Agreement with the Python runtimes | hidden states and logits bit-identical (100%, KL 0), 64/64 identical greedy tokens on each of 3 prompts | | | — | — |
| PPL (8 × 512 windows) | 14.252 | | | 14.250 | 14.250 |
| Memory | mapped file 4.32 GB + 100 MB copied execution arrays + 12 MB KV cache (79 positions); process resident peak 277 MB | | | | |

The PPL difference C++ vs Python (mean |ΔNLL| 0.031) is bf16 logit rounding, as on the earlier file; the two Python
runtimes agree exactly.

*Superseded by §3.7a: the MLX figures in this table use synchronous decode, the C++ figures pipelined decode.*

**Same-session baselines (Mac Studio, 2026-10-02, sequential runs; B:Low-G VQ decode speed (Python runtime) and
same-session baselines, B:Final Qwen3-30B-A3B SUC1 files and C++ runtime (Mac Studio)).** MLX 4-bit and BF16 were
re-measured in the e2e harness in the same Studio session as the C++ checks, run sequentially (one background
service is recorded in `other_processes`):

| Model | C++ SUC, B = 1 (ms/token) | MLX 4-bit, B = 1 (ms/token) | BF16, B = 1 (ms/token) | Sources |
|---|---:|---:|---:|---|
| Qwen3-8B, final file (4.22 bpp) | **13.3** (75.2 tok/s) | **13.3** (75.2 tok/s) | 30.3 (33.0 tok/s) | R:20261002T191814Z_cpp_generate_check, R:20261002T215816Z_e2e_baselines_qwen3_8b |
| Qwen3-30B-A3B, uniform file (4.16 bpp) | **12.9** (77.7 tok/s) | **15.75** (63.5 tok/s) | 21.2 (47.3 tok/s) | R:20261002T232117Z_cpp_check_qwen3_30b_a3b_final, R:20261002T232936Z_e2e_baselines_qwen3_30b_a3b |
| Qwen3-30B-A3B, greedy-4.0 file (4.02 bpp) | 16.5 (60.5 tok/s) | 15.75 | 21.2 | R:20261002T230743Z_cpp_check_qwen3_30b_a3b_final_greedy, same |

The MLX figures replace the earlier cross-session 13.4 ms/token (R:20261002T133705Z_e2e_abl_default_qwen3_8b); they
agree within 1%. **The harnesses still differ**: the C++ figure is the median over 64 greedy steps at 68–79 positions
(prompts of 4–13 tokens), the MLX figure is 32 decode steps after a 128-token prompt (≈ 128–160 positions), median of 3
repeats. At these lengths attention is a small part of a decode step, but the comparison is not matched. The
reading drawn at the time (8B about equal, 30B-A3B uniform −18% time per token) is **withdrawn**: the MLX decode here
was synchronous, the C++ decode pipelined (§3.7a). The 8B baseline run records a dirty working tree.

**C++ runtime on the earlier Qwen3-8B SUC1 file** (`compress_qwen3_8b`, 4.23 bpp, additive M=4 + 0.5% CSR, two-sided;
R:20261002T053015Z_cpp_generate_check, R:20261002T070454Z_cpp_generate_8b_fused, R:20261002T070501Z_cpp_generate_8b_separate):

| | C++ separate, pipelined | C++ fused, pipelined | Python default runtime |
|---|---:|---:|---:|
| Decode B=1 (ms/token) | **24.8** | 26.6 | 30.5 |
| Batch prefill (ms/token) | 4.3 | | 3.5 |
| Bit-identical vs the matching Python plan | hidden states and logits 100%, KL 0, 64/64 greedy tokens (3 prompts) | | — |
| PPL (8 × 512 windows) | 14.437 | | 14.440 |
| Memory | mapped file 4.33 GB + 166 MB execution arrays + 12 MB KV cache (79 positions); process resident peak 391 MB | | |

From the earlier to the final file C++ decode went from 24.8 to 13.3 ms/token. The files differ in codec (additive
M=4 → VQ G=2), rotation (two-sided → one-sided) and residual structure (CSR → columns), and the runtime changed in
between; this comparison does not attribute the gain to one factor. On both files fused q\|k\|v and gate\|up
launches are slower than separate ones in C++ on the M1 Ultra; the fusion should be decided per device, like the
Python fusions.

### 3.7a Matched-harness speed (`suc perf`) — supersedes the C++ vs MLX comparisons above

`suc perf` ([tools.md](tools.md)) times every system with the same tool in one session, sequentially: same raw prompt
(`experiments/perf/promessi_sposi_head.txt`), a full prefill of each context from an empty cache, then 128 greedy
tokens (EOS ignored); warmup 1, 3 repeats, median; decode ms/step = median of steps 2…n. MLX systems decode
**pipelined** (as mlx-lm), like the C++ runtime. Mac Studio M1 Ultra, clean working tree (R:20261003T015734Z_perf_qwen3_8b_final, git 02eb3be;
R:20261003T031427Z_perf_qwen3_30b_a3b_final, git b253a00). The C++ runtime decodes batch 1 only.

**Corrected finding.** The same-session comparison above (and the earlier §1.1/§1.2 headline rows) timed MLX with
synchronous per-token decode in the e2e harness and C++ with pipelined decode. Synchronous decode is slower than how
mlx-lm decodes (Qwen3-8B MLX 4-bit: 13.3 ms/token synchronous vs 10.44 ms pipelined; Qwen3-30B-A3B: 15.75 vs 13.38),
so SUC was overstated: "parity" on the 8B and "22% faster" (−18% time per token) on the 30B-A3B are withdrawn.

**Qwen3-8B, final file (4.22 bpp), B = 1:**

| ctx | MLX 4-bit decode ms (tok/s) | SUC C++ decode ms (tok/s), ratio | SUC C++ separate | SUC Python (pipelined) | BF16 | Prefill tok/s: MLX 4-bit / SUC C++ |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 10.44 (95.8) | 13.89 (72.0), 0.75× | 13.40 (74.6), 0.78× | 13.41 (74.5), 0.78× | 27.75 (36.0) | 588 / 340 (0.58×) |
| 512 | 10.81 (92.5) | 14.23 (70.3), 0.76× | 13.73 (72.8), 0.79× | 13.76 (72.6), 0.79× | 28.10 (35.6) | 754 / 334 (0.44×) |
| 2048 | 11.49 (87.0) | 14.87 (67.3), 0.77× | 14.46 (69.2), 0.79× | 14.52 (68.9), 0.79× | 28.72 (34.8) | 771 / 306 (0.40×) |
| 8192 | 15.66 (63.8) | 16.83 (59.4), 0.93× | 16.46 (60.8), 0.95× | 18.83 (53.1), 0.83× | 32.06 (31.2) | 676 / 232 (0.34×) |

Python runtime at B = 4 / 16 (ctx 128): 0.90× / 0.97× of MLX 4-bit decode; prefill ≈ 0.5× at every batch.

**Qwen3-30B-A3B, final files, B = 1:**

| ctx | MLX 4-bit decode ms (tok/s) | SUC C++ uniform (4.16 bpp), ratio | SUC C++ greedy (4.02 bpp), ratio | SUC Python greedy | BF16 | Prefill tok/s: MLX 4-bit / C++ uniform / C++ greedy |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 13.38 (74.8) | **12.60 (79.4), 1.06×** | 17.11 (58.5), 0.78× | 20.75 (48.2), 0.64× | 19.40 (51.6) | 753 / 376 (0.50×) / 201 (0.27×) |
| 512 | 13.85 (72.2) | 13.69 (73.1), 1.01× | 17.12 (58.4), 0.81× | 21.21 (47.1), 0.65× | 19.78 (50.5) | 1179 / 356 (0.30×) / 200 (0.17×) |
| 2048 | 14.80 (67.6) | **14.14 (70.7), 1.05×** | 18.25 (54.8), 0.81× | 22.14 (45.2), 0.67× | 20.73 (48.2) | 1341 / 325 (0.24×) / 183 (0.14×) |
| 8192 | 16.63 (60.1) | 19.07 (52.4), 0.87× | 22.34 (44.8), 0.74× | 24.23 (41.3), 0.69× | 23.35 (42.8) | 1083 / 225 (0.21×) / 149 (0.14×) |

**Qwen3-0.6B, final file (4.25 bpp), Apple M3 laptop (indicative, `suc perf` smoke, [tools.md](tools.md)):** MLX 4-bit
4.96 ms/step, SUC C++ 7.30 ms (0.68×), BF16 15.23 ms.

Findings: (1) **Decode is competitive only on the MoE at batch 1**: the uniform 30B-A3B file is 6% faster than MLX
4-bit at ctx 128 and 4–5% at 512–2048, but 13% slower at 8192; the greedy file is slower everywhere. (2) **Dense
models are slower**: Qwen3-8B C++ 0.75–0.77× up to ctx 2048 (0.93× at 8192, where attention dominates both);
separate projections are 3–4% faster than the default fused ones on the M1 Ultra. (3) **Prefill is slower
everywhere**: 0.34–0.58× on the 8B, 0.21–0.50× (uniform) and 0.14–0.27× (greedy) on the 30B-A3B; the C++ prefill
path (chunk 64) is the main speed gap. (4) In this harness the pipelined Python runtime equals the C++ runtime on the
8B at ctx ≤ 2048; the C++ advantage reported in the cpp checks (13.3 vs 19.9 ms) came from synchronous Python decode.
(5) One session per model, single run of the tool (3 repeats; batch-1 decode CV ≤ 1% except the 30B-A3B greedy C++
row at ctx 128, 2.3%); other processes are recorded in the results (e.g. screen sharing during the 8B run).

### 3.8 MoE from codes and expert stacks (Qwen3-30B-A3B)

| | Resident weights | Decode B=1 / B=4 (tok/s) | Prefill B=1 (tok/s) | Source |
|---|---:|---|---:|---|
| BF16 | 61.1 GB | — | — | R:20261001T163505Z_moe_e2e_check_Qwen3-30B-A3B |
| MLX 4-bit | 17.2 GB | 62.7 / 146.2 | 794 | same |
| SUC VQ G=4 (2.09 bpw linears, per-expert codebooks, timing codes) | **8.2 GB** | 40.5 / 117.7 | 430 | same |

Expert-stack codebooks (embedding/head BF16, so model bpp is above the codec rate; R:20261002T043700Z_q5x_expert_stacks_qwen3_30b_a3b):

| Codec (+ 0.5% act. residual) | Codebooks | Model bpp | PPL | KL | Encode time |
|---|---|---:|---:|---:|---:|
| RTN 4-bit | — | 4.740 | 8.836 | 0.0571 | 536 s |
| additive M=3 | per expert | 3.485 | 9.547 | 0.1287 | 16,342 s |
| additive M=3 | shared per stack | 3.426 | 9.723 | 0.1341 | 648 s |
| additive M=4 | shared per stack | 4.405 | 9.004 | 0.0564 | 814 s |
| VQ G=4 | shared per stack | 2.446 | 12.239 | 0.3495 | 364 s |

**Routing check: the compressed-vs-reconstructed KL is inherent to BF16 top-k routing**
(R:20261002T185011Z_moe_routing_check_Qwen3-30B-A3B, `benchmarks/moe_routing_check.py`; VQ G=4 codes, 256 tokens,
reference = the reconstructed BF16 model, free-running and teacher-forced, traced forward verified bit-exact against
`model(tokens)`; B:MoE routing check):

| Compared with the reconstructed BF16 model | Final KL | Top-1 | Expert-set agreement (mean of 48 layers, free-running) | Teacher-forced: set agreement / local rel. error |
|---|---:|---:|---:|---|
| SUC compressed (codes) | 0.036 | 0.922 | 0.719 | 0.974 / 1.9e-3 |
| control: same decodes, stochastic BF16 rounding, no SUC code | 0.042 | 0.914 | 0.712 | 0.974 / 2.0e-3 |
| control: same model twice | 0 | 1.000 | 1.000 | 1.000 / 0 |

A perturbation of the same size without any SUC code produces the same expert-set flips and slightly more KL. Flipped
tokens are those whose router margin at the top-k boundary is below the perturbation (layer 0: median margin 0.016
vs 0.063 for unflipped tokens; 5–12% exact BF16 ties at the boundary), and flips accumulate over the 48 layers. The
0.03–0.04 KL is therefore not a defect of the compressed execution; the reconstructed BF16 model is itself one BF16
approximation of the exact computation. One run, 256 tokens.

**Final Qwen3-30B-A3B SUC1 files** (`suc compress experiments/configs/compress_qwen3_30b_a3b_final.yaml` and
`…_final_greedy.yaml`, pinned revision `ad44e77`, every stage layer-streamed, one shared codebook set per expert stack;
R:20261002T212215Z_compress_qwen3_30b_a3b_final, R:20261002T204858Z_compress_qwen3_30b_a3b_final_greedy; C++ checks
R:20261002T232117Z_cpp_check_qwen3_30b_a3b_final, R:20261002T230743Z_cpp_check_qwen3_30b_a3b_final_greedy; same-session
baselines R:20261002T232936Z_e2e_baselines_qwen3_30b_a3b; B:Final Qwen3-30B-A3B SUC1 files and C++ runtime (Mac Studio)):

| | SUC uniform: VQ G=2 K=256 `_xs` + 0.5% act. columns | SUC greedy 4.0 over expert stacks | MLX / RTN affine 4-bit g64 | BF16 |
|---|---:|---:|---:|---:|
| Plan | 18,624 tensors, one codec | VQ G=2 K=256 cols 14,930 · additive M=3 (0.1% res) 2,688 · M=3 (0.5% res) 128 · VQ G=4 768 · BF16 110 tensors | — | — |
| bpp, model / file (all overhead) | 4.164 / 4.174 | **4.015 / 4.025** | 4.50 (MLX) / 4.51 (RTN, Q7) | 16 |
| Size, model / file | 15,892.7 MB / 15,193.9 MiB | **15,323.7 MB / 14,649.7 MiB** | 17.18 GB (MLX) | 61.06 GB |
| WikiText-2 test PPL | 9.141 (+5.1%) | 9.105 (+4.7%) | 9.062 (+4.2%, MLX Phase 0) / 9.071 (+4.3%, RTN Q7) | 8.697 |
| KL vs BF16 / top-1 | 0.0644 / 0.897 | **0.0640** / 0.893 | 0.0812 / 0.887 (RTN, Q7 protocol) | 0 |
| Decode B = 1, C++ separate pipelined, cpp check (ms/token) | 12.9 (77.7 tok/s) | 16.5 (60.5 tok/s) | 15.75 (63.5 tok/s, e2e harness, synchronous: not comparable) | 21.2 (e2e harness, synchronous) |
| **Decode B = 1, ctx 128, matched harness `suc perf` (ms/token)** | **12.60** (79.4 tok/s) | 17.11 (58.5 tok/s) | **13.38** (74.8 tok/s) | 19.40 (51.6 tok/s) |
| **Prefill B = 1, ctx 128 / 2048, matched harness (tok/s)** | 376 / 325 | 201 / 183 | 752 / 1341 | 799 / 1838 |
| C++ separate sync / fused sync (ms/token) | 15.0 / 15.0 | 18.2 / 18.2 | | |
| Python runtime decode B = 1 (ms/token) | 24.1 | 23.8 | | |
| Batch prefill, 256 tokens (ms/token): C++ / Python | 2.69 / 3.71 | 4.95 / 1.69 | | |
| Agreement C++ vs Python | hidden states and logits bit-identical, 64/64 greedy tokens on each of 3 prompts | same | | |
| PPL 8 × 512 windows, C++ / Python | 12.589 / 12.546 | 12.658 / 12.640 | | |
| Memory during C++ decode | file 15.93 GB mapped + 520 MB copied execution arrays; process resident peak 1.63 GB | 15.36 GB + 457 MB; peak 1.47 GB | decode peak 16,609 MiB | 58,381 MiB |
| `suc compress` time / max stage peak | 1994 s / 7.80 GiB (encode) | 5443 s (candidates 3497 s) / 8.05 GiB (candidates) | | |

Findings:

1. **Quality per bit**: both files have lower KL than RTN 4-bit (−21%) with 7.5% (uniform) and 10.8% (greedy) fewer
   bytes than MLX 4-bit, but slightly higher PPL (+0.4 / +0.8 points of ΔPPL). Single seed; the PPL gap is within the
   ±1% PPL noise measured on the 8B, the KL gap is not. On the MoE the greedy allocator again helps a little (same KL
   at 0.15 bpp less), as in Q7.
2. **Speed** (matched harness, §3.7a): the uniform file decodes at 12.60 ms/token in C++, 6% faster than MLX 4-bit
   (13.38 ms) at ctx 128 and 4% at ctx 2048, but its prefill is 2–4× slower. The earlier "18% less time per token"
   compared against synchronous MLX decode and is withdrawn. The greedy file is slower (17.11 ms); it differs by 110 BF16
   attention tensors and three additional codecs (additive M=3, VQ G=4), which are the likely cause (not profiled). The
   Python runtime (greedy file, pipelined) runs at 20.75 ms, 0.64× of MLX 4-bit. Compared with the earlier MoE run (VQ G=4 timing codes before K6/R5: 24.7 ms/token in Python),
   the Python runtime is about the same on the heavier final codec.
3. **C++ MoE prefill**: the greedy file's batch prefill in C++ (4.95 ms/token) is 2.9× slower than Python; the C++ MoE
   tiled prefill path is the known gap (§5.2).
4. **PPL C++ vs Python** differs more than on the 8B (mean |ΔNLL| 0.064 / 0.067 vs 0.031). Decode is bit-identical;
   the difference is in batch-prefill rounding, amplified by top-k routing flips (routing check above).
5. **Memory**: at ≈ 4 bpp the 30B-A3B needs ≈ 16.8–17.6 GB in total (mapped file + process peak), about the same as
   MLX 4-bit's decode peak (17.4 GB), so it cannot fit a 16 GB Mac (no run on a smaller Mac was made). Fitting there
   needs ≈ 2–2.5 bpp, where the measured quality loss is large (VQ G=4 expert stacks: KL 0.35, PPL 12.24, table above).

### 3.9 Entropy coding (storage only)

| File | Stored | Best generic coder | SUC entropy coding (rANS indices + Rice positions + generic rest) | Source |
|---|---:|---:|---:|---|
| 0.6B dev (M=4 + res, 4-bit embedding) | 320.36 MB, 4.300 bpp | 304.68 MB, 4.089 bpp | 303.24 MB, **4.070 bpp (−5.3%)** | R:20261002T012611Z_entropy_coding_qwen3_0.6b |
| 8B (M=4 + CSR res, 4-bit head) | 4326.5 MB, 4.226 bpp | 4221.5 MB, 4.123 bpp | 4139.4 MB, **4.043 bpp (−4.3%)** | R:20261002T142930Z_entropy_coding_qwen3_8b |

Additive indices have an order-0 entropy of 99.0% of log₂K. Context barely helps: on the 8B, held out, the
previous-codebook context gives 7.80 bits vs 7.92 order-0 (−1.5%), and the left and upper contexts lose (7.95). The savings come from residual positions (Rice: −64% of position bytes on the 8B), scales
and the affine embedding/head. Decoding the 8B indices with the numpy rANS takes 43 s. Inside the token loop it would
cost the whole decode time per token. Verdict (spec §36 Failure C): storage/distribution only; the execution layout
stays fixed-width.

### 3.10 Published-baseline comparison (no local AQLM/QuIP# run)

Full details: [comparison_published.md](comparison_published.md). Published bits count only the block linears
(embedding and LM head FP16). SUC rows give linear bpw.

| ≈ bpw | Published, Qwen3-8B | SUC Qwen3-8B (linear bpw, ΔPPL) | Reading |
|---|---|---|---|
| 2 | QuIP# +27.4%, AQLM 2×8 +87.9%, QTIP +6.9%, PV-Tuning +9.6%, LC-QAT +5.2% (arXiv:2606.10531 Tab. 1) | VQ G=4 + res 2.18: +90.8% (no FT), +69.7% (block FT); greedy 2.5: +32.0% | **SUC clearly behind** |
| 3 | no AQLM/QuIP#/QTIP number; AWQ per-channel +53.5% (arXiv:2505.02214) | greedy 3.0 (Q7): +13.2%; M=3 + res 3.18: +14.0%; VQ G=2 K=64 3.37: +9.3%, G=2 K=32 2.95: +19.0% (whole-model bpp, Q5c) | ahead of scalar methods; vs VQ methods undecidable without a local run |
| 4 | GPTQ g128 +2.6% (nominal 4 bits), AWQ g128 +16.4% | M=4 + res 4.18: +3.3–3.7%; VQ G=2 4.22: +2.9% (whole-model bpp) | about level with GPTQ g128 (different pipeline); cannot claim to beat it |

D3 requires an AQLM or QuIP# reference for every claim at ≤ 3 bpw. It is satisfied only at 2 bpw, through published
numbers. A local QuIP#/QTIP run with the SUC protocol is still missing.

### 3.11 Consolidated matrix (spec §49), Qwen3-8B

Quality from protocols A/B (KL on 16 validation sequences), except where marked †. Speed and memory from the e2e
harness (prompt 128, B = 1), except where marked ‡. "n/m" = not measured.

| Representation | bpp | Size | PPL | KL | Task score | Decode tok/s | TTFT | Peak RAM (decode) |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| BF16 | 16.00 | 16.38 GB | 9.717 | — | n/m | 34.0 | 219 ms | 15.42 GiB |
| INT8 (MLX affine g64) | 8.50 | 8.70 GB | 9.726 | 0.0018† | n/m | 55.7‡ | 204 ms‡ | 8.91 GiB‡ |
| INT4 (MLX affine g64) | 4.50 | 4.61 GB | 10.082 | 0.092 (RTN 4-bit) | n/m | 74.9; same session 75.2 (13.3 ms) ¤ | 209 ms | 4.60 GiB |
| Reference 4-bit (GPTQ/AWQ) | — | — | published only: GPTQ g128 +2.6% | — | n/m | n/m | n/m | n/m |
| SUC scalar (rotated K=16 + res) | 4.22 | 4.32 GB | 10.074 | 0.055 | n/m | n/m | n/m | n/m |
| **SUC vector, final (VQ G=2 K=256 + res), SUC1 file** | 4.22 | 4.32 GB | **9.994** | **0.049** | n/m | 75.2 (13.3 ms, C++ §) / 50.3 (19.9 ms, Python §); e2e timing codes 55.2 ¤ | 382 ms ¤ | 4.32 GB mapped + 0.28 GB (C++ §); 4.48 GiB (Python ¤) |
| SUC vector (VQ G=2 K=64 + res) | 3.37 | 3.45 GB | 10.618 | 0.124 | n/m | 49.4 ¤ | 488 ms ¤ | 3.67 GiB ¤ |
| SUC vector (VQ G=2 K=32 + res) | 2.95 | 3.02 GB | 11.560 | 0.234 | n/m | n/m (not in the low-G run) | n/m | n/m |
| SUC vector (VQ G=4 + res, head 4-bit) | 2.53 | 2.59 GB | 14.15 / 14.43 (Q5c) | 0.429 / 0.430 | n/m | 55.7 | 265 ms | 2.87 GiB |
| SUC additive (M=4, no residual) | 4.08 | 4.18 GB | 10.632 | 0.144 | n/m | 37.6 | 386 ms | 4.18 GiB |
| SUC + residual (M=4 + 0.5% act. cols) | 4.22 | 4.33 GB | 10.121 | 0.068 | n/m | 34.6 | 405 ms | 4.49 GiB |
| SUC + semantic allocation (greedy 3.5 / 3.0) | 3.50 / 3.00 | ≈ 3.58 / 3.07 GB (from bpp) | 10.65 / 11.71 | 0.115 / 0.244 | n/m | n/m (mixed plans not benchmarked) | n/m | n/m |
| SUC + entropy storage (8B file) | 4.04 (file) | 4.14 GB | as stored codec | as stored codec | n/m | not executable (decode at load ≈ 43 s numpy) | — | — |
| SUC direct Metal, C++ (earlier M=4 + CSR file, 4.23 bpp) | 4.23 | 4.33 GB | 14.437 (8 × 512 windows) | — | n/m | 40.3 (24.8 ms) § | n/m | 4.33 GB mapped + 0.18 GB |

† Phase 0 protocol (KL on the first 512 test tokens). ‡ Phase 0 harness (128 decode tokens, synchronous).
§ C++ check harness (`cpp_generate_check.py`, 64 greedy steps, 68–79 positions; R:20261002T191814Z_cpp_generate_check,
R:20261002T053015Z_cpp_generate_check): not the same harness as the other decode figures (§3.7).
¤ Low-G e2e run and same-session baselines (R:20261002T212251Z_e2e_lowg_qwen3_8b, R:20261002T215816Z_e2e_baselines_qwen3_8b),
same harness as the other e2e figures, timing codes; peak = B=1 decode peak. Sizes of the Q5c rows
are from R:20261002T175941Z_q5c_lowdim_vq_qwen3_8b (Size column, MB).

### 3.12 Dev model on the M3 laptop (16 GB)

"Small Mac" data point: Qwen3-0.6B compressed with the final recipe and run by the C++ runtime on the **dev laptop**
(Apple M3, 16 GB, on AC power). These timings are valid for that machine only, not Mac Studio numbers. Sources:
R:20261002T202145Z_compress_qwen3_0.6b_final, R:20261002T203532Z_compress_qwen3_0.6b_final_3bpp (`suc compress`, run
dirs on the local disk, 885 s / 818 s), RTN baselines R:20261002T204839Z_final_baseline_qwen3_0.6b (`suc quantize`, same
quality protocol), C++ checks R:20261002T205833Z_cpp_check_qwen3_0.6b_final_m3, R:20261002T210018Z_cpp_check_qwen3_0.6b_final_3bpp_m3,
MLX/BF16 timings R:20261002T210018Z_e2e_final_baseline_qwen3_0.6b_m3; B:Final Qwen3-0.6B SUC1 files on an Apple M3 (16 GB).
Configuration as the final 8B file; the **tied** embedding / LM head (26% of this model's parameters) is affine
4-bit g64, which lifts the whole-model bpp more than on the 8B. BF16 PPL in this protocol: 20.973.

| | SUC VQ G=2 K=256 + res (C++) | SUC VQ G=2 K=64 + res (C++) | MLX / RTN affine 4-bit g64 | RTN affine 3-bit g64 | BF16 |
|---|---:|---:|---:|---:|---:|
| bpp, model (file) | **4.251** (4.257) | **3.510** (3.517) | 4.501 | 3.502 | 16 |
| File / weights | 302.5 MiB | 249.9 MiB | 319.8 MiB (MLX resident) | 260.9 MB | 1136.9 MiB |
| WikiText-2 test PPL | **22.832** (+8.9%) | **27.806** (+32.6%) | 24.295 (+15.8%) | 62.090 (+196%) | 20.973 |
| KL vs BF16 / top-1 | **0.092** / 0.853 | **0.308** / 0.751 | 0.190 / 0.795 | 1.087 / 0.546 | 0 |
| Decode B = 1, cpp check (ms/token) | 6.95 (144 tok/s) | 7.07 (141 tok/s) | 6.72 (149 tok/s, e2e harness, synchronous: not comparable) | — | 16.88 (59 tok/s) |
| Decode B = 1, matched harness `suc perf` smoke (ms/step, indicative) | 7.30 (0.68× of MLX 4-bit) | — | 4.96 | — | 15.23 |
| Python runtime decode B = 1 (ms/token) | 12.6 | 11.2 | | | |
| Process memory during decode | mapped file + 97 MiB footprint | mapped file + 95 MiB | decode peak 551 MiB | — | 1348 MiB |
| C++ vs Python | bit-identical (hidden states, logits, 64/64 greedy tokens, 3 prompts); PPL 8 × 512: 32.465 vs 32.479 | bit-identical; 39.764 vs 39.738 | | | |

Findings: (1) at 0.25 bpp less than MLX 4-bit the 4.25-bpp file halves KL (0.092 vs 0.190) and cuts ΔPPL from +15.8% to
+8.9%; the 3.5-bpp file beats RTN 3-bit at equal size by a wide margin (KL 0.31 vs 1.09) but is worse than RTN 4-bit
(+32.6% vs +15.8% PPL), so on this model too ≈ 3.5 bpp is not "close to 4-bit". (2) The C++ runtime decodes the 4-bpp
file at 0.68× of MLX 4-bit's speed on the M3 in the matched `suc perf` smoke (7.30 vs 4.96 ms/step, indicative) and
2.1× faster than BF16 (15.23 ms). The earlier "within 3.4%" (6.95 vs 6.72 ms) compared against synchronous MLX
decode and is withdrawn. The 3.5-bpp file is not faster (7.07 ms): the 0.6B is
launch-bound on the M3, so fewer index bits save memory, not time. (3) The whole compressed model needs ≈ 0.4 GB
(file + ≈ 96 MiB process footprint) vs 1.35 GB peak for BF16 under MLX. (4) The small model tolerates compression much
less than the 8B (4.2 bpp: +8.9% vs +2.9% PPL). Qwen3-1.7B was not run.

### 3.13 QA side-by-side (qualitative)

Greedy chat generations of the final SUC files next to BF16, 20 fixed prompts (`experiments/qa/prompts.json`: 5
explanations and 5 arithmetic word problems, each in Italian and English; explanations without thinking, 300-token
limit; arithmetic with thinking, 1,500-token limit). Files: `experiments/qa/qa_8b_side_by_side.md` (SUC = final 8B
SUC1 file, C++ runtime on the M3 laptop; BF16 = mlx-lm on the Studio) and `experiments/qa/qa_30b_side_by_side.md`
(SUC = greedy-4.0 30B-A3B file, C++ on the Studio; BF16 = mlx-lm on the Studio); raw outputs `suc_8b_m3.json`,
`bf16_8b_studio.json`, `suc_30b_greedy_studio.json`, `bf16_30b_studio.json` in the same directory; B:QA side-by-side,
Qwen3-8B final file vs BF16, B:Final Qwen3-30B-A3B SUC1 files and C++ runtime (Mac Studio).

| | Qwen3-8B, SUC 4.22 bpp | Qwen3-30B-A3B, SUC greedy 4.02 bpp |
|---|---|---|
| Arithmetic prompts with a final answer (of 10) | 7, all correct and equal to BF16 (200 km, 29, 36, 61) | 8, all correct and equal to BF16 (391, 200, 29, 36, 61) |
| Arithmetic prompts cut by the 1,500-token limit | 3 (17 × 23 IT, 17 × 23 EN, 15% EN); the visible reasoning reaches the correct value (391, 391, 36) in each | 2 (17 × 23 EN, 15% EN); reasoning correct up to the cut |
| Same prompts cut for BF16 | 17 × 23 EN and 15% EN appear cut as well (the BF16 raw files do not record a stop reason) | 15% EN |
| Explanations | factually equivalent to BF16, often nearly word-identical (e.g. the sky-colour answer); 3 of 10 SUC answers reach the 300-token limit, BF16 answers of the same prompts are of similar length | same reading |
| Incoherent or hallucinated output | none observed | none observed |

The truncation counts are read from the raw files (an SUC answer without an end-of-turn token hit the limit); for the
8B this gives one more cut answer than the count in benchmarks.md. This is a **qualitative check on 20 prompts with
greedy decoding**: it supports "no visible breakage at ≈ 4 bpp", not "same capability" (§48; no task score exists,
§4 Q4).

---

### 3.14 Frontier models (Qwen3.8-Flash-Next, DeepSeek-V4-Flash, GLM-5.3-Flash)

After the project's target models, the unattended pipeline compressed three large MoE checkpoints on the Mac Studio
(111 GB available for the model at run time), all stages layer-streamed. Details, every run and every result id:
B:Qwen3.8-Flash-Next, B:DeepSeek-V4-Flash, B:GLM-5.3-Flash, B:SSD read bandwidth, B:Frontier models — summary.
All models side by side against their original full-precision versions (size, memory, TTFT, prefill, decode, quality, with charts): [compression_report.md](compression_report.md).

| | Qwen3.8-Flash-Next (qwen4_exp MoE) | DeepSeek-V4-Flash (MoE) | GLM-5.3-Flash (MoE) |
|---|---|---|---|
| Parameters | 176.9 B | 284.3 B | 313.3 B |
| SUC file (plan) | ngram4: uniform VQ G=2 + 0.5 % col. residual, 4-bit n-gram tables | native_max: greedy, experts G=4 / M=3 / native MXFP4 | greedy: G=4 / M=3 / G=2 |
| bpp (whole file, all overhead) | 4.23 | 3.06 | 3.06 |
| File size | 93.5 GB | 108.7 GB | 119.9 GB |
| Runtime memory | resident: 66.6 GB mapped + 2.2 GB arrays (n-gram tables read from the file) | resident: 107.4 GB mapped (native file) + 7.9 GB process | **expert streaming**: 5.8 GB mapped + 109.2 GB process (102 GB expert pool), 91.6 % expert hit rate |
| PPL: SUC / reference | 5.055 / 4.695 (+7.7 %) | 5.362 / 4.779 (+12.2 %) | 4.009 / 3.268 (+22.7 %) |
| KL vs reference (top-1) | 0.145 (0.856) | 0.225 (0.826) | 0.261 (0.815) |
| Same-protocol 4-bit baseline | MLX affine 4-bit (4.57 bpp on the encoded set): PPL 5.222, KL 0.202 | none (MLX 4-bit ≈ 160 GB, does not fit) | none (MLX 4-bit ≈ 176 GB, does not fit) |
| C++ decode, batch 1 | 37.6 ms/token (`suc1-generate`, ctx ≤ 4096; no matched `suc perf`) | 83.2 / 87.3 / 94.4 ms/token at ctx 128 / 2k / 8k (`suc perf`, native file) | 135.5 / 142.3 / 146.5 ms/token at ctx 128 / 2k / 8k (`suc perf`, streamed) |
| `suc eval` core suite (graded) | 30 / 30 (also MLX 4-bit 30 / 30) | 30 / 30 (Python and C++) | 27 / 30 (C++; 3 answers correct, then no end of turn) |

Reference: BF16 checkpoint (Qwen3.8, GLM; PPL from the compress run, BF16 itself does not fit), exact BF16
dequantization of the FP8 / MXFP4 checkpoint (DeepSeek). PPL: WikiText-2 test, 145 × 2048 (Qwen3.8) or 70 × 4096
windows (DeepSeek, GLM); KL: 16 × 512 validation tokens. Single seed. DeepSeek C++ speed and eval were measured on the
`native` file (3.02 bpp, KL 0.248), not on native_max (not run in C++). No AQLM / QuIP# numbers exist for these
models (local or published, [comparison_published.md](comparison_published.md)).

Key findings:

- **Native MXFP4 as an allocation candidate (DeepSeek).** The routed experts ship in MXFP4. After rotation every SUC
  expert codec sees a near-Gaussian source and gains nothing from that; VQ G=2 (4.16 bpp, rel. MSE 0.0078) is strictly
  dominated by passing the MXFP4 bytes through (4.25 bpp, error 0), and the greedy took the native candidate 1:1.
  Best file at 111 GB (native_max, 3.06 bpp): KL 0.225, PPL +12.2 % (R:20261006T115604Z_compress_deepseek_v4_flash_native_max).
- **Routed experts dominate the DeepSeek KL.** With all experts native (4.36 bpp, 155 GB, expert streaming) KL is 0.036
  and PPL +1.3 % (R:20261006T153009Z_compress_deepseek_v4_flash_native_stream): about 84–86 % of the KL of the
  ≈ 3-bpp files comes from re-quantizing experts below 4 bits. A lattice-aware sub-4-bit codec for FP4 experts is the
  obvious next codec (not built).
- **Fisher-predicted vs measured (negative).** Over four ≈ 3-bpp DeepSeek plans the predicted loss spans 5.5×, the
  measured KL 11 %; the plan predicted 5.5× worse (flat_max) is 7 % worse in KL, better in ΔCE and top-1. Neither the
  diagonal Fisher nor the rel.-MSE proxies order these plans; the additive model is not valid at the error size of
  the G=4 tier (§7 item 3). Single seed.
- **Residency-set fix (GLM).** Streaming GLM's experts from SSD, the C++ runtime first decoded at 2.58 s/token with 0.14 s
  of GPU time: each per-layer command buffer re-made its expert pool resident. Keeping pools, file views and large
  buffers in one `MTLResidencySet` gives 153 ms/token in `suc1-generate` and 135–147 ms/token in `suc perf` (16.8×).
  Next-layer prefetch raised the hit rate 91.6 → 96.3 % but was slower (269 ms/token; negative result).
- **View-splitting hang (DeepSeek).** Files above `maxBufferLength` were covered by half-overlapping GPU views whose
  residency (86.6 + 64.1 GB) exceeded the 115.3 GB working set; the first command buffer was never scheduled. Views
  are now cut only between mapping units. A later commit silently reverted this fix (stale working tree), which broke
  the first GLM runs with the residency set; restored in a5d1eb8.
- **SSD:** 7.3 GB/s (2.6-MB reads) to 9.9 GB/s (9.4-MB reads) uncached with ≥ 4 threads; `mmap` page faults are
  1.7–6× slower, so experts are read with `pread` into pools (R:20261005T023119Z_ssd_read_bench_studio).
- **Long context (DeepSeek native, Python):** KL vs reference does not grow with context (0.248 at 8k → 0.207 at 64k),
  needles 12/12 as the reference (R:20261004T201726Z_long_context_deepseek_v4_flash).

## 4. Answers to the spec §50 questions

| # | Question | Answer (Qwen3-8B unless stated) |
|---|---|---|
| 1 | Minimum actual bpw achieved? | Depends on the quality bar. Better than MLX 4-bit on PPL and KL: **4.22 bpp** (VQ G=2 K=256 + res, the final file). ≤ +10% PPL: **3.37 bpp** (VQ G=2 K=64 + res, +9.3%). ≈ +20%: 2.95 bpp (VQ G=2 K=32, +19.0%). Functional but degraded: 2.53 bpp (+46…+49%), 2.0 bpp (+307%). ≤ 1.75 bpp: broken. 30B-A3B: final files 4.02 bpp at +4.7% / KL 0.064 (greedy) and 4.16 bpp at +5.1% (uniform VQ G=2), better than RTN 4-bit in KL, not in PPL; 3.04 bpp at +13.5%, 2.55 bpp at +21% (Q7). Low-G VQ below 4 bpp not measured on the MoE. |
| 2 | True compressed size incl. all overhead? | Final 8B SUC1 file: **4,322.2 MB (4122 MiB), file bpp 4.2216** vs model bpp 4.2210 (header + directory + padding = 0.0006 bpp), against 4,607.7 MB for MLX 4-bit (R:20261002T191301Z_compress_qwen3_8b_final). Earlier M=4 + CSR file: 4,325.9 MB, 4.226 file bpp. Resident weights in the Python runtime: 4.03 GiB (M=4 + res), 2.28 GiB (VQ G=4); the C++ runtime maps the file (4.32 GB) and holds 0.28 GB more at peak. 30B-A3B final files: 14,649.7 MiB (greedy, file bpp 4.025) and 15,193.9 MiB (uniform, 4.174) vs 17.18 GB for MLX 4-bit; C++ process peak 1.47 / 1.63 GB beyond the mapping (R:20261002T204858Z_compress_qwen3_30b_a3b_final_greedy, R:20261002T212215Z_compress_qwen3_30b_a3b_final, §3.8). 8.2 GB from codes with VQ G=4 (≈ 2.1 bpw, low quality). 0.6B: 302.5 / 249.9 MiB (§3.12). |
| 3 | What quality was lost? | §3.3, §3.5: +2.9% PPL / KL 0.049 at 4.22 bpp; +9.3% / 0.124 at 3.37; +19.0% / 0.234 at 2.95; +46…+49% / 0.43 at 2.53. |
| 4 | Which tasks were affected? | **Not measured.** Only WikiText-2 PPL and KL on held-out validation text (mixed natural / code / math / instruction) were evaluated. No downstream, coding, math or OOD task score exists, so no claim about "same capability" is possible (§48). |
| 5 | Inference throughput? | §3.7a (matched harness `suc perf`, B = 1, ctx 128). Final 8B file: **72.0 tok/s in C++ (13.89 ms/token), 74.5 in the Python runtime, vs 95.8 for MLX 4-bit (10.44 ms): 0.75×**; prefill 340 vs 588 tok/s. 30B-A3B: uniform file **79.4 tok/s vs 74.8 (1.06×)**, greedy 58.5; prefill 376 vs 752 tok/s. 0.6B on the M3: 0.68× (indicative). Older, synchronous-harness figures: Final codec in the Python e2e harness: 55.2 / 113.1 / 165.5 tok/s at B = 1 / 4 / 16 vs 74.4 / 137.8 / 194.3 for MLX 4-bit (same run); G=2 K=64 + res 49.4 / 85.1 / 107.2. the earlier C++ vs MLX figures (8B 75 vs 75.2, 30B-A3B 77.7 vs 63.5, 0.6B 144 vs 149 tok/s) compared pipelined C++ with synchronous MLX and are withdrawn. |
| 6 | First-token latency? | 128-token prompt, B=1, Python runtime: 240–488 ms (SUC; final codec 382 ms, G=2 K=64 488 ms) vs 208 ms (MLX 4-bit) and 219 ms (BF16) (R:20261002T050049Z_e2e_r5_after_qwen3_8b, R:20261002T212251Z_e2e_lowg_qwen3_8b). Prefill is 14–58% slower than MLX 4-bit. Matched harness (§3.7a): TTFT at ctx 128 is 377 ms (C++) vs 218 ms (MLX 4-bit) on the 8B, 341 vs 170 ms on the 30B-A3B uniform file; C++ prefill is 0.34–0.58× (8B) and 0.21–0.50× (30B-A3B uniform) of MLX 4-bit. |
| 7 | Peak memory? | Decode peak B=1: 2.56–4.49 GiB (SUC) vs 4.60 (MLX 4-bit), 15.42 (BF16). C++ on the final file: 4.32 GB mapped + 0.28 GB process peak; 30B-A3B final files 15.36–15.93 GB mapped + 1.47–1.63 GB (MLX 4-bit decode peak 17.4 GB). Pipeline: final 8B `suc compress` peak 15.22 GiB (encode), other stages 3.2–8.1 GiB; final 30B-A3B `suc compress` max stage peak 7.80 GiB (uniform) / 8.05 GiB (greedy, candidate scoring) for a 61 GB BF16 model; streamed Fisher 31 GB (8B); block fine-tuning 30–34 GiB (8B). |
| 8 | Direct compressed execution faster or slower than conventional 4-bit? | Matched harness (§3.7a), batch 1: **decode 6% faster only on the uniform 30B-A3B file** (12.60 vs 13.38 ms/token at ctx 128; 4% at 2048; 13% slower at 8192); **slower on the dense 8B (0.75×)**, the greedy 30B-A3B file (0.78×) and the 0.6B on the M3 (0.68×, indicative); **prefill slower everywhere** (0.14–0.58×). The earlier "parity" (8B) and "−18%" (30B-A3B) came from synchronous MLX timing and are withdrawn. Older Python-runtime figures (both sides synchronous): slower at batch 1 in every configuration (final codec 74%, best 96–98% for residual-free VQ G=4 / G=2 K=16, additive M=4 + res 46%; 30B-A3B 65%). At B = 4 / 16 the final codec is slower (82% / 85%); **faster at B = 16** only for residual-free VQ G=4 (+35%) and G=2 K=16 (+20%). Slower for additive codebooks at every batch. The ≈ 3-bpp G=2 K=64 codec is slower than the 4-bpp one. |
| 9 | Which components created the improvement? | Rotation (−29% KL vs none), activation-selected residuals (−53% KL vs none, −52% vs magnitude), low-dimensional single codebooks (−27% KL vs additive M=4 at 4.22 bpp, −19% vs M=3 at 3.37 bpp), keeping the LM head at affine 4-bit (Q7: −35% KL), Fisher allocation on the MoE (−20% KL at 3 bpp). Runtime: one-sided rotation, staged codebooks in fused launches (+60% for additive), column residuals, the blocks-layout rotation (6.8 → ~0 ms/step). |
| 10 | What failed? | §1.3: ≤ 2.5 bpw quality vs QuIP#/QTIP/PV-Tuning; greedy allocation below 3 bpp; fine-tuning transfer to the 8B; residual-stream rotation folding; batch-1 decode speed of dense models (C++ 0.75× of MLX 4-bit on the 8B, matched harness) and prefill speed everywhere (0.14–0.58×); the earlier C++ "parity" / "22% faster" claims (mismatched harness, withdrawn); speed of 6-bit indices (≈ 3.4 bpp slower than 4.2 bpp); ≈ 4-bpp 30B-A3B PPL vs RTN 4-bit; entropy coding as a runtime gain; the partial-product table kernel (K1); fusing the rotation into SwiGLU (slower); fused projections in C++ on the M1 Ultra (slower). |
| 11 | Which assumptions did not hold? | (a) "Weights contain repeated patterns": block directions are isotropic; the VQ gain is space-filling only. (b) Layer-local error predicts behaviour: ρ = 0.09 on the 0.6B. (c) Additive multi-codebooks are needed for useful rates: one low-dimensional codebook is better at every rate measured (G=2 from 2.95 to 4.22 bpp, G=4 at 2.0–2.5 bpp). (d) Fewer bytes give faster decode: at 4 bits the runtime is the limit; on the MoE bytes barely matter. (e) Allocation is the main lever: only on the MoE. (f) Entropy coding of indices pays: indices are near-uniform. (g) Folding rotations into the residual stream is free: it breaks quality. (h) Data-free codebooks beat RTN at equal weight MSE: RTN keeps super weights exact, codebooks smear them. (i) One kernel configuration fits all Apple GPUs: codebook placement and fusion flip between M1 Ultra and M3. |
| 12 | Next research direction? | §7. |

---

## 5. Success level reached, and what was not done

### 5.1 Success levels (spec §47)

| Level | Criterion | Status | Evidence |
|---|---|---|---|
| **A** | vector codebook works | met | Phase 1, Q1–Q5c |
| | complete model reconstructed for evaluation | met | SUC1 files decoded and evaluated on three models (R:20261002T191301Z_compress_qwen3_8b_final, R:20261002T212215Z_compress_qwen3_30b_a3b_final, R:20261002T204858Z_compress_qwen3_30b_a3b_final_greedy, R:20261002T202145Z_compress_qwen3_0.6b_final) |
| | < 4 bpw demonstrated | met | 3.37 bpp at +9.3% PPL (8B, VQ G=2 K=64); 3.04 bpp at +13.5% (30B-A3B greedy, Q7) |
| | semantic error measured | met | KL vs BF16 everywhere; Fisher proxy validated against measured KL |
| | reproducible | met | every result from a config with full provenance; seed repeat (±3% KL); the final 8B compress run reproduces the ablation's PPL/KL exactly (same seed); every final file re-read bit-exactly |
| **B** | around 3 bpw or below | met **only without the quality criterion** | 2.95 bpp (8B, VQ G=2 K=32), 3.04 bpp (30B-A3B greedy) |
| | quality close to a strong 4-bit baseline | **met only at ≈ 4.0–4.2 bpp, not at ≈ 3 bpp** | ≈ 3 bpp: 8B +19.0% PPL / KL 0.234 (2.95 bpp), +9.3% / 0.124 (3.37 bpp); 30B-A3B +13.5% / 0.129 (3.04); 0.6B +32.6% / 0.308 (3.51) — against +3.8% / 0.092 (8B), +4.3% / 0.081 (30B-A3B), +15.8% / 0.190 (0.6B) for 4-bit. At ≈ 4 bpp: 8B 4.22 bpp better than 4-bit (+2.9% / 0.049); 30B-A3B 4.02 bpp better in KL (0.064), slightly worse in PPL (+4.7% vs +4.3%); 0.6B 4.25 bpp better (+8.9% / 0.092) |
| | direct Metal inference works | met | Python and C++ runtimes generate from codes on 0.6B, 8B and 30B-A3B; C++ bit-identical to Python on every final file |
| | no full FP16 reconstruction in the hot path | met | kernels read indices and codebooks; Ŵ never materialised; C++ maps the file and adds 0.1–1.6 GB |
| | runtime competitive | **partly: decode only on the 30B-A3B MoE at B = 1 (uniform file); not on the dense 8B; prefill not competitive; not shown at ≈ 3 bpp** | Matched harness `suc perf`, C++, B = 1, ctx 128: 30B-A3B uniform 12.60 vs 13.38 ms/token (1.06×; 1.05× at ctx 2048, 0.87× at 8192); 30B-A3B greedy 17.11 ms (0.78×); 8B 13.89 vs 10.44 ms (0.75×); 0.6B (M3, indicative) 7.30 vs 4.96 ms (0.68×). Prefill 0.34–0.58× (8B), 0.21–0.50× (30B-A3B uniform). At ≈ 3 bpp the C++ runtime ran only the 0.6B 3.5-bpp file (7.07 ms, M3); on the 8B the ≈ 3.2-bpw G=2 K=64 codec was measured in Python only (66% of MLX 4-bit at B = 1, 55% at B = 16). B > 1: Python only, final codec 82–85% of MLX 4-bit |
| **C** | ≈ 2 bpw with small degradation, direct execution, competitive throughput, much lower memory | **not met** | 2.0–2.5 bpp costs +46% to +307% PPL on the 8B; far behind QuIP#/QTIP at 2 bpw (§3.10) |

**Level A is reached. Level B is not reached as a whole**, because its bit and quality criteria are met only at
different rates. The complete evidence:

- **What holds at ≈ 4.0–4.2 bpp on two serious models**: quality at or better than a strong 4-bit baseline at 6–11%
  fewer bytes (8B: PPL +2.9% vs +3.8%, KL 0.049 vs 0.092; 30B-A3B: KL 0.064 vs 0.081, PPL +4.7% vs +4.3%), executed
  directly from codes with no FP16 reconstruction. Decode speed is competitive **only on the 30B-A3B MoE at batch 1**
  (uniform file 6% faster than MLX 4-bit at ctx 128, 4% at 2048); the dense 8B decodes 25% slower (0.75×), and prefill
  is slower on every model (2–4× on the 30B-A3B). The execution half of Level B is therefore met only in part (direct
  Metal inference and no reconstruction: yes; competitive runtime: MoE decode only), at a 4-bit-class rate, not at ≈ 3 bpw.
- **Corrected finding**: an earlier version of this assessment read "MLX 4-bit speed (8B) or below its time per
  token (30B-A3B)". That came from comparing pipelined C++ decode with synchronous MLX decode; the matched harness
  (§3.7a) replaces it.
- **What fails at ≈ 3 bpp**: quality is not close to 4-bit (8B +9.3% at 3.37 bpp and +19.0% at 2.95 bpp vs +3.8%;
  30B-A3B +13.5% vs +4.3%), and speed is not shown to be competitive there: the 6-bit-index codec is slower than the
  8-bit one in Python, and no C++ run exists at ≈ 3 bpp on the 8B or the 30B-A3B.
- **Caveats on the speed criterion**: one `suc perf` session per model (3 repeats each); batch > 1 is measured only in
  the Python runtime, where the final codec is slower than MLX 4-bit at every batch (8B: 0.90× / 0.97× at B = 4 / 16).
- **Not covered by Level B but needed for the project goal**: task-level quality (§4 Q4: PPL, KL and a 20-prompt
  qualitative check only) and the 16 GB target: at ≈ 4 bpp the 30B-A3B needs ≈ 17 GB, like MLX 4-bit.

Level C is not reached.

### 5.2 Not done

| Item | Spec | Status |
|---|---|---|
| Adaptive / multiplicative-weights allocator | §29, §40 | not implemented; only the static Fisher-greedy allocator (§39) exists |
| Local AQLM / QuIP# / QTIP run on Qwen3-8B with the SUC protocol | D3, §42 | not run; published numbers only (2 bpw) |
| Local AWQ/GPTQ 4-bit reference | §7 T0.3 | not run; published GPTQ/AWQ numbers only |
| ≈ 27B dense and larger models | D7, D8 | outside project scope (owner, after the project) |
| Energy (J/token) | §25 | not measured |
| Downstream, coding, math/reasoning, OOD task scores | §25 | not measured |
| Long-context speed (> 1024 positions) | §35 | measured to ctx 8192 at B = 1 (§3.7a); C++ two-pass attention rule on the M1 Ultra not verified for bit-exactness |
| C++ decode of the ≈ 3-bpp codecs on the target models | §47 B | not run (8B: Python only; 0.6B 3.5 bpp on the M3 only) |
| C++ batch > 1 decode | §25 | not implemented (the C++ tool decodes batch 1) |
| Qwen3-30B-A3B on a 16–32 GB Mac; Qwen3-1.7B | D7 | not run (≈ 4-bpp 30B-A3B needs ≈ 17 GB) |
| Task-level evaluation of the final files (`suc eval`) | §25 | tool built, no reported result; 20-prompt qualitative QA only (§3.13) |
| Latency and memory per Pareto point (mixed plans) | §27 | not measured (e2e benchmarks uniform codecs only) |
| End-to-end (logit) fine-tuning, PV-style index updates, tuning of expert stacks and column residuals | D3 | not done / not implemented |
| llama.cpp / GGUF interoperability | §24 | not done (optional) |
| C++ MoE tiled prefill path | cpp/README | decode is exact; prefill above 32 expert rows matches only to ~1e-4 |

---

## 6. How to use

Requirements: Apple Silicon Mac, Python 3.12, `uv`; the virtual environment on a local disk (see `CLAUDE.md`).

### 6.1 Compress a new model: `suc compress`

```bash
HF_HUB_OFFLINE=1 nohup .venv/bin/suc compress experiments/configs/<name>.yaml > logs/<name>.log 2>&1 &
# rerunning the same command resumes: finished stages and encoded blocks are skipped
```

Minimal config (fields of `CompressExperimentConfig`, `suc/config.py`; unknown keys are errors):

```yaml
name: compress_<model>
seed: 12345
model: {id: <hf-repo>, revision: <40-hex commit>, role: target}   # or local_path:
calibration: {seq_len: 512, n_calibration: 512, n_validation: 128, n_ood: 64,
              mix: {natural: 0.35, code: 0.25, math: 0.2, instruction: 0.2}, micro_batch: 8}
plan:
  mode: uniform                      # or: greedy + budget_bpw: 3.0 (+ optional head_candidates)
  codec: vq_k256_g2_rot1s+res0.005_activation_cols   # name of one candidate
  candidates:                        # final configuration (experiments/configs/compress_qwen3_8b_final.yaml)
    - {kind: vq, k: 256, block_size: 2, rotate_out: false, residual_fraction: 0.005, residual_structure: columns}
  embedding_codec: {kind: affine, bits: 4}   # keep the LM head out of the low-bit codebook
  # allocation_unit: expert_stack          # MoE: one allocation unit per (layer, projection) stack
eval: {seq_len: 2048, max_tokens: null, kl_sequences: 16}
finetune: null                       # block-wise tuning (not with codebook_scope: expert_stack)
rotation_groups: true                # shared q/k/v and gate/up input rotation (needed for the fused runtime)
verify_tensors: 16
preflight: {require_pinned_revision: true, fail_on_memory: true, memory_fraction: 0.85, disk_margin: 1.25}
# run_dir: defaults to experiments/artifacts/compress/<name>; keep it on a LOCAL disk
```

Recommended per spec D7/D8 and the findings above: pin the revision; **one low-dimensional codebook** (VQ G=2 K=256
at ≈ 4.2 bpp; G=2 K=64 / K=32 for ≈ 3.4 / 2.95 bpp; G=4 K=256 at ≈ 2.5 bpp); one-sided rotation (`rotate_out: false`
in each candidate); 0.5% activation-selected column residuals; embedding/LM head at affine 4-bit; for MoE
`codebook_scope: expert_stack` and `allocation_unit: expert_stack`; uniform codecs on dense models (greedy adds
nothing at ≥ 3.4 bpp and loses below 3 bpp); greedy on MoE. For speed, prefer 8-bit indices (K=256): the 6-bit K=64 codec is slower in
the current kernels (§3.7), and K=32 is not measured.

`suc compress` accepts this configuration, column residuals included: the final 8B run used exactly this candidate
and wrote the column residuals to SUC1 (R:20261002T191301Z_compress_qwen3_8b_final; 16 sampled encodings bit-exact on
reload).

**What happens**: `preflight` checks the pinned revision, validates the adapter against the checkpoint, estimates
peak memory and disk per stage, and fails early if the estimated peak exceeds `memory_fraction` × RAM or if free disk
is below need × `disk_margin`. It writes `preflight.json`. Every stage then writes a manifest and its artifacts under
`run_dir/<stage>/`. A stage is skipped when its fingerprint (non-default config fields, model hash, upstream output
hashes, stage version) is unchanged and its outputs exist. `encode` resumes block by block. The evaluation decodes
the tensors **from the SUC1 file**. The result JSON (+ `.md`) goes to `output_dir`.

**Measured cost** (Mac Studio M1 Ultra):

| Model | Stage | Time | Peak memory | Source |
|---|---|---:|---:|---|
| **Qwen3-8B (16.4 GB BF16), final configuration, every stage streamed** | calibrate | 295 s | 6.53 GiB | R:20261002T191301Z_compress_qwen3_8b_final |
| | reference | 359 s | 7.56 GiB | same |
| | plan | 0 s | — | same |
| | encode | 242 s | 15.22 GiB | same |
| | container | 79 s | 3.20 GiB | same |
| | evaluate | 385 s | 8.13 GiB | same |
| | **total** | **1361 s (≈ 23 min)** | max 15.22 GiB | same |
| | disk (preflight estimate) | | ≈ 15.5 GB of new run-dir data (reference 2.5 GB, encode 4.3 GB, container stage 8.7 GB) + checkpoint 16.4 GB; SUC1 file 4.32 GB | same (`preflight`, `container`) |
| Qwen3-8B, uniform M=4 + CSR, pre-streaming reference/evaluate (earlier run) | calibrate | 294 s | 6.53 GiB | R:20261001T195700Z_compress_qwen3_8b |
| | reference | 358 s | 19.89 GiB | same |
| | encode | 508 s | 15.22 GiB | same |
| | container | 82 s | 7.26 GiB | same |
| | evaluate | 386 s | 22.91 GiB | same |
| | **total** | **≈ 27 min** | | same |
| | disk (preflight estimate) | | run dir ≈ 11.2 GB (reference logits 2.5 GB, encode shards 4.3 GB, container 4.3 GB) + checkpoint 16.4 GB | same (`preflight`) |
| Qwen3-8B, streamed sweeps (`suc allocate`) | reference | 357 s | 8.1 GB | R:20261002T115608Z_pareto_allocate_qwen3_8b |
| | Fisher (streamed) + scoring of every candidate | 116 s + 1590 s | 31.2 GB | same |
| Qwen3-30B-A3B (61 GB BF16) | calibrate (streamed) | 818 s | 5.1 GiB | R:20261001T002456Z_phase2_calib_qwen3_30b_a3b |
| | reference (streamed) | 189 s | 5.8 GB | R:20261002T070150Z_q7_allocate_qwen3_30b_a3b |
| | encode, shared expert stacks: VQ G=4 / M=3 / M=4 | 364 / 648 / 814 s | — | R:20261002T043700Z_q5x_expert_stacks_qwen3_30b_a3b |
| | encode, per-expert codebooks, M=3 | 16,342 s | — | same |
| **Qwen3-30B-A3B, final uniform file (VQ G=2 `_xs` + columns), every stage streamed** | calibrate / reference / plan | 841 / 191 / 0 s | 5.11 / 5.23 GiB | R:20261002T212215Z_compress_qwen3_30b_a3b_final |
| | encode / container / evaluate | 581 / 121 / 259 s | 7.80 / 1.79 / 6.00 GiB | same |
| | **total** | **1994 s (≈ 33 min)** | max 7.80 GiB | same |
| **Qwen3-30B-A3B, final greedy-4.0 file (expert stacks)** | calibrate / reference / candidates / plan | 842 / 191 / 3497 / 0 s | 5.11 / 5.23 / 8.05 GiB | R:20261002T204858Z_compress_qwen3_30b_a3b_final_greedy |
| | encode / container / evaluate | 538 / 118 / 258 s | 7.80 / 1.79 / 6.14 GiB | same |
| | **total** | **5443 s (≈ 91 min)** | max 8.05 GiB | same |
| Qwen3-0.6B, final files, M3 laptop (indicative) | total, 4.25 / 3.51 bpp | 885 / 818 s | ≤ 4.49 GiB | R:20261002T202145Z_compress_qwen3_0.6b_final, R:20261002T203532Z_compress_qwen3_0.6b_final_3bpp |

Peak memory is the MLX peak reported per stage. With every stage streamed, reference and evaluate fall from 19.9 /
22.9 GiB to 7.6 / 8.1 GiB. The encode peak (15.22 GiB) is the same in both runs and is now the largest; its cause was
not analysed (encode is listed as a streaming stage in the result). On the 30B-A3B (61 GB of BF16 weights) no stage
exceeds 8.05 GiB (encode 7.80 GiB), so the 8B encode peak of 15.22 GiB does not grow with model size here; its
cause is still not analysed.

**Larger models (projection only, not measured).** From the Q5x expert-stack encode rate (≈ 7e-8 s/param on the
laptop for M=3, laptop/Studio ratio ≈ 2.5–3), a ~180B MoE needs ≈ 1–1.5 h per shared codec on the Studio. Streamed
calibration and reference sweeps scale with one block of weights plus the hidden states of the calibration and
evaluation sequences (spilled to disk above 4 GiB). Use `preflight` on the actual model before committing to a run.

### 6.2 Run a SUC1 file — Python/MLX runtime

```python
from suc.runtime.suc1_model import load_suc1_model, greedy
model, suc_model, report = load_suc1_model("model.suc1", group=True, fuse=True)   # default fused runtime
ids = greedy(model, prompt_token_ids, steps=64)
```

`load_suc1_model` builds the mlx-lm model of the file's architecture from the file alone (no checkpoint). Building
it calls the autotuner for the model's sizes on first use. `group=False` gives the per-projection plan used by the
C++ runtime. Tokenize with the model's HF tokenizer or `suc-tokenize`. Qwen3 dense was measured on the 8B file; qwen3_moe
on both final 30B-A3B files (24.1 / 23.8 ms/token at B = 1, bit-identical to C++; R:20261002T232117Z_cpp_check_qwen3_30b_a3b_final,
R:20261002T230743Z_cpp_check_qwen3_30b_a3b_final_greedy). Loading the 30B-A3B plan in Python takes ≈ 315–330 s.

### 6.3 Run a SUC1 file — C++ `suc1-generate`

```bash
cpp/build.sh ~/suc-local/cpp-build                     # build dir on a local disk; macOS builds the Metal tools
~/suc-local/cpp-build/suc1-generate model.suc1 --prompt "Text" --steps 64          # greedy, pipelined decode
~/suc-local/cpp-build/suc1-generate model.suc1 --chat --prompt "Question?" --temperature 0.7 --top-p 0.9 --seed 1
~/suc-local/cpp-build/suc1-generate model.suc1 --prompt "Text" --projections separate   # faster on the M1 Ultra
```

The file is memory-mapped. Weights stay in the mapping, and only small execution arrays (≈ 166 MB for the 8B) are
built at load. The tool prints load, prefill and decode timings and a memory line; `--json` writes them to a file.
Qwen3 dense and qwen3_moe are supported. Keep bit-exact validation runs below 1024 positions on the M1 Ultra until the
two-pass attention rule is verified there. See [cpp/README.md](../cpp/README.md).

### 6.4 Per-device autotuning on a new Mac

```bash
.venv/bin/python -m suc.kernels.autotune --rotation-sizes <hidden> <intermediate> --hidden <h> --intermediate <i>
.venv/bin/python -m suc.kernels.autotune --force       # re-measure every section
```

The autotuner measures codebook placement per M, rotation layout and block shape per size, element-wise fusion per
op, and the vec → MMA crossover batch. Measurements are cold-cache and in decode context. Results are cached under
the device key in `~/.cache/suc/kernel_tuning.json` (`SUC_TUNING_CACHE` overrides the path; `SUC_AUTOTUNE=0`
disables measuring). The run takes ≈ 45 s on the M3. Explicit knob values override the cache. **The C++ runtime never
tunes**: on a new machine, load the model once in Python (or run the command above) before the first C++ run.
Otherwise C++ uses built-in defaults, which can change rounding and greedy continuations.

---

## 7. Future work

Ordered by the measured gap each item addresses.

1. **Dense decode, prefill, and speed below 4 bpp and at B > 1.** In the matched harness (§3.7a) the C++ runtime runs VQ
   G=2 K=256 at 0.75× of MLX 4-bit on the 8B and 1.06× on the uniform 30B-A3B at B = 1, with prefill 2–4× slower.
   Open: the C++ prefill path (batched codebook matmul instead of chunk-64 decode kernels); the dense decode gap; a faster unpacking path for 5- and 6-bit indices (K=64 is slower than K=256 in Python,
   §3.7) and C++ runs of the ≈ 3-bpp codecs on the 8B / 30B-A3B; C++ batch > 1; then a greedy allocation over low-G VQ
   candidates (not yet run).
2. **Close the ≤ 3 bpw gap to QuIP#/QTIP**: end-to-end (logit) fine-tuning, discrete index updates (PV-Tuning style),
   and higher-dimensional lattice or trellis codes that respect the D4 ceiling (E8-style computed codebooks). Run a
   **local QuIP#/QTIP baseline** on Qwen3-8B with the SUC protocol (D3).
3. **Allocation that survives heavy degradation**: the additive Fisher proxy fails below 3 bpp. Try iterative
   re-estimation (re-score after a partial plan), interaction-aware or KL-validated steps, and the spec's adaptive
   allocator (§29/§40) measured against greedy.
4. **Batch-1 speed of the Python runtime**: with pipelined decode it equals C++ on the final 8B file (13.41 vs 13.89
   ms/token, §3.7a) but not on the 30B-A3B greedy file (20.75 vs 17.11 ms); in the last profile (K6) the SUC kernels took ≈ 12 ms per step, against 10.2 ms for all of MLX 4-bit's
   linears. Reduce launches and barriers;
   decide C++ fused-vs-separate projections per device in the autotuner; benchmark mixed (greedy) plans in the e2e
   harness.
5. **MoE**: the compressed-vs-reconstructed KL is explained (BF16 routing flips, §3.8) and the final files run in C++
   (§3.8). Still open: ≈ 3-bpp low-G VQ on expert stacks (to bring the 30B-A3B under 16 GB with usable quality), PPL
   parity with RTN 4-bit at ≈ 4 bpp, fine-tuning of shared expert stacks, the C++ MoE tiled prefill (2.9× slower than
   Python on the greedy file), and runs on 16–32 GB Macs.
6. **Evaluation breadth**: downstream / code / math / OOD tasks, long context, energy, more than one seed for the
   key rows. Without these, no capability claim is possible (§48).
7. **Scaling (owner, after the project)**: ≈ 27B dense, then larger dense and MoE models up to ≈ 180B, with the
   streamed pipeline and expert stacks.
8. Optional: a distribution container with Rice-coded residual positions and compressed scales (−4 to −5% file
   size); GGUF/llama.cpp interoperability.

---

## 8. Remaining gaps

All placeholders of the second pass are filled: P3 (§1.2, §2, §3.8, §6.1), P5 (§1.1, §3.3, §3.7, §3.11), P8 (§1.1,
§3.7, §4 Q8, §5.1), P9 (§3.12), plus the QA side-by-side (§3.13) and the re-assessed success levels (§5.1). What the
evidence still does not cover:

| Gap | Affects | Status |
|---|---|---|
| Matched-harness C++ vs MLX 4-bit timing | §1.1, §1.2, §3.7a, §5.1 "runtime competitive" | **done** (R:20261003T015734Z_perf_qwen3_8b_final, R:20261003T031427Z_perf_qwen3_30b_a3b_final); earlier cross-harness claims withdrawn. 0.6B only as an M3 smoke (indicative) |
| C++ decode of ≈ 3-bpp codecs on Qwen3-8B / 30B-A3B; VQ G=2 K=32 speed in any runtime | §5.1 Level B at ≈ 3 bpp | not run |
| Task-level scores of the final files (`suc eval`) | §4 Q4, §5.1 | tool built (`experiments/configs/eval_*.yaml`), no reported result |
| Local AQLM / QuIP# / QTIP and AWQ / GPTQ baselines | §3.10, D3 | published numbers only |
| More than one seed for the final files | §1.1, §1.2 | single seed |
| Qwen3-1.7B on the M3; 30B-A3B on a 16–32 GB Mac | §3.12, project goal | not run |
| Energy, C++ batch > 1 | §5.2 | not measured / not implemented (long-context speed now measured to 8192, §3.7a) |
