# Benchmarks

> **Current numbers** (chosen files, speed of the current build, comparison with ds4 / LM Studio / llama.cpp / MLX):
> [compression_report.md §1](compression_report.md#1-summary). This file is the phase-by-phase record; its numbers are
> those of the dated runs it cites and are superseded where §1 differs.

Protocol: spec §25 and §35. Every number below comes from a JSON file in `experiments/results/` that records model hash, resolved revision, dataset hash, git commit, seed, config hash and runtime.

## Phase 0 — dev model smoke run (not a reported result)

Machine: Apple M3, 16 GB (dev laptop). Timing and memory on this machine are **indicative only**; reported numbers must come from the Mac Studio M1 Ultra.
Config: `experiments/configs/phase0_dev_qwen3_0.6b.yaml` — Qwen/Qwen3-0.6B, WikiText-2 test (first 32,768 tokens, 15 windows of 2048, 30,720 scored tokens), KL on the first 512 tokens vs BF16, batch 1, 128 decode tokens, median of 3.
Result file: `experiments/results/20260930T205910Z_phase0_dev_qwen3_0.6b.json` (produced before the first git commit, so `git_commit` is null).

| Representation | bpp (actual) | Size (MB) | PPL | KL mean | Top-1 agree | TTFT p128 (s) | Prefill p512 (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 16.00 | 1192 | 19.96 | — | — | 0.061 | 2291 | 59.6 |
| MLX affine 8-bit g64 | 8.50 | 633 | 20.02 | 0.004 | 96.7% | 0.064 | 2105 | 101.8 |
| MLX affine 4-bit g64 | 4.50 | 335 | 22.87 | 0.239 | 73.4% | 0.065 | 2090 | 147.8 |

Observations:

- Identity check: all 310 tensors present in the mlx-lm model are bit-exact vs the checkpoint (max abs error 0). The checkpoint's 311th tensor is `lm_head.weight`, dropped by mlx-lm because embeddings are tied — so checkpoint params (751.6 M) exceed model params (596.0 M).
- 4-bit costs +14.6% PPL on a 0.6B model; small models are known to be less compression-tolerant (spec D7 note), so this is not representative of 8B.
- Decode speed scales with bytes read per token (59 → 102 → 148 tok/s), consistent with batch-1 decode being memory-bandwidth bound.
- Actual bpp includes scales and biases stored in BF16: 4 + 32/64 = 4.5.

## Open for Phase 0

- An AWQ or GPTQ reference 4-bit (spec T0.3).
- Pin `model.revision` to a commit hash in reported configs.

## Phase 0 — Qwen3-8B baselines (Mac Studio M1 Ultra, 128 GB) — reported result

Config: `experiments/configs/phase0_target_qwen3_8b.yaml` — Qwen/Qwen3-8B @ `b968826d`, full WikiText-2 test split (299,078 tokens; 145 windows of 2048, 296,960 scored), KL on the first 512 tokens vs BF16, median of 3 repeats.
Result file: `experiments/results/20260930T223113Z_phase0_target_qwen3_8b.json`.
Provenance caveat: this file records `git_commit = ef9b2b7`, `git_dirty = true`; the code that actually ran was `032b04a` (metadata used to be captured at the end of the run, after a push; fixed in `e39f7b3`). The benchmark code path is identical between the two commits.

| Representation | bpp (actual) | Size (GB) | PPL | ΔPPL | KL mean | Top-1 agree | Decode b=1 (tok/s) | Implied bandwidth b=1 | Decode b=16 (tok/s) | Prefill p=2048 (tok/s) | TTFT p=128 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 16.00 | 16.38 | 9.717 | — | — | — | 33.4 | 547 GB/s | 177.4 | 865 | 0.220 |
| MLX affine 8-bit g64 | 8.50 | 8.70 | 9.726 | +0.1% | 0.0018 | 98.8% | 55.7 | 485 GB/s | 195.0 | 720 | 0.204 |
| MLX affine 4-bit g64 | 4.50 | 4.61 | 10.082 | +3.8% | 0.109 | 87.5% | 64.9 | 299 GB/s | 189.2 | 703 | 0.222 |

Decode speed is measured with a synchronous `mx.eval` per token (conservative vs mlx-lm's pipelined generation). Implied bandwidth = weight bytes × decode tok/s (weights only, KV cache ignored), against ≈ 800 GB/s theoretical.

Findings:

1. **INT8 is essentially lossless** on this model (+0.1% PPL, 98.8% top-1 agreement); **INT4 costs +3.8% PPL** — the reference point SUC must match at fewer bits.
2. **At 4 bits the runtime, not memory bandwidth, is the limit.** BF16 decode reaches ~68% of peak bandwidth, INT4 only ~37%: shrinking weights 3.6× gives only 1.9× speed. A compressed format only turns into speed if its kernel keeps up — exactly the kernel-track question (§43).
3. **Batch 16 hides the difference.** At batch 16 all three representations decode at 177–195 tok/s: once compute dominates, fewer bytes buy little. Batch 1 is where SUC must prove itself.
4. **Prefill is compute bound** and slightly slower when quantized (865 → 703 tok/s at 2048 tokens): dequantization adds work when bandwidth is not the bottleneck.

## Phase 0 — Qwen3-30B-A3B baselines (Mac Studio) — reported result

Result: `experiments/results/20261001T005202Z_phase0_target_qwen3_30b_a3b.json` (full WikiText-2 test, re-run after the MoE quantization fix `c4c429a`; the earlier run `20260930T232606Z` left the experts in BF16 for INT8/INT4 and is superseded).

| Representation | bpp | Size (GB) | PPL | ΔPPL | KL mean | Top-1 | Decode b=1 (tok/s) | Decode b=16 (tok/s) | Prefill p=2048 (tok/s) | Peak (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 16.00 | 61.06 | 8.697 | — | — | — | 46.9 | 182.8 | 1712 | 79.0 |
| MLX affine 8-bit g64 | 8.50 | 32.44 | 8.718 | +0.2% | 0.0038 | 98.8% | 59.6 | 240.7 | 1364 | 50.7 |
| MLX affine 4-bit g64 (router 8-bit) | 4.50 | 17.18 | 9.062 | +4.2% | 0.069 | 89.1% | 56.6 | 227.3 | 1327 | 35.8 |

Findings:

1. Same picture as the dense 8B: INT8 ≈ lossless, INT4 costs ~4% PPL.
2. **For the MoE, fewer bytes barely speed up batch-1 decode** (47 → 57–60 tok/s): only ~3.3 B parameters are read per token and the expert gather/dispatch dominates. For MoE models the value of compression is **capacity** — at 4.5 bpp the 30B model needs 17 GB, i.e. it fits a 32 GB Mac; at ~2.5 bpp it would need ~10 GB.
3. Prefill gets slower when quantized (1712 → 1327–1364 tok/s): dequantization work in a compute-bound phase.

## Phase 1 — weight structure, dev model (weight-space only)

Config: `experiments/configs/phase1_weights_qwen3_0.6b.yaml` (run with G ∈ {8, 16}; later runs add G = 4). Qwen3-0.6B, layers 0 / 14 / 27, all 7 projections per layer (21 tensors), rows RMS-normalised, k-means fitted on ≤ 262,144 blocks per tensor and evaluated on held-out blocks.
Result: `20260930T220544Z_phase1_weights_qwen3_0.6b.json` and the matching `.md` report. Weight-space metrics only: they say nothing yet about model behaviour (Phase 2).

Relative MSE on held-out blocks, mean over the 21 tensors (attention and MLP behave alike):

| Index bpw | G | K | No rotation | Hadamard rotation | Gaussian R(D) bound | Best scalar quantizer at same rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 8 | 16 | 0.610 | 0.604 | 0.500 | — |
| 0.50 | 16 | 256 | 0.587 | 0.582 | 0.500 | — |
| 1.00 | 8 | 256 | 0.333 | 0.324 | 0.250 | 0.363 (1-bit Lloyd–Max) |
| 1.25 | 8 | 1024 | 0.244 | 0.237 | 0.177 | — |

Findings:

1. **Rotation makes weights Gaussian.** Excess kurtosis falls from 0.3–15.9 (worst: `layers.27.mlp.gate_proj`, 15.9) to |κ| < 0.02 for every tensor, and > 6σ outliers vanish. D3 is confirmed on this model.
2. **No shared directional structure.** Mean |cosine| between block directions equals the isotropic-random value at every G. The codebook gain is the classic VQ space-filling/shape gain, not "repeated patterns".
3. **VQ beats scalar at equal rate.** At 1 bpw, k-means VQ (0.324) beats the optimal 1-bit scalar quantizer for a Gaussian (1 − 2/π ≈ 0.363), and sits 1.1 dB above the Shannon bound — room for better codebooks (lattice / trellis) remains.
4. **Rotation equalises tensors.** The spread across tensors shrinks sharply (max − mean at 1 bpw: 0.022 → 0.002). After rotation all tensors look like the same source, which makes **shared / global codebooks** plausible (tiny codebook overhead).
5. **Entropy coding of indices will not pay.** Assignment entropy is ≥ 99% of log₂K everywhere: at most ~1% storage saving (spec Phase 8 is expected to be a negative result for plain k-means indices).
6. At equal rate, larger blocks help (G=16/K=256 beats G=8/K=16 at 0.5 bpw) but hit the D4 codebook ceiling quickly (G=16, K=1024 is not execution-viable).

Gate decision: **go** for vector codebooks, with rotation on by default. To confirm on Qwen3-8B and Qwen3-30B-A3B (queued on the Mac Studio) and at useful rates (2–4 bpw, via G = 4 and additive codebooks).

## Phase 1 — weight structure, Qwen3-8B (Mac Studio)

Config: `experiments/configs/phase1_weights_qwen3_8b.yaml` (G ∈ {4, 8, 16}, K ∈ {16, 64, 256, 1024}, layers 0 / 18 / 35, 21 tensors). Result: `experiments/results/20260930T223923Z_phase1_weights_qwen3_8b.{json,md}` (4 min 36 s on the M1 Ultra). The dev-model analysis was re-run on the Studio with G = 4 as well (`20260930T223447Z_phase1_weights_qwen3_0.6b`). Both runs record the pre-fix provenance (metadata captured at the end); no code changed during them.

Relative MSE on held-out blocks, mean over tensors, **with rotation** (attention and MLP within ±0.001):

| Index bpw | G | K | Rel. MSE | Gaussian R(D) | Gap to bound | Best scalar at same rate (Lloyd–Max, Gaussian) |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 8 | 256 | 0.324 | 0.250 | 1.1 dB | 0.363 |
| 1.00 | 4 | 16 | 0.345 | 0.250 | 1.4 dB | 0.363 |
| 1.50 | 4 | 64 | 0.186 | 0.125 | 1.7 dB | — |
| 2.00 | 4 | 256 | 0.098 | 0.0625 | 1.9 dB | 0.118 |
| 2.50 | 4 | 1024 | 0.051 | 0.031 | 2.1 dB | — |

Findings (Qwen3-8B confirms every dev-model finding):

1. **Rotation Gaussianises every tensor**: excess kurtosis up to 8.7 (`layers.35.mlp.down_proj`) → max |κ| = 0.012 after rotation; tensor-to-tensor spread collapses (max − mean ≤ 0.002).
2. **VQ beats the best scalar quantizer at equal rate** at both 1 and 2 bpw (−11% and −17% relative MSE).
3. **The gap to the Shannon bound grows with rate at fixed G** (1.1 dB → 2.1 dB): with G = 4, a 1024-entry codebook cannot fill 4-D space well. Reaching 2–3 bpw at G = 8 would need K = 2¹⁶–2²⁴ entries — far beyond the 16 KB execution ceiling (D4). **Additive (multi-codebook) representations are therefore required** for useful rates: two K = 256, G = 8 codebooks give 2 bpw with 8 KB of codebooks (Phase 5 / Q4). Lattice or trellis codebooks are the alternative to evaluate.
4. **Indices are near-uniform** (entropy ≥ 98% of log₂K): entropy coding would save ≤ 2% of index storage.

**Gate decision (Phase 1): GO.** Vector codebooks on rotated weights, rotation on by default, a small number of shared codebooks, additive codebooks to reach 2–3 bpw within the D4 ceiling. Weight-space only: behavioural impact is measured from Phase 2 on.

## Phase 2 — calibration and sensitivity, dev model (laptop)

Config: `experiments/configs/phase2_calib_qwen3_0.6b.yaml`. Result: `20260930T231235Z_phase2_calib_qwen3_0.6b.json` (+ `.md`); tokens and activation statistics in `experiments/artifacts/` (git-ignored, sha256 in the JSON).
Corpus: calibration 512 × 512 tokens (natural 179 / code 128 / math 103 / instruction 102 rows), validation 128 × 512, OOD 64 × 512 (PubMedQA); 0 shared 32-grams between any two partitions and with WikiText-2 test.
Activation statistics: 262,144 tokens, 196 tensors, layer-streamed, peak 1.6 GiB, 121 s on the M3.
Perturbation: rotated VQ, G = 8, K = 256 (1 bpw index rate) applied to one tensor at a time; measured on 16 held-out validation sequences.

Spearman rank correlation with measured logit KL over 24 sampled tensors:

| Predictor | ρ vs KL | ρ vs ΔCE |
|---|---:|---:|
| Weight relative MSE | −0.25 | — |
| Measured layer-output relative MSE (exact, on real inputs) | 0.09 | — |
| Diagonal-H proxy (GPTQ-style, predicted relative output error) | 0.45 | 0.57 |

Findings:

1. **Weight MSE carries no information about behaviour**: every tensor has rel-MSE ≈ 0.32, yet KL spans 0.016–0.141 (9×).
2. **Layer-local error does not predict end-to-end impact** (ρ = 0.09 even for the exact layer-output error). How an error propagates through the rest of the network (depth, role, residual stream) dominates. An objective that only matches layer outputs — as GPTQ-style methods do — is therefore not a sufficient allocation signal.
3. **The diagonal-H proxy is biased by role**: it overestimates the layer-output error of `q_proj`/`k_proj` by 1.4–2.3× and `gate_proj` by ~1.6×, while `o_proj`, `v_proj` and `down_proj` are within ±30%. Input channels of these projections are strongly correlated, which the diagonal ignores.
4. n = 24 gives wide uncertainty (ρ ≈ 0.4 is only marginally significant).
5. **Gradient-based proxy (empirical Fisher, ΔL ≈ ½ Σ F ΔW²)** on the same 24 tensors and the same perturbations (32 calibration sequences, M3): ρ vs KL = **0.74** (loss = ce) and 0.71 (loss = kl_self), against 0.45 for the diagonal-H proxy. Bootstrap 95% CI of the gain: [−0.06, +0.72] (ce) — suggestive, not yet conclusive at n = 24. With loss = kl_self the proxy also predicts the *size* of the KL within ~20% (median measured / predicted = 1.17). For ΔCE the Fisher proxy ranks worse (0.28 vs 0.57): ΔCE has a first-order component a second-order proxy cannot capture. The Fisher proxy is now part of `suc calibrate`; the Qwen3-8B / 30B-A3B runs will test it at scale.

## Phase 1 — weight structure, Qwen3-30B-A3B (Mac Studio)

Result: `experiments/results/20260930T233328Z_phase1_weights_qwen3_30b_a3b.{json,md}` — layers 0 / 24 / 47, attention + 4 sampled experts per layer (48 tensors), 7 min.
With rotation, MoE experts and attention are indistinguishable and match the dense 8B to the third decimal: relative MSE 0.324 at 1 bpw (G = 8, K = 256), 0.098 at 2 bpw (G = 4, K = 256), 0.051 at 2.5 bpw (G = 4, K = 1024); index entropy ≥ 98% of log₂K. Rotation brings the worst excess kurtosis from 32.4 (`layers.47.self_attn.o_proj`) to 0.006 (max 0.057 over all 48 tensors). All MoE dimensions (2048, 768, 4096, 512) are Hadamard-supported.

## Phase 2 — dev model re-run on the Mac Studio (with the Fisher proxy)

Result: `experiments/results/20260930T233540Z_phase2_calib_qwen3_0.6b.json` — same corpus hash as the laptop run; whole pipeline in 2 min (Fisher 7 s, 32 sequences).
Spearman vs measured KL (n = 24): **Fisher 0.746**, diag-H 0.447, weight MSE −0.246 — the laptop numbers are reproduced across machines.

## K1 — Metal VQ matmul, correctness kernel (laptop M3, indicative only)

`suc/kernels/vq_matmul.py`, `suc/quant/packing.py`; benchmark `benchmarks/k1_vq_matmul.py`. y = x Ŵᵀ computed on the GPU from fp16 codebook + bit-packed indices + row scales, **Ŵ never materialised**. Errors vs the float32 oracle are output rounding only (fp16 x: rel 2e-4; fp32 x: rel ≤ 4e-7); 179 tests.

Median kernel time (µs), random codebooks/indices (timing only):

| Layer (out × in) | B | FP16 dense | MLX affine 4-bit (4.5 bpw) | SUC lookup, G=8 K=256 (~1 bpw) | SUC lookup, G=4 K=256 (~2 bpw) | SUC table, G=8 |
|---|---:|---:|---:|---:|---:|---:|
| 12288 × 4096 | 1 | 1151 | 325 | **209** | **229** | 414 |
| 12288 × 4096 | 16 | 2092 | 1198 | 1873 | 1583 | 3153 |
| 3072 × 1024 | 1 | 82 | 30 | 46 | 57 | 69 |
| 1024 × 3072 | 1 | 66 | 24 | 32 | 34 | 80 |

Findings:

1. **At batch 1 on a large layer the compressed kernel already beats MLX 4-bit** (1.6× at ~1 bpw, 1.4× at ~2 bpw) — first evidence for the execution hypothesis. It is not bandwidth-bound yet (30–55 GB/s vs ~87 GB/s for dense/q4 on the M3): the limits are threadgroup-memory bank conflicts from random codebook reads and per-row re-reads of x.
2. **Small layers and batch 16 lose** to MLX 4-bit (too few threadgroups to hide latency; at B = 16 the kernel becomes arithmetic/cache-bound while q4 uses simdgroup matrix instructions).
3. **The partial-product "table" strategy is slower everywhere** in this form: device-memory gathers dominate (~15 G gathers/s) and the table leaves cache at B = 16. It needs table chunks held in threadgroup memory with many rows per threadgroup (rows ≫ K) — K2 work.
4. Negative result: staging x in threadgroup memory made "lookup" slower (occupancy + barriers; caches already absorb the reuse).

## Phase 2 — Qwen3-8B calibration and sensitivity (Mac Studio)

Result: `experiments/results/20260930T235623Z_phase2_calib_qwen3_8b.json` (+ `.md`). 21 min total: layer-streamed statistics 287 s (peak 6.5 GiB for an 8B model, D8), Fisher 155 s (32 × 512 tokens, peak 49 GiB), proxies for 252 tensors 549 s, 24 measured perturbations 231 s.

Spearman vs measured KL (n = 24, VQ 1 bpw injected into one tensor at a time, held-out validation text):

| Predictor | ρ vs KL |
|---|---:|
| **Empirical Fisher, ½ Σ F ΔW²** | **0.821** |
| Diagonal-H proxy (relative output error) | 0.489 |
| Measured layer-output error | 0.574 |
| Weight relative MSE | 0.394 |

Findings:

1. **Sensitivity spans three orders of magnitude**: the same 1-bpw error gives KL from 0.002 (`layers.4.self_attn.k_proj`) to **1.20 (`layers.6.mlp.down_proj`)** — the latter is 30× more sensitive than the next tensor. This matches the known "super weight / massive activation" behaviour of early MLP down-projections.
2. **The Fisher proxy finds it; layer-local proxies do not**: Fisher ranks `layers.6.mlp.down_proj` first by a factor of ~57, while its diagonal-H relative error (0.36) is unremarkable.
3. Fisher is the best predictor on both the dev model (0.75) and the 8B (0.82); it becomes the primary sensitivity signal for the allocator (Phase 7), with the diagonal-H proxy kept as a cheap fallback.

## Phase 2 — Qwen3-30B-A3B calibration and sensitivity (Mac Studio)

Result: `experiments/results/20261001T002456Z_phase2_calib_qwen3_30b_a3b.json` (+ `.md`). 42 min: layer-streamed statistics 818 s with a **5.1 GiB peak for a 61 GB model** (D8), Fisher 402 s (peak 71 GiB, grouped backward), proxies for 1,344 tensors (attention + 8 sampled experts per layer) 340 s, 24 measured perturbations 117 s.

| Predictor | ρ vs KL (n = 24) |
|---|---:|
| **Empirical Fisher** | **0.848** |
| Diagonal-H proxy | 0.182 |
| Weight relative MSE | −0.129 |

Findings:

1. The Fisher proxy is the best predictor on all three models (0.75 / 0.82 / 0.85); the layer-local proxy degrades on the MoE.
2. **A single expert is almost insensitive**: a 1-bpw error in one expert gives KL ≤ 0.0005 (each expert sees ~1/16 of the tokens), while attention projections reach 0.076 (`layers.47.self_attn.v_proj`). Experts — 95% of the parameters — are where low bit rates are cheapest; attention (3%) is where bits should go. This is exactly the allocation the greedy allocator (Phase 7) is designed to find.

## Q1–Q4 — uniform codecs on the whole dev model (Mac Studio) — negative result

Result: `experiments/results/20261001T013236Z_q1q4_codecs_qwen3_0.6b.json` (+ `.md`). Every eligible linear tensor **and the tied embedding** encoded with one codec; WikiText-2 test (first 32,768 tokens) and KL vs BF16 on 16 held-out validation sequences. BF16 PPL 19.95.

| Codec | Model bpp | PPL | KL |
|---|---:|---:|---:|
| RTN affine 4-bit g64 | 4.50 | 22.84 | 0.189 |
| Additive M=4 (K=256, G=8, rot) | 4.06 | 30.54 | 0.437 |
| Scalar codebook K=16 | 4.02 | 90.8 | 1.22 |
| RTN affine 3-bit | 3.50 | 54.2 | 1.09 |
| Additive M=3 | 3.05 | 149.9 | 2.03 |
| ≤ 2.5 bpp (RTN 2-bit, VQ G=4, additive M=2, VQ G=8, scalar K≤8) | 1.0–2.5 | > 10⁵ | > 8 |

**Data-free codebooks lose to RTN at the same or higher weight-MSE quality** (additive M=3 has lower weight MSE than RTN 3-bit in every role, 0.035 vs 0.038, yet 2.8× the PPL). Diagnosis on `layers.2.mlp.down_proj`, whose input has a massive-activation channel (E[x²] max/median ≈ 2·10⁶): activation-weighted error Σ_j E[x_j²]‖ΔW_:,j‖² / Σ_j E[x_j²]‖W_:,j‖² is 0.0048 for RTN 3-bit (per-group min/max keeps the extreme "super weights" exact) but 0.060 for rotated additive M=3 and 0.075 without rotation (codebooks smear them). The Fisher proxy had already ranked this tensor the most sensitive (Phase 2).

## Q5 — activation-selected residuals fix it (dev model, laptop)

Result: `experiments/results/*_q5_residual_qwen3_0.6b.json` (+ `.md`). Residual entries chosen by E[x_j²]·R² (Phase 2 activation statistics), int8 values, fixed-width positions.

| Codec | Model bpp | PPL | KL |
|---|---:|---:|---:|
| RTN affine 4-bit | 4.50 | 22.84 | 0.190 |
| **Additive M=4 + 0.5% residual (activation)** | **4.21** | **23.47** | **0.181** |
| Additive M=4 + 0.1% residual | 4.09 | 24.84 | 0.249 |
| RTN affine 3-bit | 3.50 | 54.23 | 1.087 |
| **Additive M=3 + 0.5% residual (activation)** | **3.20** | **35.77** | **0.593** |
| Additive M=3 + 0.1% residual | 3.08 | 48.85 | 0.894 |
| Additive M=3 + 0.5% residual (**magnitude**) | 3.20 | 128.5 | 1.844 |
| VQ G=4 + 0.5% residual | 2.18 | 565 | 2.66 |

Findings:

1. **Selection is what matters, not size**: at equal bits, activation-selected residuals give PPL 35.8 vs 128.5 for magnitude-selected ones.
2. With 0.5% residuals, SUC matches RTN 4-bit in KL (0.181 vs 0.190) at 0.29 bpp less, and beats RTN 3-bit clearly (PPL 35.8 vs 54.2) at 0.3 bpp less.
3. ≈ 2 bpp remains broken on this 0.6B model (PPL 565): small models are the least compression-tolerant; codebook fine-tuning (Q6) and the allocator (Q7) are the next levers. Single-tensor result for `layers.2.mlp.down_proj`: 0.1% residual brings its activation-weighted error from 0.060 to 0.0014 (+0.03 bpw).
4. Design consequence: **activation/importance-selected residuals are a required SUC component**, not an optional one; magnitude-based outlier handling is not sufficient.

## Q1–Q4 — uniform codecs on Qwen3-8B (Mac Studio)

Result: `experiments/results/*_q1q4_codecs_qwen3_8b.json` (+ `.md`). All eligible linears **and** the embedding / LM head encoded with one codec, **no residuals**; full WikiText-2 test; KL on 16 held-out validation sequences. BF16 PPL 9.717. 2 h for 13 codecs.

| Codec | Model bpp | PPL | KL |
|---|---:|---:|---:|
| RTN affine 4-bit | 4.50 | 10.088 | 0.092 |
| Additive M=4 | 4.01 | 10.557 | 0.165 |
| Scalar codebook K=16 | 4.00 | 76.1 | 1.80 |
| RTN affine 3-bit | 3.50 | 13.364 | 0.370 |
| **Additive M=3** | **3.01** | **13.407** | 0.473 |
| RTN affine 2-bit | 2.50 | 1631 | 5.09 |
| **VQ G=4 K=256 (rot)** | **2.00** | **162.5** | 2.58 |
| Additive M=2 | 2.01 | 217.7 | 2.97 |
| VQ G=8 (≈1 bpp) | 1.01 | > 10⁷ | > 14 |

Findings:

1. The 8B tolerates compression far better than the 0.6B (at ~3 bpp: PPL +38% vs +650%).
2. **Even without residuals, additive M=3 at 3.01 bpp matches the perplexity of RTN 3-bit at 3.50 bpp** (13.41 vs 13.36; KL 0.47 vs 0.37), and **VQ G=4 at 2.0 bpp is 10× better than RTN 2-bit at 2.5 bpp** (PPL 162 vs 1631).
3. The scalar codebook (the original lookup-table idea, Q1) is clearly worse than RTN at equal index bits — confirming the core motivation for vector codebooks (§2).
4. Single-codebook VQ G=4 beats additive M=2 at 2 bpp on both models (also off the allocator's hull in Q7): the cheaper kernel is also the better representation.
5. Rotation alone does not rescue 1 bpp (PPL > 10⁷ with or without).

## K1/K2 — kernel benchmarks on the Mac Studio M1 Ultra (warm cache) — not yet a valid §43 measurement

Results: `experiments/results/20261001T033545Z_k1_vq_matmul.json`, `20261001T033554Z_k2_suc_linear.json`. Median µs, random codes (timing only). Another process (`strata.cli serve`, not part of SUC) was running on the Studio.

| Layer | B | FP16 | MLX 4-bit (4.5 bpw) | SUC VQ G=8 (1 bpw) | SUC VQ G=4 (2 bpw) | SUC additive M=2 (2) | SUC additive M=3 (3) | rotation only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 12288 × 4096 | 1 | 96 | **45** | 64 | 59 | 141 | 265 | 33 |
| 12288 × 4096 | 4 | 387 | **129** | 375 | 186 | 479 | 571 | 41 |
| 12288 × 4096 | 16 | 389 | **242** | 2462 | 2496 | 2578 | 2703 | 49 |
| 4096 × 12288 | 1 | 89 | **44** | 76 | 60 | 149 | 274 | 35 |
| 4096 × 4096 | 1 | 52 | **26** | 37 | 48 | 99 | 105 | 27 |

Findings (honest negative result for now):

1. **The measurement is cache-biased**: FP16 reaches 1043 GB/s, above the M1 Ultra's ~800 GB/s DRAM bandwidth — the same 100 MB matrix is re-read and partly served from the SLC. Real decoding streams gigabytes from DRAM. This hides exactly the advantage SUC is designed for (fewer bytes), so it is **not** a valid answer to the §43 gate. Next: a cold-cache benchmark cycling through enough distinct matrices to exceed the cache.
2. Even so, on the M1 Ultra the SUC kernels are **slower than MLX 4-bit** at batch 1 (59 vs 45 µs at 2 bpw), unlike on the M3 laptop; compute/latency, not bandwidth, limits them here.
3. Batch ≥ 4 is far behind (10× at B = 16): the lookup kernel has no simdgroup-matrix path.
4. Additive codebooks cost ~M× the single-codebook time: the fused kernel scales with codebook reads.

## Q7 — greedy semantic allocation validated on the whole dev model (Mac Studio)

Result: `experiments/results/*_q7_allocate_qwen3_0.6b.json` (+ `.md`). Candidates (all with activation-selected residuals, Q5): VQ G=4 + 0.5%, additive M=3 + 0.1% / 0.5%, M=4 + 0.5%, RTN 4-bit, BF16; loss = Fisher proxy ½ Σ F ΔW² (32 calibration sequences); embedding fixed at RTN 4-bit in every plan; budgets = average bpw over the 196 allocated tensors. Candidate scoring 686 s, Fisher 6 s.

| Plan | Allocated bpw | Model bpp | PPL | KL | Predicted loss |
|---|---:|---:|---:|---:|---:|
| greedy 2.5 | 2.50 | 3.02 | 59.0 | 1.116 | 3.09e-3 |
| greedy 3.0 | 3.00 | 3.39 | 34.6 | 0.569 | 1.86e-3 |
| uniform M=3 + 0.5% | 3.20 | 3.54 | 33.0 | 0.497 | 1.85e-3 |
| greedy 3.5 | 3.50 | 3.76 | 27.3 | **0.330** | 1.14e-3 |
| greedy 4.0 | 4.00 | 4.13 | 24.3 | **0.202** | 7.14e-4 |
| uniform M=4 + 0.5% | 4.22 | 4.29 | 23.3 | 0.163 | 6.49e-4 |
| uniform RTN 4-bit | 4.50 | 4.50 | 22.8 | 0.189 | 9.69e-4 |

Findings:

1. **The Fisher proxy ranks whole-model plans correctly**: Spearman(predicted loss, measured KL) = **0.964** across the 7 plans (additivity across tensors holds well enough to choose between plans).
2. **The allocation gain is real but modest on this model**: at equal model bpp, greedy plans beat the uniform residual codecs by 0–18% in KL (greedy 3.5: 0.330 vs ≈ 0.40 interpolated between the uniform plans; greedy 4.0: 0.202 vs ≈ 0.235). The 44–48% gain predicted earlier was for candidates *without* residuals; activation-selected residuals already repair the most sensitive tensors, leaving less for the allocator.
3. Where bits go: attention `v_proj`/`k_proj` and early `down_proj` get the higher-rate codecs; most MLP tensors take VQ G=4 or M=3.
4. Greedy 4.0 matches RTN 4-bit's KL (0.202 vs 0.189) at 0.37 bpp less. The decisive test is Qwen3-8B, whose sensitivity spread is 600× (queued).

## Q6 — block-wise fine-tuning, dev model (laptop, embedding left BF16)

Script: `benchmarks/q6_finetune_dev.py` (run by the Q6 agent; raw JSON kept outside the repo). Indices fixed; codebooks, row scales and residual values tuned with Adam (lr 1e-3, 150 steps per block, 64 calibration sequences × 512) to match each BF16 block's output on the activations of the already-quantized prefix; values rounded to their stored precision with a straight-through estimator, so the bits are unchanged. Eval as in Q5 (WikiText-2 test first 32,768 tokens; KL on 16 held-out validation sequences). Model bpp includes the 0.16 B BF16 embedding; "encoded" bpp is over the linear layers.

| Representation | Encoded bpp | PPL | KL | Top-1 |
|---|---:|---:|---:|---:|
| RTN 4-bit | ≈ 4.5 | 22.54 | 0.168 | 0.807 |
| RTN 3-bit | ≈ 3.5 | 47.42 | 0.953 | 0.579 |
| Additive M=3 + 0.5% residual, untuned | 3.20 | 32.26 | 0.479 | 0.693 |
| **Additive M=3 + 0.5% residual, block-tuned** | 3.20 | **27.10** | **0.292** | 0.760 |
| VQ G=4 + 0.5% residual, untuned | 2.17 | 165.6 | 2.20 | 0.388 |
| **VQ G=4 + 0.5% residual, block-tuned** | 2.17 | **63.2** | **1.03** | 0.563 |

Every block's relative output error falls (−22% to −50%; −99.6% in block 2, the massive-activation block); no block reverted. Cost on the M3: 31–54 s per block, ~15–17 min per model, peak 5.3–5.6 GiB. Weight-space index re-assignment did not help (rejected by the guard); output-loss index updates (PV-Tuning style) remain open. MoE expert rows are not tuned yet. Integrated in `suc quantize` (`finetune:`); Studio runs for the dev model (whole-model protocol) and Qwen3-8B are queued.

## Q5 — activation-selected residuals on Qwen3-8B (Mac Studio) — reported result

Result: `experiments/results/*_q5_residual_qwen3_8b.json` (+ `.md`). Whole model (embedding and LM head encoded with the same codec), full WikiText-2 test, KL on 16 held-out validation sequences. BF16 PPL 9.717.

| Codec | Model bpp | PPL | ΔPPL | KL |
|---|---:|---:|---:|---:|
| RTN affine 4-bit | 4.50 | 10.088 | +3.8% | 0.0922 |
| **Additive M=4 + 0.5% residual** | **4.18** | **10.038** | **+3.3%** | **0.0841** |
| **Additive M=4 + 0.1% residual** | **4.04** | 10.176 | +4.7% | **0.0917** |
| RTN affine 3-bit | 3.50 | 13.364 | +37.5% | 0.370 |
| **Additive M=3 + 0.5% residual** | **3.18** | **11.316** | +16.5% | **0.248** |
| Additive M=3 + 0.1% residual | 3.04 | 11.689 | +20.3% | 0.280 |
| Additive M=3 + 0.5% residual (magnitude) | 3.18 | 13.043 | +34.2% | 0.454 |
| RTN affine 2-bit | 2.50 | 1631 | — | 5.09 |
| **VQ G=4 + 0.5% residual** | **2.18** | **18.70** | +92% | 0.742 |

Findings:

1. **SUC beats INT4 with fewer bits on the target model**: 4.18 bpp is better than RTN 4-bit at 4.50 on both PPL and KL; 4.04 bpp matches its KL (−10% bits). No fine-tuning yet.
2. **~3.2 bpp is clearly better than RTN 3-bit at 3.5** (PPL +16.5% vs +37.5%, KL 0.25 vs 0.37).
3. **~2.2 bpp is degraded but functional** (PPL 18.7) where RTN 2-bit at 2.5 collapses (1631).
4. Activation selection vs magnitude at equal bits: KL 0.248 vs 0.454 — the same pattern as on the dev model.

## Merge step 5 — generating directly from SUC codes (laptop, indicative)

`suc/runtime/compressed_model.py` (`convert_to_compressed`, `SucResidualLinear` with a CSR residual Metal kernel), `benchmarks/e2e_generate.py`. Logits equal those of the decoded-weights model (float32 KL ≈ 1e-8; bf16 differences equal the bf16 rounding floor). Qwen3-0.6B, M3, embedding MLX 4-bit in the SUC rows:

| Variant | Linear bpw | Weights in memory (MiB) | Decode B=1 (tok/s) | Decode B=16 (tok/s) | ms/token B=1 |
|---|---:|---:|---:|---:|---:|
| BF16 | 16 | 1137 | 56.7 | 465 | — |
| MLX 4-bit | 4.5 | 320 | **152.6** | **654** | 7.4 |
| SUC VQ G=4 | 2.02 | **193** | 74.8 | 552 | 12.9 |
| SUC additive M=3 + 0.5% residual | 3.20 | 255 | 53.7 | 439 | 22.3 |

Profile of a batch-1 token (ms): SUC kernels 4.8 vs MLX 4-bit linears 4.3 — the kernel itself is on par; the gap is the Hadamard rotations (5.5 ms: 8 launches per projection, 56 per block) and, with residuals, the residual pass (2.0–4.5 ms). Prefill is 2–3× slower (16-row launch tiles). Next: shared input rotation for q/k/v and gate/up, fused rotation kernels, fused projection launches (estimated ≈ 7.8 ms/token, near parity on the 0.6B; the 8B's larger layers favour SUC in the cold-cache kernel benchmark).

## K5 — batch-1 generation from codes reaches MLX 4-bit parity on the dev model (laptop, indicative)

Qwen3-0.6B, SUC VQ G=4 (2 bpw linears), M3; ms per decode step (all paths exact vs decoded weights: f32 KL ≈ 1e-8):

| Step | B=1 | B=4 | B=16 | GPU launches / block |
|---|---:|---:|---:|---:|
| Before (K2 SucLinear) | 12.16 | 13.47 | 28.49 | 56 |
| Shared input rotation (q/k/v, gate/up) + fused rotation kernel | 10.35 | 12.96 | 27.19 | 18 |
| + fused q\|k\|v and gate\|up launches | 9.17 | 12.27 | 24.90 | 12 |
| **+ `mx.compile` (all on)** | **6.83** | **10.29** | **22.74** | 12 |
| MLX 4-bit (eager, as mlx-lm runs it) | 6.58 | 9.15 | 23.99 | 7 |

Findings: the largest single cost at batch 1 was Python graph construction (~18 µs per SucLinear call, on the critical path of synchronous decoding); compiling removes it. Sharing the input rotation between projections that read the same input does not change quality (paired rel-MSE difference −4e-5 ± 6e-5, smaller than the effect of re-drawing the seed). Remaining gap at B=1: +0.25 ms; MLX 4-bit was measured eager, so a compiled baseline must be measured before claiming parity. Additive M=3: 9.14 ms; VQ + residual: 10.23 ms (residual members not yet fused). Output rotation cannot be folded into the kernel epilogue (Uᵀ mixes rows across threadgroups).

## Q7 — greedy allocation on Qwen3-8B (Mac Studio) — reported result

Result: `20261001T072625Z_q7_allocate_qwen3_8b.json` (+ `.md`). Same candidates as the dev run (all with activation-selected residuals); embedding and LM head fixed at RTN 4-bit in every plan; full WikiText-2 test; KL on 16 held-out validation sequences. BF16 PPL 9.717. Spearman(predicted, measured KL) across plans = 0.964.

| Plan | Allocated bpw | Model bpp | PPL | KL | Predicted loss |
|---|---:|---:|---:|---:|---:|
| greedy_2.5 | 2.50 | 2.80 | 12.824 | 0.3083 | 1.80e-03 |
| greedy_3.0 | 3.00 | 3.23 | 11.002 | 0.1758 | 1.04e-03 |
| uniform additive m3 + 0.5% res | 3.17 | 3.38 | 10.901 | 0.1621 | 1.12e-03 |
| greedy_3.5 | 3.50 | 3.65 | 10.551 | 0.1243 | 6.57e-04 |
| greedy_4.0 | 4.00 | 4.08 | 10.193 | 0.0840 | 4.08e-04 |
| uniform additive m4 + 0.5% res | 4.18 | 4.23 | 10.081 | 0.0751 | 4.00e-04 |
| uniform affine_b4_g64 | 4.50 | 4.50 | 10.088 | 0.0922 | 5.61e-04 |

Findings:

1. **Protecting the LM head matters more than allocation inside the network.** The same uniform codec (additive M=3 + 0.5% residual) gives KL 0.162 here, with the untied LM head at RTN 4-bit, against 0.248 in the Q5 grid where the LM head was encoded with the codebook. The LM head becomes an allocator candidate (next allocator change).
2. **Best quality per bit so far**: uniform additive M=4 + 0.5% residual at 4.18 allocated bpw — PPL 10.081, KL 0.075 — beats RTN 4-bit at 4.50 (10.088, 0.092) on both metrics.
3. **Greedy vs uniform at equal bits: +2–7% in KL** (interpolated), as on the dev model: once residuals protect the sensitive weights, per-tensor allocation adds little. Greedy at 2.5 allocated bpw (PPL 12.82) already beats RTN 3-bit at 3.5 (13.36).
4. The Fisher prediction orders the plans correctly.

## K3 — cold-cache kernel benchmark on the Mac Studio M1 Ultra (§43 gate) — reported result

Results: `20261001T072637Z_k3_m1ultra_{default,cb_threadgroup,half_math,sg4_rows8,vec2,mma_rt2}.json` (`benchmarks/k3_ab.py`, one variant at a time, sequential). Each method cycles through 16 GB of distinct matrices, so weights stream from DRAM (FP16 reaches 669 GB/s at B = 1, inside the ~800 GB/s DRAM bound — the warm-cache bias of the K1/K2 run is gone). Median of 5 samples; random codes (timing only). `strata.cli serve` (not part of SUC) was resident but idle. "Block" = all 7 linears of one Qwen3-8B decoder block.

Qwen3-8B block, µs (lower is better):

| B | FP16 | MLX q4 (4.5) | MLX q3 (3.5) | MLX q2 (2.5) | VQ G=8 (1.0) | VQ G=4 (2.0) | add. M=2 (2.0) | add. M=3 (3.0) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 — codebook in device memory | 687 | 214 | 199 | **167** | 233 | 206 | 535 | 1049 |
| 1 — codebook in threadgroup memory | 685 | 215 | 196 | **167** | 174 | 188 | 248 | 353 |
| 4 — device | 1718 | 489 | 563 | 462 | 1397 | 645 | 1837 | 2183 |
| 4 — threadgroup | 1718 | 490 | 561 | **460** | 814 | **445** | 782 | 894 |
| 16 — device | 1768 | 1264 | 1379 | 1222 | 1624 | 1667 | 2131 | 2886 |
| 16 — threadgroup | 1772 | 1262 | 1345 | **1230** | 1510 | 1486 | 1682 | 1906 |

Findings:

1. **Codebook placement flips between GPU generations.** On the M1 Ultra, staging the codebook in threadgroup memory is 1.1–3× faster (additive M=3 at B = 1: 1049 → 353 µs); on the M3 laptop device-memory reads were faster (K2). `CODEBOOK_IN` now defaults to `"auto"` (threadgroup on `applegpu_g13*`, device elsewhere). The other A/B switches (half-precision math, 4 simdgroups × 8 rows, vec2 loads, MMA_RT = 2) are neutral or worse and stay off.
2. **Single-codebook VQ is bandwidth-competitive**: at 2 bpw, VQ G=4 beats MLX 4-bit at B = 1 (188 vs 215 µs) and B = 4 (445 vs 490), but not MLX 2-bit at similar bits (167 / 460), and loses at B = 16 (1486 vs 1262).
3. **Additive codebooks are not**: M=3 costs 353 µs at B = 1, 1.8× MLX 3-bit; the quality winner (M=4 + residuals, Q5/Q7) would be slower still. Kernel time grows with M because every group reads M codebook entries.
4. §43 gate: the single-layer benchmark is now sound (cold cache, every layer shape, B = 1/4/16, against MLX 2/3/4-bit). Direct execution beats a 4-bit baseline only for single-codebook VQ; for the additive configurations that win on quality, **the compressed format is currently a memory/capacity win, not a speed win** on the M1 Ultra. Next kernel work: the partial-product table path for M ≥ 2 and an end-to-end decode measurement (queue12).

## K5 — generation from codes on Qwen3-8B (Mac Studio M1 Ultra) — reported result

Result: `20261001T074850Z_e2e_generate_qwen3_8b.json` (`benchmarks/e2e_generate.py`; 128-token prompt, 32 decode tokens, 3 repeats, all runtime fusions and `mx.compile` on, `CODEBOOK_IN = auto` → threadgroup). Embedding/LM head at MLX 4-bit in every SUC variant; codes from a fast k-means (6 iterations, timing-oriented, not the quality-track encodings). Correctness: logit KL of the code-executing model vs the same weights decoded to BF16, 256 tokens: 0.001–0.007, top-1 agreement 0.96–0.99. The recorded `git_commit` (a09c311) was read at the end of the run; the code that ran is 8974a09 (the difference is documentation only).

| Variant | Resident MiB | Linears bpw | B=1 tok/s | B=1 ms/step | B=4 tok/s | B=16 tok/s | TTFT B=1 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 15623 | 16 | 33.3 | 30.1 | 46.7 | 177.2 | 220 |
| MLX 4-bit | 4394 | 4.5 | **74.9** | **13.4** | **137.8** | 193.3 | 208 |
| SUC VQ G=4 | 2339 | 2.02 | 38.7 | 25.9 | 108.8 | **225.9** | 283 |
| SUC VQ G=4 + 0.5% residual | 2445 | 2.15 | 26.6 | 37.6 | 67.9 | 124.9 | 484 |
| SUC additive M=3 | 3169 | 3.02 | 22.9 | 43.7 | 51.6 | 134.8 | 519 |
| SUC additive M=3 + 0.5% residual | 3276 | 3.15 | 23.4 | 42.7 | 59.7 | 106.8 | 576 |

Batch-1 profile (ms per decode step): VQ G=4 — codebook kernels 12.1, **input rotations 10.8**, rest of the model 3.0 (MLX 4-bit: all linears 10.2). Residual variants add **10.6–12.6 ms** for the sparse int8 residual pass.

Findings (negative for speed at batch 1):

1. On the 8B model SUC is **~2× slower than MLX 4-bit at batch 1** (38.7 vs 74.9 tok/s at 2 bpw); the dev-model parity of the laptop run does not transfer. The kernels alone (12.1 ms) are close to MLX 4-bit's linears (10.2 ms) while reading half the bytes; the gap is the **online Hadamard rotations** (10.8 ms, ~75 µs per rotation, far above bandwidth cost) and the **sparse residual pass** (~11 ms).
2. At batch 16, VQ G=4 is the fastest variant (225.9 vs 193.3 tok/s): the MMA path pays off once the rotation cost is amortised.
3. Memory: 2.3–3.3 GB resident vs 4.4 GB for MLX 4-bit and 15.6 GB for BF16.
4. Next kernel work, in order of measured cost: (a) remove the online input rotations of q/k/v and gate/up by rotating the residual stream offline (QuaRot-style folding into the embedding, RMSNorm weights and the o/down output sides), leaving one online rotation per block (down_proj input); (b) a structured residual pass (activation-selected residuals concentrate on a few input columns → a small dense int8 GEMV on gathered columns instead of a CSR scatter); (c) the partial-product table path for additive M ≥ 2.

## Q6 — block-wise fine-tuning in the whole-model protocol, dev model (Mac Studio) — reported result

Result: `20261001T081616Z_q6_finetune_qwen3_0.6b.json` (+ `.md`), `suc quantize` with `finetune:` (100 steps per block, lr 1e-3, 64 calibration sequences, deterministic VJP). Same protocol as the Q5 dev row (every linear and the tied embedding encoded with the codec; full WikiText-2 test capped at 32,768 tokens; KL on 16 held-out validation sequences), so the untuned baseline is the Q5 dev table.

| Codec | Model bpp | PPL untuned (Q5) | PPL tuned | KL untuned | KL tuned | Top-1 tuned | Tuning time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Additive M=3 + 0.5% activation residual | 3.20 | 35.77 | **29.83** | 0.593 | **0.405** | 0.709 | 214 s |
| VQ G=4 + 0.5% activation residual | 2.18 | 565 | **107.5** | 2.66 | **1.54** | 0.448 | 177 s |

Findings: block-wise tuning cuts KL by 32% (M=3) and 42% (G=4) at unchanged bits, and the mean block output loss by 40%; no block reverted. The 3.2 bpp model is still worse than RTN 4-bit (KL 0.190) — on this 0.6B model the tied embedding/LM head encoded with the codebook is the dominant error (cf. Q7 on 8B), which the head-aware allocator (q7b) addresses. Qwen3-8B tuning is running.

## Q6 — block-wise fine-tuning on Qwen3-8B (Mac Studio) — reported result

Result: `*_q6_finetune_qwen3_8b.json` (+ `.md`); same protocol and settings as the dev Q6 run (embedding and LM head encoded with the codec, 100 steps per block, 64 calibration sequences, full WikiText-2 test, KL on 16 held-out validation sequences). Untuned baseline = the Q5 8B rows (same encodings, same seed).

| Codec | Model bpp | PPL untuned → tuned | KL untuned → tuned | Top-1 tuned | Block loss | Tuning time | Peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Additive M=3 + 0.5% activation residual | 3.18 | 11.316 → **11.223** | 0.248 → **0.228** (−8%) | 0.792 | −40% | 43 min | 34 GiB |
| VQ G=4 + 0.5% activation residual | 2.18 | 18.70 → **16.49** | 0.742 → **0.610** (−18%) | 0.676 | −45% | 33 min | 30 GiB |

Findings:

1. **On 8B the block objective transfers poorly**: every block's output error falls by 40–45% (none reverted), yet whole-model KL falls only 8–18% (dev model: 32–42%). The end-to-end error is dominated by what block-wise tuning does not touch — the codebook-encoded LM head (Q7: protecting it alone takes KL 0.248 → 0.162 at the same codec) — and by error accumulation across blocks that a per-block target ignores.
2. ΔCE on the validation sequences rises slightly for M=3 (+0.126 → +0.135) while KL and PPL fall: the tuned model moves closer to BF16 in distribution but not uniformly in likelihood; the gain at 3.2 bpp is within what the LM-head choice alone changes.
3. Next: tune on top of the head-aware allocation (LM head at higher precision, q7b), and try the end-to-end KD objective (`finetune_e2e`) on the 8B, which targets the final logits instead of block outputs.

## `suc compress` — unattended pipeline, dev model end to end (laptop, indicative)

`suc compress <config>` (`suc/pipeline/compress.py`, `suc/pipeline/stages.py`) runs these stages: preflight → calibrate → reference → [candidates] → plan → encode → [finetune] → container (SUC1) → evaluate → report. Each stage writes its artifacts and a manifest under `run_dir/<stage>/`. A rerun skips a stage when its fingerprint is unchanged and its outputs are present. The fingerprint covers the non-default config fields, the model hash, the upstream output sha256s and a stage version. The git commit is recorded but is not part of the fingerprint. Encode resumes block by block. Evaluation decodes the tensors **from the SUC1 file**. Configs: `compress_qwen3_0.6b.yaml`, `compress_qwen3_8b.yaml` (uniform M=4 + 0.5% res), `compress_qwen3_8b_greedy3.5.yaml`.

Dev run (Qwen3-0.6B, uniform additive M=4 K=256 G=8 rot + 0.5% activation-selected int8 residuals, shared q/k/v and gate/up rotation, embedding affine 4-bit g64, no fine-tuning; result `20261001T112938Z_compress_qwen3_0.6b.json`):

| | Model bpp | File bpp | File | PPL (32k tokens) | KL | Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 16 | — | — | 19.961 | — | — |
| SUC1 file | 4.293 | 4.300 | 305.5 MiB | 22.783 | 0.1616 | 0.807 |

This matches the Q7 dev row "uniform M=4 + 0.5%" (4.29 bpp, KL 0.163). The SUC1 checksum, 16 sampled encodings (bit-exact) and 4 raw tensors were verified; the reference decoder differs from the codec decode by 0. Laptop timings are not meaningful: the machine was swapping heavily under other jobs (encode took 4938 s).

Memory (D8), recorded in every result under `streaming`:
- **Not streaming-capable** (they load the whole BF16 model): `reference`, `candidates` (Fisher needs a full backward pass and keeps float32 F for every allocated weight, about 2× the BF16 size) and `evaluate`.
- **Hold every encoding** (about the compressed size): `finetune` and `container` (`write_suc1` takes an in-memory dict).
- **Layer-streamed:** `calibrate`, `encode` and `finetune` (lazy model; each block is released after tuning through the new `after_block` hook).

Preflight estimates each stage's peak and fails early. These stages are the gap to close before models larger than RAM.

Operational lessons:
- **Keep `run_dir` on a local disk.** With the run dir on the SMB share (about 14 MB/s), MLX's lazy `mx.load` of the 2.5 GB reference-logits artifact ran inside a Metal command buffer and tripped the GPU watchdog (`kIOGPUCommandBufferCallbackErrorTimeout`). Artifacts are now read on the CPU stream first. The Studio checkout is on local disk.
- **Defaulted config fields must not invalidate stages.** A `CodecConfig` field added by another change invalidated finished stages until fingerprints switched to non-default values only.

## Q5b — column-structured residuals at equal bits (dev model, laptop, indicative)

Motivation: K5 on Qwen3-8B — the CSR residual pass costs 10.6–12.6 ms per batch-1 decode step and keeps residual members out of the fused q|k|v / gate|up launches. Code: `ResidualCodec(structure="columns", column_share=…)` (`suc/quant/residual.py`; config `residual_structure` / `residual_column_share`), runtime `structured_residual_metal` + `SucFusedResidual` (`suc/runtime/compressed_model.py`), SUC1 `residual_type = columns_int8` (docs/format.md). Tests: `tests/unit/test_residual_columns.py`.

**Where residuals live.** `ResidualCodec` computes R = W − decode(base) with `decode` returning the ORIGINAL (unrotated) weight, so positions are selected and applied in the original basis; the Hadamard rotation only acts inside the base codec. This is what makes a column structure possible (in the rotated basis a massive-activation channel is spread over all columns). Caveat for residual-stream rotation folding (`suc/models/rotation_folding.py`): after folding, the input of q/k/v/gate/up IS rotated, E[x²] is flattened and the column concentration below no longer holds for those projections (o_proj / down_proj inputs are unaffected unless v-heads are folded); residual selection must then use statistics of the folded model.

**Step 1 — structure of the current selection** (additive M=3 + 0.5% int8, importance E[x_j²]·R², Phase 2 statistics; mean over the 28 layers; scratch analysis, not a result file). Coverage = fraction of selected entries in the top-c input columns; "mass" = fraction of the total predicted loss Σ E[x²]R² removed if the stored entries were exact; c_eq = columns that the CSR bit budget pays for (out·8 + 32 bits per column).

| Role | c_eq | top-8 | top-16 | top-32 | top-64 | columns hit | mass CSR | mass columns (c_eq) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q_proj | 18 | 0.67 | 0.91 | 0.99 | 1.00 | 51 | 0.460 | 0.531 |
| k_proj / v_proj | 17 | 0.67 | 0.91 | 0.99 | 1.00 | 47 | 0.46 | 0.52 |
| gate_proj / up_proj | 19 | 0.73 | 0.92 | 0.99 | 1.00 | 64 | 0.47 | 0.52 |
| o_proj | 36 | 0.25 | 0.40 | 0.58 | 0.76 | 357 | 0.166 | 0.181 |
| down_proj | 57 | 0.21 | 0.33 | 0.48 | 0.66 | 538 | 0.243 | 0.261 |

Projections reading the normed residual stream (q/k/v, gate/up) have ~99% of their residual entries in 32 columns; o_proj / down_proj are spread out, except the super-weight tensor `layers.2.mlp.down_proj` (99.9% of the mass in 8 columns). No row is concentrated (max row share ≤ 0.3%). Because a dense column costs 8 bits per entry instead of 30 (positions + value), columns remove at least as much predicted loss as CSR at equal bits on every role.

**Step 3 — quality at equal actual bits** (`experiments/configs/q5b_residual_structure_qwen3_0.6b.yaml`, result `20261001T131910Z_q5b_residual_structure_qwen3_0.6b.json`; rotation_groups on; git 78c3cb8 + uncommitted WIP — the codec code was the final one; laptop, 32k test tokens, KL on 16 validation sequences):

| Codec (0.5% budget) | bpp | PPL | KL | Top-1 |
|---|---:|---:|---:|---:|
| RTN affine 4-bit | 4.501 | 22.838 | 0.1896 | 0.795 |
| M=4, CSR | 4.214 | 22.999 | 0.1755 | 0.798 |
| **M=4, columns** | **4.212** | 22.827 | **0.1584** | 0.806 |
| M=4, columns 0.5 + CSR tail 0.5 | 4.214 | **22.518** | 0.1616 | **0.812** |
| M=3, CSR | 3.203 | 33.905 | 0.5763 | 0.665 |
| **M=3, columns** | **3.202** | 31.899 | 0.5170 | 0.682 |
| M=3, columns 0.75 + tail 0.25 | 3.203 | **31.388** | **0.5106** | 0.682 |
| M=3, columns 0.5 + tail 0.5 | 3.203 | 31.992 | 0.5329 | 0.677 |

Findings: at equal (slightly lower) bits, pure columns beat CSR on KL by 10% (M=3: 0.576 → 0.517; M=4: 0.176 → 0.158) and on PPL (−5.9% / −0.7%). A CSR tail is not needed for quality (0.75/0.25 is marginally better at M=3, 0.5/0.5 on M=4 PPL, within one run's noise); pure columns is the runtime-simplest choice. Single seed, single run: repeatability not yet measured. M=4 columns now beats RTN 4-bit on every metric at 0.29 bpp less (KL 0.158 vs 0.190).

**Step 4 — runtime.** Column residual R_c (c, out) int8, column-major, one fp16 scale per column; `structured_residual_metal`: one thread per output row and up to 8 batch rows, the threadgroup first gathers x[cols]·scale into threadgroup memory, then independent coalesced int8 loads + FMAs; an optional CSR tail in the same launch; the result is rounded to the activation dtype in the kernel. Residual members now fuse: `SucFusedResidual` (q|k|v, gate|up) = 1 SUC launch + 1 float32 output-rotation launch + 1 residual launch; `FUSE_RESIDUAL = False` restores the previous path for A/B. Launches per block (graph count, steady state): no residual 12, CSR before 32, **CSR fused 16, columns 16**. Exactness: e2e check on Qwen3-0.6B (additive M=3 + columns, float32) KL 1.5e-8, top-1 1.000; tiny-model tests vs the decoded weights (columns / hybrid / CSR / mixed groups, float32 and bf16); SUC1 reference decoder bit-identical to the codec decode.

Timing on the laptop could not be measured cleanly (other agents' jobs ran concurrently; e2e decode steps were 5× slower than quiet runs). Indicative residual-pass-only microbenchmark (`benchmarks/q5b_residual_pass.py`, compiled, min of 20, contended M3): Qwen3-0.6B shapes 2.4–4.1 ms per step for every variant, equal to the 4-launch floor (launch-bound); Qwen3-8B shapes, B=1 / 4 / 16: CSR unfused 5.0 / 16.4 / 44.2 ms, columns 4.8 / 6.3 / 22.4 ms, floor 4.4 / 2.9 / 5.0 ms (first kernel version, before the in-kernel rounding). Not a reported number: the Studio runs below decide.

Studio commands (sequential): `benchmarks/q5b_residual_pass.py --shape qwen3_8b --json …` and `benchmarks/e2e_generate.py --model Qwen/Qwen3-8B --variants add3_g8_k256_res,vq_g4_k256_res` with `--no-fuse-residual` (before), default (CSR fused) and `--residual-structure columns` (after).

## R1/R3 — removing online rotations: one-sided is free, residual-stream folding fails (dev model, laptop)

Motivation: on Qwen3-8B the online Hadamard rotations cost 10.8 ms of a 25.9 ms batch-1 decode step (K5 8B). Results: `20261001T113300Z_r1_rotation_sides_qwen3_0.6b.json`, `20261001T130811Z_r3_rotation_folding_qwen3_0.6b.json`, `20261001T085849Z_fold_exactness_qwen3_0.6b.json` (+ `…083250Z_…vheads…`). Laptop runs under contention: quality numbers are valid, no timings. Single seed.

**R1 — two-sided (U W Vᵀ) vs one-sided (W Vᵀ) rotation**, + 0.5% activation residual, shared q/k/v and gate/up rotation:

| Codec | Sides | PPL | KL | Top-1 |
|---|---|---:|---:|---:|
| additive M=3 (3.20 bpp) | two-sided | 33.91 | 0.576 | 0.665 |
| additive M=3 | **one-sided** | 34.48 | **0.525** | 0.677 |
| VQ G=4 (2.18 bpp, broken either way) | two-sided | 467.1 | 3.164 | 0.272 |
| VQ G=4 | one-sided | 293.4 | 2.279 | 0.347 |

The output rotation buys nothing (per-role rel-MSE equal or slightly lower one-sided): dropping it removes 4 of the 8 online rotations per block. Replication on Qwen3-8B queued (`r1_rotation_sides_qwen3_8b`).

**R3 — QuaRot-style folding of a residual-stream rotation into the weights.** The fold is exact (float32 folded vs original: KL 7.7e-11, top-1 1.0; tied/untied heads, dense and MoE, block-streamed). Encoded, it is a **negative result**:

| Variant (one-sided) | PPL | KL | Unfolded (R1) |
|---|---:|---:|---:|
| additive M=3, folded | 192.4 | 2.20 | 34.5 / 0.525 |
| additive M=3, folded + per-head v/o | 202.6 | 2.26 | |
| VQ G=4, folded | 50,554 | 8.24 | 293 / 2.28 |

Diagnosis: Qwen3-0.6B carries a massive activation on hidden channel 35 (|h| ≈ 7168 from layer 2 on). Unfolded, the RMSNorm weight scales it down before q/k/v and gate/up; folded, the norm weight is inside the encoded weights and the (direction-uniform) quantization error is multiplied by the raw massive value, which destroys the attention-sink tokens. Quantizing one role group at a time confirms it — only readers of the residual stream break (KL unfolded → folded: q/k/v 1.88 → 4.64, gate/up 0.95 → 11.78; o_proj 0.21 → 0.24, down_proj 1.99 → 2.15). Making folding work would need an activation-covariance-aware codec (GPTQ-like) or an exact rank-1 correction along the massive direction; not pursued.

Decision: runtime default becomes **one-sided** rotations (pending the 8B replication), optionally with the per-head v/o fold (quality-neutral on the one measured row; additive row queued as R4): online rotations per block 8 → 4 (→ 3). The remaining input rotations (shared q/k/v, shared gate/up, down_proj) can be fused into the RMSNorm and SwiGLU element-wise kernels without any change in math. 8B timings of each setting are queued.

## Q5b — column-structured residuals: runtime on Qwen3-8B (Mac Studio) — reported result

Results: `20261001T133035Z_q5b_residual_pass_qwen3_8b.json` (residual pass alone, synthetic 8B shapes, whole model of 36 blocks, median of 30) and `20261001T133057Z_e2e_q5b_csr_unfused…`, `…133900Z_e2e_q5b_csr_fused…`, `…134737Z_e2e_q5b_columns…_qwen3_8b.json` (end-to-end decode; codes from the timing-oriented fast k-means, magnitude-selected residuals — cost depends only on entry/column counts). Correctness: logit KL vs decoded weights 0.0009–0.0056 in every variant.

Residual pass alone, ms per decode step (36 blocks):

| Variant | Launches / block | B=1 | B=4 | B=16 |
|---|---:|---:|---:|---:|
| CSR, unfused | 7 | 3.90 | 13.03 | 40.37 |
| CSR, fused q\|k\|v and gate\|up | 4 | 3.42 | 12.23 | 39.25 |
| **columns** | 4 | **1.12** | **1.20** | **2.13** |
| columns 0.5 + CSR tail | 4 | 2.08 | 6.68 | 19.03 |
| launch floor (empty kernels) | 4 | 0.78 | 0.74 | 0.77 |

End-to-end decode, tok/s:

| Variant | VQ G=4 + res, B=1 / 4 / 16 | additive M=3 + res, B=1 / 4 / 16 |
|---|---|---|
| CSR, unfused (before) | 26.7 / 67.2 / 124.3 | 23.8 / 59.6 / 107.1 |
| CSR, fused | 27.5 / 57.5 / 112.9 | 19.3 / 37.9 / 95.7 |
| columns | 26.9 / **78.0** / **157.6** | 18.9 / 45.5 / 119.9 |

Findings:

1. **The column kernel itself is cheap**: 1.1 ms per step at B = 1 (0.3 ms above the launch floor) and nearly flat in batch, vs 3.4–40 ms for CSR — the dense int8 GEMV on gathered columns removes the scatter.
2. **End to end it does not yet pay off at B = 1**: decode stays at ~27 tok/s (VQ) and the profile still attributes ~9–11 ms to "residual". The pass costs ~1 ms in isolation, so the residual variants must be losing something else in the full model — the fused groups with residual members double the additive kernel time (19.6 → 31.5 ms), i.e. the residual-carrying fused path does not take the same fast kernel/compiled path as the plain one. At B = 4/16 columns help (+16% / +27% for VQ).
3. Quality (dev model): columns beat CSR at equal bits (KL −10%, Q5b dev section). Columns become the default residual structure once the fused-path regression is fixed; the 8B quality run is queued.

## Q7b — embedding and LM head as allocation candidates on Qwen3-8B (Mac Studio) — reported result

Result: `*_q7b_allocate_head_qwen3_8b.json` (+ `.md`). Same block candidates as Q7; embedding and untied LM head (1.24 B params) enter the same greedy budget with candidates affine 4/6/8-bit (g64) and BF16. Budgets are over all 254 tensors, so allocated bpw ≈ model bpp. Spearman(predicted, measured KL) across plans 0.929.

| Plan | Model bpp | PPL | KL | Blocks bpw | Embed / LM head chosen |
|---|---:|---:|---:|---:|---|
| greedy 3.0 | 3.00 | 12.150 | 0.2412 | 2.73 | affine 4 / affine 4 |
| uniform M=3 + 0.5% res | 3.38 | 10.901 | 0.1621 | 3.17 | (affine 4 fixed) |
| greedy 3.5 | 3.50 | 10.697 | 0.1378 | 3.32 | affine 4 / affine 4 |
| greedy 4.0 | 4.00 | 10.275 | 0.0925 | 3.91 | affine 4 / affine 4 |
| **uniform M=4 + 0.5% res** | **4.23** | **10.081** | **0.0751** | 4.18 | (affine 4 fixed) |
| greedy 4.25 | 4.25 | 10.182 | 0.0773 | 4.21 | affine 4 / affine 4 |
| uniform RTN 4-bit | 4.50 | 10.088 | 0.0922 | 4.50 | (affine 4 fixed) |

Findings:

1. **The allocator never buys more than 4-bit for the embedding / LM head**: at every budget the Fisher proxy prefers spending bits in the blocks. Q7's lesson was "don't encode the LM head with a low-bit codebook"; 4-bit affine already captures it, and higher head precision is predicted not to pay. (Not yet verified by a direct measurement of a uniform M=4 plan with an 8-bit head — cheap follow-up.)
2. **Greedy does not beat uniform M=4 + residual at ≈4.2 bpp** (KL 0.0773 vs 0.0751): near 4 bpp the greedy plans mix in affine 4-bit and a few BF16 tensors that the additive Fisher proxy over-values. The proxy ranks plans well (ρ 0.93) but its per-tensor trade-offs are imperfect at the top of the range. Below ~3.5 bpp, greedy remains the only way to reach the budget with good quality (3.0 bpp: PPL 12.15).
3. Best quality per bit on Qwen3-8B is unchanged: **uniform additive M=4 + 0.5% activation residual, head at affine 4-bit — 4.23 bpp, PPL 10.08, KL 0.075** (beats RTN 4-bit at 4.50 bpp on both metrics).

## Q5x — shared codebooks per MoE expert stack: encode time (laptop M3, synthetic, indicative)

Motivation: on Qwen3-30B-A3B every one of the 18,432 expert tensors ran its own k-means (~1–1.5 s per codec on the M1 Ultra, capped by `fit_blocks`, so no cheaper than an 8B tensor); Q7 on 30B-A3B was stopped at > 28 h of candidate scoring. New unit (D6): the **expert stack** = all E experts of one (layer, projection). `codebook_scope: expert_stack` (`suc.quant.stacks`, names `…_xs`) fits ONE codebook set per stack on a pooled sample (ceil(fit_blocks / E) prepared blocks per expert), shares the rotation seeds (gate/up of a layer share V), then per expert only assigns indices + row scales + residual. `allocation_unit: expert_stack` makes each stack one allocation unit (bits and Fisher loss summed). SUC1 stores the shared codebook once (content dedup); `SucSwitchLinear` keeps one (1, M, K, G) codebook and one sign row per stack.

Measurement: `scratchpad` script on synthetic stacks of Qwen3-30B-A3B shape (128 experts, Student-t(5) weights, log-normal E[x²] for the activation residual), encode + decode per expert, M3 laptop (not a reported timing):

| Stack (128 experts) | Codec | Per-expert codebooks (s/stack, extrapolated) | Shared (s/stack) | Speed-up | rel. MSE per-expert → shared | bpw shared |
|---|---|---:|---:|---:|---|---:|
| gate/up 768×2048 | additive M=3 + 0.5% res | 370 | 14.0 | 26× | 3.37e-2 → 3.55e-2 (+5%) | 3.154 |
| gate/up | additive M=4 + 0.5% res | 533 | 17.6 | 30× | 1.17e-2 → 1.26e-2 (+8%) | 4.154 |
| gate/up | VQ G=4 + 0.5% res | 161 | 7.8 | 21× | 9.48e-2 → 9.56e-2 (+1%) | 2.153 |
| down 2048×768 | additive M=3 + 0.5% res | 391 | 13.6 | 29× | 3.36e-2 → 3.54e-2 (+5%) | 3.167 |
| down | additive M=4 + 0.5% res | 476 | 17.7 | 27× | 1.16e-2 → 1.26e-2 (+8%) | 4.167 |
| down | VQ G=4 + 0.5% res | 150 | 7.8 | 19× | 9.44e-2 → 9.54e-2 (+1%) | 2.166 |

Findings (indicative):

1. **19–30× less encode time per stack**; shared cost grows with the stack's parameters (assignment), not with the number of k-means fits. Projection for Qwen3-30B-A3B on the M1 Ultra (assuming the laptop/Studio ratio of ≈ 2.5–3 seen on per-tensor encodes): one shared codec over the 144 stacks ≈ 7–16 min; Q7 candidate scoring (4 codebook + affine + BF16 candidates) ≈ 1–1.5 h instead of > 28 h, each plan's reconstruction ≈ 15–20 min instead of 5–7 h. A ~180B MoE (~170 B expert parameters) at ≈ 7e-8 s/param (laptop, M=3) ≈ 1–1.5 h per shared codec on the Studio; per-expert codebooks cost ≈ 1–1.5 s × the number of expert tensors, e.g. ≈ 10–15 h per codec for a 94-layer × 128-expert layout (36k tensors). (Fisher / reference / evaluate still load the whole BF16 model, the D8 gap of `suc compress`, so a 180B model needs that fixed first.)
2. **Sharing costs weight error on i.i.d. synthetic experts**: +1% (VQ) to +8% (M=4) relative MSE, partly offset by the codebook bits saved (≈ 0.06 bpw for M=3, 0.08 for M=4 at this shape). A 4× larger pooled sample closes only ~17% of the M=4 gap (1.264e-2 → 1.248e-2, 1.7× the time), so the gap is the cost of one codebook for 128 experts, not undersampling. Whether it matters for the model is the Q5x run (`experiments/configs/q5x_expert_stacks_qwen3_30b_a3b.yaml`, PPL / KL per-expert vs shared on 30B-A3B).
3. Fine-tuning (Q6) of shared stacks is not implemented (per-expert tuning would un-share the codebooks); the configs reject `finetune` together with `codebook_scope: expert_stack`.

## K6 — residual-variant decode regression diagnosed; input rotations fused into norm / SwiGLU (laptop, indicative)

Motivation: on Qwen3-8B (Q5b runtime, Studio) the column residual pass cost 1.1 ms in isolation but the residual variants stayed at ~27 tok/s (profile "residual" ≈ 10 ms), and fused groups with residual members doubled the additive kernel time (19.6 → 31.5 ms).

**Diagnosis.**

1. *Residual pass*: `benchmarks/q5b_residual_pass.py` launched all 4·36 residual kernels **independently**, so the GPU overlapped them (a throughput number). In a decode step every launch is on the critical path. The first column kernel (one thread per output row, a serial loop over all c columns, 1-byte loads, 16 threadgroups for a 4096-row output) was latency-bound: 37–72 µs per dependent launch on 8B shapes (M3) vs a 7 µs launch floor. New `--dependent` mode times chains of dependent launches.
2. *Fused additive kernel*: `fused_matmul_metal` (q|k|v, gate|up) always read codebooks from device memory (CBTG = 0), while the unfused `SucLinear` resolves to threadgroup-staged codebooks on the M1 Ultra (K3: 1.2–3× faster, M=3 most). The plain fused additive variant was equally slow (31.4 ms): fusion, not the residual, caused the doubling; Q5b's CSR-unfused run only looked faster because its residual members were not fused.
3. Profile attribution (`residual_ms = full − no_residual`) was correct; the cost was real.

**Fixes.** (a) `structured_residual_metal`: 4 consecutive rows per lane (one 4-element vector load per column), columns and CSR-tail entries split over 8 simdgroups, partial sums reduced in threadgroup memory. Dependent launch on 8B shapes (M3): o 37 → 12 µs, down 68 → 16 µs, q|k|v 45 → 24 µs; whole step (36 blocks, `q5b_residual_pass.py --shape qwen3_8b --dependent`) B=1 7.5 → 4.4 ms (launch floor 2.3 ms, bytes ≈ 1.4 ms on the M3). (b) Fused kernels stage **each threadgroup's own segment codebooks** when the placement resolves to threadgroup and the segment boundaries are multiples of the threadgroup rows (all Qwen3 shapes); `--no-fused-staging` restores the old path for A/B. Not measurable on the M3 (device placement wins there): the Studio run decides.

**Elementwise fusion (K6).** `RotatedRMSNorm` (input / post-attention norm → q|k|v / gate|up) and `SucMLP` (SwiGLU → down_proj) compute the op and its input rotation in one launch (`suc.kernels.rotation.rms_norm_rht` / `swiglu_rht`). They also write the unrotated output only when a residual reads it. RMSNorm output is bit-identical to `mx.fast.rms_norm`; SwiGLU is float32 math, rounded once. Launches per block (whole block, B=1, tiny Qwen3): two-sided 25 → 22, one-sided 21 → 18, one-sided + residual 25 → 22. SUC custom kernels per block: one-sided 8 = q|k|v kernel, o rotate_in + kernel, gate|up kernel, down kernel + the 3 fused norm/SwiGLU launches; no output-rotation launch. `e2e_generate.py` now defaults to one-sided (`--two-sided` to restore) and records `launches_per_block`. Exactness: f32 logit KL vs decoded weights −3e-8 … 6e-8 (rounding floor) on Qwen3-0.6B; tests `tests/unit/test_elementwise_rotation.py`.

Qwen3-0.6B e2e decode ms/step, one-sided, laptop M3 (indicative only; 0.6B is launch-bound on the M3, run-to-run noise ~10%):

| Variant | B=1 before → after | B=4 | B=16 |
|---|---|---|---|
| VQ G=4 | 6.41 → 6.46 | 9.50 → 9.91 | 22.76 → 22.20 |
| VQ G=4 + CSR residual | 12.85 → 9.59 | 36.6 → 15.1 | 67.6 → 33.1 |
| VQ G=4 + columns residual | 7.84 → 8.03 | 11.47 → 11.51 | 24.08 → 25.42 |

The 8B before/after runs on the Studio are pending.

## MoE from codes — Qwen3-30B-A3B end to end (Mac Studio) — reported result

Result: `*_moe_e2e_check_Qwen3-30B-A3B.json` (`benchmarks/moe_e2e_check.py`; every eligible tensor incl. 18,432 expert tensors encoded with VQ G=4 K256, fast k-means — timing-oriented codes; per-expert codebooks, i.e. before expert stacks; router BF16; embedding/head MLX 4-bit; run at 1ef84c5).

| | Resident weights | B=1 decode tok/s | B=4 decode tok/s | B=1 prefill tok/s |
|---|---:|---:|---:|---:|
| BF16 | 61.1 GB | — | — | — |
| MLX 4-bit | 17.2 GB | **62.7** | **146.2** | 794 |
| SUC VQ G=4 (2.09 bpw linears) | **8.2 GB** | 40.5 | 117.7 | 430 |

Module check (layer-0 expert stacks vs the reference decoder, float32): rel. error ≤ 9e-7. Whole model, compressed vs reconstructed (both BF16, 256 tokens): **KL 0.032, top-1 0.930** — an order of magnitude above the dense 8B check (0.001–0.007). Likely cause: BF16 rounding differences flip top-k expert routing (a discontinuity the dense model doesn't have); there is no float32 noise floor at this size. To check: routing agreement per layer between the two models — `benchmarks/moe_routing_check.py` (per-layer top-k agreement, router margins of flipped tokens, hidden-state error growth, free-running and teacher-forced, plus a no-SUC control: the same decodes rounded to BF16 stochastically). Tiny random Qwen3-MoE (laptop, indicative only): in float32, compressed vs reconstructed has zero flips (KL ~1e-9, rel. error ≤ 6e-7 per layer); in BF16 the compressed model (KL 4.2e-4, set agreement 0.952) is indistinguishable from the stochastic-rounding control (KL 4.9e-4, 0.945), and against the exact float32 model the reconstructed BF16 and the compressed BF16 are equally far (KL 4.6–7.8e-4 vs 6.1–7.3e-4, 3 seeds). Studio run pending.

Findings: the 30B-A3B MoE runs from codes in 8.2 GB (vs 17.2 GB for MLX 4-bit, 61 GB BF16) — the capacity win SUC targets — at 65% of MLX 4-bit's batch-1 decode speed (before the K6 runtime fixes).

## Q5b — column-structured residuals: quality on Qwen3-8B (Mac Studio) — reported result

Result: `*_q5b_residual_structure_qwen3_8b.json` (embedding/LM head encoded with the codec, Q5 protocol; full WikiText-2 test; KL on 16 held-out validation sequences). At equal bits columns are better than CSR in every codec, but by less than on the dev model:

| Codec (0.5% residual budget) | bpp | PPL CSR → columns | KL CSR → columns |
|---|---:|---:|---:|
| additive M=4 | 4.18 | 10.080 → **10.073** | 0.0844 → **0.0815** (−3.4%) |
| additive M=3 | 3.18 | 11.382 → **11.269** (cols 0.75: 11.247) | 0.2606 → **0.2557** (−1.9%; cols 0.75: 0.2549) |
| VQ G=4 | 2.18 | 18.721 → **18.536** | 0.7523 → **0.7356** (−2.2%) |
| RTN 4-bit | 4.50 | 10.088 | 0.0922 |

Decision: columns become the default residual structure (better quality, and a runtime pass that is cheap once latency-optimised — K6).

## R1 — rotation settings, decode timing on Qwen3-8B (Mac Studio) — reported result

Results: `*_e2e_generate_qwen3_8b_rot_{baseline,onesided,vfold_onesided,ceiling}.json` (run at 1ef84c5, before the K6 runtime fixes; timing codes, no residuals). Decode tok/s, and batch-1 rotation / kernel ms per step:

| Setting | Online rotations / block | VQ G=4 B=1 / 4 / 16 | rot / kern ms | add. M=3 B=1 / 4 / 16 |
|---|---:|---|---|---|
| two-sided (before) | 8 | 38.2 / 107.2 / 225.6 | 10.7 / 13.1 | 23.0 / 51.7 / 154.7 |
| **one-sided** | 4 | **46.1** / 124.3 / 242.4 | 7.0 / 12.7 | 25.8 / 54.9 / 161.7 |
| one-sided + v/o head fold | 3 | 46.3 / 126.8 / 244.9 | 6.3 / 13.1 | 26.4 / 55.4 / 162.8 |
| full fold (quality-breaking, ceiling only) | 1 | 49.5 / 133.7 / 250.5 | 4.1 / 12.6 | — |
| MLX 4-bit | — | 75.3 / 137.9 / 193.1 | — | — |

Findings: one-sided rotation gives +21% at B=1 (38.2 → 46.1 tok/s) for free in quality (dev R1). Rotation time does not scale with the count (4 rotations still cost 7.0 ms, 1 costs 4.1 ms): each standalone rotation launch is latency-bound (~50 µs), which is why K6 fuses them into the RMSNorm / SwiGLU kernels. The additive M=3 path stays kernel-bound (31 ms, fused groups reading codebooks from device memory — fixed by K6 staging).

## R1/R4 — quality replication (Mac Studio) — reported result

`*_r1_rotation_sides_qwen3_8b.json` (Q5 protocol, embedding/head encoded with the codec) and `*_r4_vhead_fold_qwen3_0.6b.json`:

| Model / codec | two-sided PPL / KL | one-sided PPL / KL | one-sided + v/o fold |
|---|---|---|---|
| 8B additive M=3 + 0.5% res (3.18 bpp) | 11.382 / 0.2606 | **11.076 / 0.2456** | — |
| 8B additive M=4 + 0.5% res (4.18 bpp) | 10.080 / 0.0844 | 10.130 / **0.0772** | — |
| 0.6B additive M=3 + 0.5% res (3.20 bpp) | 33.91 / 0.576 | 34.48 / 0.525 | 34.35 / 0.524 |

One-sided rotation lowers KL in every case (8B: −6% / −8.5%); PPL moves ±3%. With columns residuals (Q5b) the 8B M=4 configuration would combine both gains. Decision: **one-sided is the default for new configs** (`rotate_out: false`; the CodecConfig default stays two-sided so old configs reproduce), and the v/o head fold is quality-neutral (optional, −1 rotation per block).

## K6 — runtime fixes measured on Qwen3-8B (Mac Studio) — reported result

Results: `*_e2e_k6_{after,nofuse_elementwise,nostaging}_qwen3_8b.json`, `*_q5b_residual_dependent_qwen3_8b.json` (one-sided rotation, column residuals; logit KL vs decoded weights 0.0016–0.0037 everywhere). Decode tok/s:

| Variant | before K6 (one-sided, R1 table / Q5b) | K6, all on | K6, elementwise fusion off | K6, fused staging off |
|---|---|---|---|---|
| VQ G=4 | 46.1 / 124.3 / 242.4 | 43.4 / 120.1 / 244.5 | **47.3 / 126.2 / 252.9** | — |
| VQ G=4 + columns res | 26.9 / 78.0 / 157.6 (two-sided) | 43.6 / 96.5 / 219.0 | **45.4 / 103.2 / 224.6** | — |
| additive M=3 + columns res | 18.9 / 45.5 / 119.9 (two-sided) | 34.9 / 89.1 / 140.8 | **35.0 / 91.9 / 147.9** | 21.9 / 51.5 / 123.7 |
| MLX 4-bit | 75.3 / 137.9 / 193.1 | 75.0 / 137.5 / 192.0 | | |

(B = 1 / 4 / 16.) Findings:

1. **Residuals are now almost free**: VQ + residual 45.4 vs 47.3 tok/s without (was 26.9). The dependent-chain residual pass costs 3.2 ms/step at B = 1 for the whole model.
2. **Staged codebooks in fused groups: +60% for additive codebooks** (21.9 → 35.0 tok/s; kernel 31.8 → 14.5 ms/step).
3. **Fusing the rotation into RMSNorm/SwiGLU is slower on the M1 Ultra** (43.4 vs 47.3; rotation 8.3 vs 6.8 ms) → `FUSE_ELEMENTWISE` now defaults to off; candidate for per-device autotuning.
4. Batch-1 status: VQ G=4 at 63% of MLX 4-bit (47.3 vs 75.0), additive M=3 + residual at 47%; at B = 16 SUC is faster (252.9 / 147.9 vs 192.0). Remaining B = 1 cost: rotations 6–7 ms/step (each standalone launch latency-bound) and the kernels themselves (12 ms vs MLX's 10.2 ms for all linears).

## R5 — batch-1 online rotations: one launch over many cores; per-device runtime tuning (laptop, indicative)

Motivation: on Qwen3-8B (Studio, K6) the ~4 input rotations per block cost 6–7 ms per decode step (~45–50 µs each); even 1 rotation per block cost 4.1 ms. The pre-R5 kernel ran the whole n log n transform in ONE threadgroup per row (strided layout, scalar loads, H₁₂ read from memory, float32 signs): latency-bound on a single GPU core.

**Measurement.** `benchmarks/rotation_latency.py`: chains of 144 dependent launches, each reading the output of a many-threadgroup copy (the decode situation: the projection kernel's output was just written by many cores); the copy chain alone is subtracted; ops interleaved rep by rep. `--isolated` chains the launches directly — that flatters single-threadgroup kernels (the vector stays in one core's L1; there the old kernel looked as fast as the new one), so it is not used for decisions. `--step qwen3_8b` times a synthetic 8B decode step (random VQ G=4 codes, 36 blocks) with each rotation mode. The M3 laptop switches between two GPU clock regimes (all kernels ~2× slower in one) and was shared with other jobs: numbers below are min over 150 interleaved reps, three runs; the step benchmark was too noisy on the laptop to report (±3 µs per rotation).

**Design (suc/kernels/rotation.py, LAYOUT = "blocks").** One threadgroup per Q-element *output* block (Q = 256 for 4096, 512 for 12288 → 16 / 24 threadgroups on as many cores). With H_P = H_{P/Q} ⊗ H_Q, block (a, c1) = H_Q · Σ_(b,a1) H_m[a,b]·(−1)^popcount(c1 & a1)·σx_(b,a1): every threadgroup reads the whole (cached) input, its simdgroups sum the M·P/Q input blocks with ±1 (·H_m) coefficients, one barrier combines the partial sums, one simdgroup applies H_Q (in-register stages + simd_shuffle_xor), one vector store. Single launch, no grid-wide step; 12288 = 12·1024 in the same kernel (H₁₂ as compile-time constants). fp32 math; the result equals the reference up to fp32 summation order. Also: contiguous VEC = 4 layout with vector loads and constant H_m for the old "row" layout (kept as LAYOUT = "row"), int8 signs in SucLinear / SucSwitchLinear (signs were 2/3 of a blocks threadgroup's loads; `nosign` A/B: −1.7 µs at 4096), and the RMSNorm / SwiGLU prologues in the blocks layout (each threadgroup computes the norm's sum of squares itself, in MLX's exact order for n ≤ 4096: normed output bit-identical to `mx.fast.rms_norm`).

Per rotation, in decode context (M3, three runs, min µs; empty launch = the floor):

| n | empty launch | pre-R5 kernel | R5 row layout (VEC 4) | **R5 blocks** | blocks / floor |
|---:|---:|---:|---:|---:|---:|
| 4096 | 3.0–3.1 | 7.9–8.0 (median 9–12) | 7.2–8.2 | **6.9–7.2** (median 7.3–8.6) | 2.3× |
| 12288 | 3.2–3.7 | 14.4–14.5 (median 17–33) | 9.3–10.1 | **7.0–7.5** (median 7.9–10.5) | 2.2× |

Block shape (autotune chain, VQ-kernel producer, M3): 12288 — Q=128: 21 µs (96 threadgroups each re-reading 72 KB of input + float signs), Q=256: 10–11, Q=512: 9; 4096 — all blocks shapes 4.4–5 µs vs row 5.7. The ~1.5× floor goal is **not** reached on the M3 (2.2–2.3×); the remaining cost splits into sign + input loads (~2 µs) and the FWHT (~0.4 µs, `loadonly` A/B), i.e. about the cost of MLX's own `rms_norm` launch in the same chain. Exactness: unit tests vs the reference (rel ≤ 2e-6, all layouts / block shapes, segments, split outputs, per-row MoE signs, int8 = float32 signs bit-identically); bf16 outputs equal the pre-R5 kernel's except rare rounding-boundary cases (> 98% identical, ≤ 1 ulp); Qwen3-0.6B e2e f32 logit KL vs decoded weights 3e-8 (VQ G=4 and VQ + columns residual; rounding floor, as before).

Fusing rotations into element-wise ops in the blocks layout (negative for SwiGLU): every threadgroup must read gate *and* up (2× the input) and recompute silu for the whole row, so `swiglu_rht` costs 21–34 µs vs 11–20 for SwiGLU + separate rotation on the M3; `rms_norm_rht` ≈ separate (6.8 vs 7.0, 17.7 vs 18.1). Fusion is therefore decided per op and per device.

**Per-device tuning (suc/kernels/autotune.py).** The cache entry gains sections (each measured once, merged into older entries instead of discarding them): `rotation` (layout / block shape per size, measured for the sizes of the model being converted, candidates row + blocks Q 128–512), `fuse_elementwise` (per op, fused only if ≥ 5% faster than separate), `mma_min_batch` (vec → simdgroup-matrix crossover per M, alternating A/B over cold-cache weights; M3: 5 for M = 1 and M = 3, as the old constant). The runtime reads them when its knobs are left at None/"auto" (`rotation.LAYOUT`, `compressed_model.FUSE_ELEMENTWISE`, `vq_matmul.MMA_MIN_BATCH`); explicit values override. `convert_to_compressed` calls `ensure_tuned(rotation_sizes, hidden, intermediate)` outside compiled graphs; `python -m suc.kernels.autotune [--force]` measures all sections (~45 s on the M3). `e2e_generate.py`: `--fuse-elementwise auto|on|off`, `--rotation-layout`, `--rotation-strided` (pre-R5 kernel + float32 signs), `--mma-min-batch`.

The 8B before/after runs on the Studio are pending.

## Phase 8 — entropy coding of a real SUC1 file (laptop, CPU only) — storage-only, mostly negative

`benchmarks/entropy_coding.py` on the dev container `compress_qwen3_0.6b/container/model.suc1` (additive M=4 K=256 G=8 + 0.5% int8 residual on 196 linears, 4-bit affine embedding, 320.36 MB, 596.0 M params, **4.300 bpp**). Result: `*_entropy_coding_qwen3_0.6b.json`. Coders: zlib -9 / xz -6 in process, zstd -3 / -19 via the Homebrew CLI (`-T1`; the `zstandard` package is not a project dependency), a static interleaved rANS (`suc/entropy/rans.py`: 32-bit state, 16-bit words, 14-bit tables, one table per tensor and codebook, 8192 lockstep steps / 46 k lanes) and a split-plane Rice code for residual position gaps (`suc/entropy/measures.py`). rANS and Rice decodes are verified bit-exact against the file (and by `tests/unit/test_entropy.py`); every byte count includes tables, final lane states and stream headers.

**Index entropy** (bits/symbol, symbol-weighted over all tensors; conditional estimates as plug-in / two-fold held-out by row parity, add-0.5):

| Indices | log₂K | H₀ | H(·│left group) | H(·│same group, previous row) | H(·│previous codebook, same group) |
|---|---:|---:|---:|---:|---:|
| additive, 220.2 M codes | 8 | **7.924** (99.05%) | 7.744 / 8.162 | 7.745 / 8.162 | 7.612 / 8.026 |
| affine 4-bit embedding (16 k-row sample) | 4 | **3.720** (93.0%) | 3.717 / 3.717 | 3.672 / 3.672 | — |

The plug-in order-1 numbers (65 536 bins, ~4 samples per bin) are pure estimation bias: held out, every context *loses* against order-0 (8.03–8.16 vs 7.92). Additive codes carry no exploitable neighbour or cross-codebook dependence; only the later codebooks are slightly non-uniform (typical MLP tensor: H₀ 7.99 / 7.99 / 7.93 / 7.78 for m = 0..3). A decodable adaptive previous-codebook context coder (no stored tables, 8 row chunks, smoothed towards the marginal, β = 256 best of 16/64/256 on 14 tensors) confirms it: 219.63 MB vs 218.62 MB order-0 rANS for the additive codes (its learning curve only converges towards the order-0 rate). Negative result recorded.

**Per section** (bytes; best coder per section; "SUC-EC" = rANS indices + rANS residual values + Rice positions, best generic coder elsewhere):

| Section | stored | best generic (coder) | SUC-EC | saving |
|---|---:|---:|---:|---:|
| indices (additive 220.20 M + embedding 77.79 M) | 297.99 M | 291.38 M (zlib-9) | **291.12 M** (additive 218.62, embedding 72.50) | 2.3% (additive 0.72%, embedding 6.8%) |
| scales (row scales; embedding scales+biases 9.72 M) | 10.41 M | **4.69 M** (xz-6, byte-shuffled) | 4.69 M | 55% |
| residual (positions 5.91 M, values 2.20 M) | 8.11 M | 5.76 M (xz-6) | **4.58 M** (Rice 2.55 M, rANS 2.03 M) | 44% |
| codebooks (f16) | 3.21 M | **2.74 M** (xz-6, shuffled) | 2.74 M | 15% |
| directory (JSON) | 0.48 M | **0.045 M** (zstd-19) | 0.045 M | 91% |
| raw (norms, bf16) | 0.13 M | 0.06 M (xz-6, shuffled) | 0.06 M | 53% |
| **file** | **320.36 M, 4.300 bpp** | 304.68 M, 4.089 bpp | **303.24 M, 4.070 bpp** | **5.3%** |

Residual positions: fixed 22 bits each (47.3 Mbit) vs log₂C(numel, n) = 20.0 Mbit vs Rice 20.4 Mbit; residual int8 values H₀ = 6.99 bits. Whole file with zstd -3: 320.36 → 309.44 MB (3.4%).

**Where the 5.3% comes from:** about 11 MB of the 17.1 MB saved are the 4-bit affine *embedding* (indices 6.8%, f16 scales/biases) — a 0.6B-specific share (embedding = 26% of the parameters). On the SUC codec tensors alone (indices + row scales + residual, codebooks excluded) the saving is **2.2%** (229.00 → 223.89 MB): 1.6 MB from the additive codes, 3.4 MB from Rice-coded residual positions, 0.2 MB from residual values. For the 8B R1 configuration (every linear additive, ≈ 4.28 GB at 4.18 bpp) that predicts ≈ 4.09 bpp, ≈ 96 MB saved (extrapolated, not measured).

**Decode cost** (laptop M3, single thread, MB/s of fixed-width output): rANS (numpy reference) 84 MB/s (encode 4.2 s, decode 3.6 s for 378 M symbols); adaptive context rANS 9 MB/s; Rice positions 47 M positions/s; zstd -3/-19 on the indices 1.3–1.8 GB/s (CLI wall time); zlib 0.6 GB/s; xz 37 MB/s. Load-time decode of a 4.28 GB 8B file: ≈ 51 s with the numpy rANS, ≈ 3.4 s at zstd -3 speed (a compiled rANS was not measured). Saving ~96 MB of reads does not pay back that decode time at SSD speeds, and decoding inside the token loop would cost the whole decode time per token (vs 21–29 ms per token for the whole 8B decode step at 35–47 tok/s on the Studio, K6).

**Conclusion (spec §15, Failure C):** entropy coding stays **storage/distribution-only**. The execution layout keeps fixed-width packed indices (order-0 entropy 99% of log₂K, no usable context). Worth adopting if a compressed distribution format is wanted: Rice for residual positions (−57% of position bytes, decodes at 47 M/s) and byte-shuffle + a general-purpose coder for scales, codebooks and the directory; rANS over additive codes buys 0.7% and is not worth its complexity. SUC1 v1 has no entropy-table section; any of this would be a separate distribution container decoded once at install/load time, never mmapped into the hot path. Numbers are from the dev model on the laptop (CPU, shared with another job); compression ratios are hardware-independent, throughputs are indicative.

## Q5x — shared codebooks per expert stack on Qwen3-30B-A3B (Mac Studio) — reported result

Result: `*_q5x_expert_stacks_qwen3_30b_a3b.json` (+ `.md`). Embedding/LM head left BF16 so only the expert/attention coding differs (hence model bpp above the codec rate); full WikiText-2 test; KL on 16 held-out validation sequences. BF16 PPL 8.697.

| Codec (+ 0.5% activation residual) | Codebooks | Model bpp | PPL | KL | Encode time |
|---|---|---:|---:|---:|---:|
| RTN 4-bit | — | 4.740 | 8.836 | 0.0571 | 536 s |
| additive M=3 | per expert | 3.485 | **9.547** | **0.1287** | 16,342 s |
| additive M=3 | **shared per stack** | 3.426 | 9.723 | 0.1341 (+4%) | **648 s (25× faster)** |
| additive M=4 | shared per stack | 4.405 | 9.004 | **0.0564** | 814 s |
| VQ G=4 | shared per stack | 2.446 | 12.239 | 0.3495 | 364 s |

Findings: sharing codebooks across the 128 experts of a stack costs +4% KL (+1.8% PPL) at 0.06 bpp less, and makes MoE encoding 25× faster — the only way the ~180B MoE targets are practical. Additive M=4 with shared stacks beats RTN 4-bit in KL at lower bits. Allocation over stack units (Q7 30B, resumed from saved candidates with streamed validation) is queued.

## R5 — rotation kernel + per-device tuning, Qwen3-8B decode (Mac Studio) — reported result

Result: `*_e2e_r5_after_qwen3_8b.json` (one-sided rotation, column residuals, tuned rotation layout / fusion / MMA crossover from the Studio's tuning cache). Decode tok/s, B = 1 / 4 / 16:

| Variant | K6 (fusion off) | **R5** | MLX 4-bit |
|---|---|---|---|
| VQ G=4 (2 bpw) | 47.3 / 126.2 / 252.9 | **65.0 / 139.2 / 262.3** | 74.4 / 137.6 / 193.0 |
| VQ G=4 + columns res | 45.4 / 103.2 / 224.6 | **55.7 / 110.0 / 233.3** | |
| additive M=3 + columns res | 35.0 / 91.9 / 147.9 | **42.8 / 91.9 / 146.9** | |

The rotation cost for VQ fell from 6.8 ms/step to ~0 (within profile noise): batch-1 decode is now 87% of MLX 4-bit at less than half the bits, on par at B = 4 and 36% faster at B = 16.

## Q7 — allocation over expert stacks on Qwen3-30B-A3B (Mac Studio) — reported result

Result: `*_q7_allocate_qwen3_30b_a3b.json` (+ `.md`), resumed from the saved candidates artifact, streamed validation. Candidates as in the 8B Q7 with shared expert-stack codebooks (`_xs`); embedding/head affine 4-bit. BF16 PPL 8.697.

| Plan | Model bpp | PPL | KL |
|---|---:|---:|---:|
| greedy 2.5 | 2.55 | 10.536 | 0.1772 |
| greedy 3.0 | 3.04 | **9.871** | **0.1289** |
| uniform M=3 + res | 3.19 | 9.961 | 0.1610 |
| greedy 3.5 | 3.53 | 9.584 | 0.0975 |
| greedy 4.0 | 4.02 | 9.248 | **0.0783** |
| uniform M=4 + res | 4.17 | 9.235 | 0.0832 |
| RTN 4-bit | 4.51 | 9.071 | 0.0812 |

Findings: on the MoE the Fisher-guided allocation pays off clearly, unlike on the dense 8B: at 3.0 bpp greedy has 20% lower KL than uniform M=3 at 3.2 bpp, and at 4.0 bpp it beats RTN 4-bit (4.5 bpp) in KL with half a bit less (PPL slightly higher: 9.25 vs 9.07). This matches the Phase 2 sensitivity spread (single experts ~100× less sensitive than attention).

## C++ runtime on the Qwen3-8B SUC1 file (Mac Studio) — reported result

Result: `*_cpp_generate_check*.json`, `*_cpp_generate_8b_{fused,separate}.json`. File: `compress_qwen3_8b` (4.23 bpp: additive M=4 + 0.5% CSR residual, two-sided rotation — the configuration before R1/Q5b). PPL (8 × 512 WikiText-2 windows): C++ 14.437 vs Python runtimes 14.440 / 14.443 (mean |ΔNLL| 0.031, bf16 logit rounding). Decode B = 1: **C++ 24.8 ms/token** (separate projections, pipelined) / 26.6 ms (fused) vs **Python runtime 30.5 ms** on the same file; batch prefill 4.3 ms/token (Python 3.5). Resident: 4.33 GB mapped file + 166 MB execution arrays + 12 MB KV cache. Fused q|k|v / gate|up launches are slower than separate ones on the M1 Ultra here (to tune per device). Next: regenerate the file with the current best configuration (one-sided, column residuals) and measure again.

## Published baselines (AQLM/QuIP#) — literature comparison, no new measurement

Details, all tables and full citations: [comparison_published.md](comparison_published.md). This compares relative ΔPPL vs each paper's own 16-bit baseline. Published bits count only the block linears, with the embedding and LM head in FP16; SUC bpp is whole-model, so the SUC rows below give linear bpw.

| ≈ bpw | Qwen3-8B, published (same model) | SUC Qwen3-8B (linear bpw, ΔPPL) |
|---|---|---|
| 2 | QuIP# +27.4%, AQLM 2×8 +87.9%, QTIP +6.9%, PV-Tuning +9.6% (LC-QAT, arXiv:2606.10531, Table 1) | VQ G=4 + res 2.18: +90.8% (no FT), +69.7% (block FT); greedy 2.5: +32.0% |
| 3 | no AQLM/QuIP# number; AWQ per-channel +53.5% (arXiv:2505.02214, Table 2) | greedy 3.0: +13.2%; M=3 + res 3.18: +14.0% |
| 4 | GPTQ g128 +2.6% (nominal 4 bits), AWQ g128 +16.4% (arXiv:2505.02214, Table 4) | M=4 + res 4.18: +3.3–3.7%; MLX affine 4-bit 4.5: +3.8% |

Conclusion:

1. At 2 bpw SUC is clearly behind QuIP#/QTIP/PV-Tuning on the same model. This is a negative result.
2. At 4 bpw SUC is slightly behind the published GPTQ g128 and ahead of the per-channel scalar methods.
3. At 3 bpw no same-model AQLM/QuIP# reference exists, and the cross-model rows are indicative only.
4. A local QuIP#/QTIP run on Qwen3-8B with our protocol is still required by D3 for any ≤ 3 bpw claim.

## Mandatory ablations and Pareto frontier on Qwen3-8B (Mac Studio) — reported result

Results: `*_abl_{base,residual,codebook,finetune,repeat}_qwen3_8b.json`, `*_pareto_allocate_qwen3_8b.json`, `*_e2e_abl_*_qwen3_8b.json`, `*_entropy_coding_qwen3_8b.json` (plan: docs/ablations_plan.md; tables: `benchmarks/make_tables.py`). Protocol: `suc quantize`, embedding/LM head affine 4-bit, shared q/k/v and gate/up rotation, full WikiText-2 test, KL on 16 held-out sequences. BF16 PPL 9.717.

**Ablations at 4.22 bpp** (base = additive M=4 K256 G8, one-sided rotation, 0.5% activation column residual):

| Row | PPL | ΔPPL | KL |
|---|---:|---:|---:|
| base | 10.121 | +4.2% | 0.0678 |
| base, seed 2 (repeatability) | 10.081 | +3.7% | 0.0707 |
| − one-sided (two-sided rotation) | 10.113 / 10.290 (seed 2) | +4.1% / +5.9% | 0.0725 / 0.0708 |
| − rotation (none) | 10.455 | +7.6% | 0.0954 |
| − column structure (CSR residual) | 10.185 | +4.8% | 0.0691 |
| − activation selection (magnitude columns) | 10.649 | +9.6% | 0.1403 |
| − residual (none, 4.08 bpp) | 10.632 | +9.4% | 0.1443 |
| CSR residual + block fine-tuning | 10.225 | +5.2% | 0.0629 |
| − multi-codebook: **VQ G=2 K=256 (one codebook)** | **9.994** | **+2.9%** | **0.0494** |
| − vector quantization: scalar K=16 (G=1) | 10.074 | +3.7% | 0.0554 |

Seed-to-seed noise is ≈ ±3% in KL and ±1% in PPL, so: rotation (−29% KL), activation-selected residuals (−52% KL vs magnitude, −53% vs none) are clearly required components; one- vs two-sided rotation and CSR vs columns are within noise in quality (their value is runtime speed); fine-tuning gives −9% KL but slightly worse PPL. **The surprise: at 4 bpw a single low-dimensional codebook (G=2, K=256, 1 KB) beats the additive M=4 G=8 codebooks (KL −27%, well beyond noise), and even a rotated per-tensor scalar codebook does.** The additive multi-codebook is therefore not the right choice at 4 bpw; low-dimensional VQ at 2–3 bpw is being measured (Q5c).

**Pareto frontier (greedy Fisher allocation, head allocated too):**

| Plan | Model bpp | PPL | KL |
|---|---:|---:|---:|
| greedy 1.05 | 1.05 | 42,922 | 8.47 |
| greedy 1.5 | 1.50 | 915.7 | 4.43 |
| greedy 1.75 | 1.75 | 145.2 | 2.60 |
| greedy 2.0 | 2.00 | 39.53 | 1.405 |
| greedy 2.25 | 2.25 | 28.45 | 1.071 |
| greedy 2.5 | 2.50 | 18.43 | 0.725 |
| **uniform VQ G=4 + res** | 2.53 | **14.15** | **0.429** |
| greedy 3.0 | 3.00 | 11.71 | 0.244 |
| uniform M=3 + res | 3.38 | 10.65 | 0.150 |
| greedy 3.38 | 3.38 | 10.78 | 0.141 |
| greedy 3.5 | 3.50 | 10.65 | 0.115 |
| greedy 4.0 | 4.00 | 10.26 | 0.078 |
| greedy 4.22 | 4.22 | 10.27 | 0.073 |
| uniform M=4 + res | 4.22 | 10.19 | 0.069 |

Negative result: **below ~3 bpp the Fisher-greedy allocation is worse than a uniform codec** (2.5 bpp: PPL 18.4 vs 14.2 for uniform VQ G=4) — the additive ½ΣFΔW² proxy breaks down when many tensors are heavily degraded (errors interact), and the greedy mixes in very low-bit candidates. ≤ 1.75 bpp is broken with every candidate set (no "1 bpw works" claim). At equal bits (3.38 / 4.22 bpp) greedy vs uniform is within noise on the dense model (on the MoE greedy wins clearly, Q7 30B).

**Fused kernel ablation** (decode tok/s B=1 / B=16, additive M=4 + columns residual): all fusions 34.6 / 141.4; no fusion 33.9 / 126.9; no fused staging **20.5** / 109.6; residual unfused 33.8 / 151.8; two-sided 34.0 / 136.3 (MLX 4-bit 74.9 / 194.4). Staged codebooks are the one fusion that matters at B=1; additive M=4 is far slower than VQ G=4 (65.0 tok/s, R5) — another reason for low-dimensional VQ.

**Entropy coding on the 8B file** (`20261002T142930Z_entropy_coding_qwen3_8b.json`, `--fast`, file `compress_qwen3_8b`): 4.226 → 4.043 bpp (−4.3%) with the best coder per section — storage-only, as on the dev model.

## Q5c — low-dimensional single-codebook VQ across the bit range on Qwen3-8B (Mac Studio) — reported result

Result: `*_q5c_lowdim_vq_qwen3_8b.json` (+ `.md`); same protocol as the ablations (one-sided rotation, 0.5% activation column residual, head affine 4-bit, full WikiText-2 test, KL on 16 held-out sequences). Together with the ablation row VQ G=2 K=256:

| Model bpp | Codec | PPL | KL | vs best alternative at equal bits |
|---:|---|---:|---:|---|
| 4.22 | **VQ G=2 K=256** | **9.994** | **0.0494** | additive M=4: 10.121 / 0.0678 |
| 3.37 | **VQ G=2 K=64** | **10.618** | **0.1237** | additive M=3: 10.692 / 0.1526; greedy 3.38: 10.78 / 0.141 |
| 2.95 | **VQ G=2 K=32** | **11.560** | **0.2335** | greedy 3.0 (at 3.00): 11.71 / 0.244 |
| 2.53 | VQ G=2 K=16 | 15.616 | 0.5240 | — |
| 2.53 | **VQ G=4 K=256** | **14.432** | **0.4304** | (better than G=2 K=16 at equal bits) |

Findings: from ≈ 3 to 4.2 bpp a single 2-dimensional codebook beats the additive multi-codebooks and the Fisher-greedy mixtures at equal bits (KL −19% to −27%); at ≈ 2.5 bpp the 4-dimensional codebook wins (G=2 K=16 is too coarse). The best SUC operating points on Qwen3-8B are therefore **VQ G=2 K=256 (4.22 bpp, PPL +2.9%)**, **G=2 K=64 (3.37 bpp, +9.3%)**, **G=2 K=32 (2.95 bpp, +19%)** and **G=4 K=256 (2.53 bpp, +48.5%)** — all single-codebook, i.e. also the fastest kernels (one lookup per group; R5: VQ G=4 at 87% of MLX 4-bit at B = 1). G=3 is not usable on Qwen3 (hidden sizes not divisible by 3).

## MoE routing check on Qwen3-30B-A3B (Mac Studio) — reported result

Result: `*_moe_routing_check*.json` (`benchmarks/moe_routing_check.py`; 256 tokens; reference = reconstructed BF16 model; each comparison free-running and teacher-forced; traced forward verified bit-exact against `model(tokens)`).

| Compared with the reconstructed BF16 model | final KL | top-1 | expert-set agreement (mean over 48 layers, free-running) | teacher-forced: set agreement / local rel. error |
|---|---:|---:|---:|---|
| SUC compressed (codes) | 0.036 | 0.922 | 0.719 | 0.974 / 1.9e-3 |
| **control: same decodes, stochastic BF16 rounding, no SUC code** | **0.042** | 0.914 | 0.712 | 0.974 / 2.0e-3 |
| control: same model twice | 0 | 1.000 | 1.000 | 1.000 / 0 |

Finding: the 0.03–0.04 KL between the compressed and the reconstructed MoE is **not a SUC defect**. A perturbation of the same size without any SUC code (stochastic BF16 rounding) produces the same expert-set flips and slightly more KL. Flipped tokens are those whose router margin at the top-k boundary is below the perturbation (layer 0: median margin 0.016 vs 0.063 for unflipped tokens; 5–12% exact BF16 ties at the boundary), and flips accumulate across the 48 layers. The reconstructed BF16 model is itself one BF16 approximation of the exact computation, not ground truth.

## Final Qwen3-8B SUC1 file and C++ runtime — reported result (headline)

> **Superseded (speed):** the C++ vs MLX decode comparison in this section timed MLX with synchronous decode and C++ pipelined; see "Matched-harness speed (suc perf)" at the end of this file. Quality and size figures stand.

Results: `20261002T191301Z_compress_qwen3_8b_final.json` (`suc compress experiments/configs/compress_qwen3_8b_final.yaml`, every stage layer-streamed, 1361 s total) and the `cpp_check_8b_final` result (`*_cpp_generate_check*.json` on `compress_qwen3_8b_final/container/model.suc1`), Mac Studio M1 Ultra.

Configuration: VQ G=2 K=256, one-sided randomized Hadamard rotation, 0.5% activation-selected column residuals (int8), embedding/LM head affine 4-bit g64.

| | SUC final (C++ runtime) | MLX 4-bit (affine g64) | BF16 |
|---|---:|---:|---:|
| bits per parameter (whole model, all overhead) | **4.22** | 4.50 | 16 |
| file / resident weights | **4.32 GB** (4122 MiB file) | 4.61 GB (4.29 GiB) | 16.38 GB (15.26 GiB) |
| WikiText-2 test PPL (full, seq 2048) | **9.994** (+2.9%) | 10.088 (+3.8%) | 9.717 |
| KL vs BF16 (16 held-out sequences) | **0.049** | 0.092 | 0 |
| decode B=1, ms/token | **13.3** (75 tok/s) | 13.4 (74.9 tok/s, e2e harness) | 29.4 (34.0 tok/s) |

The SUC1 file was verified bit-exact (16 sampled encodings + raw tensors, reference decoder max rel. error 0); C++ and Python runtimes agree on PPL (14.252 vs 14.250 on 8 × 512 windows). The Python runtime runs the same file at 19.9 ms/token. MLX 4-bit / BF16 decode figures are from the e2e harness on the same machine (`*_e2e_abl_default_qwen3_8b.json`), not from the C++ tool; a same-harness MLX comparison inside the C++ check is not available.

## Final Qwen3-0.6B SUC1 files on an Apple M3 (16 GB) — reported result

> **Superseded (speed):** the C++ vs MLX decode comparison in this section timed MLX with synchronous decode and C++ pipelined; see "Matched-harness speed (suc perf)" at the end of this file. Quality and size figures stand.

"Small Mac" data point: the dev model compressed with the final recipe and run by the C++ runtime on the **dev laptop** (Apple M3, 16 GB unified memory, on AC power). Results: `20261002T202145Z_compress_qwen3_0.6b_final.json` and `20261002T203532Z_compress_qwen3_0.6b_final_3bpp.json` (`suc compress experiments/configs/compress_qwen3_0.6b_final{,_3bpp}.yaml`, run dirs on the local disk, 885 s / 818 s), RTN baseline `20261002T204839Z_final_baseline_qwen3_0.6b.json` (`suc quantize experiments/configs/final_baseline_qwen3_0.6b.yaml`), C++ checks `20261002T205833Z_cpp_check_qwen3_0.6b_final_m3.json` / `20261002T210018Z_cpp_check_qwen3_0.6b_final_3bpp_m3.json` (`benchmarks/cpp_generate_check.py --ppl-windows 8`), MLX baseline `20261002T210018Z_e2e_final_baseline_qwen3_0.6b_m3.json` (`benchmarks/e2e_generate.py --variants bf16,mlx_q4 --batches 1,4,16`).

Configuration: as `compress_qwen3_8b_final.yaml` — one-sided randomized Hadamard rotation, 0.5% activation-selected int8 column residuals, VQ G=2 K=256 (4-bpp file) or K=64 (3-bpp variant); the **tied** embedding / LM head is affine 4-bit g64 (it is 26% of the parameters of this model, so its 4.5 bpp lifts the whole-model bpp more than on 8B).

| | SUC VQ G=2 K=256 (C++) | SUC VQ G=2 K=64 (C++) | MLX 4-bit (affine g64) | RTN affine 3-bit | BF16 |
|---|---:|---:|---:|---:|---:|
| bits per parameter (whole model, all overhead) | **4.25** (file 4.26) | **3.51** (file 3.52) | 4.50 | 3.50 | 16 |
| file / resident weights | **302.5 MiB** (317.2 MB) | **249.9 MiB** (262.0 MB) | 319.8 MiB | 249 MiB | 1136.9 MiB |
| WikiText-2 test PPL (full, seq 2048) | **22.832** (+8.9%) | **27.806** (+32.6%) | 24.295 (+15.8%) | 62.090 (+196%) | 20.973 |
| KL vs BF16 (16 held-out sequences) / top-1 | **0.092** / 0.853 | **0.308** / 0.751 | 0.190 / 0.795 | 1.087 / 0.546 | 0 |
| decode B=1, ms/token (tok/s) | **6.95** (144) | **7.07** (141) | 6.72 (149) | — | 16.88 (59) |
| process memory during decode | 302.5 MiB mapped file + 97 MiB footprint (phys_footprint) (7 MiB arena, 6 MiB execution arrays, 9 MiB KV) | 249.9 MiB + 95 MiB | peak 551 MiB (MLX, B=1) | — | peak 1348 MiB |

Same session, sequential runs, laptop otherwise idle (no other benchmark, compression or Python job; background: desktop apps, endpoint antivirus — recorded in the e2e JSON `other_processes`). On that basis the M3 timings are valid for this machine; they are not Mac Studio numbers. PPL/KL of MLX 4-bit and 3-bit are the RTN affine g64 rows of `final_baseline_qwen3_0.6b` (every linear and the tied embedding, same protocol as `suc compress`); MLX timings are from the e2e harness (`apply_representation("mlx_affine", bits=4)`, 128-token prompt, 32 decode tokens), C++ timings from `suc1-generate` (default fused plan, pipelined decode, 13-token prompt, 64 decode tokens, median of runs) — two different harnesses on the same machine and session. MLX 4-bit decode at B=4 / 16: 9.20 / 24.41 ms/step (BF16 18.17 / 34.62); the C++ tool decodes batch 1 only.

Correctness: C++ and the Python runtime are bit-identical on all three fixed prompts for both files (hidden states and logits, 64/64 greedy tokens, separate and fused plans); PPL on 8 × 512 WikiText-2 windows C++ 32.465 vs Python 32.479 (4 bpp) and 39.764 vs 39.738 (3 bpp), mean |ΔNLL| 0.038 (bf16 batch-prefill vs per-call rounding). The 16 sampled encodings of each file re-read bit-exactly. Python runtime on the same files: 12.6 / 11.2 ms/token; C++ variants (wall): fused sync 7.59 / 7.82, separate pipelined 6.94 / 6.92 ms; batch prefill (256 tokens) 0.81 / 0.92 ms/token.

Findings:

1. **Quality**: at 0.25 bpp less than MLX 4-bit, the 4-bpp SUC file halves KL (0.092 vs 0.190) and cuts the PPL increase from +15.8% to +8.9%. The 3.5-bpp file is far better than RTN 3-bit at equal size (KL 0.31 vs 1.09, PPL 27.8 vs 62.1) and only slightly worse than RTN 4-bit in PPL, with 22% fewer bits. The small model remains much less compression-tolerant than 8B (4.2 bpp: +8.9% vs +2.9% PPL).
2. **Speed**: on the M3 the C++ runtime decodes the 4-bpp file at 6.95 ms/token, within 3.4% of MLX 4-bit (6.72, different harness) and 2.4× faster than BF16. The 3-bpp file is not faster (7.07): the 0.6B is launch-bound on the M3, so fewer index bits save memory, not time (no "faster because smaller" claim).
3. **Memory**: the whole compressed model is the mapped file (302.5 / 249.9 MiB) plus ~96 MiB of process footprint; ≈ 0.4 GB in total vs 1.35 GB peak for BF16 under MLX.

## Low-G VQ decode speed (Python runtime) and same-session baselines, Qwen3-8B (Mac Studio) — reported result

> **Superseded (C++ vs MLX only):** the Python-runtime rows are matched (synchronous decode on both sides), but the "C++ on par with MLX 4-bit" sentence compares pipelined C++ with synchronous MLX; see "Matched-harness speed (suc perf)" at the end of this file.

Results: `20261002T212251Z_e2e_lowg_qwen3_8b.json`, `20261002T215816Z_e2e_baselines_qwen3_8b.json` (`benchmarks/e2e_generate.py`, Python/MLX runtime, timing codes, column residuals). Decode tok/s B = 1 / 4 / 16 (ms/token at B=1):

| Variant | bpw (linears) | resident MiB | tok/s |
|---|---:|---:|---|
| BF16 | 16 | 15,623 | 33.0 / 52.2 / 190.2 (30.3 ms) |
| MLX 4-bit (same session) | 4.5 | 4,394 | 75.2 / 137.8 / 193.3 (13.3 ms) |
| VQ G=4 K=256 | 2 | 2,331 | 71.2 / 119.9 / 261.7 (14.0 ms) |
| VQ G=2 K=16 | 2 | 2,331 | 72.6 / 123.8 / 233.9 (13.8 ms) |
| VQ G=2 K=64 | 3 | 3,159 | 60.0 / 105.8 / 114.2 (16.7 ms) |
| VQ G=2 K=256 | 4 | 3,987 | 64.9 / 126.2 / 186.4 (15.4 ms) |
| VQ G=2 K=256 + columns res | 4.14 | 4,126 | 55.2 / 113.1 / 165.5 (18.1 ms) |

Findings: in the Python runtime the final configuration (G=2 K=256 + residual) decodes at 73% of MLX 4-bit at B=1; the C++ runtime runs the final file at 13.3 ms/token, on par with MLX 4-bit's 13.3 ms measured in the same session (section "Final Qwen3-8B SUC1 file and C++ runtime"). 6-bit indices (K=64) are slower than 8-bit ones at B=16 (index unpacking), a remaining kernel gap.

## QA side-by-side, Qwen3-8B final file vs BF16 — reported result

`experiments/qa/qa_8b_side_by_side.md` (20 prompts: Italian/English explanations and arithmetic with thinking; greedy, Qwen3 chat template; SUC = final SUC1 via the C++ runtime on the Apple M3 laptop at ≈ 7 tok/s, BF16 = mlx-lm on the Mac Studio). All completed arithmetic answers are correct and equal to BF16 (200 km, 29, 36, 61); explanations are factually equivalent and often nearly word-identical; three SUC answers were cut by the 1,500-token limit with the reasoning already at the correct value (one prompt also cut for BF16). No incoherent or hallucinated output observed. Qualitative, small sample.

## Final Qwen3-30B-A3B SUC1 files and C++ runtime (Mac Studio) — reported result (headline)

> **Superseded (speed):** the C++ vs MLX decode comparison in this section timed MLX with synchronous decode and C++ pipelined; see "Matched-harness speed (suc perf)" at the end of this file. The "22% faster" claim below is withdrawn (matched harness: 6% faster decode, 2–4× slower prefill). Quality and size figures stand.

Results: `20261002T204858Z_compress_qwen3_30b_a3b_final_greedy.json`, `20261002T212215Z_compress_qwen3_30b_a3b_final.json` (`suc compress`, all stages layer-streamed), `20261002T230743Z_cpp_check_qwen3_30b_a3b_final_greedy.json`, `20261002T232117Z_cpp_check_qwen3_30b_a3b_final.json`, same-session baselines `20261002T232936Z_e2e_baselines_qwen3_30b_a3b.json`; RTN 4-bit quality from the Q7 30B run (same protocol). BF16 PPL 8.697.

| | SUC uniform (VQ G=2 K=256 `_xs` + columns) | SUC greedy 4.0 (expert stacks) | MLX 4-bit (affine g64) | BF16 |
|---|---:|---:|---:|---:|
| bits per parameter (whole model) | 4.16 (file 4.17) | **4.02** (file 4.03) | 4.50 | 16 |
| file / resident weights | 15.89 GB | **15.32 GB** | 17.18 GB (16,385 MiB) | 61.06 GB |
| WikiText-2 test PPL | 9.141 (+5.1%) | 9.105 (+4.7%) | 9.071 (+4.3%, RTN 4-bit, Q7) | 8.697 |
| KL vs BF16 | 0.0644 | **0.0640** | 0.0812 (RTN 4-bit, Q7) | 0 |
| decode B=1, ms/token (tok/s) | **12.9 (77.7)** C++ | 16.5 (60.5) C++ | 15.75 (63.5), same session | 21.2 (47.3), same session |
| Python runtime, ms/token | 24.1 | 23.8 | — | — |

`suc compress` wall time: 1994 s (uniform), 5443 s (greedy, incl. streamed Fisher + candidate scoring). C++ and Python runtimes agree on PPL (8 × 512 windows: 12.589 / 12.546 uniform; 12.658 / 12.640 greedy); process footprint ≈ 1 GB on top of the mapped file. **The uniform file decodes 22% faster than MLX 4-bit at batch 1** with 21% lower KL and 7.5% fewer bytes; the greedy file has the best quality per bit but mixes in BF16 attention units (slower). PPL is slightly above RTN 4-bit (+0.4–0.8%) while KL is lower. QA side-by-side (20 prompts, `experiments/qa/qa_30b_side_by_side.md`): all completed arithmetic answers correct and equal to BF16; two SUC answers cut by the 1,500-token limit (reasoning correct up to the cut), one BF16 answer also cut; explanations factually equivalent, often nearly word-identical.

## Evaluation / benchmark tools `suc eval` and `suc perf`

Dedicated tools for the final SUC1 files, modelled on ds4-eval / ds4-bench (ds4, MIT): `suc eval` grades a test suite
(`experiments/eval/suite_core.json`; the hard ds4-derived `ds4_subset.json`) through the C++ and Python runtimes, with an
optional per-case comparison against a BF16 / MLX reference, regrade and drift gate; `suc perf` measures prefill, TTFT,
decode and memory of SUC (C++ / Python), MLX 4-bit and BF16 at several context lengths and batch sizes in one session
with warmup, repeats and statistics. Protocols, outputs and the exact Studio commands: [tools.md](tools.md). `suc perf`
results: next section; `suc eval` has no reported result yet (configs `experiments/configs/eval_*.yaml`).

## Matched-harness speed (suc perf) — supersedes earlier C++ vs MLX comparisons

Results: `20261003T015734Z_perf_qwen3_8b_final.json` / `.md` (git 02eb3be) and
`20261003T031427Z_perf_qwen3_30b_a3b_final.json` / `.md` (git b253a00), `suc perf experiments/configs/perf_qwen3_8b_final.yaml`
and `perf_qwen3_30b_a3b_final.yaml`, Mac Studio M1 Ultra, clean working tree. Protocol ([tools.md](tools.md)): one tool,
one session per model, systems run sequentially; same raw prompt (`experiments/perf/promessi_sposi_head.txt`), full
prefill of each context from an empty cache, then 128 greedy tokens; warmup 1, 3 repeats, median; decode ms/step =
median of steps 2…n. **MLX decode is pipelined** (as mlx-lm), like the C++ decode. The C++ runtime decodes batch 1 only.

**Why this supersedes the earlier figures.** The earlier headline comparisons (sections "Final Qwen3-8B SUC1 file…",
"Final Qwen3-30B-A3B SUC1 files…", "Final Qwen3-0.6B SUC1 files…") timed MLX 4-bit and BF16 in the e2e harness, which
evaluates each token synchronously, and SUC with the pipelined C++ decode. Synchronous decode is slower than mlx-lm's
(MLX 4-bit, 8B: 13.3 → 10.44 ms/token; 30B-A3B: 15.75 → 13.38 ms/token), so SUC was overstated. Withdrawn: "parity" on
Qwen3-8B, "22% faster" on Qwen3-30B-A3B, "within 3.4%" on Qwen3-0.6B. Likewise the C++-vs-Python gap of the cpp checks
(13.3 vs 19.9 ms on the 8B) came from synchronous Python decode.

**Qwen3-8B, final file (4.22 bpp), batch 1** — decode ms/step (tok/s), ratio of tok/s vs MLX 4-bit:

| ctx | MLX 4-bit | SUC C++ (fused, default) | SUC C++ separate | SUC Python (pipelined) | BF16 |
|---:|---:|---:|---:|---:|---:|
| 128 | 10.44 (95.8) | 13.89 (72.0) 0.75× | 13.40 (74.6) 0.78× | 13.41 (74.5) 0.78× | 27.75 (36.0) 0.38× |
| 512 | 10.81 (92.5) | 14.23 (70.3) 0.76× | 13.73 (72.8) 0.79× | 13.76 (72.6) 0.79× | 28.10 (35.6) 0.38× |
| 2048 | 11.49 (87.0) | 14.87 (67.3) 0.77× | 14.46 (69.2) 0.79× | 14.52 (68.9) 0.79× | 28.72 (34.8) 0.40× |
| 8192 | 15.66 (63.8) | 16.83 (59.4) 0.93× | 16.46 (60.8) 0.95× | 18.83 (53.1) 0.83× | 32.06 (31.2) 0.49× |

Prefill tok/s, batch 1 (ratio vs MLX 4-bit):

| ctx | MLX 4-bit | SUC C++ | SUC C++ separate | SUC Python | BF16 |
|---:|---:|---:|---:|---:|---:|
| 128 | 588 | 340 (0.58×) | 383 (0.65×) | 327 (0.56×) | 479 (0.82×) |
| 512 | 754 | 334 (0.44×) | 376 (0.50×) | 375 (0.50×) | 806 (1.07×) |
| 2048 | 771 | 306 (0.40×) | 341 (0.44×) | 392 (0.51×) | 939 (1.22×) |
| 8192 | 676 | 232 (0.34×) | 252 (0.37×) | 370 (0.55×) | 815 (1.21×) |

Python runtime at B = 4 / 16 (decode tok/s ratio vs MLX 4-bit): 0.90× / 0.97× (ctx 128), 0.90× / 0.98× (2048),
0.92× / 0.99× (8192); prefill ≈ 0.5×. Memory at ctx 128, B = 1: C++ 4122 MiB mapped file + 200 MiB process; MLX 4-bit
peak 4695 MiB; Python runtime peak 6293 MiB.

**Qwen3-30B-A3B, final files, batch 1** — decode ms/step (tok/s), ratio vs MLX 4-bit:

| ctx | MLX 4-bit | SUC C++ uniform (4.16 bpp) | SUC C++ greedy (4.02 bpp) | SUC Python greedy | BF16 |
|---:|---:|---:|---:|---:|---:|
| 128 | 13.38 (74.8) | **12.60 (79.4) 1.06×** | 17.11 (58.5) 0.78× | 20.75 (48.2) 0.64× | 19.40 (51.6) 0.69× |
| 512 | 13.85 (72.2) | 13.69 (73.1) 1.01× | 17.12 (58.4) 0.81× | 21.21 (47.1) 0.65× | 19.78 (50.5) 0.70× |
| 2048 | 14.80 (67.6) | **14.14 (70.7) 1.05×** | 18.25 (54.8) 0.81× | 22.14 (45.2) 0.67× | 20.73 (48.2) 0.71× |
| 8192 | 16.63 (60.1) | 19.07 (52.4) 0.87× | 22.34 (44.8) 0.74× | 24.23 (41.3) 0.69× | 23.35 (42.8) 0.71× |

Prefill tok/s, batch 1 (ratio vs MLX 4-bit):

| ctx | MLX 4-bit | SUC C++ uniform | SUC C++ greedy | SUC Python greedy | BF16 |
|---:|---:|---:|---:|---:|---:|
| 128 | 753 | 376 (0.50×) | 201 (0.27×) | 375 (0.50×) | 799 (1.06×) |
| 512 | 1179 | 356 (0.30×) | 200 (0.17×) | 429 (0.36×) | 1450 (1.23×) |
| 2048 | 1341 | 325 (0.24×) | 183 (0.14×) | 645 (0.48×) | 1838 (1.37×) |
| 8192 | 1083 | 225 (0.21×) | 149 (0.14×) | 562 (0.52×) | 1250 (1.15×) |

Memory at ctx 128, B = 1: C++ uniform 15,194 MiB mapped + 1065 MiB process, greedy 14,650 + 990 MiB; MLX 4-bit peak
16,602 MiB; BF16 58,357 MiB.

**Qwen3-0.6B, final file (4.25 bpp), Apple M3 laptop, `suc perf` smoke (indicative, [tools.md](tools.md)):** decode
B = 1: MLX 4-bit 4.96 ms/step, SUC C++ 7.30 ms (0.68×), BF16 15.23 ms.

Findings:

1. **Decode is competitive only on the MoE at batch 1.** The uniform Qwen3-30B-A3B file decodes 6% faster than MLX
   4-bit at ctx 128 and 4–5% at 512–2048, and 13% slower at 8192. The greedy file is slower at every context.
2. **Dense models are slower**: Qwen3-8B C++ 0.75–0.77× of MLX 4-bit up to ctx 2048 (0.93× at 8192), 0.6B 0.68× on
   the M3. Separate projections are 3–4% faster than the default fused ones on the M1 Ultra.
3. **Prefill is slower everywhere**: 8B 0.34–0.58×, 30B-A3B 0.21–0.50× (uniform) and 0.14–0.27× (greedy); the C++
   prefill (chunked, 64 tokens) is the largest speed gap. The 30B-A3B gap grows with context.
4. With pipelined decode the Python runtime equals C++ on the 8B up to ctx 2048; on the 30B-A3B greedy file C++ is
   faster (17.11 vs 20.75 ms).
5. Repeatability: 3 repeats per row; batch-1 decode CV ≤ 1% except the 30B-A3B greedy C++ row at ctx 128 (2.3%) (the
   30B-A3B Python runtime at B = 16 is unstable: CV 2.8% / 43.9% at ctx 2048 / 8192); one session per
   model; background processes (screen sharing, browser during the 8B run) are recorded in `other_processes`.

## suc eval — graded core suite on the final files (Mac Studio) — reported result

Results: `*_eval_qwen3_8b_final.json` / `*_eval_qwen3_30b_a3b_final.json` (+ `.md`); suite `experiments/eval/suite_core.json` (40 cases, 30 auto-graded: 10 arithmetic with thinking + 20 short items; 10 open explanations ungraded); greedy; no reference model (optional, off).

| Model / file | Runtime | Passed / graded |
|---|---|---:|
| Qwen3-8B final (4.22 bpp) | C++ (fused / separate) | 29 / 30 (96.7%) |
| Qwen3-8B final | Python | 28 / 30 (93.3%) |
| Qwen3-30B-A3B uniform (4.16 bpp) | C++ | **30 / 30 (100%)** |
| Qwen3-30B-A3B greedy (4.02 bpp) | C++ / Python | 28 / 30 / 29 / 30 |

Small suite: a sanity check that the compressed models answer correctly, not a capability benchmark (no BF16 run in this suite; the BF16 side-by-side is in `experiments/qa/`).

## Qwen3.8-Flash-Next (qwen4_exp, 180 B) — first frontier model (2026-10-03, M1 Ultra)

Uniform VQ G=2 K=256 + one-sided RHT + 0.5 % int8 column residual on 123.7 B eligible params (4.155 bpw), `suc compress`
from the frontier queue (11 h 31 m incl. download verify; calibrate 36 min, reference 21 min, encode ~33 s/block).
Results: `experiments/results/qwen38/`, `experiments/results/*_repack_model_ngram4.json`.

| Item | Value |
|---|---|
| SUC1 file, raw BF16 n-gram tables | 157.41 GiB (7.63 bpp overall) |
| SUC1 file after `suc repack` (row-affine 4-bit n-gram tables) | tables 95.37 → 25.03 GiB |
| Quality vs BF16 (wikitext2, queue report) | PPL 4.695 → 5.032 (+7.2 %), KL 0.151 |
| Python runtime (`suc generate`) load | 240–266 s, resident 62.08 GiB (estimate 62.20), load peak 63.31 GiB |
| C++ runtime (`suc1-generate`, 4-bit tables, ctx 4096) | decode median 37.56 ms/token (26.6 tok/s; GPU 29.1 ms + n-gram host 6.3 ms), prefill 39.5 ms/token (token-by-token) |
| C++ memory | GPU-mapped 66.6 GB + execution arrays 2.2 GB + KV 0.12 GB; n-gram tables (26.9 GB) outside GPU views, read row by row |
| torch vs MLX reference (4 blocks, float32) | worst block rel. Frobenius error 1.17e-2 before the fix, **1.89e-6 after** (bf16 run dtype vs torch fp32: 4.7e-2) |
| MLX affine 4-bit g64 baseline (same encoded set + embed/head 4-bit, tables raw) | PPL 5.222, KL 0.202, top-1 0.827 — confirmed by the rerun below (R:`20261004T102809Z_quality_eval_qwen3.8_flash_next`) |

**torch vs MLX gap — cause and fix.** Bisected inside block 0 by feeding every sub-module the torch inputs: the hyper-
connection mixers, projections, router, experts, shared expert, gated RMSNorm and the QSA attention of block 3 all agree
to ≤ 2.6e-6; only the Gated DeltaNet core differed (1.3e-3). The q/k normalization used mlx-lm qwen3_5's
`rms_norm(x, eps=1e-6)` shortcut, which adds eps to the *mean* (= 1e-6·Dk = 1.28e-4 on the sum of squares); HF / FLA
`l2norm` adds 1e-6 to the *sum*. With eps = 1e-6/Dk (Python and C++, `suc/models/mlx_qwen4_exp.py`,
`cpp/src/metal/qwen4.cpp`) block 0 drops 4.5e-4 → 1.45e-6 and the block-3 1.17e-2 disappears (it was propagation, block 3
alone is 6.6e-7 exact). Result: `experiments/results/20261003T150519Z_qwen4_exp_reference_check.json`. The compress
need not be redone: the change is ~1e-4 relative on q/k only, 40× below the bf16 noise of the run dtype (1.8e-2 at
block 0), and the codes depend on it only through calibration activations. PPL/KL numbers measured after the fix are
compared with reference logits produced before it (same small effect).

The KL (0.151 in the compress run, 0.143 after the fix) is about 2× the Qwen3-30B-A3B result; no ablation on this
model.

### Qwen3.8-Flash-Next — quality vs MLX 4-bit, `suc eval`, C++ speed (2026-10-04, M1 Ultra) — reported result

WikiText-2 test PPL (145 × 2048 windows, 296,960 tokens scored) and KL vs the BF16 reference logits of the compress run
(16 × 512 validation tokens), measured with `scripts/system_quality.py` after the l2norm-eps fix on the same windows /
sequences. R:`20261004T092508Z` / `100424Z` / `102809Z_quality_eval_qwen3.8_flash_next`. BF16 PPL 4.695 (from the
compress run; BF16 itself, 354 GB, cannot run on the Studio).

| System | File / weights | bpp, encoded set | bpp, whole file | PPL | KL mean | KL p95 (chunk-mean) | Top-1 | ΔCE | Resident (Python) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SUC raw BF16 n-gram tables | 169.01 GB | 4.16 | 7.64 | 5.033 (+7.2 %) | 0.143 | 0.600 | 0.854 | +0.013 | 68.5 GB (tables mmap) |
| SUC ngram4 (row-affine 4-bit tables) | 93.49 GB | 4.16 | 4.23 | 5.055 (+7.7 %) | 0.145 | 0.601 | 0.856 | +0.028 | 68.5 GB |
| MLX affine 4-bit g64 (same encoded set + embed/head 4-bit, tables raw) | 71.79 GB weights + raw tables | 4.57 | — | 5.222 (+11.2 %) | 0.202 | 0.861 | 0.827 | +0.058 | 71.8 GB |

- The MLX 4-bit result of the first table (marked preliminary before) is confirmed by this run (5.222 / 0.202 / 0.827). On the same encoded set
  (4.16 vs 4.57 bpp) SUC ngram4 has 28 % lower KL and 3.2 % lower PPL; single seed. Whole-file bpp = file bytes × 8 /
  176.94 B parameters; the MLX row keeps the n-gram tables raw and outside its weight bytes, so no whole-file bpp.
- 4-bit n-gram tables cost +0.4 % PPL and +1.7 % KL against raw tables (95.4 → 25.0 GiB of tables).
- The compress-run quality (KL 0.151, top-1 0.849) was measured before the l2norm-eps fix; the 0.143 above is the same
  raw file after it.
- Python runtime peak memory 74.8 GB (SUC) vs 78.1 GB (MLX 4-bit) in these runs; load 254–325 s (SUC) vs 42 s (MLX).

**`suc eval` core suite** (40 cases, 30 graded; R:`20261004T111255Z_eval_qwen3.8_flash_next`): MLX 4-bit, SUC Python
raw, SUC Python ngram4 and SUC C++ ngram4 all pass **30 / 30** (QA 10/10, SUC-basic 20/20); 10 open-ended cases are
ungraded. Suite wall time: MLX 286 s, Python raw 388 s, Python ngram4 371 s, C++ ngram4 974 s (wall time of the eval harness,
not a speed measurement).

**C++ decode** (`suc1-generate`, ngram4 file, 19-token prompt, 256 greedy tokens, ctx 4096; R:`qwen38/q4_cpp`): median
37.56 ms/token (26.6 tok/s): GPU 29.13 ms + host n-gram gather 6.29 ms. Python runtime (`suc generate`, 32 tokens):
56.1 ms/token (raw) and 60.5 ms/token (ngram4) (R:`qwen38/gen_raw`, `qwen38/gen_ngram4`). **Not measured:** a
matched-harness `suc perf` of C++ vs MLX 4-bit on this model, so no speed ratio against MLX 4-bit is claimed.

## DeepSeek-V4-Flash (deepseek_v4, 284 B) — second frontier model (2026-10-03 – 10-06, M1 Ultra) — reported result

`deepseek-ai/DeepSeek-V4-Flash-0731`, 284.33 B parameters; the checkpoint stores routed experts in MXFP4 and most other
linears in FP8. **Reference = exact BF16 dequantization of that checkpoint** (not a BF16 training checkpoint); its
WikiText-2 PPL is 4.779 (70 × 4096 windows, 286,720 tokens scored). KL: 16 × 512 validation tokens. Runtime budget for
weights: 111 GB (Studio, OS margin excluded). torch vs MLX reference, float32, 4 blocks × 2560 tokens: rel. Frobenius
3.9e-5 … 2.2e-3 (block 3), R:`20261003T174631Z_deepseek_v4_reference_check_float32`. Analysis of the plans:
[docs/arch/deepseek_v4.md §12](arch/deepseek_v4.md).

**Compressed files** (`suc compress`, greedy allocation over the same calibrate / reference / candidates; the variants
are re-plans of the base run, `plan.allocation` overrides). R:`20261003T233237Z_compress_deepseek_v4_flash`,
`20261004T125710Z_..._native`, `20261006T115604Z_..._native_max`, `20261006T135710Z_..._flat_max`,
`20261006T153009Z_..._native_stream`.

| Plan | Routed experts (33,024 tensors) | File | bpp (file) | PPL (Δ vs ref) | KL mean | KL p95 (chunk-mean) | Top-1 | ΔCE | Fisher-predicted loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | G=4 18,417 · M=3 5,559 · G=2 9,048 | 107.43 GB | 3.023 | 5.430 (+13.6 %) | 0.250 | 1.034 | 0.821 | 0.127 | 1.22e-4 |
| native | G=4 19,362 · M=3 4,422 · native MXFP4 9,240 | 107.42 GB | 3.022 | 5.390 (+12.8 %) | 0.248 | 1.065 | 0.819 | 0.132 | 6.37e-5 |
| native_max (budget 108.75 GB, no G=2) | G=4 18,822 · M=3 4,412 · native 9,790 | 108.67 GB | 3.057 | 5.362 (+12.2 %) | **0.225** | 0.906 | 0.826 | 0.116 | 5.82e-5 |
| flat_max (G=4 / M=3 only, 108.75 GB) | G=4 8,490 · M=3 24,534 | 108.68 GB | 3.058 | 5.504 (+15.2 %) | 0.241 | 0.867 | 0.831 | 0.089 | 3.22e-4 |
| native_stream (all experts native, SSD streaming) | native 33,024 | 155.05 GB | 4.362 | **4.840 (+1.3 %)** | **0.036** | 0.114 | 0.916 | 0.034 | 9.1e-7 |

Attention / shared experts are affine 8-bit (309–344 tensors) with a few BF16 / VQ G=2 tensors in every plan;
embedding / LM head affine 4-bit. Fisher-predicted losses are those of the plan stage (arch §12).

- **The routed experts dominate the KL.** With every expert kept in its native MXFP4 bytes (error 0 vs the reference)
  the KL is 0.036; every ≈ 3-bpp plan is at 0.225–0.250. About 84–86 % of the KL of the 108.75-GB files therefore comes
  from re-quantizing the routed experts below 4 bits; attention 8-bit plus embedding / head 4-bit account for the
  0.036 floor.
- **Native MXFP4 is the right candidate.** VQ G=2 (4.16 bpp, rel. MSE 0.0078 after rotation) is strictly dominated by
  passing the native MXFP4 expert through (4.25 bpp, error 0); the native plan took it 1:1 (−0.7 % KL, −0.7 % PPL at
  the same size). native_max (+1.25 GB, G=2 excluded) is the best file that fits 111 GB resident: KL −10 % vs base.
- **Fisher vs measured (negative result for the proxy).** Across the four ≈ 3-bpp plans the Fisher-predicted loss
  spans 5.5× (5.82e-5 … 3.22e-4) while the measured KL spans 11 % (0.225 … 0.250). flat_max was predicted 5.5× worse
  than native_max and is 7 % worse in KL and 2.6 % worse in PPL, but better in ΔCE (0.089 vs 0.116) and top-1 (0.831 vs
  0.826). The crude proxies of arch §12 (Σ rel. MSE, routing-count × rel. MSE) predicted flat_max *better* than
  native_max; it is not. Neither proxy orders these plans reliably; the additive diagonal-Fisher model is not valid at
  the error size of the G=4 tier (rel. MSE 0.096) with ≈ 384 routed calibration tokens per expert. One seed, one
  evaluation set; differences of a few % are not shown to be repeatable.
- **Baseline:** MLX affine 4-bit would need ≈ 160 GB (284.3 B × 4.5 bits; computed, not run) and does not fit the
  111-GB budget, so no same-protocol 4-bit baseline exists for this model; no local AQLM / QuIP# run either.
  Published community numbers ([comparison_published.md](comparison_published.md)) use other protocols and are not
  comparable.

**C++ speed** (`suc1-generate`, resident file, 12-token prompt, 256 greedy tokens; R:`20261005T174151Z_gen_cpp_deepseek_v4_flash_base`,
`20261005T174315Z_..._native`): base 86.12 ms/token (11.6 tok/s, GPU 83.1 ms), native 81.16 ms/token (12.3 tok/s, GPU
78.6 ms); load 9–12 s; process footprint 9.6 / 7.9 GB besides the 107.4 GB mapped file. The first two runs
(R:`20261005T071238Z`, `20261005T093215Z`) hung until the timeout killed them (rc −15): the file was covered by
half-overlapping GPU views (86.6 + 64.1 GB) whose residency exceeded the 115.3 GB GPU working set, so the first command
buffer was never scheduled. Views are now cut only between mapping units (commit 1be14eb; cpp/README.md).

**Matched harness** (`suc perf`, prompt from `promessi_sposi_head.txt`, 64 greedy tokens, median of 3;
R:`20261006T100712Z_perf_deepseek_v4_flash`):

| System | ctx | Decode ms/token (tok/s) | Prefill tok/s | Memory |
|---|---:|---:|---:|---|
| SUC C++ native | 128 / 2048 / 8192 | 83.19 (12.0) / 87.32 (11.5) / 94.44 (10.6) | 13.0 / 12.7 / 11.9 | 102,441 MiB file + 7,516 MiB |
| SUC C++ base | 128 / 2048 / 8192 | 88.48 (11.3) / 92.33 (10.8) / 99.61 (10.0) | 12.2 / 11.9 / 11.3 | 102,454 MiB file + 9,094 MiB |
| SUC Python native | 128 / 2048 / 8192 | 2139 / 2138 / 2150 (0.5) | 17.3 / 55.1 / 56.3 | 109,305–113,990 MiB peak |

C++ prefill is 12–13 tok/s (8192-token TTFT 687 s): a known gap (no batched codebook matmul in the C++ prefill yet, final report §7).
The Python runtime decodes at 2.1 s/token at 107–111 GiB peak MLX memory, near the machine limit. No MLX 4-bit or BF16 row
(does not fit). The native_max / flat_max / native_stream files were not run in C++ (speed not measured; native_stream
would need expert streaming, 155 GB > 111 GB).

**`suc eval` core suite** (R:`20261006T001744Z_eval_deepseek_v4_flash`, native file): SUC Python and SUC C++ both pass
**30 / 30** (QA 10/10, SUC-basic 20/20); C++ vs Python: answer agreement 100 %, 14 / 40 identical generations,
teacher-forced KL 0.0051, top-1 0.981. No BF16 / MLX 4-bit system (does not fit).

**Long context** (native file, Python runtime; R:`20261004T201726Z_long_context_deepseek_v4_flash`): 2 PG19 test
documents of 65,536 tokens, KL vs the reference (sparse indexer) logits, 12 needle cases (8k–64k, depth 0.1 / 0.5 / 0.9).

| Length | 8,192 | 16,384 | 32,768 | 65,536 |
|---|---:|---:|---:|---:|
| PPL reference / SUC | 12.02 / 13.80 | 11.21 / 12.78 | 10.72 / 12.15 | 10.41 / 11.75 |
| KL SUC vs reference (top-1) | 0.248 (0.779) | 0.231 (0.784) | 0.214 (0.791) | 0.207 (0.792) |
| KL reference with dense indexer vs sparse (top-1) | 0.010 (0.962) | 0.021 (0.939) | 0.038 (0.913) | 0.072 (0.878) |

- Needles: 12 / 12 for SUC and for the reference (answer agreement 100 %).
- The KL of the compressed model does **not** grow with context (0.248 → 0.207 at 8k → 64k), close to the 512-token
  KL of the compress run (0.248).
- Sparse-attention indexer: mean top-k selection overlap SUC vs reference 0.741 (0.867 at positions 0–8k, 0.709 at
  32–64k; chance ≈ 0.11); the needle position is selected in 97.6 % of the layer × case checks (reference 96.5 %).

## GLM-5.3-Flash (glm5_next, 313 B) — third frontier model (2026-10-05 – 10-06, M1 Ultra) — reported result

`zai-org/GLM-5.3-Flash`, 313.33 B parameters (BF16 checkpoint). torch vs MLX reference, float32, 4 blocks: rel.
Frobenius ≤ 1.4e-5 (R:`20261004T201954Z_glm5_next_reference_check_float32`). Tokenizer: 290,549 tokens encoded and
decoded with 0 mismatches vs HF tokenizers, chat template equal to `chat_template.jinja`
(R:`20261005T174021Z_glm5.3_flash_tokenizer_chat_check`; the first check failed because the C++ chat prompt did not
exist yet, R:`20261005T023127Z`).

**Compress** (R:`20261005T022627Z_compress_glm5.3_flash`): greedy allocation, 3.035 bpw allocated over 36,603
tensors (VQ G=4 16,955 · additive M=3 8,153 · VQ G=2 11,456 · affine 8-bit 39; embedding / head affine 4-bit). File
119.93 GB, **3.062 bpp**. All stages layer-streamed; 6.1 h in total (candidates 2.7 h), peak MLX memory 70.4 GiB
(candidates stage).

| | bpp | PPL (70 × 4096) | KL mean | KL p95 (chunk-mean) | Top-1 | ΔCE |
|---|---:|---:|---:|---:|---:|---:|
| BF16 reference | 16 | 3.268 | — | — | — | — |
| SUC | 3.062 | 4.009 (+22.7 %) | 0.261 | 1.063 | 0.815 | +0.128 |

No same-protocol baseline: MLX affine 4-bit would need ≈ 176 GB (313.3 B × 4.5 bits; computed, not run) and BF16 is
627 GB. The file is larger than the 111-GB budget, so the runtime streams experts from SSD: the plan's preflight
accepted it in expert-streaming mode (expert cache 102.45 GB = 89.7 % of the routed experts).

**C++ streamed decode** (`suc1-generate`, 12–13-token prompt, ≈ 256 greedy tokens, expert pool 102.0 GB with a pinned
hot set of 4,048 experts / 51.0 GB chosen from calibration routing counts, SLRU, 8 IO threads, F_NOCACHE reads):

| Run | Commit | Decode ms/token (tok/s) | GPU ms/token | Expert hit rate | Read from SSD |
|---|---|---:|---:|---:|---:|
| Python runtime, 95 GB cache (R:`20261005T115832Z_gen_python_glm5.3_flash_streamed`) | 34d0cc9 | 64,319 (0.016) | — | 90.9 % | 84.7 GB |
| C++ streamed, before the residency set (R:`20261005T175524Z_gen_cpp_glm5.3_flash_streamed`) | 1eaa0d3 | 2,577 (0.39) | 142.7 | 91.3 % | 70.3 GB |
| C++ streamed + prefetch, before (R:`20261005T191021Z`) | 0f97a45 | killed by the timeout | — | — | — |
| **C++ streamed, residency set** (R:`20261006T162521Z_gen_cpp_glm5.3_flash_streamed_resset2`) | a5d1eb8 | **153.4 (6.52)** | 89.8 | 91.6 % | 67.1 GB |
| C++ streamed + prefetch, residency set (R:`20261006T162659Z_..._resset2_prefetch`) | a5d1eb8 | 268.6 (3.72) | 172.4 | 96.3 % | 94.4 GB |

- **Residency-set fix: 16.8× faster decode.** Before it, every per-layer command buffer made its expert pool resident
  again and the driver unwired idle pools near the wired limit: 2.52 s per token of host-side encode time
  against 0.14 s of GPU time. With the pools, file views and large buffers in one `MTLResidencySet` (commit fe4d814)
  the encode overhead disappears (decode 153.4 ms, GPU 89.8 ms). The first runs after fe4d814 (R:`20261006T153424Z`,
  `153425Z`, rc 1) failed because fe4d814 had silently reverted the view-splitting fix 1be14eb; a5d1eb8 restored it.
- **Prefetch is slower (negative result).** Predicting the next layer's experts raised the hit rate from 91.6 % to
  96.3 %, but 2,385 of 7,250 prefetches were wasted, 41 % more bytes were read and decode went from 153 to 269
  ms/token. Prefetch stays off by default.
- Memory: 5.78 GB of non-expert GPU views + 5.89 GB of execution arrays + 102.0 GB expert pool; process footprint
  109.2 GB, resident peak 115.2 GB.

**Matched harness** (`suc perf`, same prompt file as DeepSeek, 64 greedy tokens, median of 3, C++ streamed, residency
set; R:`20261006T182924Z_perf_glm5.3_flash_cpp`): decode **135.45 / 142.27 / 146.51 ms/token (7.4 / 7.0 / 6.8 tok/s)**
at ctx 128 / 2048 / 8192; prefill 5.6 / 7.1 / 6.9 tok/s (TTFT 1,182 s at 8192); memory 5,508 MiB mapped file + 104,121–
104,216 MiB process footprint (expert pool included). No MLX 4-bit / BF16 row (neither fits); Python not timed in
this session (64 s/token above).

**`suc eval` core suite** (C++ streamed only; R:`20261006T200231Z_eval_glm5.3_flash_cpp`): **27 / 30** (SUC-basic 20/20,
QA 7/10), 10 ungraded, 0 errors. The three failures (it_math_1, it_math_2, it_math_5) all contain the correct answer
(391, 200, 61); the model then emits `<|user|>` instead of ending the turn and keeps generating (`think_close:
continued`), and the grader picks a later number. Whether `<|user|>` must be a stop token for GLM-5.3 in the eval
harness is not checked yet; not re-run. No BF16 / MLX 4-bit reference system (does not fit), so no answer agreement
against an uncompressed model.

## SSD read bandwidth on the Mac Studio (2026-10-05) — reported result

R:`20261005T023119Z_ssd_read_bench_studio`: internal SSD of the M1 Ultra, reads over the 107-GB DeepSeek SUC1 file.
Only uncached (`F_NOCACHE`) reads measure the device; cached `pread` partly hits the page cache (file < RAM).

| Read size | pread F_NOCACHE, 1 / 4 / 8 / 16 threads (GB/s) | pread cached, 1 / 16 threads | mmap page faults, 1 / 16 threads |
|---|---|---|---|
| 2.58 MB (Qwen3.8 expert) | 4.98 / 7.26 / 7.31 / 7.30 | 3.87 / 8.23 | 0.92 / 4.37 |
| 9.4 MB (GLM expert) | 6.70 / 9.87 / 9.89 / 9.93 | 5.89 / 13.56 | 1.12 / 4.25 |

Synthetic expert streaming (`suc-expert-bench`, uniform random routing = worst case, no hot set, 32 tokens):
GLM shape (9.4 MB experts, top-8, 42 layers, cache 83 % of experts, hit rate 34 %) 822 / 339 / 263 / 239 ms per token
at 1 / 4 / 8 / 16 IO threads; Qwen3.8 shape (2.58 MB, top-10, 48 layers, cache 25 %, hit rate 16 %) 357 / 125 / 95 /
90 ms per token. Conclusions: device bandwidth saturates at ≈ 7.3 GB/s (2.6-MB reads) and ≈ 9.9 GB/s (9.4-MB reads)
with ≥ 4 threads; `mmap` page-fault reads are 1.7–6× slower than uncached `pread`, hence explicit `pread` into pools. With the real
GLM routing and a hot set the hit rate is 91.6 % instead of 34 %, which is what makes 153 ms/token possible.

## Frontier models — summary (Mac Studio M1 Ultra, 128 GB; 111 GB budget for the model)

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

## Engine 1.4.7 refactoring, 19 October 2026

The six selected local M1 files were compared in three interleaved measured pairs (warmup excluded): [COMBINED_RESULTS.md](optimization/COMBINED_RESULTS.md). Raw records are `*_eff_comparison_combined_<model>_r{1,2,3}.json`; compiled candidate `6693b2c`, baseline `8f6d81c`. Full-reference KL, rejected chunk/half/lookup/n-gram paths and actual shipped CLI/server/app-engine checks are documented in [REFERENCE_KL.md](optimization/REFERENCE_KL.md), [INFERENCE_1.4.7.md](optimization/INFERENCE_1.4.7.md) and [PROGRESS.md](optimization/PROGRESS.md). All twelve 8k/32k comparisons and the real bundled-engine session checks passed; see [final qualification](optimization/FINAL_QUALIFICATION_1.4.7.md). Complete same-reference KL is unchanged on all six models; no new compression or competitor benchmark is implied.
