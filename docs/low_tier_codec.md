# Low tier codec (≈ 2–3 bpw): trellis-coded quantization

Status: research + prototype (2026-10-08 / 09), **integrated 2026-10-09 (§7)**. Code: `suc/quant/trellis.py` (codec
`kind: tcq`), `suc/kernels/tcq_matmul.py` (Metal kernels from the bitstreams), `benchmarks/tcq_kernel_speed.py`,
`tests/unit/test_trellis.py`, `tests/unit/test_tcq_runtime.py`, `tests/unit/test_compress_tcq.py`.

## 0. Why

The current low tier is VQ G=4 K=256 + input rotation + 0.5 % column residual (≈ 2.2 bpp; rel. MSE ≈ 0.097 on rotated
weights). On DeepSeek-V4-Flash the size variant s97 (96.8 GB, 2.72 bpp, routed experts mostly on that tier) has KL
0.387 vs 0.296 for the ds4 IQ2_XXS mix (97.6 GB), and decode did not speed up: the lookup kernel is ALU / issue-bound
on the M1 Ultra (≈ 33 GB/s of 2-bit code bytes in the single-linear benchmark of §3, vs ≈ 80 GB/s for MLX 4-bit),
not bandwidth-bound. Goal: a low tier that beats IQ2_XXS /
Q2_K at equal bits and decodes at least as fast.

## 1. Research summary

MSE figures are on i.i.d. N(0, 1) (what an incoherence rotation + row-RMS normalisation makes of a weight row);
"lit." = published / recalled from the papers, "ours" = measured here (§2).

| Method | how | 2 bpw | 2.5 bpw | 3 bpw | decode on Metal | D4 / D6 |
|---|---|---|---|---|---|---|
| Rate–distortion bound D(R) = 2^-2R | — | 0.0625 | 0.0313 | 0.0156 | — | — |
| Lloyd-Max scalar (≈ Q2_K-like grid, no rotation) | 1-D | 0.118 (lit.) | — | 0.035 (lit.) | trivial | — |
| SUC VQ G=4 K=256 (k-means) | 4-D codebook, 8-bit index | **0.094** (ours) | — | — | 1 byte + half4 LUT read per 4 w | 2 KB ✓ |
| SUC additive M=3 G=8 K=256 | 3 × 8-D codebooks | — | — | 0.030 (ours, 3.0 index bits) | 3 LUT reads per 8 w | 3 × 4 KB ✓ |
| QuIP# E8P (E8 lattice, 2^16 codewords from a 256 × 8 table + signs) | 8-D lattice | 0.089 (lit.) | — | (RVQ 2 stages) | 1 table read + sign/parity per 8 w | 4 KB ✓ |
| llama.cpp IQ2_XXS (2.06 bpw) / IQ2_XS (2.31) / IQ2_S (2.5) | E8-derived grid of 256 / 512 / 1024 8-D magnitude vectors + 7 sign bits, 4-bit sub-block scales, imatrix-weighted search | ≈ E8P-class (no rotation; imatrix weighting) | IQ2_S | — | grid + sign table reads | 2–8 KB ✓ |
| AQLM 1×16 (2 bpw) / 2×8 + PV-tuning | additive, 2^16-entry 8-D codebook | ≈ 0.08–0.09 (lit., before fine-tuning) | — | — | 1 MB codebook: random reads | 1×16 ✗ (1 MB), 2×8 = our additive M=2 ✓ |
| QTIP TCQ, bitshift trellis L=16, "1MAD" / "3INST" compute codes | trellis (Viterbi encode), value = f(16-bit window) | 0.069 (lit.), **0.069** (ours) | **0.037** (ours, V=2) | **0.018** (ours) | pure ALU: window extract + 1 LCG mad + byte sum / fp16 trick | no codebook ✓ |
| QTIP HYB (2^9 × 2 LUT, hashed) | trellis + small LUT | 0.071 (lit.), 0.075 (ours, LUT Lloyd-fitted 3 iters) | — | — | 1 LUT read per 2 w | 2 KB ✓ |

Notes.
- **Why trellis wins**: at a fixed rate a TCQ with 2^L states behaves like a very high-dimensional VQ (effective
  codebook of 2^(kT) sequences) at a decode cost of one hash per weight (or per V weights); E8P and IQ2 are 8-D codes,
  our G=4 is a 4-D code. Space-filling + granular gain: 0.094 → 0.069 is −27 % MSE ≈ +0.22 bit at equal size.
- **Decode cost on Metal.** Compute codes need no table at all (no threadgroup memory, no bank conflicts, no
  D4 issue): per step a 2-word window extract (≤ 4 int ops), one `a*s+b` (1 int mad) and a byte sum (1MAD, 5 int ops +
  1 convert) or `(x & m) ^ c` reinterpreted as half2 (3INST, 2 int ops + half add). That is more ALU per weight than
  the VQ G=4 lookup (≈ 1.5 ops + 1 LUT read per weight), so the kernel must be measured (§3), not assumed faster.
  V=2 (one window per 2 weights) halves the extract cost; "pair3" (§2) decodes 2 weights from ONE 3INST word.
- **Encode cost**: exact Viterbi over 2^L states per step. Ours (Metal step kernel, two-pass tail-biting with a
  64-step first pass): ≈ 12 µs/weight at L=16 on the M3 (≈ 100 s per 8.4 M weights) → Qwen3-0.6B ≈ 1–1.5 h,
  Qwen3-8B ≈ 2–3 h on the M1 Ultra (bandwidth-bound, ×6–8 the M3), DeepSeek-V4-Flash routed experts (≈ 280 B weights)
  ≈ 4 days on the Ultra: too slow as is (§5: options).
- **D4**: no codebook (compute codes) or a 2 KB LUT (HYB) — well inside 16 KB. **D6**: the trellis is a per-tensor (or
  per-row-group) base representation with a fixed rate per tensor; residual / outliers stay as before (the column
  residual composes unchanged: `ResidualCodec(TrellisCodec)`), so allocation candidates are just more (codec, rate)
  options per tensor.
- **imatrix / Hessian.** IQ2 types use the activation second moments to weight the grid search per column. With an
  incoherence rotation the diagonal metric becomes uniform (codecs.py `_BlockCodebookBase` note), which is why SUC
  uses activation-selected residual columns instead. QTIP / QuIP# recover the activation information with LDLQ
  (GPTQ-style error feedback on the Hessian) — not done here (§5).
- **Fine-tuning**: AQLM's PV-tuning and QTIP's fine-tuning give large gains at 2 bits; for TCQ the codes are fixed,
  only row scales (and residuals) are tunable (SUC Q6 machinery), or the HYB LUT.

## 2. Prototype and MSE (Apple M3, `suc/quant/trellis.py`)

Codec `kind: tcq` (`tcq_bits`, `tcq_L`, `tcq_code`, `tcq_V`): input-side randomized Hadamard rotation (as the
DeepSeek expert candidates; a second, output-side rotation measured no gain: 0.0718 vs 0.0717), fp16 row-RMS scale,
rows cut into tail-biting sequences of T = 256 weights along the input dim, exact Viterbi (Metal step kernel, M steps
fused per launch on closed blocks of 2^(M·kv) states), QTIP's two-pass tail-biting with a 64-step first pass. Actual
bits: T·k per sequence + 16 per row + LUT (hyb only) + 64 B descriptor; the column residual composes unchanged.
Unit tests (`tests/unit/test_trellis.py`): Viterbi = brute force on a tiny trellis, tail-biting / stream layout
round trips, bit accounting, residual wrapper.

Relative MSE ‖W − Ŵ‖² / ‖W‖² (Gaussian: 64 × 4096 i.i.d. N(0,1), no rotation; real: Qwen3-0.6B
`layers.10.self_attn.q_proj`, `layers.10.mlp.down_proj`, `layers.20.mlp.gate_proj`, one-sided rotation, mean of the
three — the three agree within ±0.0003):

| Codec | index bpw | Gaussian | Qwen3-0.6B real | vs VQ G=4 |
|---|---|---|---|---|
| VQ G=4 K=256 (current low tier) | 2 | 0.0941 | 0.0972 | — |
| tcq 1mad L=16 V=1 | 2 | 0.0692–0.0699 | **0.0698** | **−28 %** |
| tcq 3inst L=16 V=1 | 2 | 0.0693 | 0.0696 | −28 % |
| tcq 1mad L=16 V=2 | 2 | 0.0717 | 0.0717 | −26 % |
| tcq hyb L=16 (2^9 × 2 LUT, 3 Lloyd iters) | 2 (+0.06 LUT on this size) | 0.0754 | | −20 % |
| tcq pair3 L=16 V=2 (2 weights from one 3inst word) | 2 | 0.0777 | | −17 % |
| tcq 1mad L=16 V=2 | 2.5 | 0.0372 | | |
| additive M=3 G=8 K=256 (current 3-bit tier) | 3 | 0.0302 | | — |
| tcq 1mad L=16 V=1 | 3 | **0.0184** | | **−39 %** vs additive M=3 |

State size L (Gaussian, 1mad): the encode cost scales with 2^L, the MSE gain degrades slowly — every L ≥ 10 beats
VQ G=4 at 2 bpw:

| L | 10 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|
| k=2 V=1 | 0.0798 | 0.0757 | 0.0731 | 0.0717 | 0.0708 | 0.0698 |
| k=2 V=2 | 0.0871 | 0.0799 | 0.0763 | 0.0743 | 0.0729 | 0.0717 |
| k=3 V=1 | 0.0230 | 0.0207 | 0.0201 | 0.0193 | 0.0189 | 0.0184 |

Encode time (M3, L=16, idle GPU): ≈ 2.0 s per 262 k weights at k=2 V=1 (≈ 8 µs/weight; the fused-step kernel cut
35 % from the one-step kernel, fusing more than 4 bits of states was slower), ≈ 2.4 s at k=2 V=2, ≈ 6 s at k=2.5 V=2
(32 predecessors per state). Real Qwen3-0.6B tensors: ≈ 100 s per 8.4 M weights before the fused kernel.

## 3. Metal kernel (`suc/kernels/tcq_matmul.py`)

`tcq_matmul_metal`: y = x Ŵᵀ straight from the packed streams (execution layout: per row, IN/T sequences of T·k/32
little-endian uint32 words; the input rotation is applied to x by the caller as for the VQ modules). A simdgroup owns
R output rows; per sequence each of its 32 lanes takes T/32 consecutive weights, loads its x span once (shared by
the R rows), loads the 2–3 stream words covering its windows, does one funnel shift per register-sized chunk of
windows and then a compile-time shift + mask per step; the code is evaluated in registers (no LUT, no threadgroup
memory, no D4 question) and the affine part of the code (mean, 1/std, gain) is applied once per row through Σx.
Checked against the Python reference decoder for every code / rate / batch tile (`tests/unit/test_tcq_kernel.py`,
float32: rel. error < 1e-5; fp16 x / output: max abs 0.010–0.029, mean 0.003, rel. 2e-4 on 40 × 512 at L=16).

Single-linear benchmark (spec §43), **M1 Ultra**, median µs of 60 calls cycling 6 weight copies, fp16 x, random
codes (`benchmarks/tcq_kernel_speed.py`, R:`20261008T173626Z_tcq_kernel_speed`; first version without the funnel
chunks: R:`20261008T171109Z_tcq_kernel_speed`):

| kernel (bpw of the codes) | 12288×4096 B=1 | 4096×12288 B=1 | 12288×4096 B=4 | 4096×12288 B=4 | 12288×4096 B=16 | 4096×12288 B=16 |
|---|---|---|---|---|---|---|
| VQ G=4 K=256, current low tier (2) | 381 | 382 | 469 | 440 | 687 (mma) | 1049 (mma) |
| tcq pair3 V=2 (2) | **361** | **383** | **446** | 453 | 913 | 950 |
| tcq 3inst V=1 (2) | 419 | 405 | 489 | 490 | 784 | 836 |
| tcq 1mad V=2 (2) | 496 | 479 | | | | |
| tcq 1mad V=1 (2) | 525 | 510 | 561 | 534 | 838 | 883 |
| tcq pair3 V=2 (2.5) | 426 | 451 | | | | |
| tcq 1mad V=1 (3) | 425 | 503 | 484 | 499 | 829 | 905 |
| tcq pair3 V=2 (3) | 430 | 416 | 503 | 622 | 923 | 987 |
| MLX affine 2-bit g64 (2.5) | 323 | 337 | 410 | 409 | 542 | 582 |
| MLX affine 4-bit g64 (4.5) | 348 | 351 | | | | |

First version (one window extraction from a word pair per step): 1mad V=1 580 / 563, pair3 V=2 439 / 454, 3inst
V=1 421 / 517 µs at B=1. Like the VQ kernels, the trellis kernel is issue-bound on the M1 Ultra (24–45 GB/s of code
bytes vs 80 GB/s for MLX 4-bit): **pair3 V=2 decodes as fast as the current VQ G=4 tier at B ≤ 4** (within ±5 %),
3inst V=1 is 6–10 % slower, 1mad V=1 35 % slower; MLX's affine 2-bit stays 10–15 % faster than both (simpler decode,
no rotation, 0.5 bpw more). At B=16 the VQ tier has its simdgroup-matrix kernel and the trellis kernel only the
vector path (two 8-row tiles): 14–33 % slower on 12288×4096, 10–20 % faster on 4096×12288. Not measured: the ds4 / llama.cpp IQ2_XXS Metal kernel in isolation (only whole
models, §5 of compression_report).

## 4. Quality: KL at equal bits vs the current tiers and llama.cpp i-quants

Protocol: `suc quantize` (uniform codec over every eligible linear, one tensor at a time), embedding / LM head
affine 4-bit in every SUC row; KL / top-1 / ΔCE on the **16 × 512 validation ids of the model's `compress_*_final`
run** — exactly the positions the GGUF rows are scored on (`scripts/gguf_llama.py kl`, llama.cpp b11477, logits
rounded to bf16, BF16 GGUF noise floor KL 0.0008–0.0009). GGUF files: llama-quantize with an importance matrix
(`llama-imatrix` on the compress run's calibration partition, 256 × 512 tokens; `-IMAT`), and `-E4` = token embedding
and output head forced to Q4_K (4.5 bpw = the bits of the SUC rows' affine 4-bit head), so that the rows differ only in
the linears. "lin. bpw" = bits of the transformer linears (GGUF: from the per-type byte counts of the header; SUC:
encoded tensors incl. row scales, residual and metadata); "bpp" = whole file / HF parameter count. LDLQ rows: full
input second moments of 128 calibration sequences of the same run.

### 4.1 Qwen3-0.6B (dev model; SUC rows on the M3, GGUF rows on the Studio)

R:`20261008T230338Z_low_tier_qwen3_0.6b` (plain rows), R:`20261009T020458Z_low_tier_qwen3_0.6b_ldlq` (LDLQ rows), GGUF R:`20261008T1742*_gguf_llama_quality_q06_*-imat-e4`.
The 0.6B GGUF stores the tied embedding twice (token_embd + output), so its whole file is ≈ 88 MB larger than a
SUC row with the same linears; compare on lin. bpw.

| row | lin. bpw | KL | top-1 | ΔCE | PPL (SUC, WikiText-2 32k) |
|---|---|---|---|---|---|
| SUC affine 2-bit g64 (RTN) | 2.50 | 10.21 | 0.014 | 9.53 | 266148 |
| GGUF IQ2_XXS imat E4 | 2.11 | 3.464 | 0.257 | 2.934 | |
| SUC VQ G=4 K=256 + res (current low tier) | 2.17 | 1.428 | 0.498 | 1.046 | 89.6 |
| SUC tcq pair3 V=2 + res | 2.16 | 0.990 | 0.576 | 0.686 | 72.2 |
| SUC tcq 1mad V=1 + res | 2.16 | 0.843 | 0.605 | 0.548 | 44.5 |
| SUC tcq 1mad V=1 + LDLQ + res | 2.16 | 0.606 | 0.671 | 0.386 | 39.1 |
| **SUC tcq 3inst V=2 + LDLQ + res** | **2.16** | **0.548** | **0.671** | **0.319** | **36.1** |
| GGUF IQ2_S imat E4 | 2.56 | 1.251 | 0.536 | 1.032 | |
| GGUF Q2_K imat E4 | 2.95 | 0.662 | 0.657 | 0.412 | |
| GGUF IQ3_XXS imat E4 | 3.01 | 0.577 | 0.670 | 0.366 | |
| SUC additive M=3 G=8 + res (current 3-bit tier) | 3.20 | 0.374 | 0.729 | 0.193 | 29.4 |
| SUC tcq 1mad V=1 3-bit + res | 3.16 | 0.182 | 0.800 | 0.107 | 23.9 |
| **SUC tcq 1mad V=1 3-bit + LDLQ + res** | **3.16** | **0.136** | **0.823** | **0.083** | **22.9** |
| SUC affine 3-bit g64 (RTN) | 3.50 | 0.982 | 0.570 | 0.678 | 48.1 |

BF16 PPL 19.96. Without LDLQ, at equal bits tcq lowers the KL of the 2-bit tier by 41 % (VQ 1.428 → 0.843) and of
the 3-bit tier by 51 % (additive 0.374 → 0.182); the 2-bit tcq row beats IQ2_S although IQ2_S spends 0.4 bpw more,
and the 3-bit row is far below IQ3_XXS (0.182 vs 0.577, SUC +0.15 bpw). The kernel-cheap pair3 code costs +17 % KL. BlockLDLQ removes another 28–35 % (2-bit) / 25 % (3-bit) of KL:
3inst V=2 + LDLQ 0.548 (−62 % vs VQ G=4, −56 % vs IQ2_S at 0.4 bpw less) and 3-bit 0.136 (−64 % vs additive M=3,
−76 % vs IQ3_XXS). 3inst V=2 (two LCG entries per window, Gaussian MSE 0.0697 = V=1) is the best 2-bit code here.

### 4.2 Qwen3-8B (target; all rows on the Studio)

R:`20261009T013056Z_low_tier_qwen3_8b`, GGUF R:`20261008T175*_gguf_llama_quality_q8_*-imat-e4` (default-mix imat files
without E4, i.e. Q5_K output / Q2_K–IQ3_S embedding: IQ2_XXS 2.432 bpp KL 0.541, IQ2_XS 2.633 / 0.371, IQ2_S 2.798 /
0.319, Q2_K 3.205 / 0.175, IQ3_XXS 3.291 / 0.132; R:`20261008T1733*`). BF16 PPL 9.717.

| row | lin. bpw | file GB | bpp | KL | top-1 | ΔCE | PPL |
|---|---|---|---|---|---|---|---|
| SUC affine 2-bit g64 (RTN) | 2.50 | 2.871 | 2.804 | 3.815 | 0.254 | 3.250 | 596 |
| GGUF IQ2_XXS imat E4 | 2.13 | 2.558 | 2.499 | 0.518 | 0.720 | 0.253 | |
| SUC VQ G=4 K=256 + res (current low tier) | 2.17 | 2.585 | 2.525 | 0.430 | 0.750 | 0.248 | 14.43 |
| SUC tcq pair3 V=2 + LDLQ + res | 2.17 | 2.585 | 2.525 | 0.316 | 0.784 | 0.236 | 13.08 |
| **SUC tcq 1mad V=1 + LDLQ + res** | **2.17** | **2.585** | **2.525** | **0.269** | **0.793** | 0.186 | **12.31** |
| GGUF IQ2_S imat E4 | 2.49 | 2.870 | 2.803 | 0.329 | 0.778 | 0.133 | |
| GGUF Q2_K imat E4 | 2.95 | 3.267 | 3.191 | 0.185 | 0.821 | 0.066 | |
| GGUF IQ3_XXS imat E4 | 3.07 | 3.374 | 3.296 | 0.140 | 0.853 | 0.035 | |
| SUC additive M=3 G=8 + res (current 3-bit tier) | ≈ 3.17 | ≈ 3.456 | 3.376 | 0.153 | | | 10.69 |
| **SUC tcq 1mad V=1 3-bit + LDLQ + res** | ≈ 3.17 | ≈ 3.453 | **3.373** | **0.079** | | | **10.32** |

At the same file size as IQ2_XXS (+1 %), tcq 1mad + LDLQ has **48 % less KL than IQ2_XXS** (0.269 vs 0.518) and 38 %
less than the current VQ tier (0.430), and still 18 % less KL than IQ2_S, which spends 0.32 bpw / 285 MB more; top-1
0.793 vs 0.720 / 0.778. **Caveat**: ΔCE does not follow the KL ranking against IQ2_S (0.186 vs 0.133): the tcq row
moves the full distribution less but loses more NLL on the actual next tokens than IQ2_S. KL is the primary metric of
this report; the hard / answer-level suites are not run for these rows. Encode: 12,162 s for the 1mad + LDLQ row
(Studio, before the §5 speed-ups). 3-bit (from the log of the running `low_tier_qwen3_8b_3bit` config; its JSON, with
top-1 / ΔCE and exact lin. bpw, lands with the plain-tcq ablation row): tcq + LDLQ KL 0.079 vs additive M=3 0.153
(−48 %) and IQ3_XXS 0.140 / Q2_K 0.185 at 0.08–0.18 bpp more. Queued after it: the plain-tcq (no LDLQ) 2-bit ablation,
the kernel bench with 3inst V=2, and 3inst V=2 + LDLQ at L=16 / L=12 (`low_tier_qwen3_8b_3inst.yaml`).

### 4.3 State size L (encode cost) with LDLQ, Qwen3-0.6B

R:`20261009T024025Z_low_tier_qwen3_0.6b_lsweep` (M3; encode seconds for the 440 M linear weights, same machine, the
L=16 rows from §4.1 — the M3 was shared with the other runs, so ±30 %):

| row | L=12 | L=14 | L=16 |
|---|---|---|---|
| 2-bit 1mad V=1 + LDLQ: KL / encode s | 0.631 / 295 | 0.588 / 790 | 0.606 / 2942 |
| 2-bit 3inst V=2 + LDLQ | 0.631 / 327 | | 0.548 / 4088 |
| 3-bit 1mad V=1 + LDLQ | 0.159 / 287 | | 0.136 / 3472 |

L=12 costs 4–15 % KL (one run each; the L=14 / L=16 1mad pair is within run-to-run noise) for a 10× shorter encode;
every L=12 row is still far below the current tiers (VQ G=4 1.428, additive M=3 0.374) and the GGUF i-quants.

## 5. Encode speed (the blocker for DeepSeek / GLM)

The first 8B row (1mad V=1 + LDLQ, L=16) took 12,162 s on the Studio (≈ 1.75 µs / weight). Profile of one 1024 × 12288
LDLQ tensor (Studio, GPU shared with that run): Viterbi 29 s, float64 `inv` + `cholesky` of the 12288² second
moment 30 s, host copies of the back pointers 9.6 s. Changes (dev clone, branch `tcq-encode-speed`, merged; all walks
checked against the previous kernel / brute force in `tests/unit/test_trellis.py`):

| change | effect (Studio, shared GPU) |
|---|---|
| register-blocked step kernel (`STEP_KERNEL = "reg"`): one thread per closed block of 2^(M·kv) states, M steps in registers, coalesced loads / stores, no barriers; identical walks | 1.74–1.90 → 1.06–1.23 µs/w (k=2 V=1), 2.21–2.30 → 1.29–1.34 (k=2 V=2) |
| back pointers walked on the GPU (one thread per sequence) | host copies 9.6 → 2.2 s per 12.6 M weights |
| LDLQ factor U = (J chol(J H J) J)⁻¹, float32, cached for shared inputs | 12288²: 30 → 5 s (max rel. diff 6e-5) |
| half-precision path metrics between launches (relative to state 0) | −13 to −25 %; MSE equal to 4 digits, walks tail-biting-valid, cost within 0.2 % |
| adaptive Viterbi batch (2^24 / 2^L sequences) | L=12: 0.18 → 0.10 µs/w, L=10: 0.17 → 0.045 (launch-bound before) |

Net on the 1024 × 12288 LDLQ tensor: 60 s → 22 s (shared GPU). Real-model check: the 8B 3-bit 1mad + LDLQ row (k=3,
more predecessors per state than k=2) ran with all changes in 8,385 s, vs 12,162 s for the 2-bit row before
them (≈ 1.7× faster, scaling the 0.6B k=3 / k=2 encode ratio of 1.18). Remaining L=16 cost is GPU-bound (stack sample: 65 % waiting on the GPU) (fusing more steps, computed vs tabulated values, store layout: no change).

**Projected routed-expert encode for DeepSeek-V4-Flash** (277 B weights; LDLQ factors per expert input are small,
2048 / 4096): L=16 ≈ 277e9 × ~0.6–0.9 µs ≈ 46–70 h; **L=12 ≈ 0.1 µs/w ≈ 8 h** (+4–15 % KL on 0.6B, §4.3); only the
tensors the greedy puts on the low tier need it (≈ 74 % in s97), plus the Fisher scoring of the candidate on every
expert. → L=12 for the first DeepSeek / GLM re-encode, L=16 only if the night budget allows.

## 6. Integration plan (owner decision 2026-10-09: full integration, L=12 for DeepSeek / GLM; implemented, §7)

1. **Allocation candidates.** Add `tcq` rates as candidates of the expert / attention roles next to (or instead of)
   VQ G=4 and additive M=3: `{kind: tcq, tcq_bits: 2, tcq_code: 3inst, tcq_V: 2, tcq_L: 12, ldlq: true, rotate_out:
   false, residual 0.5 % columns}` (≈ 2.2 bpp; best 0.6B row, kernel ≈ pair3 cost — to be confirmed by the queued
   bench) and the 3-bit 1mad version (≈ 3.2 bpp); 2.5 bpw via `tcq_V: 2, tcq_bits: 2.5`
   (kv = 5) only if the 2/3 mix of the greedy is not enough. The candidates fingerprint changes, so the Fisher
   candidate stage is re-scored (calibrate / reference stay).
2. **Pipeline (`suc compress`).** (a) Full input second moments per dense linear in the calibrate stage (8B: ≈ 28 GB
   fp32 in RAM; DeepSeek: one shared `E[x xᵀ]` per MoE layer for gate/up (the routed experts read the same hidden
   state) and per-expert down-projection moments from the routed tokens with stronger damping, or no LDLQ for down);
   (b) `TrellisCodec` with expert-stack rotation sharing (one `seed_in` per (layer, projection) stack, as
   `codebook_scope: expert_stack` does for VQ: no codebook to share, only the rotation); (c) streaming per block as now.
3. **SUC1 + runtimes.** New tensor record `tcq` (k, L, V, T, code, gain; words (out, in/T·NW) uint32; row scales
   fp16) + the existing column residual; Python `CompressedModel` module from `tcq_matmul_metal`; C++: the kernel
   source is generated from the Python one as for the VQ kernels, plus a `suc_switch` variant (expert-gathered rows)
   and an MMA path for prefill (decode a tile of Ŵ into simdgroup matrices; the per-weight decode cost is then
   amortised over the batch).
4. **Encode budget**: §5 — L=12 (≈ 8 h of Viterbi for all DeepSeek routed experts) unless the 8B L=12 row
   (queued, `low_tier_qwen3_8b_3inst.yaml`) shows a larger KL cost than on 0.6B.
5. **Re-encodes** (after 1–4, not before): DeepSeek s97-equivalent (same 96.9 GB budget) and GLM r245 with the tcq
   tier; expected effect from §4: the low-tier experts' output error drops by the measured 8B ratio, i.e. KL of the
   s97 file from 0.387 towards / below the IQ2_XXS mix (0.296) at equal size — to be measured, not assumed.

## 7. Integration (2026-10-09)

What §6 asked for, as implemented (owner decisions: full integration, L = 12 for DeepSeek / GLM, autonomous runs):

- **SUC1 record `tcq`** (docs/format.md §5.5): `chunks` bit-packed at kv bits per sequence row, so the payload IS the
  kernels' stream layout (zero copy); `row_scales` f16; the column residual as for every base. The code normalisation
  (`code_mu`, `code_sd`, float64) is stored in the params, so the Python codec, the SUC1 reference decoder and the C++
  decoder use the same numbers (reference decoder bit-identical to the codec decode, tested). The hashed-LUT code `hyb`
  is not a SUC1 representation.
- **Kernels** (`suc/kernels/tcq_matmul.py`): constants (μ_j, c_j) as a 4-float input instead of compile-time
  constants (one pipeline per shape for every code / gain); "vec" for batch tiles < 5 rows (as before), new "mma"
  register-fragment simdgroup-matrix kernel for tiles of 5..16 rows (batched prefill; K permuted in blocks of 32
  columns, each lane decodes 8 consecutive weights of one sequence with one funnel-shifted register), and the
  expert-stack variants (tile → expert prologue) for `SucSwitchLinear`. Runtime: `SucLinear` / `SucResidualLinear` /
  `SucSwitchLinear` / mixed stacks take tcq tensors (`SucLinear(..., tcq=params)`); tcq projections are never fused.
- **Pipeline (`suc compress`)**: new resumable, layer-streamed stage `hessians` (`suc/analysis/hessians.py`), run when
  a planned codec has `ldlq`: dense linears E[xxᵀ] per input group, MoE layers ONE shared E[xxᵀ] of the routed-expert
  input for every gate / up expert and per-expert E[hhᵀ] of the routed rows for every down expert (DeepSeek: h includes
  the routing weight, applied before w2); one file per block (DeepSeek ≈ 4.4 GB per MoE layer). It is a separate stage
  (not part of `calibrate`) so the calibrate / reference / Fisher artifacts of earlier runs stay valid. Experts with
  fewer than `plan.ldlq_min_tokens` (256) routed rows are encoded without LDLQ. tcq candidates with
  `codebook_scope: expert_stack` share the stack's input rotation (gate / up: the layer's), so the runtime rotates the
  routed input once and the LDLQ factor of the shared gate/up moment is computed once per layer.
- **Candidates**: `plan.candidates_reuse` reuses the scores of every candidate whose codec config and role restriction
  are unchanged from an earlier run with the same Fisher inputs (checked: model hash, seed, calibrate outputs,
  fisher_*, allocation_unit); only the new candidates are encoded. `plan.tcq_score_rows` scores tcq on a deterministic
  row sample (the codec is row-independent; loss extrapolated, bits exact from the shape: `bits_for`). Gate / up experts
  of one stack (shared rotation + hessian) are scored with ONE batched encode of all experts' sampled rows.
- **Measured cost on the Studio (real tensors, L = 12, BlockLDLQ, shared GPU)**: DeepSeek-V4-Flash scoring per MoE layer
  (256 experts, 256 rows each): gate stack 18.5 s, down 0.20 s per expert (2-bit); GLM-5.3-Flash (288 experts) gate
  stack 20.6 / 32.5 s, down 0.19 / 0.24 s per expert (2-bit / 3-bit) → ≈ 2.5–2.7 h of tcq scoring per model.
  Full encode per expert tensor: 0.55–0.85 s (2-bit), 0.9–1.2 s (3-bit); recording the hessians of one MoE layer on 16
  sequences: 3 s. Real-weight rel. MSE (DeepSeek layer 1, with LDLQ, i.e. weighted for the activations, not minimal
  in the weight space): 2-bit 0.13 (gate / up), 0.10–0.12 (down); 3-bit 0.040 / 0.030–0.036.
- **Runs**: `experiments/configs/compress_{deepseek_v4_flash,glm5.3_flash}_tcq.yaml` (s97 / r245 budgets: 96.9 GB
  total, GLM experts 2.45 bpw fully resident) in the queue `scripts/studio_tcq_frontier.sh`; results in report §5c
  (`scripts/tcq_rows.py`).
