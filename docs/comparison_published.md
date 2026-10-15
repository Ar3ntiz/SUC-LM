# SUC vs published low-bit baselines (AQLM, QuIP#, QTIP, PV-Tuning, GPTQ/AWQ)

Purpose: spec D3 requires that every SUC target at or below 3 bpw be compared against AQLM or QuIP#. No local AQLM/QuIP# run exists yet, so this document collects **published** numbers from primary sources and puts them next to SUC's measured numbers on a **relative** scale (ΔPPL% vs the 16-bit baseline of the same paper and model). It is a literature comparison, not a new measurement: no SUC number here is new; all come from [benchmarks.md](benchmarks.md).

Compiled 2026-10-02 from the arXiv HTML versions of the papers listed in [Sources](#sources). Table numbers refer to those versions.

## 1. Read this first: what the comparison can and cannot show

1. **Different models have different sensitivity.** A ΔPPL% on Llama-2 says little about Qwen3-8B. Llama-2 is known to quantize easily. Llama-3 is clearly harder at the same bitrate: QuIP# at 2 bits costs +21% on Llama-2-7B and +42% on Llama-3-8B, and QTIP shows the same gap. Before the LC-QAT paper (§3.1), no published sub-4-bit vector-quantization number on Qwen3 existed. Cross-model rows (§4) are **indicative only**.
2. **Context length differs.** SUC evaluates at 2048 tokens: non-overlapping windows, full WikiText-2 test, BF16 PPL 9.717 for Qwen3-8B. The published papers use other lengths: QuIP# Table 4, AQLM, QTIP and PV-Tuning on Llama-2 use 4096 tokens; QTIP and PV-Tuning on Llama-3, and AQLM on Mistral/Mixtral, use 8192 tokens; QuIP# Table 2 uses 2048 tokens. Longer context lowers absolute PPL, and the relative degradation is not invariant to context length. The two Qwen3-8B sources use 2048 tokens, or have an FP16 PPL equal to ours.
3. **bpw accounting differs.** This is the largest systematic difference.
   - *Published methods* quantize only the linear layers inside the transformer blocks. They keep the embedding, the LM head and the norms in FP16, and do not count them. AQLM says this explicitly: its "average bits per parameter" counts only quantized weights. The fractional values it reports (2.02, 3.04, 4.04) come from per-layer codebooks (g·2^B·16 bits) and per-channel scales, so codebook and scale overhead is counted at layer level. This is our reading of the paper's description; no formula is given. QuIP# and QTIP report nominal bits. QuIP# states that its sign vectors / RHT add under 0.01 bpw, and its E8P codebook is a small fixed lattice table. GPTQ/AWQ "W4 g128" in the Qwen3 study is nominal; the group scales and zeros add about 0.13–0.25 bpw, which the paper does not count.
   - *SUC* reports **actual whole-model bpp**: codebooks, scales, indices, residual values and positions, metadata, **embedding and LM head** (CLAUDE.md rule). For Qwen3-8B the untied embedding plus LM head is 1.24 B of 8.19 B parameters (15%). With these two tensors in FP16, a "2-bit" model is ≈ 4.1 whole-model bpp, a "3-bit" model ≈ 5.0 and a "4-bit" model ≈ 5.8. For Llama-2-7B the share is only ~4%.
   - The like-for-like SUC number is therefore the **linear-layer ("allocated" / "blocks") bpw**. Below, every SUC row gives both values. Where the LM head is encoded with the low-bit codec (Q5 grid), SUC is *additionally* handicapped: Q7 showed that the codebook-encoded head alone takes KL 0.248 → 0.162 at the same codec. The published methods never quantize the head.
4. **Calibration and fine-tuning budgets differ by orders of magnitude.**
   - **AQLM** calibrates on 8 M RedPajama tokens. Quantization takes ~14 h for 7B on 2 GPUs and 3–4 days for 70B on 8 GPUs. The rows marked ★ add end-to-end fine-tuning on 4–16 M tokens.
   - **QuIP#** fine-tunes within each block and then end to end, on 256 RedPajama sequences at native context. This takes ~50 GPU-hours for 70B.
   - **PV-Tuning** trains on RedPajama with 1 M-token batches. Its longest run took 2 days on 8 A100s.
   - **LC-QAT** trains on ~4 B FineWeb tokens on 16 A100s, and its PV-Tuning baseline used the same data.
   - **SUC** calibrates on 512 × 512 tokens from its own mixed corpus. Its best rows have **no fine-tuning at all**. The Q6 block-wise tuning uses 100 steps per block on 64 sequences × 512 tokens, 33–43 min on one M1 Ultra. SUC has no end-to-end fine-tuning result on the 8B yet.
5. **Constraints differ.** SUC configurations obey the D4 ceiling: each codebook is ≤ 16 KB and must fit the threadgroup memory of Apple GPUs. AQLM 1×16 uses 2^16 × 8 × FP16 = 1 MiB codebooks per layer, which is not execution-viable under D4. AQLM 2×8 (2 × 4 KB) and QuIP#/QTIP (small or computed codebooks) are closer to SUC's constraint.
6. **Run-to-run reporting differs between papers.** For example, AQLM 2-bit on Llama-2-7B appears as 6.93 in QuIP# Table 4 (an earlier AQLM version), as 6.59 in AQLM v4 Table 1 (no end-to-end fine-tuning), as 6.14 in AQLM v4 Table 4 (end-to-end fine-tuning) and as 6.14 in QTIP Table 5. The tables below use each method's own paper unless stated otherwise.

ΔPPL% = PPL_quantized / PPL_16bit − 1, always against the 16-bit baseline reported **in the same table**.

## 2. SUC numbers used (Qwen3-8B, Mac Studio, from benchmarks.md)

BF16 PPL is 9.717 on the full WikiText-2 test set at seq_len 2048, with non-overlapping windows. "Linear bpw" is over the block linears; "model bpp" is the actual whole-model bpp, including the embedding and LM head. FT means fine-tuning.

| SUC row (section of benchmarks.md) | Linear bpw | Model bpp | Embedding / LM head | FT | PPL | ΔPPL |
|---|---:|---:|---|---|---:|---:|
| MLX affine 4-bit g64 (Phase 0; strong local baseline) | 4.50 | 4.50 | affine 4-bit | — | 10.082 | +3.8% |
| Additive M=4 + 0.5% act. residual (Q5) | 4.18 | 4.18 | codec (M=4) | no | 10.038 | +3.3% |
| Same, column residuals (Q5b) | 4.18 | 4.18 | codec (M=4) | no | 10.073 | +3.7% |
| Uniform M=4 + 0.5% res (Q7b) | 4.18 | 4.23 | affine 4-bit | no | 10.081 | +3.7% |
| Greedy 4.0 (Q7) | 4.00 | 4.08 | affine 4-bit | no | 10.193 | +4.9% |
| Greedy 3.5 (Q7) | 3.50 | 3.65 | affine 4-bit | no | 10.551 | +8.6% |
| Greedy 3.5 (Q7b) | 3.32 | 3.50 | affine 4-bit (chosen) | no | 10.697 | +10.1% |
| Uniform M=3 + 0.5% res (Q7) | 3.17 | 3.38 | affine 4-bit | no | 10.901 | +12.2% |
| Additive M=3 + 0.5% res, one-sided rotation (R1) | 3.18 | 3.18 | codec (M=3) | no | 11.076 | +14.0% |
| Same, block-wise FT (Q6, two-sided rotation) | 3.18 | 3.18 | codec (M=3) | block | 11.223 | +15.5% |
| Greedy 3.0 (Q7) | 3.00 | 3.23 | affine 4-bit | no | 11.002 | +13.2% |
| Greedy 3.0 (Q7b) | 2.73 | 3.00 | affine 4-bit (chosen) | no | 12.150 | +25.0% |
| Greedy 2.5 (Q7) | 2.50 | 2.80 | affine 4-bit | no | 12.824 | +32.0% |
| VQ G=4 K=256 + 0.5% res, column residuals (Q5b) | 2.18 | 2.18 | codec (VQ) | no | 18.536 | +90.8% |
| VQ G=4 + 0.5% res, block-wise FT (Q6) | 2.18 | 2.18 | codec (VQ) | block | 16.49 | +69.7% |
| VQ G=4, no residual (Q1–Q4) | 2.00 | 2.00 | codec (VQ) | no | 162.5 | +1572% |
| RTN affine 3-bit g64 (Q1–Q4) | 3.50 | 3.50 | same | no | 13.364 | +37.5% |

Qwen3-30B-A3B (MoE) has a BF16 PPL of 8.697. Its Q7 rows use shared expert-stack codebooks with the embedding and LM head at affine 4-bit, which is about 2% of the parameters, so model bpp ≈ linear bpw. The rows are:

| Plan | bpp | ΔPPL |
|---|---:|---:|
| Greedy 2.5 | 2.55 | +21.1% |
| Greedy 3.0 | 3.04 | +13.5% |
| Greedy 3.5 | 3.53 | +10.2% |
| Greedy 4.0 | 4.02 | +6.3% |
| RTN 4-bit | 4.51 | +4.3% |
| MLX 4-bit | 4.50 | +4.2% |

## 3. Same model: published numbers on Qwen3-8B (most comparable rows)

Both papers below evaluate the same checkpoint family as ours, the post-trained `Qwen3-8B` (not `-Base`): their FP16 WikiText-2 PPL is 9.71–9.72, against our 9.717. The Qwen3-8B-Base FP16 PPL in [Zheng 2025] is 6.99. These are the only published Qwen3-8B numbers for AQLM / QuIP# / QTIP that we found. No AQLM / QuIP# Qwen3 checkpoint with a reported PPL exists in the official AQLM model zoo; it lists only Qwen2-72B checkpoints, without PPL. No Qwen3-30B-A3B VQ numbers were found.

### 3.1 2-bit — LC-QAT paper, Table 1 (Qwen-3-8B, WikiText-2 test, FP16 9.72)

The paper quantizes all linear layers of all transformer blocks, so the embedding and LM head are not quantized. Bits are nominal 2-bit. The evaluation context length is not stated, but the FP16 PPL matches ours at 2048 to within 0.01. The paper does not describe the configuration or fine-tuning of the QuIP#/QTIP/YAQA rows; AQLM is the 2×8 configuration. PV-Tuning and LC-QAT are trained on ~4 B FineWeb tokens on 16 A100s.

| Method (≈2 bpw, linears only) | PPL | ΔPPL |
|---|---:|---:|
| GPTQ (scalar) | 4.68e4 | collapse |
| QuIP (scalar + incoherence) | 27.61 | +184% |
| **QuIP#** | **12.38** | **+27.4%** |
| **AQLM 2×8** | **18.26** | **+87.9%** |
| QTIP | 10.39 | +6.9% |
| YAQA | 10.50 | +8.0% |
| **PV-Tuning (AQLM + QAT, 4 B tokens)** | **10.65** | **+9.6%** |
| LC-QAT (4 B tokens) | 10.23 | +5.2% |

Next to these, the SUC rows at about 2 bits are:

| SUC row | Linear bpw | Model bpp | ΔPPL |
|---|---:|---:|---:|
| VQ G=4 + 0.5% residual, LM head *also* VQ, no FT | 2.18 | 2.18 | +90.8% |
| Same with block-wise FT | 2.18 | 2.18 | +69.7% |
| Greedy 2.5, LM head affine 4-bit, no FT | 2.50 | 2.80 | +32.0% |

### 3.2 4-bit (and 3-bit) — scalar baselines on Qwen3-8B

These come from [Zheng 2025], "An Empirical Study of Qwen3 Quantization": Tables 2 (per-channel) and 4 (group 128). The setup is 2048-token sequences, 128 C4 calibration samples, FP16 9.71 and nominal bits. A second paper, [Zhou 2026] (VQRound) Table 4, reports FP16 9.72.

| Method | Nominal bits | PPL | ΔPPL |
|---|---|---:|---:|
| RTN, per-channel | 4 | 12.0 | +23.6% |
| AWQ, per-channel | 4 | 10.5 | +8.1% |
| GPTQ, per-channel | 4 | 10.3 | +6.1% |
| AWQ, g128 | 4 (+scales) | 11.3 | +16.4% |
| **GPTQ, g128** | **4 (+≈0.13–0.25 for scales/zeros)** | **9.96** | **+2.6%** |
| GPTQ [Zhou 2026] | 4 | 10.59 | +9.0% |
| VQRound + GPTQ [Zhou 2026] | 4 | 10.18 | +4.7% |
| AWQ, per-channel | 3 | 14.9 | +53.5% |
| AWQ, g128 | 3 (+scales) | 27.5 | +183% |
| GPTQ, g128 | 2 (+scales) | 52.1 | +437% |

In [Zheng 2025], AWQ g128 is *worse* than AWQ per-channel at 3 and 4 bits, and the GPTQ 3-bit rows are empty. We report these numbers as published, without trying to explain them. No published AQLM/QuIP#/QTIP number at 3 or 4 bits on Qwen3-8B was found.

Next to these, the SUC rows at about 4 bits:

| SUC row | Linear bpw | Model bpp | ΔPPL |
|---|---:|---:|---:|
| Additive M=4 + 0.5% residual, LM head also M=4 (Q5) | 4.18 | 4.18 | +3.3% |
| Same, LM head affine 4 (Q7b) | 4.18 | 4.23 | +3.7% |
| Greedy 4.0 (Q7) | 4.00 | 4.08 | +4.9% |
| MLX affine 4-bit g64 | 4.50 | 4.50 | +3.8% |

## 4. Cross-model published numbers (indicative only)

All rows below report WikiText-2 PPL and include FT where the method has it. The † rows are without fine-tuning.

### 4.1 ≈ 2 bpw

| Model (ctx) | 16-bit PPL | QuIP# | AQLM | AQLM★ (e2e FT) | QTIP | PV-Tuning |
|---|---:|---|---|---|---|---|
| Llama-2-7B (4096) | 5.12 | 6.19 (+20.9%); † 8.22 (+60.5%) | 2.02 b: 6.59 (+28.7%) | 2.02 b: 6.14 (+19.9%) | 5.86 (+14.5%) | 2.02 b: 5.84 (+14.1%) |
| Llama-2-13B (4096) | 4.57 | 5.35 (+17.1%) | 1.97 b: 5.60 (+22.5%) | 5.33 (+16.6%) | 5.11 (+11.8%) | 1.97 b: 5.12 (+12.0%) |
| Llama-2-70B (4096) | 3.12 | 3.91 (+25.3%) | 2.07 b: 3.94 (+26.3%) | 3.83 (+22.8%) | 3.70 (+18.6%) | 2.07 b: 3.78 (+21.2%) |
| Llama-3-8B (8192) | 5.54 | 7.84 (+41.5%) | — | — | 7.33 (+32.3%) | 2.01 b: 6.99 (+26.2%) |
| Llama-3-70B (8192) | 2.59 | 5.77 (+123%) | — | — | 4.97 (+91.9%) | 2.07 b: 4.57 (+76.4%) |
| Mistral-7B (8192) | 4.77 | 2.01 b: 6.02 (+26.2%) | 2.01 b: 6.32 (+32.5%) | 5.76 (+20.8%) | — | — |
| Mixtral-8x7B (8192) | 3.46 | 2.01 b: 4.75 (+37.3%) | 1.98 b: 4.61 (+33.2%) | — | — | — |

The rows draw on these tables:

- **QuIP#:** QuIP# Table 4 for Llama-2. QTIP Table 7 for Llama-3. AQLM Tables 3 and 13 for Mistral/Mixtral, which use the authors' released QuIP# models.
- **AQLM and AQLM★:** AQLM v4 Tables 1, 3, 4 and 13.
- **QTIP:** QTIP Table 5 (hybrid code, with FT) and Table 7.
- **PV-Tuning:** PV-Tuning Table 2. Its Llama-3 baselines (5.54, 2.59) equal QTIP's ctx-8192 values.

### 4.2 ≈ 3 bpw

| Model (ctx) | 16-bit | QuIP# | QuIP# † | AQLM | AQLM★ | QTIP | GPTQ |
|---|---:|---|---|---|---|---|---|
| Llama-2-7B (4096) | 5.12 | 5.41 (+5.7%) | 5.60 (+9.4%) | 3.04 b: 5.46 (+6.6%) | 5.38 (+5.1%) | 5.28 (+3.1%) | 8.06 (+57.4%) |
| Llama-2-13B (4096) | 4.57 | 4.78 (+4.6%) | 4.90 (+7.2%) | 3.03 b: 4.82 (+5.5%) | 4.78 (+4.6%) | 4.69 (+2.6%) | 5.85 (+28.0%) |
| Llama-2-70B (4096) | 3.12 | 3.35 (+7.4%) | 3.41 (+9.3%) | 3.01 b: 3.36 (+7.7%) | 3.36 (+7.7%) | 3.26 (+4.5%) | 4.40 (+41.0%) |
| Llama-3-8B (8192) | 5.54 | 6.27 (+13.2%) | — | — | — | 6.01 (+8.5%) | — |
| Llama-3-70B (8192) | 2.59 | 3.59 (+38.6%) | — | — | — | 3.18 (+22.8%) | — |
| Mistral-7B (8192) | 4.77 | — | — | 3.04 b: 5.02 (+5.2%) | 5.12 (+7.3%) | — | — |
| Mixtral-8x7B (8192) | 3.46 | — | — | 3.02 b: 3.79 (+9.5%) | — | — | — |

The GPTQ rows are 3.00-bit, per-channel, act-order runs from AQLM Table 2.

### 4.3 ≈ 4 bpw

| Model (ctx) | 16-bit | QuIP# | QuIP# † | AQLM | QTIP | GPTQ |
|---|---:|---|---|---|---|---|
| Llama-2-7B (4096) | 5.12 | 5.19 (+1.4%) | 5.22 (+2.0%) | 4.04 b: 5.21 (+1.8%) | 5.17 (+1.0%) | 5.49 (+7.2%) |
| Llama-2-13B (4096) | 4.57 | 4.63 (+1.3%) | 4.65 (+1.8%) | 3.94 b: 4.65 (+1.8%) | 4.61 (+0.9%) | 4.78 (+4.6%) |
| Llama-2-70B (4096) | 3.12 | 3.18 (+1.9%) | 3.18 (+1.9%) | 4.14 b: 3.19 (+2.2%) | 3.16 (+1.3%) | 3.35 (+7.4%) |
| Llama-3-8B (8192) | 5.54 | 5.81 (+4.9%) | — | — | 5.67 (+2.3%) | — |
| Llama-3-70B (8192) | 2.59 | 2.99 (+15.4%) | — | — | 2.75 (+6.2%) | — |
| Mistral-7B (8192) | 4.77 | 4.01 b: 4.85 (+1.7%) | — | 4.02 b: 4.89 (+2.5%) | — | — |
| Mixtral-8x7B (8192) | 3.46 | 4.00 b: 3.60 (+4.0%) | — | 3.915 b: 3.57 (+3.2%) | — | — |

The AQLM and GPTQ columns come from AQLM v4 Tables 10, 11 and 13. AQLM's 4-bit rows have no e2e-FT variant.

### 4.4 SUC at the same linear bpw (Qwen3-8B), for reference

| Linear bpw | Best SUC ΔPPL (row) | Range of published ΔPPL on other models |
|---|---|---|
| ≈ 4.0–4.2 | +3.3% (M=4 + res, 4.18) · +4.9% (greedy 4.0) | Llama-2: +0.9 to +2.2%. Llama-3-8B: +2.3 to +4.9%. Llama-3-70B: +6 to +15% |
| ≈ 3.0–3.2 | +13.2% (greedy 3.0) · +14.0% (M=3 + res, 3.18) | Llama-2: +2.6 to +9.4%. Llama-3-8B: +8.5 to +13.2%. Llama-3-70B: +23 to +39% |
| ≈ 2.2–2.5 | +32.0% (greedy 2.5) · +69.7% (VQ 2.18, block FT) · +90.8% (VQ 2.18, no FT) | Llama-2: +12 to +29% at ~2.0. Llama-3-8B: +26 to +42% at ~2.0 |

## 5. Conclusion

The conclusion is restricted to what the numbers above support (spec §48).

- **About 2 bpw, same model (Qwen3-8B, LC-QAT Table 1):** SUC is **not competitive** with the state of the art.
  - At 2.18 bpw its best untuned result (+91% PPL) is about equal to AQLM-2×8 as run in that paper (+88%), and block-wise tuning brings SUC to +70%.
  - QuIP# (+27%), PV-Tuning (+9.6%), YAQA (+8.0%) and QTIP (+6.9%) are **far better** at nominal 2.0 bpw.
  - Even giving SUC 0.5 bpw more and a 4-bit LM head (greedy 2.5, +32%), it does not reach QuIP#. Part of the gap is accounting in the published methods' favour: SUC's 2.18-bpp rows also encode the LM head with the codec, while the papers keep the embedding and LM head in FP16, which would make their "2-bit" model ≈ 4.1 whole-model bpp. Part is method: SUC has no end-to-end fine-tuning, and it uses 2^8-entry 4-D codebooks under the 16 KB D4 ceiling, against much larger trellis or lattice codes and billion-token QAT.
  - This is a **negative result** for the current SUC 2-bit configuration, and it bounds what we may claim: SUC does not match AQLM/QuIP#-class quality at 2 bpw on Qwen3-8B.
- **About 3 bpw:** no AQLM / QuIP# / QTIP number on Qwen3-8B exists to compare against.
  - SUC's best (+13.2% at 3.0 linear bpw, LM head affine 4-bit, no fine-tuning) beats every published Qwen3-8B scalar 3-bit number: AWQ per-channel +54%, AWQ g128 +183%. It also beats our own RTN 3-bit g64 at 3.5 bpw (+37.5%).
  - Cross-model, it sits at the level QuIP# reaches on Llama-3-8B (+13.2%, at 8192 context). It is worse than what QuIP#/AQLM/QTIP reach on Llama-2 (+3 to +9%) and than QTIP on Llama-3-8B (+8.5%).
  - Because Qwen3-8B's 3-bit sensitivity under these methods is unmeasured, **we cannot say whether SUC is behind or on par with AQLM/QuIP# at 3 bpw**. Only a local QuIP#/AQLM/QTIP run on Qwen3-8B with our eval protocol can decide this.
- **About 4 bpw, same model:**
  - SUC M=4 + 0.5% residual at 4.18 bpw (+3.3–3.7%) is better than MLX affine 4-bit at 4.5 bpp (+3.8%) and than per-channel GPTQ/AWQ/RTN (+6 to +24%).
  - It is **slightly worse than the published GPTQ g128** (+2.6% at nominal 4 bits, ≈ 4.13–4.25 actual bpw for the linears, with the LM head in FP16). This run comes from a different eval pipeline, so the 0.7–1.1-point difference is within what pipeline differences can produce. Still, **SUC cannot claim to beat GPTQ at 4 bits** on this evidence.
  - SUC's 4-bit gap is similar to what AQLM/QuIP# show on Llama-3-8B (+4.9% QuIP#, +2.3% QTIP), but this is cross-model and indicative only.
- **MoE (Qwen3-30B-A3B):** no published AQLM/QuIP# numbers exist. The nearest MoE reference is Mixtral-8x7B: AQLM +33% at 1.98 bpw, +9.5% at 3.02 bpw, +3.2% at 3.9 bpw. SUC on the 30B-A3B gives +21% at 2.55 bpp, +13.5% at 3.04 bpp and +6.3% at 4.02 bpp. Different models, context lengths and bpw accounting mean these rows support no ordering claim.
- **What this implies for the plan (D3):**
  1. The D3 requirement is satisfied only for 2 bpw, and only through the LC-QAT paper's runs on Qwen3-8B, whose fine-tuning settings for QuIP#/QTIP/AQLM are not documented.
  2. A **local QuIP# or QTIP run on Qwen3-8B** with the SUC eval protocol remains needed for any ≤ 3 bpw claim.
  3. The gap at 2 bpw points to the missing components that the published methods all have: end-to-end fine-tuning (in LC-QAT Table 1, AQLM-family codes go from +88% one-shot AQLM 2×8 to +9.6% with PV-Tuning on 4 B tokens; the PV-Tuning codebook configuration is not stated there), higher-dimensional codes, and FP16 or 4-bit treatment of the LM head.

## 6. Frontier MoE models (post-project): published ≤ 3-bpw results

Models: DeepSeek-V4-Flash (284 B), GLM-5.3-Flash (~320 B), Qwen3.8-Flash-Next (125 B), compressed by the owner's
frontier queue at ≈ 3 bpp. Web search of 2026-10-05. **No local AQLM / QuIP# / QTIP run exists for any of them**, and
**no peer-reviewed paper** reports ≤ 3-bit results on the exact models: only vendor / community model cards (rows 1–5).
Rows 6–15 are the nearest comparable MoE models. Every number was seen on the cited page; rows marked (s) were read
through a summarised page and should be spot-checked against the raw table before being cited elsewhere.
Bits: **eff** = effective incl. scales / overhead, **nom** = nominal, **size** = only a file size is published (bpw not
stated; any bpw shown is our estimate).

| # | Model | Method | bpw | Metric (quantized vs reference) | Source |
|---|---|---|---|---|---|
| 1 | **DeepSeek-V4-Flash** | routed experts 2-bit VQ (K=1024, dim 4, Hadamard g64); attention / shared experts / head NVFP4; router, norms, embeddings BF16 | size 107 GB resident (source 159 GB) | PPL 4.64 vs FP4 source 3.66 (+27 %; 512-token chunks, dataset not stated); GSM8K 95.0, MMLU-Pro 66.4, HumanEval 65.2 (source ≈ 69.5) | community card, unreviewed [V4-2bit] |
| 2 | DeepSeek-V4-Flash | Unsloth UD-IQ3_XXS | size 103 GB | no accuracy / KLD published for 1–3 bit | [Unsloth-V4] |
| 3 | **GLM-5.3-Flash** | Unsloth UD-IQ1_S / UD-IQ2_XXS / UD-Q2_K_XL / UD-IQ3_XXS / UD-Q3_K_XL | size 93.1 / 101.8 / 108.7 / 120.4 / 147.5 GB (BF16 641.6) | top-1 agreement with BF16 70.9 / 76.3 / 78.3 / 81.6 / 86.3 %; mean KLD 0.670 / 0.450 / 0.380 / 0.284 / 0.160 | [Unsloth-GLM] |
| 4 | **Qwen3.8-Flash-Next** | Unsloth UD-IQ1_S / UD-IQ1_M / UD-Q2_K_XL / UD-IQ3_XXS | size 72.5 / 74.5 / 78.9 / 82.0 GB (BF16 355) | top-1 77.3 / 79.7 / 82.7 / 85.4 %; mean KLD 0.396 / 0.315 / 0.225 / 0.165 | [Unsloth-Q38] |
| 5 | Qwen3.8-Flash-Next | ISTA-DASLab GSQ-RCO GGUF (Q2_0 / IQ2_XS / IQ3_XXS / IQ3_S) | stated 2.40 / 2.50 / 3.00 / 3.50 (s) | task avg (AIME25, GPQA-D, LCB v6) 89.07 / 89.16 / 92.57 / 93.26 vs BF16 93.12 | [GSQ-Q38] |
| 6 | Qwen3-30B-A3B | KBVQ-MoE (VQ) 2 / 3 bit | 2-bit "2.08 effective" (s) | Wiki2 PPL 11.87 / 9.26 vs FP16 8.70; avg acc 63.37 / 69.09 vs 70.24 | [KBVQ] |
| 7 | Qwen3-30B-A3B | llama.cpp IQ2_M (bartowski / Unsloth UD) | size 9.7 / 10.1 GiB (≈ 2.7 / 2.8, our estimate) | wiki.test.raw PPL (ctx 512) 9.98 / 10.37 vs BF16 9.07; KLD 0.069 / 0.067 | [ubergarm] |
| 8 | Qwen3-235B-A22B | UD-Q3_K_XL / Q3_K_L | size 97 / 104 GiB (≈ 3.5 / 3.8, our estimate) | PPL 5.57 / 5.66 vs BF16 5.31; KLD 0.029 / 0.036 | [ubergarm] |
| 9 | DeepSeek-R1 671B | DQ3_K_M / Q3_K_M | eff avg 3.59 / 3.81 | weighted task avg 85.53 / 84.28 vs FP8 85.82 | [Zhao 2025] |
| 10 | DeepSeek-V3 | DQ3_K_M / Q3_K_M | eff avg 3.59 / 3.81 | DQ3_K_M avg 75.73 vs FP8 75.45; Q3_K_M MMLU 87.31 vs 88.06 | [Zhao 2025] |
| 11 | Mixtral-8x7B | AQLM / QuIP# | eff 1.98 / 2.01 | Wiki2 4.61 / 4.75, C4 5.75 / 5.89 vs FP16 3.46 / 5.02 (s) | [AQLM] Table 3 |
| 12 | Mixtral-8x7B | QuIP# 2-bit | nom 2 | Wiki2 4.69, C4 8.25 vs FP16 3.45 / 6.85 | [QuIP#] Table 6 |
| 13 | Mixtral-8x7B | KBVQ-MoE 2 / 3 bit | ≈ 2.08 / 3 | Wiki2 5.39 / 4.07 vs FP16 3.88 | [KBVQ] |
| 14 | Mixtral-8x7B | GPTQ / MxMoE | eff 2.25 / 3.25 | 2.25: 5.69 / 5.63; 3.25: 4.17 / 4.15 vs FP16 3.88 | [MxMoE] |
| 15 | DeepSeek-V2-Lite | GPTQ / MxMoE | eff 2.25 / 3.25 | 2.25: 8.49 / 7.01; 3.25: 6.18 / 6.08 vs FP16 5.92 | [MxMoE] |

Nothing found for GLM-4.5 / 4.5-Air / 4.6, Qwen3-Next-80B-A3B (files exist, no published 2–3 bit metrics), nor
QTIP / HQQ / AutoRound / MC-MoE / EAQuant numbers on these models.

How this relates to SUC (indicative only, no ordering claim): SUC on DeepSeek-V4-Flash native is 3.02 bpp actual
(107.4 GB file), KL 0.248, top-1 agreement 0.819 vs the dequantized checkpoint (WikiText-2, 4096-token windows;
`20261004T125710Z_compress_deepseek_v4_flash_native.json`). The closest published numbers (Unsloth KLD / top-1 on GLM
and Qwen3.8) use llama.cpp's own corpus and context, and GGUF sizes include higher-precision attention / embeddings,
so even the same metric name is not comparable across pipelines. Caveats:

- Metrics differ in kind: Unsloth "top-1" and KLD are agreement with BF16 logits, not task accuracy and not PPL.
- PPL protocols differ (Mixtral FP16 Wiki2 is 3.45 / 3.46 / 3.88 depending on the paper; C4 5.02 vs 6.85;
  llama.cpp rows use ctx 512). Compare deltas within one source only.
- Bits accounting is mixed (eff vs nom vs file size); embeddings, head and attention are often kept at higher precision.
- The ISTA-DASLab AQLM Mixtral HF card reports Wiki2 3.35 at 2 bits, below the paper's FP16 3.46: different protocol,
  not used.
- D3 for the frontier models therefore remains **unmet**: a ≤ 3-bit claim on them needs a local AQLM / QuIP# (or at
  least llama.cpp IQ2/IQ3 with our PPL / KL protocol) run, which does not exist.

## Sources

- **[AQLM]** V. Egiazarian, A. Panferov, D. Kuznedelev, E. Frantar, A. Babenko, D. Alistarh. *Extreme Compression of Large Language Models via Additive Quantization.* ICML 2024. arXiv:2401.06118 (v4). https://arxiv.org/abs/2401.06118
  - Table 1 (Llama-2, 2–2.8 bits), Table 2 (3–3.1 bits), Table 3 (Mixtral, 2 bits), Table 4 (Llama-2 with end-to-end FT, ★), Table 6 (3 bits with FT), Table 10 (4+ bits), Table 11 (Mixtral, 3/4 bits), Table 13 (Mistral-7B).
  - Appendix C: calibration context 4096 (Llama-2) / 8192 (Mistral/Mixtral), 8 M RedPajama tokens. Avg-bits statement in §4.
  - Model zoo: https://github.com/Vahe1994/AQLM
- **[QuIP#]** A. Tseng, J. Chee, Q. Sun, V. Kuleshov, C. De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML 2024. arXiv:2402.04396 (v2). https://arxiv.org/abs/2402.04396
  - Table 2 (ctx 2048), Table 4 (ctx 4096; FT / no-FT / no-E8 ablations).
  - Overhead: sign vectors under 0.01 bpw. FT: ~50 GPU-hours for 70B.
  - Code: https://github.com/Cornell-RelaxML/quip-sharp
- **[QTIP]** A. Tseng, Q. Sun, D. Hou, C. De Sa. *QTIP: Quantization with Trellises and Incoherence Processing.* NeurIPS 2024. arXiv:2406.11235 (v4). https://arxiv.org/abs/2406.11235
  - Table 5 (Llama-2, hybrid code, ctx 4096), Table 7 (Llama-3, ctx 8192).
- **[PV-Tuning]** V. Malinovskii, D. Mazur, I. Ilin, D. Kuznedelev, K. Burlachenko, K. Yi, D. Alistarh, P. Richtárik. *PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression.* 2024. arXiv:2405.14852. https://arxiv.org/abs/2405.14852
  - Table 2. The evaluation context length is not stated; its 16-bit Llama-3 baselines equal QTIP's ctx-8192 values.
- **[LC-QAT]** H. Wang, X. Yu, H. Zhao, F. Wang, X. Han. *LC-QAT: Data-Efficient 2-Bit QAT for LLMs via Linear-Constrained Vector Quantization.* 2026. arXiv:2606.10531 (v3). https://arxiv.org/abs/2606.10531
  - Table 1: Qwen-3-8B and Qwen-3-1.7B at 2 bits, with GPTQ, QuIP, OSTQuant, QuIP#, QTIP, YAQA, AQLM 2×8, PV-Tuning and LC-QAT.
- **[Zheng 2025]** X. Zheng, Y. Li, H. Chu, Y. Feng, X. Ma, J. Luo, J. Guo, H. Qin, M. Magno, X. Liu. *An Empirical Study of Qwen3 Quantization.* 2025. arXiv:2505.02214 (v1). https://arxiv.org/abs/2505.02214
  - Table 2 (per-channel, post-trained Qwen3), Table 4 (per-group g128). Evaluation sequence length 2048; calibration on 128 C4 samples.
- **[Zhou 2026]** Y. Zhou, Q. Chen, L. Benini, G. Sun, Y. Li. *Revisiting Adaptive Rounding with Vectorized Reparameterization for LLM Quantization* (VQRound). 2026. arXiv:2602.02151. https://arxiv.org/abs/2602.02151
  - Table 4 (Qwen3, 4 bits).
- Section 6 (accessed 2026-10-05):
  - **[V4-2bit]** https://huggingface.co/iggerask/DeepSeek-V4-Flash-2bit-GB10 (community model card)
  - **[Unsloth-V4]** https://unsloth.ai/docs/models/deepseek-v4 ; **[Unsloth-GLM]** https://unsloth.ai/docs/models/glm-5.3-flash ; **[Unsloth-Q38]** https://unsloth.ai/docs/models/qwen3.8-next
  - **[GSQ-Q38]** https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
  - **[KBVQ]** KBVQ-MoE, arXiv:2602.11184. https://arxiv.org/abs/2602.11184
  - **[ubergarm]** llama.cpp quant PPL/KLD tables. https://gist.github.com/ubergarm/0f9663fd56fc181a00ec9f634635eb38
  - **[Zhao 2025]** quantized DeepSeek-V3 / R1 evaluation, arXiv:2505.02390. https://arxiv.org/abs/2505.02390
  - **[MxMoE]** arXiv:2505.05799. https://arxiv.org/abs/2505.05799
  - Mixtral AQLM HF release: https://huggingface.co/ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf
- **SUC numbers:** [benchmarks.md](benchmarks.md), sections Phase 0 (Qwen3-8B, Qwen3-30B-A3B), Q1–Q4, Q5, Q5b, Q6, Q7, Q7b, R1/R4 and Q7 30B-A3B. Each row is backed by a result JSON in `experiments/results/`.
