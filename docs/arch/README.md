# Post-project target architectures — overview

This directory has one architecture note for each large checkpoint the owner may compress with the finished SUC
pipeline after the project (CLAUDE.md "Current state", spec D7/D8). Each note covers the architecture, every tensor
pattern with its dtype and shape, the stored quantization format and its exact dequantization, the SUC encoding plan,
Hadamard sizes, MLX work and risks:

| Note | Checkpoint (pinned revision) |
|---|---|
| [qwen4_exp.md](qwen4_exp.md) | `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| [deepseek_v4.md](deepseek_v4.md) | `deepseek-ai/DeepSeek-V4-Flash-0731` @ `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| [glm5_next.md](glm5_next.md) | `zai-org/GLM-5.3-Flash` @ `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` |
| [deepseek_v41.md](deepseek_v41.md) | `deepseek-ai/DeepSeek-V4.1-Flash` @ `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` |
| [kimi_k3.md](kimi_k3.md) | `moonshotai/Kimi-K3` @ `f831ab66814297da540d832a5235f8e904f29d06` |
| [gpt_oss.md](gpt_oss.md) | `openai/gpt-oss-120b` @ `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` |

**Method.** No weights were downloaded. The notes use `config.json`, `model.safetensors.index.json`, the modeling or
inference code shipped in the repo, transformers main and mlx-lm main, plus safetensors headers fetched with HTTP Range
requests (3–5 shards per model; the shards are listed in each note). For every model, summing shape × dtype size over
all index entries reproduces the index `total_size` to the byte, so the parameter counts below are exact.

## Comparison

| | qwen4_exp | deepseek_v4 | glm5_next | deepseek_v41 | kimi_k3 | gpt_oss |
|---|---|---|---|---|---|---|
| Total params | 180.0 B | 304.2 B (main 284.3 B) | 321.3 B | 763.2 B | 2 780 B | 116.8 B |
| Active / token | 6.0 B | 13.3 B | 17.4 B | 16.1 B | 104.2 B | 5.1 B |
| Layers | 48 (+1 MTP) | 43 (+3 MTP) | 45 (+1 MTP) | 40 (+3 MTP) | 93 | 36 |
| Attention | 36 Gated DeltaNet + 12 full GQA (24q/2kv, hd 256, sparse indexer) | 1-head shared-KV (512) + compressed KV (ratio 4/128) + indexer top-512 + SWA 128 + sinks; grouped low-rank output | 34 KDA linear + 11 MLA with DSA indexer (k-pool); **no positional encoding** | V4 family + cross-layer KV/index sharing, compress ratios 1/2 | 69 KDA linear + 24 MLA (no RoPE) | GQA 64q/8kv hd 64, alternating SWA 128 / full, sinks |
| MoE | 512 experts, top-10, + 1 gated shared; softmax | 256 experts, top-6, + 1 shared; sqrtsoftplus; layers 0–2 hash-routed | 288 experts, top-8, + 1 shared; sigmoid + bias; 3 dense layers | 384 experts, top-6, + 1 shared; sqrtsoftplus | 896 experts, top-16, + 2 shared; **latent MoE** (7168→3584); 1 dense layer | 128 experts, top-4, no shared; softmax over top-k |
| Residual | 4-stream hyper-connections | 4-stream mHC (Sinkhorn) | 4-stream mHC (Sinkhorn) | 4-stream mHC (Sinkhorn) | attention residuals (depth-wise mixture over block snapshots) | plain pre-norm |
| Extra tables | PLE n-gram tables 51.2 B (102 GB) | — | — | engram tables 196.6 B (203 GB, FP8) | — | — |
| Multimodal (dropped) | vision 0.45 B | — | vision 0.56 B | vision + aligner | vision tower | — |
| Stored format | **BF16** (only D8-compliant source) | experts FP4 E2M1 packed in I8 + UE8M0/32; FP8 E4M3 128×128 UE8M0; rest BF16/F32 | FP8 E4M3 128×128 + F32 `weight_scale_inv`; rest BF16 | experts FP4 (I8) + UE8M0/32; FP8 E4M3 32×32 UE8M0; engram FP8 | experts MXFP4 (U8 + E8M0/32); rest BF16/F32 | experts MXFP4 (U8 + E8M0/32); rest BF16 |
| Stored size | 360.0 GB | 166.9 GB | 328.3 GB | 510.3 GB | 1 560.9 GB | 65.2 GB |
| BF16-equivalent | 360.0 GB | 608.4 GB | 642.7 GB | 1 526.4 GB | 5 559.9 GB | 233.7 GB |
| Est. SUC1 @ ~4.2 bpp ¹ | 169.5 GB (94.0 with n-gram 4-bit) | 151.5 GB (150.0) | 166.8 GB (165.0) | 492.8 GB (394.5 with tables 4.25 bpp) | 1 463.9 GB | 61.5 GB (62.2 native experts) |
| Gain of SUC1 at 4.2 vs stored | 2.1× | ≈1.1× (experts already 4.25 bpw) | ≈2.0× (FP8 → 4.2) | ≈1.0× | ≈1.07× | ≈1.06× |
| Runs on 128 GB Studio at 4.2 | **yes** (~67 GB GPU-resident, n-gram mmap) | no — experts ≤ 2.5–3.0 bpw | no — ~2.8 bpp | no — experts ≤ ~1.7 bpw, tables on SSD | no (≈0.32 bpw needed); SSD expert streaming or ≥14 Macs | **yes** |
| Hadamard-incompatible eligible `in` | none (vision only, dropped) | none | none | 2304 (expert/shared down, 9·2⁸) | 33792 (layer-0 down, 33·2¹⁰) | **2880** (q/k/v, all experts, 45·2⁶) |
| mlx-lm | none (reuse qwen3_next / qwen3_5 gated delta) | none (deepseek_v41 ≈60% reusable) | none (reuse deepseek_v32, kimi_linear, deepseek_v41) | `deepseek_v41` (main only; rejects engram configs) | `kimi_k3` (main only) | `gpt_oss` (complete, native mxfp4) |
| Effort (person-days, model-specific) | 12 | 11 | 14 | 15–22 | 8–12 (+5–10 SSD runtime) | 4 |

¹ Eligible linears at 4.2 bpp; kept tensors at stored precision unless stated in parentheses (embedding/head
affine 4-bit; n-gram/engram tables quantized). The notes give the per-category arithmetic. Where the experts are
already MXFP4/FP4, "native" means the expert bytes are passed through unchanged.

**Totals over the six checkpoints:** 4.47 T params, 8.93 TB BF16-equivalent, 2.99 TB stored, ≈ 2.51 TB as SUC1
at ~4.2 bpp. Kimi-K3 alone is 1.46 TB of that.

## Cross-cutting findings

1. **D8 "full-precision source only" is violated by 5 of 6 checkpoints.** Only Qwen3.8-Flash-Next is published in
   BF16. DeepSeek V4/V4.1, Kimi-K3 and gpt-oss trained their experts with FP4/MXFP4 quantization-aware training, and GLM-5.3 ships FP8 only.
   No higher-precision weights exist. Proposed D8 amendment (needs the owner's decision): the reference model is the
   checkpoint **losslessly dequantized** per tensor at load time, and quality is reported against that model. MXFP4/FP4 and
   E4M3 values with power-of-two (UE8M0) scales are exact in BF16. GLM's F32 `weight_scale_inv` is not guaranteed to
   be a power of two, so GLM dequantizes to F32.
2. **SUC1 at ~4.2 bpp barely shrinks MXFP4 experts** (4.25 bpw native) and adds a second quantization error. Default
   for those models: pass the experts through natively (new SUC1 "foreign codec" record + MLX `mxfp4` kernel), and
   SUC-encode only the BF16/FP8 tensors. SUC pays off there only **below ~3 bpw** (Fisher-greedy allocation,
   measured against native MXFP4), which is exactly the regime needed to fit 128 GB, let alone 8–32 GB Macs.
   Transcoding pays off directly only for GLM-5.3 (FP8 → 4.2, ≈2×) and Qwen3.8 (BF16 → 4.2, 2.1×).
3. **None of these models reaches the 8–32 GB deployment goal.** The smallest at 4.2 bpp is gpt-oss at 61.5 GB.
   gpt-oss at 2.2 bpw experts is ≈33 GB.
4. **Shared infrastructure gaps** (needed by ≥ 2 models, see "Phase A" below): FP8/FP4/MXFP4 dtypes and
   dequantizers in `SafetensorsStore` (`F8_E4M3`, `F8_E8M0`, packed I8/U8 FP4); adapters for stacked, fused and
   interleaved expert tensors, weight+scale pairs, non-HF names (`layers.N.attn.wq_a`, no `model.` prefix),
   `text_config` and `model.language_model.` prefixes, and an explicit "dropped" class for MTP/vision. The current
   `unclassified()` only checks names ending in `.weight` and would silently skip `_blocks`/`.scale`/`weight_packed`.
   Also needed: new roles (linear-attention projections, MLA low-rank `q_a/q_b/kv_a/kv_b`, grouped `o_a/o_b`, indexer, compressor);
   4-stream hyper-connection residuals in the layer-streaming pipeline (calibration activations 4× wider: 5.4–8.6 GB
   per 512×512 BF16 buffer); token ids threaded through streaming (hash routing, PLE, engram); block-diagonal
   Hadamard for 2880 / 2304 / 33792 (`codecs.py` currently raises).
5. **Huge lookup tables** (Qwen PLE 102 GB, DeepSeek-V4.1 engram 203 GB) read only a few rows per token. They must stay
   memory-mapped and outside the GPU working set, and be excluded from "fits in RAM" claims (report them separately).
6. **Long-context mechanisms are untested by our evaluation windows.** The sparse indexers and compressors equal dense
   attention below their budgets (e.g. Qwen3.8's indexer is exactly dense for ≤ 2051 tokens), so the 512/2048-token
   PPL/KL protocol never exercises them. Long-context claims need separate evaluations.

## Recommended implementation order

**Phase A — shared infrastructure (≈ 6–8 person-days).** Do this first; every model below depends on it:
quantized-dtype store + dequantizers with bit-exact tests, stacked/paired-tensor adapter support with a dropped
class, SUC1 native-MXFP4 pass-through record, block-diagonal Hadamard, hyper-connection-aware streaming with token
ids, and the D8 amendment decision.

| # | Model | Why this position | Model-specific effort |
|---|---|---|---|
| 1 | **gpt-oss-120b** | mlx-lm support is complete, the architecture is the simplest, and it runs on the Studio natively and as SUC1. It exercises MXFP4 pass-through, stacked/interleaved experts and the non-power-of-two Hadamard (2880) at low risk. It is also the best candidate for the sub-3 bpw expert research tier (≈44 GB at 3.0 bpw, ≈33 GB at 2.2). | 4 d |
| 2 | **Qwen3.8-Flash-Next** | The only BF16 (D8-compliant) source, and ≈180B matches the scope "up to ≈180B MoE". SUC1 gives a real 2.1× and fits 128 GB. It brings in hyper-connections, gated DeltaNet (reuse mlx-lm `qwen3_next`), the sparse indexer and PLE tables. Needs 360 GB of disk. | 12 d |
| 3 | **DeepSeek-V4-Flash-0731** | Reuses ≈60% of mlx-lm `deepseek_v41` and builds the DeepSeek infrastructure (mHC, compressors, sinks, FP8/FP4) that V4.1 needs later. Fits only with experts ≤ 2.5–3.0 bpw, so it depends on the result of step 1's sub-3 bpw tier. | 11 d |
| 4 | **GLM-5.3-Flash** | The model where SUC transcoding pays most (FP8 → 4.2, 2×), but 167 GB at 4.2 still needs ~2.8 bpp to fit. It reuses mHC (from 3), KDA (mlx-lm `kimi_linear`) and the DSA indexer (mlx-lm `deepseek_v32`). | 14 d |
| 5 | **DeepSeek-V4.1-Flash** | 763B. Needs experts ≤ ~1.7 bpw plus 203 GB of engram tables on SSD. mlx-lm main has `deepseek_v41` but rejects engram configs, so engram must be written. Attempt only if the sub-2 bpw results from steps 1/3 are good. | 15–22 d |
| 6 | **Kimi-K3** | 2.78T, 1.46 TB even at 4.2 bpp. It cannot run on one 128 GB Mac without an SSD expert-streaming runtime (≈26 GB of expert reads per token, ≈4–5 s/token before caching). Compression itself can be layer-streamed (≈17 GB/layer, ≈3.1 TB free disk). Defer: out of reach for the project's hardware unless a multi-Mac or streaming runtime becomes a goal. | 8–12 d (+5–10 d runtime) |

Total to reach a working pipeline on models 1–4: ≈ 47–49 person-days including Phase A.

## Decisions needed from the owner

1. Approve the D8 amendment (dequantized checkpoint as the reference for FP8/FP4/MXFP4 sources).
2. Native MXFP4 pass-through as a SUC1 record type: yes or no. The alternative is always transcoding, even at ≥ 4 bpw.
3. Whether a sub-3 bpw expert tier (with matching AQLM/QuIP#-class baselines, CLAUDE.md rule) is in scope. Without it,
   only gpt-oss and Qwen3.8 run on the 128 GB Studio.
4. Disk budget on the Studio: up to 360 GB for a source checkpoint (Qwen3.8), plus SUC1 output and calibration buffers.
