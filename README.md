# SUC — Semantic Universal Compression

**A compressed representation of large language models that preserves model *behavior* per stored bit, and that the
Apple GPU executes directly from the codes — never rebuilding the FP16 weights.**

**Goal:** run large models on Macs with less unified memory than they normally need. Achieved so far: Qwen3-8B in
4.5 GB (8–16 GB Macs), Qwen3.8-27B and Qwen3-30B-A3B in ≈ 17 GB (24–32 GB Macs), and 177–313 B-parameter MoE models
(Qwen3.8-Flash-Next, DeepSeek-V4-Flash, GLM-5.3-Flash, 350–630 GB in BF16) in 74–107 GB on a single 128-GB Mac, at
decode speeds close to llama.cpp / ds4. Compressing the models needs a Mac Studio with an Ultra chip and ≥ 128 GB
(M1 Ultra or later: M2, M3 … M5 Ultra); every number here comes from an M1 Ultra 128 GB, the oldest of them.

Italian: [README-IT.md](README-IT.md) · Specification: [docs/spec.md](docs/spec.md) · All measurements:
[docs/compression_report.md](docs/compression_report.md) · C++ runtime: [cpp/README.md](cpp/README.md) · macOS app:
[SUCStudio/](SUCStudio/README.md)

## Results (Mac Studio M1 Ultra, batch 1, status 14 October 2026)

Every number comes from a result JSON (`experiments/results/`, kept in the private working repository and cited by file name) listed in
[docs/compression_report.md](docs/compression_report.md) §1. Single runs, one seed. Qwen3-0.6B was measured on an
Apple M3 laptop (16 GB).

### Table 1 — Size and quality

The chosen SUC1 file per model. Original = BF16-equivalent size of the full-precision model that quality is measured
against. KL / top-1 = next-token divergence / agreement vs the original on 16 × 512 held-out positions; PPL =
WikiText-2 test.

| Model | Type | Params | Original GB | SUC1 GB | bpp | Smaller by | Memory at run time GB | KL | top-1 | PPL original → SUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B | dense | 0.6 B | 1.50 | 0.32 | 4.26 | 4.7× | 0.42 | 0.092 | 0.853 | 20.97 → 22.83 |
| Qwen3-8B | dense | 8.2 B | 16.38 | 4.32 | 4.22 | 3.8× | 4.63 | 0.049 | 0.902 | 9.72 → 9.99 |
| Qwen3.8-27B (side8) | dense | 26.9 B | 53.79 | 16.36 | 4.87 | 3.3× | 16.75 | 0.023 | 0.950 | 7.05 → 7.15 |
| Qwen3-30B-A3B | MoE | 30.5 B | 61.06 | 15.93 | 4.17 | 3.8× | 17.11 | 0.064 | 0.897 | 8.70 → 9.14 |
| Qwen3.8-Flash-Next (side8) | MoE | 176.9 B | 353.89 | 96.32 | 4.35 | 3.7× | 73.82 ¹ | 0.064 | 0.913 | 4.70 → 4.70 |
| DeepSeek-V4-Flash (tcq) | MoE | 284.3 B | 568.75 ² | 96.82 | 2.72 | 5.9× | 102.57 | 0.235 | 0.825 | 4.78 → 5.59 |
| GLM-5.3-Flash (tcq) | MoE | 313.3 B | 626.65 ² | 99.00 | 2.53 | 6.3× | 106.59 | 0.307 | 0.793 | 3.27 → 4.27 |

bpp = file bits / all parameters, every overhead included (codebooks, scales, residuals, embedding, head, metadata).
¹ The 4-bit n-gram tables (26.9 GB) are read from the file row by row, not kept in GPU memory. ² DeepSeek and GLM are
published already low-precision (FP8 + MXFP4: 166.88 GB; FP8: 328.33 GB); against those files SUC1 is 1.7× and
3.3× smaller. All files run fully in memory; the higher-quality DeepSeek file that streams experts from SSD is in
table 3.

### Table 2 — Speed of the SUC C++ runtime

`suc1-generate`, greedy, 256 generated tokens. *Short* = a 12–13-token prompt; *2k / 32k* = the first 2,048 / 32,768
tokens of the same long text. TTFT = prompt processing + first token.

| Model | Decode tok/s (short prompt) | Prefill tok/s · TTFT, short | Prefill tok/s · TTFT, 2k | Prefill tok/s, 8k | Prefill tok/s · TTFT, 32k | Decode tok/s at 32k |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B (M3) | 143.9 | — | — | — | — | — |
| Qwen3-8B | 76.3 | 128 · 0.11 s | 304 · 6.8 s | 231 | 106 · 309 s | 42.5 |
| Qwen3.8-27B (side8) | 25.3 | 43 · 0.35 s | 93 · 22.1 s | 85 | 58 · 564 s | 20.4 |
| Qwen3-30B-A3B | 82.4 | 199 · 0.08 s | 367 · 5.6 s | 244 | 101 · 324 s | 37.5 |
| Qwen3.8-Flash-Next (side8) | 28.3 | 55 · 0.27 s | 341 · 6.0 s | 323 | 323 · 102 s | 29.2 |
| DeepSeek-V4-Flash (tcq) | 24.0 | 46 · 0.30 s | 171 · 12.0 s | 163 | 138 · 238 s | 13.7 |
| GLM-5.3-Flash (tcq) | 22.2 | 35 · 0.41 s | 116 · 17.7 s | 105 | 94 · 349 s | 13.1 |

The batched prompt prefill is not bit-identical to token-by-token prefill (KL 0.0006–0.03 between the two,
report §1c). Decode is bit-identical to the Python runtime.

### Table 3 — Against other runtimes and formats

Same machine. KL on the same positions and reference as table 1. Decode and short-prompt prefill on the same prompt.
*Long prefill*: SUC at 32k tokens on the shared long text; ds4 / llama.cpp / LM Studio = median over the hard-suite
prompts of 4k–31k tokens (a different prompt set: indicative). *Hard suite*: 120 graded questions (math, knowledge,
reasoning, long context), same prompts for every system.

| Model | System | File GB | KL | Decode tok/s | Prefill tok/s · TTFT, short | Prefill tok/s · TTFT, 2k | Long prefill tok/s | Hard suite /120 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | **SUC1** (M3) | 0.32 | 0.092 | 143.9 | — | — | — | — |
| | GGUF Q4_K_M, LM Studio | 0.48 | 0.097 | 219.8 | 594 · 0.02 s | 7,785 · 0.26 s | — | — |
| | GGUF Q8_0, LM Studio | 0.80 | 0.002 | 197.3 | 615 · 0.02 s | 8,151 · 0.25 s | — | — |
| Qwen3-8B | **SUC1** | 4.32 | 0.049 | 76.3 | 128 · 0.11 s | 304 · 6.8 s | 106 | — |
| | GGUF Q4_K_M, LM Studio | 5.03 | 0.040 | 74.8 | 176 · 0.07 s | 884 · 2.3 s | — | — |
| | GGUF Q8_0, LM Studio | 8.71 | 0.001 | 67.9 | 193 · 0.07 s | 1,007 · 2.0 s | — | — |
| | MLX 4-bit, mlx-lm ³ | 4.61 | 0.092 | 95.8 | 588 (128 tok) | — · 2.7 s | — | — |
| Qwen3.8-27B | **SUC1 side8** | 16.36 | 0.023 | 25.3 | 43 · 0.35 s | 93 · 22.1 s | 58 | 93 |
| | GGUF Q8_0, llama.cpp | 29.05 | — | 20.8 ⁴ | — | — | 252 | 92 |
| | MLX 8-bit, LM Studio | ≈ 28.6 | 0.002 | 19.4 ⁴ | — | — | 213 | 89 |
| | MLX 4-bit, mlx-lm ³ | ≈ 15.2 | 0.045 | 26.0 | 169 (128 tok) | 214 · 9.6 s | — | — |
| Qwen3-30B-A3B | **SUC1** | 15.93 | 0.064 | 82.4 | 199 · 0.08 s | 367 · 5.6 s | 101 | — |
| | GGUF Q4_K_M, LM Studio | 18.56 | 0.039 | 93.6 | 192 · 0.07 s | 1,455 · 1.4 s | — | — |
| | GGUF Q8_0, LM Studio | 32.48 | 0.006 | 79.9 | 146 · 0.09 s | 1,642 · 1.2 s | — | — |
| | MLX 4-bit, mlx-lm ³ | 17.2 | 0.081 | 74.8 | 753 (128 tok) | — · 1.5 s | — | — |
| Qwen3.8-Flash-Next | **SUC1 side8** | 96.32 | 0.064 | 28.3 | 55 · 0.27 s | 341 · 6.0 s | 323 | 97 |
| | GGUF Q4, ds4 ⁵ | 177.28 | 0.061 | 33.6 | 47 · 0.31 s | — | 558 | 99 |
| DeepSeek-V4-Flash | **SUC1 tcq** | 96.82 | 0.235 | 24.0 | 46 · 0.30 s | 171 · 12.0 s | 138 | 103 |
| | GGUF IQ2_XXS mix, ds4 | 97.59 | 0.296 | 24.5 | 42 · 0.33 s | — | 224 | 106 |
| | SUC1 V3, experts from SSD ⁶ | 155.05 | 0.036 | 13.1 | 43 · 0.36 s | 28 · 74 s | — | — |
| | GGUF MXFP4, ds4, experts from SSD | 155.98 | 0.011 | 10.0 | 7 · 1.97 s | — | — | — |
| GLM-5.3-Flash | **SUC1 tcq** | 99.00 | 0.307 | 22.2 | 35 · 0.41 s | 116 · 17.7 s | 94 | 99 |
| | GGUF Q2, ds4, experts from SSD | 96.51 | 0.456 | 24.3 | 41 · 0.36 s | — | 117 | 93 |

³ Historical `suc perf` session (October 2–8): decode at a 128-token context, prefill at 128 tokens, TTFT at 2,048.
⁴ Hard-suite median. ⁵ The GGUF keeps a 102.4-GB BF16 n-gram table on disk; its weights in memory are 74.8 GB (SUC1:
73.8 GB). ⁶ Every routed expert kept in its native MXFP4 format: the quality ceiling. ds4 rows: ds4 engine commit
`9139e2a`, runs of October 6–7 (long prefill and hard suite: October 11); LM Studio: llama.cpp runtime 2.41.0, GGUFs quantized locally without imatrix.

**Reading the tables.**

- **Quality per bit.** On the Qwen3 models SUC beats MLX 4-bit KL with fewer bits (8B 0.049 vs 0.092; 30B-A3B 0.064
  vs 0.081); on 27B side8 halves the KL at +0.36 bpp (0.023 at 4.87 vs 0.045 at 4.51) and the 4.22-bpp base file also
  beats it (0.031). GGUF Q4_K_M has lower KL than SUC on 8B and 30B-A3B at ≈ 0.7 bpp more (not on 0.6B). At 2.5–2.7 bpp SUC
  diverges less than the community 2-bit GGUFs of the same size (DeepSeek 0.235 vs 0.296, GLM 0.307 vs 0.456).
- **Decode** is within 12 % of the fastest GGUF runtime on 8B, 30B-A3B, DeepSeek and GLM, faster than the 8-bit
  files on Qwen3.8-27B (on par with MLX 4-bit), 16 % slower than ds4 on Qwen3.8-Flash-Next; MLX 4-bit decoded 8B
  26 % faster in the historical session.
- **Prefill** is where SUC loses: 1.2–1.7× slower than ds4 and 3–4.5× slower than llama.cpp / LM Studio on long prompts (TTFT at 2k: 5.6–6.8 s vs 1.2–2.3 s
  on the Qwen3 models). Qwen3.8-Flash and DeepSeek reach about 60 % of ds4's long-prompt rate, GLM 80 %.
- **Answers.** On the 120-question hard suite no paired difference is significant (SUC vs the best competitor: −3 to
  +6 questions).
- **Memory.** Every file fits the 111-GB budget of the Studio and runs fully in memory; the 155-GB DeepSeek V3 and the
  ds4 MXFP4 / GLM Q2 GGUFs stream their experts from the SSD.

Glossary: **bpp** bits per parameter of the whole file · **KL** next-token divergence from the original (0 =
identical; ≈ 0.05 small, > 0.25 large) · **top-1** share of positions with the same most likely token · **PPL**
perplexity (lower is better; compare only within a model) · **TTFT** time to first token · **prefill** prompt
reading speed · **decode** generation speed · **MoE** mixture of experts · **tcq** trellis-coded low-bit tier ·
**side8** 8-bit attention / head, 4-bit VQ elsewhere.

---

## Contents

1. [The idea](#1-the-idea)
2. [The SUC representation](#2-the-suc-representation)
3. [Sensitivity and bit allocation](#3-sensitivity-and-bit-allocation)
4. [Compression pipeline and file format](#4-compression-pipeline-and-file-format)
5. [Inference on the codes](#5-inference-on-the-codes)
6. [Usage](#6-usage)
7. [SUC Studio (macOS app)](#7-suc-studio-macos-app)
8. [Limits and claims](#8-limits-and-claims)
9. [Repository layout](#9-repository-layout)
10. [References](#10-references)

## 1. The idea

Batch-1 decoding is memory-bandwidth bound: every active weight is read once per token, so
`tokens/s ≲ bandwidth / bytes per token`, and memory capacity decides which models run at all. SUC asks how much of a
model's *behavior* can be kept per stored bit, and whether the GPU can compute directly on that compact form.

Instead of rounding each weight, a group of $G$ weights becomes one index into a learned codebook of $K$ vectors
(vector quantization): $\log_2 K / G$ index bits per weight. Quality is optimized as behavior (next-token KL),
not as weight error:

$$\min_R \; \text{Size}(R) \quad \text{s.t.} \quad D_{\text{semantic}}(M, M_R) \le \varepsilon$$

SUC combines known pieces — vector/additive codebooks (AQLM, VPTQ), incoherence rotations (QuIP#), activation-aware
errors (GPTQ/AWQ), mixed precision (EXL2, imatrix), trellis coding (QTIP), lookup-table kernels (FLUTE) — around a
behavioral objective, a global bit budget and an execution format for Apple GPUs.

## 2. The SUC representation

`Model = Codebooks + Indices + Residuals + Metadata`, chosen per tensor (expert stack for MoE).

- **Rotation:** the input side of each matrix is rotated with a randomized Hadamard
  transform ($W' = W V^\top$, applied to activations online, $O(n \log n)$). It removes outliers (worst excess
  kurtosis 32.4 → 0.057 on Qwen3-30B-A3B); removing it raises KL by 41 % on Qwen3-8B. One-sided is enough: the output
  rotation bought nothing. Folding the rotation into the residual stream fails on Qwen3 (massive activation channel).
- **Vector codebooks:** $\hat w_b = s_r\, C[i_b]$, k-means++ fitted with an activation-weighted distance. Measured best:
  a 2-D codebook ($G=2$, $K=256$, 4 bits, 1 KB) — 19–27 % lower KL than additive multi-codebooks at equal bits;
  $G=4$, $K=256$ for 2 bits.
- **Additive codebooks** ($\sum_m C_m[i_m]$): kept as an option; slower and worse at equal bits on Qwen3-8B.
- **Trellis-coded tier (tcq):** QTIP-style trellis codes with LDLQ for the 2–3-bit experts of DeepSeek / GLM
  (8B: KL 0.269 vs IQ2_XXS 0.518 at ≈ 2.5 bpp). [docs/low_tier_codec.md](docs/low_tier_codec.md).
- **Native pass-through:** experts published in MXFP4 can stay bit-exact.
- **Residuals:** 0.5 % of entries chosen by activation-weighted impact $\mathbb{E}[x_j^2] R_{ij}^2$, stored as dense
  INT8 columns. Required: removing them doubles KL on Qwen3-8B; magnitude-selected residuals are almost as bad.
- **Fine-tuning** of codebooks/scales block by block: optional (−8 % KL at 3.2 bpp on 8B); not used in the final files.
- **Honest size:** bpp includes indices, codebooks, scales, residuals, embedding, head and metadata.
- **Entropy coding** (rANS/Rice): storage only; saves < 1 % of index bytes on rotated weights, never in the hot path.

## 3. Sensitivity and bit allocation

- **Sensitivity:** layer-local error predicts the model's KL poorly (rank correlation 0.09–0.57); the empirical
  Fisher $\tfrac12 \sum F_j \Delta W_j^2$ ranks tensors at 0.75–0.85. Proxies are validated by injected perturbations.
- **Allocation:** a greedy marginal-value allocator buys upgrades by Δquality/Δbytes until the byte budget is spent
  (per tensor or expert stack; embedding/head included). It pays on MoE (Qwen3-30B-A3B: 20 % lower KL at 3.0 bpp than
  uniform 3.2 bpp) and in the frontier models (which experts go native / 3-bit / 2-bit); on dense models a uniform
  codebook is as good; below ≈ 3 bpp on dense 8B it is worse than uniform (negative result).
- **Calibration:** natural text, code, math and instructions, disjoint from every evaluation set. The published files use
  mix v1 (math 20 %, GSM8K); new configs default to v3 (math 17 % + competition math 3 %); a larger competition-math share (v2) lowered general knowledge on DeepSeek.

## 4. Compression pipeline and file format

`suc compress CONFIG.yaml` runs the resumable stages preflight → calibrate → reference → candidates → plan → encode →
container → evaluate, **one transformer block at a time** (never the whole model in memory), from the original
full-precision checkpoint (BF16, or official FP8/MXFP4 dequantized exactly). Already-quantized files (GGUF, MLX 4-bit)
are only baselines. Every run writes a JSON result with model revision and hash, dataset hash, git commit, seed,
config hash and hardware.

**SUC1** is a memory-mappable container: 64-byte header, canonical JSON directory (architecture, tokenizer id pinned by
SHA-256, tensors, codebooks), 64-byte-aligned payloads (codebooks, bit-packed indices, scales, residuals, raw
tensors), SHA-256 footer. No variable-length section. Spec: [docs/format.md](docs/format.md).

What is compressed: attention and MLP projections and every expert (SUC codecs); embedding / LM head as affine
4- or 8-bit; routers, norms and small mixers stay BF16; KV cache is runtime state.

## 5. Inference on the codes

The kernels never materialize $W$: each output row is $y_r = s_r \sum_g \langle C[i_{r,g}], x_g \rangle$, reading a
few index bits and a codebook entry from on-chip memory per group; the input rotation is one fast Hadamard per shared
input (q|k|v, gate|up), the residual a small dense INT8 pass. Projections sharing an input run as one launch; MoE runs
only the routed experts with grouped kernels. Kernel variants are auto-tuned per GPU and cached
(`~/.cache/suc/kernel_tuning.json`). Every kernel is checked against the Python reference decoder.

Runtimes: Python reference decoder (oracle), MLX/Metal Python runtime, and the **C++ runtime** (`cpp/`): mmap of the
SUC1 file, Metal kernels generated from the Python ones, graphs for Qwen3 dense/MoE, Qwen3.8 (`qwen3_5`,
`qwen4_exp`), DeepSeek-V4 and GLM-5.3, BPE tokenizers, chat templates, sampling, batched prefill, SSD expert
streaming, and the OpenAI-compatible `suc1-server`. Decode is bit-identical to the Python runtime.

## 6. Usage

### Install and test

Apple Silicon, Python 3.12, [uv](https://docs.astral.sh/uv/); keep the virtual environment on a local disk.

```bash
UV_PROJECT_ENVIRONMENT=$HOME/.venvs/suc-lm UV_LINK_MODE=copy uv sync --python 3.12
```

```bash
.venv/bin/python -m pytest
```

Build the C++ runtime (on a local disk):

```bash
cpp/build.sh ~/suc-local/cpp-build
```

### Getting a model

**Download:** the compressed SUC1 files and their tokenizers are published on Hugging Face at
[huggingface.co/Ar3ntiz](https://huggingface.co/Ar3ntiz) (uploads in progress). Download a model folder and point the
runtime or SUC Studio at its `model.suc1` and `tokenizer.json`.

**Or compress it yourself** from the original checkpoint with `suc compress`, which downloads the original into the
Hugging Face cache (`~/.cache/huggingface/hub`, or `$HF_HOME/hub`) and writes the file to
`experiments/artifacts/compress/<config name>/container/model.suc1`:

```bash
.venv/bin/suc compress experiments/configs/compress_qwen3_8b_final.yaml
```

| Model | Config (`experiments/configs/`) | Original (Hugging Face) | Revision |
|---|---|---|---|
| Qwen3-0.6B | `compress_qwen3_0.6b_final.yaml` | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de2` |
| Qwen3-8B | `compress_qwen3_8b_final.yaml` | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | `b968826d` |
| Qwen3.8-27B | `compress_qwen3.8_27b_side8.yaml` | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2` |
| Qwen3-30B-A3B | `compress_qwen3_30b_a3b_final.yaml` | [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B) | `ad44e777` |
| Qwen3.8-Flash-Next | `compress_qwen3.8_flash_next_side8.yaml` | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `de4b8e4d` |
| DeepSeek-V4-Flash | `compress_deepseek_v4_flash_tcq.yaml` | [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | `7872f01b` |
| GLM-5.3-Flash | `compress_glm5.3_flash_tcq.yaml` | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) | `eb9eb208` |

Compressing the frontier models needs an Ultra Mac with ≥ 128 GB (the 167–354 GB published checkpoints are read one block at a time from disk).
A `model.suc1` is a single self-contained file: copy it to the Mac that will run it.

### Getting the tokenizer

Each model folder on [huggingface.co/Ar3ntiz](https://huggingface.co/Ar3ntiz) includes its `tokenizer.json`. The same
file comes from the original repository of the table above, at the listed revision. The SUC1 file pins it as
`Org/Name@revision;tokenizer.json:sha256:…` (print it with `suc1-inspect model.suc1 --no-verify`); the C++ tools look
it up in the Hugging Face cache automatically and check the SHA-256. To fetch only the tokenizer (a few MB) of the
original into the cache:

```bash
hf download Qwen/Qwen3-8B tokenizer.json --revision b968826d9c46dd6066d109eabc6255188de91218
```

`hf` comes with `huggingface_hub` (in this repo: `.venv/bin/hf`); use the full 40-character revision.

Or pass a copy with `--tokenizer path/to/tokenizer.json` (not verified: it must match the SHA-256 in the tokenizer
id). Chat templates are built into the runtime, so `tokenizer.json` is the only file needed.

### Running

```bash
~/suc-local/cpp-build/suc1-generate model.suc1 --chat --prompt "Explain vector quantization." --steps 256
```

```bash
~/suc-local/cpp-build/suc1-server model.suc1 --port 8080 --max-context 32768
```

`suc1-server` speaks the OpenAI API (`/v1/chat/completions`, `/v1/completions`, streaming, `/v1/models`, `/health`;
batch 1); any OpenAI client works with `base_url="http://127.0.0.1:8080/v1"`. Expert streaming for files larger than
memory: `--expert-budget 95G --hot-experts auto`. Large files need the GPU wired limit raised once by the owner
(`sudo sysctl iogpu.wired_limit_mb=116000` on 128 GB). All options: [cpp/README.md](cpp/README.md).

Answer evals and speed benchmarks of a SUC1 file (`suc eval`, `suc perf`): [docs/usage_eval_perf.md](docs/usage_eval_perf.md).

## 7. SUC Studio (macOS app)

[SUC Studio](SUCStudio/README.md) is a native macOS app (Apple Silicon, macOS 15+) built on the same C++ engine:
a local chat client and server for SUC1 models. Installer: `SUCStudio/dist/SUC-Studio-<version>-arm64.pkg` (local
build, ad-hoc signed, no Developer ID); guide: [SUCStudio/USAGE.md](SUCStudio/USAGE.md); history:
`SUCStudio/dist/CHANGELOG.md`.

- **Models:** in Settings pick the `.suc1` file and its `tokenizer.json` (preselected when it sits next to the model).
  Automatic memory plan: the model runs fully in memory when it fits the GPU memory budget (configurable), otherwise
  the MoE experts stream from the SSD; oversized models are rejected before loading.
- **Backends:** the local engine, a remote SUC Studio / `suc1-server`, or **any inference server with an
  OpenAI-compatible API** (ds4, LM Studio, llama.cpp, LLM routers, …), with model selection, API key, streaming and
  stop. Telemetry profiles per backend show real prefill / decode speeds when the server reports them and mark
  estimates explicitly.
- **Built-in server:** exposes the loaded model as an OpenAI-compatible API, with API key and optional LAN access;
  menu-bar status and engine telemetry.
- **Chat:** sessions with export / import, Markdown rendering, separate reasoning, edit and regenerate, resizable
  composer, code editor, links opened in the browser.
- **Attachments:** text files of any extension, Word DOC/DOCX, OpenDocument, RTF, text PDFs; images for vision
  backends. Extracted text stays with the session.
- **Agent features:** global system prompt, selectable skills, MCP servers (local stdio and remote HTTP, credentials in
  the Keychain), a per-session working directory with file and command tools, manual or automatic approvals inside
  the chat, an interactive terminal, subtasks and subagents with separate contexts, configurable hooks, persistent
  artifacts for produced files, an operation log, and a configurable limit of consecutive actions.
- **Extensions and routines:** installable plugins (local folders or a marketplace) and scheduled routines with
  prompt, frequency and history.
- **Appearance and system:** independent fonts and sizes for interface and chat, start at login, keep the Mac awake.

## 8. Limits and claims

- Lossy: weights change; no "lossless" or "same intelligence" claim without task evidence.
- Prefill is 1.2–1.7× slower than ds4 and 3–4.5× slower than llama.cpp / LM Studio on long prompts; batch > 1 decode is not implemented in C++.
- At ≈ 4.4 bpp the ds4 MXFP4 GGUF of DeepSeek beats SUC V3 in KL (0.011 vs 0.036); GGUF Q4_K_M has lower KL than SUC
  on 8B and 30B-A3B at ≈ 0.7 bpp more (not on 0.6B). QuIP#/QTIP/PV-Tuning published results at ≈ 2 bpw on Qwen3-8B are better than SUC's.
- No AQLM/QuIP# same-protocol run; single seed; one machine.
- Entropy coding reduces storage only.

## 9. Repository layout

```text
suc/          io (lazy safetensors, SUC1), models (adapters, MLX graphs), quant (codecs, rotations, residuals, tcq),
              allocation, kernels (Metal + autotune), runtime, pipeline (`suc compress`), evaluation, cli
cpp/          C++ runtime: SUC1 reader/decoder, Metal kernels, model graphs, tokenizer, suc1-generate, suc1-server
SUCStudio/    macOS app on the C++ engine
experiments/  configs/ (YAML) and results/ (JSON, one per run)
docs/         spec, format, compression report, comparisons with ds4, architecture notes
scripts/      measurement queues and report generators
tests/        unit tests (tiny random models) and real-model tests (`-m model`)
```

## 10. References

AQLM (arXiv 2401.06118) · QuIP (2307.13304) · QuIP# (2402.04396) · QTIP (2406.11235) · PV-Tuning (2405.14852) ·
FLUTE (2407.10960) · CodeGEMM (2512.17970) · [llama.cpp](https://github.com/ggml-org/llama.cpp) ·
[MLX](https://ml-explore.github.io/mlx/) · T. M. Cover, "Universal Portfolios", 1991.
