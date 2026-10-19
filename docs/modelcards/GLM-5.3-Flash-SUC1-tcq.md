---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
library_name: suc
tags:
- SUC
- Semantic Universal Compression
- quantized
- vector-quantization
- apple-silicon
- metal
- mac
---

# GLM-5.3-Flash-SUC1-tcq

**GLM-5.3-Flash** compressed with **SUC** (Semantic Universal Compression) to **2.53 bits per parameter**, in the SUC1 format that the SUC C++ / Metal runtime executes directly from the codes on Apple Silicon (no FP16 reconstruction).

| | |
|---|---|
| Base model | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (revision `eb9eb208`) |
| Architecture | MoE hybrid: mHC, KDA linear attention + MLA with DSA sparse attention, 288 experts |
| Parameters | 313.3 B |
| Original size | 626.65 GB (BF16) · published checkpoint 328.33 GB |
| SUC1 file | **99.00 GB**, 2.53 bpp (every overhead included), 6.3× smaller than BF16 |
| Memory at run time | 106.59 GB (fully resident) |
| Runs on | Macs with 128 GB (GPU wired limit raised) |
| Format / runtime | SUC1 · SUC C++ runtime (`suc1-generate`, `suc1-server`) or the SUC Studio macOS app |
| Engine for Inference | https://github.com/Ar3ntiz/SUC-LM/ |
| APP for Mac OSX | https://github.com/Ar3ntiz/SUC-LM/tree/main/SUCStudio |

## Quantization

- Routed experts, Fisher-greedy allocation at 2.45 bits per weight: trellis-coded 2-bit (QTIP-style, LDLQ) 22,038 tensors, trellis-coded 3-bit 6,596, VQ G = 4 5,633, VQ G = 2 1,937, additive 3-codebook 84; all with one-sided rotation and 0.5 % column residuals.
- Attention (KDA, MLA), dense MLP and shared experts: chosen per tensor among VQ G = 2 + residuals, affine 8-bit and BF16 (4.5 bpw budget).
- Embedding and LM head: affine 4-bit, groups of 64. Routers, indexer, mHC mixers, norms: BF16. Vision tower and MTP head not included (text-only).
- Calibration: natural text, code, math and instructions (mix v1), disjoint from every evaluation set. Compressed from the original weights, one transformer block at a time.
- Reproduce: `suc compress experiments/configs/compress_glm5.3_flash_tcq.yaml`.

## Quality

C++ runtime next-token KL and top-1 against the original BF16 reference on 16 × 512 held-out tokens (8176 scored positions; lower KL is better); existing WikiText-2 test perplexity.

| | KL | top-1 | PPL original → SUC1 |
|---|---:|---:|---|
| This model | 0.304 | 0.796 | 3.27 → 4.27 |

Other formats (historical reconstructed-weight evaluation):

| Format | Size GB | KL |
|---|---:|---:|
| SUC1 tcq (this model) | 99.00 | **0.304** |
| GGUF Q2 (ds4, SSD streaming) | 96.51 | 0.456 |
| SUC1 base, 3.06 bpp (SSD streaming) | 119.93 | 0.261 |

Answer-level evaluation, 120 graded questions (20 per subset), greedy, same prompts for every system:

| System | Total | GSM8K | MATH | MMLU-Pro | ARC-C | BBH | Long context |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUC1 tcq | 99/120 | 19 | 16 | 11 | 18 | 17 | 18 |
| GGUF Q2 (ds4) | 93/120 | 19 | 14 | 8 | 17 | 18 | 17 |

Core suite (30 graded smoke-test questions): 30/30 ⁵.

## Speed

Mac Studio M1 Ultra (128 GB), batch 1, greedy, SUC C++ runtime: short input uses the earlier 256-token measurement; 2k uses three 64-token runs; 32k uses one bounded 8-token run. TTFT = prompt processing + first token.

| Decode tok/s | Prefill tok/s · TTFT, short prompt | Prefill tok/s · TTFT, 2k tokens | Prefill tok/s · TTFT, 32k tokens |
|---:|---:|---:|---:|
| 22.2 | 35 · 0.41 s | 128 · 16.1 s | 97 · 339 s |

| Runtime / format | Decode tok/s | Prefill tok/s · TTFT, 2k |
|---|---:|---:|
| SUC1 tcq (SUC C++) | 22.2 | 128 · 16.1 s |
| GGUF Q2 (ds4) | 24.3 | — ⁴ |

⁴ ds4 was measured on long prompts only within the 120-question suite (different prompts): see the project report.

Batched prompt prefill is not bit-identical to token-by-token prefill (small, measured differences); decode is bit-identical to the SUC Python reference runtime.

⁵ Measured on the earlier SUC1 files of this model (same runtime).

## Quickstart

Requires an Apple Silicon Mac and the SUC C++ runtime (build: `cpp/build.sh ~/suc-local/cpp-build` in the SUC repository).

```bash
hf download Ar3ntiz/glm5-3_flash_tcq --local-dir ~/Models/GLM-5.3-Flash-SUC1-tcq
~/suc-local/cpp-build/suc1-generate ~/Models/GLM-5.3-Flash-SUC1-tcq/glm5.3_flash_tcq.suc1 \
    --tokenizer ~/Models/GLM-5.3-Flash-SUC1-tcq/GLM-5.3-Flash-tokenizer.json --chat --prompt "Hello!" --steps 256
```

OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, streaming):

```bash
~/suc-local/cpp-build/suc1-server ~/Models/GLM-5.3-Flash-SUC1-tcq/glm5.3_flash_tcq.suc1 --tokenizer ~/Models/GLM-5.3-Flash-SUC1-tcq/GLM-5.3-Flash-tokenizer.json \
    --port 8080 --max-context 32768
```

Files of this size need the GPU wired-memory limit raised once: `sudo sysctl iogpu.wired_limit_mb=116000` (128-GB Mac).

Or open `model.suc1` and `tokenizer.json` in **SUC Studio** (macOS app: local chat, OpenAI-compatible server, web search tools).

## Limitations

- Lossy compression: outputs differ from the original model (see KL / top-1); validate on your own tasks.
- Runs only with the SUC runtime on Apple Silicon (Metal); not loadable by transformers, llama.cpp or MLX.
- Batch 1 only; long-prompt prefill is slower than llama.cpp / ds4.
- Text only: vision and multi-token-prediction heads of the original are not included.
- Single-seed measurements on one machine (Mac Studio M1 Ultra).

## License

Derivative of zai-org/GLM-5.3-Flash, under the same license (MIT). Copyright of the original weights remains with their authors.

## Citation

```bibtex
@misc{suc2026,
  title  = {SUC: Semantic Universal Compression of Large Language Models for Apple Silicon},
  author = {Arena, Tiziano},
  year   = {2026},
  url    = {https://huggingface.co/Ar3ntiz}
}
```
