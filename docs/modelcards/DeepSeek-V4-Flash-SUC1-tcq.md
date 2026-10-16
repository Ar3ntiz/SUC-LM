---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
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

# DeepSeek-V4-Flash-SUC1-tcq

**DeepSeek-V4-Flash-0731** compressed with **SUC** (Semantic Universal Compression) to **2.72 bits per parameter**, in the SUC1 format that the SUC C++ / Metal runtime executes directly from the codes on Apple Silicon (no FP16 reconstruction).

| | |
|---|---|
| Base model | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (revision `7872f01b`) |
| Architecture | MoE: hyper-connections, CSA / HCA compressed attention with lightning indexer, hash + learned routing |
| Parameters | 284.3 B |
| Original size | 568.75 GB (BF16) · published checkpoint 166.88 GB |
| SUC1 file | **96.82 GB**, 2.72 bpp (every overhead included), 5.9× smaller than BF16 |
| Memory at run time | 102.57 GB (fully resident) |
| Runs on | Macs with 128 GB (GPU wired limit raised) |
| Format / runtime | SUC1 · SUC C++ runtime (`suc1-generate`, `suc1-server`) or the SUC Studio macOS app |
| Engine for Inference | https://github.com/Ar3ntiz/SUC-LM/ |
| APP for Mac OSX | https://github.com/Ar3ntiz/SUC-LM/tree/main/SUCStudio |

## Quantization

- Routed experts, Fisher-greedy allocation per expert under a 96.9 GB file budget: trellis-coded 2-bit (QTIP-style, LDLQ) 14,316 tensors, VQ G = 4 (2 bits) 9,431, trellis-coded 3-bit 4,722, native MXFP4 kept as published 4,355, additive 3-codebook 200; all with one-sided rotation and 0.5 % column residuals.
- Attention and shared experts: chosen per tensor among VQ G = 2 + residuals, affine 8-bit and BF16.
- Embedding and LM head: affine 4-bit, groups of 64. Routers, compressors, indexer, norms: BF16. MTP head not included.
- Calibration: natural text, code, math and instructions (mix v1), disjoint from every evaluation set. Compressed from the original weights, one transformer block at a time.
- Reproduce: `suc compress experiments/configs/compress_deepseek_v4_flash_tcq.yaml`.

## Quality

Next-token KL divergence and top-1 agreement against the original model on 16 × 512 held-out positions (lower KL is better); WikiText-2 test perplexity.

| | KL | top-1 | PPL original → SUC1 |
|---|---:|---:|---|
| This model | 0.235 | 0.825 | 4.78 → 5.59 |

Same protocol, other formats:

| Format | Size GB | KL |
|---|---:|---:|
| SUC1 tcq (this model) | 96.82 | **0.235** |
| GGUF IQ2_XXS mix (ds4) | 97.59 | 0.296 |
| SUC1 V3, all experts native MXFP4 (SSD streaming) | 155.05 | 0.036 |
| GGUF MXFP4 experts (ds4, SSD streaming) | 155.98 | 0.011 |

Answer-level evaluation, 120 graded questions (20 per subset), greedy, same prompts for every system:

| System | Total | GSM8K | MATH | MMLU-Pro | ARC-C | BBH | Long context |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUC1 tcq | 103/120 | 19 | 12 | 16 | 19 | 19 | 18 |
| GGUF IQ2_XXS mix (ds4) | 106/120 | 18 | 17 | 16 | 18 | 18 | 19 |

Core suite (30 graded smoke-test questions): 30/30 ⁵.

## Speed

Mac Studio M1 Ultra (128 GB), batch 1, greedy, SUC C++ runtime. TTFT = prompt processing + first token.

| Decode tok/s | Prefill tok/s · TTFT, short prompt | Prefill tok/s · TTFT, 2k tokens | Prefill tok/s · TTFT, 32k tokens |
|---:|---:|---:|---:|
| 24.0 | 46 · 0.30 s | 171 · 12.0 s | 138 · 238 s |

| Runtime / format | Decode tok/s | Prefill tok/s · TTFT, 2k |
|---|---:|---:|
| SUC1 tcq (SUC C++) | 24.0 | 171 · 12.0 s |
| GGUF IQ2_XXS mix (ds4) | 24.5 | — ⁴ |

⁴ ds4 was measured on long prompts only within the 120-question suite (different prompts): see the project report.

Batched prompt prefill is not bit-identical to token-by-token prefill (small, measured differences); decode is bit-identical to the SUC Python reference runtime.

Competition math (MATH) is the weakest subset: 12/20 vs 17/20 for the IQ2_XXS GGUF.

⁵ Measured on the earlier SUC1 files of this model (same runtime).

## Quickstart

Requires an Apple Silicon Mac and the SUC C++ runtime (build: `cpp/build.sh ~/suc-local/cpp-build` in the SUC repository).

```bash
hf download Ar3ntiz/DeepSeek_V4_Flash_tcq --local-dir ~/Models/DeepSeek-V4-Flash-SUC1-tcq
~/suc-local/cpp-build/suc1-generate ~/Models/DeepSeek-V4-Flash-SUC1-tcq/DeepSeek-V4-Flash_tcq.suc1 \
    --tokenizer ~/Models/DeepSeek-V4-Flash-SUC1-tcq/DeepSeek-V4-Flash-tokenizer.json --chat --prompt "Hello!" --steps 256
```

OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, streaming):

```bash
~/suc-local/cpp-build/suc1-server ~/Models/DeepSeek-V4-Flash-SUC1-tcq/DeepSeek-V4-Flash_tcq.suc1 --tokenizer ~/Models/DeepSeek-V4-Flash-SUC1-tcq/DeepSeek-V4-Flash-tokenizer.json \
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

Derivative of deepseek-ai/DeepSeek-V4-Flash-0731, under the same license (MIT). Copyright of the original weights remains with their authors.

## Citation

```bibtex
@misc{suc2026,
  title  = {SUC: Semantic Universal Compression of Large Language Models for Apple Silicon},
  author = {Arena, Tiziano},
  year   = {2026},
  url    = {https://huggingface.co/Ar3ntiz}
}
```
