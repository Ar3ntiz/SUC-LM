---
license: mit
base_model: Qwen/Qwen3.8-Flash-Next
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

# Qwen3.8-Flash-Next-SUC1-side8

**Qwen3.8-Flash-Next** compressed with **SUC** (Semantic Universal Compression) to **4.35 bits per parameter**, in the SUC1 format that the SUC C++ / Metal runtime executes directly from the codes on Apple Silicon (no FP16 reconstruction).

| | |
|---|---|
| Base model | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (revision `de4b8e4d`) |
| Architecture | MoE hybrid: hyper-connections, Gated DeltaNet + gated QSA attention, 512 experts, per-layer n-gram embeddings |
| Parameters | 176.9 B |
| Original size | 353.89 GB (BF16) |
| SUC1 file | **96.32 GB**, 4.35 bpp (every overhead included), 3.7× smaller than BF16 |
| Memory at run time | 73.82 GB (fully resident) |
| Runs on | Macs with ≥ 96 GB |
| Format / runtime | SUC1 · SUC C++ runtime (`suc1-generate`, `suc1-server`) or the SUC Studio macOS app |
| Engine for Inference | https://github.com/Ar3ntiz/SUC-LM/ |
| APP for Mac OSX | https://github.com/Ar3ntiz/SUC-LM/tree/main/SUCStudio |

## Quantization

- Routed experts: 2-D vector codebook per tensor (VQ, G = 2, K = 256, 1 KB codebook), one-sided randomized Hadamard rotation of the input side, 0.5 % activation-selected INT8 column residuals, one codebook per expert stack.
- Attention, linear attention, shared expert, PLE key / value projections and LM head: affine 8-bit, groups of 64.
- Token embedding: BF16. Per-layer n-gram tables: row-wise affine 4-bit, read from the file row by row (not GPU-resident).
- Routers, hyper-connection mixers, QSA indexer, norms: BF16. Vision tower and MTP head are not included (text-only).
- Calibration: natural text, code, math and instructions (mix v1), disjoint from every evaluation set. Compressed from the original weights, one transformer block at a time.
- Reproduce: `suc compress experiments/configs/compress_qwen3.8_flash_next_side8.yaml`.

## Quality

Next-token KL divergence and top-1 agreement against the original model on 16 × 512 held-out positions (lower KL is better); WikiText-2 test perplexity.

| | KL | top-1 | PPL original → SUC1 |
|---|---:|---:|---|
| This model | 0.064 | 0.913 | 4.695 → 4.704 |

Same protocol, other formats:

| Format | Size GB | KL |
|---|---:|---:|
| SUC1 side8 (this model) | 96.32 (73.8 in memory) | 0.064 |
| GGUF Q4 (ds4) | 177.28 (74.8 in memory) | **0.061** |
| MLX affine 4-bit | — | 0.202 |

Answer-level evaluation, 120 graded questions (20 per subset), greedy, same prompts for every system:

| System | Total | GSM8K | MATH | MMLU-Pro | ARC-C | BBH | Long context |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUC1 side8 | 97/120 | 17 | 15 | 6 | 20 | 19 | 20 |
| GGUF Q4 (ds4) | 99/120 | 18 | 16 | 7 | 19 | 19 | 20 |

Core suite (30 graded smoke-test questions): 30/30.

## Speed

Mac Studio M1 Ultra (128 GB), batch 1, greedy, SUC C++ runtime. TTFT = prompt processing + first token.

| Decode tok/s | Prefill tok/s · TTFT, short prompt | Prefill tok/s · TTFT, 2k tokens | Prefill tok/s · TTFT, 32k tokens |
|---:|---:|---:|---:|
| 28.3 | 55 · 0.27 s | 341 · 6.0 s | 323 · 102 s |

| Runtime / format | Decode tok/s | Prefill tok/s · TTFT, 2k |
|---|---:|---:|
| SUC1 (SUC C++) | 28.3 | 341 · 6.0 s |
| GGUF Q4 (ds4) | 33.6 | — ⁴ |

⁴ ds4 was measured on long prompts only within the 120-question suite (different prompts): see the project report.

Batched prompt prefill is not bit-identical to token-by-token prefill (small, measured differences); decode is bit-identical to the SUC Python reference runtime.

## Quickstart

Requires an Apple Silicon Mac and the SUC C++ runtime (build: `cpp/build.sh ~/suc-local/cpp-build` in the SUC repository).

```bash
hf download Ar3ntiz/qwen3.8_flash_next_side8 --local-dir ~/Models/Qwen3.8-Flash-Next-SUC1-side8
~/suc-local/cpp-build/suc1-generate ~/Models/Qwen3.8-Flash-Next-SUC1-side8/qwen3.8_flash_next_side8.suc1 \
    --tokenizer ~/Models/Qwen3.8-Flash-Next-SUC1-side8/qwen3.8-Flash-Next-tokenizer.json --chat --prompt "Hello!" --steps 256
```

OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, streaming):

```bash
~/suc-local/cpp-build/suc1-server ~/Models/Qwen3.8-Flash-Next-SUC1-side8/qwen3.8_flash_next_side8.suc1 --tokenizer ~/Models/Qwen3.8-Flash-Next-SUC1-side8/qwen3.8-Flash-Next-tokenizer.json \
    --port 8080 --max-context 32768
```

Files of this size need the GPU wired-memory limit raised once: `sudo sysctl iogpu.wired_limit_mb=116000` (128-GB Mac; on 96 GB use about 88000).

Or open `model.suc1` and `tokenizer.json` in **SUC Studio** (macOS app: local chat, OpenAI-compatible server, web search tools).

## Limitations

- Lossy compression: outputs differ from the original model (see KL / top-1); validate on your own tasks.
- Runs only with the SUC runtime on Apple Silicon (Metal); not loadable by transformers, llama.cpp or MLX.
- Batch 1 only; long-prompt prefill is slower than llama.cpp / ds4.
- Text only: vision and multi-token-prediction heads of the original are not included.
- Single-seed measurements on one machine (Mac Studio M1 Ultra).

## License

Derivative of Qwen/Qwen3.8-Flash-Next, distributed under the base model's license: [LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). Check its terms before use.

## Citation

```bibtex
@misc{suc2026,
  title  = {SUC: Semantic Universal Compression of Large Language Models for Apple Silicon},
  author = {Arena, Tiziano},
  year   = {2026},
  url    = {https://huggingface.co/Ar3ntiz}
}
```
