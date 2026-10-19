---
license: mit
base_model: Qwen/Qwen3.8-27B
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

# Qwen3.8-27B-SUC1-side8

**Qwen3.8-27B** compressed with **SUC** (Semantic Universal Compression) to **4.87 bits per parameter**, in the SUC1 format that the SUC C++ / Metal runtime executes directly from the codes on Apple Silicon (no FP16 reconstruction).

| | |
|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (revision `1d4bf0f2`) |
| Architecture | dense hybrid: 48 Gated DeltaNet + 16 gated full-attention layers |
| Parameters | 26.9 B |
| Original size | 53.79 GB (BF16) |
| SUC1 file | **16.36 GB**, 4.87 bpp (every overhead included), 3.3× smaller than BF16 |
| Memory at run time | 16.75 GB (fully resident) |
| Runs on | Macs with ≥ 32 GB |
| Format / runtime | SUC1 · SUC C++ runtime (`suc1-generate`, `suc1-server`) or the SUC Studio macOS app |
| Engine for Inference | https://github.com/Ar3ntiz/SUC-LM/ |
| APP for Mac OSX | https://github.com/Ar3ntiz/SUC-LM/tree/main/SUCStudio |

## Quantization

- MLP and linear-attention projections: 2-D vector codebook per tensor (VQ, G = 2, K = 256, 1 KB codebook), one-sided randomized Hadamard rotation of the input side, 0.5 % activation-selected INT8 column residuals (the 17,408-wide MLP input uses an orthonormal Hartley factor m = 17).
- Full-attention q / k / v / o: affine 8-bit, groups of 64.
- Embedding and LM head: affine 8-bit, groups of 64.
- Vision tower and MTP head of the original are not included (text-only).
- Calibration: natural text, code, math and instructions (mix v1), disjoint from every evaluation set. Compressed from the original weights, one transformer block at a time.
- Reproduce: `suc compress experiments/configs/compress_qwen3.8_27b_side8.yaml`.

## Quality

C++ runtime next-token KL and top-1 against the original BF16 reference on 16 × 512 held-out tokens (8176 scored positions; lower KL is better); existing WikiText-2 test perplexity.

| | KL | top-1 | PPL original → SUC1 |
|---|---:|---:|---|
| This model | 0.023 | 0.949 | 7.05 → 7.15 |

Other formats (historical reconstructed-weight evaluation):

| Format | Size GB | KL |
|---|---:|---:|
| SUC1 side8 (this model) | 16.36 | 0.023 |
| SUC1 base (4.22 bpp) | 14.18 | 0.031 |
| MLX affine 4-bit | ≈ 15.2 | 0.045 |
| MLX affine 8-bit | ≈ 28.6 | **0.002** |

Answer-level evaluation, 120 graded questions (20 per subset), greedy, same prompts for every system:

| System | Total | GSM8K | MATH | MMLU-Pro | ARC-C | BBH | Long context |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUC1 side8 | 93/120 | 17 | 13 | 8 | 19 | 17 | 19 |
| GGUF Q8_0 (llama.cpp) | 92/120 | 18 | 13 | 6 | 18 | 17 | 20 |
| MLX 8-bit (LM Studio) | 89/120 | 17 | 12 | 5 | 18 | 17 | 20 |

Core suite (30 graded smoke-test questions): 30/30.

## Speed

Mac Studio M1 Ultra (128 GB), batch 1, greedy, SUC C++ runtime: short input uses the earlier 256-token measurement; 2k uses three 64-token runs; 32k uses one bounded 8-token run. TTFT = prompt processing + first token.

| Decode tok/s | Prefill tok/s · TTFT, short prompt | Prefill tok/s · TTFT, 2k tokens | Prefill tok/s · TTFT, 32k tokens |
|---:|---:|---:|---:|
| 25.3 | 43 · 0.35 s | 93 · 22.0 s | 59 · 558 s |

| Runtime / format | Decode tok/s | Prefill tok/s · TTFT, 2k |
|---|---:|---:|
| SUC1 (SUC C++) | 25.3 | 93 · 22.0 s |
| GGUF Q8_0 (llama.cpp) ² | 20.8 | — |
| MLX 8-bit (LM Studio) ² | 19.4 | — |
| MLX 4-bit (mlx-lm) ³ | 26.0 | 214 · 9.6 s |

² Median over the 120-question suite. ³ Historical `suc perf` session, decode at a 128-token context.

Batched prompt prefill is not bit-identical to token-by-token prefill (small, measured differences); decode is bit-identical to the SUC Python reference runtime.

## Quickstart

Requires an Apple Silicon Mac and the SUC C++ runtime (build: `cpp/build.sh ~/suc-local/cpp-build` in the SUC repository).

```bash
hf download Ar3ntiz/qwen3_27B_side8 --local-dir ~/Models/Qwen3.8-27B-SUC1-side8
~/suc-local/cpp-build/suc1-generate ~/Models/Qwen3.8-27B-SUC1-side8/Qwen3.8_27B_side8.suc1 \
    --tokenizer ~/Models/Qwen3.8-27B-SUC1-side8/Qwen3.8_27B_side8-tokenizer.json --chat --prompt "Hello!" --steps 256
```

OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, streaming):

```bash
~/suc-local/cpp-build/suc1-server ~/Models/Qwen3.8-27B-SUC1-side8/Qwen3.8_27B_side8.suc1 --tokenizer ~/Models/Qwen3.8-27B-SUC1-side8/Qwen3.8_27B_side8-tokenizer.json \
    --port 8080 --max-context 32768
```

Or open `model.suc1` and `tokenizer.json` in **SUC Studio** (macOS app: local chat, OpenAI-compatible server, web search tools).

## Limitations

- Lossy compression: outputs differ from the original model (see KL / top-1); validate on your own tasks.
- Runs only with the SUC runtime on Apple Silicon (Metal); not loadable by transformers, llama.cpp or MLX.
- Batch 1 only; long-prompt prefill is slower than llama.cpp / ds4.
- Text only: vision and multi-token-prediction heads of the original are not included.
- Single-seed measurements on one machine (Mac Studio M1 Ultra).

## License

Derivative of Qwen/Qwen3.8-27B, under the same license (Apache-2.0). Copyright of the original weights remains with their authors.

## Citation

```bibtex
@misc{suc2026,
  title  = {SUC: Semantic Universal Compression of Large Language Models for Apple Silicon},
  author = {Arena, Tiziano},
  year   = {2026},
  url    = {https://huggingface.co/Ar3ntiz}
}
```
