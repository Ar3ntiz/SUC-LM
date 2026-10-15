---
license: mit
base_model: Qwen/Qwen3-30B-A3B
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

# Qwen3-30B-A3B-SUC1

**Qwen3-30B-A3B** compressed with **SUC** (Semantic Universal Compression) to **4.17 bits per parameter**, in the SUC1 format that the SUC C++ / Metal runtime executes directly from the codes on Apple Silicon (no FP16 reconstruction).

| | |
|---|---|
| Base model | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) (revision `ad44e777`) |
| Architecture | MoE, 48 layers, 128 experts (top-8), 3.3 B active |
| Parameters | 30.5 B |
| Original size | 61.06 GB (BF16) |
| SUC1 file | **15.93 GB**, 4.17 bpp (every overhead included), 3.8× smaller than BF16 |
| Memory at run time | 17.11 GB (fully resident) |
| Runs on | Macs with ≥ 32 GB |
| Format / runtime | SUC1 · SUC C++ runtime (`suc1-generate`, `suc1-server`) or the SUC Studio macOS app |
| Engine for Inference | https://github.com/Ar3ntiz/SUC-LM/ |
| APP for Mac OSX | https://github.com/Ar3ntiz/SUC-LM/tree/main/SUCStudio |

## Quantization

- Attention projections: 2-D vector codebook per tensor (VQ, G = 2, K = 256, 1 KB codebook), one-sided randomized Hadamard rotation of the input side, 0.5 % activation-selected INT8 column residuals.
- Routed experts: the same codec, one codebook shared per expert stack (layer × projection), indices / scales / residuals per expert.
- Embedding and LM head: affine 4-bit, groups of 64.
- Router and norms: BF16.
- Calibration: natural text, code, math and instructions (mix v1), disjoint from every evaluation set. Compressed from the original weights, one transformer block at a time.
- Reproduce: `suc compress experiments/configs/compress_qwen3_30b_a3b_final.yaml`.

## Quality

Next-token KL divergence and top-1 agreement against the original model on 16 × 512 held-out positions (lower KL is better); WikiText-2 test perplexity.

| | KL | top-1 | PPL original → SUC1 |
|---|---:|---:|---|
| This model | 0.064 | 0.897 | 8.70 → 9.14 |

Same protocol, other formats:

| Format | Size GB | KL |
|---|---:|---:|
| SUC1 (this model) | 15.93 | 0.064 |
| GGUF Q4_K_M (llama.cpp, no imatrix) | 18.56 | **0.039** |
| GGUF Q3_K_M | 14.71 | 0.084 |
| MLX affine 4-bit | 17.2 | 0.081 |

Core suite (30 graded smoke-test questions): 30/30.

## Speed

Mac Studio M1 Ultra (128 GB), batch 1, greedy, SUC C++ runtime. TTFT = prompt processing + first token.

| Decode tok/s | Prefill tok/s · TTFT, short prompt | Prefill tok/s · TTFT, 2k tokens | Prefill tok/s · TTFT, 32k tokens |
|---:|---:|---:|---:|
| 82.4 | 199 · 0.08 s | 367 · 5.6 s | 101 · 324 s |

| Runtime / format | Decode tok/s | Prefill tok/s · TTFT, 2k |
|---|---:|---:|
| SUC1 (SUC C++) | 82.4 | 367 · 5.6 s |
| GGUF Q4_K_M (LM Studio) | 93.6 | 1,455 · 1.4 s |
| GGUF Q8_0 (LM Studio) | 79.9 | 1,642 · 1.2 s |
| MLX 4-bit (mlx-lm) ³ | 74.8 | — · 1.5 s |

³ Historical `suc perf` session, decode at a 128-token context.

Batched prompt prefill is not bit-identical to token-by-token prefill (small, measured differences); decode is bit-identical to the SUC Python reference runtime.

## Quickstart

Requires an Apple Silicon Mac and the SUC C++ runtime (build: `cpp/build.sh ~/suc-local/cpp-build` in the SUC repository).

```bash
hf download Ar3ntiz/Qwen3-30B-A3B-SUC1 --local-dir ~/Models/Qwen3-30B-A3B-SUC1
~/suc-local/cpp-build/suc1-generate ~/Models/Qwen3-30B-A3B-SUC1/model.suc1 \
    --tokenizer ~/Models/Qwen3-30B-A3B-SUC1/tokenizer.json --chat --prompt "Hello!" --steps 256
```

OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, streaming):

```bash
~/suc-local/cpp-build/suc1-server ~/Models/Qwen3-30B-A3B-SUC1/model.suc1 --tokenizer ~/Models/Qwen3-30B-A3B-SUC1/tokenizer.json \
    --port 8080 --max-context 32768
```

Or open `model.suc1` and `tokenizer.json` in **SUC Studio** (macOS app: local chat, OpenAI-compatible server, web search tools).

## Limitations

- Lossy compression: outputs differ from the original model (see KL / top-1); validate on your own tasks.
- Runs only with the SUC runtime on Apple Silicon (Metal); not loadable by transformers, llama.cpp or MLX.
- Batch 1 only; long-prompt prefill is slower than llama.cpp / ds4.
- Same knowledge cut-off, languages, biases and safety behaviour as the base model.
- Single-seed measurements on one machine (Mac Studio M1 Ultra).

## License

Derivative of Qwen/Qwen3-30B-A3B, under the same license (Apache-2.0). Copyright of the original weights remains with their authors.

## Citation

```bibtex
@misc{suc2026,
  title  = {SUC: Semantic Universal Compression of Large Language Models for Apple Silicon},
  author = {Arena, Tiziano},
  year   = {2026},
  url    = {https://huggingface.co/Ar3ntiz}
}
```
