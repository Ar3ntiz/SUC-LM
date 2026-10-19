---
license: mit
base_model: Qwen/Qwen3-8B
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

# Qwen3-8B-SUC1

**Qwen3-8B** compressed with **SUC** (Semantic Universal Compression) to **4.22 bits per parameter**, in the SUC1 format that the SUC C++ / Metal runtime executes directly from the codes on Apple Silicon (no FP16 reconstruction).

| | |
|---|---|
| Base model | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (revision `b968826d`) |
| Architecture | dense, 36 layers |
| Parameters | 8.2 B |
| Original size | 16.38 GB (BF16) |
| SUC1 file | **4.32 GB**, 4.22 bpp (every overhead included), 3.8× smaller than BF16 |
| Memory at run time | 4.63 GB (fully resident) |
| Runs on | Macs with ≥ 16 GB |
| Format / runtime | SUC1 · SUC C++ runtime (`suc1-generate`, `suc1-server`) or the SUC Studio macOS app |
| Engine for Inference | https://github.com/Ar3ntiz/SUC-LM/ |
| APP for Mac OSX | https://github.com/Ar3ntiz/SUC-LM/tree/main/SUCStudio |

## Quantization

- All linear layers: 2-D vector codebook per tensor (VQ, G = 2, K = 256, 1 KB codebook), one-sided randomized Hadamard rotation of the input side, 0.5 % activation-selected INT8 column residuals.
- Embedding and LM head: affine 4-bit, groups of 64.
- Norms: BF16.
- Calibration: natural text, code, math and instructions (mix v1), disjoint from every evaluation set. Compressed from the original weights, one transformer block at a time.
- Reproduce: `suc compress experiments/configs/compress_qwen3_8b_final.yaml`.

## Quality

C++ runtime next-token KL and top-1 against the original BF16 reference on 16 × 512 held-out tokens (8176 scored positions; lower KL is better); existing WikiText-2 test perplexity.

| | KL | top-1 | PPL original → SUC1 |
|---|---:|---:|---|
| This model | 0.050 | 0.902 | 9.72 → 9.99 |

Other formats (historical reconstructed-weight evaluation):

| Format | Size GB | KL |
|---|---:|---:|
| SUC1 (this model) | 4.32 | 0.050 |
| GGUF Q4_K_M (llama.cpp, no imatrix) | 5.03 | **0.040** |
| GGUF Q3_K_M | 4.12 | 0.126 |
| MLX affine 4-bit | 4.61 | 0.092 |

Core suite (30 graded smoke-test questions): 29/30.

## Speed

Mac Studio M1 Ultra (128 GB), batch 1, greedy, SUC C++ runtime: short input and 2k retain the earlier 256-token measurements; 32k uses one bounded 8-token run. TTFT = prompt processing + first token.

| Decode tok/s | Prefill tok/s · TTFT, short prompt | Prefill tok/s · TTFT, 2k tokens | Prefill tok/s · TTFT, 32k tokens |
|---:|---:|---:|---:|
| 76.3 | 128 · 0.11 s | 304 · 6.8 s | 106 · 308 s |

| Runtime / format | Decode tok/s | Prefill tok/s · TTFT, 2k |
|---|---:|---:|
| SUC1 (SUC C++) | 76.3 | 304 · 6.8 s |
| GGUF Q4_K_M (LM Studio) | 74.8 | 884 · 2.3 s |
| GGUF Q8_0 (LM Studio) | 67.9 | 1,007 · 2.0 s |

Batched prompt prefill is not bit-identical to token-by-token prefill (small, measured differences); decode is bit-identical to the SUC Python reference runtime.

## Quickstart

Requires an Apple Silicon Mac and the SUC C++ runtime (build: `cpp/build.sh ~/suc-local/cpp-build` in the SUC repository).

```bash
hf download Ar3ntiz/Qwen3-8B-SUC1 --local-dir ~/Models/Qwen3-8B-SUC1
~/suc-local/cpp-build/suc1-generate ~/Models/Qwen3-8B-SUC1/model.suc1 \
    --tokenizer ~/Models/Qwen3-8B-SUC1/tokenizer.json --chat --prompt "Hello!" --steps 256
```

OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, streaming):

```bash
~/suc-local/cpp-build/suc1-server ~/Models/Qwen3-8B-SUC1/model.suc1 --tokenizer ~/Models/Qwen3-8B-SUC1/tokenizer.json \
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

Derivative of Qwen/Qwen3-8B, under the same license (Apache-2.0). Copyright of the original weights remains with their authors.

## Citation

```bibtex
@misc{suc2026,
  title  = {SUC: Semantic Universal Compression of Large Language Models for Apple Silicon},
  author = {Arena, Tiziano},
  year   = {2026},
  url    = {https://huggingface.co/Ar3ntiz}
}
```
