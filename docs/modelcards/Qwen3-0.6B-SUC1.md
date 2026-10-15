---
license: mit
base_model: Qwen/Qwen3-0.6B
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

# Qwen3-0.6B-SUC1

**Qwen3-0.6B** compressed with **SUC** (Semantic Universal Compression) to **4.26 bits per parameter**, in the SUC1 format that the SUC C++ / Metal runtime executes directly from the codes on Apple Silicon (no FP16 reconstruction).

| | |
|---|---|
| Base model | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (revision `c1899de2`) |
| Architecture | dense, 28 layers |
| Parameters | 0.6 B |
| Original size | 1.50 GB (BF16) |
| SUC1 file | **0.32 GB**, 4.26 bpp (every overhead included), 4.7× smaller than BF16 |
| Memory at run time | 0.42 GB (fully resident) |
| Runs on | any Apple Silicon Mac (8 GB) |
| Format / runtime | SUC1 · SUC C++ runtime (`suc1-generate`, `suc1-server`) or the SUC Studio macOS app |
| Engine for Inference | https://github.com/Ar3ntiz/SUC-LM/ |
| APP for Mac OSX | https://github.com/Ar3ntiz/SUC-LM/tree/main/SUCStudio |

## Quantization

- All linear layers: 2-D vector codebook per tensor (VQ, G = 2, K = 256, 1 KB codebook), one-sided randomized Hadamard rotation of the input side, 0.5 % activation-selected INT8 column residuals.
- Tied embedding / LM head: affine 4-bit, groups of 64.
- Norms: BF16.
- Calibration: natural text, code, math and instructions (mix v1), disjoint from every evaluation set. Compressed from the original weights, one transformer block at a time.
- Reproduce: `suc compress experiments/configs/compress_qwen3_0.6b_final.yaml`.

## Quality

Next-token KL divergence and top-1 agreement against the original model on 16 × 512 held-out positions (lower KL is better); WikiText-2 test perplexity.

| | KL | top-1 | PPL original → SUC1 |
|---|---:|---:|---|
| This model | 0.092 | 0.853 | 20.97 → 22.83 |

Same protocol, other formats:

| Format | Size GB | KL |
|---|---:|---:|
| SUC1 (this model) | 0.32 | **0.092** |
| GGUF Q4_K_M (llama.cpp, no imatrix) | 0.48 | 0.097 |
| GGUF Q3_K_M | 0.41 | 0.331 |
| MLX / RTN affine 4-bit | 0.34 ¹ | 0.190 |

¹ RTN affine 4-bit at 4.5 bpw (weights only).

## Speed

Speed measured on an Apple M3 (16 GB): 143.9 tok/s decode, batch 1.

Development model of the project: use it to test the runtime, not for quality-critical work.

## Quickstart

Requires an Apple Silicon Mac and the SUC C++ runtime (build: `cpp/build.sh ~/suc-local/cpp-build` in the SUC repository).

```bash
hf download Ar3ntiz/Qwen3-0.6B-SUC1 --local-dir ~/Models/Qwen3-0.6B-SUC1
~/suc-local/cpp-build/suc1-generate ~/Models/Qwen3-0.6B-SUC1/model.suc1 \
    --tokenizer ~/Models/Qwen3-0.6B-SUC1/tokenizer.json --chat --prompt "Hello!" --steps 256
```

OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, streaming):

```bash
~/suc-local/cpp-build/suc1-server ~/Models/Qwen3-0.6B-SUC1/model.suc1 --tokenizer ~/Models/Qwen3-0.6B-SUC1/tokenizer.json \
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

Derivative of Qwen/Qwen3-0.6B, under the same license (Apache-2.0). Copyright of the original weights remains with their authors.

## Citation

```bibtex
@misc{suc2026,
  title  = {SUC: Semantic Universal Compression of Large Language Models for Apple Silicon},
  author = {Arena, Tiziano},
  year   = {2026},
  url    = {https://huggingface.co/Ar3ntiz}
}
```
