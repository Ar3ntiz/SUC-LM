# SUC C++ runtime

Italiano: [README-IT.md](README-IT.md)

Standalone runtime for SUC1 files (spec §0 D2): it memory-maps the file (zero-copy), runs Metal kernels generated from
the Python ones directly on the codes, and provides tokenizers, chat templates, sampling, batched prefill, SSD expert
streaming and an OpenAI-compatible server. No dependencies beyond the macOS SDK.

| Graph | Models | Notes |
|---|---|---|
| `qwen3`, `qwen3_moe` | Qwen3-0.6B / 8B / 30B-A3B | milestones 3–4 |
| `qwen3_5` (dense on the `qwen4_exp` graph) | Qwen3.8-27B | Gated DeltaNet + gated attention |
| `qwen4_exp` | Qwen3.8-Flash-Next | milestone 5; hyper-connections, QSA, PLE n-gram tables |
| `deepseek_v4` | DeepSeek-V4-Flash | milestone 6; CSA / HCA, native MXFP4 and mixed expert stacks |
| `glm5_next` | GLM-5.3-Flash | milestone 7; mHC, KDA, MLA + DSA |

Every codec of the format is decoded (suc1-decode: raw, bf16, affine, scalar, vq, additive, tcq, native MXFP4); the
model graphs execute all of them except `scalar`. Decode steps are
bit-identical to the Python runtime of the same file (hidden states and logits); the fast batched prefill is not
(measured KL 0.0006–0.03, docs/compression_report.md §1c).

## Current performance (Mac Studio M1 Ultra, batch 1)

`suc1-generate`, greedy, 256 tokens; chosen file per model (docs/compression_report.md §1, where the comparison with
ds4, LM Studio / llama.cpp and MLX is).

| Model (file) | SUC1 GB | Memory GB | Decode tok/s | Prefill tok/s · TTFT, short | Prefill tok/s · TTFT, 2k | Prefill tok/s, 32k | Decode tok/s at 32k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B (final) | 4.32 | 4.63 | 76.3 | 128 · 0.11 s | 304 · 6.8 s | 106 | 42.5 |
| Qwen3.8-27B (side8) | 16.36 | 16.75 | 25.3 | 43 · 0.35 s | 93 · 22.1 s | 58 | 20.4 |
| Qwen3-30B-A3B (final) | 15.93 | 17.11 | 82.4 | 199 · 0.08 s | 367 · 5.6 s | 101 | 37.5 |
| Qwen3.8-Flash-Next (side8) | 96.32 | 73.82 | 28.3 | 55 · 0.27 s | 341 · 6.0 s | 323 | 29.2 |
| DeepSeek-V4-Flash (tcq) | 96.82 | 102.57 | 24.0 | 46 · 0.30 s | 171 · 12.0 s | 138 | 13.7 |
| GLM-5.3-Flash (tcq) | 99.00 | 106.59 | 22.2 | 35 · 0.41 s | 116 · 17.7 s | 94 | 13.1 |
| DeepSeek-V4-Flash V3 (SSD experts) | 155.05 | 116.75 | 13.1 | 43 · 0.36 s | 28 · 73.9 s ¹ | — | — |
| GLM-5.3-Flash base (SSD experts) | 119.93 | 115.03 | 9.1 | 7 · 2.08 s | 16 · 125.8 s ¹ | — | — |

Qwen3-0.6B (Apple M3): 6.95 ms/token (143.9 tok/s). ¹ Row-exact prefill of an older build.

## Quick start

```bash
cpp/build.sh ~/suc-local/cpp-build
~/suc-local/cpp-build/suc1-inspect model.suc1 --no-verify          # header, tokenizer id, byte accounting
~/suc-local/cpp-build/suc1-generate model.suc1 --chat --prompt "Hello" --steps 256
~/suc-local/cpp-build/suc1-server model.suc1 --port 8080 --max-context 32768
```

The SUC1 files and tokenizers are published at [huggingface.co/Ar3ntiz](https://huggingface.co/Ar3ntiz), or produced by
`suc compress` (`experiments/artifacts/compress/<config>/container/model.suc1`). The
tokenizer is resolved from the SUC1 `tokenizer_id` (`Org/Name@rev;tokenizer.json:sha256:…`) in the Hugging Face cache
(`$HF_HUB_CACHE`, `$HF_HOME/hub`, `~/.cache/huggingface/hub`) and checked against its SHA-256; fetch it with
`hf download ORG/NAME tokenizer.json --revision REV` (REV: the full 40-character commit hash), or pass `--tokenizer path/to/tokenizer.json`.
Models and revisions: [README.md §6](../README.md#6-usage). Files larger than the default GPU working set (≈ 75 % of RAM) need the GPU wired
limit raised once (`sudo sysctl iogpu.wired_limit_mb=116000` on 128 GB); files larger than memory stream their experts:
`--expert-budget 95G --hot-experts auto`.

## Contents of this document

Build and tests, tools, library; then the implementation record by milestone (1: reader + reference decoder, 2: one
Metal linear layer, 3: Qwen3 dense graph, 4: fused plan / tokenizer / MoE / pipelined decode, 5: qwen4_exp, SSD
expert streaming, 6: deepseek_v4, 7: glm5_next, tcq tensors), `suc1-server`, and the batched prefill work. Each
milestone's "Results (M3 laptop)" are the dev-model checks of that date; speeds of record are the table above.

## Build

Build directories go on a local disk, never on the SMB share.

```bash
# CMake (C++20)
cmake -S cpp -B ~/suc-local/cpp-build -DCMAKE_BUILD_TYPE=Release && cmake --build ~/suc-local/cpp-build -j
# or without CMake: same sources and flags
cpp/build.sh ~/suc-local/cpp-build
```

Outputs: `libsuc.a`, `suc1-inspect`, `suc1-decode`, `suc-tokenize`, `suc-tests`, `suc-request-params-tests`
(request-helper int64 boundaries under `-fsanitize=float-cast-overflow`, audit F10); on macOS also `libsuc_metal.a`,
`suc1-linear`, `suc-linear-bench`, `suc-mm-bench`, `suc1-generate`, `suc1-server`, `suc1-moe`, `suc-expert-bench`,
`suc-metal-tests` (set `SUC_NO_METAL=1` to skip them). Required flag: `-ffp-contract=off`. Bit-exactness
depends on the compiler never fusing `a*b + c` into an fma, because numpy rounds the product and the sum
separately. The only fused operation is the explicit `std::fma` in the Hadamard `H_m` step (see below).

## Tests

```bash
~/suc-local/cpp-build/suc-tests                                   # C++ unit tests
~/suc-local/cpp-build/suc-request-params-tests                    # request int64 boundaries (float-cast sanitizer)
.venv/bin/python -m pytest tests/unit/test_cpp_decoder.py         # bit-exact cross-check vs Python
.venv/bin/python -m pytest tests/unit/test_cpp_decoder.py -m model # + the real dev container
~/suc-local/cpp-build/suc-metal-tests                             # Metal backend unit tests (milestone 2)
.venv/bin/python -m pytest tests/unit/test_cpp_metal.py           # GPU vs CPU reference, bit-exact vs MLX
.venv/bin/python -m pytest tests/unit/test_cpp_generate.py        # model graph vs the Python runtimes (milestones 3-4)
.venv/bin/python -m pytest tests/unit/test_cpp_moe.py             # SwitchLinear / MoeBlock vs Python (milestone 4)
.venv/bin/python -m pytest tests/unit/test_cpp_server.py          # suc1-server with the openai client
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/unit/test_cpp_tokenizer.py   # C++ tokenizer vs transformers
```

The Python cross-checks run with `SUC_AUTOTUNE=0` (`tests/unit/conftest.py` sets it) so tests never autotune
implicitly; `tests/unit/test_autotune.py` covers the tuner.

- `suc-tests` (no framework) covers: SHA-256 vectors (FIPS, 1M × 'a', hashlib output), both the portable
  path and the ARMv8 SHA-2 path, every chunking and every length 0..200. Bit unpacking at widths 1..32
  against an independent bit-by-bit packer, plus anchors from `suc.quant.packing.pack_indices`. The numpy
  PCG64 sign stream against `np.random.default_rng(seed).integers(0, 2)` for 7 seeds, including seeds ≥ 2³².
  Exhaustive fp16 → fp32 conversion. JSON parsing and its error cases. Hadamard exactness and
  orthonormality. Header, footer and directory validation on synthetic files: bad magic, version, unknown
  or inconsistent flags, counts, alignment, truncation, trailing bytes, footer fields, checksum mismatch
  (and `--no-verify`), sizes, dtype, sections and corrupt JSON.
- `tests/unit/test_cpp_decoder.py` (skipped when the tools are not built; set `SUC_CPP_BUILD` to point
  elsewhere) writes SUC1 files with the Python writer and asserts that every float32 bit pattern matches
  `SucModel.decode_tensor`. The file holds 38 codec tensors and 8 raw tensors. Codecs covered: bf16;
  affine at 2/3/4/8 bits; scalar per tensor, per row group and ragged row groups, with k = 512; vq with
  index widths 1, 3, 4, 5, 9 and 10 bits; additive with M = 2/3/4; rotation in_out, in and out; Hadamard
  sizes 2^k, 12·2^k, 20·2^k, 28·2^k (incl. 5120, 7168) and 16384 (Kronecker split); residuals sparse fp16/int8 (magnitude/importance, n = 0),
  columns int8/fp16 and columns with a sparse tail. Raw dtypes covered: bf16, f16, f32, i8, i32, u8,
  bool and 0-d.

## Tools

- `suc1-inspect FILE [--no-verify] [--codebook-hashes] [--tensors] [--filter S]`: prints the header fields,
  flags, architecture and tokenizer ids, the footer SHA-256 and its verification status (with timing),
  per-(representation, residual, rotation) counts, params, bytes and bpw, and the whole-file byte
  accounting. The accounting matches `SucModel.bits()` and sums exactly to the file size. With
  `--tensors` it adds one line per tensor: shape, index bits, bytes, bpw of its own arrays, and bpw
  including the codebooks it owns (each deduplicated codebook is charged to the first tensor that uses it,
  in canonical order).
- `suc1-decode FILE (--all | --tensor NAME ...) [--out DIR] [--format npy|raw] [--digest] [--threads N]`:
  decodes to float32 `.npy` (or raw `.f32`). `--digest` prints the SHA-256 of each decoded buffer, which
  allows comparison without writing gigabytes. The result never depends on `--threads`. Every SUC1
  representation is decoded (raw, bf16, affine, scalar, vq, additive, tcq, native `mxfp4`: E2M1 low nibble first,
  E8M0 group scales, code 8 = +0, scale 255 rejected), bit-exact with `SucModel.decode_tensor`
  (tests/unit/test_cpp_decoder.py); the codec capability table is `suc/capabilities.py`.
- `suc1-server`: numeric request fields are validated before any narrowing conversion (HTTP 400 naming the field;
  `cpp/include/suc/request_params.hpp`); an error during a streamed generation is sent as an SSE `error` event
  without a finish_reason, which the eval runner reports as ERROR.

## Library

- `suc/suc1.hpp`, `Suc1File`: memory-maps the file read-only and validates the header and footer (the
  rules of format.md §9). Verifies the SHA-256 unless disabled; it can also verify each codebook's dedup
  hash. Parses the canonical-JSON directory with the bundled minimal parser (`suc/json.hpp`), which keeps
  int64 values exact. Validates the layout: 64-byte alignment, payloads inside the payload region and not
  overlapping, `nbytes = prod(shape) · itemsize`, packed shape `[rows, ceil(n·b/32)]`, `data_offset`,
  `compressed_size`, canonical tensor order, codebook ids, and `global_flags` consistent with the
  descriptors. Exposes `TensorDesc` / `CodebookDesc` / `ArrayRecord` and zero-copy `std::span` payload views.
- `suc/decode.hpp`, `decode_tensor`: follows the operation order of format.md §10, element for element:
  - raw: any stored dtype cast to f32 (bf16 by bit shift);
  - bf16;
  - affine: `q·s + b`, two roundings;
  - scalar: `C_g[i] · s_r`;
  - vq / additive: `acc = C_0[i_0]`, then `acc + C_m[i_m]` for each m in order, then `· s_r`;
  - rotation: `rht_inverse` on rows (seed_in), then on columns (seed_out);
  - residual: columns first (`colval · colscale`, then add), then the sparse tail (`val · res_scale`, then add).
- `suc/hadamard.hpp`: the transforms of `suc.quant.hadamard` with its float32 operation order.
  - The 2^k factor is the in-place FWHT with butterfly distances 1, 2, 4, …, scaled once by
    `float(1/sqrt(2^k))`. This matches the MLX Metal `hadamard_n` kernel and the MLX CPU path. Above 8192
    it uses the `_pow2` Kronecker split.
  - The `m ∈ {12, 20, 28}` factor uses MLX's ±1 tables scaled by `float(1/sqrt(m))`, applied as a
    sequential fma chain over k (the definition `suc.quant.hadamard._mix` uses too).
- `suc/numpy_random.hpp`: Hadamard signs are not stored. They are re-derived from `seed_in` / `seed_out`
  with a bit-exact reimplementation of `np.random.default_rng(seed).integers(0, 2, n)`: SeedSequence
  (pool of 4 words, hashmix/mix), PCG64 (128-bit LCG, XSL-RR output), and the buffered 32-bit Lemire draw.
- `suc/sha256.hpp`: portable SHA-256, plus the ARMv8 SHA-2 instructions when available (always on Apple Silicon).

## Results (M3 laptop, 2026-10-01)

Dev container `compress_qwen3_0.6b/container/model.suc1` (320 MB: 310 tensors, 784 codebooks; 196 additive
M=4 K=256 G=8 tensors with an int8 sparse residual and in_out rotation, 1 affine 4-bit embedding, 113 raw
tensors):

- All 310 tensors are bit-identical to `SucModel.decode_tensor`, compared by SHA-256 of the float32 buffers.
- Open (header, directory, layout validation): ≈ 5 ms. The full SHA-256 check takes 0.16 s warm (≈ 2 GB/s)
  and 0.6–0.7 s cold.
- Decoding all 596 M elements to float32 takes 1.43 s on 8 threads and 4.3 s on 1 thread. The Python
  reference decoder takes ≈ 9.9 s for load + decode.

## Known deviations / findings

1. **Resolved: the Hadamard factors m = 20 and m = 28 are now bit-exact.** The Python reference used to apply
   `H_m` with `mx.matmul`, whose accumulation order depends on the device and shape (for m = 20 / 28 GPU, CPU and
   the fma chain differed by up to 2.1e-7 relative). `suc.quant.hadamard._mix` now defines the step as the same
   sequential fma chain as `hadamard_transform` here (`o = 0; for k: o = fma(H[i, k], y[k, c], o)`): a Metal
   kernel on the GPU, numpy (float64 round-to-odd, then float32) on the CPU. MLX GPU, MLX CPU and C++ agree bit
   for bit, including 5120 = 20·256 and 7168 = 28·256 (`vq_rot_5120`, `vq_rot_7168` in `test_bit_exact_every_codec`).
   m = 12 results are unchanged (the matmul already matched the chain for the Qwen3 sizes), so existing
   containers still decode bit-exactly. The runtime rotation kernels (`suc.kernels.rotation`) only match within
   tolerance: on the M3 the "row" layout is bit-exact, "blocks" differs by ≤ 2.3e-7 relative.
2. **Resolved: output-only rotation** (`rotate_in = False, rotate_out = True`). The Python reference decoder
   used to call `unrotate_weight(x, seed_in=None, seed_out)` (random signs) and the writer labelled such tensors
   `rotation = "in_out"`. `suc/io/suc1.py` now mirrors `_finish` in `suc.quant.codecs`, decoding from the params,
   so files with the old label still decode correctly. The writer emits the new label `rotation = "out"`
   (format.md §5.2). The C++ decoder already used the params; `test_out_only_rotation_matches_codec` checks
   C++ == SucModel == codec bit for bit.
3. CMake: `CMakeLists.txt` mirrors `build.sh` (including the Metal targets), but it was not run on the dev
   laptop (CMake is not installed there). The verified build is `build.sh`.

## Milestone 2: Metal execution of one SUC linear layer

`y = x Ŵᵀ` for one vq / additive tensor of a SUC1 file, on the GPU, with the representation read **in place** from
the mapped file. Library `libsuc_metal.a` (`suc/metal.hpp`, `src/metal/metal.cpp`), Apple's metal-cpp (vendored
unmodified under `third_party/metal-cpp`, Apache-2.0).

### Design

- **Kernels = the Python/MLX kernels.** `metal/suc_kernels.metal` is *generated* by `metal/gen_suc_kernels.py`
  from the Python sources: the helper headers and kernel bodies are verbatim copies of
  `suc.kernels.vq_matmul._HEADER / _LOOKUP_SRC / _MMA_REG_SRC`, `suc.kernels.rotation._HEADER` plus the body that
  `rotation._blocks_kernel(1, False)` builds, and `suc.runtime.compressed_model._STRUCT_HEADER / _STRUCT_SRC`. Only
  the signatures are written by hand, in the style `mx.fast.metal_kernel` generates them (template parameters in
  the same order, the same attribute names). `test_kernel_source_in_sync` fails when a Python kernel changes and
  the file was not regenerated: `.venv/bin/python cpp/metal/gen_suc_kernels.py`, then rebuild.
- **Run-time compilation.** The Command Line Tools ship no offline `metal` compiler, so the build embeds the
  `.metal` text as a string (`build.sh` / CMake generate `gen/suc_kernels_msl.cpp`), and the `Context` compiles it
  with `newLibraryWithSource`. Each template instantiation is one `template [[host_name(..)]] [[kernel]]
  decltype(k<..>) k<..>;` line, as in MLX, compiled once and cached (≈ 5–60 ms each; the OS shader cache makes
  repeats cheap). Compile options are those of MLX: fast math off, the OS default language version (the MMA body
  uses a lambda, which MSL 3.1 rejects). `$SUC_METAL_SOURCE=path.metal` overrides the embedded source.
- **Zero-copy mapping.** `MappedFile` wraps the whole `mmap` of the file (page-aligned base) in
  `newBufferWithBytesNoCopy` buffers. The length is rounded up to the page size (16 KB; `mmap` maps whole pages,
  so the tail of the last page is addressable). Above `maxBufferLength` the file is covered by several views cut
  only between *mapping units* (`mapping_units`: per transformer layer and payload section the run of that layer's
  segments, so expert stacks and fused projections addressed as one range stay in one view; plus each layer's codebook
  span), sharing at most one page. Residency is charged per view: a command buffer that binds a view makes all of it
  resident, and the GPU working set (`recommendedMaxWorkingSetSize`, ~ `iogpu.wired_limit_mb`) is charged for every
  view even where two share pages. The earlier split into half-overlapping views charged ~1.4x the file
  (DeepSeek-V4-Flash: 86.6 + 64.1 GB for 107.4 GB on the M1 Ultra, above the 115.3 GB working set) and the first
  command buffer waited in `waitUntilCompleted` forever. `MappedFile` now warns when the views exceed the working
  set; `--max-view-bytes N` caps the views (tests). SUC1 payloads are
  64-byte aligned, so every array is just (buffer, byte offset). Read in place: packed indices, f16 row scales
  (`float(half)` is exact, so reading them directly equals the float32 copy `SucLinear` holds), residual values,
  column values, and the codebooks when the M segments of a tensor are adjacent in the file (always for files
  written by `write_suc1`). Otherwise the ≤ 16 KB codebooks are packed into one small buffer.
- **Execution layout built at load (small arrays only).** Hadamard signs (from the seeds, numpy-compatible RNG),
  CSR row pointers and u16/u32 column ids from the bit-packed positions, unpacked column ids, float32 column scales,
  segment meta: a few tens of KB per tensor (see Results).
- **Execution plan** (`Linear`, = `SucLinear` / `SucResidualLinear` with `FUSE_RESIDUAL`), in one compute
  encoder (concurrent dispatch, memory barriers between dependent stages):
  1. `x̃ = V x`: rotation kernel, rounded to the activation dtype (only with `rotate_in`); "blocks" layout, or
     "row" (`rotation._kernel`, one threadgroup per row) for the sizes where the tuning cache selects it (on the M1
     Ultra n = 1024; before 2026-10 the port ran "blocks" there and was not bit-identical to Python on that machine).
  2. `ỹ = W' x̃`: lookup kernel per batch tile of ≤ 16 rows. The "vec" kernel runs below `mma_min_batch`, the
     simdgroup-matrix kernel ("reg" fragments) from it. Output is float32 when a later stage follows, else the
     activation dtype.
  3. `y = Uᵀ ỹ`: inverse rotation (only with `rotate_out`).
  4. `+ R x`: structured residual kernel (column part and/or CSR tail, one launch, rounding inside the kernel).
- **Tuning = the Python tuning.** `codebook_in`, `mma_min_batch`, the rotation layout and block shape come from the same
  per-device cache (`$SUC_TUNING_CACHE` or `~/.cache/suc/kernel_tuning.json`, key `architecture|device name`), with
  the same nearest-M rule and the same fallbacks (`applegpu_g13*` → threadgroup codebooks, MMA from B = 5,
  `default_block_config`). Every module constant (`LOOKUP_SG`, `MMA_RT`, …) is a `Tuning` field.
- **Activation dtypes**: f16, bf16 (MSL `bfloat`), f32.

Not ported (not used by the default Python configuration): the K1 / "table" strategies, `MMA_KIND = "tg"`,
`LOOKUP_HALF_MATH`, the element-wise prologues (RMSNorm / SwiGLU + rotation), the
fused multi-segment projections (`SucFusedLinear`, `SucFusedResidual`) and `suc_switch` (MoE). Those belong to
milestone 3. Rotation sizes whose 2^k factor is < 32 (Python: `mx.hadamard_transform` fallback) use a simple dense
per-element kernel (`suc_rot_dense`, C++ only).

### Tools

- `suc1-linear FILE (--all | --tensor NAME ...) [--batches 1,4,16] [--dtype f16|bf16|f32] [--path auto|vec|mma]
  [--json] [--x X.npy --y-out Y.npy] ...`: for random x (normal, rounded to the dtype), runs each supported tensor on
  the GPU. It compares with the CPU reference: `decode_tensor` → Ŵ float32, then `x Ŵᵀ` in float64. Reported:
  max abs, mean abs, relative L2, max abs / max |ref|, and the relative L2 of rounding the exact result itself to the
  dtype (the floor any f16 / bf16 output has). `--x/--y-out` runs one tensor on a given x and writes y; the
  pytest uses this for the bit-exact comparison with MLX. Tensors without a GPU kernel (affine, scalar, bf16, raw)
  are skipped.
- `suc-linear-bench [--shape 12288x4096] [--batches 1,4,16] [--gb 2] [--min-mb 512] [--samples 5] [--json OUT]`:
  the C++ counterpart of the "layers" mode of `benchmarks/k3_cold_cache.py`: same methods (`vq_G8_K256`,
  `vq_G4_K256`, `add2_G8_K256`, `add3_G8_K256` × `_kernel`, the A/B path `_vec` / `_mma` at B ≥ 4, `_full` with
  rotations), the same random representations and copy counts (N distinct copies so a cycle streams ≥ max(gb FP16
  bytes, min_mb) from DRAM), and the same timing rule (one command buffer = N independent calls, encode +
  execution, median / min per call over the samples). It also reports the GPU-only time.

### Tests

- `suc-metal-tests` (C++, needs a GPU; 696 checks): lookup kernels on synthetic host layers against float64
  CPU references:
  - index widths 1/4/5/6/8/10, aligned and unaligned;
  - M = 1–4, G = 2/3/4/8 (odd G takes the natural-order MMA path);
  - tail rows; B = 1, 3, 4, 5, 8, 16, 17, 33 (tiling); paths auto / vec / mma; f32 / f16 / bf16;
  - rotations in / out with sizes 2^k, 12·2^k, 2^14 and the dense fallback (448, 40, 24, 2);
  - residuals CSR int8 / fp16, columns int8 / fp16, both, `out % 4 ≠ 0`;
  - f16 / bf16 round-to-nearest-even conversion.
- `tests/unit/test_cpp_metal.py` (skipped when not built):
  - every vq / additive variant written with `write_suc1` (24 tensors) vs the CPU reference at B = 1/3/4/5/16/17,
    for f32, f16 and bf16;
  - **bit-exact vs MLX** `module_from_encoded` (f16, B = 1/4/16) for every tensor whose rotation sizes Python runs
    with its own kernel; the dense-fallback sizes are checked to 5e-3;
  - the multi-view mapping (forced with `--max-view-bytes`: same results as one view);
  - the kernel-source sync check, and the C++ unit tests;
  - `-m model`: all 196 SUC tensors of the 0.6B container vs the reference, plus 4 tensors bit-exact vs MLX.

### Results (M3 laptop, 2026-10-02)

Numeric agreement with the CPU reference (relative L2 of `y`, worst over tensors and B = 1/4/16; "floor" = the
relative L2 of rounding the exact result to the dtype):

| file / dtype | f32 | f16 (floor) | bf16 (floor) |
|---|---:|---:|---:|
| 0.6B container, 196 tensors (additive M=4 K=256 G=8, int8 CSR residual, in_out rotation 1024/2048/3072) | 1.0e-6 | 3.1e-4 (2.3e-4) | 2.7e-3 (1.8e-3) |
| synthetic, 24 codec variants (worst variant) | 2.2e-6 | 3.6e-4 | 3.3e-3 |

- f32: vec path ≈ 2e-7, MMA path ≤ 1e-6 (float32 simdgroup MMA). f16 / bf16 sit at ~1.3× the floor: x̃ and y are
  each rounded once to the dtype, as in the Python runtime.
- **Bit-exact with MLX**: every output bit equals the Python `SucLinear` / `SucResidualLinear`. This holds for f16
  on all synthetic variants except the dense-fallback sizes, and for f16 / bf16 / f32 on the dev-container
  tensors that were checked.
- Zero-copy: for a 0.6B tensor (3072×1024, M = 4), 1.61 MB are read in place from the mapping, and 52 KB are
  built at load (signs, CSR row pointers and u16 columns).

Indicative timing, 12288×4096 (8B gate/up), f16 x, µs per call, cold-cache cycles. Measured on the M3 laptop
while other agents' jobs shared the GPU. Two interleaved C++ / MLX runs, best of the two medians; the Mac Studio
numbers are the reference:

| method | B=1 C++ | B=1 MLX | B=4 C++ | B=4 MLX | B=16 C++ | B=16 MLX |
|---|---:|---:|---:|---:|---:|---:|
| vq_G8_K256_kernel (1.0 bpw) | 127 | 126 | 302 | 306 | 620 | 630 |
| vq_G4_K256_kernel (2.0 bpw) | 154 | 157 | 309 | 306 | 584 | 594 |
| add2_G8_K256_kernel (2.0 bpw) | 479 | 495 | 553 | 622 | 841 | 984 |
| add3_G8_K256_kernel (3.0 bpw) | 341 | 386 | 604 | 582 | 1128 | 1144 |
| vq_G8_K256_full (+ rotations) | 144 | 147 | 325 | 325 | 744 | 674 |
| add3_G8_K256_full | 403 | 403 | 651 | 678 | 1188 | 1175 |

Reading:
- The kernels are the same, and so are the times (within the ±10–30 % noise of a shared GPU).
- The C++ host side adds nothing measurable at these sizes.
- `_full` at B = 16 is ≈ 10 % slower in C++ for vq. A plausible cause is the dispatch order: C++ encodes rotation →
  kernel → rotation per copy, with a barrier at each step, while MLX may batch the N independent rotations first.
  This is not verified.
- The forced vec path at B = 16 (not taken by auto) was 2–3× slower under MLX in both runs. This is unexplained
  and possibly thermal: those rows run last in the Python script.

Raw outputs: run `suc-linear-bench` and `benchmarks/k3_cold_cache.py` (commands below); the laptop JSONs were not
committed (timings are only reportable from the Studio).

### Mac Studio commands (benchmark of record)

```bash
cd ~/Projects/SUC-LM
cpp/build.sh ~/suc-local/cpp-build
~/suc-local/cpp-build/suc-metal-tests
.venv/bin/python -m pytest tests/unit/test_cpp_metal.py tests/unit/test_cpp_decoder.py -q
mkdir -p logs && TS=$(date -u +%Y%m%dT%H%M%SZ) && nohup sh -c "
  for s in 12288x4096 4096x12288 4096x4096; do
    ~/suc-local/cpp-build/suc-linear-bench --shape \$s --batches 1,4,16 --gb 16 --min-mb 2048 --samples 7 \
      --json experiments/results/${TS}_cpp_metal_linear_\$s.json; done
  HF_HUB_OFFLINE=1 .venv/bin/python benchmarks/k3_cold_cache.py --shapes 8b --methods _K256 --no-decode \
    --gb 16 --min-mb 2048 --samples 7 --json experiments/results/${TS}_k3_cold_cache_suc_8b.json" \
  > logs/m2_bench_${TS}.log 2>&1 &
```

Sequential, never in parallel with other benchmarks. Same `--gb / --min-mb / --samples` on both sides. Both read
the same tuning cache, so the C++ runs use the same `codebook_in` / `mma_min_batch` as MLX: threadgroup codebooks
on the M1 Ultra. Compare the `time_us` of equal (`method`, `batch`, shape) rows.

## Milestone 3: Qwen3 dense model graph and greedy generation

A Qwen3 dense decoder (`suc/model.hpp`, `src/metal/model.cpp`) on the milestone-2 backend, built from a SUC1 file
alone, and the tool `suc1-generate` (token ids in, greedy token ids out).

### Design

- **Model description** from `directory.architecture` (the HF `config.json`): hidden / intermediate size, layers,
  heads, KV heads, head_dim, vocab, `rms_norm_eps`, `rope_theta`, `tie_word_embeddings`, `torch_dtype` (the activation
  dtype, bf16 for Qwen3). Unsupported settings fail loudly: MoE (`qwen3_moe`; added in milestone 4), `rope_scaling`, attention bias,
  sliding windows, `hidden_act` other than silu, head_dim not a multiple of 32. Tensor roles come from a per-family
  name table (`NameTable`, checkpoint names as in `suc.models.adapters`).
- **Weights stay in the mapping.** SUC projections are `metal::Linear` (vq / additive, rotations, residuals). The
  embedding / LM head (affine 4-bit in the current containers) and non-SUC projections (affine, raw / bf16) are read
  in place by a gather and a matvec kernel. Norm weights are read in place when stored in the activation dtype, else
  converted once (a few KB). Nothing is decoded to dense weights.
- **Ops = the Python runtime's ops, in MLX's operation order** (`metal/suc_model_kernels.metal`, hand-written; appended
  to the generated `suc_kernels.metal` at build time):

  | op | Python runtime | C++ kernel | equal bits |
  |---|---|---|---|
  | embedding | decoded weight (`q·s`, `+ b` in float32) cast to bf16, row gather | `suc_embed_affine` / `_dense` | yes |
  | RMSNorm, q/k-norm | `mx.fast.rms_norm` (`rms_single_row` order, N ≤ 4096) | `suc_rms_norm` | yes |
  | RoPE | `mx.fast.rope` (`rope_single`, non-traditional, `std::log2(base)`) | `suc_rope` | yes |
  | attention | `mx.fast.scaled_dot_product_attention` → MLX `sdpa_vector` (1 pass) | `suc_sdpa_vector` (MLX source, mask / sinks removed) | yes |
  | SwiGLU | `mlx_lm` `swiglu` (compiled `silu(g)·u`, ops in bf16) | `suc_swiglu` | yes (bf16, f32; not f16) |
  | residual add | `+` in bf16 | `suc_add` | yes |
  | q / k / v, o, gate / up, down | `SucLinear` / `SucResidualLinear` (one module per projection) | `metal::Linear` stages | yes |
  | RMSNorm + shared input rotation | `rotation.rms_norm_rht` (K6, per-device `fuse_elementwise("norm")`) | `suc_rot_prologue_blocks` (generated) | yes |
  | LM head | `embed_tokens.as_linear` = MLX matmul of the decoded bf16 weight | `suc_matvec_affine` / `_dense` (same per-element weights, float32 sum) | no: summation order |
  | argmax | `mx.argmax` (first index of the maximum) | `suc_argmax` | yes |

  The RoPE and SwiGLU kernels were checked bit-exact against MLX on 1M random bf16 values before porting
  (SwiGLU in float16 is not: 3.6 % of the outputs differ; only bf16 models are in scope).
- **Execution**: one command buffer per decode step, one concurrent compute encoder, a memory barrier on every true
  dependency, no host sync inside a step. A preallocated activation arena (sized for `--chunk` rows) and KV cache
  (`[position][kv head][head_dim]` per layer, so the v projection and the RoPE of k write their rows straight into
  the cache; `sdpa_vector` takes the strides). q|k|v and gate|up share ONE input rotation when their `seed_in` is
  equal (files of the compress pipeline: always) and run stage by stage (lookup kernels together, then the inverse
  rotations, then the residuals); with the tuned `fuse_elementwise("norm")` choice (as the Python runtime) the norm
  and that rotation are one launch. Both are A/B switches (`--no-shared-rotations`, `--fuse-norm auto|on|off`) and
  do not change a bit.
- **Prefill**: `--prefill batch` (default) runs the prompt in chunks of `--chunk` (64) tokens per command buffer: the
  projections at batch n (the "mma" lookup path from `mma_min_batch` rows), causal attention by `sdpa_vector` with
  one threadgroup per (head, query row) and `n_keys = N − n + t + 1` (MLX's `do_causal` rule). `--prefill token`
  runs the prompt as decode steps, i.e. exactly what a token-by-token Python run computes.
  The batched prefill is **not** bit-identical to the decode steps, by design (it keeps the batch kernels; the greedy
  tokens after a 2,048-token prompt first differ after a few steps on Qwen3-8B / 30B-A3B). Diagnosis (Qwen3-8B final,
  M1 Ultra, `prefill_logits_sha256` of the last prompt token now in `--json`): exactly three sources, each confirmed by
  removing it: (1) the simdgroup-matrix ("mma") lookup of tiles from `mma_min_batch` rows (`$SUC_MMA_MIN_BATCH=100000`
  keeps every tile on the vec path); (2) the structured residual's reduction split JS, which shrinks with the rows per
  threadgroup (`$SUC_RESIDUAL_EXACT=1` keeps the one-row JS); (3) attention: a decode step from 1,024 keys runs the
  two-pass kernel on 'd' devices, a batch row the single-pass one. On a 900-token prompt (1)+(2) removed gives the
  token-prefill hash; on 2,048 tokens it equals `--prefill token` with `SUC_SDPA_2PASS=off`. Removing them costs the
  speed-up (2,048 tokens: 6.7 s batched, 19.8 s with (1)+(2) removed, 27.2 s token by token), so the difference is
  accepted and measured instead: `scripts/prefill_batch_vs_token.py` (KL / top-1 per position vs token prefill;
  report §1c).
- **Python reference** (`suc/runtime/suc1_model.py`): `load_suc1_model(path, group=False)` builds the mlx-lm model of
  the file's architecture without any checkpoint: raw tensors as stored, other non-SUC tensors as their reference
  decode cast to the model dtype (= the representation the compress pipeline evaluates), SUC tensors through
  `convert_to_compressed` on the stored codes. `group=False` is the C++ execution plan (one `SucLinear` per
  projection); `group=True, fuse=True` is the default (fused) Python runtime. `step_hidden` / `run_tokens` /
  `greedy` run mlx-lm's decode step (mask None, KV cache) and return the per-layer hidden states.

### Tool

`suc1-generate FILE (--tokens 1,2,3 | --tokens-file F) [--steps N] [--prefill batch|token] [--chunk N]
[--logits-out L.npy] [--hidden-out H.npy] [--nll-out N.npy] [--force 4,5,6] [--json OUT.json] [--stop-eos]
[--max-context N] [--dtype bf16|f16|f32] [--no-shared-rotations] [--fuse-norm auto|on|off] [--tuning-cache P|-]`

Prints `generated: ids`, then (stderr) load / compile time, prefill and decode ms/token (wall, GPU time of the
command buffer, host encode time) and memory: mapped file (and how much of it is resident in the page cache,
`mincore`), execution arrays built at load, KV cache, arena, workspace, process footprint / resident size (the GPU's
reads of the mapping do not show up in the process resident size). `--logits-out` / `--hidden-out` write every
position's logits `(positions, vocab)` and hidden states `(positions, layers + 1, hidden)`; `--nll-out` the NLL of each
next prompt token (`--steps 0`: scoring only); `--force` feeds given tokens instead of the argmax (teacher forcing).

### Tests

`tests/unit/test_cpp_generate.py` (skipped when `suc1-generate` is not built):
- synthetic Qwen3 files written with `write_suc1` (affine 4-bit embedding; per role a different codec: additive
  M = 2 / 3, vq K = 16 / 64 / 256, G = 4 / 8, in_out and one-sided rotations, int8 / fp16 CSR and int8 column
  residuals), tied and untied (raw bf16 LM head), with the norm fusion and the shared rotations on / off, and one
  block at Qwen3-8B widths (hidden 4096: 1024-thread RMSNorm and the 4096 prologue; intermediate 3072 = 12·256:
  H_12): every hidden state of every position bit-identical to the Python runtime (`group=False`), logits KL
  < 1e-4, greedy agreement 1.0;
- batch prefill (chunks 3 and 64) vs token prefill: first position identical, relative L2 of the hidden states
  < 2e-2, KL < 1e-3, same greedy tokens; teacher forcing; out-of-vocabulary ids rejected;
- `-m model`: the dev container, 13-token prompt + 64 greedy tokens: all 77 × 29 hidden states bit-identical, and the
  free-running Python greedy sequence equals the C++ one.

`benchmarks/cpp_generate_check.py` (not a test) produces the numbers below: three fixed prompts, teacher-forced and
free-running comparisons against both Python runtimes, decode / prefill timing, WikiText-2 perplexity.

### Results (M3 laptop, 2026-10-02)

Dev container `compress_qwen3_0.6b/container/model.suc1` (196 additive M=4 K=256 G=8 projections with int8 CSR
residuals and in_out rotations, affine 4-bit tied embedding / LM head). Activations bf16; `--prefill token` for the
agreement rows (the decode step for every position); Python = `load_suc1_model(group=False)` unless "default".

| prompt (tokens) | positions | hidden states bit-identical | logits bit-identical | KL mean / max | top-1 (teacher-forced) | free greedy, 64 tokens |
|---|---:|---:|---:|---:|---:|---:|
| history (13) | 76 | 100 % (76 × 29) | 99.988 % | 1.4e-9 / 1.0e-7 | 1.000 | 64 / 64 identical |
| code (4) | 67 | 100 % | 99.986 % | 1.1e-10 / 7.2e-9 | 1.000 | 64 / 64 |
| short (5) | 68 | 100 % | 99.982 % | 2.6e-10 / 1.6e-8 | 1.000 | 64 / 64 |

- A longer free run (code prompt, 400 greedy tokens, 404 positions): 400 / 400 tokens identical to the Python plan
  runtime.
- The only differing values are logits (max |Δ| 0.0625, one bf16 ulp for logits of magnitude 8 – 16), from the
  LM-head summation order (deviation 1).
- Against the **default (fused) Python runtime**: KL mean 5.9e-4 – 1.0e-3 (max 1.0e-2), teacher-forced top-1 1.000,
  free greedy 64 / 64 identical on all three prompts. Its fused projections (and, where tuned on, the SwiGLU
  prologue) round differently from the per-projection plan, so this is the expected bf16 rounding spread between two
  summation plans, not a C++ error.
- **Perplexity** (WikiText-2 test, 4 windows × 512 tokens, C++ batch prefill with chunks of 128 vs a whole-window
  mlx-lm call): C++ 33.36, Python plan runtime 33.31, Python default runtime 33.33 (+0.15 % / +0.11 %); mean |ΔNLL|
  0.036 nats per token, the size of a bf16 rounding of the logits. Batch prefill vs token prefill in C++ (260
  tokens): KL mean 1.2e-4, top-1 1.000.

Timing (indicative: M3 laptop, other agents' jobs on the same GPU; B = 1, synchronous per token, medians):

| | C++ `suc1-generate` | Python plan runtime | Python default runtime |
|---|---:|---:|---:|
| decode ms/token (13-token prompt, 64 tokens) | 9.4 (GPU 8.2, host encode 0.7) | 16.1 | 14.4 |
| prefill ms/token (256-token prompt) | 1.05 (batch, chunks of 64) | 1.10 | 1.08 |

- C++ decode is 1.5 – 1.7× faster with the same kernels. The likely cause is the per-op graph building and launch
  overhead of the Python step (not profiled); the C++ step is one command buffer encoded in ≈ 0.7 ms. Prefill time
  is the same on both sides.
- Shared input rotations save ≈ 1.2 ms GPU per token (A/B, 4 interleaved runs, best GPU time: 8.4 vs 9.6 ms); the
  fused norm + rotation launch is neutral on the M3 (best 8.40 vs 8.44 ms). Prefill chunk 16 / 64 / 128: 1.49 / 1.11 /
  1.06 ms/token; token-by-token prefill 12 ms/token.

Memory (decode, 79-position context): mapped file 320.4 MB (all of it in the page cache, `mincore`);
execution arrays built at load 8.3 MB (Hadamard signs, CSR row pointers and u16 columns); KV cache 9.1 MB
(115 KB per position); arena 6.9 MB (64-row chunk); workspace 0.7 MB. Process footprint 94 MB, resident 57 MB: GPU
reads of the no-copy mapping are not counted in the process resident size, so the honest total is mapping + footprint
≈ 414 MB. The Python runtime holds the decoded bf16 embedding (311 MB) in addition to the codes.

### Known deviations / limits

1. **LM head** (closed in milestone 4: the LM head now runs MLX's gemv order, logits bit-identical; `--legacy-matvec`
   restores this behaviour): the C++ matvec sums each row in a different order than MLX's matmul of the decoded bf16 weight
   (99.98 % of the logits bit-identical, KL ≤ 1e-7). Porting MLX's `gemv` tiling would close this; MLX picks the
   tiling in host code that is not shipped with the wheel.
2. **Two-pass attention is not ported** (ported in milestone 4, verified on the M3; see there). MLX switches to `sdpa_vector_2pass` from 1024 keys on devices whose
   architecture ends in `d` / `s` (the M1 Ultra: `applegpu_g13d`), and from 4096 keys with GQA everywhere. The C++
   always runs the single-pass kernel, so beyond those lengths results agree only to rounding. Keep validation runs
   on the Mac Studio below 1024 positions.
3. **Batch prefill ≠ the Python prefill bit for bit**: mlx-lm prefills with the steel attention kernel and a causal
   mask; C++ uses `sdpa_vector` per query row. Projections, norms and element-wise ops are row-wise identical.
4. RMSNorm above 4096 (MLX `rms_looped`, e.g. hidden 5120) follows MLX's looped order from its source structure but
   is not verified bit-exact.
5. Non-SUC projections (affine / raw kept by a plan) run on the C++ matvec kernels: correct, not bit-identical to
   MLX's `QuantizedLinear` / `nn.Linear`.
6. Results depend on the tuning cache exactly as in Python (rotation block shape, `codebook_in`, `mma_min_batch`,
   `fuse_elementwise`): both read `~/.cache/suc/kernel_tuning.json`, so they agree when run on the same machine with
   the same cache. The C++ never tunes: on a new machine, load the file once in Python first (convert_to_compressed
   runs the autotune). On the laptop, a C++ run made before the Python autotune had written its entries produced a
   different greedy continuation from the 7th token, most likely through the default rotation block shape.
7. CMake mirrors `build.sh` (new sources, the second embedded kernel file, `suc1-generate`) but was not run (no CMake
   on the dev laptop).

### Mac Studio commands (Qwen3-8B)

```bash
cd ~/Projects/SUC-LM
cpp/build.sh ~/suc-local/cpp-build
.venv/bin/python -m pytest tests/unit/test_cpp_generate.py tests/unit/test_cpp_metal.py -q
F=experiments/artifacts/compress/compress_qwen3_8b/container/model.suc1
~/suc-local/cpp-build/suc1-generate $F --tokens 785,3840,315,279,18484,3493,12033,304,279,11128,45387,9294,11 --steps 64
mkdir -p logs && TS=$(date -u +%Y%m%dT%H%M%SZ) && HF_HUB_OFFLINE=1 nohup .venv/bin/python \
  benchmarks/cpp_generate_check.py --file $F --steps 64 --ppl-windows 8 --ppl-seq-len 512 \
  --out-dir experiments/results > logs/m3_check_8b_${TS}.log 2>&1 &
```

Sequential, never in parallel with other GPU jobs. The 8B file is untied: its `lm_head.weight` runs on the affine or
dense matvec, whichever the plan stored. All positions stay below 1024 (deviation 2). The Python side loads two
copies of the model (plan and default runtime, ≈ 2 × the file size).

## Milestone 4: fused plan, tokenizer, MoE, pipelined decode

### Fused projection groups (the default Python runtime's plan)

The default Python runtime (`convert_to_compressed(group=True, fuse=True)`, i.e. `load_suc1_model(path, group=True,
fuse=True)`) wraps q|k|v and gate|up into a `SucInputGroup`: one shared input rotation, and members with the same code
layout row-concatenated into ONE `SucFusedLinear` / `SucFusedResidual`. The C++ default is now that plan
(`--projections fused`; `--projections separate` keeps the milestone-3 per-projection plan = `group=False`):

- **Clusters** = `SucInputGroup.__init__`: members with equal `FusedLinear::key` (= `group_fusable_key`: in, bits,
  (M, K, G), seed_in, rotate_out, blocks-layout rotation sizes) form one `FusedLinear` (≥ 2 members); the others run
  alone on the shared x̃. Groups need one common seed_in (files of the compress pipeline: `rotation_seeds`).
- **Stages** (one launch each for the whole group, a barrier between them):
  1. `suc_fused_lookup_vec / _mma`: the vq_matmul bodies with `suc.kernels.suc_linear._PATCHES` (row r reads codebook
     set `SUC_SEG(r)`; segment boundaries SB1..SB3 as template constants; staged codebooks (CBTG) only when every
     boundary is a multiple of the threadgroup's rows, as `fused_matmul_metal`);
  2. `suc_rot_blocks{2,3}[_split]`: `rotation._blocks_kernel(nseg, split)`, the segmented inverse rotation with the
     plan of `rotation._blocks_plan` (block shape from `block_config(max size)`, clipped by the smallest segment) — the
     reason why the fused and separate plans differ by rounding on real shapes;
  3. `suc_structured_residual` with S segments (`SucFusedResidual.residual`: concatenated CSR rows, column blocks,
     one int8 scale per member, `row_quads` = every member's rows % 4 == 0), rounding inside the kernel.
  Without a residual the output rotation writes each member straight to its buffer (split); otherwise the fused
  (B, Σ out) output is handed out as views for one row (q / k / gate / up; v is copied into the V cache by
  `suc_copy_cols`) or by strided copies (`suc_copy_cols`) for batch rows.
- **In place** (default; `--fused-layout copy` for A/B): the Python layout concatenates the members' indices per
  codebook stream ((M, Σ out, words)), which the file does not have. The C++-only kernels `suc_fused_lookup_*_tab`
  (generated: the same bodies, only `packed + (m·OUT + row)·WORDS` and `scales[row]` replaced by a per-(segment,
  stream) offset table) read every member's indices and f16 scales where they lie in the mapping. Codebooks
  (S, M, K, G), output signs and the concatenated residual arrays are built at load (small).
- **SwiGLU + rotation** (`SucMLP`, `rotation.swiglu_rht`): `suc_rot_prologue_blocks` with PRO = 2, used with the fused
  plan when the tuning cache chooses `fuse_elementwise("swiglu")` (`--fuse-swiglu auto|on|off`); the Python plan
  runtime never fuses it. (Off on the M3 by its tuned choice.)

### LM head in MLX's gemv order

The tied / untied LM head (and any non-SUC projection kept affine / raw) now runs `suc_gemv_affine` /
`suc_moe_gemv`: MLX's non-transposed `gemv` accumulation order (`GEMVKernel`, TN = 4; SN = 4 for in ≤ 64, BN = 8 for
in ≥ 16·out, else SN = 32, BN = 1 — determined against MLX 0.32.3 for the MoE router, see below) on the decoded weight
element `T(q·s + b)`. Logits are therefore bit-identical to the Python runtime for one row (milestone-3 deviation 1
closed; `--legacy-matvec` keeps the old kernels for A/B). Batch-prefill rows still differ (MLX runs a gemm there).

### Two-pass decode attention

MLX switches its decode attention from `sdpa_vector` to `sdpa_vector_2pass_1` + `_2` on long contexts; the block
count is chosen in host code that the wheel does not ship. The kernels are now ported (`suc_sdpa_2pass_1`, also the
GQA-specialised `suc_sdpa_2pass_1_gqa`, and `suc_sdpa_2pass_2`: MLX source, masks / sinks removed, `blocks` a template
constant) and `Model::Impl::sdpa2_rule` reproduces the dispatch: two passes when the architecture ends in `d` and
n ≥ 1024 keys, or with GQA and n ≥ 4096; blocks = 64 if the GQA factor ≥ 4 else 32 on other devices, MLX's `d` / `s`
tables otherwise. Found and verified on the M3 by `benchmarks/cpp_sdpa_2pass_probe.py` (C++ candidates vs MLX, every
position's hidden state): GQA factors 1 / 2 / 4 / 8 up to 4160 keys, factor 4 up to 8400 keys, 0 differing positions;
single pass, other block counts and the GQA variant all differ from 4096 keys on. The `d` (M1 Ultra) and `s` branches
are MLX's host code as recalled, **not verified** — run the probe on the Studio. `--single-pass-attention` restores the
milestone-3 behaviour. Batch-prefill rows always use the single-pass kernel.

### Pipelined decode

`Model::generate(first, steps, depth)`: each step's argmax writes the next token into the GPU token buffer that the
next step's embedding gather reads, so the host encodes step t + 1 while the GPU runs step t (up to `--depth` command
buffers in flight, default 2; Metal's hazard tracking orders the shared arena / KV cache between command buffers).
Same kernels, same tokens (checked against `--sync` and the Python greedy). Used by `suc1-generate` unless a per-step
output is requested (`--logits-out`, `--hidden-out`, `--nll-out`, `--force`) or `--sync` is given. This hides the
host encode time (≈ 0.6–0.9 ms per token on the M3) behind the GPU.

Encode path: the tuning answers (`codebook_in`, `mma_min_batch`, rotation block shape) are memoized per M / size (they
were JSON lookups on every dispatch), and the hot call sites (lookup kernels, rotations, residuals, norms, RoPE,
attention, element-wise ops) look their pipeline up by an integer key (`PsoKey`) instead of formatting the template
string per dispatch. Measured A/B (sync decode, 5 runs each): encode 0.58 ms/token before and after — no measurable
change, so the remaining host time is the Metal encoder calls themselves (≈ 1 µs per dispatch). Barriers: the copy of
v into the V cache (fused plan, one row) runs alongside the q / k norms instead of after its own barrier (one barrier
less per layer; ≈ 0.7 % of the decode step, within noise).

### Tokenizer (`suc/tokenizer.hpp`, `src/tokenizer.cpp`, tool `suc-tokenize`)

The Qwen2/Qwen3 `tokenizer.json` in C++ (no dependencies): added / special tokens split out first (leftmost-longest
literal match; lstrip / rstrip / single_word / normalized flags), NFC, the Qwen pre-tokenizer regex as a hand-written
matcher (Oniguruma alternation order), ByteLevel, BPE with HF's merge loop (rank heap; word cache), ByteLevel decode
with Rust's `from_utf8_lossy` semantics. Unsupported settings (other normalizers / regexes / models, dropout,
byte_fallback, …) fail loudly. `resolve_tokenizer_path(tokenizer_id)` maps the SUC1 `tokenizer_id`
(`Org/Name@rev;tokenizer.json:sha256:…`) to the HF cache (`$HF_HUB_CACHE`, `$HF_HOME/hub`, `~/.cache/huggingface/hub`)
and checks the sha256. Unicode tables (`src/unicode_tables.inc`, generated by `tools/gen_unicode_tables.py`, `--check`
detects staleness) take `\p{L}` / `\p{N}` / `\s` from HF `tokenizers`' own regex engine (Unicode 16; Python's
`unicodedata` differs on 4,924 letters and 80 numbers) and NFC from `unicodedata` minus the 109 characters HF's older
NFC data does not normalize. The library has no chat template (the tool's `--chat` renders the single-turn Qwen3
template); invalid UTF-8 input is rejected.

`suc1-generate --prompt "text" | --prompt-file F [--tokenizer tokenizer.json] [--text]`: encodes the prompt, prints
`prompt: ids`, `generated: ids` and `text: <decoded generation>` (the last output, may span lines). `--chat [--system
TEXT] [--no-think]` wraps the prompt in the Qwen3 chat template for one user turn (+ system turn) with the generation
prompt, as `apply_chat_template(..., add_generation_prompt=True, enable_thinking=...)` renders it (ids checked against
transformers); `--stop-eos` then stops at `<|im_end|>`.

### Sampling (`suc/sampling.hpp`, `src/sampling.cpp`)

`--temperature T [--top-k K] [--top-p P] [--min-p P] [--seed S]`: on the host, from the last logits row: top-k, then
top-p (smallest descending prefix reaching P), then min-p, then a draw from `std::mt19937_64(seed)` (float64 softmax of
logits / T). Reproducible per seed, NOT the draws of mlx-lm (MLX's RNG is not reimplemented); temperature 0 or top-k 1
is the greedy argmax. Sampling runs the synchronous decode loop (the logits row is read back every step).

### MoE: qwen3_moe (`suc/switch.hpp`, `src/metal/switch.cpp`, tool `suc1-moe`)

- `ModelConfig` reads `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `decoder_sparse_step`,
  `mlp_only_layers`, `norm_topk_prob`; a layer is sparse when `num_experts > 0`, `(l + 1) % decoder_sparse_step == 0`
  and l ∉ `mlp_only_layers` (mlx-lm `Qwen3MoeDecoderLayer`); its MLP is a plain RMSNorm, then `MoeBlock`.
- `SwitchLinear` = `SucSwitchLinear` "gather" path (BT = 1, one tile per row): per-row-sign rotation
  (`suc_rot_blocks_rows` = `rotation._blocks_kernel(1, False, row_signs=True)`, or the plain launch when all experts
  share seed_in), `suc_switch_lookup_vec / _mma` (`suc_switch._PROLOGUE` + vq_matmul bodies; CBE = 0 when every expert
  uses the same codebooks: the shared expert-stack layout), inverse rotation, `suc_switch_residual` (stacked CSR),
  one rounding. Column-structured residuals (`columns_int8 / columns_fp16`, e.g. the final Qwen3-30B-A3B files) run
  through `suc_switch_structured_residual` (`suc_switch._COLRES_SRC`): per-expert column sets concatenated with a
  column-pointer table (counts may differ per expert), int8 with per-column scales or fp16, plus the optional CSR tail,
  one launch per projection for all selected experts, rounded to the activation dtype inside the kernel (copied at
  load, like the CSR arrays). Expert arrays are read in place: `StackLayout::Table` (C++-only prologue `suc_switch_lookup_*_tab`
  with a per-expert offset table) unless the experts already lie consecutively (`Stacked`, the verbatim kernel).
- `MoeBlock` = mlx-lm `Qwen3MoeSparseMoeBlock` with those experts, no host sync: router `suc_moe_gemv` (MLX gemv order,
  bit-exact), `suc_moe_route` (softmax(precise) = MLX `softmax_single_row`; top-k with `argpartition(g, -k)[..., -k:]`
  semantics = the last k of a stable ascending argsort, i.e. ascending gate order, ties by expert index (measured: 0
  mismatches in 800 rows incl. ties); take_along_axis; `norm_topk_prob` sum sequential in bf16 from 0, as MLX's row
  reduce), `suc_repeat_rows`, gate / up / swiglu / down, `suc_moe_combine` (bf16 products, sequential bf16 sum over k
  from 0 — MLX does not accumulate this reduction in float32).

- **Expert vec lookup defaults (decode, below `mma_min_batch`).** The codebooks are staged in threadgroup memory when
  the device's tuned `codebook_in` for that M is "threadgroup" (the dense lookup's per-device choice from
  `kernel_tuning.json`; built-in fallback `applegpu_g13*` → threadgroup) and they fit in 32 KB; 2 rows per simdgroup
  (Python `lookup_rows(1)` = 4), `LOOKUP_SG` = 8 simdgroups. Neither changes any row's arithmetic: logits are
  byte-identical to the Python layout (CBTG = 0, 4 rows) on every C++ model and in the tiny cross-checks
  ([decode_speed_vs_ds4.md](../docs/decode_speed_vs_ds4.md) §8). Overrides: `SUC_SWITCH_CBTG=0|1`, `SUC_SWITCH_ROWS`,
  `SUC_SWITCH_SG`.

### Tool changes

`suc1-generate` new options: `--projections fused|separate` (default fused), `--fused-layout inplace|copy`,
`--fuse-swiglu auto|on|off`, `--prompt TEXT | --prompt-file F`, `--tokenizer PATH`, `--text`, `--sync`, `--depth N`,
`--legacy-matvec`, `--single-pass-attention`, `--chat [--system TEXT] [--no-think]`, `--temperature / --top-k / --top-p / --min-p / --seed`; the
stderr summary and `--json` report the fused groups / SwiGLU fusion. qwen3_moe files run with
the same tool. `suc-tokenize (--tokenizer F | --suc1 FILE) [--encode-lines F] [--encode-json F] [--decode-json F]
[--text T] [--decode IDS] [--skip-special] [--add-special-tokens] [--json]`. `suc1-moe FILE --layer L --x X.npy
--y-out Y.npy [--stack PROJ --eidx E.npy] [--routing R.npy] [--route-out R.npy] [--layout auto|stacked|table]`: one MoE
block (or one expert stack with given expert ids) on given rows (the pytest cross-check).

### Tests

- `tests/unit/test_cpp_generate.py` (47 tests as of 2026-10-15, 9 of them `-m model`, incl. the long-context two-pass test: GQA
  factors 2 / 4 / 8, 24 positions past MLX's switch point): every comparison asserts bit-identical hidden states AND logits.
  New: fused plan vs the default Python runtime on three synthetic code layouts (q|k|v additive M = 2 with CSR / column
  residuals and gate|up vq with an fp16 residual; residual-free split rotation with q|v fused + k alone; one-sided
  rotations) × norm fusion auto / on / off (incl. the SwiGLU prologue forced on, Python `FUSE_ELEMENTWISE = True`);
  shared seeds with the separate plan; fused batch prefill = separate batch prefill bit for bit; pipelined = `--sync`
  = Python greedy; qwen3_moe (top-2, top-4 with a dense `mlp_only_layers` block, `norm_topk_prob` False) × plan / default
  runtime; MoE batch prefill; `-m model`: the dev container with the fused plan vs the default runtime (77 × 29 hidden
  states, 64 free greedy tokens) and `--prompt` text (ids = transformers, printed text = transformers decode).
- `tests/unit/test_cpp_moe.py` (20, 2026-10-15): SwitchLinear vs `SucSwitchLinear.rows(x, eidx, "gather")` bit for bit (four MoE
  variants incl. a shared expert-stack file written with `suc.quant.stacks`, 3 projections, N = 1/3/8/40 with repeated
  experts, f16 / bf16, layouts auto / stacked / table); MoeBlock vs `layers[L].mlp(x)` (routing ids, scores and outputs
  bit-identical, n = 1/3/32/k; given-routing path); a 30B-A3B-width block (hidden 2048, E = 128, k = 8,
  moe_intermediate 768 = 12·64 with H_12 rotations).
- `tests/unit/test_cpp_tokenizer.py` (23, 2026-10-15): exact ids vs `transformers` (Qwen3-0.6B tokenizer, offline) on 2,891
  WikiText-2 test lines (298,938 tokens), the first 2,000 lines as one document (216,940 tokens), 59 hand-written edge
  cases (NFC / NFD, CJK, jamo, emoji ZWJ / skin tones / flags, Arabic, Devanagari, whitespace runs, tabs, \r\n,
  contractions in both cases, digits, code, embedded special tokens, empty string) and 2,330 random strings: 0
  mismatches, with `add_special_tokens` False / True; decode (with / without special tokens, 300 random id sequences
  incl. partial UTF-8) = transformers; round trip on NFC input; added-token flags on a modified tokenizer.json; table
  staleness; tokenizer_id resolution and sha256 mismatch.
- Also: `--chat` prompt ids = `apply_chat_template` (`-m model`; with / without system turn, enable_thinking on /
  off); sampling on the tiny file: same seed → same tokens, temperature 0 / top-k 1 → greedy, different seeds →
  different draws.
- C++: `suc-tests` 1078 checks (2026-10-13 build; tokenizer pieces: byte map, UTF-8, regex splits, NFC, BPE on a toy file; sampler:
  greedy cases, top-k / top-p / min-p supports, reproducibility, frequencies within 4σ of the softmax);
  `suc-metal-tests` 705 checks (+ MoE gemv / combine / top-k ties).

### Results (M3 laptop, 2026-10-02)

Dev container `compress_qwen3_0.6b/container/model.suc1` (as in milestone 3), `benchmarks/cpp_generate_check.py`
(three prompts, `--prefill token`, 64 greedy tokens, teacher-forced comparison at every position):

| C++ plan | Python runtime | hidden states bit-identical | logits bit-identical | KL max | free greedy (64) |
|---|---|---:|---:|---:|---:|
| `--projections separate` | plan (`group=False`) | 100 % (3 prompts, 67–77 × 29) | 100 % | 0 | 64 / 64 identical |
| fused (default) | default (`group=True, fuse=True`) | 100 % | 100 % | 0 | 64 / 64 identical |

- The two plans differ from each other (C++ fused vs C++ separate on the 13-token prompt + 64 tokens: relative L2 of
  the hidden states 3.7e-3; the segmented rotations use other block shapes), and each C++ plan equals its Python
  counterpart bit for bit, so the comparison discriminates. In-place and copied fused layouts give identical bits.
- Perplexity (WikiText-2 test, 4 × 512 tokens, C++ batch prefill, fused plan): C++ 33.369, Python plan 33.312, Python
  default 33.326 (unchanged from milestone 3: batch rows still differ by the attention / gemm kernels).
- Tokenizer: 0 mismatches against transformers on 2,891 WikiText-2 test lines and the edge-case / random corpora
  (see Tests); `--prompt` on the dev container gives the 13 ids of the milestone-3 prompt. Encode ≈ 10–19 MB/s
  (2.3–4.5 M tokens/s) on one M3 core, vs 3 MB/s for HF `tokenizers` from Python with one thread.
- MoE: every tiny qwen3_moe file (top-2 / top-4 / dense layer / no `norm_topk_prob`, plan and default runtime) and the
  30B-A3B-width block: bit-identical hidden states and logits.

Timing (indicative: M3 laptop, B = 1, medians of 63–127 decode steps; two benchmark runs an hour apart, the second
slower throughout, so ranges):

| | ms/token | GPU ms/token |
|---|---:|---:|
| C++ fused, pipelined (default) | 8.2 – 10.1 | 8.2 – 10.2 |
| C++ fused, `--sync` | 10.8 – 11.8 | 8.2 – 10.4 |
| C++ separate, pipelined | 8.2 – 9.5 | 8.3 – 9.5 |
| C++ separate, `--sync` | 9.5 – 11.0 | 8.2 – 9.6 |
| Python default runtime | 14.4 – 15.8 | |
| Python plan runtime | 14.5 – 16.7 | |

- Pipelined decode removes the host part of the step: wall time = GPU time (sync: GPU + ≈ 1.4–1.6 ms of encode,
  commit and wait). C++ default vs Python default runtime: 1.5–1.75× fewer ms per token with the same kernels.
- The fused plan does **not** reduce the GPU time on the M3 (fused vs separate within ±10 %, the second run 8 %
  slower fused). The M1 Ultra, whose staged-codebook kernels the fusion was designed for, is where it must be measured
  (Studio commands below); no claim from the laptop.
- Memory (decode, 79 positions): mapped file 320.4 MB; execution arrays built at load 17.3 MB (in-place fused
  layout; the copied layout needs 164.7 MB and raised the peak resident size from 194 to 380 MB in an A/B run); KV
  cache 9.1 MB; arena 7.6 MB; workspace 0.6 MB; process footprint 111 MB, resident 74 MB.

MoE, indicative: a synthetic 2-layer file at Qwen3-30B-A3B widths (hidden 2048, 32 / 4 heads of 128, 128 experts,
top-8, moe_intermediate 768; vq K = 256 G = 8 ≈ 1 bpw + 0.5 % int8 residual on attention and up_proj; vocab 4096;
176 MB): C++ 0.73 ms/token pipelined (1.03 ms `--sync`, GPU 0.73 ms), Python runtimes 3.30 ms/token (plan and default
alike): ≈ 0.36 ms per MoE layer in C++. Not a Qwen3-30B-A3B measurement (synthetic weights, no real LM head).

### Known deviations / limits

1. **MoE batch prefill above 32 expert rows**: Python's `SucSwitchLinear.rows` switches to its "tiled" path (host-side
   grouping by expert, then BT-row tiles) and `SwitchGLU` sorts the rows from 64 indices; the C++ always runs the
   sync-free gather path. Measured: one stack at N = 128 matches 99.98–99.99 % of the values (relative L2 7e-9 to
   3e-5); a block at n = 32, k = 2 matches 98.9–99.7 % (relative L2 2–4e-4). Decode (N = top_k rows) is exact.
2. Two-pass attention (milestone-3 deviation 2): ported and verified on the M3; the M1 Ultra's block-count rule is
   unverified until `benchmarks/cpp_sdpa_2pass_probe.py` has run there (keep bit-exact validation below 1024 positions
   until then).
3. The fused plan needs the members' arrays in one mapped view for the in-place layout; otherwise (files above
   `maxBufferLength / 2` per view straddling a view boundary, or host layers) it falls back to the copied layout.
4. RMSNorm + rotation fusion is only used for N ≤ 4096 (the bit-exact MLX order); Python fuses larger sizes too
   (different rounding for hidden 5120+, not a Qwen3-8B / 30B-A3B size).
5. The MoE `mma` branch at BT = 1 (only if a tuning cache sets `mma_min_batch` 1) and per-row rotations with 2^k < 32
   are untested / rejected.

### Mac Studio commands (Qwen3-8B; Qwen3-30B-A3B when its container exists)

```bash
cd ~/Projects/SUC-LM
cpp/build.sh ~/suc-local/cpp-build
~/suc-local/cpp-build/suc-tests && ~/suc-local/cpp-build/suc-metal-tests
.venv/bin/python -m pytest tests/unit/test_cpp_generate.py tests/unit/test_cpp_moe.py tests/unit/test_cpp_metal.py -q
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/unit/test_cpp_tokenizer.py -q    # needs the Qwen3-0.6B tokenizer cached
# M1 Ultra two-pass attention rule (expect "auto ... 0"; if not, the candidate with 0 is MLX's choice -> fix sdpa2_rule)
for kv in 2 1 4; do .venv/bin/python benchmarks/cpp_sdpa_2pass_probe.py --heads 8 --kv-heads $kv \
  --candidates auto,off,std:64,std:128,std:256,gqa:2:128,gqa:4:128; done
.venv/bin/python benchmarks/cpp_sdpa_2pass_probe.py --heads 8 --kv-heads 2 --positions 9300 --start 8192 \
  --candidates auto,std:128,std:256
.venv/bin/python -m pytest tests/unit/test_cpp_generate.py -q -m model -k long_context
F=experiments/artifacts/compress/compress_qwen3_8b/container/model.suc1
~/suc-local/cpp-build/suc1-generate $F --prompt "The history of the printing press begins in the fifteenth century," --steps 64
for p in fused separate; do ~/suc-local/cpp-build/suc1-generate $F --tokens 785,3840,315,279,18484,3493,12033,304,279,11128,45387,9294,11 \
  --steps 128 --projections $p; done            # decode ms/token (pipelined), memory line
mkdir -p logs && TS=$(date -u +%Y%m%dT%H%M%SZ) && HF_HUB_OFFLINE=1 nohup .venv/bin/python \
  benchmarks/cpp_generate_check.py --file $F --steps 64 --ppl-windows 8 --ppl-seq-len 512 \
  --out-dir experiments/results > logs/m4_check_8b_${TS}.log 2>&1 &
```

Sequential, never in parallel with other GPU jobs; all validation positions below 1024 (deviation 2). The benchmark
now compares the C++ separate plan with `group=False` and the C++ fused plan with the default runtime (hidden states
and logits bit-identical fractions, KL, greedy), and times fused / separate × pipelined / sync. Load the 8B file once
in Python on the Studio before the first C++ run if its tuning cache lacks the 8B sizes (the C++ never tunes).

## Milestone 5: qwen4_exp (Qwen3.8-Flash-Next)

`suc/qwen4.hpp`, `src/metal/qwen4.cpp`, kernels `metal/suc_qwen4_kernels.metal` (appended to the embedded source).
`suc1-generate` dispatches on `model_type == "qwen4_exp"`. Reference: `suc/models/mlx_qwen4_exp.py` on the Python
SUC1 runtime (`load_suc1_model(path, group=False, affine_modules=False)`, `step_hidden` per token); architecture:
[docs/arch/qwen4_exp.md](../docs/arch/qwen4_exp.md).

### Design

- **Graph** (one token per step, as `Model.step_hidden`): embedding gathered into `hc` identical streams; per layer the
  PLE update (PLE layers), the attention gated residual (`hc_norm` → BF16 mixers `input_mix_weight_down / up` → `u`,
  injection weights `2 σ(W_inj x̂ / hc)`), Gated DeltaNet or gated attention on `u`, injection into every stream, the MLP
  gated residual and the MoE; the final mixer feeds the LM head (no final norm).
- **Weights in place**: SUC projections are `metal::Linear` (one per projection: the Python plan runtime), the expert
  stacks `metal::SwitchLinear` through `MoeBlock::encode_experts`; the BF16 tensors (mixers, routers, `in_proj_a / b`,
  `index_qk_proj`, `shared_expert_gate`) run MLX's gemv order (`suc_moe_gemv`), the affine LM head `suc_gemv_affine`,
  norms / conv weights / `A_log` / `dt_bias` are read in place in their stored dtype.
- **Ops** (`suc_qwen4_kernels.metal`): every tensor the Python model materializes in bf16 is rounded at the same point;
  Python float scalars meet bf16 arrays as bf16 values (MLX weak typing: `/ hc`, `/ sqrt(H)`, the q / k scales of the
  delta rule, `1e-6`); column sums (stream mean, expert combine) accumulate in bf16; the PLE gate `<k_i, q_i>` follows
  MLX's bf16 row reduce (measured: N ≤ 64 sequential; ≤ 512 one simdgroup; ≤ 1024 128 threads; above 1024 threads, 4
  contiguous elements per thread per step). Gated delta rule = the mlx-lm kernel (state float32, in place); attention =
  MLX `sdpa_vector` / `sdpa_vector_2pass` with MLX's bool-mask branch fed by the QSA selection (head dims 64 / 96 / 128 /
  256, the two-pass rule of milestone 4), MLX's unfused fallback for other head dims; router softmax in float32, top-k in
  argpartition order, renormalized, scores rounded to bf16.
- **QSA indexer**: raw indexer keys cached per token; a block key (RoPE(k_layernorm(mean of 4 raw keys)) at the block
  start) is computed once when its block completes; from `budget / ratio` complete blocks on, block scores
  `Σ_h relu(q_h · k_b) / √d` and an exact top-k (radix threshold search; ties by lowest block index — Python uses
  argpartition, so exact ties may be broken differently) give the selection, the incomplete tail always visible.
- **N-gram tables (never resident)**: per token the host hashes the token history (EOS-aware 2-token context, xor of
  products with the layer multipliers, mod the per-head primes; the stored I64 hash buffers are checked against the
  recomputation at load) and reads the 16 rows from the mapping: raw BF16 rows copied, row-affine records
  (`suc repack`, `group_size` = row width, `scheme: row_minmax`) decoded `q · s + b` in float32 on the CPU. **The GPU
  views of the mapping leave the tables out** (`MappedFile(ctx, f, qwen4_table_ranges(f))`; `Qwen4Model` refuses a
  mapping whose views cover them): a command buffer that uses a no-copy buffer makes all of its pages resident — measured
  on the M3, one float read per 128 MB of an 8 GB mapped file faulted in all 8 GB.
- **Caches**: K / V (12 layers on the real model), indexer raw keys and block keys, Gated DeltaNet conv state (3 rows) and
  float32 recurrent state, PLE conv state (9 rows, dilation 3), n-gram context; `reset()` clears them.
- **Row-exact prefill** (`--prefill exact` / `$SUC_PREFILL_EXACT=1` since 2026-10-10; previously the default; batched,
  layer-major, `Qwen4Options::layer_prefill`): the prompt in chunks of `--chunk` (64)
  tokens, every chunk layer by layer. Over all chunk rows at once, every row with the arithmetic of its decode step:
  the affine projections (`suc_gemv_affine_rows`: each weight element decoded once and applied to a group of rows,
  4 by default, `$SUC_AFFINE_ROWS_NB`; every row's arithmetic and summation order are those of the one-row
  `suc_gemv_affine`), the dense (raw) projections (`suc_moe_gemv_rows`, same idea for `suc_moe_gemv`, 4 rows,
  `$SUC_GEMV_ROWS_NB`), the SUC projections (`Linear::encode_rows_exact`: rotations and residual over all rows, the
  residual with the one-row reduction split, the lookup in tiles of up to `max_exact_rows` rows on the vec path —
  2 by default, `$SUC_EXACT_TILE`: the vec lookup is ALU-bound beyond ~2 rows per weight read on the M1 Ultra —
  whose CHUNK switch is that of the one-row launch; `$SUC_PREFILL_ROW_PROJ=1` = one row at a time, A/B), the
  Gated DeltaNet step (`q4_conv_rows`, `q4_delta_prep_rows`, `q4_gated_delta_rows`: conv, gate prep and recurrence of
  all chunk tokens in one launch each, the state carried in registers; `$SUC_PREFILL_LIN_TOKENS=1` = the per-token
  loop), the router / shared expert, and the routed experts in **expert tiles** (the chunk's routed
  rows sorted by expert into tiles of `bt` rows, one expert per tile; `suc_switch_lookup_vec` with BT = bt reads the
  expert's codes once per tile; bt is a power of two up to the largest tile height for which the vec lookup keeps its
  per-row arithmetic (`SwitchLinear::max_tile_rows`: its CHUNK switch), chosen per layer by
  Σ tiles · (1 + 0.3 (bt − 1)), `$SUC_TILE_BT` forces it); token by token in position order on the one-token buffers:
  PLE, KV cache, indexer and attention (the hyper-connection mixers run stage by stage over the rows).
  `$SUC_Q4_PROFILE=1` splits each chunk into one command buffer per stage and prints the GPU time per stage. One host sync per MoE layer per
  chunk (the tile plan; streamed: the expert fetch, in token groups whose experts fit the slots). Same bits as decode
  steps: the last prompt token's logits and streams sha256 (`prefill_logits_sha256` / `prefill_streams_sha256` in
  `--json`) are compared with `--prefill token` (decode steps, 16 per command buffer) in
  `test_batched_prefill_bit_identical` (raw / repacked / affine-8-bit projection files, chunks 5 / 7 / 64, forced
  tiles of 2 and 16 rows with padding, streamed experts with top-k + 1 slots). The n-gram rows of the prompt are
  gathered up front per chunk. With per-position outputs (`--logits-out` ...) one step per command buffer. Dense
  qwen3_5 files take the same path (projections and MLP over the rows). The default is now the fast batched prefill
  (section 'Fast batched prefill').
- **Tokenizer / chat**: the Qwen3.8 `tokenizer.json` uses the Qwen3.5 Split regex (letter runs `[\p{L}\p{M}]+`, the
  punctuation class also excludes `\p{M}`): implemented as a second matcher mode, `\p{M}` added to the Unicode tables
  (`tools/gen_unicode_tables.py`, class bit 8). `--chat` on a qwen4_exp file renders the model's `chat_template.jinja`
  (single user turn, optional system turn; with thinking a reasoning-effort system turn, `--reasoning-effort
  xhigh|medium|low`, default xhigh, and the `<think>\n` generation prompt; `--no-think`: `<think>\n\n</think>\n\n`).
  `--stop-eos` stops at the text `eos_token_id` or the tokenizer's end tokens (`<|im_end|>`, `<|endoftext|>`).
  `--print-prompt` prints the prompt ids without loading the model.

### Tests

- `tests/unit/test_cpp_qwen4.py` (CTest `suc-qwen4-crosscheck`; skipped unless built): tiny qwen4_exp checkpoints
  (`tiny_models.write_tiny_qwen4_exp`) through `suc compress` (VQ K = 16 G = 2, one-sided input rotation, column int8
  residuals, expert-stack codebooks, affine embedding / head), raw and `suc repack`ed (row-affine) tables, two shapes:
  the tiny one (hidden 64, head_dim 16 → MLX's attention fallback) and one at the real widths where kernels branch
  (hidden 2560 = 10240-wide streams, head_dim 256, linear key dim 128, 16 n-gram heads × 160). Prompt + 30 greedy
  tokens (33 positions, QSA sparse from position 11), teacher-forced Python run on the C++ sequence: logits / streams
  error, KL, argmax and the free-running greedy tokens; chunked prefill = per-step prefill = Python greedy (EOS inside
  the prompt); teacher forcing; vocabulary check; the GPU views exclude the tables; the Qwen3.8 chat template against
  jinja2 + HF `tokenizers` (when the snapshot is in the HF cache).
- `tests/unit/test_cpp_tokenizer.py`: `test_qwen38_*` — the Qwen3.8 tokenizer.json against HF `tokenizers` on WikiText-2
  (2,891 lines, 297,053 tokens), the edge cases and the random corpus plus combining-mark strings: 0 mismatches
  (encode with / without special tokens, decode, round trip). Needs the snapshot in the HF cache or
  `$SUC_QWEN38_TOKENIZER`.

### Results (M3 laptop, 2026-10-03)

Every test file, both table formats, both shapes: **logits and every stream bit-identical** to the Python plan runtime
at all 33 positions (KL 0), identical greedy tokens; against `affine_modules=True` (MLX quantized matmul head) logits
differ by ≤ 1 bf16 ulp (KL ≤ 2e-6), same tokens. Not compared: the default Python runtime (`group=True, fuse=True`; its
segmented rotations round differently, as for Qwen3). Tiny file decode ≈ 1.8 ms/token (synchronous, one host round trip
per token for the n-gram rows; GPU 1.3 ms) — no timing claim; real-model numbers only from the Studio.

### Not supported / limits

- MTP head and vision tower (ignored, as in the Python runtime); tied embeddings, attention bias, other layer types.
- No pipelined decode (the next token's n-gram rows need the host).
- Long contexts: the QSA path is exact on the M3 within the measured MLX dispatch; the M1 Ultra two-pass rule is the
  unverified one of milestone 4. Exact ties between block scores may select different blocks than argpartition.
- Real file measured on the Studio: see the table at the top (73.82 GB, 28.3 tok/s).

### Mac Studio: the real file

```bash
cd ~/Projects/SUC-LM && cpp/build.sh ~/suc-local/cpp-build
.venv/bin/python -m pytest tests/unit/test_cpp_qwen4.py tests/unit/test_cpp_tokenizer.py -q -k "qwen4 or qwen38"
F=experiments/artifacts/compress/compress_qwen3.8_flash_next_side8/container/model.suc1   # chosen file (ngram4)
~/suc-local/cpp-build/suc1-generate $F --prompt "The history of the printing press begins" --steps 64 --text
~/suc-local/cpp-build/suc1-generate $F --prompt "Explain QSA attention briefly." --chat --no-think --stop-eos \
  --steps 256 --max-context 4096 --json logs/q4_cpp_$(date -u +%Y%m%dT%H%M%SZ).json
```

Sequential, never alongside other GPU jobs. The C++ never tunes: if the tuning cache lacks the qwen4_exp sizes (2560,
640, 6144), run `suc generate $F --tokens 1 --steps 1` once first (Python autotunes on load).

Expected memory (estimate from docs/arch/qwen4_exp.md §6; the stderr `memory:` line reports the measured values): GPU
views = the file minus the n-gram tables ≈ 67 GB (eligible codes ≈ 65 GB + BF16 kept ≈ 1.5 GB + affine embedding / head
≈ 0.7 GB), all made resident on the first step (≈ 15–20 s from SSD); execution arrays built at load (per-expert sign
tables, column residuals) ≈ 1 GB; KV + indexer keys ≈ 28 KB per position (115 MB at 4,096); Gated DeltaNet states
≈ 115 MB; arena < 50 MB. Total ≈ 68–70 GB resident (measured: 73.82 GB); the n-gram tables (102 GB raw / 27 GB row-affine) stay in the page
cache only as far as rows are touched (16 rows per token). Fits the 128 GB Studio, not a 32 GB Mac.

## SSD expert streaming (`suc/expert_stream.hpp`, `src/metal/expert_stream.cpp`, tool `suc-expert-bench`)

For MoE files larger than the memory the runtime may use (docs/expert_streaming.md). `suc1-generate FILE
--expert-budget BYTES[K|M|G] [--hot-experts FILE|auto] [--expert-policy lru|slru] [--expert-io-threads N]
[--expert-pin-fraction F] [--expert-prefetch] [--expert-cached-io]`; the JSON gets an `expert_streaming` record
(pools, pinned set, requests / hits / misses / evictions / prefetch hits and waste, bytes and read calls, IO and blocked
time; warm-up excluded) and `excluded_from_gpu_views`.

- **Pools.** One shared-storage MTLBuffer per MoE layer of S_l slots; a slot holds one expert's streamed arrays in the
  file's own layout (runs of adjacent arrays, 64-byte aligned), so a miss is a read straight into the slot: the stored
  bytes are the kernels' operands, no conversion. S_l = the same capacity fraction for every layer, at least top-k,
  leftover budget handed out slot by slot; Σ pools ≤ budget (checked in the tests). A budget below every layer's top-k
  slots is an error.
- **Reads.** `pread` on one descriptor with `F_NOCACHE` (the slot is the only copy) on a pool of IO threads (default
  8); all misses of a layer are submitted together, runs split into ≤ 1 MiB reads.
- **Policy.** Per layer LRU or segmented LRU (probation + protected 80 %, the default). Hot set: the routing counts of
  `hot_experts.json` (`suc compress` with preflight.expert_streaming; keys `"<prefix>.mlp"`), pinned greedily by count
  over all layers up to `--expert-pin-fraction` of the budget (default 0.5), leaving top-k unpinned slots per layer.
- **Prefetch** (`--expert-prefetch`, off by default): the next MoE layer's router applied to the current layer's MoE
  input (pre-gating heuristic), its top-k read asynchronously while the GPU runs the current experts; a layer's fetch
  first waits for its pending prefetch reads.
- **GPU views.** The routed-expert payloads (`routed_expert_ranges(file)`: every array of every `….experts.<e>.…`
  tensor) are left out of the MappedFile views (whole pages; tiny test files have none), as the qwen4_exp n-gram tables.
- **Codec-agnostic hook.** `ExpertStream::add_layer(name, per-expert file ranges, min_slots, bind)`; on install the
  stream calls `bind(expert, pool, byte offsets of the ranges)`. `SwitchLinear` gained `StackLayout::Streamed` (the
  `_tab` kernels, packed indices / row scales based at the pool, offset table entries rewritten by `bind_expert`;
  codebooks, sign tables and residuals stay resident as in `Table`), `MoeBlock::attach_stream / stream_in /
  stream_prefetch`. Another expert codec (e.g. native MXFP4) registers its own arrays the same way.
- **Graphs.** `ModelOptions::expert_stream` / `Qwen4Options::expert_stream`. Per streamed layer: router, end the
  command buffer and wait (`split_command_buffer`), fetch the routed ids on the host, encode the experts. Streamed
  decode keeps one command buffer in flight (`generate` forces depth 1); qwen3_moe batch prefill runs chunks of at most
  ⌊min unpinned slots / top-k⌋ tokens. Streamed and resident are bit-identical (same kernels, same operand bytes).

- **deepseek_v4** (`Ds4Options::expert_stream`): every layer's three `ExpertStack`s are built streamed (mixed-codec
  sub-stacks = `SwitchLinear::from_file_slots(..., streamed)` over the global expert ids, native experts =
  `Mxfp4Switch` with its codes / scales bases at the pool and its byte-offset table rewritten on install); per expert 6
  ranges ({gate, up, down} x {packed indices or MXFP4 codes, row scales or E8M0 scales}), `bind` dispatched to the
  sub-stack holding the expert. Per layer: router + gate on the GPU, split, fetch, experts; prefetch = the next layer's
  gate (router + bias, or `tid2eid[token]` on hash layers, i.e. exact) on this layer's MoE input. `--mixed-experts all`
  is rejected with a budget (it reads each sub-stack's first expert for foreign rows, which need not be resident).
  Chunked prompt prefill keeps working (every MoE layer of every step splits the command buffer).

- **Residency set** (`Context::make_resident`, macOS 15+, `SUC_RESIDENCY_SET=0` disables): the pools, the file views
  and every `new_buffer` of at least 64 KiB join one `MTLResidencySet` attached to the queue (up to
  max(recommendedMaxWorkingSetSize, `iogpu.wired_limit_mb`); the JSON reports `residency_set_bytes` /
  `residency_skipped_bytes`). Without it each per-layer command buffer re-made its pool resident, and near the wired
  limit the driver unwired idle pools between command buffers: GLM-5.3-Flash on the Studio decoded at 2.58 s/token
  with 0.14 s of GPU time and 0.07 s blocked in reads (≈ 55 ms per split, 42 splits). `suc-expert-bench --gpu`
  (one command buffer per layer reading the routed slots; `--experts E` wraps the ranges around the file so pools can
  exceed it, `--prefetch` = perfect next-layer prediction) on the M3, GLM shape (9.4 MB experts, 42 layers, top-8,
  27 experts per layer, 8.85 GB of pools), 8 IO threads: 968 ms/token without the set (409 ms/token outside GPU and
  reads), 567 ms/token with it (37 ms/token outside GPU and reads); per commit 23 ms p50 / 335 ms p90 → 0.2 ms in a
  standalone 8 x 1.2 GB test. Studio after the fix: GLM-5.3-Flash base streamed 9.1 tok/s, 91 % hit rate (table at the top).

Not streamed (resident, built at load as in `Table`): expert residual arrays (CSR / columns; ≈ 1–3 % of an expert),
per-expert sign tables and codebooks. Not done: prefetch-accuracy numbers on a real routing trace.

Read path on the M3 laptop (2026-10-04; `suc-expert-bench ~/suc-local/qwen3_8b_final/model.suc1`: synthetic experts
over a 4.3 GB file, 48 layers, uniform random top-10, 16 tokens, SLRU, F_NOCACHE, no compute between layers; laptop
numbers, re-run on the Studio and the target Macs):

| expert MB | budget | IO threads | IO ms / token | MB read / token | GB/s while blocked | layer miss p50 / p90 ms |
|---|---|---|---|---|---|---|
| 2.58 | 35 % (hit 0.32) | 1 | 546 | 836 | 1.53 | 11.0 / 14.8 |
| 2.58 | 35 % | 4 | 259 | 835 | 3.23 | 5.3 / 7.2 |
| 2.58 | 35 % | 8 | 224 | 839 | 3.77 | 4.6 / 6.0 |
| 1.54 | 25 % (hit 0.23) | 4 | 203 | 573 | 2.83 | 4.1 / 5.3 |
| 1.54 | 25 % | 8 | 187 | 569 | 3.06 | 3.9 / 4.8 |

One layer's ~7 misses reach 3–3.8 GB/s, below the ~5 GB/s of `scripts/pread_bench.py` with a deep queue: per layer the
reads are dependent on the router, so the queue drains every layer (prefetch is what can deepen it).

Tests: `tests/unit/test_cpp_expert_stream.py` (Qwen3-MoE per-expert codecs with CSR residuals and column residuals, LRU /
SLRU / prefetch, both projection plans, hot set, budget error, batch prefill, pipelined decode, GPU-view exclusion;
qwen4_exp with and without prefetch): logits, hidden states and tokens bit-identical to the resident C++ runtime under
budgets of top-k + 1 slots per layer (evictions on every step). `tests/unit/test_cpp_deepseek_v4.py::test_streamed_experts_equal_resident`:
deepseek_v4 tiny / wide / real shapes, base (mixed VQ / additive stacks) and native (+ MXFP4 experts, an all-native
layer), SLRU and LRU + prefetch, chunked prefill: logits, streams and tokens bit-identical to resident at top-k + 1
slots per layer; `--mixed-experts all` with a budget fails.

## Milestone 6: deepseek_v4 (DeepSeek-V4-Flash)

`suc/deepseek_v4.hpp`, `src/metal/deepseek_v4.cpp`; expert stacks `suc/expert_mixed.hpp`, `src/metal/expert_mixed.cpp`;
kernels `metal/suc_ds4_kernels.metal` (appended to the embedded source). `suc1-generate` dispatches on
`model_type == "deepseek_v4"`. Reference: `suc/models/mlx_deepseek_v4.py` on the Python SUC1 runtime
(`load_suc1_model(path, group=False, affine_modules=False)`, `step_hidden` per token); architecture:
[docs/arch/deepseek_v4.md](../docs/arch/deepseek_v4.md).

### Design

- **Graph** (one token per step): the embedding gathered into the 4 hyper-connection streams; per layer
  `hc_attn.pre` (float32 RMS scale of the 16384-wide stream, the 24 mixer logits in MLX's gemv order, sigmoid gates,
  row softmax + 20 Sinkhorn normalisations in one thread, the stream collapse), attention, `hc_attn.post` (the 4 x 4
  mixing matmul as MLX's fma chain), the same around the MoE; `hc_head` + norm + LM head. `--hidden-out` writes the
  Python stream tensor (streams + the 3 token-id digit channels of hash routing).
- **Attention**: wq_a / wq_b / wkv / wo_a / wo_b are `metal::Linear` (wo_a as the Python SUC module: every row group on
  every input group, the diagonal blocks taken; dense wo_a: one gemv per group); per-head float32 RMS norm of q, RoPE
  on the last 64 channels (base theta on window layers, YaRN + compress_rope_theta on compressed layers; the inverse
  frequencies in numpy's float32 arithmetic), the FP8 round trip of the non-RoPE KV channels (MLX's E4M3 conversion),
  a ring of the last `sliding_window` keys. Compressors (ratio 4 with the overlapping previous group, ratio 128) keep
  the raw float32 rows of the current and previous group and emit an entry when a group completes (softmax over the
  rows, pooling, norm, RoPE at the group start, FP8 or Hadamard + FP4). The lightning indexer (its own ratio-4
  compressor) scores every entry once more than `index_topk` exist and selects the top k (exact threshold search,
  ties by the lower index = argpartition's order). Attention runs MLX's dispatch: `sdpa_vector` with the bool mask and
  sinks for head dims 64 / 96 / 128 / 256 with at most 32 query heads, else MLX's unfused fallback — the real head_dim
  512 takes it (an array mask always rules out MLX's head-dim-512 vector kernel). MLX folds the H query heads of the
  one KV head into M = H rows of each matmul, so the C++ follows MLX's matmul dispatch: one head: scores in gemv order,
  p @ V in gemv_t order; H > 1: q' Kᵀ through `gemv_wide` (`ds4_fb_scores_wide`; Apple GPU generation >= 15, 2-15
  heads, bf16 / f16, >= 2 keys; 32 lanes per key with one pass or <= 64 keys, else 16) or MLX's steel GEMM order
  (`ds4_mma`: fused with the K % BK tail first, split-k when K >= max(M, N) and K / 16 >= 8; the order and plan of
  `glm_mma` / `gemm_plan` of the GLM graph), one key: gemv over the H rows (BN by H); p[1:] @ V always the steel GEMM
  (split-k from 512 keys on). Keys are read in place (window ring + entries); softmax(precise) single-row up to 4096
  values, MLX's looped kernel with 1024 threads above.
- **MoE**: router logits in float32 (gemv), sqrt(softplus) scores (one thread per expert), hash layers from
  `tid2eid[token]` (any integer dtype), others the top k of scores + bias (descending, ties by index), weights
  normalised and scaled; clamped SwiGLU in float32 with the routing weight before w2; the k expert rows summed in
  float32 in MLX's column order; shared expert added in float32.
- **Expert stacks** (`ExpertStack`, = `switch_from_encodings`): one `SwitchLinear` when every expert has one code
  layout; `Mxfp4Switch` when all are native MXFP4 (SUC1 `mxfp4` records read in place: a port of MLX's
  `fp_gather_qmv` / `fp_gather_qmv_fast` for `gather_qmm(mode="mxfp4")`, the fast kernel when in % 512 == 0 and out %
  8 == 0); otherwise a **mixed stack** = `SucSwitchMixed`: one sub-stack per layout (native experts form one), each
  over the GLOBAL expert ids (`SwitchLinear::from_file_slots`: empty slots carry a ~0 packed offset that the `_tab`
  lookup prologue tests and skips), then one select launch takes row n from the sub-stack of its expert.
  `--mixed-experts routed` (default): rows routed to other sub-stacks are skipped (no weight reads); `all`: every
  sub-stack computes every row (the Python semantics, A/B). Same bits either way (tested).
- **Numerics** — measured against MLX 0.32.3 with probes (`mx.fast.metal_kernel` candidates vs the MLX ops):
  MLX rounds every element-wise op, so the multi-op kernels disable fma contraction (`#pragma clang fp
  contract(off)`); precise exp / log / sqrt / rsqrt / sin / cos except where MLX writes `fast::exp` (softmax,
  attention); row sums (`mean(x²)`) with MLX's row-reduce threads (32 up to 512 values, 128 at 1024, 512 at 2048,
  1024 above, 4 contiguous per thread, simd sums); column sums (streams, compressor rows, indexer heads, expert rows)
  as 8 interleaved sequential partials up to 16 rows, 32 partials + `simd_sum` from 32 rows; a non-last-axis softmax is
  max, precise exp, column sum, division; `mx.hadamard_transform` on bf16 is `hadamard_n` with radix 16 and bf16
  intermediates; the bf16 indexer matmul agrees after rounding (sequential fma).
- **Row-exact prefill** (`--prefill exact` / `$SUC_PREFILL_EXACT=1` since 2026-10-10; previously the default; batched,
  layer-major, `Ds4Options::layer_prefill`): as the qwen4_exp graph above, chunks of
  `--chunk` (64) tokens layer by layer: `wq_a`, `wkv`, `wq_b`, `wo_a` (per group), `wo_b` and the shared expert over the
  chunk rows (affine / dense / SUC row-exact as for qwen4_exp), q / kv norms over the rows,
  the routed experts in expert tiles (SUC sub-stacks: the vec lookup with BT rows; native MXFP4:
  `ds4_mxfp4_qmv_tile`, MLX's `fp_qmv` / `fp_qmv_fast` bodies over BT rows sharing one expert's codes, every row's
  arithmetic unchanged; tiles planned in (gate, up, down) sub-stack order and each sub-stack of a mixed stack run
  only over the tile range of its own experts, `$SUC_MIXED_ALL_TILES=1` = all tiles), hyper-connections, compressors
  (rows + emitted entries), window KV, indexer and attention token by token in position order; the kv rope and q
  norm + rope of every row run before that loop and the inverse rope of the outputs after it, without barriers
  between tokens (`$SUC_DS4_SEQ_TOKENS=1` = inside the loop, A/B). One host sync per MoE layer per chunk; streamed experts are fetched per chunk in
  token groups whose distinct experts fit the slots. Bit-identity with `--prefill token`
  (`test_batched_prefill_bit_identical`: base / native / affine-8-bit projections x tiny / wide / real shapes, chunks
  5 / 7 / 64 across compressor groups and the indexer's top-k regime, forced 2- and 16-row tiles, streamed). The default is now the fast batched prefill (section 'Fast batched prefill').
- **Tokenizer / chat**: DeepSeek's `tokenizer.json` (empty `Sequence` normalizer, three `Split` pre-tokenizers:
  digit groups of up to 3, CJK runs, the punctuation / letter / whitespace regex with `\p{P}` and `\p{S}` — new Unicode
  table classes from HF `tokenizers`' engine, `tools/gen_unicode_tables.py`). `--chat` renders the single-turn
  `encode_messages` of the checkpoint's `encoding/encoding_dsv4.py` (DeepSeek-V4-Flash-0731 rev 7872f01b): BOS, in
  thinking mode the `REASONING_EFFORT_PROMPTS` text of `--reasoning-effort low|high|max` (default low = none), the
  system text, `<｜User｜>`, the prompt, `<｜Assistant｜>`, then `<think>` (thinking) or `</think>` (`--no-think`);
  tokenized as `inference/generate.py` does (`tokenizer.encode`, `add_bos_token` false). `--stop-eos` stops at
  `<｜end▁of▁sentence｜>`. Tools, tasks and `latest_reminder` are not rendered; multi-turn history (suc1-server) uses the
  per-turn markup, not checked against `encode_messages`.
- **Router**: the checkpoint's `scoring_func` / `score_func` is `sqrtsoftplus` (config.json and inference/config.json at
  the pinned revision); `inference/model.py` `Gate` = `F.softplus(x @ Wᵀ).sqrt()` in float32, top-k of scores + bias,
  weights = unbiased scores normalised x `route_scale`; `suc/models/mlx_deepseek_v4.py` and the C++ `ds4_route`
  implement exactly that (checked 2026-10-04; `test_gate_scores_follow_the_reference_score_func`), so the Studio
  compress runs need no redo for the router.

### Tests

- `tests/unit/test_cpp_deepseek_v4.py` (CTest `suc-deepseek-v4-crosscheck`; skipped unless built): tiny DeepSeek-V4
  checkpoints in the original FP8 / MXFP4 layout encoded tensor by tensor (attention: VQ / additive, rotations, int8
  column and fp16 CSR residuals; experts: VQ K=16 G=2 + column residual, VQ K=64 G=4, additive M=3 per expert = mixed
  stacks in every layer; the "native" variant adds verbatim MXFP4 experts (4 codecs in one stack) and an all-native
  last layer), three shapes: tiny (head_dim 128: sdpa_vector; fp_gather_qmv), wide (head_dim 512: MLX's fallback;
  fp_gather_qmv_fast) and real (hidden 4096 = 16384-wide streams, head_dim 512, 64 indexer heads, experts 2048 x 4096,
  a ratio-32 compressor). 40 positions (indexer top-k active from 20), teacher-forced Python run on the C++ sequence:
  **every stream and logit bit-identical** is asserted, free greedy = Python greedy; `--mixed-experts all` = routed;
  chunked prefill = per-token = Python greedy; teacher forcing; vocabulary check; the chat prompt ids against the
  checkpoint's `encode_messages` (fetched into the HF cache at the pinned revision; thinking / chat, system turn,
  reasoning effort low / high / max); streamed experts = resident (above).
- `tests/unit/test_cpp_tokenizer.py`: `test_dsv4_*` — DeepSeek-V4-Flash tokenizer vs HF `tokenizers` on WikiText-2
  (2,891 lines, 287,597 tokens), the edge cases and random corpora with CJK / digits / punctuation / symbols / special
  tokens: 0 mismatches (encode with / without special tokens, decode, round trip). Needs the snapshot in the HF cache
  or `$SUC_DSV4_TOKENIZER`.

### Results (M3 laptop, 2026-10-04)

All 15 cross-check tests: logits and streams bit-identical at all 39 compared positions for the 6 files (tiny / wide /
real x base / native; all with 2 attention heads; 64 heads and long sequences: the folded-attention item below). Indicative timing (M3, synthetic 433 MB file at the real widths, 2 layers, 16 experts top-6,
layer 0 a mixed stack of 4 codecs (native + VQ G=2 + VQ G=4 + additive M=3), layer 1 all native; 64 greedy steps,
3 runs): C++ `--mixed-experts routed` 3.9 ms/token (GPU 3.5), `all` 6.4 - 7.2 ms/token (GPU 6.1 - 6.9): routing rows
per codec is 1.65 - 1.85x faster on this file; Python runtime 12.3 ms/token. No claim for the real model from these.

### Not supported / limits

- MTP / DSpark head (dropped, as in Python), `simulate_act_fp8` (off in the Python model too), tied embeddings,
  n_shared_experts != 1, num_key_value_heads != 1, `--mixed-experts all` together with `--expert-budget`.
- **Folded fallback attention (fixed 2026-10-04).** Until then the C++ ran per-head gemv / gemv_t for H > 1 heads
  while MLX runs the folded matmuls above; the float32 sums differ in order and flip a bf16 rounding of the attention
  output now and then: probe (`mx.fast.scaled_dot_product_attention` vs the same ops per head, random data) 0-1 of
  4 x 2 x 512 outputs differed with 2 heads, 3-346 of 4 x 64 x 512 with 64 heads (the real model), already at 40 keys;
  on tiny files the logits first differed at position 182 (2 heads) / 20 (64 heads), then KL ~1e-3 at every position.
  The 15 cross-check tests passed only because their files have 2 heads and 40 positions. Now bit-identical:
  `test_folded_attention_heads` (8 and 64 heads, 600 positions: gemv_wide with 32 and 16 lanes, GEMM, split-k on both
  matmuls) and `test_long_context_bit_exact` (`SUC_LONG_CONTEXT=1`, ~4 min per case; 2 and 64 heads, 8,400
  teacher-forced positions with a sliding window of 8192, i.e. >= 8k keys from position 8,191 on and ~10,300 keys on
  the ratio-4 layer): logits identical at all 8,399 positions. The `gemm_plan` 'd' (M1 Ultra) BK table is the GLM
  graph's ("as recalled"); it only matters for fused GEMMs whose K is not a multiple of 16, i.e. p V below 512 keys.
- Verified: MLX's looped softmax (rows > 4096 keys) uses 1024 threads on the M3 (a `mx.fast.metal_kernel` copy of
  `softmax_looped` with 1024 threads equals `mx.softmax(precise=True)` on bf16 rows of 4,097-16,385 values, 0 of
  4.2 M elements differ; 512 / 256 threads differ), which is what `ds4_softmax_looped` runs.
- MLX's two-pass decode attention for the vector path (head dims <= 256, > 1024 keys on M1 Ultra-class 'd' devices,
  > 4096 keys with GQA, i.e. always for deepseek_v4's single KV head): not ported (the real head_dim 512 never takes
  it), so tiny head_dim-128 files are not bit-exact beyond those key counts.
- softmax scoring (`scoring_func: softmax`) runs in one thread (not MLX's softmax order); the DeepSeek-V4 checkpoints
  use sqrtsoftplus.
- No pipelined decode (one command buffer per step).

### Mac Studio: the real file

```bash
cd ~/Projects/SUC-LM && cpp/build.sh ~/suc-local/cpp-build
~/suc-local/cpp-build/suc-tests && ~/suc-local/cpp-build/suc-metal-tests
.venv/bin/python -m pytest tests/unit/test_cpp_deepseek_v4.py -q
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/unit/test_cpp_tokenizer.py -q -k dsv4   # needs the V4 tokenizer cached
F=experiments/artifacts/compress/compress_deepseek_v4_flash_tcq/container/model.suc1   # chosen file (resident)
~/suc-local/cpp-build/suc1-generate $F --prompt "The history of the printing press begins" --steps 64 --text \
  --json logs/ds4_cpp_$(date -u +%Y%m%dT%H%M%SZ).json
~/suc-local/cpp-build/suc1-generate $F --prompt "Explain sparse attention briefly." --chat --no-think --stop-eos \
  --steps 256 --max-context 4096
for m in routed all; do ~/suc-local/cpp-build/suc1-generate $F --tokens 0,128803,2058,128804 --steps 64 --mixed-experts $m; done
```

Sequential, never alongside other GPU jobs. The C++ never tunes: if the tuning cache lacks the deepseek_v4 sizes
(4096, 1024, 2048, 512, 8192, 32768 / 8 groups), run `suc generate $F --tokens 1 --steps 1` once first. Agreement
with Python on the real file: run the teacher-forced comparison of `tests/unit/test_cpp_deepseek_v4.py`
(`--logits-out` / `--hidden-out` vs `run_tokens`); the M1 Ultra's GEMM tile table ('d') is unverified, so a mismatch
there points at `gemm_plan` first. Streaming: add `--expert-budget BYTES [--hot-experts auto] [--expert-prefetch]`.
The affine 8-bit gemv (deepseek_v4 and glm5_next, unsplit columns) runs 4 row simdgroups per threadgroup by default
(bit-identical for any value; 1 → 4 is 49.2 → 45.6 ms/token on V1 alone, §8 of
[decode_speed_vs_ds4.md](../docs/decode_speed_vs_ds4.md)); `SUC_AFFINE_SG=N` overrides it.

Expected memory (from the directory; the stderr `memory:` line reports the measured values): the GPU views map the
whole file (107.5 GB at the configured budget) and every page becomes resident on the first step; execution arrays
built at load (expert offset tables, per-expert sign tables, residual tables) ~1-2 GB; KV + compressor state
~6.9 KB per position + 5.6 MB window keys + 22 MB compressor group rows (0.23 GB at 32k positions); fallback-attention
scores 64 heads x (128 + positions / 4) x 4 B; arena < 100 MB. Total ~110 GB: fits the 128 GB Studio (the
runtime_budget of the compress config is 111 GB), not a 32 GB Mac without SSD expert streaming (`--expert-budget`,
see the SSD expert streaming section). Measured (tcq file): 102.57 GB.
Measured speed: the "Current performance" table at the top (V1 native_max: 25.5 tok/s; tcq: 24.0 tok/s).

## Milestone 7: glm5_next (GLM-5.3-Flash)

`suc/glm5.hpp`, `src/metal/glm5.cpp`, kernels `metal/suc_glm5_kernels.metal` (appended to the embedded source; reuses
the ds4 / qwen4 / model kernels where the Python model has the same op). `suc1-generate` dispatches on
`model_type == "glm5_next"` (`tools/suc1_generate_glm5.inc`, included by `suc1_generate.cpp`). Reference:
`suc/models/mlx_glm5_next.py` on the Python SUC1 runtime (`load_suc1_model(path, group=False, affine_modules=False)`,
`step_hidden` per token); architecture: [docs/arch/glm5_next.md](../docs/arch/glm5_next.md).

### Design

- **Graph** (one token per step): embedding gathered into 4 identical streams; per layer mHC pre (MLX row reduce of
  mean(x²) over the 4 · hidden stream, the stream normalized BEFORE the 24 mixer logits (float32 gemv), sigmoid gates,
  softmax + 20 Sinkhorn normalisations in one thread, stream collapse), input RMSNorm, KDA or MLA + DSA, mHC post in the
  model dtype (`T(T(post) f) + T(combᵀ res)`, the 4 x 4 bf16 matmul a float32 fma chain), the same around the dense MLP /
  MoE; head: stream mean (float32), norm, LM head (affine codes in place or dense).
- **KDA**: q / k / v projections; causal depthwise conv4 + SiLU in float32 over a rolling bf16 tail (3 rows); q / k
  `x / sqrt(Σx² + 1e-6)` (row sums of <= 64 values sequential in one thread = MLX row_reduce_small, longer rows MLX
  row_reduce_simple); the safe-gate decay `exp(-5 σ(exp(A_log) (f_b f_a x + dt_bias)))` per channel, beta = σ_bf16(b x);
  the mlx-lm gated-delta step with the vectorized gate (float32 state in place); o rounded to bf16, float32 RMSNorm
  (`o_norm` weight cast up), `· σ(g_b g_a x)`, o_proj.
- **MLA + DSA**: q_a → RMSNorm → q_b; the kv_a latent (RMSNorm) appended to a per-layer latent cache; kv_b applied
  absorbed per head (`q'_h = q_h W_k,h`: MLX gemv_t per head; `o_h = (P c) W_v,hᵀ`: MLX gemv per head). Indexer: key
  `LayerNorm(wk x)` (MLX layer_norm_single_row) and gate score `kpool_gate x` kept as float32 rows of the incomplete pool;
  every 4 tokens a pooled key (softmax over the pool of gate + ape, `ds4_comp_pool`); once more than index_topk / 4
  complete pools exist, pool scores `Σ_h w_h relu(q_h · K_p / √d)`, exact top-k (`q4_index_select`, ties by lower
  index = MLX's argpartition, a stable sort on the GPU), and the selected pools' tokens plus the incomplete tail are the
  attention mask. Attention: MLX `sdpa_vector` (kv_lora_rank 64 / 96 / 128 / 256 and <= 32 heads) or MLX's unfused
  fallback (the real rank 512).
- **MLX folds the fallback's heads into GEMMs.** The H query heads share the one latent KV head, so `q' Kᵀ`, `P V` and
  the indexer's `q poolsᵀ` are single steel GEMMs in MLX (B broadcast over the batch -> M = heads), not per-head gemvs:
  `glm_mma` reproduces their float32 order with 8 x 8 x 8 simdgroup MMAs: split-k when K >= max(M, N) and K / 16 >= 8
  (P = min(max(2, next_pow2((K/16) / (⌈M/32⌉ ⌈N/32⌉))), 32) partitions of `((K/16)/P) · 16`, partials summed from 0),
  else the fused kernel with the unaligned K tail first (BK by MLX's device-class table). Measured with float32 operands
  against MLX 0.32.3 on the M3 (every shape bit for bit); a single key (first token) or a single head takes MLX's gemv.
  The deepseek_v4 fallback attention (64 heads on one KV head, head_dim 512) uses the same folded GEMM order (fixed
  2026-10-04, milestone 6).
- **MoE**: router logits float32 (gemv), sigmoid, top-k of scores + correction bias (descending, ties by index),
  `w / (Σw + 1e-20) · 2.5`; routed experts = `ExpertStack` (SUC codes; mixed codecs per stack = sub-stacks + select);
  clamped SwiGLU (no weight), down, `T(Σ_k float(y_k) w_k)` (MLX column order) `+ shared` in bf16. Dense MLP layers and
  the shared expert: SUC / affine / dense projections + the clamped SwiGLU.
- **Weights in place**: SUC projections (`metal::Linear`), affine projections (`suc_gemv_affine` = the Python bf16
  decode + gemv), raw tensors in their stored dtype; float32 copies only where the Python parameter is float32 and the
  file is not (conv weights, `o_norm` after the bf16 cast, A_log / dt_bias / router bias / mHC base and scale).
- **SSD expert streaming** (`--expert-budget`, as milestone 6's stacks): `ExpertStack::from_file(..., streamed)`, per
  expert the {gate, up, down} x {indices, scales} ranges registered with `ExpertStream::add_layer("<prefix>.mlp")`
  (= the `hot_experts.json` keys of `suc compress`); per MoE layer: router, `split_command_buffer`, `fetch`, experts;
  `--expert-prefetch` routes the next MoE layer's router on this layer's MoE input. Same bits as resident.
- **Chat**: `--chat` renders the GLM-5.3-Flash chat_template.jinja (rev eb9eb208):
  `[gMASK]<sop><|system|>Reasoning Effort: Max` (`--reasoning-effort low|high|max`, default max),
  `[<|system|>TEXT]<|user|>PROMPT<|assistant|><think>`. The template ignores enable_thinking, so `--no-think` does not
  change the prompt. Prompt ids are checked against transformers by `test_chat_prompt_format`. The
  tokenizer is the generic byte-level BPE; a `tokenizer.json` with a pre-tokenizer regex the C++ tokenizer does not know
  fails loudly at load.

### Tests

`tests/unit/test_cpp_glm5.py` (CTest `suc-glm5-crosscheck`; skipped unless built): tiny GLM-5.3-Flash-structured
checkpoints in the FP8 layout encoded tensor by tensor (KDA / MLA / MLP / shared projections: VQ / additive with
rotations, int8 column and fp16 CSR residuals, affine 8-bit; routed experts: VQ K=16 G=2 + column residual, VQ K=64 G=4,
additive M=3 per expert = mixed stacks; affine 4-bit embedding / head), shapes tiny (hidden 128, KDA head_dim 32, MLA
rank 128: sdpa_vector), real widths (hidden 4096 = 16384-wide streams, KDA head_dim 128, rank 512 / nope 256 / v 256,
32 x 128 indexer, experts 2048 x 4096) and 64 MLA heads over 564 positions; 40 positions with the pool selection active
from position 11: **every stream and logit bit-identical** to the Python runtime (teacher-forced), greedy tokens equal;
float32 activations bit-identical on every shape; chunked prefill = per token; streamed (LRU / SLRU / prefetch, top-k + 1
slots per layer, a pinned hot set) = resident; a `suc compress` container (expert-stack codebooks, `hot_experts.json`
`auto`) resident = Python and streamed = resident.

### Results (M3 laptop, 2026-10-04)

All cross-checks bit-identical (logits and streams, bf16 and float32). Indicative only: the 2-layer real-width test
file decodes at 2.4 ms/token (GPU 2.0 ms); no claim for the real model.

### Not supported / limits

- RoPE (qk_rope_head_dim > 0), shared indexer top-k, `index_kpool_always_select_tail = false`, routers other than
  sigmoid / one group, n_shared_experts != 1, attention bias, tied embeddings (all fail loudly); MTP and vision dropped.
- Long contexts: the fallback softmax above 4096 keys uses MLX's looped softmax (1024 threads, verified on the M3 in
  milestone 6); MLX's two-pass sdpa_vector (ranks <= 256 at long contexts) is not ported (the real rank 512 never takes
  it). The 'd'-class (M1 Ultra) GEMM tile table only matters for float32 activations with K % 16 != 0; bf16 shapes of
  the real model are device-independent.
- No pipelined decode (one command buffer per step, plus one host sync per MoE layer when streamed). Batched prefill:
  see 'Fast batched prefill' below.

### Mac Studio: the real file

```bash
cd ~/Projects/SUC-LM && cpp/build.sh ~/suc-local/cpp-build
~/suc-local/cpp-build/suc-tests && ~/suc-local/cpp-build/suc-metal-tests
.venv/bin/python -m pytest tests/unit/test_cpp_glm5.py -q
# owner, once: sudo sysctl iogpu.wired_limit_mb=116000
F=experiments/artifacts/compress/compress_glm5.3_flash/container/model.suc1   # 119.9-GB streamed file; the chosen tcq file is resident (no --expert-budget)
~/suc-local/cpp-build/suc1-generate $F --tokens 5,6,7,8 --steps 64 --expert-budget 95G --hot-experts auto \
  --json logs/glm53_cpp_$(date -u +%Y%m%dT%H%M%SZ).json
~/suc-local/cpp-build/suc1-generate $F --prompt "Explain sparse attention briefly." --chat --no-think --stop-eos \
  --steps 256 --max-context 4096 --expert-budget 95G --hot-experts auto --expert-prefetch
```

Sequential, never alongside other GPU jobs. The C++ never tunes: if the tuning cache lacks the glm5_next sizes, run
`suc generate $F --tokens 1 --steps 1 --expert-budget 1.02e11` once first (the Python CLI takes bytes). Bit-exactness
against Python on the real file: the teacher-forced comparison of `tests/unit/test_cpp_glm5.py` (`--logits-out` /
`--hidden-out` vs `run_tokens` of `load_streamed_suc1_model`) with prompts below ~4,000 tokens.

Expected memory (estimate from the compress plan, ~120 GB file: routed experts 114.2 GB, the rest ~5.8 GB; the stderr
`memory:` / `expert streaming:` lines report the measured values): GPU views = the file minus the routed experts
≈ 5.8 GB (all resident after the first step); execution arrays built at load ≈ 2 GB (per-expert sign tables ≈ 0.5 GB,
the experts' column residuals ≈ 1.5 GB, which stay resident); KV ≈ 11.7 KB per position (0.38 GB at 32k) + KDA states
0.15 GB; expert pools = `--expert-budget` (95 GiB = 102 GB ≈ 89 % of the routed experts, half of it pinned hot set).
Total ≈ 110 GB: inside the 111 GB runtime budget of the compress config, only with the raised wired limit; not a
32 GB Mac. Measured: 9.1 tok/s with a 91 % expert hit rate (the
resident tcq file: 22.2 tok/s; table at the top).

## Trellis-coded (tcq) tensors

SUC1 representation `tcq` (docs/format.md §5.5, docs/low_tier_codec.md; Python: `suc.kernels.tcq_matmul`,
`SucLinear` / `SucSwitchLinear` in tcq mode): per row, sequences of T weights coded as tail-biting little-endian
bitstreams of NW = T·k/32 words (codes `1mad`, `3inst`, `pair3`; V = 1 or 2 weights per step; L-bit states), one f16
scale per row, the one-sided input rotation, optional column / CSR residual.

- **Kernels**: `cpp/metal/gen_suc_kernels.py` emits `tcq_matmul._TCQ_HEADER`, the `vec` (`_SRC`) and `mma` (`_MMA_SRC`)
  bodies verbatim as `suc_tcq_vec` / `suc_tcq_mma`, behind `_SWITCH_PROLOGUE` as `suc_tcq_switch_{vec,mma}` (stacked
  expert arrays) and behind a C++-only offset-table prologue as `suc_tcq_switch_{vec,mma}_tab` (experts read in place,
  empty slots of mixed-codec sub-stacks, streamed slot pools). The f16 row scales are bound straight from the file
  (`float(half)` is exact, so this equals SucLinear's float32 copy); `cst` = float32 [mu_0, c_0, mu_1, c_1] is computed
  on the host from the directory (`TcqLayout::from_params`, = `tcq_matmul.kernel_consts`: c_j = float(gain / code_sd[j])
  with the division in double).
- **Reference decoder** (`src/decode.cpp`): states from the stream windows, raw code values exact in double,
  float32((raw - mu) / sd) · float32(gain) · row scale, then the shared Hadamard inverse: bit-identical to
  `SucModel.decode_tensor`.
- **`metal::Linear`**: the streams are the stored `chunks` payload viewed as (out, in/T · NW) u32 (zero-copy). Dispatch =
  `tcq_matmul.launch_plan`: batch tiles of <= 16 rows; a tile of bt rows runs `vec` below 5 rows (SG = 8, R = 4 for
  bt <= 2 else 2), `mma` from 5 (SG = 4, RT = 4, NB = ceil(bt / 8)). Rotations, residual and output dtype are the stages
  of the vq Linear. Never fused (`FusedLinear::key` = ""; tcq members still share the input rotation of their group).
  `encode_rows_exact` runs vec tiles of up to 4 rows: no row's arithmetic depends on the other rows of the tile, so
  every row equals its one-row (decode) launch bit for bit (`suc1-linear --rows-exact`, checked by the tests).
- **Expert stacks**: `SwitchLinear` takes uniform tcq stacks (Stacked when the experts' streams / scales are consecutive
  in the file, else Table; Streamed for SSD expert streaming: the same two file ranges per expert as vq), shared input
  signs (1, in) when every expert has the same `seed_in` (what `suc compress` writes for tcq experts), else per-row
  sign tables. Decode = the gather path (BT = 1, vec); expert tiles of the batched prefill up to 4 rows (vec, row-exact).
  Mixed-codec stacks (`ExpertStack`): tcq experts form their own sub-stack(s) (one per tcq layout key) next to vq /
  additive / native MXFP4 ones, as `switch_from_encodings`.
- **Graphs**: qwen3 / qwen3_moe, qwen4_exp / qwen3_5, deepseek_v4 and glm5_next accept tcq projections and experts.

Tests: `tests/unit/test_cpp_tcq.py` (decoder; Linear vs the CPU reference and bit-exact vs Python at f16 / bf16 on both
kernels and forced paths; row-exact batches; stacks in every layout, tiles, mixed vq + tcq stacks, a MoE block; the six
graph families teacher-forced against Python, deepseek_v4 batched prefill and streamed experts, glm5_next streamed
experts). Tools: `suc1-linear --rows-exact`, `suc1-moe --stack P --mixed routed|all --tiles BT`.

Limits: the kernels take V in {1, 2}, T a multiple of 128 dividing in, kv <= L <= 24 (the SUC1 compute codes; `hyb` is not
a SUC1 representation). No per-device tuning (fixed launch constants of the Python kernels). Speeds: the DeepSeek / GLM tcq rows of the table
at the top.

## suc1-server: OpenAI-compatible HTTP server

`suc1-server FILE [options]` (`tools/suc1_server.cpp`) loads one SUC1 file once and serves it over a minimal,
dependency-free HTTP/1.1 server (POSIX sockets; keep-alive; `Content-Length` request bodies; streamed responses use
chunked transfer encoding). Every model family of `suc1-generate` runs (qwen3, qwen3_moe, qwen3_5, qwen4_exp,
deepseek_v4, glm5_next), with all of its runtime and expert-streaming flags (`--expert-budget`, `--hot-experts`, ...).

```bash
~/suc-local/cpp-build/suc1-server model.suc1 [--host 127.0.0.1] [--port 8080] [--model-name NAME] \
    [--max-context 4096] [--max-tokens 256] [--temperature T --top-k K --top-p P --min-p P] \
    [--no-think] [--reasoning-effort LEVEL] [--tokenizer tokenizer.json] [--quiet] [suc1-generate runtime flags]
```

- Endpoints: `POST /v1/chat/completions`, `POST /v1/completions` (`"stream": true` = server-sent events, one
  `chat.completion.chunk` / `text_completion` per text delta, a final chunk with `finish_reason`, a usage chunk with
  `stream_options.include_usage`, then `data: [DONE]`), `GET /v1/models`, `GET /health`. Errors use the OpenAI
  `{"error": {...}}` body (400 for unsupported parameters).
- Parameters: `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`
  (string or list; the stop text is excluded), `n` = 1; chat: `messages` (system / developer / user / assistant, string
  or text-part content), `reasoning_effort`, `chat_template_kwargs.enable_thinking`. Completions take a text prompt or a
  list of token ids (one prompt per request). Unset sampling parameters take the server's flags (default: greedy).
  Not supported: `n` > 1, logprobs, echo, suffix, penalties, tools.
- Generation: requests are serialized (batch 1, a mutex around the model; each connection has its own thread, so
  `/health` answers during a generation), the KV cache / states are reset per request. Greedy qwen3 / qwen3_moe
  requests run `suc1-generate`'s pipelined decode (`Model::generate`, `--depth` command buffers in flight, its
  `on_token` callback consumes each token and sends the SSE chunk while the next step runs on the GPU); sampling,
  `--sync` and the other families decode synchronously (one command buffer per token; the sampler of
  `suc/sampling.hpp`, seeded per request, a random seed when `seed` is unset), as `suc1-generate` does. It stops at the architecture's `eos_token_id` and the tokenizer's end tokens (`finish_reason: "stop"`, the
  end token counts in `usage.completion_tokens` but is not in the text), at a stop string, or at `max_tokens` / the
  context (`--max-context`, default 4096: the KV cache is allocated for it at load; `finish_reason: "length"`).
  Streamed deltas hold back an incomplete UTF-8 tail and a possible stop-string prefix, so they join to the
  non-streamed text.
- Chat template: the per-family template of `suc1-generate --chat` (now `chat_prompt_text` in `tools/tool_common.hpp`,
  shared by both tools). One user turn (optionally after a system turn) renders exactly as before; earlier turns of a
  multi-turn history use the same per-turn markup, with an assistant turn's `<think>...</think>` reasoning dropped (as
  the Qwen3 template does; for qwen4_exp, deepseek_v4 and glm5_next the history markup is not checked against their
  reference templates).
- Tokenizer: required (`--tokenizer`, else the tokenizer.json the SUC1 `tokenizer_id` pins). Prompt ids >= the model's
  vocab are rejected.
- Speed: greedy decode ms/token (logged per request) equals `suc1-generate`'s on the tiny test file (M3: 0.18 vs
  0.18 ms/token non-stream and stream; 0.83 ms/token with `--sync`, the former synchronous loop); the test checks
  <= 1.1x.

Test: `tests/unit/test_cpp_server.py` (also a CTest entry) starts the server on a tiny random Qwen3 file with a
synthetic byte-level tokenizer and drives it with the `openai` Python client (dev dependency): greedy text of
`/v1/completions` (token-id and text prompts) and `/v1/chat/completions` (with / without system turn, `enable_thinking`
false) equals `suc1-generate`'s tokens cut at the first end token, streamed deltas join to the same text, prompt token
counts equal `suc1-generate --chat`'s prompt, seeded sampling equals `suc1-generate --temperature/--top-k/--top-p
--seed`, stop strings, the context bound and request errors.

## Fast batched prefill of the frontier graphs (2026-10-10)

Owner decision: the batched prefill may differ from token-by-token decode by small rounding differences (no
bit-identity requirement); the size of the difference is measured instead (`scripts/prefill_batch_vs_token.py`:
per-position KL / top-1 of every prompt row and of a teacher-forced continuation, report §1c). The deepseek_v4,
glm5_next, qwen4_exp and qwen3_5 graphs therefore run, by default, a batched prefill built on simdgroup-matrix
kernels; the previous row-exact batched prefill stays available as `--prefill exact` (or `$SUC_PREFILL_EXACT=1`,
`*Options::exact_prefill`), `--prefill token` = decode steps. The Qwen3 / Qwen3-MoE graph is unchanged (already not
row-exact).

- **GEMM kernels** (`cpp/metal/suc_model_kernels.metal`, `suc_mm_*`; host `encode_mm_affine` / `encode_mm_dense` /
  `encode_tcq_mm` in `suc/metal.hpp`): Y (n, out) = X (n, in) · Ŵᵀ over all chunk rows, 8 x 8 simdgroup MMAs,
  float32 accumulation, one rounding. Threadgroup tile BM (8 / 16 / 32 / 64) rows x 64 outputs, k steps of 32; the
  weight tile is decoded into threadgroup memory (affine: T(q·s + b) as the decode gemv; dense: T(w), or the stored
  value for float32 outputs such as the router / compressor gemvs; tcq: random-access decode of each weight's
  trellis window). Used for every affine / dense projection of a chunk (n >= 8, in % 32 == 0); SUC projections run
  `Linear::encode(..., Path::Prefill)` (the "mma" lookup in 16-row tiles; tcq tensors: `suc_tcq_mm` over all rows).
- **Expert tiles** (`Tiles::exact = false`): up to 16 rows per tile (tcq sub-stacks run each 16-row tile as two
  8-row tcq_matmul mma tiles, whose 16-row tile was 3.5x slower, multiplying float32 matrices instead of tcq_matmul's
  activation-dtype ones: with bf16 matrices the tiny glm5 tcq file reached per-position KL 0.245), the vq / additive lookup on its mma kernel from the tuned `mma_min_batch` rows (the switch mma kernel
  now gets NB = ceil(bt / 8): with NB fixed at 1, 16-row tiles computed only half of each tile), native MXFP4
  sub-stacks on `ds4_mxfp4_mm_tile` from 8 rows (merging consecutive tiles of one expert into one threadgroup row of
  <= 64 rows when the host plan is known). `plan_tiles` keeps its cost model (α = 0.3: smaller α, i.e. more 16-row
  tiles, was slower).
- **deepseek_v4 attention** (`attn_chunk_fast`): q / kv projections as GEMMs, q norm + RoPE and kv RoPE over all rows
  (`ds4_qnorm_rope_rows`, `ds4_rope_rows`), keys in a linear buffer (the window before the chunk gathered from the
  ring, then the chunk's FP8-simulated keys, `ds4_kv_store_rows`), both compressors once per chunk (`comp_chunk`: one
  GEMM per compressor weight into float32 rows, the groups completed inside the chunk pooled in one
  `ds4_comp_pool_rows` launch, normed, roped, Hadamard + FP4 / FP8, group buffers left as the decode step expects), ONE
  `ds4_attn_chunk` launch per layer (one simdgroup per row and head over its window keys and visible entries), the
  ring refreshed with the chunk's last window keys, inverse RoPE over all rows, `wo_a` / `wo_b` as GEMMs. The kernel
  keeps the decode step's roundings (sink in T; for the unfused 64-head path q' = T(T(scale) q), scores in T and the
  probabilities rounded to T before p·V, two passes): with them the chunk attention is bit-identical to the decode
  steps on the tiny test files; without the sink rounding KL reached 4e-2 at single positions, without the probability
  rounding 1.2e-2 on the wide file. Layers whose indexer would select (entries > index_topk, i.e. contexts beyond
  4 · index_topk tokens) keep the sequential per-token attention.
- **qwen4_exp / qwen3_5**: chunk projections as GEMMs / mma lookups, expert tiles as above; the recurrent parts as
  before (Gated DeltaNet in one launch per chunk, full attention per token).
- **glm5_next** (`encode_chunk_fast`, resident and streamed): projections as GEMMs / mma lookups, the router as a
  float32 GEMM, routed experts in MMA tiles (passes of <= 2048 tile rows, `$SUC_GLM_EXPERT_PASS_ROWS`), row-batched mHC
  kernels, KDA causal conv over the chunk in one launch, the gated-delta recurrence of the whole chunk in one launch
  per layer, MLA / DSA with batched latents, indexer LayerNorm and pools, the absorbed queries and per-head value
  projections as GEMMs (W_k transposed per DSA layer at first use, ~184 MB on the real file) and one `glm_attn_chunk`
  launch per layer with each row's DSA selection. With `SUC_FAST_OFF=ptra` (only these row-batched kernels left) the
  batched path is bit-identical to token prefill on the real file (512 positions). `$SUC_GLM_PROFILE=1`: stage profile.
  Server: batched by default with chunks of 512.
- Chunk defaults: superseded on 2026-10-12 (next section: 4,096 / 1,024 tokens); `--prefill exact` keeps the
  previous chunks.
- Size of the difference (dev runs on the M1 Ultra, `prefill_batch_vs_token.py`): the MoE frontier models are far
  more sensitive than Qwen3-8B (KL 1e-3): DeepSeek-V4-Flash tcq 512 tokens KL mean 0.031 / top-1 94.1 %, GLM-5.3-Flash
  tcq 0.011-0.013 / 93.9 %. Any single fast part alone (even only the router GEMM, whose scores differ from the gemv's
  in the last float bits) gives KL ~5e-3 on DeepSeek and ~1.2e-2 on GLM: near-tie expert routing across 43 / 42 MoE
  layers amplifies float-level changes; with every fast part off the paths are bit-identical. The queue
  (`scripts/studio_fast_prefill_queue.sh`) records the 2,048-token values for every model (report §1c).
- Per-row logits: `--logits-out` together with an explicit `--prefill batch|exact` computes every prompt row's logits
  inside the batched chunks (`*Model::forward_rows`), which `scripts/prefill_batch_vs_token.py` compares with
  `--prefill token`. Without `--prefill`, `--logits-out` still runs decode steps (the cross-checks against Python).
- Diagnostics: `$SUC_FAST_OFF` (letters of parts to keep row-exact: p projections, a attention, t expert tiles, r router),
  `$SUC_MM_BM`, `$SUC_TCQ_MM=0`, `$SUC_TCQ_RUNS=1`, `$SUC_MXFP4_RUNS=0`, `$SUC_DS4_ATTN_RND`.

### Prefill after the ds4 comparison (2026-10-11, docs/prefill_vs_ds4.md)

- **deepseek_v4: batched lightning indexer.** Chunks whose CSA rows pass `index_topk` entries (contexts beyond
  4 · 512 tokens) no longer fall back to token-by-token attention.
  - `attn_chunk_fast` projects the indexer queries / weights of all rows as GEMMs, then runs rope / Hadamard / FP4
    over all rows.
  - Scores: `ds4_index_scores_mm`, 32-entry × 64-head simdgroup-matrix tiles. `$SUC_IDX_MM=0` falls back to
    `ds4_index_scores_rows`, the decode arithmetic per entry.
  - Selection: `ds4_index_select_rows`, `q4_index_select`'s exact threshold search and tie rule, written as an
    ascending entry list per row.
  - `ds4_attn_chunk` attends the window plus that list.
  - `$SUC_FAST_OFF=i` restores the sequential path for those layers.
- **Row-batched launches (`$SUC_FAST_OFF=h` = the per-row launches, same bits).** Every row-local kernel that ran once
  per chunk row now runs once per chunk:
  - deepseek_v4: hyper-connection pre and post, routing, MoE output, embedding.
  - qwen4_exp / qwen3_5: hyper-mix norm, low-rank act and mix; inject; routing; shared-expert add; dense norm / add;
    the Gated DeltaNet q / k norms.
  - The kernels take the row from the grid. The chunk's last row now lives in the ring like the others and is copied
    to `hs` per layer.
- **qwen4_exp: batched full attention** (`full_attn_core_rows`; `$SUC_FAST_OFF=a` = per token). Over all chunk rows:
  q / k norms and RoPE with per-row positions, the indexer keys, the blocks completed in the chunk in one
  `q4_block_key` launch, `q4_index_scores_rows`, the top-k blocks of every row as ascending block lists
  (`ds4_index_select_rows`), and one `q4_sdpa_vector_sel` launch over (heads, rows) with each row's causal key count.
  Rows past the budget visit only their selected blocks plus the incomplete tail (list mode), not every key with a
  mask test. This is always the single-pass kernel; decode
  switches to two passes from 1,024 keys on the M1 Ultra. The sigmoid gate also runs over all rows.
- **glm5_next: DSA attention over the selected pools only.**
  - `glm_attn_chunk` walks each row's ascending pool list plus the incomplete tail, instead of all `p + 1` positions
    with a mask test. Same keys, same order, same bits.
  - The per-row `q4_index_select` launches became one `ds4_index_select_rows`.
  - The pool scores use `ds4_index_scores_mm` (MODE 1).
- **Scores kept in threadgroup memory.** `ds4_attn_chunk` and `glm_attn_chunk` (`RND = 1`) compute each score once and
  keep it, a T value, in threadgroup memory: up to 2,048 / 2,560 keys per head, 16 / 20 KB. Previously they
  recomputed it in each pass. Same bits.
- Stage profile: `$SUC_DS4_PROFILE=1` now also splits `attn_proj_comp`, `attn_indexer`, `attn_kernel` and `attn_wo`.
- **Default chunks** (suc1-generate, when `--chunk` is not given, at most the prompt): 4,096 tokens for deepseek_v4 and
  glm5_next, and 1,024 beyond 8,192-token prompts. A 32k prompt in chunks of 4,096 ran out of GPU memory on the
  M1 Ultra with the 97 GB DeepSeek file. qwen4_exp / qwen3_5 use 1,024. These come from the 8k-token sweeps on the M1 Ultra: DeepSeek tcq 134.5 →
  147.9 tok/s, GLM tcq 99.8 → 105.9, Qwen3.8 side8 GPU 3.03 → 2.80 ms/token.
  `suc1-server`: chunks of 64 (qwen4_exp, qwen3_5, deepseek_v4) and 512 (glm5_next) unless `--chunk` is given.
- **qwen4_exp n-gram rows**: the host gathers of a chunk's tokens run in parallel. With a cold page cache these are
  random SSD reads; the serial loop cost up to ~1.9 ms per prompt token of host time at 8k on the Studio.
  `prefill_ngram_s` is now in `--json`.

Tests: `test_fast_batched_prefill_close_to_token` in `tests/unit/test_cpp_{deepseek_v4,qwen4,qwen3_5,glm5}.py`
(every position's KL / top-1 against token prefill on the tiny random files, chunks splitting compressor groups /
pools, forced tile heights, streamed experts); the bit-identity tests now pass `--prefill exact`.

## Open items

- Batch > 1 decode (one request at a time today).
- Prefill: the routed-expert tiles (8–16 rows vs ds4's 32-token dequantized tiles) and the scalar chunk attention
  are the remaining gap to ds4 (docs/prefill_vs_ds4.md).
- Decode: Qwen3.8-Flash-Next is 16 % slower than ds4 (docs/decode_speed_vs_ds4.md).
- CMake mirrors `build.sh` but is not part of the verified build.
