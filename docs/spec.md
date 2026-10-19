# Semantic Universal Compression (SUC)

Italiano: [spec-IT.md](spec-IT.md)

**Implementation specification for an LLM coding agent**  
**Development hardware:** Apple Silicon Mac Studio with an Ultra chip and ≥ 128 GB of unified memory — M1 Ultra and later (M2 Ultra, M3 Ultra, … up to the current M5 Ultra). The project's reference measurements were taken on an M1 Ultra 128 GB, the oldest supported configuration; newer Ultra chips (more bandwidth and GPU cores) are expected to be faster but are not measured here.  
**Primary goal:** research and prototype a model representation that minimizes storage/memory while preserving model behavior and enabling direct inference from the compressed representation, without first reconstructing the entire FP16 weight matrix.

---

## 1. Mission

Build a research system that tests this hypothesis:

> An LLM can potentially be represented much more compactly than conventional low-bit scalar quantization by combining semantic/data-aware sensitivity, vector/codebook quantization, adaptive bit allocation, residual/outlier preservation, optional entropy coding, and hardware-aware fused inference.

The system must optimize **model behavior per stored bit**, not merely numeric weight reconstruction error.

Formally, given original model `M` and compressed model `Mq`:

\[
\min \; Size(M_q)
\]

subject to:

\[
D_{semantic}(M,M_q) \le \epsilon
\]

and optionally:

\[
Latency(M_q) \le L_{max}
\]

\[
PeakMemory(M_q) \le M_{max}
\]

\[
Energy(M_q) \le E_{max}
\]

The project must distinguish four different goals:

1. **Lossless compression:** exact reconstruction of original weights.
2. **Lossy quantization:** weights are approximated.
3. **Storage compression:** compressed weights are expanded before inference.
4. **Compute-native compression:** the inference kernel consumes the compressed representation directly.

The primary research objective is #4.

---

# 0. Design decisions (amendments, 2026-09-30)

These decisions were agreed with the project owner after review of the original spec. **Where they conflict with a later section, this section wins.** Each later section affected carries a `(see D#)` note.

### D1 — Two parallel tracks instead of one sequence

The original order (former §51, since rewritten per D1) required full-model compression phases before the single-layer kernel milestone, contradicting the §43 gate. Work is split into:

- **Quality track** — analysis, calibration, codebooks, residuals, allocation. Evaluated by decoding in Python and running the *reconstructed* weights in MLX. Full-model perplexity/KL is allowed here from the start; it measures representation quality, not runtime.
- **Kernel track** — direct execution over compressed data. Starts as soon as a basic vector codebook (Phase 4) exists. The §43 single-linear-layer gate applies to **this track**: no full-transformer direct execution until the single-layer benchmark is sound.

The tracks meet at Phase 15 (transformer integration).

### D2 — C++ runtime deferred, not dropped (done)

> **Status 2026-10-14: delivered.** The C++ runtime was built at the end of the kernel track as planned and is now the
> main inference path: SUC1 reader / decoder, Metal kernels generated from the Python ones, graphs for Qwen3 dense /
> MoE, Qwen3.8 (`qwen3_5`, `qwen4_exp`), DeepSeek-V4 and GLM-5.3, tokenizers, batched prefill, SSD expert streaming and
> the OpenAI-compatible `suc1-server` ([cpp/README.md](../cpp/README.md)). It also powers the macOS app SUC Studio
> (`SUCStudio/`). "Deferred" below records the original plan (2026-09-30).
> Runtime update 1.4.7: [inference behavior](optimization/INFERENCE_1.4.7.md), [combined measurements](optimization/COMBINED_RESULTS.md) and [qualification status](optimization/PROGRESS.md). This adds runtime optimizations without changing D2, the SUC1 representation or the memory policy.

Kernel prototyping uses MLX custom Metal kernels (`mx.fast.metal_kernel`) called from Python. The standalone C++ runtime (§18, Phase 11) remains a required deliverable but moves to the end of the kernel track, after the fused kernels are validated. The strong 4-bit baseline on Apple Silicon is MLX's own affine quantization (`mx.quantize` / fused quantized matmul), in addition to AWQ/GPTQ where tooling permits.

### D3 — Incoherence processing and codebook fine-tuning are first-class components

Known sub-3-bpw methods depend on two ingredients the original architecture omitted:

- **Incoherence processing** — randomized Hadamard (or similar orthogonal) rotations of weights/activations before quantization (QuIP/QuIP#), to suppress outliers and make blocks more "codebook-friendly". The rotation must be either fused into the kernel or folded into adjacent layers; its runtime cost is measured.
- **Codebook / representation fine-tuning** — after initial fitting, tune codebooks (and optionally scales/norm parameters) against a semantic objective on calibration data (AQLM-style block/global fine-tuning, PV-tuning-style discrete updates).

Both are optional, switchable components and both get a row in the ablation table (§26): `SUC - incoherence`, `SUC - fine-tuning`. For targets at or below 3 bpw the reference baselines must include at least one of **AQLM** or **QuIP#** (published numbers on the same model/eval, or a local run), not only a 4-bit method.

> **Status 2026-10-14:** incoherence processing is in the final recipe as a one-sided (input-side) randomized Hadamard rotation; codebook fine-tuning was implemented (Q6) but is optional and unused in the chosen files; both have ablation rows ([final_report.md](final_report.md) §3.6). AQLM / QuIP#: published numbers only (see D7).

### D4 — Codebook size is bounded by the hardware from day one

At batch=1 decode is memory-bandwidth bound (M1 / M2 Ultra ≈ 800 GB/s theoretical, M3 Ultra ≈ 819 GB/s, more on later chips). A codebook the kernel reads per output tile must fit in threadgroup memory (32 KB per threadgroup on Apple GPUs). Default ceiling: **codebook bytes = K × G × bytes_per_entry ≤ 16 KB** per codebook actively used by a threadgroup (e.g. K=256, G=8, FP16 = 4 KB). Configurations exceeding the ceiling may be studied in the quality track but are flagged as "not execution-viable" unless a kernel demonstrates otherwise.

### D5 — Sensitivity uses cheap proxies first

Per-block perturbation tests (§9 T2.3) do not scale to 7B+. Sensitivity is estimated first with cheap proxies accumulated during calibration:

- diagonal / blockwise input second moment `diag(XᵀX)` (GPTQ-style Hessian proxy) per linear layer;
- optionally empirical Fisher (squared gradients) of the loss w.r.t. weights.

Real perturbation tests are run only on a **sampled subset** of tensors/blocks to validate and calibrate the proxy (report rank correlation proxy vs measured ΔKL). The proxy is used for allocation only if that correlation is reported.

### D6 — Allocation granularity follows the kernel

Mixing representations *inside* a tensor at fine block granularity breaks the regularity the kernel needs. Therefore:

- the **base representation** (codebook family, G, K, bits) is chosen per **tensor** or per **row-group** (contiguous output rows, size a multiple of the kernel tile);
- **fine-grained** (per-block / per-element) decisions are restricted to residuals and outliers (Phase 6), which the kernel handles as a separate sparse pass.

### D7 — The model is a configuration input, not a code constant

The model to compress is selected by configuration (Hugging Face repo id or local path + revision). Code must not hard-code a model or layer names; architecture-specific details (which linear layers are eligible, naming, attention layout) live in a small per-family adapter. Two roles exist:

- **dev model** — small (≈0.5–2B), used for fast iteration and CI-scale tests;
- **target model** — the model being seriously optimized (first serious target 7B/8B, §7 T0.1).

Current plan (2026-09-30), two targets in sequence:

| Stage | Model | Role |
|---|---|---|
| Development / tests | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B` | dev: minute-scale iteration, automated tests |
| Foundation, quality track Q1–Q5, §43 kernel gate | `Qwen/Qwen3-8B` | target 1 (dense) |
| From Q6–Q7 (fine-tuning, allocator) onwards | `Qwen/Qwen3-30B-A3B` | target 2 (MoE, ~3B active) |
| After the project (run by the owner) | ≈27B dense, then larger dense and MoE models (up to ≈180B MoE) | out of project scope |

Phase 0 already includes loading `Qwen3-30B-A3B` through the adapter and its BF16 / MLX-4-bit baselines, so D8 is exercised on a real MoE early. Few AQLM/QuIP# numbers are published for Qwen3, so those baselines must be run locally.

Supported families are those loadable by `mlx-lm`; adding a family means adding an adapter plus a numerical identity-conversion test. Every result records the exact model id, revision and weight hash (§7 T0.4).

---

> **Status 2026-10-14:** besides the two targets, the pipeline compressed Qwen3.8-27B (dense) and the MoE models Qwen3.8-Flash-Next (177 B), DeepSeek-V4-Flash (284 B) and GLM-5.3-Flash (313 B) on the 128-GB Studio — chosen files in [compression_report.md](compression_report.md) §1. Qwen3-1.7B was never used (dev model: Qwen3-0.6B). AQLM / QuIP# were not run locally: published numbers only ([comparison_published.md](comparison_published.md)).

### D8 — Scalability built in from Phase 0

> **Status 2026-10-14:** the pipeline has since compressed Qwen3.8-27B and three 177–313 B MoE models
> (Qwen3.8-Flash-Next, DeepSeek-V4-Flash, GLM-5.3-Flash) layer-streamed on the 128-GB Studio
> ([compression_report.md](compression_report.md)).

The project targets Qwen3-8B (dense) and Qwen3-30B-A3B (MoE) and ends with the C++ runtime and the final report (decision 2026-10-01); scaling to ≈27B dense and beyond (up to ≈180B MoE, which does not fit in 128 GB at BF16) is done by the owner after the project with the finished pipeline — so the pipeline must run unattended end to end on such models. The pipeline must not need re-architecting to reach them:

- **Layer-streaming pipeline.** Weights are read lazily (mmap of safetensors shards), and calibration, sensitivity estimation and compression proceed **one transformer block at a time** (GPTQ-style sequential pass: propagate calibration activations through the already-processed block, then free it). No step may require the full model resident in BF16/FP16. Peak memory of each pipeline stage is measured and reported.
- **Architecture adapter interface ready for MoE and non-standard attention.** The adapter enumerates eligible linear tensors with roles (`attn_q/k/v/o`, `mlp_gate/up/down`, `expert[e].*`, `shared_expert.*`, `router`, linear-attention projections, …). Router/gating weights default to high precision. Per-expert activation counts are collected during calibration and exposed to the allocator; calibration size must be sufficient for rarely-routed experts (report min tokens seen per expert).
- **Full-precision source only.** Input is the original Hugging Face checkpoint (safetensors, BF16/FP16) plus tokenizer. Already-quantized artifacts (GGUF, MLX 4-bit, AWQ/GPTQ, FP8) are never used as input — only as comparison baselines. Source shard hashes are verified and recorded.

---

# 2. Core idea

The user's original intuition was:

```text
weight value
    ↓
small index
    ↓
lookup table
    ↓
value
```

A scalar lookup table does not by itself beat strong 4-bit quantization. A table of 1000 scalar values requires about 10 bits per scalar index:

\[
\lceil \log_2(1000) \rceil = 10
\]

The promising generalization is:

```text
block of many weights
        ↓
one compact code/index
        ↓
codebook vector
```

For example, if one 8-bit index selects a 16-weight vector:

\[
8/16 = 0.5\;bits/weight
\]

before codebook, scale, residual, alignment, and metadata overhead.

This is vector/codebook quantization.

The critical research question is then:

> How much model behavior can be retained per stored bit, and can that compressed representation be used directly by the inference kernel?

---

# 3. What this project is and is not

## 3.1 It is

- a research prototype;
- a low-bit/vector/codebook compression framework;
- a semantic-aware bit allocator;
- an inference-kernel experiment;
- a benchmark platform for memory/quality/latency trade-offs.

## 3.2 It is not

- a claim that a new compression primitive has already been discovered;
- a claim that all LLMs can be compressed to 1–2 bpw without meaningful degradation;
- a claim that entropy coding automatically makes inference faster;
- a direct inheritance of Cover's Universal Portfolio theorem by neural networks.

The Universal Portfolio / information-theoretic concepts are the inspiration for the **allocation and universalization principle**, while vector quantization, low-bit LLM quantization and LUT-centric inference already have substantial prior work.

The research novelty, if any, must come from the **combination and objective**, not from pretending that codebooks or 2-bit quantization are new.

---

# 4. Technology choice for Apple Silicon Ultra (M1 Ultra and later)

## 4.1 Primary language stack

Use a hybrid stack.

### Python
Use for:

- model loading;
- calibration data;
- weight/activation statistics;
- codebook training;
- optimization;
- experiment orchestration;
- benchmark reporting;
- serialization tooling.

### C++ (deferred in the plan, delivered — see D2)
Use for:

- runtime;
- packed representation;
- memory layout;
- low-level reference kernels;
- eventual inference integration.

### Metal Shading Language
Use for:

- GPU kernels;
- lookup/decode/GEMM fusion;
- direct execution over compressed weights.

### MLX
Use as the first Apple-native research framework. Apple Silicon has unified CPU/GPU memory, and MLX is specifically designed around that model. This makes it an excellent prototyping environment for compressed representations that should be visible to CPU and GPU without ordinary device-copy semantics.

### llama.cpp / GGML

> **Status 2026-10-14:** used only as a comparison baseline (llama.cpp, ds4, LM Studio); no interoperability layer was built (§24).
Use later as an interoperability/runtime target, not as the first development environment.

The current llama.cpp project supports Apple Silicon through Metal and has a broad range of low-bit formats, making it an important final comparison target.

---

# 5. Architecture

```text
                         ORIGINAL MODEL
                              │
                              ▼
                    ┌────────────────────┐
                    │ Model Analyzer     │
                    └─────────┬──────────┘
                              │
               ┌──────────────┼─────────────────┐
               ▼              ▼                 ▼
         weight stats    activation stats   sensitivity
               │              │                 │
               └──────────────┼─────────────────┘
                              ▼
                    ┌────────────────────┐
                    │ Semantic Allocator │
                    └─────────┬──────────┘
                              │
                      per-block budgets
                              │
                              ▼
                    ┌────────────────────┐
                    │ Codebook Builder   │
                    └─────────┬──────────┘
                              │
                     codebooks + indices
                              │
                ┌─────────────┼────────────┐
                ▼             ▼            ▼
            residuals      outliers     metadata
                │             │            │
                └─────────────┼────────────┘
                              ▼
                    ┌────────────────────┐
                    │ Optional Entropy   │
                    │ Coding             │
                    └─────────┬──────────┘
                              │
                              ▼
                       SUC Model File
                              │
                ┌─────────────┴──────────────┐
                ▼                            ▼
        Reference decoder              Direct runtime
        (correctness)                  (C++ / Metal)
                │                            │
                ▼                            ▼
          reconstructed                fused inference
             weights                   on compressed data
```

---

# 6. Repository structure

> **Status 2026-10-14:** the layout as built differs: C++ runtime and its Metal sources in `cpp/`, Python Metal kernels in `suc/kernels/`, runtime in `suc/runtime/`, pipeline in `suc/pipeline/`, queues in `suc/orchestrate/`; tests in `tests/unit` and `tests/model`; docs include `format.md`, `benchmarks.md`, `final_report.md`, `compression_report.md`; the macOS app is `SUCStudio/`. No LICENSE file yet.

```text
SUC-LM/
├── README.md
├── LICENSE
├── pyproject.toml
├── CMakeLists.txt
├── docs/
│   ├── architecture.md
│   ├── format.md
│   ├── algorithms.md
│   ├── benchmarks.md
│   └── experiments.md
├── suc/
│   ├── io/
│   ├── models/
│   ├── analysis/
│   ├── quant/
│   ├── allocation/
│   ├── entropy/
│   ├── evaluation/
│   └── cli/
├── runtime/
│   ├── cpp/
│   └── metal/
├── tests/
│   ├── unit/
│   ├── numerical/
│   ├── model/
│   ├── runtime/
│   └── regression/
├── experiments/
│   ├── configs/
│   ├── scripts/
│   └── results/
└── benchmarks/
```

---

# 7. Phase 0 — Baseline and reproducibility

## Goal

Establish a trustworthy reference before implementing SUC.

## Tasks

### T0.1 — model loader

Support at least one Hugging Face decoder-only transformer.

Start with a 1B–8B model so experiments are fast. Use a 7B/8B model as the first serious target. Only after stability, scale to 30B/70B-class models.

### T0.2 — baseline runtime

Measure:

- parameter count;
- raw tensor sizes;
- FP16 size;
- baseline inference latency;
- tokens/s;
- prompt throughput;
- peak unified memory;
- model load time.

### T0.3 — baseline representations

> **Status 2026-10-14:** the strong 4-bit reference is MLX affine 4-bit, plus llama.cpp K-quants and ds4 GGUFs; AWQ / GPTQ were not run locally.

At minimum compare:

- FP16;
- INT8;
- INT4;
- one strong existing 4-bit reference such as AWQ or GPTQ where architecture/tooling permit.

### T0.4 — deterministic benchmark harness

All runs must capture:

```json
{
  "model": "...",
  "model_hash": "...",
  "git_commit": "...",
  "dataset_hash": "...",
  "seed": 12345,
  "runtime": "...",
  "timestamp": "..."
}
```

## Acceptance

- identity conversion gives negligible numerical error;
- the benchmark harness produces machine-readable JSON;
- all experiments can be reproduced from a configuration file.

---

# 8. Phase 1 — Weight structure analysis

## Goal

Determine whether the model has enough local structure for vector/codebook compression.

Do not assume that it does.

## Tasks

### T1.1 — scalar statistics

Per tensor calculate:

- min/max;
- mean/std;
- quantiles;
- histogram;
- kurtosis;
- outlier rates.

### T1.2 — block statistics

Test block sizes:

```text
8
16
32
64
128
```

For every block collect:

- L1/L2 norm;
- variance;
- max absolute value;
- sparsity;
- mean;
- normalized direction.

### T1.3 — block similarity

Estimate pairwise/clustering similarity using:

- Euclidean distance;
- normalized Euclidean distance;
- cosine distance;
- optionally activation-weighted distance.

### T1.4 — occupancy analysis

For candidate codebooks, measure:

- number of occupied codes;
- long-tail distribution;
- repetition;
- average nearest-codebook distance;
- p95/p99 distance.

### Deliverable

Generate a report:

```text
layer/tensor
block size
number of blocks
candidate K
entropy of assignments
reconstruction error
```

This phase decides whether vector coding is worth pursuing and which block sizes are plausible.

---

# 9. Phase 2 — Calibration and semantic sensitivity

*(see D5: proxies first, perturbation tests on a sampled subset)*

## Goal

Measure which compression errors actually matter to the model.

Weight MSE alone is not sufficient.

## Tasks

### T2.1 — calibration dataset

Start with roughly 1k–10k sequences, configurable by YAML/JSON.

Include multiple task types:

- natural language;
- coding;
- mathematics/reasoning;
- instruction following.

Do not use the final benchmark set for tuning.

### T2.2 — activation statistics

For important linear layers collect streaming statistics of their inputs:

- mean;
- variance;
- max magnitude;
- per-channel scale;
- second-moment/Hessian proxies where practical.

Do not persist full activation tensors unless needed for a specific experiment.

### T2.3 — local perturbation tests

For each layer/block:

1. create a compressed version;
2. run calibration samples;
3. measure change in layer output;
4. measure logit divergence;
5. measure cross-entropy/perplexity change.

### T2.4 — sensitivity score

Create a score such as:

\[
S_i = \frac{\Delta D_{semantic}}{\Delta E_i}
\]

where `Ei` is the injected representation error.

The exact score can be refined later. The first implementation must be transparent and measurable.

### Deliverable

Produce a sensitivity map per tensor/block:

```json
{
  "layer.12.mlp.down_proj": {
    "block_size": 16,
    "sensitivity_mean": 0.013,
    "sensitivity_p95": 0.041
  }
}
```

---

# 10. Phase 3 — Scalar codebook baseline

## Goal

Implement the user's original lookup-table concept as a controlled baseline.

Representation:

```text
codebook = [K scalar values]
indices  = one index per weight
```

Test:

```text
K = 4, 8, 16, 32, 64, 256, 1024
```

## Tasks

- implement k-means and k-means++;
- support per-tensor and per-block codebooks;
- support deterministic seeds;
- measure reconstruction error;
- measure perplexity;
- measure task accuracy;
- measure compressed size including codebook overhead;
- optionally entropy-code the indices.

## Expected outcome

This is primarily a sanity check. It is not expected to beat a strong 4-bit implementation dramatically.

---

# 11. Phase 4 — Vector codebook quantization

*(see D4: codebook size ceiling; D3: incoherence processing)*

## Goal

Represent groups of weights with one code.

For block `b`:

\[
\hat W_b = C[index_b]
\]

where `C` contains vector entries.

## Test matrix

```text
block size G = 4, 8, 16, 32
K = 16, 32, 64, 128, 256, 512, 1024
codebook dtype = FP16, BF16 where supported
```

## Tasks

1. Vector k-means.
2. Initialization strategies.
3. Codebook refinement.
4. Optional activation-weighted distance.
5. Compression size calculator with all overheads.
6. Decoder implementation.

## Required output

For every configuration report:

- nominal index bpw;
- actual bpw;
- codebook size;
- reconstruction error;
- calibration loss;
- held-out loss;
- model perplexity.

---

# 12. Phase 5 — Additive / multi-codebook representation

> **Status 2026-10-14:** implemented and measured; low-dimensional VQ (G=2, K=256) beat additive codebooks at every tested rate, so additive is optional and unused in the final recipe ([final_report.md](final_report.md) §2).

## Goal

Avoid requiring a huge single codebook.

Represent a block as:

\[
\hat W_b = C_1[i_1] + C_2[i_2] + ... + C_N[i_N]
\]

Example:

```text
codebook 1: 2-bit index
codebook 2: 2-bit index
codebook 3: 2-bit index
```

## Tasks

### T5.1 — greedy residual fitting

```python
residual = block
approx = 0
for codebook in codebooks:
    idx = nearest(codebook, residual)
    approx += codebook[idx]
    residual -= codebook[idx]
```

### T5.2 — optional joint refinement

Once the greedy version works, optimize all codebooks jointly where computationally justified.

### T5.3 — scaling study

Measure:

- 1 codebook;
- 2 codebooks;
- 3 codebooks;
- 4 codebooks.

Plot bits/weight vs semantic quality.

---

# 13. Phase 6 — Residual and outlier representation

## Goal

Spend extra bits only where they have high semantic value.

General form:

\[
W = W_{base} + R
\]

where `R` is sparse, low-bit, INT8, or FP16 for selected exceptions.

## Candidate residual modes

```text
none
sparse top-k FP16
sparse top-k INT8
small residual codebook
per-block high-precision exception
```

## Tasks

- compute residual after codebook approximation;
- rank residual elements by impact or magnitude;
- encode only a configurable fraction;
- benchmark quality/size trade-off;
- compare magnitude-based vs semantic-sensitivity-based outlier selection.

The preferred outcome is something like:

```text
98–99% of representation → very low-bit base
1–2% → extra precision
```

but the actual fraction must be measured.

---

# 14. Phase 7 — Semantic adaptive bit allocation

*(see D6: base representation per tensor/row-group; fine-grained only for residuals)*

## Goal

This is the central “Universal Allocation” component.

Instead of fixing the same representation for every block, choose the representation that produces the best quality under a global memory budget.

## Representation candidates per block

Examples:

```text
FP16
INT8
INT4
INT3
INT2
scalar-codebook
vector-codebook
multi-codebook
vector-codebook + residual
```

Every candidate must have measured/estimated:

```text
size
semantic loss
latency
```

## Optimization problem

Let `a_i` select a representation for block `i`:

\[
\min_{a_i} \sum_i D_i(a_i)
\]

subject to:

\[
\sum_i Size_i(a_i) \le B
\]

and optionally:

\[
Latency(a) \le L_{max}
\]

## First allocator

Implement a greedy marginal-value allocator.

For every possible upgrade:

\[
Value = \frac{\Delta Quality}{\Delta Bytes}
\]

Sort by value and spend the available budget.

## Advanced allocator

Experiment with multiplicative-weights style updates among candidate representation choices.

This is an experimental analogy to Universal Portfolio / expert aggregation. Do **not** claim that Cover's theorem directly proves optimality here.

---

# 15. Phase 8 — Entropy and universal coding

## Goal

Exploit non-uniform distributions of codebook indices.

Calculate:

\[
H(X) = -\sum_x p(x)\log_2 p(x)
\]

Test:

- fixed-width indices;
- Huffman;
- ANS/rANS;
- other practical entropy coding if justified.

## Important

Treat entropy coding first as a **storage/bandwidth optimization**.

Do not automatically place a variable-length bitstream inside the hot inference path.

Recommended design:

```text
SUC storage stream
       ↓
optional decode/cache
       ↓
fixed/aligned execution blocks
       ↓
Metal kernel
```

An alternative direct-streaming mode can be tested later.

---

# 16. Phase 9 — SUC binary container

> **Status 2026-10-14:** implemented as `format_version` 1 — byte-level spec in [format.md](format.md); `architecture_id` / `tokenizer_id` live in the JSON directory, not the fixed header; no entropy-table section (entropy coding is storage-only).

Design an independent `SUC1` format before touching GGUF.

## File structure

```text
SUC1
├── header
├── model metadata
├── tensor directory
├── codebook directory
├── codebook payloads
├── index payloads
├── residual payloads
├── optional entropy tables
└── checksum/footer
```

## Header fields

At minimum:

```text
magic = SUC1
format_version
architecture_id
tokenizer_id
number_of_tensors
number_of_codebooks
global_flags
```

`SUC1` is a fixed 4-byte file identifier and never changes for format evolution; all compatibility is governed by `format_version` (a new magic only for a fully incompatible redesign). Decision 2026-09-30.


## Tensor descriptor

```text
name
shape
layout
representation_type
block_size
codebook_id
index_bits
residual_type
data_offset
compressed_size
logical_size
```

## Requirements

- deterministic output;
- versioned;
- checksummed;
- 64-bit offsets;
- memory-map friendly;
- endian defined;
- alignment defined;
- no hidden assumptions in metadata.

---

# 17. Phase 10 — Python reference decoder

Implement a deliberately simple decoder.

```python
model = SucModel.load("model.suc")
W = model.decode_tensor("layer.0.mlp.down_proj.weight")
```

Purpose:

- correctness oracle;
- format validator;
- unit-test reference;
- generator of reconstructed tensors for semantic evaluation.

It is allowed to be slow.

The reference decoder must be simpler than the optimized runtime so that independent correctness checks are possible.

---

# 18. Phase 11 — C++ runtime data structures

*(see D2: deferred to the end of the kernel track; delivered — [cpp/README.md](../cpp/README.md))*

Suggested structures:

```cpp
struct SucCodebook {
    uint32_t entries;
    uint32_t vector_length;
    DataType dtype;
    const void* data;
};

struct SucTensor {
    Shape shape;
    uint32_t block_size;
    RepresentationType representation;
    const uint8_t* indices;
    const uint8_t* residuals;
    const SucCodebook* codebook;
};
```

Important goals:

- tightly packed indices;
- aligned data;
- contiguous codebook storage;
- low metadata overhead;
- zero-copy or mmap loading where practical.

---

# 19. Phase 12 — Metal kernel development

This is the critical runtime phase.

## Kernel A — correctness baseline

Implement:

```text
load index
→ lookup vector
→ reconstruct small block
→ dot product
```

inside the kernel.

Do not materialize the entire model in FP16.

## Kernel B — fused block operation

Target:

```text
activation block
+
compressed index block
+
codebook
        ↓
 fused dot product
        ↓
output
```

## Kernel C — multi-codebook

Compute additive codebook contributions without materializing full weights.

## Kernel D — sparse residual

Add the residual contribution only for selected entries.

---

# 20. Phase 13 — Eliminate the LUT bottleneck

A naïve codebook kernel may be too slow because of random lookup and repeated reconstruction.

Therefore test progressively:

### Strategy A

Direct per-element lookup.

### Strategy B

Block/vector lookup.

### Strategy C

Vectorized codebook access.

### Strategy D

Precompute codebook-derived partial dot products.

Conceptually:

```text
activation block
      ↓
partial products for codebook entries
      ↓
index
      ↓
select partial result
```

This is analogous to the direction taken by codebook-centric GEMM research: the goal is to avoid repeatedly reconstructing weight values during inference.

---

# 21. Phase 14 — Linear layer integration

Do not immediately rewrite the full transformer.

First replace one linear operation:

\[
Y = XW^T
\]

with direct SUC execution.

Compare four implementations:

```text
A = FP16 GEMM
B = strong INT4 baseline
C = SUC decode-then-GEMM
D = SUC direct Metal GEMM
```

Measure:

- numerical error;
- kernel latency;
- throughput;
- memory use;
- effective bandwidth.

This is the first meaningful go/no-go milestone.

---

# 22. Phase 15 — Transformer integration

Support mixed representation inside the same transformer:

```text
Layer A → FP16
Layer B → INT4
Layer C → SUC vector
Layer D → SUC + residual
```

Prioritize:

1. MLP projections;
2. attention projections;
3. output projection;
4. embeddings if beneficial.

Normalization and small tensors can remain conventional.

The allocator must be allowed to select a different representation per tensor/block.

---

# 23. Phase 16 — MLX integration

> **Status 2026-10-14:** exposed as CLI commands (`suc compress`, `suc generate`, `suc eval`, `suc perf`) rather than a Python API; backends are the Python / MLX runtime from codes and the C++ runtime (`suc1-generate`, `suc1-server`); no llama.cpp backend.

Expose a high-level Python interface:

```python
compressed = suc.compress(model, config)
outputs = suc.generate(compressed, prompt, backend="metal")
```

Proposed backends:

```text
reference
mlx
metal
llama_cpp (later)
```

Avoid duplicating model logic across backends. The compressed format and planner should remain backend-neutral.

---

# 24. Phase 17 — llama.cpp / GGUF interoperability

> **Status 2026-10-14:** not done (optional). No SUC tensor type for GGML / GGUF exists; llama.cpp, ds4 and LM Studio appear only as comparison baselines ([compression_report.md](compression_report.md) §5–§6).

Do this only after the standalone SUC runtime is stable.

Possible future path:

```text
SUC tensor type
      ↓
GGML/GGUF metadata
      ↓
CPU reference kernel
      ↓
Metal kernel
```

The current llama.cpp development model is relevant here because new tensor/runtime support requires conversion, graph handling, quantization tooling, and backend testing.

Do not use a GGUF extension as the first source of truth for SUC.

---

# 25. Phase 18 — Quality benchmark protocol

Every model representation must use the same protocol.

## Compression

Report:

```text
original size
compressed size
actual bits/parameter
codebook overhead
index overhead
residual overhead
entropy metadata overhead
```

Use:

\[
bpp = \frac{compressed\_bits}{parameter\_count}
\]

Do not report only nominal index bits.

## Quality

At minimum:

- held-out perplexity;
- next-token KL divergence;
- downstream benchmark score;
- coding task;
- math/reasoning task.

The calibration set must not be used as the final test set.

## Speed

Measure separately:

- load time;
- first-token latency;
- batch=1 token generation;
- batch >1 throughput;
- prompt processing;
- long context.

## Memory

Measure:

- file size;
- resident memory;
- peak unified memory;
- optionally effective memory bandwidth.

## Energy

If measurable:

```text
joules/token
energy/model-load
```

Otherwise explicitly state that energy was not measured.

---

# 26. Phase 19 — Mandatory ablations

> **Status 2026-10-14:** done on Qwen3-8B at 4.22 bpp — [final_report.md](final_report.md) §3.6.

*(see D3: add `SUC - incoherence` and `SUC - fine-tuning`)*

Test the final system and remove one component at a time:

```text
SUC
SUC - semantic sensitivity
SUC - vector quantization
SUC - multi-codebook
SUC - residuals
SUC - adaptive bit allocation
SUC - entropy coding
SUC - fused kernel
```

This identifies where the actual improvement comes from.

No performance claim is valid without an ablation table.

---

# 27. Phase 20 — Pareto frontier

> **Status 2026-10-14:** Qwen3-8B frontier done (Fisher-greedy allocation, [final_report.md](final_report.md) §3.5); ≤ 1.75 bpp breaks the model; latency / memory per mixed-plan point not measured.

Generate models around target sizes:

```text
4.0 bpw
3.5 bpw
3.0 bpw
2.5 bpw
2.25 bpw
2.0 bpw
1.75 bpw
1.5 bpw
~1.0 bpw
```

For every point measure:

- semantic quality;
- size;
- latency;
- peak memory;
- energy if available.

Plot:

```text
quality
  ^
  |                *
  |            *
  |         *
  |      *
  |   *
  +----------------------> bits/parameter
```

The goal is to move the frontier toward lower bpw without destroying quality or runtime.

---

# 28. Phase 21 — semantic objective experiments

Compare several definitions of model similarity.

### A. Weight MSE

\[
||W-W_q||_2^2
\]

### B. Layer output MSE

Compare layer outputs on calibration data.

### C. Logit MSE

Compare next-token logits.

### D. KL divergence

\[
D_{KL}(P||Q)
\]

### E. Perplexity

### F. Composite objective

For example:

\[
L = \lambda_1 KL + \lambda_2 PPL + \lambda_3 TaskLoss + \lambda_4 Memory + \lambda_5 Latency
\]

The implementation must make the weights configurable.

---

# 29. Phase 22 — universal/adaptive allocation experiments

Create two modes.

## Model-specific mode

Heavily optimized for one model and calibration distribution.

## Adaptive mode

A representation family is available and the allocator adjusts resource weights using observed performance.

Purpose:

> determine whether the Universal Allocation idea produces measurable benefits over static hand-tuned quantization.

This is an empirical question, not a theorem inherited from Cover.

---

# 30. Phase 23 — large-model scaling

> **Status 2026-10-14:** no scaling phase was run as such (D7 / D8), but the finished pipeline compressed Qwen3-30B-A3B, Qwen3.8-27B and 177–313 B MoE models ([compression_report.md](compression_report.md)).

Only after a stable 7B/8B implementation.

Test, if memory/time permit:

```text
13B/14B
30B-class
70B-class
```

Approximate raw parameter storage for 70B weights:

```text
FP16 ≈ 140 GB
8-bit ≈ 70 GB
4-bit ≈ 35 GB
```

These are parameter-only approximations; real deployed footprint includes metadata, scales, runtime buffers and KV cache.

An Ultra-class Mac with ≥ 128 GB of unified memory (M1 Ultra or later) is a useful research target because it permits experimentation with models that are difficult to keep resident in conventional GPU-only memory configurations.

---

# 31. Phase 24 — optional model-specific semantic compression

After generic SUC works, investigate task-conditioned representations.

Examples:

```text
coding-optimized
math-optimized
general-chat
long-context
```

The idea is not to create different base weights manually, but to measure whether semantic objectives shift where precision/codebook capacity should be allocated.

A successful result would look like:

```text
same base model
same approximate total size

coding workload:
  preserve blocks important for coding

math workload:
  preserve blocks important for math
```

Any improvement must be validated on held-out data.

---

# 32. Data/model leakage controls

The agent must prevent:

- calibration/test leakage;
- benchmark-specific overfitting;
- tuning directly on public benchmark examples that will later be reported;
- hidden future knowledge;
- changing the evaluation protocol after results are seen.

Maintain three data partitions:

```text
calibration
validation
final test
```

and a separate out-of-domain evaluation set.

---

# 33. Unit tests

Implement tests for:

- scalar quantizer;
- vector quantizer;
- codebook serialization;
- index packing;
- index unpacking;
- additive codebook reconstruction;
- residual encoding;
- entropy codec;
- SUC serialization;
- SUC deserialization;
- deterministic seeds.

Property tests should verify:

```text
encode → decode
```

matches the Python reference exactly for the representable representation.

---

# 34. Numerical/runtime tests

For every Metal kernel compare against the reference implementation:

```text
max_abs_error
mean_abs_error
relative_error
logit KL divergence
```

Use tolerances appropriate to the numerical format.

Do not accept visual/single-prompt similarity as sufficient evidence.

---

# 35. Performance test matrix

At minimum:

```text
batch = 1
batch = 4
batch = 16
```

and several sequence lengths.

Measure:

```text
prefill tok/s
decode tok/s
first-token latency
kernel latency
load latency
peak memory
```

Keep batch=1 as a first-class benchmark because local LLM usage is frequently memory-bound and batch=1 exposes lookup/latency overhead clearly.

---

# 36. Failure modes to actively investigate

## Failure A — excellent compression, bad quality

Likely causes:

- blocks too large;
- codebook too small;
- poor distance metric;
- no semantic calibration.

## Failure B — excellent compression, slow runtime

Likely causes:

- random lookup;
- cache pressure;
- reconstruction overhead;
- poor threadgroup/layout choice.

## Failure C — entropy coding saves storage but increases runtime

Expected to be possible.

Keep entropy coding out of the hot path or decode into an execution-friendly packed form.

## Failure D — allocator overfits calibration

Use held-out and OOD evaluation.

## Failure E — 2 bpw is attractive numerically but not hardware-efficient

Accept that a slightly larger representation can be superior on the Pareto frontier.

The actual objective is not “minimum bpw at any cost”.

---

# 37. Important engineering rule: compressed representation must be execution-aware

Do not design storage first and discover later that the GPU cannot use it efficiently.

Maintain two layouts:

## Research layout

Simple arrays/tensors for experimentation.

## Execution layout

Packed and aligned data optimized for Metal memory access.

The conversion between them must be explicit and benchmarked.

---

# 38. Codebook design research space

Test:

### Scalar codebook

```text
one value per code
```

### Vector codebook

```text
one vector per code
```

### Additive codebook

```text
sum of codebook vectors
```

### Hierarchical codebook

```text
global codebook
+
layer codebook
+
block residual
```

### Product-style codebook

Split the vector into subspaces and code each subspace separately.

For every family compare:

```text
quality
size
lookup complexity
kernel cost
```

---

# 39. Semantic bit allocation algorithm — first concrete version

Implement the following greedy algorithm before any sophisticated optimizer.

### Step 1

Split each eligible tensor into blocks.

### Step 2

For each block and candidate representation produce a record:

```json
{
  "block": 123,
  "candidate": "suc_vq_3bit",
  "bytes": 192,
  "semantic_loss": 0.0017,
  "latency_cost": 0.04
}
```

### Step 3

Establish a minimum representation for every block.

### Step 4

Create possible upgrades.

### Step 5

Rank upgrades by:

\[
score = \frac{\Delta quality}{\Delta bytes}
\]

or a configurable weighted objective.

### Step 6

Spend the remaining budget.

### Step 7

Run full-model validation.

### Step 8

Iterate if the local sensitivity estimate diverges from full-model behavior.

This first implementation should be simple enough to inspect manually.

---

# 40. Advanced allocator — multiplicative-weights research

> **Status 2026-10-14:** not implemented; only the static Fisher-greedy allocator (§39) exists.

Once the greedy allocator works, create a research implementation in which each candidate representation has a weight.

For candidate `j`:

\[
w_{j,t+1}=w_{j,t}e^{-\eta\ell_{j,t}}
\]

where `ell` is measured loss and `eta` is the learning rate.

Do this in simulation first.

Questions to answer:

1. Does the allocator converge to good representations?
2. Is it stable under noisy measurements?
3. How much exploration is required?
4. Does it outperform greedy selection?
5. Does it overreact to measurement noise?

This connects the compression problem to expert aggregation without assuming an unproven equivalence with Cover's original theorem.

---

# 41. Search strategy for direct inference

A direct SUC kernel should be explored in this order:

```text
1. decode + GEMM
2. fused decode + GEMM
3. vector LUT + dot product
4. additive codebook fused kernel
5. partial-sum/codebook-centric kernel
```

The purpose of step 1 is correctness only.

Steps 2–5 are where speed improvements must be demonstrated.

---

# 42. Benchmark against a real low-bit baseline

Never compare SUC only with FP16.

At minimum compare:

```text
FP16
strong 4-bit baseline
SUC 4 bpw
SUC 3 bpw
SUC 2 bpw
```

A meaningful success condition is not:

> “SUC is smaller than FP16.”

That is trivial.

A meaningful result is:

> “SUC is significantly smaller than a strong 4-bit baseline while preserving similar quality and remaining competitive in latency.”

---

# 43. First milestone — single linear layer

*(see D1: this gate applies to the kernel track)*

This is the primary implementation gate.

Input:

```text
W = FP16 matrix
X = FP16 activations
```

Build:

```text
W ≈ codebook[index]
```

Compute:

\[
Y=XW^T
\]

without reconstructing the full `W` into FP16 during the timed path.

Compare:

```text
FP16
INT4
SUC decode+GEMM
SUC direct Metal
```

Measure:

- output error;
- kernel time;
- throughput;
- memory use;
- effective bandwidth.

**Do not proceed to full-model compression until this milestone is sound.**

---

# 44. Second milestone — complete 7B/8B model

> **Status 2026-10-14:** on the dense 8B ≈ 3 bpp was explored (VQ G=2 K=64 at 3.37 bpp: PPL +9.3 %, KL 0.124) but the chosen file is 4.22 bpp (KL 0.049); chosen files below 3 bpp exist only for the MoE frontier models through the trellis-coded (tcq) expert tier (DeepSeek-V4-Flash 2.72 bpp, GLM-5.3-Flash 2.53 bpp).

Target a complete model with:

```text
vector codebooks
+
semantic sensitivity
+
mixed precision
+
residual/outlier support
```

First target:

```text
~3 bpw
```

Then investigate:

```text
~2.5
~2.0
<2.0
```

Do not optimize aggressively for 1 bpw before proving stability around 3 bpw.

---

# 45. Third milestone — automatic allocator

The user should be able to specify:

```yaml
budget_gb: 20
quality_floor: 0.98
latency_ceiling_ms: 100
```

and receive a representation plan automatically.

The final plan should contain:

```text
tensor
block
representation
bits
codebook
residual policy
estimated size
```

No final layer-by-layer precision values should be hard-coded in the allocator implementation.

---

# 46. Fourth milestone — full direct Metal execution

The model must be executable with a mixture of:

```text
conventional low-bit tensors
+
SUC tensors
+
FP16 exceptions
```

without reconstructing the entire model to FP16 first.

---

# 47. Success levels

> **Status 2026-10-14:** Level A reached; Level B only in part (quality ≈ 4-bit reached at ≈ 4.0–4.2 bpp, not at ≈ 3 bpp; execution competitive in decode, not in prefill); Level C not reached — [final_report.md](final_report.md) §5.1.

## Level A — proof of concept

Success if:

- vector codebook works;
- complete model can be reconstructed for evaluation;
- <4 bpw is demonstrated;
- semantic error is measured;
- output is reproducible.

## Level B — strong research prototype

Success if:

- around 3 bpw or below;
- quality close to a strong 4-bit baseline;
- direct Metal inference works;
- no full FP16 reconstruction in the hot path;
- runtime is competitive.

## Level C — major result

A major result would require simultaneously demonstrating on at least one serious model:

```text
~2 bpw or below
+
small measured quality degradation
+
direct GPU execution
+
competitive throughput
+
substantially lower memory footprint
```

These are aspirations, not guarantees.

---

# 48. What must not be claimed without evidence

Never claim:

- “lossless semantic compression” if weights are altered;
- “same intelligence” without task-specific evidence;
- “faster because smaller” without timing;
- “universal” in the mathematical Cover sense without a corresponding proof;
- “1 bpw works” based only on reconstruction MSE;
- “works for all LLMs” after one successful model;
- “entropy coding improves inference” when it only reduces file size.

---

# 49. Final experiment matrix

At the end, generate a consolidated matrix such as:

| Representation | bpw | Size | PPL | KL | Task score | tok/s | TTFT | Peak RAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | | | | | | | | |
| INT8 | | | | | | | | |
| INT4 | | | | | | | | |
| Reference 4-bit | | | | | | | | |
| SUC scalar | | | | | | | | |
| SUC vector | | | | | | | | |
| SUC additive | | | | | | | | |
| SUC + residual | | | | | | | | |
| SUC + semantic allocation | | | | | | | | |
| SUC + entropy storage | | | | | | | | |
| SUC direct Metal | | | | | | | | |

---

# 50. Final report requirements

Create:

```text
docs/final_report.md
```

The report must answer:

1. What minimum actual bpw was achieved?
2. What was the true compressed size including all overhead?
3. What quality was lost?
4. Which tasks were affected?
5. What was the inference throughput?
6. What was first-token latency?
7. What was peak memory?
8. Was direct compressed execution faster or slower than conventional 4-bit?
9. Which components created the improvement?
10. What failed?
11. Which assumptions did not hold?
12. What is the next research direction?

Include all negative results.

---

# 51. Recommended execution order

The coding agent must follow this order unless a documented experiment demonstrates a reason to change it. Per D1 the work runs on two tracks; within a track, order is strict.

```text
SHARED FOUNDATION
 1. Baseline inference (FP16 + MLX 4-bit + AWQ/GPTQ where possible)
 2. Reproducible benchmark harness
 3. Weight structure analysis
 4. Calibration + sensitivity proxies (D5)

QUALITY TRACK (reconstructed weights, evaluated in MLX)
 Q1. Scalar codebook baseline
 Q2. Vector codebook (within D4 ceiling)
 Q3. Incoherence processing (D3)
 Q4. Multi-codebook
 Q5. Residual/outlier support
 Q6. Codebook fine-tuning (D3)
 Q7. Greedy semantic bit allocator (granularity per D6)
 Q8. SUC container + Python reference decoder

KERNEL TRACK (starts after Q2)
 K1. Metal correctness kernel (mx.fast.metal_kernel)
 K2. Fused Metal kernel
 K3. Single linear layer benchmark  <-- §43 gate
 K4. Multi-codebook / residual / rotation kernels

MERGE
 5. Full transformer integration (mixed representations)
 6. Entropy coding (storage only)
 7. Ablation studies
 8. Pareto frontier
 9. Universal/adaptive allocation experiments
10. Large-model scaling (out of scope per D7 / D8; done after all with the finished pipeline, see D8)
11. C++ runtime (D2 — required, deferred; done)
12. Optional llama.cpp/GGUF integration (not done; GGUF files used only as comparison baselines)
```

If a phase fails, diagnose and correct it before scaling up.

---

# 52. Deep research direction: Semantic Universal Compression

The strongest long-term formulation is:

\[
Model = Codebooks + Indices + Residuals + Metadata
\]

where the representation minimizes model-behavior loss rather than numerical weight error alone.

A conceptual optimization is:

\[
\min_{R}
\left[
\lambda_s Size(R)
+
\lambda_q D_{semantic}(M,M_R)
+
\lambda_l Latency(R)
+
\lambda_e Energy(R)
\right]
\]

The representation may contain:

```text
very-low-bit vector codebooks
+
multiple codebooks
+
layer-specific codebooks
+
semantic-sensitive high precision blocks
+
sparse residuals
+
optional entropy coding
+
hardware-oriented packing
```

The central hypothesis is:

> The minimum useful representation of an LLM is determined by the information required to preserve its behavior, not simply by the numerical precision used to store every original weight.

The central engineering hypothesis is:

> A sufficiently structured compressed representation can be consumed directly by optimized inference kernels without a full dequantization pass.

The allocation hypothesis is:

> Under a fixed memory/bit budget, adaptive allocation can spend additional precision where it provides the largest marginal preservation of model behavior.

All three hypotheses must be evaluated independently.

---

# 53. Useful references

## Apple MLX

Unified memory documentation:

https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html

## llama.cpp

https://github.com/ggml-org/llama.cpp

Quantization tooling:

https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize

Build / Metal:

https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md

## AQLM

“AQLM: Extreme Compression of Large Language Models via Additive Quantization”

https://arxiv.org/abs/2401.06118

## QuIP

“QuIP: 2-Bit Quantization of Large Language Models With Guarantees”

https://arxiv.org/abs/2307.13304

## QuIP#

“QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Adaptive Rounding”

https://arxiv.org/abs/2402.04396

## PV-Tuning

"PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression"

https://arxiv.org/abs/2405.14852

## MLX custom Metal kernels

https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html

## FLUTE

“FLUTE: Fast Lookup Table-based Matrix Multiplication for LLM Inference”

https://arxiv.org/abs/2407.10960

## CodeGEMM

“CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs”

https://arxiv.org/abs/2512.17970

---

# 54. Final instruction to the coding agent

Treat this repository as **research engineering**.

Do not optimize for producing a large amount of code. Optimize for producing a small number of independently verifiable experiments.

Every claim must have:

```text
baseline
+
change
+
measurement
+
repeatability
+
trade-off
```

Prefer a 1-layer kernel benchmark that is scientifically correct over a full-model implementation whose performance is not understood.

The project is successful even if the hypothesis is disproven, provided the negative result is rigorous and identifies the limiting factor.
