# Expert streaming: routed MoE experts from SSD under a RAM budget

Status 2026-10-03: research notes, a graph-independent expert cache (`suc/runtime/expert_cache.py`), a budget
calculator (`scripts/expert_budget.py`) and an SSD read micro-benchmark (`scripts/pread_bench.py`). 2026-10-04: the C++
runtime streams experts (§3, end; `cpp/tools/suc_expert_bench.cpp` measures its read path). No result on the Studio yet. Laptop (Apple M3, 16 GB) measurements are marked as such. Every
tokens/s figure below comes from a **model**, not from a measurement.

## 1. Findings from published work and existing tools

| Source | What they do | Measured result | Relevance for us |
|---|---|---|---|
| Apple, *LLM in a flash* (Alizadeh et al., ACL 2024) | Dense FFN sparsity: load only predicted-active neurons from flash; "windowing" reuses neurons of recent tokens; "row-column bundling" stores up/down rows together so each read is larger | Flash throughput goes from < 1 GB/s for small reads to > 5 GB/s for large parallel reads (M1 Max). 4–5x faster than naive loading on CPU | Two rules: (1) **reads must be large and parallel**; (2) keep one unit's data contiguous. A SUC1 expert is already ~3 contiguous ranges of 0.01–2.5 MB |
| flash-moe (danveloper / forks; Qwen3.5-397B-A17B: 512 experts, K=4, ~6.75 MB/expert at 4-bit) | C/Metal engine. Non-expert weights resident (5.5 GB). Experts read with parallel `pread` (GCD) per layer once the router has run | M3 Max 48 GB: 4.4 tok/s at 4-bit, 5.7 at 2-bit (≈ 9.7 after kernel tuning). Its log of what failed: **mmap of the expert files was 5x slower** (page-fault cost). **A custom Metal LRU cache was slower than letting the OS page cache cache the files** (+38% after removing it; the OS cache got ~71% hits on its own). F_NOCACHE gave +3% (2-bit only). LZ4 on experts −13%. Temporal expert prediction: 25% hits, −18% net. A small MLP routing predictor: 31% accuracy | Our pread results agree (mmap 2–5x slower, §2). Their cache result is a warning: a cache inside unified memory **competes with the page cache and with GPU memory**. Ours has a fixed byte budget and uses F_NOCACHE, so there is only one copy, but it has to be A/B-tested against "mmap + OS cache" on the target Mac (§5) |
| llama.cpp (mmap default; discussions #18758, #19825) | MoE models of 100–150% of RAM run from the page cache through mmap | mmap beats `--direct-io` when the model is close to RAM size: the OS cache keeps hot experts across tokens. On M5 Pro (Qwen3-Next-80B-A3B, 48 × 512 experts) explicit slice reads gave +13–14% over mmap faulting, and **laying experts out to reduce reads per token (1418 → 370) gave 2.2x on cold I/O**. On a 24 GB M4 Pro, swap thrash from mmap reached kernel panics | Layout and read size matter more than which syscall is used. When the cache is much smaller than the model, an explicit budget avoids memory-pressure failures |
| Mixtral offloading (Eliseev & Mazur 2023), Pre-gated MoE (ISCA 2024), HOBBIT, SpecMD 2026 | **Next-layer expert prediction**: apply layer l+1's router to layer l's hidden state (the residual stream changes slowly) and prefetch while layer l computes | Mixtral-8x7B: top-1 next-layer guess is correct 96% of the time on average, ~90% two or three layers ahead. Speculative prefetch: up to 14% lower time per token than on-demand loading (CPU→GPU). LRU per layer in that work: a few experts cached out of 8 | Should hold for a residual architecture. Qwen3.8's hyper-connection streams add an input-dependent mixer before every router, which may lower accuracy. **Measure it before relying on it** (flash-moe's *temporal* predictor failed; the router heuristic is a different and better-founded one) |
| MoE-Infinity (LFU), SeqMoE 2026, survey 2412.14219 | Cache policies | At 15/25/35/45% capacity, LRU gets 60/74/84/89% hits on their traces (SeqMoE's own predictor: 85–97%). MoE-Infinity-style LFU at 20 of 60 experts per layer: 33–44% | Real routing is skewed: hit rate > capacity fraction. Uniform routing (hit rate = capacity fraction) is a **pessimistic bound** |

Apple SSD and page-cache behaviour:
- Apple SSDs deliver their bandwidth only with **large (≥ 256 KB) and concurrent** reads. One thread reads 2.3 GB/s on
  the M3. Four or more threads reach about 5 GB/s (§2). The internal SSD of the M1 Ultra Studio is specified around
  7 GB/s sequential. We have not measured it yet.
- `fcntl(fd, F_NOCACHE, 1)` stops reads on that descriptor from filling the unified buffer cache. The bytes then exist
  only in our cache. Without it, every expert read is held **twice** (page cache + our copy) until the OS evicts it.
- mmap faults go through 16 KB pages plus read-ahead. Copying out of a mapping is 2–5x slower than pread at expert
  sizes (§2; flash-moe measured 5x).
- GPU memory on macOS is limited by the wired limit (`iogpu.wired_limit_mb`, about 65–75% of RAM by default). The
  "usable" budgets below assume it has been raised, or that the cache fits under it. We do not change this setting
  for the user.

## 2. Measured: SSD random reads (laptop M3, 16 GB, internal SSD)

`scripts/pread_bench.py`. Random 16 KB-aligned offsets in an 8 GB file, 1.2–1.5 GB read per row. These are laptop
numbers and must be re-run on the Studio and on the target Macs.

| Read size | Mode | T=1 | T=4 | T=8 | T=16 |
|---|---|---|---|---|---|
| 2.58 MB (one 4.2-bpp expert) | pread + F_NOCACHE | 2.30 GB/s | **5.06** | 4.94 | 4.71 |
| 2.58 MB | mmap copy | 0.85 | 1.13 | 1.44 | 2.51 |
| 1.54 MB (one 2.5-bpp expert) | pread + F_NOCACHE | 1.71 | — | 5.37 | 5.42 |
| 0.25 MB | pread + F_NOCACHE | 2.09 | — | 5.39 | 5.51 |
| 8 MB | pread + F_NOCACHE | 3.99 | — | 4.91 | 5.66 |
| 10 KB | pread + F_NOCACHE | 0.17 (p50 latency 70 µs) | | | |

A first run made right after writing the test file gave 3.7 GB/s at T=16, because the SSD was still busy with the
write. Use rows measured after the device has settled.

Python cache overhead, measured with a synthetic 474 MB SUC1 file of VQ G=2 K=256 experts (2.47 MB each) served from
the page cache: a miss costs 0.24 ms per expert end to end with 8 IO threads (10 GB/s, including the `mx.array` copy).
A `get` of 10 resident experts costs 7 µs, which is 0.34 ms per token over 48 layers. So the Python path is not the
bottleneck at SSD speeds.

## 3. Design (`suc/runtime/expert_cache.py`)

- **ExpertLayout** works from the directory alone. For each MoE layer prefix (`….experts.{e}.{proj}.weight`, regex
  configurable) it records each expert's tensors, payload records and *coalesced read ranges*. SUC1 groups payloads
  by section and sorts tensors by name, so an expert's three projections are contiguous inside each section. Each
  expert is therefore about 2–3 reads (indices ≈ 97% of the bytes, scales, residual), with no change to the file
  format. Shared stack codebooks (≤ 16 KB) stay in the file mapping.
- **ExpertRecord** holds the expert's stored payloads as mx arrays in the storage layout: packed u32 index words (the
  execution packing already), scales and residual arrays. Its `nbytes` is exactly its compressed size, and the budget
  counts that. `record.encoded(proj)` rebuilds the file's `Encoded`, reusing the SucModel decoder logic.
- **ExpertCache(source, budget_bytes, policy="lru"|"slru", io="pread"|"mmap", nocache=True, io_threads=8)**:
  - `get(layer, ids)` serves the router ids of one layer. It issues **all misses at once** to the IO pool (parallel
    reads, which the SSD needs), then installs them, evicting under the budget. Experts requested in the same call
    are never evicted by that call.
  - `prefetch(layer, ids)` starts background reads. The worker threads only read into host buffers, and the
    `mx.array`s are created on the calling thread. Predictions that are still unused when that layer's `get` arrives
    are counted as wasted and dropped (or installed, with `keep_unused_prefetch`).
  - `pin(layer, ids)` adds a static hot set, for example the most-routed experts of a calibration trace. Pinned
    experts are never evicted.
  - Policies: `lru`, and `slru` (segmented LRU: probation + protected 80%), which keeps experts used twice through a
    scan of one-off experts. Both are O(1) per access.
  - `stats`: requests, hits, prefetch hits, misses, prefetch issued/wasted, evictions, bytes read (all and sync),
    read calls, read seconds, blocked seconds, resident and pinned bytes.
- **Graph-independent execution**: `compact(ids)` takes unique experts plus remapped indices, and
  `switch_from_records(records, proj)` turns them into a `SucSwitchLinear` over the selected experts only.
- **predict_experts(router_{l+1}, h_l, k)** implements the pre-gating heuristic.
- **simulate(trace, …)** replays a (tokens, layers, k) routing trace through the same policy code. **budget_plan(…)**
  is the analytic model below. `scripts/expert_budget.py` prints both from a SUC1 file or a preset.

Tests (`tests/unit/test_expert_cache.py`, 26 passing) use tiny Qwen3-MoE SUC1 files in two layouts: per-expert codecs
with CSR residuals, and a shared expert stack with int8 column residuals. They check that:
- records rebuild the exact `Encoded` (pread and mmap);
- the compact SucSwitchLinear equals the resident one **bit for bit** (bf16 and f16, repeated ids);
- a full token-by-token decode with every MoE block fetching its experts through a cache smaller than one layer's 8
  experts (evictions on every step, LRU and SLRU) gives logits **bit-identical** to `load_suc1_model`;
- eviction order, budget and pinning behave as specified, and prefetch hit/waste accounting is correct;
- the simulator's hit rate equals the cache's on the same trace;
- the budget model reproduces the Qwen3.8 numbers.

**Integration note.** `switch_from_records` rebuilds a SucSwitchLinear on every call (unpack + repack). That is
correct but slow, and meant for tests. The production path is a **slot pool** per layer: fixed-capacity
`packed_indices (S, M, out, words)` / `row_scales (S, out)` / residual tables, with an expert→slot table. A miss
copies the record's stored words into a free slot, because the file packing *is* the kernel packing, and the kernel
gets slot ids instead of expert ids. Ragged column residuals need a per-slot capacity: with a fixed column fraction
the count is the same for every expert. This belongs in the qwen4_exp runtime and the C++ `SwitchLinear`.

**C++ runtime (2026-10-04): implemented** (`cpp/src/metal/expert_stream.cpp`, cpp/README.md "SSD expert streaming";
`suc1-generate --expert-budget BYTES --hot-experts FILE`). Per-layer GPU slot pools under one byte budget; a miss is a
`pread` (F_NOCACHE, 4–8 IO threads, all misses of a layer together) straight into the slot, whose bytes are the
kernel operands; the switch kernels keep expert ids and read through a per-expert offset table that the stream rewrites
on install (no slot-id remap). LRU / SLRU, hot set from `hot_experts.json`, optional next-layer prefetch (router of
layer l+1 on layer l's MoE input). Expert residuals, sign tables and codebooks stay resident. Streamed == resident bit
for bit (Qwen3-MoE, qwen4_exp, and deepseek_v4 with mixed-codec and native MXFP4 expert stacks since 2026-10-04: each
sub-stack's offset table is rewritten on install, `--mixed-experts routed` only). The Python runtime streams native
MXFP4 expert records too (DeepSeek-V4 test).

**Budget model** (`budget_plan`). Definitions:
- cache = RAM − resident non-expert set − overhead (KV, activations, runtime, OS; default 6 GB);
- capacity fraction f = cache / all expert bytes;
- hit rate h = f under uniform routing (pessimistic), or from a trace.

Per token:
- miss bytes = L·k·(1−h)·expert bytes;
- IO = miss bytes / SSD bandwidth + L·P(layer has a miss)·latency. A layer's experts are known only after its router
  runs, so each layer with a miss costs one dependent round trip;
- serial tok/s = 1/(compute + IO). Overlapped = 1/max(compute, IO) is the ceiling with perfect next-layer prefetch.

## 4. Numbers for Qwen3.8-Flash-Next

The arithmetic follows from the architecture:
- One expert has 1280·2560 + 2560·640 = 4.92 M parameters.
- Per token the model uses 48 layers × top-10 = 480 experts.
- Non-expert resident set ≈ 4 GB (attention / linear attention / shared experts in SUC ≈ 2.8 GB, kept BF16 tensors
  1.5 GB, embed/head 4-bit). The n-gram table stays on mmap: 16 row reads per token, ~1 ms at the measured 70 µs per
  10 KB read if issued serially, less in parallel.

| Expert tier | MB / expert | all experts | per token if all miss |
|---|---|---|---|
| 4.2 bpp (VQ G=2 K=256 + 0.5% col. residual, current plan) | 2.58 | 63.4 GB | 1.24 GB |
| ≈ 2.5 bpp (VQ G=4 K=256 + scales/residual) | 1.54 | 37.7 GB | 0.74 GB |

Assumptions for the estimates:
- SSD 5.0 GB/s (the M3 measurement in §2; the Studio is probably faster);
- 150 µs per layer round trip;
- overhead 6 GB;
- **compute 35 ms/token with every expert resident.** This is an estimate: about 2x the Qwen3-30B-A3B SUC decode,
  17.5 ms at 57 tok/s, for the same 48 layers plus hyper-connections, DeltaNet and 10 instead of 8 experts. Replace it
  with the measured value of the qwen4_exp runtime.

The table has three hit-rate rows per budget:
- "uniform": the pessimistic bound;
- "zipf s": synthetic skew, a what-if and **not a measurement**;
- real hit rates need a Qwen3.8 routing trace: `--trace`.

The table is `scripts/expert_budget.py --preset qwen3.8-{4.2,2.5} --ram 111,64,32 --ssd-gbps 5 --compute-ms 35 --zipf 0.6,1.0`.

| Budget | Tier | cache GB | capacity | hit (uniform / s=0.6 / s=1) | miss MB/token | tok/s serial | tok/s ceiling (prefetch) |
|---|---|---|---|---|---|---|---|
| **111 GB** (Studio) | 4.2 bpp | 101 | 1.00 | all resident | 0 | 28.6 | 28.6 |
| 111 GB | 2.5 bpp | 101 | 1.00 | all resident | 0 | 28.6 | 28.6 |
| **64 GB** | 4.2 bpp | 54 | 0.85 | 0.85 / 0.83 / 0.88 | 184 / 208 / 147 | 12.9 / 12.3 / 14.6 | 23.5 / 21.5 / 28.6 |
| 64 GB | 2.5 bpp | 54 | 1.00 | all resident | 0 | 28.6 | 28.6 |
| 48 GB | 4.2 bpp | 38 | 0.60 | 0.60 / 0.69 / 0.83 | 496 / 381 / 206 | 7.1 / 8.5 / 12.2 | 9.4 / 12.0 / 21.3 |
| 48 GB | 2.5 bpp | 38 | 1.00 | all resident (47.7 GB) | 0 | 28.6 | 28.6 |
| **32 GB** | 4.2 bpp | 22 | 0.35 | 0.35 / 0.49 / 0.72 | 809 / 631 / 342 | 4.9 / 5.9 / 9.1 | 5.9 / 7.5 / 13.3 |
| 32 GB | 2.5 bpp | 22 | 0.58 | 0.58 / 0.68 / 0.83 | 308 / 235 / 126 | 9.6 / 11.2 / 15.1 | 14.6 / 18.5 / 28.6 |

What the table says:
- **Studio (111 GB usable):** no streaming is needed. Resident memory is 63.4 (experts) + 4 + 6 = 73 GB, plus the
  4-bit n-gram table on mmap/CPU (27–32 GB), for ≈ 100–105 GB. The margin is ≈ 6–11 GB, so the n-gram table should
  stay page-cache backed (evictable) rather than wired. Expert streaming here is only useful as a test bed and as a
  memory-pressure valve (for example a small cache at 4.2 bpp that frees ~20 GB for long contexts).
- **64 GB:** at 4.2 bpp, streaming costs about 2x in tokens/s (12–15 vs 29) unless prefetch hides IO. The ≈ 2.5 bpp
  expert tier makes the model **fully resident** at 64 GB and even at 48 GB (47.7 GB, tight).
- **32 GB:** streaming is mandatory. 2.5 bpp is worth about 2x over 4.2 bpp: 9.6–15 vs 4.9–9 tok/s serial, because it
  halves both miss bytes and the per-expert cost, and it raises capacity from 0.35 to 0.58. At 32 GB, IO dominates
  compute, so the bandwidth of the target SSD sets the speed.
- The latency term is small: ≤ 48 × 150 µs = 7 ms. Bandwidth is the cost.

## 5. Recommendation

1. **Studio:** run Qwen3.8 fully resident at 4.2 bpp. Do not build streaming into the critical path for the 128 GB
   target.
2. **32–64 GB Macs:** the **sub-3 bpw expert tier is the first lever** and should be tried before any cache logic. It
   removes streaming at 48–64 GB and doubles throughput at 32 GB. It needs the quality check of the ≈ 2.5 bpp tier
   (VQ G=4 K=256 + residual) on the experts only: per D3 and §48 this means PPL/KL against the 4.2 bpp file and
   against AQLM/QuIP#-class ≤ 3 bpw baselines.
3. Then add streaming with the cache as implemented:
   - explicit byte budget;
   - pread + F_NOCACHE with 4–8 IO threads;
   - all misses of a layer issued at once;
   - LRU or SLRU plus a pinned hot set taken from a calibration routing trace.
4. Measurements still needed, in this order:
   - (a) a **routing trace** of Qwen3.8 on calibration text: real hit rates (`--trace`), per-expert frequency (hot
     set), and next-layer router-heuristic accuracy (`predict_experts`). This decides whether async prefetch is worth
     building; flash-moe saw a net loss from a weaker predictor;
   - (b) `pread_bench.py` on the Studio and on a 32 GB target;
   - (c) an A/B on the target of explicit cache + F_NOCACHE versus mmap + OS page cache, the configuration that won in
     flash-moe.
5. Integration (qwen4_exp runtime / C++): a slot-pool layout (§3) so that misses are memcpy's into fixed GPU slots
   and the switch kernel takes slot ids. Mixed-precision tiers (hot experts at 4.2 bpp, cold ones at ≈ 2.5 bpp, as in
   HOBBIT and flash-moe's tiered quantization) are a follow-up once (4a) gives frequencies.

## Sources

- Alizadeh et al., *LLM in a flash: Efficient Large Language Model Inference with Limited Memory*, ACL 2024 —
  https://aclanthology.org/2024.acl-long.678.pdf, https://machinelearning.apple.com/research/efficient-large-language
- flash-moe — https://github.com/Alexintosh/flash-moe, https://github.com/awas-k/flash-moe-M1,
  https://starlog.is/articles/llm-engineering/danveloper-flash-moe/, https://github.com/jundot/omlx/issues/986
- llama.cpp — https://github.com/ggml-org/llama.cpp/discussions/18758, https://github.com/ggml-org/llama.cpp/issues/19825,
  https://github.com/ggml-org/llama.cpp/discussions/19163
- Eliseev & Mazur, *Fast Inference of Mixture-of-Experts Language Models with Offloading* — https://arxiv.org/pdf/2312.17238
- SpecMD — https://arxiv.org/html/2602.03921; HOBBIT — https://arxiv.org/pdf/2411.01433;
  *In-depth Analysis on Caching and Pre-fetching in MoE Offloading* — https://arxiv.org/pdf/2511.05814
- SeqMoE — https://arxiv.org/pdf/2609.12978; MoE inference survey — https://arxiv.org/pdf/2412.14219
