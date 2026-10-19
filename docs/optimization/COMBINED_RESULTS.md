# Combined automatic-default results — 1.4.7

M1 Ultra 128 GiB; baseline compiled commit `8f6d81c`, candidate `6693b2c`. Same selected local SUC1 per pair. Three measured A/B pairs per model, alternating order, first warmup excluded. CLI generation 64 tokens (63 timed decode steps); 2048-token raw text input except Qwen3-8B (12-token short input, T=0.7, top-k=0, top-p=1, min-p=0, seed 42). Others greedy. Median of per-run medians. No exclusive host reservation/OS cache purge; sub-percent changes are not established improvements.

| Model | Prefill tok/s before → after | Decode tok/s before → after | TTFT s before → after | Continuation identical |
|---|---:|---:|---:|---|
| Qwen3-8B | 119.53 → 119.64 | 36.71 → 67.51 | 0.128 → 0.115 | False |
| Qwen3.8-Flash-Next | 341.57 → 366.07 | 29.23 → 33.92 | 6.030 → 5.624 | True |
| DeepSeek-V4-Flash | 169.37 → 182.19 | 21.46 → 21.49 | 12.138 → 11.288 | True |
| GLM-5.3-Flash | 116.73 → 127.83 | 20.25 → 20.22 | 17.594 → 16.071 | True |
| Qwen3.8-27B | 93.41 → 93.29 | 24.49 → 24.84 | 21.965 → 21.994 | True |
| Qwen3-30B-A3B | 367.69 → 367.50 | 71.20 → 71.61 | 5.584 → 5.587 | True |

Last-prefill logits are byte-identical for all six models in this matrix. Qwen3-8B sampled strings differ because categorical CDF order changed, not because model logits changed. This is not a full-context BF16 KL measurement. Full-reference qualification is separate in [REFERENCE_KL.md](REFERENCE_KL.md).

CPU row-scratch reuse/serial n-gram behavior and cache locality can affect Qwen3.8 timing; this is the measured 2k profile, not a universal speed guarantee. Prefix reuse is measured separately and must not be presented as faster decode. Query tiling, half TCQ and lookup were not enabled. File sizes and compressed payloads are unchanged; memory/context closeout is separate.

Raw evidence: `experiments/results/*_eff_comparison_combined_{model}_r{1,2,3}.json`; paired raw runs carry build commits, input hash, effective chunk, parameters and memory accounting. See [release status](RELEASE_1.4.7.md) for the final qualified package revision; [final qualification](FINAL_QUALIFICATION_1.4.7.md) records all twelve long-context checks and six full-reference KL comparisons.

Reproduction uses `scripts/efficiency_compare_m1.py --case combined --repeats 3`. Qwen3-8B raw short text is `The history of the printing press begins in the fifteenth century` (no trailing comma/chat template). Long inputs are exactly the first 2048 raw token IDs of `experiments/perf/promessi_sposi_head.txt`, encoded with each model's own tokenizer; hashes are stored in the raw records. Do not substitute the older shared 256-token speed protocol or its comma-terminated short prompt.
