# M1 inference efficiency results

Candidate binary commit: `bc82cf4e8b71c069356b660bdebcdcc5222218fa`. Warm-up pairs and obsolete candidate builds excluded.

| Experiment | Model | Measured pairs | Prefill tok/s A → B | Decode tok/s A → B | Token identity | Last-position KL max |
|---|---|---:|---:|---:|---|---:|
| ngram | q38 | 3 | 51.5 → 51.8 | 35.8 → 34.2 | True | 0 |
<!-- ngram: B workers=[4, 4, 4], tasks=[16, 16, 16] -->
| sampling | q8 | 3 | 119.9 → 120.2 | 36.9 → 67.7 | False | 0 |
| chunks | q38 | 2 | 106.3 → 340.8 | 32.7 → 32.0 | False | 0.055342 |
| attention | q8 | 2 | 231.5 → 246.6 | 62.5 → 62.4 | True | 0 |
| tcq | ds | 2 | 169.1 → 182.0 | 21.5 → 21.5 | True | 0 |
| tcq | glm | 2 | 116.0 → 133.2 | 20.2 → 20.8 | True | 0 |
| lookup | q8 | 2 | 308.3 → 306.8 | 75.9 → 52.7 | False | 0 |
| coverage | q27 | 1 | 93.0 → 93.5 | 24.6 → 24.8 | True | 0 |
| coverage | q30 | 1 | 368.5 → 368.4 | 72.6 → 72.5 | True | 0 |

## Prefix reuse

| Model | New-request tokens | Cached tokens | Request wall seconds off → on | Speedup | First-token identity |
|---|---:|---:|---:|---:|---|
| q8 | 16 | 2048 | 6.8099 → 0.0965 | 70.60× | True |
| q38 | 16 | 2048 | 5.7709 → 0.2751 | 20.98× | True |

## Interpretation and limits

- A/B order alternates; each experiment excludes its first warm-up pair. No forced OS cache purge or exclusive machine reservation was performed.
- Sampling compares historical sorted and linear categorical CDF order. Seeded strings may differ while the probability distribution is preserved; greedy sampling is unchanged.
- Chunk comparisons reproduce the old application chunk of 64 versus the shared policy. Fast batched arithmetic can differ across chunk boundaries.
- Query tiling and TCQ reuse remain opt-in. Half-operand TCQ failed the synthetic quality budget and is rejected.
- Lookup verification is a greedy prompt-lookup prototype, not MTP. Amortized generation timing is not per-token latency. Hybrid graphs/SSD experts/sampled speculative generation are explicitly unsupported.
- Cache results cover an exact, chunk-aligned live prefix. Divergent history, equal/shorter prompts, cancellation and reload invalidate or replay safely. This is not an LRU snapshot cache for every chat.
- Last-position KL and short continuations do not replace a full held-out quality/hard-suite qualification. No new model compression or production/main deployment was performed.
- Real MTP remains a separate pending milestone; see MTP_PLAN.md.
