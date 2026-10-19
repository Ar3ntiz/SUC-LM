# Qwen3.8 paired hard-suite qualification

Source: `20261019T030052Z_eval_efficiency_q38_chunks.json`. Candidate binary: `bc82cf4e8b71c069356b660bdebcdcc5222218fa`. Same selected SUC1, greedy sampler and serial n-gram gathers; 120 cases per configuration.

| Source | Chunk 64 | Chunk 1024 |
|---|---:|---:|
| ARC-Challenge | 20/20 | 20/20 |
| BBH | 20/20 | 19/20 |
| GSM8K | 17/20 | 17/20 |
| LongCtx | 20/20 | 20/20 |
| MATH | 15/20 | 15/20 |
| MMLU-Pro | 8/20 | 7/20 |

Total: 100/120 vs 98/120. No execution errors. Candidate wins: 3; losses: 5. Identical outputs: 55/120.

Candidate wins: math500/counting_and_probability/1003, mmlu_pro/3817, mmlu_pro/1803

Candidate losses: math500/geometry/465, mmlu_pro/9135, mmlu_pro/8195, mmlu_pro/12204, bbh/causal_judgement-118

Decision: reject automatic chunk-1024 promotion for this release under the predeclared no-loss criterion. This is an engineering acceptance decision, not evidence of a statistically significant population-level quality difference. Explicit chunk selection stays supported. No reference-KL equivalence can be inferred from these grades; reference-KL qualification is separate.
