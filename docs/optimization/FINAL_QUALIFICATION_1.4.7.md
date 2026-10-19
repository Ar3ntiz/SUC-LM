# Engine 1.4.7 final qualification

All tests use only the selected complete local files on M1 Ultra. Long-context checks are one bounded A/B pair per model/length, not repeated throughput or full-context BF16 evaluation. The repeated 2k/sampling matrix is in COMBINED_RESULTS.md.

## Context and memory checks

| Model | Tokens | Same final logits / continuation | KV MiB | States MiB | Arena + workspace MiB | Process footprint MiB |
|---|---:|---|---:|---:|---:|---:|
| q8 | 8192 | True / True | 1153.3 | 0.0 | 36.9 | 1339.3 |
| q8 | 32768 | True / True | 4609.3 | 0.0 | 37.0 | 4796.4 |
| q27 | 8192 | True / True | 512.6 | 146.8 | 486.7 | 1245.1 |
| q27 | 32768 | True / True | 2048.6 | 146.8 | 488.9 | 2784.7 |
| q30 | 8192 | True / True | 768.8 | 0.0 | 45.4 | 1827.4 |
| q30 | 32768 | True / True | 3072.8 | 0.0 | 45.5 | 4135.3 |
| q38 | 8192 | True / True | 222.2 | 110.3 | 1013.1 | 5320.9 |
| q38 | 32768 | True / True | 888.2 | 110.3 | 1017.1 | 6031.4 |
| ds | 8192 | True / True | 80.9 | 0.0 | 11301.6 | 17584.0 |
| ds | 32768 | True / True | 242.1 | 0.0 | 3248.5 | 9131.3 |
| glm | 8192 | True / True | 93.6 | 140.8 | 3698.3 | 11585.6 |
| glm | 32768 | True / True | 374.1 | 140.8 | 1601.2 | 9372.3 |

Process footprint is not total unified/model memory. GPU-mapped file views, CPU n-gram tables, page cache, driver residency and shared OS memory have different accounting; do not add this column to file size or compare it to a whole-system RAM measurement. All raw runs retain detailed memory fields. No memory-policy or compressed-payload change was made.

## Real bundled-engine sessions

| Model | Cache-on/off outputs | Maximum replay-footprint growth MiB | Cancellation / reload |
|---|---|---:|---|
| q8 | True | 0.00 | passed |
| q38 | True | 0.20 | passed |

Each session experiment covers strict aligned-prefix extension, divergent/equal/shorter histories, cancellation after streamed output, eight replay cycles and model unload/reload on the actual bundled engine. Cache-off is the oracle for the same requests. This is neither a visual GUI test nor a proof of zero leaks under every workload.

## Complete reference KL

| Model | Chunk | Scored positions | Mean KL before / after | Global p95 before / after | Raw logits identical |
|---|---:|---:|---:|---:|---|
| q8 | 512 | 8176 | 0.04957650 / 0.04957650 | 0.14676428 / 0.14676428 | True |
| q27 | 512 | 8176 | 0.02339630 / 0.02339630 | 0.06915852 / 0.06915852 | True |
| q30 | 512 | 8176 | 0.06536214 / 0.06536214 | 0.19141348 / 0.19141348 | True |
| q38 | 512 | 8176 | 0.05983845 / 0.05983845 | 0.27339077 / 0.27339077 | True |
| ds | 4096 | 8176 | 0.23485560 / 0.23485560 | 0.99598897 / 0.99598897 | True |
| glm | 512 | 8176 | 0.30379513 / 0.30379513 | 1.23923683 / 1.23923683 | True |

The same existing BF16 references, validation rows, BF16 logit casts and canonical float32 divergence/global-p95 aggregation are used within every pair. These runtime figures must not replace reconstructed-weight compression metrics or be advertised as improved compression. Seeded sampled strings are a separate CDF-order issue.

## Service profiles and final artifacts

- ds: standalone and bundled-app defaults match the explicit historical-kernel CLI oracle at chunk 64.
- glm: standalone and bundled-app defaults match the explicit historical-kernel CLI oracle at chunk 512.
- DS service-64 arithmetic: all 8176 validation positions match the historical TCQ path; mean BF16-reference KL 0.23566283, with identical raw logits.
- Final seven-check shipped-artifact smoke and standalone distribution refer to compiled commit `24b2ad7ff7aed65b7b41160ab35f6df8738e3c88`. Shared C++/Metal input hash: `f6bb844d6d5d03e35e2cffc673a500b3a1c68bacbd41abd58037feebc3af577f`.
- The final generation CLI is byte-for-byte identical to the binary used for the six-model matrix and long-context qualification: True.

Manual GUI acceptance of build 28 on physical Intel/other Apple Silicon machines is not claimed. The unchanged UI already passed native and simulated remote-only Swift suites; final app/helpers retain Universal slices and the local engine remains ARM-only. Intel uses remote inference. MTP, new compression, online model-card publication, installation and main-branch merge are outside this closeout.

Status: QUALIFIED within the scope and limits above.
