# Full same-reference runtime KL — engine 1.4.7

Same stored BF16 reference and validation rows within every pair; 16x512 tokens, final position excluded (8176 scored positions). Raw float logits must match; scoring casts to BF16 then uses canonical float32 divergence and global p95. These are C++ runtime qualification values, separate from historical reconstructed-weight compression metrics.

| Model | Effective chunk | Mean KL before / after | Global p95 before / after | Raw logits identical |
|---|---:|---:|---:|---|
| q8 | 512 | 0.04957650 / 0.04957650 | 0.14676428 / 0.14676428 | True |
| q27 | 512 | 0.02339630 / 0.02339630 | 0.06915852 / 0.06915852 | True |
| q30 | 512 | 0.06536214 / 0.06536214 | 0.19141348 / 0.19141348 | True |
| q38 | 512 | 0.05983845 / 0.05983845 | 0.27339077 / 0.27339077 | True |
| ds | 4096 | 0.23485560 / 0.23485560 | 0.99598897 / 0.99598897 | True |
| glm | 512 | 0.30379513 / 0.30379513 | 1.23923683 / 1.23923683 | True |

The first Qwen chunk-64 command-line comparison used mismatched effective chunks and remains diagnostic only. It is excluded here. DeepSeek service chunk 64 is a separate full-profile check in FINAL_QUALIFICATION_1.4.7.md. CLI/service profiles must not be silently mixed.
- q8: `20261019T100440Z_eff_reference_kl_q8.json`; compiled candidate `24b2ad7ff7aed65b7b41160ab35f6df8738e3c88`.
- q27: `20261019T101247Z_eff_reference_kl_q27.json`; compiled candidate `24b2ad7ff7aed65b7b41160ab35f6df8738e3c88`.
- q30: `20261019T101558Z_eff_reference_kl_q30.json`; compiled candidate `24b2ad7ff7aed65b7b41160ab35f6df8738e3c88`.
- q38: `20261019T050112Z_eff_reference_kl_q38.json`; compiled candidate `f464a603e0d18f62671765c0555988b1fa6eae55`.
- ds: `20261019T041458Z_eff_reference_kl_ds.json`; compiled candidate `f464a603e0d18f62671765c0555988b1fa6eae55`.
- glm: `20261019T054052Z_eff_reference_kl_glm.json`; compiled candidate `f464a603e0d18f62671765c0555988b1fa6eae55`.
