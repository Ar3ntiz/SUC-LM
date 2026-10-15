# Ablations and Pareto frontier — Qwen3-8B run plan

Spec §26 (mandatory ablations, with the D3 rows `SUC − incoherence` and `SUC − fine-tuning`) and §27 (Pareto
frontier). Model: `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`. Every run below happens on the Mac
Studio through `scripts/studio_queue_ablations.sh`. Tables and plot: `benchmarks/make_tables.py`. Row keys are
checked against the configs by `tests/unit/test_ablation_configs.py`.

## Best configuration ("SUC")

Additive M=4, K=256, G=8 (4.0 index bpw, 4 × 4 KB codebooks per tensor, within the D4 ceiling), one-sided
randomized Hadamard (input side only, R1), 0.5% activation-selected int8 residual in the column structure (Q5b),
shared q/k/v and gate/up input rotation (`rotation_groups`), embedding and untied LM head affine 4-bit g64 (Q7/Q7b).
No fine-tuning: block-wise tuning of column residuals is not implemented, see the fine-tuning row. Expected
≈ 4.22 model bpp. No run with exactly this configuration exists yet. The closest are uniform M=4 + 0.5% CSR,
two-sided, head affine 4 (Q7: 4.225 bpp, PPL 10.081, KL 0.0751) and M=4 + 0.5% columns with the head encoded by
the codec (Q5b: 4.178 bpp, PPL 10.073, KL 0.0815).

## Protocols

Rows are differenced only against rows from the same protocol.

| Protocol | Runner | Embedding / LM head | Rotation seeds | Used by |
|---|---|---|---|---|
| **A** | `suc quantize` | fixed affine 4-bit (new `embedding_codec` option) | `rotation_groups: true` | `abl_*` configs |
| **B** | `suc allocate` | allocated (`head_candidates`), or affine 4-bit for the uniform rows | per tensor (allocate has no `rotation_groups`) | `pareto_allocate`, Q7, Q7b |
| Q5 | `suc quantize` | encoded with the row's codec | Q5/Q6: per tensor; q5b/R1: groups | earlier Q1–Q6 grids |

In every protocol, PPL is measured on the full WikiText-2 test split (299,078 tokens, 145 windows of 2048). KL is
measured against BF16 on 16 held-out validation sequences of the Phase 2 corpus. bpp counts actual bits over all
parameters: codebooks, scales, indices, residuals and metadata.

The SUC codec appears in both A (`abl_base`) and B (`pareto_allocate`, uniform M=4). This links the two
protocols, but it is not a repeat. Repeatability comes from `abl_repeat`: the same protocol A with a second seed.

## Mandatory ablation table

| Spec row | Variant (what changes vs SUC) | Config → codec / plan | Existing evidence (other protocol) | Status |
|---|---|---|---|---|
| SUC | — | `abl_base` → `additive_m4_k256_g8_rot1s+res0.005_activation_cols` | Q7 uniform M=4 + CSR (`20261001T072625Z_q7_allocate_qwen3_8b`); Q5b M=4 cols (`20261001T180551Z_q5b_residual_structure_qwen3_8b`) | **new run** |
| repeatability | SUC and two-sided re-encoded with seed 23456 | `abl_repeat` → same two codecs | — | **new run** |
| SUC − incoherence | no rotation (`rotate: false`) | `abl_base` → `additive_m4_k256_g8+res0.005_activation_cols` | q1q4: VQ G8 with / without rotation, both broken at 1 bpp (`20261001T033522Z_q1q4_codecs_qwen3_8b`); Phase 1 weight-space only | **new run** (only an equal-bit rotation on/off pair at a working rate is missing) |
| (incoherence, sides) | two-sided U W Vᵀ | `abl_base` → `additive_m4_k256_g8_rot+res0.005_activation_cols` | R1 8B, CSR, Q5 protocol: one-sided KL −8.6% (`20261001T192950Z_r1_rotation_sides_qwen3_8b`) | covered in Q5; re-run in A (cheap, also feeds the seed check) |
| SUC − vector quantization | G=1, K=16: rotated scalar codebook at the same 4 bpw index rate | `abl_codebook` → `vq_k16_g1_rot1s+res0.005_activation_cols` | q1q4: unrotated per-tensor scalar K=16 without residual, 4.0 bpp, PPL 76 vs 10.56 for M=4 | **new run** (equal bits, everything else equal) |
| SUC − multi-codebook | M=1 single codebook, G=2, K=256 (4 bpw index; codebook 1 KB) | `abl_codebook` → `vq_k256_g2_rot1s+res0.005_activation_cols` | q1q4 at 2 bpp: VQ G4 (M=1) beats additive M=2 (PPL 162 vs 218) | **new run** at the SUC rate |
| SUC − residuals | none | `abl_residual` → `additive_m4_k256_g8_rot1s` | q1q4 M=4 (4.008 bpp, KL 0.165) vs Q5 M=4 + 0.5% (4.179, KL 0.084) | **new run** (bits differ by ≈ 0.17 bpp, reported as such) |
| (residuals, selection) | magnitude-selected columns, same bits | `abl_residual` → `additive_m4_k256_g8_rot1s+res0.005_magnitude_cols` | Q5 M=3: magnitude KL 0.454 vs activation 0.248 (`20261001T053826Z_q5_residual_qwen3_8b`) | **new run** |
| (residuals, structure) | activation-selected CSR, same bits | `abl_residual` → `additive_m4_k256_g8_rot1s+res0.005_activation` | q5b: columns KL −3.4% vs CSR (M=4) | **new run** |
| SUC − fine-tuning | fine-tuning on vs off, CSR residual (column tuning not implemented) | `abl_finetune` → `…_activation+ft`, paired with the CSR row of `abl_residual` | Q6 8B: M=3 KL 0.248 → 0.228, G4 0.742 → 0.610 (`20261001T095733Z_q6_finetune_qwen3_8b`) | **new run** |
| SUC − semantic sensitivity | uniform codec vs Fisher-greedy plan at equal bpp (4.22 and 3.38) | `pareto_allocate` → `uniform_<M4 cols>` vs `greedy_4.2`, `uniform_<M3 cols>` vs `greedy_3.4` | Q7 / Q7b: greedy vs uniform +2–7% KL near 4 bpp | **new run** (part of the Pareto run) |
| (semantic sensitivity, residual selection) | see the magnitude row | | | |
| SUC − adaptive bit allocation | spec §29 adaptive mode / §40 multiplicative weights | — | — | **not done: not implemented** (only the static Fisher-greedy allocator of §39 exists) |
| SUC − entropy coding | execution layout without entropy coding = SUC; entropy coding is storage-only | `entropy_8b` (optional, CPU) on the existing 8B SUC1 file | Phase 8, 0.6B: 4.300 → 4.070 bpp file (−5.3%), additive codes −0.7%; 8B extrapolated ≈ −96 MB (`20261002T012611Z_entropy_coding_qwen3_0.6b`) | covered (0.6B); 8B measurement queued |
| SUC − fused kernel | decode from codes vs decoded dense BF16; within-kernel fusions off one at a time; two-sided online rotation | `e2e_abl_{default,nofuse,nostaging,nofuseres,twosided}` | R1 rotation timing (one-sided +21% B=1, `…rot_{baseline,onesided}`); K6 staging +60% for additive, residual fusion (`*_e2e_k6_*`, on the Studio); Q5b CSR fused / unfused | covered for VQ G4 / M=3; **new runs** for the SUC configuration (M=4 + columns) |

Notes:

- "Equal bits" is checked by the actual bpp in each result. The G=1 and G=2 rows have the same index rate as SUC but
  different scale and codebook overheads. The table reports the measured bpp.
- The magnitude and CSR residual rows also stand in for the "semantic sensitivity" component on the residual side:
  activation selection uses the Phase 2 calibration statistics E[x²], magnitude selection uses none.
- The `e2e` runs use timing-oriented codes (fast k-means, magnitude-selected residuals). Kernel cost depends only on
  shapes and counts. They say nothing about quality. Logit KL vs decoded weights is recorded as the exactness check.

## Pareto frontier

Config `pareto_allocate` (protocol B). Fisher (32 sequences) is computed once and every candidate is scored once;
then one greedy plan per budget. Budgets are average bpw over all 254 tensors (embedding and LM head included), so
they match model bpp closely (Q7b: 3.000 → 3.0005).

Block candidates: all one-sided with 0.5% activation column residuals unless noted.

| Codec | ≈ bpw |
|---|---:|
| VQ G8 K256, no residual | 1.0 |
| VQ G8 K256 | 1.2 |
| VQ G4 K64 | 1.7 |
| VQ G4 K256 | 2.2 |
| additive M=2 | 2.2 |
| additive M=3 | 3.2 |
| additive M=4 | 4.2 |
| RTN affine 2 / 3 / 4-bit | 2.5 / 3.5 / 4.5 |
| BF16 | 16 |

Head candidates: VQ G8 / G4 (no residual: there are no input activations for the embedding), affine 2 / 3 / 4 / 8.

| Target bpp | Budget | Plan in the result | Notes |
|---:|---:|---|---|
| 4.0 | 4.0 | `greedy_4.0` | |
| 3.5 | 3.5 | `greedy_3.5` | |
| 3.0 | 3.0 | `greedy_3.0` | |
| 2.5 | 2.5 | `greedy_2.5` | |
| 2.25 | 2.25 | `greedy_2.2` | plan names use `{budget:.1f}` |
| 2.0 | 2.0 | `greedy_2.0` | |
| 1.75 | 1.75 | `greedy_1.8` | |
| 1.5 | 1.5 | `greedy_1.5` | |
| ~1.0 | 1.05 | `greedy_1.1` | cheapest plan (all VQ G8) ≈ 1.006 bpp; expected broken (q1q4: PPL > 10⁷ at 1 bpp) — a measured negative result, not a "1 bpw works" claim |
| (ablation) | 4.22, 3.38 | `greedy_4.2`, `greedy_3.4` | equal-bpp partners of the uniform M=4 / M=3 rows |

Uniform rows in the same run, all with the head at affine 4: M=4 cols, M=3 cols, VQ G4 cols.

Baselines:

| Baseline | Source |
|---|---|
| RTN affine 4 / 3 / 2-bit g64 (whole model, head included) | `20261001T033522Z_q1q4_codecs_qwen3_8b` (PPL 10.088 / 13.364 / 1631) |
| MLX affine 4-bit | `20260930T223113Z_phase0_target_qwen3_8b` |
| BF16 reference PPL | 9.717 |
| Earlier greedy points (Q7b: CSR, two-sided) | plotted as a separate series |

Not covered by this plan:

- **AQLM / QuIP# reference at ≤ 3 bpp (D3).** Not run locally. No published Qwen3-8B numbers are known (D7 note).
- **Latency and peak memory per Pareto point (§27).** `e2e_generate.py` runs uniform variants only, not mixed
  greedy plans. Quality-pipeline peak memory is in each result (`peak_memory_bytes`). Runtime speed is reported
  for the uniform SUC configurations in the fused-kernel rows.
- **Energy:** not measured.
- **Downstream / coding / math task scores (§25):** not part of this plan.
- **Single seed everywhere except `abl_repeat`.** Read every Δ smaller than the seed-to-seed spread of the SUC row
  as "no measurable difference".

## Studio queue

`scripts/studio_queue_ablations.sh` runs every step sequentially. Before each step it waits until no other SUC job
is running. After each step it commits the new results on the Studio. Finished steps are skipped on restart (a
`.done` marker under `logs/abl_queue/`).

Timing estimates come from the timings recorded in the existing 8B result JSONs:

| Operation | Time |
|---|---:|
| reference / evaluate | ≈ 6 min per model |
| encode, additive M=4 | ≈ 7 min |
| encode, M=3 | ≈ 5.5 min |
| encode, VQ | ≈ 2–2.5 min |
| Q7/Q7b candidate scoring, 6 candidates | 21–22 min |
| Q7 plan validation | 4–7 min per plan |
| Q6 8B tuning | 33–43 min |
| e2e run | ≈ 7 min |

| Step | Command | Estimate |
|---|---|---:|
| `abl_base` | `suc quantize experiments/configs/abl_base_qwen3_8b.yaml` | 46 min |
| `abl_residual` | `suc quantize …/abl_residual_qwen3_8b.yaml` | 45 min |
| `abl_codebook` | `suc quantize …/abl_codebook_qwen3_8b.yaml` | 30 min |
| `pareto_allocate` | `suc allocate …/pareto_allocate_qwen3_8b.yaml` (Fisher 3 + scoring ≈ 35 + 14 plans × 6–7 + reference 6) | 2.2 h |
| `abl_finetune` | `suc quantize …/abl_finetune_qwen3_8b.yaml` | 75 min |
| `abl_repeat` | `suc quantize …/abl_repeat_qwen3_8b.yaml` | 32 min |
| `e2e_*` (5 runs) | `benchmarks/e2e_generate.py --variants … --residual-structure columns [--no-fuse / --no-fused-staging / --no-fuse-residual / --two-sided]` (first run has 4 variants ≈ 14 min, others ≈ 7) | 40 min |
| `entropy_8b` | `benchmarks/entropy_coding.py --suc1 …/compress_qwen3_8b/container/model.suc1 --fast` (CPU; skipped if the file is absent) | 30–60 min |
| **total** | | **≈ 7.5 h** (6.5–8.5 h) |

Start (on the Studio, after `git push studio main`):

```bash
nohup scripts/studio_queue_ablations.sh > logs/abl_queue.log 2>&1 &
```

To check the setup without running anything: `scripts/studio_queue_ablations.sh --dry-run`.

If `pareto_allocate` fails after candidate scoring, do not start it again from scratch. The scored candidates are
saved as `experiments/artifacts/*_pareto_allocate_qwen3_8b_candidates.json`. Set `candidates_artifact:` to that
path in the config and rerun the step: Fisher and scoring are skipped.

Afterwards, on the laptop: `scripts/sync_studio.sh`, then render the tables and plot. The Studio and laptop
environments have no matplotlib, and the script is stdlib-only, so any Python with matplotlib can draw the plot:

```bash
python3 benchmarks/make_tables.py --md docs/ablation_tables.md --plot docs/figures/pareto_qwen3_8b.svg
```

## Code changes made for this plan

- `suc quantize` gained `embedding_codec`: a fixed codec for the embedding and LM head in every row, the same field
  as in `suc allocate` / `suc compress`. The default `None` keeps the Q5 behaviour. Test:
  `tests/unit/test_quantize_run.py::test_run_quantize_with_fixed_embedding_codec`. Adding the field changes
  `config_hash` for re-runs of earlier quantize configs. It does not change their results.
- `benchmarks/e2e_generate.py` gained the `add4_g8_k256[_res]` variant (additive M=4). The default variant list is
  unchanged.
- Known gap: `suc compress` rejects `residual_structure: columns` (CodecTable guard). The best configuration
  therefore cannot yet be written as a SUC1 file, so the ablations use `suc quantize`. Block-wise fine-tuning of
  column residuals is not implemented either.
