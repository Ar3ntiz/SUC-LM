# Using `suc eval` and `suc perf`

A practical guide to the two evaluation tools for finished SUC1 files. For the design background (what was taken from
ds4-eval / ds4-bench and why, provenance and licenses) see [tools.md](tools.md); this page is about running the tools,
configuring them and reading their output.

Contents: [1 What they measure](#1-what-the-tools-measure) · [2 Prerequisites](#2-prerequisites) ·
[3 Quick start](#3-quick-start) · [4 Config reference](#4-configuration-reference) · [5 Grading](#5-how-grading-works)
· [6 Own suites](#6-adding-your-own-suite) · [7 Regrade, drift, perf comparison](#7-regrade-drift-checks-and-perf-comparisons)
· [8 Reading the outputs](#8-reading-the-outputs) · [9 Pitfalls](#9-common-pitfalls) ·
[10 Troubleshooting](#10-troubleshooting)

## 1. What the tools measure

| | `suc eval CONFIG` | `suc perf CONFIG` |
|---|---|---|
| Question answered | Does the compressed model still answer correctly through the **real inference path** (chat template, prefill, token-by-token greedy decode)? | How fast is it — prefill throughput, time to first token (TTFT), decode ms/step — and how much memory does it use, next to MLX 4-bit and BF16? |
| Input | a test suite (`experiments/eval/*.json`) | a fixed text (`experiments/perf/promessi_sposi_head.txt`), cut at several context lengths |
| Output | pass rate per system and per source, per-case states and answers; optional agreement with a reference system; optional drift vs an older run | one row per (system, context, batch) with median / stdev / CV over repeats, ratios vs a reference system; optional ratios vs an older run |
| Use it when | you produced a new SUC1 file, changed the C++ or Python runtime, changed a kernel or a codec, or want to check that C++ and Python give the same answers | you changed a kernel / the runtime / the file layout and want speed numbers, or need the speed table of a report |
| Not for | perplexity / KL on WikiText-2 (that is `suc compress`'s quality report and `suc bench`) | quality (it never grades anything) |

Both tools run every configured *system* (see §4.1) **in one session, sequentially**, write a machine-readable JSON
(with `model`, `model_hash`, `git_commit`, `dataset_hash`, `seed`, `runtime`, `timestamp`, `config_hash`) plus a
Markdown summary next to it, and never stop on a per-case / per-row error (it is recorded and the run continues).

## 2. Prerequisites

1. **Python environment** as in [CLAUDE.md](../CLAUDE.md) "Commands" (`.venv` on a local disk).
2. **C++ tool** for `suc1_cpp` systems — built on every machine that runs it (Metal kernels are compiled at run time):
   ```bash
   cpp/build.sh ~/suc-local/cpp-build      # -> ~/suc-local/cpp-build/suc1-generate (the default cpp_build)
   ```
   Keep the build directory on a local disk. A different directory is set per system with `cpp_build:`.
3. **SUC1 files**, produced by `suc compress` (they are `run_dir/container/model.suc1`):
   - laptop, 0.6B: `~/suc-local/compress/compress_qwen3_0.6b_final{,_3bpp}/container/model.suc1`
     (`suc compress experiments/configs/compress_qwen3_0.6b_final.yaml`, also `_final_3bpp`);
   - Mac Studio: `experiments/artifacts/compress/compress_qwen3_8b_final/container/model.suc1`,
     `.../compress_qwen3_30b_a3b_final{,_greedy}/container/model.suc1` (relative to the repository checkout, git-ignored).
4. **HF cache.** The tokenizer and chat template always come from the checkpoint named in `model:` (exact `revision`).
   Only the tokenizer files (`*.json`, `*.txt`, `*.model`, `*.jinja`) are needed when all systems are SUC1; the
   `*.safetensors` weights are needed only for `bf16` / `mlx_affine` systems. Download once while online, e.g.
   ```bash
   .venv/bin/huggingface-cli download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218
   ```
   Alternatively set `model.local_path` to a local snapshot directory (no hub lookup at all).
5. **`HF_HUB_OFFLINE=1`** for every run: no network access, no surprise downloads mid-run, and a clear error if
   something is not cached.
6. **An idle machine** for `suc perf` (and preferably for `suc eval`, whose per-case timings are also recorded).

## 3. Quick start

Results land in `output_dir` (default `experiments/results/`) as `<UTC timestamp>_<name>.json` and the matching
`<UTC timestamp>_<name>.md`, e.g. `experiments/results/20261003T101500Z_eval_qwen3_8b_final.json` + `.md`. The
`name` comes from the config; files are never overwritten (a `_1` suffix is added on a collision). The JSON path is
printed at the end of the run.

Runtimes below are **rough, unmeasured estimates** (no reported run yet) derived from the token budgets and from the
decode speeds in [benchmarks.md](benchmarks.md); real runs vary with how much each model writes.

### Laptop (Apple M3, 16 GB), Qwen3-0.6B

```bash
HF_HUB_OFFLINE=1 .venv/bin/suc eval experiments/configs/eval_qwen3_0.6b_final.yaml   # core suite, 3 systems
HF_HUB_OFFLINE=1 .venv/bin/suc perf experiments/configs/perf_qwen3_0.6b_final.yaml   # MLX 4-bit, BF16, SUC C++ / Python
.venv/bin/suc eval --regrade experiments/results/<ts>_eval_qwen3_0.6b_final.json       # re-grade, no model
```

Expected: `suc eval` roughly 10–30 min (40 cases × 3 systems); `suc perf` roughly 10–30 min (4 systems × 4 contexts
× batch 1 / 4). Results: `<ts>_eval_qwen3_0.6b_final.{json,md}`, `<ts>_perf_qwen3_0.6b_final.{json,md}`. Laptop speed
numbers are indicative only.

A fast smoke test of a config before a long run: copy it and add `limit: 3` (and e.g. `max_tokens: 256`).

### Mac Studio (M1 Ultra, 128 GB), Qwen3-8B and Qwen3-30B-A3B

Code reaches the Studio only via git (`git push studio main`); results are committed **on the Studio** and pulled back
(`git pull --ff-only studio main`), see [CLAUDE.md](../CLAUDE.md) "Machines". Run one tool at a time:

```bash
ssh <user>@<studio-host>
cd ~/Projects/SUC-LM
cpp/build.sh ~/suc-local/cpp-build                   # once per C++ change
cat > logs/eval_perf_final.sh <<'EOF'
set -x
export HF_HUB_OFFLINE=1
.venv/bin/suc eval experiments/configs/eval_qwen3_8b_final.yaml
.venv/bin/suc perf experiments/configs/perf_qwen3_8b_final.yaml
.venv/bin/suc eval experiments/configs/eval_qwen3_30b_a3b_final.yaml
.venv/bin/suc perf experiments/configs/perf_qwen3_30b_a3b_final.yaml
EOF
nohup bash logs/eval_perf_final.sh > logs/eval_perf_final.log 2>&1 &
tail -f logs/eval_perf_final.log
# when it has finished:
git add experiments/results && git commit -m "eval/perf results (8B, 30B-A3B)"
```

The lines in the script run one after the other (no `&` inside), which is what keeps the timings valid.

| Config | Systems | Rough runtime | Results |
|---|---|---|---|
| `eval_qwen3_8b_final.yaml` | C++ fused, C++ `--projections separate`, Python runtime | 30–90 min | `<ts>_eval_qwen3_8b_final.{json,md}` |
| `perf_qwen3_8b_final.yaml` | MLX 4-bit, BF16, C++ fused / separate, Python; ctx 128–8192; B 1/4/16 | 1–2 h | `<ts>_perf_qwen3_8b_final.{json,md}` |
| `eval_qwen3_30b_a3b_final.yaml` | C++ greedy, C++ uniform, Python greedy | 30–90 min | `<ts>_eval_qwen3_30b_a3b_final.{json,md}` |
| `perf_qwen3_30b_a3b_final.yaml` | MLX 4-bit, BF16 (61 GB), C++ greedy / uniform, Python greedy | 1–3 h | `<ts>_perf_qwen3_30b_a3b_final.{json,md}` |
| `eval_qwen3_8b_final_ds4.yaml` (optional, hard) | C++ separate | several hours (67 thinking cases, up to 8192 tokens each) | `<ts>_eval_qwen3_8b_final_ds4.{json,md}` |
| `eval_qwen3_30b_a3b_final_ds4.yaml` (optional, hard) | C++ greedy | several hours | `<ts>_eval_qwen3_30b_a3b_final_ds4.{json,md}` |

## 4. Configuration reference

Configs are YAML. Unknown keys are errors (typos fail at load time, before anything runs); lists are accepted where
the dataclass has a tuple. Relative paths are resolved against the current directory (run from the repository root).
`~` is expanded **only** in SUC1 `file` and `cpp_build`; `suite`, `prompt_file`, `compare_to` and `output_dir` must
be relative or absolute paths without `~`.

### 4.1 `systems` entries (`SystemConfig`, shared by both tools)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | required | Unique label used in tables, `reference`, drift and comparisons. Keep it stable across runs: drift / `--compare-to` match systems by name. |
| `kind` | `suc1_cpp` \| `suc1_python` \| `bf16` \| `mlx_affine` | required | Which inference system (below). |
| `file` | str | `None` | SUC1 file; required for `suc1_*`, invalid otherwise. `~` allowed. |
| `bits` | int | `None` | `mlx_affine` only, required: one of 2, 3, 4, 5, 6, 8. |
| `group_size` | int | 64 (for `mlx_affine`) | `mlx_affine` only. |
| `cpp_build` | str | `~/suc-local/cpp-build` | Directory containing `suc1-generate` (`suc1_cpp`). |
| `cpp_args` | list[str] | `[]` | Extra `suc1-generate` flags, `suc1_cpp` only, e.g. `[--projections, separate]`, `[--dtype, f16]`. Do not pass flags the runner sets itself (`--tokens-file`, `--steps`, `--json`, `--stop-eos`, `--force`, `--logits-out`, `--quiet`). |
| `python_plan` | `fused` \| `separate` | `fused` | `suc1_python` only: fused = grouped / fused projections (the default Python runtime and the C++ default plan); separate = one module per projection. |

The four kinds:

- **`suc1_cpp`** — the C++ runtime (`cpp/`, tool `suc1-generate`) on the SUC1 file. One process per call: the file
  mapping and the Metal pipeline compilation happen per call and are reported (`load_s`, `compile_s`) but are outside
  every timing window. Batch 1 only.
- **`suc1_python`** — the same file through the Python/MLX runtime (`suc.runtime.suc1_model.load_suc1_model`), decoded
  with the mlx-lm model object.
- **`bf16`** — the reference checkpoint through mlx-lm, as loaded. Needs the safetensors weights.
- **`mlx_affine`** — the reference checkpoint through mlx-lm after MLX affine round-to-nearest quantization
  (`mx.quantize`, `bits` / `group_size`): the strong low-bit baseline (`{bits: 4, group_size: 64}` = "MLX 4-bit").

Examples:

```yaml
systems:
  - {name: suc_cpp, kind: suc1_cpp, file: ~/suc-local/compress/compress_qwen3_0.6b_final/container/model.suc1}
  - {name: suc_cpp_separate, kind: suc1_cpp, file: experiments/artifacts/compress/compress_qwen3_8b_final/container/model.suc1,
     cpp_args: [--projections, separate]}
  - {name: suc_python, kind: suc1_python, file: ~/suc-local/compress/compress_qwen3_0.6b_final/container/model.suc1}
  - {name: mlx_q4, kind: mlx_affine, bits: 4, group_size: 64}
  - {name: bf16, kind: bf16}
```

### 4.2 `model` (`ModelConfig`, both tools)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | str | required | HF id of the checkpoint the systems derive from (tokenizer and chat template for every system; weights for `bf16` / `mlx_affine`). |
| `revision` | str | `main` | Pin the exact commit (all shipped configs do). |
| `local_path` | str | `None` | Use this local snapshot instead of the HF cache. |
| `role` | `dev` \| `target` | `dev` | Recorded only. |

`model_hash` in the result is computed from the local weights when they are present, else taken from the first SUC1
file's metadata (the checkpoint it was compressed from).

### 4.3 `suc eval` (`EvalExperimentConfig`)

| Field | Type | Default | Meaning / example |
|---|---|---|---|
| `name` | str | required | Result file stem, e.g. `eval_qwen3_8b_final`. |
| `model` | mapping | required | §4.2. |
| `suite` | str | required | Suite JSON, e.g. `experiments/eval/suite_core.json`. Its sha256 is the run's `dataset_hash`. |
| `systems` | list | required | §4.1, non-empty, unique names. |
| `reference` | str \| null | `null` | Name of a system the others are compared with per case (§8.1). **Off by default**: every system is only graded. When set, the reference system runs first. |
| `case_ids` | list[str] \| null | `null` | Run only these case ids, in this order, e.g. `[basic/en_int_02, qa/it_math_3]`. Unknown ids are an error. |
| `limit` | int \| null | `null` | Then keep only the first N cases (suite order or `case_ids` order). |
| `max_tokens` | int \| null | `null` | Upper cap on every case's own `max_tokens` budget (the budget is `min(case, cap)`), e.g. `8192` for the ds4 suite (cases carry 16000). |
| `think` | `case` \| `on` \| `off` | `case` | Thinking mode: per case (`think` field of the case), or forced on / off for all cases (`enable_thinking` of the chat template). |
| `think_reply_reserve` | int ≥ 0 | 256 | Thinking-budget control (§4.5); `0` = off. The ds4 configs use 512. |
| `teacher_forced_positions` | int ≥ 0 | 128 | With a `reference`: per case the first N tokens of the reference output are scored by every other system (KL, top-1, NLL); `0` = off. Ignored without a reference. |
| `compare_to` | str \| null | `null` | Path of an older `suc eval` result JSON for the drift check (§7.2). |
| `seed` | int | 12345 | Recorded and used to seed the RNGs; it does not change greedy outputs. |
| `output_dir` | str | `experiments/results` | Where the JSON and `.md` are written. |

### 4.4 `suc perf` (`PerfExperimentConfig`)

| Field | Type | Default | Meaning / example |
|---|---|---|---|
| `name` | str | required | Result file stem. |
| `model`, `systems` | | required | §4.2, §4.1. |
| `prompt_file` | str | `experiments/perf/promessi_sposi_head.txt` | Text tokenized raw (no chat template) into one fixed token sequence; must have at least as many tokens as the largest context. |
| `contexts` | list[int] \| null | `null` | Explicit context lengths, e.g. `[128, 512, 2048, 8192]`. Give either this **or** the sweep below. |
| `ctx_start`, `ctx_max` | int \| null | `null` | ds4-bench style frontier sweep from `ctx_start` to `ctx_max` (`ctx_max` always included)… |
| `step_incr` | int \| null | `null` | …linear: `+step_incr` per frontier (e.g. 1024 → 1024, 2048, …)… |
| `step_mul` | float \| null | `null` | …or geometric: `×step_mul` (> 1, rounded up), e.g. `ctx_start: 128, ctx_max: 8192, step_mul: 2` → 128, 256, …, 8192. Exactly one of `step_incr` / `step_mul`. |
| `gen_tokens` | int ≥ 2 | 128 | Greedy tokens per timed run; EOS is ignored (fixed length). |
| `batch_sizes` | list[int] | `[1]` | e.g. `[1, 4, 16]`. Batch b uses b different windows of the text (overlapping when b × ctx exceeds it). C++ rows with b > 1 are recorded as `unsupported`. |
| `warmup` | int ≥ 0 | 1 | Untimed runs per row (8 tokens each) before the timed ones. |
| `repeats` | int ≥ 1 | 3 | Timed runs per row; statistics are over these. |
| `mlx_decode` | `pipelined` \| `sync` | `pipelined` | MLX decode loop for the mlx-lm systems (§9.1). Keep `pipelined` when comparing with C++. |
| `prefill_step_size` | int | 2048 | Chunk size of the mlx-lm prefill (as `mlx_lm.generate_step`). The C++ tool uses its own `--chunk` (64). |
| `reference` | str \| null | first system | System the ratio tables are taken against, e.g. `mlx_q4`. |
| `compare_to` | str \| null | `null` | Older `suc perf` result for row-by-row ratios (§7.3); the CLI flag `--compare-to` overrides it. |
| `seed`, `output_dir` | | 12345, `experiments/results` | As for `suc eval`. |

### 4.5 Thinking budgets and `think_close`

Qwen3 thinks inside `<think> … </think>` before the visible answer, and only the visible answer is graded. A long
reasoning chain can use up the whole budget and never answer. `suc eval` applies ds4-eval's *hard limit*, adapted to
stateless runners: for a thinking case with `budget > think_reply_reserve > 0`,

1. generate `budget − think_reply_reserve` tokens;
2. if the model stopped (EOS) — done (`think_close: none`);
3. if it did not stop and never wrote `</think>`, the tokens `</think>\n\n` are **forced** and `think_reply_reserve`
   more tokens are generated from prompt + thoughts + close (`think_close: hard`);
4. if it closed `</think>` but did not stop, it continues for `think_reply_reserve` tokens (`think_close: continued`).

So the reserve is the room the model always keeps to write its `Answer:` line. Non-thinking cases, and thinking cases
with a budget not above the reserve, run in one call. ds4's *soft* limit (closing early when `</think>` ranks high) is
not implemented. In the per-case table a `(hard)` / `(continued)` mark shows which rule fired; many `hard` closes mean
the budget is too small for the model, not necessarily that the compression broke it.

### 4.6 Sampling

There is no sampling option: both tools decode **greedily** (argmax). The C++ sampler is not mlx-lm's RNG, so sampled
outputs would not be comparable across runtimes, and greedy makes reruns reproducible (the basis of the drift check).
`seed` therefore does not change any output.

### 4.7 Suites shipped

| Suite | Cases | Content | Budget | Use |
|---|---|---|---|---|
| `experiments/eval/suite_core.json` (`suc_core`) | 40 | 10 Italian/English explanations (`open`, not graded, 300 tokens, no thinking), 10 math QA prompts (`integer`, thinking, 1500 tokens, raw prompts from `experiments/qa/prompts.json`), 20 short authored items (10 integer + 10 multiple choice, English and Italian, no thinking, 512 tokens, `Answer:` line) | short | every file, every runtime; the default |
| `experiments/eval/ds4_subset.json` (`ds4_subset`) | 67 | ds4-eval set without GPQA: 25 SuperGPQA (choice), 25 AIME 2025 (integer), 17 COMPSEC (C/C++ vulnerable lines) — all thinking | 16000 per case (configs cap at 8192) | hard capability regression; long |
| `experiments/eval/suite_hard.json` (`suc_hard`) | 30 | 5 deliberately chosen items per subset, built by `scripts/build_hard_suite.py` from pinned datasets (selection rules and ids in the file's `selection` block): GSM8K test (most steps + distractor numbers), MATH-500 L3–L5 integer answers, MMLU-Pro 10 options (law / scibench / stemez / theoremQA, key E–J), ARC-Challenge (5 exam sources), BBH (multistep arithmetic, logical deduction 7, shuffled objects 7, date understanding, causal judgement), synthetic multi-hop needles in PG19 books at ~8k / 8k / 16k / 16k / 32k tokens — all thinking off, `Answer:` line | 512–2048 per subset | discriminating SUC1 vs GGUF pairs where `suc_core` saturates (`scripts/studio_hard_suite.sh`, summary `scripts/hard_suite_report.py`); ds4 systems need `ctx` ≥ 40960 |

**GPQA Diamond (opt-in, not committed).** ds4-eval also embeds 25 GPQA Diamond items. They are CC BY 4.0, but the GPQA
authors ask that the questions not be republished in plain text (to keep them out of training sets), so they are not
in the repository. To use them, build a local suite from a ds4 checkout and keep it outside the repository:

```bash
.venv/bin/python scripts/import_ds4_eval_cases.py /path/to/ds4/ds4_eval.c ~/suc-local/eval/ds4_with_gpqa.json --include-gpqa
# then in a config (absolute path, no ~):  suite: /Users/<you>/suc-local/eval/ds4_with_gpqa.json
```

`--think-max-tokens N` sets the per-case budget written into the file (default 16000). The same script regenerates
`ds4_subset.json` without `--include-gpqa`.

## 5. How grading works

Implemented in `suc/evaluation/answers.py` (a port of the ds4-eval extractors; the ds4 self-tests are unit tests in
`tests/unit/test_eval_answers.py`). Grading is purely textual and needs no model.

**Visible text.** Only the text after the first `</think>` is graded (the whole text when there is none).

**The `Answer:` line.** Prompts with `prompt_style: ds4` end with an instruction to write exactly one final line
`Answer: <letter | integer | line numbers>`. The *strict marker* is the **last** standalone word `answer`
(any case) followed by optional spaces and `:`. Without one, the *loose marker* is the first standalone `answer`.

**`choice`** (2–10 options, letters A…):
1. no strict marker and a `\boxed{X}` with a single in-range letter → that letter;
2. else, within 96 characters after the marker, the first standalone in-range letter that does not start a prose word
   (`A careful …`, `I'll`) and is not rejected (`not B`, `except C`, `rules out D`, `isn't A`);
3. else the last standalone in-range letter of the visible text; else `?`.

**`integer`:**
1. on the strict-marker line: a `\boxed{…}` there wins, else the first integer right of the **last** `=`, else the
   first integer of the line;
2. else the last `\boxed{…}` of the visible text;
3. else (no strict marker) the loose-marker line, same rules;
4. else the last integer of the visible text; else `?`.

Leading zeros are dropped and thousands separators kept (`1,024` → `1024`). So `Answer: 3 × 14 = 42` grades `42`.

**`lines`** (COMPSEC): the marker line is normalized to a line spec (`line 9 and 15` → `9,15`, `20 - 22` → `20-22`); it
passes when **every** line the model names lies inside the accepted set of the case (e.g. answer key `17-20`) and at
least one line is named. Naming a line outside the set fails.

**`open`**: not graded (`UNGRADED`); compared only through agreement with a reference system.

States: `PASSED`, `FAILED`, `UNGRADED` (open cases), `ERROR` (the runner raised: the error text is stored, the run
continues). Pass rate = passed / graded (ungraded and error cases are excluded from the denominator — check the
`errors` column).

## 6. Adding your own suite

A suite is one JSON file (validated by `suc.evaluation.suite.load_suite`; any validation error names the case index):

```json
{
  "suite": "my_suite",
  "version": 1,
  "description": "What it tests and what it is not.",
  "attribution": "Who wrote the questions; source and license.",
  "licenses": {"MySource": "CC BY 4.0"},
  "system": null,
  "cases": [
    {"id": "my/int_01", "source": "MySource", "domain": "arithmetic", "kind": "integer",
     "question": "What is 17 times 23?", "answer": "391",
     "prompt_style": "ds4", "think": false, "max_tokens": 256},
    {"id": "my/mc_01", "source": "MySource", "kind": "choice",
     "question": "Which planet is closest to the Sun?", "choices": ["Venus", "Mercury", "Mars"], "answer": "B",
     "prompt_style": "ds4", "think": true, "max_tokens": 2048},
    {"id": "my/lines_01", "source": "MySource", "kind": "lines",
     "question": "Which lines of this C function contain the out-of-bounds write?\n1 ...\n2 ...",
     "answer": "4-5", "prompt_style": "ds4", "think": true, "max_tokens": 4096},
    {"id": "my/open_01", "source": "MySource", "kind": "open",
     "question": "Explain in two sentences what a KV cache is.", "prompt_style": "raw", "think": false,
     "max_tokens": 200}
  ]
}
```

Case keys:

| Key | Required | Meaning |
|---|---|---|
| `id` | yes | Unique in the suite; used by `case_ids`, drift and regrade. Never reuse an id for a different question. |
| `source` | yes | Group for the pass-rate-by-source table. |
| `domain`, `title` | no | Free text. |
| `kind` | yes | `choice`, `integer`, `lines` or `open`. |
| `question` | yes | The user turn. |
| `choices` | `choice` only | 2–10 strings, rendered as `A. …` lines. |
| `answer` | all but `open` | `choice`: one letter within the choices; `integer`: digits; `lines`: accepted line set (`17-20`, `9,15`). Forbidden for `open`. |
| `prompt_style` | no (`ds4`) | `ds4` appends the choices and the "write exactly one final line `Answer: …`" instruction; `raw` sends the question unchanged (then make sure the question asks for a recognisable final answer, e.g. `\boxed{}`). `open` cases are always raw. |
| `think` | yes | `enable_thinking` for this case (overridable by the config's `think`). |
| `max_tokens` | yes | Generation budget of the case (capped by the config's `max_tokens`). |

Suite-level `system` is an optional system prompt for every case (`null` = none). Put the file under
`experiments/eval/` only if its content may be published (see GPQA above); every change of the file changes its
sha256, i.e. the `dataset_hash`, so results of different versions are distinguishable. Keep questions you will report
on disjoint from anything used for calibration or tuning (CLAUDE.md, data partitions).

## 7. Regrade, drift checks and perf comparisons

### 7.1 `suc eval --regrade RESULT.json`

Re-grades the stored output texts with the **current** extractors, without loading any model, and prints every case
whose state or picked answer changed, plus one summary line per system:

```bash
.venv/bin/suc eval --regrade experiments/results/20261003T101500Z_eval_qwen3_8b_final.json
# suc_cpp basic/en_int_07: stored FAILED picked=12 -> regrade PASSED picked=120 expected=120
# suc eval regrade suc_cpp: passed=27 failed=3 changed=1 not_graded=10
```

Use it after changing `answers.py` (or to check that an extractor fix does not move other cases). It does not rewrite
the JSON. Exit code 1 when some system has no gradable case.

### 7.2 Drift check (`compare_to` in an eval config)

Set `compare_to: experiments/results/<old>_eval_....json` and rerun the same suite. Per (system name, case id) the run
reports changes of `status`, `picked`, `generated_tokens` and `tokens_sha` (hash of the generated token ids) — ds4's
deterministic gate. Interpretation:

- same machine, same software, same file: a greedy rerun must show **0 changes**; anything else is a regression or a
  non-determinism bug;
- after an inference change (kernel, runtime) that should be exact: still 0 changes expected;
- across machines or runtimes (C++ vs Python, laptop vs Studio): small rounding differences can move long generations
  (different `tokens_sha` / token count) without changing the answer — judge the `status` / `picked` changes.

Systems or cases missing in one of the two runs are skipped (`compared` counts the matched pairs).

### 7.3 Comparing perf runs (`--compare-to`)

```bash
HF_HUB_OFFLINE=1 .venv/bin/suc perf experiments/configs/perf_qwen3_8b_final.yaml \
    --compare-to experiments/results/<old>_perf_qwen3_8b_final.json
```

Rows are matched on (system name, context, batch); the `.md` gets a "Comparison" table with old → new decode tok/s and
the new / old ratios of decode tok/s, prefill tok/s (> 1 = faster) and TTFT (< 1 = faster). Only meaningful when both
runs come from the same machine under the same conditions (§9).

## 8. Reading the outputs

### 8.1 `suc eval` Markdown

1. Header: suite, number of cases, suite sha256, model@revision, git commit (`(dirty)` if uncommitted changes —
   such a run is not a result of record), chip, timestamp, reference or "no reference comparison".
2. **Summary table** per system: `kind`, `passed / graded`, `pass rate`, `ungraded`, `errors`, `gen tokens` (total
   generated), `time s` (sum of per-case wall time, including C++ process start and pipeline compilation). With a
   reference, five more columns: `identical to ref` (cases with an identical token sequence), `answer agreement`
   (same picked answer, over graded cases where the reference found one), `mean common prefix` (common token prefix /
   reference length, averaged), `TF KL vs ref` and `TF top-1` (below).
3. **Pass rate by source** (`passed/graded` per `source`).
4. **Per case**: one cell per system, `✓` / `✗` / `·` (ungraded) / `ERR`, generated tokens, picked answer in
   backticks, and `(hard)` / `(continued)` when the thinking rule fired. Column 3 is the correct answer.
5. With a reference: **agreement per case**, `= ` for identical output, `common prefix / reference length`,
   `ans =` / `ans ≠`, and `KL …`.
6. With `compare_to`: **Drift** table (`field: old → new`).

Teacher-forced metrics: the reference's own output (first `teacher_forced_positions` tokens) is fed to each other
system; per position KL(reference ‖ system) of the next-token distributions, top-1 agreement of the argmaxes, and the
mean NLL of the reference tokens under both (`nll_ref`, `nll_test` in the JSON). They measure distribution closeness
on identical inputs, independent of where the free-running outputs diverge.

The JSON additionally holds per system `info` (kind, file path / bytes / sha256, `load_s`, `resident_bytes` for
mlx-lm systems, bits / bpp for MLX systems) and per case: `prompt_tokens`, `think`, `budget`, `status`, `picked`,
`generated_tokens`, `tokens_sha`, `text`, `tokens`, `stopped`, `think_close`, `seconds`, `prefill_s`,
`decode_ms_median`, and `error` for ERROR cases; `other_processes` (top CPU processes at the start); `protocol`
(stop ids, think-close ids, reserve).

### 8.2 `suc perf` Markdown

Main table, one row per (system, ctx, B):

| Column | Meaning |
|---|---|
| `prefill tok/s` | median ± stdev over repeats of B·ctx / prefill time (full prefill from an empty cache). |
| `TTFT ms` | median prefill time until the first generated token is available. |
| `decode ms/step` | median ± stdev over repeats of the per-run **median of steps 2…n** (steady state). |
| `decode tok/s` | B × 1000 / decode ms/step (aggregate over the batch). |
| `first step ms` | the first decode step after prefill, reported separately (often slower). |
| `CV decode` | coefficient of variation (stdev / mean) of decode ms/step over the repeats. Above a few % = noisy run: repeat it. |
| `memory MiB` | mlx-lm systems: MLX **peak** memory of the run (weights + activations + KV). C++: `<mapped SUC1 file> file + <process phys_footprint>` — the GPU reads the mapping, so add both for the real footprint. |
| `KV MiB` | KV cache bytes (C++: sized exactly for prompt + steps + 1). |

Rows with status `unsupported` (C++ at B > 1) or `error` show only the status. Then two ratio tables: **decode tok/s**
and **prefill tok/s** per (ctx, B) for every system, with `(×)` = value / reference value (> 1 = faster than the
reference). With `--compare-to`, the comparison table (§7.3).

The JSON per row also has `mean`, `min`, `max`, `n` for each statistic, `decode_p90_ms`, `peak_memory_bytes`,
`kv_cache_bytes`, `first_tokens` and `deterministic` (were the first 8 generated tokens identical across repeats — a
cheap sanity check), the raw `runs`, and for C++ `memory` (mapped file, file resident in page cache, copied execution
arrays, KV, arena, workspace, phys_footprint, resident, resident peak), `load_s`, `compile_s` and
`decode_gpu_ms_median` (GPU time of the command buffers: wall minus host overhead). `systems` holds each system's
`info`; `other_processes` the top CPU processes at the start.

## 9. Common pitfalls

1. **Pipelined vs synchronous MLX decode.** mlx-lm queues step n+1 (`mx.async_eval`) before reading token n; the C++
   decode is pipelined too. A synchronous loop (one blocking eval per token) is slower: on the laptop (0.6B, MLX 4-bit,
   indicative) pipelined 4.92–5.15 ms/step vs sync 6.47–6.50 ms/step. The earlier end-to-end MLX decode figures in
   benchmarks.md came from synchronous per-token evaluation (`suc.evaluation.speed`, the e2e harness); that is why
   they **overstated SUC's speed relative to MLX**. Compare SUC with the `pipelined` MLX numbers (the default) and say
   which mode a number comes from; use `mlx_decode: sync` only to reproduce or explain old figures.
2. **The C++ runtime is batch 1 only.** B > 1 rows are `unsupported`; batch comparisons are between mlx-lm systems
   (including the Python SUC runtime). Do not read a missing C++ B=16 row as a failure.
3. **Sequential only.** Never run two benchmarks (or a benchmark and a compression, a download, a test run) at the
   same time: they share the GPU, memory bandwidth and the page cache. Queue them in one script.
4. **Thermal state and background load.** Check `other_processes` in the JSON; close heavy apps, indexers, browsers;
   keep the laptop on power. Repeat a run whose CV is high. Numbers of record come from the Mac Studio only.
5. **Compare within one session.** All systems of one `suc perf` run share the conditions; ratios across separate runs
   (or machines) are much less reliable. `--compare-to` is for same-machine before/after checks.
6. **First access of a SUC1 file** may read it from disk; the warmup run absorbs this for the timed runs. C++ `load_s`
   / `compile_s` are excluded from TTFT by design — report them separately if start-up time matters.
7. **Eval timings are not perf numbers.** `suc eval` records seconds and ms/token per case, but C++ starts one process
   per call (twice for a forced-close thinking case, whose second call re-prefills the prompt plus the thoughts);
   use `suc perf` for speed claims.
8. **Pass rate on a 40-item suite is coarse.** One case is 2.5 points. Use the per-case table, the agreement with a
   reference, and the quality report (PPL / KL) together; do not claim a quality difference from a one-case change.
9. **Dirty git tree.** `git_dirty: true` results are not reproducible from a commit; commit first (on the Studio the
   code comes from git anyway).

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: …/suc1-generate not found (build it with cpp/build.sh)` | Build the C++ tool on this machine (`cpp/build.sh ~/suc-local/cpp-build`) or set `cpp_build:` to where it is. |
| `FileNotFoundError` on a `model.suc1` path | The file does not exist on this machine: run the matching `suc compress` config, or fix `file:`. The 8B / 30B configs use repo-relative `experiments/artifacts/…` paths that exist only on the Studio; run from the repo root. |
| `LocalEntryNotFoundError` / "offline mode" errors from huggingface_hub | The tokenizer files (or, for `bf16` / `mlx_affine`, the weights) of that exact `revision` are not cached. Download once without `HF_HUB_OFFLINE`, or set `model.local_path`. |
| `config: unknown key(s) …` / `missing required key(s)` | Typo or a field of the other tool (e.g. `contexts` in an eval config). See §4. |
| `system 'x': 'file' is required for suc1_* kinds and invalid otherwise` / `bits/group_size are only valid for mlx_affine` / `cpp_args is only valid for suc1_cpp` | Field does not belong to that `kind`. |
| `reference 'x' is not a system name` | `reference` must equal one of the `name`s. |
| `perf: give either contexts or ctx_start + ctx_max …` / `exactly one of step_incr / step_mul` | Use one context specification only. |
| `prompt has N tokens, need at least M` | The largest context exceeds the tokenized `prompt_file`; lower `contexts` / `ctx_max` or use a longer text. |
| `unknown case id(s) […]` | `case_ids` contains ids not in the suite (ids include the prefix, e.g. `basic/en_int_02`, `AIME2025/…`). |
| `No such file` for `suite`, `prompt_file` or `compare_to` starting with `~` | `~` is not expanded for these fields; use an absolute path. |
| Cases with `ERROR` / rows with `error` | The runner raised; the message is in the record's `error` field (JSON). Typical: `suc1-generate failed (…)` with the tool's stderr tail — rerun that command by hand with the same flags to debug. The rest of the run is valid. |
| Many `(hard)` closes / answers `?` on thinking cases | Budget too small for the model: raise the case budget or `max_tokens`, or `think_reply_reserve`; check the stored `text`. |
| A case fails although the text looks right | Run `--regrade` on the result and inspect the `text`; if the extractor is wrong, fix `answers.py` with a unit test in `tests/unit/test_eval_answers.py` and regrade. |
| Drift reports changes on an unchanged setup | Different machine / software / file (check `file_sha256`, `git_commit`, `runtime` in both JSONs), or a real non-determinism — investigate before trusting either run. |
| Out of memory / heavy swapping | BF16 of the 8B / 30B needs the Studio (and BF16 30B at B=16, ctx 8192 is the heaviest row); drop `bf16` or large `batch_sizes` on smaller machines. A teacher-forced C++ comparison writes full logits (positions × vocab × 4 bytes) for the prompt plus the forced tokens. |
| Perf numbers jump between runs / CV is high | Background load or thermal throttling (§9.4); check `other_processes`, rerun on an idle machine. |
| `git push` / `git pull` problems around results on the Studio | Results are committed on the Studio and pulled back with `git pull --ff-only studio main` (or `scripts/sync_studio.sh`); never scp them (CLAUDE.md). |
