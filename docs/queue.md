# `suc queue`: unattended multi-model compression

`suc queue <queue.yaml>` compresses a list of Hugging Face models one after the other, without supervision, on the
Mac Studio. For each model it runs a persisted state machine:

```
check -> space -> download -> verify -> compress -> report -> done
```

| Step | What it does | On problems |
|---|---|---|
| `check` | HF API metadata at the pinned revision: does it exist, file sizes, size per dtype, `config.json` (architecture, experts, source quantization). Is there a registered SUC adapter (`suc.models.adapters._REGISTRY`)? | no adapter / no safetensors → `unsupported` (with the reason), next model; not found / gated → `failed` |
| `space` | free disk vs remaining download + estimated compress work space + SUC1 size, per volume (HF cache and run dirs), × `space.margin`, keeping `space.min_free_gib` free | does not fit → `awaiting_decision` + `AWAITING_DECISION.md`, next model that fits |
| `download` | `snapshot_download` of the pinned revision in a child process (`suc.orchestrate.fetch`): allow patterns for weights / config / tokenizer, ignore patterns for duplicate weight folders (`original/`, `metal/`), retries with exponential backoff, resumable. `HF_HUB_OFFLINE` is removed from its environment. Bytes, time and MB/s are recorded. | low-disk watchdog → `awaiting_decision`; child failure after all retries → `failed` |
| `verify` | every selected file present with the Hub size; LFS sha256 == blob name in the HF cache (free: the cache names blobs by sha256); shard set == `model.safetensors.index.json`; every safetensors header consistent with the file size | bad files are deleted and downloaded again once; then `failed` |
| `compress` | `suc compress <queue_dir>/configs/<slug>.yaml` in a child process with `HF_HUB_OFFLINE=1` and its own log; the log is parsed for stage / block k/n / ETA | exit ≠ 0 → `failed` (log tail in the report); compress preflight "insufficient disk" → `awaiting_decision` |
| `report` | copies the compress result JSON / md to `<queue_dir>/results/`, extracts bpp, SUC1 size, PPL (BF16 → SUC), KL, stage timings | `failed` |

Any exception in any step marks the model `failed` (step, full traceback, log tail) and the queue moves on.
States: `pending`, `running`, `awaiting_decision`, `done`, `failed`, `unsupported`, `skipped`.

## Queue file

`experiments/queues/frontier_models.yaml` is the queue for the six frontier models (in the requested order). Main
keys (unknown keys are errors; see `suc/orchestrate/config.py`):

- `queue_dir` (local disk), `hf_cache` (`HF_HUB_CACHE` for download and compress), `work_root` (compress run
  dirs, default `<queue_dir>/runs`).
- `models`: `id`, pinned 40-hex `revision`, optional `priority` (higher first, ties in file order), `template`,
  `overrides`, `source_policy`, `estimated_bpp`, `enabled`, `notes`.
- Compress config = template + queue `overrides` + model `overrides` (deep merge, lists replace); `name`, `model`,
  `run_dir`, `output_dir` are set by the queue. Default templates: the final recipe —
  `compress_qwen3_8b_final.yaml` for dense models, `compress_qwen3_30b_a3b_final.yaml` (expert stacks `_xs`,
  uniform) when `config.json` declares experts. Budget / codec / expert-stack / source-dequant choices go in
  `overrides` (the generated config is validated by `load_compress_config` before the child starts, so a key the
  compress schema does not know yet fails that model with a clear message).
- `compress.calibration_mix` (default `v2`): calibration mix preset written into every generated config whose queue /
  model `overrides` do not set `calibration.mix` (it replaces the template's mix; `null` keeps the template's). v2 =
  the math share split between GSM8K train and competition-style MATH-train problems (chat-rendered, disjoint from
  MATH-500 and every eval suite item: `experiments/eval/calibration_disjointness.json`); v1 = the mix of every run
  before 2026-10-11 (`suc.config.CALIB_MIXES`). Stage fingerprints measure the mix against v1, so a model whose
  earlier run must be resumed / reused keeps it with `overrides: {calibration: {mix: v1}}` (done for
  Qwen3.8-Flash-Next in `frontier_models.yaml`). A hand-written compress config gets v2 by omitting `mix` or
  writing `mix: v2`; every committed config states the v1 mix explicitly and keeps its fingerprints.
- `source_policy`: `keep` (default: originals stay in the HF cache while space allows) or `delete_after_success`.
- `space`: `min_free_gib` (floor, also the watchdog threshold), `margin`, `estimated_bpp`, `workspace_factor`,
  `workspace_fixed_gib`. The estimate is deliberately coarse and on the safe side (packed U8/I8 weights count as 2
  weights each).

## Start (Mac Studio)

From the laptop: commit, `git push studio main`. Then on the Studio:

```bash
ssh <user>@<studio-host>
cd ~/Projects/SUC-LM
scripts/studio_queue.sh experiments/queues/frontier_models.yaml --dry-run   # HF check + space + configs, no download
scripts/studio_queue.sh experiments/queues/frontier_models.yaml             # nohup + caffeinate -dimsu, returns at once
```

The launcher unsets `HF_HUB_OFFLINE` (the download needs the network; the compress child gets `HF_HUB_OFFLINE=1`
from the orchestrator), exports `HF_HUB_CACHE`, keeps the Mac awake with `caffeinate -dimsu` and appends the
orchestrator's own output to `<queue_dir>/orchestrator.log`. Gated repos need `hf auth login` (or `HF_TOKEN`) on
the Studio first.

## Watch

```bash
tail -f ~/suc-queue/frontier_models/progress.log
cat ~/suc-queue/frontier_models/SUMMARY.md          # or: .venv/bin/suc queue experiments/queues/frontier_models.yaml --status
```

`progress.log` has one flushed line per event:

```
2026-10-03T12:00:00Z INFO  [2/6] deepseek-ai/DeepSeek-V4-Flash-0731 download: 45.2% 120.30 GiB/266.10 GiB, 85.1 MB/s (avg 80.3), ETA 30m12s
2026-10-03T15:10:00Z INFO  [2/6] deepseek-ai/DeepSeek-V4-Flash-0731 compress: encode block 12/61 (20%), ETA 3h05m | 1546 tensors, 4.25 bpw, 180.2 s
2026-10-03T15:11:00Z WARN  [3/6] zai-org/GLM-5.3-Flash space: AWAITING DECISION (space): short by 210.00 GiB: ...
2026-10-03T15:11:00Z ERROR [4/6] deepseek-ai/DeepSeek-V4.1-Flash compress: FAILED: StepFailed: suc compress exited with code 1 ...
```

Format: `<UTC timestamp> <INFO|WARN|ERROR> [<i>/<N>] <model id | queue> <step>: <message>`.

Other files in the queue dir:

- `status.json`: machine-readable snapshot of every model (state, step, per-step timings, check / space / download
  / verify / compress / result records) + `heartbeat` (refreshed every `poll.heartbeat_seconds`) + current
  sub-progress. A heartbeat older than a few minutes means the orchestrator is not running.
- `SUMMARY.md`: table (model, state, step, duration, sizes, bpp, PPL, KL, error summary, link to the report),
  regenerated after every state change.
- `reports/<slug>.md` / `.json`: everything about one model, the traceback and the log tail on failure, the
  environment (git commit, MLX versions, machine) and the disk usage before / after.
- `logs/<slug>.download.log`, `logs/<slug>.compress.log`, `configs/<slug>.yaml`, `results/`, `runs/<slug>/`.

## Decide

When a model does not fit, the queue writes `AWAITING_DECISION.md` (model, sizes, free space per volume, options,
cached sources of processed models that could be deleted) and continues with the next model that fits. Answer in
`<queue_dir>/decisions.yaml`:

```yaml
"zai-org/GLM-5.3-Flash": approve          # run it anyway (the min_free_gib watchdog still protects the disk)
"moonshotai/Kimi-K3": skip                # give up on this model
"openai/gpt-oss-120b": retry              # re-check / retry now (also: a failed or unsupported model)
"deepseek-ai/DeepSeek-V4.1-Flash":        # delete the cached original of a processed model, then re-check
  delete_source_of: Qwen/Qwen3.8-Flash-Next
  then: approve                           # optional
```

The file is re-read between models and every `poll.decisions_seconds` while only blocked models remain. Each edit of
the file is applied once (a standing `retry` does not loop on a model that keeps failing; edit the file again to
retry again). `delete_source_of` is refused for a model that has not finished (it still needs its source). The
queue also re-checks blocked models by itself: freeing space by hand is enough for a `space` block; a model paused
by the watchdog or the compress preflight resumes by itself only once ≥ 5 GiB more is free than at the pause.

## Resume, stop

- Rerun the same launcher after a crash, a reboot or a stop: finished steps are never repeated; the interrupted
  step resumes (the download continues partial files, `suc compress` skips its completed stages and blocks). A
  child left running by a crashed orchestrator is stopped first. A second instance refuses to start (flock on
  `<queue_dir>/.lock`).
- Stop cleanly: `.venv/bin/suc queue experiments/queues/frontier_models.yaml --stop` (SIGTERM to the orchestrator:
  it stops the running child, marks the model `pending` at its step, exits with 143).
- `--once`: exit (code 3) instead of waiting when only blocked models remain.
- Results: the compress result JSON is written to `experiments/results/` (as for every `suc compress` run): commit it
  on the Studio (`git add experiments/results && git commit`) and pull it with `git pull --ff-only studio main`.

## Tests

`tests/unit/test_queue.py` runs the orchestrator against a fake Hub, a fake download child and a fake `suc compress`
child (`tests/unit/queue_fakes.py`): unsupported architecture → skip, failure → continue, insufficient space →
awaiting decision → resumed after approval / after space was freed / after `delete_source_of`, crash and stop
resume without repeating finished steps, low-disk watchdog, re-download of a bad file, progress-line format.
