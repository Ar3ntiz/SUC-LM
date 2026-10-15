"""Human-readable outputs of `suc queue`: progress.log lines, compress-log parsing, SUMMARY.md, AWAITING_DECISION.md,
per-model reports. Pure functions over the persisted queue state (suc.orchestrate.orchestrator)."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from suc.provenance import utc_timestamp

LEVELS = ("INFO", "WARN", "ERROR")
# 2026-10-03T12:00:00Z INFO  [2/6] org/model download: 45.2% 12.30/27.20 GiB, 85.1 MB/s, ETA 3m10s
PROGRESS_LINE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) (INFO|WARN|ERROR) +\[(\d+)/(\d+)\] (\S+) (\w+): (.*)$")


def gib(n: float | None) -> str:
    """Human-readable size (GiB above 1 GiB, MiB above 1 MiB, else KiB)."""
    if n is None:
        return "—"
    for unit, k in (("TiB", 2 ** 40), ("GiB", 2 ** 30), ("MiB", 2 ** 20)):
        if abs(n) >= k:
            return f"{n / k:.2f} {unit}"
    return f"{n / 2 ** 10:.1f} KiB"


def duration(s: float | None) -> str:
    if s is None or s != s or s < 0:
        return "—"
    s = int(round(s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"{s // 86400}d{s % 86400 // 3600:02d}h"


class ProgressLog:
    """Append-only, flushed after every line (tail -f). Thread-safe."""

    def __init__(self, path: Path, total: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.total = total
        self._lock = threading.Lock()

    def line(self, level: str, index: int, who: str, step: str, msg: str) -> str:
        assert level in LEVELS
        text = f"{utc_timestamp()} {level:<5} [{index}/{self.total}] {who} {step}: {' '.join(str(msg).split())}"
        with self._lock, open(self.path, "a") as f:
            f.write(text + "\n")
            f.flush()
            os.fsync(f.fileno())
        return text


# ------------------------------------------------------------------------------------------- compress log

_PREFIX = re.compile(r"^\S+ \[compress\] ")
_STAGE = re.compile(r"^stage (\w+): (start|done in [\d.]+ s.*|up to date, skipped.*|FAILED.*)$")
_BLOCK = re.compile(r"^(\w+) (\S+) \((\d+)/(\d+)\)")


class CompressProgress:
    """Turns `suc compress` log lines into progress events: stage start/done, block k/n with ETA, warnings, errors."""

    def __init__(self, clock=time.time) -> None:
        self.clock = clock
        self.stage: str | None = None
        self.block: tuple[int, int] | None = None
        self._first: dict[str, tuple[float, int]] = {}   # stage -> (time, k) of the first block line this attempt
        self.eta_seconds: float | None = None

    def feed(self, raw: str) -> tuple[str, str] | None:
        """(level, message) for a line worth a progress event, else None."""
        line = _PREFIX.sub("", raw.rstrip("\n"))
        if not line.strip():
            return None
        if (m := _STAGE.match(line)) is not None:
            self.stage, what = m.group(1), m.group(2)
            self.block, self.eta_seconds = None, None
            if what.startswith("FAILED"):
                return "ERROR", f"stage {self.stage} {what}"
            return "INFO", f"stage {self.stage} {what}"
        if (m := _BLOCK.match(line)) is not None:
            stage, k, n = m.group(1), int(m.group(3)), int(m.group(4))
            self.stage, self.block = stage, (k, n)
            now = self.clock()
            t0, k0 = self._first.setdefault(stage, (now, k))
            self.eta_seconds = (now - t0) / (k - k0) * (n - k) if k > k0 else None
            eta = f", ETA {duration(self.eta_seconds)}" if self.eta_seconds is not None else ""
            return "INFO", f"{stage} block {k}/{n} ({100 * k / n:.0f}%){eta} | {line[m.end():].lstrip(': ')[:160]}"
        if "WARNING" in line:
            return "WARN", line[:300]
        if line.startswith(("Traceback", "PreflightError", "RuntimeError", "ValueError")) or "Error:" in line:
            return "ERROR", line[:300]
        if line.startswith(("preflight:", "run ", "done ", "reference:", "evaluate:", "container:", "plan:",
                            "finetune:", "candidates ")):
            return "INFO", line[:300]
        return None

    def snapshot(self) -> dict:
        return {"stage": self.stage, "block": list(self.block) if self.block else None,
                "eta_seconds": self.eta_seconds}


def tail(path: Path, n: int = 60) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            data = f.read().decode(errors="replace")
    except OSError:
        return ""
    return "\n".join(data.splitlines()[-n:])


# ------------------------------------------------------------------------------------------- markdown

def _fmt(v, spec: str) -> str:
    return "—" if v is None else format(v, spec)


def _model_duration(r: dict) -> float | None:
    return sum(s.get("seconds") or 0 for s in r.get("steps", {}).values()) or None


def summary_md(state: dict) -> str:
    models = state["models"]
    counts: dict[str, int] = {}
    for r in models:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    lines = [f"# Queue `{state['queue']}`", "",
             f"Updated {utc_timestamp()} · orchestrator {state.get('status', '?')} (pid {state.get('pid')}, host "
             f"{state.get('host')}) · heartbeat in `status.json` · live log: `tail -f progress.log`", "",
             "States: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
             "| # | Model | State | Step | Duration | Download | SUC1 | bpp | PPL (BF16 → SUC) | KL | Error / note "
             "| Report |",
             "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for i, r in enumerate(models, 1):
        res = r.get("result") or {}
        size = (r.get("check") or {}).get("sizes", {}).get("download_bytes")
        ppl = (f"{_fmt(res.get('ppl_bf16'), '.3f')} → {_fmt(res.get('ppl'), '.3f')}" if res else "—")
        note = ""
        if r.get("error"):
            e = r["error"]
            note = f"{e['step']}: {e['type']}: {e['message']}"
        elif r.get("awaiting"):
            note = f"awaiting decision ({r['awaiting']['reason']}): {r['awaiting'].get('summary', '')}"
        elif r.get("reason"):
            note = r["reason"]
        note = note.replace("|", "\\|").replace("\n", " ")[:160]
        lines.append(f"| {i} | `{r['id']}` | {r['state']} | {r.get('step') or '—'} | {duration(_model_duration(r))} "
                     f"| {gib(size)} | {gib(res.get('suc1_bytes'))} | {_fmt(res.get('bpp'), '.3f')} | {ppl} "
                     f"| {_fmt(res.get('kl_mean'), '.4f')} | {note} | [report](reports/{r['slug']}.md) |")
    return "\n".join(lines) + "\n"


DECISIONS_HELP = """\
Write your decision into `decisions.yaml` in this directory (the queue re-reads it between models and every
`poll.decisions_seconds` while it waits). One entry per model id:

```yaml
"<model id>": approve            # run it anyway (the low-disk watchdog floor still applies)
"<model id>": skip               # give up on this model
"<model id>": retry              # re-check / retry now (e.g. after you freed space by hand)
"<model id>":                    # delete the HF-cache copy of another model's source, then re-check
  delete_source_of: <other model id>     # or a list of ids
  then: approve                  # optional: approve | retry
```
"""


def awaiting_md(state: dict, free: dict) -> str:
    waiting = [r for r in state["models"] if r["state"] == "awaiting_decision"]
    lines = [f"# Awaiting decision — queue `{state['queue']}`", "", f"Updated {utc_timestamp()}.", ""]
    for r in waiting:
        a = r["awaiting"]
        d = a.get("details", {})
        lines += [f"## `{r['id']}` @ `{r['revision'][:12]}`", "",
                  f"- Reason: **{a['reason']}** (step `{a['step']}`, since {a['since']}): {a.get('summary', '')}"]
        sizes = (r.get("check") or {}).get("sizes", {})
        if sizes:
            lines.append(f"- Download (selected files): {gib(sizes.get('download_bytes'))}; already in the cache: "
                         f"{gib(d.get('present_bytes'))}; estimated SUC1 file: {gib(d.get('suc1_estimate_bytes'))}; "
                         f"estimated compress work space: {gib(d.get('work_estimate_bytes'))}")
        for v in d.get("volumes", []):
            lines.append(f"- Volume `{v['path']}`: need {gib(v['need_bytes'])} × margin {v['margin']} + floor "
                         f"{gib(v['floor_bytes'])}; free {gib(v['free_bytes'])} → short by {gib(v['short_bytes'])}")
        if a.get("log_line"):
            lines.append(f"- Log: `{a['log_line']}`")
        lines.append("")
    done = [r for r in state["models"] if r["state"] in ("done", "failed", "skipped", "unsupported")
            and (r.get("disk") or {}).get("source_bytes")]
    lines += ["## Options", "", "- `approve`: proceed although the estimate does not fit.", "- `skip`: drop the model.",
              "- free space by hand (other volumes, old run dirs under the queue's `runs/`), then `retry` (the queue "
              "also re-checks by itself every poll).",
              "- `delete_source_of`: delete the cached original safetensors of a model already processed:"]
    for r in done:
        lines.append(f"  - `{r['id']}` ({r['state']}): {gib(r['disk']['source_bytes'])} in the HF cache")
    if not done:
        lines.append("  - (no processed model with a cached source)")
    lines += ["", "Free space now: " + ", ".join(f"`{k}` {gib(v)}" for k, v in free.items()), "", DECISIONS_HELP]
    return "\n".join(lines) + "\n"


def model_report_md(r: dict, env: dict) -> str:
    lines = [f"# `{r['id']}` — {r['state']}", "",
             f"Revision `{r['revision']}` · slug `{r['slug']}` · step `{r.get('step')}` · started {r.get('started')} · "
             f"finished {r.get('finished') or '—'} · attempts {r.get('attempts', 0)}", ""]
    if r.get("reason"):
        lines += [f"Reason: {r['reason']}", ""]
    if r.get("awaiting"):
        lines += [f"Awaiting decision: {r['awaiting']['reason']} — {r['awaiting'].get('summary', '')}", ""]
    lines += ["| Step | Status | Started | Seconds |", "|---|---|---|---:|"]
    for k, s in r.get("steps", {}).items():
        lines.append(f"| {k} | {s.get('status')} | {s.get('started')} | {_fmt(s.get('seconds'), '.1f')} |")
    c = r.get("check") or {}
    if c:
        a, sz = c.get("arch", {}), c.get("sizes", {})
        lines += ["", "## Check", "",
                  f"- Architecture: model_type `{a.get('model_type')}`, {a.get('architectures')}, layers "
                  f"{a.get('num_layers')}, experts {a.get('num_experts')}, source quantization "
                  f"{a.get('source_quantization')}",
                  f"- SUC adapter: {c.get('adapter')}",
                  f"- Download {gib(sz.get('download_bytes'))} in {sz.get('files')} files "
                  f"({sz.get('safetensors_files')} safetensors); whole repo {gib(sz.get('repo_bytes'))}",
                  "- Parameters by stored dtype (Hub): " + ", ".join(f"{k} {n / 1e9:.2f} B" for k, n in
                                                                    sz.get("params_by_dtype", {}).items())
                  + f"; stored bits / weight {_fmt(sz.get('stored_bits_per_weight'), '.2f')}"]
    for key in ("space", "download", "verify", "compress", "result", "disk"):
        if r.get(key):
            lines += ["", f"## {key.capitalize()}", "", "```json", json.dumps(r[key], indent=1, sort_keys=True,
                                                                             default=str)[:6000], "```"]
    if r.get("error"):
        e = r["error"]
        lines += ["", "## Error", "", f"Step `{e['step']}` at {e['at']}: `{e['type']}: {e['message']}`", "",
                  "```", e.get("traceback", ""), "```"]
        if e.get("log_tail"):
            lines += ["", f"Log tail (`{e.get('log')}`):", "", "```", e["log_tail"], "```"]
    lines += ["", "## Environment", "", "```json", json.dumps(env, indent=1, sort_keys=True, default=str), "```"]
    return "\n".join(lines) + "\n"
