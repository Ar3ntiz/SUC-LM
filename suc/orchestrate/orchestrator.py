"""`suc queue`: unattended multi-model compression (download -> verify -> suc compress -> report), crash-safe.

Per model, a persisted state machine over the steps

    check -> space -> download -> verify -> compress -> report -> done

  check     HF metadata (existence, file sizes, size per dtype, config.json); is there a registered SUC adapter for
            the architecture? If not: `unsupported` (with the reason) and the queue moves on.
  space     free disk vs download size + estimated compress work space + SUC1 size, per volume. Does not fit:
            `awaiting_decision` + AWAITING_DECISION.md; the queue continues with the next model that fits. The user
            answers in decisions.yaml (approve / skip / retry / delete_source_of), re-read between models and every
            poll.decisions_seconds while only blocked models remain; the space is also re-checked by itself.
  download  snapshot_download of the pinned revision in a child process (suc.orchestrate.fetch: retries with
            backoff, resumable); HF_HUB_OFFLINE is removed from its environment.
  verify    every selected file present with the Hub size, LFS sha256 == blob name (free: the HF cache names blobs by
            sha256), shard set == model.safetensors.index.json, safetensors headers consistent with the file sizes.
            A bad file is deleted and downloaded again once.
  compress  `suc compress <generated config>` in a child process (HF_HUB_OFFLINE=1, own log); its resumable stages
            make a rerun continue where it stopped.
  report    the compress result JSON / md copied to <queue_dir>/results/, size / bpp / PPL / KL / timings extracted.

Any exception in a step: `failed` (step + full traceback + log tail in the model's report), next model. A step
that completed is never rerun (state in status.json, written atomically after every transition); a crash or a
reboot resumes the interrupted step (download and compress resume themselves). SIGTERM / SIGINT stop the running
child cleanly and leave the model `pending` at its step. A low-disk watchdog stops a download / compress child when
free space falls below space.min_free_gib and marks the model `awaiting_decision` instead of filling the disk.

Files in <queue_dir>: status.json (snapshot + heartbeat), progress.log (one line per event, tail -f),
SUMMARY.md, AWAITING_DECISION.md (only while a model waits), decisions.yaml (user), reports/<slug>.{md,json},
configs/<slug>.yaml (generated compress configs), logs/<slug>.{download,compress}.log, results/, runs/ (compress
run dirs, unless work_root is set), .lock (single instance).
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

import yaml

from suc.orchestrate import hub as H
from suc.orchestrate.config import ModelEntry, QueueConfig, deep_merge
from suc.orchestrate.report import (CompressProgress, ProgressLog, awaiting_md, duration, gib, model_report_md,
                                    summary_md, tail)
from suc.provenance import utc_timestamp

STEPS = ("check", "space", "download", "verify", "compress", "report")
TERMINAL = ("done", "failed", "unsupported", "skipped")
GIB = 2 ** 30
AUTO_RESUME_GAIN = 5 * GIB      # free-space gain that resumes a model paused by the watchdog / compress preflight


class AwaitDecision(Exception):
    def __init__(self, reason: str, summary: str, details: dict | None = None, log_line: str | None = None) -> None:
        super().__init__(f"{reason}: {summary}")
        self.reason, self.summary, self.details, self.log_line = reason, summary, details or {}, log_line


class Unsupported(Exception):
    pass


class StepFailed(RuntimeError):
    def __init__(self, msg: str, log: Path | None = None) -> None:
        super().__init__(msg)
        self.log = log


class StopRequested(BaseException):
    """SIGTERM / SIGINT: unwind to the main loop, leave the model resumable."""


class QueueLocked(RuntimeError):
    pass


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp.{path.name}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _existing(p: Path) -> Path:
    p = Path(p).expanduser().resolve()
    while not p.exists():
        p = p.parent
    return p


def default_disk_free(path: Path) -> int:
    return shutil.disk_usage(_existing(path)).free


def _device(path: Path) -> int:
    return os.stat(_existing(path)).st_dev


def dir_bytes(d: Path) -> int:
    total = 0
    if Path(d).is_dir():
        for root, _, files in os.walk(d):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    pass
    return total


def _pid_command(pid: int) -> str | None:
    """Command line of a live process (None if it does not exist or is a zombie)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    try:
        out = subprocess.run(["ps", "-o", "stat=,command=", "-p", str(pid)], capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    stat, _, cmd = out.partition(" ")
    return None if not out or stat.startswith("Z") else cmd.strip() or None


def environment(repo_dir: Path) -> dict:
    from suc.provenance import git_info, runtime_info

    return {"git": git_info(repo_dir), "runtime": runtime_info(), "argv": sys.argv, "cwd": str(repo_dir),
            "hf_hub_offline_in_orchestrator": os.environ.get("HF_HUB_OFFLINE")}


class Orchestrator:
    def __init__(self, cfg: QueueConfig, *, queue_file: str | Path | None = None, hub: H.HubClient | None = None,
                 disk_free: Callable[[Path], int] | None = None, repo_dir: str | Path | None = None,
                 env_info: dict | None = None, clock: Callable[[], float] = time.time,
                 install_signals: bool = True) -> None:
        self.cfg = cfg
        self.queue_file = str(queue_file) if queue_file else None
        self.qdir = cfg.queue_path
        self.cache = cfg.cache_path
        self.work = cfg.work_path
        self.repo = Path(repo_dir or os.getcwd()).resolve()
        self.hub = hub or H.HfHubClient(self.cache)
        self.disk_free = disk_free or default_disk_free
        self.clock = clock
        self.install_signals = install_signals
        self.env = env_info if env_info is not None else environment(self.repo)
        self.entries = cfg.ordered()
        self.index = {e.id: i + 1 for i, e in enumerate(self.entries)}
        self.plog = ProgressLog(self.qdir / "progress.log", len(self.entries))
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = "{}"
        self._child: subprocess.Popen | None = None
        self._decisions: dict = {}
        self._decisions_mtime: float | None = None
        self._warned: set[str] = set()
        self.state: dict = {}
        self.current: dict = {}

    # --------------------------------------------------------------------------------------- state / outputs

    def _load_state(self) -> None:
        p = self.qdir / "status.json"
        old = {}
        if p.exists():
            try:
                old = {r["id"]: r for r in json.loads(p.read_text()).get("models", [])}
            except (ValueError, KeyError, TypeError):
                bad = p.with_name(f"status.corrupt.{int(time.time())}.json")
                p.rename(bad)
                self.event("WARN", 0, "queue", "start", f"unreadable status.json moved to {bad.name}; starting fresh")
        models = []
        for e in self.entries:
            r = old.get(e.id)
            if r is not None and r.get("revision") != e.revision:
                self.event("WARN", self.index[e.id], e.id, "start",
                           f"revision changed {r.get('revision', '')[:12]} -> {e.revision[:12]}: state reset")
                r = None
            if r is None:
                r = {"id": e.id, "revision": e.revision, "slug": e.slug, "state": "pending", "step": None,
                     "steps": {}, "error": None, "awaiting": None, "attempts": 0}
            models.append(r)
        self.state = {"queue": self.cfg.name, "queue_file": self.queue_file, "queue_dir": str(self.qdir),
                      "pid": os.getpid(), "host": socket.gethostname(), "started": utc_timestamp(),
                      "status": "running", "heartbeat": utc_timestamp(), "current": {}, "models": models}

    def rec(self, model_id: str) -> dict:
        return next(r for r in self.state["models"] if r["id"] == model_id)

    def save(self) -> None:
        with self._lock:
            self.state["heartbeat"] = utc_timestamp()
            self.state["current"] = self.current
            self.state["counts"] = {s: sum(r["state"] == s for r in self.state["models"])
                                    for s in ("pending", "running", "awaiting_decision", *TERMINAL)}
            self._snapshot = json.dumps(self.state, indent=1, sort_keys=True, default=str)
            _atomic_write(self.qdir / "status.json", self._snapshot + "\n")

    def changed(self, r: dict | None = None) -> None:
        """After every state change: status.json, SUMMARY.md, AWAITING_DECISION.md and the model's report."""
        self.save()
        _atomic_write(self.qdir / "SUMMARY.md", summary_md(self.state))
        aw = self.qdir / "AWAITING_DECISION.md"
        if any(m["state"] == "awaiting_decision" for m in self.state["models"]):
            _atomic_write(aw, awaiting_md(self.state, self.free_now()))
        elif aw.exists():
            aw.unlink()
        if r is not None:
            self.write_report(r)

    def write_report(self, r: dict) -> None:
        d = self.qdir / "reports"
        _atomic_write(d / f"{r['slug']}.json", json.dumps({**r, "environment": self.env}, indent=1, sort_keys=True,
                                                          default=str) + "\n")
        _atomic_write(d / f"{r['slug']}.md", model_report_md(r, self.env))

    def _heartbeat(self) -> None:
        while not self._hb_stop.wait(self.cfg.poll.heartbeat_seconds):
            with self._lock:
                try:
                    obj = json.loads(self._snapshot)
                except ValueError:
                    continue
                obj["heartbeat"] = utc_timestamp()
                try:
                    _atomic_write(self.qdir / "status.json", json.dumps(obj, indent=1, sort_keys=True) + "\n")
                except OSError:
                    pass

    def event(self, level: str, index: int, who: str, step: str, msg: str) -> None:
        self.plog.line(level, index, who, step, msg)

    def ev(self, e: ModelEntry | dict, step: str, msg: str, level: str = "INFO") -> None:
        mid = e.id if isinstance(e, ModelEntry) else e["id"]
        self.event(level, self.index.get(mid, 0), mid, step, msg)

    def free_now(self) -> dict:
        out = {}
        for p in {str(self.cache), str(self.work)}:
            try:
                out[p] = self.disk_free(Path(p))
            except OSError:
                pass
        return out

    def disk_record(self, e: ModelEntry) -> dict:
        return {"free": self.free_now(), "source_bytes": dir_bytes(H.repo_cache_dir(self.cache, e.id)),
                "run_dir_bytes": dir_bytes(self.work / e.slug), "at": utc_timestamp()}

    # --------------------------------------------------------------------------------------- decisions

    def read_decisions(self) -> dict:
        p = self.qdir / "decisions.yaml"
        if not p.exists():
            self._decisions = {}
            return self._decisions
        mtime = p.stat().st_mtime
        if mtime == self._decisions_mtime:
            return self._decisions
        self._decisions_mtime = mtime
        try:
            d = yaml.safe_load(p.read_text()) or {}
            if not isinstance(d, dict):
                raise ValueError("decisions.yaml must be a mapping {model id: decision}")
            if isinstance(d.get("decisions"), dict):
                d = d["decisions"]
        except Exception as ex:
            self.event("WARN", 0, "queue", "decisions", f"decisions.yaml ignored: {type(ex).__name__}: {ex}")
            return self._decisions
        out = {}
        for k, v in d.items():
            try:
                out[str(k)] = self._normalize(v)
            except ValueError as ex:
                self.event("WARN", 0, "queue", "decisions", f"decision for {k} ignored: {ex}")
        self._decisions = out
        self.event("INFO", 0, "queue", "decisions", f"read decisions.yaml: "
                   + (", ".join(f"{k}={v['action'] or 'delete_source_of'}" for k, v in out.items()) or "empty"))
        return out

    @staticmethod
    def _normalize(v) -> dict:
        if isinstance(v, str):
            v = {"action": v}
        if not isinstance(v, dict):
            raise ValueError(f"expected a string or a mapping, got {v!r}")
        action = v.get("action", v.get("then"))
        if action not in (None, "approve", "skip", "retry"):
            raise ValueError(f"unknown action {action!r} (approve | skip | retry)")
        dels = v.get("delete_source_of") or []
        dels = [dels] if isinstance(dels, str) else list(dels)
        unknown = set(v) - {"action", "then", "delete_source_of"}
        if unknown:
            raise ValueError(f"unknown key(s) {sorted(unknown)}")
        if action is None and not dels:
            raise ValueError("empty decision")
        return {"action": action, "delete_source_of": dels}

    def decision(self, model_id: str) -> dict:
        return self._decisions.get(model_id) or {"action": None, "delete_source_of": []}

    def apply_decisions(self) -> None:
        dec = self.read_decisions()
        ids = {r["id"] for r in self.state["models"]}
        for mid, d in dec.items():
            if mid not in ids:
                if mid not in self._warned:
                    self._warned.add(mid)
                    self.event("WARN", 0, "queue", "decisions", f"decision for {mid}: not in the queue, ignored")
                continue
            r = self.rec(mid)
            # a decision acts once per edit of decisions.yaml (a standing `retry` must not loop on a failing model);
            # `approve` additionally keeps letting the space check pass (step_space reads it every time)
            token = json.dumps(d, sort_keys=True) + "|" + str(self._decisions_mtime)
            if r.get("decision_applied") == token:
                continue
            applied = False
            for target in d["delete_source_of"]:
                applied |= self.delete_source(target, requested_by=mid)
            if d["action"] == "skip" and r["state"] in ("pending", "awaiting_decision", "failed"):
                r.update(state="skipped", reason="skipped by decisions.yaml", awaiting=None, finished=utc_timestamp())
                self.ev(r, "decision", "skip: model skipped")
                applied = True
            elif d["action"] in ("approve", "retry") or d["delete_source_of"]:
                if r["state"] in ("awaiting_decision", "failed") or (d["action"] == "retry"
                                                                     and r["state"] == "unsupported"):
                    if r["state"] in ("failed", "unsupported"):
                        r["retried"] = r.get("retried", 0) + 1
                        r.pop("reason", None)
                        if r["state"] == "unsupported":
                            r["steps"].pop("check", None)
                    blocked = r.get("step")
                    if blocked in r["steps"] and r["steps"][blocked].get("status") != "done":
                        r["steps"][blocked]["status"] = "retry"
                    r.update(state="pending", awaiting=None, error=None)
                    self.ev(r, "decision", f"{d['action'] or 'delete_source_of'}: back to pending at step {blocked}")
                    applied = True
            r["decision_applied"] = token          # processed (applied, refused or not applicable now): once
            if applied:
                self.changed(r)
            else:
                self.save()

    def delete_source(self, target: str, requested_by: str) -> bool:
        """Delete the HF-cache copy of `target` (a user decision). Refused for a model that still needs it."""
        if target in {r["id"] for r in self.state["models"]}:
            t = self.rec(target)
            if t["state"] not in TERMINAL:
                self.event("WARN", self.index.get(requested_by, 0), requested_by, "decision",
                           f"delete_source_of {target} refused: it is {t['state']} and still needs its source")
                return False
        d = H.repo_cache_dir(self.cache, target)
        if not d.exists():
            self.event("INFO", self.index.get(requested_by, 0), requested_by, "decision",
                       f"delete_source_of {target}: nothing cached at {d}")
            return True
        n = dir_bytes(d)
        shutil.rmtree(d)
        self.event("WARN", self.index.get(requested_by, 0), requested_by, "decision",
                   f"delete_source_of {target}: deleted {d} ({gib(n)})")
        if target in {r["id"] for r in self.state["models"]}:
            t = self.rec(target)
            t.setdefault("disk", {})["source_deleted"] = {"at": utc_timestamp(), "bytes": n, "by": requested_by}
            t["disk"]["source_bytes"] = 0
            self.write_report(t)
        return True

    # --------------------------------------------------------------------------------------- children

    def run_child(self, cmd: list[str], log: Path, env: dict, watch: list[Path], on_poll: Callable[[], None],
                  r: dict, step: str) -> int:
        """Run cmd (own session, output appended to log) until it exits; stop it on SIGTERM / low disk."""
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(f"\n===== {utc_timestamp()} {step} attempt: {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=self.repo, env=env,
                                    start_new_session=True)
        self._child = proc
        r["child"] = {"pid": proc.pid, "marker": cmd[-1] if step == "compress" else "suc.orchestrate.fetch",
                      "step": step}
        self.save()
        floor = self.cfg.space.min_free_gib * GIB
        last_watch = 0.0
        try:
            while True:
                rc = proc.poll()
                on_poll()
                if rc is not None:
                    return rc
                if self.stop.is_set():
                    raise StopRequested()
                now = time.time()
                if now - last_watch >= self.cfg.space.watchdog_seconds:
                    last_watch = now
                    for p in watch:
                        free = self.disk_free(p)
                        if free < floor:
                            self.ev(r, step, f"low disk: {gib(free)} free at {p} < floor {gib(floor)}; stopping "
                                    f"the {step} child", "WARN")
                            self._terminate(proc)
                            raise AwaitDecision("low_disk", f"free space {gib(free)} at {p} fell below the floor "
                                                f"{gib(floor)} during {step}",
                                                {"volumes": [{"path": str(p), "free_bytes": free, "need_bytes": 0,
                                                              "margin": 1, "floor_bytes": floor,
                                                              "short_bytes": floor - free}], "paused_free": free})
                self.stop.wait(self.cfg.poll.child_seconds)
        except BaseException:
            self._terminate(proc)
            raise
        finally:
            self._child = None
            r.pop("child", None)

    @staticmethod
    def _terminate(proc: subprocess.Popen, grace: float = 120.0) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    def kill_orphans(self) -> None:
        """A child left running by a crashed orchestrator would collide with the resumed step (run-dir lock)."""
        for r in self.state["models"]:
            c = r.pop("child", None)
            if not c:
                continue
            cmd = _pid_command(int(c["pid"]))
            if cmd and c["marker"] in cmd:
                self.ev(r, c["step"], f"stopping orphaned {c['step']} child pid {c['pid']} of a previous run", "WARN")
                try:
                    os.killpg(int(c["pid"]), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    try:
                        os.kill(int(c["pid"]), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
                for _ in range(120):
                    if _pid_command(int(c["pid"])) is None:
                        break
                    time.sleep(1)

    def _env(self, offline: bool) -> dict:
        env = dict(os.environ)
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
        if offline:
            env["HF_HUB_OFFLINE"] = "1"
        env["HF_HUB_CACHE"] = str(self.cache)
        # Measured on the Studio (2026-10-03): the Xet transfer backend ran at ~0.5 MB/s vs ~8 MB/s over plain HTTP.
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        env["PYTHONUNBUFFERED"] = "1"
        return env

    # --------------------------------------------------------------------------------------- steps

    def step_check(self, e: ModelEntry, r: dict) -> None:
        meta = self.hub.metadata(e.id, e.revision)
        if meta.get("sha") and meta["sha"] != e.revision:
            raise StepFailed(f"the Hub resolved {e.id}@{e.revision} to commit {meta['sha']}")
        dl = self.cfg.download
        sel = H.select_files(meta["files"], dl.allow_patterns, dl.ignore_patterns)
        sizes = H.size_summary(meta, dl.allow_patterns, dl.ignore_patterns)
        arch = H.architecture(meta["config"])
        ok, why = H.adapter_support(meta["config"])
        r["check"] = {"arch": arch, "sizes": sizes, "adapter": why, "supported": ok, "gated": meta.get("gated"),
                      "files": [{"name": f["name"], "size": f["size"], "sha256": f.get("sha256")} for f in sel]}
        self.ev(e, "check", f"{arch['model_type']} {arch['architectures']}, {arch['num_layers']} layers, experts "
                f"{arch['num_experts']}, source quant {arch['source_quantization']}; download {gib(sizes['download_bytes'])}"
                f" in {sizes['files']} files; params " + H.params_text(sizes))
        if not ok:
            raise Unsupported(f"no registered SUC architecture adapter: {why}")
        if not sizes["safetensors_files"]:
            raise Unsupported("no safetensors weights in the selected files (D8: input only from HF safetensors)")
        if arch["source_quantization"]:
            self.ev(e, "check", f"source checkpoint is quantized ({arch['source_quantization']}): needs a "
                    "quantized-source reader / dequant options in the compress config", "WARN")

    def estimate_space(self, e: ModelEntry, r: dict) -> dict:
        c, sp = r["check"], self.cfg.space
        present = H.present_bytes(self.cache, e.id, e.revision, c["files"])
        dl_need = max(0, c["sizes"]["download_bytes"] - present)
        bpp = e.estimated_bpp or sp.estimated_bpp
        suc1 = c["sizes"]["logical_weights"] * bpp / 8
        run_dir = self.work / e.slug
        existing = dir_bytes(run_dir)
        work_need = max(0.0, suc1 * sp.workspace_factor + sp.workspace_fixed_gib * GIB - existing)
        needs: dict[int, dict] = {}
        for path, need in ((self.cache, dl_need), (self.work, work_need)):
            dev = _device(path)
            v = needs.setdefault(dev, {"path": str(path), "need_bytes": 0})
            v["need_bytes"] += int(need)
        vols = []
        for v in needs.values():
            free = int(self.disk_free(Path(v["path"])))
            floor = int(sp.min_free_gib * GIB)
            short = int(v["need_bytes"] * sp.margin + floor - free)
            vols.append({**v, "free_bytes": free, "margin": sp.margin, "floor_bytes": floor,
                         "short_bytes": max(0, short), "fits": short <= 0})
        return {"fits": all(v["fits"] for v in vols), "present_bytes": present, "download_need_bytes": dl_need,
                "suc1_estimate_bytes": int(suc1), "estimated_bpp": bpp, "work_estimate_bytes": int(work_need),
                "existing_run_dir_bytes": existing, "volumes": vols, "at": utc_timestamp()}

    def step_space(self, e: ModelEntry, r: dict) -> None:
        est = self.estimate_space(e, r)
        r["space"] = est
        msg = (f"download need {gib(est['download_need_bytes'])}, SUC1 ~{gib(est['suc1_estimate_bytes'])}, work "
               f"~{gib(est['work_estimate_bytes'])}; " + "; ".join(f"{v['path']} free {gib(v['free_bytes'])}"
                                                                   for v in est["volumes"]))
        if est["fits"]:
            self.ev(e, "space", "fits: " + msg)
            return
        if self.decision(e.id)["action"] == "approve":
            est["approved"] = True
            self.ev(e, "space", "does NOT fit but approved in decisions.yaml: " + msg, "WARN")
            return
        short = max(v["short_bytes"] for v in est["volumes"])
        raise AwaitDecision("space", f"short by {gib(short)}: {msg}", est)

    def step_download(self, e: ModelEntry, r: dict) -> None:
        files = r["check"]["files"]
        total = r["check"]["sizes"]["download_bytes"]
        present0 = H.present_bytes(self.cache, e.id, e.revision, files)
        snap = H.snapshot_dir(self.cache, e.id, e.revision)
        if present0 >= total and all((snap / f["name"]).exists() for f in files):
            r["download"] = {"skipped": "already complete in the HF cache", "bytes": 0, "seconds": 0.0,
                             "snapshot": str(snap)}
            self.ev(e, "download", f"already in the cache ({gib(total)}): {snap}")
            return
        dl = self.cfg.download
        base = list(dl.command) if dl.command else [sys.executable, "-m", "suc.orchestrate.fetch"]
        cmd = base + ["--repo", e.id, "--revision", e.revision, "--cache-dir", str(self.cache),
                      "--allow", *dl.allow_patterns, "--ignore", *dl.ignore_patterns, "--retries", str(dl.retries),
                      "--backoff", str(dl.backoff_seconds), "--max-backoff", str(dl.max_backoff_seconds),
                      "--max-workers", str(dl.max_workers)]
        log = self.qdir / "logs" / f"{e.slug}.download.log"
        start = H.cached_bytes(self.cache, e.id)
        t0 = time.time()
        last = {"t": t0, "b": start, "print": 0.0}
        self.ev(e, "download", f"start: {gib(total - present0)} to fetch ({gib(present0)} already cached) -> "
                f"{self.cache}")

        def poll() -> None:
            now = time.time()
            if now - last["print"] < self.cfg.poll.progress_seconds:
                return
            cur = H.cached_bytes(self.cache, e.id)
            done = min(total, present0 + max(0, cur - start))
            rate = (cur - last["b"]) / max(1e-6, now - last["t"])
            avg = (cur - start) / max(1e-6, now - t0)
            eta = (total - done) / avg if avg > 0 else None
            if last["print"]:
                self.ev(e, "download", f"{100 * done / max(1, total):.1f}% {gib(done)}/{gib(total)}, "
                        f"{rate / 1e6:.1f} MB/s (avg {avg / 1e6:.1f}), ETA {duration(eta)}")
            self.current.update(progress={"pct": 100 * done / max(1, total), "bytes": done, "total": total,
                                          "mb_per_s": rate / 1e6, "eta_seconds": eta})
            last.update(t=now, b=cur, print=now)
            self.save()

        rc = self.run_child(cmd, log, self._env(offline=False), [self.cache], poll, r, "download")
        seconds = time.time() - t0
        if rc != 0:
            raise StepFailed(f"download child exited with code {rc}", log)
        now_present = H.present_bytes(self.cache, e.id, e.revision, files)
        if now_present < total:          # verify names the bad files and downloads them once more
            self.ev(e, "download", f"child finished but only {gib(now_present)} of {gib(total)} are complete",
                    "WARN")
        got = max(0, H.cached_bytes(self.cache, e.id) - start)
        r["download"] = {"bytes": got, "seconds": seconds, "mb_per_s": got / max(1e-6, seconds) / 1e6,
                         "snapshot": str(snap), "log": str(log)}
        self.ev(e, "download", f"done: {gib(got)} in {duration(seconds)} ({r['download']['mb_per_s']:.1f} MB/s)")

    def verify_files(self, e: ModelEntry, r: dict) -> tuple[dict, list[str], list[str]]:
        """(info, problems, bad files). Bad files (missing / wrong size / wrong hash / broken header) can be fixed
        by downloading them again."""
        snap = H.snapshot_dir(self.cache, e.id, e.revision)
        files = r["check"]["files"]
        bad: list[str] = []
        problems: list[str] = []
        n_sha = n_hdr = 0
        for f in files:
            if self.stop.is_set():
                raise StopRequested()
            p = snap / f["name"]
            if not p.exists():
                bad.append(f["name"])
                problems.append(f"missing {f['name']}")
                continue
            size = p.stat().st_size
            if f["size"] is not None and size != f["size"]:
                bad.append(f["name"])
                problems.append(f"{f['name']}: size {size} != Hub size {f['size']}")
                continue
            if f.get("sha256") and p.is_symlink():
                blob = os.path.basename(os.path.realpath(p))
                if len(blob) == 64 and blob != f["sha256"]:
                    bad.append(f["name"])
                    problems.append(f"{f['name']}: blob {blob[:12]} != Hub sha256 {f['sha256'][:12]}")
                    continue
                n_sha += len(blob) == 64
            if f["name"].endswith(".safetensors"):
                if (why := _safetensors_header_error(p)) is not None:
                    bad.append(f["name"])
                    problems.append(f"{f['name']}: {why}")
                    continue
                n_hdr += 1
        shards = sorted(f["name"] for f in files if f["name"].endswith(".safetensors"))
        index = [f["name"] for f in files if f["name"].endswith("model.safetensors.index.json")]
        info = {"files": len(files), "safetensors": len(shards), "sha256_checked": n_sha, "headers_checked": n_hdr,
                "index": index[0] if index else None}
        if index and (snap / index[0]).exists():
            wm = json.loads((snap / index[0]).read_text()).get("weight_map", {})
            prefix = os.path.dirname(index[0])
            listed = sorted({os.path.join(prefix, v) if prefix else v for v in wm.values()})
            in_dir = sorted(s for s in shards if os.path.dirname(s) == prefix)
            info.update(index_shards=len(listed), index_tensors=len(wm))
            if listed != in_dir:
                problems.append(f"index lists {len(listed)} shards, snapshot has {len(in_dir)}: missing "
                                f"{sorted(set(listed) - set(in_dir))[:5]}, unlisted {sorted(set(in_dir) - set(listed))[:5]}")
        return info, problems, bad

    def step_verify(self, e: ModelEntry, r: dict) -> None:
        info, problems, bad = self.verify_files(e, r)
        if bad and r.get("redownloads", 0) < 1:
            r["redownloads"] = r.get("redownloads", 0) + 1
            self.ev(e, "verify", f"{len(bad)} bad file(s) ({', '.join(bad[:3])}): deleting them and downloading "
                    "again once", "WARN")
            snap = H.snapshot_dir(self.cache, e.id, e.revision)
            for name in bad:
                p = snap / name
                if p.is_symlink():
                    blob = Path(os.path.realpath(p))
                    p.unlink()
                    blob.unlink(missing_ok=True)
                elif p.exists():
                    p.unlink()
            self.step_download(e, r)
            info, problems, bad = self.verify_files(e, r)
        r["verify"] = {**info, "problems": problems}
        if problems:
            raise StepFailed("verification failed: " + "; ".join(problems[:10]))
        self.ev(e, "verify", f"ok: {info['files']} files, {info['safetensors']} shards (index "
                f"{info.get('index_shards', '—')} shards / {info.get('index_tensors', '—')} tensors), "
                f"{info['sha256_checked']} sha256 via blob names, {info['headers_checked']} headers")

    def compress_config(self, e: ModelEntry, r: dict) -> dict:
        moe = r["check"]["arch"]["is_moe"]
        tpl = e.template or (self.cfg.compress.template_moe if moe else self.cfg.compress.template_dense)
        tp = Path(tpl).expanduser()
        tp = tp if tp.is_absolute() else self.repo / tp
        d = yaml.safe_load(tp.read_text()) or {}
        d = deep_merge(d, self.cfg.overrides)
        d = deep_merge(d, e.overrides)
        mix = self.cfg.compress.calibration_mix
        if mix is not None and not any("mix" in ((o or {}).get("calibration") or {})
                                       for o in (self.cfg.overrides, e.overrides)):
            d["calibration"] = {**(d.get("calibration") or {}), "mix": mix}
        if "name" not in e.overrides:
            d["name"] = f"queue_{e.slug}"
        model = {k: v for k, v in (d.get("model") or {}).items() if k != "local_path"}
        d["model"] = {**model, "id": e.id, "revision": e.revision, "role": "target"}
        d["run_dir"] = str(self.work / e.slug)
        if "output_dir" not in e.overrides and "output_dir" not in self.cfg.overrides:
            d["output_dir"] = self.cfg.compress.output_dir
        r.setdefault("compress", {})["template"] = str(tp)
        return d

    def step_compress(self, e: ModelEntry, r: dict) -> None:
        from suc.config import load_compress_config

        d = self.compress_config(e, r)
        cfg_path = self.qdir / "configs" / f"{e.slug}.yaml"
        _atomic_write(cfg_path, f"# generated by suc queue ({self.cfg.name}) from {r['compress']['template']} + "
                                f"overrides; regenerated on every attempt\n" + yaml.safe_dump(d, sort_keys=False))
        load_compress_config(cfg_path)            # fail here (with the reason) rather than inside the child
        cc = self.cfg.compress
        base = list(cc.command) if cc.command else [sys.executable, "-m", "suc.cli.main", "compress"]
        cmd = base + [str(cfg_path)]
        log = self.qdir / "logs" / f"{e.slug}.compress.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        pos = log.stat().st_size if log.exists() else 0
        prog = CompressProgress()
        buf = {"pos": pos, "rest": ""}
        t0 = time.time()
        self.ev(e, "compress", f"start: {' '.join(cmd)} (log {log}, run dir {d['run_dir']})")

        def poll() -> None:
            try:
                with open(log, "r", errors="replace") as f:
                    f.seek(buf["pos"])
                    data = f.read()
                    buf["pos"] = f.tell()
            except OSError:
                return
            text = buf["rest"] + data
            lines = text.split("\n")
            buf["rest"] = lines.pop()
            for line in lines:
                if (out := prog.feed(line)) is not None:
                    self.ev(e, "compress", out[1], out[0])
                    self.current.update(progress={**prog.snapshot(), "elapsed_seconds": time.time() - t0})
                    self.save()

        rc = self.run_child(cmd, log, self._env(offline=True), [self.work], poll, r, "compress")
        buf["rest"] += "\n"
        poll()
        seconds = time.time() - t0
        r["compress"].update(config=str(cfg_path), log=str(log), returncode=rc, seconds=seconds,
                             run_dir=d["run_dir"], last_stage=prog.stage)
        if rc != 0:
            t = tail(log, 80)
            hit = next((ln for ln in reversed(t.splitlines()) if "insufficient disk" in ln), None)
            if hit:
                raise AwaitDecision("space", "suc compress preflight: insufficient disk", r.get("space") or {},
                                    log_line=hit.strip()[:300])
            raise StepFailed(f"suc compress exited with code {rc} (last stage {prog.stage})", log)
        st = Path(d["run_dir"]) / "status.json"
        status = json.loads(st.read_text()) if st.exists() else {}
        if status.get("state") != "done" or not status.get("result"):
            raise StepFailed(f"suc compress exited 0 but {st} says {status.get('state')!r}", log)
        r["compress"]["result_json"] = status["result"]
        self.ev(e, "compress", f"done in {duration(seconds)} -> {status['result']}")

    def step_report(self, e: ModelEntry, r: dict) -> None:
        src = Path(r["compress"]["result_json"])
        src = src if src.is_absolute() else self.repo / src
        res = json.loads(src.read_text())
        out = self.qdir / "results"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / src.name)
        if src.with_suffix(".md").exists():
            shutil.copy2(src.with_suffix(".md"), out / src.with_suffix(".md").name)
        size, q, ref = res.get("size", {}), res.get("quality", {}), res.get("reference", {})
        cont = res.get("container", {})
        r["result"] = {"result_json": str(src), "copy": str(out / src.name), "bpp": size.get("bpp"),
                       "file_bpp": size.get("file_bpp"), "bytes_total": size.get("bytes_total"),
                       "suc1_bytes": cont.get("bytes"), "suc1_path": cont.get("path"),
                       "ppl": (q.get("perplexity") or {}).get("ppl"),
                       "ppl_bf16": (ref.get("perplexity") or {}).get("ppl"),
                       "kl_mean": (q.get("kl") or {}).get("kl_mean"),
                       "top1_agreement": (q.get("kl") or {}).get("top1_agreement"),
                       "stage_seconds": {k: v.get("seconds") for k, v in (res.get("stages") or {}).items()},
                       "git_commit": (res.get("metadata") or {}).get("git_commit")}
        x = r["result"]
        self.ev(e, "report", f"bpp {x['bpp']}, SUC1 {gib(x['suc1_bytes'])}, PPL {x['ppl_bf16']} -> {x['ppl']}, "
                f"KL {x['kl_mean']}")

    # --------------------------------------------------------------------------------------- driver

    def process(self, e: ModelEntry) -> None:
        r = self.rec(e.id)
        r.update(state="running", awaiting=None, error=None, attempts=r.get("attempts", 0) + 1)
        r.setdefault("started", utc_timestamp())
        r.setdefault("disk", {})["before"] = r.get("disk", {}).get("before") or self.disk_record(e)
        self.current = {"model": e.id, "index": self.index[e.id], "step": None, "progress": {}}
        self.changed(r)
        fns = {s: getattr(self, f"step_{s}") for s in STEPS}
        step, t0 = STEPS[0], time.time()
        try:
            for step in STEPS:
                if r["steps"].get(step, {}).get("status") == "done":
                    continue
                if self.stop.is_set():
                    raise StopRequested()
                r["step"] = step
                prev = r["steps"].get(step, {})
                r["steps"][step] = {"status": "running", "started": utc_timestamp(),
                                    "previous_seconds": (prev.get("seconds") or 0) + (prev.get("previous_seconds") or 0)}
                self.current.update(step=step, progress={})
                self.ev(e, step, "begin" + (f" (resuming, was {prev['status']})" if prev else ""))
                self.changed(r)
                t0 = time.time()
                fns[step](e, r)
                r["steps"][step].update(status="done", seconds=time.time() - t0, finished=utc_timestamp())
                self.changed(r)
            r.update(state="done", step="done", finished=utc_timestamp())
            r["disk"]["after"] = self.disk_record(e)
            self.post_success(e, r)
            self.ev(e, "done", f"model done in {duration(sum(s.get('seconds') or 0 for s in r['steps'].values()))}")
        except StopRequested:
            r["steps"].setdefault(step, {}).update(status="interrupted", seconds=time.time() - t0)
            r.update(state="pending", interrupted=utc_timestamp())
            self.ev(e, step, "interrupted by a stop request; rerun the queue to resume this step", "WARN")
            raise
        except AwaitDecision as ex:
            r["steps"].setdefault(step, {}).update(status="blocked", seconds=time.time() - t0)
            r.update(state="awaiting_decision", awaiting={"reason": ex.reason, "summary": ex.summary, "step": step,
                                                          "since": utc_timestamp(), "details": ex.details,
                                                          "log_line": ex.log_line,
                                                          "paused_free": min(self.free_now().values(), default=0)})
            self.ev(e, step, f"AWAITING DECISION ({ex.reason}): {ex.summary} -> see AWAITING_DECISION.md, answer in "
                    "decisions.yaml; continuing with the next model", "WARN")
        except Unsupported as ex:
            r["steps"].setdefault(step, {}).update(status="unsupported", seconds=time.time() - t0)
            r.update(state="unsupported", reason=str(ex), finished=utc_timestamp())
            self.ev(e, step, f"UNSUPPORTED: {ex}; continuing with the next model", "WARN")
        except Exception as ex:
            log = getattr(ex, "log", None)
            r["steps"].setdefault(step, {}).update(status="failed", seconds=time.time() - t0)
            r.update(state="failed", finished=utc_timestamp(),
                     error={"step": step, "type": type(ex).__name__, "message": str(ex)[:2000],
                            "traceback": traceback.format_exc(), "at": utc_timestamp(),
                            "log": str(log) if log else None, "log_tail": tail(log, 80) if log else None})
            self.ev(e, step, f"FAILED: {type(ex).__name__}: {str(ex)[:300]}; continuing with the next model", "ERROR")
        finally:
            r.setdefault("disk", {})["last"] = self.disk_record(e)
            self.current = {}
            self.changed(r)

    def post_success(self, e: ModelEntry, r: dict) -> None:
        policy = e.source_policy or self.cfg.source_policy
        r["source_policy"] = policy
        if policy == "delete_after_success":
            d = H.repo_cache_dir(self.cache, e.id)
            n = dir_bytes(d)
            shutil.rmtree(d, ignore_errors=True)
            r["disk"]["source_deleted"] = {"at": utc_timestamp(), "bytes": n, "by": "source_policy"}
            r["disk"]["source_bytes"] = 0
            self.ev(e, "done", f"source_policy delete_after_success: deleted {d} ({gib(n)})")
        else:
            r["disk"]["source_bytes"] = dir_bytes(H.repo_cache_dir(self.cache, e.id))

    def refresh_awaiting(self) -> None:
        """Space may have been freed by hand (or by a finished model's source policy): re-check blocked models."""
        for e in self.entries:
            r = self.rec(e.id)
            if r["state"] != "awaiting_decision" or "check" not in r:
                continue
            a = r["awaiting"]
            est = self.estimate_space(e, r)
            if not est["fits"]:
                continue
            if not (a["reason"] == "space" and a["step"] == "space"):
                # the estimate already said "fits" when the watchdog / the compress preflight disagreed: resume
                # by itself only once clearly more space is free than at the pause (no stop-and-go loop)
                if min(v["free_bytes"] for v in est["volumes"]) < a.get("paused_free", 0) + AUTO_RESUME_GAIN:
                    continue
            r["space"] = est
            if a["step"] in r["steps"]:
                r["steps"][a["step"]]["status"] = "retry"
            r.update(state="pending", awaiting=None)
            self.ev(e, "space", "space now fits: back to pending")
            self.changed(r)

    def runnable(self) -> ModelEntry | None:
        for e in self.entries:
            if self.rec(e.id)["state"] in ("pending", "running"):
                return e
        return None

    def _on_signal(self, signum, frame) -> None:
        self.stop.set()
        self.event("WARN", 0, "queue", "signal", f"received {signal.Signals(signum).name}: stopping the running step "
                   "(it resumes on the next start)")

    def run(self) -> int:
        """Process the queue; returns 0 when every model is terminal, 3 when models still await a decision, 143 when
        stopped by a signal."""
        self.qdir.mkdir(parents=True, exist_ok=True)
        lockf = open(self.qdir / ".lock", "a+")
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lockf.seek(0)
            raise QueueLocked(f"queue {self.qdir} is already run by pid {lockf.read().strip() or '?'}") from None
        lockf.seek(0)
        lockf.truncate()
        lockf.write(str(os.getpid()))
        lockf.flush()
        old = {}
        if self.install_signals and threading.current_thread() is threading.main_thread():
            for s in (signal.SIGTERM, signal.SIGINT):
                old[s] = signal.signal(s, self._on_signal)
        self._hb_stop = threading.Event()
        hb = threading.Thread(target=self._heartbeat, daemon=True)
        rc = 0
        try:
            self._load_state()
            os.environ.pop("HF_HUB_OFFLINE", None)
            for r in self.state["models"]:
                if r["state"] == "running":
                    r["state"] = "pending"
                    self.ev(r, r.get("step") or "start", f"resuming at step {r.get('step')} after an unclean stop",
                            "WARN")
            self.kill_orphans()
            counts = {s: sum(r['state'] == s for r in self.state['models']) for s in ('pending', *TERMINAL,
                                                                                    'awaiting_decision')}
            self.event("INFO", 0, "queue", "start", f"{self.cfg.name}: {len(self.entries)} models "
                       + ", ".join(f"{k} {v}" for k, v in counts.items() if v)
                       + f"; pid {os.getpid()}; cache {self.cache}; work {self.work}; git "
                       f"{str((self.env.get('git') or {}).get('commit'))[:10]}")
            self.changed()
            hb.start()
            while not self.stop.is_set():
                self.apply_decisions()
                self.refresh_awaiting()
                e = self.runnable()
                if e is not None:
                    self.process(e)
                    continue
                waiting = [r for r in self.state["models"] if r["state"] == "awaiting_decision"]
                if not waiting:
                    break
                if not self.cfg.wait_for_decisions:
                    rc = 3
                    break
                self.state["status"] = "waiting for decisions"
                self.save()
                key = "waiting:" + ",".join(r["id"] for r in waiting)
                if key not in self._warned:
                    self._warned.add(key)
                    self.event("WARN", 0, "queue", "idle", f"{len(waiting)} model(s) await a decision ("
                               + ", ".join(r["id"] for r in waiting) + "): see AWAITING_DECISION.md; re-checking "
                               f"every {self.cfg.poll.decisions_seconds:.0f} s")
                self.stop.wait(self.cfg.poll.decisions_seconds)
                self.state["status"] = "running"
            if self.stop.is_set():
                raise StopRequested()
            self.state["status"] = "finished" if rc == 0 else "blocked (awaiting decisions)"
            c = self.state["models"]
            self.event("INFO", 0, "queue", "finish", f"{self.state['status']}: " + ", ".join(
                f"{s} {n}" for s in ("done", "failed", "unsupported", "skipped", "awaiting_decision")
                if (n := sum(r['state'] == s for r in c))))
        except StopRequested:
            self.state["status"] = "stopped"
            self.event("WARN", 0, "queue", "stop", "stopped; restart the same command to resume")
            rc = 143
        finally:
            self._hb_stop.set()
            if hb.is_alive():
                hb.join(5)
            self.changed()
            for s, h in old.items():
                signal.signal(s, h)
            lockf.seek(0)
            lockf.truncate()             # no stale pid for the launcher's "already running" check
            fcntl.flock(lockf, fcntl.LOCK_UN)
            lockf.close()
        return rc


def _safetensors_header_error(p: Path) -> str | None:
    """Cheap structural check: header length and the largest data offset are consistent with the file size."""
    try:
        size = p.stat().st_size
        with open(p, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return "shorter than the 8-byte header length"
            n = int.from_bytes(raw, "little")
            if n <= 0 or 8 + n > size:
                return f"header length {n} inconsistent with file size {size}"
            hdr = json.loads(f.read(n))
    except (OSError, ValueError) as ex:
        return f"unreadable header: {ex}"
    end = max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
    if 8 + n + end != size:
        return f"data ends at {8 + n + end}, file size {size}"
    return None


def stop_queue(cfg: QueueConfig) -> str:
    p = cfg.queue_path / ".lock"
    try:
        pid = int(p.read_text().strip())
    except (OSError, ValueError):
        return f"no running queue at {cfg.queue_path}"
    if _pid_command(pid) is None:
        return f"queue at {cfg.queue_path} is not running (stale pid {pid})"
    os.kill(pid, signal.SIGTERM)
    return f"sent SIGTERM to the queue orchestrator pid {pid}; it stops the running step and exits"
