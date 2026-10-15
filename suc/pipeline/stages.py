"""Resumable stage runner for unattended pipelines (suc compress).

A run directory holds one sub-directory per stage. A stage is COMPLETE when <stage>/manifest.json exists, its
fingerprint equals the fingerprint of the stage's current inputs, and every recorded output still exists with
its recorded size. The fingerprint is the sha256 of the canonical JSON of {stage, stage version, inputs}; the
inputs of a stage are the relevant config sections, the model hash and the OUTPUT HASHES of the upstream stages
it reads, so a changed config field or a changed upstream artifact invalidates exactly the dependent stages.

The manifest is written last, atomically (tmp + rename), so a crash inside a stage never leaves a stage that
looks complete. Outputs are written atomically too (atomic_path). The git commit that produced a stage is
recorded in its manifest but is NOT part of the fingerprint: committing unrelated code must not force a
multi-hour recompute; the report lists the commit of every stage instead.

Skip checks compare sizes, not hashes (re-hashing ~100 GB of artifacts per rerun is too slow); hashes are
computed once, when the stage runs, and propagate into downstream fingerprints.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import mlx.core as mx

from suc.io.hashing import sha256_file
from suc.provenance import utc_timestamp


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def tmp_path_for(path: Path) -> Path:
    """Temporary sibling with the same suffix (some writers append their own extension)."""
    return path.with_name(f".tmp.{path.name}")


@contextmanager
def atomic_path(path: str | Path) -> Iterator[Path]:
    """Yield a temporary path next to `path`; rename it onto `path` only if the block succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path_for(path)
    if tmp.exists():
        tmp.unlink()
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_text(path: str | Path, text: str) -> None:
    with atomic_path(path) as tmp:
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())


def atomic_write_json(path: str | Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, sort_keys=True, indent=1, default=str) + "\n")


def log(msg: str) -> None:
    """One flushed line per event (nohup-friendly)."""
    print(f"{utc_timestamp()} [compress] {msg}", flush=True)
    sys.stdout.flush()


def _gib(n: float) -> str:
    return f"{n / 2 ** 30:.2f} GiB"


class StageRunner:
    """Runs stages in a run directory; skips the complete ones (module docstring)."""

    def __init__(self, root: str | Path, *, git_commit: str | None, versions: dict[str, int]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.git_commit = git_commit
        self.versions = versions
        self.status: list[dict] = []
        self.manifests: dict[str, dict] = {}

    def stage_dir(self, stage: str) -> Path:
        return self.root / stage

    def _manifest_path(self, stage: str) -> Path:
        return self.stage_dir(stage) / "manifest.json"

    def complete(self, stage: str, fp: str) -> dict | None:
        p = self._manifest_path(stage)
        if not p.exists():
            return None
        try:
            m = json.loads(p.read_text())
        except ValueError:
            return None
        if m.get("fingerprint") != fp:
            return None
        for rel, o in m["outputs"].items():
            f = self._resolve(stage, rel)
            if not f.exists() or f.stat().st_size != o["bytes"]:
                return None
        return m

    def _stale_manifest(self, stage: str, fp: str) -> dict | None:
        """The manifest of a finished earlier run of `stage` whose fingerprint differs from fp, else None."""
        p = self._manifest_path(stage)
        try:
            m = json.loads(p.read_text()) if p.exists() else None
        except ValueError:
            return None
        return m if m and m.get("fingerprint") != fp and m.get("finished") else None

    def _resolve(self, stage: str, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.stage_dir(stage) / p

    def path(self, stage: str, rel: str) -> Path:
        """Absolute path of an output of `stage` (relative names live in the stage directory)."""
        return self._resolve(stage, rel)

    def outputs(self, stage: str) -> dict:
        """{relative path: {"sha256", "bytes"}} of a stage that has run or was skipped in this process."""
        return self.manifests[stage]["outputs"]

    def run(self, stage: str, inputs: dict, fn: Callable[[Path, str], dict], *, resumable: bool = False,
            keep_stale: bool = False) -> dict:
        """Run fn(stage_dir, fingerprint) -> {"outputs": [paths], "info": {...}, "sha256": {path: sha}?} unless the
        stage is complete. Without `resumable` the stage directory is emptied before fn runs; a resumable stage
        manages partial state itself (keyed by the fingerprint it receives). A COMPLETE earlier run of the stage with
        other inputs is invalidated explicitly: the changed input keys are logged and, with keep_stale, its directory
        is kept as <stage>.stale-<fingerprint prefix> instead of being overwritten (small, expensive stages)."""
        fp = fingerprint({"stage": stage, "version": self.versions[stage], "inputs": inputs})
        done = self.complete(stage, fp)
        if done is not None:
            self.manifests[stage] = done
            self.status.append({"stage": stage, "status": "skipped", "seconds": 0.0})
            log(f"stage {stage}: up to date, skipped (ran {done['finished']}, git {str(done['git_commit'])[:10]})")
            return done
        d = self.stage_dir(stage)
        old = self._stale_manifest(stage, fp)
        if old is not None:
            oi = old.get("inputs") or {}
            changed = sorted(k for k in set(oi) | set(inputs) if oi.get(k) != inputs.get(k))
            if old.get("version") != self.versions[stage]:
                changed.append(f"version {old.get('version')} -> {self.versions[stage]}")
            msg = f"stage {stage}: earlier complete run is stale (changed: {', '.join(changed) or '?'})"
            if keep_stale:
                keep = d.with_name(f"{stage}.stale-{str(old.get('fingerprint'))[:12]}")
                if not keep.exists():
                    d.rename(keep)
                    msg += f"; kept as {keep.name}"
            log(msg)
        if d.exists() and not resumable:
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        mp = self._manifest_path(stage)
        if mp.exists():
            mp.unlink()
        log(f"stage {stage}: start")
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        res = fn(d, fp)
        peak = int(mx.get_peak_memory())
        known = res.get("sha256", {})
        outputs = {}
        for p in res["outputs"]:
            p = Path(p)
            try:
                rel = str(p.relative_to(d))
            except ValueError:
                rel = str(p)
            outputs[rel] = {"sha256": known.get(str(p)) or sha256_file(p), "bytes": p.stat().st_size}
        seconds = time.perf_counter() - t0          # includes hashing the outputs
        manifest = {"stage": stage, "version": self.versions[stage], "fingerprint": fp, "inputs": inputs,
                    "outputs": outputs, "info": res.get("info", {}), "seconds": seconds,
                    "peak_memory_bytes": peak, "git_commit": self.git_commit, "finished": utc_timestamp()}
        atomic_write_json(mp, manifest)
        self.manifests[stage] = manifest
        self.status.append({"stage": stage, "status": "ran", "seconds": seconds, "peak_memory_bytes": peak})
        log(f"stage {stage}: done in {seconds:.1f} s, peak MLX memory {_gib(peak)}")
        return manifest
