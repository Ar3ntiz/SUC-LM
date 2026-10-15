"""Run provenance: git state, runtime environment, mandatory result metadata (spec §7 T0.4)."""
from __future__ import annotations

import datetime as _dt
import json
import platform
import random
import re
import socket
import subprocess
import sys
from pathlib import Path

REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "model", "model_hash", "git_commit", "dataset_hash", "seed", "runtime", "timestamp")


def _run(cmd: list[str]) -> str | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


RESULT_DIRS = ("experiments/results/",)  # run outputs: untracked files here do not make the tree dirty


def git_info(repo_dir: str | Path = ".") -> dict:
    """Commit hash (None before the first commit), dirty flag and branch. Never raises.

    Dirty means modified tracked files or untracked files outside RESULT_DIRS, i.e. code or
    configs that the commit hash alone would not reproduce.
    """
    g = ["git", "-C", str(repo_dir)]
    if _run([*g, "rev-parse", "--git-dir"]) is None:
        return {"commit": None, "dirty": None, "branch": None}
    status = _run([*g, "status", "--porcelain", "--untracked-files=all"])
    branch = _run([*g, "symbolic-ref", "--short", "-q", "HEAD"])
    return {"commit": _run([*g, "rev-parse", "--verify", "-q", "HEAD"]) or None,
            "dirty": None if status is None else any(
                not (line.startswith("?? ") and line[3:].startswith(RESULT_DIRS)) for line in status.splitlines()),
            "branch": branch or None}


def _sysctl(key: str) -> str | None:
    return _run(["sysctl", "-n", key]) if sys.platform == "darwin" else None


def runtime_info() -> dict:
    mem = _sysctl("hw.memsize")
    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor() or None,
        "memory_bytes": int(mem) if mem and mem.isdigit() else None,
        "hostname": socket.gethostname(),
        "mlx": None, "mlx_lm": None, "mlx_device": None,
    }
    try:
        import mlx.core as mx
        info["mlx"] = mx.__version__
        info["mlx_device"] = str(mx.default_device())
    except Exception:
        pass
    try:
        from importlib.metadata import version
        info["mlx_lm"] = version("mlx-lm")
    except Exception:
        pass
    return info


def utc_timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_metadata(*, model: str, model_revision: str, model_hash: str, dataset_hash: str, seed: int,
                 config_hash: str, repo_dir: str | Path = ".") -> dict:
    git = git_info(repo_dir)
    return {"model": model, "model_revision": model_revision, "model_hash": model_hash,
            "git_commit": git["commit"], "git_dirty": git["dirty"], "dataset_hash": dataset_hash,
            "seed": seed, "config_hash": config_hash, "runtime": runtime_info(),
            "timestamp": utc_timestamp()}


def write_result(result: dict, output_dir: str | Path, name: str) -> Path:
    """Write a result JSON as <output_dir>/<YYYYmmddTHHMMSSZ>_<name>.json; never overwrites."""
    md = result.get("metadata")
    if not isinstance(md, dict):
        raise ValueError("result must contain a 'metadata' dict")
    missing = [k for k in REQUIRED_METADATA_KEYS if k not in md]
    if missing:
        raise ValueError(f"result metadata missing required key(s) {missing}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{_dt.datetime.now(_dt.timezone.utc):%Y%m%dT%H%M%SZ}_{re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_') or 'result'}"
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    for i in range(10_000):
        path = out / (f"{stem}.json" if i == 0 else f"{stem}_{i}.json")
        try:
            with open(path, "x") as f:  # exclusive create: atomic no-overwrite
                f.write(text)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"could not find a free filename for {stem}")


def seed_everything(seed: int) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import mlx.core as mx
        mx.random.seed(seed)
    except ImportError:
        pass
