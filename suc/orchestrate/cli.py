"""`suc queue <queue.yaml>` entry point (run / --once / --dry-run / --status / --stop)."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

from suc.orchestrate.config import load_queue_config


def dry_run(path: str | Path, *, hub=None, disk_free=None, repo_dir=None) -> str:
    """HF check + space estimate + generated compress config for every model; downloads / writes nothing else."""
    from suc.config import compress_config_from_dict
    from suc.orchestrate.orchestrator import Orchestrator, StepFailed, Unsupported
    from suc.orchestrate.hub import params_text
    from suc.orchestrate.report import gib

    os.environ.pop("HF_HUB_OFFLINE", None)
    cfg = load_queue_config(path)
    o = Orchestrator(cfg, queue_file=path, env_info={}, install_signals=False, hub=hub, disk_free=disk_free,
                     repo_dir=repo_dir)
    o.ev = lambda *a, **k: None                          # no progress.log lines from a dry run
    rows = [f"queue {cfg.name}: {len(o.entries)} models, queue_dir {cfg.queue_path}, cache {cfg.cache_path}, "
            f"work {cfg.work_path}"]
    for i, e in enumerate(o.entries, 1):
        r: dict = {"id": e.id}
        try:
            o.step_check(e, r)
            est = o.estimate_space(e, r)
            compress_config_from_dict(o.compress_config(e, r))
            a, s = r["check"]["arch"], r["check"]["sizes"]
            rows.append(f"[{i}] {e.id}: {a['model_type']} (experts {a['num_experts']}, source quant "
                        f"{a['source_quantization']}), params {params_text(s)}, download {gib(s['download_bytes'])}, SUC1 ~"
                        f"{gib(est['suc1_estimate_bytes'])}, fits {est['fits']}, template {r['compress']['template']}")
        except Unsupported as ex:
            c = r.get("check", {})
            a, s = c.get("arch", {}), c.get("sizes", {})
            rows.append(f"[{i}] {e.id}: UNSUPPORTED: {ex} | {a.get('model_type')} (experts {a.get('num_experts')}, "
                        f"source quant {a.get('source_quantization')}), download {gib(s.get('download_bytes'))}, "
                        f"params " + params_text(s))
        except (StepFailed, Exception) as ex:   # report every model, never stop at the first problem
            rows.append(f"[{i}] {e.id}: ERROR {type(ex).__name__}: {ex}")
    return "\n".join(rows)


def status(path: str | Path) -> str:
    cfg = load_queue_config(path)
    p = cfg.queue_path / "SUMMARY.md"
    return p.read_text() if p.exists() else f"no SUMMARY.md yet in {cfg.queue_path}"


def main(path: str, *, once: bool = False, dry: bool = False, show_status: bool = False, stop: bool = False) -> int:
    if dry:
        print(dry_run(path))
        return 0
    if show_status:
        print(status(path))
        return 0
    from suc.orchestrate.orchestrator import Orchestrator, QueueLocked, stop_queue

    cfg = load_queue_config(path)
    if stop:
        print(stop_queue(cfg))
        return 0
    if once:
        cfg = dataclasses.replace(cfg, wait_for_decisions=False)
    try:
        rc = Orchestrator(cfg, queue_file=path).run()
    except QueueLocked as ex:
        print(str(ex))
        return 2
    print(json.dumps({"queue_dir": str(cfg.queue_path), "exit": rc}))
    return rc
