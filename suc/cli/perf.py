"""`suc perf`: prefill / time-to-first-token / decode speed and memory of several inference systems (SUC1 via C++ and
Python, MLX affine, BF16) in ONE session, sequentially, on the same token sequence.

Design adapted from ds4-bench (ds4_bench.c; ds4, https://github.com/antirez/ds4, MIT License, Copyright (c) 2026 The
ds4.c authors): one fixed real-text token sequence (a public-domain text, experiments/perf/promessi_sposi_head.txt =
the first 160 kB of ds4's cleaned Project Gutenberg #45334 "I promessi sposi"), measured at context frontiers
(explicit list, or ctx_start → ctx_max by +step_incr / ×step_mul), a fixed greedy decode that does not stop at EOS,
first-step and steady decode reported separately, KV-cache bytes per row, machine-readable rows plus a table.
Differences from ds4-bench, by design: every (system, context, batch) row is a fresh full prefill of the context (the
C++ tool has no session snapshot API; ds4 measures the incremental interval between frontiers on one live session),
batch sizes > 1 for the mlx-lm systems, `warmup` untimed runs and `repeats` timed runs with median / mean / stdev /
min / max / CV, several systems compared in the same session, and a row-by-row comparison with a previous result.
"""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from suc.config import PerfExperimentConfig, config_hash, config_to_dict, load_perf_config
from suc.provenance import run_metadata, seed_everything, write_result

MiB = 1024 * 1024
# 1 (absent): openai_server decode_ms / decode_tok_s were SSE chunk gaps; 2: those are null for HTTP rows, which
# report chunk_gap_ms / client_e2e_tok_s / server_decode_tok_s instead (audit F04). Native rows are unchanged.
PERF_METRICS_SCHEMA = 2


def stats(xs: list[float]) -> dict | None:
    xs = [float(x) for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {"median": statistics.median(xs), "mean": mean, "stdev": sd, "min": min(xs), "max": max(xs),
            "cv": sd / mean if mean else None, "n": len(xs)}


def summarize_runs(runs: list[dict], ctx: int, batch: int) -> dict:
    """Row statistics over the timed repeats; per run: prefill tok/s, TTFT, median / p90 / first decode step."""
    per = []
    for r in runs:
        d = r["decode_ms"]
        steady = d[1:] if len(d) > 1 else d
        med = statistics.median(steady) if steady else None
        per.append({"prefill_tok_s": batch * ctx / r["prefill_s"], "ttft_ms": 1e3 * r["prefill_s"],
                    "decode_ms": med, "decode_tok_s": batch * 1e3 / med if med else None,
                    "decode_first_ms": d[0] if d else None,
                    "decode_p90_ms": sorted(steady)[int(0.9 * (len(steady) - 1))] if steady else None})
        if "chunk_gap_ms" in r:
            # HTTP runner (audit F04): no per-token steps; chunk cadence and aggregate throughput, labelled as such
            g = r["chunk_gap_ms"]
            per[-1].update(chunk_gap_ms=statistics.median(g) if g else None,
                           client_e2e_tok_s=r.get("client_e2e_tok_s"), server_decode_tok_s=r.get("server_decode_tok_s"))
    out = {k: stats([p.get(k) for p in per]) for k in per[0]}
    peaks = [r.get("peak_memory_bytes") for r in runs if r.get("peak_memory_bytes") is not None]
    kv = [r.get("kv_cache_bytes") for r in runs if r.get("kv_cache_bytes")]
    out["peak_memory_bytes"] = max(peaks) if peaks else None
    out["kv_cache_bytes"] = max(kv) if kv else None
    extra = {k: runs[-1][k] for k in ("memory", "load_s", "compile_s", "decode_gpu_ms_median") if k in runs[-1]}
    return {**out, **extra}


def batch_windows(tokens: list[int], ctx: int, b: int) -> list[list[int]]:
    """b windows of ctx tokens: consecutive when the sequence is long enough, else evenly spread (overlapping)."""
    stride = ctx if b * ctx <= len(tokens) else (len(tokens) - ctx) // max(b - 1, 1)
    return [tokens[i * stride:i * stride + ctx] for i in range(b)]


def run_perf(cfg: PerfExperimentConfig, *, tokens: list[int] | None = None, runner_factory=None,
             model_dir: Path | None = None, model_hash_value: str | None = None, dataset_hash: str | None = None,
             config_path: str | None = None) -> dict:
    """Run the sweep and return the result dict (not written); tokens / runner_factory injectable for tests."""
    from suc.evaluation import runners as rn
    from suc.io.hashing import sha256_file

    seed_everything(cfg.seed)
    need_weights = any(s.kind in ("bf16", "mlx_affine") for s in cfg.systems)
    if tokens is None or (runner_factory is None and need_weights and model_dir is None):
        model_dir = model_dir or rn.resolve_model_dir(cfg.model, need_weights)
    if tokens is None:
        from mlx_lm.utils import load_tokenizer

        tok = load_tokenizer(model_dir)
        tokens = list(tok.encode(Path(cfg.prompt_file).read_text(), add_special_tokens=False))
        dataset_hash = sha256_file(cfg.prompt_file)
    if runner_factory is None:
        def runner_factory(sc):
            return rn.open_system(sc, model_dir, decode=cfg.mlx_decode, prefill_step_size=cfg.prefill_step_size)
    if model_hash_value is None:
        model_hash_value = rn.reference_model_hash(model_dir, cfg.systems)
    frontiers = cfg.frontiers()
    if len(tokens) < max(frontiers):
        raise ValueError(f"prompt has {len(tokens)} tokens, need at least {max(frontiers)} (largest context)")
    other = rn.busy_processes()
    rows, systems = [], {}
    t_run = time.perf_counter()
    for sc in cfg.systems:
        print(f"[suc perf] system {sc.name} ({sc.kind})", flush=True)
        runner = runner_factory(sc)
        try:
            for ctx in frontiers:
                for b in cfg.batch_sizes:
                    row = {"system": sc.name, "kind": sc.kind, "ctx": ctx, "batch": b, "gen_tokens": cfg.gen_tokens}
                    if not runner.supports_batch(b):
                        rows.append({**row, "status": "unsupported"})
                        continue
                    prompts = batch_windows(tokens, ctx, b)
                    try:
                        for _ in range(cfg.warmup):
                            runner.perf(prompts, min(cfg.gen_tokens, 8))
                        runs = [runner.perf(prompts, cfg.gen_tokens) for _ in range(cfg.repeats)]
                        row.update(status="ok", **summarize_runs(runs, ctx, b),
                                   first_tokens=runs[0].get("tokens"),
                                   deterministic=len({json.dumps(r.get("tokens")) for r in runs}) == 1,
                                   runs=[{k: v for k, v in r.items() if k != "memory"} for r in runs])
                    except rn.UnsupportedRun as e:
                        row.update(status="unsupported", note=str(e))
                    except Exception as e:  # unattended: record and continue
                        row.update(status="error", error=f"{type(e).__name__}: {e}")
                    rows.append(row)
                    if row["status"] == "ok" and row.get("decode_ms") is None:     # HTTP: aggregate only
                        e2e = _med(row, "client_e2e_tok_s")
                        print(f"  ctx={ctx:<6} B={b:<3} prefill {row['prefill_tok_s']['median']:9.1f} tok/s  "
                              f"TTFT {row['ttft_ms']['median']:8.1f} ms  client end-to-end "
                              f"{e2e if e2e is not None else float('nan'):7.1f} tok/s (no per-token timing)",
                              flush=True)
                    elif row["status"] == "ok":
                        print(f"  ctx={ctx:<6} B={b:<3} prefill {row['prefill_tok_s']['median']:9.1f} tok/s  "
                              f"TTFT {row['ttft_ms']['median']:8.1f} ms  decode {row['decode_ms']['median']:7.2f} "
                              f"ms/step ({row['decode_tok_s']['median']:7.1f} tok/s)", flush=True)
                    else:
                        print(f"  ctx={ctx:<6} B={b:<3} {row['status']} {row.get('error', row.get('note', ''))}",
                              flush=True)
        finally:
            systems[sc.name] = dict(runner.info)
            runner.close()
    md = run_metadata(model=cfg.model.id, model_revision=cfg.model.revision, model_hash=model_hash_value,
                      dataset_hash=dataset_hash or "injected", seed=cfg.seed, config_hash=config_hash(cfg))
    res = {"metadata": md, "config": config_to_dict(cfg), "config_path": config_path, "kind": "suc_perf",
           "metrics_schema": PERF_METRICS_SCHEMA,
           "reference": cfg.reference or cfg.systems[0].name, "frontiers": frontiers, "prompt_tokens": len(tokens),
           "systems": systems, "rows": rows, "other_processes": other, "wall_s": time.perf_counter() - t_run,
           "protocol": {"prefill": "full prefill of each context from an empty cache (TTFT = prefill + first token)",
                        "decode": f"greedy, {cfg.gen_tokens} tokens, EOS not honoured; decode_ms = median of steps "
                                  "2..n per run (first step reported separately); statistics over repeats",
                        "http": "openai_server rows (metrics_schema >= 2): decode_ms / decode_tok_s are null (an SSE "
                                "chunk may carry several tokens); chunk_gap_ms = median gap between text chunks, "
                                "client_e2e_tok_s = completion_tokens / (last chunk - request start), "
                                "server_decode_tok_s = the server's own decode timing when it reports one",
                        "mlx_decode": cfg.mlx_decode, "warmup": cfg.warmup, "repeats": cfg.repeats,
                        "batch": "b different windows of the sequence (overlapping when b × ctx exceeds it); the C++ tool decodes batch 1 only"}}
    if cfg.compare_to:
        res["comparison"] = compare(json.loads(Path(cfg.compare_to).read_text()), res)
    return res


def _key(r: dict) -> tuple:
    return r["system"], r["ctx"], r["batch"]


def _med(r: dict, k: str):
    v = r.get(k)
    return v.get("median") if isinstance(v, dict) else None


def compare(old: dict, new: dict) -> list[dict]:
    """Row-by-row ratios new / old (matching system, context, batch) of decode and prefill throughput."""
    old_by = {_key(r): r for r in old.get("rows", []) if r.get("status") == "ok"}
    out = []
    for r in new["rows"]:
        o = old_by.get(_key(r))
        if r.get("status") != "ok" or o is None:
            continue
        ratio = {}
        for k in ("decode_tok_s", "prefill_tok_s", "ttft_ms"):
            a, b = _med(o, k), _med(r, k)
            ratio[k] = b / a if a and b else None
        out.append({"system": r["system"], "ctx": r["ctx"], "batch": r["batch"],
                    "old_decode_tok_s": _med(o, "decode_tok_s"), "new_decode_tok_s": _med(r, "decode_tok_s"),
                    "ratio": ratio})
    return out


def _mem(r: dict) -> str:
    m = r.get("memory")
    if m:   # C++: mapped file + process footprint
        return f"{m.get('mapped_file', 0) / MiB:.0f} file + {m.get('phys_footprint', 0) / MiB:.0f}"
    p = r.get("peak_memory_bytes")
    return f"{p / MiB:.0f}" if p else "—"


def render_markdown(res: dict) -> str:
    md, cfg = res["metadata"], res["config"]
    ref = res["reference"]
    lines = [f"# suc perf — {cfg['name']}", "",
             f"Model {md['model']}@{md['model_revision'][:12]}, git {str(md['git_commit'])[:12]}"
             f"{' (dirty)' if md.get('git_dirty') else ''}, {md['runtime'].get('chip')}, "
             f"{(md['runtime'].get('memory_bytes') or 0) / 2**30:.0f} GiB, {md['timestamp']}. Prompt "
             f"`{cfg['prompt_file']}` ({res['prompt_tokens']} tokens); {cfg['gen_tokens']} greedy tokens; warmup "
             f"{cfg['warmup']}, repeats {cfg['repeats']} (median ± stdev); MLX decode {cfg['mlx_decode']}; all systems "
             "in this session, sequentially.", "",
             "| system | ctx | B | prefill tok/s | TTFT ms | decode ms/step | decode tok/s | first step ms | CV decode "
             "| memory MiB | KV MiB |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res["rows"]:
        if r["status"] != "ok":
            lines.append(f"| {r['system']} | {r['ctx']} | {r['batch']} | {r['status']} | | | | | | | |")
            continue
        p, t, d, s, f = (r[k] for k in ("prefill_tok_s", "ttft_ms", "decode_ms", "decode_tok_s", "decode_first_ms"))
        kv = r.get("kv_cache_bytes")
        if d is None:   # HTTP row (metrics_schema 2): no per-token decode timing
            e2e = _med(r, "client_e2e_tok_s")
            lines.append(f"| {r['system']} | {r['ctx']} | {r['batch']} | {p['median']:.1f} ± {p['stdev']:.1f} | "
                         f"{t['median']:.1f} | — (HTTP) | client e2e {e2e or 0:.1f} | — | — | {_mem(r)} | — |")
            continue
        lines.append(f"| {r['system']} | {r['ctx']} | {r['batch']} | {p['median']:.1f} ± {p['stdev']:.1f} | "
                     f"{t['median']:.1f} | {d['median']:.2f} ± {d['stdev']:.2f} | {s['median']:.1f} | "
                     f"{f['median']:.2f} | {d['cv'] or 0:.1%} | {_mem(r)} | {(kv or 0) / MiB:.1f} |")
    names = list(res["systems"])
    cells = sorted({(r["ctx"], r["batch"]) for r in res["rows"]})
    by = {_key(r): r for r in res["rows"] if r["status"] == "ok"}
    lines += ["", f"Decode tok/s (ratio vs `{ref}`):", "", "| ctx | B | " + " | ".join(names) + " |",
              "|---:|---:|" + "---:|" * len(names)]
    for ctx, b in cells:
        base = _med(by.get((ref, ctx, b), {}), "decode_tok_s")
        row = []
        for n in names:
            v = _med(by.get((n, ctx, b), {}), "decode_tok_s")
            row.append("—" if v is None else f"{v:.1f}" + (f" ({v / base:.2f}×)" if base and n != ref else ""))
        lines.append(f"| {ctx} | {b} | " + " | ".join(row) + " |")
    lines += ["", f"Prefill tok/s (ratio vs `{ref}`):", "", "| ctx | B | " + " | ".join(names) + " |",
              "|---:|---:|" + "---:|" * len(names)]
    for ctx, b in cells:
        base = _med(by.get((ref, ctx, b), {}), "prefill_tok_s")
        row = []
        for n in names:
            v = _med(by.get((n, ctx, b), {}), "prefill_tok_s")
            row.append("—" if v is None else f"{v:.0f}" + (f" ({v / base:.2f}×)" if base and n != ref else ""))
        lines.append(f"| {ctx} | {b} | " + " | ".join(row) + " |")
    if res.get("comparison") is not None:
        lines += ["", f"## Comparison with `{cfg['compare_to']}`", "",
                  "| system | ctx | B | decode tok/s old → new | decode ratio | prefill ratio | TTFT ratio |",
                  "|---|---:|---:|---|---:|---:|---:|"]
        for c in res["comparison"]:
            q = c["ratio"]
            lines.append(f"| {c['system']} | {c['ctx']} | {c['batch']} | {c['old_decode_tok_s']:.1f} → "
                         f"{c['new_decode_tok_s']:.1f} | {q['decode_tok_s'] or 0:.3f} | {q['prefill_tok_s'] or 0:.3f} | "
                         f"{q['ttft_ms'] or 0:.3f} |")
    lines += ["", "Memory: mlx-lm systems = peak MLX memory of the run (weights + activations + KV); C++ = mapped "
              "SUC1 file + process phys_footprint. Timings are only comparable within one session on one machine."]
    return "\n".join(lines) + "\n"


def run(config_path: str | Path, compare_to: str | None = None) -> Path:
    cfg = load_perf_config(config_path)
    if compare_to:
        import dataclasses

        cfg = dataclasses.replace(cfg, compare_to=compare_to)
    res = run_perf(cfg, config_path=str(config_path))
    path = write_result(res, cfg.output_dir, cfg.name)
    path.with_suffix(".md").write_text(render_markdown(res))
    return path


__all__ = ["run", "run_perf", "batch_windows", "render_markdown", "compare", "stats", "summarize_runs"]
