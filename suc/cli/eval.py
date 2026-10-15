"""`suc eval`: answer a test suite with several inference systems (SUC1 via C++ / Python, BF16, MLX affine) in one
session, grade the answers and compare every system with a reference system.

Design adapted from ds4-eval (ds4_eval.c; ds4, https://github.com/antirez/ds4, MIT License, Copyright (c) 2026 The
ds4.c authors): a small fixed, audited suite run through the real inference path (chat template, prefill, greedy
token-by-token decode), the final answer graded by robust extractors (suc.evaluation.answers), a thinking-budget rule
that keeps a reserve for the visible answer, a per-case report (state, prompt / generated tokens, given / correct
answer), a replayable record (--regrade re-grades a stored result with the current extractors without any model) and a
deterministic-drift gate (compare_to: generated-token counts and answers per case must not move between runs).
SUC additions: several systems per run (typically one SUC1 file through the C++ and the Python runtime) and, optional
(config `reference`, off by default), agreement with a reference system per case (same answer, identical output, common
token prefix, first divergence) plus teacher-forced next-token KL / top-1 agreement on the reference output.

Everything is greedy, so a rerun on the same machine and software reproduces the token sequences; the C++ tool and the
Python runtime differ only by rounding (batch prefill) and may diverge after many tokens.

Time cap for unattended queues: when the environment variable SUC_EVAL_DEADLINE (Unix time, seconds) is set, a case
that would start at or after it is not run and is recorded with status SKIPPED (neither graded nor an error); the run
still finishes and writes its result (block "deadline"). Cases are taken in suite order, so the suite order decides
what a cap drops first.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from suc.config import EvalExperimentConfig, config_hash, config_to_dict, load_eval_config
from suc.evaluation.answers import grade
from suc.evaluation.suite import load_suite, render_prompt_ids, select_cases
from suc.provenance import run_metadata, seed_everything, write_result

THINK_CLOSE = "</think>"


def tokens_sha(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()[:16]


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def tf_metrics(ref_logp: np.ndarray, test_logits: np.ndarray, forced: list[int]) -> dict:
    """Teacher-forced comparison on the reference output: KL(ref || test) per position, top-1 agreement, mean NLL of
    the forced (reference) tokens under both."""
    lq = _log_softmax(test_logits)
    lp = ref_logp.astype(np.float64)
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    idx = np.arange(len(forced))
    f = np.asarray(forced)
    return {"positions": int(len(forced)), "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
            "top1_agreement": float((lp.argmax(-1) == lq.argmax(-1)).mean()),
            "nll_ref": float(-lp[idx, f].mean()), "nll_test": float(-lq[idx, f].mean())}


def generate_case(runner, prompt: list[int], budget: int, think: bool, reserve: int, stop_ids: set[int],
                  close_ids: list[int]) -> dict:
    """Greedy answer of one case with the thinking-budget rule (ds4-eval hard limit, adapted to stateless runners):
    a thinking case first runs budget - reserve tokens; if it neither stopped nor closed </think>, the close tokens
    are forced and `reserve` more tokens follow; if it closed but did not stop, it continues for `reserve` tokens."""
    t0 = time.perf_counter()
    split = think and reserve > 0 and budget > reserve and close_ids
    first = runner.generate(prompt, budget - reserve if split else budget, stop_ids)
    toks, stopped, close = list(first.tokens), first.stopped, "none"
    decode = list(first.decode_ms)
    if split and not stopped:
        forced = [] if close_ids[0] in toks else list(close_ids)
        second = runner.generate(prompt + toks + forced, reserve, stop_ids)
        toks = toks + forced + list(second.tokens)
        stopped = second.stopped
        decode += list(second.decode_ms)
        close = "hard" if forced else "continued"
    return {"tokens": toks, "stopped": stopped, "think_close": close, "seconds": time.perf_counter() - t0,
            "prefill_s": first.prefill_s, "decode_ms_median": statistics.median(decode) if decode else None}


def eval_deadline() -> float | None:
    """SUC_EVAL_DEADLINE (Unix time) or None when unset / empty."""
    v = os.environ.get("SUC_EVAL_DEADLINE", "").strip()
    return float(v) if v else None


def _status(passed: bool | None) -> str:
    return "UNGRADED" if passed is None else ("PASSED" if passed else "FAILED")


def summarize_system(records: list[dict], cases: list[dict]) -> dict:
    graded = [r for r in records if r["status"] in ("PASSED", "FAILED")]
    by_source: dict[str, list[int]] = {}
    for r, c in zip(records, cases):
        if r["status"] in ("PASSED", "FAILED"):
            s = by_source.setdefault(c["source"], [0, 0])
            s[0] += r["status"] == "PASSED"
            s[1] += 1
    passed = sum(r["status"] == "PASSED" for r in graded)
    return {"passed": passed, "failed": len(graded) - passed, "graded": len(graded),
            "ungraded": sum(r["status"] == "UNGRADED" for r in records),
            "errors": sum(r["status"] == "ERROR" for r in records),
            "skipped": sum(r["status"] == "SKIPPED" for r in records),
            "pass_rate": passed / len(graded) if graded else None,
            "by_source": {k: {"passed": v[0], "graded": v[1]} for k, v in sorted(by_source.items())},
            "generated_tokens": sum(r.get("generated_tokens", 0) for r in records),
            "seconds": sum(r.get("seconds", 0.0) for r in records)}


def agreement(ref: list[dict], test: list[dict]) -> tuple[list[dict], dict]:
    """Per case and aggregate agreement of `test` with the reference system's records."""
    rows = []
    for r, t in zip(ref, test):
        if {"ERROR", "SKIPPED"} & {r["status"], t["status"]}:
            rows.append({"id": r["id"], "error": True})
            continue
        n = common_prefix(r["tokens"], t["tokens"])
        row = {"id": r["id"], "identical": r["tokens"] == t["tokens"], "common_prefix": n,
               "prefix_fraction": n / max(len(r["tokens"]), 1),
               "first_divergence": None if r["tokens"] == t["tokens"] else n,
               "answer_agree": None if r["picked"] in (None, "?") else r["picked"] == t["picked"]}
        if t.get("teacher_forced"):
            row["tf_kl_mean"] = t["teacher_forced"]["kl_mean"]
            row["tf_top1"] = t["teacher_forced"]["top1_agreement"]
        rows.append(row)
    ok = [x for x in rows if not x.get("error")]
    ans = [x["answer_agree"] for x in ok if x["answer_agree"] is not None]
    kls = [x["tf_kl_mean"] for x in ok if "tf_kl_mean" in x]
    top = [x["tf_top1"] for x in ok if "tf_top1" in x]
    summ = {"cases": len(ok), "identical_outputs": sum(x["identical"] for x in ok),
            "answer_agreement": (sum(ans) / len(ans)) if ans else None, "answer_compared": len(ans),
            "mean_prefix_fraction": float(np.mean([x["prefix_fraction"] for x in ok])) if ok else None,
            "tf_kl_mean": float(np.mean(kls)) if kls else None, "tf_top1": float(np.mean(top)) if top else None}
    return rows, summ


def drift(old: dict, new: dict) -> dict:
    """ds4-eval deterministic gate: per system and case, changes of status / answer / generated tokens."""
    changes, compared = [], 0
    for name, sys_new in new["systems"].items():
        sys_old = old.get("systems", {}).get(name)
        if not sys_old:
            continue
        old_by = {r["id"]: r for r in sys_old["cases"]}
        for r in sys_new["cases"]:
            o = old_by.get(r["id"])
            if o is None:
                continue
            compared += 1
            diff = {k: (o.get(k), r.get(k)) for k in ("status", "picked", "generated_tokens", "tokens_sha")
                    if o.get(k) != r.get(k)}
            if diff:
                changes.append({"system": name, "id": r["id"], **{k: {"old": a, "new": b} for k, (a, b) in diff.items()}})
    return {"compared": compared, "changed": len(changes), "changes": changes}


# ------------------------------------------------------------------------------------------------ markdown

def _f(x, fmt="{:.3f}", none="—") -> str:
    return none if x is None else fmt.format(x)


def render_markdown(res: dict) -> str:
    md, cfg = res["metadata"], res["config"]
    ref = res["reference"]
    lines = [f"# suc eval — {cfg['name']}", "",
             f"Suite `{res['suite']['name']}` ({len(res['cases'])} cases, sha256 {res['suite']['sha256'][:12]}), "
             f"model {md['model']}@{md['model_revision'][:12]}, git {str(md['git_commit'])[:12]}"
             f"{' (dirty)' if md.get('git_dirty') else ''}, {md['runtime'].get('chip')}, {md['timestamp']}. "
             "Greedy; " + (f"reference system `{ref}`." if ref else "no reference comparison."), "",
             "| system | kind | passed / graded | pass rate | ungraded | errors | gen tokens | time s |"
             + (" identical to ref | answer agreement | mean common prefix | TF KL vs ref | TF top-1 |" if ref else ""),
             "|---|---|---:|---:|---:|---:|---:|---:|" + ("---:|" * 5 if ref else "")]
    for name, s in res["systems"].items():
        sm, ag = s["summary"], s.get("agreement_summary") or {}
        row = (f"| {name} | {s['info'].get('kind')} | {sm['passed']} / {sm['graded']} | "
               f"{_f(sm['pass_rate'], '{:.1%}')} | {sm['ungraded']} | {sm['errors']} | {sm['generated_tokens']} | "
               f"{sm['seconds']:.0f} |")
        if ref:
            row += (f" {'ref' if name == ref else (str(ag['identical_outputs']) + ' / ' + str(ag['cases']) if ag else '—')} | "
                    f"{_f(ag.get('answer_agreement'), '{:.1%}')} | {_f(ag.get('mean_prefix_fraction'), '{:.1%}')} | "
                    f"{_f(ag.get('tf_kl_mean'), '{:.4f}')} | {_f(ag.get('tf_top1'), '{:.3f}')} |")
        lines.append(row)
    sources = sorted({c["source"] for c in res["cases"]})
    lines += ["", "Pass rate by source:", "", "| system | " + " | ".join(sources) + " |",
              "|---|" + "---:|" * len(sources)]
    for name, s in res["systems"].items():
        bs = s["summary"]["by_source"]
        lines.append(f"| {name} | " + " | ".join(
            f"{bs[x]['passed']}/{bs[x]['graded']}" if x in bs else "—" for x in sources) + " |")
    lines += ["", "## Per case", "", "State / generated tokens / given answer per system (correct answer in column 3).",
              "", "| # | case | correct | " + " | ".join(res["systems"]) + " |",
              "|---:|---|---|" + "---|" * len(res["systems"])]
    for i, c in enumerate(res["cases"]):
        cells = []
        for name, s in res["systems"].items():
            r = s["cases"][i]
            mark = {"PASSED": "✓", "FAILED": "✗", "UNGRADED": "·", "ERROR": "ERR", "SKIPPED": "skip"}[r["status"]]
            cells.append(f"{mark} {r.get('generated_tokens', 0)} `{r.get('picked') or '-'}`"
                         + ("" if r.get("think_close", "none") == "none" else f" ({r['think_close']})"))
        lines.append(f"| {i + 1} | {c['id']} | {c.get('answer', '—')} | " + " | ".join(cells) + " |")
    others = [n for n in res["systems"] if n != ref] if ref else []
    if others:
        lines += ["", f"## Agreement with `{ref}` per case", "",
                  "Common token prefix / reference length, answer agreement, teacher-forced KL (first "
                  f"{cfg['teacher_forced_positions']} reference tokens).", "",
                  "| case | " + " | ".join(others) + " |", "|---|" + "---|" * len(others)]
        for i, c in enumerate(res["cases"]):
            cells = []
            for n in others:
                a = res["systems"][n]["agreement"][i]
                if a.get("error"):
                    cells.append("ERR")
                    continue
                reflen = len(res["systems"][ref]["cases"][i]["tokens"])
                s = ("= " if a["identical"] else "") + f"{a['common_prefix']}/{reflen}"
                if a["answer_agree"] is not None:
                    s += " ans " + ("=" if a["answer_agree"] else "≠")
                if "tf_kl_mean" in a:
                    s += f" KL {a['tf_kl_mean']:.3f}"
                cells.append(s)
            lines.append(f"| {c['id']} | " + " | ".join(cells) + " |")
    if res.get("drift"):
        d = res["drift"]
        lines += ["", f"## Drift vs `{cfg['compare_to']}`", "",
                  f"{d['changed']} of {d['compared']} (system, case) pairs changed.", ""]
        if d["changes"]:
            lines += ["| system | case | change |", "|---|---|---|"]
            for ch in d["changes"]:
                lines.append(f"| {ch['system']} | {ch['id']} | " + "; ".join(
                    f"{k}: {v['old']} → {v['new']}" for k, v in ch.items() if isinstance(v, dict)) + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ run

def run_eval(cfg: EvalExperimentConfig, *, tokenizer=None, runner_factory=None, model_dir: Path | None = None,
             model_hash_value: str | None = None, config_path: str | None = None) -> dict:
    """Run the evaluation and return the result dict (not written). tokenizer / runner_factory(system_cfg) /
    model_dir / model_hash_value are injectable for tests; by default the checkpoint snapshot is resolved and the
    systems are opened with suc.evaluation.runners.open_system."""
    from suc.evaluation import runners as rn

    seed_everything(cfg.seed)
    suite = load_suite(cfg.suite)
    cases = select_cases(suite["cases"], cfg.case_ids, cfg.limit)
    need_weights = any(s.kind in ("bf16", "mlx_affine") for s in cfg.systems)
    if tokenizer is None or (runner_factory is None and need_weights and model_dir is None):
        model_dir = model_dir or rn.resolve_model_dir(cfg.model, need_weights)
    if tokenizer is None:
        tokenizer = rn.load_tokenizer(model_dir)
    if runner_factory is None:
        def runner_factory(sc):
            return rn.open_system(sc, model_dir, tokenizer=tokenizer)
    if model_hash_value is None:
        model_hash_value = rn.reference_model_hash(model_dir, cfg.systems)
    # the tokenizer's eos plus the checkpoint's end-of-turn ids (generation_config: GLM <|user|>, <|observation|>)
    stop_ids = set(int(t) for t in tokenizer.eos_token_ids) | rn.model_eos_ids(model_dir)
    close_ids = list(tokenizer.encode(THINK_CLOSE + "\n\n", add_special_tokens=False))
    ref_name = cfg.reference                    # None: grade every system, no cross-system comparison
    order = sorted(cfg.systems, key=lambda s: s.name != ref_name) if ref_name else list(cfg.systems)
    prepared = []
    for c in cases:
        think = c["think"] if cfg.think == "case" else cfg.think == "on"
        budget = min(c["max_tokens"], cfg.max_tokens or c["max_tokens"])
        prepared.append((c, think, budget, render_prompt_ids(tokenizer, c, suite.get("system"), think)))
    other_processes = rn.busy_processes()
    deadline = eval_deadline()
    systems: dict[str, dict] = {}
    t_run = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="suc_eval_") as tmp:
        for sc in order:
            print(f"[suc eval] system {sc.name} ({sc.kind})", flush=True)
            runner = runner_factory(sc)
            records = []
            try:
                for i, (c, think, budget, ids) in enumerate(prepared):
                    rec = {"id": c["id"], "prompt_tokens": len(ids), "think": think, "budget": budget}
                    if deadline is not None and time.time() >= deadline:
                        rec.update(status="SKIPPED", picked=None, tokens=[], generated_tokens=0, seconds=0.0,
                                   skipped="SUC_EVAL_DEADLINE reached before the case started")
                        records.append(rec)
                        print(f"  [{i + 1}/{len(prepared)}] SKIPPED  {c['id']} (deadline)", flush=True)
                        continue
                    try:
                        g = generate_case(runner, ids, budget, think, cfg.think_reply_reserve, stop_ids, close_ids)
                        text = tokenizer.decode(g["tokens"])
                        picked, passed = grade(c, text)
                        rec.update(status=_status(passed), picked=picked, generated_tokens=len(g["tokens"]),
                                   tokens_sha=tokens_sha(g["tokens"]), text=text, **g)
                        npos = min(cfg.teacher_forced_positions, len(g["tokens"])) if ref_name else 0
                        if npos > 0:
                            forced_path = Path(tmp, f"ref_{i}.npy")
                            if sc.name == ref_name:
                                np.save(forced_path, _log_softmax(runner.score(ids, g["tokens"][:npos])).astype(np.float16))
                            elif forced_path.exists() and systems[ref_name]["cases"][i].get("tokens"):
                                ref_toks = systems[ref_name]["cases"][i]["tokens"][:cfg.teacher_forced_positions]
                                rec["teacher_forced"] = tf_metrics(np.load(forced_path), runner.score(ids, ref_toks),
                                                                   ref_toks)
                    except Exception as e:  # unattended: record and continue
                        rec.update(status="ERROR", error=f"{type(e).__name__}: {e}", picked=None, tokens=[],
                                   generated_tokens=0, seconds=0.0)
                        if isinstance(e, rn.ServerStreamError):     # diagnostic only: never graded (audit F03)
                            rec.update(partial_text=e.partial_text, partial_chunks=e.chunks)
                    records.append(rec)
                    print(f"  [{i + 1}/{len(prepared)}] {rec['status']:<8} {c['id']} prompt={len(ids)} "
                          f"gen={rec.get('generated_tokens', 0)} picked={rec.get('picked')} "
                          f"expected={c.get('answer', '-')}", flush=True)
            finally:
                info = dict(runner.info)
                runner.close()
            systems[sc.name] = {"info": info, "cases": records, "summary": summarize_system(records, cases)}
    for name, s in systems.items():
        if ref_name and name != ref_name:
            s["agreement"], s["agreement_summary"] = agreement(systems[ref_name]["cases"], s["cases"])
    md = run_metadata(model=cfg.model.id, model_revision=cfg.model.revision, model_hash=model_hash_value,
                      dataset_hash=suite["sha256"], seed=cfg.seed, config_hash=config_hash(cfg))
    res = {"metadata": md, "config": config_to_dict(cfg), "config_path": config_path, "kind": "suc_eval",
           "suite": {"name": suite.get("suite"), "path": cfg.suite, "sha256": suite["sha256"],
                     "attribution": suite.get("attribution"), "licenses": suite.get("licenses"),
                     "system_prompt": suite.get("system")},
           "cases": cases, "reference": ref_name, "systems": systems, "other_processes": other_processes,
           "wall_s": time.perf_counter() - t_run,
           "protocol": {"decoding": "greedy", "stop_ids": sorted(stop_ids), "think_close_ids": close_ids,
                        "think_reply_reserve": cfg.think_reply_reserve,
                        "teacher_forced_positions": cfg.teacher_forced_positions}}
    if deadline is not None:
        res["deadline"] = {"env": "SUC_EVAL_DEADLINE", "epoch": deadline,
                           "skipped": {n: s["summary"]["skipped"] for n, s in systems.items()}}
    if cfg.compare_to:
        res["drift"] = drift(json.loads(Path(cfg.compare_to).read_text()), res)
    return res


def run(config_path: str | Path) -> Path:
    cfg = load_eval_config(config_path)
    res = run_eval(cfg, config_path=str(config_path))
    path = write_result(res, cfg.output_dir, cfg.name)
    path.with_suffix(".md").write_text(render_markdown(res))
    for name, s in res["systems"].items():
        sm = s["summary"]
        print(f"suc eval: {name}: {sm['passed']}/{sm['graded']} passed, {sm['ungraded']} ungraded, "
              f"{sm['errors']} errors" + (f", {sm['skipped']} skipped (deadline)" if sm.get("skipped") else ""))
    return path


def regrade(result_path: str | Path) -> int:
    """Re-grade a stored `suc eval` result with the current extractors (no model; ds4-eval --regrade-trace)."""
    res = json.loads(Path(result_path).read_text())
    by_id = {c["id"]: c for c in res["cases"]}
    rc = 0
    for name, s in res["systems"].items():
        passed = failed = changed = not_graded = 0
        for r in s["cases"]:
            if r["status"] == "ERROR" or "text" not in r:
                not_graded += 1
                continue
            picked, ok = grade(by_id[r["id"]], r["text"])
            if ok is None:
                not_graded += 1
                continue
            passed += ok
            failed += not ok
            st = _status(ok)
            if st != r["status"] or picked != r["picked"]:
                changed += 1
                print(f"{name} {r['id']}: stored {r['status']} picked={r['picked']} -> regrade {st} picked={picked} "
                      f"expected={by_id[r['id']].get('answer')}")
        print(f"suc eval regrade {name}: passed={passed} failed={failed} changed={changed} not_graded={not_graded}")
        if passed + failed == 0:
            rc = 1
    return rc


__all__ = ["run", "run_eval", "regrade", "render_markdown", "generate_case", "agreement", "drift", "tf_metrics"]
