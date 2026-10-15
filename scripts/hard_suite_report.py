#!/usr/bin/env python3
"""Summary of the `hard` suite runs (experiments/eval/suite_hard.json): per system accuracy by subset, and per model
the paired agreement between the SUC1 file and its community GGUF (ds4 engine), written into
docs/compression_report.md between the markers below (section appended once if missing). Uses the newest result JSON
of each eval name in experiments/results; missing runs are listed as not measured.

Second block, the ENLARGED suite (experiments/eval/suite_hard20.json, 120 cases; scripts/studio_hard20.sh): accuracy
per subset per system, and per pair (SUC1 vs GGUF / LM Studio / llama-server) the per-item 2x2 table (both correct /
only SUC / only the other / both wrong) overall and per subset, with the exact two-sided McNemar p-value on the
discordant items; items SKIPPED by a time cap (SUC_EVAL_DEADLINE), ERROR or missing on either side are excluded from a
pair and counted.
Written between its own markers right after the 30-case block.

    .venv/bin/python scripts/hard_suite_report.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments" / "results"
REPORT = REPO / "docs" / "compression_report.md"
BEGIN, END = "<!-- hard-suite:begin (scripts/hard_suite_report.py) -->", "<!-- hard-suite:end -->"
SUBSETS = ["GSM8K", "MATH", "MMLU-Pro", "ARC-Challenge", "BBH", "LongCtx"]
# (model, SUC label, SUC eval name, GGUF label, GGUF eval name)
PAIRS = [
    ("DeepSeek-V4-Flash", "SUC V1 native_max (108.7 GB)", "eval_hard_ds_v1", "ds4 IQ2_XXS mix", "gguf_eval_hard_ds_iq2"),
    ("DeepSeek-V4-Flash", "SUC V3 native_stream (155.1 GB)", "eval_hard_ds_v3", "ds4 MXFP4 experts", "gguf_eval_hard_ds_mxfp4"),
    ("DeepSeek-V4-Flash", "SUC tcq (96.9 GB, resident)", "eval_hard_ds_tcq", "ds4 IQ2_XXS mix", "gguf_eval_hard_ds_iq2"),
    ("GLM-5.3-Flash", "SUC (streamed)", "eval_hard_glm", "ds4 Q2", "gguf_eval_hard_glm_q2"),
    ("GLM-5.3-Flash", "SUC tcq (experts 2.45 bpw, resident)", "eval_hard_glm_tcq", "ds4 Q2", "gguf_eval_hard_glm_q2"),
    ("Qwen3.8-Flash-Next", "SUC side8", "eval_hard_q38_side8", "ds4 Q4", "gguf_eval_hard_q38_q4"),
    ("Qwen3.8-Flash-Next", "SUC base 4.16", "eval_hard_q38_base", "ds4 Q4", "gguf_eval_hard_q38_q4"),
]


BEGIN20, END20 = "<!-- hard-suite-20:begin (scripts/hard_suite_report.py) -->", "<!-- hard-suite-20:end -->"
SUITE20 = REPO / "experiments" / "eval" / "suite_hard20.json"
# (model, SUC label, SUC eval name, other label, other eval name)
PAIRS20 = [
    ("DeepSeek-V4-Flash", "SUC tcq (96.9 GB, resident)", "eval_hard20_ds_tcq", "ds4 IQ2_XXS mix", "gguf_eval_hard20_ds_iq2"),
    # the same tcq configuration calibrated with mix v2 (GSM8K + MATH-train; compress_deepseek_v4_flash_tcq_mathcal):
    # vs the mix-v1 file and vs ds4 IQ2_XXS (scripts/studio_ds_tcq_mathcal.sh); MATH subset highlighted (HIGHLIGHT20)
    ("DeepSeek-V4-Flash", "SUC tcq calib v2 (96.9 GB, resident)", "eval_hard20_ds_tcq_mathcal",
     "SUC tcq calib v1 (96.9 GB)", "eval_hard20_ds_tcq"),
    ("DeepSeek-V4-Flash", "SUC tcq calib v2 (96.9 GB, resident)", "eval_hard20_ds_tcq_mathcal", "ds4 IQ2_XXS mix",
     "gguf_eval_hard20_ds_iq2"),
    ("GLM-5.3-Flash", "SUC tcq (experts 2.45 bpw, resident)", "eval_hard20_glm_tcq", "ds4 Q2", "gguf_eval_hard20_glm_q2"),
    ("Qwen3.8-Flash-Next", "SUC side8", "eval_hard20_q38_side8", "ds4 Q4", "gguf_eval_hard20_q38_q4"),
    # Q8_0 GGUF: llama.cpp llama-server (b11477; the engine LM Studio's GGUF runtime is built on), run by
    # scripts/studio_llama_server_eval.sh after the LM Studio Q8_0 row failed to load (`lms load` FAILED 2026-10-11)
    ("Qwen3.8-27B", "SUC side8 (16.4 GB)", "eval_hard20_q38_27b_side8", "GGUF Q8_0 (llama.cpp llama-server)",
     "gguf_eval_hard20_llamacpp_q38_27b_q8_0"),
    ("Qwen3.8-27B", "SUC side8 (16.4 GB)", "eval_hard20_q38_27b_side8", "LM Studio MLX 8-bit",
     "gguf_eval_hard20_lms_q38_27b_8bit"),
]
GRADED = ("PASSED", "FAILED")
# SUC eval name -> subset whose pairs get their own line (the question the run was made for)
HIGHLIGHT20 = {"eval_hard20_ds_tcq_mathcal": "MATH"}


def newest(name: str) -> Path | None:
    found = sorted(RESULTS.glob(f"*_{name}.json"))
    return found[-1] if found else None


def load(name: str):
    p = newest(name)
    if p is None:
        return None
    d = json.loads(p.read_text())
    (sysname, s), = d["systems"].items()
    by_src = {c["id"]: c["source"] for c in d["cases"]}
    return {"path": p, "cases": {r["id"]: r for r in s["cases"]}, "source": by_src, "summary": s["summary"],
            "suite_sha": d["suite"]["sha256"], "commit": str(d["metadata"].get("git_commit"))[:9]}


def acc_cells(r) -> list[str]:
    if r is None:
        return ["not measured"] + [""] * len(SUBSETS)
    out = []
    for sub in SUBSETS:
        ids = [i for i, s in r["source"].items() if s == sub]
        ok = sum(r["cases"][i]["status"] == "PASSED" for i in ids if i in r["cases"])
        out.append(f"{ok}/{len(ids)}")
    sm = r["summary"]
    errs = f" ({sm['errors']} err)" if sm["errors"] else ""
    return [f"**{sm['passed']}/{sm['graded']}**{errs}"] + out


def pair_cells(a, b) -> list[str]:
    if a is None or b is None:
        return ["—"] * 5
    if a["suite_sha"] != b["suite_sha"]:
        return ["suite mismatch"] + ["—"] * 4
    ids = [i for i in a["cases"] if i in b["cases"] and "ERROR" not in (a["cases"][i]["status"], b["cases"][i]["status"])]
    same_out = sum((a["cases"][i]["status"] == "PASSED") == (b["cases"][i]["status"] == "PASSED") for i in ids)
    same_ans = sum(a["cases"][i].get("picked") == b["cases"][i].get("picked") for i in ids)
    only_a = [i for i in ids if a["cases"][i]["status"] == "PASSED" and b["cases"][i]["status"] != "PASSED"]
    only_b = [i for i in ids if b["cases"][i]["status"] == "PASSED" and a["cases"][i]["status"] != "PASSED"]
    return [f"{same_out}/{len(ids)}", f"{same_ans}/{len(ids)}", str(len(only_a)), str(len(only_b)),
            "; ".join([f"SUC: {i}" for i in only_a] + [f"GGUF: {i}" for i in only_b]) or "none"]


def render() -> str:
    lines = [BEGIN, "", "### Hard suite: does the KL gap show in answers? (SUC1 vs GGUF, 30 questions)", "",
             "Suite `experiments/eval/suite_hard.json` (scripts/build_hard_suite.py): 5 deliberately chosen items per "
             "subset (GSM8K test, MATH-500 L3-L5, MMLU-Pro 10 options, ARC-Challenge, BBH 5 tasks, synthetic multi-hop "
             "needles in PG19 at ~8k/8k/16k/16k/32k tokens), greedy, thinking off, ds4-style `Answer:` line, same "
             "prompt token ids for both systems of a model (SUC1 via the C++ runtime, GGUF via the ds4 engine). With 5 "
             "items per subset a single question moves a subset by 20 points: read the paired columns (same outcome / "
             "which system alone solved what), not the accuracy as a benchmark score. Queue: scripts/studio_hard_suite.sh.",
             "", "| model | system | total | " + " | ".join(SUBSETS) + " | result |",
             "|---|---|---:|" + "---:|" * len(SUBSETS) + "---|"]
    seen = set()
    loaded = {}
    for model, la, na, lb, nb in PAIRS:
        for lab, name in ((la, na), (lb, nb)):
            if name in seen:
                continue
            seen.add(name)
            loaded[name] = r = load(name)
            res = f"`{r['path'].stem}` ({r['commit']})" if r else "—"
            lines.append(f"| {model} | {lab} | " + " | ".join(acc_cells(r)) + f" | {res} |")
    lines += ["", "Paired agreement per model (cases where both systems produced an answer):", "",
              "| model | SUC1 | GGUF | same pass/fail | same extracted answer | only SUC correct | only GGUF correct | "
              "differing cases |", "|---|---|---|---:|---:|---:|---:|---|"]
    for model, la, na, lb, nb in PAIRS:
        lines.append(f"| {model} | {la} | {lb} | " + " | ".join(pair_cells(loaded[na], loaded[nb])) + " |")
    lines += ["", END]
    return "\n".join(lines)


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test: binomial(b + c, 1/2) on the discordant counts."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def acc20_cells(r, sha: str | None) -> list[str]:
    if r is None:
        return ["not measured"] + [""] * len(SUBSETS)
    out = []
    for sub in SUBSETS:
        recs = [r["cases"].get(i) for i, s in r["source"].items() if s == sub]
        g = [x for x in recs if x and x["status"] in GRADED]
        ok = sum(x["status"] == "PASSED" for x in g)
        miss = len(recs) - len(g)
        out.append(f"{ok}/{len(g)}" + (f" ({miss} n/a)" if miss else ""))
    sm = r["summary"]
    extra = [f"{sm[k]} {lab}" for k, lab in (("errors", "err"), ("skipped", "skipped")) if sm.get(k)]
    tot = f"**{sm['passed']}/{sm['graded']}**" + (f" ({', '.join(extra)})" if extra else "")
    if sha and r["suite_sha"] != sha:
        tot += " (suite file changed since)"
    return [tot] + out


def pair20(a, b) -> dict | None:
    """Per-item 2x2 of two runs over the items both graded; None if a run is missing / suites differ."""
    if a is None or b is None or a["suite_sha"] != b["suite_sha"]:
        return None
    ids = list(a["source"])
    both = [i for i in ids if a["cases"].get(i, {}).get("status") in GRADED and b["cases"].get(i, {}).get("status") in GRADED]
    out = {"n": len(both), "excluded": len(ids) - len(both), "sub": {}}
    for sub in [None] + SUBSETS:
        sel = [i for i in both if sub is None or a["source"][i] == sub]
        pa = {i: a["cases"][i]["status"] == "PASSED" for i in sel}
        pb = {i: b["cases"][i]["status"] == "PASSED" for i in sel}
        cell = {"both": sum(pa[i] and pb[i] for i in sel), "only_a": [i for i in sel if pa[i] and not pb[i]],
                "only_b": [i for i in sel if pb[i] and not pa[i]], "neither": sum(not pa[i] and not pb[i] for i in sel),
                "same_answer": sum(a["cases"][i].get("picked") == b["cases"][i].get("picked") for i in sel), "n": len(sel)}
        if sub is None:
            out.update(cell)
        else:
            out["sub"][sub] = cell
    return out


def render20() -> str:
    sha = None
    if SUITE20.exists():
        import hashlib

        sha = hashlib.sha256(SUITE20.read_bytes()).hexdigest()
    lines = [BEGIN20, "", "### Hard suite, enlarged (120 cases): SUC1 vs GGUF / LM Studio per item", "",
             "Suite `experiments/eval/suite_hard20.json` (`scripts/build_hard_suite.py --per-subset 20`, sha256 "
             f"`{sha[:12] if sha else 'missing'}`): the selection rules of the 30-case suite applied in rounds, 20 items "
             "per subset (round 0 = the 30 cases above, same ids and prompts; selection rules and ids in the suite's "
             "`selection` block). Greedy, thinking off, ds4-style `Answer:` line, same prompt token ids for every "
             "system of a model (SUC1 via the C++ runtime; GGUF via the ds4 engine; LM Studio MLX and the Qwen3.8-27B Q8_0 "
             "GGUF via an OpenAI-compatible server -- LM Studio, resp. llama.cpp's llama-server, the engine LM Studio "
             "uses for GGUF -- with the audit F03 / F04 harness, prompt = the same rendered text, ctx 40960, parallel 1, "
             "no speculative / MTP decoding). Queue: scripts/studio_hard20.sh (+ scripts/studio_llama_server_eval.sh); "
             "per-job time cap: cases not started before it are SKIPPED (\"n/a\" below) and left out of the pair "
             "statistics. Still small samples: read a difference through the McNemar column (exact two-sided p on the "
             "discordant items), not the accuracy.", "",
             "| model | system | total | " + " | ".join(SUBSETS) + " | result |",
             "|---|---|---:|" + "---:|" * len(SUBSETS) + "---|"]
    seen, loaded = set(), {}
    for model, la, na, lb, nb in PAIRS20:
        for lab, name in ((la, na), (lb, nb)):
            if name in seen:
                continue
            seen.add(name)
            loaded[name] = r = load(name)
            res = f"`{r['path'].stem}` ({r['commit']})" if r else "—"
            lines.append(f"| {model} | {lab} | " + " | ".join(acc20_cells(r, sha)) + f" | {res} |")
    lines += ["", "Paired per item (items graded by both systems): A = the SUC1 file, B = the other system.", "",
              "| model | A (SUC1) | B | items | both correct | only A | only B | both wrong | same answer | "
              "McNemar p | excluded |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    details, highlight = [], []
    for model, la, na, lb, nb in PAIRS20:
        a, b = loaded[na], loaded[nb]
        pr = pair20(a, b)
        if pr is None:
            why = "not measured" if a is None or b is None else "suite mismatch"
            lines.append(f"| {model} | {la} | {lb} | {why} | | | | | | | |")
            continue
        lines.append(f"| {model} | {la} | {lb} | {pr['n']} | {pr['both']} | {len(pr['only_a'])} | {len(pr['only_b'])} "
                     f"| {pr['neither']} | {pr['same_answer']} | {mcnemar_p(len(pr['only_a']), len(pr['only_b'])):.3g} "
                     f"| {pr['excluded']} |")
        details.append((model, la, lb, pr))
        highlight.append((model, la, lb, pr, na))
    if details:
        lines += ["", "Per subset (both correct / only A / only B / both wrong):", "",
                  "| model | B | " + " | ".join(SUBSETS) + " |", "|---|---|" + "---:|" * len(SUBSETS)]
        for model, la, lb, pr in details:
            lines.append(f"| {model} | {lb} | " + " | ".join(
                f"{c['both']}/{len(c['only_a'])}/{len(c['only_b'])}/{c['neither']}" if c["n"] else "—"
                for c in (pr["sub"][s] for s in SUBSETS)) + " |")
        hl = [(model, la, lb, pr, HIGHLIGHT20[na]) for model, la, lb, pr, na in highlight if na in HIGHLIGHT20]
        if hl:
            lines += ["", "Highlighted subsets (the question of the run; paired items graded by both):", ""]
            for model, la, lb, pr, sub in hl:
                c = pr["sub"][sub]
                lines.append(f"- **{model}, {sub}: {la} {c['both'] + len(c['only_a'])}/{c['n']} vs {lb} "
                             f"{c['both'] + len(c['only_b'])}/{c['n']}** (only A {len(c['only_a'])}, only B "
                             f"{len(c['only_b'])}, McNemar p {mcnemar_p(len(c['only_a']), len(c['only_b'])):.3g}); "
                             f"only A: " + (", ".join(f"`{i}`" for i in c["only_a"]) or "none") + "; only B: "
                             + (", ".join(f"`{i}`" for i in c["only_b"]) or "none") + ".")
        lines += ["", "Discordant items:", ""]
        for model, la, lb, pr in details:
            lines.append(f"- {model}, {la} vs {lb}: only A: " + (", ".join(f"`{i}`" for i in pr["only_a"]) or "none")
                         + "; only B: " + (", ".join(f"`{i}`" for i in pr["only_b"]) or "none") + ".")
    lines += ["", END20]
    return "\n".join(lines)


def _put(text: str, begin: str, end: str, block: str, after: str | None = None) -> str:
    """Replace the block between begin / end; else insert it right after the marker `after`; else append."""
    if begin in text and end in text:
        pre, rest = text.split(begin, 1)
        return pre + block + rest.split(end, 1)[1]
    if after and after in text:
        pre, rest = text.split(after, 1)
        return pre + after + "\n\n" + block + rest
    return text.rstrip("\n") + "\n\n" + block + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=["hard", "hard20"], help="regenerate one block only (default both)")
    a = ap.parse_args()
    blocks = [(BEGIN, END, render, None), (BEGIN20, END20, render20, END)]
    blocks = [b for b in blocks if a.only is None or (b[0] == BEGIN20) == (a.only == "hard20")]
    if a.dry_run:
        for *_, fn, _ in blocks:
            print(fn())
        return
    text = REPORT.read_text()
    for begin, end, fn, after in blocks:
        text = _put(text, begin, end, fn(), after)
    REPORT.write_text(text)
    print(f"updated {REPORT}")


if __name__ == "__main__":
    main()
