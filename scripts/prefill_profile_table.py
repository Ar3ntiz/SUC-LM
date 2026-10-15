"""Tables of the prefill profile runs (scripts/studio_prefill_profile.sh) for docs/prefill_vs_ds4.md.

Reads the newest `*_prof_<model>_<build>_<N>[_stages|_c<chunk>].json` results (scripts/frontier_measure.py generate):
wall prefill tok/s per build and prompt length, the chunk sweep, and the stage profiles (the "prefill profile <stage>
GPU <ms> ms total" lines of suc1-generate's stderr under SUC_DS4_PROFILE / SUC_GLM_PROFILE / SUC_Q4_PROFILE; every stage
boundary commits and waits, so the stage times add up to more than the wall time of the unprofiled run).

    .venv/bin/python scripts/prefill_profile_table.py [--models ds glm q38] [--top 12]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
LABEL = {"ds": "DeepSeek-V4-Flash tcq", "glm": "GLM-5.3-Flash tcq", "q38": "Qwen3.8-Flash-Next side8"}
STAGE = re.compile(r"prefill profile (\S+)\s+GPU\s+([0-9.]+) ms")


def newest(name: str) -> dict | None:
    files = sorted(p for p in RESULTS.glob(f"*_{name}.json") if p.name.split("_", 1)[1] == f"{name}.json")
    return json.loads(files[-1].read_text()) | {"_stem": files[-1].stem} if files else None


def tok_s(rec: dict | None) -> float | None:
    s = (rec or {}).get("summary") or {}
    return s.get("prefill_tok_s")


Q4 = re.compile(r"^  (\S+)\s+([0-9.]+) s\s+[0-9.]+ %", re.M)


def stages(rec: dict | None) -> dict[str, float]:
    """stage -> GPU ms of the prompt's prefill (qwen4: the last "qwen4 prefill profile" block, i.e. not the pipeline
    warm-up chunk)."""
    out: dict[str, float] = {}
    text = "\n".join((rec or {}).get("stderr_profile") or []) or (rec or {}).get("stderr_tail") or ""
    if "qwen4 prefill profile" in text:
        for m in Q4.finditer(text.rsplit("qwen4 prefill profile", 1)[1]):
            out[m.group(1)] = 1e3 * float(m.group(2))
        return out
    for m in STAGE.finditer(text):
        out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def fmt(x, nd=1):
    return "—" if x is None else f"{x:,.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="*", default=["ds", "glm", "q38"])
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    lens = (2048, 8192, 32768)
    print("| Model | build | " + " | ".join(f"{n // 1024}k tok/s" for n in lens) + " | results |")
    print("|---|---|" + "---|" * len(lens) + "---|")
    for m in a.models:
        for b in ("base", "new"):
            recs = [newest(f"prof_{m}_{b}_{n}") for n in lens]
            src = ", ".join(f"R:`{r['_stem']}`" for r in recs if r)
            print(f"| {LABEL[m]} | {b} | " + " | ".join(fmt(tok_s(r)) for r in recs) + f" | {src} |")
    print()
    print("Chunk sweep (this build, 8,192 tokens):")
    print()
    for m in a.models:
        row = [(c, tok_s(newest(f"prof_{m}_new_8192_c{c}"))) for c in (1024, 2048, 4096)]
        row = [(c, v) for c, v in row if v is not None]
        if row:
            print(f"- {LABEL[m]}: default {fmt(tok_s(newest(f'prof_{m}_new_8192')))} tok/s; "
                  + ", ".join(f"chunk {c}: {fmt(v)}" for c, v in row))
    print()
    for m in a.models:
        for n in lens:
            cols = {b: stages(newest(f"prof_{m}_{b}_{n}_stages")) for b in ("base", "new")}
            if not any(cols.values()):
                continue
            keys = sorted(set(cols["base"]) | set(cols["new"]),
                          key=lambda k: -max(cols["base"].get(k, 0.0), cols["new"].get(k, 0.0)))[: a.top]
            print(f"Stage profile, {LABEL[m]}, {n:,} tokens (GPU ms per token; stage-synced runs):")
            print()
            print("| stage | base | new |")
            print("|---|---|---|")
            for k in keys:
                print(f"| {k} | " + " | ".join(fmt(cols[b].get(k, 0.0) / n if cols[b] else None, 3)
                                               for b in ("base", "new")) + " |")
            tot = {b: sum(cols[b].values()) / n if cols[b] else None for b in ("base", "new")}
            print(f"| total | {fmt(tot['base'], 3)} | {fmt(tot['new'], 3)} |")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
