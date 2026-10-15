#!/usr/bin/env python3
"""Per-question comparison of `suc eval` systems from different result JSONs (same suite): passed / graded per
system, the questions where the pass / fail outcome differs, and agreement of the graded answers. Markdown to stdout.

    eval_pair_compare.py LABEL=RESULT.json:SYSTEM [LABEL=RESULT.json:SYSTEM ...]

RESULT.json may be a path or a result stem under experiments/results; SYSTEM is the system name inside it (default: the
only system). The first system is the comparison base for "differs" / "same answer".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(spec: str):
    label, rest = spec.split("=", 1)
    path, _, system = rest.partition(":")
    p = Path(path)
    if not p.exists():
        p = ROOT / "experiments" / "results" / (path if path.endswith(".json") else path + ".json")
    d = json.loads(p.read_text())
    if not system:
        (system,) = d["systems"].keys()
    cases = {c["id"]: c for c in d["systems"][system]["cases"]}
    return label, p.stem, cases, {c["id"]: c for c in d["cases"]}


def main(argv: list[str]) -> int:
    sys_ = [load(s) for s in argv]
    suite = sys_[0][3]
    graded = [i for i, c in suite.items() if c.get("kind") != "open"]
    print("| system | result | passed / graded |")
    print("|---|---|---|")
    for label, stem, cases, _ in sys_:
        n = sum(cases[i]["status"] == "PASSED" for i in graded if i in cases)
        print(f"| {label} | R:`{stem}` | {n}/{len(graded)} |")
    base = sys_[0]
    print()
    diff = [i for i in graded if len({s[2].get(i, {}).get("status") for s in sys_}) > 1]
    same_ans = [sum(s[2][i].get("picked") == base[2][i].get("picked") for i in graded if i in s[2] and i in base[2])
                for s in sys_[1:]]
    for s, k in zip(sys_[1:], same_ans):
        print(f"- {s[0]} vs {base[0]}: same graded answer on {k}/{len(graded)} questions.")
    if not diff:
        print("- No question passes on one system and fails on the other.")
    for i in diff:
        q = suite[i]
        cells = "; ".join(f"{s[0]}: {s[2][i]['status'].lower()} (answer {s[2][i].get('picked')!r})" for s in sys_ if i in s[2])
        print(f"- `{i}` ({q.get('question', '')[:90]!r}, expected {q.get('answer')!r}): {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
