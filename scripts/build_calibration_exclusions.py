#!/usr/bin/env python3
"""Exclusion list and disjointness record of the competition-math calibration domain ("math_competition",
calibration mix v2; suc.data.sources.EXTRA_SOURCES = the MATH TRAIN split, EleutherAI/hendrycks_math, pinned).

Evaluation problems = all 500 MATH-500 items (HuggingFaceH4/MATH-500, the revision scripts/build_hard_suite.py pins)
+ every case of every eval suite committed under experiments/eval (read from git HEAD, so a stale working tree on the
SMB share cannot drop a suite). Checks, per MATH train record:
  - by id: the train files carry no ids; every MATH-500 unique_id is checked to be "test/..." (MATH TEST split), i.e.
    disjoint from the train split by construction;
  - exact: normalized problem text (lowercase, whitespace collapsed; suc.data.sources.problem_key) equal to an
    evaluation problem's;
  - near-duplicate: shingle containment >= 0.5 with an evaluation problem: shingles = word 5-grams (words = runs of
    [a-z0-9] of the lowercased text, so LaTeX / Asymptote punctuation is ignored; a text of < 5 words is one shingle),
    containment = shared shingles / shingles of the smaller of the two texts. (A plain shared-13-gram rule over
    tokens incl. punctuation flagged 1,280 of 7,500 train problems on LaTeX / [asy] boilerplate alone; the 0.5
    containment rule flags ~1.7 %, its top hits are the same problem with one sign / one asked quantity changed.)
Writes
  suc/data/exclusions/math_competition_eval.json   {"hashes": sorted problem_keys}: every evaluation problem + every
                                                   flagged train problem (the loader drops a record whose key is listed)
  experiments/eval/calibration_disjointness.json   the record: pins, file sha256, counts, every flagged train item
and re-checks that no kept train record still matches (exact or near-duplicate).

    .venv/bin/python scripts/build_calibration_exclusions.py [--check]   # --check: verify only, write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from suc.data.sources import (  # noqa: E402
    EXCLUSIONS_DIR, EXTRA_SOURCES, MATH_EXCLUSIONS, _raw_records, problem_key,
)

MATH500 = ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test.jsonl")
RECORD = REPO / "experiments" / "eval" / "calibration_disjointness.json"
N = 5                   # shingle length (words)
THRESHOLD = 0.5         # containment that flags a near-duplicate
_WORD = re.compile(r"[a-z0-9]+")


def shingles(text: str, n: int = N) -> set[tuple[str, ...]]:
    w = _WORD.findall(text.lower())
    if not w:
        return set()
    if len(w) < n:
        return {tuple(w)}
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


def eval_problems() -> tuple[list[tuple[str, str]], dict]:
    """[(eval id, problem text)] of MATH-500 and every committed suite; provenance dict."""
    from huggingface_hub import hf_hub_download

    repo, rev, fn = MATH500
    p = Path(hf_hub_download(repo, fn, repo_type="dataset", revision=rev))
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    not_test = [r["unique_id"] for r in rows if not r["unique_id"].startswith("test/")]
    if not_test:
        raise SystemExit(f"MATH-500 ids outside the MATH test split: {not_test[:5]}")
    out = [(f"math500/{r['unique_id']}", r["problem"]) for r in rows]
    prov = {"math500": {"repo": repo, "revision": rev, "file": fn, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        "items": len(rows), "ids_all_test_split": True}, "suites": {}}
    head = _git("rev-parse", "HEAD").strip()
    for path in sorted(_git("ls-tree", "--name-only", "HEAD", "experiments/eval/").split()):
        if not path.endswith(".json") or path.endswith(RECORD.name):
            continue
        raw = _git("show", f"HEAD:{path}")
        d = json.loads(raw)
        cases = d.get("cases") if isinstance(d, dict) else None
        if not cases:
            continue
        n = 0
        for c in cases:
            q = c.get("question") or ""
            if q.strip():
                out.append((f"{Path(path).stem}:{c.get('id')}", q))
                n += 1
        prov["suites"][path] = {"sha256": hashlib.sha256(raw.encode()).hexdigest(), "cases": len(cases),
                                "questions": n}
    prov["git_head"] = head
    return out, prov


def train_records() -> tuple[list[tuple[str, int, str]], dict]:
    """[(source name, row, problem)] of every MATH train file (unfiltered); per file sha256 / rows."""
    from huggingface_hub import hf_hub_download

    out, prov = [], {}
    for name, s in EXTRA_SOURCES.items():
        if s.domain != "math_competition":
            continue
        p = hf_hub_download(s.repo_id, s.filename, repo_type="dataset", revision=s.revision)
        rows = list(_raw_records(p, s.format, ("problem",)))
        out += [(name, i, r.get("problem") or "") for i, r in enumerate(rows)]
        prov[name] = {"repo": s.repo_id, "revision": s.revision, "file": s.filename, "license": s.license,
                      "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(), "rows": len(rows)}
    return out, prov


def flag(train, evals) -> list[dict]:
    from collections import Counter, defaultdict

    ekeys = {problem_key(t): i for i, t in evals}
    esh = [shingles(t) for _, t in evals]
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for k, s in enumerate(esh):
        for g in s:
            index[g].append(k)
    hits = []
    for name, row, prob in train:
        key = problem_key(prob)
        if key in ekeys:
            hits.append({"source": name, "row": row, "rule": "exact", "eval_id": ekeys[key], "key": key})
            continue
        s = shingles(prob)
        cnt = Counter(k for g in s for k in index.get(g, ()))
        best = max(((c / min(len(s), len(esh[k])), k) for k, c in cnt.items()), default=(0.0, None))
        if best[0] >= THRESHOLD:
            hits.append({"source": name, "row": row, "rule": "near-duplicate", "eval_id": evals[best[1]][0],
                         "key": key, "containment": round(best[0], 3), "problem": prob[:200]})
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the committed exclusion file only")
    a = ap.parse_args()
    evals, eprov = eval_problems()
    train, tprov = train_records()
    hits = flag(train, evals)
    hashes = sorted({problem_key(t) for _, t in evals} | {h["key"] for h in hits})
    xfile = EXCLUSIONS_DIR / MATH_EXCLUSIONS
    if a.check:
        have = set(json.loads(xfile.read_text())["hashes"])
        miss = sorted(set(hashes) - have)
        print(f"{len(evals)} evaluation problems, {len(train)} train records, {len(hits)} flagged; "
              f"exclusion file {'OK' if not miss else f'MISSING {len(miss)} keys'}")
        sys.exit(1 if miss else 0)
    hs = set(hashes)
    kept = [(n, r, p) for n, r, p in train if problem_key(p) not in hs]
    residual = flag(kept, evals)
    if residual:
        raise SystemExit(f"{len(residual)} kept train records still overlap: {residual[:3]}")
    xfile.parent.mkdir(parents=True, exist_ok=True)
    xbody = json.dumps({"description": "problem_key (suc.data.sources) of every evaluation problem (MATH-500 + all "
                                       "experiments/eval suites) and of every MATH-train problem that duplicates one "
                                       "(exact or near-duplicate); written by scripts/build_calibration_exclusions.py",
                        "hashes": hashes}, indent=0) + "\n"
    xfile.write_text(xbody)
    rec = {
        "description": ("Disjointness of the calibration domain math_competition (MATH train, chat-rendered; "
                        "calibration mix v2) from every evaluation item: MATH-500 and every case of every suite "
                        "under experiments/eval. By id: MATH-500 ids are all MATH test split, the source is the train "
                        "split. By text: records whose normalized problem equals an evaluation problem, or whose word "
                        f"{N}-gram shingle containment with one is >= {THRESHOLD}, are dropped at load (suc.data.sources.load_source, exclusion file "
                        "below). Re-checked after exclusion: 0 overlaps."),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": eprov["git_head"],
        "evaluation": {k: v for k, v in eprov.items() if k != "git_head"},
        "evaluation_problems": len(evals),
        "calibration_sources": tprov,
        "train_records": len(train),
        "near_duplicate_rule": {"shingle_words": N, "containment_threshold": THRESHOLD},
        "flagged": {"exact": sum(h["rule"] == "exact" for h in hits),
                    "near_duplicate": sum(h["rule"] != "exact" for h in hits), "items": hits},
        "kept_records": len(kept),
        "residual_overlap_after_exclusion": 0,
        "exclusion_file": {"path": str(xfile.relative_to(REPO)), "sha256": hashlib.sha256(xbody.encode()).hexdigest(),
                           "hashes": len(hashes)},
    }
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{len(evals)} evaluation problems, {len(train)} train records: {rec['flagged']['exact']} exact, "
          f"{rec['flagged']['near_duplicate']} near-duplicates flagged, {len(kept)} kept -> {xfile.relative_to(REPO)}, "
          f"{RECORD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
