"""Import the embedded question set of ds4-eval (ds4_eval.c of the ds4 project) into a `suc eval` suite JSON.

ds4 (https://github.com/antirez/ds4, MIT License, Copyright (c) 2026 The ds4.c authors) embeds a 92-item audited
regression set in ds4_eval.c: GPQA Diamond (CC BY 4.0), an audited SuperGPQA slice (ODC-BY), AIME 2025 (MIT mirror)
and COMPSEC (reduced C/C++ vulnerability-localization items written by the ds4 authors, MIT). This script parses the
`eval_cases[]` initializer and writes the cases in the suite format of suc.evaluation.suite (kind choice / integer /
lines), keeping source, id, domain and title, plus the attribution and per-source licenses in the suite header.

GPQA Diamond items are excluded unless --include-gpqa is given: the GPQA authors ask that the questions not be
republished in plain text (training-set contamination); keep such a file local (do not commit it).

Usage:
  .venv/bin/python scripts/import_ds4_eval_cases.py /path/to/ds4/ds4_eval.c experiments/eval/ds4_subset.json
         [--include-gpqa] [--think-max-tokens 16000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

_FIELD = re.compile(r'^\s*\.(source|id|domain|title|question|answer|choice\[(\d+)\])\s*=\s*"(.*)",\s*$')
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "'": "'", "a": "\a", "b": "\b", "f": "\f",
            "v": "\v", "?": "?"}

LICENSES = {
    "GPQA Diamond": "CC BY 4.0 (https://huggingface.co/datasets/Idavidrein/gpqa)",
    "GPQA Diamond (modified)": "CC BY 4.0 (https://huggingface.co/datasets/Idavidrein/gpqa), modified by the ds4 authors",
    "SuperGPQA": "ODC-BY (https://huggingface.co/datasets/m-a-p/SuperGPQA), audited slice selected by the ds4 authors",
    "AIME2025": "MIT (mirror https://huggingface.co/datasets/test-time-compute/aime_2025)",
    "COMPSEC": "MIT (ds4 authors; reduced from public CVE write-ups)",
}


def c_unescape(s: str) -> str:
    """Decode a C string literal body (UTF-8 source, standard escapes, \\x and octal)."""
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        n = s[i + 1]
        if n in _ESCAPES:
            out.append(_ESCAPES[n])
            i += 2
        elif n == "x":
            m = re.match(r"[0-9a-fA-F]+", s[i + 2:])
            out.append(chr(int(m.group(0), 16)))
            i += 2 + len(m.group(0))
        elif n in "01234567":
            m = re.match(r"[0-7]{1,3}", s[i + 1:])
            out.append(chr(int(m.group(0), 8)))
            i += 1 + len(m.group(0))
        else:
            raise ValueError(f"unknown C escape \\{n} in {s[max(0, i - 20):i + 20]!r}")
    return "".join(out)


def parse_cases(source: str) -> list[dict]:
    start = source.index("static const eval_case eval_cases[] = {")
    end = source.index("\n};", start)
    cases, cur = [], None
    for line in source[start:end].splitlines()[1:]:
        s = line.strip()
        if s == "{":
            cur = {"choices": {}}
            continue
        if s in ("},", "}"):
            if cur is not None:
                ch = cur.pop("choices")
                cur["choices"] = [ch[k] for k in sorted(ch)]
                if list(sorted(ch)) != list(range(len(ch))):
                    raise ValueError(f"{cur.get('id')}: non-contiguous choices")
                cases.append(cur)
            cur = None
            continue
        m = _FIELD.match(line)
        if not m:
            if s:
                raise ValueError(f"unparsed line: {line[:120]!r}")
            continue
        key, idx, val = m.group(1), m.group(2), c_unescape(m.group(3))
        if idx is not None:
            cur["choices"][int(idx)] = val
        else:
            cur[key] = val
    return cases


def to_suite_case(c: dict, think_max_tokens: int) -> dict:
    kind = "choice" if c["choices"] else ("lines" if c["source"] == "COMPSEC" else "integer")
    out = {"id": f"{c['source']}/{c['id']}", "source": c["source"], "domain": c.get("domain"),
           "title": c.get("title"), "kind": kind, "question": c["question"], "answer": c["answer"],
           "prompt_style": "ds4", "think": True, "max_tokens": think_max_tokens}
    if kind == "choice":
        out["choices"] = c["choices"]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ds4_eval_c")
    ap.add_argument("out")
    ap.add_argument("--include-gpqa", action="store_true")
    ap.add_argument("--think-max-tokens", type=int, default=16000)
    a = ap.parse_args(argv)
    raw = Path(a.ds4_eval_c).read_bytes()
    cases = parse_cases(raw.decode("utf-8"))
    kept = [c for c in cases if a.include_gpqa or not c["source"].startswith("GPQA")]
    sources = sorted({c["source"] for c in kept})
    suite = {
        "suite": "ds4_subset" + ("_with_gpqa" if a.include_gpqa else ""),
        "version": 1,
        "description": "Embedded ds4-eval regression set (ds4_eval.c), in ds4 order"
                       + ("" if a.include_gpqa else ", GPQA Diamond items excluded") +
                       ". A hard capability regression suite, not a leaderboard score.",
        "attribution": "Questions, keys and audit from ds4 (https://github.com/antirez/ds4), MIT License, "
                       "Copyright (c) 2026 The ds4.c authors; imported by scripts/import_ds4_eval_cases.py from "
                       f"ds4_eval.c sha256 {hashlib.sha256(raw).hexdigest()}.",
        "licenses": {s: LICENSES.get(s, "see ds4_eval.c") for s in sources},
        "system": "You are solving a hard benchmark question. Reason carefully. "
                  "The final answer must follow the requested format exactly.",
        "cases": [to_suite_case(c, a.think_max_tokens) for c in kept],
    }
    Path(a.out).write_text(json.dumps(suite, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(cases)} cases parsed, {len(kept)} written to {a.out} ({', '.join(sources)})")


if __name__ == "__main__":
    main()
