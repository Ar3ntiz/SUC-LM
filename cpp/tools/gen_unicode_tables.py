"""Generate cpp/src/unicode_tables.inc: the Unicode data used by the C++ tokenizer (cpp/src/tokenizer.cpp).

Run with the project venv (needs `tokenizers`):

    .venv/bin/python cpp/tools/gen_unicode_tables.py            # writes cpp/src/unicode_tables.inc
    .venv/bin/python cpp/tools/gen_unicode_tables.py --check    # fails if the checked-in file is stale

The C++ tokenizer must reproduce HF `tokenizers` exactly, and the two Unicode-dependent parts of that library use
different Unicode versions, neither of which is necessarily Python's `unicodedata` version:

* The Split pre-tokenizer regex runs on Oniguruma. Its `\\p{L}`, `\\p{N}`, `\\s`, `\\p{M}` (Qwen3.5 variant) and
  `\\p{P}` / `\\p{S}` (DeepSeek-V4) classes are taken from the
  engine itself: every code point is pushed through `tokenizers.pre_tokenizers.Split` with that class as pattern.
  (With tokenizers 0.23 / Oniguruma this is Unicode 16, Python 3.12's unicodedata is Unicode 15.)
* The NFC normalizer (crate unicode-normalization-alignments) is based on an older Unicode version. Canonical
  combining classes, decompositions and compositions are stable once a character is assigned, so the tables are
  Python's `unicodedata` data restricted to the characters the engine knows: a character whose combining class,
  decomposition or composition the engine does not apply is treated as unassigned (ccc 0, no mapping). Each
  restriction is detected by probing `tokenizers.normalizers.NFC` / `NFD`.

The differences against plain `unicodedata` are printed, so a tokenizers upgrade that changes them is visible.
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

from tokenizers import Regex, normalizers
from tokenizers import pre_tokenizers as pt

OUT = Path(__file__).resolve().parents[1] / "src" / "unicode_tables.inc"
CPS = [c for c in range(0x110000) if not 0xD800 <= c < 0xE000]
SBASE, SCOUNT = 0xAC00, 11172


def engine_class(pattern: str) -> set[int]:
    """Code points matched by a one-character regex class in the tokenizers (Oniguruma) engine."""
    text = "".join(map(chr, CPS))
    rest: set[str] = set()
    for piece, _ in pt.Split(Regex(pattern), "removed").pre_tokenize_str(text):
        rest.update(piece)
    return {c for c in CPS if chr(c) not in rest}


def category_ranges() -> tuple[list[tuple[int, int, int]], dict]:
    L, N, S, M = engine_class(r"\p{L}"), engine_class(r"\p{N}"), engine_class(r"\s"), engine_class(r"\p{M}")
    P, Y = engine_class(r"\p{P}"), engine_class(r"\p{S}")
    uL = {c for c in CPS if unicodedata.category(chr(c))[0] == "L"}
    uN = {c for c in CPS if unicodedata.category(chr(c))[0] == "N"}
    stats = {"L": len(L), "N": len(N), "S": len(S), "M": len(M), "P": len(P), "Sym": len(Y),
             "L_vs_unicodedata": len(L ^ uL),
             "N_vs_unicodedata": len(N ^ uN)}
    if L & N or (L | N) & S or (L | N | S) & M or (L | N | S | M) & P or (L | N | S | M | P) & Y:
        raise SystemExit("unexpected overlap between \\p{L}, \\p{N}, \\s, \\p{M}, \\p{P} and \\p{S}")
    cls = {}
    for c in L:
        cls[c] = 1
    for c in N:
        cls[c] = 2
    for c in S:
        cls[c] = 4
    for c in M:
        cls[c] = 8
    for c in P:
        cls[c] = 16
    for c in Y:
        cls[c] = 32
    ranges: list[tuple[int, int, int]] = []
    for c in sorted(cls):
        if ranges and ranges[-1][1] + 1 == c and ranges[-1][2] == cls[c]:
            ranges[-1] = (ranges[-1][0], c, cls[c])
        else:
            ranges.append((c, c, cls[c]))
    return ranges, stats


def nfc_tables():
    nfc, nfd = normalizers.NFC(), normalizers.NFD()
    unknown: set[int] = set()
    # Combining classes: "A" + U+0345 (ccc 240) + X is reordered by NFC iff 0 < ccc(X) < 240.
    ccc = {c: unicodedata.combining(chr(c)) for c in CPS if unicodedata.combining(chr(c))}
    for c in ccc:
        for mark in ("ͅ", "̖"):
            s = "A" + mark + chr(c)
            if nfc.normalize_str(s) != unicodedata.normalize("NFC", s):
                unknown.add(c)
                break
    # Canonical decompositions (Hangul syllables are algorithmic and handled in C++).
    decomp = {}
    for c in CPS:
        if SBASE <= c < SBASE + SCOUNT:
            continue
        d = unicodedata.normalize("NFD", chr(c))
        if d != chr(c):
            if nfd.normalize_str(chr(c)) != d:
                unknown.add(c)
            else:
                decomp[c] = [ord(x) for x in d]
    # Primary composites: canonical pair decompositions that NFC recomposes.
    compose = {}
    for c in CPS:
        if SBASE <= c < SBASE + SCOUNT or c in unknown:
            continue
        raw = unicodedata.decomposition(chr(c))
        if not raw or raw.startswith("<"):
            continue
        parts = [int(x, 16) for x in raw.split()]
        if len(parts) != 2:
            continue
        pair = "".join(map(chr, parts))
        if unicodedata.normalize("NFC", pair) != chr(c):
            continue
        if nfc.normalize_str(pair) != chr(c):
            unknown.add(c)
            continue
        compose[(parts[0], parts[1])] = c
    ccc = {c: v for c, v in ccc.items() if c not in unknown}
    decomp = {c: v for c, v in decomp.items() if c not in unknown and not any(x in unknown for x in v)}
    compose = {k: v for k, v in compose.items() if v not in unknown and k[0] not in unknown and k[1] not in unknown}
    return ccc, decomp, compose, unknown


def render() -> tuple[str, dict]:
    ranges, stats = category_ranges()
    ccc, decomp, compose, unknown = nfc_tables()
    stats.update(ccc=len(ccc), decomp=len(decomp), compose=len(compose), nfc_unknown_to_engine=len(unknown))
    import tokenizers

    out = [
        "// Generated by cpp/tools/gen_unicode_tables.py -- do not edit by hand.",
        f"// unicodedata {unicodedata.unidata_version}, tokenizers {tokenizers.__version__}; {stats}",
        "// Regex classes from the tokenizers (Oniguruma) engine; NFC data from unicodedata restricted to the",
        "// characters known to the tokenizers NFC normalizer (see the generator's docstring).",
        "",
        "// [first, last] code point ranges: 1 = \\p{L}, 2 = \\p{N}, 4 = \\s (White_Space), 8 = \\p{M}, 16 = \\p{P}, 32 = \\p{S}.",
        "struct CatRange { uint32_t first, last; uint8_t cls; };",
        f"static const CatRange kCatRanges[{len(ranges)}] = {{",
    ]
    out += [f"  {{0x{a:X}, 0x{b:X}, {k}}}," for a, b, k in ranges]
    out += ["};", "", "// Nonzero canonical combining classes.", "struct CccEntry { uint32_t cp; uint8_t ccc; };",
            f"static const CccEntry kCcc[{len(ccc)}] = {{"]
    out += [f"  {{0x{c:X}, {v}}}," for c, v in sorted(ccc.items())]
    flat: list[int] = []
    rows = []
    for c, d in sorted(decomp.items()):
        rows.append((c, len(flat), len(d)))
        flat += d
    out += ["};", "", "// Full canonical decompositions (Hangul syllables excluded): code point, offset, length.",
            "struct DecompEntry { uint32_t cp; uint16_t offset; uint8_t len; };",
            f"static const DecompEntry kDecomp[{len(rows)}] = {{"]
    out += [f"  {{0x{c:X}, {o}, {n}}}," for c, o, n in rows]
    out += ["};", f"static const uint32_t kDecompData[{len(flat)}] = {{"]
    for i in range(0, len(flat), 12):
        out.append("  " + " ".join(f"0x{x:X}," for x in flat[i:i + 12]))
    out += ["};", "", "// Primary composites, sorted by (first, second).",
            "struct ComposeEntry { uint32_t first, second, composite; };",
            f"static const ComposeEntry kCompose[{len(compose)}] = {{"]
    out += [f"  {{0x{a:X}, 0x{b:X}, 0x{c:X}}}," for (a, b), c in sorted(compose.items())]
    out += ["};", ""]
    assert len(flat) < 65536
    return "\n".join(out), stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if the checked-in file differs")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    text, stats = render()
    print(stats, file=sys.stderr)
    if a.check:
        same = a.out.exists() and a.out.read_text() == text
        print("up to date" if same else f"STALE: {a.out}", file=sys.stderr)
        return 0 if same else 1
    a.out.write_text(text)
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
