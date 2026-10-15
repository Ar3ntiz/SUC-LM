"""Final-answer extraction and grading for `suc eval` (multiple choice, exact integer, source-line sets).

The logic is a Python port of the ds4-eval answer extractors (ds4_eval.c: find_last_answer_marker,
mc_letter_is_negated, find_answer_letter, find_integer_answer, normalize_compsec_line_spec, compsec_answer_matches;
ds4, https://github.com/antirez/ds4, MIT License, Copyright (c) 2026 The ds4.c authors), re-implemented, not
translated line by line. Semantics kept from ds4:
  - grading looks only at the text after the first `</think>` (the visible answer);
  - the strict marker is the LAST "answer" word followed by optional spaces and ':'; without one, the first loose
    "answer" word is used as the marker;
  - multiple choice: the first standalone in-range letter after the marker (96-character window) that does not begin a
    prose word ("A careful ...", "I'll") and is not the object of a rejection cue ("not B", "rules out C"); otherwise
    the last standalone in-range letter of the visible text;
  - integer: on the marker line, the first integer right of the LAST '=' (else the first integer of the line), leading
    zeros dropped; otherwise the last integer of the visible text;
  - line sets: the marker line normalized to "9,15" / "20-22"; a model answer matches when every line it names lies in
    the accepted (audited) set and at least one does.
SUC additions (prompts that do not ask for an "Answer:" line, e.g. experiments/qa/prompts.json): without a strict marker,
the last `\\boxed{...}` (Qwen3's usual final-answer form) is taken before the loose marker, a `\\boxed{...}` on the
answer line wins over other numbers of that line; integers keep thousands separators ("1,024" -> 1024).
"""
from __future__ import annotations

import re

_THINK_CLOSE = "</think>"
_NEG_CUES = {"not", "except", "excluding", "exclude", "excludes", "eliminate", "eliminates", "eliminated",
             "reject", "rejects", "rejected", "rejecting"}
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_INT = re.compile(r"\d{1,3}(?:,\d{3})+(?!\d)|\d+")


def visible_text(text: str) -> str:
    """Text after the first `</think>` (the whole text when there is none)."""
    i = text.find(_THINK_CLOSE)
    return text if i < 0 else text[i + len(_THINK_CLOSE):]


def _strict_marker(vis: str) -> int | None:
    """Index of the LAST standalone 'answer' followed by optional whitespace and ':'."""
    last = None
    for m in re.finditer(r"answer", vis, flags=re.IGNORECASE):
        a, b = m.start(), m.end()
        before = vis[a - 1] if a > 0 else " "
        after = vis[b] if b < len(vis) else " "
        if before.isalpha() or after.isalpha():
            continue
        q = b
        while q < len(vis) and vis[q].isspace():
            q += 1
        if q < len(vis) and vis[q] == ":":
            last = a
    return last


def _loose_marker(vis: str) -> int | None:
    m = re.search(r"answer", vis, flags=re.IGNORECASE)
    return m.start() if m else None


def _last_boxed(vis: str) -> str | None:
    found = _BOXED.findall(vis)
    return found[-1] if found else None


def _negated(seg: str, pos: int) -> bool:
    """ds4 mc_letter_is_negated: the word right before seg[pos] (same line) is a rejection cue."""
    p = pos
    while p > 0:
        c = seg[p - 1]
        if c == "\n":
            return False
        if c in " \t,;":
            p -= 1
        else:
            break
    wend = p
    while p > 0 and (seg[p - 1].isalpha() or seg[p - 1] == "'"):
        p -= 1
    w = seg[p:wend].lower()
    if not w or len(w) >= 16:
        return False
    if len(w) >= 3 and w.endswith("n't"):
        return True
    if w in _NEG_CUES:
        return True
    if w == "out":
        q = p
        while q > 0 and seg[q - 1] in " \t":
            q -= 1
        rend = q
        while q > 0 and seg[q - 1].isalpha():
            q -= 1
        return seg[q:rend].lower() in ("rule", "rules", "ruled")
    return False


def _letter_in(seg: str, nchoices: int, base_before: str = " ") -> str | None:
    """First acceptable standalone choice letter in `seg` (ds4 forward scan after the marker)."""
    top = chr(ord("A") + nchoices - 1)
    for i, ch in enumerate(seg):
        c = ch.upper()
        if not ("A" <= c <= top) or not ch.isalpha():
            continue
        before = seg[i - 1] if i > 0 else base_before
        after = seg[i + 1] if i + 1 < len(seg) else " "
        if before.isalpha() or after.isalpha():
            continue
        if after == "'":
            continue
        if after in " \t":
            w = i + 1
            while w < len(seg) and seg[w] in " \t":
                w += 1
            if w < len(seg) and seg[w].islower():
                continue
        if _negated(seg, i):
            continue
        return c
    return None


def _choice(vis: str, nchoices: int) -> str:
    if nchoices <= 0:
        return "?"
    top = chr(ord("A") + nchoices - 1)
    strict = _strict_marker(vis)
    if strict is None:
        boxed = _last_boxed(vis)
        if boxed is not None:
            b = boxed.strip().upper().rstrip(".")
            if len(b) == 1 and "A" <= b <= top:
                return b
    marker = strict if strict is not None else _loose_marker(vis)
    if marker is not None:
        got = _letter_in(vis[marker:marker + 96], nchoices, vis[marker - 1] if marker > 0 else " ")
        if got:
            return got
    for i in range(len(vis) - 1, -1, -1):
        c = vis[i].upper()
        if vis[i].isalpha() and "A" <= c <= top:
            before = vis[i - 1] if i > 0 else " "
            after = vis[i + 1] if i + 1 < len(vis) else " "
            if not before.isalpha() and not after.isalpha():
                return c
    return "?"


def _norm_int(s: str) -> str:
    s = s.replace(",", "").lstrip("0")
    return s or "0"


def _first_int(s: str) -> str | None:
    m = _INT.search(s)
    return _norm_int(m.group(0)) if m else None


def _marker_line(vis: str, at: int, width: int = 160) -> str:
    line = vis[at:at + width]
    nl = line.find("\n")
    return line if nl < 0 else line[:nl]


def _int_on_line(line: str) -> str | None:
    boxed = _BOXED.findall(line)          # SUC: "Final Answer: 15% of 240 is \\boxed{36}" grades the boxed value
    if boxed:
        got = _first_int(boxed[-1].split("=")[-1])
        if got is not None:
            return got
    eq = line.rfind("=")
    if eq >= 0:
        got = _first_int(line[eq + 1:])
        if got is not None:
            return got
    return _first_int(line)


def _integer(vis: str) -> str:
    strict = _strict_marker(vis)
    if strict is not None:
        got = _int_on_line(_marker_line(vis, strict))
        if got is not None:
            return got
    boxed = _last_boxed(vis)
    if boxed is not None:
        got = _int_on_line(boxed)
        if got is not None:
            return got
    if strict is None:
        loose = _loose_marker(vis)
        if loose is not None:
            got = _int_on_line(_marker_line(vis, loose))
            if got is not None:
                return got
    found = _INT.findall(vis)
    return _norm_int(found[-1]) if found else "?"


def _normalize_line_spec(s: str) -> str:
    """ds4 normalize_compsec_line_spec: "line 9 and 15" -> "9,15", "20 - 22" -> "20-22"."""
    parts = []
    for m in re.finditer(r"(\d+)(?:\s*-\s*(\d+))?", s):
        parts.append(m.group(1) + (f"-{m.group(2)}" if m.group(2) else ""))
    return ",".join(parts) or "?"


def _lines(vis: str) -> str:
    marker = _strict_marker(vis)
    if marker is None:
        marker = _loose_marker(vis)
    if marker is not None:
        got = _normalize_line_spec(_marker_line(vis, marker))
        if got != "?":
            return got
    return _integer(vis)


def _line_set(spec: str, limit: int = 256) -> set[int] | None:
    out: set[int] = set()
    for m in re.finditer(r"(\d+)(?:-(\d+))?", spec):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        a, b = min(a, b), max(a, b)
        out.update(range(max(a, 0), min(b, limit - 1) + 1))
    return out or None


def line_spec_matches(expected: str, got: str) -> bool:
    """Every line the model names is in the accepted set and at least one is named (ds4 compsec_answer_matches)."""
    e, g = _line_set(expected), _line_set(got)
    return e is not None and g is not None and g <= e


def extract_answer(kind: str, text: str, nchoices: int = 0) -> str:
    """The graded answer of `text` for a case kind ('choice' needs nchoices); '?' when nothing is found."""
    vis = visible_text(text)
    if kind == "choice":
        return _choice(vis, nchoices)
    if kind == "integer":
        return _integer(vis)
    if kind == "lines":
        return _lines(vis)
    raise ValueError(f"no extractor for case kind {kind!r}")


def grade(case: dict, text: str) -> tuple[str | None, bool | None]:
    """(extracted answer, passed) for a case dict (kind, answer, choices); (None, None) for ungraded 'open' cases."""
    kind = case["kind"]
    if kind == "open":
        return None, None
    got = extract_answer(kind, text, len(case.get("choices") or ()))
    want = case["answer"]
    if kind == "choice":
        ok = got == want.strip().upper()
    elif kind == "integer":
        ok = got == _norm_int(want.strip())
    else:
        ok = line_spec_matches(want, got)
    return got, ok


__all__ = ["visible_text", "extract_answer", "grade", "line_spec_matches"]
