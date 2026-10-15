"""Test suites of `suc eval`: loading, case selection, prompt rendering (experiments/eval/*.json).

Suite file: {"suite", "version", "description", "attribution", "licenses"?, "system": str | null, "cases": [...]};
case: {"id", "source", "domain"?, "title"?, "kind": choice | integer | lines | open, "question", "choices"? (choice,
2..10), "answer"? (every kind but open), "prompt_style": ds4 | raw, "think": bool, "max_tokens": int}.

prompt_style ds4 renders the question the way ds4-eval does (ds4_eval.c build_question_prompt; ds4, MIT, The ds4.c
authors): the choices as "A. ..." lines and a final instruction to end with exactly one "Answer: <...>" line; raw sends
the question unchanged (the QA prompts of experiments/qa/prompts.json). The prompt token ids are rendered once, in
Python, with the checkpoint's chat template, and fed to every system as ids, so all runtimes see the same tokens.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

KINDS = ("choice", "integer", "lines", "open")
_FINAL = "At the end, write exactly one final line in this format and do not write anything after it:\n"
_TAIL = {"choice": "\nSolve the question. " + _FINAL + "Answer: <letter>",
         "lines": "\n" + _FINAL + "Answer: <line number or comma-separated line numbers>",
         "integer": "\nSolve the problem. " + _FINAL + "Answer: <integer>"}


def _check_case(c: dict, where: str) -> None:
    for k in ("id", "source", "kind", "question", "think", "max_tokens"):
        if k not in c:
            raise ValueError(f"{where}: missing key {k!r}")
    unknown = sorted(set(c) - {"id", "source", "domain", "title", "kind", "question", "choices", "answer",
                               "prompt_style", "think", "max_tokens"})
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}")
    if c["kind"] not in KINDS:
        raise ValueError(f"{where}: kind {c['kind']!r} not in {KINDS}")
    if (c["kind"] == "open") != ("answer" not in c):
        raise ValueError(f"{where}: 'answer' is required for graded kinds and invalid for 'open'")
    if (c["kind"] == "choice") != ("choices" in c):
        raise ValueError(f"{where}: 'choices' is required for kind choice and invalid otherwise")
    if c["kind"] == "choice":
        n = len(c["choices"])
        if not 2 <= n <= 10 or not ("A" <= c["answer"] <= chr(ord("A") + n - 1)) or len(c["answer"]) != 1:
            raise ValueError(f"{where}: 2..10 choices and a letter answer within them required")
    if c.get("prompt_style", "ds4") not in ("ds4", "raw"):
        raise ValueError(f"{where}: prompt_style must be ds4 or raw")
    if not isinstance(c["max_tokens"], int) or c["max_tokens"] < 1:
        raise ValueError(f"{where}: max_tokens must be a positive integer")


def load_suite(path: str | Path) -> dict:
    """The suite dict plus 'sha256' (of the file bytes: the dataset_hash of the run); cases validated."""
    raw = Path(path).read_bytes()
    suite = json.loads(raw)
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: 'cases' must be a non-empty list")
    for i, c in enumerate(cases):
        _check_case(c, f"{path} cases[{i}]")
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate case ids")
    return {**suite, "sha256": hashlib.sha256(raw).hexdigest()}


def select_cases(cases: list[dict], ids: tuple[str, ...] | None = None, limit: int | None = None) -> list[dict]:
    """Cases with the given ids (in the given order), else the suite order; then the first `limit`."""
    if ids is not None:
        by_id = {c["id"]: c for c in cases}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(f"unknown case id(s) {missing}")
        cases = [by_id[i] for i in ids]
    return cases[:limit] if limit is not None else list(cases)


def user_prompt(case: dict) -> str:
    """The user-turn text of a case (ds4-eval build_question_prompt for prompt_style ds4)."""
    if case.get("prompt_style", "ds4") == "raw" or case["kind"] == "open":
        return case["question"]
    text = case["question"] + "\n"
    if case["kind"] == "choice":
        text += "\nChoices:\n" + "".join(f"{chr(ord('A') + i)}. {ch}\n" for i, ch in enumerate(case["choices"]))
    return text + _TAIL[case["kind"]]


def render_prompt_ids(tokenizer, case: dict, system: str | None, think: bool) -> list[int]:
    """Chat-template token ids (generation prompt appended; enable_thinking = think)."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user_prompt(case)}]
    text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=think)
    return list(tokenizer.encode(text, add_special_tokens=False))


__all__ = ["KINDS", "load_suite", "select_cases", "user_prompt", "render_prompt_ids"]
