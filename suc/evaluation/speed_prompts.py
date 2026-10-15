"""Shared prompts of the batch-1 speed protocols, so every runtime (SUC C++ / Python, ds4, LM Studio / llama-server)
is timed on the same input:

  SHORT_PROMPT   the frontier short prompt (scripts/frontier_measure.py generate): raw text, no chat template,
                 256 greedy tokens;
  long prompt    exactly LONG_PROMPT_TOKENS (2048) token ids: the first 2048 ids of the raw tokenization (no special
                 tokens) of experiments/perf/promessi_sposi_head.txt (the ds4-bench / `suc perf` prompt) with the
                 model's own tokenizer, then 256 greedy tokens.

Runtimes that take token ids (suc1-generate --tokens-file, ds4_score serve) get long_prompt_ids(tok) directly; text
servers get long_prompt_text(tok) = the decoded ids (re-tokenization is checked against the server's prompt token
count by the caller).
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHORT_PROMPT = "The history of the printing press begins in the fifteenth century,"
GEN_TOKENS = 256
LONG_PROMPT_FILE = REPO / "experiments" / "perf" / "promessi_sposi_head.txt"
LONG_PROMPT_TOKENS = 2048


def long_prompt_ids(tokenizer, n: int = LONG_PROMPT_TOKENS) -> list[int]:
    ids = list(tokenizer.encode(LONG_PROMPT_FILE.read_text(), add_special_tokens=False))
    if len(ids) < n:
        raise ValueError(f"{LONG_PROMPT_FILE} has only {len(ids)} tokens, {n} requested")
    return ids[:n]


def long_prompt_text(tokenizer, n: int = LONG_PROMPT_TOKENS) -> str:
    return tokenizer.decode(long_prompt_ids(tokenizer, n))


def write_tokens_file(ids: list[int], path: str | Path) -> Path:
    """Comma-separated ids, the format of suc1-generate --tokens-file."""
    p = Path(path)
    p.write_text(",".join(map(str, ids)))
    return p


__all__ = ["SHORT_PROMPT", "GEN_TOKENS", "LONG_PROMPT_FILE", "LONG_PROMPT_TOKENS", "long_prompt_ids",
           "long_prompt_text", "write_tokens_file"]
