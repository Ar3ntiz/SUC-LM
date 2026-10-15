"""Held-out perplexity over non-overlapping windows."""
from __future__ import annotations

import math

import mlx.core as mx


def perplexity(model, tokens: mx.array, seq_len: int, max_tokens: int | None = None,
               batch_size: int = 1) -> dict:
    """Perplexity over non-overlapping windows of seq_len+1 tokens.

    `tokens` (1-D) is optionally truncated to `max_tokens`, then cut into windows of seq_len+1 tokens;
    the incomplete tail window is dropped. Each window feeds tokens[0:seq_len] and scores the
    predictions of tokens[1:seq_len+1], so every window scores exactly seq_len tokens with fresh
    context (no cache, no overlap). `model(x)` must map int (B, T) -> logits (B, T, V).
    NLL is accumulated in float32 per window and summed in Python float64.
    """
    tokens = tokens.reshape(-1)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    n_win = tokens.size // (seq_len + 1)
    if n_win == 0:
        raise ValueError(f"need at least {seq_len + 1} tokens, got {tokens.size}")
    windows = tokens[: n_win * (seq_len + 1)].reshape(n_win, seq_len + 1)
    total = 0.0
    for s in range(0, n_win, batch_size):
        w = windows[s:s + batch_size]
        logits = model(w[:, :-1]).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        nll = -mx.take_along_axis(logp, w[:, 1:, None], axis=-1).sum()
        total += nll.item()
    n = n_win * seq_len
    nll_mean = total / n
    return {"ppl": math.exp(nll_mean), "nll_mean": nll_mean, "tokens_scored": n,
            "windows": n_win, "seq_len": seq_len}
