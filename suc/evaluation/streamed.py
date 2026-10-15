"""Layer-streamed quality evaluation (spec D8): WikiText-2 PPL and KL vs stored reference logits with at most one
transformer block resident (suc.models.streaming.stream_eval), for models larger than RAM.

One block-sequential sweep serves both measurements: the PPL windows (batch 1, as suc.evaluation.perplexity) and
the KL sequences (micro-batches of `chunk` rows, as suc.evaluation.reconstruct.reference_logits / _kl) go through
every block together; the LM head is applied one micro-batch at a time. Each micro-batch sees exactly the
operations of the whole-model forward with the same batch composition, so PPL, the reference logits and the KL
statistics equal those of evaluate_quality / reference_logits (tests/unit/test_streamed_eval.py checks this).

Reference logits are stored IN FULL (bf16, N x T x V, the same safetensors tensor "logits" as before) but written
and read in `chunk`-row slices (RowStreamWriter / SafetensorsStore.load_rows), so neither stage holds them all.
Approximation-free by design: a top-k / fp16-log-softmax store would shrink the file but change the KL numbers.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from suc.analysis.sensitivity import logit_divergence
from suc.evaluation.reconstruct import aggregate_kl
from suc.io.safetensors_store import RowStreamWriter, SafetensorsStore, read_header
from suc.models.streaming import BlockHook, stream_eval

REF_KEY = "logits"


def ppl_windows(tokens: mx.array, seq_len: int, max_tokens: int | None) -> mx.array:
    """(n_win, seq_len + 1) non-overlapping windows exactly as suc.evaluation.perplexity.perplexity."""
    tokens = tokens.reshape(-1)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    n_win = tokens.size // (seq_len + 1)
    if n_win == 0:
        raise ValueError(f"need at least {seq_len + 1} tokens, got {tokens.size}")
    return tokens[: n_win * (seq_len + 1)].reshape(n_win, seq_len + 1)


class RefLogits:
    """Reader of the stored reference logits, one row slice at a time."""

    def __init__(self, path: str | Path) -> None:
        self.store = SafetensorsStore(read_header(Path(path)))
        self.shape = self.store.info(REF_KEY).shape

    def rows(self, start: int, stop: int) -> mx.array:
        return self.store.load_rows(REF_KEY, start, stop)




def streamed_quality(model: nn.Module, *, ppl_tokens: mx.array, seq_len: int, max_tokens: int | None,
                     kl_tokens: mx.array | None = None, chunk: int = 4, ref_logits: RefLogits | None = None,
                     ref_out: str | Path | None = None, prepare_block: BlockHook | None = None,
                     prepare_head: Callable[[nn.Module], None] | None = None, release: bool = True,
                     spill_dir: str | Path | None = None) -> dict:
    """{"perplexity": as perplexity(...), "kl": as evaluate_quality's KL vs ref_logits (or None), "streaming": {...}}.

    With kl_tokens and ref_out, the bf16 logits of the KL sequences are written to ref_out (the reference stage);
    with kl_tokens and ref_logits, KL / top-1 / ΔCE vs those logits are computed (the evaluate stage). The model is
    run block by block (lazily loaded model: one block of weights resident; prepare_block installs e.g. decoded
    weights); its blocks are released unless release=False."""
    windows = ppl_windows(ppl_tokens, seq_len, max_tokens)
    n_win = windows.shape[0]
    batches = [windows[j:j + 1, :-1] for j in range(n_win)]
    kl_rows = []
    if kl_tokens is not None:
        if ref_logits is None and ref_out is None:
            raise ValueError("kl_tokens given without ref_logits or ref_out")
        if ref_logits is not None and ref_logits.shape[:2] != tuple(kl_tokens.shape):
            raise ValueError("ref_logits rows do not match kl_tokens")
        kl_rows = [(s, min(s + chunk, kl_tokens.shape[0])) for s in range(0, kl_tokens.shape[0], chunk)]
        batches += [kl_tokens[s:e] for s, e in kl_rows]
    writer = None
    nll_total = 0.0
    kl_parts: list[tuple[dict, int]] = []

    def consume(j: int, logits: mx.array) -> None:
        nonlocal nll_total, writer
        if j < n_win:                                    # PPL window: the arithmetic of perplexity()
            w = windows[j:j + 1]
            lg = logits.astype(mx.float32)
            logp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            nll_total += (-mx.take_along_axis(logp, w[:, 1:, None], axis=-1).sum()).item()
            return
        s, e = kl_rows[j - n_win]
        tok = kl_tokens[s:e]
        if ref_out is not None:                          # reference: reference_logits() rows, written in place
            if writer is None:
                writer = RowStreamWriter(ref_out, REF_KEY, (*kl_tokens.shape, logits.shape[-1]), mx.bfloat16)
            lg = logits.astype(mx.bfloat16)
            mx.eval(lg)
            writer.write(lg)
        if ref_logits is not None:                       # evaluate: one sub-chunk of _kl()
            r = ref_logits.rows(s, e)
            test = logits.astype(r.dtype)
            d = logit_divergence(r[:, :-1], test[:, :-1], tok[:, 1:], return_tokens=True)
            kl_parts.append((d, tok.shape[0] * (tok.shape[1] - 1)))

    try:
        info = stream_eval(model, batches, consume, prepare_block=prepare_block, prepare_head=prepare_head,
                           release=release, spill_dir=spill_dir)
    finally:
        if writer is not None:
            writer._f.close()
    if writer is not None:
        writer.close()
    n = n_win * seq_len
    nll_mean = nll_total / n
    ppl = {"ppl": math.exp(nll_mean), "nll_mean": nll_mean, "tokens_scored": n, "windows": n_win,
           "seq_len": seq_len}
    kl = None
    if ref_logits is not None and kl_tokens is not None:
        kl = aggregate_kl(kl_parts)          # exact means, global kl_p95 (suc.evaluation.reconstruct)
    return {"perplexity": ppl, "kl": kl,
            "streaming": {"peak_memory_bytes": info["peak_memory_bytes"], "seconds": info["seconds"],
                          "batches": len(batches)}}
