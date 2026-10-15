"""Load / prefill / decode timing on Apple Silicon (spec §25, §35)."""
from __future__ import annotations

import statistics
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def _vocab_size(model) -> int:
    args = getattr(model, "args", None)
    if args is not None and getattr(args, "vocab_size", None):
        return int(args.vocab_size)
    return int(model.vocab_size)


def _one_run(model, prompt: mx.array, decode_tokens: int) -> dict:
    cache = make_prompt_cache(model)
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.synchronize()
    t0 = time.perf_counter()
    y = mx.argmax(model(prompt, cache=cache)[:, -1, :], axis=-1)
    mx.eval(y)
    t1 = time.perf_counter()
    out = [y]
    for _ in range(decode_tokens):
        y = mx.argmax(model(y[:, None], cache=cache)[:, -1, :], axis=-1)
        mx.eval(y)
        out.append(y)
    t2 = time.perf_counter()
    b, p = prompt.shape
    ttft, dec = t1 - t0, t2 - t1
    toks = mx.stack(out[:decode_tokens], axis=1).tolist()  # first decode_tokens generated tokens
    return {"ttft_s": ttft, "prefill_tok_s": b * p / ttft, "decode_s": dec,
            "decode_tok_s": b * decode_tokens / dec, "peak_memory_bytes": mx.get_peak_memory(),
            "tokens": toks}


def measure_generation(model, prompt_len: int, decode_tokens: int, batch_size: int = 1,
                       warmup: int = 1, repeats: int = 3, seed: int = 0) -> dict:
    """Greedy generation timing on a synthetic prompt (random ids, deterministic from `seed`).

    Per repeat: a fresh KV cache (mlx_lm make_prompt_cache); ttft_s = prefill of the whole prompt +
    argmax of the first token (evaluated); then `decode_tokens` further single-token steps, each
    evaluated synchronously (batch=1 latency, no async pipelining). prefill_tok_s =
    batch*prompt_len/ttft_s, decode_tok_s = batch*decode_tokens/decode_s. peak_memory_bytes is
    mx.get_peak_memory() since a reset at the start of the repeat (includes resident weights).
    Returned scalars are medians over repeats (peak memory: max). `tokens` in each repeat holds the
    first `decode_tokens` generated ids per sequence, for determinism checks.
    """
    rng = np.random.default_rng(seed)
    prompt = mx.array(rng.integers(0, _vocab_size(model), size=(batch_size, prompt_len)).astype(np.int32))
    for _ in range(warmup):
        _one_run(model, prompt, min(decode_tokens, 2))
    reps = [_one_run(model, prompt, decode_tokens) for _ in range(repeats)]
    med = lambda k: statistics.median(r[k] for r in reps)  # noqa: E731
    return {"prompt_len": prompt_len, "decode_tokens": decode_tokens, "batch_size": batch_size,
            "ttft_s": med("ttft_s"), "prefill_tok_s": med("prefill_tok_s"),
            "decode_tok_s": med("decode_tok_s"),
            "peak_memory_bytes": max(r["peak_memory_bytes"] for r in reps), "repeats": reps}


def timed_load(model_id_or_path: str, revision: str | None = None):
    """mlx_lm.load (non-lazy: weights evaluated) timed with perf_counter -> (model, tokenizer, seconds)."""
    t0 = time.perf_counter()
    model, tokenizer = load(model_id_or_path, revision=revision)
    mx.synchronize()
    return model, tokenizer, time.perf_counter() - t0
