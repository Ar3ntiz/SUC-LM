"""Block-streaming pass on the real Qwen3-0.6B dev model: equivalence and peak memory (spec D8)."""
import gc
import time
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from suc.models.streaming import load_lazy, model_blocks, stream_blocks

MODEL_DIR = Path.home() / (".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
                           "c1899de289a04d12100db370d81485cdf75e47ca")
B, T = 8, 256

pytestmark = [pytest.mark.model,
              pytest.mark.skipif(not MODEL_DIR.exists(), reason="Qwen3-0.6B not downloaded")]


def _fresh() -> None:
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def test_streamed_logits_match_full_model_with_lower_peak_memory():
    t_start = time.perf_counter()
    model, _ = load_lazy(MODEL_DIR)
    tokens = mx.random.randint(0, model.args.vocab_size, (B, T), key=mx.random.key(12345))
    mx.eval(tokens)

    _fresh()
    out = stream_blocks(model, tokens, micro_batch=4, release=True, return_logits=True)
    stream_peak, block_s = out["peak_memory_bytes"], sum(out["block_seconds"])
    streamed = out["logits"].astype(mx.float32)
    mx.eval(streamed)
    assert all(sum(v.nbytes for _, v in tree_flatten(b.parameters())) == 0 for b in model_blocks(model))
    del model, out
    _fresh()

    import mlx_lm
    full_model, _ = mlx_lm.load(str(MODEL_DIR))  # non-lazy: all weights resident
    full = full_model(tokens).astype(mx.float32)
    mx.eval(full)
    full_peak = mx.get_peak_memory()

    max_abs = mx.abs(streamed - full).max().item()
    lp = full - mx.logsumexp(full, axis=-1, keepdims=True)
    lq = streamed - mx.logsumexp(streamed, axis=-1, keepdims=True)
    kl = (mx.exp(lp) * (lp - lq)).sum(-1).mean().item()
    print(f"\nmax_abs_diff={max_abs:.4g} KL(full||stream)={kl:.3g} "
          f"stream_peak={stream_peak / 2**20:.0f}MiB full_peak={full_peak / 2**20:.0f}MiB "
          f"block_s={block_s:.2f} total_s={time.perf_counter() - t_start:.1f}")
    assert kl < 1e-3
    assert stream_peak < full_peak
