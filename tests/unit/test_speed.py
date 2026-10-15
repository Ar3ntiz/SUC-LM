import mlx.core as mx

from suc.evaluation.speed import measure_generation
from tiny_models import tiny_qwen3


KEYS = {"prompt_len", "decode_tokens", "batch_size", "ttft_s", "prefill_tok_s", "decode_tok_s",
        "peak_memory_bytes", "repeats"}


def test_measure_generation_keys_and_positive_rates():
    r = measure_generation(tiny_qwen3(), prompt_len=16, decode_tokens=4, warmup=0, repeats=1)
    assert KEYS <= set(r)
    assert (r["prompt_len"], r["decode_tokens"], r["batch_size"]) == (16, 4, 1)
    assert r["ttft_s"] > 0 and r["prefill_tok_s"] > 0 and r["decode_tok_s"] > 0
    assert r["peak_memory_bytes"] > 0
    assert len(r["repeats"]) == 1
    rep = r["repeats"][0]
    assert {"ttft_s", "prefill_tok_s", "decode_tok_s", "decode_s", "peak_memory_bytes"} <= set(rep)


def test_measure_generation_batch_and_greedy_determinism():
    model = tiny_qwen3()
    r1 = measure_generation(model, prompt_len=8, decode_tokens=3, batch_size=2, warmup=1, repeats=2, seed=3)
    r2 = measure_generation(model, prompt_len=8, decode_tokens=3, batch_size=2, warmup=0, repeats=1, seed=3)
    assert r1["batch_size"] == 2 and len(r1["repeats"]) == 2
    # greedy decode from the same seeded prompt produces identical tokens
    assert r1["repeats"][0]["tokens"] == r1["repeats"][1]["tokens"] == r2["repeats"][0]["tokens"]
    assert len(r1["repeats"][0]["tokens"]) == 2 and len(r1["repeats"][0]["tokens"][0]) == 3
