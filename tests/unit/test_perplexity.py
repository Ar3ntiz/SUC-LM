import math

import mlx.core as mx
import numpy as np
import pytest

from suc.evaluation.perplexity import perplexity
from tiny_models import tiny_qwen3


class FakeModel:
    """Logit `a` on token (x+1) % V at every position, 0 elsewhere."""

    def __init__(self, vocab: int, a: float):
        self.vocab, self.a = vocab, a
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, x: mx.array) -> mx.array:
        self.calls.append(tuple(x.shape))
        return mx.array(self.a * np.eye(self.vocab, dtype=np.float32)[(np.array(x) + 1) % self.vocab])


def expected_nll(tokens: np.ndarray, vocab: int, a: float, seq_len: int, max_tokens=None) -> tuple[float, int]:
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    n_win = len(tokens) // (seq_len + 1)
    hit, miss = -(a - math.log(math.exp(a) + vocab - 1)), math.log(math.exp(a) + vocab - 1)
    total, n = 0.0, 0
    for w in range(n_win):
        win = tokens[w * (seq_len + 1):(w + 1) * (seq_len + 1)]
        for i in range(seq_len):
            total += hit if win[i + 1] == (win[i] + 1) % vocab else miss
            n += 1
    return total / n, n


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_perplexity_exact_on_fake_model(batch_size):
    vocab, a, seq_len = 11, 2.0, 5
    rng = np.random.default_rng(0)
    # mix of predictable (+1) and random transitions; 40 tokens -> 6 windows of 6, tail of 4 dropped
    toks = [0]
    for _ in range(39):
        toks.append((toks[-1] + 1) % vocab if rng.random() < 0.6 else int(rng.integers(vocab)))
    toks = np.array(toks, dtype=np.int32)
    model = FakeModel(vocab, a)
    r = perplexity(model, mx.array(toks), seq_len=seq_len, batch_size=batch_size)
    nll, n = expected_nll(toks, vocab, a, seq_len)
    assert r["windows"] == 6 and r["tokens_scored"] == n == 30 and r["seq_len"] == seq_len
    assert r["nll_mean"] == pytest.approx(nll, rel=1e-5)
    assert r["ppl"] == pytest.approx(math.exp(nll), rel=1e-5)
    assert all(shape[1] == seq_len for shape in model.calls)


def test_perplexity_max_tokens_truncates():
    vocab, a, seq_len = 7, 1.0, 4
    toks = np.arange(50, dtype=np.int32) % vocab
    r = perplexity(FakeModel(vocab, a), mx.array(toks), seq_len=seq_len, max_tokens=23)
    nll, n = expected_nll(toks, vocab, a, seq_len, max_tokens=23)
    assert r["windows"] == 4 and r["tokens_scored"] == n == 16
    assert r["nll_mean"] == pytest.approx(nll, rel=1e-5)


def test_perplexity_too_few_tokens_raises():
    with pytest.raises(ValueError):
        perplexity(FakeModel(5, 1.0), mx.array(np.arange(4, dtype=np.int32)), seq_len=4)


def test_perplexity_tiny_model_finite_and_deterministic():
    model = tiny_qwen3(seed=1)
    toks = mx.array(np.random.default_rng(0).integers(0, 128, size=200).astype(np.int32))
    r1 = perplexity(model, toks, seq_len=32, batch_size=2)
    r2 = perplexity(model, toks, seq_len=32, batch_size=1)
    assert math.isfinite(r1["ppl"]) and r1["ppl"] > 1.0
    assert r1["windows"] == 6
    assert r1["nll_mean"] == pytest.approx(r2["nll_mean"], rel=1e-4)
    assert perplexity(model, toks, seq_len=32, batch_size=2)["nll_mean"] == r1["nll_mean"]
