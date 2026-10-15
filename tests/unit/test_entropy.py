"""Tests for the storage-only entropy coder (static interleaved rANS) and the entropy estimators (spec §15)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from suc.entropy import measures, rans


def _streams(rng):
    return [
        rng.integers(0, 256, 5000).astype(np.uint8),                                     # near uniform, K=256
        np.minimum(rng.geometric(0.3, 3001) - 1, 15).astype(np.uint8),                   # skewed, K=16
        np.full(17, 3, np.uint8),                                                        # single symbol
        np.zeros(0, np.uint8),                                                           # empty stream
        rng.integers(0, 2, 1).astype(np.uint8),                                          # one symbol
        (rng.standard_normal(20000) * 20 + 128).clip(0, 255).astype(np.uint8),           # bell shaped
    ]


def _tables(streams, a=256, pb=rans.PROB_BITS):
    out = np.zeros((len(streams), a), np.uint32)
    for i, s in enumerate(streams):
        c = np.bincount(s.astype(np.int64), minlength=a) if s.size else np.ones(a, np.int64)
        out[i] = rans.quantize_freqs(c, pb)
    return out


def test_quantize_freqs_sum_and_support():
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = rng.integers(0, 1000, 256) * (rng.random(256) < 0.7)
        c[0] = max(c[0], 1)
        f = rans.quantize_freqs(c, 12)
        assert int(f.sum()) == 1 << 12
        assert np.all((f > 0) == (c > 0))
    assert rans.quantize_freqs(np.array([0, 5, 0]), 4).tolist() == [0, 16, 0]


@pytest.mark.parametrize("steps", [1, 7, 64, 30000])
@pytest.mark.parametrize("pb", [8, 14, 16])
def test_rans_round_trip_bit_exact(steps, pb):
    rng = np.random.default_rng(steps + pb)
    streams = _streams(rng)
    enc = rans.encode(streams, _tables(streams, pb=pb), steps=steps, prob_bits=pb)
    dec = rans.decode(enc)
    assert len(dec) == len(streams)
    for s, d in zip(streams, dec):
        assert d.dtype == np.uint8 and np.array_equal(s, d)
    assert int(enc.stream_words.sum()) == enc.words.size


def test_rans_size_close_to_entropy():
    rng = np.random.default_rng(1)
    s = (rng.standard_normal(400000) * 12 + 128).clip(0, 255).astype(np.uint8)
    enc = rans.encode([s], _tables([s]), steps=4096)
    h = measures.order0(s, 256)
    payload_bits = 16 * enc.words.size
    assert payload_bits / s.size < h + 0.01            # table quantization + flush overhead < 0.01 bit/symbol
    assert payload_bits / s.size > h - 1e-6 - 32 * enc.lanes[0] / s.size


def test_rans_alphabet_u16_and_bad_inputs():
    rng = np.random.default_rng(2)
    s = rng.integers(0, 1000, 3000).astype(np.uint16)
    f = rans.quantize_freqs(np.bincount(s, minlength=1024), 14)[None]
    assert np.array_equal(rans.decode(rans.encode([s], f, steps=50))[0], s)
    with pytest.raises(ValueError):
        rans.encode([np.array([5], np.uint8)], rans.quantize_freqs(np.array([1, 1, 0, 0, 0, 0]))[None])


def test_rans_detects_corruption():
    rng = np.random.default_rng(3)
    s = rng.integers(0, 256, 4000).astype(np.uint8)
    enc = rans.encode([s], _tables([s]), steps=16)
    enc.words = enc.words[:-1]
    with pytest.raises(ValueError):
        rans.decode(enc)


def test_entropy_estimators():
    assert measures.entropy_bits(np.array([1, 1, 1, 1])) == pytest.approx(2.0)
    rng = np.random.default_rng(4)
    x = rng.integers(0, 16, 200000)
    assert measures.order0(x, 16) == pytest.approx(4.0, abs=1e-3)
    y = (x + (rng.random(x.size) < 0.1)) % 16           # y depends on x
    assert measures.conditional_plugin(x, y, 16) < 0.5
    fold = np.arange(x.size) % 2 == 0
    held = measures.two_fold(x, y, fold, 16)
    assert measures.conditional_plugin(x, y, 16) <= held < 0.6
    # independent symbols: held-out order-1 >= true entropy (no spurious gain), plug-in is biased low
    z = rng.integers(0, 256, 20000)
    c = rng.integers(0, 256, 20000)
    assert measures.conditional_plugin(c, z, 256) < 7.5
    assert measures.two_fold(c, z, np.arange(z.size) % 2 == 0, 256) > 7.9
    assert measures.log2_binomial(10, 3) == pytest.approx(math.log2(120))
    r, bits = measures.rice_bits(np.array([1, 2, 3, 4]))
    assert bits == min(int(((np.array([0, 1, 2, 3]) >> k).sum()) + 4 * (1 + k)) for k in range(8)) and r >= 0


@pytest.mark.parametrize("r", [0, 1, 5, 9])
def test_rice_round_trip_bit_exact(r):
    rng = np.random.default_rng(r)
    pos = np.sort(rng.choice(3_000_000, 15000, replace=False))
    gaps = np.diff(np.concatenate([[-1], pos]))
    u, lo = measures.rice_encode(gaps - 1, r)
    assert np.array_equal(measures.rice_decode(u, lo, r, gaps.size), gaps - 1)
    _, best_bits = measures.rice_bits(gaps)
    assert 8 * (u.size + lo.size) >= best_bits                 # never below the optimal parameter's exact size
    assert 8 * (u.size + lo.size) - 16 <= measures.rice_bits_at(gaps, r)
    empty = measures.rice_encode(np.zeros(0, np.int64), r)
    assert measures.rice_decode(*empty, r, 0).size == 0


def test_freqs_from_counts_exact():
    rng = np.random.default_rng(5)
    c = rng.integers(0, 50, (300, 256)) * (rng.random((300, 256)) < 0.3)
    c[0] = 0                                                    # empty context -> uniform
    f = rans.freqs_from_counts(c, 14)
    assert f.shape == c.shape and np.all(f.sum(axis=1) == 1 << 14) and f.min() >= 1
    assert np.all(f[0] == 64)
    assert np.array_equal(rans.freqs_from_counts(c[5], 14), f[5])


@pytest.mark.parametrize("steps", [1, 33, 4096])
def test_rans_context_tables_round_trip(steps):
    """Order-1 context coding: symbol of stream B uses the table selected by stream A's symbol."""
    rng = np.random.default_rng(steps)
    a = rng.integers(0, 256, 9000).astype(np.uint8)
    b = ((a.astype(np.int64) + rng.integers(0, 4, a.size)) % 256).astype(np.uint8)
    counts = np.zeros((256, 256), np.int64)
    np.add.at(counts, (a[:4500], b[:4500]), 1)
    tables = rans.freqs_from_counts(counts)
    short = rng.integers(0, 256, 10).astype(np.uint8)
    streams, tids = [b, short], [a.astype(np.int64), np.full(10, 7, np.int64)]
    enc = rans.encode(streams, tables, steps=steps, table_ids=tids)
    dec = rans.decode(enc, table_ids=tids)
    assert all(np.array_equal(x, y) for x, y in zip(streams, dec))
    assert 16 * enc.words.size < 0.5 * 8 * b.size              # strongly predictable given the context
    # binary-search path also exact for per-stream tables
    s = [rng.integers(0, 200, 3000).astype(np.uint8)]
    f = rans.quantize_freqs(np.bincount(s[0], minlength=256))[None]
    enc2 = rans.encode(s, f, steps=steps)
    assert np.array_equal(rans.decode(enc2, table_ids=[np.zeros(3000, np.int64)])[0], s[0])
