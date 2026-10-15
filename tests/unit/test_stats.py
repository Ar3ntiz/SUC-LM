"""Tests for per-tensor scalar statistics and per-block statistics (spec §8 T1.1, T1.2)."""
from __future__ import annotations

import json
import math

import mlx.core as mx
import numpy as np
import pytest

from suc.analysis.stats import block_stats, row_rms_normalize, scalar_stats, to_blocks

QKEYS = {"q0.001", "q0.01", "q0.05", "q0.25", "q0.5", "q0.75", "q0.95", "q0.99", "q0.999"}


def _normal(shape, seed=0):
    return mx.random.normal(shape, key=mx.random.key(seed))


def test_scalar_stats_constant_tensor():
    s = scalar_stats(mx.full((4, 8), 2.5))
    assert s["numel"] == 32
    assert s["min"] == s["max"] == s["mean"] == 2.5
    assert s["std"] == 0.0 and s["abs_mean"] == 2.5 and s["abs_max"] == 2.5
    assert set(s["quantiles"]) == QKEYS and all(v == 2.5 for v in s["quantiles"].values())
    assert s["outlier_rate"] == {"3.0": 0.0, "6.0": 0.0}
    assert sum(s["histogram"]["counts"]) == 32
    json.dumps(s)  # plain Python types only


def test_scalar_stats_hand_values():
    s = scalar_stats(mx.array([[-2.0, 0.0], [1.0, 1.0]]), hist_bins=4)
    assert s["mean"] == 0.0 and s["min"] == -2.0 and s["max"] == 1.0
    assert s["std"] == pytest.approx(math.sqrt(1.5))
    assert s["abs_mean"] == 1.0 and s["abs_max"] == 2.0
    # E[x^4] = (16+0+1+1)/4 = 4.5 ; sigma^4 = 2.25 -> excess kurtosis = 2 - 3 = -1
    assert s["kurtosis"] == pytest.approx(-1.0, abs=1e-6)
    assert s["histogram"]["edges"] == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert s["histogram"]["counts"] == [1, 0, 1, 2]


def test_scalar_stats_normal_and_laplace_kurtosis():
    sn = scalar_stats(_normal((512, 512)))
    assert sn["kurtosis"] == pytest.approx(0.0, abs=0.05)
    assert sn["mean"] == pytest.approx(0.0, abs=0.01) and sn["std"] == pytest.approx(1.0, abs=0.01)
    assert sn["quantiles"]["q0.5"] == pytest.approx(0.0, abs=0.01)
    assert sn["quantiles"]["q0.95"] == pytest.approx(1.6449, abs=0.02)
    assert sn["outlier_rate"]["3.0"] == pytest.approx(0.0027, abs=0.0005)
    assert sn["outlier_rate"]["6.0"] == 0.0
    lap = np.random.default_rng(0).laplace(size=(512, 512)).astype(np.float32)
    assert scalar_stats(mx.array(lap))["kurtosis"] == pytest.approx(3.0, abs=0.3)


def test_scalar_stats_injected_outlier():
    w = _normal((256, 256))
    base = scalar_stats(w)
    w2 = mx.where(mx.arange(256 * 256).reshape(256, 256) == 777, 50.0, w)
    s = scalar_stats(w2)
    assert s["abs_max"] == 50.0 and base["abs_max"] < 10
    assert s["outlier_rate"]["6.0"] == pytest.approx(1 / 65536)
    assert s["kurtosis"] > base["kurtosis"] + 10


def test_scalar_stats_bf16_input():
    w = _normal((64, 64)).astype(mx.bfloat16)
    s = scalar_stats(w)
    ref = np.array(w.astype(mx.float32))
    assert s["mean"] == pytest.approx(float(ref.mean()), abs=1e-5)
    assert s["abs_max"] == pytest.approx(float(np.abs(ref).max()))


def test_to_blocks_layout():
    w = mx.arange(2 * 16).reshape(2, 16)
    b = to_blocks(w, 8)
    assert b.shape == (4, 8)
    assert b.tolist() == [list(range(i * 8, i * 8 + 8)) for i in range(4)]
    b4 = to_blocks(mx.arange(12).reshape(3, 4), 4)
    assert b4[1].tolist() == [4, 5, 6, 7]


def test_to_blocks_errors():
    with pytest.raises(ValueError):
        to_blocks(mx.zeros((4, 10)), 8)
    with pytest.raises(ValueError):
        to_blocks(mx.zeros((64,)), 8)
    with pytest.raises(ValueError):
        to_blocks(mx.zeros((2, 4, 8)), 8)


def test_row_rms_normalize():
    w = _normal((32, 64)) * mx.arange(1, 33).reshape(32, 1)
    wn, rms = row_rms_normalize(w.astype(mx.bfloat16))
    assert wn.dtype == mx.float32 and rms.dtype == mx.float32 and rms.shape == (32,)
    np.testing.assert_allclose(np.sqrt(np.mean(np.array(wn) ** 2, axis=1)), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.array(wn * rms[:, None]), np.array(w.astype(mx.bfloat16).astype(mx.float32)),
                               rtol=1e-5, atol=1e-5)


def test_block_stats_hand_values():
    w = mx.array([[1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 4.0]])  # G=4 -> two blocks
    s = block_stats(w, 4, normalize_rows=False)
    assert s["num_blocks"] == 2 and s["num_blocks_used"] == 2 and s["block_size"] == 4
    # block0: l1=4, l2=2, mean=0, var=1, max_abs=1, zeros=0 ; block1: l1=4, l2=4, mean=1, var=3, max_abs=4, zeros=.75
    assert s["l1"]["mean"] == 4.0 and s["l2"]["mean"] == 3.0
    assert s["mean"]["mean"] == 0.5 and s["variance"]["mean"] == 2.0
    assert s["max_abs"]["mean"] == 2.5
    assert s["sparsity_exact"]["mean"] == pytest.approx(0.375)
    assert s["sparsity_1e-3"]["mean"] == pytest.approx(0.375)
    for q in ("l1", "l2", "variance", "max_abs", "mean", "sparsity_exact", "sparsity_1e-3"):
        assert set(s[q]) == {"mean", "std", "p05", "p50", "p95", "p99"}
    json.dumps(s)


def test_block_stats_subsample_deterministic():
    w = _normal((64, 256))
    a = block_stats(w, 16, max_blocks=100, seed=3)
    b = block_stats(w, 16, max_blocks=100, seed=3)
    c = block_stats(w, 16, max_blocks=100, seed=4)
    assert a == b
    assert a["num_blocks"] == 1024 and a["num_blocks_used"] == 100
    assert a["l2"] != c["l2"]


def test_block_stats_bf16_and_row_normalization():
    w = (_normal((16, 128)) * 7.0).astype(mx.bfloat16)
    s = block_stats(w, 128, normalize_rows=True)
    # G == in_features: every block is a whole row -> L2 = sqrt(G) after row-RMS normalization
    assert s["l2"]["mean"] == pytest.approx(math.sqrt(128), rel=1e-4)
    assert s["l2"]["std"] == pytest.approx(0.0, abs=1e-3)


def test_direction_concentration():
    g = 16
    noise = block_stats(_normal((256, 256)), g)
    # few repeated block patterns (random signs/scales) -> strongly shared directions
    patterns = _normal((2, g), seed=1)
    idx = mx.random.randint(0, 2, (256 * 256 // g,), key=mx.random.key(2))
    scale = mx.random.uniform(-2, 2, (256 * 256 // g, 1), key=mx.random.key(3))
    structured = (patterns[idx] * scale).reshape(256, 256)
    st = block_stats(structured, g)
    assert noise["direction_concentration"] == pytest.approx(noise["direction_concentration_isotropic"], abs=0.02)
    assert st["direction_concentration"] > noise["direction_concentration"] + 0.2
