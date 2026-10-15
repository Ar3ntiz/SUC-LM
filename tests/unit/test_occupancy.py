import math

import mlx.core as mx
import numpy as np
import pytest

from suc.analysis.occupancy import codebook_fit_report, occupancy_metrics


def test_uniform_assignments():
    m = occupancy_metrics(mx.array(np.tile(np.arange(16), 10), dtype=mx.uint32), 16)
    assert m["k"] == 16 and m["occupied"] == 16 and m["occupied_frac"] == 1.0
    assert m["entropy_bits"] == pytest.approx(4.0) and m["entropy_frac"] == pytest.approx(1.0)
    assert m["top1_share"] == pytest.approx(1 / 16)
    assert m["top10pct_share"] == pytest.approx(2 / 16)  # ceil(1.6) = 2 codes
    assert m["gini"] == pytest.approx(0.0, abs=1e-9)


def test_single_code():
    m = occupancy_metrics(mx.full((50,), 3, dtype=mx.uint32), 8)
    assert m["occupied"] == 1 and m["occupied_frac"] == 1 / 8
    assert m["entropy_bits"] == 0.0 and m["entropy_frac"] == 0.0
    assert m["top1_share"] == 1.0 and m["top10pct_share"] == 1.0
    assert m["gini"] == pytest.approx(1 - 1 / 8)  # maximal concentration over k codes


def test_two_codes_hand_computed():
    a = mx.array([0] * 3 + [1], dtype=mx.uint32)
    m = occupancy_metrics(a, 4)
    h = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert m["entropy_bits"] == pytest.approx(h) and m["entropy_frac"] == pytest.approx(h / 2)
    assert m["top1_share"] == 0.75


def structured_blocks(n=6000, g=8, n_centers=64, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_centers, g))
    x = centers[rng.integers(0, n_centers, n)] + 0.1 * rng.normal(size=(n, g))
    return mx.array(x.astype(np.float32))


def test_fit_report_fields_and_monotone_in_k():
    b = structured_blocks()
    reps = [codebook_fit_report(b, k, seed=0, iters=15) for k in (4, 16, 64)]
    r = reps[0]
    assert r["k"] == 4 and r["block_size"] == 8 and r["num_blocks"] == 6000
    assert r["heldout_blocks"] == 1200 and r["fit_blocks"] == 4800
    assert r["nominal_index_bpw"] == pytest.approx(2 / 8)
    assert r["codebook_bytes_fp16"] == 4 * 8 * 2 and r["execution_viable"] is True
    assert set(r["occupancy"]) >= {"occupied", "entropy_bits", "entropy_frac", "gini"}
    assert r["dist_mean"] <= r["dist_p95"] <= r["dist_p99"]
    fit = [x["rel_mse_fit"] for x in reps]
    held = [x["rel_mse_heldout"] for x in reps]
    assert fit[0] > fit[1] > fit[2] and held[0] > held[1] > held[2]
    assert held[2] < 0.05  # 64 true centers; noise floor ~0.01 plus k-means local minima


def test_fit_report_subsample_and_determinism():
    b = structured_blocks(n=5000)
    r1 = codebook_fit_report(b, 16, seed=3, iters=5, max_fit_blocks=1000)
    r2 = codebook_fit_report(b, 16, seed=3, iters=5, max_fit_blocks=1000)
    assert r1["fit_blocks"] == 1000 and r1["heldout_blocks"] == 1000
    assert r1 == r2


@pytest.mark.parametrize("k,g,viable", [(1024, 8, True), (512, 16, True), (1024, 16, False), (2048, 4, True)])
def test_execution_viable_boundary(k, g, viable):
    b = mx.random.normal((k + 200, g), key=mx.random.key(0))
    r = codebook_fit_report(b, k, iters=1, holdout_frac=0.05)
    assert r["codebook_bytes_fp16"] == k * g * 2
    assert r["execution_viable"] is viable


def test_gaussian_blocks_rel_mse_decreases():
    b = mx.random.normal((8000, 8), key=mx.random.key(1))
    held = [codebook_fit_report(b, k, iters=15)["rel_mse_heldout"] for k in (2, 16, 128)]
    assert all(0 < h < 1 for h in held)
    assert held[0] > held[1] > held[2]
