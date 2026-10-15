import mlx.core as mx
import numpy as np
import pytest

from suc.quant.kmeans import KMeansResult, kmeans, nearest


def blobs(k=8, g=4, per=200, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, g)).astype(np.float32) * 5
    labels = np.repeat(np.arange(k), per)
    x = centers[labels] + spread * rng.normal(size=(k * per, g)).astype(np.float32)
    perm = rng.permutation(len(x))
    return mx.array(x[perm]), labels[perm], centers


def purity(assign: np.ndarray, labels: np.ndarray) -> float:
    return sum(np.bincount(labels[assign == c]).max() for c in np.unique(assign)) / len(labels)


def test_recovers_separated_clusters():
    x, labels, centers = blobs()
    r = kmeans(x, 8, seed=1)
    assert isinstance(r, KMeansResult)
    assert r.codebook.shape == (8, 4) and r.codebook.dtype == mx.float32
    assert r.assignments.shape == (x.shape[0],) and r.assignments.dtype == mx.uint32
    a = np.array(r.assignments)
    assert purity(a, labels) == 1.0
    cb = np.array(r.codebook)
    d = ((cb[:, None, :] - centers[None]) ** 2).sum(-1)
    assert d.min(axis=1).max() < 0.01  # every learned code sits on a true center
    assert sorted(d.argmin(axis=1)) == list(range(8))
    assert r.converged and 1 <= r.iterations <= 25
    assert r.inertia == pytest.approx(float(nearest(x, r.codebook)[1].sum()), rel=1e-4)


@pytest.mark.parametrize("init", ["kmeans++", "random"])
def test_deterministic_by_seed(init):
    x = mx.random.normal((3000, 8), key=mx.random.key(3))
    r1, r2 = kmeans(x, 32, iters=5, seed=7, init=init), kmeans(x, 32, iters=5, seed=7, init=init)
    assert mx.array_equal(r1.codebook, r2.codebook) and mx.array_equal(r1.assignments, r2.assignments)
    r3 = kmeans(x, 32, iters=5, seed=8, init=init)
    assert not mx.array_equal(r1.codebook, r3.codebook)


def test_k_larger_than_distinct_points():
    base = mx.array(np.arange(5 * 3, dtype=np.float32).reshape(5, 3))
    x = mx.concatenate([base] * 20)  # 100 points, 5 distinct
    r = kmeans(x, 16, seed=0)
    cb = np.array(r.codebook)
    assert np.isfinite(cb).all()
    assert r.inertia == pytest.approx(0.0, abs=1e-6)
    assert len(np.unique(np.array(r.assignments))) == 5


def test_empty_cluster_reseeded_from_farthest_points():
    # random init on heavily duplicated data forces empty clusters; the far outlier must get its own code
    x = mx.concatenate([mx.zeros((500, 2)), mx.array([[100.0, 100.0]])])
    r = kmeans(x, 2, init="random", seed=0)
    assert r.inertia == pytest.approx(0.0, abs=1e-6)
    assert int(np.array(r.assignments)[-1]) != int(np.array(r.assignments)[0])


def test_weighted_distance_changes_assignment():
    cb = mx.array([[1.0, 0.0], [0.0, 1.5]])
    x = mx.array([[0.0, 0.0]])
    assert int(nearest(x, cb)[0][0]) == 0  # 1 < 2.25
    h = mx.array([10.0, 1.0])  # coordinate 0 matters much more
    a, d = nearest(x, cb, weights=h)
    assert int(a[0]) == 1 and float(d[0]) == pytest.approx(2.25)
    a2, _ = nearest(x, cb, weights=mx.array([[10.0, 1.0]]))  # per-row (N,G) form
    assert int(a2[0]) == 1


def test_weighted_kmeans_centroid_is_weighted_mean():
    x = mx.array([[0.0, 0.0], [1.0, 1.0]])
    w = mx.array([[3.0, 1.0], [1.0, 3.0]])
    r = kmeans(x, 1, weights=w)
    assert np.allclose(np.array(r.codebook), [[0.25, 0.75]], atol=1e-6)


def test_nearest_chunked_equals_unchunked():
    x = mx.random.normal((1037, 8), key=mx.random.key(0))
    cb = mx.random.normal((64, 8), key=mx.random.key(1))
    w = mx.random.uniform(0.1, 2.0, (1037, 8), key=mx.random.key(2))
    for weights in (None, w[0], w):
        a1, d1 = nearest(x, cb, weights=weights, chunk=100)
        a2, d2 = nearest(x, cb, weights=weights, chunk=1 << 20)
        assert mx.array_equal(a1, a2) and mx.allclose(d1, d2).item()
    ref = ((np.array(x)[:, None] - np.array(cb)[None]) ** 2).sum(-1)
    assert np.array_equal(np.array(nearest(x, cb, chunk=100)[0]), ref.argmin(1))
    assert np.allclose(np.array(nearest(x, cb)[1]), ref.min(1), rtol=1e-5, atol=1e-5)


def test_rejects_k_above_n():
    with pytest.raises(ValueError):
        kmeans(mx.zeros((3, 2)), 4)
