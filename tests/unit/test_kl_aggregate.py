"""kl_p95 is the global 95th percentile over all scored tokens, independent of the chunking (audit observation:
the old value, a token-weighted mean of per-chunk p95, is kept as kl_p95_chunk_mean)."""
import mlx.core as mx
import numpy as np
import pytest

from suc.analysis.sensitivity import logit_divergence
from suc.evaluation.reconstruct import aggregate_kl


def test_global_percentile_independent_of_chunking():
    rng = np.random.default_rng(0)
    ref = mx.array(rng.standard_normal((8, 33, 50)).astype(np.float32))
    test = ref + mx.array((rng.standard_normal((8, 33, 50)) * rng.exponential(0.3, (8, 1, 1))).astype(np.float32))
    tg = mx.array(rng.integers(0, 50, (8, 33)))
    full = logit_divergence(ref, test, tg, return_tokens=True)
    for chunk in (1, 2, 8):
        parts = [(logit_divergence(ref[s:s + chunk], test[s:s + chunk], tg[s:s + chunk], return_tokens=True),
                  min(chunk, 8 - s) * 33) for s in range(0, 8, chunk)]
        a = aggregate_kl(parts)
        assert a["kl_p95"] == pytest.approx(full["kl_p95"], rel=1e-6) and a["kl_p95_definition"] == "global"
        assert a["kl_mean"] == pytest.approx(full["kl_mean"], rel=1e-5) and a["tokens"] == 8 * 33
        if chunk == 1:
            assert a["kl_p95_chunk_mean"] != pytest.approx(a["kl_p95"], rel=1e-3)   # the old value is not a percentile
