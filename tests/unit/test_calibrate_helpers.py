"""Tests for the pure helpers of the Phase 2 runner."""
import pytest

from suc.analysis.calibrate import aggregate_chunks, sample_measured
from suc.models.adapters import LinearSpec


def _specs(layers: int = 6) -> list[LinearSpec]:
    roles = ["attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"]
    return [LinearSpec(f"model.layers.{i}.{r}.weight", r, i, None, True) for i in range(layers) for r in roles]


def test_sample_measured_is_deterministic_and_sized():
    a = sample_measured(_specs(), 10, seed=3)
    assert a == sample_measured(_specs(), 10, seed=3) and len(a) == 10 and len(set(a)) == 10


def test_sample_measured_covers_every_role_and_spreads_layers():
    picked = sample_measured(_specs(), 14, seed=0)
    by_name = {s.name: s for s in _specs()}
    assert {by_name[n].role for n in picked} == {s.role for s in _specs()}
    assert len({by_name[n].layer for n in picked}) >= 4


def test_sample_measured_caps_at_available():
    assert len(sample_measured(_specs(1), 100, seed=0)) == 7


def test_aggregate_chunks_token_weighted():
    chunks = [({"kl_mean": 1.0, "delta_ce": 0.5, "top1_agreement": 1.0}, 100),
              ({"kl_mean": 3.0, "delta_ce": 1.5, "top1_agreement": 0.0}, 300)]
    agg = aggregate_chunks(chunks)
    assert agg["kl_mean"] == pytest.approx(2.5) and agg["delta_ce"] == pytest.approx(1.25)
    assert agg["top1_agreement"] == pytest.approx(0.25) and agg["tokens"] == 400


def test_aggregate_chunks_ignores_non_numeric_and_missing():
    agg = aggregate_chunks([({"kl_mean": 1.0, "note": "x"}, 10), ({"kl_mean": 2.0}, 10)])
    assert agg["kl_mean"] == pytest.approx(1.5) and "note" not in agg
