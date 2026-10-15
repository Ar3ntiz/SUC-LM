"""Tests for Phase 1 result aggregation and the Markdown report."""
import pytest

from suc.analysis.report import markdown_report, summarize


def _cb(rotation: bool, g: int, k: int, mse: float, ent: float) -> dict:
    return {"rotation": rotation, "block_size": g, "k": k, "num_blocks": 1000, "rel_mse_heldout": mse,
            "rel_mse_fit": mse * 0.9, "nominal_index_bpw": {16: 4, 256: 8}[k] / g,
            "execution_viable": k * g * 2 <= 16384, "occupancy": {"entropy_frac": ent, "occupied_frac": 1.0}}


def _tensor(name: str, role: str, mses: dict) -> dict:
    return {"name": name, "role": role, "layer": 0, "expert": None, "shape": [64, 64],
            "scalar": {"kurtosis": 1.0, "outlier_rate": {"6.0": 0.0}},
            "scalar_rotated": {"kurtosis": 0.1, "outlier_rate": {"6.0": 0.0}},
            "codebooks": [_cb(rot, g, k, m, 0.9) for (rot, g, k), m in mses.items()]}


def _result() -> dict:
    t1 = _tensor("a.q", "attn_q", {(False, 8, 256): 0.20, (True, 8, 256): 0.10, (False, 8, 16): 0.5})
    t2 = _tensor("a.k", "attn_k", {(False, 8, 256): 0.40, (True, 8, 256): 0.30, (False, 8, 16): 0.7})
    return {"name": "phase1_x", "metadata": {"model": "M", "model_hash": "h", "git_commit": "c"},
            "tensors": [t1, t2]}


def test_summarize_groups_by_role_family_rotation_g_k():
    rows = summarize(_result()["tensors"])
    row = next(r for r in rows if r["rotation"] and r["block_size"] == 8 and r["k"] == 256)
    assert row["group"] == "attn" and row["tensors"] == 2
    assert row["rel_mse_heldout_mean"] == pytest.approx(0.20)
    assert row["rel_mse_heldout_max"] == pytest.approx(0.30)
    assert row["nominal_index_bpw"] == pytest.approx(1.0)
    assert row["execution_viable"] is True
    assert row["gaussian_rd_bound"] == pytest.approx(0.25)


def test_summarize_sorted_by_group_rotation_bpw():
    rows = summarize(_result()["tensors"])
    keys = [(r["group"], r["rotation"], r["nominal_index_bpw"]) for r in rows]
    assert keys == sorted(keys)


def test_markdown_report_contains_deliverable_columns_and_rotation_gain():
    md = markdown_report(_result())
    assert "# Phase 1 weight-structure report: phase1_x" in md
    for col in ("Tensor", "Block size", "Blocks", "K", "Entropy", "Rel. MSE"):
        assert col in md
    assert "a.q" in md and "a.k" in md
    assert "Rotation" in md and "Gaussian R(D)" in md
