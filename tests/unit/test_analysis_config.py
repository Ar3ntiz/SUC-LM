"""Tests for the Phase 1 weight-analysis config (spec §8)."""
from pathlib import Path

import pytest

from suc.config import (
    AnalysisConfig, ModelConfig, analysis_config_from_dict, config_hash, config_to_dict, load_analysis_config,
)

REPO = Path(__file__).resolve().parents[2]


def _minimal() -> dict:
    return {"name": "weights", "model": {"id": "Qwen/Qwen3-0.6B"}}


def test_minimal_defaults():
    cfg = analysis_config_from_dict(_minimal())
    assert cfg.model == ModelConfig(id="Qwen/Qwen3-0.6B")
    a = cfg.analysis
    assert a == AnalysisConfig()
    assert a.layers is None  # None = first, middle and last layer
    assert a.block_sizes == (8, 16, 32, 64, 128)
    assert a.kmeans_block_sizes == (8, 16)
    assert a.codebook_sizes == (16, 64, 256, 1024)
    assert a.rotations == (False, True)


def test_unknown_analysis_key_rejected():
    with pytest.raises(ValueError, match="block_size"):
        analysis_config_from_dict({**_minimal(), "analysis": {"block_size": [8]}})


def test_kmeans_block_sizes_must_be_subset_of_block_sizes():
    with pytest.raises(ValueError, match="kmeans_block_sizes"):
        analysis_config_from_dict({**_minimal(), "analysis": {"block_sizes": [8], "kmeans_block_sizes": [16]}})


def test_unknown_role_rejected():
    with pytest.raises(ValueError, match="roles"):
        analysis_config_from_dict({**_minimal(), "analysis": {"roles": ["mlp_up", "attention"]}})


def test_round_trip_and_hash():
    d = {**_minimal(), "analysis": {"layers": [0, 5], "codebook_sizes": [16, 256], "experts_per_layer": 2}}
    cfg = analysis_config_from_dict(d)
    assert cfg.analysis.layers == (0, 5)
    again = analysis_config_from_dict(config_to_dict(cfg))
    assert again == cfg and config_hash(again) == config_hash(cfg)


def test_example_phase1_config_loads():
    cfg = load_analysis_config(REPO / "experiments/configs/phase1_weights_qwen3_0.6b.yaml")
    assert cfg.model.id == "Qwen/Qwen3-0.6B" and len(cfg.model.revision) == 40
