"""Tests for the Phase 2 calibration / sensitivity config (spec §9, D5)."""
from pathlib import Path

import pytest

from suc.config import (
    CalibrationConfig, SensitivityConfig, calibration_config_from_dict, config_hash, config_to_dict,
    load_calibration_config,
)

REPO = Path(__file__).resolve().parents[2]


def _minimal() -> dict:
    return {"name": "calib", "model": {"id": "Qwen/Qwen3-0.6B"}}


def test_defaults():
    cfg = calibration_config_from_dict(_minimal())
    c, s = cfg.calibration, cfg.sensitivity
    assert c == CalibrationConfig() and s == SensitivityConfig()
    assert c.mix == {"natural": 0.35, "code": 0.25, "math": 0.17, "math_competition": 0.03, "instruction": 0.2}
    assert (c.seq_len, c.n_calibration, c.n_validation, c.n_ood) == (512, 512, 128, 64)
    assert s.kind == "vq" and (s.block_size, s.k, s.rotate) == (8, 256, True)
    assert (s.fisher, s.fisher_loss, s.fisher_sequences, s.fisher_micro_batch, s.fisher_group_size) == (True, "ce", 32, 2, None)


def test_mix_must_sum_to_one_and_use_known_domains():
    with pytest.raises(ValueError, match="sum"):
        calibration_config_from_dict({**_minimal(), "calibration": {"mix": {"natural": 0.5, "code": 0.2}}})
    with pytest.raises(ValueError, match="domain"):
        calibration_config_from_dict({**_minimal(), "calibration": {"mix": {"natural": 0.5, "poetry": 0.5}}})


def test_unknown_keys_rejected():
    with pytest.raises(ValueError, match="levels"):
        calibration_config_from_dict({**_minimal(), "sensitivity": {"levels": [0.1]}})


def test_measured_tensors_bounds():
    with pytest.raises(ValueError, match="measured_tensors"):
        calibration_config_from_dict({**_minimal(), "sensitivity": {"measured_tensors": 0}})


def test_round_trip_and_hash():
    cfg = calibration_config_from_dict({**_minimal(), "sensitivity": {"k": 64, "measured_tensors": 8}})
    again = calibration_config_from_dict(config_to_dict(cfg))
    assert again == cfg and config_hash(again) == config_hash(cfg)


def test_example_phase2_config_loads():
    cfg = load_calibration_config(REPO / "experiments/configs/phase2_calib_qwen3_0.6b.yaml")
    assert cfg.model.id == "Qwen/Qwen3-0.6B" and len(cfg.model.revision) == 40
