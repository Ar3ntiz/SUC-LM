"""End-to-end test of the allocation runner (suc allocate) on a tiny random Qwen3."""
import json
from pathlib import Path

import mlx.core as mx
import pytest
import yaml

import suc.allocation.run as arun
from suc.config import load_allocate_config
from test_calibrate_run import _tiny_model_dir


def _setup(tmp_path: Path, monkeypatch) -> Path:
    mx.random.seed(0)
    tokens = mx.random.randint(0, 128, (2048,), dtype=mx.int32)
    monkeypatch.setattr(arun, "load_wikitext2", lambda split: ("unused", "wikihash"))
    monkeypatch.setattr(arun, "tokenize", lambda tok, text: tokens)
    monkeypatch.setattr(arun, "ARTIFACT_DIR", str(tmp_path / "artifacts"))
    art = tmp_path / "tiny_phase2_data.safetensors"
    mx.save_safetensors(str(art), {"validation": mx.random.randint(0, 128, (8, 64), dtype=mx.int32),
                                   "calibration": mx.random.randint(0, 128, (16, 64), dtype=mx.int32)})
    cfg = {"name": "a_tiny", "output_dir": str(tmp_path / "results"),
           "model": {"id": "tiny/qwen3", "local_path": str(_tiny_model_dir(tmp_path))},
           "eval": {"seq_len": 128, "max_tokens": 1024, "kl_sequences": 4, "chunk": 2, "calib_artifact": str(art)},
           "candidates": [{"kind": "affine", "bits": 2}, {"kind": "affine", "bits": 4}, {"kind": "affine", "bits": 8}],
           "budgets_bpp": [3.0, 5.0], "uniform": ["affine_b4_g64"],
           "fisher_sequences": 8, "fisher_micro_batch": 2}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_config_defaults_and_validation(tmp_path, monkeypatch):
    cfg = load_allocate_config(_setup(tmp_path, monkeypatch))
    assert cfg.budgets_bpp == (3.0, 5.0) and cfg.embedding_codec is None and cfg.fisher_loss == "ce"
    bad = yaml.safe_load(Path(tmp_path / "cfg.yaml").read_text())
    bad["uniform"] = ["vq_k256_g8_rot"]
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="uniform"):
        load_allocate_config(tmp_path / "bad.yaml")


def test_run_allocate_end_to_end(tmp_path, monkeypatch):
    r = json.loads(arun.run_allocate(_setup(tmp_path, monkeypatch)).read_text())
    plans = {p["plan"]: p for p in r["plans"]}
    assert set(plans) == {"greedy_3.0", "greedy_5.0", "uniform_affine_b4_g64"}
    for name in ("greedy_3.0", "greedy_5.0"):
        assert plans[name]["allocated_bpw"] <= float(name.split("_")[1]) + 1e-9
    assert plans["greedy_5.0"]["predicted_loss"] <= plans["greedy_3.0"]["predicted_loss"]
    assert plans["greedy_5.0"]["quality"]["kl"]["kl_mean"] <= plans["greedy_3.0"]["quality"]["kl"]["kl_mean"] * 1.5
    assert sum(plans["greedy_3.0"]["by_codec"].values()) == r["allocated_tensors"]
    assert Path(r["candidates_artifact"]).exists()
    assert Path(r["report"]).read_text().startswith("# Allocation report: a_tiny")
