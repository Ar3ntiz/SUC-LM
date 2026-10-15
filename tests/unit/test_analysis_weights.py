"""End-to-end test of the Phase 1 analysis runner on a tiny random Qwen3 saved to disk."""
import json
from pathlib import Path

import mlx.core as mx
import yaml
from mlx.utils import tree_flatten

from suc.analysis.weights import analyze_tensor, gaussian_rd_bound, run_analysis
from suc.config import AnalysisConfig
from suc.provenance import REQUIRED_METADATA_KEYS
from tiny_models import tiny_qwen3

FIX = Path(__file__).parent / "fixtures"


def _tiny_model_dir(tmp_path: Path) -> Path:
    model = tiny_qwen3()
    d = tmp_path / "model"
    d.mkdir()
    mx.save_safetensors(str(d / "model.safetensors"), dict(tree_flatten(model.parameters())))
    cfg = json.loads((FIX / "qwen3_0.6b_config.json").read_text())
    cfg.update(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4,
               num_key_value_heads=2, head_dim=16, vocab_size=128, tie_word_embeddings=True)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def _small_analysis() -> dict:
    return {"block_sizes": [8, 16], "kmeans_block_sizes": [8], "codebook_sizes": [16, 64],
            "max_blocks": 4096, "kmeans_iters": 5}


def test_gaussian_rd_bound():
    assert gaussian_rd_bound(1.0) == 0.25
    assert gaussian_rd_bound(0.5) == 0.5


def test_analyze_tensor_rotation_lowers_kurtosis_of_outlier_matrix():
    mx.random.seed(0)
    w = mx.random.normal((64, 128)) * 0.02
    w = w.at[3, 5].add(5.0)
    res = analyze_tensor(w, AnalysisConfig(**{**_small_analysis(), "block_sizes": (8,), "codebook_sizes": (16,)}),
                         seed=1, name="t")
    assert res["rotation_supported"]
    assert res["scalar_rotated"]["kurtosis"] < res["scalar"]["kurtosis"]
    assert {c["rotation"] for c in res["codebooks"]} == {False, True}


def test_analyze_tensor_skips_rotation_for_unsupported_size():
    w = mx.random.normal((64, 72))  # 72 = 9 * 8: no Hadamard of that size
    res = analyze_tensor(w, AnalysisConfig(block_sizes=(8,), kmeans_block_sizes=(8,), codebook_sizes=(16,),
                                           max_blocks=1024, kmeans_iters=3), seed=1, name="t")
    assert not res["rotation_supported"] and res["scalar_rotated"] is None
    assert {c["rotation"] for c in res["codebooks"]} == {False}


def test_run_analysis_writes_json_and_markdown(tmp_path):
    cfg = {"name": "p1_tiny", "output_dir": str(tmp_path / "results"),
           "model": {"id": "tiny/qwen3", "local_path": str(_tiny_model_dir(tmp_path))},
           "analysis": _small_analysis()}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))

    json_path, md_path = run_analysis(path)

    result = json.loads(json_path.read_text())
    assert set(REQUIRED_METADATA_KEYS) <= set(result["metadata"])
    assert result["metadata"]["dataset_hash"] == "none"
    names = {t["name"] for t in result["tensors"]}
    assert "model.layers.0.mlp.down_proj.weight" in names and "model.layers.1.self_attn.q_proj.weight" in names
    assert all("embed" not in n for n in names)
    t = result["tensors"][0]
    assert set(t["blocks"]) == {"8", "16"}
    assert len(t["codebooks"]) == 2 * 1 * 2  # rotations x kmeans block sizes x K
    assert result["summary"] and md_path.read_text().startswith("# Phase 1 weight-structure report: p1_tiny")


def test_run_metadata_is_captured_before_any_tensor_work(tmp_path, monkeypatch):
    import suc.analysis.weights as w

    calls = []
    real_meta, real_analyze = w.run_metadata, w.analyze_tensor
    monkeypatch.setattr(w, "run_metadata", lambda **kw: calls.append("metadata") or real_meta(**kw))
    monkeypatch.setattr(w, "analyze_tensor", lambda *a, **kw: calls.append("tensor") or real_analyze(*a, **kw))
    cfg = {"name": "p1_order", "output_dir": str(tmp_path / "results"),
           "model": {"id": "tiny/qwen3", "local_path": str(_tiny_model_dir(tmp_path))},
           "analysis": {**_small_analysis(), "layers": [0], "roles": ["mlp_down"], "codebook_sizes": [16]}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run_analysis(path)
    assert calls[0] == "metadata" and "tensor" in calls
