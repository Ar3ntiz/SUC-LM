"""End-to-end test of the Phase 2 runner on a tiny random Qwen3 with injected data (no network)."""
import json
from pathlib import Path

import mlx.core as mx
import yaml
from mlx.utils import tree_flatten

import suc.analysis.calibrate as calib
from suc.provenance import REQUIRED_METADATA_KEYS
from tiny_models import tiny_qwen3

FIX = Path(__file__).parent / "fixtures"


class FakeTokenizer:
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [2 + (ord(ch) % 120) for ch in text]


def _tiny_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    mx.save_safetensors(str(d / "model.safetensors"), dict(tree_flatten(tiny_qwen3().parameters())))
    cfg = json.loads((FIX / "qwen3_0.6b_config.json").read_text())
    cfg.update(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4,
               num_key_value_heads=2, head_dim=16, vocab_size=128, tie_word_embeddings=True,
               rope_theta=10000.0, max_position_embeddings=512)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def _docs() -> tuple[dict, list]:
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    make = lambda tag, n: [f"{tag} {i} " + " ".join(words[(i + j) % 10] for j in range(40)) for i in range(n)]
    return {d: make(d, 60) for d in ("natural", "code", "math", "math_competition", "instruction")}, make("ood", 40)


def test_run_calibration_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(calib, "ARTIFACT_DIR", str(tmp_path / "artifacts"))
    cfg = {"name": "p2_tiny", "output_dir": str(tmp_path / "results"),
           "model": {"id": "tiny/qwen3", "local_path": str(_tiny_model_dir(tmp_path))},
           "calibration": {"seq_len": 32, "n_calibration": 16, "n_validation": 8, "n_ood": 4, "micro_batch": 4},
           "sensitivity": {"k": 16, "measured_tensors": 6, "eval_sequences": 4, "eval_chunk": 2,
                           "fisher_sequences": 8, "fisher_micro_batch": 2, "fisher_group_size": 4}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    docs, ood = _docs()

    json_path = calib.run_calibration(path, tokenizer=FakeTokenizer(), docs=docs, ood_docs=ood)

    r = json.loads(json_path.read_text())
    assert set(REQUIRED_METADATA_KEYS) <= set(r["metadata"])
    assert r["metadata"]["dataset_hash"] == r["data"]["partitions_hash"]
    assert r["data"]["partitions"]["calibration"]["rows"] == 16 and r["data"]["partitions"]["ood"]["rows"] == 4
    assert r["activation_stats"]["tokens"] == 16 * 32
    assert len(r["tensors"]) == 2 * 7  # every eligible projection of the 2 layers gets a proxy score
    measured = [t for t in r["tensors"].values() if "measured" in t]
    assert len(measured) == 6
    for t in measured:
        assert t["measured"]["kl_mean"] >= 0 and t["measured"]["tokens"] == 4 * 31
    assert r["proxy_validation"]["n"] == 6
    assert all(t["fisher_proxy"] >= 0 for t in r["tensors"].values())
    assert "spearman_fisher_vs_kl" in r["proxy_validation"] and r["fisher"]["sequences"] == 8
    assert Path(r["activation_stats"]["artifact"]).exists() and Path(r["data"]["artifact"]).exists()
    assert json_path.with_suffix(".md").read_text().startswith("# Phase 2 calibration & sensitivity report: p2_tiny")
