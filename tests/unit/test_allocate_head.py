"""Embedding / untied LM head as allocation candidates (head_candidates in AllocateExperimentConfig)."""
import json
from pathlib import Path

import mlx.core as mx
import pytest
import yaml
from mlx.utils import tree_flatten

import suc.allocation.run as arun
from suc.analysis.fisher import fisher_diag
from suc.config import allocate_config_from_dict, config_hash, config_to_dict, load_allocate_config
from suc.models.adapters import get_adapter
from test_allocate_run import _setup
from test_calibrate_run import FIX
from tiny_models import tiny_qwen3

HEAD = [{"kind": "affine", "bits": 4}, {"kind": "affine", "bits": 8}, {"kind": "bf16"}]


def _untied_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "model_untied"
    d.mkdir()
    mx.save_safetensors(str(d / "model.safetensors"), dict(tree_flatten(tiny_qwen3(tie=False).parameters())))
    cfg = json.loads((FIX / "qwen3_0.6b_config.json").read_text())
    cfg.update(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4,
               num_key_value_heads=2, head_dim=16, vocab_size=128, tie_word_embeddings=False,
               rope_theta=10000.0, max_position_embeddings=512)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def _head_cfg(tmp_path: Path, monkeypatch, *, untied: bool = True, **over) -> Path:
    path = _setup(tmp_path, monkeypatch)
    cfg = yaml.safe_load(path.read_text())
    if untied:
        cfg["model"]["local_path"] = str(_untied_model_dir(tmp_path))
    cfg.update(head_candidates=HEAD, uniform_head="affine_b8_g64", **over)
    path.write_text(yaml.safe_dump(cfg))
    return path


# ------------------------------------------------------------------------------------------- config
def _base_dict(**over) -> dict:
    d = {"name": "x", "model": {"id": "tiny/qwen3"}, "candidates": [{"kind": "affine", "bits": 2}],
         "budgets_bpp": [3.0], "head_candidates": HEAD, "uniform_head": "bf16"}
    d.update(over)
    return d


def test_head_candidates_parse_and_hash_round_trip():
    cfg = allocate_config_from_dict(_base_dict())
    assert [c.name for c in cfg.head_candidates] == ["affine_b4_g64", "affine_b8_g64", "bf16"]
    assert cfg.uniform_head == "bf16"
    again = allocate_config_from_dict(config_to_dict(cfg))
    assert again == cfg and config_hash(again) == config_hash(cfg)


def test_head_candidates_default_none_keeps_old_hash_fields():
    d = _base_dict()
    del d["head_candidates"], d["uniform_head"]
    cfg = allocate_config_from_dict(d)
    assert cfg.head_candidates is None and cfg.uniform_head is None


@pytest.mark.parametrize("over, match", [
    ({"head_candidates": [{"kind": "affine", "bits": 4, "bogus": 1}]}, "unknown key"),
    ({"head_candidates": []}, "head_candidates"),
    ({"uniform_head": "affine_b2_g64"}, "uniform_head"),
    ({"embedding_codec": {"kind": "affine", "bits": 4}}, "embedding_codec"),
    ({"head_candidates": [{"kind": "affine", "bits": 4, "residual_fraction": 0.01}]}, "activation"),
])
def test_head_candidates_validation(over, match):
    with pytest.raises(ValueError, match=match):
        allocate_config_from_dict(_base_dict(**over))


def test_uniform_head_required_with_uniform_baselines():
    d = _base_dict(uniform=["affine_b2_g64"])
    del d["uniform_head"]
    with pytest.raises(ValueError, match="uniform_head"):
        allocate_config_from_dict(d)
    d["uniform"] = []
    assert allocate_config_from_dict(d).uniform_head is None   # not needed without uniform baselines


def test_uniform_head_without_head_candidates_is_an_error():
    d = _base_dict()
    del d["head_candidates"]
    with pytest.raises(ValueError, match="uniform_head"):
        allocate_config_from_dict(d)


# ----------------------------------------------------------------------------------- head selection
def test_head_specs_tied_vs_untied(tmp_path):
    d = _untied_model_dir(tmp_path)
    cfg = json.loads((d / "config.json").read_text())
    names = list(dict(tree_flatten(tiny_qwen3(tie=False).parameters())))
    assert [s.name for s in arun.head_specs(get_adapter(cfg), names)] == ["model.embed_tokens.weight", "lm_head.weight"]
    cfg["tie_word_embeddings"] = True
    tied_names = [n for n in names if n != "lm_head.weight"]
    assert [s.name for s in arun.head_specs(get_adapter(cfg), tied_names)] == ["model.embed_tokens.weight"]


def test_fisher_for_embedding_and_untied_head():
    model = tiny_qwen3(tie=False)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 2, dtype=mx.int32)   # only tokens 1..8 appear
    f = fisher_diag(model, tokens, ["model.embed_tokens.weight", "lm_head.weight"], micro_batch=1)
    emb, head = f["model.embed_tokens.weight"], f["lm_head.weight"]
    assert emb.shape == (128, 64) and head.shape == (128, 64)
    row_sum = mx.sum(emb, axis=1)
    # rows of tokens never fed as input (the last token is only a target) get exactly zero Fisher
    assert mx.all(row_sum[mx.array([0] + list(range(8, 128)))] == 0).item()
    assert mx.all(row_sum[1:8] > 0).item()
    assert mx.all(mx.sum(head, axis=1) > 0).item()   # softmax gradient touches every vocab row of the head


# --------------------------------------------------------------------------------------- end to end
def test_run_allocate_with_head_candidates_untied(tmp_path, monkeypatch):
    r = json.loads(arun.run_allocate(_head_cfg(tmp_path, monkeypatch)).read_text())
    head_names = ["model.embed_tokens.weight", "lm_head.weight"]
    block_params = 73728                              # eligible linears of the tiny Qwen3 (as in the old path)
    assert r["head_tensors"] == head_names
    assert r["allocated_params"] == block_params + 2 * 128 * 64
    assert r["allocated_tensors"] == 14 + 2
    plans = {p["plan"]: p for p in r["plans"]}
    assert set(plans) == {"greedy_3.0", "greedy_5.0", "uniform_affine_b4_g64"}
    for name in ("greedy_3.0", "greedy_5.0"):
        p = plans[name]
        assert p["allocated_bpw"] <= float(name.split("_")[1]) + 1e-9
        assert set(p["head"]) == set(head_names)
        assert {row["tensor"] for row in p["rows"]} >= set(head_names)
        # every allocated tensor (blocks + head) is encoded exactly once by reconstruction
        assert p["size"]["encoded_params"] == r["allocated_params"]
        # candidate bits (scoring path) == reconstruction bits (same codec objects, same seeds)
        enc_bits = p["size"]["bits_total"] - p["size"]["breakdown"]["unencoded"]
        assert enc_bits == round(p["allocated_bpw"] * r["allocated_params"])
    u = plans["uniform_affine_b4_g64"]
    assert u["head"] == dict.fromkeys(head_names, "affine_b8_g64")
    assert plans["greedy_5.0"]["predicted_loss"] <= plans["greedy_3.0"]["predicted_loss"]
    assert "head" in Path(r["report"]).read_text()


def test_run_allocate_with_head_candidates_tied(tmp_path, monkeypatch):
    r = json.loads(arun.run_allocate(_head_cfg(tmp_path, monkeypatch, untied=False)).read_text())
    assert r["head_tensors"] == ["model.embed_tokens.weight"]
    assert r["allocated_params"] == 73728 + 128 * 64


def test_old_path_has_no_head_tensors(tmp_path, monkeypatch):
    r = json.loads(arun.run_allocate(_setup(tmp_path, monkeypatch)).read_text())
    assert r["allocated_params"] == 73728 and r["head_tensors"] == []
    for p in r["plans"]:
        assert "model.embed_tokens.weight" not in {row["tensor"] for row in p["rows"]} and p["head"] == {}
