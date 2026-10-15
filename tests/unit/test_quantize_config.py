"""Tests for the quality-track experiment config (Q1–Q4: codecs applied to a whole model)."""
from pathlib import Path

import pytest

from suc.config import CodecConfig, QuantEvalConfig, config_hash, config_to_dict, load_quantize_config, quantize_config_from_dict

REPO = Path(__file__).resolve().parents[2]


def _minimal(**codec) -> dict:
    return {"name": "q", "model": {"id": "Qwen/Qwen3-0.6B"}, "codecs": [{"kind": "vq", **codec}]}


def test_defaults():
    cfg = quantize_config_from_dict(_minimal())
    c = cfg.codecs[0]
    assert (c.kind, c.k, c.block_size, c.num_codebooks, c.rotate) == ("vq", 256, 8, 1, True)
    assert c.name == "vq_k256_g8_rot"
    assert cfg.roles is None and cfg.eval == QuantEvalConfig()
    assert cfg.eval.seq_len == 2048 and cfg.eval.kl_sequences == 16


def test_names_per_kind():
    names = [quantize_config_from_dict(_minimal()).codecs[0].name]
    cfg = quantize_config_from_dict({**_minimal(), "codecs": [
        {"kind": "bf16"}, {"kind": "affine", "bits": 4}, {"kind": "scalar", "k": 16},
        {"kind": "vq", "k": 256, "block_size": 4, "rotate": False},
        {"kind": "additive", "k": 256, "num_codebooks": 2, "refine_iters": 2}]})
    names += [c.name for c in cfg.codecs]
    assert names == ["vq_k256_g8_rot", "bf16", "affine_b4_g64", "scalar_k16_tensor", "vq_k256_g4", "additive_m2_k256_g8_rot_r2"]


def test_codec_validation():
    with pytest.raises(ValueError, match="bits"):
        quantize_config_from_dict({**_minimal(), "codecs": [{"kind": "affine"}]})
    with pytest.raises(ValueError, match="kind"):
        quantize_config_from_dict({**_minimal(), "codecs": [{"kind": "zip"}]})
    with pytest.raises(ValueError, match="num_codebooks"):
        quantize_config_from_dict({**_minimal(), "codecs": [{"kind": "additive", "num_codebooks": 1}]})
    with pytest.raises(ValueError, match="codecs"):
        quantize_config_from_dict({"name": "q", "model": {"id": "x"}, "codecs": []})


def test_codec_to_dict_for_make_codec():
    c = quantize_config_from_dict(_minimal(k=64)).codecs[0]
    assert c.codec_dict() == {"kind": "vq", "k": 64, "block_size": 8, "rotate": True, "kmeans_iters": 25,
                              "fit_blocks": 262144}


def test_round_trip_and_hash():
    cfg = quantize_config_from_dict({**_minimal(), "roles": ["mlp_down"], "eval": {"max_tokens": 4096}})
    again = quantize_config_from_dict(config_to_dict(cfg))
    assert again == cfg and config_hash(again) == config_hash(cfg)


def test_example_q_config_loads():
    cfg = load_quantize_config(REPO / "experiments/configs/q1q4_codecs_qwen3_0.6b.yaml")
    assert cfg.model.id == "Qwen/Qwen3-0.6B" and len(cfg.codecs) >= 8


def test_residual_codec_fields_and_name():
    c = quantize_config_from_dict({**_minimal(), "codecs": [
        {"kind": "additive", "num_codebooks": 3, "residual_fraction": 0.001}]}).codecs[0]
    assert (c.residual_fraction, c.residual_dtype, c.residual_selection) == (0.001, "int8", "activation")
    assert c.name == "additive_m3_k256_g8_rot+res0.001_activation"
    assert "residual_fraction" not in c.codec_dict()  # the base codec dict stays make_codec-compatible
    with pytest.raises(ValueError, match="residual_fraction"):
        quantize_config_from_dict({**_minimal(), "codecs": [{"kind": "vq", "residual_fraction": 1.5}]})
