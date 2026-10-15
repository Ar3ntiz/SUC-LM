"""Tests for suc.config: YAML/dict parsing, validation, round-trip, hashing."""
from pathlib import Path

import pytest
import yaml

from suc.config import (
    BenchConfig, DatasetConfig, ExperimentConfig, ModelConfig, RepresentationConfig,
    config_from_dict, config_hash, config_to_dict, load_config,
)

REPO = Path(__file__).resolve().parents[2]


def _minimal() -> dict:
    return {"name": "exp", "model": {"id": "Qwen/Qwen3-0.6B"}, "representations": [{"kind": "bf16"}]}


def test_minimal_defaults():
    cfg = config_from_dict(_minimal())
    assert cfg.name == "exp"
    assert cfg.model == ModelConfig(id="Qwen/Qwen3-0.6B")
    assert cfg.model.revision == "main" and cfg.model.role == "dev" and cfg.model.local_path is None
    assert cfg.dataset == DatasetConfig()
    assert cfg.bench == BenchConfig()
    assert cfg.bench.prompt_lengths == (128, 512)
    assert cfg.seed == 12345
    assert cfg.output_dir == "experiments/results"
    assert cfg.representations[0].name == "bf16"


def test_frozen():
    cfg = config_from_dict(_minimal())
    with pytest.raises(Exception):
        cfg.seed = 1  # type: ignore[misc]


def test_mlx_affine_defaults_and_name():
    r = config_from_dict({**_minimal(), "representations": [{"kind": "mlx_affine", "bits": 4}]}).representations[0]
    assert r.bits == 4 and r.group_size == 64 and r.name == "mlx_affine_b4_g64"
    r = config_from_dict({**_minimal(), "representations": [
        {"kind": "mlx_affine", "bits": 3, "group_size": 32, "name": "q3"}]}).representations[0]
    assert r.name == "q3" and r.group_size == 32


@pytest.mark.parametrize("rep", [
    {"kind": "mlx_affine"},                      # bits missing
    {"kind": "mlx_affine", "bits": 7},           # invalid bits
    {"kind": "bf16", "bits": 4},                 # bits not allowed
    {"kind": "fp16", "group_size": 64},          # group_size not allowed
    {"kind": "int4"},                            # unknown kind
])
def test_invalid_representation(rep):
    with pytest.raises(ValueError):
        config_from_dict({**_minimal(), "representations": [rep]})


def test_representations_required_nonempty():
    d = _minimal(); d["representations"] = []
    with pytest.raises(ValueError):
        config_from_dict(d)
    d.pop("representations")
    with pytest.raises(ValueError, match="representations"):
        config_from_dict(d)


def test_missing_required():
    d = _minimal(); d.pop("name")
    with pytest.raises(ValueError, match="name"):
        config_from_dict(d)
    d = _minimal(); d["model"] = {}
    with pytest.raises(ValueError, match="id"):
        config_from_dict(d)


@pytest.mark.parametrize("path", [(), ("model",), ("dataset",), ("bench",)])
def test_unknown_key_rejected(path):
    d = _minimal()
    tgt = d
    for p in path:
        tgt = tgt.setdefault(p, {})
    tgt["bogus_key"] = 1
    with pytest.raises(ValueError, match="bogus_key"):
        config_from_dict(d)


def test_unknown_key_in_representation():
    d = {**_minimal(), "representations": [{"kind": "bf16", "bogus_key": 1}]}
    with pytest.raises(ValueError, match="bogus_key"):
        config_from_dict(d)


def test_invalid_literals():
    for d in (
        {**_minimal(), "model": {"id": "x", "role": "prod"}},
        {**_minimal(), "dataset": {"name": "c4"}},
        {**_minimal(), "dataset": {"split": "train"}},
    ):
        with pytest.raises(ValueError):
            config_from_dict(d)


def test_roundtrip_and_hash():
    d = {
        "name": "full", "seed": 7, "output_dir": "out",
        "model": {"id": "Qwen/Qwen3-8B", "revision": "abc123", "local_path": "/m", "role": "target"},
        "dataset": {"name": "wikitext2", "split": "validation", "seq_len": 1024, "max_tokens": 4096},
        "representations": [{"kind": "bf16"}, {"kind": "mlx_affine", "bits": 8, "group_size": 64}],
        "bench": {"prompt_lengths": [64], "decode_tokens": 16, "batch_sizes": [1, 4], "warmup": 0, "repeats": 2},
    }
    cfg = config_from_dict(d)
    assert cfg.bench.batch_sizes == (1, 4)
    out = config_to_dict(cfg)
    assert config_from_dict(out) == cfg
    import json
    json.dumps(out)  # JSON-serialisable
    h = config_hash(cfg)
    assert len(h) == 64 and h == config_hash(config_from_dict(out))
    assert h != config_hash(config_from_dict({**d, "seed": 8}))


def test_load_config_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(_minimal()))
    assert load_config(p) == config_from_dict(_minimal())
    assert isinstance(load_config(str(p)), ExperimentConfig)


def test_example_phase0_config_loads():
    cfg = load_config(REPO / "experiments/configs/phase0_dev_qwen3_0.6b.yaml")
    assert cfg.model.id == "Qwen/Qwen3-0.6B" and cfg.model.role == "dev"
    assert cfg.dataset == DatasetConfig(name="wikitext2", split="test", seq_len=2048, max_tokens=32768)
    assert [r.name for r in cfg.representations] == ["bf16", "mlx_affine_b8_g64", "mlx_affine_b4_g64"]
    assert all(isinstance(r, RepresentationConfig) for r in cfg.representations)
