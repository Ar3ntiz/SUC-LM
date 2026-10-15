"""Tests for architecture adapters (D7/D8). Tensor names are generated from real configs; no weights."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from suc.models.adapters import (
    LinearSpec,
    Qwen3Adapter,
    Qwen3MoeAdapter,
    UnsupportedArchitecture,
    get_adapter,
)

FIX = Path(__file__).parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


def _attn(i: int) -> list[str]:
    p = f"model.layers.{i}"
    return [f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight",
            f"{p}.self_attn.q_norm.weight", f"{p}.self_attn.k_norm.weight",
            *(f"{p}.self_attn.{x}_proj.weight" for x in "qkvo")]


def _dense_mlp(i: int) -> list[str]:
    return [f"model.layers.{i}.mlp.{x}_proj.weight" for x in ("gate", "up", "down")]


def _head(cfg: dict) -> list[str]:
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    return names if cfg.get("tie_word_embeddings") else names + ["lm_head.weight"]


def dense_names(cfg: dict) -> list[str]:
    out = _head(cfg)
    for i in range(cfg["num_hidden_layers"]):
        out += _attn(i) + _dense_mlp(i)
    return out


def is_sparse(cfg: dict, i: int) -> bool:
    return i not in cfg.get("mlp_only_layers", []) and (i + 1) % cfg.get("decoder_sparse_step", 1) == 0


def moe_names(cfg: dict) -> list[str]:
    out = _head(cfg)
    for i in range(cfg["num_hidden_layers"]):
        out += _attn(i)
        if is_sparse(cfg, i):
            out.append(f"model.layers.{i}.mlp.gate.weight")
            out += [f"model.layers.{i}.mlp.experts.{e}.{x}_proj.weight"
                    for e in range(cfg["num_experts"]) for x in ("gate", "up", "down")]
        else:
            out += _dense_mlp(i)
    return out


def small_moe(**kw) -> dict:
    cfg = load("qwen3_30b_a3b_config.json")
    cfg.update(num_hidden_layers=4, num_experts=8, **kw)
    return cfg


# --- dispatch ---------------------------------------------------------------

def test_get_adapter_dispatch():
    assert isinstance(get_adapter(load("qwen3_8b_config.json")), Qwen3Adapter)
    assert isinstance(get_adapter(load("qwen3_30b_a3b_config.json")), Qwen3MoeAdapter)


@pytest.mark.parametrize("cfg", [{"model_type": "llama"}, {}])
def test_unknown_model_type_raises(cfg):
    with pytest.raises(UnsupportedArchitecture):
        get_adapter(cfg)
    assert issubclass(UnsupportedArchitecture, ValueError)


# --- dense ------------------------------------------------------------------

def test_dense_tied_0_6b():
    cfg = load("qwen3_0.6b_config.json")
    a, names = get_adapter(cfg), dense_names(cfg)
    assert (a.num_layers, a.is_moe, a.num_experts, a.family) == (28, False, 0, "qwen3")
    specs = a.linears(names)
    assert len(specs) == 28 * 7 + 1
    assert not any(s.role == "lm_head" for s in specs)
    assert specs[0] == LinearSpec("model.embed_tokens.weight", "embed", None, None, False)
    assert sum(s.eligible for s in specs) == 28 * 7
    assert a.unclassified(names) == []
    a.validate(names)


def test_dense_untied_8b_and_order():
    cfg = load("qwen3_8b_config.json")
    a, names = get_adapter(cfg), dense_names(cfg)
    specs = a.linears(reversed(names))
    assert len(specs) == 36 * 7 + 2
    assert specs[-1].role == "lm_head" and not specs[-1].eligible and specs[-1].layer is None
    assert [s.role for s in specs[1:8]] == ["attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"]
    assert all(s.layer == 0 and s.expert is None for s in specs[1:8])
    assert [s.layer for s in specs[1:-1]] == sorted(s.layer for s in specs[1:-1])
    a.validate(names)


def test_norms_not_returned():
    cfg = load("qwen3_8b_config.json")
    names = {s.name for s in get_adapter(cfg).linears(dense_names(cfg))}
    assert not any("norm" in n for n in names)


def test_untied_missing_lm_head_invalid():
    cfg = load("qwen3_8b_config.json")
    names = [n for n in dense_names(cfg) if n != "lm_head.weight"]
    with pytest.raises(ValueError, match="lm_head"):
        get_adapter(cfg).validate(names)


def test_unknown_tensor_detected():
    cfg = load("qwen3_0.6b_config.json")
    a = get_adapter(cfg)
    names = dense_names(cfg) + ["model.layers.0.mlp.mystery_proj.weight", "model.layers.0.mlp.up_proj.bias"]
    assert a.unclassified(names) == ["model.layers.0.mlp.mystery_proj.weight"]
    with pytest.raises(ValueError, match="mystery"):
        a.validate(names)


def test_layer_out_of_range_invalid():
    cfg = load("qwen3_0.6b_config.json")
    with pytest.raises(ValueError, match="layer"):
        get_adapter(cfg).validate(dense_names(cfg) + _dense_mlp(28))


def test_missing_projection_invalid():
    cfg = load("qwen3_0.6b_config.json")
    names = [n for n in dense_names(cfg) if n != "model.layers.5.self_attn.v_proj.weight"]
    with pytest.raises(ValueError, match="attn_v"):
        get_adapter(cfg).validate(names)


# --- MoE --------------------------------------------------------------------

def test_moe_small():
    cfg = small_moe()
    a, names = get_adapter(cfg), moe_names(cfg)
    assert (a.num_layers, a.is_moe, a.num_experts, a.family) == (4, True, 8, "qwen3_moe")
    specs = a.linears(names)
    routers = [s for s in specs if s.role == "router"]
    assert len(routers) == 4 and not any(s.eligible for s in routers)
    assert all(s.expert is None for s in routers)
    experts = [s for s in specs if s.role.startswith("expert_")]
    assert len(experts) == 4 * 8 * 3 and all(s.eligible for s in experts)
    e = next(s for s in experts if s.name == "model.layers.2.mlp.experts.5.down_proj.weight")
    assert (e.role, e.layer, e.expert) == ("expert_down", 2, 5)
    # within a layer: non-expert tensors first, then experts by index
    l1 = [s for s in specs if s.layer == 1]
    assert [s.role for s in l1[:5]] == ["attn_q", "attn_k", "attn_v", "attn_o", "router"]
    assert [s.expert for s in l1[5:]] == sorted(s.expert for s in l1[5:])
    assert a.unclassified(names) == []
    a.validate(names)


def test_moe_full_30b_a3b_counts():
    cfg = load("qwen3_30b_a3b_config.json")
    a, names = get_adapter(cfg), moe_names(cfg)
    assert len(names) == 18867  # matches the real model.safetensors.index.json weight_map
    specs = a.linears(names)
    assert sum(s.role.startswith("expert_") and s.eligible for s in specs) == 48 * 128 * 3
    assert sum(s.role == "router" for s in specs) == 48
    assert sum(s.eligible for s in specs) == 48 * (128 * 3 + 4)
    a.validate(names)


def test_moe_missing_expert_invalid():
    cfg = small_moe()
    names = [n for n in moe_names(cfg) if not n.startswith("model.layers.3.mlp.experts.7.")]
    with pytest.raises(ValueError, match="expert"):
        get_adapter(cfg).validate(names)


def test_moe_missing_single_expert_projection_invalid():
    cfg = small_moe()
    names = [n for n in moe_names(cfg) if n != "model.layers.1.mlp.experts.3.up_proj.weight"]
    with pytest.raises(ValueError):
        get_adapter(cfg).validate(names)


def test_moe_expert_index_out_of_range_invalid():
    cfg = small_moe()
    extra = [f"model.layers.0.mlp.experts.8.{x}_proj.weight" for x in ("gate", "up", "down")]
    with pytest.raises(ValueError, match="expert"):
        get_adapter(cfg).validate(moe_names(cfg) + extra)


@pytest.mark.parametrize("kw", [{"mlp_only_layers": [0, 2]}, {"decoder_sparse_step": 2}])
def test_moe_dense_layers_honored(kw):
    cfg = small_moe(**kw)
    a, names = get_adapter(cfg), moe_names(cfg)
    a.validate(names)
    dense_layers = {s.layer for s in a.linears(names) if s.role == "mlp_gate"}
    assert dense_layers == {i for i in range(4) if not is_sparse(cfg, i)} and dense_layers
    # a sparse checkpoint does not match a config declaring dense layers
    with pytest.raises(ValueError):
        a.validate(moe_names(small_moe()))
