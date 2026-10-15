"""Tests for choosing which tensors the Phase 1 analysis visits."""
import json
from pathlib import Path

from suc.analysis.select import select_tensors
from suc.config import AnalysisConfig
from suc.models.adapters import get_adapter

FIX = Path(__file__).parent / "fixtures"


def _dense_names(n_layers: int) -> list[str]:
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    for i in range(n_layers):
        p = f"model.layers.{i}"
        names += [f"{p}.self_attn.{x}_proj.weight" for x in "qkvo"]
        names += [f"{p}.self_attn.q_norm.weight", f"{p}.self_attn.k_norm.weight",
                  f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight"]
        names += [f"{p}.mlp.{x}_proj.weight" for x in ("gate", "up", "down")]
    return names


def _moe_config(layers: int = 4, experts: int = 8) -> dict:
    cfg = json.loads((FIX / "qwen3_30b_a3b_config.json").read_text())
    return {**cfg, "num_hidden_layers": layers, "num_experts": experts, "mlp_only_layers": [], "decoder_sparse_step": 1}


def _moe_names(layers: int, experts: int) -> list[str]:
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for i in range(layers):
        p = f"model.layers.{i}"
        names += [f"{p}.self_attn.{x}_proj.weight" for x in "qkvo"]
        names += [f"{p}.mlp.gate.weight"]
        names += [f"{p}.mlp.experts.{e}.{x}_proj.weight" for e in range(experts) for x in ("gate", "up", "down")]
    return names


def test_default_layers_are_first_middle_last_and_only_eligible():
    cfg = json.loads((FIX / "qwen3_0.6b_config.json").read_text())
    adapter = get_adapter(cfg)
    specs = select_tensors(adapter, _dense_names(cfg["num_hidden_layers"]), AnalysisConfig(), seed=0)
    n = cfg["num_hidden_layers"]
    assert sorted({s.layer for s in specs}) == [0, n // 2, n - 1]
    assert all(s.eligible for s in specs)
    assert len(specs) == 3 * 7  # q, k, v, o, gate, up, down per layer


def test_explicit_layers_and_roles():
    cfg = json.loads((FIX / "qwen3_0.6b_config.json").read_text())
    specs = select_tensors(get_adapter(cfg), _dense_names(cfg["num_hidden_layers"]),
                           AnalysisConfig(layers=(2,), roles=("mlp_down",)), seed=0)
    assert [s.name for s in specs] == ["model.layers.2.mlp.down_proj.weight"]


def test_moe_samples_experts_deterministically_and_skips_router():
    cfg = _moe_config()
    names = _moe_names(4, 8)
    a = AnalysisConfig(layers=(0, 3), experts_per_layer=2)
    s1 = select_tensors(get_adapter(cfg), names, a, seed=7)
    s2 = select_tensors(get_adapter(cfg), names, a, seed=7)
    assert s1 == s2
    assert all(s.role != "router" for s in s1)
    for layer in (0, 3):
        experts = {s.expert for s in s1 if s.layer == layer and s.expert is not None}
        assert len(experts) == 2
    assert sum(1 for s in s1 if s.expert is not None) == 2 * 2 * 3


def test_moe_expert_sample_varies_with_seed():
    cfg = _moe_config(layers=1, experts=64)
    names = _moe_names(1, 64)
    a = AnalysisConfig(layers=(0,), experts_per_layer=4)
    e1 = {s.expert for s in select_tensors(get_adapter(cfg), names, a, seed=1) if s.expert is not None}
    e2 = {s.expert for s in select_tensors(get_adapter(cfg), names, a, seed=2) if s.expert is not None}
    assert e1 != e2
