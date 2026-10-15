"""Choose which checkpoint tensors the Phase 1 analysis visits (spec §8, D7, D8)."""

from __future__ import annotations

import random
from collections.abc import Iterable

from suc.config import AnalysisConfig
from suc.models.adapters import ArchitectureAdapter, LinearSpec


def default_layers(num_layers: int) -> tuple[int, ...]:
    """First, middle and last transformer layer."""
    return tuple(sorted({0, num_layers // 2, num_layers - 1}))


def select_tensors(adapter: ArchitectureAdapter, tensor_names: Iterable[str], cfg: AnalysisConfig,
                   seed: int) -> list[LinearSpec]:
    """Eligible linear tensors in the chosen layers and roles; MoE experts sampled deterministically per layer."""
    layers = set(cfg.layers if cfg.layers is not None else default_layers(adapter.num_layers))
    specs = [s for s in adapter.linears(tensor_names)
             if s.eligible and s.layer in layers and s.role in cfg.roles]
    chosen: dict[int, set[int]] = {}
    for layer in layers:
        experts = sorted({s.expert for s in specs if s.layer == layer and s.expert is not None})
        k = min(cfg.experts_per_layer, len(experts))
        chosen[layer] = set(random.Random(seed * 100_003 + layer).sample(experts, k))
    return [s for s in specs if s.expert is None or s.expert in chosen[s.layer]]
