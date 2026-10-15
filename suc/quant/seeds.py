"""Encode-seed and rotation-seed derivation shared by candidate scoring (suc.allocation.greedy.build_candidates) and
final encoding (suc.evaluation.reconstruct.reconstruct_model / suc compress), audit F08: the allocator must score
exactly the rotated encoding that is written.

  encode seed      seed + crc32(tensor name)                       (codec-internal seeds: k-means, seed_in / seed_out)
  rotation groups  q/k/v of a block and gate/up of a dense MLP read the same input; with rotation_groups the codec
                   gets ONE input rotation per group: seed_in = rotation_seeds(name, seed) (runtime fusion, K5)
  expert stacks    tcq codecs with codebook scope "expert_stack" get the stack's seed_in (suc.quant.stacks)
"""
from __future__ import annotations

import inspect
import re
import zlib

import numpy as np

from suc.quant.stacks import shares_rotation, stack_of, stack_seeds

# Tensors that read the same input (README §5): q/k/v of a block (input_layernorm output) and gate/up of a dense
# MLP (post_attention_layernorm output). MoE expert / shared-expert rows are not grouped.
_ROT_GROUP = re.compile(r"^(.*\.layers\.\d+)\.(?:(self_attn)\.[qkv]_proj|(mlp)\.(?:gate|up)_proj)\.weight$")

# Version of the scoring / encoding seed contract (stage fingerprints of runs that use rotation groups):
# 1 = candidates scored with the per-tensor seed_in while encode used the group seed_in (before audit F08);
# 2 = both use rotation_kwargs.
SEED_CONTRACT = 2


def encode_seed(name: str, seed: int) -> int:
    """Per-tensor encode seed: depends only on the name (not on the order / subset of tensors)."""
    return seed + zlib.crc32(name.encode())


def rotation_group(name: str) -> str | None:
    """Rotation group key of a checkpoint tensor ("model.layers.3.self_attn.qkv", "model.layers.3.mlp.gate_up")
    or None when the tensor shares its input with no other eligible tensor."""
    if (m := _ROT_GROUP.match(name)) is None:
        return None
    return f"{m.group(1)}.self_attn.qkv" if m.group(2) else f"{m.group(1)}.mlp.gate_up"


def rotation_seeds(name: str, seed: int) -> dict[str, int]:
    """{"seed_in": s} shared by every tensor of name's rotation group (s derived from seed + crc32(group key)
    only, so it does not depend on the order or subset of tensors), or {} for ungrouped tensors. seed_out stays
    per tensor (derived from the tensor's own encode seed)."""
    g = rotation_group(name)
    if g is None:
        return {}
    return {"seed_in": int(np.random.default_rng(seed + zlib.crc32(g.encode())).integers(0, 2 ** 31 - 1))}


def accepts_rotation_seeds(codec) -> bool:
    """True when codec.encode takes seed_in= (VectorCodebook / AdditiveCodebook / TrellisCodec / ResidualCodec)."""
    ps = inspect.signature(codec.encode).parameters
    return "seed_in" in ps or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ps.values())


def rotation_kwargs(codec, name: str, seed: int, rotation_groups: bool) -> dict:
    """The rotation-seed keyword arguments codec.encode gets for tensor `name` (scoring and encoding alike): the
    group seed_in with rotation_groups (codecs that accept it), then the expert stack's seed_in for tcq codecs that
    share it (shares_rotation)."""
    kw = {}
    if rotation_groups and accepts_rotation_seeds(codec):
        kw.update(rotation_seeds(name, seed))
    if (st := stack_of(name)) is not None and shares_rotation(codec):
        kw["seed_in"] = stack_seeds(st[0], seed)["seed_in"]
    return kw
