"""Residual-stream rotation folding (suc.models.rotation_folding): exactness on tiny dense / MoE Qwen3 models."""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from suc.io.safetensors_store import SafetensorsStore
from suc.models.adapters import get_adapter
from suc.models.rotation_folding import (FoldedStore, ResidualFold, RotatedFinalNorm, fold_model)
from suc.quant.hadamard import rht
from tests.unit.tiny_models import tiny_qwen3, tiny_qwen3_moe


def _cfg(model) -> dict:
    return {k: v for k, v in dataclasses.asdict(model.args).items()}


def _randomize_norms(model, seed=7) -> None:
    """Non-trivial RMSNorm weights (random init leaves them at 1, which would hide a missing fold)."""
    rng = np.random.default_rng(seed)
    upd = {k: mx.array(rng.uniform(0.3, 2.0, v.shape).astype(np.float32)).astype(v.dtype)
           for k, v in tree_flatten(model.parameters()) if k.endswith("norm.weight")}
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(list(upd.items())))
    mx.eval(model.parameters())


def _tokens(n=2, t=17, vocab=128, seed=0):
    return mx.random.randint(0, vocab, (n, t), key=mx.random.key(seed)).astype(mx.int32)


def _kl(a, b) -> float:
    """Mean next-token KL(a || b) in float64 (float32 KL of near-identical logits is dominated by rounding)."""
    la, lb = (np.array(x.astype(mx.float32)).astype(np.float64) for x in (a, b))
    la, lb = (x - x.max(-1, keepdims=True) for x in (la, lb))
    la, lb = (x - np.log(np.exp(x).sum(-1, keepdims=True)) for x in (la, lb))
    return float((np.exp(la) * (la - lb)).sum(-1).mean())


def _make(kind, tie, seed=0):
    m = tiny_qwen3(seed=seed, tie=tie) if kind == "dense" else tiny_qwen3_moe(seed=seed, tie=tie, experts=4)
    m.set_dtype(mx.float32)
    _randomize_norms(m)
    return m


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("tie", [True, False])
@pytest.mark.parametrize("v_heads", [False, True])
def test_folded_model_logits_match_in_float32(kind, tie, v_heads):
    ref = _make(kind, tie)
    model = _make(kind, tie)
    adapter = get_adapter(_cfg(model))
    fold = ResidualFold.for_adapter(adapter, seed=11, v_heads=v_heads)
    rep = fold_model(model, adapter, fold)
    assert rep["tied_head"] == tie
    assert isinstance(model.model.norm, RotatedFinalNorm) == tie
    toks = _tokens()
    a, b = ref(toks), model(toks)
    np.testing.assert_allclose(np.array(b), np.array(a), atol=2e-4, rtol=1e-4)
    assert _kl(a, b) < 1e-9
    # the weights really changed (not a no-op)
    w0 = dict(tree_flatten(ref.parameters()))["model.layers.0.self_attn.q_proj.weight"]
    w1 = dict(tree_flatten(model.parameters()))["model.layers.0.self_attn.q_proj.weight"]
    assert float(mx.max(mx.abs(w0 - w1))) > 1e-2


def test_folded_norms_are_ones_and_hidden_states_are_rotated():
    ref, model = _make("dense", False), _make("dense", False)
    adapter = get_adapter(_cfg(model))
    fold = ResidualFold.for_adapter(adapter, seed=3)
    fold_model(model, adapter, fold)
    p = dict(tree_flatten(model.parameters()))
    for name in ("model.layers.0.input_layernorm.weight", "model.layers.1.post_attention_layernorm.weight",
                 "model.norm.weight"):
        assert mx.array_equal(p[name], mx.ones_like(p[name]))
    # q_norm / k_norm are untouched
    q0 = dict(tree_flatten(ref.parameters()))["model.layers.0.self_attn.q_norm.weight"]
    assert mx.array_equal(p["model.layers.0.self_attn.q_norm.weight"], q0)
    # embedding output is the rotated embedding
    toks = _tokens(1, 5)
    np.testing.assert_allclose(np.array(model.model.embed_tokens(toks)),
                               np.array(rht(ref.model.embed_tokens(toks), 3)), atol=1e-5)


def test_moe_router_logits_unchanged():
    ref, model = _make("moe", True), _make("moe", True)
    adapter = get_adapter(_cfg(model))
    fold = ResidualFold.for_adapter(adapter, seed=5)
    fold_model(model, adapter, fold)
    x = mx.random.normal((3, 64), key=mx.random.key(1))
    blk_r, blk_f = ref.model.layers[0], model.model.layers[0]
    xn_r = blk_r.post_attention_layernorm(x)
    xn_f = blk_f.post_attention_layernorm(fold.rotate(x))
    np.testing.assert_allclose(np.array(blk_f.mlp.gate(xn_f)), np.array(blk_r.mlp.gate(xn_r)), atol=1e-4)


def _save(tmp_path, model) -> SafetensorsStore:
    weights = {}
    for k, v in tree_flatten(model.parameters()):
        if ".switch_mlp." in k:
            for e in range(v.shape[0]):
                weights[k.replace(".switch_mlp.", f".experts.{e}.")] = v[e]
        else:
            weights[k] = v
    d = tmp_path / "ckpt"
    d.mkdir()
    mx.save_safetensors(str(d / "model.safetensors"), weights)
    return SafetensorsStore.from_dir(d)


@pytest.mark.parametrize("kind,tie", [("dense", True), ("dense", False), ("moe", True)])
@pytest.mark.parametrize("v_heads", [False, True])
def test_folded_store_matches_fold_model_tensor_by_tensor(tmp_path, kind, tie, v_heads):
    src = _make(kind, tie)
    store = _save(tmp_path, src)
    model = _make(kind, tie)
    adapter = get_adapter(_cfg(model))
    fold = ResidualFold.for_adapter(adapter, seed=9, v_heads=v_heads)
    fs = FoldedStore(store, adapter, fold)
    fold_model(model, adapter, fold)
    p = dict(tree_flatten(model.parameters()))
    assert fs.names() == store.names()
    for name in store.names():
        w = fs.load(name)
        if ".experts." in name:
            stack, e = name.split(".experts.")
            e, proj = e.split(".", 1)
            want = p[f"{stack}.switch_mlp.{proj}"][int(e)]
        elif name == "model.norm.weight" and tie:
            want = p["model.norm.norm.weight"]       # kept, wrapped by RotatedFinalNorm
        else:
            want = p[name]
        np.testing.assert_allclose(np.array(w.astype(mx.float32)), np.array(want), atol=1e-5, err_msg=name)


def test_rotated_input_roles():
    fold = ResidualFold(64, 0)
    assert all(fold.rotated_input(r) for r in ("attn_q", "attn_k", "attn_v", "mlp_gate", "mlp_up", "expert_gate",
                                               "expert_up", "lm_head", "embed"))
    assert not any(fold.rotated_input(r) for r in ("attn_o", "mlp_down", "expert_down"))
    assert dataclasses.replace(fold, v_heads=True, head_dim=16).rotated_input("attn_o")


def test_unsupported_hidden_size_raises():
    with pytest.raises(ValueError, match="Hadamard"):
        ResidualFold.for_adapter(get_adapter({"model_type": "qwen3", "num_hidden_layers": 1, "hidden_size": 72,
                                              "num_attention_heads": 4}), seed=0)


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_fold_block_streaming_matches_fold_model(kind):
    from suc.models.rotation_folding import fold_block
    from suc.models.streaming import block_prefix, model_blocks, stream_blocks
    ref, model = _make(kind, True), _make(kind, True)
    adapter = get_adapter(_cfg(model))
    fold = ResidualFold.for_adapter(adapter, seed=2)
    toks = _tokens()
    want = fold.rotate(stream_blocks(ref, toks, release=False)["hidden"])
    model.model.embed_tokens.weight = fold.fold("embed", model.model.embed_tokens.weight)
    got = stream_blocks(model, toks, release=False,
                        on_block=lambda i, b: fold_block(b, block_prefix(i), adapter, fold))["hidden"]
    np.testing.assert_allclose(np.array(got), np.array(want), atol=2e-4)
    assert len(model_blocks(model)) == 2


@pytest.mark.parametrize("v_heads,rotate_out", [(False, True), (False, False), (True, False)])
def test_folded_codes_run_without_input_rotations(v_heads, rotate_out):
    """Folded tiny Qwen3 → q/k/v, gate/up (and o_proj with v_heads) encoded without V → SUC runtime: no input
    rotation launch for them, logits equal the decoded-weights model."""
    from suc.analysis.sensitivity import get_weight, set_weight
    from suc.kernels.suc_linear import SucLinear
    from suc.quant.codecs import VectorCodebook
    from suc.runtime.compressed_model import SucInputGroup, convert_to_compressed
    comp, ref = _make("dense", True), _make("dense", True)
    adapter = get_adapter(_cfg(comp))
    fold = ResidualFold.for_adapter(adapter, seed=4, v_heads=v_heads)
    for m in (comp, ref):
        fold_model(m, adapter, fold)
    names = {s.name: s.role for s in adapter.linears(k for k, _ in tree_flatten(comp.parameters())) if s.eligible}
    enc = {}
    for i, (name, role) in enumerate(names.items()):
        codec = VectorCodebook(16, 4, kmeans_iters=3, rotate_in=not fold.rotated_input(role), rotate_out=rotate_out)
        enc[name] = codec.encode(get_weight(comp, name), seed=i)
        set_weight(ref, name, codec.decode(enc[name]))
    convert_to_compressed(comp, enc)
    blk = comp.model.layers[0]
    assert isinstance(blk.self_attn["suc_qkv"], SucInputGroup) and blk.self_attn["suc_qkv"].parts[0].seed_in is None
    assert blk.mlp["suc_gate_up"].parts[0].seed_in is None
    assert blk.mlp.down_proj.seed_in is not None
    assert (blk.self_attn.o_proj.seed_in is None) == v_heads
    lins = [m for _, m in comp.named_modules() if isinstance(m, SucLinear)]
    assert all((m.seed_out is not None) == rotate_out for m in lins)
    toks = _tokens()
    a, b = ref(toks), comp(toks)
    assert _kl(a, b) < 1e-6


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_head_only_fold_is_exact_and_touches_only_v_and_o(tmp_path, kind):
    ref, model = _make(kind, True), _make(kind, True)
    adapter = get_adapter(_cfg(model))
    fold = ResidualFold.for_adapter(adapter, seed=6, v_heads=True, residual=False)
    rep = fold_model(model, adapter, fold)
    assert rep["folded"] and all(n.endswith(("v_proj.weight", "o_proj.weight")) for n in rep["folded"])
    assert not isinstance(model.model.norm, RotatedFinalNorm)
    toks = _tokens()
    assert _kl(ref(toks), model(toks)) < 1e-9
    assert fold.rotated_input("attn_o") and not fold.rotated_input("attn_q")
    fs = FoldedStore(_save(tmp_path, ref), adapter, fold)
    p = dict(tree_flatten(model.parameters()))
    for name in ("model.layers.0.self_attn.v_proj.weight", "model.layers.1.self_attn.o_proj.weight",
                 "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight"):
        np.testing.assert_allclose(np.array(fs.load(name).astype(mx.float32)), np.array(p[name]), atol=1e-5)


def test_nothing_to_fold_raises():
    with pytest.raises(ValueError, match="nothing to fold"):
        ResidualFold.for_adapter(get_adapter(_cfg(_make("dense", True))), seed=0, residual=False)
