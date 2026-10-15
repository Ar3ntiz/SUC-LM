"""Tests for Q6 representation fine-tuning (README §2, spec D3/D8)."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
import mlx.optimizers as optim

from suc.analysis.sensitivity import get_weight, logit_divergence
from suc.quant.codecs import AdditiveCodebook, VectorCodebook
from suc.quant.finetune import (
    FinetuneConfig, TrainableSucLinear, TrainableSucSwitchLinear, finetune_blockwise, finetune_e2e,
)
from suc.quant.residual import ResidualCodec
from tests.unit.tiny_models import tiny_qwen3, tiny_qwen3_moe


def _randn(shape, seed=0, scale=0.02) -> mx.array:
    return mx.array((scale * np.random.default_rng(seed).standard_normal(shape)).astype(np.float32))


def _np(x: mx.array) -> np.ndarray:
    return np.array(x.astype(mx.float32), dtype=np.float64)


def _tokens(seed=0, b=8, t=16, vocab=128) -> mx.array:
    return mx.array(np.random.default_rng(seed).integers(0, vocab, size=(b, t)).astype(np.int32))


W = _randn((64, 128), 1)
CODECS = {
    "vq_rot": VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=5),
    "vq_norot": VectorCodebook(16, block_size=8, rotate=False, kmeans_iters=5),
    "add2": AdditiveCodebook(16, block_size=8, num_codebooks=2, rotate=True, kmeans_iters=5),
    "vq_res_fp16": ResidualCodec(VectorCodebook(16, block_size=8, kmeans_iters=5), 0.05, "fp16"),
    "add2_res_int8": ResidualCodec(AdditiveCodebook(16, 8, num_codebooks=2, kmeans_iters=5), 0.05, "int8"),
    # column-structured residuals (Q5b): dense int8 / fp16 columns, with and without a CSR tail; one-sided G=2 VQ
    "vq_g2_1s_cols_int8": ResidualCodec(VectorCodebook(16, block_size=2, rotate=True, rotate_out=False, kmeans_iters=5),
                                        0.02, "int8", "magnitude", structure="columns"),
    "add2_cols_tail_fp16": ResidualCodec(AdditiveCodebook(16, 8, num_codebooks=2, kmeans_iters=5), 0.03, "fp16",
                                         "magnitude", structure="columns", column_share=0.5),
    "vq_cols_tail_int8": ResidualCodec(VectorCodebook(16, block_size=8, kmeans_iters=5), 0.03, "int8", "magnitude",
                                       structure="columns", column_share=0.6),
}


# ------------------------------------------------------------------ TrainableSucLinear
@pytest.mark.parametrize("name", list(CODECS))
def test_weight_equals_codec_decode(name) -> None:
    c = CODECS[name]
    e = c.encode(W, seed=3)
    lin = TrainableSucLinear.from_encoded(e)
    np.testing.assert_allclose(_np(lin.weight()), _np(c.decode(e)), rtol=0, atol=1e-6)
    x = _randn((5, 128), 2, 1.0)
    np.testing.assert_allclose(_np(lin(x)), _np(x @ c.decode(e).T), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", list(CODECS))
def test_gradients_flow_to_continuous_parts_only(name) -> None:
    e = CODECS[name].encode(W, seed=3)
    lin = TrainableSucLinear.from_encoded(e)
    keys = sorted(k for k, _ in tree_flatten(lin.trainable_parameters()))
    expect = ["codebooks", "row_scales"] + (["res_val"] if "res" in name or "tail" in name else []) \
        + (["res_colval"] if "cols" in name else [])
    assert keys == sorted(expect)  # indices / positions are not parameters at all
    x = _randn((7, 128), 4, 1.0)
    y = _randn((7, 64), 5, 1.0)
    _, g = nn.value_and_grad(lin, lambda xx: mx.sum((lin(xx) - y) ** 2))(x)
    for k, v in tree_flatten(g):
        assert float(mx.abs(v).sum()) > 0, k


@pytest.mark.parametrize("name", list(CODECS))
def test_to_encoded_round_trip(name) -> None:
    c = CODECS[name]
    e = c.encode(W, seed=3)
    lin = TrainableSucLinear.from_encoded(e)
    # perturb all continuous parts as an optimizer would
    lin.update({k: v * 1.01 + 0.003 for k, v in lin.trainable_parameters().items()})
    e2 = lin.to_encoded()
    assert e2.codec == e.codec and type(e2) is type(e)
    assert e2.bits() == e.bits()
    for k, v in e.arrays.items():
        assert e2.arrays[k].dtype == v.dtype and e2.arrays[k].shape == v.shape, k
    np.testing.assert_allclose(_np(c.decode(e2)), _np(lin.weight()), rtol=0, atol=1e-6)
    assert not np.allclose(_np(c.decode(e2)), _np(c.decode(e)))


def test_reassign_recovers_nearest_codes() -> None:
    c = VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=5)
    e = c.encode(W, seed=3)
    lin = TrainableSucLinear.from_encoded(e)
    good = _np(lin.weight())
    rng = np.random.default_rng(0)
    lin._idx = mx.array(rng.integers(0, 16, size=lin._idx.shape).astype(np.int32))
    bad = _np(lin.weight())
    changed = lin.reassign(W)
    assert changed > 0.5
    err = lambda a: float(((a - _np(W)) ** 2).sum())  # noqa: E731
    assert err(_np(lin.weight())) < err(bad)
    np.testing.assert_allclose(_np(lin.weight()), good, atol=1e-6)  # k-means codes were already nearest


# ------------------------------------------------------------------ whole-model helpers
def _linear_names(model) -> list[str]:
    return [k for k, _ in tree_flatten(model.parameters())
            if k.startswith("model.layers.") and k.endswith("_proj.weight") and ".experts." not in k
            and ".switch_mlp." not in k and ".mlp.gate." not in k]


def _encode_all(model, codec, names) -> dict:
    return {n: codec.encode(get_weight(model, n), seed=i) for i, n in enumerate(names)}


def _setup(seed=0, codec=None):
    model = tiny_qwen3(seed)
    names = _linear_names(model)
    ref = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
    codec = codec or VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=5)
    enc = _encode_all(model, codec, names)
    return model, names, ref, codec, enc


def _set_decoded(model, codec, enc) -> None:
    from suc.analysis.sensitivity import set_weight
    for n, e in enc.items():
        set_weight(model, n, codec.decode(e))


def _kl(model, ref_logits, toks) -> float:
    return logit_divergence(ref_logits, model(toks))["kl_mean"]


CFG = FinetuneConfig(block_steps=40, lr=3e-3, micro_batch=4, sequences=8)


def test_blockwise_lowers_block_error_and_kl() -> None:
    model, names, ref, codec, enc = _setup()
    calib, held = _tokens(0), _tokens(1)
    ref_logits = model(held)
    _set_decoded(model, codec, enc)
    kl_before = _kl(model, ref_logits, held)
    tuned, log = finetune_blockwise(model, enc, calib, CFG, reference_weights=lambda n: ref[n])
    assert set(tuned) == set(enc)
    assert len(log["blocks"]) == 2
    for b in log["blocks"].values():
        assert b["loss_after"] < b["loss_before"], b
        assert b["seconds"] >= 0
    # the model now holds the decode of the tuned encodings
    for n in names:
        np.testing.assert_allclose(_np(get_weight(model, n)), _np(codec.decode(tuned[n])), atol=1e-6)
    assert _kl(model, ref_logits, held) < kl_before
    for n, e in tuned.items():
        assert e.bits() == enc[n].bits()
        assert np.array_equal(np.array(e.arrays["indices"]), np.array(enc[n].arrays["indices"]))


def test_non_representation_parameters_unchanged() -> None:
    model, names, ref, codec, enc = _setup(1)
    finetune_blockwise(model, enc, _tokens(0), CFG, reference_weights=lambda n: ref[n])
    for k, v in tree_flatten(model.parameters()):
        if k not in names:
            assert np.array_equal(np.array(v), np.array(ref[k])), k


def test_blockwise_with_residual_and_tune_subset() -> None:
    codec = ResidualCodec(AdditiveCodebook(16, 8, num_codebooks=2, kmeans_iters=5), 0.02, "int8")
    model, names, ref, codec, enc = _setup(2, codec)
    cfg = FinetuneConfig(block_steps=20, lr=3e-3, micro_batch=4, sequences=8, tune=("codebooks",))
    tuned, log = finetune_blockwise(model, enc, _tokens(0), cfg, reference_weights=lambda n: ref[n])
    for n, e in tuned.items():
        for k in ("base.row_scales", "res_val", "res_scale", "res_pos", "base.indices"):
            assert np.array_equal(np.array(e.arrays[k]), np.array(enc[n].arrays[k])), (n, k)
        assert not np.array_equal(np.array(e.arrays["base.codebooks"]), np.array(enc[n].arrays["base.codebooks"]))
    for b in log["blocks"].values():
        assert b["loss_after"] < b["loss_before"]


@pytest.mark.parametrize("share", [1.0, 0.5])
def test_blockwise_with_column_residual(share) -> None:
    """Column residual values (and the CSR tail) are tuned with the STE to their stored precision; columns,
    positions, indices and bits stay fixed; the model holds the decode of the tuned encodings."""
    codec = ResidualCodec(VectorCodebook(16, block_size=2, rotate=True, rotate_out=False, kmeans_iters=5), 0.03,
                          "int8", "magnitude", structure="columns", column_share=share)
    model, names, ref, codec, enc = _setup(6, codec)
    assert all(e.params["columns"] > 0 for e in enc.values())
    cfg = FinetuneConfig(block_steps=20, lr=3e-3, micro_batch=4, sequences=8, tune=("residual",))
    tuned, log = finetune_blockwise(model, enc, _tokens(0), cfg, reference_weights=lambda n: ref[n])
    for b in log["blocks"].values():
        assert b["loss_after"] < b["loss_before"]
    changed = 0
    for n, e in tuned.items():
        assert e.bits() == enc[n].bits() and e.params == enc[n].params
        for k, v in enc[n].arrays.items():
            assert e.arrays[k].dtype == v.dtype and e.arrays[k].shape == v.shape, (n, k)
        for k in ("res_cols", "res_pos", "base.indices", "base.codebooks", "base.row_scales"):
            assert np.array_equal(np.array(e.arrays[k]), np.array(enc[n].arrays[k])), (n, k)
        changed += not np.array_equal(np.array(e.arrays["res_colval"]), np.array(enc[n].arrays["res_colval"]))
        np.testing.assert_allclose(_np(get_weight(model, n)), _np(codec.decode(e)), atol=1e-6)
    assert changed > 0


def test_reassign_does_not_increase_block_loss() -> None:
    model, names, ref, codec, enc = _setup(3)
    cfg = FinetuneConfig(block_steps=20, lr=3e-3, micro_batch=4, sequences=8, reassign=True)
    _, log = finetune_blockwise(model, enc, _tokens(0), cfg, reference_weights=lambda n: ref[n])
    for b in log["blocks"].values():
        assert "reassign" in b
        assert b["loss_after"] <= b["loss_after_tune"] + 1e-12
        assert b["loss_after"] < b["loss_before"]


def test_determinism() -> None:
    outs = []
    for _ in range(2):
        model, names, ref, codec, enc = _setup(4)
        tuned, log = finetune_blockwise(model, enc, _tokens(0), CFG, reference_weights=lambda n: ref[n])
        outs.append((tuned, [b["loss_after"] for b in log["blocks"].values()]))
    assert outs[0][1] == outs[1][1]
    for n in outs[0][0]:
        for k, v in outs[0][0][n].arrays.items():
            assert np.array_equal(np.array(v), np.array(outs[1][0][n].arrays[k])), (n, k)


def test_e2e_lowers_kl() -> None:
    model, names, ref, codec, enc = _setup(5)
    teacher = tiny_qwen3(5)
    calib, held = _tokens(0), _tokens(1)
    ref_logits = teacher(held)
    _set_decoded(model, codec, enc)
    kl_before = _kl(model, ref_logits, held)
    cfg = FinetuneConfig(e2e_steps=40, lr=3e-3, micro_batch=4, sequences=8)
    tuned, log = finetune_e2e(model, enc, calib, lambda t: teacher(t), cfg)
    assert log["kl_after"] < log["kl_before"]
    assert _kl(model, ref_logits, held) < kl_before
    for n in names:
        np.testing.assert_allclose(_np(get_weight(model, n)), _np(codec.decode(tuned[n])), atol=1e-6)
    for k, v in tree_flatten(model.parameters()):
        if k not in names:
            assert np.array_equal(np.array(v), np.array(ref[k])), k


# ------------------------------------------------------------------ MoE: expert stacks tuned
EXPERT_PROJ = ("gate_proj", "up_proj", "down_proj")


def _expert_names(layers=(0, 1), n_exp=8, proj=EXPERT_PROJ):
    return [f"model.layers.{i}.mlp.experts.{e}.{p}.weight" for i in layers for e in range(n_exp) for p in proj]


def _moe_setup(seed=0, codec=None):
    model = tiny_qwen3_moe(seed, tie=False)
    ref = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
    codec = codec or VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=5)
    names = _linear_names(model)
    enc = {n: codec.encode(_ref_weight(ref, n), seed=i) for i, n in enumerate(names + _expert_names())}
    return model, names, ref, codec, enc


def _switch_from(codec, n_exp=6, out=32, inn=64, seed=0):
    from mlx_lm.models.switch_layers import SwitchLinear
    encs = [codec.encode(_randn((out, inn), seed + e, 0.1), seed=e) for e in range(n_exp)]
    lin = SwitchLinear(inn, out, n_exp, bias=False)
    lin.weight = mx.stack([codec.decode(e) for e in encs])
    t = TrainableSucSwitchLinear(encs, [f"x.experts.{e}.p.weight" for e in range(n_exp)])
    return encs, lin, t


@pytest.mark.parametrize("name", ["vq_rot", "add2_res_int8"])
def test_trainable_switch_matches_switch_linear(name) -> None:
    _, lin, t = _switch_from(CODECS[name])
    x = mx.expand_dims(_randn((2, 5, 64), 3, 1.0), (-2, -3))
    inds = mx.array(np.random.default_rng(0).integers(0, 6, size=(2, 5, 2)).astype(np.uint32))
    np.testing.assert_allclose(_np(t(x, inds)), _np(lin(x, inds)), rtol=1e-5, atol=1e-5)


def test_only_routed_experts_get_gradients_and_change() -> None:
    _, _, t = _switch_from(CODECS["add2_res_int8"])
    t0 = {k: np.array(v) for k, v in tree_flatten(t.parameters())}
    x = mx.expand_dims(_randn((16, 64), 4, 1.0), (-2, -3))
    inds = mx.array(np.random.default_rng(1).choice([0, 2, 5], size=(16, 2)).astype(np.uint32))
    target = _randn((16, 2, 1, 32), 5, 1.0)

    def loss_fn(x, y):
        return mx.mean((t(x, inds) - y) ** 2)
    vg = nn.value_and_grad(t, loss_fn)
    _, g = vg(x, target)
    g = dict(tree_flatten(g))
    for e in range(6):
        nz = any(float(mx.max(mx.abs(v))) > 0 for k, v in g.items() if k.startswith(f"experts.{e}."))
        assert nz == (e in (0, 2, 5)), e
    assert sorted(k for k in g if k.startswith("experts.0.")) == ["experts.0.codebooks", "experts.0.res_val",
                                                                    "experts.0.row_scales"]
    opt = optim.Adam(learning_rate=1e-2, bias_correction=True)
    for _ in range(3):
        _, g = vg(x, target)
        opt.update(t, g)
        mx.eval(t.parameters(), opt.state)
    for k, v in tree_flatten(t.parameters()):
        e = int(k.split(".")[1])
        assert np.array_equal(np.array(v), t0[k]) == (e not in (0, 2, 5)), k


def test_moe_blockwise_tunes_experts_and_attention() -> None:
    model, names, ref, codec, enc = _moe_setup(0)
    experts = _expert_names()
    trainable = sorted(k for k, _ in tree_flatten(model.trainable_parameters()))
    calib, held = _tokens(0), _tokens(1)
    ref_logits = model(held)
    _set_decoded_moe(model, codec, enc)
    kl_before = _kl(model, ref_logits, held)
    tuned, log = finetune_blockwise(model, enc, calib, CFG, reference_weights=lambda n: _ref_weight(ref, n))
    assert set(tuned) == set(enc) and log["skipped"] == []
    for b in log["blocks"].values():
        assert b["expert_tensors"] == 24 and b["tensors"] == 24 + 4
        assert b["loss_after"] < b["loss_before"], b
    assert _kl(model, ref_logits, held) < kl_before
    changed = 0
    for n in names + experts:
        np.testing.assert_allclose(_np(get_weight(model, n)), _np(codec.decode(tuned[n])), atol=1e-6)
        assert tuned[n].bits() == enc[n].bits()
        assert np.array_equal(np.array(tuned[n].arrays["indices"]), np.array(enc[n].arrays["indices"]))
        changed += not np.array_equal(np.array(tuned[n].arrays["codebooks"]), np.array(enc[n].arrays["codebooks"]))
    assert changed > len(experts) // 2
    stacks = {f"model.layers.{i}.mlp.switch_mlp.{p}.weight" for i in range(2) for p in EXPERT_PROJ}
    for k, v in tree_flatten(model.parameters()):      # router, norms, embedding, head untouched
        if k not in names and k not in stacks:
            assert np.array_equal(np.array(v), np.array(ref[k])), k
    assert type(model.model.layers[0].mlp.switch_mlp.gate_proj).__name__ == "SwitchLinear"
    assert sorted(k for k, _ in tree_flatten(model.trainable_parameters())) == trainable


def test_moe_blockwise_deterministic_with_residual() -> None:
    codec = ResidualCodec(AdditiveCodebook(16, 8, num_codebooks=2, kmeans_iters=5), 0.02, "int8")
    outs = []
    for _ in range(2):
        model, names, ref, codec, enc = _moe_setup(1, codec)
        cfg = FinetuneConfig(block_steps=10, lr=3e-3, micro_batch=4, sequences=8, reassign=True)
        tuned, log = finetune_blockwise(model, enc, _tokens(0), cfg, reference_weights=lambda n: _ref_weight(ref, n))
        outs.append((tuned, [b["loss_after"] for b in log["blocks"].values()]))
        for b in log["blocks"].values():
            assert b["loss_after"] < b["loss_before"] and "reassign" in b
    assert outs[0][1] == outs[1][1]
    for n in outs[0][0]:
        for k, v in outs[0][0][n].arrays.items():
            assert np.array_equal(np.array(v), np.array(outs[1][0][n].arrays[k])), (n, k)


def test_moe_incomplete_expert_stack_is_decoded_untuned() -> None:
    model, names, ref, codec, enc = _moe_setup(2)
    partial = [n for n in _expert_names(layers=(0,), proj=("up_proj",)) if ".experts.7." not in n]
    enc = {n: e for n, e in enc.items() if ".experts." not in n or n in partial}
    tuned, log = finetune_blockwise(model, enc, _tokens(0), CFG, reference_weights=lambda n: _ref_weight(ref, n))
    assert sorted(s["name"] for s in log["skipped"]) == sorted(partial)
    for n in partial:
        assert tuned[n] is enc[n]
        np.testing.assert_allclose(_np(get_weight(model, n)), _np(codec.decode(enc[n])), atol=1e-6)


def test_moe_e2e_tunes_experts() -> None:
    model, names, ref, codec, enc = _moe_setup(3)
    teacher = tiny_qwen3_moe(3, tie=False)
    _set_decoded_moe(model, codec, enc)
    cfg = FinetuneConfig(e2e_steps=20, lr=3e-3, micro_batch=4, sequences=8)
    tuned, log = finetune_e2e(model, enc, _tokens(0), lambda t: teacher(t), cfg)
    assert log["skipped"] == [] and log["kl_after"] < log["kl_before"]
    for n in _expert_names():
        np.testing.assert_allclose(_np(get_weight(model, n)), _np(codec.decode(tuned[n])), atol=1e-6)


def _set_decoded_moe(model, codec, enc) -> None:
    from suc.quant.finetune import _set_all
    _set_all(model, {n: codec.decode(e) for n, e in enc.items()})


def _ref_weight(ref: dict, name: str) -> mx.array:
    import re
    m = re.match(r"^(.*)\.experts\.(\d+)\.(\w+)\.weight$", name)
    if m:
        return ref[f"{m.group(1)}.switch_mlp.{m.group(3)}.weight"][int(m.group(2))]
    return ref[name]


def test_fixed_encodings_are_installed_untuned_and_returned_unchanged() -> None:
    """Mixed plans (suc compress): non-codebook encodings (e.g. affine) of a block are decoded by the caller's
    fixed_weights for the student, never tuned, and returned unchanged; the teacher still uses the reference."""
    from suc.quant.codecs import AffineRTN

    model, names, ref, codec, enc = _setup(5)
    affine = AffineRTN(4, 64)
    fixed_names = [n for n in names if n.endswith("o_proj.weight")]
    for n in fixed_names:
        enc[n] = affine.encode(ref[n], seed=0)
    with pytest.raises(ValueError, match="fixed_weights"):
        finetune_blockwise(tiny_qwen3(5), enc, _tokens(0), CFG, reference_weights=lambda n: ref[n])
    calls = []
    tuned, log = finetune_blockwise(model, enc, _tokens(0), CFG, reference_weights=lambda n: ref[n],
                                    fixed_weights=lambda n: affine.decode(enc[n]),
                                    after_block=lambda i, b: calls.append(i))
    assert calls == [0, 1]
    for n in fixed_names:
        assert tuned[n] is enc[n]
        np.testing.assert_allclose(_np(get_weight(model, n)), _np(affine.decode(enc[n])), atol=1e-6)
    assert all(b["fixed"] == len(fixed_names) // 2 for b in log["blocks"].values())
    for b in log["blocks"].values():
        assert b["loss_after"] < b["loss_before"]


def test_blockwise_streamed_encodings_match_in_memory() -> None:
    """D8: encodings read per block (callable) + results delivered per block give exactly the in-memory result,
    and each block's encodings are requested once, only for that block."""
    model, names, ref, codec, enc = _setup(2)
    calib = _tokens(0)
    tuned, log = finetune_blockwise(model, enc, calib, CFG, reference_weights=lambda n: ref[n])

    model2, _, _, _, _ = _setup(2)
    asked, got = [], {}

    def block_enc(i):
        asked.append(i)
        return {n: e for n, e in enc.items() if n.startswith(f"model.layers.{i}.")}

    def sink(i, t):
        assert all(n.startswith(f"model.layers.{i}.") for n in t)
        got.update(t)

    out, log2 = finetune_blockwise(model2, block_enc, calib, CFG, reference_weights=lambda n: ref[n],
                                   on_block_tuned=sink)
    assert out == {} and asked == [0, 1] and set(got) == set(tuned)
    for n in tuned:
        for k in tuned[n].arrays:
            assert np.array_equal(np.array(tuned[n].arrays[k]), np.array(got[n].arrays[k])), (n, k)
    assert [b["loss_after"] for b in log["blocks"].values()] == [b["loss_after"] for b in log2["blocks"].values()]
