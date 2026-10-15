"""Tests for suc.analysis.sensitivity (spec §9 T2.3/T2.4, D5)."""
import math

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_moe

from suc.analysis.sensitivity import (
    get_weight, logit_divergence, measure_tensor_sensitivity, perturb_weight, proxy_output_error,
    proxy_output_error_full, sensitivity_score, set_weight, spearman, weight_error,
)
from tiny_models import tiny_qwen3


def tiny_moe(seed: int = 0) -> qwen3_moe.Model:
    mx.random.seed(seed)
    args = qwen3_moe.ModelArgs(
        model_type="qwen3_moe", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
        mlp_only_layers=[], moe_intermediate_size=32, rms_norm_eps=1e-6, vocab_size=128,
        num_key_value_heads=2, head_dim=16, rope_theta=10000.0, tie_word_embeddings=False,
        max_position_embeddings=512, norm_topk_prob=True,
    )
    model = qwen3_moe.Model(args)
    mx.eval(model.parameters())
    return model


def snapshot(model) -> dict:
    return {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())}


def assert_same(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def tokens(seed: int = 0, b: int = 2, t: int = 16, vocab: int = 128) -> mx.array:
    return mx.array(np.random.default_rng(seed).integers(0, vocab, size=(b, t)).astype(np.int32))


# ---------------------------------------------------------------- perturbations
def test_gaussian_rel_mse_is_level_squared():
    w = mx.random.normal((256, 512), key=mx.random.key(1)) * mx.linspace(0.1, 3.0, 256)[:, None]
    for level in (0.05, 0.3):
        w_hat = perturb_weight(w, "gaussian", seed=3, level=level)
        assert w_hat.shape == w.shape and w_hat.dtype == w.dtype
        assert weight_error(w, w_hat)["rel_mse"] == pytest.approx(level ** 2, rel=0.02)
    assert np.array_equal(np.array(perturb_weight(w, "gaussian", seed=3, level=0.1)),
                          np.array(perturb_weight(w, "gaussian", seed=3, level=0.1)))
    assert weight_error(w, perturb_weight(w, "gaussian", seed=3, level=0.0))["rel_mse"] == 0.0


def test_vq_error_decreases_with_k_and_keeps_shape_dtype():
    w = (mx.random.normal((128, 256), key=mx.random.key(2)) * 0.02).astype(mx.bfloat16)
    errs = []
    for k in (4, 32, 256):
        w_hat = perturb_weight(w, "vq", seed=0, block_size=4, k=k)
        assert w_hat.shape == w.shape and w_hat.dtype == mx.bfloat16
        errs.append(weight_error(w, w_hat)["rel_mse"])
    assert errs[0] > errs[1] > errs[2]


def test_vq_rotate_on_outlier_heavy_matrix():
    rng = np.random.default_rng(0)
    a = rng.standard_t(df=2, size=(128, 256)).astype(np.float32)  # heavy tails
    w = mx.array(a)
    for rotate in (True, False):
        e = weight_error(w, perturb_weight(w, "vq", seed=1, block_size=8, k=64, rotate=rotate,
                                           fit_blocks=2048))["rel_mse"]
        assert math.isfinite(e) and 0 < e < 1


def test_weight_error_values():
    w = mx.array([[1.0, 2.0], [3.0, 4.0]])
    e = weight_error(w, w + mx.array([[1.0, 0.0], [0.0, 1.0]]))
    assert e["mse"] == pytest.approx(0.5)
    assert e["rel_mse"] == pytest.approx(2.0 / 30.0)


# ---------------------------------------------------------------- proxies
def test_proxy_full_equals_exact_mean_output_error():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 32)) @ rng.standard_normal((32, 32))  # correlated columns
    dw = rng.standard_normal((16, 32))
    exact = np.mean(np.sum((x @ dw.T) ** 2, axis=1))
    m2 = mx.array((x.T @ x / x.shape[0]).astype(np.float32))
    got = proxy_output_error_full(mx.array(dw.astype(np.float32)), m2)
    assert got == pytest.approx(exact, rel=1e-4)


def test_proxy_diag_matches_full_for_independent_columns():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((20000, 32)) * rng.uniform(0.1, 3.0, 32)).astype(np.float32)
    dw = mx.array(rng.standard_normal((16, 32)).astype(np.float32))
    m2 = x.T @ x / x.shape[0]
    full = proxy_output_error_full(dw, mx.array(m2))
    diag = proxy_output_error(dw, mx.array(np.diag(m2).copy()))
    assert diag == pytest.approx(full, rel=0.05)


# ---------------------------------------------------------------- divergence
def test_logit_divergence_zero_and_positive():
    ref = mx.random.normal((2, 5, 11), key=mx.random.key(0))
    tg = mx.zeros((2, 5), dtype=mx.int32)
    same = logit_divergence(ref, ref, tg)
    assert abs(same["kl_mean"]) < 1e-6 and same["top1_agreement"] == 1.0 and abs(same["delta_ce"]) < 1e-6
    other = logit_divergence(ref, ref + mx.random.normal(ref.shape, key=mx.random.key(1)))
    assert other["kl_mean"] > 0 and other["kl_p95"] >= 0 and "ce_ref" not in other


def test_logit_divergence_delta_ce_sign():
    v = 6
    tg = mx.array([[2, 4]], dtype=mx.int32)
    ref = mx.zeros((1, 2, v))
    better = ref + 5.0 * mx.eye(v)[tg]  # boost the target token
    worse = ref - 5.0 * mx.eye(v)[tg]
    assert logit_divergence(ref, better, tg)["delta_ce"] < 0
    d = logit_divergence(ref, worse, tg)
    assert d["delta_ce"] > 0 and d["ce_test"] > d["ce_ref"]
    assert d["ce_ref"] == pytest.approx(math.log(v), rel=1e-5)


# ---------------------------------------------------------------- get / set
def test_get_set_dense_roundtrip():
    m = tiny_qwen3(0)
    name = "model.layers.1.mlp.down_proj.weight"
    w = get_weight(m, name)
    assert w is m.model.layers[1].mlp.down_proj.weight
    set_weight(m, name, w * 2)
    assert np.allclose(np.array(m.model.layers[1].mlp.down_proj.weight), np.array(w) * 2)
    with pytest.raises((KeyError, ValueError)):
        set_weight(m, name, mx.zeros((3, 3)))
    with pytest.raises(KeyError):
        get_weight(m, "model.layers.1.mlp.nope.weight")


def test_get_set_moe_expert_row_only():
    m = tiny_moe(0)
    before = snapshot(m)
    name = "model.layers.0.mlp.experts.2.up_proj.weight"
    w = get_weight(m, name)
    stack = m.model.layers[0].mlp.switch_mlp.up_proj.weight
    assert w.shape == stack.shape[1:]
    assert np.array_equal(np.array(w), np.array(stack[2]))
    set_weight(m, name, mx.ones_like(w))
    after = snapshot(m)
    key = "model.layers.0.mlp.switch_mlp.up_proj.weight"
    for e in (0, 1, 3):
        assert np.array_equal(after[key][e], before[key][e])
    assert np.all(after[key][2] == 1.0)
    for k in before:
        if k != key:
            assert np.array_equal(before[k], after[k]), k


# ---------------------------------------------------------------- measurement
@pytest.mark.parametrize("builder,name", [
    (tiny_qwen3, "model.layers.0.self_attn.q_proj.weight"),
    (tiny_moe, "model.layers.1.mlp.experts.1.down_proj.weight"),
])
def test_measure_restores_and_zero_perturbation(builder, name):
    m = builder(0)
    toks = tokens()
    ref = m(toks)
    before = snapshot(m)
    w = get_weight(m, name)
    r0 = measure_tensor_sensitivity(m, name, w, toks, ref)
    assert abs(r0["kl_mean"]) < 1e-5 and abs(r0["delta_ce"]) < 1e-5 and r0["rel_mse"] == 0.0
    assert r0["layer_out_rel_mse"] == pytest.approx(0.0, abs=1e-12)
    r1 = measure_tensor_sensitivity(m, name, perturb_weight(w, "gaussian", seed=0, level=0.5), toks, ref)
    assert r1["kl_mean"] > 0 and r1["layer_out_rel_mse"] > 0
    assert r1["layer_out_rel_mse"] == pytest.approx(0.25, rel=0.6)  # ~ level^2 for isotropic-ish inputs
    assert_same(before, snapshot(m))
    assert np.array_equal(np.array(m(toks)), np.array(ref))


def test_measure_restores_on_exception():
    m = tiny_moe(0)
    toks = tokens()
    ref = m(toks)
    before = snapshot(m)
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    w_hat = perturb_weight(get_weight(m, name), "gaussian", seed=0, level=1.0)
    with pytest.raises(Exception):
        measure_tensor_sensitivity(m, name, w_hat, toks[:, :, None, None], ref)  # bad shape
    assert_same(before, snapshot(m))
    assert type(m.model.layers[0].mlp.switch_mlp.gate_proj).__name__ == "SwitchLinear"


def test_kl_monotone_in_gaussian_level():
    m = tiny_qwen3(1)
    toks = tokens(1, b=4, t=32)
    ref = m(toks)
    names = [f"model.layers.{i}.{p}.weight" for i in (0, 1)
             for p in ("self_attn.v_proj", "mlp.down_proj")]
    kls = []
    for level in (0.01, 0.1, 0.5):
        kls.append(np.mean([measure_tensor_sensitivity(
            m, n, perturb_weight(get_weight(m, n), "gaussian", seed=7, level=level), toks, ref,
            layer_output=False)["kl_mean"] for n in names]))
    assert kls[0] < kls[1] < kls[2]


# ---------------------------------------------------------------- stats helpers
def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # ties: ranks x = [1.5, 1.5, 3, 4], y = [1, 2, 3, 4] -> pearson of ranks
    assert spearman([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832980505138)
    assert spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_sensitivity_score():
    assert sensitivity_score(0.2, 0.1) == pytest.approx(2.0)
    assert math.isfinite(sensitivity_score(0.2, 0.0))
