"""Tests for suc.analysis.fisher (empirical diagonal Fisher, spec §6 / D5)."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_moe

from suc.analysis.fisher import fisher_diag, fisher_proxy, fisher_summary
from suc.analysis.sensitivity import get_weight, measure_tensor_sensitivity, perturb_weight, spearman
from tiny_models import tiny_qwen3

DENSE = [f"model.layers.{i}.{m}.weight" for i in range(2)
         for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                   "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]


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


def tokens(seed: int = 0, b: int = 4, t: int = 16, vocab: int = 128) -> mx.array:
    return mx.array(np.random.default_rng(seed).integers(0, vocab, size=(b, t)).astype(np.int32))


def trainable_keys(model) -> list[str]:
    return sorted(k for k, _ in tree_flatten(model.trainable_parameters()))


def ce_grad_sq(model, toks: mx.array, name: str) -> np.ndarray:
    """Independent reference: squared gradient of the mean next-token CE over all of toks."""
    holder_path, key = name.rsplit(".", 1)
    model.freeze()
    holder = model
    for p in holder_path.split("."):
        holder = holder[int(p)] if isinstance(holder, list) else holder[p]
    holder.unfreeze(keys=[key], recurse=False)

    def loss():
        logits = model(toks[:, :-1]).astype(mx.float32)
        return nn.losses.cross_entropy(logits, toks[:, 1:], reduction="mean")

    _, g = nn.value_and_grad(model, loss)()
    model.unfreeze()
    (_, gw), = tree_flatten(g)
    return np.array(gw.astype(mx.float32)) ** 2


def test_shapes_names_dtype_dense_and_moe():
    m = tiny_qwen3(0)
    f = fisher_diag(m, tokens(), DENSE[:3])
    assert list(f) == DENSE[:3]
    for n, v in f.items():
        assert v.shape == get_weight(m, n).shape and v.dtype == mx.float32
        assert float(mx.min(v)) >= 0 and float(mx.sum(v)) > 0
    moe = tiny_moe(0)
    names = ["model.layers.1.mlp.experts.2.down_proj.weight", "model.layers.0.mlp.experts.0.gate_proj.weight",
             "model.layers.0.self_attn.o_proj.weight"]
    f = fisher_diag(moe, tokens(), names)
    assert list(f) == names
    for n in names:
        assert f[n].shape == get_weight(moe, n).shape
    with pytest.raises(KeyError):
        fisher_diag(m, tokens(), ["model.layers.9.mlp.up_proj.weight"])


def test_frozen_state_restored():
    m = tiny_qwen3(0)
    m.freeze()
    m.model.layers[0].mlp.unfreeze()
    before = trainable_keys(m)
    fisher_diag(m, tokens(), DENSE[:2] + ["model.layers.0.mlp.up_proj.weight"], group_size=1)
    assert trainable_keys(m) == before
    m2 = tiny_qwen3(0)
    before = trainable_keys(m2)
    fisher_diag(m2, tokens(), DENSE[:2], loss="kl_self")
    assert trainable_keys(m2) == before


@pytest.mark.parametrize("loss", ["ce", "kl_self"])
def test_group_size_does_not_change_result(loss):
    m = tiny_moe(1)
    names = DENSE[:4] + ["model.layers.0.mlp.experts.1.up_proj.weight", "model.layers.0.mlp.experts.3.up_proj.weight"]
    a = fisher_diag(m, tokens(1), names, loss=loss, seed=5)
    b = fisher_diag(m, tokens(1), names, loss=loss, seed=5, group_size=1)
    for n in names:
        np.testing.assert_allclose(np.array(a[n]), np.array(b[n]), rtol=1e-5, atol=1e-12)


def test_micro_batch_estimator():
    m = tiny_qwen3(2)
    toks = tokens(2, b=4)
    name = "model.layers.1.mlp.down_proj.weight"
    # micro_batch = B: one micro-batch -> the squared full-batch gradient
    f = np.array(fisher_diag(m, toks, [name], micro_batch=4)[name])
    ref = ce_grad_sq(m, toks, name)
    np.testing.assert_allclose(f, ref, rtol=1e-4, atol=1e-6 * ref.max())
    # micro_batch = 1: mean over sequences of the per-sequence squared gradient (> squared mean gradient)
    f1 = np.array(fisher_diag(m, toks, [name], micro_batch=1)[name])
    ref1 = np.mean([ce_grad_sq(m, toks[i:i + 1], name) for i in range(4)], axis=0)
    np.testing.assert_allclose(f1, ref1, rtol=1e-4, atol=1e-6 * ref1.max())
    assert f1.sum() > f.sum()


def test_kl_self_is_seeded_and_differs_from_ce():
    m = tiny_qwen3(0)
    n = DENSE[:1]
    a = fisher_diag(m, tokens(), n, loss="kl_self", seed=1)[n[0]]
    b = fisher_diag(m, tokens(), n, loss="kl_self", seed=1)[n[0]]
    c = fisher_diag(m, tokens(), n, loss="kl_self", seed=2)[n[0]]
    d = fisher_diag(m, tokens(), n, loss="ce")[n[0]]
    assert np.array_equal(np.array(a), np.array(b))
    assert not np.allclose(np.array(a), np.array(c))
    assert not np.allclose(np.array(a), np.array(d))


def test_fisher_proxy_hand_computation():
    f = mx.array([[1.0, 2.0], [0.0, 4.0]])
    d = mx.array([[0.5, -1.0], [3.0, 0.25]], dtype=mx.bfloat16)
    assert fisher_proxy(d, f) == pytest.approx(0.5 * (1 * 0.25 + 2 * 1 + 0 + 4 * 0.0625))


def test_fisher_summary():
    f = mx.array(np.arange(12, dtype=np.float32).reshape(3, 4))
    s = fisher_summary(f)
    assert s["total"] == pytest.approx(66.0) and s["mean"] == pytest.approx(5.5)
    assert s["p50"] == pytest.approx(5.5) and s["p99"] == pytest.approx(np.percentile(np.arange(12), 99))
    col = np.array([12.0, 15.0, 18.0, 21.0])
    assert s["col_sum_p50"] == pytest.approx(np.percentile(col, 50))
    assert s["col_sum_p99"] == pytest.approx(np.percentile(col, 99))
    assert s["col_sum_max"] == pytest.approx(21.0)


def test_moe_fisher_nonzero_only_for_routed_experts():
    class MaskedGate(nn.Module):
        """Router whose logits for experts 2 and 3 are pushed to -1e4: they are never routed."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __call__(self, x):
            return self.inner(x) + mx.array([0.0, 0.0, -1e4, -1e4], dtype=x.dtype)

    m = tiny_moe(3)
    m.model.layers[0].mlp.gate = MaskedGate(m.model.layers[0].mlp.gate)
    names = [f"model.layers.0.mlp.experts.{e}.{p}.weight" for e in range(4) for p in ("gate_proj", "down_proj")]
    f = fisher_diag(m, tokens(3), names)
    for n in names:
        e = int(n.split(".")[5])
        total = float(mx.sum(f[n]))
        assert (total > 0) if e < 2 else (total == 0.0), (n, total)


@pytest.mark.parametrize("loss", ["kl_self", "ce"])
def test_fisher_proxy_predicts_measured_kl(loss):
    """Rank prediction of measured KL (ΔCE on a random-init model is dominated by the first-order term and
    is not used). For kl_self the documented scale also holds: mean KL ≈ n_tokens_per_micro_batch · proxy."""
    m = tiny_qwen3(4)
    toks = tokens(4, b=8, t=32)
    f = fisher_diag(m, toks, DENSE, loss=loss, seed=0, micro_batch=1)
    ref = m(toks)
    mx.eval(ref)
    proxy, measured = [], []
    for i, n in enumerate(DENSE):
        w = get_weight(m, n)
        w_hat = perturb_weight(w, "gaussian", seed=100 + i, level=0.02)
        proxy.append(fisher_proxy(w_hat.astype(mx.float32) - w.astype(mx.float32), f[n]))
        measured.append(measure_tensor_sensitivity(m, n, w_hat, toks, ref, layer_output=False)["kl_mean"])
    rho = spearman(proxy, measured)
    assert len(DENSE) >= 8 and rho > 0, rho
    if loss == "kl_self":
        ratio = np.median(np.array(measured) / (np.array(proxy) * (toks.shape[1] - 1)))
        assert 0.5 < ratio < 2.0, ratio
