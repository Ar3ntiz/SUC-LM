import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from suc.evaluation.baselines import apply_representation, logits_kl, size_report
from tiny_models import tiny_qwen3


def n_params(model) -> int:
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(model.parameters()))


@pytest.mark.parametrize("kind,dt", [("bf16", mx.bfloat16), ("fp16", mx.float16)])
def test_cast_representation_bpp16(kind, dt):
    model = tiny_qwen3()
    n = n_params(model)
    assert apply_representation(model, kind) is model
    r = size_report(model)
    assert r["parameter_count"] == n
    assert r["bpp"] == 16.0
    assert r["bytes_total"] == 2 * n == r["fp16_bytes"]
    assert r["bytes_by_dtype"] == {str(dt): 2 * n}
    assert r["bytes_weights_packed"] == 0 and r["bytes_scales_biases"] == 0


@pytest.mark.parametrize("bits,gs", [(4, 64), (4, 32), (8, 64)])
def test_affine_size_matches_hand_calculation(bits, gs):
    model = tiny_qwen3(tie=False)
    apply_representation(model, "bf16")
    n = n_params(model)
    apply_representation(model, "mlx_affine", bits=bits, group_size=gs)
    assert isinstance(model.model.layers[0].mlp.down_proj, nn.QuantizedLinear)
    assert isinstance(model.model.embed_tokens, nn.QuantizedEmbedding)
    r = size_report(model)
    assert r["parameter_count"] == n  # logical count survives packing
    # hand calculation: every Linear/Embedding weight is quantized; norms stay bf16
    h, v, inter, L = 64, 128, 128, 2
    per_layer = h * 64 + 2 * (h * 32) + 64 * h + 3 * h * inter  # q,k,v,o + gate,up,down
    qn = L * per_layer + 2 * v * h  # + embed + lm_head
    assert r["parameter_count_quantized"] == qn
    assert r["bytes_weights_packed"] == qn * bits // 8
    assert r["bytes_scales_biases"] == (qn // gs) * 2 * 2  # fp16/bf16 scale + bias per group
    q_bpp = (r["bytes_weights_packed"] + r["bytes_scales_biases"]) * 8 / qn
    assert q_bpp == pytest.approx(bits + 32 / gs)
    assert r["bytes_other"] == (n - qn) * 2
    assert r["bytes_total"] == r["bytes_weights_packed"] + r["bytes_scales_biases"] + r["bytes_other"]
    assert r["bpp"] == pytest.approx(r["bytes_total"] * 8 / n)
    assert bits < r["bpp"] < 16


def test_affine_skips_indivisible_layers():
    model = tiny_qwen3()
    apply_representation(model, "mlx_affine", bits=4, group_size=128)  # hidden 64 not divisible
    assert isinstance(model.model.layers[0].mlp.down_proj, nn.QuantizedLinear)  # in=128
    assert isinstance(model.model.layers[0].self_attn.q_proj, nn.Linear)
    assert not isinstance(model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)


def test_affine_model_still_runs():
    model = tiny_qwen3()
    x = mx.array([[1, 2, 3, 4]])
    ref = model(x)
    apply_representation(model, "mlx_affine", bits=4)
    out = model(x)
    assert out.shape == ref.shape
    kl = logits_kl(ref, out)
    assert kl["kl_mean"] > 0


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        apply_representation(tiny_qwen3(), "int3_magic")


def test_logits_kl_identical_and_different():
    rng = np.random.default_rng(0)
    a = mx.array(rng.normal(size=(2, 5, 17)).astype(np.float32))
    r = logits_kl(a, a)
    assert r["kl_mean"] == pytest.approx(0.0, abs=1e-6) and r["kl_max"] == pytest.approx(0.0, abs=1e-6)
    assert r["top1_agreement"] == 1.0 and r["tokens"] == 10
    b = mx.array(rng.normal(size=(2, 5, 17)).astype(np.float32))
    r2 = logits_kl(a, b)
    assert r2["kl_mean"] > 0 and r2["kl_max"] >= r2["kl_mean"]
    # hand check on one row
    p = np.exp(np.array(a[0, 0])) / np.exp(np.array(a[0, 0])).sum()
    q = np.exp(np.array(b[0, 0])) / np.exp(np.array(b[0, 0])).sum()
    one = logits_kl(a[0, 0][None], b[0, 0][None])
    assert one["kl_mean"] == pytest.approx(float((p * np.log(p / q)).sum()), rel=1e-4)


def test_logits_kl_bf16_inputs_computed_in_fp32():
    rng = np.random.default_rng(1)
    a = mx.array(rng.normal(size=(4, 9)).astype(np.float32)).astype(mx.bfloat16)
    r = logits_kl(a, a)
    assert isinstance(r["kl_mean"], float) and abs(r["kl_mean"]) < 1e-6


def _tiny_moe():
    from mlx_lm.models import qwen3_moe

    mx.random.seed(0)
    args = qwen3_moe.ModelArgs(
        model_type="qwen3_moe", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, vocab_size=128,
        decoder_sparse_step=1, mlp_only_layers=[], num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
        rope_theta=10000.0, max_position_embeddings=512, tie_word_embeddings=False, norm_topk_prob=True,
    )
    model = qwen3_moe.Model(args)
    mx.eval(model.parameters())
    return model


def test_affine_quantizes_moe_experts_like_mlx_lm():
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    model = _tiny_moe()
    n_before = size_report(model)["parameter_count"]
    apply_representation(model, "mlx_affine", bits=4, group_size=64)
    mlp = model.model.layers[0].mlp
    assert isinstance(mlp.switch_mlp.up_proj, QuantizedSwitchLinear)
    assert mlp.gate.bits == 8  # router kept at 8 bits, as mlx-lm's qwen3_moe quant_predicate does
    rep = size_report(model)
    assert rep["parameter_count"] == n_before  # logical count unchanged by packing
    experts = 2 * 3 * 4 * 64 * 64               # layers x projections x experts x (64 x 64)
    assert rep["parameter_count_quantized"] >= experts
    assert rep["bpp"] < 6.0
