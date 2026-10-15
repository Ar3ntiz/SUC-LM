"""K6: online input rotations fused into the element-wise ops that produce the projection inputs (RMSNorm → q|k|v
and gate|up, SwiGLU → down_proj). Same math as rotating the op's output: exactness vs the decoded-weights model and
vs the unfused runtime, and launches per block."""
from __future__ import annotations

import re
import tempfile

import mlx.core as mx
import numpy as np
import pytest

import suc.kernels.rotation as rot
import suc.kernels.suc_linear as sl
import suc.runtime.compressed_model as cm
from suc.analysis.sensitivity import set_weight
from suc.evaluation.baselines import logits_kl
from suc.evaluation.reconstruct import rotation_seeds
from suc.quant.codecs import make_codec
from suc.quant.hadamard import random_signs
from suc.quant.residual import ResidualCodec
from tiny_models import tiny_qwen3


@pytest.fixture(autouse=True)
def _fusion_on(monkeypatch):
    """These tests exercise the fused path, which is off by default (slower on the M1 Ultra, K6)."""
    monkeypatch.setattr(cm, "FUSE_ELEMENTWISE", True)

PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
_VIEWS = {"Reshape", "Broadcast", "ExpandDims", "Squeeze", "Transpose", "Slice", "StopGradient", "Flatten",
          "Unflatten"}


# ------------------------------------------------------------------------------------------- kernels

@pytest.fixture(params=[dict(LAYOUT="blocks"), dict(LAYOUT="blocks", BLOCK_Q=512, BLOCK_SG=4),
                        dict(LAYOUT="blocks", BLOCK_Q=128, BLOCK_SG=2), dict(LAYOUT="row")],
                ids=["blocks", "blocks-q512", "blocks-q128-sg2", "row"])
def layout(request, monkeypatch):
    """The prologue kernels exist for both rotation layouts (R5); each must match its own plain rotation."""
    for k, v in request.param.items():
        monkeypatch.setattr(rot, k, v)


@pytest.mark.parametrize("n", [64, 384, 1024, 3072, 4096, 12288])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("b", [1, 3])
def test_rms_norm_rht_equals_norm_then_rotation(n, dtype, b, layout) -> None:
    x = (3 * mx.random.normal((b, n), key=mx.random.key(n))).astype(dtype)
    w = (1 + 0.3 * mx.random.normal((n,), key=mx.random.key(1))).astype(dtype)
    s = random_signs(n, 5)
    xr, xn = rot.rms_norm_rht(x, w, 1e-6, s, with_input=True)
    xn_ref = mx.fast.rms_norm(x, w, 1e-6)
    assert xr.dtype == xn.dtype == dtype
    if n > 4096:   # MLX's looped kernel: a different mean-square summation order (≤ 1 ulp of the output)
        a, r = np.array(xn.astype(mx.float32)), np.array(xn_ref.astype(mx.float32))
        ulp = {mx.float32: 2 ** -22, mx.float16: 2 ** -10, mx.bfloat16: 2 ** -7}[dtype]
        assert np.all(np.abs(a - r) <= np.abs(r) * ulp + 1e-30)
        xn_ref = xn
    # bit-identical normed output; the rotation of identical inputs is identical
    assert np.array_equal(np.array(xn.astype(mx.float32)), np.array(xn_ref.astype(mx.float32)))
    assert np.array_equal(np.array(xr.astype(mx.float32)), np.array(rot.rht_fused(xn_ref, s).astype(mx.float32)))
    xr_only, none = rot.rms_norm_rht(x, w, 1e-6, s)
    assert none is None and np.array_equal(np.array(xr_only.astype(mx.float32)), np.array(xr.astype(mx.float32)))


@pytest.mark.parametrize("n", [64, 384, 3072, 12288, 17 * 32, 17408])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_swiglu_rht_equals_swiglu_then_rotation(n, dtype, layout) -> None:
    g = mx.random.normal((2, 3, n), key=mx.random.key(7)).astype(dtype)
    u = mx.random.normal((2, 3, n), key=mx.random.key(8)).astype(dtype)
    s = random_signs(n, 9)
    xr, h = rot.swiglu_rht(g, u, s, with_input=True)
    g32, u32 = g.astype(mx.float32), u.astype(mx.float32)
    h_ref = (g32 * mx.sigmoid(g32) * u32).astype(dtype)               # float32 math, one rounding
    assert h.shape == xr.shape == g.shape and h.dtype == dtype
    d = np.abs(np.array(h.astype(mx.float32)) - np.array(h_ref.astype(mx.float32)))
    ulp = {mx.float32: 1e-6, mx.float16: 1e-3, mx.bfloat16: 8e-3}[dtype]
    assert d.max() <= ulp * (1 + np.abs(np.array(h_ref.astype(mx.float32))).max())
    ref = rot.rht_fused(h.reshape(-1, n), s).reshape(g.shape)      # rotation of the kernel's own h: identical
    assert np.array_equal(np.array(xr.astype(mx.float32)), np.array(ref.astype(mx.float32)))


def test_prologue_rejects_unsupported_sizes() -> None:
    with pytest.raises(ValueError):
        rot.rms_norm_rht(mx.ones((1, 96)), mx.ones((96,)), 1e-6, mx.ones((96,)))   # 96 = 12·8: 2^k < 32


# ------------------------------------------------------------------------------------------- model

def _get(model, name):
    cur = model
    for p in name.split("."):
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    return cur


def _codec(res: str | None, one_sided: bool):
    base = make_codec({"kind": "vq", "k": 16, "block_size": 4, "rotate": True, "kmeans_iters": 3,
                       "rotate_out": not one_sided})
    if res is None:
        return base
    return ResidualCodec(base, 0.03, "int8", "magnitude", structure=res)


def _models(res, one_sided, dtype=mx.float32):
    ref, comp = tiny_qwen3(seed=0), tiny_qwen3(seed=0)
    if dtype != mx.float32:
        ref.set_dtype(dtype)
        comp.set_dtype(dtype)
    codec = _codec(res, one_sided)
    enc = {}
    for i, n in enumerate(f"model.layers.{l}.{p}.weight" for l in range(2) for p in PROJ):
        enc[n] = codec.encode(_get(comp, n[:-7]).weight, seed=i, **rotation_seeds(n, 0))
        set_weight(ref, n, codec.decode(enc[n]))
    mx.eval(ref.parameters())
    cm.convert_to_compressed(comp, enc)
    return ref, comp


def _toks(b=3, t=12):
    return mx.array(np.random.default_rng(0).integers(0, 128, size=(b, t)).astype(np.int32))


@pytest.mark.parametrize("res", [None, "columns", "csr"])
@pytest.mark.parametrize("one_sided", [True, False])
# float32 KL is at the rounding floor (|KL| ≲ 5e-8 on this tiny model, also unfused and of either sign)
@pytest.mark.parametrize("dtype,tol,kl_tol", [(mx.float32, 1e-4, 1e-7), (mx.bfloat16, 3e-2, 3e-4)])
def test_fused_model_matches_decoded_weights(res, one_sided, dtype, tol, kl_tol) -> None:
    ref, comp = _models(res, one_sided, dtype)
    blk = comp.model.layers[0]
    assert type(blk.input_layernorm) is cm.RotatedRMSNorm and type(blk.post_attention_layernorm) is cm.RotatedRMSNorm
    assert type(blk.mlp) is cm.SucMLP
    # parameter paths unchanged (norm weights where they were)
    assert "model.layers.0.input_layernorm.weight" in dict(__import__("mlx.utils").utils.tree_flatten(
        comp.parameters()))
    lr, lc = ref(_toks()).astype(mx.float32), comp(_toks()).astype(mx.float32)
    rel = float(mx.linalg.norm(lc - lr) / mx.linalg.norm(lr))
    kl = logits_kl(lr, lc)
    assert rel < tol, rel
    assert kl["kl_mean"] < kl_tol and kl["top1_agreement"] > 0.95, kl


@pytest.mark.parametrize("res", [None, "columns"])
def test_fused_equals_unfused_runtime(res, monkeypatch) -> None:
    _, comp = _models(res, True)
    toks = _toks()
    fused = np.array(comp(toks))
    monkeypatch.setattr(cm, "FUSE_ELEMENTWISE", False)
    unfused = np.array(comp(toks))
    # float32: identical math, only the float32 summation order of the SwiGLU sigmoid may differ
    np.testing.assert_allclose(fused, unfused, rtol=0, atol=1e-4 * np.abs(unfused).max())


def test_rotations_happen_inside_the_elementwise_kernels(monkeypatch) -> None:
    """Eager trace: only o_proj still calls rotate_in; the norms / SwiGLU launch the fused kernels."""
    monkeypatch.setattr(sl, "COMPILE", False)
    _, comp = _models("columns", True)
    calls = {"rotate_in": 0, "norm": 0, "swiglu": 0}
    orig_ri, orig_n, orig_s = sl.SucLinear.rotate_in, rot.rms_norm_rht, rot.swiglu_rht

    def ri(self, x2):
        calls["rotate_in"] += 1
        return orig_ri(self, x2)

    def nr(*a, **k):
        calls["norm"] += 1
        return orig_n(*a, **k)

    def sw(*a, **k):
        calls["swiglu"] += 1
        return orig_s(*a, **k)
    monkeypatch.setattr(sl.SucLinear, "rotate_in", ri)
    monkeypatch.setattr(rot, "rms_norm_rht", nr)
    monkeypatch.setattr(rot, "swiglu_rht", sw)
    mx.eval(comp(_toks()))
    assert calls == {"rotate_in": 2, "norm": 4, "swiglu": 2}, calls


def _launches(layer, hid) -> list[str]:
    x = mx.random.normal((1, 1, hid))
    mx.eval(x)
    y = layer(x, None, None)
    with tempfile.NamedTemporaryFile("w+", suffix=".dot") as f:
        mx.export_to_dot(f.name, y)
        ops = re.findall(r'label ="(\w+)", shape=rectangle', open(f.name).read())
    mx.eval(y)
    return [o for o in ops if o not in _VIEWS]


@pytest.mark.parametrize("res,one_sided,suc_before,suc_after", [
    (None, False, 12, 12), (None, True, 8, 8), ("columns", True, 12, 12)])
def test_launches_per_block(res, one_sided, suc_before, suc_after, monkeypatch) -> None:
    """The 3 input rotations move into the norm / SwiGLU launches: the 2 block RMSNorms and the compiled SwiGLU of
    MLX disappear, the SUC custom kernels stay (one-sided: no output-rotation launch at all)."""
    _, comp = _models(res, one_sided)
    blk = comp.model.layers[0]
    after = _launches(blk, 64)
    monkeypatch.setattr(cm, "FUSE_ELEMENTWISE", False)
    before = _launches(blk, 64)
    assert before.count("CustomKernel") == suc_before and after.count("CustomKernel") == suc_after
    assert before.count("RMSNorm") == 4 and after.count("RMSNorm") == 2           # q_norm / k_norm remain
    assert any(o.startswith("Compiled") for o in before) and not any(o.startswith("Compiled") for o in after)
    assert len(before) - len(after) == 3
    if one_sided:   # per unit: q|k|v kernel (+res), o rotate_in + kernel (+res), gate|up kernel (+res), down kernel (+res)
        assert after.count("CustomKernel") == (5 if res is None else 9) + 3
