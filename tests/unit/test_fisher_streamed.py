"""Layer-streamed exact Fisher (suc.analysis.fisher.fisher_diag_streamed) equals the whole-model fisher_diag."""
import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.utils import load_model

from suc.analysis.fisher import fisher_diag, fisher_diag_streamed
from suc.models.streaming import load_lazy
from test_streamed_eval import dense_dir, moe_dir

BLOCK = [f"model.layers.{i}.{m}.weight" for i in range(2)
         for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                   "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
MOE = ["model.layers.0.self_attn.q_proj.weight", "model.layers.1.mlp.experts.2.down_proj.weight",
       "model.layers.0.mlp.experts.0.gate_proj.weight", "model.layers.0.mlp.experts.7.up_proj.weight",
       "model.layers.1.self_attn.o_proj.weight"]


def _tokens(seed=0, b=6, t=48):
    return mx.array(np.random.default_rng(seed).integers(0, 128, size=(b, t)).astype(np.int32))


def _streamed(d, toks, names, **kw):
    got, order = {}, []

    def sink(g, f):
        order.append(g)
        got.update(f)

    info = fisher_diag_streamed(load_lazy(d)[0], toks, names, model_dir=d, on_group=sink, **kw)
    assert info["groups"] == order
    return got, order


def _compare(want, got):
    """Block tensors (dense and MoE experts): bit-identical. Embedding: its lookup gradient is an atomic scatter-add whose bf16 summation
    order varies run to run (the whole-model fisher_diag differs from itself by ~3e-5 relative L1), so only
    agreement to that noise level is required."""
    assert set(want) == set(got)
    for n in want:
        a, b = np.array(want[n]), np.array(got[n])
        assert a.shape == b.shape and b.dtype == np.float32
        if n != "model.embed_tokens.weight":
            np.testing.assert_array_equal(a, b, err_msg=n)
        else:
            assert np.abs(a - b).sum() / np.abs(a).sum() < 1e-3, n
            assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.9999, n


@pytest.mark.parametrize("kind,loss", [("f32_tied", "ce"), ("bf16_tied", "ce"), ("bf16_untied", "kl_self"),
                                       ("f32_tied", "kl_self"), ("moe", "ce"), ("bf16_tied_fused", "ce"),
                                       ("bf16_untied_fused", "kl_self")])
@pytest.mark.parametrize("disk", [False, True])
def test_streamed_fisher_equals_full_model(tmp_path, kind, loss, disk):
    d = moe_dir(tmp_path) if kind == "moe" else dense_dir(tmp_path, mx.bfloat16 if "bf16" in kind else None,
                                                           tie="untied" not in kind,
                                                           head_dim=128 if "fused" in kind else 16)
    heads = ["lm_head.weight", "model.embed_tokens.weight"] if "untied" in kind or kind == "moe" \
        else ["model.embed_tokens.weight"]
    names = (MOE if kind == "moe" else BLOCK) + heads
    toks = _tokens()
    want = fisher_diag(load_model(d)[0], toks, names, micro_batch=2, loss=loss, seed=3)
    got, order = _streamed(d, toks, names, micro_batch=2, loss=loss, seed=3,
                           work_dir=tmp_path / "work" if disk else None, fingerprint="x")
    assert order == ["block_00001", "block_00000", "head"]
    _compare(want, got)
    if disk:
        assert not (tmp_path / "work" / "cot").exists() and not any((tmp_path / "work" / "act").rglob("*.npy"))


@pytest.mark.parametrize("loss", ["ce", "kl_self"])
def test_streamed_fisher_resumes_after_crash(tmp_path, loss):
    d = dense_dir(tmp_path, mx.bfloat16)
    names = BLOCK + ["model.embed_tokens.weight"]
    toks = _tokens(1)
    ref, _ = _streamed(d, toks, names, micro_batch=3, loss=loss, seed=5)
    work = tmp_path / "w"
    seen = []

    def crash_after_first(g, f):
        seen.append(g)
        if g == "block_00000":
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated"):
        fisher_diag_streamed(load_lazy(d)[0], toks, names, model_dir=d, on_group=crash_after_first,
                             micro_batch=3, work_dir=work, fingerprint="fp1", loss=loss, seed=5)
    assert seen == ["block_00001", "block_00000"]
    got, order = _streamed(d, toks, names, micro_batch=3, work_dir=work, fingerprint="fp1", loss=loss, seed=5)
    assert order == ["block_00000", "head"]                    # block 1 was not recomputed
    _compare({n: ref[n] for n in got}, got)
    got2, order2 = _streamed(d, toks, names, micro_batch=3, work_dir=work, fingerprint="fp2", loss=loss,
                             seed=5)  # new inputs: everything is redone
    assert order2 == ["block_00001", "block_00000", "head"]


def test_streamed_fisher_errors(tmp_path):
    d = dense_dir(tmp_path)
    with pytest.raises(KeyError):
        fisher_diag_streamed(load_lazy(d)[0], _tokens(), ["model.layers.5.mlp.up_proj.weight"], model_dir=d,
                             on_group=lambda g, f: None)
    with pytest.raises(KeyError, match="tied"):
        fisher_diag_streamed(load_lazy(d)[0], _tokens(), ["lm_head.weight"], model_dir=d, on_group=lambda g, f: None)
    with pytest.raises(ValueError, match="loss"):
        fisher_diag_streamed(load_lazy(d)[0], _tokens(), BLOCK, model_dir=d, on_group=lambda g, f: None, loss="x")
