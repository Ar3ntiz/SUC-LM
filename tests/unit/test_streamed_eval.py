"""Layer-streamed evaluation (D8) reproduces the whole-model PPL, reference logits and KL (suc.evaluation.streamed)."""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from suc.analysis.sensitivity import set_weight
from suc.evaluation.reconstruct import evaluate_quality, reference_logits
from suc.evaluation.streamed import RefLogits, streamed_quality
from suc.io.safetensors_store import RowStreamWriter, SafetensorsStore, read_header
from suc.models.streaming import HiddenBuffer, load_lazy, model_blocks
from tiny_models import tiny_qwen3, tiny_qwen3_moe

FIX = Path(__file__).parent / "fixtures"


def _save(model, d: Path, cfg: dict, dtype=None) -> Path:
    d.mkdir()
    params = dict(tree_flatten(model.parameters()))
    if dtype is not None:
        params = {k: v.astype(dtype) for k, v in params.items()}
    mx.save_safetensors(str(d / "model.safetensors"), params)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def dense_dir(tmp_path: Path, dtype=None, tie=True, head_dim=16) -> Path:
    """Tiny Qwen3 checkpoint. head_dim=128 (2 heads) takes MLX's fused SDPA kernel, whose forward differs from the
    composed implementation used under gradient tracing (the case the streamed Fisher must reproduce)."""
    cfg = json.loads((FIX / "qwen3_0.6b_config.json").read_text())
    heads = 4 if head_dim == 16 else 2
    cfg.update(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=heads,
               num_key_value_heads=heads // 2, head_dim=head_dim, vocab_size=128, tie_word_embeddings=tie,
               rope_theta=10000.0, max_position_embeddings=512)
    if head_dim == 16:
        model = tiny_qwen3(0, tie=tie)
    else:
        from mlx_lm.models import qwen3
        mx.random.seed(0)
        model = qwen3.Model(qwen3.ModelArgs(
            model_type="qwen3", hidden_size=64, num_hidden_layers=2, intermediate_size=128, num_attention_heads=2,
            rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=1, max_position_embeddings=512,
            rope_theta=10000.0, head_dim=head_dim, tie_word_embeddings=tie))
        mx.eval(model.parameters())
    return _save(model, tmp_path / f"dense_{dtype}_{tie}_{head_dim}", cfg, dtype)


def moe_dir(tmp_path: Path) -> Path:
    """HF-layout checkpoint (per-expert tensors) of the tiny MoE: the inverse of qwen3_moe.sanitize."""
    m = tiny_qwen3_moe(0, tie=False)
    params = {}
    for k, v in tree_flatten(m.parameters()):
        if ".switch_mlp." in k:
            pre, proj = k.split(".switch_mlp.")
            for e in range(v.shape[0]):
                params[f"{pre}.experts.{e}.{proj}"] = v[e]
        else:
            params[k] = v
    d = tmp_path / "moe"
    d.mkdir()
    mx.save_safetensors(str(d / "model.safetensors"), params)
    a = m.args
    cfg = {"model_type": "qwen3_moe", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
           "num_attention_heads": 4, "num_experts": a.num_experts, "num_experts_per_tok": a.num_experts_per_tok,
           "decoder_sparse_step": 1, "mlp_only_layers": [], "moe_intermediate_size": a.moe_intermediate_size,
           "rms_norm_eps": 1e-6, "vocab_size": 128, "num_key_value_heads": 2, "head_dim": 16,
           "rope_theta": 10000.0, "tie_word_embeddings": False, "max_position_embeddings": 512,
           "norm_topk_prob": True}
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def _tokens(seed, shape, vocab=128):
    return mx.array(np.random.default_rng(seed).integers(0, vocab, size=shape).astype(np.int32))


def _full(d: Path):
    from mlx_lm.utils import load_model

    return load_model(d)[0]


def _bits(a: mx.array) -> np.ndarray:
    return np.array(a.view(mx.uint16)) if a.dtype == mx.bfloat16 else np.array(a)


def test_row_stream_writer_and_load_rows(tmp_path):
    x = mx.random.normal((7, 3, 5)).astype(mx.bfloat16)
    p = tmp_path / "x.safetensors"
    with RowStreamWriter(p, "logits", x.shape, mx.bfloat16) as w:
        for s in range(0, 7, 3):
            w.write(x[s:s + 3])
    np.testing.assert_array_equal(_bits(mx.load(str(p))["logits"]), _bits(x))   # standard reader agrees
    st = SafetensorsStore(read_header(p))
    np.testing.assert_array_equal(_bits(st.load_rows("logits", 2, 6)), _bits(x[2:6]))
    with pytest.raises(ValueError, match="rows"):
        with RowStreamWriter(tmp_path / "y.safetensors", "a", (4, 2), mx.float32) as w:
            w.write(mx.zeros((2, 2)))


@pytest.mark.parametrize("spill", [False, True])
def test_hidden_buffer_round_trip(tmp_path, spill):
    hb = HiddenBuffer(2, tmp_path / "h" if spill else None)
    a, b = mx.random.normal((2, 3, 4)).astype(mx.bfloat16), mx.random.normal((1, 3, 4))
    hb.put(0, a)
    hb.put(1, b)
    np.testing.assert_array_equal(_bits(hb.get(0)), _bits(a))
    assert hb.get(0).dtype == mx.bfloat16 and hb.get(1).dtype == mx.float32
    hb.put(0, b)
    np.testing.assert_array_equal(np.array(hb.get(0)), np.array(b))
    hb.clear()
    assert not (tmp_path / "h").exists()


@pytest.mark.parametrize("kind", ["f32_tied", "bf16_tied", "bf16_untied", "moe"])
@pytest.mark.parametrize("spill", [False, True])
def test_streamed_quality_equals_whole_model(tmp_path, kind, spill):
    d = moe_dir(tmp_path) if kind == "moe" else dense_dir(tmp_path, mx.bfloat16 if "bf16" in kind else None,
                                                           tie=kind != "bf16_untied")
    ppl_tokens = _tokens(1, (700,))
    kl_tokens = _tokens(2, (5, 24))
    full = _full(d)
    ref = evaluate_quality(full, ppl_tokens=ppl_tokens, seq_len=64, max_tokens=600)
    ref_lg = mx.concatenate(reference_logits(full, kl_tokens, chunk=2), axis=0)

    out = tmp_path / "ref.safetensors"
    lazy = load_lazy(d)[0]
    r = streamed_quality(lazy, ppl_tokens=ppl_tokens, seq_len=64, max_tokens=600, kl_tokens=kl_tokens, chunk=2,
                         ref_out=out, spill_dir=tmp_path / "spill" if spill else None)
    assert r["perplexity"] == ref["perplexity"]                     # bit-identical NLL sum
    np.testing.assert_array_equal(_bits(mx.load(str(out))["logits"]), _bits(ref_lg))
    assert r["kl"] is None
    assert all(p.size == 0 for blk in model_blocks(lazy) for _, p in tree_flatten(blk.parameters()))  # released

    # a perturbed "compressed" model: install modified weights block by block, compare with the whole-model KL
    def perturb(model, i):
        for name in (f"model.layers.{i}.self_attn.q_proj.weight", f"model.layers.{i}.mlp.down_proj.weight"
                     if kind != "moe" else f"model.layers.{i}.mlp.experts.1.down_proj.weight"):
            from suc.analysis.sensitivity import get_weight
            w = get_weight(model, name)
            set_weight(model, name, w * 1.05)

    test_full = _full(d)
    for i in range(2):
        perturb(test_full, i)
    want = evaluate_quality(test_full, ppl_tokens=ppl_tokens, seq_len=64, max_tokens=600, kl_tokens=kl_tokens,
                            ref_logits=reference_logits(full, kl_tokens, chunk=2), chunk=2)
    lazy = load_lazy(d)[0]
    got = streamed_quality(lazy, ppl_tokens=ppl_tokens, seq_len=64, max_tokens=600, kl_tokens=kl_tokens, chunk=2,
                           ref_logits=RefLogits(out), prepare_block=lambda i, b: perturb(lazy, i),
                           spill_dir=tmp_path / "spill2" if spill else None)
    assert got["perplexity"] == want["perplexity"]
    assert got["kl"] == want["kl"]
    assert got["kl"]["kl_mean"] > 0
