"""Tests for the block-streaming (GPTQ-style sequential) forward pass (spec D8)."""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_moe

from suc.models.streaming import block_prefix, embed, head, load_lazy, model_blocks, stream_blocks
from tiny_models import tiny_qwen3

FIX = Path(__file__).parent / "fixtures"
B, T = 4, 16


def tiny_qwen3_moe(seed: int = 0) -> qwen3_moe.Model:
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


def _tokens(vocab: int = 128, seed: int = 1) -> mx.array:
    return mx.random.randint(0, vocab, (B, T), key=mx.random.key(seed))


MODELS = {
    "qwen3_tied": lambda: tiny_qwen3(tie=True),
    "qwen3_untied": lambda: tiny_qwen3(tie=False),
    "qwen3_moe": tiny_qwen3_moe,
}


@pytest.mark.parametrize("name", list(MODELS))
def test_streamed_logits_match_full_forward(name):
    model = MODELS[name]()
    tokens = _tokens()
    ref = model(tokens)
    out = stream_blocks(model, tokens, micro_batch=2, release=False, return_logits=True)
    assert out["logits"].shape == ref.shape
    assert mx.allclose(out["logits"], ref, atol=1e-5, rtol=1e-5).item()
    assert out["hidden"].shape == (B, T, 64)
    assert len(out["block_seconds"]) == 2 and out["peak_memory_bytes"] > 0


def test_head_and_embed_compose_to_model():
    for tie in (True, False):
        model = tiny_qwen3(tie=tie)
        tokens = _tokens()
        h = embed(model, tokens)
        mask = "causal"
        for blk in model_blocks(model):
            h = blk(h, mask)
        assert mx.allclose(head(model, h), model(tokens), atol=1e-5).item()


def test_block_prefix():
    assert block_prefix(3) == "model.layers.3"


def test_hooks_called_once_per_block_in_order():
    model = tiny_qwen3()
    calls = []
    stream_blocks(model, _tokens(), on_block=lambda i, b: calls.append(("on", i)),
                  after_block=lambda i, b: calls.append(("after", i)))
    assert calls == [("on", 0), ("after", 0), ("on", 1), ("after", 1)]


class _Recorder(nn.Module):
    """Wraps input_layernorm to record the block input (the norm's input is the block's residual input)."""

    def __init__(self, inner: nn.Module, store: list):
        super().__init__()
        self.inner = inner
        self._store = store

    def __call__(self, x):
        self._store.append(x)
        return self.inner(x)


def test_instrumented_block0_input_equals_embeddings():
    model = tiny_qwen3()
    tokens = _tokens()
    seen: dict[int, list] = {}

    def on_block(i, blk):
        seen[i] = []
        blk.input_layernorm = _Recorder(blk.input_layernorm, seen[i])

    def after_block(i, blk):
        blk.input_layernorm = blk.input_layernorm.inner

    stream_blocks(model, tokens, on_block=on_block, after_block=after_block, micro_batch=3, release=False)
    x0 = mx.concatenate(seen[0], axis=0)
    assert len(seen[0]) == 2  # ceil(4 / 3) micro-batches
    assert mx.array_equal(x0, embed(model, tokens)).item()
    assert all(not isinstance(b.input_layernorm, _Recorder) for b in model_blocks(model))


def test_micro_batch_size_does_not_change_result():
    model = tiny_qwen3_moe()
    tokens = _tokens()
    a = stream_blocks(model, tokens, micro_batch=1, release=False, return_logits=True)
    b = stream_blocks(model, tokens, micro_batch=B, release=False, return_logits=True)
    assert mx.allclose(a["hidden"], b["hidden"], atol=1e-5).item()
    assert mx.allclose(a["logits"], b["logits"], atol=1e-5).item()


def _block_nbytes(blk: nn.Module) -> int:
    return sum(v.nbytes for _, v in tree_flatten(blk.parameters()))


def test_release_frees_block_parameters():
    model = tiny_qwen3()
    before = [_block_nbytes(b) for b in model_blocks(model)]
    out = stream_blocks(model, _tokens(), release=True, return_logits=True)
    assert out["logits"] is not None
    assert all(_block_nbytes(b) == 0 for b in model_blocks(model))
    assert model.model.embed_tokens.weight.size > 0  # embeddings / head are kept
    kept = tiny_qwen3()
    stream_blocks(kept, _tokens(), release=False)
    assert [_block_nbytes(b) for b in model_blocks(kept)] == before


def test_hidden_dtype_is_used_between_blocks():
    model = tiny_qwen3()
    out = stream_blocks(model, _tokens(), hidden_dtype=mx.float16, release=False)
    assert out["hidden"].dtype == mx.float16
    assert stream_blocks(model, _tokens(), release=False)["hidden"].dtype == mx.float32


def _save_tiny(model: nn.Module, d: Path, tie: bool) -> Path:
    d.mkdir()
    mx.save_safetensors(str(d / "model.safetensors"), dict(tree_flatten(model.parameters())))
    cfg = json.loads((FIX / "qwen3_0.6b_config.json").read_text())
    cfg.update(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4,
               num_key_value_heads=2, head_dim=16, vocab_size=128, tie_word_embeddings=tie,
               rope_theta=10000.0, max_position_embeddings=512)  # fixture is trimmed: add rope fields
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_works_on_lazily_loaded_model(tmp_path):
    ref_model = tiny_qwen3(tie=False)
    tokens = _tokens()
    ref = ref_model(tokens)
    model, tokenizer = load_lazy(_save_tiny(ref_model, tmp_path / "m", tie=False))
    assert tokenizer is None  # no tokenizer files in the directory
    out = stream_blocks(model, tokens, return_logits=True)
    assert mx.allclose(out["logits"], ref, atol=1e-5).item()
