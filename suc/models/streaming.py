"""Block-streaming forward pass over a lazily loaded mlx-lm model (spec D8).

GPTQ-style sequential pass: all calibration sequences are embedded, then the
network runs ONE transformer block at a time over every sequence. The caller may
instrument a block before it runs (``on_block``) and read / remove the
instrumentation afterwards (``after_block``); the block's parameters are then
released, so at most one block's weights (plus embeddings / head and the hidden
states of all sequences) are resident at once when the model was loaded lazily.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model, load_tokenizer

_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")

BlockHook = Callable[[int, nn.Module], None]

# model families implemented in SUC itself (mlx-lm has no module for them): model_type -> module
LOCAL_MODELS = {"qwen4_exp": "suc.models.mlx_qwen4_exp", "qwen3_5": "suc.models.mlx_qwen3_5",
                "deepseek_v4": "suc.models.mlx_deepseek_v4",
                "glm5_next": "suc.models.mlx_glm5_next"}


def get_model_classes(config: dict):
    """(Model, ModelArgs) for a config: SUC's own implementations first (LOCAL_MODELS), else mlx-lm's."""
    import importlib

    from mlx_lm.utils import _get_classes

    mod = LOCAL_MODELS.get(config.get("model_type"))
    if mod is not None:
        arch = importlib.import_module(mod)
        return arch.Model, arch.ModelArgs
    return _get_classes(config)


def load_lazy(model_dir: str | Path) -> tuple[nn.Module, object]:
    """Lazily load a local mlx-lm model (weights memory-mapped, materialized on first use).

    Equivalent to ``mlx_lm.load(model_dir, lazy=True)``. If the directory holds no
    tokenizer files (e.g. tiny test checkpoints) the tokenizer is ``None``. Models with memory-mapped lookup tables
    (qwen4_exp n-gram tables) get them attached from the checkpoint (``attach_checkpoint``).
    """
    path = Path(model_dir)
    model, config = load_model(path, lazy=True, get_model_classes=get_model_classes)
    if hasattr(model, "attach_checkpoint"):
        model.attach_checkpoint(path)
    tokenizer = None
    if any((path / f).exists() for f in _TOKENIZER_FILES):
        tokenizer = load_tokenizer(path, eos_token_ids=config.get("eos_token_id"))
    return model, tokenizer


def model_blocks(model: nn.Module) -> list[nn.Module]:
    """The transformer blocks, in execution order (qwen3 / qwen3_moe: ``model.model.layers``)."""
    return model.model.layers


def block_prefix(i: int) -> str:
    """Checkpoint parameter-name prefix of block ``i``."""
    return f"model.layers.{i}"


def embed(model: nn.Module, tokens: mx.array) -> mx.array:
    """Token embeddings (B, T, h): the hidden state entering block 0 (models with a wider residual stream, e.g.
    qwen4_exp hyper-connections, define ``embed_stream``)."""
    if hasattr(model, "embed_stream"):
        return model.embed_stream(tokens)
    return model.model.embed_tokens(tokens)


def head(model: nn.Module, h: mx.array) -> mx.array:
    """Final norm + LM head (tied embeddings via ``embed_tokens.as_linear``) -> logits (B, T, V); ``head_stream``
    when the model defines it."""
    if hasattr(model, "head_stream"):
        return model.head_stream(h)
    out = model.model.norm(h)
    if model.args.tie_word_embeddings:
        return model.model.embed_tokens.as_linear(out)
    return model.lm_head(out)


def _release(block: nn.Module) -> None:
    """Replace every parameter of ``block`` by an empty array so its buffers can be freed."""
    block.update(tree_map(lambda p: mx.array([], dtype=p.dtype), block.parameters()))


def stream_blocks(model: nn.Module, tokens: mx.array, *,
                  on_block: BlockHook | None = None, after_block: BlockHook | None = None,
                  micro_batch: int = 8, release: bool = True, return_logits: bool = False,
                  hidden_dtype: mx.Dtype | None = None) -> dict:
    """Run ``model`` block by block over all sequences of ``tokens`` (B, T).

    Uses the causal mask mlx-lm builds without a KV cache. Returns
    ``{"hidden": (B,T,h) before the final norm, "logits": (B,T,V) or None,
    "peak_memory_bytes": int, "block_seconds": [float per block]}``.
    """
    mx.reset_peak_memory()
    n = tokens.shape[0]
    spans = [(s, min(s + micro_batch, n)) for s in range(0, n, micro_batch)]
    hs = []
    for s, e in spans:
        h = embed(model, tokens[s:e])
        hs.append(h.astype(hidden_dtype) if hidden_dtype is not None else h)
    mx.eval(hs)

    block_seconds = []
    for i, block in enumerate(model_blocks(model)):
        t0 = time.perf_counter()
        if on_block is not None:
            on_block(i, block)
        new = []
        for h in hs:
            out = block(h, create_attention_mask(h, None))
            out = out.astype(hidden_dtype) if hidden_dtype is not None else out
            mx.eval(out)  # per micro-batch: bounds the activation working set
            new.append(out)
        hs = new
        if after_block is not None:
            after_block(i, block)
        if release:
            _release(block)
            mx.clear_cache()
        block_seconds.append(time.perf_counter() - t0)

    logits = None
    if return_logits:
        logits = []
        for h in hs:
            lg = head(model, h)
            mx.eval(lg)
            logits.append(lg)
        logits = mx.concatenate(logits, axis=0)
    hidden = mx.concatenate(hs, axis=0)
    mx.eval(hidden)
    return {"hidden": hidden, "logits": logits,
            "peak_memory_bytes": int(mx.get_peak_memory()), "block_seconds": block_seconds}


# ------------------------------------------------------------------ block-sequential evaluation (D8)

class HiddenBuffer:
    """Hidden states of every micro-batch between two blocks of a block-sequential pass.

    In memory (a list) by default; with ``spill_dir`` each micro-batch is a raw ``.npy`` file (bf16 stored as its
    uint16 bit pattern, so the round trip is bit-exact) read back through a memory map, so only the micro-batch
    being processed is resident. ``put`` may overwrite the micro-batch that was just read (``get`` copies it)."""

    def __init__(self, n: int, spill_dir: str | Path | None = None) -> None:
        self.n = n
        self.dir = Path(spill_dir) if spill_dir is not None else None
        self._mem: list[mx.array | None] = [None] * n
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, j: int, bf16: bool) -> Path:
        # the dtype is part of the name (bf16 is stored as its uint16 bit pattern), so a resumed run reads it back
        return self.dir / f"{j:06d}{'.bf16' if bf16 else ''}.npy"

    def put(self, j: int, a: mx.array) -> None:
        if self.dir is None:
            self._mem[j] = a
            return
        mx.eval(a)
        bf16 = a.dtype == mx.bfloat16
        raw = np.array(a.view(mx.uint16)) if bf16 else np.array(a)
        tmp = self.dir / f".tmp.{j:06d}.npy"
        np.save(tmp, raw)
        other = self._path(j, not bf16)
        if other.exists():
            other.unlink()
        os.replace(tmp, self._path(j, bf16))

    def get(self, j: int) -> mx.array:
        if self.dir is None:
            return self._mem[j]
        bf16 = self._path(j, True).exists()
        raw = np.load(self._path(j, bf16), mmap_mode="r")
        a = mx.array(np.ascontiguousarray(raw))
        return a.view(mx.bfloat16) if bf16 else a

    def complete(self) -> bool:
        """Every micro-batch is present (in memory or on disk; used to resume a sweep)."""
        if self.dir is None:
            return all(a is not None for a in self._mem)
        return all(self._path(j, True).exists() or self._path(j, False).exists() for j in range(self.n))

    def clear(self) -> None:
        self._mem = [None] * self.n
        if self.dir is not None:
            shutil.rmtree(self.dir, ignore_errors=True)


def stream_eval(model: nn.Module, batches: list[mx.array], consume: Callable[[int, mx.array], None], *,
                prepare_block: BlockHook | None = None, prepare_head: Callable[[nn.Module], None] | None = None,
                release: bool = True, spill_dir: str | Path | None = None) -> dict:
    """Block-sequential forward of ``model`` over token micro-batches, then ``consume(j, logits_j)`` per batch.

    Every batch is embedded, then block i runs over ALL batches (``prepare_block(i, block)`` first, e.g. to install
    decoded weights; then the block's parameters are released); finally the final norm + LM head is applied ONE
    batch at a time and the logits are handed to ``consume`` (never all logits at once). Each batch goes through
    exactly the operations of ``model(batch)`` (same causal mask, same dtypes), so the logits are those of the
    whole-model forward. ``prepare_head(model)`` runs before the embedding (e.g. to install a decoded embedding).
    Peak memory: one block + the hidden states (on disk with ``spill_dir``) + one batch's logits."""
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    if prepare_head is not None:
        prepare_head(model)
    hs = HiddenBuffer(len(batches), spill_dir)
    for j, b in enumerate(batches):
        h = embed(model, b)
        mx.eval(h)
        hs.put(j, h)
    block_seconds = []
    for i, block in enumerate(model_blocks(model)):
        tb = time.perf_counter()
        if prepare_block is not None:
            prepare_block(i, block)
        for j in range(len(batches)):
            h = hs.get(j)
            out = block(h, create_attention_mask(h, None))
            mx.eval(out)
            hs.put(j, out)
            del h, out
        if release:
            _release(block)
        mx.clear_cache()
        block_seconds.append(time.perf_counter() - tb)
    for j in range(len(batches)):
        lg = head(model, hs.get(j))
        consume(j, lg)
        del lg
    hs.clear()
    mx.clear_cache()
    return {"peak_memory_bytes": int(mx.get_peak_memory()), "block_seconds": block_seconds,
            "seconds": time.perf_counter() - t0}


def fresh_block_weights(model: nn.Module, model_dir: str | Path, i: int) -> dict[str, mx.array]:
    """Lazily loaded (memory-mapped, not yet read) checkpoint parameters of block ``i`` in the mlx-lm parameter
    namespace of the block, exactly as ``mlx_lm.utils.load_model`` maps them (``model.sanitize``, e.g. MoE experts
    stacked into ``switch_mlp``). Fresh arrays on every call: a block released after use (``_release``) can be
    loaded again (reverse sweeps) without keeping earlier blocks resident."""
    import glob

    weights: dict[str, mx.array] = {}
    for wf in sorted(glob.glob(str(Path(model_dir) / "model*.safetensors"))):
        weights.update(mx.load(wf))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    prefix = f"{block_prefix(i)}."
    return {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}


def load_block(model: nn.Module, model_dir: str | Path, i: int) -> nn.Module:
    """(Re)load block ``i``'s checkpoint parameters into ``model`` (lazily) and return the block."""
    from mlx.utils import tree_unflatten

    block = model_blocks(model)[i]
    block.update(tree_unflatten(list(fresh_block_weights(model, model_dir, i).items())))
    return block
