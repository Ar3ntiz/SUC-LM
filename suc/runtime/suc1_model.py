"""The Python/MLX runtime of a SUC1 file: an mlx-lm model built from the file alone (the reference for the C++ model
graph of cpp/README.md milestone 3).

load_suc1_model(path) builds the mlx-lm model of `directory.architecture` (no checkpoint needed), in the checkpoint
dtype (torch_dtype), and fills it from the file:
  - raw tensors (norms, q/k-norm) as stored;
  - other non-SUC tensors (affine / scalar / bf16: the embedding, a kept LM head, kept projections) as their
    reference decode (SucModel.decode_tensor, float32) cast to the model dtype, i.e. the representation the
    compress pipeline evaluates (suc.pipeline.compress._install); with affine_modules=True an affine embedding /
    LM head (bits 2/4/8, MLX group size) instead runs from its stored codes (AffineEmbedding / AffineLinear: the
    SUC1 packed layout is the MLX affine layout);
  - vq / additive tensors as SUC modules (suc.runtime.compressed_model.convert_to_compressed) on the stored codes;
  - native MXFP4 pass-through tensors (representation "mxfp4") on their stored checkpoint codes through MLX's mxfp4
    quantized matmul / gather_qmm (suc.kernels.mxfp4), alone or inside a mixed-codec expert stack;
  - lookup tables the model class flags (`is_table_tensor`: qwen4_exp n-gram shards, raw or row-affine) are attached
    as memory-mapped views (`attach_suc1`) and never made resident.
group / fuse are passed to convert_to_compressed; group=False keeps one SucLinear / SucResidualLinear per projection,
the execution plan of the C++ runtime (metal::Linear), so a decode step can be compared bit for bit.

Memory (D8 for the runtime): the file stays memory-mapped. SUC tensors are converted one transformer block at a time
from SucModel.encoded_packed (the packed index words are copied once from the mmap, never unpacked / repacked), so
the peak is the final resident set plus one block's transient (one expert stack: about twice its packed size);
decoded tensors are evaluated one at a time (one float32 copy transient). estimate_resident_bytes(sm) predicts the
resident set from the directory alone.

step_hidden(model, token, cache) runs one token as mlx-lm does in decode (mask None, KV cache) and also returns the
hidden state after the embedding and after every block.
"""
from __future__ import annotations

import gc
import math
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from suc.io.suc1 import SucModel
from suc.runtime.compressed_model import (convert_to_compressed, expert_stack, group_projections, merge_reports)

_DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
_LAYER = re.compile(r"^model\.layers\.(\d+)\.")
_MLX_AFFINE = {"bits": (2, 4, 8), "group_size": (32, 64, 128)}
_AFFINE_TARGETS = ("model.embed_tokens.weight", "lm_head.weight")
_BYTES = {"bf16": 2, "f16": 2, "f32": 4, "f64": 8, "i8": 1, "u8": 1, "i16": 2, "u16": 2, "i32": 4, "u32": 4,
          "i64": 8, "u64": 8, "bool": 1}


def model_dtype(architecture: dict) -> mx.Dtype:
    return _DTYPES.get(architecture.get("torch_dtype") or "bfloat16", mx.bfloat16)


# ------------------------------------------------------------------------------------------- affine from codes

class _AffineCodes(nn.Module):
    """Packed affine codes (out, in*bits/32) uint32 + per-group scales / biases (out, in/gs) as stored in SUC1 (the
    MLX affine layout for bits 2/4/8). Results are computed in float32 (scales / biases cast up) and returned in
    `dtype`, so the values equal the reference decode q * s + b."""

    def __init__(self, weight: mx.array, scales: mx.array, biases: mx.array, *, group_size: int, bits: int,
                 dtype: mx.Dtype):
        super().__init__()
        self.weight, self.scales, self.biases = weight, scales, biases
        self.group_size, self.bits, self.out_dtype = group_size, bits, dtype
        self.freeze()


class AffineEmbedding(_AffineCodes):
    def __call__(self, ids: mx.array) -> mx.array:
        return mx.dequantize(self.weight[ids], self.scales[ids].astype(mx.float32),
                             self.biases[ids].astype(mx.float32), group_size=self.group_size,
                             bits=self.bits).astype(self.out_dtype)

    def as_linear(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(x, self.weight, self.scales.astype(mx.float32), self.biases.astype(mx.float32),
                                   transpose=True, group_size=self.group_size, bits=self.bits).astype(x.dtype)


class AffineLinear(_AffineCodes):
    def __call__(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(x, self.weight, self.scales.astype(mx.float32), self.biases.astype(mx.float32),
                                   transpose=True, group_size=self.group_size, bits=self.bits).astype(x.dtype)


def _affine_runnable(d: dict) -> bool:
    p = d["params"]
    return (d["representation_type"] == "affine" and d["residual_type"] == "none" and len(d["shape"]) == 2
            and p.get("bits") in _MLX_AFFINE["bits"] and p.get("group_size") in _MLX_AFFINE["group_size"])


def _affine_module(sm: SucModel, name: str, dtype: mx.Dtype, cls):
    d = sm.descriptor(name)
    p = d["params"]

    def arr(key: str) -> mx.array:
        rec = sm.record(name, key)
        a = mx.array(np.array(sm.payload(name, key)))
        return a.view(mx.bfloat16) if rec["dtype"] == "bf16" else a

    m = cls(arr("indices"), arr("scales"), arr("biases"), group_size=int(p["group_size"]), bits=int(p["bits"]),
            dtype=dtype)
    mx.eval(m.parameters())
    return m


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    *path, attr = name[: -len(".weight")].split(".")
    cur = model
    for k in path:
        cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    cur[attr] = module


# ------------------------------------------------------------------------------------------- loader

def _layer_of(name: str) -> int:
    m = _LAYER.match(name)
    return int(m.group(1)) if m else -1


def load_suc1_model(path: str | Path, *, group: bool = False, fuse: bool = False, dtype: mx.Dtype | None = None,
                    affine_modules: bool = False, log=None, skip=None):
    """(mlx-lm model, SucModel, report) for a SUC1 file; see the module docstring. log(str) receives progress.
    skip(name) -> True leaves a vq / additive tensor out entirely (never read, never converted): the routed experts
    of suc.runtime.expert_streaming, which fetches them per call from an ExpertCache."""
    from suc.models.streaming import get_model_classes

    sm = SucModel.load(path, verify=False)
    arch = dict(sm.architecture)
    model_cls, args_cls = get_model_classes(arch)
    model = model_cls(args_cls.from_dict(arch))
    dt = dtype or model_dtype(arch)
    model.set_dtype(dt)
    params = dict(tree_flatten(model.parameters()))
    suc_names: dict[int, list[str]] = {}
    weights, affine = [], []
    expert_rows: dict[str, dict[int, mx.array]] = {}       # non-SUC expert tensors, stacked per switch_mlp weight
    is_table = getattr(model, "is_table_tensor", lambda n: False)
    for name in sm.names():
        d = sm.descriptor(name)
        rep = d["representation_type"]
        if skip is not None and skip(name):
            if rep not in ("vq", "additive", "tcq", "mxfp4"):   # code tensors only (native mxfp4: codes + scales)
                raise ValueError(f"{name}: only vq / additive / tcq / mxfp4 tensors can be skipped (streamed), "
                                 f"got {rep}")
            continue
        if rep in ("vq", "additive", "tcq", "mxfp4"):
            suc_names.setdefault(_layer_of(name), []).append(name)
            continue
        if is_table(name):                    # memory-mapped lookup tables / hash buffers (model.attach_suc1)
            continue
        if affine_modules and name in _AFFINE_TARGETS and name in params and _affine_runnable(d):
            affine.append(name)
            continue
        st = expert_stack(name)
        if name not in params and st is not None and st[0] in params:
            w = (sm.encoded(name) if rep == "raw" else sm.decode_tensor(name)).astype(params[st[0]].dtype)
            mx.eval(w)
            expert_rows.setdefault(st[0], {})[st[1]] = w
            continue
        if name not in params:
            if name == "lm_head.weight" and arch.get("tie_word_embeddings"):
                continue
            raise KeyError(f"{name}: not a parameter of the {arch.get('model_type')} model")
        w = (sm.encoded(name) if rep == "raw" else sm.decode_tensor(name)).astype(params[name].dtype)
        mx.eval(w)                            # one float32 decode transient at a time
        weights.append((name, w))
    for stack, rows in expert_rows.items():
        if sorted(rows) != list(range(params[stack].shape[0])):
            raise KeyError(f"{stack}: not every expert row is stored")
        weights.append((stack, mx.stack([rows[e] for e in range(len(rows))])))
    expert_rows.clear()
    if hasattr(model, "attach_suc1"):
        model.attach_suc1(sm)
    model.load_weights(weights, strict=False)
    loaded = dict(tree_flatten(model.parameters()))
    mx.eval([loaded[n] for n, _ in weights])   # the init weights of the SUC targets are never materialized
    del weights, loaded
    for name in affine:
        cls = AffineEmbedding if name == "model.embed_tokens.weight" else AffineLinear
        _set_module(model, name, _affine_module(sm, name, dt, cls))
    reports = []
    for layer in sorted(suc_names):           # one block's codes in flight at a time
        enc = {n: (sm.encoded(n) if sm.descriptor(n)["representation_type"] == "mxfp4" else sm.encoded_packed(n))
               for n in suc_names[layer]}
        reports.append(convert_to_compressed(model, enc, group=False))
        del enc
        if log is not None:
            log(f"load: block {layer} converted ({len(suc_names[layer])} tensors), active "
                f"{mx.get_active_memory() / 2**30:.2f} GiB")
    report = merge_reports(reports)
    report["groups"] = group_projections(model, fuse=fuse) if group else []
    report["affine_modules"] = affine
    gc.collect()
    mx.clear_cache()
    report["active_memory_after"] = mx.get_active_memory()
    report["memory_estimate"] = estimate_resident_bytes(sm, affine_modules=affine_modules, dtype=dt)
    model.eval()
    return model, sm, report


def estimate_resident_bytes(sm: SucModel, *, affine_modules: bool = False, dtype: mx.Dtype = mx.bfloat16) -> dict:
    """Resident weight bytes of load_suc1_model(sm.path) predicted from the directory alone (no payload read):
      codes      = Σ over vq / additive tensors of packed indices + residual payloads (as stored)
                   + native mxfp4 tensors as stored (codes + E8M0 scales)
                   + row scales as float32 + codebooks as f16 (once per expert stack when shared)
                   + rotation signs (int8, in_features per tensor with an input rotation)
      dense      = Σ over other non-table tensors of numel × model-dtype bytes (raw / decoded)
      affine     = embedding / LM head kept as codes (affine_modules): stored packed + scales + biases
      tables_mmap = bytes of the memory-mapped lookup tables (page cache only, never resident)
    Excludes the KV / recurrent caches, activations, the load transient (one block) and Python objects."""
    from suc.models.streaming import get_model_classes

    try:
        model_cls, _ = get_model_classes(dict(sm.architecture))
        is_table = getattr(model_cls, "is_table_tensor", lambda n: False)
    except Exception:
        is_table = lambda n: False            # noqa: E731
    elt = {mx.bfloat16: 2, mx.float16: 2, mx.float32: 4}[dtype]
    codes = dense = aff = tables = 0
    seen_cb: set[tuple] = set()
    for name in sm.names():
        d = sm.descriptor(name)
        rep = d["representation_type"]
        n_el = math.prod(d["shape"])
        if is_table(name):
            tables += int(d["compressed_size"])
            continue
        if rep == "mxfp4":                     # native codes + E8M0 scales, resident as stored
            codes += int(d["compressed_size"])
            continue
        if rep in ("vq", "additive", "tcq"):
            p = d["params"].get("base", d["params"]) if d["residual_type"] != "none" else d["params"]
            for r in d["arrays"]:
                k = r["key"].removeprefix("base.")
                if k == "row_scales":
                    codes += 4 * math.prod(r["shape"])
                else:
                    codes += int(r["nbytes"])
            st = expert_stack(name)
            key = ((st[0] if st else name), tuple(d.get("codebook_ids") or []))
            if key not in seen_cb:
                seen_cb.add(key)
                for cid in d.get("codebook_ids") or []:
                    c = sm.codebooks()[cid]
                    codes += 2 * c["entries"] * c["vector_length"]
            if p.get("rotate") and p.get("rotate_in", True):
                codes += int(d["shape"][1])
            if p.get("rotate") and p.get("rotate_out", rep != "tcq"):
                codes += int(d["shape"][0])
            continue
        if affine_modules and name in _AFFINE_TARGETS and _affine_runnable(d):
            aff += int(d["compressed_size"])
            continue
        dense += n_el * (_BYTES[d["arrays"][0]["dtype"]] if rep == "raw" and d["arrays"][0]["dtype"]
                         in ("i64", "i32", "u32", "u64") else elt)
    return {"codes": codes, "dense": dense, "affine": aff, "total": codes + dense + aff, "tables_mmap": tables}


def make_cache(model):
    from mlx_lm.models.cache import make_prompt_cache

    return make_prompt_cache(model)


def lm_head(model, h: mx.array) -> mx.array:
    inner = model.model
    if getattr(model.args, "tie_word_embeddings", False) or not hasattr(model, "lm_head"):
        return inner.embed_tokens.as_linear(h)
    return model.lm_head(h)


def step_hidden(model, token: int, cache) -> tuple[mx.array, list[mx.array]]:
    """(logits (vocab,), [h0, h1, …, h_L] each (hidden,)) for one token at the cache position (decode, mask None)."""
    if hasattr(model, "step_hidden"):
        return model.step_hidden(token, cache)
    inner = model.model
    h = inner.embed_tokens(mx.array([[int(token)]], dtype=mx.uint32))
    hs = [h]
    for layer, c in zip(inner.layers, cache):
        h = layer(h, None, c)
        hs.append(h)
    logits = lm_head(model, inner.norm(h))
    return logits[0, -1], [x[0, -1] for x in hs]


def run_tokens(model, tokens: list[int], *, hidden: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """Token-by-token pass over `tokens` (teacher forcing): logits (n, vocab) float32 and hidden (n, L+1, hidden)."""
    cache = make_cache(model)
    out_l, out_h = [], []
    for t in tokens:
        logits, hs = step_hidden(model, t, cache)
        mx.eval(logits, hs)
        out_l.append(np.array(logits.astype(mx.float32)))
        if hidden:
            out_h.append(np.stack([np.array(x.astype(mx.float32)) for x in hs]))
    return np.stack(out_l), (np.stack(out_h) if hidden else None)


def greedy(model, prompt: list[int], steps: int) -> list[int]:
    """Greedy generation, prompt fed token by token (the C++ prefill), `steps` new tokens."""
    cache = make_cache(model)
    nxt = None
    for t in prompt:
        logits, _ = step_hidden(model, t, cache)
        nxt = int(mx.argmax(logits).item())
    out = []
    for _ in range(steps):
        out.append(nxt)
        if len(out) == steps:
            break
        logits, _ = step_hidden(model, nxt, cache)
        nxt = int(mx.argmax(logits).item())
    return out


def kl_rows(p_logits: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
    """KL(softmax(p) || softmax(q)) per row, float64."""
    p = p_logits.astype(np.float64)
    q = q_logits.astype(np.float64)
    lp = p - p.max(-1, keepdims=True)
    lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
    lq = q - q.max(-1, keepdims=True)
    lq -= np.log(np.exp(lq).sum(-1, keepdims=True))
    return (np.exp(lp) * (lp - lq)).sum(-1)


__all__ = ["load_suc1_model", "estimate_resident_bytes", "AffineEmbedding", "AffineLinear", "make_cache", "step_hidden",
           "run_tokens", "greedy", "kl_rows", "lm_head"]
