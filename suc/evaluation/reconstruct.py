"""Quality-track evaluator: whole-model codec reconstruction, actual size accounting, PPL / KL (spec §25).

Every eligible linear weight is replaced by codec.decode(codec.encode(w)) one tensor at a time
(read from the SafetensorsStore, so no second full copy of the model is held), then quality is
measured on the modified model: WikiText-2 perplexity (FINAL-TEST set, never used for tuning, §32)
and next-token KL vs the original model on held-out VALIDATION sequences of the Phase 2 corpus.

Size accounting covers ALL checkpoint tensors: encoded tensors at the bits their codec reports,
every other tensor at its checkpoint dtype. Tensor names are in the checkpoint namespace; MoE
experts "…mlp.experts.{e}.<proj>.weight" are rows of the stacked mlx-lm "…mlp.switch_mlp.<proj>.weight".
"""
from __future__ import annotations

import inspect
import re
import time
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from suc.analysis.sensitivity import get_weight, logit_divergence, set_weight, weight_error
from suc.evaluation.perplexity import perplexity
from suc.io.safetensors_store import SafetensorsStore
from suc.models.adapters import LinearSpec
# rotation groups / seed derivation: suc.quant.seeds (shared with candidate scoring, audit F08); re-exported here
from suc.quant.seeds import _ROT_GROUP, accepts_rotation_seeds, rotation_group, rotation_kwargs, rotation_seeds  # noqa: F401
from suc.quant.stacks import encode_stack, shares_codebooks, stack_of

if TYPE_CHECKING:
    from suc.quant.codecs import Codec

_EXPERT = re.compile(r"^(.*)\.experts\.(\d+)\.(\w+)\.weight$")
_DTYPE_BITS = {"BF16": 16, "F16": 16, "F32": 32, "F64": 64, "I8": 8, "U8": 8, "BOOL": 8,
               "I16": 16, "U16": 16, "I32": 32, "U32": 32, "I64": 64, "U64": 64}
_PARTS = ("indices", "codebooks", "scales", "residual", "metadata")
_EMBED, _LM_HEAD = "model.embed_tokens.weight", "lm_head.weight"


def _stable_index(name: str) -> int:
    """Per-tensor seed offset that depends only on the name (not on the order/subset of specs)."""
    return zlib.crc32(name.encode())


# ---------------------------------------------------------------- reconstruction
def _flush_experts(model: nn.Module, pending: dict[str, dict[int, mx.array]]) -> None:
    """Write buffered expert rows into their stacked tensors, one copy per stack."""
    for stack, rows in pending.items():
        cur = get_weight(model, stack)
        new = mx.stack([rows.get(r, cur[r]) for r in range(cur.shape[0])])
        set_weight(model, stack, new)
        mx.eval(get_weight(model, stack))
    pending.clear()


def _work_items(specs: Sequence[LinearSpec], expert_stacks: bool) -> list[tuple[str | None, list[LinearSpec]]]:
    """(stack key or None, specs) in processing order. Without expert_stacks every spec is its own item (the original
    order); with it the experts of one stack form one item at the position of the stack's first expert."""
    if not expert_stacks:
        return [(None, [s]) for s in specs]
    items: list[tuple[str | None, list[LinearSpec]]] = []
    where: dict[str, int] = {}
    for s in specs:
        st = stack_of(s.name)
        if st is None:
            items.append((None, [s]))
        elif st[0] in where:
            items[where[st[0]]][1].append(s)
        else:
            where[st[0]] = len(items)
            items.append((st[0], [s]))
    for key, ss in items:
        if key is not None:
            ss.sort(key=lambda s: stack_of(s.name)[1])
    return items


def reconstruct_model(model: nn.Module, store: SafetensorsStore, specs: Sequence[LinearSpec],
                      codec_for: Callable[[LinearSpec], "Codec | None"], *, seed: int,
                      col_weights: Mapping[str, mx.array] | None = None,
                      importance: Mapping[str, mx.array] | None = None, log_every: int = 0,
                      keep_encodings: bool = False, rotation_groups: bool = False,
                      expert_stacks: bool = False, mixed_stacks: bool = False,
                      hessians: Mapping[str, mx.array] | None = None) -> dict:
    """Replace each spec's weight in `model` by its codec reconstruction (in place).

    codec_for(spec) -> None keeps the original tensor (then it is not listed and counts as
    unencoded in model_size). Encode seed = seed + crc32(name). col_weights are keyed by the
    tensor name without ".weight". importance (e.g. empirical Fisher, keyed by the full tensor name)
    is passed as encode(..., importance=...) only to codecs whose encode accepts it. rotation_groups=True passes
    encode(..., seed_in=rotation_seeds(name, seed)["seed_in"]) for q/k/v and gate/up (codecs that accept it), so
    tensors reading the same input share the input rotation V and the runtime rotates that input once (README
    §11.5); the default False keeps the per-tensor seeds of earlier results. Expert rows are buffered per stacked tensor and written once
    the layer is finished (one stack copy per layer/projection instead of one per expert); an
    expert's row is never modified before its own turn.

    expert_stacks=True processes the experts of one MoE projection stack together (suc.quant.stacks): every expert
    of a stack must get the same codec (ValueError otherwise), and a codec with codebook scope "expert_stack" is
    fitted once per stack on a pooled sample (shared codebooks and rotation seeds; encode_stack). Codecs with scope
    "tensor" are encoded per expert exactly as without expert_stacks (only the processing order differs).
    hessians (keyed like col_weights, full E[x x^T] of the input) go to codecs whose encode takes hessian= (LDLQ).
    mixed_stacks=True (allocation unit "expert") lets the experts of a stack have different codecs: each shared-scope
    codec is fitted on the pooled sample of the WHOLE stack (as when it was scored) and encodes its own experts.
    """
    t_start = time.perf_counter()
    tensors: list[dict] = []
    encodings: dict = {}
    pending: dict[str, dict[int, mx.array]] = {}
    enc_params = enc_bits = 0
    last_layer = None
    n_done = 0

    def record(spec: LinearSpec, codec, w: mx.array, e, t0: float) -> None:
        nonlocal enc_params, enc_bits
        w_hat = codec.decode(e)
        bits = {k: int(v) for k, v in e.bits().items()}
        if sum(bits[k] for k in _PARTS) != bits["total"]:
            raise ValueError(f"{codec.name}: bits parts do not sum to total for {spec.name}: {bits}")
        err = weight_error(w, w_hat)
        if (m := _EXPERT.match(spec.name)) is not None:
            stack = f"{m.group(1)}.switch_mlp.{m.group(3)}.weight"
            row = w_hat.astype(get_weight(model, stack).dtype)
            mx.eval(row)
            pending.setdefault(stack, {})[int(m.group(2))] = row
        else:
            set_weight(model, spec.name, w_hat)
            mx.eval(get_weight(model, spec.name))
        tensors.append({"name": spec.name, "role": spec.role, "layer": spec.layer, "expert": spec.expert,
                        "codec": codec.name, "shape": list(w.shape), "bits": bits, "bpw": e.bpw(),
                        "rel_mse": err["rel_mse"], "seconds": time.perf_counter() - t0})
        enc_params += w.size
        enc_bits += bits["total"]
        if keep_encodings:
            encodings[spec.name] = e
        if len(tensors) % 64 == 0:
            mx.clear_cache()
        if log_every and len(tensors) % log_every == 0:
            t = tensors[-1]
            print(f"[reconstruct] {n_done}/{len(specs)} {t['name']} bpw={t['bpw']:.3f} rel_mse={t['rel_mse']:.3e}",
                  flush=True)

    def encode_one(spec: LinearSpec, codec) -> None:
        t0 = time.perf_counter()
        w = store.load(spec.name)
        cw = col_weights.get(spec.name[:-7]) if col_weights is not None else None
        kw = {"importance": importance.get(spec.name) if importance is not None else None} \
            if "importance" in inspect.signature(codec.encode).parameters else {}
        kw.update(rotation_kwargs(codec, spec.name, seed, rotation_groups))   # = candidate scoring (suc.quant.seeds)
        if (hessians is not None and getattr(codec, "ldlq", True) and spec.name[:-7] in hessians
                and "hessian" in inspect.signature(codec.encode).parameters):
            kw["hessian"] = hessians[spec.name[:-7]]     # full input second moment (LDLQ, suc.quant.trellis)
        src = store.quant_source(spec.name) if hasattr(store, "quant_source") else None
        if hasattr(codec, "encode_source") and src is not None and src.kind == codec.name:
            # native pass-through (MXFP4 experts): the checkpoint bytes verbatim; must decode to the reference exactly
            e = codec.encode_source(*store.load_source(spec.name)[1:])
            if not mx.array_equal(codec.decode(e), w.astype(mx.float32)).item():
                raise ValueError(f"{spec.name}: native {codec.name} pass-through does not decode to the reference")
        else:
            e = codec.encode(w, seed=seed + _stable_index(spec.name), col_weights=cw, **kw)
        record(spec, codec, w, e, t0)

    for key, item in _work_items(specs, expert_stacks):
        if pending and item[0].layer != last_layer:
            _flush_experts(model, pending)
        last_layer = item[0].layer
        if key is None:
            n_done += 1
            codec = codec_for(item[0])
            if codec is not None:
                encode_one(item[0], codec)
            continue
        codecs = [codec_for(s) for s in item]
        if (not mixed_stacks and len({id(c) for c in codecs}) != 1
                and len({getattr(c, "name", None) for c in codecs}) != 1):
            raise ValueError(f"{key}: the experts of a stack must share one codec, got "
                             f"{sorted({getattr(c, 'name', None) for c in codecs}, key=str)}")
        n_done += len(item)
        by_codec: dict[int, tuple[object, list[LinearSpec]]] = {}
        for s, c in zip(item, codecs):
            if c is not None:
                by_codec.setdefault(id(c), (c, []))[1].append(s)
        by_name = {s.name: s for s in item}
        for codec, sub in by_codec.values():
            if not shares_codebooks(codec):
                for s in sub:
                    encode_one(s, codec)
                continue
            t0 = time.perf_counter()
            for name, w, e in encode_stack(codec, store.load, [s.name for s in sub], seed=seed, stack=key,
                                           col_weights=col_weights, importance=importance,
                                           fit_names=[s.name for s in item]):
                record(by_name[name], codec, w, e, t0)
                t0 = time.perf_counter()
    _flush_experts(model, pending)
    mx.clear_cache()
    out = {"tensors": tensors, "encoded_params": enc_params, "encoded_bits": enc_bits,
           "seconds": time.perf_counter() - t_start}
    if keep_encodings:
        out["encodings"] = encodings   # name -> Encoded (compressed form, e.g. for Q6 fine-tuning)
    return out


# ---------------------------------------------------------------- size
def _same_bytes(store: SafetensorsStore, a: str, b: str, chunk: int = 1 << 24) -> bool:
    ia, ib = store.info(a), store.info(b)
    if (ia.dtype, ia.shape, ia.nbytes) != (ib.dtype, ib.shape, ib.nbytes):
        return False
    with open(ia.shard, "rb") as fa, open(ib.shard, "rb") as fb:
        fa.seek(ia.offset)
        fb.seek(ib.offset)
        for _ in range(0, ia.nbytes, chunk):
            n = min(chunk, ia.nbytes - fa.tell() + ia.offset)
            if fa.read(n) != fb.read(n):
                return False
    return True


def model_size(store: SafetensorsStore, recon: dict) -> dict:
    """Actual whole-model size: encoded tensors at codec bits, all others at checkpoint dtype bits.

    A checkpoint lm_head byte-identical to the embedding (tied embeddings stored twice) is counted
    once, unless the lm_head itself was encoded. bpp = bits_total / parameter_count.
    """
    enc = {t["name"]: t for t in recon["tensors"]}
    skip = set()
    if _LM_HEAD in store and _EMBED in store and _LM_HEAD not in enc and _same_bytes(store, _LM_HEAD, _EMBED):
        skip.add(_LM_HEAD)
    breakdown = dict.fromkeys((*_PARTS, "unencoded"), 0)
    n_total = n_enc = 0
    for name in store.names():
        if name in skip:
            continue
        info = store.info(name)
        n = 1
        for d in info.shape:
            n *= d
        n_total += n
        if name in enc:
            n_enc += n
            for k in _PARTS:
                breakdown[k] += enc[name]["bits"][k]
        else:
            breakdown["unencoded"] += n * _DTYPE_BITS[info.dtype]
    if set(enc) - set(store.names()):
        raise KeyError(f"encoded tensors missing from the store: {sorted(set(enc) - set(store.names()))[:5]}")
    bits_total = sum(breakdown.values())
    enc_bits = bits_total - breakdown["unencoded"]
    return {"parameter_count": n_total, "bits_total": bits_total, "bpp": bits_total / n_total,
            "bytes_total": -(-bits_total // 8), "encoded_params": n_enc,
            "encoded_bpp": enc_bits / n_enc if n_enc else float("nan"),
            "other_params": n_total - n_enc, "breakdown": breakdown}


# ---------------------------------------------------------------- quality
def reference_logits(model: nn.Module, tokens: mx.array, chunk: int = 4) -> list[mx.array]:
    """Logits of `model` on tokens (B, T), as bf16 chunks of `chunk` rows (bounded memory)."""
    out = []
    for s in range(0, tokens.shape[0], chunk):
        lg = model(tokens[s:s + chunk]).astype(mx.bfloat16)
        mx.eval(lg)
        out.append(lg)
    return out


KL_P95_DEFINITION = "global"     # kl_p95 = 95th percentile over all scored tokens (kl_p95_chunk_mean: the old value)


def aggregate_kl(parts: list[tuple[dict, int]]) -> dict:
    """Combine logit_divergence(..., return_tokens=True) results of several (sub)chunks: token-weighted means of the
    mean fields (exact), kl_p95 = the GLOBAL 95th percentile of the per-token KL, and, for continuity with results
    before 2026-10-09, kl_p95_chunk_mean = the token-weighted mean of per-chunk p95 values (what kl_p95 used to be:
    not a percentile, and dependent on the chunking)."""
    keys = ("kl_mean", "top1_agreement", "ce_ref", "ce_test")
    n_tok = sum(n for _, n in parts)
    out = {k: sum(d[k] * n for d, n in parts) / n_tok for k in keys}
    out["kl_p95_chunk_mean"] = sum(d["kl_p95"] * n for d, n in parts) / n_tok
    out["kl_p95"] = float(np.percentile(np.concatenate([d["kl_tokens"] for d, _ in parts]), 95))
    out["kl_p95_definition"] = KL_P95_DEFINITION
    out["delta_ce"] = out["ce_test"] - out["ce_ref"]
    out["tokens"] = n_tok
    return out


def _kl(model: nn.Module, kl_tokens: mx.array, ref_logits: list[mx.array], chunk: int) -> dict:
    """KL / top-1 / CE vs the reference over (sub)chunks (aggregate_kl: exact means, global kl_p95)."""
    if sum(r.shape[0] for r in ref_logits) != kl_tokens.shape[0]:
        raise ValueError("ref_logits rows do not match kl_tokens")
    parts, row = [], 0
    for ref in ref_logits:
        for s in range(0, ref.shape[0], chunk):
            r = ref[s:s + chunk]
            tok = kl_tokens[row + s: row + s + r.shape[0]]
            test = model(tok).astype(r.dtype)
            d = logit_divergence(r[:, :-1], test[:, :-1], tok[:, 1:], return_tokens=True)
            parts.append((d, tok.shape[0] * (tok.shape[1] - 1)))
            del test
        row += ref.shape[0]
    return aggregate_kl(parts)


def evaluate_quality(model: nn.Module, *, ppl_tokens: mx.array, seq_len: int, max_tokens: int | None,
                     kl_tokens: mx.array | None = None, ref_logits: list[mx.array] | None = None,
                     chunk: int = 4) -> dict:
    """{"perplexity": perplexity(...) dict, "kl": KL/top-1/ΔCE vs ref_logits on kl_tokens, or None}."""
    ppl = perplexity(model, ppl_tokens, seq_len, max_tokens)
    kl = None
    if kl_tokens is not None:
        if ref_logits is None:
            raise ValueError("kl_tokens given without ref_logits")
        kl = _kl(model, kl_tokens, ref_logits, chunk)
    return {"perplexity": ppl, "kl": kl}


def load_validation_tokens(artifact: str | Path, n: int) -> mx.array:
    """First n rows of the held-out "validation" partition of a Phase 2 *_data.safetensors artifact."""
    val = mx.load(str(artifact))["validation"]
    if n > val.shape[0]:
        raise ValueError(f"requested {n} validation sequences, artifact has {val.shape[0]}")
    return val[:n]
