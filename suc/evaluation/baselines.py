"""Conventional baseline representations (BF16/FP16, MLX affine) and size/quality reporting."""
from __future__ import annotations

from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

_CASTS = {"bf16": mx.bfloat16, "fp16": mx.float16}


def apply_representation(model, kind: str, bits: int | None = None, group_size: int | None = None):
    """Apply a baseline representation in place and return the model.

    - "bf16"/"fp16": cast every floating parameter.
    - "mlx_affine": MLX affine (per-group scale+bias in the model dtype), mirroring
      `mlx_lm.utils.quantize_model`: every module with `to_quantized` (Linear, Embedding, MoE
      SwitchLinear) whose input dim is divisible by group_size (default 64); bits default 4; the
      model's own `quant_predicate` is honoured (Qwen3-MoE keeps the router at 8 bits).
    """
    if kind in _CASTS:
        model.set_dtype(_CASTS[kind])
    elif kind == "mlx_affine":
        gs, b = group_size or 64, bits or 4

        model_pred = getattr(model, "quant_predicate", None)

        def pred(path, m):
            if not hasattr(m, "to_quantized") or m.weight.shape[-1] % gs != 0:
                return False
            return model_pred(path, m) if model_pred is not None else True

        nn.quantize(model, group_size=gs, bits=b, mode="affine", class_predicate=pred)
    else:
        raise ValueError(f"unknown representation kind: {kind!r}")
    # One leaf module at a time: a lazily loaded model (suc.models.streaming.load_lazy) then never holds more than
    # one module's source weights next to the converted ones (models larger than RAM in BF16, e.g. 180 B qwen4_exp).
    for _, m in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        mx.eval(m.parameters())
    mx.eval(model.parameters())
    return model


def _is_quantized(m) -> bool:
    return hasattr(m, "bits") and hasattr(m, "scales")  # QuantizedLinear / Embedding / SwitchLinear


def size_report(model) -> dict:
    """Actual storage of the model's parameters as currently represented (spec §25).

    parameter_count is the LOGICAL weight count (quantized weights are unpacked: cols*32/bits).
    bytes_weights_packed: packed quantized weights; bytes_scales_biases: their per-group scales and
    biases; bytes_other: every non-quantized parameter (norms, unquantized layers, linear biases).
    bpp = bytes_total*8/parameter_count, i.e. including all scale/bias overhead.
    """
    n_logical = n_quant = 0
    by_dtype: dict[str, int] = defaultdict(int)
    packed = sb = other = 0
    for _, m in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        for name, v in tree_flatten(m.parameters()):
            by_dtype[str(v.dtype)] += v.nbytes
            if _is_quantized(m) and name == "weight":
                logical = v.size * 32 // m.bits  # also 3-D stacked expert weights
                n_logical += logical
                n_quant += logical
                packed += v.nbytes
            elif _is_quantized(m) and name in ("scales", "biases"):
                sb += v.nbytes
            else:
                n_logical += v.size
                other += v.nbytes
    total = packed + sb + other
    return {
        "parameter_count": n_logical, "parameter_count_quantized": n_quant,
        "bytes_total": total, "bytes_by_dtype": dict(by_dtype),
        "bytes_weights_packed": packed, "bytes_scales_biases": sb, "bytes_other": other,
        "bpp": total * 8 / n_logical, "fp16_bytes": n_logical * 2,
    }


def logits_kl(ref_logits: mx.array, test_logits: mx.array) -> dict:
    """Per-token KL(P_ref || P_test) over the last axis, in float32; plus top-1 agreement."""
    v = ref_logits.shape[-1]
    r = ref_logits.reshape(-1, v).astype(mx.float32)
    t = test_logits.reshape(-1, v).astype(mx.float32)
    lp = r - mx.logsumexp(r, axis=-1, keepdims=True)
    lq = t - mx.logsumexp(t, axis=-1, keepdims=True)
    kl = (mx.exp(lp) * (lp - lq)).sum(axis=-1)
    agree = (mx.argmax(r, axis=-1) == mx.argmax(t, axis=-1)).astype(mx.float32).mean()
    return {"kl_mean": kl.mean().item(), "kl_max": kl.max().item(),
            "top1_agreement": agree.item(), "tokens": int(r.shape[0])}
