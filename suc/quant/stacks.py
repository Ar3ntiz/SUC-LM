"""Expert stacks: the E routed experts of one MoE projection in one layer as ONE representation unit (D6).

A Qwen3-MoE checkpoint stores every expert projection as its own tensor ("<p>.experts.{e}.<proj>.weight"); mlx-lm and
the runtime (suc.kernels.suc_switch.SucSwitchLinear) hold them as one stacked weight "<p>.switch_mlp.<proj>.weight"
(E, out, in). The STACK KEY of an expert tensor is that stacked name. Treating the stack as one unit gives

  allocation  one candidate list per stack (bits and Fisher loss summed over its experts); every expert of a stack
              gets the same codec, which the runtime requires anyway (units / expand_plan).
  codebooks   with codebook scope "expert_stack" (vq / additive bases, optionally residual-wrapped) the codebooks are
              fitted ONCE per stack on a pooled sample of prepared blocks of all experts, then every expert only gets
              its indices, row scales and residual (encode_stack). One k-means per stack instead of one per expert.

Shared-stack encoding (encode_stack):
  rotation  every expert of a stack uses the SAME rotation seeds (stack_seeds): seed_in is shared by the gate and up
            stacks of a layer (they read the same token input, so the runtime rotates it once for all routed
            experts), seed_out per stack. Experts of one stack then live in one rotated basis, which is what makes a
            pooled codebook well defined; the per-row Hadamard sign tables of the runtime collapse to one row.
  sampling  pass 1 loads each expert (one at a time, D8), prepares it exactly as encode does (rotation, row-RMS
            normalisation) and keeps ceil(fit_blocks / E) blocks drawn uniformly without replacement with the
            expert's own encode seed; the pooled sample (<= fit_blocks + E blocks) is fitted with the codec's own fit
            (vq: k-means; additive: greedy residual fit + refine_iters joint refinement on the sample).
  encode    pass 2 re-loads each expert and encodes it against the fixed codebooks (vq: nearest code; additive:
            greedy assignment + refine_iters coordinate-descent re-assignment passes). Row scales and the sparse
            residual stay per expert. Expert 0 owns the codebooks (counted in its bits()); the others reference them.
Encode seeds stay per expert (seed + crc32(name)), as in suc.evaluation.reconstruct.
"""
from __future__ import annotations

import inspect
import math
import re
import zlib
from typing import Callable, Iterator, Literal, Mapping, Sequence

import mlx.core as mx
import numpy as np

from suc.quant.codecs import Encoded, _BlockCodebookBase

Scope = Literal["tensor", "expert_stack"]
SCOPES: tuple[str, ...] = ("tensor", "expert_stack")
_EXPERT = re.compile(r"^(.*)\.experts\.(\d+)\.(\w+)\.weight$")
_GATE_UP = ("gate_proj", "up_proj")


def stack_of(name: str) -> tuple[str, int] | None:
    """(stack key "<p>.switch_mlp.<proj>.weight", expert index) of an expert tensor name, else None."""
    if (m := _EXPERT.match(name)) is None:
        return None
    return f"{m.group(1)}.switch_mlp.{m.group(3)}.weight", int(m.group(2))


def units(names: Sequence[str]) -> dict[str, list[str]]:
    """Allocation units in first-appearance order: every non-expert tensor is its own unit ({name: [name]}); the
    experts of one stack form one unit keyed by the stack key, with its tensor names sorted by expert index."""
    out: dict[str, list[str]] = {}
    for n in names:
        st = stack_of(n)
        out.setdefault(n if st is None else st[0], []).append(n)
    for k, v in out.items():
        if len(v) > 1:
            v.sort(key=lambda n: stack_of(n)[1])
    return out


def expand_plan(plan: Mapping[str, object], unit_map: Mapping[str, Sequence[str]]) -> dict[str, object]:
    """Per-unit plan -> per-tensor plan (every tensor of a unit gets the unit's value)."""
    return {n: v for u, v in plan.items() for n in unit_map.get(u, [u])}


# ------------------------------------------------------------------------------------------- codebook scope

def set_codebook_scope(codec, scope: str):
    """Tag a codec object with its codebook scope (returns it). "expert_stack" needs a vq / additive base (shared
    codebooks + rotation) or a tcq base (shared input rotation only: shares_rotation)."""
    if scope not in SCOPES:
        raise ValueError(f"unknown codebook scope {scope!r}; expected one of {SCOPES}")
    if scope == "expert_stack" and _block_base(codec) is None and _trellis_base(codec) is None:
        raise ValueError(f"codec {getattr(codec, 'name', codec)!r}: codebook scope 'expert_stack' needs a vq / "
                         "additive / tcq base codec")
    codec.codebook_scope = scope
    return codec


def _trellis_base(codec):
    base = getattr(codec, "base", codec)
    return base if getattr(base, "name", None) == "tcq" else None


def shares_rotation(codec) -> bool:
    """tcq with scope "expert_stack": every expert of a stack is encoded with the stack's input rotation seed
    (stack_seeds: shared by the gate and up stacks of a layer), so the runtime rotates the routed input once."""
    return codebook_scope(codec) == "expert_stack" and _trellis_base(codec) is not None


def codebook_scope(codec) -> str:
    return getattr(codec, "codebook_scope", "tensor")


def _block_base(codec) -> _BlockCodebookBase | None:
    base = getattr(codec, "base", codec)     # ResidualCodec wraps its base codec
    return base if isinstance(base, _BlockCodebookBase) else None


def shares_codebooks(codec) -> bool:
    return codebook_scope(codec) == "expert_stack" and _block_base(codec) is not None


def _derived(seed: int, key: str) -> int:
    return int(np.random.default_rng(seed + zlib.crc32(key.encode())).integers(0, 2 ** 31 - 1))


def stack_seeds(stack: str, seed: int) -> dict[str, int]:
    """Rotation seeds of every expert of `stack` (module docstring): seed_in shared by the gate and up stacks of a
    layer, seed_out per stack; both depend only on (seed, stack key)."""
    prefix, proj = stack[: -len(".weight")].rsplit(".", 1)
    group = f"{prefix}.gate_up" if proj in _GATE_UP else f"{prefix}.{proj}"
    return {"seed_in": _derived(seed, group + ".in"), "seed_out": _derived(seed, f"{prefix}.{proj}.out")}


def encode_stack(codec, load: Callable[[str], mx.array], names: Sequence[str], *, seed: int, stack: str,
                 col_weights: Mapping[str, mx.array] | None = None,
                 importance: Mapping[str, mx.array] | None = None,
                 fit_names: Sequence[str] | None = None) -> Iterator[tuple[str, mx.array, Encoded]]:
    """Encode the experts `names` (expert order) of `stack` with ONE shared codebook set (module docstring).

    Yields (name, w, Encoded) one expert at a time (w as loaded); every Encoded decodes with codec.decode.
    col_weights are keyed by the tensor name without ".weight", importance by the full name (as reconstruct_model).
    fit_names (default: names) are the experts whose pooled sample fits the codebooks: allocation unit "expert" fits
    on the WHOLE stack and encodes only the experts that got this codec, so every encoding equals the one scored
    (same codebooks, same seeds); the first of `names` owns the codebooks."""
    base = _block_base(codec)
    if base is None:
        raise ValueError(f"codec {getattr(codec, 'name', codec)!r} cannot share codebooks")
    if not names:
        return
    fit_names = list(names) if fit_names is None else list(fit_names)
    rot = stack_seeds(stack, seed)
    per = math.ceil(base.fit_blocks / len(fit_names))
    takes_importance = "importance" in inspect.signature(codec.encode).parameters

    def cw(n: str) -> mx.array | None:
        return col_weights.get(n[:-7]) if col_weights is not None else None

    xs, hs = [], []
    for n in fit_names:                      # pass 1: pooled fitting sample (one expert resident at a time)
        x, h = base.sample_blocks(load(n).astype(mx.float32), seed=seed + zlib.crc32(n.encode()), n=per,
                                  col_weights=cw(n), **rot)
        mx.eval(x, *(() if h is None else (h,)))
        xs.append(x)
        hs.append(h)
    if any(h is None for h in hs) and not all(h is None for h in hs):
        raise ValueError(f"{stack}: column weights are given for some experts only")
    hpool = None if hs[0] is None else mx.concatenate(hs)
    cb = base.fit(mx.concatenate(xs), hpool, seed=seed + zlib.crc32(stack.encode()))
    mx.eval(cb)
    del xs, hs, hpool
    for i, n in enumerate(names):            # pass 2: indices / row scales / residual per expert
        w = load(n)
        kw = {"importance": importance.get(n) if importance is not None else None} if takes_importance else {}
        e = codec.encode(w, seed=seed + zlib.crc32(n.encode()), col_weights=cw(n), codebooks=cb,
                         codebook_owner=i == 0, **rot, **kw)
        e.params.get("base", e.params)["codebook_stack"] = stack    # next to codebook_shared / codebook_owner
        yield n, w, e
