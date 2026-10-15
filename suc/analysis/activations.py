"""Streaming per-linear-layer input statistics over calibration data (spec §9 T2.2, D5).

For every linear layer the input moments over tokens are accumulated without storing
activations: mean, variance, max |x| and the GPTQ-style Hessian proxy H = E[x xᵀ] (diagonal
always, full matrix on request). Statistics are keyed by the checkpoint tensor name without
".weight" so they join with ``suc.models.adapters.LinearSpec.name``; MoE experts are keyed
per expert ("model.layers.3.mlp.experts.7.up_proj") even though mlx-lm stacks them.

Internal state is (count, mean, centered sum of squares, max |x|, centered co-moment), merged
exactly across updates with the pairwise formulas of Chan et al.; all math in float32.

Sibling linears that consume the SAME input tensor share one ``InputMoments`` object, which is
stored under every sibling name: gate_proj/up_proj (dense MLP and each MoE expert) and
q_proj/k_proj/v_proj. Only the first sibling called in a forward records; the others check that
they receive the identical array object and raise otherwise, so a non-standard block fails loudly
instead of silently mixing inputs.

Instrumentation swaps the ``__class__`` of each recorded module for a recording subclass (no
change to the parameter tree, names or weights); ``remove`` restores the original classes.
State is evaluated after every update so no activation graph is retained across batches.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwitchGLU

_SHARED_INPUT_GROUPS = (("gate_proj", "up_proj"), ("q_proj", "k_proj", "v_proj"))
_PCTS = {"p50": 50.0, "p95": 95.0, "p99": 99.0}


class InputMoments:
    """Streaming first/second moments of a layer's input over rows (tokens). Exact merge across updates (Chan et al.)."""

    def __init__(self, dim: int, full_second_moment: bool = False):
        self.dim = dim
        self.full = full_second_moment
        self.count = 0
        self._mean = mx.zeros((dim,), mx.float32)
        self._m2 = mx.zeros((dim,), mx.float32)          # sum (x - mean)^2
        self._max = mx.zeros((dim,), mx.float32)
        self._c2 = mx.zeros((dim, dim), mx.float32) if full_second_moment else None  # sum (x-mean)(x-mean)^T

    # -- accumulation --------------------------------------------------------------------------
    def _merge_stats(self, n: int, mean: mx.array, m2: mx.array, max_abs: mx.array, c2: mx.array | None) -> None:
        if n == 0:
            return
        if self.count == 0:
            self._mean, self._m2, self._max = mean, m2, max_abs
            if self.full:
                self._c2 = c2
        else:
            na, tot = self.count, self.count + n
            delta = mean - self._mean
            f = na * n / tot
            self._mean = self._mean + delta * (n / tot)
            self._m2 = self._m2 + m2 + delta * delta * f
            self._max = mx.maximum(self._max, max_abs)
            if self.full:
                self._c2 = self._c2 + c2 + mx.outer(delta, delta) * f
        self.count += n
        mx.eval(self._state())

    def _state(self) -> list[mx.array]:
        return [self._mean, self._m2, self._max] + ([self._c2] if self.full else [])

    def update(self, x: mx.array) -> None:
        rows = x.reshape(-1, self.dim).astype(mx.float32)
        n = rows.shape[0]
        if n == 0:
            return
        mean = mx.mean(rows, axis=0)
        d = rows - mean
        self._merge_stats(n, mean, mx.sum(d * d, axis=0), mx.max(mx.abs(rows), axis=0),
                          d.T @ d if self.full else None)

    def merge(self, other: "InputMoments") -> None:
        if other.dim != self.dim:
            raise ValueError(f"dim mismatch {other.dim} != {self.dim}")
        c2 = None
        if self.full:
            if not other.full and other.count:
                raise ValueError("cannot merge moments without full second moment into full ones")
            c2 = other._c2
        self._merge_stats(other.count, other._mean, other._m2, other._max, c2)

    # -- statistics ------------------------------------------------------------------------------
    def mean(self) -> mx.array:
        return self._mean

    def var(self) -> mx.array:
        return self._m2 / max(self.count, 1)

    def max_abs(self) -> mx.array:
        return self._max

    def second_moment_diag(self) -> mx.array:
        return self.var() + self._mean * self._mean

    def second_moment(self) -> mx.array | None:
        if not self.full:
            return None
        return self._c2 / max(self.count, 1) + mx.outer(self._mean, self._mean)

    def summary(self, top_k: int = 16) -> dict:
        out: dict = {"count": self.count, "dim": self.dim}
        if self.count == 0:
            return out
        diag = np.array(self.second_moment_diag())
        for key, v in (("var", np.array(self.var())), ("max_abs", np.array(self._max)), ("second_moment_diag", diag)):
            q = {k: float(p) for k, p in zip(_PCTS, np.percentile(v, list(_PCTS.values())))}
            out[key] = {**q, "max": float(v.max())}
        med = float(np.median(diag))
        out["outlier_channel_ratio"] = float(diag.max()) / med if med > 0 else float("inf")
        out["top_channels"] = [int(i) for i in np.argsort(-diag, kind="stable")[:top_k]]
        return out

    # -- serialization ---------------------------------------------------------------------------
    def to_arrays(self, prefix: str) -> dict[str, mx.array]:
        """``.mean``, ``.m2diag`` = E[x^2], ``.max_abs``, ``.m2full`` = E[x xᵀ] (if enabled), ``.count`` (int64)."""
        out = {f"{prefix}.mean": self._mean, f"{prefix}.m2diag": self.second_moment_diag(),
               f"{prefix}.max_abs": self._max, f"{prefix}.count": mx.array(self.count, dtype=mx.int64)}
        if self.full:
            out[f"{prefix}.m2full"] = self.second_moment()
        return out

    @classmethod
    def from_arrays(cls, arrays: dict[str, mx.array], prefix: str) -> "InputMoments":
        """Inverse of ``to_arrays``; centered sums are recovered as n*(E[x^2] - mean^2) in float32."""
        mean = arrays[f"{prefix}.mean"].astype(mx.float32)
        full = f"{prefix}.m2full" in arrays
        m = cls(mean.shape[0], full)
        n = int(arrays[f"{prefix}.count"].item())
        m2 = (arrays[f"{prefix}.m2diag"] - mean * mean) * n
        c2 = (arrays[f"{prefix}.m2full"] - mx.outer(mean, mean)) * n if full else None
        m._merge_stats(n, mean, m2, arrays[f"{prefix}.max_abs"].astype(mx.float32), c2)
        return m


class _SharedInput:
    """Call-cycle tracker for sibling linears fed the same input: only the first call records."""

    def __init__(self, members: int):
        self.members, self.pos, self.last = members, 0, None

    def should_record(self, x: mx.array) -> bool:
        first = self.pos == 0
        if first:
            self.last = x
        elif x is not self.last:
            raise RuntimeError("siblings expected to share one input received different arrays")
        self.pos += 1
        if self.pos == self.members:
            self.pos, self.last = 0, None
        return first


_RECORDING_CLASSES: dict[type, type] = {}


def _recording_class(cls: type) -> type:
    if cls not in _RECORDING_CLASSES:
        def __call__(self, x, *args, **kwargs):
            self._suc_hook(x, *args, **kwargs)
            return cls.__call__(self, x, *args, **kwargs)
        _RECORDING_CLASSES[cls] = type(f"Recording{cls.__name__}", (cls,), {"__call__": __call__})
    return _RECORDING_CLASSES[cls]


def _in_dims(m: nn.Module) -> int:
    if isinstance(m, nn.QuantizedLinear):
        return m.scales.shape[-1] * m.group_size
    if "scales" in m:  # QuantizedSwitchLinear
        return m.input_dims
    return m.weight.shape[-1]


def _join(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


class Recorder:
    """Instruments nn.Linear / SwitchGLU modules inside a module tree to record their inputs; forward outputs unchanged."""

    def __init__(self, full_second_moment: Iterable[str] = ()):
        self.full = set(full_second_moment)
        self.moments: dict[str, InputMoments] = {}
        self.expert_counts: dict[str, mx.array] = {}
        self._swapped: list[tuple[nn.Module, type]] = []

    def _new(self, names: list[str], dim: int) -> InputMoments:
        m = InputMoments(dim, any(n in self.full for n in names))
        for n in names:
            self.moments[n] = m
        return m

    def _swap(self, mod: nn.Module, hook) -> None:
        self._swapped.append((mod, type(mod)))
        mod._suc_hook = hook
        mod.__class__ = _recording_class(type(mod))

    # -- instrumentation -------------------------------------------------------------------------
    def instrument(self, module: nn.Module, prefix: str) -> list[str]:
        """Record every linear under `module`, whose checkpoint name is `prefix` ("" for the model root)."""
        before = set(self.moments)
        self._walk(module, prefix)
        return [n for n in self.moments if n not in before]

    def _walk(self, mod: nn.Module, path: str) -> None:
        linears: dict[str, nn.Module] = {}
        for key, val in mod.items():
            children = [(key, val)] if isinstance(val, nn.Module) else (
                [(f"{key}.{i}", v) for i, v in enumerate(val) if isinstance(v, nn.Module)]
                if isinstance(val, (list, tuple)) else [])
            for k, child in children:
                if isinstance(child, (nn.Linear, nn.QuantizedLinear)):
                    linears[k] = child
                elif isinstance(child, SwitchGLU):
                    self._instrument_switch(child, _join(path, k))
                else:
                    self._walk(child, _join(path, k))
        for group in _SHARED_INPUT_GROUPS:
            if all(g in linears for g in group):
                names = [_join(path, g) for g in group]
                moments, shared = self._new(names, _in_dims(linears[group[0]])), _SharedInput(len(group))
                for g in group:
                    self._swap(linears.pop(g), self._dense_hook(moments, shared))
        for k, lin in linears.items():
            self._swap(lin, self._dense_hook(self._new([_join(path, k)], _in_dims(lin)), None))

    @staticmethod
    def _dense_hook(moments: InputMoments, shared: _SharedInput | None):
        def hook(x, *_, **__):
            if shared is None or shared.should_record(x):
                moments.update(x)
        return hook

    def _instrument_switch(self, glu: SwitchGLU, path: str) -> None:
        """mlx-lm SwitchGLU at "<L>.mlp.switch_mlp": experts are "<L>.mlp.experts.{e}.{gate,up,down}_proj"."""
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        base = _join(parent, "experts")
        E, dim, hid = glu.gate_proj.num_experts, _in_dims(glu.gate_proj), _in_dims(glu.down_proj)
        gate_up = [self._new([f"{base}.{e}.gate_proj", f"{base}.{e}.up_proj"], dim) for e in range(E)]
        down = [self._new([f"{base}.{e}.down_proj"], hid) for e in range(E)]
        self.expert_counts[parent] = mx.zeros((E,), mx.int64)
        shared = _SharedInput(2)

        def gate_up_hook(x, indices, *_, **__):
            if shared.should_record(x):
                counts = self._expert_update(gate_up, x, indices)
                self.expert_counts[parent] = self.expert_counts[parent] + mx.array(counts, dtype=mx.int64)

        def down_hook(x, indices, *_, **__):
            self._expert_update(down, x, indices)

        self._swap(glu.gate_proj, gate_up_hook)
        self._swap(glu.up_proj, gate_up_hook)
        self._swap(glu.down_proj, down_hook)

    @staticmethod
    def _expert_update(moments: list[InputMoments], x: mx.array, indices: mx.array) -> np.ndarray:
        """Scatter rows to experts. x broadcasts to indices.shape + (1, D) in both the unsorted
        ((..., 1, 1, D) / (..., k, 1, H)) and the sorted ((N*k, 1, D), indices (N*k,)) SwitchGLU paths."""
        E, D = len(moments), x.shape[-1]
        rows = mx.broadcast_to(x, indices.shape + x.shape[-2:]).reshape(-1, D).astype(mx.float32)
        e = indices.reshape(-1)
        n = mx.zeros((E,), mx.float32).at[e].add(1.0)
        mean = mx.zeros((E, D), mx.float32).at[e].add(rows) / mx.maximum(n, 1.0)[:, None]
        d = rows - mean[e]
        m2 = mx.zeros((E, D), mx.float32).at[e].add(d * d)
        mxa = mx.zeros((E, D), mx.float32).at[e].maximum(mx.abs(rows))
        mx.eval(n, mean, m2, mxa)
        counts = np.array(n).astype(np.int64)
        for k, m in enumerate(moments):
            if counts[k]:
                c2 = None
                if m.full:
                    dk = d * (e == k).astype(mx.float32)[:, None]
                    c2 = dk.T @ dk
                m._merge_stats(int(counts[k]), mean[k], m2[k], mxa[k], c2)
        return counts

    def remove(self, module: nn.Module) -> None:
        """Restore the original classes of the instrumented modules inside `module` (moments are kept)."""
        inside = {id(m) for m in module.modules()}
        keep = []
        for mod, cls in reversed(self._swapped):
            if id(mod) in inside:
                mod.__class__ = cls
                del mod._suc_hook
            else:
                keep.append((mod, cls))
        self._swapped = keep[::-1]

    # -- persistence -----------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """One safetensors file (each shared object stored once, aliases in metadata) + a JSON sidecar of summaries."""
        path = Path(path)
        canon: dict[int, str] = {}
        aliases: dict[str, str] = {}
        arrays: dict[str, mx.array] = {}
        for name, m in self.moments.items():
            if id(m) in canon:
                aliases[name] = canon[id(m)]
            else:
                canon[id(m)] = name
                arrays.update(m.to_arrays(f"moments.{name}"))
        for name, c in self.expert_counts.items():
            arrays[f"expert_counts.{name}"] = c
        meta = {"moments": json.dumps(list(canon.values())), "aliases": json.dumps(aliases),
                "expert_counts": json.dumps(list(self.expert_counts)), "full": json.dumps(sorted(self.full))}
        mx.save_safetensors(str(path), arrays, metadata=meta)
        side = {"moments": {n: m.summary() for n, m in self.moments.items()},
                "expert_counts": {n: np.array(c).tolist() for n, c in self.expert_counts.items()}}
        path.with_suffix(".json").write_text(json.dumps(side, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Recorder":
        arrays, meta = mx.load(str(path), return_metadata=True)
        rec = cls(json.loads(meta["full"]))
        for name in json.loads(meta["moments"]):
            rec.moments[name] = InputMoments.from_arrays(arrays, f"moments.{name}")
        for alias, name in json.loads(meta["aliases"]).items():
            rec.moments[alias] = rec.moments[name]
        for name in json.loads(meta["expert_counts"]):
            rec.expert_counts[name] = arrays[f"expert_counts.{name}"]
        return rec
