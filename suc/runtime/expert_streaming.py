"""Expert streaming in the Python SUC1 runtime: routed experts read from SSD through a byte-budgeted ExpertCache.

For models whose SUC1 file is larger than the memory the runtime may use (GLM-5.3-Flash at ~3 bpw: ~118 GB file,
111 GB runtime budget on the 128 GB Mac Studio; `suc compress` accepts such a plan only with
preflight.expert_streaming, which checks that the non-expert weights + KV + activations leave a large enough expert
cache). Everything except the routed experts is loaded as usual (suc.runtime.suc1_model.load_suc1_model, the routed
experts skipped); every MoE layer's experts module gets an ExpertStream:

  per call   the router's expert ids -> compact() (distinct ids, remapped indices) -> ExpertCache.get (resident /
             pinned records, misses read with pread in parallel) -> one SUC switch module per projection over the
             selected experts only (switch_from_encodings: mixed codecs per stack allowed) -> the model's usual expert
             math with the remapped indices. The selected experts' records are exactly their stored payloads, so the
             computation equals the fully resident model's (tested bit for bit on a tiny glm5_next).
  pinning    the most-routed experts of the calibration pass (run_dir/container/hot_experts.json, written by `suc
             compress` with preflight.expert_streaming) are pinned first, up to pin_fraction of the budget, greedily
             by routing count over all layers (every layer sees the same tokens x top-k, so counts compare across
             layers); the rest of the budget is an LRU / segmented-LRU cache.

Expert records may be SUC-coded (vq / additive, with residuals) or native MXFP4 pass-through (codes + E8M0 scales,
run by MXFP4SwitchLinear); a layer may mix both (SucSwitchMixed over the selected experts).

Models opt in by giving their routed-experts module a `_stream` attribute consulted for the projections
(suc.models.mlx_glm5_next.RoutedExperts, suc.models.mlx_deepseek_v4.RoutedExperts). Memory: resident non-expert
weights + budget_bytes of expert records + one layer's in-flight read buffers.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from suc.runtime.expert_cache import EXPERT_RE, ExpertCache, compact

_PROJS = ("gate_proj", "up_proj", "down_proj")


class ExpertStream:
    """The routed experts of one MoE layer, fetched per call from an ExpertCache (module docstring).

    A call whose distinct experts do not fit the unpinned part of the budget (e.g. a long prefill on a small cache)
    is split into expert groups that do; each group runs with the routing weights of the other experts set to 0 and
    the partial sums are added (same terms, different float32 summation order than one pass)."""

    def __init__(self, cache: ExpertCache, layer: str):
        self.cache, self.layer = cache, layer
        self.li = cache.layout.layer_index(layer)

    def _groups(self, ids: list[int]) -> list[list[int]]:
        c, li = self.cache, self.li
        free = c.budget - c.stats.pinned_bytes
        out, cur, used = [], [], 0
        for e in ids:
            n = 0 if (li, e) in c._pinned else c.layout.entry(li, e).nbytes
            if cur and used + n > free:
                out.append(cur)
                cur, used = [], 0
            cur.append(e)
            used += n
        return out + ([cur] if cur else [])

    def modules(self, ids: list[int]):
        from suc.kernels.suc_switch import switch_from_encodings

        recs = self.cache.get(self.li, ids)
        return [switch_from_encodings([r.encoded(p) for r in recs]) for p in _PROJS]

    def run(self, experts, x: mx.array, inds: mx.array, w: mx.array) -> mx.array:
        """experts.compute over the experts selected by `inds` (the model's routed-experts module)."""
        ids, local = compact(np.array(inds))
        groups = self._groups(ids)
        if len(groups) == 1:
            return experts.compute(*self.modules(ids), x, mx.array(local).astype(mx.uint32), w)
        y = None
        for g in groups:
            pos = np.full(len(ids), -1, np.int64)
            pos[[ids.index(e) for e in g]] = np.arange(len(g))
            loc = pos[local]
            part = experts.compute(*self.modules(g), x, mx.array(np.maximum(loc, 0)).astype(mx.uint32),
                                   mx.where(mx.array(loc >= 0), w, mx.zeros_like(w)))
            mx.eval(part)                                 # before the next group's reads may evict these records
            y = part if y is None else y + part
        return y


class _NoWeights(nn.Module):
    """Placeholder for the resident expert projections of a streamed layer (never called, no parameters)."""

    def __call__(self, *a, **k):  # pragma: no cover - guarded by ExpertStream
        raise RuntimeError("streamed expert layer: projections come from the ExpertCache")


def is_routed_expert(name: str) -> bool:
    return EXPERT_RE.match(name) is not None


def load_hot_experts(path: str | Path | None) -> dict[str, list[int]] | None:
    """{MoE layer prefix: routing counts} from a hot_experts.json (None when path is None or missing)."""
    if path is None or not Path(path).exists():
        return None
    return {k: [int(c) for c in v] for k, v in json.loads(Path(path).read_text())["layers"].items()}


def choose_pins(cache: ExpertCache, counts: dict[str, list[int]], pin_bytes: int) -> dict[str, list[int]]:
    """Greedy hot set: (layer, expert) by routing count, most-routed first, while the pinned bytes fit pin_bytes;
    experts never routed in calibration are never pinned. Layers absent from the file layout are ignored."""
    order = []
    for layer, cs in counts.items():
        if layer not in cache.layout.layers:
            continue
        order += [(c, layer, e) for e, c in enumerate(cs) if c > 0]
    order.sort(key=lambda t: (-t[0], t[1], t[2]))
    out: dict[str, list[int]] = {}
    used = 0
    for _, layer, e in order:
        n = cache.layout.entry(layer, e).nbytes
        if used + n > pin_bytes:
            continue
        out.setdefault(layer, []).append(e)
        used += n
    return out


def attach_expert_streaming(model: nn.Module, cache: ExpertCache) -> list[str]:
    """Give every MoE layer of `model` whose experts are in the cache's layout an ExpertStream; returns the layer
    prefixes. The experts module lives at "<prefix>.switch_mlp" (the SUC / mlx-lm stacking convention)."""
    done = []
    for prefix in cache.layout.layers:
        mod = model
        for k in prefix.split("."):
            mod = mod[int(k)] if isinstance(mod, list) else mod[k]
        sw = mod["switch_mlp"]
        if not hasattr(sw, "_stream") or not hasattr(sw, "compute"):
            raise TypeError(f"{prefix}.switch_mlp ({type(sw).__name__}) does not support expert streaming")
        for p in _PROJS:
            setattr(sw, p, _NoWeights())          # drop the (never materialized) resident stacks
        sw._stream = ExpertStream(cache, prefix)
        done.append(prefix)
    return done


def load_streamed_suc1_model(path: str | Path, budget_bytes: int, *, hot_experts: str | Path | None = "auto",
                             pin_fraction: float = 0.5, policy: str = "slru", io: str = "pread",
                             nocache: bool = True, io_threads: int = 8, dtype: mx.Dtype | None = None,
                             group: bool = False, fuse: bool = False, affine_modules: bool = False, log=None):
    """(model, SucModel, ExpertCache, report): the SUC1 model with its routed experts streamed through an
    ExpertCache of budget_bytes (module docstring). hot_experts: a hot_experts.json, "auto" (the one next to the SUC1
    file, i.e. run_dir/container/hot_experts.json, if present) or None (no pinning)."""
    from suc.runtime.suc1_model import load_suc1_model

    path = Path(path)
    model, sm, report = load_suc1_model(path, group=group, fuse=fuse, affine_modules=affine_modules, dtype=dtype,
                                        log=log, skip=is_routed_expert)
    cache = ExpertCache(sm, budget_bytes, policy=policy, io=io, nocache=nocache, io_threads=io_threads)
    hp = path.parent / "hot_experts.json" if hot_experts == "auto" else hot_experts
    counts = load_hot_experts(hp)
    pins = choose_pins(cache, counts, int(pin_fraction * budget_bytes)) if counts else {}
    for layer, experts in pins.items():
        cache.pin(layer, experts)
    layers = attach_expert_streaming(model, cache)
    est = report["memory_estimate"]                  # the directory estimate counts every expert: replace them
    est = {**est, "codes": est["codes"] - cache.layout.expert_bytes_total}
    est["total"] = est["codes"] + est["dense"] + est["affine"]
    report = {**report, "memory_estimate": {**est, "expert_cache_budget": int(budget_bytes)}, "expert_streaming": {
        "budget_bytes": int(budget_bytes), "policy": policy, "io": io, "layers": len(layers),
        "hot_experts": str(hp) if counts else None, "pinned_experts": sum(len(v) for v in pins.values()),
        "pinned_bytes": cache.stats.pinned_bytes, "layout": cache.layout.summary()}}
    if log is not None:
        log(f"expert streaming: {len(layers)} MoE layers, budget {budget_bytes / 2**30:.2f} GiB, pinned "
            f"{report['expert_streaming']['pinned_experts']} experts ({cache.stats.pinned_bytes / 2**30:.2f} GiB)")
    return model, sm, cache, report


__all__ = ["ExpertStream", "attach_expert_streaming", "choose_pins", "is_routed_expert", "load_hot_experts",
           "load_streamed_suc1_model"]
