"""Greedy semantic bit allocator (spec §14 Phase 7, §39 first concrete version, §45; granularity D6).

Problem. Each eligible tensor i gets exactly one candidate representation a_i (a codec; D6: the base
representation is chosen per tensor). A "tensor" here is an allocation UNIT: one checkpoint tensor, or one MoE expert
stack (all E experts of a (layer, projection), build_candidates(stacks=...), suc.quant.stacks) whose bits and loss are
the sums over its experts. Minimise the total predicted semantic loss under a bit budget:

    min Σ_i D_i(a_i)   s.t.   Σ_i bits_i(a_i) <= B

bits_i are ACTUAL encoded bits (Encoded.bits()["total"]: indices + codebooks + scales + metadata).
D_i is a per-tensor predicted loss, normally the empirical-Fisher proxy ½ Σ F ⊙ ΔW² (docs/benchmarks.md,
Phase 2: best tensor-sensitivity ranking, Spearman ≈ 0.75 on Qwen3-0.6B and ≈ 0.82 on Qwen3-8B).

ASSUMPTION (additivity). The total loss is modelled as the SUM of per-tensor losses, each measured
with every other tensor at full precision. This ignores interactions between errors of different
tensors (the cross terms of the second-order expansion are not diagonal across tensors, and errors
compound through depth). Full-model validation (PPL / KL of the reconstructed model) checks it; §39
step 8 says to iterate if it diverges.

Algorithm (multiple-choice knapsack, greedy on convex hulls):
1. per tensor, keep only the lower convex hull of its (bits, loss) points (dominated and non-convex
   candidates removed) so that successive upgrades have decreasing marginal value;
2. start every tensor at its cheapest hull point (ValueError if that alone exceeds B);
3. repeatedly apply the pending upgrade with the largest Δloss / Δbits that still fits (a heap holds
   one pending upgrade per tensor: the next hull step); an upgrade that does not fit freezes its
   tensor, since the remaining budget only shrinks; stop when the heap is empty.
With equal bit steps across tensors greedy is exactly optimal; in general it is within one hull step
of the LP (hull) relaxation optimum, i.e. loss <= OPT + max single-step loss drop. Ties are broken by
tensor name, so the plan is deterministic.
"""
from __future__ import annotations

import heapq
import math
import inspect
import time
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import mlx.core as mx

import numpy as np

from suc.quant.codecs import Codec
from suc.quant.seeds import encode_seed, rotation_kwargs
from suc.quant.stacks import _trellis_base, encode_stack, shares_codebooks, shares_rotation, stack_of, stack_seeds


def trellis_kwargs(codec, name: str, seed: int, hessians: Mapping[str, mx.array] | None) -> dict:
    """encode kwargs a tcq codec gets for tensor `name` in suc compress (scoring and encode alike): the stack's input
    rotation seed for experts (shares_rotation) and the hessian when the codec is tagged ldlq and one is recorded."""
    kw = {}
    if (st := stack_of(name)) is not None and shares_rotation(codec):
        kw["seed_in"] = stack_seeds(st[0], seed)["seed_in"]
    key = name.removesuffix(".weight")
    if hessians is not None and getattr(codec, "ldlq", False) and key in hessians:
        kw["hessian"] = hessians[key]
    return kw


def _score_trellis(codec, n: str, w: mx.array, kw: dict, *, seed: int, loss_fn, hessians, row_sample: int):
    """(bits, loss, err, ‖W‖²) of a tcq codec on tensor n (build_candidates docstring: rotation / LDLQ / row sample)."""
    kw = {**kw, **trellis_kwargs(codec, n, seed, hessians)}
    out = int(w.shape[0])
    rows = _sample_rows(n, out, seed, row_sample)
    if rows is not None:
        ri = mx.array(rows.astype(np.int32))
        ws = w[ri]
        if "importance" in kw and kw["importance"] is not None:
            kw["importance"] = kw["importance"][ri]
    else:
        ws = w
    e = codec.encode(ws, seed=seed + zlib.crc32(n.encode()), **kw)
    w_hat = codec.decode(e)
    f = 1.0 if rows is None else out / len(rows)
    loss = loss_fn(n, ws, w_hat) if rows is None else loss_fn(n, ws, w_hat, rows=rows)
    err, w2 = mx.sum((w_hat - ws) ** 2).item(), mx.sum(ws * ws).item()
    bits = int(codec.bits_for(w.shape)) if rows is not None else int(e.bits()["total"])
    return bits, float(loss) * f, err * f, w2 * f


def _sample_rows(n: str, out: int, seed: int, row_sample: int) -> np.ndarray | None:
    if not row_sample or out <= row_sample:
        return None
    rng = np.random.default_rng(seed + zlib.crc32(n.encode()) + 1)
    return np.sort(rng.choice(out, row_sample, replace=False))


def _batchable(codec, members: Sequence[str], seed: int, hessians) -> bool:
    """A tcq codec can score the experts of one stack in ONE encode when they share the input rotation and the
    (LDLQ) hessian object (MoE gate / up: one shared second moment per layer): the trellis encode is row-independent,
    so the rows of all members are encoded together (Viterbi batches of thousands of sequences instead of a few
    hundred per expert)."""
    if len(members) < 2 or _trellis_base(codec) is None or not shares_rotation(codec):
        return False
    st = {(stack_of(n) or (None,))[0] for n in members}
    if len(st) != 1 or None in st:
        return False
    kws = [trellis_kwargs(codec, n, seed, hessians) for n in members]
    h0 = kws[0].get("hessian")
    return all(k.get("seed_in") == kws[0].get("seed_in") and k.get("hessian") is h0 for k in kws)


def _score_trellis_stack(codec, members: Sequence[str], get, *, seed: int, loss_fn, hessians, row_sample: int,
                         importance) -> dict[str, tuple]:
    """_score_trellis for every member of a stack with one base encode of the (sampled) rows of all members; the
    residual (if any) is then fitted per member on its own rows (ResidualCodec.encode(base_encoded=))."""
    from suc.quant.codecs import Encoded

    base = _trellis_base(codec)
    kw0 = trellis_kwargs(codec, members[0], seed, hessians)
    parts, rows_of, shapes = [], {}, {}
    for n in members:
        w = get(n).astype(mx.float32)
        shapes[n] = tuple(w.shape)
        rows = _sample_rows(n, int(w.shape[0]), seed, row_sample)
        rows_of[n] = rows
        ws = w if rows is None else w[mx.array(rows.astype(np.int32))]
        mx.eval(ws)
        parts.append(ws)
        del w
    big = mx.concatenate(parts, axis=0)
    be = base.encode(big, seed=seed + zlib.crc32(members[0].encode()), **kw0)
    del big
    T, V = int(be.params["T"]), int(be.params["V"])
    seq_per_row = shapes[members[0]][1] // T
    out, r0 = {}, 0
    for n, ws in zip(members, parts):
        nr = int(ws.shape[0])
        arrays = {"chunks": be.arrays["chunks"][r0 * seq_per_row:(r0 + nr) * seq_per_row],
                  "row_scales": be.arrays["row_scales"][r0:r0 + nr]}
        p = dict(be.params)
        p["seed"] = seed + zlib.crc32(n.encode())
        p["stream_bits"] = int(round(base.k * nr * ws.shape[1]))
        sub = Encoded(be.codec, (nr, int(ws.shape[1])), p, arrays)
        r0 += nr
        rows = rows_of[n]
        if codec is not base:
            imp = importance.get(n) if importance is not None else None
            if imp is not None and rows is not None:
                imp = imp[mx.array(rows.astype(np.int32))]
            ikw = {"importance": imp} if imp is not None else {}
            e = codec.encode(ws, seed=seed + zlib.crc32(n.encode()), base_encoded=sub, **ikw)
        else:
            e = sub
        w_hat = codec.decode(e)
        f = 1.0 if rows is None else shapes[n][0] / len(rows)
        loss = loss_fn(n, ws, w_hat) if rows is None else loss_fn(n, ws, w_hat, rows=rows)
        err, w2 = mx.sum((w_hat - ws) ** 2).item(), mx.sum(ws * ws).item()
        bits = int(codec.bits_for(shapes[n])) if rows is not None else int(e.bits()["total"])
        out[n] = (bits, float(loss) * f, err * f, w2 * f)
    return out


@dataclass(frozen=True)
class Candidate:
    codec: str          # codec name
    bits: int           # actual bits of the encoded tensor (Encoded.bits()["total"])
    loss: float         # predicted semantic loss (e.g. fisher_proxy(ΔW, F)); lower is better
    rel_mse: float = float("nan")


def build_candidates(tensors: Mapping[str, mx.array] | Callable[[str], mx.array], names: Sequence[str],
                     codecs: Mapping[str, Codec], loss_fn: Callable[[str, mx.array, mx.array], float], *,
                     seed: int, log_every: int = 0,
                     importance: Mapping[str, mx.array] | None = None,
                     stacks: Mapping[str, Sequence[str]] | None = None,
                     per_member: bool = False, hessians: Mapping[str, mx.array] | None = None,
                     row_sample: int = 0, rotation_groups: bool = False) -> dict[str, list[Candidate]]:
    """Encode every unit with every codec and score it: {unit: [Candidate per codec, in codec order]}.

    Tensors are processed one at a time (loaded lazily when `tensors` is a callable, e.g. store.load);
    encode seed = seed + crc32(name), as in suc.evaluation.reconstruct.
    loss_fn(name, w, w_hat) -> predicted loss of replacing w by w_hat. `importance[name]` (per-weight) is passed to
    codecs whose encode accepts it (importance-selected residuals, Q5).

    stacks: {unit name: [tensor names]} for units made of several tensors (MoE expert stacks, suc.quant.stacks.units);
    a name in `names` that is a key of `stacks` is scored as ONE unit: bits and loss are summed over its tensors and
    rel_mse is Σ‖ΔW‖² / Σ‖W‖². Codecs with codebook scope "expert_stack" share one codebook set per stack
    (encode_stack, as reconstruct_model(expert_stacks=True)); the others are encoded per tensor.

    per_member=True scores the members of a stack unit SEPARATELY ({member name: [Candidate per codec]}; allocation
    unit "expert"): shared-codebook codecs still fit one pooled codebook per stack, and member 0 carries its bits.

    Trellis (tcq) codecs: experts get their stack's input rotation seed when the codec shares it (shares_rotation, as
    reconstruct_model), codecs tagged `ldlq = True` get hessians[name without ".weight"] (BlockLDLQ) when present,
    and with row_sample > 0 a tensor with more rows is scored on a deterministic sample of row_sample rows (the codec
    is row-independent): loss and squared errors are scaled by out / row_sample and loss_fn is called as
    loss_fn(name, w_rows, w_hat_rows, rows=rows); bits are the exact full-tensor bits (codec.bits_for(shape)).

    rotation_groups: as reconstruct_model / suc compress rotation_groups - per-tensor codecs get the SAME rotation
    seeds as the final encode (suc.quant.seeds.rotation_kwargs: one shared seed_in for q/k/v and dense gate/up), so
    the scored distortion is that of the encoding that is written (audit F08)."""
    get = tensors if callable(tensors) else tensors.__getitem__
    stacks = stacks or {}
    out: dict[str, list[Candidate]] = {}
    t0 = time.perf_counter()

    def kw_for(n: str, codec) -> dict:
        if importance is not None and n in importance and "importance" in inspect.signature(codec.encode).parameters:
            return {"importance": importance[n]}
        return {}

    def score(codec, n: str, w: mx.array, e) -> tuple[int, float, float, float]:
        w = w.astype(mx.float32)
        w_hat = codec.decode(e)
        err, w2 = mx.sum((w_hat - w) ** 2).item(), mx.sum(w * w).item()
        return int(e.bits()["total"]), float(loss_fn(n, w, w_hat)), err, w2

    for i, name in enumerate(names):
        members = list(stacks.get(name, [name]))
        acc = {cname: [0, 0.0, 0.0, 0.0] for cname in codecs}       # bits, loss, err, ‖W‖² summed over members
        each: dict[str, dict[str, list]] = {n: {} for n in members}  # per_member: member -> codec -> [bits, loss, ..]

        def add(cname: str, n: str, r: tuple) -> None:
            acc[cname] = [a + b for a, b in zip(acc[cname], r)]
            if per_member:
                each[n][cname] = list(r)

        shared = [cn for cn, c in codecs.items() if len(members) > 1 and shares_codebooks(c)]
        for cname in shared:                                         # one pooled fit per stack and codec
            codec = codecs[cname]
            for n, w, e in encode_stack(codec, get, members, seed=seed, stack=name, importance=importance):
                add(cname, n, score(codec, n, w, e))
                del w, e
        rest = [cn for cn in codecs if cn not in shared]
        for cname in [cn for cn in rest if _batchable(codecs[cn], members, seed, hessians)]:
            for n, r in _score_trellis_stack(codecs[cname], members, get, seed=seed, loss_fn=loss_fn,
                                             hessians=hessians, row_sample=row_sample, importance=importance).items():
                add(cname, n, r)
            rest.remove(cname)
        if rest:
            for n in members:                                        # each tensor loaded once for all its codecs
                w = get(n).astype(mx.float32)
                for cname in rest:
                    codec = codecs[cname]
                    kw = {**kw_for(n, codec), **rotation_kwargs(codec, n, seed, rotation_groups)}
                    if _trellis_base(codec) is not None:
                        add(cname, n, _score_trellis(codec, n, w, kw, seed=seed, loss_fn=loss_fn, hessians=hessians,
                                                     row_sample=row_sample))
                        continue
                    e = codec.encode(w, seed=encode_seed(n, seed), **kw)
                    add(cname, n, score(codec, n, w, e))
                    del e
                del w
        if per_member:
            for n in members:
                out[n] = [Candidate(cn, int(a[0]), float(a[1]), a[2] / a[3] if a[3] > 0 else float("nan"))
                          for cn in codecs for a in [each[n][cn]]]
        else:
            out[name] = [Candidate(cn, int(a[0]), float(a[1]), a[2] / a[3] if a[3] > 0 else float("nan"))
                         for cn, a in acc.items()]
        mx.clear_cache()
        if log_every and (i + 1) % log_every == 0:
            print(f"[candidates] {i + 1}/{len(names)} units, {time.perf_counter() - t0:.1f}s", flush=True)
    return out


def lower_hull(cands: Sequence[Candidate]) -> list[Candidate]:
    """Non-dominated candidates on the lower convex hull of (bits, loss), sorted by bits.

    Along the result bits strictly increase, loss strictly decreases and the marginal value
    Δloss/Δbits strictly decreases (collinear middle points are dropped)."""
    front: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (c.bits, c.loss)):
        if not front or c.loss < front[-1].loss:  # drop dominated: more (or equal) bits, not less loss
            front.append(c)
    hull: list[Candidate] = []
    for c in front:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # pop b if the rate a->b is not larger than the rate b->c (b on or above the chord a-c)
            if (a.loss - b.loss) * (c.bits - b.bits) <= (b.loss - c.loss) * (b.bits - a.bits):
                hull.pop()
            else:
                break
        hull.append(c)
    return hull


def greedy_allocate(cands: Mapping[str, Sequence[Candidate]], budget_bits: int) -> dict[str, Candidate]:
    """Greedy marginal-value allocation on each tensor's lower hull (module docstring). Plan in input order."""
    hulls = {n: lower_hull(c) for n, c in cands.items()}
    if empty := [n for n, h in hulls.items() if not h]:
        raise ValueError(f"tensors without candidates: {empty[:5]}")
    level = dict.fromkeys(hulls, 0)
    used = sum(h[0].bits for h in hulls.values())
    if used > budget_bits:
        raise ValueError(f"infeasible budget: the cheapest assignment needs {used} bits > {budget_bits}")

    def push(heap: list, n: str) -> None:
        h, j = hulls[n], level[n]
        if j + 1 < len(h):
            a, b = h[j], h[j + 1]
            heapq.heappush(heap, (-(a.loss - b.loss) / (b.bits - a.bits), n))

    heap: list = []
    for n in hulls:
        push(heap, n)
    while heap:
        _, n = heapq.heappop(heap)
        a, b = hulls[n][level[n]], hulls[n][level[n] + 1]
        if used + b.bits - a.bits > budget_bits:
            continue  # this tensor is frozen: the remaining budget can only shrink
        used += b.bits - a.bits
        level[n] += 1
        push(heap, n)
    return {n: hulls[n][level[n]] for n in cands}


def plan_summary(plan: Mapping[str, Candidate], cands: Mapping[str, Sequence[Candidate]] | None = None) -> dict:
    """JSON-able summary; with cands also the cheapest / most expensive possible totals (bits_min/max)."""
    out = {"bits_total": sum(c.bits for c in plan.values()), "loss_total": sum(c.loss for c in plan.values()),
           "tensors": len(plan), "by_codec": dict(Counter(c.codec for c in plan.values())),
           "rows": [{"tensor": n, **asdict(c)} for n, c in plan.items()]}
    if cands is not None:
        out["bits_min"] = sum(min(c.bits for c in v) for v in cands.values())
        out["bits_max"] = sum(max(c.bits for c in v) for v in cands.values())
    return out


def budget_for_bpp(target_bpp: float, total_params: int, fixed_bits: int) -> int:
    """Bits left for the allocated tensors so that (allocated + fixed_bits) / total_params <= target_bpp."""
    b = math.floor(target_bpp * total_params) - fixed_bits
    if b < 0:
        raise ValueError(f"target {target_bpp} bpp is below the fixed part alone ({fixed_bits / total_params:.4f} bpp)")
    return int(b)


def uniform_plan(cands: Mapping[str, Sequence[Candidate]], codec: str) -> dict[str, Candidate]:
    """Every tensor gets the same codec (baseline); KeyError if a tensor lacks it."""
    out = {}
    for n, cs in cands.items():
        match = [c for c in cs if c.codec == codec]
        if not match:
            raise KeyError(f"tensor {n!r} has no candidate {codec!r}")
        out[n] = match[0]
    return out
