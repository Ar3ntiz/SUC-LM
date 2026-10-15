"""Empirical diagonal Fisher of the weights as a loss-aware sensitivity proxy (spec §6, D5).

Layer-local proxies (weight MSE, diag(E[x xᵀ])) ignore how an error propagates to the logits. The
Optimal-Brain-Damage style second-order estimate uses the loss itself:

    ΔL ≈ ½ Σ F ⊙ ΔW²,   F = diag of the Fisher / Gauss-Newton matrix of the loss w.r.t. W.

Estimator (precisely). tokens (N, T) are split into consecutive micro-batches of `micro_batch` rows
(the last may be smaller). For micro-batch m, g_m = ∂ℓ_m/∂W where ℓ_m is the MEAN next-token loss
over its n_m = rows·(T-1) predicted positions (inputs tokens[:, :-1]). The returned value is

    F = (1/M) Σ_m g_m ⊙ g_m        (unweighted mean over the M micro-batches)

i.e. the "batch" empirical Fisher, not the per-token one. With per-token gradients g_t,
g_m = (1/n_m) Σ_t g_t, so E[g_m²] = (1/n_m²) Σ_t E[g_t²] + cross terms. Consequences:
- loss="ce": targets are the data tokens tokens[:, 1:]. Cross terms (correlated per-token gradients,
  non-zero mean gradient on finite data) do not vanish, so F is NOT simply F_token / n.
- loss="kl_self": targets are sampled from the model's own predictive distribution at each position
  (seeded, one key per micro-batch, identical for every parameter group). Given the context, the
  sampled positions are independent and each score has zero mean, so cross terms vanish in
  expectation and n_m · F is an unbiased estimator of the per-token TRUE Fisher diagonal, which is
  the exact Hessian of the mean next-token KL(P_W || P_{W+ΔW}) at ΔW = 0. Hence for kl_self the
  predicted mean KL is ≈ n_m · fisher_proxy(ΔW, F).
Rankings across tensors computed from the same tokens and micro_batch are unaffected by the 1/n_m scale.

Only the requested tensors get gradients: every parameter is frozen and only the parameters holding
the requested tensors are unfrozen. MoE experts "…mlp.experts.{e}.{proj}.weight" are row e of the
stacked "…mlp.switch_mlp.{proj}.weight"; the whole stack gets a gradient but only requested rows are
accumulated. With group_size set, the underlying parameters are processed in groups of that many (one
forward/backward over all micro-batches per group), so peak memory = weights + grads(group) +
activations. The frozen/trainable state of the model is restored on exit.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Literal, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from suc.analysis.sensitivity import _EXPERT, _resolve


@contextmanager
def _integer_indices_without_gradient() -> Iterator[None]:
    """Some mlx-lm MoE routers (e.g. qwen3_moe) gather with argpartition indices without stop_gradient,
    which makes the backward pass fail ("Cannot calculate VJP with respect to indices"). Integer
    indices carry no gradient, so wrapping mx.argpartition / mx.argsort in stop_gradient for the
    duration is semantically a no-op. Process-global patch, restored on exit (not thread-safe)."""
    saved = {k: getattr(mx, k) for k in ("argpartition", "argsort")}
    try:
        for k, f in saved.items():
            setattr(mx, k, lambda *a, _f=f, **kw: mx.stop_gradient(_f(*a, **kw)))
        yield
    finally:
        for k, f in saved.items():
            setattr(mx, k, f)


def _param_path(tensor_name: str) -> tuple[str, int | None]:
    """mlx-lm parameter path and expert row for a checkpoint tensor name."""
    if (m := _EXPERT.match(tensor_name)) is not None:
        return f"{m.group(1)}.switch_mlp.{m.group(3)}.weight", int(m.group(2))
    return tensor_name, None


def _unfreeze_path(model: nn.Module, path: str) -> None:
    """Unfreeze the parameter at a flattened path: the deepest Module on the path owns the local key."""
    parts = path.split(".")
    cur, owner, key = model, model, parts[0]
    for i, p in enumerate(parts[:-1]):
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
        if isinstance(cur, nn.Module):
            owner, key = cur, parts[i + 1]
    owner.unfreeze(keys=[key], recurse=False)


def _set_trainable(model: nn.Module, paths: Sequence[str]) -> None:
    model.freeze()
    for p in paths:
        _unfreeze_path(model, p)


def fisher_diag(model: nn.Module, tokens: mx.array, tensor_names: Sequence[str], *, micro_batch: int = 2,
                group_size: int | None = None, loss: Literal["ce", "kl_self"] = "ce",
                seed: int = 0) -> dict[str, mx.array]:
    """Batch empirical diagonal Fisher (see module docstring) for each checkpoint tensor name.

    Returns {name: float32 array with the tensor's shape}, in the order of tensor_names.
    """
    if loss not in ("ce", "kl_self"):
        raise ValueError(f"unknown loss {loss!r}")
    for n in tensor_names:
        _resolve(model, n)  # KeyError for unknown names / expert rows
    targets_of = {n: _param_path(n) for n in tensor_names}
    params = list(dict.fromkeys(p for p, _ in targets_of.values()))
    groups = [params] if group_size is None else [params[i:i + group_size] for i in range(0, len(params), group_size)]

    x_mb = [tokens[s:s + micro_batch] for s in range(0, tokens.shape[0], micro_batch)]
    if loss == "ce":
        y_mb = [t[:, 1:] for t in x_mb]
    else:  # sample targets once so that every parameter group sees the same ones
        keys = mx.random.split(mx.random.key(seed), len(x_mb))
        y_mb = []
        for t, k in zip(x_mb, keys):
            y = mx.random.categorical(model(t[:, :-1]).astype(mx.float32), key=k)
            mx.eval(y)
            y_mb.append(y)

    def mean_ce(x, y):
        return nn.losses.cross_entropy(model(x).astype(mx.float32), y, reduction="mean")

    trainable_before = [k for k, _ in tree_flatten(model.trainable_parameters())]
    out: dict[str, mx.array] = {}
    try:
        with _integer_indices_without_gradient():
            for group in groups:
                _set_trainable(model, group)
                vg = nn.value_and_grad(model, mean_ce)
                names = [n for n in tensor_names if targets_of[n][0] in group]
                acc: dict[str, mx.array] = {}
                for t, y in zip(x_mb, y_mb):
                    _, g = vg(t[:, :-1], y)
                    g = dict(tree_flatten(g))
                    for n in names:
                        path, row = targets_of[n]
                        gn = (g[path] if row is None else g[path][row]).astype(mx.float32)
                        acc[n] = gn * gn if n not in acc else acc[n] + gn * gn
                    mx.eval(list(acc.values()))
                    del g
                for n in names:
                    out[n] = acc[n] / len(x_mb)
                mx.eval(list(out.values()))
                del acc, vg
    finally:
        _set_trainable(model, trainable_before)
    return {n: out[n] for n in tensor_names}


_BLOCK = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_HEAD_TENSORS = ("model.embed_tokens.weight", "lm_head.weight")


def _mean_ce(logits: mx.array, y: mx.array) -> mx.array:
    return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


def fisher_diag_streamed(model: nn.Module, tokens: mx.array, tensor_names: Sequence[str], *,
                         model_dir: str | Path, on_group: Callable[[str, dict[str, mx.array]], None],
                         micro_batch: int = 2, loss: Literal["ce", "kl_self"] = "ce", seed: int = 0,
                         work_dir: str | Path | None = None, fingerprint: str = "") -> dict:
    """fisher_diag with at most ONE transformer block resident (spec D8), exact (no proxy): two sweeps.

    Forward sweep: every micro-batch is embedded and run block by block, each block evaluated as inside a gradient
    trace (see `traced`: MLX's fused kernels are replaced by composed ones there, which changes bf16 values); the
    INPUT of every block (and the final hidden state) is kept for every micro-batch (on disk under work_dir: (L+1) x N x T x h at the model dtype).
    The upstream gradient dℓ_m/dh_L of each micro-batch's mean loss is computed at the head. Reverse sweep: block
    i = L-1..0 is reloaded fresh from the checkpoint (suc.models.streaming.load_block), and for each micro-batch
    one VJP of the block at its stored input with the upstream gradient gives ∂ℓ_m/∂W (W of block i) and
    dℓ_m/dh_i (the next upstream gradient). These are the same per-micro-batch gradients g_m as the full-model
    backward pass of fisher_diag (same micro-batches, losses and sampled kl_self targets), so F = mean_m g_m²
    is the same estimator. Embedding / LM head (tensor names model.embed_tokens.weight, lm_head.weight) are done
    last: head-side gradient (recomputed from the stored h_L) + embedding-lookup gradient (from dℓ_m/dh_0), summed
    per micro-batch before squaring, as autodiff does for tied embeddings.

    F of each group is handed to on_group(group, {name: float32 F}) as soon as it is final — groups
    "block_{i:05d}" in reverse block order, then "head" — and is not kept: the caller scores or stores it.
    on_group must be idempotent per group. With work_dir the sweep is resumable at block granularity
    (progress.json keyed by `fingerprint`: a crash resumes at the first unfinished block; a group may be
    delivered again if the crash happened between on_group and the progress update). Only parameters of
    blocks with requested tensors get weight gradients; every block gets an input VJP.

    Returns {"groups": [...], "forward_seconds", "reverse_seconds", "peak_memory_bytes"}."""
    import json
    import os
    import shutil
    import time

    from mlx.utils import tree_unflatten
    from mlx_lm.models.base import create_attention_mask

    from suc.models.streaming import HiddenBuffer, _release, embed, head, load_block, model_blocks

    if loss not in ("ce", "kl_self"):
        raise ValueError(f"unknown loss {loss!r}")
    blocks = model_blocks(model)
    n_layers = len(blocks)
    by_block: dict[int, list[str]] = {}
    head_names = []
    for n in tensor_names:
        if n in _HEAD_TENSORS:
            if n == "lm_head.weight" and model.args.tie_word_embeddings:
                raise KeyError(f"{n}: tied embeddings have no separate LM head (use model.embed_tokens.weight)")
            head_names.append(n)
            continue
        m = _BLOCK.match(n)
        if m is None or int(m.group(1)) >= n_layers:
            raise KeyError(f"unknown or unsupported tensor {n!r} (block linears, embedding, LM head)")
        by_block.setdefault(int(m.group(1)), []).append(n)
    work = Path(work_dir) if work_dir is not None else None
    prog_path = work / "progress.json" if work is not None else None
    prog = {"fingerprint": fingerprint, "forward": False, "reverse_next": None, "head": False}
    if work is not None:
        if prog_path.exists():
            old = json.loads(prog_path.read_text())
            if old.get("fingerprint") == fingerprint:
                prog = old
        if not prog["forward"] and work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)

    def save_prog() -> None:
        if prog_path is not None:
            tmp = prog_path.with_name(".tmp.progress.json")
            tmp.write_text(json.dumps(prog))
            os.replace(tmp, prog_path)

    def buf(kind: str, level: int) -> HiddenBuffer:
        return HiddenBuffer(n_mb, work / kind / f"{level:05d}" if work is not None else None)

    x_mb = [tokens[s:s + micro_batch] for s in range(0, tokens.shape[0], micro_batch)]
    n_mb = len(x_mb)
    inputs = [t[:, :-1] for t in x_mb]
    acts = [buf("act", i) for i in range(n_layers + 1)]
    mx.reset_peak_memory()
    t0 = time.perf_counter()

    def traced(block, h):
        """Block output as computed INSIDE a gradient trace: under tracing MLX replaces fused kernels (e.g.
        fast.scaled_dot_product_attention) by their composed implementation, whose bf16 forward differs. The
        whole-model backward of fisher_diag sees these values, so the stored block inputs must be the traced
        ones. Only the primal output of the vjp is evaluated (lazy graph): no backward work is done here."""
        with _integer_indices_without_gradient():
            (out,), _ = mx.vjp(lambda x: block(x, create_attention_mask(x, None)), [h], [mx.zeros_like(h)])
        return out

    sample = loss == "kl_self"
    tpath = work / "targets.npy" if work is not None and sample else None
    if prog["forward"] and tpath is not None and not tpath.exists():
        prog = {"fingerprint": fingerprint, "forward": False, "reverse_next": None, "head": False}   # redo all
    if not prog["forward"]:
        # kl_self targets are sampled from the logits of the PLAIN forward model(x) (as fisher_diag does), so a
        # second, untraced chain of hidden states is kept (one level at a time) for kl_self only
        plain = HiddenBuffer(n_mb, work / "plain" if work is not None else None) if sample else None
        for m, x in enumerate(inputs):
            e = embed(model, x)
            acts[0].put(m, e)
            if sample:
                plain.put(m, e)
        for i, block in enumerate(blocks):
            for m in range(n_mb):
                h = acts[i].get(m)
                out = traced(block, h)
                mx.eval(out)
                acts[i + 1].put(m, out)
                if sample:
                    hp = plain.get(m)
                    op = block(hp, create_attention_mask(hp, None))
                    mx.eval(op)
                    plain.put(m, op)
                    del hp, op
                del h, out
            _release(block)
            mx.clear_cache()
        if sample:
            keys = mx.random.split(mx.random.key(seed), n_mb)
            targets = []
            for m, k in enumerate(keys):
                y = mx.random.categorical(head(model, plain.get(m)).astype(mx.float32), key=k)
                mx.eval(y)
                targets.append(y)
            plain.clear()
            if tpath is not None:
                np.save(tpath, np.concatenate([np.array(y) for y in targets]))
    if not sample:
        targets = [t[:, 1:] for t in x_mb]
    elif prog["forward"]:                           # resumed after a finished forward sweep
        arr = np.load(tpath)
        targets = [mx.array(arr[s:s + micro_batch]) for s in range(0, arr.shape[0], micro_batch)]
    if not prog["forward"]:
        prog["forward"] = True
        save_prog()
    t_fwd = time.perf_counter() - t0

    t1 = time.perf_counter()
    groups = []
    with _integer_indices_without_gradient():
        if prog["reverse_next"] is None:          # upstream gradient at the final hidden state
            cot = buf("cot", n_layers)
            for m in range(n_mb):
                g = mx.grad(lambda h, y=targets[m]: _mean_ce(head(model, h), y))(acts[n_layers].get(m))
                mx.eval(g)
                cot.put(m, g)
            prog["reverse_next"] = n_layers - 1
            save_prog()
        else:                                     # resumed: upstream gradient of the first unfinished block
            cot = buf("cot", prog["reverse_next"] + 1)
        for i in range(prog["reverse_next"], -1, -1):
            block = load_block(model, model_dir, i)
            names = by_block.get(i, [])
            pre = f"model.layers.{i}."
            targets_of = {n: (p[len(pre):], r) for n in names for p, r in [_param_path(n)]}
            keys = list(dict.fromkeys(p for p, _ in targets_of.values()))
            flat = dict(tree_flatten(block.parameters()))
            orig = [flat[k] for k in keys]
            acc: dict[str, mx.array] = {}
            new = buf("cot", i)

            def fn(*args):
                *ps, hh = args
                if ps:
                    block.update(tree_unflatten(list(zip(keys, ps))))
                return block(hh, create_attention_mask(hh, None))

            for m in range(n_mb):
                h = acts[i].get(m)
                _, vjps = mx.vjp(fn, [*orig, h], [cot.get(m)])
                grads, g_in = vjps[:-1], vjps[-1]
                by_key = dict(zip(keys, grads))
                for n in names:
                    path, row = targets_of[n]
                    gn = (by_key[path] if row is None else by_key[path][row]).astype(mx.float32)
                    acc[n] = gn * gn if n not in acc else acc[n] + gn * gn
                mx.eval(g_in, list(acc.values()))
                new.put(m, g_in)
                del h, vjps, grads, g_in, by_key
            if keys:
                block.update(tree_unflatten(list(zip(keys, orig))))
            if names:
                for n in names:        # in place, one tensor at a time: never two copies of a block's F (an MoE block
                    acc[n] = acc[n] / n_mb   # of DeepSeek-V4 holds 6.4 B expert weights: 26 GB of float32 F)
                    mx.eval(acc[n])
                f = acc
                on_group(f"block_{i:05d}", f)
                groups.append(f"block_{i:05d}")
                del f
            del acc, orig, flat
            _release(block)
            prog["reverse_next"] = i - 1
            save_prog()
            cot.clear()
            acts[i].clear()
            cot = new
            mx.clear_cache()
        if head_names and not prog["head"]:
            emb = model.model.embed_tokens
            acc = {}
            for m in range(n_mb):
                h_last, y, x = acts[n_layers].get(m), targets[m], inputs[m]
                g = {}
                if "model.embed_tokens.weight" in head_names:
                    w0 = emb.weight

                    def lookup(w):
                        emb.update({"weight": w})
                        return emb(x)
                    _, (g_lookup,) = mx.vjp(lookup, [w0], [cot.get(m)])
                    emb.update({"weight": w0})
                    g["model.embed_tokens.weight"] = g_lookup
                hm = model.model.embed_tokens if model.args.tie_word_embeddings else model.lm_head
                want_head = model.args.tie_word_embeddings and "model.embed_tokens.weight" in head_names \
                    or "lm_head.weight" in head_names
                if want_head:
                    w1 = hm.weight

                    def head_loss(w):
                        hm.update({"weight": w})
                        return _mean_ce(head(model, h_last), y)
                    g_head = mx.grad(head_loss)(w1)
                    hm.update({"weight": w1})
                    if model.args.tie_word_embeddings:
                        g["model.embed_tokens.weight"] = g_head + g["model.embed_tokens.weight"]
                    else:
                        g["lm_head.weight"] = g_head
                for n in head_names:
                    gn = g[n].astype(mx.float32)
                    acc[n] = gn * gn if n not in acc else acc[n] + gn * gn
                mx.eval(list(acc.values()))
                del g
            f = {n: acc[n] / n_mb for n in head_names}
            mx.eval(list(f.values()))
            on_group("head", f)
            groups.append("head")
            prog["head"] = True
            save_prog()
            del f, acc
    for a in acts:
        a.clear()
    if work is not None:
        shutil.rmtree(work / "cot", ignore_errors=True)
    mx.clear_cache()
    return {"groups": groups, "forward_seconds": t_fwd, "reverse_seconds": time.perf_counter() - t1,
            "peak_memory_bytes": int(mx.get_peak_memory())}


def fisher_proxy(delta_w: mx.array, fisher: mx.array) -> float:
    """½ Σ F ⊙ ΔW² (second-order predicted loss change; scale as documented in the module docstring)."""
    d = delta_w.astype(mx.float32)
    return 0.5 * mx.sum(fisher.astype(mx.float32) * d * d).item()


def fisher_summary(fisher: mx.array) -> dict:
    """Sum, mean, p50/p99 over entries and quantiles of the per-input-column sums Σ_i F_ij."""
    f = np.array(fisher.astype(mx.float32), dtype=np.float64)
    col = f.reshape(-1, f.shape[-1]).sum(axis=0)
    return {"total": float(f.sum()), "mean": float(f.mean()), "p50": float(np.percentile(f, 50)),
            "p99": float(np.percentile(f, 99)), "col_sum_p50": float(np.percentile(col, 50)),
            "col_sum_p99": float(np.percentile(col, 99)), "col_sum_max": float(col.max())}
