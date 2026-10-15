"""Local perturbation tests and cheap sensitivity proxies (spec §9 T2.3/T2.4, D5).

A controlled representation error is injected into ONE tensor (gaussian noise or an actual VQ
reconstruction), the model is run on evaluation tokens and the behavioural change is measured
(next-token KL vs the reference logits, cross-entropy change, the layer's own output error).
Proxy predictions of the layer output error from the Hessian proxy H = E[x xᵀ] of the layer
input are provided so that their rank correlation with the measurements can be reported (D5:
a proxy is used for allocation only if that correlation is reported).

Tensor names are in the CHECKPOINT namespace. MoE experts
"…mlp.experts.{e}.{gate,up,down}_proj.weight" map to row e of the stacked mlx-lm tensor
"…mlp.switch_mlp.{gate,up,down}_proj.weight". All statistics are computed in float32.
"""
from __future__ import annotations

import math
import re
from typing import Literal, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from suc.analysis.stats import row_rms_normalize, to_blocks
from suc.quant.hadamard import rotate_weight, unrotate_weight
from suc.quant.kmeans import kmeans, nearest

_EXPERT = re.compile(r"^(.*)\.experts\.(\d+)\.(\w+)\.weight$")


# ---------------------------------------------------------------- perturbations
def perturb_weight(w: mx.array, kind: Literal["gaussian", "vq"], *, seed: int, level: float = 0.0,
                   block_size: int = 8, k: int = 256, rotate: bool = True, fit_blocks: int = 65536) -> mx.array:
    """Return a perturbed copy of the 2-D weight w (same shape and dtype).

    gaussian: w + n with n_ij ~ N(0, (level * rms(row_i))²), hence E[rel_mse] = level² exactly
      (relative Frobenius error ||ΔW||²/||W||²; the realised value fluctuates by O(1/sqrt(numel))).
    vq: a real vector-codebook reconstruction: optional randomized Hadamard rotation
      (seeds 2*seed, 2*seed+1 for in/out dims) -> row-RMS normalisation -> k-means with K=k on at
      most fit_blocks blocks of size block_size (deterministic subsample) -> nearest code for all
      blocks -> reconstruct, rescale rows, unrotate. The per-row RMS is kept exact (float32),
      i.e. its storage precision is not modelled here.
    """
    if kind == "gaussian":
        x = w.astype(mx.float32)
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True))
        noise = mx.random.normal(x.shape, key=mx.random.key(seed))
        return (x + noise * (level * rms)).astype(w.dtype)
    if kind != "vq":
        raise ValueError(f"unknown perturbation kind {kind!r}")
    s_in, s_out = 2 * seed, 2 * seed + 1
    x = rotate_weight(w, s_in, s_out) if rotate else w.astype(mx.float32)
    xn, rms = row_rms_normalize(x)
    blocks = to_blocks(xn, block_size)
    n = blocks.shape[0]
    fit = blocks
    if n > fit_blocks:
        fit = blocks[mx.sort(mx.random.permutation(n, key=mx.random.key(seed))[:fit_blocks])]
    cb = kmeans(fit, k, seed=seed).codebook
    a, _ = nearest(blocks, cb)
    rec = cb[a].reshape(x.shape) * rms[:, None]
    out = unrotate_weight(rec, s_in, s_out) if rotate else rec
    return out.astype(w.dtype)


def weight_error(w: mx.array, w_hat: mx.array) -> dict:
    """{"rel_mse": ||ΔW||²/||W||², "mse": mean(ΔW²)} in float32."""
    x = w.astype(mx.float32)
    d = w_hat.astype(mx.float32) - x
    sq, ref = mx.sum(d * d), mx.sum(x * x)
    mx.eval(sq, ref)
    return {"rel_mse": sq.item() / max(ref.item(), 1e-30), "mse": sq.item() / d.size}


# ---------------------------------------------------------------- proxies
def proxy_output_error(delta_w: mx.array, m2_diag: mx.array) -> float:
    """Diagonal-H prediction of E||ΔW x||²: Σ_j H_jj ||ΔW[:, j]||² (exact iff H is diagonal)."""
    d = delta_w.astype(mx.float32)
    return mx.sum(mx.sum(d * d, axis=0) * m2_diag.astype(mx.float32)).item()


def proxy_output_error_full(delta_w: mx.array, m2_full: mx.array) -> float:
    """tr(ΔW H ΔWᵀ): exact E||ΔW x||² when H = E[x xᵀ]."""
    d = delta_w.astype(mx.float32)
    return mx.sum((d @ m2_full.astype(mx.float32)) * d).item()


# ---------------------------------------------------------------- divergence
def logit_divergence(ref_logits: mx.array, test_logits: mx.array, targets: mx.array | None = None,
                     return_tokens: bool = False) -> dict:
    """Per-token KL(P_ref || P_test) summary, top-1 agreement and (with targets) cross-entropies.

    logits (B, T, V), targets (B, T) = the token each position should predict. Computed in float32,
    one batch row at a time to bound memory for large vocabularies. kl_p95 is the exact 95th percentile over the
    tokens of THIS call; return_tokens adds "kl_tokens" (float32 per-token KL) so that callers aggregating several
    calls can compute a global percentile (suc.evaluation.reconstruct.aggregate_kl).
    """
    kls, agree, ce_r, ce_t = [], [], [], []
    for b in range(ref_logits.shape[0]):
        lr = nn.log_softmax(ref_logits[b].astype(mx.float32), axis=-1)
        lt = nn.log_softmax(test_logits[b].astype(mx.float32), axis=-1)
        kl = mx.sum(mx.exp(lr) * (lr - lt), axis=-1)
        ag = mx.argmax(lr, axis=-1) == mx.argmax(lt, axis=-1)
        outs = [kl, ag]
        if targets is not None:
            tg = targets[b].astype(mx.int32)[:, None]
            outs += [-mx.take_along_axis(lr, tg, axis=-1)[:, 0], -mx.take_along_axis(lt, tg, axis=-1)[:, 0]]
        mx.eval(*outs)
        kls.append(np.array(kl))
        agree.append(np.array(ag))
        if targets is not None:
            ce_r.append(np.array(outs[2]))
            ce_t.append(np.array(outs[3]))
    kl_all = np.maximum(np.concatenate(kls), 0.0)  # clamp tiny negative round-off
    out = {"kl_mean": float(kl_all.mean()), "kl_p95": float(np.percentile(kl_all, 95)),
           "top1_agreement": float(np.concatenate(agree).mean())}
    if return_tokens:
        out["kl_tokens"] = kl_all.astype(np.float32)
    if targets is not None:
        r, t = float(np.concatenate(ce_r).mean()), float(np.concatenate(ce_t).mean())
        out.update({"ce_ref": r, "ce_test": t, "delta_ce": t - r})
    return out


# ---------------------------------------------------------------- get / set
def _resolve(model: nn.Module, tensor_name: str) -> tuple[nn.Module, str, int | None]:
    """(module holding the parameter, parameter key, expert row or None)."""
    row = None
    if (m := _EXPERT.match(tensor_name)) is not None:
        tensor_name, row = f"{m.group(1)}.switch_mlp.{m.group(3)}.weight", int(m.group(2))
    *path, key = tensor_name.split(".")
    cur = model
    try:
        for p in path:
            cur = cur[int(p)] if isinstance(cur, list) else cur[p]
        if not isinstance(cur, nn.Module) or not isinstance(cur.get(key), mx.array):
            raise KeyError(key)
    except (KeyError, IndexError, ValueError, TypeError):
        raise KeyError(f"tensor {tensor_name!r} not found in model") from None
    if row is not None and not 0 <= row < cur[key].shape[0]:
        raise KeyError(f"expert {row} out of range for {tensor_name!r}")
    return cur, key, row


def get_weight(model: nn.Module, tensor_name: str) -> mx.array:
    holder, key, row = _resolve(model, tensor_name)
    return holder[key] if row is None else holder[key][row]


def set_weight(model: nn.Module, tensor_name: str, w: mx.array) -> None:
    """Replace the tensor (cast to the stored dtype). For an expert only its row of the stacked
    tensor changes; the stack is rebuilt functionally so the previous array object is untouched."""
    holder, key, row = _resolve(model, tensor_name)
    cur = holder[key]
    shape = cur.shape if row is None else cur.shape[1:]
    if tuple(w.shape) != tuple(shape):
        raise ValueError(f"shape {tuple(w.shape)} != {tuple(shape)} for {tensor_name!r}")
    w = w.astype(cur.dtype)
    holder[key] = w if row is None else mx.concatenate([cur[:row], w[None], cur[row + 1:]])


# ---------------------------------------------------------------- measurement
class _OutputProbe:
    """Transparent stand-in for a Linear / SwitchLinear that accumulates Σ||ΔW x||² and Σ||W x||²
    over the inputs it actually receives (for an expert only inputs routed to that expert)."""

    def __init__(self, inner: nn.Module, w: mx.array, w_hat: mx.array, row: int | None):
        self.inner, self.row = inner, row
        self.w = w.astype(mx.float32)
        self.dw = w_hat.astype(mx.float32) - self.w
        self.num: list[mx.array] = []
        self.den: list[mx.array] = []

    def __call__(self, x, *args, **kwargs):
        y = self.inner(x, *args, **kwargs)
        xf = x.astype(mx.float32)
        d, r = xf @ self.dw.T, xf @ self.w.T
        d2, r2 = mx.sum(d * d, axis=-1), mx.sum(r * r, axis=-1)
        if self.row is not None:  # gather_mm output batch = broadcast(x.shape[:-2], indices.shape)
            idx = args[0] if args else kwargs["indices"]
            sel = (idx == self.row).astype(mx.float32)[..., None]
            d2, r2 = d2 * sel, r2 * sel
        self.num.append(mx.sum(d2))
        self.den.append(mx.sum(r2))
        return y

    def rel_mse(self) -> float:
        if not self.num:
            return float("nan")  # never called (e.g. an expert that received no tokens)
        num, den = sum(self.num), sum(self.den)
        mx.eval(num, den)
        return num.item() / max(den.item(), 1e-30)


def _probe_parent(model: nn.Module, tensor_name: str) -> tuple[nn.Module, str] | None:
    """Parent module and attribute key of the linear layer owning the tensor, if probe-able."""
    holder, _, row = _resolve(model, tensor_name)
    if row is None and type(holder) is not nn.Linear:
        return None
    if row is not None and type(holder).__name__ != "SwitchLinear":
        return None
    name = _EXPERT.sub(r"\1.switch_mlp.\3.weight", tensor_name)
    *path, attr, _ = name.split(".")
    parent = model
    for p in path:
        parent = parent[int(p)] if isinstance(parent, list) else parent[p]
    return parent, attr


def measure_tensor_sensitivity(model: nn.Module, tensor_name: str, w_hat: mx.array, eval_tokens: mx.array,
                               ref_logits: mx.array, *, layer_output: bool = True) -> dict:
    """Swap w_hat into ONE tensor, forward eval_tokens (B, T) and measure the behavioural change.

    KL/top-1 are over the first T-1 positions, CE targets are eval_tokens[:, 1:]. layer_out_rel_mse
    = Σ||(W_hat - W) x||² / Σ||W x||² on the layer's actual inputs in this forward (weight-only
    product, bias ignored); nan if not measured or the tensor is not a (Switch)Linear weight.
    The original parameter array object is always put back (bit-exact), also on exceptions.
    """
    holder, key, row = _resolve(model, tensor_name)
    original = holder[key]
    w = original if row is None else original[row]
    probe_at = _probe_parent(model, tensor_name) if layer_output else None
    probe = None
    try:
        set_weight(model, tensor_name, w_hat)
        if probe_at is not None:
            parent, attr = probe_at
            probe = _OutputProbe(parent[attr], w, w_hat, row)
            parent[attr] = probe
        logits = model(eval_tokens)
        out = logit_divergence(ref_logits[:, :-1], logits[:, :-1], eval_tokens[:, 1:])
        out["layer_out_rel_mse"] = probe.rel_mse() if probe is not None else float("nan")
    finally:
        if probe is not None:
            probe_at[0][probe_at[1]] = probe.inner
        holder[key] = original
    out.update(weight_error(w, w_hat))
    return out


# ---------------------------------------------------------------- statistics
def _rank(a: np.ndarray) -> np.ndarray:
    """1-based ranks, ties get the average rank."""
    order = np.argsort(a, kind="stable")
    sorted_a = a[order]
    ranks = np.empty(a.size, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson on average ranks); nan if either input is constant."""
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("x and y must be 1-D of equal length")
    ra, rb = _rank(a) - (a.size + 1) / 2, _rank(b) - (b.size + 1) / 2
    den = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb) / den if den > 0 else float("nan")


def sensitivity_score(delta_semantic: float, delta_error: float, eps: float = 1e-12) -> float:
    """S = ΔD_semantic / ΔE (T2.4), with eps guarding a zero injected error."""
    return delta_semantic / max(delta_error, eps)
