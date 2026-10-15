"""Q6 representation fine-tuning (README §2, spec D3, D8): tune the CONTINUOUS parts of SUC encodings.

The discrete indices (and residual positions) stay fixed; codebooks (M, K, G), row scales and sparse
residual values are tuned against a behavioural loss on calibration data:

  block-wise (AQLM-style, layer-streamed): for block i, X_i = output of the already tuned quantized blocks
    0..i-1, Y_i = BF16 block i on X_i (teacher weights from `reference_weights`); minimise
    sum||block_q(X_i) - Y_i||² / sum||Y_i||² per micro-batch with Adam; then X_{i+1} = block_q(X_i).
  end-to-end: mean next-token KL(P_teacher || P_q) over calibration positions, all blocks jointly.

TrainableSucLinear decodes its weight on the fly and differentiably. Stored-precision rounding is applied
in the forward pass with a straight-through estimator (fp16 codebooks / row scales / residual values,
int8 residual values with their per-tensor scale max|v|/127 recomputed from the current values), so
decode(to_encoded()) equals weight() and the stored bits never change. Parameters are float32 masters:
  codebooks   (M, K, G) as stored (entries are O(1): row-normalised, rotated space)
  row_scales  (out,) multiplicative factor on the initial fp16 scales (init 1)
  res_val     (n,) residual values divided by the tensor's initial max|R| (init in [-1, 1])
  res_colval  (c, out) column-structured residual values (ResidualCodec structure="columns") divided by each
              column's initial max|R| (init in [-1, 1]); int8 columns are rounded with their per-column fp16 scale
              max|v|/127 recomputed from the current values (the encode rule), fp16 columns to fp16. The column
              indices, like the CSR positions, stay fixed.
so a single Adam learning rate acts as a relative step size on every part (tune part "residual" = res_val and
res_colval).
The codebook-gather VJP is a fixed-order chunked one-hot GEMM (MLX's atomic scatter-add is not
run-to-run deterministic on the GPU); FinetuneConfig.deterministic=False uses the faster atomic path.

Scope: dense nn.Linear tensors and MoE experts of the encodings. The experts of one mlx-lm SwitchLinear stack
(checkpoint rows "…experts.{e}.<proj>.weight", all E given) are tuned through a TrainableSucSwitchLinear: each call
decodes only the routed experts, so only those receive gradients (unrouted experts stay bit-identical). Expert rows
of an incomplete stack are set to their (untuned) decode for the student and to the reference for the teacher, and
listed in log["skipped"]. The router is never an encoding (not an eligible role) and stays frozen.
On return the model holds decode(tuned encoding) of every encoded tensor (cast to the parameter dtype);
all other parameters are untouched and the frozen/trainable state of the model is restored.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models.base import create_attention_mask

from suc.analysis.fisher import _integer_indices_without_gradient, _set_trainable
from suc.analysis.sensitivity import _resolve, get_weight, set_weight
from suc.models.streaming import block_prefix, embed, model_blocks
from suc.quant.codecs import Encoded
from suc.quant.hadamard import rht, rht_inverse, rotate_weight, unrotate_weight
from suc.quant.kmeans import nearest
from suc.quant.residual import _PREFIX, _ResidualEncoded, _int8_sym

_EXPERT = re.compile(r"^(.*)\.experts\.(\d+)\.(\w+)\.weight$")
_TUNE_KEYS = {"codebooks": ("codebooks",), "row_scales": ("row_scales",), "residual": ("res_val", "res_colval")}


def _ste(x: mx.array, q: mx.array) -> mx.array:
    """Forward q, backward identity w.r.t. x."""
    return x + mx.stop_gradient(q - x)


def _ste16(x: mx.array) -> mx.array:
    return _ste(x, x.astype(mx.float16).astype(mx.float32))


_GATHER_CHUNK = 1 << 24  # one-hot elements per chunk in the deterministic gather VJP


@mx.custom_function
def _gather(cb: mx.array, idx: mx.array) -> mx.array:
    """cb[idx] (K, G) x (N,) -> (N, G). MLX's own VJP is an atomic scatter-add whose float summation order
    varies run to run on the GPU; this VJP sums chunked one-hot GEMMs in a fixed order (deterministic)."""
    return cb[idx]


@_gather.vjp
def _gather_vjp(primals, cotangent, _output):
    cb, idx = primals
    k = cb.shape[0]
    step = max(1, _GATHER_CHUNK // k)
    codes = mx.arange(k)[None]
    g = mx.zeros(cb.shape, cotangent.dtype)
    for s in range(0, idx.shape[0], step):
        g = g + (idx[s:s + step, None] == codes).astype(cotangent.dtype).T @ cotangent[s:s + step]
    return g.astype(cb.dtype), mx.zeros_like(idx)


def _int8_quant(v: mx.array) -> tuple[mx.array, mx.array]:
    """ResidualCodec int8 rule: (codes int8, fp16 scale max|v|/127)."""
    s16 = (mx.max(mx.abs(v)) / 127 if v.size else mx.array(0.0)).astype(mx.float16)
    s = s16.astype(mx.float32)
    return mx.clip(mx.round(v / mx.where(s > 0, s, 1.0)), -127, 127).astype(mx.int8), s16


class TrainableSucLinear(nn.Module):
    """Linear whose weight is decoded on the fly from trainable codebooks / row scales / residual values."""

    def __init__(self, e: Encoded, bias: mx.array | None = None,
                 tune: tuple[str, ...] = ("codebooks", "row_scales", "residual"), deterministic: bool = True) -> None:
        super().__init__()
        self._take = _gather if deterministic else (lambda c, i: c[i])
        res = isinstance(e, _ResidualEncoded)
        be = e.base() if res else e
        if be.codec not in ("vq", "additive"):
            raise ValueError(f"TrainableSucLinear supports vq / additive bases, got {be.codec!r}")
        bad = set(tune) - set(_TUNE_KEYS)
        if bad:
            raise ValueError(f"unknown tune parts {sorted(bad)}; expected a subset of {sorted(_TUNE_KEYS)}")
        self._enc, self._base_params, self._shape = e, be.params, tuple(e.shape)
        self._idx_dtype = be.arrays["indices"].dtype
        m = be.arrays["codebooks"].shape[0]
        self._idx = be.arrays["indices"].astype(mx.int32).reshape(m, -1)
        self.codebooks = be.arrays["codebooks"].astype(mx.float32)
        self._s0 = be.arrays["row_scales"].astype(mx.float32)
        self.row_scales = mx.ones_like(self._s0)
        self._res_dtype = self._col_dtype = None
        if res and e.arrays.get("res_cols") is not None and e.arrays["res_cols"].size:
            out = self._shape[0]
            v = e.arrays["res_colval"].astype(mx.float32).reshape(-1, out)
            if "res_colscale" in e.arrays:
                v = v * e.arrays["res_colscale"].astype(mx.float32)[:, None]
            c = mx.max(mx.abs(v), axis=1, keepdims=True)
            self._col_c = mx.where(c > 0, c, 1.0)
            self._cols = e.arrays["res_cols"].astype(mx.int32)
            self._col_dtype = e.params["value_dtype"]
            self.res_colval = v / self._col_c
        if res and e.params["n"] > 0:
            v = e.arrays["res_val"].astype(mx.float32)
            if "res_scale" in e.arrays:
                v = v * e.arrays["res_scale"].astype(mx.float32)
            c = mx.max(mx.abs(v))
            self._res_c = mx.where(c > 0, c, 1.0)
            self._res_pos = e.arrays["res_pos"].astype(mx.int32)
            self._res_dtype = e.params["value_dtype"]
            self.res_val = v / self._res_c
        if bias is not None:
            self.bias = bias
        self._tune = tuple(k for t in tune for k in _TUNE_KEYS[t] if k in self)
        self.activate()

    @classmethod
    def from_encoded(cls, e: Encoded, bias: mx.array | None = None,
                     tune: tuple[str, ...] = ("codebooks", "row_scales", "residual"),
                     deterministic: bool = True) -> "TrainableSucLinear":
        return cls(e, bias, tune, deterministic)

    def activate(self) -> None:
        """Make exactly the tuned parts trainable."""
        self.freeze()
        if self._tune:
            self.unfreeze(keys=list(self._tune), recurse=False)

    # -- decode ------------------------------------------------------------------------------
    def _scales(self) -> mx.array:
        return _ste16(self._s0 * self.row_scales)

    def _res_values(self) -> mx.array:
        v = self.res_val * self._res_c
        if self._res_dtype == "fp16":
            return _ste16(v)
        q, s16 = _int8_quant(mx.stop_gradient(v))
        return _ste(v, q.astype(mx.float32) * s16.astype(mx.float32))

    def _col_values(self) -> mx.array:
        """(c, out) dequantized column residual: fp16, or int8 with one fp16 scale per column (max|v| / 127,
        recomputed from the current values as ResidualCodec does), straight-through."""
        v = self.res_colval * self._col_c
        if self._col_dtype == "fp16":
            return _ste16(v)
        q, s16 = _int8_sym(mx.stop_gradient(v), axis=1)
        return _ste(v, q.astype(mx.float32) * s16.astype(mx.float32)[:, None])

    def _residual(self, w: mx.array, sign: float = 1.0, frozen: bool = False) -> mx.array:
        """w + sign * (column residual, then CSR residual), in the codec's decode order."""
        if self._col_dtype is not None:
            cv = self._col_values()
            cv = mx.stop_gradient(cv) if frozen else cv
            w = w.at[:, self._cols].add(sign * cv.T)
        if self._res_dtype is not None:
            rv = self._res_values()
            rv = mx.stop_gradient(rv) if frozen else rv
            w = w.reshape(-1).at[self._res_pos].add(sign * rv).reshape(self._shape)
        return w

    def _unrotate(self, x: mx.array) -> mx.array:
        p = self._base_params
        if not p["rotate"]:
            return x
        if p.get("rotate_out", True):
            return unrotate_weight(x, p["seed_in"], p["seed_out"])
        return rht_inverse(x, p["seed_in"])

    def weight(self) -> mx.array:
        """Differentiable decode (out, in), float32, in the original (unrotated) basis."""
        out, inn = self._shape
        cb = _ste16(self.codebooks)
        blocks = sum(self._take(cb[m], self._idx[m]) for m in range(cb.shape[0]))
        return self._residual(self._unrotate(blocks.reshape(out, inn) * self._scales()[:, None]))

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight().astype(x.dtype).T
        return y + self.bias if "bias" in self else y

    # -- discrete update -----------------------------------------------------------------------
    def reassign(self, w_ref: mx.array) -> float:
        """Re-assign indices to the nearest code of the reference weight minus the current residual, in the
        rotated row-normalised space (one coordinate-descent pass over the M codebooks). Returns the
        fraction of changed indices."""
        out, inn = self._shape
        p = self._base_params
        w = self._residual(w_ref.astype(mx.float32), -1.0, frozen=True)
        if p["rotate"]:
            w = rotate_weight(w, p["seed_in"], p["seed_out"]) if p.get("rotate_out", True) else rht(w, p["seed_in"])
        s = mx.stop_gradient(self._scales())
        blocks = (w / mx.where(s > 0, s, 1.0)[:, None]).reshape(-1, self.codebooks.shape[-1])
        cb = self.codebooks.astype(mx.float16).astype(mx.float32)
        old, idx = self._idx, [self._idx[m] for m in range(cb.shape[0])]
        for m in range(cb.shape[0]):
            target = blocks - sum((cb[j][idx[j]] for j in range(cb.shape[0]) if j != m), mx.zeros_like(blocks))
            idx[m] = nearest(target, cb[m])[0].astype(mx.int32)
        self._idx = mx.stack(idx)
        mx.eval(self._idx)
        return float(mx.mean((self._idx != old).astype(mx.float32)))

    # -- back to the codec layout --------------------------------------------------------------
    def to_encoded(self) -> Encoded:
        """Codec layout with the tuned fp16/int8 arrays; shapes, dtypes and bits are unchanged."""
        e = self._enc
        base = {"codebooks": self.codebooks.astype(mx.float16),
                "indices": self._idx.reshape(e.arrays[(_PREFIX if isinstance(e, _ResidualEncoded) else "")
                                                      + "indices"].shape).astype(self._idx_dtype),
                "row_scales": (self._s0 * self.row_scales).astype(mx.float16)}
        if not isinstance(e, _ResidualEncoded):
            return Encoded(e.codec, e.shape, dict(e.params), base)
        arrays = dict(e.arrays)
        arrays.update({_PREFIX + k: v for k, v in base.items()})
        if self._res_dtype is not None:
            v = mx.stop_gradient(self.res_val * self._res_c)
            if self._res_dtype == "fp16":
                arrays["res_val"] = v.astype(mx.float16)
            else:
                arrays["res_val"], arrays["res_scale"] = _int8_quant(v)
        if self._col_dtype is not None:
            v = mx.stop_gradient(self.res_colval * self._col_c)
            if self._col_dtype == "fp16":
                arrays["res_colval"] = v.astype(mx.float16)
            else:
                arrays["res_colval"], arrays["res_colscale"] = _int8_sym(v, axis=1)
        return _ResidualEncoded(e.codec, e.shape, dict(e.params), arrays)


class TrainableSucSwitchLinear(nn.Module):
    """Stand-in for an mlx-lm SwitchLinear (MoE expert stack (E, out, in)) whose experts are TrainableSucLinear.

    Each call decodes only the experts routed in that call (host-side np.unique of the indices; the router output
    carries no gradient) and runs mx.gather_mm on the decoded (U, out, in) stack with remapped indices, so only
    routed experts enter the graph and receive gradients. An expert never routed during tuning gets exact zero
    gradients, and Adam (zero moments) leaves it bit-identical. Memory: the decoded weights of the routed experts
    of the block (float32) plus their gradients, i.e. block-local (D8); for a 128-expert block nearly all experts
    are routed in a micro-batch, so this is ~2 x 4 bytes x the block's expert parameters."""

    def __init__(self, encodings: list[Encoded], names: list[str], bias: mx.array | None = None,
                 tune: tuple[str, ...] = ("codebooks", "row_scales", "residual"), deterministic: bool = True) -> None:
        super().__init__()
        if len(encodings) != len(names) or not encodings:
            raise ValueError("one name per expert encoding is required")
        self.experts = [TrainableSucLinear(e, tune=tune, deterministic=deterministic) for e in encodings]
        self._names = tuple(names)
        if bias is not None:
            self.bias = bias
        self.activate()

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def activate(self) -> None:
        self.freeze()
        for t in self.experts:
            t.activate()

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        idx = mx.stop_gradient(indices)
        used = np.unique(np.asarray(idx)).astype(np.int64)
        remap = np.zeros(len(self.experts), np.uint32)
        remap[used] = np.arange(used.size, dtype=np.uint32)
        w = mx.stack([self.experts[int(e)].weight() for e in used]).astype(x.dtype)
        y = mx.gather_mm(x, w.swapaxes(-1, -2), rhs_indices=mx.array(remap)[idx.astype(mx.uint32)],
                         sorted_indices=sorted_indices)
        if "bias" in self:
            y = y + mx.expand_dims(self.bias[idx], -2)
        return y


@dataclass(frozen=True)
class FinetuneConfig:
    block_steps: int = 100            # Adam steps per block (block-wise stage)
    e2e_steps: int = 0                # Adam steps of the end-to-end KL stage
    lr: float = 1e-4                  # relative step size (see module docstring)
    micro_batch: int = 4              # sequences per step
    sequences: int = 64               # calibration sequences used (first rows of calib_tokens)
    tune: tuple[str, ...] = ("codebooks", "row_scales", "residual")
    reassign: bool = False            # re-assign indices after block tuning (kept only if the loss does not rise)
    seed: int = 0                     # micro-batch order
    deterministic: bool = True        # fixed-order codebook gradient (about 2x slower step than MLX's atomic scatter)


# ---------------------------------------------------------------------------------------- helpers

def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    *path, attr, _ = name.split(".")
    cur = model
    for p in path:
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    return cur, attr


def _split(model: nn.Module, names: list[str]) -> tuple[list[str], list[str]]:
    """(dense nn.Linear tensors, other tensors such as MoE expert rows)."""
    dense, other = [], []
    for n in names:
        holder, _, row = _resolve(model, n)
        (dense if row is None and type(holder) is nn.Linear else other).append(n)
    return dense, other


def _expert_stacks(model: nn.Module, names: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """({stack tensor "<p>.switch_mlp.<proj>.weight": expert names in expert order}, remaining names) for the
    expert rows among `names` whose stack is an mlx-lm SwitchLinear with ALL its experts listed; other expert rows
    (incomplete stacks) and non-expert names are returned as remaining."""
    from mlx_lm.models.switch_layers import SwitchLinear
    rows: dict[str, dict[int, str]] = {}
    rest = []
    for n in names:
        if (m := _EXPERT.match(n)) is None:
            rest.append(n)
        else:
            rows.setdefault(f"{m.group(1)}.switch_mlp.{m.group(3)}.weight", {})[int(m.group(2))] = n
    stacks = {}
    for stack, r in rows.items():
        parent, attr = _parent(model, stack)
        lin = parent[attr]
        if type(lin) is SwitchLinear and sorted(r) == list(range(lin.weight.shape[0])):
            stacks[stack] = [r[e] for e in range(len(r))]
        else:
            rest += list(r.values())
    return stacks, rest


def _set_all(model: nn.Module, weights: dict[str, mx.array]) -> None:
    """set_weight for dense tensors; expert rows are written once per stacked tensor."""
    stacks: dict[str, dict[int, mx.array]] = {}
    for n, w in weights.items():
        if (m := _EXPERT.match(n)) is not None:
            stacks.setdefault(f"{m.group(1)}.switch_mlp.{m.group(3)}.weight", {})[int(m.group(2))] = w
        else:
            set_weight(model, n, w)
    for stack, rows in stacks.items():
        cur = get_weight(model, stack)
        set_weight(model, stack, mx.stack([rows[r].astype(cur.dtype) if r in rows else cur[r]
                                           for r in range(cur.shape[0])]))
    mx.eval([get_weight(model, n) for n in weights])


def _tunable(e: Encoded) -> bool:
    """vq / additive base (optionally with a sparse residual): the encodings TrainableSucLinear supports."""
    return (e.base() if isinstance(e, _ResidualEncoded) else e).codec in ("vq", "additive")


def _decode(e: Encoded) -> mx.array:
    """Codec decode of a vq / additive (+ residual) encoding (same arithmetic as the codecs)."""
    return mx.stop_gradient(TrainableSucLinear(e, tune=()).weight())


def _install(model: nn.Module, encodings: dict[str, Encoded], names: list[str], tune: tuple[str, ...],
             deterministic: bool = True) -> dict[str, tuple[nn.Module, str, nn.Module, TrainableSucLinear]]:
    out = {}
    for n in names:
        parent, attr = _parent(model, n)
        lin = parent[attr]
        t = TrainableSucLinear.from_encoded(encodings[n], bias=lin.get("bias"), tune=tune, deterministic=deterministic)
        parent[attr] = t
        out[n] = (parent, attr, lin, t)
    return out


def _install_experts(model: nn.Module, encodings: dict[str, Encoded], stacks: dict[str, list[str]],
                     tune: tuple[str, ...], deterministic: bool = True) -> dict:
    """Replace each SwitchLinear of `stacks` by a TrainableSucSwitchLinear; keyed by the stack tensor name."""
    out = {}
    for stack, names in stacks.items():
        parent, attr = _parent(model, stack)
        lin = parent[attr]
        t = TrainableSucSwitchLinear([encodings[n] for n in names], names, bias=lin.get("bias"), tune=tune,
                                     deterministic=deterministic)
        parent[attr] = t
        out[stack] = (parent, attr, lin, t)
    return out


def _leaves(installed: dict):
    """(checkpoint tensor name, TrainableSucLinear) of every installed tensor (experts of a stack in order)."""
    for n, (*_, t) in installed.items():
        if isinstance(t, TrainableSucSwitchLinear):
            yield from zip(t.names, t.experts)
        else:
            yield n, t


def _uninstall(model: nn.Module, installed: dict) -> dict[str, Encoded]:
    tuned = {}
    rows = {}
    for n, (parent, attr, lin, t) in installed.items():
        parent[attr] = lin
        if isinstance(t, TrainableSucSwitchLinear):
            for name, leaf in zip(t.names, t.experts):
                rows[name] = mx.stop_gradient(leaf.weight())
                tuned[name] = leaf.to_encoded()
            continue
        set_weight(model, n, mx.stop_gradient(t.weight()))
        tuned[n] = t.to_encoded()
    if rows:
        _set_all(model, rows)   # one stack copy per projection
    mx.eval([get_weight(model, n) for n in installed])
    return tuned


def _activate(installed: dict, root: nn.Module) -> None:
    root.freeze()
    for *_, t in installed.values():
        t.activate()


def _snapshot(installed: dict) -> dict:
    return {n: (tree_map(lambda a: mx.array(a), t.parameters()), t._idx) for n, t in _leaves(installed)}


def _restore(installed: dict, snap: dict) -> None:
    for n, t in _leaves(installed):
        params, idx = snap[n]
        t.update(params)
        t._idx = idx


def _order(n: int, steps: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    out: list[int] = []
    while len(out) < steps:
        out += rng.permutation(n).tolist()
    return out[:steps]


def _rel(out: mx.array, y: mx.array) -> tuple[mx.array, mx.array]:
    d = out.astype(mx.float32) - y.astype(mx.float32)
    yf = y.astype(mx.float32)
    return mx.sum(d * d), mx.sum(yf * yf)


def _block_loss(block: nn.Module, xs: list[mx.array], ys: list[mx.array]) -> float:
    num = den = 0.0
    for x, y in zip(xs, ys):
        a, b = _rel(block(x, create_attention_mask(x, None)), y)
        mx.eval(a, b)
        num, den = num + a.item(), den + b.item()
    return num / max(den, 1e-30)


def _trainable_paths(model: nn.Module) -> list[str]:
    return [k for k, _ in tree_flatten(model.trainable_parameters())]


# ---------------------------------------------------------------------------------------- stages

def finetune_blockwise(model: nn.Module, encodings: dict[str, Encoded] | Callable[[int], dict[str, Encoded]],
                       calib_tokens: mx.array, cfg: FinetuneConfig, *,
                       reference_weights: Callable[[str], mx.array],
                       fixed_weights: Callable[[str], mx.array] | None = None,
                       after_block: Callable[[int, nn.Module], None] | None = None,
                       on_block_tuned: Callable[[int, dict[str, Encoded]], None] | None = None
                       ) -> tuple[dict[str, Encoded], dict]:
    """Block-wise output matching (module docstring). `reference_weights(name)` returns the original weight
    of a checkpoint tensor name (e.g. SafetensorsStore.load). Returns (tuned encodings, log) with
    log = {"blocks": {i: {loss_before, loss_after_tune, loss_after, reverted, [reassign], seconds, tensors, fixed}},
           "skipped": [{name, reason}], "peak_memory_bytes", "seconds"}. Losses are over all used sequences.

    Encodings whose base codec is not vq / additive (e.g. affine tensors of a mixed allocation plan) are not
    tunable: the student uses fixed_weights(name) (their decode, supplied by the caller), the teacher the
    reference, and they are returned unchanged (ValueError if such encodings exist and fixed_weights is None).
    after_block(i, block) runs once block i is finished and its output computed (e.g. to release the block's
    weights of a lazily loaded model, D8); the block is not used again afterwards.

    Streaming (D8): `encodings` may be a callable i -> {name: Encoded} of block i only (e.g. read from that block's
    encode shard), and on_block_tuned(i, {name: tuned Encoded}) receives each block's result as soon as it is
    final; with on_block_tuned the returned dict is empty, so no more than one block of encodings is resident."""
    t_start = time.perf_counter()
    mx.reset_peak_memory()
    tokens = calib_tokens[:cfg.sequences]
    trainable_before = _trainable_paths(model)
    xs = [embed(model, tokens[s:s + cfg.micro_batch]) for s in range(0, tokens.shape[0], cfg.micro_batch)]
    mx.eval(xs)
    streamed = callable(encodings)
    tuned = {} if on_block_tuned is not None else ({} if streamed else dict(encodings))
    log: dict = {"blocks": {}, "skipped": []}
    try:
        with _integer_indices_without_gradient():
            for i, block in enumerate(model_blocks(model)):
                t0 = time.perf_counter()
                pre = block_prefix(i) + "."
                block_enc = encodings(i) if streamed else {n: e for n, e in encodings.items() if n.startswith(pre)}
                names = [n for n in block_enc if n.startswith(pre)]
                fixed = [n for n in names if not _tunable(block_enc[n])]
                if fixed and fixed_weights is None:
                    raise ValueError(f"non-codebook encodings {fixed[:3]} need fixed_weights (their decode)")
                dense, other = _split(model, [n for n in names if n not in fixed])
                stacks, other = _expert_stacks(model, other)
                log["skipped"] += [{"name": n, "reason": "not a dense nn.Linear nor part of a complete MoE "
                                    "expert stack: decoded without tuning"} for n in other]
                _set_all(model, {n: reference_weights(n) for n in names})
                ys = []
                for x in xs:
                    y = block(x, create_attention_mask(x, None))
                    mx.eval(y)
                    ys.append(y)
                _set_all(model, {n: _decode(block_enc[n]) for n in other})
                _set_all(model, {n: fixed_weights(n) for n in fixed})
                installed = _install(model, block_enc, dense, cfg.tune, cfg.deterministic)
                installed.update(_install_experts(model, block_enc, stacks, cfg.tune, cfg.deterministic))
                entry = {"tensors": len(dense) + sum(map(len, stacks.values())), "fixed": len(fixed),
                         "loss_before": _block_loss(block, xs, ys)}
                if stacks:
                    entry["expert_tensors"] = sum(map(len, stacks.values()))
                if installed and cfg.block_steps > 0:
                    snap = _snapshot(installed)
                    _activate(installed, block)

                    def loss_fn(x, y):
                        a, b = _rel(block(x, create_attention_mask(x, None)), y)
                        return a / b

                    vg = nn.value_and_grad(block, loss_fn)
                    opt = optim.Adam(learning_rate=cfg.lr, bias_correction=True)
                    for j in _order(len(xs), cfg.block_steps, cfg.seed + i):
                        loss, g = vg(xs[j], ys[j])
                        opt.update(block, g)
                        mx.eval(block.trainable_parameters(), opt.state, loss)
                    entry["loss_after_tune"] = _block_loss(block, xs, ys)
                    entry["reverted"] = entry["loss_after_tune"] >= entry["loss_before"]
                    if entry["reverted"]:
                        _restore(installed, snap)
                        entry["loss_after_tune"] = entry["loss_before"]
                    if cfg.reassign:
                        before = {n: t._idx for n, t in _leaves(installed)}
                        changed = [t.reassign(reference_weights(n)) for n, t in _leaves(installed)]
                        new = _block_loss(block, xs, ys)
                        accepted = new <= entry["loss_after_tune"]
                        if not accepted:
                            for n, t in _leaves(installed):
                                t._idx = before[n]
                        entry["reassign"] = {"changed_fraction": float(np.mean(changed)), "loss": new,
                                             "accepted": accepted}
                    entry["loss_after"] = _block_loss(block, xs, ys)
                    del vg, opt, snap
                else:
                    entry["loss_after_tune"] = entry["loss_after"] = entry["loss_before"]
                    entry["reverted"] = False
                block_tuned = {n: block_enc[n] for n in names}
                block_tuned.update(_uninstall(model, installed))
                if on_block_tuned is not None:
                    on_block_tuned(i, block_tuned)
                else:
                    tuned.update(block_tuned)
                del block_tuned, block_enc
                new_xs = []
                for x in xs:
                    h = block(x, create_attention_mask(x, None))
                    mx.eval(h)
                    new_xs.append(h)
                xs = new_xs
                del ys
                if after_block is not None:
                    after_block(i, block)
                mx.clear_cache()
                entry["seconds"] = time.perf_counter() - t0
                log["blocks"][i] = entry
    finally:
        _set_trainable(model, trainable_before)
    log["peak_memory_bytes"] = int(mx.get_peak_memory())
    log["seconds"] = time.perf_counter() - t_start
    return tuned, log


def _kl_loss(logits_q: mx.array, logits_t: mx.array) -> mx.array:
    lt = nn.log_softmax(logits_t.astype(mx.float32), axis=-1)
    lq = nn.log_softmax(logits_q.astype(mx.float32), axis=-1)
    return mx.mean(mx.sum(mx.exp(lt) * (lt - lq), axis=-1))


def finetune_e2e(model: nn.Module, encodings: dict[str, Encoded], calib_tokens: mx.array,
                 teacher_logits_fn: Callable[[mx.array], mx.array], cfg: FinetuneConfig
                 ) -> tuple[dict[str, Encoded], dict]:
    """End-to-end KL(P_teacher || P_q) over all calibration positions, all dense encoded tensors jointly.
    `teacher_logits_fn(tokens (B, T)) -> logits (B, T, V)` is called per micro-batch (never kept for the
    whole set). Returns (tuned encodings, {"kl_before", "kl_after", "reverted", "skipped", "seconds",
    "peak_memory_bytes"}), KL = token mean over the used calibration sequences."""
    t_start = time.perf_counter()
    mx.reset_peak_memory()
    tokens = calib_tokens[:cfg.sequences]
    mbs = [tokens[s:s + cfg.micro_batch] for s in range(0, tokens.shape[0], cfg.micro_batch)]
    trainable_before = _trainable_paths(model)
    dense, other = _split(model, list(encodings))
    stacks, other = _expert_stacks(model, other)
    _set_all(model, {n: _decode(encodings[n]) for n in other})
    installed = _install(model, encodings, dense, cfg.tune, cfg.deterministic)
    installed.update(_install_experts(model, encodings, stacks, cfg.tune, cfg.deterministic))

    def kl_all() -> float:
        tot = 0.0
        for t in mbs:
            k = _kl_loss(model(t), mx.stop_gradient(teacher_logits_fn(t)))
            mx.eval(k)
            tot += k.item() * t.shape[0]
        return tot / tokens.shape[0]

    log: dict = {"skipped": [{"name": n, "reason": "not a dense nn.Linear nor part of a complete MoE expert stack: "
                              "decoded without tuning"} for n in other]}
    try:
        with _integer_indices_without_gradient():
            log["kl_before"] = kl_all()
            if installed and cfg.e2e_steps > 0:
                snap = _snapshot(installed)
                _activate(installed, model)

                def loss_fn(t, lt):
                    return _kl_loss(model(t), lt)

                vg = nn.value_and_grad(model, loss_fn)
                opt = optim.Adam(learning_rate=cfg.lr, bias_correction=True)
                for j in _order(len(mbs), cfg.e2e_steps, cfg.seed):
                    lt = teacher_logits_fn(mbs[j])
                    mx.eval(lt)
                    loss, g = vg(mbs[j], mx.stop_gradient(lt))
                    opt.update(model, g)
                    mx.eval(model.trainable_parameters(), opt.state, loss)
                    del lt, g
                log["kl_after"] = kl_all()
                log["reverted"] = log["kl_after"] >= log["kl_before"]
                if log["reverted"]:
                    _restore(installed, snap)
                    log["kl_after"] = log["kl_before"]
            else:
                log["kl_after"], log["reverted"] = log["kl_before"], False
            tuned = dict(encodings)
            tuned.update(_uninstall(model, installed))
    finally:
        _set_trainable(model, trainable_before)
    log["peak_memory_bytes"] = int(mx.get_peak_memory())
    log["seconds"] = time.perf_counter() - t_start
    return tuned, log
