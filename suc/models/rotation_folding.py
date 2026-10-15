"""Residual-stream rotation folding (QuaRot-style; README §2, docs/benchmarks.md K5 next step (a)).

Pick ONE randomized Hadamard R = H diag(s) on the hidden size and run the network on the rotated residual stream
h' = R h. Because RMSNorm is invariant under orthogonal maps up to its per-channel weight g (n(Rh) = R n(h)), the
weights can be rewritten exactly (row-vector form: x' = suc.quant.hadamard.rht(x, seed) on the last axis):

  readers  (input = g ⊙ n(h): q/k/v, gate/up, experts' gate/up, router, untied LM head)
           W' = W diag(g) Rᵀ          — rows:    W'_r = rht(g ⊙ W_r);  the norm weight becomes 1
  writers  (output added to h: o_proj, down_proj, experts' down)
           W' = R W                   — columns: W' = rht(Wᵀ)ᵀ   (a bias b would become rht(b))
  embed    E' = E Rᵀ                  — rows:    E'_t = rht(E_t)
  final norm with a TIED LM head (E' serves both ends): logits = E' R (g ⊙ Rᵀ n(h')), i.e. the final norm stays
           unfolded and is wrapped by two online rotations per token (RotatedFinalNorm) — once per forward, not
           per block. An untied LM head is a reader (g folded, no online rotation).

Attention-internal norms (Qwen3 q_norm / k_norm, on the head dim after the projections) see q = W_q (g ⊙ n(h)),
which is unchanged, so they are untouched. The router's logits are unchanged as well (it is a reader).

Optional per-head rotation of the value path (v_heads=True): attention is linear in V per head, so a Hadamard R_h on
the head dim folded into v_proj's rows (per kv head: W_v' = R_h W_v) makes the attention output per head R_h o,
and o_proj's columns absorb it (W_o' = W_o blockdiag(R_hᵀ)). o_proj then reads an input already rotated by a
block-diagonal (head_dim-sized) Hadamard.

Every rewrite is per tensor given R and that tensor's norm weight (D8: FoldedStore folds while streaming, one
tensor at a time). Roles come from the architecture adapter (suc.models.adapters), never from hard-coded names.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from suc.models.adapters import ArchitectureAdapter, LinearSpec
from suc.quant.hadamard import rht, rht_inverse, supported_size

# role -> norm position of the normed residual stream it reads
READER_NORMS: dict[str, str] = {
    "attn_q": "attn", "attn_k": "attn", "attn_v": "attn",
    "mlp_gate": "mlp", "mlp_up": "mlp", "expert_gate": "mlp", "expert_up": "mlp",
    "shared_expert_gate": "mlp", "shared_expert_up": "mlp", "router": "mlp", "lm_head": "final",
}
WRITERS: frozenset[str] = frozenset({"attn_o", "mlp_down", "expert_down", "shared_expert_down"})
# roles whose (folded) input is the rotated residual stream: no online input rotation is needed after folding
ROTATED_INPUT_ROLES: frozenset[str] = frozenset(READER_NORMS) | {"embed"}


@dataclass(frozen=True)
class ResidualFold:
    """The folding transform: residual rotation seed (hidden size) and optional per-head value rotation."""
    hidden: int
    seed: int
    v_heads: bool = False
    head_dim: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    residual: bool = True      # False: only the per-head value rotation (v_heads) is folded; residual stream as is

    @classmethod
    def for_adapter(cls, adapter: ArchitectureAdapter, seed: int, v_heads: bool = False,
                    residual: bool = True) -> "ResidualFold":
        c = adapter.config
        hidden = int(c["hidden_size"])
        if not residual and not v_heads:
            raise ValueError("nothing to fold: residual=False needs v_heads=True")
        if residual and not supported_size(hidden):
            raise ValueError(f"hidden size {hidden} has no Hadamard: residual rotation folding unsupported")
        n_heads = int(c["num_attention_heads"])
        head_dim = int(c.get("head_dim") or hidden // n_heads)
        if v_heads and not supported_size(head_dim):
            raise ValueError(f"head_dim {head_dim} has no Hadamard: per-head value rotation unsupported")
        return cls(hidden, int(seed), bool(v_heads), head_dim, n_heads,
                   int(c.get("num_key_value_heads") or n_heads), bool(residual))

    @property
    def head_seed(self) -> int:
        return int(np.random.default_rng(self.seed).integers(0, 2 ** 31 - 1))

    # -- per-tensor rewrites (float32; any leading dims, e.g. stacked experts (E, out, in)) -------------------------
    def rotate(self, x: mx.array) -> mx.array:
        """h' = R h on the last axis."""
        return rht(x, self.seed)

    def unrotate(self, x: mx.array) -> mx.array:
        """h = Rᵀ h' on the last axis."""
        return rht_inverse(x, self.seed)

    def fold_reader(self, w: mx.array, norm: mx.array | None) -> mx.array:
        w = w.astype(mx.float32)
        if norm is not None:
            w = w * norm.astype(mx.float32)
        return rht(w, self.seed)

    def fold_writer(self, w: mx.array) -> mx.array:
        return mx.swapaxes(rht(mx.swapaxes(w.astype(mx.float32), -1, -2), self.seed), -1, -2)

    def _v_rows(self, w: mx.array) -> mx.array:
        """W_v' = R_h W_v per kv head (rows grouped by head)."""
        *lead, out, inn = w.shape
        hd = self.head_dim
        x = w.reshape(*lead, out // hd, hd, inn)
        x = mx.swapaxes(rht(mx.swapaxes(x, -1, -2), self.head_seed), -1, -2)
        return x.reshape(*lead, out, inn)

    def _o_cols(self, w: mx.array) -> mx.array:
        """W_o' = W_o blockdiag(R_hᵀ) (columns grouped by head)."""
        *lead, out, inn = w.shape
        hd = self.head_dim
        return rht(w.reshape(*lead, out, inn // hd, hd), self.head_seed).reshape(*lead, out, inn)

    def fold(self, role: str, w: mx.array, norm: mx.array | None = None) -> mx.array:
        """The folded float32 tensor of `role` (norm = the weight of the RMSNorm it reads, readers only)."""
        if not self.residual:
            w = w.astype(mx.float32)
            return self._v_rows(w) if role == "attn_v" else self._o_cols(w) if role == "attn_o" else w
        if role == "embed":
            return self.rotate(w.astype(mx.float32))
        if role in READER_NORMS:
            out = self.fold_reader(w, norm)
            return self._v_rows(out) if (self.v_heads and role == "attn_v") else out
        if role in WRITERS:
            out = self.fold_writer(w)
            return self._o_cols(out) if (self.v_heads and role == "attn_o") else out
        raise ValueError(f"role {role!r} is not touched by the residual rotation")

    def rotated_input(self, role: str) -> bool:
        """True when the folded tensor of `role` reads an input that is already rotated (no online V needed)."""
        return (self.residual and role in ROTATED_INPUT_ROLES) or (self.v_heads and role == "attn_o")

    def touches(self, role: str) -> bool:
        """True when the folding rewrites tensors of `role`."""
        return self.residual or (self.v_heads and role in ("attn_v", "attn_o"))


# --------------------------------------------------------------------------------------- name -> role / norm
_SWITCH = re.compile(r"^(.*)\.switch_mlp\.(\w+)\.weight$")


def _spec(adapter: ArchitectureAdapter, name: str) -> LinearSpec | None:
    """LinearSpec of a checkpoint name, or of an mlx-lm stacked expert name "<p>.switch_mlp.<proj>.weight"."""
    s = adapter._classify(name)
    if s is None and (m := _SWITCH.match(name)) is not None:
        s = adapter._classify(f"{m.group(1)}.experts.0.{m.group(2)}.weight")
    return s


def _norm_positions(adapter: ArchitectureAdapter, num_layers: int) -> dict[str, tuple[str, int | None]]:
    out = {adapter.norm_name("final"): ("final", None)}
    for i in range(num_layers):
        for pos in ("attn", "mlp"):
            out[adapter.norm_name(pos, i)] = (pos, i)
    return out


def _norm_of(adapter: ArchitectureAdapter, spec: LinearSpec) -> str | None:
    pos = READER_NORMS.get(spec.role)
    if pos is None:
        return None
    return adapter.norm_name(pos, spec.layer if pos != "final" else None)


class FoldedStore:
    """A SafetensorsStore view whose load() returns the FOLDED tensors (float32 for rewritten ones), one at a time.

    Readers fold their own norm weight (a small extra read); folded norm weights load as ones, except the final norm
    of a tied model (kept: RotatedFinalNorm applies it). info() / names() / shards are those of the checkpoint."""

    def __init__(self, store, adapter: ArchitectureAdapter, fold: ResidualFold):
        self.store, self.adapter, self.fold = store, adapter, fold
        self._norms = _norm_positions(adapter, adapter.num_layers)
        self.tied = adapter.tied_embeddings
        self.norms_folded = fold.residual

    def names(self) -> list[str]:
        return self.store.names()

    def info(self, name: str):
        return self.store.info(name)

    def __contains__(self, name: object) -> bool:
        return name in self.store

    def __len__(self) -> int:
        return len(self.store)

    def __getattr__(self, attr):   # shards, total_bytes, ... of the underlying store
        return getattr(self.store, attr)

    def load(self, name: str) -> mx.array:
        if name in self._norms:
            pos, _ = self._norms[name]
            w = self.store.load(name)
            keep = (pos == "final" and self.tied) or not self.norms_folded
            return w if keep else mx.ones(w.shape, dtype=w.dtype)
        spec = _spec(self.adapter, name)
        if spec is None or not self.fold.touches(spec.role):
            return self.store.load(name)          # attention-internal norms etc.: unchanged
        if spec.role == "lm_head" and self.tied:
            raise ValueError("a tied model has no separate LM head to fold")
        nn_ = _norm_of(self.adapter, spec)
        return self.fold.fold(spec.role, self.store.load(name), None if nn_ is None else self.store.load(nn_))


# --------------------------------------------------------------------------------------- in-memory model
class RotatedFinalNorm(nn.Module):
    """Final RMSNorm of a folded model with a tied LM head: R (g ⊙ n(Rᵀ h')) — two online rotations per token."""

    def __init__(self, norm: nn.Module, seed: int):
        super().__init__()
        self.norm = norm
        self.seed = int(seed)

    def __call__(self, h: mx.array) -> mx.array:
        z = self.norm(rht_inverse(h, self.seed))     # float32 through the norm, one rounding at the end
        return rht(z, self.seed).astype(h.dtype)


def fold_params(params: dict[str, mx.array], adapter: ArchitectureAdapter, fold: ResidualFold, *,
                dtype: mx.Dtype | None = None) -> dict[str, mx.array]:
    """Folded replacements for the entries of `params` (mlx-lm parameter names, e.g. one block's or the whole
    model's; stacked "switch_mlp" experts allowed). Every reader's norm weight must be among `params`. Returns only
    the changed entries, cast to their dtype (or `dtype`): folded linears / embedding and the folded norms as ones
    (the final norm of a tied model is kept). A tied LM head is skipped (it is the embedding)."""
    tied = adapter.tied_embeddings
    norms = _norm_positions(adapter, adapter.num_layers)
    new = {}
    for name, w in params.items():
        if name in norms:
            if fold.residual and not (norms[name][0] == "final" and tied):
                new[name] = mx.ones(w.shape, dtype=dtype or w.dtype)
            continue
        spec = _spec(adapter, name)
        if spec is None or (spec.role == "lm_head" and tied) or not fold.touches(spec.role):
            continue
        nn_ = _norm_of(adapter, spec) if fold.residual else None
        if nn_ is not None and nn_ not in params:
            raise KeyError(f"{name!r} reads norm {nn_!r}, which is not among the parameters to fold")
        new[name] = fold.fold(spec.role, w, None if nn_ is None else params[nn_]).astype(dtype or w.dtype)
        mx.eval(new[name])     # one tensor's float32 temporaries at a time
    return new


def _module(model: nn.Module, path: str):
    *pre, attr = path.split(".")
    parent = model
    for p in pre:
        parent = parent[int(p)] if isinstance(parent, list) else parent[p]
    return parent, attr


def fold_block(block: nn.Module, prefix: str, adapter: ArchitectureAdapter, fold: ResidualFold) -> None:
    """Fold one transformer block in place (its parameters named `prefix`.…, e.g. "model.layers.3"): the D8 path
    for block-streamed passes (suc.models.streaming.stream_blocks on_block hook)."""
    params = {f"{prefix}.{k}": v for k, v in tree_flatten(block.parameters())}
    new = fold_params(params, adapter, fold)
    block.update(tree_unflatten([(k[len(prefix) + 1:], v) for k, v in new.items()]))


def fold_model(model: nn.Module, adapter: ArchitectureAdapter, fold: ResidualFold, *,
               dtype: mx.Dtype | None = None) -> dict:
    """Rewrite an mlx-lm model (dense or MoE, unquantized) in place into its folded form. Weights are cast back to
    their stored dtype (or `dtype`). Tied LM head: the final norm is wrapped by RotatedFinalNorm. Returns
    {"folded": [names], "tied_head": bool}. Whole-model convenience; fold_block / FoldedStore stream (D8)."""
    if dtype is not None:
        model.set_dtype(dtype)
    new = fold_params(dict(tree_flatten(model.parameters())), adapter, fold)
    model.update(tree_unflatten(list(new.items())))
    mx.eval(model.parameters())
    tied = adapter.tied_embeddings and fold.residual
    if tied:
        parent, attr = _module(model, adapter.norm_name("final")[: -len(".weight")])
        parent[attr] = RotatedFinalNorm(parent[attr], fold.seed)
    return {"folded": sorted(new), "tied_head": tied}
