"""MLX implementation of `deepseek_v4` (deepseek-ai/DeepSeek-V4-Flash-0731), see docs/arch/deepseek_v4.md.

Ported from the checkpoint's reference implementation inference/model.py + kernel.py (rev 7872f01b); mlx-lm has no
`deepseek_v4` module. The DSpark / MTP speculative head (`mtp.*`) is dropped (text LM only).

Parameter namespace (= the SUC tensor namespace of suc.models.adapters.DeepseekV4Adapter.source_store):
  model.embed_tokens / lm_head / model.norm / model.hc_head.{fn,base,scale}
  model.layers.N.{attn_norm,ffn_norm}.weight, model.layers.N.hc_{attn,ffn}.{fn,base,scale}
  model.layers.N.self_attn.{wq_a,q_norm,wq_b,wkv,kv_norm,wo_a,wo_b}, .attn_sink, .compressor.{wkv,wgate,ape,norm},
      .indexer.{wq_b,weights_proj,compressor.*}   (reference names inside the attention)
  model.layers.N.mlp.gate.{weight, e_score_correction_bias | tid2eid}, mlp.shared_experts.{gate,up,down}_proj,
      mlp.switch_mlp.{gate,up,down}_proj (w1 / w3 / w2 of the 256 routed experts, stacked)
sanitize() maps the checkpoint names there and dequantizes FP8 (128 x 128 E8M0 blocks) / MXFP4 tensors exactly to
BF16, lazily (suc.io.quantized_source). Hyper-connection parameters, sinks, ape and the gate bias stay float32 (the
reference computes them in float32; set_dtype keeps them).

Stream layout. The residual is hc_mult = 4 streams of hidden_size (16384 wide, model.py:916). Hash-routed layers
(layers < num_hash_layers) select experts from the TOKEN ID (tid2eid), so the stream tensor also carries the token id
as 3 base-64 digit channels behind the streams (exact in bf16 / fp16 / fp32): [x_0 | x_1 | x_2 | x_3 | d0 d1 d2]
in every block, so blocks stay shape-preserving and the block-sequential machinery of suc.models.streaming (Fisher
VJPs, calibration, evaluation) runs unchanged with `block(h, mask)`. head_stream applies hc_head + norm + LM head.

Attention (MQA, one 512-wide K = V vector per token, sliding window 128, attention sinks) over the window keys plus
the layer's compressed entries: ratio 4 (CSA, overlapping gated pooling, lightning-indexer top-k) or ratio 128 (HCA,
every causal entry). Prefill and cached decode share ONE code path: the per-layer cache keeps the last window - 1
keys, the raw rows of the incomplete compressor group and of the previous group (overlap), and every compressed entry,
so a step computes exactly what the full prefill computes for those positions (same formulas, same order of
operations). The attention itself runs through mx.fast.scaled_dot_product_attention with a boolean mask over
[window keys | compressed entries] and `sinks`.

QAT simulation (docs/arch/deepseek_v4.md §8 item 7), ModelArgs flags:
  simulate_kv_fp8 (default on)       window KV and compressed KV: non-RoPE dims FP8 round trip, block 64, ue8m0 scale
  simulate_indexer_fp4 (default on)  indexer q / keys: Hadamard rotation + FP4 round trip, block 32
  simulate_act_fp8 (default OFF)     activations FP8 round trip (block 128) before every FP8 / FP4 GEMM of the
                                     reference (wq_a, wq_b, wkv, wo_b, indexer wq_b, shared + routed experts); off as in
                                     mlx-lm deepseek_v41 (the KL of switching it is measured by
                                     scripts/deepseek_v4_reference_check.py)
Round trips use a straight-through estimator (value = the quantized one, gradient = identity) so the Fisher VJPs
of suc compress pass through them; the indexer selection is under stop_gradient (a discrete top-k).

Known deviations from the reference (rounding level): the indexer score and the unweighted q RMS norm are computed in
float32 (the reference uses bf16 einsum / ops), routed expert outputs are summed in one float32 reduction, top-k ties
are broken by argpartition, the FP4 cast rounds to nearest-even (tilelang's T.Cast(FP4) rounding mode is unverified),
and logits are produced in the model dtype (the reference head is float32).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.switch_layers import SwitchGLU

F32 = mx.float32
_DIGIT = 64                      # token id carried as 3 base-64 digits (< 262144 ids)
_N_DIGITS = 3
_KEEP_F32 = (".hc_attn.", ".hc_ffn.", "model.hc_head.", ".attn_sink", ".ape", ".e_score_correction_bias")


# ------------------------------------------------------------------------------------------- configuration

@dataclass
class ModelArgs:
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_groups: int = 8
    o_lora_rank: int = 1024
    sliding_window: int = 128
    compress_ratios: Optional[list] = None
    compress_rope_theta: float = 160000.0
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2048
    n_shared_experts: int = 1
    num_hash_layers: int = 0
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    swiglu_limit: float = 10.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    eos_token_id: Any = 1
    simulate_kv_fp8: bool = True
    simulate_indexer_fp4: bool = True
    simulate_act_fp8: bool = False
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params: dict) -> "ModelArgs":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in names})

    def __post_init__(self):
        n = self.num_hidden_layers
        self.compress_ratios = list(self.compress_ratios or [0] * n)[:n]   # the config also lists the MTP stages
        if len(self.compress_ratios) != n:
            raise ValueError(f"compress_ratios has {len(self.compress_ratios)} entries for {n} layers")
        if self.tie_word_embeddings:
            raise NotImplementedError("deepseek_v4 with tied embeddings")
        if self.n_shared_experts != 1:
            raise NotImplementedError("deepseek_v4 with n_shared_experts != 1")
        if self.scoring_func not in ("sqrtsoftplus", "sigmoid", "softmax"):
            raise ValueError(f"unknown scoring_func {self.scoring_func!r}")
        if self.num_hash_layers and self.vocab_size > _DIGIT ** _N_DIGITS:
            raise NotImplementedError(f"hash routing carries token ids < {_DIGIT ** _N_DIGITS} in the stream")

    # attributes read by generic SUC code (autotune shapes, preflight)
    @property
    def intermediate_size(self) -> int:
        return self.moe_intermediate_size

    @property
    def stream_width(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def token_channels(self) -> int:
        return _N_DIGITS if self.num_hash_layers else 0


# ------------------------------------------------------------------------------------------- numerics helpers

def _pow2_ceil_scale(amax: mx.array, inv_max: float) -> mx.array:
    """2^ceil(log2(amax * inv_max)) by IEEE-754 bit manipulation (kernel.py fast_round_scale), float32."""
    v = (amax.astype(F32) * mx.array(inv_max, dtype=F32)).view(mx.uint32)
    e = ((v >> 23) & 0xFF).astype(mx.int32) - 127 + ((v & 0x7FFFFF) != 0).astype(mx.int32)
    e = mx.clip(e, -126, 127)
    return ((e + 127).astype(mx.uint32) << 23).view(F32)


def _ste(x: mx.array, q: mx.array) -> mx.array:
    """Value q (exactly), gradient of x (straight-through)."""
    xf = x.astype(F32)
    return (xf + mx.stop_gradient(q.astype(F32) - xf)).astype(x.dtype)


def fp8_round_trip(x: mx.array, block: int) -> mx.array:
    """act_quant(x, block, "ue8m0", inplace=True) of kernel.py: per `block` values of the last axis a power-of-two
    scale (amax / 448 rounded up, amax >= 1e-4), E4M3 round to nearest even, back to x.dtype. Straight-through."""
    shape = x.shape
    xf = x.astype(F32).reshape(*shape[:-1], shape[-1] // block, block)
    s = _pow2_ceil_scale(mx.maximum(mx.abs(xf).max(-1, keepdims=True), 1e-4), 1.0 / 448.0)
    q = mx.from_fp8(mx.to_fp8(mx.clip(xf / s, -448.0, 448.0)), dtype=F32) * s
    return _ste(x, q.reshape(shape).astype(x.dtype))


_FP4_MAGS = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=F32)


def fp4_round_trip(x: mx.array, block: int = 32) -> mx.array:
    """fp4_act_quant(x, block, inplace=True) of kernel.py: power-of-two scale per block (amax / 6 rounded up), E2M1
    round to nearest (ties to the even code), back to x.dtype. Straight-through."""
    shape = x.shape
    xf = x.astype(F32).reshape(*shape[:-1], shape[-1] // block, block)
    s = _pow2_ceil_scale(mx.maximum(mx.abs(xf).max(-1, keepdims=True), 6.0 * 2.0 ** -126), 1.0 / 6.0)
    y = mx.clip(xf / s, -6.0, 6.0)
    a = mx.abs(y)
    code = ((a > 0.25).astype(mx.uint32) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5) + (a >= 3.5)
            + (a > 5.0))
    q = mx.take(_FP4_MAGS, code) * mx.where(y < 0, -1.0, 1.0) * s
    return _ste(x, q.reshape(shape).astype(x.dtype))


def hadamard(x: mx.array) -> mx.array:
    """rotate_activation of model.py: orthonormal Walsh-Hadamard transform of the last axis."""
    return mx.hadamard_transform(x, scale=x.shape[-1] ** -0.5)


def rope_inv_freq(dim: int, base: float, scaling: dict | None) -> np.ndarray:
    """Per-pair frequencies of precompute_freqs_cis (model.py:206), float32; YaRN when scaling is given."""
    freqs = (np.float32(1.0) / (np.float32(base) ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim))))
    freqs = freqs.astype(np.float32)
    if scaling:
        factor = float(scaling.get("factor", 1.0))
        orig = int(scaling.get("original_max_position_embeddings", 0))
        beta_fast, beta_slow = float(scaling.get("beta_fast", 32)), float(scaling.get("beta_slow", 1))
        if orig > 0:
            def corr_dim(rot):
                return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
            low = max(math.floor(corr_dim(beta_fast)), 0)
            high = min(math.ceil(corr_dim(beta_slow)), dim - 1)
            if low == high:
                high += 0.001
            ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - np.float32(low)) / np.float32(high - low), 0, 1)
            smooth = (np.float32(1) - ramp).astype(np.float32)
            freqs = (freqs / np.float32(factor) * (np.float32(1) - smooth) + freqs * smooth).astype(np.float32)
    return freqs


class Rope:
    """Interleaved-pair RoPE on the LAST `dim` channels (apply_rotary_emb, model.py:238), float32 math."""

    def __init__(self, dim: int, base: float, scaling: dict | None):
        self.dim = dim
        self.inv_freq = mx.array(rope_inv_freq(dim, base, scaling))

    def cos_sin(self, positions: mx.array) -> tuple[mx.array, mx.array]:
        ang = positions.astype(F32)[:, None] * self.inv_freq[None, :]
        return mx.cos(ang), mx.sin(ang)

    def apply(self, x: mx.array, cs: tuple[mx.array, mx.array], *, head_axis: bool, inverse: bool = False):
        """x (..., L, [H,] D): rotate the last self.dim channels at the positions of cs (L entries)."""
        cos, sin = cs
        if head_axis:
            cos, sin = cos[:, None, :], sin[:, None, :]
        if inverse:
            sin = -sin
        d = self.dim
        keep, r = x[..., :-d], x[..., -d:].astype(F32)
        r = r.reshape(*r.shape[:-1], d // 2, 2)
        a, b = r[..., 0], r[..., 1]
        out = mx.stack([a * cos - b * sin, a * sin + b * cos], axis=-1).reshape(*r.shape[:-2], d)
        return mx.concatenate([keep, out.astype(x.dtype)], axis=-1) if keep.shape[-1] else out.astype(x.dtype)


# ------------------------------------------------------------------------------------------- modules

class RMSNorm(nn.Module):
    """model.py RMSNorm: float32 math, weight applied in float32, rounded once to the input dtype."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class _CompressorState:
    """Rows of the incomplete group (raw kv / score), the previous complete group (overlap), emitted entries."""

    __slots__ = ("cur_kv", "cur_sc", "prev_kv", "prev_sc", "entries", "n")

    def __init__(self):
        self.cur_kv = self.cur_sc = self.prev_kv = self.prev_sc = self.entries = None
        self.n = 0


class Compressor(nn.Module):
    """Gated pooling of `ratio` consecutive tokens into one KV entry (model.py:285); ratio 4 overlaps with the
    previous group (entry i pools the first-half channels of group i-1 and the second-half channels of group i)."""

    def __init__(self, a: ModelArgs, ratio: int, head_dim: int, rotate: bool, rope: Rope):
        super().__init__()
        self.ratio, self.d, self.rotate = ratio, head_dim, rotate
        self.overlap = ratio == 4
        coff = 2 if self.overlap else 1
        self.ape = mx.zeros((ratio, coff * head_dim), dtype=F32)
        self.wkv = nn.Linear(a.hidden_size, coff * head_dim, bias=False)
        self.wgate = nn.Linear(a.hidden_size, coff * head_dim, bias=False)
        self.norm = RMSNorm(head_dim, a.rms_norm_eps)
        self._rope = rope
        self._rd = a.qk_rope_head_dim
        self._sim_kv, self._sim_fp4 = a.simulate_kv_fp8, a.simulate_indexer_fp4

    def __call__(self, x: mx.array, st: _CompressorState) -> mx.array | None:
        """New entries (B, n, head_dim) from the rows of x (B, L, dim) appended to the state, or None; updates st."""
        B = x.shape[0]
        r, d = self.ratio, self.d
        xf = x.astype(F32)
        kv, sc = self.wkv(xf).astype(F32), self.wgate(xf).astype(F32)       # compression runs in float32
        if st.cur_kv is not None:
            kv, sc = mx.concatenate([st.cur_kv, kv], axis=1), mx.concatenate([st.cur_sc, sc], axis=1)
        n = kv.shape[1] // r
        st.cur_kv, st.cur_sc = kv[:, n * r:], sc[:, n * r:]
        if n == 0:
            return None
        cd = kv.shape[-1]
        gk = kv[:, : n * r].reshape(B, n, r, cd)
        gs = sc[:, : n * r].reshape(B, n, r, cd) + self.ape
        if self.overlap:
            pk = st.prev_kv[:, None] if st.prev_kv is not None else mx.zeros((B, 1, r, cd), F32)
            ps = st.prev_sc[:, None] if st.prev_sc is not None else mx.full((B, 1, r, cd), -mx.inf, F32)
            pk = mx.concatenate([pk, gk[:, :-1]], axis=1)
            ps = mx.concatenate([ps, gs[:, :-1]], axis=1)
            st.prev_kv, st.prev_sc = gk[:, -1], gs[:, -1]
            gk = mx.concatenate([pk[..., :d], gk[..., d:]], axis=2)
            gs = mx.concatenate([ps[..., :d], gs[..., d:]], axis=2)
        pooled = (gk * mx.softmax(gs, axis=2)).sum(axis=2)
        e = self.norm(pooled.astype(x.dtype))
        pos = (st.n + mx.arange(n)) * r
        e = self._rope.apply(e, self._rope.cos_sin(pos), head_axis=False)
        if self.rotate:
            e = hadamard(e)
            if self._sim_fp4:
                e = fp4_round_trip(e, 32)
        elif self._sim_kv:
            e = mx.concatenate([fp8_round_trip(e[..., :-self._rd], 64), e[..., -self._rd:]], axis=-1)
        st.n += n
        st.entries = e if st.entries is None else mx.concatenate([st.entries, e], axis=1)
        return e


class Indexer(nn.Module):
    """Lightning indexer (model.py:386): scores every visible compressed entry of its OWN ratio-4 compressor and
    selects the top `index_topk` per query; returns a boolean selection (B, L, N) or None (every visible entry)."""

    def __init__(self, a: ModelArgs, rope: Rope):
        super().__init__()
        self.n_heads, self.head_dim, self.topk = a.index_n_heads, a.index_head_dim, a.index_topk
        self.wq_b = nn.Linear(a.q_lora_rank, a.index_n_heads * a.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(a.hidden_size, a.index_n_heads, bias=False)
        self.compressor = Compressor(a, 4, a.index_head_dim, True, rope)
        self._rope = rope
        self._sim_fp4, self._sim_act = a.simulate_indexer_fp4, a.simulate_act_fp8

    def __call__(self, x: mx.array, qr: mx.array, cs, pos: mx.array, st: _CompressorState) -> mx.array | None:
        B, L, _ = x.shape
        self.compressor(mx.stop_gradient(x), st)
        n = st.n
        if n == 0 or n <= self.topk:
            return None                       # at most topk entries exist: every visible one is selected
        x, qr = mx.stop_gradient(x), mx.stop_gradient(qr)
        q = self.wq_b(fp8_round_trip(qr, 128) if self._sim_act else qr).reshape(B, L, self.n_heads, self.head_dim)
        q = hadamard(self._rope.apply(q, cs, head_axis=True))
        if self._sim_fp4:
            q = fp4_round_trip(q, 32)
        w = self.weights_proj(x).astype(F32) * (self.head_dim ** -0.5 * self.n_heads ** -0.5)
        k = mx.stop_gradient(st.entries)                                   # (B, N, d)
        s = mx.maximum(mx.matmul(q, k[:, None].swapaxes(-1, -2)).astype(F32), 0.0)   # (B, L, H, N)
        s = (s * w[..., None]).sum(axis=2)                                 # (B, L, N)
        visible = (mx.arange(n)[None, :] + 1) * 4 <= pos[:, None] + 1      # (L, N)
        s = mx.where(visible[None], s, -mx.inf)
        top = mx.argpartition(-s, kth=self.topk - 1, axis=-1)[..., : self.topk]
        sel = mx.zeros((B, L, n), dtype=mx.bool_)
        sel = mx.put_along_axis(sel, top, mx.array(True), axis=-1)
        return mx.stop_gradient(sel)


class LayerCache:
    """Per-layer decode state: tokens seen, the last window - 1 keys, compressor / indexer states."""

    def __init__(self):
        self.offset = 0
        self.win: mx.array | None = None
        self.comp = _CompressorState()
        self.index = _CompressorState()

    @property
    def state(self):
        return [a for a in (self.win, self.comp.entries, self.index.entries) if a is not None]

    def is_trimmable(self) -> bool:
        return False


class Attention(nn.Module):
    """model.py Attention: low-rank q (wq_a -> q_norm -> wq_b, unweighted per-head RMS norm), single K = V head,
    sliding window + compressed entries, attention sinks, inverse RoPE on the output, grouped low-rank output."""

    def __init__(self, a: ModelArgs, layer: int):
        super().__init__()
        self.n_heads, self.head_dim, self.rd = a.num_attention_heads, a.head_dim, a.qk_rope_head_dim
        self.window, self.ratio, self.eps = a.sliding_window, a.compress_ratios[layer], a.rms_norm_eps
        self.groups, self.o_rank = a.o_groups, a.o_lora_rank
        self.scale = a.head_dim ** -0.5
        self._sim_kv, self._sim_act = a.simulate_kv_fp8, a.simulate_act_fp8
        if self.ratio:
            self.rope = Rope(a.qk_rope_head_dim, a.compress_rope_theta, a.rope_scaling)
        else:                                 # pure sliding window: base theta, no YaRN (model.py:483)
            self.rope = Rope(a.qk_rope_head_dim, a.rope_theta, None)
        self.attn_sink = mx.zeros((a.num_attention_heads,), dtype=F32)
        self.wq_a = nn.Linear(a.hidden_size, a.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(a.q_lora_rank, a.rms_norm_eps)
        self.wq_b = nn.Linear(a.q_lora_rank, a.num_attention_heads * a.head_dim, bias=False)
        self.wkv = nn.Linear(a.hidden_size, a.head_dim, bias=False)
        self.kv_norm = RMSNorm(a.head_dim, a.rms_norm_eps)
        self.wo_a = nn.Linear(a.num_attention_heads * a.head_dim // a.o_groups, a.o_groups * a.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(a.o_groups * a.o_lora_rank, a.hidden_size, bias=False)
        if self.ratio:
            self.compressor = Compressor(a, self.ratio, a.head_dim, False, self.rope)
            if self.ratio == 4:
                self.indexer = Indexer(a, self.rope)

    def _aq(self, x: mx.array) -> mx.array:
        return fp8_round_trip(x, 128) if self._sim_act else x

    def _wo_a(self, o: mx.array) -> mx.array:
        """o (B, L, G, in_g) -> (B, L, G, o_rank): row group g of wo_a applied to input group g."""
        G, R = self.groups, self.o_rank
        lin = self.wo_a
        if type(lin) is nn.Linear or (isinstance(lin, nn.Linear) and hasattr(lin, "_suc_hook")):
            hook = getattr(lin, "_suc_hook", None)        # activation recorder (suc.analysis.activations)
            if hook is not None:
                hook(o.reshape(-1, o.shape[-1]))
            w = lin.weight.reshape(G, R, -1)
            return mx.einsum("blgd,grd->blgr", o, w)
        y = lin(o).reshape(*o.shape[:3], G, R)           # SUC module: every row group on every input group
        idx = mx.arange(G).reshape(1, 1, G, 1, 1)
        return mx.take_along_axis(y, idx, axis=3).squeeze(3)

    def __call__(self, x: mx.array, cache: LayerCache | None = None) -> mx.array:
        B, L, _ = x.shape
        c = cache if cache is not None else LayerCache()
        p0 = c.offset
        pos = p0 + mx.arange(L)
        cs = self.rope.cos_sin(pos)
        xa = self._aq(x)
        qr = self.q_norm(self.wq_a(xa))
        q = self.wq_b(self._aq(qr)).reshape(B, L, self.n_heads, self.head_dim)
        qf = q.astype(F32)
        q = (qf * mx.rsqrt(mx.mean(qf * qf, axis=-1, keepdims=True) + self.eps)).astype(x.dtype)
        q = self.rope.apply(q, cs, head_axis=True)
        kv = self.rope.apply(self.kv_norm(self.wkv(xa)), cs, head_axis=False)
        if self._sim_kv:
            kv = mx.concatenate([fp8_round_trip(kv[..., :-self.rd], 64), kv[..., -self.rd:]], axis=-1)

        keys = kv if c.win is None else mx.concatenate([c.win, kv], axis=1)
        kpos = mx.arange(p0 + L - keys.shape[1], p0 + L)
        mask = (kpos[None, :] <= pos[:, None]) & (kpos[None, :] > pos[:, None] - self.window)   # (L, Kw)
        mask = mx.broadcast_to(mask[None], (B, L, keys.shape[1]))
        if self.ratio:
            sel = self.indexer(x, qr, cs, pos, c.index) if "indexer" in self else None
            self.compressor(x, c.comp)
            n = c.comp.n
            if n:
                visible = (mx.arange(n)[None, :] + 1) * self.ratio <= pos[:, None] + 1
                vis = visible[None] & sel if sel is not None else mx.broadcast_to(visible[None], (B, L, n))
                keys = mx.concatenate([keys, c.comp.entries.astype(keys.dtype)], axis=1)
                mask = mx.concatenate([mask, vis], axis=-1)
        k = keys[:, None]
        o = mx.fast.scaled_dot_product_attention(q.transpose(0, 2, 1, 3), k, k, scale=self.scale,
                                                 mask=mask[:, None], sinks=self.attn_sink.astype(q.dtype))
        o = self.rope.apply(o.transpose(0, 2, 1, 3), cs, head_axis=True, inverse=True)
        o = self._wo_a(o.reshape(B, L, self.groups, -1))
        out = self.wo_b(self._aq(o.reshape(B, L, self.groups * self.o_rank)))
        if cache is not None:
            keep = self.window - 1
            c.win = kv[:, -keep:] if c.win is None else mx.concatenate([c.win, kv], axis=1)[:, -keep:]
            c.offset = p0 + L
        return out


class Gate(nn.Module):
    """model.py Gate: float32 scores (sqrt(softplus) by default); hash layers take the experts from tid2eid[token],
    the others the top-k of scores + e_score_correction_bias (selection only). Weights = the unbiased scores of the
    chosen experts, normalised (non-softmax scoring), x routed_scaling_factor."""

    def __init__(self, a: ModelArgs, layer: int):
        super().__init__()
        self.top_k, self.func, self.route_scale = a.num_experts_per_tok, a.scoring_func, a.routed_scaling_factor
        self.hash = layer < a.num_hash_layers
        self.weight = mx.zeros((a.n_routed_experts, a.hidden_size))
        if self.hash:
            self.tid2eid = mx.zeros((a.vocab_size, a.num_experts_per_tok), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros((a.n_routed_experts,), dtype=F32)

    def __call__(self, x: mx.array, tokens: mx.array | None) -> tuple[mx.array, mx.array]:
        s = x.astype(F32) @ self.weight.astype(F32).T
        if self.func == "softmax":
            s = mx.softmax(s, axis=-1)
        elif self.func == "sigmoid":
            s = mx.sigmoid(s)
        else:
            s = mx.sqrt(mx.logaddexp(s, 0.0))
        if self.hash:
            if tokens is None:
                raise ValueError("hash-routed layer reached without token ids in the stream (use embed_stream)")
            inds = self.tid2eid[tokens].astype(mx.uint32)
        else:
            biased = mx.stop_gradient(s + self.e_score_correction_bias)
            inds = mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[..., : self.top_k].astype(mx.uint32)
        inds = mx.stop_gradient(inds)
        w = mx.take_along_axis(s, inds, axis=-1)
        if self.func != "softmax":
            w = w / w.sum(axis=-1, keepdims=True)
        return inds, w * self.route_scale


def _clamped_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    g, u = gate.astype(F32), up.astype(F32)
    if limit > 0:
        u = mx.clip(u, -limit, limit)
        g = mx.minimum(g, limit)
    return nn.silu(g) * u


class RoutedExperts(SwitchGLU):
    """SwitchGLU (gate_proj = w1, up_proj = w3, down_proj = w2) with the reference expert math: float32 clamped
    SwiGLU, the routing weight applied BEFORE w2 (model.py:601-611), outputs summed in float32. Rows are sorted by
    expert for large batches (as mlx-lm's SwitchGLU); the activation recorder sees the usual (x, indices) calls."""

    def __init__(self, dims: int, hidden: int, n: int, limit: float):
        super().__init__(dims, hidden, n, bias=False)
        self.limit = limit
        self._stream = None       # suc.runtime.expert_streaming.ExpertStream (routed experts read from SSD)

    def __call__(self, x: mx.array, inds: mx.array, w: mx.array) -> mx.array:
        if self._stream is not None:
            return self._stream.run(self, x, inds, w)
        return self.compute(self.gate_proj, self.up_proj, self.down_proj, x, inds, w)

    def compute(self, gate_p, up_p, down_p, x: mx.array, inds: mx.array, w: mx.array) -> mx.array:
        """The expert math with the given projection modules (the resident stacks, or an ExpertStream's modules over
        the selected experts with remapped indices)."""
        k = inds.shape[-1]
        lead = inds.shape
        xe = mx.expand_dims(x, (-2, -3))                  # (..., 1, 1, D)
        do_sort = inds.size >= 64
        if do_sort:
            flat = inds.flatten()
            order = mx.argsort(flat)
            inv = mx.argsort(order)
            xe = xe.flatten(0, -3)[order // k]            # (N*k, 1, D)
            idx, wr = flat[order], w.flatten()[order]
        else:
            idx, wr = inds, w
        up = up_p(xe, idx, sorted_indices=do_sort)
        gate = gate_p(xe, idx, sorted_indices=do_sort)
        h = (_clamped_swiglu(gate, up, self.limit) * wr[..., None, None]).astype(x.dtype)
        y = down_p(h, idx, sorted_indices=do_sort)
        if do_sort:
            y = y[inv].reshape(*lead, 1, -1)
        return y.squeeze(-2).astype(F32).sum(axis=-2)


class SharedExpert(nn.Module):
    def __init__(self, dims: int, hidden: int, limit: float, sim_act: bool):
        super().__init__()
        self.gate_proj = nn.Linear(dims, hidden, bias=False)
        self.up_proj = nn.Linear(dims, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dims, bias=False)
        self.limit, self._sim_act = limit, sim_act

    def __call__(self, x: mx.array) -> mx.array:
        h = _clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.limit).astype(x.dtype)
        return self.down_proj(fp8_round_trip(h, 128) if self._sim_act else h)


class MoE(nn.Module):
    def __init__(self, a: ModelArgs, layer: int):
        super().__init__()
        self.gate = Gate(a, layer)
        self.switch_mlp = RoutedExperts(a.hidden_size, a.moe_intermediate_size, a.n_routed_experts, a.swiglu_limit)
        self.shared_experts = SharedExpert(a.hidden_size, a.moe_intermediate_size, a.swiglu_limit, a.simulate_act_fp8)
        self._sim_act = a.simulate_act_fp8

    def __call__(self, x: mx.array, tokens: mx.array | None) -> mx.array:
        inds, w = self.gate(x, tokens)
        xa = fp8_round_trip(x, 128) if self._sim_act else x
        y = self.switch_mlp(xa, inds, w)
        return (y + self.shared_experts(xa).astype(F32)).astype(x.dtype)


class HyperConnection(nn.Module):
    """Manifold hyper-connection mixer of one sub-layer (Block.hc_pre / hc_post, kernel.py hc_split_sinkhorn)."""

    def __init__(self, a: ModelArgs):
        super().__init__()
        hc = a.hc_mult
        mix = (2 + hc) * hc
        self.fn = mx.zeros((mix, hc * a.hidden_size), dtype=F32)
        self.base = mx.zeros((mix,), dtype=F32)
        self.scale = mx.zeros((3,), dtype=F32)
        self.hc, self.iters, self.eps, self.norm_eps = hc, a.hc_sinkhorn_iters, a.hc_eps, a.rms_norm_eps

    def pre(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """x (B, L, hc, D) -> (y (B, L, D) in x.dtype, post (B, L, hc), comb (B, L, hc, hc)) (float32)."""
        B, L, hc, D = x.shape
        xf = x.reshape(B, L, hc * D).astype(F32)
        rs = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.norm_eps)
        mixes = (xf @ self.fn.T) * rs
        sc, b = self.scale, self.base
        pre = mx.sigmoid(mixes[..., :hc] * sc[0] + b[:hc]) + self.eps
        post = 2 * mx.sigmoid(mixes[..., hc:2 * hc] * sc[1] + b[hc:2 * hc])
        comb = (mixes[..., 2 * hc:] * sc[2] + b[2 * hc:]).reshape(B, L, hc, hc)
        comb = mx.softmax(comb, axis=-1) + self.eps
        comb = comb / (comb.sum(axis=-2, keepdims=True) + self.eps)
        for _ in range(self.iters - 1):
            comb = comb / (comb.sum(axis=-1, keepdims=True) + self.eps)
            comb = comb / (comb.sum(axis=-2, keepdims=True) + self.eps)
        y = (pre[..., None] * xf.reshape(B, L, hc, D)).sum(axis=2)
        return y.astype(x.dtype), post, comb

    @staticmethod
    def post(f: mx.array, res: mx.array, post: mx.array, comb: mx.array) -> mx.array:
        """y_j = post_j f + Σ_i comb_ij res_i (float32), rounded to f.dtype; res (B, L, hc, D)."""
        y = post[..., None] * f.astype(F32)[..., None, :] + mx.swapaxes(comb, -1, -2) @ res.astype(F32)
        return y.astype(f.dtype)


class HCHead(nn.Module):
    """Final collapse of the streams (Block.hc_head): pre = sigmoid(mixes * scale + base) + eps."""

    def __init__(self, a: ModelArgs):
        super().__init__()
        self.fn = mx.zeros((a.hc_mult, a.hc_mult * a.hidden_size), dtype=F32)
        self.base = mx.zeros((a.hc_mult,), dtype=F32)
        self.scale = mx.zeros((1,), dtype=F32)
        self.eps, self.norm_eps = a.hc_eps, a.rms_norm_eps

    def __call__(self, x: mx.array) -> mx.array:
        B, L, hc, D = x.shape
        xf = x.reshape(B, L, hc * D).astype(F32)
        rs = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.norm_eps)
        pre = mx.sigmoid((xf @ self.fn.T) * rs * self.scale + self.base) + self.eps
        return (pre[..., None] * xf.reshape(B, L, hc, D)).sum(axis=2).astype(x.dtype)


def tokens_from_stream(extra: mx.array) -> mx.array:
    """Token ids (B, L) int32 from the 3 base-64 digit channels of the stream tensor."""
    d = mx.round(mx.stop_gradient(extra[..., :_N_DIGITS]).astype(F32)).astype(mx.int32)
    return d[..., 0] + _DIGIT * d[..., 1] + _DIGIT * _DIGIT * d[..., 2]


class DecoderLayer(nn.Module):
    """One block on the stream tensor [hc streams | token-id digits] (shape-preserving)."""

    def __init__(self, a: ModelArgs, layer: int):
        super().__init__()
        self.hc, self.dim, self.width = a.hc_mult, a.hidden_size, a.stream_width
        self.hash = layer < a.num_hash_layers
        self.self_attn = Attention(a, layer)
        self.mlp = MoE(a, layer)
        self.attn_norm = RMSNorm(a.hidden_size, a.rms_norm_eps)
        self.ffn_norm = RMSNorm(a.hidden_size, a.rms_norm_eps)
        self.hc_attn = HyperConnection(a)
        self.hc_ffn = HyperConnection(a)

    def __call__(self, h: mx.array, mask=None, cache=None) -> mx.array:
        """`mask` is accepted for the generic block API and ignored (window / compressed masks are built here)."""
        B, L, _ = h.shape
        x, extra = h[..., : self.width].reshape(B, L, self.hc, self.dim), h[..., self.width:]
        y, post, comb = self.hc_attn.pre(x)
        x = self.hc_attn.post(self.self_attn(self.attn_norm(y), cache), x, post, comb)
        y, post, comb = self.hc_ffn.pre(x)
        tokens = tokens_from_stream(extra) if self.hash else None
        x = self.hc_ffn.post(self.mlp(self.ffn_norm(y), tokens), x, post, comb)
        x = x.reshape(B, L, self.width)
        return mx.concatenate([x, extra.astype(x.dtype)], axis=-1) if extra.shape[-1] else x


class TextModel(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(a.vocab_size, a.hidden_size)
        self.layers = [DecoderLayer(a, i) for i in range(a.num_hidden_layers)]
        self.norm = RMSNorm(a.hidden_size, a.rms_norm_eps)
        self.hc_head = HCHead(a)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = TextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.layers

    # -- stream API (suc.models.streaming) --------------------------------------
    def embed_stream(self, tokens: mx.array, cache=None) -> mx.array:
        """Token ids (B, L) -> stream tensor (B, L, hc * D [+ 3 token-id digits])."""
        a = self.args
        e = self.model.embed_tokens(tokens)
        parts = [e] * a.hc_mult
        if a.token_channels:
            t = tokens.astype(mx.int32)
            digits = mx.stack([t % _DIGIT, (t // _DIGIT) % _DIGIT, t // (_DIGIT * _DIGIT)], axis=-1)
            parts.append(digits.astype(e.dtype))
        return mx.concatenate(parts, axis=-1)

    def head_stream(self, h: mx.array) -> mx.array:
        a = self.args
        B, L, _ = h.shape
        x = self.model.hc_head(h[..., : a.stream_width].reshape(B, L, a.hc_mult, a.hidden_size))
        return self.lm_head(self.model.norm(x))

    def __call__(self, inputs: mx.array, cache=None, **_) -> mx.array:
        h = self.embed_stream(inputs)
        for i, layer in enumerate(self.model.layers):
            h = layer(h, None, cache[i] if cache is not None else None)
        return self.head_stream(h)

    def step_hidden(self, token: int, cache) -> tuple[mx.array, list[mx.array]]:
        """(logits (vocab,), [stream after the embedding, after every block]) for one token (decode)."""
        h = self.embed_stream(mx.array([[int(token)]], dtype=mx.uint32))
        hs = [h]
        for layer, c in zip(self.model.layers, cache):
            h = layer(h, None, c)
            hs.append(h)
        return self.head_stream(h)[0, -1], [x[0, -1] for x in hs]

    def make_cache(self) -> list:
        return [LayerCache() for _ in self.model.layers]

    def set_dtype(self, dtype: mx.Dtype, predicate=lambda x: mx.issubdtype(x, mx.floating)) -> None:
        """As nn.Module.set_dtype, but hyper-connection parameters, sinks, ape and the gate bias stay float32."""
        keep = [(k, v) for k, v in tree_flatten(self.parameters()) if any(t in k for t in _KEEP_F32)]
        super().set_dtype(dtype, predicate)
        self.update(tree_unflatten([(k, v.astype(F32)) for k, v in keep]))

    # -- loading ----------------------------------------------------------------
    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Checkpoint (reference) names -> parameter names: drop mtp.*, consume *.scale companions by dequantizing their
        FP8 / MXFP4 weight exactly to BF16, stack the routed experts into switch_mlp, tid2eid -> int32. Lazy."""
        from suc.io.quantized_source import dequantize
        from suc.models.adapters import DeepseekV4Adapter
        from suc.quant.stacks import stack_of

        out: dict[str, mx.array] = {}
        experts: dict[str, dict[int, mx.array]] = {}
        for k, v in weights.items():
            name = DeepseekV4Adapter.to_suc_name(k)
            if name is None:
                continue
            sk = k[: -len(".weight")] + ".scale" if k.endswith(".weight") else None
            if sk is not None and sk in weights:
                kind = "mxfp4" if v.dtype == mx.int8 else "fp8_block"
                v = dequantize(kind, v, weights[sk], name=name, check=False)
            if name.endswith(".tid2eid"):
                v = v.astype(mx.int32)
            st = stack_of(name)
            if st is not None:
                experts.setdefault(st[0], {})[st[1]] = v
            else:
                out[name] = v
        for stack, rows in experts.items():
            if sorted(rows) != list(range(len(rows))):
                raise ValueError(f"{stack}: experts {sorted(rows)[:4]}... are not 0..{len(rows) - 1}")
            out[stack] = mx.stack([rows[e] for e in range(len(rows))])
        return out


__all__ = ["ModelArgs", "Model", "LayerCache", "fp8_round_trip", "fp4_round_trip", "rope_inv_freq",
           "tokens_from_stream"]
