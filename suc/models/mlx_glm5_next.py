"""MLX implementation of the `glm5_next` text model (zai-org/GLM-5.3-Flash), see docs/arch/glm5_next.md.

Ported from transformers 5.18 `models/glm5_next/modeling_glm5_next.py` (TF below); mlx-lm has no `glm5_next` module.
Reused: the vectorized gated-delta recurrence of mlx-lm (`gated_delta.py`, through suc.models.mlx_qwen4_exp._gated_delta
with its ops VJP), SwitchGLU, ArraysCache. Vision tower and the MTP layer (index num_hidden_layers) are dropped.

Parameter namespace (= the SUC tensor namespace of suc.models.adapters.Glm5NextAdapter.source_store): the checkpoint
names with `model.language_model.` -> `model.`, unchanged otherwise (q/k/v short convolutions stay three [C, 1, K]
tensors, hyper-connection parameters stay `layers.N.hc_{attn,ffn}_{fn,base,scale}`), plus the routed experts stacked
into `mlp.switch_mlp.{gate,up,down}_proj.weight`. sanitize() maps the checkpoint names there and dequantizes the FP8
(E4M3 + F32 128 x 128 block multiplier) weights to BF16 lazily (suc.io.quantized_source "fp8_block_f32").
Hyper-connection parameters, A_log, dt_bias, the short-convolution weights and the router bias stay float32 (HF keeps
them in float32: `_keep_in_fp32_modules_strict`, TF:1379; set_dtype keeps them).

Stream layout. The residual is hc_mult = 4 streams of hidden_size (TF:1495): blocks map the (B, L, 4 * D) stream tensor
to itself, so the block-sequential machinery of suc.models.streaming (Fisher VJPs, calibration, evaluation) runs
unchanged with `block(h, mask)`; head_stream applies the unweighted stream mean (TF:337), the final norm and the head.

Layers (TF:1280-1350): mHC collapse (pre) -> input_layernorm -> KDA (34 layers) or MLA + DSA indexer (11 layers) ->
mHC expand (post / comb, computed in the model dtype as TF:1337) -> mHC collapse -> post_attention_layernorm -> dense
MLP (layers < first_k_dense_replace) or MoE (288 experts top-8, sigmoid router with correction bias, shared expert;
SwiGLU with the swiglu_limit clamps) -> mHC expand.

KDA (TF:622-771): q, k, v = SiLU(causal depthwise conv4(W x)) (conv and SiLU in float32), q / k L2-normalized in float32
(x / sqrt(sum x^2 + 1e-6)), q / sqrt(Dk); per-channel log-decay g = -5 * sigmoid(exp(A_log) * (f_b f_a x + dt_bias))
(the "safe gate", TF:355-365), beta = sigmoid(b_proj x); recurrence S <- S diag(exp g); S <- S + k beta (v - S^T k)^T;
o = S^T q (float32, the vectorized gated-delta kernel; HF runs the chunked form for prefill, equal up to float32
rounding). Output o_proj(RMSNorm_w(o) * sigmoid(g_b g_a x)), gated norm in float32.

MLA (TF:1102-1277, NoPE: qk_rope_head_dim = 0): q = q_b(RMSNorm(q_a x)), latent c = RMSNorm(kv_a x) is cached; kv_b is
applied ABSORBED per head (q'_h = W_k,h^T q_h scores against c directly, o_h = W_v,h (P c)): mathematically the HF
expand_kv attention, without materializing per-head keys / values for every cached token. Scale qk_head_dim^-0.5.

DSA indexer with k-pool compression (TF:774-1062): per token the key k = LayerNorm(wk x) and the gate score
gamma = W_gate x; every complete pool of index_kpool consecutive tokens (aligned at token 0: no padding here) gets the
key sum_j softmax_j(gamma_j + ape_j) * k_j (per channel). A query at position t scores the complete pools whose last
token is visible: sum_h w_h * relu(q_h . K_p / sqrt(d)), w = weights_proj x / sqrt(n_heads), and attends to the tokens
of its top index_topk / index_kpool pools plus the current incomplete pool (always_select_tail). When a query sees at
most index_topk / index_kpool complete pools, every visible token is selected and the attention is exactly causal.
Selection is under stop_gradient (TF runs the indexer under no_grad); argpartition breaks exact score ties
differently from torch.topk.

Prefill and cached decode share one code path (DSACache keeps the latents, the indexer keys / gates of the incomplete
pool and the pooled keys; KDA keeps the conv tail and the recurrent state), so a decode step computes what the
prefill computes for that position (up to the float32 rounding of the recurrence order).

Known deviations from HF (rounding level): mx.fast.rms_norm rounds once (HF rounds the normalized value to the input
dtype, then multiplies by the weight), routed expert outputs are summed in one float32 reduction (HF index_add in the
model dtype), the SwiGLU activation runs in float32, the indexer pooled keys are summed in float32, absorbed MLA.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.switch_layers import SwitchGLU

from suc.models.mlx_qwen4_exp import _gated_delta

F32 = mx.float32
_HF_PREFIX = "model.language_model."
_DSA_TYPES = ("deepseek_sparse_attention", "indexed_attention", "full_attention")
_KEEP_F32 = (".hc_attn_", ".hc_ffn_", ".A_log", ".dt_bias", "_conv1d.", ".e_score_correction_bias")
# kept (non-SUC) linears of the MLX affine baseline (suc.evaluation.baselines.apply_representation)
_KEPT_LINEARS = ("mlp.gate", "self_attn.f_a_proj", "self_attn.f_b_proj", "self_attn.g_a_proj", "self_attn.g_b_proj",
                 "self_attn.b_proj", "self_attn.kv_b_proj", "indexer.wq_b", "indexer.wk", "indexer.weights_proj")


# ------------------------------------------------------------------------------------------- configuration

@dataclass
class TextArgs:
    vocab_size: int = 154880
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    scoring_func: str = "sigmoid"
    swiglu_limit: float = 10.0
    first_k_dense_replace: int = 3
    mlp_layer_types: Optional[list] = None
    layer_types: Optional[list] = None
    indexer_types: Optional[list] = None
    linear_attn_config: Optional[dict] = None
    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: Optional[float] = -5.0
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 4
    index_kpool_always_select_tail: bool = True
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    eos_token_id: Any = None

    @classmethod
    def from_dict(cls, d: dict) -> "TextArgs":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def __post_init__(self):
        n = self.num_hidden_layers
        la = self.linear_attn_config or {}
        self.linear_num_heads = int(la.get("num_heads", self.linear_num_heads))
        self.linear_head_dim = int(la.get("head_dim", self.linear_head_dim))
        self.linear_conv_kernel_dim = int(la.get("short_conv_kernel_size", self.linear_conv_kernel_dim))
        self.linear_lower_bound = la.get("gate_lower_bound", self.linear_lower_bound)
        if self.layer_types:
            kinds = ["dsa" if t in _DSA_TYPES else "kda" for t in self.layer_types[:n]]
            if set(self.layer_types[:n]) - set(_DSA_TYPES) - {"linear_attention"}:
                raise ValueError(f"unsupported glm5_next layer types {sorted(set(self.layer_types))}")
        elif la.get("kda_layers") is not None:
            kinds = ["kda" if i in la["kda_layers"] else "dsa" for i in range(n)]
        else:
            kinds = ["kda" if i % 4 != 3 else "dsa" for i in range(n)]
        self.layer_kinds = kinds
        if self.mlp_layer_types:
            self.sparse = [t == "sparse" for t in self.mlp_layer_types[:n]]
        else:
            self.sparse = [i >= self.first_k_dense_replace for i in range(n)]
        if self.indexer_types and any(t != "full" for t in self.indexer_types[:n]):
            raise NotImplementedError("glm5_next with shared (cross-layer) indexer top-k")
        if self.qk_rope_head_dim:
            raise NotImplementedError("glm5_next with RoPE (the checkpoint is NoPE)")
        if self.linear_lower_bound is None:
            raise NotImplementedError("glm5_next KDA without the safe gate lower bound")
        if self.scoring_func != "sigmoid" or self.n_group != 1 or self.topk_group != 1:
            raise NotImplementedError("glm5_next router other than sigmoid / one group")
        if self.n_shared_experts != 1 or self.attention_bias or self.tie_word_embeddings:
            raise NotImplementedError("glm5_next with n_shared_experts != 1, attention bias or tied embeddings")
        if self.index_topk % self.index_kpool:
            raise ValueError("index_topk must be a multiple of index_kpool")

    @property
    def stream_width(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


@dataclass
class ModelArgs:
    model_type: str
    text: TextArgs
    tie_word_embeddings: bool = False
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params: dict) -> "ModelArgs":
        text = dict(params.get("text_config") or params)
        return cls(model_type=params.get("model_type", "glm5_next"), text=TextArgs.from_dict(text))

    # attributes read by generic SUC code (autotune shapes, preflight)
    @property
    def hidden_size(self) -> int:
        return self.text.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self.text.moe_intermediate_size

    @property
    def vocab_size(self) -> int:
        return self.text.vocab_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text.num_hidden_layers

    @property
    def stream_width(self) -> int:
        return self.text.stream_width


# ------------------------------------------------------------------------------------------- building blocks

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class LayerNorm(nn.Module):
    """torch.nn.LayerNorm (weight + bias), float32 statistics, result in the input dtype."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.layer_norm(x, self.weight, self.bias, self.eps)


class ConvWeight(nn.Module):
    """A depthwise conv weight in the HF layout [C, 1, K] (parameter name `<conv>.weight`)."""

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.weight = mx.zeros((channels, 1, kernel))


def _causal_conv_silu(x: mx.array, w: mx.array, out_len: int) -> mx.array:
    """SiLU(depthwise causal conv) in float32 (TF causal_conv1d_fn: input cast to the float32 conv weight), x
    (B, out_len + K - 1, C) with the left context included; returned in float32."""
    k = w.shape[-1]
    wf = w[:, 0, :].astype(F32)
    acc = None
    for i in range(k):
        term = x[:, i: i + out_len, :].astype(F32) * wf[:, i]
        acc = term if acc is None else acc + term
    return nn.silu(acc)


def _l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """TF l2norm (FLA): x / sqrt(sum x^2 + eps), float32."""
    x = x.astype(F32)
    return x / mx.sqrt((x * x).sum(axis=-1, keepdims=True) + eps)


class LinearAttention(nn.Module):
    """Glm5NextTextLinearAttention (KDA with the safe-gate decay), module docstring."""

    def __init__(self, a: TextArgs):
        super().__init__()
        self.heads, self.head_dim = a.linear_num_heads, a.linear_head_dim
        d = self.heads * self.head_dim
        self.kernel = a.linear_conv_kernel_dim
        self.lower_bound = float(a.linear_lower_bound)
        self.eps = a.rms_norm_eps
        self.q_proj = nn.Linear(a.hidden_size, d, bias=False)
        self.k_proj = nn.Linear(a.hidden_size, d, bias=False)
        self.v_proj = nn.Linear(a.hidden_size, d, bias=False)
        self.q_conv1d = ConvWeight(d, self.kernel)
        self.k_conv1d = ConvWeight(d, self.kernel)
        self.v_conv1d = ConvWeight(d, self.kernel)
        self.f_a_proj = nn.Linear(a.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, d, bias=False)
        self.dt_bias = mx.zeros((d,), dtype=F32)
        self.A_log = mx.zeros((self.heads,), dtype=F32)
        self.b_proj = nn.Linear(a.hidden_size, self.heads, bias=False)
        self.g_a_proj = nn.Linear(a.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, d, bias=False)
        self.o_norm = RMSNorm(self.head_dim, a.rms_norm_eps)
        self.o_proj = nn.Linear(d, a.hidden_size, bias=False)

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        B, L, _ = x.shape
        H, Dk = self.heads, self.head_dim
        d = H * Dk
        qkv = mx.concatenate([self.q_proj(x), self.k_proj(x), self.v_proj(x)], axis=-1)
        n_keep = self.kernel - 1
        prev = cache[0] if cache is not None and cache[0] is not None else mx.zeros((B, n_keep, 3 * d), qkv.dtype)
        conv_in = mx.concatenate([prev.astype(qkv.dtype), qkv], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_in[:, -n_keep:, :])
        w = mx.concatenate([self.q_conv1d.weight, self.k_conv1d.weight, self.v_conv1d.weight], axis=0)
        conv = _causal_conv_silu(conv_in, w, L).astype(x.dtype)            # TF casts back to the input dtype
        q, k, v = (t.reshape(B, L, H, Dk) for t in mx.split(conv, [d, 2 * d], axis=-1))
        q = _l2norm(q) * (Dk ** -0.5)
        k = _l2norm(k)
        f = self.f_b_proj(self.f_a_proj(x)).astype(F32) + self.dt_bias.astype(F32)
        g = self.lower_bound * mx.sigmoid(mx.exp(self.A_log.astype(F32))[:, None] * f.reshape(B, L, H, Dk))
        beta = mx.sigmoid(self.b_proj(x)).astype(F32)
        state = cache[1] if cache is not None and cache[1] is not None else mx.zeros((B, H, Dk, Dk), dtype=F32)
        o, state = _gated_delta(q, k, v.astype(F32), mx.exp(g), beta, state)
        if cache is not None:
            cache[1] = state
        o = o.astype(x.dtype).astype(F32)                                   # TF: core_attn_out.to(input dtype)
        gate = self.g_b_proj(self.g_a_proj(x)).reshape(B, L, H, Dk).astype(F32)
        o = mx.fast.rms_norm(o, self.o_norm.weight.astype(F32), self.eps) * mx.sigmoid(gate)
        return self.o_proj(o.astype(x.dtype).reshape(B, L, d))


class DSACache:
    """Per-layer decode state of an MLA + DSA layer: MLA latents of every token, the indexer keys / gate scores of the
    incomplete pool and the pooled keys of every complete pool (one per index_kpool tokens)."""

    def __init__(self):
        self.offset = 0
        self.latent: mx.array | None = None          # (B, T, kv_lora_rank)
        self.raw_k: mx.array | None = None           # (B, r, d) r < kpool
        self.raw_g: mx.array | None = None
        self.pools: mx.array | None = None           # (B, P, d)

    @property
    def state(self):
        return [a for a in (self.latent, self.raw_k, self.raw_g, self.pools) if a is not None]

    def is_trimmable(self) -> bool:
        return False


class Indexer(nn.Module):
    """Glm5NextTextIndexer (k-pool DSA), module docstring. Returns a boolean token selection (B, L, T) or None when
    every visible token is selected (dense causal). Updates the pool state of the cache."""

    def __init__(self, a: TextArgs):
        super().__init__()
        self.n_heads, self.head_dim, self.kpool = a.index_n_heads, a.index_head_dim, a.index_kpool
        self.topk_pools = a.index_topk // a.index_kpool
        self.tail = a.index_kpool_always_select_tail
        self.wq_b = nn.Linear(a.q_lora_rank, a.index_n_heads * a.index_head_dim, bias=False)
        self.wk = nn.Linear(a.hidden_size, a.index_head_dim, bias=False)
        self.k_norm = LayerNorm(a.index_head_dim, 1e-6)
        self.weights_proj = nn.Linear(a.hidden_size, a.index_n_heads, bias=False)
        self.index_kpool_compress_gate = mx.zeros((a.index_head_dim, a.hidden_size))
        self.index_kpool_compress_ape = mx.zeros((a.index_kpool, a.index_head_dim))

    def _pool(self, k: mx.array, g: mx.array) -> mx.array:
        """(B, n * kpool, d) keys / gate scores -> (B, n, d) pooled keys (per-channel softmax over the pool)."""
        B, T, d = k.shape
        n = T // self.kpool
        logits = g.reshape(B, n, self.kpool, d).astype(F32) + self.index_kpool_compress_ape.astype(F32)
        p = mx.softmax(logits, axis=2)
        return (p * k.reshape(B, n, self.kpool, d).astype(F32)).sum(axis=2).astype(k.dtype)

    def __call__(self, x: mx.array, q_resid: mx.array, c: DSACache) -> mx.array | None:
        B, L, _ = x.shape
        x, q_resid = mx.stop_gradient(x), mx.stop_gradient(q_resid)
        p0 = c.offset
        k = self.k_norm(self.wk(x))
        g = x @ self.index_kpool_compress_gate.T.astype(x.dtype)
        if c.raw_k is not None:
            k, g = mx.concatenate([c.raw_k, k], axis=1), mx.concatenate([c.raw_g, g], axis=1)
        n_new = k.shape[1] // self.kpool
        if n_new:
            new = self._pool(k[:, : n_new * self.kpool], g[:, : n_new * self.kpool])
            c.pools = new if c.pools is None else mx.concatenate([c.pools, new], axis=1)
        c.raw_k, c.raw_g = k[:, n_new * self.kpool:], g[:, n_new * self.kpool:]
        T = p0 + L
        n_pools = T // self.kpool
        if n_pools <= self.topk_pools:
            return None                                   # every query sees <= topk pools: dense causal
        pos = p0 + mx.arange(L)
        n_vis = (pos + 1) // self.kpool                   # complete pools visible to each query
        q = self.wq_b(q_resid).reshape(B, L, self.n_heads, self.head_dim).astype(F32)
        s = mx.maximum(mx.matmul(q, c.pools[:, None].astype(F32).swapaxes(-1, -2)) * self.head_dim ** -0.5, 0.0)
        w = self.weights_proj(x).astype(F32) * self.n_heads ** -0.5        # (B, L, H)
        score = (w[..., None] * s).sum(axis=2)                              # (B, L, P)
        visible = mx.arange(n_pools)[None, :] < n_vis[:, None]              # (L, P)
        score = mx.where(visible[None], score, -mx.inf)
        top = mx.argpartition(-score, kth=self.topk_pools - 1, axis=-1)[..., : self.topk_pools]
        chosen = mx.put_along_axis(mx.zeros(score.shape, dtype=mx.bool_), top, mx.array(True), axis=-1)
        chosen = chosen & visible[None]
        tok = mx.arange(T)
        in_pool = tok[None, :] < (n_vis * self.kpool)[:, None]              # (L, T) token of a visible complete pool
        sel = mx.take(chosen, mx.minimum(tok // self.kpool, n_pools - 1), axis=-1) & in_pool[None]
        causal = tok[None, :] <= pos[:, None]
        if self.tail:
            sel = sel | (~in_pool & causal)[None]
        return mx.stop_gradient(sel & causal[None])


class Attention(nn.Module):
    """Glm5NextTextAttention: NoPE MLA (absorbed kv_b) restricted to the indexer's token selection."""

    def __init__(self, a: TextArgs):
        super().__init__()
        self.heads, self.nope, self.v_dim, self.rank = a.num_attention_heads, a.qk_nope_head_dim, a.v_head_dim, \
            a.kv_lora_rank
        self.scale = a.qk_head_dim ** -0.5
        self.q_a_proj = nn.Linear(a.hidden_size, a.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(a.q_lora_rank, a.rms_norm_eps)
        self.q_b_proj = nn.Linear(a.q_lora_rank, a.num_attention_heads * a.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(a.hidden_size, a.kv_lora_rank + a.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(a.kv_lora_rank, a.rms_norm_eps)
        self.kv_b_proj = nn.Linear(a.kv_lora_rank, a.num_attention_heads * (a.qk_nope_head_dim + a.v_head_dim),
                                   bias=False)
        self.o_proj = nn.Linear(a.num_attention_heads * a.v_head_dim, a.hidden_size, bias=False)
        self.indexer = Indexer(a)

    def __call__(self, x: mx.array, cache: DSACache | None = None) -> mx.array:
        B, L, _ = x.shape
        c = cache if cache is not None else DSACache()
        p0 = c.offset
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_resid).reshape(B, L, self.heads, self.nope).transpose(0, 2, 1, 3)   # (B, H, L, nope)
        lat = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))                                    # (B, L, rank)
        sel = self.indexer(x, q_resid, c)
        lat_all = lat if c.latent is None else mx.concatenate([c.latent, lat], axis=1)
        T = lat_all.shape[1]
        w = self.kv_b_proj.weight.reshape(self.heads, self.nope + self.v_dim, self.rank)
        wk, wv = w[:, : self.nope], w[:, self.nope:]                       # (H, nope, rank), (H, v, rank)
        qa = mx.matmul(q, wk[None].astype(q.dtype))                         # (B, H, L, rank) = W_k,h^T q_h
        if sel is not None:
            mask = sel[:, None]
        elif L == 1:
            mask = None
        else:
            mask = (mx.arange(T)[None, :] <= (p0 + mx.arange(L))[:, None])[None, None]
        kv = lat_all[:, None]                                                # (B, 1, T, rank)
        o = mx.fast.scaled_dot_product_attention(qa, kv, kv, scale=self.scale, mask=mask)   # (B, H, L, rank)
        o = mx.matmul(o, wv.swapaxes(-1, -2)[None].astype(o.dtype))        # (B, H, L, v) = W_v,h (P c)
        out = self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, self.heads * self.v_dim))
        if cache is not None:
            c.latent = lat_all
            c.offset = p0 + L
        return out


def _clamped_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    """TF Glm5NextTextMLP / Experts._apply_gate: silu(min(gate, limit)) * clip(up, -limit, limit), float32."""
    g, u = gate.astype(F32), up.astype(F32)
    if limit > 0:
        g = mx.minimum(g, limit)
        u = mx.clip(u, -limit, limit)
    return nn.silu(g) * u


class MLP(nn.Module):
    def __init__(self, dims: int, hidden: int, limit: float):
        super().__init__()
        self.gate_proj = nn.Linear(dims, hidden, bias=False)
        self.up_proj = nn.Linear(dims, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dims, bias=False)
        self.limit = limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.limit).astype(x.dtype))


class Router(nn.Module):
    """Glm5NextTextTopkRouter: float32 sigmoid scores, top-k of scores + e_score_correction_bias (selection only), the
    unbiased scores of the chosen experts normalised (+1e-20) and scaled by routed_scaling_factor."""

    def __init__(self, a: TextArgs):
        super().__init__()
        self.top_k, self.scale, self.norm = a.num_experts_per_tok, a.routed_scaling_factor, a.norm_topk_prob
        self.weight = mx.zeros((a.n_routed_experts, a.hidden_size))
        self.e_score_correction_bias = mx.zeros((a.n_routed_experts,), dtype=F32)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        s = mx.sigmoid(x.astype(F32) @ self.weight.astype(F32).T)
        biased = mx.stop_gradient(s + self.e_score_correction_bias.astype(F32))
        inds = mx.stop_gradient(mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[..., : self.top_k])
        inds = inds.astype(mx.uint32)
        w = mx.take_along_axis(s, inds, axis=-1)
        if self.norm:
            w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
        return inds, w * self.scale


class RoutedExperts(SwitchGLU):
    """SwitchGLU (mlx-lm layout, the activation recorder sees the usual (x, indices) calls) with the clamped SwiGLU and
    the routing weight applied AFTER down_proj (TF:133-135); outputs summed in float32.

    Expert streaming (suc.runtime.expert_streaming): with a stream attached, the call is delegated to it; it fetches
    the selected experts from an ExpertCache and runs `compute` with SUC modules over those experts only (remapped
    indices), i.e. the same arithmetic as with the resident stack."""

    def __init__(self, dims: int, hidden: int, n: int, limit: float):
        super().__init__(dims, hidden, n, bias=False)
        self.limit = limit
        self._stream = None

    def __call__(self, x: mx.array, inds: mx.array, w: mx.array) -> mx.array:
        if self._stream is not None:
            return self._stream.run(self, x, inds, w)
        return self.compute(self.gate_proj, self.up_proj, self.down_proj, x, inds, w)

    def compute(self, gate_p, up_p, down_p, x: mx.array, inds: mx.array, w: mx.array) -> mx.array:
        """sum_k w_k down_k(SwiGLU(gate_k x, up_k x)) (float32) with the given expert projection modules."""
        k = inds.shape[-1]
        lead = inds.shape
        xe = mx.expand_dims(x, (-2, -3))                  # (..., 1, 1, D)
        do_sort = inds.size >= 64
        if do_sort:
            flat = inds.flatten()
            order = mx.argsort(flat)
            inv = mx.argsort(order)
            xe = xe.flatten(0, -3)[order // k]            # (N*k, 1, D)
            idx = flat[order]
        else:
            idx = inds
        up = up_p(xe, idx, sorted_indices=do_sort)
        gate = gate_p(xe, idx, sorted_indices=do_sort)
        h = _clamped_swiglu(gate, up, self.limit).astype(x.dtype)
        y = down_p(h, idx, sorted_indices=do_sort)
        if do_sort:
            y = y[inv].reshape(*lead, 1, -1)
        y = y.squeeze(-2).astype(F32) * w[..., None].astype(F32)
        return y.sum(axis=-2)


class MoE(nn.Module):
    def __init__(self, a: TextArgs):
        super().__init__()
        self.gate = Router(a)
        self.switch_mlp = RoutedExperts(a.hidden_size, a.moe_intermediate_size, a.n_routed_experts, a.swiglu_limit)
        self.shared_experts = MLP(a.hidden_size, a.moe_intermediate_size * a.n_shared_experts, a.swiglu_limit)

    def __call__(self, x: mx.array) -> mx.array:
        inds, w = self.gate(x)
        y = self.switch_mlp(x, inds, w).astype(x.dtype)
        return y + self.shared_experts(x)


class DecoderLayer(nn.Module):
    """One block on the stream tensor (B, L, hc * D) (shape-preserving); mHC parameters are direct parameters
    `hc_{attn,ffn}_{fn,base,scale}` (the checkpoint names)."""

    def __init__(self, a: TextArgs, layer: int):
        super().__init__()
        self.kind = a.layer_kinds[layer]
        self.hc, self.dim = a.hc_mult, a.hidden_size
        self.iters, self.hc_eps, self.norm_eps = a.hc_sinkhorn_iters, a.hc_eps, a.rms_norm_eps
        mix = (2 + a.hc_mult) * a.hc_mult
        for k in ("attn", "ffn"):
            setattr(self, f"hc_{k}_fn", mx.zeros((mix, a.hc_mult * a.hidden_size), dtype=F32))
            setattr(self, f"hc_{k}_base", mx.zeros((mix,), dtype=F32))
            setattr(self, f"hc_{k}_scale", mx.ones((3,), dtype=F32))
        self.self_attn = LinearAttention(a) if self.kind == "kda" else Attention(a)
        self.mlp = MoE(a) if a.sparse[layer] else MLP(a.hidden_size, a.intermediate_size, a.swiglu_limit)
        self.input_layernorm = RMSNorm(a.hidden_size, a.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(a.hidden_size, a.rms_norm_eps)

    def _hc_pre(self, x: mx.array, k: str) -> tuple[mx.array, mx.array, mx.array]:
        """Glm5NextTextHyperConnection.forward (TF:287-331): x (B, L, hc, D) -> (post (B, L, hc), comb (B, L, hc, hc)
        float32, collapsed input (B, L, D) in x.dtype)."""
        B, L, hc, D = x.shape
        fn, base, sc = (getattr(self, f"hc_{k}_{n}").astype(F32) for n in ("fn", "base", "scale"))
        xf = x.reshape(B, L, hc * D).astype(F32)
        xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.norm_eps)
        m = xf @ fn.T
        pre = mx.sigmoid(m[..., :hc] * sc[0] + base[:hc]) + self.hc_eps
        post = 2 * mx.sigmoid(m[..., hc:2 * hc] * sc[1] + base[hc:2 * hc])
        comb = mx.softmax((m[..., 2 * hc:] * sc[2] + base[2 * hc:]).reshape(B, L, hc, hc), axis=-1) + self.hc_eps
        comb = comb / (comb.sum(axis=-2, keepdims=True) + self.hc_eps)
        for _ in range(self.iters - 1):
            comb = comb / (comb.sum(axis=-1, keepdims=True) + self.hc_eps)
            comb = comb / (comb.sum(axis=-2, keepdims=True) + self.hc_eps)
        y = (pre[..., None] * x.astype(F32)).sum(axis=2).astype(x.dtype)
        return post, comb, y

    @staticmethod
    def _hc_post(f: mx.array, res: mx.array, post: mx.array, comb: mx.array) -> mx.array:
        """post_j f + sum_i comb_ij res_i in the model dtype (TF:1337)."""
        dt = res.dtype
        return post.astype(dt)[..., None] * f[..., None, :] + mx.swapaxes(comb.astype(dt), -1, -2) @ res

    def __call__(self, h: mx.array, mask=None, cache=None) -> mx.array:
        """`mask` is accepted for the generic block API and ignored (causal / DSA masks are built here, no padding)."""
        B, L, _ = h.shape
        x = h.reshape(B, L, self.hc, self.dim)
        post, comb, y = self._hc_pre(x, "attn")
        x = self._hc_post(self.self_attn(self.input_layernorm(y), cache), x, post, comb)
        post, comb, y = self._hc_pre(x, "ffn")
        x = self._hc_post(self.mlp(self.post_attention_layernorm(y)), x, post, comb)
        return x.reshape(B, L, self.hc * self.dim)


class TextModel(nn.Module):
    def __init__(self, a: TextArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(a.vocab_size, a.hidden_size)
        self.layers = [DecoderLayer(a, i) for i in range(a.num_hidden_layers)]
        self.norm = RMSNorm(a.hidden_size, a.rms_norm_eps)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.text = args.text
        self.model = TextModel(args.text)
        self.lm_head = nn.Linear(args.text.hidden_size, args.text.vocab_size, bias=False)

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.layers

    @staticmethod
    def quant_predicate(path: str, module: nn.Module) -> bool:
        """MLX affine baseline: quantize exactly the tensors SUC encodes (Glm5NextAdapter eligible set) plus the
        embedding / LM head; routers, KDA gate factors, kv_b and the indexer stay BF16 as in `suc compress`."""
        return not path.endswith(_KEPT_LINEARS)

    # -- stream API (suc.models.streaming) --------------------------------------
    def embed_stream(self, tokens: mx.array, cache=None) -> mx.array:
        e = self.model.embed_tokens(tokens)
        return mx.concatenate([e] * self.text.hc_mult, axis=-1)

    def head_stream(self, h: mx.array) -> mx.array:
        a = self.text
        B, L, _ = h.shape
        x = h.reshape(B, L, a.hc_mult, a.hidden_size).astype(F32).mean(axis=2).astype(h.dtype)
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
        return [ArraysCache(size=2) if layer.kind == "kda" else DSACache() for layer in self.model.layers]

    def set_dtype(self, dtype: mx.Dtype, predicate=lambda x: mx.issubdtype(x, mx.floating)) -> None:
        """As nn.Module.set_dtype, but the parameters HF keeps in float32 stay float32."""
        keep = [(k, v) for k, v in tree_flatten(self.parameters()) if any(t in k for t in _KEEP_F32)]
        super().set_dtype(dtype, predicate)
        self.update(tree_unflatten([(k, v.astype(F32)) for k, v in keep]))

    # -- loading ----------------------------------------------------------------
    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Checkpoint names -> parameter names: drop the vision tower, the MTP layer and the scale companions,
        dequantize FP8 weights with their weight_scale_inv (BF16, lazily), strip `model.language_model.`, stack the
        routed experts into switch_mlp. Names already in the SUC namespace pass through."""
        from suc.io.quantized_source import dequantize
        from suc.quant.stacks import stack_of

        n_layers = self.text.num_hidden_layers
        out: dict[str, mx.array] = {}
        experts: dict[str, dict[int, mx.array]] = {}
        for k, v in weights.items():
            if k.startswith("model.visual.") or k.endswith(".weight_scale_inv"):
                continue
            name = "model." + k[len(_HF_PREFIX):] if k.startswith(_HF_PREFIX) else k
            parts = name.split(".")
            if name.startswith("model.layers.") and int(parts[2]) >= n_layers:
                continue                                                    # MTP layer
            sk = k + "_scale_inv"
            if sk in weights:
                v = dequantize("fp8_block_f32", v, weights[sk], name=name, check=False)
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


__all__ = ["ModelArgs", "TextArgs", "Model", "DSACache", "RoutedExperts"]
