"""MLX implementation of the `qwen4_exp` text model (Qwen/Qwen3.8-Flash-Next), see docs/arch/qwen4_exp.md.

Ported from transformers `models/qwen4_exp/{modular,modeling}_qwen4_exp.py` (main, fetched 2026-10-03); mlx-lm has no
`qwen4_exp` module. Reused from mlx-lm: the gated-delta recurrence (`gated_delta.py`: Metal kernel + ops reference),
SwitchGLU, KVCache / ArraysCache. Vision tower and MTP head are dropped (text-only LM).

Parameter namespace (= the SUC tensor namespace of `suc.models.adapters.Qwen4ExpAdapter.source_store`): the HF names
with `model.language_model.` -> `model.`; the fused expert stacks `mlp.experts.gate_up_proj` [E, 2I, H] /
`mlp.experts.down_proj` [E, H, I] become `mlp.switch_mlp.{gate,up,down}_proj.weight` (gate = rows [0, I), up = rows
[I, 2I), as `Qwen4ExpTextExperts.forward` chunks them). Every other tensor keeps its HF shape (conv weights stay
[C, 1, K]: the depthwise convolutions are written out, no layout change; RMSNorm weights stay raw, the (1 + w) is applied
in float32 at run time as HF does).

Stream layout. The residual of this model is `hc_count` streams of `hidden_size` (4 x 2560 = 10240 wide, HF
`hidden_states.repeat(1, 1, hc)`). A transformer block maps the stream tensor to the stream tensor, so the
block-sequential machinery of suc.models.streaming runs unchanged with `block(h, mask)`. Layers with PLE (per-layer
n-gram embeddings, `ple_layer_ids`, one-indexed) also need the hashed n-gram embedding rows of the TOKENS; they are
gathered once at embedding time (`embed_stream`) and carried as extra channels behind the streams (the stream tensor
is `hc*H + E * len(ple_layer_ids)` wide in EVERY block, so blocks are shape-preserving as the generic code — Fisher
VJPs, fine-tuning losses — assumes; PLE layer k reads segment k, the other blocks pass the segments through). The
n-gram rows depend only on the token ids, so this is exactly the HF computation. `head_stream` applies the
final hyper-connection mixer (there is no final RMSNorm) and the LM head.

N-gram tables (51.2 B parameters, 102 GB BF16) are NOT parameters: they stay memory-mapped (safetensors shards or SUC1
payload views, raw BF16 or 4-bit row-affine after `suc repack`) and are gathered on the CPU (`NgramTable`), 16 rows per
token; they are never concatenated and never made resident.

QSA indexer: the selection is computed only when a query sees more than `indexer_budget / indexer_compress_ratio`
complete key blocks; below that (every sequence of <= budget + ratio - 1 tokens) it selects every visible token and the
attention is exactly dense causal, as in HF. Selection uses exact top-k (argpartition); HF uses torch.topk, so ties
between equal block scores (e.g. several all-ReLU-zero blocks) may be broken differently.

Known numerical deviations from HF (all bf16 rounding-order level): q/k L2 normalization of the gated delta rule as in
mlx-lm (eps inside the mean instead of the sum), RoPE through mx.fast.rope (float32 internally), expert outputs summed in
one reduction (HF accumulates with index_add in bf16). A torch reference check on the real checkpoint is
scripts/qwen4_exp_reference_check.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops
from mlx_lm.models.switch_layers import SwitchGLU

_HF_PREFIX = "model.language_model."
_BUFFERS = ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")
_KEPT_LINEARS = ("mlp.gate", "mlp.shared_expert_gate", "linear_attn.in_proj_a", "linear_attn.in_proj_b",
                 "indexer.index_qk_proj")


# ------------------------------------------------------------------------------------------- configuration

@dataclass
class TextArgs:
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    layer_types: Optional[list] = None
    full_attention_interval: int = 4
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    hc_count: int = 4
    hc_lowrank: int = 320
    ple_layer_ids: Optional[list] = None
    ple_embed_dim: Optional[int] = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = 512
    indexer_n_heads: Optional[int] = None
    indexer_kv_heads: Optional[int] = None
    indexer_head_dim: Optional[int] = None
    indexer_budget: Optional[int] = None
    indexer_compress_ratio: Optional[int] = None
    output_gate_type: Optional[str] = None
    hidden_act: str = "silu"
    eos_token_id: Any = None
    rope_parameters: dict = field(default_factory=lambda: {"rope_theta": 1e7, "partial_rotary_factor": 0.25})
    max_position_embeddings: int = 262144
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "TextArgs":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["linear_attention" if (i + 1) % self.full_attention_interval else "full_attention"
                                for i in range(self.num_hidden_layers)]
        self.layer_types = ["full_attention" if t in ("full_attention", "indexed_attention") else t
                            for t in self.layer_types]
        if set(self.layer_types) - {"linear_attention", "full_attention"}:
            raise ValueError(f"unsupported qwen4_exp layer types {sorted(set(self.layer_types))}")
        self.ple_layer_ids = sorted(set(self.ple_layer_ids or []))
        self.ple_embed_dim = self.ple_embed_dim or self.hidden_size
        if self.attention_bias:
            raise NotImplementedError("qwen4_exp with attention_bias")
        gate = self.output_gate_type or self.hidden_act
        if gate not in ("sigmoid", "silu"):
            raise ValueError(f"unsupported output gate activation {gate!r}")
        for i in self.ple_layer_ids:
            if self.layer_types[i - 1] != "linear_attention":
                raise ValueError("qwen4_exp PLE is only defined on linear-attention layers")

    @property
    def eos(self) -> int:
        e = self.eos_token_id
        return int(e[0] if isinstance(e, (list, tuple)) else e)

    @property
    def stream_width(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    def ple_index(self, layer: int) -> int | None:
        """Index into ple_layer_ids of 0-indexed `layer`, or None (ple_layer_ids is one-indexed)."""
        return self.ple_layer_ids.index(layer + 1) if layer + 1 in self.ple_layer_ids else None

    @property
    def rope_theta(self) -> float:
        return float(self.rope_parameters.get("rope_theta", 1e7))

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * float(self.rope_parameters.get("partial_rotary_factor", 1.0)))


@dataclass
class ModelArgs:
    model_type: str
    text: TextArgs
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params: dict) -> "ModelArgs":
        text = dict(params.get("text_config") or params)
        tie = bool(params.get("tie_word_embeddings", text.get("tie_word_embeddings", False)))
        if tie:
            raise NotImplementedError("qwen4_exp with tied embeddings")
        return cls(model_type=params["model_type"], text=TextArgs.from_dict(text), tie_word_embeddings=tie)

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


# ------------------------------------------------------------------------------------------- n-gram hashing (PLE)

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> list[int]:
    """HF `_build_layer_multipliers` (odd multipliers below (2^63 - 1) // vocab, so token * m never overflows int64)."""
    max_long = (1 << 63) - 1
    half_bound = max(1, (max_long // max(unigram_vocab_size, 1)) // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    return [2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1
            for i in range(ngram_size)]


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


@lru_cache(maxsize=None)
def _primes_after(start: int, count: int) -> tuple[int, ...]:
    """The `count` consecutive primes > start (HF `_find_nth_prime_after(start, n)` for n = 1..count)."""
    out, p = [], start
    while len(out) < count:
        p += 1
        if _is_prime(p):
            out.append(p)
    return tuple(out)


def ngram_head_sizes(a: TextArgs, ple_layer_index: int) -> tuple[list[int], list[int], int]:
    """(per-head prime vocab sizes, head row offsets, padded table rows) of one PLE layer."""
    heads = a.ngram_heads
    primes = _primes_after(a.ngram_vocab_size_base - 1, (ple_layer_index + 1) * heads)
    sizes = list(primes[ple_layer_index * heads:])
    offsets = [int(x) for x in np.concatenate([[0], np.cumsum(sizes)[:-1]])]
    total = sum(sizes)
    div = a.make_ngram_vocab_size_divisible_by
    return sizes, offsets, int(math.ceil(total / div) * div)


def _shift_right_ignore_eos(tok: np.ndarray, shift: int, eos: int) -> np.ndarray:
    """HF `_shift_right_ignore_eos` on (B, T) int64: token `shift` positions back within the EOS-delimited segment,
    EOS where that position lies before the segment start."""
    if shift == 0:
        return tok
    b, t = tok.shape
    pos = np.arange(t, dtype=np.int64)
    eos_pos = np.where(tok == eos, pos, -1)
    prev_incl = np.maximum.accumulate(eos_pos, axis=1)
    prev = np.concatenate([np.full((b, 1), -1, dtype=np.int64), prev_incl[:, :-1]], axis=1)
    in_seg = pos[None] - (prev + 1)
    src = pos - shift
    shifted = tok[:, np.clip(src, 0, None)]
    valid = (in_seg >= shift) & (src[None] >= 0)
    return np.where(valid, shifted, eos)


def ngram_ids(tokens: np.ndarray, context: np.ndarray, a: TextArgs, ple_layer_index: int) -> np.ndarray:
    """Table row ids (B, T, ngram_heads) of the current `tokens` (B, T) given the `context` (B, ngram_size - 1) of
    previous ids (EOS-filled at the start of a sequence), exactly as HF `Qwen4ExpTextNGramEmbedding.forward`."""
    mult = layer_multipliers(a.vocab_size, a.ngram_size, ple_layer_index, a.seed)
    sizes, offsets, _ = ngram_head_sizes(a, ple_layer_index)
    hist = np.concatenate([context.astype(np.int64), tokens.astype(np.int64)], axis=1)
    shifted = [_shift_right_ignore_eos(hist, s, a.eos) for s in range(a.ngram_size)]
    sizes_np, offs_np = np.array(sizes, dtype=np.int64), np.array(offsets, dtype=np.int64)
    blocks = []
    for n in range(2, a.ngram_size + 1):
        lo, hi = (n - 2) * a.heads_per_ngram, (n - 1) * a.heads_per_ngram
        mix = shifted[0] * mult[0]
        for p in range(1, n):
            mix = np.bitwise_xor(mix, shifted[p] * mult[p])
        blocks.append(np.remainder(mix[..., None], sizes_np[lo:hi]) + offs_np[lo:hi])
    return np.concatenate(blocks, axis=-1)[:, -tokens.shape[1]:]


class RowAffineShard:
    """One n-gram table shard stored row-wise affine (suc.quant.row_affine; SUC1 `affine` record with group_size = row
    width): packed code words (rows, words) uint32, scales / biases (rows, 1) f16 or bf16 bit patterns — all
    memory-mapped views. take(rows) decodes only the requested rows (float32 q * s + b, = the reference decoder)."""

    def __init__(self, words: np.ndarray, scales: np.ndarray, biases: np.ndarray, *, bits: int, dim: int,
                 scale_dtype: str = "f16"):
        if words.shape[0] != scales.shape[0] or scales.shape != biases.shape:
            raise ValueError("row-affine shard: words / scales / biases disagree in rows")
        self.words, self.scales, self.biases = words, scales, biases
        self.bits, self.scale_dtype = bits, scale_dtype
        self.shape = (int(words.shape[0]), int(dim))
        self.dtype = np.dtype(np.float32)          # decoded dtype

    def take(self, rows: np.ndarray) -> np.ndarray:
        from suc.quant.row_affine import dequantize_rows

        return dequantize_rows(self.words[rows], self.scales[rows], self.biases[rows], self.bits, self.shape[1],
                               self.scale_dtype)


class NgramTable:
    """One PLE layer's n-gram embedding: the row-split shards as memory-mapped arrays, gathered on the CPU (only the
    requested rows are ever read; the table is never resident).

    shards: arrays (rows_j, dim) — uint16 = bf16 bit patterns (safetensors BF16 / SUC1 raw bf16), or float32 — or
    RowAffineShard (4-bit row-affine SUC1 records, decoded to float32 per gathered row)."""

    def __init__(self, shards: list):
        if not shards:
            raise ValueError("no n-gram shards")
        self.shards = shards
        self.dim = int(shards[0].shape[1])
        self.starts = np.concatenate([[0], np.cumsum([s.shape[0] for s in shards])]).astype(np.int64)
        kinds = {s.dtype for s in shards}
        if len(kinds) != 1:
            raise ValueError(f"mixed n-gram shard dtypes {kinds}")
        self.np_dtype = kinds.pop()
        self.quantized = isinstance(shards[0], RowAffineShard)
        if any(isinstance(s, RowAffineShard) != self.quantized for s in shards):
            raise ValueError("mixed raw / row-affine n-gram shards")

    @property
    def rows(self) -> int:
        return int(self.starts[-1])

    def gather_np(self, ids: np.ndarray) -> np.ndarray:
        """Rows of `ids` (any shape, int64) -> numpy (*ids.shape, dim): bf16 bit patterns (raw bf16 shards), float32
        (raw f32 or row-affine shards)."""
        flat = ids.reshape(-1)
        if flat.size and (flat.min() < 0 or flat.max() >= self.rows):
            raise IndexError("n-gram row id out of range")
        out = np.empty((flat.size, self.dim), dtype=self.np_dtype)
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        for j in np.unique(shard):
            sel = shard == j
            local = flat[sel] - self.starts[j]
            s = self.shards[j]
            out[sel] = s.take(local) if self.quantized else s[local]
        return out.reshape(*ids.shape, self.dim)

    def gather(self, ids: np.ndarray) -> mx.array:
        """Rows of `ids` (any shape, int64) -> mx.array (*ids.shape, dim) (bf16 for raw bf16 shards, else float32)."""
        a = mx.array(self.gather_np(ids))
        return a.view(mx.bfloat16) if self.np_dtype == np.uint16 else a


class _Tables:
    """Holder of the NgramTable of each PLE layer (a plain object, so nn.Module does not treat it as parameters)."""

    def __init__(self) -> None:
        self.by_layer: dict[int, NgramTable] = {}


def ngram_shard_name(layer: int, j: int) -> str:
    return f"model.layers.{layer}.ple.ple_embedding.ngram_embedding.shard_{j}.weight"


# ------------------------------------------------------------------------------------------- building blocks

class RMSNorm1p(nn.Module):
    """Qwen4ExpTextRMSNorm: float32 x * rsqrt(mean(x^2) + eps) * (1 + w), optionally per `group` channels."""

    def __init__(self, dims: int, eps: float = 1e-6, group: int | None = None):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps, self.group = eps, group

    def __call__(self, x: mx.array) -> mx.array:
        xs = x.astype(mx.float32)
        if self.group is not None:
            xs = xs.reshape(*x.shape[:-1], -1, self.group)
        y = mx.fast.rms_norm(xs, None, self.eps).reshape(x.shape)
        return (y * (1.0 + self.weight.astype(mx.float32))).astype(x.dtype)


class RMSNormGated(nn.Module):
    """Qwen4ExpTextRMSNormGated: plain-w RMSNorm of x times act(gate) (sigmoid for output_gate_type 'sigmoid')."""

    def __init__(self, dims: int, eps: float, activation: str):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps, self.activation = eps, activation

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        h = self.weight * mx.fast.rms_norm(x.astype(mx.float32), None, self.eps).astype(x.dtype)
        g = gate.astype(mx.float32)
        g = mx.sigmoid(g) if self.activation == "sigmoid" else nn.silu(g)
        return (h.astype(mx.float32) * g).astype(x.dtype)


class ConvWeight(nn.Module):
    """Holds a depthwise conv weight in the HF layout [C, 1, K] (parameter name `<conv>.weight`)."""

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.weight = mx.zeros((channels, 1, kernel))


def depthwise_causal_conv(x: mx.array, w: mx.array, out_len: int, dilation: int = 1) -> mx.array:
    """y[:, t, c] = sum_k w[c, 0, k] * x[:, t + k * dilation, c] for t < out_len; x is (B, out_len + (K-1)*dilation, C)
    (left context included). Accumulated in float32, returned in x's dtype (torch F.conv1d with groups=C)."""
    k = w.shape[-1]
    wf = w[:, 0, :].astype(mx.float32)
    acc = None
    for i in range(k):
        term = x[:, i * dilation: i * dilation + out_len, :].astype(mx.float32) * wf[:, i]
        acc = term if acc is None else acc + term
    return acc.astype(x.dtype)


@mx.custom_function
def _gated_delta(q, k, v, g, beta, state):
    if mx.default_device() == mx.gpu and mx.metal.is_available() and k.shape[-1] % 32 == 0 and v.shape[-1] % 4 == 0:
        y, s = gated_delta_kernel(q, k, v, g, beta, state)
    else:
        y, s = gated_delta_ops(q, k, v, g, beta, state)
    return y, s


@_gated_delta.vjp
def _gated_delta_vjp(primals, cotangents, outputs):
    """The Metal kernel has no VJP: differentiate the ops reference (needed by the streamed Fisher / fine-tuning)."""
    _, grads = mx.vjp(lambda *p: gated_delta_ops(*p), list(primals), list(cotangents))
    return tuple(grads)


class GatedDeltaNet(nn.Module):
    """Qwen4ExpTextGatedDeltaNet (= Qwen3.5 GatedDeltaNet with a sigmoid output gate, mlx-lm qwen3_5.py)."""

    def __init__(self, a: TextArgs):
        super().__init__()
        self.num_v_heads, self.num_k_heads = a.linear_num_value_heads, a.linear_num_key_heads
        self.head_k_dim, self.head_v_dim = a.linear_key_head_dim, a.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("linear_num_value_heads must be a multiple of linear_num_key_heads")
        self.conv_kernel_size = a.linear_conv_kernel_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv1d = ConvWeight(self.conv_dim, self.conv_kernel_size)
        self.in_proj_qkv = nn.Linear(a.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(a.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(a.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(a.hidden_size, self.num_v_heads, bias=False)
        self.dt_bias = mx.ones((self.num_v_heads,))
        self.A_log = mx.zeros((self.num_v_heads,))
        self.norm = RMSNormGated(self.head_v_dim, a.rms_norm_eps, a.output_gate_type or a.hidden_act)
        self.out_proj = nn.Linear(self.value_dim, a.hidden_size, bias=False)

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        B, S, _ = x.shape
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        n_keep = self.conv_kernel_size - 1
        state0 = cache[0] if cache is not None and cache[0] is not None else mx.zeros((B, n_keep, self.conv_dim),
                                                                                  dtype=qkv.dtype)
        conv_in = mx.concatenate([state0.astype(qkv.dtype), qkv], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_in[:, -n_keep:, :])
        conv_out = nn.silu(depthwise_causal_conv(conv_in, self.conv1d.weight, S))
        q, k, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(B, S, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, S, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, S, self.num_v_heads, self.head_v_dim)
        # HF / FLA l2norm: x * rsqrt(sum(x^2) + 1e-6). rms_norm adds eps to the MEAN, so eps = 1e-6 / Dk makes
        # inv * rms_norm(x) exactly that l2norm (mlx-lm qwen3_5 passes 1e-6, i.e. 1e-6 * Dk on the sum: a 1e-4
        # relative error on q/k, measured 4.5e-4 on block 0's output vs torch float32).
        inv = self.head_k_dim ** -0.5
        eps = 1e-6 / self.head_k_dim
        q = (inv ** 2) * mx.fast.rms_norm(q, None, eps)        # = l2norm(q) / sqrt(Dk)
        k = inv * mx.fast.rms_norm(k, None, eps)               # = l2norm(k)
        beta = mx.sigmoid(b)
        g = mx.exp(-mx.exp(self.A_log.astype(mx.float32))
                   * nn.softplus(a.astype(mx.float32) + self.dt_bias.astype(mx.float32)))
        state = cache[1] if cache is not None and cache[1] is not None else \
            mx.zeros((B, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=mx.float32)
        out, state = _gated_delta(q, k, v, g, beta, state)
        if cache is not None:
            cache[1] = state
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


def _rope_at(x: mx.array, positions: mx.array, dims: int, base: float) -> mx.array:
    """Non-interleaved (rotate_half) RoPE of the first `dims` channels of x (..., n, D) at integer `positions` (n,)."""
    inv = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    ang = positions.astype(mx.float32)[:, None] * inv[None]
    cos = mx.concatenate([mx.cos(ang)] * 2, axis=-1)
    sin = mx.concatenate([mx.sin(ang)] * 2, axis=-1)
    xr = x[..., :dims].astype(mx.float32)
    half = dims // 2
    rot = mx.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
    y = (xr * cos + rot * sin).astype(x.dtype)
    return mx.concatenate([y, x[..., dims:]], axis=-1) if dims < x.shape[-1] else y


class Indexer(nn.Module):
    """Qwen4ExpTextQSAIndexer: per query, the top `budget / ratio` blocks of `ratio` consecutive tokens (block key =
    RoPE(k_layernorm(mean of raw keys)) at the block start, score = sum_h relu(q_h . k_b) / sqrt(d)) plus the
    incomplete tail. Returns a bool mask (B, 1, L, T_kv), or None when every visible token is selected."""

    def __init__(self, a: TextArgs):
        super().__init__()
        self.n_heads, self.kv_heads, self.head_dim = a.indexer_n_heads, a.indexer_kv_heads, a.indexer_head_dim
        if self.kv_heads != 1:
            raise ValueError("qwen4_exp QSA requires indexer_kv_heads = 1")
        self.ratio = a.indexer_compress_ratio
        self.topk = a.indexer_budget // self.ratio
        self.rot, self.base = a.rotary_dim, a.rope_theta
        self.index_qk_proj = nn.Linear(a.hidden_size, (self.n_heads + 1) * self.head_dim, bias=False)
        self.q_layernorm = RMSNorm1p(self.head_dim, a.rms_norm_eps)
        self.k_layernorm = RMSNorm1p(self.head_dim, a.rms_norm_eps)

    def __call__(self, x: mx.array, offset: int, cache=None) -> mx.array | None:
        B, L, _ = x.shape
        t_kv = offset + L
        n_blocks = t_kv // self.ratio
        sparse = n_blocks > self.topk
        if cache is None and not sparse:
            return None                                     # dense regime: nothing to compute or cache
        qk = self.index_qk_proj(x)
        d = self.head_dim
        raw = qk[..., self.n_heads * d:]                    # (B, L, d) raw token keys
        if cache is not None:
            keys = raw if cache.index_keys is None else mx.concatenate([cache.index_keys, raw], axis=1)
            cache.index_keys = keys
        else:
            keys = raw
        if not sparse:
            return None
        q = self.q_layernorm(qk[..., : self.n_heads * d].reshape(B, L, self.n_heads, d)).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.rot, traditional=False, base=self.base, scale=1.0, offset=offset)  # (B, nh, L, d)
        pooled = keys[:, : n_blocks * self.ratio].reshape(B, n_blocks, self.ratio, d)
        pooled = pooled.astype(mx.float32).mean(axis=2).astype(keys.dtype)
        bk = _rope_at(self.k_layernorm(pooled), mx.arange(n_blocks) * self.ratio, self.rot, self.base)  # (B, nb, d)
        scores = mx.maximum(q.astype(mx.float32) @ bk.astype(mx.float32)[:, None].swapaxes(-1, -2), 0.0)
        scores = scores.sum(axis=1) / math.sqrt(d)          # (B, L, nb)
        qpos = offset + mx.arange(L)
        n_valid = (qpos + 1) // self.ratio                   # complete visible blocks per query
        blk = mx.arange(n_blocks)
        valid = blk[None, :] < n_valid[:, None]              # (L, nb)
        s = mx.where(valid[None], scores, -mx.inf)
        top = mx.argpartition(-s, kth=self.topk - 1, axis=-1)[..., : self.topk]
        chosen = mx.put_along_axis(mx.zeros(s.shape, dtype=mx.bool_), top, mx.array(True), axis=-1) & valid[None]
        tok = mx.arange(t_kv)
        in_blocks = tok[None, :] < (n_valid * self.ratio)[:, None]           # (L, T): token lies in a complete block
        tok_block = mx.minimum(tok // self.ratio, n_blocks - 1)
        sel = mx.where(in_blocks[None], mx.take(chosen, tok_block, axis=-1), (tok[None, :] <= qpos[:, None])[None])
        sel = sel & (tok[None, :] <= qpos[:, None])[None]
        return sel[:, None]


class AttentionCache(KVCache):
    """KV cache of a full-attention layer plus the raw indexer keys of every cached token."""

    def __init__(self) -> None:
        super().__init__()
        self.index_keys = None


class Attention(nn.Module):
    """Qwen4ExpTextAttention: gated GQA (q_proj doubled: per head [query | gate]), q/k (1+w) RMSNorm, partial RoPE,
    output * sigmoid(gate), QSA indexer mask."""

    def __init__(self, a: TextArgs):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = a.num_attention_heads, a.num_key_value_heads, a.head_dim
        self.scale = self.head_dim ** -0.5
        self.rot, self.base = a.rotary_dim, a.rope_theta
        self.q_proj = nn.Linear(a.hidden_size, self.n_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(a.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(a.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, a.hidden_size, bias=False)
        self.q_norm = RMSNorm1p(self.head_dim, a.rms_norm_eps)
        self.k_norm = RMSNorm1p(self.head_dim, a.rms_norm_eps)
        if a.indexer_n_heads is not None:
            self.indexer = Indexer(a)

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        B, L, _ = x.shape
        offset = cache.offset if cache is not None else 0
        sel = self.indexer(x, offset, cache) if "indexer" in self else None
        q, gate = mx.split(self.q_proj(x).reshape(B, L, self.n_heads, 2 * self.head_dim), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.rot, traditional=False, base=self.base, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.rot, traditional=False, base=self.base, scale=1.0, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        if sel is not None:
            mask = sel
        elif L == 1:
            mask = None
        elif offset == 0:
            mask = "causal"
        else:
            t = mx.arange(offset + L)
            mask = (t[None, :] <= (offset + mx.arange(L))[:, None])[None, None]
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o * mx.sigmoid(gate))


class MLP(nn.Module):
    def __init__(self, dims: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dims, hidden, bias=False)
        self.up_proj = nn.Linear(dims, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    """Qwen4ExpTextSparseMoeBlock: float32 softmax router, top-k, renormalized; SwiGLU experts (SwitchGLU); shared
    SwiGLU expert scaled by sigmoid(shared_expert_gate)."""

    def __init__(self, a: TextArgs):
        super().__init__()
        self.top_k, self.norm_topk_prob = a.num_experts_per_tok, a.norm_topk_prob
        self.gate = nn.Linear(a.hidden_size, a.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(a.hidden_size, a.moe_intermediate_size, a.num_experts)
        self.shared_expert = MLP(a.hidden_size, a.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(a.hidden_size, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        probs = mx.softmax(self.gate(x).astype(mx.float32), axis=-1)
        k = self.top_k
        inds = mx.argpartition(probs, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(probs, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores.astype(x.dtype)[..., None]).sum(axis=-2)
        return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


class GatedResidual(nn.Module):
    """Qwen4ExpTextGatedResidual: hc-stream mixer. Returns (u, injection weights) or, without combine (the final
    mixer), u = mean_i(sigmoid(W_up silu(W_down xn / hc))_i * xn_i), xn = per-stream (1+w) RMSNorm of x."""

    def __init__(self, a: TextArgs, combine: bool = True):
        super().__init__()
        self.hc, self.h = a.hc_count, a.hidden_size
        w = a.stream_width
        self.hc_norm = RMSNorm1p(w, a.rms_norm_eps, group=a.hidden_size)
        self.input_mix_weight_down = nn.Linear(w, a.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(a.hc_lowrank, w, bias=False)
        if combine:
            self.block_inject_weight = nn.Linear(w, self.hc, bias=False)

    def __call__(self, x: mx.array):
        xn = self.hc_norm(x)
        m = mx.sigmoid(self.input_mix_weight_up(nn.silu(self.input_mix_weight_down(xn) / self.hc)))
        shape = (*x.shape[:-1], self.hc, self.h)
        u = (m.reshape(shape) * xn.reshape(shape)).mean(axis=-2)
        if "block_inject_weight" not in self:
            return u
        return u, 2 * mx.sigmoid(self.block_inject_weight(xn) / self.hc)


def _inject(x: mx.array, f: mx.array, c: mx.array) -> mx.array:
    """x + flatten(f[..., None, :] * c[..., :, None]) (the sub-layer output added to every stream, weighted)."""
    return x + (f[..., None, :] * c[..., :, None]).reshape(x.shape)


class PLELayer(nn.Module):
    """Qwen4ExpTextPLELayer (the n-gram rows e are gathered at embedding time): k = norm_key(key_proj e) per stream,
    v = value_proj e, gate_i = signed_sqrt(<k_i, norm_query(x)_i> / sqrt(H)), g_i = sigmoid(gate_i) v,
    out = g + silu(dilated depthwise conv(norm_conv(g)))."""

    def __init__(self, a: TextArgs):
        super().__init__()
        self.hc, self.h = a.hc_count, a.hidden_size
        w = a.stream_width
        self.dilation = a.ngram_size
        self.kernel = a.ple_conv_kernel_size
        self.state_len = (self.kernel - 1) * self.dilation
        self.key_proj = nn.Linear(a.ple_embed_dim, w, bias=False)
        self.value_proj = nn.Linear(a.ple_embed_dim, a.hidden_size, bias=False)
        self.norm_key = RMSNorm1p(w, a.rms_norm_eps, group=a.hidden_size)
        self.norm_query = RMSNorm1p(w, a.rms_norm_eps, group=a.hidden_size)
        self.norm_conv = RMSNorm1p(w, a.rms_norm_eps, group=a.hidden_size)
        self.conv1d = ConvWeight(w, self.kernel)

    def __call__(self, x: mx.array, e: mx.array, cache=None) -> mx.array:
        B, S, W = x.shape
        shape = (B, S, self.hc, self.h)
        k = self.norm_key(self.key_proj(e)).reshape(shape)
        v = self.value_proj(e)
        q = self.norm_query(x).reshape(shape)
        gate = (k * q).sum(axis=-1, keepdims=True) / math.sqrt(self.h)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        g = (mx.sigmoid(gate) * v[..., None, :]).reshape(B, S, W)
        gn = self.norm_conv(g)
        prev = cache[2] if cache is not None and cache[2] is not None else mx.zeros((B, self.state_len, W),
                                                                                  dtype=gn.dtype)
        conv_in = mx.concatenate([prev.astype(gn.dtype), gn], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(conv_in[:, -self.state_len:, :])
        return g + nn.silu(depthwise_causal_conv(conv_in, self.conv1d.weight, S, self.dilation))


class DecoderLayer(nn.Module):
    """One block on the stream tensor [streams (hc*H) | n-gram rows (E per PLE layer)] (shape-preserving)."""

    def __init__(self, a: TextArgs, layer: int):
        super().__init__()
        self.layer_type = a.layer_types[layer]
        self.width = a.stream_width
        self.ple_dim = a.ple_embed_dim
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(a)
        else:
            self.self_attn = Attention(a)
        self.mlp = SparseMoeBlock(a)
        self.ple_segment = a.ple_index(layer)
        if self.ple_segment is not None:
            self.ple = PLELayer(a)
        self.attn_hyper_connection = GatedResidual(a)
        self.mlp_hyper_connection = GatedResidual(a)

    def __call__(self, h: mx.array, mask=None, cache=None) -> mx.array:
        """`mask` is accepted for the generic block API and ignored: causal / QSA masks are built here (no padding)."""
        x, extra = h[..., : self.width], h[..., self.width:]
        if "ple" in self:
            lo = self.ple_segment * self.ple_dim
            if extra.shape[-1] < lo + self.ple_dim:
                raise ValueError("PLE layer reached without n-gram rows in the stream tensor (use embed_stream)")
            x = x + self.ple(x, extra[..., lo: lo + self.ple_dim], cache)
        u, c = self.attn_hyper_connection(x)
        f = self.linear_attn(u, cache) if self.layer_type == "linear_attention" else self.self_attn(u, cache)
        x = _inject(x, f, c)
        u, c = self.mlp_hyper_connection(x)
        x = _inject(x, self.mlp(u), c)
        return mx.concatenate([x, extra], axis=-1) if extra.shape[-1] else x


class TextModel(nn.Module):
    def __init__(self, a: TextArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(a.vocab_size, a.hidden_size)
        self.layers = [DecoderLayer(a, i) for i in range(a.num_hidden_layers)]
        self.hyper_connection_mixer = GatedResidual(a, combine=False)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.text = args.text
        self.model = TextModel(args.text)
        self.lm_head = nn.Linear(args.text.hidden_size, args.text.vocab_size, bias=False)
        self._tables = _Tables()

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.layers

    # -- n-gram tables ----------------------------------------------------------
    def ple_layers(self) -> list[int]:
        return [i for i in range(self.text.num_hidden_layers) if self.text.ple_index(i) is not None]

    def attach_tables(self, get: Callable[[str], np.ndarray]) -> None:
        """Attach the n-gram shards of every PLE layer; get(name) returns the (rows, dim) array of a shard name
        (`ngram_shard_name`), memory-mapped (never read whole)."""
        for i in self.ple_layers():
            table = NgramTable([get(ngram_shard_name(i, j)) for j in range(self.text.split_ngram_parts)])
            _, _, rows = ngram_head_sizes(self.text, self.text.ple_index(i))
            if table.rows != rows or table.dim * self.text.ngram_heads != self.text.ple_embed_dim:
                raise ValueError(f"layer {i}: n-gram table {table.rows} x {table.dim}, expected {rows} x "
                                 f"{self.text.ple_embed_dim // self.text.ngram_heads}")
            self._tables.by_layer[i] = table

    def attach_checkpoint(self, model_dir: str | Path) -> None:
        """Attach the n-gram tables from the safetensors shards of an HF checkpoint directory (memory-mapped)."""
        if not self.ple_layers():
            return
        from suc.io.safetensors_store import SafetensorsStore

        store = SafetensorsStore.from_dir(model_dir)

        def get(name: str) -> np.ndarray:
            hf = _HF_PREFIX + name[len("model."):]
            return memmap_tensor(store.info(hf if hf in store else name))

        self.attach_tables(get)

    def attach_suc1(self, sm) -> None:
        """Attach the n-gram tables of a SUC1 file as zero-copy payload views of its mmap: raw shards (bf16) or
        row-affine shards (`suc repack`, suc.io.repack)."""
        if not self.ple_layers():
            return

        def get(name: str):
            d = sm.descriptor(name)
            if d["representation_type"] == "raw":
                return sm.payload(name, "weight")
            p = d["params"]
            if d["representation_type"] != "affine" or p.get("group_size") != d["shape"][1] \
                    or d["residual_type"] != "none":
                raise ValueError(f"{name}: n-gram shard stored as {d['representation_type']} (group "
                                 f"{p.get('group_size')}); expected raw or row-wise affine")
            return RowAffineShard(sm.payload(name, "indices"), sm.payload(name, "scales"), sm.payload(name, "biases"),
                                  bits=int(p["bits"]), dim=int(d["shape"][1]),
                                  scale_dtype=sm.record(name, "scales")["dtype"])

        self.attach_tables(get)

    @staticmethod
    def quant_predicate(path: str, module: nn.Module) -> bool:
        """MLX affine baseline (suc.evaluation.baselines.apply_representation): quantize exactly the tensors SUC
        encodes (Qwen4ExpAdapter eligible set: attention q/k/v/o, linear-attention qkv/z/out, shared and routed
        experts, PLE key/value) plus the embedding and the LM head; keep the routers, the shared-expert scalar
        gate, in_proj_a/b, the QSA indexer projection and every hyper-connection mixer in BF16, as `suc compress`
        does. The n-gram tables are not modules (memory-mapped, raw)."""
        return not (path.endswith(_KEPT_LINEARS) or "hyper_connection" in path)

    @staticmethod
    def is_table_tensor(name: str) -> bool:
        """Tensors of the SUC namespace that are not parameters of this model (n-gram shards, hash buffers)."""
        return ".ngram_embedding.shard_" in name or name.rsplit(".", 1)[-1] in _BUFFERS

    # -- stream API (suc.models.streaming) --------------------------------------
    def embed_stream(self, tokens: mx.array, cache=None) -> mx.array:
        """Token ids (B, T) -> stream tensor (B, T, hc*H [+ E per PLE layer]). With a cache, the n-gram context
        (previous ngram_size - 1 ids) is read from / written to the PLE layer's cache entry."""
        a = self.text
        e = self.model.embed_tokens(tokens)
        parts = [mx.concatenate([e] * a.hc_count, axis=-1)]
        ple = self.ple_layers()
        if ple:
            tok = np.array(tokens).astype(np.int64)
            for i in ple:
                if i not in self._tables.by_layer:
                    raise RuntimeError("n-gram tables not attached (attach_checkpoint / attach_suc1)")
                ctx_len = a.ngram_size - 1
                c = cache[i] if cache is not None else None
                ctx = getattr(c, "ngram_ctx", None)
                if ctx is None:
                    ctx = np.full((tok.shape[0], ctx_len), a.eos, dtype=np.int64)
                ids = ngram_ids(tok, ctx, a, a.ple_index(i))
                if c is not None:
                    c.ngram_ctx = np.concatenate([ctx, tok], axis=1)[:, -ctx_len:]
                rows = self._tables.by_layer[i].gather(ids)                 # (B, T, heads, dim)
                parts.append(rows.reshape(*tok.shape, -1).astype(e.dtype))
        return mx.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]

    def head_stream(self, h: mx.array) -> mx.array:
        return self.lm_head(self.model.hyper_connection_mixer(h[..., : self.text.stream_width]))

    def __call__(self, inputs: mx.array, cache=None, **_) -> mx.array:
        h = self.embed_stream(inputs, cache)
        for i, layer in enumerate(self.model.layers):
            h = layer(h, None, cache[i] if cache is not None else None)
        return self.head_stream(h)

    def step_hidden(self, token: int, cache) -> tuple[mx.array, list[mx.array]]:
        """(logits (vocab,), [stream after the embedding, after every block]) for one token (decode)."""
        h = self.embed_stream(mx.array([[int(token)]], dtype=mx.uint32), cache)
        hs = [h]
        for layer, c in zip(self.model.layers, cache):
            h = layer(h, None, c)
            hs.append(h)
        return self.head_stream(h)[0, -1], [x[0, -1] for x in hs]

    def make_cache(self) -> list:
        out = []
        for i, layer in enumerate(self.model.layers):
            if layer.layer_type == "linear_attention":
                out.append(ArraysCache(size=3 if "ple" in layer else 2))
            else:
                out.append(AttentionCache())
        return out

    # -- loading ----------------------------------------------------------------
    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """HF checkpoint names -> parameter names: drop vision / MTP / n-gram tables / hash buffers, strip
        `model.language_model.`, split the fused expert stacks. Lazy (slices of memory-mapped arrays)."""
        out = {}
        for k, v in weights.items():
            if k.startswith(("model.visual.", "mtp.")):
                continue
            if k.startswith(_HF_PREFIX):
                k = "model." + k[len(_HF_PREFIX):]
            if self.is_table_tensor(k):
                continue
            if k.endswith(".mlp.experts.gate_up_proj"):
                p = k[: -len("experts.gate_up_proj")]
                inter = v.shape[1] // 2
                out[p + "switch_mlp.gate_proj.weight"] = v[:, :inter]
                out[p + "switch_mlp.up_proj.weight"] = v[:, inter:]
            elif k.endswith(".mlp.experts.down_proj"):
                out[k[: -len("experts.down_proj")] + "switch_mlp.down_proj.weight"] = v
            else:
                out[k] = v
        return out


def memmap_tensor(info) -> np.ndarray:
    """Read-only numpy memmap of one safetensors tensor (BF16 as its uint16 bit pattern)."""
    np_dtype = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[info.dtype]
    return np.memmap(info.shard, dtype=np_dtype, mode="r", offset=info.offset, shape=tuple(info.shape))
