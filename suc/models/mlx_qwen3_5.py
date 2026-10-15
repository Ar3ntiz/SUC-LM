"""MLX implementation of the dense `qwen3_5` text model (Qwen/Qwen3.8-27B), text-only.

Architecture (HF `Qwen3_5ForConditionalGeneration`, mlx-lm `qwen3_5.py`): pre-norm decoder blocks
`h = x + mixer(input_layernorm(x)); out = h + mlp(post_attention_layernorm(h))`, the mixer being a Gated DeltaNet
(three of every four layers, `layer_types`) or gated GQA full attention (q_proj doubled per head [query | gate], q / k
(1 + w) RMSNorm, partial RoPE, output * sigmoid(gate)); dense SwiGLU MLP; final (1 + w) RMSNorm; untied LM head.
Vision tower and MTP head are dropped.

The mixers are the qwen4_exp ones (suc.models.mlx_qwen4_exp: Qwen3.8-Flash-Next is the same Gated DeltaNet / gated
attention without its hyper-connections, PLE, MoE and QSA indexer), so the HF-exact details carry over: q / k L2 norm
with eps on the sum (mlx-lm puts it on the mean), RMSNorm weights kept RAW and applied as float32 (1 + w) at run time
(mlx-lm's sanitize adds 1 to the BF16 weight instead, one more rounding), conv weights in the HF layout [C, 1, K].
`output_gate_type: swish` (= silu, the Qwen3.5 RMSNormGated) is accepted.

Parameter namespace (= the SUC namespace of suc.models.adapters.Qwen35Adapter.source_store): the HF names with
`model.language_model.` -> `model.`; every tensor keeps its HF shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache, KVCache

from suc.models.mlx_qwen4_exp import MLP, Attention, GatedDeltaNet, RMSNorm1p, TextArgs

_HF_PREFIX = "model.language_model."
_DROPPED = ("model.visual.", "mtp.")
_KEPT_LINEARS = ("linear_attn.in_proj_a", "linear_attn.in_proj_b")


def text_args(text: dict) -> TextArgs:
    """qwen4_exp TextArgs of a qwen3_5 text config (no experts / hyper-connections / PLE / indexer)."""
    d = dict(text)
    gate = d.get("output_gate_type")
    if gate == "swish":
        d["output_gate_type"] = "silu"
    if d.get("hidden_act", "silu") != "silu":
        raise ValueError(f"qwen3_5: unsupported hidden_act {d.get('hidden_act')!r}")
    if int(d.get("num_experts", 0) or 0):
        raise NotImplementedError("qwen3_5 MoE (Qwen3_5Moe) is not supported by this module")
    d.update(num_experts=0, hc_count=1, ple_layer_ids=None, indexer_n_heads=None)
    return TextArgs.from_dict(d)


@dataclass
class ModelArgs:
    model_type: str
    text: TextArgs
    intermediate_size: int
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params: dict) -> "ModelArgs":
        text = dict(params.get("text_config") or params)
        tie = bool(params.get("tie_word_embeddings", text.get("tie_word_embeddings", False)))
        if text.get("attn_output_gate", True) is False:
            raise NotImplementedError("qwen3_5 without the attention output gate")
        return cls(model_type=params["model_type"], text=text_args(text),
                   intermediate_size=int(text["intermediate_size"]), tie_word_embeddings=tie)

    # attributes read by generic SUC code (autotune shapes, preflight)
    @property
    def hidden_size(self) -> int:
        return self.text.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text.vocab_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text.num_hidden_layers


class DecoderLayer(nn.Module):
    def __init__(self, a: TextArgs, intermediate: int, layer: int):
        super().__init__()
        self.layer_type = a.layer_types[layer]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(a)
        else:
            self.self_attn = Attention(a)
        self.input_layernorm = RMSNorm1p(a.hidden_size, a.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm1p(a.hidden_size, a.rms_norm_eps)
        self.mlp = MLP(a.hidden_size, intermediate)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        """`mask` is accepted for the generic block API and ignored (causal, no padding; built by the mixers)."""
        u = self.input_layernorm(x)
        r = self.linear_attn(u, cache) if self.layer_type == "linear_attention" else self.self_attn(u, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class TextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        a = args.text
        self.embed_tokens = nn.Embedding(a.vocab_size, a.hidden_size)
        self.layers = [DecoderLayer(a, args.intermediate_size, i) for i in range(a.num_hidden_layers)]
        self.norm = RMSNorm1p(a.hidden_size, a.rms_norm_eps)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.text = args.text
        self.model = TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.text.hidden_size, args.text.vocab_size, bias=False)

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.layers

    def __call__(self, inputs: mx.array, cache=None, **_) -> mx.array:
        h = self.model.embed_tokens(inputs)
        for i, layer in enumerate(self.model.layers):
            h = layer(h, None, cache[i] if cache is not None else None)
        h = self.model.norm(h)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    def make_cache(self) -> list:
        return [ArraysCache(size=2) if l.layer_type == "linear_attention" else KVCache() for l in self.model.layers]

    @staticmethod
    def quant_predicate(path: str, module: nn.Module) -> bool:
        """MLX affine baseline: quantize exactly the tensors SUC encodes (Qwen35Adapter eligible set) plus the
        embedding and the LM head; in_proj_a / in_proj_b stay BF16 as in `suc compress`."""
        return not path.endswith(_KEPT_LINEARS)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """HF names -> parameter names: drop vision / MTP, strip `model.language_model.` (lazy, no copies)."""
        out = {}
        for k, v in weights.items():
            if k.startswith(_DROPPED):
                continue
            if k.startswith(_HF_PREFIX):
                k = "model." + k[len(_HF_PREFIX):]
            if k == "lm_head.weight" and self.args.tie_word_embeddings:
                continue
            out[k] = v
        return out


__all__ = ["Model", "ModelArgs", "DecoderLayer", "text_args"]
