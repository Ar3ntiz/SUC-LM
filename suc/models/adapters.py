"""Architecture adapters: map checkpoint tensor names to linear roles without loading weights (D7, D8).

An adapter is built from a Hugging Face ``config.json`` dict and classifies the tensor names of the
original (un-sanitized) HF checkpoint. Adding a family = subclass ``ArchitectureAdapter`` + ``_REGISTRY``.
"""
from __future__ import annotations

import math
import re
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, Iterable, Literal, get_args

Role = Literal["attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down",
               "expert_gate", "expert_up", "expert_down",
               "shared_expert_gate", "shared_expert_up", "shared_expert_down",
               "router", "embed", "lm_head",
               # qwen4_exp (Gated DeltaNet, QSA indexer, hyper-connections, PLE)
               "linattn_qkv", "linattn_z", "linattn_out", "linattn_a", "linattn_b", "indexer_qk",
               "hc_down", "hc_up", "hc_inject", "shared_expert_router", "ple_key", "ple_value",
               # deepseek_v4 (low-rank MQA with grouped output, compressed-KV attention, lightning indexer)
               "attn_q_a", "attn_q_b", "attn_kv", "attn_o_a", "attn_o_b", "indexer_q_b", "indexer_weights",
               "compressor_kv", "compressor_gate",
               # glm5_next (KDA linear attention, NoPE MLA with a k-pool DSA indexer)
               "kda_q", "kda_k", "kda_v", "kda_o", "kda_f_a", "kda_f_b", "kda_g_a", "kda_g_b", "kda_beta",
               "attn_kv_a", "attn_kv_b", "indexer_k"]
ROLES: tuple[Role, ...] = get_args(Role)
_ROLE_ORDER = {r: i for i, r in enumerate(ROLES)}
_NOT_ELIGIBLE: frozenset[Role] = frozenset({"router", "embed", "lm_head",   # router stays high precision (D8)
                                            "linattn_a", "linattn_b", "indexer_qk", "hc_down", "hc_up", "hc_inject",
                                            "shared_expert_router",
                                            # deepseek_v4: the indexer only selects compressed entries (a discrete
                                            # top-k the Fisher proxy cannot see); compressors run in float32
                                            "indexer_q_b", "indexer_weights", "compressor_kv", "compressor_gate",
                                            # glm5_next: KDA decay / beta / output-gate low-rank factors (tiny, they
                                            # drive the recurrence), kv_b (vendor-kept BF16, run absorbed per head),
                                            # the indexer key (discrete top-k selection)
                                            "kda_f_a", "kda_f_b", "kda_g_a", "kda_g_b", "kda_beta", "attn_kv_b",
                                            "indexer_k"})

ATTN_ROLES: tuple[Role, ...] = ("attn_q", "attn_k", "attn_v", "attn_o")
MLP_ROLES: tuple[Role, ...] = ("mlp_gate", "mlp_up", "mlp_down")
EXPERT_ROLES: tuple[Role, ...] = ("expert_gate", "expert_up", "expert_down")


@dataclass(frozen=True)
class LinearSpec:
    name: str             # checkpoint tensor name
    role: Role
    layer: int | None     # None for embed / lm_head
    expert: int | None    # None unless an expert tensor
    eligible: bool        # compressible by SUC by default


class UnsupportedArchitecture(ValueError):
    pass


_L = r"model\.layers\.(?P<layer>\d+)"
_E = r"experts\.(?P<expert>\d+)"


def _compile(table: dict[str, Role]) -> list[tuple[re.Pattern[str], Role]]:
    return [(re.compile(p), r) for p, r in table.items()]


class ArchitectureAdapter(ABC):
    """Base adapter: regex-table classification plus per-layer structural validation."""

    family: ClassVar[str]
    # full-match regex -> role; named groups `layer` / `expert` are optional
    LINEAR_PATTERNS: ClassVar[dict[str, Role]] = {
        r"model\.embed_tokens\.weight": "embed",
        r"lm_head\.weight": "lm_head",
        rf"{_L}\.self_attn\.q_proj\.weight": "attn_q",
        rf"{_L}\.self_attn\.k_proj\.weight": "attn_k",
        rf"{_L}\.self_attn\.v_proj\.weight": "attn_v",
        rf"{_L}\.self_attn\.o_proj\.weight": "attn_o",
        rf"{_L}\.mlp\.gate_proj\.weight": "mlp_gate",
        rf"{_L}\.mlp\.up_proj\.weight": "mlp_up",
        rf"{_L}\.mlp\.down_proj\.weight": "mlp_down",
    }
    # known non-linear ".weight" tensors (norms)
    NONLINEAR_PATTERNS: ClassVar[tuple[str, ...]] = (
        r"model\.norm\.weight",
        rf"{_L}\.(input_layernorm|post_attention_layernorm)\.weight",
        rf"{_L}\.self_attn\.(q_norm|k_norm)\.weight",
    )

    # RMSNorm weights by position ({layer} substituted): "attn" = the norm whose output the attention projections
    # read, "mlp" = the one read by the MLP / experts / router, "final" = the one before the LM head
    # (suc.models.rotation_folding folds them into the reading linears).
    NORMS: ClassVar[dict[str, str]] = {
        "attn": "model.layers.{layer}.input_layernorm.weight",
        "mlp": "model.layers.{layer}.post_attention_layernorm.weight",
        "final": "model.norm.weight",
    }

    def __init__(self, config: dict):
        self.config = config
        self._linear_re = _compile(self.LINEAR_PATTERNS)
        self._nonlinear_re = [re.compile(p) for p in self.NONLINEAR_PATTERNS]

    # -- structure -------------------------------------------------------------
    @property
    def text_config(self) -> dict:
        """The language-model config (multimodal checkpoints nest it under "text_config")."""
        return self.config

    @property
    def num_layers(self) -> int:
        return int(self.text_config["num_hidden_layers"])

    @property
    def residual_width(self) -> int:
        """Width of the hidden state passed between transformer blocks (hidden_size; wider with hyper-connections)."""
        return int(self.text_config.get("hidden_size", 0))

    def source_store(self, store):
        """The SafetensorsStore in the SUC tensor namespace (identity unless the family renames / splits tensors)."""
        return store

    def table_tensors(self, names: Iterable[str]) -> set[str]:
        """Row-gather lookup tables that stay memory-mapped (never resident as block weights)."""
        return set()

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """KV-cache bytes one token adds at inference (all layers; plain K and V of every KV head)."""
        c = self.text_config
        kv = int(c.get("num_key_value_heads") or c.get("num_attention_heads", 1))
        hd = int(c.get("head_dim") or int(c["hidden_size"]) // int(c.get("num_attention_heads", 1)))
        return 2 * self.num_layers * kv * hd * dtype_bytes

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def num_experts(self) -> int:
        return 0

    @property
    def tied_embeddings(self) -> bool:
        return bool(self.config.get("tie_word_embeddings", False))

    def norm_name(self, position: str, layer: int | None = None) -> str:
        """Checkpoint name of the RMSNorm weight at `position` ("attn" / "mlp" of `layer`, or "final")."""
        if position not in self.NORMS:
            raise KeyError(f"unknown norm position {position!r}; expected one of {sorted(self.NORMS)}")
        if position != "final" and layer is None:
            raise ValueError(f"norm position {position!r} needs a layer")
        return self.NORMS[position].format(layer=layer)

    def expected_roles(self, layer: int) -> tuple[Role, ...]:
        """Non-expert roles each layer must contain."""
        return ATTN_ROLES + MLP_ROLES

    def expected_experts(self, layer: int) -> int:
        """Number of routed experts expected in `layer` (0 = dense MLP)."""
        return 0

    # -- classification --------------------------------------------------------
    def _classify(self, name: str) -> LinearSpec | None:
        for rx, role in self._linear_re:
            m = rx.fullmatch(name)
            if m:
                g = m.groupdict()
                layer = int(g["layer"]) if g.get("layer") is not None else None
                expert = int(g["expert"]) if g.get("expert") is not None else None
                return LinearSpec(name, role, layer, expert, role not in _NOT_ELIGIBLE)
        return None

    @staticmethod
    def _key(s: LinearSpec) -> tuple:
        layer = -1 if s.role == "embed" else (math.inf if s.layer is None else s.layer)
        return (layer, -1 if s.expert is None else s.expert, _ROLE_ORDER[s.role], s.name)

    def linears(self, tensor_names: Iterable[str]) -> list[LinearSpec]:
        specs = (self._classify(n) for n in set(tensor_names))
        return sorted((s for s in specs if s is not None), key=self._key)

    def unclassified(self, tensor_names: Iterable[str]) -> list[str]:
        return sorted(n for n in set(tensor_names) if n.endswith(".weight") and self._classify(n) is None
                      and not any(rx.fullmatch(n) for rx in self._nonlinear_re))

    def validate(self, tensor_names: Iterable[str]) -> None:
        names = list(tensor_names)
        errors: list[str] = []
        if unknown := self.unclassified(names):
            errors.append(f"unclassified tensors: {unknown[:5]}{' ...' if len(unknown) > 5 else ''}")
        specs = self.linears(names)
        roles: dict[int, set[Role]] = defaultdict(set)
        experts: dict[int, dict[int, set[Role]]] = defaultdict(lambda: defaultdict(set))
        globals_: set[Role] = set()
        for s in specs:
            if s.layer is None:
                globals_.add(s.role)
            elif s.layer >= self.num_layers:
                errors.append(f"{s.name}: layer {s.layer} >= num_layers {self.num_layers}")
            elif s.expert is None:
                roles[s.layer].add(s.role)
            else:
                experts[s.layer][s.expert].add(s.role)
        if "embed" not in globals_:
            errors.append("missing embed")
        if not self.tied_embeddings and "lm_head" not in globals_:
            errors.append("missing lm_head (tie_word_embeddings is false)")
        for i in range(self.num_layers):
            want = set(self.expected_roles(i))
            if missing := want - roles[i]:
                errors.append(f"layer {i}: missing {sorted(missing, key=_ROLE_ORDER.get)}")
            if extra := roles[i] - want:
                errors.append(f"layer {i}: unexpected {sorted(extra, key=_ROLE_ORDER.get)}")
            n_exp = self.expected_experts(i)
            got = experts[i]
            if sorted(got) != list(range(n_exp)):
                errors.append(f"layer {i}: expert count {len(got)} (indices {min(got, default=None)}.."
                              f"{max(got, default=None)}) != expected {n_exp}")
            for e, rs in sorted(got.items()):
                if rs != set(EXPERT_ROLES):
                    errors.append(f"layer {i} expert {e}: missing {sorted(set(EXPERT_ROLES) - rs)}")
        if errors:
            raise ValueError(f"{self.family} checkpoint invalid:\n  " + "\n  ".join(errors))


class Qwen3Adapter(ArchitectureAdapter):
    family = "qwen3"


class Qwen3MoeAdapter(ArchitectureAdapter):
    """Qwen3-MoE: routed experts, no shared expert; some layers may be dense
    (`mlp_only_layers`, `decoder_sparse_step`), mirroring mlx_lm.models.qwen3_moe."""

    family = "qwen3_moe"
    LINEAR_PATTERNS = {
        **ArchitectureAdapter.LINEAR_PATTERNS,
        rf"{_L}\.mlp\.gate\.weight": "router",
        rf"{_L}\.mlp\.{_E}\.gate_proj\.weight": "expert_gate",
        rf"{_L}\.mlp\.{_E}\.up_proj\.weight": "expert_up",
        rf"{_L}\.mlp\.{_E}\.down_proj\.weight": "expert_down",
    }

    @property
    def num_experts(self) -> int:
        return int(self.config.get("num_experts", 0))

    def is_sparse_layer(self, layer: int) -> bool:
        step = int(self.config.get("decoder_sparse_step", 1) or 1)
        return (layer not in self.config.get("mlp_only_layers", [])
                and self.num_experts > 0 and (layer + 1) % step == 0)

    def expected_roles(self, layer: int) -> tuple[Role, ...]:
        return ATTN_ROLES + (("router",) if self.is_sparse_layer(layer) else MLP_ROLES)

    def expected_experts(self, layer: int) -> int:
        return self.num_experts if self.is_sparse_layer(layer) else 0


_Q4_HF = "model.language_model."
_Q4_DROPPED = ("model.visual.", "mtp.")          # vision tower, MTP head: text-only LM (docs/arch/qwen4_exp.md §6)
_Q4_BUFFERS = ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")
_Q4_TABLE = re.compile(rf"{_L}\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight")


class Qwen4ExpAdapter(ArchitectureAdapter):
    """qwen4_exp (Qwen/Qwen3.8-Flash-Next), docs/arch/qwen4_exp.md.

    Two namespaces. The HF checkpoint ("model.language_model.layers.N...", fused 3-D expert stacks
    "mlp.experts.gate_up_proj" [E, 2I, H] / "mlp.experts.down_proj" [E, H, I], vision tower, MTP head) and the SUC
    namespace used everywhere downstream (= the MLX parameter names of suc.models.mlx_qwen4_exp, plus virtual
    per-expert names): prefix "model.", one tensor per expert and projection "…mlp.experts.{e}.{gate,up,down}_proj
    .weight" (gate = rows [0, I) of gate_up_proj[e], up = rows [I, 2I)), vision / MTP dropped. Every expert tensor is a
    CONTIGUOUS byte range of the fused stack, so source_store() only re-points TensorInfo offsets (no copy) and
    store.load() / the SUC1 raw path / model_size work unchanged. validate() accepts either namespace.

    Eligible: attention q/k/v/o, linear-attention qkv/z/out, shared expert, routed experts, PLE key/value. Kept: router,
    shared-expert scalar gate, in_proj_a/b, indexer, all hyper-connection mixers, norms, conv weights, A_log / dt_bias,
    n-gram tables (row-gather, memory-mapped) and their I64 hash buffers.
    """

    family = "qwen4_exp"
    LINEAR_PATTERNS = {
        r"model\.embed_tokens\.weight": "embed",
        r"lm_head\.weight": "lm_head",
        rf"{_L}\.self_attn\.q_proj\.weight": "attn_q",
        rf"{_L}\.self_attn\.k_proj\.weight": "attn_k",
        rf"{_L}\.self_attn\.v_proj\.weight": "attn_v",
        rf"{_L}\.self_attn\.o_proj\.weight": "attn_o",
        rf"{_L}\.self_attn\.indexer\.index_qk_proj\.weight": "indexer_qk",
        rf"{_L}\.linear_attn\.in_proj_qkv\.weight": "linattn_qkv",
        rf"{_L}\.linear_attn\.in_proj_z\.weight": "linattn_z",
        rf"{_L}\.linear_attn\.out_proj\.weight": "linattn_out",
        rf"{_L}\.linear_attn\.in_proj_a\.weight": "linattn_a",
        rf"{_L}\.linear_attn\.in_proj_b\.weight": "linattn_b",
        rf"{_L}\.(?:attn|mlp)_hyper_connection\.input_mix_weight_down\.weight": "hc_down",
        rf"{_L}\.(?:attn|mlp)_hyper_connection\.input_mix_weight_up\.weight": "hc_up",
        rf"{_L}\.(?:attn|mlp)_hyper_connection\.block_inject_weight\.weight": "hc_inject",
        r"model\.hyper_connection_mixer\.input_mix_weight_down\.weight": "hc_down",
        r"model\.hyper_connection_mixer\.input_mix_weight_up\.weight": "hc_up",
        rf"{_L}\.mlp\.gate\.weight": "router",
        rf"{_L}\.mlp\.shared_expert_gate\.weight": "shared_expert_router",
        rf"{_L}\.mlp\.shared_expert\.gate_proj\.weight": "shared_expert_gate",
        rf"{_L}\.mlp\.shared_expert\.up_proj\.weight": "shared_expert_up",
        rf"{_L}\.mlp\.shared_expert\.down_proj\.weight": "shared_expert_down",
        rf"{_L}\.mlp\.{_E}\.gate_proj\.weight": "expert_gate",
        rf"{_L}\.mlp\.{_E}\.up_proj\.weight": "expert_up",
        rf"{_L}\.mlp\.{_E}\.down_proj\.weight": "expert_down",
        rf"{_L}\.ple\.key_proj\.weight": "ple_key",
        rf"{_L}\.ple\.value_proj\.weight": "ple_value",
    }
    NONLINEAR_PATTERNS = (
        r"model\.hyper_connection_mixer\.hc_norm\.weight",
        rf"{_L}\.(?:attn|mlp)_hyper_connection\.hc_norm\.weight",
        rf"{_L}\.linear_attn\.(?:A_log|dt_bias|conv1d\.weight|norm\.weight)",
        rf"{_L}\.self_attn\.(?:q_norm|k_norm)\.weight",
        rf"{_L}\.self_attn\.indexer\.(?:q_layernorm|k_layernorm)\.weight",
        rf"{_L}\.ple\.(?:conv1d|norm_key|norm_query|norm_conv)\.weight",
        _Q4_TABLE.pattern,
        rf"{_L}\.ple\.ple_embedding\.(?:{'|'.join(_Q4_BUFFERS)})",
    )
    NORMS: ClassVar[dict[str, str]] = {}   # no foldable RMSNorm (gated stream mixer): rotation folding does not apply

    @property
    def text_config(self) -> dict:
        return self.config.get("text_config", self.config)

    @property
    def tied_embeddings(self) -> bool:
        return bool(self.config.get("tie_word_embeddings", self.text_config.get("tie_word_embeddings", False)))

    @property
    def num_experts(self) -> int:
        return int(self.text_config.get("num_experts", 0))

    @property
    def residual_width(self) -> int:
        """hc streams + the n-gram rows of every PLE layer (suc.models.mlx_qwen4_exp stream layout)."""
        t = self.text_config
        h = int(t["hidden_size"])
        return int(t.get("hc_count", 4)) * h + len(t.get("ple_layer_ids") or []) * int(t.get("ple_embed_dim") or h)

    def layer_type(self, layer: int) -> str:
        t = self.text_config
        types = t.get("layer_types") or ["linear_attention" if (i + 1) % int(t.get("full_attention_interval", 4))
                                         else "full_attention" for i in range(self.num_layers)]
        return "linear_attention" if types[layer] == "linear_attention" else "full_attention"

    def has_ple(self, layer: int) -> bool:
        return layer + 1 in (self.text_config.get("ple_layer_ids") or [])

    def expected_roles(self, layer: int) -> tuple[Role, ...]:
        mixer = ("linattn_qkv", "linattn_z", "linattn_out", "linattn_a", "linattn_b") \
            if self.layer_type(layer) == "linear_attention" else ATTN_ROLES + ("indexer_qk",)
        ple = ("ple_key", "ple_value") if self.has_ple(layer) else ()
        return mixer + ("hc_down", "hc_up", "hc_inject", "router", "shared_expert_router", "shared_expert_gate",
                        "shared_expert_up", "shared_expert_down") + ple

    def expected_experts(self, layer: int) -> int:
        return self.num_experts

    # -- namespaces --------------------------------------------------------------
    @staticmethod
    def is_dropped(hf_name: str) -> bool:
        return hf_name.startswith(_Q4_DROPPED)

    def to_suc_names(self, hf_name: str, shape: tuple[int, ...] | None = None) -> list[str]:
        """SUC names of one HF tensor ([] = dropped; unknown names pass through unchanged so validate reports them).
        Expert stacks need the stack length (shape[0]; default num_experts)."""
        if self.is_dropped(hf_name):
            return []
        n = "model." + hf_name[len(_Q4_HF):] if hf_name.startswith(_Q4_HF) else hf_name
        e_count = int(shape[0]) if shape is not None else self.num_experts
        for fused, projs in ((".mlp.experts.gate_up_proj", ("gate_proj", "up_proj")),
                             (".mlp.experts.down_proj", ("down_proj",))):
            if n.endswith(fused):
                p = n[: -len(fused)]
                return [f"{p}.mlp.experts.{e}.{proj}.weight" for e in range(e_count) for proj in projs]
        return [n]

    def source_store(self, store):
        """View of `store` (HF names) in the SUC namespace: renamed infos, expert slices as offset ranges."""
        from suc.io.safetensors_store import SafetensorsStore, TensorInfo

        infos: dict[str, TensorInfo] = {}
        for name in store.names():
            info = store.info(name)
            if name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")):
                n_exp, rows, cols = info.shape
                per_expert = info.nbytes // n_exp
                parts = 2 if name.endswith("gate_up_proj") else 1
                out_rows = rows // parts
                names = self.to_suc_names(name, info.shape)
                for k, vname in enumerate(names):
                    e, part = divmod(k, parts)
                    nb = per_expert // parts
                    infos[vname] = TensorInfo(vname, info.dtype, (out_rows, cols), info.shard,
                                              info.offset + e * per_expert + part * nb, nb)
                continue
            for vname in self.to_suc_names(name):
                infos[vname] = TensorInfo(vname, info.dtype, info.shape, info.shard, info.offset, info.nbytes)
        return SafetensorsStore(infos)

    def _suc_namespace(self, names: Iterable[str]) -> list[str]:
        names = list(names)
        if not any(n.startswith(_Q4_HF) or self.is_dropped(n) for n in names):
            return names
        return [v for n in names for v in self.to_suc_names(n)]

    def table_tensors(self, names: Iterable[str]) -> set[str]:
        return {n for n in self._suc_namespace(names) if _Q4_TABLE.fullmatch(n)}

    def unclassified(self, tensor_names: Iterable[str]) -> list[str]:
        # every tensor, not only "*.weight" (A_log, dt_bias and the I64 buffers have no suffix)
        return sorted(n for n in set(self._suc_namespace(tensor_names)) if self._classify(n) is None
                      and not any(rx.fullmatch(n) for rx in self._nonlinear_re))

    def linears(self, tensor_names: Iterable[str]) -> list[LinearSpec]:
        return super().linears(self._suc_namespace(tensor_names))

    def validate(self, tensor_names: Iterable[str]) -> None:
        names = self._suc_namespace(tensor_names)
        super().validate(names)
        t = self.text_config
        tables = self.table_tensors(names)
        want = len(t.get("ple_layer_ids") or []) * int(t.get("split_ngram_parts", 0))
        if len(tables) != want:
            raise ValueError(f"{self.family} checkpoint invalid:\n  n-gram table shards {len(tables)} != expected {want}")


class Qwen35Adapter(Qwen4ExpAdapter):
    """Dense qwen3_5 (Qwen/Qwen3.8-27B; HF Qwen3_5ForConditionalGeneration), suc.models.mlx_qwen3_5.

    Same two namespaces as qwen4_exp (HF "model.language_model.layers.N..." with vision tower and MTP head; SUC
    "model.layers.N...", vision / MTP dropped), no fused tensors. Layers: Gated DeltaNet (in_proj_qkv / z / a / b,
    out_proj) or gated full attention (q doubled with the output gate, k, v, o), pre-norm RMSNorms, dense SwiGLU MLP.
    Eligible: attention q/k/v/o, linear-attention qkv/z/out, MLP gate/up/down. Kept: in_proj_a/b (per-head decay and
    beta, 48 rows), norms, conv weights, A_log / dt_bias, embedding / LM head (codec chosen by the plan)."""

    family = "qwen3_5"
    LINEAR_PATTERNS = {
        r"model\.embed_tokens\.weight": "embed",
        r"lm_head\.weight": "lm_head",
        rf"{_L}\.self_attn\.q_proj\.weight": "attn_q",
        rf"{_L}\.self_attn\.k_proj\.weight": "attn_k",
        rf"{_L}\.self_attn\.v_proj\.weight": "attn_v",
        rf"{_L}\.self_attn\.o_proj\.weight": "attn_o",
        rf"{_L}\.linear_attn\.in_proj_qkv\.weight": "linattn_qkv",
        rf"{_L}\.linear_attn\.in_proj_z\.weight": "linattn_z",
        rf"{_L}\.linear_attn\.out_proj\.weight": "linattn_out",
        rf"{_L}\.linear_attn\.in_proj_a\.weight": "linattn_a",
        rf"{_L}\.linear_attn\.in_proj_b\.weight": "linattn_b",
        rf"{_L}\.mlp\.gate_proj\.weight": "mlp_gate",
        rf"{_L}\.mlp\.up_proj\.weight": "mlp_up",
        rf"{_L}\.mlp\.down_proj\.weight": "mlp_down",
    }
    NONLINEAR_PATTERNS = (
        r"model\.norm\.weight",
        rf"{_L}\.(?:input_layernorm|post_attention_layernorm)\.weight",
        rf"{_L}\.linear_attn\.(?:A_log|dt_bias|conv1d\.weight|norm\.weight)",
        rf"{_L}\.self_attn\.(?:q_norm|k_norm)\.weight",
    )
    NORMS: ClassVar[dict[str, str]] = {}   # (1 + w) RMSNorms: rotation folding is not implemented for them

    @property
    def num_experts(self) -> int:
        return 0

    @property
    def residual_width(self) -> int:
        return int(self.text_config["hidden_size"])

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """K / V of the full-attention layers only (the linear-attention state is per sequence, not per token)."""
        c = self.text_config
        full = sum(self.layer_type(i) == "full_attention" for i in range(self.num_layers))
        return 2 * full * int(c["num_key_value_heads"]) * int(c["head_dim"]) * dtype_bytes

    def has_ple(self, layer: int) -> bool:
        return False

    def expected_roles(self, layer: int) -> tuple[Role, ...]:
        mixer = ("linattn_qkv", "linattn_z", "linattn_out", "linattn_a", "linattn_b") \
            if self.layer_type(layer) == "linear_attention" else ATTN_ROLES
        return mixer + MLP_ROLES

    def expected_experts(self, layer: int) -> int:
        return 0

    def table_tensors(self, names: Iterable[str]) -> set[str]:
        return set()


_DS4_SCALE = re.compile(r"^(.*)\.scale$")
_DS4_EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.weight$")
_DS4_PROJ = {"1": "gate_proj", "3": "up_proj", "2": "down_proj"}


class DeepseekV4Adapter(ArchitectureAdapter):
    """deepseek_v4 (deepseek-ai/DeepSeek-V4-Flash-0731), docs/arch/deepseek_v4.md.

    Two namespaces. The checkpoint uses the reference-implementation names (inference/model.py: "layers.N.attn.wq_a
    .weight" + "layers.N.attn.wq_a.scale", "layers.N.ffn.experts.E.w1.weight", "hc_head_fn", DSpark stages "mtp.*") and
    stores the attention / shared-expert / indexer linears as FP8 E4M3 with 128 x 128 E8M0 block scales and the routed
    experts as MXFP4 (packed I8 + per-32 E8M0 scales). The SUC namespace (= the MLX parameter names of
    suc.models.mlx_deepseek_v4, plus virtual per-expert names) is "model.layers.N.self_attn.wq_a.weight",
    "model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight" (w1 / w3 / w2), "model.layers.N.hc_attn.fn", ...;
    source_store() maps every logical tensor there and attaches its QuantSource, so store.load() returns the exact BF16
    dequantization (the D8 exception: the full-precision reference IS the dequantized checkpoint). Scale companions
    are consumed by their weight; mtp.* (DSpark speculative head) is dropped. validate() accepts either namespace.

    Eligible: wq_a, wq_b, wkv, wo_a (one [o_groups * o_lora_rank, in] tensor whose row-group g reads input group g),
    wo_b, shared and routed experts. Kept: router weight / e_score_correction_bias / tid2eid (hash layers), indexer wq_b
    and weights_proj (they only drive the top-k selection), compressor and indexer-compressor wkv / wgate (float32 in
    the reference), every norm, attention sinks, ape, hyper-connection parameters (F32).
    """

    family = "deepseek_v4"
    _P = r"model\.layers\.(?P<layer>\d+)"
    LINEAR_PATTERNS = {
        r"model\.embed_tokens\.weight": "embed",
        r"lm_head\.weight": "lm_head",
        rf"{_P}\.self_attn\.wq_a\.weight": "attn_q_a",
        rf"{_P}\.self_attn\.wq_b\.weight": "attn_q_b",
        rf"{_P}\.self_attn\.wkv\.weight": "attn_kv",
        rf"{_P}\.self_attn\.wo_a\.weight": "attn_o_a",
        rf"{_P}\.self_attn\.wo_b\.weight": "attn_o_b",
        rf"{_P}\.self_attn\.indexer\.wq_b\.weight": "indexer_q_b",
        rf"{_P}\.self_attn\.indexer\.weights_proj\.weight": "indexer_weights",
        rf"{_P}\.self_attn\.(?:indexer\.)?compressor\.wkv\.weight": "compressor_kv",
        rf"{_P}\.self_attn\.(?:indexer\.)?compressor\.wgate\.weight": "compressor_gate",
        rf"{_P}\.mlp\.gate\.weight": "router",
        rf"{_P}\.mlp\.shared_experts\.gate_proj\.weight": "shared_expert_gate",
        rf"{_P}\.mlp\.shared_experts\.up_proj\.weight": "shared_expert_up",
        rf"{_P}\.mlp\.shared_experts\.down_proj\.weight": "shared_expert_down",
        rf"{_P}\.mlp\.{_E}\.gate_proj\.weight": "expert_gate",
        rf"{_P}\.mlp\.{_E}\.up_proj\.weight": "expert_up",
        rf"{_P}\.mlp\.{_E}\.down_proj\.weight": "expert_down",
    }
    NONLINEAR_PATTERNS = (
        r"model\.norm\.weight",
        r"model\.hc_head\.(?:fn|base|scale)",
        rf"{_P}\.(?:attn_norm|ffn_norm)\.weight",
        rf"{_P}\.hc_(?:attn|ffn)\.(?:fn|base|scale)",
        rf"{_P}\.self_attn\.(?:q_norm|kv_norm)\.weight",
        rf"{_P}\.self_attn\.attn_sink",
        rf"{_P}\.self_attn\.(?:indexer\.)?compressor\.(?:ape|norm\.weight)",
        rf"{_P}\.mlp\.gate\.(?:e_score_correction_bias|tid2eid)",
    )
    NORMS: ClassVar[dict[str, str]] = {}   # hyper-connection streams: no residual-stream rotation folding (doc §9.11)

    # -- structure -------------------------------------------------------------
    @property
    def num_experts(self) -> int:
        return int(self.config.get("n_routed_experts", 0))

    @property
    def residual_width(self) -> int:
        """hc_mult streams of hidden_size + 3 token-id channels (hash routing; suc.models.mlx_deepseek_v4)."""
        c = self.config
        return int(c.get("hc_mult", 4)) * int(c["hidden_size"]) + (3 if int(c.get("num_hash_layers", 0)) else 0)

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """Asymptotic KV bytes per token: one K = V vector of head_dim per compressed entry (CSA: 1 per 4 tokens + the
        indexer's index_head_dim key; HCA: 1 per 128) — the sliding window (window x head_dim per layer) is fixed.
        For DeepSeek-V4-Flash: (21 x (512 + 128) / 4 + 20 x 512 / 128) x 2 B = 6.9 KB per token."""
        c = self.config
        hd, ihd = int(c["head_dim"]), int(c.get("index_head_dim", 0))
        per = 0.0
        for i in range(self.num_layers):
            r = self.compress_ratio(i)
            if r:
                per += (hd + (ihd if r == 4 else 0)) / r
        return int(math.ceil(per * dtype_bytes))

    def kv_fixed_bytes(self, dtype_bytes: int = 2) -> int:
        """Sliding-window keys of every layer (independent of the context length)."""
        c = self.config
        return self.num_layers * int(c.get("sliding_window", 0)) * int(c["head_dim"]) * dtype_bytes

    def compress_ratio(self, layer: int) -> int:
        return int((self.config.get("compress_ratios") or [0] * self.num_layers)[layer])

    def is_hash_layer(self, layer: int) -> bool:
        return layer < int(self.config.get("num_hash_layers", 0))

    def expected_roles(self, layer: int) -> tuple[Role, ...]:
        r = self.compress_ratio(layer)
        attn = ("attn_q_a", "attn_q_b", "attn_kv", "attn_o_a", "attn_o_b")
        comp = ("compressor_kv", "compressor_gate") if r else ()
        idx = ("indexer_q_b", "indexer_weights") if r == 4 else ()
        return attn + comp + idx + ("router", "shared_expert_gate", "shared_expert_up", "shared_expert_down")

    def expected_experts(self, layer: int) -> int:
        return self.num_experts

    # -- namespaces --------------------------------------------------------------
    @staticmethod
    def is_dropped(name: str) -> bool:
        return name.startswith("mtp.")

    @staticmethod
    def to_suc_name(name: str) -> str | None:
        """SUC name of a checkpoint (reference-namespace) tensor; None for dropped tensors and scale companions.
        Names already in the SUC namespace (and unknown names) pass through unchanged."""
        if name.startswith("model.") or name == "lm_head.weight":
            return name                       # SUC namespace (its hc_*.scale are hyper-connection parameters)
        if name.startswith("mtp.") or _DS4_SCALE.match(name):
            return None
        if (m := _DS4_EXPERT.match(name)) is not None:
            return f"model.layers.{m.group(1)}.mlp.experts.{m.group(2)}.{_DS4_PROJ[m.group(3)]}.weight"
        top = {"embed.weight": "model.embed_tokens.weight", "head.weight": "lm_head.weight",
               "norm.weight": "model.norm.weight", "hc_head_fn": "model.hc_head.fn",
               "hc_head_base": "model.hc_head.base", "hc_head_scale": "model.hc_head.scale"}
        if name in top:
            return top[name]
        m = re.match(r"^layers\.(\d+)\.(.*)$", name)
        if m is None:
            return name
        p, rest = f"model.layers.{m.group(1)}.", m.group(2)
        if (h := re.match(r"^hc_(attn|ffn)_(fn|base|scale)$", rest)) is not None:
            return f"{p}hc_{h.group(1)}.{h.group(2)}"
        if rest.startswith("attn."):
            return p + "self_attn." + rest[len("attn."):]
        if (s := re.match(r"^ffn\.shared_experts\.w([123])\.weight$", rest)) is not None:
            return f"{p}mlp.shared_experts.{_DS4_PROJ[s.group(1)]}.weight"
        if rest == "ffn.gate.bias":
            return p + "mlp.gate.e_score_correction_bias"
        if rest.startswith("ffn."):
            return p + "mlp." + rest[len("ffn."):]
        return p + rest

    def _suc_namespace(self, names: Iterable[str]) -> list[str]:
        return [v for n in names if (v := self.to_suc_name(n)) is not None]

    def source_store(self, store):
        """View of `store` (checkpoint names) in the SUC namespace: renamed infos; FP8 / MXFP4 weights become logical
        BF16 tensors dequantized exactly on load (QuantSource with their scale companion)."""
        from suc.io.safetensors_store import QuantSource, SafetensorsStore, TensorInfo

        infos: dict[str, TensorInfo] = {}
        quant: dict[str, QuantSource] = {}
        for name in store.names():
            v = self.to_suc_name(name)
            if v is None:
                continue
            info = store.info(name)
            kind = {"F8_E4M3": "fp8_block", "I8": "mxfp4"}.get(info.dtype)
            if kind is None:
                infos[v] = TensorInfo(v, info.dtype, info.shape, info.shard, info.offset, info.nbytes)
                continue
            sname = name[: -len(".weight")] + ".scale"
            if not name.endswith(".weight") or sname not in store:
                raise ValueError(f"{name}: {info.dtype} tensor without its scale companion {sname}")
            shape = info.shape if kind == "fp8_block" else (info.shape[0], 2 * info.shape[1])
            infos[v] = TensorInfo(v, "BF16", tuple(shape), info.shard, info.offset, 2 * shape[0] * shape[1])
            quant[v] = QuantSource(kind, info, store.info(sname))
        return SafetensorsStore(infos, quant)

    def unclassified(self, tensor_names: Iterable[str]) -> list[str]:
        # every tensor, not only "*.weight" (hc parameters, attn_sink, ape, gate bias / tid2eid have no suffix)
        return sorted(n for n in set(self._suc_namespace(tensor_names)) if self._classify(n) is None
                      and not any(rx.fullmatch(n) for rx in self._nonlinear_re))

    def linears(self, tensor_names: Iterable[str]) -> list[LinearSpec]:
        return super().linears(self._suc_namespace(tensor_names))

    def validate(self, tensor_names: Iterable[str]) -> None:
        raw = list(tensor_names)
        names = self._suc_namespace(raw)
        super().validate(names)
        have = set(names)
        errors = []
        for i in range(self.num_layers):
            p = f"model.layers.{i}.mlp.gate."
            want, other = ("tid2eid", "e_score_correction_bias") if self.is_hash_layer(i) else \
                ("e_score_correction_bias", "tid2eid")
            if p + want not in have:
                errors.append(f"layer {i}: missing mlp.gate.{want}")
            if p + other in have:
                errors.append(f"layer {i}: unexpected mlp.gate.{other}")
            r = self.compress_ratio(i)
            for k in ("ape", "norm.weight"):
                if bool(r) != (f"model.layers.{i}.self_attn.compressor.{k}" in have):
                    errors.append(f"layer {i}: compressor.{k} {'missing' if r else 'unexpected'} (ratio {r})")
        if not any(n.startswith("model.") for n in raw):     # checkpoint namespace: every expert has its scale
            raw_set = set(raw)
            if missing := [n for n in raw if _DS4_EXPERT.match(n) and n[:-7] + ".scale" not in raw_set]:
                errors.append(f"{len(missing)} expert weight(s) without a scale companion, e.g. {missing[0]}")
        if errors:
            raise ValueError(f"{self.family} checkpoint invalid:\n  " + "\n  ".join(errors))


_GLM_HF = "model.language_model."
_GLM_DSA_TYPES = ("deepseek_sparse_attention", "indexed_attention", "full_attention")


class Glm5NextAdapter(ArchitectureAdapter):
    """glm5_next (zai-org/GLM-5.3-Flash), docs/arch/glm5_next.md.

    Two namespaces. The checkpoint ("model.language_model.layers.N...", FP8 E4M3 weights with an F32 128 x 128 block
    multiplier "<name>.weight_scale_inv", vision tower "model.visual.*", MTP layer N = num_hidden_layers) and the SUC
    namespace (= the MLX parameter names of suc.models.mlx_glm5_next): prefix "model.", everything else unchanged, MTP
    layer / vision tower / scale companions dropped. source_store() attaches a QuantSource("fp8_block_f32") to every FP8
    weight, so store.load() returns its BF16 dequantization (the reference implementations' own, suc.io
    .quantized_source). validate() accepts either namespace.

    Eligible: KDA q/k/v/o (the vendor kept them BF16: give them priority in the allocator), MLA q_a / q_b / kv_a / o,
    dense MLP (layers < first_k_dense_replace), shared and routed experts. Kept: router + bias, KDA f_a/f_b/g_a/g_b/
    b_proj, A_log, dt_bias, the q/k/v short convolutions, o_norm, MLA kv_b (BF16 in the source; the MLX model runs it
    absorbed per head), the whole indexer (wq_b, wk, k_norm, weights_proj, k-pool gate / ape), hyper-connection
    parameters (F32), every norm.
    """

    family = "glm5_next"
    LINEAR_PATTERNS = {
        r"model\.embed_tokens\.weight": "embed",
        r"lm_head\.weight": "lm_head",
        rf"{_L}\.self_attn\.q_proj\.weight": "kda_q",
        rf"{_L}\.self_attn\.k_proj\.weight": "kda_k",
        rf"{_L}\.self_attn\.v_proj\.weight": "kda_v",
        rf"{_L}\.self_attn\.o_proj\.weight": "attn_o",            # kda_o on KDA layers (_classify)
        rf"{_L}\.self_attn\.f_a_proj\.weight": "kda_f_a",
        rf"{_L}\.self_attn\.f_b_proj\.weight": "kda_f_b",
        rf"{_L}\.self_attn\.g_a_proj\.weight": "kda_g_a",
        rf"{_L}\.self_attn\.g_b_proj\.weight": "kda_g_b",
        rf"{_L}\.self_attn\.b_proj\.weight": "kda_beta",
        rf"{_L}\.self_attn\.q_a_proj\.weight": "attn_q_a",
        rf"{_L}\.self_attn\.q_b_proj\.weight": "attn_q_b",
        rf"{_L}\.self_attn\.kv_a_proj_with_mqa\.weight": "attn_kv_a",
        rf"{_L}\.self_attn\.kv_b_proj\.weight": "attn_kv_b",
        rf"{_L}\.self_attn\.indexer\.wq_b\.weight": "indexer_q_b",
        rf"{_L}\.self_attn\.indexer\.wk\.weight": "indexer_k",
        rf"{_L}\.self_attn\.indexer\.weights_proj\.weight": "indexer_weights",
        rf"{_L}\.mlp\.gate_proj\.weight": "mlp_gate",
        rf"{_L}\.mlp\.up_proj\.weight": "mlp_up",
        rf"{_L}\.mlp\.down_proj\.weight": "mlp_down",
        rf"{_L}\.mlp\.gate\.weight": "router",
        rf"{_L}\.mlp\.shared_experts\.gate_proj\.weight": "shared_expert_gate",
        rf"{_L}\.mlp\.shared_experts\.up_proj\.weight": "shared_expert_up",
        rf"{_L}\.mlp\.shared_experts\.down_proj\.weight": "shared_expert_down",
        rf"{_L}\.mlp\.{_E}\.gate_proj\.weight": "expert_gate",
        rf"{_L}\.mlp\.{_E}\.up_proj\.weight": "expert_up",
        rf"{_L}\.mlp\.{_E}\.down_proj\.weight": "expert_down",
    }
    NONLINEAR_PATTERNS = (
        r"model\.norm\.weight",
        rf"{_L}\.(?:input_layernorm|post_attention_layernorm)\.weight",
        rf"{_L}\.hc_(?:attn|ffn)_(?:fn|base|scale)",
        rf"{_L}\.self_attn\.(?:q|k|v)_conv1d\.weight",
        rf"{_L}\.self_attn\.(?:A_log|dt_bias|o_norm\.weight)",
        rf"{_L}\.self_attn\.(?:q_a_layernorm|kv_a_layernorm)\.weight",
        rf"{_L}\.self_attn\.indexer\.(?:k_norm\.weight|k_norm\.bias|index_kpool_compress_gate"
        r"|index_kpool_compress_ape)",
        rf"{_L}\.mlp\.gate\.e_score_correction_bias",
    )
    NORMS: ClassVar[dict[str, str]] = {}   # hyper-connection streams: no residual-stream rotation folding (doc §7)

    # -- structure -------------------------------------------------------------
    @property
    def text_config(self) -> dict:
        return self.config.get("text_config", self.config)

    @property
    def tied_embeddings(self) -> bool:
        return bool(self.config.get("tie_word_embeddings", self.text_config.get("tie_word_embeddings", False)))

    @property
    def num_experts(self) -> int:
        return int(self.text_config.get("n_routed_experts", 0))

    @property
    def residual_width(self) -> int:
        t = self.text_config
        return int(t.get("hc_mult", 4)) * int(t["hidden_size"])

    def layer_type(self, layer: int) -> str:
        """"kda" (linear attention) or "dsa" (MLA + indexer) of 0-indexed `layer`."""
        t = self.text_config
        types = t.get("layer_types")
        if types:
            return "dsa" if types[layer] in _GLM_DSA_TYPES else "kda"
        kda = (t.get("linear_attn_config") or {}).get("kda_layers")
        if kda is not None:
            return "kda" if layer in kda else "dsa"
        return "kda" if layer % 4 != 3 else "dsa"

    def is_sparse_layer(self, layer: int) -> bool:
        t = self.text_config
        kinds = t.get("mlp_layer_types")
        if kinds:
            return kinds[layer] == "sparse"
        return layer >= int(t.get("first_k_dense_replace", 3))

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """Per token and DSA layer: the MLA latent (kv_lora_rank) + the indexer key and k-pool gate score
        (index_head_dim each; suc.models.mlx_glm5_next.DSACache). GLM-5.3-Flash: 11 x (512 + 2 x 128) x 2 B = 16.9 KB."""
        t = self.text_config
        per = int(t["kv_lora_rank"]) + 2 * int(t.get("index_head_dim", 0))
        return sum(per for i in range(self.num_layers) if self.layer_type(i) == "dsa") * dtype_bytes

    def kv_fixed_bytes(self, dtype_bytes: int = 2) -> int:
        """KDA state of every linear-attention layer (float32 heads x Dk x Dv + the conv tail of q / k / v)."""
        t = self.text_config
        la = t.get("linear_attn_config") or {}
        heads = int(la.get("num_heads", t.get("linear_num_heads", 64)))
        hd = int(la.get("head_dim", t.get("linear_head_dim", 128)))
        conv = int(la.get("short_conv_kernel_size", t.get("linear_conv_kernel_dim", 4)))
        per = heads * hd * hd * 4 + 3 * heads * hd * (conv - 1) * dtype_bytes
        return sum(per for i in range(self.num_layers) if self.layer_type(i) == "kda")

    def expected_roles(self, layer: int) -> tuple[Role, ...]:
        if self.layer_type(layer) == "kda":
            attn = ("kda_q", "kda_k", "kda_v", "kda_o", "kda_f_a", "kda_f_b", "kda_g_a", "kda_g_b", "kda_beta")
        else:
            attn = ("attn_q_a", "attn_q_b", "attn_kv_a", "attn_kv_b", "attn_o", "indexer_q_b", "indexer_k",
                    "indexer_weights")
        mlp = ("router", "shared_expert_gate", "shared_expert_up", "shared_expert_down") \
            if self.is_sparse_layer(layer) else MLP_ROLES
        return attn + mlp

    def expected_experts(self, layer: int) -> int:
        return self.num_experts if self.is_sparse_layer(layer) else 0

    def _classify(self, name: str) -> LinearSpec | None:
        s = super()._classify(name)
        if s is not None and s.role == "attn_o" and s.layer is not None and s.layer < self.num_layers \
                and self.layer_type(s.layer) == "kda":
            return LinearSpec(s.name, "kda_o", s.layer, None, True)
        return s

    # -- namespaces --------------------------------------------------------------
    def is_dropped(self, name: str) -> bool:
        """Vision tower, the MTP layer (index num_hidden_layers) and FP8 scale companions."""
        if name.startswith("model.visual.") or name.endswith(".weight_scale_inv"):
            return True
        m = re.match(r"^model\.(?:language_model\.)?layers\.(\d+)\.", name)
        return m is not None and int(m.group(1)) >= self.num_layers

    def to_suc_name(self, name: str) -> str | None:
        """SUC name of a checkpoint tensor (None = dropped); SUC-namespace and unknown names pass through."""
        if self.is_dropped(name):
            return None
        return "model." + name[len(_GLM_HF):] if name.startswith(_GLM_HF) else name

    def _suc_namespace(self, names: Iterable[str]) -> list[str]:
        return [v for n in names if (v := self.to_suc_name(n)) is not None]

    def source_store(self, store):
        """View of `store` (checkpoint names) in the SUC namespace: renamed infos; FP8 weights become logical BF16
        tensors dequantized on load (QuantSource "fp8_block_f32" with their weight_scale_inv companion)."""
        from suc.io.safetensors_store import QuantSource, SafetensorsStore, TensorInfo

        infos: dict[str, TensorInfo] = {}
        quant: dict[str, QuantSource] = {}
        for name in store.names():
            v = self.to_suc_name(name)
            if v is None:
                continue
            info = store.info(name)
            if info.dtype != "F8_E4M3":
                infos[v] = TensorInfo(v, info.dtype, info.shape, info.shard, info.offset, info.nbytes)
                continue
            sname = name + "_scale_inv"
            if not name.endswith(".weight") or sname not in store:
                raise ValueError(f"{name}: F8_E4M3 tensor without its scale companion {sname}")
            infos[v] = TensorInfo(v, "BF16", tuple(info.shape), info.shard, info.offset, 2 * math.prod(info.shape))
            quant[v] = QuantSource("fp8_block_f32", info, store.info(sname))
        return SafetensorsStore(infos, quant)

    def unclassified(self, tensor_names: Iterable[str]) -> list[str]:
        # every tensor, not only "*.weight" (A_log, dt_bias, hc_* parameters, kpool gate / ape have no suffix)
        return sorted(n for n in set(self._suc_namespace(tensor_names)) if self._classify(n) is None
                      and not any(rx.fullmatch(n) for rx in self._nonlinear_re))

    def linears(self, tensor_names: Iterable[str]) -> list[LinearSpec]:
        return super().linears(self._suc_namespace(tensor_names))

    def validate(self, tensor_names: Iterable[str]) -> None:
        raw = list(tensor_names)
        names = self._suc_namespace(raw)
        super().validate(names)
        have = set(names)
        errors = []
        for i in range(self.num_layers):
            p = f"model.layers.{i}."
            want = [p + f"hc_{k}_{v}" for k in ("attn", "ffn") for v in ("fn", "base", "scale")]
            if self.layer_type(i) == "kda":
                want += [p + f"self_attn.{k}" for k in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
                                                        "A_log", "dt_bias", "o_norm.weight")]
            else:
                want += [p + f"self_attn.indexer.{k}" for k in ("index_kpool_compress_gate",
                                                                "index_kpool_compress_ape", "k_norm.weight")]
            if self.is_sparse_layer(i):
                want.append(p + "mlp.gate.e_score_correction_bias")
            if missing := [w for w in want if w not in have]:
                errors.append(f"layer {i}: missing {[m[len(p):] for m in missing]}")
        raw_set = set(raw)
        if missing := [n for n in raw if n.startswith(_GLM_HF) and n.endswith(".weight") and ".experts." in n
                       and n + "_scale_inv" not in raw_set and not self.is_dropped(n)]:
            errors.append(f"{len(missing)} expert weight(s) without a weight_scale_inv companion, e.g. {missing[0]}")
        if errors:
            raise ValueError(f"{self.family} checkpoint invalid:\n  " + "\n  ".join(errors))


_REGISTRY: dict[str, type[ArchitectureAdapter]] = {
    "qwen3": Qwen3Adapter,
    "qwen3_moe": Qwen3MoeAdapter,
    "qwen4_exp": Qwen4ExpAdapter,
    "qwen3_5": Qwen35Adapter,
    "deepseek_v4": DeepseekV4Adapter,
    "glm5_next": Glm5NextAdapter,
}


def get_adapter(config: dict) -> ArchitectureAdapter:
    model_type = config.get("model_type")
    try:
        cls = _REGISTRY[model_type]
    except KeyError:
        raise UnsupportedArchitecture(
            f"unsupported model_type {model_type!r}; supported: {sorted(_REGISTRY)}") from None
    return cls(config)
