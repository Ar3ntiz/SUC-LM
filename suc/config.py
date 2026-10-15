"""Experiment configuration: frozen dataclasses loaded from YAML/dict (spec §7 T0.4, D7).

Every experiment is reproducible from one config file. Unknown keys are rejected so that
typos cannot silently fall back to defaults.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

_AFFINE_BITS = (2, 3, 4, 5, 6, 8)
_DEFAULT_GROUP_SIZE = 64


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str = "main"
    local_path: str | None = None
    role: Literal["dev", "target"] = "dev"


@dataclass(frozen=True)
class DatasetConfig:
    name: Literal["wikitext2"] = "wikitext2"
    split: Literal["test", "validation"] = "test"
    seq_len: int = 2048
    max_tokens: int | None = None


@dataclass(frozen=True)
class RepresentationConfig:
    kind: Literal["bf16", "fp16", "mlx_affine"]
    bits: int | None = None
    group_size: int | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("bf16", "fp16", "mlx_affine"):
            raise ValueError(f"unknown representation kind {self.kind!r}")
        if self.kind == "mlx_affine":
            if self.bits not in _AFFINE_BITS:
                raise ValueError(f"mlx_affine requires bits in {_AFFINE_BITS}, got {self.bits!r}")
            if self.group_size is None:
                object.__setattr__(self, "group_size", _DEFAULT_GROUP_SIZE)
            if self.group_size <= 0:
                raise ValueError(f"group_size must be positive, got {self.group_size}")
        elif self.bits is not None or self.group_size is not None:
            raise ValueError(f"bits/group_size are only valid for mlx_affine, not {self.kind!r}")
        if self.name is None:
            name = self.kind if self.kind != "mlx_affine" else f"mlx_affine_b{self.bits}_g{self.group_size}"
            object.__setattr__(self, "name", name)


@dataclass(frozen=True)
class BenchConfig:
    prompt_lengths: tuple[int, ...] = (128, 512)
    decode_tokens: int = 128
    batch_sizes: tuple[int, ...] = (1,)
    warmup: int = 1
    repeats: int = 3


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: ModelConfig
    dataset: DatasetConfig
    representations: tuple[RepresentationConfig, ...]
    bench: BenchConfig
    seed: int = 12345
    output_dir: str = "experiments/results"


_ANALYSIS_ROLES = ("attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down",
                   "expert_gate", "expert_up", "expert_down",
                   "shared_expert_gate", "shared_expert_up", "shared_expert_down")


@dataclass(frozen=True)
class AnalysisConfig:
    """Phase 1 weight-structure analysis (spec §8). Only SUC-eligible linear tensors are analysed."""
    layers: tuple[int, ...] | None = None      # None = first, middle and last layer
    roles: tuple[str, ...] = _ANALYSIS_ROLES   # roles absent from the checkpoint are skipped
    experts_per_layer: int = 4                 # MoE: deterministic sample of experts per layer
    block_sizes: tuple[int, ...] = (8, 16, 32, 64, 128)
    kmeans_block_sizes: tuple[int, ...] = (8, 16)
    codebook_sizes: tuple[int, ...] = (16, 64, 256, 1024)
    rotations: tuple[bool, ...] = (False, True)  # randomized Hadamard off / on (D3)
    normalize_rows: bool = True
    max_blocks: int = 262144                   # per-tensor cap on blocks used for statistics and k-means
    kmeans_iters: int = 25

    def __post_init__(self) -> None:
        unknown = sorted(set(self.roles) - set(_ANALYSIS_ROLES))
        if unknown:
            raise ValueError(f"analysis.roles: unknown role(s) {unknown}")
        extra = sorted(set(self.kmeans_block_sizes) - set(self.block_sizes))
        if extra:
            raise ValueError(f"analysis.kmeans_block_sizes {extra} must be a subset of block_sizes")
        if self.experts_per_layer < 1 or self.max_blocks < 1 or self.kmeans_iters < 1:
            raise ValueError("analysis: experts_per_layer, max_blocks and kmeans_iters must be positive")


@dataclass(frozen=True)
class AnalysisExperimentConfig:
    name: str
    model: ModelConfig
    analysis: AnalysisConfig
    seed: int = 12345
    output_dir: str = "experiments/results"


_CALIB_DOMAINS = ("natural", "code", "math", "math_competition", "instruction")
# Calibration mixes (domain -> share of the calibration / validation windows; suc.data.sources):
#   v1: every run before 2026-10-11 (math = GSM8K train only). Every committed config states it explicitly.
#   v2: the math share split evenly between GSM8K train ("math") and competition-style MATH-train problems with
#       worked solutions ("math_competition", chat-rendered, disjoint from MATH-500 and every eval suite item:
#       experiments/eval/calibration_disjointness.json). Measured once (DeepSeek-V4-Flash tcq, report §5c): MATH
#       16/20 vs 12/20, but MMLU-Pro 10/20 vs 16/20, hard suite 101 vs 102 of 119, KL 0.246 vs 0.234 -- not adopted.
#   v3: the default for NEW configs: v1 with a small competition-math share (3% of the windows, taken from GSM8K).
# A config opts in by omitting `calibration.mix` or writing `mix: v3`; `mix: v1` (or the explicit v1 dict) keeps the
# old corpus. Stage fingerprints measure the mix against CALIB_MIX_V1 (field metadata "fingerprint_default", see
# suc.pipeline.compress.config_delta), so a v1 run's fingerprints are unchanged and a v2 run gets new ones.
CALIB_MIX_V1 = {"natural": 0.35, "code": 0.25, "math": 0.2, "instruction": 0.2}
CALIB_MIX_V2 = {"natural": 0.35, "code": 0.25, "math": 0.1, "math_competition": 0.1, "instruction": 0.2}
CALIB_MIX_V3 = {"natural": 0.35, "code": 0.25, "math": 0.17, "math_competition": 0.03, "instruction": 0.2}
CALIB_MIXES = {"v1": CALIB_MIX_V1, "v2": CALIB_MIX_V2, "v3": CALIB_MIX_V3}


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration corpus and activation statistics (spec §9 T2.1/T2.2, §32). `mix`: a domain -> share mapping or
    a preset name of CALIB_MIXES ("v1" / "v2" / "v3", resolved to its mapping); default v3 (see CALIB_MIXES above)."""
    seq_len: int = 512
    n_calibration: int = 512
    n_validation: int = 128
    n_ood: int = 64
    mix: dict = field(default_factory=lambda: dict(CALIB_MIX_V3), metadata={"fingerprint_default": CALIB_MIX_V1})
    micro_batch: int = 8
    full_second_moment: tuple[str, ...] = ()   # tensor names (without .weight) that keep the full E[x xᵀ]

    def __post_init__(self) -> None:
        if isinstance(self.mix, str):
            if self.mix not in CALIB_MIXES:
                raise ValueError(f"calibration.mix: unknown preset {self.mix!r}; allowed {sorted(CALIB_MIXES)} "
                                 "or a domain -> share mapping")
            object.__setattr__(self, "mix", dict(CALIB_MIXES[self.mix]))
        if not isinstance(self.mix, dict):
            raise ValueError(f"calibration.mix must be a mapping or a preset name, got {type(self.mix).__name__}")
        unknown = sorted(set(self.mix) - set(_CALIB_DOMAINS))
        if unknown:
            raise ValueError(f"calibration.mix: unknown domain(s) {unknown}; allowed {_CALIB_DOMAINS}")
        if abs(sum(self.mix.values()) - 1.0) > 1e-6:
            raise ValueError(f"calibration.mix must sum to 1, got {sum(self.mix.values())}")
        if min(self.seq_len, self.n_calibration, self.n_validation, self.n_ood, self.micro_batch) < 1:
            raise ValueError("calibration: sizes must be positive")


@dataclass(frozen=True)
class SensitivityConfig:
    """Proxy sensitivity for every eligible tensor, measured perturbations on a sampled subset (D5)."""
    kind: Literal["vq", "gaussian"] = "vq"
    block_size: int = 8
    k: int = 256
    rotate: bool = True
    gaussian_level: float = 0.1
    measured_tensors: int = 24        # sampled tensors with real perturbation tests (validates the proxy)
    eval_sequences: int = 16          # validation sequences used for measured KL / ΔCE
    eval_chunk: int = 4               # sequences per forward during measurement (bounds logits memory)
    experts_per_layer: int = 8        # MoE: experts per layer that get a proxy score (deterministic sample)
    fisher: bool = True               # empirical-Fisher proxy ½ Σ F ΔW² (captures downstream propagation)
    fisher_loss: Literal["ce", "kl_self"] = "ce"
    fisher_sequences: int = 32        # calibration sequences used for F
    fisher_micro_batch: int = 2
    fisher_group_size: int | None = None   # parameters per backward pass; set for large models to bound memory

    def __post_init__(self) -> None:
        if self.measured_tensors < 1:
            raise ValueError("sensitivity.measured_tensors must be >= 1")
        if min(self.eval_sequences, self.eval_chunk, self.experts_per_layer, self.block_size) < 1 or self.k < 2:
            raise ValueError("sensitivity: eval_sequences, k and block_size must be positive")


@dataclass(frozen=True)
class CalibrationExperimentConfig:
    name: str
    model: ModelConfig
    calibration: CalibrationConfig
    sensitivity: SensitivityConfig
    seed: int = 12345
    output_dir: str = "experiments/results"


@dataclass(frozen=True)
class CodecConfig:
    """One weight representation of the quality track (Q1–Q4); maps 1:1 onto suc.quant.codecs.make_codec.
    kind "mxfp4" is the native pass-through of MXFP4-at-source tensors (suc.quant.codecs.MXFP4Native: checkpoint codes +
    E8M0 scales verbatim, 4.25 bpw, zero error vs the D8 reference); in a greedy plan it is not scored on the Fisher
    (suc.pipeline.compress adds it with loss 0 to the tensors whose source is MXFP4) and takes no other field."""
    kind: Literal["bf16", "affine", "scalar", "vq", "additive", "mxfp4", "tcq"]
    bits: int | None = None               # affine
    group_size: int = 64                  # affine
    k: int = 256                          # scalar / vq / additive codebook entries
    block_size: int = 8                   # vq / additive (G)
    num_codebooks: int | None = None      # additive (M >= 2); vq = 1
    rotate: bool = True                   # vq / additive: randomized Hadamard (D3)
    rotate_out: bool = True               # vq / additive: also rotate the output side (False = one-sided, input only)
    # Rotate the residual stream offline (suc.models.rotation_folding: norms folded, embedding / readers ·Q, writers
    # Qᵀ·); vq / additive tensors that read the rotated stream are then encoded without an input rotation.
    fold_rotation: bool = False
    fold_v_heads: bool = False            # per-head Hadamard folded into v_proj rows / o_proj columns (alone or with
                                          # fold_rotation): o_proj then needs no online input rotation
    scope: Literal["tensor", "row_group"] = "tensor"   # scalar codebook scope
    row_group: int = 64
    refine_iters: int = 0                 # additive joint refinement
    kmeans_iters: int = 25
    fit_blocks: int = 262144
    activation_weighted: bool = False     # weight the codebook distance by E[x_j^2] (Phase 2 statistics)
    residual_fraction: float = 0.0        # Q5: keep this fraction of residual entries exactly (0 = off)
    residual_dtype: Literal["int8", "fp16"] = "int8"
    residual_selection: Literal["activation", "magnitude"] = "activation"   # activation: E[x_j^2]·R² (Phase 2 stats)
    # Q5b: "csr" (sparse positions) or "columns" (dense int8 residual on the input columns with the largest
    # importance, same bit budget as the CSR of residual_fraction; suc.quant.residual)
    residual_structure: Literal["csr", "columns"] = "csr"
    residual_column_share: float = 1.0    # columns: share of the budget spent on columns, the rest is a CSR tail
    # vq / additive on MoE experts: "expert_stack" fits ONE codebook set per (layer, projection) stack on a pooled
    # sample of all its experts and shares the rotation seeds (suc.quant.stacks); "tensor" = one per expert tensor.
    # tcq (no codebook): "expert_stack" only shares the input rotation seed of the stack (gate / up: of the layer).
    # Non-expert tensors are always encoded per tensor.
    codebook_scope: Literal["tensor", "expert_stack"] = "tensor"
    # tcq (suc.quant.trellis, docs/low_tier_codec.md): bitshift-trellis quantizer, tcq_bits per weight (tcq_bits * V
    # integer), L-bit states, code "1mad" / "3inst" (pure compute) or "hyb" (hashed 2^9 x 2 LUT), V weights per step
    tcq_bits: float = 2.0
    tcq_L: int = 16
    tcq_code: Literal["1mad", "3inst", "pair3", "hyb"] = "1mad"
    tcq_V: int | None = None
    tcq_T: int = 256                      # tcq: weights per tail-biting sequence (kernels: T % (32 V) == 0, T >= 128)
    # tcq: BlockLDLQ with the full input second moment of every dense linear (recorded by suc quantize on
    # ldlq_sequences calibration sequences); name suffix "_ldlq"
    ldlq: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("bf16", "affine", "scalar", "vq", "additive", "mxfp4", "tcq"):
            raise ValueError(f"codec kind {self.kind!r} unknown")
        if self.kind == "mxfp4" and (self.residual_fraction or self.activation_weighted or self.fold_rotation
                                     or self.fold_v_heads):
            raise ValueError("codec kind 'mxfp4' (native source pass-through) takes no residual / rotation folding")
        if self.kind == "affine" and self.bits not in _AFFINE_BITS:
            raise ValueError(f"affine codec requires bits in {_AFFINE_BITS}, got {self.bits!r}")
        if not 0.0 <= self.residual_fraction <= 1.0:
            raise ValueError(f"residual_fraction must be in [0, 1], got {self.residual_fraction}")
        if self.residual_structure not in ("csr", "columns"):
            raise ValueError(f"residual_structure must be 'csr' or 'columns', got {self.residual_structure!r}")
        if not 0.0 < self.residual_column_share <= 1.0 or (self.residual_structure == "csr"
                                                            and self.residual_column_share != 1.0):
            raise ValueError(f"residual_column_share must be in (0, 1] and only set with residual_structure: "
                             f"columns, got {self.residual_column_share}")
        if self.codebook_scope not in ("tensor", "expert_stack"):
            raise ValueError(f"codebook_scope must be 'tensor' or 'expert_stack', got {self.codebook_scope!r}")
        if self.ldlq and (self.kind != "tcq" or not self.rotate or self.rotate_out):
            raise ValueError("ldlq needs kind tcq with the one-sided input rotation (rotate: true, rotate_out: false)")
        if self.codebook_scope == "expert_stack" and self.kind not in ("vq", "additive", "tcq"):
            raise ValueError(f"codebook_scope 'expert_stack' needs a vq / additive / tcq codec, got kind {self.kind!r}")
        m = self.num_codebooks if self.num_codebooks is not None else (2 if self.kind == "additive" else 1)
        if self.kind == "additive" and m < 2:
            raise ValueError("additive codec requires num_codebooks >= 2")
        object.__setattr__(self, "num_codebooks", m)
        if self.name is None:
            rot = ("_rot" + ("" if self.rotate_out else "1s")) if self.rotate else ""
            aw = "_aw" if self.activation_weighted else ""
            name = {"bf16": "bf16", "mxfp4": "mxfp4_native",
                    "affine": f"affine_b{self.bits}_g{self.group_size}",
                    "scalar": f"scalar_k{self.k}_{self.scope}",
                    "vq": f"vq_k{self.k}_g{self.block_size}{rot}",
                    "additive": f"additive_m{m}_k{self.k}_g{self.block_size}{rot}"
                                + (f"_r{self.refine_iters}" if self.refine_iters else ""),
                    "tcq": f"tcq_{self.tcq_code}_b{self.tcq_bits:g}_L{self.tcq_L}"
                           + (f"_v{self.tcq_V}" if self.tcq_V else "") + (f"_t{self.tcq_T}" if self.tcq_T != 256 else "")
                           + rot}[self.kind] + aw
            if self.codebook_scope == "expert_stack":
                name += "_xs"
            if self.fold_rotation:
                name += "_fold" + ("v" if self.fold_v_heads else "")
            elif self.fold_v_heads:
                name += "_vfold"
            if self.ldlq:
                name += "_ldlq"
            if self.residual_fraction:
                name += f"+res{self.residual_fraction:g}_{self.residual_selection}"
                if self.residual_structure == "columns":
                    name += "_cols" + (f"{self.residual_column_share:g}" if self.residual_column_share != 1.0 else "")
            object.__setattr__(self, "name", name)

    def codec_dict(self) -> dict:
        keys = {"bf16": (), "mxfp4": (), "affine": ("bits", "group_size"), "scalar": ("k", "scope", "row_group"),
                "vq": ("k", "block_size", "rotate", "kmeans_iters", "fit_blocks"),
                "additive": ("k", "block_size", "num_codebooks", "rotate", "refine_iters", "kmeans_iters", "fit_blocks"),
                "tcq": ("rotate", "rotate_out")}
        d = {"kind": self.kind, **{key: getattr(self, key) for key in keys[self.kind]}}
        if self.kind == "tcq":
            d.update(k=self.tcq_bits, L=self.tcq_L, code=self.tcq_code, V=self.tcq_V)
            if self.tcq_T != 256:
                d["T"] = self.tcq_T
        if self.kind in ("vq", "additive") and not self.rotate_out:
            d["rotate_out"] = False          # only when set: codec dicts of earlier configs are unchanged
        return d


@dataclass(frozen=True)
class QuantEvalConfig:
    wikitext_split: Literal["test", "validation"] = "test"   # final-test perplexity (spec §32: never tuned on)
    seq_len: int = 2048
    max_tokens: int | None = None
    kl_sequences: int = 16            # held-out validation sequences of the Phase 2 corpus for KL vs BF16
    chunk: int = 4
    calib_artifact: str | None = None # Phase 2 *_data.safetensors; None = newest matching the model in experiments/artifacts


@dataclass(frozen=True)
class FinetuneSpec:
    """Q6 block-wise fine-tuning of codebooks / row scales / residual values (suc.quant.finetune)."""
    block_steps: int = 150
    lr: float = 1e-3
    sequences: int = 64               # calibration sequences of the Phase 2 corpus
    micro_batch: int = 4
    deterministic: bool = True


@dataclass(frozen=True)
class QuantizeExperimentConfig:
    name: str
    model: ModelConfig
    codecs: tuple[CodecConfig, ...]
    eval: QuantEvalConfig
    roles: tuple[str, ...] | None = None      # None = every eligible linear tensor
    include_embeddings: bool = True           # also encode embedding / untied LM head (fair whole-model bpp)
    ldlq_sequences: int = 128                 # calibration sequences for the full input second moments (ldlq codecs)
    # Fixed codec for the embedding / untied LM head in EVERY row (as embedding_codec in suc allocate / compress);
    # None = they get the row's codec (Q5 protocol). Requires include_embeddings; no residual.
    embedding_codec: CodecConfig | None = None
    finetune: FinetuneSpec | None = None      # Q6: applied after encoding to every vq / additive codec
    rotation_groups: bool = False             # share the input rotation of q/k/v and gate/up (runtime fusion, K5)
    activation_stats: str | None = None       # Phase 2 *_activations.safetensors (for activation_weighted codecs)
    seed: int = 12345
    output_dir: str = "experiments/results"

    def __post_init__(self) -> None:
        if self.finetune is not None and any(c.codebook_scope == "expert_stack" for c in self.codecs):
            raise ValueError("finetune is not supported with codebook_scope 'expert_stack' yet (per-expert tuning "
                             "would un-share the stack codebooks)")
        if self.embedding_codec is not None:
            if not self.include_embeddings:
                raise ValueError("embedding_codec requires include_embeddings: true")
            if self.embedding_codec.residual_fraction:
                raise ValueError("embedding_codec cannot carry a residual")


@dataclass(frozen=True)
class AllocateExperimentConfig:
    """Phase 7 / Q7: Fisher-scored candidates → greedy plans per budget → full-model validation (spec §39)."""
    name: str
    model: ModelConfig
    candidates: tuple[CodecConfig, ...]
    budgets_bpp: tuple[float, ...]            # average bits per weight over the allocated tensors
    eval: QuantEvalConfig
    uniform: tuple[str, ...] = ()             # candidate codec names evaluated as uniform baselines
    embedding_codec: CodecConfig | None = None   # fixed codec for embedding / untied LM head; None = keep BF16
    # Embedding (+ untied LM head) as allocation candidates in the same greedy budget; None = old behaviour
    # (excluded from allocation, encoded with embedding_codec). Mutually exclusive with embedding_codec.
    head_candidates: tuple[CodecConfig, ...] | None = None
    uniform_head: str | None = None           # head candidate used for embed / LM head in the uniform baselines
    fisher_loss: Literal["ce", "kl_self"] = "ce"
    fisher_sequences: int = 32
    fisher_micro_batch: int = 2
    fisher_group_size: int | None = None
    # "expert_stack": the experts of one MoE (layer, projection) stack are ONE allocation unit (same codec for all,
    # bits / Fisher loss summed); "tensor": every expert tensor is its own unit (earlier results)
    allocation_unit: Literal["tensor", "expert_stack"] = "tensor"
    # Resume: a *_candidates.json artifact written by an earlier run of THIS config; Fisher and candidate scoring
    # are skipped and the saved candidates are used (units and codec names are checked against the config)
    candidates_artifact: str | None = None
    seed: int = 12345
    output_dir: str = "experiments/results"

    def __post_init__(self) -> None:
        _check_unit(self.allocation_unit, self.candidates)
        if self.candidates_artifact is not None and not str(self.candidates_artifact).strip():
            raise ValueError("candidates_artifact must be a path (or omitted)")
        names = {c.name for c in self.candidates}
        missing = sorted(set(self.uniform) - names)
        if missing:
            raise ValueError(f"uniform baselines {missing} are not among the candidates {sorted(names)}")
        if not self.budgets_bpp or min(self.budgets_bpp) <= 0:
            raise ValueError("budgets_bpp must be a non-empty list of positive numbers")
        if self.head_candidates is None:
            if self.uniform_head is not None:
                raise ValueError("uniform_head requires head_candidates")
            return
        if not self.head_candidates:
            raise ValueError("head_candidates must be a non-empty list (or omitted)")
        if self.embedding_codec is not None:
            raise ValueError("embedding_codec and head_candidates are mutually exclusive")
        head_names = [c.name for c in self.head_candidates]
        if len(set(head_names)) != len(head_names):
            raise ValueError(f"duplicate head_candidates names {head_names}")
        if any(c.residual_fraction and c.residual_selection == "activation" for c in self.head_candidates):
            raise ValueError("head_candidates cannot use activation-selected residuals (no input activations "
                             "for the embedding); use residual_selection: magnitude")
        if self.uniform and self.uniform_head is None:
            raise ValueError("uniform baselines with head_candidates require uniform_head")
        if self.uniform_head is not None and self.uniform_head not in head_names:
            raise ValueError(f"uniform_head {self.uniform_head!r} is not among the head_candidates {head_names}")


def _check_unit(unit: str, candidates, allowed: tuple[str, ...] = ("tensor", "expert_stack")) -> None:
    if unit not in allowed:
        raise ValueError(f"allocation_unit must be one of {list(allowed)}, got {unit!r}")
    shared = [c.name for c in candidates if c.codebook_scope == "expert_stack"]
    if shared and unit not in ("expert_stack", "expert"):
        raise ValueError(f"candidates {shared} share codebooks per expert stack: they need allocation_unit "
                         "'expert_stack' (every expert of a stack must get the same codec)")


def _check_literal(cls: type, field: str, value: Any) -> None:
    hint = cls.__annotations__[field]
    if isinstance(hint, str) and hint.startswith("Literal["):
        allowed = eval(hint, {"Literal": Literal})  # annotations are strings (postponed evaluation)
        if value not in get_args(allowed):
            raise ValueError(f"{cls.__name__}.{field}: {value!r} not in {get_args(allowed)}")


def _build(cls: type, d: Any, where: str, **nested: Any) -> Any:
    """Construct a flat dataclass from a dict, rejecting unknown and missing keys."""
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(d).__name__}")
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - names)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}")
    required = [f.name for f in fields(cls)
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    missing = [k for k in required if k not in d and k not in nested]
    if missing:
        raise ValueError(f"{where}: missing required key(s) {missing}")
    kw = {**d, **nested}
    for f in fields(cls):
        if f.name in kw and f.name not in nested:
            v = kw[f.name]
            if isinstance(v, list) and (isinstance(f.default, tuple) or "tuple" in str(f.type)):
                kw[f.name] = v = tuple(v)
            _check_literal(cls, f.name, v)
    return cls(**kw)


def config_from_dict(d: dict) -> ExperimentConfig:
    if not isinstance(d, dict):
        raise ValueError("config: expected a mapping")
    d = dict(d)
    if "representations" not in d:
        raise ValueError("config: missing required key(s) ['representations']")
    reps = d.pop("representations")
    if not isinstance(reps, (list, tuple)) or not reps:
        raise ValueError("config: 'representations' must be a non-empty list")
    if "model" not in d:
        raise ValueError("config: missing required key(s) ['model']")
    nested = {
        "model": _build(ModelConfig, d.pop("model"), "model"),
        "dataset": _build(DatasetConfig, d.pop("dataset", {}) or {}, "dataset"),
        "bench": _build(BenchConfig, d.pop("bench", {}) or {}, "bench"),
        "representations": tuple(_build(RepresentationConfig, r, f"representations[{i}]")
                                 for i, r in enumerate(reps)),
    }
    return _build(ExperimentConfig, d, "config", **nested)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        return config_from_dict(yaml.safe_load(f) or {})


def analysis_config_from_dict(d: dict) -> AnalysisExperimentConfig:
    if not isinstance(d, dict):
        raise ValueError("config: expected a mapping")
    d = dict(d)
    if "model" not in d:
        raise ValueError("config: missing required key(s) ['model']")
    nested = {
        "model": _build(ModelConfig, d.pop("model"), "model"),
        "analysis": _build(AnalysisConfig, d.pop("analysis", {}) or {}, "analysis"),
    }
    return _build(AnalysisExperimentConfig, d, "config", **nested)


def load_analysis_config(path: str | Path) -> AnalysisExperimentConfig:
    with open(path) as f:
        return analysis_config_from_dict(yaml.safe_load(f) or {})


def calibration_config_from_dict(d: dict) -> CalibrationExperimentConfig:
    if not isinstance(d, dict):
        raise ValueError("config: expected a mapping")
    d = dict(d)
    if "model" not in d:
        raise ValueError("config: missing required key(s) ['model']")
    nested = {
        "model": _build(ModelConfig, d.pop("model"), "model"),
        "calibration": _build(CalibrationConfig, d.pop("calibration", {}) or {}, "calibration"),
        "sensitivity": _build(SensitivityConfig, d.pop("sensitivity", {}) or {}, "sensitivity"),
    }
    return _build(CalibrationExperimentConfig, d, "config", **nested)


def load_calibration_config(path: str | Path) -> CalibrationExperimentConfig:
    with open(path) as f:
        return calibration_config_from_dict(yaml.safe_load(f) or {})


def quantize_config_from_dict(d: dict) -> QuantizeExperimentConfig:
    if not isinstance(d, dict):
        raise ValueError("config: expected a mapping")
    d = dict(d)
    for key in ("model", "codecs"):
        if key not in d:
            raise ValueError(f"config: missing required key(s) [{key!r}]")
    codecs = d.pop("codecs")
    if not isinstance(codecs, (list, tuple)) or not codecs:
        raise ValueError("config: 'codecs' must be a non-empty list")
    nested = {
        "model": _build(ModelConfig, d.pop("model"), "model"),
        "eval": _build(QuantEvalConfig, d.pop("eval", {}) or {}, "eval"),
        "codecs": tuple(_build(CodecConfig, c, f"codecs[{i}]") for i, c in enumerate(codecs)),
    }
    ft = d.pop("finetune", None)
    nested["finetune"] = None if ft is None else _build(FinetuneSpec, ft, "finetune")
    emb = d.pop("embedding_codec", None)
    nested["embedding_codec"] = None if emb is None else _build(CodecConfig, emb, "embedding_codec")
    return _build(QuantizeExperimentConfig, d, "config", **nested)


def load_quantize_config(path: str | Path) -> QuantizeExperimentConfig:
    with open(path) as f:
        return quantize_config_from_dict(yaml.safe_load(f) or {})


def allocate_config_from_dict(d: dict) -> AllocateExperimentConfig:
    if not isinstance(d, dict):
        raise ValueError("config: expected a mapping")
    d = dict(d)
    for key in ("model", "candidates", "budgets_bpp"):
        if key not in d:
            raise ValueError(f"config: missing required key(s) [{key!r}]")
    cands = d.pop("candidates")
    if not isinstance(cands, (list, tuple)) or not cands:
        raise ValueError("config: 'candidates' must be a non-empty list")
    emb = d.pop("embedding_codec", None)
    head = d.pop("head_candidates", None)
    if head is not None and not isinstance(head, (list, tuple)):
        raise ValueError("config: 'head_candidates' must be a list")
    nested = {
        "model": _build(ModelConfig, d.pop("model"), "model"),
        "eval": _build(QuantEvalConfig, d.pop("eval", {}) or {}, "eval"),
        "candidates": tuple(_build(CodecConfig, c, f"candidates[{i}]") for i, c in enumerate(cands)),
        "embedding_codec": None if emb is None else _build(CodecConfig, emb, "embedding_codec"),
        "head_candidates": None if head is None else tuple(_build(CodecConfig, c, f"head_candidates[{i}]")
                                                           for i, c in enumerate(head)),
    }
    return _build(AllocateExperimentConfig, d, "config", **nested)


def load_allocate_config(path: str | Path) -> AllocateExperimentConfig:
    with open(path) as f:
        return allocate_config_from_dict(yaml.safe_load(f) or {})


AnyConfig = AllocateExperimentConfig | ExperimentConfig | AnalysisExperimentConfig | CalibrationExperimentConfig | QuantizeExperimentConfig


def config_to_dict(cfg: AnyConfig) -> dict:
    def conv(x: Any) -> Any:
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        return x
    return conv(dataclasses.asdict(cfg))


def config_hash(cfg: AnyConfig) -> str:
    blob = json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------- suc compress (unattended end-to-end pipeline)

@dataclass(frozen=True)
class CompressPlanConfig:
    """Per-tensor codec plan of `suc compress`: one uniform codec, or ONE greedy Fisher plan for a budget.

    budget_bpw is the average bits per weight over the allocated tensors (as budgets_bpp in Q7). budget_bytes is the
    alternative TOTAL-size budget: every tensor of the model (allocated ones at their candidate bits, all others at their
    stored dtype / fixed embedding codec) + the SUC1 container overhead must fit in budget_bytes (suc.pipeline.compress
    .fixed_bits). The embedding / untied LM head is either encoded with the fixed embedding_codec (outside the budget;
    None = keep BF16) or, in greedy mode, allocated inside the budget from head_candidates (not both).

    allocation_unit "expert" (greedy): every routed-expert tensor is its own allocation unit, but candidates with
    codebook_scope expert_stack keep ONE pooled codebook fit per (stack, codec): the experts of a stack that get the
    same codec share it (suc.quant.stacks.encode_stack fit_names), so experts of one stack may get different codecs
    (the runtime runs such a stack as one SucSwitchLinear per codec, suc.runtime.compressed_model).
    candidate_roles restricts candidates to roles ({codec name: [role, ...]}; unlisted candidates apply to every
    allocated tensor), e.g. low-bit codecs for routed experts only.

    role_budgets (greedy): separate budgets for groups of roles, [{roles: [...], budget_bpw: x}, ...] (disjoint roles):
    the tensors of a group are allocated by their own greedy pass for an average of x bits per weight over the group,
    the remaining allocated tensors share budget_bpw (or what budget_bytes leaves after the fixed part and the group
    budgets). E.g. routed experts at exactly 3.0 bpw on average, whatever the other tensors get.

    allocation (greedy): ALLOCATION-ONLY overrides, {budget_bytes: x, exclude: {candidate name: [roles]},
    role_budgets: [...]}. They change what the greedy may choose (budget_bytes, only for budget_bytes plans, replaces
    plan.budget_bytes for the plan, the encode memory estimate and the runtime check; exclude removes candidates for
    the listed roles; role_budgets replaces plan.role_budgets) but not what the candidates stage scores: the candidates fingerprint leaves this field out (suc.pipeline.compress
    .scored_plan_delta), so a re-plan variant reuses the calibrate / reference / candidates artifacts of a run with
    the same scored candidates and plan.budget_bytes.

    role_codecs (uniform): {codec name: [roles]}: the tensors of the listed roles get that codec instead of plan.codec
    (a fixed per-role plan without a Fisher pass, e.g. affine 8-bit attention + VQ experts). Block roles take a
    candidate; the roles "embed" / "lm_head" take a candidate or the embedding_codec and override embedding_codec
    for that tensor (e.g. BF16 embedding + affine 8-bit LM head). Roles must not repeat across codecs."""
    mode: Literal["uniform", "greedy"]
    candidates: tuple[CodecConfig, ...]
    codec: str | None = None                  # uniform: name of the candidate applied to every eligible tensor
    budget_bpw: float | None = None           # greedy: average bpw over the allocated tensors
    budget_bytes: float | None = None         # greedy: total model size budget (bytes), alternative to budget_bpw
    candidate_roles: dict | None = None       # greedy: {candidate name: [roles it may be used for]}
    role_budgets: tuple | None = None         # greedy: ({roles: [...], budget_bpw: x}, ...) allocated separately
    embedding_codec: CodecConfig | None = None
    head_candidates: tuple[CodecConfig, ...] | None = None
    fisher_loss: Literal["ce", "kl_self"] = "ce"
    fisher_sequences: int = 32
    fisher_micro_batch: int = 2
    fisher_group_size: int | None = None
    allocation_unit: Literal["tensor", "expert_stack", "expert"] = "tensor"   # as AllocateExperimentConfig's + "expert"
    allocation: dict | None = None            # greedy: allocation-only overrides {budget_bytes, exclude, role_budgets}
    role_codecs: dict | None = None           # uniform: {codec name: [roles]} per-role overrides of plan.codec
    # tcq + ldlq codecs (suc.quant.trellis BlockLDLQ): input second moments recorded by the `hessians` stage on the
    # first ldlq_sequences calibration sequences (layer-streamed): dense linears E[x xᵀ] per input group, MoE layers
    # ONE shared E[x xᵀ] of the expert input for every gate / up expert and one per-expert E[h hᵀ] of the routed rows
    # for each down expert (suc.analysis.hessians). ldlq_damp: H + damp · mean(diag H) · I; a down expert with fewer
    # than ldlq_min_tokens routed rows is encoded without LDLQ.
    ldlq_sequences: int = 512
    ldlq_damp: float = 0.01
    ldlq_min_tokens: int = 256
    # Candidates stage: tcq candidates are scored on a deterministic sample of tcq_score_rows rows per tensor (the
    # trellis codec is row-independent; loss extrapolated by out / rows, bits exact from the shape); 0 = all rows.
    tcq_score_rows: int = 0
    # Candidates stage: an earlier run directory (or its candidates/ directory) whose scores are reused for every
    # candidate whose codec config, Fisher inputs (model, seed, calibration outputs, fisher_*) are identical; only the
    # other candidates are encoded and scored. Not part of the candidates fingerprint (the scores do not depend on it).
    candidates_reuse: str | None = None

    def __post_init__(self) -> None:
        _check_unit(self.allocation_unit, self.candidates, ("tensor", "expert_stack", "expert"))
        if self.allocation is not None:
            self._check_allocation()
        if self.allocation_unit == "expert" and self.mode != "greedy":
            raise ValueError("plan: allocation_unit 'expert' is only valid with mode 'greedy'")
        if self.candidate_roles is not None:
            from suc.models.adapters import ROLES

            if self.mode != "greedy" or not isinstance(self.candidate_roles, dict):
                raise ValueError("plan.candidate_roles must be a mapping and is only valid with mode 'greedy'")
            names = {c.name for c in self.candidates}
            for k, v in self.candidate_roles.items():
                if k not in names:
                    raise ValueError(f"plan.candidate_roles: {k!r} is not a candidate ({sorted(names)})")
                if not isinstance(v, (list, tuple)) or not v or any(r not in ROLES for r in v):
                    raise ValueError(f"plan.candidate_roles[{k!r}] must be a non-empty list of roles from {ROLES}")
        if self.role_budgets is not None:
            self._check_role_budgets(self.role_budgets, "plan.role_budgets")
        if self.mode not in ("uniform", "greedy"):
            raise ValueError(f"plan.mode must be 'uniform' or 'greedy', got {self.mode!r}")
        if self.role_codecs is not None:
            self._check_role_codecs()
        if not self.candidates:
            raise ValueError("plan.candidates must be a non-empty list")
        names = [c.name for c in self.candidates]
        if len(set(names)) != len(names):
            raise ValueError(f"plan.candidates: duplicate codec names {names}")
        if self.mode == "uniform":
            if self.codec not in names:
                raise ValueError(f"plan.codec {self.codec!r} is not among the candidates {names}")
            if self.budget_bpw is not None or self.budget_bytes is not None or self.head_candidates is not None:
                raise ValueError("plan: budget_bpw / budget_bytes / head_candidates are only valid with mode 'greedy'")
        else:
            if (self.budget_bpw is None) == (self.budget_bytes is None):
                raise ValueError("plan: mode 'greedy' requires exactly one of budget_bpw / budget_bytes")
            for k in ("budget_bpw", "budget_bytes"):
                v = getattr(self, k)
                if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
                    raise ValueError(f"plan.{k} must be a number, got {v!r} (YAML: write 107500000000, not 107.5e9)")
            if (self.budget_bpw if self.budget_bpw is not None else self.budget_bytes) <= 0:
                raise ValueError("plan: the greedy budget must be positive")
            if self.codec is not None:
                raise ValueError("plan.codec is only valid with mode 'uniform'")
        if self.head_candidates is not None and self.embedding_codec is not None:
            raise ValueError("plan: give either embedding_codec (fixed) or head_candidates (allocated), not both")
        every = list(self.candidates) + list(self.head_candidates or ()) + \
            ([self.embedding_codec] if self.embedding_codec is not None else [])
        by_name: dict[str, CodecConfig] = {}
        for c in every:
            if c.activation_weighted:
                raise ValueError(f"plan: activation_weighted codecs are not supported by suc compress ({c.name})")
            if c.name in by_name and by_name[c.name] != c:
                raise ValueError(f"plan: two different codec configs share the name {c.name!r}")
            by_name[c.name] = c
        from suc.capabilities import export_errors   # audit F07: SUC1-exportable codecs only, before any model load

        errs = [e for c in by_name.values() for e in export_errors(c)]
        if errs:
            raise ValueError("plan: " + "; ".join(errs))
        if min(self.fisher_sequences, self.fisher_micro_batch) < 1:
            raise ValueError("plan: fisher_sequences and fisher_micro_batch must be positive")
        if self.ldlq_sequences < 1 or not 0 <= self.ldlq_damp < 1 or self.ldlq_min_tokens < 0 or self.tcq_score_rows < 0:
            raise ValueError("plan: ldlq_sequences >= 1, 0 <= ldlq_damp < 1, ldlq_min_tokens >= 0, tcq_score_rows >= 0")

    def _check_allocation(self) -> None:
        from suc.models.adapters import ROLES

        a = self.allocation
        if self.mode != "greedy":
            raise ValueError("plan.allocation is only valid with mode 'greedy'")
        if not isinstance(a, dict) or not a or set(a) - {"budget_bytes", "exclude", "role_budgets"}:
            raise ValueError("plan.allocation must be a non-empty mapping with keys 'budget_bytes', 'exclude' and / or "
                             "'role_budgets'")
        if "budget_bytes" in a and self.budget_bytes is None:
            raise ValueError("plan.allocation.budget_bytes is only valid with plan.budget_bytes")
        if "role_budgets" in a:
            self._check_role_budgets(a["role_budgets"], "plan.allocation.role_budgets")
        if "budget_bytes" in a:
            v = a["budget_bytes"]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                raise ValueError(f"plan.allocation.budget_bytes must be a positive number, got {v!r} "
                                 "(YAML: write 108750000000, not 108.75e9)")
        ex = a.get("exclude", {})
        if not isinstance(ex, dict):
            raise ValueError("plan.allocation.exclude must be a mapping {candidate name: [roles]}")
        names = {c.name for c in self.candidates}
        for k, v in ex.items():
            if k not in names:
                raise ValueError(f"plan.allocation.exclude: {k!r} is not a candidate ({sorted(names)})")
            if not isinstance(v, (list, tuple)) or not v or any(r not in ROLES for r in v):
                raise ValueError(f"plan.allocation.exclude[{k!r}] must be a non-empty list of roles from {ROLES}")

    def _check_role_budgets(self, rb, where: str) -> None:
        from suc.models.adapters import ROLES

        if self.mode != "greedy" or not isinstance(rb, (list, tuple)) or not rb:
            raise ValueError(f"{where} must be a non-empty list and is only valid with mode 'greedy'")
        seen: set = set()
        for i, g in enumerate(rb):
            if not isinstance(g, dict) or set(g) != {"roles", "budget_bpw"}:
                raise ValueError(f"{where}[{i}] must be a mapping with exactly 'roles' and 'budget_bpw'")
            roles, bpw = g["roles"], g["budget_bpw"]
            if not isinstance(roles, (list, tuple)) or not roles or any(r not in ROLES for r in roles):
                raise ValueError(f"{where}[{i}].roles must be a non-empty list of roles from {ROLES}")
            if seen & set(roles):
                raise ValueError(f"{where}: roles {sorted(seen & set(roles))} appear in two groups")
            seen |= set(roles)
            if isinstance(bpw, bool) or not isinstance(bpw, (int, float)) or bpw <= 0:
                raise ValueError(f"{where}[{i}].budget_bpw must be a positive number")

    def _check_role_codecs(self) -> None:
        from suc.models.adapters import ROLES

        rc = self.role_codecs
        if self.mode != "uniform" or not isinstance(rc, dict) or not rc:
            raise ValueError("plan.role_codecs must be a non-empty mapping and is only valid with mode 'uniform'")
        names = {c.name for c in self.candidates}
        head_names = names | ({self.embedding_codec.name} if self.embedding_codec is not None else set())
        seen: set = set()
        for k, v in rc.items():
            if not isinstance(v, (list, tuple)) or not v or any(r not in ROLES for r in v):
                raise ValueError(f"plan.role_codecs[{k!r}] must be a non-empty list of roles from {ROLES}")
            for r in v:
                ok = head_names if r in ("embed", "lm_head") else names
                if k not in ok:
                    raise ValueError(f"plan.role_codecs: {k!r} is not a codec usable for role {r!r} ({sorted(ok)})")
            if seen & set(v):
                raise ValueError(f"plan.role_codecs: roles {sorted(seen & set(v))} appear under two codecs")
            seen |= set(v)
        if seen & {"embed", "lm_head"} and self.embedding_codec is None:
            raise ValueError("plan.role_codecs for 'embed' / 'lm_head' needs an embedding_codec (the default)")

    def uniform_codec(self, role: str) -> str:
        """Codec name of a tensor of `role` in a uniform plan: plan.role_codecs override, else plan.codec
        (embedding / LM head: else embedding_codec)."""
        for k, v in (self.role_codecs or {}).items():
            if role in v:
                return k
        if role in ("embed", "lm_head"):
            return self.embedding_codec.name
        return self.codec

    def allocation_budget_bytes(self) -> float | None:
        """The total-size budget the greedy plan allocates: plan.allocation.budget_bytes if given, else budget_bytes."""
        a = self.allocation or {}
        return float(a["budget_bytes"]) if "budget_bytes" in a else self.budget_bytes

    def allocation_role_budgets(self) -> tuple:
        """The role groups the greedy plan allocates separately: plan.allocation.role_budgets if given, else
        role_budgets (empty: none)."""
        a = self.allocation or {}
        return tuple(a["role_budgets"]) if "role_budgets" in a else tuple(self.role_budgets or ())

    def excluded(self, candidate: str, role: str) -> bool:
        """plan.allocation.exclude: the greedy may not choose `candidate` for a tensor of `role`."""
        return role in ((self.allocation or {}).get("exclude") or {}).get(candidate, ())

    def codec_configs(self) -> dict[str, CodecConfig]:
        """Every codec config of the plan by name (candidates, head candidates, embedding codec)."""
        every = list(self.candidates) + list(self.head_candidates or ()) + \
            ([self.embedding_codec] if self.embedding_codec is not None else [])
        return {c.name: c for c in every}


@dataclass(frozen=True)
class PreflightConfig:
    """Checks before any heavy work (suc.pipeline.compress.preflight)."""
    require_pinned_revision: bool = True      # model.revision must be a 40-hex commit unless local_path is given
    fail_on_memory: bool = True               # estimated peak > memory_fraction × physical RAM is an error
    memory_fraction: float = 0.85
    disk_margin: float = 1.25                 # free disk must exceed the estimated need × disk_margin
    # Inference-side check of a plan.budget_bytes plan: weights (<= budget_bytes) + KV cache at runtime_context tokens
    # + runtime_activation_bytes must fit runtime_budget_bytes (the memory the SUC1 runtime may use on the target Mac).
    runtime_budget_bytes: float | None = None
    runtime_context: int = 32768
    runtime_activation_bytes: float = 2e9
    # Expert-streaming mode (suc.runtime.expert_streaming): a model LARGER than runtime_budget_bytes is accepted when
    # everything but the routed experts + KV + activations fits, and the rest of the budget (the expert cache) holds
    # at least expert_cache_min_layers layers' worth of top-k experts; the least-used experts are read from SSD.
    # The container stage then also writes hot_experts.json (calibration routing counts) for the runtime's pinned set.
    expert_streaming: bool = False
    expert_cache_min_layers: int = 2

    def __post_init__(self) -> None:
        if not 0 < self.memory_fraction <= 1 or self.disk_margin < 1:
            raise ValueError("preflight: memory_fraction must be in (0, 1] and disk_margin >= 1")
        for k in ("runtime_budget_bytes", "runtime_activation_bytes"):
            v = getattr(self, k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
                raise ValueError(f"preflight.{k} must be a number, got {v!r} (YAML: write 111000000000, not 111e9)")
        if self.runtime_budget_bytes is not None and (self.runtime_budget_bytes <= 0 or self.runtime_context < 1):
            raise ValueError("preflight: runtime_budget_bytes and runtime_context must be positive")
        if self.expert_streaming and (self.runtime_budget_bytes is None or self.expert_cache_min_layers < 1):
            raise ValueError("preflight: expert_streaming needs runtime_budget_bytes and expert_cache_min_layers >= 1")


@dataclass(frozen=True)
class NgramCodecConfig:
    """How `suc compress` stores row-gather lookup tables (qwen4_exp n-gram shards; adapter.table_tensors).

    raw: the checkpoint bytes (BF16), memory-mapped by the runtime. row_affine: the record `suc repack` produces
    (suc.io.repack / suc.quant.row_affine: one affine group per row, group_size = row width, scheme row_minmax,
    scale / bias in scale_dtype), written by the container stage one shard at a time (D8)."""
    kind: Literal["raw", "row_affine"] = "raw"
    bits: int = 4
    scale_dtype: Literal["f16", "bf16"] = "f16"

    def __post_init__(self) -> None:
        from suc.quant.row_affine import ROW_BITS

        if self.kind not in ("raw", "row_affine"):
            raise ValueError(f"ngram_codec.kind must be 'raw' or 'row_affine', got {self.kind!r}")
        if isinstance(self.bits, bool) or self.bits not in ROW_BITS:
            raise ValueError(f"ngram_codec.bits must be one of {ROW_BITS}, got {self.bits!r}")
        if self.scale_dtype not in ("f16", "bf16"):
            raise ValueError(f"ngram_codec.scale_dtype must be f16 or bf16, got {self.scale_dtype!r}")


@dataclass(frozen=True)
class CompressExperimentConfig:
    """`suc compress`: calibrate → (Fisher candidates) → plan → encode → (fine-tune) → SUC1 → evaluate → report.

    The KL / fine-tuning / Fisher sequences come from this run's own calibration stage, so eval.calib_artifact
    must stay unset. run_dir holds the resumable stage artifacts (default experiments/artifacts/compress/<name>)."""
    name: str
    model: ModelConfig
    plan: CompressPlanConfig
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    eval: QuantEvalConfig = field(default_factory=QuantEvalConfig)
    finetune: FinetuneSpec | None = None
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    ngram_codec: NgramCodecConfig = field(default_factory=NgramCodecConfig)   # default raw: fingerprints unchanged
    rotation_groups: bool = False             # shared input rotation for q/k/v and gate/up (runtime fusion, K5)
    verify_tensors: int = 16                  # encoded tensors re-read from the SUC1 file and compared bit-exactly
    seed: int = 12345
    output_dir: str = "experiments/results"
    run_dir: str | None = None

    def __post_init__(self) -> None:
        if self.eval.calib_artifact is not None:
            raise ValueError("eval.calib_artifact must not be set: suc compress uses its own calibration stage")
        if self.eval.kl_sequences > self.calibration.n_validation:
            raise ValueError(f"eval.kl_sequences {self.eval.kl_sequences} > calibration.n_validation "
                             f"{self.calibration.n_validation}")
        if self.verify_tensors < 1:
            raise ValueError("verify_tensors must be >= 1")
        if self.finetune is not None and any(c.codebook_scope == "expert_stack"
                                             for c in self.plan.codec_configs().values()):
            raise ValueError("finetune is not supported with codebook_scope 'expert_stack' yet (per-expert tuning "
                             "would un-share the stack codebooks)")


def _codec_list(v: Any, where: str) -> tuple[CodecConfig, ...]:
    if not isinstance(v, (list, tuple)):
        raise ValueError(f"config: {where!r} must be a list")
    return tuple(_build(CodecConfig, c, f"{where}[{i}]") for i, c in enumerate(v))


def compress_config_from_dict(d: dict) -> CompressExperimentConfig:
    if not isinstance(d, dict):
        raise ValueError("config: expected a mapping")
    d = dict(d)
    for key in ("model", "plan"):
        if key not in d:
            raise ValueError(f"config: missing required key(s) [{key!r}]")
    plan = d.pop("plan")
    if not isinstance(plan, dict):
        raise ValueError("config: 'plan' must be a mapping")
    plan = dict(plan)
    if "candidates" not in plan:
        raise ValueError("plan: missing required key(s) ['candidates']")
    emb, head = plan.pop("embedding_codec", None), plan.pop("head_candidates", None)
    plan_nested = {"candidates": _codec_list(plan.pop("candidates"), "plan.candidates"),
                   "embedding_codec": None if emb is None else _build(CodecConfig, emb, "plan.embedding_codec"),
                   "head_candidates": None if head is None else _codec_list(head, "plan.head_candidates")}
    ft = d.pop("finetune", None)
    nested = {
        "model": _build(ModelConfig, d.pop("model"), "model"),
        "plan": _build(CompressPlanConfig, plan, "plan", **plan_nested),
        "calibration": _build(CalibrationConfig, d.pop("calibration", {}) or {}, "calibration"),
        "eval": _build(QuantEvalConfig, d.pop("eval", {}) or {}, "eval"),
        "finetune": None if ft is None else _build(FinetuneSpec, ft, "finetune"),
        "preflight": _build(PreflightConfig, d.pop("preflight", {}) or {}, "preflight"),
        "ngram_codec": _build(NgramCodecConfig, d.pop("ngram_codec", {}) or {}, "ngram_codec"),
    }
    return _build(CompressExperimentConfig, d, "config", **nested)


def load_compress_config(path: str | Path) -> CompressExperimentConfig:
    with open(path) as f:
        return compress_config_from_dict(yaml.safe_load(f) or {})


# ---------------------------------------------------------------- suc eval / suc perf (runtime comparison tools)

_SYSTEM_KINDS = ("bf16", "mlx_affine", "suc1_cpp", "suc1_python", "ds4", "openai_server")


@dataclass(frozen=True)
class SystemConfig:
    """One inference system compared by `suc eval` / `suc perf`.

    bf16 / mlx_affine: the reference checkpoint (model) through mlx-lm, as loaded (bf16) or MLX affine RTN
    (suc.evaluation.baselines.apply_representation). suc1_cpp: a SUC1 file through the C++ tool suc1-generate in
    cpp_build (extra command-line flags in cpp_args, e.g. ["--projections", "separate"]). suc1_python: the same file
    through the Python/MLX runtime (suc.runtime.suc1_model.load_suc1_model; python_plan fused = grouped / fused
    projections, the default runtime and the C++ default plan; separate = one module per projection).
    ds4: a GGUF file (`file`) through the ds4 engine (`scripts/gguf/ds4_score.c` serve mode, built into ds4_dir by
    scripts/gguf/build_ds4_score.sh; extra engine flags in engine_args, e.g. SSD streaming; KV context ctx), fed the
    same prompt token ids as every other system. openai_server: an OpenAI-compatible HTTP server (LM Studio `lms
    server`, llama-server, mlx_lm.server) at base_url serving server_model, through /v1/completions with the prompt
    text rendered by suc (the detokenized prompt ids, chat template included), so the server tokenizes the same text;
    the server's prompt token count is checked against the ids per request (suc.evaluation.runners
    OpenAIServerRunner)."""
    name: str
    kind: Literal["bf16", "mlx_affine", "suc1_cpp", "suc1_python", "ds4", "openai_server"]
    file: str | None = None
    bits: int | None = None
    group_size: int | None = None
    cpp_build: str = "~/suc-local/cpp-build"
    cpp_args: tuple[str, ...] = ()
    python_plan: Literal["fused", "separate"] = "fused"
    ds4_dir: str = "~/Projects/ds4"
    engine_args: tuple[str, ...] = ()
    ctx: int = 16384
    base_url: str | None = None
    server_model: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _SYSTEM_KINDS:
            raise ValueError(f"system {self.name!r}: unknown kind {self.kind!r}")
        needs_file = self.kind.startswith("suc1_") or self.kind == "ds4"
        if needs_file != (self.file is not None):
            raise ValueError(f"system {self.name!r}: 'file' is required for suc1_* / ds4 kinds and invalid otherwise")
        if self.engine_args and self.kind != "ds4":
            raise ValueError(f"system {self.name!r}: engine_args is only valid for ds4")
        server = self.kind == "openai_server"
        if server != (self.base_url is not None and self.server_model is not None):
            raise ValueError(f"system {self.name!r}: base_url + server_model are required for openai_server and "
                             "invalid otherwise")
        if self.ctx < 1:
            raise ValueError(f"system {self.name!r}: ctx must be positive")
        if self.kind == "mlx_affine":
            if self.bits not in _AFFINE_BITS:
                raise ValueError(f"system {self.name!r}: mlx_affine requires bits in {_AFFINE_BITS}")
            if self.group_size is None:
                object.__setattr__(self, "group_size", _DEFAULT_GROUP_SIZE)
        elif self.bits is not None or self.group_size is not None:
            raise ValueError(f"system {self.name!r}: bits/group_size are only valid for mlx_affine")
        if self.cpp_args and self.kind != "suc1_cpp":
            raise ValueError(f"system {self.name!r}: cpp_args is only valid for suc1_cpp")


def _check_systems(systems: tuple[SystemConfig, ...]) -> None:
    if not systems:
        raise ValueError("config: 'systems' must be a non-empty list")
    names = [s.name for s in systems]
    if len(set(names)) != len(names):
        raise ValueError(f"config: duplicate system names {names}")


@dataclass(frozen=True)
class EvalExperimentConfig:
    """`suc eval`: answer a test suite (experiments/eval/*.json) with every system, greedy, and grade it.

    The core is the automatic scoring of each system (typically one SUC1 file) on the suite; the comparison with a
    reference system is optional. model: the checkpoint the systems derive from (weights only for bf16 / mlx_affine
    systems) and the tokenizer / chat template of every system. reference: name of a system the others are compared
    with per case (same answer, identical output, common token prefix, teacher-forced KL / top-1 on the reference
    output); None (default) = no comparison, every system is only graded. case_ids / limit select cases (ids, or
    the first `limit` in suite order). max_tokens caps every case's own budget; think: per case, or forced on / off.
    think_reply_reserve: thinking budget control (ds4-eval's hard limit): a thinking case that has not closed
    </think> with this many tokens of budget left gets </think> forced and keeps the reserve for the visible answer
    (0 = off). teacher_forced_positions: with a reference, per case the first N tokens of the reference output are
    scored by every other system (next-token KL vs the reference, top-1 agreement; 0 = off). compare_to: a previous `suc eval` result
    JSON; per system and case, changes of state / answer / generated tokens are reported (ds4-eval's deterministic
    token-count gate)."""
    name: str
    model: ModelConfig
    suite: str
    systems: tuple[SystemConfig, ...]
    reference: str | None = None
    case_ids: tuple[str, ...] | None = None
    limit: int | None = None
    max_tokens: int | None = None
    think: Literal["case", "on", "off"] = "case"
    think_reply_reserve: int = 256
    teacher_forced_positions: int = 128
    compare_to: str | None = None
    seed: int = 12345
    output_dir: str = "experiments/results"

    def __post_init__(self) -> None:
        _check_systems(self.systems)
        if self.reference is not None and self.reference not in [s.name for s in self.systems]:
            raise ValueError(f"reference {self.reference!r} is not a system name")
        if self.limit is not None and self.limit < 1 or self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("limit / max_tokens must be positive")
        if self.think_reply_reserve < 0 or self.teacher_forced_positions < 0:
            raise ValueError("think_reply_reserve / teacher_forced_positions must be >= 0")


@dataclass(frozen=True)
class PerfExperimentConfig:
    """`suc perf`: prefill / TTFT / decode speed and memory of every system, in one session, sequentially.

    One fixed token sequence (prompt_file, tokenized raw, as ds4-bench --prompt-file) is cut at each context length
    (contexts, or ds4-bench frontiers ctx_start → ctx_max by step_incr or ×step_mul); per (system, context, batch):
    `warmup` untimed runs, then `repeats` timed runs of a full prefill of the context + gen_tokens greedy tokens (EOS
    not honoured: fixed length). Batch b uses b different windows of the sequence. mlx_decode: pipelined (mlx-lm
    generate_step: the next step is queued with async_eval before the current token is read, like the C++
    pipelined decode) or sync (one blocking eval per token). reference: the system the speed ratios are taken
    against (default: the first). compare_to: a previous `suc perf` result JSON (row-by-row ratios)."""
    name: str
    model: ModelConfig
    systems: tuple[SystemConfig, ...]
    prompt_file: str = "experiments/perf/promessi_sposi_head.txt"
    contexts: tuple[int, ...] | None = None
    ctx_start: int | None = None
    ctx_max: int | None = None
    step_incr: int | None = None
    step_mul: float | None = None
    gen_tokens: int = 128
    batch_sizes: tuple[int, ...] = (1,)
    warmup: int = 1
    repeats: int = 3
    mlx_decode: Literal["pipelined", "sync"] = "pipelined"
    prefill_step_size: int = 2048
    reference: str | None = None
    compare_to: str | None = None
    seed: int = 12345
    output_dir: str = "experiments/results"

    def __post_init__(self) -> None:
        _check_systems(self.systems)
        if self.reference is not None and self.reference not in [s.name for s in self.systems]:
            raise ValueError(f"reference {self.reference!r} is not a system name")
        sweep = (self.ctx_start, self.ctx_max)
        if (self.contexts is None) == (None in sweep):
            raise ValueError("perf: give either contexts or ctx_start + ctx_max (+ step_incr or step_mul)")
        if self.contexts is not None and (not self.contexts or min(self.contexts) < 1):
            raise ValueError("perf: contexts must be a non-empty list of positive lengths")
        if self.contexts is None:
            if (self.step_incr is None) == (self.step_mul is None):
                raise ValueError("perf: give exactly one of step_incr / step_mul with ctx_start / ctx_max")
            if self.ctx_start < 1 or self.ctx_max < self.ctx_start:
                raise ValueError("perf: need 1 <= ctx_start <= ctx_max")
            if self.step_incr is not None and self.step_incr < 1 or self.step_mul is not None and self.step_mul <= 1:
                raise ValueError("perf: step_incr must be >= 1 and step_mul > 1")
        if self.gen_tokens < 2 or self.warmup < 0 or self.repeats < 1 or not self.batch_sizes \
                or min(self.batch_sizes) < 1 or self.prefill_step_size < 1:
            raise ValueError("perf: gen_tokens >= 2, warmup >= 0, repeats >= 1, positive batch sizes required")

    def frontiers(self) -> list[int]:
        """Context lengths of the sweep (ds4-bench next_frontier: linear or geometric, ctx_max always included)."""
        if self.contexts is not None:
            return sorted(set(self.contexts))
        out, cur = [], self.ctx_start
        while True:
            out.append(cur)
            if cur >= self.ctx_max:
                return out
            nxt = cur + self.step_incr if self.step_incr is not None else max(cur + 1, math.ceil(cur * self.step_mul))
            cur = min(nxt, self.ctx_max)


def _tool_config(cls: type, d: dict):
    if not isinstance(d, dict):
        raise ValueError("config: expected a mapping")
    d = dict(d)
    for key in ("model", "systems"):
        if key not in d:
            raise ValueError(f"config: missing required key(s) [{key!r}]")
    systems = d.pop("systems")
    if not isinstance(systems, (list, tuple)):
        raise ValueError("config: 'systems' must be a list")
    nested = {"model": _build(ModelConfig, d.pop("model"), "model"),
              "systems": tuple(_build(SystemConfig, s, f"systems[{i}]") for i, s in enumerate(systems))}
    return _build(cls, d, "config", **nested)


def eval_config_from_dict(d: dict) -> EvalExperimentConfig:
    return _tool_config(EvalExperimentConfig, d)


def load_eval_config(path: str | Path) -> EvalExperimentConfig:
    with open(path) as f:
        return eval_config_from_dict(yaml.safe_load(f) or {})


def perf_config_from_dict(d: dict) -> PerfExperimentConfig:
    return _tool_config(PerfExperimentConfig, d)


def load_perf_config(path: str | Path) -> PerfExperimentConfig:
    with open(path) as f:
        return perf_config_from_dict(yaml.safe_load(f) or {})
