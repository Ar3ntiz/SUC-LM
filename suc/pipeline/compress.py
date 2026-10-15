"""`suc compress <config.yaml>`: unattended full-precision HF checkpoint -> SUC1 file + quality report.

Stages (each writes its artifacts under run_dir/<stage>/ and a manifest; see suc.pipeline.stages for the skip rule):

  preflight  (always)  revision pinned, adapter validation, every planned codec can encode the shapes it gets
                       (block_size / group_size divide in_features, Hadamard size), peak-memory and disk
                       estimates, free-disk check.
  calibrate            calibration / validation / OOD partitions + layer-streamed activation statistics (Phase 2).
  reference            BF16 WikiText-2 PPL (final-test set) + BF16 logits on held-out validation sequences.
  candidates (greedy)  empirical Fisher + every candidate codec scored on every allocated tensor (Q7).
  plan                 ONE per-tensor codec plan: a uniform codec, or the greedy plan for budget_bpw
                       (allocate_plan is the single entry point into the allocator).
  encode               per-block encode with the planned codecs; one SUC1 shard per block, re-read and compared
                       bit-exactly; resumable at block granularity (progress.json).
  finetune (optional)  Q6 block-wise fine-tuning of codebooks / row scales / residual values, per-block shards.
  container            the final SUC1 file (all encodings + every other checkpoint tensor raw), reloaded with
                       checksum verification; a sample of tensors compared bit-exactly with the encodings, and the
                       reference decoder compared with the codec decode.
  evaluate             the model decoded FROM THE SUC1 FILE: PPL, KL vs BF16, actual size (all overhead).
  report               result JSON (+ md) via suc.provenance.write_result; metadata captured at the start.

Memory (D8). Every stage is layer-streamed: at most one transformer block of weights (plus embedding / LM head,
the hidden states of the stage's sequences and one micro-batch's working set) is resident.
  calibrate, encode    GPTQ-style block-sequential pass over a lazily loaded model (blocks released after use).
  reference, evaluate  suc.evaluation.streamed: one block-sequential sweep over the PPL windows and the KL
                       sequences; the LM head runs one micro-batch at a time; reference logits are written / read
                       in chunk-row slices (full bf16 logits on disk, no approximation). evaluate installs the
                       decoded weights of block i from the SUC1 file just before block i runs.
  candidates           exact streamed Fisher (suc.analysis.fisher.fisher_diag_streamed: forward sweep storing every
                       block input on disk, reverse sweep reloading one block at a time); the candidates of a block
                       are scored as soon as its F is final and F is dropped (never all F at once). Resumable.
  finetune             block encodings read from that block's encode shard, tuned shard written per block.
  container            suc.io.suc1.write_suc1_streamed: one tensor resident, byte-identical to write_suc1.
Hidden states that exceed HIDDEN_IN_MEMORY_MAX are spilled to the stage directory (memory-mapped). preflight
estimates each stage's peak memory and disk need and fails early.
"""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import math
import os
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from suc.allocation.greedy import Candidate, build_candidates, greedy_allocate, plan_summary
from suc.analysis.activations import Recorder
from suc.analysis.calibrate import collect_activation_stats, load_domain_docs
from suc.analysis.fisher import fisher_diag_streamed, fisher_proxy
from suc.analysis.sensitivity import get_weight, set_weight
from suc.cli.bench import resolve_model_dir, resolved_revision
from suc.config import CodecConfig, CompressExperimentConfig, config_hash, config_to_dict, load_compress_config
from suc.data.calibration import build_ood, build_partitions, partitions_manifest
from suc.evaluation.data import load_wikitext2, tokenize
from suc.evaluation.reconstruct import _EXPERT, _flush_experts, _same_bytes, model_size, reconstruct_model
from suc.evaluation.streamed import RefLogits, streamed_quality
from suc.io import repack
from suc.io.hashing import model_hash, sha256_file
from suc.io.safetensors_store import SafetensorsStore
from suc.io.suc1 import SucModel, write_suc1, write_suc1_streamed
from suc.models.adapters import LinearSpec, get_adapter
from suc.models.streaming import _release, load_lazy, model_blocks
from suc.pipeline.stages import StageRunner, atomic_path, atomic_write_json, atomic_write_text, fingerprint, log
from suc.provenance import run_metadata, seed_everything, write_result
from suc.quant.codecs import Encoded, make_codec
from suc.quant.hadamard import supported_size
from suc.quant import stacks
from suc.quant.residual import ResidualCodec, _ResidualEncoded
from suc.quant.stacks import set_codebook_scope
from suc.quant.run import _per_role, select_specs
from suc.pipeline.runlock import run_lock
from suc.quant.seeds import SEED_CONTRACT, accepts_rotation_seeds, rotation_group

ARTIFACT_ROOT = "experiments/artifacts/compress"
STAGE_VERSIONS = {"calibrate": 1, "reference": 1, "hessians": 1, "candidates": 1, "plan": 1, "encode": 1,
                  "finetune": 1, "container": 1, "evaluate": 1, "report": 1}
FULL_MODEL_STAGES: tuple[str, ...] = ()        # stages that load the whole BF16 model: none (all layer-streamed)
ALL_ENCODINGS_STAGES: tuple[str, ...] = ()     # stages that hold every encoding in memory: none
STREAMING_STAGES = ("calibrate", "reference", "hessians", "candidates", "encode", "finetune", "container", "evaluate")
HIDDEN_IN_MEMORY_MAX = 4 << 30                 # hidden states of a streamed sweep above this size go to disk
_LAYER = re.compile(r"^model\.layers\.(\d+)\.")
_HEAD_ROLES = ("embed", "lm_head")


class PreflightError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------- environment

def free_disk_bytes(path: str | Path) -> int:
    p = Path(path).resolve()
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def physical_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def config_delta(x):
    """Config view used in stage fingerprints: dataclass fields equal to their default are dropped (recursively).

    New config fields added later with a behaviour-preserving default then leave existing fingerprints, and the
    hours of finished stages behind them, valid. Changing what an existing default MEANS must bump STAGE_VERSIONS,
    or keep the old default as the field's metadata "fingerprint_default" (the value compared here instead):
    calibration.mix became v2 for new configs while fingerprints still measure against v1, so every v1 run keeps
    its fingerprints and a v2 run (explicit or by default) gets new ones.
    """
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        out = {}
        for f in dataclasses.fields(x):
            v = getattr(x, f.name)
            if "fingerprint_default" in f.metadata:
                default = f.metadata["fingerprint_default"]
            elif f.default is not dataclasses.MISSING:
                default = f.default
            elif f.default_factory is not dataclasses.MISSING:
                default = f.default_factory()
            else:
                default = dataclasses.MISSING
            if default is not dataclasses.MISSING and v == default:
                continue
            out[f.name] = config_delta(v)
        return out
    if isinstance(x, (list, tuple)):
        return [config_delta(v) for v in x]
    if isinstance(x, dict):
        return {k: config_delta(v) for k, v in x.items()}
    return x


def _load_artifact(path: Path) -> dict[str, mx.array]:
    """Read a safetensors artifact completely on the CPU stream. mx.load is lazy: a multi-GB file on a slow
    (network) disk would otherwise be read inside a Metal command buffer and trip the GPU watchdog
    (kIOGPUCommandBufferCallbackErrorTimeout, seen with the run dir on the SMB share)."""
    with mx.stream(mx.cpu):
        arrays = mx.load(str(path))
        mx.eval(arrays)
    return arrays


def _free(*objs) -> None:
    del objs
    gc.collect()
    mx.clear_cache()


def _numel(shape) -> int:
    return math.prod(shape)


def _gib(n: float) -> str:
    return f"{n / 2 ** 30:.2f} GiB"


_run_lock = run_lock     # flock-based, see suc/pipeline/runlock.py (audit F01)


# ------------------------------------------------------------------------------------------- context

@dataclass
class Ctx:
    cfg: CompressExperimentConfig
    model_dir: Path
    store: SafetensorsStore
    hf_config: dict
    adapter: object
    specs: list[LinearSpec]               # eligible block linears
    head_specs: list[LinearSpec]          # embedding (+ untied LM head) that get a codec
    model_hash: str
    revision: str
    tokenizer: object = None
    docs: dict | None = None
    ood_docs: list | None = None
    _recorder: Recorder | None = field(default=None, repr=False)

    @property
    def needs_stats(self) -> bool:
        return any(c.residual_fraction and c.residual_selection == "activation"
                   for c in self.cfg.plan.codec_configs().values())

    @property
    def tokenizer_id(self) -> str:
        tid = f"{self.cfg.model.id}@{self.revision}"
        tj = self.model_dir / "tokenizer.json"
        return tid + (f";tokenizer.json:sha256:{sha256_file(tj)}" if tj.exists() else "")


class CodecTable:
    """Codec object per tensor, identical for encode and decode. Activation-selected residual codecs fall back to
    magnitude selection for tensors without activation statistics (embedding / LM head), as in suc quantize."""

    def __init__(self, ctx: Ctx, plan: dict[str, str], stats_names: set[str]) -> None:
        self.cfgs = ctx.cfg.plan.codec_configs()
        self.ldlq_damp = ctx.cfg.plan.ldlq_damp
        self.plan, self.stats_names = plan, stats_names
        self._objs: dict[tuple[str, bool], object] = {}

    def obj(self, cname: str, fallback: bool):
        key = (cname, fallback)
        if key not in self._objs:
            c: CodecConfig = self.cfgs[cname]
            base = set_codebook_scope(make_codec(c.codec_dict()), c.codebook_scope)
            if c.kind == "tcq":                 # BlockLDLQ only for ldlq configs (reconstruct / build_candidates)
                base.ldlq, base.hessian_damp = bool(c.ldlq), float(self.ldlq_damp)
            if c.residual_fraction:
                sel = "magnitude" if fallback or c.residual_selection == "magnitude" else "importance"
                # the fallback keeps the residual structure (columns stay columns, selected by magnitude)
                res = ResidualCodec(base, c.residual_fraction, c.residual_dtype, sel, structure=c.residual_structure,
                                    column_share=c.residual_column_share)
                if c.kind == "tcq":
                    res.ldlq = bool(c.ldlq)
                self._objs[key] = set_codebook_scope(res, c.codebook_scope)
            else:
                self._objs[key] = base
        return self._objs[key]

    def needs_importance(self, cname: str) -> bool:
        c = self.cfgs[cname]
        return bool(c.residual_fraction) and c.residual_selection == "activation"

    def for_tensor(self, name: str, cname: str | None = None):
        cname = cname or self.plan[name]
        return self.obj(cname, self.needs_importance(cname) and name not in self.stats_names)

    def decode(self, name: str, e: Encoded) -> mx.array:
        return self.for_tensor(name).decode(e)


def _recorder(ctx: Ctx, runner: StageRunner) -> Recorder | None:
    if not ctx.needs_stats:
        return None
    if ctx._recorder is None:
        ctx._recorder = Recorder.load(runner.path("calibrate", "activations.safetensors"))
    return ctx._recorder


def _importance(ctx: Ctx, rec: Recorder | None, specs: list[LinearSpec]) -> dict[str, mx.array] | None:
    """Per-weight importance E[x_j^2] broadcast over rows (lazy views), for tensors with statistics."""
    if rec is None:
        return None
    return {s.name: mx.broadcast_to(rec.moments[s.name[:-7]].second_moment_diag()[None], ctx.store.info(s.name).shape)
            for s in specs if s.name[:-7] in rec.moments}


def _stats_names(ctx: Ctx, rec: Recorder | None) -> set[str]:
    return {s.name for s in ctx.specs + ctx.head_specs if rec is not None and s.name[:-7] in rec.moments}


def _same_encoded(a: Encoded | mx.array, b: Encoded | mx.array) -> bool:
    def same(x: mx.array, y: mx.array) -> bool:
        if x.dtype != y.dtype or tuple(x.shape) != tuple(y.shape):
            return False
        if x.dtype == mx.bfloat16:
            x, y = x.view(mx.uint16), y.view(mx.uint16)
        return bool(np.array_equal(np.array(x), np.array(y)))

    if isinstance(a, mx.array) or isinstance(b, mx.array):
        return isinstance(a, mx.array) and isinstance(b, mx.array) and same(a, b)
    return (type(a) is type(b) and a.codec == b.codec and tuple(a.shape) == tuple(b.shape) and a.params == b.params
            and sorted(a.arrays) == sorted(b.arrays) and all(same(a.arrays[k], b.arrays[k]) for k in a.arrays))


def _load_shards(files: list[Path], verify: bool = True) -> dict[str, Encoded]:
    out = {}
    for f in files:
        sm = SucModel.load(f, verify=verify)
        for n in sm.names():
            out[n] = sm.encoded(n)
    return out


# ------------------------------------------------------------------------------------------- preflight

def _est_bpw(c: CodecConfig, in_features: int, numel: int) -> float:
    """Rough upper estimate of a codec's actual bpw (preflight only; the report uses the real bits)."""
    if c.kind == "bf16":
        b = 16.0
    elif c.kind == "mxfp4":
        b = 4.0 + 8 / 32
    elif c.kind == "affine":
        b = c.bits + 32 / c.group_size
    elif c.kind == "scalar":
        b = math.ceil(math.log2(c.k)) + 16 / in_features
    elif c.kind == "tcq":
        b = c.tcq_bits + 16 / in_features
    else:
        b = c.num_codebooks * math.ceil(math.log2(c.k)) / c.block_size + 16 / in_features
    if c.residual_fraction:
        b += c.residual_fraction * (max(1, math.ceil(math.log2(numel))) + (8 if c.residual_dtype == "int8" else 16))
    return b + 0.01


def estimate_resources(ctx: Ctx) -> dict:
    """Peak-memory estimate per stage and the disk need of the run (bytes). Coarse by design: it must catch
    'does not fit' days before a long unattended run reaches the stage, not predict to the MiB."""
    cfg, store = ctx.cfg, ctx.store
    names = store.names()
    hc = ctx.adapter.text_config
    h = ctx.adapter.residual_width            # hidden state between blocks (4-stream residual for qwen4_exp)
    tables = ctx.adapter.table_tensors(names)  # memory-mapped lookup tables: never resident as block weights
    inter = int(hc.get("moe_intermediate_size") or hc.get("intermediate_size") or h)
    vocab = int(hc.get("vocab_size", 0))
    layers = int(hc.get("num_hidden_layers", 1))
    blocks: dict[int, int] = defaultdict(int)
    other = 0
    table_bytes = sum(store.info(n).nbytes for n in tables)
    for n in names:
        if n in tables:
            continue
        if (m := _LAYER.match(n)) is not None:
            blocks[int(m.group(1))] += store.info(n).nbytes
        else:
            other += store.info(n).nbytes
    total = store.total_bytes()
    largest = max(blocks.values(), default=0)
    largest_tensor = max(_numel(store.info(s.name).shape) for s in ctx.specs + ctx.head_specs) \
        if ctx.specs or ctx.head_specs else 0
    p = cfg.plan
    plan_codecs = p.codec_configs()

    def worst(cands, s: LinearSpec) -> float:
        shape = store.info(s.name).shape
        return max(_est_bpw(c, shape[-1], _numel(shape)) for c in cands)

    block_cands = [plan_codecs[p.codec]] if p.mode == "uniform" else list(p.candidates)
    head_cands = list(p.head_candidates or ()) or ([p.embedding_codec] if p.embedding_codec is not None else [])

    def allowed(s: LinearSpec) -> list:
        if p.mode == "uniform":
            return [plan_codecs[p.uniform_codec(s.role)]]
        if not p.candidate_roles:
            return block_cands
        return [plan_codecs[c] for c in _allowed_candidates(ctx, s.role)]

    enc_block = sum(_numel(store.info(s.name).shape) * worst(allowed(s), s) / 8 for s in ctx.specs)
    if p.mode == "greedy" and p.budget_bytes is not None:
        enc_block = min(enc_block, p.allocation_budget_bytes())  # the plan's whole file fits budget_bytes
    if p.role_codecs is not None and head_cands:
        head_cands = head_cands + [plan_codecs[c] for c in p.role_codecs]   # worst case over the head overrides
    enc_head = sum(_numel(store.info(s.name).shape) * worst(head_cands, s) / 8 for s in ctx.head_specs) \
        if head_cands else 0.0
    enc = enc_block + enc_head
    encoded_src = sum(store.info(s.name).nbytes for s in ctx.specs + (ctx.head_specs if head_cands else []))
    raw_other = total - encoded_src
    nc = cfg.ngram_codec
    qt = quantized_tables(ctx, names)          # stored row-affine instead of their checkpoint bytes
    raw_other -= sum(store.info(n).nbytes - repack.row_affine_bytes(store.info(n).shape, nc.bits) for n in qt)
    c, ev = cfg.calibration, cfg.eval
    act = c.micro_batch * c.seq_len * max(inter, h) * 4 * 4                       # one block forward, float32
    hidden_all = c.n_calibration * c.seq_len * h * 2 * 2                          # bf16 in + out of a block
    ref_logits = ev.kl_sequences * c.seq_len * vocab * 2
    # PPL: one window of ev.seq_len (bf16 logits + float32 copy + log-probs); KL: ev.chunk rows of calibration length
    eval_logits = max(ev.seq_len * vocab * 4 * 3, ev.chunk * c.seq_len * vocab * 4 * 4)
    # Layer-streamed sweeps (D8): one block of weights + embedding / head ("other") + the hidden states of all
    # sequences (spilled to disk above HIDDEN_IN_MEMORY_MAX: then one micro-batch) + one block's forward working set.
    inter_dense = int(hc.get("intermediate_size") or inter)
    heads_n = int(hc.get("num_attention_heads", 1))
    ppl_tok = ev.max_tokens or 350_000                  # WikiText-2 test is ~300k tokens for current tokenizers
    hidden_eval = (ppl_tok + ev.kl_sequences * c.seq_len) * h * 2
    hidden_eval_res = hidden_eval if hidden_eval <= HIDDEN_IN_MEMORY_MAX else 2 * ev.chunk * c.seq_len * h * 2
    t_eval = max(ev.seq_len, ev.chunk * c.seq_len)
    act_eval = t_eval * max(inter_dense, h) * 4 * 4 + heads_n * max(ev.seq_len, c.seq_len) ** 2 * 4
    mem = {
        "calibrate": largest + other + hidden_all + act,
        "reference": largest + other + hidden_eval_res + act_eval + eval_logits,
        "plan": 0,
        # float32 working copies of the largest tensor (w, w_hat, residual, importance, sort keys): measured
        # 3.36 GiB on Qwen3-0.6B, whose largest tensor is the 0.16 B-parameter embedding
        "encode": 4 * largest + other + enc_block / max(layers, 1) + 6 * 4 * largest_tensor,
        # one tensor + its packed payloads, the directory (~2 KiB per tensor), verification decodes (float32)
        "container": 2 * 4 * largest_tensor + 2048 * len(names),
        # + float32 decode temporaries of the largest encoded tensor
        "evaluate": largest + other + hidden_eval_res + act_eval + eval_logits + 3 * 4 * largest_tensor,
    }
    fisher_act = fisher_cot = 0
    if p.mode == "greedy":
        # reverse sweep: block weights + its bf16 gradients + float32 F accumulators (2x) of one block, one
        # micro-batch's backward through one block; head step: vocab x hidden float32 gradients; candidate scoring
        back = p.fisher_micro_batch * c.seq_len * (h * 16 + inter_dense * 8) * 2
        head_grad = 4 * vocab * h * 4 if p.head_candidates is not None else 0
        mem["candidates"] = other + 4 * largest + back + max(head_grad, 6 * 4 * largest_tensor)
        fisher_act = (layers + 1) * p.fisher_sequences * c.seq_len * h * 2      # every block input, on disk
        fisher_cot = 2 * p.fisher_sequences * c.seq_len * h * 2
    if cfg.finetune is not None:
        f = cfg.finetune
        mem["finetune"] = (enc_block / max(layers, 1) + enc_head + other + 8 * largest
                           + 3 * f.sequences * c.seq_len * h * 2 + f.micro_batch * c.seq_len * max(inter, h) * 4 * 8)
    ld_dense, ld_moe = ldlq_targets(ctx)
    hess_disk = hess_mem = 0
    if ld_dense or ld_moe:
        # per MoE layer: shared gate/up D² + per-expert down E · I² (float32); dense: in² per input group (upper bound:
        # per tensor); recording holds one block's accumulators twice (sum + one micro-batch's segmented product)
        n_exp = int(ctx.adapter.num_experts or 0)
        per_moe = 4 * (h * h + n_exp * inter * inter)
        dense_b = sum(4 * store.info(n + ".weight").shape[-1] ** 2 for n in ld_dense)
        hess_disk = len(ld_moe) * per_moe + dense_b
        hess_mem = largest + other + hidden_all + act + 2 * per_moe
        mem["hessians"] = hess_mem
    spill_eval = hidden_eval if hidden_eval > HIDDEN_IN_MEMORY_MAX else 0
    disk = {
        "hessians": hess_disk + (hidden_all if hess_disk else 0),
        "calibrate": (c.n_calibration + c.n_validation + c.n_ood) * c.seq_len * 4
                     + sum(store.info(s.name).shape[-1] for s in ctx.specs) * 4 * 8,
        "reference": ref_logits + spill_eval,
        "candidates": fisher_act + fisher_cot,
        "encode": enc,
        "finetune": enc_block if cfg.finetune is not None else 0,
        "container": 2 * (enc + raw_other),          # the file + the transient section files of the streamed writer
        "evaluate": spill_eval,
    }
    peak_stage = max(mem, key=mem.get)
    return {"memory": {"per_stage_bytes": {k: int(v) for k, v in mem.items()}, "peak_stage": peak_stage,
                       "peak_estimate_bytes": int(mem[peak_stage]), "largest_block_bf16_bytes": int(largest),
                       "checkpoint_bytes": int(total), "mmap_table_bytes": int(table_bytes),
                       "full_model_stages": [s for s in FULL_MODEL_STAGES if s in mem],
                       "all_encodings_stages": [s for s in ALL_ENCODINGS_STAGES if s in mem]},
            "disk": {"per_stage_bytes": {k: int(v) for k, v in disk.items()}, "need_bytes": int(sum(disk.values())),
                     "estimated_compressed_bytes": int(enc + raw_other)}}


def predicted_weight_bytes(ctx: Ctx) -> dict:
    """Model size a greedy plan will have (bytes, before running it): budget_bytes, or the fixed part (fixed_bits)
    + every role_budgets group at its bpw + the rest at budget_bpw; also the routed-expert share of it (their group's
    bpw, else the main bpw) and the largest single routed expert (3 projections at the worst allowed candidate)."""
    p, store = ctx.cfg.plan, ctx.store
    heads = _head_names_allocated(ctx)
    specs = ctx.specs + [s for s in ctx.head_specs if s.name in heads]
    fixed = fixed_bits(ctx, {s.name for s in specs})["total"]
    groups = list(p.allocation_role_budgets())

    def bpw_of(role: str) -> float | None:
        return next((g["budget_bpw"] for g in groups if role in g["roles"]), None)

    group_bits = rest_params = 0
    expert_params = 0
    for s in specs:
        n = _numel(store.info(s.name).shape)
        b = bpw_of(s.role)
        if b is None:
            rest_params += n
        else:
            group_bits += b * n
        if s.expert is not None:
            expert_params += n
    if p.budget_bytes is not None:
        total = p.allocation_budget_bytes()
        rest_bpw = (total * 8 - fixed - group_bits) / max(rest_params, 1)
    else:
        rest_bpw = p.budget_bpw
        total = (fixed + group_bits + rest_bpw * rest_params) / 8
    ebpw = bpw_of("expert_gate") or rest_bpw
    per_expert: dict[tuple, float] = defaultdict(float)
    for s in ctx.specs:
        if s.expert is not None:
            shape = store.info(s.name).shape
            worst = max(_est_bpw(p.codec_configs()[c], shape[-1], _numel(shape)) for c in
                        _allowed_candidates(ctx, s.role))
            per_expert[(s.layer, s.expert)] += _numel(shape) * worst / 8
    return {"total": total, "fixed": fixed / 8, "routed_experts": ebpw * expert_params / 8,
            "routed_expert_bpw": ebpw, "routed_expert_max_bytes": max(per_expert.values(), default=0.0)}


def _dir_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0


def check_revision(cfg: CompressExperimentConfig) -> None:
    if (cfg.preflight.require_pinned_revision and cfg.model.local_path is None
            and not re.fullmatch(r"[0-9a-f]{40}", cfg.model.revision)):
        raise PreflightError(f"model.revision {cfg.model.revision!r} is not pinned: give the 40-hex commit of "
                             f"{cfg.model.id} (or set preflight.require_pinned_revision: false)")


def _codec_shape_error(c: CodecConfig, shape) -> str | None:
    """Why codec c cannot encode a tensor of `shape` (the codecs would raise mid-run), or None."""
    inn = int(shape[-1])
    if c.kind in ("vq", "additive"):
        if inn % c.block_size:
            return f"in_features {inn} not divisible by block_size {c.block_size}"
        if c.rotate and not c.fold_rotation and not supported_size(inn):
            return f"no Hadamard rotation for in_features {inn}"
    if c.kind == "affine" and inn % c.group_size:
        return f"in_features {inn} not divisible by group_size {c.group_size}"
    if c.kind == "mxfp4" and inn % 32:
        return f"in_features {inn} not divisible by the MXFP4 group size 32"
    if c.kind == "tcq":
        if inn % c.tcq_T:
            return f"in_features {inn} not divisible by the trellis sequence length {c.tcq_T}"
        if c.rotate and not supported_size(inn):
            return f"no Hadamard rotation for in_features {inn}"
    return None


def codec_shape_errors(ctx: Ctx) -> list[str]:
    """Every (codec, tensor shape) pair the plan may encode that the codec cannot handle: the uniform codec or every
    greedy candidate on the eligible block linears, the embedding codec / head candidates on the embedding / head."""
    p = ctx.cfg.plan
    cfgs = p.codec_configs()
    block = [cfgs[p.codec]] if p.mode == "uniform" else list(p.candidates)
    head = list(p.head_candidates or ()) or ([p.embedding_codec] if p.embedding_codec is not None else [])
    errors: dict[tuple[str, str], list[str]] = defaultdict(list)
    for codecs, specs in ((block, ctx.specs), (head, ctx.head_specs)):
        for s in specs:
            shape = ctx.store.info(s.name).shape
            if p.role_codecs is not None:
                codecs = [cfgs[p.uniform_codec(s.role)]]
            for c in codecs:
                if (why := _codec_shape_error(c, shape)) is not None:
                    errors[(c.name, why)].append(s.name)
    return [f"codec {c} cannot encode {len(ns)} tensor(s), e.g. {ns[0]}: {why}" for (c, why), ns in errors.items()]


def runtime_memory_check(ctx: Ctx) -> tuple[dict | None, list[str], list[str]]:
    """Inference-side memory check of a greedy plan against preflight.runtime_budget_bytes: (record, errors,
    warnings). Resident mode: predicted weights + KV at runtime_context + activations must fit. With
    preflight.expert_streaming a larger model is accepted when the non-expert weights + KV + activations leave an
    expert cache of at least expert_cache_min_layers x top-k of the largest routed expert."""
    cfg = ctx.cfg
    pf, ad = cfg.preflight, ctx.adapter
    if pf.runtime_budget_bytes is None or cfg.plan.mode != "greedy":
        return None, [], []
    weights = predicted_weight_bytes(ctx)
    errors: list[str] = []
    warnings: list[str] = []
    kv = ad.kv_bytes_per_token() * pf.runtime_context + getattr(ad, "kv_fixed_bytes", lambda: 0)()
    need = weights["total"] + kv + pf.runtime_activation_bytes
    rec = {"weights_budget_bytes": int(weights["total"]), "weights_estimate": weights,
           "kv_bytes": int(kv), "context": pf.runtime_context,
           "activation_bytes": pf.runtime_activation_bytes, "need_bytes": int(need),
           "runtime_budget_bytes": pf.runtime_budget_bytes, "mode": "resident"}
    msg = (f"runtime memory: weights {weights['total'] / 1e9:.2f} GB + KV {kv / 1e9:.2f} GB "
           f"({pf.runtime_context} tokens) + activations {pf.runtime_activation_bytes / 1e9:.2f} GB = "
           f"{need / 1e9:.2f} GB > runtime_budget_bytes {pf.runtime_budget_bytes / 1e9:.2f} GB")
    if need > pf.runtime_budget_bytes and not pf.expert_streaming:
        errors.append(msg)
    elif need > pf.runtime_budget_bytes:
        # expert streaming: everything but the routed experts resident, the rest of the budget caches experts
        resident = weights["total"] - weights["routed_experts"]
        cache = pf.runtime_budget_bytes - resident - kv - pf.runtime_activation_bytes
        top_k = int(ad.text_config.get("num_experts_per_tok", 1))
        min_cache = pf.expert_cache_min_layers * top_k * weights["routed_expert_max_bytes"]
        rec.update(mode="expert_streaming", resident_bytes=int(resident),
                   routed_expert_bytes=int(weights["routed_experts"]), expert_cache_bytes=int(cache),
                   expert_cache_min_bytes=int(min_cache),
                   expert_cache_fraction=cache / max(weights["routed_experts"], 1))
        if cache < min_cache:
            errors.append(f"{msg}; expert streaming: non-expert weights {resident / 1e9:.2f} GB leave an expert "
                          f"cache of {cache / 1e9:.2f} GB < {pf.expert_cache_min_layers} layers x top-{top_k} "
                          f"experts ({min_cache / 1e9:.3f} GB)")
        else:
            warnings.append(f"{msg}: accepted in expert-streaming mode (expert cache {cache / 1e9:.2f} GB = "
                            f"{100 * cache / max(weights['routed_experts'], 1):.1f}% of the routed experts; the "
                            "rest is read from SSD at run time)")
    return rec, errors, warnings


def preflight(ctx: Ctx, run_dir: Path) -> dict:
    """Estimates + checks; writes run_dir/preflight.json; raises PreflightError (after writing) on failure."""
    cfg = ctx.cfg
    est = estimate_resources(ctx)
    ram = physical_memory_bytes()
    existing = _dir_bytes(run_dir)
    need_new = max(0, est["disk"]["need_bytes"] - existing)
    free = free_disk_bytes(run_dir)
    est["memory"]["physical_bytes"] = ram
    est["disk"].update(free_bytes=int(free), existing_run_dir_bytes=int(existing), need_new_bytes=int(need_new))
    errors = codec_shape_errors(ctx) + native_candidate_errors(ctx)
    if hasattr(ctx.store, "quant_names") and ctx.store.quant_names():
        # quantized-at-source checkpoint (D8 exception): the reference is the EXACT BF16 dequantization; verify it
        # from the scale tensors alone before any heavy work (cached per model hash: ~9 GB of scales on V4-Flash)
        cache = run_dir / "source_scales.json"
        rec = json.loads(cache.read_text()) if cache.exists() else {}
        if rec.get("model_hash") != ctx.model_hash:
            from suc.io.quantized_source import NotLossless
            try:
                rec = {"model_hash": ctx.model_hash, "ranges": ctx.store.check_quant_scales(), "lossless": True}
            except NotLossless as ex:
                rec = {"model_hash": ctx.model_hash, "lossless": False, "error": str(ex)}
            atomic_write_json(cache, rec)
        est["source_precision"] = {"quantized_tensors": len(ctx.store.quant_names()), **rec}
        if not rec.get("lossless"):
            errors.append(f"source dequantization is not exact: {rec.get('error')}")
    rt, rt_errors, rt_warnings = runtime_memory_check(ctx)
    if rt is not None:
        est["runtime"] = rt
    errors += rt_errors
    if rt_warnings:
        est.setdefault("warnings", []).extend(rt_warnings)
    if free < need_new * cfg.preflight.disk_margin:
        errors.append(f"insufficient disk at {run_dir}: need ~{_gib(need_new)} (x{cfg.preflight.disk_margin} margin), "
                      f"free {_gib(free)}")
    peak = est["memory"]["peak_estimate_bytes"]
    if ram is not None and peak > cfg.preflight.memory_fraction * ram:
        msg = (f"estimated peak memory {_gib(peak)} in stage {est['memory']['peak_stage']!r} exceeds "
               f"{cfg.preflight.memory_fraction:.0%} of physical memory ({_gib(ram)}); full-model stages: "
               f"{est['memory']['full_model_stages'] or 'none (all stages layer-streamed)'}")
        (errors if cfg.preflight.fail_on_memory else est.setdefault("warnings", [])).append(msg)
    rec = {"ok": not errors, "errors": errors, "model": cfg.model.id, "revision": ctx.revision,
           "model_hash": ctx.model_hash, **est}
    atomic_write_json(run_dir / "preflight.json", rec)
    m = est["memory"]
    log(f"preflight: {cfg.model.id}@{ctx.revision[:12]} checkpoint {_gib(m['checkpoint_bytes'])}, largest block "
        f"{_gib(m['largest_block_bf16_bytes'])}; est. peak memory {_gib(peak)} ({m['peak_stage']}), RAM "
        f"{_gib(ram) if ram else '?'}; disk need {_gib(need_new)}, free {_gib(free)}; full-model stages "
        f"{m['full_model_stages']}")
    for w in est.get("warnings", []):
        log(f"preflight WARNING: {w}")
    if errors:
        raise PreflightError("preflight failed: " + "; ".join(errors))
    return rec


# ------------------------------------------------------------------------------------------- stages

_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json", "vocab.txt",
                    "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
                    "chat_template.json", "encoding/*.py")


def calibration_input_identity(ctx: Ctx) -> dict:
    """Identity of the calibration inputs that model_hash does not cover (audit observation "resume input identity"):
    sha256 of the checkpoint's tokenizer / chat-template files and of the corpus definition (the pinned
    suc.data.sources specs, or the injected documents). Recorded in the calibrate manifest and checked whenever the
    stage is reused; NOT a fingerprint input, so finished runs from before this check stay valid (they are reported
    as unverified)."""
    h = hashlib.sha256()
    for pat in _TOKENIZER_FILES:
        for f in sorted(ctx.model_dir.glob(pat)):
            h.update(f.relative_to(ctx.model_dir).as_posix().encode() + b"\0")
            h.update(hashlib.sha256(f.read_bytes()).digest())
    if ctx.docs is not None:                    # injected corpus (tests): its content
        corpus = hashlib.sha256(json.dumps([ctx.docs, ctx.ood_docs], sort_keys=True, default=str).encode())
        corpus = corpus.hexdigest()
    else:                                       # v1 mix: the same value as before the extra domains existed
        from suc.data.sources import corpus_identity

        corpus = corpus_identity(ctx.cfg.calibration.mix)
    return {"tokenizer_sha256": h.hexdigest(), "corpus_sha256": corpus}


def _check_calibration_identity(ctx: Ctx, runner: StageRunner) -> None:
    rec = (runner.manifests["calibrate"].get("info") or {}).get("input_identity")
    now = calibration_input_identity(ctx)
    if rec is None:
        log("calibrate: tokenizer / corpus identity not recorded (stage ran before 2026-10-09): reused unverified")
    elif rec != now:
        bad = sorted(k for k in now if rec.get(k) != now[k])
        raise PreflightError(f"calibrate stage of {runner.root} was produced with a different {', '.join(bad)}; "
                             f"move {runner.stage_dir('calibrate')} away (or use another run_dir) to recalibrate")


def _stage_calibrate(ctx: Ctx, d: Path, fp: str) -> dict:
    cfg, c = ctx.cfg, ctx.cfg.calibration
    tok = ctx.tokenizer if ctx.tokenizer is not None else load_lazy(ctx.model_dir)[1]
    docs, ood, sources, hashes = ctx.docs, ctx.ood_docs, {}, {}
    if docs is None:
        all_docs, sources, hashes = load_domain_docs(tok, c.mix)
        ood = all_docs.pop("ood", [])
        docs = all_docs
    parts = build_partitions({k: docs[k] for k in c.mix}, tok, seq_len=c.seq_len, mix=c.mix, seed=cfg.seed,
                             n_sequences={"calibration": c.n_calibration, "validation": c.n_validation})
    parts["ood"] = build_ood(ood or [], tok, seq_len=c.seq_len, n_sequences=c.n_ood, seed=cfg.seed)
    manifest = partitions_manifest(parts, sources, hashes)
    data = d / "data.safetensors"
    with atomic_path(data) as tmp:
        mx.save_safetensors(str(tmp), {p: v["tokens"] for p, v in parts.items()})
    rec, stats = collect_activation_stats(ctx.model_dir, parts["calibration"]["tokens"], c.micro_batch,
                                          c.full_second_moment)
    acts = d / "activations.safetensors"
    with atomic_path(acts) as tmp:
        rec.save(tmp)
        os.replace(tmp.with_suffix(".json"), d / "activations.json")
    del rec
    _free()
    atomic_write_json(d / "calibration.json", {**manifest, "activation_stats": stats})
    return {"outputs": [data, acts, d / "activations.json", d / "calibration.json"],
            "info": {"partitions_hash": manifest["partitions_hash"], "activation_stats": stats,
                     "input_identity": calibration_input_identity(ctx)}}


def _ppl_tokens(ctx: Ctx) -> tuple[mx.array, str, str]:
    text, wiki_hash = load_wikitext2(ctx.cfg.eval.wikitext_split)
    toks = tokenize(load_lazy(ctx.model_dir)[1], text)
    digest = hashlib.sha256(np.asarray(toks, dtype=np.int32).astype("<i4").tobytes()).hexdigest()
    return toks, wiki_hash, digest


def _hidden_bytes(ctx: Ctx, n_tokens: int) -> int:
    """Bytes of the hidden states of n_tokens positions at the checkpoint dtype of the embedding."""
    h = ctx.adapter.residual_width
    e = "model.embed_tokens.weight"
    itemsize = ctx.store.info(e).nbytes // max(1, _numel(ctx.store.info(e).shape)) if e in ctx.store else 2
    return n_tokens * h * itemsize


def _eval_tokens(ev, ppl_tokens: mx.array, kl_tokens: mx.array) -> int:
    n = ppl_tokens.size if ev.max_tokens is None else min(ppl_tokens.size, ev.max_tokens)
    return (n // (ev.seq_len + 1)) * ev.seq_len + kl_tokens.size


def _spill(ctx: Ctx, d: Path, n_tokens: int) -> Path | None:
    """Spill directory for the hidden states of a streamed sweep, or None when they fit HIDDEN_IN_MEMORY_MAX."""
    return d / "hidden" if _hidden_bytes(ctx, n_tokens) > HIDDEN_IN_MEMORY_MAX else None


def _stage_reference(ctx: Ctx, runner: StageRunner, d: Path, fp: str) -> dict:
    """BF16 PPL + reference logits, layer-streamed (suc.evaluation.streamed): bit-identical to the whole-model
    evaluate_quality / reference_logits, with one block of weights resident."""
    ev = ctx.cfg.eval
    ppl_tokens, wiki_hash, tok_sha = _ppl_tokens(ctx)
    kl_tokens = _load_artifact(runner.path("calibrate", "data.safetensors"))["validation"][: ev.kl_sequences]
    model = load_lazy(ctx.model_dir)[0]
    lp = d / "ref_logits.safetensors"
    with atomic_path(lp) as tmp:
        r = streamed_quality(model, ppl_tokens=ppl_tokens, seq_len=ev.seq_len, max_tokens=ev.max_tokens,
                             kl_tokens=kl_tokens, chunk=ev.chunk, ref_out=tmp,
                             spill_dir=_spill(ctx, d, _eval_tokens(ev, ppl_tokens, kl_tokens)))
    _free(model)
    quality = {"perplexity": r["perplexity"], "kl": None}
    rec = {"quality": quality, "wikitext_split": ev.wikitext_split, "wikitext_hash": wiki_hash,
           "ppl_tokens": int(ppl_tokens.size), "ppl_tokens_sha256": tok_sha,
           "kl_sequences": int(kl_tokens.shape[0]), "kl_seq_len": int(kl_tokens.shape[1]),
           "streaming": r["streaming"]}
    atomic_write_json(d / "reference.json", rec)
    log(f"reference: BF16 PPL {quality['perplexity']['ppl']:.3f}")
    return {"outputs": [lp, d / "reference.json"], "info": {"ppl": quality["perplexity"]["ppl"]}}


def _head_names_allocated(ctx: Ctx) -> list[str]:
    return [s.name for s in ctx.head_specs] if ctx.cfg.plan.head_candidates is not None else []


def _stage_candidates(ctx: Ctx, runner: StageRunner, d: Path, fp: str) -> dict:
    """Exact empirical Fisher, layer-streamed (fisher_diag_streamed), and every candidate codec scored on every
    allocated tensor. The candidates of a block are scored as soon as its F is final (reverse block order, then
    the embedding / LM head) and written to parts/<group>.json; F itself is never stored, so neither memory nor
    disk ever holds more than one block's F. Resumable: finished parts and the Fisher sweep's block progress
    (fisher_work/progress.json) survive a crash; everything is keyed by the stage fingerprint.
    plan.fisher_group_size is not used any more (the streamed sweep is already one block at a time)."""
    cfg, p = ctx.cfg, ctx.cfg.plan
    marker = d / "fingerprint.json"
    if not marker.exists() or json.loads(marker.read_text()).get("fingerprint") != fp:
        for f in d.iterdir():
            f.unlink() if f.is_file() else shutil.rmtree(f)
        atomic_write_json(marker, {"fingerprint": fp})
    parts = d / "parts"
    parts.mkdir(exist_ok=True)
    calib = _load_artifact(runner.path("calibrate", "data.safetensors"))["calibration"][: p.fisher_sequences]
    heads = _head_names_allocated(ctx)
    block_names = [s.name for s in ctx.specs]
    rec = _recorder(ctx, runner)
    table = CodecTable(ctx, {}, _stats_names(ctx, rec))
    importance = _importance(ctx, rec, ctx.specs)
    layer_of = {s.name: s.layer for s in ctx.specs}
    role_of = {s.name: s.role for s in ctx.specs}
    hstore = _hessian_store(ctx, runner)
    reuse = _candidates_reuse(ctx, runner)
    if reuse is not None:
        log(f"candidates: reusing the scores of {sorted(reuse[1])} from {reuse[0]}")

    def loss_fn(n, w, w_hat, rows=None, fisher=None):
        f = fisher[n] if rows is None else fisher[n][mx.array(np.asarray(rows, np.int32))]
        return fisher_proxy(w_hat.astype(mx.float32) - w.astype(mx.float32), f)

    def score(group: str, fisher: dict[str, mx.array]) -> None:
        out = parts / f"{group}.json"
        if out.exists():                       # delivered again after a crash: already scored
            return
        lf = lambda n, w, w_hat, rows=None: loss_fn(n, w, w_hat, rows, fisher)  # noqa: E731
        old = {}
        if reuse is not None and group != "head" and (reuse[0] / f"{group}.json").exists():
            old = json.loads((reuse[0] / f"{group}.json").read_text())
        cands: dict[str, list[Candidate]] = {}
        if group == "head":
            codecs = {c.name: table.obj(c.name, table.needs_importance(c.name)) for c in p.head_candidates}
            cands.update(build_candidates(ctx.store.load, heads, codecs, lf, seed=cfg.seed,
                                          rotation_groups=cfg.rotation_groups))
        else:
            i = int(group.split("_")[1])
            names = [n for n in block_names if layer_of[n] == i]
            split: dict[tuple, list[str]] = defaultdict(list)     # (stats?, allowed candidates) -> names
            for n in names:
                split[(n in table.stats_names, _allowed_candidates(ctx, role_of[n], scored=True))].append(n)
            for (with_stats, allowed), ns in split.items():
                fallback = not with_stats
                umap = _score_map(ctx, ns)
                units = list(umap)
                keys = [m for u in units for m in (umap[u] if p.allocation_unit == "expert" else [u])]
                reused = {c for c in allowed if reuse is not None and c in reuse[1]
                          and all(k in old and any(x["codec"] == c for x in old[k]) for k in keys)
                          and (reuse[2] or not _seed_contract_matters(ctx, table.obj(c, False), keys))}
                codecs = {c: table.obj(c, fallback and table.needs_importance(c)) for c in allowed if c not in reused}
                got = build_candidates(ctx.store.load, units, codecs, lf, seed=cfg.seed,
                                       importance=importance, log_every=64, stacks=umap,
                                       per_member=p.allocation_unit == "expert", hessians=hstore,
                                       row_sample=p.tcq_score_rows, rotation_groups=cfg.rotation_groups) if codecs else {}
                for k in keys:
                    have = {c.codec: c for c in got.get(k, [])}
                    have.update({x["codec"]: Candidate(**x) for x in old.get(k, []) if x["codec"] in reused})
                    cands[k] = [have[c] for c in allowed]
        atomic_write_json(out, {n: [asdict(c) for c in cs] for n, cs in cands.items()})
        log(f"candidates {group}: {len(cands)} units scored")

    groups = {f"block_{i:05d}" for i in layer_of.values()} | ({"head"} if heads else set())
    finfo = {"forward_seconds": 0.0, "reverse_seconds": 0.0}
    if not all((parts / f"{g}.json").exists() for g in groups):
        model = load_lazy(ctx.model_dir)[0]
        finfo = fisher_diag_streamed(model, calib, block_names + heads, model_dir=ctx.model_dir, on_group=score,
                                     micro_batch=p.fisher_micro_batch, loss=p.fisher_loss, seed=cfg.seed,
                                     work_dir=d / "fisher_work", fingerprint=fp)
        _free(model)
    cands = {}
    for f in sorted(parts.glob("*.json")):
        cands.update(json.loads(f.read_text()))
    expected = set(_unit_map(ctx, block_names)) | set(heads)
    if set(cands) != expected:
        raise RuntimeError(f"candidates: scored units differ from the allocated units "
                           f"(missing {sorted(expected - set(cands))[:3]}, extra {sorted(set(cands) - expected)[:3]})")
    out = d / "candidates.json"
    atomic_write_json(out, cands)
    shutil.rmtree(d / "fisher_work", ignore_errors=True)
    return {"outputs": [out], "info": {"tensors": len(cands), "fisher_forward_seconds": finfo["forward_seconds"],
                                       "fisher_reverse_seconds": finfo["reverse_seconds"]}}


def _unit_map(ctx: Ctx, names: list[str]) -> dict[str, list[str]]:
    """Allocation units over block tensor names: one per tensor (allocation_unit "tensor" / "expert"), or one per MoE
    expert stack ("expert_stack")."""
    if ctx.cfg.plan.allocation_unit == "expert_stack":
        return stacks.units(names)
    return {n: [n] for n in names}


def _score_map(ctx: Ctx, names: list[str]) -> dict[str, list[str]]:
    """Scoring units: the expert stacks for "expert_stack" and "expert" (one pooled codebook fit per stack and shared
    candidate; "expert" then scores every expert separately, build_candidates(per_member=True)), else per tensor."""
    if ctx.cfg.plan.allocation_unit in ("expert_stack", "expert"):
        return stacks.units(names)
    return {n: [n] for n in names}


def _allowed_candidates(ctx: Ctx, role: str, *, scored: bool = False) -> tuple[str, ...]:
    """Candidate names usable for a tensor of `role` (plan.candidate_roles; unlisted candidates: every role).
    scored=True: only the candidates the candidates stage encodes and scores (not the native pass-through ones);
    scored=False (what the greedy may choose) also drops plan.allocation.exclude."""
    p = ctx.cfg.plan
    cr = p.candidate_roles or {}
    native = native_candidate_names(p) if scored else set()
    return tuple(c.name for c in p.candidates
                 if (c.name not in cr or role in cr[c.name]) and c.name not in native
                 and (scored or not p.excluded(c.name, role)))


# ------------------------------------------------------------------- LDLQ second moments (tcq + ldlq codecs)

def ldlq_targets(ctx: Ctx) -> tuple[list[str], list[str]]:
    """(dense linear names without ".weight", MoE layer prefixes) of the block tensors that may get an ldlq codec:
    the uniform codec, or any scored candidate allowed for their role. ([], []) without ldlq codecs."""
    from suc.analysis.hessians import moe_prefixes

    p = ctx.cfg.plan
    ld = {c.name for c in p.codec_configs().values() if c.ldlq}
    if not ld:
        return [], []
    dense, experts = [], []
    for s in ctx.specs:
        allowed = {p.uniform_codec(s.role)} if p.mode == "uniform" else set(_allowed_candidates(ctx, s.role,
                                                                                               scored=True))
        if ld & allowed:
            (experts if stacks.stack_of(s.name) is not None else dense).append(s.name)
    return sorted(n[:-7] for n in dense), sorted(moe_prefixes(experts))


def _stage_hessians(ctx: Ctx, runner: StageRunner, d: Path, fp: str) -> dict:
    """BlockLDLQ input second moments (suc.analysis.hessians.record_hessians), layer-streamed and resumable."""
    from suc.analysis.hessians import record_hessians

    p = ctx.cfg.plan
    dense, moe = ldlq_targets(ctx)
    toks = _load_artifact(runner.path("calibrate", "data.safetensors"))["calibration"][: p.ldlq_sequences]
    r = record_hessians(ctx.model_dir, toks, d, dense=dense, moe=moe, micro_batch=ctx.cfg.calibration.micro_batch,
                        fingerprint=fp, log=log)
    files = [d / f for f in r["files"]]
    info = {"sequences": int(toks.shape[0]), "dense": len(dense), "moe_layers": len(moe), "files": len(files),
            "bytes": sum(f.stat().st_size for f in files), "seconds": r["seconds"]}
    atomic_write_json(d / "hessians.json", info)
    log(f"hessians: {len(files)} block files, {info['bytes'] / 2 ** 30:.1f} GiB ({len(dense)} dense, {len(moe)} MoE "
        f"layers, {info['sequences']} sequences)")
    return {"outputs": files + [d / "hessians.json"], "info": info}


def _hessian_store(ctx: Ctx, runner: StageRunner):
    """HessianStore over the hessians stage files, None when the plan has no ldlq codec."""
    from suc.analysis.hessians import HessianStore

    if "hessians" not in runner.manifests:
        return None
    files = [runner.path("hessians", rel) for rel in sorted(runner.outputs("hessians")) if rel.endswith(".safetensors")]
    return HessianStore(files, min_tokens=ctx.cfg.plan.ldlq_min_tokens)


_FISHER_KEYS = ("fisher_loss", "fisher_sequences", "fisher_micro_batch", "allocation_unit")


def _candidates_reuse(ctx: Ctx, runner: StageRunner) -> tuple[Path, set[str], bool] | None:
    """plan.candidates_reuse: (parts directory of the earlier candidates stage, candidate names whose scores are
    reused, whether the source used the same scoring seed contract - else rotation-grouped tensors are re-scored). A candidate is reused when its codec config and role restriction are identical and the Fisher inputs
    (model hash, seed, calibration outputs, fisher_* and allocation_unit) are identical; anything else in the plan
    may differ. A source that does not match the Fisher inputs is a configuration error."""
    src = ctx.cfg.plan.candidates_reuse
    if not src:
        return None
    d = Path(src).expanduser()
    if (d / "candidates").is_dir():
        d = d / "candidates"
    m = json.loads((d / "manifest.json").read_text())
    inp = m["inputs"]
    new = scored_plan_delta(ctx.cfg.plan)
    old = inp.get("plan", {})
    bad = [k for k, a, b in (("model_hash", inp.get("model_hash"), ctx.model_hash), ("seed", inp.get("seed"),
                                                                                    ctx.cfg.seed),
                             ("calibrate", inp.get("calibrate"), runner.outputs("calibrate"))) if a != b]
    bad += [k for k in _FISHER_KEYS if old.get(k) != new.get(k)]
    if bad:
        raise ValueError(f"plan.candidates_reuse {d}: Fisher inputs differ ({bad}); scores cannot be reused")
    oc = {c["name"]: c for c in old.get("candidates", [])}
    orl, nrl = old.get("candidate_roles") or {}, new.get("candidate_roles") or {}
    names = {c["name"] for c in new.get("candidates", []) if oc.get(c["name"]) == c
             and orl.get(c["name"]) == nrl.get(c["name"])}
    # audit F08: scores of rotation-grouped tensors are reusable only from a source with the same seed contract
    same_contract = _candidates_seed_inputs(ctx.cfg) == {k: inp.get(k) for k in ("rotation_groups", "seed_contract")
                                                         if k in inp}
    return d / "parts", names, same_contract


def _candidates_seed_inputs(cfg) -> dict:
    """Candidates-stage fingerprint inputs of the scoring seed contract (audit F08). Present only with
    rotation_groups (the only setting under which scoring and encoding seeds differed before SEED_CONTRACT 2), so
    runs without rotation groups keep their fingerprints; runs with them are invalidated explicitly."""
    return {"rotation_groups": True, "seed_contract": SEED_CONTRACT} if cfg.rotation_groups else {}


def _seed_contract_matters(ctx: Ctx, codec, keys) -> bool:
    """True when a reused score of `codec` on `keys` would depend on the seed contract: rotation groups on, a codec
    that takes seed_in, and a rotation-grouped tensor among the keys."""
    return (ctx.cfg.rotation_groups and accepts_rotation_seeds(codec)
            and any(rotation_group(k) is not None for k in keys))


# ------------------------------------------------------------------- native MXFP4 pass-through candidates

def native_candidate_names(p) -> set[str]:
    """Greedy candidates of kind "mxfp4": the checkpoint's MXFP4 codes + scales kept verbatim (MXFP4Native)."""
    return {c.name for c in p.candidates if c.kind == "mxfp4"}


def scored_plan_delta(p) -> dict:
    """config_delta of the plan WITHOUT the native pass-through candidates (and their candidate_roles entries): the
    plan input of the candidates stage. Native candidates are never scored (error 0, cost known from the shape), so
    adding one to a config leaves the Fisher / candidate artifacts of an earlier run valid (resumable re-plan)."""
    d = dict(config_delta(p))
    d.pop("allocation", None)            # allocation-only overrides (budget, exclusions) never change the scores
    d.pop("candidates_reuse", None)      # reusing identical scores of an earlier run does not change them
    native = native_candidate_names(p)
    if not native:
        return d
    d["candidates"] = [cd for cd, c in zip(d["candidates"], p.candidates) if c.name not in native]
    if d.get("candidate_roles"):
        d["candidate_roles"] = {k: v for k, v in d["candidate_roles"].items() if k not in native}
    return d


def _is_mxfp4_source(ctx: Ctx, name: str) -> bool:
    q = ctx.store.quant_source(name) if hasattr(ctx.store, "quant_source") else None
    return q is not None and q.kind == "mxfp4"


def add_native_candidates(ctx: Ctx, cands: dict[str, list[Candidate]]) -> tuple[dict[str, list[Candidate]], dict]:
    """Every allocation unit of block tensors whose members are ALL MXFP4 at source (and whose role allows it) gets
    each native candidate with loss 0 (bit-exact vs the reference) and its exact bits (MXFP4Native.bits_for: what
    the encode stage will store). Returns (candidates, {native candidate: units offered})."""
    from suc.quant.codecs import MXFP4Native

    native = native_candidate_names(ctx.cfg.plan)
    if not native:
        return cands, {}
    role_of = {s.name: s.role for s in ctx.specs}
    members = _unit_map(ctx, [s.name for s in ctx.specs])
    out, offered = {}, dict.fromkeys(sorted(native), 0)
    for u, cs in cands.items():
        cs = list(cs)
        ms = members.get(u)
        if ms and all(_is_mxfp4_source(ctx, m) for m in ms):
            have = {c.codec for c in cs}
            for name in sorted(native):
                if name in have or name not in _allowed_candidates(ctx, role_of[ms[0]]):
                    continue
                bits = sum(MXFP4Native.bits_for(ctx.store.info(m).shape) for m in ms)
                cs.append(Candidate(name, int(bits), 0.0, 0.0))
                offered[name] += 1
        out[u] = cs
    return out, offered


def native_candidate_errors(ctx: Ctx) -> list[str]:
    """A native candidate that no allowed tensor can use (no MXFP4-at-source tensor of its roles) is a config error."""
    p = ctx.cfg.plan
    native = native_candidate_names(p)
    if not native:
        return []
    if p.mode != "greedy":
        return [f"native candidates {sorted(native)} are only supported in a greedy plan"]
    errs = []
    for name in sorted(native):
        if not any(_is_mxfp4_source(ctx, s.name) for s in ctx.specs if name in _allowed_candidates(ctx, s.role)):
            errs.append(f"native candidate {name}: no tensor of its roles is MXFP4 at source")
    return errs


SUC1_OVERHEAD_PER_TENSOR = 4096     # bytes reserved per tensor for the SUC1 directory entry + 64-byte alignment


def fixed_bits(ctx: Ctx, allocated: set[str]) -> dict:
    """Bits of every tensor OUTSIDE the greedy allocation as the container will store them: raw tensors at their
    stored dtype, the embedding / LM head at the fixed embedding_codec, + a SUC1 overhead reserve per tensor
    (header, directory, alignment); lookup tables at the ngram_codec. The budget_bytes plan allocates
    budget_bytes * 8 minus this total."""
    from suc.evaluation.reconstruct import _DTYPE_BITS

    p, store = ctx.cfg.plan, ctx.store
    head = {s.name for s in ctx.head_specs} if p.embedding_codec is not None else set()
    tables = quantized_tables(ctx, store.names())
    raw = codec = 0
    for n in store.names():
        if n in allocated:
            continue
        shape = store.info(n).shape
        if n in tables:
            codec += 8 * repack.row_affine_bytes(shape, ctx.cfg.ngram_codec.bits)
        elif n in head:
            c = p.embedding_codec
            numel = _numel(shape)
            if c.kind == "affine":
                codec += numel * c.bits + (numel // c.group_size) * 2 * 16
            else:
                codec += int(math.ceil(_est_bpw(c, shape[-1], numel) * numel))
        else:
            raw += _numel(shape) * _DTYPE_BITS[store.info(n).dtype]
    overhead = 8 * SUC1_OVERHEAD_PER_TENSOR * len(store.names())
    return {"raw_bits": raw, "fixed_codec_bits": codec, "overhead_bits": overhead, "total": raw + codec + overhead}


def _role_budget_groups(ctx: Ctx, cands: dict, heads: list[str]) -> tuple[list[tuple[dict, list[str], int]], list[str]]:
    """plan.role_budgets (or its plan.allocation override): ([(group, its allocation units, their parameter count)],
    remaining units), candidate order.
    A unit's role is its first member's (the experts of an expert-stack unit share one role)."""
    role_of = {s.name: s.role for s in ctx.specs + ctx.head_specs}
    members = {**_unit_map(ctx, [s.name for s in ctx.specs]), **{h: [h] for h in heads}}
    groups = [(g, [], 0) for g in ctx.cfg.plan.allocation_role_budgets()]
    rest = []
    for u in cands:
        ms = members.get(u, [u])
        role = role_of.get(ms[0])
        for k, (g, units, n) in enumerate(groups):
            if role in g["roles"]:
                units.append(u)
                groups[k] = (g, units, n + sum(_numel(ctx.store.info(m).shape) for m in ms))
                break
        else:
            rest.append(u)
    return groups, rest


def _apply_allocation_exclude(ctx: Ctx, cands: dict[str, list[Candidate]]) -> dict[str, list[Candidate]]:
    """plan.allocation.exclude: drop the excluded candidates of every unit (role of its first member)."""
    p = ctx.cfg.plan
    role_of = {s.name: s.role for s in ctx.specs + ctx.head_specs}
    members = _unit_map(ctx, [s.name for s in ctx.specs])
    out = {}
    for u, cs in cands.items():
        role = role_of.get(members.get(u, [u])[0])
        kept = [c for c in cs if not p.excluded(c.codec, role)]
        if not kept:
            raise ValueError(f"plan.allocation.exclude leaves no candidate for {u!r} (role {role})")
        out[u] = kept
    return out


def allocate_plan(ctx: Ctx, cands: dict[str, list[Candidate]] | None) -> tuple[dict[str, str], dict]:
    """The ONE call into the allocator: {tensor name: codec name} for every tensor that gets a codec, + summary.

    uniform: plan.codec on every eligible block linear; greedy: greedy_allocate on the Fisher candidates for
    budget_bpw × allocated params (allocated = eligible linears, plus embedding / LM head with head_candidates).
    A fixed embedding_codec is applied outside the budget."""
    p = ctx.cfg.plan
    block_names = [s.name for s in ctx.specs]
    heads = _head_names_allocated(ctx)
    alloc_params = sum(_numel(ctx.store.info(n).shape) for n in block_names + heads)
    summary: dict = {"mode": p.mode, "allocated_tensors": len(block_names) + len(heads),
                     "allocated_params": alloc_params}
    if p.mode == "uniform":
        plan = {s.name: p.uniform_codec(s.role) for s in ctx.specs}
        if p.role_codecs is not None:
            summary["role_codecs"] = p.role_codecs
    else:
        cands, offered = add_native_candidates(ctx, cands)
        if offered:
            summary["native_candidates_offered"] = offered
        if p.allocation is not None:
            cands = _apply_allocation_exclude(ctx, cands)
            summary["allocation"] = p.allocation
        groups, rest = _role_budget_groups(ctx, cands, heads)
        chosen: dict[str, Candidate] = {}
        group_bits = 0
        rows = []
        for g, units, params in groups:
            sub = {u: cands[u] for u in units}
            gb = int(g["budget_bpw"] * params)
            got = greedy_allocate(sub, gb) if sub else {}
            chosen.update(got)
            group_bits += gb
            bits = sum(c.bits for c in got.values())
            rows.append({"roles": list(g["roles"]), "budget_bpw": g["budget_bpw"], "units": len(units),
                         "params": params, "budget_bits": gb, "predicted_bits": bits,
                         "predicted_bpw": bits / params if params else None})
        rest_params = alloc_params - sum(r["params"] for r in rows)
        if p.budget_bytes is not None:
            fixed = fixed_bits(ctx, set(block_names) | set(heads))
            budget_bytes = p.allocation_budget_bytes()
            budget = int(budget_bytes * 8) - fixed["total"] - group_bits
            if budget <= 0:
                raise ValueError(f"plan: budget_bytes {budget_bytes:.4g} is below the fixed part alone "
                                 f"({(fixed['total'] + group_bits) / 8:.4g} bytes incl. role_budgets)")
            summary.update(budget_bytes=budget_bytes, fixed_bits=fixed, budget_bits_allocated=budget + group_bits,
                           budget_bpw_allocated=(budget + group_bits) / alloc_params)
        else:
            budget = int(p.budget_bpw * rest_params)
            summary.update(budget_bpw=p.budget_bpw)
        if rest:
            chosen.update(greedy_allocate({u: cands[u] for u in rest}, budget))
        chosen = {u: chosen[u] for u in cands}               # plan in candidate order
        if rows:
            summary["role_budgets"] = rows
        s = plan_summary(chosen, cands)
        plan = stacks.expand_plan({n: c.codec for n, c in chosen.items()}, _unit_map(ctx, block_names))
        summary.update(predicted_bits=s["bits_total"], predicted_loss=s["loss_total"],
                       bits_min=s["bits_min"], bits_max=s["bits_max"])
        if p.budget_bytes is not None:
            summary["predicted_total_bytes"] = (s["bits_total"] + summary["fixed_bits"]["total"]) / 8
    if p.allocation_unit != "tensor":     # only when set: plan.json (an upstream hash) of earlier runs is unchanged
        summary["allocation_unit"] = p.allocation_unit
        summary["allocation_units"] = len(_unit_map(ctx, block_names)) + len(heads)
    if p.embedding_codec is not None:
        plan.update({s.name: p.uniform_codec(s.role) if p.mode == "uniform" else p.embedding_codec.name
                     for s in ctx.head_specs})
    return plan, summary


def _stage_plan(ctx: Ctx, runner: StageRunner, d: Path, fp: str) -> dict:
    cands = None
    if ctx.cfg.plan.mode == "greedy":
        raw = json.loads(runner.path("candidates", "candidates.json").read_text())
        cands = {n: [Candidate(**c) for c in cs] for n, cs in raw.items()}
    plan, summary = allocate_plan(ctx, cands)
    out = d / "plan.json"
    atomic_write_json(out, {"tensors": plan, "summary": summary})
    log(f"plan: {summary['mode']}, {len(plan)} tensors with a codec")
    return {"outputs": [out], "info": summary}


def _read_plan(runner: StageRunner) -> tuple[dict[str, str], dict]:
    r = json.loads(runner.path("plan", "plan.json").read_text())
    return r["tensors"], r["summary"]


def _groups(ctx: Ctx, plan: dict[str, str]) -> list[tuple[str, list[LinearSpec]]]:
    """Encode groups in processing order: "head" (embedding / LM head) then one group per transformer block."""
    out = []
    head = [s for s in ctx.head_specs if s.name in plan]
    if head:
        out.append(("head", head))
    by_layer: dict[int, list[LinearSpec]] = defaultdict(list)
    for s in ctx.specs:
        if s.name in plan:
            by_layer[s.layer].append(s)
    out += [(f"block_{i:05d}", by_layer[i]) for i in sorted(by_layer)]
    return out


def _release_group(model, g: str) -> None:
    if g == "head":
        _release(model.model.embed_tokens)
        if "lm_head" in model:
            _release(model.lm_head)
    else:
        _release(model_blocks(model)[int(g.split("_")[1])])
    mx.clear_cache()


def _stage_encode(ctx: Ctx, runner: StageRunner, d: Path, fp: str) -> dict:
    cfg = ctx.cfg
    plan, _ = _read_plan(runner)
    rec = _recorder(ctx, runner)
    table = CodecTable(ctx, plan, _stats_names(ctx, rec))
    importance = _importance(ctx, rec, ctx.specs)
    prog_path = d / "progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else None
    if prog is None or prog.get("fingerprint") != fp:
        for f in d.iterdir():
            f.unlink() if f.is_file() else shutil.rmtree(f)
        prog = {"fingerprint": fp, "groups": {}}
    model = load_lazy(ctx.model_dir)[0]     # lazy (mmap): only the block being encoded is materialized
    hstore = _hessian_store(ctx, runner)      # BlockLDLQ second moments (tcq + ldlq codecs), one block at a time
    groups = _groups(ctx, plan)
    for k, (g, gspecs) in enumerate(groups):
        shard = d / f"{g}.suc1"
        done = prog["groups"].get(g)
        if done is not None and shard.exists() and shard.stat().st_size == done["bytes"]:
            log(f"encode {g}: resumed from disk ({len(done['tensors'])} tensors)")
            _release_group(model, g)
            continue
        recon = reconstruct_model(model, ctx.store, gspecs, lambda s: table.for_tensor(s.name), seed=cfg.seed,
                                  importance=importance, keep_encodings=True, rotation_groups=cfg.rotation_groups,
                                  expert_stacks=cfg.plan.allocation_unit in ("expert_stack", "expert"),
                                  mixed_stacks=cfg.plan.allocation_unit == "expert", hessians=hstore)
        encs = recon["encodings"]
        for t in recon["tensors"]:
            t["plan_codec"] = plan[t["name"]]        # config codec name (t["codec"] is the codec object's name)
        with atomic_path(shard) as tmp:
            info = write_suc1(tmp, architecture=ctx.hf_config, tokenizer_id=ctx.tokenizer_id, tensors=encs,
                              metadata={"group": g, "stage": "encode"})
        back = SucModel.load(shard, verify=True)
        bad = [n for n in encs if not _same_encoded(back.encoded(n), encs[n])]
        if bad:
            raise RuntimeError(f"encode {g}: SUC1 shard does not round-trip bit-exactly for {bad[:3]}")
        prog["groups"][g] = {"file": shard.name, "bytes": info["bytes"], "sha256": info["sha256"],
                             "tensors": recon["tensors"], "seconds": recon["seconds"]}
        atomic_write_json(prog_path, prog)
        bpw = sum(t["bits"]["total"] for t in recon["tensors"]) / max(1, sum(_numel(t["shape"]) for t in recon["tensors"]))
        log(f"encode {g} ({k + 1}/{len(groups)}): {len(encs)} tensors, {bpw:.3f} bpw, {recon['seconds']:.1f} s")
        del recon, encs, back
        _release_group(model, g)
        gc.collect()
    _free(model)
    tensors = [t for g, _ in groups for t in prog["groups"][g]["tensors"]]
    summary = {"tensors": tensors, "groups": {g: {k: v for k, v in r.items() if k != "tensors"}
                                              for g, r in prog["groups"].items()},
               "seconds": sum(r["seconds"] for r in prog["groups"].values())}
    atomic_write_json(d / "encode.json", summary)
    shards = [d / prog["groups"][g]["file"] for g, _ in groups]
    return {"outputs": shards + [d / "encode.json"],
            "sha256": {str(d / prog["groups"][g]["file"]): prog["groups"][g]["sha256"] for g, _ in groups},
            "info": {"groups": len(groups), "tensors": len(tensors)}}


def _shards(runner: StageRunner, stage: str) -> list[Path]:
    return [runner.path(stage, rel) for rel in sorted(runner.outputs(stage)) if rel.endswith(".suc1")]


def _stage_finetune(ctx: Ctx, runner: StageRunner, d: Path, fp: str) -> dict:
    from suc.quant.finetune import FinetuneConfig, finetune_blockwise

    f = ctx.cfg.finetune
    plan, _ = _read_plan(runner)
    table = CodecTable(ctx, plan, _stats_names(ctx, _recorder(ctx, runner)))
    shards = _shards(runner, "encode")
    head = _load_shards([s for s in shards if s.name == "head.suc1"])
    block_shards = {s.stem: s for s in shards if s.name != "head.suc1"}
    model = load_lazy(ctx.model_dir)[0]
    for n, e in head.items():   # the student's inputs start from the encoded embedding, as in suc quantize
        set_weight(model, n, table.decode(n, e))
    del head
    calib = _load_artifact(runner.path("calibrate", "data.safetensors"))["calibration"]
    cur: dict[str, Encoded] = {}             # encodings of the block being tuned (D8: one block resident)

    def block_encodings(i: int) -> dict[str, Encoded]:
        cur.clear()
        src = block_shards.get(f"block_{i:05d}")
        if src is not None:
            cur.update(_load_shards([src]))
        return dict(cur)

    outputs, sha = [], {}

    def write_block(i: int, tuned: dict[str, Encoded]) -> None:
        src = block_shards[f"block_{i:05d}"]
        out = d / src.name
        with atomic_path(out) as tmp:
            info = write_suc1(tmp, architecture=ctx.hf_config, tokenizer_id=ctx.tokenizer_id, tensors=tuned,
                              metadata={"group": src.stem, "stage": "finetune"})
        back = SucModel.load(out, verify=True)
        if set(back.names()) != set(cur) or any(not _same_encoded(back.encoded(n), tuned[n]) for n in tuned):
            raise RuntimeError(f"finetune {src.stem}: SUC1 shard does not round-trip bit-exactly")
        outputs.append(out)
        sha[str(out)] = info["sha256"]

    _, flog = finetune_blockwise(
        model, block_encodings, calib, FinetuneConfig(block_steps=f.block_steps, lr=f.lr, sequences=f.sequences,
                                                      micro_batch=f.micro_batch, deterministic=f.deterministic),
        reference_weights=ctx.store.load, fixed_weights=lambda n: table.decode(n, cur[n]),
        after_block=lambda i, b: _release(b), on_block_tuned=lambda i, t: write_block(i, t) if t else None)
    _free(model)
    if sorted(outputs) != sorted(d / s.name for s in block_shards.values()):
        raise RuntimeError("finetune: not every encode block shard got a fine-tuned shard")
    outputs.sort()
    blocks = list(flog["blocks"].values())
    summary = {"blocks": len(blocks), "seconds": flog["seconds"], "peak_memory_bytes": flog["peak_memory_bytes"],
               "loss_before_mean": sum(b["loss_before"] for b in blocks) / max(len(blocks), 1),
               "loss_after_mean": sum(b["loss_after"] for b in blocks) / max(len(blocks), 1),
               "reverted": sum(1 for b in blocks if b.get("reverted")), "skipped": len(flog["skipped"]),
               "fixed": sum(b.get("fixed", 0) for b in blocks),
               "per_block": {str(i): {k: b[k] for k in ("loss_before", "loss_after")} for i, b in flog["blocks"].items()}}
    atomic_write_json(d / "finetune.json", summary)
    log(f"finetune: {len(blocks)} blocks, mean block loss {summary['loss_before_mean']:.4f} -> "
        f"{summary['loss_after_mean']:.4f}")
    return {"outputs": outputs + [d / "finetune.json"], "sha256": sha, "info": {k: v for k, v in summary.items()
                                                                              if k != "per_block"}}


class ShardIndex:
    """Lazy view of the final encodings: tensor name -> the per-block SUC1 shard holding it (fine-tuned shards
    override encode shards). Only directories are read up front; `get` reads one tensor (mmap)."""

    def __init__(self, files: list[Path]) -> None:
        self._sm: dict[str, SucModel] = {}
        for f in files:
            sm = SucModel.load(f, verify=False)       # verified (checksum + bit-exact round trip) when written
            for n in sm.names():
                self._sm[n] = sm

    def names(self) -> list[str]:
        return sorted(self._sm)

    def __contains__(self, name: object) -> bool:
        return name in self._sm

    def __len__(self) -> int:
        return len(self._sm)

    def get(self, name: str) -> Encoded | mx.array:
        return self._sm[name].encoded(name)


def _final_encodings(ctx: Ctx, runner: StageRunner) -> ShardIndex:
    files = _shards(runner, "encode")
    if ctx.cfg.finetune is not None:
        files += _shards(runner, "finetune")
    return ShardIndex(files)


def _tied_duplicate(store: SafetensorsStore, encoded: set[str]) -> set[str]:
    """lm_head stored byte-identical to the embedding (tied weights stored twice) is kept once (as model_size)."""
    e, h = "model.embed_tokens.weight", "lm_head.weight"
    if h in store and e in store and h not in encoded and _same_bytes(store, h, e):
        return {h}
    return set()


def quantized_tables(ctx: Ctx, names) -> set[str]:
    """Row-gather tables the container stores row-affine (ngram_codec row_affine): the adapter's table tensors
    ("*.weight" shards; the I64 hash buffers stay raw). Empty with the default ngram_codec (raw)."""
    if ctx.cfg.ngram_codec.kind == "raw":
        return set()
    return {n for n in ctx.adapter.table_tensors(names) if n.endswith(".weight")}


def _stage_container(ctx: Ctx, runner: StageRunner, d: Path, fp: str) -> dict:
    cfg = ctx.cfg
    plan, _ = _read_plan(runner)
    table = CodecTable(ctx, plan, _stats_names(ctx, _recorder(ctx, runner)))
    encs = _final_encodings(ctx, runner)
    skip = _tied_duplicate(ctx.store, set(encs.names()))
    raw = [n for n in ctx.store.names() if n not in encs and n not in skip]
    tables = quantized_tables(ctx, raw)
    raw = [n for n in raw if n not in tables]
    path = d / "model.suc1"
    counts: dict[str, int] = defaultdict(int)
    for c in plan.values():
        counts[c] += 1
    # content-only metadata (no config hash / git commit): the file is a pure function of the stage inputs
    meta = {"model": cfg.model.id, "model_revision": ctx.revision, "model_hash": ctx.model_hash,
            "plan_codecs": dict(sorted(counts.items())), "skipped_duplicates": sorted(skip)}
    nc = cfg.ngram_codec
    source = repack.safetensors_table_source(ctx.store)
    # one shard quantized at a time from the checkpoint mmap in row chunks (D8): write_suc1_streamed produces every
    # segment of a tensor before the next tensor, and the quantizer keeps only the current shard's result
    quant = repack._TableQuantizer(source, nc.bits, nc.scale_dtype)
    if tables:
        meta["ngram_tables"] = {"codec": nc.kind, "bits": nc.bits, "scale_dtype": nc.scale_dtype,
                                "scheme": "row_minmax", "tensors": len(tables)}

    def get(n: str):
        if n in encs:
            return encs.get(n)
        if n in tables:
            return repack.row_affine_prepared(n, ctx.store.info(n).shape, quant)
        return ctx.store.load(n)

    with atomic_path(path) as tmp:     # one tensor resident (D8); byte-identical to write_suc1 on the same content
        info = write_suc1_streamed(tmp, architecture=ctx.hf_config, tokenizer_id=ctx.tokenizer_id,
                                   names=encs.names() + raw + sorted(tables), metadata=meta, get=get)
    quant._name = quant._res = None
    _free()
    sm = SucModel.load(path, verify=True)
    if set(sm.names()) != set(encs.names()) | set(raw) | tables:
        raise RuntimeError("container: tensor set of the SUC1 file differs from the expected set")
    rng = np.random.default_rng(cfg.seed)
    names = encs.names()
    sample = [names[i] for i in sorted(rng.choice(len(names), min(cfg.verify_tensors, len(names)), replace=False))]
    raw_sample = [raw[i] for i in sorted(rng.choice(len(raw), min(4, len(raw)), replace=False))] if raw else []
    exact, abs_err, rel_err = 0, 0.0, 0.0
    for n in sample:
        e = encs.get(n)
        if not _same_encoded(sm.encoded(n), e):
            raise RuntimeError(f"container: {n} in the SUC1 file is not bit-identical to its encoding")
        exact += 1
        w = table.decode(n, e).astype(mx.float32)
        diff = mx.max(mx.abs(sm.decode_tensor(n) - w)).item()
        abs_err = max(abs_err, diff)
        rel_err = max(rel_err, diff / max(mx.max(mx.abs(w)).item(), 1e-30))
    for n in raw_sample:
        if not _same_encoded(sm.encoded(n), ctx.store.load(n)):
            raise RuntimeError(f"container: raw tensor {n} is not bit-identical to the checkpoint")
    if rel_err > 1e-2:
        raise RuntimeError(f"container: reference decoder disagrees with the codec decode (max rel err {rel_err:.2e})")
    tables_rec = None
    if tables:     # sampled rows of every stored table decode within half a quantization step of the checkpoint
        checked = repack.check_table_rows(source, repack.suc1_table_reader(sm), tables, quant, 64, cfg.seed)
        before = sum(ctx.store.info(n).nbytes for n in tables)
        after = sum(int(sm.descriptor(n)["compressed_size"]) for n in tables)
        st = quant.stats
        tables_rec = {"converted": len(tables), "codec": nc.kind, "bits": nc.bits, "scale_dtype": nc.scale_dtype,
                      "bytes_before": before, "bytes_after": after,
                      "bpp_after": 8 * after / max(1, sum(_numel(ctx.store.info(n).shape) for n in tables)),
                      "rel_rmse_max": max(s["rel_rmse"] for s in st.values()),
                      "max_abs_err": max(s["max_abs_err"] for s in st.values()),
                      "rows_checked": checked, "per_tensor": st}
    rec = {"path": str(path.resolve()), "bytes": info["bytes"], "sha256": info["sha256"], "tensors": info["tensors"],
           "codebooks": info["codebooks"], "encoded_tensors": len(encs), "raw_tensors": len(raw),
           **({"tables": tables_rec} if tables_rec is not None else {}),
           "skipped_duplicates": sorted(skip), "file_bits": sm.bits(),
           "verify": {"checksum": True, "checked": len(sample), "bit_exact": exact, "raw_checked": len(raw_sample),
                      "ref_decoder_max_abs_err": abs_err, "ref_decoder_max_rel_err": rel_err}}
    atomic_write_json(d / "container.json", rec)
    log(f"container: {path} {info['bytes'] / 2 ** 20:.1f} MiB, {info['tensors']} tensors; verified {exact} encoded "
        f"(bit-exact) + {len(raw_sample)} raw, ref-decoder max rel err {rel_err:.1e}")
    if tables_rec is not None:
        log(f"container: {len(tables)} tables row-affine {nc.bits}-bit ({tables_rec['bytes_before'] / 2 ** 30:.2f} -> "
            f"{tables_rec['bytes_after'] / 2 ** 30:.2f} GiB), rel RMSE max {tables_rec['rel_rmse_max']:.4f}")
    outputs = [path, d / "container.json"]
    if cfg.preflight.expert_streaming:
        outputs.append(write_hot_experts(runner, d / "hot_experts.json"))
    return {"outputs": outputs, "sha256": {str(path): info["sha256"]}, "info": {"bytes": info["bytes"]}}


def write_hot_experts(runner: StageRunner, out: Path) -> Path:
    """Routing counts of the calibration pass (calibrate/activations.json, the Recorder's expert_counts per MoE layer
    prefix "model.layers.N.mlp") as the hot-expert source of the expert-streaming runtime
    (suc.runtime.expert_streaming pins the most-routed experts first)."""
    side = json.loads(runner.path("calibrate", "activations.json").read_text())
    counts = {k: [int(c) for c in v] for k, v in side.get("expert_counts", {}).items()}
    if not counts:
        raise RuntimeError("container: expert streaming requested but the calibration pass recorded no expert routing")
    atomic_write_json(out, {"source": "calibrate/activations.json (calibration partition routing counts)",
                            "layers": counts, "tokens_x_top_k": {k: sum(v) for k, v in counts.items()}})
    return out


def _install(model, sm: SucModel, table: CodecTable, names: list[str]) -> list[dict]:
    """Set every encoded tensor of the file into the model as its codec decode (expert rows buffered per layer)."""
    rows, pending, last = [], {}, None
    for n in sorted(names, key=lambda n: (int(m.group(1)) if (m := _LAYER.match(n)) else -1, n)):
        layer = int(m.group(1)) if (m := _LAYER.match(n)) else -1
        if pending and layer != last:
            _flush_experts(model, pending)
        last = layer
        e = sm.encoded(n)
        w = table.decode(n, e)
        if (m := _EXPERT.match(n)) is not None:
            stack = f"{m.group(1)}.switch_mlp.{m.group(3)}.weight"
            row = w.astype(get_weight(model, stack).dtype)
            mx.eval(row)
            pending.setdefault(stack, {})[int(m.group(2))] = row
        else:
            set_weight(model, n, w)
            mx.eval(get_weight(model, n))
        rows.append({"name": n, "bits": {k: int(v) for k, v in e.bits().items()}})
        del e, w
    _flush_experts(model, pending)
    return rows


def _stored_bits(sm: SucModel, name: str) -> dict:
    """Encoded.bits()-style accounting of a stored (non-raw) tensor from its payload records (no decode)."""
    from suc.quant.codecs import METADATA_BITS

    out = {"indices": 0, "codebooks": 0, "scales": 0, "residual": 0, "metadata": METADATA_BITS}
    for r in sm.descriptor(name)["arrays"]:
        out[r["part"]] += 8 * int(r["nbytes"])
    out["total"] = sum(out.values())
    return out


def _stage_evaluate(ctx: Ctx, runner: StageRunner, d: Path, fp: str) -> dict:
    ev = ctx.cfg.eval
    plan, _ = _read_plan(runner)
    table = CodecTable(ctx, plan, _stats_names(ctx, _recorder(ctx, runner)))
    ref = json.loads(runner.path("reference", "reference.json").read_text())
    ppl_tokens, _, tok_sha = _ppl_tokens(ctx)
    if tok_sha != ref["ppl_tokens_sha256"]:
        raise RuntimeError("evaluate: WikiText-2 tokens differ from those of the reference stage")
    kl_tokens = _load_artifact(runner.path("calibrate", "data.safetensors"))["validation"][: ev.kl_sequences]
    ref_logits = RefLogits(runner.path("reference", "ref_logits.safetensors"))   # read chunk by chunk
    sm = SucModel.load(runner.path("container", "model.suc1"), verify=False)  # verified by the container stage
    tables = ctx.adapter.table_tensors(sm.names())          # memory-mapped lookup tables, attached from the file
    encoded = [n for n in sm.names() if sm.descriptor(n)["representation_type"] != "raw" and n not in tables]
    by_layer: dict[int, list[str]] = defaultdict(list)
    for n in encoded:
        by_layer[int(m.group(1)) if (m := _LAYER.match(n)) else -1].append(n)
    rows: list[dict] = []
    model = load_lazy(ctx.model_dir)[0]      # lazy: block i is materialized (from its decoded encodings) only when run
    if hasattr(model, "attach_suc1"):        # the lookup tables as STORED (raw or row-affine payload views of the file)
        model.attach_suc1(sm)
    rows.extend({"name": n, "bits": _stored_bits(sm, n)} for n in sorted(tables)
                if sm.descriptor(n)["representation_type"] != "raw")
    r = streamed_quality(model, ppl_tokens=ppl_tokens, seq_len=ev.seq_len, max_tokens=ev.max_tokens,
                         kl_tokens=kl_tokens, chunk=ev.chunk, ref_logits=ref_logits,
                         prepare_head=lambda mdl: rows.extend(_install(mdl, sm, table, by_layer.get(-1, []))),
                         prepare_block=lambda i, b: rows.extend(_install(model, sm, table, by_layer.get(i, []))),
                         spill_dir=_spill(ctx, d, _eval_tokens(ev, ppl_tokens, kl_tokens)))
    _free(model)
    quality = {"perplexity": r["perplexity"], "kl": r["kl"]}
    size = model_size(ctx.store, {"tensors": rows})
    fb = sm.bits()["total"]
    size.update(file_bits=fb, file_bytes=fb // 8, file_bpp=fb / size["parameter_count"])
    out = d / "eval.json"
    atomic_write_json(out, {"quality": quality, "size": size, "streaming": r["streaming"]})
    log(f"evaluate: PPL {quality['perplexity']['ppl']:.3f} (BF16 {ref['quality']['perplexity']['ppl']:.3f}), "
        f"KL {quality['kl']['kl_mean']:.4f}, bpp {size['bpp']:.3f} (file {size['file_bpp']:.3f})")
    return {"outputs": [out], "info": {"ppl": quality["perplexity"]["ppl"], "kl": quality["kl"]["kl_mean"],
                                       "bpp": size["bpp"]}}


# ------------------------------------------------------------------------------------------- report

def _streaming_record(stages: list[str]) -> dict:
    return {"requires_full_model": [s for s in FULL_MODEL_STAGES if s in stages],
            "holds_all_encodings": [s for s in ALL_ENCODINGS_STAGES if s in stages],
            "streaming": [s for s in STREAMING_STAGES if s in stages],
            "note": "streaming stages keep at most one transformer block of weights (plus embedding / LM head, the "
                    "hidden states of their sequences, spilled to disk above HIDDEN_IN_MEMORY_MAX, and one "
                    "micro-batch's working set) resident; requires_full_model / holds_all_encodings list stages "
                    "that still need the whole BF16 model / every encoding in memory."}


def _stage_report(ctx: Ctx, runner: StageRunner, metadata: dict, pre: dict, d: Path, fp: str) -> dict:
    cfg = ctx.cfg
    calib = json.loads(runner.path("calibrate", "calibration.json").read_text())
    ref = json.loads(runner.path("reference", "reference.json").read_text())
    plan, psum = _read_plan(runner)
    enc = json.loads(runner.path("encode", "encode.json").read_text())
    ft = json.loads(runner.path("finetune", "finetune.json").read_text()) if cfg.finetune is not None else None
    cont = json.loads(runner.path("container", "container.json").read_text())
    cont["path"] = str(runner.path("container", "model.suc1").resolve())   # the run dir may have been moved
    ev = json.loads(runner.path("evaluate", "eval.json").read_text())
    alloc = {s.name for s in ctx.specs} | set(_head_names_allocated(ctx))
    at = [t for t in enc["tensors"] if t["name"] in alloc]
    by_codec: dict[str, int] = defaultdict(int)
    for t in at:
        by_codec[plan[t["name"]]] += 1
    psum = {**psum, "by_codec": dict(sorted(by_codec.items())),
            "allocated_bpw": sum(t["bits"]["total"] for t in at) / max(1, sum(_numel(t["shape"]) for t in at)),
            "head": {s.name: plan.get(s.name, "bf16 (kept)") for s in ctx.head_specs}}
    status = {s["stage"]: s["status"] for s in runner.status}
    stages = {name: {"status": status.get(name), **{k: m[k] for k in ("seconds", "peak_memory_bytes", "git_commit",
                                                                       "finished", "fingerprint", "outputs")}}
              for name, m in runner.manifests.items()}
    commits = {m["git_commit"] for m in runner.manifests.values()}
    md = dict(metadata)
    md["dataset_hashes"] = {"calibration_partitions": calib["partitions_hash"], "wikitext": ref["wikitext_hash"]}
    md["dataset_hash"] = fingerprint(md["dataset_hashes"])
    result = {
        "metadata": md, "name": cfg.name, "config": config_to_dict(cfg),
        "architecture": {"family": ctx.adapter.family, "model_type": ctx.hf_config.get("model_type"),
                         "num_layers": ctx.adapter.num_layers, "is_moe": ctx.adapter.is_moe,
                         "num_experts": ctx.adapter.num_experts},
        "preflight": pre, "stages": stages, "stages_from_mixed_git_commits": len(commits) > 1,
        "calibration": {"partitions_hash": calib["partitions_hash"], "partitions": calib["partitions"],
                        "ngram_overlap": calib["ngram_overlap"], "activation_stats": calib["activation_stats"]},
        "plan": psum,
        "encode": {"tensors": enc["tensors"], "per_role": _per_role(enc["tensors"]), "seconds": enc["seconds"]},
        "finetune": ft, "container": cont, "size": ev["size"],
        "reference": ref["quality"], "quality": ev["quality"],
        "eval_data": {k: ref[k] for k in ("wikitext_split", "ppl_tokens", "ppl_tokens_sha256", "kl_sequences",
                                         "kl_seq_len")},
        "streaming": _streaming_record(list(runner.manifests)),
    }
    json_path = write_result(result, cfg.output_dir, cfg.name).resolve()
    md_path = json_path.with_suffix(".md")
    atomic_write_text(md_path, compress_report(result))
    return {"outputs": [json_path, md_path], "info": {"result": str(json_path)}}


def compress_report(r: dict) -> str:
    meta, q, ref, s, p = r["metadata"], r["quality"], r["reference"], r["size"], r["plan"]
    base = ref["perplexity"]["ppl"]
    kl = q["kl"]
    lines = [f"# Compression report: {r['name']}", "",
             f"Model `{meta['model']}` @ `{meta['model_revision']}` · weights `{meta['model_hash'][:16]}` · "
             f"git `{meta.get('git_commit')}`{' (dirty)' if meta.get('git_dirty') else ''}", "",
             f"SUC1 file: `{r['container']['path']}` ({r['container']['bytes'] / 2 ** 20:.1f} MiB, sha256 "
             f"`{r['container']['sha256'][:16]}`); {r['container']['verify']['bit_exact']} sampled encodings "
             f"bit-exact on reload, reference decoder max rel. error "
             f"{r['container']['verify']['ref_decoder_max_rel_err']:.1e}.", "",
             f"Plan: {p['mode']}" + (f" (budget {p['budget_bpw']} bpw)" if p.get("budget_bpw") else "")
             + (f" (budget {p['budget_bytes'] / 1e9:.2f} GB total, predicted {p['predicted_total_bytes'] / 1e9:.2f} GB)"
                if p.get("budget_bytes") else "")
             + f"; allocated bpw {p['allocated_bpw']:.3f} over {p['allocated_tensors']} tensors; codecs "
             + ", ".join(f"{k} {v}" for k, v in p["by_codec"].items()) + "; embedding / LM head: "
             + ", ".join(f"`{k}` {v}" for k, v in p["head"].items()) + ".", "",
             "| | bpp (model) | bpp (file) | Size (MB) | PPL | ΔPPL | KL mean | Top-1 | ΔCE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
             f"| BF16 | 16 | — | — | {base:.3f} | — | — | — | — |",
             f"| SUC | {s['bpp']:.3f} | {s['file_bpp']:.3f} | {s['bytes_total'] / 1e6:.1f} | {q['perplexity']['ppl']:.3f} "
             f"| {100 * (q['perplexity']['ppl'] / base - 1):+.1f}% | {kl['kl_mean']:.4f} | {kl['top1_agreement']:.3f} "
             f"| {kl['delta_ce']:+.4f} |", "",
             f"PPL: WikiText-2 {r['eval_data']['wikitext_split']} ({q['perplexity']['tokens_scored']:,} tokens scored); KL: "
             f"{r['eval_data']['kl_sequences']} held-out validation sequences × {r['eval_data']['kl_seq_len']} tokens. "
             "bpp (model) = codec bits of every tensor / parameters; bpp (file) = SUC1 file bits incl. header, "
             "directory and alignment padding.", "",
             "| Stage | Status | Seconds | Peak MLX memory (GiB) | git |", "|---|---|---:|---:|---|"]
    for name, st in r["stages"].items():
        lines.append(f"| {name} | {st['status']} | {st['seconds']:.1f} | {st['peak_memory_bytes'] / 2 ** 30:.2f} "
                     f"| {str(st['git_commit'])[:10]} |")
    if r["finetune"]:
        f = r["finetune"]
        lines += ["", f"Fine-tuning: {f['blocks']} blocks, mean block loss {f['loss_before_mean']:.4f} → "
                      f"{f['loss_after_mean']:.4f}, reverted {f['reverted']}, untuned (fixed) tensors {f['fixed']}."]
    st = r["streaming"]
    lines += ["", f"Not streaming-capable (whole BF16 model resident): {', '.join(st['requires_full_model']) or '—'}; "
                  f"all encodings in memory: {', '.join(st['holds_all_encodings']) or '—'}."]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------- driver

def _write_status(run_dir: Path, runner: StageRunner | None, state: str, extra: dict | None = None) -> None:
    atomic_write_json(run_dir / "status.json", {"state": state, "stages": runner.status if runner else [],
                                                **(extra or {})})


def run_compress(config_path: str | Path, *, tokenizer=None, docs=None, ood_docs=None) -> Path:
    """Run (or resume) the whole pipeline; returns the result JSON path. `tokenizer`/`docs`/`ood_docs` can be
    injected for the calibration corpus (tests); the default is the model tokenizer + suc.data.sources."""
    import time

    t_start = time.perf_counter()
    cfg = load_compress_config(config_path)
    seed_everything(cfg.seed)
    run_dir = Path(cfg.run_dir or f"{ARTIFACT_ROOT}/{cfg.name}").expanduser()   # "~/..." = a local disk
    with _run_lock(run_dir):
        log(f"run {cfg.name}: config {config_path} (hash {config_hash(cfg)[:12]}), run dir {run_dir}")
        try:
            check_revision(cfg)
        except PreflightError as ex:
            atomic_write_json(run_dir / "preflight.json", {"ok": False, "errors": [str(ex)]})
            raise
        model_dir = resolve_model_dir(cfg)
        revision = resolved_revision(model_dir, cfg.model.revision)
        if cfg.model.local_path is None and revision != cfg.model.revision:
            raise PreflightError(f"resolved snapshot {revision} != configured revision {cfg.model.revision}")
        hf_config = json.loads((model_dir / "config.json").read_text())
        adapter = get_adapter(hf_config)
        hf_store = SafetensorsStore.from_dir(model_dir)
        adapter.validate(hf_store.names())
        store = adapter.source_store(hf_store)     # SUC namespace (renamed / per-expert views; identity for Qwen3)
        names = store.names()
        specs = select_specs(adapter, names, None, include_embeddings=False)
        head_specs = [s for s in select_specs(adapter, names, None, include_embeddings=True) if s.role in _HEAD_ROLES]
        if cfg.plan.embedding_codec is None and cfg.plan.head_candidates is None:
            head_specs = []
        ctx = Ctx(cfg=cfg, model_dir=model_dir, store=store, hf_config=hf_config, adapter=adapter, specs=specs,
                  head_specs=head_specs, model_hash=model_hash(model_dir), revision=revision,
                  tokenizer=tokenizer, docs=docs, ood_docs=ood_docs)
        # Captured before the heavy work so git commit/dirty describe the code that actually runs.
        metadata = run_metadata(model=cfg.model.id, model_revision=revision, model_hash=ctx.model_hash,
                                dataset_hash="pending", seed=cfg.seed, config_hash=config_hash(cfg))
        pre = preflight(ctx, run_dir)
        runner = StageRunner(run_dir, git_commit=metadata["git_commit"], versions=STAGE_VERSIONS)
        o = runner.outputs
        m = ctx.model_hash

        def step(stage: str, inputs: dict, fn, **kw) -> dict:
            _write_status(run_dir, runner, f"running {stage}")
            try:
                out = runner.run(stage, inputs, fn, **kw)
            except BaseException as ex:   # recorded for unattended monitoring; rerun the same config to resume
                _write_status(run_dir, runner, f"failed in {stage}", {"error": f"{type(ex).__name__}: {ex}"})
                log(f"stage {stage}: FAILED ({type(ex).__name__}: {ex}); rerun the same command to resume")
                raise
            _write_status(run_dir, runner, f"finished {stage}")
            return out

        c = {k: config_delta(getattr(cfg, k)) for k in ("calibration", "eval", "plan", "finetune")}
        step("calibrate", {"model_hash": m, "calibration": c["calibration"], "seed": cfg.seed},
             lambda d, fp: _stage_calibrate(ctx, d, fp))
        _check_calibration_identity(ctx, runner)
        step("reference", {"model_hash": m, "eval": c["eval"], "calibrate": o("calibrate")["data.safetensors"]},
             lambda d, fp: _stage_reference(ctx, runner, d, fp))
        ld_dense, ld_moe = ldlq_targets(ctx)
        hess_in = {}
        if ld_dense or ld_moe:
            # BlockLDLQ second moments of the tensors that may get a tcq + ldlq codec (only then: other runs'
            # candidates / encode fingerprints are unchanged)
            step("hessians", {"model_hash": m, "calibrate": o("calibrate")["data.safetensors"],
                              "sequences": cfg.plan.ldlq_sequences, "micro_batch": cfg.calibration.micro_batch,
                              "dense": ld_dense, "moe": ld_moe},
                 lambda d, fp: _stage_hessians(ctx, runner, d, fp), resumable=True)
            hess_in = {"hessians": o("hessians")}
        plan_in = {"model_hash": m, "plan": c["plan"], "tensors": [s.name for s in specs + head_specs]}
        if cfg.ngram_codec.kind != "raw" and cfg.plan.mode == "greedy":    # fixed_bits counts the stored tables
            plan_in["ngram_codec"] = config_delta(cfg.ngram_codec)
        if cfg.plan.mode == "greedy":
            # the native pass-through candidates are not scored: leaving them out keeps earlier Fisher runs reusable
            step("candidates", {"model_hash": m, "plan": scored_plan_delta(cfg.plan), "seed": cfg.seed,
                                "calibrate": o("calibrate"), **hess_in, **_candidates_seed_inputs(cfg)},
                 lambda d, fp: _stage_candidates(ctx, runner, d, fp), resumable=True, keep_stale=True)
            plan_in["candidates"] = o("candidates")
        step("plan", plan_in, lambda d, fp: _stage_plan(ctx, runner, d, fp))
        stats = o("calibrate")["activations.safetensors"] if ctx.needs_stats else None
        step("encode", {"model_hash": m, "plan": o("plan"), "codecs": c["plan"], "seed": cfg.seed,
                        "rotation_groups": cfg.rotation_groups, "activations": stats, **hess_in},
             lambda d, fp: _stage_encode(ctx, runner, d, fp), resumable=True)
        upstream = {"encode": o("encode")}
        if cfg.finetune is not None:
            step("finetune", {"model_hash": m, "finetune": c["finetune"], "encode": o("encode"), "plan": o("plan"),
                              "calibrate": o("calibrate")["data.safetensors"], "activations": stats},
                 lambda d, fp: _stage_finetune(ctx, runner, d, fp))
            upstream["finetune"] = o("finetune")
        if cfg.preflight.expert_streaming:      # only then: the container fingerprint of other runs is unchanged
            upstream["routing"] = o("calibrate")["activations.json"]
        if cfg.ngram_codec.kind != "raw":       # only then: the container fingerprint of raw-table runs is unchanged
            upstream["ngram_codec"] = config_delta(cfg.ngram_codec)
        step("container", {"model_hash": m, "revision": revision, "verify_tensors": cfg.verify_tensors, "plan": o("plan"), **upstream},
             lambda d, fp: _stage_container(ctx, runner, d, fp))
        step("evaluate", {"model_hash": m, "eval": c["eval"], "container": o("container")["model.suc1"],
                          "reference": o("reference"), "plan": o("plan")},
             lambda d, fp: _stage_evaluate(ctx, runner, d, fp))
        rep = step("report", {"config_hash": config_hash(cfg),
                              "upstream": {k: v["outputs"] for k, v in runner.manifests.items()}},
                   lambda d, fp: _stage_report(ctx, runner, metadata, pre, d, fp))
        json_path = Path(next(p for p in rep["outputs"] if p.endswith(".json")))
        ev = json.loads(runner.path("evaluate", "eval.json").read_text())
        ran = [s["stage"] for s in runner.status if s["status"] == "ran"]
        _write_status(run_dir, runner, "done", {"result": str(json_path)})
        log(f"done {cfg.name}: bpp {ev['size']['bpp']:.3f} (file {ev['size']['file_bpp']:.3f}), PPL "
            f"{ev['quality']['perplexity']['ppl']:.3f}, KL {ev['quality']['kl']['kl_mean']:.4f}; ran {ran or 'nothing'}; "
            f"{time.perf_counter() - t_start:.0f} s total -> {json_path}")
        return json_path
