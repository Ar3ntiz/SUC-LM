"""E2E generation from SUC codes (spec §22, §35, §46): an mlx-lm Qwen3 whose eligible linears are SucLinear /
SucResidualLinear modules (suc.runtime.compressed_model), vs the same model in BF16 and MLX affine 4-bit.

Not a test. Usage:
  .venv/bin/python benchmarks/e2e_generate.py [--model Qwen/Qwen3-0.6B] [--revision <sha>] [--variants a,b,...]
         [--batches 1,4,16] [--prompt-len 128] [--decode-tokens 32] [--repeats 3] [--embed q4|bf16]
         [--kmeans-iters 6] [--fit-blocks 65536] [--residual-fraction 0.005] [--check-seqs 4] [--no-profile]
         [--rotation fused|mlx] [--no-rotation-groups] [--no-fuse] [--no-compile] [--fold] [--fold-v-heads]
         [--one-sided | --two-sided] [--no-fuse-elementwise] [--json out.json]
  Mac Studio: --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --json <out>

Variants: bf16, mlx_q4 (apply_representation("mlx_affine", bits=4): every Linear AND the embedding / LM head),
vq_g4_k256 (VQ G=4 K=256, rotated, ≈2 bpw), add3_g8_k256 (additive M=3 G=8 K=256, rotated, ≈3 bpw), and the same
two with "_res"; on request (--variants) also add4_g8_k256 and the low-dimensional single codebooks vq_g2_k256 /
vq_g2_k64 / vq_g2_k16 (≈4 / 3 / 2 bpw) and vq_g3_k512 (≈3 bpw; skipped when an input width is not a multiple of 3),
each with "_res" too (+ sparse residual, --residual-fraction of the weights, int8 values, MAGNITUDE-selected: timing
only; quality needs activation-selected residuals, docs/benchmarks.md Q5).
Codes are real encodings of the model's weights, but with a cheap k-means (--kmeans-iters, --fit-blocks): the
timing does not depend on code quality; quality is NOT reported here (see the Q-track results).

Runtime switches of the SUC variants (A/B of the batch-1 work, docs/benchmarks.md E2E sections):
  --rotation fused (default): one Metal launch per rotation side (suc.kernels.rotation); mlx: the 3 + 3 MLX ops.
  rotation groups (default on; --no-rotation-groups = per-tensor seeds as before): q/k/v and gate/up are encoded
      with a shared seed_in (suc.evaluation.reconstruct.rotation_seeds) and wrapped by group_projections, so their
      common input is rotated once per forward.
  --no-compile: eager graph building (default: SucLinear / group calls go through mx.compile, ~18 → ~3.5 µs of
      Python per call on the M3 — measured with synchronous per-token eval, so Python time is on the critical path).
  --residual-structure columns [--residual-column-share f]: the _res variants store the residual as dense int8
      input columns at the CSR bit budget of --residual-fraction (+ a CSR tail with share < 1; docs/benchmarks.md
      Q5b); --no-fuse-residual: residual members run their own CSR launch outside the fused q|k|v / gate|up part
      (the earlier path; default: SucFusedResidual, one residual launch per group).
  --no-fuse: groups share the rotation only; default: members with the same code layout are row-concatenated into
      one SUC launch + one segmented output-rotation launch (SucFusedLinear).

Rotation removal (docs/benchmarks.md R1-R4):
  --fold: the SUC variants (and their check copy) first get a residual-stream rotation folded into the BF16 weights
      (suc.models.rotation_folding: norms folded, embedding / readers ·Rᵀ, writers R·); q/k/v, gate/up (and experts'
      gate/up) are then encoded WITHOUT an input rotation, so only o_proj and down_proj inputs are rotated online.
      A tied LM head costs two online rotations per token at the final norm (RotatedFinalNorm), none per block.
  --fold-v-heads: fold a per-head Hadamard into v_proj rows / o_proj columns (with or without --fold): o_proj needs
      no online input rotation. NOTE: --fold alone destroys quality on Qwen3 (docs/benchmarks.md R3: massive
      activations); the quality-preserving setting is --fold-v-heads --one-sided.
  One-sided rotations (W' = W Vᵀ, no inverse rotation launches) are the default (R1: quality-neutral);
  --two-sided restores U W Vᵀ (--one-sided is accepted for older command lines and is the default).
  --no-fused-staging: fused q|k|v / gate|up kernels never stage codebooks in threadgroup memory (the path before
      K6; on the M1 Ultra staging is the faster placement, so this isolates the fused-group kernel regression).
  --fuse-elementwise auto|on|off: input rotations fused into the input / post-attention RMSNorm and the SwiGLU
      (suc.runtime.compressed_model.fuse_elementwise_rotations, K6) or run as their own launches; auto (default) =
      the per-device autotune choice (suc.kernels.autotune), else off. --no-fuse-elementwise = off.
  --rotation-layout auto|blocks|row: online rotation kernel layout (suc.kernels.rotation.LAYOUT; R5: "blocks" =
      one threadgroup per output block, "row" = one threadgroup per row); --rotation-strided = the pre-R5 kernel
      (row layout, strided scalar loads, H_m from memory, float32 signs) for before/after runs.
  --mma-min-batch auto|<int>: batch tile from which the simdgroup-matrix kernel is used (auto: per-device tuned
      crossover, suc.kernels.autotune, else 5 — the pre-R5 constant).
  The JSON records rotations_per_block (online rotation launches in block 0; "input_fused" = input rotations done
  inside a norm / SwiGLU launch) and launches_per_block (non-view primitives of one block call, B = 1).

Non-SUC tensors of the SUC variants: norms BF16; embedding (+ untied lm_head) MLX affine 4-bit g64 with --embed q4
(default, the same representation as in mlx_q4, so SUC vs mlx_q4 differs only in the 7 projections per block) or
BF16 with --embed bf16. Conversion is tensor-by-tensor (encode → SucLinear → dense weight dropped), so the peak is
one BF16 model plus one tensor's encoding.

Correctness (--check-seqs > 0, SUC variants): a second copy of the model gets every converted weight replaced by
codec.decode() (cast to BF16 as the model stores it; same embedding representation); next-token KL(decoded ||
compressed) and top-1 agreement on --check-seqs fixed text windows. With --check-f32 (default for < 3e9 params)
also in float32 (then the gap is rounding only; BF16 differences include BF16 rounding of the decoded weights and
of the rotated activations). The SUC module arrays are restored bit-exactly after the float32 pass.

Speed: suc.evaluation.speed.measure_generation (random prompt, greedy, synchronous per-token eval) per batch.
Resident weights = bytes of every array in the module tree after conversion; active memory = mx.get_active_memory
after gc + mx.clear_cache.

Profile (batch 1 decode, --no-profile to skip): the decode step is re-timed with the 7 projections of every block
replaced by stand-ins of the same output shape:
  null         zeros (no linear work)              → "rest" = embedding / LM head / attention / norms / glue
  kernel_only  SUC Metal kernel only (no rotations, no residual)
  no_residual  SucLinear incl. rotations, residual skipped
  full         the converted model
  kernel = kernel_only − null, rotations = no_residual − kernel_only, residual = full − no_residual.
For bf16 / mlx_q4 only full and null. Stand-ins are compiled like the modules when compile is on. With groups, kernel_only / no_residual are group modes (SucInputGroup.mode).
MLX op count per call is taken from the lazy graph (mx.export_to_dot) of one call, B = 1, per unit of the first
block (a group counts once, for all its members).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suc.analysis.sensitivity import get_weight, set_weight  # noqa: E402
from suc.evaluation.baselines import apply_representation, logits_kl  # noqa: E402
from suc.evaluation.speed import measure_generation, timed_load  # noqa: E402
import suc.kernels.rotation as rotation  # noqa: E402
import suc.kernels.suc_linear as suc_linear  # noqa: E402
import suc.runtime.compressed_model as compressed_model  # noqa: E402
from suc.evaluation.reconstruct import rotation_seeds  # noqa: E402
from suc.kernels.suc_linear import SucLinear  # noqa: E402
from suc.models.adapters import get_adapter  # noqa: E402
from suc.models.rotation_folding import ResidualFold, fold_model  # noqa: E402
from suc.provenance import git_info  # noqa: E402
from suc.quant.codecs import make_codec  # noqa: E402
from suc.quant.residual import ResidualCodec  # noqa: E402
from suc.runtime.compressed_model import (SucGroupMember, SucInputGroup, SucResidualLinear,  # noqa: E402
                                          convert_to_compressed, group_projections, merge_reports,
                                          resident_weight_bytes)

SUC_VARIANTS = {"vq_g4_k256": {"kind": "vq", "k": 256, "block_size": 4, "rotate": True},
                "add3_g8_k256": {"kind": "additive", "k": 256, "block_size": 8, "num_codebooks": 3, "rotate": True},
                # best quality configuration (ablation baseline, docs/ablations_plan.md); not in the default list
                "add4_g8_k256": {"kind": "additive", "k": 256, "block_size": 8, "num_codebooks": 4, "rotate": True},
                # low-dimensional single codebooks (Q5c; not in the default list): 4 / 3 / 2 bpw at G = 2, 3 bpw at
                # G = 3 (9-bit indices). G = 3 needs every eligible in_features % 3 == 0 — not the case for Qwen3
                # (hidden 1024 / 4096): such variants are reported as skipped, the run goes on.
                "vq_g2_k256": {"kind": "vq", "k": 256, "block_size": 2, "rotate": True},
                "vq_g2_k64": {"kind": "vq", "k": 64, "block_size": 2, "rotate": True},
                "vq_g2_k16": {"kind": "vq", "k": 16, "block_size": 2, "rotate": True},
                "vq_g3_k512": {"kind": "vq", "k": 512, "block_size": 3, "rotate": True}}
NON_DEFAULT = ("add4_", "vq_g2_", "vq_g3_")
VARIANTS = ["bf16", "mlx_q4"] + [f"{v}{r}" for v in SUC_VARIANTS for r in ("", "_res")]
DEFAULT_REVISIONS = {"Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
                     "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218"}
CHECK_TEXT = ("The history of the printing press begins in the fifteenth century, when movable metal type made it "
              "possible to reproduce books quickly and cheaply. def fib(n):\n    return n if n < 2 else "
              "fib(n - 1) + fib(n - 2)\nSolve for x: 3x + 7 = 22, so x = 5. Please summarise the main argument of the "
              "essay in two sentences, then list three counter-arguments a careful reader might raise. ")
SEED = 12345


# ------------------------------------------------------------------------------------------------ helpers

def busy_processes() -> str:
    try:
        return subprocess.run(["pgrep", "-fl", "suc quantize|python|strata"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


def release() -> None:
    gc.collect()
    mx.clear_cache()


def eligible_names(model) -> list[str]:
    return list(eligible_roles(model))


def eligible_roles(model) -> dict[str, str]:
    """{checkpoint name: role} of the eligible linears, in adapter order."""
    adapter = get_adapter({"model_type": model.args.model_type})
    return {s.name: s.role for s in adapter.linears(k for k, _ in tree_flatten(model.parameters())) if s.eligible}


def model_adapter(model):
    return get_adapter(dataclasses.asdict(model.args))


def residual_fold(model, args) -> ResidualFold:
    return ResidualFold.for_adapter(model_adapter(model), seed=SEED, v_heads=args.fold_v_heads, residual=args.fold)


def apply_fold(model, args) -> dict:
    """Residual-stream rotation folded into the BF16 weights (suc.models.rotation_folding), in place."""
    rep = fold_model(model, model_adapter(model), residual_fold(model, args))
    return {"tensors": len(rep["folded"]), "residual": args.fold, "tied_head": rep["tied_head"],
            "v_heads": args.fold_v_heads}


def module_at(model, path: str):
    *pre, attr = path.split(".")
    cur = model
    for p in pre:
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    return cur, attr


def projection_paths(model) -> list[str]:
    """Module paths of the 7 projections per block (in whatever representation they currently are)."""
    pat = re.compile(r"^model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)$")
    return [p for p, m in model.named_modules() if pat.match(p)]


def quantize_embedding(model) -> None:
    nn.quantize(model, group_size=64, bits=4, mode="affine",
                class_predicate=lambda p, m: p in ("model.embed_tokens", "lm_head") and hasattr(m, "to_quantized"))
    mx.eval(model.parameters())


def codec_for(variant: str, args, rotate_in: bool = True):
    base = SUC_VARIANTS[variant.removesuffix("_res")]
    base = make_codec(dict(base, kmeans_iters=args.kmeans_iters, fit_blocks=args.fit_blocks, rotate_in=rotate_in,
                           rotate_out=args.rotate_out))
    if variant.endswith("_res"):
        return ResidualCodec(base, args.residual_fraction, value_dtype="int8", selection="magnitude",
                             structure=args.residual_structure, column_share=args.residual_column_share)
    return base


def check_tokens(tokenizer, n: int, length: int = 64) -> mx.array:
    ids = tokenizer.encode(CHECK_TEXT * (1 + n * length // 40))
    return mx.array([ids[i * length:(i + 1) * length] for i in range(n)])


def suc_modules(model) -> dict[str, SucLinear]:
    """Every module holding SUC arrays (standalone, group parts and fused parts)."""
    return {p: m for p, m in model.named_modules() if isinstance(m, SucLinear)}


# ------------------------------------------------------------------------------------------- build models

class UnsupportedVariant(ValueError):
    """The variant's code layout does not fit this model (e.g. G = 3 on 4096-wide inputs)."""


def build(variant: str, args) -> tuple[nn.Module, dict]:
    model, tokenizer, load_s = timed_load(args.model, revision=args.revision)
    info: dict = {"load_s": load_s, "parameters_bf16_bytes": resident_weight_bytes(model)}
    if variant == "bf16":
        return model, info
    if variant == "mlx_q4":
        t0 = time.perf_counter()
        apply_representation(model, "mlx_affine", bits=4)
        info["quantize_s"] = time.perf_counter() - t0
        release()
        return model, info
    fold = None
    if args.fold or args.fold_v_heads:
        info["fold"] = apply_fold(model, args)
        fold = residual_fold(model, args)
    if args.embed == "q4":
        quantize_embedding(model)
    ref = None
    if args.check_seqs:
        ref, _, _ = timed_load(args.model, revision=args.revision)
        if args.fold or args.fold_v_heads:
            apply_fold(ref, args)
        if args.embed == "q4":
            quantize_embedding(ref)
        if args.check_f32:
            ref.set_dtype(mx.float32)   # decoded weights kept exact; the bf16 copy is derived in check()
    g = SUC_VARIANTS[variant.removesuffix("_res")]["block_size"]
    bad = sorted({get_weight(model, n).shape[1] for n in eligible_roles(model) if get_weight(model, n).shape[1] % g})
    if bad:
        raise UnsupportedVariant(f"{variant}: in_features {bad} not divisible by G={g}")
    codec = codec_for_plain = codec_for(variant, args)
    # folded model: tensors reading the rotated residual stream need no input rotation
    codec_rot = codec_for(variant, args, rotate_in=False) if fold is not None else codec
    reports, enc_bits, enc_params = [], 0, 0
    t0 = time.perf_counter()
    for i, (name, role) in enumerate(eligible_roles(model).items()):
        codec = codec_rot if (fold is not None and fold.rotated_input(role)) else codec_for_plain
        w = get_weight(model, name)
        e = codec.encode(w, seed=SEED + i, **(rotation_seeds(name, SEED) if args.rotation_groups else {}))
        mx.eval(e.arrays)
        enc_bits += e.bits()["total"]
        enc_params += w.size
        if ref is not None:
            set_weight(ref, name, codec.decode(e))
            mx.eval(get_weight(ref, name))
        del w
        reports.append(convert_to_compressed(model, {name: e}, clear_cache=(i % 32 == 0), group=False))
        del e
        if args.verbose and i % 32 == 0:
            print(f"#   [{variant}] {i + 1} {name} {time.perf_counter() - t0:.0f}s", flush=True)
    groups = group_projections(model, fuse=args.fuse)
    release()
    rep = merge_reports(reports)
    rep["active_memory_after"] = mx.get_active_memory()
    info.update(encode_convert_s=time.perf_counter() - t0, codec=codec.name,
                encoded_params=enc_params, encoded_bits_storage=enc_bits, encoded_bpw_storage=enc_bits / enc_params,
                encoded_bpw_resident=rep["bytes_after"] * 8 / enc_params,
                conversion={k: v for k, v in rep.items() if k != "converted" and not k.startswith("active_memory")},
                n_residual=sum(r["n_residual"] for r in rep["converted"]),
                groups=len(groups), fused_groups=sum(g["fused_parts"] > 0 for g in groups),
                runtime={"rotation": suc_linear.ROTATION, "rotation_groups": args.rotation_groups, "fuse": args.fuse,
                         "compile": suc_linear.COMPILE, "fuse_residual": compressed_model.FUSE_RESIDUAL,
                         "residual_structure": args.residual_structure,
                         "residual_column_share": args.residual_column_share, "fold": args.fold, "fold_v_heads": args.fold_v_heads,
                         "rotate_out": args.rotate_out, "fuse_elementwise": {op: compressed_model.fuse_elementwise(op) for op in ("norm", "swiglu")},
                         "rotation_kernel": rotation_kernel_info(model),
                         "sign_dtype": str(suc_linear.SIGN_DTYPE),
                         "mma_min_batch": {m: suc_linear._vqm.mma_min_batch(m) for m in (1, 3)},
                         "fused_staging": suc_linear.FUSED_STAGING,
                         "codebook_in": {m: suc_linear._vqm.resolve_codebook_in(suc_linear._vqm.CODEBOOK_IN, m=m)
                                         for m in (1, 3)}},
                rotations_per_block=rotations_per_block(model))
    if ref is not None:
        info["check"] = check(ref, model, tokenizer, args)
        del ref
        release()
    return model, info


def rotation_kernel_info(model) -> dict:
    """Rotation kernel settings and, per online rotation size of block 0, the resolved layout / block shape."""
    from suc.quant.hadamard import _factor
    try:
        sizes = sorted({m.in_features for _, m in model.model.layers[0].named_modules()
                        if isinstance(m, SucLinear) and m.seed_in is not None})
    except AttributeError:
        sizes = []
    per = {}
    for n in sizes:
        lay = rotation.layout_for(n)
        if lay == "blocks" and rotation.supported(n):
            m_, p_ = _factor(n)
            q, t = rotation.blocks_shape((p_,), (m_,), *rotation.block_config(n))
            per[n] = {"layout": lay, "q": q, "threads": t}
        else:
            per[n] = {"layout": lay}
    return {"LAYOUT": rotation.LAYOUT, "VEC": rotation.VEC, "HM_CONST": rotation.HM_CONST, "sizes": per}


def rotations_per_block(model) -> dict:
    """Online rotation launches per decoder block of the first block (input side: one per distinct input that is
    rotated — a group shares it; output side: one per standalone SucLinear / fused part with seed_out)."""
    pre = "model.layers.0."
    n_in = n_out = n_fused = 0
    fused_groups = {id(m._group.obj) for p, m in model.named_modules()
                    if p.startswith(pre) and isinstance(m, compressed_model.RotatedRMSNorm)}
    fused_down = {id(m.down_proj) for p, m in model.named_modules()
                  if p.startswith(pre) and isinstance(m, compressed_model.SucMLP)}
    for p, m in model.named_modules():
        if not p.startswith(pre):
            continue
        if isinstance(m, SucInputGroup):
            r = m.parts[0].seed_in is not None
            n_fused += r and id(m) in fused_groups
            n_in += r and id(m) not in fused_groups
            n_out += sum(part.seed_out is not None for part in m.parts)
        elif isinstance(m, SucLinear) and ".suc_" not in p and ".parts." not in p:
            r = m.seed_in is not None
            n_fused += r and id(m) in fused_down
            n_in += r and id(m) not in fused_down
            n_out += m.seed_out is not None
    return {"input": n_in, "input_fused": n_fused, "output": n_out, "total": n_in + n_out}


def check(ref, comp, tokenizer, args) -> dict:
    """KL(decoded || compressed). bf16 always; with check_f32, ref holds the exact (float32) decoded weights:
    f32 = KL(decoded f32 || compressed f32) (rounding only), bf16 = KL(decoded f32 || compressed bf16) next to the
    noise floor KL(decoded f32 || decoded bf16), and bf16_vs_decoded_bf16 = KL(decoded bf16 || compressed bf16)."""
    toks = check_tokens(tokenizer, args.check_seqs)
    l_c16 = comp(toks)
    if not args.check_f32:
        return {"tokens": int(toks.size), "bf16": logits_kl(ref(toks), l_c16)}
    l_ref32 = ref(toks)
    saved = {p: dict(tree_flatten(m.parameters())) for p, m in suc_modules(comp).items()}
    comp.set_dtype(mx.float32)
    l_c32 = comp(toks)
    comp.set_dtype(mx.bfloat16)
    for p, m in suc_modules(comp).items():
        m.update(dict(saved[p]))     # fp16 codebooks / f32 scales / f32 signs back, bit-exact
    mx.eval(comp.parameters())
    ref.set_dtype(mx.bfloat16)
    l_ref16 = ref(toks)
    return {"tokens": int(toks.size), "f32": logits_kl(l_ref32, l_c32), "bf16": logits_kl(l_ref32, l_c16),
            "bf16_noise_floor": logits_kl(l_ref32, l_ref16), "bf16_vs_decoded_bf16": logits_kl(l_ref16, l_c16)}


# --------------------------------------------------------------------------------------------- profiling

class _Null(nn.Module):
    def __init__(self, out: int):
        super().__init__()
        self.out = out

    def __call__(self, x):
        return mx.zeros((*x.shape[:-1], self.out), dtype=x.dtype)


class _KernelOnly(nn.Module):
    def __init__(self, m: SucLinear):
        super().__init__()
        self.m = m

    def __call__(self, x):
        f = lambda x2: self.m.kernel(x2)
        x2 = x.reshape(-1, x.shape[-1])
        y = suc_linear.compiled_call(self.m, f, "kernel_only", x2) if suc_linear.COMPILE else f(x2)
        return y.reshape(*x.shape[:-1], self.m.out_features)


class _NoResidual(nn.Module):
    def __init__(self, m: SucLinear):
        super().__init__()
        self.m = m

    def __call__(self, x):
        x2 = x.reshape(-1, self.m.in_features)
        f = lambda x2: SucLinear.from_rotated(self.m, self.m.rotate_in(x2), x2)   # base class: residual skipped
        y = suc_linear.compiled_call(self.m, f, "no_residual", x2) if suc_linear.COMPILE else f(x2)
        return y.reshape(*x.shape[:-1], self.m.out_features)


def _out_features(m) -> int:
    if isinstance(m, (SucLinear, SucGroupMember)):
        return m.out_features
    return int(m.weight.shape[0])


def with_stand_ins(model, mode: str):
    """Swap the 7 projections per block for stand-ins (groups: set their mode); returns a restore() callable."""
    swapped, groups = [], [m for _, m in model.named_modules() if isinstance(m, SucInputGroup)]
    mlps = [m for _, m in model.named_modules() if isinstance(m, compressed_model.SucMLP)]
    for g in groups:
        g.mode = mode           # "null": the fused norms fall back to plain RMSNorm (the group is not called)
    for p in projection_paths(model):
        parent, attr = module_at(model, p)
        m = parent[attr]
        if mode == "no_residual" and isinstance(parent, compressed_model.SucMLP) and attr == "down_proj":
            parent.mode = mode  # keeps the SwiGLU-fused input rotation, skips the residual
            continue
        if mode == "null":
            new = _Null(_out_features(m))
        elif isinstance(m, SucGroupMember):
            new = m
        elif mode == "kernel_only":
            new = _KernelOnly(m)
        elif mode == "no_residual":
            new = _NoResidual(m) if isinstance(m, SucResidualLinear) else m
        else:
            raise ValueError(mode)
        parent[attr] = new
        swapped.append((parent, attr, m))

    def restore():
        for parent, attr, m in swapped:
            parent[attr] = m
        for g in groups:
            g.mode = "full"
        for m in mlps:
            m.mode = "full"
    return restore


def decode_ms(model, args, batch: int = 1) -> float:
    r = measure_generation(model, args.prompt_len, args.decode_tokens, batch_size=batch, warmup=1,
                           repeats=args.repeats, seed=SEED)
    return 1e3 * batch / r["decode_tok_s"]


_VIEWS = {"Reshape", "Broadcast", "ExpandDims", "Squeeze", "Transpose", "Slice", "StopGradient", "Flatten", "Unflatten"}


def op_count(module, in_features: int, dtype=mx.bfloat16) -> dict | None:
    """Primitives in the lazy graph of one call (B = 1, evaluated input) via mx.export_to_dot: all of them, and
    those that launch work (views — reshape / broadcast / expand_dims / squeeze / transpose / slice — excluded)."""
    try:
        x = mx.zeros((1, 1, in_features), dtype=dtype)
        mx.eval(x)
        y = module(x)
        with tempfile.NamedTemporaryFile("w+", suffix=".dot") as f:
            mx.export_to_dot(f.name, *(y if isinstance(y, list) else [y]))
            ops = re.findall(r'label ="(\w+)", shape=rectangle', open(f.name).read())
        mx.eval(y)
        return {"all": len(ops), "kernels": sum(o not in _VIEWS for o in ops),
                "names": [o for o in ops if o not in _VIEWS]}
    except Exception as ex:  # noqa: BLE001 — diagnostic only
        return {"error": repr(ex)}


def profile(model, variant: str, args) -> dict:
    out = {"full_ms": decode_ms(model, args)}
    modes = ["null"] + (["kernel_only"] if variant not in ("bf16", "mlx_q4") else []) + \
            (["no_residual"] if variant.endswith("_res") else [])
    for mode in modes:
        restore = with_stand_ins(model, mode)
        try:
            out[f"{mode}_ms"] = decode_ms(model, args)
        finally:
            restore()
    out["rest_ms"] = out["null_ms"]
    out["linears_ms"] = out["full_ms"] - out["null_ms"]
    out["linears_fraction"] = out["linears_ms"] / out["full_ms"]
    if "kernel_only_ms" in out:
        nores = out.get("no_residual_ms", out["full_ms"])
        out["kernel_ms"] = out["kernel_only_ms"] - out["null_ms"]
        out["rotation_ms"] = nores - out["kernel_only_ms"]
        out["residual_ms"] = out["full_ms"] - nores
    units = {}
    for p in projection_paths(model)[:7]:
        par, a = module_at(model, p)
        m = par[a]
        if isinstance(m, SucGroupMember):
            units.setdefault("+".join(m.group.names), (m.group, m.in_features))
        else:
            units[a] = (m, _in_features(m))
    out["ops_per_call"] = {k: op_count(mod, n) for k, (mod, n) in units.items()}
    out["kernels_per_block"] = sum(o.get("kernels", 0) for o in out["ops_per_call"].values())
    out["launches_per_block"] = block_op_count(model)
    return out


def block_op_count(model, dtype=mx.bfloat16) -> dict:
    """Non-view primitives of one call of decoder block 0 (B = 1, one token, no cache): every launch of the block
    — projections, fused norm / SwiGLU rotations, attention, norms, residual adds."""
    try:
        layer = model.model.layers[0]
        x = mx.zeros((1, 1, model.args.hidden_size), dtype=dtype)
        mx.eval(x)
        y = layer(x, None, None)
        with tempfile.NamedTemporaryFile("w+", suffix=".dot") as f:
            mx.export_to_dot(f.name, y)
            ops = re.findall(r'label ="(\w+)", shape=rectangle', open(f.name).read())
        mx.eval(y)
        ks = [o for o in ops if o not in _VIEWS]
        return {"kernels": len(ks), "custom_kernels": ks.count("CustomKernel"),
                "names": {o: ks.count(o) for o in sorted(set(ks))}}
    except Exception as ex:  # noqa: BLE001 — diagnostic only
        return {"error": repr(ex)}


def _in_features(m) -> int:
    if isinstance(m, (SucLinear, SucGroupMember)):
        return m.in_features
    if isinstance(m, nn.QuantizedLinear):
        return m.weight.shape[1] * 32 // m.bits
    return int(m.weight.shape[1])


# ------------------------------------------------------------------------------------------------- main

def run_variant(variant: str, args) -> dict:
    release()
    try:
        model, info = build(variant, args)
    except UnsupportedVariant as ex:
        print(f"{variant:18s} skipped: {ex}", flush=True)
        release()
        return {"variant": variant, "skipped": str(ex), "generation": []}
    release()
    res = {"variant": variant, **info, "resident_weight_bytes": resident_weight_bytes(model),
           "active_memory_bytes": mx.get_active_memory(), "generation": []}
    for b in args.batches:
        r = measure_generation(model, args.prompt_len, args.decode_tokens, batch_size=b, warmup=args.warmup,
                               repeats=args.repeats, seed=SEED)
        r["repeats"] = [{k: v for k, v in rep.items() if k != "tokens"} for rep in r["repeats"]]
        res["generation"].append(r)
        print(f"{variant:18s} B={b:<3d} ttft={r['ttft_s'] * 1e3:8.1f} ms  prefill={r['prefill_tok_s']:8.0f} tok/s  "
              f"decode={r['decode_tok_s']:7.1f} tok/s  peak={r['peak_memory_bytes'] / 2**20:7.0f} MiB", flush=True)
    if args.profile:
        res["profile_b1"] = profile(model, variant, args)
        pr = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["profile_b1"].items()
              if k != "ops_per_call"}
        print(f"{variant:18s} profile {json.dumps(pr)}", flush=True)
    if "check" in res:
        print(f"{variant:18s} check {json.dumps(res['check'])}", flush=True)
    print(f"{variant:18s} resident weights {res['resident_weight_bytes'] / 2**20:.0f} MiB, active "
          f"{res['active_memory_bytes'] / 2**20:.0f} MiB", flush=True)
    del model
    release()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--variants", default=",".join(v for v in VARIANTS if not v.startswith(NON_DEFAULT)))
    ap.add_argument("--batches", default="1,4,16")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--embed", choices=("q4", "bf16"), default="q4")
    ap.add_argument("--kmeans-iters", type=int, default=6)
    ap.add_argument("--fit-blocks", type=int, default=65536)
    ap.add_argument("--residual-fraction", type=float, default=0.005)
    ap.add_argument("--residual-structure", choices=("csr", "columns"), default="csr",
                    help="_res variants: CSR positions, or dense int8 input columns at the same bit budget (Q5b)")
    ap.add_argument("--residual-column-share", type=float, default=1.0,
                    help="columns: budget share of the columns, the rest is a CSR tail")
    ap.add_argument("--no-fuse-residual", dest="fuse_residual", action="store_false",
                    help="residual members are not fused into q|k|v / gate|up launches (the earlier path)")
    ap.add_argument("--check-seqs", type=int, default=4)
    ap.add_argument("--check-f32", choices=("auto", "yes", "no"), default="auto")
    ap.add_argument("--no-profile", dest="profile", action="store_false")
    ap.add_argument("--rotation", choices=("fused", "mlx"), default="fused")
    ap.add_argument("--no-rotation-groups", dest="rotation_groups", action="store_false")
    ap.add_argument("--no-fuse", dest="fuse", action="store_false")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--fold", action="store_true", help="fold a residual-stream rotation into the weights first")
    ap.add_argument("--fold-v-heads", action="store_true", help="per-head Hadamard folded into v_proj / o_proj")
    ap.add_argument("--one-sided", dest="rotate_out", action="store_false",
                    help="encode without output-side rotations (the default; kept for older command lines)")
    ap.add_argument("--two-sided", dest="rotate_out", action="store_true",
                    help="encode with output-side rotations too (U W Vᵀ; online inverse rotation launches)")
    ap.add_argument("--no-fused-staging", dest="fused_staging", action="store_false",
                    help="fused q|k|v / gate|up kernels read codebooks from device memory (the earlier path)")
    ap.add_argument("--fuse-elementwise", choices=("auto", "on", "off"), default="auto",
                    help="input rotations fused into RMSNorm / SwiGLU (auto: per-device tuned choice, else off)")
    ap.add_argument("--no-fuse-elementwise", dest="fuse_elementwise", action="store_const", const="off",
                    help="= --fuse-elementwise off")
    ap.add_argument("--rotation-layout", choices=("auto", "blocks", "row"), default="auto",
                    help="online rotation kernel layout (suc.kernels.rotation.LAYOUT)")
    ap.add_argument("--rotation-strided", action="store_true",
                    help="the pre-R5 rotation kernel (row layout, strided scalar loads, H_m read from memory, "
                         "float32 signs)")
    ap.add_argument("--mma-min-batch", default="auto",
                    help="vec → simdgroup-matrix crossover batch (int), or auto (per-device tuned, else 5)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", default=None)
    ap.set_defaults(rotate_out=False)
    args = ap.parse_args()
    args.revision = args.revision or DEFAULT_REVISIONS.get(args.model)
    args.batches = [int(b) for b in args.batches.split(",")]
    suc_linear.ROTATION = args.rotation
    suc_linear.COMPILE = args.compile
    compressed_model.FUSE_RESIDUAL = args.fuse_residual
    compressed_model.FUSE_ELEMENTWISE = {"auto": None, "on": True, "off": False}[args.fuse_elementwise]
    rotation.LAYOUT = args.rotation_layout
    if args.rotation_strided:
        rotation.LAYOUT, rotation.VEC, rotation.HM_CONST = "row", 1, False
        suc_linear.SIGN_DTYPE = mx.float32
    suc_linear._vqm.MMA_MIN_BATCH = None if args.mma_min_batch == "auto" else int(args.mma_min_batch)
    suc_linear.FUSED_STAGING = args.fused_staging
    variants = args.variants.split(",")
    if bad := [v for v in variants if v not in VARIANTS]:
        raise SystemExit(f"unknown variants {bad}; known: {VARIANTS}")
    is_small = "0.6B" in args.model or "1.7B" in args.model
    args.check_f32 = {"auto": is_small, "yes": True, "no": False}[args.check_f32]
    dev = mx.device_info().get("device_name", platform.machine())
    others = busy_processes()
    print(f"# device={dev} mlx={mx.__version__} mlx_lm={mlx_lm.__version__} model={args.model}@{args.revision}")
    print(f"# other processes at start:\n{others}" if others else "# no other python/suc processes")
    t0 = time.time()
    results = [run_variant(v, args) for v in variants]
    summary = summarize(results)
    print(json.dumps(summary, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(benchmark="e2e_generate", device=dev, mlx=mx.__version__, mlx_lm=mlx_lm.__version__,
                           memory=mx.device_info(), platform=platform.platform(), git=git_info(),
                           model=args.model, revision=args.revision, args=vars(args), seed=SEED,
                           timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), other_processes=others,
                           other_processes_end=busy_processes(), wall_s=time.time() - t0,
                           results=results, summary=summary), f, indent=1)
        print(f"# wrote {args.json}")


def summarize(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        if "skipped" in r:
            rows.append({"variant": r["variant"], "skipped": r["skipped"]})
            continue
        row = {"variant": r["variant"], "resident_MiB": round(r["resident_weight_bytes"] / 2**20, 1),
               "linears_bpw_resident": round(r.get("encoded_bpw_resident", float("nan")), 3)
               if "encoded_bpw_resident" in r else None}
        for g in r["generation"]:
            b = g["batch_size"]
            row[f"B{b}_ttft_ms"] = round(g["ttft_s"] * 1e3, 1)
            row[f"B{b}_prefill_tok_s"] = round(g["prefill_tok_s"])
            row[f"B{b}_decode_tok_s"] = round(g["decode_tok_s"], 1)
            row[f"B{b}_decode_ms_step"] = round(1e3 * b / g["decode_tok_s"], 2)
            row[f"B{b}_peak_MiB"] = round(g["peak_memory_bytes"] / 2**20)
        if "profile_b1" in r:
            row["profile_b1"] = {k: round(v, 3) for k, v in r["profile_b1"].items() if isinstance(v, float)}
        if "check" in r:
            row["check_kl_bf16"] = r["check"]["bf16"]["kl_mean"]
            if "f32" in r["check"]:
                row["check_kl_f32"] = r["check"]["f32"]["kl_mean"]
                row["check_kl_bf16_noise_floor"] = r["check"]["bf16_noise_floor"]["kl_mean"]
        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
