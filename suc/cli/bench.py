"""Phase 0 baseline benchmark: one config -> one machine-readable JSON result (spec §7)."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from suc.config import ExperimentConfig, RepresentationConfig, config_hash, config_to_dict, load_config
from suc.evaluation.baselines import apply_representation, logits_kl, size_report
from suc.evaluation.data import load_wikitext2, tokenize
from suc.evaluation.perplexity import perplexity
from suc.evaluation.speed import measure_generation, timed_load
from suc.io.hashing import model_hash
from suc.io.safetensors_store import SafetensorsStore
from suc.models.adapters import get_adapter
from suc.provenance import run_metadata, seed_everything, write_result

KL_TOKENS = 512  # tokens of the first eval window used for logit KL vs the first representation


def resolve_model_dir(cfg: ExperimentConfig) -> Path:
    if cfg.model.local_path:
        return Path(cfg.model.local_path)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        cfg.model.id, revision=cfg.model.revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    ))


def resolved_revision(model_dir: Path, requested: str) -> str:
    """Exact commit of a Hugging Face cache snapshot (snapshots/<sha>); `requested` for plain local dirs."""
    if model_dir.parent.name == "snapshots" and len(model_dir.name) == 40:
        return model_dir.name
    return requested


def architecture_summary(model_dir: Path, store: SafetensorsStore) -> dict:
    hf_config = json.loads((model_dir / "config.json").read_text())
    adapter = get_adapter(hf_config)
    names = store.names()
    adapter.validate(names)
    linears = adapter.linears(names)
    eligible = [s for s in linears if s.eligible]
    eligible_params = sum(_numel(store.info(s.name).shape) for s in eligible)
    return {
        "family": adapter.family,
        "num_layers": adapter.num_layers,
        "is_moe": adapter.is_moe,
        "num_experts": adapter.num_experts,
        "checkpoint_tensors": len(names),
        "checkpoint_bytes": store.total_bytes(),
        "checkpoint_params": sum(_numel(store.info(n).shape) for n in names),
        "eligible_linear_tensors": len(eligible),
        "eligible_linear_params": eligible_params,
        "checkpoint_dtypes": sorted({store.info(n).dtype for n in names}),
    }


def identity_check(model, store: SafetensorsStore) -> dict:
    """Loaded model parameters must equal the raw checkpoint tensors bit-exactly (T0 acceptance)."""
    params = dict(tree_flatten(model.parameters()))
    compared, max_abs, mismatched = 0, 0.0, []
    for name in store.names():
        if name not in params:
            continue
        ref, got = store.load(name), params[name]
        if ref.shape != got.shape or ref.dtype != got.dtype:
            mismatched.append(name)
            continue
        err = mx.max(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))).item()
        max_abs = max(max_abs, err)
        compared += 1
    return {
        "tensors_compared": compared,
        "tensors_in_checkpoint": len(store),
        "shape_or_dtype_mismatch": mismatched,
        "max_abs_error": max_abs,
        "exact": max_abs == 0.0 and not mismatched,
    }


def run_representation(rep: RepresentationConfig, cfg: ExperimentConfig, model_dir: Path,
                       tokens: mx.array, ref_logits: mx.array | None,
                       store: SafetensorsStore) -> tuple[dict, mx.array]:
    mx.reset_peak_memory()
    model, _tok, load_s = timed_load(str(model_dir))
    identity = identity_check(model, store)
    t0 = time.perf_counter()
    apply_representation(model, rep.kind, bits=rep.bits, group_size=rep.group_size)
    mx.eval(model.parameters())
    convert_s = time.perf_counter() - t0

    kl_window = tokens[: min(KL_TOKENS, cfg.dataset.seq_len)][None]
    logits = model(kl_window)[0].astype(mx.float32)
    mx.eval(logits)
    kl = logits_kl(ref_logits, logits) if ref_logits is not None else None

    ppl = perplexity(model, tokens, cfg.dataset.seq_len, max_tokens=cfg.dataset.max_tokens)
    speed = [
        measure_generation(model, p, cfg.bench.decode_tokens, batch_size=b,
                           warmup=cfg.bench.warmup, repeats=cfg.bench.repeats, seed=cfg.seed)
        for b in cfg.bench.batch_sizes for p in cfg.bench.prompt_lengths
    ]
    result = {
        "representation": {"name": rep.name, "kind": rep.kind, "bits": rep.bits, "group_size": rep.group_size},
        "load_s": load_s,
        "convert_s": convert_s,
        "identity_check": identity,
        "size": size_report(model),
        "perplexity": ppl,
        "kl_vs_reference": kl,
        "speed": speed,
        "peak_memory_bytes": mx.get_peak_memory(),
    }
    del model
    gc.collect()
    mx.clear_cache()
    return result, logits


def run(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    seed_everything(cfg.seed)
    model_dir = resolve_model_dir(cfg)
    store = SafetensorsStore.from_dir(model_dir)
    mhash = model_hash(model_dir)
    arch = architecture_summary(model_dir, store)

    text, dhash = load_wikitext2(cfg.dataset.split)
    _model, tokenizer, _ = timed_load(str(model_dir))
    del _model
    gc.collect()
    tokens = tokenize(tokenizer, text)
    if cfg.dataset.max_tokens is not None:
        tokens = tokens[: cfg.dataset.max_tokens]

    # Captured before any work so git commit/dirty describe the code that actually runs.
    metadata = run_metadata(
        model=cfg.model.id, model_revision=resolved_revision(model_dir, cfg.model.revision), model_hash=mhash,
        dataset_hash=dhash, seed=cfg.seed, config_hash=config_hash(cfg),
    )
    reps, ref_logits = [], None
    for rep in cfg.representations:
        res, logits = run_representation(rep, cfg, model_dir, tokens, ref_logits, store)
        if ref_logits is None:
            ref_logits = logits
            res["kl_vs_reference"] = {"reference": True}
        reps.append(res)

    result = {
        "metadata": metadata,
        "config": config_to_dict(cfg),
        "dataset": {"name": cfg.dataset.name, "split": cfg.dataset.split,
                    "tokens_total": int(tokens.size), "kl_reference_tokens": min(KL_TOKENS, cfg.dataset.seq_len)},
        "architecture": arch,
        "kl_reference": cfg.representations[0].name,
        "representations": reps,
    }
    return write_result(result, cfg.output_dir, cfg.name)


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n
