"""Quality-track runner (Q1–Q4): encode every selected tensor with one codec at a time, measure the whole model.

Per codec: fresh BF16 model → reconstruct (one tensor at a time, D8) → actual whole-model size (spec §25) →
WikiText-2 perplexity (final-test set) and next-token KL vs the BF16 model on held-out validation sequences of
the Phase 2 corpus (spec §32: nothing here is tuned on the test set).
"""

from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import load_model

from suc.analysis.activations import Recorder
from suc.cli.bench import resolved_revision, resolve_model_dir
from suc.config import QuantizeExperimentConfig, config_hash, config_to_dict, load_quantize_config
from suc.evaluation.data import load_wikitext2, tokenize
from suc.evaluation.reconstruct import (
    evaluate_quality, load_validation_tokens, model_size, reconstruct_model, reference_logits,
)
from suc.io.hashing import model_hash, sha256_file
from suc.io.safetensors_store import SafetensorsStore
from suc.models.adapters import get_adapter
from suc.models.rotation_folding import FoldedStore, ResidualFold, fold_block, fold_model, fold_params
from suc.models.streaming import block_prefix, load_lazy, stream_blocks
from suc.provenance import run_metadata, seed_everything, write_result
from suc.quant.codecs import make_codec
from suc.quant.residual import ResidualCodec
from suc.quant.stacks import set_codebook_scope

ARTIFACT_DIR = "experiments/artifacts"


def select_specs(adapter, names, roles, include_embeddings: bool):
    """Eligible linear tensors (optionally filtered by role) plus embedding / untied LM head if requested."""
    out = []
    for s in adapter.linears(names):
        if s.eligible and (roles is None or s.role in roles):
            out.append(s)
        elif include_embeddings and roles is None and s.role == "embed":
            out.append(s)
        elif include_embeddings and roles is None and s.role == "lm_head" and not adapter.tied_embeddings:
            out.append(s)
    return out


def default_calib_artifact(model_id: str) -> Path:
    slug = model_id.split("/")[-1].lower().replace("-", "_")
    found = sorted(Path(ARTIFACT_DIR).glob(f"*phase2_calib_{slug}_data.safetensors"))
    if not found:
        raise FileNotFoundError(f"no Phase 2 corpus artifact for {model_id} in {ARTIFACT_DIR}; run `suc calibrate` first")
    return found[-1]


_HEAD_ROLES = ("embed", "lm_head")


def default_activation_stats(model_id: str) -> Path:
    slug = model_id.split("/")[-1].lower().replace("-", "_")
    found = sorted(Path(ARTIFACT_DIR).glob(f"*phase2_calib_{slug}_activations.safetensors"))
    if not found:
        raise FileNotFoundError(f"no Phase 2 activation statistics for {model_id} in {ARTIFACT_DIR}")
    return found[-1]


def folded_activation_stats(model_dir: Path, adapter, fold, tokens: mx.array, micro_batch: int = 8) -> dict:
    """E[x_j^2] of every linear's input in the FOLDED model over `tokens` (calibration partition), block-streamed (D8):
    the embedding (and other non-block tensors) are folded first, each block is folded right before it runs."""
    model, _ = load_lazy(model_dir)
    outer = {k: v for k, v in tree_flatten(model.parameters()) if not k.startswith(block_prefix(0)[:-1])}
    model.update(tree_unflatten(list(fold_params(outer, adapter, fold).items())))
    rec = Recorder()

    def on_block(i, b):
        fold_block(b, block_prefix(i), adapter, fold)
        rec.instrument(b, block_prefix(i))

    stream_blocks(model, tokens, micro_batch=micro_batch, release=True, on_block=on_block,
                  after_block=lambda i, b: rec.remove(b))
    out = {k: m.second_moment_diag() for k, m in rec.moments.items()}
    mx.eval(out)
    del model
    gc.collect()
    mx.clear_cache()
    return out


def full_input_moments(model_dir: Path, names: list[str], tokens: mx.array, micro_batch: int = 8) -> dict:
    """E[x x^T] (in, in) of the input of every named dense linear over `tokens` (calibration partition),
    block-streamed (D8); q/k/v and gate/up share one matrix (same input). Keyed by name without ".weight"."""
    model, _ = load_lazy(model_dir)
    rec = Recorder(full_second_moment=names)
    stream_blocks(model, tokens, micro_batch=micro_batch, release=True,
                  on_block=lambda i, b: rec.instrument(b, block_prefix(i)), after_block=lambda i, b: rec.remove(b))
    want, cache, out = set(names), {}, {}
    for k, m in rec.moments.items():
        if k in want and m.full:
            if id(m) not in cache:
                cache[id(m)] = m.second_moment()
            out[k] = cache[id(m)]
    mx.eval(out)
    del model
    gc.collect()
    mx.clear_cache()
    return out


def _free(*objs) -> None:
    del objs
    gc.collect()
    mx.clear_cache()


def _per_role(tensors: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in tensors:
        groups[t["role"]].append(t)
    return {role: {"tensors": len(ts), "rel_mse_mean": sum(t["rel_mse"] for t in ts) / len(ts),
                   "rel_mse_max": max(t["rel_mse"] for t in ts), "bpw_mean": sum(t["bpw"] for t in ts) / len(ts)}
            for role, ts in sorted(groups.items())}


def run_quantize(config_path: str | Path) -> Path:
    cfg: QuantizeExperimentConfig = load_quantize_config(config_path)
    ev = cfg.eval
    seed_everything(cfg.seed)
    model_dir = resolve_model_dir(cfg)
    store = SafetensorsStore.from_dir(model_dir)
    adapter = get_adapter(json.loads((model_dir / "config.json").read_text()))
    names = store.names()
    adapter.validate(names)
    specs = select_specs(adapter, names, cfg.roles, cfg.include_embeddings)

    text, wiki_hash = load_wikitext2(ev.wikitext_split)
    ppl_tokens = tokenize(load_lazy(model_dir)[1], text)
    artifact = Path(ev.calib_artifact).expanduser() if ev.calib_artifact else default_calib_artifact(cfg.model.id)
    kl_tokens = load_validation_tokens(artifact, ev.kl_sequences)
    col_weights = None
    needs_stats = any(c.activation_weighted or (c.residual_fraction and c.residual_selection == "activation")
                      for c in cfg.codecs)
    if needs_stats:
        stats_path = (Path(cfg.activation_stats).expanduser() if cfg.activation_stats
                      else default_activation_stats(cfg.model.id))
        rec = Recorder.load(stats_path)
        col_weights = {k: m.second_moment_diag() for k, m in rec.moments.items()}
    # per-weight importance for activation-selected residuals: E[x_j^2] broadcast over rows (lazy views)
    importance = ({s.name: mx.broadcast_to(col_weights[s.name[:-7]][None], store.info(s.name).shape)
                   for s in specs if s.name[:-7] in col_weights} if col_weights is not None else None)
    # Captured before the heavy work so git commit/dirty describe the code that actually runs.
    metadata = run_metadata(model=cfg.model.id, model_revision=resolved_revision(model_dir, cfg.model.revision),
                            model_hash=model_hash(model_dir), dataset_hash=wiki_hash, seed=cfg.seed,
                            config_hash=config_hash(cfg))

    hessians = None
    if any(c.ldlq for c in cfg.codecs):
        hessians = full_input_moments(model_dir, [s.name[:-7] for s in specs if s.role not in _HEAD_ROLES],
                                      mx.load(str(artifact))["calibration"][:cfg.ldlq_sequences])
        print(f"[quantize] full input second moments of {len(hessians)} linears (LDLQ)", flush=True)

    model, _ = load_model(model_dir)
    t0 = time.perf_counter()
    reference = evaluate_quality(model, ppl_tokens=ppl_tokens, seq_len=ev.seq_len, max_tokens=ev.max_tokens)
    ref_logits = reference_logits(model, kl_tokens, chunk=ev.chunk)
    reference["seconds"] = time.perf_counter() - t0
    _free(model)

    print(f"[quantize] reference PPL {reference['perplexity']['ppl']:.3f} ({reference['seconds']:.0f} s)", flush=True)
    results = []
    folded_stats: dict[tuple, dict] = {}    # (fold_rotation, fold_v_heads) -> folded-basis E[x_j^2] per linear
    for c in cfg.codecs:
        mx.reset_peak_memory()
        model, _ = load_model(model_dir)
        c_store, c_col_weights, c_importance, fold, fold_info = store, col_weights, importance, None, None
        if c.fold_rotation or c.fold_v_heads:
            # residual-stream rotation folded into the weights (suc.models.rotation_folding); activation statistics
            # are re-recorded in the folded basis (the stored ones are diagonal in the original basis)
            fold = ResidualFold.for_adapter(adapter, seed=cfg.seed, v_heads=c.fold_v_heads, residual=c.fold_rotation)
            fold_model(model, adapter, fold)
            c_store = FoldedStore(store, adapter, fold)
            if needs_stats:
                key = (c.fold_rotation, c.fold_v_heads)
                if key not in folded_stats:
                    folded_stats[key] = folded_activation_stats(
                        model_dir, adapter, fold, mx.load(str(artifact))["calibration"])
                c_col_weights = folded_stats[key]
                c_importance = {s.name: mx.broadcast_to(c_col_weights[s.name[:-7]][None], store.info(s.name).shape)
                                for s in specs if s.name[:-7] in c_col_weights}
            fold_info = {"seed": fold.seed, "residual": fold.residual, "v_heads": fold.v_heads,
                         "tied_head": adapter.tied_embeddings and fold.residual,
                         "stats_recorded": bool(needs_stats)}

        def make_codec_for(base):
            set_codebook_scope(base, c.codebook_scope)
            if c.residual_fraction:
                sel = "importance" if c.residual_selection == "activation" else "magnitude"
                res = set_codebook_scope(ResidualCodec(base, c.residual_fraction, c.residual_dtype, sel,
                                                       structure=c.residual_structure,
                                                       column_share=c.residual_column_share), c.codebook_scope)
                fallback = set_codebook_scope(ResidualCodec(base, c.residual_fraction, c.residual_dtype, "magnitude"),
                                              c.codebook_scope)  # no stats
                imp_ = c_importance if sel == "importance" else None
                return lambda s: res if (imp_ is not None and s.name in imp_) or sel == "magnitude" else fallback
            return lambda _s: base

        cw = c_col_weights if c.activation_weighted else None
        imp = c_importance if (c.residual_fraction and c.residual_selection == "activation") else None
        codec_for = make_codec_for(make_codec(c.codec_dict()))
        if fold is not None and c.kind in ("vq", "additive"):
            # tensors reading the (already incoherent) rotated residual stream: no input rotation
            rotated_for = make_codec_for(make_codec({**c.codec_dict(), "rotate_in": False}))
            plain_for = codec_for
            codec_for = lambda s: rotated_for(s) if fold.rotated_input(s.role) else plain_for(s)  # noqa: E731
        if cfg.embedding_codec is not None:
            # fixed codec for embedding / untied LM head in every row (as embedding_codec in suc allocate)
            head_codec, body_for = make_codec(cfg.embedding_codec.codec_dict()), codec_for
            codec_for = lambda s: head_codec if s.role in _HEAD_ROLES else body_for(s)  # noqa: E731
        tune = cfg.finetune is not None and c.kind in ("vq", "additive")
        recon = reconstruct_model(model, c_store, specs, codec_for, seed=cfg.seed, col_weights=cw, importance=imp,
                                  keep_encodings=tune, rotation_groups=cfg.rotation_groups,
                                  expert_stacks=c.codebook_scope == "expert_stack",
                                  hessians=hessians if c.ldlq else None)
        ft_log = None
        if tune:
            from suc.quant.finetune import FinetuneConfig, finetune_blockwise

            f = cfg.finetune
            calib = mx.load(str(artifact))["calibration"]
            block_enc = {n: e for n, e in recon.pop("encodings").items() if ".layers." in n}
            _, log = finetune_blockwise(model, block_enc, calib,
                                        FinetuneConfig(block_steps=f.block_steps, lr=f.lr, sequences=f.sequences,
                                                       micro_batch=f.micro_batch, deterministic=f.deterministic),
                                        reference_weights=store.load)
            blocks = list(log["blocks"].values())
            ft_log = {"blocks": len(blocks), "seconds": log["seconds"], "peak_memory_bytes": log["peak_memory_bytes"],
                      "loss_before_mean": sum(b["loss_before"] for b in blocks) / max(len(blocks), 1),
                      "loss_after_mean": sum(b["loss_after"] for b in blocks) / max(len(blocks), 1),
                      "reverted": sum(1 for b in blocks if b.get("reverted")), "skipped": len(log["skipped"]),
                      "per_block": {str(i): {k: b[k] for k in ("loss_before", "loss_after")} for i, b in log["blocks"].items()}}
            del block_enc
        t0 = time.perf_counter()
        quality = evaluate_quality(model, ppl_tokens=ppl_tokens, seq_len=ev.seq_len, max_tokens=ev.max_tokens,
                                   kl_tokens=kl_tokens, ref_logits=ref_logits, chunk=ev.chunk)
        results.append({
            "codec": c.name + ("+ft" if tune else ""), "codec_config": c.codec_dict(), "finetune": ft_log,
            "embedding_codec": cfg.embedding_codec.name if cfg.embedding_codec is not None else None,
            "activation_weighted": c.activation_weighted, "fold": fold_info,
            "size": model_size(store, recon), "quality": quality,
            "per_role": _per_role(recon["tensors"]),
            "tensor_names": [t["name"] for t in recon["tensors"]],
            "tensors": recon["tensors"],
            "seconds": {"encode": recon["seconds"], "evaluate": time.perf_counter() - t0},
            "peak_memory_bytes": mx.get_peak_memory(),
        })
        r = results[-1]
        print(f"[quantize] {c.name}: bpp {r['size']['bpp']:.3f} PPL {quality['perplexity']['ppl']:.3f} "
              f"KL {quality['kl']['kl_mean']:.4f} (encode {recon['seconds']:.0f} s)", flush=True)
        _free(model, recon)

    result = {
        "metadata": metadata, "name": cfg.name, "config": config_to_dict(cfg),
        "architecture": {"family": adapter.family, "num_layers": adapter.num_layers,
                         "is_moe": adapter.is_moe, "num_experts": adapter.num_experts},
        "eval_data": {"wikitext_split": ev.wikitext_split, "ppl_tokens": int(ppl_tokens.size),
                      "kl_artifact": str(artifact), "kl_artifact_sha256": sha256_file(artifact),
                      "kl_sequences": int(kl_tokens.shape[0]), "kl_seq_len": int(kl_tokens.shape[1])},
        "reference": reference,
        "results": results,
    }
    json_path = write_result(result, cfg.output_dir, cfg.name)
    md_path = json_path.with_suffix(".md")
    md_path.write_text(quantize_report(result))
    result["report"] = str(md_path)
    json_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return json_path


def quantize_report(result: dict) -> str:
    meta, ref = result["metadata"], result["reference"]
    e = result["eval_data"]
    out = [f"# Quality track report: {result['name']}", "",
           f"Model `{meta['model']}` · weights `{meta['model_hash'][:16]}` · git `{meta.get('git_commit')}`", "",
           f"PPL: WikiText-2 {e['wikitext_split']} ({e['ppl_tokens']:,} tokens, max {result['config']['eval']['max_tokens']}); "
           f"KL: {e['kl_sequences']} held-out validation sequences × {e['kl_seq_len']} tokens vs BF16.",
           "bpp = whole-model actual bits / parameters (every overhead included); encoded bpp = over encoded tensors only.", "",
           f"BF16 reference PPL: {ref['perplexity']['ppl']:.3f}", "",
           "| Codec | bpp (model) | bpp (encoded) | Size (MB) | PPL | ΔPPL | KL mean | Top-1 | ΔCE | Encode (s) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    base = ref["perplexity"]["ppl"]
    for r in sorted(result["results"], key=lambda r: -r["size"]["bpp"]):
        s, q = r["size"], r["quality"]
        kl = q["kl"] or {}
        out.append(f"| {r['codec']} | {s['bpp']:.3f} | {s['encoded_bpp']:.3f} | {s['bytes_total'] / 1e6:.1f} "
                   f"| {q['perplexity']['ppl']:.3f} | {100 * (q['perplexity']['ppl'] / base - 1):+.1f}% "
                   f"| {kl.get('kl_mean', float('nan')):.4f} | {kl.get('top1_agreement', float('nan')):.3f} "
                   f"| {kl.get('delta_ce', float('nan')):+.4f} | {r['seconds']['encode']:.0f} |")
    return "\n".join(out) + "\n"
