"""Phase 2 runner: calibration corpus, activation statistics, proxy and measured sensitivity (spec §9, D5, D8)."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from suc.models.adapters import LinearSpec


def sample_measured(specs: Sequence[LinearSpec], n: int, seed: int) -> list[str]:
    """Deterministic stratified sample of tensor names: round-robin over roles, layers spread within each role."""
    rng = random.Random(seed)
    by_role: dict[str, list[LinearSpec]] = defaultdict(list)
    for s in specs:
        by_role[s.role].append(s)
    queues = {}
    for role, items in sorted(by_role.items()):
        items = sorted(items, key=lambda s: (s.layer or 0, s.expert or 0, s.name))
        # interleave evenly spaced positions so that the first picks of a role span the depth of the model
        order, step = [], len(items)
        while step >= 1 and len(order) < len(items):
            offset = rng.randrange(step)
            order += [i for i in range(offset, len(items), step) if i not in order]
            step //= 2
        order += [i for i in range(len(items)) if i not in order]
        queues[role] = [items[i].name for i in order]
    picked: list[str] = []
    while len(picked) < min(n, len(specs)):
        for role in sorted(queues):
            if queues[role] and len(picked) < n:
                picked.append(queues[role].pop(0))
    return picked


def aggregate_chunks(chunks: Sequence[tuple[dict, int]]) -> dict:
    """Token-weighted mean of the numeric fields of per-chunk measurement dicts; adds "tokens"."""
    sums: dict[str, float] = defaultdict(float)
    weights: dict[str, int] = defaultdict(int)
    for d, tokens in chunks:
        for key, v in d.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sums[key] += float(v) * tokens
                weights[key] += tokens
    out = {key: sums[key] / weights[key] for key in sums}
    out["tokens"] = sum(t for _, t in chunks)
    return out


# ---------------------------------------------------------------- runner

import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402

from suc.analysis.activations import Recorder  # noqa: E402
from suc.analysis.fisher import fisher_diag, fisher_proxy, fisher_summary  # noqa: E402
from suc.analysis.sensitivity import (  # noqa: E402
    measure_tensor_sensitivity, perturb_weight, proxy_output_error, sensitivity_score, spearman, weight_error,
)
from suc.cli.bench import resolved_revision, resolve_model_dir  # noqa: E402
from suc.config import CalibrationExperimentConfig, config_hash, config_to_dict, load_calibration_config  # noqa: E402
from suc.data.calibration import build_ood, build_partitions, partitions_hash, partitions_manifest  # noqa: E402
from suc.io.hashing import model_hash, sha256_file  # noqa: E402
from suc.io.safetensors_store import SafetensorsStore  # noqa: E402
from suc.models.adapters import get_adapter  # noqa: E402
from suc.models.streaming import block_prefix, load_lazy, stream_blocks  # noqa: E402
from suc.provenance import run_metadata, seed_everything, utc_timestamp, write_result  # noqa: E402

ARTIFACT_DIR = "experiments/artifacts"   # large binary outputs, git-ignored; their sha256 goes into the result JSON


def load_domain_docs(tokenizer, domains=()) -> tuple[dict[str, list[str]], dict, dict[str, str]]:
    """Render the curated sources of a mix (suc.data.sources.sources_for(domains): the v1 corpus SOURCES
    plus the extra sources of the named domains): domain -> documents, specs, file hashes."""
    from suc.data.sources import load_source, render, sources_for

    srcs = sources_for(domains)
    docs: dict[str, list[str]] = defaultdict(list)
    hashes = {}
    for name, spec in srcs.items():
        records, hashes[name] = load_source(spec)
        docs[spec.domain] += [t for t in (render(spec, r, tokenizer) for r in records) if t]
    return dict(docs), srcs, hashes


def proxy_specs(adapter, names: Sequence[str], experts_per_layer: int, seed: int) -> list[LinearSpec]:
    """Eligible tensors that get a proxy score; MoE experts sampled deterministically per layer."""
    specs = [s for s in adapter.linears(names) if s.eligible]
    keep: dict[int, set[int]] = {}
    for layer in {s.layer for s in specs if s.expert is not None}:
        experts = sorted({s.expert for s in specs if s.layer == layer and s.expert is not None})
        keep[layer] = set(random.Random(seed * 100_003 + layer).sample(experts, min(experts_per_layer, len(experts))))
    return [s for s in specs if s.expert is None or s.expert in keep[s.layer]]


def collect_activation_stats(model_dir: Path, tokens: mx.array, micro_batch: int,
                             full_second_moment: Sequence[str] = ()) -> tuple[Recorder, dict]:
    """Layer-streamed pass (D8): instrument each block before it runs, un-instrument after, release its weights."""
    model, _ = load_lazy(model_dir)
    rec = Recorder(full_second_moment=full_second_moment)
    out = stream_blocks(model, tokens, micro_batch=micro_batch, release=True,
                        on_block=lambda i, b: rec.instrument(b, block_prefix(i)),
                        after_block=lambda i, b: rec.remove(b))
    info = {"peak_memory_bytes": out["peak_memory_bytes"], "seconds": sum(out["block_seconds"]),
            "tokens": int(tokens.size), "recorded_tensors": len(rec.moments)}
    del model, out
    gc.collect()
    mx.clear_cache()
    return rec, info


def _proxy_entry(w: mx.array, w_hat: mx.array, m2diag: mx.array) -> dict:
    """Weight error and the diagonal-H prediction of the layer's relative output error."""
    w32 = w.astype(mx.float32)
    pred = proxy_output_error((w_hat.astype(mx.float32) - w32), m2diag)
    base = proxy_output_error(w32, m2diag)          # ≈ E||W x||² under the same diagonal approximation
    return {**weight_error(w, w_hat), "proxy_out_err": pred, "proxy_rel_out_err": pred / max(base, 1e-30)}


def run_calibration(config_path: str | Path, *, tokenizer=None, docs=None, ood_docs=None) -> Path:
    """Phase 2 end to end. `tokenizer`/`docs`/`ood_docs` can be injected (tests); default = model tokenizer + SOURCES."""
    cfg: CalibrationExperimentConfig = load_calibration_config(config_path)
    c, sc = cfg.calibration, cfg.sensitivity
    seed_everything(cfg.seed)
    model_dir = resolve_model_dir(cfg)
    store = SafetensorsStore.from_dir(model_dir)
    adapter = get_adapter(json.loads((model_dir / "config.json").read_text()))
    names = store.names()
    adapter.validate(names)
    stem = f"{utc_timestamp().replace('-', '').replace(':', '')}_{cfg.name}"
    art = Path(ARTIFACT_DIR)
    art.mkdir(parents=True, exist_ok=True)

    # 1. data (T2.1): disjoint calibration / validation partitions + OOD set
    if tokenizer is None:
        tokenizer = load_lazy(model_dir)[1]
    sources, source_hashes = {}, {}
    if docs is None:
        all_docs, sources, source_hashes = load_domain_docs(tokenizer, c.mix)
        ood_docs = all_docs.pop("ood", [])
        docs = all_docs
    parts = build_partitions({d: docs[d] for d in c.mix}, tokenizer, seq_len=c.seq_len, mix=c.mix, seed=cfg.seed,
                             n_sequences={"calibration": c.n_calibration, "validation": c.n_validation})
    parts["ood"] = build_ood(ood_docs, tokenizer, seq_len=c.seq_len, n_sequences=c.n_ood, seed=cfg.seed)
    manifest = partitions_manifest(parts, sources, source_hashes)
    data_path = art / f"{stem}_data.safetensors"
    mx.save_safetensors(str(data_path), {p: v["tokens"] for p, v in parts.items()})
    dataset_hash = partitions_hash(parts)
    # Captured before the heavy work so git commit/dirty describe the code that actually runs.
    metadata = run_metadata(model=cfg.model.id, model_revision=resolved_revision(model_dir, cfg.model.revision),
                            model_hash=model_hash(model_dir), dataset_hash=dataset_hash, seed=cfg.seed,
                            config_hash=config_hash(cfg))

    # 2. activation statistics (T2.2), layer-streamed
    rec, stats_info = collect_activation_stats(model_dir, parts["calibration"]["tokens"], c.micro_batch,
                                               c.full_second_moment)
    stats_path = art / f"{stem}_activations.safetensors"
    rec.save(stats_path)

    # 3. proxy sensitivity for every eligible tensor (D5: cheap signals)
    specs = proxy_specs(adapter, names, sc.experts_per_layer, cfg.seed)
    from mlx_lm.utils import load_model

    fisher, fisher_info = {}, None
    if sc.fisher:
        t0 = time.perf_counter()
        mx.reset_peak_memory()
        fmodel, _ = load_model(model_dir)
        fisher = fisher_diag(fmodel, parts["calibration"]["tokens"][: sc.fisher_sequences], [s.name for s in specs],
                             micro_batch=sc.fisher_micro_batch, group_size=sc.fisher_group_size,
                             loss=sc.fisher_loss, seed=cfg.seed)
        fisher_info = {"sequences": min(sc.fisher_sequences, int(parts["calibration"]["tokens"].shape[0])),
                       "loss": sc.fisher_loss, "micro_batch": sc.fisher_micro_batch,
                       "seconds": time.perf_counter() - t0, "peak_memory_bytes": mx.get_peak_memory()}
        del fmodel
        gc.collect()
        mx.clear_cache()
    t0 = time.perf_counter()
    tensors, w_hats = {}, {}
    measured_names = set(sample_measured(specs, sc.measured_tensors, cfg.seed))
    for i, s in enumerate(specs):
        key = s.name[: -len(".weight")]
        if key not in rec.moments:
            continue
        w = store.load(s.name)
        w_hat = perturb_weight(w, sc.kind, seed=cfg.seed + i, level=sc.gaussian_level, block_size=sc.block_size,
                               k=sc.k, rotate=sc.rotate)
        entry = {"role": s.role, "layer": s.layer, "expert": s.expert, "shape": list(w.shape)}
        entry.update(_proxy_entry(w, w_hat, rec.moments[key].second_moment_diag()))
        entry.update(rec.moments[key].summary(top_k=4))
        if s.name in fisher:
            f = fisher.pop(s.name)
            entry["fisher_proxy"] = fisher_proxy(w_hat.astype(mx.float32) - w.astype(mx.float32), f)
            entry["fisher"] = fisher_summary(f)
            del f
        tensors[s.name] = entry
        if s.name in measured_names:
            w_hats[s.name] = w_hat
        del w
        if i % 16 == 0:
            gc.collect()
            mx.clear_cache()
    proxy_seconds = time.perf_counter() - t0
    del rec
    gc.collect()
    mx.clear_cache()

    # 4. measured perturbations on the sampled subset (T2.3), on held-out validation text
    model, _ = load_model(model_dir)
    ev = parts["validation"]["tokens"][: sc.eval_sequences]
    chunks = [ev[s:s + sc.eval_chunk] for s in range(0, ev.shape[0], sc.eval_chunk)]
    refs = []
    for ch in chunks:
        r = model(ch)
        mx.eval(r)
        refs.append(r)
    t0 = time.perf_counter()
    for name, w_hat in w_hats.items():
        per_chunk = [(measure_tensor_sensitivity(model, name, w_hat, ch, ref), int(ch.shape[0] * (ch.shape[1] - 1)))
                     for ch, ref in zip(chunks, refs)]
        m = aggregate_chunks(per_chunk)
        m["sensitivity_kl_per_rel_mse"] = sensitivity_score(m["kl_mean"], tensors[name]["rel_mse"])
        tensors[name]["measured"] = m
    measure_seconds = time.perf_counter() - t0
    del model, refs
    gc.collect()
    mx.clear_cache()

    # 5. validate the proxy (D5): rank correlations over the measured subset
    measured = [t for t in tensors.values() if "measured" in t]
    kl = [t["measured"]["kl_mean"] for t in measured]
    validation = {
        "n": len(measured),
        "spearman_proxy_rel_vs_kl": spearman([t["proxy_rel_out_err"] for t in measured], kl),
        "spearman_proxy_abs_vs_kl": spearman([t["proxy_out_err"] for t in measured], kl),
        "spearman_weight_rel_mse_vs_kl": spearman([t["rel_mse"] for t in measured], kl),
        "spearman_proxy_rel_vs_layer_out": spearman([t["proxy_rel_out_err"] for t in measured],
                                                    [t["measured"]["layer_out_rel_mse"] for t in measured]),
        "spearman_proxy_rel_vs_delta_ce": spearman([t["proxy_rel_out_err"] for t in measured],
                                                   [t["measured"]["delta_ce"] for t in measured]),
    }
    if fisher_info is not None:
        validation["spearman_fisher_vs_kl"] = spearman([t["fisher_proxy"] for t in measured], kl)
        validation["spearman_fisher_vs_delta_ce"] = spearman([t["fisher_proxy"] for t in measured],
                                                             [t["measured"]["delta_ce"] for t in measured])

    result = {
        "metadata": metadata,
        "name": cfg.name,
        "config": config_to_dict(cfg),
        "architecture": {"family": adapter.family, "num_layers": adapter.num_layers,
                         "is_moe": adapter.is_moe, "num_experts": adapter.num_experts},
        "data": {**manifest, "artifact": str(data_path), "artifact_sha256": sha256_file(data_path)},
        "activation_stats": {**stats_info, "artifact": str(stats_path), "artifact_sha256": sha256_file(stats_path)},
        "timing_s": {"proxy": proxy_seconds, "measure": measure_seconds},
        "fisher": fisher_info,
        "proxy_validation": validation,
        "tensors": tensors,
    }
    json_path = write_result(result, cfg.output_dir, cfg.name)
    json_path.with_suffix(".md").write_text(calibration_report(result))
    return json_path


def calibration_report(result: dict) -> str:
    """Markdown summary: data, proxy validation, most/least sensitive tensors."""
    v, d, a = result["proxy_validation"], result["data"], result["activation_stats"]
    meta = result["metadata"]
    out = [f"# Phase 2 calibration & sensitivity report: {result['name']}", "",
           f"Model `{meta['model']}` · weights `{meta['model_hash'][:16]}` · data `{meta['dataset_hash'][:16]}` "
           f"· git `{meta.get('git_commit')}`", "",
           "## Data", "", "| Partition | Rows | Rows per domain |", "|---|---:|---|"]
    for p, info in d["partitions"].items():
        out.append(f"| {p} | {info['rows']} | " + ", ".join(f"{k} {n}" for k, n in info["rows_per_domain"].items()) + " |")
    out += ["", "32-gram overlap between partitions: " + ", ".join(f"{k} = {n}" for k, n in d["ngram_overlap"].items()),
            "", f"Activation statistics: {a['tokens']:,} tokens, {a['recorded_tensors']} tensors, "
            f"layer-streamed peak {a['peak_memory_bytes'] / 2**30:.2f} GiB, {a['seconds']:.1f} s.", "",
            "## Proxy validation (D5)", "",
            f"Measured tensors: {v['n']} (held-out validation text).", "",
            "| Rank correlation (Spearman) | ρ |", "|---|---:|"]
    for k in ("spearman_fisher_vs_kl", "spearman_fisher_vs_delta_ce", "spearman_proxy_rel_vs_kl",
              "spearman_proxy_abs_vs_kl", "spearman_weight_rel_mse_vs_kl", "spearman_proxy_rel_vs_layer_out",
              "spearman_proxy_rel_vs_delta_ce"):
        if k not in v:
            continue
        out.append(f"| {k.removeprefix('spearman_').replace('_', ' ')} | {v[k]:.3f} |")
    measured = sorted(((n, t) for n, t in result["tensors"].items() if "measured" in t),
                      key=lambda x: -x[1]["measured"]["kl_mean"])
    out += ["", "## Measured tensors (by KL)", "",
            "| Tensor | Weight rel. MSE | Proxy rel. out err | Fisher proxy | Layer out rel. MSE | KL | ΔCE | Top-1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n, t in measured:
        m = t["measured"]
        out.append(f"| {n} | {t['rel_mse']:.4f} | {t['proxy_rel_out_err']:.4f} | {t.get('fisher_proxy', float('nan')):.2e} "
                   f"| {m['layer_out_rel_mse']:.4f} "
                   f"| {m['kl_mean']:.4f} | {m['delta_ce']:+.4f} | {m['top1_agreement']:.3f} |")
    ranked = sorted(result["tensors"].items(), key=lambda x: -x[1]["proxy_rel_out_err"])
    out += ["", "## Proxy ranking (all scored tensors, top 15)", "", "| Tensor | Proxy rel. out err | Outlier channel ratio |",
            "|---|---:|---:|"]
    for n, t in ranked[:15]:
        out.append(f"| {n} | {t['proxy_rel_out_err']:.4f} | {t.get('outlier_channel_ratio', float('nan')):.1f} |")
    return "\n".join(out) + "\n"
