"""Phase 1 runner: weight-structure analysis, one tensor at a time (spec §8, D3, D4, D8)."""

from __future__ import annotations

import gc
import json
import zlib
from pathlib import Path

import mlx.core as mx

from suc.analysis.occupancy import codebook_fit_report
from suc.analysis.report import markdown_report, summarize
from suc.analysis.select import select_tensors
from suc.analysis.stats import block_stats, row_rms_normalize, scalar_stats, to_blocks
from suc.cli.bench import resolved_revision, resolve_model_dir
from suc.config import AnalysisConfig, config_hash, config_to_dict, load_analysis_config
from suc.io.hashing import model_hash
from suc.io.safetensors_store import SafetensorsStore
from suc.models.adapters import get_adapter
from suc.provenance import run_metadata, seed_everything, write_result
from suc.quant.hadamard import rotate_weight, supported_size


def gaussian_rd_bound(bpw: float) -> float:
    """Shannon rate–distortion bound for an iid Gaussian source: relative MSE >= 2^(-2R) at R bits/weight.

    A reference line, not a target: row-normalised LLM weights are close to (but not exactly) iid Gaussian,
    so the gap between a codebook's relative MSE and this bound measures how much a better quantizer could gain.
    """
    return 2.0 ** (-2.0 * bpw)


def analyze_tensor(w: mx.array, cfg: AnalysisConfig, seed: int, name: str) -> dict:
    """Scalar/block statistics and codebook fits for one 2-D weight, with and without rotation."""
    w = w.astype(mx.float32)
    out_dim, in_dim = w.shape
    tseed = (seed * 1_000_003 + zlib.crc32(name.encode())) % (2**31)
    rotation_supported = supported_size(in_dim) and supported_size(out_dim)
    variants = {False: w}
    if True in cfg.rotations and rotation_supported:
        variants[True] = rotate_weight(w, seed_in=tseed, seed_out=tseed + 1)
    res = {
        "shape": [out_dim, in_dim],
        "rotation_supported": rotation_supported,
        "scalar": scalar_stats(w),
        "scalar_rotated": scalar_stats(variants[True]) if True in variants else None,
        "blocks": {str(g): block_stats(w, g, normalize_rows=cfg.normalize_rows, max_blocks=cfg.max_blocks, seed=tseed)
                   for g in cfg.block_sizes if in_dim % g == 0},
        "codebooks": [],
    }
    for rot in (r for r in cfg.rotations if r in variants):
        src = row_rms_normalize(variants[rot])[0] if cfg.normalize_rows else variants[rot]
        for g in (g for g in cfg.kmeans_block_sizes if in_dim % g == 0):
            blocks = to_blocks(src, g)
            for k in cfg.codebook_sizes:
                rep = codebook_fit_report(blocks, k, seed=tseed, iters=cfg.kmeans_iters, max_fit_blocks=cfg.max_blocks)
                rep.update(rotation=rot, gaussian_rd_bound=gaussian_rd_bound(rep["nominal_index_bpw"]))
                res["codebooks"].append(rep)
    return res


def run_analysis(config_path: str | Path) -> tuple[Path, Path]:
    """Analyse the configured tensors; writes <ts>_<name>.json and a matching .md report."""
    cfg = load_analysis_config(config_path)
    seed_everything(cfg.seed)
    model_dir = resolve_model_dir(cfg)
    store = SafetensorsStore.from_dir(model_dir)
    adapter = get_adapter(json.loads((model_dir / "config.json").read_text()))
    names = store.names()
    adapter.validate(names)
    # Captured before any work so git commit/dirty describe the code that actually runs.
    metadata = run_metadata(
        model=cfg.model.id, model_revision=resolved_revision(model_dir, cfg.model.revision),
        model_hash=model_hash(model_dir), dataset_hash="none", seed=cfg.seed, config_hash=config_hash(cfg),
    )

    tensors = []
    for spec in select_tensors(adapter, names, cfg.analysis, seed=cfg.seed):
        entry = {"name": spec.name, "role": spec.role, "layer": spec.layer, "expert": spec.expert}
        entry.update(analyze_tensor(store.load(spec.name), cfg.analysis, cfg.seed, spec.name))
        tensors.append(entry)
        gc.collect()
        mx.clear_cache()

    result = {
        "metadata": metadata,
        "name": cfg.name,
        "config": config_to_dict(cfg),
        "architecture": {"family": adapter.family, "num_layers": adapter.num_layers,
                         "is_moe": adapter.is_moe, "num_experts": adapter.num_experts},
        "summary": summarize(tensors),
        "tensors": tensors,
    }
    json_path = write_result(result, cfg.output_dir, cfg.name)
    md_path = json_path.with_suffix(".md")
    md_path.write_text(markdown_report(result))
    return json_path, md_path
