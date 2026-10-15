"""Allocation runner (Phase 7 / Q7, spec §39 steps 1–8), layer-streamed end to end (spec D8).

Fisher on calibration data → every candidate codec scored on every allocation unit (predicted ΔL = ½ Σ F ΔW²)
→ greedy plans per bit budget (+ uniform baselines) → each plan reconstructed and validated on the full model:
WikiText-2 perplexity (final-test set) and KL vs BF16 on held-out validation sequences.

Memory: no stage holds the whole model. The empirical Fisher is the exact streamed one
(suc.analysis.fisher.fisher_diag_streamed): the candidates of a block are scored as soon as its F is final and F is
then dropped, so at most one block's F exists (block tensors are bit-identical to the whole-model fisher_diag; the
embedding's F only up to its scatter-add summation noise). Validation is block-sequential
(suc.evaluation.streamed): the BF16 reference logits are streamed to disk, and for every plan each block is
reconstructed with the plan's codecs (one tensor / expert stack at a time from the SafetensorsStore) right before
it runs, so peak memory ≈ one block + hidden states (spilled to disk when large) + one micro-batch of logits.
PPL / KL / sizes / rows equal those of the earlier whole-model loop bit for bit (tests/unit/test_allocate_streamed.py).

Scratch state (Fisher sweep, scored parts, reference logits, spilled hidden states) lives in
<ARTIFACT_DIR>/<name>_work keyed by a fingerprint of the inputs: a crashed run resumes the Fisher / scoring at
block granularity; it is removed after success. The candidates artifact (<ts>_<name>_candidates.json) is written
atomically before validation; `candidates_artifact` in the config resumes from it (Fisher and scoring skipped; its
units and codec names are checked against the config).

With `head_candidates` the embedding (and the LM head, if untied) are allocated in the same budget with their own
candidate list; otherwise they are encoded with the fixed `embedding_codec` (or kept BF16) outside the budget.
"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path

import mlx.core as mx
import numpy as np

from suc.allocation.greedy import Candidate, build_candidates, greedy_allocate, plan_summary, uniform_plan
from suc.analysis.fisher import fisher_diag_streamed, fisher_proxy
from suc.analysis.sensitivity import spearman
from suc.cli.bench import resolved_revision, resolve_model_dir
from suc.config import AllocateExperimentConfig, config_hash, config_to_dict, load_allocate_config
from suc.evaluation.data import load_wikitext2, tokenize
from suc.evaluation.reconstruct import model_size, reconstruct_model
from suc.evaluation.streamed import RefLogits, streamed_quality
from suc.io.hashing import model_hash, sha256_file
from suc.io.safetensors_store import SafetensorsStore
from suc.models.adapters import get_adapter
from suc.models.streaming import load_lazy
from suc.pipeline.stages import atomic_path, atomic_write_text
from suc.provenance import run_metadata, seed_everything, utc_timestamp, write_result
from suc.quant.codecs import make_codec
from suc.analysis.activations import Recorder
from suc.quant import stacks
from suc.quant.residual import ResidualCodec
from suc.quant.run import default_activation_stats, default_calib_artifact, select_specs

ARTIFACT_DIR = "experiments/artifacts"
HIDDEN_IN_MEMORY_MAX = 4 << 30      # hidden states of a streamed sweep above this size are spilled to disk


def _log(msg: str) -> None:
    print(f"[allocate] {msg}", flush=True)


def _free(*objs) -> None:
    del objs
    gc.collect()
    mx.clear_cache()


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def head_specs(adapter, names) -> list:
    """Embedding and (only if untied) LM head specs, in checkpoint order."""
    return [s for s in select_specs(adapter, names, None, include_embeddings=True) if s.role in ("embed", "lm_head")]


def _make_codec(c):
    base = make_codec(c.codec_dict())
    sel = "importance" if c.residual_selection == "activation" else "magnitude"
    codec = ResidualCodec(base, c.residual_fraction, c.residual_dtype, sel, structure=c.residual_structure,
                          column_share=c.residual_column_share) if c.residual_fraction else base
    return stacks.set_codebook_scope(codec, c.codebook_scope)


def _importance(rec: Recorder | None, store: SafetensorsStore, specs) -> dict[str, mx.array] | None:
    """Per-weight importance E[x_j^2] (broadcast over rows) of the given block linears, from Phase 2 statistics."""
    if rec is None:
        return None
    return {s.name: mx.broadcast_to(rec.moments[s.name[:-7]].second_moment_diag()[None], store.info(s.name).shape)
            for s in specs}


# ------------------------------------------------------------------------------------- candidates artifact
_CAND_FIELDS = {f.name for f in fields(Candidate)}


def _dump_candidates(cands: dict[str, list[Candidate]]) -> str:
    return json.dumps({n: [asdict(c) for c in cs] for n, cs in cands.items()}, sort_keys=True)


def load_candidates(path: str | Path, expected: dict[str, list[str]]) -> dict[str, list[Candidate]]:
    """Candidates of a *_candidates.json artifact, checked against the config: exactly the expected allocation units
    (expected = {unit: [candidate codec names in config order]}), each with exactly those codecs in that order.
    Returned in the order of `expected` (the order the scoring run built them in: greedy plans and their rows
    follow it). ValueError on any mismatch."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"candidates artifact {path} not found")
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: not a candidates artifact (expected a JSON object)")
    missing = [u for u in expected if u not in raw]
    extra = sorted(set(raw) - set(expected))
    if missing or extra:
        raise ValueError(f"{path}: allocation units do not match the config: {len(missing)} missing "
                         f"{missing[:3]}, {len(extra)} extra {extra[:3]}")
    out: dict[str, list[Candidate]] = {}
    for u, want in expected.items():
        got = [c.get("codec") if isinstance(c, dict) else None for c in raw[u]]
        if got != want:
            raise ValueError(f"{path}: unit {u!r} has candidate codecs {got}, the config expects {want}")
        for c in raw[u]:
            if set(c) - _CAND_FIELDS or not {"codec", "bits", "loss"} <= set(c):
                raise ValueError(f"{path}: unit {u!r}: candidate field(s) {sorted(set(c) ^ _CAND_FIELDS)} "
                                 f"do not match {sorted(_CAND_FIELDS)}")
        out[u] = [Candidate(**c) for c in raw[u]]
    return out


# ------------------------------------------------------------------------------------- Fisher + scoring
def _fingerprint(cfg: AllocateExperimentConfig, model_id_hash: str, calib: mx.array, stats: Path | None) -> str:
    """Inputs of the Fisher sweep and scoring (the scratch state of a crashed run is reused only if they match)."""
    d = config_to_dict(cfg)
    keep = ("model", "candidates", "head_candidates", "fisher_loss", "fisher_sequences", "fisher_micro_batch",
            "allocation_unit", "seed")
    blob = {"config": {k: d[k] for k in keep}, "model_hash": model_id_hash,
            "calib_sha256": hashlib.sha256(np.asarray(calib, dtype=np.int32).astype("<i4").tobytes()).hexdigest(),
            "activation_stats": sha256_file(stats) if stats is not None else None}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()


def score_candidates(*, cfg: AllocateExperimentConfig, model_dir: Path, store: SafetensorsStore, specs, head_names,
                     unit_map: dict[str, list[str]], codecs: dict, head_codecs: dict, rec: Recorder | None,
                     calib: mx.array, work: Path, fingerprint: str) -> tuple[dict[str, list[Candidate]], dict]:
    """Streamed exact Fisher with every block's units scored as soon as its F is final (then F is dropped).

    Each scored group is written to work/parts/<group>.json, so after a crash the Fisher sweep (work/fisher_work)
    and the scoring resume at block granularity. Returns the candidates in the order unit_map + head_names (that of
    the earlier single build_candidates call) and timings."""
    marker = work / "fingerprint.json"
    if work.exists() and (not marker.exists() or json.loads(marker.read_text()).get("fingerprint") != fingerprint):
        shutil.rmtree(work)
    parts = work / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    atomic_write_text(marker, json.dumps({"fingerprint": fingerprint}))
    layer_of = {s.name: s.layer for s in specs}
    units_of: dict[int, list[str]] = defaultdict(list)
    for u, members in unit_map.items():
        units_of[layer_of[members[0]]].append(u)
    specs_of: dict[int, list] = defaultdict(list)
    for s in specs:
        specs_of[s.layer].append(s)
    groups = [f"block_{i:05d}" for i in sorted(units_of)] + (["head"] if head_names else [])
    scoring = [0.0]

    def score(group: str, fisher: dict[str, mx.array]) -> None:
        out = parts / f"{group}.json"
        if out.exists():                       # delivered again after a crash: already scored
            return
        t0 = time.perf_counter()
        loss_fn = lambda n, w, w_hat: fisher_proxy(w_hat.astype(mx.float32) - w.astype(mx.float32), fisher[n])  # noqa: E731
        if group == "head":
            cands = build_candidates(store.load, head_names, head_codecs, loss_fn, seed=cfg.seed)
        else:
            i = int(group.split("_")[1])
            us = units_of[i]
            cands = build_candidates(store.load, us, codecs, loss_fn, seed=cfg.seed,
                                     importance=_importance(rec, store, specs_of[i]),
                                     stacks={u: unit_map[u] for u in us},
                                     log_every=16 if cfg.allocation_unit == "expert_stack" else 0)
        atomic_write_text(out, _dump_candidates(cands))
        scoring[0] += time.perf_counter() - t0
        _log(f"candidates {group}: {len(cands)} units scored ({time.perf_counter() - t0:.1f} s)")

    t0 = time.perf_counter()
    finfo = {"forward_seconds": 0.0, "reverse_seconds": 0.0, "peak_memory_bytes": None}
    if not all((parts / f"{g}.json").exists() for g in groups):
        model = load_lazy(model_dir)[0]
        finfo = fisher_diag_streamed(model, calib, [s.name for s in specs] + list(head_names), model_dir=model_dir,
                                     on_group=score, micro_batch=cfg.fisher_micro_batch, loss=cfg.fisher_loss,
                                     seed=cfg.seed, work_dir=work / "fisher_work", fingerprint=fingerprint)
        _free(model)
    total = time.perf_counter() - t0
    raw: dict[str, list[dict]] = {}
    for g in groups:
        raw.update(json.loads((parts / f"{g}.json").read_text()))
    order = list(unit_map) + list(head_names)
    if set(raw) != set(order):
        raise RuntimeError(f"candidates: scored units differ from the allocation units "
                           f"(missing {sorted(set(order) - set(raw))[:3]}, extra {sorted(set(raw) - set(order))[:3]})")
    cands = {u: [Candidate(**c) for c in raw[u]] for u in order}
    timing = {"fisher": total - scoring[0], "candidates": scoring[0],
              "fisher_forward": finfo["forward_seconds"], "fisher_reverse_incl_scoring": finfo["reverse_seconds"],
              "fisher_peak_memory_bytes": finfo["peak_memory_bytes"]}
    return cands, timing


# ------------------------------------------------------------------------------------- streamed validation
def _spill_dir(store: SafetensorsStore, hf_config: dict, ev, ppl_tokens: mx.array, kl_tokens: mx.array,
               work: Path) -> Path | None:
    """Spill directory for the hidden states of a validation sweep, or None when they fit HIDDEN_IN_MEMORY_MAX."""
    n = ppl_tokens.size if ev.max_tokens is None else min(ppl_tokens.size, ev.max_tokens)
    n_tok = (n // (ev.seq_len + 1)) * ev.seq_len + kl_tokens.size
    e = "model.embed_tokens.weight"
    itemsize = store.info(e).nbytes // max(1, _numel(store.info(e).shape)) if e in store else 2
    return work / "hidden" if n_tok * int(hf_config.get("hidden_size", 0)) * itemsize > HIDDEN_IN_MEMORY_MAX else None


def streamed_plan_quality(*, model_dir: Path, store: SafetensorsStore, specs, head_part, codec_for, seed: int,
                          rec: Recorder | None, expert_stacks: bool, ppl_tokens: mx.array, kl_tokens: mx.array,
                          ref_logits: RefLogits, ev, spill_dir: Path | None,
                          log_every: int = 0) -> tuple[dict, dict, dict]:
    """(quality, recon, streaming) of one plan: the lazily loaded model is run block by block and each block is
    reconstructed with codec_for (reconstruct_model, one tensor / expert stack at a time) right before it runs; the
    embedding / LM head (head_part) before the embedding lookup. recon is the merged reconstruct_model record."""
    by_layer: dict[int, list] = defaultdict(list)
    for s in specs:
        by_layer[s.layer].append(s)
    recons: list[dict] = []
    model = load_lazy(model_dir)[0]

    def rebuild(ss, importance) -> None:
        if ss:
            recons.append(reconstruct_model(model, store, ss, codec_for, seed=seed, importance=importance,
                                            expert_stacks=expert_stacks))
            if log_every and ss[0].layer is not None and (ss[0].layer + 1) % log_every == 0:
                _log(f"  block {ss[0].layer + 1}/{len(by_layer)} reconstructed "
                     f"({sum(x['seconds'] for x in recons):.1f} s encode so far)")

    r = streamed_quality(model, ppl_tokens=ppl_tokens, seq_len=ev.seq_len, max_tokens=ev.max_tokens,
                         kl_tokens=kl_tokens, chunk=ev.chunk, ref_logits=ref_logits,
                         prepare_head=lambda m: rebuild(head_part, None),
                         prepare_block=lambda i, b: rebuild(by_layer.get(i, []),
                                                            _importance(rec, store, by_layer.get(i, []))),
                         spill_dir=spill_dir)
    _free(model)
    recon = {"tensors": [t for x in recons for t in x["tensors"]],
             "encoded_params": sum(x["encoded_params"] for x in recons),
             "encoded_bits": sum(x["encoded_bits"] for x in recons),
             "seconds": sum(x["seconds"] for x in recons)}
    return {"perplexity": r["perplexity"], "kl": r["kl"]}, recon, r["streaming"]


def run_allocate(config_path: str | Path) -> Path:
    cfg: AllocateExperimentConfig = load_allocate_config(config_path)
    ev = cfg.eval
    seed_everything(cfg.seed)
    resume = cfg.candidates_artifact is not None
    if resume and not Path(cfg.candidates_artifact).is_file():     # fail before any heavy work
        raise FileNotFoundError(f"candidates artifact {cfg.candidates_artifact} not found")
    model_dir = resolve_model_dir(cfg)
    store = SafetensorsStore.from_dir(model_dir)
    hf_config = json.loads((model_dir / "config.json").read_text())
    adapter = get_adapter(hf_config)
    names = store.names()
    adapter.validate(names)
    specs = select_specs(adapter, names, None, include_embeddings=False)
    stack_units = cfg.allocation_unit == "expert_stack"
    # head_candidates: embed (+ untied lm_head) are allocated like block linears (same Fisher, same budget);
    # otherwise they are encoded with the fixed embedding_codec (or kept BF16) outside the budget.
    hspecs = head_specs(adapter, names) if cfg.head_candidates is not None else []
    emb_specs = head_specs(adapter, names) if cfg.embedding_codec is not None else []
    head_names = [s.name for s in hspecs]
    alloc_params = sum(_numel(store.info(s.name).shape) for s in specs + hspecs)
    head_params = sum(_numel(store.info(s.name).shape) for s in hspecs)
    # allocation units: one per tensor, or one per MoE expert stack (all its experts get the same codec)
    unit_map = stacks.units([s.name for s in specs]) if stack_units else {s.name: [s.name] for s in specs}
    codecs = {c.name: _make_codec(c) for c in cfg.candidates}
    head_codecs = {c.name: _make_codec(c) for c in cfg.head_candidates or ()}
    expected = {**{u: list(codecs) for u in unit_map}, **{n: list(head_codecs) for n in head_names}}

    text, wiki_hash = load_wikitext2(ev.wikitext_split)
    ppl_tokens = tokenize(load_lazy(model_dir)[1], text)
    artifact = Path(ev.calib_artifact) if ev.calib_artifact else default_calib_artifact(cfg.model.id)
    corpus = mx.load(str(artifact))
    calib, kl_tokens = corpus["calibration"][: cfg.fisher_sequences], corpus["validation"][: ev.kl_sequences]
    del corpus
    stats_path = None
    if any(c.residual_fraction and c.residual_selection == "activation" for c in cfg.candidates):
        stats_path = Path(default_activation_stats(cfg.model.id))
    rec = Recorder.load(stats_path) if stats_path is not None else None
    mhash = model_hash(model_dir)
    # Captured before the heavy work so git commit/dirty describe the code that actually runs.
    metadata = run_metadata(model=cfg.model.id, model_revision=resolved_revision(model_dir, cfg.model.revision),
                            model_hash=mhash, dataset_hash=wiki_hash, seed=cfg.seed, config_hash=config_hash(cfg))
    art = Path(ARTIFACT_DIR)
    art.mkdir(parents=True, exist_ok=True)
    work = art / f"{cfg.name}_work"

    # 1–2. Fisher and candidate scoring (layer-streamed), or resume from a saved candidates artifact
    if resume:
        cand_path = Path(cfg.candidates_artifact)
        _log(f"resuming from candidates artifact {cand_path} (Fisher and scoring skipped)")
        cands = load_candidates(cand_path, expected)
        timing: dict = {"fisher": None, "candidates": None}
    else:
        cand_path = art / f"{utc_timestamp().replace('-', '').replace(':', '')}_{cfg.name}_candidates.json"
        _log(f"candidates artifact will be written to {cand_path}")
        cands, timing = score_candidates(cfg=cfg, model_dir=model_dir, store=store, specs=specs,
                                         head_names=head_names, unit_map=unit_map, codecs=codecs,
                                         head_codecs=head_codecs, rec=rec, calib=calib, work=work,
                                         fingerprint=_fingerprint(cfg, mhash, calib, stats_path))
        atomic_write_text(cand_path, _dump_candidates(cands))
        _log(f"candidates artifact written: {cand_path}")
        shutil.rmtree(work / "fisher_work", ignore_errors=True)
    del calib
    gc.collect()
    mx.clear_cache()

    # 3–6. plans
    plans: dict[str, dict[str, Candidate]] = {}
    for b in cfg.budgets_bpp:
        plans[f"greedy_{b:.1f}"] = greedy_allocate(cands, int(b * alloc_params))
    block_cands = {n: cs for n, cs in cands.items() if n not in set(head_names)}
    for u in cfg.uniform:
        plans[f"uniform_{u}"] = uniform_plan(block_cands, u)
        if hspecs:
            plans[f"uniform_{u}"].update(uniform_plan({n: cands[n] for n in head_names}, cfg.uniform_head))

    # 7. full-model validation, block-sequential (reference logits streamed to disk, read back chunk by chunk)
    spill = _spill_dir(store, hf_config, ev, ppl_tokens, kl_tokens, work)
    ref_path = work / "ref_logits.safetensors"
    model = load_lazy(model_dir)[0]
    with atomic_path(ref_path) as tmp:
        r = streamed_quality(model, ppl_tokens=ppl_tokens, seq_len=ev.seq_len, max_tokens=ev.max_tokens,
                             kl_tokens=kl_tokens, chunk=ev.chunk, ref_out=tmp, spill_dir=spill)
    _free(model)
    reference = {"perplexity": r["perplexity"], "kl": None}
    streaming = {"reference": r["streaming"], "hidden_spilled": spill is not None}
    _log(f"reference: BF16 PPL {reference['perplexity']['ppl']:.3f}, peak "
         f"{r['streaming']['peak_memory_bytes'] / 2 ** 30:.2f} GiB")
    ref_logits = RefLogits(ref_path)
    emb_codec = make_codec(cfg.embedding_codec.codec_dict()) if cfg.embedding_codec is not None else None
    out = []
    for pname, plan in plans.items():
        chosen = stacks.expand_plan({n: c.codec for n, c in plan.items()}, unit_map)
        codec_for = lambda s: (head_codecs if s.name in head_names else codecs)[chosen[s.name]]             if s.name in chosen else emb_codec  # noqa: E731
        quality, recon, sinfo = streamed_plan_quality(
            model_dir=model_dir, store=store, specs=specs, head_part=hspecs or emb_specs, codec_for=codec_for,
            seed=cfg.seed, rec=rec, expert_stacks=stack_units, ppl_tokens=ppl_tokens, kl_tokens=kl_tokens,
            ref_logits=ref_logits, ev=ev, spill_dir=spill, log_every=8 if stack_units else 0)
        summ = plan_summary(plan)
        head_bits = sum(plan[n].bits for n in head_names)
        out.append({"plan": pname, "allocated_bpw": summ["bits_total"] / alloc_params,
                    "blocks_bpw": (summ["bits_total"] - head_bits) / (alloc_params - head_params),
                    "head": {n: plan[n].codec for n in head_names}, "head_bits": head_bits,
                    "predicted_loss": summ["loss_total"], "by_codec": summ["by_codec"],
                    "size": model_size(store, recon), "quality": quality, "rows": summ["rows"],
                    "seconds": recon["seconds"], "streaming": sinfo})
        _log(f"{pname}: alloc bpw {out[-1]['allocated_bpw']:.3f} PPL {quality['perplexity']['ppl']:.3f} "
             f"KL {quality['kl']['kl_mean']:.4f} predicted {summ['loss_total']:.3e} "
             f"peak {sinfo['peak_memory_bytes'] / 2 ** 30:.2f} GiB")
        del recon
        gc.collect()
    del ref_logits

    result = {
        "metadata": metadata, "name": cfg.name, "config": config_to_dict(cfg),
        "allocated_tensors": len(specs) + len(hspecs), "allocated_params": alloc_params,
        "allocation_unit": cfg.allocation_unit, "allocation_units": len(unit_map) + len(hspecs),
        "head_tensors": head_names, "head_params": head_params,
        "candidates_artifact": str(cand_path), "candidates_sha256": sha256_file(cand_path),
        "resumed_from_candidates": resume,
        "timing_s": timing,
        "eval_data": {"wikitext_split": ev.wikitext_split, "ppl_tokens": int(ppl_tokens.size),
                      "kl_artifact": str(artifact), "kl_sequences": int(kl_tokens.shape[0])},
        "reference": reference,
        "streaming": streaming,
        "plans": out,
        "predicted_vs_measured_spearman": spearman([p["predicted_loss"] for p in out],
                                                   [p["quality"]["kl"]["kl_mean"] for p in out]),
    }
    json_path = write_result(result, cfg.output_dir, cfg.name)
    md_path = json_path.with_suffix(".md")
    md_path.write_text(allocate_report(result))
    result["report"] = str(md_path)
    json_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    return json_path


def allocate_report(result: dict) -> str:
    meta, ref = result["metadata"], result["reference"]
    base = ref["perplexity"]["ppl"]
    heads = result.get("head_tensors") or []
    out = [f"# Allocation report: {result['name']}", "",
           f"Model `{meta['model']}` · weights `{meta['model_hash'][:16]}` · git `{meta.get('git_commit')}`", "",
           f"{result['allocated_tensors']} allocated tensors ({result['allocated_params']:,} params"
           + (f"; {result['allocation_units']} allocation units, one per MoE expert stack"
              if result.get("allocation_unit") == "expert_stack" else "") + "). Predicted loss = "
           "Σ ½ F ΔW² (empirical Fisher, additive). BF16 reference PPL "
           f"{base:.3f}. Spearman(predicted, measured KL) across plans: {result['predicted_vs_measured_spearman']:.3f}."
           + (f" Embedding / LM head allocated in the same budget: {', '.join(f'`{n}`' for n in heads)} "
              f"({result['head_params']:,} params; Fisher is zero on rows of tokens absent from the calibration inputs)."
              if heads else ""), "",
           "| Plan | Alloc. bpw | Model bpp | PPL | ΔPPL | KL mean | Top-1 | Predicted loss | Codecs |"
           + (" Blocks bpw | Embed / LM head |" if heads else ""),
           "|---|---:|---:|---:|---:|---:|---:|---:|---|" + ("---:|---|" if heads else "")]
    for p in sorted(result["plans"], key=lambda p: p["allocated_bpw"]):
        q, kl = p["quality"], p["quality"]["kl"]
        codecs = ", ".join(f"{k} {v}" for k, v in sorted(p["by_codec"].items(), key=lambda kv: -kv[1]))
        row = (f"| {p['plan']} | {p['allocated_bpw']:.3f} | {p['size']['bpp']:.3f} | {q['perplexity']['ppl']:.3f} "
               f"| {100 * (q['perplexity']['ppl'] / base - 1):+.1f}% | {kl['kl_mean']:.4f} | {kl['top1_agreement']:.3f} "
               f"| {p['predicted_loss']:.3e} | {codecs} |")
        if heads:
            row += f" {p['blocks_bpw']:.3f} | {' / '.join(p['head'][n] for n in heads)} |"
        out.append(row)
    return "\n".join(out) + "\n"
