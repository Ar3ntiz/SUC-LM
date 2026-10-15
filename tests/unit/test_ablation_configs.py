"""The Qwen3-8B ablation / Pareto configs (docs/ablations_plan.md) load, share one protocol, remove exactly one
component each, and match the row keys of benchmarks/make_tables.py."""
import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

from suc.config import CodecConfig, load_allocate_config, load_quantize_config

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "experiments" / "configs"
ABL = ("abl_base", "abl_codebook", "abl_residual", "abl_finetune", "abl_repeat")
SUC = "additive_m4_k256_g8_rot1s+res0.005_activation_cols"
REV = "b968826d9c46dd6066d109eabc6255188de91218"


def _abl(name: str):
    return load_quantize_config(CFG / f"{name}_qwen3_8b.yaml")


def _pareto():
    return load_allocate_config(CFG / "pareto_allocate_qwen3_8b.yaml")


def _tables():
    spec = importlib.util.spec_from_file_location("make_tables", ROOT / "benchmarks" / "make_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                         # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _fields(c: CodecConfig) -> dict:
    return {k: v for k, v in dataclasses.asdict(c).items() if k != "name"}


def _suc() -> CodecConfig:
    return _abl("abl_base").codecs[0]


@pytest.mark.parametrize("name", ABL)
def test_ablation_configs_share_one_protocol(name):
    cfg = _abl(name)
    assert cfg.name == f"{name}_qwen3_8b"
    assert (cfg.model.id, cfg.model.revision, cfg.model.role) == ("Qwen/Qwen3-8B", REV, "target")
    assert cfg.seed == (23456 if name == "abl_repeat" else 12345)
    assert cfg.rotation_groups and cfg.include_embeddings and cfg.roles is None
    assert cfg.embedding_codec is not None and cfg.embedding_codec.name == "affine_b4_g64"
    assert (cfg.eval.wikitext_split, cfg.eval.max_tokens, cfg.eval.kl_sequences, cfg.eval.seq_len) == ("test", None, 16, 2048)
    assert cfg.eval.calib_artifact is None and cfg.activation_stats is None
    assert (cfg.finetune is not None) == (name == "abl_finetune")


def test_suc_row_is_the_best_configuration():
    c = _suc()
    assert c.name == SUC
    assert (c.kind, c.k, c.block_size, c.num_codebooks, c.rotate, c.rotate_out) == ("additive", 256, 8, 4, True, False)
    assert (c.residual_fraction, c.residual_selection, c.residual_structure) == (0.005, "activation", "columns")


# each ablation row differs from SUC only in the fields of the component it removes
REMOVED = {
    "additive_m4_k256_g8_rot+res0.005_activation_cols": {"rotate_out"},
    "additive_m4_k256_g8+res0.005_activation_cols": {"rotate"},
    "vq_k16_g1_rot1s+res0.005_activation_cols": {"kind", "k", "block_size", "num_codebooks"},
    "vq_k256_g2_rot1s+res0.005_activation_cols": {"kind", "block_size", "num_codebooks"},
    "additive_m4_k256_g8_rot1s": {"residual_fraction", "residual_structure"},
    "additive_m4_k256_g8_rot1s+res0.005_magnitude_cols": {"residual_selection"},
    "additive_m4_k256_g8_rot1s+res0.005_activation": {"residual_structure"},
}


def test_each_ablation_row_removes_one_component():
    suc = _fields(_suc())
    rows = [c for n in ("abl_base", "abl_codebook", "abl_residual") for c in _abl(n).codecs if c.name != SUC]
    assert sorted(c.name for c in rows) == sorted(REMOVED)
    for c in rows:
        diff = {k for k, v in _fields(c).items() if suc[k] != v}
        assert diff == REMOVED[c.name], c.name


def test_codebook_rows_keep_the_index_rate_and_d4_ceiling():
    for c in _abl("abl_codebook").codecs:
        bits = c.num_codebooks * (c.k - 1).bit_length() / c.block_size
        assert bits == 4.0, c.name                                     # same index bpw as M=4 x 8 bits / G=8
        assert c.k * c.block_size * 2 <= 16 * 1024                     # D4: codebook bytes (fp16) <= 16 KB


def test_repeat_rows_are_abl_base_rows():
    base = {c.name: c for c in _abl("abl_base").codecs}
    rep = _abl("abl_repeat").codecs
    assert rep[0].name == SUC and all(base[c.name] == c for c in rep)


def test_finetune_pair_uses_the_untuned_csr_row():
    tuned = _abl("abl_finetune").codecs
    untuned = {c.name: c for c in _abl("abl_residual").codecs}
    assert len(tuned) == 1 and tuned[0] == untuned[tuned[0].name]
    assert tuned[0].residual_structure == "csr"          # column-residual tuning is not implemented


def test_pareto_config_covers_the_spec_points():
    cfg = _pareto()
    t = _tables()
    budgets = set(cfg.budgets_bpp)
    for target in t.PARETO_TARGETS:
        assert t.PARETO_BUDGETS.get(target, target) in budgets
    assert set(t.EXTRA_BUDGETS) <= budgets
    names = [t.greedy_name(b) for b in cfg.budgets_bpp]
    assert len(set(names)) == len(names)                 # plan names (greedy_{b:.1f}) do not collide
    assert cfg.head_candidates is not None and cfg.uniform_head == "affine_b4_g64"
    assert (cfg.model.id, cfg.model.revision, cfg.seed, cfg.eval.kl_sequences) == ("Qwen/Qwen3-8B", REV, 12345, 16)
    for c in cfg.candidates:
        if c.kind in ("vq", "additive"):
            assert c.rotate and not c.rotate_out, c.name
            assert c.residual_fraction in (0.0, 0.005) and (not c.residual_fraction or c.residual_structure == "columns")
    assert SUC in {c.name for c in cfg.candidates} and f"uniform_{SUC}".removeprefix("uniform_") in cfg.uniform
    # cheapest assignment (VQ G=8 everywhere, ≈ 1.006 bpp) stays below the lowest budget
    assert min(cfg.budgets_bpp) >= 1.03


def test_make_tables_rows_point_at_existing_codecs_and_plans():
    t = _tables()
    codecs = {f"*_{n}_qwen3_8b.json": {c.name for c in _abl(n).codecs} for n in ABL}
    codecs["*_abl_finetune_qwen3_8b.json"] = {c.name + "+ft" for c in _abl("abl_finetune").codecs}
    p = _pareto()
    plans = {"uniform_" + u for u in p.uniform} | {t.greedy_name(b) for b in p.budgets_bpp}
    for row in t.ABLATIONS:
        if row.src.pattern in codecs:
            assert row.src.key in codecs[row.src.pattern], row.label
        else:
            assert row.src.pattern == t.PARETO and row.src.key in plans, row.label
    labels = {r.label for r in t.ABLATIONS}
    assert all(r.vs is None or r.vs in labels for r in t.ABLATIONS)
    assert all(r.vs is None or r.vs in {x.label for x in t.EVIDENCE} for r in t.EVIDENCE)
    assert all(r.vs is None or r.vs in {x.label for x in t.SPEED} for r in t.SPEED)


def test_make_tables_renders_with_missing_results(tmp_path):
    md, series = _tables().render(tmp_path)              # empty results dir: every row pending, no crash
    assert "pending (`abl_base_qwen3_8b`)" in md and "not implemented" in md
    assert all(not v for v in series.values())


def test_make_tables_deltas_against_the_suc_row(tmp_path):
    import json

    def row(codec, kl):
        return {"codec": codec, "size": {"bpp": 4.2, "bits_total": 8e9},
                "quality": {"perplexity": {"ppl": 10.0}, "kl": {"kl_mean": kl, "top1_agreement": 0.9}}}
    (tmp_path / "20990101T000000Z_abl_base_qwen3_8b.json").write_text(json.dumps(
        {"reference": {"perplexity": {"ppl": 9.7}},
         "results": [row(SUC, 0.08), row("additive_m4_k256_g8_rot+res0.005_activation_cols", 0.088)]}))
    md, _ = _tables().render(tmp_path)
    two_sided = next(line for line in md.splitlines() if "rotation two-sided" in line)
    assert "| 0.0880 | +10.0% |" in two_sided and "20990101T000000Z_abl_base_qwen3_8b.json" in two_sided
