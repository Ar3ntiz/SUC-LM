"""Calibration mix v2 (GSM8K + MATH-train math share): opt-in semantics, unchanged v1 fingerprints / corpus identity,
the exclusion filter and the committed disjointness record (no network: downloads are faked or not needed)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from suc.config import CALIB_MIX_V1, CALIB_MIX_V2, CALIB_MIX_V3, CalibrationConfig
from suc.data import sources as src
from suc.pipeline.compress import config_delta

REPO = Path(__file__).resolve().parents[2]


def test_mix_presets_and_default():
    assert CalibrationConfig().mix == CALIB_MIX_V3
    assert CalibrationConfig(mix="v3").mix == CALIB_MIX_V3
    assert abs(sum(CALIB_MIX_V3.values()) - 1) < 1e-9
    assert CALIB_MIX_V3["math"] + CALIB_MIX_V3["math_competition"] == pytest.approx(CALIB_MIX_V1["math"])
    assert CALIB_MIX_V3["math_competition"] == 0.03     # small share: the v2 run lost MMLU-Pro (report §5c)
    assert CalibrationConfig(mix="v1").mix == CALIB_MIX_V1 and CalibrationConfig(mix="v2").mix == CALIB_MIX_V2
    assert abs(sum(CALIB_MIX_V2.values()) - 1) < 1e-9
    assert CALIB_MIX_V2["math"] + CALIB_MIX_V2["math_competition"] == CALIB_MIX_V1["math"]
    with pytest.raises(ValueError, match="preset"):
        CalibrationConfig(mix="v9")


def test_v1_fingerprint_inputs_unchanged_v2_differs():
    # a v1 config (explicit dict, as every committed config, or the preset) contributes nothing to the stage
    # fingerprints -- exactly as before the default changed; v2 (explicit or by default) does
    assert config_delta(CalibrationConfig(mix=dict(CALIB_MIX_V1))) == {}
    assert config_delta(CalibrationConfig(mix="v1")) == {}
    assert config_delta(CalibrationConfig()) == {"mix": CALIB_MIX_V3}
    assert config_delta(CalibrationConfig(mix="v2", n_calibration=64)) == {"n_calibration": 64, "mix": CALIB_MIX_V2}


def test_committed_configs_all_pin_v1():
    import yaml

    for f in sorted((REPO / "experiments" / "configs").glob("compress_*.yaml")):
        d = yaml.safe_load(f.read_text())
        want = "v2" if f.name == "compress_deepseek_v4_flash_tcq_mathcal.yaml" else CALIB_MIX_V1  # the v2 experiment
        assert (d.get("calibration") or {}).get("mix") == want, f.name


def test_v1_corpus_identity_is_the_old_value():
    old = hashlib.sha256(json.dumps({k: repr(v) for k, v in sorted(src.SOURCES.items())}).encode()).hexdigest()
    assert src.sources_for(CALIB_MIX_V1) == src.SOURCES
    assert src.corpus_identity(CALIB_MIX_V1) == old
    v2 = src.sources_for(CALIB_MIX_V2)
    assert set(v2) - set(src.SOURCES) == {f"math_train_{s}" for s in src.MATH_SUBJECTS}
    assert src.corpus_identity(CALIB_MIX_V2) != old
    assert src.sources_for(CALIB_MIX_V3) == v2 and src.corpus_identity(CALIB_MIX_V3) == src.corpus_identity(CALIB_MIX_V2)
    for s in v2.values():
        if s.domain == "math_competition":
            assert len(s.revision) == 40 and s.license == "mit" and s.template == "chat"
            assert "/train-" in s.filename and isinstance(s, src.ExcludingSourceSpec)


def test_excluding_source_drops_listed_problems(tmp_path, monkeypatch):
    path = tmp_path / "train.parquet"
    pq.write_table(pa.table({"problem": ["Find  X.", "Keep me", "eval item"], "solution": ["s1", "s2", "s3"]}), path)
    monkeypatch.setattr(src, "hf_hub_download", lambda **kw: str(path))
    (tmp_path / "x.json").write_text(json.dumps({"hashes": [src.problem_key("find x."), src.problem_key("EVAL  item")]}))
    monkeypatch.setattr(src, "EXCLUSIONS_DIR", tmp_path)
    spec = src.ExcludingSourceSpec("m", "math_competition", "r", "f", "rev", "parquet", ("problem", "solution"),
                                   template="chat", exclude_field="problem", exclude_file="x.json")
    recs, _ = src.load_source(spec)
    assert recs == [{"problem": "Keep me", "solution": "s2"}]


def test_exclusion_file_covers_every_committed_eval_question():
    have = src.load_exclusions(src.MATH_EXCLUSIONS)
    n = 0
    for f in sorted((REPO / "experiments" / "eval").glob("*.json")):
        d = json.loads(f.read_text())
        for c in (d.get("cases") or []) if isinstance(d, dict) else []:
            if (c.get("question") or "").strip():
                assert src.problem_key(c["question"]) in have, (f.name, c.get("id"))
                n += 1
    assert n > 0
    rec = json.loads((REPO / "experiments" / "eval" / "calibration_disjointness.json").read_text())
    xf = src.EXCLUSIONS_DIR / src.MATH_EXCLUSIONS
    assert rec["exclusion_file"]["sha256"] == hashlib.sha256(xf.read_bytes()).hexdigest()
    assert rec["residual_overlap_after_exclusion"] == 0 and rec["evaluation"]["math500"]["ids_all_test_split"]
    assert rec["kept_records"] + len(rec["flagged"]["items"]) == rec["train_records"]
    assert {s for s in rec["calibration_sources"]} == {f"math_train_{s}" for s in src.MATH_SUBJECTS}


def test_chat_rendering_of_a_math_record():
    class Tok:
        chat_template = "x"

        def apply_chat_template(self, msgs, tokenize=False, enable_thinking=True):
            return "|".join(f"{m['role']}:{m['content']}" for m in msgs)

    spec = src.EXTRA_SOURCES["math_train_algebra"]
    assert src.render(spec, {"problem": "P?", "solution": "S."}, Tok()) == "user:P?|assistant:S."


def test_yaml_preset_through_the_loader():
    from suc.config import calibration_config_from_dict

    cfg = calibration_config_from_dict({"name": "c", "model": {"id": "Qwen/Qwen3-0.6B"}, "calibration": {"mix": "v1"}})
    assert cfg.calibration.mix == CALIB_MIX_V1 and config_delta(cfg.calibration) == {}


def test_frontier_queue_pins_the_finished_model_to_v1():
    from suc.orchestrate.config import load_queue_config

    cfg = load_queue_config(REPO / "experiments/queues/frontier_models.yaml")
    assert cfg.compress.calibration_mix == "v3"
    q38 = next(m for m in cfg.models if m.id == "Qwen/Qwen3.8-Flash-Next")
    assert q38.overrides["calibration"]["mix"] == "v1"
