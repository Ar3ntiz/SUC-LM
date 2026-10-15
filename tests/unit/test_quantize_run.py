"""End-to-end test of the quality-track runner (suc quantize) on a tiny random Qwen3."""
import json
from pathlib import Path

import mlx.core as mx
import pytest
import yaml

import suc.quant.run as qrun
from suc.provenance import REQUIRED_METADATA_KEYS
from test_calibrate_run import _tiny_model_dir


def _setup(tmp_path: Path, monkeypatch, codecs: list[dict], **extra) -> Path:
    mx.random.seed(0)
    tokens = mx.random.randint(0, 128, (2048,), dtype=mx.int32)
    monkeypatch.setattr(qrun, "load_wikitext2", lambda split: ("unused", "wikihash"))
    monkeypatch.setattr(qrun, "tokenize", lambda tok, text: tokens)
    art = tmp_path / "tiny_phase2_data.safetensors"
    mx.save_safetensors(str(art), {"validation": mx.random.randint(0, 128, (8, 64), dtype=mx.int32)})
    cfg = {"name": "q_tiny", "output_dir": str(tmp_path / "results"),
           "model": {"id": "tiny/qwen3", "local_path": str(_tiny_model_dir(tmp_path))},
           "eval": {"seq_len": 128, "max_tokens": 1024, "kl_sequences": 4, "chunk": 2, "calib_artifact": str(art)},
           "codecs": codecs, **extra}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_run_quantize_codecs_and_reference(tmp_path, monkeypatch):
    path = _setup(tmp_path, monkeypatch, [{"kind": "bf16"}, {"kind": "affine", "bits": 4},
                                          {"kind": "vq", "k": 16, "block_size": 8}])
    r = json.loads(qrun.run_quantize(path).read_text())
    assert set(REQUIRED_METADATA_KEYS) <= set(r["metadata"]) and r["metadata"]["dataset_hash"] == "wikihash"
    rows = {row["codec"]: row for row in r["results"]}
    assert set(rows) == {"bf16", "affine_b4_g64", "vq_k16_g8_rot"}
    # the tiny checkpoint is float32, so the BF16 codec rounds it (on real BF16 checkpoints it is an identity)
    assert rows["bf16"]["quality"]["kl"]["kl_mean"] < 1e-3
    assert rows["bf16"]["quality"]["perplexity"]["ppl"] == pytest.approx(r["reference"]["perplexity"]["ppl"], rel=1e-3)
    # 4 bits + 32/64 scale+bias bits; the fixed 512-bit metadata per tensor weighs ~0.1 bpw on 64x64 tensors
    assert rows["affine_b4_g64"]["size"]["encoded_bpp"] == pytest.approx(4.5, abs=0.15)
    assert rows["vq_k16_g8_rot"]["quality"]["kl"]["kl_mean"] > 0
    assert "model.embed_tokens.weight" in rows["vq_k16_g8_rot"]["tensor_names"]  # include_embeddings default
    md = Path(r["report"]).read_text()
    assert md.startswith("# Quality track report: q_tiny") and "affine_b4_g64" in md


def test_run_quantize_without_embeddings_and_role_filter(tmp_path, monkeypatch):
    path = _setup(tmp_path, monkeypatch, [{"kind": "affine", "bits": 4}], include_embeddings=False,
                  roles=["mlp_down"])
    r = json.loads(qrun.run_quantize(path).read_text())
    names = r["results"][0]["tensor_names"]
    assert names and all(n.endswith("mlp.down_proj.weight") for n in names)


def _stats_artifact(tmp_path: Path) -> Path:
    from suc.analysis.activations import Recorder
    from mlx_lm.utils import load_model

    model, _ = load_model(tmp_path / "model")
    rec = Recorder()
    rec.instrument(model, "")
    model(mx.random.randint(0, 128, (2, 32), dtype=mx.int32))
    path = tmp_path / "tiny_activations.safetensors"
    rec.save(path)
    return path


def test_run_quantize_with_activation_residual(tmp_path, monkeypatch):
    path = _setup(tmp_path, monkeypatch, [{"kind": "vq", "k": 16, "block_size": 8},
                                          {"kind": "vq", "k": 16, "block_size": 8, "residual_fraction": 0.01}])
    cfg = yaml.safe_load(path.read_text())
    cfg["activation_stats"] = str(_stats_artifact(tmp_path))
    path.write_text(yaml.safe_dump(cfg))
    r = json.loads(qrun.run_quantize(path).read_text())
    base, res = r["results"]
    assert res["codec"] == "vq_k16_g8_rot+res0.01_activation"
    assert res["size"]["breakdown"]["residual"] > 0 and base["size"]["breakdown"]["residual"] == 0
    assert res["size"]["bpp"] > base["size"]["bpp"]


def test_run_quantize_with_blockwise_finetune(tmp_path, monkeypatch):
    path = _setup(tmp_path, monkeypatch, [{"kind": "vq", "k": 16, "block_size": 8}])
    cfg = yaml.safe_load(path.read_text())
    art = Path(cfg["eval"]["calib_artifact"])
    mx.save_safetensors(str(art), {"validation": mx.random.randint(0, 128, (8, 64), dtype=mx.int32),
                                   "calibration": mx.random.randint(0, 128, (8, 64), dtype=mx.int32)})
    cfg["finetune"] = {"block_steps": 3, "lr": 1e-3, "sequences": 4, "micro_batch": 2}
    path.write_text(yaml.safe_dump(cfg))
    r = json.loads(qrun.run_quantize(path).read_text())
    ft = r["results"][0]["finetune"]
    assert ft["blocks"] == 2 and ft["seconds"] > 0
    assert ft["loss_after_mean"] <= ft["loss_before_mean"]
    assert r["results"][0]["codec"] == "vq_k16_g8_rot+ft"


def test_run_quantize_with_rotation_folding(tmp_path, monkeypatch):
    seen = []
    real = qrun.make_codec
    monkeypatch.setattr(qrun, "make_codec", lambda d: seen.append(dict(d)) or real(d))
    path = _setup(tmp_path, monkeypatch, [
        {"kind": "bf16", "fold_rotation": True},
        {"kind": "vq", "k": 16, "block_size": 8, "rotate_out": False, "fold_rotation": True,
         "residual_fraction": 0.01},
        {"kind": "vq", "k": 16, "block_size": 8, "fold_rotation": True, "fold_v_heads": True},
        {"kind": "vq", "k": 16, "block_size": 8, "rotate_out": False, "fold_v_heads": True}])
    cfg = yaml.safe_load(path.read_text())
    cfg["activation_stats"] = str(_stats_artifact(tmp_path))
    art = Path(cfg["eval"]["calib_artifact"])
    arrs = dict(mx.load(str(art)))
    arrs["calibration"] = mx.random.randint(0, 128, (4, 32), dtype=mx.int32)
    mx.save_safetensors(str(art), arrs)
    path.write_text(yaml.safe_dump(cfg))
    r = json.loads(qrun.run_quantize(path).read_text())
    bf, res, vh, ho = r["results"]
    assert [x["codec"] for x in r["results"]] == ["bf16_fold", "vq_k16_g8_rot1s_fold+res0.01_activation",
                                                  "vq_k16_g8_rot_foldv", "vq_k16_g8_rot1s_vfold"]
    assert ho["fold"]["v_heads"] and not ho["fold"]["residual"]
    # folding is exact: only the BF16 rounding of the float32 tiny checkpoint remains
    assert bf["quality"]["kl"]["kl_mean"] < 1e-3 and bf["fold"]["tied_head"]
    assert res["fold"]["stats_recorded"] and res["size"]["breakdown"]["residual"] > 0
    assert {"kind": "vq", "k": 16, "block_size": 8, "rotate": True, "kmeans_iters": 25, "fit_blocks": 262144,
            "rotate_out": False, "rotate_in": False} in seen
    assert vh["fold"]["v_heads"]


def test_run_quantize_with_fixed_embedding_codec(tmp_path, monkeypatch):
    path = _setup(tmp_path, monkeypatch, [{"kind": "vq", "k": 16, "block_size": 8}],
                  embedding_codec={"kind": "affine", "bits": 4})
    r = json.loads(qrun.run_quantize(path).read_text())
    row = r["results"][0]
    assert row["codec"] == "vq_k16_g8_rot" and row["embedding_codec"] == "affine_b4_g64"
    by_name = {t["name"]: t for t in row["tensors"]}
    emb = by_name["model.embed_tokens.weight"]
    assert emb["bpw"] == pytest.approx(4.5, abs=0.15)             # affine 4-bit g64, not the 0.5-bpw VQ
    blk = [t for n, t in by_name.items() if ".layers." in n]
    assert blk and all(t["codec"] == "vq" for t in blk) and emb["codec"] == "affine"
