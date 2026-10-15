from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from suc.cli.bench import identity_check, resolved_revision
from suc.io.safetensors_store import SafetensorsStore
from tiny_models import tiny_qwen3


def _saved_store(model, tmp_path: Path) -> SafetensorsStore:
    mx.save_safetensors(str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    return SafetensorsStore.from_dir(tmp_path)


def test_identity_check_exact_for_unmodified_model(tmp_path):
    model = tiny_qwen3()
    store = _saved_store(model, tmp_path)
    res = identity_check(model, store)
    assert res["exact"] and res["max_abs_error"] == 0.0
    assert res["tensors_compared"] == len(store)


def test_identity_check_detects_perturbed_weight(tmp_path):
    model = tiny_qwen3()
    store = _saved_store(model, tmp_path)
    layer = model.model.layers[0].mlp.down_proj
    layer.weight = layer.weight + 1e-2
    res = identity_check(model, store)
    assert not res["exact"] and res["max_abs_error"] > 0.0


def test_identity_check_reports_dtype_mismatch(tmp_path):
    model = tiny_qwen3()
    store = _saved_store(model, tmp_path)
    model.set_dtype(mx.float16)
    res = identity_check(model, store)
    assert not res["exact"] and res["shape_or_dtype_mismatch"]


def test_resolved_revision_from_hf_snapshot_path(tmp_path):
    snap = tmp_path / "models--Qwen--Qwen3-0.6B" / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    assert resolved_revision(snap, "main") == "a" * 40


def test_resolved_revision_falls_back_for_local_dir(tmp_path):
    assert resolved_revision(tmp_path, "main") == "main"
