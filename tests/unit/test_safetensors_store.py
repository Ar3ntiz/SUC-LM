"""Tests for the header-only safetensors store (spec D8)."""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from suc.io.safetensors_store import SafetensorsStore, TensorInfo


def _arrays() -> dict[str, mx.array]:
    mx.random.seed(0)
    return {
        "a.bf16": mx.random.normal((4, 8)).astype(mx.bfloat16),
        "b.f16": mx.random.normal((3, 5)).astype(mx.float16),
        "c.f32": mx.random.normal((7,)).astype(mx.float32),
        "d.i32": mx.arange(6, dtype=mx.int32).reshape(2, 3),
        "e.u8": mx.arange(10, dtype=mx.uint8),
    }


def _assert_equal(a: mx.array, b: mx.array) -> None:
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert np.array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)))


def test_single_file(tmp_path: Path) -> None:
    arrs = _arrays()
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), arrs, metadata={"format": "mlx"})
    ref = mx.load(str(path))
    store = SafetensorsStore.from_dir(tmp_path)

    assert store.names() == sorted(arrs)
    assert len(store) == len(arrs)
    assert "c.f32" in store and "zzz" not in store
    assert store.shards() == [path]
    for name in arrs:
        _assert_equal(store.load(name), ref[name])
    info = store.info("a.bf16")
    assert isinstance(info, TensorInfo)
    assert info.dtype == "BF16" and info.shape == (4, 8) and info.nbytes == 64
    assert store.total_bytes() == sum(a.nbytes for a in arrs.values())


def test_load_middle_tensor_reads_only_its_bytes(tmp_path: Path) -> None:
    arrs = {f"t{i}": mx.full((16,), float(i), dtype=mx.float32) for i in range(5)}
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), arrs)
    store = SafetensorsStore.from_dir(tmp_path)
    info = store.info("t2")
    raw = path.read_bytes()[info.offset : info.offset + info.nbytes]
    assert np.array_equal(np.frombuffer(raw, dtype=np.float32), np.full(16, 2.0, np.float32))
    assert np.array_equal(np.array(store.load("t2")), np.full(16, 2.0, np.float32))


def test_sharded_index(tmp_path: Path) -> None:
    arrs = _arrays()
    names = sorted(arrs)
    s1, s2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    mx.save_safetensors(str(tmp_path / s1), {n: arrs[n] for n in names[:2]})
    mx.save_safetensors(str(tmp_path / s2), {n: arrs[n] for n in names[2:]})
    weight_map = {n: (s1 if i < 2 else s2) for i, n in enumerate(names)}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    store = SafetensorsStore.from_dir(tmp_path)
    assert store.names() == names
    assert store.shards() == [tmp_path / s1, tmp_path / s2]
    assert store.info(names[3]).shard == tmp_path / s2
    for n in names:
        _assert_equal(store.load(n), arrs[n])


def test_fallback_glob_and_duplicates(tmp_path: Path) -> None:
    mx.save_safetensors(str(tmp_path / "x.safetensors"), {"a": mx.zeros((2,))})
    mx.save_safetensors(str(tmp_path / "y.safetensors"), {"b": mx.ones((2,))})
    store = SafetensorsStore.from_dir(tmp_path)
    assert store.names() == ["a", "b"]
    mx.save_safetensors(str(tmp_path / "z.safetensors"), {"a": mx.ones((2,))})
    with pytest.raises(ValueError):
        SafetensorsStore.from_dir(tmp_path)


def test_unknown_name_raises_keyerror(tmp_path: Path) -> None:
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"a": mx.zeros((2,))})
    store = SafetensorsStore.from_dir(tmp_path)
    with pytest.raises(KeyError):
        store.load("missing")
    with pytest.raises(KeyError):
        store.info("missing")


def test_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SafetensorsStore.from_dir(tmp_path)
