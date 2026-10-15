"""Tests for deterministic model hashing."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from suc.io.hashing import model_hash, sha256_file


def _make_model(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "model-00001-of-00002.safetensors").write_bytes(b"\x01" * 1000)
    (d / "model-00002-of-00002.safetensors").write_bytes(b"\x02" * 500)
    (d / "config.json").write_text('{"a": 1}')
    (d / "tokenizer.json").write_text("ignored")


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello" * 1000)
    assert sha256_file(p, chunk_size=7) == hashlib.sha256(b"hello" * 1000).hexdigest()


def test_model_hash_deterministic_and_sensitive(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _make_model(a)
    _make_model(b)
    h = model_hash(a)
    assert len(h) == 64
    assert h == model_hash(a) == model_hash(b)
    (b / "tokenizer.json").write_text("changed")  # not part of the hash
    assert model_hash(b) == h
    data = bytearray((b / "model-00002-of-00002.safetensors").read_bytes())
    data[10] ^= 0xFF
    (b / "model-00002-of-00002.safetensors").write_bytes(bytes(data))
    assert model_hash(b) != h
    (a / "config.json").write_text('{"a": 2}')
    assert model_hash(a) != h


def _make_hf_cache(root: Path, corrupt: bool = False) -> tuple[Path, Path]:
    """Build a fake HF cache: blobs/<sha256> for LFS, blobs/<sha1> for config, snapshot symlinks."""
    plain = root / "plain"
    _make_model(plain)
    blobs = root / "blobs"
    snap = root / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    for f in plain.iterdir():
        data = f.read_bytes()
        if f.suffix == ".safetensors":
            name = hashlib.sha256(data).hexdigest()
            if corrupt and f.name.startswith("model-00001"):
                data = data[:-1] + b"\x00"
        else:
            name = hashlib.sha1(data).hexdigest()
        (blobs / name).write_bytes(data)
        os.symlink(os.path.relpath(blobs / name, snap), snap / f.name)
    return plain, snap


def test_hf_cache_fast_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plain, snap = _make_hf_cache(tmp_path)
    expected = model_hash(plain)

    import suc.io.hashing as hashing

    calls: list[Path] = []
    real = hashing.sha256_file

    def spy(path, chunk_size=16 << 20):  # type: ignore[no-untyped-def]
        calls.append(Path(path))
        return real(path, chunk_size)

    monkeypatch.setattr(hashing, "sha256_file", spy)
    assert model_hash(snap) == expected
    # Only config.json (git-sha1 blob) must be read; LFS shards use the blob name.
    assert [p.name for p in calls] == ["config.json"]
    calls.clear()
    assert model_hash(snap, verify=True) == expected
    assert len(calls) == 3


def test_hf_cache_verify_detects_mismatch(tmp_path: Path) -> None:
    plain, snap = _make_hf_cache(tmp_path, corrupt=True)
    # Fast path trusts the blob name, so it still reports the expected hash.
    assert model_hash(snap) == model_hash(plain)
    with pytest.raises(ValueError):
        model_hash(snap, verify=True)
