"""Audit F02: artifact identity is the sha256 of the complete file, never a neighbouring sidecar's claim."""
from __future__ import annotations

import hashlib
import json
import os
import shutil

import pytest

from suc.io import hashing
from suc.io.hashing import artifact_identity


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setattr(hashing, "_HASH_CACHE", tmp_path / "cache" / "h.json")
    monkeypatch.delenv("SUC_TRUST_HASH_CACHE", raising=False)


def _sidecar(d, target, data: bytes):
    (d / "container.json").write_text(json.dumps({"path": str(target), "bytes": len(data), "sha256": _sha(data)}))


def test_same_sized_neighbour_does_not_inherit_sidecar_hash(tmp_path):
    (tmp_path / "model.suc1").write_bytes(b"AAAA")
    (tmp_path / "other.suc1").write_bytes(b"BBBB")
    _sidecar(tmp_path, tmp_path / "model.suc1", b"AAAA")
    assert artifact_identity(tmp_path / "other.suc1")["sha256"] == _sha(b"BBBB")
    assert artifact_identity(tmp_path / "model.suc1")["sha256"] == _sha(b"AAAA")


def test_same_path_same_size_replacement(tmp_path):
    f = tmp_path / "model.suc1"
    f.write_bytes(b"AAAA")
    _sidecar(tmp_path, f, b"AAAA")
    assert artifact_identity(f)["sha256"] == _sha(b"AAAA")
    f.write_bytes(b"CCCC")
    assert artifact_identity(f)["sha256"] == _sha(b"CCCC")
    # even a trusted stat cache misses: mtime/ctime changed
    assert artifact_identity(f, trust_cache=True) == {"path": str(f.resolve()), "bytes": 4, "sha256": _sha(b"CCCC"),
                                                       "sha256_source": "stat_cache"}


def test_moved_and_copied_files_keep_identity(tmp_path):
    f = tmp_path / "a" / "model.suc1"
    f.parent.mkdir()
    f.write_bytes(b"payload")
    h = artifact_identity(f)["sha256"]
    moved = tmp_path / "b.suc1"
    shutil.move(f, moved)
    copied = tmp_path / "c.suc1"
    shutil.copy(moved, copied)
    for p in (moved, copied):
        r = artifact_identity(p, trust_cache=True)
        assert r["sha256"] == h and r["sha256_source"] == "full_hash"     # new path: no cache entry, hashed


@pytest.mark.parametrize("sidecar", [None, "not json", json.dumps({"bytes": "x"}), json.dumps([1, 2])])
def test_missing_or_malformed_sidecar(tmp_path, sidecar):
    f = tmp_path / "model.suc1"
    f.write_bytes(b"xyz")
    if sidecar is not None:
        (tmp_path / "container.json").write_text(sidecar)
    assert artifact_identity(f)["sha256"] == _sha(b"xyz")


def test_unchanged_file_cache_is_opt_in_and_labelled(tmp_path, monkeypatch):
    f = tmp_path / "model.suc1"
    f.write_bytes(b"z" * 1000)
    assert artifact_identity(f)["sha256_source"] == "full_hash"
    assert artifact_identity(f)["sha256_source"] == "full_hash"           # default: verify every time
    assert artifact_identity(f, trust_cache=True)["sha256_source"] == "stat_cache"
    monkeypatch.setenv("SUC_TRUST_HASH_CACHE", "1")
    r = artifact_identity(f)
    assert r["sha256_source"] == "stat_cache" and r["sha256"] == _sha(b"z" * 1000)
    hashing._HASH_CACHE.write_text("garbage")                            # corrupt cache: re-hash
    assert artifact_identity(f)["sha256_source"] == "full_hash"


def test_runner_helper_uses_full_file(tmp_path):
    rn = pytest.importorskip("suc.evaluation.runners")
    (tmp_path / "other.suc1").write_bytes(b"BBBB")
    _sidecar(tmp_path, tmp_path / "model.suc1", b"AAAA")
    assert rn.suc1_sha256(tmp_path / "other.suc1") == _sha(b"BBBB")
    assert os.path.exists(tmp_path / "container.json")
