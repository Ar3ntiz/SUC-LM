"""Tests for suc.provenance: git info, runtime info, metadata, result writing, seeding."""
import json
import random
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from suc.provenance import (
    REQUIRED_METADATA_KEYS, git_info, run_metadata, runtime_info, seed_everything,
    utc_timestamp, write_result,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path) -> Path:
    r = tmp_path / "repo"; r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    return r


def test_git_info_no_commits(repo):
    info = git_info(repo)
    assert info["commit"] is None and info["dirty"] is False
    (repo / "a.txt").write_text("x")
    assert git_info(repo)["dirty"] is True


def test_git_info_with_commit(repo):
    (repo / "a.txt").write_text("x")
    _git(repo, "add", "a.txt"); _git(repo, "commit", "-q", "-m", "init")
    info = git_info(repo)
    assert info["commit"] == _git(repo, "rev-parse", "HEAD") and len(info["commit"]) == 40
    assert info["dirty"] is False and info["branch"] == "main"
    (repo / "a.txt").write_text("y")
    assert git_info(repo)["dirty"] is True


def test_git_info_not_a_repo(tmp_path):
    info = git_info(tmp_path / "nope")
    assert info == {"commit": None, "dirty": None, "branch": None}


def test_runtime_info():
    info = runtime_info()
    for k in ("python", "platform", "machine", "chip", "memory_bytes", "mlx", "mlx_lm", "mlx_device", "hostname"):
        assert k in info
    json.dumps(info)


def test_utc_timestamp():
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", utc_timestamp())


def test_run_metadata(repo):
    md = run_metadata(model="m", model_revision="main", model_hash="h", dataset_hash="d", seed=1,
                      config_hash="c", repo_dir=repo)
    assert set(REQUIRED_METADATA_KEYS) <= set(md)
    assert set(md) == {"model", "model_revision", "model_hash", "git_commit", "git_dirty", "dataset_hash",
                       "seed", "config_hash", "runtime", "timestamp"}
    assert md["git_commit"] is None and md["seed"] == 1
    json.dumps(md)


def _result(repo) -> dict:
    md = run_metadata(model="m", model_revision="main", model_hash="h", dataset_hash="d", seed=1,
                      config_hash="c", repo_dir=repo)
    return {"metadata": md, "b": 2, "a": 1}


def test_write_result(tmp_path, repo):
    out = tmp_path / "out"
    p = write_result(_result(repo), out, "my run/x")
    assert p.parent == out and p.suffix == ".json"
    assert re.fullmatch(r"\d{8}T\d{6}Z_[A-Za-z0-9._-]+\.json", p.name)
    assert "/" not in p.name[17:] and " " not in p.name
    text = p.read_text()
    assert text.endswith("\n")
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n"
    assert json.loads(text)["a"] == 1


def test_write_result_never_overwrites(tmp_path, repo):
    paths = {write_result(_result(repo), tmp_path, "same") for _ in range(3)}
    assert len(paths) == 3 and all(p.exists() for p in paths)


@pytest.mark.parametrize("drop", REQUIRED_METADATA_KEYS)
def test_write_result_requires_metadata(tmp_path, repo, drop):
    r = _result(repo); del r["metadata"][drop]
    with pytest.raises(ValueError, match=drop):
        write_result(r, tmp_path, "x")


def test_write_result_missing_metadata_block(tmp_path):
    with pytest.raises(ValueError):
        write_result({"a": 1}, tmp_path, "x")


def test_seed_everything():
    seed_everything(3); a = (random.random(), np.random.rand())
    seed_everything(3); b = (random.random(), np.random.rand())
    assert a == b
    import mlx.core as mx
    seed_everything(3); x = mx.random.uniform(shape=(4,)).tolist()
    seed_everything(3); y = mx.random.uniform(shape=(4,)).tolist()
    assert x == y


def test_git_info_ignores_untracked_result_files(repo):
    (repo / "a.txt").write_text("x")
    _git(repo, "add", "a.txt"); _git(repo, "commit", "-q", "-m", "init")
    (repo / "experiments" / "results").mkdir(parents=True)
    (repo / "experiments" / "results" / "r.json").write_text("{}")
    assert git_info(repo)["dirty"] is False
    (repo / "new_module.py").write_text("x = 1")
    assert git_info(repo)["dirty"] is True
