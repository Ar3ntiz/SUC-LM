"""Fakes for the `suc queue` tests: a fake Hub (metadata from a local 'remote' directory), a fake download child
(copies the remote files into an HF-cache layout) and a fake `suc compress` child (log lines + result JSON)."""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

FAKE_FETCH = r'''
import argparse, hashlib, os, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--repo"); p.add_argument("--revision"); p.add_argument("--cache-dir")
p.add_argument("--allow", nargs="*"); p.add_argument("--ignore", nargs="*")
a, _ = p.parse_known_args()
from huggingface_hub.utils import filter_repo_objects
remote = Path(os.environ["FAKE_REMOTE"])
with open(remote / "fetch_calls.txt", "a") as f:
    f.write(a.repo + "\n")
(remote / "fetch_started").write_text(a.repo)
src = remote / a.repo.replace("/", "--")
repo = Path(a.cache_dir) / ("models--" + a.repo.replace("/", "--"))
snap = repo / "snapshots" / a.revision
truncate = remote / "truncate_once"
files = sorted(x for x in src.rglob("*") if x.is_file())
for f in filter_repo_objects(files, allow_patterns=a.allow, ignore_patterns=a.ignore,
                             key=lambda x: str(x.relative_to(src))):
    time.sleep(float(os.environ.get("FAKE_FETCH_SLEEP", "0")))
    data = f.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if f.suffix == ".safetensors" and truncate.exists():
        truncate.unlink()
        data = data[:-3]
    blob = repo / "blobs" / sha
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(data)
    link = snap / f.relative_to(src)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(os.path.relpath(blob, link.parent), link)
print("SNAPSHOT", snap, flush=True)
'''

FAKE_COMPRESS = r'''
import json, os, sys, time
from pathlib import Path
import yaml
cfg = yaml.safe_load(open(sys.argv[1]))
run = Path(cfg["run_dir"]); run.mkdir(parents=True, exist_ok=True)
with open(run / "attempts.txt", "a") as f:
    f.write("x\n")
def log(m):
    print(f"2026-10-03T00:00:00Z [compress] {m}", flush=True)
log(f"run {cfg['name']}: config {sys.argv[1]}")
if "broken" in cfg["model"]["id"]:
    log("stage encode: start")
    print("Traceback (most recent call last):\nValueError: fake encode failure", flush=True)
    sys.exit(1)
if os.environ.get("FAKE_COMPRESS_NODISK"):
    print("suc.pipeline.compress.PreflightError: preflight failed: insufficient disk at x: need ~9 GiB", flush=True)
    sys.exit(1)
log("preflight WARNING: fake warning")
log("stage calibrate: up to date, skipped (ran x, git y)")
log("stage encode: start")
for k in range(3):
    time.sleep(float(os.environ.get("FAKE_COMPRESS_SLEEP", "0")))
    log(f"encode model.layers.{k} ({k + 1}/3): 7 tensors, 4.250 bpw, 0.1 s")
log("stage encode: done in 0.3 s, peak MLX memory 0.10 GiB")
out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
res = out / f"20261003T000000Z_{cfg['name']}.json"
res.write_text(json.dumps({"metadata": {"git_commit": "abc"}, "size": {"bpp": 4.31, "file_bpp": 4.32, "bytes_total": 1000},
    "quality": {"perplexity": {"ppl": 10.5}, "kl": {"kl_mean": 0.05, "top1_agreement": 0.9}},
    "reference": {"perplexity": {"ppl": 10.0}}, "container": {"bytes": 1234, "path": str(run / "model.suc1")},
    "stages": {"encode": {"seconds": 0.3}}}))
res.with_suffix(".md").write_text("# report\n")
(run / "status.json").write_text(json.dumps({"state": "done", "result": str(res)}))
log(f"done {cfg['name']}: bpp 4.310")
'''


def safetensors_bytes(n_values: int, fill: int = 1) -> bytes:
    data = bytes([fill % 256, 2]) * n_values
    hdr = json.dumps({"w": {"dtype": "BF16", "shape": [n_values], "data_offsets": [0, len(data)]}}).encode()
    return struct.pack("<Q", len(hdr)) + hdr + data


def make_remote(remote: Path, repo: str, *, model_type: str = "qwen3", values: int = 64, shards: int = 2,
                fixture: str = "qwen3_0.6b_config.json") -> None:
    d = remote / repo.replace("/", "--")
    d.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((FIXTURES / fixture).read_text())
    cfg["model_type"] = model_type
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer.json").write_text("{}")
    names = [f"model-{i + 1:05d}-of-{shards:05d}.safetensors" for i in range(shards)]
    for i, n in enumerate(names):
        (d / n).write_bytes(safetensors_bytes(values, fill=i + 1))        # distinct shards = distinct blobs
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"t{i}": n for i, n in
                                                                             enumerate(names)}}))
    (d / "original").mkdir(exist_ok=True)
    (d / "original" / "consolidated.safetensors").write_bytes(safetensors_bytes(values))   # must be ignored


class FakeHub:
    def __init__(self, remote: Path) -> None:
        self.remote = remote
        self.calls: list[str] = []

    def metadata(self, repo: str, revision: str) -> dict:
        self.calls.append(repo)
        d = self.remote / repo.replace("/", "--")
        if not d.exists():
            from suc.orchestrate.hub import HubError
            raise HubError(f"{repo} not found")
        files = []
        params = 0
        for f in sorted(x for x in d.rglob("*") if x.is_file()):
            name = str(f.relative_to(d))
            data = f.read_bytes()
            lfs = f.suffix == ".safetensors"
            files.append({"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest() if lfs else None})
            if lfs and not name.startswith("original/"):
                params += (len(data) - 8 - struct.unpack("<Q", data[:8])[0]) // 2
        return {"sha": revision, "files": files, "dtypes": {"BF16": params},
                "config": json.loads((d / "config.json").read_text())}


def write_scripts(tmp: Path) -> tuple[list[str], list[str]]:
    (tmp / "fake_fetch.py").write_text(FAKE_FETCH)
    (tmp / "fake_compress.py").write_text(FAKE_COMPRESS)
    return [sys.executable, str(tmp / "fake_fetch.py")], [sys.executable, str(tmp / "fake_compress.py")]
