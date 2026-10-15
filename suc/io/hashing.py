"""Deterministic hashing of model weights (spec T0.4 / D8)."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path, chunk_size: int = 16 << 20) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


_HASH_CACHE = Path(os.environ.get("SUC_HASH_CACHE_FILE", "~/.cache/suc/file_sha256.json")).expanduser()


def _stat_key(st: os.stat_result) -> list:
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def artifact_identity(path: str | Path, trust_cache: bool | None = None) -> dict:
    """Identity of an artifact file (audit F02): streaming sha256 of the complete file.

    Never trusts a neighbouring sidecar (container.json) - a same-sized other file or a same-path replacement would
    otherwise inherit its hash. Every full hash is recorded in a cache keyed by the resolved path and its stat
    identity (device, inode, size, mtime, ctime); the cache is only *used* when ``trust_cache`` is true (default: env
    ``SUC_TRUST_HASH_CACHE=1``), and the result then says ``sha256_source: "stat_cache"`` instead of ``"full_hash"``
    - a stat match is a performance/trust trade-off, not proof of equal content.
    """
    import json

    if trust_cache is None:
        trust_cache = os.environ.get("SUC_TRUST_HASH_CACHE") == "1"
    f = Path(path).expanduser().resolve()
    st = f.stat()
    key = str(f)
    try:
        cache = json.loads(_HASH_CACHE.read_text())
        if not isinstance(cache, dict):
            cache = {}
    except (OSError, ValueError):
        cache = {}
    hit = cache.get(key)
    if trust_cache and isinstance(hit, dict) and hit.get("stat") == _stat_key(st) and _SHA256_RE.match(
            str(hit.get("sha256", ""))):
        return {"path": key, "bytes": st.st_size, "sha256": hit["sha256"], "sha256_source": "stat_cache"}
    digest = sha256_file(f)
    st2 = f.stat()
    if _stat_key(st2) == _stat_key(st):                    # unchanged while hashing: cacheable
        cache[key] = {"stat": _stat_key(st), "sha256": digest}
        try:
            _HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _HASH_CACHE.with_name(f"{_HASH_CACHE.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(cache, indent=0, sort_keys=True))
            os.replace(tmp, _HASH_CACHE)
        except OSError:
            pass
    return {"path": key, "bytes": st.st_size, "sha256": digest, "sha256_source": "full_hash"}


def _file_sha256(path: Path, verify: bool) -> str:
    """sha256 of a file, using the HF-cache LFS blob name when available."""
    if path.is_symlink():
        blob = os.path.basename(os.readlink(path))
        if _SHA256_RE.match(blob):
            if not verify:
                return blob
            actual = sha256_file(path)
            if actual != blob:
                raise ValueError(f"sha256 mismatch for {path}: blob {blob}, content {actual}")
            return actual
    return sha256_file(path)


def model_hash(model_dir: str | Path, verify: bool = False) -> str:
    """Hash identifying the exact weights: all *.safetensors files plus config.json.

    sha256 over the sorted lines "<relative filename>\\t<sha256 of file>\\n".
    """
    d = Path(model_dir)
    files = sorted(d.glob("*.safetensors"))
    if (d / "config.json").exists():
        files.append(d / "config.json")
    if not files:
        raise FileNotFoundError(f"no safetensors/config.json in {d}")
    lines = sorted(f"{f.relative_to(d).as_posix()}\t{_file_sha256(f, verify)}\n" for f in files)
    return hashlib.sha256("".join(lines).encode()).hexdigest()
