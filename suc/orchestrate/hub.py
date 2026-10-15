"""Hugging Face metadata (check step) and HF-cache layout helpers.

The orchestrator talks to the Hub only through a `HubClient` (`metadata(repo, revision) -> dict`), so the tests use
a fake one. The real client needs network access: HF_HUB_OFFLINE must not be set in the orchestrator process.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable, Protocol

# The Hub's safetensors "parameters" are LOGICAL weight counts per stored dtype, also for packed quantized
# checkpoints (checked on openai/gpt-oss-120b: BF16 2.2 B + U8 114.7 B = 116.8 B parameters, 61 GiB of files), so
# the parameter count is their plain sum; stored bytes come from the file sizes, never from count x dtype width.

_EXPERT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts")


class HubError(RuntimeError):
    """Permanent metadata failure (repository / revision not found, gated without access)."""


class HubClient(Protocol):
    def metadata(self, repo: str, revision: str) -> dict:
        """{"sha": commit, "files": [{"name", "size", "sha256" | None}], "dtypes": {dtype: n_params},
        "config": config.json dict, "gated": ..., "library": ...}"""
        ...


class HfHubClient:
    """Real client (huggingface_hub). Transient errors are retried with backoff; permanent ones raise HubError."""

    def __init__(self, cache_dir: Path, retries: int = 6, backoff_seconds: float = 10.0) -> None:
        self.cache_dir = Path(cache_dir)
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    def metadata(self, repo: str, revision: str) -> dict:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        import huggingface_hub.constants as hf_constants
        from huggingface_hub import HfApi, hf_hub_download

        hf_constants.HF_HUB_OFFLINE = False      # read at import time: undo an inherited HF_HUB_OFFLINE=1
        from huggingface_hub.errors import (EntryNotFoundError, GatedRepoError, RepositoryNotFoundError,
                                            RevisionNotFoundError)

        delay = self.backoff_seconds
        for attempt in range(self.retries):
            try:
                info = HfApi().model_info(repo, revision=revision, files_metadata=True)
                cfg_file = hf_hub_download(repo, "config.json", revision=revision, cache_dir=str(self.cache_dir))
                break
            except GatedRepoError as ex:
                raise HubError(f"{repo} is gated: accept its license on huggingface.co and log in on this machine "
                               f"(`hf auth login` or HF_TOKEN): {ex}") from ex
            except (RepositoryNotFoundError, RevisionNotFoundError) as ex:
                raise HubError(f"{repo}@{revision} not found on the Hub: {ex}") from ex
            except EntryNotFoundError as ex:
                raise HubError(f"{repo}@{revision} has no config.json: {ex}") from ex
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 600)
        files = [{"name": s.rfilename, "size": s.size, "sha256": s.lfs.sha256 if s.lfs else None}
                 for s in info.siblings or []]
        st = getattr(info, "safetensors", None)
        return {"sha": info.sha, "files": files, "dtypes": dict(st.parameters) if st else {},
                "config": json.loads(Path(cfg_file).read_text()), "gated": getattr(info, "gated", None),
                "library": getattr(info, "library_name", None)}


def select_files(files: Iterable[dict], allow: Iterable[str], ignore: Iterable[str]) -> list[dict]:
    """The files snapshot_download(allow_patterns, ignore_patterns) would fetch."""
    from huggingface_hub.utils import filter_repo_objects

    return list(filter_repo_objects(list(files), allow_patterns=list(allow), ignore_patterns=list(ignore),
                                    key=lambda f: f["name"]))


def _sub_configs(cfg: dict) -> list[dict]:
    return [cfg] + [v for k, v in cfg.items() if k.endswith("_config") and isinstance(v, dict)]


def num_experts(cfg: dict) -> int:
    return max((int(c.get(k) or 0) for c in _sub_configs(cfg) for k in _EXPERT_KEYS), default=0)


def architecture(cfg: dict) -> dict:
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    q = cfg.get("quantization_config") or text.get("quantization_config") or None
    return {"model_type": cfg.get("model_type"), "architectures": cfg.get("architectures"),
            "num_layers": text.get("num_hidden_layers"), "hidden_size": text.get("hidden_size"),
            "num_experts": num_experts(cfg), "is_moe": num_experts(cfg) > 0,
            "source_quantization": (q.get("quant_method") or q.get("quant_type") or "unknown") if isinstance(q, dict)
            else None}


def adapter_support(cfg: dict) -> tuple[bool, str]:
    """(supported, reason): is there a registered SUC architecture adapter for this config.json?"""
    from suc.models.adapters import get_adapter

    try:
        a = get_adapter(cfg)
    except Exception as ex:          # UnsupportedArchitecture, or an adapter rejecting the config
        return False, f"{type(ex).__name__}: {ex}"
    return True, f"adapter {type(a).__name__} (family {getattr(a, 'family', '?')})"


def size_summary(meta: dict, allow: Iterable[str], ignore: Iterable[str]) -> dict:
    sel = select_files(meta["files"], allow, ignore)
    dtypes = meta.get("dtypes") or {}
    weights = sum(dtypes.values())
    st = [f for f in sel if f["name"].endswith(".safetensors")]
    if not weights:                  # Hub safetensors metadata missing: assume BF16 storage
        weights = sum(f["size"] or 0 for f in st) // 2
    return {"download_bytes": int(sum(f["size"] or 0 for f in sel)), "files": len(sel),
            "safetensors_files": len(st), "safetensors_bytes": int(sum(f["size"] or 0 for f in st)),
            "repo_bytes": int(sum(f["size"] or 0 for f in meta["files"])),
            "params_by_dtype": dict(dtypes), "logical_weights": int(weights),
            "stored_bits_per_weight": 8 * sum(f["size"] or 0 for f in st) / weights if weights else None}


def params_text(sizes: dict) -> str:
    return ", ".join(f"{k} {n / 1e9:.2f} B" for k, n in sizes.get("params_by_dtype", {}).items()) or "?"


# ------------------------------------------------------------------------------------------- HF cache layout

def repo_cache_dir(cache: Path, repo: str) -> Path:
    return Path(cache) / f"models--{repo.replace('/', '--')}"


def snapshot_dir(cache: Path, repo: str, revision: str) -> Path:
    return repo_cache_dir(cache, repo) / "snapshots" / revision


def cached_bytes(cache: Path, repo: str) -> int:
    """Bytes in the repo's blob store, partial (.incomplete) downloads included: download progress."""
    blobs = repo_cache_dir(cache, repo) / "blobs"
    total = 0
    if blobs.is_dir():
        for f in os.scandir(blobs):
            try:
                total += f.stat(follow_symlinks=False).st_size
            except OSError:
                pass
    return total


def present_bytes(cache: Path, repo: str, revision: str, files: Iterable[dict]) -> int:
    """Bytes of the selected files already complete in the snapshot (size matches)."""
    snap = snapshot_dir(cache, repo, revision)
    n = 0
    for f in files:
        p = snap / f["name"]
        try:
            if p.exists() and (f["size"] is None or p.stat().st_size == f["size"]):
                n += p.stat().st_size
        except OSError:
            pass
    return n
