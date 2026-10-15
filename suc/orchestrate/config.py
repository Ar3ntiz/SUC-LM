"""Queue file of `suc queue`: the models to compress unattended, in order, with their compress-config recipe.

Unknown keys are errors (as in suc.config). Paths starting with "~" are expanded; relative paths are relative to
the repository root (the working directory of `suc queue`), except queue_dir-derived defaults.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from suc.config import _build

SourcePolicy = Literal["keep", "delete_after_success"]

# Weights, config, tokenizer (incl. custom tokenizer code / tiktoken vocabularies). Same semantics as
# huggingface_hub.snapshot_download (fnmatch; "*" also matches "/").
DEFAULT_ALLOW = ("*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py", "*.tiktoken")
# Duplicate weight copies some repos ship next to the HF safetensors (gpt-oss: original/, metal/), other formats.
DEFAULT_IGNORE = ("original/*", "metal/*", "*.bin", "*.pth", "*.pt", "*.gguf", "*.onnx", "*.msgpack", "*.h5")


@dataclass(frozen=True)
class SpaceConfig:
    """Disk-space policy. A model fits when free - need * margin >= min_free_gib on every volume it uses."""
    min_free_gib: float = 50.0          # hard floor: also the low-disk watchdog threshold during download/compress
    margin: float = 1.15                # multiplier on the estimated need
    estimated_bpp: float = 4.6          # SUC1 size estimate: bits per (logical) parameter, incl. head / raw tensors
    workspace_factor: float = 3.0       # compress run dir ~ factor x SUC1 size (encode shards + container + temp)
    workspace_fixed_gib: float = 30.0   # + calibration data, reference logits, spilled hidden states
    watchdog_seconds: float = 15.0      # free-disk poll interval while a child runs

    def __post_init__(self) -> None:
        if self.margin < 1 or self.min_free_gib < 0 or self.estimated_bpp <= 0 or self.workspace_factor < 0:
            raise ValueError("space: margin >= 1, min_free_gib >= 0, estimated_bpp > 0, workspace_factor >= 0")


@dataclass(frozen=True)
class DownloadConfig:
    allow_patterns: tuple[str, ...] = DEFAULT_ALLOW
    ignore_patterns: tuple[str, ...] = DEFAULT_IGNORE
    retries: int = 12                   # attempts of the whole (resumable) snapshot download
    backoff_seconds: float = 30.0       # first retry delay; doubles up to max_backoff_seconds
    max_backoff_seconds: float = 900.0
    max_workers: int = 8
    command: tuple[str, ...] | None = None   # None: [python, -m, suc.orchestrate.fetch] (tests use a fake)


@dataclass(frozen=True)
class CompressStepConfig:
    template_dense: str = "experiments/configs/compress_qwen3_8b_final.yaml"
    template_moe: str = "experiments/configs/compress_qwen3_30b_a3b_final.yaml"
    output_dir: str = "experiments/results"  # result JSON (committed on the Studio); a copy goes to queue_dir
    command: tuple[str, ...] | None = None   # None: [python, -m, suc.cli.main, compress] (tests use a fake)
    # calibration mix preset (suc.config.CALIB_MIXES) written into every generated config whose queue / model
    # overrides do not set calibration.mix themselves (it replaces the template's mix); None = keep the template's.
    # Default v3 (v1 with a 3% MATH-train share; v2's even split lost MMLU-Pro) for new compressions; pin `overrides: {calibration: {mix: v1}}` on
    # a model whose earlier run must keep its fingerprints.
    calibration_mix: str | None = "v3"

    def __post_init__(self) -> None:
        from suc.config import CALIB_MIXES

        if self.calibration_mix is not None and self.calibration_mix not in CALIB_MIXES:
            raise ValueError(f"compress.calibration_mix must be one of {sorted(CALIB_MIXES)} or null, "
                             f"got {self.calibration_mix!r}")


@dataclass(frozen=True)
class PollConfig:
    progress_seconds: float = 30.0      # download progress lines
    heartbeat_seconds: float = 30.0     # status.json heartbeat
    decisions_seconds: float = 60.0     # re-read decisions.yaml / re-check space while only blocked models remain
    child_seconds: float = 2.0          # child process / log poll


@dataclass(frozen=True)
class ModelEntry:
    id: str
    revision: str
    priority: int = 0                   # higher first; ties keep the file order
    template: str | None = None         # compress config template; None = template_dense / template_moe by config
    overrides: dict = field(default_factory=dict)   # deep-merged into the template (lists replace)
    source_policy: SourcePolicy | None = None        # None = queue default
    estimated_bpp: float | None = None
    enabled: bool = True
    notes: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError(f"models[{self.id}].revision must be a pinned 40-hex commit, got {self.revision!r}")
        if not isinstance(self.overrides, dict):
            raise ValueError(f"models[{self.id}].overrides must be a mapping")
        if self.source_policy not in (None, "keep", "delete_after_success"):
            raise ValueError(f"models[{self.id}].source_policy must be 'keep' or 'delete_after_success'")

    @property
    def slug(self) -> str:
        return slug(self.id)


@dataclass(frozen=True)
class QueueConfig:
    name: str
    queue_dir: str
    models: tuple[ModelEntry, ...]
    hf_cache: str = "~/.cache/huggingface/hub"     # HF_HUB_CACHE for download and compress (local disk)
    work_root: str | None = None                    # compress run dirs; None = <queue_dir>/runs
    source_policy: SourcePolicy = "keep"
    overrides: dict = field(default_factory=dict)   # applied to every model before its own overrides
    wait_for_decisions: bool = True                 # keep running (polling) while only blocked models remain
    space: SpaceConfig = field(default_factory=SpaceConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    compress: CompressStepConfig = field(default_factory=CompressStepConfig)
    poll: PollConfig = field(default_factory=PollConfig)

    def __post_init__(self) -> None:
        ids = [m.id for m in self.models]
        if len(set(ids)) != len(ids):
            raise ValueError(f"queue: duplicate model ids {ids}")
        if not self.models:
            raise ValueError("queue: 'models' must be a non-empty list")
        if self.source_policy not in ("keep", "delete_after_success"):
            raise ValueError("queue: source_policy must be 'keep' or 'delete_after_success'")

    @property
    def queue_path(self) -> Path:
        return Path(self.queue_dir).expanduser()

    @property
    def cache_path(self) -> Path:
        return Path(self.hf_cache).expanduser()

    @property
    def work_path(self) -> Path:
        return Path(self.work_root).expanduser() if self.work_root else self.queue_path / "runs"

    def ordered(self) -> list[ModelEntry]:
        """Enabled models by priority (descending), file order within a priority."""
        idx = {m.id: i for i, m in enumerate(self.models)}
        return sorted((m for m in self.models if m.enabled), key=lambda m: (-m.priority, idx[m.id]))


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.replace("/", "__"))


def deep_merge(base: Any, over: Any) -> Any:
    """Recursive dict merge; any non-dict value (lists included) in `over` replaces the base value."""
    if isinstance(base, dict) and isinstance(over, dict):
        out = copy.deepcopy(base)
        for k, v in over.items():
            out[k] = deep_merge(out[k], v) if k in out else copy.deepcopy(v)
        return out
    return copy.deepcopy(over)


def _tuples(d: dict, keys: tuple[str, ...]) -> dict:
    return {k: (tuple(v) if k in keys and isinstance(v, list) else v) for k, v in d.items()}


def queue_config_from_dict(d: dict) -> QueueConfig:
    if not isinstance(d, dict):
        raise ValueError("queue: expected a mapping")
    d = dict(d)
    models = d.pop("models", None)
    if not isinstance(models, list):
        raise ValueError("queue: 'models' must be a list")
    entries = tuple(_build(ModelEntry, m, f"models[{i}]", overrides=(m or {}).get("overrides") or {})
                    if isinstance(m, dict) else _build(ModelEntry, m, f"models[{i}]") for i, m in enumerate(models))
    nested = {
        "models": entries,
        "overrides": d.pop("overrides", None) or {},
        "space": _build(SpaceConfig, d.pop("space", None) or {}, "space"),
        "download": _build(DownloadConfig, _tuples(d.pop("download", None) or {},
                                                   ("allow_patterns", "ignore_patterns", "command")), "download"),
        "compress": _build(CompressStepConfig, _tuples(d.pop("compress", None) or {}, ("command",)), "compress"),
        "poll": _build(PollConfig, d.pop("poll", None) or {}, "poll"),
    }
    return _build(QueueConfig, d, "queue", **nested)


def load_queue_config(path: str | Path) -> QueueConfig:
    with open(path) as f:
        return queue_config_from_dict(yaml.safe_load(f) or {})
