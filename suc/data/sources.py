"""Pinned raw-text sources for the calibration corpus (spec §9 T2.1, §32).

One small, non-gated file per domain, pinned to a dataset commit. Selection rules:
- never a public benchmark's test split (GSM8K: train only; PubMedQA: the unlabeled subset, not
  the labeled benchmark); never Wikipedia-derived text (WikiText-2 test is the final-test set,
  suc/evaluation/data.py);
- the OOD source is never used for calibration or validation.

Licenses are the dataset-card strings (C4 ODC-BY; GSM8K MIT; Dolly CC-BY-SA-3.0; PubMedQA MIT;
MATH (Hendrycks et al., EleutherAI/hendrycks_math) MIT). codeparrot-clean-valid has no card license:
every record carries its GitHub repository license and only permissively licensed files are kept
(`record_filter`).

Base corpus vs extra domains. `SOURCES` is the corpus of every run before 2026-10-11 and must not
change (its repr is the corpus identity checked when a calibrate stage is reused). Domains added
later live in `EXTRA_SOURCES` and are loaded only when the calibration mix names their domain
(`sources_for`), so the corpus identity of a v1-mix run is unchanged:

- "math_competition" (calibration mix v2, suc.config.CALIB_MIX_V2): the MATH TRAIN split (7 subject
  files, 7,500 problems with worked solutions), chat-rendered (user = problem, assistant =
  solution). MATH-500 is a subset of the MATH TEST split, so it is disjoint by construction; on top
  of that every record whose normalized problem text (`problem_key`) is listed in the pinned
  exclusion file is dropped: the problems of all 500 MATH-500 items and of every case of every eval
  suite in experiments/eval, plus train problems that near-duplicate any of them (word 13-gram
  overlap). The file and the disjointness record are written by
  scripts/build_calibration_exclusions.py (record: experiments/eval/calibration_disjointness.json).
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from suc.evaluation.data import _sha256

Domain = Literal["natural", "code", "math", "math_competition", "instruction", "ood"]
EXCLUSIONS_DIR = Path(__file__).resolve().parent / "exclusions"
PERMISSIVE_CODE_LICENSES = ("mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "unlicense", "cc0-1.0")


@dataclass(frozen=True)
class SourceSpec:
    """A pinned source file. `text_fields` may be dotted paths into nested records ("a.b"); list
    values are joined with "\\n". `record_filter` = ((field, allowed values), ...) keeps only
    records whose field value is allowed (applied before `max_docs`, which keeps the first N
    records in file order)."""
    name: str
    domain: Domain
    repo_id: str
    filename: str
    revision: str
    format: Literal["parquet", "jsonl", "jsonl.gz", "json.gz"]
    text_fields: tuple[str, ...]
    template: Literal["plain", "qa", "chat"] = "plain"
    license: str = ""
    max_docs: int | None = None
    record_filter: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class ExcludingSourceSpec(SourceSpec):
    """A SourceSpec whose records are dropped when `problem_key(record[exclude_field])` is listed in
    the exclusion file EXCLUSIONS_DIR / exclude_file ({"hashes": [...]}; applied before `max_docs`).
    A subclass, so the repr of the base SOURCES (the v1 corpus identity) stays unchanged."""
    exclude_field: str = ""
    exclude_file: str = ""


SOURCES: dict[str, SourceSpec] = {s.name: s for s in (
    SourceSpec("c4_en_val0", "natural", "allenai/c4", "en/c4-validation.00000-of-00008.json.gz",
               "1588ec454efa1a09f29cd18ddd04fe05fc8653a2", "json.gz", ("text",), license="odc-by"),
    SourceSpec("codeparrot_clean_valid", "code", "codeparrot/codeparrot-clean-valid", "file-000000000054.json.gz",
               "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd", "json.gz", ("content",),
               license="per-file GitHub license, filtered to: " + ", ".join(PERMISSIVE_CODE_LICENSES),
               max_docs=20000, record_filter=(("license", PERMISSIVE_CODE_LICENSES),)),
    SourceSpec("gsm8k_train", "math", "openai/gsm8k", "main/train-00000-of-00001.parquet",
               "740312add88f781978c0658806c59bc2815b9866", "parquet", ("question", "answer"),
               template="qa", license="mit"),
    SourceSpec("dolly15k", "instruction", "databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl",
               "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", "jsonl", ("instruction", "context", "response"),
               template="chat", license="cc-by-sa-3.0"),
    SourceSpec("pubmedqa_unlabeled", "ood", "qiaojin/PubMedQA", "pqa_unlabeled/train-00000-of-00001.parquet",
               "9001f2853fb87cab8d220904e0de81ac6973b318", "parquet", ("context.contexts", "long_answer"),
               license="mit"),
)}


_MATH_REV = "21a5633873b6a120296cce3e2df9d5550074f4a3"
MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                 "prealgebra", "precalculus")
MATH_EXCLUSIONS = "math_competition_eval.json"

# Domains outside the v1 corpus: loaded only when the calibration mix names them (sources_for).
EXTRA_SOURCES: dict[str, SourceSpec] = {s.name: s for s in (
    ExcludingSourceSpec(f"math_train_{sub}", "math_competition", "EleutherAI/hendrycks_math",
                        f"{sub}/train-00000-of-00001.parquet", _MATH_REV, "parquet", ("problem", "solution"),
                        template="chat", license="mit", exclude_field="problem", exclude_file=MATH_EXCLUSIONS)
    for sub in MATH_SUBJECTS
)}


def sources_for(domains: Iterable[str]) -> dict[str, SourceSpec]:
    """The sources a calibration mix over `domains` loads: the whole v1 corpus (SOURCES, incl. the OOD
    source) plus the EXTRA_SOURCES of the named domains. For a v1 mix this is exactly SOURCES."""
    ds = set(domains)
    return {**SOURCES, **{k: s for k, s in EXTRA_SOURCES.items() if s.domain in ds}}


def corpus_identity(domains: Iterable[str]) -> str:
    """sha256 of the corpus definition of a mix: repr of every spec of sources_for(domains) (for a v1 mix
    the same value as before EXTRA_SOURCES existed) plus, for an ExcludingSourceSpec, its exclusion file."""
    srcs = sources_for(domains)
    blob = {k: repr(v) for k, v in sorted(srcs.items())}
    files = sorted({s.exclude_file for s in srcs.values() if isinstance(s, ExcludingSourceSpec)})
    if files:
        blob = {"sources": blob, "exclusions": {f: hashlib.sha256((EXCLUSIONS_DIR / f).read_bytes()).hexdigest()
                                                for f in files}}
    return hashlib.sha256(json.dumps(blob).encode()).hexdigest()


def problem_key(text: str) -> str:
    """sha256 of a problem text after lowercasing and collapsing whitespace (the exclusion-list key)."""
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def load_exclusions(file: str) -> frozenset[str]:
    return frozenset(json.loads((EXCLUSIONS_DIR / file).read_text())["hashes"])


def _get(record: dict, field: str) -> str:
    v = record
    for k in field.split("."):
        v = v.get(k) if isinstance(v, dict) else None
    if v is None:
        return ""
    if isinstance(v, list):
        return "\n".join(str(x) for x in v if x)
    return str(v)


def _raw_records(path: str, fmt: str, fields: tuple[str, ...]) -> Iterator[dict]:
    if fmt == "parquet":
        cols = sorted({f.split(".")[0] for f in fields})
        yield from pq.read_table(path, columns=cols).to_pylist()
        return
    opener = gzip.open if fmt.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_source(spec: SourceSpec, cache_dir: str | None = None) -> tuple[list[dict], str]:
    """Download the pinned file; return (records {text_field: str}, sha256 of the file).

    Records whose text fields are all empty are dropped. JSON(.gz) files are JSON Lines.
    """
    path = hf_hub_download(repo_id=spec.repo_id, repo_type="dataset", filename=spec.filename,
                           revision=spec.revision, cache_dir=cache_dir)
    fields = spec.text_fields + tuple(f for f, _ in spec.record_filter)
    excl, xf = frozenset(), ""
    if isinstance(spec, ExcludingSourceSpec) and spec.exclude_file:
        excl, xf = load_exclusions(spec.exclude_file), spec.exclude_field
        fields += (xf,)
    out: list[dict] = []
    for r in _raw_records(path, spec.format, fields):
        if any(_get(r, f) not in allowed for f, allowed in spec.record_filter):
            continue
        if excl and problem_key(_get(r, xf)) in excl:
            continue
        rec = {f: _get(r, f) for f in spec.text_fields}
        if not any(rec.values()):
            continue
        out.append(rec)
        if spec.max_docs is not None and len(out) >= spec.max_docs:
            break
    return out, _sha256(path)


def _model_encoder(tokenizer):
    """`encode_messages` from the checkpoint's `encoding/encoding_*.py`, or None. The tokenizer's
    `name_or_path` is the snapshot directory or a repo id resolved from the local HF cache."""
    import importlib.util
    from pathlib import Path
    root = Path(str(getattr(tokenizer, "name_or_path", "") or ""))
    if not (root / "encoding").is_dir():
        from huggingface_hub.constants import HF_HUB_CACHE   # repo id: newest cached snapshot with encoding/
        snaps = sorted(Path(HF_HUB_CACHE).glob(f"models--{str(root).replace('/', '--')}/snapshots/*/encoding"),
                       key=lambda d: d.stat().st_mtime)
        if not snaps:
            return None
        root = snaps[-1].parent
    files = sorted((root / "encoding").glob("encoding_*.py")) if (root / "encoding").is_dir() else []
    files = [f for f in files if not f.name.startswith("test_")]
    if not files:
        return None
    spec = importlib.util.spec_from_file_location(f"_suc_model_encoding_{files[0].stem}", files[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "encode_messages", None)


def render(spec: SourceSpec, record: dict, tokenizer=None) -> str:
    """Render one record into a document.

    plain: non-empty text fields joined by "\\n\\n".
    qa:    "Question: {f0}\\nAnswer: {f1}" (exactly two fields).
    chat:  user = non-empty fields[:-1] joined by "\\n\\n", assistant = fields[-1];
           `tokenizer.apply_chat_template([user, assistant], tokenize=False, enable_thinking=False)`
           (enable_thinking dropped if the template does not accept it). For Qwen3 this yields
           the non-thinking format with an empty think block in the assistant turn.
    """
    vals = [record.get(f) or "" for f in spec.text_fields]
    if spec.template == "plain":
        return "\n\n".join(v for v in vals if v)
    if spec.template == "qa":
        if len(vals) != 2:
            raise ValueError(f"qa template needs exactly 2 fields, got {spec.text_fields}")
        return f"Question: {vals[0]}\nAnswer: {vals[1]}"
    if spec.template == "chat":
        if tokenizer is None:
            raise ValueError("chat template needs a tokenizer with apply_chat_template")
        msgs = [{"role": "user", "content": "\n\n".join(v for v in vals[:-1] if v)},
                {"role": "assistant", "content": vals[-1]}]
        if not getattr(tokenizer, "chat_template", None):
            encode = _model_encoder(tokenizer)
            if encode is not None:     # e.g. DeepSeek-V4: no chat_template, encoding/encoding_dsv4.py instead
                return encode(msgs, thinking_mode="chat", add_default_bos_token=False)
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(msgs, tokenize=False)
    raise ValueError(f"unknown template {spec.template!r}")
