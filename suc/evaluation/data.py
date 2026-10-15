"""Evaluation text datasets.

WikiText-2 (raw, v1) is used ONLY as a final-test / comparability set (perplexity numbers comparable
with the literature). It must never be used for calibration, sensitivity estimation or any tuning
(spec §32, leakage controls).
"""
from __future__ import annotations

import hashlib

import mlx.core as mx
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

WIKITEXT_REPO = "Salesforce/wikitext"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_wikitext2(split: str = "test", cache_dir: str | None = None) -> tuple[str, str]:
    """Return (text, sha256 of the source parquet) for WikiText-2-raw-v1 `split`.

    Joining convention: `"\\n\\n".join(rows)` over the raw `text` column, rows kept verbatim (each
    non-empty row already ends with "\\n"; empty rows are kept). This matches the common
    HF-transformers / GPTQ perplexity convention, so numbers are comparable to published ones.

    FINAL-TEST SET: the "test" split must never be used for calibration or tuning (spec §32).
    """
    path = hf_hub_download(
        repo_id=WIKITEXT_REPO, repo_type="dataset",
        filename=f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", cache_dir=cache_dir,
    )
    rows = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    return "\n\n".join(rows), _sha256(path)


def tokenize(tokenizer, text: str) -> mx.array:
    """1-D int32 token ids, no special tokens (BOS/EOS) added."""
    return mx.array(tokenizer.encode(text, add_special_tokens=False), dtype=mx.int32)
