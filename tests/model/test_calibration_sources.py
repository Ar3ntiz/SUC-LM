"""Real calibration corpus: all pinned SOURCES, Qwen3-0.6B tokenizer, overlap vs WikiText-2 test."""
import json
from pathlib import Path

import numpy as np
import pytest

from suc.data.calibration import build_ood, build_partitions, ngram_overlap, partitions_manifest
from suc.data.sources import SOURCES, load_source, render
from suc.evaluation.data import load_wikitext2, tokenize

MODEL_DIR = Path.home() / (".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
                           "c1899de289a04d12100db370d81485cdf75e47ca")
SEQ_LEN = 512
MIX = {"natural": 0.35, "code": 0.25, "math": 0.2, "instruction": 0.2}


@pytest.mark.model
def test_real_calibration_corpus():
    from mlx_lm.utils import load_tokenizer
    tok = load_tokenizer(MODEL_DIR)
    docs: dict[str, list[str]] = {}
    hashes = {}
    for name, spec in SOURCES.items():
        recs, hashes[name] = load_source(spec)
        docs.setdefault(spec.domain, []).extend(render(spec, r, tok) for r in recs)
    ood_docs = docs.pop("ood")
    parts = build_partitions(docs, tok, seq_len=SEQ_LEN, n_sequences={"calibration": 512, "validation": 128},
                             mix=MIX, seed=12345)
    parts["ood"] = build_ood(ood_docs, tok, seq_len=SEQ_LEN, n_sequences=64, seed=12345)

    assert parts["calibration"]["tokens"].shape == (512, SEQ_LEN)
    assert parts["validation"]["tokens"].shape == (128, SEQ_LEN)
    assert parts["ood"]["tokens"].shape == (64, SEQ_LEN)
    assert parts["calibration"]["domains"].count("natural") == 179  # 0.35*512 = 179.2

    # zero document overlap: a document's partition is a function of its text alone
    from suc.data.calibration import _doc_key, _partition_of
    fr = {"calibration": 0.8, "validation": 0.2}
    for d, ds in docs.items():
        keys = {_doc_key(x, "suc-calib-v1") for x in ds}
        cal_n = sum(_partition_of(k, fr) == "calibration" for k in keys)
        assert cal_n == parts["calibration"]["n_docs"][d]
        assert len(keys) - cal_n == parts["validation"]["n_docs"][d]

    wiki, wiki_sha = load_wikitext2("test")
    wiki_tok = tokenize(tok, wiki)
    manifest = partitions_manifest(parts, SOURCES, hashes)
    manifest["ngram_overlap_vs_wikitext2_test"] = {p: ngram_overlap(v["tokens"], wiki_tok, 32)
                                                   for p, v in parts.items()}
    manifest["wikitext2_test_sha256"] = wiki_sha
    for n in (8, 13):
        manifest[f"ngram{n}_overlap"] = {
            "calibration|validation": ngram_overlap(parts["calibration"]["tokens"], parts["validation"]["tokens"], n),
            "calibration|wikitext2_test": ngram_overlap(parts["calibration"]["tokens"], wiki_tok, n)}
    ids = np.array(parts["calibration"]["tokens"])
    manifest["calibration_eos_fraction"] = float((ids == tok.eos_token_id).mean())
    print(json.dumps(manifest, indent=1))
    assert manifest["ngram_overlap"]["calibration|validation"] < 50
