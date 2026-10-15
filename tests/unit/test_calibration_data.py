import gzip
import hashlib
import json

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from suc.data import calibration as cal
from suc.data import sources as src
from suc.data.sources import SourceSpec

EOS = 0


class FakeTok:
    """Whitespace tokenizer: each distinct word gets a stable id >= 1; eos = 0."""
    eos_token_id = EOS

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.special_flags: list[bool] = []

    def encode(self, text, add_special_tokens=True):
        self.special_flags.append(add_special_tokens)
        return [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.split()]


class FakeChatTok:
    def __init__(self):
        self.kw = None

    def apply_chat_template(self, messages, tokenize=True, **kw):
        self.kw = dict(kw, tokenize=tokenize)
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)


def _docs(domain: str, n: int, words: int = 12) -> list[str]:
    return [" ".join(f"{domain}{i}w{j}" for j in range(words)) for i in range(n)]


# ---------------------------------------------------------------- assignment

def test_assign_partition_deterministic_and_normalized():
    fr = {"calibration": 0.8, "validation": 0.2}
    a = cal.assign_partition("Hello   World\n", fr, "s")
    assert a in cal.PARTITIONS
    assert cal.assign_partition("hello world", fr, "s") == a
    assert all(cal.assign_partition("Hello World", fr, "s") == a for _ in range(5))


def test_assign_partition_fractions_roughly_respected():
    fr = {"calibration": 0.75, "validation": 0.25}
    got = [cal.assign_partition(f"doc {i}", fr, "salt") for i in range(4000)]
    frac = got.count("calibration") / len(got)
    assert 0.72 < frac < 0.78


def test_assign_partition_salt_changes_assignment():
    fr = {"calibration": 0.5, "validation": 0.5}
    a = [cal.assign_partition(f"doc {i}", fr, "a") for i in range(200)]
    b = [cal.assign_partition(f"doc {i}", fr, "b") for i in range(200)]
    assert a != b


# ---------------------------------------------------------------- build_partitions

def _build(docs, tok=None, seed=0, n=None, mix=None, seq_len=8):
    return cal.build_partitions(
        docs, tok or FakeTok(), seq_len=seq_len,
        n_sequences=n or {"calibration": 12, "validation": 4},
        mix=mix or {"natural": 0.5, "code": 0.5}, seed=seed)


def test_build_partitions_shapes_quotas_domains():
    docs = {"natural": _docs("nat", 200), "code": _docs("code", 200)}
    parts = _build(docs, n={"calibration": 10, "validation": 5}, mix={"natural": 0.7, "code": 0.3})
    assert set(parts) == {"calibration", "validation"}
    c, v = parts["calibration"], parts["validation"]
    assert c["tokens"].shape == (10, 8) and c["tokens"].dtype == mx.int32
    assert v["tokens"].shape == (5, 8)
    assert c["domains"].count("natural") == 7 and c["domains"].count("code") == 3
    # 0.7*5 = 3.5, 0.3*5 = 1.5 -> largest remainder keeps the total exact
    assert len(v["domains"]) == 5 and sorted(set(v["domains"])) == ["code", "natural"]
    assert set(c["n_docs"]) == {"natural", "code"} and all(k > 0 for k in c["n_docs"].values())


def test_build_partitions_no_document_in_two_partitions():
    tok = FakeTok()
    docs = {"natural": _docs("nat", 300), "code": _docs("code", 300)}
    parts = _build(docs, tok=tok, n={"calibration": 30, "validation": 10})
    fr = {"calibration": 0.75, "validation": 0.25}
    rows = {p: set(np.array(parts[p]["tokens"]).reshape(-1).tolist()) for p in parts}
    # every token (word) belongs to exactly one document, so disjoint docs -> disjoint token sets
    assert not (rows["calibration"] - {EOS}) & (rows["validation"] - {EOS})
    # and each token's document was assigned to the partition it appears in
    inv = {i: w for w, i in tok.vocab.items()}
    by_word = {w: d for dom in docs.values() for d in dom for w in d.split()}
    for p, toks in rows.items():
        for t in toks - {EOS}:
            assert cal.assign_partition(by_word[inv[t]], fr, "suc-calib-v1") == p
    assert len(rows["calibration"]) > 100 and len(rows["validation"]) > 30
    assert all(flag is False for flag in tok.special_flags)


def test_build_partitions_stable_under_corpus_reordering():
    nat = _docs("nat", 200)
    a = _build({"natural": nat, "code": _docs("code", 200)})
    b = _build({"natural": nat[::-1], "code": _docs("code", 200)[::-1]})
    for p in a:  # input order is irrelevant: documents are canonically ordered before seeded sampling
        assert a[p]["n_docs"] == b[p]["n_docs"]
        assert mx.array_equal(a[p]["tokens"], b[p]["tokens"]).item()


def test_build_partitions_deterministic_by_seed():
    docs = {"natural": _docs("nat", 200), "code": _docs("code", 200)}
    tok = FakeTok()
    a = _build(docs, tok=tok, seed=1)
    b = _build(docs, tok=tok, seed=1)
    c = _build(docs, tok=tok, seed=2)
    assert all(mx.array_equal(a[p]["tokens"], b[p]["tokens"]).item() for p in a)
    assert a["calibration"]["domains"] == b["calibration"]["domains"]
    assert not mx.array_equal(a["calibration"]["tokens"], c["calibration"]["tokens"]).item()


def test_build_partitions_eos_between_documents():
    docs = {"natural": _docs("nat", 100, words=5)}
    parts = _build(docs, n={"calibration": 6, "validation": 2}, mix={"natural": 1.0}, seq_len=12)
    t = np.array(parts["calibration"]["tokens"]).reshape(-1)
    # 5 words then eos: the stream is periodic with period 6 inside each window
    assert (t == EOS).sum() >= 6
    for r in np.array(parts["calibration"]["tokens"]):
        pos = np.flatnonzero(r == EOS)
        assert np.all(np.diff(pos) == 6)


def test_build_partitions_raises_when_quota_unfillable():
    docs = {"natural": _docs("nat", 3), "code": _docs("code", 200)}
    with pytest.raises(ValueError, match="natural.*available"):
        _build(docs, n={"calibration": 40, "validation": 10})


def test_build_partitions_rejects_bad_keys():
    with pytest.raises(ValueError):
        _build({"natural": _docs("n", 50)}, mix={"natural": 0.5, "code": 0.5})
    with pytest.raises(ValueError):
        _build({"natural": _docs("n", 50)}, n={"test": 4}, mix={"natural": 1.0})


def test_build_ood():
    out = cal.build_ood(_docs("bio", 100), FakeTok(), seq_len=8, n_sequences=6, seed=0)
    assert out["tokens"].shape == (6, 8)
    assert out["domains"] == ["ood"] * 6 and out["n_docs"]["ood"] > 0
    with pytest.raises(ValueError):
        cal.build_ood(_docs("bio", 2), FakeTok(), seq_len=8, n_sequences=6, seed=0)


# ---------------------------------------------------------------- overlap / hash / manifest

def test_ngram_overlap():
    a = mx.array(np.arange(40).reshape(2, 20), dtype=mx.int32)
    b = mx.array(np.arange(100, 140).reshape(2, 20), dtype=mx.int32)
    assert cal.ngram_overlap(a, b, n=4) == 0
    shared = np.arange(100, 140).reshape(2, 20)
    shared[1, 5:12] = np.arange(3, 10)  # 7 tokens from a's row 0 -> 4 shared 4-grams
    assert cal.ngram_overlap(a, mx.array(shared, dtype=mx.int32), n=4) == 4
    assert cal.ngram_overlap(a, a, n=4) == 2 * 17
    # n-grams never cross row boundaries: a's last of row 0 + first of row 1
    c = mx.array(np.array([[18, 19, 20, 21]]), dtype=mx.int32)
    assert cal.ngram_overlap(a, c, n=4) == 0
    assert cal.ngram_overlap(a, mx.arange(0, 10), n=4) == 7  # 1-D input is one row


def test_partitions_hash_changes_with_tokens():
    docs = {"natural": _docs("nat", 200), "code": _docs("code", 200)}
    parts = _build(docs)
    h = cal.partitions_hash(parts)
    assert h == cal.partitions_hash(dict(reversed(list(parts.items()))))
    t = np.array(parts["validation"]["tokens"])
    t[0, 0] += 1
    mod = {**parts, "validation": {**parts["validation"], "tokens": mx.array(t)}}
    assert cal.partitions_hash(mod) != h


def test_partitions_manifest_json():
    docs = {"natural": _docs("nat", 200), "code": _docs("code", 200)}
    tok = FakeTok()
    parts = _build(docs, tok=tok)
    parts["ood"] = cal.build_ood(_docs("bio", 50), tok, seq_len=8, n_sequences=3, seed=0)
    spec = SourceSpec("c4", "natural", "allenai/c4", "x.json.gz", "abc", "json.gz", ("text",), license="odc-by")
    m = cal.partitions_manifest(parts, {"c4": spec}, {"c4": "deadbeef"})
    json.dumps(m)
    assert m["seq_len"] == 8 and m["partitions_hash"] == cal.partitions_hash(parts)
    assert m["partitions"]["calibration"]["rows"] == 12
    assert m["partitions"]["calibration"]["rows_per_domain"] == {"code": 6, "natural": 6}
    assert m["sources"]["c4"]["sha256"] == "deadbeef" and m["sources"]["c4"]["license"] == "odc-by"
    assert set(m["ngram_overlap"]) == {"calibration|ood", "calibration|validation", "ood|validation"}
    assert m["ngram_overlap"]["calibration|validation"] == 0


# ---------------------------------------------------------------- sources

def test_render_templates():
    plain = SourceSpec("p", "natural", "r", "f", "rev", "jsonl", ("title", "text"))
    assert src.render(plain, {"title": "T", "text": "body"}) == "T\n\nbody"
    assert src.render(plain, {"title": "", "text": "body"}) == "body"
    qa = SourceSpec("q", "math", "r", "f", "rev", "parquet", ("question", "answer"), template="qa")
    assert src.render(qa, {"question": "1+1?", "answer": "2"}) == "Question: 1+1?\nAnswer: 2"
    chat = SourceSpec("c", "instruction", "r", "f", "rev", "jsonl", ("instruction", "context", "response"),
                      template="chat")
    tok = FakeChatTok()
    out = src.render(chat, {"instruction": "Do X", "context": "ctx", "response": "done"}, tok)
    assert out == "<user>Do X\n\nctx</user><assistant>done</assistant>"
    assert tok.kw == {"tokenize": False, "enable_thinking": False}
    assert src.render(chat, {"instruction": "Do X", "context": "", "response": "ok"}, tok) \
        == "<user>Do X</user><assistant>ok</assistant>"
    with pytest.raises(ValueError):
        src.render(chat, {"instruction": "a", "context": "", "response": "b"})


def test_render_chat_template_without_enable_thinking():
    class OldTok:
        def apply_chat_template(self, messages, tokenize=True):
            return "|".join(m["content"] for m in messages)
    chat = SourceSpec("c", "instruction", "r", "f", "rev", "jsonl", ("q", "a"), template="chat")
    assert src.render(chat, {"q": "hi", "a": "yo"}, OldTok()) == "hi|yo"


def _patch_download(monkeypatch, path):
    calls = {}

    def fake(**kw):
        calls.update(kw)
        return str(path)
    monkeypatch.setattr(src, "hf_hub_download", fake)
    return calls


def test_load_source_parquet(tmp_path, monkeypatch):
    path = tmp_path / "train.parquet"
    pq.write_table(pa.table({"question": ["q1", "q2", "q3"], "answer": ["a1", "a2", "a3"], "x": [1, 2, 3]}), path)
    calls = _patch_download(monkeypatch, path)
    spec = SourceSpec("g", "math", "openai/gsm8k", "main/train.parquet", "sha1", "parquet",
                      ("question", "answer"), template="qa", max_docs=2)
    recs, sha = src.load_source(spec, cache_dir="c")
    assert recs == [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert calls == {"repo_id": "openai/gsm8k", "repo_type": "dataset", "filename": "main/train.parquet",
                     "revision": "sha1", "cache_dir": "c"}


def test_load_source_parquet_nested_field(tmp_path, monkeypatch):
    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"context": [{"contexts": ["s1", "s2"]}], "long_answer": ["la"]}), path)
    _patch_download(monkeypatch, path)
    spec = SourceSpec("pm", "ood", "r", "f", "rev", "parquet", ("context.contexts", "long_answer"))
    recs, _ = src.load_source(spec)
    assert recs == [{"context.contexts": "s1\ns2", "long_answer": "la"}]


def test_load_source_jsonl_gz_with_filter(tmp_path, monkeypatch):
    rows = [{"content": "a", "license": "mit"}, {"content": "b", "license": "gpl-3.0"},
            {"content": "", "license": "mit"}, {"content": "c", "license": "apache-2.0"}]
    for fmt in ("jsonl.gz", "json.gz"):
        path = tmp_path / f"d.{fmt}"
        with gzip.open(path, "wt") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
        _patch_download(monkeypatch, path)
        spec = SourceSpec("cp", "code", "r", "f", "rev", fmt, ("content",),
                          record_filter=(("license", ("mit", "apache-2.0")),))
        recs, sha = src.load_source(spec)
        assert recs == [{"content": "a"}, {"content": "c"}]  # filtered, empty dropped
        assert sha == hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_source_plain_jsonl(tmp_path, monkeypatch):
    path = tmp_path / "d.jsonl"
    path.write_text('{"instruction": "i", "context": null, "response": "r"}\n\n')
    _patch_download(monkeypatch, path)
    spec = SourceSpec("d", "instruction", "r", "f", "rev", "jsonl", ("instruction", "context", "response"))
    recs, _ = src.load_source(spec)
    assert recs == [{"instruction": "i", "context": "", "response": "r"}]


def test_sources_registry_is_pinned():
    domains = {s.domain for s in src.SOURCES.values()}
    assert {"natural", "code", "math", "instruction", "ood"} <= domains
    for name, s in src.SOURCES.items():
        assert s.name == name and len(s.revision) == 40 and s.license
        assert "test" not in s.filename.lower()
