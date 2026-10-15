import hashlib

import mlx.core as mx
import pyarrow as pa
import pyarrow.parquet as pq

from suc.evaluation import data


def test_load_wikitext2_with_local_parquet(tmp_path, monkeypatch):
    rows = ["", " = Title = \n", "", " Some text here .\n", " More @-@ text .\n"]
    path = tmp_path / "test-00000-of-00001.parquet"
    pq.write_table(pa.table({"text": rows}), path)
    calls = {}

    def fake_download(**kw):
        calls.update(kw)
        return str(path)

    monkeypatch.setattr(data, "hf_hub_download", fake_download)
    text, sha = data.load_wikitext2("test", cache_dir=str(tmp_path / "c"))
    assert text == "\n\n".join(rows)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert calls["repo_id"] == "Salesforce/wikitext" and calls["repo_type"] == "dataset"
    assert calls["filename"] == "wikitext-2-raw-v1/test-00000-of-00001.parquet"
    assert calls["cache_dir"] == str(tmp_path / "c")


class FakeTokenizer:
    def __init__(self):
        self.kw = None

    def encode(self, text, add_special_tokens=True):
        self.kw = add_special_tokens
        return ([999] if add_special_tokens else []) + [ord(c) for c in text]


def test_tokenize_no_special_tokens():
    tok = FakeTokenizer()
    ids = data.tokenize(tok, "abc")
    assert tok.kw is False
    assert ids.dtype == mx.int32 and ids.ndim == 1
    assert ids.tolist() == [97, 98, 99]
