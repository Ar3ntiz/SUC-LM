"""scripts/long_context_eval.py on tiny deepseek_v4 / qwen4_exp models with the sparse indexer selection active: the
chunked block-sequential forward equals the whole-model forward, the indexer tap records real choices and its dense
mode equals an indexer that selects everything, needle scoring by teacher forcing equals greedy decoding, and the
script end to end (layer-streamed reference, a raw SUC1 file through the Python runtime, dense control)."""
import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from mlx.utils import tree_flatten

from suc.models.streaming import load_lazy
from tiny_models import write_tiny_deepseek_v4, write_tiny_qwen4_exp

_SPEC = importlib.util.spec_from_file_location("long_context_eval", Path(__file__).resolve().parents[2]
                                               / "scripts/long_context_eval.py")
lc = importlib.util.module_from_spec(_SPEC)
sys.modules["long_context_eval"] = lc          # dataclasses resolve annotations through sys.modules
_SPEC.loader.exec_module(lc)


class FakeTokenizer:
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [2 + (ord(ch) % 120) for ch in text]


@pytest.fixture(scope="module")
def ds4_dir(tmp_path_factory):
    return write_tiny_deepseek_v4(tmp_path_factory.mktemp("lc_ds4") / "model", seed=3)


@pytest.fixture(scope="module")
def q4_dir(tmp_path_factory):
    return write_tiny_qwen4_exp(tmp_path_factory.mktemp("lc_q4") / "model", seed=3, dtype=mx.float32)


def _f32(d):
    model, _ = load_lazy(d)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


def _chunked_logits(model, tokens: np.ndarray, chunk: int, tap=None, record=None):
    seq = lc.Seq("s", "needle", tokens.astype(np.int32), record if record is not None else np.zeros(0, np.int64))
    parts = {}
    lc.forward_sequences(lambda: model, [seq], lambda s, c0, lg: parts.__setitem__(c0, lg), chunk=chunk,
                         streamed=False, budget_bytes=1e12, tap=tap, log=lambda m: None)
    return mx.concatenate([parts[k] for k in sorted(parts)], axis=1)


@pytest.mark.parametrize("family", ["ds4", "q4"])
@pytest.mark.parametrize("chunk", [13, 32])
def test_chunked_block_sequential_equals_whole_model(family, chunk, ds4_dir, q4_dir):
    """Sparse regime (ds4: top-4 of up to 24 entries; q4: top-2 of up to 24 blocks) at 96 tokens."""
    model = _f32(ds4_dir if family == "ds4" else q4_dir)
    vocab = 300 if family == "ds4" else 128
    tokens = np.random.default_rng(0).integers(2, vocab, 96)
    whole = model(mx.array(tokens[None], dtype=mx.int32))
    tap = lc.IndexerTap()
    with tap:
        got = _chunked_logits(model, tokens, chunk, tap, record=np.arange(96))
    err = float(mx.abs(got - whole).max())
    assert err <= 1e-4 * float(mx.abs(whole).max()), err
    # real choices were recorded in every indexer layer, |S| = top-k
    layers = lc.indexer_layers(model)
    assert layers == ([1, 3] if family == "ds4" else [3])
    topk = 4 if family == "ds4" else 2
    for layer in layers:
        r = tap.records[("s", layer)]
        assert len(r["pos"]) > 40 and all(len(i) == topk for i in r["idx"])
        assert all(i.max() < nv for i, nv in zip(r["idx"], r["nvis"]))


def _set_topk(model, k):
    from suc.models import mlx_deepseek_v4 as ds4
    from suc.models import mlx_qwen4_exp as q4

    for m in model.modules():
        if isinstance(m, (ds4.Indexer, q4.Indexer)):
            m.topk = k


@pytest.mark.parametrize("family", ["ds4", "q4"])
def test_dense_mode_equals_an_indexer_selecting_everything(family, ds4_dir, q4_dir):
    model = _f32(ds4_dir if family == "ds4" else q4_dir)
    tokens = np.random.default_rng(1).integers(2, 128, 80)
    with lc.IndexerTap(dense=True) as tap:
        dense = _chunked_logits(model, tokens, 16, tap)
    assert tap.calls > 0 and not tap.records
    sparse = _chunked_logits(model, tokens, 16)
    _set_topk(model, 10 ** 9)
    everything = model(mx.array(tokens[None], dtype=mx.int32))
    assert float(mx.abs(dense - everything).max()) <= 1e-4 * float(mx.abs(everything).max())
    assert float(mx.abs(dense - sparse).max()) > 1e-3           # the indexer matters at this length


def test_needle_teacher_forcing_equals_greedy_decoding(ds4_dir, tmp_path):
    model = _f32(ds4_dir)
    prompt = np.random.default_rng(2).integers(2, 300, 60).tolist()
    cache = model.make_cache()
    y = mx.argmax(model(mx.array([prompt]), cache=cache)[0, -1])
    greedy = [int(y.item())]
    for _ in range(4):
        y = mx.argmax(model(y.reshape(1, 1), cache=cache)[0, -1])
        greedy.append(int(y.item()))
    wrong = greedy[:2] + [(greedy[2] + 1) % 300] + greedy[3:]
    seqs = []
    for i, ans in enumerate((greedy, wrong)):
        t = np.asarray(prompt + ans, np.int32)
        seqs.append(lc.Seq(f"n{i}", "needle", t, np.arange(len(prompt) - 1, len(t) - 1),
                           {"length": len(t), "depth": 0.5, "answer": ans, "answer_start": len(prompt),
                            "needle_span": [10, 20]}))
    data = {"docs": [], "needles": seqs}
    sc = lc.Scorer(data, 4, tmp_path / "ref", is_reference=True)
    lc.forward_sequences(lambda: model, seqs, sc, chunk=16, streamed=False, budget_bytes=1e12, log=lambda m: None)
    sc.finish(tmp_path, data)
    cases = json.loads((tmp_path / "needle.json").read_text())
    assert cases[0]["correct"] and cases[0]["token_accuracy"] == 1.0
    assert not cases[1]["correct"] and cases[1]["token_accuracy"] < 1.0   # the token after the error is still scored


def test_selection_agreement_and_needle_hits():
    ref = {"pos": np.array([10, 20, 30]), "nvis": np.array([8, 8, 8]), "ratio": 4,
           "idx": np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6]], np.int32)}
    test = {"pos": np.array([20, 30, 40]), "nvis": np.array([8, 8, 8]), "ratio": 4,
            "idx": np.array([[4, 5, 6, 7], [0, 1, 2, -1], [0, 1, 2, 3]], np.int32)}
    r = lc.selection_agreement({"s": {3: ref}}, {"s": {3: test}}, ["s"], (25, 64))
    assert r["layers"]["3"]["n"] == 2 and r["layers"]["3"]["mean"] == pytest.approx((1.0 + 0.5) / 2)
    assert r["layers"]["3"]["chance_mean"] == pytest.approx(0.5)
    assert r["by_position"]["0-25"]["mean"] == 1.0 and r["by_position"]["25-64"]["mean"] == 0.5
    seq = lc.Seq("s", "needle", np.zeros(64, np.int32), np.arange(3), {"needle_span": [9, 12]})   # units 2 (8-11)
    h = lc.needle_hits({"s": {3: ref}}, [seq])
    assert h["layers"]["3"]["hit_rate"] == pytest.approx(2 / 3)


def _raw_suc1(model_dir: Path, path: Path, noise: float = 0.02) -> Path:
    """A SUC1 file holding every parameter of the tiny deepseek_v4 raw, floating weights perturbed by `noise`."""
    from suc.io.suc1 import write_suc1

    model, _ = load_lazy(model_dir)
    rng = np.random.default_rng(5)

    def jitter(k, v):
        if not mx.issubdtype(v.dtype, mx.floating) or v.ndim < 2:
            return v
        e = mx.array(rng.normal(size=v.shape).astype(np.float32))
        return (v.astype(mx.float32) * (1 + noise * e)).astype(v.dtype)

    params = {k: jitter(k, v) for k, v in tree_flatten(model.parameters())}
    arch = json.loads((model_dir / "config.json").read_text())
    write_suc1(path, architecture=arch, tokenizer_id="tiny/deepseek_v4", tensors=params,
               metadata={"model_hash": "tiny"})
    return path


def test_script_end_to_end_deepseek(ds4_dir, tmp_path):
    rng = np.random.default_rng(0)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa. lambda mu nu xi\n".split(" ")
    texts = ["short"] + [" ".join(rng.choice(words, 400)) for _ in range(4)]
    src = tmp_path / "books.parquet"
    pq.write_table(pa.table({"text": texts}), src)
    suc1 = _raw_suc1(ds4_dir, tmp_path / "model.suc1")
    cfg = {"name": "lc_tiny_ds4", "seed": 3, "output_dir": str(tmp_path / "results"),
           "model": {"id": "tiny/deepseek_v4", "local_path": str(ds4_dir)},
           "source": {"local_path": str(src), "sha256": None},
           "lengths": [256, 512], "ppl_docs": 2, "kl_stride": 3, "sel_stride": 8,
           "needle": {"depths": [0.2, 0.8], "lengths": [256, 512]},
           "chunk": 64, "hidden_budget_gb": 1.0e-4, "work_dir": str(tmp_path / "work"),
           "systems": [{"name": "reference", "kind": "reference"},
                       {"name": "suc_python", "kind": "suc1_python", "file": str(suc1), "chunk": 48},
                       {"name": "reference_dense", "kind": "reference", "indexer": "dense"}]}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    out = lc.main(["--config", str(p)], tokenizer=FakeTokenizer())
    r = json.loads(out.read_text())
    for k in ("model", "model_hash", "git_commit", "dataset_hash", "seed", "runtime", "timestamp"):
        assert k in r["metadata"]
    s = r["systems"]
    ref, suc, dense = s["reference"], s["suc_python"], s["reference_dense"]
    assert ref["forward"]["groups"] > 1                               # sequences grouped by the hidden budget
    assert set(ref["ppl"]) == {"256", "512"} and ref["ppl"]["512"]["tokens"] == 1024
    assert ref["info"]["indexer_layers"] == [1, 3]
    assert "kl" not in ref and suc["kl"]["512"]["all"]["kl_mean"] > 0
    assert suc["kl"]["512"]["all"]["positions"] == 2 * len(range(0, 512, 3))
    assert dense["kl"]["512"]["all"]["kl_mean"] > 0                   # indexer off changes the reference
    ov = suc["selection_overlap"]["documents"]
    assert set(ov["layers"]) == {"1", "3"} and 0.0 <= ov["all_layers"]["mean"] <= 1.0
    assert ov["all_layers"]["n"] > 50
    assert "selection_overlap" not in dense and "needle_hit_rate" not in dense
    assert ref["needle"]["cases"] == 4 and set(ref["needle"]["by_length"]) == {"256", "512"}
    assert ref["needle_hit_rate"]["mean"] is not None
    assert not (tmp_path / "work" / "lc_tiny_ds4" / "ref_logits").exists()
    # rerun of one system: the reference is reused only while its logits exist -> recomputed, same numbers
    out2 = lc.main(["--config", str(p), "--systems", "suc_python"], tokenizer=FakeTokenizer())
    r2 = json.loads(out2.read_text())["systems"]
    assert r2["suc_python"]["kl"]["512"]["all"]["kl_mean"] == pytest.approx(suc["kl"]["512"]["all"]["kl_mean"])
    assert set(r2) == {"reference", "suc_python"}


def test_script_end_to_end_qwen4(q4_dir, tmp_path):
    """qwen4_exp: layer-streamed reference (n-gram tables memory-mapped) vs the dense control; no SUC1 here (the
    deepseek test covers the runtime path)."""
    rng = np.random.default_rng(1)
    words = "one two three four five six seven. eight nine\n".split(" ")
    src = tmp_path / "books.parquet"
    pq.write_table(pa.table({"text": [" ".join(rng.choice(words, 500)) for _ in range(3)]}), src)
    cfg = {"name": "lc_tiny_q4", "model": {"id": "tiny/qwen4_exp", "local_path": str(q4_dir)},
           "output_dir": str(tmp_path / "results"), "source": {"local_path": str(src), "sha256": None},
           "lengths": [256], "ppl_docs": 1, "kl_stride": 2, "sel_stride": 4, "needle": {"depths": [0.5]},
           "chunk": 40, "work_dir": str(tmp_path / "work"),
           "systems": [{"name": "reference", "kind": "reference"},
                       {"name": "dense", "kind": "reference", "indexer": "dense"}]}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    r = json.loads(lc.main(["--config", str(p)], tokenizer=FakeTokenizer()).read_text())["systems"]
    assert r["reference"]["info"]["indexer_layers"] == [3]
    assert r["dense"]["kl"]["256"]["all"]["kl_mean"] > 0
    assert r["reference"]["needle_hit_rate"]["layers"]["3"]["n"] > 0


def test_config_validation(tmp_path):
    base = {"name": "x", "model": {"id": "m"}, "systems": [{"name": "r", "kind": "reference"}]}
    assert lc.lc_config_from_dict(base).lengths == (8192, 16384, 32768, 65536)
    with pytest.raises(ValueError, match="first system"):
        lc.lc_config_from_dict({**base, "systems": [{"name": "s", "kind": "suc1_python", "file": "f"}]})
    with pytest.raises(ValueError, match="unknown key"):
        lc.lc_config_from_dict({**base, "lenghts": [1]})
    with pytest.raises(ValueError, match="file"):
        lc.lc_config_from_dict({**base, "systems": base["systems"] + [{"name": "s", "kind": "suc1_python"}]})
    for f in ("long_context_deepseek_v4_flash.yaml", "long_context_qwen3.8_flash_next.yaml"):
        cfg = lc.load_lc_config(Path(__file__).resolve().parents[2] / "experiments/configs" / f)
        assert cfg.systems[0].kind == "reference" and any(s.kind == "suc1_python" for s in cfg.systems)
