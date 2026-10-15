"""qwen4_exp from a SUC1 file: the Python/MLX runtime on the stored codes (block-by-block conversion from the packed
index words, affine embedding / LM head from codes, n-gram tables memory-mapped), `suc repack` (raw n-gram shards ->
row-wise affine, everything else byte for byte) and the memory estimate. Tiny random qwen4_exp only."""
import ctypes
import ctypes.util
import hashlib
import json
import mmap
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import yaml

from suc.analysis.sensitivity import set_weight
from suc.io.repack import repack_cli, repack_suc1, table_names
from suc.io.suc1 import SucModel, stored, write_suc1
from suc.kernels.suc_linear import SucLinear
from suc.kernels.suc_switch import SucSwitchLinear
from suc.models import mlx_qwen4_exp as q4
from suc.models.streaming import load_lazy
from suc.quant.row_affine import dequantize_rows, pack_rows, quantize_rows, unpack_rows
from suc.runtime.compressed_model import resident_weight_bytes
from suc.runtime.suc1_model import AffineEmbedding, AffineLinear, greedy, load_suc1_model
from tiny_models import write_tiny_qwen4_exp

XS = {"kind": "vq", "k": 16, "block_size": 2, "rotate_out": False, "kmeans_iters": 3, "residual_fraction": 0.02,
      "residual_structure": "columns", "codebook_scope": "expert_stack"}
XS_NAME = "vq_k16_g2_rot1s_xs+res0.02_activation_cols"


def _compress(tmp: Path, name: str, **extra) -> tuple[dict, Path]:
    """`suc compress` of a tiny qwen4_exp (seed 2) with the XS recipe; extra top-level config keys (ngram_codec)."""
    import suc.pipeline.compress as comp
    from test_calibrate_run import FakeTokenizer, _docs

    mx.random.seed(0)
    tokens = mx.random.randint(2, 128, (2048,), dtype=mx.int32)
    model_dir = write_tiny_qwen4_exp(tmp / "model", seed=2, dtype=mx.bfloat16)
    cfg = {"name": name, "seed": 7, "output_dir": str(tmp / "results"), "run_dir": str(tmp / "run"),
           "model": {"id": "tiny/qwen4_exp", "local_path": str(model_dir)},
           "calibration": {"seq_len": 32, "n_calibration": 16, "n_validation": 8, "n_ood": 4, "micro_batch": 4},
           "plan": {"mode": "uniform", "codec": XS_NAME, "allocation_unit": "expert_stack", "candidates": [XS],
                    "embedding_codec": {"kind": "affine", "bits": 4}},
           "eval": {"seq_len": 64, "max_tokens": 1024, "kl_sequences": 4, "chunk": 2},
           "finetune": None, "rotation_groups": True, "verify_tensors": 8, **extra}
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    docs, ood = _docs()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(comp, "load_wikitext2", lambda split: ("unused", "wikihash"))
        mp.setattr(comp, "tokenize", lambda tok, text: tokens)
        r = json.loads(comp.run_compress(p, tokenizer=FakeTokenizer(), docs=docs, ood_docs=ood).read_text())
    return r, model_dir


@pytest.fixture(scope="module")
def q4_file(tmp_path_factory):
    """(SUC1 path, checkpoint dir) of a tiny qwen4_exp compressed with `suc compress` (raw n-gram tables)."""
    r, model_dir = _compress(tmp_path_factory.mktemp("q4rt"), "c_q4rt")
    return Path(r["container"]["path"]), model_dir


def _reference(model_dir: Path, sm: SucModel):
    """The checkpoint model with every non-raw tensor of the file set to its reference decode (raw n-gram tables)."""
    ref, _ = load_lazy(model_dir)
    for n in sm.names():
        if sm.descriptor(n)["representation_type"] != "raw" and not q4.Model.is_table_tensor(n):
            set_weight(ref, n, sm.decode_tensor(n))
    return ref


def _kl(lp: mx.array, lq: mx.array) -> float:
    p = mx.softmax(lp.astype(mx.float32), axis=-1)
    q = mx.softmax(lq.astype(mx.float32), axis=-1)
    return (p * (mx.log(p + 1e-9) - mx.log(q + 1e-9))).sum(-1).mean().item()


# ------------------------------------------------------------------ runtime from codes

def test_packed_path_equals_the_repacking_path(q4_file):
    """encoded_packed (stored words, no unpack) builds the same SUC modules as the research-layout path."""
    path, _ = q4_file
    sm = SucModel.load(path)
    n = "model.layers.0.linear_attn.in_proj_qkv.weight"
    a, b = SucLinear.from_encoded(sm.encoded_packed(n).base()), SucLinear.from_encoded(sm.encoded(n).base())
    assert mx.array_equal(a.packed_indices, b.packed_indices) and mx.array_equal(a.codebooks, b.codebooks)
    names = [f"model.layers.2.mlp.experts.{e}.down_proj.weight" for e in range(8)]
    s1 = SucSwitchLinear.from_encodings([sm.encoded_packed(x) for x in names])
    s2 = SucSwitchLinear.from_encodings([sm.encoded(x) for x in names])
    for k, v in s2.parameters().items():
        assert mx.array_equal(s1.parameters()[k], v), k


def test_runtime_from_codes_matches_the_reference_decode(q4_file):
    path, model_dir = q4_file
    sm = SucModel.load(path)
    assert sm.descriptor("model.embed_tokens.weight")["representation_type"] == "affine"
    model, _, rep = load_suc1_model(path, group=False, affine_modules=True)
    assert isinstance(model.model.embed_tokens, AffineEmbedding)
    assert set(rep["affine_modules"]) <= {"model.embed_tokens.weight", "lm_head.weight"}
    if "lm_head.weight" in rep["affine_modules"]:
        assert isinstance(model.lm_head, AffineLinear)
    kinds = {c["kind"] for c in rep["converted"]}
    assert "SucSwitchLinear" in kinds and rep["tensors_converted"] == sum(
        sm.descriptor(n)["representation_type"] in ("vq", "additive") for n in sm.names())
    ref = _reference(model_dir, sm)
    x = mx.random.randint(2, 128, (2, 24))
    lr, lq = ref(x), model(x)
    assert _kl(lr, lq) < 2e-2
    dec, _, _ = load_suc1_model(path, group=False, affine_modules=False)   # decoded embedding / head
    assert mx.abs(dec(x).astype(mx.float32) - lq.astype(mx.float32)).max().item() < 0.1
    grouped, _, _ = load_suc1_model(path, group=True, fuse=True, affine_modules=True)
    assert mx.abs(grouped(x).astype(mx.float32) - lq.astype(mx.float32)).max().item() < 0.25
    # generation with the KV / recurrent / PLE caches = the full forward, token by token
    out = greedy(model, [5, 6, 7, 8], 6)
    full = list(map(int, [5, 6, 7, 8]))
    for t in out[:-1]:
        full.append(t)
    lf = model(mx.array([full]))
    assert [int(t) for t in mx.argmax(lf[0, 3:], axis=-1).tolist()] == out


def test_memory_estimate_and_tables_never_resident(q4_file):
    path, _ = q4_file
    sm = SucModel.load(path)
    import gc

    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    model, _, rep = load_suc1_model(path, group=False, affine_modules=True)
    resident = resident_weight_bytes(model)
    est = rep["memory_estimate"]
    tables = table_names(sm)
    assert est["tables_mmap"] == sum(sm.descriptor(n)["compressed_size"] for n in sm.names()
                                     if q4.Model.is_table_tensor(n)) > sum(sm.descriptor(n)["compressed_size"]
                                                                           for n in tables) > 0
    assert abs(est["total"] - resident) <= 0.05 * resident, (est, resident)
    assert mx.get_active_memory() - before < 1.5 * est["total"] + (8 << 20)
    for t in model._tables.by_layer.values():                   # the tables are views of the file mapping
        for s in t.shards:
            assert isinstance(s, np.ndarray) and not s.flags["OWNDATA"] and s.base is not None


# ------------------------------------------------------------------ row-affine tables + repack

def test_row_affine_pack_quantize_round_trip():
    rng = np.random.default_rng(0)
    for bits in (2, 4, 8):
        q = rng.integers(0, 1 << bits, (5, 160)).astype(np.uint8)
        assert np.array_equal(unpack_rows(pack_rows(q, bits), bits, 160), q)
    from suc.quant.packing import pack_indices
    q = rng.integers(0, 16, (3, 160)).astype(np.uint8)
    assert np.array_equal(pack_rows(q, 4), np.array(pack_indices(mx.array(q), 4)))   # = the SUC1 packed layout
    x = (rng.standard_normal((300, 160)) * 0.05).astype(np.float32)
    w, s, b = quantize_rows(x, 4)
    y = dequantize_rows(w, s, b, 4, 160)
    assert np.all(np.abs(y - x) <= 0.5 * s.astype(np.float32) + 1e-7)
    assert np.sqrt(((y - x) ** 2).mean() / (x ** 2).mean()) < 0.12
    with pytest.raises(ValueError):
        quantize_rows(np.full((1, 160), 1e6, np.float32) * np.arange(160), 4)        # f16 overflow


def test_repack_round_trip_and_runtime(q4_file, tmp_path):
    path, model_dir = q4_file
    src = SucModel.load(path)
    tables = table_names(src)
    assert tables == sorted(q4.ngram_shard_name(1, j) for j in range(2))
    out = tmp_path / "row4.suc1"
    line = repack_cli(str(path), str(out), output_dir=str(tmp_path / "results"))
    assert "2 tables to 4-bit" in line
    res = json.loads(Path(f"{out}.repack.json").read_text())
    assert res["metadata"]["git_commit"] is not None and res["repack"]["verify"]["checksum"]
    sm = SucModel.load(out, verify=True)
    assert sm.metadata["repack"][0]["ngram_tables"]["bits"] == 4
    for n in src.names():
        d0, d1 = src.descriptor(n), sm.descriptor(n)
        if n in tables:
            assert d1["representation_type"] == "affine" and d1["params"]["group_size"] == d0["shape"][1]
            x = src.decode_tensor(n)
            y = sm.decode_tensor(n)
            step = sm.payload(n, "scales").astype(np.float32)
            assert np.all(np.abs(np.array(y - x)) <= 0.5 * step + 1e-6)
            assert d1["compressed_size"] < 0.4 * d0["compressed_size"]     # 6 bpp at width 16 (4.2 at 160)
        else:
            for ra, rb in zip(d0["arrays"], d1["arrays"]):
                assert src._view_bytes(ra) == sm._view_bytes(rb), n
    # gather-dequant of the mmap rows == the reference decode of those rows
    model, _, _ = load_suc1_model(out, group=False, affine_modules=True)
    t = model._tables.by_layer[1]
    assert t.quantized
    ids = np.array([[0, 5, t.rows - 1], [17, 2031, 2032]])
    full = np.concatenate([np.array(sm.decode_tensor(n)) for n in tables])
    assert np.array_equal(t.gather_np(ids), full[ids])
    # the model with 4-bit tables vs raw tables: same codes elsewhere, small logit change
    raw, _, _ = load_suc1_model(path, group=False, affine_modules=True)
    x = mx.random.randint(2, 128, (2, 24))
    assert _kl(raw(x), model(x)) < 1e-2
    assert len(greedy(model, [5, 6, 7], 4)) == 4
    # repacking a file without raw tables copies it byte for byte (the Prepared writer reproduces the layout)
    again = tmp_path / "again.suc1"
    rec = repack_suc1(out, again)
    assert rec["tables"]["converted"] == 0
    assert hashlib.sha256(again.read_bytes()).hexdigest() == hashlib.sha256(out.read_bytes()).hexdigest()


def test_stored_prepared_rewrite_is_byte_identical(q4_file, tmp_path):
    path, _ = q4_file
    sm = SucModel.load(path)
    dst = tmp_path / "copy.suc1"
    write_suc1(dst, architecture=dict(sm.architecture), tokenizer_id=sm.tokenizer_id,
               tensors={n: stored(sm, n) for n in sm.names()}, metadata=sm.metadata)
    assert dst.read_bytes() == Path(path).read_bytes()


# ------------------------------------------------------------------ suc compress writing row-affine tables directly

def _record(sm: SucModel, n: str) -> tuple[dict, list[bytes]]:
    """Descriptor without absolute payload offsets + the payload bytes of every array."""
    d = json.loads(json.dumps(sm.descriptor(n)))
    d.pop("data_offset")
    for r in d["arrays"]:
        r.pop("offset")
    return d, [sm._view_bytes(r) for r in sm.descriptor(n)["arrays"]]


def test_compress_ngram_codec_row_affine_equals_repack(q4_file, tmp_path_factory, monkeypatch):
    """ngram_codec {kind: row_affine, bits: 4}: the container writes the n-gram shards directly as the record
    `suc repack` produces (byte-identical descriptor + payloads), quantizing one shard at a time from the
    checkpoint mmap in row chunks (D8); the evaluate stage runs the model with those stored 4-bit tables."""
    import suc.io.repack as rp

    raw_path, _ = q4_file
    computed, chunks, views, takes = [], [], [], []
    orig_q, orig_rows, orig_take = rp._TableQuantizer._quantize, rp._rows_f32, q4.RowAffineShard.take

    def spy_q(self, name):
        computed.append((name, self._res is None))         # the previous shard's result is released first
        return orig_q(self, name)

    def spy_rows(view, dtype, lo, hi):
        views.append(isinstance(view, np.memmap) or isinstance(view.base, (mmap.mmap, np.memmap)))
        chunks.append(hi - lo)
        return orig_rows(view, dtype, lo, hi)

    def spy_take(self, rows):
        takes.append(len(rows))
        return orig_take(self, rows)

    monkeypatch.setattr(rp._TableQuantizer, "_quantize", spy_q)
    monkeypatch.setattr(rp, "_rows_f32", spy_rows)
    monkeypatch.setattr(rp, "CHUNK_ROWS", 300)
    monkeypatch.setattr(q4.RowAffineShard, "take", spy_take)
    r, _ = _compress(tmp_path_factory.mktemp("q4row"), "c_q4row", ngram_codec={"kind": "row_affine", "bits": 4})
    path = Path(r["container"]["path"])
    tables = sorted(q4.ngram_shard_name(1, j) for j in range(2))
    # D8: each shard quantized exactly once, one after the other, from memory-mapped rows in chunks
    assert [n for n, _ in computed[:len(tables)]] == tables and all(free for _, free in computed)
    assert max(chunks) <= 300 and all(views)
    assert takes, "evaluate did not gather from the stored row-affine tables"
    # == compress raw + suc repack
    rep = tmp_path_factory.mktemp("q4rep") / "repacked.suc1"
    repack_suc1(raw_path, rep)
    a, b = SucModel.load(path, verify=True), SucModel.load(rep, verify=True)
    assert a.names() == b.names()
    for n in a.names():           # every tensor: same record (the files differ only in their metadata)
        assert _record(a, n) == _record(b, n), n
    assert all(a.descriptor(n)["representation_type"] == "affine" for n in tables)
    assert a.metadata["ngram_tables"] == {"codec": "row_affine", "bits": 4, "scale_dtype": "f16",
                                          "scheme": "row_minmax", "tensors": 2}
    assert r["container"]["tables"]["converted"] == 2 and r["container"]["tables"]["rel_rmse_max"] < 0.2
    # the evaluate stage's size counts the tables at their stored bits
    numel = sum(int(np.prod(a.descriptor(n)["shape"])) for n in a.names()
                if a.descriptor(n)["representation_type"] != "raw")
    assert r["size"]["encoded_params"] == numel
    # the runtime loads it, with the same tables as the repacked file
    model, _, _ = load_suc1_model(path, group=False, affine_modules=True)
    assert model._tables.by_layer[1].quantized
    ids = np.array([[0, 5, 2031]])
    other, _, _ = load_suc1_model(rep, group=False, affine_modules=True)
    assert np.array_equal(model._tables.by_layer[1].gather_np(ids), other._tables.by_layer[1].gather_np(ids))
    x = mx.random.randint(2, 128, (2, 24))
    assert mx.array_equal(model(x), other(x))
    assert len(greedy(model, [5, 6, 7], 3)) == 3


# ------------------------------------------------------------------ mmap rows are not made resident

def _resident_pages(a: np.ndarray) -> int:
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    page = mmap.PAGESIZE
    addr = a.ctypes.data
    start = addr - addr % page
    length = addr + a.nbytes - start
    n = -(-length // page)
    vec = (ctypes.c_char * n)()
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    if libc.mincore(ctypes.c_void_p(start), ctypes.c_size_t(length), vec) != 0:
        pytest.skip("mincore unavailable")
    return sum(1 for c in vec.raw if c & 1)


@pytest.mark.parametrize("quantized", [False, True])
def test_gather_touches_only_the_requested_rows(tmp_path, quantized):
    """A sparse table file of 0.6-5 GB (never written): gathering 64 rows brings in only the pages around those rows
    (read-ahead clusters included), a few pages per row and array, about 1 % of this table; the table is never made resident."""
    rows, dim = 8_000_000, 160 if quantized else 320    # sparse: no disk used
    f = tmp_path / "table.bin"
    words = dim * 4 // 32
    width = (words * 4 + 4) if quantized else dim * 2
    with open(f, "wb") as fh:
        fh.truncate(rows * width)
    if quantized:
        w = np.memmap(f, dtype=np.uint32, mode="r", shape=(rows, words))
        s = np.memmap(f, dtype=np.float16, mode="r", offset=rows * words * 4, shape=(rows, 1))
        b = np.memmap(f, dtype=np.float16, mode="r", offset=rows * words * 4 + rows * 2, shape=(rows, 1))
        shard = q4.RowAffineShard(w, s, b, bits=4, dim=dim)
        arrays = [w, s, b]
    else:
        shard = np.memmap(f, dtype=np.uint16, mode="r", shape=(rows, dim))
        arrays = [shard]
    t = q4.NgramTable([shard])
    before = sum(_resident_pages(a) for a in arrays)
    ids = np.random.default_rng(0).choice(rows, 64, replace=False)
    g = t.gather(ids.reshape(4, 16))
    mx.eval(g)
    assert g.shape == (4, 16, dim) and float(mx.abs(g.astype(mx.float32)).max().item()) == 0.0
    touched = sum(_resident_pages(a) for a in arrays) - before
    total = sum(-(-a.nbytes // mmap.PAGESIZE) for a in arrays)
    assert touched <= 4 * 64 * len(arrays) + 16 and touched < 0.02 * total, (touched, total)


def test_suc_generate_cli(q4_file, tmp_path):
    from suc.cli.main import main

    path, _ = q4_file
    out = tmp_path / "gen.json"
    main(["generate", str(path), "--tokens", "5,6,7,8", "--steps", "5", "--json", str(out)])
    rec = json.loads(out.read_text())
    assert len(rec["generated"]) == 5 and rec["prompt_tokens"] == 4 and rec["affine_modules"]
    m = rec["memory"]
    assert m["estimate"]["total"] > 0 and m["resident_weights"] > 0 and m["peak_mlx"] >= m["resident_weights"]
    model, _, _ = load_suc1_model(path, group=True, fuse=True, affine_modules=True)
    assert rec["generated"] == greedy(model, [5, 6, 7, 8], 5)
