"""Tests for the SUC1 container and the Python reference decoder (spec §16, §17, §33)."""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from suc.io import suc1
from suc.io.suc1 import ALIGNMENT, FORMAT_VERSION, SucFormatError, SucModel, write_suc1
from suc.models.adapters import get_adapter
from suc.quant.codecs import (AdditiveCodebook, AffineRTN, BF16Codec, Encoded, ScalarCodebook,
                              VectorCodebook)
from suc.quant.packing import pack_indices
from suc.quant.residual import ResidualCodec
from tiny_models import tiny_qwen3

ARCH = {"model_type": "qwen3", "hidden_size": 64}
TOK = "tok-sha256:abc"


def _randn(shape, seed=0, scale=0.02) -> mx.array:
    return mx.array((scale * np.random.default_rng(seed).standard_normal(shape)).astype(np.float32))


def _np(x: mx.array) -> np.ndarray:
    return np.array(x.astype(mx.float32))


def _same_array(a: mx.array, b: mx.array) -> None:
    assert a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape)
    if a.dtype == mx.bfloat16:
        a, b = a.view(mx.uint16), b.view(mx.uint16)
    np.testing.assert_array_equal(np.array(a), np.array(b))


def _same_encoded(a: Encoded, b: Encoded) -> None:
    assert type(a) is type(b) and a.codec == b.codec and tuple(a.shape) == tuple(b.shape)
    assert a.params == b.params and sorted(a.arrays) == sorted(b.arrays)
    for k in a.arrays:
        _same_array(a.arrays[k], b.arrays[k])


CODECS = {
    "bf16": BF16Codec(),
    "affine4": AffineRTN(4, 64),
    "affine3": AffineRTN(3, 32),
    "scalar_tensor": ScalarCodebook(16, kmeans_iters=3),
    "scalar_rowgroup": ScalarCodebook(8, scope="row_group", row_group=16, kmeans_iters=3),
    "scalar_k512": ScalarCodebook(512, kmeans_iters=2),
    "vq_rot": VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=3),
    "vq_norot": VectorCodebook(32, block_size=4, rotate=False, kmeans_iters=3),
    "additive_rot": AdditiveCodebook(16, block_size=8, num_codebooks=3, rotate=True, refine_iters=1, kmeans_iters=3),
    "res_fp16_vq": ResidualCodec(VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=3), 0.02),
    "res_int8_additive": ResidualCodec(AdditiveCodebook(16, block_size=8, num_codebooks=2, kmeans_iters=3), 0.01, "int8"),
    "res_fp16_affine": ResidualCodec(AffineRTN(2, 64), 0.005),
    "res_zero": ResidualCodec(VectorCodebook(16, block_size=8, kmeans_iters=3), 0.0, "int8"),
}


def _write(tmp_path: Path, tensors: dict, name="m.suc", **kw) -> tuple[Path, dict]:
    p = tmp_path / name
    return p, write_suc1(p, architecture=ARCH, tokenizer_id=TOK, tensors=tensors, **kw)


@pytest.mark.parametrize("key", sorted(CODECS))
def test_round_trip_every_codec(tmp_path, key):
    codec = CODECS[key]
    w = _randn((64, 128), 1)
    e = codec.encode(w, seed=7)
    p, info = _write(tmp_path, {"layer.w": e})
    m = SucModel.load(p)
    assert m.names() == ["layer.w"]
    _same_encoded(m.encoded("layer.w"), e)
    assert m.encoded("layer.w").bits() == e.bits()
    ref = _np(codec.decode(e))
    np.testing.assert_array_equal(_np(m.decode_tensor("layer.w")), ref)  # bit-exact vs the codec decoder
    assert m.decode_tensor("layer.w").dtype == mx.float32
    assert info["bytes"] == p.stat().st_size and info["tensors"] == 1


def test_vq_rotate_out_false(tmp_path):
    # 72 output rows have no Hadamard -> rotate_out=False (only the input side is rotated)
    codec = VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=3)
    e = codec.encode(_randn((72, 64), 2), seed=1)
    assert e.params["rotate_out"] is False
    p, _ = _write(tmp_path, {"w": e})
    m = SucModel.load(p)
    _same_encoded(m.encoded("w"), e)
    np.testing.assert_array_equal(_np(m.decode_tensor("w")), _np(codec.decode(e)))
    assert m.descriptor("w")["rotation"] == "in"


@pytest.mark.parametrize("legacy_label", [False, True])
def test_vq_rotate_in_false(tmp_path, monkeypatch, legacy_label):
    """Output-side rotation only (rotate_in=False): label "out", decode == codec.decode bit-exactly (regression:
    the reference decoder used to call unrotate_weight with seed_in=None, i.e. random signs). Files written before
    the "out" label existed carry "in_out"; the params decide, so they still decode correctly."""
    if legacy_label:
        monkeypatch.setattr(suc1, "_rotation_label", lambda bp: "in_out" if bp.get("rotate") else "none")
    codec = VectorCodebook(16, block_size=8, rotate=True, rotate_in=False, kmeans_iters=2)
    e = codec.encode(_randn((64, 128), 2), seed=5)
    assert e.params["rotate_in"] is False and e.params["rotate_out"] is True and e.params["seed_in"] is None
    p, _ = _write(tmp_path, {"w": e})
    m = SucModel.load(p)
    assert m.descriptor("w")["rotation"] == ("in_out" if legacy_label else "out")
    ref = _np(codec.decode(e))
    for _ in range(2):  # deterministic (no random signs)
        np.testing.assert_array_equal(_np(m.decode_tensor("w")).view(np.uint32), ref.view(np.uint32))


def test_raw_tensors(tmp_path):
    raw = {"a.bf16": _randn((5, 7), 3).astype(mx.bfloat16), "b.f32": _randn((11,), 4),
           "c.f16": _randn((2, 3, 4), 5).astype(mx.float16), "d.i32": mx.arange(-5, 5, dtype=mx.int32),
           "e.u8": mx.array(np.arange(9, dtype=np.uint8)), "f.scalar": mx.array(1.5)}
    p, _ = _write(tmp_path, raw)
    m = SucModel.load(p)
    assert m.names() == sorted(raw)
    for k, v in raw.items():
        got = m.encoded(k)
        assert isinstance(got, mx.array)
        _same_array(got, v)
        np.testing.assert_array_equal(_np(m.decode_tensor(k)), _np(v))
        d = m.descriptor(k)
        assert d["representation_type"] == "raw" and d["shape"] == list(v.shape)
        assert d["logical_size"] == v.size and d["compressed_size"] == v.nbytes


def _mixed() -> dict:
    w = _randn((64, 128), 9)
    return {"z.vq": CODECS["vq_rot"].encode(w, seed=1), "a.raw": _randn((4, 4), 1).astype(mx.bfloat16),
            "m.res": CODECS["res_int8_additive"].encode(w, seed=2), "b.aff": CODECS["affine4"].encode(w, seed=3)}


def test_determinism_and_canonical_order(tmp_path):
    t = _mixed()
    p1, i1 = _write(tmp_path, t, "1.suc", metadata={"b": 1, "a": [1, 2]})
    p2, i2 = _write(tmp_path, dict(reversed(list(t.items()))), "2.suc", metadata={"a": [1, 2], "b": 1})
    assert p1.read_bytes() == p2.read_bytes() and i1 == i2
    assert i1["sha256"] == hashlib.sha256(p1.read_bytes()).hexdigest()
    m = SucModel.load(p1)
    assert m.names() == sorted(t)
    assert m.metadata == {"a": [1, 2], "b": 1}
    assert m.architecture == ARCH and m.tokenizer_id == TOK and m.architecture_id == "qwen3"
    # within a section, payloads follow the canonical (sorted-name) tensor order
    offs = [r["offset"] for n in m.names() for r in m.descriptor(n)["arrays"] if r["part"] == "indices"]
    assert len(offs) == 3 and offs == sorted(offs)


def test_checksum_detects_flipped_byte(tmp_path):
    p, _ = _write(tmp_path, _mixed())
    data = bytearray(p.read_bytes())
    off = SucModel.load(p).descriptor("z.vq")["data_offset"]
    data[off + 3] ^= 0x10
    p.write_bytes(bytes(data))
    with pytest.raises(SucFormatError, match="checksum"):
        SucModel.load(p)
    SucModel.load(p, verify=False)  # structure still parses


def test_footer_and_truncation(tmp_path):
    p, _ = _write(tmp_path, _mixed())
    data = p.read_bytes()
    p.write_bytes(data[:-1])
    with pytest.raises(SucFormatError):
        SucModel.load(p)


def test_magic_and_version_mismatch(tmp_path):
    p, _ = _write(tmp_path, _mixed())
    data = bytearray(p.read_bytes())
    bad = bytearray(data)
    bad[:4] = b"GGUF"
    p.write_bytes(bytes(bad))
    with pytest.raises(SucFormatError, match="magic"):
        SucModel.load(p, verify=False)
    bad = bytearray(data)
    bad[4:8] = struct.pack("<I", FORMAT_VERSION + 1)
    p.write_bytes(bytes(bad))
    with pytest.raises(SucFormatError, match="format_version"):
        SucModel.load(p, verify=False)


def test_fixed_header_little_endian(tmp_path):
    p, info = _write(tmp_path, _mixed())
    data = p.read_bytes()
    magic, ver, flags, nt, nc, doff, dsize, poff, foff = struct.unpack_from("<4sIQQQQQQQ", data, 0)
    assert magic == b"SUC1" and ver == FORMAT_VERSION and nt == 4 and nc == info["codebooks"]
    assert doff == 64 and poff % ALIGNMENT == 0 and foff % ALIGNMENT == 0 and poff >= doff + dsize
    assert flags & suc1.FLAG_CODEBOOKS and flags & suc1.FLAG_RESIDUAL and flags & suc1.FLAG_ROTATION
    fmagic, algo, foff2, digest = struct.unpack_from("<4sIQ32s", data, foff)
    assert fmagic == b"SUCF" and algo == 1 and foff2 == foff and len(data) == foff + suc1.FOOTER_SIZE
    assert digest == hashlib.sha256(data[:foff]).digest()
    # payload bytes are explicit little-endian: raw bf16 bits and packed u32 index words
    m = SucModel.load(p)
    a = m.descriptor("a.raw")["arrays"][0]
    raw = np.array(_mixed()["a.raw"].view(mx.uint16)).astype("<u2").tobytes()
    assert data[a["offset"]:a["offset"] + a["nbytes"]] == raw
    e = _mixed()["z.vq"]
    ia = next(r for r in m.descriptor("z.vq")["arrays"] if r["key"] == "indices")
    words = np.array(pack_indices(e.arrays["indices"].reshape(-1, e.arrays["indices"].shape[-1]), 4)).astype("<u4")
    assert data[ia["offset"]:ia["offset"] + ia["nbytes"]] == words.tobytes()


def test_alignment(tmp_path):
    p, _ = _write(tmp_path, _mixed())
    m = SucModel.load(p)
    for n in m.names():
        d = m.descriptor(n)
        assert d["data_offset"] % ALIGNMENT == 0
        for r in d["arrays"]:
            assert r["offset"] % ALIGNMENT == 0
    for c in m.codebooks():
        assert c["offset"] % ALIGNMENT == 0


def test_lazy_mmap_payloads(tmp_path):
    t = _mixed()
    p, _ = _write(tmp_path, t)
    m0 = SucModel.load(p)
    view = m0.payload("a.raw", "weight")
    assert isinstance(view, np.ndarray) and np.shares_memory(view, m0._mm)  # zero-copy view into the mmap
    # corrupt the z.vq payload: without verify, other tensors load and decode correctly (no full read)
    off = m0.descriptor("z.vq")["data_offset"]
    data = bytearray(p.read_bytes())
    data[off:off + 16] = b"\xff" * 16
    p.write_bytes(bytes(data))
    m = SucModel.load(p, verify=False)
    assert m.descriptor("z.vq")["shape"] == [64, 128]
    _same_encoded(m.encoded("b.aff"), t["b.aff"])
    _same_array(m.encoded("a.raw"), t["a.raw"])


def test_codebook_dedup(tmp_path):
    w = _randn((64, 128), 4)
    e = CODECS["vq_rot"].encode(w, seed=5)
    p1, i1 = _write(tmp_path, {"x": e}, "1.suc")
    p2, i2 = _write(tmp_path, {"x": e, "y": CODECS["vq_rot"].encode(w, seed=5)}, "2.suc")
    assert i1["codebooks"] == i2["codebooks"] == 1
    m = SucModel.load(p2)
    assert m.descriptor("x")["codebook_ids"] == m.descriptor("y")["codebook_ids"] == [0]
    assert m.descriptor("x")["codebook_id"] == 0
    _same_encoded(m.encoded("y"), e)
    b1, b2 = SucModel.load(p1).bits(), m.bits()
    assert b1["codebooks"] == b2["codebooks"] == 16 * 8 * 16


def test_additive_codebooks_in_directory(tmp_path):
    e = CODECS["additive_rot"].encode(_randn((64, 128), 4), seed=5)
    p, info = _write(tmp_path, {"x": e})
    m = SucModel.load(p)
    d = m.descriptor("x")
    assert info["codebooks"] == 3 and d["codebook_ids"] == [0, 1, 2]
    assert [(c["entries"], c["vector_length"], c["dtype"]) for c in m.codebooks()] == [(16, 8, "f16")] * 3
    assert d["representation_type"] == "additive" and d["block_size"] == 8 and d["index_bits"] == 4
    assert d["residual_type"] == "none" and d["logical_size"] == 64 * 128 and d["layout"] == suc1.LAYOUT


def test_bits_reconcile_with_file_size(tmp_path):
    t = _mixed()
    p, _ = _write(tmp_path, t, metadata={"note": "x" * 100})
    b = SucModel.load(p).bits()
    parts = ("header", "directory", "footer", "padding", "indices", "codebooks", "scales", "residual", "raw")
    assert set(parts) | {"total"} == set(b)
    assert sum(b[k] for k in parts) == b["total"] == p.stat().st_size * 8
    assert b["header"] == 512 and b["footer"] == 8 * suc1.FOOTER_SIZE
    assert b["raw"] == 16 * 16
    # scales: vq row scales (64 fp16) + additive row scales (64) + affine scales/biases (2 * 64 * 2)
    assert b["scales"] == 16 * (64 + 64 + 2 * 64 * 2) + 0
    # residual: packed positions + int8 values + fp16 scale, each in its own 64-aligned payload
    res = SucModel.load(p).descriptor("m.res")
    assert b["residual"] == 8 * sum(r["nbytes"] for r in res["arrays"] if r["part"] == "residual")


def test_descriptor_residual_fields(tmp_path):
    e = CODECS["res_fp16_vq"].encode(_randn((64, 128), 3), seed=1)
    p, _ = _write(tmp_path, {"w": e})
    d = SucModel.load(p).descriptor("w")
    assert d["representation_type"] == "vq" and d["residual_type"] == "sparse_fp16"
    assert d["codec"] == e.codec and d["index_bits"] == 4 and d["block_size"] == 8
    pos = next(r for r in d["arrays"] if r["key"] == "res_pos")
    assert pos["packed"]["bits"] == e.params["position_bits"] == 13
    assert d["compressed_size"] == sum(r["nbytes"] for r in d["arrays"])


def test_pack_fast_path_matches_packing():
    rng = np.random.default_rng(0)
    for bits, n in [(8, 16), (8, 3), (16, 6), (4, 32), (3, 11), (13, 40), (32, 2)]:
        idx = rng.integers(0, 2 ** bits, size=(3, n), dtype=np.uint64).astype(np.uint32)
        ref = np.array(pack_indices(mx.array(idx), bits)).astype("<u4").tobytes()
        assert suc1._pack_bytes(idx, bits) == ref
        np.testing.assert_array_equal(suc1._unpack(np.frombuffer(ref, "<u4").reshape(3, -1), bits, n), idx)


def test_write_errors(tmp_path):
    with pytest.raises(ValueError, match="model_type"):
        write_suc1(tmp_path / "x.suc", architecture={}, tokenizer_id=TOK, tensors={"a": mx.zeros(2)})
    with pytest.raises(TypeError):
        _write(tmp_path, {"a": np.zeros(2)})
    bad = Encoded("mystery", (2, 2), {}, {"weight": mx.zeros((2, 2))})
    with pytest.raises(ValueError, match="mystery"):
        _write(tmp_path, {"a": bad})


def test_tiny_qwen3_multi_tensor(tmp_path):
    model = tiny_qwen3(seed=1, tie=True)
    weights = dict(tree_flatten(model.parameters()))
    cfg = {"model_type": "qwen3", "num_hidden_layers": 2, "tie_word_embeddings": True}
    specs = {s.name: s for s in get_adapter(cfg).linears(weights) if s.eligible}
    assert len(specs) == 14
    vq = VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=3)
    add_res = ResidualCodec(AdditiveCodebook(16, block_size=8, num_codebooks=2, refine_iters=1, kmeans_iters=3), 0.005)
    codec_of, tensors = {}, {}
    for i, (name, w) in enumerate(sorted(weights.items())):
        spec = specs.get(name)
        if spec is None:
            tensors[name] = w
            continue
        codec_of[name] = add_res if spec.role == "mlp_down" else vq
        tensors[name] = codec_of[name].encode(w, seed=100 + i)
    p, info = _write(tmp_path, tensors, metadata={"source": "tiny"})
    assert info["tensors"] == len(weights)
    m = SucModel.load(p)
    assert m.names() == sorted(weights)
    for name, w in weights.items():
        if name in codec_of:
            ref = _np(codec_of[name].decode(tensors[name]))
            np.testing.assert_array_equal(_np(m.decode_tensor(name)), ref)
            _same_encoded(m.encoded(name), tensors[name])
        else:
            np.testing.assert_array_equal(_np(m.decode_tensor(name)), _np(w))
    b = m.bits()
    assert b["total"] == p.stat().st_size * 8


def _streamed(tmp_path: Path, tensors: dict, name="s.suc", **kw) -> tuple[Path, dict]:
    p = tmp_path / name
    loads = []

    def get(n):
        loads.append(n)
        return tensors[n]

    info = suc1.write_suc1_streamed(p, architecture=ARCH, tokenizer_id=TOK, names=list(reversed(list(tensors))),
                                    get=get, **kw)
    assert loads == sorted(tensors)                       # canonical order, each tensor loaded exactly once
    assert not list(tmp_path.glob(".*.sections"))         # section scratch files removed
    return p, info


@pytest.mark.parametrize("case", ["every_codec", "mixed", "raw", "dedup", "tiny_qwen3", "empty_payloads"])
def test_streamed_writer_is_byte_identical(tmp_path, case):
    """write_suc1_streamed (one tensor resident, D8) produces exactly the bytes of the in-memory write_suc1."""
    if case == "every_codec":
        w = _randn((64, 128), 1)
        tensors = {f"t.{k}": c.encode(w, seed=i) for i, (k, c) in enumerate(sorted(CODECS.items()))}
    elif case == "mixed":
        tensors = _mixed()
    elif case == "raw":
        tensors = {"a.bf16": _randn((5, 7), 3).astype(mx.bfloat16), "b.f32": _randn((11,), 4),
                   "f.scalar": mx.array(1.5), "d.i32": mx.arange(-5, 5, dtype=mx.int32)}
    elif case == "dedup":
        e = CODECS["vq_rot"].encode(_randn((64, 128), 4), seed=5)
        tensors = {"x": e, "y": CODECS["vq_rot"].encode(_randn((64, 128), 4), seed=5), "r": _randn((3,), 1)}
    elif case == "empty_payloads":
        tensors = {"z": mx.zeros((0,), dtype=mx.float32), "a": CODECS["res_zero"].encode(_randn((64, 128), 2), seed=1)}
    else:
        model = tiny_qwen3(seed=1, tie=True)
        weights = dict(tree_flatten(model.parameters()))
        vq = VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=3)
        tensors = {n: (vq.encode(w, seed=i) if w.ndim == 2 and "embed" not in n else w)
                   for i, (n, w) in enumerate(sorted(weights.items()))}
    meta = {"source": case, "n": [1, 2]}
    p1, i1 = _write(tmp_path, tensors, "ref.suc", metadata=meta)
    p2, i2 = _streamed(tmp_path, tensors, metadata=meta)
    assert i1 == i2
    assert p1.read_bytes() == p2.read_bytes()
    SucModel.load(p2, verify=True)


def test_streamed_writer_errors(tmp_path):
    with pytest.raises(ValueError, match="model_type"):
        suc1.write_suc1_streamed(tmp_path / "x", architecture={}, tokenizer_id=TOK, names=["a"], get=lambda n: None)
    with pytest.raises(ValueError, match="duplicate"):
        suc1.write_suc1_streamed(tmp_path / "x", architecture=ARCH, tokenizer_id=TOK, names=["a", "a"],
                                 get=lambda n: mx.zeros((2,)))
    with pytest.raises(TypeError):
        suc1.write_suc1_streamed(tmp_path / "x", architecture=ARCH, tokenizer_id=TOK, names=["a"],
                                 get=lambda n: np.zeros(2))
    assert not list(tmp_path.glob(".*.sections"))
