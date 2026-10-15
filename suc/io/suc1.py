"""SUC1 container: writer, mmap reader and Python reference decoder (spec §16, §17; byte layout in docs/format.md).

File = 64-byte fixed header | canonical-JSON directory | 64-byte-aligned payload sections
(codebooks, indices, scales, residual, raw) | 64-byte footer with the sha256 of every preceding byte.
All integers and payloads are little-endian; every payload offset is an absolute u64 multiple of 64.
Indices are stored in the bit-packed execution layout of suc.quant.packing; identical codebooks are
stored once (content-hash dedup). Tensors are stored in canonical order: sorted by name (code-point order).

The reference decoder (SucModel.decode_tensor) is deliberately simple and independent of the codec
decode paths: its own unpack, numpy gather / sum / scale / scatter, and only the shared Hadamard
primitive from suc.quant.hadamard. It is the correctness oracle for codecs and kernels.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import mlx.core as mx
import numpy as np

from suc.quant.codecs import Encoded
from suc.quant.hadamard import rht_inverse, unrotate_weight
from suc.quant.packing import pack_indices
from suc.quant.residual import _ResidualEncoded

MAGIC, FOOTER_MAGIC = b"SUC1", b"SUCF"
FORMAT_VERSION = 1
ALIGNMENT = 64
HEADER_SIZE = FOOTER_SIZE = 64
CHECKSUM_SHA256 = 1
LAYOUT = "rows_u32le_bitstream"           # packed index layout (suc.quant.packing)
FLAG_CODEBOOKS, FLAG_RESIDUAL, FLAG_ROTATION = 1, 2, 4
_KNOWN_FLAGS = FLAG_CODEBOOKS | FLAG_RESIDUAL | FLAG_ROTATION
_HEADER = struct.Struct("<4sIQQQQQQQ")     # magic, version, flags, n_tensors, n_codebooks, dir off/size, payload off, footer off
_FOOTER = struct.Struct("<4sIQ32s16x")    # magic, checksum algorithm, footer offset, sha256, reserved
SECTIONS = ("codebooks", "indices", "scales", "residual", "raw")
_CODEC_KINDS = ("bf16", "affine", "scalar", "vq", "additive", "mxfp4", "tcq")
LAYOUT_MXFP4 = "mxfp4_u8_nibbles"         # native MXFP4: checkpoint bytes, two E2M1 codes per byte (low nibble first)
_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)

# dtype name -> (mlx dtype, little-endian numpy storage dtype); bf16 is stored as its u16 bit pattern
_DTYPES: dict[str, tuple[mx.Dtype, np.dtype]] = {
    "bf16": (mx.bfloat16, np.dtype("<u2")), "f16": (mx.float16, np.dtype("<f2")),
    "f32": (mx.float32, np.dtype("<f4")), "f64": (mx.float64, np.dtype("<f8")),
    "i8": (mx.int8, np.dtype("i1")), "i16": (mx.int16, np.dtype("<i2")), "i32": (mx.int32, np.dtype("<i4")),
    "i64": (mx.int64, np.dtype("<i8")), "u8": (mx.uint8, np.dtype("u1")), "u16": (mx.uint16, np.dtype("<u2")),
    "u32": (mx.uint32, np.dtype("<u4")), "u64": (mx.uint64, np.dtype("<u8")), "bool": (mx.bool_, np.dtype("?")),
}
_DTYPE_NAME = {mx_dt: name for name, (mx_dt, _) in _DTYPES.items()}


class SucFormatError(ValueError):
    pass


def _align(n: int) -> int:
    return -(-n // ALIGNMENT) * ALIGNMENT


def _canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _dtype_name(a: mx.array) -> str:
    try:
        return _DTYPE_NAME[a.dtype]
    except KeyError:
        raise ValueError(f"unsupported dtype {a.dtype}") from None


def _to_np(a: mx.array) -> np.ndarray:
    """Little-endian numpy copy of an mlx array (bf16 as its uint16 bit pattern)."""
    name = _dtype_name(a)
    src = a.view(mx.uint16) if name == "bf16" else a
    return np.ascontiguousarray(np.array(src).astype(_DTYPES[name][1], copy=False))


def _pack_bytes(idx: np.ndarray, bits: int) -> bytes:
    """(rows, n) indices -> bytes of the suc.quant.packing layout as little-endian u32 words."""
    rows, n = idx.shape
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= 1 << bits):
        raise ValueError(f"indices must lie in [0, 2**{bits})")
    if bits in (8, 16, 32) and n * bits % 32 == 0:   # byte-aligned rows: the packed words ARE the LE bytes
        return idx.astype({8: "u1", 16: "<u2", 32: "<u4"}[bits]).tobytes()
    if rows == 0 or n == 0:
        return b""
    return np.array(pack_indices(mx.array(idx.astype(np.uint32)), bits)).astype("<u4").tobytes()


def _unpack(words: np.ndarray, bits: int, n: int) -> np.ndarray:
    """Independent inverse of the packing layout: (rows, w) u32 words -> (rows, n) uint32."""
    rows = words.shape[0]
    w64 = np.concatenate([words.astype(np.uint64), np.zeros((rows, 1), np.uint64)], axis=1)
    p = np.arange(n, dtype=np.uint64) * np.uint64(bits)
    word, shift = (p >> np.uint64(5)).astype(np.int64), p & np.uint64(31)
    both = w64[:, word] | (w64[:, word + 1] << np.uint64(32))
    return ((both >> shift) & np.uint64((1 << bits) - 1)).astype(np.uint32)


# ------------------------------------------------------------------------------------- writer

@dataclass
class _Item:                       # one payload segment
    section: str
    nbytes: int
    produce: Callable[[], bytes]
    record: dict = field(default_factory=dict)


def _index_bits_for(key: str, params: dict) -> int | None:
    if key == "res_pos":
        return int(params["position_bits"])
    if key == "res_cols":
        return int(params["column_index_bits"])
    if key in ("indices", "base.indices"):
        return int((params["base"] if key.startswith("base.") else params)["index_bits"])
    if key in ("chunks", "base.chunks"):      # tcq: kv-bit step chunks; packed rows ARE the tail-biting streams
        return int((params["base"] if key.startswith("base.") else params)["kv"])
    return None


def _part_of(key: str) -> str:
    """Accounting part / payload section of an Encoded array key."""
    if key.startswith("res_"):
        return "residual"
    k = key.removeprefix("base.")
    return "indices" if k in ("indices", "codes", "chunks") else "raw" if k == "weight" else "scales"


def _array_item(key: str, a: mx.array, bits: int | None) -> _Item:
    dtype = _dtype_name(a)
    if bits is None:
        npa = _to_np(a)
        return _Item(_part_of(key), npa.nbytes, lambda npa=npa: npa.tobytes(),
                     {"key": key, "part": _part_of(key), "dtype": dtype, "shape": list(a.shape)})
    shape = list(a.shape)
    n = shape[-1] if shape else 1
    rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    words = -(-n * bits // 32)
    rec = {"key": key, "part": _part_of(key), "dtype": "u32", "shape": [rows, words],
           "packed": {"bits": bits, "logical_dtype": dtype, "logical_shape": shape}}
    return _Item(_part_of(key), rows * words * 4,
                 lambda: _pack_bytes(np.array(a).reshape(rows, n).astype(np.int64), bits), rec)


def _rotation_label(bp: dict) -> str:
    """Descriptor `rotation` label from the base codec params (format.md §5.2); decoding uses the params."""
    if not bp.get("rotate"):
        return "none"
    rin, rout = bp.get("rotate_in", True), bp.get("rotate_out", True)
    return {(True, True): "in_out", (True, False): "in", (False, True): "out"}.get((rin, rout), "none")


def _describe(name: str, t: Encoded | mx.array) -> tuple[dict, list[_Item], list[mx.array]]:
    """Descriptor skeleton, own payload items, and codebook slices (K, G) of one tensor."""
    if isinstance(t, mx.array):
        d = {"name": name, "shape": list(t.shape), "layout": "row_major", "representation_type": "raw",
             "codec": None, "params": {}, "block_size": None, "index_bits": None, "residual_type": "none",
             "rotation": "none", "codebook_key": None, "logical_size": int(t.size)}
        return d, [_array_item("weight", t, None)], []
    if not isinstance(t, Encoded):
        raise TypeError(f"{name}: expected Encoded or mx.array, got {type(t).__name__}")
    p = t.params
    residual = isinstance(t, _ResidualEncoded)
    bp = p["base"] if residual else p
    rep = p["base_codec"] if residual else t.codec
    if rep not in _CODEC_KINDS:
        raise ValueError(f"{name}: unknown codec {rep!r}")
    if rep == "tcq" and bp.get("code") not in ("1mad", "3inst", "pair3"):
        raise ValueError(f"{name}: tcq code {bp.get('code')!r} is not a SUC1 representation (compute codes only)")
    block = {"affine": bp.get("group_size"), "scalar": 1, "vq": bp.get("block_size"),
             "additive": bp.get("block_size"), "mxfp4": bp.get("group_size"), "tcq": bp.get("T")}.get(rep)
    rotation = _rotation_label(bp)
    d = {"name": name, "shape": list(t.shape), "layout": LAYOUT_MXFP4 if rep == "mxfp4" else LAYOUT,
         "representation_type": rep, "codec": t.codec,
         "params": p, "block_size": block, "index_bits": bp.get("index_bits"),
         "residual_type": (f"{'columns' if p.get('structure') == 'columns' else 'sparse'}_{p['value_dtype']}"
                           if residual else "none"), "rotation": rotation,
         "codebook_key": None, "logical_size": int(t.shape[0] * t.shape[1])}
    items, cbs = [], []
    for key in sorted(t.arrays):
        a = t.arrays[key]
        if key.removeprefix("base.") == "codebooks":
            d["codebook_key"] = key
            cbs = [a[m] for m in range(a.shape[0])]
        else:
            items.append(_array_item(key, a, _index_bits_for(key, p)))
    return d, items, cbs


@dataclass
class StoredCodebook:
    """A codebook already in storage form (dtype name, shape, little-endian bytes), e.g. read from a SUC1 file."""
    dtype: str
    shape: tuple
    data: bytes


def _codebook_entry(c) -> tuple[str, dict, bytes]:
    """(content key, directory fields, bytes) of a codebook given as an mx.array slice or a StoredCodebook. The key is
    the sha256 of [dtype, shape] + bytes (identical codebooks are stored once)."""
    if isinstance(c, StoredCodebook):
        dtype, shape, data = c.dtype, list(c.shape), c.data
    else:
        npc = _to_np(c)
        dtype, shape, data = _dtype_name(c), list(c.shape), npc.tobytes()
    key = hashlib.sha256(_canonical_json([dtype, shape]) + data).hexdigest()
    return key, {"entries": int(shape[0]), "vector_length": int(shape[1]) if len(shape) > 1 else 1,
                 "dtype": dtype}, data


@dataclass
class Prepared:
    """One tensor in storage form for write_suc1 / write_suc1_streamed: its descriptor (without offsets / sizes /
    codebook ids, which the writer assigns), its payload segments (section, nbytes, produce, record; produced lazily,
    in section order) and its codebooks. Lets the writers copy or re-encode tensors without an Encoded in memory."""
    descriptor: dict
    items: list
    codebooks: list = field(default_factory=list)

    def parts(self) -> tuple[dict, list, list]:
        d = {k: v for k, v in self.descriptor.items()
             if k not in ("data_offset", "arrays", "codebook_ids", "codebook_id", "compressed_size")}
        items = [_Item(it.section, it.nbytes, it.produce,
                       {k: v for k, v in it.record.items() if k not in ("offset", "nbytes")}) for it in self.items]
        return d, items, list(self.codebooks)


def stored(sm: "SucModel", name: str) -> Prepared:
    """Tensor `name` of an open SUC1 file as a Prepared whose payloads are copied byte for byte from its mmap: writing
    it reproduces the descriptor and bytes the original writer produced (repacking without decode / re-encode)."""
    d = sm.descriptor(name)
    items = [_Item(r["part"], int(r["nbytes"]), lambda r=r: sm._view_bytes(r), dict(r)) for r in d["arrays"]]
    cbs = []
    for cid in d.get("codebook_ids") or []:
        c = sm.codebooks()[cid]
        cbs.append(StoredCodebook(c["dtype"], (c["entries"], c["vector_length"]),
                                  bytes(sm._mm[c["offset"]:c["offset"] + c["nbytes"]])))
    return Prepared(dict(d), items, cbs)


def write_suc1(path: str | Path, *, architecture: dict, tokenizer_id: str,
               tensors: dict[str, Encoded | mx.array], metadata: dict | None = None) -> dict:
    """Write a SUC1 file; returns {"bytes", "sha256", "tensors", "codebooks"}. Output is a pure
    function of the content (no timestamps; canonical order and JSON), hence byte-deterministic.

    A tensor may also be a `Prepared` (descriptor + lazily produced payload segments, e.g. `stored(sm, name)` copying
    an existing file's bytes): only descriptors are held, each payload is produced when its section is written."""
    if "model_type" not in architecture:
        raise ValueError("architecture must contain 'model_type' (the architecture_id)")
    descs, items_by_section = [], {s: [] for s in SECTIONS}
    cb_items, cb_ids = [], {}
    flags = 0
    for name in sorted(tensors):
        t = tensors[name]
        d, items, cbs = t.parts() if isinstance(t, Prepared) else _describe(name, t)
        ids = []
        for c in cbs:
            key, fields, data = _codebook_entry(c)
            if key not in cb_ids:
                cb_ids[key] = len(cb_ids)
                cb_items.append(_Item("codebooks", len(data), lambda data=data: data,
                                      {"id": cb_ids[key], **fields, "sha256": key}))
            ids.append(cb_ids[key])
        d["codebook_ids"], d["codebook_id"] = ids, (ids[0] if ids else None)
        flags |= (FLAG_CODEBOOKS if ids else 0) | (FLAG_RESIDUAL if d["residual_type"] != "none" else 0) \
            | (FLAG_ROTATION if d["rotation"] != "none" else 0)
        d["arrays"] = [it.record for it in items]
        d["compressed_size"] = sum(it.nbytes for it in items)
        for it in items:
            items_by_section[it.section].append(it)
        descs.append(d)
    ordered = cb_items + [it for s in SECTIONS[1:] for it in items_by_section[s]]

    def directory(payload_off: int) -> bytes:
        off = payload_off
        for it in ordered:
            it.record["offset"], it.record["nbytes"] = off, it.nbytes
            off = _align(off + it.nbytes)
        for d in descs:
            d["data_offset"] = min(r["offset"] for r in d["arrays"])
        return _canonical_json({
            "format": "SUC1", "format_version": FORMAT_VERSION, "alignment": ALIGNMENT, "byte_order": "little",
            "architecture_id": str(architecture["model_type"]), "architecture": architecture,
            "tokenizer_id": tokenizer_id, "metadata": metadata or {}, "global_flags": flags,
            "sections": list(SECTIONS), "codebooks": [it.record for it in cb_items], "tensors": descs})

    payload_off = _align(HEADER_SIZE)
    while True:   # absolute offsets live in the directory: grow the reserved space until it fits
        dir_bytes = directory(payload_off)
        need = _align(HEADER_SIZE + len(dir_bytes))
        if need <= payload_off:
            break
        payload_off = need
    end = payload_off
    for it in ordered:
        end = _align(it.record["offset"] + it.nbytes)
    footer_off = end
    h = hashlib.sha256()
    with open(path, "wb") as f:
        def put(b: bytes) -> None:
            f.write(b)
            h.update(b)
        put(_HEADER.pack(MAGIC, FORMAT_VERSION, flags, len(descs), len(cb_items), HEADER_SIZE, len(dir_bytes),
                         payload_off, footer_off))
        put(dir_bytes)
        pos = HEADER_SIZE + len(dir_bytes)
        for it in ordered:
            put(b"\0" * (it.record["offset"] - pos))
            b = it.produce()
            if len(b) != it.nbytes:
                raise AssertionError(f"payload size mismatch for {it.record.get('key', 'codebook')}")
            put(b)
            pos = it.record["offset"] + it.nbytes
        put(b"\0" * (footer_off - pos))
        put(_FOOTER.pack(FOOTER_MAGIC, CHECKSUM_SHA256, footer_off, h.copy().digest()))
    return {"bytes": footer_off + FOOTER_SIZE, "sha256": h.hexdigest(), "tensors": len(descs),
            "codebooks": len(cb_items)}


def write_suc1_streamed(path: str | Path, *, architecture: dict, tokenizer_id: str, names: Iterable[str],
                        get: Callable[[str], Encoded | mx.array], metadata: dict | None = None,
                        work_dir: str | Path | None = None, chunk: int = 16 << 20) -> dict:
    """Byte-identical to write_suc1(path, tensors={n: get(n) for n in names}) with ONE tensor resident at a time
    (spec D8: the container of a model larger than RAM).

    Every payload offset lives in the directory, which precedes the payloads, and payloads are grouped by section
    across tensors (docs/format.md §2). Single pass in canonical order: each tensor is loaded once (get(name)), its
    segments are appended to one temporary file per section at section-relative aligned offsets; then the absolute
    offsets are fixed (payload_offset + section base + relative offset, the same arithmetic as write_suc1), the
    header and directory are written and the section files are copied behind them in a streamed, hashed copy.
    Transient disk: about the payload size in `work_dir` (default: a hidden directory next to `path`, removed on
    exit). Memory: the directory (descriptors of every tensor) plus one tensor."""
    if "model_type" not in architecture:
        raise ValueError("architecture must contain 'model_type' (the architecture_id)")
    path = Path(path)
    order = sorted(names)
    if len(set(order)) != len(order):
        raise ValueError("duplicate tensor names")
    work = Path(work_dir) if work_dir is not None else path.parent / f".{path.name}.sections"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    files = {s: open(work / f"{s}.bin", "w+b") for s in SECTIONS}
    pos = dict.fromkeys(SECTIONS, 0)
    placed: list[tuple[dict, str, int]] = []           # (record, section, section-relative offset)

    def place(section: str, rec: dict, data: bytes, nbytes: int) -> None:
        if len(data) != nbytes:
            raise AssertionError(f"payload size mismatch for {rec.get('key', 'codebook')}")
        rel = _align(pos[section])
        f = files[section]
        if rel > pos[section]:
            f.write(b"\0" * (rel - pos[section]))
        f.write(data)
        pos[section] = rel + nbytes
        rec["nbytes"] = nbytes
        placed.append((rec, section, rel))

    try:
        descs, cb_recs, cb_ids, flags = [], [], {}, 0
        for name in order:
            t = get(name)
            d, items, cbs = t.parts() if isinstance(t, Prepared) else _describe(name, t)
            ids = []
            for c in cbs:
                key, fields, data = _codebook_entry(c)
                if key not in cb_ids:
                    cb_ids[key] = len(cb_ids)
                    rec = {"id": cb_ids[key], **fields, "sha256": key}
                    cb_recs.append(rec)
                    place("codebooks", rec, data, len(data))
                ids.append(cb_ids[key])
            d["codebook_ids"], d["codebook_id"] = ids, (ids[0] if ids else None)
            flags |= (FLAG_CODEBOOKS if ids else 0) | (FLAG_RESIDUAL if d["residual_type"] != "none" else 0) \
                | (FLAG_ROTATION if d["rotation"] != "none" else 0)
            d["arrays"] = [it.record for it in items]
            d["compressed_size"] = sum(it.nbytes for it in items)
            for it in items:
                place(it.section, it.record, it.produce(), it.nbytes)
            descs.append(d)
            del items, cbs
        sizes = {s: _align(pos[s]) for s in SECTIONS}
        base, acc = {}, 0
        for s in SECTIONS:
            base[s], acc = acc, acc + sizes[s]

        def directory(payload_off: int) -> bytes:
            for rec, s, rel in placed:
                rec["offset"] = payload_off + base[s] + rel
            for d in descs:
                d["data_offset"] = min(r["offset"] for r in d["arrays"])
            return _canonical_json({
                "format": "SUC1", "format_version": FORMAT_VERSION, "alignment": ALIGNMENT, "byte_order": "little",
                "architecture_id": str(architecture["model_type"]), "architecture": architecture,
                "tokenizer_id": tokenizer_id, "metadata": metadata or {}, "global_flags": flags,
                "sections": list(SECTIONS), "codebooks": cb_recs, "tensors": descs})

        payload_off = _align(HEADER_SIZE)
        while True:
            dir_bytes = directory(payload_off)
            need = _align(HEADER_SIZE + len(dir_bytes))
            if need <= payload_off:
                break
            payload_off = need
        footer_off = payload_off + acc
        h = hashlib.sha256()
        with open(path, "wb") as out:
            def put(b: bytes) -> None:
                out.write(b)
                h.update(b)
            put(_HEADER.pack(MAGIC, FORMAT_VERSION, flags, len(descs), len(cb_recs), HEADER_SIZE, len(dir_bytes),
                             payload_off, footer_off))
            put(dir_bytes)
            put(b"\0" * (payload_off - HEADER_SIZE - len(dir_bytes)))
            for s in SECTIONS:
                f = files[s]
                f.flush()
                f.seek(0)
                left = pos[s]
                while left > 0:
                    b = f.read(min(chunk, left))
                    if not b:
                        raise IOError(f"short read of section {s} in {work}")
                    put(b)
                    left -= len(b)
                put(b"\0" * (sizes[s] - pos[s]))
            put(_FOOTER.pack(FOOTER_MAGIC, CHECKSUM_SHA256, footer_off, h.copy().digest()))
    finally:
        for f in files.values():
            f.close()
        shutil.rmtree(work, ignore_errors=True)
    return {"bytes": footer_off + FOOTER_SIZE, "sha256": h.hexdigest(), "tensors": len(descs),
            "codebooks": len(cb_recs)}


def _sha256_file(path: str | Path, end: int | None = None, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        left = end
        while left is None or left > 0:
            b = f.read(chunk if left is None else min(chunk, left))
            if not b:
                break
            h.update(b)
            left = None if left is None else left - len(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------- reader

def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _check_payload_records(path, directory: dict, p_off: int, f_off: int) -> None:
    """Directory consistency the C++ reader also enforces (audit observation "reader hardening"): known dtypes,
    nbytes == prod(shape) * itemsize, 64-byte aligned offsets and every payload inside [payload_offset,
    footer_offset), unique tensor names. Python ints do not wrap, so a malformed size cannot alias a valid one."""
    def seg(what: str, dtype: str, shape, off, nbytes) -> None:
        if dtype not in _DTYPES:
            raise SucFormatError(f"{path}: {what}: unknown dtype {dtype!r}")
        if not all(isinstance(x, int) and x >= 0 for x in (*shape, off, nbytes)):
            raise SucFormatError(f"{path}: {what}: shape / offset / nbytes must be non-negative integers")
        if nbytes != math.prod(shape) * np.dtype(_DTYPES[dtype][1]).itemsize:
            raise SucFormatError(f"{path}: {what}: nbytes != prod(shape) * itemsize")
        if off % ALIGNMENT or off < p_off or off + nbytes > f_off:
            raise SucFormatError(f"{path}: {what}: payload outside the payload region or misaligned")

    try:
        for i, c in enumerate(directory["codebooks"]):
            seg(f"codebook {i}", c["dtype"], [c["entries"], c["vector_length"]], c["offset"], c["nbytes"])
        names = set()
        for t in directory["tensors"]:
            if t["name"] in names:
                raise SucFormatError(f"{path}: duplicate tensor {t['name']!r}")
            names.add(t["name"])
            for a in t["arrays"]:
                seg(f"{t['name']}/{a['key']}", a["dtype"], a["shape"], a["offset"], a["nbytes"])
    except (KeyError, TypeError) as ex:
        raise SucFormatError(f"{path}: malformed directory record: {ex!r}") from None


class SucModel:
    """Read-only SUC1 file. Header and directory are parsed eagerly; payloads are zero-copy views
    of a read-only np.memmap, touched only when a tensor is requested."""

    def __init__(self, path: Path, header: tuple, directory: dict) -> None:
        self.path, self._header, self._dir = path, header, directory
        self._tensors = {d["name"]: d for d in directory["tensors"]}
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.architecture, self.architecture_id = directory["architecture"], directory["architecture_id"]
        self.tokenizer_id, self.metadata = directory["tokenizer_id"], directory["metadata"]
        self.global_flags = header[2]

    @classmethod
    def load(cls, path: str | Path, verify: bool = True) -> "SucModel":
        path = Path(path)
        size = path.stat().st_size
        with open(path, "rb") as f:
            raw = f.read(HEADER_SIZE)
            if len(raw) < HEADER_SIZE:
                raise SucFormatError(f"{path}: file too small for a SUC1 header")
            hdr = _HEADER.unpack(raw)
            magic, version, flags, n_t, n_c, d_off, d_size, p_off, f_off = hdr
            if magic != MAGIC:
                raise SucFormatError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")
            if version != FORMAT_VERSION:
                raise SucFormatError(f"{path}: unsupported format_version {version} (reader supports {FORMAT_VERSION})")
            if flags & ~_KNOWN_FLAGS:
                raise SucFormatError(f"{path}: unknown global_flags bits {flags & ~_KNOWN_FLAGS:#x}")
            if size != f_off + FOOTER_SIZE or f_off % ALIGNMENT or p_off % ALIGNMENT or p_off < d_off + d_size:
                raise SucFormatError(f"{path}: inconsistent layout (size {size}, footer at {f_off}); truncated?")
            f.seek(f_off)
            fmagic, algo, f_off2, digest = _FOOTER.unpack(f.read(FOOTER_SIZE))
            if fmagic != FOOTER_MAGIC or algo != CHECKSUM_SHA256 or f_off2 != f_off:
                raise SucFormatError(f"{path}: bad footer")
            if verify and _sha256_file(path, f_off) != digest.hex():
                raise SucFormatError(f"{path}: checksum mismatch")
            f.seek(d_off)
            try:
                directory = json.loads(f.read(d_size))
            except ValueError as ex:
                raise SucFormatError(f"{path}: corrupt directory: {ex}") from None
        if (directory.get("format_version"), len(directory["tensors"]), len(directory["codebooks"])) != (version, n_t, n_c):
            raise SucFormatError(f"{path}: directory disagrees with the fixed header")
        _check_payload_records(path, directory, p_off, f_off)
        return cls(path, hdr, directory)

    # -- directory -------------------------------------------------------------
    def names(self) -> list[str]:
        return [d["name"] for d in self._dir["tensors"]]

    def descriptor(self, name: str) -> dict:
        try:
            return self._tensors[name]
        except KeyError:
            raise KeyError(f"unknown tensor {name!r}") from None

    def codebooks(self) -> list[dict]:
        return self._dir["codebooks"]

    # -- payload access --------------------------------------------------------
    def _view(self, rec: dict) -> np.ndarray:
        dt = _DTYPES[rec["dtype"]][1]
        return self._mm[rec["offset"]:rec["offset"] + rec["nbytes"]].view(dt).reshape(rec["shape"])

    def _view_bytes(self, rec: dict) -> bytes:
        return self._mm[rec["offset"]:rec["offset"] + rec["nbytes"]].tobytes()

    def record(self, name: str, key: str) -> dict:
        """Directory record (dtype, shape, offset, nbytes, packed) of one stored payload."""
        for rec in self.descriptor(name)["arrays"]:
            if rec["key"] == key:
                return rec
        raise KeyError(f"{name}: no payload {key!r}")

    def payload(self, name: str, key: str) -> np.ndarray:
        """Zero-copy view of one stored payload (packed u32 words for indices)."""
        for rec in self.descriptor(name)["arrays"]:
            if rec["key"] == key:
                return self._view(rec)
        raise KeyError(f"{name}: no payload {key!r}")

    def _logical(self, rec: dict) -> np.ndarray:
        """Payload in its research-layout numpy form (indices unpacked, storage dtype otherwise)."""
        v = self._view(rec)
        if "packed" not in rec:
            return v
        pk = rec["packed"]
        shape = pk["logical_shape"]
        n = shape[-1] if shape else 1
        return _unpack(v, pk["bits"], n).reshape(shape).astype(_DTYPES[pk["logical_dtype"]][1])

    def _codebook(self, cid: int) -> np.ndarray:
        c = self._dir["codebooks"][cid]
        shape = [c["entries"], c["vector_length"]]
        return self._view({"dtype": c["dtype"], "offset": c["offset"], "nbytes": c["nbytes"], "shape": shape})

    def _arrays(self, d: dict) -> dict[str, np.ndarray]:
        out = {r["key"]: self._logical(r) for r in d["arrays"]}
        if d["codebook_key"]:
            out[d["codebook_key"]] = np.stack([self._codebook(i) for i in d["codebook_ids"]])
        return out

    def encoded(self, name: str) -> Encoded | mx.array:
        """The in-memory research-layout Encoded (or raw mx.array) exactly as it was written."""
        d = self.descriptor(name)
        arrays = {}
        for key, v in self._arrays(d).items():
            rec = next((r for r in d["arrays"] if r["key"] == key), None)
            dtype = (rec["packed"]["logical_dtype"] if "packed" in rec else rec["dtype"]) if rec else \
                self._dir["codebooks"][d["codebook_ids"][0]]["dtype"]
            a = mx.array(np.array(v))
            arrays[key] = a.view(mx.bfloat16) if dtype == "bf16" else a
        if d["representation_type"] == "raw":
            return arrays["weight"]
        cls = _ResidualEncoded if d["residual_type"] != "none" else Encoded
        return cls(d["codec"], tuple(d["shape"]), d["params"], arrays)

    def encoded_packed(self, name: str) -> Encoded:
        """Encoded of a vq / additive / tcq tensor (optionally with a residual) for the runtime: the index streams stay
        in the stored execution layout ("packed_indices" (M, out, words) uint32, copied once from the mmap, never
        unpacked; tcq: "packed_words" (out, in/T · NW), the kernel layout), every other array as in `encoded`.
        suc.kernels.suc_linear.SucLinear.from_encoded accepts it."""
        d = self.descriptor(name)
        if d["representation_type"] not in ("vq", "additive", "tcq"):
            raise ValueError(f"{name}: encoded_packed needs a vq / additive / tcq tensor, not "
                             f"{d['representation_type']!r}")
        residual = d["residual_type"] != "none"
        pre = "base." if residual else ""
        arrays = {}
        for rec in d["arrays"]:
            key = rec["key"]
            if key == pre + "chunks":
                bp = d["params"]["base"] if residual else d["params"]
                if rec["packed"]["bits"] != int(bp["kv"]) or rec["shape"][-1] * 32 != int(round(bp["T"] * bp["k"])):
                    raise SucFormatError(f"{name}: stored tcq streams do not match the codec params")
                arrays[pre + "packed_words"] = mx.array(np.array(self._view(rec))).reshape(d["shape"][0], -1)
                continue
            if key == pre + "indices":
                pk = rec["packed"]
                lshape = pk["logical_shape"]
                words = rec["shape"][-1]
                if pk["bits"] != (d["params"]["base"] if residual else d["params"])["index_bits"]:
                    raise SucFormatError(f"{name}: stored index bits differ from the codec params")
                arrays[pre + "packed_indices"] = mx.array(np.array(self._view(rec))).reshape(*lshape[:-1], words)
                continue
            v = self._logical(rec)
            dtype = rec["packed"]["logical_dtype"] if "packed" in rec else rec["dtype"]
            a = mx.array(np.array(v))
            arrays[key] = a.view(mx.bfloat16) if dtype == "bf16" else a
        if d["codebook_key"]:
            cbs = [self._codebook(i) for i in d["codebook_ids"]]
            a = mx.array(np.stack(cbs))
            dtype = self._dir["codebooks"][d["codebook_ids"][0]]["dtype"]
            arrays[d["codebook_key"]] = a.view(mx.bfloat16) if dtype == "bf16" else a
        cls = _ResidualEncoded if residual else Encoded
        return cls(d["codec"], tuple(d["shape"]), d["params"], arrays)

    # -- reference decoder -----------------------------------------------------
    def decode_tensor(self, name: str) -> mx.array:
        """Reference decode to float32 (spec §17): slow, simple, independent of the codec decoders."""
        d = self.descriptor(name)
        a = self._arrays(d)
        for r in d["arrays"]:          # bf16 scales / biases / codebooks are stored as bit patterns
            if r["dtype"] == "bf16" and "packed" not in r and r["key"].removeprefix("base.") != "weight":
                a[r["key"]] = _bf16_to_f32(a[r["key"]])
        if d["codebook_key"] and self._dir["codebooks"][d["codebook_ids"][0]]["dtype"] == "bf16":
            a[d["codebook_key"]] = _bf16_to_f32(a[d["codebook_key"]])
        if d["representation_type"] == "raw":
            w = a["weight"]
            return mx.array(_bf16_to_f32(w) if d["arrays"][0]["dtype"] == "bf16" else w.astype(np.float32))
        residual = d["residual_type"] != "none"
        pre = "base." if residual else ""
        base = {k[len(pre):]: v for k, v in a.items() if k.startswith(pre)}
        w = _decode_base(d["representation_type"], d["params"]["base"] if residual else d["params"],
                         tuple(d["shape"]), base)
        if residual and "res_cols" in a and a["res_cols"].size:
            cols = a["res_cols"].astype(np.int64)
            cv = a["res_colval"].astype(np.float32).reshape(cols.size, w.shape[0])
            if "res_colscale" in a:
                cv = cv * a["res_colscale"].astype(np.float32)[:, None]
            w = w.copy()
            w[:, cols] += cv.T
        if residual and d["params"]["n"] > 0:
            v = a["res_val"].astype(np.float32)
            if "res_scale" in a:
                v = v * a["res_scale"].astype(np.float32)
            flat = w.reshape(-1).copy()
            flat[a["res_pos"].astype(np.int64)] += v
            w = flat.reshape(w.shape)
        return mx.array(w)

    # -- accounting ------------------------------------------------------------
    def bits(self) -> dict:
        """Whole-file accounting in bits; the parts sum exactly to 8 * file size."""
        out = {"header": 8 * HEADER_SIZE, "directory": 8 * self._header[6], "footer": 8 * FOOTER_SIZE}
        out.update({s: 0 for s in SECTIONS})
        segs = [(HEADER_SIZE, self._header[6])]
        for c in self._dir["codebooks"]:
            out["codebooks"] += 8 * c["nbytes"]
            segs.append((c["offset"], c["nbytes"]))
        for d in self._dir["tensors"]:
            for r in d["arrays"]:
                out[r["part"]] += 8 * r["nbytes"]
                segs.append((r["offset"], r["nbytes"]))
        pad, pos = 0, HEADER_SIZE
        for off, n in sorted(segs):
            pad += off - pos
            pos = off + n
        out["padding"] = 8 * (pad + self._header[8] - pos)
        out["total"] = 8 * self.path.stat().st_size
        return out


def _decode_base(rep: str, p: dict, shape: tuple[int, int], a: dict[str, np.ndarray]) -> np.ndarray:
    out, inn = shape
    if rep == "bf16":
        return _bf16_to_f32(a["weight"])
    if rep == "mxfp4":
        b = a["codes"].astype(np.uint8)
        codes = np.stack([b & 0x0F, b >> 4], axis=-1).reshape(out, inn)
        scale = np.ldexp(np.float32(1.0), a["scales"].astype(np.int32) - 127).astype(np.float32)
        return (_E2M1[codes].reshape(out, inn // p["group_size"], p["group_size"]) * scale[..., None]).reshape(out, inn)
    if rep == "affine":
        gs = p["group_size"]
        q = a["indices"].astype(np.float32).reshape(out, inn // gs, gs)
        return (q * a["scales"].astype(np.float32)[..., None] + a["biases"].astype(np.float32)[..., None]).reshape(out, inn)
    s = a["row_scales"].astype(np.float32)
    if rep == "tcq":
        return _decode_tcq(p, shape, a["chunks"], s)
    cb = a["codebooks"].astype(np.float32)
    if rep == "scalar":
        grp = np.arange(out) // p["row_group"]
        return cb[grp[:, None], a["indices"].astype(np.int64), 0] * s[:, None]
    if rep in ("vq", "additive"):
        idx = a["indices"].astype(np.int64)                       # (M, out, in // G)
        acc = cb[0][idx[0]]
        for m in range(1, cb.shape[0]):
            acc = acc + cb[m][idx[m]]
        x = acc.reshape(out, inn) * s[:, None]
        if not p["rotate"]:
            return x
        # mirrors suc.quant.codecs._BlockCodebookBase._finish; the params (not the descriptor label, which files
        # written before the "out" label existed set to "in_out" for output-only rotation) decide
        rin, rout = p.get("rotate_in", True), p.get("rotate_out", True)
        if rin and rout:
            return np.array(unrotate_weight(mx.array(x), p["seed_in"], p["seed_out"]))
        if rin:
            return np.array(rht_inverse(mx.array(x), p["seed_in"]))
        if rout:
            return np.array(rht_inverse(mx.array(x).T, p["seed_out"]).T)
        raise SucFormatError("rotate=true without an input or output rotation")
    raise SucFormatError(f"unknown representation {rep!r}")


# tcq reference decode (docs/format.md §5.5): independent of suc.quant.trellis except for the shared Hadamard primitive
_TCQ_LCG = {"1mad": [(34038481, 76625530), (2654435761, 1013904223)],
            "3inst": [(89226354, 64248484), (2654435761, 1013904223)], "pair3": [(89226354, 64248484)]}


def _tcq_raw(code: str, V: int, s: np.ndarray) -> np.ndarray:
    """(n, V) float64 raw code values of states s (exact)."""
    s = s.astype(np.uint64)
    out = np.empty((s.size, V), np.float64)
    for j in range(V):
        a, b = _TCQ_LCG[code][0 if code == "pair3" else j]
        x = (s * np.uint64(a) + np.uint64(b)) & np.uint64(0xFFFFFFFF)
        if code == "1mad":
            out[:, j] = sum(((x >> np.uint64(8 * i)) & np.uint64(0xFF)) for i in range(4)).astype(np.float64)
        else:
            h = (((x & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)).astype(np.uint32)).view(np.float16)
            h = h.reshape(-1, 2).astype(np.float64)
            out[:, j] = h[:, 0] - h[:, 1] if (code == "pair3" and j == 1) else h[:, 0] + h[:, 1]
    return out


def _decode_tcq(p: dict, shape: tuple[int, int], chunks: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Ŵ = rotation⁻¹(s_r · gain · (raw(state) - code_mu) / code_sd). The state of step i of a sequence is the L-bit
    window of its circular little-endian stream ending at chunk i: s_i = Σ_t chunk_{i-t} << (L - kv - t·kv)."""
    out, inn = shape
    L, kv, V = int(p["L"]), int(p["kv"]), int(p["V"])
    c = chunks.astype(np.int64).reshape(-1, int(p["T"]) // V)
    st = np.zeros_like(c)
    t = 0
    while L - kv - t * kv > -kv:                  # every chunk overlapping the window
        sh = L - kv - t * kv
        ct = np.roll(c, t, axis=1)
        st |= (ct << sh) if sh >= 0 else (ct >> -sh)
        t += 1
    st &= (1 << L) - 1
    raw = _tcq_raw(p["code"], V, st.reshape(-1))
    mu, sd = np.array(p["code_mu"], np.float64), np.array(p["code_sd"], np.float64)
    vals = (((raw - mu) / sd).astype(np.float32) * np.float32(p["gain"])).reshape(out, inn)
    x = vals * s[:, None]
    if not p["rotate"]:
        return x
    rin, rout = p.get("rotate_in", True), p.get("rotate_out", False)
    if rin and rout:
        return np.array(unrotate_weight(mx.array(x), p["seed_in"], p["seed_out"]))
    if rin:
        return np.array(rht_inverse(mx.array(x), p["seed_in"]))
    return np.array(rht_inverse(mx.array(x).T, p["seed_out"]).T)
