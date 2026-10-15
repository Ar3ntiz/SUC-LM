"""`suc repack`: rewrite a SUC1 file converting its raw lookup tables (qwen4_exp n-gram embedding shards) to row-wise
affine (suc.quant.row_affine), copying every other tensor byte for byte (no decode / re-encode).

Table tensor = a raw 2-D tensor the architecture's model class flags with `is_table_tensor` and whose name ends in
".weight" (the I64 hash buffers stay raw). The converted tensor is an ordinary SUC1 `affine` record (docs/format.md
§5): `indices` u8 (rows, n) packed at `bits`, `scales` / `biases` (rows, 1) f16 (or bf16), group_size = n = the row
width, params {"bits", "group_size", "index_bits", "scheme": "row_minmax"}. Its reference decode is the generic affine
decode, so readers need nothing new; suc.models.mlx_qwen4_exp.NgramTable gathers rows of it directly from the mmap.

D8: one table shard at a time, read from the source mmap in row chunks (a 2.5 M x 160 BF16 shard is 800 MB; peak
transient per chunk is a few hundred MB). The output is written by suc.io.suc1.write_suc1 with lazily produced
payloads (Prepared), straight to the destination: no temporary section files, so the disk needed is the output size.
Each shard is quantized twice (once for the `indices` section, once for `scales`, where its biases and scales are
consecutive and share one cached result); the error statistics come from the first pass.

The output is byte-identical to write_suc1 of {unchanged tensors as stored, tables as row-affine}; with no table to
convert it is byte-identical to the source.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

from suc.io.suc1 import (FOOTER_SIZE, LAYOUT, Prepared, SucModel, _FOOTER, _Item, stored, write_suc1)
from suc.quant.row_affine import ROW_BITS, bf16_to_f32, dequantize_rows, quantize_rows, words_per_row

_SOURCE_DTYPES = {"bf16", "f16", "f32"}


def table_names(sm: SucModel) -> list[str]:
    """Raw 2-D table tensors of the file (weights flagged by the model class's is_table_tensor)."""
    from suc.models.streaming import get_model_classes

    try:
        model_cls, _ = get_model_classes(dict(sm.architecture))
    except Exception:        # an architecture without a model class has no tables to convert
        return []
    is_table = getattr(model_cls, "is_table_tensor", None)
    if is_table is None:
        return []
    out = []
    for n in sm.names():
        d = sm.descriptor(n)
        if d["representation_type"] == "raw" and n.endswith(".weight") and is_table(n) and len(d["shape"]) == 2:
            if d["arrays"][0]["dtype"] not in _SOURCE_DTYPES:
                raise ValueError(f"{n}: table dtype {d['arrays'][0]['dtype']} cannot be quantized")
            out.append(n)
    return out


CHUNK_ROWS = 1 << 18          # rows per quantization chunk (160-wide f32: 168 MB transient)
_NP_DTYPES = {"BF16": (np.uint16, "bf16"), "F16": (np.float16, "f16"), "F32": (np.float32, "f32")}


def _rows_f32(view: np.ndarray, dtype: str, lo: int, hi: int) -> np.ndarray:
    v = np.asarray(view[lo:hi])
    return bf16_to_f32(v) if dtype == "bf16" else v.astype(np.float32)


def suc1_table_source(sm: SucModel):
    """Table source of a SUC1 file: name -> (memory-mapped raw payload view (rows, n), dtype)."""
    return lambda name: (sm.payload(name, "weight"), sm.descriptor(name)["arrays"][0]["dtype"])


def safetensors_table_source(store):
    """Table source of a SafetensorsStore (suc compress, SUC namespace): name -> (read-only np.memmap of the
    checkpoint tensor, dtype); BF16 as its uint16 bit pattern. Raises on a dtype that cannot be quantized."""
    def get(name: str):
        info = store.info(name)
        if info.dtype not in _NP_DTYPES or len(info.shape) != 2:
            raise ValueError(f"{name}: table {info.dtype} {list(info.shape)} cannot be quantized row-wise")
        np_dtype, dt = _NP_DTYPES[info.dtype]
        return np.memmap(info.shard, dtype=np_dtype, mode="r", offset=info.offset, shape=tuple(info.shape)), dt
    return get


class _TableQuantizer:
    """Quantizes one table shard on demand (chunked), caching the last shard's result and its error statistics.
    source(name) -> (memory-mapped (rows, n) view, dtype in bf16 / f16 / f32): suc1_table_source /
    safetensors_table_source. Only ONE shard's result is held: it is released before the next shard is read."""

    def __init__(self, source, bits: int, scale_dtype: str, chunk_rows: int | None = None) -> None:
        if bits not in ROW_BITS:
            raise ValueError(f"bits must be one of {ROW_BITS}")
        self.source, self.bits, self.scale_dtype = source, bits, scale_dtype
        self.chunk_rows = chunk_rows or CHUNK_ROWS
        self._name: str | None = None
        self._res: tuple | None = None
        self.stats: dict[str, dict] = {}

    def __call__(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if name == self._name:
            return self._res
        self._name, self._res = None, None
        res = self._quantize(name)
        self._name, self._res = name, res
        return res

    def _quantize(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        view, dtype = self.source(name)
        rows = view.shape[0]
        words, scales, biases = [], [], []
        se = ss = 0.0
        max_abs = 0.0
        t0 = time.perf_counter()
        for lo in range(0, rows, self.chunk_rows):
            x = _rows_f32(view, dtype, lo, min(rows, lo + self.chunk_rows))
            w, s, b = quantize_rows(x, self.bits, self.scale_dtype)
            if name not in self.stats:
                err = dequantize_rows(w, s, b, self.bits, x.shape[1], self.scale_dtype) - x
                se += float(np.square(err, dtype=np.float64).sum())
                ss += float(np.square(x, dtype=np.float64).sum())
                max_abs = max(max_abs, float(np.abs(err).max(initial=0.0)))
            words.append(w)
            scales.append(s)
            biases.append(b)
            del x
        res = (np.concatenate(words), np.concatenate(scales), np.concatenate(biases))
        if name not in self.stats:
            self.stats[name] = {"rows": int(rows), "rel_rmse": float(np.sqrt(se / max(ss, 1e-300))),
                                "max_abs_err": max_abs, "seconds": time.perf_counter() - t0}
        return res


def row_affine_prepared(name: str, shape, quant: _TableQuantizer) -> Prepared:
    """The row-affine SUC1 record of table `name` of `shape` (rows, n) (payloads produced lazily by `quant`)."""
    rows, n = (int(x) for x in shape)
    bits, sdt = quant.bits, quant.scale_dtype
    w = words_per_row(n, bits)
    params = {"bits": bits, "group_size": n, "index_bits": bits, "scheme": "row_minmax"}
    d = {"name": name, "shape": [rows, n], "layout": LAYOUT, "representation_type": "affine", "codec": "affine",
         "params": params, "block_size": n, "index_bits": bits, "residual_type": "none", "rotation": "none",
         "codebook_key": None, "logical_size": rows * n}
    sb = 2 * rows
    items = [
        _Item("scales", sb, lambda: quant(name)[2].tobytes(),
              {"key": "biases", "part": "scales", "dtype": sdt, "shape": [rows, 1]}),
        _Item("indices", rows * w * 4, lambda: quant(name)[0].astype("<u4").tobytes(),
              {"key": "indices", "part": "indices", "dtype": "u32", "shape": [rows, w],
               "packed": {"bits": bits, "logical_dtype": "u8", "logical_shape": [rows, n]}}),
        _Item("scales", sb, lambda: quant(name)[1].tobytes(),
              {"key": "scales", "part": "scales", "dtype": sdt, "shape": [rows, 1]}),
    ]
    return Prepared(d, items, [])


def row_affine_bytes(shape, bits: int) -> int:
    """Payload bytes of a row-affine record of `shape` (rows, n): packed codes + f16/bf16 scale and bias per row."""
    rows, n = (int(x) for x in shape)
    return rows * (words_per_row(n, bits) * 4 + 4)


def check_table_rows(source, read_out, tables, quant: _TableQuantizer, rows: int, seed: int) -> int:
    """Sampled rows of every table: the stored row-affine record (read_out(name) -> (indices, scales, biases)
    memory-mapped views, scale dtype) decodes within half a quantization step of the source rows. Returns the
    number of rows checked; raises on a violation."""
    rng = np.random.default_rng(seed)
    checked = 0
    for n in sorted(tables):
        view, dtype = source(n)
        n_rows, width = view.shape
        idx = np.sort(rng.choice(n_rows, min(rows, n_rows), replace=False))
        x = np.stack([_rows_f32(view, dtype, i, i + 1)[0] for i in idx])
        words, s_all, b_all, sdt = read_out(n)
        s = s_all[idx]
        y = dequantize_rows(words[idx], s, b_all[idx], quant.bits, width, sdt)
        step = (bf16_to_f32(s) if sdt == "bf16" else s.astype(np.float32)).reshape(-1, 1)
        excess = np.abs(y - x) - (0.5 * step + 1e-6 * np.abs(x) + 1e-12)
        if float(excess.max()) > 0:
            raise RuntimeError(f"{n}: decoded rows exceed half a quantization step")
        checked += len(idx)
    return checked


def suc1_table_reader(sm: SucModel):
    """read_out of check_table_rows for a SUC1 file."""
    return lambda n: (sm.payload(n, "indices"), sm.payload(n, "scales"), sm.payload(n, "biases"),
                      sm.record(n, "scales")["dtype"])


def footer_sha256(path: str | Path) -> str:
    """The sha256 recorded in a SUC1 footer (of every preceding byte), without rehashing the file."""
    size = Path(path).stat().st_size
    with open(path, "rb") as f:
        f.seek(size - FOOTER_SIZE)
        return _FOOTER.unpack(f.read(FOOTER_SIZE))[3].hex()


def repack_suc1(src: str | Path, dst: str | Path, *, bits: int = 4, scale_dtype: str = "f16",
                chunk_rows: int | None = None, verify: bool = True, verify_rows: int = 64, seed: int = 0) -> dict:
    """Rewrite `src` as `dst` with every raw table tensor converted to row-affine `bits` (module docstring).
    Returns a record {source, output, bytes, sha256, tables: {converted, bits, scale_dtype, bytes_before,
    bytes_after, rel_rmse_max, per_tensor}, verify}. `dst` is written atomically (temporary name + rename)."""
    if bits not in ROW_BITS:
        raise ValueError(f"bits must be one of {ROW_BITS}")
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("repack: destination must differ from the source")
    t0 = time.perf_counter()
    sm = SucModel.load(src, verify=False)
    tables = set(table_names(sm))
    quant = _TableQuantizer(suc1_table_source(sm), bits, scale_dtype, chunk_rows)
    tensors = {n: (row_affine_prepared(n, sm.descriptor(n)["shape"], quant) if n in tables else stored(sm, n))
               for n in sm.names()}
    src_sha = footer_sha256(src)
    meta = dict(sm.metadata)
    if tables:
        hist = list(meta.get("repack", []))
        hist.append({"source_sha256": src_sha, "ngram_tables": {"tensors": len(tables), "bits": bits,
                                                               "scale_dtype": scale_dtype, "scheme": "row_minmax"}})
        meta["repack"] = hist
    tmp = dst.with_name(f".{dst.name}.tmp")
    try:
        info = write_suc1(tmp, architecture=dict(sm.architecture), tokenizer_id=sm.tokenizer_id, tensors=tensors,
                          metadata=meta)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()
    before = sum(int(sm.descriptor(n)["compressed_size"]) for n in tables)
    out = SucModel.load(dst, verify=verify)
    after = sum(int(out.descriptor(n)["compressed_size"]) for n in tables)
    check = _verify(sm, out, tables, quant, verify_rows, seed) if verify else None
    stats = quant.stats
    return {"source": str(src.resolve()), "source_sha256": src_sha, "output": str(dst.resolve()),
            "bytes": info["bytes"], "source_bytes": src.stat().st_size, "sha256": info["sha256"],
            "seconds": time.perf_counter() - t0,
            "tables": {"converted": len(tables), "bits": bits, "scale_dtype": scale_dtype,
                       "bytes_before": before, "bytes_after": after,
                       "bpp_after": 8 * after / max(1, sum(int(np.prod(out.descriptor(n)["shape"])) for n in tables)),
                       "rel_rmse_max": max((s["rel_rmse"] for s in stats.values()), default=0.0),
                       "max_abs_err": max((s["max_abs_err"] for s in stats.values()), default=0.0),
                       "per_tensor": stats},
            "verify": check}


def _verify(src: SucModel, out: SucModel, tables: set[str], quant: _TableQuantizer, rows: int, seed: int) -> dict:
    """Checksum (SucModel.load verify) + every non-table payload byte-identical (hash compare per tensor, sampled to
    at most 256 tensors) + sampled table rows: decoded rows within half a quantization step of the source."""
    rng = np.random.default_rng(seed)
    if set(out.names()) != set(src.names()):
        raise RuntimeError("repack: tensor set changed")
    others = [n for n in src.names() if n not in tables]
    pick = [others[i] for i in sorted(rng.choice(len(others), min(256, len(others)), replace=False))] if others else []
    for n in pick:
        a, b = src.descriptor(n), out.descriptor(n)
        if [r["key"] for r in a["arrays"]] != [r["key"] for r in b["arrays"]]:
            raise RuntimeError(f"repack: {n} payload keys changed")
        for ra, rb in zip(a["arrays"], b["arrays"]):
            if hashlib.sha256(src._view_bytes(ra)).digest() != hashlib.sha256(out._view_bytes(rb)).digest():
                raise RuntimeError(f"repack: {n}/{ra['key']} not byte-identical")
    checked = check_table_rows(suc1_table_source(src), suc1_table_reader(out), tables, quant, rows, seed + 1)
    return {"checksum": True, "payload_identical_checked": len(pick), "table_rows_checked": checked}


def repack_cli(source: str, output: str, *, bits: int = 4, scale_dtype: str = "f16", verify: bool = True,
               output_dir: str | None = "experiments/results") -> str:
    """`suc repack`: run repack_suc1, write the result JSON (provenance metadata; suc.provenance.write_result) to
    output_dir and next to the output file (<output>.repack.json); returns the one-line report."""
    import json

    from suc.provenance import run_metadata, write_result

    rec = repack_suc1(source, output, bits=bits, scale_dtype=scale_dtype, verify=verify)
    meta = SucModel.load(output, verify=False).metadata
    cfg = {"source": rec["source"], "bits": bits, "scale_dtype": scale_dtype, "verify": verify}
    result = {"metadata": run_metadata(model=str(meta.get("model", "unknown")),
                                       model_revision=str(meta.get("model_revision", "unknown")),
                                       model_hash=str(meta.get("model_hash", "unknown")), dataset_hash="none", seed=0,
                                       config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()),
              "config": cfg, "repack": rec}
    Path(f"{output}.repack.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    line = repack_report(rec)
    if output_dir:
        line += f"\nresult: {write_result(result, output_dir, 'repack_' + Path(output).stem)}"
    return line


def repack_report(rec: dict) -> str:
    t = rec["tables"]
    return (f"repack: {rec['source']} ({rec['source_bytes'] / 2**30:.2f} GiB) -> {rec['output']} "
            f"({rec['bytes'] / 2**30:.2f} GiB) in {rec['seconds']:.0f} s; {t['converted']} tables to {t['bits']}-bit "
            f"row affine ({t['bytes_before'] / 2**30:.2f} -> {t['bytes_after'] / 2**30:.2f} GiB, {t['bpp_after']:.3f} "
            f"bpp), rel RMSE max {t['rel_rmse_max']:.4f}")


__all__ = ["CHUNK_ROWS", "table_names", "row_affine_prepared", "row_affine_bytes", "check_table_rows",
           "suc1_table_source", "suc1_table_reader", "safetensors_table_source", "repack_suc1", "repack_cli", "repack_report", "footer_sha256"]
