"""Header-only safetensors store: lazy, per-tensor loading (spec D8).

The store parses safetensors headers itself (8-byte little-endian u64 header
length followed by a JSON header) and loads a single tensor by reading only its
bytes, so no step requires the full model resident in memory.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

# safetensors dtype -> (numpy dtype used for the raw read, optional mlx view dtype)
_DTYPES: dict[str, tuple[np.dtype, mx.Dtype | None]] = {
    "BF16": (np.dtype("<u2"), mx.bfloat16),
    "F16": (np.dtype("<f2"), None),
    "F32": (np.dtype("<f4"), None),
    "F64": (np.dtype("<f8"), None),
    "I8": (np.dtype("i1"), None),
    "I16": (np.dtype("<i2"), None),
    "I32": (np.dtype("<i4"), None),
    "I64": (np.dtype("<i8"), None),
    "U8": (np.dtype("u1"), None),
    "U16": (np.dtype("<u2"), None),
    "U32": (np.dtype("<u4"), None),
    "U64": (np.dtype("<u8"), None),
    "BOOL": (np.dtype("?"), None),
    # FP8 checkpoints (DeepSeek-V4): raw bytes; decoded by suc.io.quantized_source (load of a QuantSource tensor)
    "F8_E4M3": (np.dtype("u1"), None),
    "F8_E8M0": (np.dtype("u1"), None),
}

_INDEX_NAME = "model.safetensors.index.json"
_SINGLE_NAME = "model.safetensors"


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str  # safetensors dtype string, e.g. "BF16", "F16", "F32", "I32", "U8"
    shape: tuple[int, ...]
    shard: Path
    offset: int  # absolute byte offset of the tensor data in the shard file
    nbytes: int


def read_header(path: Path) -> dict[str, TensorInfo]:
    """Parse one shard's header into TensorInfo records (ignores ``__metadata__``)."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    base = 8 + hlen
    out: dict[str, TensorInfo] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        out[name] = TensorInfo(name, meta["dtype"], tuple(meta["shape"]), path, base + start, end - start)
    return out


@dataclass(frozen=True)
class QuantSource:
    """A logical tensor stored quantized at source: raw weight + E8M0 scale tensors (suc.io.quantized_source)."""
    kind: str             # "fp8_block" | "mxfp4"
    weight: TensorInfo
    scale: TensorInfo


class SafetensorsStore:
    """Read-only view over the safetensors shards of a model directory.

    `quant` maps logical tensor names to their QuantSource (adapters of quantized-at-source checkpoints build it in
    source_store): info(name) then describes the LOGICAL tensor (dtype BF16, dequantized shape, nbytes = numel x 2,
    shard / offset of the raw weight) and load(name) returns the exact BF16 dequantization (one tensor at a time)."""

    def __init__(self, infos: dict[str, TensorInfo], quant: dict[str, QuantSource] | None = None) -> None:
        self._infos = infos
        self._quant = dict(quant or {})

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "SafetensorsStore":
        d = Path(model_dir)
        index = d / _INDEX_NAME
        if index.is_file():
            weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]
            shards = sorted({d / s for s in weight_map.values()})
        elif (d / _SINGLE_NAME).is_file():
            shards = [d / _SINGLE_NAME]
        else:
            shards = sorted(d.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no safetensors files in {d}")
        infos: dict[str, TensorInfo] = {}
        for shard in shards:
            for name, info in read_header(shard).items():
                if name in infos:
                    raise ValueError(f"duplicate tensor {name!r} in {infos[name].shard} and {shard}")
                infos[name] = info
        return cls(infos)

    def names(self) -> list[str]:
        return sorted(self._infos)

    def info(self, name: str) -> TensorInfo:
        try:
            return self._infos[name]
        except KeyError:
            raise KeyError(f"unknown tensor {name!r}") from None

    def quant_source(self, name: str) -> QuantSource | None:
        return self._quant.get(name)

    def quant_names(self) -> list[str]:
        """Logical tensors stored quantized at source (dequantized exactly on load)."""
        return sorted(self._quant)

    def check_quant_scales(self) -> dict:
        """Read every E8M0 scale tensor (never the weights) and verify the BF16 dequantization is exact
        (suc.io.quantized_source.SCALE_RANGE); NotLossless otherwise. {kind: {"tensors", "min", "max"}}."""
        from suc.io.quantized_source import check_scales

        out: dict[str, dict] = {}
        for name in sorted(self._quant):
            q = self._quant[name]
            lo, hi = check_scales(q.kind, self._read(q.scale), name)
            r = out.setdefault(q.kind, {"tensors": 0, "min": lo, "max": hi})
            r["tensors"] += 1
            r["min"], r["max"] = min(r["min"], lo), max(r["max"], hi)
        return out

    def load_source(self, name: str) -> tuple[str, mx.array, mx.array]:
        """(kind, raw weight, raw scale) of a quantized-at-source tensor exactly as stored in the checkpoint (no
        dequantization; e.g. mxfp4: int8 [out, in/2] packed E2M1 nibbles + uint8 E8M0 [out, in/32]). KeyError for
        tensors that are not quantized at source."""
        q = self._quant.get(name)
        if q is None:
            raise KeyError(f"{name!r} is not quantized at source")
        return q.kind, self._read(q.weight), self._read(q.scale)

    def load(self, name: str) -> mx.array:
        """Load one tensor in its original dtype, reading only its bytes (quantized-at-source tensors: exact BF16)."""
        q = self._quant.get(name)
        if q is not None:
            from suc.io.quantized_source import dequantize

            return dequantize(q.kind, self._read(q.weight), self._read(q.scale), name=name)
        return self._read(self.info(name))

    def _read(self, info: TensorInfo) -> mx.array:
        name = info.name
        if info.dtype not in _DTYPES:
            raise NotImplementedError(f"unsupported safetensors dtype {info.dtype}")
        np_dtype, view = _DTYPES[info.dtype]
        with open(info.shard, "rb") as f:
            f.seek(info.offset)
            raw = f.read(info.nbytes)
        if len(raw) != info.nbytes:
            raise IOError(f"short read for {name!r} in {info.shard}")
        arr = mx.array(np.frombuffer(raw, dtype=np_dtype).reshape(info.shape))
        return arr.view(view) if view is not None else arr

    def load_rows(self, name: str, start: int, stop: int) -> mx.array:
        """Rows [start, stop) of the first dimension of one tensor, reading only their bytes (row-major data)."""
        if name in self._quant:
            return self.load(name)[max(0, start): stop]
        info = self.info(name)
        np_dtype, view = _DTYPES[info.dtype]
        rows = info.shape[0]
        start, stop = max(0, start), min(stop, rows)
        row_bytes = info.nbytes // rows if rows else 0
        with open(info.shard, "rb") as f:
            f.seek(info.offset + start * row_bytes)
            raw = f.read((stop - start) * row_bytes)
        if len(raw) != (stop - start) * row_bytes:
            raise IOError(f"short read for {name!r} in {info.shard}")
        arr = mx.array(np.frombuffer(raw, dtype=np_dtype).reshape((stop - start, *info.shape[1:])))
        return arr.view(view) if view is not None else arr

    def shards(self) -> list[Path]:
        return sorted({i.shard for i in self._infos.values()})

    def total_bytes(self) -> int:
        return sum(i.nbytes for i in self._infos.values())

    def __contains__(self, name: object) -> bool:
        return name in self._infos

    def __len__(self) -> int:
        return len(self._infos)


_MX_TO_ST = {mx.bfloat16: "BF16", mx.float16: "F16", mx.float32: "F32", mx.int32: "I32", mx.uint32: "U32",
             mx.int64: "I64", mx.uint8: "U8", mx.int8: "I8", mx.uint16: "U16", mx.int16: "I16", mx.bool_: "BOOL"}


class RowStreamWriter:
    """Write ONE tensor of a known shape to a safetensors file in row chunks along the first dimension, without
    holding the whole tensor (e.g. reference logits N x T x V). The header is written up front (8-byte length +
    JSON padded with spaces to a multiple of 8, as the safetensors spec requires); ``close`` checks that every row
    was written. Readable by mx.load / safetensors and SafetensorsStore (load_rows)."""

    def __init__(self, path: str | Path, name: str, shape: tuple[int, ...], dtype: mx.Dtype) -> None:
        self.path, self.name, self.shape, self.dtype = Path(path), name, tuple(int(d) for d in shape), dtype
        np_dtype, _ = _DTYPES[_MX_TO_ST[dtype]]
        self._row_bytes = int(np.prod(self.shape[1:], dtype=np.int64)) * np_dtype.itemsize
        nbytes = self._row_bytes * self.shape[0]
        header = json.dumps({name: {"dtype": _MX_TO_ST[dtype], "shape": list(self.shape),
                                    "data_offsets": [0, nbytes]}}, separators=(",", ":")).encode()
        header += b" " * (-len(header) % 8)
        self._f = open(self.path, "wb")
        self._f.write(struct.pack("<Q", len(header)) + header)
        self.rows = 0

    def write(self, chunk: mx.array) -> None:
        if chunk.dtype != self.dtype or tuple(chunk.shape[1:]) != self.shape[1:]:
            raise ValueError(f"chunk {chunk.dtype} {tuple(chunk.shape)} does not match {self.dtype} {self.shape}")
        if self.rows + chunk.shape[0] > self.shape[0]:
            raise ValueError("more rows than the declared shape")
        a = chunk.view(mx.uint16) if chunk.dtype == mx.bfloat16 else chunk
        self._f.write(np.ascontiguousarray(np.array(a)).tobytes())
        self.rows += chunk.shape[0]

    def close(self) -> None:
        self._f.close()
        if self.rows != self.shape[0]:
            raise ValueError(f"{self.path}: wrote {self.rows} of {self.shape[0]} rows")

    def __enter__(self) -> "RowStreamWriter":
        return self

    def __exit__(self, exc_type, *_) -> None:
        if exc_type is None:
            self.close()
        else:
            self._f.close()
