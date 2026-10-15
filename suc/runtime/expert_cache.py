"""Expert cache: routed-expert records of a SUC1 file streamed from SSD into a byte-budgeted resident cache.

Motivation (docs/expert_streaming.md): the routed experts of large MoE models (Qwen3.8-Flash-Next: 48 layers x 512
experts, ~63 GB at 4.2 bpp) do not fit a 32-64 GB Mac, but only top-k of them are used per layer and token. The
cache keeps the experts that are used often in (unified, GPU-visible) memory and reads the others on demand.

Independent of the model graph: the caller passes the expert ids chosen by the router of one layer and gets back one
ExpertRecord per id. A record holds the expert's stored payloads (packed index words, row scales, residual arrays) in
the SUC1 storage layout as mx arrays, so its size is exactly its compressed size in the file; shared stack codebooks
are small and stay in the file mapping. switch_from_records builds a SucSwitchLinear over the selected experts only
(remapped expert indices), which computes bit-identical rows to the fully resident SucSwitchLinear.

Pieces:
  ExpertLayout   directory-only map (layer prefix, expert) -> tensors, payload records, coalesced read ranges, bytes.
                 In a SUC1 file payloads are grouped by section (docs/format.md §2) and tensors sorted by name, so one
                 expert's three projections are contiguous inside each section: ~3 reads per expert (indices,
                 scales, residual), the first one ~97 % of the bytes.
  ExpertReader   positional reads (os.pread / preadv; optional F_NOCACHE so the OS page cache does not double the
                 memory of the cached experts) or copies out of an mmap; thread-safe.
  ExpertCache    LRU or segmented-LRU (frequency-aware, O(1)) eviction under a byte budget, optional pinned hot set,
                 parallel miss loads, optional async prefetch (reads in worker threads into host staging buffers; the
                 mx arrays are created on the calling thread), statistics.
  predict_experts  next-layer expert guess: top-k of the NEXT layer's router applied to the CURRENT hidden state
                 (pre-gating heuristic, Eliseev & Mazur 2023; docs/expert_streaming.md §1).
  simulate / budget_plan   trace-driven hit rates with the same eviction policies, and the analytic budget model
                 used by scripts/expert_budget.py.
"""
from __future__ import annotations

import os
import re
import sys
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import mlx.core as mx
import numpy as np

from suc.io.suc1 import _DTYPES, SucModel

EXPERT_RE = re.compile(r"^(?P<prefix>.*)\.experts\.(?P<e>\d+)\.(?P<proj>\w+)\.weight$")
_LAYER_NUM = re.compile(r"layers\.(\d+)")
F_NOCACHE = 48            # <sys/fcntl.h> on macOS (fcntl.F_NOCACHE where Python exposes it)

Key = tuple[int, int]     # (layer index in ExpertLayout.layers, expert id)


# ------------------------------------------------------------------------------------------------- layout

@dataclass(frozen=True)
class ExpertEntry:
    tensors: tuple[str, ...]                       # tensor names (sorted), e.g. …experts.7.{down,gate,up}_proj.weight
    arrays: tuple[tuple[str, dict], ...]           # (tensor name, payload record) in file order
    ranges: tuple[tuple[int, int], ...]            # coalesced (offset, nbytes) reads covering every payload
    nbytes: int                                    # payload bytes (= sum of compressed_size; no shared codebooks)

    @property
    def read_bytes(self) -> int:
        return sum(n for _, n in self.ranges)


class ExpertLayout:
    """Directory-only view of the routed experts of a SUC1 file (no payload is touched)."""

    def __init__(self, sm: SucModel, *, pattern: re.Pattern = EXPERT_RE, merge_gap: int = 4096) -> None:
        self.sm = sm
        groups: dict[str, dict[int, list[str]]] = {}
        for name in sm.names():
            m = pattern.match(name)
            if m is not None:
                groups.setdefault(m.group("prefix"), {}).setdefault(int(m.group("e")), []).append(name)

        def order(prefix: str):
            n = _LAYER_NUM.search(prefix)
            return (prefix.split(".layers.")[0] if n else prefix, int(n.group(1)) if n else -1, prefix)

        self.layers: list[str] = sorted(groups, key=order)
        self._index = {p: i for i, p in enumerate(self.layers)}
        self._entries: dict[Key, ExpertEntry] = {}
        for li, prefix in enumerate(self.layers):
            for e, names in groups[prefix].items():
                self._entries[(li, e)] = self._entry(sorted(names), merge_gap)
        self.num_experts = [len(groups[p]) for p in self.layers]
        for li, p in enumerate(self.layers):
            if sorted(groups[p]) != list(range(self.num_experts[li])):
                raise ValueError(f"{p}: expert ids are not 0..{self.num_experts[li] - 1}")
        self.expert_bytes_total = sum(en.nbytes for en in self._entries.values())
        self.file_bytes = sm.path.stat().st_size

    def _entry(self, names: list[str], merge_gap: int) -> ExpertEntry:
        arrays = sorted(((n, r) for n in names for r in self.sm.descriptor(n)["arrays"]), key=lambda t: t[1]["offset"])
        ranges: list[list[int]] = []
        for _, r in arrays:
            if r["nbytes"] == 0:
                continue
            if ranges and r["offset"] - (ranges[-1][0] + ranges[-1][1]) <= merge_gap:
                ranges[-1][1] = r["offset"] + r["nbytes"] - ranges[-1][0]
            else:
                ranges.append([r["offset"], r["nbytes"]])
        return ExpertEntry(tuple(names), tuple(arrays), tuple((a, b) for a, b in ranges),
                           sum(r["nbytes"] for _, r in arrays))

    def layer_index(self, layer: int | str) -> int:
        if isinstance(layer, str):
            try:
                return self._index[layer]
            except KeyError:
                raise KeyError(f"no routed experts under {layer!r}") from None
        if not 0 <= layer < len(self.layers):
            raise IndexError(f"layer {layer} out of range (0..{len(self.layers) - 1})")
        return layer

    def entry(self, layer: int | str, expert: int) -> ExpertEntry:
        li = self.layer_index(layer)
        try:
            return self._entries[(li, int(expert))]
        except KeyError:
            raise KeyError(f"{self.layers[li]}: no expert {expert}") from None

    def summary(self) -> dict:
        sizes = np.array([en.nbytes for en in self._entries.values()], np.int64)
        reads = np.array([len(en.ranges) for en in self._entries.values()], np.int64)
        return {"layers": len(self.layers), "experts_per_layer": sorted(set(self.num_experts)),
                "experts": int(sizes.size), "expert_bytes_total": int(self.expert_bytes_total),
                "expert_bytes_mean": float(sizes.mean()) if sizes.size else 0.0,
                "expert_bytes_max": int(sizes.max()) if sizes.size else 0,
                "reads_per_expert_mean": float(reads.mean()) if reads.size else 0.0,
                "file_bytes": int(self.file_bytes), "non_expert_bytes": int(self.file_bytes - self.expert_bytes_total)}


# ------------------------------------------------------------------------------------------------- IO

class ExpertReader:
    """Thread-safe reads of byte ranges of one file. mode "pread": positional reads on one descriptor (with
    nocache=True, F_NOCACHE on macOS: no page-cache copy); mode "mmap": copies out of a read-only mapping (faults go
    through the page cache; flash-moe and llama.cpp both measured mmap faulting slower than explicit reads)."""

    def __init__(self, path: str | Path, mode: str = "pread", nocache: bool = True) -> None:
        if mode not in ("pread", "mmap"):
            raise ValueError(f"mode must be 'pread' or 'mmap', got {mode!r}")
        self.path, self.mode = Path(path), mode
        self.nocache = bool(nocache and mode == "pread" and sys.platform == "darwin")
        self._fd = os.open(self.path, os.O_RDONLY)
        if self.nocache:
            import fcntl
            fcntl.fcntl(self._fd, getattr(fcntl, "F_NOCACHE", F_NOCACHE), 1)
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r") if mode == "mmap" else None

    def read(self, offset: int, nbytes: int) -> np.ndarray:
        """(nbytes,) uint8, a fresh buffer owned by the caller."""
        if self._mm is not None:
            return np.array(self._mm[offset:offset + nbytes])
        buf = np.empty(nbytes, np.uint8)
        mv, done = memoryview(buf), 0
        while done < nbytes:
            if hasattr(os, "preadv"):
                n = os.preadv(self._fd, [mv[done:]], offset + done)
            else:
                b = os.pread(self._fd, nbytes - done, offset + done)
                n = len(b)
                mv[done:done + n] = b
            if n <= 0:
                raise IOError(f"short read at {offset + done} in {self.path}")
            done += n
        return buf

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        self._mm = None


# ------------------------------------------------------------------------------------------------- records

class _RecordView(SucModel):
    """SucModel whose payload views of one expert come from that expert's loaded arrays (codebooks: the file)."""

    def __init__(self, sm: SucModel, arrays: dict[int, np.ndarray]) -> None:   # noqa: super().__init__ not wanted
        self.path, self._header, self._dir, self._tensors = sm.path, sm._header, sm._dir, sm._tensors
        self._mm, self._loaded = sm._mm, arrays

    def _view(self, rec: dict) -> np.ndarray:
        a = self._loaded.get(id(rec))       # records are the directory's own dicts (zero-size ones share offsets)
        return a if a is not None else super()._view(rec)


class ExpertRecord:
    """One expert's stored payloads (mx arrays in the SUC1 storage dtype / shape: packed u32 words, scales, residual
    arrays) — `nbytes` is exactly the expert's compressed size."""

    __slots__ = ("layer", "expert", "tensors", "arrays", "nbytes", "_sm")

    def __init__(self, layer: int, expert: int, entry: ExpertEntry, arrays: dict[str, mx.array], sm: SucModel):
        self.layer, self.expert, self.tensors, self.arrays = layer, expert, entry.tensors, arrays
        self.nbytes = entry.nbytes
        self._sm = sm

    def _loaded(self) -> dict[int, np.ndarray]:
        out = {}
        for name in self.tensors:
            for r in self._sm.descriptor(name)["arrays"]:
                out[id(r)] = np.asarray(self.arrays[f"{name}:{r['key']}"])
        return out

    def encoded(self, proj: str):
        """Research-layout Encoded of the tensor whose name contains `.{proj}.` (or the full tensor name), identical
        to SucModel.encoded on the file."""
        name = next((n for n in self.tensors if n == proj or f".{proj}." in n), None)
        if name is None:
            raise KeyError(f"expert {self.expert}: no tensor for {proj!r} in {self.tensors}")
        return _RecordView(self._sm, self._loaded()).encoded(name)


def _arrays_from_buffers(entry: ExpertEntry, bufs: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Slice the read buffers into the expert's payload arrays (storage dtype and shape, zero-copy views)."""
    out = {}
    for name, r in entry.arrays:
        for (off, n), buf in zip(entry.ranges, bufs):
            if off <= r["offset"] and r["offset"] + r["nbytes"] <= off + n:
                s = r["offset"] - off
                out[f"{name}:{r['key']}"] = buf[s:s + r["nbytes"]].view(_DTYPES[r["dtype"]][1]).reshape(r["shape"])
                break
        else:
            if r["nbytes"]:
                raise AssertionError(f"{name}:{r['key']} not covered by the read ranges")
            out[f"{name}:{r['key']}"] = np.zeros(r["shape"], _DTYPES[r["dtype"]][1])
    return out


# ------------------------------------------------------------------------------------------------- policy

class _Policy:
    """Byte-budgeted eviction order. "lru": one recency list. "slru": segmented LRU — new entries enter a probation
    segment, a hit promotes to the protected segment (≤ `protected` of the budget, overflow demoted to probation);
    victims come from probation first, so experts used twice survive a scan of one-off experts (frequency-aware,
    O(1) per operation)."""

    def __init__(self, kind: str, budget: int, protected: float = 0.8) -> None:
        if kind not in ("lru", "slru"):
            raise ValueError(f"policy must be 'lru' or 'slru', got {kind!r}")
        self.kind, self.budget = kind, budget
        self.prob: OrderedDict[Key, int] = OrderedDict()
        self.prot: OrderedDict[Key, int] = OrderedDict()
        self.prot_cap = int(protected * budget)
        self.used = self.prot_used = 0

    def __contains__(self, k: Key) -> bool:
        return k in self.prob or k in self.prot

    def touch(self, k: Key) -> None:
        if k in self.prot:
            self.prot.move_to_end(k)
        elif self.kind == "lru":
            self.prob.move_to_end(k)
        else:
            n = self.prob.pop(k)
            self.prot[k] = n
            self.prot_used += n
            while self.prot_used > self.prot_cap and len(self.prot) > 1:
                d, dn = self.prot.popitem(last=False)
                self.prot_used -= dn
                self.prob[d] = dn

    def insert(self, k: Key, n: int) -> None:
        self.prob[k] = n
        self.used += n

    def remove(self, k: Key) -> None:
        if k in self.prob:
            self.used -= self.prob.pop(k)
        else:
            n = self.prot.pop(k)
            self.used -= n
            self.prot_used -= n

    def victim(self, keep: set) -> Key | None:
        for seg in (self.prob, self.prot):
            for k in seg:
                if k not in keep:
                    return k
        return None


# ------------------------------------------------------------------------------------------------- cache

@dataclass
class CacheStats:
    requests: int = 0            # expert lookups (one per requested id)
    hits: int = 0                # resident (cache or pinned)
    prefetch_hits: int = 0       # served by a prefetch issued earlier (in flight or completed)
    misses: int = 0              # synchronous loads
    prefetch_issued: int = 0
    prefetch_wasted: int = 0     # prefetched experts dropped unused
    evictions: int = 0
    bytes_read: int = 0          # all reads (sync + prefetch)
    bytes_read_sync: int = 0
    read_calls: int = 0
    read_seconds: float = 0.0    # summed per-read wall time in the IO threads
    wait_seconds: float = 0.0    # time get() blocked on IO
    resident_bytes: int = 0
    pinned_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        return (self.hits + self.prefetch_hits) / self.requests if self.requests else 0.0

    def as_dict(self) -> dict:
        return {**asdict(self), "hit_rate": self.hit_rate}


@dataclass
class _Loaded:
    arrays: dict[str, np.ndarray]
    nbytes: int
    reads: int
    seconds: float
    prefetch: bool = False
    future: Future | None = field(default=None, repr=False)


class ExpertCache:
    """Byte-budgeted cache of routed-expert records of a SUC1 file (module docstring).

    budget_bytes bounds the resident records (cache + pinned); host staging buffers of loads in flight are extra
    (at most one layer's worth of sync loads plus the outstanding prefetches)."""

    def __init__(self, source: str | Path | SucModel, budget_bytes: int, *, policy: str = "lru",
                 io: str = "pread", nocache: bool = True, io_threads: int = 8, merge_gap: int = 4096,
                 pattern: re.Pattern = EXPERT_RE, keep_unused_prefetch: bool = False) -> None:
        self.sm = source if isinstance(source, SucModel) else SucModel.load(source, verify=False)
        self.layout = ExpertLayout(self.sm, pattern=pattern, merge_gap=merge_gap)
        self.budget = int(budget_bytes)
        self._policy = _Policy(policy, self.budget)
        self._reader = ExpertReader(self.sm.path, io, nocache)
        self._pool = ThreadPoolExecutor(max_workers=max(1, io_threads), thread_name_prefix="suc-expert-io")
        self._records: dict[Key, ExpertRecord] = {}
        self._pinned: dict[Key, ExpertRecord] = {}
        self._inflight: dict[Key, Future] = {}
        self.keep_unused_prefetch = keep_unused_prefetch
        self.stats = CacheStats()

    # -- IO --------------------------------------------------------------------------------------------
    def _load(self, key: Key) -> _Loaded:          # worker thread: reads only, no mx calls
        entry = self.layout.entry(*key)
        t0 = time.perf_counter()
        bufs = [self._reader.read(off, n) for off, n in entry.ranges]
        return _Loaded(_arrays_from_buffers(entry, bufs), entry.read_bytes, len(entry.ranges),
                       time.perf_counter() - t0)

    def _submit(self, key: Key) -> Future:
        return self._pool.submit(self._load, key)

    def _record(self, key: Key, ld: _Loaded) -> ExpertRecord:
        st = self.stats
        st.bytes_read += ld.nbytes
        st.read_calls += ld.reads
        st.read_seconds += ld.seconds
        arrays = {k: mx.array(v) for k, v in ld.arrays.items()}
        return ExpertRecord(key[0], key[1], self.layout.entry(*key), arrays, self.sm)

    def _install(self, key: Key, rec: ExpertRecord, keep: set) -> None:
        cap = self.budget - self.stats.pinned_bytes
        while self._policy.used + rec.nbytes > cap:
            v = self._policy.victim(keep)
            if v is None:
                raise MemoryError(f"expert cache budget {self.budget} B cannot hold the requested experts "
                                  f"(pinned {self.stats.pinned_bytes} B)")
            self._policy.remove(v)
            del self._records[v]
            self.stats.evictions += 1
        self._policy.insert(key, rec.nbytes)
        self._records[key] = rec
        self.stats.resident_bytes = self._policy.used + self.stats.pinned_bytes

    # -- public API ------------------------------------------------------------------------------------
    def get(self, layer: int | str, experts: Iterable[int]) -> list[ExpertRecord]:
        """Records of `experts` (router ids of one layer, any order, repeats allowed) in the same order."""
        li = self.layout.layer_index(layer)
        ids = [int(e) for e in np.asarray(list(experts)).reshape(-1)]
        keys = [(li, e) for e in ids]
        want = set(keys)
        need = sum(self.layout.entry(*k).nbytes for k in want if k not in self._pinned)
        if need > self.budget - self.stats.pinned_bytes:
            raise MemoryError(f"{len(want)} experts need {need} B > cache budget {self.budget} B "
                              f"(pinned {self.stats.pinned_bytes} B)")
        st = self.stats
        st.requests += len(keys)
        pending: dict[Key, tuple[Future, bool]] = {}
        for k in dict.fromkeys(keys):                  # issue every miss before waiting on any (parallel reads)
            if k in self._pinned or k in self._policy:
                continue
            if k in self._inflight:
                pending[k] = (self._inflight.pop(k), True)
            else:
                pending[k] = (self._submit(k), False)
        for k in keys:                                 # hit accounting per requested id
            if k in self._pinned:
                st.hits += 1
            elif k in self._policy:
                st.hits += 1
                self._policy.touch(k)
            elif pending[k][1]:
                st.prefetch_hits += 1
            else:
                st.misses += 1
        t0 = time.perf_counter()
        for k, (fut, pre) in pending.items():
            ld = fut.result()
            if not pre:
                st.bytes_read_sync += ld.nbytes
            self._install(k, self._record(k, ld), want)
        st.wait_seconds += time.perf_counter() - t0
        for k in [k for k in self._inflight if k[0] == li]:   # this layer has been served: unused predictions
            fut = self._inflight.pop(k)
            st.prefetch_wasted += 1
            if self.keep_unused_prefetch:
                self._install(k, self._record(k, fut.result()), want)
            else:
                ld = fut.result()
                st.bytes_read += ld.nbytes
                st.read_calls += ld.reads
                st.read_seconds += ld.seconds
        return [self._pinned.get(k) or self._records[k] for k in keys]

    def prefetch(self, layer: int | str, experts: Iterable[int]) -> int:
        """Start background reads of the non-resident `experts` of `layer` (consumed by a later get). Returns the
        number of reads issued."""
        li = self.layout.layer_index(layer)
        n = 0
        for e in {int(e) for e in np.asarray(list(experts)).reshape(-1)}:
            k = (li, e)
            if k in self._pinned or k in self._policy or k in self._inflight:
                continue
            self._inflight[k] = self._submit(k)
            n += 1
        self.stats.prefetch_issued += n
        return n

    def pin(self, layer: int | str, experts: Iterable[int]) -> None:
        """Load and pin `experts` (a static hot set, e.g. the most-routed experts of a calibration trace): pinned
        records are never evicted and count against the budget."""
        li = self.layout.layer_index(layer)
        for e in sorted({int(e) for e in experts}):
            k = (li, e)
            if k in self._pinned:
                continue
            n = self.layout.entry(*k).nbytes
            if k in self._policy:
                rec = self._records.pop(k)
                self._policy.remove(k)
            else:
                rec = self._record(k, self._load(k))
            while self._policy.used + self.stats.pinned_bytes + n > self.budget:
                v = self._policy.victim(set())
                if v is None:
                    raise MemoryError(f"pinning {self.layout.layers[li]} expert {e} exceeds the budget")
                self._policy.remove(v)
                del self._records[v]
                self.stats.evictions += 1
            self._pinned[k] = rec
            self.stats.pinned_bytes += n
            self.stats.resident_bytes = self._policy.used + self.stats.pinned_bytes

    def resident(self, layer: int | str) -> list[int]:
        li = self.layout.layer_index(layer)
        return sorted(e for (l, e) in list(self._records) + list(self._pinned) if l == li)

    def reset_stats(self) -> None:
        self.stats = CacheStats(resident_bytes=self.stats.resident_bytes, pinned_bytes=self.stats.pinned_bytes)

    def close(self) -> None:
        for f in self._inflight.values():
            f.cancel()
        self._inflight.clear()
        self._pool.shutdown(wait=True)
        self._reader.close()

    def __enter__(self) -> "ExpertCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------------------------------------------- execution helpers

def compact(experts: np.ndarray | Sequence[int]) -> tuple[list[int], np.ndarray]:
    """(unique experts in first-use order, indices remapped into that list, same shape as `experts`)."""
    a = np.asarray(experts).astype(np.int64)
    uniq, first, inv = np.unique(a.reshape(-1), return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return [int(u) for u in uniq[order]], rank[inv].reshape(a.shape).astype(np.uint32)


def switch_from_records(records: Sequence[ExpertRecord], proj: str):
    """SucSwitchLinear over `records` (one per distinct expert, in the order of compact()) for projection `proj`
    (e.g. "gate_proj"); call it with the remapped indices of compact()."""
    from suc.kernels.suc_switch import SucSwitchLinear
    return SucSwitchLinear.from_encodings([r.encoded(proj) for r in records])


def predict_experts(router_weight: mx.array | np.ndarray, h: mx.array | np.ndarray, k: int) -> np.ndarray:
    """Top-k experts of `router_weight` (E, d) on `h` (d,) or (n, d): the next layer's router applied to the current
    hidden state (pre-gating heuristic). Returns (k,) or (n, k) int64, best first."""
    w = mx.array(router_weight) if not isinstance(router_weight, mx.array) else router_weight
    x = mx.array(h) if not isinstance(h, mx.array) else h
    logits = np.asarray((x.astype(mx.float32) @ w.astype(mx.float32).T))
    idx = np.argsort(-logits, axis=-1, kind="stable")[..., :k]
    return idx.astype(np.int64)


# ------------------------------------------------------------------------------------------------- budget model

def simulate(trace: np.ndarray, expert_bytes: float | np.ndarray, budget_bytes: float, *, policy: str = "lru",
             pinned: dict[int, Sequence[int]] | None = None) -> dict:
    """Replay a routing trace (tokens, layers, k) through the eviction policy (no IO). expert_bytes: scalar or
    (layers, experts). Returns hit rate, miss bytes per token and P(layer has ≥ 1 miss)."""
    tr = np.asarray(trace)
    if tr.ndim != 3:
        raise ValueError(f"trace must be (tokens, layers, k), got {tr.shape}")
    T, L, _ = tr.shape
    size = (lambda l, e: float(expert_bytes)) if np.isscalar(expert_bytes) else \
        (lambda l, e: float(expert_bytes[l][e]))
    pin = {(l, int(e)) for l, es in (pinned or {}).items() for e in es}
    pinned_bytes = sum(size(l, e) for l, e in pin)
    pol = _Policy(policy, int(budget_bytes - pinned_bytes))
    if pol.budget < 0:
        raise ValueError("pinned set exceeds the budget")
    req = hit = 0
    miss_bytes = 0.0
    layer_miss = 0
    for t in range(T):
        for l in range(L):
            ks = [(l, int(e)) for e in tr[t, l]]
            want, missed = set(ks), False
            for k in ks:
                req += 1
                if k in pin:
                    hit += 1
                elif k in pol:
                    hit += 1
                    pol.touch(k)
            for k in dict.fromkeys(ks):
                if k in pin or k in pol:
                    continue
                missed = True
                n = size(*k)
                miss_bytes += n
                while pol.used + n > pol.budget:
                    v = pol.victim(want)
                    if v is None:
                        raise MemoryError("budget smaller than one layer's experts")
                    pol.remove(v)
                pol.insert(k, int(n))
            layer_miss += missed
    return {"hit_rate": hit / req if req else 0.0, "miss_bytes_per_token": miss_bytes / T,
            "p_layer_miss": layer_miss / (T * L), "tokens": T}


def budget_plan(*, ram_bytes: float, resident_bytes: float, overhead_bytes: float, n_layers: int, n_experts: int,
                top_k: int, expert_bytes: float, ssd_bytes_per_s: float, compute_s: float,
                read_latency_s: float = 150e-6, hit_rate: float | None = None,
                p_layer_miss: float | None = None) -> dict:
    """Analytic per-token cost of expert streaming under a RAM budget (docs/expert_streaming.md §3).

    cache = ram - resident - overhead; capacity fraction f = cache / all expert bytes. Without a measured hit_rate
    the uniform-routing value h = f is used (exact for LRU under independent uniform routing; real routing is skewed,
    so this is a pessimistic bound). Per token: miss bytes = L·k·(1-h)·expert_bytes; IO time = miss bytes / SSD
    bandwidth + L·P(layer has a miss)·read latency (one dependent round trip per layer: a layer's experts are
    known only after its router ran). Serial tok/s = 1 / (compute + IO); overlapped = 1 / max(compute, IO) is the
    ceiling of perfect next-layer prefetch."""
    total = n_layers * n_experts * expert_bytes
    cache = ram_bytes - resident_bytes - overhead_bytes
    layer_need = top_k * expert_bytes
    out = {"ram_gb": ram_bytes / 1e9, "resident_gb": resident_bytes / 1e9, "overhead_gb": overhead_bytes / 1e9,
           "experts_gb": total / 1e9, "cache_gb": max(cache, 0.0) / 1e9}
    if cache < layer_need:
        return {**out, "feasible": False, "reason": "budget below resident set + one layer of experts"}
    f = min(1.0, cache / total)
    h = f if hit_rate is None else hit_rate
    if f >= 1.0:
        h = 1.0
    p_miss = (1.0 - h ** top_k) if p_layer_miss is None else p_layer_miss
    if h >= 1.0:
        p_miss = 0.0
    miss_bytes = n_layers * top_k * (1.0 - h) * expert_bytes
    io_s = miss_bytes / ssd_bytes_per_s + n_layers * p_miss * read_latency_s
    return {**out, "feasible": True, "capacity_fraction": f, "hit_rate": h, "all_experts_resident": f >= 1.0,
            "miss_mb_per_token": miss_bytes / 1e6, "io_ms_per_token": 1e3 * io_s, "compute_ms_per_token": 1e3 * compute_s,
            "tok_s_serial": 1.0 / (compute_s + io_s), "tok_s_overlapped": 1.0 / max(compute_s, io_s),
            "ssd_gb_per_s": ssd_bytes_per_s / 1e9}


def zipf_trace(tokens: int, layers: int, experts: int, k: int, s: float, *, seed: int = 0) -> np.ndarray:
    """Synthetic routing trace: per layer a random permutation of a Zipf(s) popularity, k distinct experts per token
    (s = 0: uniform). For what-if tables only — real hit rates need a routing trace of the model."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, experts + 1) ** s
    p /= p.sum()
    out = np.empty((tokens, layers, k), np.int64)
    for l in range(layers):
        perm = rng.permutation(experts)
        pl = p[np.argsort(perm)]
        g = rng.gumbel(size=(tokens, experts)) + np.log(pl)       # Gumbel top-k = sampling without replacement
        out[:, l] = np.argsort(-g, axis=1)[:, :k]
    return out


__all__ = ["EXPERT_RE", "ExpertLayout", "ExpertEntry", "ExpertReader", "ExpertRecord", "ExpertCache", "CacheStats",
           "compact", "switch_from_records", "predict_experts", "simulate", "budget_plan", "zipf_trace"]
