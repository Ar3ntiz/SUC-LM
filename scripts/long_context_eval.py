"""Long-context measurement of the sparse attention indexers (deepseek_v4 CSA lightning indexer, qwen4_exp QSA) on a
compressed model vs its reference, at 8k-64k tokens.

Three measurements, every system run block-sequentially over all sequences (as scripts/system_quality.py):
  (a) PPL / KL on long documents. `ppl_docs` books of the pinned PG19 test parquet (emozilla/pg19-test; disjoint from
      every calibration source of suc.data.sources and from WikiText-2), each cut to max(lengths) + 1 tokens and run
      ONCE: the model is causal, so the NLL of the first L positions of a 64k run IS the PPL at context L (also
      reported: the "tail" [L/2, L), i.e. only predictions with long contexts). KL / top-1 / delta-CE vs the reference
      at every `kl_stride`-th position (the reference logits of those positions are stored in bf16 in the work dir,
      read back row slice by row slice; exact per position, subsampled positions only).
  (b) Needle in a haystack, `depths` x `lengths`: a haystack from the next PG19 book, the sentence "The special magic
      number for <city> is <7 digits>." inserted at a sentence boundary at the given depth, then a question ending in
      "The special magic number for <city> is" and the answer " <digits>". Scored by teacher forcing: the case is
      correct iff the argmax at every answer position is the answer token, which is exactly "greedy decoding produces
      the answer" (also: per-token accuracy, answer NLL). Raw text, no chat template (same for every system).
  (c) Indexer agreement: the selected compressed entries (deepseek_v4: CSA entries of 4 tokens) / key blocks (qwen4_exp:
      blocks of indexer_compress_ratio tokens) per layer, recorded at every `sel_stride`-th query of the documents and
      at the answer positions of the needles, only where the selection is a real choice (more visible units than
      top-k). Overlap = |S_test & S_ref| / |S_ref| per query (chance level |S| / visible units is reported next to
      it), per layer and per position bucket. Each system selects on ITS OWN hidden states (end-to-end agreement).
      For the needles also the "needle hit rate": fraction of answer-position queries whose selection contains a unit
      that overlaps the needle sentence.

Systems (`kind`): `reference` = the checkpoint (native / dequantized BF16 weights) loaded lazily and layer-streamed —
one block of weights resident, released after use, never the whole model; `suc1_python` = a SUC1 file through the
Python runtime (resident, as `suc eval`). `indexer: dense` turns the indexer off (deepseek_v4: every visible
compressed entry; qwen4_exp: dense causal attention) — the control. The first system must be the sparse reference.

Memory (D8): sequences are grouped so their hidden states fit `hidden_budget_gb` (streamed reference) or
`resident_hidden_budget_gb` (resident SUC1 weights; usually one long sequence per group); inside a block every
sequence runs in chunks of `chunk` tokens through the layer's decode cache (the models compute exactly the full
prefill in chunks: tests/unit/test_long_context_eval.py), so temporaries are bounded by the chunk, not by 64k x 64k
scores. `spill_dir` keeps the hidden states on disk instead (suc.models.streaming.HiddenBuffer).

Resumable: each finished system writes work_dir/<name>/<system>/done.json (fingerprinted); a rerun reuses it.

  HF_HUB_OFFLINE=1 .venv/bin/python scripts/long_context_eval.py --config experiments/configs/long_context_X.yaml \\
      [--systems a,b] [--max-len N] [--no-needle] [--keep-work]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from suc.config import ModelConfig, SystemConfig, _build, config_hash, config_to_dict  # noqa: E402
from suc.evaluation import runners as rn  # noqa: E402
from suc.models.streaming import HiddenBuffer, _release, head, load_lazy, model_blocks  # noqa: E402
from suc.provenance import run_metadata, seed_everything, write_result  # noqa: E402

GB = 1e9


# ------------------------------------------------------------------------------------------- configuration

@dataclass(frozen=True)
class SourceConfig:
    """Pinned long-document source: a parquet with one document per row (PG19 test split by default)."""
    repo_id: str = "emozilla/pg19-test"
    filename: str = "data/test-00000-of-00001-29a571947c0b5ccc.parquet"
    revision: str = "c5e39bf32e33f9111323aa68d7d9000d22722035"
    sha256: str | None = "9aae5ddf035760257458cff08d2575d78a15f84eff867af7a87eff0681b01bfc"
    text_field: str = "text"
    local_path: str | None = None          # a local parquet instead of the hub file (tests)


@dataclass(frozen=True)
class NeedleConfig:
    depths: tuple[float, ...] = (0.1, 0.5, 0.9)
    lengths: tuple[int, ...] | None = None  # default: the PPL lengths
    cases_per_cell: int = 1


@dataclass(frozen=True)
class LCSystem:
    name: str
    kind: Literal["reference", "suc1_python"]
    file: str | None = None
    indexer: Literal["sparse", "dense"] = "sparse"
    python_plan: Literal["fused", "separate"] = "fused"
    chunk: int | None = None               # overrides the config chunk for this system


@dataclass(frozen=True)
class LongContextConfig:
    name: str
    model: ModelConfig
    systems: tuple[LCSystem, ...]
    seed: int = 12345
    output_dir: str = "experiments/results"
    source: SourceConfig = field(default_factory=SourceConfig)
    lengths: tuple[int, ...] = (8192, 16384, 32768, 65536)
    ppl_docs: int = 2
    kl_stride: int = 4
    sel_stride: int = 32
    needle: NeedleConfig | None = field(default_factory=NeedleConfig)
    chunk: int = 256
    hidden_budget_gb: float = 24.0
    resident_hidden_budget_gb: float = 4.0
    add_bos: bool = True
    work_dir: str = "~/suc-local/long_context"
    spill_dir: str | None = None


def lc_config_from_dict(d: dict) -> LongContextConfig:
    if not isinstance(d, dict):
        raise ValueError("config: expected a mapping")
    d = dict(d)
    for key in ("model", "systems"):
        if key not in d:
            raise ValueError(f"config: missing required key(s) [{key!r}]")
    systems = d.pop("systems")
    if not isinstance(systems, (list, tuple)) or not systems:
        raise ValueError("config: 'systems' must be a non-empty list")
    nested = {"model": _build(ModelConfig, d.pop("model"), "model"),
              "systems": tuple(_build(LCSystem, s, f"systems[{i}]") for i, s in enumerate(systems))}
    if "source" in d:
        nested["source"] = _build(SourceConfig, d.pop("source"), "source")
    if "needle" in d:
        n = d.pop("needle")
        nested["needle"] = None if n is None else _build(NeedleConfig, n, "needle")
    cfg = _build(LongContextConfig, d, "config", **nested)
    names = [s.name for s in cfg.systems]
    if len(set(names)) != len(names):
        raise ValueError(f"config: duplicate system names {names}")
    if cfg.systems[0].kind != "reference" or cfg.systems[0].indexer != "sparse":
        raise ValueError("config: the first system must be the sparse reference (kind reference, indexer sparse)")
    for s in cfg.systems:
        if (s.kind == "suc1_python") != (s.file is not None):
            raise ValueError(f"system {s.name!r}: 'file' is required for suc1_python and invalid for reference")
    if list(cfg.lengths) != sorted(set(cfg.lengths)) or cfg.lengths[0] < 2:
        raise ValueError("config: lengths must be increasing and >= 2")
    if cfg.kl_stride < 1 or cfg.sel_stride < 1 or cfg.chunk < 1 or cfg.ppl_docs < 0:
        raise ValueError("config: kl_stride, sel_stride, chunk must be >= 1 and ppl_docs >= 0")
    return cfg


def load_lc_config(path: str | Path) -> LongContextConfig:
    with open(path) as f:
        return lc_config_from_dict(yaml.safe_load(f) or {})


# ------------------------------------------------------------------------------------------- data

@dataclass
class Seq:
    id: str
    kind: str                      # "doc" | "needle"
    tokens: np.ndarray             # int32 model input (the doc target is tokens_next)
    record: np.ndarray             # query positions whose indexer selection is recorded
    meta: dict = field(default_factory=dict)


_KEYS = ("Lisbon", "Kyoto", "Valparaiso", "Tromso", "Windhoek", "Hobart", "Ushuaia", "Tbilisi", "Reykjavik",
         "Cusco", "Zanzibar", "Bergen")


def _enc(tok, text: str) -> list[int]:
    return list(tok.encode(text, add_special_tokens=False))


def _source_path(src: SourceConfig) -> str:
    if src.local_path:
        return str(Path(src.local_path).expanduser())
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=src.repo_id, repo_type="dataset", filename=src.filename, revision=src.revision)


def _boundary(tok, hay: list[int], k: int, scan: int = 512) -> int:
    """First index j >= k (within `scan` tokens) right after a sentence end (a token ending in '.', '!' or '?'; PG19
    lines are hard-wrapped, so newlines are not boundaries), else k."""
    dec = getattr(tok, "decode", None)
    if dec is None:
        return k
    for j in range(max(k, 1), min(k + scan, len(hay))):
        if dec([hay[j - 1]]).rstrip().endswith((".", "!", "?")):
            return j
    return k


def build_data(cfg: LongContextConfig, tok, lengths: tuple[int, ...], needle: NeedleConfig | None) -> dict:
    """{"docs": [Seq], "needles": [Seq], "source_sha256", "tokens_sha256"}; documents in parquet file order."""
    import pyarrow.parquet as pq

    from suc.evaluation.data import _sha256

    path = _source_path(cfg.source)
    sha = _sha256(path)
    if cfg.source.sha256 and sha != cfg.source.sha256:
        raise RuntimeError(f"{path}: sha256 {sha} != pinned {cfg.source.sha256}")
    texts = pq.read_table(path, columns=[cfg.source.text_field]).column(cfg.source.text_field).to_pylist()
    bos_id = getattr(tok, "bos_token_id", None) if cfg.add_bos else None
    bos = [int(bos_id)] if bos_id is not None else []
    max_len = max(lengths)
    n_lens = tuple(needle.lengths or lengths) if needle is not None else ()
    hay_len = max(n_lens) if n_lens else 0
    docs, hay, used = [], None, []
    for i, text in enumerate(texts):
        if len(docs) >= cfg.ppl_docs and (hay is not None or not hay_len):
            break
        ids = _enc(tok, (text or "")[: 12 * max(max_len, hay_len) + 4096])
        if len(docs) < cfg.ppl_docs:
            t = bos + ids
            if len(t) >= max_len + 1:
                arr = np.asarray(t[: max_len + 1], dtype=np.int32)
                docs.append(Seq(f"doc{len(docs)}", "doc", arr[:-1],
                                np.arange(cfg.sel_stride - 1, max_len, cfg.sel_stride), {"row": i, "next": arr[1:]}))
                used.append(i)
            continue
        if hay is None and hay_len and len(ids) >= hay_len:
            hay, hay_row = ids[:hay_len], i
            used.append(i)
    if len(docs) < cfg.ppl_docs:
        raise RuntimeError(f"only {len(docs)} source documents have {max_len + 1} tokens")
    needles = []
    if hay_len:
        if hay is None:
            raise RuntimeError(f"no source document with {hay_len} tokens for the needle haystack")
        rng = np.random.default_rng(cfg.seed)
        for L in n_lens:
            for d in needle.depths:
                for c in range(needle.cases_per_cell):
                    key = _KEYS[len(needles) % len(_KEYS)]
                    value = str(int(rng.integers(10 ** 6, 10 ** 7)))
                    nd = _enc(tok, f" The special magic number for {key} is {value}. ")
                    qu = _enc(tok, f"\n\nQuestion: What is the special magic number for {key} mentioned in the text "
                                   f"above?\nAnswer: The special magic number for {key} is")
                    an = _enc(tok, f" {value}")
                    n_hay = L - len(bos) - len(nd) - len(qu) - len(an)
                    if n_hay < 16:
                        raise ValueError(f"needle length {L} too short for the needle / question")
                    h = hay[:n_hay]
                    k = _boundary(tok, h, int(round(d * n_hay)))
                    t = np.asarray(bos + h[:k] + nd + h[k:] + qu + an, dtype=np.int32)
                    a0 = len(t) - len(an)
                    needles.append(Seq(f"needle_L{L}_d{d:g}_{c}", "needle", t, np.arange(a0 - 1, len(t) - 1),
                                       {"length": L, "depth": d, "key": key, "value": value, "answer": an,
                                        "answer_start": a0, "needle_span": [len(bos) + k, len(bos) + k + len(nd)],
                                        "row": hay_row}))
    h = hashlib.sha256()
    for s in docs + needles:
        h.update(s.tokens.astype("<i4").tobytes())
        if s.kind == "doc":
            h.update(s.meta["next"][-1:].astype("<i4").tobytes())
    return {"docs": docs, "needles": needles, "source_sha256": sha, "tokens_sha256": h.hexdigest(),
            "source_rows": used, "source_path": path}


# ------------------------------------------------------------------------------------------- indexer tap

_TAP: "IndexerTap | None" = None


class IndexerTap:
    """While active: the deepseek_v4 / qwen4_exp Indexer.__call__ are wrapped (class level) to record the selection at
    the wanted query positions of the current (sequence, layer), or to switch the indexer off (dense=True: the
    indexer returns None = every visible unit, without computing anything)."""

    def __init__(self, dense: bool = False):
        self.dense = dense
        self.seq: str | None = None
        self.layer: int | None = None
        self.want: np.ndarray | None = None
        self.records: dict[tuple[str, int], dict] = {}
        self.calls = 0
        self._saved: list = []

    def begin(self, seq: str, layer: int, want: np.ndarray | None) -> None:
        self.seq, self.layer, self.want = seq, layer, want

    def _rows(self, pos: np.ndarray) -> np.ndarray:
        if self.want is None or self.seq is None:
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(np.isin(pos, self.want))

    def _add(self, pos: int, nvis: int, ratio: int, idx: np.ndarray) -> None:
        r = self.records.setdefault((self.seq, self.layer), {"pos": [], "nvis": [], "idx": [], "ratio": ratio})
        r["pos"].append(int(pos))
        r["nvis"].append(int(nvis))
        r["idx"].append(idx.astype(np.int32))

    def entries(self, sel: mx.array, pos: mx.array, ratio: int, topk: int) -> None:
        """deepseek_v4: sel (B, L, N) over compressed entries; visible entries of query p: j < (p + 1) // ratio."""
        p = np.array(pos).astype(np.int64)
        rows = self._rows(p)
        if rows.size == 0:
            return
        s = np.array(sel[0][mx.array(rows)])
        for r, row in zip(rows, s):
            nv = int((p[r] + 1) // ratio)
            if nv > topk:
                self._add(p[r], nv, ratio, np.flatnonzero(row[:nv]))

    def blocks(self, sel: mx.array, offset: int, ratio: int, topk: int) -> None:
        """qwen4_exp: sel (B, 1, L, T) token mask; block b (complete, visible) is chosen iff token b * ratio is."""
        L = sel.shape[2]
        p = offset + np.arange(L, dtype=np.int64)
        rows = self._rows(p)
        if rows.size == 0:
            return
        s = np.array(sel[0, 0][mx.array(rows)])
        for r, row in zip(rows, s):
            nv = int((p[r] + 1) // ratio)
            if nv > topk:
                self._add(p[r], nv, ratio, np.flatnonzero(row[: nv * ratio: ratio]))

    def __enter__(self) -> "IndexerTap":
        global _TAP
        from suc.models import mlx_deepseek_v4 as ds4
        from suc.models import mlx_qwen4_exp as q4

        ds4_orig, q4_orig = ds4.Indexer.__call__, q4.Indexer.__call__

        def ds4_call(ix, x, qr, cs, pos, st):
            tap = _TAP
            if tap is not None:
                tap.calls += 1
                if tap.dense:
                    return None
            sel = ds4_orig(ix, x, qr, cs, pos, st)
            if tap is not None and sel is not None:
                tap.entries(sel, pos, ix.compressor.ratio, ix.topk)
            return sel

        def q4_call(ix, x, offset, cache=None):
            tap = _TAP
            if tap is not None:
                tap.calls += 1
                if tap.dense:
                    return None
            sel = q4_orig(ix, x, offset, cache)
            if tap is not None and sel is not None:
                tap.blocks(sel, int(offset), ix.ratio, ix.topk)
            return sel

        self._saved = [(ds4.Indexer, ds4_orig), (q4.Indexer, q4_orig)]
        ds4.Indexer.__call__, q4.Indexer.__call__ = ds4_call, q4_call
        _TAP = self
        return self

    def __exit__(self, *_) -> None:
        global _TAP
        for cls, orig in self._saved:
            cls.__call__ = orig
        _TAP = None

    def save(self, seq: str, path: Path) -> None:
        out = {}
        for (s, layer), r in sorted(self.records.items(), key=lambda kv: kv[0][1]):
            if s != seq:
                continue
            w = max((len(i) for i in r["idx"]), default=0)
            idx = np.full((len(r["idx"]), w), -1, dtype=np.int32)
            for j, i in enumerate(r["idx"]):
                idx[j, : len(i)] = i
            out.update({f"L{layer}_pos": np.asarray(r["pos"], np.int64), f"L{layer}_nvis": np.asarray(r["nvis"]),
                        f"L{layer}_ratio": np.asarray(r["ratio"]), f"L{layer}_idx": idx})
        np.savez(path, **out)
        for k in [k for k in self.records if k[0] == seq]:
            del self.records[k]


def load_selections(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    z = np.load(path)
    layers = sorted({int(k.split("_")[0][1:]) for k in z.files})
    return {L: {"pos": z[f"L{L}_pos"], "nvis": z[f"L{L}_nvis"], "ratio": int(z[f"L{L}_ratio"]),
                "idx": z[f"L{L}_idx"]} for L in layers}


def indexer_layers(model) -> list[int]:
    from suc.models import mlx_deepseek_v4 as ds4
    from suc.models import mlx_qwen4_exp as q4

    out = []
    for i, block in enumerate(model_blocks(model)):
        if any(isinstance(m, (ds4.Indexer, q4.Indexer)) for m in block.modules()):
            out.append(i)
    return out


# ------------------------------------------------------------------------------------------- block-sequential forward

def _make_caches(model) -> list:
    if hasattr(model, "make_cache"):
        return model.make_cache()
    from mlx_lm.models.cache import make_prompt_cache

    return make_prompt_cache(model)


def _embed(model, toks: mx.array, caches) -> mx.array:
    if hasattr(model, "embed_stream"):
        return model.embed_stream(toks, caches)
    return model.model.embed_tokens(toks)


def _groups(seqs: list[Seq], bytes_per_token: int, budget: float) -> list[list[Seq]]:
    out, cur, used = [], [], 0
    for s in seqs:
        b = len(s.tokens) * bytes_per_token
        if cur and used + b > budget:
            out.append(cur)
            cur, used = [], 0
        cur.append(s)
        used += b
    if cur:
        out.append(cur)
    return out


def forward_sequences(get_model: Callable[[], object], seqs: list[Seq], consume: Callable[[Seq, int, mx.array], None],
                      *, chunk: int, streamed: bool, budget_bytes: float, tap: IndexerTap | None = None,
                      spill_dir: Path | None = None, log=print) -> dict:
    """Block-sequential forward of every sequence; consume(seq, start, logits (1, C, V)) per chunk, in order.

    Inside block i each sequence runs chunk by chunk through a fresh decode cache of layer i (the cache path of the
    model computes exactly the full prefill). streamed: the model is a lazy checkpoint; block i is materialized
    before it runs and released after it (a fresh lazy model per sequence group)."""
    model = get_model()
    probe = _embed(model, mx.zeros((1, 1), dtype=mx.int32), _make_caches(model))
    bpt = probe.shape[-1] * probe.dtype.size
    del probe
    groups = _groups(seqs, bpt, budget_bytes)
    t0 = time.perf_counter()
    block_s = []
    for gi, group in enumerate(groups):
        if gi > 0 and streamed:
            del model
            gc.collect()
            mx.clear_cache()
            model = get_model()
        items = [(s, c0, min(c0 + chunk, len(s.tokens))) for s in group for c0 in range(0, len(s.tokens), chunk)]
        hb = HiddenBuffer(len(items), spill_dir / f"g{gi}" if spill_dir is not None else None)
        k = 0
        for s in group:
            ec = _make_caches(model)
            for _, c0, c1 in [it for it in items if it[0] is s]:
                h = _embed(model, mx.array(s.tokens[None, c0:c1]), ec)
                mx.eval(h)
                hb.put(k, h)
                k += 1
        blocks = model_blocks(model)
        for i, block in enumerate(blocks):
            tb = time.perf_counter()
            if streamed:
                mx.eval(block.parameters())
            k = 0
            for s in group:
                cache = _make_caches(model)[i]
                if tap is not None:
                    tap.begin(s.id, i, s.record)
                for _, c0, c1 in [it for it in items if it[0] is s]:
                    h = hb.get(k)
                    out = block(h, "causal" if h.shape[1] > 1 else None, cache)
                    mx.eval(out)
                    hb.put(k, out)
                    del h, out
                    k += 1
                del cache
            if tap is not None:
                tap.begin(None, None, None)
            if streamed:
                _release(block)
            mx.clear_cache()
            block_s.append(time.perf_counter() - tb)
            log(f"  group {gi + 1}/{len(groups)} ({sum(len(s.tokens) for s in group)} tokens) block {i + 1}/"
                f"{len(blocks)}: {block_s[-1]:.1f} s, active {mx.get_active_memory() / GB:.1f} GB, "
                f"peak {mx.get_peak_memory() / GB:.1f} GB")
        for k, (s, c0, _) in enumerate(items):
            lg = head(model, hb.get(k))
            consume(s, c0, lg)
            del lg
        hb.clear()
        mx.clear_cache()
    del model
    gc.collect()
    mx.clear_cache()
    return {"groups": len(groups), "seconds": time.perf_counter() - t0, "block_seconds": block_s,
            "hidden_bytes_per_token": int(bpt)}


# ------------------------------------------------------------------------------------------- scoring

class Scorer:
    """consume() of one system: document NLL, reference-logit rows (reference) or KL vs them (others), needle
    answers. Arrays are written to the system's work directory by finish()."""

    def __init__(self, data: dict, kl_stride: int, ref_dir: Path, is_reference: bool):
        self.kl_stride, self.ref_dir, self.is_ref = kl_stride, ref_dir, is_reference
        self.nll = {s.id: np.zeros(len(s.tokens), np.float32) for s in data["docs"]}
        self.kl = {s.id: {k: [] for k in ("pos", "kl", "top1", "ce_ref", "ce_test")} for s in data["docs"]}
        self.needle = {}
        self._ref: dict[str, np.ndarray] = {}

    def _ref_rows(self, sid: str, n_pos: int, vocab: int) -> np.ndarray:
        if sid not in self._ref:
            p = self.ref_dir / f"{sid}.npy"
            if self.is_ref:
                self.ref_dir.mkdir(parents=True, exist_ok=True)
                self._ref[sid] = np.lib.format.open_memmap(p, mode="w+", dtype=np.uint16, shape=(n_pos, vocab))
            else:
                self._ref[sid] = np.load(p, mmap_mode="r")
        return self._ref[sid]

    def __call__(self, s: Seq, c0: int, logits: mx.array) -> None:
        lg = logits[0].astype(mx.float32)                         # (C, V)
        C, V = lg.shape
        logp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        if s.kind == "doc":
            tgt = mx.array(s.meta["next"][c0:c0 + C])
            nll = -mx.take_along_axis(logp, tgt[:, None], axis=-1)[:, 0]
            self.nll[s.id][c0:c0 + C] = np.array(nll)
            st = self.kl_stride
            first = (-c0) % st
            rows = np.arange(first, C, st)
            if rows.size == 0:
                return
            pos = c0 + rows
            ref = self._ref_rows(s.id, (len(s.tokens) + st - 1) // st, V)
            sel = mx.array(rows)
            if self.is_ref:
                ref[pos // st] = np.array(logits[0][sel].astype(mx.bfloat16).view(mx.uint16))
                return
            lr = mx.array(np.ascontiguousarray(ref[pos // st])).view(mx.bfloat16).astype(mx.float32)
            lr = lr - mx.logsumexp(lr, axis=-1, keepdims=True)
            lt = logp[sel]
            kl = mx.maximum(mx.sum(mx.exp(lr) * (lr - lt), axis=-1), 0.0)
            top1 = mx.argmax(lr, axis=-1) == mx.argmax(lt, axis=-1)
            t = tgt[sel][:, None]
            ce_r, ce_t = -mx.take_along_axis(lr, t, axis=-1)[:, 0], -mx.take_along_axis(lt, t, axis=-1)[:, 0]
            mx.eval(kl, top1, ce_r, ce_t)
            acc = self.kl[s.id]
            for k, v in (("pos", pos), ("kl", kl), ("top1", top1), ("ce_ref", ce_r), ("ce_test", ce_t)):
                acc[k].append(np.asarray(v))
            return
        a0, ans = s.meta["answer_start"], s.meta["answer"]
        rec = self.needle.setdefault(s.id, {"argmax": [None] * len(ans), "logp": [None] * len(ans)})
        for m, t in enumerate(ans):
            p = a0 - 1 + m
            if c0 <= p < c0 + C:
                rec["argmax"][m] = int(mx.argmax(logp[p - c0]).item())
                rec["logp"][m] = float(logp[p - c0, t].item())

    def finish(self, sdir: Path, data: dict) -> None:
        for sid, ref in self._ref.items():
            if self.is_ref:
                ref.flush()
        self._ref.clear()
        for sid, a in self.nll.items():
            np.save(sdir / f"nll_{sid}.npy", a)
        if not self.is_ref:
            for sid, acc in self.kl.items():
                np.savez(sdir / f"kl_{sid}.npz", **{k: np.concatenate(v) if v else np.zeros(0) for k, v in acc.items()})
        cases = []
        for s in data["needles"]:
            r = self.needle[s.id]
            ans = s.meta["answer"]
            hits = [a == t for a, t in zip(r["argmax"], ans)]
            cases.append({"id": s.id, "length": s.meta["length"], "depth": s.meta["depth"],
                          "needle_span": s.meta["needle_span"], "answer_tokens": len(ans),
                          "correct": bool(all(hits)), "token_accuracy": float(np.mean(hits)),
                          "answer_nll": float(-sum(r["logp"])), "argmax": r["argmax"]})
        (sdir / "needle.json").write_text(json.dumps(cases, indent=1))


# ------------------------------------------------------------------------------------------- systems

def open_model(sc: LCSystem, model_dir: Path) -> tuple[Callable[[], object], bool, dict, Callable[[], None]]:
    """(get_model, streamed, info, close) for a system. reference: a fresh lazy checkpoint per call (layer-streamed);
    suc1_python: the SUC1 file loaded once through the Python runtime (suc.evaluation.runners.open_system)."""
    if sc.kind == "reference":
        return (lambda: load_lazy(model_dir)[0]), True, {"kind": "reference", "model_dir": str(model_dir)}, \
            (lambda: None)
    runner = rn.open_system(SystemConfig(name=sc.name, kind="suc1_python", file=sc.file, python_plan=sc.python_plan),
                            model_dir)
    return (lambda: runner.model), False, dict(runner.info), runner.close


def _fingerprint(cfg: LongContextConfig, sc: LCSystem, data: dict, chunk: int) -> str:
    blob = json.dumps({"system": asdict(sc), "model": asdict(cfg.model), "tokens": data["tokens_sha256"],
                       "kl_stride": cfg.kl_stride, "sel_stride": cfg.sel_stride, "chunk": chunk},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def run_system(cfg: LongContextConfig, sc: LCSystem, data: dict, model_dir: Path, work: Path, log=print) -> dict:
    sdir = work / sc.name
    chunk = sc.chunk or cfg.chunk
    fp = _fingerprint(cfg, sc, data, chunk)
    done = sdir / "done.json"
    is_ref = sc is cfg.systems[0]
    ref_dir = work / "ref_logits"
    if done.exists():
        rec = json.loads(done.read_text())
        if rec.get("fingerprint") == fp and (not is_ref or ref_dir.exists()):
            log(f"[long-context] system {sc.name}: reusing {done}")
            return rec
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    if is_ref and ref_dir.exists():
        shutil.rmtree(ref_dir)
    log(f"[long-context] system {sc.name} ({sc.kind}, indexer {sc.indexer}, chunk {chunk})")
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    get_model, streamed, info, close = open_model(sc, model_dir)
    info["load_s"] = time.perf_counter() - t0
    m = get_model()
    info["indexer_layers"] = indexer_layers(m)
    del m
    scorer = Scorer(data, cfg.kl_stride, ref_dir, is_ref)
    seqs = data["docs"] + data["needles"]
    budget = (cfg.hidden_budget_gb if streamed else cfg.resident_hidden_budget_gb) * GB
    spill = Path(cfg.spill_dir).expanduser() / cfg.name / sc.name if cfg.spill_dir else None
    with IndexerTap(dense=sc.indexer == "dense") as tap:
        fw = forward_sequences(get_model, seqs, scorer, chunk=chunk, streamed=streamed, budget_bytes=budget,
                               tap=tap, spill_dir=spill, log=log)
        for s in seqs:
            tap.save(s.id, sdir / f"sel_{s.id}.npz")
        info["indexer_calls"] = tap.calls
    close()
    scorer.finish(sdir, data)
    n_tok = sum(len(s.tokens) for s in seqs)
    rec = {"fingerprint": fp, "system": asdict(sc), "chunk": chunk, "info": info, "forward": fw, "tokens": n_tok,
           "seconds": time.perf_counter() - t0, "tokens_per_s": n_tok / max(fw["seconds"], 1e-9),
           "peak_memory_bytes": int(mx.get_peak_memory())}
    done.write_text(json.dumps(rec, indent=1))
    gc.collect()
    mx.clear_cache()
    log(f"[long-context] system {sc.name}: {rec['seconds']:.0f} s, {rec['tokens_per_s']:.1f} tok/s, peak "
        f"{rec['peak_memory_bytes'] / GB:.1f} GB")
    return rec


# ------------------------------------------------------------------------------------------- aggregation

def ppl_by_length(nll: list[np.ndarray], lengths: tuple[int, ...]) -> dict:
    out = {}
    for L in lengths:
        full = np.concatenate([a[:L] for a in nll]).astype(np.float64)
        tail = np.concatenate([a[L // 2:L] for a in nll]).astype(np.float64)
        out[str(L)] = {"ppl": math.exp(full.mean()), "nll_mean": float(full.mean()), "tokens": int(full.size),
                       "ppl_tail": math.exp(tail.mean()), "tail_tokens": int(tail.size), "tail_from": L // 2}
    return out


def kl_by_length(kl: list[dict], lengths: tuple[int, ...]) -> dict:
    out = {}
    for L in lengths:
        for name, lo in (("all", 0), ("tail", L // 2)):
            parts = [{k: v[(d["pos"] >= lo) & (d["pos"] < L)] for k, v in d.items()} for d in kl]
            n = sum(p["kl"].size for p in parts)
            if n == 0:
                continue
            cat = {k: np.concatenate([p[k] for p in parts]).astype(np.float64) for k in ("kl", "top1", "ce_ref", "ce_test")}
            out.setdefault(str(L), {})[name] = {
                "kl_mean": float(cat["kl"].mean()), "kl_p95": float(np.percentile(cat["kl"], 95)),
                "top1_agreement": float(cat["top1"].mean()), "ce_ref": float(cat["ce_ref"].mean()),
                "ce_test": float(cat["ce_test"].mean()),
                "delta_ce": float(cat["ce_test"].mean() - cat["ce_ref"].mean()), "positions": int(n)}
    return out


def _overlaps(ref: dict, test: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(positions, overlap |T & R| / |R|, chance |R| / visible) over the queries recorded by both."""
    common, ir, it = np.intersect1d(ref["pos"], test["pos"], return_indices=True)
    ov, ch = np.zeros(common.size), np.zeros(common.size)
    for j, (a, b) in enumerate(zip(ir, it)):
        r = ref["idx"][a]
        r = r[r >= 0]
        t = test["idx"][b]
        t = t[t >= 0]
        ov[j] = np.intersect1d(r, t, assume_unique=True).size / max(r.size, 1)
        ch[j] = r.size / max(int(ref["nvis"][a]), 1)
    return common, ov, ch


def _stats(v: np.ndarray) -> dict:
    if v.size == 0:
        return {"n": 0}
    return {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "min": float(v.min()), "n": int(v.size)}


def selection_agreement(ref_sel: dict[str, dict], test_sel: dict[str, dict], seq_ids: list[str],
                        edges: tuple[int, ...]) -> dict:
    """Per layer and per position bucket [0, e0), [e0, e1), ... of the overlap of the test selection with the
    reference one (mean over queries), with chance level."""
    per_layer: dict[int, list] = {}
    for sid in seq_ids:
        r, t = ref_sel.get(sid, {}), test_sel.get(sid, {})
        for layer in sorted(set(r) & set(t)):
            per_layer.setdefault(layer, []).append(_overlaps(r[layer], t[layer]))
    layers, all_pos, all_ov = {}, [], []
    for layer, parts in sorted(per_layer.items()):
        pos = np.concatenate([p[0] for p in parts])
        ov = np.concatenate([p[1] for p in parts])
        ch = np.concatenate([p[2] for p in parts])
        layers[str(layer)] = {**_stats(ov), "chance_mean": float(ch.mean()) if ch.size else None}
        all_pos.append(pos)
        all_ov.append(ov)
    out = {"layers": layers}
    if all_ov:
        pos, ov = np.concatenate(all_pos), np.concatenate(all_ov)
        out["all_layers"] = _stats(ov)
        buckets, lo = {}, 0
        for e in edges:
            m = (pos >= lo) & (pos < e)
            buckets[f"{lo}-{e}"] = _stats(ov[m])
            lo = e
        out["by_position"] = buckets
    return out


def needle_hits(sel: dict[str, dict], needles: list[Seq]) -> dict:
    """Per layer: fraction of answer-position queries whose selection holds a unit overlapping the needle."""
    per_layer: dict[int, list] = {}
    for s in needles:
        a, b = s.meta["needle_span"]
        for layer, r in sel.get(s.id, {}).items():
            ratio = r["ratio"]
            for row in r["idx"]:
                u = row[row >= 0]
                per_layer.setdefault(layer, []).append(bool(np.any((u * ratio < b) & ((u + 1) * ratio > a))))
    layers = {str(k): {"hit_rate": float(np.mean(v)), "n": len(v)} for k, v in sorted(per_layer.items())}
    allv = [x for v in per_layer.values() for x in v]
    return {"layers": layers, "mean": float(np.mean(allv)) if allv else None}


def needle_summary(cases: list[dict]) -> dict:
    if not cases:
        return {}
    acc = lambda cs: float(np.mean([c["correct"] for c in cs])) if cs else None   # noqa: E731
    lens = sorted({c["length"] for c in cases})
    deps = sorted({c["depth"] for c in cases})
    return {"accuracy": acc(cases), "cases": len(cases),
            "token_accuracy": float(np.mean([c["token_accuracy"] for c in cases])),
            "answer_nll_mean": float(np.mean([c["answer_nll"] for c in cases])),
            "by_length": {str(L): acc([c for c in cases if c["length"] == L]) for L in lens},
            "by_depth": {f"{d:g}": acc([c for c in cases if c["depth"] == d]) for d in deps},
            "per_case": cases}


def aggregate(cfg: LongContextConfig, systems: list[LCSystem], recs: dict, data: dict, work: Path,
              lengths: tuple[int, ...]) -> dict:
    ref = cfg.systems[0]
    doc_ids = [s.id for s in data["docs"]]
    needle_ids = [s.id for s in data["needles"]]
    seq_ids = doc_ids + needle_ids
    sels = {sc.name: {sid: load_selections(work / sc.name / f"sel_{sid}.npz") for sid in seq_ids} for sc in systems}
    ref_cases = json.loads((work / ref.name / "needle.json").read_text())
    out = {}
    for sc in systems:
        sdir = work / sc.name
        nll = [np.load(sdir / f"nll_{sid}.npy") for sid in doc_ids]
        cases = json.loads((sdir / "needle.json").read_text())
        r = {k: recs[sc.name][k] for k in ("info", "chunk", "forward", "tokens", "seconds", "tokens_per_s",
                                            "peak_memory_bytes")}
        r["indexer"] = sc.indexer
        r["ppl"] = ppl_by_length(nll, lengths) if nll else {}
        if sc is not ref:
            kl = [dict(np.load(sdir / f"kl_{sid}.npz")) for sid in doc_ids]
            r["kl"] = kl_by_length(kl, lengths) if kl else {}
            agree = [a["correct"] == b["correct"] for a, b in zip(cases, ref_cases)]
            r["needle_agreement_with_reference"] = float(np.mean(agree)) if agree else None
        r["needle"] = needle_summary(cases)
        if sc.indexer == "sparse":
            r["needle_hit_rate"] = needle_hits(sels[sc.name], data["needles"])
            if sc is not ref:
                r["selection_overlap"] = {
                    "documents": selection_agreement(sels[ref.name], sels[sc.name], doc_ids, lengths),
                    "needle_answers": selection_agreement(sels[ref.name], sels[sc.name], needle_ids, lengths)}
        out[sc.name] = r
    return out


# ------------------------------------------------------------------------------------------- main

def main(argv=None, tokenizer=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--systems", default=None, help="comma-separated subset (the reference is always included)")
    ap.add_argument("--max-len", type=int, default=None, help="drop lengths above N (smoke tests; not reported)")
    ap.add_argument("--no-needle", action="store_true")
    ap.add_argument("--keep-work", action="store_true", help="keep the stored reference logits after a full run")
    a = ap.parse_args(argv)
    cfg = load_lc_config(a.config)
    seed_everything(cfg.seed)
    lengths = tuple(L for L in cfg.lengths if a.max_len is None or L <= a.max_len)
    if not lengths:
        raise ValueError("--max-len drops every length")
    needle = None if a.no_needle else cfg.needle
    if needle is not None and a.max_len is not None:
        nl = tuple(L for L in (needle.lengths or lengths) if L <= a.max_len)
        needle = NeedleConfig(needle.depths, nl, needle.cases_per_cell) if nl else None
    pick = set(a.systems.split(",")) if a.systems else None
    systems = [s for i, s in enumerate(cfg.systems) if i == 0 or pick is None or s.name in pick]
    model_dir = rn.resolve_model_dir(cfg.model, any(s.kind == "reference" for s in systems))
    if tokenizer is None:
        from mlx_lm.utils import load_tokenizer

        tokenizer = load_tokenizer(model_dir)
    data = build_data(cfg, tokenizer, lengths, needle)
    print(f"[long-context] {len(data['docs'])} documents x {max(lengths)} tokens, {len(data['needles'])} needle "
          f"cases, source rows {data['source_rows']}", flush=True)
    work = Path(cfg.work_dir).expanduser() / cfg.name
    work.mkdir(parents=True, exist_ok=True)
    recs = {}
    for sc in systems:
        recs[sc.name] = run_system(cfg, sc, data, model_dir, work, log=lambda m: print(m, flush=True))
    res_systems = aggregate(cfg, systems, recs, data, work, lengths)
    md = run_metadata(model=cfg.model.id, model_revision=cfg.model.revision,
                      model_hash=rn.reference_model_hash(model_dir, systems),
                      dataset_hash=f"{cfg.source.repo_id}@{cfg.source.revision}:{data['source_sha256']};"
                                   f"tokens:{data['tokens_sha256']}",
                      seed=cfg.seed, config_hash=config_hash(cfg))
    res = {"metadata": md, "kind": "long_context_eval", "config": config_to_dict(cfg), "config_path": a.config,
           "lengths": list(lengths), "max_len_override": a.max_len, "needle_enabled": needle is not None,
           "data": {"source": asdict(cfg.source), "source_rows": data["source_rows"],
                    "documents": [{"id": s.id, "row": s.meta["row"], "tokens": int(len(s.tokens))}
                                  for s in data["docs"]],
                    "needles": [{"id": s.id, **{k: v for k, v in s.meta.items() if k != "answer"}}
                                for s in data["needles"]]},
           "reference_system": cfg.systems[0].name, "systems": res_systems,
           "notes": "PPL at L = NLL of the first L positions of max(lengths)-token documents (causal); tail = "
                    "positions [L/2, L). KL at every kl_stride-th position vs bf16 reference logits. Needle accuracy "
                    "by teacher forcing (= greedy decode of the answer). Selection overlap per query = |T & R| / |R| "
                    "on each system's own hidden states."}
    p = write_result(res, cfg.output_dir, cfg.name)
    print(f"wrote {p}", flush=True)
    if not a.keep_work and pick is None:
        shutil.rmtree(work / "ref_logits", ignore_errors=True)   # the large part; per-system arrays are kept
    return p


if __name__ == "__main__":
    main()
