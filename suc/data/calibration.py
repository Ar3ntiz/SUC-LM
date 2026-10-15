"""Calibration / validation token partitions and the OOD set (spec §9 T2.1, §32).

Leakage control is by construction: every (rendered) document is assigned to exactly one
partition by a salted hash of its normalized text, so the assignment does not depend on corpus
order, sampling seed or which other documents exist. Exact duplicates (after normalization)
collapse to one document. The final-test set (WikiText-2 test) and the OOD set live elsewhere;
residual n-gram overlap between all of them is measured and recorded in the manifest.

Packing convention: per partition and domain, documents are put in canonical order (by hash),
shuffled with a seed derived from (seed, partition, domain), tokenized without special tokens,
each followed by `tokenizer.eos_token_id`, concatenated, and cut into consecutive seq_len windows
until the domain quota is filled (windows may span document boundaries). Rows of all domains are
then shuffled together with a seed derived from (seed, partition).
"""
from __future__ import annotations

import dataclasses
import hashlib
from itertools import combinations

import mlx.core as mx
import numpy as np

PARTITIONS = ("calibration", "validation")
_P64 = np.uint64(0x100000001B3)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _doc_key(text: str, salt: str) -> str:
    return hashlib.sha256((salt + _normalize(text)).encode("utf-8")).hexdigest()


def _partition_of(key: str, fractions: dict[str, float]) -> str:
    u = int(key[:16], 16) / 2.0 ** 64
    total = sum(fractions.values())
    acc = 0.0
    names = [p for p in PARTITIONS if p in fractions] + [p for p in fractions if p not in PARTITIONS]
    for p in names:
        acc += fractions[p] / total
        if u < acc:
            return p
    return names[-1]


def assign_partition(text: str, fractions: dict[str, float], salt: str) -> str:
    """Partition of a document: u = first 64 bits of sha256(salt + normalized text) / 2^64, mapped
    onto cumulative `fractions` in PARTITIONS order. Normalization = lowercase + collapse whitespace."""
    return _partition_of(_doc_key(text, salt), fractions)


def _rng(seed: int, *names: str) -> np.random.Generator:
    h = hashlib.sha256("|".join([str(seed), *names]).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _quotas(mix: dict[str, float], n: int) -> dict[str, int]:
    """round(mix * n) by largest remainder, so quotas sum exactly to n (ties: mix order)."""
    raw = {d: w * n for d, w in mix.items()}
    q = {d: int(np.floor(v)) for d, v in raw.items()}
    order = sorted(mix, key=lambda d: -(raw[d] - q[d]))
    for d in order[: n - sum(q.values())]:
        q[d] += 1
    return q


def _pack(docs: list[str], tokenizer, seq_len: int, n_windows: int, rng: np.random.Generator,
          label: str) -> tuple[np.ndarray, int]:
    """(n_windows, seq_len) int32 from `docs` shuffled by rng; returns (windows, docs used)."""
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos_token_id")
    need = n_windows * seq_len
    buf: list[int] = []
    used = 0
    for i in rng.permutation(len(docs)):
        if len(buf) >= need:
            break
        buf.extend(tokenizer.encode(docs[i], add_special_tokens=False))
        buf.append(eos)
        used += 1
    if len(buf) < need:
        raise ValueError(f"{label}: needs {n_windows} windows of {seq_len} tokens, only "
                         f"{len(buf) // seq_len} available from {len(docs)} docs")
    return np.asarray(buf[:need], dtype=np.int32).reshape(n_windows, seq_len), used


def _unique_sorted(docs: list[str], salt: str) -> list[tuple[str, str]]:
    """[(key, doc)] deduplicated by normalized text, in canonical (key) order."""
    return sorted({_doc_key(d, salt): d for d in docs if d.strip()}.items())


def build_partitions(docs: dict[str, list[str]], tokenizer, *, seq_len: int, n_sequences: dict[str, int],
                     mix: dict[str, float], seed: int, salt: str = "suc-calib-v1") -> dict[str, dict]:
    """partition -> {"tokens": (n, seq_len) int32, "domains": per-row domain, "n_docs": {domain: docs
    assigned to the partition}, "n_docs_used": {domain: docs that contributed tokens}}.

    Document fractions are proportional to n_sequences. Per-domain window quota is
    round(mix[d] * n_sequences[p]) by largest remainder (rows sum exactly to n_sequences[p]).
    """
    if not n_sequences or set(n_sequences) - set(PARTITIONS):
        raise ValueError(f"n_sequences keys must be a subset of {PARTITIONS}, got {list(n_sequences)}")
    if set(mix) - set(docs):
        raise ValueError(f"mix domains {sorted(set(mix) - set(docs))} have no documents")
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise ValueError(f"mix must sum to 1, got {sum(mix.values())}")
    total = sum(n_sequences.values())
    fractions = {p: n_sequences[p] / total for p in PARTITIONS if p in n_sequences}
    assigned: dict[str, dict[str, list[str]]] = {p: {d: [] for d in mix} for p in fractions}
    for d in mix:
        for key, doc in _unique_sorted(docs[d], salt):
            assigned[_partition_of(key, fractions)][d].append(doc)
    out = {}
    for p in fractions:
        rows, domains, used = [], [], {}
        for d, q in _quotas(mix, n_sequences[p]).items():
            w, used[d] = _pack(assigned[p][d], tokenizer, seq_len, q, _rng(seed, p, d), f"domain {d!r} / {p}")
            rows.append(w)
            domains += [d] * q
        perm = _rng(seed, p).permutation(len(domains))
        out[p] = {"tokens": mx.array(np.concatenate(rows)[perm]), "domains": [domains[i] for i in perm],
                  "n_docs": {d: len(v) for d, v in assigned[p].items()}, "n_docs_used": used}
    return out


def build_ood(docs: list[str], tokenizer, *, seq_len: int, n_sequences: int, seed: int) -> dict:
    """OOD set: same packing as build_partitions, one domain "ood", no partition assignment."""
    uniq = [d for _, d in _unique_sorted(docs, "suc-ood")]
    w, used = _pack(uniq, tokenizer, seq_len, n_sequences, _rng(seed, "ood", "ood"), "ood")
    return {"tokens": mx.array(w), "domains": ["ood"] * n_sequences, "n_docs": {"ood": len(uniq)},
            "n_docs_used": {"ood": used}}


def _ngram_hashes(t: mx.array, n: int) -> np.ndarray:
    a = np.asarray(t, dtype=np.int64)
    a = a.reshape(1, -1) if a.ndim == 1 else a
    if a.shape[1] < n:
        return np.zeros(0, dtype=np.uint64)
    win = np.lib.stride_tricks.sliding_window_view(a, n, axis=1).astype(np.uint64)
    h = np.zeros(win.shape[:2], dtype=np.uint64)
    for j in range(n):
        h = h * _P64 + win[..., j] + np.uint64(1)
    return np.unique(h)


def ngram_overlap(a: mx.array, b: mx.array, n: int = 32) -> int:
    """Number of distinct n-gram hashes (64-bit polynomial, within rows only) shared by a and b.
    1-D inputs are treated as a single row."""
    return int(np.intersect1d(_ngram_hashes(a, n), _ngram_hashes(b, n), assume_unique=True).size)


def partitions_hash(parts: dict[str, dict]) -> str:
    """sha256 over (name, shape, int32 token bytes) of every partition, in sorted-name order."""
    h = hashlib.sha256()
    for name in sorted(parts):
        t = np.asarray(parts[name]["tokens"], dtype=np.int32)
        h.update(name.encode())
        h.update(str(t.shape).encode())
        h.update(t.astype("<i4").tobytes())
    return h.hexdigest()


def partitions_manifest(parts: dict[str, dict], sources: dict, source_hashes: dict[str, str], n: int = 32) -> dict:
    """JSON-able record: rows/domains/docs per partition, sources with file sha256 and licenses,
    n-gram overlap for every pair of partitions (including "ood" if present), partitions_hash."""
    seq_lens = {int(v["tokens"].shape[1]) for v in parts.values()}
    if len(seq_lens) != 1:
        raise ValueError(f"partitions have different seq_len: {seq_lens}")
    info = {}
    for p, v in parts.items():
        doms = v["domains"]
        info[p] = {"rows": len(doms), "rows_per_domain": {d: doms.count(d) for d in sorted(set(doms))},
                   "n_docs": dict(v["n_docs"]), "n_docs_used": dict(v.get("n_docs_used", {}))}
    return {
        "seq_len": seq_lens.pop(),
        "partitions": info,
        "sources": {k: {**dataclasses.asdict(s), "sha256": source_hashes.get(k, "")} for k, s in sources.items()},
        "ngram_n": n,
        "ngram_overlap": {f"{a}|{b}": ngram_overlap(parts[a]["tokens"], parts[b]["tokens"], n)
                          for a, b in combinations(sorted(parts), 2)},
        "partitions_hash": partitions_hash(parts),
    }
