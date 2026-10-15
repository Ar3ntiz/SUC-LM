"""Static, interleaved rANS over many symbol streams at once (numpy reference; storage-only, spec §15).

Coder: 32-bit state, 16-bit renormalization words, lower bound L = 2**16, PROB_BITS-bit quantized frequencies
(the classic rans_byte construction with 16-bit I/O): encode x -> ((x // f) << n) + x % f + c after emitting the
low 16 bits while x >= f << (32 - n); decode slot = x & (2**n - 1), x -> f * (x >> n) + slot - c, then read one
16-bit word while x < L. One renormalization step suffices in both directions because n <= 16.

Interleaving: every stream s (one frequency table, n_s symbols) is split over lanes_s = ceil(n_s / steps) lanes;
symbol i of the stream goes to step i // lanes_s, lane i % lanes_s. All lanes of all streams advance in lockstep
over the same `steps`, so one numpy operation processes one symbol of every lane (the structure of SIMD / GPU
interleaved rANS). The word stream is ordered by step, then by global lane; within a step the lanes of one stream
are contiguous, so a per-stream file layout is the same words regrouped by stream (sizes are reported per stream).
Stored overhead per stream: its frequency table (u16 per alphabet entry) and the final 32-bit state of every lane.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PROB_BITS = 14
RANS_L = 1 << 16
WORD_BITS = 16


def quantize_freqs(counts: np.ndarray, prob_bits: int = PROB_BITS) -> np.ndarray:
    """Counts -> integer frequencies summing to 2**prob_bits, every present symbol >= 1 (absent ones 0)."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("empty histogram")
    m = 1 << prob_bits
    if int((counts > 0).sum()) > m:
        raise ValueError(f"alphabet with {(counts > 0).sum()} present symbols does not fit {prob_bits} bits")
    f = np.where(counts > 0, np.maximum(1, np.floor(counts * m / total + 0.5)), 0).astype(np.int64)
    diff = m - int(f.sum())
    # fix the sum on the symbols where the change costs the least: largest frequencies first
    order = np.argsort(-f, kind="stable")
    i = 0
    while diff != 0:
        s = order[i % order.size]
        if diff > 0:
            f[s] += 1
            diff -= 1
        elif f[s] > 1:
            f[s] -= 1
            diff += 1
        i += 1
    return f.astype(np.uint32)


@dataclass
class RansEncoded:
    words: np.ndarray            # uint16, ordered by step then global lane
    final_states: np.ndarray     # uint32 per global lane (decoder start state)
    freqs: np.ndarray            # (n_streams, alphabet) uint32, rows sum to 2**prob_bits
    lengths: np.ndarray          # int64 symbols per stream
    lanes: np.ndarray            # int64 lanes per stream
    stream_words: np.ndarray     # int64 16-bit words per stream
    steps: int
    prob_bits: int

    def stream_bytes(self, alphabet_sizes: np.ndarray | None = None) -> np.ndarray:
        """Stored bytes per stream: words + final lane states + u16 table + 3 u64 header fields."""
        a = self.freqs.shape[1] if alphabet_sizes is None else np.asarray(alphabet_sizes)
        return 2 * self.stream_words + 4 * self.lanes + 2 * a + 24

    def total_bytes(self, alphabet_sizes: np.ndarray | None = None) -> int:
        return int(self.stream_bytes(alphabet_sizes).sum())


def _layout(lengths: np.ndarray, steps: int):
    lanes = np.maximum(1, -(-lengths // steps))
    stream_of_lane = np.repeat(np.arange(lengths.size), lanes)
    first = np.concatenate([[0], np.cumsum(lanes)[:-1]])
    local = np.arange(int(lanes.sum())) - np.repeat(first, lanes)
    ln, nl = lengths[stream_of_lane], lanes[stream_of_lane]
    count = np.where(local < ln, (ln - local + nl - 1) // nl, 0)   # symbols carried by each lane
    return lanes, stream_of_lane, local, count


def _grid(streams: list[np.ndarray], lanes: np.ndarray, steps: int, dt=None) -> np.ndarray:
    """(steps, total_lanes) grid of symbols (zeros at padding positions)."""
    if dt is None:
        dt = np.uint8 if all(s.dtype.itemsize == 1 for s in streams) else np.uint16
    out = np.zeros((steps, int(lanes.sum())), dtype=dt)
    col = 0
    for s, nl in zip(streams, lanes):
        nl = int(nl)
        pad = np.zeros(steps * nl, dtype=dt)
        pad[:s.size] = s
        out[:, col:col + nl] = pad.reshape(steps, nl)
        col += nl
    return out


def freqs_from_counts(counts: np.ndarray, prob_bits: int = PROB_BITS) -> np.ndarray:
    """Vectorized, integer-exact table construction for context models: (..., a) counts -> frequencies with
    every symbol >= 1 (unseen symbols stay codable) summing to 2**prob_bits per row. Each symbol gets 1, the
    remaining 2**prob_bits - a are split proportionally (floor), the leftover goes to the largest fractional
    parts (ties: lower symbol). Encoder and decoder call it on identical counts, so tables need not be stored."""
    c = np.asarray(counts, dtype=np.int64)
    a = c.shape[-1]
    m = 1 << prob_bits
    if a > m:
        raise ValueError("alphabet larger than 2**prob_bits")
    tot = c.sum(axis=-1, keepdims=True)
    c = np.where(tot > 0, c, 1)
    tot = np.where(tot > 0, tot, a)
    num = c * (m - a)
    f = num // tot + 1
    rem = m - f.sum(axis=-1, keepdims=True)
    order = np.argsort(-(num % tot), axis=-1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(a), order.shape), axis=-1)
    return (f + (rank < rem)).astype(np.uint32)


def _cum(freqs: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros((freqs.shape[0], 1), np.uint32),
                           np.cumsum(freqs, axis=1, dtype=np.uint32)[:, :-1]], axis=1)


def _check(streams, freqs, prob_bits, table_ids):
    if not 1 <= prob_bits <= 16:
        raise ValueError("prob_bits must be in [1, 16]")
    if np.any(freqs.sum(axis=1) != 1 << prob_bits):
        raise ValueError("every frequency table must sum to 2**prob_bits")
    if table_ids is None:
        if freqs.shape[0] != len(streams):
            raise ValueError("one frequency table per stream (or pass table_ids)")
        table_ids = [None] * len(streams)
    elif len(table_ids) != len(streams):
        raise ValueError("one table-id array per stream")
    for i, (s, t) in enumerate(zip(streams, table_ids)):
        if not s.size:
            continue
        if t is not None and (t.shape != s.shape or int(t.min()) < 0 or int(t.max()) >= freqs.shape[0]):
            raise ValueError(f"stream {i}: bad table ids")
        if int(s.max()) >= freqs.shape[1] or np.any(freqs[i if t is None else t, s] == 0):
            raise ValueError(f"stream {i}: symbol outside its table")


def encode(streams: list[np.ndarray], freqs: np.ndarray, steps: int = 8192, prob_bits: int = PROB_BITS,
           table_ids: list[np.ndarray] | None = None) -> RansEncoded:
    """Encode every stream s with table freqs[s], or, with `table_ids`, symbol i of stream s with table
    freqs[table_ids[s][i]] (context modelling: the decoder must be given the same table ids)."""
    freqs = np.asarray(freqs, dtype=np.uint32)
    _check(streams, freqs, prob_bits, table_ids)
    lengths = np.array([s.size for s in streams], dtype=np.int64)
    steps = max(1, min(steps, int(lengths.max()) if lengths.size else 1))
    lanes, sol, _, count = _layout(lengths, steps)
    grid = _grid(streams, lanes, steps)
    tgrid = None if table_ids is None else _grid([t.astype(np.int64) for t in table_ids], lanes, steps, np.int64)
    a = freqs.shape[1]
    fbase = (sol * a).astype(np.int64)
    ff, cf = freqs.reshape(-1).astype(np.uint64), _cum(freqs).reshape(-1).astype(np.uint64)
    n = np.uint64(prob_bits)
    x = np.full(sol.size, RANS_L, dtype=np.uint64)
    chunks: list[np.ndarray] = []
    lane_ids: list[np.ndarray] = []
    for t in range(steps - 1, -1, -1):
        valid = t < count
        idx = (fbase if tgrid is None else tgrid[t] * a) + grid[t]
        f, c = np.where(valid, ff[idx], np.uint64(1)), cf[idx]     # padding: dummy f = 1, result discarded
        emit = valid & (x >= (f << np.uint64(32 - prob_bits)))
        e = np.flatnonzero(emit)
        chunks.append((x[e] & np.uint64(0xFFFF)).astype(np.uint16))
        lane_ids.append(e)
        x[e] >>= np.uint64(WORD_BITS)
        enc = ((x // f) << n) + (x % f) + c
        x = np.where(valid, enc, x)
    words = np.concatenate(chunks[::-1]) if chunks else np.zeros(0, np.uint16)
    per_lane = np.bincount(np.concatenate(lane_ids), minlength=sol.size) if lane_ids else np.zeros(sol.size, np.int64)
    stream_words = np.bincount(sol, weights=per_lane, minlength=len(streams)).astype(np.int64)
    return RansEncoded(words, x.astype(np.uint32), freqs, lengths, lanes, stream_words, steps, prob_bits)


def decode(enc: RansEncoded, table_ids: list[np.ndarray] | None = None) -> list[np.ndarray]:
    """Exact inverse of encode (pass the same table_ids if the encoder used them)."""
    pb, steps = enc.prob_bits, enc.steps
    lanes, sol, _, count = _layout(enc.lengths, steps)
    freqs = enc.freqs
    ns, a = freqs.shape
    m = 1 << pb
    cum = _cum(freqs)
    odt = np.uint16 if a > 256 else np.uint8
    tgrid = None if table_ids is None else _grid([t.astype(np.int64) for t in table_ids], lanes, steps, np.int64)
    use_lut = tgrid is None and ns * m <= 1 << 26
    if use_lut:   # slot -> symbol lookup per table
        lut = np.empty((ns, m), dtype=odt)
        for i in range(ns):
            lut[i] = np.repeat(np.arange(a), freqs[i])
        lut_flat, lbase = lut.reshape(-1), (sol * m).astype(np.int64)
    else:         # binary search on the cumulative table (a + 1 columns, cum[a] = m)
        cumx = np.concatenate([cum, np.full((ns, 1), m, np.uint32)], axis=1).reshape(-1).astype(np.int64)
        nbits = max(1, (a - 1).bit_length())
    fbase = (sol * a).astype(np.int64)
    ff, cf = freqs.reshape(-1).astype(np.uint64), cum.reshape(-1).astype(np.uint64)
    x = enc.final_states.astype(np.uint64)
    words = enc.words
    out = np.zeros((steps, sol.size), dtype=odt)
    mask, n, lo = np.uint64(m - 1), np.uint64(pb), np.uint64(RANS_L)
    ptr = 0
    for t in range(steps):
        valid = t < count
        slot = x & mask
        if use_lut:
            s = lut_flat[lbase + slot.astype(np.int64)]
            idx = fbase + s
        else:
            tb = fbase // a if tgrid is None else tgrid[t]
            cb = tb * (a + 1)
            sl = slot.astype(np.int64)
            s = np.zeros(sol.size, np.int64)
            for b in range(nbits - 1, -1, -1):
                cand = s + (1 << b)
                ok = cand < a
                ok[ok] = cumx[cb[ok] + cand[ok]] <= sl[ok]
                s = np.where(ok, cand, s)
            idx = tb * a + s
        xn = ff[idx] * (x >> n) + slot - cf[idx]
        x = np.where(valid, xn, x)
        out[t] = s
        need = np.flatnonzero(valid & (x < lo))
        k = need.size
        if k:
            x[need] = (x[need] << np.uint64(WORD_BITS)) | words[ptr:ptr + k].astype(np.uint64)
            ptr += k
    if ptr != words.size or np.any(x != RANS_L):
        raise ValueError("corrupt rANS stream (words left over or final state mismatch)")
    res, col = [], 0
    for ln, nl in zip(enc.lengths, lanes):
        nl = int(nl)
        res.append(np.ascontiguousarray(out[:, col:col + nl]).reshape(-1)[:int(ln)])
        col += nl
    return res
