"""Phase 8 (spec §15): entropy of a real SUC1 file and what storage-only entropy coding can actually save.

Not a test. CPU only (numpy, zlib/lzma in process, the `zstd` command line tool if present). Usage:
  .venv/bin/python benchmarks/entropy_coding.py --suc1 ~/suc-local/compress/compress_qwen3_0.6b/container/model.suc1 \
      [--out-dir experiments/results] [--steps 8192] [--fast]

Measured, per tensor and per section:
  * code indices: order-0 entropy per codebook, conditional entropy given the previous group in the row ("left"),
    the same group in the previous row ("up") and, for additive codebooks, the same group's code of the previous
    codebook ("prev_cb"); plug-in AND two-fold held-out (fold = row parity, add-0.5 smoothing). Plug-in order-1
    estimates with 65536 bins are biased low; only the held-out numbers are achievable rates.
  * residual positions (sorted unique flat positions): fixed position_bits vs log2 C(numel, n) vs exact Rice code
    of the gaps; residual int8 values: order-0 entropy; scales / codebooks / raw: byte-plane entropies.
  * real coders on every section blob (payload bytes concatenated, padding excluded): zlib -9, xz/lzma -6,
    zstd -3 / -19 (CLI, -T1), each also on a byte-shuffled copy (f16/bf16 sections: low bytes then high bytes);
    a static interleaved rANS (suc.entropy.rans, one table per tensor and codebook) on all indices and residual
    values; a split-plane Rice code (suc.entropy.measures) on residual position gaps. rANS and Rice decodes are
    verified bit-exact against the file.
  * decode throughput (MB/s of decoded output, single thread) and the load-time decode seconds of an 8B-sized file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

from suc.entropy import measures as em
from suc.entropy import rans
from suc.io.suc1 import FOOTER_SIZE, HEADER_SIZE, SECTIONS, SucModel

ZSTD = shutil.which("zstd")
QWEN3_8B_PARAMS = 8.19e9          # Qwen3-8B incl. untied embedding and head
BPP_8B = 4.18                     # R1: 8B additive M=4 + 0.5% residual, actual bpp


# ------------------------------------------------------------------------------------------- helpers

def _section_blobs(m: SucModel) -> dict[str, bytes]:
    segs = {s: [] for s in SECTIONS}
    for c in m.codebooks():
        segs["codebooks"].append((c["offset"], c["nbytes"]))
    for d in m._dir["tensors"]:
        for r in d["arrays"]:
            segs[r["part"]].append((r["offset"], r["nbytes"]))
    mm = m._mm
    out = {s: b"".join(bytes(mm[o:o + n]) for o, n in sorted(v)) for s, v in segs.items()}
    out["directory"] = bytes(mm[HEADER_SIZE:HEADER_SIZE + m._header[6]])
    return out


def _shuffle(b: bytes, width: int = 2) -> bytes:
    a = np.frombuffer(b[:len(b) // width * width], np.uint8).reshape(-1, width)
    return a.T.tobytes() + b[len(b) // width * width:]


def _timeit(fn, repeats: int = 1):
    best, out = None, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return out, best


def _zstd(blob: bytes, level: int, tmp: Path) -> tuple[int, float, float]:
    src, dst = tmp / "blob.bin", tmp / "blob.zst"
    src.write_bytes(blob)
    t0 = time.perf_counter()
    subprocess.run([ZSTD, "-q", "-f", "-T1", f"-{level}", "--ultra" if level > 19 else "-q", str(src), "-o", str(dst)],
                   check=True)
    tc = time.perf_counter() - t0
    best = None
    for _ in range(2):
        t0 = time.perf_counter()
        r = subprocess.run([ZSTD, "-q", "-d", "-c", "-T1", str(dst)], check=True, stdout=subprocess.PIPE)
        td = time.perf_counter() - t0
        best = td if best is None else min(best, td)
    if r.stdout != blob:
        raise AssertionError("zstd round trip mismatch")
    return dst.stat().st_size, tc, best


def _generic(name: str, blob: bytes, tmp: Path, fast: bool) -> list[dict]:
    rows = []
    variants = [("plain", blob)]
    if name in ("codebooks", "scales", "raw"):
        variants.append(("shuffle2", _shuffle(blob)))
    big = len(blob) > 64 << 20
    for vname, data in variants:
        coders = [("zlib-9", lambda d: zlib.compress(d, 9), zlib.decompress)]
        if not (fast and big):
            coders.append(("xz-6", lambda d: lzma.compress(d, preset=6), lzma.decompress))
        for cname, comp, decomp in coders:
            c, tc = _timeit(lambda: comp(data))
            d, td = _timeit(lambda: decomp(c), 2)
            assert d == data
            rows.append({"section": name, "variant": vname, "coder": cname, "in_bytes": len(data),
                         "out_bytes": len(c), "compress_s": tc, "decompress_s": td,
                         "decode_MBps": len(data) / td / 1e6})
        if ZSTD:
            for lvl in (3, 19):
                if fast and big and lvl == 19:
                    continue
                n, tc, td = _zstd(data, lvl, tmp)
                rows.append({"section": name, "variant": vname, "coder": f"zstd-{lvl}", "in_bytes": len(data),
                             "out_bytes": n, "compress_s": tc, "decompress_s": td,
                             "decode_MBps": len(data) / td / 1e6, "note": "CLI wall time incl. process start"})
    return rows


def _index_stats(idx: np.ndarray, k: int, prev: np.ndarray | None) -> dict:
    """idx (rows, groups) of one codebook; prev: same-shape codes of the previous codebook or None."""
    rows, g = idx.shape
    fold = (np.arange(rows) % 2 == 0)[:, None] & np.ones((1, g), bool)
    s = idx.astype(np.int64)
    out = {"n": int(s.size), "H0": em.order0(s, k),
           "H0_heldout": em.two_fold(np.zeros_like(s), s, fold, k)}
    if g > 1:
        c, y, f = s[:, :-1], s[:, 1:], fold[:, 1:]
        out["H_left_plugin"], out["H_left_heldout"] = em.conditional_plugin(c, y, k), em.two_fold(c, y, f, k)
    if rows > 1:
        c, y = s[:-1], s[1:]
        f = (np.arange(1, rows) % 2 == 0)[:, None] & np.ones((1, g), bool)
        out["H_up_plugin"], out["H_up_heldout"] = em.conditional_plugin(c, y, k), em.two_fold(c, y, f, k)
    if prev is not None:
        c = prev.astype(np.int64)
        out["H_prevcb_plugin"], out["H_prevcb_heldout"] = em.conditional_plugin(c, s, k), em.two_fold(c, s, fold, k)
    return out


def _plane_entropy(blob: bytes, width: int) -> list[float]:
    a = np.frombuffer(blob[:len(blob) // width * width], np.uint8).reshape(-1, width)
    return [em.order0(a[:, j], 256) for j in range(width)]


def _prevcb_adaptive(additive: list[tuple[str, np.ndarray]], chunks: int, ctx_steps: int, o0_steps: int,
                     beta: int) -> dict:
    """Additive codes with the code of the previous codebook at the same position as context, no stored tables.

    Codebook 0 of every tensor and row chunk 0 of codebooks 1..M-1 use static order-0 rANS (tables stored).
    Row chunk j >= 1 of codebook m >= 1 uses one table per context value (256 x 256 per tensor) built by
    rans.freqs_from_counts from the (code m-1, code m) pair counts of chunks < j, which the decoder has already
    decoded, smoothed towards the marginal of code m: p(y | c) ∝ n(c, y) + beta * (marg(y) + 1) / sum(marg + 1)
    (integer arithmetic); codebooks are decoded in order m = 0, 1, ..., so the context is known. All tensors are coded in
    lockstep, one rANS call per (m, j). Decoding (timed) rebuilds the tables exactly as the encoder did."""
    if not additive:
        return {"total_bytes": 0, "decode_s": 1e-9, "per_tensor_bytes": {}}
    names = [n for n, _ in additive]
    M = additive[0][1].shape[0]
    bounds = [[(r * j) // chunks for j in range(chunks + 1)] for r in (i.shape[1] for _, i in additive)]

    def chunk(i, m, j):
        idx = additive[i][1]
        b = bounds[i]
        return np.ascontiguousarray(idx[m, b[j]:b[j + 1]]).reshape(-1)

    per_tensor = dict.fromkeys(names, 0)
    # static order-0 part
    o0_streams, o0_owner = [], []
    for i, (_, idx) in enumerate(additive):
        o0_streams.append(np.ascontiguousarray(idx[0]).reshape(-1))
        o0_owner.append(i)
        for m in range(1, M):
            o0_streams.append(chunk(i, m, 0))
            o0_owner.append(i)
    f0 = np.stack([rans.quantize_freqs(np.bincount(s, minlength=256)) for s in o0_streams])
    e0 = rans.encode(o0_streams, f0, steps=o0_steps)
    b0 = e0.stream_bytes(np.full(len(o0_streams), 256))
    for i, b in zip(o0_owner, b0):
        per_tensor[names[i]] += int(b)
    t0 = time.perf_counter()
    d0 = rans.decode(e0)
    t_dec = time.perf_counter() - t0
    if not all(np.array_equal(a, b) for a, b in zip(o0_streams, d0)):
        raise AssertionError("order-0 part not bit-exact")
    del d0
    # context passes
    nt = len(additive)
    curve = []
    ctx_bytes, lanes_total = 0, 0
    t_build = 0.0
    for m in range(1, M):
        counts = np.zeros((nt, 256, 256), np.int64)
        marg = np.ones((nt, 256), np.int64)
        for j in range(1, chunks):
            tb0 = time.perf_counter()
            for i in range(nt):      # counts of chunk j-1 (decoded by now)
                c, y = chunk(i, m - 1, j - 1).astype(np.int64), chunk(i, m, j - 1).astype(np.int64)
                counts[i] += np.bincount(c * 256 + y, minlength=65536).reshape(256, 256)
                marg[i] += np.bincount(y, minlength=256)
            sm = counts * marg.sum(axis=1)[:, None, None] + beta * marg[:, None, :]
            tables = rans.freqs_from_counts(sm.reshape(nt * 256, 256))
            t_build += time.perf_counter() - tb0
            syms = [chunk(i, m, j) for i in range(nt)]
            tids = [i * 256 + chunk(i, m - 1, j).astype(np.int64) for i in range(nt)]
            enc = rans.encode(syms, tables, steps=ctx_steps, table_ids=tids)
            t0 = time.perf_counter()
            dec = rans.decode(enc, table_ids=tids)
            t_dec += time.perf_counter() - t0
            if not all(np.array_equal(a, b) for a, b in zip(syms, dec)):
                raise AssertionError(f"context pass m={m} j={j} not bit-exact")
            bs = enc.stream_bytes(np.zeros(nt, np.int64))      # tables are derived, not stored
            for i, b in enumerate(bs):
                per_tensor[names[i]] += int(b)
            n_sym = sum(x.size for x in syms)
            ctx_bytes += int(bs.sum())
            lanes_total += int(enc.lanes.sum())
            curve.append({"m": m, "chunk": j, "symbols": n_sym, "bits_per_symbol": 8 * float(bs.sum()) / n_sym})
    # pair counting + table construction is part of the decoder's work
    t_dec += t_build
    return {"chunks": chunks, "ctx_steps": ctx_steps, "beta": beta, "order0_part_bytes": int(b0.sum()), "context_part_bytes": ctx_bytes,
            "total_bytes": int(b0.sum()) + ctx_bytes, "context_final_state_bytes": 4 * lanes_total,
            "decode_s": t_dec, "table_build_s": t_build, "bit_exact": True, "learning_curve": curve,
            "per_tensor_bytes": per_tensor,
            "note": "decode_s = order-0 part + context passes + pair counting and table rebuild"}


# ------------------------------------------------------------------------------------------- main

def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suc1", required=True)
    ap.add_argument("--out-dir", help="write <out-dir>/<timestamp>_<name>.json (suc.provenance.write_result)")
    ap.add_argument("--name", default="entropy_coding_qwen3_0.6b")
    ap.add_argument("--steps", type=int, default=8192, help="rANS interleave steps (lanes = n / steps per stream)")
    ap.add_argument("--fast", action="store_true", help="skip xz and zstd-19 on sections > 64 MB")
    ap.add_argument("--chunks", type=int, default=8, help="row chunks of the adaptive previous-codebook context model")
    ap.add_argument("--ctx-beta", type=int, default=256,
                    help="pseudo-count mass of the marginal in the context model (256: best of 16/64/256 on 14 tensors)")
    ap.add_argument("--ctx-steps", type=int, default=2048, help="rANS interleave steps of each context pass")
    args = ap.parse_args(argv)
    path = Path(args.suc1).expanduser()
    t_start = time.time()
    m = SucModel.load(path, verify=True)
    file_bytes = path.stat().st_size
    file_sha = json.load(open(path.parent / "container.json")).get("sha256") if (path.parent / "container.json").exists() else None
    params = sum(int(d["logical_size"]) for d in m._dir["tensors"])
    bits = m.bits()
    print(f"# {path.name}: {file_bytes} bytes, {params} params, {8 * file_bytes / params:.4f} bpp", flush=True)

    # ---- per-tensor entropy + collect rANS streams ----
    tensors, streams, alph, stream_meta = [], [], [], []
    pos_rows, additive = [], []
    for d in m._dir["tensors"]:
        rep = d["representation_type"]
        if rep == "raw":
            continue
        recs = {r["key"]: r for r in d["arrays"]}
        ikey = "base.indices" if "base.indices" in recs else "indices"
        idx = m._logical(recs[ikey])
        bits_idx = int(recs[ikey]["packed"]["bits"])
        k = 1 << bits_idx
        row = {"name": d["name"], "rep": rep, "codec": d["codec"], "shape": d["shape"], "index_bits": bits_idx,
               "array_bytes": {r["key"]: int(r["nbytes"]) for r in d["arrays"]}}
        if rep == "additive" or rep == "vq":
            per = []
            for j in range(idx.shape[0]):
                per.append(_index_stats(idx[j], k, idx[j - 1] if j > 0 else None))
                streams.append(np.ascontiguousarray(idx[j]).reshape(-1).astype(np.uint8 if k <= 256 else np.uint16))
                alph.append(k)
                stream_meta.append((d["name"], f"indices[{j}]", bits_idx))
            row["codebooks"] = per
            additive.append((d["name"], idx))
        else:   # affine: (out, in) codes; large tensors (the embedding): 16 evenly spaced 1024-row chunks
            rows_n = idx.shape[0]
            if idx.size > 1 << 24 and rows_n > 16 * 1024:
                starts = np.linspace(0, rows_n - 1024, 16).astype(int)
                row["codebooks"] = [_index_stats(idx[a:a + 1024], k, None) for a in starts]
                row["stats_subsampled_rows"] = 16 * 1024
            else:
                row["codebooks"] = [_index_stats(idx, k, None)]
            streams.append(np.ascontiguousarray(idx).reshape(-1).astype(np.uint8))
            alph.append(k)
            stream_meta.append((d["name"], "indices", bits_idx))
        if d["residual_type"] != "none" and d["params"].get("n", 0) > 0:
            p = d["params"]
            pos = m._logical(recs["res_pos"]).astype(np.int64)
            numel = int(d["shape"][0]) * int(d["shape"][1])
            gaps = np.diff(np.concatenate([[-1], pos]))
            r_best, rice = em.rice_bits(gaps)
            val = m._logical(recs["res_val"]).view(np.uint8) if "res_val" in recs else None
            row["residual"] = {"n": int(pos.size), "numel": numel, "position_bits": int(p["position_bits"]),
                               "pos_fixed_bits": int(pos.size * p["position_bits"]),
                               "pos_log2_binomial_bits": em.log2_binomial(numel, int(pos.size)),
                               "pos_rice_bits": rice, "rice_r": r_best,
                               "val_bits_fixed": None if val is None else int(8 * val.size),
                               "val_H0": None if val is None else em.order0(val, 256)}
            pos_rows.append((d["name"], gaps, r_best, pos))
            if val is not None and recs["res_val"]["dtype"] == "i8":
                streams.append(np.ascontiguousarray(val))
                alph.append(256)
                stream_meta.append((d["name"], "res_val", 8))
        tensors.append(row)
    print(f"# entropy stats: {len(tensors)} tensors, {len(streams)} rANS streams, {time.time() - t_start:.0f}s", flush=True)

    # ---- rANS ----
    freqs = np.zeros((len(streams), max(alph)), np.uint32)
    for i, s in enumerate(streams):
        freqs[i] = rans.quantize_freqs(np.bincount(s.astype(np.int64), minlength=freqs.shape[1]))
    enc, t_enc = _timeit(lambda: rans.encode(streams, freqs, steps=args.steps))
    dec, t_dec = _timeit(lambda: rans.decode(enc))
    exact = all(np.array_equal(a, b) for a, b in zip(streams, dec))
    if not exact:
        raise AssertionError("rANS decode is not bit-exact")
    del dec
    sb = enc.stream_bytes(np.array(alph))
    fixed_b = np.array([s.size * b / 8 for s, (_, _, b) in zip(streams, stream_meta)])
    is_idx = np.array([not k.startswith("res_") for _, k, _ in stream_meta])
    n_sym = sum(s.size for s in streams)
    out_bytes_fixed = float(fixed_b.sum())
    rans_row = {"streams": len(streams), "steps": enc.steps, "lanes": int(enc.lanes.sum()), "prob_bits": enc.prob_bits,
                "symbols": int(n_sym), "bit_exact": exact,
                "indices_fixed_bytes": float(fixed_b[is_idx].sum()), "indices_rans_bytes": int(sb[is_idx].sum()),
                "res_val_fixed_bytes": float(fixed_b[~is_idx].sum()), "res_val_rans_bytes": int(sb[~is_idx].sum()),
                "overhead_bytes": {"tables": int(2 * np.array(alph).sum()), "final_states": int(4 * enc.lanes.sum()),
                                   "stream_headers": 24 * len(streams)},
                "encode_s": t_enc, "decode_s": t_dec, "decode_MBps_fixed_width_output": out_bytes_fixed / t_dec / 1e6,
                "decode_Msym_per_s": n_sym / t_dec / 1e6,
                "note": "numpy-vectorized reference (one op per step over all lanes), single thread"}
    per_stream = {}
    for (name, key, _), b, fb in zip(stream_meta, sb, fixed_b):
        e = per_stream.setdefault(name, {"fixed_bytes": 0.0, "rans_bytes": 0})
        e["fixed_bytes"] += float(fb)
        e["rans_bytes"] += int(b)
    for row in tensors:
        if row["name"] in per_stream:
            row["rans"] = per_stream[row["name"]]
    print(f"# rANS: {rans_row['indices_fixed_bytes'] / 1e6:.2f} -> {rans_row['indices_rans_bytes'] / 1e6:.2f} MB "
          f"indices; decode {t_dec:.1f}s ({rans_row['decode_MBps_fixed_width_output']:.1f} MB/s)", flush=True)

    # ---- adaptive previous-codebook context model (additive indices) ----
    ctx_row = _prevcb_adaptive(additive, args.chunks, args.ctx_steps, args.steps, args.ctx_beta)
    fixed_add = sum(i.size for _, i in additive)          # 8-bit codes: bytes = symbols
    o0_add = sum(int(b) for (name, key, _), b in zip(stream_meta, sb) if key.startswith("indices[")
                 and any(name == a for a, _ in additive))
    ctx_row.update({"additive_fixed_bytes": fixed_add, "additive_order0_rans_bytes": o0_add,
                    "decode_MBps_fixed_width_output": fixed_add / ctx_row["decode_s"] / 1e6})
    for row in tensors:
        if row["name"] in ctx_row["per_tensor_bytes"]:
            row["prevcb_ctx_bytes"] = ctx_row["per_tensor_bytes"][row["name"]]
    print(f"# prevcb context: additive {fixed_add / 1e6:.2f} MB -> order-0 {o0_add / 1e6:.2f} -> context "
          f"{ctx_row['total_bytes'] / 1e6:.2f} MB; decode {ctx_row['decode_s']:.1f}s", flush=True)

    # ---- Rice on residual positions (verified) ----
    t_rice, rice_bytes, pos_fixed = 0.0, 0, 0
    rice_per = {}
    for name, gaps, r, pos in pos_rows:
        u, lo = em.rice_encode(gaps - 1, r)
        g2, td = _timeit(lambda: em.rice_decode(u, lo, r, gaps.size))
        t_rice += td
        if not np.array_equal(np.cumsum(g2 + 1) - 1, pos):
            raise AssertionError(f"{name}: Rice decode is not bit-exact")
        rice_bytes += u.size + lo.size + 8           # + r and n
        rice_per[name] = u.size + lo.size + 8
    for row in tensors:
        if "residual" in row:
            pos_fixed += -(-row["residual"]["pos_fixed_bits"] // 32) * 4
    rice_row = {"tensors": len(pos_rows), "pos_packed_bytes": pos_fixed, "pos_rice_bytes": rice_bytes,
                "decode_s": t_rice, "bit_exact": True,
                "decode_Mpos_per_s": sum(g.size for _, g, _, _ in pos_rows) / max(t_rice, 1e-9) / 1e6}

    # ---- generic coders per section ----
    blobs = _section_blobs(m)
    plane = {"scales_f16_byte_H0": _plane_entropy(blobs["scales"], 2),
             "codebooks_f16_byte_H0": _plane_entropy(blobs["codebooks"], 2),
             "raw_bf16_byte_H0": _plane_entropy(blobs["raw"], 2)}
    generic = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for s in ["directory", *SECTIONS]:
            print(f"# generic coders: {s} ({len(blobs[s]) / 1e6:.1f} MB) ...", flush=True)
            generic.extend(_generic(s, blobs[s], tmp, args.fast))
        whole = None
        if ZSTD:
            whole = bytes(m._mm)
            n3, tc3, td3 = _zstd(whole, 3, tmp)
            whole = {"coder": "zstd-3", "in_bytes": file_bytes, "out_bytes": n3, "compress_s": tc3, "decompress_s": td3,
                     "decode_MBps": file_bytes / td3 / 1e6}

    # ---- totals ----
    best_generic = {}
    for r in generic:
        b = best_generic.get(r["section"])
        if b is None or r["out_bytes"] < b["out_bytes"]:
            best_generic[r["section"]] = r
    fixed_overhead = HEADER_SIZE + FOOTER_SIZE
    before = {s: len(blobs[s]) for s in ["directory", *SECTIONS]}
    after_generic = {s: min(before[s], best_generic[s]["out_bytes"]) for s in before}
    # entropy-coded composite: rANS indices, rANS res values + Rice positions, generic best for the rest
    res_other = before["residual"] - pos_fixed - int(rans_row["res_val_fixed_bytes"])
    after_ec = dict(after_generic)
    after_ec["indices"] = min(before["indices"], rans_row["indices_rans_bytes"])
    after_ctx = dict(after_ec)
    after_ctx["indices"] = min(before["indices"], rans_row["indices_rans_bytes"] - o0_add + ctx_row["total_bytes"])
    after_ec["residual"] = min(before["residual"], rice_bytes + rans_row["res_val_rans_bytes"] + res_other)
    padding = bits["padding"] // 8

    def total(d, pad=0):
        return fixed_overhead + pad + sum(d.values())

    summary = {
        "params": params, "file_bytes": file_bytes, "bpp_file": 8 * file_bytes / params,
        "payload_bytes_no_padding": total(before), "bpp_no_padding": 8 * total(before) / params,
        "generic_best_bytes": total(after_generic), "bpp_generic_best": 8 * total(after_generic) / params,
        "entropy_coded_bytes": total(after_ec), "bpp_entropy_coded": 8 * total(after_ec) / params,
        "reduction_vs_file_generic": 1 - total(after_generic) / file_bytes,
        "reduction_vs_file_entropy_coded": 1 - total(after_ec) / file_bytes,
        "per_section_before": before, "per_section_generic_best": {s: [after_generic[s], best_generic[s]["coder"],
                                                                     best_generic[s]["variant"]] for s in before},
        "per_section_entropy_coded": after_ec, "padding_bytes_removed": padding,
        "context_coded_bytes": total(after_ctx), "bpp_context_coded": 8 * total(after_ctx) / params,
        "reduction_vs_file_context_coded": 1 - total(after_ctx) / file_bytes, "per_section_context_coded": after_ctx,
    }
    # additive (codec) tensors only: own payloads (indices, row scales, residual), codebooks excluded.
    # After = order-0 rANS indices + rANS res values + Rice positions; scales kept raw (conservative).
    a_before = a_after = 0
    for row in tensors:
        if row["rep"] != "additive":
            continue
        ab = row["array_bytes"]
        a_before += sum(ab.values())
        a_after += sum(v for k, v in ab.items() if k not in ("base.indices", "res_pos", "res_val"))
        a_after += row["rans"]["rans_bytes"] + rice_per.get(row["name"], ab.get("res_pos", 0))
    summary["additive_tensors_only"] = {"before_bytes": a_before, "after_bytes": a_after,
                                        "reduction": 1 - a_after / a_before}
    # load-time extrapolation to an 8B-sized SUC1 file (R1 config: every linear incl. embedding/head additive)
    bytes_8b = QWEN3_8B_PARAMS * BPP_8B / 8
    zrow = next((r for r in generic if r["section"] == "indices" and r["coder"] == "zstd-3" and r["variant"] == "plain"), None)
    extrap = {"assumed_8b_file_bytes": bytes_8b, "assumed": f"{QWEN3_8B_PARAMS:.3g} params x {BPP_8B} bpp (R1)",
              "rans_numpy_decode_s": bytes_8b / (rans_row["decode_MBps_fixed_width_output"] * 1e6),
              "zstd3_decode_s": None if zrow is None else bytes_8b / (zrow["decode_MBps"] * 1e6),
              "bytes_saved_8b_entropy_coded": bytes_8b * summary["reduction_vs_file_entropy_coded"],
              "bytes_saved_8b_additive_fraction": bytes_8b * summary["additive_tensors_only"]["reduction"],
              "bytes_saved_8b_context_coded": bytes_8b * summary["reduction_vs_file_context_coded"],
              "context_numpy_decode_s": bytes_8b / (ctx_row["decode_MBps_fixed_width_output"] * 1e6),
              "hot_path_note": "decoding inside the token loop would cost the full decode time per token "
                               "(every weight is read once per decode step)"}
    # aggregate entropies over all indices of each representation
    agg = {}
    for row in tensors:
        a = agg.setdefault(row["rep"], {"k_bits": row["index_bits"], "_n": {}})
        for c in row["codebooks"]:
            for key, v in c.items():
                if key != "n":      # each estimate weighted by the symbols it was measured on
                    a[key] = a.get(key, 0.0) + v * c["n"]
                    a["_n"][key] = a["_n"].get(key, 0) + c["n"]
    for a in agg.values():
        ns = a.pop("_n")
        for key, n in ns.items():
            a[key] /= n
        a["n"] = ns["H0"]
        a["H0_fraction_of_max"] = a["H0"] / a["k_bits"]
    res_tot = {k: sum(r["residual"][k] for r in tensors if "residual" in r)
               for k in ("n", "pos_fixed_bits", "pos_log2_binomial_bits", "pos_rice_bits", "val_bits_fixed")}
    res_tot["val_H0_weighted"] = sum(r["residual"]["val_H0"] * r["residual"]["n"] for r in tensors
                                     if "residual" in r) / max(1, res_tot["n"])

    from suc.provenance import run_metadata, write_result
    cfg = {"suc1": str(path), "steps": args.steps, "fast": args.fast, "prob_bits": rans.PROB_BITS,
           "chunks": args.chunks, "ctx_steps": args.ctx_steps, "ctx_beta": args.ctx_beta}
    md = run_metadata(model=m.metadata.get("model", "unknown"), model_revision=m.metadata.get("model_revision", ""),
                      model_hash=m.metadata.get("model_hash", "unknown"), dataset_hash="none", seed=0,
                      config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest())
    out = {"metadata": md, "benchmark": "entropy_coding", "config": cfg, "suc1_sha256": file_sha,
           "zstd_cli": None if not ZSTD else subprocess.run([ZSTD, "--version"], capture_output=True, text=True).stdout.strip(),
           "summary": summary, "index_entropy": agg, "residual_totals": res_tot, "byte_plane_entropy": plane,
           "rans": rans_row, "prevcb_context": ctx_row, "rice_positions": rice_row, "generic_coders": generic, "whole_file": whole,
           "extrapolation_8b": extrap, "tensors": tensors, "wall_s": time.time() - t_start}
    print(json.dumps({"summary": summary, "index_entropy": agg, "residual_totals": res_tot, "rans": rans_row,
                      "prevcb_context": {k: v for k, v in ctx_row.items() if k != "per_tensor_bytes"}, "rice": rice_row, "extrapolation_8b": extrap, "whole_file": whole}, indent=1, default=str))
    for r in generic:
        print(f"{r['section']:10s} {r['variant']:8s} {r['coder']:8s} {r['in_bytes']:>11d} -> {r['out_bytes']:>11d} "
              f"({r['out_bytes'] / max(1, r['in_bytes']):.4f})  dec {r['decode_MBps']:.0f} MB/s")
    if args.out_dir:
        print(f"# wrote {write_result(out, args.out_dir, args.name)}")
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
