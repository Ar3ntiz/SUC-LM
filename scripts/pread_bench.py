#!/usr/bin/env python3
"""Random-read bandwidth of the local SSD for expert streaming (docs/expert_streaming.md).

Reads `--chunk`-sized blocks (default 2.6 MB = one Qwen3.8 expert at 4.2 bpp) at random aligned offsets of a file
with `--threads` concurrent readers, for each IO mode:
  pread_nocache  os.preadv on a descriptor with F_NOCACHE (macOS): bypasses the unified buffer cache, so the numbers
                 are the device bandwidth even when the file was read before;
  pread          os.preadv through the page cache (repeated offsets can hit RAM);
  mmap           memcpy out of a read-only mapping (page faults, 16 KB pages + read-ahead).
Only offsets never read before in the run are used for the cached modes, but a file smaller than RAM may still be
cached from earlier runs: use F_NOCACHE numbers as the device bandwidth, or a file larger than RAM.

  python scripts/pread_bench.py --file /path/model.suc1                    # an existing big file (read-only)
  python scripts/pread_bench.py --create /tmp/bench.bin --size-gb 8         # writes a test file first
Prints one line per (mode, threads) and, with --json, writes the results (with machine metadata) there.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

F_NOCACHE = 48


def _open(path: Path, nocache: bool) -> int:
    fd = os.open(path, os.O_RDONLY)
    if nocache and sys.platform == "darwin":
        import fcntl
        fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", F_NOCACHE), 1)
    return fd


def _pread(fd: int, buf: np.ndarray, off: int) -> None:
    mv, done = memoryview(buf), 0
    while done < buf.size:
        if hasattr(os, "preadv"):
            n = os.preadv(fd, [mv[done:]], off + done)
        else:
            b = os.pread(fd, buf.size - done, off + done)
            n = len(b)
            mv[done:done + n] = b
        if n <= 0:
            raise IOError("short read")
        done += n


def run(path: Path, mode: str, threads: int, chunk: int, total: int, seed: int) -> dict:
    size = path.stat().st_size
    align = 16384
    n = max(1, total // chunk)
    rng = np.random.default_rng(seed)
    slots = (size - chunk) // align
    offs = (rng.choice(slots, size=min(n, slots), replace=False) * align).astype(np.int64)
    bufs = [np.empty(chunk, np.uint8) for _ in range(threads)]
    lat: list[float] = []
    if mode == "mmap":
        mm = np.memmap(path, dtype=np.uint8, mode="r")

        def one(i: int) -> float:
            t = time.perf_counter()
            np.copyto(bufs[i % threads], mm[offs[i]:offs[i] + chunk])
            return time.perf_counter() - t
    else:
        fd = _open(path, mode == "pread_nocache")

        def one(i: int) -> float:
            t = time.perf_counter()
            _pread(fd, bufs[i % threads], int(offs[i]))
            return time.perf_counter() - t
    t0 = time.perf_counter()
    if threads == 1:
        lat = [one(i) for i in range(offs.size)]
    else:
        # thread i handles requests i, i+T, …: each owns one buffer, no sharing
        def worker(w: int) -> list[float]:
            return [one(i) for i in range(w, offs.size, threads)]
        with ThreadPoolExecutor(threads) as ex:
            for part in ex.map(worker, range(threads)):
                lat.extend(part)
    dt = time.perf_counter() - t0
    if mode != "mmap":
        os.close(fd)
    la = np.array(lat) * 1e3
    return {"mode": mode, "threads": threads, "chunk_bytes": chunk, "reads": int(offs.size),
            "gb_per_s": offs.size * chunk / dt / 1e9, "ms_p50": float(np.median(la)), "ms_p99": float(np.percentile(la, 99))}


def _machine() -> dict:
    chip = ""
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    return {"platform": platform.platform(), "chip": chip, "hostname": platform.node()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path, help="existing file to read (never written)")
    src.add_argument("--create", type=Path, help="create a random test file here first (removed unless --keep)")
    ap.add_argument("--size-gb", type=float, default=4.0, help="size of the --create file")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--chunk-mb", type=float, default=2.58, help="read size (MB, 1e6); default one 4.2-bpp expert")
    ap.add_argument("--total-gb", type=float, default=2.0, help="bytes read per (mode, threads)")
    ap.add_argument("--threads", default="1,4,8,16")
    ap.add_argument("--modes", default="pread_nocache,pread,mmap")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    path = a.file or a.create
    if a.create:
        blk = np.random.default_rng(0).integers(0, 256, 64 << 20, dtype=np.uint8).tobytes()
        with open(path, "wb") as f:
            for _ in range(int(a.size_gb * 1e9) // len(blk) + 1):
                f.write(blk)
            f.flush()
            os.fsync(f.fileno())
    chunk = int(a.chunk_mb * 1e6) // 4096 * 4096
    res = []
    try:
        for seed, mode in enumerate(a.modes.split(",")):
            for t in (int(x) for x in a.threads.split(",")):
                r = run(path, mode, t, chunk, int(a.total_gb * 1e9), seed * 100 + t)
                res.append(r)
                print(f"{r['mode']:14s} T={t:3d} chunk={chunk / 1e6:6.2f} MB  {r['gb_per_s']:6.2f} GB/s  "
                      f"p50 {r['ms_p50']:7.2f} ms  p99 {r['ms_p99']:7.2f} ms  ({r['reads']} reads)", flush=True)
    finally:
        if a.create and not a.keep:
            path.unlink(missing_ok=True)
    if a.json:
        a.json.write_text(json.dumps({"benchmark": "pread_bench", "file_bytes": path.stat().st_size if path.exists()
                                      else int(a.size_gb * 1e9), "machine": _machine(), "results": res}, indent=1))


if __name__ == "__main__":
    main()
