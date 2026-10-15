"""Single-linear benchmark (spec §43) of the TCQ kernel (suc.kernels.tcq_matmul) vs the VQ lookup kernel and MLX
affine quantized_matmul: median µs per call over cycled weight copies (cold-ish cache), GB/s of weight bytes read.
Random codes (timing does not depend on the values). Usage: python benchmarks/tcq_kernel_speed.py [--batches 1,4]
Writes a result JSON (provenance) to experiments/results unless --no-write."""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suc.kernels.tcq_matmul import tcq_matmul_metal          # noqa: E402
from suc.kernels.vq_matmul import vq_matmul_metal            # noqa: E402
from suc.provenance import run_metadata, write_result        # noqa: E402

SHAPES = [(12288, 4096), (4096, 12288)]                       # (out, in): Qwen3-8B gate/up and down


def _time(fn, copies, reps):
    for c in copies[:2]:
        mx.eval(fn(*c))
    ts = []
    for i in range(reps):
        c = copies[i % len(copies)]
        t = time.perf_counter()
        mx.eval(fn(*c))
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--copies", type=int, default=6)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for out, inn in SHAPES:
        n = out * inn
        for b in [int(v) for v in a.batches.split(",")]:
            x = mx.random.normal((b, inn), key=mx.random.key(b)).astype(mx.float16)
            cases = {}
            # VQ G=4 K=256 (8-bit indices) and G=2 K=256 (4 bpw), the current low / mid tiers
            for g in (4, 2):
                cb = mx.random.normal((256, g)).astype(mx.float16)
                cps = [(mx.array(rng.integers(0, 2**32, (out, inn * 8 // g // 32), dtype=np.uint32)),) for _ in range(a.copies)]
                s = mx.ones((out,), mx.float16)
                cases[f"vq_g{g}_k256"] = (lambda w, cb=cb, s=s, x=x: vq_matmul_metal(x, cb, w, s, bits=8, in_features=inn,
                                                                                    strategy="lookup"), cps, n * 8 / g / 8)
            for code, k, V in [("1mad", 2, 1), ("1mad", 2, 2), ("3inst", 2, 1), ("3inst", 2, 2), ("pair3", 2, 2),
                               ("1mad", 2.5, 2), ("3inst", 2.5, 2), ("pair3", 2.5, 2), ("1mad", 3, 1), ("3inst", 3, 2),
                               ("pair3", 3, 2)]:
                nw = int(256 * k) // 32
                s = mx.ones((out,), mx.float16)
                cps = [(mx.array(rng.integers(0, 2**32, (out, inn // 256 * nw), dtype=np.uint32)),) for _ in range(a.copies)]
                cases[f"tcq_{code}_k{k:g}_v{V}"] = (
                    lambda w, s=s, x=x, k=k, V=V, code=code: tcq_matmul_metal(x, w, s, k=k, L=16, V=V, T=256, code=code,
                                                                             gain=1.0), cps, n * k / 8)
            for bits in (2, 3, 4):
                wq = [mx.quantize(mx.random.normal((out, inn)).astype(mx.float16), group_size=64, bits=bits)
                      for _ in range(a.copies)]
                cases[f"mlx_affine_q{bits}"] = (lambda w, sc, bi, x=x, bits=bits: mx.quantized_matmul(
                    x, w, sc, bi, transpose=True, group_size=64, bits=bits), wq, n * (bits + 0.5) / 8)
            for name, (fn, cps, nbytes) in cases.items():
                us = _time(fn, cps, a.reps)
                rows.append({"shape": [out, inn], "batch": b, "kernel": name, "us_median": us,
                             "weight_bytes": nbytes, "gbps": nbytes / us / 1e3})
                print(f"{out}x{inn} B={b} {name:24s} {us:8.1f} us {nbytes / us / 1e3:6.1f} GB/s", flush=True)
    if not a.no_write:
        meta = run_metadata(model="none (random codes)", model_revision=None, model_hash=None, dataset_hash=None,
                            seed=0, config_hash=None)
        print(write_result({"metadata": meta, "name": "tcq_kernel_speed", "rows": rows,
                            "protocol": f"median of {a.reps} calls cycling {a.copies} weight copies, fp16 x"},
                           "experiments/results", "tcq_kernel_speed"))


if __name__ == "__main__":
    main()
