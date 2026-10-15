"""K1 timing: direct VQ matmul (lookup / table) vs dense fp16 matmul and MLX 4-bit affine quantized_matmul.

Not a test. Usage: .venv/bin/python benchmarks/k1_vq_matmul.py [--json out.json]
Each sample evaluates `reps` independent calls in one mx.eval (amortizes launch/sync overhead); the reported
time is the median over samples of (sample time / reps). Effective bandwidth = weight-representation bytes /
time (x and y traffic excluded). Laptop numbers are indicative only.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import mlx.core as mx
import numpy as np

from suc.kernels.vq_matmul import vq_matmul_metal
from suc.quant.packing import pack_indices

SHAPES = [(3072, 1024), (1024, 3072), (12288, 4096)]   # (out, in): Qwen3-0.6B MLP, 8B-sized MLP
BATCHES = [1, 16]
CONFIGS = [(8, 256), (4, 256)]                         # (G, K): 1 bpw and 2 bpw of index


def bench(fn, samples: int = 30, reps: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(samples):
        t0 = time.perf_counter()
        mx.eval([fn() for _ in range(reps)])
        ts.append((time.perf_counter() - t0) / reps)
    return statistics.median(ts)


def nbytes(*arrs: mx.array) -> int:
    return sum(a.nbytes for a in arrs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for out, inp in SHAPES:
        w = mx.random.normal((out, inp), key=mx.random.key(0)).astype(mx.float16) * 0.02
        wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
        mx.eval(w, wq, sc, bi)
        for b in BATCHES:
            x = mx.random.normal((b, inp), key=mx.random.key(1)).astype(mx.float16)
            mx.eval(x)
            cands = {"fp16_dense": (lambda: x @ w.T, nbytes(w)),
                     "mlx_affine_q4_g64": (lambda: mx.quantized_matmul(x, wq, sc, bi, transpose=True,
                                                                       group_size=64, bits=4), nbytes(wq, sc, bi))}
            for g, k in CONFIGS:
                bits = int(np.log2(k))
                cb = mx.array(rng.normal(size=(k, g)).astype(np.float16))
                pk = pack_indices(mx.array(rng.integers(0, k, size=(out, inp // g), dtype=np.uint32)), bits)
                s = mx.array(rng.uniform(0.01, 0.1, size=(out,)).astype(np.float32))
                mx.eval(cb, pk, s)
                for st in ("lookup", "table"):
                    cands[f"suc_{st}_G{g}_K{k}"] = (
                        lambda cb=cb, pk=pk, s=s, bits=bits, st=st:
                        vq_matmul_metal(x, cb, pk, s, bits=bits, in_features=inp, strategy=st),
                        nbytes(cb, pk, s))
            for name, (fn, wb) in cands.items():
                t = bench(fn)
                rows.append(dict(out=out, inp=inp, batch=b, method=name, time_us=t * 1e6, weight_bytes=wb,
                                 bpw=8 * wb / (out * inp), gbps=wb / t / 1e9))
                r = rows[-1]
                print(f"{out:>6}x{inp:<5} B={b:<3} {name:<22} {r['time_us']:9.1f} us  "
                      f"{r['bpw']:6.3f} bpw  {r['gbps']:7.1f} GB/s", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(device=platform.machine(), mlx=mx.__version__, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
