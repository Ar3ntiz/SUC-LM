"""K2/K4 timing: SucLinear (vq / additive, with and without rotation) vs dense fp16 and MLX 4-bit affine.

Not a test. Usage: .venv/bin/python benchmarks/k2_suc_linear.py [--json out.json] [--quick] [--samples N]
Each sample evaluates `reps` independent calls in one mx.eval (amortizes launch/sync overhead); the reported time
is the median over samples of (sample time / reps), the minimum is printed too (useful on a shared GPU).
Effective bandwidth = weight-representation bytes / median time (x, y and rotation traffic excluded).
Codebooks/indices are random (timing only; numerics are covered by tests/unit/test_suc_linear.py).

Methods per (layer, batch):
  fp16_dense            x @ Wᵀ
  mlx_affine_q4_g64     mx.quantized_matmul, 4-bit affine, group 64 (4.5 bpw + metadata)
  <suc>_k1              K1 lookup kernel (vq only, reference point for the K2 changes)
  <suc>_kernel          K2 lookup kernel alone on fp16 x (no rotation), codebook read from device memory (default)
  <suc>_kernel_tg       same kernel with the codebooks staged in threadgroup memory (placement comparison)
  <suc>_full            SucLinear.__call__ incl. V x and Uᵀ ỹ (fast Hadamard, float32) around the kernel
  rotation_only         fast_rht(x) + fast_rht_inverse(ỹ), no kernel — the rotation overhead
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time

import mlx.core as mx
import numpy as np

from suc.kernels.suc_linear import SucLinear, fast_rht, fast_rht_inverse
from suc.kernels.vq_matmul import additive_matmul_metal, vq_matmul_metal
from suc.quant.hadamard import random_signs
from suc.quant.packing import pack_indices

SHAPES = [(12288, 4096), (4096, 12288), (4096, 4096), (3072, 1024)]   # (out, in): 8B MLP up/down, 8B attn, 0.6B MLP
BATCHES = [1, 4, 16]
SUC = {"vq_G8_K256": (1, 8, 256), "vq_G4_K256": (1, 4, 256),           # name: (M, G, K)
       "add2_G8_K256": (2, 8, 256), "add3_G8_K256": (3, 8, 256)}
TG_COMPARE = ("vq_G8_K256", "add2_G8_K256")   # also time the threadgroup-staged codebook variant


def bench(fn, samples: int, reps: int = 20, warmup: int = 5) -> tuple[float, float]:
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(samples):
        t0 = time.perf_counter()
        mx.eval([fn() for _ in range(reps)])
        ts.append((time.perf_counter() - t0) / reps)
    return statistics.median(ts), min(ts)


def nbytes(*arrs: mx.array) -> int:
    return sum(a.nbytes for a in arrs)


def make_layer(rng: np.random.Generator, out: int, inp: int, m: int, g: int, k: int) -> SucLinear:
    bits = int(np.log2(k))
    cb = mx.array(rng.normal(size=(m, k, g)).astype(np.float16))
    pk = mx.stack([pack_indices(mx.array(rng.integers(0, k, size=(out, inp // g), dtype=np.uint32)), bits)
                   for _ in range(m)])
    s = mx.array(rng.uniform(0.01, 0.1, size=(out,)).astype(np.float32))
    layer = SucLinear(cb, pk, s, bits=bits, in_features=inp, seed_in=1, seed_out=2)
    mx.eval(layer.parameters())
    return layer


def busy_processes() -> str:
    try:
        return subprocess.run(["pgrep", "-fl", "suc quantize|python"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="batch 1 only, fewer samples")
    args = ap.parse_args()
    batches, samples = ([1], min(args.samples, 11)) if args.quick else (BATCHES, args.samples)
    rng = np.random.default_rng(0)
    rows = []
    print(f"# device={mx.device_info().get('device_name', platform.machine())} mlx={mx.__version__}")
    print(f"{'layer':>12} {'B':>3} {'method':<26} {'median_us':>10} {'min_us':>9} {'bpw':>7} {'GB/s':>7}")
    for out, inp in SHAPES:
        w = mx.random.normal((out, inp), key=mx.random.key(0)).astype(mx.float16) * 0.02
        wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
        layers = {name: make_layer(rng, out, inp, *cfg) for name, cfg in SUC.items()}
        mx.eval(w, wq, sc, bi)
        s_in, s_out = random_signs(inp, 1), random_signs(out, 2)
        for b in batches:
            x = mx.random.normal((b, inp), key=mx.random.key(1)).astype(mx.float16)
            yt = mx.random.normal((b, out), key=mx.random.key(2))
            mx.eval(x, yt)
            cands = {"fp16_dense": (lambda: x @ w.T, nbytes(w)),
                     "mlx_affine_q4_g64": (lambda: mx.quantized_matmul(x, wq, sc, bi, transpose=True,
                                                                       group_size=64, bits=4), nbytes(wq, sc, bi))}
            for name, L in layers.items():
                wb = nbytes(L.codebooks, L.packed_indices, L.row_scales)
                if L.codebooks.shape[0] == 1:
                    cands[f"{name}_k1"] = (lambda L=L: vq_matmul_metal(
                        x, L.codebooks[0], L.packed_indices[0], L.row_scales, bits=L.bits, in_features=inp,
                        strategy="lookup_k1"), wb)
                cands[f"{name}_kernel"] = (lambda L=L: L.kernel(x), wb)
                if name in TG_COMPARE:
                    cands[f"{name}_kernel_tg"] = (lambda L=L: additive_matmul_metal(
                        x, L.codebooks, L.packed_indices, L.row_scales, bits=L.bits, in_features=inp,
                        codebook_in="threadgroup"), wb)
                cands[f"{name}_full"] = (lambda L=L: L(x), wb)
            cands["rotation_only"] = (lambda: (fast_rht(x, s_in), fast_rht_inverse(yt, s_out)), 0)
            for name, (fn, wb) in cands.items():
                med, mn = bench(fn, samples)
                r = dict(out=out, inp=inp, batch=b, method=name, time_us=med * 1e6, min_us=mn * 1e6,
                         weight_bytes=wb, bpw=8 * wb / (out * inp), gbps=wb / med / 1e9)
                rows.append(r)
                print(f"{out:>6}x{inp:<5} {b:>3} {name:<26} {r['time_us']:10.1f} {r['min_us']:9.1f} "
                      f"{r['bpw']:7.3f} {r['gbps']:7.1f}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(device=mx.device_info().get("device_name", platform.machine()), mlx=mx.__version__,
                           samples=samples, other_processes=busy_processes(), rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
