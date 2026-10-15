"""K3 cold-cache timing: y = x Ŵᵀ streamed from DRAM, not from the SLC / L2 (fixes the K1/K2 warm-cache bias).

Not a test. Usage:
  .venv/bin/python benchmarks/k3_cold_cache.py [--gb 2] [--json out.json] [--samples 5] [--batches 1,4,16]
         [--shapes 8b,30b,0.6b] [--methods substr,...] [--no-layers] [--no-decode] [--no-vec-compare]

Why: K1/K2 re-read one matrix, so FP16 reached 1043 GB/s on the M1 Ultra (> ~800 GB/s DRAM) — results were partly
served from the system-level cache, hiding the bytes-read advantage SUC exists for. Here every method gets N
distinct copies of its weight representation and the timed loop cycles through all of them.

Copies: per (shape, method) N = max(ceil(gb·1e9 / FP16 bytes of one copy), ceil(min_mb·1e6 / bytes of one copy of
this method)): the FP16 copies together exceed --gb (default 2 GB; use --gb 8..32 on the 128 GB M1 Ultra), and
every method streams at least --min-mb (default 512 MB, ≫ the 96 MB SLC of an M1 Ultra) per cycle.
Representations are random (timing only; numerics are covered by tests/unit/test_vq_matmul.py and
test_suc_linear.py); packed indices are random uint32 words, valid for K = 2**bits.

Timing ("layers" mode): one sample = one full cycle over the N copies with the same x, graph built first, then
ONE mx.eval over the N outputs timed (Metal encode + GPU execution; CPU graph construction excluded).
µs/call = sample time / N; reported median and min over --samples cycles after one warm-up cycle. The N calls are
independent, so MLX may overlap neighbouring kernels (hides launch / tail latency; cannot exceed DRAM bandwidth).
GB/s = representation bytes of one copy (codebooks + indices + scales, or q + scales + biases) / median µs/call.

"decode" mode (closest to real decoding): N distinct Qwen3-8B transformer blocks (7 projections each:
q 4096x4096, k/v 1024x4096, o 4096x4096, gate/up 12288x4096, down 4096x12288) applied as a dependency chain
h → q,k,v(h); a = o(q); gate,up(a); h' = down(gate) → next block (attention / SiLU omitted, no extra kernels),
one mx.eval per cycle over all N blocks; µs per block. Kernels of consecutive projections in the chain cannot
overlap, as in a real decode step (q/k/v and gate/up may, as they could in a real model).

Methods: fp16_dense, mlx_affine_q{4,3,2}_g64 (mx.quantized_matmul), <suc>_kernel (Metal kernel only, fp16 x,
auto path: vec for B < MMA_MIN_BATCH, simdgroup-matrix for B ≥ MMA_MIN_BATCH), at every B ≥ 4 also the path auto
does not take: <suc>_vec (B ≥ MMA_MIN_BATCH) or <suc>_mma (B < MMA_MIN_BATCH) — the A/B of MMA_MIN_BATCH —
<suc>_table (Strategy D, vq_matmul strategy "table", single codebooks with K ≤ TABLE_MAX_K, with the A/B rows)
and <suc>_full (SucLinear incl. Hadamard rotations V x and Uᵀ ỹ). SUC configurations need in_features % G == 0:
G = 3 rows exist for 12288-wide inputs only (layers mode), and no G = 3 decode block.
Random representations are scaled to be ≈ norm-preserving (row std ≈ 1/√in): with arbitrary scales the decode
chain overflowed to inf/NaN after one block, which slowed mx.quantized_matmul ~4x (q4 B=4 6679 vs 1687 µs/block).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import statistics
import subprocess
import time
from typing import Callable

import mlx.core as mx

import suc.kernels.vq_matmul as vm
from suc.kernels.suc_linear import SucLinear

SHAPES = {  # name: [(out, in, label)]
    "8b": [(12288, 4096, "8B mlp gate/up"), (4096, 12288, "8B mlp down"), (4096, 4096, "8B attn q/o")],
    "30b": [(4096, 2048, "30B-A3B attn q"), (2048, 4096, "30B-A3B attn o"),
            (768, 2048, "30B-A3B expert gate/up"), (2048, 768, "30B-A3B expert down")],
    "0.6b": [(3072, 1024, "0.6B mlp gate/up")],
}
BLOCK_8B = {"q": (4096, 4096), "k": (1024, 4096), "v": (1024, 4096), "o": (4096, 4096),
            "gate": (12288, 4096), "up": (12288, 4096), "down": (4096, 12288)}
SUC = {"vq_G8_K256": (1, 8, 256), "vq_G4_K256": (1, 4, 256),           # name: (M, G, K)
       "add2_G8_K256": (2, 8, 256), "add3_G8_K256": (3, 8, 256),
       # low-dimensional single codebooks (Q5c): 4 / 3 / 2 bpw at G = 2, 3 / 2 bpw at G = 3 (9- and 6-bit indices)
       "vq_G2_K256": (1, 2, 256), "vq_G2_K64": (1, 2, 64), "vq_G2_K16": (1, 2, 16),
       "vq_G3_K512": (1, 3, 512), "vq_G3_K64": (1, 3, 64)}
TABLE_MAX_K = 64   # <suc>_table rows (Strategy D partial-product tables) for single codebooks with K <= this
AFFINE_BITS = (4, 3, 2)
SEED = 12345

Layer = Callable[[mx.array], mx.array]


# ------------------------------------------------------------------------------------- representations

def _affine_shapes(out: int, inp: int, bits: int) -> list[tuple[tuple[int, ...], mx.Dtype]]:
    q = mx.quantize(mx.zeros((out, inp), dtype=mx.float16), group_size=64, bits=bits)
    return [(a.shape, a.dtype) for a in q]


def random_words(shape: tuple[int, ...], key: mx.array) -> mx.array:
    """Uniform random uint32 words (two exact 16-bit halves; randint over 2**32 would zero the low bits)."""
    ka, kb = mx.random.split(key)
    hi = mx.random.randint(0, 1 << 16, shape, dtype=mx.uint32, key=ka)
    lo = mx.random.randint(0, 1 << 16, shape, dtype=mx.uint32, key=kb)
    return (hi << 16) | lo


def make_copy(method: str, out: int, inp: int, key: mx.array) -> tuple[Layer, list[mx.array]]:
    """One random copy of `method` for an (out, in) layer → (callable x ↦ y, representation arrays)."""
    k1, k2, k3 = mx.random.split(key, 3)
    sd = 1.0 / math.sqrt(inp)            # every representation ≈ norm-preserving: chained activations stay finite
    if method == "fp16_dense":
        w = (mx.random.normal((out, inp), key=k1) * sd).astype(mx.float16)
        return (lambda x: x @ w.T), [w]
    if method.startswith("mlx_affine_q"):
        bits = int(method[len("mlx_affine_q")])
        (qs, _), (ss, st), _ = _affine_shapes(out, inp, bits)
        wq = random_words(qs, k1)
        lv = (1 << bits) - 1                # uniform codes 0..lv: std sqrt(((lv+1)^2 - 1) / 12), centred by the bias
        s0 = sd / math.sqrt(((lv + 1) ** 2 - 1) / 12)
        sc = mx.random.uniform(0.9 * s0, 1.1 * s0, ss, key=k2).astype(st)
        bi = (-lv / 2 * sc.astype(mx.float32)).astype(st)
        del k3
        return (lambda x: mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=bits)), [wq, sc, bi]
    name, variant = method.rsplit("_", 1)
    if name not in SUC or variant not in ("kernel", "vec", "mma", "full", "table"):
        raise ValueError(f"unknown method {method!r}")
    m, g, k = SUC[name]
    bits = int(math.log2(k))
    cb = mx.random.normal((m, k, g), key=k1).astype(mx.float16)
    pk = random_words((m, out, -(-(inp // g) * bits // 32)), k2)
    s = mx.random.uniform(0.9, 1.1, (out,), key=k3) * (sd / math.sqrt(m))   # Σ of M N(0,1) entries: std √M
    layer = SucLinear(cb, pk, s, bits=bits, in_features=inp, seed_in=1, seed_out=2)
    reps = [layer.codebooks, layer.packed_indices, layer.row_scales]
    if variant == "kernel":
        return layer.kernel, reps
    if variant in ("vec", "mma"):
        return (lambda x: layer.kernel(x, path=variant)), reps
    if variant == "table":
        cb1, pk1 = layer.codebooks[0], layer.packed_indices[0]
        return (lambda x: vm.vq_matmul_metal(x, cb1, pk1, layer.row_scales, bits=bits, in_features=inp,
                                             strategy="table")), reps
    return layer, reps


def methods(batch: int, select: list[str] | None, vec_compare: bool) -> list[str]:
    ms = ["fp16_dense"] + [f"mlx_affine_q{b}_g64" for b in AFFINE_BITS]
    for name in SUC:
        ms.append(f"{name}_kernel")
        if vec_compare and batch >= 4:   # the path "auto" does NOT take at this batch (A/B of MMA_MIN_BATCH)
            ms.append(f"{name}_vec" if batch >= vm.mma_min_batch(1) else f"{name}_mma")
        if vec_compare and SUC[name][0] == 1 and SUC[name][2] <= TABLE_MAX_K:
            ms.append(f"{name}_table")
        ms.append(f"{name}_full")
    return [m for m in ms if not select or any(s in m for s in select)]


def applicable(method: str, inp: int) -> bool:
    """SUC methods need in_features % G == 0 (G = 3 only fits 12288-wide inputs among the 8B shapes)."""
    name = method.rsplit("_", 1)[0]
    return name not in SUC or inp % SUC[name][1] == 0


def nbytes(arrs: list[mx.array]) -> int:
    return sum(a.nbytes for a in arrs)


def n_copies(fp16_bytes: int, method_bytes: int, gb: float, min_mb: float) -> int:
    return max(math.ceil(gb * 1e9 / fp16_bytes), math.ceil(min_mb * 1e6 / method_bytes), 1)


# ------------------------------------------------------------------------------------------------ timing

def time_cycles(build: Callable[[], list[mx.array]], samples: int) -> list[float]:
    """Seconds per full cycle: graph built outside the timer, one mx.eval over all outputs inside."""
    mx.eval(build())                                  # warm-up cycle (kernel compilation, first touch)
    ts = []
    for _ in range(samples):
        outs = build()
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(outs)
        ts.append(time.perf_counter() - t0)
    return ts


def release() -> None:
    """Call after dropping the last reference to a method's copies: returns their buffers to the OS."""
    mx.synchronize()
    mx.clear_cache()


def run_layers(shapes, batches, select, vec_compare, gb, min_mb, samples, rows) -> None:
    print(f"{'layer':>12} {'B':>3} {'method':<26} {'N':>5} {'median_us':>10} {'min_us':>9} {'bpw':>7} {'GB/s':>7}")
    key = mx.random.key(SEED)
    for out, inp, label in shapes:
        fp16_bytes = out * inp * 2
        xs = {b: mx.random.normal((b, inp), key=mx.random.key(b)).astype(mx.float16) for b in batches}
        mx.eval(list(xs.values()))
        all_methods = list(dict.fromkeys(m for b in batches for m in methods(b, select, vec_compare)))
        for method in [m for m in all_methods if applicable(m, inp)]:
            proto_fn, proto = make_copy(method, out, inp, key)
            per_copy = nbytes(proto)
            n = n_copies(fp16_bytes, per_copy, gb, min_mb)
            copies = [make_copy(method, out, inp, mx.random.key(SEED + 1000 * len(method) + i)) for i in range(n)]
            mx.eval([a for _, arrs in copies for a in arrs])
            del proto_fn, proto
            for b in batches:
                if method not in methods(b, select, vec_compare):
                    continue
                x = xs[b]
                ts = time_cycles(lambda: [fn(x) for fn, _ in copies], samples)
                med, mn = statistics.median(ts) / n, min(ts) / n
                r = dict(mode="layers", label=label, out=out, inp=inp, batch=b, method=method, copies=n,
                         time_us=med * 1e6, min_us=mn * 1e6, weight_bytes=per_copy, total_bytes=per_copy * n,
                         bpw=8 * per_copy / (out * inp), gbps=per_copy / med / 1e9, samples_us=[t / n * 1e6 for t in ts])
                rows.append(r)
                print(f"{out:>6}x{inp:<5} {b:>3} {method:<26} {n:>5} {r['time_us']:10.1f} {r['min_us']:9.1f} "
                      f"{r['bpw']:7.3f} {r['gbps']:7.1f}", flush=True)
            copies = None
            release()


def run_decode(batches, select, vec_compare, gb, samples, rows) -> None:
    print(f"\n# decode-like: Qwen3-8B blocks (7 projections) as a dependency chain, µs per block")
    print(f"{'B':>3} {'method':<26} {'N':>5} {'median_us':>10} {'min_us':>9} {'MB/block':>9} {'GB/s':>7}")
    fp16_block = sum(o * i * 2 for o, i in BLOCK_8B.values())
    n = max(math.ceil(gb * 1e9 / fp16_block), 2)
    key = mx.random.key(SEED + 1)
    all_methods = list(dict.fromkeys(m for b in batches for m in methods(b, select, vec_compare)))
    for method in [m for m in all_methods if all(applicable(m, i) for _, i in BLOCK_8B.values())]:
        blocks, per_block = [], 0
        for i in range(n):
            blk = {}
            for j, (p, (o, inp)) in enumerate(BLOCK_8B.items()):
                blk[p] = make_copy(method, o, inp, mx.random.key(SEED + 1 + i * 16 + j))
            blocks.append(blk)
        mx.eval([a for blk in blocks for _, arrs in blk.values() for a in arrs])
        per_block = sum(nbytes(arrs) for _, arrs in blocks[0].values())

        def chain(h: mx.array) -> list[mx.array]:
            side = []
            for blk in blocks:
                q = blk["q"][0](h)
                side += [blk["k"][0](h), blk["v"][0](h)]
                a = blk["o"][0](q)
                g = blk["gate"][0](a)
                side.append(blk["up"][0](a))
                h = blk["down"][0](g)
            return side + [h]

        for b in batches:
            if method not in methods(b, select, vec_compare):
                continue
            h0 = mx.random.normal((b, 4096), key=mx.random.key(b)).astype(mx.float16)
            ts = time_cycles(lambda: chain(h0), samples)
            med, mn = statistics.median(ts) / n, min(ts) / n
            r = dict(mode="decode_block", label="Qwen3-8B block", batch=b, method=method, copies=n,
                     time_us=med * 1e6, min_us=mn * 1e6, weight_bytes=per_block, total_bytes=per_block * n,
                     bpw=8 * per_block / (fp16_block / 2), gbps=per_block / med / 1e9,
                     samples_us=[t / n * 1e6 for t in ts])
            rows.append(r)
            print(f"{b:>3} {method:<26} {n:>5} {r['time_us']:10.1f} {r['min_us']:9.1f} {per_block / 1e6:9.1f} "
                  f"{r['gbps']:7.1f}", flush=True)
        blocks = chain = None
        release()


def busy_processes() -> str:
    try:
        return subprocess.run(["pgrep", "-fl", "suc quantize|python|strata"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


def kernel_config() -> dict:
    return {"mma_min_batch": vm.mma_min_batch(1),
            **{k: getattr(vm, k) for k in ("MMA_MIN_BATCH", "MMA_SG", "MMA_RT", "MMA_KC", "MMA_TILE", "LOOKUP_SG",
                                           "LOOKUP_VEC", "LOOKUP_ROWS", "CODEBOOK_IN", "MAX_BATCH_TILE")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=2.0, help="FP16 bytes per cycle (GB); N copies exceed this")
    ap.add_argument("--min-mb", type=float, default=512.0, help="minimum bytes streamed per cycle by any method")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--batches", default="1,4,16")
    ap.add_argument("--shapes", default="8b,30b,0.6b")
    ap.add_argument("--methods", default=None, help="comma-separated substrings to keep (default all)")
    ap.add_argument("--no-layers", action="store_true")
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--no-vec-compare", action="store_true", help="skip the <suc>_vec / <suc>_mma A/B rows")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    select = args.methods.split(",") if args.methods else None
    shapes = [s for name in args.shapes.split(",") for s in SHAPES[name]]
    others = busy_processes()
    dev = mx.device_info().get("device_name", platform.machine())
    print(f"# device={dev} mlx={mx.__version__} gb={args.gb} min_mb={args.min_mb} kernel={kernel_config()}")
    if others:
        print(f"# other processes:\n{others}")
    rows: list[dict] = []
    t0 = time.time()
    if not args.no_layers:
        run_layers(shapes, batches, select, not args.no_vec_compare, args.gb, args.min_mb, args.samples, rows)
    if not args.no_decode:
        run_decode(batches, select, not args.no_vec_compare, args.gb, args.samples, rows)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(benchmark="k3_cold_cache", device=dev, mlx=mx.__version__, memory=mx.device_info(),
                           timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), seed=SEED, args=vars(args),
                           kernel_config=kernel_config(), other_processes=others,
                           other_processes_end=busy_processes(), wall_s=time.time() - t0,
                           peak_memory_bytes=mx.get_peak_memory(), rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
