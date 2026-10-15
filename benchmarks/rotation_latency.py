"""Batch-1 latency of one online randomized Hadamard rotation (suc.kernels.rotation) vs the launch floor.

    .venv/bin/python benchmarks/rotation_latency.py [--sizes 4096 12288] [--batches 1] [--chain 144]
        [--repeats 30] [--isolated] [--step qwen3_8b] [--json out.json] [--out-dir experiments/results --name N]

Every timed graph is a chain of --chain DEPENDENT launches (launch l+1 reads launch l's output), mx.compile'd, as
in a decode step where each rotation sits on the critical path (independent launches would overlap on the GPU and
measure throughput instead). By default each launch reads the output of a many-threadgroup copy (spread), like a
rotation reading the projection kernel's output in decode, and the spread chain alone is subtracted; --isolated
chains the launches directly (the vector then stays in one core's L1, which flatters single-threadgroup kernels:
on the M3 the "row" layout looked faster than "blocks" that way, while the decode-step context shows the opposite).
Reported per launch: min / median over --repeats of wall time / chain; repeats are interleaved over the ops.

Rows per size n and batch B:
  empty    a Metal kernel with one 32-thread threadgroup that touches one element (the launch floor)
  copy     a Metal kernel with the rotation's grid copying the vector (floor + reading / writing n values)
  rms_norm mx.fast.rms_norm (an MLX-native latency-bound reference)
  rotate   suc.kernels.rotation.rht_fused (the runtime's input rotation, current default layout, int8 signs),
           x bf16 → x̃ bf16
  rotate_row      LAYOUT = "row" (one threadgroup per row, contiguous VEC layout, constant H_m)
  rotate_strided  the pre-R5 kernel (VEC = 1 strided layout with scalar loads, H_m read from memory, float32 signs)
The 8B decode step has 36 blocks × 4 input rotations (q|k|v, o, gate|up at 4096; down at 12288 = 12·1024).

--step: additionally time a synthetic Qwen3-8B-shaped decode step in context (random VQ G=4 K=256 codes, distinct
per block; per block rms_norm → [rotation] → q|k|v kernel → [rotation] → o kernel → add → rms_norm → [rotation] →
gate|up kernel → SwiGLU → [rotation] → down kernel → add; attention omitted), with rotations "none" | "separate" |
"strided" (the pre-R5 kernel) | "fused" (into RMSNorm / SwiGLU) | "empty" / "copy" (the empty / copy kernel in place of each rotation: the in-context
launch floor). (mode − none) / (4 · blocks) is the in-context cost of one rotation (or one floor launch).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suc.kernels import rotation as rot  # noqa: E402
from suc.kernels.suc_linear import SIGN_DTYPE  # noqa: E402
from suc.quant.hadamard import random_signs  # noqa: E402

_EMPTY = mx.fast.metal_kernel(
    name="suc_bench_empty", input_names=["x"], output_names=["y"],
    source="    if (thread_position_in_grid.x == 0) y[0] = x[0];")
_COPY = mx.fast.metal_kernel(
    name="suc_bench_copy", input_names=["x"], output_names=["y"],
    source="""    uint t = thread_position_in_threadgroup.x, b = threadgroup_position_in_grid.z;
    for (uint i = t; i < N; i += T) y[b * N + i] = x[b * N + i];""")


_SPREAD = mx.fast.metal_kernel(
    name="suc_bench_spread", input_names=["x"], output_names=["y"],
    source="    uint i = thread_position_in_grid.x; y[i] = x[i];")


def spread(x: mx.array) -> mx.array:
    """Copy with one thread per element in 32-thread threadgroups: the vector is written by many GPU cores, as the
    projection kernel's output is in a decode step (an isolated chain of single-threadgroup kernels would keep it in
    one core's L1)."""
    return _SPREAD(inputs=[x], grid=(x.size, 1, 1), threadgroup=(32, 1, 1), output_shapes=[x.shape],
                   output_dtypes=[x.dtype])[0]


def empty(x: mx.array) -> mx.array:
    return _EMPTY(inputs=[x], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[x.shape],
                  output_dtypes=[x.dtype])[0]


def copy(x: mx.array) -> mx.array:
    n = x.shape[-1]
    return _COPY(inputs=[x], template=[("N", n), ("T", 256)], grid=(256, 1, x.shape[0]), threadgroup=(256, 1, 1),
                 output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


def time_chains(ops: dict, x: mx.array, chain: int, repeats: int, warmup_s: float = 3.0,
                in_context: bool = True) -> dict:
    """{name: {min_us, median_us}} per launch; the ops' repeats are interleaved (GPU clock drift hits all alike).
    in_context: every launch reads the output of a spread() launch (many-core producer, as in decode) and the
    chain of spread() launches alone ("spread" row) is subtracted."""
    def chained(op):
        def f(x):
            for _ in range(chain):
                x = op(spread(x)) if in_context else op(x)
            return x
        return mx.compile(f)
    fns = {k: chained(op) for k, op in ops.items()}
    if in_context:
        fns["spread"] = chained(lambda x: x)
    t_end = time.perf_counter() + warmup_s
    while True:                                  # compile + bring the GPU clock up
        for fn in fns.values():
            mx.eval(fn(x))
        if time.perf_counter() > t_end:
            break
    ts = {k: [] for k in fns}
    for _ in range(repeats):
        for k, fn in fns.items():
            t0 = time.perf_counter()
            mx.eval(fn(x))
            ts[k].append((time.perf_counter() - t0) / chain * 1e6)
    out = {k: {"min_us": float(min(v)), "median_us": float(np.median(v))} for k, v in ts.items()}
    if in_context:
        base = out.pop("spread")
        out = {k: {"min_us": v["min_us"] - base["min_us"], "median_us": v["median_us"] - base["median_us"],
                   "spread_min_us": base["min_us"]} for k, v in out.items()}
    return out


def ops_for(n: int) -> dict:
    s = random_signs(n, 1)                 # float32 signs: what the pre-R5 runtime stored
    s8 = s.astype(SIGN_DTYPE)              # what SucLinear stores now
    w = mx.ones((n,), dtype=mx.bfloat16)
    mx.eval(s, s8, w)
    return {"empty": empty, "copy": copy, "rms_norm": lambda x: mx.fast.rms_norm(x, w, 1e-6),
            "rotate": lambda x: rot.rht_fused(x, s8),
            "rotate_row": lambda x: with_cfg(dict(LAYOUT="row"), rot.rht_fused, x, s8),
            "rotate_strided": lambda x: with_cfg(STRIDED, rot.rht_fused, x, s)}


# the pre-R5 kernel: one threadgroup per row, strided layout with scalar loads, H_m read from memory
STRIDED = dict(LAYOUT="row", VEC=1, HM_CONST=False, THREADS_MAX=256)


def with_cfg(cfg: dict, fn, *args, **kw):
    """fn(*args) with suc.kernels.rotation module settings cfg (read when the graph is traced)."""
    old = {k: getattr(rot, k) for k in cfg}
    for k, v in cfg.items():
        setattr(rot, k, v)
    try:
        return fn(*args, **kw)
    finally:
        for k, v in old.items():
            setattr(rot, k, v)


STEP_MODES = ("none", "separate", "strided", "fused", "empty", "copy")
# hidden, intermediate, q|k|v out, layers
STEP_SHAPES = {"qwen3_8b": (4096, 12288, 6144, 36), "qwen3_0.6b": (1024, 3072, 4096, 28)}


def _codes(out: int, inp: int, key) -> tuple:
    from suc.quant.packing import pack_indices
    idx = mx.random.randint(0, 256, (out, inp // 4), key=key).astype(mx.uint32)
    return pack_indices(idx, 8)[None], mx.ones((out,), mx.float32) * 0.02


def step_codes(shape: str, seed: int = 0, layers: int | None = None) -> dict:
    """Random VQ G=4 K=256 codes (distinct per block) for a synthetic decode step of this shape."""
    h, inter, qkv, n_layers = STEP_SHAPES[shape]
    key = mx.random.key(seed)
    blocks = []
    for i in range(layers or n_layers):
        ks = mx.random.split(mx.random.key(seed + 1 + i), 4)
        blocks.append([_codes(qkv, h, ks[0]), _codes(h, h, ks[1]), _codes(2 * inter, h, ks[2]),
                       _codes(h, inter, ks[3])])
        mx.eval(blocks[-1])
    c = {"shape": shape, "h": h, "inter": inter, "blocks": blocks,
         "cb": (0.5 * mx.random.normal((1, 256, 4), key=key)).astype(mx.float16),
         "sh": random_signs(h, 1), "si": random_signs(inter, 2), "w": mx.ones((h,), mx.bfloat16)}
    c["sh8"], c["si8"] = c["sh"].astype(SIGN_DTYPE), c["si"].astype(SIGN_DTYPE)
    mx.eval(c["cb"], c["sh"], c["si"], c["w"], c["sh8"], c["si8"])
    return c


def build_step(codes: dict, rotations: str):
    """fn(x) -> x: one synthetic decode step (see --step) on `codes`, rotations one of STEP_MODES."""
    from suc.kernels.vq_matmul import additive_matmul_metal
    h, inter, cb, w = (codes[k] for k in ("h", "inter", "cb", "w"))
    # int8 signs (the runtime's) except for the pre-R5 "strided" mode (float32 signs, as stored then)
    sh, si = (codes["sh"], codes["si"]) if rotations == "strided" else (codes["sh8"], codes["si8"])

    def lin(x, c, n_in):
        return additive_matmul_metal(x, cb, c[0], c[1], bits=8, in_features=n_in)

    def rot_(x, s):
        if rotations in ("empty", "copy"):         # a launch with the rotation's dependency but no work
            return empty(x) if rotations == "empty" else copy(x)
        if rotations == "strided":
            return with_cfg(STRIDED, rot.rht_fused, x, s)
        return x if rotations == "none" else rot.rht_fused(x, s)

    def norm_rot(x):
        if rotations == "fused":
            return rot.rms_norm_rht(x, w, 1e-6, sh)[0]
        return rot_(mx.fast.rms_norm(x, w, 1e-6), sh)

    def swiglu_rot(g, u):
        if rotations == "fused":
            return rot.swiglu_rht(g, u, si)[0]
        return rot_(nn_silu(g) * u, si)

    def step(x):
        for qkv_c, o_c, gu_c, d_c in codes["blocks"]:
            a = lin(norm_rot(x), qkv_c, h)[:, :h]          # attention omitted: q stands in for its output
            x = x + lin(rot_(a, sh), o_c, h)
            gu = lin(norm_rot(x), gu_c, h)
            x = x + lin(swiglu_rot(gu[:, :inter], gu[:, inter:]), d_c, inter)
        return x
    return mx.compile(step)


def nn_silu(g):
    return g * mx.sigmoid(g)


def time_fns(fns: dict, x0: mx.array, repeats: int, warmup_s: float = 3.0) -> dict:
    """{name: {min_ms, median_ms}} per call of each fn(x0), repeats interleaved over the fns."""
    t_end = time.perf_counter() + warmup_s
    while True:
        for fn in fns.values():
            mx.eval(fn(x0))
        if time.perf_counter() > t_end:
            break
    ts = {k: [] for k in fns}
    for _ in range(repeats):
        for k, fn in fns.items():
            t0 = time.perf_counter()
            mx.eval(fn(x0))
            ts[k].append((time.perf_counter() - t0) * 1e3)
    return {k: {"min_ms": float(min(v)), "median_ms": float(np.median(v))} for k, v in ts.items()}


def time_step(codes: dict, batch: int, repeats: int, modes=STEP_MODES) -> dict:
    x0 = (0.1 * mx.random.normal((batch, codes["h"]), key=mx.random.key(0))).astype(mx.bfloat16)
    mx.eval(x0)
    return time_fns({m: build_step(codes, m) for m in modes}, x0, repeats)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[4096, 12288])
    ap.add_argument("--batches", type=int, nargs="+", default=[1])
    ap.add_argument("--chain", type=int, default=144)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--isolated", action="store_true",
                    help="chain the launches directly (no many-core producer in between; favours one-core kernels)")
    ap.add_argument("--step", choices=sorted(STEP_SHAPES), help="also time a synthetic decode step of this shape")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--out-dir", help="also write <out-dir>/<timestamp>_<name>.json (suc.provenance.write_result)")
    ap.add_argument("--name", default="rotation_latency")
    args = ap.parse_args(argv)
    rows, steps = [], []
    for n in args.sizes:
        ops = ops_for(n)
        for b in args.batches:
            x = mx.random.normal((b, n), key=mx.random.key(n)).astype(mx.bfloat16)
            mx.eval(x)
            for name, t in time_chains(ops, x, args.chain, args.repeats, in_context=not args.isolated).items():
                r = {"n": n, "batch": b, "op": name, **t}
                rows.append(r)
                print(f"n={n:6d} B={b:2d} {name:9s} min {r['min_us']:7.2f} µs  median {r['median_us']:7.2f} µs",
                      flush=True)
    if args.step:
        layers = STEP_SHAPES[args.step][3]
        codes = step_codes(args.step)
        for b in args.batches:
            res = time_step(codes, b, args.repeats)
            per = {m: (res[m]["min_ms"] - res["none"]["min_ms"]) * 1e3 / (4 * layers) for m in STEP_MODES[1:]}
            steps.append({"shape": args.step, "batch": b, **{f"{m}_ms": res[m] for m in res},
                          "per_rotation_us": per})
            print(f"step {args.step} B={b}: " + "  ".join(f"{m} {res[m]['min_ms']:.2f}/{res[m]['median_ms']:.2f} ms"
                                                       for m in res)
                  + "  per rotation (min): " + ", ".join(f"{m} {v:.1f} µs" for m, v in per.items()),
                  flush=True)
    from suc.kernels import autotune
    from suc.provenance import run_metadata, write_result
    md = run_metadata(model=f"synthetic:{args.step or 'rotation'}", model_revision="", model_hash="synthetic",
                      dataset_hash="none", seed=0, config_hash="none")
    out = {"metadata": md, "benchmark": "rotation_latency", "rows": rows, "steps": steps, "chain": args.chain,
           "repeats": args.repeats, "in_context": not args.isolated,
           "rotation_settings": {k: getattr(rot, k) for k in ("LAYOUT", "VEC", "HM_CONST", "BLOCK_Q", "BLOCK_SG")},
           "resolved": {n: {"layout": rot.layout_for(n), "block_config": rot.block_config(n)} for n in args.sizes},
           "tuning_entry": autotune._load().get(autotune.device_key())}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1) + "\n")
    if args.out_dir:
        print(f"# wrote {write_result(out, args.out_dir, args.name)}")
    return out


if __name__ == "__main__":
    main()
