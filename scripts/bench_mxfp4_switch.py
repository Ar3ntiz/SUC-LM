"""Decode-time kernel cost of one routed-expert projection: native MXFP4 (MLX gather_qmm, suc.kernels.mxfp4) vs the
SUC expert codecs of compress_deepseek_v4_flash.yaml (VQ G=4 / G=2, K=256, one-sided input rotation, 0.5% column
residual, shared stack codebook; SucSwitchLinear) and a mixed stack (half native, half VQ G=4: SucSwitchMixed).

Decode = one token, top-k experts of a stack of E (DeepSeek-V4-Flash: top-6; E is reduced, the per-call cost depends
on the k selected experts, not on E). Shapes (out, in): gate / up 2048 x 4096, down 4096 x 2048. Weights are random
(timing only). Two timings per call: `sync` = one call + mx.eval per iteration (median; includes the launch / sync
overhead a batch-1 decode step pays per projection) and `chained` = `--chain` dependent calls in one graph, evaluated
once (amortized kernel time). Timings are only meaningful per device; reported numbers come from the Mac Studio.

  .venv/bin/python scripts/bench_mxfp4_switch.py [--experts 16] [--top-k 6] [--iters 200] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import mlx.core as mx
import numpy as np

from suc.io.quantized_source import np_quantize_mxfp4
from suc.kernels.suc_switch import switch_from_encodings
from suc.quant.codecs import make_codec
from suc.quant.residual import ResidualCodec

SHAPES = ((2048, 4096), (4096, 2048))


def vq_codec(g: int):
    base = make_codec({"kind": "vq", "k": 256, "block_size": g, "rotate": True, "rotate_out": False,
                       "kmeans_iters": 4, "fit_blocks": 65536})
    return ResidualCodec(base, 0.005, "int8", "magnitude", structure="columns")


def stacks(out: int, inn: int, n_exp: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((out, inn)).astype(np.float32) * 0.02
    packed, scale = np_quantize_mxfp4(w)
    nat = make_codec({"kind": "mxfp4"}).encode_source(mx.array(packed), mx.array(scale))
    ref = make_codec({"kind": "mxfp4"}).decode(nat)
    encs = {"native_mxfp4": nat}
    for g in (4, 2):
        encs[f"vq_g{g}+res0.005_cols"] = vq_codec(g).encode(ref, seed=seed, seed_in=11)
    out_mods = {name: switch_from_encodings([e] * n_exp) for name, e in encs.items()}
    out_mods["mixed native+vq_g4"] = switch_from_encodings(
        [encs["native_mxfp4"] if i % 2 == 0 else encs["vq_g4+res0.005_cols"] for i in range(n_exp)])
    bpp = {name: e.bpw() for name, e in encs.items()}
    return out_mods, bpp


def bench(m, x: mx.array, idx: mx.array, iters: int, chain: int) -> dict:
    for _ in range(20):
        mx.eval(m(x, idx))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(m(x, idx))
        ts.append(time.perf_counter() - t0)
    reps = max(3, iters // chain)
    tc = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ys = [m(x, idx) for _ in range(chain)]
        mx.eval(ys)
        tc.append((time.perf_counter() - t0) / chain)
    return {"sync_us": 1e6 * statistics.median(ts), "chained_us": 1e6 * statistics.median(tc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--chain", type=int, default=32)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = []
    for out, inn in SHAPES:
        mods, bpp = stacks(out, inn, a.experts, seed=out)
        x = mx.random.normal((1, 1, 1, inn)).astype(mx.bfloat16)
        idx = mx.array(np.arange(a.top_k)[None] * (a.experts // a.top_k), dtype=mx.uint32)   # spread over the stack
        for name, m in mods.items():
            r = bench(m, x, idx, a.iters, a.chain)
            r.update(shape=[out, inn], codec=name, bpp=bpp.get(name), top_k=a.top_k,
                     chained_us_per_expert=r["chained_us"] / a.top_k)
            rows.append(r)
            print(f"{out}x{inn}  {name:22s} bpp {bpp.get(name, float('nan')):5.2f}  sync {r['sync_us']:8.1f} us  "
                  f"chained {r['chained_us']:8.1f} us ({r['chained_us_per_expert']:6.1f} us/expert)", flush=True)
    if a.json:
        rec = {"device": mx.metal.device_info().get("device_name") if hasattr(mx, "metal") else platform.machine(),
               "mlx": mx.__version__, "experts": a.experts, "top_k": a.top_k, "rows": rows}
        with open(a.json, "w") as f:
            json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
