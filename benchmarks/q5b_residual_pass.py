"""Q5b: cost of the residual pass alone per decode step, CSR vs column-structured residuals (docs/benchmarks.md Q5b).

Not a test. Usage:
  .venv/bin/python benchmarks/q5b_residual_pass.py [--shape qwen3_0.6b|qwen3_8b] [--batches 1,4,16]
         [--fraction 0.005] [--repeats 30] [--dependent] [--json out.json]

Per layer the 7 projections of a Qwen3 block are given a residual of the SAME bit budget (ResidualCodec, int8,
importance selection with a synthetic E[x_j²]: a few lognormal "massive" channels, so the selected positions
concentrate on few input columns as on the real model, docs/benchmarks.md Q5b step 1). The base codec is MLX
affine 2-bit (fast to encode; the residual pass never reads the base codes). Weights are N(0, 1/in) with a few
large "super weights" in the hot columns.

Timed per decode step = the residual launches of all layers for one input x (B, in), evaluated together:
  csr_unfused  7 sparse_residual_metal launches per block (the K5 path, compressed_model.FUSE_RESIDUAL = False)
  csr_fused    4 launches per block: q|k|v and gate|up as one SucFusedResidual each, o and down alone
  columns      4 launches, structured_residual_metal on the dense int8 columns (column_share 1)
  columns0.5   4 launches, half of the budget on columns, half on a CSR tail
  floor        4 launches per block of the same kernel with no residual work (dispatch / launch floor)
Only the residual is timed (codes are dummies). The step is mx.compile'd (as the runtime's group calls are), so
Python graph building is off the measurement. min / median over --repeats of mx.eval wall time.

--dependent: each launch reads the previous launch's output (y_in of launch l+1 = output of launch l, per unit,
summed over the units), as in a decode step, where every launch is on the critical path. Without it the 4·layers
launches are independent and the GPU overlaps them: that measures throughput, not the latency a decode step pays
(the first column kernel looked cheap that way — 1.1 ms on the 8B shapes — but cost ~10 ms end to end).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suc.kernels.suc_linear import SucLinear  # noqa: E402
from suc.provenance import git_info  # noqa: E402
from suc.quant.codecs import AffineRTN  # noqa: E402
from suc.quant.packing import words_per_row  # noqa: E402
from suc.quant.residual import ResidualCodec  # noqa: E402
from suc.runtime.compressed_model import SucFusedResidual, module_from_encoded  # noqa: E402,F401
import suc.runtime.compressed_model as cm  # noqa: E402

SHAPES = {  # hidden, intermediate, q out, kv out, layers
    "qwen3_0.6b": (1024, 3072, 2048, 1024, 28),
    "qwen3_8b": (4096, 12288, 4096, 1024, 36),
}
VARIANTS = {"csr_unfused": ("csr", 1.0), "csr_fused": ("csr", 1.0), "columns": ("columns", 1.0),
            "columns0.5": ("columns", 0.5), "floor": ("csr", 1.0)}


def synthetic(out: int, inn: int, rng: np.random.Generator) -> tuple[mx.array, mx.array]:
    ex2 = np.exp(rng.normal(0, 1, inn)).astype(np.float32)
    hot = rng.choice(inn, 6, replace=False)
    ex2[hot] *= 1e3
    w = rng.normal(0, 1 / np.sqrt(inn), (out, inn)).astype(np.float32)
    w[rng.integers(0, out, 6), hot] += 0.5
    return mx.array(w), mx.broadcast_to(mx.array(ex2)[None], (out, inn))


def residual_module(w: mx.array, imp: mx.array, codec: ResidualCodec, seed: int) -> cm.SucResidualLinear:
    e = codec.encode(w, seed=seed, importance=imp)
    out, inn = w.shape
    a = e.arrays
    cb = mx.zeros((1, 256, 4), mx.float16)
    packed = mx.zeros((1, out, words_per_row(inn // 4, 8)), mx.uint32)
    return cm.SucResidualLinear(cb, packed, mx.ones((out,)), bits=8, in_features=inn,
                                res_pos=a["res_pos"], res_val=a["res_val"], res_scale=a.get("res_scale"),
                                res_cols=a.get("res_cols"), res_colval=a.get("res_colval"),
                                res_colscale=a.get("res_colscale"))


def build(shape: str, variant: str, fraction: float, seed: int = 0) -> tuple[list, dict]:
    hid, inter, qo, kvo, layers = SHAPES[shape]
    structure, share = VARIANTS[variant]
    codec = ResidualCodec(AffineRTN(2, 64), 0.0 if variant == "floor" else fraction, "int8", "importance",
                          structure=structure,
                          column_share=share)
    rng = np.random.default_rng(seed)
    # one layer's modules, reused for every layer (same sizes and costs; avoids encoding 36 layers)
    projs = {"q": (qo, hid), "k": (kvo, hid), "v": (kvo, hid), "o": (hid, qo), "gate": (inter, hid),
             "up": (inter, hid), "down": (hid, inter)}
    mods = {}
    for i, (n, (o, i_)) in enumerate(projs.items()):
        w, imp = synthetic(o, i_, rng)
        mods[n] = residual_module(w, imp, codec, seed + i)
    if variant == "csr_unfused":
        units = [(mods[n], projs[n][1]) for n in projs]
    else:
        units = [(SucFusedResidual([mods["q"], mods["k"], mods["v"]]), hid), (mods["o"], qo),
                 (SucFusedResidual([mods["gate"], mods["up"]]), hid), (mods["down"], inter)]
    for u, _ in units:
        mx.eval(u.parameters())
    info = {"launches_per_block": len(units), "layers": layers,
            "residual_bytes_per_block": sum(sum(v.nbytes for k, v in _flat(u) if "res" in k) for u, _ in units),
            "columns_per_block": {n: m.n_columns for n, m in mods.items()},
            "csr_entries_per_block": {n: m.n_residual for n, m in mods.items()}}
    return units, info


def _flat(m):
    from mlx.utils import tree_flatten
    return tree_flatten(m.parameters())


def _floor(x: mx.array, y: mx.array) -> mx.array:
    meta = mx.array(np.array([0, y.shape[1], 0, 0, 0], np.uint32))
    return cm.structured_residual_metal(x, y, meta, None, None, None, None, None, None, mx.ones((1,)), 1)


def time_dependent(units: list, layers: int, b: int, repeats: int, floor: bool = False) -> dict:
    """Sum over units of a chain of `layers` dependent launches (the output of one is the y_in of the next)."""
    tot_min = tot_med = 0.0
    for u, i_ in units:
        x = mx.random.normal((b, i_), key=mx.random.key(i_)).astype(mx.bfloat16)
        y0 = mx.zeros((b, u.out_features), mx.float32)
        mx.eval(x, y0)

        def chain(x, y, u=u):
            for _ in range(layers):
                y = _floor(x, y) if floor else u.residual(x, y)
            return y
        fn = mx.compile(chain)
        for _ in range(3):
            mx.eval(fn(x, y0))
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mx.eval(fn(x, y0))
            ts.append(time.perf_counter() - t0)
        tot_min += min(ts)
        tot_med += float(np.median(ts))
    return {"batch": b, "min_ms": 1e3 * tot_min, "median_ms": 1e3 * tot_med, "dependent": True}


def time_step(units: list, layers: int, b: int, repeats: int, floor: bool = False) -> dict:
    xs = {i_: mx.random.normal((b, i_), key=mx.random.key(i_)).astype(mx.bfloat16) for _, i_ in units}
    ys = [mx.zeros((b, u.out_features), mx.float32) for u, _ in units]
    mx.eval(list(xs.values()), ys)

    def outs_fn(*args):
        x_in, y_in = args[:len(xs)], args[len(xs):]
        xd = dict(zip(xs, x_in))
        f = (lambda u, x, y: _floor(x, y)) if floor else (lambda u, x, y: u.residual(x, y))
        return [f(u, xd[i_], y) for _ in range(layers) for (u, i_), y in zip(units, y_in)]
    fn = mx.compile(outs_fn)
    args = [*xs.values(), *ys]

    def step():
        mx.eval(fn(*args))
    for _ in range(3):
        step()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        step()
        ts.append(time.perf_counter() - t0)
    return {"batch": b, "min_ms": 1e3 * min(ts), "median_ms": 1e3 * float(np.median(ts))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=sorted(SHAPES), default="qwen3_0.6b")
    ap.add_argument("--batches", default="1,4,16")
    ap.add_argument("--fraction", type=float, default=0.005)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--dependent", action="store_true", help="time chains of dependent launches (decode latency)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    res = []
    for v in args.variants.split(","):
        units, info = build(args.shape, v, args.fraction)
        timer = time_dependent if args.dependent else time_step
        rows = [timer(units, info["layers"], b, args.repeats, floor=v == "floor") for b in batches]
        res.append({"variant": v, **info, "timing": rows})
        print(f"{v:12s} launches/block={info['launches_per_block']} res MiB/model="
              f"{info['residual_bytes_per_block'] * info['layers'] / 2**20:.1f} "
              + "  ".join(f"B={r['batch']}: {r['min_ms']:.2f} ms (med {r['median_ms']:.2f})" for r in rows), flush=True)
        del units
        mx.clear_cache()
    if args.json:
        Path(args.json).write_text(json.dumps(dict(
            benchmark="q5b_residual_pass", shape=args.shape, args=vars(args), device=mx.device_info(),
            mlx=mx.__version__, platform=platform.platform(), git=git_info(),
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), results=res), indent=1))
        print(f"# wrote {args.json}")


if __name__ == "__main__":
    main()
