"""Per-device kernel autotuning.

The best kernel configuration differs between Apple GPU generations and sizes (K3: staging the codebook in
threadgroup memory is 1.1-3x faster on the M1 Ultra, device-memory reads were faster on the M3; K6: fusing the input
rotation into RMSNorm / SwiGLU was slower on the M1 Ultra), and SUC must run on any M-series chip, including ones
never measured here. So choices are *measured* once per device and cached:

    key   = "<architecture>|<device name>"            e.g. "applegpu_g13d|Apple M1 Ultra"
    cache = $SUC_TUNING_CACHE or ~/.cache/suc/kernel_tuning.json   {key: entry}
    entry = {"codebook_in": {"<M>": "device" | "threadgroup"}, "measurements_us": {...}, "mlx": ..., "timestamp": ...,
             "rotation":         {"choice": {"<n>": {"layout": "blocks", "q": Q, "sg": S} | {"layout": "row"}}, ...},
             "fuse_elementwise": {"choice": {"norm": bool, "swiglu": bool}, ...},
             "mma_min_batch":    {"choice": {"<M>": B}, ...}}

Sections (each with its own measurements and timestamp; a missing section is measured on the next ensure_tuned,
so entries written by older versions are completed, not discarded):
  codebook_in       codebook placement per number of codebooks M (suc.kernels.vq_matmul.resolve_codebook_in)
  rotation          online-rotation kernel layout / block shape per size n (suc.kernels.rotation.layout_for /
                    block_config); measured for the sizes of the model being converted
  fuse_elementwise  input rotation fused into RMSNorm / into SwiGLU vs separate launches, per op
                    (suc.runtime.compressed_model.fuse_elementwise)
  mma_min_batch     batch tile from which the simdgroup-matrix kernel beats the vec kernel, per M
                    (suc.kernels.vq_matmul.mma_min_batch)

Kernels only *read* the cache (`lookup`, `rotation_choice`, ...); measuring happens explicitly, outside any
traced/compiled graph, via `ensure_tuned()` (called when a compressed model is built) or `autotune()` /
`python -m suc.kernels.autotune`. Set SUC_AUTOTUNE=0 to never measure (built-in defaults where nothing is cached).

The codebook / MMA measurements stream enough distinct weight copies (target_mb) to defeat the system-level cache,
like K3, at decode-relevant batch sizes. The rotation / fusion measurements are dependent chains in decode context:
every link is a VQ projection kernel (distinct weights, many cores) whose output feeds the timed op, whose output
feeds the next link — a rotation's cost is its latency on the critical path, not its throughput.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
from typing import Callable

import mlx.core as mx

PLACEMENTS = ("device", "threadgroup")
_memory: dict | None = None


def cache_path() -> Path:
    return Path(os.environ.get("SUC_TUNING_CACHE", Path.home() / ".cache" / "suc" / "kernel_tuning.json"))


def clear_memory_cache() -> None:
    global _memory
    _memory = None


@lru_cache(maxsize=None)
def device_key() -> str:
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return f"{info.get('architecture', '')}|{info.get('device_name', '')}"


def _load() -> dict:
    global _memory
    if _memory is None:
        try:
            _memory = json.loads(cache_path().read_text())
        except (OSError, ValueError):
            _memory = {}
    return _memory


def _entry(key: str | None) -> dict:
    return _load().get(key or device_key(), {})


def _save_section(key: str, name: str, section: dict) -> None:
    entry = dict(_load().get(key, {}))
    entry[name] = section
    _save(key, entry)


def _nearest(choices: dict, m: int):
    if not choices:
        return None
    return choices[min(choices, key=lambda s: (abs(int(s) - m), -int(s)))]


def _rotation_section(entry: dict) -> dict:
    """The entry's "rotation" section if it was measured under the current launch policy
    (suc.kernels.rotation.LAUNCH_POLICY, audit F09), else {}. Sections without a "launch_policy" field predate it;
    they are accepted: every candidate they measured already launched ≤ 256 threads (THREADS_MAX was always 256,
    blocks used ≤ 8 simdgroups), i.e. they satisfy the first policy, "tg256-pow2-v1"."""
    from suc.kernels.rotation import LAUNCH_POLICY
    sec = entry.get("rotation", {})
    return sec if sec.get("launch_policy", "tg256-pow2-v1") == LAUNCH_POLICY else {}


def rotation_choice(n: int, key: str | None = None) -> dict | None:
    """Cached rotation kernel choice for size n on this device ({"layout": ..., "q": ..., "sg": ...}), else None
    (also when the cached section was measured under another launch policy)."""
    return _rotation_section(_entry(key)).get("choice", {}).get(str(n))


def fuse_elementwise_choice(op: str = "norm", key: str | None = None) -> bool | None:
    """Cached choice: fuse the input rotation into op ("norm" = RMSNorm, "swiglu") on this device, else None."""
    c = _entry(key).get("fuse_elementwise", {}).get("choice")
    return c.get(op) if isinstance(c, dict) else c


def mma_min_batch_choice(m: int = 1, key: str | None = None) -> int | None:
    """Cached vec → simdgroup-matrix crossover batch for M codebooks (nearest measured M), else None."""
    return _nearest(_entry(key).get("mma_min_batch", {}).get("choice", {}), m)


def _save(key: str, entry: dict) -> None:
    data = dict(_load())
    data[key] = entry
    p = cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, indent=1) + "\n")
    tmp.replace(p)
    clear_memory_cache()


def lookup(key: str, m: int) -> str | None:
    """Cached codebook placement for this device and number of codebooks M (nearest measured M), else None."""
    return _nearest(_load().get(key, {}).get("codebook_in", {}), m)


def _measure_kernel(placement: str, m: int, batch: int, *, out: int, inp: int, target_mb: float,
                    reps: int, path: str | None = None, g: int = 8) -> float:
    """Median µs per layer call, cycling through distinct random weight copies (cold cache). path None: the vec
    kernel below the current crossover, else "auto"."""
    from suc.kernels.vq_matmul import additive_matmul_metal, mma_min_batch
    from suc.quant.packing import pack_indices

    k, bits = 256, 8
    copy_bytes = m * out * inp // g * bits // 8
    n = max(2, int(target_mb * 2 ** 20 // copy_bytes))
    key = mx.random.key(0)
    cbs = mx.random.normal((m, k, g), key=key).astype(mx.float16)
    scales = mx.ones((out,), dtype=mx.float32)
    base = pack_indices(mx.random.randint(0, k, (m * out, inp // g), key=key).astype(mx.uint32), bits)
    copies = [(base + i).reshape(m, out, -1) for i in range(n)]   # distinct buffers, same layout
    mx.eval(copies)
    x = mx.random.normal((batch, inp), key=key).astype(mx.float16)
    path = path or ("vec" if batch < mma_min_batch(m) else "auto")
    run = lambda: mx.eval([additive_matmul_metal(x, cbs, c, scales, bits=bits, in_features=inp,  # noqa: E731
                                                 codebook_in=placement, path=path) for c in copies])
    run()                                                            # compile + warm the pipeline
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run()
        samples.append((time.perf_counter() - t0) / n * 1e6)
    del copies
    mx.clear_cache()
    return sorted(samples)[len(samples) // 2]


def _measure_paths(placement: str, m: int, batch: int, *, out: int, inp: int, target_mb: float, reps: int,
                   g: int, paths=("vec", "mma")) -> dict:
    """{path: min µs per layer call}, the paths alternating rep by rep over the same cold-cache weight copies (an
    A/B less exposed to GPU clock drift than separate medians)."""
    from suc.kernels.vq_matmul import additive_matmul_metal
    from suc.quant.packing import pack_indices

    k, bits = 256, 8
    n = max(2, int(target_mb * 2 ** 20 // (m * out * inp // g * bits // 8)))
    key = mx.random.key(0)
    cbs = mx.random.normal((m, k, g), key=key).astype(mx.float16)
    scales = mx.ones((out,), dtype=mx.float32)
    base = pack_indices(mx.random.randint(0, k, (m * out, inp // g), key=key).astype(mx.uint32), bits)
    copies = [(base + i).reshape(m, out, -1) for i in range(n)]
    mx.eval(copies)
    x = mx.random.normal((batch, inp), key=key).astype(mx.float16)

    def run(path):
        mx.eval([additive_matmul_metal(x, cbs, c, scales, bits=bits, in_features=inp, codebook_in=placement,
                                       path=path) for c in copies])
    for p in paths:
        run(p)
    best = {p: float("inf") for p in paths}
    for _ in range(reps):
        for p in paths:
            t0 = time.perf_counter()
            run(p)
            best[p] = min(best[p], (time.perf_counter() - t0) / n * 1e6)
    del copies
    mx.clear_cache()
    return best


def autotune(*, key: str | None = None, ms: tuple[int, ...] = (1, 2, 3, 4), batches: tuple[int, ...] = (1, 4),
             measure: Callable[[str, int, int], float] | None = None, out: int = 12288, inp: int = 4096,
             target_mb: float = 256, reps: int = 3, save: bool = True) -> dict:
    """Measure both codebook placements per M (summed over `batches`), pick the faster, cache and return it."""
    key = key or device_key()
    if measure is None:
        measure = lambda p, m, b: _measure_kernel(p, m, b, out=out, inp=inp,  # noqa: E731
                                                  target_mb=target_mb, reps=reps)
    meas, choice = {}, {}
    for m in ms:
        total = {}
        for p in PLACEMENTS:
            total[p] = 0.0
            for b in batches:
                us = measure(p, m, b)
                meas[f"m{m}_b{b}_{p}"] = us
                total[p] += us
        choice[str(m)] = min(PLACEMENTS, key=lambda p: total[p])
    entry = {"codebook_in": choice, "measurements_us": meas, "batches": list(batches), "shape": [out, inp],
             "mlx": mx.__version__, "timestamp": datetime.now(timezone.utc).isoformat()}
    if save:   # keep the other sections of this device
        _save(key, {**{k: v for k, v in _load().get(key, {}).items() if k in SECTIONS[1:]}, **entry})
    return entry


# ----------------------------------------------------------------------------- decode-context chains (R5)
SECTIONS = ("codebook_in", "rotation", "fuse_elementwise", "mma_min_batch")
ROTATION_CANDIDATES = ({"layout": "row"},
                       {"layout": "blocks", "q": 128, "sg": 8}, {"layout": "blocks", "q": 256, "sg": 4},
                       {"layout": "blocks", "q": 256, "sg": 8}, {"layout": "blocks", "q": 512, "sg": 4},
                       {"layout": "blocks", "q": 512, "sg": 8})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _rotation_cfg:
    """Context manager: suc.kernels.rotation settings for one candidate (read while a graph is traced)."""

    def __init__(self, cand: dict):
        self.cfg = {"LAYOUT": cand["layout"]}
        if cand["layout"] == "blocks":
            self.cfg.update(BLOCK_Q=cand["q"], BLOCK_SG=cand["sg"])

    def __enter__(self):
        from suc.kernels import rotation as rot
        self.old = {k: getattr(rot, k) for k in self.cfg}
        for k, v in self.cfg.items():
            setattr(rot, k, v)

    def __exit__(self, *exc):
        from suc.kernels import rotation as rot
        for k, v in self.old.items():
            setattr(rot, k, v)


CHAIN_DIN = 1024   # input width of the VQ link of a decode-context chain (clipped to the timed op's output)


def _context_chain(ops: dict, out: int, *, din: int | None = None, links: int = 24, reps: int = 31,
                   warmup_s: float = 1.0, seed: int = 0) -> dict:
    """Median µs per link of each op(y) in a dependent decode-like chain: y = VQ kernel(x[:, :din]) (out × din,
    distinct weights per link, G=4 K=256), x = op(y). ops: {name: (fn, cand)} (cand: rotation settings while
    tracing, or None); op "none" (identity) is added and subtracted rep by rep, so the result is the op's own
    cost. din (default min(CHAIN_DIN, out)) must not exceed the width of every op's output (and of y, for the
    identity op): the next link reads x[:, :din]."""
    from suc.kernels.vq_matmul import additive_matmul_metal
    from suc.quant.packing import pack_indices

    din = min(CHAIN_DIN, out) if din is None else din
    if din > out or din % 4:
        raise ValueError(f"chain input width {din} must be a multiple of 4 and <= the link output {out}")
    key = mx.random.key(seed)
    cb = (0.5 * mx.random.normal((1, 256, 4), key=key)).astype(mx.float16)
    ws = [(pack_indices(mx.random.randint(0, 256, (out, din // 4), key=mx.random.key(seed + 1 + i))
                        .astype(mx.uint32), 8)[None], mx.full((out,), 0.02, mx.float32)) for i in range(links)]
    x0 = (0.1 * mx.random.normal((1, din), key=key)).astype(mx.bfloat16)
    mx.eval(cb, ws, x0)

    def chain(op):
        def f(x):
            for w in ws:
                x = op(additive_matmul_metal(x[:, :din], cb, w[0], w[1], bits=8, in_features=din))
            return x
        return mx.compile(f)
    fns = {"none": (chain(lambda y: y), None)}
    fns.update({k: (chain(fn), cand) for k, (fn, cand) in ops.items()})

    def run(name):
        fn, cand = fns[name]
        if cand is None:
            return mx.eval(fn(x0))
        with _rotation_cfg(cand):
            return mx.eval(fn(x0))
    t_end = time.perf_counter() + warmup_s
    while True:
        for k in fns:
            run(k)
        if time.perf_counter() > t_end:
            break
    ts = {k: [] for k in fns}
    for _ in range(reps):
        for k in fns:
            t0 = time.perf_counter()
            run(k)
            ts[k].append((time.perf_counter() - t0) / links * 1e6)
    # median over reps of (op − none) within the same rep: clock-regime changes between reps cancel
    diff = {k: sorted(a - z for a, z in zip(ts[k], ts["none"]))[reps // 2] for k in ops}
    del ws, fns
    mx.clear_cache()
    return diff


def _rotation_candidates(n: int) -> list[dict]:
    """ROTATION_CANDIDATES that differ for size n after clipping to its 2^k factor (blocks_shape)."""
    from suc.kernels import rotation as rot
    from suc.quant.hadamard import _factor
    m, p = _factor(n)
    seen, out = set(), []
    for c in ROTATION_CANDIDATES:
        # launch-policy guard (audit F09): a candidate whose launch would be rejected is never measured
        if c["layout"] == "row":
            shape = ("row",)
            t = rot.threads_and_vec((p,), rot.row_threads_max(), rot.VEC)[0]
        else:
            shape = ("blocks", *rot.blocks_shape((p,), (m,), c["q"], c["sg"]))
            t = shape[2]
        try:
            rot._check_launch(t, c["layout"])
        except ValueError:
            continue
        if shape not in seen:
            seen.add(shape)
            out.append(c)
    return out


def _cand_name(c: dict) -> str:
    return "row" if c["layout"] == "row" else f"blocks_q{c['q']}_sg{c['sg']}"


def measure_rotation(n: int, **kw) -> dict:
    """{candidate name: µs per rotation of size n in decode context} (see _context_chain)."""
    from suc.kernels import rotation as rot
    from suc.quant.hadamard import random_signs
    s = random_signs(n, 1)
    mx.eval(s)
    ops = {_cand_name(c): (lambda y: rot.rotate(y, s), c) for c in _rotation_candidates(n)}
    return _context_chain(ops, n, **kw)


def tune_rotation(sizes, *, key: str | None = None, measure: Callable[[int], dict] | None = None,
                  save: bool = True) -> dict:
    """Measure the rotation candidates for every size in `sizes`, merge the fastest into the "rotation" section."""
    from suc.kernels import rotation as rot
    key = key or device_key()
    measure = measure or measure_rotation
    sec = dict(_rotation_section(_load().get(key, {})))   # another launch policy's choices are not merged
    choice, meas = dict(sec.get("choice", {})), dict(sec.get("measurements_us", {}))
    names = {_cand_name(c): c for c in ROTATION_CANDIDATES}
    for n in sorted(set(int(n) for n in sizes)):
        if not rot.supported(n):   # the MLX fallback rotates these sizes: no kernel layout to choose
            continue
        try:
            raw = measure(n)
            us = {k: float(v) for k, v in raw.items() if k in names}
            if not us:
                raise ValueError(f"no known candidate measured (got {sorted(raw)})")
        except Exception as e:   # noqa: BLE001 — a size that cannot be measured keeps the built-in default
            warnings.warn(f"[suc] rotation tuning failed for n={n} ({type(e).__name__}: {e}); "
                          f"using the default layout", RuntimeWarning, stacklevel=2)
            continue
        meas[str(n)] = us
        choice[str(n)] = names[min(us, key=us.get)]
    sec = {"choice": choice, "measurements_us": meas, "launch_policy": rot.LAUNCH_POLICY, "mlx": mx.__version__,
           "timestamp": _now()}
    if save:
        _save_section(key, "rotation", sec)
    return sec


def measure_fuse_elementwise(hidden: int, intermediate: int, **kw) -> dict:
    """µs per op in decode context: RMSNorm → rotation (hidden) and SwiGLU → rotation (intermediate), each fused
    (one launch, suc.kernels.rotation.rms_norm_rht / swiglu_rht) and separate (MLX op + rotate)."""
    from suc.kernels import rotation as rot
    from suc.quant.hadamard import random_signs
    sh, si = random_signs(hidden, 1), random_signs(intermediate, 2)
    w = mx.ones((hidden,), mx.bfloat16)
    mx.eval(sh, si, w)

    def swiglu(y):
        g, u = y[:, :intermediate], y[:, intermediate:]
        return g * mx.sigmoid(g) * u
    res = {}
    if rot.supported(hidden):   # otherwise the fused kernel does not exist (MLX fallback rotation)
        norm = _context_chain({"fused": (lambda y: rot.rms_norm_rht(y, w, 1e-6, sh)[0], None),
                               "separate": (lambda y: rot.rotate(mx.fast.rms_norm(y, w, 1e-6), sh), None)},
                              hidden, **kw)
        res.update(norm_fused=norm["fused"], norm_separate=norm["separate"])
    if rot.supported(intermediate):   # the SwiGLU output (the next link's input) is intermediate wide
        mlp = _context_chain({"fused": (lambda y: rot.swiglu_rht(y[:, :intermediate], y[:, intermediate:], si)[0],
                                        None),
                              "separate": (lambda y: rot.rotate(swiglu(y), si), None)}, 2 * intermediate,
                             **{"din": min(CHAIN_DIN, intermediate), **kw})
        res.update(swiglu_fused=mlp["fused"], swiglu_separate=mlp["separate"])
    return res


FUSE_MARGIN = 0.05


def tune_fuse_elementwise(hidden: int = 4096, intermediate: int = 12288, *, key: str | None = None,
                          measure: Callable[[int, int], dict] | None = None, save: bool = True) -> dict:
    """Per op (RMSNorm → q|k|v / gate|up, SwiGLU → down_proj): fused iff at least FUSE_MARGIN faster than
    separate launches (the default), so measurement noise does not flip it."""
    key = key or device_key()
    us = (measure or measure_fuse_elementwise)(hidden, intermediate)
    # an op without measurements (size not supported by the fused kernels) is never fused
    choice = {op: bool(f"{op}_fused" in us and us[f"{op}_fused"] < (1 - FUSE_MARGIN) * us[f"{op}_separate"])
              for op in ("norm", "swiglu")}
    sec = {"choice": choice, "measurements_us": us, "sizes": [hidden, intermediate], "mlx": mx.__version__,
           "timestamp": _now()}
    if save:
        _save_section(key, "fuse_elementwise", sec)
    return sec


def tune_mma_min_batch(*, key: str | None = None, ms: tuple[int, ...] = (1, 3), batches=(2, 3, 4, 5, 6, 8),
                       measure: Callable[[str, int, int], float] | None = None, out: int = 12288, inp: int = 4096,
                       target_mb: float = 128, reps: int = 7, save: bool = True) -> dict:
    """Per M: the smallest measured batch from which the simdgroup-matrix path is faster than the vec path at that
    and every larger measured batch (max(batches) + 1 when it never is). VQ G=4 for M = 1, G=8 otherwise."""
    from suc.kernels.vq_matmul import resolve_codebook_in
    key = key or device_key()
    if measure is None:
        pairs = {}

        def measure(path, m, b):
            if (m, b) not in pairs:
                pairs[(m, b)] = _measure_paths(resolve_codebook_in("auto", key=key, m=m), m, b, out=out, inp=inp,
                                               target_mb=target_mb, reps=reps, g=4 if m == 1 else 8)
            return pairs[(m, b)][path]
    meas, choice = {}, {}
    for m in ms:
        wins = {}
        for b in batches:
            vec, mma = measure("vec", m, b), measure("mma", m, b)
            meas[f"m{m}_b{b}_vec"], meas[f"m{m}_b{b}_mma"] = vec, mma
            wins[b] = mma < vec
        bs = sorted(batches)
        choice[str(m)] = next((b for i, b in enumerate(bs) if all(wins[c] for c in bs[i:])), bs[-1] + 1)
    sec = {"choice": choice, "measurements_us": meas, "shape": [out, inp], "mlx": mx.__version__, "timestamp": _now()}
    if save:
        _save_section(key, "mma_min_batch", sec)
    return sec


def ensure_tuned(*, key: str | None = None, sections: tuple[str, ...] = SECTIONS, rotation_sizes=(),
                 hidden: int | None = None, intermediate: int | None = None, **kwargs) -> dict | None:
    """Measure what this device's cache entry lacks (no-op when complete or SUC_AUTOTUNE=0), return the entry.

    codebook_in / mma_min_batch: once per device; rotation: each of rotation_sizes not measured yet;
    fuse_elementwise: once, when hidden / intermediate are given (the model's sizes). kwargs go to autotune()
    (codebook_in). Never call inside mx.compile.

    Never raises: a section whose measurement fails is skipped with a RuntimeWarning (the kernels then use their
    built-in defaults for it), so tuning cannot break model conversion."""
    if os.environ.get("SUC_AUTOTUNE", "1") == "0":
        return None
    try:
        key = key or device_key()
        entry = _load().get(key, {})
    except Exception as e:   # noqa: BLE001
        warnings.warn(f"[suc] kernel tuning skipped ({type(e).__name__}: {e}); using defaults", RuntimeWarning,
                      stacklevel=2)
        return None
    todo = []
    if "codebook_in" in sections and "codebook_in" not in entry:
        todo.append(("codebook_in", lambda: autotune(key=key, **kwargs)))
    from suc.kernels.rotation import supported
    # sizes the rotation kernels do not support use the MLX fallback: nothing to tune (and nothing ever cached)
    missing = sorted({int(n) for n in rotation_sizes if supported(int(n))}
                     - {int(n) for n in _rotation_section(entry).get("choice", {})})
    if "rotation" in sections and missing:
        todo.append((f"rotation {missing}", lambda: tune_rotation(missing, key=key)))
    if "mma_min_batch" in sections and "mma_min_batch" not in entry:
        todo.append(("mma_min_batch", lambda: tune_mma_min_batch(key=key)))
    if "fuse_elementwise" in sections and "fuse_elementwise" not in entry and hidden and intermediate:
        todo.append(("fuse_elementwise", lambda: tune_fuse_elementwise(hidden, intermediate, key=key)))
    for name, fn in todo:   # in this order: rotation before fusion (the fusion A/B uses the tuned rotation)
        print(f"[suc] tuning {name} for {key} (once; cached in {cache_path()})", flush=True)
        try:
            fn()
        except Exception as e:   # noqa: BLE001 — defaults stay in effect for this section
            warnings.warn(f"[suc] tuning {name} failed ({type(e).__name__}: {e}); using defaults", RuntimeWarning,
                          stacklevel=2)
    try:
        return _load().get(key)
    except Exception:   # noqa: BLE001
        return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Measure and cache this device's kernel choices (all sections).")
    ap.add_argument("--rotation-sizes", type=int, nargs="*", default=[4096, 12288])
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--intermediate", type=int, default=12288)
    ap.add_argument("--force", action="store_true", help="re-measure every section")
    a = ap.parse_args()
    if a.force:
        autotune()
        tune_rotation(a.rotation_sizes)
        tune_mma_min_batch()
        tune_fuse_elementwise(a.hidden, a.intermediate)
    else:
        ensure_tuned(rotation_sizes=a.rotation_sizes, hidden=a.hidden, intermediate=a.intermediate)
    print(json.dumps({device_key(): _load().get(device_key())}, indent=1))
