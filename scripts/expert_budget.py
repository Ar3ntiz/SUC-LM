#!/usr/bin/env python3
"""Expert-streaming budget calculator (docs/expert_streaming.md §3; model in suc.runtime.expert_cache.budget_plan).

Given a SUC1 file (or a model preset) and RAM budgets, prints the resident set, the expert-cache size, the expected
expert miss traffic per token and estimated decode tokens/s from an SSD bandwidth (measure it with
scripts/pread_bench.py on the target Mac and pass --ssd-gbps or --bench-json).

  python scripts/expert_budget.py --suc1 model.suc1 --ram 111,64,32 --ssd-gbps 5.5
  python scripts/expert_budget.py --preset qwen3.8-4.2 --ram 111,64,32 --trace routing.npy
  python scripts/expert_budget.py --preset qwen3.8-2.5 --ram 64,32 --zipf 0,0.8,1.2

Hit rate: by default the uniform-routing value (= cache capacity fraction; exact for LRU when routing is uniform and
independent, pessimistic for real skewed routing). --trace (tokens, layers, k) int array of router choices replays
the trace through the eviction policy; --zipf s adds synthetic Zipf(s)-popularity what-if rows (not a measurement).
Compute time per token: --compute-ms, or active bytes / --compute-gbps (effective bandwidth of the resident decode;
SUC on Qwen3-8B reaches ≈ 200 GB/s on the M1 Ultra: 4.3 GB x 47 tok/s, docs/benchmarks.md K6).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suc.runtime.expert_cache import budget_plan, simulate, zipf_trace  # noqa: E402

EXPERT_PARAMS_QWEN38 = 1280 * 2560 + 2560 * 640            # gate|up 1280x2560 + down 2560x640 = 4 915 200
# Qwen3.8-Flash-Next (docs/arch/qwen4_exp.md): 48 MoE layers x 512 experts, top-10; non-expert resident ≈ 4 GB
# (attention / linear attention / shared experts ≈ 2.8 GB SUC + 1.5 GB BF16 kept tensors, embed/head 4-bit 0.7 GB);
# n-gram table on mmap (16 row reads per token, not counted). Active bytes per token ≈ resident read per token
# (≈ 3.3 GB: everything but the 0.7 GB embedding/head, of which only one row / the head is read) + 10 x 48 experts.
PRESETS = {
    "qwen3.8-4.2": dict(n_layers=48, n_experts=512, top_k=10, bpp=4.2, resident_gb=4.0, active_resident_gb=3.6),
    "qwen3.8-2.5": dict(n_layers=48, n_experts=512, top_k=10, bpp=2.5, resident_gb=4.0, active_resident_gb=3.6),
}


def from_suc1(path: Path, top_k: int | None) -> dict:
    from suc.io.suc1 import SucModel
    from suc.runtime.expert_cache import ExpertLayout
    sm = SucModel.load(path, verify=False)
    lay = ExpertLayout(sm)
    s = lay.summary()
    arch = sm.architecture.get("text_config", sm.architecture)
    k = top_k or int(arch.get("num_experts_per_tok") or arch.get("moe_top_k") or 0)
    if not k:
        raise SystemExit("top-k not found in the architecture: pass --top-k")
    n_exp = max(lay.num_experts)
    sizes = np.zeros((len(lay.layers), n_exp))
    for li in range(len(lay.layers)):
        for e in range(lay.num_experts[li]):
            sizes[li, e] = lay.entry(li, e).nbytes
    return dict(n_layers=s["layers"], n_experts=n_exp, top_k=k, expert_bytes=s["expert_bytes_mean"],
                resident_gb=s["non_expert_bytes"] / 1e9, active_resident_gb=s["non_expert_bytes"] / 1e9, sizes=sizes,
                source=str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--suc1", type=Path)
    src.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--ram", default="111,64,32", help="RAM budgets in GB (usable for the process)")
    ap.add_argument("--overhead-gb", type=float, default=6.0, help="KV cache, activations, runtime, OS headroom")
    ap.add_argument("--resident-gb", type=float, help="override the non-expert resident bytes")
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--ssd-gbps", type=float, default=5.0, help="random-read bandwidth at expert-sized reads")
    ap.add_argument("--bench-json", type=Path, help="pread_bench.py --json output: best pread_nocache bandwidth")
    ap.add_argument("--read-latency-us", type=float, default=150.0, help="dependent round trip per layer with a miss")
    ap.add_argument("--compute-ms", type=float, help="decode time per token with every expert resident")
    ap.add_argument("--compute-gbps", type=float, default=200.0, help="effective bandwidth when --compute-ms unset")
    ap.add_argument("--trace", type=Path, help=".npy (tokens, layers, k) router choices to replay")
    ap.add_argument("--policy", default="lru", choices=["lru", "slru"])
    ap.add_argument("--zipf", default="", help="comma list of Zipf exponents for synthetic what-if rows")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    if a.suc1:
        m = from_suc1(a.suc1, a.top_k)
    else:
        m = dict(PRESETS[a.preset], source=a.preset)
        m["expert_bytes"] = EXPERT_PARAMS_QWEN38 * m["bpp"] / 8
        if a.top_k:
            m["top_k"] = a.top_k
    if a.resident_gb is not None:
        m["resident_gb"] = m["active_resident_gb"] = a.resident_gb
    bw = a.ssd_gbps * 1e9
    if a.bench_json:
        r = json.loads(a.bench_json.read_text())["results"]
        bw = max(x["gb_per_s"] for x in r if x["mode"] == "pread_nocache") * 1e9
    L, E, k, eb = m["n_layers"], m["n_experts"], m["top_k"], m["expert_bytes"]
    active = m["active_resident_gb"] * 1e9 + L * k * eb
    compute_s = a.compute_ms / 1e3 if a.compute_ms else active / (a.compute_gbps * 1e9)
    print(f"source {m['source']}: {L} layers x {E} experts, top-{k}, expert {eb / 1e6:.2f} MB, "
          f"all experts {L * E * eb / 1e9:.1f} GB, resident (non-expert) {m['resident_gb']:.2f} GB")
    print(f"per token: {L * k} expert uses = {L * k * eb / 1e9:.2f} GB if all miss; compute {compute_s * 1e3:.1f} ms "
          f"(active {active / 1e9:.2f} GB); SSD {bw / 1e9:.2f} GB/s, latency {a.read_latency_us:.0f} us/layer-miss")
    rows = []
    hdr = f"{'RAM GB':>7} {'cache GB':>8} {'cap':>5} {'hit model':>14} {'hit':>5} {'miss MB/tok':>11} " \
          f"{'IO ms':>7} {'tok/s serial':>12} {'tok/s overlap':>13}"
    print(hdr)
    traces = []
    if a.trace:
        traces.append(("trace", np.load(a.trace)))
    for s in (float(x) for x in a.zipf.split(",") if x):
        traces.append((f"zipf s={s:g}", zipf_trace(400, min(L, 8), E, k, s, seed=0)))
    for ram in (float(x) for x in a.ram.split(",")):
        base = dict(ram_bytes=ram * 1e9, resident_bytes=m["resident_gb"] * 1e9, overhead_bytes=a.overhead_gb * 1e9,
                    n_layers=L, n_experts=E, top_k=k, expert_bytes=eb, ssd_bytes_per_s=bw, compute_s=compute_s,
                    read_latency_s=a.read_latency_us * 1e-6)
        variants = [("uniform", budget_plan(**base))]
        cache = base["ram_bytes"] - base["resident_bytes"] - base["overhead_bytes"]
        for name, tr in traces:
            if cache < k * eb or cache >= L * E * eb:
                continue
            frac = cache / (L * E * eb)                    # same capacity fraction on the trace's layers
            sim = simulate(tr, eb, frac * tr.shape[1] * E * eb, policy=a.policy)
            variants.append((name, budget_plan(**base, hit_rate=sim["hit_rate"], p_layer_miss=sim["p_layer_miss"])))
        for name, p in variants:
            rows.append({"model": name, **p})
            if not p["feasible"]:
                print(f"{ram:7.0f} {p['cache_gb']:8.1f}  infeasible: {p['reason']}")
                continue
            print(f"{ram:7.0f} {p['cache_gb']:8.1f} {p['capacity_fraction']:5.2f} {name:>14} {p['hit_rate']:5.2f} "
                  f"{p['miss_mb_per_token']:11.0f} {p['io_ms_per_token']:7.1f} {p['tok_s_serial']:12.1f} "
                  f"{p['tok_s_overlapped']:13.1f}")
    if a.json:
        a.json.write_text(json.dumps({"source": m["source"], "rows": rows}, indent=1, default=float))


if __name__ == "__main__":
    main()
