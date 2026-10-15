"""Render the mandatory ablation table (spec §26) and the Pareto frontier (spec §27) for Qwen3-8B from result JSONs.

Not a test. Reads only experiments/results/*.json (the newest file per pattern); rows whose run has not been made
yet are printed as "pending" with the config that produces them (docs/ablations_plan.md has the full map).

    .venv/bin/python benchmarks/make_tables.py                                   # markdown to stdout
    .venv/bin/python benchmarks/make_tables.py --md docs/ablation_tables.md --plot docs/figures/pareto_qwen3_8b.svg

The plot (PPL and KL vs actual bpp) needs matplotlib; without it only the markdown is written.
Protocols: A = suc quantize, rotation_groups on, embedding / LM head fixed at affine 4-bit (abl_* configs);
B = suc allocate (per-tensor rotation seeds; head allocated or affine 4-bit); Q5 = embedding / LM head encoded with
the row's codec (earlier Q1-Q6 grids). Rows of different protocols are never differenced against each other.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

RESULTS = Path("experiments/results")
PARETO_TARGETS = (4.0, 3.5, 3.0, 2.5, 2.25, 2.0, 1.75, 1.5, 1.0)
PARETO_BUDGETS = {1.0: 1.05}        # target -> budget actually used (the cheapest plan is ≈ 1.006 bpp)
EXTRA_BUDGETS = (4.22, 3.38)        # greedy at the bpp of the uniform M=4 / M=3 rows (semantic-sensitivity ablation)

SUC = "additive_m4_k256_g8_rot1s+res0.005_activation_cols"
SUC_CSR = "additive_m4_k256_g8_rot1s+res0.005_activation"
M3 = "additive_m3_k256_g8_rot1s+res0.005_activation_cols"
G4 = "vq_k256_g4_rot1s+res0.005_activation_cols"


def greedy_name(budget: float) -> str:
    """Plan name used by suc.allocation.run for a budget."""
    return f"greedy_{budget:.1f}"


@dataclass(frozen=True)
class Src:
    kind: str           # quantize | allocate | compress | e2e | entropy
    pattern: str        # glob under RESULTS; the newest match is used
    key: str = ""       # codec (quantize), plan (allocate), variant (e2e)


@dataclass(frozen=True)
class Row:
    group: str
    label: str
    src: Src
    vs: str | None = "SUC"          # label of the row the deltas are taken against (same table, same protocol)
    note: str = ""


def _q(p: str) -> str:
    return f"*_{p}.json"


ABL_BASE, ABL_RES, ABL_CB, ABL_FT, ABL_REP = (_q(f"abl_{n}_qwen3_8b")
                                              for n in ("base", "residual", "codebook", "finetune", "repeat"))
PARETO = _q("pareto_allocate_qwen3_8b")

# Mandatory ablation table, protocol A unless stated (B: the allocator rows come from the Pareto run).
ABLATIONS: tuple[Row, ...] = (
    Row("SUC", "SUC", Src("quantize", ABL_BASE, SUC), vs=None,
        note="additive M=4 K256 G8, one-sided rotation, 0.5% activation column residual, head affine 4"),
    Row("repeatability", "SUC, second seed (23456)", Src("quantize", ABL_REP, SUC),
        note="seed-to-seed spread = noise floor of every Δ"),
    Row("incoherence", "rotation two-sided (U W Vᵀ)", Src("quantize", ABL_BASE,
                                                         "additive_m4_k256_g8_rot+res0.005_activation_cols")),
    Row("repeatability", "rotation two-sided, second seed (23456)",
        Src("quantize", ABL_REP, "additive_m4_k256_g8_rot+res0.005_activation_cols"),
        vs="SUC, second seed (23456)"),
    Row("incoherence", "SUC − incoherence (no rotation)", Src("quantize", ABL_BASE,
                                                              "additive_m4_k256_g8+res0.005_activation_cols")),
    Row("vector quantization", "SUC − VQ: rotated scalar codebook G=1 K=16 (4 bpw index)",
        Src("quantize", ABL_CB, "vq_k16_g1_rot1s+res0.005_activation_cols")),
    Row("multi-codebook", "SUC − multi-codebook: one codebook M=1, G=2 K=256 (4 bpw index)",
        Src("quantize", ABL_CB, "vq_k256_g2_rot1s+res0.005_activation_cols")),
    Row("residuals", "SUC − residuals (none)", Src("quantize", ABL_RES, "additive_m4_k256_g8_rot1s")),
    Row("residuals", "residual selected by magnitude (columns, no calibration statistics)",
        Src("quantize", ABL_RES, "additive_m4_k256_g8_rot1s+res0.005_magnitude_cols")),
    Row("residuals", "residual activation-selected, CSR structure", Src("quantize", ABL_RES, SUC_CSR)),
    Row("fine-tuning", "SUC (CSR residual) + block-wise fine-tuning", Src("quantize", ABL_FT, SUC_CSR + "+ft"),
        vs="residual activation-selected, CSR structure",
        note="column-residual tuning not implemented; tuned vs untuned CSR pair"),
    Row("semantic sensitivity", "uniform M=4 (protocol B)", Src("allocate", PARETO, "uniform_" + SUC), vs=None),
    Row("semantic sensitivity", "Fisher greedy at the same bpp (4.22, protocol B)",
        Src("allocate", PARETO, greedy_name(4.22)), vs="uniform M=4 (protocol B)"),
    Row("semantic sensitivity", "uniform M=3 (protocol B)", Src("allocate", PARETO, "uniform_" + M3), vs=None),
    Row("semantic sensitivity", "Fisher greedy at the same bpp (3.38, protocol B)",
        Src("allocate", PARETO, greedy_name(3.38)), vs="uniform M=3 (protocol B)"),
)

# Earlier runs that already answer part of an ablation (Q5 protocol unless stated): pairs at equal bits.
EVIDENCE: tuple[Row, ...] = (
    Row("incoherence", "VQ G8 K256 no rotation (1 bpp)", Src("quantize", _q("q1q4_codecs_qwen3_8b"), "vq_k256_g8"),
        vs=None),
    Row("incoherence", "VQ G8 K256 rotated (1 bpp)", Src("quantize", _q("q1q4_codecs_qwen3_8b"), "vq_k256_g8_rot"),
        vs="VQ G8 K256 no rotation (1 bpp)"),
    Row("incoherence", "M=4 + 0.5% CSR, two-sided", Src("quantize", _q("r1_rotation_sides_qwen3_8b"),
                                                       "additive_m4_k256_g8_rot+res0.005_activation"), vs=None),
    Row("incoherence", "M=4 + 0.5% CSR, one-sided", Src("quantize", _q("r1_rotation_sides_qwen3_8b"),
                                                       "additive_m4_k256_g8_rot1s+res0.005_activation"),
        vs="M=4 + 0.5% CSR, two-sided"),
    Row("vector quantization", "additive M=4, no residual (4.0 bpp)",
        Src("quantize", _q("q1q4_codecs_qwen3_8b"), "additive_m4_k256_g8_rot"), vs=None),
    Row("vector quantization", "scalar codebook K=16 per tensor, unrotated (4.0 bpp)",
        Src("quantize", _q("q1q4_codecs_qwen3_8b"), "scalar_k16_tensor"), vs="additive M=4, no residual (4.0 bpp)"),
    Row("multi-codebook", "VQ G4 K256, M=1 (2.0 bpp)", Src("quantize", _q("q1q4_codecs_qwen3_8b"), "vq_k256_g4_rot"),
        vs=None),
    Row("multi-codebook", "additive M=2 G8 K256 (2.0 bpp)",
        Src("quantize", _q("q1q4_codecs_qwen3_8b"), "additive_m2_k256_g8_rot"), vs="VQ G4 K256, M=1 (2.0 bpp)"),
    Row("residuals", "M=3 + 0.5% magnitude", Src("quantize", _q("q5_residual_qwen3_8b"),
                                                "additive_m3_k256_g8_rot+res0.005_magnitude"), vs=None),
    Row("residuals", "M=3 + 0.5% activation", Src("quantize", _q("q5_residual_qwen3_8b"),
                                                 "additive_m3_k256_g8_rot+res0.005_activation"),
        vs="M=3 + 0.5% magnitude"),
    Row("residuals", "M=4 + 0.5% CSR (q5b)", Src("quantize", _q("q5b_residual_structure_qwen3_8b"),
                                                "additive_m4_k256_g8_rot+res0.005_activation"), vs=None),
    Row("residuals", "M=4 + 0.5% columns (q5b)", Src("quantize", _q("q5b_residual_structure_qwen3_8b"),
                                                    "additive_m4_k256_g8_rot+res0.005_activation_cols"),
        vs="M=4 + 0.5% CSR (q5b)"),
    Row("fine-tuning", "M=3 + 0.5% untuned (Q5)", Src("quantize", _q("q5_residual_qwen3_8b"),
                                                     "additive_m3_k256_g8_rot+res0.005_activation"), vs=None),
    Row("fine-tuning", "M=3 + 0.5% block-tuned (Q6)", Src("quantize", _q("q6_finetune_qwen3_8b"),
                                                         "additive_m3_k256_g8_rot+res0.005_activation+ft"),
        vs="M=3 + 0.5% untuned (Q5)"),
    Row("semantic sensitivity", "Q7 uniform M=4 + 0.5% (head affine 4, B)",
        Src("allocate", _q("q7_allocate_qwen3_8b"), "uniform_additive_m4_k256_g8_rot+res0.005_activation"), vs=None),
    Row("semantic sensitivity", "Q7b greedy 4.25 (head allocated, B)",
        Src("allocate", _q("q7b_allocate_head_qwen3_8b"), "greedy_4.2"),
        vs="Q7 uniform M=4 + 0.5% (head affine 4, B)"),
)

# Fused kernel: decode from codes (timing-oriented codes, MLX 4-bit embedding / head).
SPEED: tuple[Row, ...] = (
    Row("fused kernel", "BF16 dense (decoded weights, no compressed kernel)", Src("e2e", "*_e2e_abl_default_qwen3_8b.json", "bf16"), vs=None),
    Row("fused kernel", "MLX affine 4-bit", Src("e2e", "*_e2e_abl_default_qwen3_8b.json", "mlx_q4"), vs=None),
    Row("fused kernel", "SUC M=4, no residual", Src("e2e", "*_e2e_abl_default_qwen3_8b.json", "add4_g8_k256"), vs=None),
    Row("fused kernel", "SUC M=4 + columns residual (all fusions on)",
        Src("e2e", "*_e2e_abl_default_qwen3_8b.json", "add4_g8_k256_res"), vs=None),
    Row("fused kernel", "− fused q\\|k\\|v, gate\\|up launches", Src("e2e", "*_e2e_abl_nofuse_qwen3_8b.json", "add4_g8_k256_res"),
        vs="SUC M=4 + columns residual (all fusions on)"),
    Row("fused kernel", "− staged codebooks in fused groups",
        Src("e2e", "*_e2e_abl_nostaging_qwen3_8b.json", "add4_g8_k256_res"),
        vs="SUC M=4 + columns residual (all fusions on)"),
    Row("fused kernel", "− residual fused into the group launch",
        Src("e2e", "*_e2e_abl_nofuseres_qwen3_8b.json", "add4_g8_k256_res"),
        vs="SUC M=4 + columns residual (all fusions on)"),
    Row("fused kernel", "two-sided rotation (4 extra online rotations / block)",
        Src("e2e", "*_e2e_abl_twosided_qwen3_8b.json", "add4_g8_k256_res"),
        vs="SUC M=4 + columns residual (all fusions on)"),
    # earlier runs (VQ G4 / additive M=3), kept for reference
    Row("fused kernel (earlier)", "R1: VQ G4 two-sided", Src("e2e", "*_e2e_generate_qwen3_8b_rot_baseline.json", "vq_g4_k256"), vs=None),
    Row("fused kernel (earlier)", "R1: VQ G4 one-sided", Src("e2e", "*_e2e_generate_qwen3_8b_rot_onesided.json", "vq_g4_k256"),
        vs="R1: VQ G4 two-sided"),
    Row("fused kernel (earlier)", "K6: M=3 + columns, all on", Src("e2e", "*_e2e_k6_after_qwen3_8b.json", "add3_g8_k256_res"), vs=None),
    Row("fused kernel (earlier)", "K6: M=3 + columns, staging off", Src("e2e", "*_e2e_k6_nostaging_qwen3_8b.json", "add3_g8_k256_res"),
        vs="K6: M=3 + columns, all on"),
)

ENTROPY = (Src("entropy", "*_entropy_coding_qwen3_0.6b.json"), Src("entropy", "*_entropy_coding_qwen3_8b.json"))
RTN = (("RTN affine 4-bit g64", "affine_b4_g64"), ("RTN affine 3-bit g64", "affine_b3_g64"),
       ("RTN affine 2-bit g64", "affine_b2_g64"))


# ------------------------------------------------------------------------------------------------ loading

_cache: dict[str, dict | None] = {}


def newest(pattern: str, root: Path) -> Path | None:
    found = sorted(root.glob(pattern))
    return found[-1] if found else None


def _load(pattern: str, root: Path) -> tuple[dict | None, Path | None]:
    p = newest(pattern, root)
    if p is None:
        return None, None
    if str(p) not in _cache:
        _cache[str(p)] = json.loads(p.read_text())
    return _cache[str(p)], p


def quality_row(src: Src, root: Path) -> dict | None:
    """bpp / PPL / KL / top-1 / size of one quantize, allocate or compress entry (None if not run yet)."""
    d, p = _load(src.pattern, root)
    if d is None:
        return None
    if src.kind == "quantize":
        e = next((r for r in d["results"] if r["codec"] == src.key), None)
    elif src.kind == "allocate":
        e = next((r for r in d["plans"] if r["plan"] == src.key), None)
    elif src.kind == "compress":
        e = d
    else:
        raise ValueError(src.kind)
    if e is None:
        return None
    kl = e["quality"].get("kl") or {}
    return {"bpp": e["size"]["bpp"], "gb": e["size"]["bits_total"] / 8e9, "ppl": e["quality"]["perplexity"]["ppl"],
            "kl": kl.get("kl_mean"), "top1": kl.get("top1_agreement"), "ref_ppl": d["reference"]["perplexity"]["ppl"],
            "file": p.name, "head": e.get("head") or e.get("embedding_codec"), "by_codec": e.get("by_codec")}


def speed_row(src: Src, root: Path) -> dict | None:
    d, p = _load(src.pattern, root)
    if d is None:
        return None
    s = next((r for r in d["summary"] if r["variant"] == src.key), None)
    if s is None:
        return None
    return {**{k: s.get(k) for k in ("B1_decode_tok_s", "B4_decode_tok_s", "B16_decode_tok_s", "B1_decode_ms_step",
                                     "resident_MiB")}, "file": p.name}


# ------------------------------------------------------------------------------------------------ rendering

def _f(x, fmt: str) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def _rel(a, b) -> str:
    return "—" if a is None or b is None or not b else f"{100 * (a / b - 1):+.1f}%"


def _pending(src: Src) -> str:
    return f"pending (`{src.pattern.removeprefix('*_').removesuffix('.json')}`)"


def quality_table(rows: tuple[Row, ...], root: Path) -> list[str]:
    out = ["| Component | Variant | bpp | Size (GB) | PPL | ΔPPL | KL | ΔKL | Top-1 | Source |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    got = {r.label: quality_row(r.src, root) for r in rows}
    for r in rows:
        q = got[r.label]
        if q is None:
            out.append(f"| {r.group} | {r.label} | | | | | | | | {_pending(r.src)} |")
            continue
        base = got.get(r.vs) if r.vs else None
        dppl = _rel(q["ppl"], base["ppl"]) if base else "—"
        dkl = _rel(q["kl"], base["kl"]) if base else "—"
        src = f"`{q['file']}`" + (f"; {r.note}" if r.note else "")
        out.append(f"| {r.group} | {r.label} | {q['bpp']:.3f} | {q['gb']:.2f} | {q['ppl']:.3f} | {dppl} "
                   f"| {_f(q['kl'], '.4f')} | {dkl} | {_f(q['top1'], '.3f')} | {src} |")
    return out


def speed_table(root: Path) -> list[str]:
    out = ["| Group | Variant | Decode B=1 (tok/s) | Δ B=1 | B=4 | B=16 | ms/step B=1 | Resident (MiB) | Source |",
           "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    got = {r.label: speed_row(r.src, root) for r in SPEED}
    for r in SPEED:
        s = got[r.label]
        if s is None:
            out.append(f"| {r.group} | {r.label} | | | | | | | {_pending(r.src)} |")
            continue
        base = got.get(r.vs) if r.vs else None
        d1 = _rel(s["B1_decode_tok_s"], base["B1_decode_tok_s"]) if base else "—"
        out.append(f"| {r.group} | {r.label} | {_f(s['B1_decode_tok_s'], '.1f')} | {d1} "
                   f"| {_f(s['B4_decode_tok_s'], '.1f')} | {_f(s['B16_decode_tok_s'], '.1f')} "
                   f"| {_f(s['B1_decode_ms_step'], '.2f')} | {_f(s['resident_MiB'], '.0f')} | `{s['file']}` |")
    return out


def entropy_lines(root: Path) -> list[str]:
    out = ["| Model | SUC1 file bpp (fixed-width, execution layout) | entropy-coded bpp (storage only) | saving | Source |",
           "|---|---:|---:|---:|---|"]
    for src in ENTROPY:
        d, p = _load(src.pattern, root)
        model = src.pattern.split("_qwen3_")[-1].removesuffix(".json").upper()
        if d is None:
            out.append(f"| Qwen3-{model} | | | | {_pending(src)} |")
            continue
        s = d["summary"]
        out.append(f"| Qwen3-{model} | {s['bpp_file']:.3f} | {s['bpp_entropy_coded']:.3f} "
                   f"| {_rel(s['bpp_entropy_coded'], s['bpp_file'])} | `{p.name}` |")
    return out


def pareto_points(root: Path) -> dict[str, list[dict]]:
    """Series of (bpp, PPL, KL) points: new greedy plans, earlier Q7b plans, uniform codecs, RTN baselines."""
    series: dict[str, list[dict]] = {"SUC greedy (pareto_allocate)": [], "SUC uniform (pareto_allocate)": [],
                                     "Q7b greedy (CSR, two-sided)": [], "RTN affine (all tensors)": []}
    for t in PARETO_TARGETS:
        q = quality_row(Src("allocate", PARETO, greedy_name(PARETO_BUDGETS.get(t, t))), root)
        if q:
            series["SUC greedy (pareto_allocate)"].append({**q, "target": t, "name": greedy_name(PARETO_BUDGETS.get(t, t))})
    for name in (SUC, M3, G4):
        q = quality_row(Src("allocate", PARETO, "uniform_" + name), root)
        if q:
            series["SUC uniform (pareto_allocate)"].append({**q, "target": None, "name": name})
    for b in (3.0, 3.5, 4.0, 4.25):
        q = quality_row(Src("allocate", _q("q7b_allocate_head_qwen3_8b"), greedy_name(b)), root)
        if q:
            series["Q7b greedy (CSR, two-sided)"].append({**q, "target": b, "name": greedy_name(b)})
    for label, key in RTN:
        q = quality_row(Src("quantize", _q("q1q4_codecs_qwen3_8b"), key), root)
        if q:
            series["RTN affine (all tensors)"].append({**q, "target": None, "name": label})
    return series


def pareto_table(series: dict[str, list[dict]]) -> list[str]:
    pts = [(s, p) for s, ps in series.items() for p in ps]
    # non-dominated in (bpp, KL): no other point with lower-or-equal bpp AND lower KL
    def dominated(p):
        return any(o is not p and o["kl"] is not None and p["kl"] is not None and o["bpp"] <= p["bpp"] + 1e-9
                   and o["kl"] < p["kl"] for _, o in pts)
    out = ["| Target bpp | Series | Plan / codec | bpp | Size (GB) | PPL | ΔPPL vs BF16 | KL | Top-1 | Head | Frontier |",
           "|---:|---|---|---:|---:|---:|---:|---:|---:|---|:---:|"]
    sent = {s: {p["target"] for p in ps} for s, ps in series.items()}
    for t in PARETO_TARGETS:
        if t not in sent["SUC greedy (pareto_allocate)"]:
            out.append(f"| {t:g} | SUC greedy (pareto_allocate) | {greedy_name(PARETO_BUDGETS.get(t, t))}: "
                       f"{_pending(Src('allocate', PARETO))} | | | | | | | | |")
    for s, p in sorted(pts, key=lambda sp: -sp[1]["bpp"]):
        head = p["head"]
        if isinstance(head, dict):
            head = "/".join(sorted(set(head.values())))
        out.append(f"| {_f(p['target'], 'g')} | {s} | {p['name']} | {p['bpp']:.3f} | {p['gb']:.2f} | {p['ppl']:.3f} "
                   f"| {_rel(p['ppl'], p['ref_ppl'])} | {_f(p['kl'], '.4f')} | {_f(p['top1'], '.3f')} "
                   f"| {head or '—'} | {'●' if not dominated(p) else ''} |")
    return out


def plot(series: dict[str, list[dict]], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    markers = {"SUC greedy (pareto_allocate)": "o-", "SUC uniform (pareto_allocate)": "s",
               "Q7b greedy (CSR, two-sided)": "^--", "RTN affine (all tensors)": "x:"}
    ref = next((p["ref_ppl"] for ps in series.values() for p in ps), None)
    for ax, key, title in ((axes[0], "ppl", "WikiText-2 test perplexity"), (axes[1], "kl", "KL vs BF16 (held-out)")):
        for s, ps in series.items():
            ps = sorted((p for p in ps if p[key] is not None), key=lambda p: p["bpp"])
            if ps:
                ax.plot([p["bpp"] for p in ps], [p[key] for p in ps], markers[s], label=s)
        if key == "ppl" and ref:
            ax.axhline(ref, color="grey", lw=0.8, ls="--", label=f"BF16 ({ref:.2f})")
        ax.set_yscale("log")
        ax.set_xlabel("actual bits per parameter (whole model)")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    fig.suptitle("Qwen3-8B: quality vs size")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return True


def render(root: Path) -> tuple[str, dict]:
    series = pareto_points(root)
    lines = ["# Qwen3-8B — ablations and Pareto frontier", "",
             "Generated by `benchmarks/make_tables.py` from `experiments/results/` (newest file per run); "
             "row → config map in `docs/ablations_plan.md`. bpp = actual whole-model bits per parameter "
             "(codebooks, scales, indices, residuals, metadata). PPL: full WikiText-2 test; KL: 16 held-out "
             "validation sequences vs BF16. Δ columns are relative to the row named in the plan, same protocol only.",
             "", "## Mandatory ablations (protocol A: suc quantize, head affine 4-bit, rotation groups)", ""]
    lines += quality_table(ABLATIONS, root)
    lines += ["", "Adaptive (multiplicative-weights / online) allocation, spec §29 / §40: **not implemented** — no row. "
              "The allocator rows above are the static Fisher-greedy allocator (§39).", "",
              "## Entropy coding (storage only, spec §15)", ""]
    lines += entropy_lines(root)
    lines += ["", "## Fused kernel (decode from codes on the M1 Ultra; timing-oriented codes)", ""]
    lines += speed_table(root)
    lines += ["", "## Earlier evidence (Q5 protocol unless marked B; equal-bit pairs)", ""]
    lines += quality_table(EVIDENCE, root)
    lines += ["", "## Pareto frontier", "", "Frontier ● = no other listed point has lower-or-equal bpp and lower KL.", ""]
    lines += pareto_table(series)
    return "\n".join(lines) + "\n", series


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--md", default=None, help="write the markdown here instead of stdout")
    ap.add_argument("--plot", default=None, help="PNG or SVG path for PPL / KL vs bpp (needs matplotlib)")
    args = ap.parse_args()
    md, series = render(Path(args.results))
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md)
    else:
        print(md)
    if args.plot and not plot(series, Path(args.plot)):
        print("matplotlib not available: plot skipped (markdown only)")


if __name__ == "__main__":
    main()
