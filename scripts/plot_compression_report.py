"""Figures and the summary table of docs/compression_report.md, read from the result JSONs in experiments/results.

Every number comes from a committed result file (listed in SOURCES); nothing is typed in by hand except the on-disk
size of the three frontier checkpoints (index `total_size`, from docs/arch/*.md, not a result JSON) and the 111 GB
runtime budget of the Mac Studio. Output: four SVGs under docs/figures/ and a Markdown table on stdout.

matplotlib is not a project dependency; run it in a throwaway environment (no project venv involved):

  uv run --no-project --python 3.12 --with matplotlib python scripts/plot_compression_report.py
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "docs" / "figures"

SOURCES = {
    "c06": "20261002T202145Z_compress_qwen3_0.6b_final.json",
    "cpp06": "20261002T205833Z_cpp_check_qwen3_0.6b_final_m3.json",
    "e2e06": "20261002T210018Z_e2e_final_baseline_qwen3_0.6b_m3.json",
    "rtn06": "20261002T204839Z_final_baseline_qwen3_0.6b.json",
    "c8": "20261002T191301Z_compress_qwen3_8b_final.json",
    "perf8": "20261003T015734Z_perf_qwen3_8b_final.json",
    "eval8": "20261003T032730Z_eval_qwen3_8b_final.json",
    "q7_8": "20261001T072625Z_q7_allocate_qwen3_8b.json",
    "c30": "20261002T212215Z_compress_qwen3_30b_a3b_final.json",
    "perf30": "20261003T031427Z_perf_qwen3_30b_a3b_final.json",
    "eval30": "20261003T034721Z_eval_qwen3_30b_a3b_final.json",
    "q7_30": "20261002T070150Z_q7_allocate_qwen3_30b_a3b.json",
    "q38": "20261003T132100Z_queue_Qwen__Qwen3.8-Flash-Next.json",
    "q38_suc": "20261004T092508Z_quality_eval_qwen3.8_flash_next.json",
    "q38_mlx": "20261004T102809Z_quality_eval_qwen3.8_flash_next.json",
    "q38_cpp": "20261006T213008Z_gen_cpp_qwen3.8_flash_next_ngram4.json",
    "eval38": "20261004T111255Z_eval_qwen3.8_flash_next.json",
    "ds_max": "20261006T115604Z_compress_deepseek_v4_flash_native_max.json",
    "ds_stream": "20261006T153009Z_compress_deepseek_v4_flash_native_stream.json",
    "perf_ds": "20261006T100712Z_perf_deepseek_v4_flash.json",
    "gen_ds_stream": "20261006T204955Z_gen_cpp_deepseek_v4_flash_native_stream.json",
    "gen_ds_max": "20261006T212714Z_gen_cpp_deepseek_v4_flash_native_max.json",
    "eval_ds": "20261006T001744Z_eval_deepseek_v4_flash.json",
    "glm": "20261005T022627Z_compress_glm5.3_flash.json",
    "perf_glm": "20261006T182924Z_perf_glm5.3_flash_cpp.json",
    "eval_glm": "20261006T200231Z_eval_glm5.3_flash_cpp.json",
}
# on-disk checkpoint size (safetensors index total_size) where it differs from BF16: docs/arch/*.md
ON_DISK = {"qwen38": 359_999_963_128, "ds_max": 166_878_536_440, "ds_stream": 166_878_536_440,
           "glm": 328_326_771_576}
BUDGET_GB = 111.0
PREVIEW = os.environ.get("SUC_PLOT_PREVIEW")   # optional directory for PNG previews

# palette (dataviz reference, first three categorical slots: validated all-pairs) + neutral ink readable on light and
# dark backgrounds (the SVGs are transparent)
C_SUC, C_BF16, C_MLX = "#2a78d6", "#eb6834", "#1baf7a"
INK, GRID = "#777777", "#9a9a9a"


def load(key):
    return json.loads((RES / SOURCES[key]).read_text())


def perf_row(perf, system, ctx, batch=1):
    for r in perf["rows"]:
        if r["system"] == system and r["ctx"] == ctx and r["batch"] == batch:
            return r
    raise KeyError((system, ctx))


def med(x):
    return x["median"] if isinstance(x, dict) else x


def eval_score(ev, system):
    s = ev["systems"][system]["summary"]
    return f"{s['passed']}/{s['graded']}"


def collect():
    """One dict per model (smallest to largest); None = not measured."""
    out = []

    # --- Qwen3-0.6B (Apple M3 laptop; dev model, never run on the Studio in the final form)
    c, cpp, e2e = load("c06"), load("cpp06"), load("e2e06")
    t = cpp["timing"]
    e = {r["variant"]: r for r in e2e["summary"]}
    rtn = {r["codec"]: r for r in load("rtn06")["results"]}["affine_b4_g64"]
    pre_ms = t["prefill"]["cpp_batch_ms_per_token"]
    out.append(dict(
        key="q06", name="Qwen3-0.6B", short="0.6B", params=c["size"]["parameter_count"], machine="M3 16 GB",
        orig=c["preflight"]["memory"]["checkpoint_bytes"], disk=None, file=c["size"]["file_bytes"],
        bpp=c["size"]["file_bpp"], resident=t["cpp_memory"]["mapped_file"] + t["cpp_memory"]["phys_footprint"],
        decode=t["cpp_ms_per_token"], prefill=1000.0 / pre_ms, prefill_note=f"batch prefill, {t['prefill']['tokens']} tok",
        ttft128=None, ttft2k=None,
        ppl=c["quality"]["perplexity"]["ppl"], ref_ppl=c["reference"]["perplexity"]["ppl"],
        kl=c["quality"]["kl"]["kl_mean"], top1=c["quality"]["kl"]["top1_agreement"], delta_ce=c["quality"]["kl"]["delta_ce"], score=None,
        bf16=dict(decode=e["bf16"]["B1_decode_ms_step"], prefill=e["bf16"]["B1_prefill_tok_s"],
                  ttft128=e["bf16"]["B1_ttft_ms"] / 1000, mem=e["bf16"]["B1_peak_MiB"] * 2**20),
        mlx=dict(decode=e["mlx_q4"]["B1_decode_ms_step"], prefill=e["mlx_q4"]["B1_prefill_tok_s"],
                 ttft128=e["mlx_q4"]["B1_ttft_ms"] / 1000, mem=e["mlx_q4"]["B1_peak_MiB"] * 2**20,
                 bpp=rtn["size"]["bpp"] if "size" in rtn else 4.50, kl=rtn["quality"]["kl"]["kl_mean"],
                 ppl=rtn["quality"]["perplexity"]["ppl"]),
        src=["c06", "cpp06", "e2e06", "rtn06"]))

    # --- Qwen3-8B / Qwen3-30B-A3B (Studio, suc perf matched harness)
    for key, ck, pk, ek, qk, sysname, evsys, name, short in [
            ("q8", "c8", "perf8", "eval8", "q7_8", "suc_cpp", "suc_cpp", "Qwen3-8B", "8B"),
            ("q30", "c30", "perf30", "eval30", "q7_30", "suc_cpp_uniform", "suc_cpp_uniform", "Qwen3-30B-A3B",
             "30B-A3B")]:
        c, p, ev = load(ck), load(pk), load(ek)
        r128, r2k = perf_row(p, sysname, 128), perf_row(p, sysname, 2048)
        rtn = load(qk)["plans"][-1]                     # last plan of the Q7 run = uniform RTN affine 4-bit
        sysd = p["systems"]

        def base(s):
            a, b = perf_row(p, s, 128), perf_row(p, s, 2048)
            return dict(decode=med(a["decode_ms"]), decode2k=med(b["decode_ms"]), prefill=med(a["prefill_tok_s"]),
                        prefill2k=med(b["prefill_tok_s"]), ttft128=med(a["ttft_ms"]) / 1000, ttft2k=med(b["ttft_ms"]) / 1000,
                        mem=med(a["peak_memory_bytes"]))
        mlx = base("mlx_q4")
        mlx.update(bpp=sysd["mlx_q4"]["bpp"], kl=rtn["quality"]["kl"]["kl_mean"], ppl=rtn["quality"]["perplexity"]["ppl"])
        out.append(dict(
            key=key, name=name, short=short, params=c["size"]["parameter_count"], machine="M1 Ultra",
            orig=c["preflight"]["memory"]["checkpoint_bytes"], disk=None, file=c["size"]["file_bytes"],
            bpp=c["size"]["file_bpp"], resident=c["size"]["file_bytes"] + med(r128["peak_memory_bytes"]),
            decode=med(r128["decode_ms"]), decode2k=med(r2k["decode_ms"]), prefill=med(r128["prefill_tok_s"]),
            prefill2k=med(r2k["prefill_tok_s"]), prefill_note="suc perf ctx 128",
            ttft128=med(r128["ttft_ms"]) / 1000, ttft2k=med(r2k["ttft_ms"]) / 1000,
            ppl=c["quality"]["perplexity"]["ppl"], ref_ppl=c["reference"]["perplexity"]["ppl"],
            kl=c["quality"]["kl"]["kl_mean"], top1=c["quality"]["kl"]["top1_agreement"], delta_ce=c["quality"]["kl"]["delta_ce"], score=eval_score(ev, evsys),
            bf16=base("bf16"), mlx=mlx, src=[ck, pk, ek, qk]))

    # --- Qwen3.8-Flash-Next (ngram4 file; C++ suc1-generate, 256 greedy tokens, same harness as the DeepSeek rows)
    q, qs, qm, ev = load("q38"), load("q38_suc"), load("q38_mlx"), load("eval38")
    g = load("q38_cpp")["run"]
    s = qs["systems"]["suc_python_ngram4"]
    m = qm["systems"]["mlx_q4"]
    ev_mlx = statistics.median(x["decode_ms_median"] for x in ev["systems"]["mlx_q4"]["cases"])
    ev_cpp = statistics.median(x["decode_ms_median"] for x in ev["systems"]["suc_cpp_ngram4"]["cases"])
    fb = s["info"]["file_bytes"]
    out.append(dict(
        key="qwen38", name="Qwen3.8-Flash-Next", short="Qwen3.8", params=q["size"]["parameter_count"],
        machine="M1 Ultra", orig=q["preflight"]["memory"]["checkpoint_bytes"], disk=ON_DISK["qwen38"], file=fb,
        bpp=fb * 8 / q["size"]["parameter_count"], resident=g["memory"]["mapped_file"] + g["memory"]["phys_footprint"],
        decode=g["decode_ms_per_token_median"], prefill=g["prompt_tokens"] / g["prefill_s"],
        prefill_note=f"suc1-generate, {g['prompt_tokens']}-token prompt", ttft128=None, ttft2k=None,
        ttft_short=g["prefill_s"] + g["decode_ms_per_token"][0] / 1000,
        ppl=s["perplexity"]["ppl"], ref_ppl=qs["reference"]["bf16_ppl"]["ppl"], kl=s["kl"]["kl_mean"],
        delta_ce=s["kl"]["delta_ce"],
        top1=s["kl"]["top1_agreement"], score=eval_score(ev, "suc_cpp_ngram4"),
        eval_harness=dict(mlx=ev_mlx, cpp=ev_cpp),
        bf16=None,
        mlx=dict(decode=None, mem=m["info"]["resident_bytes"], bpp=m["info"]["bpp"], kl=m["kl"]["kl_mean"],
                 ppl=m["perplexity"]["ppl"]),
        src=["q38", "q38_suc", "q38_mlx", "q38_cpp", "eval38"]))

    # --- DeepSeek-V4-Flash V1 native_max (in memory) / V3 native_stream (SSD expert streaming)
    ev = load("eval_ds")
    c, r = load("ds_max"), load("gen_ds_max")["run"]
    out.append(dict(
        key="ds_max", name="DeepSeek-V4-Flash V1 (native_max)", short="DS-V4 V1", params=c["size"]["parameter_count"],
        machine="M1 Ultra", orig=c["preflight"]["memory"]["checkpoint_bytes"], disk=ON_DISK["ds_max"],
        file=c["size"]["file_bytes"], bpp=c["size"]["file_bpp"],
        resident=r["memory"]["mapped_file"] + r["memory"]["phys_footprint"],
        decode=r["decode_ms_per_token_median"], decode_mean=statistics.mean(r["decode_ms_per_token"]),
        decode_gpu=r["decode_gpu_ms_per_token_median"],
        prefill=r["prompt_tokens"] / r["prefill_s"], prefill_note=f"suc1-generate, {r['prompt_tokens']}-token prompt",
        ttft128=None, ttft2k=None, ttft_short=r["prefill_s"] + r["decode_ms_per_token"][0] / 1000,
        ppl=c["quality"]["perplexity"]["ppl"], ref_ppl=c["reference"]["perplexity"]["ppl"],
        kl=c["quality"]["kl"]["kl_mean"], top1=c["quality"]["kl"]["top1_agreement"], delta_ce=c["quality"]["kl"]["delta_ce"],
        score=eval_score(ev, "suc_cpp_native") + " (native)", bf16=None, mlx=None,
        src=["ds_max", "gen_ds_max", "eval_ds"]))
    c, g = load("ds_stream"), load("gen_ds_stream")
    r = g["run"]
    out.append(dict(
        key="ds_stream", name="DeepSeek-V4-Flash V3 (native_stream)", short="DS-V4 V3",
        params=c["size"]["parameter_count"], machine="M1 Ultra", orig=c["preflight"]["memory"]["checkpoint_bytes"],
        disk=ON_DISK["ds_stream"], file=c["size"]["file_bytes"], bpp=c["size"]["file_bpp"],
        resident=r["memory"]["mapped_file"] + r["memory"]["phys_footprint"], streamed=True,
        decode=r["decode_ms_per_token_median"], decode_mean=statistics.mean(r["decode_ms_per_token"]),
        decode_gpu=r["decode_gpu_ms_per_token_median"], hit=r["expert_streaming"]["hit_rate"],
        prefill=r["prompt_tokens"] / r["prefill_s"], prefill_note=f"suc1-generate, {r['prompt_tokens']}-token prompt",
        ttft128=None, ttft2k=None, ttft_short=r["prefill_s"] + r["decode_ms_per_token"][0] / 1000,
        ppl=c["quality"]["perplexity"]["ppl"], ref_ppl=c["reference"]["perplexity"]["ppl"],
        kl=c["quality"]["kl"]["kl_mean"], top1=c["quality"]["kl"]["top1_agreement"], delta_ce=c["quality"]["kl"]["delta_ce"], score=None,
        bf16=None, mlx=None, src=["ds_stream", "gen_ds_stream"]))

    # --- GLM-5.3-Flash (streamed)
    c, p, ev = load("glm"), load("perf_glm"), load("eval_glm")
    a, b = perf_row(p, "suc_cpp_streamed", 128), perf_row(p, "suc_cpp_streamed", 2048)
    out.append(dict(
        key="glm", name="GLM-5.3-Flash (streamed)", short="GLM-5.3", params=c["size"]["parameter_count"],
        machine="M1 Ultra", orig=c["preflight"]["memory"]["checkpoint_bytes"], disk=ON_DISK["glm"],
        file=c["size"]["file_bytes"], bpp=c["size"]["file_bpp"],
        resident=a["memory"]["mapped_file"] + a["memory"]["phys_footprint"], streamed=True,
        decode=med(a["decode_ms"]), decode2k=med(b["decode_ms"]), prefill=med(a["prefill_tok_s"]),
        prefill2k=med(b["prefill_tok_s"]), prefill_note="suc perf ctx 128",
        ttft128=med(a["ttft_ms"]) / 1000, ttft2k=med(b["ttft_ms"]) / 1000,
        ppl=c["quality"]["perplexity"]["ppl"], ref_ppl=c["reference"]["perplexity"]["ppl"],
        kl=c["quality"]["kl"]["kl_mean"], top1=c["quality"]["kl"]["top1_agreement"], delta_ce=c["quality"]["kl"]["delta_ce"],
        score=eval_score(ev, "suc_cpp_streamed"), bf16=None, mlx=None, src=["glm", "perf_glm", "eval_glm"]))
    return out


# ------------------------------------------------------------------------------------------------ figures
def style(ax):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.grid(True, color=GRID, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def legend(ax, **kw):
    lg = ax.legend(frameon=False, fontsize=9, **kw)
    for t in lg.get_texts():
        t.set_color(INK)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.patch.set_alpha(0)
    fig.savefig(FIG / name, format="svg", transparent=True, bbox_inches="tight", metadata={"Date": None})
    if PREVIEW:                                  # PNG copies on a white background, for a visual check only
        fig.savefig(Path(PREVIEW) / (name + ".png"), format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


GB = FuncFormatter(lambda v, _: f"{v:g}")


def fig_size(models):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    style(ax)
    n = len(models)
    ys = list(range(n))[::-1]
    h = 0.36
    for y, m in zip(ys, models):
        ax.barh(y + h / 2, m["orig"] / 1e9, height=h - 0.04, color=C_BF16)
        ax.barh(y - h / 2, m["file"] / 1e9, height=h - 0.04, color=C_SUC)
        ax.text(m["orig"] / 1e9 * 1.08, y + h / 2, f"{m['orig'] / 1e9:,.1f} GB", va="center", fontsize=8, color=INK)
        ax.text(m["file"] / 1e9 * 1.08, y - h / 2,
                f"{m['file'] / 1e9:,.1f} GB  ({m['orig'] / m['file']:.1f}x, {m['bpp']:.2f} bpp)",
                va="center", fontsize=8, color=INK)
        if m.get("disk"):
            ax.plot([m["disk"] / 1e9] * 2, [y + 0.02, y + h - 0.02], color=INK, linewidth=2)
    ax.axvline(BUDGET_GB, color=INK, linestyle="--", linewidth=1)
    ax.text(BUDGET_GB * 1.03, n - 0.45, "111 GB runtime budget", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 8000)
    ax.xaxis.set_major_formatter(GB)
    ax.set_yticks(ys)
    ax.set_yticklabels([m["name"] for m in models], fontsize=9, color=INK)
    ax.set_xlabel("GB (log scale)")
    handles = [Patch(color=C_BF16, label="original, BF16-equivalent"), Patch(color=C_SUC, label="SUC1 file"),
               Line2D([], [], color=INK, linewidth=2, label="on-disk checkpoint (FP8 / MXFP4)")]
    legend(ax, handles=handles, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=3)
    save(fig, "compression_size.svg")


# ------------------------------------------------------------------------------------------------ community GGUF
# newest gguf_quality_* / gguf_perf_* result per key (scripts/gguf_compare.py); SUC counterpart = key in collect()
GGUF = [("ds_iq2", "ds_max", "DS-V4 IQ2_XXS mix"), ("ds_mxfp4", "ds_stream", "DS-V4 MXFP4 (streamed)"),
        ("glm_q2", "glm", "GLM-5.3 Q2 (streamed)"), ("q38_q4", "qwen38", "Qwen3.8 Q4")]
C_GGUF = "#9b59d0"


def _newest(kind, key):
    f = sorted(RES.glob(f"*_{kind}_{key}.json"))
    return (json.loads(f[-1].read_text()), f[-1].name) if f else (None, None)


def collect_gguf(models):
    by = {m["key"]: m for m in models}
    out = []
    for key, suc, short in GGUF:
        q, qf = _newest("gguf_quality", key)
        p, pf = _newest("gguf_perf", key)
        if q is None and p is None:
            continue
        g = (q or p)["gguf"]
        m = by[suc]
        sp = (p or {}).get("summary", {}).get("short_prompt") or {}
        out.append(dict(key=key, short=short, suc=m, file=g["file_bytes"], bpp=g["file_bytes"] * 8 / m["params"],
                        kl=(q or {}).get("kl"), ppl=(q or {}).get("perplexity"), decode=sp.get("decode_ms_median"),
                        prefill_s=sp.get("prefill_s"), prompt_tokens=sp.get("prompt_tokens"), ttft=sp.get("ttft_s"),
                        streamed="--ssd-streaming" in (q or p)["config"]["ds4_args"], src=[x for x in (qf, pf) if x]))
    return out


def gguf_table(gg):
    def f(x, nd=2):
        return "—" if x is None else f"{x:,.{nd}f}"
    print("| Model | file | GB | bpp | KL | top-1 | ΔCE | decode ms/tok | prefill tok/s | TTFT s | GGUF PPL, first windows |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in gg:
        m, kl = g["suc"], g["kl"] or {}
        pp = g["ppl"]
        ppl = "—" if not pp else f"{pp['ppl']:.3f} ({pp['windows']} × {pp['seq_len']})"
        pre = None if not g["prefill_s"] else g["prompt_tokens"] / g["prefill_s"]
        print(f"| {g['short']} | GGUF (ds4){', streamed' if g['streamed'] else ''} | {g['file'] / 1e9:.2f} | "
              f"{g['bpp']:.3f} | {f(kl.get('kl_mean'), 4)} | {f(kl.get('top1_agreement'), 3)} | "
              f"{f(kl.get('delta_ce'), 4)} | {f(g['decode'])} | {f(pre, 1)} | {f(g['ttft'])} | {ppl} |")
        print(f"| | SUC1 {m['short']}{', streamed' if m.get('streamed') else ''} | {m['file'] / 1e9:.2f} | "
              f"{m['bpp']:.3f} | {m['kl']:.4f} | {m['top1']:.3f} | {f(m.get('delta_ce'), 4)} | {f(m['decode'])} | "
              f"{f(m['prefill'], 1)} | {f(m.get('ttft_short'))} | — |")
    for g in gg:
        print("  gguf sources:", g["key"], g["src"])


def fig_decode(models):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    style(ax)
    seen = set()

    def pt(x, y, color, label, marker="o", filled=True):
        lab = label if label not in seen else None
        seen.add(label)
        ax.scatter([x], [y], s=60, marker=marker, color=color if filled else "none", edgecolors=color,
                   linewidths=2, label=lab, zorder=3)
    xs, ys = [], []
    for m in models:
        x = m["params"] / 1e9
        if m.get("streamed"):
            pt(x, m["decode"], C_SUC, "SUC C++, SSD expert streaming", marker="D", filled=False)
        elif m.get("speed_proxy"):
            pt(x, m["decode"], C_SUC, "SUC C++, resident", marker="o")
        else:
            pt(x, m["decode"], C_SUC, "SUC C++, resident", marker="o")
            xs.append(x), ys.append(m["decode"])
        ax.annotate(m["short"] + ("*" if m.get("speed_proxy") else ""), (x, m["decode"]), textcoords="offset points", xytext=(7, -3), fontsize=8, color=INK)
        if m.get("bf16") and m["bf16"].get("decode"):
            pt(x, m["bf16"]["decode"], C_BF16, "original BF16 (MLX)", marker="s")
        if m.get("mlx") and m["mlx"].get("decode"):
            pt(x, m["mlx"]["decode"], C_MLX, "MLX affine 4-bit", marker="^")
    ax.text(0.98, 0.03, "Qwen3.8, DeepSeek, GLM: BF16 and MLX 4-bit do not fit 128 GB (not measurable)\n"
            "Qwen3.8, DS-V4: suc1-generate, short prompt, 256 tokens; others: suc perf ctx 128\n"
            "0.6B: Apple M3 laptop; all others: M1 Ultra", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(4, 700)
    ax.xaxis.set_major_formatter(GB)
    ax.yaxis.set_major_formatter(GB)
    ax.set_xlabel("total parameters, billions (log)")
    ax.set_ylabel("decode ms / token, batch 1 (log; lower is faster)")
    legend(ax, loc="upper left")
    save(fig, "compression_decode.svg")


def fig_quality(models):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    style(ax)
    done_mlx = False
    for m in models:
        ax.scatter([m["bpp"]], [m["kl"]], s=60, marker="D" if m.get("streamed") else "o",
                   color="none" if m.get("streamed") else C_SUC, edgecolors=C_SUC, linewidths=2, zorder=3)
        ax.annotate(m["short"], (m["bpp"], m["kl"]), textcoords="offset points", xytext=(7, -3), fontsize=8,
                    color=INK)
        if m.get("mlx") and m["mlx"].get("kl") is not None:
            ax.scatter([m["mlx"]["bpp"]], [m["mlx"]["kl"]], s=60, marker="^", color=C_MLX, zorder=3,
                       label=None if done_mlx else "MLX / RTN affine 4-bit, same model")
            ax.plot([m["bpp"], m["mlx"]["bpp"]], [m["kl"], m["mlx"]["kl"]], color=GRID, linewidth=1, zorder=2)
            done_mlx = True
    for g in GG:
        if not g["kl"]:
            continue
        ax.scatter([g["bpp"]], [g["kl"]["kl_mean"]], s=60, marker="s", color="none" if g["streamed"] else C_GGUF,
                   edgecolors=C_GGUF, linewidths=2, zorder=3)
        ax.annotate(g["short"], (g["bpp"], g["kl"]["kl_mean"]), textcoords="offset points", xytext=(7, -3),
                    fontsize=8, color=INK)
        ax.plot([g["bpp"], g["suc"]["bpp"]], [g["kl"]["kl_mean"], g["suc"]["kl"]], color=C_GGUF, alpha=0.4,
                linewidth=1, linestyle=":", zorder=2)
    if any(g["kl"] for g in GG):
        ax.scatter([], [], s=60, marker="s", color=C_GGUF, label="community GGUF (ds4), same KL positions")
    ax.scatter([], [], s=60, marker="o", color=C_SUC, label="SUC1 file, resident")
    ax.scatter([], [], s=60, marker="D", color="none", edgecolors=C_SUC, linewidths=2,
               label="SUC1 file, SSD expert streaming")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.02, 0.05, 0.1, 0.2, 0.5]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(0.008, 0.6)
    ax.set_xlabel("bits per parameter, whole file incl. all overhead")
    ax.set_ylabel("KL vs full-precision reference (log)")
    ax.set_xlim(2.3, 5.0)
    legend(ax, loc="upper right")
    save(fig, "compression_quality.svg")


def fig_prefill(models):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax in (a1, a2):
        style(ax)
    names = [m["short"] for m in models]
    ys = list(range(len(models)))[::-1]
    h = 0.26
    for y, m in zip(ys, models):
        matched = "suc perf" in m["prefill_note"]
        a1.barh(y, m["prefill"], height=h, color=C_SUC if matched else "none", edgecolor=C_SUC, linewidth=1.5,
                hatch=None if matched else "///")
        a1.text(m["prefill"] * 1.1, y, f"{m['prefill']:,.1f}", va="center", fontsize=8, color=INK)
        if m.get("mlx") and m["mlx"].get("prefill"):
            a1.barh(y + h, m["mlx"]["prefill"], height=h, color=C_MLX)
        if m.get("bf16") and m["bf16"].get("prefill"):
            a1.barh(y + 2 * h, m["bf16"]["prefill"], height=h, color=C_BF16)
        if m.get("ttft2k"):
            a2.barh(y, m["ttft2k"], height=h, color=C_SUC)
            a2.text(m["ttft2k"] * 1.1, y, f"{m['ttft2k']:,.1f} s", va="center", fontsize=8, color=INK)
            if m.get("mlx") and m["mlx"].get("ttft2k"):
                a2.barh(y + h, m["mlx"]["ttft2k"], height=h, color=C_MLX)
            if m.get("bf16") and m["bf16"].get("ttft2k"):
                a2.barh(y + 2 * h, m["bf16"]["ttft2k"], height=h, color=C_BF16)
        else:
            a2.text(0.12, y, "not measured at 2048", va="center", fontsize=8, color=INK)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(GB)
        ax.set_yticks([y + h for y in ys])
    a1.set_yticklabels(names, fontsize=9, color=INK)
    a2.set_yticklabels(names, fontsize=9, color=INK)
    a1.set_xlim(1, 20000)
    a2.set_xlim(0.1, 3000)
    a1.set_xlabel("prefill tok/s, batch 1 (log; ctx 128)")
    a2.set_xlabel("time to first token at ctx 2048, s (log)")
    handles = [Patch(color=C_SUC, label="SUC C++ (suc perf)"),
               Patch(facecolor="none", edgecolor=C_SUC, hatch="///", label="SUC C++, other protocol (see table)"),
               Patch(color=C_MLX, label="MLX affine 4-bit"), Patch(color=C_BF16, label="original BF16")]
    lg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8,
                    bbox_to_anchor=(0.5, -0.06))
    for t in lg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    save(fig, "compression_prefill_ttft.svg")


def table(models):
    def gb(x):
        return "—" if x is None else f"{x / 1e9:,.2f}"

    def f(x, nd=2, unit=""):
        return "—" if x is None else f"{x:,.{nd}f}{unit}"
    print("| Model | params | original GB (BF16-eq.) | on-disk GB | SUC1 GB | ratio | bpp | resident GB | "
          "decode ms/tok | prefill tok/s | TTFT 128 / 2048 s | PPL ref → SUC | KL | top-1 | eval |")
    print("|" + "---|" * 15)
    for m in models:
        ttft = "—" if m.get("ttft128") is None else f"{m['ttft128']:.2f} / {m['ttft2k']:.1f}"
        print(f"| {m['name']} | {m['params'] / 1e9:.1f} B | {gb(m['orig'])} | {gb(m.get('disk'))} | {gb(m['file'])} | "
              f"{m['orig'] / m['file']:.2f}x | {m['bpp']:.3f} | {gb(m['resident'])} | {f(m['decode'])} | "
              f"{f(m['prefill'], 1)} | {ttft} | {m['ref_ppl']:.3f} → {m['ppl']:.3f} | {m['kl']:.4f} | "
              f"{m['top1']:.3f} | {m['score'] or '—'} |")
    print()
    for m in models:
        print(m["name"], {k: v for k, v in m.items() if k in ("bf16", "mlx", "eval_harness", "decode_mean", "hit",
                                                              "decode_gpu", "ttft_short", "decode2k", "prefill2k")})
        print("   sources:", ", ".join(SOURCES[s] for s in m["src"]))


def main():
    global GG
    models = collect()
    GG = collect_gguf(models)
    fig_size(models)
    fig_decode(models)
    fig_quality(models)
    fig_prefill(models)
    table(models)
    print()
    gguf_table(GG)


GG: list = []

if __name__ == "__main__":
    main()
