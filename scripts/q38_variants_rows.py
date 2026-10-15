#!/usr/bin/env python3
"""Insert the Qwen3.8-Flash-Next iso / side8 rows into docs/compression_report.md §5 (after the base SUC1 row).

Reads: the compress evaluate records (eval.json) of the two variant run dirs, the scripts/system_quality.py result of
the iso file with 4-bit n-gram tables, and the two `frontier_measure.py generate` results. Columns as the §5 table;
bpp = file bits / 176.94 B parameters (§1). Idempotent: rows already present (same label) are not inserted again.
  python scripts/q38_variants_rows.py --sq SQ.json --gen-iso G1.json --gen-side8 G2.json
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PARAMS = 176943899555
ART = REPO / "experiments/artifacts/compress"
ANCHOR = "| | SUC1 (ngram4), resident | 93.49 |"


def kl_cells(k: dict) -> str:
    return f"{k['kl_mean']:.4f} | {k['top1_agreement']:.3f} | {k['delta_ce']:.4f}"


def speed_cells(g: dict | None) -> str:
    if g is None or g["summary"].get("returncode") != 0:
        return "— | — | —"
    r = g["run"]
    ttft = r["prefill_s"] + r["decode_ms_per_token"][0] / 1000
    return f"{r['decode_ms_per_token_median']:.2f} | {r['prompt_tokens'] / r['prefill_s']:.1f} | {ttft:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sq", required=True)
    ap.add_argument("--gen-iso")
    ap.add_argument("--gen-side8")
    a = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text()) if p and Path(p).exists() else None  # noqa: E731
    iso_dir, s8_dir = ART / "compress_qwen3.8_flash_next_iso", ART / "compress_qwen3.8_flash_next_side8"
    iso_raw = load(iso_dir / "evaluate/eval.json")["quality"]["kl"]
    sq = load(a.sq)["systems"]["suc_python_iso_ngram4"]["kl"]
    s8 = load(s8_dir / "evaluate/eval.json")["quality"]["kl"]
    f_raw = (iso_dir / "container/model.suc1").stat().st_size
    f_iso4 = (iso_dir / "container/model_ngram4.suc1").stat().st_size
    f_s8 = (s8_dir / "container/model.suc1").stat().st_size
    row = lambda label, f, k, g: (f"| | {label} | {f / 1e9:.2f} | {8 * f / PARAMS:.3f} | {kl_cells(k)} | "  # noqa: E731
                                  f"{speed_cells(g)} | — |")
    rows = [row("SUC1 iso, raw BF16 tables ¹¹", f_raw, iso_raw, None),
            row("SUC1 iso (ngram4), resident ¹¹", f_iso4, sq, load(a.gen_iso)),
            row("SUC1 side8 (ngram4), resident ¹¹", f_s8, s8, load(a.gen_side8))]
    doc = REPO / "docs/compression_report.md"
    lines = doc.read_text().split("\n")
    i = next(j for j, l in enumerate(lines) if l.startswith(ANCHOR))
    new = [r for r in rows if not any(l.startswith(r.split(" | ")[0] + " | " + r.split(" | ")[1]) for l in lines)]
    lines[i + 1:i + 1] = new
    doc.write_text("\n".join(lines))
    print("\n".join(new))


if __name__ == "__main__":
    main()
