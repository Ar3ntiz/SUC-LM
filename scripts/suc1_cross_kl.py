#!/usr/bin/env python3
"""KL / top-1 / delta-CE of a SUC1 file through the C++ runtime on the 16 x 512 protocol of ANY `suc compress` run
directory (its held-out validation rows calibrate/data.safetensors and its BF16 reference logits
reference/ref_logits.safetensors), with exactly the arithmetic of the GGUF rows (scripts/gguf_compare.kl_vs_reference:
suc's logit_divergence, test logits rounded to bfloat16 as the evaluate stage does).

Why: the validation partition depends on the calibration mix, so a run calibrated with mix v2 reports its evaluate-stage
KL on other sequences than the v1 runs (and the ds4 GGUF rows, scored on the v1 base run). Scoring the files on one
run directory's protocol makes them comparable; scoring a file on its own run directory checks the C++ path against
its evaluate stage (Python, reconstructed weights).

Per validation row: `suc1-generate --tokens-file <first token> --force <the other 511> --logits-out` (decode path,
teacher-forced, one process per row; the file stays in the page cache), i.e. suc.evaluation.runners.CppRunner.score.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/suc1_cross_kl.py --name NAME --suc1 FILE --run-dir RUN_DIR \\
        [--build ~/suc-local/cpp-build] [--sequences 16]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import frontier_measure as fm  # noqa: E402
from gguf_compare import kl_vs_reference  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="result name (experiments/results/<ts>_<name>.json)")
    ap.add_argument("--suc1", required=True)
    ap.add_argument("--run-dir", required=True, help="suc compress run directory whose protocol is used")
    ap.add_argument("--build", default="~/suc-local/cpp-build")
    ap.add_argument("--sequences", type=int, default=16)
    a = ap.parse_args()

    import mlx.core as mx

    from suc.evaluation.runners import CppRunner

    run = Path(a.run_dir).expanduser()
    val = mx.load(str(run / "calibrate" / "data.safetensors"))["validation"][: a.sequences]
    tok = np.asarray(val).astype("<i4")
    n, L = tok.shape
    runner = CppRunner("suc1", a.suc1, a.build)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as d:
        tf, lf = Path(d, "tokens.i32"), Path(d, "logits.bf16")
        tok.tofile(tf)
        vocab = None
        with open(lf, "wb") as out:
            for i in range(n):
                lg = runner.score([int(tok[i, 0])], [int(x) for x in tok[i, 1:]])   # (L-1, vocab) float32
                if lg.shape[0] != L - 1:
                    raise SystemExit(f"row {i}: {lg.shape[0]} logit rows, expected {L - 1}")
                vocab = lg.shape[1]
                out.write(np.asarray(mx.array(lg).astype(mx.bfloat16).view(mx.uint16)).tobytes())
                print(f"row {i + 1}/{n}: {time.perf_counter() - t0:.0f} s", flush=True)
        kl = kl_vs_reference(run, tf, lf, n, L, vocab)
    sec = time.perf_counter() - t0
    print(f"KL {kl['kl_mean']:.4f}, top-1 {kl['top1_agreement']:.4f}, dCE {kl['delta_ce']:.4f} ({sec:.0f} s)")
    ref_json = run / "reference" / "reference.json"
    fm._write(a.name, a.suc1, hashlib.sha256(tok.tobytes()).hexdigest(),
              {"tool": "suc1-generate (teacher-forced decode, --logits-out)", "file": a.suc1, "run_dir": str(run),
               "sequences": n, "seq_len": L, "build": a.build},
              {"kl": kl, "seconds": sec, "protocol": {
                  "run_dir": str(run),
                  "calibrate_sha256": hashlib.sha256((run / "calibrate" / "data.safetensors").read_bytes()).hexdigest(),
                  "reference_json_sha256": hashlib.sha256(ref_json.read_bytes()).hexdigest() if ref_json.exists() else None}})


if __name__ == "__main__":
    main()
