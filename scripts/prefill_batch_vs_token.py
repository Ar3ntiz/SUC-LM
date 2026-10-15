#!/usr/bin/env python3
"""Batched vs token-by-token prefill of the C++ graphs (Qwen3 / Qwen3-MoE, and since 2026-10-10 deepseek_v4 / glm5_next /
qwen4_exp / qwen3_5, whose default batched prefill runs simdgroup-matrix GEMMs): the measured size of the difference
that the batched path accepts by design (it is not bit-identical; cpp/README.md).

    prefill_batch_vs_token.py --name N --suc1 FILE [--long-prompt 2048] [--steps 32] [--free-steps 256] [--build DIR]

On the shared long speed prompt (suc.evaluation.speed_prompts, first N ids, the file's own tokenizer):
  1. `--prefill token` with --logits-out (every prompt position, then `steps` greedy steps): reference logits R and
     the greedy tokens G;
  2. the default batched prefill with --logits-out and --force G (teacher-forced on the same tokens, so every position
     sees the same context): logits B. --logits-out makes the batched path compute the logits of every row
     (Logits::All), the production path only the last row;
  3. production runs (no per-position output, `free_steps` greedy tokens) of both modes with $SUC_PREFILL_DUMP: the
     last prompt token's logits as the default run computes them, and the free-running greedy continuations.
Recorded: per-position KL(R || B) (float64 log-softmax), top-1 agreement, max |logit difference| over the prompt
positions and over the teacher-forced continuation; the same for the last prompt token of the production runs; the
first step at which the free-running greedy tokens differ (None = identical). One result JSON
(scripts/frontier_measure.py metadata).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import frontier_measure as fm  # noqa: E402


def _run(binary: Path, suc1: str, prompt: Path, args: list[str], env: dict | None = None) -> dict:
    import os

    with tempfile.TemporaryDirectory() as d:
        j = Path(d) / "run.json"
        cmd = [str(binary), suc1, "--tokens-file", str(prompt), "--json", str(j), "--quiet", *args]
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})
        if r.returncode:
            raise SystemExit(f"suc1-generate failed: {r.stderr[-2000:]}")
        return json.loads(j.read_text())


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def _compare(ref: np.ndarray, got: np.ndarray, chunk: int = 64) -> dict:
    """Row-wise KL(ref || got), top-1 agreement, max |difference| of the raw logits."""
    kl, top1, dmax, same = [], [], 0.0, 0
    for i in range(0, ref.shape[0], chunk):
        a, b = np.asarray(ref[i:i + chunk]), np.asarray(got[i:i + chunk])
        la, lb = _log_softmax(a), _log_softmax(b)
        kl.append((np.exp(la) * (la - lb)).sum(-1))
        top1.append(a.argmax(-1) == b.argmax(-1))
        dmax = max(dmax, float(np.abs(a - b).max()))
        same += int((a == b).all(-1).sum())
    kl, top1 = np.concatenate(kl), np.concatenate(top1)
    return {"positions": int(kl.size), "kl_mean": float(kl.mean()), "kl_p95": float(np.quantile(kl, 0.95)),
            "kl_max": float(kl.max()), "top1_agreement": float(top1.mean()), "top1_mismatches": int((~top1).sum()),
            "max_abs_logit_diff": dmax, "bit_identical_rows": same}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--suc1", required=True)
    ap.add_argument("--long-prompt", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=32, help="teacher-forced continuation steps with logits")
    ap.add_argument("--free-steps", type=int, default=256, help="free-running greedy steps of the production runs")
    ap.add_argument("--build", default="~/suc-local/cpp-build")
    ap.add_argument("extra", nargs="*", help="further suc1-generate arguments (after --), e.g. --expert-budget 95G")
    a = ap.parse_args()
    build = Path(a.build).expanduser()
    binary = build / "suc1-generate"
    suc1 = str(Path(a.suc1).expanduser())
    ids = fm._long_prompt_ids(build, suc1, a.long_prompt)
    out: dict = {"prompt_tokens": len(ids), "steps": a.steps, "free_steps": a.free_steps}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        prompt = d / "prompt.txt"
        prompt.write_text(",".join(map(str, ids)))
        # 1-2: every position, teacher-forced on the token run's greedy tokens
        rt = _run(binary, suc1, prompt, ["--prefill", "token", "--steps", str(a.steps), "--logits-out", str(d / "t.npy"),
                                         *a.extra])
        g = rt["generated"]
        rb = _run(binary, suc1, prompt, ["--prefill", "batch", "--steps", str(len(g)), "--force", ",".join(map(str, g)),
                                         "--logits-out", str(d / "b.npy"), *a.extra])
        t = np.load(d / "t.npy", mmap_mode="r")
        b = np.load(d / "b.npy", mmap_mode="r")
        if t.shape != b.shape:
            raise SystemExit(f"logits shapes differ: {t.shape} vs {b.shape}")
        n = len(ids)
        out["prompt_positions"] = _compare(t[:n], b[:n])
        out["continuation_positions"] = _compare(t[n:], b[n:]) if t.shape[0] > n else None
        out["modes"] = {"token": {k: rt.get(k) for k in ("prefill", "chunk", "prefill_s")},
                        "batch": {k: rb.get(k) for k in ("prefill", "chunk", "prefill_s")}}
        del t, b
        # 3: production runs (last-row logits only), free-running greedy
        last = {}
        for mode, args in (("token", ["--prefill", "token"]), ("batch", [])):
            r = _run(binary, suc1, prompt, [*args, "--steps", str(a.free_steps), *a.extra],
                     env={"SUC_PREFILL_DUMP": str(d / mode)})
            last[mode] = (np.load(d / f"{mode}.logits.npy")[None, :], r)
        out["last_prompt_token"] = _compare(last["token"][0], last["batch"][0])
        out["last_prompt_token"]["sha256"] = {m: last[m][1].get("prefill_logits_sha256") for m in last}
        gt, gb = last["token"][1]["generated"], last["batch"][1]["generated"]
        out["free_running"] = {"first_divergence_step": next((i for i, (x, y) in enumerate(zip(gt, gb)) if x != y), None),
                               "tokens_equal": sum(x == y for x, y in zip(gt, gb)), "tokens": min(len(gt), len(gb))}
    prompt_hash = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    fm._write(a.name, a.suc1, prompt_hash, {"tool": "prefill_batch_vs_token.py", "file": a.suc1,
                                             "long_prompt": a.long_prompt, "steps": a.steps,
                                             "free_steps": a.free_steps, "extra": a.extra}, out)
    print(json.dumps(out, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
