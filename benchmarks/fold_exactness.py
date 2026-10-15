"""Exactness of residual-stream rotation folding (suc.models.rotation_folding) on a real checkpoint.

Not a test. Compares the logits of the original model and of the folded model on held-out validation sequences of
the Phase 2 corpus (never the final test set):
  float32  folded (float32 weights) vs original (float32)      → should be rounding only
  bf16     folded (folded in float32, stored BF16) vs original BF16, and both BF16 models vs the float32 original
           (bf16_folded_vs_float32_original next to the noise floor bf16_noise_floor = original BF16 vs float32)
KL is next-token KL(original || folded) in float64; also max |Δlogit| and top-1 agreement.

Usage: .venv/bin/python benchmarks/fold_exactness.py [--model Qwen/Qwen3-0.6B] [--revision <sha>] [--seqs 4]
       [--seq-len 256] [--v-heads] [--json out.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suc.models.adapters import get_adapter  # noqa: E402
from suc.models.rotation_folding import ResidualFold, fold_model  # noqa: E402
from suc.provenance import git_info, runtime_info  # noqa: E402
from suc.quant.run import default_calib_artifact  # noqa: E402

DEFAULT_REVISIONS = {"Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
                     "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218"}


def compare(a: mx.array, b: mx.array) -> dict:
    """KL / max |Δ| / top-1 over all (sequence, position) rows, float64, one sequence at a time (bounded memory)."""
    kls, d, agree = [], 0.0, []
    for i in range(a.shape[0]):
        la, lb = (np.array(x[i].astype(mx.float32)).astype(np.float64) for x in (a, b))
        d = max(d, float(np.abs(la - lb).max()))
        la, lb = (x - x.max(-1, keepdims=True) for x in (la, lb))
        la, lb = (x - np.log(np.exp(x).sum(-1, keepdims=True)) for x in (la, lb))
        kls.append((np.exp(la) * (la - lb)).sum(-1))
        agree.append(la.argmax(-1) == lb.argmax(-1))
    kl = np.concatenate(kls)
    return {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "max_abs_logit_diff": d,
            "top1_agreement": float(np.concatenate(agree).mean())}


def logits(model, toks: mx.array) -> mx.array:
    out = []
    for i in range(toks.shape[0]):
        y = model(toks[i:i + 1]).astype(mx.float32)
        mx.eval(y)
        out.append(y)
    return mx.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--v-heads", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    rev = args.revision or DEFAULT_REVISIONS.get(args.model)
    model_dir = Path(snapshot_download(args.model, revision=rev, allow_patterns=["*.json", "*.safetensors"]))
    adapter = get_adapter(json.loads((model_dir / "config.json").read_text()))
    toks = mx.load(str(default_calib_artifact(args.model)))["validation"][:args.seqs, :args.seq_len]
    fold = ResidualFold.for_adapter(adapter, seed=args.seed, v_heads=args.v_heads)
    res = {}
    for dtype in (mx.float32, mx.bfloat16):
        key = "float32" if dtype == mx.float32 else "bf16"
        ref, _ = load_model(model_dir)
        ref.set_dtype(dtype)
        l_ref = logits(ref, toks)
        if dtype == mx.bfloat16:
            res["bf16_noise_floor"] = compare(ref32, l_ref)
        else:
            ref32 = l_ref
        del ref
        model, _ = load_model(model_dir)
        model.set_dtype(mx.float32)
        fold_model(model, adapter, fold, dtype=None)
        model.set_dtype(dtype)
        l_fold = logits(model, toks)
        res[key] = compare(l_ref, l_fold)
        if dtype == mx.bfloat16:
            res["bf16_folded_vs_float32_original"] = compare(ref32, l_fold)
        del model
        mx.clear_cache()
        print(key, json.dumps(res[key]), flush=True)
    for k in ("bf16_noise_floor", "bf16_folded_vs_float32_original"):
        print(k, json.dumps(res[k]))
    out = {"benchmark": "fold_exactness", "model": args.model, "revision": rev, "seed": args.seed,
           "v_heads": args.v_heads, "tokens": int(toks.size), "git": git_info(), "runtime": runtime_info(),
           "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "results": res}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"# wrote {args.json}")


if __name__ == "__main__":
    main()
