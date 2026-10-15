"""Q6 dev check: block-wise fine-tuning of SUC encodings on Qwen3-0.6B (not a test).

Usage: .venv/bin/python benchmarks/q6_finetune_dev.py [--steps 150] [--lr 1e-4] [--only a,b,rtn] [--json out.json]

All eligible linears are encoded; the tied embedding / LM head stays BF16 here (the Q5 table also encoded it,
so model bpp is not directly comparable). Eval = the Q5 protocol: WikiText-2 test, first 32,768 tokens at
seq_len 2048 (PPL) and next-token KL vs BF16 on 16 held-out "validation" sequences of the Phase 2 corpus.
Calibration = the first `--sequences` rows of its "calibration" partition (disjoint from validation / test).
  a    additive M=3 K=256 G=8 rot + 0.5% int8 residual selected by E[x_j²]·R²
  b    vq K=256 G=4 rot + 0.5% int8 residual (same selection)
  rtn  MLX affine RTN 4-bit and 3-bit, g64 (reference points)
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import zlib
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model

from suc.analysis.activations import Recorder
from suc.analysis.sensitivity import set_weight
from suc.evaluation.data import load_wikitext2, tokenize
from suc.evaluation.reconstruct import (
    evaluate_quality, load_validation_tokens, model_size, reconstruct_model, reference_logits,
)
from suc.io.safetensors_store import SafetensorsStore
from suc.models.adapters import get_adapter
from suc.models.streaming import load_lazy
from suc.quant.codecs import AdditiveCodebook, AffineRTN, VectorCodebook
from suc.quant.finetune import FinetuneConfig, finetune_blockwise
from suc.quant.residual import ResidualCodec
from suc.quant.run import select_specs

MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
ART = Path("experiments/artifacts")
SEED = 12345


def newest(pattern: str) -> Path:
    found = sorted(ART.glob(pattern))
    if not found:
        raise FileNotFoundError(pattern)
    return found[-1]


def free(*objs) -> None:
    del objs
    gc.collect()
    mx.clear_cache()


def summary(q: dict) -> dict:
    return {"ppl": q["perplexity"]["ppl"], "kl": q["kl"]["kl_mean"], "top1": q["kl"]["top1_agreement"],
            "delta_ce": q["kl"]["delta_ce"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sequences", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--reassign", action="store_true")
    ap.add_argument("--only", default="rtn,a,b")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    only = set(args.only.split(","))

    store = SafetensorsStore.from_dir(MODEL_DIR)
    adapter = get_adapter(json.loads((MODEL_DIR / "config.json").read_text()))
    specs = select_specs(adapter, store.names(), None, include_embeddings=False)
    data = newest("*phase2_calib_qwen3_0.6b*_data.safetensors")
    calib = mx.load(str(data))["calibration"][:args.sequences]
    kl_tokens = load_validation_tokens(data, 16)
    ppl_tokens = tokenize(load_lazy(MODEL_DIR)[1], load_wikitext2("test")[0])
    ev = dict(ppl_tokens=ppl_tokens, seq_len=2048, max_tokens=32768, kl_tokens=kl_tokens)

    model, _ = load_model(MODEL_DIR)
    ref_q = evaluate_quality(model, ppl_tokens=ppl_tokens, seq_len=2048, max_tokens=32768)
    ref_logits = reference_logits(model, kl_tokens)
    free(model)
    out: dict = {"model": str(MODEL_DIR), "calib_artifact": str(data), "eval": "wikitext2 test 32768 tok @2048; "
                 "KL 16 validation seqs", "embedding": "bf16 (not encoded)", "bf16_ppl": ref_q["perplexity"]["ppl"],
                 "finetune": {"steps": args.steps, "lr": args.lr, "sequences": args.sequences,
                              "micro_batch": args.micro_batch, "reassign": args.reassign}, "results": {}}
    print(f"BF16 PPL {out['bf16_ppl']:.3f}", flush=True)

    if "rtn" in only:
        for bits in (4, 3):
            model, _ = load_model(MODEL_DIR)
            codec = AffineRTN(bits, 64)
            recon = reconstruct_model(model, store, specs, lambda _s: codec, seed=SEED)
            q = evaluate_quality(model, ref_logits=ref_logits, **ev)
            r = {"bpp": model_size(store, recon)["bpp"], **summary(q)}
            out["results"][f"rtn{bits}"] = r
            print(f"RTN {bits}-bit: {r}", flush=True)
            free(model, recon)

    rec = Recorder.load(newest("*phase2_calib_qwen3_0.6b_activations.safetensors"))
    m2 = {k: m.second_moment_diag() for k, m in rec.moments.items()}
    bases = {"a": ("additive M=3 K=256 G=8 rot + 0.5% res", AdditiveCodebook(256, 8, num_codebooks=3, rotate=True)),
             "b": ("vq K=256 G=4 rot + 0.5% res", VectorCodebook(256, block_size=4, rotate=True))}
    for key, (label, base) in bases.items():
        if key not in only:
            continue
        res = ResidualCodec(base, 0.005, "int8", "importance")
        model, _ = load_model(MODEL_DIR)
        t0 = time.perf_counter()
        enc = {}
        for s in specs:
            w = store.load(s.name)
            imp = mx.broadcast_to(m2[s.name[:-7]][None], w.shape)
            e = res.encode(w, seed=SEED + zlib.crc32(s.name.encode()), importance=imp)
            set_weight(model, s.name, res.decode(e))
            mx.eval(e.arrays)
            enc[s.name] = e
            del w
        mx.clear_cache()
        t_enc = time.perf_counter() - t0
        size = model_size(store, {"tensors": [{"name": n, "bits": e.bits()} for n, e in enc.items()]})
        before = summary(evaluate_quality(model, ref_logits=ref_logits, **ev))
        print(f"{label}: bpp {size['bpp']:.3f} (encode {t_enc:.0f} s) before {before}", flush=True)
        cfg = FinetuneConfig(block_steps=args.steps, lr=args.lr, micro_batch=args.micro_batch,
                             sequences=args.sequences, reassign=args.reassign, seed=SEED)
        tuned, log = finetune_blockwise(model, enc, calib, cfg, reference_weights=store.load)
        after = summary(evaluate_quality(model, ref_logits=ref_logits, **ev))
        size_after = model_size(store, {"tensors": [{"name": n, "bits": e.bits()} for n, e in tuned.items()]})
        assert size_after["bits_total"] == size["bits_total"]
        blocks = {i: {k: b[k] for k in ("loss_before", "loss_after", "reverted", "seconds")} for i, b in log["blocks"].items()}
        out["results"][key] = {"label": label, "bpp": size["bpp"], "encoded_bpp": size["encoded_bpp"],
                               "encode_seconds": t_enc, "before": before, "after": after,
                               "finetune_seconds": log["seconds"], "finetune_peak_memory_bytes": log["peak_memory_bytes"],
                               "blocks": blocks}
        print(f"{label}: after {after}  finetune {log['seconds']:.0f} s, peak {log['peak_memory_bytes'] / 2**30:.2f} GiB",
              flush=True)
        for i in sorted(blocks)[:3] + sorted(blocks)[-2:]:
            b = blocks[i]
            print(f"  block {i:2d}: {b['loss_before']:.4f} -> {b['loss_after']:.4f} ({b['seconds']:.1f} s)", flush=True)
        free(model, enc, tuned)

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out["results"], indent=1, default=float))


if __name__ == "__main__":
    main()
