"""C++/Metal model graph (cpp/ milestone 3, tool suc1-generate) vs the Python/MLX runtime of the same SUC1 file.

Not a test (see tests/unit/test_cpp_generate.py). Usage:
  .venv/bin/python benchmarks/cpp_generate_check.py [--file model.suc1] [--steps 64] [--build ~/suc-local/cpp-build]
         [--timing-tokens 64] [--repeats 3] [--no-default-runtime] [--out-dir experiments/results] [--name cpp_generate_check]

Per fixed prompt (Qwen3 tokenizer ids, fed token by token = the C++ prefill):
  1. C++ greedy generation of --steps tokens with the per-projection plan (--projections separate; logits and
     per-layer hidden states of every position).
  2. Python runtime with that plan (load_suc1_model(group=False): one SucLinear / SucResidualLinear per projection),
     teacher-forced on the C++ sequence: hidden states after the embedding and every block (fraction bit-identical,
     max abs difference), logits (fraction bit-identical, KL(python || C++) mean / max), top-1 agreement; then
     free-running Python greedy: identical-prefix length and agreement with the C++ tokens.
  3. The C++ default plan (--projections fused: row-concatenated q|k|v / gate|up launches, milestone 4) against the
     Python default runtime (convert_to_compressed defaults: grouped / fused projections, per-device element-wise
     fusion), the same comparisons (hidden states bit-identical fraction, logits KL, greedy agreement).
Perplexity (--ppl-windows N > 0): WikiText-2 test, N non-overlapping windows of --ppl-seq-len + 1 tokens (the
suc.evaluation.perplexity convention): C++ scores each window with a batch prefill (suc1-generate --nll-out) and Python
with a whole-window mlx-lm call of both runtimes; PPL and the mean |ΔNLL| per token are reported. Evaluation only
(the final-test split is never used for tuning).
Timing (batch 1, after the prompt): C++ ms/token (wall, and GPU time of the command buffer) of the default (fused
plan, pipelined decode: the next step is encoded while the GPU runs) and of the fused --sync / separate variants, vs
Python ms/token of both runtimes (model call + argmax .item() per token). Perplexity uses the C++ default plan. Laptop timings are indicative
only; benchmarks of record come from the Mac Studio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suc.provenance import run_metadata, write_result  # noqa: E402
from suc.runtime.suc1_model import greedy, kl_rows, load_suc1_model, make_cache, run_tokens  # noqa: E402

DEFAULT_FILE = Path.home() / "suc-local/compress/compress_qwen3_0.6b/container/model.suc1"
# Qwen3 tokenizer ids of fixed prompts (text in the comments)
PROMPTS = {
    # "The history of the printing press begins in the fifteenth century,"
    "history": [785, 3840, 315, 279, 18484, 3493, 12033, 304, 279, 11128, 45387, 9294, 11],
    # "def fibonacci(n):\n"
    "code": [750, 75698, 1445, 982],
    # "The history of France is"
    "short": [785, 3840, 315, 9625, 374],
}


def cpp_generate(build: Path, file: Path, prompt: list[int], steps: int, out: Path, *extra: str) -> dict:
    j = out / "run.json"
    cmd = [str(build / "suc1-generate"), str(file), "--tokens", ",".join(map(str, prompt)), "--steps", str(steps),
           "--logits-out", str(out / "logits.npy"), "--hidden-out", str(out / "hidden.npy"), "--json", str(j),
           "--prefill", "token", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    res = json.loads(j.read_text())
    res["stderr"] = r.stderr.strip()
    return res


def python_decode_ms(model, prompt: list[int], tokens: int, repeats: int) -> dict:
    """Median ms/token of synchronous greedy decode after a whole-prompt prefill (model call + argmax .item())."""
    per = []
    for _ in range(repeats + 1):            # first repeat = warm-up
        cache = make_cache(model)
        logits = model(mx.array([prompt]), cache=cache)
        nxt = int(mx.argmax(logits[0, -1]).item())
        times = []
        for _ in range(tokens):
            t0 = time.perf_counter()
            logits = model(mx.array([[nxt]]), cache=cache)
            nxt = int(mx.argmax(logits[0, -1]).item())
            times.append(time.perf_counter() - t0)
        per.append(1e3 * statistics.median(times))
    return {"ms_per_token_median": statistics.median(per[1:]), "repeats_ms": per[1:]}


def python_prefill_ms(model, prompt: list[int], repeats: int) -> float:
    """Median ms/token of a whole-prompt prefill (mlx-lm model call on the prompt, last-token argmax evaluated)."""
    per = []
    for _ in range(repeats + 1):
        cache = make_cache(model)
        t0 = time.perf_counter()
        logits = model(mx.array([prompt]), cache=cache)
        int(mx.argmax(logits[0, -1]).item())
        per.append(1e3 * (time.perf_counter() - t0) / len(prompt))
    return statistics.median(per[1:])


def cpp_run_json(build: Path, file: Path, prompt: list[int], steps: int, *extra: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([str(build / "suc1-generate"), str(file), "--tokens", ",".join(map(str, prompt)),
                            "--steps", str(steps), "--json", f"{d}/t.json", *extra], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
        return json.loads(Path(d, "t.json").read_text())


def ppl_check(args, sm, models: dict) -> dict:
    """C++ (batch-prefill NLL) vs Python (whole-window logits) perplexity on WikiText-2 test windows."""
    from transformers import AutoTokenizer

    from suc.evaluation.data import load_wikitext2, tokenize

    repo, rev = sm.tokenizer_id.split(";")[0].split("@")
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    text, wt_hash = load_wikitext2("test")
    ids = np.array(tokenize(tok, text))
    L = args.ppl_seq_len
    wins = ids[: args.ppl_windows * (L + 1)].reshape(args.ppl_windows, L + 1)
    cpp_nll, py_nll = [], {k: [] for k in models}
    for w in wins:
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([str(args.build / "suc1-generate"), str(args.file), "--tokens", ",".join(map(str, w)),
                                "--steps", "0", "--prefill", "batch", "--chunk", "128", "--nll-out", f"{d}/nll.npy",
                                "--quiet"], capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(r.stderr)
            cpp_nll.append(np.load(f"{d}/nll.npy").astype(np.float64))
        for k, m in models.items():
            logits = m(mx.array(w[None, :-1])).astype(mx.float32)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            nll = -mx.take_along_axis(logp[0], mx.array(w[1:, None]), axis=-1)[:, 0]
            py_nll[k].append(np.array(nll).astype(np.float64))
    c = np.concatenate(cpp_nll)
    out = {"windows": int(args.ppl_windows), "seq_len": L, "tokens_scored": int(c.size), "wikitext_hash": wt_hash,
           "cpp_ppl": float(np.exp(c.mean()))}
    for k, v in py_nll.items():
        p = np.concatenate(v)
        out[f"{k}_ppl"] = float(np.exp(p.mean()))
        out[f"{k}_mean_abs_dnll"] = float(np.abs(p - c).mean())
    if len(py_nll) == 2:   # noise floor: the two Python runtimes against each other
        a, b = (np.concatenate(v) for v in py_nll.values())
        out["python_runtimes_mean_abs_dnll"] = float(np.abs(a - b).mean())
    return out


def prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE)
    ap.add_argument("--build", type=Path, default=Path(os.environ.get("SUC_CPP_BUILD", Path.home() / "suc-local/cpp-build")))
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--timing-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prefill-tokens", type=int, default=256)
    ap.add_argument("--ppl-windows", type=int, default=4)
    ap.add_argument("--ppl-seq-len", type=int, default=512)
    ap.add_argument("--no-default-runtime", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None, help="write a result JSON (write_result) here")
    ap.add_argument("--name", default="cpp_generate_check", help="result name (file <timestamp>_<name>.json)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    plan_model, sm, _ = load_suc1_model(args.file, group=False)
    load_plan_s = time.perf_counter() - t0
    default_model = None
    if not args.no_default_runtime:
        default_model, _, _ = load_suc1_model(args.file, group=True, fuse=True)
    rows = {}
    for name, prompt in PROMPTS.items():
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            c = cpp_generate(args.build, args.file, prompt, args.steps, d, "--projections", "separate")
            cl, ch = np.load(d / "logits.npy"), np.load(d / "hidden.npy")
            if default_model is not None:   # the C++ default plan (fused projection groups)
                cf = cpp_generate(args.build, args.file, prompt, args.steps, d, "--projections", "fused")
                fl, fh = np.load(d / "logits.npy"), np.load(d / "hidden.npy")
        gen = c["generated"]
        seq = prompt + gen[:-1]
        pl, ph = run_tokens(plan_model, seq)
        same_h = (ph == ch).all(axis=-1)                       # (positions, layers + 1)
        kl = kl_rows(pl, cl)
        tf_agree = float(np.mean(pl.argmax(-1)[len(prompt) - 1:] == np.array(gen)))
        free = greedy(plan_model, prompt, args.steps)
        row = {"prompt_tokens": len(prompt), "generated": gen, "positions": len(seq),
               "hidden_bit_identical_fraction": float(same_h.mean()),
               "hidden_max_abs_diff": float(np.abs(ph - ch).max()),
               "logits_bit_identical_fraction": float(np.mean(pl == cl)),
               "logits_max_abs_diff": float(np.abs(pl - cl).max()),
               "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
               "teacher_forced_top1_agreement": tf_agree,
               "free_greedy_identical_prefix": prefix_len(free, gen),
               "free_greedy_agreement": float(np.mean(np.array(free) == np.array(gen))),
               "cpp": {k: c[k] for k in ("decode_ms_per_token_median", "decode_gpu_ms_per_token_median",
                                         "prefill_ms_per_token_median", "load_s", "compile_s", "memory")}}
        if default_model is not None:
            # C++ fused plan vs the default Python runtime, teacher-forced on the C++ fused sequence
            fgen = cf["generated"]
            fseq = prompt + fgen[:-1]
            dl, dh = run_tokens(default_model, fseq)
            dfree = greedy(default_model, prompt, args.steps)
            kd = kl_rows(dl, fl)
            row["default_runtime"] = {"cpp_generated": fgen,
                                      "hidden_bit_identical_fraction": float((dh == fh).all(axis=-1).mean()),
                                      "hidden_max_abs_diff": float(np.abs(dh - fh).max()),
                                      "logits_bit_identical_fraction": float(np.mean(dl == fl)),
                                      "kl_mean": float(kd.mean()), "kl_max": float(kd.max()),
                                      "teacher_forced_top1_agreement":
                                          float(np.mean(dl.argmax(-1)[len(prompt) - 1:] == np.array(fgen))),
                                      "free_greedy_identical_prefix": prefix_len(dfree, fgen),
                                      "free_greedy_agreement": float(np.mean(np.array(dfree) == np.array(fgen)))}
        rows[name] = row
        print(f"{name}: hidden identical {row['hidden_bit_identical_fraction']:.4f}, logits identical "
              f"{row['logits_bit_identical_fraction']:.5f}, KL mean {row['kl_mean']:.2e} max {row['kl_max']:.2e}, "
              f"top-1 (TF) {tf_agree:.3f}, free greedy prefix {row['free_greedy_identical_prefix']}/{args.steps}"
              + (f"; C++ fused vs default runtime: hidden identical "
                 f"{row['default_runtime']['hidden_bit_identical_fraction']:.4f}, KL mean "
                 f"{row['default_runtime']['kl_mean']:.2e}, free greedy prefix "
                 f"{row['default_runtime']['free_greedy_identical_prefix']}" if default_model is not None else ""))

    # timing: C++ (a separate run with --timing-tokens decode steps) vs both Python runtimes
    prompt = PROMPTS["history"]
    ct = cpp_run_json(args.build, args.file, prompt, args.timing_tokens + 1)   # default: fused, pipelined decode
    variants = {"fused_sync": ("--sync",), "separate_pipelined": ("--projections", "separate"),
                "separate_sync": ("--projections", "separate", "--sync")}
    cv = {k: cpp_run_json(args.build, args.file, prompt, args.timing_tokens + 1, *a) for k, a in variants.items()}
    timing = {"cpp_ms_per_token": ct["decode_ms_per_token_median"], "cpp_gpu_ms_per_token": ct["decode_gpu_ms_per_token_median"],
              "cpp_encode_ms_per_token": ct["decode_encode_ms_per_token_median"],
              "cpp_variants": {k: {m: v[m] for m in ("decode_ms_per_token_median", "decode_gpu_ms_per_token_median",
                                                     "decode_encode_ms_per_token_median")} for k, v in cv.items()},
              "cpp_memory": ct["memory"], "cpp_device": ct["device"],
              "python_plan_runtime": python_decode_ms(plan_model, prompt, args.timing_tokens, args.repeats)}
    if default_model is not None:
        timing["python_default_runtime"] = python_decode_ms(default_model, prompt, args.timing_tokens, args.repeats)
    # prefill of a long prompt: C++ batch prefill (chunks of 64) vs a whole-prompt mlx-lm call
    long_prompt = (PROMPTS["history"] * ((args.prefill_tokens + 12) // 13))[: args.prefill_tokens]
    cps = [cpp_run_json(args.build, args.file, long_prompt, 1, "--prefill", "batch")["prefill_ms_per_token"]
           for _ in range(args.repeats)]
    timing["prefill"] = {"tokens": len(long_prompt), "cpp_batch_ms_per_token": statistics.median(cps),
                         "python_plan_runtime_ms_per_token": python_prefill_ms(plan_model, long_prompt, args.repeats)}
    if default_model is not None:
        timing["prefill"]["python_default_runtime_ms_per_token"] = python_prefill_ms(default_model, long_prompt,
                                                                                     args.repeats)
    print("C++ variants (ms/token wall / GPU):", {k: (round(v["decode_ms_per_token_median"], 2),
                                                       round(v["decode_gpu_ms_per_token_median"], 2))
                                                   for k, v in timing["cpp_variants"].items()})
    print(f"timing (B=1 decode, ms/token): C++ fused pipelined {timing['cpp_ms_per_token']:.2f} (GPU {timing['cpp_gpu_ms_per_token']:.2f}), "
          f"Python plan runtime {timing['python_plan_runtime']['ms_per_token_median']:.2f}"
          + (f", Python default runtime {timing['python_default_runtime']['ms_per_token_median']:.2f}"
             if default_model is not None else ""))
    print(f"prefill {timing['prefill']['tokens']} tokens (ms/token): C++ batch {timing['prefill']['cpp_batch_ms_per_token']:.2f}, "
          f"Python plan runtime {timing['prefill']['python_plan_runtime_ms_per_token']:.2f}"
          + (f", Python default runtime {timing['prefill']['python_default_runtime_ms_per_token']:.2f}"
             if default_model is not None else ""))

    ppl = None
    if args.ppl_windows > 0:
        models = {"python_plan_runtime": plan_model}
        if default_model is not None:
            models["python_default_runtime"] = default_model
        ppl = ppl_check(args, sm, models)
        print("perplexity:", json.dumps(ppl))

    md = sm.metadata
    tok_hash = hashlib.sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest()
    result = {"metadata": run_metadata(model=md.get("model", "?"), model_revision=md.get("model_revision", "?"),
                                       model_hash=md.get("model_hash", "?"), dataset_hash=tok_hash, seed=0,
                                       config_hash=hashlib.sha256(json.dumps(vars(args), default=str,
                                                                             sort_keys=True).encode()).hexdigest()),
              "file": str(args.file), "steps": args.steps, "load_python_plan_s": load_plan_s,
              "prompts": rows, "timing": timing, "perplexity": ppl}
    if args.out_dir:
        print("wrote", write_result(result, args.out_dir, args.name))
    else:
        print(json.dumps({k: v for k, v in result.items() if k != "prompts"}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
