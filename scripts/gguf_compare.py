#!/usr/bin/env python3
"""Community GGUF quantizations vs the SUC1 files, measured with the ds4 engine on the Mac Studio
(docs/compression_report.md, "Community GGUF comparison"; queue: scripts/studio_gguf_compare.sh).

Every GGUF is scored on exactly the token ids of the `suc compress` evaluate stage of the same model:

  - PPL: WikiText-2 test, tokenized with the HF tokenizer as suc compress does (sha256 checked against the compress run's
    reference.json), cut into the same non-overlapping windows of eval.seq_len + 1 tokens, every window from an empty
    context; NLL of every position from the full next-token logits (scripts/gguf/ds4_score.c, decode path).
  - KL / top-1 / delta-CE: the 16 x 512 held-out validation sequences of the compress run (calibrate/data.safetensors)
    against the BF16 reference logits stored by its reference stage (reference/ref_logits.safetensors), with suc's own
    logit_divergence, test logits rounded to bfloat16 exactly as the evaluate stage does. This is the SAME KL as the
    SUC rows of the report (same reference logits, tokens and arithmetic), not an approximation.
  - speed: ds4-bench (ctx 128 and 2048 from an empty context, prompt experiments/perf/promessi_sposi_head.txt, 256
    greedy tokens, 3 repeats) and the frontier short-prompt protocol of scripts/frontier_measure.py generate (the same
    prompt sentence, 256 greedy tokens) through ds4_score serve.
  - eval: the suc eval core suite (experiments/eval/suite_core.json) with suc eval's own prompt rendering, thinking
    budget rule and graders (suc.cli.eval.run_eval with an injected runner on ds4_score serve).

  gguf_compare.py prepare KEY        token files + GGUF tensor inventory (bytes / params, bpw) -> logs/gguf/KEY/
  gguf_compare.py quality KEY        PPL / KL result JSON from the ds4_score outputs (partial PPL allowed, flagged)
  gguf_compare.py perf KEY           ds4-bench + short-prompt runs -> result JSON
  gguf_compare.py eval KEY [--limit N] [--suite F --ctx N]  suc eval suite (default core) on the GGUF (kind ds4)
  gguf_compare.py rows-cmd KEY WHAT  print the ds4_score command line for WHAT in {kl, kl_b, ppl} (used by the queue)
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shlex
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

DS4 = Path(os.environ.get("DS4_DIR", "~/Projects/ds4")).expanduser()
GGUF_DIR = Path(os.environ.get("GGUF_DIR", "~/Projects/ds4/GGUF")).expanduser()
SCORE_BIN = DS4 / "ds4_score_suc"
WORK = REPO / "logs" / "gguf"
PERF_PROMPT = REPO / "experiments" / "perf" / "promessi_sposi_head.txt"
from suc.evaluation.speed_prompts import SHORT_PROMPT  # noqa: E402  (frontier_measure.py generate)
Q38_RUN = Path("~/suc-queue/frontier_models/runs/Qwen__Qwen3.8-Flash-Next").expanduser()

MODELS = {
    "ds_iq2": dict(
        label="DeepSeek-V4-Flash 0731 IQ2_XXS mix (ds4 imatrix)",
        gguf="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8"
             "-chat-v2-imatrix-fixed-0731.gguf",
        compress="experiments/configs/compress_deepseek_v4_flash.yaml",
        run_dir=REPO / "experiments/artifacts/compress/compress_deepseek_v4_flash",
        eval_cfg="experiments/configs/eval_deepseek_v4_flash.yaml", ds4_args=[], batch=4, ppl_rows=8),
    "ds_mxfp4": dict(
        label="DeepSeek-V4-Flash 0731 MXFP4 experts (ds4, SSD streaming)",
        gguf="DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf",
        compress="experiments/configs/compress_deepseek_v4_flash.yaml",
        run_dir=REPO / "experiments/artifacts/compress/compress_deepseek_v4_flash",
        eval_cfg="experiments/configs/eval_deepseek_v4_flash.yaml",
        # explicit expert-cache target: never let the 156 GB file approach the 116 GB GPU wired limit
        ds4_args=["--ssd-streaming", "--ssd-streaming-cache-experts", "80GB"], batch=1,
        ppl_rows=8),
    "glm_q2": dict(
        label="GLM-5.3-Flash Q2 (ds4)",
        gguf="GLM-5.3-Flash-Q2.gguf",
        compress="experiments/configs/compress_glm5.3_flash.yaml",
        run_dir=REPO / "experiments/artifacts/compress/compress_glm5.3_flash",
        eval_cfg="experiments/configs/eval_glm5.3_flash_cpp.yaml",
        # SSD expert streaming (as the SUC1 GLM file), cache target 70 GB: no native session batching when streaming
        ds4_args=["--ssd-streaming", "--ssd-streaming-cache-experts", "70GB"], batch=1, ppl_rows=8),
    "q38_q4": dict(
        label="Qwen3.8-Flash-Next Q4 (ds4 qwen38-q4k: Q4_K gate/up, MXFP4 down, BF16 n-grams)",
        gguf="Qwen3.8-Flash-Next-Q4.gguf",
        compress="experiments/configs/compress_qwen3.8_flash_next.yaml",
        run_dir=Q38_RUN,
        eval_cfg="experiments/configs/eval_qwen3.8_flash_next.yaml", ds4_args=[], batch=1, ppl_rows=8),
}


def _m(key: str) -> dict:
    return MODELS[key]


def _work(key: str) -> Path:
    d = WORK / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _gguf(key: str) -> Path:
    return GGUF_DIR / _m(key)["gguf"]


# ------------------------------------------------------------------------------------------------ GGUF inventory

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_inventory(path: Path) -> dict:
    """Header metadata (scalars / strings, arrays summarized) and per-tensor params / bytes of a GGUF file; tensor bytes
    are the distance between consecutive data offsets (the last one runs to the end of the file), so no quant-type size
    table is needed. Categories: mtp (blk index >= block_count, 'mtp' / 'nextn' in the name), ngram ('ngram' / 'engram',
    Qwen3.8's per_layer_token_embd n-gram table), model (everything else)."""
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def rstr():
            return f.read(rd("<Q")).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                at, n = rd("<I"), rd("<Q")
                items = [rval(at) for _ in range(n)]
                return {"array_type": at, "len": n} if n > 16 else items
            return rd(_SCALAR[t])

        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version, n_t, n_kv = rd("<I"), rd("<Q"), rd("<Q")
        kv = {}
        for _ in range(n_kv):
            k = rstr()
            kv[k] = rval(rd("<I"))
        tensors = []
        for _ in range(n_t):
            name = rstr()
            dims = [rd("<Q") for _ in range(rd("<I"))]
            ttype, off = rd("<I"), rd("<Q")
            tensors.append([name, dims, ttype, off])
        align = int(kv.get("general.alignment", 32))
        start = (f.tell() + align - 1) // align * align
    size = path.stat().st_size
    order = sorted(range(len(tensors)), key=lambda i: tensors[i][3])
    for j, i in enumerate(order):
        end = tensors[order[j + 1]][3] if j + 1 < len(order) else size - start
        tensors[i].append(end - tensors[i][3])
    arch = kv.get("general.architecture", "")
    n_block = kv.get(f"{arch}.block_count")
    cats: dict[str, dict] = {}
    types: dict[int, int] = {}
    for name, dims, ttype, _, nbytes in tensors:
        n = 1
        for d in dims:
            n *= d
        parts = name.split(".")
        blk = int(parts[1]) if len(parts) > 1 and parts[0] == "blk" and parts[1].isdigit() else None
        cat = ("mtp" if ("mtp" in name or "nextn" in name
                         or (blk is not None and n_block is not None and blk >= n_block))
               else "ngram" if ("ngram" in name or "engram" in name or name.startswith("per_layer_token_embd"))
               else "model")
        c = cats.setdefault(cat, {"tensors": 0, "params": 0, "bytes": 0})
        c["tensors"] += 1
        c["params"] += n
        c["bytes"] += nbytes
        types[ttype] = types.get(ttype, 0) + nbytes
    for c in cats.values():
        c["bpw"] = c["bytes"] * 8 / c["params"] if c["params"] else None
    head = hashlib.sha256()
    with open(path, "rb") as f:
        head.update(f.read(64 << 20))
    meta = {k: v for k, v in kv.items() if not k.startswith("tokenizer.") or isinstance(v, (int, float, str))}
    return {"path": str(path), "file_bytes": size, "gguf_version": version, "tensor_count": n_t,
            "architecture": arch, "block_count": n_block, "data_offset": start, "categories": cats,
            "bytes_by_ggml_type": {str(k): v for k, v in sorted(types.items())},
            "head64m_sha256": head.hexdigest(), "mtime": path.stat().st_mtime,
            "fingerprint": f"gguf:{size}:{head.hexdigest()[:32]}", "metadata": meta}


# ------------------------------------------------------------------------------------------------ prepare

def _tokenizer_and_cfg(key: str):
    from mlx_lm.utils import load_tokenizer

    from suc.cli.bench import resolve_model_dir
    from suc.config import load_compress_config

    cfg = load_compress_config(REPO / _m(key)["compress"])
    model_dir = resolve_model_dir(cfg)
    hf = json.loads((model_dir / "config.json").read_text())
    return cfg, model_dir, load_tokenizer(model_dir, eos_token_ids=hf.get("eos_token_id"))


def cmd_prepare(key: str) -> None:
    import mlx.core as mx
    import numpy as np

    from suc.evaluation.data import load_wikitext2, tokenize
    from suc.evaluation.streamed import RefLogits, ppl_windows

    m, w = _m(key), _work(key)
    cfg, model_dir, tok = _tokenizer_and_cfg(key)
    ref = json.loads((m["run_dir"] / "reference" / "reference.json").read_text())
    text, wiki_hash = load_wikitext2(cfg.eval.wikitext_split)
    toks = tokenize(tok, text)
    sha = hashlib.sha256(np.asarray(toks, dtype=np.int32).astype("<i4").tobytes()).hexdigest()
    if sha != ref["ppl_tokens_sha256"]:
        raise SystemExit(f"{key}: WikiText-2 token sha {sha} != reference {ref['ppl_tokens_sha256']}")
    seq_len = ref["quality"]["perplexity"]["seq_len"]
    win = np.asarray(ppl_windows(toks, seq_len, cfg.eval.max_tokens), dtype="<i4")
    win.tofile(w / "ppl_tokens.i32")
    kl = np.asarray(mx.load(str(m["run_dir"] / "calibrate" / "data.safetensors"))["validation"][: ref["kl_sequences"]],
                    dtype="<i4")
    kl.tofile(w / "kl_tokens.i32")
    rl = RefLogits(m["run_dir"] / "reference" / "ref_logits.safetensors")
    if tuple(rl.shape[:2]) != kl.shape:
        raise SystemExit(f"{key}: reference logits {rl.shape} do not match the KL tokens {kl.shape}")
    sample = text[:6000]
    (w / "vocab_sample.txt").write_text(sample)
    inv = gguf_inventory(_gguf(key))
    meta = {"key": key, "label": m["label"], "model": cfg.model.id, "revision": cfg.model.revision,
            "model_dir": str(model_dir), "run_dir": str(m["run_dir"]), "wikitext_hash": wiki_hash,
            "ppl_tokens_sha256": sha, "ppl_windows": int(win.shape[0]), "seq_len": int(seq_len),
            "kl_shape": list(kl.shape), "kl_tokens_sha256": hashlib.sha256(kl.tobytes()).hexdigest(),
            "ref_logits_shape": list(rl.shape), "ref_ppl": ref["quality"]["perplexity"],
            "vocab_sample_hf_ids": list(tok.encode(sample, add_special_tokens=False)),
            "short_prompt_ids": list(tok.encode(SHORT_PROMPT, add_special_tokens=False)),
            "gguf": inv}
    (w / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{key}: {win.shape[0]} PPL windows x {seq_len}, KL {kl.shape}, GGUF {inv['file_bytes'] / 1e9:.2f} GB, "
          f"categories { {k: (round(v['bytes'] / 1e9, 2), round(v['bpw'] or 0, 3)) for k, v in inv['categories'].items()} }")


def cmd_rows_cmd(key: str, what: str) -> None:
    m, w = _m(key), _work(key)
    meta = json.loads((w / "meta.json").read_text())
    if what == "ppl":
        n, L = meta["ppl_windows"], meta["seq_len"] + 1
        args = ["rows", str(_gguf(key)), str(w / "ppl_tokens.i32"), str(n), str(L), str(w / "ppl_rows.jsonl"),
                "--batch", str(m["batch"])] + (["--max-rows", str(m["ppl_rows"])] if m["ppl_rows"] else [])
    else:
        b = 1 if what == "kl" else m["batch"]
        n, L = meta["kl_shape"]
        args = ["rows", str(_gguf(key)), str(w / "kl_tokens.i32"), str(n), str(L), str(w / f"{what}_rows.jsonl"),
                "--batch", str(b), "--logits-out", str(w / f"{what}_logits.bf16")]
    print(shlex.join([str(SCORE_BIN)] + args + list(m["ds4_args"])))


# ------------------------------------------------------------------------------------------------ results

def _ds4_commit() -> str:
    try:
        r = subprocess.run(["git", "-C", str(DS4), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write(key: str, kind: str, body: dict, dataset_hash: str) -> Path:
    from suc.provenance import run_metadata, write_result

    meta = json.loads((_work(key) / "meta.json").read_text())
    config = {"key": key, "kind": kind, "gguf": meta["gguf"]["path"], "ds4_dir": str(DS4), "ds4_commit": _ds4_commit(),
              "ds4_args": _m(key)["ds4_args"], "batch": _m(key)["batch"], "ppl_rows_cap": _m(key)["ppl_rows"],
              "compress_config": _m(key)["compress"], "run_dir": meta["run_dir"]}
    ch = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    md = run_metadata(model=meta["model"], model_revision=meta["revision"], model_hash=meta["gguf"]["fingerprint"],
                      dataset_hash=dataset_hash, seed=0, config_hash=ch, repo_dir=REPO)
    res = {"metadata": md, "kind": kind, "config": config, "gguf": {k: v for k, v in meta["gguf"].items()
                                                                     if k != "metadata"},
           "gguf_metadata": meta["gguf"]["metadata"], "label": meta["label"], **body}
    path = write_result(res, REPO / "experiments" / "results", f"{kind}_{key}")
    print(path)
    return path


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def kl_vs_reference(run_dir: Path, tokens_i32: Path, logits_bf16: Path, n: int, L: int, vg: int) -> dict:
    """KL / top-1 / CE of test logits (n, L-1, vg) bf16 on the KL token rows (n, L) of a `suc compress` run against
    its stored BF16 reference logits, with suc's own logit_divergence (the evaluate stage's arithmetic)."""
    import mlx.core as mx
    import numpy as np

    from suc.analysis.sensitivity import logit_divergence
    from suc.evaluation.streamed import RefLogits

    test = np.memmap(logits_bf16, dtype=np.uint16, mode="r", shape=(n, L - 1, vg))
    tok = mx.array(np.fromfile(tokens_i32, dtype="<i4").reshape(n, L))
    ref = RefLogits(Path(run_dir) / "reference" / "ref_logits.safetensors")
    vr = ref.shape[2]
    keys = ("kl_mean", "kl_p95", "top1_agreement", "ce_ref", "ce_test")
    acc, ntok = dict.fromkeys(keys, 0.0), 0
    for s in range(0, n, 2):
        e = min(s + 2, n)
        r = ref.rows(s, e)[:, :-1]
        t = mx.array(np.ascontiguousarray(test[s:e])).view(mx.bfloat16)
        if vg < vr:      # GGUF vocabulary shorter than the checkpoint's padded LM head: absent ids get -inf-like logits
            t = mx.concatenate([t, mx.full((*t.shape[:2], vr - vg), -1e30, dtype=mx.bfloat16)], axis=-1)
        elif vg > vr:
            t = t[..., :vr]
        d = logit_divergence(r, t.astype(r.dtype), tok[s:e, 1:])
        k = (e - s) * (L - 1)
        for x in keys:
            acc[x] += d[x] * k
        ntok += k
    out = {x: v / ntok for x, v in acc.items()}
    out.update(delta_ce=out["ce_test"] - out["ce_ref"], tokens=ntok, vocab_gguf=vg, vocab_ref=vr,
               kl_p95_note="mean of per-chunk p95 (2 sequences per chunk), as the evaluate stage")
    return out


def _kl_of(key: str, what: str) -> dict | None:
    w, m = _work(key), _m(key)
    meta = json.loads((w / "meta.json").read_text())
    rows = _rows(w / f"{what}_rows.jsonl")
    n, L = meta["kl_shape"]
    if len(rows) < n:
        return None
    return kl_vs_reference(m["run_dir"], w / "kl_tokens.i32", w / f"{what}_logits.bf16", n, L, rows[0]["vocab"])


def _batch_consistency(key: str) -> dict | None:
    """KL between the batch-1 and the batched decode of the same KL sequences (ds4 native session batching)."""
    import mlx.core as mx
    import numpy as np

    w = _work(key)
    a, b = _rows(w / "kl_rows.jsonl"), _rows(w / "kl_b_rows.jsonl")
    meta = json.loads((w / "meta.json").read_text())
    n, L = meta["kl_shape"]
    if len(a) < n or len(b) < n:
        return None
    v = a[0]["vocab"]
    x = np.memmap(w / "kl_logits.bf16", dtype=np.uint16, mode="r", shape=(n, L - 1, v))
    y = np.memmap(w / "kl_b_logits.bf16", dtype=np.uint16, mode="r", shape=(n, L - 1, v))
    kls, agree, maxabs = [], [], 0.0
    for i in range(n):
        p = mx.array(np.ascontiguousarray(x[i])).view(mx.bfloat16).astype(mx.float32)
        q = mx.array(np.ascontiguousarray(y[i])).view(mx.bfloat16).astype(mx.float32)
        lp, lq = p - mx.logsumexp(p, -1, keepdims=True), q - mx.logsumexp(q, -1, keepdims=True)
        kls.append(float((mx.exp(lp) * (lp - lq)).sum(-1).mean()))
        agree.append(float((p.argmax(-1) == q.argmax(-1)).mean()))
        maxabs = max(maxabs, float(mx.abs(p - q).max()))
    nll_a = sum(r["nll_sum"] for r in a) / sum(r["tokens"] for r in a)
    nll_b = sum(r["nll_sum"] for r in b) / sum(r["tokens"] for r in b)
    return {"batch": b[0]["batch"], "kl_batch1_vs_batched": statistics.mean(kls), "top1_agreement": statistics.mean(agree),
            "max_abs_logit_diff": maxabs, "nll_batch1": nll_a, "nll_batched": nll_b}


def cmd_quality(key: str) -> None:
    import math

    w, m = _work(key), _m(key)
    meta = json.loads((w / "meta.json").read_text())
    ppl_rows = _rows(w / "ppl_rows.jsonl")
    kl = _kl_of(key, "kl")
    body: dict = {"protocol": {
        "ppl": f"WikiText-2 {meta['ppl_windows']} non-overlapping windows of {meta['seq_len']} + 1 tokens (suc compress "
               "evaluate windows, same token ids), each from an empty context, decoded token by token by ds4; NLL from "
               "the full fp32 logits (log-sum-exp in double)",
        "kl": "16 x 512 compress validation sequences, logits rounded to bf16, KL vs the stored BF16 reference logits with "
              "suc.analysis.sensitivity.logit_divergence (identical to the SUC evaluate stage)",
        "decode_batch_ppl": m["batch"]}}
    if ppl_rows:
        nll = sum(r["nll_sum"] for r in ppl_rows)
        nt = sum(r["tokens"] for r in ppl_rows)
        ppl = {"ppl": math.exp(nll / nt), "nll_mean": nll / nt, "tokens_scored": nt, "windows": len(ppl_rows),
               "windows_total": meta["ppl_windows"], "seq_len": meta["seq_len"],
               "complete": len(ppl_rows) == meta["ppl_windows"],
               "window_ids": sorted(r["row"] for r in ppl_rows),
               "top1_next_token_accuracy": sum(r["top1_hits"] for r in ppl_rows) / nt,
               "seconds": sum(r["seconds"] for r in ppl_rows)}
        body["perplexity"] = ppl
    body["reference_perplexity"] = meta["ref_ppl"]
    body["kl"] = kl
    body["batch_consistency"] = _batch_consistency(key) if m["batch"] > 1 else None
    body["kl_rows_seconds"] = sum(r["seconds"] for r in _rows(w / "kl_rows.jsonl"))
    _write(key, "gguf_quality", body, meta["wikitext_hash"])


def cmd_tokcheck(key: str) -> dict:
    """ds4's own tokenization of a WikiText sample vs the HF ids (vocabulary identity check; the measurements always
    feed the HF ids, so this only has to show that the ids mean the same strings)."""
    w = _work(key)
    meta = json.loads((w / "meta.json").read_text())
    r = subprocess.run([str(SCORE_BIN), "tokenize", str(_gguf(key)), str(w / "vocab_sample.txt"), *_m(key)["ds4_args"]],
                       cwd=DS4, capture_output=True, text=True)
    ids = [int(x) for x in r.stdout.split()] if r.returncode == 0 else []
    hf = meta["vocab_sample_hf_ids"]
    same = sum(a == b for a, b in zip(ids, hf))
    out = {"returncode": r.returncode, "ds4_tokens": len(ids), "hf_tokens": len(hf), "identical": ids == hf,
           "position_matches": same, "stderr_tail": r.stderr[-1500:]}
    (w / "tokcheck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "stderr_tail"}))
    return out


# ------------------------------------------------------------------------------------------------ ds4_score serve runner

def _ds4_runner(key: str, ctx: int, log: Path):
    """suc.evaluation.runners.Ds4Runner (system kind ds4) on the key's GGUF with its engine flags."""
    from suc.evaluation.runners import Ds4Runner

    return Ds4Runner(f"gguf_{key}", _gguf(key), DS4, tuple(_m(key)["ds4_args"]), ctx=ctx, log=log,
                     info={"kind": "ds4", "file_bytes": _gguf(key).stat().st_size})


def _peak_footprint(stderr: str) -> int | None:
    for line in stderr.splitlines():
        if "peak memory footprint" in line:
            return int(line.split()[0])
    return None


def cmd_perf(key: str, repeats: int = 3, short_only: bool = False) -> None:
    w, m = _work(key), _m(key)
    meta = json.loads((w / "meta.json").read_text())
    runs = []
    if short_only:
        repeats = 1
    for ctx in (() if short_only else (128, 2048)):
        for rep in range(repeats):
            csv = w / f"bench_{ctx}_{rep}.csv"
            cmd = ["/usr/bin/time", "-l", str(DS4 / "ds4-bench"), "--model", str(_gguf(key)), "--prompt-file",
                   str(PERF_PROMPT), "--ctx-start", str(ctx), "--ctx-max", str(ctx), "--gen-tokens", "256",
                   "--csv", str(csv), *m["ds4_args"]]
            t0 = time.perf_counter()
            r = subprocess.run(cmd, cwd=DS4, capture_output=True, text=True)
            row = {"ctx": ctx, "repeat": rep, "returncode": r.returncode, "wall_s": time.perf_counter() - t0,
                   "peak_memory_footprint": _peak_footprint(r.stderr), "stderr_tail": r.stderr[-3000:], "cmd": cmd}
            if r.returncode == 0 and csv.exists():
                lines = csv.read_text().strip().splitlines()
                hdr, val = lines[0].split(","), lines[-1].split(",")
                row["bench"] = {h: float(v) for h, v in zip(hdr, val)}
                b = row["bench"]
                row["prefill_s"] = b["prefill_tokens"] / b["prefill_tps"]
                row["ttft_s"] = row["prefill_s"] + b["gen_first_ms"] / 1000
                row["decode_ms_per_token_steady"] = 1000 / b["gen_steady_tps"]
            runs.append(row)
            print(f"{key} ds4-bench ctx {ctx} #{rep}: rc {r.returncode} {row.get('bench')}", flush=True)
    # frontier short-prompt protocol: same sentence (HF ids), 256 greedy tokens, no stop, warm-up + repeats
    short = []
    try:
        runner = _ds4_runner(key, 1024, w / "serve_perf.log")
        for rep in range(repeats + 1):
            g = runner.generate(meta["short_prompt_ids"], 256, set())
            short.append({"repeat": rep - 1, "prefill_s": g.prefill_s, "prompt_tokens": len(meta["short_prompt_ids"]),
                          "decode_ms_median": statistics.median(g.decode_ms), "decode_ms_mean": statistics.mean(g.decode_ms),
                          "first_step_ms": g.decode_ms[0], "ttft_s": g.prefill_s + g.decode_ms[0] / 1000,
                          "tokens": len(g.tokens), "first_tokens": g.tokens[:16]})
        load_s = runner.info["load_s"]
        runner.close()
    except Exception as e:  # unattended: record and continue
        short.append({"error": f"{type(e).__name__}: {e}"})
        load_s = None

    def med(ctx, k):
        v = [r[k] for r in runs if r["ctx"] == ctx and k in r]
        return statistics.median(v) if v else None

    summary = {}
    for c in (128, 2048):
        pf = [r["bench"]["prefill_tps"] for r in runs if r["ctx"] == c and "bench" in r]
        summary[f"ctx{c}"] = {"prefill_tok_s": statistics.median(pf) if pf else None, "ttft_s": med(c, "ttft_s"),
                              "decode_ms_per_token": med(c, "decode_ms_per_token_steady"),
                              "peak_memory_footprint": med(c, "peak_memory_footprint")}
    sh = [s for s in short if s.get("repeat", -1) >= 0 and "error" not in s]
    summary["short_prompt"] = ({"decode_ms_median": statistics.median(s["decode_ms_median"] for s in sh),
                                "prefill_s": statistics.median(s["prefill_s"] for s in sh),
                                "ttft_s": statistics.median(s["ttft_s"] for s in sh),
                                "prompt_tokens": sh[0]["prompt_tokens"], "load_s": load_s} if sh else None)
    body = {"short_only": short_only, "protocol": {"bench": "ds4-bench, prompt experiments/perf/promessi_sposi_head.txt (ds4 tokenization), "
                                  "full prefill of ctx tokens from an empty context, then 256 greedy tokens without EOS; "
                                  "decode = 1000 / gen_steady_tps (steps 2..n), TTFT = prefill + first step; median of "
                                  f"{repeats} fresh processes; peak footprint from /usr/bin/time -l",
                         "short_prompt": "frontier_measure.py generate protocol: same sentence (HF ids), 256 greedy "
                                         "tokens, ds4_score serve, one process, 1 warm-up + "
                                         f"{repeats} repeats, decode median of steps 2..n"},
            "summary": summary, "runs": runs, "short_prompt_runs": short}
    _write(key, "gguf_perf", body, hashlib.sha256(PERF_PROMPT.read_bytes()).hexdigest())


def cmd_eval(key: str, limit: int | None = None, suite: str | None = None, ctx: int | None = None) -> None:
    """The suc eval core suite on the GGUF: one system of kind ds4 (suc.evaluation.runners.Ds4Runner) in the model's
    eval config (same suite, prompt rendering incl. the DeepSeek encoder patch, stop ids, budgets, thinking rule,
    graders); greedy, no reference. limit: smoke test on the first cases, written to logs/gguf/KEY/ only. suite: another
    suite file (e.g. experiments/eval/suite_hard.json; result named gguf_eval_<suite name>_KEY); ctx: the ds4 KV
    context (default 16384; the hard suite needs ~33k + budget)."""
    from frontier_measure import _patch_suc

    _patch_suc()
    import suc.cli.eval as ev
    from suc.config import SystemConfig, load_eval_config
    from suc.provenance import write_result

    m, w = _m(key), _work(key)
    meta = json.loads((w / "meta.json").read_text())
    cfg = load_eval_config(REPO / m["eval_cfg"])
    sys0 = SystemConfig(name=f"gguf_{key}", kind="ds4", file=str(_gguf(key)), ds4_dir=str(DS4),
                        engine_args=tuple(m["ds4_args"]), **({"ctx": ctx} if ctx else {}))
    tag = "" if suite is None else Path(suite).stem.removeprefix("suite_") + "_"
    cfg = dataclasses.replace(cfg, name=f"gguf_eval_{tag}{key}", systems=(sys0,), reference=None, compare_to=None,
                              limit=limit, **({"suite": suite, "teacher_forced_positions": 0} if suite else {}))
    os.environ["SUC_DS4_LOG"] = str(w / "serve_eval.log")
    res = ev.run_eval(cfg, model_hash_value=meta["gguf"]["fingerprint"], config_path=m["eval_cfg"])
    res["gguf"] = {k: v for k, v in meta["gguf"].items() if k != "metadata"}
    res["config"]["systems_note"] = (f"system gguf_{key} = {meta['gguf']['path']} through ds4_score serve (ds4 engine, "
                                     "kind ds4); same suite / prompt ids / stops / budgets / graders as the SUC rows "
                                     f"of {m['eval_cfg']}")
    if limit is not None:
        path = w / f"eval_{tag}smoke.json"
        path.write_text(json.dumps(res, indent=1, default=str))
    else:
        path = write_result(res, REPO / "experiments" / "results", cfg.name)
    path.with_suffix(".md").write_text(ev.render_markdown(res))
    print(path, res["systems"][f"gguf_{key}"]["summary"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "quality", "perf", "eval", "rows-cmd", "tokcheck", "inventory"])
    ap.add_argument("key", choices=sorted(MODELS))
    ap.add_argument("what", nargs="?")
    ap.add_argument("--short-only", action="store_true", help="perf: only the short-prompt run (1 warm-up + 1)")
    ap.add_argument("--limit", type=int, help="eval: smoke test on the first N cases (written to logs/gguf/KEY/)")
    ap.add_argument("--suite", help="eval: suite file instead of the eval config's (e.g. experiments/eval/suite_hard.json)")
    ap.add_argument("--ctx", type=int, help="eval: ds4 KV context (default 16384)")
    a = ap.parse_args()
    if a.cmd == "prepare":
        cmd_prepare(a.key)
    elif a.cmd == "rows-cmd":
        cmd_rows_cmd(a.key, a.what)
    elif a.cmd == "quality":
        cmd_quality(a.key)
    elif a.cmd == "perf":
        cmd_perf(a.key, short_only=a.short_only)
    elif a.cmd == "eval":
        cmd_eval(a.key, a.limit, a.suite, a.ctx)
    elif a.cmd == "tokcheck":
        cmd_tokcheck(a.key)
    else:
        inv = gguf_inventory(_gguf(a.key))
        print(json.dumps({k: v for k, v in inv.items() if k != "metadata"}, indent=1))


if __name__ == "__main__":
    main()
