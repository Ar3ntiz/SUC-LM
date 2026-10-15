"""`suc generate`: greedy generation from a SUC1 file through the Python/MLX runtime (suc.runtime.suc1_model), with
load time, resident memory (measured vs estimated from the directory) and prefill / decode speed. A smoke run, not a
benchmark (`suc perf` / `suc eval` are the measured tools); prints one JSON record and optionally writes it."""
from __future__ import annotations

import gc
import json
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx


def _tokenizer(sm, tokenizer: str | None):
    from suc.evaluation.runners import load_tokenizer

    if tokenizer and Path(tokenizer).expanduser().exists():
        return load_tokenizer(Path(tokenizer).expanduser())
    from huggingface_hub import snapshot_download

    repo = tokenizer or sm.metadata.get("model")
    rev = None if tokenizer else sm.metadata.get("model_revision")
    if not repo:
        raise ValueError("no tokenizer: pass --tokenizer (the file metadata names no model)")
    d = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"])
    return load_tokenizer(Path(d))


def run(file: str, *, prompt: str | None = None, tokens: str | None = None, steps: int = 32,
        tokenizer: str | None = None, affine_modules: bool = True, group: bool = True, chat: bool = False,
        json_out: str | None = None, quiet: bool = False, expert_budget: float | None = None,
        hot_experts: str | None = "auto") -> dict:
    from mlx_lm.models.cache import make_prompt_cache

    from suc.runtime.suc1_model import load_suc1_model

    log = (lambda m: None) if quiet else (lambda m: print(m, file=sys.stderr, flush=True))
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    t0 = time.perf_counter()
    cache_x = None
    if expert_budget:                       # routed experts streamed from SSD (suc.runtime.expert_streaming)
        from suc.runtime.expert_streaming import load_streamed_suc1_model

        model, sm, cache_x, rep = load_streamed_suc1_model(file, int(expert_budget), hot_experts=hot_experts,
                                                           group=group, fuse=group, affine_modules=affine_modules,
                                                           log=log)
    else:
        model, sm, rep = load_suc1_model(file, group=group, fuse=group, affine_modules=affine_modules, log=log)
    mx.synchronize()
    load_s = time.perf_counter() - t0
    resident = mx.get_active_memory() - base
    est = rep["memory_estimate"]
    log(f"loaded in {load_s:.1f} s: resident {resident / 2**30:.2f} GiB (estimate {est['total'] / 2**30:.2f} GiB, "
        f"tables mmap {est['tables_mmap'] / 2**30:.2f} GiB), load peak {mx.get_peak_memory() / 2**30:.2f} GiB")
    tok = None
    if tokens:
        ids = [int(t) for t in tokens.split(",") if t.strip()]
    else:
        tok = _tokenizer(sm, tokenizer)
        text = prompt if prompt is not None else "The capital of France is"
        if chat and getattr(tok, "chat_template", None):
            ids = list(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True))
        else:
            ids = list(tok.encode(text))
    cache = make_prompt_cache(model)
    t1 = time.perf_counter()
    logits = model(mx.array([ids], dtype=mx.uint32), cache=cache)[0, -1]
    nxt = int(mx.argmax(logits).item())
    prefill_s = time.perf_counter() - t1
    out, step_ms = [nxt], []
    eos = set(getattr(tok, "eos_token_ids", []) or []) if tok is not None else set()
    for _ in range(steps - 1):
        if nxt in eos:
            break
        t2 = time.perf_counter()
        logits = model(mx.array([[nxt]], dtype=mx.uint32), cache=cache)[0, -1]
        nxt = int(mx.argmax(logits).item())
        step_ms.append(1e3 * (time.perf_counter() - t2))
        out.append(nxt)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # bytes on macOS
    rec = {"file": str(Path(file).resolve()), "architecture": sm.architecture_id, "prompt_tokens": len(ids),
           "generated": out, "text": tok.decode(out) if tok is not None else None,
           "load_s": load_s, "prefill_s": prefill_s,
           "decode_ms_median": sorted(step_ms)[len(step_ms) // 2] if step_ms else None,
           "memory": {"resident_weights": int(resident), "estimate": est, "peak_mlx": int(mx.get_peak_memory()),
                      "active_end": int(mx.get_active_memory()), "max_rss": int(rss)},
           "affine_modules": rep.get("affine_modules"), "grouped": group}
    if cache_x is not None:
        rec["expert_streaming"] = {**rep["expert_streaming"], "stats": cache_x.stats.as_dict()}
        cache_x.close()
    if json_out:
        Path(json_out).write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return rec


__all__ = ["run"]
