#!/usr/bin/env python3
"""Batch-1 speed of a model behind an OpenAI-compatible server (LM Studio `lms server`, llama-server, mlx_lm.server):
TTFT, prefill tok/s and decode tok/s, on the shared prompts of suc.evaluation.speed_prompts (the same inputs as the
SUC C++ / ds4 speed rows):

  short   SHORT_PROMPT (frontier_measure.py generate), raw text, 256 greedy tokens;
  long    the 2048-token prompt (long_prompt_text: decoded first 2048 ids of experiments/perf/promessi_sposi_head.txt),
          256 greedy tokens.

Per prompt: `warmup` untimed rounds, then `repeats` timed ones. LM Studio keeps prompt caches beyond the active slot
(a 2048-token prompt repeated after an unrelated request still came back in ~19 ms), so with --reload CMD every round
first runs CMD (e.g. `lms unload --all && lms load ... --parallel 1`), then one untimed 8-token request on a DIFFERENT
prompt of the same length (the next window of the prompt file: kernels / pipelines warm, nothing to reuse), then the
timed request. Without --reload only the different-prompt request precedes each timed one, and a run whose prefill
exceeds 20k tok/s is flagged prefix_cache_suspected. Every request also sends cache_prompt=false (llama-server). Greedy (suc.evaluation.runners.greedy_sampling), ignore_eos so all 256 tokens are
generated (the completion token count is recorded and checked). Client-side timing over the streamed response:
  TTFT = request start -> first streamed token (HTTP and first-token sampling included);
  prefill tok/s = server prompt tokens / TTFT;
  decode tok/s = (completion tokens - 1) / (last chunk - first chunk).

LM Studio (Studio: ~/.lmstudio/bin/lms; models under ~/.lmstudio/models, our GGUFs hard-linked under suc-local/ by
scripts/gguf_llama.py link):
  lms server start --port 1234
  lms ls                                                      # model keys
  lms load <model key> --identifier suc-speed --context-length 16384 --gpu max --parallel 1 \\
      --no-speculative-draft-mtp -y
  lms ps                                                      # what is loaded
  .venv/bin/python scripts/server_speed.py --name lms_qwen3_8b_q4_k_m --server-model suc-speed \\
      --model-id Qwen/Qwen3-8B --revision <rev> [--gguf FILE] \\
      --reload "lms unload --all && lms load <model key> --identifier suc-speed ... -y"
  lms unload --all; lms server stop
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

LMS = Path("~/.lmstudio/bin/lms").expanduser()


def _tokenizer(model_id: str, revision: str | None):
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer

    d = snapshot_download(model_id, revision=revision, allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"])
    return load_tokenizer(Path(d)), d


def _lms_ps() -> str | None:
    if not LMS.exists():
        return None
    try:
        return subprocess.run([str(LMS), "ps", "--json"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def measure(runner, prompt_text: str, other_text: str, n_ids: int, gen: int, warmup: int, repeats: int,
            reload: str | None = None) -> dict:
    runs = []
    for i in range(warmup + repeats):
        if reload:
            subprocess.run(reload, shell=True, check=True, capture_output=True)
        runner.request(other_text, 8, None, ignore_eos=True)
        r = runner.request(prompt_text, gen, None, ignore_eos=True)
        u = r["usage"] or {}
        dec_s = sum(r["chunk_gap_ms"]) / 1e3          # first text chunk -> last text chunk (audit F04)
        ct, pt = u.get("completion_tokens"), u.get("prompt_tokens")
        row = {"warmup": i < warmup, "ttft_s": r["ttft_s"], "prompt_tokens_server": pt, "prompt_ids": n_ids,
               "completion_tokens": ct, "chunks": r["chunks"], "finish_reason": r["finish_reason"],
               "decode_s": dec_s, "wall_s": r["wall_s"], "server_stats": r["stats"], "reloaded": bool(reload),
               "prefill_tok_s": (pt / r["ttft_s"]) if pt else None,
               # aggregate over the window first -> last text chunk; assumes the first chunk holds one token
               # (exact for one-token-per-chunk servers, an estimate otherwise): see decode_tok_s_definition
               "decode_tok_s": ((ct - 1) / dec_s) if ct and dec_s > 0 else None,
               "decode_tok_s_definition": "aggregate (completion_tokens - 1) / (last chunk - first chunk)",
               "client_e2e_tok_s": r["client_e2e_tok_s"], "server_decode_tok_s": r["server_decode_tok_s"],
               "chunk_gap_ms_median": statistics.median(r["chunk_gap_ms"]) if r["chunk_gap_ms"] else None}
        row["prefix_cache_suspected"] = bool(row["prefill_tok_s"] and pt and pt > 256 and row["prefill_tok_s"] > 2e4)
        runs.append(row)
        print(json.dumps({k: row[k] for k in ("warmup", "ttft_s", "prompt_tokens_server", "completion_tokens",
                                              "prefill_tok_s", "decode_tok_s", "prefix_cache_suspected")}), flush=True)
    timed = [r for r in runs if not r["warmup"]]

    def med(k):
        v = [r[k] for r in timed if r[k] is not None]
        return statistics.median(v) if v else None

    return {"runs": runs, "summary": {"ttft_s": med("ttft_s"), "prefill_tok_s": med("prefill_tok_s"),
                                      "decode_tok_s": med("decode_tok_s"),
                                      "decode_ms_per_token": (1e3 / med("decode_tok_s")) if med("decode_tok_s") else None,
                                      "prompt_tokens_server": timed[0]["prompt_tokens_server"], "prompt_ids": n_ids,
                                      "completion_tokens": timed[0]["completion_tokens"],
                                      "prefix_cache_suspected": any(r["prefix_cache_suspected"] for r in timed)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="result name (experiments/results/<ts>_server_speed_<name>.json)")
    ap.add_argument("--base-url", default="http://127.0.0.1:1234")
    ap.add_argument("--server-model", required=True, help="model id / identifier the server knows")
    ap.add_argument("--model-id", required=True, help="HF checkpoint (tokenizer; metadata model)")
    ap.add_argument("--revision")
    ap.add_argument("--gguf", help="the GGUF file the server runs (fingerprint = model_hash)")
    ap.add_argument("--engine", default="lmstudio", help="free-text engine label (lmstudio-llama.cpp, lmstudio-mlx...)")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prompts", default="short,long")
    ap.add_argument("--reload", help="shell command run before every round (unload + load the model: cold caches)")
    ap.add_argument("--out-dir", default=str(REPO / "experiments" / "results"))
    a = ap.parse_args()

    from suc.evaluation.runners import OpenAIServerRunner, busy_processes
    from suc.evaluation.speed_prompts import LONG_PROMPT_FILE, LONG_PROMPT_TOKENS, SHORT_PROMPT, long_prompt_ids
    from suc.provenance import run_metadata, write_result

    tok, tok_dir = _tokenizer(a.model_id, a.revision)
    runner = OpenAIServerRunner(a.name, a.base_url, a.server_model, tok, info={"kind": "openai_server"})
    n_short = len(tok.encode(SHORT_PROMPT, add_special_tokens=False))
    all_ids = long_prompt_ids(tok, 2 * LONG_PROMPT_TOKENS)
    long_ids = long_prompt_ids(tok)
    prompts = {"short": (SHORT_PROMPT, tok.decode(all_ids[LONG_PROMPT_TOKENS:LONG_PROMPT_TOKENS + n_short]), n_short),
               "long": (tok.decode(long_ids), tok.decode(all_ids[LONG_PROMPT_TOKENS:]), len(long_ids))}
    other = busy_processes()
    body = {"prompts": {}}
    for p in a.prompts.split(","):
        text, other_text, n = prompts[p]
        print(f"[server_speed] {a.name} prompt {p} ({n} ids)", flush=True)
        body["prompts"][p] = measure(runner, text, other_text, n, a.gen, a.warmup, a.repeats, a.reload)
    gguf, mh = None, f"server:{a.server_model}"
    if a.gguf:
        from gguf_compare import gguf_inventory

        inv = gguf_inventory(Path(a.gguf).expanduser())
        gguf, mh = {k: v for k, v in inv.items() if k != "metadata"}, inv["fingerprint"]
    config = {"name": a.name, "base_url": a.base_url, "server_model": a.server_model, "engine": a.engine,
              "gguf": a.gguf, "gen": a.gen, "warmup": a.warmup, "repeats": a.repeats, "prompts": a.prompts,
              "reload": a.reload,
              "sampling": runner.info["sampling"]}
    dh = hashlib.sha256(SHORT_PROMPT.encode() + LONG_PROMPT_FILE.read_bytes()).hexdigest()
    md = run_metadata(model=a.model_id, model_revision=a.revision or "unknown", model_hash=mh, dataset_hash=dh,
                      seed=0, config_hash=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                      repo_dir=REPO)
    res = {"metadata": md, "kind": "server_speed", "config": config, "gguf": gguf, "lms_ps": _lms_ps(),
           "tokenizer_dir": tok_dir, "other_processes": other, "runner_info": runner.info,
           "protocol": "\n\n".join(__doc__.split("\n\n")[:3]), **body}
    path = write_result(res, Path(a.out_dir), f"server_speed_{a.name}")
    print(path, json.dumps({p: v["summary"] for p, v in body["prompts"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
