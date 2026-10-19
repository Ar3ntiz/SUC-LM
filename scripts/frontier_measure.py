#!/usr/bin/env python3
"""Helpers of the frontier measurement queue (scripts/studio_frontier_measure.sh): DeepSeek-V4-Flash, GLM-5.3-Flash.

  frontier_measure.py generate --name N --suc1 FILE [--prompt P | --long-prompt 2048] [--steps S] [--timeout SEC]
                               -- [suc1-generate flags]
      one C++ `suc1-generate` run (--json captured: load / prefill / decode speed, memory, expert_streaming record
      with hit rate when streamed), written as a result JSON with run metadata (model / model_hash from the file).
      --timeout: the run gets SIGTERM, then SIGKILL 15 s later (its whole process group), and is recorded with
      summary.timed_out = true and the stderr captured so far.
  frontier_measure.py wrap --name N --suc1 FILE --input A.json [--input B.json ...] [--note TEXT]
      wraps JSONs written by other tools (scripts/pread_bench.py, suc-expert-bench, `suc generate --json`) into one
      result JSON with run metadata.
  frontier_measure.py glm-check --snapshot DIR --suc1 GLM_FILE [--build DIR]
      GLM-5.3-Flash on the real snapshot: (a) C++ tokenizer (suc-tokenize) vs HF `tokenizers` on the WikiText-2 test
      lines + edge cases (encode, encode with special tokens, decode); (b) `suc1-generate --chat --print-prompt` ids vs
      transformers `apply_chat_template(add_generation_prompt=True, enable_thinking=...)` with / without system turn.
  frontier_measure.py suc {eval|perf} CONFIG
      the `suc` CLI with two run-time extensions that the YAML schema does not have (yet):
        - suc1_python systems whose name ends in "_streamed" load with SSD expert streaming
          (suc.runtime.expert_streaming.load_streamed_suc1_model, budget $SUC_PY_EXPERT_BUDGET bytes, default 95e9,
          hot set auto); the budget, pinned set and cache statistics are recorded in the system's info;
        - tokenizers without a chat_template (DeepSeek-V4: encoding/encoding_dsv4.py) render the eval prompts with the
          checkpoint's own encode_messages (thinking_mode "thinking" / "chat"), as suc.data.sources does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _wired_limit() -> int | None:
    try:
        return int(subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True,
                                  timeout=10).stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _metadata(suc1: str | None, dataset_hash: str, config: dict) -> dict:
    from suc.provenance import run_metadata

    model, rev, mh = "unknown", "unknown", "unknown"
    if suc1 and Path(suc1).expanduser().exists():
        from suc.io.suc1 import SucModel

        meta = SucModel.load(Path(suc1).expanduser(), verify=False).metadata
        model = str(meta.get("model") or meta.get("model_id") or model)
        rev = str(meta.get("model_revision") or meta.get("revision") or rev)
        mh = str(meta.get("model_hash") or mh)
    ch = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    return run_metadata(model=model, model_revision=rev, model_hash=mh, dataset_hash=dataset_hash, seed=0,
                        config_hash=ch, repo_dir=REPO)


def _write(name: str, suc1: str | None, dataset_hash: str, config: dict, body: dict) -> Path:
    from suc.provenance import write_result

    metadata = _metadata(suc1, dataset_hash, config)
    if build_commit := os.environ.get("SUC_BENCH_BUILD_COMMIT"):
        # Isolated baseline binaries can be measured from a newer benchmark checkout.
        import re
        if not re.fullmatch(r"[0-9a-f]{40}", build_commit):
            raise ValueError("SUC_BENCH_BUILD_COMMIT must be a full Git hash")
        metadata["measurement_git_commit"] = metadata["git_commit"]
        metadata["git_commit"] = build_commit
    res = {"metadata": metadata, "kind": name, "config": config,
           "iogpu_wired_limit_mb": _wired_limit(), **body}
    path = write_result(res, Path(os.environ.get("SUC_RESULTS_DIR", REPO / "experiments" / "results")), name)
    print(path)
    return path


# ------------------------------------------------------------------------------------------------ generate / wrap

def _run_with_timeout(cmd: list[str], timeout: float) -> tuple[subprocess.CompletedProcess, bool]:
    """Runs cmd in its own process group; past `timeout` seconds (0 = none) the group gets SIGTERM, then SIGKILL after
    15 s. subprocess.run(timeout=) only kills the direct child and then blocks reading its pipes, which any surviving
    descendant (or a child stuck in the kernel) keeps open: the queue then never moves on."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout or None)
        return subprocess.CompletedProcess(cmd, p.returncode, out, err), False
    except subprocess.TimeoutExpired:
        pass
    for sig, grace in ((signal.SIGTERM, 15), (signal.SIGKILL, 30)):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            pass
        try:
            out, err = p.communicate(timeout=grace)
            break
        except subprocess.TimeoutExpired:
            out, err = "", ""
    else:   # still not gone (uninterruptible): stop waiting, close our ends of the pipes
        for f in (p.stdout, p.stderr):
            f and f.close()
    err = (err or "") + f"\n[frontier_measure] timeout after {timeout:g} s: process group killed"
    return subprocess.CompletedProcess(cmd, p.returncode if p.returncode is not None else -9, out or "", err), True


def _long_prompt_ids(build: Path, suc1: str, n: int) -> list[int]:
    """The shared long speed prompt (suc.evaluation.speed_prompts): the first n ids of the raw tokenization of
    LONG_PROMPT_FILE with the file's own tokenizer (C++ suc-tokenize = HF `tokenizers`, tested bit for bit)."""
    from suc.evaluation.speed_prompts import LONG_PROMPT_FILE

    with tempfile.TemporaryDirectory() as d:
        src = Path(d, "text.json")
        src.write_text(json.dumps([LONG_PROMPT_FILE.read_text()]))
        r = subprocess.run([str(build / "suc-tokenize"), "--suc1", str(Path(suc1).expanduser()), "--encode-json",
                            str(src)], capture_output=True, text=True, check=True)
    ids = [int(x) for x in r.stdout.strip().split(",")]
    if len(ids) < n:
        raise SystemExit(f"{LONG_PROMPT_FILE} has only {len(ids)} tokens, {n} requested")
    return ids[:n]


def cmd_generate(a, extra: list[str]) -> int:
    binary = Path(a.build).expanduser() / "suc1-generate"
    with tempfile.TemporaryDirectory() as d:
        if a.long_prompt:
            ids = _long_prompt_ids(Path(a.build).expanduser(), a.suc1, a.long_prompt)
            Path(d, "prompt.txt").write_text(",".join(map(str, ids)))
            src = ["--tokens-file", f"{d}/prompt.txt"]
            a.prompt = "long:" + hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
        else:
            src = ["--prompt", a.prompt]
        cmd = [str(binary), str(Path(a.suc1).expanduser()), *src, "--steps", str(a.steps),
               "--json", f"{d}/run.json", *extra]
        print(" ".join(cmd), flush=True)
        r, timed_out = _run_with_timeout(cmd, a.timeout)
        sys.stderr.write(r.stderr[-4000:])
        run = json.loads(Path(d, "run.json").read_text()) if Path(d, "run.json").exists() else None
    for k in ("logits", "hidden"):          # keep the JSON small
        if isinstance(run, dict):
            run.pop(k, None)
    es = (run or {}).get("expert_streaming") or {}
    st = es.get("stats", es)
    req, hits = st.get("requests"), st.get("hits")
    summary = {"returncode": r.returncode, "hit_rate": (hits / req) if req else None, "timed_out": timed_out}
    if run and run.get("decode_ms_per_token"):
        ms = sorted(float(x) for x in run["decode_ms_per_token"])
        summary["decode_ms_median"] = ms[len(ms) // 2]
        summary["decode_tok_s_median"] = 1000.0 / ms[len(ms) // 2]
    if run and run.get("prefill_s"):
        # prefill = the prompt's forward (wall, excluding pipeline compiles it triggered); TTFT = prefill + the first
        # decode step (the first generated token comes from the prefill's logits, the step makes the second one
        # available; reported as in §1b: prefill + one decode step)
        pf = float(run["prefill_s"]) - float(run.get("prefill_compile_s") or 0.0)
        summary["prompt_tokens"] = run.get("prompt_tokens")
        summary["prefill_s"] = pf
        summary["prefill_tok_s"] = run["prompt_tokens"] / pf if pf > 0 else None
        summary["prefill_mode"] = run.get("prefill")
        summary["ttft_s"] = pf + (summary.get("decode_ms_median") or 0.0) / 1000.0
    _write(a.name, a.suc1, hashlib.sha256(a.prompt.encode()).hexdigest(),
           {"tool": "suc1-generate", "command": cmd[:1] + cmd[2:], "file": a.suc1, "steps": a.steps,
            "experiment_environment": {key: os.environ.get(key) for key in
                ("SUC_NGRAM_THREADS", "SUC_PREFIX_CACHE", "SUC_PREFILL_QUERY_TILE", "SUC_TCQ_PREFILL_HALF", "SUC_SAMPLING_LEGACY", "SUC_TCQ_REUSE")}},
           {"summary": summary, "run": run, "stderr_tail": r.stderr[-3000:],
            # stage profiles (SUC_*_PROFILE=1) in full: the tail can cut their first lines
            "stderr_profile": [ln for ln in r.stderr.splitlines()
                               if "profile" in ln or re.match(r"^  \S+\s+[0-9.]+ s\s+[0-9.]+ %", ln)]})
    print(json.dumps(summary), flush=True)
    return r.returncode


def cmd_wrap(a) -> int:
    inputs = {}
    for p in a.input:
        p = Path(p)
        inputs[p.name] = json.loads(p.read_text()) if p.exists() else {"missing": str(p)}
    _write(a.name, a.suc1, "none", {"inputs": a.input, "note": a.note}, {"note": a.note, "inputs": inputs})
    return 0


# ------------------------------------------------------------------------------------------------ GLM checks

def cmd_glm_check(a) -> int:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    snap = Path(a.snapshot).expanduser()
    tj = snap / "tokenizer.json"
    build = Path(a.build).expanduser()
    tok = Tokenizer.from_file(str(tj))

    from suc.evaluation.data import WIKITEXT_REPO
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    pqf = hf_hub_download(repo_id=WIKITEXT_REPO, repo_type="dataset",
                          filename="wikitext-2-raw-v1/test-00000-of-00001.parquet")
    lines = [t for t in pq.read_table(pqf).column("text").to_pylist() if t.strip()]
    try:
        sys.path.insert(0, str(REPO / "tests" / "unit"))
        from test_cpp_tokenizer import EDGE_CASES
    except Exception:  # noqa: BLE001  (pytest-only imports)
        EDGE_CASES = ["", " ", "Hello world", "  leading", "trailing  ", "tabs\tand\nnewlines\r\n",
                      "Ciao, come stai? 😀", "日本語のテキスト", "123456789 3.14159", "[gMASK]<sop><|user|>hi"]
    texts = lines + list(EDGE_CASES)

    def run_tok(args: list[str]) -> list[str]:
        r = subprocess.run([str(build / "suc-tokenize"), "--tokenizer", str(tj), *args], capture_output=True,
                           text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])
        return r.stdout.split("\n")[:-1]

    out: dict = {"tokenizer_json": str(tj), "lines": len(texts)}
    with tempfile.TemporaryDirectory() as d:
        Path(d, "in.json").write_text(json.dumps(texts, ensure_ascii=True))
        for special in (False, True):
            try:
                got = run_tok(["--encode-json", f"{d}/in.json"] + (["--add-special-tokens"] if special else []))
                ours = [[int(x) for x in ln.split(",")] if ln else [] for ln in got]
                ref = [tok.encode(t, add_special_tokens=special).ids for t in texts]
                bad = [i for i, (x, y) in enumerate(zip(ours, ref)) if x != y] + \
                    ([-1] if len(ours) != len(ref) else [])
                out[f"encode{'_special' if special else ''}"] = {
                    "tokens": sum(map(len, ref)), "mismatches": len(bad),
                    "examples": [{"text": texts[i], "ours": ours[i][:40], "ref": ref[i][:40]} for i in bad[:5] if i >= 0]}
            except Exception as e:  # noqa: BLE001
                out[f"encode{'_special' if special else ''}"] = {"error": f"{type(e).__name__}: {e}"}
        ref = [tok.encode(t, add_special_tokens=False).ids for t in texts]
        Path(d, "ids.json").write_text(json.dumps(ref))
        try:
            dec = [json.loads(ln) for ln in run_tok(["--decode-json", f"{d}/ids.json"])]
            bad = [i for i, t in enumerate(dec) if t != tok.decode(ref[i], skip_special_tokens=False)]
            out["decode"] = {"mismatches": len(bad) + abs(len(dec) - len(ref))}
        except Exception as e:  # noqa: BLE001
            out["decode"] = {"error": f"{type(e).__name__}: {e}"}

    hf = AutoTokenizer.from_pretrained(str(snap))
    chat = []
    for prompt, system, think in [("Ciao, come stai?", None, False), ("Ciao, come stai?", None, True),
                                  ("Hello world", "You are helpful.", True),
                                  ("What is 17 * 23? Answer with a number.", "You are helpful.", False)]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        text = hf.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=think)
        ref_ids = hf.encode(text, add_special_tokens=False)
        args = [str(build / "suc1-generate"), str(Path(a.suc1).expanduser()), "--prompt", prompt, "--chat",
                "--tokenizer", str(tj), "--print-prompt", "--quiet"]
        args += (["--system", system] if system else []) + ([] if think else ["--no-think"])
        r = subprocess.run(args, capture_output=True, text=True)
        try:
            ours = [int(x) for x in r.stdout.split("prompt:")[1].split()[0].split(",")]
        except (IndexError, ValueError):
            ours = None
        chat.append({"prompt": prompt, "system": system, "think": think, "hf_text": text, "match": ours == ref_ids,
                     "ours": ours, "ref": ref_ids, "returncode": r.returncode, "stderr": r.stderr[-500:]})
    out["chat_template"] = chat
    out["all_ok"] = all(out.get(k, {}).get("mismatches", 1) == 0 for k in ("encode", "encode_special", "decode")) \
        and all(c["match"] for c in chat)
    _write("glm5.3_flash_tokenizer_chat_check", a.suc1, hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
           {"snapshot": str(snap), "suc1": a.suc1}, out)
    print(json.dumps({k: v for k, v in out.items() if k != "chat_template"} |
                     {"chat_match": [c["match"] for c in chat]}, default=str)[:2000])
    return 0 if out["all_ok"] else 1


# ------------------------------------------------------------------------------------------------ patched suc CLI

def _patch_suc() -> None:
    import suc.cli.eval as ev
    import suc.evaluation.runners as rn
    import suc.evaluation.suite as suite
    from suc.data.sources import _model_encoder

    orig_render = suite.render_prompt_ids

    def render_prompt_ids(tokenizer, case, system, think):
        if not getattr(tokenizer, "chat_template", None):
            enc = _model_encoder(tokenizer)
            if enc is not None:
                msgs = ([{"role": "system", "content": system}] if system else []) + [
                    {"role": "user", "content": suite.user_prompt(case)}]
                text = enc(msgs, thinking_mode="thinking" if think else "chat")
                return list(tokenizer.encode(text, add_special_tokens=False))
        return orig_render(tokenizer, case, system, think)

    suite.render_prompt_ids = render_prompt_ids
    ev.render_prompt_ids = render_prompt_ids

    orig_open = rn.open_system

    def open_system(sc, model_dir, **kw):
        if not (sc.kind == "suc1_python" and sc.name.endswith("_streamed")):
            return orig_open(sc, model_dir, **kw)
        import gc
        import time

        import mlx.core as mx
        from suc.runtime.expert_streaming import load_streamed_suc1_model

        budget = int(float(os.environ.get("SUC_PY_EXPERT_BUDGET", "95e9")))
        f = Path(sc.file).expanduser()
        gc.collect()
        mx.clear_cache()
        base = mx.get_active_memory()
        t0 = time.perf_counter()
        fused = sc.python_plan == "fused"
        model, _, cache, rep = load_streamed_suc1_model(f, budget, hot_experts="auto", group=fused, fuse=fused,
                                                        log=lambda s: print(s, flush=True))
        mx.eval(model.parameters())
        mx.synchronize()
        es = dict(rep["expert_streaming"])
        info = {"kind": sc.kind, "file": str(f), "file_bytes": f.stat().st_size,
                **{"file_" + k: v for k, v in rn.suc1_identity(f).items() if k in ("sha256", "sha256_source")},
                "python_plan": sc.python_plan, "expert_streaming": es, "load_s": time.perf_counter() - t0,
                "resident_bytes": int(mx.get_active_memory() - base)}
        runner = rn.MlxRunner(sc.name, model, info=info, **kw)
        close = runner.close

        def close_with_stats():
            es["stats"] = cache.stats.as_dict()     # shared with the copy the CLI took of runner.info
            print(f"[{sc.name}] expert cache: {es['stats']}", flush=True)
            close()

        runner.close = close_with_stats
        return runner

    rn.open_system = open_system


def cmd_suc(rest: list[str]) -> int:
    _patch_suc()
    from suc.cli.main import main

    main(rest)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "suc":
        return cmd_suc(sys.argv[2:])
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--name", required=True)
    g.add_argument("--suc1", required=True)
    from suc.evaluation.speed_prompts import SHORT_PROMPT

    g.add_argument("--prompt", default=SHORT_PROMPT)
    g.add_argument("--steps", type=int, default=256)
    g.add_argument("--long-prompt", type=int, default=0, metavar="N",
                   help="instead of --prompt: the first N ids of the shared long speed prompt (2048 in the protocol)")
    g.add_argument("--build", default="~/suc-local/cpp-build")
    g.add_argument("--timeout", type=float, default=0.0,
                   help="seconds (0 = none): the run's process group is killed and the run recorded as timed out")
    w = sub.add_parser("wrap")
    w.add_argument("--name", required=True)
    w.add_argument("--suc1")
    w.add_argument("--input", action="append", required=True)
    w.add_argument("--note", default="")
    c = sub.add_parser("glm-check")
    c.add_argument("--snapshot", required=True)
    c.add_argument("--suc1", required=True)
    c.add_argument("--build", default="~/suc-local/cpp-build")
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        return cmd_generate(a, extra)
    if a.cmd == "wrap":
        return cmd_wrap(a)
    return cmd_glm_check(a)


if __name__ == "__main__":
    sys.exit(main())
