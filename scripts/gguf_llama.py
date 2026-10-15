#!/usr/bin/env python3
"""GGUF quantizations of the small models (Qwen3-0.6B, Qwen3-8B, Qwen3-30B-A3B, Qwen3.8-27B) vs the SUC1 files,
through a pinned llama.cpp (scripts/gguf/build_llama_cpp.sh: b11477, Metal, ~/suc-local/llama.cpp) -- the engine of
LM Studio's GGUF runtime. Counterpart of scripts/gguf_compare.py (ds4 engine) for the models ds4 does not run.

KL method (strictly comparable with the SUC rows): the GGUF is scored on exactly the 16 x 512 held-out validation
token ids of the model's `suc compress` run (calibrate/data.safetensors), every row decoded from an empty context by
scripts/gguf/llama_score.cpp (libllama, full logits of every position, rounded to bf16), and compared with the BF16
reference logits stored by that run's reference stage using suc's own logit_divergence
(gguf_compare.kl_vs_reference) -- the same tokens, reference and arithmetic as the SUC KL of the report. llama.cpp's
own `llama-perplexity --kl-divergence` was not used: it tokenizes a text file itself, scores only the second half of
each window and compares with a BF16 GGUF run instead of the checkpoint's MLX BF16 logits, so the SUC KL would have
had to be recomputed on its windows. The BF16 GGUF converted from the same checkpoint is scored too: its KL vs the
MLX BF16 reference is the engine noise floor of every GGUF row.

GGUF files (no download: converted locally from the HF checkpoint snapshot already in the HF cache):
  BF16 via llama.cpp convert_hf_to_gguf.py, then llama-quantize (no imatrix) to the K-quants LM Studio ships
  (Q4_K_M, Q8_0, ...). Files: ~/suc-local/gguf/KEY/<Model>-<TYPE>.gguf; `link` hard-links them into
  ~/.lmstudio/models/suc-local/<Model>-GGUF/ so LM Studio lists them (lms ls) for the speed / eval runs.

  gguf_llama.py convert KEY                  BF16 GGUF from the HF snapshot (CPU only)
  gguf_llama.py quantize KEY TYPE [TYPE..]   llama-quantize BF16 -> TYPE (CPU only); TYPE-IMAT uses the imatrix,
                                             TYPE-IMAT-E4 also embedding / head Q4_K
  gguf_llama.py imatrix KEY [--run-dir D]    llama-imatrix on the compress run's calibration partition (GPU)
  gguf_llama.py prepare KEY [--run-dir D]    KL token rows + vocabulary check (llama-tokenize vs HF ids)
  gguf_llama.py kl KEY TYPE|PATH [--run-dir D]   llama_score rows -> KL result JSON (GPU)
  gguf_llama.py link KEY                     hard links into ~/.lmstudio/models/suc-local/
  gguf_llama.py lms-eval KEY LABEL SERVER_MODEL [--gguf F] [--limit N] [--suite F]
                                             suc eval core suite through the running server (system kind
                                             openai_server; same suite / prompt text / stops / budgets / graders as the
                                             model's SUC eval config) -> gguf_eval_lms_KEY_LABEL result JSON; --suite:
                                             another suite file (e.g. experiments/eval/suite_hard20.json; no teacher
                                             forcing) -> gguf_eval_<suite name>_lms_KEY_LABEL (as gguf_compare.py eval);
                                             --engine llama-server: the server is llama.cpp's llama-server (this
                                             pinned build; started by the caller, e.g.
                                             scripts/studio_llama_server_eval.sh) -> ..._llamacpp_KEY_LABEL
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

LLAMA = Path(os.environ.get("LLAMA_DIR", "~/suc-local/llama.cpp")).expanduser()
BIN = LLAMA / "build" / "bin"
GGUF_ROOT = Path(os.environ.get("SUC_GGUF_DIR", "~/suc-local/gguf")).expanduser()
LMS_MODELS = Path("~/.lmstudio/models/suc-local").expanduser()
WORK = REPO / "logs" / "gguf_llama"

MODELS = {
    "q06": dict(name="Qwen3-0.6B", compress="experiments/configs/compress_qwen3_0.6b_final.yaml",
                eval_cfg="experiments/configs/eval_qwen3_0.6b_final.yaml"),
    "q8": dict(name="Qwen3-8B", compress="experiments/configs/compress_qwen3_8b_final.yaml",
               eval_cfg="experiments/configs/eval_qwen3_8b_final.yaml"),
    "q30": dict(name="Qwen3-30B-A3B", compress="experiments/configs/compress_qwen3_30b_a3b_final.yaml",
                eval_cfg="experiments/configs/eval_qwen3_30b_a3b_final.yaml"),
    "q38_27b": dict(name="Qwen3.8-27B", compress="experiments/configs/compress_qwen3.8_27b_side8.yaml",
                    eval_cfg="experiments/configs/eval_qwen3.8_27b.yaml"),
}


def _cfg(key: str):
    from suc.config import load_compress_config

    return load_compress_config(REPO / MODELS[key]["compress"])


def _run_dir(key: str, override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser()
    cfg = _cfg(key)
    return Path(cfg.run_dir).expanduser() if cfg.run_dir else REPO / "experiments" / "artifacts" / "compress" / cfg.name


def _model_dir(key: str) -> Path:
    from suc.cli.bench import resolve_model_dir

    return resolve_model_dir(_cfg(key))


def _gguf(key: str, qtype: str) -> Path:
    p = Path(qtype).expanduser()
    if p.suffix == ".gguf":
        return p
    return GGUF_ROOT / key / f"{MODELS[key]['name']}-{qtype.upper()}.gguf"


def _work(key: str) -> Path:
    d = WORK / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _llama_commit() -> str:
    r = subprocess.run(["git", "-C", str(LLAMA), "describe", "--tags", "--always"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def _run(cmd: list[str], **kw) -> None:
    print("+ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


# ------------------------------------------------------------------------------------------------ files

def cmd_convert(key: str) -> Path:
    out = _gguf(key, "BF16")
    if out.exists():
        print(f"exists: {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    _run([LLAMA / ".venv" / "bin" / "python", LLAMA / "convert_hf_to_gguf.py", _model_dir(key), "--outtype", "bf16",
          "--outfile", tmp])
    tmp.rename(out)
    return out


def _imatrix_path(key: str) -> Path:
    return GGUF_ROOT / key / f"{MODELS[key]['name']}-imatrix.gguf"


def cmd_imatrix(key: str, run_dir: str | None, chunks: int = 256) -> Path:
    """llama-imatrix on the CALIBRATION partition of the model's compress run (calibrate/data.safetensors
    "calibration" rows, detokenized; never validation / test data), 512-token chunks, BF16 GGUF on the GPU."""
    import mlx.core as mx
    from transformers import AutoTokenizer

    out = _imatrix_path(key)
    if out.exists():
        print(f"exists: {out}")
        return out
    rd, w = _run_dir(key, run_dir), _work(key)
    cal = mx.load(str(rd / "calibrate" / "data.safetensors"))["calibration"]
    tok = AutoTokenizer.from_pretrained(str(_model_dir(key)))
    txt = w / "imatrix_calibration.txt"
    txt.write_text("\n\n".join(tok.decode(r) for r in cal.tolist()))
    tmp = out.with_suffix(".part.gguf")
    _run([BIN / "llama-imatrix", "-m", _gguf(key, "BF16"), "-f", txt, "-o", tmp, "-c", "512", "--chunks", str(chunks),
          "-ngl", "99"])
    tmp.rename(out)
    return out


def cmd_quantize(key: str, types: list[str]) -> None:
    """TYPE[-IMAT][-E4]: -IMAT uses the importance matrix of cmd_imatrix (required by the IQ2 / IQ1 types), -E4 keeps
    token embedding and output head at Q4_K (4.5 bpw, the bits of SUC's affine 4-bit embedding_codec)."""
    src = _gguf(key, "BF16")
    if not src.exists():
        raise SystemExit(f"{src} missing: run convert first")
    for t in types:
        out = _gguf(key, t)
        if out.exists():
            print(f"exists: {out}")
            continue
        tmp = out.with_suffix(".part")
        qt, extra = t.upper(), []
        if qt.endswith("-E4"):          # embedding + output head Q4_K (4.5 bpw = SUC's affine 4-bit head / embedding)
            qt, extra = qt[:-3], ["--token-embedding-type", "q4_k", "--output-tensor-type", "q4_k"]
        if qt.endswith("-IMAT"):
            qt, extra = qt[:-5], ["--imatrix", _imatrix_path(key), *extra]
        _run([BIN / "llama-quantize", *extra, src, tmp, qt, str(os.cpu_count() or 8)])
        tmp.rename(out)


def cmd_link(key: str) -> None:
    """Hard links (same volume, no extra space) under ~/.lmstudio/models/suc-local/<Model>-GGUF/ (LM Studio's
    publisher/repo/file layout), so `lms ls` / `lms load` see the locally made files."""
    d = LMS_MODELS / f"{MODELS[key]['name']}-GGUF"
    d.mkdir(parents=True, exist_ok=True)
    for f in sorted((GGUF_ROOT / key).glob("*.gguf")):
        if f.name.endswith("-BF16.gguf"):
            continue                    # the BF16 file is the KL noise floor only, not an LM Studio row
        dst = d / f.name
        if not dst.exists():
            os.link(f, dst)
        print(dst)


# ------------------------------------------------------------------------------------------------ KL

def cmd_prepare(key: str, run_dir: str | None) -> dict:
    import mlx.core as mx
    import numpy as np
    from mlx_lm.utils import load_tokenizer

    rd, w = _run_dir(key, run_dir), _work(key)
    ref = json.loads((rd / "reference" / "reference.json").read_text())
    kl = np.asarray(mx.load(str(rd / "calibrate" / "data.safetensors"))["validation"][: ref["kl_sequences"]],
                    dtype="<i4")
    kl.tofile(w / "kl_tokens.i32")
    tok = load_tokenizer(_model_dir(key))
    sample = (REPO / "experiments" / "perf" / "promessi_sposi_head.txt").read_text()[:6000]
    (w / "vocab_sample.txt").write_text(sample)
    meta = {"key": key, "model": _cfg(key).model.id, "revision": _cfg(key).model.revision, "run_dir": str(rd),
            "kl_shape": list(kl.shape), "kl_tokens_sha256": hashlib.sha256(kl.tobytes()).hexdigest(),
            "vocab_sample_hf_ids": list(tok.encode(sample, add_special_tokens=False))}
    (w / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{key}: KL rows {kl.shape} from {rd}")
    return meta


def tokcheck(key: str, gguf: Path) -> dict:
    """llama.cpp's tokenization of a text sample vs the HF ids (the rows are fed as ids; this shows that the ids mean
    the same strings in the GGUF vocabulary)."""
    w = _work(key)
    meta = json.loads((w / "meta.json").read_text())
    r = subprocess.run([str(BIN / "llama-tokenize"), "-m", str(gguf), "-f", str(w / "vocab_sample.txt"), "--ids",
                        "--no-bos", "--log-disable"], capture_output=True, text=True)
    ids = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else []
    hf = meta["vocab_sample_hf_ids"]
    return {"returncode": r.returncode, "llama_tokens": len(ids), "hf_tokens": len(hf), "identical": ids == hf,
            "position_matches": sum(a == b for a, b in zip(ids, hf)), "stderr_tail": r.stderr[-800:]}


def gguf_summary(path: Path) -> tuple[dict, dict]:
    from gguf_compare import gguf_inventory

    inv = gguf_inventory(path)
    return {k: v for k, v in inv.items() if k != "metadata"}, inv["metadata"]


def weights_dir_summary(path: Path) -> dict:
    """Identity of a safetensors model directory served by LM Studio's MLX engine (e.g. an MLX 8-bit quant): file
    names and sizes plus config.json; fingerprint = sha256 of that record (no full-file hashing of ~30 GB)."""
    files = sorted((f.name, f.stat().st_size) for f in path.iterdir() if f.is_file() and not f.name.startswith("."))
    cfg = (path / "config.json").read_text() if (path / "config.json").exists() else ""
    blob = json.dumps({"files": files, "config.json": cfg}, sort_keys=True).encode()
    return {"path": str(path), "kind": "weights_dir", "files": [{"name": n, "bytes": b} for n, b in files],
            "bytes": sum(b for _, b in files), "fingerprint": hashlib.sha256(blob).hexdigest(),
            "fingerprint_source": "sha256 of file names / sizes + config.json"}


def cmd_kl(key: str, qtype: str, run_dir: str | None) -> Path:
    from gguf_compare import kl_vs_reference

    from suc.provenance import run_metadata, write_result

    w = _work(key)
    meta = json.loads((w / "meta.json").read_text())
    rd = Path(meta["run_dir"]) if run_dir is None else Path(run_dir).expanduser()
    g = _gguf(key, qtype)
    tag = g.stem
    n, L = meta["kl_shape"]
    rows_path, logits_path = w / f"{tag}_rows.jsonl", w / f"{tag}_logits.bf16"
    tc = tokcheck(key, g)
    t0 = time.perf_counter()
    with open(rows_path, "w") as fo:
        _run([BIN / "llama_score", g, w / "kl_tokens.i32", n, L, logits_path], stdout=fo)
    rows = [json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]
    if len(rows) != n:
        raise SystemExit(f"llama_score wrote {len(rows)} of {n} rows")
    kl = kl_vs_reference(rd, w / "kl_tokens.i32", logits_path, n, L, rows[0]["vocab"])
    nll = sum(r["nll_sum"] for r in rows) / sum(r["tokens"] for r in rows)
    inv, gmeta = gguf_summary(g)
    config = {"key": key, "gguf": str(g), "llama_cpp": _llama_commit(), "tool": "scripts/gguf/llama_score.cpp",
              "run_dir": str(rd), "compress_config": MODELS[key]["compress"], "quant": qtype}
    ch = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    md = run_metadata(model=meta["model"], model_revision=meta["revision"], model_hash=inv["fingerprint"],
                      dataset_hash=meta["kl_tokens_sha256"], seed=0, config_hash=ch, repo_dir=REPO)
    res = {"metadata": md, "kind": "gguf_llama_quality", "config": config, "gguf": inv, "gguf_metadata": gmeta,
           "protocol": {"kl": f"{n} x {L} compress validation sequences (token ids of calibrate/data.safetensors), "
                              "each decoded from an empty context in one batch by llama.cpp, logits rounded to bf16, "
                              "KL vs the run's stored BF16 reference logits with suc.analysis.sensitivity."
                              "logit_divergence (identical to the SUC evaluate stage)"},
           "kl": kl, "nll_mean_fp32": nll, "tokcheck": tc, "seconds": time.perf_counter() - t0,
           "rows_seconds": sum(r["seconds"] for r in rows)}
    path = write_result(res, REPO / "experiments" / "results", f"gguf_llama_quality_{key}_{qtype.lower()}")
    print(path, json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in kl.items()}))
    logits_path.unlink()                # 2.5 GB per 0.6B / 8B file; the rows and the result JSON stay
    return path


def cmd_lms_eval(key: str, label: str, server_model: str, base_url: str, gguf: str | None,
                 limit: int | None, suite: str | None = None, engine: str = "lmstudio",
                 server_cmd: str | None = None) -> Path:
    import dataclasses

    import suc.cli.eval as ev
    from suc.config import SystemConfig, load_eval_config
    from suc.provenance import write_result

    cfg = load_eval_config(REPO / MODELS[key]["eval_cfg"])
    name = f"{'lms' if engine == 'lmstudio' else 'llamacpp'}_{key}_{label.lower()}"
    sys0 = SystemConfig(name=name, kind="openai_server", base_url=base_url, server_model=server_model)
    tag = "" if suite is None else Path(suite).stem.removeprefix("suite_") + "_"
    cfg = dataclasses.replace(cfg, name=f"gguf_eval_{tag}{name}", systems=(sys0,), reference=None, compare_to=None,
                              limit=limit, **({"suite": suite, "teacher_forced_positions": 0} if suite else {}))
    mh = None
    inv = None
    if gguf:
        gp = Path(gguf).expanduser()
        inv = weights_dir_summary(gp) if gp.is_dir() else gguf_summary(gp)[0]
        mh = inv["fingerprint"]
    res = ev.run_eval(cfg, model_hash_value=mh, config_path=MODELS[key]["eval_cfg"])
    res["gguf"] = inv
    if engine == "lmstudio":
        from server_speed import _lms_ps

        res["lms_ps"] = _lms_ps()
        who = f"{server_model} at {base_url}"
    else:
        res["llama_server"] = {"engine": "llama.cpp llama-server (the engine LM Studio uses for GGUF)",
                               "llama_cpp": _llama_commit(), "binary": str(BIN / "llama-server"),
                               "command": server_cmd}
        who = f"llama.cpp llama-server {_llama_commit()} ({server_model}) at {base_url}"
    res["config"]["systems_note"] = (f"system {name} = {who} (/v1/completions, prompt text = "
                                     "suc's rendered chat template); same " + ("suite" if suite is None else
                                                                               f"model / tokenizer (suite {suite})")
                                     + f" / stops / budgets / graders as the SUC rows of {MODELS[key]['eval_cfg']}")
    if limit is not None:
        path = _work(key) / f"eval_smoke_{tag}{label.lower()}.json"
        path.write_text(json.dumps(res, indent=1, default=str))
    else:
        path = write_result(res, REPO / "experiments" / "results", cfg.name)
        path.with_suffix(".md").write_text(ev.render_markdown(res))
    s = res["systems"][name]
    print(path, s["summary"], {k: s["info"].get(k) for k in ("requests", "prompt_token_mismatches")})
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["convert", "quantize", "imatrix", "prepare", "kl", "link", "lms-eval"])
    ap.add_argument("key", choices=sorted(MODELS))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--run-dir", help="the suc compress run dir (default: from the compress config)")
    ap.add_argument("--base-url", default="http://127.0.0.1:1234")
    ap.add_argument("--gguf", help="lms-eval: the GGUF (or the MLX safetensors directory) the server runs "
                                   "(fingerprint = model_hash)")
    ap.add_argument("--limit", type=int, help="lms-eval: smoke test on the first N cases (logs/gguf_llama/KEY/)")
    ap.add_argument("--suite", help="lms-eval: suite file instead of the eval config's (e.g. experiments/eval/suite_hard20.json)")
    ap.add_argument("--engine", choices=["lmstudio", "llama-server"], default="lmstudio",
                    help="lms-eval: which OpenAI-compatible server runs at --base-url (names the result)")
    ap.add_argument("--server-cmd", help="lms-eval --engine llama-server: the server command line (recorded)")
    a = ap.parse_args()
    if a.cmd == "convert":
        cmd_convert(a.key)
    elif a.cmd == "quantize":
        cmd_quantize(a.key, a.args or ["Q4_K_M"])
    elif a.cmd == "imatrix":
        cmd_imatrix(a.key, a.run_dir)
    elif a.cmd == "prepare":
        cmd_prepare(a.key, a.run_dir)
    elif a.cmd == "kl":
        for t in a.args or ["BF16", "Q4_K_M"]:
            cmd_kl(a.key, t, a.run_dir)
    elif a.cmd == "lms-eval":
        cmd_lms_eval(a.key, a.args[0], a.args[1], a.base_url, a.gguf, a.limit, a.suite, a.engine, a.server_cmd)
    else:
        cmd_link(a.key)


if __name__ == "__main__":
    main()
