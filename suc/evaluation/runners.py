"""Inference systems of `suc eval` / `suc perf`: one interface over mlx-lm models (BF16, MLX affine, the Python SUC1
runtime), the C++ tool suc1-generate, the ds4 engine on a GGUF file (ds4_score serve) and any OpenAI-compatible HTTP
server (LM Studio, llama-server, mlx_lm.server).

Every runner takes prompt TOKEN IDS (rendered once in Python), decodes greedily and exposes:
  generate(prompt, steps, stop_ids) -> GenOutput   greedy continuation, cut at the first stop id (excluded);
  score(prompt, forced) -> (len(forced), vocab) float32 logits predicting each forced token (teacher forcing);
  perf(prompts, gen_tokens) -> dict                one timed run: full prefill of the prompts (batch = len(prompts)),
                                                   then gen_tokens greedy tokens, EOS not honoured.
Timing conventions (same for every runner): prefill_s = prompt processing up to the first generated token being
available (= TTFT); decode_ms = wall time of each following step (gen_tokens - 1 values). mlx-lm runners prefill in
chunks of prefill_step_size like mlx_lm.generate_step and decode either pipelined (the next step is queued with
mx.async_eval before the current token is read, as generate_step and the C++ pipelined decode do) or synchronously.
The C++ runner starts one process per call (file mapping and pipeline compilation are reported separately as load_s /
compile_s and are not inside any timing window); it decodes batch 1 only. The ds4 runner keeps one `ds4_score serve`
process (fresh context per request; prefill_s and per-step decode times measured inside the engine). The server runner
measures from the client: prefill_s = time to the first streamed chunk (TTFT, HTTP overhead included). A streamed
chunk may carry any number of tokens, so the server runner reports NO per-token decode_ms (audit F04, metric schema
HTTP_METRICS_VERSION): the chunk gaps are kept as chunk_gap_ms, and throughput is aggregate - client_e2e_tok_s =
usage.completion_tokens / (last chunk - request start) and, when the server reports its own timing (llama-server
`timings`, LM Studio `stats`), server_decode_tok_s. A stream that carries an error event or ends without a
finish_reason raises ServerStreamError (the eval harness records ERROR, never a graded answer).
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np


@dataclass
class GenOutput:
    tokens: list[int]
    stopped: bool
    prefill_s: float
    decode_ms: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class UnsupportedRun(RuntimeError):
    """The runner cannot execute this configuration (e.g. batch > 1 on the C++ tool)."""


class ServerStreamError(RuntimeError):
    """A streamed completion failed: an error event inside the SSE stream, or the stream ended (EOF / [DONE]) before a
    finish_reason. partial_text / chunks keep the diagnostic partial output; the request is NOT a completed answer."""

    def __init__(self, message: str, partial_text: str = "", chunks: int = 0) -> None:
        super().__init__(message)
        self.partial_text, self.chunks = partial_text, chunks


def _cut(tokens: list[int], stop_ids: set[int]) -> tuple[list[int], bool]:
    for i, t in enumerate(tokens):
        if t in stop_ids:
            return tokens[:i], True
    return tokens, False


# ------------------------------------------------------------------------------------------------ mlx-lm models

class MlxRunner:
    """An mlx-lm model object (any representation) decoded greedily with a fresh KV cache per call."""

    def __init__(self, name: str, model, *, info: dict | None = None, decode: str = "pipelined",
                 prefill_step_size: int = 2048) -> None:
        if decode not in ("pipelined", "sync"):
            raise ValueError(f"decode must be pipelined or sync, got {decode!r}")
        self.name, self.model, self.decode, self.step = name, model, decode, prefill_step_size
        self.info = dict(info or {})
        self.info.update({"runtime": "mlx", "decode": decode, "prefill_step_size": prefill_step_size})

    def supports_batch(self, b: int) -> bool:
        return True

    def _cache(self):
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(self.model)

    def _prefill(self, toks: mx.array, cache) -> mx.array:
        while toks.shape[1] > 1:
            n = min(self.step, toks.shape[1] - 1)
            self.model(toks[:, :n], cache=cache)
            mx.eval([c.state for c in cache])
            toks = toks[:, n:]
        return mx.argmax(self.model(toks, cache=cache)[:, -1, :], axis=-1)

    def _step(self, y: mx.array, cache) -> mx.array:
        return mx.argmax(self.model(y[:, None], cache=cache)[:, -1, :], axis=-1)

    def _run(self, prompts: list[list[int]], steps: int,
             stop_ids: set[int] | None) -> tuple[list[list[int]], float, list[float], object]:
        """(tokens per row, prefill_s, per-step decode ms, cache); stops early only for batch 1 with stop_ids."""
        cache = self._cache()
        pipe = self.decode == "pipelined"
        stop = stop_ids if (stop_ids and len(prompts) == 1) else None
        mx.synchronize()
        t0 = time.perf_counter()
        y = self._prefill(mx.array(prompts, dtype=mx.int32), cache)
        nxt = None
        if pipe and steps > 1:          # queue step 2 before reading token 1 (mlx_lm.generate_step)
            mx.async_eval(y)
            nxt = self._step(y, cache)
            mx.async_eval(nxt)
        mx.eval(y)
        t_prev = time.perf_counter()
        prefill_s = t_prev - t0
        out, times = [y], []
        if not (stop and int(y.item()) in stop):
            while len(out) < steps:
                if pipe:
                    y = nxt
                    if len(out) + 1 < steps:
                        nxt = self._step(y, cache)
                        mx.async_eval(nxt)
                else:
                    y = self._step(y, cache)
                mx.eval(y)
                t = time.perf_counter()
                times.append(1e3 * (t - t_prev))
                t_prev = t
                out.append(y)
                if stop and int(y.item()) in stop:
                    break
        mx.synchronize()
        return mx.stack(out, axis=1).tolist(), prefill_s, times, cache

    def generate(self, prompt: list[int], steps: int, stop_ids: set[int]) -> GenOutput:
        toks, prefill_s, times, _ = self._run([prompt], steps, stop_ids)
        tokens, stopped = _cut(toks[0], stop_ids)
        return GenOutput(tokens, stopped, prefill_s, times)

    def score(self, prompt: list[int], forced: list[int]) -> np.ndarray:
        seq = mx.array([prompt + forced[:-1]], dtype=mx.int32)
        logits = self.model(seq)[0, len(prompt) - 1:].astype(mx.float32)
        mx.eval(logits)
        return np.array(logits)

    def perf(self, prompts: list[list[int]], gen_tokens: int) -> dict:
        mx.clear_cache()
        mx.reset_peak_memory()
        toks, prefill_s, times, cache = self._run(prompts, gen_tokens, None)
        kv = sum(int(getattr(c, "nbytes", 0) or 0) for c in cache)
        return {"prefill_s": prefill_s, "decode_ms": times, "peak_memory_bytes": int(mx.get_peak_memory()),
                "kv_cache_bytes": kv, "tokens": toks[0][:8]}

    def close(self) -> None:
        self.model = None
        gc.collect()
        mx.clear_cache()


# ------------------------------------------------------------------------------------------------ C++ suc1-generate

class CppRunner:
    """The C++ runtime (cpp/, tool suc1-generate) on a SUC1 file; one process per call."""

    def __init__(self, name: str, file: str | Path, build: str | Path, args: tuple[str, ...] = (),
                 info: dict | None = None) -> None:
        self.name, self.file = name, Path(file).expanduser()
        self.binary = Path(build).expanduser() / "suc1-generate"
        if not self.binary.exists():
            raise FileNotFoundError(f"{self.binary} not found (build it with cpp/build.sh)")
        if not self.file.exists():
            raise FileNotFoundError(self.file)
        self.args = tuple(args)
        self.info = dict(info or {})
        self.info.update({"runtime": "cpp", "binary": str(self.binary), "cpp_args": list(self.args)})

    def supports_batch(self, b: int) -> bool:
        return b == 1

    def _call(self, prompt: list[int], steps: int, *extra: str, files: dict | None = None) -> dict:
        with tempfile.TemporaryDirectory() as d:
            tf = Path(d, "prompt.txt")
            tf.write_text(",".join(map(str, prompt)))
            cmd = [str(self.binary), str(self.file), "--tokens-file", str(tf), "--steps", str(steps),
                   "--json", f"{d}/run.json", *self.args, *[a.replace("{dir}", d) for a in extra]]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"suc1-generate failed ({r.returncode}): {r.stderr.strip()[-2000:]}")
            res = json.loads(Path(d, "run.json").read_text())
            for key, name in (files or {}).items():
                res[key] = np.load(Path(d, name))
            return res

    def generate(self, prompt: list[int], steps: int, stop_ids: set[int]) -> GenOutput:
        res = self._call(prompt, steps, "--stop-eos")
        tokens, stopped = _cut([int(t) for t in res["generated"]], stop_ids)
        return GenOutput(tokens, stopped, float(res["prefill_s"]), [float(x) for x in res.get("decode_ms_per_token", [])],
                         {k: res.get(k) for k in ("load_s", "compile_s", "memory")})

    def score(self, prompt: list[int], forced: list[int]) -> np.ndarray:
        res = self._call(prompt, len(forced), "--force", ",".join(map(str, forced)), "--logits-out",
                         "{dir}/logits.npy", "--quiet", files={"logits": "logits.npy"})
        logits = res["logits"]
        return np.ascontiguousarray(logits[len(prompt) - 1:len(prompt) - 1 + len(forced)], dtype=np.float32)

    def perf(self, prompts: list[list[int]], gen_tokens: int) -> dict:
        if len(prompts) != 1:
            raise UnsupportedRun("suc1-generate decodes batch 1 only")
        res = self._call(prompts[0], gen_tokens, "--quiet")
        mem = res.get("memory") or {}
        return {"prefill_s": float(res["prefill_s"]), "decode_ms": [float(x) for x in res["decode_ms_per_token"]],
                "decode_gpu_ms_median": res.get("decode_gpu_ms_per_token_median"),
                "peak_memory_bytes": mem.get("phys_footprint"), "kv_cache_bytes": mem.get("kv_cache"),
                "memory": mem, "load_s": res.get("load_s"), "compile_s": res.get("compile_s"),
                "tokens": [int(t) for t in res["generated"][:8]]}

    def close(self) -> None:
        pass


# ------------------------------------------------------------------------------------------------ ds4 (GGUF)

def _git_head(path: Path) -> str:
    try:
        r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class Ds4Runner:
    """A GGUF file through the ds4 engine: one persistent `ds4_score serve` process (scripts/gguf/ds4_score.c, built
    into ds4_dir as ds4_score_suc by scripts/gguf/build_ds4_score.sh), greedy, fresh context per request, prompt fed
    as token ids. Protocol per line: "GEN <max_new> <n_stop> <stops...> <n_prompt> <ids...>" ->
    "OK <stopped> <prefill_s> <n> <tokens...> <decode ms per step...>" (stop token excluded)."""

    BINARY = "ds4_score_suc"

    def __init__(self, name: str, gguf: str | Path, ds4_dir: str | Path = "~/Projects/ds4", args: tuple[str, ...] = (),
                 ctx: int = 16384, log: str | Path | None = None, info: dict | None = None, popen=subprocess.Popen):
        self.name, self.gguf = name, Path(gguf).expanduser()
        self.ds4_dir = Path(ds4_dir).expanduser()
        binary = self.ds4_dir / self.BINARY
        if popen is subprocess.Popen:
            if not binary.exists():
                raise FileNotFoundError(f"{binary} not found (build it with scripts/gguf/build_ds4_score.sh)")
            if not self.gguf.exists():
                raise FileNotFoundError(self.gguf)
        self.log = open(log, "a") if log else open(os.devnull, "w")
        t0 = time.perf_counter()
        self.p = popen([str(binary), "serve", str(self.gguf), "--ctx", str(ctx), *args], cwd=str(self.ds4_dir),
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        line = self.p.stdout.readline()
        if not line.startswith("READY"):
            self.close()
            raise RuntimeError(f"ds4_score serve did not start: {line!r}")
        self.info = dict(info or {})
        self.info.update({"runtime": "ds4", "file": str(self.gguf), "ds4_dir": str(self.ds4_dir),
                          "ds4_commit": _git_head(self.ds4_dir), "engine_args": list(args), "ctx": ctx,
                          "load_s": time.perf_counter() - t0})

    def supports_batch(self, b: int) -> bool:
        return b == 1

    def generate(self, prompt: list[int], steps: int, stop_ids: set[int]) -> GenOutput:
        stops = sorted(stop_ids)
        self.p.stdin.write(f"GEN {steps} {len(stops)} {' '.join(map(str, stops))} {len(prompt)} "
                           f"{' '.join(map(str, prompt))}\n")
        self.p.stdin.flush()
        ans = self.p.stdout.readline().split()
        if not ans or ans[0] != "OK":
            raise RuntimeError(f"ds4_score serve: {' '.join(ans)[:500] or 'no answer (process exited?)'}")
        stopped, prefill, n = ans[1] == "1", float(ans[2]), int(ans[3])
        toks = [int(x) for x in ans[4:4 + n]]
        return GenOutput(toks, stopped, prefill, [float(x) for x in ans[4 + n:]])

    def score(self, prompt: list[int], forced: list[int]) -> np.ndarray:
        raise UnsupportedRun("ds4 runner: teacher-forced scoring is not implemented (use reference: null)")

    def perf(self, prompts: list[list[int]], gen_tokens: int) -> dict:
        if len(prompts) != 1:
            raise UnsupportedRun("ds4 runner decodes batch 1 only")
        g = self.generate(prompts[0], gen_tokens, set())
        return {"prefill_s": g.prefill_s, "decode_ms": g.decode_ms, "peak_memory_bytes": None, "kv_cache_bytes": None,
                "tokens": g.tokens[:8]}

    def close(self) -> None:
        try:
            if self.p.stdin:
                self.p.stdin.close()
            self.p.wait(timeout=120)
        except Exception:
            self.p.kill()
        self.log.close()


# ------------------------------------------------------------------------------------------------ OpenAI server

def greedy_sampling() -> dict:
    """Request fields that make llama.cpp / LM Studio / mlx_lm.server sampling greedy and penalty-free (LM Studio
    defaults to repeat_penalty 1.1, top_k 40, min_p 0.05); unknown fields are ignored by the servers."""
    return {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "min_p": 0.0, "repeat_penalty": 1.0,
            "frequency_penalty": 0.0, "presence_penalty": 0.0, "seed": 0}


HTTP_METRICS_VERSION = 2       # 1: decode_ms = SSE chunk gaps labelled as token steps; 2: chunk_gap_ms + aggregates
SSE_DONE = {"__sse_done__": True}   # yielded by _http_post_stream for "data: [DONE]"


def _http_post_stream(url: str, body: dict, timeout: float):
    """POST a JSON body and yield (arrival time, parsed SSE event) for every `data:` line; "[DONE]" yields SSE_DONE
    and ends. A connection that closes without [DONE] simply ends the iteration (the caller checks completion)."""
    import urllib.request

    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                yield time.perf_counter(), SSE_DONE
                return
            yield time.perf_counter(), json.loads(data)


def _server_decode_tok_s(stats: dict) -> float | None:
    """Decode throughput measured by the server itself: llama-server timings.predicted_per_second, LM Studio
    stats.tokens_per_second; None when the server reports none (suc1-server, mlx_lm.server)."""
    for k, f in (("timings", "predicted_per_second"), ("stats", "tokens_per_second")):
        v = (stats.get(k) or {}).get(f) if isinstance(stats.get(k), dict) else None
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


class OpenAIServerRunner:
    """An OpenAI-compatible server (LM Studio `lms server`, llama-server, mlx_lm.server) through /v1/completions,
    streamed. The prompt ids rendered by suc (chat template included) are detokenized and sent as text, so the server
    tokenizes the same text; usage.prompt_tokens is compared with len(ids) on every request (mismatches counted in
    info). Generated tokens are the server's text re-tokenized with the checkpoint tokenizer (grading uses the text).
    Greedy (greedy_sampling), max_tokens = steps, stop = the stop ids' strings plus the server's own end of turn;
    stopped = finish_reason "stop". Timing is client-side (see the module docstring); the server's own statistics
    (LM Studio `stats`, llama-server `timings`) are kept in GenOutput.extra when present."""

    def __init__(self, name: str, base_url: str, model: str, tokenizer, *, timeout: float = 3600.0,
                 info: dict | None = None, post=_http_post_stream):
        self.name, self.url, self.model, self.tok = name, base_url.rstrip("/"), model, tokenizer
        self.timeout, self._post = timeout, post
        self.info = dict(info or {})
        self.info.update({"runtime": "openai_server", "base_url": self.url, "server_model": model,
                          "endpoint": "/v1/completions", "sampling": greedy_sampling(), "requests": 0,
                          "prompt_token_mismatches": 0, "prompt_token_mismatch_examples": []})

    def supports_batch(self, b: int) -> bool:
        return b == 1

    def _stop_strings(self, stop_ids: set[int]) -> list[str]:
        out = []
        for t in sorted(stop_ids):
            s = self.tok.decode([t])
            if s and s not in out:
                out.append(s)
        return out[:4]                       # the OpenAI API allows at most 4 stop sequences

    def request(self, prompt_text: str, steps: int, stop: list[str] | None = None, ignore_eos: bool = False) -> dict:
        """One streamed completion: text, finish_reason, usage, ttft_s, chunk gaps, aggregate throughput, server stats.
        Raises ServerStreamError on an error event or when the stream ends before a finish_reason (audit F03)."""
        body = {"model": self.model, "prompt": prompt_text, "max_tokens": steps, "stream": True,
                "stream_options": {"include_usage": True}, "cache_prompt": False, **greedy_sampling()}
        if stop:
            body["stop"] = stop
        if ignore_eos:
            body["ignore_eos"] = True
        t0 = time.perf_counter()
        parts, times, finish, usage, stats = [], [], None, None, {}
        self.info["requests"] += 1
        for t, ev in self._post(self.url + "/v1/completions", body, self.timeout):
            if ev is SSE_DONE or ev.get("__sse_done__"):
                break
            if ev.get("error") is not None:
                err = ev["error"]
                msg = err.get("message") if isinstance(err, dict) else str(err)
                self.info["stream_errors"] = self.info.get("stream_errors", 0) + 1
                raise ServerStreamError(f"server error event after {len(times)} chunks: {msg}", "".join(parts),
                                        len(times))
            for ch in ev.get("choices") or []:
                txt = ch.get("text") or ""
                if txt:
                    parts.append(txt)
                    times.append(t)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if ev.get("usage"):
                usage = ev["usage"]
            for k in ("stats", "timings"):
                if ev.get(k):
                    stats[k] = ev[k]
        if finish is None:
            self.info["stream_errors"] = self.info.get("stream_errors", 0) + 1
            raise ServerStreamError(f"stream ended without a finish_reason after {len(times)} chunks (truncated)",
                                    "".join(parts), len(times))
        t_end = time.perf_counter()
        ct = (usage or {}).get("completion_tokens")
        last = times[-1] if times else None
        return {"text": "".join(parts), "finish_reason": finish, "usage": usage, "stats": stats,
                "ttft_s": (times[0] - t0) if times else t_end - t0,
                "chunk_gap_ms": [1e3 * (b - a) for a, b in zip(times, times[1:])], "chunks": len(times),
                "completion_tokens": ct,
                # aggregate client throughput over [request start, last text chunk]: independent of the chunking
                "client_e2e_tok_s": (ct / (last - t0)) if ct and last is not None and last > t0 else None,
                "server_decode_tok_s": _server_decode_tok_s(stats),
                "metrics_version": HTTP_METRICS_VERSION, "wall_s": t_end - t0}

    def _check_prompt(self, n_ids: int, usage: dict | None) -> None:
        n = (usage or {}).get("prompt_tokens")
        if n is not None and int(n) != n_ids:
            self.info["prompt_token_mismatches"] += 1
            if len(self.info["prompt_token_mismatch_examples"]) < 10:
                self.info["prompt_token_mismatch_examples"].append({"ids": n_ids, "server": int(n)})

    def generate(self, prompt: list[int], steps: int, stop_ids: set[int]) -> GenOutput:
        r = self.request(self.tok.decode(prompt), steps, self._stop_strings(stop_ids))
        self._check_prompt(len(prompt), r["usage"])
        tokens = list(self.tok.encode(r["text"], add_special_tokens=False)) if r["text"] else []
        return GenOutput(tokens, r["finish_reason"] == "stop", r["ttft_s"], [],      # no per-token timing over HTTP
                         {"usage": r["usage"], "stats": r["stats"], "finish_reason": r["finish_reason"],
                          **{k: r[k] for k in ("chunk_gap_ms", "chunks", "client_e2e_tok_s", "server_decode_tok_s",
                                               "metrics_version")}})

    def score(self, prompt: list[int], forced: list[int]) -> np.ndarray:
        raise UnsupportedRun("openai_server runner: no logits over HTTP (use reference: null)")

    def perf(self, prompts: list[list[int]], gen_tokens: int) -> dict:
        if len(prompts) != 1:
            raise UnsupportedRun("openai_server runner: batch 1 only")
        r = self.request(self.tok.decode(prompts[0]), gen_tokens, None, ignore_eos=True)
        self._check_prompt(len(prompts[0]), r["usage"])
        return {"prefill_s": r["ttft_s"], "decode_ms": [], "peak_memory_bytes": None,
                "kv_cache_bytes": None, "usage": r["usage"], "server_stats": r["stats"], "chunks": r["chunks"],
                "chunk_gap_ms": r["chunk_gap_ms"], "client_e2e_tok_s": r["client_e2e_tok_s"],
                "server_decode_tok_s": r["server_decode_tok_s"], "metrics_version": r["metrics_version"],
                "finish_reason": r["finish_reason"], "tokens": []}

    def close(self) -> None:
        pass


# ------------------------------------------------------------------------------------------------ factory

def suc1_sha256(file: str | Path) -> str:
    """sha256 of the complete SUC1 file (audit F02: a neighbouring container.json is never trusted)."""
    return suc1_identity(file)["sha256"]


def suc1_identity(file: str | Path) -> dict:
    """{path, bytes, sha256, sha256_source} of an artifact (suc.io.hashing.artifact_identity)."""
    from suc.io.hashing import artifact_identity

    return artifact_identity(file)


def open_system(sc, model_dir: Path | None, *, decode: str = "pipelined", prefill_step_size: int = 2048,
                tokenizer=None):
    """A runner for a SystemConfig (suc.config); model_dir = the reference checkpoint (bf16 / mlx_affine; its
    tokenizer for openai_server when `tokenizer` is not given)."""
    info = {"kind": sc.kind}
    if sc.kind == "ds4":
        f = Path(sc.file).expanduser()
        info.update(file_bytes=f.stat().st_size)
        return Ds4Runner(sc.name, f, sc.ds4_dir, sc.engine_args, sc.ctx, info=info,
                         log=os.environ.get("SUC_DS4_LOG"))
    if sc.kind == "openai_server":
        if tokenizer is None:
            if model_dir is None:
                raise ValueError(f"system {sc.name!r} needs the checkpoint tokenizer")
            tokenizer = load_tokenizer(model_dir)
        return OpenAIServerRunner(sc.name, sc.base_url, sc.server_model, tokenizer, info=info)
    if sc.kind == "suc1_cpp":
        ident = suc1_identity(sc.file)
        info.update(file=str(Path(sc.file).expanduser()), file_bytes=ident["bytes"], file_sha256=ident["sha256"],
                    file_sha256_source=ident["sha256_source"])
        return CppRunner(sc.name, sc.file, sc.cpp_build, sc.cpp_args, info)
    gc.collect()
    mx.clear_cache()
    base = mx.get_active_memory()
    t0 = time.perf_counter()
    if sc.kind == "suc1_python":
        from suc.runtime.suc1_model import load_suc1_model

        fused = sc.python_plan == "fused"
        model, _, _ = load_suc1_model(Path(sc.file).expanduser(), group=fused, fuse=fused)
        ident = suc1_identity(sc.file)
        info.update(file=str(Path(sc.file).expanduser()), file_bytes=ident["bytes"], file_sha256=ident["sha256"],
                    file_sha256_source=ident["sha256_source"], python_plan=sc.python_plan)
    else:
        if model_dir is None:
            raise ValueError(f"system {sc.name!r} needs the reference checkpoint")
        from suc.evaluation.baselines import apply_representation, size_report
        from suc.models.streaming import load_lazy

        model, _ = load_lazy(model_dir)     # = mlx_lm.load(lazy=True) + SUC model classes / memory-mapped tables
        if sc.kind == "mlx_affine":
            apply_representation(model, "mlx_affine", bits=sc.bits, group_size=sc.group_size)
            info.update(bits=sc.bits, group_size=sc.group_size)
        rep = size_report(model)
        info.update(weight_bytes=rep["bytes_total"], bpp=rep["bpp"])
    mx.eval(model.parameters())
    mx.synchronize()
    gc.collect()
    info["load_s"] = time.perf_counter() - t0
    info["resident_bytes"] = int(mx.get_active_memory() - base)
    return MlxRunner(sc.name, model, info=info, decode=decode, prefill_step_size=prefill_step_size)


def reference_model_hash(model_dir: Path | None, systems) -> str:
    """model_hash of the reference checkpoint (suc.io.hashing); without local weights the hash recorded in the first
    SUC1 file's metadata (the checkpoint it was compressed from)."""
    if model_dir is not None and any(Path(model_dir).glob("*.safetensors")):
        from suc.io.hashing import model_hash

        return model_hash(model_dir)
    for sc in systems:
        if sc.file and sc.kind.startswith("suc1_"):
            from suc.io.suc1 import SucModel

            h = SucModel.load(Path(sc.file).expanduser(), verify=False).metadata.get("model_hash")
            if h:
                return str(h)
    return "unknown"


def busy_processes(n: int = 8) -> str:
    """Top CPU consumers at the start of a run (benchmark validity record)."""
    try:
        out = subprocess.run(["ps", "-Ao", "pid,pcpu,pmem,comm", "-r"], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    return "\n".join(out.splitlines()[: n + 1])


def _eos_list(v) -> list[int]:
    if isinstance(v, bool) or v is None:
        return []
    if isinstance(v, int):
        return [v]
    if isinstance(v, (list, tuple)):
        return [int(t) for t in v if isinstance(t, int) and not isinstance(t, bool)]
    return []


def model_eos_ids(model_dir: str | Path | None) -> set[int]:
    """End-of-turn token ids declared by a checkpoint: eos_token_id of config.json (top level and the nested
    text_config of multimodal checkpoints) and of generation_config.json. GLM chat models list <|user|> and
    <|observation|> there besides <|endoftext|>; the tokenizer's own eos_token_id is only <|endoftext|>."""
    if model_dir is None:
        return set()
    ids: set[int] = set()
    d = Path(model_dir)
    for name in ("config.json", "generation_config.json"):
        f = d / name
        if not f.exists():
            continue
        try:
            c = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        ids.update(_eos_list(c.get("eos_token_id")))
        if isinstance(c.get("text_config"), dict):
            ids.update(_eos_list(c["text_config"].get("eos_token_id")))
    return ids


def load_tokenizer(model_dir: str | Path):
    """mlx-lm tokenizer whose eos_token_ids are the tokenizer's eos_token_id plus model_eos_ids(model_dir)."""
    from mlx_lm.utils import load_tokenizer as _load

    tok = _load(Path(model_dir))
    base = getattr(tok, "eos_token_id", None)
    ids = set(tok.eos_token_ids) | ({int(base)} if isinstance(base, int) else set()) | model_eos_ids(model_dir)
    return _load(Path(model_dir), eos_token_ids=sorted(ids))


def resolve_model_dir(model_cfg, need_weights: bool) -> Path:
    """Local snapshot of the reference checkpoint (weights only when needed; tokenizer files always)."""
    if model_cfg.local_path:
        return Path(model_cfg.local_path).expanduser()
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.txt", "*.model", "*.jinja"] + (["*.safetensors"] if need_weights else [])
    return Path(snapshot_download(model_cfg.id, revision=model_cfg.revision, allow_patterns=patterns))


__all__ = ["GenOutput", "UnsupportedRun", "MlxRunner", "CppRunner", "Ds4Runner", "OpenAIServerRunner",
           "greedy_sampling", "ServerStreamError", "open_system", "suc1_sha256", "suc1_identity", "busy_processes",
           "resolve_model_dir", "reference_model_hash",
           "model_eos_ids", "load_tokenizer"]
