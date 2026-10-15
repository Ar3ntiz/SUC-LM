"""Audit F03 / F04: the OpenAI-compatible streaming runner distinguishes transport / server failures from answers
(error events, truncated streams -> ServerStreamError -> ERROR in run_eval) and never labels SSE chunk gaps as token
steps (aggregate throughput independent of the chunking)."""
from __future__ import annotations

import http.server
import json
import statistics
import threading
import time

import pytest

from suc.cli import eval as ev
from suc.cli.perf import summarize_runs
from suc.config import eval_config_from_dict
from suc.evaluation import runners as rn
from suc.evaluation.runners import OpenAIServerRunner, ServerStreamError
from test_eval_tools import ANSWERS, MODEL, FakeTokenizer, _suite


@pytest.fixture(autouse=True)
def _fast_metadata(monkeypatch, tmp_path):
    import suc.provenance as prov

    real = prov.run_metadata
    monkeypatch.setattr(ev, "run_metadata", lambda **kw: real(**kw, repo_dir=tmp_path))


def _scripted(*events, done=True, t0=None, dt=0.0):
    """post() replaying events (dicts) at t0 + i * dt; done appends the [DONE] sentinel."""
    def post(url, body, timeout):
        t = time.perf_counter() if t0 is None else t0
        for i, e in enumerate(events):
            yield t + i * dt, e
        if done:
            yield t + len(events) * dt, rn.SSE_DONE
    return post


def _runner(post, tok=None):
    return OpenAIServerRunner("s", "http://unused", "m", tok or FakeTokenizer(), post=post)


def _ch(text="", finish=None):
    return {"choices": [{"text": text, "finish_reason": finish}]}


# ------------------------------------------------------------------------------------------------ F03

def test_error_before_text():
    r = _runner(_scripted({"error": {"message": "generation failed", "type": "server_error"}}))
    with pytest.raises(ServerStreamError, match="generation failed") as e:
        r.request("p", 10)
    assert e.value.partial_text == "" and e.value.chunks == 0
    assert r.info["stream_errors"] == 1


def test_error_after_partial_text_keeps_diagnostics():
    r = _runner(_scripted(_ch("The answer"), _ch(" is"), {"error": {"message": "metal oom"}}))
    with pytest.raises(ServerStreamError) as e:
        r.request("p", 10)
    assert e.value.partial_text == "The answer is" and e.value.chunks == 2


@pytest.mark.parametrize("done", [True, False])
def test_eof_before_finish_reason(done):
    r = _runner(_scripted(_ch("partial"), done=done))
    with pytest.raises(ServerStreamError, match="without a finish_reason"):
        r.request("p", 10)


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_ordinary_finish_with_usage_after(finish):
    r = _runner(_scripted(_ch("a"), _ch("b"), _ch("", finish),
                          {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}))
    out = r.request("p", 10)
    assert out["text"] == "ab" and out["finish_reason"] == finish and out["usage"]["completion_tokens"] == 2


def test_finish_in_last_text_chunk_and_no_usage():
    out = _runner(_scripted(_ch("a"), _ch("b", "length"))).request("p", 2)
    assert out["text"] == "ab" and out["finish_reason"] == "length" and out["client_e2e_tok_s"] is None


def test_valid_empty_completion():
    out = _runner(_scripted(_ch("", "stop"), {"choices": [], "usage": {"completion_tokens": 0}})).request("p", 5)
    assert out["text"] == "" and out["finish_reason"] == "stop" and out["chunks"] == 0
    g = _runner(_scripted(_ch("", "stop"))).generate([1, 2], 5, {1000})
    assert g.tokens == [] and g.stopped


def test_run_eval_records_error_not_failed(tmp_path):
    """Error / truncated streams are ERROR (excluded from grading and timing), ordinary answers are graded."""
    tok = FakeTokenizer()
    tok.eos_token_ids = [1000]
    calls = []

    def post(url, body, timeout):
        calls.append(body)
        t = time.perf_counter()
        n = len(calls)
        if n == 1:                                       # server error after partial text
            yield t, _ch("4")
            yield t, {"error": {"message": "generation failed"}}
            return
        if n == 2:                                       # connection dropped before the finish
            yield t, _ch(ANSWERS[next(k for k in ANSWERS if k in body["prompt"])][:2])
            return
        ans = ANSWERS[next(k for k in ANSWERS if k in body["prompt"])]
        yield t, _ch(ans)
        yield t, _ch("", "stop")
        yield t, rn.SSE_DONE

    cfg = eval_config_from_dict({"name": "t", "model": MODEL, "suite": str(_suite(tmp_path)),
                                 "output_dir": str(tmp_path), "systems": [
                                     {"name": "srv", "kind": "openai_server", "base_url": "http://x",
                                      "server_model": "m"}]})
    res = ev.run_eval(cfg, tokenizer=tok, runner_factory=lambda sc: _runner(post, tok), model_hash_value="h")
    s = res["systems"]["srv"]
    assert [c["status"] for c in s["cases"]] == ["ERROR", "ERROR", "UNGRADED"]
    assert s["cases"][0]["partial_text"] == "4" and "ServerStreamError" in s["cases"][0]["error"]
    assert s["summary"]["errors"] == 2 and s["summary"]["graded"] == 0
    assert s["summary"]["generated_tokens"] == s["cases"][2]["generated_tokens"]   # errors add no tokens / time
    assert s["info"]["stream_errors"] == 2


class _SSEHandler(http.server.BaseHTTPRequestHandler):
    script: list = []

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for line in self.script:
            self.wfile.write(f"data: {line}\n\n".encode())
            self.wfile.flush()
        self.close_connection = True                       # no [DONE] unless scripted

    def log_message(self, *a):
        pass


@pytest.mark.parametrize("script,ok", [
    ([json.dumps(_ch("hi")), json.dumps(_ch("", "stop")), "[DONE]"], True),
    ([json.dumps(_ch("hi"))], False),                                             # early EOF on a real socket
    ([json.dumps(_ch("hi")), json.dumps({"error": {"message": "boom"}}), "[DONE]"], False),   # suc1-server shape
])
def test_real_http_stream(script, ok):
    handler = type("H", (_SSEHandler,), {"script": script})
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        r = OpenAIServerRunner("s", f"http://127.0.0.1:{srv.server_port}", "m", FakeTokenizer(), timeout=10)
        if ok:
            assert r.request("p", 4)["text"] == "hi"
        else:
            with pytest.raises(ServerStreamError):
                r.request("p", 4)
    finally:
        srv.shutdown()
        srv.server_close()


# ------------------------------------------------------------------------------------------------ F04

def _partitioned(n_tokens: int, n_chunks: int, t0: float, t_first: float, t_last: float):
    """An answer of n_tokens split into n_chunks text chunks, first at t_first, last at t_last (one chunk: the whole
    answer flushed at t_last); the window request start -> last chunk is the same for every partitioning."""
    sizes = [n_tokens // n_chunks + (i < n_tokens % n_chunks) for i in range(n_chunks)]
    times = [t_last] if n_chunks == 1 else [t_first + (t_last - t_first) * i / (n_chunks - 1) for i in range(n_chunks)]
    evs = [(t, _ch("x" * k)) for t, k in zip(times, sizes)]
    evs.append((times[-1], _ch("", "length")))
    evs.append((times[-1], {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": n_tokens}}))
    evs.append((times[-1], rn.SSE_DONE))

    def post(url, body, timeout):
        yield from evs
    return post


@pytest.mark.parametrize("n_chunks", [100, 50, 2, 1])
def test_aggregate_throughput_independent_of_chunking(monkeypatch, n_chunks):
    """1 token per chunk, many per chunk, one chunk for the whole answer: same token count and window -> same
    client_e2e_tok_s; chunk gaps are never reported as per-token decode steps."""
    t0 = 1000.0
    clock = iter([t0, t0 + 1.0])                     # request start; end-of-stream
    monkeypatch.setattr(rn.time, "perf_counter", lambda: next(clock, t0 + 1.0))
    r = _runner(_partitioned(100, n_chunks, t0, t0 + 0.2, t0 + 1.0))
    run = r.perf([[1, 2, 3]], 100)
    assert run["decode_ms"] == [] and run["chunks"] == n_chunks
    assert run["client_e2e_tok_s"] == pytest.approx(100 / 1.0)
    assert run["prefill_s"] == pytest.approx(0.2 if n_chunks > 1 else 1.0)
    s = summarize_runs([run], 3, 1)
    assert s["decode_ms"] is None and s["decode_tok_s"] is None            # not manufactured from chunks
    assert s["client_e2e_tok_s"]["median"] == pytest.approx(100.0)
    if n_chunks > 1:
        assert s["chunk_gap_ms"]["median"] == pytest.approx(800.0 / (n_chunks - 1))


def test_delayed_flush_and_usage_only_events(monkeypatch):
    """Empty / usage-only events carry no text: they are not chunks; a delayed flush of everything in one chunk
    still yields the same aggregate."""
    t0 = 50.0
    clock = iter([t0, t0 + 2.0])
    monkeypatch.setattr(rn.time, "perf_counter", lambda: next(clock, t0 + 2.0))
    evs = [(t0 + 0.1, {"choices": []}), (t0 + 0.5, _ch("")), (t0 + 2.0, _ch("y" * 40)),
           (t0 + 2.0, _ch("", "length")), (t0 + 2.0, {"choices": [], "usage": {"completion_tokens": 40}}),
           (t0 + 2.0, rn.SSE_DONE)]
    run = _runner(lambda *a: iter(evs)).perf([[1]], 40)
    assert run["chunks"] == 1 and run["chunk_gap_ms"] == [] and run["client_e2e_tok_s"] == pytest.approx(20.0)


def test_server_timing_preferred_when_reported():
    evs = [_ch("a"), _ch("", "stop"), {"choices": [], "usage": {"completion_tokens": 1},
                                       "timings": {"predicted_per_second": 42.5}}]
    assert _runner(_scripted(*evs)).request("p", 3)["server_decode_tok_s"] == 42.5
    evs[-1] = {"choices": [], "usage": {"completion_tokens": 1}, "stats": {"tokens_per_second": 17.0}}
    assert _runner(_scripted(*evs)).request("p", 3)["server_decode_tok_s"] == 17.0


def test_native_perf_rows_unchanged():
    run = {"prefill_s": 0.5, "decode_ms": [10.0, 20.0, 20.0, 20.0]}
    s = summarize_runs([run], 100, 1)
    assert s["decode_ms"]["median"] == 20.0 and s["decode_tok_s"]["median"] == 50.0 and "chunk_gap_ms" not in s
    assert statistics.median([20.0, 20.0, 20.0]) == 20.0
