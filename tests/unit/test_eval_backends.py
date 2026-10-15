"""`suc eval` system kinds ds4 (GGUF through ds4_score serve) and openai_server (LM Studio / llama-server) with fake
backends: configs, the wire protocols, identical prompt ids / text, greedy request fields, grading through run_eval."""
from __future__ import annotations

import io
import time

import pytest

from suc.cli import eval as ev
from suc.config import eval_config_from_dict
from suc.evaluation.runners import Ds4Runner, OpenAIServerRunner, UnsupportedRun, greedy_sampling
from suc.evaluation.suite import render_prompt_ids
from test_eval_tools import ANSWERS, MODEL, FakeTokenizer, _suite

EOS = 1000


@pytest.fixture(autouse=True)
def _fast_metadata(monkeypatch, tmp_path):
    import suc.provenance as prov

    real = prov.run_metadata
    monkeypatch.setattr(ev, "run_metadata", lambda **kw: real(**kw, repo_dir=tmp_path))


def _answer(text: str) -> str:
    return ANSWERS[next(k for k in ANSWERS if k in text)]


# ------------------------------------------------------------------------------------------------ configs

def test_backend_system_configs_validate():
    cfg = eval_config_from_dict({"name": "e", "model": MODEL, "suite": "s.json", "systems": [
        {"name": "g", "kind": "ds4", "file": "m.gguf", "engine_args": ["--ssd-streaming"], "ctx": 8192},
        {"name": "lms", "kind": "openai_server", "base_url": "http://127.0.0.1:1234", "server_model": "qwen3-8b"}]})
    assert cfg.systems[0].engine_args == ("--ssd-streaming",) and cfg.systems[0].ctx == 8192
    assert cfg.systems[1].server_model == "qwen3-8b"
    bad = [{"name": "g", "kind": "ds4"},                                             # no file
           {"name": "s", "kind": "openai_server", "base_url": "http://x"},           # no server_model
           {"name": "b", "kind": "bf16", "base_url": "http://x", "server_model": "m"},
           {"name": "c", "kind": "suc1_cpp", "file": "m.suc1", "engine_args": ["-x"]}]
    for s in bad:
        with pytest.raises(ValueError):
            eval_config_from_dict({"name": "e", "model": MODEL, "suite": "s", "systems": [s]})


# ------------------------------------------------------------------------------------------------ ds4

class FakeDs4Process:
    """Speaks the ds4_score serve protocol: answers each GEN line from the decoded prompt ids."""

    def __init__(self, tok):
        self.tok, self.requests = tok, []
        self.stdin = self
        self.stdout = io.StringIO("READY\n")

    # stdin side
    def write(self, line):
        f = line.split()
        assert f[0] == "GEN"
        steps, n_stop = int(f[1]), int(f[2])
        stops = {int(x) for x in f[3:3 + n_stop]}
        n = int(f[3 + n_stop])
        prompt = [int(x) for x in f[4 + n_stop:4 + n_stop + n]]
        assert len(prompt) == n
        self.requests.append({"steps": steps, "stops": stops, "prompt": prompt})
        out = (self.tok.encode(_answer(self.tok.decode(prompt))) + [EOS])[:steps]
        stopped = int(any(t in stops for t in out))
        out = [t for t in out if t not in stops]
        ms = " ".join(["1.5"] * max(len(out) - 1, 0))
        self.stdout = io.StringIO(f"OK {stopped} 0.25 {len(out)} {' '.join(map(str, out))} {ms}\n")

    def flush(self):
        pass

    def close(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def test_ds4_runner_protocol_and_eval(tmp_path):
    tok = FakeTokenizer()
    procs = []

    def popen(cmd, **kw):
        assert cmd[1] == "serve" and cmd[2].endswith("m.gguf") and "--ssd-streaming" in cmd
        assert kw["cwd"].endswith("ds4")
        procs.append(FakeDs4Process(tok))
        return procs[-1]

    cfg = eval_config_from_dict({"name": "t", "model": MODEL, "suite": str(_suite(tmp_path)),
                                 "output_dir": str(tmp_path), "systems": [
                                     {"name": "g", "kind": "ds4", "file": str(tmp_path / "m.gguf"),
                                      "ds4_dir": str(tmp_path / "ds4"), "engine_args": ["--ssd-streaming"]}]})

    def factory(sc):
        return Ds4Runner(sc.name, sc.file, sc.ds4_dir, sc.engine_args, sc.ctx, popen=popen, info={"kind": sc.kind})

    tok.eos_token_ids = [EOS]
    res = ev.run_eval(cfg, tokenizer=tok, runner_factory=factory, model_hash_value="h")
    s = res["systems"]["g"]
    assert [r["status"] for r in s["cases"]] == ["PASSED", "PASSED", "UNGRADED"]
    assert s["info"]["runtime"] == "ds4" and s["info"]["engine_args"] == ["--ssd-streaming"]
    # the engine received exactly suc's rendered prompt ids and the eval's stop ids
    reqs = procs[0].requests
    suite_cases = res["cases"]
    for req, c in zip(reqs, suite_cases):
        assert req["prompt"] == render_prompt_ids(tok, c, None, False)
        assert req["stops"] == {EOS} and req["steps"] == c["max_tokens"]
    assert s["cases"][0]["prefill_s"] == 0.25 and s["cases"][0]["decode_ms_median"] == 1.5
    with pytest.raises(UnsupportedRun):
        factory(cfg.systems[0]).score([1], [2])


# ------------------------------------------------------------------------------------------------ OpenAI server

class FakeServer:
    """/v1/completions over SSE: answers from the prompt text, one chunk per character, usage at the end."""

    def __init__(self, tok, prompt_token_offset=0):
        self.tok, self.bodies, self.offset = tok, [], prompt_token_offset

    def __call__(self, url, body, timeout):
        assert url.endswith("/v1/completions") and body["stream"] is True
        self.bodies.append(body)
        ans = _answer(body["prompt"])[:body["max_tokens"]]
        t = time.perf_counter() + 0.05
        for ch in ans:
            t += 0.01
            yield t, {"choices": [{"text": ch, "finish_reason": None}]}
        yield t, {"choices": [{"text": "", "finish_reason": "stop"}]}
        n_prompt = len(self.tok.encode(body["prompt"])) + self.offset
        yield t, {"choices": [], "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(ans)}}


def test_openai_server_runner_sends_suc_text_greedy_and_grades(tmp_path):
    tok = FakeTokenizer()
    tok.eos_token_ids = [EOS]
    server = FakeServer(tok)
    cfg = eval_config_from_dict({"name": "t", "model": MODEL, "suite": str(_suite(tmp_path)),
                                 "output_dir": str(tmp_path), "systems": [
                                     {"name": "lms", "kind": "openai_server", "base_url": "http://127.0.0.1:1234/",
                                      "server_model": "qwen3-0.6b"}]})

    def factory(sc):
        return OpenAIServerRunner(sc.name, sc.base_url, sc.server_model, tok, post=server, info={"kind": sc.kind})

    res = ev.run_eval(cfg, tokenizer=tok, runner_factory=factory, model_hash_value="h")
    s = res["systems"]["lms"]
    assert [r["status"] for r in s["cases"]] == ["PASSED", "PASSED", "UNGRADED"]
    assert s["info"]["requests"] == 3 and s["info"]["prompt_token_mismatches"] == 0
    for body, c in zip(server.bodies, res["cases"]):
        assert body["prompt"] == tok.decode(render_prompt_ids(tok, c, None, False))
        assert body["model"] == "qwen3-0.6b" and body["max_tokens"] == c["max_tokens"]
        for k, v in greedy_sampling().items():
            assert body[k] == v
        assert "ignore_eos" not in body
    assert s["cases"][0]["text"] == ANSWERS["Q-int"]


def test_openai_server_runner_counts_prompt_mismatch_and_perf():
    tok = FakeTokenizer()
    r = OpenAIServerRunner("x", "http://h", "m", tok, post=FakeServer(tok, prompt_token_offset=1))
    ids = tok.encode("Q-int please")
    g = r.generate(ids, 64, {EOS})
    assert g.stopped and g.prefill_s > 0 and g.decode_ms == []           # no per-token timing over HTTP (F04)
    assert len(g.extra["chunk_gap_ms"]) == len(ANSWERS["Q-int"]) - 1
    assert r.info["prompt_token_mismatches"] == 1
    assert r.info["prompt_token_mismatch_examples"][0] == {"ids": len(ids), "server": len(ids) + 1}
    p = r.perf([ids], 5)
    assert p["usage"]["completion_tokens"] == 5 and p["decode_ms"] == [] and len(p["chunk_gap_ms"]) == 4
    assert r._post.bodies[-1]["ignore_eos"] is True
    with pytest.raises(UnsupportedRun):
        r.perf([ids, ids], 5)


def test_speed_prompts_long_prompt_is_2048_ids(tmp_path):
    from suc.evaluation.speed_prompts import LONG_PROMPT_TOKENS, long_prompt_ids, long_prompt_text, write_tokens_file

    tok = FakeTokenizer()
    ids = long_prompt_ids(tok)
    assert len(ids) == LONG_PROMPT_TOKENS == 2048
    assert tok.encode(long_prompt_text(tok))[:100] == ids[:100]
    assert write_tokens_file(ids[:3], tmp_path / "t.txt").read_text() == ",".join(map(str, ids[:3]))
