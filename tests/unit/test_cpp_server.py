"""suc1-server (cpp/tools/suc1_server.cpp): the OpenAI-compatible HTTP server of the C++/Metal runtime, driven with the
`openai` Python client on a tiny random Qwen3 SUC1 file and a synthetic byte-level tokenizer.

Greedy generations (/v1/completions with token-id and text prompts, /v1/chat/completions with the chat template) must
equal suc1-generate on the same file, cut at the first end token (the architecture's eos_token_id and the tokenizer's
<|endoftext|> / <|im_end|>); streamed (SSE) deltas must join to the non-streamed text; stop strings, seeded sampling,
usage counts, /v1/models, /health and request errors are checked as well.

Skipped unless suc1-server and suc1-generate are built (cpp/build.sh) into $SUC_CPP_BUILD (default ~/suc-local/cpp-build).
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from tests.unit.test_cpp_generate import ARCH, _tiny_file   # untied LM head: varied greedy text on the tiny model

tokenizers = pytest.importorskip("tokenizers")
httpx = pytest.importorskip("httpx")
try:
    import openai
except ImportError:      # only the raw-HTTP tests (HTTP_ONLY) run without the SDK (e.g. a venv without the dev extra)
    openai = None

BUILD = Path(os.environ.get("SUC_CPP_BUILD", Path.home() / "suc-local" / "cpp-build"))
SERVER, GENERATE = BUILD / "suc1-server", BUILD / "suc1-generate"

pytestmark = pytest.mark.skipif(not (SERVER.exists() and GENERATE.exists()),
                                reason=f"suc1-server / suc1-generate not built in {BUILD} (run cpp/build.sh)")

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]   # ids 256, 257, 258 (special)
STOP_IDS = {ARCH["eos_token_id"], 256, 258}                   # eos_token_id + the tokenizer's end tokens


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _tokenizer(path: Path, vocab_size: int) -> Path:
    """Byte-level BPE without merges: ids 0..255 the bytes, 256.. the special tokens and <think> / </think>, then
    two-letter tokens up to the model's vocab (decodable, never produced by encode)."""
    b2u = _bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    added = []
    for i, t in enumerate(SPECIAL + ["<think>", "</think>"]):
        vocab[t] = 256 + i
        added.append({"id": 256 + i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
                      "normalized": False, "special": t in SPECIAL})
    letters = "abcdefghijklmnopqrstuvwxyz"
    pairs = ("".join(c) for n in (2, 3) for c in itertools.product(letters, repeat=n))
    while len(vocab) < vocab_size:
        vocab[next(pairs)] = len(vocab)
    tj = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
          "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
          "post_processor": None,
          "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
          "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                    "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                    "vocab": vocab, "merges": []}}
    path.write_text(json.dumps(tj))
    return path


def _start(f: Path, tok: Path, tmp: Path, *extra: str):
    log = open(tmp / "server.log", "w")
    p = subprocess.Popen([str(SERVER), str(f), "--tokenizer", str(tok), "--port", "0", "--max-context", "512",
                          "--model-name", "tiny", *extra], stdout=subprocess.PIPE, stderr=log, text=True)
    line = p.stdout.readline()
    if not line.startswith("listening on "):
        p.kill()
        log.close()
        pytest.fail(f"suc1-server did not start: {line!r}\n{(tmp / 'server.log').read_text()}")
    url = line.split()[-1]
    client = openai.OpenAI(base_url=url + "/v1", api_key="none", max_retries=0, timeout=120) if openai else None
    return p, log, {"url": url, "file": f, "tok": tok, "client": client, "log": tmp / "server.log",
                    "dec": tokenizers.Tokenizer.from_file(str(tok))}


def _stop(p, log) -> None:
    p.terminate()
    p.wait(timeout=10)
    log.close()


# tests that talk raw HTTP (httpx / urllib) and need no openai SDK
HTTP_ONLY = {"test_numeric_fields_validated_before_narrowing", "test_token_ids_checked_before_narrowing",
             "test_documented_sentinels_accepted", "test_runner_against_server_stream_and_metrics",
             "test_seed_int64_boundary_rejected", "test_seed_int64_boundary_accepted"}


@pytest.fixture(autouse=True)
def _needs_sdk(request):
    if openai is None and request.node.originalname not in HTTP_ONLY:
        pytest.skip("openai SDK not installed: it is in pyproject's `dev` dependency group (uv sync installs it)")


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("server")
    p, log, s = _start(_tiny_file(tmp / "tiny.suc1", False), _tokenizer(tmp / "tokenizer.json", ARCH["vocab_size"]),
                       tmp)
    yield s
    _stop(p, log)


def _generate(srv, *args: str) -> tuple[list[int], list[int]]:
    """(prompt ids, generated ids) of suc1-generate (greedy, no stop) on the server's file and tokenizer."""
    r = subprocess.run([str(GENERATE), str(srv["file"]), "--tokenizer", str(srv["tok"]), "--quiet", *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = {ln.split(":", 1)[0]: ln.split(":", 1)[1] for ln in r.stdout.splitlines() if ":" in ln}
    ids = lambda k: [int(x) for x in out.get(k, "").split(",") if x.strip()]   # noqa: E731
    return ids("prompt"), ids("generated")


def _expected(srv, gen: list[int]) -> tuple[str, str, int]:
    """(text, finish_reason, completion_tokens) the server must return for suc1-generate's tokens."""
    cut = next((i for i, t in enumerate(gen) if t in STOP_IDS), None)
    if cut is None:
        return srv["dec"].decode(gen, skip_special_tokens=False), "length", len(gen)
    return srv["dec"].decode(gen[:cut], skip_special_tokens=False), "stop", cut + 1


def _stream_text(stream, chat: bool) -> tuple[str, str | None, object]:
    text, finish, usage = "", None, None
    for ch in stream:
        if ch.usage is not None:
            usage = ch.usage
        for c in ch.choices:
            text += (c.delta.content or "") if chat else c.text
            finish = c.finish_reason or finish
    return text, finish, usage


def test_health_and_models(srv) -> None:
    r = httpx.get(srv["url"] + "/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert [m.id for m in srv["client"].models.list().data] == ["tiny"]
    assert httpx.get(srv["url"] + "/nope").status_code == 404


@pytest.mark.parametrize("prompt", [[5, 17, 300], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [400]])
def test_completions_token_prompt_greedy_matches_generate(srv, prompt) -> None:
    steps = 24
    _, gen = _generate(srv, "--tokens", ",".join(map(str, prompt)), "--steps", str(steps))
    text, finish, n = _expected(srv, gen)
    c = srv["client"].completions.create(model="tiny", prompt=prompt, max_tokens=steps, temperature=0)
    assert c.choices[0].text == text
    assert c.choices[0].finish_reason == finish
    assert (c.usage.prompt_tokens, c.usage.completion_tokens) == (len(prompt), n)
    s = srv["client"].completions.create(model="tiny", prompt=prompt, max_tokens=steps, temperature=0, stream=True,
                                         stream_options={"include_usage": True})
    st, sf, su = _stream_text(s, chat=False)
    assert (st, sf) == (text, finish)
    assert su.completion_tokens == n


def test_completions_text_prompt(srv) -> None:
    ids, gen = _generate(srv, "--prompt", "Hello, world", "--steps", "16")
    text, finish, n = _expected(srv, gen)
    c = srv["client"].completions.create(model="tiny", prompt="Hello, world", max_tokens=16, temperature=0)
    assert (c.choices[0].text, c.choices[0].finish_reason) == (text, finish)
    assert c.usage.prompt_tokens == len(ids) == len("Hello, world")


@pytest.mark.parametrize("system", [None, "Be brief."])
def test_chat_greedy_matches_generate(srv, system) -> None:
    steps = 20
    extra = ["--system", system] if system else []
    ids, gen = _generate(srv, "--prompt", "What is SUC?", "--chat", *extra, "--steps", str(steps))
    text, finish, n = _expected(srv, gen)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": "What is SUC?"}]
    c = srv["client"].chat.completions.create(model="tiny", messages=msgs, max_tokens=steps, temperature=0)
    assert c.choices[0].message.role == "assistant"
    assert (c.choices[0].message.content, c.choices[0].finish_reason) == (text, finish)
    assert (c.usage.prompt_tokens, c.usage.completion_tokens) == (len(ids), n)
    s = srv["client"].chat.completions.create(model="tiny", messages=msgs, max_tokens=steps, temperature=0,
                                              stream=True, stream_options={"include_usage": True})
    st, sf, su = _stream_text(s, chat=True)
    assert (st, sf) == (text, finish)
    assert (su.prompt_tokens, su.completion_tokens) == (len(ids), n)


def test_chat_no_think_and_multi_turn(srv) -> None:
    ids, gen = _generate(srv, "--prompt", "hi", "--chat", "--no-think", "--steps", "8")
    text, finish, _ = _expected(srv, gen)
    c = srv["client"].chat.completions.create(model="tiny", messages=[{"role": "user", "content": "hi"}], max_tokens=8,
                                              temperature=0, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    assert (c.choices[0].message.content, c.choices[0].finish_reason) == (text, finish)
    assert c.usage.prompt_tokens == len(ids)
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "<think>\nx\n</think>\n\nhello"},
            {"role": "user", "content": [{"type": "text", "text": "again"}]}]
    c = srv["client"].chat.completions.create(model="tiny", messages=msgs, max_tokens=4, temperature=0)
    expect = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>user\nagain<|im_end|>\n" \
             "<|im_start|>assistant\n"
    assert c.usage.prompt_tokens == len(srv["dec"].encode(expect, add_special_tokens=False).ids)


def test_stop_strings(srv) -> None:
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    full = srv["client"].completions.create(model="tiny", prompt=prompt, max_tokens=24, temperature=0).choices[0].text
    probe = full.replace("�", "")
    if len(probe) < 6:
        pytest.skip("the tiny model's greedy text is too short for a stop-string probe")
    stop = probe[len(probe) // 2: len(probe) // 2 + 2]
    want = full[:full.index(stop)]
    c = srv["client"].completions.create(model="tiny", prompt=prompt, max_tokens=24, temperature=0, stop=[stop, "@@@"])
    assert (c.choices[0].text, c.choices[0].finish_reason) == (want, "stop")
    st, sf, _ = _stream_text(srv["client"].completions.create(model="tiny", prompt=prompt, max_tokens=24, temperature=0,
                                                              stop=stop, stream=True), chat=False)
    assert (st, sf) == (want, "stop")


def test_sampling_seeded(srv) -> None:
    kw = dict(model="tiny", prompt=[5, 17, 300], max_tokens=16, temperature=1.5, top_p=0.95, seed=3,
              extra_body={"top_k": 50})
    a = srv["client"].completions.create(**kw)
    b = srv["client"].completions.create(**kw)
    assert a.choices[0].text == b.choices[0].text
    st, _, _ = _stream_text(srv["client"].completions.create(**kw, stream=True), chat=False)
    assert st == a.choices[0].text
    _, gen = _generate(srv, "--tokens", "5,17,300", "--steps", "16", "--temperature", "1.5", "--top-p", "0.95",
                       "--top-k", "50", "--seed", "3")
    text, finish, n = _expected(srv, gen)
    assert (a.choices[0].text, a.choices[0].finish_reason, a.usage.completion_tokens) == (text, finish, n)


def test_request_errors(srv) -> None:
    cl = srv["client"]
    with pytest.raises(openai.BadRequestError):
        cl.completions.create(model="tiny", prompt=[5, 600], max_tokens=4)   # id >= vocab
    with pytest.raises(openai.BadRequestError):
        cl.completions.create(model="tiny", prompt=[5], max_tokens=4, n=2)
    with pytest.raises(openai.BadRequestError):
        cl.chat.completions.create(model="tiny", messages=[{"role": "tool", "content": "x"}], max_tokens=4)
    with pytest.raises(openai.BadRequestError):
        cl.completions.create(model="tiny", prompt=list(range(1, 200)) * 3, max_tokens=4)   # >= --max-context
    r = httpx.post(srv["url"] + "/v1/completions", content=b"{not json")
    assert r.status_code == 400 and "error" in r.json()
    # the context bounds max_tokens: 500 prompt tokens + 100 requested in a 512 context -> 12 tokens, "length"
    c = cl.completions.create(model="tiny", prompt=[7] * 500, max_tokens=100, temperature=0)
    assert c.usage.completion_tokens <= 12


def _post(srv, body: dict) -> httpx.Response:
    return httpx.post(srv["url"] + "/v1/completions", json=body, timeout=60)


@pytest.mark.parametrize("field,value", [
    ("max_tokens", -1), ("max_tokens", 0), ("max_tokens", 1.5), ("max_tokens", 2 ** 31), ("max_tokens", 1e30),
    ("max_tokens", "8"), ("max_completion_tokens", -5), ("temperature", -0.5), ("top_p", 1.5), ("min_p", -0.1),
    ("top_k", -2), ("top_k", 2.5), ("seed", -2), ("seed", 0.5)])
def test_numeric_fields_validated_before_narrowing(srv, field, value) -> None:
    """Audit F06: invalid numeric values are HTTP 400 naming the field, never silently narrowed."""
    r = _post(srv, {"model": "tiny", "prompt": [5, 17], "max_tokens": 4, field: value})
    assert r.status_code == 400, (field, value, r.text)
    assert field in r.json()["error"]["message"]


def _post_raw(srv, body: str) -> httpx.Response:
    return httpx.post(srv["url"] + "/v1/completions", content=body.encode(),
                      headers={"Content-Type": "application/json"}, timeout=60)


@pytest.mark.parametrize("seed", ["9223372036854775808", "9223372036854775808.0", "9.223372036854775808e18",
                                  "9.223372036854776e18", "9223372036854775807.0", "1e19", "-2", "-2.0", "0.5"])
def test_seed_int64_boundary_rejected(srv, seed) -> None:
    """Audit F10: 2^63 in every JSON spelling (the parser keeps it as a double) is HTTP 400 naming `seed`, never an
    undefined double-to-int64 conversion that could fall back to the random-seed sentinel."""
    r = _post_raw(srv, '{"model": "tiny", "prompt": [5, 17], "max_tokens": 2, "seed": %s}' % seed)
    assert r.status_code == 400, (seed, r.text)
    assert "seed" in r.json()["error"]["message"]


@pytest.mark.parametrize("seed", ["9223372036854775807", "9223372036854774784.0", "9.223372036854775e18", "256.0",
                                  "-1", "-1.0"])
def test_seed_int64_boundary_accepted(srv, seed) -> None:
    """INT64_MAX (exact integer), the largest double below 2^63, integral doubles and the -1 sentinel are valid."""
    r = _post_raw(srv, '{"model": "tiny", "prompt": [5, 17], "max_tokens": 2, "temperature": 0.8, "seed": %s}' % seed)
    assert r.status_code == 200, (seed, r.text)


def test_token_ids_checked_before_narrowing(srv) -> None:
    vocab = ARCH["vocab_size"]
    for bad in (2 ** 32 + 1, 2 ** 32, -1, vocab):          # 2^32 + 1 used to wrap to token 1
        r = _post(srv, {"model": "tiny", "prompt": [5, bad], "max_tokens": 2})
        assert r.status_code == 400 and "prompt" in r.json()["error"]["message"], bad
    for ok in (0, vocab - 1):                              # boundary-valid ids
        r = _post(srv, {"model": "tiny", "prompt": [5, ok], "max_tokens": 2, "temperature": 0})
        assert r.status_code == 200, r.text


def test_documented_sentinels_accepted(srv) -> None:
    base = {"model": "tiny", "prompt": [5, 17, 300], "max_tokens": 4}
    for extra in ({"top_k": -1}, {"top_k": 0}, {"seed": -1, "temperature": 0.8}, {"max_tokens": 4.0},
                  {"top_p": 1, "min_p": 0}, {"temperature": 0}):
        r = _post(srv, {**base, **extra})
        assert r.status_code == 200, (extra, r.text)


def test_runner_against_server_stream_and_metrics(srv) -> None:
    """Audit F03 / F04 on the real server: the eval runner's streamed text equals the non-streamed answer, the
    stream ends with a finish_reason, no per-token decode_ms is reported over HTTP, usage drives the aggregate."""
    from suc.evaluation.runners import OpenAIServerRunner

    class Tok:
        def decode(self, ids):
            return srv["dec"].decode(ids, skip_special_tokens=False)

        def encode(self, text, add_special_tokens=False):
            return srv["dec"].encode(text, add_special_tokens=False).ids

    r = OpenAIServerRunner("srv", srv["url"], "tiny", Tok(), timeout=120)
    prompt = "Hello, world"
    out = r.request(prompt, 12)
    full = _post(srv, {"model": "tiny", "prompt": prompt, "max_tokens": 12, "temperature": 0}).json()
    assert out["text"] == full["choices"][0]["text"] and out["finish_reason"] == full["choices"][0]["finish_reason"]
    assert out["usage"]["completion_tokens"] == full["usage"]["completion_tokens"]
    run = r.perf([Tok().encode(prompt)], 8)
    assert run["decode_ms"] == [] and run["metrics_version"] == 2
    assert run["client_e2e_tok_s"] and run["client_e2e_tok_s"] > 0


def _family_file(kind: str, tmp: Path) -> Path:
    if kind == "qwen3_moe":
        from tests.unit.test_cpp_generate import ARCH_MOE, _moe_file
        return _moe_file(tmp / "moe.suc1", ARCH_MOE)
    if kind == "glm5_next":
        from tests.unit.test_cpp_glm5 import TINY, glm5_file
        return glm5_file(tmp / "glm.suc1", over=TINY)
    if kind == "deepseek_v4":
        from tests.unit.test_cpp_deepseek_v4 import TINY, ds4_file
        return ds4_file(tmp / "ds4.suc1", native=False, over=TINY)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))   # test_cpp_qwen4 imports its helpers as top-level modules
    from tests.unit.test_cpp_qwen4 import _compress
    return _compress(tmp)


@pytest.mark.parametrize("kind,extra", [("qwen3_moe", ()), ("qwen3_moe", ("--expert-budget", "64M")),
                                        ("glm5_next", ()), ("deepseek_v4", ()), ("qwen4_exp", ())])
def test_model_families_greedy_matches_generate(tmp_path, kind, extra) -> None:
    """Every graph of suc1-generate behind the server (and expert streaming): greedy /v1/completions text = the
    suc1-generate tokens up to the server's end token; streamed deltas join to it; chat renders the family template
    (prompt tokens = suc1-generate --chat --print-prompt)."""
    f = _family_file(kind, tmp_path)
    p, log, s = _start(f, _tokenizer(tmp_path / "tokenizer.json", 4096), tmp_path, *extra)
    try:
        prompt, steps = [5, 6, 7, 8], 12
        _, gen = _generate(s, "--tokens", "5,6,7,8", "--steps", str(steps), *extra)
        c = s["client"].completions.create(model="tiny", prompt=prompt, max_tokens=steps, temperature=0)
        n = c.usage.completion_tokens
        if c.choices[0].finish_reason == "length":
            assert n == steps and c.choices[0].text == s["dec"].decode(gen, skip_special_tokens=False)
        else:
            assert c.choices[0].text == s["dec"].decode(gen[:n - 1], skip_special_tokens=False)
        st, sf, _ = _stream_text(s["client"].completions.create(model="tiny", prompt=prompt, max_tokens=steps,
                                                                temperature=0, stream=True), chat=False)
        assert (st, sf) == (c.choices[0].text, c.choices[0].finish_reason)
        ids, _ = _generate(s, "--prompt", "hi", "--chat", "--print-prompt")
        try:
            ch = s["client"].chat.completions.create(model="tiny", messages=[{"role": "user", "content": "hi"}],
                                                     max_tokens=2, temperature=0)
            assert ch.usage.prompt_tokens == len(ids)
        except openai.BadRequestError as e:   # tiny vocabs (128 / 256) lack the template's special-token ids
            assert "out of range" in str(e) and max(ids) >= 128
    finally:
        _stop(p, log)


def _server_decode_ms(srv, rid: str) -> float:
    """Decode ms/token the server logged for request `rid` (written after the response; polled briefly)."""
    for _ in range(100):
        for ln in srv["log"].read_text().splitlines():
            if ln.startswith(rid + ":") and "ms/token" in ln:
                return float(ln.split("decode ")[1].split(" ms/token")[0])
        time.sleep(0.02)
    pytest.fail(f"no timing line for {rid} in the server log")


@pytest.fixture(scope="module")
def srv_long(tmp_path_factory):
    """Server on a tied-head tiny file: its greedy continuations run 200 tokens without an end token (the untied
    `srv` file reaches EOS within 200 steps on every probe prompt, which suits the stop / EOS tests). Kept separate
    so EOS behaviour is never suppressed to get a long decode."""
    tmp = tmp_path_factory.mktemp("server_long")
    p, log, s = _start(_tiny_file(tmp / "tied.suc1", True), _tokenizer(tmp / "tokenizer.json", ARCH["vocab_size"]),
                       tmp)
    yield s
    _stop(p, log)


def test_greedy_pipelined_decode_speed_matches_generate(srv_long) -> None:
    """Greedy requests run suc1-generate's pipelined decode loop: the same tokens (non-stream and stream) and a decode
    time per token within 1.1x of suc1-generate's (best of 3 runs each, mean over the decode steps). The fixture
    must provide a 200-token continuation without an end token: a violated precondition fails, it does not skip."""
    srv = srv_long
    steps = 200
    for prompt in ([5, 17, 300], [1, 2, 3], [42], [7, 7, 7, 7], [100, 200]):
        _, gen = _generate(srv, "--tokens", ",".join(map(str, prompt)), "--steps", str(steps))
        if not STOP_IDS & set(gen):
            break
    else:
        pytest.fail("speed fixture precondition: every probe prompt reaches an end token within 200 steps")
    text, finish, n = _expected(srv, gen)
    assert (finish, n) == ("length", steps)
    gen_ms, srv_ms, sse_ms = [], [], []
    for _ in range(3):
        js = srv["log"].parent / "gen.json"
        _generate(srv, "--tokens", ",".join(map(str, prompt)), "--steps", str(steps), "--json", str(js))
        d = json.loads(js.read_text())["decode_ms_per_token"]
        assert len(d) == steps - 1
        gen_ms.append(sum(d) / len(d))
        c = srv["client"].completions.create(model="tiny", prompt=prompt, max_tokens=steps, temperature=0)
        assert (c.choices[0].text, c.choices[0].finish_reason, c.usage.completion_tokens) == (text, finish, n)
        srv_ms.append(_server_decode_ms(srv, c.id))
        s = srv["client"].completions.create(model="tiny", prompt=prompt, max_tokens=steps, temperature=0,
                                             stream=True, stream_options={"include_usage": True})
        chunks = list(s)
        assert "".join(c.text for ch in chunks for c in ch.choices) == text
        assert len([ch for ch in chunks if ch.choices and ch.choices[0].text]) > 1   # streamed incrementally
        sse_ms.append(_server_decode_ms(srv, chunks[0].id))
    g, sv, se = min(gen_ms), min(srv_ms), min(sse_ms)
    print(f"decode ms/token: suc1-generate {g:.3f}, suc1-server {sv:.3f} (stream {se:.3f})")
    assert sv <= 1.1 * g and se <= 1.1 * g, (gen_ms, srv_ms, sse_ms)
