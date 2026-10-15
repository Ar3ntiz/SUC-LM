"""`suc eval` / `suc perf` (suc.cli.eval, suc.cli.perf, suc.evaluation.suite, suc.evaluation.runners): configs, suites,
the drivers with scripted runners, the mlx-lm runner on a tiny Qwen3 and the C++ runner on a tiny SUC1 file."""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from suc.cli import eval as ev
from suc.cli import perf as pf
from suc.config import eval_config_from_dict, perf_config_from_dict
from suc.evaluation.runners import CppRunner, GenOutput, MlxRunner, UnsupportedRun
from suc.evaluation.suite import load_suite, select_cases, user_prompt
from tiny_models import tiny_qwen3

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fast_metadata(monkeypatch, tmp_path):
    """run_metadata outside the repository (git status on the SMB share takes seconds)."""
    import suc.provenance as prov

    real = prov.run_metadata
    for mod in (ev, pf):
        monkeypatch.setattr(mod, "run_metadata", lambda **kw: real(**kw, repo_dir=tmp_path))


MODEL = {"id": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"}


# ------------------------------------------------------------------------------------------------ configs

def test_eval_config_parses_and_validates():
    cfg = eval_config_from_dict({"name": "e", "model": MODEL, "suite": "s.json", "systems": [
        {"name": "bf16", "kind": "bf16"}, {"name": "q4", "kind": "mlx_affine", "bits": 4},
        {"name": "cpp", "kind": "suc1_cpp", "file": "m.suc1", "cpp_args": ["--projections", "separate"]}],
        "case_ids": ["a", "b"]})
    assert cfg.systems[1].group_size == 64 and cfg.systems[2].cpp_args == ("--projections", "separate")
    assert cfg.case_ids == ("a", "b")
    with pytest.raises(ValueError, match="unknown key"):
        eval_config_from_dict({"name": "e", "model": MODEL, "suite": "s", "systems": [{"name": "a", "kind": "bf16"}],
                               "typo": 1})
    with pytest.raises(ValueError, match="file"):
        eval_config_from_dict({"name": "e", "model": MODEL, "suite": "s", "systems": [{"name": "a", "kind": "suc1_cpp"}]})
    with pytest.raises(ValueError, match="duplicate"):
        eval_config_from_dict({"name": "e", "model": MODEL, "suite": "s",
                               "systems": [{"name": "a", "kind": "bf16"}, {"name": "a", "kind": "bf16"}]})
    with pytest.raises(ValueError, match="reference"):
        eval_config_from_dict({"name": "e", "model": MODEL, "suite": "s", "reference": "x",
                               "systems": [{"name": "a", "kind": "bf16"}]})


def test_perf_frontiers_follow_ds4_bench():
    base = {"name": "p", "model": MODEL, "systems": [{"name": "a", "kind": "bf16"}]}
    lin = perf_config_from_dict({**base, "ctx_start": 2048, "ctx_max": 7000, "step_incr": 2048})
    assert lin.frontiers() == [2048, 4096, 6144, 7000]
    geo = perf_config_from_dict({**base, "ctx_start": 100, "ctx_max": 1000, "step_mul": 2})
    assert geo.frontiers() == [100, 200, 400, 800, 1000]
    assert perf_config_from_dict({**base, "contexts": [512, 128, 512]}).frontiers() == [128, 512]
    with pytest.raises(ValueError):
        perf_config_from_dict(base)                                     # neither contexts nor a sweep
    with pytest.raises(ValueError):
        perf_config_from_dict({**base, "ctx_start": 1, "ctx_max": 8})   # sweep without a step


@pytest.mark.parametrize("path", sorted((ROOT / "experiments/configs").glob("eval_*.yaml")))
def test_repo_eval_configs_load(path):
    from suc.config import load_eval_config

    cfg = load_eval_config(path)
    assert (ROOT / cfg.suite).exists()


@pytest.mark.parametrize("path", sorted((ROOT / "experiments/configs").glob("perf_*.yaml")))
def test_repo_perf_configs_load(path):
    from suc.config import load_perf_config

    cfg = load_perf_config(path)
    assert (ROOT / cfg.prompt_file).exists() and cfg.frontiers()


# ------------------------------------------------------------------------------------------------ suites

@pytest.mark.parametrize("name", ["suite_core.json", "ds4_subset.json", "suite_hard.json", "suite_hard20.json"])
def test_repo_suites_are_valid(name):
    s = load_suite(ROOT / "experiments/eval" / name)
    assert len(s["sha256"]) == 64 and s["cases"]


def test_hard20_suite_extends_the_hard_suite():
    """suite_hard20.json (scripts/build_hard_suite.py --per-subset 20): 20 per subset, round 0 = the 30-case suite."""
    small, big = (load_suite(ROOT / "experiments/eval" / n) for n in ("suite_hard.json", "suite_hard20.json"))
    by_id = {c["id"]: c for c in big["cases"]}
    assert all(by_id.get(c["id"]) == c for c in small["cases"])
    for sub, sel in big["selection"].items():
        assert len(sel["ids"]) == 20 and sel["ids"][:5] == small["selection"][sub]["ids"]
        assert [c["id"] for c in big["cases"] if c["source"] == sub] == sel["ids"]
    assert big["cases"][-1]["source"] == "LongCtx"     # a deadline drops LongCtx rounds first


def test_ds4_prompt_format_and_selection():
    case = {"id": "x", "source": "S", "kind": "choice", "question": "Q?", "choices": ["a", "b"], "answer": "B",
            "think": False, "max_tokens": 8}
    p = user_prompt(case)
    assert p.startswith("Q?\n\nChoices:\nA. a\nB. b\n") and p.endswith("Answer: <letter>")
    assert user_prompt({**case, "kind": "integer", "prompt_style": "ds4"}).endswith("Answer: <integer>")
    assert user_prompt({**case, "prompt_style": "raw"}) == "Q?"
    cases = [{"id": str(i)} for i in range(5)]
    assert [c["id"] for c in select_cases(cases, ("3", "1"))] == ["3", "1"]
    assert [c["id"] for c in select_cases(cases, limit=2)] == ["0", "1"]
    with pytest.raises(ValueError):
        select_cases(cases, ("9",))


# ------------------------------------------------------------------------------------------------ scripted eval

EOS, THINK_CLOSE = 1000, 1001


class FakeTokenizer:
    """Bytes as token ids; '</think>' is one id, EOS = 1000."""
    eos_token_ids = [EOS]

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize, enable_thinking):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + ("<a>" if enable_thinking else "<a></think>")

    def encode(self, text, add_special_tokens=False):
        out = []
        for i, part in enumerate(text.split("</think>")):
            if i:
                out.append(THINK_CLOSE)
            out += list(part.encode())
        return out

    def decode(self, ids):
        s, buf = "", []
        for t in ids:
            if t == THINK_CLOSE:
                s += bytes(buf).decode(errors="replace") + "</think>"
                buf = []
            elif t < 256:
                buf.append(t)
        return s + bytes(buf).decode(errors="replace")


class ScriptedRunner:
    """Answers each prompt with a fixed text (looked up by a substring of the prompt)."""

    def __init__(self, name, answers, vocab=1002, info=None):
        self.name, self.answers, self.vocab = name, answers, vocab
        self.info = info or {"kind": "scripted"}
        self.calls = []
        self.tok = FakeTokenizer()

    def generate(self, prompt, steps, stop_ids):
        self.calls.append((list(prompt), steps))
        text = self.tok.decode(prompt)
        key = next(k for k in self.answers if k in text)
        cont = self.answers[key]
        out = (self.tok.encode(cont) + [EOS])[:steps]
        stopped = EOS in out
        return GenOutput([t for t in out if t != EOS], stopped, 0.01, [1.0] * max(len(out) - 1, 0))

    def score(self, prompt, forced):
        rng = np.random.default_rng(len(prompt))
        logits = rng.standard_normal((len(forced), self.vocab)).astype(np.float32)
        logits[np.arange(len(forced)), forced] += 20.0
        return logits

    def supports_batch(self, b):
        return b == 1

    def perf(self, prompts, gen_tokens):
        if len(prompts) > 1:
            raise UnsupportedRun("batch")
        return {"prefill_s": 0.5, "decode_ms": [2.0] + [1.0] * (gen_tokens - 2), "peak_memory_bytes": 1 << 20,
                "kv_cache_bytes": 1 << 10, "tokens": [1, 2]}

    def close(self):
        pass


def _suite(tmp_path: Path) -> Path:
    cases = [
        {"id": "int", "source": "A", "kind": "integer", "question": "Q-int", "answer": "42", "think": False,
         "max_tokens": 64},
        {"id": "mc", "source": "B", "kind": "choice", "question": "Q-mc", "choices": ["x", "y", "z"], "answer": "C",
         "think": False, "max_tokens": 64},
        {"id": "open", "source": "A", "kind": "open", "question": "Q-open", "think": False, "max_tokens": 64},
    ]
    p = tmp_path / "suite.json"
    p.write_text(json.dumps({"suite": "t", "version": 1, "system": None, "cases": cases}))
    return p


def _eval_cfg(tmp_path, **kw):
    return eval_config_from_dict({"name": "t", "model": MODEL, "suite": str(_suite(tmp_path)),
                                  "systems": [{"name": "ref", "kind": "bf16"}, {"name": "same", "kind": "bf16"},
                                              {"name": "diff", "kind": "bf16"}],
                                  "reference": "ref", "teacher_forced_positions": 4, "output_dir": str(tmp_path),
                                  **kw})


ANSWERS = {"Q-int": "Answer: 42", "Q-mc": "Answer: C", "Q-open": "blue sky"}


def _run(tmp_path, **kw):
    runners = {"ref": ScriptedRunner("ref", ANSWERS), "same": ScriptedRunner("same", ANSWERS),
               "diff": ScriptedRunner("diff", {**ANSWERS, "Q-int": "Answer: 41", "Q-open": "blue sea"})}
    cfg = _eval_cfg(tmp_path, **kw)
    return ev.run_eval(cfg, tokenizer=FakeTokenizer(), runner_factory=lambda sc: runners[sc.name],
                       model_hash_value="h"), runners


def test_eval_grades_and_compares_with_reference(tmp_path):
    res, _ = _run(tmp_path)
    ref, same, diff = (res["systems"][k] for k in ("ref", "same", "diff"))
    assert [r["status"] for r in ref["cases"]] == ["PASSED", "PASSED", "UNGRADED"]
    assert ref["summary"]["passed"] == 2 and ref["summary"]["graded"] == 2
    assert same["agreement_summary"]["identical_outputs"] == 3 and same["agreement_summary"]["answer_agreement"] == 1
    assert [r["status"] for r in diff["cases"]][:2] == ["FAILED", "PASSED"]
    a = diff["agreement"]
    assert a[0]["identical"] is False and a[0]["first_divergence"] == len("Answer: 4") and a[0]["answer_agree"] is False
    assert a[2]["answer_agree"] is None and a[2]["common_prefix"] == len("blue s")
    # teacher forcing: identical scripted logits for equal prompt lengths -> KL 0 for "same"
    assert same["cases"][0]["teacher_forced"]["kl_mean"] == pytest.approx(0, abs=1e-6)
    assert same["cases"][0]["teacher_forced"]["positions"] == 4
    md = ev.render_markdown(res)
    assert "| ref | scripted | 2 / 2 | 100.0% | 1 | 0 |" in md and "## Agreement with `ref` per case" in md
    from suc.provenance import write_result

    write_result(res, tmp_path, "t")   # required metadata present


def test_eval_without_reference_only_grades(tmp_path):
    res, runners = _run(tmp_path, reference=None)
    assert res["reference"] is None
    assert all("agreement" not in s and "teacher_forced" not in s["cases"][0] for s in res["systems"].values())
    assert res["systems"]["diff"]["summary"]["passed"] == 1
    md = ev.render_markdown(res)
    assert "no reference comparison" in md and "## Agreement" not in md


def test_eval_drift_and_regrade(tmp_path):
    res, _ = _run(tmp_path)
    old = json.loads(json.dumps(res))
    old["systems"]["diff"]["cases"][0]["generated_tokens"] += 1
    d = ev.drift(old, res)
    assert d["changed"] == 1 and d["changes"][0]["id"] == "int" and "generated_tokens" in d["changes"][0]
    stored = tmp_path / "r.json"
    res["systems"]["ref"]["cases"][0]["picked"] = "7"          # pretend an older extractor picked another answer
    stored.write_text(json.dumps(res))
    assert ev.regrade(stored) == 0


def test_eval_think_budget_forces_close(tmp_path):
    class Rambler(ScriptedRunner):
        def generate(self, prompt, steps, stop_ids):
            self.calls.append((list(prompt), steps))
            if THINK_CLOSE in prompt[-3:]:
                toks = FakeTokenizer().encode("Answer: 42")
                return GenOutput(toks[:steps], True, 0.0, [])
            return GenOutput([ord("x")] * steps, False, 0.0, [])

    r = Rambler("ref", {})
    p = _suite(tmp_path)
    s = json.loads(p.read_text())
    s["cases"] = [{**s["cases"][0], "think": True, "max_tokens": 40}]
    p.write_text(json.dumps(s))
    cfg = eval_config_from_dict({"name": "t", "model": MODEL, "suite": str(p), "systems": [{"name": "ref", "kind": "bf16"}],
                                 "think_reply_reserve": 16, "teacher_forced_positions": 0})
    res = ev.run_eval(cfg, tokenizer=FakeTokenizer(), runner_factory=lambda sc: r, model_hash_value="h")
    rec = res["systems"]["ref"]["cases"][0]
    assert r.calls[0][1] == 24 and r.calls[1][1] == 16            # budget - reserve, then the reserve
    assert rec["think_close"] == "hard" and rec["status"] == "PASSED"
    assert rec["generated_tokens"] == 24 + len(res["protocol"]["think_close_ids"]) + len("Answer: 42")


def test_model_eos_ids_reads_config_text_config_and_generation_config(tmp_path):
    from suc.evaluation.runners import model_eos_ids

    assert model_eos_ids(None) == set() and model_eos_ids(tmp_path) == set()
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"eos_token_id": [7, 8]}}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [7, 9], "pad_token_id": 7}))
    assert model_eos_ids(tmp_path) == {7, 8, 9}
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 5}))
    assert model_eos_ids(tmp_path) == {5, 7, 9}


def test_eval_stops_at_generation_config_end_of_turn(tmp_path):
    """GLM: the tokenizer's eos is <|endoftext|> only; <|user|> (generation_config eos_token_id) must end the turn,
    otherwise the reply runs on (and a thinking case gets a forced </think> and a second answer)."""
    from suc.evaluation.runners import _cut

    USER = 1002

    class TurnRunner(ScriptedRunner):
        def generate(self, prompt, steps, stop_ids):
            self.calls.append((list(prompt), steps))
            toks = FakeTokenizer().encode("Answer: 42") + [USER] + FakeTokenizer().encode("Answer: 41")
            toks, stopped = _cut(toks[:steps], stop_ids)
            return GenOutput(toks, stopped, 0.0, [])

    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [EOS, USER]}))
    p = _suite(tmp_path)
    s = json.loads(p.read_text())
    s["cases"] = [{**s["cases"][0], "think": True, "max_tokens": 40}]
    p.write_text(json.dumps(s))
    cfg = eval_config_from_dict({"name": "t", "model": MODEL, "suite": str(p), "systems": [{"name": "ref", "kind": "bf16"}],
                                 "think_reply_reserve": 16, "teacher_forced_positions": 0})
    r = TurnRunner("ref", {}, vocab=1003)
    res = ev.run_eval(cfg, tokenizer=FakeTokenizer(), runner_factory=lambda sc: r, model_dir=tmp_path,
                      model_hash_value="h")
    rec = res["systems"]["ref"]["cases"][0]
    assert rec["stopped"] is True and rec["status"] == "PASSED" and len(r.calls) == 1
    assert rec["text"].endswith("Answer: 42")


def test_eval_records_runner_errors(tmp_path):
    class Broken(ScriptedRunner):
        def generate(self, prompt, steps, stop_ids):
            raise RuntimeError("boom")

    runners = {"ref": ScriptedRunner("ref", ANSWERS), "same": Broken("same", ANSWERS), "diff": Broken("diff", ANSWERS)}
    res = ev.run_eval(_eval_cfg(tmp_path), tokenizer=FakeTokenizer(), runner_factory=lambda sc: runners[sc.name],
                      model_hash_value="h")
    assert res["systems"]["same"]["summary"]["errors"] == 3
    assert all(a.get("error") for a in res["systems"]["same"]["agreement"])


def test_eval_deadline_skips_remaining_cases(tmp_path, monkeypatch):
    monkeypatch.setenv("SUC_EVAL_DEADLINE", "1")          # long past: every case is skipped, the run still finishes
    res, runners = _run(tmp_path)
    for s in res["systems"].values():
        assert [r["status"] for r in s["cases"]] == ["SKIPPED"] * 3
        assert s["summary"]["skipped"] == 3 and s["summary"]["graded"] == 0 and s["summary"]["errors"] == 0
    assert all(a.get("error") for a in res["systems"]["same"]["agreement"])
    assert res["deadline"]["skipped"] == {"ref": 3, "same": 3, "diff": 3}
    assert "skip" in ev.render_markdown(res)
    monkeypatch.setenv("SUC_EVAL_DEADLINE", "")
    res, _ = _run(tmp_path)
    assert "deadline" not in res and res["systems"]["ref"]["summary"]["skipped"] == 0


# ------------------------------------------------------------------------------------------------ scripted perf

def test_perf_driver_rows_stats_and_compare(tmp_path):
    cfg = perf_config_from_dict({"name": "p", "model": MODEL, "systems": [{"name": "a", "kind": "bf16"},
                                                                          {"name": "b", "kind": "bf16"}],
                                 "contexts": [16, 32], "batch_sizes": [1, 2], "gen_tokens": 5, "repeats": 2})
    runners = {"a": ScriptedRunner("a", {}), "b": ScriptedRunner("b", {})}
    res = pf.run_perf(cfg, tokens=list(range(200)), runner_factory=lambda sc: runners[sc.name], model_hash_value="h")
    ok = [r for r in res["rows"] if r["status"] == "ok"]
    assert len(ok) == 4 and sum(r["status"] == "unsupported" for r in res["rows"]) == 4
    r = ok[0]
    assert r["prefill_tok_s"]["median"] == pytest.approx(16 / 0.5) and r["ttft_ms"]["median"] == pytest.approx(500)
    assert r["decode_ms"]["median"] == 1.0 and r["decode_first_ms"]["median"] == 2.0
    assert r["decode_tok_s"]["median"] == pytest.approx(1000) and r["deterministic"]
    cmp = pf.compare(res, res)
    assert cmp and all(c["ratio"]["decode_tok_s"] == pytest.approx(1) for c in cmp)
    res["comparison"] = cmp
    res["config"]["compare_to"] = "old.json"
    md = pf.render_markdown(res)
    assert "| a | 16 | 1 |" in md and "unsupported" in md and "(1.00×)" in md and "## Comparison" in md
    w = pf.batch_windows(list(range(100)), 40, 4)
    assert [x[0] for x in w] == [0, 20, 40, 60] and all(len(x) == 40 for x in w)
    assert [x[0] for x in pf.batch_windows(list(range(100)), 20, 2)] == [0, 20]
    with pytest.raises(ValueError, match="need"):
        pf.run_perf(cfg, tokens=list(range(10)), runner_factory=lambda sc: runners[sc.name], model_hash_value="h")


def test_stats():
    s = pf.stats([1.0, 2.0, 3.0])
    assert s["median"] == 2 and s["min"] == 1 and s["max"] == 3 and s["n"] == 3 and s["cv"] == pytest.approx(0.5)
    assert pf.stats([]) is None


# ------------------------------------------------------------------------------------------------ mlx runner

def test_mlx_runner_greedy_score_and_perf():
    model = tiny_qwen3(seed=1, vocab=128)
    prompt = [3, 17, 5, 9, 33, 2]
    pipe = MlxRunner("p", model, decode="pipelined", prefill_step_size=4)
    sync = MlxRunner("s", model, decode="sync")
    a, b = pipe.generate(prompt, 12, set()), sync.generate(prompt, 12, set())
    assert a.tokens == b.tokens and len(a.tokens) == 12 and len(a.decode_ms) == 11
    # reference greedy: whole-sequence forward pass
    seq = list(prompt)
    for _ in range(12):
        seq.append(int(mx.argmax(model(mx.array([seq]))[0, -1]).item()))
    assert a.tokens == seq[len(prompt):]
    logits = pipe.score(prompt, a.tokens)
    assert logits.shape == (12, 128) and list(logits.argmax(-1)) == a.tokens
    stop = {a.tokens[3]}
    c = pipe.generate(prompt, 12, stop)
    assert c.stopped and c.tokens == a.tokens[:a.tokens.index(a.tokens[3])]
    r = pipe.perf([prompt, prompt[::-1]], 6)
    assert len(r["decode_ms"]) == 5 and r["peak_memory_bytes"] > 0 and r["kv_cache_bytes"] > 0


# ------------------------------------------------------------------------------------------------ C++ runner

def test_cpp_runner_matches_python_runtime(tmp_path):
    import test_cpp_generate as tcg

    if not tcg.GENERATE.exists():
        pytest.skip("suc1-generate not built")
    from suc.runtime.suc1_model import load_suc1_model

    f = tcg._tiny_file(tmp_path / "t.suc1", tie=True)
    cpp = CppRunner("cpp", f, tcg.BUILD, ("--projections", "separate"))
    py = MlxRunner("py", load_suc1_model(f, group=False)[0])
    prompt = [5, 77, 300, 12, 9, 41]
    g = cpp.generate(prompt, 10, set())
    assert len(g.tokens) == 10 and len(g.decode_ms) == 9 and g.prefill_s > 0
    lc, lp = cpp.score(prompt, g.tokens), py.score(prompt, g.tokens)
    assert lc.shape == lp.shape == (10, 512)
    assert list(lc.argmax(-1)) == g.tokens                      # its own greedy output, teacher-forced
    kl = ev.tf_metrics(ev._log_softmax(lp), lc, g.tokens)
    assert kl["top1_agreement"] >= 0.9 and kl["kl_mean"] < 1e-2
    r = cpp.perf([prompt], 6)
    assert len(r["decode_ms"]) == 5 and r["memory"]["mapped_file"] > 0
    assert not cpp.supports_batch(2)
    with pytest.raises(UnsupportedRun):
        cpp.perf([prompt, prompt], 6)


def test_cli_eval_requires_config_or_regrade():
    from suc.cli.main import main

    with pytest.raises(SystemExit):
        main(["eval"])


def test_ds4_case_importer_parses_c_initializers():
    import importlib.util

    spec = importlib.util.spec_from_file_location("imp", ROOT / "scripts/import_ds4_eval_cases.py")
    imp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imp)
    src = ('static const eval_case eval_cases[] = {\n    {\n        .source = "AIME2025",\n        .id = "a1",\n'
           '        .question = "Find $\\\\frac{1}{2}$ of \\"x\\".\\nThen \\u00e8",\n        .answer = "7",\n    },\n'
           '    {\n        .source = "SuperGPQA",\n        .id = "s1",\n        .question = "Q",\n'
           '        .choice[0] = "p",\n        .choice[1] = "q",\n        .answer = "B",\n    },\n};\n')
    with pytest.raises(ValueError, match="unknown C escape"):
        imp.parse_cases(src)
    cases = imp.parse_cases(src.replace("\\u00e8", "è"))
    assert cases[0]["question"] == 'Find $\\frac{1}{2}$ of "x".\nThen è' and cases[1]["choices"] == ["p", "q"]
    assert imp.to_suite_case(cases[1], 100)["kind"] == "choice" and imp.to_suite_case(cases[0], 100)["kind"] == "integer"
