"""Cross-check of the C++/Metal qwen4_exp graph (cpp/src/metal/qwen4.cpp, `suc1-generate` on a qwen4_exp file) with the
Python/MLX SUC1 runtime of the same file (suc.runtime.suc1_model.load_suc1_model(group=False, affine_modules=False):
one SucLinear per projection, decoded embedding / LM head = the C++ execution plan).

Tiny random qwen4_exp checkpoints (tests/unit/tiny_models.py) compressed with `suc compress` (VQ K=16 G=2, one-sided
input rotation, column residuals, expert-stack codebooks, affine 4-bit embedding / LM head), then optionally
`suc repack`ed (row-affine 4-bit n-gram tables). Per position (token-by-token decode, teacher-forced on the C++ token
sequence, prompt + generated tokens well into the QSA sparse regime): logits and every stream after the embedding and
after every block are compared (max / mean abs error, KL, bit-identical fraction), the free-running greedy tokens must
be identical. Two shapes: the tiny one (hidden 64, head_dim 16: MLX's unfused attention fallback) and one at
Qwen3.8-Flash-Next widths where it matters (hidden 2560 = 10240-wide streams, head_dim 256 with sdpa_vector and the
QSA bool-mask branch, linear key dim 128, 16 n-gram heads of 160 = the real row width).

Skipped unless suc1-generate is built (cpp/build.sh) into $SUC_CPP_BUILD (default ~/suc-local/cpp-build).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import yaml

from suc.io.repack import repack_suc1
from suc.runtime.suc1_model import greedy, kl_rows, load_suc1_model, run_tokens
from tiny_models import write_tiny_qwen4_exp

BUILD = Path(os.environ.get("SUC_CPP_BUILD", Path.home() / "suc-local" / "cpp-build"))
GENERATE = BUILD / "suc1-generate"
HUB = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
QWEN38 = HUB / "models--Qwen--Qwen3.8-Flash-Next" / "snapshots" / "de4b8e4d43b917e7706784d8bb445c9af86a3540"

pytestmark = pytest.mark.skipif(not GENERATE.exists(), reason=f"suc1-generate not built in {BUILD} (run cpp/build.sh)")

XS = {"kind": "vq", "k": 16, "block_size": 2, "rotate_out": False, "kmeans_iters": 3, "residual_fraction": 0.02,
      "residual_structure": "columns", "codebook_scope": "expert_stack"}
XS_NAME = "vq_k16_g2_rot1s_xs+res0.02_activation_cols"
# Qwen3.8-Flash-Next widths where the kernels branch (fewer heads / experts / layers to stay small)
WIDE = {"hidden_size": 2560, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 256,
        "linear_num_key_heads": 2, "linear_num_value_heads": 6, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "ple_embed_dim": 2560, "heads_per_ngram": 8, "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64,
        "hc_lowrank": 32, "indexer_n_heads": 2, "indexer_head_dim": 128, "num_experts": 4, "num_experts_per_tok": 2}
PROMPT = [5, 6, 7, 8]
STEPS = 30          # 33 positions: the QSA selection is sparse from 12 visible tokens on (budget 8, ratio 4)


A8 = {"kind": "affine", "bits": 8, "group_size": 32}   # 32: the tiny shared-expert widths
# the projections the real Qwen3.8-Flash-Next SUC1 files store as affine 8-bit codes
A8_ROLES = ["attn_q", "attn_k", "attn_v", "attn_o", "linattn_qkv", "linattn_z", "linattn_out", "shared_expert_gate",
            "shared_expert_up", "shared_expert_down", "ple_key", "ple_value"]


def _compress(tmp: Path, affine: bool = False, **over) -> Path:
    """SUC1 file of a tiny qwen4_exp (raw n-gram tables), as tests/unit/test_qwen4_exp_runtime.py builds it.
    affine: the attention / linear-attention / shared-expert / PLE projections as affine 8-bit codes (real files)."""
    import suc.pipeline.compress as comp
    from test_calibrate_run import FakeTokenizer, _docs

    mx.random.seed(0)
    tokens = mx.random.randint(2, 128, (2048,), dtype=mx.int32)
    model_dir = write_tiny_qwen4_exp(tmp / "model", seed=2, dtype=mx.bfloat16, **over)
    cfg = {"name": "c_q4cpp", "seed": 7, "output_dir": str(tmp / "results"), "run_dir": str(tmp / "run"),
           "model": {"id": "tiny/qwen4_exp", "local_path": str(model_dir)},
           "calibration": {"seq_len": 32, "n_calibration": 16, "n_validation": 8, "n_ood": 4, "micro_batch": 4},
           "plan": {"mode": "uniform", "codec": XS_NAME, "allocation_unit": "expert_stack",
                    "candidates": [XS, A8] if affine else [XS],
                    **({"role_codecs": {"affine_b8_g32": A8_ROLES}} if affine else {}),
                    "embedding_codec": {"kind": "affine", "bits": 4}},
           "eval": {"seq_len": 64, "max_tokens": 1024, "kl_sequences": 4, "chunk": 2},
           "finetune": None, "rotation_groups": True, "verify_tensors": 8}
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    docs, ood = _docs()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(comp, "load_wikitext2", lambda split: ("unused", "wikihash"))
        mp.setattr(comp, "tokenize", lambda tok, text: tokens)
        r = json.loads(comp.run_compress(p, tokenizer=FakeTokenizer(), docs=docs, ood_docs=ood).read_text())
    return Path(r["container"]["path"])


@pytest.fixture(scope="module", params=["tiny", "wide"])
def q4_files(request, tmp_path_factory):
    """{"raw": SUC1 path, "row4": the same file after `suc repack`} for the tiny and the wide shape."""
    tmp = tmp_path_factory.mktemp(f"q4cpp_{request.param}")
    raw = _compress(tmp, **(WIDE if request.param == "wide" else {}))
    row4 = tmp / "row4.suc1"
    rec = repack_suc1(raw, row4)
    assert rec["tables"]["converted"] == 2
    aff = _compress(tmp / "affine", affine=True, **(WIDE if request.param == "wide" else {}))
    return {"raw": raw, "row4": row4, "affine": aff}


def q4_kind(path: Path) -> str:
    """"tiny" / "wide" fixture of a SUC1 path (tmp dir q4cpp_<kind>N)."""
    return "wide" if any(part.startswith("q4cpp_wide") for part in path.parts) else "tiny"


def _cpp(path: Path, tmp: Path, *args: str, tokens=PROMPT, steps=STEPS) -> tuple[list[int], dict]:
    j = tmp / "out.json"
    r = subprocess.run([str(GENERATE), str(path), "--tokens", ",".join(map(str, tokens)), "--steps", str(steps),
                        "--json", str(j), *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    gen = [int(x) for x in r.stdout.split("generated:")[1].split()[0].split(",")]
    return gen, json.loads(j.read_text())


@pytest.mark.parametrize("tables", ["raw", "row4"])
def test_graph_matches_python_runtime(q4_files, tables, tmp_path):
    path = q4_files[tables]
    gen, rec = _cpp(path, tmp_path, "--logits-out", str(tmp_path / "l.npy"), "--hidden-out", str(tmp_path / "h.npy"))
    assert rec["model_type"] == "qwen4_exp" and rec["ngram_quantized"] == (tables == "row4")
    m = rec["memory"]
    assert m["ngram_tables"] > 0
    if m["ngram_tables"] > (1 << 20):           # the GPU views leave the tables out (all but shared edge pages)
        assert m["excluded_from_gpu_views"] >= 0.5 * m["ngram_tables"]
        assert m["mapped_file"] <= m["file_size"] - m["excluded_from_gpu_views"] + (1 << 16)
    lc, hc = np.load(tmp_path / "l.npy"), np.load(tmp_path / "h.npy")
    seq = PROMPT + gen[:-1]
    assert lc.shape[0] == hc.shape[0] == len(seq)
    model, _, _ = load_suc1_model(path, group=False, affine_modules=False)
    lp, hp = run_tokens(model, seq)
    hp = hp[:, :, : hc.shape[2]]                     # the streams (Python also carries the n-gram rows behind them)
    kl = kl_rows(lp, lc)
    err_l, err_h = np.abs(lp - lc), np.abs(hp - hc)
    same_l, same_h = (lp == lc).mean(), (hp == hc).mean()
    print(f"\n[{tables}] positions {len(seq)}: logits max {err_l.max():.3g} mean {err_l.mean():.3g} bit-identical "
          f"{same_l:.4f}; KL max {kl.max():.3g} mean {kl.mean():.3g}; streams max {err_h.max():.3g} mean "
          f"{err_h.mean():.3g} bit-identical {same_h:.4f}")
    # The wide shape is not bit-identical (different bf16 kernels / reduction orders): a 1-ulp difference can flip a
    # NEAR-TIED discrete choice downstream (measured: the wide raw file diverges at one position whose QSA indexer
    # 2nd/3rd block scores are 0.696 / 0.684; wide raw or row4 failed before and after the l2norm-eps fix).
    # Tolerated at <= 1 position (KL < 1e-2); every other position must agree to KL < 1e-4.
    tol_h = 0.05 * max(1.0, float(np.abs(hp).max()))
    bad = (kl >= 1e-4) | (err_l.max(-1) > 0.0625) | (err_h.reshape(err_h.shape[0], -1).max(-1) > tol_h)
    assert bad.sum() <= (1 if q4_kind(path) == "wide" else 0), np.flatnonzero(bad)
    assert kl.max() < 1e-2
    assert (lp.argmax(-1) == lc.argmax(-1)).all()
    assert greedy(model, PROMPT, STEPS) == gen


def test_chunked_prefill_and_sync_steps_agree(q4_files, tmp_path):
    """The prompt as decode steps several per command buffer (--chunk) = one per command buffer = Python greedy."""
    path = q4_files["raw"]
    prompt = [3, 9, 1, 4, 4, 17, 2, 1, 1, 30, 5, 6, 7, 8, 11, 12]   # EOS (1) inside: the n-gram context resets
    a, rec = _cpp(path, tmp_path, "--chunk", "5", tokens=prompt, steps=12)
    b, _ = _cpp(path, tmp_path, "--prefill", "token", tokens=prompt, steps=12)
    assert rec["chunk"] == 5 and a == b
    model, _, _ = load_suc1_model(path, group=False, affine_modules=False)
    assert greedy(model, prompt, 12) == a


LONG_PROMPT = [int(x) for x in np.random.default_rng(7).integers(2, 127, size=45)] + [1, 3, 9, 4]   # EOS inside


@pytest.mark.parametrize("variant", ["raw", "row4", "affine"])
def test_batched_prefill_bit_identical(q4_files, variant, tmp_path, monkeypatch):
    """Row-exact batched (layer-major) prefill (--prefill exact) = token-by-token prefill: the last prompt token's logits and streams (sha256) and
    the greedy continuation, for several chunk sizes and forced expert-tile heights (padding rows); streamed experts
    with few slots (several fetch groups per chunk)."""
    path = q4_files[variant]
    ref, rr = _cpp(path, tmp_path, "--prefill", "token", tokens=LONG_PROMPT, steps=6)
    assert rr["prefill"] == "token"
    for chunk, bt in [("7", None), ("64", None), ("64", "16"), ("5", "2")]:
        if bt:
            monkeypatch.setenv("SUC_TILE_BT", bt)
        else:
            monkeypatch.delenv("SUC_TILE_BT", raising=False)
        gen, rec = _cpp(path, tmp_path, "--prefill", "exact", "--chunk", chunk, tokens=LONG_PROMPT, steps=6)
        assert rec["prefill"] == "batch" and rec["chunk"] == int(chunk)
        assert rec["prefill_logits_sha256"] == rr["prefill_logits_sha256"], (chunk, bt)
        assert rec["prefill_streams_sha256"] == rr["prefill_streams_sha256"], (chunk, bt)
        assert gen == ref
    monkeypatch.delenv("SUC_TILE_BT", raising=False)
    _, rec = _cpp(path, tmp_path, "--expert-budget", str(1 << 34), steps=2)
    st = rec["expert_streaming"]
    budget = st["layers"] * 3 * st["slot_bytes_max"]
    gen, rec = _cpp(path, tmp_path, "--prefill", "exact", "--chunk", "16", "--expert-budget", str(budget),
                    tokens=LONG_PROMPT, steps=6)
    assert rec["prefill_logits_sha256"] == rr["prefill_logits_sha256"]
    assert rec["prefill_streams_sha256"] == rr["prefill_streams_sha256"]
    assert gen == ref and rec["prefill_chunk_fetches"] > st["layers"] * 3


def _row_kl(ref: np.ndarray, got: np.ndarray) -> np.ndarray:
    def ls(x):
        x = x.astype(np.float64)
        m = x.max(-1, keepdims=True)
        return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
    a, b = ls(ref), ls(got)
    return (np.exp(a) * (a - b)).sum(-1)


def _fast_vs_token(path, tmp_path, monkeypatch, prompt, runs, extra=()):
    """Default batched prefill (simdgroup-matrix GEMMs, MMA expert tiles; not bit-identical) vs token prefill: logits
    of every prompt position (--prefill batch --logits-out) and of the teacher-forced continuation."""
    ref, _ = _cpp(path, tmp_path, "--prefill", "token", "--logits-out", str(tmp_path / "t.npy"), *extra, tokens=prompt,
                  steps=6)
    lt = np.load(tmp_path / "t.npy")
    for chunk, bt in runs:
        if bt:
            monkeypatch.setenv("SUC_TILE_BT", bt)
        else:
            monkeypatch.delenv("SUC_TILE_BT", raising=False)
        _, rec = _cpp(path, tmp_path, "--prefill", "batch", "--chunk", chunk, "--logits-out", str(tmp_path / "b.npy"),
                      "--force", ",".join(map(str, ref)), *extra, tokens=prompt, steps=6)
        lb = np.load(tmp_path / "b.npy")
        assert rec["prefill"] == "batch" and lb.shape == lt.shape, (lb.shape, lt.shape)
        kl = _row_kl(lt, lb)
        top1 = (lt.argmax(-1) == lb.argmax(-1)).mean()
        # tiny random models: bf16 roundings of GEMM vs gemv accumulation orders, amplified by near-tie routing;
        # real-model sizes: scripts/prefill_batch_vs_token.py (report §1c)
        assert kl.mean() < 2e-2 and kl.max() < 0.3, (chunk, bt, kl.mean(), kl.max(), int(kl.argmax()))
        assert top1 >= 0.85, (chunk, bt, top1)
    monkeypatch.delenv("SUC_TILE_BT", raising=False)


@pytest.mark.parametrize("variant", ["raw", "row4", "affine"])
def test_fast_batched_prefill_close_to_token(q4_files, variant, tmp_path, monkeypatch):
    path = q4_files[variant]
    _fast_vs_token(path, tmp_path, monkeypatch, LONG_PROMPT, [("7", None), ("48", None), ("48", "16"), ("16", "8")])
    _, rec = _cpp(path, tmp_path, "--expert-budget", str(1 << 34), steps=2)
    st = rec["expert_streaming"]
    budget = st["layers"] * 3 * st["slot_bytes_max"]
    _fast_vs_token(path, tmp_path, monkeypatch, LONG_PROMPT, [("16", None)], ("--expert-budget", str(budget)))


@pytest.mark.parametrize("variant", ["raw", "affine"])
def test_fast_attention_rows_close_to_token(q4_files, variant, tmp_path, monkeypatch):
    """Row-batched full attention of the fast prefill (SUC_FAST_OFF=pt: projections / expert tiles row-exact, only
    the attention core batched): q / k norms and RoPE over the chunk rows, block keys, QSA scores and selection of
    every row past the budget (the tiny file selects from 12 tokens on), one sdpa_vector launch over (heads, rows) =
    the token-by-token prefill to rounding level (single-pass kernel; decode switches to two passes on long contexts);
    the row-batched hyper-connection / inject / routing launches are bit-identical to their per-row form."""
    path = q4_files[variant]
    ref, _ = _cpp(path, tmp_path, "--prefill", "token", "--logits-out", str(tmp_path / "t.npy"), tokens=LONG_PROMPT,
                  steps=2)
    lt = np.load(tmp_path / "t.npy")
    monkeypatch.setenv("SUC_FAST_OFF", "pt")
    for chunk in ("16", "48"):
        _cpp(path, tmp_path, "--prefill", "batch", "--chunk", chunk, "--logits-out", str(tmp_path / "b.npy"),
             "--force", ",".join(map(str, ref)), tokens=LONG_PROMPT, steps=2)
        lb = np.load(tmp_path / "b.npy")
        kl = _row_kl(lt, lb)
        # rounding level (device-dependent key grouping vs the decode sdpa kernels; near-tie routing can amplify it
        # at single positions)
        assert kl.mean() < 5e-3 and kl.max() < 0.1, (chunk, kl.mean(), kl.max(), int(kl.argmax()))
        assert (lt.argmax(-1) == lb.argmax(-1)).mean() >= 0.95, chunk
        # the row-batched hyper-connection / inject / routing launches = their per-row launches bit for bit
        monkeypatch.setenv("SUC_FAST_OFF", "pth")
        _cpp(path, tmp_path, "--prefill", "batch", "--chunk", chunk, "--logits-out", str(tmp_path / "h.npy"),
             "--force", ",".join(map(str, ref)), tokens=LONG_PROMPT, steps=2)
        assert (np.load(tmp_path / "h.npy") == lb).all(), chunk
        monkeypatch.setenv("SUC_FAST_OFF", "pt")


def test_teacher_forcing_and_vocab_check(q4_files, tmp_path):
    path = q4_files["row4"]
    gen, rec = _cpp(path, tmp_path, "--force", "1,2,3,4,5", steps=5)
    assert rec["teacher_forced"] and len(gen) == 5
    r = subprocess.run([str(GENERATE), str(path), "--tokens", "5,999"], capture_output=True, text=True)
    assert r.returncode != 0 and "vocab" in r.stderr


@pytest.mark.skipif(not (QWEN38 / "chat_template.jinja").exists(), reason="Qwen3.8-Flash-Next tokenizer not cached")
def test_qwen38_chat_template(q4_files):
    """--chat on a qwen4_exp file renders the Qwen3.8-Flash-Next chat_template.jinja (ids = HF tokenizers on the
    jinja2 rendering, transformers' environment settings)."""
    jinja2 = pytest.importorskip("jinja2")
    tk = pytest.importorskip("tokenizers")
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(m):
        raise ValueError(m)

    env.globals["raise_exception"] = raise_exception
    tpl = env.from_string((QWEN38 / "chat_template.jinja").read_text())
    tok = tk.Tokenizer.from_file(str(QWEN38 / "tokenizer.json"))
    cases = [("Ciao, come stai?", None, True, None), ("  Hello world \n", "You are helpful. ", True, "low"),
             ("Hi", None, False, None), ("Hi", "Sys", False, None), ("x", None, True, "medium")]
    for prompt, system, think, effort in cases:
        msgs = ([{"role": "system", "content": system}] if system is not None else []) + [{"role": "user", "content": prompt}]
        kw = {"add_generation_prompt": True, **({} if think else {"enable_thinking": False}),
              **({"reasoning_effort": effort} if effort else {})}
        ref = tok.encode(tpl.render(messages=msgs, **kw), add_special_tokens=False).ids
        args = [str(GENERATE), str(q4_files["raw"]), "--prompt", prompt, "--chat", "--tokenizer",
                str(QWEN38 / "tokenizer.json"), "--print-prompt", "--quiet"]
        args += (["--system", system] if system is not None else []) + ([] if think else ["--no-think"])
        args += ["--reasoning-effort", effort] if effort else []
        r = subprocess.run(args, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert [int(x) for x in r.stdout.split("prompt:")[1].split()[0].split(",")] == ref
