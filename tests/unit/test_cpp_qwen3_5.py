"""Cross-check of the C++/Metal dense qwen3_5 graph (cpp/src/metal/qwen4.cpp with Qwen4Config::dense, `suc1-generate`
on a qwen3_5 file) with the Python/MLX SUC1 runtime of the same file (load_suc1_model(group=False,
affine_modules=False)).

Tiny random qwen3_5 checkpoints (tests/unit/tiny_models.py) compressed with `suc compress` (VQ K=16 G=2 one-sided
rotation + column residual for the bulk, affine 8-bit attention, affine 4-bit embedding / LM head). Per position
(token-by-token decode, teacher-forced on the C++ tokens): logits and the hidden state after the embedding and every
block are compared; the free-running greedy tokens must be identical. Two shapes: tiny (hidden 64, head_dim 16: MLX's
unfused attention fallback) and one at Qwen3.8-27B widths where kernels branch (hidden 5120 > 4096: looped RMSNorm,
head_dim 256: sdpa_vector, linear key dim 128).

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

from suc.runtime.suc1_model import greedy, kl_rows, load_suc1_model, run_tokens
from tiny_models import write_tiny_qwen3_5

BUILD = Path(os.environ.get("SUC_CPP_BUILD", Path.home() / "suc-local" / "cpp-build"))
GENERATE = BUILD / "suc1-generate"

pytestmark = pytest.mark.skipif(not GENERATE.exists(), reason=f"suc1-generate not built in {BUILD} (run cpp/build.sh)")

VQ = {"kind": "vq", "k": 16, "block_size": 2, "rotate_out": False, "kmeans_iters": 3, "residual_fraction": 0.02,
      "residual_structure": "columns"}
VQ_NAME = "vq_k16_g2_rot1s+res0.02_activation_cols"
A8 = {"kind": "affine", "bits": 8, "group_size": 64}
WIDE = {"hidden_size": 5120, "intermediate_size": 17 * 32, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 256, "linear_num_key_heads": 2, "linear_num_value_heads": 6, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128}
PROMPT = [5, 6, 7, 8]
STEPS = 24


def _compress(tmp: Path, **over) -> Path:
    import suc.pipeline.compress as comp
    from test_calibrate_run import FakeTokenizer, _docs

    mx.random.seed(0)
    tokens = mx.random.randint(2, 128, (2048,), dtype=mx.int32)
    over = {"intermediate_size": 17 * 32, **over}    # down_proj input rotation with the m = 17 factor (27B: 17 * 2^10)
    model_dir = write_tiny_qwen3_5(tmp / "model", seed=2, dtype=mx.bfloat16, **over)
    cfg = {"name": "c_q35cpp", "seed": 7, "output_dir": str(tmp / "results"), "run_dir": str(tmp / "run"),
           "model": {"id": "tiny/qwen3_5", "local_path": str(model_dir)},
           "calibration": {"seq_len": 32, "n_calibration": 16, "n_validation": 8, "n_ood": 4, "micro_batch": 4},
           "plan": {"mode": "uniform", "codec": VQ_NAME, "candidates": [VQ, A8],
                    "role_codecs": {"affine_b8_g64": ["attn_q", "attn_k", "attn_v", "attn_o"]},
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
def q35_file(request, tmp_path_factory):
    return _compress(tmp_path_factory.mktemp(f"q35cpp_{request.param}"), **(WIDE if request.param == "wide" else {}))


def _cpp(path: Path, tmp: Path, *args: str, tokens=PROMPT, steps=STEPS) -> tuple[list[int], dict]:
    j = tmp / "out.json"
    r = subprocess.run([str(GENERATE), str(path), "--tokens", ",".join(map(str, tokens)), "--steps", str(steps),
                        "--json", str(j), *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    gen = [int(x) for x in r.stdout.split("generated:")[1].split()[0].split(",")]
    return gen, json.loads(j.read_text())


def test_graph_matches_python_runtime(q35_file, tmp_path):
    gen, rec = _cpp(q35_file, tmp_path, "--logits-out", str(tmp_path / "l.npy"), "--hidden-out", str(tmp_path / "h.npy"))
    assert rec["model_type"] == "qwen3_5" and rec["suc_projections"] > 0
    lc, hc = np.load(tmp_path / "l.npy"), np.load(tmp_path / "h.npy")
    seq = PROMPT + gen[:-1]
    assert lc.shape[0] == hc.shape[0] == len(seq)
    model, _, _ = load_suc1_model(q35_file, group=False, affine_modules=False)
    lp, hp = run_tokens(model, seq)
    kl = kl_rows(lp, lc)
    err_l, err_h = np.abs(lp - lc), np.abs(hp - hc)
    print(f"\npositions {len(seq)}: logits max {err_l.max():.3g} mean {err_l.mean():.3g} bit-identical "
          f"{(lp == lc).mean():.4f}; KL max {kl.max():.3g} mean {kl.mean():.3g}; hidden max {err_h.max():.3g} "
          f"bit-identical {(hp == hc).mean():.4f}")
    # Bit-identical on the M3. On the M1 Ultra the float32 delta-rule recurrence drifts by ulps after ~10 steps (the
    # qwen4_exp graph shows the same, tests/unit/test_cpp_qwen4.py); with the tiny random weights (PPL ~200) that can
    # flip a near-tied argmax, so argmax / greedy agreement is required only where the top-2 margin is not a near tie.
    assert kl.max() < 1e-3
    top2 = np.sort(lp, axis=-1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0]) > 0.25
    assert (lp.argmax(-1) == lc.argmax(-1))[clear].all()
    assert err_h.max() <= 0.05 * max(1.0, float(np.abs(hp).max()))
    py = greedy(model, PROMPT, STEPS)
    first = next((i for i, (x, y) in enumerate(zip(py, gen)) if x != y), None)
    assert first is None or not clear[len(PROMPT) - 1 + first], (first, py, gen)


def test_chunked_prefill_and_sync_steps_agree(q35_file, tmp_path):
    prompt = [3, 9, 1, 4, 4, 17, 2, 1, 1, 30, 5, 6, 7, 8, 11, 12]
    a, rec = _cpp(q35_file, tmp_path, "--chunk", "5", tokens=prompt, steps=12)
    b, _ = _cpp(q35_file, tmp_path, "--prefill", "token", tokens=prompt, steps=12)
    assert rec["chunk"] == 5 and a == b


def test_batched_prefill_bit_identical(q35_file, tmp_path):
    """Row-exact batched (layer-major) prefill (--prefill exact) = token-by-token prefill: the last prompt token's logits
    and streams (sha256) and the greedy continuation, for chunks that split the prompt (7) and hold it whole (64)."""
    prompt = [int(x) for x in np.random.default_rng(7).integers(2, 127, size=45)]
    ref, rr = _cpp(q35_file, tmp_path, "--prefill", "token", tokens=prompt, steps=6)
    assert rr["prefill"] == "token"
    for chunk in ("7", "64"):
        gen, rec = _cpp(q35_file, tmp_path, "--prefill", "exact", "--chunk", chunk, tokens=prompt, steps=6)
        assert rec["prefill"] == "batch" and rec["chunk"] == int(chunk)
        assert rec["prefill_logits_sha256"] == rr["prefill_logits_sha256"], chunk
        assert rec["prefill_streams_sha256"] == rr["prefill_streams_sha256"], chunk
        assert gen == ref


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


def test_fast_batched_prefill_close_to_token(q35_file, tmp_path, monkeypatch):
    prompt = [int(x) for x in np.random.default_rng(7).integers(2, 127, size=45)]
    _fast_vs_token(q35_file, tmp_path, monkeypatch, prompt, [("7", None), ("48", None)])
