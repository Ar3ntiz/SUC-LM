"""benchmarks/moe_routing_check.py on a random tiny Qwen3-MoE: traced forward, router top-k, stochastic BF16
rounding control, and the end-to-end comparison (float32: compressed == reconstructed up to float32 summation)."""
import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from tiny_models import tiny_qwen3_moe

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rc():
    spec = importlib.util.spec_from_file_location("moe_routing_check", ROOT / "benchmarks" / "moe_routing_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _neighbours_bf16(w: np.ndarray):
    """Largest BF16 ≤ |w| and smallest BF16 ≥ |w| (signed), via bit truncation of the float32 pattern."""
    u = w.view(np.uint32)
    lo = (u & np.uint32(0xFFFF0000)).view(np.float32)
    hi = ((u & np.uint32(0xFFFF0000)) + np.uint32(0x10000)).view(np.float32)
    exact = (u & np.uint32(0xFFFF)) == 0
    return lo, np.where(exact, lo, hi)


def test_stochastic_round_is_a_neighbour_and_unbiased(rc):
    w = np.random.default_rng(0).normal(size=(256, 256)).astype(np.float32)
    sr = np.asarray(rc.stochastic_round_bf16(mx.array(w), mx.random.key(1)).astype(mx.float32))
    lo, hi = _neighbours_bf16(w)
    assert np.all((sr == lo) | (sr == hi))
    # differs from nearest with probability E[min(f, 1 - f)] = 1/4 (f = dropped fraction of an ulp)
    assert 0.2 < np.mean(sr != np.asarray(mx.array(w).astype(mx.bfloat16).astype(mx.float32))) < 0.3
    # unbiased: mean rounding error ≪ the error of always rounding the magnitude down
    assert abs(np.mean(sr - w)) < 0.05 * np.mean(np.abs(lo - w))
    near = np.asarray(rc.stochastic_round_bf16(mx.array(w), mx.random.key(1), 0.0).astype(mx.float32))
    assert np.array_equal(near, np.asarray(mx.array(w).astype(mx.bfloat16).astype(mx.float32)))


def test_router_topk_matches_block_selection(rc):
    logits = mx.random.normal((3, 5, 16), key=mx.random.key(2)).astype(mx.bfloat16)
    k = 4
    got = np.asarray(rc.router_topk(logits, k))
    gates = mx.softmax(logits, axis=-1, precise=True)
    ref = np.asarray(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
    assert np.array_equal(np.sort(got, -1), np.sort(ref, -1))
    g = np.asarray(gates.astype(mx.float32))
    ranked = np.take_along_axis(g, got, -1)
    assert np.all(np.diff(ranked, axis=-1) <= 0)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_traced_forward_is_bit_exact(rc, dtype):
    model = tiny_qwen3_moe(seed=3, experts=8, top_k=2, layers=3, mlp_only_layers=(1,))
    model.set_dtype(dtype)
    toks = mx.array(np.random.default_rng(0).integers(0, 128, size=(2, 16)).astype(np.int32))
    tr = rc.traced_forward(model, toks)
    assert np.array_equal(tr["logits"], np.asarray(model(toks).astype(mx.float32)))
    assert sorted(tr["topk"]) == [0, 2] and tr["topk"][0].shape == (2, 16, 2)
    tf = rc.traced_forward(model, toks, teacher=tr["inputs"])    # teacher = own inputs: identical run
    assert all(np.array_equal(a, b) for a, b in zip(tf["outputs"], tr["outputs"]))
    res = rc.compare(tr, tf, model, local=True)
    assert res["summary"]["set_agreement_mean"] == 1.0 and res["summary"]["rel_err_out_mean"] == 0.0


def test_end_to_end_float32_has_no_routing_flips(rc, tmp_path):
    path = rc.main(["--model", "tiny-moe", "--tiny-layers", "2", "--dtype", "float32", "--kmeans-iters", "2",
                    "--check-seqs", "2", "--check-len", "32", "--controls", "repeat", "--quiet",
                    "--out-dir", str(tmp_path)])
    d = json.loads(Path(path).read_text())
    assert d["traced_forward_bit_exact"] and d["metadata"]["model"] == "tiny-moe"
    rep = d["comparisons"]["repeat"]["free"]
    assert rep["final_bit_identical"] and rep["summary"]["flipped_token_layers"] == 0
    comp = d["comparisons"]["compressed"]
    assert comp["free"]["summary"]["flipped_token_layers"] == 0
    assert comp["teacher_forced"]["summary"]["flipped_token_layers"] == 0
    assert comp["free"]["final"]["top1_agreement"] == 1.0 and abs(comp["free"]["final"]["kl_mean"]) < 1e-6
    assert comp["free"]["summary"]["rel_err_out_last"] < 1e-5
