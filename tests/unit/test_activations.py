import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_moe
from mlx_lm.models.switch_layers import SwitchLinear

from suc.analysis.activations import InputMoments, Recorder
from suc.models.adapters import get_adapter
from tests.unit.tiny_models import tiny_qwen3

TOL = dict(rtol=1e-5, atol=1e-5)


def _np_stats(x: np.ndarray) -> dict:
    x = x.astype(np.float64)
    return {"mean": x.mean(0), "var": x.var(0), "max_abs": np.abs(x).max(0), "m2": (x * x).mean(0),
            "full": x.T @ x / len(x)}


def _check(m: InputMoments, ref: dict, full: bool = True):
    np.testing.assert_allclose(np.array(m.mean()), ref["mean"], **TOL)
    np.testing.assert_allclose(np.array(m.var()), ref["var"], **TOL)
    np.testing.assert_allclose(np.array(m.max_abs()), ref["max_abs"], **TOL)
    np.testing.assert_allclose(np.array(m.second_moment_diag()), ref["m2"], **TOL)
    if full:
        np.testing.assert_allclose(np.array(m.second_moment()), ref["full"], **TOL)


def _data(seed=0, n=300, d=12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, d)) * rng.uniform(0.1, 3, size=d) + rng.normal(size=d)).astype(np.float32)


def test_moments_match_numpy_over_updates_and_merge():
    x = _data()
    m = InputMoments(12, full_second_moment=True)
    for chunk in (x[:7], x[7:100], x[100:]):
        m.update(mx.array(chunk.reshape(-1, 1, 12)))  # leading dims are flattened
    assert m.count == 300
    _check(m, _np_stats(x))

    a, b = InputMoments(12, True), InputMoments(12, True)
    a.update(mx.array(x[:50]))
    b.update(mx.array(x[50:]))
    a.merge(b)
    assert a.count == 300
    _check(a, _np_stats(x))
    empty = InputMoments(12, True)
    empty.merge(a)
    _check(empty, _np_stats(x))


def test_moments_without_full_and_bf16_input():
    x = _data(1)
    m = InputMoments(12)
    m.update(mx.array(x).astype(mx.bfloat16))
    assert m.second_moment() is None
    xb = np.array(mx.array(x).astype(mx.bfloat16).astype(mx.float32))
    _check(m, _np_stats(xb), full=False)
    assert m.mean().dtype == mx.float32


def test_summary_and_array_roundtrip():
    x = _data(2)
    x[:, 3] *= 50
    m = InputMoments(12, True)
    m.update(mx.array(x))
    s = m.summary(top_k=3)
    json.dumps(s)
    assert s["count"] == 300 and s["dim"] == 12
    assert s["top_channels"][0] == 3 and len(s["top_channels"]) == 3
    assert s["outlier_channel_ratio"] > 100
    assert set(s["var"]) == {"p50", "p95", "p99", "max"} and set(s["max_abs"]) == set(s["second_moment_diag"])
    arrs = m.to_arrays("p")
    assert set(arrs) == {"p.mean", "p.m2diag", "p.max_abs", "p.m2full", "p.count"}
    assert arrs["p.count"].dtype == mx.int64 and arrs["p.count"].ndim == 0
    r = InputMoments.from_arrays(arrs, "p")
    assert r.count == 300
    _check(r, _np_stats(x))
    assert "p.m2full" not in InputMoments(4).to_arrays("p")


def _expected_dense_names(model, tie=True) -> set[str]:
    cfg = {"model_type": "qwen3", "num_hidden_layers": 2, "tie_word_embeddings": tie}
    names = [k for k, _ in tree_flatten(model.parameters())]
    return {s.name[:-len(".weight")] for s in get_adapter(cfg).linears(names) if s.role not in ("embed", "lm_head")}


def test_dense_recording_is_transparent_and_complete():
    model = tiny_qwen3(seed=0)
    tokens = mx.array(np.random.default_rng(0).integers(0, 128, size=(2, 9)))
    ref = model(tokens)
    rec = Recorder(full_second_moment=["model.layers.0.mlp.down_proj"])
    names = rec.instrument(model, "")
    assert set(names) == _expected_dense_names(model) == set(rec.moments)
    out = model(tokens)
    assert mx.array_equal(out, ref)
    model(tokens[:, :4])
    for n in names:
        assert rec.moments[n].count == 2 * 9 + 2 * 4
    # gate/up and q/k/v share one input -> one shared object
    L = "model.layers.1"
    assert rec.moments[f"{L}.mlp.gate_proj"] is rec.moments[f"{L}.mlp.up_proj"]
    assert rec.moments[f"{L}.self_attn.q_proj"] is rec.moments[f"{L}.self_attn.v_proj"]
    assert rec.moments[f"{L}.mlp.down_proj"] is not rec.moments[f"{L}.mlp.gate_proj"]
    assert rec.moments["model.layers.0.mlp.down_proj"].second_moment() is not None
    assert rec.moments[f"{L}.mlp.down_proj"].second_moment() is None
    assert rec.moments[f"{L}.mlp.down_proj"].dim == 128
    rec.remove(model)
    assert all(type(m) is nn.Linear for _, m in model.named_modules() if isinstance(m, nn.Linear))
    assert not any(hasattr(m, "_suc_hook") for _, m in model.named_modules())
    before = rec.moments[f"{L}.mlp.down_proj"].count
    assert mx.array_equal(model(tokens), ref)
    assert rec.moments[f"{L}.mlp.down_proj"].count == before


def test_dense_moments_match_captured_inputs():
    model = tiny_qwen3(seed=1)
    tokens = mx.array(np.random.default_rng(1).integers(0, 128, size=(3, 5)))
    layer = model.model.layers[0]
    mlp = layer.mlp
    h = model.model.embed_tokens(tokens)
    h = h + layer.self_attn(layer.input_layernorm(h), mask="causal")
    x = np.array(layer.post_attention_layernorm(h).astype(mx.float32)).reshape(-1, 64)
    g = x @ np.array(mlp.gate_proj.weight).T
    down_in = g / (1 + np.exp(-g)) * (x @ np.array(mlp.up_proj.weight).T)
    rec = Recorder(full_second_moment=["model.layers.0.mlp.down_proj"])
    rec.instrument(layer, "model.layers.0")  # instrument one block only
    model(tokens)
    _check(rec.moments["model.layers.0.mlp.gate_proj"], _np_stats(x), full=False)
    _check(rec.moments["model.layers.0.mlp.down_proj"], _np_stats(down_in))
    rec.remove(layer)


def _tiny_moe_args() -> qwen3_moe.ModelArgs:
    return qwen3_moe.ModelArgs(
        model_type="qwen3_moe", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
        mlp_only_layers=[0], moe_intermediate_size=32, rms_norm_eps=1e-6, vocab_size=128,
        num_key_value_heads=2, head_dim=16, rope_theta=10000.0, tie_word_embeddings=True,
        max_position_embeddings=512, norm_topk_prob=True)


def _tiny_moe(seed=0) -> qwen3_moe.Model:
    mx.random.seed(seed)
    model = qwen3_moe.Model(_tiny_moe_args())
    mx.eval(model.parameters())
    return model


def _moe_hf_names(a: qwen3_moe.ModelArgs) -> list[str]:
    names = ["model.embed_tokens.weight"]
    for i in range(a.num_hidden_layers):
        L = f"model.layers.{i}"
        names += [f"{L}.self_attn.{p}.weight" for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
        if i in a.mlp_only_layers:
            names += [f"{L}.mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")]
        else:
            names.append(f"{L}.mlp.gate.weight")
            names += [f"{L}.mlp.experts.{e}.{p}.weight" for e in range(a.num_experts)
                      for p in ("gate_proj", "up_proj", "down_proj")]
    return names


def test_moe_model_names_and_transparency():
    a = _tiny_moe_args()
    model = _tiny_moe()
    cfg = {"model_type": "qwen3_moe", "num_hidden_layers": 2, "num_experts": 4, "decoder_sparse_step": 1,
           "mlp_only_layers": [0], "tie_word_embeddings": True}
    expected = {s.name[:-len(".weight")] for s in get_adapter(cfg).linears(_moe_hf_names(a))
                if s.role not in ("embed", "lm_head")}
    tokens = mx.array(np.random.default_rng(0).integers(0, 128, size=(2, 40)))  # sorted path (>= 64 routes)
    ref = model(tokens)
    rec = Recorder()
    assert set(rec.instrument(model, "")) == expected == set(rec.moments)
    assert mx.array_equal(model(tokens), ref)
    c = rec.expert_counts["model.layers.1.mlp"]
    assert c.dtype == mx.int64 and c.shape == (4,) and int(c.sum()) == 2 * 40 * 2
    assert rec.moments["model.layers.1.mlp.gate"].count == 80
    rec.remove(model)
    assert all(type(m) is SwitchLinear for _, m in model.named_modules() if isinstance(m, SwitchLinear))
    assert mx.array_equal(model(tokens), ref)


def _moe_reference(block, xs: list[np.ndarray], e: int):
    """Rows routed to expert e (gate/up input) and its down input, by running the router ourselves."""
    k = block.top_k
    rows, downs = [], []
    wg = np.array(block.switch_mlp.gate_proj.weight[e])
    wu = np.array(block.switch_mlp.up_proj.weight[e])
    for x in xs:
        gates = mx.softmax(block.gate(mx.array(x)), axis=-1, precise=True)
        inds = np.array(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]).reshape(-1, k)
        flat = x.reshape(-1, x.shape[-1])
        sel = flat[(inds == e).any(1)]
        g = sel @ wg.T
        rows.append(sel)
        downs.append(g / (1 + np.exp(-g)) * (sel @ wu.T))
    return np.concatenate(rows), np.concatenate(downs)


def test_moe_per_expert_moments_and_counts():
    model = _tiny_moe(seed=3)
    block = model.model.layers[1].mlp
    rng = np.random.default_rng(3)
    xs = [rng.normal(size=(1, 5, 64)).astype(np.float32),     # 10 routes: unsorted path
          rng.normal(size=(4, 30, 64)).astype(np.float32)]    # 240 routes: sorted path
    refs = [block(mx.array(x)) for x in xs]
    P = "model.layers.1.mlp"
    rec = Recorder(full_second_moment=[f"{P}.experts.2.down_proj"])
    rec.instrument(block, P)
    for x, r in zip(xs, refs):
        assert mx.array_equal(block(mx.array(x)), r)
    counts = np.array(rec.expert_counts[P])
    assert counts.sum() == (5 + 120) * 2
    for e in range(4):
        gate, up, down = (rec.moments[f"{P}.experts.{e}.{p}"] for p in ("gate_proj", "up_proj", "down_proj"))
        assert gate is up and gate.count == counts[e] == down.count
        rows, drows = _moe_reference(block, xs, e)
        assert len(rows) == counts[e]
        _check(gate, _np_stats(rows), full=False)
        _check(down, _np_stats(drows), full=(e == 2))
    assert rec.moments[f"{P}.experts.1.down_proj"].second_moment() is None
    rec.remove(block)


def test_save_load_roundtrip(tmp_path):
    model = _tiny_moe(seed=4)
    tokens = mx.array(np.random.default_rng(4).integers(0, 128, size=(2, 12)))
    rec = Recorder(full_second_moment=["model.layers.0.mlp.down_proj", "model.layers.1.mlp.experts.0.down_proj"])
    rec.instrument(model, "")
    model(tokens)
    rec.remove(model)
    path = tmp_path / "acts.safetensors"
    rec.save(path)
    side = json.loads(path.with_suffix(".json").read_text())
    assert "model.layers.1.mlp.experts.3.down_proj" in side["moments"]
    assert side["expert_counts"]["model.layers.1.mlp"] == np.array(rec.expert_counts["model.layers.1.mlp"]).tolist()
    got = Recorder.load(path)
    assert set(got.moments) == set(rec.moments)
    L = "model.layers.1"
    assert got.moments[f"{L}.self_attn.q_proj"] is got.moments[f"{L}.self_attn.k_proj"]
    assert got.moments[f"{L}.mlp.experts.1.gate_proj"] is got.moments[f"{L}.mlp.experts.1.up_proj"]
    for n, m in rec.moments.items():
        g = got.moments[n]
        assert g.count == m.count
        for f in ("mean", "var", "max_abs", "second_moment_diag"):
            np.testing.assert_allclose(np.array(getattr(g, f)()), np.array(getattr(m, f)()), **TOL)
        assert (g.second_moment() is None) == (m.second_moment() is None)
    np.testing.assert_allclose(np.array(got.moments["model.layers.0.mlp.down_proj"].second_moment()),
                               np.array(rec.moments["model.layers.0.mlp.down_proj"].second_moment()), **TOL)
    assert mx.array_equal(got.expert_counts[f"{L}.mlp"], rec.expert_counts[f"{L}.mlp"])


def test_mismatched_shared_input_raises():
    lin = nn.Module()
    lin.gate_proj, lin.up_proj = nn.Linear(4, 3), nn.Linear(4, 3)
    rec = Recorder()
    rec.instrument(lin, "blk")
    lin.gate_proj(mx.ones((2, 4)))
    with pytest.raises(RuntimeError):
        lin.up_proj(mx.zeros((2, 4)))
