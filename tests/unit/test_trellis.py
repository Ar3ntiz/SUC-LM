"""Trellis-coded quantization (suc.quant.trellis): stream layout, tail-biting, bit accounting, Gaussian MSE."""
import mlx.core as mx
import numpy as np
import pytest

from suc.quant.codecs import make_codec
from suc.quant.residual import ResidualCodec
from suc.quant.trellis import (
    TrellisCodec, chunks_to_states, compute_code, pack_stream, states_to_chunks, stream_states, viterbi,
    viterbi_tailbiting,
)


@pytest.mark.parametrize("L,kv", [(10, 2), (10, 4), (12, 5), (12, 3)])
def test_tailbiting_walk_and_stream_layout(L, kv):
    n = 64
    x = mx.random.normal((4, n, 1), key=mx.random.key(0))
    vals = mx.array(compute_code("1mad", L))
    st = viterbi_tailbiting(x, vals, L, kv, window=16)
    R = 1 << (L - kv)
    # bitshift transitions, and the wrap: s_0's low L-kv bits are s_{n-1}'s high L-kv bits
    assert np.all((st[:, 1:] & (R - 1)) == (st[:, :-1] >> kv))
    assert np.all((st[:, 0] & (R - 1)) == (st[:, -1] >> kv))
    ch = states_to_chunks(st, L, kv)
    assert np.array_equal(chunks_to_states(ch, L, kv), st)
    assert np.array_equal(stream_states(pack_stream(ch, kv), L, kv), st)


def test_viterbi_is_optimal_on_a_tiny_trellis():
    L, kv, n = 4, 2, 5
    vals = np.random.default_rng(0).normal(size=(1 << L, 1)).astype(np.float32)
    x = np.random.default_rng(1).normal(size=(1, n, 1)).astype(np.float32)
    st = viterbi(mx.array(x), mx.array(vals), L, kv)[0]
    best = min(
        sum((x[0, i, 0] - vals[s, 0]) ** 2 for i, s in enumerate(path))
        for path in _paths(L, kv, n))
    got = sum((x[0, i, 0] - vals[s, 0]) ** 2 for i, s in enumerate(st))
    assert got == pytest.approx(best, rel=1e-5)


def _paths(L, kv, n):
    R = 1 << (L - kv)
    out = [[s] for s in range(1 << L)]
    for _ in range(n - 1):
        out = [p + [((p[-1] >> kv) | (c << (L - kv)))] for p in out for c in range(1 << kv)]
    assert all((p[i] & (R - 1)) == (p[i - 1] >> kv) for p in out[:5] for i in range(1, n))
    return out


@pytest.mark.parametrize("cfg", [dict(code="1mad", k=2), dict(code="1mad", k=2.5, V=2), dict(code="3inst", k=2),
                                 dict(code="hyb", k=2, Q=6, lut_iters=1, fit_seqs=64)])
def test_codec_roundtrip_bits_and_mse(cfg):
    w = mx.random.normal((8, 512), key=mx.random.key(3)) * 0.02
    c = TrellisCodec(L=10, T=128, rotate=True, **cfg)
    e = c.encode(w, seed=5)
    d = c.decode(e)
    rel = float(mx.sum((d - w) ** 2) / mx.sum(w ** 2))
    assert rel < (0.12 if cfg["k"] == 2 else 0.07)
    b = e.bits()
    assert b["indices"] == int(cfg["k"] * w.size)
    assert b["scales"] == 8 * 16
    assert b["codebooks"] == (e.arrays["lut"].size * 16 if "lut" in e.arrays else 0)


def test_make_codec_and_residual_wrapper():
    w = mx.random.normal((16, 256), key=mx.random.key(4))
    base = make_codec({"kind": "tcq", "k": 2, "L": 10, "T": 128, "code": "1mad"})
    r = ResidualCodec(base, 0.01, "int8", "magnitude")
    e = r.encode(w, seed=1)
    d = r.decode(e)
    assert e.bits()["indices"] == 2 * w.size and e.bits()["residual"] > 0
    assert float(mx.sum((d - w) ** 2) / mx.sum(w ** 2)) < float(
        mx.sum((base.decode(base.encode(w, seed=1)) - w) ** 2) / mx.sum(w ** 2))


def test_ldlq_lowers_the_hessian_proxy_loss():
    rng = np.random.default_rng(0)
    inn, out = 512, 16
    x = rng.normal(size=(4096, inn)) * (1 + 20 * (rng.random(inn) < 0.02))[None]
    x[:, 1:] += 0.5 * x[:, :-1]                                                    # correlated, anisotropic input
    h = (x.T @ x / x.shape[0]).astype(np.float32)
    w = mx.array(rng.normal(size=(out, inn)).astype(np.float32))
    c = TrellisCodec(k=2, L=10, T=128, code="1mad", rotate=True, rotate_out=False)

    def proxy(d):
        e = np.array(d - w)
        return float(np.einsum("ij,jk,ik->", e, h, e))

    plain = proxy(c.decode(c.encode(w, seed=1)))
    e = c.encode(w, seed=1, hessian=mx.array(h))
    assert e.params["ldlq"] == {"damp": 0.01} and e.bits()["indices"] == 2 * w.size
    assert proxy(c.decode(e)) < 0.8 * plain
    with pytest.raises(ValueError):
        TrellisCodec(k=2, L=10, T=128, rotate=False).encode(w, seed=1, hessian=mx.array(h))


@pytest.mark.parametrize("L,kv,V", [(10, 2, 1), (12, 3, 1), (12, 4, 2), (12, 5, 2)])
def test_step_kernels_agree(L, kv, V, monkeypatch):
    import suc.quant.trellis as T
    x = mx.random.normal((16, 64 // V, V), key=mx.random.key(0))
    vals = mx.array(compute_code("1mad", L, V))
    out = {}
    monkeypatch.setattr(T, "COST_HALF", False)
    for kind in ("tg", "reg"):
        monkeypatch.setattr(T, "STEP_KERNEL", kind)
        out[kind] = viterbi_tailbiting(x, vals, L, kv, window=16, code=("1mad", 1.0))
    assert np.array_equal(out["tg"], out["reg"])


def test_half_path_metrics_valid_and_near_optimal(monkeypatch):
    import suc.quant.trellis as T
    L, kv = 12, 2
    x = mx.random.normal((32, 128, 1), key=mx.random.key(5))
    vals = compute_code("1mad", L)
    err = {}
    for half in (False, True):
        monkeypatch.setattr(T, "COST_HALF", half)
        st = viterbi_tailbiting(x, mx.array(vals), L, kv, code=("1mad", 1.0))
        R = 1 << (L - kv)
        assert np.all((st[:, 1:] & (R - 1)) == (st[:, :-1] >> kv)) and np.all((st[:, 0] & (R - 1)) == (st[:, -1] >> kv))
        err[half] = float(np.sum((vals[st, 0] - np.array(x)[..., 0]) ** 2))
    assert err[True] <= err[False] * 1.002
