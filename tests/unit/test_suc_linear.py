"""Numerical tests (spec §34) for SucLinear (README §5): additive kernel, fast rotations, codec round trip."""
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from suc.kernels.suc_linear import SucLinear, fast_rht, fast_rht_inverse
from suc.kernels.vq_matmul import additive_matmul_metal, additive_matmul_reference
from suc.quant.codecs import make_codec
from suc.quant.hadamard import random_signs, rht, rht_inverse
from suc.quant.packing import pack_indices

REL_TOL = {mx.float16: 1e-3, mx.bfloat16: 8e-3, mx.float32: 1e-5}   # K1 tolerances


def errors(y, ref):
    d = np.abs(np.array(y.astype(mx.float32)) - np.array(ref.astype(mx.float32)))
    r = np.array(ref.astype(mx.float32))
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                rel=float(np.linalg.norm(d) / max(np.linalg.norm(r), 1e-30)))


def additive_problem(m, out, inp, g, k, seed=0):
    rng = np.random.default_rng(seed)
    cb = mx.array(rng.normal(size=(m, k, g)).astype(np.float16))
    idx = mx.array(rng.integers(0, k, size=(m, out, inp // g), dtype=np.uint32))
    s = mx.array(rng.uniform(0.01, 0.1, size=(out,)).astype(np.float32))
    return cb, idx, s


def pack_all(idx, bits):
    return mx.stack([pack_indices(idx[m], bits) for m in range(idx.shape[0])])


# ------------------------------------------------------------------------------------- additive kernel

@pytest.mark.parametrize("m", [1, 2, 3, 4])
@pytest.mark.parametrize("out,inp,g,k", [(1024, 1024, 8, 256), (3072, 1024, 8, 256), (1024, 3072, 4, 256),
                                         (50, 100, 4, 32), (37, 296, 8, 16)])
@pytest.mark.parametrize("b,path", [(1, "auto"), (4, "auto"), (16, "auto"), (20, "auto"), (4, "vec"), (16, "vec"),
                                    (1, "mma"), (2, "mma")])
def test_additive_kernel_matches_oracle(b, path, out, inp, g, k, m):
    cb, idx, s = additive_problem(m, out, inp, g, k)
    bits = int(np.log2(k))
    x = mx.random.normal((b, inp), key=mx.random.key(1)).astype(mx.float16)
    y = additive_matmul_metal(x, cb, pack_all(idx, bits), s, bits=bits, in_features=inp, path=path)
    ref = additive_matmul_reference(x, cb, idx, s)
    assert y.shape == (b, out) and y.dtype == mx.float16
    e = errors(y, ref)
    assert e["rel"] < REL_TOL[mx.float16], e


def test_additive_reference_is_sum_of_codebooks():
    cb, idx, s = additive_problem(3, 8, 32, 4, 16)
    x = mx.random.normal((2, 32), key=mx.random.key(0))
    c, i = np.array(cb).astype(np.float32), np.array(idx)
    w = np.stack([np.concatenate([sum(c[m][i[m, r, gi]] for m in range(3)) for gi in range(8)]) for r in range(8)])
    w *= np.array(s)[:, None]
    np.testing.assert_allclose(np.array(additive_matmul_reference(x, cb, idx, s)), np.array(x) @ w.T,
                               rtol=1e-5, atol=1e-5)


def test_additive_rejects_codebooks_over_d4_budget():
    cb, idx, s = additive_problem(5, 8, 64, 8, 256)          # 5 * 256 * 8 * 2 B = 20 KB > 16 KB together
    with pytest.raises(ValueError, match="16 KB"):
        additive_matmul_metal(mx.zeros((1, 64), dtype=mx.float16), cb, pack_all(idx, 8), s, bits=8, in_features=64)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("m", [1, 3])
def test_additive_mma_other_dtypes(dtype, m):
    cb, idx, s = additive_problem(m, 300, 1024, 8, 256)
    x = mx.random.normal((16, 1024), key=mx.random.key(1)).astype(dtype)
    y = additive_matmul_metal(x, cb, pack_all(idx, 8), s, bits=8, in_features=1024, path="mma")
    assert y.dtype == dtype
    e = errors(y, additive_matmul_reference(x, cb, idx, s))
    assert e["rel"] < REL_TOL[dtype], e


def test_additive_float32_out_dtype_from_fp16_x():
    cb, idx, s = additive_problem(2, 128, 512, 8, 256)
    x = mx.random.normal((8, 512), key=mx.random.key(1)).astype(mx.float16)
    for path in ("vec", "mma"):
        y = additive_matmul_metal(x, cb, pack_all(idx, 8), s, bits=8, in_features=512, path=path,
                                  out_dtype=mx.float32)
        assert y.dtype == mx.float32
        assert errors(y, additive_matmul_reference(x, cb, idx, s))["rel"] < 1e-5


def test_additive_at_d4_budget_is_accepted():
    cb, idx, s = additive_problem(4, 64, 64, 8, 256)         # exactly 16 KB
    x = mx.random.normal((1, 64), key=mx.random.key(0)).astype(mx.float16)
    y = additive_matmul_metal(x, cb, pack_all(idx, 8), s, bits=8, in_features=64)
    assert errors(y, additive_matmul_reference(x, cb, idx, s))["rel"] < REL_TOL[mx.float16]


# ------------------------------------------------------------------------------------------- rotations

@pytest.mark.parametrize("n", [1024, 3072, 4096, 12288, 5120, 7168, 16384])
def test_fast_rht_matches_reference(n):
    x = mx.random.normal((3, n), key=mx.random.key(n))
    s = random_signs(n, 7)
    np.testing.assert_allclose(np.array(fast_rht(x, s)), np.array(rht(x, 7)), atol=5e-6)
    np.testing.assert_allclose(np.array(fast_rht_inverse(x, s)), np.array(rht_inverse(x, 7)), atol=5e-6)
    np.testing.assert_allclose(np.array(fast_rht_inverse(fast_rht(x, s), s)), np.array(x), atol=5e-6)


# --------------------------------------------------------------------------------------------- module

@lru_cache(maxsize=None)
def encoded(kind, out, inp, rotate, m):
    cfg = dict(kind=kind, k=256, block_size=8, rotate=rotate, kmeans_iters=3, fit_blocks=20000)
    if kind == "additive":
        cfg["num_codebooks"] = m
    w = mx.random.normal((out, inp), key=mx.random.key(out + inp)) * 0.02
    codec = make_codec(cfg)
    e = codec.encode(w, seed=3)
    return e, codec.decode(e)


CASES = [("vq", True, 1), ("vq", False, 1), ("additive", True, 2), ("additive", False, 3)]


@pytest.mark.parametrize("kind,rotate,m", CASES)
@pytest.mark.parametrize("out,inp", [(1024, 1024), (3072, 1024), (1024, 3072)])
@pytest.mark.parametrize("lead", [(1,), (4,), (2, 3)])
def test_suc_linear_matches_decoded_weight(lead, out, inp, kind, rotate, m):
    e, w_hat = encoded(kind, out, inp, rotate, m)
    layer = SucLinear.from_encoded(e)
    x = mx.random.normal((*lead, inp), key=mx.random.key(5)).astype(mx.float16)
    y = layer(x)
    assert y.shape == (*lead, out) and y.dtype == mx.float16
    ref = x.astype(mx.float32) @ w_hat.T
    e_ = errors(y, ref)
    assert e_["rel"] < REL_TOL[mx.float16], e_


@pytest.mark.parametrize("kind,rotate,m", CASES)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_suc_linear_other_dtypes(dtype, kind, rotate, m):
    e, w_hat = encoded(kind, 1024, 1024, rotate, m)
    x = mx.random.normal((4, 1024), key=mx.random.key(6)).astype(dtype)
    y = SucLinear.from_encoded(e)(x)
    assert y.dtype == dtype
    e_ = errors(y, x.astype(mx.float32) @ w_hat.T)
    assert e_["rel"] < REL_TOL[dtype], e_


def test_suc_linear_bias_and_module():
    e, w_hat = encoded("vq", 1024, 1024, True, 1)
    bias = mx.random.normal((1024,), key=mx.random.key(9))
    layer = SucLinear.from_encoded(e)
    layer_b = SucLinear(layer.codebooks, layer.packed_indices, layer.row_scales, bits=layer.bits,
                        in_features=1024, seed_in=layer.seed_in, seed_out=layer.seed_out, bias=bias)
    assert isinstance(layer_b, nn.Module)
    x = mx.random.normal((3, 1024), key=mx.random.key(2)).astype(mx.float32)
    e_ = errors(layer_b(x), x @ w_hat.T + bias)
    assert e_["rel"] < REL_TOL[mx.float32], e_


def test_suc_linear_rejects_other_codecs_and_bad_input():
    w = mx.random.normal((64, 64), key=mx.random.key(0))
    with pytest.raises(ValueError, match="vq or additive"):
        SucLinear.from_encoded(make_codec(dict(kind="affine", bits=4)).encode(w, seed=0))
    e, _ = encoded("vq", 1024, 1024, False, 1)
    with pytest.raises(ValueError):
        SucLinear.from_encoded(e)(mx.zeros((2, 512), dtype=mx.float16))


@pytest.mark.parametrize("kind,m", [("vq", 1), ("additive", 2)])
def test_suc_linear_input_only_rotation(kind, m):
    e, w_hat = encoded(kind, 1000, 1024, True, m)              # 1000 rows: no Hadamard on the output side
    assert e.params["rotate_out"] is False and e.params["seed_out"] is None
    layer = SucLinear.from_encoded(e)
    assert layer.seed_in is not None and layer.seed_out is None
    x = mx.random.normal((2, 3, 1024), key=mx.random.key(4)).astype(mx.float16)
    e_ = errors(layer(x), x.astype(mx.float32) @ w_hat.T)
    assert e_["rel"] < REL_TOL[mx.float16], e_


def test_suc_linear_legacy_params_mean_both_rotations():
    import copy
    e, w_hat = encoded("vq", 1024, 1024, True, 1)
    legacy = copy.copy(e)
    legacy.params = {k: v for k, v in e.params.items() if k not in ("rotate_in", "rotate_out")}
    layer = SucLinear.from_encoded(legacy)
    assert layer.seed_in == e.params["seed_in"] and layer.seed_out == e.params["seed_out"]
    x = mx.random.normal((4, 1024), key=mx.random.key(8)).astype(mx.float16)
    assert errors(layer(x), x.astype(mx.float32) @ w_hat.T)["rel"] < REL_TOL[mx.float16]


@pytest.mark.parametrize("path", ["auto", "vec", "mma"])
@pytest.mark.parametrize("b", [1, 4, 16])
def test_suc_linear_kernel_paths(b, path):
    e, w_hat = encoded("additive", 1024, 1024, False, 3)
    layer = SucLinear.from_encoded(e)
    x = mx.random.normal((b, 1024), key=mx.random.key(11)).astype(mx.float16)
    assert errors(layer.kernel(x, path=path), x.astype(mx.float32) @ w_hat.T)["rel"] < REL_TOL[mx.float16]


# ------------------------------------------------------------------- fused rotations / pre-rotated input / fusion
import suc.kernels.suc_linear as sl  # noqa: E402
from suc.kernels.suc_linear import SucFusedLinear, fusable_key, fused_matmul_metal  # noqa: E402


@pytest.mark.parametrize("rotation", ["fused", "mlx"])
@pytest.mark.parametrize("kind,rotate,m", CASES)
def test_rotation_paths_agree(rotation, kind, rotate, m, monkeypatch):
    e, w_hat = encoded(kind, 3072, 1024, rotate, m)
    monkeypatch.setattr(sl, "ROTATION", rotation)
    layer = SucLinear.from_encoded(e)
    x = mx.random.normal((3, 1024), key=mx.random.key(12)).astype(mx.bfloat16)
    assert errors(layer(x), x.astype(mx.float32) @ w_hat.T)["rel"] < REL_TOL[mx.bfloat16]
    x32 = x.astype(mx.float32)
    assert errors(layer(x32), x32 @ w_hat.T)["rel"] < REL_TOL[mx.float32]


def test_from_rotated_equals_call():
    e, _ = encoded("vq", 3072, 1024, True, 1)
    layer = SucLinear.from_encoded(e)
    x = mx.random.normal((2, 1024), key=mx.random.key(13)).astype(mx.float16)
    a = layer(x)
    b = layer.from_rotated(layer.rotate_in(x), x)
    assert np.array_equal(np.array(a), np.array(b))


def shared_seed_members(kind="vq", m=1, sizes=(2048, 1024, 1024), inp=1024, g=4, k=256, seed_in=777):
    """Encodings of independent weights with one shared input rotation (rotation group)."""
    cfg = dict(kind=kind, k=k, block_size=g, rotate=True, kmeans_iters=2, fit_blocks=8192)
    if kind == "additive":
        cfg["num_codebooks"] = m
    codec = make_codec(cfg)
    encs = [codec.encode(mx.random.normal((o, inp), key=mx.random.key(20 + i)) * 0.02, seed=50 + i, seed_in=seed_in)
            for i, o in enumerate(sizes)]
    return encs, [codec.decode(e) for e in encs]


@pytest.mark.parametrize("kind,m,g", [("vq", 1, 4), ("vq", 1, 8), ("additive", 2, 8), ("additive", 3, 4)])
@pytest.mark.parametrize("b,path", [(1, "auto"), (4, "auto"), (16, "auto"), (20, "auto"), (2, "mma"), (8, "vec")])
def test_fused_kernel_matches_members(b, path, kind, m, g):
    encs, _ = shared_seed_members(kind, m, g=g)
    mods = [SucLinear.from_encoded(e) for e in encs]
    x = mx.random.normal((b, 1024), key=mx.random.key(3)).astype(mx.float16)
    f = SucFusedLinear(mods)
    y = f.kernel(x, mx.float32, path=path)
    ref = mx.concatenate([md.kernel(x, mx.float32, path=path) for md in mods], axis=1)
    assert y.shape == (b, 4096)
    assert errors(y, ref)["rel"] < 1e-6


@pytest.mark.parametrize("placement", ["threadgroup", "device"])
@pytest.mark.parametrize("sizes", [(2048, 1024, 1024), (96, 48, 40)])   # aligned / unaligned (device fallback)
@pytest.mark.parametrize("kind,m,g", [("vq", 1, 4), ("additive", 3, 8)])
@pytest.mark.parametrize("b,path", [(1, "auto"), (4, "auto"), (16, "auto"), (2, "mma")])
def test_fused_kernel_codebook_placement(b, path, kind, m, g, sizes, placement):
    """Staged (threadgroup) codebooks: each threadgroup stages its own segment's codebooks."""
    encs, _ = shared_seed_members(kind, m, sizes=sizes, g=g)
    mods = [SucLinear.from_encoded(e) for e in encs]
    x = mx.random.normal((b, 1024), key=mx.random.key(3)).astype(mx.float16)
    y = fused_matmul_metal(x, mx.stack([md.codebooks for md in mods]),
                           mx.concatenate([md.packed_indices for md in mods], axis=1),
                           mx.concatenate([md.row_scales for md in mods]), bits=mods[0].bits, in_features=1024,
                           seg_rows=sizes, out_dtype=mx.float32, path=path, codebook_in=placement)
    ref = mx.concatenate([md.kernel(x, mx.float32, path=path) for md in mods], axis=1)
    assert errors(y, ref)["rel"] < 1e-6


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("sizes,inp", [((2048, 1024, 1024), 1024), ((3072, 3072), 1024), ((64, 32, 32), 64)])
@pytest.mark.parametrize("lead", [(1,), (5,), (2, 3)])
def test_fused_module_matches_decoded_weights(lead, sizes, inp, dtype):
    encs, w_hats = shared_seed_members(sizes=sizes, inp=inp, k=16 if inp == 64 else 256)
    mods = [SucLinear.from_encoded(e) for e in encs]
    assert len({fusable_key(md) for md in mods}) == 1
    f = SucFusedLinear(mods)
    x = mx.random.normal((*lead, inp), key=mx.random.key(4)).astype(dtype)
    x2 = x.reshape(-1, inp)
    ys = f.from_rotated_split(f.rotate_in(x2))
    for y, w in zip(ys, w_hats):
        assert y.dtype == dtype and y.shape == (x2.shape[0], w.shape[0])
        assert errors(y, x2.astype(mx.float32) @ w.T)["rel"] < REL_TOL[dtype]
    full = f(x)
    assert full.shape == (*lead, sum(sizes))
    assert errors(full.reshape(-1, sum(sizes)), mx.concatenate(ys, axis=1))["rel"] == 0.0
    # same memory as the members (concatenation, nothing duplicated)
    nbytes = lambda md: sum(v.nbytes for _, v in __import__("mlx.utils").utils.tree_flatten(md.parameters()))
    assert nbytes(f) == sum(nbytes(md) for md in mods) - (len(mods) - 1) * mods[0].signs_in.nbytes


def test_fusable_key_rejects_mixed_layouts():
    encs, _ = shared_seed_members(sizes=(1024, 1024))
    a, b = (SucLinear.from_encoded(e) for e in encs)
    other, _ = shared_seed_members(g=8, sizes=(1024,))
    c = SucLinear.from_encoded(other[0])
    assert fusable_key(a) == fusable_key(b) != fusable_key(c)
    with pytest.raises(ValueError, match="fusable"):
        SucFusedLinear([a, c])
    d, _ = shared_seed_members(sizes=(1024,), seed_in=5)
    assert fusable_key(SucLinear.from_encoded(d[0])) != fusable_key(a)
    with pytest.raises(ValueError, match="segments"):
        fused_matmul_metal(mx.zeros((1, 1024), dtype=mx.float16), mx.stack([a.codebooks] * 5),
                           mx.concatenate([a.packed_indices] * 5, axis=1), mx.concatenate([a.row_scales] * 5),
                           bits=8, in_features=1024, seg_rows=(1024,) * 5)


def test_compiled_call_equals_eager_and_sees_updates(monkeypatch):
    e, w_hat = encoded("vq", 3072, 1024, True, 1)
    layer = SucLinear.from_encoded(e)
    x = mx.random.normal((2, 1024), key=mx.random.key(14)).astype(mx.bfloat16)
    monkeypatch.setattr(sl, "COMPILE", False)
    eager = layer(x)
    monkeypatch.setattr(sl, "COMPILE", True)
    assert np.array_equal(np.array(layer(x).astype(mx.float32)), np.array(eager.astype(mx.float32)))
    layer.row_scales = layer.row_scales * 2          # module arrays are implicit inputs: no stale trace
    assert errors(layer(x), 2 * eager)["rel"] < 1e-2
    layer.set_dtype(mx.float32)                     # dtype change: retraced
    assert errors(layer(x.astype(mx.float32)), 2 * (x.astype(mx.float32) @ w_hat.T))["rel"] < 1e-4
