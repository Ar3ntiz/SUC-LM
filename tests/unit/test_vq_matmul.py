"""Numerical tests (spec §34) for the direct VQ matmul kernels against the float32 reference oracle."""
import mlx.core as mx
import numpy as np
import pytest

from suc.kernels.vq_matmul import vq_decode, vq_matmul_metal, vq_matmul_reference
from suc.quant.packing import pack_indices

# relative Frobenius error tolerance per x dtype; float32 accumulation, error dominated by output rounding
REL_TOL = {mx.float16: 1e-3, mx.bfloat16: 8e-3, mx.float32: 1e-5}


def make_problem(out, inp, g, k, seed=0):
    rng = np.random.default_rng(seed)
    cb = rng.normal(size=(k, g)).astype(np.float16)
    idx = rng.integers(0, k, size=(out, inp // g), dtype=np.uint32)
    s = rng.uniform(0.01, 0.1, size=(out,)).astype(np.float32)
    return mx.array(cb), mx.array(idx), mx.array(s)


def errors(y, ref):
    d = np.abs(np.array(y.astype(mx.float32)) - np.array(ref))
    r = np.array(ref)
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                rel=float(np.linalg.norm(d) / max(np.linalg.norm(r), 1e-30)))


def run(b, out, inp, g, k, strategy, dtype=mx.float16, seed=0):
    cb, idx, s = make_problem(out, inp, g, k, seed)
    bits = int(np.log2(k))
    x = mx.random.normal((b, inp), key=mx.random.key(seed + 1)).astype(dtype)
    y = vq_matmul_metal(x, cb, pack_indices(idx, bits), s, bits=bits, in_features=inp, strategy=strategy)
    ref = vq_matmul_reference(x, cb, idx, s)
    assert y.shape == (b, out) and y.dtype == dtype
    return y, ref, errors(y, ref)


def test_vq_decode_matches_numpy():
    cb, idx, s = make_problem(16, 32, 4, 16)
    w = vq_decode(cb, idx, s)
    assert w.shape == (16, 32) and w.dtype == mx.float32
    c, i, sc = np.array(cb).astype(np.float32), np.array(idx), np.array(s)
    expect = np.stack([np.concatenate([sc[r] * c[i[r, gi]] for gi in range(i.shape[1])]) for r in range(16)])
    np.testing.assert_array_equal(np.array(w), expect)


def test_reference_is_dense_matmul():
    cb, idx, s = make_problem(8, 16, 4, 16)
    x = mx.random.normal((3, 16), key=mx.random.key(0)).astype(mx.float16)
    ref = vq_matmul_reference(x, cb, idx, s)
    assert ref.dtype == mx.float32
    np.testing.assert_allclose(np.array(ref), np.array(x.astype(mx.float32)) @ np.array(vq_decode(cb, idx, s)).T,
                               rtol=1e-5, atol=1e-5)


STRATEGIES = ["lookup", "lookup_vec", "lookup_mma", "lookup_k1", "table"]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("g,k", [(4, 16), (4, 256), (8, 16), (8, 256)])
@pytest.mark.parametrize("out,inp", [(64, 128), (1024, 1024), (3072, 1024), (1024, 3072)])
@pytest.mark.parametrize("b", [1, 4, 16])
def test_metal_matches_reference_fp16(b, out, inp, g, k, strategy):
    _, _, e = run(b, out, inp, g, k, strategy)
    assert e["rel"] < REL_TOL[mx.float16], e


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("b", [1, 16])
def test_metal_matches_reference_other_dtypes(b, dtype, strategy):
    _, _, e = run(b, 1024, 1024, 8, 256, strategy, dtype=dtype)
    assert e["rel"] < REL_TOL[dtype], e


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("b,out,inp,g,k", [
    (1, 37, 296, 8, 256),     # 37 groups: not a multiple of the 32-wide simdgroup; odd row count
    (3, 50, 100, 4, 32),      # 25 groups, 5-bit indices straddling words
    (2, 64, 192, 8, 1024),    # 10-bit indices, codebook exactly at the 16 KB ceiling
    (5, 33, 64, 4, 8),        # 3-bit indices
    (20, 70, 128, 8, 64),     # batch larger than one kernel batch tile
    (17, 70, 128, 8, 64),     # tail tile of one row (auto: mma tile + vec tile)
    (6, 21, 40, 2, 4),        # G=2, 2-bit indices (16 groups per word), 5 groups per row
    (4, 19, 96, 16, 64),      # G=16, 6-bit straddling indices
    (8, 40, 600, 8, 256),     # 75 groups: partial K chunk
    (4, 20, 96, 3, 16),       # odd G: a fragment's two columns may straddle groups
])
def test_edge_cases(b, out, inp, g, k, strategy):
    _, _, e = run(b, out, inp, g, k, strategy)
    assert e["rel"] < REL_TOL[mx.float16], e


def test_lookup_rejects_codebook_over_16kb():
    cb, idx, s = make_problem(8, 64, 16, 1024)  # 1024 * 16 * 2 B = 32 KB > 16 KB (D4)
    x = mx.zeros((1, 64), dtype=mx.float16)
    with pytest.raises(ValueError, match="16 KB"):
        vq_matmul_metal(x, cb, pack_indices(idx, 10), s, bits=10, in_features=64, strategy="lookup")


def test_rejects_bad_shapes():
    cb, idx, s = make_problem(8, 64, 8, 16)
    p = pack_indices(idx, 4)
    with pytest.raises(ValueError):
        vq_matmul_metal(mx.zeros((1, 32), dtype=mx.float16), cb, p, s, bits=4, in_features=64)
    with pytest.raises(ValueError):
        vq_matmul_metal(mx.zeros((1, 64), dtype=mx.float16), cb, p, s, bits=5, in_features=64)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_deterministic(strategy):
    y1, _, _ = run(4, 1024, 1024, 8, 256, strategy)
    y2, _, _ = run(4, 1024, 1024, 8, 256, strategy)
    np.testing.assert_array_equal(np.array(y1), np.array(y2))


@pytest.mark.parametrize("path", ["auto", "vec", "mma"])
@pytest.mark.parametrize("codebook_in", ["device", "threadgroup"])
@pytest.mark.parametrize("b,out,inp,g,k,m", [(1, 1024, 1024, 8, 256, 1), (4, 37, 296, 8, 256, 2),
                                             (16, 50, 100, 4, 32, 3), (20, 70, 128, 8, 64, 4)])
def test_lookup_codebook_placement(b, out, inp, g, k, m, codebook_in, path):
    from suc.kernels.vq_matmul import additive_matmul_metal, additive_matmul_reference
    rng = np.random.default_rng(0)
    cb = mx.array(rng.normal(size=(m, k, g)).astype(np.float16))
    idx = mx.array(rng.integers(0, k, size=(m, out, inp // g), dtype=np.uint32))
    s = mx.array(rng.uniform(0.01, 0.1, size=(out,)).astype(np.float32))
    bits = int(np.log2(k))
    x = mx.random.normal((b, inp), key=mx.random.key(1)).astype(mx.float16)
    pk = mx.stack([pack_indices(idx[j], bits) for j in range(m)])
    y = additive_matmul_metal(x, cb, pk, s, bits=bits, in_features=inp, codebook_in=codebook_in, path=path)
    assert errors(y, additive_matmul_reference(x, cb, idx, s))["rel"] < REL_TOL[mx.float16]


@pytest.mark.parametrize("b", [1, 3, 4, 7, 8, 9, 16])
def test_auto_dispatch_matches_forced_paths(b):
    """'lookup' = lookup_vec below MMA_MIN_BATCH, lookup_mma from it; all three agree with the oracle."""
    import suc.kernels.vq_matmul as vm
    ys = {s: run(b, 256, 512, 8, 256, s)[0] for s in ("lookup", "lookup_vec", "lookup_mma")}
    same = "lookup_mma" if b >= vm.mma_min_batch(1) else "lookup_vec"
    np.testing.assert_array_equal(np.array(ys["lookup"]), np.array(ys[same]))


@pytest.mark.parametrize("hm", [False, True])
@pytest.mark.parametrize("vec", [1, 2, 4])
@pytest.mark.parametrize("sg,rows", [(8, 4), (4, 4), (2, 8)])
@pytest.mark.parametrize("b,out,inp,g,k", [(1, 1024, 1024, 8, 256), (2, 37, 296, 8, 256), (1, 50, 100, 4, 32),
                                           (1, 33, 64, 4, 8), (1, 64, 4096, 4, 256)])
def test_lookup_vec_ab_switches(b, out, inp, g, k, vec, sg, rows, hm):
    """Batch-1 A/B knobs of the vec kernel (LOOKUP_VEC words per lane step, LOOKUP_SG, LOOKUP_ROWS,
    LOOKUP_HALF_MATH) keep numerics within the fp16 tolerance."""
    import suc.kernels.vq_matmul as vm
    old = vm.LOOKUP_VEC, vm.LOOKUP_SG, dict(vm.LOOKUP_ROWS), vm.LOOKUP_HALF_MATH
    try:
        vm.LOOKUP_VEC, vm.LOOKUP_SG, vm.LOOKUP_HALF_MATH = vec, sg, hm
        vm.LOOKUP_ROWS.update({1: rows, 2: rows})
        _, _, e = run(b, out, inp, g, k, "lookup_vec")
    finally:
        vm.LOOKUP_VEC, vm.LOOKUP_SG, vm.LOOKUP_HALF_MATH = old[0], old[1], old[3]
        vm.LOOKUP_ROWS.clear(); vm.LOOKUP_ROWS.update(old[2])
    assert e["rel"] < REL_TOL[mx.float16], e


@pytest.mark.parametrize("kind", ["reg", "tg"])
@pytest.mark.parametrize("sg,rt", [(4, 2), (2, 4), (8, 1), (4, 1), (2, 8)])
@pytest.mark.parametrize("b,g,k", [(4, 8, 256), (16, 8, 256), (9, 4, 32), (16, 3, 16)])
def test_lookup_mma_tile_switches(b, g, k, sg, rt, kind):
    """A/B knobs of the simdgroup-matrix path (MMA_KIND, MMA_SG, MMA_RT) keep numerics."""
    import suc.kernels.vq_matmul as vm
    old = vm.MMA_SG, vm.MMA_RT, vm.MMA_KIND
    try:
        vm.MMA_SG, vm.MMA_RT, vm.MMA_KIND = sg, rt, kind
        _, _, e = run(b, 300, 960, g, k, "lookup_mma")
    finally:
        vm.MMA_SG, vm.MMA_RT, vm.MMA_KIND = old
    assert e["rel"] < REL_TOL[mx.float16], e


def test_lookup_mma_tg_kind_threadgroup_codebook_and_dtypes():
    import suc.kernels.vq_matmul as vm
    old = vm.MMA_KIND
    try:
        vm.MMA_KIND = "tg"
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            _, _, e = run(16, 200, 512, 8, 256, "lookup_mma", dtype=dtype)
            assert e["rel"] < REL_TOL[dtype], (dtype, e)
    finally:
        vm.MMA_KIND = old


def test_codebook_in_auto_resolves_per_gpu_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("SUC_TUNING_CACHE", str(tmp_path / "none.json"))
    import suc.kernels.autotune as at
    at.clear_memory_cache()
    from suc.kernels.vq_matmul import resolve_codebook_in
    assert resolve_codebook_in("auto", "applegpu_g13d") == "threadgroup"   # M1 Ultra (K3 A/B)
    assert resolve_codebook_in("auto", "applegpu_g15g") == "device"        # M3 (K2 A/B)
    assert resolve_codebook_in("device", "applegpu_g13d") == "device"
    with pytest.raises(ValueError):
        resolve_codebook_in("shared", "applegpu_g13d")


# ------------------------------------------------------------------ low-dimensional single codebooks (Q5c)
# G = 2 / 3 with 8-, 6-, 4-, 9-bit indices: non-power-of-2 widths straddle words, 9-bit indices cross words in a
# 9-word period; the vec kernel reads whole word chunks (lcm(bits, 32) bits) per lane step, the odd-G mma path
# owns two whole groups per lane.
LOW_G = [(2, 256), (2, 64), (2, 16), (3, 512), (3, 64)]


@pytest.mark.parametrize("strategy", ["lookup_vec", "lookup_mma", "table"])
@pytest.mark.parametrize("g,k", LOW_G)
@pytest.mark.parametrize("out,inp", [(64, 384), (130, 390), (37, 3072), (8, 12288)])
@pytest.mark.parametrize("b", [1, 4, 16])
def test_low_g_matches_reference_fp32(b, out, inp, g, k, strategy):
    """float32 x: the kernels reproduce the float32 oracle to accumulation rounding (rel ~1e-6)."""
    if inp % g:
        pytest.skip("in_features % G != 0")
    _, _, e = run(b, out, inp, g, k, strategy, dtype=mx.float32)
    assert e["rel"] < 1e-5, e


@pytest.mark.parametrize("strategy", ["lookup", "lookup_vec", "lookup_mma"])
@pytest.mark.parametrize("g,k", LOW_G)
@pytest.mark.parametrize("b,out,inp", [(1, 1024, 3072), (5, 33, 66), (17, 70, 1086), (3, 9, 6)])
def test_low_g_edge_cases_fp16(b, out, inp, g, k, strategy):
    """Partial word chunks at the row end (e.g. 6-bit: 16 groups / 3 words, 9-bit: 32 groups / 9 words),
    rows not a multiple of the tiles, tails of the odd-G 8-group blocks."""
    if inp % g:
        pytest.skip("in_features % G != 0")
    _, _, e = run(b, out, inp, g, k, strategy)
    assert e["rel"] < REL_TOL[mx.float16], e


@pytest.mark.parametrize("path", ["vec", "mma"])
@pytest.mark.parametrize("codebook_in", ["device", "threadgroup"])
@pytest.mark.parametrize("g,k", LOW_G)
@pytest.mark.parametrize("b", [1, 4, 16])
def test_low_g_codebook_placement_fp32(b, g, k, codebook_in, path):
    from suc.kernels.vq_matmul import additive_matmul_metal, additive_matmul_reference
    out, inp = 70, 768
    rng = np.random.default_rng(3)
    cb = mx.array(rng.normal(size=(1, k, g)).astype(np.float16))
    idx = mx.array(rng.integers(0, k, size=(1, out, inp // g), dtype=np.uint32))
    s = mx.array(rng.uniform(0.5, 1.0, size=(out,)).astype(np.float32))
    bits = int(np.log2(k))
    x = mx.random.normal((b, inp), key=mx.random.key(1))
    y = additive_matmul_metal(x, cb, pack_indices(idx[0], bits)[None], s, bits=bits, in_features=inp,
                              codebook_in=codebook_in, path=path)
    assert errors(y, additive_matmul_reference(x, cb, idx, s))["rel"] < 1e-5


@pytest.mark.parametrize("vec", [1, 2, 4])
@pytest.mark.parametrize("g,k", LOW_G)
def test_low_g_vec_words_per_step(g, k, vec):
    """LOOKUP_VEC consecutive word chunks per lane step also for straddling widths."""
    import suc.kernels.vq_matmul as vm
    old = vm.LOOKUP_VEC
    try:
        vm.LOOKUP_VEC = vec
        _, _, e = run(1, 50, 1086, g, k, "lookup_vec", dtype=mx.float32)
    finally:
        vm.LOOKUP_VEC = old
    assert e["rel"] < 1e-5, e


@pytest.mark.parametrize("strategy", ["lookup_vec", "lookup_mma"])
@pytest.mark.parametrize("b,out,inp,g,k", [
    (4, 30, 200, 2, 1024),    # G = 2, 10-bit: 4 groups per lane span exceed one word pair (per-group fetch)
    (4, 30, 300, 3, 1024),    # odd G, 10-bit straddling indices
    (8, 30, 202, 2, 64),      # NG = 101: partial 2-group x span and partial 16-group chunk
    (4, 12, 18, 3, 512),      # NG = 6 < one 8-group block / 32-group chunk
    (1, 33, 3, 3, 512),       # a single group per row
])
def test_low_g_partial_spans_fp32(b, out, inp, g, k, strategy):
    _, _, e = run(b, out, inp, g, k, strategy, dtype=mx.float32)
    assert e["rel"] < 1e-5, e
