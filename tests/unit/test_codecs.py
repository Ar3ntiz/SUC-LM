"""Tests for the research-layout weight codecs (spec §10-§12, README §2)."""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from suc.quant.codecs import (AdditiveCodebook, AffineRTN, BF16Codec, Encoded, ScalarCodebook,
                              VectorCodebook, codebook_bytes, make_codec)

META = 512  # metadata bits per tensor


def _randn(shape: tuple[int, int], seed: int = 0, scale: float = 1.0) -> mx.array:
    return mx.array((scale * np.random.default_rng(seed).standard_normal(shape)).astype(np.float32))


def rel_mse(w: mx.array, wh: mx.array) -> float:
    a, b = np.array(w.astype(mx.float32)), np.array(wh)
    return float(((a - b) ** 2).sum() / (a ** 2).sum())


ALL = [BF16Codec(), AffineRTN(4), ScalarCodebook(8), ScalarCodebook(8, scope="row_group", row_group=16),
       VectorCodebook(16, rotate=False, kmeans_iters=5), VectorCodebook(16, kmeans_iters=5),
       AdditiveCodebook(16, num_codebooks=2, kmeans_iters=5, refine_iters=1)]


@pytest.mark.parametrize("codec", ALL, ids=lambda c: f"{c.name}")
def test_roundtrip_shape_dtype(codec) -> None:
    w = _randn((64, 128), 1, 0.02)
    e = codec.encode(w, seed=0)
    assert isinstance(e, Encoded) and e.codec == codec.name and e.shape == (64, 128)
    wh = codec.decode(e)
    assert wh.shape == (64, 128) and wh.dtype == mx.float32
    b = e.bits()
    assert set(b) == {"indices", "codebooks", "scales", "residual", "metadata", "total"}
    assert all(isinstance(v, int) for v in b.values())
    assert b["total"] == sum(v for k, v in b.items() if k != "total")
    assert e.bpw() == pytest.approx(b["total"] / (64 * 128))
    assert rel_mse(w, wh) < 1.0
    import json
    json.dumps(e.params)


def test_bf16_identity_bits() -> None:
    w = _randn((32, 64), 2)
    e = BF16Codec().encode(w, seed=0)
    np.testing.assert_array_equal(np.array(BF16Codec().decode(e)), np.array(w.astype(mx.bfloat16).astype(mx.float32)))
    assert e.bits() == {"indices": 16 * 2048, "codebooks": 0, "scales": 0, "residual": 0, "metadata": META,
                        "total": 16 * 2048 + META}


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("gs", [32, 64, 128])
def test_affine_matches_mlx(bits: int, gs: int) -> None:
    w = _randn((96, 256), bits * gs, 0.02).astype(mx.float16)
    e = AffineRTN(bits, gs).encode(w, seed=0)
    wq, s, b = mx.quantize(w, group_size=gs, bits=bits, mode="affine")
    ref = np.array(mx.dequantize(wq, s, b, group_size=gs, bits=bits, mode="affine").astype(mx.float32))
    got = np.array(AffineRTN(bits, gs).decode(e))
    # our decode is float32 q*s+b; MLX returns it rounded to fp16
    np.testing.assert_allclose(got, ref, rtol=2 ** -10, atol=1e-7)
    n = 96 * 256
    assert e.bits()["total"] - META == bits * n + 32 * n // gs
    assert e.bits()["indices"] == bits * n and e.bits()["scales"] == 32 * n // gs
    assert e.bpw() == pytest.approx(bits + 32 / gs + META / n)
    # codes stored one per weight in [0, 2^bits)
    idx = np.array(e.arrays["indices"])
    assert idx.shape == (96, 256) and idx.max() < 2 ** bits


def test_affine_bf16_input_close_to_mlx() -> None:
    w = _randn((64, 128), 5, 0.02).astype(mx.bfloat16)
    e = AffineRTN(4, 64).encode(w, seed=0)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    ref = np.array(mx.dequantize(wq, s, b, group_size=64, bits=4).astype(mx.float32))
    err_ref = ((ref - np.array(w.astype(mx.float32))) ** 2).mean()
    err = ((np.array(AffineRTN(4, 64).decode(e)) - np.array(w.astype(mx.float32))) ** 2).mean()
    assert err == pytest.approx(err_ref, rel=0.05)


def test_error_decreases_with_k() -> None:
    w = _randn((128, 256), 3)
    sc = [rel_mse(w, (c := ScalarCodebook(k)).decode(c.encode(w, seed=0))) for k in (2, 4, 16)]
    vq = [rel_mse(w, (c := VectorCodebook(k, 4, kmeans_iters=10)).decode(c.encode(w, seed=0))) for k in (4, 16, 64)]
    af = [rel_mse(w, (c := AffineRTN(b)).decode(c.encode(w, seed=0))) for b in (2, 3, 4)]
    for errs in (sc, vq, af):
        assert errs[0] > errs[1] > errs[2], errs


def test_scalar_vs_affine_report() -> None:
    w = _randn((128, 512), 4)
    for b in (2, 3, 4):
        es = rel_mse(w, (c := ScalarCodebook(2 ** b)).decode(c.encode(w, seed=0)))
        ea = rel_mse(w, (c := AffineRTN(b)).decode(c.encode(w, seed=0)))
        print(f"b={b}: scalar k={2 ** b} rel_mse={es:.4f}  affine g64 rel_mse={ea:.4f}")
        assert es < 1 and ea < 1


def test_scalar_bits_hand_checked() -> None:
    w = _randn((32, 64), 6)
    e = ScalarCodebook(8).encode(w, seed=0)
    assert e.arrays["codebooks"].shape == (1, 8, 1) and e.arrays["codebooks"].dtype == mx.float16
    assert e.arrays["indices"].shape == (32, 64) and e.arrays["row_scales"].dtype == mx.float16
    assert e.bits() == {"indices": 3 * 2048, "codebooks": 8 * 16, "scales": 32 * 16, "residual": 0,
                        "metadata": META, "total": 3 * 2048 + 128 + 512 + META}
    e = ScalarCodebook(5, scope="row_group", row_group=8).encode(w, seed=0)
    assert e.arrays["codebooks"].shape == (4, 5, 1)
    assert e.bits()["indices"] == 3 * 2048 and e.bits()["codebooks"] == 4 * 5 * 16
    assert codebook_bytes(e) == 5 * 2


def test_scalar_row_group_not_worse() -> None:
    w = _randn((64, 128), 7) * mx.array(np.linspace(0.2, 3, 128, dtype=np.float32))[None]
    et = rel_mse(w, (c := ScalarCodebook(4)).decode(c.encode(w, seed=0)))
    eg = rel_mse(w, (c := ScalarCodebook(4, scope="row_group", row_group=16)).decode(c.encode(w, seed=0)))
    assert eg <= et * 1.02


@pytest.mark.parametrize("make", [lambda: ScalarCodebook(4), lambda: VectorCodebook(16, 4, rotate=False)])
def test_col_weights_change_fit(make) -> None:
    w = np.array(_randn((128, 64), 8))
    w[:, 5] = 3.0 + 0.05 * w[:, 5]  # a heavy column far from the bulk that an unweighted fit neglects
    w = mx.array(w)
    cw = mx.ones((64,)).at[5].add(999.0)
    c = make()
    e0, e1 = c.encode(w, seed=0), c.encode(w, seed=0, col_weights=cw)
    err0 = float(((np.array(c.decode(e0)) - np.array(w))[:, 5] ** 2).mean())
    err1 = float(((np.array(c.decode(e1)) - np.array(w))[:, 5] ** 2).mean())
    print(f"col_weights heavy-column err {err0:.4g} -> {err1:.4g}")
    assert err1 < 0.7 * err0, (err0, err1)


def test_col_weights_ignored_when_rotated() -> None:
    w = _randn((64, 64), 9)
    c = VectorCodebook(16, 4, rotate=True, kmeans_iters=5)
    cw = mx.ones((64,)).at[5].add(999.0)
    e0, e1 = c.encode(w, seed=0), c.encode(w, seed=0, col_weights=cw)
    np.testing.assert_array_equal(np.array(e0.arrays["indices"]), np.array(e1.arrays["indices"]))


def test_vq_gaussian_rel_mse_and_bits() -> None:
    out, inn = 512, 1024
    w = _randn((out, inn), 10)
    e = (c := VectorCodebook(256, 8, rotate=True)).encode(w, seed=0)
    err = rel_mse(w, c.decode(e))
    print(f"vq k256 g8 rot gaussian rel_mse={err:.4f} bpw={e.bpw():.5f}")
    assert err == pytest.approx(0.32, abs=0.03)
    n = out * inn
    assert e.arrays["codebooks"].shape == (1, 256, 8) and e.arrays["codebooks"].dtype == mx.float16
    assert e.arrays["indices"].shape == (1, out, inn // 8) and e.arrays["indices"].dtype == mx.uint8
    assert e.bits() == {"indices": n // 8 * 8, "codebooks": 256 * 8 * 16, "scales": 16 * out, "residual": 0,
                        "metadata": META, "total": n + 256 * 8 * 16 + 16 * out + META}
    assert e.bpw() == pytest.approx(1 + (32768 + 8192 + 512) / n)
    assert codebook_bytes(e) == 4096
    assert e.params["rotate"] and "seed_in" in e.params and "seed_out" in e.params


def test_vq_bits_non_power_of_two_k() -> None:
    w = _randn((16, 32), 11)
    e = VectorCodebook(300, 4, rotate=False, kmeans_iters=2).encode(mx.tile(w, (8, 8)), seed=0)
    assert e.arrays["indices"].dtype == mx.uint16
    assert e.bits()["indices"] == math.ceil(math.log2(300)) * (128 * 256 // 4)


def test_rotation_roundtrip_large_k() -> None:
    w = _randn((16, 32), 12)
    c = VectorCodebook(64, 8, rotate=True, kmeans_iters=10)  # 64 blocks, one code each
    wh = c.decode(c.encode(w, seed=3))
    assert rel_mse(w, wh) < 1e-5
    np.testing.assert_allclose(np.array(wh), np.array(w), atol=5e-3)


def test_vq_unsupported_hadamard_size() -> None:
    with pytest.raises(ValueError):
        VectorCodebook(4, 8).encode(_randn((16, 72), 0), seed=0)


def test_additive_improves_with_m_and_bits() -> None:
    out, inn = 256, 512
    w = _randn((out, inn), 13)
    errs, es = [], []
    for m in (1, 2):
        c = AdditiveCodebook(256, 8, num_codebooks=m, kmeans_iters=15)
        e = c.encode(w, seed=0)
        errs.append(rel_mse(w, c.decode(e)))
        es.append(e)
    print(f"additive gaussian M=1 {errs[0]:.4f} M=2 {errs[1]:.4f}")
    assert errs[1] < 0.5 * errs[0]
    n = out * inn
    e2 = es[1]
    assert e2.arrays["codebooks"].shape == (2, 256, 8) and e2.arrays["indices"].shape == (2, out, inn // 8)
    assert e2.bits()["total"] == 2 * n + 2 * 256 * 8 * 16 + 16 * out + META
    assert e2.bpw() == pytest.approx(2 + (65536 + 16 * out + META) / n)
    assert codebook_bytes(e2) == 4096


def test_additive_decode_is_sum() -> None:
    w = _randn((32, 64), 14)
    c = AdditiveCodebook(16, 8, num_codebooks=3, rotate=False, kmeans_iters=5)
    e = c.encode(w, seed=0)
    cb = np.array(e.arrays["codebooks"].astype(mx.float32))
    idx = np.array(e.arrays["indices"]).astype(np.int64)
    blocks = sum(cb[m][idx[m]] for m in range(3))  # (out, in//G, G)
    ref = blocks.reshape(32, 64) * np.array(e.arrays["row_scales"].astype(mx.float32))[:, None]
    np.testing.assert_allclose(np.array(c.decode(e)), ref, rtol=1e-6, atol=1e-6)


def test_additive_refine_not_worse() -> None:
    w = _randn((128, 256), 15)
    e0 = rel_mse(w, (c := AdditiveCodebook(64, 8, num_codebooks=2, kmeans_iters=10)).decode(c.encode(w, seed=0)))
    e2 = rel_mse(w, (c := AdditiveCodebook(64, 8, num_codebooks=2, kmeans_iters=10, refine_iters=2))
                 .decode(c.encode(w, seed=0)))
    print(f"additive refine0 {e0:.4f} refine2 {e2:.4f}")
    assert e2 <= e0 * (1 + 1e-4)


@pytest.mark.parametrize("codec", ALL[2:], ids=lambda c: c.name)
def test_deterministic_by_seed(codec) -> None:
    w = _randn((64, 128), 16)
    a, b, c = codec.encode(w, seed=4), codec.encode(w, seed=4), codec.encode(w, seed=5)
    for k in a.arrays:
        np.testing.assert_array_equal(np.array(a.arrays[k]), np.array(b.arrays[k]))
    assert a.params == b.params
    diff = any(not np.array_equal(np.array(a.arrays[k]), np.array(c.arrays[k])) for k in a.arrays)
    assert diff or a.params != c.params


def test_fit_blocks_subsample() -> None:
    w = _randn((64, 256), 17)
    c = VectorCodebook(32, 8, rotate=False, kmeans_iters=5, fit_blocks=500)
    e = c.encode(w, seed=0)
    assert e.arrays["indices"].shape == (1, 64, 32)
    assert rel_mse(w, c.decode(e)) < 1


def test_make_codec() -> None:
    assert isinstance(make_codec({"kind": "bf16"}), BF16Codec)
    a = make_codec({"kind": "affine", "bits": 3, "group_size": 32})
    assert isinstance(a, AffineRTN) and a.bits == 3 and a.group_size == 32
    s = make_codec({"kind": "scalar", "k": 16, "scope": "row_group"})
    assert isinstance(s, ScalarCodebook) and s.k == 16 and s.scope == "row_group"
    v = make_codec({"kind": "vq", "k": 256, "block_size": 4, "rotate": False})
    assert isinstance(v, VectorCodebook) and v.block_size == 4 and not v.rotate
    ad = make_codec({"kind": "additive", "k": 256, "num_codebooks": 3})
    assert isinstance(ad, AdditiveCodebook) and ad.num_codebooks == 3
    with pytest.raises(ValueError):
        make_codec({"kind": "nope"})
    with pytest.raises(ValueError):
        make_codec({"k": 4})
    with pytest.raises(ValueError):
        make_codec({"kind": "vq", "k": 4, "bogus": 1})


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        AffineRTN(4, 64).encode(_randn((4, 96)), seed=0)  # 96 % 64
    with pytest.raises(ValueError):
        VectorCodebook(4, 8, rotate=False).encode(_randn((4, 12)), seed=0)
    with pytest.raises(ValueError):
        ScalarCodebook(1)
    with pytest.raises(ValueError):
        BF16Codec().encode(mx.zeros((4,)), seed=0)
    with pytest.raises(ValueError):
        VectorCodebook(8, 8).decode(BF16Codec().encode(_randn((4, 8)), seed=0))


def test_rotation_falls_back_to_input_side_when_output_dim_unsupported():
    # embeddings / LM heads: the vocabulary size (e.g. 151936) has no Hadamard; only V (input side) is applied
    mx.random.seed(3)
    w = mx.random.normal((72, 64))  # 72 = 9 * 8: unsupported output size, 64 supported
    for codec in (VectorCodebook(k=64, block_size=8, rotate=True), AdditiveCodebook(k=64, block_size=8, num_codebooks=2)):
        e = codec.encode(w, seed=1)
        assert e.params["rotate"] and e.params["rotate_in"] and not e.params["rotate_out"]
        err = float(mx.mean((codec.decode(e) - w) ** 2) / mx.mean(w ** 2))
        assert err < 0.6
    big = VectorCodebook(k=72 * 8, block_size=8, rotate=True, fit_blocks=72 * 8)
    assert float(mx.max(mx.abs(big.decode(big.encode(w, seed=0)) - w))) < 1e-2  # exact up to fp16 codebook rounding


def test_rotation_needs_supported_input_dim():
    with pytest.raises(ValueError, match="rotation"):
        VectorCodebook(k=16, block_size=8, rotate=True).encode(mx.random.normal((64, 72)), seed=0)


# ------------------------------------------------------------------ explicit rotation seeds (rotation groups)
@pytest.mark.parametrize("make", [lambda: VectorCodebook(16, block_size=4, kmeans_iters=3),
                                  lambda: AdditiveCodebook(16, block_size=4, num_codebooks=2, kmeans_iters=3)])
def test_explicit_rotation_seeds_override_and_roundtrip(make) -> None:
    w = _randn((64, 128), seed=4)
    c = make()
    base = c.encode(w, seed=7)
    e = c.encode(w, seed=7, seed_in=1234)
    assert e.params["seed_in"] == 1234 and e.params["seed_out"] == base.params["seed_out"]
    e2 = c.encode(w, seed=7, seed_in=1234, seed_out=99)
    assert e2.params["seed_out"] == 99
    # decode only uses the stored seeds; quality is statistically unchanged
    assert abs(rel_mse(w, c.decode(e)) - rel_mse(w, c.decode(base))) < 0.25 * rel_mse(w, c.decode(base))
    same = c.encode(w, seed=7, seed_in=None, seed_out=None)
    assert same.params == base.params and np.array_equal(np.array(same.arrays["indices"]),
                                                         np.array(base.arrays["indices"]))


def test_explicit_seed_in_with_unrotated_output_and_residual() -> None:
    from suc.quant.residual import ResidualCodec
    w = _randn((36, 64), seed=5)   # 36 rows: no Hadamard → rotate_out False
    e = VectorCodebook(16, block_size=4, kmeans_iters=3).encode(w, seed=1, seed_in=55, seed_out=66)
    assert e.params["seed_in"] == 55 and e.params["seed_out"] is None and not e.params["rotate_out"]
    r = ResidualCodec(VectorCodebook(16, block_size=4, kmeans_iters=3), 0.02)
    er = r.encode(w, seed=1, seed_in=55)
    assert er.base().params["seed_in"] == 55
    assert rel_mse(w, r.decode(er)) < rel_mse(w, VectorCodebook(16, block_size=4, kmeans_iters=3).decode(e))
    # a base without rotation silently ignores the seeds (rotation groups may mix codecs)
    ra = ResidualCodec(AffineRTN(4), 0.02)
    assert ra.decode(ra.encode(w, seed=1, seed_in=55)).shape == w.shape
    with pytest.raises(TypeError):
        r.encode(w, seed=1, bogus=3)


# ------------------------------------------------------------------ rotation sides (one-sided / input-free)
@pytest.mark.parametrize("cls", [VectorCodebook, AdditiveCodebook])
@pytest.mark.parametrize("rin,rout", [(True, False), (False, True), (False, False)])
def test_rotation_sides_roundtrip_and_params(cls, rin, rout) -> None:
    w = _randn((16, 32), 21)
    kw = {"num_codebooks": 2} if cls is AdditiveCodebook else {}
    c = cls(64, 8, rotate=True, kmeans_iters=10, rotate_in=rin, rotate_out=rout, **kw)  # 64 blocks: exact codes
    e = c.encode(w, seed=4)
    p = e.params
    assert p["rotate"] == (rin or rout)
    if rin or rout:
        assert p["rotate_in"] == rin and p["rotate_out"] == rout
        assert (p["seed_in"] is None) == (not rin) and (p["seed_out"] is None) == (not rout)
    np.testing.assert_allclose(np.array(c.decode(e)), np.array(w), atol=5e-3)


def test_rotate_in_false_stores_output_rotated_weight() -> None:
    from suc.quant.hadamard import rht
    w = _randn((16, 32), 22)
    c = VectorCodebook(64, 8, rotate=True, kmeans_iters=10, rotate_in=False)
    e = c.encode(w, seed=1)
    stored = c._sum_codes(e).reshape(16, 32) * e.arrays["row_scales"].astype(mx.float32)[:, None]
    np.testing.assert_allclose(np.array(stored), np.array(rht(w.T, e.params["seed_out"]).T), atol=5e-3)


def test_rotate_in_false_allows_unsupported_input_size_and_uses_col_weights() -> None:
    w = _randn((16, 72), 23)    # 72: no Hadamard on the input side — fine when V is not applied
    cw = mx.array(np.linspace(0.1, 10, 72).astype(np.float32))
    c = VectorCodebook(8, 8, rotate=True, kmeans_iters=5, rotate_in=False)
    e = c.encode(w, seed=0, col_weights=cw)
    assert not e.params["rotate_in"] and e.params["rotate_out"]
    e2 = c.encode(w, seed=0)
    assert not mx.array_equal(e.arrays["indices"], e2.arrays["indices"])   # the column metric was used


def test_make_codec_rotation_sides() -> None:
    c = make_codec({"kind": "additive", "k": 16, "block_size": 8, "num_codebooks": 2, "rotate_out": False})
    assert c.rotate and c.rotate_in and not c.rotate_out
