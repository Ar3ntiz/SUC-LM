"""Trellis-coded (tcq) tensors end to end on tiny shapes: exact bit accounting, the SUC1 `tcq` record (reference decoder
bit-identical to the codec decode), the Metal kernels (vec / mma / expert-stack prologue) and the runtime modules
(SucLinear / SucResidualLinear / SucSwitchLinear / mixed stacks) against the Python reference decoder."""
import mlx.core as mx
import numpy as np
import pytest

from suc.io.suc1 import SucModel, write_suc1
from suc.kernels.suc_linear import SucLinear
from suc.kernels.suc_switch import SucSwitchLinear, SucSwitchMixed, switch_from_encodings
from suc.kernels.tcq_matmul import (
    kernel_consts, pack_encoded, tcq_gather_matmul_metal, tcq_matmul_metal, tcq_matmul_reference,
)
from suc.quant.codecs import make_codec
from suc.quant.residual import ResidualCodec
from suc.quant.trellis import TrellisCodec, code_stats
from suc.runtime.compressed_model import module_from_encoded

ARCH = {"model_type": "qwen3", "hidden_size": 256}


def _w(out, inn, seed):
    return mx.array(np.random.default_rng(seed).normal(size=(out, inn)).astype(np.float32)) / np.sqrt(inn)


def _codec(code="3inst", k=2, V=2, L=10, T=128, residual=None):
    base = TrellisCodec(k=k, L=L, code=code, V=V, T=T, rotate=True, rotate_out=False)
    if residual is None:
        return base
    return ResidualCodec(base, residual, "int8", "magnitude", structure="columns")


def _rel(y, ref):
    d = y.astype(mx.float32) - ref
    return float(mx.sqrt(mx.sum(d * d) / mx.sum(ref * ref)))


# ------------------------------------------------------------------------------------------------ accounting

@pytest.mark.parametrize("cfg", [dict(code="3inst", k=2, V=2), dict(code="1mad", k=3, V=1), dict(code="3inst", k=3, V=2)])
@pytest.mark.parametrize("residual", [None, 0.01])
def test_bits_for_equals_encoded_bits(cfg, residual):
    c = _codec(L=8, residual=residual, **cfg)
    w = _w(24, 256, 0)
    e = c.encode(w, seed=3)
    assert c.bits_for(w.shape) == e.bits()["total"]
    p = e.params["base"] if residual else e.params
    assert p["index_bits"] == p["kv"] and p["code_mu"] == list(code_stats(cfg["code"], 8, cfg["V"])[0])


def test_decode_uses_stored_code_stats():
    c = _codec(L=8)
    e = c.encode(_w(8, 128, 1), seed=0)
    d0 = c.decode(e)
    e.params["code_mu"] = [m + 1.0 for m in e.params["code_mu"]]       # the stored numbers decide the values
    assert not mx.array_equal(c.decode(e), d0).item()


# ------------------------------------------------------------------------------------------------ SUC1 record

@pytest.mark.parametrize("residual", [None, 0.02])
@pytest.mark.parametrize("cfg", [dict(code="3inst", k=2, V=2), dict(code="1mad", k=3, V=1), dict(code="pair3", k=2, V=2)])
def test_suc1_record_roundtrip_and_reference_decoder(tmp_path, residual, cfg):
    c = _codec(L=10, residual=residual, **cfg)
    w = _w(40, 256, 2)
    e = c.encode(w, seed=7)
    path = tmp_path / "m.suc1"
    write_suc1(path, architecture=ARCH, tokenizer_id="t", tensors={"model.layers.0.mlp.down_proj.weight": e})
    sm = SucModel.load(path)
    n = "model.layers.0.mlp.down_proj.weight"
    d = sm.descriptor(n)
    assert d["representation_type"] == "tcq" and d["block_size"] == 128 and d["rotation"] == "in"
    back = sm.encoded(n)
    pre = "base." if residual else ""
    assert np.array_equal(np.array(back.arrays[pre + "chunks"]), np.array(e.arrays[pre + "chunks"]))
    assert mx.array_equal(sm.decode_tensor(n), c.decode(e)).item()         # bit-identical reference decoder
    rec = sm.record(n, pre + "chunks")
    kv = (e.params["base"] if residual else e.params)["kv"]
    assert rec["packed"]["bits"] == kv and rec["shape"] == [40 * 2, int(128 * cfg["k"]) // 32]
    # the stored rows are the kernel's streams: encoded_packed == pack_encoded
    pk = sm.encoded_packed(n)
    base = e.base() if residual else e
    assert np.array_equal(np.array(pk.arrays[pre + "packed_words"]), np.array(pack_encoded(base)))
    assert sm.bits()["indices"] == 40 * 256 * cfg["k"]


def test_suc1_rejects_hyb(tmp_path):
    e = TrellisCodec(k=2, L=8, code="hyb", Q=4, T=128, lut_iters=1, fit_seqs=16).encode(_w(4, 128, 0), seed=0)
    with pytest.raises(ValueError, match="compute codes"):
        write_suc1(tmp_path / "x.suc1", architecture=ARCH, tokenizer_id="t", tensors={"a.weight": e})


# ------------------------------------------------------------------------------------------------ kernels

@pytest.mark.parametrize("cfg", [dict(code="3inst", k=2, V=2), dict(code="1mad", k=3, V=1), dict(code="3inst", k=3, V=2),
                                 dict(code="1mad", k=2, V=1), dict(code="pair3", k=2, V=2)])
@pytest.mark.parametrize("batch,path", [(1, "auto"), (4, "auto"), (5, "auto"), (16, "auto"), (21, "auto"), (3, "mma"),
                                        (9, "vec")])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_kernel_paths_match_reference(cfg, batch, path, dtype):
    L, T = 10, 128
    e = TrellisCodec(L=L, T=T, rotate=False, **cfg).encode(_w(72, 512, 3), seed=1)
    x = mx.random.normal((batch, 512), key=mx.random.key(batch)).astype(dtype)
    ref = tcq_matmul_reference(x, e)
    y = tcq_matmul_metal(x, pack_encoded(e), e.arrays["row_scales"], k=cfg["k"], L=L, V=cfg["V"], T=T,
                         code=cfg["code"], cst=kernel_consts(e.params), path=path, out_dtype=mx.float32)
    tol = 1e-5 if dtype == mx.float32 and (path == "vec" or (path == "auto" and batch < 5)) else 3e-3
    assert y.shape == ref.shape and _rel(y, ref) < tol


@pytest.mark.parametrize("tile", [1, 3, 8, 16])
def test_gather_kernel_matches_per_expert(tile):
    codec = TrellisCodec(k=2, L=10, code="3inst", V=2, T=128, rotate=False)
    encs = [codec.encode(_w(40, 256, 10 + i), seed=i) for i in range(5)]
    words = mx.stack([pack_encoded(e) for e in encs])
    scales = mx.stack([e.arrays["row_scales"].astype(mx.float32) for e in encs])
    te = mx.array(np.array([3, 0, 3, 4], np.uint32))
    x = mx.random.normal((4 * tile, 256), key=mx.random.key(tile)).astype(mx.float16)
    y = tcq_gather_matmul_metal(x, words, scales, te, k=2, L=10, V=2, T=128, code="3inst",
                                cst=kernel_consts(encs[0].params), tile=tile, out_dtype=mx.float32)
    for t, ex in enumerate(np.array(te).tolist()):
        ref = tcq_matmul_metal(x[t * tile:(t + 1) * tile], words[ex], scales[ex], k=2, L=10, V=2, T=128,
                               code="3inst", cst=kernel_consts(encs[ex].params), out_dtype=mx.float32)
        assert mx.array_equal(y[t * tile:(t + 1) * tile], ref).item()      # same kernel body, same numerics


# ------------------------------------------------------------------------------------------------ modules

@pytest.mark.parametrize("residual", [None, 0.02])
@pytest.mark.parametrize("batch", [1, 7])
def test_linear_module_matches_decoded_weight(residual, batch):
    c = _codec(L=10, residual=residual)
    w = _w(48, 256, 4)
    e = c.encode(w, seed=2)
    m = module_from_encoded(e)
    assert isinstance(m, SucLinear) and m.tcq is not None and m.seed_in is not None
    x = mx.random.normal((batch, 256), key=mx.random.key(5))
    ref = x @ c.decode(e).T
    assert _rel(m(x), ref) < (2e-5 if batch < 5 else 3e-3)


@pytest.mark.parametrize("path", ["gather", "tiled", "grouped"])
def test_switch_module_matches_reference(path):
    c = _codec(L=10, residual=0.02)
    encs = [c.encode(_w(32, 256, 20 + i), seed=100 + i, seed_in=777) for i in range(6)]
    m = SucSwitchLinear.from_encodings(encs)
    assert m.tcq is not None and m.signs_in.shape[0] == 1          # shared stack rotation: one sign row
    dec = [c.decode(e) for e in encs]
    n = 24
    x = mx.random.normal((n, 256), key=mx.random.key(9))
    eidx = mx.array(np.random.default_rng(1).integers(0, 6, n).astype(np.uint32))
    ref = mx.stack([x[i] @ dec[k].T for i, k in enumerate(np.array(eidx).tolist())])
    y = m.rows(x, eidx, path=path)
    assert _rel(y, ref) < 3e-3


def test_mixed_stack_vq_and_tcq():
    vq = ResidualCodec(make_codec({"kind": "vq", "k": 16, "block_size": 4, "rotate": True, "rotate_out": False,
                                   "kmeans_iters": 3}), 0.02, "int8", "magnitude", structure="columns")
    tq = _codec(L=10, residual=0.02)
    encs = [(tq if i % 2 else vq).encode(_w(32, 256, 40 + i), seed=i) for i in range(4)]
    m = switch_from_encodings(encs)
    assert isinstance(m, SucSwitchMixed) and len(m.subs) == 2
    dec = [(tq if i % 2 else vq).decode(e) for i, e in enumerate(encs)]
    x = mx.random.normal((3, 1, 256), key=mx.random.key(2))
    idx = mx.array(np.array([[0, 1], [3, 2], [1, 1]], np.uint32))
    y = m(x[:, None], idx)
    ref = mx.stack([mx.stack([x[b, 0] @ dec[k].T for k in row]) for b, row in enumerate(np.array(idx).tolist())])
    assert _rel(y.reshape(ref.shape), ref) < 2e-5
