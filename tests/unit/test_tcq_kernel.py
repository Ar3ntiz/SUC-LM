"""TCQ Metal gemv (suc.kernels.tcq_matmul) vs the Python reference decoder: max / mean abs and relative error."""
import mlx.core as mx
import pytest

from suc.kernels.tcq_matmul import pack_encoded, tcq_matmul_metal, tcq_matmul_reference
from suc.quant.trellis import TrellisCodec


@pytest.mark.parametrize("code,k,V", [("1mad", 2, 1), ("1mad", 2, 2), ("3inst", 2, 1), ("3inst", 2, 2), ("pair3", 2, 2),
                                      ("1mad", 2.5, 2), ("1mad", 3, 1), ("pair3", 3, 2)])
@pytest.mark.parametrize("batch", [1, 3, 9])
def test_tcq_kernel_matches_reference(code, k, V, batch):
    L = 10
    w = mx.random.normal((24, 512), key=mx.random.key(1))
    e = TrellisCodec(k=k, L=L, code=code, V=V, rotate=False).encode(w, seed=0)
    x = mx.random.normal((batch, 512), key=mx.random.key(2))
    ref = tcq_matmul_reference(x, e)
    y = tcq_matmul_metal(x, pack_encoded(e), e.arrays["row_scales"], k=k, L=L, V=V, T=256, code=code,
                         gain=e.params["gain"])
    err = mx.abs(y - ref)
    rel = float(mx.sqrt(mx.sum(err ** 2) / mx.sum(ref ** 2)))
    assert y.shape == ref.shape and rel < 1e-5, (float(err.max()), float(err.mean()), rel)
