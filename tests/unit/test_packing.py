import mlx.core as mx
import numpy as np
import pytest

from suc.quant.packing import pack_indices, unpack_indices


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8, 10, 12])
@pytest.mark.parametrize("n", [1, 7, 31, 32, 33, 100, 1024])
def test_round_trip(bits, n):
    rng = np.random.default_rng(bits * 1000 + n)
    idx = rng.integers(0, 2 ** bits, size=(5, n), dtype=np.uint32)
    packed = pack_indices(mx.array(idx), bits)
    assert packed.dtype == mx.uint32 and packed.shape == (5, -(-n * bits // 32))
    out = unpack_indices(packed, bits, n)
    assert out.dtype == mx.uint32 and out.shape == (5, n)
    np.testing.assert_array_equal(np.array(out), idx)


def test_little_endian_bit_order():
    # 11 3-bit values: element j occupies stream bits [3j, 3j+3), LSB first; element 10 straddles words
    idx = ((np.arange(11, dtype=np.uint32) * 5 + 3) % 8)[None]
    packed = np.array(pack_indices(mx.array(idx), 3))
    stream = sum(int(v) << (3 * j) for j, v in enumerate(idx[0]))
    assert packed.shape == (1, 2)
    assert int(packed[0, 0]) == stream & 0xFFFFFFFF and int(packed[0, 1]) == stream >> 32


def test_rows_padded_independently():
    idx = np.full((3, 5), 7, dtype=np.uint32)  # 15 bits per row, one word per row
    packed = np.array(pack_indices(mx.array(idx), 3))
    assert packed.shape == (3, 1) and (packed == (1 << 15) - 1).all()


def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        pack_indices(mx.array(np.array([[4]], dtype=np.uint32)), 2)
