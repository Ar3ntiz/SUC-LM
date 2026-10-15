"""Cross-check of the C++ reference decoder (cpp/, tool suc1-decode) against the Python reference decoder
SucModel.decode_tensor: bit-exact float32 equality for every codec / representation, plus the real dev container.

Skipped unless the C++ tools are built (cpp/build.sh or CMake) into $SUC_CPP_BUILD (default ~/suc-local/cpp-build).
The dev-container test is marked `model` (needs ~/suc-local/compress/compress_qwen3_0.6b/container/model.suc1 or
$SUC1_DEV_FILE).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from suc.io.suc1 import SucModel, write_suc1
from suc.quant.codecs import AdditiveCodebook, AffineRTN, BF16Codec, ScalarCodebook, VectorCodebook
from suc.quant.residual import ResidualCodec

BUILD = Path(os.environ.get("SUC_CPP_BUILD", Path.home() / "suc-local" / "cpp-build"))
DECODE, INSPECT, UNIT = BUILD / "suc1-decode", BUILD / "suc1-inspect", BUILD / "suc-tests"
DEV_FILE = Path(os.environ.get("SUC1_DEV_FILE", Path.home() / "suc-local/compress/compress_qwen3_0.6b/container/model.suc1"))

pytestmark = pytest.mark.skipif(not DECODE.exists(), reason=f"C++ tools not built in {BUILD} (run cpp/build.sh)")

ARCH = {"model_type": "qwen3", "hidden_size": 64}


def _randn(shape, seed=0, scale=0.02) -> mx.array:
    return mx.array((scale * np.random.default_rng(seed).standard_normal(shape)).astype(np.float32))


def _imp(shape, seed=0) -> mx.array:
    return mx.array(np.random.default_rng(seed).exponential(1.0, shape).astype(np.float32))


# name -> (codec, weight shape). Covers every representation, index widths 1..10 bits (packed at exact width),
# rotations in_out / in / out, Hadamard sizes 2^k, m*2^k (m = 12, 20, 28) and > 8192, and every residual type.
CASES = {
    "bf16": (BF16Codec(), (64, 128)),
    "affine4": (AffineRTN(4, 64), (64, 128)),
    "affine3": (AffineRTN(3, 32), (64, 128)),
    "affine2_g128": (AffineRTN(2, 128), (32, 256)),
    "affine8": (AffineRTN(8, 64), (16, 128)),
    "scalar_tensor": (ScalarCodebook(16, kmeans_iters=3), (64, 128)),
    "scalar_rowgroup": (ScalarCodebook(8, scope="row_group", row_group=16, kmeans_iters=3), (64, 128)),
    "scalar_rowgroup_ragged": (ScalarCodebook(4, scope="row_group", row_group=24, kmeans_iters=2), (64, 128)),
    "scalar_k512": (ScalarCodebook(512, kmeans_iters=2), (64, 128)),
    "vq_k2_1bit": (VectorCodebook(2, block_size=8, rotate=True, kmeans_iters=2), (64, 128)),
    "vq_k8_3bit_g4": (VectorCodebook(8, block_size=4, rotate=True, kmeans_iters=2), (64, 128)),
    "vq_rot": (VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=3), (64, 128)),
    "vq_norot": (VectorCodebook(32, block_size=4, rotate=False, kmeans_iters=3), (64, 128)),
    "vq_k512_9bit": (VectorCodebook(512, block_size=4, rotate=True, kmeans_iters=1), (64, 128)),
    "vq_k1024_10bit_g2": (VectorCodebook(1024, block_size=2, rotate=False, kmeans_iters=1), (64, 128)),
    "vq_rot_in_only": (VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2), (72, 64)),   # 72: no Hadamard
    "vq_rot_onesided": (VectorCodebook(16, block_size=8, rotate=True, rotate_out=False, kmeans_iters=2), (64, 128)),
    "vq_rot_m12": (VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2), (96, 3072)),  # 12*8 x 12*256
    "vq_rot_16384": (VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=1), (4, 16384)),  # split 2^14
    "additive_rot": (AdditiveCodebook(16, block_size=8, num_codebooks=3, rotate=True, refine_iters=1, kmeans_iters=3),
                     (64, 128)),
    "additive_k64_6bit": (AdditiveCodebook(64, block_size=8, num_codebooks=4, rotate=True, kmeans_iters=2), (64, 256)),
    "res_fp16_vq": (ResidualCodec(VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=3), 0.02), (64, 128)),
    "res_int8_additive": (ResidualCodec(AdditiveCodebook(16, block_size=8, num_codebooks=2, kmeans_iters=3), 0.01,
                                        "int8"), (64, 128)),
    "res_fp16_affine": (ResidualCodec(AffineRTN(2, 64), 0.005), (64, 128)),
    "res_int8_scalar": (ResidualCodec(ScalarCodebook(16, kmeans_iters=2), 0.01, "int8"), (64, 128)),
    "res_fp16_bf16": (ResidualCodec(BF16Codec(), 0.01), (32, 64)),
    "res_zero": (ResidualCodec(VectorCodebook(16, block_size=8, kmeans_iters=3), 0.0, "int8"), (64, 128)),
    "res_int8_importance": (ResidualCodec(VectorCodebook(16, block_size=8, kmeans_iters=2), 0.01, "int8",
                                          "importance"), (64, 128)),
    "cols_int8": (ResidualCodec(VectorCodebook(16, block_size=8, kmeans_iters=2), 0.02, "int8", "importance",
                                structure="columns"), (64, 256)),
    "cols_fp16": (ResidualCodec(VectorCodebook(16, block_size=8, kmeans_iters=2), 0.05, "fp16", "magnitude",
                                structure="columns"), (64, 256)),
    "cols_int8_tail": (ResidualCodec(AdditiveCodebook(16, block_size=8, num_codebooks=2, kmeans_iters=2), 0.02, "int8",
                                     "importance", structure="columns", column_share=0.5), (64, 256)),
    "cols_fp16_tail_affine": (ResidualCodec(AffineRTN(4, 64), 0.03, "fp16", "magnitude", structure="columns",
                                            column_share=0.6), (64, 256)),
}

# The H_m step of the Hadamard transform is a sequential fma chain in both decoders (suc.quant.hadamard._mix,
# cpp/src/hadamard.cpp), so m = 20 / 28 sizes are bit-exact too. 5120 = 20*256 (Qwen3-14B/32B hidden), 7168 = 28*256.
CASES.update({
    "vq_rot_m20": (VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2), (40, 640)),    # 20*2 x 20*32
    "vq_rot_m28": (VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2), (56, 448)),    # 28*2 x 28*16
    "vq_rot_5120": (VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=1), (20, 5120)),
    "vq_rot_7168": (VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=1), (28, 7168)),
    "vq_rot_5120_out": (VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=1), (5120, 16)),
    "vq_rot_out_only": (VectorCodebook(16, block_size=8, rotate=True, rotate_in=False, kmeans_iters=2), (64, 128)),
    # m = 17 Hartley factor (suc.quant.hadamard): 544 = 17 * 32 on both sides; 17408 = 17 * 2^10 (Qwen3.8-27B MLP)
    "vq_rot_m17": (VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2), (544, 544)),
    "vq_g2_k256_onesided_cols_17408": (ResidualCodec(VectorCodebook(256, block_size=2, rotate=True, rotate_out=False,
                                                                    kmeans_iters=1), 0.005, "int8", "importance",
                                                    structure="columns"), (8, 17408)),
    # low-dimensional single-codebook VQ of the final configurations: one-sided, G = 2 / 3, 8- / 9-bit indices,
    # with column residuals (96 = 12 * 8: Hadamard size divisible by 3)
    "vq_g2_k256_onesided_cols": (ResidualCodec(VectorCodebook(256, block_size=2, rotate=True, rotate_out=False,
                                                              kmeans_iters=1), 0.01, "int8", "importance",
                                               structure="columns"), (64, 256)),
    "vq_g3_k512_onesided_cols_tail": (ResidualCodec(VectorCodebook(512, block_size=3, rotate=True, rotate_out=False,
                                                                   kmeans_iters=1), 0.02, "int8", "importance",
                                                    structure="columns", column_share=0.5), (64, 96)),
})


def _encode(codec, shape, seed):
    w = _randn(shape, seed)
    kw = {"importance": _imp(shape, seed)} if getattr(codec, "selection", None) == "importance" else {}
    return codec.encode(w, seed=seed, **kw)


def _raw_tensors() -> dict:
    return {"raw.bf16": _randn((5, 7), 3).astype(mx.bfloat16), "raw.f32": _randn((11,), 4),
            "raw.f16": _randn((2, 3, 4), 5).astype(mx.float16), "raw.i32": mx.arange(-5, 5, dtype=mx.int32),
            "raw.u8": mx.array(np.arange(9, dtype=np.uint8)), "raw.scalar": mx.array(1.5),
            "raw.bool": mx.array([True, False, True]), "raw.i8": mx.array(np.arange(-4, 4, dtype=np.int8))}


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _cpp_decode(path: Path, out: Path) -> None:
    r = subprocess.run([str(DECODE), str(path), "--all", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _write(tmp_path: Path, tensors: dict, name: str) -> Path:
    p = tmp_path / name
    write_suc1(p, architecture=ARCH, tokenizer_id="tok", tensors=tensors)
    return p


def test_cpp_unit_tests() -> None:
    if not UNIT.exists():
        pytest.skip("suc-tests not built")
    r = subprocess.run([str(UNIT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_bit_exact_every_codec(tmp_path: Path) -> None:
    tensors = {f"t.{k}": _encode(c, shape, i + 1) for i, (k, (c, shape)) in enumerate(sorted(CASES.items()))}
    tensors.update(_raw_tensors())
    p = _write(tmp_path, tensors, "all.suc1")
    out = tmp_path / "cpp"
    _cpp_decode(p, out)
    m = SucModel.load(p)
    bad = []
    for name in m.names():
        ref = np.array(m.decode_tensor(name))
        got = np.load(out / f"{name}.npy")
        assert got.dtype == np.float32 and got.shape == ref.shape, name
        if not np.array_equal(_bits(got), _bits(ref)):
            bad.append((name, float(np.abs(got.astype(np.float64) - ref).max())))
    assert not bad, f"C++ decode differs from SucModel.decode_tensor: {bad}"


def test_inspect_runs(tmp_path: Path) -> None:
    p = _write(tmp_path, {"a": _encode(*CASES["cols_int8_tail"], 1), "b": _raw_tensors()["raw.f16"]}, "i.suc1")
    r = subprocess.run([str(INSPECT), str(p), "--tensors", "--codebook-hashes"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "checksum         OK" in r.stdout and "(consistent)" in r.stdout
    bits = SucModel.load(p).bits()
    assert f"total      {bits['total'] // 8:>14}" in r.stdout


def test_cpp_rejects_corruption(tmp_path: Path) -> None:
    p = _write(tmp_path, {"a": _encode(*CASES["vq_rot"], 1)}, "c.suc1")
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0xFF
    q = tmp_path / "corrupt.suc1"
    q.write_bytes(bytes(b))
    r = subprocess.run([str(DECODE), str(q), "--all"], capture_output=True, text=True)
    assert r.returncode != 0 and "checksum mismatch" in r.stderr


def test_out_only_rotation_matches_codec(tmp_path: Path) -> None:
    """rotate_in=False (output-side rotation only, descriptor rotation = "out"): C++ == SucModel == codec.decode."""
    codec = VectorCodebook(16, block_size=8, rotate=True, rotate_in=False, kmeans_iters=2)
    e = codec.encode(_randn((64, 128), 2), seed=5)
    assert e.params["rotate_in"] is False and e.params["seed_in"] is None
    p = _write(tmp_path, {"w": e}, "o.suc1")
    assert SucModel.load(p).descriptor("w")["rotation"] == "out"
    _cpp_decode(p, tmp_path / "cpp")
    got = np.load(tmp_path / "cpp" / "w.npy")
    assert np.array_equal(_bits(got), _bits(np.array(codec.decode(e))))
    assert np.array_equal(_bits(got), _bits(np.array(SucModel.load(p).decode_tensor("w"))))


@pytest.mark.model
def test_dev_container_bit_exact() -> None:
    if not DEV_FILE.exists():
        pytest.skip(f"{DEV_FILE} not found")
    t0 = time.perf_counter()
    r = subprocess.run([str(DECODE), str(DEV_FILE), "--all", "--digest"], capture_output=True, text=True)
    cpp_s = time.perf_counter() - t0
    assert r.returncode == 0, r.stderr
    digests = dict(reversed(line.split(" ", 1)) for line in r.stdout.splitlines())
    t0 = time.perf_counter()
    m = SucModel.load(DEV_FILE)
    assert sorted(digests) == sorted(m.names())
    bad = [n for n in m.names()
           if hashlib.sha256(_bits(np.array(m.decode_tensor(n))).tobytes()).hexdigest() != digests[n]]
    py_s = time.perf_counter() - t0
    print(f"\ndev container: {len(digests)} tensors; C++ open+decode {cpp_s:.2f} s ({r.stderr.strip()}); "
          f"Python load+decode {py_s:.2f} s")
    assert not bad, f"{len(bad)} tensors differ, e.g. {bad[:5]}"


def test_mxfp4_native_bit_exact(tmp_path: Path) -> None:
    """Audit F05: native MXFP4 tensors (verbatim E2M1 codes + E8M0 scales) decode in C++ bit-exactly like
    SucModel.decode_tensor and MXFP4Native.decode: every nibble code, several rows / groups, extreme valid scales."""
    from suc.quant.codecs import MXFP4Native

    rng = np.random.default_rng(7)
    out, inn = 6, 128
    packed = rng.integers(0, 256, (out, inn // 2), dtype=np.uint8)
    packed[0, :8] = np.arange(16, dtype=np.uint8)[0::2] | (np.arange(16, dtype=np.uint8)[1::2] << 4)
    scales = rng.integers(2, 252, (out, inn // 32), dtype=np.uint8)
    scales[1, :2] = (2, 251)                              # suc.io.quantized_source.SCALE_RANGE["mxfp4"] limits
    codec = MXFP4Native()
    e = codec.encode_source(mx.array(packed), mx.array(scales))
    e_int8 = codec.encode_source(mx.array(packed.view(np.int8)), mx.array(scales))   # checkpoint int8 view
    p = _write(tmp_path, {"w": e, "w8": e_int8}, "mx.suc1")
    _cpp_decode(p, tmp_path / "cpp")
    m = SucModel.load(p)
    for name in ("w", "w8"):
        got = np.load(tmp_path / "cpp" / f"{name}.npy")
        assert np.array_equal(_bits(got), _bits(np.array(m.decode_tensor(name)))), name
        assert np.array_equal(_bits(got), _bits(np.array(codec.decode(e)))), name
