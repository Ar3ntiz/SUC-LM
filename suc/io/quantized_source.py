"""Lossless dequantization of quantized-at-source checkpoints (D8 exception, docs/arch/deepseek_v4.md §4).

Some frontier checkpoints are published only in a quantization-aware-trained low-precision format; the full-precision
reference of every SUC measurement is then the checkpoint DEQUANTIZED EXACTLY to BF16, one tensor (one expert) at a
time. Two formats (DeepSeek-V4 reference implementation, inference/model.py + convert.py):

  fp8_block  weight F8_E4M3 [out, in] + scale F8_E8M0 [ceil(out/B0), ceil(in/B1)] (B = 128 x 128):
             W[i, j] = e4m3(W8[i, j]) * 2^(S[i // B0, j // B1] - 127)
  mxfp4      weight I8 [out, in/2] (two E2M1 nibbles per byte: element 2c in the LOW nibble, 2c+1 in the high one)
             + scale F8_E8M0 [out, in/32]:  W[i, k] = E2M1(nib(i, k)) * 2^(S[i, k // 32] - 127)

Exactness. E4M3 values have at most 4 significant bits and E2M1 at most 2, BF16 has 8, and a power-of-two scale only
moves the exponent, so the product is exactly representable in BF16 as long as it stays in the BF16 NORMAL range
(Metal may flush subnormals). check_scales() enforces that from the scale tensor alone: every E8M0 exponent must lie
in a range for which the largest and the smallest non-zero element of the format stay normal and finite; anything
else raises NotLossless. The products are formed in float32 (exact for the same reason) and rounded once to BF16
(a no-op on exact values). Tests compare bit for bit with a numpy port of convert.py (tests/unit/test_deepseek_v4.py).

A third format (GLM-5.3-Flash, docs/arch/glm5_next.md §4; transformers integrations/finegrained_fp8.py
Fp8Dequantize, mlx-lm deepseek_v32 sanitize):

  fp8_block_f32  weight F8_E4M3 [out, in] + scale `weight_scale_inv` F32 [ceil(out/128), ceil(in/128)], a MULTIPLIER
             despite its name:  W[i, j] = bf16( float32(e4m3(W8[i, j])) * S[i // 128, j // 128] )
Here the scale is an arbitrary float32, so the product is NOT exactly representable in BF16 in general. The reference
of every SUC measurement is then the dequantization the reference implementations themselves perform: the float32
product (one IEEE multiply, identical in numpy / torch / MLX) rounded once to BF16 (round to nearest even). That is
what dequant_fp8_block_f32 returns (bit-identical to transformers' FP8 dequantize to bfloat16, tested against a numpy
port); check_scales("fp8_block_f32") only rejects non-finite or non-positive scales and reports the range.
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

FP8_BLOCK = (128, 128)
MXFP4_GROUP = 32
# E2M1 code -> value (convert.py FP4_TABLE): sign bit 3, magnitude [0, .5, 1, 1.5, 2, 3, 4, 6]
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
# E8M0 exponent ranges for which every element * 2^(s-127) is a finite NORMAL bf16 (or zero):
#   e4m3: smallest non-zero 2^-9, largest 448 < 2^9  ->  2^(s-136) >= 2^-126 and 2^(s-118) < 2^128
#   e2m1: smallest non-zero 0.5, largest 6 < 2^3     ->  2^(s-128) >= 2^-126 and 2^(s-124) < 2^128
SCALE_RANGE = {"fp8_block": (10, 245), "mxfp4": (2, 251)}
FLOAT_SCALE_KINDS = ("fp8_block_f32",)      # float32 multiplier scales (no exponent range: see the module docstring)


class NotLossless(ValueError):
    """The scale exponents of a tensor would make its BF16 dequantization inexact (or are NaN / out of range)."""


def e8m0_to_float32(s: mx.array) -> mx.array:
    """2^(s - 127) for uint8 exponents 1..254 (bit construction: exact)."""
    return (s.astype(mx.uint32) << 23).view(mx.float32)


def check_scales(kind: str, scale_u8: mx.array, name: str = "") -> tuple[int, int] | tuple[float, float]:
    """(min, max) exponent of the scale tensor; NotLossless when outside SCALE_RANGE[kind] (255 = NaN in E8M0).
    Float kinds (fp8_block_f32): (min, max) scale value; NotLossless when a scale is not finite or not positive."""
    if kind in FLOAT_SCALE_KINDS:
        s = scale_u8.astype(mx.float32)
        mn, mxv = float(s.min().item()), float(s.max().item())
        if not (math.isfinite(mn) and math.isfinite(mxv) and mn > 0):
            raise NotLossless(f"{name}: {kind} scales span [{mn}, {mxv}] (must be finite and positive)")
        return mn, mxv
    lo, hi = SCALE_RANGE[kind]
    s = scale_u8.astype(mx.int32)
    mn, mxv = int(s.min().item()), int(s.max().item())
    if mn < lo or mxv > hi:
        raise NotLossless(f"{name}: {kind} scale exponents span [{mn}, {mxv}], outside [{lo}, {hi}] where the BF16 "
                          "dequantization is exact (finite, normal)")
    return mn, mxv


def dequant_fp8_block(w_u8: mx.array, scale_u8: mx.array, block: tuple[int, int] = FP8_BLOCK,
                      dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """E4M3 weight (uint8 view, [out, in]) x per-block E8M0 scale -> dtype (lazy; no scale-range check)."""
    out, inn = w_u8.shape
    b0, b1 = block
    if tuple(scale_u8.shape) != (-(-out // b0), -(-inn // b1)):
        raise ValueError(f"fp8 scale shape {tuple(scale_u8.shape)} does not match weight {tuple(w_u8.shape)} with "
                         f"block {block}")
    w = mx.from_fp8(w_u8, dtype=mx.float32)
    s = e8m0_to_float32(scale_u8)
    if out % b0 == 0 and inn % b1 == 0:
        y = (w.reshape(out // b0, b0, inn // b1, b1) * s[:, None, :, None]).reshape(out, inn)
    else:
        y = w * mx.repeat(mx.repeat(s, b0, axis=0), b1, axis=1)[:out, :inn]
    return y.astype(dtype)


def dequant_fp8_block_f32(w_u8: mx.array, scale_f32: mx.array, block: tuple[int, int] = FP8_BLOCK,
                          dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """E4M3 weight (uint8 view, [out, in]) x per-block float32 multiplier (`weight_scale_inv`) -> dtype: the float32
    product rounded once (lazy; no scale check)."""
    out, inn = w_u8.shape
    b0, b1 = block
    if tuple(scale_f32.shape) != (-(-out // b0), -(-inn // b1)):
        raise ValueError(f"fp8 scale shape {tuple(scale_f32.shape)} does not match weight {tuple(w_u8.shape)} with "
                         f"block {block}")
    w = mx.from_fp8(w_u8.view(mx.uint8) if w_u8.dtype != mx.uint8 else w_u8, dtype=mx.float32)
    s = scale_f32.astype(mx.float32)
    if out % b0 == 0 and inn % b1 == 0:
        y = (w.reshape(out // b0, b0, inn // b1, b1) * s[:, None, :, None]).reshape(out, inn)
    else:
        y = w * mx.repeat(mx.repeat(s, b0, axis=0), b1, axis=1)[:out, :inn]
    return y.astype(dtype)


def dequant_mxfp4(w_packed: mx.array, scale_u8: mx.array, group: int = MXFP4_GROUP,
                  dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Packed E2M1 weight ([out, in/2], int8 / uint8 bytes, low nibble first) x E8M0 scale per `group` inputs of a row
    -> dtype [out, in] (lazy; no scale-range check)."""
    out, half = w_packed.shape
    inn = 2 * half
    if tuple(scale_u8.shape) != (out, inn // group):
        raise ValueError(f"mxfp4 scale shape {tuple(scale_u8.shape)} does not match weight [{out}, {inn}] / {group}")
    b = w_packed.view(mx.uint8) if w_packed.dtype == mx.int8 else w_packed
    codes = mx.stack([b & 0x0F, b >> 4], axis=-1).reshape(out, inn)
    vals = mx.take(mx.array(FP4_VALUES, dtype=mx.float32), codes.astype(mx.uint32))
    s = e8m0_to_float32(scale_u8)
    y = (vals.reshape(out, inn // group, group) * s[:, :, None]).reshape(out, inn)
    return y.astype(dtype)


def dequantize(kind: str, weight: mx.array, scale: mx.array, *, name: str = "", check: bool = True,
               dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """One tensor of `kind` ("fp8_block" | "mxfp4" | "fp8_block_f32") from its raw weight bytes and scales;
    check=True verifies the scales first (NotLossless)."""
    if check:
        check_scales(kind, scale, name)
    if kind == "fp8_block":
        return dequant_fp8_block(weight, scale, dtype=dtype)
    if kind == "fp8_block_f32":
        return dequant_fp8_block_f32(weight, scale, dtype=dtype)
    if kind == "mxfp4":
        return dequant_mxfp4(weight, scale, dtype=dtype)
    raise ValueError(f"unknown source quantization {kind!r}")


# ------------------------------------------------------------------------------------------- numpy references

def e4m3_table() -> np.ndarray:
    """float64 value of every E4M3FN byte (bias 7, no infinities, 0x7F / 0xFF = NaN), from the bit fields."""
    out = np.empty(256, dtype=np.float64)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 0x7
        if e == 0xF and m == 0x7:
            out[b] = np.nan
        elif e == 0:
            out[b] = sign * (m / 8.0) * 2.0 ** -6
        else:
            out[b] = sign * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return out


def np_dequant_fp8_block(w_u8: np.ndarray, scale_u8: np.ndarray, block: tuple[int, int] = FP8_BLOCK) -> np.ndarray:
    """numpy port of convert.py:126 in float64: weight * scale broadcast over 128 x 128 blocks."""
    w = e4m3_table()[w_u8.astype(np.int64)]
    s = np.ldexp(1.0, scale_u8.astype(np.int64) - 127)
    s = np.repeat(np.repeat(s, block[0], axis=0), block[1], axis=1)[: w.shape[0], : w.shape[1]]
    return w * s


def np_dequant_fp8_block_f32(w_u8: np.ndarray, scale_f32: np.ndarray,
                             block: tuple[int, int] = FP8_BLOCK) -> np.ndarray:
    """numpy port of transformers Fp8Dequantize._dequantize_one (F32 math, q * s), float32 [out, in] (round it to
    BF16 to get the reference weights)."""
    w = e4m3_table()[w_u8.astype(np.int64)].astype(np.float32)
    s = np.repeat(np.repeat(scale_f32.astype(np.float32), block[0], axis=0), block[1], axis=1)
    return (w * s[: w.shape[0], : w.shape[1]]).astype(np.float32)


def np_quantize_fp8_block_f32(w: np.ndarray, block: tuple[int, int] = FP8_BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """Reference quantizer for test checkpoints in the GLM layout: float32 block scale amax / 448 (not a power of two)
    + E4M3 bytes of w / scale."""
    out, inn = w.shape
    bo, bi = -(-out // block[0]), -(-inn // block[1])
    pad = np.zeros((bo * block[0], bi * block[1]), dtype=np.float32)
    pad[:out, :inn] = w
    s = (np.abs(pad.reshape(bo, block[0], bi, block[1])).max(axis=(1, 3)) / 448.0).astype(np.float32)
    s = np.maximum(s, np.float32(1e-12))
    full = np.repeat(np.repeat(s, block[0], 0), block[1], 1)[:out, :inn]
    q = np.array(mx.to_fp8(mx.array((w / full).astype(np.float32))))
    return q, s


def np_dequant_mxfp4(w_i8: np.ndarray, scale_u8: np.ndarray, group: int = MXFP4_GROUP) -> np.ndarray:
    """numpy port of convert.py:28-33 + the per-32 E8M0 scale (model.py:139-143) in float64."""
    x = w_i8.view(np.uint8)
    low, high = x & 0x0F, (x >> 4) & 0x0F
    v = np.stack([np.array(FP4_VALUES)[low], np.array(FP4_VALUES)[high]], axis=-1).reshape(x.shape[0], -1)
    return v * np.repeat(np.ldexp(1.0, scale_u8.astype(np.int64) - 127), group, axis=1)


def np_quantize_fp8_block(w: np.ndarray, block: tuple[int, int] = FP8_BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """Reference quantizer for test checkpoints: power-of-two block scale (amax / 448 rounded up) + E4M3 bytes."""
    out, inn = w.shape
    bo, bi = -(-out // block[0]), -(-inn // block[1])
    pad = np.zeros((bo * block[0], bi * block[1]), dtype=np.float32)
    pad[:out, :inn] = w
    amax = np.abs(pad.reshape(bo, block[0], bi, block[1])).max(axis=(1, 3))
    e = np.ceil(np.log2(np.maximum(amax, 1e-30) / 448.0)).astype(np.int64)
    s = np.clip(e + 127, 1, 254).astype(np.uint8)
    full = np.repeat(np.repeat(np.ldexp(1.0, s.astype(np.int64) - 127), block[0], 0), block[1], 1)[:out, :inn]
    q = np.array(mx.to_fp8(mx.array((w / full).astype(np.float32))))
    return q, s


def np_quantize_mxfp4(w: np.ndarray, group: int = MXFP4_GROUP) -> tuple[np.ndarray, np.ndarray]:
    """Reference quantizer for test checkpoints: power-of-two scale per `group` inputs (amax / 6 rounded up), nearest
    E2M1 code; returns (packed int8 [out, in/2], scale uint8 [out, in/group])."""
    out, inn = w.shape
    g = w.reshape(out, inn // group, group)
    e = np.ceil(np.log2(np.maximum(np.abs(g).max(-1), 1e-30) / 6.0)).astype(np.int64)
    s = np.clip(e + 127, 1, 254).astype(np.uint8)
    y = (g / np.ldexp(1.0, s.astype(np.int64) - 127)[..., None]).reshape(out, inn)
    mags = np.array(FP4_VALUES[:8])
    code = np.abs(np.abs(y)[..., None] - mags).argmin(-1) + 8 * (y < 0)
    code = code.astype(np.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8).view(np.int8)
    return packed, s


__all__ = ["FP8_BLOCK", "MXFP4_GROUP", "FP4_VALUES", "SCALE_RANGE", "FLOAT_SCALE_KINDS", "NotLossless",
           "e8m0_to_float32", "check_scales", "dequant_fp8_block", "dequant_fp8_block_f32", "dequant_mxfp4",
           "dequantize", "e4m3_table", "np_dequant_fp8_block", "np_dequant_fp8_block_f32", "np_dequant_mxfp4",
           "np_quantize_fp8_block", "np_quantize_fp8_block_f32", "np_quantize_mxfp4"]
