"""Native MXFP4 execution of pass-through tensors (suc.quant.codecs.MXFP4Native, SUC1 representation "mxfp4").

The stored layout IS MLX's mxfp4 quantized layout: the packed E2M1 bytes (two codes per byte, low nibble first) viewed
as little-endian uint32 words (8 codes per word, element j at bits 4j) and one uint8 E8M0 exponent per 32 inputs of a
row. So the codes run directly in MLX's own kernels (mx.quantized_matmul / mx.gather_qmm with mode="mxfp4"), no
custom Metal kernel and no repacking: the arrays are reinterpreted, not converted.

MXFP4Linear        nn.Linear replacement (one tensor).
MXFP4SwitchLinear  mlx-lm SwitchLinear replacement over E experts (gather_qmm, one launch for the selected experts);
                   also a sub-stack of suc.kernels.suc_switch.SucSwitchMixed (native experts next to SUC-coded ones).
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from suc.quant.codecs import MXFP4_GROUP, Encoded


def _words(e: Encoded) -> tuple[mx.array, mx.array]:
    """(uint32 [out, in/8], uint8 [out, in/32]) MLX mxfp4 operands of one native Encoded."""
    if e.codec != "mxfp4":
        raise ValueError(f"expected a native mxfp4 encoding, got {e.codec!r}")
    out, inn = e.shape
    if inn % MXFP4_GROUP:
        raise ValueError(f"in_features {inn} not divisible by {MXFP4_GROUP}")
    w = e.arrays["codes"].astype(mx.uint8).reshape(out, inn // 2).view(mx.uint32)     # reinterpret, no repack
    return w, e.arrays["scales"].astype(mx.uint8)


class MXFP4Linear(nn.Module):
    """y = x W^T (+ b) with W in native MXFP4 codes (MLX quantized_matmul, mode mxfp4)."""
    n_residual = 0

    def __init__(self, weight: mx.array, scales: mx.array, bias: mx.array | None = None):
        super().__init__()
        self.weight, self.scales = weight, scales
        if bias is not None:
            self.bias = bias
        self.in_features, self.out_features = int(weight.shape[-1]) * 8, int(weight.shape[-2])
        self.freeze()

    @classmethod
    def from_encoded(cls, e: Encoded, bias: mx.array | None = None) -> "MXFP4Linear":
        return cls(*_words(e), bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.quantized_matmul(x, self.weight, self.scales, transpose=True, group_size=MXFP4_GROUP, bits=4,
                                mode="mxfp4")
        return y + self.bias if "bias" in self else y


class MXFP4SwitchLinear(nn.Module):
    """Drop-in for mlx-lm SwitchLinear on E experts in native MXFP4 (mx.gather_qmm, mode mxfp4)."""
    n_residual = 0

    def __init__(self, weight: mx.array, scales: mx.array, bias: mx.array | None = None):
        super().__init__()
        if weight.ndim != 3 or scales.shape[:2] != weight.shape[:2]:
            raise ValueError(f"inconsistent stacks: weight {weight.shape}, scales {scales.shape}")
        self.weight, self.scales = weight, scales
        if bias is not None:
            self.bias = bias
        self.in_features, self.out_features = int(weight.shape[-1]) * 8, int(weight.shape[-2])
        self.freeze()

    @classmethod
    def from_encodings(cls, encodings: list[Encoded], bias: mx.array | None = None) -> "MXFP4SwitchLinear":
        if not encodings or len({tuple(e.shape) for e in encodings}) != 1:
            raise ValueError("MXFP4SwitchLinear: needs >= 1 expert encodings of one shape")
        ws, ss = zip(*(_words(e) for e in encodings))
        return cls(mx.stack(ws), mx.stack(ss), bias=bias)

    @property
    def num_experts(self) -> int:
        return int(self.weight.shape[0])

    @property
    def input_dims(self) -> int:
        return self.in_features

    @property
    def output_dims(self) -> int:
        return self.out_features

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        """Same contract as SwitchLinear: x (*bx, R, in), indices (*bi) -> (*broadcast, R, out)."""
        y = mx.gather_qmm(x, self.weight, self.scales, rhs_indices=indices, transpose=True, group_size=MXFP4_GROUP,
                          bits=4, mode="mxfp4", sorted_indices=sorted_indices)
        if "bias" in self:
            y = y + mx.expand_dims(self.bias[indices], -2)
        return y


__all__ = ["MXFP4Linear", "MXFP4SwitchLinear"]
