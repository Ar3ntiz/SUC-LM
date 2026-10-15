"""Central codec capability table (audit F07): which stage of the project supports which representation.

A codec can be available for research (`suc quantize`: encode + quality decode in MLX) without being exportable to a
SUC1 file or runnable by a runtime. `suc compress` produces a complete SUC1 file, so its configuration is gated on
`suc1_export` before any model loading / calibration (CompressPlanConfig); the runtime columns document what the
Python (`suc.runtime`) and C++ (`cpp/`) engines execute and are checked by tests/unit/test_capabilities.py against
the code paths that implement them.

Columns:
  encode / quality_decode   suc.quant.codecs / trellis encode + decode (research path, `suc quantize`)
  suc1_export               suc.io.suc1 writer / SucModel reader (container + Python reference decode)
  cpp_reference_decode      cpp/src/decode.cpp (suc1-decode, CPU)
  python_runtime            suc.runtime: "codes" (kernels run on the stored codes), "affine" (MLX quantized matmul),
                            "dense" (decoded once at load), False
  cpp_runtime               cpp Metal graphs: "codes", "affine", "dense", "experts" (expert stacks only), False
"""
from __future__ import annotations

# TCQ trellis codes: compute codes have a SUC1 layout and kernels; "hyb" (hashed LUT) is research-only.
TCQ_SUC1_CODES = ("1mad", "3inst", "pair3")

CAPABILITIES: dict[str, dict] = {
    "bf16":     dict(encode=True, quality_decode=True, suc1_export=True, cpp_reference_decode=True,
                     python_runtime="dense", cpp_runtime="dense"),
    "affine":   dict(encode=True, quality_decode=True, suc1_export=True, cpp_reference_decode=True,
                     python_runtime="affine", cpp_runtime="affine"),
    "scalar":   dict(encode=True, quality_decode=True, suc1_export=True, cpp_reference_decode=True,
                     python_runtime="dense", cpp_runtime=False),
    "vq":       dict(encode=True, quality_decode=True, suc1_export=True, cpp_reference_decode=True,
                     python_runtime="codes", cpp_runtime="codes"),
    "additive": dict(encode=True, quality_decode=True, suc1_export=True, cpp_reference_decode=True,
                     python_runtime="codes", cpp_runtime="codes"),
    "mxfp4":    dict(encode=True, quality_decode=True, suc1_export=True, cpp_reference_decode=True,
                     python_runtime="codes", cpp_runtime="experts"),
    "tcq":      dict(encode=True, quality_decode=True, suc1_export=True, cpp_reference_decode=True,
                     python_runtime="codes", cpp_runtime="codes"),
    "tcq:hyb":  dict(encode=True, quality_decode=True, suc1_export=False, cpp_reference_decode=False,
                     python_runtime=False, cpp_runtime=False),
}


def capability_key(codec) -> str:
    """Row of CAPABILITIES for a CodecConfig (duck-typed: .kind, .tcq_code)."""
    if codec.kind == "tcq" and getattr(codec, "tcq_code", None) not in TCQ_SUC1_CODES:
        return f"tcq:{codec.tcq_code}"
    return codec.kind


def capability(codec) -> dict:
    return CAPABILITIES[capability_key(codec)]


def export_errors(codec) -> list[str]:
    """Why `codec` cannot be written by `suc compress` (SUC1 export), as actionable messages; [] when it can."""
    errs = []
    key = capability_key(codec)
    if not CAPABILITIES[key]["suc1_export"]:
        errs.append(f"codec {codec.name!r} ({key}) has no SUC1 representation: it is research-only (use `suc quantize` "
                    f"to evaluate it); SUC1-exportable tcq codes are {list(TCQ_SUC1_CODES)}")
    if codec.kind == "tcq" and key != f"tcq:{codec.tcq_code}":
        v = codec.tcq_V if codec.tcq_V is not None else 1
        if v not in (1, 2):
            errs.append(f"codec {codec.name!r}: tcq V={v} has no SUC1 decoder / kernel (V in 1, 2)")
        elif codec.tcq_code == "pair3" and v != 2:
            errs.append(f"codec {codec.name!r}: tcq code pair3 needs V = 2")
        elif codec.tcq_T % v:
            errs.append(f"codec {codec.name!r}: tcq T={codec.tcq_T} not divisible by V={v}")
    return errs
