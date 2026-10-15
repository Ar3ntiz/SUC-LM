"""A/B the K3 kernel switches on the current machine (meant for the Mac Studio M1 Ultra).

Runs benchmarks/k3_cold_cache.py once per variant with module constants overridden; each JSON
records `kernel_config`. Usage: .venv/bin/python benchmarks/k3_ab.py <out_prefix> [--gb 16]
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

VARIANTS = {
    "default": {},
    "half_math": {"LOOKUP_HALF_MATH": True},
    "sg4_rows8": {"LOOKUP_SG": 4, "LOOKUP_ROWS": {1: 8, 2: 4, 4: 2, 8: 2, 16: 1}},
    "vec2": {"LOOKUP_VEC": 2},
    "cb_threadgroup": {"CODEBOOK_IN": "threadgroup"},
    "mma_rt2": {"MMA_RT": 2},
}


def main() -> None:
    prefix, gb = sys.argv[1], (sys.argv[sys.argv.index("--gb") + 1] if "--gb" in sys.argv else "16")
    sys.path.insert(0, str(Path(__file__).parent))
    for name, consts in VARIANTS.items():
        vq = importlib.import_module("suc.kernels.vq_matmul")
        saved = {k: getattr(vq, k) for k in consts}
        for k, v in consts.items():
            setattr(vq, k, v)
        try:
            bench = importlib.import_module("k3_cold_cache")
            shapes = "8b,30b,0.6b" if name == "default" else "8b"
            sys.argv = ["k3_cold_cache.py", "--gb", gb, "--shapes", shapes, "--json", f"{prefix}_{name}.json"]
            print(f"== variant {name} {consts}", flush=True)
            bench.main()
        finally:
            for k, v in saved.items():
                setattr(vq, k, v)


if __name__ == "__main__":
    main()
