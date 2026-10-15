"""Probe MLX's two-pass decode attention dispatch against the C++ ports (cpp/ milestone 4; not a test).

The C++ runtime reproduces MLX's choice between the single-pass sdpa_vector and the two-pass kernels
(sdpa_vector_2pass_1 / _1_gqa + _2) and its block count from MLX's host code, which is not shipped with the wheel
(Model::Impl::sdpa2_rule in cpp/src/metal/model.cpp). On the M3 the rule is verified bit for bit; on another device
(e.g. the Mac Studio's M1 Ultra, architecture "...d") run this probe:

  .venv/bin/python benchmarks/cpp_sdpa_2pass_probe.py [--heads 8 --kv-heads 2] [--positions 1100] [--start 1024]
         [--candidates auto,off,std:32,std:64,std:128,std:256,gqa:2:128,...] [--build ~/suc-local/cpp-build]

It writes a tiny one-block Qwen3 SUC1 file (tests/unit/test_cpp_generate.py `_tiny_file`, head_dim 64), runs the
Python runtime token by token over --positions random tokens (the reference, cached as .npy next to the file), then
suc1-generate --prefill token once per candidate with SUC_SDPA_2PASS=<candidate> applied from --start keys
(SUC_SDPA_2PASS_MIN), and prints how many positions' final hidden states differ from MLX. "auto" is the C++ rule; a
candidate with 0 mismatches is MLX's choice for that (device, gqa factor, key count). Candidates: "off" (single
pass), "std:BLOCKS" (sdpa_vector_2pass_1), "gqa:HPT:BLOCKS" (sdpa_vector_2pass_1_gqa).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("SUC_AUTOTUNE", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "unit"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import test_cpp_generate as t  # noqa: E402
from suc.runtime.suc1_model import load_suc1_model, make_cache, step_hidden  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--positions", type=int, default=None, help="default: start + 64")
    ap.add_argument("--start", type=int, default=None, help="keys from which candidates apply (default: MLX's "
                                                            "switch point, 1024 on '...d' devices, else 4096)")
    ap.add_argument("--candidates", default="auto,off,std:32,std:64,std:128,std:256,std:512")
    ap.add_argument("--build", type=Path, default=Path(os.environ.get("SUC_CPP_BUILD", Path.home() / "suc-local/cpp-build")))
    ap.add_argument("--dir", type=Path, default=Path(tempfile.gettempdir()) / "suc_sdpa_probe")
    a = ap.parse_args()
    arch_name = mx.device_info().get("architecture", "")
    start = a.start or (1024 if arch_name.endswith("d") else 4096)
    n = a.positions or start + 64
    a.dir.mkdir(parents=True, exist_ok=True)
    arch = dict(t.ARCH, num_attention_heads=a.heads, num_key_value_heads=a.kv_heads, num_hidden_layers=1,
                max_position_embeddings=max(8192, n + 8))
    f = a.dir / f"probe_{a.heads}_{a.kv_heads}.suc1"
    if not f.exists():
        t._tiny_file(f, True, arch=arch)
    toks = [int(x) for x in np.random.default_rng(0).integers(0, arch["vocab_size"], n)]
    ref = a.dir / f"py_{a.heads}_{a.kv_heads}_{n}.npy"
    if not ref.exists():
        m, _, _ = load_suc1_model(f, group=False)
        c = make_cache(m)
        out = []
        for tok in toks:
            _, hs = step_hidden(m, tok, c)
            out.append(np.array(hs[-1].astype(mx.float32)))
        np.save(ref, np.stack(out))
    py = np.load(ref)
    print(f"device {arch_name}, gqa factor {a.heads // a.kv_heads}, {n} positions, candidates from {start} keys")
    for cand in a.candidates.split(","):
        env = dict(os.environ, SUC_SDPA_2PASS=cand, SUC_SDPA_2PASS_MIN=str(start))
        r = subprocess.run([str(a.build / "suc1-generate"), str(f), "--tokens", ",".join(map(str, toks)), "--steps", "1",
                            "--prefill", "token", "--hidden-out", str(a.dir / "c.npy"), "--quiet", "--max-context",
                            str(n + 4)], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print(f"{cand:14s} failed: {r.stderr.strip()[:200]}")
            continue
        ch = np.load(a.dir / "c.npy")[:, -1]
        bad = [p for p in range(n) if not np.array_equal(ch[p], py[p])]
        print(f"{cand:14s} positions with differing hidden states: {len(bad)}" + (f" (first {bad[:3]})" if bad else ""))


if __name__ == "__main__":
    main()
