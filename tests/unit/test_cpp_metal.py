"""Cross-checks of the C++ Metal backend (cpp/, milestone 2; tools suc1-linear and suc-metal-tests).

1. y = x Ŵᵀ on the GPU (zero-copy over the mapped SUC1 file) vs the C++ CPU reference decoder (Ŵ float32, matmul
   in float64), for synthetic files with every vq / additive variant: index widths 1..10 (aligned and unaligned),
   G = 2/3/4/8, M = 1..4, rotations in / out / in_out with Hadamard sizes 2^k, 12·2^k, 2^14 and the small-2^k dense
   fallback, residuals sparse fp16 / int8 and column-structured int8 / fp16 with and without a sparse tail.
2. Bit-exact equality with the Python/MLX runtime (suc.runtime.compressed_model.module_from_encoded, i.e. SucLinear /
   SucResidualLinear running the same Metal kernels) wherever Python uses those kernels (rotation sizes with
   2^k ≥ 32); the dense fallback (Python: mx.hadamard_transform) only to float rounding.
3. cpp/metal/suc_kernels.metal is in sync with the Python kernel sources (cpp/metal/gen_suc_kernels.py).

Skipped unless the tools are built (cpp/build.sh) into $SUC_CPP_BUILD (default ~/suc-local/cpp-build) and a Metal GPU
is present. The dev-container test is marked `model`.
"""
from __future__ import annotations

import importlib.util
import json
import os
import resource
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from suc.io.suc1 import SucModel, write_suc1
from suc.quant.codecs import AdditiveCodebook, VectorCodebook
from suc.quant.residual import ResidualCodec

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("SUC_CPP_BUILD", Path.home() / "suc-local" / "cpp-build"))
LINEAR, UNIT = BUILD / "suc1-linear", BUILD / "suc-metal-tests"
DEV_FILE = Path(os.environ.get("SUC1_DEV_FILE", Path.home() / "suc-local/compress/compress_qwen3_0.6b/container/model.suc1"))

pytestmark = pytest.mark.skipif(not LINEAR.exists(), reason=f"C++ Metal tools not built in {BUILD} (run cpp/build.sh)")

# relative-L2 thresholds vs the float64 reference, per activation dtype (f16/bf16: x̃ and y are rounded to the
# dtype, ~1-1.5x the error of rounding the exact result itself; f32: float32 accumulation)
REL_L2 = {"f32": 2e-5, "f16": 2e-3, "bf16": 1.5e-2}


def _randn(shape, seed=0, scale=0.02) -> mx.array:
    return mx.array((scale * np.random.default_rng(seed).standard_normal(shape)).astype(np.float32))


def _imp(shape, seed=0) -> mx.array:
    return mx.array(np.random.default_rng(seed).exponential(1.0, shape).astype(np.float32))


VQ = VectorCodebook
CASES = {
    "vq_g8_k256_rot": (VQ(256, block_size=8, rotate=True, kmeans_iters=2), (128, 256)),
    "vq_g4_k16_norot": (VQ(16, block_size=4, rotate=False, kmeans_iters=2), (64, 128)),
    "vq_1bit": (VQ(2, block_size=8, rotate=True, kmeans_iters=2), (64, 128)),
    "vq_3bit_g4": (VQ(8, block_size=4, rotate=True, kmeans_iters=2), (64, 128)),
    "vq_9bit_g4": (VQ(512, block_size=4, rotate=True, kmeans_iters=1), (64, 128)),
    "vq_10bit_g2": (VQ(1024, block_size=2, rotate=False, kmeans_iters=1), (64, 128)),
    "vq_g3_5bit": (VQ(32, block_size=3, rotate=False, kmeans_iters=2), (64, 96)),
    # low-dimensional single codebooks (Q5c): G = 2 with 8 / 6 / 4-bit and G = 3 with 9 / 6-bit indices
    "vq_g2_k256_rot": (VQ(256, block_size=2, rotate=True, rotate_out=False, kmeans_iters=1), (96, 512)),
    "vq_g2_k64_rot": (VQ(64, block_size=2, rotate=True, rotate_out=False, kmeans_iters=1), (96, 512)),
    "vq_g2_k16": (VQ(16, block_size=2, rotate=False, kmeans_iters=2), (64, 132)),
    "vq_g3_k512": (VQ(512, block_size=3, rotate=False, kmeans_iters=1), (64, 390)),
    "vq_g3_k64_rot": (VQ(64, block_size=3, rotate=True, rotate_out=False, kmeans_iters=1), (64, 384)),
    "vq_rot_in_only": (VQ(16, block_size=8, rotate=True, kmeans_iters=2), (72, 64)),       # 72: no Hadamard
    "vq_rot_out_only": (VQ(16, block_size=8, rotate=True, rotate_in=False, kmeans_iters=2), (64, 128)),
    "vq_rot_onesided": (VQ(16, block_size=8, rotate=True, rotate_out=False, kmeans_iters=2), (64, 128)),
    "vq_rot_m12": (VQ(16, block_size=8, rotate=True, kmeans_iters=2), (384, 3072)),       # 12*32 x 12*256
    "vq_rot_16384": (VQ(16, block_size=8, rotate=True, rotate_out=False, kmeans_iters=1), (8, 16384)),
    "additive_m3_rot": (AdditiveCodebook(16, block_size=8, num_codebooks=3, rotate=True, kmeans_iters=2), (64, 128)),
    "additive_m4_6bit": (AdditiveCodebook(64, block_size=8, num_codebooks=4, rotate=True, kmeans_iters=2), (64, 256)),
    "additive_m2_norot": (AdditiveCodebook(16, block_size=4, num_codebooks=2, rotate=False, kmeans_iters=2), (64, 128)),
    "res_fp16_vq": (ResidualCodec(VQ(16, block_size=8, rotate=True, kmeans_iters=2), 0.02), (64, 128)),
    "res_int8_additive": (ResidualCodec(AdditiveCodebook(16, block_size=8, num_codebooks=2, rotate=True,
                                                         kmeans_iters=2), 0.01, "int8"), (64, 128)),
    "res_zero": (ResidualCodec(VQ(16, block_size=8, kmeans_iters=2), 0.0, "int8"), (64, 128)),
    "res_int8_importance": (ResidualCodec(VQ(16, block_size=8, kmeans_iters=2), 0.01, "int8", "importance"),
                            (64, 128)),
    "cols_int8": (ResidualCodec(VQ(16, block_size=8, rotate=True, kmeans_iters=2), 0.02, "int8", "importance",
                                structure="columns"), (64, 256)),
    "cols_fp16": (ResidualCodec(VQ(16, block_size=8, kmeans_iters=2), 0.05, "fp16", "magnitude",
                                structure="columns"), (64, 256)),
    "cols_int8_tail": (ResidualCodec(AdditiveCodebook(16, block_size=8, num_codebooks=2, rotate=True, kmeans_iters=2),
                                     0.02, "int8", "importance", structure="columns", column_share=0.5), (64, 256)),
    # Hadamard sizes whose 2^k factor is < 32: C++ dense fallback, Python mx.hadamard_transform (not bit-exact)
    "vq_rot_m20": (VQ(16, block_size=8, rotate=True, kmeans_iters=2), (40, 640)),     # 20*2 x 20*32
    "vq_rot_m28": (VQ(16, block_size=8, rotate=True, kmeans_iters=2), (56, 448)),     # 28*2 x 28*16
}
NOT_BIT_EXACT = {"vq_rot_m20", "vq_rot_m28"}


def _encode(codec, shape, seed):
    w = _randn(shape, seed)
    kw = {"importance": _imp(shape, seed)} if getattr(codec, "selection", None) == "importance" else {}
    return codec.encode(w, seed=seed, **kw)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("metal") / "codecs.suc1"
    tensors = {f"t.{k}": _encode(c, shape, i + 1) for i, (k, (c, shape)) in enumerate(sorted(CASES.items()))}
    tensors["raw.bias"] = _randn((64,), 9)            # not a linear codec: must be skipped, not fail
    write_suc1(p, architecture={"model_type": "qwen3"}, tokenizer_id="tok", tensors=tensors)
    return p


def _run(*args: str) -> subprocess.CompletedProcess:
    r = subprocess.run([str(LINEAR), *args], capture_output=True, text=True)
    if r.returncode != 0 and "no Metal device" in r.stderr:
        pytest.skip("no Metal device")
    assert r.returncode == 0, r.stderr
    return r


def _rows(r: subprocess.CompletedProcess) -> list[dict]:
    return [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]


def test_metal_unit_tests() -> None:
    if not UNIT.exists():
        pytest.skip("suc-metal-tests not built")
    r = subprocess.run([str(UNIT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert " 0 failures" in r.stdout


def test_kernel_source_in_sync() -> None:
    spec = importlib.util.spec_from_file_location("gen_suc_kernels", ROOT / "cpp/metal/gen_suc_kernels.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert (ROOT / "cpp/metal/suc_kernels.metal").read_text() == gen.generate(), \
        "cpp/metal/suc_kernels.metal is stale: run .venv/bin/python cpp/metal/gen_suc_kernels.py and rebuild"


@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
def test_gpu_matches_cpu_reference(synthetic: Path, dtype: str) -> None:
    rows = _rows(_run(str(synthetic), "--all", "--json", "--dtype", dtype, "--batches", "1,3,4,5,16,17"))
    names = {r["tensor"] for r in rows}
    assert names == {f"t.{k}" for k in CASES}, names ^ {f"t.{k}" for k in CASES}
    bad = [(r["tensor"], r["batch"], r["rel_l2"]) for r in rows
           if not r["finite"] or r["rel_l2"] > REL_L2[dtype] or r["max_rel"] > 10 * REL_L2[dtype]]
    assert not bad, bad
    assert {r["path"] for r in rows} >= {"vec", "mma", "mma+vec"}
    zc = {r["tensor"]: r["codebooks_zero_copy"] for r in rows}
    assert all(zc.values()), "codebooks of a fresh file are adjacent: expected zero-copy"


def test_bit_exact_vs_mlx(synthetic: Path, tmp_path: Path) -> None:
    from suc.runtime.compressed_model import module_from_encoded

    m = SucModel.load(synthetic)
    xp, yp = tmp_path / "x.npy", tmp_path / "y.npy"
    bad, near = [], []
    for k in sorted(CASES):
        name = f"t.{k}"
        mod = module_from_encoded(m.encoded(name))
        for b in (1, 4, 16):
            x = np.random.default_rng(b).standard_normal((b, mod.in_features)).astype(np.float32)
            np.save(xp, x)
            _run(str(synthetic), "--tensor", name, "--x", str(xp), "--y-out", str(yp), "--dtype", "f16", "--json")
            got = np.load(yp)
            ref = np.array(mod(mx.array(x).astype(mx.float16)).astype(mx.float32))
            if k in NOT_BIT_EXACT:
                near.append(float(np.abs(got - ref).max() / np.abs(ref).max()))
            elif not np.array_equal(got.view(np.uint32), ref.view(np.uint32)):
                bad.append((name, b, int((got != ref).sum()), float(np.abs(got - ref).max())))
    assert not bad, f"C++ GPU output differs from the MLX SucLinear: {bad}"
    assert max(near) < 5e-3, near           # f16 outputs, different rotation summation order


def test_multi_view_mapping(synthetic: Path) -> None:
    """Files above maxBufferLength are mapped as several no-copy views (cut between mapping units, sharing at most one
    page); force that with small views."""
    one = _rows(_run(str(synthetic), "--all", "--json", "--dtype", "f32", "--batches", "4"))
    view = 160 * 1024     # > the largest array (73 KB), < the file: several views
    many = _rows(_run(str(synthetic), "--all", "--json", "--dtype", "f32", "--batches", "4",
                      "--max-view-bytes", str(view)))
    assert one[0]["views"] == 1 and many[0]["views"] >= 2
    page = resource.getpagesize()
    assert many[0]["view_bytes"] <= many[0]["mapped_bytes"] + (many[0]["views"] - 1) * page
    assert [(r["tensor"], r["max_abs"]) for r in one] == [(r["tensor"], r["max_abs"]) for r in many]


def test_rejects_unsupported_tensor(synthetic: Path) -> None:
    r = subprocess.run([str(LINEAR), str(synthetic), "--tensor", "raw.bias"], capture_output=True, text=True)
    assert "not supported on the GPU" in r.stderr


@pytest.mark.model
def test_dev_container(tmp_path: Path) -> None:
    """All 196 SUC tensors of the Qwen3-0.6B container vs the CPU reference, and a few bit-exact vs MLX."""
    if not DEV_FILE.exists():
        pytest.skip(f"{DEV_FILE} not found")
    rows = _rows(_run(str(DEV_FILE), "--all", "--json", "--dtype", "f16", "--batches", "1,4,16", "--no-verify"))
    assert len({r["tensor"] for r in rows}) == 196
    bad = [(r["tensor"], r["batch"], r["rel_l2"]) for r in rows if not r["finite"] or r["rel_l2"] > REL_L2["f16"]]
    assert not bad, bad
    assert all(r["codebooks_zero_copy"] for r in rows)

    from suc.runtime.compressed_model import module_from_encoded
    m = SucModel.load(DEV_FILE, verify=False)
    xp, yp = tmp_path / "x.npy", tmp_path / "y.npy"
    for name in ("model.layers.0.self_attn.q_proj.weight", "model.layers.13.mlp.gate_proj.weight",
                 "model.layers.27.mlp.down_proj.weight", "model.layers.7.self_attn.k_proj.weight"):
        mod = module_from_encoded(m.encoded(name))
        for b in (1, 4, 16):
            x = np.random.default_rng(b).standard_normal((b, mod.in_features)).astype(np.float32)
            np.save(xp, x)
            _run(str(DEV_FILE), "--tensor", name, "--x", str(xp), "--y-out", str(yp), "--dtype", "f16", "--no-verify")
            ref = np.array(mod(mx.array(x).astype(mx.float16)).astype(mx.float32))
            assert np.array_equal(np.load(yp).view(np.uint32), ref.view(np.uint32)), (name, b)
