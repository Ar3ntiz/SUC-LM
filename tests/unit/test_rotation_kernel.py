"""Numerical tests (spec §34) for the fused Hadamard rotation kernel (suc.kernels.rotation) vs suc.quant.hadamard."""
import mlx.core as mx
import numpy as np
import pytest

from suc.kernels.rotation import rht_fused, rht_inverse_fused, rotate, supported
from suc.quant.hadamard import random_signs, rht, rht_inverse

QWEN3_SIZES = [768, 1024, 2048, 3072, 4096, 6144, 12288]


def rel(a, b):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)), float(np.abs(a - b).max())


@pytest.mark.parametrize("n", QWEN3_SIZES + [32, 64, 128, 20 * 64, 28 * 32, 8192, 16384, 5120, 7168, 17 * 32, 17408])
@pytest.mark.parametrize("b", [1, 3])
def test_fused_matches_reference_float32(n, b):
    x = mx.random.normal((b, n), key=mx.random.key(n))
    s = random_signs(n, 7)
    r, mx_abs = rel(rht_fused(x, s), rht(x, 7))
    assert r < 2e-6 and mx_abs < 1e-5, (r, mx_abs)
    r, mx_abs = rel(rht_inverse_fused(x, s), rht_inverse(x, 7))
    assert r < 2e-6 and mx_abs < 1e-5, (r, mx_abs)
    # exact inverse pair
    r, _ = rel(rht_inverse_fused(rht_fused(x, s), s), x)
    assert r < 3e-6


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("n", [1024, 3072, 12288])
def test_fused_casts_once(dtype, n):
    """Low-precision in/out: x cast to float32, rotated in float32, rounded once (== reference then cast)."""
    x = mx.random.normal((2, n), key=mx.random.key(1)).astype(dtype)
    s = random_signs(n, 3)
    y = rht_fused(x, s)
    assert y.dtype == dtype
    ref = rht(x.astype(mx.float32), 3).astype(dtype)
    d = np.abs(np.array(y.astype(mx.float32)) - np.array(ref.astype(mx.float32)))
    ulp = np.abs(np.array(ref.astype(mx.float32))) * (2 ** -7 if dtype == mx.bfloat16 else 2 ** -10) + 1e-6
    assert np.mean(d == 0) > 0.97 and np.all(d <= ulp), (np.mean(d == 0), d.max())
    y32 = rht_inverse_fused(x.astype(mx.float32), s, out_dtype=dtype)
    assert y32.dtype == dtype
    assert rel(y32, rht_inverse(x.astype(mx.float32), 3))[0] < 5e-3


@pytest.mark.parametrize("sizes", [(2048, 1024, 1024), (4096, 1024, 1024), (3072, 3072), (12288, 12288), (64, 32, 32),
                                   (768,)])
@pytest.mark.parametrize("b", [1, 5])
def test_segmented_inverse_matches_per_segment(sizes, b):
    n = sum(sizes)
    x = mx.random.normal((b, n), key=mx.random.key(2))
    seeds = [11 + i for i in range(len(sizes))]
    signs = mx.concatenate([random_signs(m, sd) for m, sd in zip(sizes, seeds)])
    y = rotate(x, signs, sizes, inverse=True)
    parts, o = [], 0
    for m, sd in zip(sizes, seeds):
        parts.append(rht_inverse(x[:, o:o + m], sd))
        o += m
    r, mx_abs = rel(y, mx.concatenate(parts, axis=1))
    assert r < 2e-6 and mx_abs < 1e-5
    yf = rotate(x, signs, sizes, inverse=False)
    parts, o = [], 0
    for m, sd in zip(sizes, seeds):
        parts.append(rht(x[:, o:o + m], sd))
        o += m
    assert rel(yf, mx.concatenate(parts, axis=1))[0] < 2e-6


def test_leading_dims_and_validation():
    x = mx.random.normal((2, 3, 1024), key=mx.random.key(0))
    s = random_signs(1024, 0)
    assert rel(rht_fused(x, s), rht(x, 0))[0] < 2e-6
    assert supported(1024) and supported(768) and not supported(16) and not supported(9728)
    with pytest.raises(ValueError):
        rotate(x, s, (512, 256))
    with pytest.raises(ValueError):
        rotate(mx.zeros((1, 9728)), mx.ones(9728))


@pytest.mark.parametrize("b", [1, 4])
def test_split_outputs_equal_concatenated(b):
    sizes = (2048, 1024, 1024)
    x = mx.random.normal((b, 1, sum(sizes)), key=mx.random.key(3))
    signs = random_signs(sum(sizes), 1)
    whole = rotate(x, signs, sizes, inverse=True, out_dtype=mx.bfloat16)
    parts = rotate(x, signs, sizes, inverse=True, out_dtype=mx.bfloat16, split=True)
    assert [p.shape for p in parts] == [(b, 1, n) for n in sizes]
    assert np.array_equal(np.array(mx.concatenate(parts, axis=-1).astype(mx.float32)),
                          np.array(whole.astype(mx.float32)))


# ------------------------------------------------------------------------------------------- R5 layouts
import suc.kernels.rotation as rot  # noqa: E402

LAYOUTS = [dict(LAYOUT="blocks", BLOCK_Q=q, BLOCK_SG=sg) for q, sg in ((32, 1), (64, 2), (128, 8), (256, 4),
                                                                          (256, 8), (512, 4), (512, 8))] + \
          [dict(LAYOUT="row", VEC=v, THREADS_MAX=t, HM_CONST=h) for v, t, h in ((1, 256, False), (2, 512, True),
                                                                                  (4, 256, True), (4, 128, True))]
LAYOUT_IDS = ["-".join(f"{k[0]}{v}" for k, v in c.items()) for c in LAYOUTS]


@pytest.fixture(params=LAYOUTS, ids=LAYOUT_IDS)
def layout(request, monkeypatch):
    for k, v in request.param.items():
        monkeypatch.setattr(rot, k, v)
    return request.param


@pytest.mark.parametrize("n", [32, 128, 768, 3072, 4096, 12288, 20 * 64, 28 * 128, 5120, 7168, 17 * 32, 17408])
def test_layouts_match_reference(layout, n):
    """Every layout / block shape computes the same rotation (float32 summation order aside), both directions."""
    x = mx.random.normal((2, n), key=mx.random.key(n + 1))
    s = random_signs(n, 5)
    for inv, ref in ((False, rht(x, 5)), (True, rht_inverse(x, 5))):
        r, mx_abs = rel(rot.rotate(x, s, inverse=inv), ref)
        assert r < 2e-6 and mx_abs < 1e-5, (inv, r, mx_abs)


@pytest.mark.parametrize("vec", [1, 2, 4])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_row_hm_const_half_precision_m17(monkeypatch, vec, dtype):
    """Row layout, H_17 as compile-time constants, half-precision I/O, n = 17408: VEC = 1 bf16 forward stored ~0 for
    sub-vectors a = 0, 1 on the M3 until the simdgroup_barrier between the core and the store (2026-10-13)."""
    for k, v in dict(LAYOUT="row", VEC=vec, THREADS_MAX=256, HM_CONST=True).items():
        monkeypatch.setattr(rot, k, v)
    n = 17408
    x = mx.random.normal((2, n), key=mx.random.key(n + 1)).astype(dtype)
    s = random_signs(n, 5)
    xf = x.astype(mx.float32)
    for inv, ref in ((False, rht(xf, 5)), (True, rht_inverse(xf, 5))):
        r, mx_abs = rel(rot.rotate(x, s, inverse=inv), ref)
        assert r < 5e-3 and mx_abs < 0.1, (inv, r, mx_abs)


@pytest.mark.parametrize("sizes", [(4096, 1024, 1024), (12288, 12288), (64, 32, 32)])
def test_layouts_segments_split_and_row_signs(layout, sizes):
    n = sum(sizes)
    x = mx.random.normal((3, n), key=mx.random.key(4))
    signs = mx.concatenate([random_signs(m, 20 + i) for i, m in enumerate(sizes)])
    ref, o = [], 0
    for i, m in enumerate(sizes):
        ref.append(rht_inverse(x[:, o:o + m], 20 + i))
        o += m
    ref = mx.concatenate(ref, axis=1)
    assert rel(rot.rotate(x, signs, sizes, inverse=True), ref)[0] < 2e-6
    parts = rot.rotate(x, signs, sizes, inverse=True, split=True)
    assert rel(mx.concatenate(parts, axis=1), ref)[0] < 2e-6
    # per-row sign tables (MoE): row b uses table rows[b]
    table = mx.stack([signs, -signs])
    rows = mx.array([1, 0, 1], dtype=mx.uint32)
    y = rot.rotate(x, table, sizes, inverse=False, rows=rows)
    for b in range(3):
        sb = table[int(rows[b])]
        yb = rot.rotate(x[b:b + 1], sb, sizes, inverse=False)
        assert np.array_equal(np.array(y[b:b + 1]), np.array(yb))


@pytest.mark.parametrize("n", [4096, 12288])
def test_new_kernel_agrees_with_pre_r5_kernel_in_bf16(n):
    """bf16 in / out (the runtime case): the R5 kernels round to the same bf16 values as the pre-R5 kernel except
    for rare float32 rounding-boundary cases (different summation order), never by more than one bf16 ulp."""
    x = mx.random.normal((4, n), key=mx.random.key(9)).astype(mx.bfloat16)
    s = random_signs(n, 2)
    old = {"LAYOUT": rot.LAYOUT, "VEC": rot.VEC, "HM_CONST": rot.HM_CONST}
    try:
        rot.LAYOUT, rot.VEC, rot.HM_CONST = "row", 1, False
        y0 = np.array(rot.rht_fused(x, s).astype(mx.float32))
    finally:
        for k, v in old.items():
            setattr(rot, k, v)
    for lay in ("blocks", "row"):
        rot.LAYOUT = lay
        try:
            y1 = np.array(rot.rht_fused(x, s).astype(mx.float32))
        finally:
            rot.LAYOUT = old["LAYOUT"]
        d = np.abs(y1 - y0)
        assert np.mean(d == 0) > 0.98 and np.all(d <= np.abs(y0) * 2 ** -7 + 1e-6), (lay, np.mean(d == 0))


def test_default_block_config_and_tuned_choice(tmp_path, monkeypatch):
    import json
    import suc.kernels.autotune as at
    assert rot.default_block_config(4096) == (256, 8)
    assert rot.default_block_config(12288) == (512, 8)
    assert rot.default_block_config(1024) == (128, 8)
    monkeypatch.setenv("SUC_TUNING_CACHE", str(tmp_path / "t.json"))
    monkeypatch.setattr(rot, "LAYOUT", "auto")
    at.clear_memory_cache()
    try:
        assert rot.layout_for(4096) == "blocks" and rot.block_config(4096) == (256, 8)
        (tmp_path / "t.json").write_text(json.dumps({at.device_key(): {"rotation": {"choice": {
            "4096": {"layout": "row"}, "12288": {"layout": "blocks", "q": 256, "sg": 4}}}}}))
        at.clear_memory_cache()
        assert rot.layout_for(4096) == "row" and rot.layout_for(12288) == "blocks"
        assert rot.block_config(12288) == (256, 4)
        monkeypatch.setattr(rot, "BLOCK_Q", 128)            # explicit overrides win
        assert rot.block_config(12288) == (128, 4)
        x = mx.random.normal((1, 4096), key=mx.random.key(0))
        assert rel(rot.rht_fused(x, random_signs(4096, 1)), rht(x, 1))[0] < 2e-6
    finally:
        at.clear_memory_cache()


@pytest.mark.parametrize("n", [1024, 4096, 12288])
def test_int8_signs_equal_float32_signs(layout, n):
    """SucLinear stores int8 signs (R5); the kernels read them as is, with bit-identical results."""
    x = mx.random.normal((2, n), key=mx.random.key(3)).astype(mx.bfloat16)
    s = random_signs(n, 8)
    for inv in (False, True):
        a = rot.rotate(x, s, inverse=inv)
        b = rot.rotate(x, s.astype(mx.int8), inverse=inv)
        assert np.array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)))


# ------------------------------------------------------------------------- launch policy (audit F09)
# A threadgroup above the compiled pipeline's maxTotalThreadsPerThreadgroup aborts the interpreter from MLX's native
# code (row layout, T = 512, VEC = 2, n = 17408 on the M1 Ultra: limit 384). These checks run in a subprocess so a
# regression shows up as a failed test, not as a crashed pytest session.
_F09_CHILD = r"""
import sys
import mlx.core as mx
import numpy as np
import suc.kernels.rotation as rot
from suc.quant.hadamard import random_signs, rht, rht_inverse

n = int(sys.argv[1])
rot.LAYOUT, rot.THREADS_MAX, rot.HM_CONST = "row", 512, True


def rel(a, b):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)), float(np.abs(a - b).max())


s = random_signs(n, 5)
for vec in (1, 2, 4):
    rot.VEC = vec
    t = rot._plan((n,), False, False, rot.row_threads_max(), rot.BUF_FLOATS, False, vec)[3]
    assert t <= rot.LAUNCH_MAX_THREADS and t & (t - 1) == 0, t
    for dt in (mx.float32, mx.float16, mx.bfloat16):
        x = mx.random.normal((2, n), key=mx.random.key(n + 1)).astype(dt)
        xf = x.astype(mx.float32)
        for inv, ref in ((False, rht(xf, 5)), (True, rht_inverse(xf, 5))):
            y = rot.rotate(x, s, inverse=inv)
            assert y.dtype == dt
            r, m = rel(y, ref)
            tol = (2e-6, 1e-5) if dt == mx.float32 else (5e-3, 0.1)
            assert r < tol[0] and m < tol[1], (vec, dt, inv, r, m)
    # segmented (fused q|k|v style), split outputs and per-row sign tables (MoE)
    sizes = (n, 1024, 1024)
    x = mx.random.normal((3, sum(sizes)), key=mx.random.key(4))
    sg = mx.concatenate([random_signs(k, 20 + i) for i, k in enumerate(sizes)])
    ref, o = [], 0
    for i, k in enumerate(sizes):
        ref.append(rht_inverse(x[:, o:o + k], 20 + i))
        o += k
    ref = mx.concatenate(ref, axis=1)
    assert rel(rot.rotate(x, sg, sizes, inverse=True), ref)[0] < 2e-6
    assert rel(mx.concatenate(rot.rotate(x, sg, sizes, inverse=True, split=True), axis=1), ref)[0] < 2e-6
    table, rows = mx.stack([sg, -sg]), mx.array([1, 0, 1], dtype=mx.uint32)
    y = rot.rotate(x, table, sizes, rows=rows)
    for b in range(3):
        assert np.array_equal(np.array(y[b:b + 1]), np.array(rot.rotate(x[b:b + 1], table[int(rows[b])], sizes)))
    # fused prologues (RMSNorm / SwiGLU + rotation) share the row plan
    w = mx.ones((n,), dtype=mx.float32)
    xr = mx.random.normal((2, n), key=mx.random.key(7))
    yn, _ = rot.rms_norm_rht(xr, w, 1e-6, s)
    assert rel(yn, rht(mx.fast.rms_norm(xr, w, 1e-6), 5))[0] < 2e-6
print("F09_OK", n)
"""


@pytest.mark.parametrize("n", [1024, 12 * 1024, 17 * 1024, 20 * 1024, 28 * 1024])
def test_f09_row_threads_512_request_runs_in_subprocess(n):
    """THREADS_MAX = 512 (the F09 reproducer, factors 1/12/17/20/28) falls back to a 256-thread power-of-two plan:
    the process exits normally and forward / inverse / segmented / row-sign / prologue results match the reference
    in float32, float16 and bfloat16."""
    import os
    import subprocess
    import sys
    env = dict(os.environ, SUC_AUTOTUNE="0")
    r = subprocess.run([sys.executable, "-c", _F09_CHILD, str(n)], capture_output=True, text=True, timeout=900,
                       env=env)
    assert r.returncode == 0 and f"F09_OK {n}" in r.stdout, (r.returncode, r.stdout[-2000:], r.stderr[-2000:])


def test_launch_policy_contract(monkeypatch):
    """Configurations the policy cannot run are rejected before any GPU dispatch (ValueError, not an abort);
    requests above the ceiling fall back to compatible plans."""
    assert rot.threads_and_vec((1024,), 256, 2) == (256, (2,))
    for tmax, vec in ((384, 2), (512 + 32, 1), (16, 1), (256, 3)):
        with pytest.raises(ValueError):
            rot.threads_and_vec((1024,), tmax, vec)
    for t in (512, 384, 1024, 16, 96):
        with pytest.raises(ValueError):
            rot._check_launch(t, "row")
    rot._check_launch(96, "blocks")            # blocks: any multiple of 32 up to the ceiling
    with pytest.raises(ValueError):
        rot._check_launch(512, "blocks")
    monkeypatch.setattr(rot, "THREADS_MAX", 512)
    assert rot.row_threads_max() == rot.LAUNCH_MAX_THREADS == 256
    monkeypatch.setattr(rot, "THREADS_MAX", 384)
    with pytest.raises(ValueError):
        rot.row_threads_max()
    monkeypatch.setattr(rot, "LAYOUT", "row")
    with pytest.raises(ValueError):            # raised before dispatch
        rot.rotate(mx.zeros((1, 1024)), mx.ones(1024))
    # blocks: an explicit simdgroup override above the ceiling is clipped to LAUNCH_MAX_THREADS / 32
    assert rot.blocks_shape((1024,), (17,), 32, 16) == (32, 256)
    monkeypatch.setattr(rot, "LAYOUT", "blocks")
    monkeypatch.setattr(rot, "BLOCK_Q", 32)
    monkeypatch.setattr(rot, "BLOCK_SG", 16)
    x = mx.random.normal((2, 17408), key=mx.random.key(0))
    assert rel(rot.rotate(x, random_signs(17408, 5)), rht(x, 5))[0] < 2e-6
