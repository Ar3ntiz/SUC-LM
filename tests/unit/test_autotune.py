"""Per-device kernel autotuning: measured choices cached per GPU, read by the kernels without re-measuring."""
import json

import pytest

import suc.kernels.autotune as at
from suc.kernels.vq_matmul import resolve_codebook_in


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SUC_TUNING_CACHE", str(tmp_path / "tuning.json"))
    at.clear_memory_cache()
    yield
    at.clear_memory_cache()


def test_cached_choice_overrides_builtin_default(tmp_path):
    key = "applegpu_g99x|Apple M9 Max"
    (tmp_path / "tuning.json").write_text(json.dumps(
        {key: {"codebook_in": {"1": "threadgroup", "3": "device"}}}))
    assert resolve_codebook_in("auto", key=key, m=1) == "threadgroup"
    assert resolve_codebook_in("auto", key=key, m=3) == "device"
    # an M without a measured entry falls back to the nearest measured M
    assert resolve_codebook_in("auto", key=key, m=4) == "device"


def test_unknown_device_falls_back_to_builtin_table():
    assert resolve_codebook_in("auto", key="applegpu_g13d|Apple M1 Ultra", m=1) == "threadgroup"
    assert resolve_codebook_in("auto", key="applegpu_g15g|Apple M3", m=1) == "device"


def test_explicit_placement_is_never_overridden(tmp_path):
    key = "applegpu_g99x|Apple M9 Max"
    (tmp_path / "tuning.json").write_text(json.dumps({key: {"codebook_in": {"1": "threadgroup"}}}))
    assert resolve_codebook_in("device", key=key, m=1) == "device"
    with pytest.raises(ValueError):
        resolve_codebook_in("shared", key=key, m=1)


def test_autotune_picks_the_faster_placement_and_persists(tmp_path):
    timings = {("device", 1): 10.0, ("threadgroup", 1): 5.0, ("device", 2): 7.0, ("threadgroup", 2): 9.0}
    calls = []

    def fake_measure(placement, m, batch):
        calls.append((placement, m, batch))
        return timings[(placement, m)] * batch

    key = "applegpu_g99x|Apple M9 Max"
    entry = at.autotune(key=key, ms=(1, 2), batches=(1, 4), measure=fake_measure)
    assert entry["codebook_in"] == {"1": "threadgroup", "2": "device"}
    saved = json.loads((tmp_path / "tuning.json").read_text())
    assert saved[key]["codebook_in"] == {"1": "threadgroup", "2": "device"}
    assert "measurements_us" in saved[key] and "mlx" in saved[key]
    # ensure_tuned does not re-measure a device that already has an entry
    n = len(calls)
    at.ensure_tuned(key=key, ms=(1, 2), batches=(1, 4), measure=fake_measure)
    assert len(calls) == n


def test_ensure_tuned_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SUC_AUTOTUNE", "0")
    called = []
    assert at.ensure_tuned(key="k|x", measure=lambda *a: called.append(a) or 1.0) is None
    assert not called


def test_real_measurement_smoke():
    """Runs the actual kernels once on this machine (tiny problem): both placements must produce timings."""
    entry = at.autotune(ms=(1,), batches=(1,), out=256, inp=256, target_mb=1, reps=1, save=False)
    assert entry["codebook_in"]["1"] in ("device", "threadgroup")
    assert all(v > 0 for v in entry["measurements_us"].values())


# ------------------------------------------------------------------------------ R5: runtime knobs per device
KEY = "applegpu_g99x|Apple M9 Max"


def test_rotation_section_picks_fastest_candidate_per_size_and_keeps_other_sections(tmp_path):
    (tmp_path / "tuning.json").write_text(json.dumps({KEY: {"codebook_in": {"1": "device"}}}))
    fake = {4096: {"row": 9.0, "blocks_q256_sg8": 4.0, "blocks_q512_sg4": 5.0},
            12288: {"row": 7.0, "blocks_q256_sg8": 8.0, "blocks_q512_sg4": 6.5}}
    sec = at.tune_rotation([4096, 12288, 9728], key=KEY, measure=lambda n: fake[n])   # 9728: unsupported, skipped
    assert sec["choice"] == {"4096": {"layout": "blocks", "q": 256, "sg": 8},
                             "12288": {"layout": "blocks", "q": 512, "sg": 4}}
    saved = json.loads((tmp_path / "tuning.json").read_text())[KEY]
    assert saved["codebook_in"] == {"1": "device"} and saved["rotation"]["choice"]["4096"]["q"] == 256
    at.clear_memory_cache()
    assert at.rotation_choice(4096, key=KEY) == {"layout": "blocks", "q": 256, "sg": 8}
    assert at.rotation_choice(1024, key=KEY) is None
    # a later size is merged in, earlier sizes stay
    at.tune_rotation([1024], key=KEY, measure=lambda n: {"row": 1.0, "blocks_q128_sg8": 2.0})
    at.clear_memory_cache()
    assert at.rotation_choice(1024, key=KEY) == {"layout": "row"} and at.rotation_choice(4096, key=KEY) is not None
    # re-running the codebook tuning keeps the other sections
    at.autotune(key=KEY, ms=(1,), batches=(1,), measure=lambda p, m, b: 1.0 if p == "device" else 2.0)
    assert set(json.loads((tmp_path / "tuning.json").read_text())[KEY]) >= {"codebook_in", "rotation"}


def test_fuse_elementwise_is_decided_per_op():
    us = {"norm_fused": 5.0, "norm_separate": 6.0, "swiglu_fused": 9.0, "swiglu_separate": 7.0}
    sec = at.tune_fuse_elementwise(key=KEY, measure=lambda h, i: us)
    assert sec["choice"] == {"norm": True, "swiglu": False}
    at.clear_memory_cache()
    assert at.fuse_elementwise_choice("norm", key=KEY) is True
    assert at.fuse_elementwise_choice("swiglu", key=KEY) is False


def test_mma_min_batch_is_the_first_batch_from_which_mma_always_wins():
    t = {1: {2: (1, 2), 3: (1, 2), 4: (2, 1), 5: (1, 2), 6: (2, 1), 8: (3, 1)},     # wins at 4 but loses at 5
         3: {2: (1, 2), 3: (2, 1), 4: (2, 1), 5: (2, 1), 6: (2, 1), 8: (2, 1)}}
    sec = at.tune_mma_min_batch(key=KEY, ms=(1, 3), measure=lambda path, m, b: t[m][b][path == "mma"])
    assert sec["choice"] == {"1": 6, "3": 3}
    never = at.tune_mma_min_batch(key=KEY, ms=(1,), batches=(2, 4), measure=lambda path, m, b: 2.0 if path == "mma"
                                  else 1.0, save=False)
    assert never["choice"] == {"1": 5}
    at.clear_memory_cache()
    assert at.mma_min_batch_choice(1, key=KEY) == 6 and at.mma_min_batch_choice(2, key=KEY) == 3


def test_runtime_reads_tuned_knobs_unless_forced(tmp_path, monkeypatch):
    import suc.kernels.vq_matmul as vm
    import suc.runtime.compressed_model as cm
    monkeypatch.setattr(at, "device_key", lambda: KEY)
    monkeypatch.setattr(vm, "MMA_MIN_BATCH", None)
    monkeypatch.setattr(cm, "FUSE_ELEMENTWISE", None)
    assert vm.mma_min_batch(1) == vm.MMA_MIN_BATCH_DEFAULT and cm.fuse_elementwise() is False   # nothing cached
    (tmp_path / "tuning.json").write_text(json.dumps({KEY: {"mma_min_batch": {"choice": {"1": 3, "3": 7}},
                                                            "fuse_elementwise": {"choice": {"norm": True,
                                                                                            "swiglu": False}}}}))
    at.clear_memory_cache()
    assert vm.mma_min_batch(1) == 3 and vm.mma_min_batch(4) == 7
    assert cm.fuse_elementwise("norm") is True and cm.fuse_elementwise("swiglu") is False and cm.fuse_elementwise()
    monkeypatch.setattr(vm, "MMA_MIN_BATCH", 5)
    monkeypatch.setattr(cm, "FUSE_ELEMENTWISE", False)
    assert vm.mma_min_batch(1) == 5 and cm.fuse_elementwise() is False


def test_ensure_tuned_measures_only_missing_sections_and_sizes(tmp_path, monkeypatch):
    monkeypatch.setenv("SUC_AUTOTUNE", "1")
    calls = []
    monkeypatch.setattr(at, "autotune", lambda **kw: calls.append("codebook_in"))
    monkeypatch.setattr(at, "tune_rotation", lambda sizes, key: calls.append(("rotation", tuple(sizes))))
    monkeypatch.setattr(at, "tune_mma_min_batch", lambda key: calls.append("mma_min_batch"))
    monkeypatch.setattr(at, "tune_fuse_elementwise", lambda h, i, key: calls.append(("fuse", h, i)))
    (tmp_path / "tuning.json").write_text(json.dumps({KEY: {"codebook_in": {"1": "device"},
                                                            "rotation": {"choice": {"4096": {"layout": "row"}}}}}))
    at.clear_memory_cache()
    at.ensure_tuned(key=KEY, rotation_sizes=(4096, 12288), hidden=4096, intermediate=12288)
    assert calls == [("rotation", (12288,)), "mma_min_batch", ("fuse", 4096, 12288)]
    calls.clear()
    at.ensure_tuned(key=KEY, sections=("codebook_in",), rotation_sizes=(1024,))
    assert calls == []


def test_real_rotation_and_fusion_measurement_smoke():
    """The decode-context chains run on this machine (small sizes, few links): every candidate gets a timing."""
    us = at.measure_rotation(1024, links=4, reps=2, warmup_s=0.0)
    assert set(us) == {at._cand_name(c) for c in at._rotation_candidates(1024)} and "row" in us
    f = at.measure_fuse_elementwise(1024, 3072, links=4, reps=2, warmup_s=0.0)
    assert set(f) == {"norm_fused", "norm_separate", "swiglu_fused", "swiglu_separate"}


# Regression: the decode-context chain read x[:, :1024] of an op output narrower than 1024, so every rotation size
# < 1024 (and RMSNorm / SwiGLU fusion for hidden / intermediate < 1024) failed with "x must be (B, 1024)".
@pytest.mark.parametrize("n", [32, 64, 128, 256, 384, 768])
def test_rotation_tuning_small_sizes(n):
    sec = at.tune_rotation([n], key=KEY, save=False,
                           measure=lambda s: at.measure_rotation(s, links=2, reps=1, warmup_s=0.0))
    assert set(sec["measurements_us"][str(n)]) == {at._cand_name(c) for c in at._rotation_candidates(n)}
    assert sec["choice"][str(n)] in at.ROTATION_CANDIDATES


def test_rotation_tuning_skips_unsupported_and_failing_sizes():
    # 48 = 3·16, 96 = 3·32 (m = 3 not supported), 16 (2^k < 32): MLX fallback, never measured
    def measure(n):
        if n == 1024:
            raise RuntimeError("boom")
        return {"row": 2.0, "blocks_q128_sg8": 1.0, "stale_candidate": 0.5}
    with pytest.warns(RuntimeWarning, match="n=1024"):
        sec = at.tune_rotation([16, 48, 96, 1024, 4096], key=KEY, measure=measure, save=False)
    assert set(sec["choice"]) == {"4096"}
    assert sec["choice"]["4096"] == {"layout": "blocks", "q": 128, "sg": 8}   # unknown candidate names ignored


def test_fusion_tuning_small_and_unsupported_sizes():
    f = at.measure_fuse_elementwise(64, 128, links=2, reps=1, warmup_s=0.0)
    assert set(f) == {"norm_fused", "norm_separate", "swiglu_fused", "swiglu_separate"}
    sec = at.tune_fuse_elementwise(64, 96, key=KEY, save=False,          # 96: no fused SwiGLU kernel
                                   measure=lambda h, i: at.measure_fuse_elementwise(h, i, links=2, reps=1,
                                                                                    warmup_s=0.0))
    assert set(sec["measurements_us"]) == {"norm_fused", "norm_separate"} and sec["choice"]["swiglu"] is False


def test_ensure_tuned_never_raises(monkeypatch):
    monkeypatch.setenv("SUC_AUTOTUNE", "1")
    calls = []

    def boom(*a, **kw):
        calls.append(1)
        raise RuntimeError("tuning failed")
    for name in ("autotune", "tune_rotation", "tune_mma_min_batch", "tune_fuse_elementwise"):
        monkeypatch.setattr(at, name, boom)
    with pytest.warns(RuntimeWarning, match="using defaults"):
        at.ensure_tuned(key=KEY, rotation_sizes=(48, 1024), hidden=1024, intermediate=3072)
    assert len(calls) == 4      # every section attempted despite the earlier failures
    monkeypatch.setattr(at, "device_key", boom)
    with pytest.warns(RuntimeWarning, match="skipped"):
        assert at.ensure_tuned() is None


def test_ensure_tuned_ignores_unsupported_rotation_sizes(monkeypatch):
    monkeypatch.setenv("SUC_AUTOTUNE", "1")
    calls = []
    monkeypatch.setattr(at, "tune_rotation", lambda sizes, key: calls.append(tuple(sizes)))
    at.ensure_tuned(key=KEY, sections=("rotation",), rotation_sizes=(16, 48, 96, 64))
    assert calls == [(64,)]


def test_rotation_cache_identity_includes_launch_policy(tmp_path, monkeypatch):
    """Audit F09: rotation choices are tied to suc.kernels.rotation.LAUNCH_POLICY. Sections written without the
    field (before the policy, all within it) stay valid; a section from another policy is ignored, re-tuned and
    never merged."""
    import suc.kernels.rotation as rot
    sec = at.tune_rotation([1024], key=KEY, measure=lambda n: {"row": 1.0, "blocks_q128_sg8": 2.0})
    assert sec["launch_policy"] == rot.LAUNCH_POLICY
    at.clear_memory_cache()
    assert at.rotation_choice(1024, key=KEY) == {"layout": "row"}
    legacy = {KEY: {"rotation": {"choice": {"4096": {"layout": "row"}}}}}
    (tmp_path / "tuning.json").write_text(json.dumps(legacy))
    at.clear_memory_cache()
    assert at.rotation_choice(4096, key=KEY) == {"layout": "row"}
    legacy[KEY]["rotation"]["launch_policy"] = "tg512-other"
    (tmp_path / "tuning.json").write_text(json.dumps(legacy))
    at.clear_memory_cache()
    assert at.rotation_choice(4096, key=KEY) is None
    calls = []
    monkeypatch.setattr(at, "tune_rotation", lambda sizes, key: calls.append(tuple(sizes)))
    monkeypatch.setenv("SUC_AUTOTUNE", "1")
    at.ensure_tuned(key=KEY, sections=("rotation",), rotation_sizes=(4096,))
    assert calls == [(4096,)]


def test_rotation_candidates_respect_launch_policy(monkeypatch):
    """Candidates whose launch the policy rejects are never measured (no native abort while tuning)."""
    import suc.kernels.rotation as rot
    names = {at._cand_name(c) for c in at._rotation_candidates(17408)}
    assert "row" in names
    real = rot._check_launch

    def reject_row(t, layout):
        if layout == "row":
            raise ValueError("rejected")
        real(t, layout)
    monkeypatch.setattr(rot, "_check_launch", reject_row)
    assert "row" not in {at._cand_name(c) for c in at._rotation_candidates(17408)}
