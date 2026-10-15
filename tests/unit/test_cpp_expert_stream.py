"""SSD expert streaming in the C++/Metal runtime (cpp/src/metal/expert_stream.cpp, `suc1-generate --expert-budget`):
the routed experts read with pread into per-layer GPU slot pools give the SAME logits, hidden states and tokens as the
resident runtime (bit for bit), with budgets that force evictions on every step, for Qwen3-MoE (per-expert codecs with
CSR residuals, column-structured residuals; LRU / SLRU; next-layer prefetch; a pinned hot set; batch prefill; the
pipelined decode path) and qwen4_exp; the pools respect the budget and the routed-expert payloads stay out of the GPU
views of the file mapping.

Skipped unless suc1-generate is built (cpp/build.sh) into $SUC_CPP_BUILD (default ~/suc-local/cpp-build).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

BUILD = Path(os.environ.get("SUC_CPP_BUILD", Path.home() / "suc-local" / "cpp-build"))
GENERATE = BUILD / "suc1-generate"

pytestmark = pytest.mark.skipif(not GENERATE.exists(), reason=f"suc1-generate not built in {BUILD} (run cpp/build.sh)")

PROMPT = [5, 17, 300, 42, 7, 99]


def _run(file: Path, tmp: Path, *extra: str, prompt=PROMPT, steps: int = 10, rows: bool = True, ok: bool = True):
    """(generated, logits, hidden, json record) of one suc1-generate run."""
    j = tmp / "out.json"
    args = [str(GENERATE), str(file), "--tokens", ",".join(map(str, prompt)), "--steps", str(steps), "--json", str(j)]
    if rows:
        args += ["--logits-out", str(tmp / "l.npy"), "--hidden-out", str(tmp / "h.npy")]
    r = subprocess.run(args + list(extra), capture_output=True, text=True)
    if not ok:
        assert r.returncode != 0
        return r.stderr
    assert r.returncode == 0, r.stderr
    gen = [int(x) for x in r.stdout.split("generated:")[1].split()[0].split(",")]
    rec = json.loads(j.read_text())
    if rows:
        return gen, np.load(tmp / "l.npy"), np.load(tmp / "h.npy"), rec
    return gen, None, None, rec


def _slot_bytes(file: Path, tmp: Path, *extra: str, prompt=(5, 6, 7)) -> tuple[int, int]:
    """(largest slot bytes, MoE layers) from a run whose budget holds every expert."""
    rec = _run(file, tmp, "--expert-budget", str(1 << 34), *extra, prompt=prompt, steps=2, rows=False)[3]["expert_streaming"]
    return rec["slot_bytes_max"], rec["layers"]


def _same(a, b) -> None:
    ga, la, ha, _ = a
    gb, lb, hb, _ = b
    assert ga == gb
    assert la.shape == lb.shape and np.array_equal(la, lb), f"logits differ at {np.argwhere(la != lb)[:5].tolist()}"
    assert ha.shape == hb.shape and np.array_equal(ha, hb)


@pytest.fixture(scope="module")
def moe_files(tmp_path_factory):
    from test_cpp_generate import ARCH_MOE, EXPERT_CODECS, EXPERT_COLS, _moe_file

    tmp = tmp_path_factory.mktemp("xs_moe")
    return {"per_expert": _moe_file(tmp / "pe.suc1", ARCH_MOE, expert_codecs=EXPERT_CODECS),
            "cols": _moe_file(tmp / "cols.suc1", dict(ARCH_MOE, num_experts_per_tok=4), expert_codecs=EXPERT_COLS)}


@pytest.mark.parametrize("variant", ["per_expert", "cols"])
@pytest.mark.parametrize("opts", [("--expert-policy", "lru"), ("--expert-policy", "slru"),
                                  ("--expert-prefetch", "--expert-io-threads", "2")])
def test_qwen3_moe_streamed_equals_resident(moe_files, variant, opts, tmp_path):
    f = moe_files[variant]
    for sep in ((), ("--projections", "separate")):
        base = ("--prefill", "token", *sep)
        resident = _run(f, tmp_path, *base)
        slot, layers = _slot_bytes(f, tmp_path)
        k = 2 if variant == "per_expert" else 4
        budget = layers * (k + 1) * slot                  # one slot more than top-k per layer: evictions every step
        got = _run(f, tmp_path, *base, "--expert-budget", str(budget), *opts)
        _same(resident, got)
        st = got[3]["expert_streaming"]
        assert st["pool_bytes"] <= budget and st["slots"] == layers * (k + 1)
        assert st["misses"] > 0 and st["evictions"] > 0 and st["bytes_read"] > 0 and st["hit_rate"] < 1.0
        assert st["requests"] == st["hits"] + st["misses"]
        if "--expert-prefetch" in opts:
            assert st["prefetch_issued"] > 0


def test_qwen3_moe_hot_set_and_budget_errors(moe_files, tmp_path):
    f = moe_files["per_expert"]
    resident = _run(f, tmp_path, "--prefill", "token")
    slot, layers = _slot_bytes(f, tmp_path)
    hot = tmp_path / "hot_experts.json"
    hot.write_text(json.dumps({"layers": {f"model.layers.{i}.mlp": [9, 1, 0, 7, 3, 0, 2, 5] for i in range(layers)}}))
    budget = layers * 4 * slot
    got = _run(f, tmp_path, "--prefill", "token", "--expert-budget", str(budget), "--hot-experts", str(hot))
    _same(resident, got)
    st = got[3]["expert_streaming"]
    # 4 slots per layer, top-2 needs 2 unpinned: at most 2 pinned per layer (the most-routed, experts 0 and 3)
    assert st["pinned_experts"] == 2 * layers and st["pinned_bytes"] <= 0.5 * budget and st["pool_bytes"] <= budget
    # "auto" finds hot_experts.json next to the file only (none here): no pins
    rec = _run(f, tmp_path, "--expert-budget", str(budget), "--hot-experts", "auto", steps=2, rows=False)[3]
    assert rec["expert_streaming"]["pinned_experts"] == 0
    err = _run(f, tmp_path, "--expert-budget", str(layers * 2 * slot - 64), ok=False)
    assert "cannot hold the minimum slots" in err


def test_qwen3_moe_batch_prefill_and_pipelined_decode(moe_files, tmp_path):
    f = moe_files["per_expert"]
    slot, layers = _slot_bytes(f, tmp_path)
    prompt = PROMPT + [11, 250, 3, 64, 128, 77, 401, 9]
    # every expert fits: chunks of 8 slots / top-2 = 4 tokens (the resident run with --chunk 4 has the same batches)
    resident = _run(f, tmp_path, "--prefill", "batch", "--chunk", "4", prompt=prompt)
    got = _run(f, tmp_path, "--prefill", "batch", "--expert-budget", str(layers * 8 * slot), prompt=prompt)
    _same(resident, got)
    # a 6-slot pool: chunks of 3 tokens
    resident = _run(f, tmp_path, "--prefill", "batch", "--chunk", "3", prompt=prompt)
    got = _run(f, tmp_path, "--prefill", "batch", "--expert-budget", str(layers * 6 * slot), prompt=prompt)
    _same(resident, got)
    assert got[3]["expert_streaming"]["evictions"] > 0
    # pipelined greedy decode (no per-step outputs): same tokens
    a = _run(f, tmp_path, rows=False, steps=16)[0]
    b = _run(f, tmp_path, "--expert-budget", str(layers * 3 * slot), "--expert-prefetch", rows=False, steps=16)
    assert a == b[0] and b[3]["expert_streaming"]["misses"] > 0


def test_routed_experts_stay_out_of_the_gpu_views(tmp_path):
    """Experts large enough to span whole pages: the streamed run's GPU views leave them out (the resident run maps
    the whole file), with the same results."""
    from test_cpp_generate import ARCH_MOE, EXPERT_CODECS, _moe_file

    f = _moe_file(tmp_path / "big.suc1", dict(ARCH_MOE, num_experts=16, moe_intermediate_size=1024),
                  expert_codecs=EXPERT_CODECS)
    resident = _run(f, tmp_path, "--prefill", "token", steps=4)
    slot, layers = _slot_bytes(f, tmp_path)
    got = _run(f, tmp_path, "--prefill", "token", "--expert-budget", str(layers * 3 * slot), steps=4)
    _same(resident, got)
    st = got[3]["expert_streaming"]
    excluded = got[3]["excluded_from_gpu_views"]
    assert excluded >= 0.8 * st["expert_bytes_total"]
    assert got[3]["memory"]["mapped_file"] <= resident[3]["memory"]["mapped_file"] - excluded
    assert st["pool_bytes"] <= layers * 3 * slot


@pytest.fixture(scope="module")
def q4_file(tmp_path_factory):
    from test_cpp_qwen4 import _compress

    return _compress(tmp_path_factory.mktemp("xs_q4"))


@pytest.mark.parametrize("opts", [(), ("--expert-prefetch", "--expert-policy", "lru")])
def test_qwen4_exp_streamed_equals_resident(q4_file, opts, tmp_path):
    resident = _run(q4_file, tmp_path, prompt=[5, 6, 7, 8], steps=12)
    slot, layers = _slot_bytes(q4_file, tmp_path)
    k = 2                                             # tiny qwen4_exp: 8 experts, top-2
    budget = layers * (k + 1) * slot
    got = _run(q4_file, tmp_path, "--expert-budget", str(budget), *opts, prompt=[5, 6, 7, 8], steps=12)
    _same(resident, got)
    st = got[3]["expert_streaming"]
    assert st["pool_bytes"] <= budget and st["misses"] > 0 and st["evictions"] > 0


def test_streamed_decode_speed_close_to_resident(moe_files, tmp_path):
    """Perf regression guard: with a budget of top-k + 1 slots per layer (misses and evictions on most steps), the
    streamed greedy decode stays within a fixed per-MoE-layer allowance of the resident decode, and the slot pools are
    in the queue's residency set. (GLM-5.3-Flash on the Studio decoded at 2.58 s/token with GPU time 0.14 s: each of
    the 42 per-layer command buffers re-made its multi-GB pool resident; the residency set removes that.) Medians of
    three runs each; the allowance (1 ms per MoE layer: one host sync plus the reads of a few small experts, against
    ~0.45 ms measured on the M3) is loose for a busy machine but far below a per-split residency cost."""
    f = moe_files["per_expert"]
    slot, layers = _slot_bytes(f, tmp_path)
    budget = layers * 3 * slot

    def median_ms(*extra):
        best, rec = None, None
        for _ in range(3):
            rec = _run(f, tmp_path, *extra, rows=False, steps=48)[3]
            ms = sorted(rec["decode_ms_per_token"])
            best = ms[len(ms) // 2] if best is None else min(best, ms[len(ms) // 2])
        return best, rec

    resident, _ = median_ms()
    streamed, rec = median_ms("--expert-budget", str(budget))
    st = rec["expert_streaming"]
    assert st["misses"] > 0 and st["evictions"] > 0
    if rec.get("residency_set"):
        assert rec["residency_set_bytes"] >= st["pool_bytes"] and rec["residency_skipped_bytes"] == 0
    print(f"decode ms/token: resident {resident:.3f}, streamed {streamed:.3f} ({layers} MoE layers, "
          f"hit rate {st['hit_rate']:.3f})")
    assert streamed <= 1.5 * resident + 1.0 * layers, (resident, streamed)
