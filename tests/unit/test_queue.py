"""`suc queue` orchestrator: fake Hub / download / compress children; no network, no MLX work."""
from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from suc.config import load_compress_config
from suc.orchestrate import hub as H
from suc.orchestrate.config import load_queue_config, queue_config_from_dict
from suc.orchestrate.orchestrator import Orchestrator, QueueLocked, dir_bytes
from suc.orchestrate.report import PROGRESS_LINE, CompressProgress, ProgressLog
from tests.unit.queue_fakes import FakeHub, make_remote, write_scripts

REPO = Path(__file__).resolve().parents[2]
REV = "a" * 40
REV2 = "b" * 40
FAST_POLL = {"progress_seconds": 0.0, "heartbeat_seconds": 0.05, "decisions_seconds": 0.05, "child_seconds": 0.02}


@pytest.fixture
def env(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    monkeypatch.setenv("FAKE_REMOTE", str(remote))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")          # must be removed for the download child
    fetch, compress = write_scripts(tmp_path)
    return {"tmp": tmp_path, "remote": remote, "fetch": fetch, "compress": compress}


def queue(env, models, **kw) -> dict:
    t = env["tmp"]
    d = {"name": "test_queue", "queue_dir": str(t / "q"), "hf_cache": str(t / "cache"),
         "models": models, "poll": dict(FAST_POLL),
         "space": {"min_free_gib": 0.0, "workspace_fixed_gib": 0.0, "workspace_factor": 1.0, "estimated_bpp": 16.0,
                   "watchdog_seconds": 0.02},
         "download": {"command": env["fetch"]},
         "compress": {"command": env["compress"], "output_dir": str(t / "results_out"),
                      "template_dense": str(REPO / "experiments/configs/compress_qwen3_8b_final.yaml"),
                      "template_moe": str(REPO / "experiments/configs/compress_qwen3_30b_a3b_final.yaml")}}
    for k, v in kw.items():
        d[k] = {**d[k], **v} if isinstance(v, dict) and isinstance(d.get(k), dict) else v
    return d


def orch(env, d, disk=lambda p: 10 ** 15) -> Orchestrator:
    return Orchestrator(queue_config_from_dict(d), hub=FakeHub(env["remote"]), disk_free=disk, repo_dir=env["tmp"],
                        env_info={}, install_signals=False)


def status(env) -> dict:
    s = json.loads((env["tmp"] / "q" / "status.json").read_text())
    return {r["id"]: r for r in s["models"]}


def progress(env) -> list[str]:
    return (env["tmp"] / "q" / "progress.log").read_text().splitlines()


# ------------------------------------------------------------------------------------------- formats / config

def test_progress_line_format(tmp_path):
    log = ProgressLog(tmp_path / "progress.log", 6)
    line = log.line("INFO", 2, "org/model", "download", "45.2% 1.00/2.00 GiB,\n 85.1 MB/s")
    m = PROGRESS_LINE.match(line)
    assert m and m.group(2) == "INFO" and m.group(3) == "2" and m.group(4) == "6" and m.group(5) == "org/model"
    assert m.group(6) == "download" and "\n" not in line
    assert (tmp_path / "progress.log").read_text() == line + "\n"


def test_compress_log_parsing_and_eta():
    t = [100.0]
    p = CompressProgress(clock=lambda: t[0])
    assert p.feed("2026-10-03T00:00:00Z [compress] stage encode: start") == ("INFO", "stage encode start")
    assert "1/4" in p.feed("2026-10-03T00:00:00Z [compress] encode model.layers.0 (1/4): 7 tensors")[1]
    t[0] = 110.0
    lvl, msg = p.feed("2026-10-03T00:00:00Z [compress] encode model.layers.1 (2/4): 7 tensors")
    assert lvl == "INFO" and "2/4" in msg and "ETA 20s" in msg and p.snapshot()["block"] == [2, 4]
    assert p.feed("x [compress] preflight WARNING: low RAM")[0] == "WARN"
    assert p.feed("x [compress] stage encode: FAILED (ValueError: boom); rerun")[0] == "ERROR"
    assert p.feed("random noise") is None


def test_frontier_queue_file_loads_in_order():
    cfg = load_queue_config(REPO / "experiments/queues/frontier_models.yaml")
    assert [(m.id, m.revision) for m in cfg.models] == [
        ("Qwen/Qwen3.8-Flash-Next", "de4b8e4d43b917e7706784d8bb445c9af86a3540"),
        ("deepseek-ai/DeepSeek-V4-Flash-0731", "7872f01b1d1fe23eabc4c98b48bffcef5a386062"),
        ("zai-org/GLM-5.3-Flash", "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"),
        ("deepseek-ai/DeepSeek-V4.1-Flash", "2cba9e42aa026125f3ed06c6d98c1db82f7ca027"),
        ("moonshotai/Kimi-K3", "f831ab66814297da540d832a5235f8e904f29d06"),
        ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a")]
    assert [m.id for m in cfg.ordered()] == ["Qwen/Qwen3.8-Flash-Next"]   # user decision: Qwen3.8 first, then ask
    assert cfg.source_policy == "keep"
    for t in (cfg.compress.template_dense, cfg.compress.template_moe):
        assert (REPO / t).exists()


def test_queue_config_rejects_unknown_keys_and_unpinned(env):
    with pytest.raises(ValueError, match="unknown"):
        queue_config_from_dict(queue(env, [{"id": "a/b", "revision": REV, "bogus": 1}]))
    with pytest.raises(ValueError, match="pinned"):
        queue_config_from_dict(queue(env, [{"id": "a/b", "revision": "main"}]))


def test_priority_order(env):
    cfg = queue_config_from_dict(queue(env, [{"id": "a/one", "revision": REV}, {"id": "a/two", "revision": REV,
                                                                                "priority": 5}]))
    assert [m.id for m in cfg.ordered()] == ["a/two", "a/one"]


def test_generated_compress_config_moe_and_overrides(env):
    make_remote(env["remote"], "org/moe", model_type="qwen3_moe", fixture="qwen3_30b_a3b_config.json")
    d = queue(env, [{"id": "org/moe", "revision": REV,
                     "overrides": {"eval": {"kl_sequences": 8}, "calibration": {"n_calibration": 128}}}],
              overrides={"verify_tensors": 4})
    o = orch(env, d)
    e = o.entries[0]
    r: dict = {"id": e.id}
    o.step_check(e, r)
    assert r["check"]["arch"]["is_moe"] and r["check"]["supported"]
    c = o.compress_config(e, r)
    assert c["model"] == {"id": "org/moe", "revision": REV, "role": "target"}
    assert c["plan"]["allocation_unit"] == "expert_stack" and c["verify_tensors"] == 4
    assert c["eval"]["kl_sequences"] == 8 and c["calibration"]["n_calibration"] == 128
    assert c["calibration"]["seq_len"] == 512               # template value kept by the deep merge
    p = env["tmp"] / "c.yaml"
    p.write_text(yaml.safe_dump(c))
    assert load_compress_config(p).run_dir == str(env["tmp"] / "q" / "runs" / "org__moe")
    assert c["calibration"]["mix"] == "v3"                  # queue default for new runs replaces the template's v1
    assert load_compress_config(p).calibration.mix["math_competition"] == 0.03


def test_generated_compress_config_calibration_mix_pins(env):
    from suc.config import CALIB_MIX_V1

    make_remote(env["remote"], "org/moe", model_type="qwen3_moe", fixture="qwen3_30b_a3b_config.json")
    for models, kw, want in (
            ([{"id": "org/moe", "revision": REV, "overrides": {"calibration": {"mix": "v1"}}}], {}, "v1"),
            ([{"id": "org/moe", "revision": REV}], {"overrides": {"calibration": {"mix": "v1"}}}, "v1"),
            ([{"id": "org/moe", "revision": REV}], {"compress": {"calibration_mix": None}}, CALIB_MIX_V1)):
        o = orch(env, queue(env, models, **kw))
        e = o.entries[0]
        r: dict = {"id": e.id}
        o.step_check(e, r)
        assert o.compress_config(e, r)["calibration"]["mix"] == want
    with pytest.raises(ValueError, match="calibration_mix"):
        queue_config_from_dict(queue(env, [{"id": "a/b", "revision": REV}], compress={"calibration_mix": "v9"}))


def test_size_summary_selects_like_snapshot_download(env):
    make_remote(env["remote"], "org/m")
    meta = FakeHub(env["remote"]).metadata("org/m", REV)
    from suc.orchestrate.config import DEFAULT_ALLOW, DEFAULT_IGNORE
    s = H.size_summary(meta, DEFAULT_ALLOW, DEFAULT_IGNORE)
    assert s["safetensors_files"] == 2 and s["download_bytes"] < s["repo_bytes"]     # original/ ignored


# ------------------------------------------------------------------------------------------- state machine

def test_unsupported_architecture_is_skipped_and_queue_continues(env):
    make_remote(env["remote"], "org/alien", model_type="alien_arch")
    make_remote(env["remote"], "org/good")
    d = queue(env, [{"id": "org/alien", "revision": REV}, {"id": "org/good", "revision": REV}])
    assert orch(env, d).run() == 0
    s = status(env)
    assert s["org/alien"]["state"] == "unsupported" and "alien_arch" in s["org/alien"]["reason"]
    assert "download" not in s["org/alien"]["steps"]
    assert s["org/good"]["state"] == "done" and s["org/good"]["result"]["bpp"] == 4.31
    q = env["tmp"] / "q"
    summary = (q / "SUMMARY.md").read_text()
    assert "org/alien" in summary and "unsupported" in summary and "4.310" in summary
    assert (q / "reports" / "org__alien.md").exists() and (q / "reports" / "org__good.json").exists()
    assert (q / "results" / "20261003T000000Z_queue_org__good.json").exists()
    lines = progress(env)
    assert all(PROGRESS_LINE.match(ln) for ln in lines), [ln for ln in lines if not PROGRESS_LINE.match(ln)]
    assert any("UNSUPPORTED" in ln for ln in lines)
    assert any("encode block 3/3" in ln for ln in lines)                    # compress sub-progress parsed
    # downloaded files are verified and original/ was never fetched into the snapshot
    snap = H.snapshot_dir(env["tmp"] / "cache", "org/good", REV)
    assert s["org/good"]["verify"]["sha256_checked"] == 2 and not (snap / "original").exists()


def test_failure_is_recorded_and_queue_continues(env):
    make_remote(env["remote"], "org/broken")
    make_remote(env["remote"], "org/good")
    d = queue(env, [{"id": "org/broken", "revision": REV}, {"id": "org/good", "revision": REV}])
    assert orch(env, d).run() == 0
    s = status(env)
    err = s["org/broken"]["error"]
    assert s["org/broken"]["state"] == "failed" and err["step"] == "compress"
    assert "Traceback" in err["traceback"] and "fake encode failure" in err["log_tail"]
    assert s["org/good"]["state"] == "done"
    rep = (env["tmp"] / "q" / "reports" / "org__broken.md").read_text()
    assert "fake encode failure" in rep and "StepFailed" in rep
    assert any(" ERROR " in ln and "org/broken" in ln for ln in progress(env))


def test_hub_not_found_fails_and_continues(env):
    make_remote(env["remote"], "org/good")
    d = queue(env, [{"id": "org/missing", "revision": REV}, {"id": "org/good", "revision": REV}])
    orch(env, d).run()
    s = status(env)
    assert s["org/missing"]["state"] == "failed" and s["org/missing"]["error"]["step"] == "check"
    assert s["org/good"]["state"] == "done"


def test_insufficient_space_awaits_decision_then_resumes_after_approval(env):
    make_remote(env["remote"], "org/big", values=20_000)
    make_remote(env["remote"], "org/small", values=64)
    d = queue(env, [{"id": "org/big", "revision": REV}, {"id": "org/small", "revision": REV}],
              wait_for_decisions=False)
    free = 60_000                                  # big needs ~2 x 80 kB, small ~2 x 0.5 kB
    assert orch(env, d, disk=lambda p: free).run() == 3
    s = status(env)
    assert s["org/big"]["state"] == "awaiting_decision" and s["org/big"]["awaiting"]["reason"] == "space"
    assert s["org/small"]["state"] == "done"                  # the next model that fits ran
    q = env["tmp"] / "q"
    aw = (q / "AWAITING_DECISION.md").read_text()
    assert "org/big" in aw and "short by" in aw and "decisions.yaml" in aw
    assert "download" not in s["org/big"]["steps"]
    (q / "decisions.yaml").write_text('"org/big": approve\n')
    assert orch(env, d, disk=lambda p: free).run() == 0
    s = status(env)
    assert s["org/big"]["state"] == "done" and s["org/big"]["space"]["approved"]
    assert not (q / "AWAITING_DECISION.md").exists()


def test_waiting_queue_rereads_decisions_periodically(env):
    make_remote(env["remote"], "org/big", values=20_000)
    d = queue(env, [{"id": "org/big", "revision": REV}])           # wait_for_decisions: true (default)
    q = env["tmp"] / "q"

    def decide():
        for _ in range(200):
            if (q / "AWAITING_DECISION.md").exists():
                (q / "decisions.yaml").write_text('"org/big": skip\n')
                return
            time.sleep(0.02)
    t = threading.Thread(target=decide)
    t.start()
    assert orch(env, d, disk=lambda p: 1000).run() == 0
    t.join()
    assert status(env)["org/big"]["state"] == "skipped"


def test_space_freed_by_hand_resumes_without_decision(env):
    make_remote(env["remote"], "org/big", values=20_000)
    d = queue(env, [{"id": "org/big", "revision": REV}])
    free = {"v": 1000}
    o = orch(env, d, disk=lambda p: free["v"])

    def free_space():
        for _ in range(200):
            if (env["tmp"] / "q" / "AWAITING_DECISION.md").exists():
                free["v"] = 10 ** 12
                return
            time.sleep(0.02)
    t = threading.Thread(target=free_space)
    t.start()
    assert o.run() == 0
    t.join()
    assert status(env)["org/big"]["state"] == "done"


def test_delete_source_of_frees_space_for_the_next_model(env):
    make_remote(env["remote"], "org/first", values=20_000)
    make_remote(env["remote"], "org/second", values=20_000)
    d = queue(env, [{"id": "org/first", "revision": REV}, {"id": "org/second", "revision": REV2}],
              wait_for_decisions=False)
    cap = 250_000                  # one model (download 80 kB + work 80 kB, x1.15) fits; two cached sources do not

    def disk(p):
        return cap - dir_bytes(env["tmp"] / "cache") - dir_bytes(env["tmp"] / "q" / "runs")
    assert orch(env, d, disk=disk).run() == 3
    s = status(env)
    assert s["org/first"]["state"] == "done" and s["org/second"]["state"] == "awaiting_decision"
    assert "org/first" in (env["tmp"] / "q" / "AWAITING_DECISION.md").read_text()
    (env["tmp"] / "q" / "decisions.yaml").write_text('"org/second":\n  delete_source_of: org/first\n')
    assert orch(env, d, disk=disk).run() == 0
    s = status(env)
    assert s["org/second"]["state"] == "done"
    assert not H.repo_cache_dir(env["tmp"] / "cache", "org/first").exists()
    assert s["org/first"]["disk"]["source_deleted"]["by"] == "org/second"


def test_delete_source_of_refuses_a_model_that_still_needs_it(env):
    make_remote(env["remote"], "org/a", values=20_000)
    make_remote(env["remote"], "org/b", values=20_000)
    d = queue(env, [{"id": "org/a", "revision": REV}, {"id": "org/b", "revision": REV}], wait_for_decisions=False)
    (env["tmp"] / "q").mkdir()
    (env["tmp"] / "q" / "decisions.yaml").write_text('"org/a":\n  delete_source_of: org/b\n')
    orch(env, d, disk=lambda p: 1000).run()
    assert any("refused" in ln for ln in progress(env))


def test_crash_resume_does_not_redo_finished_steps(env, monkeypatch):
    make_remote(env["remote"], "org/m")
    d = queue(env, [{"id": "org/m", "revision": REV}])
    o = orch(env, d)

    class Crash(BaseException):
        pass

    def crash(e, r):
        raise Crash()
    monkeypatch.setattr(o, "step_compress", crash)
    with pytest.raises(Crash):
        o.run()
    s = status(env)["org/m"]
    assert s["state"] == "running" and s["step"] == "compress"
    assert {k for k, v in s["steps"].items() if v["status"] == "done"} == {"check", "space", "download", "verify"}
    o2 = orch(env, d)                                  # a new process after the crash: the lock was released
    assert o2.run() == 0
    s = status(env)["org/m"]
    assert s["state"] == "done" and o2.hub.calls == []                 # check not repeated
    assert (env["remote"] / "fetch_calls.txt").read_text().splitlines() == ["org/m"]   # downloaded once
    assert any("resuming at step compress" in ln for ln in progress(env))


def test_stop_request_interrupts_child_and_resumes(env, monkeypatch):
    make_remote(env["remote"], "org/m")
    monkeypatch.setenv("FAKE_COMPRESS_SLEEP", "1.0")
    d = queue(env, [{"id": "org/m", "revision": REV}])
    o = orch(env, d)

    def stopper():
        for _ in range(500):
            r = next((m for m in o.state.get("models", []) if m["id"] == "org/m"), {})
            if r.get("child", {}).get("step") == "compress":
                time.sleep(0.3)
                o._on_signal(15, None)
                return
            time.sleep(0.02)
    t = threading.Thread(target=stopper)
    t.start()
    assert o.run() == 143
    t.join()
    s = status(env)["org/m"]
    assert s["state"] == "pending" and s["steps"]["compress"]["status"] == "interrupted" and "child" not in s
    monkeypatch.setenv("FAKE_COMPRESS_SLEEP", "0")
    assert orch(env, d).run() == 0
    assert status(env)["org/m"]["state"] == "done"
    assert any("SIGTERM" in ln for ln in progress(env))


def test_low_disk_watchdog_stops_download(env, monkeypatch):
    make_remote(env["remote"], "org/m", shards=4)
    monkeypatch.setenv("FAKE_FETCH_SLEEP", "0.3")
    d = queue(env, [{"id": "org/m", "revision": REV}], wait_for_decisions=False, space={"min_free_gib": 1e-6})

    def disk(p):
        return 1 if (env["remote"] / "fetch_started").exists() else 10 ** 12
    assert orch(env, d, disk=disk).run() == 3
    s = status(env)["org/m"]
    assert s["state"] == "awaiting_decision" and s["awaiting"]["reason"] == "low_disk"
    assert s["awaiting"]["step"] == "download"
    assert any("low disk" in ln and " WARN " in ln for ln in progress(env))


def test_compress_preflight_disk_error_awaits_decision(env, monkeypatch):
    make_remote(env["remote"], "org/m")
    monkeypatch.setenv("FAKE_COMPRESS_NODISK", "1")
    d = queue(env, [{"id": "org/m", "revision": REV}], wait_for_decisions=False)
    assert orch(env, d).run() == 3
    s = status(env)["org/m"]
    assert s["state"] == "awaiting_decision" and "insufficient disk" in s["awaiting"]["log_line"]


def test_truncated_download_is_fetched_again_once(env):
    make_remote(env["remote"], "org/m")
    (env["remote"] / "truncate_once").write_text("1")
    d = queue(env, [{"id": "org/m", "revision": REV}])
    assert orch(env, d).run() == 0
    s = status(env)["org/m"]
    assert s["state"] == "done" and s["redownloads"] == 1 and s["verify"]["problems"] == []
    assert len((env["remote"] / "fetch_calls.txt").read_text().splitlines()) == 2


def test_retry_decision_after_failure(env):
    make_remote(env["remote"], "org/broken")
    d = queue(env, [{"id": "org/broken", "revision": REV}])
    orch(env, d).run()
    assert status(env)["org/broken"]["state"] == "failed"
    (env["tmp"] / "q" / "decisions.yaml").write_text('"org/broken": retry\n')
    orch(env, d).run()
    s = status(env)["org/broken"]
    assert s["state"] == "failed" and s["retried"] == 1          # retried once (fails again), not in a loop
    runs = (env["tmp"] / "q" / "runs" / "org__broken" / "attempts.txt").read_text().splitlines()
    assert len(runs) == 2


def test_source_policy_delete_after_success(env):
    make_remote(env["remote"], "org/m")
    d = queue(env, [{"id": "org/m", "revision": REV, "source_policy": "delete_after_success"}])
    orch(env, d).run()
    assert status(env)["org/m"]["state"] == "done"
    assert not H.repo_cache_dir(env["tmp"] / "cache", "org/m").exists()


def test_single_instance_lock(env):
    make_remote(env["remote"], "org/m")
    d = queue(env, [{"id": "org/m", "revision": REV}])
    q = env["tmp"] / "q"
    q.mkdir()
    with open(q / ".lock", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(QueueLocked):
            orch(env, d).run()


def test_download_child_env_has_no_offline_flag_and_compress_child_is_offline(env):
    make_remote(env["remote"], "org/m")
    o = orch(env, queue(env, [{"id": "org/m", "revision": REV}]))
    assert "HF_HUB_OFFLINE" not in o._env(offline=False)
    assert o._env(offline=True)["HF_HUB_OFFLINE"] == "1"
    assert o._env(offline=False)["HF_HUB_CACHE"] == str(env["tmp"] / "cache")


def test_orphaned_child_of_a_crashed_run_is_stopped(env):
    make_remote(env["remote"], "org/m")
    o = orch(env, queue(env, [{"id": "org/m", "revision": REV}]))
    marker = str(env["tmp"] / "marker_config.yaml")
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", marker], start_new_session=True)
    o._load_state()
    o.rec("org/m")["child"] = {"pid": p.pid, "marker": marker, "step": "compress"}
    o.kill_orphans()
    assert p.wait(10) is not None and "child" not in o.rec("org/m")


def test_heartbeat_updates_status(env, monkeypatch):
    make_remote(env["remote"], "org/m")
    monkeypatch.setenv("FAKE_COMPRESS_SLEEP", "0.3")
    o = orch(env, queue(env, [{"id": "org/m", "revision": REV}]))
    beats = set()
    finished = threading.Event()

    def watch():
        while not finished.is_set():
            try:
                beats.add(json.loads((env["tmp"] / "q" / "status.json").read_text())["heartbeat"])
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.05)
    t = threading.Thread(target=watch)
    t.start()
    o.run()
    finished.set()
    t.join()
    assert len(beats) >= 2


# ------------------------------------------------------------------------------------------- CLI / fetch child

def test_dry_run_reports_every_model_without_downloading(env):
    from suc.orchestrate.cli import dry_run

    make_remote(env["remote"], "org/alien", model_type="alien_arch")
    make_remote(env["remote"], "org/good")
    p = env["tmp"] / "queue.yaml"
    p.write_text(yaml.safe_dump(queue(env, [{"id": "org/alien", "revision": REV}, {"id": "org/good", "revision": REV},
                                            {"id": "org/missing", "revision": REV}])))
    out = dry_run(p, hub=FakeHub(env["remote"]), disk_free=lambda p: 10 ** 12, repo_dir=env["tmp"])
    assert "org/alien: UNSUPPORTED" in out and "org/good: qwen3" in out and "fits True" in out
    assert "org/missing: ERROR" in out
    assert not (env["tmp"] / "cache").exists() and not (env["tmp"] / "q" / "status.json").exists()


def test_cli_status_and_stop_without_a_running_queue(env, capsys):
    from suc.cli.main import main

    p = env["tmp"] / "queue.yaml"
    p.write_text(yaml.safe_dump(queue(env, [{"id": "org/m", "revision": REV}])))
    with pytest.raises(SystemExit) as ex:
        main(["queue", str(p), "--status"])
    assert ex.value.code == 0 and "no SUMMARY.md" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["queue", str(p), "--stop"])
    assert "no running queue" in capsys.readouterr().out


def test_fetch_child_retries_with_backoff(monkeypatch, capsys):
    import huggingface_hub

    from suc.orchestrate import fetch

    calls = []

    def flaky(repo, **kw):
        calls.append(kw)
        if len(calls) < 3:
            raise ConnectionError("network down")
        return "/cache/snap"
    monkeypatch.setattr(huggingface_hub, "snapshot_download", flaky)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    rc = fetch.main(["--repo", "a/b", "--revision", REV, "--cache-dir", "/c", "--allow", "*.json", "--ignore",
                     "original/*", "--backoff", "0", "--retries", "5"])
    out = capsys.readouterr().out
    assert rc == 0 and len(calls) == 3 and "SNAPSHOT /cache/snap" in out and "attempt 2/5 failed" in out
    assert calls[0]["allow_patterns"] == ["*.json"] and calls[0]["revision"] == REV
    import os
    assert "HF_HUB_OFFLINE" not in os.environ
