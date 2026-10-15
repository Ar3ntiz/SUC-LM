"""scripts/studio_wait_idle.sh: only real SUC workloads (python running our entry points, suc1-* binaries, named queue
scripts) keep a queue waiting; a `tail -f` on a log whose name looks like a workload does not."""
from __future__ import annotations

import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "studio_wait_idle.sh"
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS ps columns (ucomm)")


@pytest.fixture
def env(tmp_path):
    token = f"suctest{uuid.uuid4().hex[:10]}"   # --filter: ignore real processes on the machine
    procs: list[subprocess.Popen] = []

    def spawn(argv, executable=None):
        p = subprocess.Popen(argv, executable=executable, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
        time.sleep(0.3)
        return p

    def check(*opts):
        r = subprocess.run(["bash", str(SCRIPT), "--check", "--filter", token, *opts], capture_output=True, text=True)
        assert r.returncode in (0, 1), r.stderr
        return r.returncode == 1

    yield token, tmp_path, spawn, check
    for p in procs:
        p.kill()
        p.wait()


def test_ignores_tail_and_non_python(env):
    token, tmp, spawn, check = env
    log = tmp / f"{token}_suc.cli.main_long_context_eval_suc1-generate.log"
    log.write_text("x\n")
    spawn(["tail", "-f", str(log)])
    spawn(["/bin/sh", "-c", f"sleep 60; : suc.cli.main system_quality.py {token}"])   # matching args, wrong executable
    assert not check()


def test_python_workload(env):
    token, _, spawn, check = env
    p = spawn([sys.executable, "-c", "import time; time.sleep(60)", "suc.cli.main", token])
    assert check()
    p.kill(); p.wait()
    assert not check()


def test_renamed_argv0(env):
    token, _, spawn, check = env
    spawn([f"/opt/{token}/suc1-generate", "60"], executable="/bin/sleep")   # sleep with argv[0] = suc1-generate
    assert check()


def test_fetch_optional(env):
    token, _, spawn, check = env
    spawn([sys.executable, "-c", "import time; time.sleep(60)", "-m", "suc.orchestrate.fetch", token])
    assert not check()
    assert check("--fetch")


def test_queue_script_exact_path(env):
    token, tmp, spawn, check = env
    q = tmp / f"{token}_queue.sh"
    q.write_text("sleep 60\n")
    (tmp / f"{token}_queue.sh.log").write_text("x\n")
    spawn(["tail", "-f", str(tmp / f"{token}_queue.sh.log")])
    assert not check("--script", str(q))
    spawn(["bash", str(q)])
    assert not check()
    assert check("--script", str(q))
    assert not check("--script", str(tmp / f"{token}_other.sh"))


def test_wait_returns_when_idle(env):
    token, _, spawn, _ = env
    p = spawn([sys.executable, "-c", "import time; time.sleep(60)", "long_context_eval.py", token])
    w = subprocess.Popen(["bash", str(SCRIPT), "--filter", token, "--interval", "0.2", "--quiet-checks", "2"],
                         stdout=subprocess.PIPE, text=True)
    time.sleep(1.0)
    assert w.poll() is None
    p.kill(); p.wait()
    out, _ = w.communicate(timeout=10)
    assert w.returncode == 0
    assert "waiting for:" in out and "long_context_eval.py" in out
