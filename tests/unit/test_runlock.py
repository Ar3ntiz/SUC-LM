"""Audit F01: the compression run lock is a real exclusive flock (multiprocess tests, CPU only)."""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

import pytest

from suc.pipeline.runlock import RunLocked, run_lock

_CTX = mp.get_context("spawn")


def _enter(run_dir: str, start, hold_s: float, q) -> None:
    """Wait for the start event, try the lock, report 'entered'/'locked', hold it for hold_s."""
    start.wait(10)
    try:
        with run_lock(Path(run_dir)):
            q.put(("entered", os.getpid()))
            time.sleep(hold_s)
    except RunLocked:
        q.put(("locked", os.getpid()))


def _hold_forever(run_dir: str, ready) -> None:
    with run_lock(Path(run_dir)):
        ready.set()
        time.sleep(60)


def _raise_inside(run_dir: str) -> None:
    with run_lock(Path(run_dir)):
        raise ValueError("boom")


def _results(q, n):
    return [q.get(timeout=20) for _ in range(n)]


def test_simultaneous_first_start_has_one_owner(tmp_path):
    for _ in range(3):                                   # repeat: the race must never admit two
        d = tmp_path / f"run{_}"
        start, q = _CTX.Event(), _CTX.Queue()
        ps = [_CTX.Process(target=_enter, args=(str(d), start, 1.0, q)) for _ in range(4)]
        for p in ps:
            p.start()
        time.sleep(1.0)                                  # every process is waiting on the event
        start.set()
        res = _results(q, 4)
        for p in ps:
            p.join(20)
        assert sorted(r[0] for r in res) == ["entered", "locked", "locked", "locked"]


def test_existing_owner_blocks_and_reports_pid(tmp_path):
    ready = _CTX.Event()
    owner = _CTX.Process(target=_hold_forever, args=(str(tmp_path), ready))
    owner.start()
    try:
        assert ready.wait(20)
        with pytest.raises(RunLocked, match=str(owner.pid)):
            with run_lock(tmp_path):
                pass
    finally:
        owner.kill()
        owner.join(20)


def test_owner_death_releases_lock(tmp_path):
    ready = _CTX.Event()
    owner = _CTX.Process(target=_hold_forever, args=(str(tmp_path), ready))
    owner.start()
    assert ready.wait(20)
    os.kill(owner.pid, signal.SIGKILL)                   # no finally block runs
    owner.join(20)
    assert (tmp_path / ".lock").exists()                 # stale PID file left behind ...
    with run_lock(tmp_path):                             # ... but the kernel released the flock
        assert (tmp_path / ".lock").read_text() == str(os.getpid())


def test_exception_releases_lock(tmp_path):
    p = _CTX.Process(target=_raise_inside, args=(str(tmp_path),))
    p.start()
    p.join(20)
    assert p.exitcode != 0
    with run_lock(tmp_path):
        pass
    with pytest.raises(ValueError):                      # same process: released on exception too
        with run_lock(tmp_path):
            raise ValueError
    with run_lock(tmp_path):
        pass


def test_third_entrant_while_another_waits(tmp_path):
    """Owner holds; a second process is refused; after the owner exits a third enters (same inode, never unlinked)."""
    start, q = _CTX.Event(), _CTX.Queue()
    owner = _CTX.Process(target=_enter, args=(str(tmp_path), start, 2.0, q))
    owner.start()
    start.set()
    assert q.get(timeout=20)[0] == "entered"
    ino = (tmp_path / ".lock").stat().st_ino
    s2 = _CTX.Event()
    second = _CTX.Process(target=_enter, args=(str(tmp_path), s2, 0.0, q))
    second.start()
    s2.set()
    assert q.get(timeout=20)[0] == "locked"
    owner.join(20)
    second.join(20)
    s3 = _CTX.Event()
    third = _CTX.Process(target=_enter, args=(str(tmp_path), s3, 0.0, q))
    third.start()
    s3.set()
    assert q.get(timeout=20)[0] == "entered"
    third.join(20)
    assert (tmp_path / ".lock").stat().st_ino == ino


def test_pid_file_is_not_the_lock(tmp_path):
    """A leftover PID file (old implementation, or a live unrelated PID) does not block a new owner."""
    (tmp_path / ".lock").write_text(str(os.getppid()))
    with run_lock(tmp_path):
        assert (tmp_path / ".lock").read_text() == str(os.getpid())
