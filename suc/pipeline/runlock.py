"""Exclusive per-run-directory lock (audit F01).

The lock is an OS advisory ``flock`` on an open descriptor of ``<run_dir>/.lock``, acquired non-blocking and held for
the whole run; the kernel releases it when the descriptor is closed or the process dies. The PID written into the
file is diagnostic metadata only, written after ownership is acquired. The file is never unlinked, so every process
locks the same inode (unlinking would let a waiter lock a stale inode while a newcomer locks a fresh one).

Kept free of MLX imports so that multiprocess tests can run on any machine.
"""
from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class RunLocked(RuntimeError):
    """Another process owns the run directory."""


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """One process per run directory (two unattended runs on the same artifacts would corrupt them)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / ".lock"
    fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            owner = os.pread(fd, 64, 0).decode(errors="replace").strip() or "?"
            raise RunLocked(f"run directory {run_dir} is in use by process {owner}") from None
        os.ftruncate(fd, 0)
        os.pwrite(fd, str(os.getpid()).encode(), 0)
        try:
            yield
        finally:
            try:
                os.ftruncate(fd, 0)        # diagnostic only; the inode (and the path) stay
            except OSError:
                pass
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
