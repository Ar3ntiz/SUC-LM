"""Download child of `suc queue`: one pinned snapshot into the HF cache, with retries and exponential backoff.

    python -m suc.orchestrate.fetch --repo ORG/NAME --revision SHA --cache-dir DIR \
        [--allow PAT ...] [--ignore PAT ...] [--retries N] [--backoff S] [--max-backoff S] [--max-workers W]

snapshot_download resumes partial files and skips complete ones, so a retry (or a rerun after a crash / SIGTERM)
continues where it stopped. Runs in its own process so the orchestrator can stop it (SIGTERM, low-disk watchdog)
and measure progress from the blob store. Prints `SNAPSHOT <path>` on success.
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="suc.orchestrate.fetch")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--allow", nargs="*", default=None)
    p.add_argument("--ignore", nargs="*", default=None)
    p.add_argument("--retries", type=int, default=12)
    p.add_argument("--backoff", type=float, default=30.0)
    p.add_argument("--max-backoff", type=float, default=900.0)
    p.add_argument("--max-workers", type=int, default=8)
    a = p.parse_args(argv)
    os.environ.pop("HF_HUB_OFFLINE", None)          # must be unset before huggingface_hub reads its constants
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

    delay = a.backoff
    for attempt in range(1, a.retries + 1):
        t0 = time.time()
        try:
            path = snapshot_download(a.repo, revision=a.revision, cache_dir=a.cache_dir, allow_patterns=a.allow,
                                     ignore_patterns=a.ignore, max_workers=a.max_workers)
            print(f"SNAPSHOT {path}", flush=True)
            return 0
        except (GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError) as ex:
            print(f"FATAL {type(ex).__name__}: {ex}", flush=True)
            return 2
        except KeyboardInterrupt:
            return 130
        except Exception as ex:      # network / server errors: retry, the download resumes
            print(f"attempt {attempt}/{a.retries} failed after {time.time() - t0:.0f} s: {type(ex).__name__}: {ex}",
                  flush=True)
            if attempt == a.retries:
                return 1
            print(f"retrying in {delay:.0f} s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, a.max_backoff)
    return 1


if __name__ == "__main__":
    sys.exit(main())
