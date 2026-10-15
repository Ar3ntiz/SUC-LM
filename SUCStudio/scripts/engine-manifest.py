#!/usr/bin/env python3
"""Fingerprint actual parent engine inputs, including uncommitted edits."""
import hashlib, json, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
files = {}
for base in ("include", "src", "metal", "third_party", "tools", "tests"):
    for path in sorted((root / base).rglob("*")):
        if path.is_file() and path.name != ".DS_Store" and "__pycache__" not in path.parts:
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
try:
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
except (OSError, subprocess.CalledProcessError):
    commit = None
content = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
print(json.dumps({"source": "../cpp", "repository_commit": commit,
    "content_sha256": content, "files": files}, indent=2, sort_keys=True))
