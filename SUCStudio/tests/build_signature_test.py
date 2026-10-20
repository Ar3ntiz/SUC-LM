import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
script=(Path(__file__).resolve().parents[1]/"scripts/build-engine.sh").read_text()
code=re.search(r"<<'PYHASH'\n(.*?)\nPYHASH",script,re.S).group(1)
with tempfile.TemporaryDirectory() as td:
 p=Path(td)/"manifest.json"
 def signature(commit="a",content="same",flags="flags",compiler="clang"):
  p.write_text(json.dumps({"repository_commit":commit,"content_sha256":content}))
  return subprocess.check_output([sys.executable,"-c",code,str(p),flags,compiler],text=True).strip()
 baseline=signature()
 assert baseline==signature(commit="documentation-only")
 assert baseline!=signature(content="changed source")
 assert baseline!=signature(flags="different flags")
 assert baseline!=signature(compiler="new compiler")
print("Build cache: source/flags/compiler invalidate; documentation-only revision preserves objects: PASS")
