#!/bin/bash
# Sequential baseline/candidate measurements; selected complete local-disk files only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:?isolated build directory}"
TAG="${2:?result tag}"
SOURCE="$HOME/Projects/SUC-LM/experiments/artifacts/compress"
PYTHON="$HOME/Projects/SUC-LM/.venv/bin/python"
export SUC_BENCH_BUILD_COMMIT="$(cat "$BUILD/GIT_COMMIT")"
export PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1 SUC_CPP_BUILD="$BUILD"
cd "$ROOT"
models=("compress_qwen3_8b_final/container/model.suc1" "compress_qwen3.8_flash_next_side8/container/qwen3.8_flash_next_side8/qwen3.8_flash_next_side8.suc1")
names=(q8 q38)
for index in 0 1; do
  file="$SOURCE/${models[$index]}"
  "$PYTHON" - "$file" <<'PY'
from pathlib import Path
import os,subprocess,sys
p=Path(sys.argv[1]);assert p.is_file() and not p.is_symlink() and p.stat().st_size>100_000_000,p
assert p.open('rb').read(4)==b'SUC1',p
mount=subprocess.check_output(['df','-P',str(p.resolve())],text=True).splitlines()[-1].split()[0]
assert mount.startswith('/dev/'),f'Non-local filesystem: {mount}'
print('LOCAL MODEL',p,p.stat().st_size,flush=True)
PY
  for length in short 2048; do
    for repeat in 1 2 3; do
      bash scripts/studio_wait_idle.sh --check
      args=(--prompt "The history of the printing press begins in the fifteenth century");if [[ "$length" != short ]];then args=(--long-prompt "$length");fi
      "$PYTHON" scripts/frontier_measure.py generate --build "$BUILD" --name "eff_${TAG}_${names[$index]}_${length}_r${repeat}" --suc1 "$file" --steps 64 "${args[@]}" -- --quiet
    done
  done
done
