#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:?isolated candidate build}"
PYTHON="$HOME/Projects/SUC-LM/.venv/bin/python"
export PYTHONPATH="$ROOT:$ROOT/tests/unit" HF_HUB_OFFLINE=1 SUC_CPP_BUILD="$BUILD"
cd "$ROOT"
bash scripts/studio_wait_idle.sh --check
"$BUILD/suc-tests"
"$BUILD/suc-prefix-tests"
"$BUILD/suc-request-params-tests"
"$BUILD/suc-sampling-bench"
"$BUILD/suc-attention-bench"
"$PYTHON" -m pytest tests/unit/test_cpp_efficiency.py tests/unit/test_cpp_server.py tests/unit/test_cpp_qwen4.py -q -s --tb=short
