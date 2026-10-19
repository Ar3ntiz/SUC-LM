#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/scripts/test-universal.sh"
# The engine remains ARM-only. Never attempt to execute it on Intel.
if [[ "$(uname -m)" == arm64 && -x "$ROOT/.build/arm64/suc-tests" ]]; then
  "$ROOT/.build/arm64/suc-tests"
fi
