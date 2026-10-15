#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
source "$ROOT/scripts/engine-path.sh"
"$PYTHON_BIN" "$ROOT/tests/build_dependency_test.py"
ARCH="$(uname -m)"; export ARCH
"$ROOT/scripts/build-engine.sh"
OUT="$ROOT/.build/$ARCH"; SDK="$(xcrun --show-sdk-path)"
"$OUT/suc-tests"
xcrun clang++ -std=c++20 -isysroot "$SDK" -isystem "$SDK/usr/include/c++/v1" "$ROOT/tests/memory_policy_tests.cpp" -o "$OUT/memory-policy-tests"
"$OUT/memory-policy-tests"
xcrun clang++ -std=c++20 -O1 -ffp-contract=off -isysroot "$SDK" -isystem "$SDK/usr/include/c++/v1" -I"$ENGINE_ROOT/include" -isystem "$ENGINE_ROOT/third_party/metal-cpp" -I"$ENGINE_ROOT/tools" "$ROOT/tests/service_tests.cpp" "$OUT"/obj/*.o -framework Metal -framework Foundation -framework QuartzCore -o "$OUT/service-tests"
"$OUT/service-tests"
"$PYTHON_BIN" "$ROOT/tests/protocol_test.py" "$OUT/suc-studio-engine"
"$ROOT/scripts/test-swift.sh"
