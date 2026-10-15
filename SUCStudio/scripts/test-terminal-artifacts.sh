#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
OUT="$ROOT/.build/$(uname -m)"
mkdir -p "$OUT/swift-cache"
xcrun clang++ -std=c++17 "$ROOT/Engine/studio-terminal.cpp" -o "$OUT/studio-terminal-test"
"$PYTHON_BIN" "$ROOT/tests/terminal_control_fixture.py" "$OUT/studio-terminal-test"
CORE=()
for file in "$ROOT"/Sources/*.swift; do
  case "$file" in */*Views.swift|*/Views.swift|*/SUCStudioApp.swift) ;; *) CORE+=("$file");; esac
done
xcrun swiftc -swift-version 5 -parse-as-library -module-cache-path "$OUT/swift-cache" "${CORE[@]}" "$ROOT/tests/TerminalArtifactTests.swift" -o "$OUT/terminal-artifact-tests"
"$OUT/terminal-artifact-tests" "$OUT/studio-terminal-test"
