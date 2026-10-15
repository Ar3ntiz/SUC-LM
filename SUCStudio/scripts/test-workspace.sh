#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.build/$(uname -m)"
xcrun clang++ -std=c++17 "$ROOT/Engine/studio-command.cpp" -o "$OUT/studio-command-test"
CORE=()
for file in "$ROOT"/Sources/*.swift; do
  case "$file" in */*Views.swift|*/Views.swift|*/SUCStudioApp.swift) ;; *) CORE+=("$file");; esac
done
xcrun swiftc -swift-version 5 -parse-as-library -module-cache-path "$OUT/swift-cache" "${CORE[@]}" "$ROOT/tests/WorkspaceTests.swift" -o "$OUT/workspace-tests"
"$OUT/workspace-tests" "$OUT/studio-command-test"
xcrun swiftc -swift-version 5 -parse-as-library -module-cache-path "$OUT/swift-cache" "${CORE[@]}" "$ROOT/tests/AutomationTests.swift" -o "$OUT/automation-tests"
"$OUT/automation-tests" "$OUT/studio-command-test"
