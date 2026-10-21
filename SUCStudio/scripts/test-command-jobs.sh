#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-$(xcrun --show-sdk-path)}"
OUT="$ROOT/.build/command-jobs-tests"
mkdir -p "$OUT/cache"
xcrun clang++ -std=c++17 "$ROOT/Engine/studio-command.cpp" -o "$OUT/studio-command-test"
CORE=()
for file in "$ROOT"/Sources/*.swift; do
 case "$file" in */*Views.swift|*/Views.swift|*/SUCStudioApp.swift) ;; *) CORE+=("$file");; esac
done
xcrun swiftc -swift-version 5 -whole-module-optimization -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/cache" "${CORE[@]}" "$ROOT/tests/CommandJobsTests.swift" -o "$OUT/command-jobs-tests"
"$OUT/command-jobs-tests" "$OUT/studio-command-test"
