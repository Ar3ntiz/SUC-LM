#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk}"
OUT="$ROOT/.build/customization14602-tests"
mkdir -p "$OUT/swift-cache"
CORE=()
for file in "$ROOT"/Sources/*.swift; do
 case "$file" in */*Views.swift|*/Views.swift|*/SUCStudioApp.swift) ;; *) CORE+=("$file");; esac
done
xcrun clang++ -std=c++17 "$ROOT/Engine/studio-command.cpp" -o "$OUT/command-helper"
xcrun swiftc -swift-version 5 -whole-module-optimization -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/swift-cache" "${CORE[@]}" "$ROOT/tests/Customization14602Tests.swift" -o "$OUT/customization-tests"
"$OUT/customization-tests" "$OUT/command-helper" "$ROOT/tests/mcp_fixture.py" "$ROOT/examples/portable-review"
