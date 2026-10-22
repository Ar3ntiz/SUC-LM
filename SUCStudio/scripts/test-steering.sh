#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-$(xcrun --show-sdk-path)}"
if [[ -z "${SDKROOT:-}" && -d /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk ]]; then SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk; fi
OUT="$ROOT/.build/steering-tests"
mkdir -p "$OUT/swift-cache"
CORE=()
for file in "$ROOT"/Sources/*.swift; do
 case "$file" in */*Views.swift|*/Views.swift|*/SUCStudioApp.swift) ;; *) CORE+=("$file");; esac
done
xcrun swiftc -swift-version 5 -whole-module-optimization -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/swift-cache" "${CORE[@]}" "$ROOT/tests/SteeringTests.swift" -o "$OUT/steering-tests"
xcrun clang++ -std=c++17 "$ROOT/Engine/studio-command.cpp" -o "$OUT/studio-command-test"
"$OUT/steering-tests" "$ROOT/tests/steering_fixture.py" "$OUT/studio-command-test"
