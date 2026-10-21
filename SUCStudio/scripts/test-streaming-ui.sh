#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk}"
OUT="$ROOT/.build/streaming-ui"
mkdir -p "$OUT/swift-cache"
FILES=()
for file in "$ROOT"/Sources/*.swift; do
 case "$file" in */SUCStudioApp.swift) ;; *) FILES+=("$file");; esac
done
xcrun swiftc -swift-version 5 -whole-module-optimization -O -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/swift-cache" "${FILES[@]}" "$ROOT/tests/StreamingUITests.swift" -o "$OUT/streaming-ui"
rm -f "$OUT/result.json"
"$OUT/streaming-ui" "$OUT/result.json"
[[ -s "$OUT/result.json" ]]
