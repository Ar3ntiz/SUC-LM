#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk}"
OUT="$ROOT/.build/universal-tests"
mkdir -p "$OUT/swift-cache"
SOURCES=()
for file in "$ROOT"/Sources/*.swift; do [[ "$file" == */SUCStudioApp.swift ]] || SOURCES+=("$file"); done
xcrun swiftc -swift-version 5 -whole-module-optimization -sdk "$SDK" -module-cache-path "$OUT/swift-cache" -parse-as-library "${SOURCES[@]}" "$ROOT/tests/ArtifactInspectorUITests.swift" -o "$OUT/artifact-inspector-tests"
RESULT="$OUT/artifact-inspector-ui-result.json"
rm -f "$RESULT"
"$OUT/artifact-inspector-tests" "$RESULT"
[[ -s "$RESULT" ]] || { echo "Test UI interrotto prima della conferma finale" >&2; exit 1; }
