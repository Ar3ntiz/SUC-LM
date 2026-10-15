#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.build/$(uname -m)"
SOURCES=()
for file in "$ROOT"/Sources/*.swift; do [[ "$file" == */SUCStudioApp.swift ]] || SOURCES+=("$file");done
xcrun swiftc -swift-version 5 -parse-as-library -module-cache-path "$OUT/swift-cache" "${SOURCES[@]}" "$ROOT/tests/NativeEditorTests.swift" -o "$OUT/native-editor-tests"
"$OUT/native-editor-tests"
