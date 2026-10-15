#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.build/$(uname -m)"
SOURCES=()
for file in "$ROOT"/Sources/*.swift; do
  [[ "$file" == */SUCStudioApp.swift ]] || SOURCES+=("$file")
done
xcrun swiftc -swift-version 5 -module-cache-path "$OUT/swift-cache" -parse-as-library "${SOURCES[@]}" "$ROOT/tests/RenderPreview.swift" -o "$OUT/render-preview"
"$OUT/render-preview" "$@"
