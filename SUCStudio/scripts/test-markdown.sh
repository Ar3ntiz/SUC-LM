#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.build/$(uname -m)"
xcrun swiftc -swift-version 5 -parse-as-library -module-cache-path "$OUT/swift-cache" "$ROOT/Sources/Markdown.swift" "$ROOT/tests/MarkdownTests.swift" -o "$OUT/markdown-tests"
"$OUT/markdown-tests"
