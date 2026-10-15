#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.build/$(uname -m)"
xcrun swiftc -swift-version 5 -parse-as-library -module-cache-path "$OUT/swift-cache" "$ROOT/Sources/Models.swift" "$ROOT/Sources/Markdown.swift" "$ROOT/Sources/MessagePresentation.swift" "$ROOT/Sources/RemoteTelemetry.swift" "$ROOT/tests/PresentationTests.swift" -o "$OUT/presentation-tests"
"$OUT/presentation-tests"
