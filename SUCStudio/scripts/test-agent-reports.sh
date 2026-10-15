#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.build/$(uname -m)"
CORE=()
for file in "$ROOT"/Sources/*.swift; do
 case "$file" in */*Views.swift|*/Views.swift|*/SUCStudioApp.swift) ;; *) CORE+=("$file");; esac
done
xcrun swiftc -swift-version 5 -parse-as-library -module-cache-path "$OUT/swift-cache" "${CORE[@]}" "$ROOT/tests/AgentReportTests.swift" -o "$OUT/agent-report-tests"
"$OUT/agent-report-tests"
