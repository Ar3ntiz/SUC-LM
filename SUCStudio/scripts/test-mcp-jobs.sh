#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk}"
OUT="$ROOT/.build/mcp-jobs-tests"
mkdir -p "$OUT/cache"
CORE=()
for file in "$ROOT"/Sources/*.swift; do
 case "$file" in */*Views.swift|*/Views.swift|*/SUCStudioApp.swift) ;; *) CORE+=("$file");; esac
done
xcrun swiftc -swift-version 5 -whole-module-optimization -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/cache" "${CORE[@]}" "$ROOT/tests/MCPJobsTests.swift" -o "$OUT/mcp-jobs-tests"
"$OUT/mcp-jobs-tests" "$ROOT/tests/mcp_jobs_fixture.py"
