#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-$(xcrun --show-sdk-path)}"
OUT="$ROOT/.build/agent-continuation-tests"
mkdir -p "$OUT/cache"
CORE=()
for file in "$ROOT"/Sources/*.swift; do
 case "$file" in */*Views.swift|*/Views.swift|*/SUCStudioApp.swift) ;; *) CORE+=("$file");; esac
done
xcrun swiftc -swift-version 5 -whole-module-optimization -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/cache" "${CORE[@]}" "$ROOT/tests/AgentContinuationTests.swift" -o "$OUT/agent-continuation-tests"
"$OUT/agent-continuation-tests" "$ROOT/tests/agent_continuation_fixture.py"
