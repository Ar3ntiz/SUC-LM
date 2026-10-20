#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-$(xcrun --show-sdk-path)}"
if [[ -z "${SDKROOT:-}" && -d /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk ]]; then SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk; fi
OUT="$ROOT/.build/context-tests"
mkdir -p "$OUT/swift-cache"
CORE=()
for file in "$ROOT"/Sources/*.swift; do
 case "$file" in */*Views.swift|*/Views.swift|*/SUCStudioApp.swift) ;; *) CORE+=("$file");; esac
done
xcrun swiftc -swift-version 5 -whole-module-optimization -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/swift-cache" "${CORE[@]}" "$ROOT/tests/ContextTests.swift" -o "$OUT/context-tests"
"$OUT/context-tests" "$ROOT/tests/context_fixture.py"

xcrun swiftc -swift-version 5 -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/swift-cache" "$ROOT/Sources/Models.swift" "$ROOT/Sources/PluginCatalog.swift" "$ROOT/tests/PluginPolicyTests.swift" -o "$OUT/plugin-policy-tests"
"$OUT/plugin-policy-tests"
