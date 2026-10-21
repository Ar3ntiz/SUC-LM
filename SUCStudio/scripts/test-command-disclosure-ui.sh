#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-$(xcrun --show-sdk-path)}"
if [[ -z "${SDKROOT:-}" && -d /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk ]]; then SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk; fi
OUT="$ROOT/.build/command-disclosure-ui"
mkdir -p "$OUT/cache"
SOURCES=()
for file in "$ROOT"/Sources/*.swift; do [[ "$file" == */SUCStudioApp.swift ]] || SOURCES+=("$file"); done
xcrun swiftc -swift-version 5 -whole-module-optimization -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/cache" "${SOURCES[@]}" "$ROOT/tests/CommandDisclosureUITests.swift" -o "$OUT/disclosure-ui-tests"
"$OUT/disclosure-ui-tests"
