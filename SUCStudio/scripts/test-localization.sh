#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-$(xcrun --show-sdk-path)}"
OUT="$ROOT/.build/localization-tests"
mkdir -p "$OUT/cache"
xcrun swiftc -swift-version 5 -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/cache" "$ROOT/Sources/Models.swift" "$ROOT/Sources/Localization.swift" "$ROOT/Sources/LocalizationCatalog.swift" "$ROOT/tests/LocalizationTests.swift" -o "$OUT/localization-tests"
"$OUT/localization-tests"
