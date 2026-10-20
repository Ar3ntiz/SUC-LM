#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-$(xcrun --show-sdk-path)}"
if [[ -z "${SDKROOT:-}" && -d /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk ]]; then SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk; fi
"${PYTHON_BIN:-/usr/bin/python3}" "$ROOT/tests/release_identity_test.py"
OUT="$ROOT/.build/universal-tests"
mkdir -p "$OUT/swift-cache"
CORE=()
for file in "$ROOT"/Sources/*.swift; do
  case "$file" in */*Views.swift|*/Views.swift|*/SUCStudioApp.swift) ;; *) CORE+=("$file");; esac
done
for mode in native remote-only; do
  FLAGS=(-D SUC_UNIVERSAL_TESTS)
  if [[ "$mode" == remote-only ]]; then FLAGS+=(-D SUC_REMOTE_ONLY); fi
  xcrun swiftc -swift-version 5 -whole-module-optimization -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/swift-cache" "${FLAGS[@]}" "${CORE[@]}" "$ROOT/tests/UniversalPlatformTests.swift" -o "$OUT/platform-$mode"
  "$OUT/platform-$mode"
done
xcrun swiftc -swift-version 5 -whole-module-optimization -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/swift-cache" "$ROOT/Sources/Models.swift" "$ROOT/Sources/EngineClient.swift" "$ROOT/Sources/Attachments.swift" "$ROOT/Sources/ImageAttachments.swift" "$ROOT/Sources/MCPClient.swift" "$ROOT/Sources/RemoteInference.swift" "$ROOT/Sources/RemoteTelemetry.swift" "$ROOT/tests/RemoteTests.swift" -o "$OUT/remote-tests"
"$OUT/remote-tests" "$ROOT/tests/remote_fixture.py"

"$ROOT/scripts/test-release146.sh"

"$ROOT/scripts/test-customization14602.sh"
"$ROOT/scripts/test-mcp-http14602.sh"

"$ROOT/scripts/test-mcp-schemas.sh"

"$ROOT/scripts/test-chat-queue.sh"

"$ROOT/scripts/test-context.sh"
