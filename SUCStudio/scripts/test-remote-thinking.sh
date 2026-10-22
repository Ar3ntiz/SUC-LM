#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-$(xcrun --show-sdk-path)}"
if [[ -z "${SDKROOT:-}" && -d /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk ]]; then SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk; fi
OUT="$ROOT/.build/remote-thinking-tests"
mkdir -p "$OUT/cache"
xcrun swiftc -swift-version 5 -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/cache" "$ROOT/Sources/Models.swift" "$ROOT/Sources/RemoteTelemetry.swift" "$ROOT/tests/TelemetryTests.swift" -o "$OUT/telemetry-tests"
"$OUT/telemetry-tests"
xcrun swiftc -swift-version 5 -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/cache" "$ROOT/Sources/Models.swift" "$ROOT/Sources/EngineClient.swift" "$ROOT/Sources/Attachments.swift" "$ROOT/Sources/ImageAttachments.swift" "$ROOT/Sources/MCPClient.swift" "$ROOT/Sources/RemoteInference.swift" "$ROOT/Sources/RemoteTelemetry.swift" "$ROOT/tests/RemoteTests.swift" -o "$OUT/remote-tests"
"$OUT/remote-tests" "$ROOT/tests/remote_fixture.py"
