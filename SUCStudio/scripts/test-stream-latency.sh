#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-$(xcrun --show-sdk-path)}"
if [[ -z "${SDKROOT:-}" && -d /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk ]]; then SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk; fi
OUT="$ROOT/.build/diagnostics"
mkdir -p "$OUT/swift-cache"
xcrun swiftc -swift-version 5 -whole-module-optimization -O -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/swift-cache" "$ROOT/Sources/Models.swift" "$ROOT/Sources/EngineClient.swift" "$ROOT/Sources/Attachments.swift" "$ROOT/Sources/ImageAttachments.swift" "$ROOT/Sources/MCPClient.swift" "$ROOT/Sources/RemoteInference.swift" "$ROOT/Sources/RemoteTelemetry.swift" "$ROOT/Sources/ModelIOLog.swift" "$ROOT/tests/StreamLatencyProbe.swift" -o "$OUT/stream-latency-probe"
"$OUT/stream-latency-probe" "$ROOT/tests/stream_latency_fixture.py" "${1:-$ROOT/../docs/optimization/LONG_CHAT_CLIENT_1.4.7.02.json}"
