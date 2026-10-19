#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk}"
OUT="$ROOT/.build/universal-tests"
mkdir -p "$OUT/swift-cache"
FILES=()
for name in Models EngineClient Attachments ImageAttachments MCPClient RemoteInference RemoteTelemetry;do FILES+=("$ROOT/Sources/$name.swift");done
xcrun swiftc -swift-version 5 -whole-module-optimization -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/swift-cache" "${FILES[@]}" "$ROOT/tests/MCPHTTP14602Tests.swift" -o "$OUT/mcp-http14602-tests"
"$OUT/mcp-http14602-tests" "$ROOT/tests/mcp_fixture.py"
