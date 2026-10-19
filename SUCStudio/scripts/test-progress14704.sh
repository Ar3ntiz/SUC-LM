#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${SDKROOT:-$(xcrun --show-sdk-path)}"
if [[ -z "${SDKROOT:-}" && -d /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk ]]; then SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk; fi
OUT="$ROOT/.build/progress-tests"
mkdir -p "$OUT/swift-cache"
xcrun clang++ -std=c++20 -isysroot "$SDK" -isystem "$SDK/usr/include/c++/v1" -I"$ROOT/../cpp/include" "$ROOT/../cpp/tests/test_prefill_progress.cpp" -o "$OUT/progress-cpu"
"$OUT/progress-cpu"
xcrun swiftc -swift-version 5 -whole-module-optimization -sdk "$SDK" -parse-as-library -module-cache-path "$OUT/swift-cache" "$ROOT/Sources/Models.swift" "$ROOT/Sources/EngineClient.swift" "$ROOT/Sources/Attachments.swift" "$ROOT/Sources/ImageAttachments.swift" "$ROOT/Sources/MCPClient.swift" "$ROOT/Sources/RemoteInference.swift" "$ROOT/Sources/RemoteTelemetry.swift" "$ROOT/tests/RemoteTests.swift" -o "$OUT/remote-tests"
"$OUT/remote-tests" "$ROOT/tests/remote_fixture.py"
echo 'Progress observer and actual remote SSE regressions: PASS'
