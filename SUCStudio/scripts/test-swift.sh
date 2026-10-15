#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.build/$(uname -m)"
mkdir -p "$OUT/swift-cache"
CORE=()
for file in "$ROOT"/Sources/*.swift; do
  case "$file" in */*Views.swift|*/Views.swift|*/SUCStudioApp.swift) ;; *) CORE+=("$file");; esac
done
xcrun swiftc -swift-version 5 -parse-as-library -module-cache-path "$OUT/swift-cache" "$ROOT/Sources/Models.swift" "$ROOT/tests/ArchiveTests.swift" -o "$OUT/archive-tests"
"$OUT/archive-tests"
xcrun swiftc -swift-version 5 -parse-as-library -module-cache-path "$OUT/swift-cache" "${CORE[@]}" "$ROOT/tests/SessionRoutingTests.swift" -o "$OUT/session-tests"
"$OUT/session-tests"

xcrun swiftc -swift-version 5 -parse-as-library -module-cache-path "$OUT/swift-cache" "$ROOT/Sources/Models.swift" "$ROOT/Sources/Attachments.swift" "$ROOT/Sources/ImageAttachments.swift" "$ROOT/tests/AttachmentTests.swift" -o "$OUT/attachment-tests"
"$OUT/attachment-tests"

xcrun swiftc -swift-version 5 -parse-as-library -module-cache-path "$OUT/swift-cache" "${CORE[@]}" "$ROOT/tests/CustomizationTests.swift" -o "$OUT/customization-tests"
"$OUT/customization-tests" "$ROOT/tests/mcp_fixture.py"

xcrun swiftc -swift-version 5 -parse-as-library -module-cache-path "$OUT/swift-cache" "$ROOT/Sources/Models.swift" "$ROOT/Sources/EngineClient.swift" "$ROOT/Sources/Attachments.swift" "$ROOT/Sources/ImageAttachments.swift" "$ROOT/Sources/MCPClient.swift" "$ROOT/Sources/RemoteInference.swift" "$ROOT/Sources/RemoteTelemetry.swift" "$ROOT/tests/RemoteTests.swift" -o "$OUT/remote-tests"
"$OUT/remote-tests" "$ROOT/tests/remote_fixture.py"

"$ROOT/scripts/test-workspace.sh"
xcrun swiftc -swift-version 5 -parse-as-library -module-cache-path "$OUT/swift-cache" "${CORE[@]}" "$ROOT/tests/PluginRoutineTests.swift" -o "$OUT/plugin-routine-tests"
"$OUT/plugin-routine-tests"
"$ROOT/scripts/test-preferences.sh"

"$ROOT/scripts/test-markdown.sh"

"$ROOT/scripts/test-images.sh"

"$ROOT/scripts/test-telemetry.sh"

"$ROOT/scripts/test-agent-reports.sh"

"$ROOT/scripts/test-scroll-journal.sh"

"$ROOT/scripts/test-terminal-artifacts.sh"

"$ROOT/scripts/test-presentation.sh"
