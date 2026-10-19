#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
(cd "$ROOT" && npm run build && npm test)
"$ROOT/gradlew" -p "$ROOT" :app:testDebugUnitTest
