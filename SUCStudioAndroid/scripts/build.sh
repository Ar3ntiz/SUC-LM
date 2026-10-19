#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
[[ -d "$SDK/platforms/android-34" ]] || { echo 'Imposta ANDROID_HOME su una SDK con platforms;android-34.' >&2; exit 1; }
printf 'sdk.dir=%s\n' "$SDK" > "$ROOT/local.properties"
(cd "$ROOT" && npm ci --no-fund && npm run build)
mkdir -p "$ROOT/app/src/main/assets/licenses"
cp "$ROOT/licenses/"* "$ROOT/app/src/main/assets/licenses/"
"$ROOT/gradlew" -p "$ROOT" :app:assembleDebug
