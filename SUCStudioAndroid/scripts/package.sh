#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
(cd "$ROOT" && npm run build)
if [[ -n "${SUC_ANDROID_KEYSTORE:-}" ]]; then
  : "${SUC_ANDROID_STORE_PASSWORD:?Set store password outside the repository}"
  : "${SUC_ANDROID_KEY_ALIAS:?Set key alias}"
  : "${SUC_ANDROID_KEY_PASSWORD:?Set key password}"
  "$ROOT/gradlew" -p "$ROOT" :app:assembleRelease
  INPUT="$ROOT/app/build/outputs/apk/release/app-release.apk"
else
  "$ROOT/gradlew" -p "$ROOT" :app:assembleDebug
  INPUT="$ROOT/app/build/outputs/apk/debug/app-debug.apk"
fi
VERSION=$(sed -n "s/.*versionName '\([^']*\)'.*/\1/p" "$ROOT/app/build.gradle")
[[ -n "$VERSION" ]] || { echo 'Versione Android non trovata.' >&2; exit 1; }
mkdir -p "$ROOT/dist"
cp "$INPUT" "$ROOT/dist/SUC-Studio-Android-${VERSION}.apk"
(cd "$ROOT/dist" && shasum -a 256 SUC-Studio-Android-${VERSION}.apk > SHA256SUMS.txt)
printf 'APK: %s\n' "$ROOT/dist/SUC-Studio-Android-${VERSION}.apk"
