#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/SUC Studio.app"
plutil -lint "$APP/Contents/Info.plist"
codesign --verify --deep --strict "$APP"
[[ -n "$(/usr/libexec/PlistBuddy -c 'Print NSAppleEventsUsageDescription' "$APP/Contents/Info.plist")" ]]
ENTITLEMENTS="$(codesign -d --entitlements :- "$APP" 2>/dev/null)"
[[ "$(printf '%s' "$ENTITLEMENTS" | /usr/libexec/PlistBuddy -c 'Print com.apple.security.automation.apple-events' /dev/stdin)" == true ]]
for bin in "$APP/Contents/MacOS/SUC Studio" "$APP/Contents/Helpers/suc-studio-command" "$APP/Contents/Helpers/suc-studio-terminal"; do
  # Some Apple lipo versions reject multiple verify_arch arguments.
  for arch in arm64 x86_64; do lipo "$bin" -verify_arch "$arch"; done
done
lipo "$APP/Contents/Helpers/suc-studio-engine" -verify_arch arm64
for bin in "$APP/Contents/MacOS/SUC Studio" "$APP/Contents/Helpers/"*; do
  if otool -L "$bin" | awk '/^[[:space:]]/ {print $1}' | grep -Ev '^(/usr/lib/|/System/Library/|@rpath/libswift)' ; then
    echo "Unexpected external dependency" >&2; exit 1
  fi
done
[[ "$(/usr/libexec/PlistBuddy -c 'Print CFBundleIdentifier' "$APP/Contents/Info.plist")" == local.sucstudio.universal.app ]]
[[ -s "$APP/Contents/Resources/USAGE.md" && -s "$APP/Contents/Resources/USAGE.html" ]]
echo 'Universal bundle verification passed (app/helpers: arm64+x86_64; engine: arm64).'
