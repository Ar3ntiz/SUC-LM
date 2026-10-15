#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/SUC Studio.app"
plutil -lint "$APP/Contents/Info.plist"
codesign --verify --deep --strict "$APP"
for bin in "$APP/Contents/MacOS/SUC Studio" "$APP/Contents/Helpers/suc-studio-engine" "$APP/Contents/Helpers/suc-studio-command" "$APP/Contents/Helpers/suc-studio-terminal"; do
  lipo "$bin" -verify_arch arm64
  if otool -L "$bin" | tail -n +2 | awk '{print $1}' | grep -Ev '^(/usr/lib/|/System/Library/|@rpath/libswift)' ; then
    echo "Unexpected external dependency" >&2; exit 1
  fi
done
[[ -f "$APP/Contents/Resources/USAGE.md" ]]
[[ -f "$APP/Contents/Resources/USAGE.html" ]]
echo 'Bundle verification passed.'
