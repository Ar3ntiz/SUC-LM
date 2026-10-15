#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
[[ -x "$PYTHON_BIN" ]] || { echo "Python non disponibile: $PYTHON_BIN" >&2; exit 1; }
source "$ROOT/scripts/engine-path.sh"
ARCH="${ARCH:-arm64}"
export ARCH
"$ROOT/scripts/build-engine.sh"
OUT="$ROOT/.build/$ARCH"
APP="$ROOT/dist/SUC Studio.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Helpers" "$APP/Contents/Resources" "$OUT/swift-cache"
SDK="$(xcrun --show-sdk-path)"
xcrun swiftc -swift-version 5 -O -target "$ARCH-apple-macosx15.0" -sdk "$SDK" -module-cache-path "$OUT/swift-cache" -parse-as-library "$ROOT"/Sources/*.swift -o "$APP/Contents/MacOS/SUC Studio"
xcrun clang++ -std=c++17 -O2 -arch "$ARCH" -mmacosx-version-min=15.0 "$ROOT/Engine/studio-command.cpp" -o "$APP/Contents/Helpers/suc-studio-command"
codesign --force --sign - "$APP/Contents/Helpers/suc-studio-command"
xcrun clang++ -std=c++17 -O2 -arch "$ARCH" -mmacosx-version-min=15.0 "$ROOT/Engine/studio-terminal.cpp" -o "$APP/Contents/Helpers/suc-studio-terminal"
codesign --force --sign - "$APP/Contents/Helpers/suc-studio-terminal"
cp "$OUT/ENGINE_BUILD_MANIFEST.json" "$APP/Contents/Resources/"
cp "$OUT/ENGINE_BUILD_MANIFEST.json" "$ROOT/dist/"
cp "$OUT/suc-studio-engine" "$APP/Contents/Helpers/"
cp "$ROOT/Resources/Info.plist" "$APP/Contents/Info.plist"
"$PYTHON_BIN" "$ROOT/scripts/generate-help.py"
[[ -s "$ROOT/Resources/USAGE.html" ]] || { echo "La guida HTML non è stata generata." >&2; exit 1; }
cp "$ROOT/Resources/USAGE.html" "$APP/Contents/Resources/"
cp "$ROOT/USAGE.md" "$APP/Contents/Resources/"
mkdir -p "$APP/Contents/Resources/docs"
for document in QA_REPORT.md ARCHITECTURE.md; do
  if [[ -f "$ROOT/docs/$document" ]]; then cp "$ROOT/docs/$document" "$APP/Contents/Resources/docs/"; fi
done
cp "$ENGINE_ROOT/third_party/metal-cpp/LICENSE.txt" "$APP/Contents/Resources/Metal-cpp-LICENSE.txt"
if [[ -f "$ROOT/Resources/AppIcon.icns" ]]; then cp "$ROOT/Resources/AppIcon.icns" "$APP/Contents/Resources/"; fi
codesign --force --sign - "$APP/Contents/Helpers/suc-studio-engine"
codesign --force --sign - "$APP"
codesign --verify --deep --strict "$APP"
printf 'App built: %s\n' "$APP"
