#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
source "$ROOT/scripts/engine-path.sh"
# SDK 27 on the development machine lacks the required SwiftUI macro plugin.
SDK="${SDKROOT:-$(xcrun --show-sdk-path)}"
if [[ -z "${SDKROOT:-}" && -d /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk ]]; then
  SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
fi
export SDKROOT="$SDK"
ARCH=arm64 "$ROOT/scripts/build-engine.sh"
APP="$ROOT/dist/SUC Studio.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Helpers" "$APP/Contents/Resources/docs"
for arch in arm64 x86_64; do
  OUT="$ROOT/.build/$arch"
  mkdir -p "$OUT/swift-cache-14602"
  printf 'Compiling SwiftUI app (%s)…\n' "$arch"
  xcrun swiftc -swift-version 5 -whole-module-optimization -O -target "$arch-apple-macosx15.0" -sdk "$SDK" -module-cache-path "$OUT/swift-cache-14602" -parse-as-library "$ROOT"/Sources/*.swift -o "$OUT/SUC Studio"
  for helper in command terminal; do
    xcrun clang++ -std=c++17 -O2 -arch "$arch" -isysroot "$SDK" -mmacosx-version-min=15.0 "$ROOT/Engine/studio-$helper.cpp" -o "$OUT/suc-studio-$helper"
  done
done
printf 'Assembling Universal bundle…\n'
lipo -create "$ROOT/.build/arm64/SUC Studio" "$ROOT/.build/x86_64/SUC Studio" -output "$APP/Contents/MacOS/SUC Studio"
for helper in command terminal; do
  lipo -create "$ROOT/.build/arm64/suc-studio-$helper" "$ROOT/.build/x86_64/suc-studio-$helper" -output "$APP/Contents/Helpers/suc-studio-$helper"
done
cp "$ROOT/.build/arm64/suc-studio-engine" "$APP/Contents/Helpers/"
cp "$ROOT/.build/arm64/ENGINE_BUILD_MANIFEST.json" "$APP/Contents/Resources/"
cp "$ROOT/.build/arm64/ENGINE_BUILD_MANIFEST.json" "$ROOT/dist/"
cp "$ROOT/Resources/Info.plist" "$APP/Contents/Info.plist"
"$PYTHON_BIN" "$ROOT/scripts/generate-help.py"
"$PYTHON_BIN" "$ROOT/scripts/generate-help.py" en
cp "$ROOT/Resources/USAGE.html" "$ROOT/Resources/USAGE-en.html" "$ROOT/USAGE-EN.md" "$ROOT/USAGE.md" "$ROOT/USAGE-IT.md" "$APP/Contents/Resources/"
cp "$ROOT/../cpp/README-IT.md" "$APP/Contents/Resources/docs/CPP_README-IT.md"
cp "$ROOT/docs/QA_REPORT.md" "$ROOT/docs/ARCHITECTURE.md" "$APP/Contents/Resources/docs/"
cp "$ROOT/../docs/optimization/INFERENCE_1.4.7.md" "$ROOT/../docs/optimization/REFERENCE_KL.md" "$ROOT/../docs/optimization/COMBINED_RESULTS.md" "$ROOT/../docs/optimization/FINAL_QUALIFICATION_1.4.7.md" "$ROOT/../docs/optimization/RELEASE_1.4.7.md" "$APP/Contents/Resources/docs/"
cp "$ROOT/../docs/optimization/LONG_CHAT_DIAGNOSTICS.md" "$ROOT/../docs/optimization/RELEASE_1.4.7.02.md" "$APP/Contents/Resources/docs/"
cp "$ROOT/../docs/optimization/CHAT_PREFIX_CHECKPOINTS.md" "$ROOT/../docs/optimization/RELEASE_1.4.7.03.md" "$APP/Contents/Resources/docs/"
cp "$ROOT/../docs/optimization/RELEASE_1.4.7.04.md" "$APP/Contents/Resources/docs/"
cp "$ROOT/../docs/optimization/RELEASE_1.4.7.05.md" "$ROOT/../docs/optimization/RELEASE_1.4.8.md" "$ROOT/../docs/optimization/RELEASE_1.4.8.02.md" "$ROOT/../docs/optimization/RELEASE_1.4.8.03.md" "$APP/Contents/Resources/docs/"
cp "$ROOT/../docs/optimization/RELEASE_1.4.9_IT.md" "$APP/Contents/Resources/docs/"
cp -R "$ROOT/Resources/en.lproj" "$ROOT/Resources/it.lproj" "$APP/Contents/Resources/"
cp "$ROOT/docs/CUSTOMIZATION_1.4.6.02_IT.md" "$APP/Contents/Resources/docs/"
mkdir -p "$APP/Contents/Resources/licenses"
cp "$ROOT/licenses/"* "$APP/Contents/Resources/licenses/"
mkdir -p "$APP/Contents/Resources/examples"
cp -R "$ROOT/examples/portable-review" "$APP/Contents/Resources/examples/"
cp "$ENGINE_ROOT/third_party/metal-cpp/LICENSE.txt" "$APP/Contents/Resources/Metal-cpp-LICENSE.txt"
if [[ -f "$ROOT/Resources/AppIcon.icns" ]]; then cp "$ROOT/Resources/AppIcon.icns" "$APP/Contents/Resources/"; fi
for helper in command terminal engine; do codesign --force --sign - "$APP/Contents/Helpers/suc-studio-$helper"; done
codesign --force --sign - "$APP"
"$ROOT/scripts/verify.sh"
printf 'Universal app built: %s\n' "$APP"
