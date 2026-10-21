#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
[[ -x "$PYTHON_BIN" ]] || { echo "Python non disponibile: $PYTHON_BIN" >&2; exit 1; }
APP="$ROOT/dist/SUC Studio.app"
[[ -d "$APP" ]] || "$ROOT/scripts/build.sh"
VERSION=$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$ROOT/Resources/Info.plist")
BUNDLED_VERSION=$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$APP/Contents/Info.plist")
[[ "$VERSION" == "$BUNDLED_VERSION" ]] || { echo 'Rebuild the app before packaging a new version' >&2; exit 1; }
"$PYTHON_BIN" "$ROOT/scripts/generate-help.py"
"$PYTHON_BIN" "$ROOT/scripts/generate-help.py" en
[[ -s "$ROOT/Resources/USAGE.html" ]] || { echo "La guida HTML non è stata generata." >&2; exit 1; }
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
codesign --force --sign - "$APP"
"$ROOT/scripts/verify.sh"
STAGE="$(mktemp -d "$ROOT/.build/package.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/root/Applications"
ditto "$APP" "$STAGE/root/Applications/SUC Studio.app"
cat > "$STAGE/components.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><array><dict>
<key>RootRelativeBundlePath</key><string>Applications/SUC Studio.app</string>
<key>BundleIsRelocatable</key><false/>
<key>BundleIsVersionChecked</key><true/>
<!-- preinstall allows only the historical or retained SUC Studio identifier. -->
<key>BundleHasStrictIdentifier</key><false/>
<key>BundleOverwriteAction</key><string>upgrade</string>
</dict></array></plist>
PLIST
pkgbuild --scripts "$ROOT/scripts/installer" --root "$STAGE/root" --component-plist "$STAGE/components.plist" --install-location / --identifier local.sucstudio.universal.app --version "$VERSION" "$ROOT/dist/SUC-Studio-${VERSION}-universal.pkg"
(cd "$ROOT/dist" && shasum -a 256 SUC-Studio-${VERSION}-universal.pkg > SHA256SUMS.txt)
cp "$ROOT/dist/SHA256SUMS.txt" "$ROOT/dist/SHA256SUMS-${VERSION}.txt"
