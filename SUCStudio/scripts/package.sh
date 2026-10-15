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
[[ -s "$ROOT/Resources/USAGE.html" ]] || { echo "La guida HTML non è stata generata." >&2; exit 1; }
cp "$ROOT/Resources/USAGE.html" "$ROOT/USAGE.md" "$APP/Contents/Resources/"
cp "$ROOT/docs/QA_REPORT.md" "$ROOT/docs/ARCHITECTURE.md" "$APP/Contents/Resources/docs/"
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
<key>BundleHasStrictIdentifier</key><true/>
<key>BundleOverwriteAction</key><string>upgrade</string>
</dict></array></plist>
PLIST
pkgbuild --root "$STAGE/root" --component-plist "$STAGE/components.plist" --install-location / --identifier local.sucstudio.app --version "$VERSION" "$ROOT/dist/SUC-Studio-${VERSION}-arm64.pkg"
(cd "$ROOT/dist" && shasum -a 256 SUC-Studio-${VERSION}-arm64.pkg > SHA256SUMS.txt)
