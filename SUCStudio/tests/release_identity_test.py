#!/usr/bin/env python3
"""Release identity and installer migration guards, without touching installed apps."""
from pathlib import Path
import plistlib
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
info = plistlib.loads((root / 'Resources/Info.plist').read_bytes())
assert info['CFBundleDisplayName'] == info['CFBundleName'] == info['CFBundleExecutable'] == 'SUC Studio'
assert info['CFBundleShortVersionString'] == '1.4.9.05'
assert info['CFBundleVersion'] == '46'
assert info.get('NSAppleEventsUsageDescription', '').strip()
entitlements = plistlib.loads((root / 'Resources/SUCStudio.entitlements').read_bytes())
assert entitlements['com.apple.security.automation.apple-events'] is True
for language in ('en', 'it'):
    assert 'NSAppleEventsUsageDescription' in (root / f'Resources/{language}.lproj/InfoPlist.strings').read_text()
for script in ('build.sh', 'package.sh'):
    assert '--entitlements "$ROOT/Resources/SUCStudio.entitlements" "$APP"' in (root / 'scripts' / script).read_text()
identity = 'local.sucstudio.universal.app'
assert info['CFBundleIdentifier'] == identity
with tempfile.TemporaryDirectory() as temporary:
    volume = Path(temporary)
    apps = volume / 'Applications'
    apps.mkdir()
    current, preview = apps / 'SUC Studio.app', apps / 'SUC Studio Universal Preview.app'
    def bundle(path, identifier):
        (path / 'Contents').mkdir(parents=True, exist_ok=True)
        (path / 'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier': identifier}))
    def run(script):
        return subprocess.run([str(root / 'scripts/installer' / script), '', '', str(volume)], capture_output=True).returncode
    assert run('preinstall') == 0
    bundle(current, 'local.sucstudio.app')
    assert run('preinstall') == 0
    bundle(current, 'unrelated.app')
    assert run('preinstall') != 0
    bundle(current, identity)
    assert run('preinstall') == 0
    bundle(preview, 'unrelated.app')
    assert run('postinstall') == 0 and preview.exists()
    bundle(preview, identity)
    assert run('postinstall') == 0 and not preview.exists() and current.exists()
    outside = volume / 'do-not-remove'
    bundle(outside, identity)
    preview.symlink_to(outside, target_is_directory=True)
    assert run('postinstall') == 0 and preview.is_symlink() and outside.exists()
    preview.unlink()
    current.rename(volume / 'saved-current')
    bundle(preview, identity)
    assert run('postinstall') == 0 and preview.exists()
    current.symlink_to(outside, target_is_directory=True)
    assert run('preinstall') != 0
print('Release identity and installer migration guards: PASS')
