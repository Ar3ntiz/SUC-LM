"""Render the actual menu-bar SF Symbol with AppKit, then package a macOS ICNS."""
from pathlib import Path
import struct, subprocess, tempfile
root = Path(__file__).resolve().parents[1]
iconset = root / 'Resources/AppIcon.iconset'
with tempfile.TemporaryDirectory(prefix='suc-icon-') as temp:
    binary = Path(temp) / 'render-icon'
    subprocess.run(['xcrun', 'swiftc', '-swift-version', '5', '-module-cache-path', str(root/'.build/icon-swift-cache'), str(root/'scripts/make-icon.swift'), '-o', str(binary)], check=True)
    subprocess.run([str(binary), str(iconset)], check=True)
entries = []
for tag, filename in [
    (b'icp4','icon_16x16.png'), (b'icp5','icon_32x32.png'), (b'icp6','icon_32x32@2x.png'),
    (b'ic07','icon_128x128.png'), (b'ic08','icon_256x256.png'), (b'ic09','icon_512x512.png'),
    (b'ic10','icon_512x512@2x.png')]:
    data = (iconset/filename).read_bytes()
    entries.append(tag + struct.pack('>I', len(data)+8) + data)
body = b''.join(entries)
(root/'Resources/AppIcon.icns').write_bytes(b'icns' + struct.pack('>I', len(body)+8) + body)
