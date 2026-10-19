#!/usr/bin/env python3
"""Package standalone CLI/server binaries from a verified isolated macOS build."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', required=True, type=Path)
    ap.add_argument('--version', required=True)
    ap.add_argument('--output', type=Path, default=Path('SUCStudio/dist'))
    a = ap.parse_args(); build = a.build.expanduser(); a.output.mkdir(parents=True, exist_ok=True)
    commit = (build/'GIT_COMMIT').read_text().strip()
    repo = Path(__file__).resolve().parents[1]
    # Documentation-only commits may differ; the actual C++/Metal inputs must not.
    assert not subprocess.check_output(['git','-C',str(repo),'diff',commit,'HEAD','--','cpp',':(exclude)cpp/README.md',':(exclude)cpp/README-IT.md'])
    app_manifest = json.loads((a.output/'ENGINE_BUILD_MANIFEST.json').read_text())
    current_manifest = json.loads(subprocess.check_output(
        ['python3',str(repo/'SUCStudio/scripts/engine-manifest.py'),str(repo/'cpp')],text=True))
    assert current_manifest['content_sha256'] == app_manifest['content_sha256']
    names = ['suc1-generate', 'suc1-server', 'suc1-inspect', 'suc-tokenize']
    minimum = {}
    for name in names:
        info = subprocess.check_output(['vtool', '-show-build', str(build/name)], text=True)
        found = re.search(r'\bminos\s+([0-9.]+)', info)
        assert found, f'Missing deployment target: {name}'
        minimum[name] = found.group(1)
    required = max(minimum.values(), key=lambda x: tuple(map(int, x.split('.'))))
    manifest = {'version': a.version, 'binary_commit': commit, 'architecture': 'arm64',
                'minimum_system': f'macOS {required}+ (Apple Silicon, Metal)',
                'minimum_macos_by_binary': minimum, 'files': {},
                'engine_input_sha256': app_manifest['content_sha256'],
                'app_repository_commit': app_manifest['repository_commit']}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)/f'SUC-CLI-Server-{a.version}-arm64'; root.mkdir()
        for name in names:
            source = build/name
            subprocess.run(['lipo', str(source), '-verify_arch', 'arm64'], check=True)
            shutil.copy2(source, root/name)
            manifest['files'][name] = hashlib.sha256(source.read_bytes()).hexdigest()
        (root/'BUILD.json').write_text(json.dumps(manifest, indent=2)+'\n')
        usage = '''# SUC CLI and server

Apple Silicon / Metal binaries. The Universal macOS app is distributed separately;
Intel uses its remote-client mode. No model weights or tokenizer are bundled.
These particular standalone binaries require macOS @MINIMUM@ or later, as
recorded from their Mach-O deployment targets in BUILD.json. The separately
packaged Universal app targets macOS 15 or later; do not confuse these targets.

From this directory (replace both absolute model/tokenizer paths):

```sh
./suc1-inspect /path/model.suc1
./suc1-generate /path/model.suc1 --tokenizer /path/tokenizer.json --prompt "Hello" --steps 128
./suc1-server --help
./suc1-server /path/model.suc1 --tokenizer /path/tokenizer.json
```

The standalone server defaults to loopback and has no API-key authentication;
the app's embedded server supplies the API-key layer. Use the app server for
authenticated LAN access. Model files and tokenizers must match; existing SUC1
files are reusable.

DeepSeek/GLM automatically use qualified FP32 TCQ reuse; SUC_TCQ_REUSE=0 selects
the historical path. Other model families retain their previous kernel default.
Qwen hybrid graphs retain one bounded recurrent/convolution checkpoint (128 MiB
maximum CPU state copies); KV/indexer prefix rows remain in place. Matching chat
turns restore that prefix without changing the existing SUC template. First requests
and changes before the checkpoint still run full prefill.
SUC_PREFIX_CHECKPOINTS=0 retains strict live-prefix reuse only.
SUC_PREFIX_CACHE=0 disables both prefix policies; SUC_SAMPLING_LEGACY=1 selects
historical sampling order. Sampling CDF order can change seeded text, not its
distribution. N-gram parallelism, query tiling, half TCQ and prompt lookup are
not release defaults. MTP is not implemented. Memory policy is unchanged.

Keep the CLI automatic Qwen prefill profile and the app/server conservative
profile distinct when comparing measurements. Explicit --chunk is honored.
BUILD.json records the actual compiled revision and binary hashes.

This local distribution is not Developer-ID signed or notarized.
No installer should be run automatically during validation.
'''
        (root/'USAGE.md').write_text(usage.replace('@MINIMUM@', required))
        if (repo/'LICENSE').exists(): shutil.copy2(repo/'LICENSE', root/'LICENSE')
        (root/'docs').mkdir()
        for name in ['INFERENCE_1.4.7.md','REFERENCE_KL.md','COMBINED_RESULTS.md','FINAL_QUALIFICATION_1.4.7.md','RELEASE_1.4.7.md','CHAT_PREFIX_CHECKPOINTS.md','RELEASE_1.4.7.03.md']:
            shutil.copy2(repo/'docs/optimization'/name, root/'docs'/name)
        shutil.copy2(repo/'cpp/README-IT.md', root/'docs/CPP_README-IT.md')
        archive = a.output/(root.name+'.tar.gz')
        with tarfile.open(archive, 'w:gz') as tf: tf.add(root, arcname=root.name)
    (a.output/'CLI_SERVER_BUILD.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(archive)

if __name__ == '__main__': main()
