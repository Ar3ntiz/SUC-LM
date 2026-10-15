"""Exercise real build scripts with a fake compiler; real compilation runs separately."""
import json, os, pathlib, shutil, subprocess, tempfile
root = pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='suc-parent-build-') as temp:
    base = pathlib.Path(temp); app = base/'SUCStudio'; engine = base/'cpp'; tools = base/'bin'
    (app/'scripts').mkdir(parents=True); (app/'Engine').mkdir(); tools.mkdir()
    for name in ('build-engine.sh', 'engine-path.sh', 'engine-manifest.py'):
        shutil.copy2(root/'scripts'/name, app/'scripts'/name)
    def run(ok=True):
        p = subprocess.run(['/bin/bash', str(app/'scripts/build-engine.sh')], env=env, capture_output=True, text=True)
        assert (p.returncode == 0) == ok, p.stdout + p.stderr
        return p
    env = dict(os.environ, PATH=str(tools)+':'+os.environ['PATH'], ARCH='arm64', FAKE_BUILD_ROOT=str(base))
    # Missing parent fails even if someone reintroduces a local runtime snapshot.
    (app/'runtime').mkdir()
    assert 'richiede il motore del progetto padre' in run(False).stderr
    for directory in ('include/suc','src/metal','tools','tests','metal','third_party/metal-cpp'):
        (engine/directory).mkdir(parents=True)
    for name in ('tools/tool_common.hpp','tests/test_main.cpp','src/first.cpp','src/removed.cpp','src/metal/device.cpp'):
        (engine/name).write_text('// fixture\n')
    for name in ('suc_kernels','suc_model_kernels','suc_qwen4_kernels','suc_ds4_kernels','suc_glm5_kernels'):
        (engine/'metal'/f'{name}.metal').write_text('// metal fixture\n')
    (tools/'xcrun').write_text('''#!/bin/sh
if [ "$1" = "--find" ]; then echo "$FAKE_BUILD_ROOT/bin/fake-clang"; else echo "$FAKE_BUILD_ROOT/sdk"; fi
''')
    (tools/'fake-clang').write_text('''#!/usr/bin/python3
import os,sys,pathlib
if '--version' in sys.argv: print('test compiler');sys.exit(0)
with open(os.environ['FAKE_BUILD_ROOT']+'/calls','a') as log:log.write(' '.join(sys.argv[1:])+'\\n')
pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_text('compiled fixture')
''')
    for name in ('xcrun','fake-clang'): (tools/name).chmod(0o755)
    def compiles(): return sum(' -c ' in ' '+line+' ' for line in (base/'calls').read_text().splitlines())
    run(); first = compiles(); assert first > 0
    manifest = app/'.build/arm64/ENGINE_BUILD_MANIFEST.json'
    before = json.loads(manifest.read_text())['content_sha256']
    run(); assert compiles() == first, 'unchanged inputs should reuse objects'
    changed = engine/'src/first.cpp'; changed.write_text('// changed parent source\n'); os.utime(changed, (1,1))
    run(); assert compiles() > first, 'older timestamp must not hide content change'
    assert json.loads(manifest.read_text())['content_sha256'] != before
    stale = app/'.build/arm64/obj/src_removed.o'; assert stale.exists()
    (engine/'src/removed.cpp').unlink(); run(); assert not stale.exists(), 'removed parent source must not leave a linked object'
    print('BuildDependencyTests: missing parent/no fallback, cache reuse, older-mtime edit and removed-source cleanup passed')
