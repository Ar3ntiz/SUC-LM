#!/bin/bash
# Sequential final artifact build + real graph checks. Does not touch installed apps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${SUC_CHECKPOINT_BUILD:-$HOME/suc-local/cpp-chat-checkpoint}"
PY="${SUC_TEST_PYTHON:-$HOME/Projects/SUC-LM/.venv/bin/python}"
WIDE="${1:?usage: qualify_checkpoint_finish.sh PATH_TO_WIDE_FIXTURE}"
cd "$ROOT"
[[ -f "$WIDE" ]]
"$PY" -c 'import json;assert json.load(open("docs/optimization/CHAT_PREFIX_REAL_QWEN.json"))["passed"]'
bash scripts/studio_wait_idle.sh --check
mkdir -p logs
/usr/bin/python3 - "$BUILD" <<'PY'
import hashlib,json,sys
from pathlib import Path
build=Path(sys.argv[1]);files=[build/'suc1-server',build/'suc1-generate',Path('SUCStudio/.build/arm64/suc-studio-engine')]
Path('logs/chat-checkpoint-before-rebuild.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
PY
bash cpp/build.sh "$BUILD" > logs/chat-checkpoint-final-cpp-build.log 2>&1
git rev-parse HEAD > "$BUILD/GIT_COMMIT"
bash SUCStudio/scripts/build-engine.sh > logs/chat-checkpoint-final-helper-build.log 2>&1
c++ -std=c++20 -O2 -ffp-contract=off -Icpp/include -Icpp/tools -isystem cpp/third_party/metal-cpp SUCStudio/tests/ChatPromptProbe.cpp "$BUILD/libsuc_metal.a" "$BUILD/libsuc.a" -framework Metal -framework Foundation -framework QuartzCore -o SUCStudio/.build/chat-prompt-probe
APP="$ROOT/SUCStudio/dist/SUC Studio.app"
cp SUCStudio/.build/arm64/suc-studio-engine "$APP/Contents/Helpers/"
cp SUCStudio/.build/arm64/ENGINE_BUILD_MANIFEST.json "$APP/Contents/Resources/"
cp SUCStudio/.build/arm64/ENGINE_BUILD_MANIFEST.json SUCStudio/dist/
codesign --force --sign - "$APP/Contents/Helpers/suc-studio-engine"
codesign --force --sign - "$APP"
"$PY" scripts/qualify_qwen_chat_helper.py --helper "$APP/Contents/Helpers/suc-studio-engine" \
 --model "$HOME/Projects/SUC-LM/experiments/artifacts/compress/compress_qwen3.8_flash_next_side8/container/qwen3.8_flash_next_side8/qwen3.8_flash_next_side8.suc1" \
 --tokenizer "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer.json" \
 --prompt-probe "$ROOT/SUCStudio/.build/chat-prompt-probe" --output docs/optimization/CHAT_PREFIX_APP_QWEN.json > logs/chat-checkpoint-app-qwen.log 2>&1
SUC_CHECKPOINT_TEST_CHUNK=63 "$BUILD/suc-qwen-checkpoint-tests" "$WIDE" 195 > logs/chat-checkpoint-partial-qsa.log 2>&1
DENSE_MODEL="$("$PY" -c 'import sys;sys.path.insert(0,"scripts");from efficiency_compare_m1 import local_model;print(local_model("q27"))')"
"$BUILD/suc-qwen-checkpoint-tests" "$DENSE_MODEL" 195 > logs/chat-checkpoint-q27.log 2>&1
"$PY" scripts/efficiency_release_smoke.py --build "$BUILD" --app "$APP" --version 1.4.7.03 --output docs/optimization/RELEASE_SMOKE_1.4.7.03.json > logs/chat-checkpoint-release-smoke.log 2>&1
/usr/bin/python3 - "$BUILD" <<'PY'
import hashlib,json,sys
from pathlib import Path
before=json.loads(Path('logs/chat-checkpoint-before-rebuild.json').read_text());after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}
cases={}
for label,file in [('wide_partial_qsa_chunk63','chat-checkpoint-partial-qsa.log'),('qwen38_27b_side8_chunk64','chat-checkpoint-q27.log')]:
    rows=[json.loads(x) for x in Path('logs',file).read_text().splitlines() if x.startswith('{"case":')]
    assert len(rows)==5 and all(x['max_abs']==0 and x['equal_logits']==x['paired_logits'] for x in rows)
    cases[label]=rows
out={'passed':True,'binary_commit':Path(sys.argv[1],'GIT_COMMIT').read_text().strip(),'binary_sha256_before':before,'binary_sha256_after':after,'rebuild_byte_identical':before==after,'cases':cases}
Path('docs/optimization/CHAT_PREFIX_FINAL_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n')
PY
printf 'CHECKPOINT_FINAL_QUALIFICATION_PASS\n'
