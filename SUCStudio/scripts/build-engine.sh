#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/engine-path.sh"
ARCH="${ARCH:-arm64}"
OUT="$ROOT/.build/$ARCH"
mkdir -p "$OUT/obj" "$OUT/gen"
CXX="$(xcrun --find clang++)"
SDK="${SDKROOT:-$(xcrun --show-sdk-path)}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
[[ -x "$PYTHON_BIN" ]] || { echo "Python non disponibile: $PYTHON_BIN" >&2; exit 1; }
FLAGS=(-std=c++20 -O2 -ffp-contract=off -arch "$ARCH" -mmacosx-version-min=15.0 -isysroot "$SDK" -isystem "$SDK/usr/include/c++/v1" -I"$ENGINE_ROOT/include" -isystem "$ENGINE_ROOT/third_party/metal-cpp" -I"$ENGINE_ROOT/tools" -pthread)
write_manifest() {
  local target="$1" temporary="${1}.tmp.$$"
  rm -f "$temporary"
  if ! "$PYTHON_BIN" "$ROOT/scripts/engine-manifest.py" "$ENGINE_ROOT" > "$temporary" || [[ ! -s "$temporary" ]]; then
    rm -f "$temporary"
    echo "Impossibile generare il manifest non vuoto del motore padre." >&2
    exit 1
  fi
  "$PYTHON_BIN" -c 'import json,sys; value=json.load(open(sys.argv[1])); assert value.get("files") and value.get("content_sha256")' "$temporary"
  mv "$temporary" "$target"
}
write_manifest "$OUT/engine-inputs.json"
# Only local, embedded Metal sources. No download and no offline Metal compiler.
"$PYTHON_BIN" - "$ENGINE_ROOT" "$OUT" <<'PY'
from pathlib import Path
import sys
root,out=map(Path,sys.argv[1:]);names=['','Model','Qwen4','Ds4','Glm5'];files=['suc_kernels','suc_model_kernels','suc_qwen4_kernels','suc_ds4_kernels','suc_glm5_kernels']
s='namespace suc::metal {\n'
for n,f in zip(names,files):
 s+=f'extern const char* const kEmbedded{n}KernelSource;\nconst char* const kEmbedded{n}KernelSource = R"SUCMSL('+ (root/'metal'/f'{f}.metal').read_text()+')SUCMSL";\n'
s+='}\n';p=out/'gen/kernels.cpp'
if not p.exists() or p.read_text()!=s:p.write_text(s)
PY
# Fingerprint content, not mtimes: checkout switches and restored older files
# must rebuild too. Removed source files must not survive as stale linked objects.
BUILD_SIG="$("$PYTHON_BIN" - "$OUT/engine-inputs.json" "${FLAGS[*]}" "$($CXX --version)" <<'PYHASH'
import hashlib,sys
from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()+(sys.argv[2]+sys.argv[3]).encode()).hexdigest())
PYHASH
)"
if [[ ! -f "$OUT/build.signature" || "$(cat "$OUT/build.signature")" != "$BUILD_SIG" || "${CLEAN:-0}" == 1 ]]; then
  CLEAN=1
  find "$OUT/obj" -type f -name '*.o' -delete
fi
OBJS=()
for src in "$ENGINE_ROOT"/src/*.cpp "$ENGINE_ROOT"/src/metal/*.cpp "$OUT/gen/kernels.cpp"; do
  name="${src#"$ENGINE_ROOT/"}"; [[ "$src" == "$OUT/gen/kernels.cpp" ]] && name="embedded_kernels.cpp"
  name="${name//\//_}"; obj="$OUT/obj/${name%.cpp}.o"
  if [[ ! -f "$obj" || "$src" -nt "$obj" || "${CLEAN:-0}" == 1 ]]; then "$CXX" "${FLAGS[@]}" -c "$src" -o "$obj"; fi
  OBJS+=("$obj")
done
"$CXX" "${FLAGS[@]}" "$ROOT/Engine/studio_engine.cpp" "${OBJS[@]}" -framework Metal -framework Foundation -framework QuartzCore -o "$OUT/suc-studio-engine"
"$CXX" "${FLAGS[@]}" "$ENGINE_ROOT/tests/test_main.cpp" "${OBJS[@]}" -framework Metal -framework Foundation -framework QuartzCore -o "$OUT/suc-tests"
write_manifest "$OUT/engine-inputs-after.json"
if ! cmp -s "$OUT/engine-inputs.json" "$OUT/engine-inputs-after.json"; then
  echo "Il motore padre è cambiato durante la compilazione. Ripeti la build." >&2
  exit 1
fi
cp "$OUT/engine-inputs.json" "$OUT/ENGINE_BUILD_MANIFEST.json"
printf '%s\n' "$BUILD_SIG" > "$OUT/build.signature"
printf 'Engine built: %s\n' "$OUT/suc-studio-engine"
