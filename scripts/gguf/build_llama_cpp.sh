#!/bin/bash
# Pinned llama.cpp (Metal, static libraries) for the GGUF comparison of the small models, plus scripts/gguf/llama_score
# linked against it. Tools: $LLAMA_DIR/build/bin (llama-perplexity, llama-quantize, llama-server, llama-tokenize, ...),
# converter venv $LLAMA_DIR/.venv (convert_hf_to_gguf.py requirements, cmake, ninja), $LLAMA_DIR/build/bin/llama_score.
# No Homebrew on the Studio; no sudo.   scripts/gguf/build_llama_cpp.sh
set -euo pipefail
TAG=${LLAMA_TAG:-b11477}
LLAMA_DIR=${LLAMA_DIR:-$HOME/suc-local/llama.cpp}
SRC=$(cd "$(dirname "$0")" && pwd)/llama_score.cpp
export PATH=$HOME/.local/bin:$PATH
if [ ! -d "$LLAMA_DIR" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
[ "$(git describe --tags)" = "$TAG" ] || { echo "llama.cpp checkout is $(git describe --tags), expected $TAG"; exit 1; }
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
uv pip install -q --python .venv/bin/python cmake ninja -r requirements/requirements-convert_hf_to_gguf.txt
.venv/bin/cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF >/dev/null
.venv/bin/cmake --build build -j 8 --target llama-perplexity llama-quantize llama-server llama-tokenize llama-imatrix llama
c++ -O3 -std=c++17 -I include -I ggml/include -o build/bin/llama_score "$SRC" \
  build/src/libllama.a build/ggml/src/libggml.a build/ggml/src/ggml-metal/libggml-metal.a \
  build/ggml/src/ggml-blas/libggml-blas.a build/ggml/src/libggml-cpu.a build/ggml/src/libggml-base.a \
  -framework Foundation -framework Metal -framework MetalKit -framework Accelerate -lc++
echo "$LLAMA_DIR/build/bin/llama_score (llama.cpp $TAG)"
