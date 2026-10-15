#!/bin/bash
# Build scripts/gguf/ds4_score.c against a ds4 checkout (default ~/Projects/ds4), linking its core objects; the
# objects are (re)built by ds4's own Makefile through the ds4-bench target. Output: $DS4/ds4_score_suc (run it with
# cwd = $DS4, as every ds4 tool).
set -euo pipefail
DS4=${DS4:-$HOME/Projects/ds4}
SRC=$(cd "$(dirname "$0")" && pwd)/ds4_score.c
make -C "$DS4" -s ds4-bench >/dev/null
cd "$DS4"
cc -O3 -mcpu=native -Wall -Wextra -std=c11 -I. -o ds4_score_suc "$SRC" \
   ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o ds4_engram.o \
   -lm -pthread -framework Foundation -framework Metal
echo "$DS4/ds4_score_suc (ds4 $(git -C "$DS4" rev-parse --short HEAD))"
