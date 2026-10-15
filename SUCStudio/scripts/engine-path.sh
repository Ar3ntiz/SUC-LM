#!/bin/bash
# Shared by all build/test entry points. No vendored fallback or external override.
SUC_APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -d "$SUC_APP_ROOT/../cpp/include/suc" || ! -f "$SUC_APP_ROOT/../cpp/tools/tool_common.hpp" ]]; then
  echo "SUC Studio richiede il motore del progetto padre: $SUC_APP_ROOT/../cpp. Mantieni SUCStudio dentro SUC-LM; nessuna copia runtime locale viene usata." >&2
  exit 1
fi
ENGINE_ROOT="$(cd "$SUC_APP_ROOT/../cpp" && pwd)"
unset SUC_APP_ROOT
