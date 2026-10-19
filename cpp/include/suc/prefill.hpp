// Shared conservative prompt chunk policy. Explicit requests always win; callers
// still enforce expert slot limits. Long contexts reserve space for KV and scratch.
#pragma once
#include <algorithm>
#include <cstddef>
#include <string_view>
namespace suc {
inline std::size_t prefill_chunk(std::string_view family, std::size_t tokens,
                                std::size_t requested = 0, bool exact = false,
                                bool batched = true, bool streamed = false) {
  if (!batched) return 1;
  if (requested) return requested;
  if (exact) return std::min<std::size_t>(streamed ? 64 : 16, std::max<std::size_t>(tokens, 1));
  std::size_t cap = 64;
  if (!streamed) {
    // Larger Qwen chunks failed paired hard-suite non-regression (98 vs 100/120).
    // They remain available through an explicit request, not automatic policy.
    if (family == "deepseek_v4" || family == "glm5_next") cap = tokens > 8192 ? 1024 : 4096;
  }
  return std::min(cap, std::max<std::size_t>(tokens, 1));
}

// Service profiles predate the CLI's large-chunk policy. Preserve their
// arithmetic boundaries unless a caller explicitly requests another chunk.
inline std::size_t service_prefill_chunk(std::string_view family, std::size_t context,
                                        std::size_t requested = 0, bool exact = false,
                                        bool batched = true, bool streamed = false) {
  if (!batched || exact)
    return prefill_chunk(family, context, requested, exact, batched, streamed);
  if (requested) return requested;
  return family == "glm5_next" ? 512 : 64;
}
}
