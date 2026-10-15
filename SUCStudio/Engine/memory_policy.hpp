#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

// App policy only. Estimates are conservative heuristics, not allocation guarantees.
struct StudioMemoryPlan {
  uint64_t limit = 0, resident = 0, reserve = 0, expert_budget = 0;
  bool streamed = false;
};
inline StudioMemoryPlan studio_memory_plan(uint64_t physical, uint64_t metal,
    uint64_t file_bytes, uint64_t removable, uint64_t reserve,
    const std::string& mode, uint64_t manual) {
  if (mode != "auto" && mode != "resident" && mode != "manual")
    throw std::runtime_error("Modalità memoria non valida");
  if (!physical || !metal || removable > file_bytes)
    throw std::runtime_error("Impossibile determinare un budget memoria affidabile");
  StudioMemoryPlan p;
  // The total process + OS target is 96 % of physical memory. Metal's recommended working set remains a hard upper
  // bound for GPU-backed allocations (and can be raised explicitly by iogpu.wired_limit_mb). The previous 12.5 %
  // physical margin duplicated the OS cost already observed outside Metal and unnecessarily reduced long contexts.
  const uint64_t physical_96 = physical - physical / 25;
  if (!physical_96) throw std::runtime_error("Memoria fisica insufficiente");
  p.limit = std::min(metal, physical_96);
  p.reserve = reserve;
  p.resident = file_bytes;
  const bool fits = reserve < p.limit && file_bytes <= p.limit - reserve;
  if (mode == "resident" || (mode == "auto" && fits)) {
    if (!fits) throw std::runtime_error("Stima memoria oltre il limite prudenziale. Riduci contesto/modello o usa Automatico per gli expert MoE.");
    return p;
  }
  if (!removable)
    throw std::runtime_error("Modello senza expert instradati: streaming SSD non disponibile nel motore attuale. Scegli un modello più piccolo o riduci il contesto. Lo swap macOS non sostituisce lo streaming dei pesi.");
  p.resident = file_bytes - removable;
  if (reserve >= p.limit || p.resident >= p.limit - reserve)
    throw std::runtime_error("Anche i pesi non instradati e la riserva superano il limite: riduci contesto o modello.");
  auto available = p.limit - reserve - p.resident;
  p.expert_budget = mode == "manual" ? manual : available;
  if (!p.expert_budget || p.expert_budget > available)
    throw std::runtime_error("Budget expert nullo o superiore alla memoria stimata disponibile");
  p.streamed = true;
  return p;
}
inline uint64_t studio_parse_budget(std::string s) {
  if (s.empty()) throw std::runtime_error("Inserisci un budget expert, ad esempio 8G");
  double scale=1;
  char last=s.back();
  if(last=='K'||last=='M'||last=='G') { scale=last=='K'?1024.0:last=='M'?1048576.0:1073741824.0; s.pop_back(); }
  if(s.empty() || s.find_first_not_of("0123456789.")!=std::string::npos)
    throw std::runtime_error("Budget non valido: usa un numero positivo e K/M/G");
  size_t n=0; double v=std::stod(s,&n)*scale;
  if(n!=s.size() || !std::isfinite(v) || v<1 || v>1099511627776.0)
    throw std::runtime_error("Budget expert fuori intervallo (1 byte–1 TiB)");
  return uint64_t(v);
}
