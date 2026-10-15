#include "suc/parallel.hpp"

#include <cstdlib>

namespace suc {

static unsigned g_threads = 0;

unsigned num_threads() {
  if (g_threads) return g_threads;
  if (const char* e = std::getenv("SUC_THREADS")) {
    int v = std::atoi(e);
    if (v > 0) return unsigned(v);
  }
  unsigned h = std::thread::hardware_concurrency();
  return h ? h : 1;
}

void set_num_threads(unsigned n) { g_threads = n; }

}  // namespace suc
