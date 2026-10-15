// Tiny static-partition parallel_for over std::thread (no thread pool; work items are coarse rows).
// The result never depends on the thread count: every index is processed by exactly one thread with
// the same sequential arithmetic.
#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <thread>
#include <vector>

namespace suc {

// Number of worker threads (SUC_THREADS environment variable, default hardware_concurrency).
unsigned num_threads();
void set_num_threads(unsigned n);   // 0 = default

template <typename F>
void parallel_for(std::size_t n, std::size_t min_per_thread, F&& f) {
  const std::size_t want = std::max<std::size_t>(1, n / std::max<std::size_t>(1, min_per_thread));
  const std::size_t t = std::min<std::size_t>(num_threads(), want);
  if (t <= 1) {
    for (std::size_t i = 0; i < n; ++i) f(i);
    return;
  }
  std::vector<std::thread> pool;
  std::vector<std::exception_ptr> errors(t);
  for (std::size_t k = 0; k < t; ++k) {
    pool.emplace_back([&, k] {
      try {
        for (std::size_t i = n * k / t; i < n * (k + 1) / t; ++i) f(i);
      } catch (...) {
        errors[k] = std::current_exception();
      }
    });
  }
  for (auto& th : pool) th.join();
  for (auto& e : errors)
    if (e) std::rethrow_exception(e);
}

}  // namespace suc
