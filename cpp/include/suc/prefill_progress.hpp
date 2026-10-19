// Optional request-local observation only; never splits or reschedules GPU work.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <utility>
namespace suc {
struct PrefillProgress {
  using Clock = std::chrono::steady_clock;
  std::function<void(std::size_t,std::size_t,std::size_t)> emit;
  std::size_t total=0, processed=0, cached=0;
  Clock::time_point last{};
  bool begun=false;
  void publish(bool force=false) {
    auto now=Clock::now();
    if (emit && (force || now-last>=std::chrono::milliseconds(250))) {
      last=now; emit(processed,total,cached);
    }
  }
};
inline thread_local PrefillProgress* current_prefill_progress=nullptr;
struct PrefillProgressScope {
  PrefillProgress value;
  PrefillProgress* previous;
  explicit PrefillProgressScope(std::function<void(std::size_t,std::size_t,std::size_t)> f)
      : value{std::move(f)}, previous(current_prefill_progress) { current_prefill_progress=&value; }
  ~PrefillProgressScope(){ current_prefill_progress=previous; }
  PrefillProgressScope(const PrefillProgressScope&)=delete;
  PrefillProgressScope& operator=(const PrefillProgressScope&)=delete;
};
inline void prefill_progress_begin(std::size_t total,std::size_t cached) {
  if(auto* p=current_prefill_progress){p->total=total;p->cached=std::min(cached,total);p->processed=p->cached;p->begun=true;p->publish(true);}
}
inline void prefill_progress_advance(std::size_t count) {
  if(auto* p=current_prefill_progress; p && p->begun){p->processed+=std::min(count,p->total-p->processed);p->publish(p->processed==p->total);}
}
}
