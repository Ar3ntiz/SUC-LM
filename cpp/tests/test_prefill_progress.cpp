#include "suc/prefill_progress.hpp"
#include <cassert>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>
int main(){
  using Row=std::tuple<std::size_t,std::size_t,std::size_t>;
  std::vector<Row> rows;
  {
    suc::PrefillProgressScope scope([&](auto p,auto t,auto c){rows.emplace_back(p,t,c);});
    suc::prefill_progress_begin(2165,2048);
    suc::prefill_progress_advance(64);suc::prefill_progress_advance(53);
    assert(rows.front()==Row(2048,2165,2048));assert(rows.back()==Row(2165,2165,2048));
    assert(rows.size()==2); // bounded UI overhead: start + completion for a fast request
    std::thread other([]{suc::prefill_progress_begin(100,0);suc::prefill_progress_advance(100);});other.join();assert(rows.size()==2);
  }
  assert(!suc::current_prefill_progress);
  try {
    suc::PrefillProgressScope scope([](auto,auto,auto){throw std::runtime_error("cancel");});
    suc::prefill_progress_begin(100,0);assert(false);
  }catch(const std::runtime_error&){}
  assert(!suc::current_prefill_progress);
  suc::prefill_progress_advance(99); // no observer outside prefill/decode
}
