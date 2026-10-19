#include "suc/parallel.hpp"

#include <cstdlib>
#include <atomic>
#include <condition_variable>
#include <mutex>

namespace suc {

struct ParallelWorkers::Impl {
  std::mutex mu;
  std::condition_variable start, done;
  std::vector<std::thread> workers;
  std::function<void(std::size_t)> fn;
  std::atomic<std::size_t> next{0};
  std::size_t n=0, remaining=0, epoch=0;
  bool stopping=false;
  std::exception_ptr error;
  explicit Impl(unsigned count) {
    for(unsigned i=0;i<std::max(1u,count);++i) workers.emplace_back([this]{
      std::size_t seen=0;
      for(;;){
        {std::unique_lock lock(mu);start.wait(lock,[&]{return stopping||epoch!=seen;});if(stopping)return;seen=epoch;}
        try {for(std::size_t k;(k=next.fetch_add(1,std::memory_order_relaxed))<n;) fn(k);}
        catch(...){std::lock_guard lock(mu);if(!error)error=std::current_exception();}
        {std::lock_guard lock(mu);if(--remaining==0)done.notify_one();}
      }
    });
  }
  ~Impl(){ {std::lock_guard lock(mu);stopping=true;}start.notify_all();for(auto& t:workers)t.join(); }
};
ParallelWorkers::ParallelWorkers(unsigned count):p_(std::make_unique<Impl>(count)){}
ParallelWorkers::~ParallelWorkers()=default;
void ParallelWorkers::run(std::size_t n,const std::function<void(std::size_t)>& fn){
  if(!n)return;
  auto& p=*p_;std::unique_lock lock(p.mu);
  p.fn=fn;p.n=n;p.next=0;p.error=nullptr;p.remaining=p.workers.size();++p.epoch;p.start.notify_all();
  p.done.wait(lock,[&]{return p.remaining==0;});p.fn={};if(p.error)std::rethrow_exception(p.error);
}

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
