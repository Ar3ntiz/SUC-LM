#include "suc/adaptive_prefill.hpp"
#include "suc/live_prefix.hpp"
#include <cassert>
#include <cstdio>
struct Graph:suc::InferenceEngine {
 std::vector<std::size_t> calls;std::vector<uint32_t> fed,cp;
 void reset()override{fed.clear();cp.clear();}
 std::size_t checkpoint_stride()const override{return 1024;}
 bool save_checkpoint()override{cp=fed;return true;}
 bool restore_checkpoint()override{if(cp.empty())return false;fed=cp;return true;}
 void clear_checkpoint()override{cp.clear();}
 uint32_t prefill(const std::vector<uint32_t>& p)override{calls.push_back(p.size());fed.insert(fed.end(),p.begin(),p.end());return fed.back();}
 uint32_t step(uint32_t t)override{fed.push_back(t);return t;}
 std::vector<float> last_logits()override{return {float(fed.size())};}
};
int main(){
 for(std::size_t length:{1,63,64,65,1023,1024,1025,2048,2177}) {
  std::size_t done=0;while(done<length){auto n=suc::history_prefill_chunk(length-done);assert(n>0&&n<=1024);assert(n==1024||n<=64);done+=n;}assert(done==length);
 }
 auto g=std::make_unique<Graph>();auto* raw=g.get();
 auto adaptive=std::make_unique<suc::AdaptivePrefillEngine>(std::move(g));
 suc::LivePrefixEngine<suc::InferenceEngine> cached(std::move(adaptive));
 std::vector<uint32_t> prompt(2177);for(std::size_t i=0;i<prompt.size();++i)prompt[i]=i;
 cached.prepare(prompt);assert(raw->fed==prompt);
 assert((raw->calls==std::vector<std::size_t>{1024,1024,64,64,1}));
 raw->calls.clear();cached.prepare(prompt);assert(cached.cached_tokens()==2112);assert(raw->fed==prompt);
 assert((raw->calls==std::vector<std::size_t>{64,1}));
 cached.step(9);prompt.push_back(9);prompt.push_back(12);raw->calls.clear();cached.prepare(prompt);
 assert(cached.cached_tokens()==2178&&raw->fed==prompt);assert(raw->calls==std::vector<std::size_t>{1});
 raw->calls.clear();prompt.insert(prompt.end(),1100,7);cached.prepare(prompt);assert(raw->fed==prompt);for(auto n:raw->calls)assert(n<=64);
 cached.reset();assert(cached.cached_tokens()==0);cached.prepare(prompt);assert(raw->fed==prompt);
 puts("Adaptive 1024/64 scheduling, checkpoint replay and live-prefix extension: PASS");
}
