#include "suc/live_prefix.hpp"
#include <cassert>
#include <stdexcept>
#include <cstdlib>
struct Graph {
  virtual ~Graph()=default;
  virtual void reset()=0;
  virtual void invalidate_prefix(){}
  virtual std::size_t cached_tokens() const{return 0;}
  virtual std::size_t checkpoint_stride()const{return 0;}
  virtual bool save_checkpoint(){return false;}
  virtual bool restore_checkpoint(){return false;}
  virtual void clear_checkpoint(){}
  virtual uint32_t prepare(const std::vector<uint32_t>& p){reset();return prefill(p);}
  virtual uint32_t prefill(const std::vector<uint32_t>& p)=0;
  virtual uint32_t step(uint32_t t)=0;
  virtual std::vector<float> last_logits()=0;
  virtual bool generate(uint32_t,std::size_t,const std::function<bool(uint32_t)>&){return false;}
};
struct Recurrent:Graph {
  uint32_t state=0; int resets=0;
  void reset()override{state=0;++resets;}
  uint32_t step(uint32_t t)override{state=state*31+t;return state%997;}
  uint32_t prefill(const std::vector<uint32_t>& p)override{uint32_t out=0;for(auto t:p)out=step(t);return out;}
  std::vector<float> last_logits()override{return {float(state)};}
  bool generate(uint32_t first,std::size_t n,const std::function<bool(uint32_t)>& cb)override{for(std::size_t i=0;i<n;++i){first=step(first);if(!cb(first))break;}return true;}
};
struct CheckpointGraph final:Recurrent {
 uint32_t saved=0; bool valid=false,fail=false;
 std::size_t checkpoint_stride()const override{return 8;}
 bool save_checkpoint()override{saved=state;valid=true;return true;}
 bool restore_checkpoint()override{if(!valid)return false;state=saved;return true;}
 void clear_checkpoint()override{valid=false;}
 uint32_t prefill(const std::vector<uint32_t>& p)override{
   if(fail)throw std::runtime_error("prefill failed");return Recurrent::prefill(p);
 }
};
int main(){
 auto graph=std::make_unique<Recurrent>();auto* raw=graph.get();suc::LivePrefixEngine<Graph> cache(std::move(graph));
 cache.prepare({1,2,3});assert(cache.cached_tokens()==0);
 auto r=cache.prepare({1,2,3,4});assert(cache.cached_tokens()==3);assert(raw->resets==1);
 Recurrent fresh;assert(r==fresh.prefill({1,2,3,4}));
 cache.prepare({1,2});assert(cache.cached_tokens()==0);assert(raw->resets==2);
 cache.prepare({9,2,3});assert(cache.cached_tokens()==0);
 cache.invalidate_prefix();cache.prepare({9,2,3,4});assert(cache.cached_tokens()==0);
 cache.reset();cache.prepare({1,2});auto first=raw->state%997;auto out=first;
 cache.generate(first,3,[&](uint32_t t){out=t;return false;});
 cache.prepare({1,2,first,7});assert(cache.cached_tokens()==3);
 fresh.reset();assert(raw->state%997==fresh.prefill({1,2,first,7}));
 cache.prepare({1,2,first,7});assert(cache.cached_tokens()==0); // equal input safely replays
 (void)out;
 auto g=std::make_unique<CheckpointGraph>();auto* cp=g.get();suc::LivePrefixEngine<Graph> c(std::move(g));
 std::vector<uint32_t> a(160,2),b=a;b.resize(184,3);b[159]=4;
 auto check=[&](const auto& p,std::size_t expected){auto result=c.prepare(p);Recurrent f;
   assert(result==f.prefill(p));assert(c.cached_tokens()==expected);};
 check(a,0);assert(cp->valid); // checkpoint at 96
 c.step(9);c.step(10);check(b,96); // rewritten historical assistant / thinking marker
 check(b,120); // equal request: restore earlier prefix; checkpoint advanced to 120
 auto shorter=b;shorter.resize(150);check(shorter,120);
 auto changed=shorter;changed[0]=8;check(changed,0); // changed system / another session
 c.invalidate_prefix();assert(!cp->valid);check(changed,0);
 c.reset();assert(!cp->valid);
 check(a,0);cp->fail=true;try{c.prepare(b);assert(false);}catch(const std::runtime_error&){}
 assert(!cp->valid);cp->fail=false;check(b,0);
 setenv("SUC_PREFIX_CHECKPOINTS","0",1);
 auto no=std::make_unique<CheckpointGraph>();auto* np=no.get();suc::LivePrefixEngine<Graph> nc(std::move(no));
 nc.prepare(a);assert(!np->valid);nc.prepare(b);assert(nc.cached_tokens()==0);
 unsetenv("SUC_PREFIX_CHECKPOINTS");setenv("SUC_PREFIX_CACHE","0",1);
 suc::LivePrefixEngine<Graph> off(std::make_unique<CheckpointGraph>());off.prepare(a);off.prepare(a);
 assert(off.cached_tokens()==0);unsetenv("SUC_PREFIX_CACHE");
}
