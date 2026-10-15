#include "../Engine/memory_policy.hpp"
#include <cassert>
#include <iostream>
#include <functional>
int main() {
  const uint64_t G=1ull<<30;
  auto rejects=[](std::function<void()> f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}assert(failed);};
  auto dense=studio_memory_plan(32*G,24*G,8*G,0,3*G,"auto",0);
  assert(!dense.streamed && dense.expert_budget==0);
  auto moe=studio_memory_plan(32*G,24*G,64*G,60*G,4*G,"auto",0);
  assert(moe.streamed && moe.resident==4*G && moe.expert_budget==moe.limit-8*G);
  auto manual=studio_memory_plan(32*G,24*G,8*G,6*G,3*G,"manual",2*G);
  assert(manual.streamed && manual.expert_budget==2*G);
  rejects([&]{studio_memory_plan(32*G,24*G,64*G,0,4*G,"auto",0);});
  rejects([&]{studio_memory_plan(32*G,24*G,64*G,60*G,4*G,"resident",0);});
  rejects([&]{studio_memory_plan(32*G,24*G,64*G,40*G,4*G,"auto",0);});
  rejects([&]{studio_memory_plan(32*G,24*G,8*G,6*G,3*G,"manual",24*G);});
  rejects([&]{studio_memory_plan(32*G,24*G,8*G,0,3*G,"manual",2*G);});
  rejects([&]{studio_memory_plan(32*G,0,8*G,6*G,3*G,"auto",0);});
  rejects([&]{studio_memory_plan(32*G,24*G,8*G,6*G,24*G,"auto",0);});
  rejects([&]{studio_memory_plan(32*G,24*G,8*G,6*G,3*G,"invalid",0);});
  // Target total use is 96 % of physical memory, while Metal's working-set recommendation remains authoritative.
  // With iogpu.wired_limit_mb=116000 the 128 GiB Studio therefore uses the complete ~113.28 GiB Metal allowance.
  const uint64_t metal116=116000ull<<20;
  assert(studio_memory_plan(128*G,metal116,8*G,0,G,"auto",0).limit==metal116);
  assert(studio_memory_plan(128*G,96*G,8*G,0,G,"auto",0).limit==96*G);
  assert(studio_memory_plan(16*G,uint64_t(10.7*G),8*G,0,G,"auto",0).limit==uint64_t(10.7*G));
  assert(studio_memory_plan(16*G,16*G,8*G,0,G,"auto",0).limit==16*G-(16*G)/25);
  auto ds4=studio_memory_plan(128*G,116000ull<<20,96800000000ull,90*G,5*G,"auto",0);
  assert(!ds4.streamed && ds4.expert_budget==0);
  auto glm=studio_memory_plan(128*G,116000ull<<20,99000000000ull,92*G,6*G,"resident",0);
  assert(!glm.streamed);
  auto boundary=studio_memory_plan(128*G,metal116,metal116-12*G,0,12*G,"auto",0);
  assert(!boundary.streamed);
  rejects([&]{studio_memory_plan(128*G,metal116,metal116-12*G+1,0,12*G,"auto",0);});
  assert(studio_memory_plan(4*G,4*G,G,0,G/4,"auto",0).limit==4*G-(4*G)/25);
  assert(studio_parse_budget("1.5G")==G+G/2);
  assert(studio_parse_budget("1024M")==G);
  for(auto s:{"", "-1G", "nan", "inf", "1e300", "1.2.3G", "0", "99999999999999999999999999999999G"})
    rejects([&]{studio_parse_budget(s);});
  std::cout << "MemoryPolicyTests: dense/MoE decisions, boundaries, reserves and invalid budgets passed\n";
}
