// Compare query-reuse prefill attention with the vector reference, synthetic data.
#include "suc/metal.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace suc::metal;
int main(){
 Context ctx; std::mt19937 rng(17);std::uniform_real_distribution<float> random(-.7f,.7f);
 for(auto act:{Act::BF16,Act::F16,Act::F32})for(std::size_t rows:{3u,64u}){
  const std::size_t D=128,Hq=16,Hkv=4,N=rows==3?17:2048,a=act_size(act);
  auto q=ctx.new_buffer(rows*Hq*D*a),k=ctx.new_buffer(N*Hkv*D*a),v=ctx.new_buffer(N*Hkv*D*a);
  auto ref=ctx.new_buffer(rows*Hq*D*a),tile=ctx.new_buffer(rows*Hq*D*a);
  for(auto b:{q,k,v}){std::vector<float>x(b->length()/a);for(auto& f:x)f=random(rng);f32_to_act(x.data(),b->contents(),x.size(),act);}
  auto run=[&](bool tiled,int iterations){
   auto cb=ctx.queue()->commandBuffer();auto enc=cb->computeCommandEncoder();
   auto args=std::string(act_msl(act))+", 128, 128"+(tiled?", 2":"");
   enc->setComputePipelineState(ctx.pipeline(tiled?"suc_sdpa_queries":"suc_sdpa_vector",args));
   enc->setBuffer(q.get(),0,0);enc->setBuffer(k.get(),0,1);enc->setBuffer(v.get(),0,2);enc->setBuffer((tiled?tile:ref).get(),0,3);
   const int gqa=4,nn=int(N);const std::size_t hs=D,ss=Hkv*D;const float scale=1/std::sqrt(float(D));const uint32_t nr=uint32_t(rows);
   enc->setBytes(&gqa,4,4);enc->setBytes(&nn,4,5);enc->setBytes(&hs,sizeof hs,6);enc->setBytes(&ss,sizeof ss,7);
   enc->setBytes(&hs,sizeof hs,8);enc->setBytes(&ss,sizeof ss,9);enc->setBytes(&scale,4,10);if(tiled)enc->setBytes(&nr,4,11);
   for(int it=0;it<iterations;++it){enc->dispatchThreadgroups(MTL::Size(Hq,tiled?(rows+1)/2:rows,1),MTL::Size(1024,1,1));enc->memoryBarrier(MTL::BarrierScopeBuffers);}
   enc->endEncoding();cb->commit();cb->waitUntilCompleted();
   if(cb->status()!=MTL::CommandBufferStatusCompleted)throw std::runtime_error("attention command failed");
   return 1000*(cb->GPUEndTime()-cb->GPUStartTime())/iterations;
  };
  run(false,1);run(true,1);double old=run(false,10),now=run(true,10);
  std::vector<float>x(rows*Hq*D),y(x.size());act_to_f32(ref->contents(),x.data(),x.size(),act);act_to_f32(tile->contents(),y.data(),y.size(),act);
  float error=0;for(std::size_t i=0;i<x.size();++i)error=std::max(error,std::abs(x[i]-y[i]));
  std::printf("{\"dtype\":\"%s\",\"rows\":%zu,\"context\":%zu,\"reference_ms\":%.6f,\"tiled_ms\":%.6f,\"max_error\":%.9g,\"identical\":%s}\n",act_msl(act),rows,N,old,now,error,std::memcmp(ref->contents(),tile->contents(),rows*Hq*D*a)==0?"true":"false");
  if(!std::isfinite(error)||error>1e-5)return 1;
 }
}
