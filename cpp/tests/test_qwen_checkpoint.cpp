// GPU qualification of in-place hybrid checkpoints. One model, paired identical positions, no downloads.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include "suc/inference_engine.hpp"
#include "suc/live_prefix.hpp"
#include "suc/qwen4.hpp"
#include "suc/tokenizer.hpp"
#include "../tools/tool_common.hpp"
using namespace suc;
using namespace suc::metal;
using namespace suc::tools;
static void require(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
struct Graph final:InferenceEngine {
 Qwen4Model m;
 std::vector<std::pair<std::size_t,std::vector<float>>> rows;
 std::size_t record_from=~std::size_t(0);
 Graph(Context& c,const MappedFile& f,const Qwen4Options& o):m(c,f,o){}
 void reset() override{m.reset();}
 std::size_t checkpoint_stride()const override{return m.checkpoint_stride();}
 bool save_checkpoint()override{return m.save_checkpoint();}
 bool restore_checkpoint()override{return m.restore_checkpoint();}
 void clear_checkpoint()override{m.clear_checkpoint();}
 uint32_t prefill(const std::vector<uint32_t>& p)override{
   const auto start=m.position();
   if(start>=record_from){rows.emplace_back(start,m.forward_rows(p));auto l=m.logits();return std::max_element(l.begin(),l.end())-l.begin();}
   return m.forward(p);
 }
 uint32_t step(uint32_t t)override{return m.step(t);}
 std::vector<float> last_logits()override{return m.logits();}
};
static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(int argc,char** argv){try{
 require(argc>=2,"usage: suc-qwen-checkpoint-tests MODEL [prefix-tokens] [tokenizer.json]");
 const auto length=argc>2?std::stoul(argv[2]):192;
 auto f=Suc1File::open(argv[1]);Context ctx;MappedFile mf(ctx,f,qwen4_table_ranges(f));
 Qwen4Options opt;opt.chunk=64;opt.layer_prefill=true;opt.max_context=length+2048;
 if(const char* chunk=std::getenv("SUC_CHECKPOINT_TEST_CHUNK"))opt.chunk=std::stoul(chunk);
 require(opt.chunk>0&&opt.chunk<=1024,"test chunk out of range");
 auto graph=std::make_unique<Graph>(ctx,mf,opt);auto* g=graph.get();LivePrefixEngine<InferenceEngine> cache(std::move(graph));
 const auto vocab=g->m.config().vocab;
 std::vector<std::vector<uint32_t>> prompts;
 if(argc>3){
  auto tok=Tokenizer::from_file(argv[3]);const auto arch=f.directory()["architecture"];
  std::string text;for(std::size_t i=0;i<length/8;++i)text+="Record "+std::to_string(i)+": alpha beta gamma delta.\n";
  for(bool think:{false,true}){
   std::vector<ChatMessage> msgs={{"system","Answer briefly."},{"user",text+"Summarize."}};
   prompts.push_back(tok.encode(chat_prompt_text(arch,msgs,think,"medium")));
   msgs.push_back({"assistant","<think>Internal calculation omitted.</think>\n\nA short summary."});
   msgs.push_back({"user","Give one detail."});prompts.push_back(tok.encode(chat_prompt_text(arch,msgs,think,"medium")));
   msgs.back().content="Give two details.";prompts.push_back(tok.encode(chat_prompt_text(arch,msgs,think,"medium")));
   msgs[0].content="Answer in Italian.";prompts.push_back(tok.encode(chat_prompt_text(arch,msgs,think,"medium")));
  }
 }else{
  std::vector<uint32_t> p(length);for(std::size_t i=0;i<p.size();++i)p[i]=2+(i*7)%(vocab-2);
  prompts.push_back(p);p.insert(p.end(),{9,12,17,4,7});p[length-1]=5;prompts.push_back(p);
  prompts.push_back(p);p.resize(p.size()-2);prompts.push_back(p);p[0]=13;prompts.push_back(p);
 }
 for(std::size_t i=0;i<prompts.size();++i){
  const auto& p=prompts[i];g->rows.clear();g->record_from=~std::size_t(0);
  const double t0=now();auto next=cache.prepare(p);const double cached_s=now()-t0;
  const auto cached=cache.cached_tokens();auto actual=g->m.logits();auto bytes=g->m.checkpoint_bytes();
  // Save decode behavior before reset; never keep two copies of the model's weights/arena.
  std::vector<uint32_t> continuation;for(int n=0;n<4;++n){continuation.push_back(next);next=cache.step(next);}
  // The whole teacher-forced suffix is compared, with the same globally aligned chunk boundaries.
  // Save actual rows after restoring the checkpoint. The first call may have zero cached tokens.
  g->rows.clear();g->record_from=0;cache.prepare(p);auto captured=std::move(g->rows);
  const auto restored=cache.cached_tokens();
  require(!captured.empty(),"no evaluated rows");
  cache.reset();g->record_from=~std::size_t(0);const double cold_start=now();
  if(captured.front().first)g->m.forward(std::vector<uint32_t>(p.begin(),p.begin()+captured.front().first),false);
  std::size_t paired=0;double max_abs=0,max_kl=0;std::size_t bit_equal=0;
  for(const auto& [start,values]:captured){
   const auto count=values.size()/vocab;
   auto ref=g->m.forward_rows(std::vector<uint32_t>(p.begin()+start,p.begin()+start+count));
   require(ref.size()==values.size(),"row shape mismatch");
   for(std::size_t row=0;row<count;++row){
    const float* a=values.data()+row*vocab;const float* b=ref.data()+row*vocab;
    double ma=*std::max_element(a,a+vocab),mb=*std::max_element(b,b+vocab),sa=0,sb=0;
    for(std::size_t j=0;j<vocab;++j){require(std::isfinite(a[j])&&std::isfinite(b[j]),"nonfinite logits");sa+=std::exp(a[j]-ma);sb+=std::exp(b[j]-mb);}
    double kl=0;for(std::size_t j=0;j<vocab;++j){max_abs=std::max(max_abs,double(std::abs(a[j]-b[j])));bit_equal+=a[j]==b[j];double la=a[j]-ma-std::log(sa),lb=b[j]-mb-std::log(sb);kl+=std::exp(la)*(la-lb);}
    max_kl=std::max(max_kl,kl);++paired;
   }
  }
  auto reference=g->m.logits();require(actual==reference,"normal prefill logits differ from cold reference");
  next=std::max_element(reference.begin(),reference.end())-reference.begin();
  for(auto expected:continuation){require(next==expected,"greedy continuation mismatch");next=g->m.step(next);}
  require(max_abs==0,"checkpoint logits differ at paired suffix positions");
  const double cold_s=now()-cold_start;
  std::printf("{\"case\":%zu,\"tokens\":%zu,\"cached_tokens\":%zu,\"repeated_cached_tokens\":%zu,\"checkpoint_bytes\":%zu,\"paired_positions\":%zu,\"paired_logits\":%zu,\"equal_logits\":%zu,\"max_abs\":%.9g,\"max_kl\":%.9g,\"cached_prepare_s\":%.6f,\"reference_with_rows_s\":%.6f}\n",i,p.size(),cached,restored,bytes,paired,paired*vocab,bit_equal,max_abs,max_kl,cached_s,cold_s);std::fflush(stdout);
  // Re-establish the previous request's checkpoint and generated state for the following conversation turn.
  cache.reset();g->rows.clear();g->record_from=~std::size_t(0);next=cache.prepare(p);
  for(int n=0;n<4;++n)next=cache.step(next);
 }
 // Allocation cap, reset invalidation and unavailable checkpoint must safely fall back.
 require(!g->m.save_checkpoint(0),"zero allocation cap accepted");require(!g->m.restore_checkpoint(),"failed save retained stale checkpoint");
 cache.reset();require(!g->m.restore_checkpoint(),"reset retained checkpoint");
 return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
