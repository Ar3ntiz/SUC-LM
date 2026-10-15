#include "../Engine/inference_service.hpp"
#include <cassert>
#include <future>

// Deterministic fake graph ONLY in this test executable. Production always loads the real Metal graph.
struct TestEngine : Engine {
  void reset() override {}
  uint32_t prefill(const std::vector<uint32_t>&) override { return 0; }
  uint32_t step(uint32_t) override { return 0; }
  std::vector<float> last_logits() override { return {1,0}; }
};
static std::string response_for(const std::string& request) {
  int fds[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,fds)==0);
  auto worker=std::async(std::launch::async,[&]{std::string buffer;HttpRequest r;bool ok=read_request(fds[1],buffer,r);::shutdown(fds[1],SHUT_WR);return ok;});
  send_all(fds[0],request);::shutdown(fds[0],SHUT_WR);
  std::string out;char b[1024];ssize_t n;while((n=::read(fds[0],b,sizeof b))>0)out.append(b,size_t(n));
  bool ok=worker.get();::close(fds[0]);::close(fds[1]);return ok?"accepted":out;
}
int main(){
  std::vector<std::string> events;output_sink=[&](const std::string& e){events.push_back(e);};
  assert(!model_io_logging);model_io_logging=true;
  assert(constant_equal("abc","abc"));assert(!constant_equal("abc","abd"));assert(!constant_equal("abc","abcd"));
  assert(bounded_json("{\"x\":\"[\\\"\",\"y\":[1]}"));assert(!bounded_json(std::string(65,'[')+std::string(65,']')));
  api_secret=std::string(64,'a');
  assert(response_for("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n").find("401")!=std::string::npos);
  assert(response_for("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")=="accepted");
  auto authorized="GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer "+api_secret+"\r\n";
  assert(response_for(authorized+"\r\n")=="accepted");
  ++api_epoch;assert(response_for(authorized+"\r\n").find("401")!=std::string::npos);api_epoch=0;
  assert(response_for(authorized+"Origin: https://evil.example\r\n\r\n").find("401")!=std::string::npos);
  assert(response_for("GET /health HTTP/1.1\r\nHost: evil.example\r\n\r\n").find("401")!=std::string::npos);
  assert(response_for(authorized+"Content-Length: 3000000\r\n\r\n").find("413")!=std::string::npos);
  assert(response_for(authorized+"Authorization: Bearer duplicate\r\n\r\n")!="accepted");
  Server s;TestEngine engine;s.engine=&engine;s.model_name="tiny";s.vocab=2;s.max_ctx=16;s.stop_ids={0};
  auto params=parse_generation(s,Json::parse("{\"model\":\"tiny\",\"prompt\":[1],\"max_tokens\":2}"),false);
  assert(params.prompt==std::vector<uint32_t>{1});
  bool rejected=false;try{parse_generation(s,Json::parse("{\"model\":\"wrong\",\"prompt\":[1]}"),false);}catch(...){rejected=true;}assert(rejected);
  rejected=false;try{parse_generation(s,Json::parse("{\"prompt\":[1],\"max_tokens\":16}"),false);}catch(...){rejected=true;}assert(rejected);
  {
    StudioRequestContext request("test","chat");StudioTurn turn;assert(turn.enter());
    auto result=generate(s,params,[](const std::string&){return true;});assert(result.completion_tokens==1);assert(result.decode_ms_per_token()==0);
    current_request->cancelled=true;rejected=false;try{generate(s,params,[](const std::string&){return true;});}catch(...){rejected=true;}assert(rejected);
  }
  bool saw_prefill=false,saw_metrics=false;
  for(auto& line:events){auto e=Json::parse(line);if(e["event"].as_string()=="prefill")saw_prefill=true;if(e["event"].as_string()=="metrics")saw_metrics=true;}
  assert(saw_prefill&&saw_metrics);
  // Exercise the actual HTTP/SSE renderer with a deterministic test graph and tokenizer.
  std::string vocab="{";
  for(int byte=0;byte<256;++byte){
    auto cp=suc::tok_detail::byte_to_unicode(uint8_t(byte));std::string character;
    if(cp<128)character+=char(cp);else{character+=char(0xc0|(cp>>6));character+=char(0x80|(cp&63));}
    vocab+=(byte?",":"")+jstr(character)+":"+std::to_string(byte);
  }
  vocab+="}";
  auto tokenizer=Tokenizer::from_json("{\"model\":{\"type\":\"BPE\",\"vocab\":"+vocab+",\"merges\":[]},\"added_tokens\":[],\"decoder\":{\"type\":\"ByteLevel\"},\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"use_regex\":false}}");
  s.tok=&tokenizer;s.stop_ids.clear();
  int pair[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,pair)==0);
  auto http=std::async(std::launch::async,[&]{serve_connection(s,pair[1]);});
  std::string body=R"({"model":"tiny","prompt":[1],"max_tokens":2,"stream":true,"stream_options":{"include_usage":true}})";
  send_all(pair[0],"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer "+api_secret+"\r\nContent-Length: "+std::to_string(body.size())+"\r\n\r\n"+body);
  std::string answer;char data[4096];ssize_t bytes;while((bytes=::read(pair[0],data,sizeof data))>0)answer.append(data,size_t(bytes));::close(pair[0]);http.get();
  assert(answer.find("200 OK")!=std::string::npos);assert(answer.find("data: [DONE]")!=std::string::npos);assert(answer.find("\"completion_tokens\": 2")!=std::string::npos);
  assert(answer.find("\"suc_metrics\"")!=std::string::npos);assert(answer.find("\"prefill_s\"")!=std::string::npos);assert(answer.find("\"decode_s\"")!=std::string::npos);
  bool saw_model_input=false,saw_model_output=false;
  for(auto& line:events){auto e=Json::parse(line);saw_model_input|=e["event"].as_string()=="model_input";saw_model_output|=e["event"].as_string()=="model_output";}
  assert(saw_model_input&&saw_model_output);
  model_io_logging=false;
  // Cancellation removes queued tickets; a following request must not deadlock.
  StudioTurn first;assert(first.enter());
  std::promise<void> registered;auto registration=registered.get_future();
  auto queued=std::async(std::launch::async,[&]{StudioRequestContext request("cancel-me","chat");registered.set_value();StudioTurn turn;return turn.enter();});
  registration.wait();{std::lock_guard<std::mutex> l(requests_mutex);requests.at("cancel-me")->cancelled=true;}queue_cv.notify_all();assert(!queued.get());
  std::puts("service_tests: authentication, limits, metrics, context and cancellation passed");
}
