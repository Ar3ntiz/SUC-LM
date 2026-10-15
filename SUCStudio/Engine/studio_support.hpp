#pragma once
#include <condition_variable>
#include <deque>
#include <iostream>
#include <set>
#include "suc/sha256.hpp"

inline std::mutex output_mutex, queue_mutex;
inline std::condition_variable queue_cv;
inline std::deque<uint64_t> waiting;
inline uint64_t next_ticket = 1;
inline bool running_turn = false;
inline std::atomic<bool> exiting{false};
inline std::atomic<bool> model_io_logging{false};
inline std::atomic<int> live_requests{0};
inline std::function<void(const std::string&)> output_sink;
inline std::string json_quote(const std::string& s) {
  std::string out="\"";
  for (unsigned char c:s) {
    if(c=='"'||c=='\\'){out+='\\';out+=char(c);}
    else if(c<32){char b[8];std::snprintf(b,sizeof b,"\\u%04x",c);out+=b;}
    else out+=char(c);
  } return out+'"';
}
inline void event(const std::string& name,const std::string& id="",const std::string& payload="{}") {
  std::lock_guard<std::mutex> l(output_mutex);
  static uint64_t seq=0;
  std::string line="{\"version\":1,\"sequence\":"+std::to_string(++seq)+",\"event\":"+json_quote(name)+",\"id\":"+json_quote(id)+",\"payload\":"+payload+"}";
  if(output_sink) output_sink(line); else {std::cout<<line<<'\n';std::cout.flush();}
}
inline std::atomic<uint64_t> api_epoch{0};
inline thread_local uint64_t connection_epoch=0;
struct RequestState {uint64_t epoch=0;std::string id,origin;std::atomic<bool> cancelled{false};};
inline thread_local std::shared_ptr<RequestState> current_request;
inline std::mutex requests_mutex;
inline std::map<std::string,std::shared_ptr<RequestState>> requests;
struct StudioRequestContext {
  explicit StudioRequestContext(std::string id,std::string origin) {
    current_request=std::make_shared<RequestState>();current_request->id=std::move(id);current_request->origin=std::move(origin);current_request->epoch=connection_epoch;
    {std::lock_guard<std::mutex> l(requests_mutex);requests[current_request->id]=current_request;}
    ++live_requests;
    event("queued",current_request->id,"{\"origin\":"+json_quote(current_request->origin)+"}");
  }
  ~StudioRequestContext(){event("idle",current_request->id);{std::lock_guard<std::mutex> l(requests_mutex);requests.erase(current_request->id);}current_request.reset();--live_requests;}
};
inline bool studio_cancelled(){return exiting || (current_request && (current_request->cancelled || (current_request->origin=="api" && current_request->epoch!=api_epoch.load())));}
inline std::string studio_request_id(){return current_request?current_request->id:"";}
inline void studio_started(){event("started",studio_request_id());}
struct StudioTurn {
  bool owned=false;
  bool enter(){
    std::unique_lock<std::mutex> l(queue_mutex);
    if(waiting.size()>=8)return false;
    auto ticket=next_ticket++;waiting.push_back(ticket);
    queue_cv.wait(l,[&]{return studio_cancelled()||(!running_turn&&waiting.front()==ticket);});
    if(studio_cancelled()){waiting.erase(std::find(waiting.begin(),waiting.end(),ticket));queue_cv.notify_all();return false;}
    waiting.pop_front();running_turn=true;owned=true;return true;
  }
  ~StudioTurn(){if(owned){std::lock_guard<std::mutex> l(queue_mutex);running_turn=false;queue_cv.notify_all();}}
};
inline void studio_prefill(size_t n,double seconds){event("prefill",studio_request_id(),"{\"prompt_tokens\":"+std::to_string(n)+",\"prefill_s\":"+std::to_string(seconds)+"}");}
inline void studio_metrics(size_t p,size_t n,double prefill,double decode){event("metrics",studio_request_id(),"{\"prompt_tokens\":"+std::to_string(p)+",\"completion_tokens\":"+std::to_string(n)+",\"prefill_s\":"+std::to_string(prefill)+",\"decode_s\":"+std::to_string(decode)+",\"source\":\"suc-server\",\"aggregation\":\"mean\"}");}
inline std::mutex auth_mutex;
inline std::string api_secret;
inline bool constant_equal(const std::string& a,const std::string& b){size_t v=a.size()^b.size();for(size_t i=0;i<std::max(a.size(),b.size());++i)v|=(i<a.size()?a[i]:0)^(i<b.size()?b[i]:0);return v==0;}
inline bool studio_authorize(int fd,const std::string& path,const std::map<std::string,std::string>& headers){
  bool allowed=false;
  {std::lock_guard<std::mutex> l(auth_mutex);auto it=headers.find("authorization");allowed=!api_secret.empty()&&it!=headers.end()&&constant_equal(it->second,"Bearer "+api_secret);}
  // No browser cross-origin access. CLI/SDK clients have no Origin.
  if(headers.count("origin"))allowed=false;
  if((path=="/health"||path=="/v1/health")&&!headers.count("origin"))allowed=true;
  auto h=headers.find("host");
  if(h==headers.end())allowed=false;
  else { auto host=h->second.substr(0,h->second.find(':'));in_addr addr{};if(host!="localhost" && ::inet_pton(AF_INET,host.c_str(),&addr)!=1)allowed=false; }
  if(connection_epoch != api_epoch.load())allowed=false;
  if(allowed)return true;
  const std::string body="{\"error\":{\"message\":\"Unauthorized\",\"type\":\"authentication_error\"}}";
  const std::string response="HTTP/1.1 401 Unauthorized\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: "+std::to_string(body.size())+"\r\n\r\n"+body;
  ::send(fd,response.data(),response.size(),0);return false;
}
inline void verify_tokenizer(const std::string& id,const std::string& path){
  const auto p=id.find("tokenizer.json:sha256:");if(p==std::string::npos)return;
  auto expected=id.substr(p+std::string("tokenizer.json:sha256:").size(),64);
  if(expected.size()!=64)throw std::runtime_error("Invalid tokenizer hash");
  std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Tokenizer unreadable");
  suc::Sha256 h;char buf[65536];while(in){in.read(buf,sizeof buf);h.update(buf,size_t(in.gcount()));}
  auto hash=h.finish();if(suc::to_hex(hash.data(),hash.size())!=expected)throw std::runtime_error("Tokenizer SHA-256 mismatch");
}

inline bool bounded_json(const std::string& text){
  int depth=0;bool quoted=false,escaped=false;
  for(char c:text){if(quoted){if(escaped)escaped=false;else if(c=='\\')escaped=true;else if(c=='"')quoted=false;}else if(c=='"')quoted=true;else if(c=='{'||c=='['){if(++depth>64)return false;}else if(c=='}'||c==']'){if(--depth<0)return false;}}
  return depth==0&&!quoted;
}
