#include "inference_service.hpp"
#include <future>
#include <sys/poll.h>
#include <mach/mach.h>

static std::mutex model_mutex;
static std::shared_ptr<LoadedEngine> loaded;
static std::atomic<bool> loading{false};
static std::atomic<int> connections{0}, tasks{0};
static std::atomic<int> listener{-1};
static std::thread accept_thread;
static std::mutex jobs_mutex;
static std::vector<std::future<void>> jobs;
static void spawn(std::function<void()> work) {
  std::lock_guard<std::mutex> l(jobs_mutex);
  jobs.erase(std::remove_if(jobs.begin(),jobs.end(),[](auto& f){return f.wait_for(std::chrono::seconds(0))==std::future_status::ready;}),jobs.end());
  jobs.push_back(std::async(std::launch::async,[work=std::move(work)]() mutable {
    try{work();}catch(const std::exception& e){event("error","", "{\"message\":"+jstr(e.what())+"}");}
    // A completed async shared state can outlive its callable's execution in jobs. Release captures
    // now: queued work often owns the model graph, which must not survive an explicit unload.
    work={};
  }));
}
static std::shared_ptr<LoadedEngine> model() {
  std::lock_guard<std::mutex> l(model_mutex);if(!loaded)throw std::runtime_error("Load a model first");return loaded;
}
static void cancel_request(const std::string& id) {
  std::lock_guard<std::mutex> l(requests_mutex);auto it=requests.find(id);if(it!=requests.end())it->second->cancelled=true;queue_cv.notify_all();
}
static void stop_server() {
  ++api_epoch;
  int fd=listener.exchange(-1);
  if(fd>=0){::shutdown(fd,SHUT_RDWR);::close(fd);}
  if(accept_thread.joinable())accept_thread.join();
  {std::lock_guard<std::mutex> l(requests_mutex);for(auto& [id,r]:requests)if(r->origin=="api")r->cancelled=true;}
  queue_cv.notify_all();event("server","","{\"running\":false}");
}
static void start_server(const Json& command) {
  auto m=model(); if(listener>=0)throw std::runtime_error("Server already running");
  auto host=command["host"].as_string();auto port=command["port"].as_int();auto key=command["key"].as_string();
  if(key.size()<32||key.size()>256||key.find_first_of("\r\n")!=std::string::npos)throw std::runtime_error("API key requires 32–256 characters");
  if(port<1||port>65535)throw std::runtime_error("Port must be 1–65535");
  if(host!="127.0.0.1"&&host!="0.0.0.0")throw std::runtime_error("Unsupported bind address");
  int fd=::socket(AF_INET,SOCK_STREAM,0);if(fd<0)throw std::runtime_error("Socket failed");
  int one=1;::setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one);
  sockaddr_in address{};address.sin_family=AF_INET;address.sin_port=htons(uint16_t(port));::inet_pton(AF_INET,host.c_str(),&address.sin_addr);
  if(::bind(fd,reinterpret_cast<sockaddr*>(&address),sizeof address)<0||::listen(fd,16)<0){auto error=std::string(std::strerror(errno));::close(fd);throw std::runtime_error("Cannot start server: "+error);}
  {std::lock_guard<std::mutex> l(auth_mutex);api_secret=key;}
  listener=fd;
  const auto epoch=api_epoch.load();
  accept_thread=std::thread([m,fd,epoch]{
    while(listener==fd&&!exiting){
      pollfd p{fd,POLLIN,0};if(::poll(&p,1,200)<=0)continue;
      int client=::accept(fd,nullptr,nullptr);if(client<0)continue;
      if(++connections>32){--connections;::close(client);continue;}
      timeval timeout{15,0};int one=1;
      ::setsockopt(client,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof timeout);::setsockopt(client,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof timeout);
      ::setsockopt(client,SOL_SOCKET,SO_NOSIGPIPE,&one,sizeof one);
      spawn([m,client,epoch]{connection_epoch=epoch;serve_connection(m->S,client);--connections;});
    }
  });
  event("server","","{\"running\":true,\"host\":"+jstr(host)+",\"port\":"+std::to_string(port)+"}");
}
static void command(const Json& c) {
  if(!c.is_object())throw std::runtime_error("Expected command object");
  const auto cmd=c["command"].as_string();
  const auto id=c.find("id")?c["id"].as_string():"";
  if(cmd=="hello"){event("ready",id,"{\"protocol\":1,\"version\":\"1.4.7\",\"phase_metrics\":true}");return;}
  if(cmd=="shutdown"){exiting=true;queue_cv.notify_all();return;}
  if(cmd=="set_model_log"){model_io_logging=c["enabled"].as_bool();return;}
  if(cmd=="cancel"){cancel_request(id);return;}
  if(cmd=="stop_server"){stop_server();return;}
  if(cmd=="start_server"){start_server(c);return;}
  if(cmd=="status"){
    task_vm_info_data_t info{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
    bool valid=task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count)==KERN_SUCCESS;
    event("status",id,"{\"requests\":"+std::to_string(live_requests.load())+",\"connections\":"+std::to_string(connections.load())+",\"footprint_bytes\":"+(valid?std::to_string(info.phys_footprint):"null")+"}");return;
  }
  if(cmd=="unload"){
    if(tasks||loading||live_requests||connections||listener>=0)throw std::runtime_error("Stop server and finish requests before unloading");
    {std::lock_guard<std::mutex> l(model_mutex);loaded.reset();}event("unloaded",id);return;
  }
  if(cmd=="load"){
    {std::lock_guard<std::mutex> l(model_mutex);if(loaded||loading.exchange(true))throw std::runtime_error("Unload current model first");}
    spawn([c,id]{try{
      RuntimeFlags flags;flags.max_ctx=4096;
      const auto path=c["path"].as_string();flags.tokenizer_path=c["tokenizer"].as_string();
      auto n=c["context"].as_int();if(n<128||n>1048576)throw std::runtime_error("Context must be 128–1048576");flags.max_ctx=size_t(n);
      if(const auto* a=c.find("runtime_args")){const auto& args=a->as_array();for(size_t i=0;i<args.size();++i){auto next=[&](){if(++i>=args.size())throw std::runtime_error("Missing runtime argument");return args[i].as_string();};const auto option=args[i].as_string(); if(option=="--expert-budget"){flags.xo.budget_bytes=studio_parse_budget(next());continue;} if(!parse_runtime_flag(option,next,flags))throw std::runtime_error("Unknown runtime argument");}}
      const auto mode=c.find("memory_mode")?c["memory_mode"].as_string():(flags.xo.budget_bytes?"manual":"auto");
      event("loading",id);auto m=std::make_shared<LoadedEngine>(path,flags,mode,id);
      {std::lock_guard<std::mutex> l(model_mutex);loaded=m;}loading=false;
      event("loaded",id,"{\"model\":"+jstr(m->S.model_name)+",\"family\":"+jstr(m->family)+",\"context\":"+std::to_string(m->S.max_ctx)+",\"load_s\":"+std::to_string(m->load_s)+"}");
    }catch(const std::exception& e){loading=false;event("error",id,"{\"message\":"+jstr(e.what())+"}");}});return;
  }
  if(cmd=="generate"){
    if(id.empty())throw std::runtime_error("Request id required");
    auto m=model();if(++tasks>9){--tasks;throw std::runtime_error("Chat queue full");}
    // Register synchronously so an immediate cancel cannot race request creation.
    auto state=std::make_shared<RequestState>();state->id=id;state->origin="chat";
    {std::lock_guard<std::mutex> l(requests_mutex);if(requests.count(id)){--tasks;throw std::runtime_error("Duplicate request id");}requests[id]=state;}
    spawn([m,c,id,state]{
      current_request=state;++live_requests;event("queued",id,"{\"origin\":\"chat\"}");
      try{
        StudioTurn turn;if(!turn.enter())throw std::runtime_error(studio_cancelled()?"Request cancelled":"Queue full");
        studio_started();auto g=parse_generation(m->S,c["body"],true);
        auto result=generate(m->S,g,[&](const std::string& delta){if(studio_cancelled())return false;event("delta",id,"{\"text\":"+jstr(delta)+"}");return true;});
        event(studio_cancelled()||result.aborted?"cancelled":"finished",id,"{\"reason\":"+jstr(result.finish)+"}");
      }catch(const std::exception& e){event(studio_cancelled()?"cancelled":"error",id,"{\"message\":"+jstr(e.what())+"}");}
      {std::lock_guard<std::mutex> l(requests_mutex);requests.erase(id);}current_request.reset();--live_requests;--tasks;
    });return;
  }
  throw std::runtime_error("Unknown command");
}
int main() {
  std::signal(SIGPIPE,SIG_IGN);
  event("ready","","{\"protocol\":1,\"version\":\"1.4.7\"}");
  // Bounded incremental input, also handles multiple commands per read without getline allocations.
  std::string pending;char buffer[16384];
  while(!exiting){
    auto n=::read(STDIN_FILENO,buffer,sizeof buffer);if(n<0&&errno==EINTR)continue;if(n<=0)break;
    pending.append(buffer,size_t(n));
    size_t end;
    while((end=pending.find('\n'))!=std::string::npos){
      auto line=pending.substr(0,end);pending.erase(0,end+1);
      if(line.size()>4*1024*1024){exiting=true;break;}
      std::string id;
      try{if(!bounded_json(line))throw std::runtime_error("Invalid JSON nesting");auto c=Json::parse(line);if(c.find("id"))id=c["id"].as_string();command(c);}catch(const std::exception& e){event("error",id,"{\"message\":"+jstr(e.what())+"}");}
      if(exiting)break;
    }
    if(pending.size()>4*1024*1024)break;
  }
  exiting=true;queue_cv.notify_all();stop_server();
  // EOF must not leave a model-owning orphan, even during opaque prefill or model loading.
  std::thread watchdog([]{std::this_thread::sleep_for(std::chrono::seconds(5));_exit(0);});watchdog.detach();
  for(;;){std::vector<std::future<void>> current;{std::lock_guard<std::mutex> l(jobs_mutex);current.swap(jobs);}if(current.empty())break;for(auto& j:current)j.wait();}
  loaded.reset();return 0;
}
