// Print the EXACT shared server rendering for independent HF token-count checks. No GPU/model execution.
#include <fstream>
#include <iostream>
#include <iterator>
#include "suc/suc1.hpp"
#include "tool_common.hpp"
int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::runtime_error("usage: chat-prompt-probe MODEL REQUEST.json");
    suc::OpenOptions options;options.verify_checksum=false; // metadata-only probe; model payload is not executed
    auto file=suc::Suc1File::open(argv[1],options);std::ifstream in(argv[2]);
    if(!in)throw std::runtime_error("request unreadable");
    auto body=suc::Json::parse(std::string(std::istreambuf_iterator<char>(in),{}));
    std::vector<suc::tools::ChatMessage> messages;
    for(const auto& m:body["messages"].as_array())messages.push_back({m["role"].as_string(),m["content"].as_string()});
    bool think=true;
    if(const auto* k=body.find("chat_template_kwargs"))if(const auto* t=k->find("enable_thinking"))think=t->as_bool();
    const auto effort=body.find("reasoning_effort")?body["reasoning_effort"].as_string():"";
    std::cout<<suc::tools::chat_prompt_text(file.directory()["architecture"],messages,think,effort);
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
