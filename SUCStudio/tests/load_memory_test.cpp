#include "load_memory.hpp"
#include <cassert>
#include <atomic>
#include <fstream>
#include <cstdio>
int main(){
 char file[]="/tmp/suc-load-memory-XXXXXX";int fd=mkstemp(file);assert(fd>=0);assert(ftruncate(fd,32768)==0);close(fd);
 std::atomic<int> samples{0};
 {LoadMemoryMonitor m(file,[&](const std::string&,const LoadMemoryMonitor::Sample& s){assert(s.target==32768);assert(s.resident<=s.target);assert(s.initial<=s.target);assert(s.valid);++samples;});
  m.phase("Compilazione Metal");std::this_thread::sleep_for(std::chrono::milliseconds(1200));}
 assert(samples>=2);auto n=samples.load();std::this_thread::sleep_for(std::chrono::milliseconds(100));assert(samples==n);
 unlink(file);
 {LoadMemoryMonitor m("/not/a/model",[](const std::string&,const LoadMemoryMonitor::Sample& s){assert(!s.valid);});}
 puts("Resident-page observation, stop/join and unavailable file fallback: PASS");
}
