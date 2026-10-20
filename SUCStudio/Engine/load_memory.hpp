#pragma once
// Observe resident file pages; never fault pages in or change residency policy.
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <chrono>
#include <cstdint>
#include <string>

class LoadMemoryMonitor {
 public:
  struct Sample { uint64_t resident=0, initial=0, target=0, footprint=0, baseline=0; bool valid=false; double query_s=0; };
  using Callback=std::function<void(const std::string&,const Sample&)>;
  LoadMemoryMonitor(const std::string& path, Callback callback):callback_(std::move(callback)) {
    baseline_=footprint();
    int fd=::open(path.c_str(),O_RDONLY);
    struct stat st{};
    if(fd>=0) {
      if(!::fstat(fd,&st)&&st.st_size>0) {
        size_=uint64_t(st.st_size);page_=uint64_t(::getpagesize());
        mapping_=::mmap(nullptr,size_,PROT_READ,MAP_PRIVATE,fd,0);
      }
      ::close(fd);
    }
    auto first=sample();initial_=first.resident;
    thread_=std::thread([this] {
      for(;;) {
        std::string phase;
        {std::unique_lock<std::mutex> lock(mutex_);if(stop_)break;phase=phase_;}
        const auto measured=sample();
        try {callback_(phase,measured);} catch(...) {} // Telemetry must not fail a load.
        // Keep observation near 5% of elapsed load time, with bounded UI refresh gaps.
        const auto pause=std::chrono::milliseconds(std::clamp<int64_t>(int64_t(measured.query_s*20000),1000,5000));
        std::unique_lock<std::mutex> lock(mutex_);
        if(cv_.wait_for(lock,pause,[this]{return stop_;}))break;
      }
    });
  }
  ~LoadMemoryMonitor() {
    {std::lock_guard<std::mutex> lock(mutex_);stop_=true;}cv_.notify_all();
    if(thread_.joinable())thread_.join();
    if(mapping_!=MAP_FAILED)::munmap(mapping_,size_);
  }
  void phase(std::string value) {std::lock_guard<std::mutex> lock(mutex_);phase_=std::move(value);}
  LoadMemoryMonitor(const LoadMemoryMonitor&)=delete;
  LoadMemoryMonitor& operator=(const LoadMemoryMonitor&)=delete;
 private:
  static uint64_t footprint() {
    task_vm_info_data_t info{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count)==KERN_SUCCESS?info.phys_footprint:0;
  }
  Sample sample() {
    Sample s;s.target=size_;s.initial=initial_;s.baseline=baseline_;s.footprint=footprint();
    const auto begin=std::chrono::steady_clock::now();
    if(mapping_!=MAP_FAILED) {
      mach_vm_address_t address=reinterpret_cast<uintptr_t>(mapping_);mach_vm_size_t size=0;
      vm_region_extended_info_data_t info{};mach_msg_type_number_t count=VM_REGION_EXTENDED_INFO_COUNT;
      mach_port_t object=MACH_PORT_NULL;
      const auto result=mach_vm_region(mach_task_self(),&address,&size,VM_REGION_EXTENDED_INFO,
        reinterpret_cast<vm_region_info_t>(&info),&count,&object);
      if(object!=MACH_PORT_NULL)mach_port_deallocate(mach_task_self(),object);
      if(result==KERN_SUCCESS && address==reinterpret_cast<uintptr_t>(mapping_) && size>=size_) {
        s.valid=true;s.resident=std::min<uint64_t>(size_,uint64_t(info.pages_resident)*page_);
      }
    }
    s.query_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    return s;
  }
  Callback callback_;uint64_t size_=0,page_=4096,initial_=0,baseline_=0;
  void* mapping_=MAP_FAILED;std::thread thread_;
  std::mutex mutex_;std::condition_variable cv_;bool stop_=false;
  std::string phase_="Verifica del file modello";
};
