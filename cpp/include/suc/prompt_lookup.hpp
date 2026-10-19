// Draft continuation from previous text only; requires no additional model weights.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>
namespace suc {
inline std::vector<uint32_t> lookup_draft(const std::vector<uint32_t>& history,std::size_t window) {
  if(history.empty()||!window)return {};
  std::vector<uint32_t> out{history.back()};
  for(std::size_t n=std::min<std::size_t>(4,history.size()/2);n>=2;--n){
    for(std::size_t end=history.size()-1;end>=n;--end){
      const auto begin=end-n;
      if(std::equal(history.begin()+begin,history.begin()+end,history.end()-n)){
        for(std::size_t j=end;j<history.size()&&out.size()<window;++j)out.push_back(history[j]);
        return out;
      }
    }
  }
  return out;
}
}
