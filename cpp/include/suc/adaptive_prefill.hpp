// Adaptive service prefill: large history blocks, conservative final tail.
#pragma once
#include "suc/inference_engine.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
namespace suc {
inline std::size_t history_prefill_chunk(std::size_t remaining,std::size_t tail=64,std::size_t bulk=1024) {
  if(!tail || bulk<tail)throw std::invalid_argument("Invalid prefill chunk limits");
  return remaining>=bulk?bulk:std::min(tail,remaining);
}
class AdaptivePrefillEngine final:public InferenceEngine {
  std::unique_ptr<InferenceEngine> inner_;
  std::size_t tail_,bulk_,position_=0,checkpoint_position_=0;
 public:
  AdaptivePrefillEngine(std::unique_ptr<InferenceEngine> inner,std::size_t tail=64,std::size_t bulk=1024)
    :inner_(std::move(inner)),tail_(tail),bulk_(bulk) {
    if(!inner_ || !tail || bulk<tail)throw std::invalid_argument("Invalid adaptive prefill engine");
  }
  void reset() override {inner_->reset();position_=checkpoint_position_=0;}
  std::size_t checkpoint_stride() const override {return inner_->checkpoint_stride()?tail_:0;}
  bool save_checkpoint() override {const bool saved=inner_->save_checkpoint();checkpoint_position_=saved?position_:0;return saved;}
  bool restore_checkpoint() override {if(!inner_->restore_checkpoint())return false;position_=checkpoint_position_;return true;}
  void clear_checkpoint() override {inner_->clear_checkpoint();checkpoint_position_=0;}
  uint32_t prefill(const std::vector<uint32_t>& tokens) override {
    if(tokens.empty())throw std::invalid_argument("Empty prefill");
    uint32_t next=0;
    const auto bulk=position_==0?bulk_:tail_; // Warm suffixes keep the conservative profile.
    for(std::size_t i=0;i<tokens.size();) {
      const auto count=history_prefill_chunk(tokens.size()-i,tail_,bulk);
      next=inner_->prefill(std::vector<uint32_t>(tokens.begin()+i,tokens.begin()+i+count));i+=count;position_+=count;
    }
    return next;
  }
  uint32_t step(uint32_t t) override {const auto result=inner_->step(t);++position_;return result;}
  std::vector<float> last_logits() override {return inner_->last_logits();}
  bool generate(uint32_t first,std::size_t count,const std::function<bool(uint32_t)>& sink) override {
    return inner_->generate(first,count,[&](uint32_t token){++position_;return sink(token);});
  }
};
}
