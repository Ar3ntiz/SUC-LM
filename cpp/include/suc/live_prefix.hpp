// Exact live-prefix reuse, with one optional bounded in-place recurrent prefix checkpoint.
#pragma once
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
namespace suc {
template<class Base> class LivePrefixEngine final : public Base {
  std::unique_ptr<Base> inner_;
  std::vector<uint32_t> fed_, checkpoint_tokens_;
  std::size_t cached_ = 0;
  bool enabled_ = true, checkpoints_ = true;
 public:
  explicit LivePrefixEngine(std::unique_ptr<Base> inner) : inner_(std::move(inner)) {
    if (const char* e=std::getenv("SUC_PREFIX_CACHE");e&&e[0]=='0') enabled_=false;
    if (const char* e=std::getenv("SUC_PREFIX_CHECKPOINTS");e&&e[0]=='0') checkpoints_=false;
  }
  void reset() override { invalidate_prefix(); inner_->reset(); }
  void invalidate_prefix() override { fed_.clear(); checkpoint_tokens_.clear(); cached_=0; inner_->clear_checkpoint(); }
  std::size_t cached_tokens() const override { return cached_; }
  uint32_t prepare(const std::vector<uint32_t>& prompt) override {
    cached_=0;
    const auto extends = [&](const std::vector<uint32_t>& prefix) {
      return !prefix.empty() && prompt.size()>prefix.size() &&
             std::equal(prefix.begin(),prefix.end(),prompt.begin());
    };
    try {
      // Equal/shorter/divergent histories can restore the earlier checkpoint, never truncate hybrid state.
      if (enabled_ && extends(fed_)) cached_=fed_.size();
      else if (enabled_ && checkpoints_ && extends(checkpoint_tokens_) && inner_->restore_checkpoint()) {
        fed_=checkpoint_tokens_; cached_=fed_.size();
      } else reset();
      const auto stride=enabled_ && checkpoints_ ? inner_->checkpoint_stride() : 0;
      // Leave the generation marker and a tokenization margin outside the saved prefix. Align to the
      // model's existing chunk boundaries so splitting a cold prefill does not change projection tiling.
      const auto anchor=stride && prompt.size()>64 ? ((prompt.size()-64)/stride)*stride : 0;
      if (anchor>cached_) {
        prefill(std::vector<uint32_t>(prompt.begin()+cached_,prompt.begin()+anchor));
        checkpoint_tokens_.clear();
        if (inner_->save_checkpoint()) checkpoint_tokens_=fed_;
      }
      return prefill(std::vector<uint32_t>(prompt.begin()+fed_.size(),prompt.end()));
    } catch (...) { invalidate_prefix(); throw; }
  }
  uint32_t prefill(const std::vector<uint32_t>& prompt) override {
    auto t=inner_->prefill(prompt);fed_.insert(fed_.end(),prompt.begin(),prompt.end());return t;
  }
  uint32_t step(uint32_t token) override {auto t=inner_->step(token);fed_.push_back(token);return t;}
  std::vector<float> last_logits() override {return inner_->last_logits();}
  bool generate(uint32_t first,std::size_t steps,const std::function<bool(uint32_t)>& on_token) override {
    uint32_t input=first;
    return inner_->generate(first,steps,[&](uint32_t output){fed_.push_back(input);input=output;return on_token(output);});
  }
};
}
