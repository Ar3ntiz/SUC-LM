// Shared inference frontend contract. Checkpoints are optional and owned by a single loaded graph.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>
namespace suc {
class InferenceEngine {
 public:
  virtual ~InferenceEngine() = default;
  virtual void reset() = 0;
  virtual uint32_t prepare(const std::vector<uint32_t>& prompt) { reset(); return prefill(prompt); }
  virtual void invalidate_prefix() {}
  virtual std::size_t cached_tokens() const { return 0; }
  virtual std::size_t checkpoint_stride() const { return 0; }
  virtual bool save_checkpoint() { return false; }
  virtual bool restore_checkpoint() { return false; }
  virtual void clear_checkpoint() {}
  virtual uint32_t prefill(const std::vector<uint32_t>& prompt) = 0;   // argmax of the last prompt position
  virtual uint32_t step(uint32_t token) = 0;                           // argmax of the new position
  virtual std::vector<float> last_logits() = 0;                        // (vocab) of the last position
  // Pipelined greedy decode of up to `steps` tokens after `first` (suc1-generate's loop), each passed to on_token as
  // it completes (false stops); false when this engine has none (the caller then runs step()).
  virtual bool generate(uint32_t, std::size_t, const std::function<bool(uint32_t)>&) { return false; }
};
}
