// Token sampling for generation (milestone 4): temperature, top-k, top-p (nucleus) and min-p filtering with a seeded
// RNG, CPU-only, on one row of logits.
//
// Order (as mlx-lm's make_sampler applies its filters to the logits, then mx.random.categorical(logits / temp)):
//   1. temperature <= 0 (or top_k == 1): greedy, the first index of the maximum (= mx.argmax);
//   2. z = logits / temperature (float64 from here on);
//   3. top_k > 0: keep the k largest (ties: lower index first);
//   4. top_p in (0, 1): keep the smallest prefix of the descending order whose probability mass reaches top_p;
//   5. min_p in (0, 1): keep tokens with p >= min_p * p_max;
//   6. draw from the renormalised distribution with a 53-bit uniform from std::mt19937_64(seed).
// The draws are reproducible for a seed but NOT the same as mlx-lm's (MLX's RNG is not reimplemented).
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>

namespace suc {

struct SamplerConfig {
  double temperature = 0.0;   // <= 0: greedy
  std::size_t top_k = 0;      // 0: off
  double top_p = 1.0;         // >= 1: off
  double min_p = 0.0;         // <= 0: off
  uint64_t seed = 0;
  bool greedy() const { return temperature <= 0.0 || top_k == 1; }
};

class Sampler {
 public:
  explicit Sampler(const SamplerConfig& c) : cfg_(c), rng_(c.seed) {}
  const SamplerConfig& config() const { return cfg_; }
  // One token id from logits[0:n] (n > 0; NaN logits are not handled).
  uint32_t sample(const float* logits, std::size_t n);

 private:
  SamplerConfig cfg_;
  std::mt19937_64 rng_;
};

}  // namespace suc
