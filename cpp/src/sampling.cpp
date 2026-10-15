// Token sampling (see suc/sampling.hpp).
#include "suc/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace suc {

uint32_t Sampler::sample(const float* logits, std::size_t n) {
  if (n == 0) throw std::invalid_argument("sample: no logits");
  if (cfg_.greedy()) return uint32_t(std::max_element(logits, logits + n) - logits);   // first maximum
  const double inv_t = 1.0 / cfg_.temperature;
  std::vector<uint32_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0u);
  auto desc = [&](uint32_t a, uint32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
  std::size_t keep = n;
  if (cfg_.top_k > 0 && cfg_.top_k < n) {
    std::partial_sort(idx.begin(), idx.begin() + cfg_.top_k, idx.end(), desc);
    keep = cfg_.top_k;
  } else {
    std::sort(idx.begin(), idx.end(), desc);
  }
  // probabilities of the kept candidates (descending), relative to the maximum
  const double zmax = double(logits[idx[0]]) * inv_t;
  std::vector<double> p(keep);
  double tot = 0.0;
  for (std::size_t i = 0; i < keep; ++i) tot += p[i] = std::exp(double(logits[idx[i]]) * inv_t - zmax);
  if (cfg_.top_p > 0.0 && cfg_.top_p < 1.0) {
    double acc = 0.0;
    std::size_t k = 0;
    while (k < keep) {
      acc += p[k++] / tot;
      if (acc >= cfg_.top_p) break;
    }
    keep = std::max<std::size_t>(k, 1);
  }
  if (cfg_.min_p > 0.0 && cfg_.min_p < 1.0) {
    std::size_t k = 1;   // p[0] is the maximum (= 1 relative)
    while (k < keep && p[k] >= cfg_.min_p * p[0]) ++k;
    keep = k;
  }
  tot = 0.0;
  for (std::size_t i = 0; i < keep; ++i) tot += p[i];
  const double u = double(rng_() >> 11) * (1.0 / 9007199254740992.0) * tot;   // [0, tot)
  double acc = 0.0;
  for (std::size_t i = 0; i < keep; ++i) {
    acc += p[i];
    if (u < acc) return idx[i];
  }
  return idx[keep - 1];
}

}  // namespace suc
