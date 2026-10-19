// CPU-only reproducible legacy/linear sampling comparison, no model speed claim.
#include "suc/sampling.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
int main() {
  std::mt19937 rng(123); std::normal_distribution<float> d(0, 3);
  std::vector<float> logits(151936); for (auto& x : logits) x = d(rng);
  uint32_t checksum = 0;
  for (bool legacy : {true, false}) {
    suc::SamplerConfig c; c.temperature = .7; c.seed = 42; c.legacy_order = legacy;
    suc::Sampler sampler(c); std::vector<double> times;
    for (int pass = 0; pass < 5; ++pass) {
      const auto t = std::chrono::steady_clock::now();
      for (int i = 0; i < 50; ++i) checksum ^= sampler.sample(logits.data(), logits.size());
      times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t).count()/50);
    }
    std::sort(times.begin(), times.end());
    std::printf("{\"legacy\":%s,\"vocab\":151936,\"temperature\":0.7,\"top_k\":0,\"top_p\":1,\"seed\":42,\"median_ms\":%.6f}\n",legacy?"true":"false",times[2]);
  }
  std::fprintf(stderr,"checksum=%u\n",checksum);
}
