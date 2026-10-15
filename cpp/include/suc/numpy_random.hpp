// Bit-exact reimplementation of the numpy primitives the SUC1 decoder needs to re-derive Hadamard
// signs that are not stored in the file: suc.quant.hadamard.random_signs(n, seed) is
//   (2 * np.random.default_rng(seed).integers(0, 2, size=n) - 1).astype(np.float32)
// i.e. SeedSequence(seed) -> PCG64 (XSL-RR 128/64) -> bounded 32-bit Lemire draws with range 2.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace suc {

class SeedSequence {
 public:
  explicit SeedSequence(uint64_t entropy);   // non-negative Python int < 2**64
  std::vector<uint32_t> generate_state_u32(std::size_t n_words) const;
  std::vector<uint64_t> generate_state_u64(std::size_t n_words) const;

 private:
  uint32_t pool_[4];
};

class Pcg64 {
 public:
  explicit Pcg64(const SeedSequence& ss);    // numpy.random.PCG64(SeedSequence)
  uint64_t next64();
  uint32_t next32();                          // numpy's buffered 32-bit draw (low half first)

 private:
  unsigned __int128 state_ = 0, inc_ = 0;
  bool has32_ = false;
  uint32_t buf32_ = 0;
};

// np.random.default_rng(seed).integers(0, 2, size=n) as 0/1 bytes.
std::vector<uint8_t> numpy_random_bits(std::size_t n, uint64_t seed);
// suc.quant.hadamard.random_signs(n, seed): +-1 float32.
std::vector<float> random_signs(std::size_t n, uint64_t seed);

}  // namespace suc
