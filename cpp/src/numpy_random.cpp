#include "suc/numpy_random.hpp"

namespace suc {
namespace {

// numpy/random/bit_generator.pyx constants (pool_size 4, 32-bit words)
constexpr uint32_t INIT_A = 0x43b0d7e5, MULT_A = 0x931e8875;
constexpr uint32_t INIT_B = 0x8b51f9dd, MULT_B = 0x58f38ded;
constexpr uint32_t MIX_MULT_L = 0xca01f9dd, MIX_MULT_R = 0x4973f715;
constexpr unsigned XSHIFT = 16;

uint32_t hashmix(uint32_t value, uint32_t& hash_const) {
  value ^= hash_const;
  hash_const *= MULT_A;
  value *= hash_const;
  value ^= value >> XSHIFT;
  return value;
}

uint32_t mix(uint32_t x, uint32_t y) {
  uint32_t r = MIX_MULT_L * x - MIX_MULT_R * y;
  r ^= r >> XSHIFT;
  return r;
}

}  // namespace

SeedSequence::SeedSequence(uint64_t entropy) {
  // _coerce_to_uint32_array(int): little-endian 32-bit words, 0 -> [0]
  uint32_t ent[2];
  std::size_t n_ent = 0;
  if (entropy == 0) {
    ent[n_ent++] = 0;
  } else {
    while (entropy) {
      ent[n_ent++] = uint32_t(entropy & 0xFFFFFFFFu);
      entropy >>= 32;
    }
  }
  uint32_t hc = INIT_A;
  for (std::size_t i = 0; i < 4; ++i) pool_[i] = hashmix(i < n_ent ? ent[i] : 0u, hc);
  for (std::size_t s = 0; s < 4; ++s)
    for (std::size_t d = 0; d < 4; ++d)
      if (s != d) pool_[d] = mix(pool_[d], hashmix(pool_[s], hc));
  // entropy longer than the pool (impossible for <= 2 words) would be mixed in here
}

std::vector<uint32_t> SeedSequence::generate_state_u32(std::size_t n_words) const {
  std::vector<uint32_t> out(n_words);
  uint32_t hc = INIT_B;
  for (std::size_t i = 0; i < n_words; ++i) {
    uint32_t v = pool_[i % 4];
    v ^= hc;
    hc *= MULT_B;
    v *= hc;
    v ^= v >> XSHIFT;
    out[i] = v;
  }
  return out;
}

std::vector<uint64_t> SeedSequence::generate_state_u64(std::size_t n_words) const {
  auto w = generate_state_u32(2 * n_words);
  std::vector<uint64_t> out(n_words);
  for (std::size_t i = 0; i < n_words; ++i) out[i] = uint64_t(w[2 * i]) | (uint64_t(w[2 * i + 1]) << 32);
  return out;
}

static constexpr unsigned __int128 kPcgMult =
    (static_cast<unsigned __int128>(0x2360ED051FC65DA4ull) << 64) | 0x4385DF649FCCF645ull;

Pcg64::Pcg64(const SeedSequence& ss) {
  auto v = ss.generate_state_u64(4);
  const unsigned __int128 initstate = (static_cast<unsigned __int128>(v[0]) << 64) | v[1];
  const unsigned __int128 initseq = (static_cast<unsigned __int128>(v[2]) << 64) | v[3];
  state_ = 0;
  inc_ = (initseq << 1) | 1;
  state_ = state_ * kPcgMult + inc_;
  state_ += initstate;
  state_ = state_ * kPcgMult + inc_;
}

uint64_t Pcg64::next64() {
  state_ = state_ * kPcgMult + inc_;
  const uint64_t hi = uint64_t(state_ >> 64), lo = uint64_t(state_);
  const unsigned rot = unsigned(state_ >> 122);
  const uint64_t x = hi ^ lo;
  return (x >> rot) | (x << ((64 - rot) & 63));
}

uint32_t Pcg64::next32() {
  if (has32_) {
    has32_ = false;
    return buf32_;
  }
  const uint64_t n = next64();
  has32_ = true;
  buf32_ = uint32_t(n >> 32);
  return uint32_t(n & 0xFFFFFFFFu);
}

std::vector<uint8_t> numpy_random_bits(std::size_t n, uint64_t seed) {
  // Generator.integers(0, 2, dtype=int64) -> random_bounded_uint64_fill(off=0, rng=1, use_masked=False)
  // -> buffered_bounded_lemire_uint32(rng=1): m = next32 * 2, threshold (2^32-2) % 2 = 0, result m >> 32.
  Pcg64 g{SeedSequence(seed)};
  std::vector<uint8_t> out(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = uint8_t((uint64_t(g.next32()) * 2u) >> 32);
  return out;
}

std::vector<float> random_signs(std::size_t n, uint64_t seed) {
  auto b = numpy_random_bits(n, seed);
  std::vector<float> s(n);
  for (std::size_t i = 0; i < n; ++i) s[i] = b[i] ? 1.0f : -1.0f;
  return s;
}

}  // namespace suc
