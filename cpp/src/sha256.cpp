#include "suc/sha256.hpp"

#include <algorithm>
#include <cstring>

#if defined(__ARM_FEATURE_SHA2) || defined(__ARM_FEATURE_CRYPTO)
#include <arm_neon.h>
#define SUC_SHA256_ARM 1
#endif

namespace suc {
namespace {

constexpr uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void blocks_portable(uint32_t h[8], const uint8_t* p, std::size_t nblocks) {
  uint32_t w[64];
  while (nblocks--) {
    for (int t = 0; t < 16; ++t)
      w[t] = uint32_t(p[4 * t]) << 24 | uint32_t(p[4 * t + 1]) << 16 | uint32_t(p[4 * t + 2]) << 8 | p[4 * t + 3];
    for (int t = 16; t < 64; ++t) {
      uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
      uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
      w[t] = w[t - 16] + s0 + w[t - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int t = 0; t < 64; ++t) {
      uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = hh + S1 + ch + K[t] + w[t];
      uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + maj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    p += 64;
  }
}

#ifdef SUC_SHA256_ARM
void blocks_arm(uint32_t h[8], const uint8_t* p, std::size_t nblocks) {
  uint32x4_t abcd = vld1q_u32(h), efgh = vld1q_u32(h + 4);
  while (nblocks--) {
    const uint32x4_t abcd0 = abcd, efgh0 = efgh;
    uint32x4_t msg[4];
    for (int i = 0; i < 4; ++i) msg[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p + 16 * i)));
    for (int i = 0; i < 16; ++i) {   // 16 groups of 4 rounds
      const uint32x4_t wk = vaddq_u32(msg[i & 3], vld1q_u32(K + 4 * i));
      const uint32x4_t prev = abcd;
      abcd = vsha256hq_u32(abcd, efgh, wk);
      efgh = vsha256h2q_u32(efgh, prev, wk);
      if (i < 12)   // schedule W[4(i+4) .. 4(i+4)+3] into the slot just consumed
        msg[i & 3] = vsha256su1q_u32(vsha256su0q_u32(msg[i & 3], msg[(i + 1) & 3]), msg[(i + 2) & 3], msg[(i + 3) & 3]);
    }
    abcd = vaddq_u32(abcd, abcd0);
    efgh = vaddq_u32(efgh, efgh0);
    p += 64;
  }
  vst1q_u32(h, abcd);
  vst1q_u32(h + 4, efgh);
}
#endif

}  // namespace

bool Sha256::hardware_available() {
#ifdef SUC_SHA256_ARM
  return true;
#else
  return false;
#endif
}

void Sha256::reset() {
  static constexpr uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::memcpy(h_, init, sizeof h_);
  buf_len_ = 0;
  total_ = 0;
}

void Sha256::blocks(const uint8_t* p, std::size_t nblocks) {
#ifdef SUC_SHA256_ARM
  if (!portable_) return blocks_arm(h_, p, nblocks);
#endif
  blocks_portable(h_, p, nblocks);
}

void Sha256::update(const void* data, std::size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  total_ += n;
  if (buf_len_) {
    std::size_t take = std::min<std::size_t>(64 - buf_len_, n);
    std::memcpy(buf_ + buf_len_, p, take);
    buf_len_ += take;
    p += take;
    n -= take;
    if (buf_len_ < 64) return;
    blocks(buf_, 1);
    buf_len_ = 0;
  }
  if (n >= 64) {
    blocks(p, n / 64);
    p += n / 64 * 64;
    n %= 64;
  }
  std::memcpy(buf_, p, n);
  buf_len_ = n;
}

std::array<uint8_t, 32> Sha256::finish() {
  const uint64_t bits = total_ * 8;
  uint8_t pad[72] = {0x80};
  std::size_t padlen = (buf_len_ < 56 ? 56 : 120) - buf_len_;
  for (int i = 0; i < 8; ++i) pad[padlen + i] = uint8_t(bits >> (56 - 8 * i));
  update(pad, padlen + 8);
  std::array<uint8_t, 32> out{};
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j) out[4 * i + j] = uint8_t(h_[i] >> (24 - 8 * j));
  reset();
  return out;
}

std::array<uint8_t, 32> sha256(const void* data, std::size_t n) {
  Sha256 h;
  h.update(data, n);
  return h.finish();
}

std::string to_hex(const uint8_t* p, std::size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s(2 * n, '0');
  for (std::size_t i = 0; i < n; ++i) {
    s[2 * i] = d[p[i] >> 4];
    s[2 * i + 1] = d[p[i] & 15];
  }
  return s;
}

}  // namespace suc
