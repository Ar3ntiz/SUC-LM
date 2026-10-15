#include "suc/hadamard.hpp"

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "suc/numpy_random.hpp"
#include "suc/parallel.hpp"

namespace suc {

namespace {

// MLX's Hadamard tables (mlx/backend/common/hadamard.h, from http://neilsloane.com/hadamard/);
// suc.quant.hadamard._h_m(m) == sign(table) * float32(1/sqrt(m)) element for element.
constexpr const char* kH12 =
    "+-++++++++++" "--+-+-+-+-+-" "+++-++----++" "+---+--+-++-" "+++++-++----" "+-+---+--+-+"
    "++--+++-++--" "+--++---+--+" "++----+++-++" "+--+-++---+-" "++++----+++-" "+-+--+-++---";

constexpr const char* kH20 =
    "+----+----++--++-++-" "-+----+---+++---+-++" "--+----+---+++-+-+-+" "---+----+---+++++-+-"
    "----+----++--++-++-+" "-+++++-----+--+++--+" "+-+++-+---+-+--+++--" "++-++--+---+-+--+++-"
    "+++-+---+---+-+--+++" "++++-----++--+-+--++" "--++-+-++-+-----++++" "---++-+-++-+---+-+++"
    "+---++-+-+--+--++-++" "++---++-+----+-+++-+" "-++---++-+----+++++-" "-+--+--++-+----+----"
    "+-+-----++-+----+---" "-+-+-+---+--+----+--" "--+-+++------+----+-" "+--+--++------+----+";

constexpr const char* kH28 =
    "+------++----++-+--+-+--++--" "-+-----+++-----+-+--+-+--++-" "--+-----+++---+-+-+----+--++"
    "---+-----+++---+-+-+-+--+--+" "----+-----+++---+-+-+++--+--" "-----+-----++++--+-+--++--+-"
    "------++----++-+--+-+--++--+" "--++++-+-------++--+++-+--+-" "---++++-+-----+-++--+-+-+--+"
    "+---+++--+----++-++--+-+-+--" "++---++---+----++-++--+-+-+-" "+++---+----+----++-++--+-+-+"
    "++++--------+-+--++-++--+-+-" "-++++--------+++--++--+--+-+" "-+-++-++--++--+--------++++-"
    "+-+-++--+--++--+--------++++" "-+-+-++--+--++--+----+---+++" "+-+-+-++--+--+---+---++---++"
    "++-+-+-++--+------+--+++---+" "-++-+-+-++--+------+-++++---" "+-++-+---++--+------+-++++--"
    "-++--++-+-++-+++----++------" "+-++--++-+-++-+++-----+-----" "++-++---+-+-++-+++-----+----"
    "-++-++-+-+-+-+--+++-----+---" "--++-++++-+-+----+++-----+--" "+--++-+-++-+-+----+++-----+-"
    "++--++-+-++-+-+----++------+";

// m = 17 (no Hadamard matrix of that order exists; 17408 = 17 * 2^10 is the Qwen3.5 / Qwen3.8-27B MLP width): the
// orthonormal discrete Hartley matrix (cos + sin)(2 pi j k / 17) / sqrt(17), symmetric, float32 row-major --
// bit-identical to suc.quant.hadamard._H17_HEX.
constexpr float kH17[17 * 17] = {
    0x1.f0b6840000000p-3f, 0x1.f0b6840000000p-3f, 0x1.f0b6840000000p-3f, 0x1.f0b6840000000p-3f,
    0x1.f0b6840000000p-3f, 0x1.f0b6840000000p-3f, 0x1.f0b6840000000p-3f, 0x1.f0b6840000000p-3f,
    0x1.f0b6840000000p-3f, 0x1.f0b6840000000p-3f, 0x1.f0b6840000000p-3f, 0x1.f0b6840000000p-3f,
    0x1.f0b6840000000p-3f, 0x1.f0b6840000000p-3f, 0x1.f0b6840000000p-3f, 0x1.f0b6840000000p-3f,
    0x1.f0b6840000000p-3f, 0x1.f0b6840000000p-3f, 0x1.414d5e0000000p-2f, 0x1.5edab60000000p-2f,
    0x1.4d05880000000p-2f, 0x1.0e36660000000p-2f, 0x1.55d1be0000000p-3f, 0x1.84323e0000000p-5f,
    -0x1.41a8080000000p-4f, -0x1.8cfc160000000p-3f, -0x1.21c35c0000000p-2f, -0x1.55e6580000000p-2f,
    -0x1.5bdc680000000p-2f, -0x1.32d7740000000p-2f, -0x1.c0c3600000000p-3f, -0x1.be786a0000000p-4f,
    0x1.03891c0000000p-6f, 0x1.1bbcd40000000p-3f, 0x1.f0b6840000000p-3f, 0x1.5edab60000000p-2f,
    0x1.0e36660000000p-2f, 0x1.84323e0000000p-5f, -0x1.8cfc160000000p-3f, -0x1.55e6580000000p-2f,
    -0x1.32d7740000000p-2f, -0x1.be786a0000000p-4f, 0x1.1bbcd40000000p-3f, 0x1.414d5e0000000p-2f,
    0x1.4d05880000000p-2f, 0x1.55d1be0000000p-3f, -0x1.41a8080000000p-4f, -0x1.21c35c0000000p-2f,
    -0x1.5bdc680000000p-2f, -0x1.c0c3600000000p-3f, 0x1.03891c0000000p-6f, 0x1.f0b6840000000p-3f,
    0x1.4d05880000000p-2f, 0x1.84323e0000000p-5f, -0x1.21c35c0000000p-2f, -0x1.32d7740000000p-2f,
    0x1.03891c0000000p-6f, 0x1.414d5e0000000p-2f, 0x1.0e36660000000p-2f, -0x1.41a8080000000p-4f,
    -0x1.55e6580000000p-2f, -0x1.c0c3600000000p-3f, 0x1.1bbcd40000000p-3f, 0x1.5edab60000000p-2f,
    0x1.55d1be0000000p-3f, -0x1.8cfc160000000p-3f, -0x1.5bdc680000000p-2f, -0x1.be786a0000000p-4f,
    0x1.f0b6840000000p-3f, 0x1.0e36660000000p-2f, -0x1.8cfc160000000p-3f, -0x1.32d7740000000p-2f,
    0x1.1bbcd40000000p-3f, 0x1.4d05880000000p-2f, -0x1.41a8080000000p-4f, -0x1.5bdc680000000p-2f,
    0x1.03891c0000000p-6f, 0x1.5edab60000000p-2f, 0x1.84323e0000000p-5f, -0x1.55e6580000000p-2f,
    -0x1.be786a0000000p-4f, 0x1.414d5e0000000p-2f, 0x1.55d1be0000000p-3f, -0x1.21c35c0000000p-2f,
    -0x1.c0c3600000000p-3f, 0x1.f0b6840000000p-3f, 0x1.55d1be0000000p-3f, -0x1.55e6580000000p-2f,
    0x1.03891c0000000p-6f, 0x1.4d05880000000p-2f, -0x1.8cfc160000000p-3f, -0x1.c0c3600000000p-3f,
    0x1.414d5e0000000p-2f, 0x1.84323e0000000p-5f, -0x1.5bdc680000000p-2f, 0x1.1bbcd40000000p-3f,
    0x1.0e36660000000p-2f, -0x1.21c35c0000000p-2f, -0x1.be786a0000000p-4f, 0x1.5edab60000000p-2f,
    -0x1.41a8080000000p-4f, -0x1.32d7740000000p-2f, 0x1.f0b6840000000p-3f, 0x1.84323e0000000p-5f,
    -0x1.32d7740000000p-2f, 0x1.414d5e0000000p-2f, -0x1.41a8080000000p-4f, -0x1.c0c3600000000p-3f,
    0x1.5edab60000000p-2f, -0x1.8cfc160000000p-3f, -0x1.be786a0000000p-4f, 0x1.4d05880000000p-2f,
    -0x1.21c35c0000000p-2f, 0x1.03891c0000000p-6f, 0x1.0e36660000000p-2f, -0x1.55e6580000000p-2f,
    0x1.1bbcd40000000p-3f, 0x1.55d1be0000000p-3f, -0x1.5bdc680000000p-2f, 0x1.f0b6840000000p-3f,
    -0x1.41a8080000000p-4f, -0x1.be786a0000000p-4f, 0x1.0e36660000000p-2f, -0x1.5bdc680000000p-2f,
    0x1.414d5e0000000p-2f, -0x1.8cfc160000000p-3f, 0x1.03891c0000000p-6f, 0x1.55d1be0000000p-3f,
    -0x1.32d7740000000p-2f, 0x1.5edab60000000p-2f, -0x1.21c35c0000000p-2f, 0x1.1bbcd40000000p-3f,
    0x1.84323e0000000p-5f, -0x1.c0c3600000000p-3f, 0x1.4d05880000000p-2f, -0x1.55e6580000000p-2f,
    0x1.f0b6840000000p-3f, -0x1.8cfc160000000p-3f, 0x1.1bbcd40000000p-3f, -0x1.41a8080000000p-4f,
    0x1.03891c0000000p-6f, 0x1.84323e0000000p-5f, -0x1.be786a0000000p-4f, 0x1.55d1be0000000p-3f,
    -0x1.c0c3600000000p-3f, 0x1.0e36660000000p-2f, -0x1.32d7740000000p-2f, 0x1.4d05880000000p-2f,
    -0x1.5bdc680000000p-2f, 0x1.5edab60000000p-2f, -0x1.55e6580000000p-2f, 0x1.414d5e0000000p-2f,
    -0x1.21c35c0000000p-2f, 0x1.f0b6840000000p-3f, -0x1.21c35c0000000p-2f, 0x1.414d5e0000000p-2f,
    -0x1.55e6580000000p-2f, 0x1.5edab60000000p-2f, -0x1.5bdc680000000p-2f, 0x1.4d05880000000p-2f,
    -0x1.32d7740000000p-2f, 0x1.0e36660000000p-2f, -0x1.c0c3600000000p-3f, 0x1.55d1be0000000p-3f,
    -0x1.be786a0000000p-4f, 0x1.84323e0000000p-5f, 0x1.03891c0000000p-6f, -0x1.41a8080000000p-4f,
    0x1.1bbcd40000000p-3f, -0x1.8cfc160000000p-3f, 0x1.f0b6840000000p-3f, -0x1.55e6580000000p-2f,
    0x1.4d05880000000p-2f, -0x1.c0c3600000000p-3f, 0x1.84323e0000000p-5f, 0x1.1bbcd40000000p-3f,
    -0x1.21c35c0000000p-2f, 0x1.5edab60000000p-2f, -0x1.32d7740000000p-2f, 0x1.55d1be0000000p-3f,
    0x1.03891c0000000p-6f, -0x1.8cfc160000000p-3f, 0x1.414d5e0000000p-2f, -0x1.5bdc680000000p-2f,
    0x1.0e36660000000p-2f, -0x1.be786a0000000p-4f, -0x1.41a8080000000p-4f, 0x1.f0b6840000000p-3f,
    -0x1.5bdc680000000p-2f, 0x1.55d1be0000000p-3f, 0x1.1bbcd40000000p-3f, -0x1.55e6580000000p-2f,
    0x1.0e36660000000p-2f, 0x1.03891c0000000p-6f, -0x1.21c35c0000000p-2f, 0x1.4d05880000000p-2f,
    -0x1.be786a0000000p-4f, -0x1.8cfc160000000p-3f, 0x1.5edab60000000p-2f, -0x1.c0c3600000000p-3f,
    -0x1.41a8080000000p-4f, 0x1.414d5e0000000p-2f, -0x1.32d7740000000p-2f, 0x1.84323e0000000p-5f,
    0x1.f0b6840000000p-3f, -0x1.32d7740000000p-2f, -0x1.41a8080000000p-4f, 0x1.5edab60000000p-2f,
    -0x1.be786a0000000p-4f, -0x1.21c35c0000000p-2f, 0x1.0e36660000000p-2f, 0x1.1bbcd40000000p-3f,
    -0x1.5bdc680000000p-2f, 0x1.84323e0000000p-5f, 0x1.414d5e0000000p-2f, -0x1.c0c3600000000p-3f,
    -0x1.8cfc160000000p-3f, 0x1.4d05880000000p-2f, 0x1.03891c0000000p-6f, -0x1.55e6580000000p-2f,
    0x1.55d1be0000000p-3f, 0x1.f0b6840000000p-3f, -0x1.c0c3600000000p-3f, -0x1.21c35c0000000p-2f,
    0x1.55d1be0000000p-3f, 0x1.414d5e0000000p-2f, -0x1.be786a0000000p-4f, -0x1.55e6580000000p-2f,
    0x1.84323e0000000p-5f, 0x1.5edab60000000p-2f, 0x1.03891c0000000p-6f, -0x1.5bdc680000000p-2f,
    -0x1.41a8080000000p-4f, 0x1.4d05880000000p-2f, 0x1.1bbcd40000000p-3f, -0x1.32d7740000000p-2f,
    -0x1.8cfc160000000p-3f, 0x1.0e36660000000p-2f, 0x1.f0b6840000000p-3f, -0x1.be786a0000000p-4f,
    -0x1.5bdc680000000p-2f, -0x1.8cfc160000000p-3f, 0x1.55d1be0000000p-3f, 0x1.5edab60000000p-2f,
    0x1.1bbcd40000000p-3f, -0x1.c0c3600000000p-3f, -0x1.55e6580000000p-2f, -0x1.41a8080000000p-4f,
    0x1.0e36660000000p-2f, 0x1.414d5e0000000p-2f, 0x1.03891c0000000p-6f, -0x1.32d7740000000p-2f,
    -0x1.21c35c0000000p-2f, 0x1.84323e0000000p-5f, 0x1.4d05880000000p-2f, 0x1.f0b6840000000p-3f,
    0x1.03891c0000000p-6f, -0x1.c0c3600000000p-3f, -0x1.5bdc680000000p-2f, -0x1.21c35c0000000p-2f,
    -0x1.41a8080000000p-4f, 0x1.55d1be0000000p-3f, 0x1.4d05880000000p-2f, 0x1.414d5e0000000p-2f,
    0x1.1bbcd40000000p-3f, -0x1.be786a0000000p-4f, -0x1.32d7740000000p-2f, -0x1.55e6580000000p-2f,
    -0x1.8cfc160000000p-3f, 0x1.84323e0000000p-5f, 0x1.0e36660000000p-2f, 0x1.5edab60000000p-2f,
    0x1.f0b6840000000p-3f, 0x1.1bbcd40000000p-3f, 0x1.03891c0000000p-6f, -0x1.be786a0000000p-4f,
    -0x1.c0c3600000000p-3f, -0x1.32d7740000000p-2f, -0x1.5bdc680000000p-2f, -0x1.55e6580000000p-2f,
    -0x1.21c35c0000000p-2f, -0x1.8cfc160000000p-3f, -0x1.41a8080000000p-4f, 0x1.84323e0000000p-5f,
    0x1.55d1be0000000p-3f, 0x1.0e36660000000p-2f, 0x1.4d05880000000p-2f, 0x1.5edab60000000p-2f,
    0x1.414d5e0000000p-2f};

constexpr std::size_t kMaxPow2 = 8192;   // mx.hadamard_transform float32 limit (suc.quant.hadamard._MAX_POW2)

struct Factor {
  std::size_t m = 0, p = 0;
};

Factor factor(std::size_t n) {
  for (std::size_t m : {1u, 12u, 20u, 28u, 17u})
    if (n > 0 && n % m == 0) {
      std::size_t p = n / m;
      if ((p & (p - 1)) == 0) return {m, p};
    }
  return {};
}

// H_m as float32 (row-major), plain and transposed.
struct HmTable {
  std::vector<float> h, ht;
};

const HmTable& hm_table(std::size_t m) {
  static std::once_flag once;
  static HmTable t12, t20, t28, t17;
  std::call_once(once, [] {
    auto build = [](HmTable& t, const char* s, std::size_t m) {
      const float c = static_cast<float>(1.0 / std::sqrt(static_cast<double>(m)));
      t.h.resize(m * m);
      t.ht.resize(m * m);
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
          float v = s[i * m + j] == '+' ? c : -c;
          t.h[i * m + j] = v;
          t.ht[j * m + i] = v;
        }
    };
    build(t12, kH12, 12);
    build(t20, kH20, 20);
    build(t28, kH28, 28);
    t17.h.assign(kH17, kH17 + 17 * 17);
    t17.ht.resize(17 * 17);
    for (std::size_t i = 0; i < 17; ++i)
      for (std::size_t j = 0; j < 17; ++j) t17.ht[j * 17 + i] = kH17[i * 17 + j];
  });
  if (m == 12) return t12;
  if (m == 20) return t20;
  if (m == 28) return t28;
  if (m == 17) return t17;
  throw std::logic_error("no Hadamard table for m=" + std::to_string(m));
}

inline float inv_sqrt(std::size_t p) { return static_cast<float>(1.0 / std::sqrt(static_cast<double>(p))); }

// Unscaled in-place FWHT, butterfly distance increasing (h = 1, 2, 4, ...).
void fwht(float* x, std::size_t n) {
  for (std::size_t h = 1; h < n; h <<= 1)
    for (std::size_t i = 0; i < n; i += 2 * h)
      for (std::size_t j = i; j < i + h; ++j) {
        const float a = x[j], b = x[j + h];
        x[j] = a + b;
        x[j + h] = a - b;
      }
}

// suc.quant.hadamard._pow2 on one contiguous vector of length p (a power of two).
void pow2(float* x, std::size_t p, float* scratch) {
  if (p <= kMaxPow2) {
    fwht(x, p);
    const float s = inv_sqrt(p);
    for (std::size_t i = 0; i < p; ++i) x[i] = x[i] * s;
    return;
  }
  const std::size_t q = p / kMaxPow2;
  for (std::size_t b = 0; b < q; ++b) pow2(x + b * kMaxPow2, kMaxPow2, scratch);
  // strided factor: for every position c inside a block, transform (x[c], x[c + 8192], ...)
  float* v = scratch;
  float* inner = scratch + q;
  for (std::size_t c = 0; c < kMaxPow2; ++c) {
    for (std::size_t j = 0; j < q; ++j) v[j] = x[j * kMaxPow2 + c];
    pow2(v, q, inner);
    for (std::size_t j = 0; j < q; ++j) x[j * kMaxPow2 + c] = v[j];
  }
}

}  // namespace

bool hadamard_supported(std::size_t n) { return factor(n).m != 0; }

HadamardFactor hadamard_factor(std::size_t n) {
  const Factor f = factor(n);
  return {f.m, f.p};
}

std::vector<float> hadamard_hm(std::size_t m) {
  if (m == 1) return {1.0f};
  return hm_table(m).h;
}

void hadamard_transform(float* x, std::size_t n, bool transpose, float* scratch) {
  const Factor f = factor(n);
  if (!f.m) throw std::invalid_argument("unsupported Hadamard size " + std::to_string(n));
  if (f.m == 1) return pow2(x, n, scratch);
  for (std::size_t k = 0; k < f.m; ++k) pow2(x + k * f.p, f.p, scratch);
  const HmTable& t = hm_table(f.m);
  const float* h = transpose ? t.ht.data() : t.h.data();
  // out[i, c] = sum_k h[i, k] * y[k, c] as a sequential fma chain (scratch holds the output)
  for (std::size_t i = 0; i < f.m; ++i) {
    float* o = scratch + i * f.p;
    for (std::size_t c = 0; c < f.p; ++c) o[c] = 0.0f;
    for (std::size_t k = 0; k < f.m; ++k) {
      const float hk = h[i * f.m + k];
      const float* y = x + k * f.p;
      for (std::size_t c = 0; c < f.p; ++c) o[c] = std::fma(hk, y[c], o[c]);
    }
  }
  for (std::size_t i = 0; i < n; ++i) x[i] = scratch[i];
}

void rht_inverse_rows(float* x, std::size_t rows, std::size_t n, uint64_t seed) {
  if (!hadamard_supported(n)) throw std::invalid_argument("unsupported Hadamard size " + std::to_string(n));
  const std::vector<float> s = random_signs(n, seed);
  parallel_for(rows, 8, [&](std::size_t r) {
    thread_local std::vector<float> scratch;
    if (scratch.size() < 2 * n) scratch.resize(2 * n);
    float* row = x + r * n;
    hadamard_transform(row, n, /*transpose=*/true, scratch.data());
    for (std::size_t j = 0; j < n; ++j) row[j] = row[j] * s[j];
  });
}

}  // namespace suc
