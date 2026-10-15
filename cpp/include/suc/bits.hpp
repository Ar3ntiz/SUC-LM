// Low-level helpers: SUC1 dtypes, exact fp16/bf16 -> fp32 conversion, and the packed index layout
// `rows_u32le_bitstream` (docs/format.md §6, §7; = suc.quant.packing).
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace suc {

static_assert(std::endian::native == std::endian::little, "SUC1 payloads are little-endian; big-endian hosts are unsupported");

enum class DType { BF16, F16, F32, F64, I8, I16, I32, I64, U8, U16, U32, U64, Bool };

// Parses a format dtype name ("bf16", "f16", ...); throws std::invalid_argument on unknown names.
DType parse_dtype(std::string_view name);
std::string_view dtype_name(DType t);
std::size_t dtype_size(DType t);

// Exact conversions (identical to numpy's float16 -> float32 cast and the reference bf16 bit shift).
inline float bf16_to_f32(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  return std::bit_cast<float>(u);
}

inline float f16_to_f32(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000) << 16;
  const uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t u;
  if (exp == 0x1F) {                 // inf / nan (payload preserved)
    u = sign | 0x7F800000u | (mant << 13);
  } else if (exp != 0) {             // normal
    u = sign | ((exp + 112) << 23) | (mant << 13);
  } else if (mant == 0) {            // signed zero
    u = sign;
  } else {                           // subnormal: renormalize
    int e = 113;
    while (!(mant & 0x400)) { mant <<= 1; --e; }
    u = sign | (uint32_t(e) << 23) | ((mant & 0x3FF) << 13);
  }
  return std::bit_cast<float>(u);
}

// Reads one little-endian scalar of the given dtype at p and casts it to float32 the way numpy's
// astype(np.float32) does (round to nearest even for wide types).
float load_as_f32(const uint8_t* p, DType t);
// Same for integer dtypes -> int64 (throws for floating dtypes).
int64_t load_as_i64(const uint8_t* p, DType t);

inline uint32_t load_u32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

inline std::size_t words_per_row(std::size_t n, unsigned bits) { return (n * bits + 31) / 32; }

// Element j of one packed row (bits in [1, 32]); `row` points at the row's first word (raw bytes,
// so unaligned/mmap pointers are fine). Reads at most the words the element occupies.
inline uint32_t unpack_one(const uint8_t* row, unsigned bits, std::size_t j) {
  const uint64_t p = uint64_t(j) * bits;
  const std::size_t w = std::size_t(p >> 5);
  const unsigned sh = unsigned(p & 31);
  uint64_t v = load_u32(row + 4 * w);
  if (sh + bits > 32) v |= uint64_t(load_u32(row + 4 * (w + 1))) << 32;
  const uint64_t mask = bits == 32 ? 0xFFFFFFFFull : ((1ull << bits) - 1);
  return uint32_t((v >> sh) & mask);
}

// Unpacks n elements of one row into out.
void unpack_row(const uint8_t* row, unsigned bits, std::size_t n, uint32_t* out);

}  // namespace suc
