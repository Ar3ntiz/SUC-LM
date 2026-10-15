#include "suc/bits.hpp"

#include <stdexcept>
#include <string>

namespace suc {

DType parse_dtype(std::string_view n) {
  if (n == "bf16") return DType::BF16;
  if (n == "f16") return DType::F16;
  if (n == "f32") return DType::F32;
  if (n == "f64") return DType::F64;
  if (n == "i8") return DType::I8;
  if (n == "i16") return DType::I16;
  if (n == "i32") return DType::I32;
  if (n == "i64") return DType::I64;
  if (n == "u8") return DType::U8;
  if (n == "u16") return DType::U16;
  if (n == "u32") return DType::U32;
  if (n == "u64") return DType::U64;
  if (n == "bool") return DType::Bool;
  throw std::invalid_argument("unknown SUC1 dtype '" + std::string(n) + "'");
}

std::string_view dtype_name(DType t) {
  switch (t) {
    case DType::BF16: return "bf16";
    case DType::F16: return "f16";
    case DType::F32: return "f32";
    case DType::F64: return "f64";
    case DType::I8: return "i8";
    case DType::I16: return "i16";
    case DType::I32: return "i32";
    case DType::I64: return "i64";
    case DType::U8: return "u8";
    case DType::U16: return "u16";
    case DType::U32: return "u32";
    case DType::U64: return "u64";
    case DType::Bool: return "bool";
  }
  return "?";
}

std::size_t dtype_size(DType t) {
  switch (t) {
    case DType::I8: case DType::U8: case DType::Bool: return 1;
    case DType::BF16: case DType::F16: case DType::I16: case DType::U16: return 2;
    case DType::F32: case DType::I32: case DType::U32: return 4;
    case DType::F64: case DType::I64: case DType::U64: return 8;
  }
  return 0;
}

template <typename T>
static T rd(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof v);
  return v;
}

float load_as_f32(const uint8_t* p, DType t) {
  switch (t) {
    case DType::BF16: return bf16_to_f32(rd<uint16_t>(p));
    case DType::F16: return f16_to_f32(rd<uint16_t>(p));
    case DType::F32: return rd<float>(p);
    case DType::F64: return static_cast<float>(rd<double>(p));
    case DType::I8: return static_cast<float>(rd<int8_t>(p));
    case DType::I16: return static_cast<float>(rd<int16_t>(p));
    case DType::I32: return static_cast<float>(rd<int32_t>(p));
    case DType::I64: return static_cast<float>(rd<int64_t>(p));
    case DType::U8: return static_cast<float>(rd<uint8_t>(p));
    case DType::U16: return static_cast<float>(rd<uint16_t>(p));
    case DType::U32: return static_cast<float>(rd<uint32_t>(p));
    case DType::U64: return static_cast<float>(rd<uint64_t>(p));
    case DType::Bool: return rd<uint8_t>(p) ? 1.0f : 0.0f;
  }
  return 0.0f;
}

int64_t load_as_i64(const uint8_t* p, DType t) {
  switch (t) {
    case DType::I8: return rd<int8_t>(p);
    case DType::I16: return rd<int16_t>(p);
    case DType::I32: return rd<int32_t>(p);
    case DType::I64: return rd<int64_t>(p);
    case DType::U8: return rd<uint8_t>(p);
    case DType::U16: return rd<uint16_t>(p);
    case DType::U32: return rd<uint32_t>(p);
    case DType::U64: return static_cast<int64_t>(rd<uint64_t>(p));
    case DType::Bool: return rd<uint8_t>(p) ? 1 : 0;
    default: throw std::invalid_argument("load_as_i64: floating dtype " + std::string(dtype_name(t)));
  }
}

void unpack_row(const uint8_t* row, unsigned bits, std::size_t n, uint32_t* out) {
  if (bits == 0 || bits > 32) throw std::invalid_argument("unpack_row: bits must be in [1, 32]");
  if (bits == 8) {
    for (std::size_t j = 0; j < n; ++j) out[j] = row[j];
    return;
  }
  if (bits == 16) {
    for (std::size_t j = 0; j < n; ++j) out[j] = rd<uint16_t>(row + 2 * j);
    return;
  }
  if (bits == 32) {
    for (std::size_t j = 0; j < n; ++j) out[j] = rd<uint32_t>(row + 4 * j);
    return;
  }
  // Generic path: a 64-bit window of two words, refilled per word boundary.
  const uint64_t mask = (1ull << bits) - 1;
  uint64_t acc = 0;
  unsigned have = 0;           // valid bits in acc
  std::size_t w = 0;
  for (std::size_t j = 0; j < n; ++j) {
    if (have < bits) {
      acc |= uint64_t(load_u32(row + 4 * w++)) << have;
      have += 32;
    }
    out[j] = uint32_t(acc & mask);
    acc >>= bits;
    have -= bits;
  }
}

}  // namespace suc
