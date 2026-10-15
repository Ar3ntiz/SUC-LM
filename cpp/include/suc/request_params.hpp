// Validation of numeric HTTP request fields (suc1-server, audit F06): JSON type, sign, integrality, finiteness and
// range are checked BEFORE any narrowing conversion, so an invalid request can never silently change meaning
// (max_tokens -1 -> SIZE_MAX, 1.5 -> 1, token id 2^32 + 1 -> 1). Every function throws std::invalid_argument naming
// the field; the server maps it to HTTP 400.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "suc/json.hpp"

namespace suc::req {

[[noreturn]] inline void reject(const char* name, const std::string& why) {
  throw std::invalid_argument(std::string("'") + name + "' " + why);
}

// A finite JSON number in [lo, hi].
inline double number(const Json& v, const char* name, double lo, double hi) {
  if (!v.is_number()) reject(name, "must be a number");
  const double x = v.is_int() ? double(v.as_int()) : v.as_double();
  if (!std::isfinite(x)) reject(name, "must be finite");
  if (x < lo || x > hi) reject(name, "must be in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
  return x;
}

// An integral JSON number in [lo, hi] (an integral double such as 256.0 is accepted, 1.5 is not).
inline int64_t integer(const Json& v, const char* name, int64_t lo, int64_t hi) {
  if (!v.is_number()) reject(name, "must be an integer");
  if (v.is_int()) {
    const int64_t x = v.as_int();
    if (x < lo || x > hi) reject(name, "must be in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    return x;
  }
  const double d = v.as_double();
  if (!std::isfinite(d) || std::floor(d) != d) reject(name, "must be an integer");
  // Audit F10: range-check against the exact int64 domain [-2^63, 2^63) BEFORE the conversion (an out-of-range
  // double-to-int64 cast is undefined). double(INT64_MAX) rounds up to 2^63, so it can never serve as an inclusive
  // bound; both limits here are exact powers of two. The [lo, hi] check is then done exactly on the int64 value.
  constexpr double kTwo63 = 9223372036854775808.0;   // 2^63, exactly representable
  static_assert(kTwo63 == double(uint64_t(1) << 63));
  const std::string range = "must be in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]";
  if (d < -kTwo63 || d >= kTwo63) reject(name, range);
  const int64_t x = int64_t(d);
  if (x < lo || x > hi) reject(name, range);
  return x;
}

// A generation length (max_tokens / max_completion_tokens): integer in [1, 2^31 - 1]; the context still caps it.
inline std::size_t length(const Json& v, const char* name) {
  return std::size_t(integer(v, name, 1, std::numeric_limits<int32_t>::max()));
}

// top_k: integer >= -1; -1 and 0 are the documented "off" sentinels (vLLM convention) and map to 0.
inline std::size_t top_k(const Json& v) {
  const int64_t k = integer(v, "top_k", -1, std::numeric_limits<int32_t>::max());
  return k <= 0 ? 0 : std::size_t(k);
}

// seed: integer in [0, 2^63 - 1], or -1 = draw a random seed (returns false then).
inline bool seed(const Json& v, uint64_t& out) {
  const int64_t s = integer(v, "seed", -1, std::numeric_limits<int64_t>::max());
  if (s < 0) return false;
  out = uint64_t(s);
  return true;
}

// One prompt token id: integer in [0, vocab).
inline uint32_t token_id(const Json& v, uint64_t vocab) {
  if (!v.is_int()) reject("prompt", "token ids must be integers");
  const int64_t t = v.as_int();
  if (t < 0 || uint64_t(t) >= vocab || uint64_t(t) > std::numeric_limits<uint32_t>::max())
    reject("prompt", "token id " + std::to_string(t) + " out of range (vocab " + std::to_string(vocab) + ")");
  return uint32_t(t);
}

}  // namespace suc::req
