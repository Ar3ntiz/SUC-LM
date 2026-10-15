// Audit F10: int64 boundaries of the request helpers (req::integer / req::seed) under
// -fsanitize=float-cast-overflow -fno-sanitize-recover (CMakeLists.txt / build.sh add the flags when the compiler
// supports them), so any out-of-range double-to-int64 conversion aborts this test instead of silently passing.
// The JSON parser keeps integer literals outside int64 (e.g. 9223372036854775808) as doubles, so 2^63 reaches the
// double branch in every spelling.
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

#include "suc/json.hpp"
#include "suc/request_params.hpp"

using namespace suc;

static int g_fail = 0, g_checks = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_fail;                                                              \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
    }                                                                        \
  } while (0)

static Json J(const char* t) { return Json::parse(t); }

// True iff the call throws std::invalid_argument whose message names `field`.
template <typename F>
static bool rejects(F&& f, const char* field) {
  try {
    f();
  } catch (const std::invalid_argument& e) {
    return std::string(e.what()).find(field) != std::string::npos;
  }
  return false;
}

int main() {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  uint64_t sd = 7;

  // INT64_MAX as an exact JSON integer: accepted (exact-integer branch).
  CHECK(J("9223372036854775807").is_int());
  CHECK(req::seed(J("9223372036854775807"), sd) && sd == uint64_t(kMax));

  // 2^63 in integer / decimal / exponent spellings: all doubles equal to 2^63, all rejected naming `seed`.
  for (const char* bad : {"9223372036854775808", "9223372036854775808.0", "9.223372036854775808e18",
                          "9.223372036854776e18", "9.223372036854776E+18", "92233720368547758.08e2",
                          "9223372036854775807.0", "9223372036854775807.5",     // both round to 2^63
                          "1e19", "1.8446744073709552e19", "1e300"}) {
    CHECK(!J(bad).is_int());
    CHECK(rejects([&] { req::seed(J(bad), sd); }, "seed"));
  }

  // Largest double below 2^63 (2^63 - 1024): accepted in decimal and exponent spellings, exact value.
  for (const char* ok : {"9223372036854774784.0", "9.223372036854774784e18", "9.223372036854775e18"}) {
    sd = 0;
    CHECK(req::seed(J(ok), sd) && sd == uint64_t(9223372036854774784ull));
  }

  // Valid integral doubles.
  CHECK(req::seed(J("256.0"), sd) && sd == 256);
  CHECK(req::seed(J("1e3"), sd) && sd == 1000);
  CHECK(req::seed(J("0.0"), sd) && sd == 0);
  CHECK(req::seed(J("-0.0"), sd) && sd == 0);

  // Fractions and non-finite values: rejected.
  for (const char* bad : {"0.5", "-0.5", "1.000001", "4503599627370495.5", "1e999", "-1e999"})
    CHECK(rejects([&] { req::seed(J(bad), sd); }, "seed"));

  // Documented sentinel -1 (integer and integral double): random seed, `out` untouched. -2 is invalid.
  sd = 7;
  CHECK(!req::seed(J("-1"), sd) && sd == 7);
  CHECK(!req::seed(J("-1.0"), sd) && sd == 7);
  CHECK(rejects([&] { req::seed(J("-2"), sd); }, "seed"));
  CHECK(rejects([&] { req::seed(J("-2.0"), sd); }, "seed"));

  // Lower boundary of the generic helper: -2^63 is representable and exact, anything below is rejected before the
  // conversion (-9223372036854775809 is a double equal to -2^63 after rounding, so it is accepted as -2^63).
  CHECK(req::integer(J("-9223372036854775808"), "v", kMin, kMax) == kMin);
  CHECK(req::integer(J("-9223372036854775808.0"), "v", kMin, kMax) == kMin);
  CHECK(req::integer(J("-9.223372036854775808e18"), "v", kMin, kMax) == kMin);
  for (const char* bad : {"-9.223372036854777856e18", "-1e19", "-1e300"})
    CHECK(rejects([&] { req::integer(J(bad), "v", kMin, kMax); }, "'v'"));
  CHECK(rejects([&] { req::integer(J("9223372036854775808"), "v", kMin, kMax); }, "'v'"));
  CHECK(req::integer(J("9223372036854774784.0"), "v", kMin, kMax) == 9223372036854774784ll);

  // Exact [lo, hi] comparison on the int64 after the conversion (no rounded double bound).
  CHECK(req::integer(J("2147483647.0"), "max_tokens", 1, 2147483647) == 2147483647);
  CHECK(rejects([&] { req::integer(J("2147483648.0"), "max_tokens", 1, 2147483647); }, "max_tokens"));
  CHECK(rejects([&] { req::integer(J("0.0"), "max_tokens", 1, 2147483647); }, "max_tokens"));

  std::printf("%d checks, %d failures\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
