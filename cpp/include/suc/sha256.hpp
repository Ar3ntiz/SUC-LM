// SHA-256 (FIPS 180-4). Portable implementation plus the ARMv8 SHA-2 instructions when the target
// supports them (__ARM_FEATURE_SHA2, e.g. every Apple Silicon Mac). Both paths are covered by the
// unit tests (cpp/tests/test_main.cpp).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace suc {

class Sha256 {
 public:
  Sha256() { reset(); }
  void reset();
  void update(const void* data, std::size_t n);
  std::array<uint8_t, 32> finish();   // resets the state afterwards

  // Forces the portable block function (tests compare both paths).
  void set_portable(bool on) { portable_ = on; }
  static bool hardware_available();

 private:
  void blocks(const uint8_t* p, std::size_t nblocks);
  uint32_t h_[8];
  uint8_t buf_[64];
  std::size_t buf_len_ = 0;
  uint64_t total_ = 0;
  bool portable_ = false;
};

std::array<uint8_t, 32> sha256(const void* data, std::size_t n);
std::string to_hex(const uint8_t* p, std::size_t n);

}  // namespace suc
