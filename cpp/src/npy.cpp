#include "suc/npy.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace suc {

void write_raw(const std::string& path, const void* data, std::size_t nbytes) {
  std::FILE* fp = std::fopen(path.c_str(), "wb");
  if (!fp) throw std::runtime_error("cannot write " + path);
  const bool ok = nbytes == 0 || std::fwrite(data, 1, nbytes, fp) == nbytes;
  if (std::fclose(fp) != 0 || !ok) throw std::runtime_error("write failed: " + path);
}

void write_npy_f32(const std::string& path, const std::vector<uint64_t>& shape, const float* data) {
  std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
  uint64_t n = 1;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    dict += std::to_string(shape[i]);
    if (shape.size() == 1 || i + 1 < shape.size()) dict += ",";
    if (i + 1 < shape.size()) dict += " ";
    n *= shape[i];
  }
  dict += "), }";
  // magic(6) + version(2) + header_len(2) + dict + padding + '\n' is a multiple of 64
  std::size_t total = 10 + dict.size() + 1;
  dict.append((64 - total % 64) % 64, ' ');
  dict += '\n';
  std::string out = "\x93NUMPY";
  out += char(1);
  out += char(0);
  const uint16_t hl = uint16_t(dict.size());
  out += char(hl & 0xFF);
  out += char(hl >> 8);
  out += dict;
  std::FILE* fp = std::fopen(path.c_str(), "wb");
  if (!fp) throw std::runtime_error("cannot write " + path);
  bool ok = std::fwrite(out.data(), 1, out.size(), fp) == out.size();
  ok = ok && (n == 0 || std::fwrite(data, sizeof(float), n, fp) == n);
  if (std::fclose(fp) != 0 || !ok) throw std::runtime_error("write failed: " + path);
}

std::vector<float> read_npy_f32(const std::string& path, std::vector<uint64_t>& shape) {
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) throw std::runtime_error("cannot read " + path);
  auto fail = [&](const char* msg) {
    std::fclose(fp);
    throw std::runtime_error(path + ": " + msg);
  };
  unsigned char pre[12];
  if (std::fread(pre, 1, 10, fp) != 10 || std::memcmp(pre, "\x93NUMPY", 6) != 0) fail("not a .npy file");
  std::size_t hl = pre[8] | (std::size_t(pre[9]) << 8);
  if (pre[6] >= 2) {
    if (std::fread(pre + 10, 1, 2, fp) != 2) fail("truncated header");
    hl |= (std::size_t(pre[10]) << 16) | (std::size_t(pre[11]) << 24);
  } else if (pre[6] != 1) {
    fail("unsupported .npy version");
  }
  std::string dict(hl, '\0');
  if (std::fread(dict.data(), 1, hl, fp) != hl) fail("truncated header");
  if (dict.find("'descr': '<f4'") == std::string::npos) fail("dtype must be '<f4' (float32 little-endian)");
  if (dict.find("'fortran_order': False") == std::string::npos) fail("Fortran order is not supported");
  const std::size_t a = dict.find("'shape': ("), b = a == std::string::npos ? a : dict.find(')', a);
  if (b == std::string::npos) fail("no shape");
  shape.clear();
  uint64_t n = 1;
  for (std::size_t i = a + 10; i < b;) {
    while (i < b && (dict[i] == ' ' || dict[i] == ',')) ++i;
    if (i >= b) break;
    uint64_t d = 0;
    while (i < b && dict[i] >= '0' && dict[i] <= '9') d = d * 10 + uint64_t(dict[i++] - '0');
    shape.push_back(d);
    n *= d;
  }
  std::vector<float> out(n);
  if (n && std::fread(out.data(), sizeof(float), n, fp) != n) fail("truncated data");
  std::fclose(fp);
  return out;
}

}  // namespace suc
