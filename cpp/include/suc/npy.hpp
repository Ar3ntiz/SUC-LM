// Minimal .npy (format 1.0) writer and reader for float32 C-order arrays, for comparisons with numpy.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace suc {

void write_npy_f32(const std::string& path, const std::vector<uint64_t>& shape, const float* data);
void write_raw(const std::string& path, const void* data, std::size_t nbytes);
// Reads a little-endian float32 ('<f4'), C-order .npy file (format 1.0 / 2.0); throws on anything else.
std::vector<float> read_npy_f32(const std::string& path, std::vector<uint64_t>& shape);

}  // namespace suc
