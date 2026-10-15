// C++ reference decoder: SUC1 tensor -> dense float32, with the exact float32 operation order of the
// Python reference decoder SucModel.decode_tensor (suc/io/suc1.py, docs/format.md §10), so results are
// bit-identical (all Hadamard sizes, m = 1 / 12 / 20 / 28; see cpp/README.md).
//
// Supported: raw (any stored dtype), bf16, affine, scalar, vq, additive, tcq (trellis, §5.5); bit-packed indices of any
// width 1..32; f16 row scales; Hadamard rotations in / in_out / out (output-only, rotate_in = false);
// residuals sparse_fp16 / sparse_int8 / columns_fp16 / columns_int8 (with optional sparse tail).
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "suc/suc1.hpp"

namespace suc {

struct DecodedTensor {
  std::vector<uint64_t> shape;
  std::vector<float> data;   // row-major
};

DecodedTensor decode_tensor(const Suc1File& f, const TensorDesc& t);
inline DecodedTensor decode_tensor(const Suc1File& f, std::string_view name) { return decode_tensor(f, f.tensor(name)); }

// Payload helpers (research layout): any array as float32 / as unsigned ints (packed arrays unpacked).
std::vector<float> array_as_f32(const Suc1File& f, const ArrayRecord& r);
std::vector<uint32_t> array_as_u32(const Suc1File& f, const ArrayRecord& r);
std::vector<float> codebook_as_f32(const Suc1File& f, uint64_t id);   // (entries, vector_length)

}  // namespace suc
