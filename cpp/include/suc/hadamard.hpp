// Randomized Hadamard transforms with the exact float32 operation order of suc.quant.hadamard
// (which runs on MLX): n = m * 2^k, m in {1, 12, 20, 28, 17}.
//
//  * 2^k factor: in-place Sylvester FWHT with butterfly distances 1, 2, 4, ... (the order of MLX's
//    `hadamard_n` Metal kernel and CPU path), then one multiply by float(1/sqrt(2^k)); above 8192 the
//    factor is split as in `_pow2` (8192-blocks first, then the strided factor, each scaled).
//  * m factor: H_m (MLX's +-1 tables, scaled by float(1/sqrt(m))) applied per column of the (m, 2^k)
//    view as a sequential fused-multiply-add chain over k = 0..m-1 starting from 0. This is the
//    definition of the step in suc.quant.hadamard._mix as well (Metal kernel on the GPU, exact numpy
//    emulation on the CPU), so all m are bit-identical to the Python reference.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace suc {

bool hadamard_supported(std::size_t n);

// (m, 2^k) with n = m * 2^k, m in {1, 12, 20, 28, 17}; {0, 0} when n is not supported.
struct HadamardFactor {
  std::size_t m = 0, p = 0;
};
HadamardFactor hadamard_factor(std::size_t n);

// MLX's orthonormal H_m (= suc.quant.hadamard._h_m(m)) as m*m float32, row-major ({1.0f} for m = 1).
std::vector<float> hadamard_hm(std::size_t m);

// x <- H x (transpose = false) or x <- H^T x (transpose = true), orthonormal, on one contiguous
// vector of length n. `scratch` must hold n floats.
void hadamard_transform(float* x, std::size_t n, bool transpose, float* scratch);

// rht_inverse(y, seed) = signs(seed) * (H^T y), applied to each of `rows` contiguous rows of length n.
void rht_inverse_rows(float* x, std::size_t rows, std::size_t n, uint64_t seed);

}  // namespace suc
