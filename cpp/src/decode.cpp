#include "suc/decode.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "suc/hadamard.hpp"
#include "suc/parallel.hpp"

namespace suc {
namespace {

[[noreturn]] void bad(const TensorDesc& t, const std::string& msg) { throw FormatError(t.name + ": " + msg); }

uint64_t numel(const std::vector<uint64_t>& s) {
  uint64_t n = 1;
  for (uint64_t d : s)
    if (__builtin_mul_overflow(n, d, &n)) throw FormatError("tensor shape product overflows 64 bits");
  return n;
}

const std::vector<uint64_t>& logical_shape(const ArrayRecord& r) { return r.packed ? r.logical_shape : r.shape; }

void expect_shape(const TensorDesc& t, const ArrayRecord& r, const std::vector<uint64_t>& want) {
  if (logical_shape(r) != want) bad(t, "array '" + r.key + "' has an unexpected logical shape");
}

bool param_bool(const Json& p, std::string_view key, bool dflt) {
  const Json* v = p.find(key);
  return (v && !v->is_null()) ? v->as_bool() : dflt;
}

uint64_t seed_param(const TensorDesc& t, const Json& p, std::string_view key) {
  const Json* v = p.find(key);
  if (!v || v->is_null()) bad(t, "rotation needs params." + std::string(key));
  return v->as_uint();
}

// x (rows, cols) -> transpose (cols, rows)
std::vector<float> transpose(const std::vector<float>& x, std::size_t rows, std::size_t cols) {
  std::vector<float> y(x.size());
  constexpr std::size_t B = 64;
  parallel_for((rows + B - 1) / B, 4, [&](std::size_t rb) {
    const std::size_t r0 = rb * B;
    for (std::size_t c0 = 0; c0 < cols; c0 += B)
      for (std::size_t r = r0; r < std::min(rows, r0 + B); ++r)
        for (std::size_t c = c0; c < std::min(cols, c0 + B); ++c) y[c * rows + r] = x[r * cols + c];
  });
  return y;
}

// Inverse rotation on the output (column) side: x <- (rht_inverse(x^T, seed))^T.
void unrotate_out(std::vector<float>& x, std::size_t out, std::size_t inn, uint64_t seed) {
  std::vector<float> xt = transpose(x, out, inn);
  rht_inverse_rows(xt.data(), inn, out, seed);
  x = transpose(xt, inn, out);
}

// tcq raw code value of entry j of state s (exact in double; docs/format.md §5.5)
double tcq_raw(int code, int j, uint32_t s) {
  static constexpr uint32_t A[2][2] = {{34038481u, 2654435761u}, {89226354u, 2654435761u}};
  static constexpr uint32_t B[2][2] = {{76625530u, 1013904223u}, {64248484u, 1013904223u}};
  const int jj = code == 2 ? 0 : j;                       // pair3: both entries from entry 0's word
  const uint32_t x = A[code == 0 ? 0 : 1][jj] * s + B[code == 0 ? 0 : 1][jj];   // mod 2^32
  if (code == 0) return double((x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24));
  const uint32_t h = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
  const double h0 = double(f16_to_f32(uint16_t(h & 0xFFFFu))), h1 = double(f16_to_f32(uint16_t(h >> 16)));
  return (code == 2 && j == 1) ? h0 - h1 : h0 + h1;
}

// = suc.io.suc1._decode_tcq: W'[r, q·T + i·V + j] = float32((raw_j(s_i) - code_mu[j]) / code_sd[j]) · float32(gain)
// · row_scales[r] (float32), s_i the L-bit window of sequence q's circular stream ending at chunk i; then the inverse
// rotation as vq (tcq defaults to rotate_out = false).
std::vector<float> decode_tcq(const Suc1File& f, const TensorDesc& t, const Json& p, const std::string& pre,
                              const std::vector<float>& s) {
  const std::size_t out = t.shape[0], inn = t.shape[1];
  const int L = int(p["L"].as_uint()), kv = int(p["kv"].as_uint()), V = int(p["V"].as_uint());
  const std::size_t T = p["T"].as_uint();
  const std::string& cs = p["code"].as_string();
  const int code = cs == "1mad" ? 0 : cs == "3inst" ? 1 : cs == "pair3" ? 2 : -1;
  if (code < 0) bad(t, "tcq code '" + cs + "' is not a SUC1 representation");
  if ((V != 1 && V != 2) || (code == 2 && V != 2)) bad(t, "tcq V unsupported");
  if (T == 0 || T % std::size_t(V) || inn % T || L < 1 || L > 32 || kv < 1) bad(t, "tcq params out of range");
  const std::size_t n = T / std::size_t(V), nseq = out * (inn / T);
  const ArrayRecord& ch = t.array(pre + "chunks");
  expect_shape(t, ch, {nseq, n});
  const auto c = array_as_u32(f, ch);
  const Json& jmu = p["code_mu"];
  const Json& jsd = p["code_sd"];
  if (jmu.size() != std::size_t(V) || jsd.size() != std::size_t(V)) bad(t, "tcq code_mu / code_sd need V entries");
  double mu[2] = {0, 0}, sd[2] = {1, 1};
  for (int j = 0; j < V; ++j) mu[j] = jmu.at(std::size_t(j)).as_double(), sd[j] = jsd.at(std::size_t(j)).as_double();
  const float gain = float(p["gain"].as_double());
  const uint64_t mask = (uint64_t(1) << L) - 1;
  std::vector<float> x(out * inn);
  parallel_for(nseq, 64, [&](std::size_t q) {
    const uint32_t* cq = c.data() + q * n;
    float* xq = x.data() + q * T;          // sequence q = row q / (in / T), columns (q % (in / T)) · T ...
    const float sr = s[q / (inn / T)];
    for (std::size_t i = 0; i < n; ++i) {
      uint64_t st = 0;
      for (int u = 0; L - kv - u * kv > -kv; ++u) {   // every chunk overlapping the window
        const int sh = L - kv - u * kv;
        const uint64_t cu = cq[(i + n * (std::size_t(u) / n + 1) - std::size_t(u)) % n];
        st |= sh >= 0 ? (cu << sh) : (cu >> -sh);
      }
      st &= mask;
      for (int j = 0; j < V; ++j) {
        const float v = float((tcq_raw(code, j, uint32_t(st)) - mu[j]) / sd[j]) * gain;
        xq[i * std::size_t(V) + std::size_t(j)] = v * sr;
      }
    }
  });
  if (!param_bool(p, "rotate", false)) return x;
  const bool rin = param_bool(p, "rotate_in", true), rout = param_bool(p, "rotate_out", false);
  if (rin) rht_inverse_rows(x.data(), out, inn, seed_param(t, p, "seed_in"));
  if (rout) unrotate_out(x, out, inn, seed_param(t, p, "seed_out"));
  if (!rin && !rout) bad(t, "rotate=true without an input or output rotation");
  return x;
}

std::vector<float> decode_base(const Suc1File& f, const TensorDesc& t, const Json& p, const std::string& pre) {
  const std::string& rep = t.representation_type;
  if (t.shape.size() != 2) bad(t, "codec tensors must be 2-D");
  const std::size_t out = t.shape[0], inn = t.shape[1];
  auto arr = [&](const char* k) -> const ArrayRecord& { return t.array(pre + k); };

  if (rep == "bf16") {
    const ArrayRecord& w = arr("weight");
    expect_shape(t, w, {out, inn});
    if (w.dtype != DType::BF16) bad(t, "bf16 weight must be stored as bf16");
    return array_as_f32(f, w);
  }
  if (rep == "mxfp4") {
    // Native MXFP4 (docs/format.md §5.4, = suc.io.suc1._decode_base): codes u8 [out, in/2], element 2c in the low
    // nibble and 2c+1 in the high nibble; E8M0 scales u8 [out, in/32]; W = E2M1(code) · 2^(S - 127). Code 8 (-0)
    // decodes to +0 like the Python table. S = 255 (NaN in E8M0) is rejected; every other S gives an exact float32.
    static constexpr float kE2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    const Json* gsv = p.find("group_size");
    const std::size_t gs = (gsv && !gsv->is_null()) ? gsv->as_uint() : 32;
    if (gs != 32) bad(t, "mxfp4 group_size must be 32");
    if (inn % gs) bad(t, "mxfp4 in_features not divisible by 32");
    const ArrayRecord &cr = arr("codes"), &sr = arr("scales");
    if (cr.packed || sr.packed || cr.dtype != DType::U8 || sr.dtype != DType::U8)
      bad(t, "mxfp4 codes / scales must be unpacked u8 arrays");
    expect_shape(t, cr, {out, inn / 2});
    expect_shape(t, sr, {out, inn / gs});
    const auto cb = f.payload(cr), sb = f.payload(sr);
    if (cb.size() != out * (inn / 2) || sb.size() != out * (inn / gs)) bad(t, "mxfp4 payload size mismatch");
    for (uint8_t s : sb)
      if (s == 255) bad(t, "mxfp4 scale 255 (E8M0 NaN)");
    std::vector<float> w(out * inn);
    parallel_for(out, 64, [&](std::size_t r) {
      for (std::size_t c = 0; c < inn / 2; ++c) {
        const uint8_t byte = cb[r * (inn / 2) + c];
        for (std::size_t h = 0; h < 2; ++h) {
          const std::size_t k = 2 * c + h;
          const float scale = std::ldexp(1.0f, int(sb[r * (inn / gs) + k / gs]) - 127);
          w[r * inn + k] = kE2M1[h ? (byte >> 4) : (byte & 0x0F)] * scale;
        }
      }
    });
    return w;
  }
  if (rep == "affine") {
    const std::size_t gs = p["group_size"].as_uint();
    if (gs == 0 || inn % gs) bad(t, "in_features not divisible by group_size");
    const ArrayRecord &qi = arr("indices"), &sc = arr("scales"), &bi = arr("biases");
    expect_shape(t, qi, {out, inn});
    expect_shape(t, sc, {out, inn / gs});
    expect_shape(t, bi, {out, inn / gs});
    const auto q = array_as_u32(f, qi);
    const auto s = array_as_f32(f, sc), b = array_as_f32(f, bi);
    std::vector<float> w(out * inn);
    parallel_for(out, 64, [&](std::size_t r) {
      for (std::size_t i = r * inn; i < (r + 1) * inn; ++i) {
        const std::size_t g = i / gs;   // (row, group) flat index since inn % gs == 0
        w[i] = float(q[i]) * s[g] + b[g];
      }
    });
    return w;
  }

  const ArrayRecord& rs = arr("row_scales");
  expect_shape(t, rs, {out});
  const auto s = array_as_f32(f, rs);
  if (rep == "tcq") return decode_tcq(f, t, p, pre, s);
  if (t.codebook_ids.empty()) bad(t, "codebook representation without codebooks");
  if (t.codebook_key != pre + "codebooks") bad(t, "unexpected codebook_key");

  if (rep == "scalar") {
    const std::size_t rg = p["row_group"].as_uint();
    if (rg == 0) bad(t, "row_group must be positive");
    const std::size_t ngroups = t.codebook_ids.size();
    if ((out + rg - 1) / rg != ngroups) bad(t, "number of scalar codebooks != row groups");
    std::vector<std::vector<float>> cbs;
    uint64_t K = 0;
    for (uint64_t id : t.codebook_ids) {
      const auto& c = f.codebooks()[id];
      if (c.vector_length != 1) bad(t, "scalar codebook with vector_length != 1");
      if (cbs.empty()) K = c.entries;
      else if (c.entries != K) bad(t, "scalar codebooks of different sizes");
      cbs.push_back(codebook_as_f32(f, id));
    }
    const ArrayRecord& qi = arr("indices");
    expect_shape(t, qi, {out, inn});
    const auto q = array_as_u32(f, qi);
    std::vector<float> w(out * inn);
    for (std::size_t r = 0; r < out; ++r) {
      const std::vector<float>& cb = cbs[r / rg];
      for (std::size_t j = 0; j < inn; ++j) {
        const uint32_t k = q[r * inn + j];
        if (k >= K) bad(t, "codebook index out of range");
        w[r * inn + j] = cb[k] * s[r];
      }
    }
    return w;
  }

  if (rep != "vq" && rep != "additive") bad(t, "unknown representation '" + rep + "'");
  const std::size_t M = t.codebook_ids.size();
  const auto& c0 = f.codebooks()[t.codebook_ids[0]];
  const std::size_t K = c0.entries, G = c0.vector_length;
  if (G == 0 || inn % G) bad(t, "in_features not divisible by the codebook vector length");
  std::vector<std::vector<float>> cbs;
  for (uint64_t id : t.codebook_ids) {
    const auto& c = f.codebooks()[id];
    if (c.entries != K || c.vector_length != G) bad(t, "codebooks of different shapes");
    cbs.push_back(codebook_as_f32(f, id));
  }
  const ArrayRecord& qi = arr("indices");
  const std::size_t nb = inn / G;
  expect_shape(t, qi, {M, out, nb});
  const auto q = array_as_u32(f, qi);
  for (uint32_t k : q)
    if (k >= K) bad(t, "codebook index out of range");
  std::vector<float> x(out * inn);
  // acc = C_0[i_0]; acc = acc + C_m[i_m] for m = 1..M-1 (float32, in order); x = acc * s_r
  parallel_for(out, 64, [&](std::size_t r) {
    float* xr = x.data() + r * inn;
    for (std::size_t b = 0; b < nb; ++b) {
      const float* v = cbs[0].data() + std::size_t(q[r * nb + b]) * G;
      for (std::size_t g = 0; g < G; ++g) xr[b * G + g] = v[g];
    }
    for (std::size_t m = 1; m < M; ++m) {
      const uint32_t* qm = q.data() + (m * out + r) * nb;
      for (std::size_t b = 0; b < nb; ++b) {
        const float* v = cbs[m].data() + std::size_t(qm[b]) * G;
        for (std::size_t g = 0; g < G; ++g) xr[b * G + g] = xr[b * G + g] + v[g];
      }
    }
    for (std::size_t j = 0; j < inn; ++j) xr[j] = xr[j] * s[r];
  });

  if (!param_bool(p, "rotate", false)) return x;
  const bool rin = param_bool(p, "rotate_in", true), rout = param_bool(p, "rotate_out", true);
  if (rin) rht_inverse_rows(x.data(), out, inn, seed_param(t, p, "seed_in"));   // x V   (rows, seed_in)
  if (rout) unrotate_out(x, out, inn, seed_param(t, p, "seed_out"));            // U^T x (columns, seed_out)
  if (!rin && !rout) bad(t, "rotate=true without an input or output rotation");
  return x;
}

}  // namespace

std::vector<float> array_as_f32(const Suc1File& f, const ArrayRecord& r) {
  if (r.packed) {
    auto u = array_as_u32(f, r);
    return std::vector<float>(u.begin(), u.end());
  }
  auto b = f.payload(r);
  const std::size_t n = numel(r.shape), sz = dtype_size(r.dtype);
  std::vector<float> out(n);
  constexpr std::size_t chunk = 1 << 16;
  parallel_for((n + chunk - 1) / chunk, 4, [&](std::size_t c) {
    for (std::size_t i = c * chunk; i < std::min(n, (c + 1) * chunk); ++i) out[i] = load_as_f32(b.data() + i * sz, r.dtype);
  });
  return out;
}

std::vector<uint32_t> array_as_u32(const Suc1File& f, const ArrayRecord& r) {
  auto b = f.payload(r);
  if (r.packed) {
    const auto& ls = r.logical_shape;
    const std::size_t n = ls.empty() ? 1 : ls.back();
    const std::size_t rows = r.shape[0], wpr = r.shape[1];
    std::vector<uint32_t> out(rows * n);
    parallel_for(rows, 256, [&](std::size_t i) { unpack_row(b.data() + 4 * i * wpr, r.bits, n, out.data() + i * n); });
    return out;
  }
  const std::size_t n = numel(r.shape), sz = dtype_size(r.dtype);
  std::vector<uint32_t> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const int64_t v = load_as_i64(b.data() + i * sz, r.dtype);
    if (v < 0 || v > int64_t(UINT32_MAX)) throw FormatError(r.key + ": index value outside [0, 2^32)");
    out[i] = uint32_t(v);
  }
  return out;
}

std::vector<float> codebook_as_f32(const Suc1File& f, uint64_t id) {
  const CodebookDesc& c = f.codebooks().at(id);
  auto b = f.payload(c);
  const std::size_t n = c.entries * c.vector_length, sz = dtype_size(c.dtype);
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = load_as_f32(b.data() + i * sz, c.dtype);
  return out;
}

DecodedTensor decode_tensor(const Suc1File& f, const TensorDesc& t) {
  DecodedTensor d;
  d.shape = t.shape;
  if (t.representation_type == "raw") {
    if (t.arrays.size() != 1 || t.arrays[0].key != "weight") bad(t, "raw tensor must have one 'weight' array");
    if (t.arrays[0].shape != t.shape) bad(t, "raw weight shape != tensor shape");
    d.data = array_as_f32(f, t.arrays[0]);
    return d;
  }
  const Json& params = *t.params;
  const bool residual = t.has_residual();
  const Json& bp = residual ? params["base"] : params;
  d.data = decode_base(f, t, bp, residual ? "base." : "");
  if (!residual) return d;

  const std::size_t out = t.shape[0], inn = t.shape[1];
  std::vector<float>& w = d.data;
  // column-structured residual: w[:, cols[j]] += colval[j] (* colscale[j]), float32
  if (const ArrayRecord* rc = t.find_array("res_cols"); rc && rc->logical_numel()) {
    const auto cols = array_as_u32(f, *rc);
    const ArrayRecord& rv = t.array("res_colval");
    if (rv.logical_numel() != cols.size() * out) bad(t, "res_colval size != columns * out");
    auto cv = array_as_f32(f, rv);
    if (const ArrayRecord* rsc = t.find_array("res_colscale")) {
      const auto cs = array_as_f32(f, *rsc);
      if (cs.size() != cols.size()) bad(t, "res_colscale size != columns");
      for (std::size_t j = 0; j < cols.size(); ++j)
        for (std::size_t r = 0; r < out; ++r) cv[j * out + r] = cv[j * out + r] * cs[j];
    }
    for (std::size_t j = 0; j < cols.size(); ++j) {
      if (cols[j] >= inn) bad(t, "residual column out of range");
      for (std::size_t r = 0; r < out; ++r) w[r * inn + cols[j]] = w[r * inn + cols[j]] + cv[j * out + r];
    }
  }
  // sparse residual: w.flat[pos] += val (* res_scale)
  if (params["n"].as_uint() > 0) {
    const auto pos = array_as_u32(f, t.array("res_pos"));
    auto v = array_as_f32(f, t.array("res_val"));
    if (pos.size() != v.size()) bad(t, "res_pos / res_val length mismatch");
    if (const ArrayRecord* rsc = t.find_array("res_scale")) {
      const auto sc = array_as_f32(f, *rsc);
      if (sc.size() != 1) bad(t, "res_scale must be a scalar");
      for (float& x : v) x = x * sc[0];
    }
    for (std::size_t i = 0; i < pos.size(); ++i) {
      if (pos[i] >= w.size()) bad(t, "residual position out of range");
      w[pos[i]] = w[pos[i]] + v[i];
    }
  }
  return d;
}

}  // namespace suc
