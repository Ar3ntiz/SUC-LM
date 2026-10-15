// MoE expert stacks and the Qwen3-MoE sparse block on the Metal backend (see suc/switch.hpp).
#include "suc/switch.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <tuple>

#include "suc/decode.hpp"
#include "suc/expert_stream.hpp"
#include "suc/hadamard.hpp"
#include "suc/json.hpp"
#include "suc/numpy_random.hpp"

namespace suc::metal {

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("metal: " + msg); }

// Constants of the Python kernels (duplicated from metal.cpp: suc.kernels.vq_matmul / rotation).
constexpr std::size_t kCodebookLimit = 16 * 1024;
constexpr int kTgFloats = 8192 - 64;
constexpr std::size_t kColresMaxColumns = 4096;   // suc_switch.COLRES_MAX_COLUMNS
int mma_kcg(int g, int bits, int kc) {
  const int gpu = 32 % bits == 0 ? 32 / bits : 1;
  const int step = std::lcm(gpu, 8 / std::gcd(g, 8));
  const int want = (kc / g + step - 1) / step * step;
  return std::max(step, want);
}

bool param_bool(const Json& p, std::string_view key, bool dflt) {
  const Json* v = p.find(key);
  return (v && !v->is_null()) ? v->as_bool() : dflt;
}

uint64_t seed_of(const Json& p, std::string_view key) {
  const Json* v = p.find(key);
  if (!v || v->is_null()) throw FormatError(std::string("rotation needs params.") + std::string(key));
  return v->as_uint();
}

void barrier(MTL::ComputeCommandEncoder* enc) { enc->memoryBarrier(MTL::BarrierScopeBuffers); }

std::string join(std::initializer_list<std::string> xs) {
  std::string s;
  for (const auto& x : xs) s += (s.empty() ? "" : ", ") + x;
  return s;
}
std::string I(long v) { return std::to_string(v); }

void ensure(Context& ctx, BufPtr& b, std::size_t& cap, std::size_t bytes) {
  if (cap < bytes) {
    b = ctx.new_buffer(bytes);
    cap = bytes;
  }
}

void set(MTL::ComputeCommandEncoder* enc, BufRef r, int i) { enc->setBuffer(r.buffer, r.offset, i); }

std::optional<Act> act_of(DType d) {
  if (d == DType::F16) return Act::F16;
  if (d == DType::BF16) return Act::BF16;
  if (d == DType::F32) return Act::F32;
  return std::nullopt;
}

struct Pool {
  NS::AutoreleasePool* p = NS::AutoreleasePool::alloc()->init();
  ~Pool() { p->release(); }
};

// One expert array per expert: where they are and whether they are consecutive in expert order.
struct Segs {
  std::vector<uint64_t> off;
  uint64_t nbytes = 0;
  bool consecutive() const {
    for (std::size_t e = 1; e < off.size(); ++e)
      if (off[e] != off[0] + e * nbytes) return false;
    return true;
  }
  uint64_t lo() const { return *std::min_element(off.begin(), off.end()); }
  uint64_t hi() const { return *std::max_element(off.begin(), off.end()) + nbytes; }
};

}  // namespace

// ------------------------------------------------------------------------------------------ SwitchLinear

std::string SwitchLinear::expert_name(const std::string& prefix, std::size_t e, const std::string& proj) {
  return prefix + ".experts." + std::to_string(e) + "." + proj + ".weight";
}

std::size_t SwitchLinear::count_experts(const Suc1File& f, const std::string& prefix, const std::string& proj) {
  std::size_t e = 0;
  while (f.find_tensor(expert_name(prefix, e, proj))) ++e;
  return e;
}

SwitchLinear SwitchLinear::from_file(Context& ctx, const MappedFile& mf, const std::string& prefix,
                                     const std::string& proj, std::size_t num_experts, StackLayout layout) {
  const std::size_t E = num_experts ? num_experts : count_experts(mf.file(), prefix, proj);
  std::vector<int64_t> slots(E);
  for (std::size_t e = 0; e < E; ++e) slots[e] = int64_t(e);
  return build(ctx, mf, prefix, proj, slots, layout, false);
}

SwitchLinear SwitchLinear::from_file_slots(Context& ctx, const MappedFile& mf, const std::string& prefix,
                                           const std::string& proj, const std::vector<int64_t>& slots,
                                           bool streamed) {
  return build(ctx, mf, prefix, proj, slots, streamed ? StackLayout::Streamed : StackLayout::Table, true);
}

// slots[s]: the expert read by slot s, or -1 (empty slot: skipped by the lookup kernel, Table layout only). Arrays of
// empty slots repeat the first member's (signs, offsets) or are empty (residuals); only the packed-index offset carries
// the ~0 sentinel the _tab prologue tests.
SwitchLinear SwitchLinear::build(Context& ctx, const MappedFile& mf, const std::string& prefix, const std::string& proj,
                                 const std::vector<int64_t>& slots, StackLayout layout, bool subset) {
  const Suc1File& f = mf.file();
  const std::size_t E = slots.size();
  SwitchLinear S;
  S.name_ = prefix + ".switch_mlp." + proj;
  if (!E) throw FormatError(S.name_ + ": no expert tensors " + expert_name(prefix, 0, proj));
  S.E_ = E;
  std::vector<const TensorDesc*> ts(E, nullptr);
  std::size_t first = E;
  for (std::size_t e = 0; e < E; ++e) {
    if (slots[e] < 0) continue;
    if (first == E) first = e;
    const std::string n = expert_name(prefix, std::size_t(slots[e]), proj);
    ts[e] = f.find_tensor(n);
    if (!ts[e]) throw FormatError(S.name_ + ": missing expert tensor " + n + " (all experts must be present)");
    std::string why;
    if (!Linear::supported(*ts[e], &why)) throw FormatError(n + ": " + why);
    const std::string& rt = ts[e]->residual_type;
    if (rt != "none" && rt != "sparse_fp16" && rt != "sparse_int8" && rt != "columns_fp16" && rt != "columns_int8")
      throw FormatError(n + ": residual " + rt + " is not supported for expert stacks");
  }
  if (first == E) throw FormatError(S.name_ + ": a sub-stack without experts");
  bool identity = true;
  for (std::size_t e = 0; e < E; ++e) identity = identity && slots[e] == int64_t(e);
  if (!identity && layout != StackLayout::Streamed) layout = StackLayout::Table;
  // empty slots alias the first member (only its packed offset is replaced by the sentinel below)
  for (std::size_t e = 0; e < E; ++e)
    if (!ts[e]) ts[e] = ts[first];
  auto empty = [&](std::size_t e) { return slots[e] < 0; };
  for (std::size_t e = 0; e < E && layout == StackLayout::Streamed; ++e)   // streamed slots: slot s = expert s
    if (slots[e] >= 0 && slots[e] != int64_t(e)) throw FormatError(S.name_ + ": streamed slots must be identity or empty");
  (void)subset;
  // ---- common layout (= suc_switch._layout: codec, index bits, codebook shape, rotation sides; and the shape)
  const TensorDesc& t0 = *ts[first];
  S.out_ = t0.shape[0];
  S.in_ = t0.shape[1];
  auto base_params = [](const TensorDesc& t) -> const Json& { return t.has_residual() ? (*t.params)["base"] : *t.params; };
  S.tcq_ = t0.representation_type == "tcq";
  const bool tcq = S.tcq_;
  std::size_t cb_bytes = 0;
  if (tcq) {   // trellis-coded experts: no codebooks, streams (out, in/T · NW) u32 per expert
    S.tcq_l_ = TcqLayout::from_params(base_params(t0), S.in_, t0.name);
    S.M_ = 1;
    S.bits_ = unsigned(S.tcq_l_.KV);
    S.words_ = S.tcq_l_.words(S.in_);
  } else {
    S.M_ = t0.codebook_ids.size();
    const CodebookDesc& c00 = f.codebooks().at(t0.codebook_ids[0]);
    S.K_ = c00.entries;
    S.G_ = c00.vector_length;
    S.bits_ = unsigned(*t0.index_bits);
    if (S.G_ == 0 || S.in_ % S.G_) throw FormatError(S.name_ + ": in_features not divisible by G");
    if (S.K_ > (uint64_t(1) << S.bits_)) throw FormatError(S.name_ + ": K exceeds 2^index_bits");
    cb_bytes = S.M_ * S.K_ * S.G_ * 2;
    if (cb_bytes > kCodebookLimit) throw FormatError(S.name_ + ": codebooks exceed the 16 KB ceiling (D4)");
    S.words_ = words_per_row(S.in_ / S.G_, S.bits_);
  }
  auto rot_sides = [&](const TensorDesc& t) {
    const Json& p = base_params(t);
    const bool rot = param_bool(p, "rotate", false);
    // tcq tensors default to rotate_out = false (SucLinear._from_tcq)
    return std::pair<bool, bool>{rot && param_bool(p, "rotate_in", true),
                                 rot && param_bool(p, "rotate_out", t.representation_type != "tcq")};
  };
  std::tie(S.rot_in_, S.rot_out_) = rot_sides(t0);
  std::vector<uint64_t> seeds_in(E), seeds_out(E);
  Segs pk, sc, cbs;
  pk.off.resize(E);
  sc.off.resize(E);
  bool same_cb = true;
  std::vector<uint8_t> cb0;
  for (std::size_t e = 0; e < E; ++e) {
    const TensorDesc& t = *ts[e];
    const std::string pre = t.has_residual() ? "base." : "";
    if (t.shape != t0.shape || t.representation_type != t0.representation_type ||
        (!tcq && (t.index_bits != t0.index_bits || t.codebook_ids.size() != S.M_)) || rot_sides(t) != rot_sides(t0))
      throw FormatError(t.name + ": experts differ in shape or code layout (codec, bits, codebooks, rotation sides)");
    if (tcq) {
      if (TcqLayout::from_params(base_params(t), S.in_, t.name).key != S.tcq_l_.key)
        throw FormatError(t.name + ": experts differ in tcq code layout (k, L, V, T, code, gain, code statistics)");
      const ArrayRecord& ch = t.array(pre + "chunks");
      if (!ch.packed || ch.bits != S.bits_ || ch.nbytes != S.out_ * S.words_ * 4 ||
          ch.shape != std::vector<uint64_t>{S.out_ * (S.in_ / std::size_t(S.tcq_l_.T)), uint64_t(S.tcq_l_.NW)})
        throw FormatError(t.name + ": stored tcq streams do not match the codec params");
      const ArrayRecord& rs = t.array(pre + "row_scales");
      if (rs.dtype != DType::F16 || rs.shape != std::vector<uint64_t>{S.out_}) throw FormatError(t.name + ": row_scales");
      pk.off[e] = ch.offset;
      pk.nbytes = ch.nbytes;
      sc.off[e] = rs.offset;
      sc.nbytes = rs.nbytes;
      const Json& p = base_params(t);
      if (S.rot_in_) seeds_in[e] = seed_of(p, "seed_in");
      if (S.rot_out_) seeds_out[e] = seed_of(p, "seed_out");
      continue;
    }
    for (std::size_t m = 0; m < S.M_; ++m) {
      const CodebookDesc& c = f.codebooks().at(t.codebook_ids[m]);
      if (c.entries != S.K_ || c.vector_length != S.G_ || c.dtype != DType::F16)
        throw FormatError(t.name + ": codebooks of a different shape / dtype");
    }
    const ArrayRecord& qi = t.array(pre + "indices");
    if (!qi.packed || qi.bits != S.bits_ || qi.shape != std::vector<uint64_t>{S.M_ * S.out_, S.words_})
      throw FormatError(t.name + ": unexpected packed index layout");
    const ArrayRecord& rs = t.array(pre + "row_scales");
    if (rs.dtype != DType::F16 || rs.shape != std::vector<uint64_t>{S.out_}) throw FormatError(t.name + ": row_scales");
    pk.off[e] = qi.offset;
    pk.nbytes = qi.nbytes;
    sc.off[e] = rs.offset;
    sc.nbytes = rs.nbytes;
    const Json& p = base_params(t);
    if (S.rot_in_) seeds_in[e] = seed_of(p, "seed_in");
    if (S.rot_out_) seeds_out[e] = seed_of(p, "seed_out");
    // codebook set of expert e (bytes, in id order)
    std::vector<uint8_t> cb(cb_bytes);
    for (std::size_t m = 0; m < S.M_; ++m) {
      auto s = f.payload(f.codebooks().at(t.codebook_ids[m]));
      std::memcpy(cb.data() + m * S.K_ * S.G_ * 2, s.data(), s.size());
    }
    if (e == 0) cb0 = cb;
    else same_cb = same_cb && (t.codebook_ids == t0.codebook_ids || cb == cb0);
  }
  // codebooks of expert e adjacent in the file (M segments in id order)?
  auto cb_adjacent = [&](const TensorDesc& t) {
    const CodebookDesc& c0 = f.codebooks().at(t.codebook_ids[0]);
    for (std::size_t m = 0; m < S.M_; ++m)
      if (f.codebooks().at(t.codebook_ids[m]).offset != c0.offset + m * S.K_ * S.G_ * 2) return false;
    return true;
  };
  S.cbe_ = !tcq && E > 1 && !same_cb;
  bool cb_inplace_each = true;
  cbs.off.resize(tcq ? 0 : S.cbe_ ? E : 1);
  cbs.nbytes = cb_bytes;
  for (std::size_t e = 0; e < cbs.off.size(); ++e) {
    cbs.off[e] = f.codebooks().at(ts[e]->codebook_ids[0]).offset;
    cb_inplace_each = cb_inplace_each && cb_adjacent(*ts[e]);
  }
  const bool cb_stack_inplace = tcq || (cb_inplace_each && cbs.consecutive());
  if (layout == StackLayout::Auto)
    layout = (pk.consecutive() && sc.consecutive() && cb_stack_inplace) ? StackLayout::Stacked : StackLayout::Table;
  S.layout_ = layout;
  const bool streamed = layout == StackLayout::Streamed;
  const bool table = layout == StackLayout::Table || streamed;   // per-expert offset table (the _tab kernels)

  // copy E arrays (or the codebook sets) into one dense stack
  auto stack_copy = [&](const std::vector<const uint8_t*>& srcs, std::size_t each) {
    BufPtr b = ctx.new_buffer(srcs.size() * each);
    for (std::size_t e = 0; e < srcs.size(); ++e)
      std::memcpy(static_cast<uint8_t*>(b->contents()) + e * each, srcs[e], each);
    S.owned_.push_back(b);
    S.copied_bytes_ += srcs.size() * each;
    return BufRef{b.get(), 0};
  };
  auto cb_set_bytes = [&](std::size_t e) {
    std::vector<uint8_t> cb(cb_bytes);
    for (std::size_t m = 0; m < S.M_; ++m) {
      auto s = f.payload(f.codebooks().at(ts[e]->codebook_ids[m]));
      std::memcpy(cb.data() + m * S.K_ * S.G_ * 2, s.data(), s.size());
    }
    return cb;
  };
  std::vector<uint64_t> eoff(3 * E, 0);
  // ---- codebooks (tcq: none; the kernels have no codebook input)
  if (tcq) {
    S.codebooks_ = ctx.dummy();
  } else if (cb_stack_inplace) {
    S.codebooks_ = mf.at(cbs.off[0], cbs.off.size() * cb_bytes);
    S.zero_copy_bytes_ += cbs.off.size() * cb_bytes;
    for (std::size_t e = 0; S.cbe_ && e < E; ++e) eoff[3 * e] = e * S.M_ * S.K_ * S.G_;
  } else if (table && cb_inplace_each) {
    const BufRef r = mf.at(cbs.lo(), cbs.hi() - cbs.lo());
    S.codebooks_ = r;
    S.zero_copy_bytes_ += cbs.off.size() * cb_bytes;
    for (std::size_t e = 0; S.cbe_ && e < E; ++e) eoff[3 * e] = (cbs.off[e] - cbs.lo()) / 2;
  } else {
    std::vector<std::vector<uint8_t>> sets;
    std::vector<const uint8_t*> ptrs;
    for (std::size_t e = 0; e < cbs.off.size(); ++e) sets.push_back(cb_set_bytes(e));
    for (auto& s : sets) ptrs.push_back(s.data());
    S.codebooks_ = stack_copy(ptrs, cb_bytes);
    for (std::size_t e = 0; S.cbe_ && e < E; ++e) eoff[3 * e] = e * S.M_ * S.K_ * S.G_;
  }
  S.weight_bytes_ += cbs.off.size() * cb_bytes;
  // ---- packed indices and row scales
  auto place = [&](const Segs& s, BufRef& dst, int slot, std::size_t elem) {
    S.weight_bytes_ += E * s.nbytes;
    if (streamed) {   // bound to slot pool offsets by bind_expert (never mapped for the GPU)
      dst = ctx.dummy();
      (slot == 1 ? S.pk_off_ : S.sc_off_) = s.off;
      (slot == 1 ? S.pk_bytes_ : S.sc_bytes_) = s.nbytes;
      (void)elem;
    } else if (s.consecutive()) {
      dst = mf.at(s.off[0], E * s.nbytes);
      S.zero_copy_bytes_ += E * s.nbytes;
      for (std::size_t e = 0; e < E; ++e) eoff[3 * e + slot] = e * s.nbytes / elem;
    } else if (table) {
      dst = mf.at(s.lo(), s.hi() - s.lo());
      S.zero_copy_bytes_ += E * s.nbytes;
      for (std::size_t e = 0; e < E; ++e) eoff[3 * e + slot] = (s.off[e] - s.lo()) / elem;
    } else {
      std::vector<const uint8_t*> ptrs;
      for (std::size_t e = 0; e < E; ++e) ptrs.push_back(f.bytes(s.off[e], s.nbytes).data());
      dst = stack_copy(ptrs, s.nbytes);
      for (std::size_t e = 0; e < E; ++e) eoff[3 * e + slot] = e * s.nbytes / elem;
    }
  };
  place(pk, S.packed_, 1, 4);
  place(sc, S.scales_, 2, 2);
  S.empty_.assign(E, 0);
  for (std::size_t e = 0; e < E; ++e)   // empty slots: the _tab prologue skips their rows
    if (empty(e)) eoff[3 * e + 1] = ~uint64_t(0), S.empty_[e] = 1;
  if (table) {
    S.eoff_ = ctx.upload(eoff.data(), eoff.size() * 8);
    S.copied_bytes_ += eoff.size() * 8;
  }
  if (tcq) {
    S.tcq_cst_ = ctx.upload(S.tcq_l_.cst, sizeof S.tcq_l_.cst);
    S.copied_bytes_ += sizeof S.tcq_l_.cst;
  }

  // ---- rotations: one shared sign vector when every expert has the same seed, else a per-row table (E, n)
  auto signs = [&](std::size_t n, const std::vector<uint64_t>& seeds, BufPtr& s, BufPtr& hm, bool& rows) {
    rows = std::any_of(seeds.begin(), seeds.end(), [&](uint64_t v) { return v != seeds[0]; });
    const HadamardFactor hf = hadamard_factor(n);
    if (!hf.m) throw FormatError(S.name_ + ": unsupported rotation size " + std::to_string(n));
    if (rows && hf.p < 32)
      throw FormatError(S.name_ + ": per-expert rotations of size " + std::to_string(n) +
                        " need the blocks layout (2^k >= 32)");
    std::vector<float> v;
    for (std::size_t e = 0; e < (rows ? E : 1); ++e) {
      const std::vector<float> r = random_signs(n, seeds[e]);
      v.insert(v.end(), r.begin(), r.end());
    }
    s = ctx.upload(v.data(), v.size() * 4);
    const std::vector<float> h = hadamard_hm(hf.m);
    hm = ctx.upload(h.data(), h.size() * 4);
    S.copied_bytes_ += v.size() * 4;
  };
  if (S.rot_in_) signs(S.in_, seeds_in, S.signs_in_, S.hm_in_, S.rows_in_);
  if (S.rot_out_) signs(S.out_, seeds_out, S.signs_out_, S.hm_out_, S.rows_out_);

  // ---- residual: CSR over the E·out stacked rows (= SucSwitchLinear._residual)
  const bool u16 = S.in_ <= 65536;
  S.rcols_u16_ = u16;
  std::vector<uint32_t> row_ptr(E * S.out_ + 1, 0);
  std::vector<uint16_t> c16;
  std::vector<uint32_t> c32;
  std::vector<uint8_t> vals;
  std::vector<float> scales(E, 1.0f);
  std::optional<DType> vdt;
  for (std::size_t e = 0; e < E; ++e) {
    const TensorDesc& t = *ts[e];
    if (empty(e) || !t.has_residual() || (*t.params)["n"].as_uint() == 0) continue;
    const auto pos = array_as_u32(f, t.array("res_pos"));
    if (pos.empty()) continue;
    const ArrayRecord& rv = t.array("res_val");
    if (rv.logical_numel() != pos.size()) throw FormatError(t.name + ": res_pos / res_val length mismatch");
    if (rv.dtype != DType::I8 && rv.dtype != DType::F16) throw FormatError(t.name + ": res_val dtype");
    if (vdt && *vdt != rv.dtype) throw FormatError(S.name_ + ": experts store residual values in different dtypes");
    vdt = rv.dtype;
    for (std::size_t i = 0; i < pos.size(); ++i) {
      const uint64_t r = pos[i] / S.in_, c = pos[i] % S.in_;
      if (r >= S.out_ || (i && pos[i] <= pos[i - 1])) throw FormatError(t.name + ": res_pos not sorted / in range");
      ++row_ptr[e * S.out_ + r + 1];
      if (u16) c16.push_back(uint16_t(c));
      else c32.push_back(uint32_t(c));
    }
    const auto b = f.payload(rv);
    vals.insert(vals.end(), b.begin(), b.end());
    if (const ArrayRecord* s = t.find_array("res_scale")) scales[e] = array_as_f32(f, *s).at(0);
    S.n_res_ += pos.size();
  }
  // ---- column residual: per-expert column sets concatenated in expert order (= SucSwitchLinear._column_residual)
  std::vector<uint32_t> cptr(E + 1, 0), ccols;
  std::vector<uint8_t> cvals;
  std::vector<float> cscale;
  std::optional<DType> cdt;
  for (std::size_t e = 0; e < E; ++e) {
    const TensorDesc& t = *ts[e];
    const ArrayRecord* rc = !empty(e) && t.has_residual() ? t.find_array("res_cols") : nullptr;
    std::size_t c = 0;
    if (rc && rc->logical_numel()) {
      const auto cols = array_as_u32(f, *rc);
      for (std::size_t j = 0; j < cols.size(); ++j)
        if (cols[j] >= S.in_ || (j && cols[j] <= cols[j - 1])) throw FormatError(t.name + ": res_cols not sorted / in range");
      const ArrayRecord& cv = t.array("res_colval");
      if (cv.shape != std::vector<uint64_t>{cols.size(), S.out_}) throw FormatError(t.name + ": res_colval shape");
      if (cv.dtype != DType::I8 && cv.dtype != DType::F16) throw FormatError(t.name + ": res_colval dtype");
      if (cdt && *cdt != cv.dtype)
        throw FormatError(S.name_ + ": experts store column residual values in different dtypes");
      cdt = cv.dtype;
      std::vector<float> cs(cols.size(), 1.0f);
      if (const ArrayRecord* sc = t.find_array("res_colscale")) cs = array_as_f32(f, *sc);
      if (cs.size() != cols.size()) throw FormatError(t.name + ": res_colscale size");
      ccols.insert(ccols.end(), cols.begin(), cols.end());
      const auto b = f.payload(cv);
      cvals.insert(cvals.end(), b.begin(), b.end());
      cscale.insert(cscale.end(), cs.begin(), cs.end());
      c = cols.size();
    }
    cptr[e + 1] = cptr[e] + uint32_t(c);
    S.cmax_ = std::max(S.cmax_, c);
  }
  S.n_cols_ = cptr[E];
  if (S.n_cols_) {
    if (S.cmax_ > kColresMaxColumns)
      throw FormatError(S.name_ + ": more residual columns per expert than the gather buffer (4096)");
    S.ccols_u16_ = u16;
    S.cval_i8_ = *cdt == DType::I8;
    S.cptr_ = ctx.upload(cptr.data(), cptr.size() * 4);
    if (u16) {
      std::vector<uint16_t> c(ccols.begin(), ccols.end());
      S.ccols_ = ctx.upload(c.data(), c.size() * 2);
    } else {
      S.ccols_ = ctx.upload(ccols.data(), ccols.size() * 4);
    }
    S.cval_ = ctx.upload(cvals.data(), cvals.size());
    S.cscale_ = ctx.upload(cscale.data(), cscale.size() * 4);
    const std::size_t b = cptr.size() * 4 + S.n_cols_ * ((u16 ? 2 : 4) + 4) + cvals.size();
    S.copied_bytes_ += b;
    S.weight_bytes_ += b;
  }
  if (S.n_res_ || S.n_cols_) S.rscale_ = ctx.upload(scales.data(), scales.size() * 4);
  if (S.n_res_) {
    for (std::size_t r = 0; r < E * S.out_; ++r) row_ptr[r + 1] += row_ptr[r];
    S.rvals_i8_ = *vdt == DType::I8;
    S.row_ptr_ = ctx.upload(row_ptr.data(), row_ptr.size() * 4);
    S.rcols_ = u16 ? ctx.upload(c16.data(), c16.size() * 2) : ctx.upload(c32.data(), c32.size() * 4);
    S.rvals_ = ctx.upload(vals.data(), vals.size());
    const std::size_t b = row_ptr.size() * 4 + S.n_res_ * (u16 ? 2 : 4) + vals.size() + scales.size() * 4;
    S.copied_bytes_ += b;
    S.weight_bytes_ += b;
  }
  return S;
}

void SwitchLinear::encode_rotation(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef eidx, BufRef y,
                                   std::size_t N, bool inverse, Act xa, Act ya) const {
  const std::size_t n = inverse ? out_ : in_;
  const bool rows = inverse ? rows_out_ : rows_in_;
  const BufRef s{(inverse ? signs_out_ : signs_in_).get(), 0};
  if (!rows) {   // = rotation.rotate(x2, signs[0]) (one rotation shared by every expert)
    encode_rht(ctx, enc, x, y, N, n, s, {(inverse ? hm_out_ : hm_in_).get(), 0}, inverse, xa, ya);
    return;
  }
  // = rotation.rotate(x2, signs (E, n), rows=eidx): rotation._blocks_plan(row_signs=True), one segment
  if (ctx.rotation_row_layout(n)) {   // or the "row" layout (rotation._plan(row_signs=True))
    encode_rot_row(ctx, enc, x, s, {y}, {n}, N, inverse, xa, ya, eidx);
    return;
  }
  const HadamardFactor f = hadamard_factor(n);
  const int m = int(f.m), p = int(f.p);
  const auto [qmax, sgmax] = ctx.rotation_block_config(n);
  const int q = std::min(qmax, p);
  const int sg = std::min({sgmax, m * p / q, kTgFloats / q});
  const int t = 32 * std::max(1, sg);
  const int ge = m * p / q;
  const std::string targs = join({act_msl(xa), act_msl(ya), I(q), I(t), I(q / 32), I(inverse), I(long(n)),
                                  I((t / 32) * q), I(m), I(p), "0", "0", I(ge)});
  enc->setComputePipelineState(ctx.pipeline("suc_rot_blocks_rows", targs));
  set(enc, x, 0);
  set(enc, s, 1);
  set(enc, eidx, 2);
  set(enc, y, 3);
  enc->dispatchThreads(MTL::Size(t, ge, N), MTL::Size(t, 1, 1));
}

// Rows per simdgroup of the expert vec lookup at decode batch sizes (Python suc_switch: lookup_rows(1) = 4). 2 was
// faster on the M1 Ultra with either codebook source (DeepSeek V1: 49.1 -> 47.2 ms/token device, 45.6 -> 41.9
// threadgroup; docs/decode_speed_vs_ds4.md §8).
constexpr int kSwitchVecRows = 2;

// = suc_switch.gather_matmul_metal(x, ..., tile_experts = eidx, tile = 1), HM = 0; the codebook source and the row
// blocking of the vec kernel may differ from Python's (CBTG = 0, lookup_rows(1)) without changing any output bit.
std::size_t SwitchLinear::max_tile_rows(const Context& ctx) const {
  if (tcq_) return kTcqExactRows;   // tcq vec kernel: no row's arithmetic depends on the tile height
  if (1 >= ctx.mma_min_batch(int(M_))) return 1;   // decode runs the mma lookup: no row-exact batched variant
  // suc_lookup_vec's CHUNK switch (the per-lane group assignment) as a function of BT
  const unsigned low = bits_ & (~bits_ + 1u), gcd = low < 32u ? low : 32u;
  auto chunk = [&](std::size_t bt) {
    return gcd == bits_ || (G_ % 2 == 0 && bt <= 6) || (bt <= 2 && bits_ / gcd <= 3);
  };
  std::size_t b = 1;
  while (b < 16 && chunk(b + 1) == chunk(1)) ++b;
  return b;
}

void SwitchLinear::encode_kernel(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef eidx, BufRef y,
                                 std::size_t N, Act xa, Act ya, const Tiles* tiles) const {
  const Tuning& tu = ctx.tuning();
  const int m = int(M_);
  const std::size_t bt = tiles ? tiles->bt : 1;
  const bool fast = tiles && !tiles->exact;
  const bool mma = fast ? int(bt) >= ctx.mma_min_batch(m) : 1 >= ctx.mma_min_batch(m);
  if (tiles) {
    if (fast && bt > 16) fail(name_ + ": tile height " + std::to_string(bt) + " > 16");
    if (!fast && bt > max_tile_rows(ctx)) fail(name_ + ": tile height " + std::to_string(bt) + " is not row-exact");
    if (N != tiles->count * bt) fail(name_ + ": tiled rows must be count * bt");
    eidx = tiles->teidx;
    N = tiles->count;
  }
  if (tcq_) {   // = tcq_gather_matmul_metal(x, words, scales, tile_experts, tile = bt): vec for bt < 5
    const bool tab = layout_ == StackLayout::Table || layout_ == StackLayout::Streamed;
    // batched prefill: the GEMM kernel over runs of consecutive tiles of one expert (its weights decoded once per run of
    // up to 64 rows; per tile it was slower than tcq_matmul's 8-row mma tiles: random-access decode, DeepSeek tcq 8.2 vs
    // 6.6 ms/token).
    // default off: 8.34 vs 7.34 ms/token on DeepSeek-V4-Flash tcq (786 tokens, chunk 1024, M1 Ultra); $SUC_TCQ_RUNS=1 on
    static const bool runs_on = [] { const char* e = std::getenv("SUC_TCQ_RUNS"); return e && e[0] == '1'; }();
    if (runs_on && fast && tiles->host_teidx && bt >= 4 && in_ % 32 == 0 && xa != Act::F32) {
      const auto runs = tile_runs(tiles->host_teidx, tiles->count, 64 / bt);
      std::size_t mx = 0;
      for (const auto& r : runs) mx = std::max<std::size_t>(mx, r.second);
      BufPtr rb = ctx.upload(runs.data(), runs.size() * 8);   // retained by the command buffer
      encode_tcq_mm(ctx, enc, tcq_l_, tab ? 2 : 1, x, packed_, scales_, true, {tcq_cst_.get(), 0}, eidx,
                    tab ? BufRef{eoff_.get(), 0} : BufRef{}, y, bt, N, out_, in_, xa, ya, {rb.get(), 0}, runs.size(),
                    mx * bt);
      return;
    }
    if (fast && bt > 8 && bt % 8 == 0 && tiles->host_teidx) {
      // 16-row tiles as 8-row halves (tcq_matmul's 16-row mma tile was 3.5x slower than two 8-row tiles on the M1
      // Ultra); the other sub-stacks of a mixed stack keep their 16-row tiles
      const std::size_t h = bt / 8;
      std::vector<uint32_t> te(N * h);
      for (std::size_t t = 0; t < N; ++t)
        for (std::size_t k = 0; k < h; ++k) te[t * h + k] = tiles->host_teidx[t];
      BufPtr tb = ctx.upload(te.data(), te.size() * 4);   // retained by the command buffer
      encode_tcq(ctx, enc, tcq_l_, tab ? 2 : 1, x, packed_, scales_, true, {tcq_cst_.get(), 0}, {tb.get(), 0},
                 tab ? BufRef{eoff_.get(), 0} : BufRef{}, y, 8, N * h, out_, in_, xa, ya, Path::Prefill);
      return;
    }
    encode_tcq(ctx, enc, tcq_l_, tab ? 2 : 1, x, packed_, scales_, true, {tcq_cst_.get(), 0}, eidx,
               tab ? BufRef{eoff_.get(), 0} : BufRef{}, y, bt, N, out_, in_, xa, ya, fast ? Path::Prefill : Path::Auto);
    return;
  }
  const std::string head = join({act_msl(xa), I(long(bt)), I(long(out_)), I(long(in_)), I(long(in_ / G_)), I(long(G_)),
                                 I(long(K_)), I(long(bits_)), I(long(words_)), I(cbe_)});
  std::string kernel, targs;
  int sgs, rows;
  if (mma) {
    sgs = tu.mma_sg;
    rows = 8 * tu.mma_rt;
    const char* tt = (xa == Act::F16 && M_ == 1) ? "half" : "float";
    kernel = "suc_switch_lookup_mma";
    targs = head + ", " + join({I(sgs), I(m), I(tu.mma_rt), I(long((bt + 7) / 8)), I(mma_kcg(int(G_), int(bits_), tu.mma_kc)), tt, "0",
                                act_msl(ya), "half"});
  } else {
    // Codebook source and rows per simdgroup / simdgroups per threadgroup of the vec lookup: bit-identical for any
    // value (no row's arithmetic changes; checked on every C++ model, docs/decode_speed_vs_ds4.md §8). Defaults:
    // codebooks staged in threadgroup memory when the device's tuned `codebook_in` for M says so (the dense lookup's
    // per-device choice, kernel_tuning.json) and they fit in 32 KB; kSwitchVecRows rows per simdgroup; LOOKUP_SG
    // simdgroups. SUC_SWITCH_CBTG=0|1, SUC_SWITCH_ROWS, SUC_SWITCH_SG override.
    static const int env_cbtg = [] { const char* v = std::getenv("SUC_SWITCH_CBTG"); return v ? std::atoi(v) : -1; }();
    static const int env_rows = [] { const char* v = std::getenv("SUC_SWITCH_ROWS"); return v ? std::atoi(v) : 0; }();
    static const int env_sg = [] { const char* v = std::getenv("SUC_SWITCH_SG"); return v ? std::atoi(v) : 0; }();
    const bool want_tg = env_cbtg >= 0 ? env_cbtg == 1 : ctx.codebook_in(m) == "threadgroup";
    const bool cbtg = want_tg && M_ * K_ * G_ * 2 <= 32 * 1024;
    sgs = env_sg > 0 ? env_sg : tu.lookup_sg;
    rows = env_rows > 0 ? env_rows : kSwitchVecRows;
    kernel = "suc_switch_lookup_vec";
    targs = head + ", " + join({I(sgs), I(m), I(rows), I(tu.lookup_vec), cbtg ? "1" : "0", "0", act_msl(ya), "half"});
  }
  const bool tab = layout_ == StackLayout::Table || layout_ == StackLayout::Streamed;
  if (tab) kernel += "_tab";
  enc->setComputePipelineState(ctx.pipeline(kernel, targs));
  set(enc, x, 0);
  set(enc, codebooks_, 1);
  set(enc, packed_, 2);
  set(enc, scales_, 3);
  set(enc, eidx, 4);
  set(enc, y, 5);
  if (tab) enc->setBuffer(eoff_.get(), 0, 6);
  const std::size_t n_tg = (out_ + std::size_t(sgs * rows) - 1) / std::size_t(sgs * rows);
  enc->dispatchThreads(MTL::Size(n_tg * sgs * 32, N, 1), MTL::Size(sgs * 32, 1, 1));
}

std::pair<std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>> SwitchLinear::expert_ranges(std::size_t e) const {
  if (layout_ != StackLayout::Streamed) fail(name_ + ": expert_ranges needs StackLayout::Streamed");
  return {{pk_off_.at(e), pk_bytes_}, {sc_off_.at(e), sc_bytes_}};
}

void SwitchLinear::bind_expert(std::size_t e, MTL::Buffer* pool, uint64_t packed_at, uint64_t scales_at) {
  if (layout_ != StackLayout::Streamed) fail(name_ + ": bind_expert needs StackLayout::Streamed");
  if (e >= E_) fail(name_ + ": bind_expert: expert id out of range");
  if (empty_slot(e)) fail(name_ + ": bind_expert: expert " + std::to_string(e) + " is not in this sub-stack");
  if (packed_at % 4 || scales_at % 2) fail(name_ + ": bind_expert: misaligned slot offsets");
  if (packed_ .buffer != pool || scales_.buffer != pool) {   // first binding: the pool becomes both bases
    packed_ = {pool, 0};
    scales_ = {pool, 0};
  }
  auto* t = static_cast<uint64_t*>(eoff_->contents());
  t[3 * e + 1] = packed_at / 4;
  t[3 * e + 2] = scales_at / 2;
}

// = SucSwitchLinear._forward(x2, eidx, "gather")
void SwitchLinear::encode(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef eidx, BufRef y,
                          std::size_t N, Act act, Workspace& ws, const Tiles* tiles) const {
  if (!N) return;
  BufRef xr = x;
  if (rot_in_) {
    ensure(ctx, ws.xr, ws.xr_bytes, N * in_ * act_size(act));
    xr = {ws.xr.get(), 0};
    encode_rotation(ctx, enc, x, eidx, xr, N, false, act, act);
    barrier(enc);
  }
  const bool res = n_res_ > 0 || n_cols_ > 0;
  if (!rot_out_ && !res) {
    encode_kernel(ctx, enc, xr, eidx, y, N, act, act, tiles);
    return;
  }
  ensure(ctx, ws.y32, ws.y32_bytes, N * out_ * 4);
  BufRef a{ws.y32.get(), 0};
  encode_kernel(ctx, enc, xr, eidx, a, N, act, Act::F32, tiles);
  barrier(enc);
  ensure(ctx, ws.y32b, ws.y32b_bytes, N * out_ * 4);
  BufRef b{ws.y32b.get(), 0};
  if (rot_out_) {
    if (!res) {
      encode_rotation(ctx, enc, a, eidx, y, N, true, Act::F32, act);
      return;
    }
    encode_rotation(ctx, enc, a, eidx, b, N, true, Act::F32, Act::F32);
    barrier(enc);
    std::swap(a, b);
  }
  if (n_cols_) {   // = switch_structured_residual_metal(..., out_dtype = dtype): rounding inside the kernel
    encode_column_residual(ctx, enc, x, eidx, a, y, N, act);
    return;
  }
  // = gather_residual_metal (float32 out), then .astype(dtype)
  const std::string targs = join({act_msl(act), rcols_u16_ ? "uint16_t" : "uint32_t", rvals_i8_ ? "int8_t" : "half",
                                  I(long(in_)), I(long(out_))});
  enc->setComputePipelineState(ctx.pipeline("suc_switch_residual", targs));
  set(enc, x, 0);
  set(enc, a, 1);
  enc->setBuffer(row_ptr_.get(), 0, 2);
  enc->setBuffer(rcols_.get(), 0, 3);
  enc->setBuffer(rvals_.get(), 0, 4);
  enc->setBuffer(rscale_.get(), 0, 5);
  set(enc, eidx, 6);
  set(enc, b, 7);
  const std::size_t tx = std::min<std::size_t>(256, out_);
  enc->dispatchThreads(MTL::Size((out_ + tx - 1) / tx * tx, N, 1), MTL::Size(tx, 1, 1));
  barrier(enc);
  const uint32_t cnt = uint32_t(N * out_);
  enc->setComputePipelineState(ctx.pipeline("suc_cast", join({"float", act_msl(act)})));
  set(enc, b, 0);
  set(enc, y, 1);
  enc->setBytes(&cnt, 4, 2);
  enc->dispatchThreads(MTL::Size(cnt, 1, 1), MTL::Size(std::min<uint32_t>(256, cnt), 1, 1));
}

void SwitchLinear::prepare(Context& ctx, Workspace& ws, std::size_t N, Act act) const {
  if (rot_in_) ensure(ctx, ws.xr, ws.xr_bytes, N * in_ * act_size(act));
  if (rot_out_ || n_res_ > 0 || n_cols_ > 0) {
    ensure(ctx, ws.y32, ws.y32_bytes, N * out_ * 4);
    ensure(ctx, ws.y32b, ws.y32b_bytes, N * out_ * 4);
  }
}

// encode() between its barriers (see the header). Buffers: xr = ws.xr after the input rotation; the lookup writes y
// directly (no output rotation, no residual) or a = ws.y32; the output rotation writes y (no residual) or ws.y32b, which
// then becomes a; the column residual reads a and writes y; the CSR residual writes b, cast to y in the last stage.
bool SwitchLinear::encode_stage(int st, Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef eidx, BufRef y,
                                std::size_t N, Act act, Workspace& ws, bool masked, const Tiles* tiles) const {
  if (!N) return false;
  const bool res = n_res_ > 0 || n_cols_ > 0, direct = !rot_out_ && !res;
  if (masked && !routed_writes()) fail(name_ + ": masked writes need a column residual over an offset table");
  const BufRef xr = rot_in_ ? BufRef{ws.xr.get(), 0} : x;
  BufRef a{ws.y32.get(), 0}, b{ws.y32b.get(), 0};
  if (rot_out_ && res) std::swap(a, b);
  switch (st) {
    case 0:
      if (!rot_in_) return false;
      encode_rotation(ctx, enc, x, eidx, xr, N, false, act, act);
      return true;
    case 1:
      if (direct) encode_kernel(ctx, enc, xr, eidx, y, N, act, act, tiles);
      else encode_kernel(ctx, enc, xr, eidx, BufRef{ws.y32.get(), 0}, N, act, Act::F32, tiles);
      return true;
    case 2:
      if (!rot_out_) return false;
      if (!res) encode_rotation(ctx, enc, BufRef{ws.y32.get(), 0}, eidx, y, N, true, Act::F32, act);
      else encode_rotation(ctx, enc, BufRef{ws.y32.get(), 0}, eidx, BufRef{ws.y32b.get(), 0}, N, true, Act::F32, Act::F32);
      return true;
    case 3:
      if (!res) return false;
      if (n_cols_) {
        encode_column_residual(ctx, enc, x, eidx, a, y, N, act, masked);
        return true;
      }
      {
        const std::string targs = join({act_msl(act), rcols_u16_ ? "uint16_t" : "uint32_t", rvals_i8_ ? "int8_t" : "half",
                                        I(long(in_)), I(long(out_))});
        enc->setComputePipelineState(ctx.pipeline("suc_switch_residual", targs));
        set(enc, x, 0);
        set(enc, a, 1);
        enc->setBuffer(row_ptr_.get(), 0, 2);
        enc->setBuffer(rcols_.get(), 0, 3);
        enc->setBuffer(rvals_.get(), 0, 4);
        enc->setBuffer(rscale_.get(), 0, 5);
        set(enc, eidx, 6);
        set(enc, b, 7);
        const std::size_t tx = std::min<std::size_t>(256, out_);
        enc->dispatchThreads(MTL::Size((out_ + tx - 1) / tx * tx, N, 1), MTL::Size(tx, 1, 1));
      }
      return true;
    case 4: {
      if (!res || n_cols_) return false;
      const uint32_t cnt = uint32_t(N * out_);
      enc->setComputePipelineState(ctx.pipeline("suc_cast", join({"float", act_msl(act)})));
      set(enc, b, 0);
      set(enc, y, 1);
      enc->setBytes(&cnt, 4, 2);
      enc->dispatchThreads(MTL::Size(cnt, 1, 1), MTL::Size(std::min<uint32_t>(256, cnt), 1, 1));
      return true;
    }
    default:
      return false;
  }
}

// = suc_switch.switch_structured_residual_metal (launch shape of suc_switch.colres_launch). masked: the same body with
// rows of empty slots skipped (ds4_switch_colres_masked, cpp/metal/suc_ds4_kernels.metal).
void SwitchLinear::encode_column_residual(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef eidx,
                                          BufRef y32, BufRef y, std::size_t N, Act act, bool masked) const {
  const bool has_csr = n_res_ > 0;
  const int rq = out_ % 4 == 0 ? 4 : 1;
  std::size_t js = 8;
  while (js > 1 && js > cmax_) js /= 2;
  const std::size_t red = js > 1 ? js * 32 * rq : 1;
  const char* trc = has_csr ? (rcols_u16_ ? "uint16_t" : "uint32_t") : "uint32_t";
  const char* trv = has_csr ? (rvals_i8_ ? "int8_t" : "half") : "int8_t";
  const std::string targs = join({act_msl(act), ccols_u16_ ? "uint16_t" : "uint32_t", cval_i8_ ? "int8_t" : "half",
                                  trc, trv, I(long(in_)), I(long(out_)), I(long(cmax_)), I(rq), I(long(js)),
                                  I(long(red)), has_csr ? "true" : "false", act_msl(act)});
  enc->setComputePipelineState(ctx.pipeline(masked ? "ds4_switch_colres_masked" : "suc_switch_structured_residual", targs));
  if (masked) enc->setBuffer(eoff_.get(), 0, 12);
  const BufRef d = ctx.dummy();
  set(enc, x, 0);
  set(enc, y32, 1);
  set(enc, eidx, 2);
  enc->setBuffer(cptr_.get(), 0, 3);
  enc->setBuffer(ccols_.get(), 0, 4);
  enc->setBuffer(cval_.get(), 0, 5);
  enc->setBuffer(cscale_.get(), 0, 6);
  set(enc, has_csr ? BufRef{row_ptr_.get(), 0} : d, 7);
  set(enc, has_csr ? BufRef{rcols_.get(), 0} : d, 8);
  set(enc, has_csr ? BufRef{rvals_.get(), 0} : d, 9);
  enc->setBuffer(rscale_.get(), 0, 10);
  set(enc, y, 11);
  const std::size_t n_tg = (out_ + 32 * rq - 1) / (32 * rq);
  enc->dispatchThreads(MTL::Size(n_tg * 32 * js, N, 1), MTL::Size(32 * js, 1, 1));
}

namespace {
void run(Context& ctx, const std::function<void(MTL::ComputeCommandEncoder*)>& body) {
  MTL::CommandBuffer* cb = ctx.queue()->commandBuffer();
  MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  body(enc);
  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();
  if (cb->status() != MTL::CommandBufferStatusCompleted)
    fail(std::string("command buffer failed: ") +
         (cb->error() ? cb->error()->localizedDescription()->utf8String() : "?"));
}
}  // namespace

std::vector<float> SwitchLinear::forward(Context& ctx, const std::vector<float>& x, const std::vector<uint32_t>& eidx,
                                         Act act) const {
  const std::size_t N = eidx.size();
  if (x.size() != N * in_) fail("SwitchLinear::forward: x has " + std::to_string(x.size()) + " values, expected N * in");
  for (uint32_t e : eidx)
    if (e >= E_) fail("SwitchLinear::forward: expert id " + std::to_string(e) + " out of range");
  Pool pool;
  BufPtr xb = ctx.new_buffer(N * in_ * act_size(act));
  f32_to_act(x.data(), xb->contents(), x.size(), act);
  BufPtr eb = ctx.upload(eidx.data(), N * 4);
  BufPtr yb = ctx.new_buffer(N * out_ * act_size(act));
  Workspace ws;
  run(ctx, [&](MTL::ComputeCommandEncoder* enc) { encode(ctx, enc, {xb.get(), 0}, {eb.get(), 0}, {yb.get(), 0}, N, act, ws); });
  std::vector<float> y(N * out_);
  act_to_f32(yb->contents(), y.data(), y.size(), act);
  return y;
}

// ------------------------------------------------------------------------------------------ MoeBlock

std::size_t MoeScratch::bytes() const {
  return gates_b + eidx_b + scores_b + xk_b + g_b + u_b + h_b + yk_b + ykr_b + ws_gate.xr_bytes + ws_gate.y32_bytes +
         ws_gate.y32b_bytes + ws_up.xr_bytes + ws_up.y32_bytes + ws_up.y32b_bytes + ws_down.xr_bytes +
         ws_down.y32_bytes + ws_down.y32b_bytes;
}

MoeBlock MoeBlock::from_file(Context& ctx, const MappedFile& mf, const std::string& layer_prefix, const MoeConfig& cfg,
                             Act act, StackLayout layout) {
  const Suc1File& f = mf.file();
  if (!cfg.hidden || !cfg.num_experts || !cfg.top_k || !cfg.intermediate || cfg.top_k > cfg.num_experts)
    throw FormatError(layer_prefix + ": invalid MoE config");
  if (cfg.num_experts > 4096) throw FormatError(layer_prefix + ": more than 4096 experts (MLX softmax_looped)");
  MoeBlock B;
  B.cfg_ = cfg;
  B.act_ = act;
  B.prefix_ = layer_prefix;
  const std::string p = layer_prefix + ".mlp";
  const TensorDesc& r = f.tensor(p + ".gate.weight");
  if (r.shape != std::vector<uint64_t>{cfg.num_experts, cfg.hidden})
    throw FormatError(r.name + ": router shape != (num_experts, hidden_size)");
  if (r.representation_type == "raw" && r.arrays.size() == 1 && act_of(r.arrays[0].dtype) == act) {
    B.router_ = mf.at(r.arrays[0]);
  } else {   // = load_suc1_model: raw as stored / other codecs as their reference decode, cast to the model dtype
    const DecodedTensor d = decode_tensor(f, r);
    B.router_owned_ = ctx.new_buffer(d.data.size() * act_size(act));
    f32_to_act(d.data.data(), B.router_owned_->contents(), d.data.size(), act);
    B.router_ = {B.router_owned_.get(), 0};
  }
  B.gate_ = SwitchLinear::from_file(ctx, mf, p, "gate_proj", cfg.num_experts, layout);
  B.up_ = SwitchLinear::from_file(ctx, mf, p, "up_proj", cfg.num_experts, layout);
  B.down_ = SwitchLinear::from_file(ctx, mf, p, "down_proj", cfg.num_experts, layout);
  for (const SwitchLinear* s : {&B.gate_, &B.up_})
    if (s->out_features() != cfg.intermediate || s->in_features() != cfg.hidden)
      throw FormatError(s->name() + ": shape != (moe_intermediate_size, hidden_size)");
  if (B.down_.out_features() != cfg.hidden || B.down_.in_features() != cfg.intermediate)
    throw FormatError(B.down_.name() + ": shape != (hidden_size, moe_intermediate_size)");
  return B;
}

std::size_t MoeBlock::copied_bytes() const {
  return gate_.copied_bytes() + up_.copied_bytes() + down_.copied_bytes() +
         (router_owned_ ? cfg_.num_experts * cfg_.hidden * act_size(act_) : 0);
}

std::size_t MoeBlock::weight_bytes() const {
  return gate_.weight_bytes() + up_.weight_bytes() + down_.weight_bytes() + cfg_.num_experts * cfg_.hidden * act_size(act_);
}

void MoeBlock::encode_route(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef xn, BufRef eidx, BufRef scores,
                            std::size_t n, MoeScratch& s) const {
  const std::size_t H = cfg_.hidden, E = cfg_.num_experts, K = cfg_.top_k;
  ensure(ctx, s.gates, s.gates_b, n * E * act_size(act_));
  // gates = x Wᵀ: MLX gemv (non-transposed matrix) tiling: K <= 64: SN = 4 (8 rows per simdgroup); K >= 16·out:
  // 8 simdgroups along K; else 1. TN = 4 (checked bit for bit against MLX in tests/unit/test_cpp_moe.py).
  const int sn = H <= 64 ? 4 : 32;
  const int bn = (H > 64 && H >= 16 * E) ? 8 : 1;
  const int sm = 32 / sn;
  enc->setComputePipelineState(ctx.pipeline(
      "suc_moe_gemv", join({act_msl(act_), act_msl(act_), I(long(H)), I(long(E)), I(sn), "4", I(bn)})));
  set(enc, xn, 0);
  set(enc, router_, 1);
  enc->setBuffer(s.gates.get(), 0, 2);
  enc->dispatchThreadgroups(MTL::Size((E + sm - 1) / sm, n, 1), MTL::Size(32 * bn, 1, 1));
  barrier(enc);
  const std::size_t tpg = 32 * (((E + 3) / 4 + 31) / 32);
  enc->setComputePipelineState(ctx.pipeline(
      "suc_moe_route", join({act_msl(act_), I(long(E)), I(long(K)), I(cfg_.norm_topk_prob), I(long(tpg))})));
  enc->setBuffer(s.gates.get(), 0, 0);
  set(enc, eidx, 1);
  set(enc, scores, 2);
  enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(tpg, 1, 1));
}

void MoeBlock::encode_experts(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef xn, BufRef eidx, BufRef scores,
                              BufRef y, std::size_t n, MoeScratch& s) const {
  const std::size_t H = cfg_.hidden, K = cfg_.top_k, F = cfg_.intermediate, N = n * K;
  const std::size_t as = act_size(act_);
  ensure(ctx, s.xk, s.xk_b, N * H * as);
  ensure(ctx, s.g, s.g_b, N * F * as);
  ensure(ctx, s.u, s.u_b, N * F * as);
  ensure(ctx, s.h, s.h_b, N * F * as);
  ensure(ctx, s.yk, s.yk_b, N * H * as);
  // x2: each token row repeated for its K experts (SucSwitchLinear.__call__ broadcast, row n·K + j)
  const uint32_t cols = uint32_t(H), rep = uint32_t(K);
  enc->setComputePipelineState(ctx.pipeline("suc_repeat_rows", act_msl(act_)));
  set(enc, xn, 0);
  enc->setBuffer(s.xk.get(), 0, 1);
  enc->setBytes(&cols, 4, 2);
  enc->setBytes(&rep, 4, 3);
  enc->dispatchThreads(MTL::Size(H, N, 1), MTL::Size(std::min<std::size_t>(256, H), 1, 1));
  barrier(enc);
  const BufRef xk{s.xk.get(), 0}, g{s.g.get(), 0}, u{s.u.get(), 0}, h{s.h.get(), 0}, yk{s.yk.get(), 0};
  up_.encode(ctx, enc, xk, eidx, u, N, act_, s.ws_up);
  gate_.encode(ctx, enc, xk, eidx, g, N, act_, s.ws_gate);
  barrier(enc);
  // SwitchGLU activation(x_up, x_gate) = swiglu(x_gate, x_up) = silu(gate) · up
  const uint32_t cnt = uint32_t(N * F);
  enc->setComputePipelineState(ctx.pipeline("suc_swiglu", act_msl(act_)));
  set(enc, g, 0);
  set(enc, u, 1);
  set(enc, h, 2);
  enc->setBytes(&cnt, 4, 3);
  enc->dispatchThreads(MTL::Size(cnt, 1, 1), MTL::Size(std::min<uint32_t>(256, cnt), 1, 1));
  barrier(enc);
  down_.encode(ctx, enc, h, eidx, yk, N, act_, s.ws_down);
  barrier(enc);
  enc->setComputePipelineState(ctx.pipeline("suc_moe_combine", join({act_msl(act_), I(long(K)), I(long(H))})));
  set(enc, yk, 0);
  set(enc, scores, 1);
  set(enc, y, 2);
  enc->dispatchThreads(MTL::Size(H, n, 1), MTL::Size(std::min<std::size_t>(256, H), 1, 1));
}

std::vector<std::pair<uint32_t, uint32_t>> tile_runs(const uint32_t* teidx, std::size_t count, std::size_t max_tiles) {
  std::vector<std::pair<uint32_t, uint32_t>> r;
  for (std::size_t t = 0; t < count;) {
    std::size_t u = t + 1;
    while (u < count && teidx[u] == teidx[t] && u - t < max_tiles) ++u;
    r.push_back({uint32_t(t), uint32_t(u - t)});
    t = u;
  }
  return r;
}

TilePlan plan_tiles(const std::vector<std::pair<uint32_t, uint32_t>>& rows, std::size_t max_bt,
                    const std::vector<uint32_t>* rank, double alpha_arg) {
  TilePlan P;
  if (rows.empty()) return P;
  max_bt = std::max<std::size_t>(1, std::min<std::size_t>(max_bt, 16));
  std::vector<std::pair<uint32_t, uint32_t>> r(rows.begin(), rows.end());   // (expert, row) sorted, rows stable
  for (auto& x : r) std::swap(x.first, x.second);
  if (rank) {
    for (const auto& x : r)
      if (x.first >= rank->size()) throw std::out_of_range("plan_tiles: expert id beyond the rank table");
    std::stable_sort(r.begin(), r.end(), [&](const auto& a, const auto& b) {
      const uint32_t ra = (*rank)[a.first], rb = (*rank)[b.first];
      return ra != rb ? ra < rb : a.first < b.first;
    });
  } else {
    std::stable_sort(r.begin(), r.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  }
  std::vector<std::size_t> counts;
  for (std::size_t i = 0; i < r.size();) {
    std::size_t j = i;
    while (j < r.size() && r[j].first == r[i].first) ++j;
    counts.push_back(j - i);
    i = j;
  }
  static const char* alpha_env = std::getenv("SUC_TILE_ALPHA");
  const double alpha = alpha_env ? std::atof(alpha_env) : alpha_arg >= 0.0 ? alpha_arg : 0.3;
  static const std::size_t force = [] { const char* v = std::getenv("SUC_TILE_BT"); return v ? std::size_t(std::atoi(v)) : 0; }();
  std::size_t bt = 1;
  if (force) {
    while (bt * 2 <= std::min(force, max_bt)) bt *= 2;
  } else {
    double best = -1.0;
    for (std::size_t b = 1; b <= max_bt; b *= 2) {
      std::size_t tiles = 0;
      for (std::size_t c : counts) tiles += (c + b - 1) / b;
      const double cost = double(tiles) * (1.0 + alpha * double(b - 1));
      if (best < 0 || cost < best) best = cost, bt = b;
    }
  }
  P.bt = bt;
  for (std::size_t i = 0; i < r.size();) {
    std::size_t j = i;
    while (j < r.size() && r[j].first == r[i].first) ++j;
    for (std::size_t t0 = i; t0 < j; t0 += bt) {
      const uint32_t e = r[i].first;
      P.teidx.push_back(e);
      for (std::size_t b = 0; b < bt; ++b) {
        const bool live = t0 + b < j;
        P.src.push_back(live ? r[t0 + b].second : r[t0].second);
        P.dst.push_back(live ? r[t0 + b].second : ~0u);
        P.eidx.push_back(e);
      }
    }
    i = j;
  }
  return P;
}

std::size_t MoeBlock::fast_tile_rows() const { return gate_.tcq() || up_.tcq() || down_.tcq() ? 8 : 16; }

std::size_t MoeBlock::max_tile_rows(const Context& ctx) const {
  return std::min({gate_.max_tile_rows(ctx), up_.max_tile_rows(ctx), down_.max_tile_rows(ctx)});
}

void MoeBlock::encode_experts_tiled(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef xn, BufRef tok, BufRef dst,
                                    BufRef reid, const Tiles& tiles, BufRef scores, BufRef y, std::size_t n,
                                    MoeScratch& s, bool combine) const {
  const std::size_t H = cfg_.hidden, K = cfg_.top_k, F = cfg_.intermediate, N = tiles.count * tiles.bt;
  const std::size_t as = act_size(act_);
  if ((H * as) % 4) fail("encode_experts_tiled: hidden row bytes must be a multiple of 4");
  ensure(ctx, s.xk, s.xk_b, N * H * as);
  ensure(ctx, s.g, s.g_b, N * F * as);
  ensure(ctx, s.u, s.u_b, N * F * as);
  ensure(ctx, s.h, s.h_b, N * F * as);
  ensure(ctx, s.yk, s.yk_b, N * H * as);
  ensure(ctx, s.ykr, s.ykr_b, n * K * H * as);
  const uint32_t words = uint32_t(H * as / 4);
  enc->setComputePipelineState(ctx.pipeline("suc_gather_words", "uint32_t"));
  set(enc, xn, 0);
  enc->setBuffer(s.xk.get(), 0, 1);
  set(enc, tok, 2);
  enc->setBytes(&words, 4, 3);
  enc->dispatchThreads(MTL::Size(words, N, 1), MTL::Size(std::min<uint32_t>(256, words), 1, 1));
  barrier(enc);
  const BufRef xk{s.xk.get(), 0}, g{s.g.get(), 0}, u{s.u.get(), 0}, h{s.h.get(), 0}, yk{s.yk.get(), 0},
      ykr{s.ykr.get(), 0};
  up_.encode(ctx, enc, xk, reid, u, N, act_, s.ws_up, &tiles);
  gate_.encode(ctx, enc, xk, reid, g, N, act_, s.ws_gate, &tiles);
  barrier(enc);
  const uint32_t cnt = uint32_t(N * F);
  enc->setComputePipelineState(ctx.pipeline("suc_swiglu", act_msl(act_)));
  set(enc, g, 0);
  set(enc, u, 1);
  set(enc, h, 2);
  enc->setBytes(&cnt, 4, 3);
  enc->dispatchThreads(MTL::Size(cnt, 1, 1), MTL::Size(std::min<uint32_t>(256, cnt), 1, 1));
  barrier(enc);
  down_.encode(ctx, enc, h, reid, yk, N, act_, s.ws_down, &tiles);
  barrier(enc);
  enc->setComputePipelineState(ctx.pipeline("suc_scatter_words", "uint32_t"));
  set(enc, yk, 0);
  set(enc, ykr, 1);
  set(enc, dst, 2);
  enc->setBytes(&words, 4, 3);
  enc->dispatchThreads(MTL::Size(words, N, 1), MTL::Size(std::min<uint32_t>(256, words), 1, 1));
  barrier(enc);
  (void)ykr;
  if (combine) encode_combine(ctx, enc, scores, y, n, s);
}

void MoeBlock::encode_combine(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef scores, BufRef y, std::size_t n,
                              MoeScratch& s) const {
  const std::size_t H = cfg_.hidden, K = cfg_.top_k;
  enc->setComputePipelineState(ctx.pipeline("suc_moe_combine", join({act_msl(act_), I(long(K)), I(long(H))})));
  enc->setBuffer(s.ykr.get(), 0, 0);
  set(enc, scores, 1);
  set(enc, y, 2);
  enc->dispatchThreads(MTL::Size(H, n, 1), MTL::Size(std::min<std::size_t>(256, H), 1, 1));
}

void MoeBlock::attach_stream(ExpertStream& s) {
  const std::size_t E = cfg_.num_experts;
  for (const SwitchLinear* w : {&gate_, &up_, &down_})
    if (w->layout() != StackLayout::Streamed) fail(w->name() + ": attach_stream needs StackLayout::Streamed stacks");
  if (stream_) fail(prefix_ + ": already attached to an expert stream");
  std::vector<std::vector<ExpertStream::Range>> r(E);
  for (std::size_t e = 0; e < E; ++e)
    for (const SwitchLinear* w : {&gate_, &up_, &down_}) {
      const auto [pk, sc] = w->expert_ranges(e);
      r[e].push_back(pk);
      r[e].push_back(sc);
    }
  stream_layer_ = s.add_layer(prefix_ + ".mlp", std::move(r), cfg_.top_k,
                              [this](uint32_t e, MTL::Buffer* pool, const uint64_t* at) {
                                gate_.bind_expert(e, pool, at[0], at[1]);
                                up_.bind_expert(e, pool, at[2], at[3]);
                                down_.bind_expert(e, pool, at[4], at[5]);
                              });
  stream_ = &s;
}

void MoeBlock::stream_in(const uint32_t* eidx, std::size_t count) const {
  if (!stream_) fail(prefix_ + ": stream_in without an attached expert stream");
  stream_->fetch(stream_layer_, eidx, count);
}

void MoeBlock::stream_prefetch(const uint32_t* eidx, std::size_t count) const {
  if (!stream_) fail(prefix_ + ": stream_prefetch without an attached expert stream");
  stream_->prefetch(stream_layer_, eidx, count);
}

void MoeBlock::encode(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef xn, BufRef y, std::size_t n,
                      MoeScratch& s) const {
  if (!n) return;
  if (stream_)
    fail(prefix_ + ": a streamed block needs the host between routing and experts (encode_route, wait, stream_in, "
                   "encode_experts)");
  const std::size_t K = cfg_.top_k;
  ensure(ctx, s.eidx, s.eidx_b, n * K * 4);
  ensure(ctx, s.scores, s.scores_b, n * K * act_size(act_));
  const BufRef e{s.eidx.get(), 0}, sc{s.scores.get(), 0};
  encode_route(ctx, enc, xn, e, sc, n, s);
  barrier(enc);
  encode_experts(ctx, enc, xn, e, sc, y, n, s);
}

}  // namespace suc::metal
