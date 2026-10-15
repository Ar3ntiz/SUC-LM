// Native MXFP4 expert stacks and mixed-codec expert stacks (see suc/expert_mixed.hpp).
#include "suc/expert_mixed.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <tuple>

#include "suc/json.hpp"

namespace suc::metal {

namespace {

void barrier(MTL::ComputeCommandEncoder* enc) { enc->memoryBarrier(MTL::BarrierScopeBuffers); }
void set(MTL::ComputeCommandEncoder* enc, BufRef r, int i) { enc->setBuffer(r.buffer, r.offset, i); }

void ensure(Context& ctx, BufPtr& b, std::size_t& cap, std::size_t bytes) {
  if (cap < bytes) {
    b = ctx.new_buffer(bytes);
    cap = bytes;
  }
}

std::string expert_name(const std::string& prefix, std::size_t e, const std::string& proj) {
  return SwitchLinear::expert_name(prefix, e, proj);
}

bool param_bool(const Json& p, std::string_view key, bool dflt) {
  const Json* v = p.find(key);
  return (v && !v->is_null()) ? v->as_bool() : dflt;
}

// Code layout of a vq / additive / tcq expert (= suc_switch._layout of its base, plus the residual value dtypes, which one
// SwitchLinear requires to agree).
std::string layout_key(const Suc1File& f, const TensorDesc& t) {
  const Json& p = t.has_residual() ? (*t.params)["base"] : *t.params;
  const bool rot = param_bool(p, "rotate", false);
  std::string k;
  if (t.representation_type == "tcq") {   // suc_switch._layout of a tcq tensor: (kv, k, L, V, T, code, gain, mu, sd)
    k = "tcq|" + TcqLayout::from_params(p, t.shape.at(1), t.name).key + "|" +
        std::to_string(rot && param_bool(p, "rotate_in", true)) + std::to_string(rot && param_bool(p, "rotate_out", false));
  } else {
    const CodebookDesc& c = f.codebooks().at(t.codebook_ids.at(0));
    k = t.representation_type + "|" + std::to_string(t.index_bits.value_or(0)) + "|" +
        std::to_string(t.codebook_ids.size()) + "x" + std::to_string(c.entries) + "x" +
        std::to_string(c.vector_length) + "|" + std::to_string(rot && param_bool(p, "rotate_in", true)) +
        std::to_string(rot && param_bool(p, "rotate_out", true));
  }
  if (t.has_residual()) {
    if (const ArrayRecord* v = t.find_array("res_val"); v && v->logical_numel()) k += "|csr" + std::string(dtype_name(v->dtype));
    if (const ArrayRecord* v = t.find_array("res_colval"); v && v->logical_numel()) k += "|col" + std::string(dtype_name(v->dtype));
  }
  return k;
}

}  // namespace

// ------------------------------------------------------------------------------------------ Mxfp4Switch

Mxfp4Switch Mxfp4Switch::from_file(Context& ctx, const MappedFile& mf, const std::string& prefix,
                                   const std::string& proj, const std::vector<int64_t>& slots, bool streamed) {
  const Suc1File& f = mf.file();
  Mxfp4Switch S;
  S.streamed_ = streamed;
  const std::size_t E = slots.size();
  std::vector<const TensorDesc*> ts(E, nullptr);
  uint64_t clo = UINT64_MAX, chi = 0, slo = UINT64_MAX, shi = 0;
  for (std::size_t e = 0; e < E; ++e) {
    if (slots[e] < 0) continue;
    if (streamed && slots[e] != int64_t(e)) throw FormatError(prefix + "." + proj + ": streamed slots must be identity or empty");
    const std::string n = expert_name(prefix, std::size_t(slots[e]), proj);
    const TensorDesc& t = f.tensor(n);
    if (t.representation_type != "mxfp4") throw FormatError(n + ": expected an mxfp4 tensor");
    if (t.shape.size() != 2) throw FormatError(n + ": mxfp4 tensor is not 2-D");
    const Json& p = *t.params;
    if (const Json* g = p.find("group_size"); g && g->as_uint() != 32) throw FormatError(n + ": mxfp4 group_size != 32");
    if (!S.members_) {
      S.out_ = t.shape[0];
      S.in_ = t.shape[1];
    } else if (t.shape[0] != S.out_ || t.shape[1] != S.in_) {
      throw FormatError(n + ": experts of one stack differ in shape");
    }
    if (S.in_ % 32) throw FormatError(n + ": in_features not a multiple of 32");
    const ArrayRecord& c = t.array("codes");
    const ArrayRecord& s = t.array("scales");
    if (c.dtype != DType::U8 || c.nbytes != S.out_ * S.in_ / 2) throw FormatError(n + ": mxfp4 codes layout");
    if (s.dtype != DType::U8 || s.nbytes != S.out_ * S.in_ / 32) throw FormatError(n + ": mxfp4 scales layout");
    clo = std::min(clo, c.offset), chi = std::max(chi, c.offset + c.nbytes);
    slo = std::min(slo, s.offset), shi = std::max(shi, s.offset + s.nbytes);
    ts[e] = &t;
    ++S.members_;
    S.weight_bytes_ += c.nbytes + s.nbytes;
  }
  if (!S.members_) throw FormatError(prefix + "." + proj + ": an mxfp4 stack without experts");
  std::vector<uint64_t> eoff(2 * E, ~uint64_t(0));
  if (streamed) {   // bound to slot pool offsets by bind_expert (never mapped for the GPU)
    S.codes_ = S.scales_ = ctx.dummy();
    S.c_off_.assign(E, ~uint64_t(0));
    S.s_off_.assign(E, ~uint64_t(0));
    for (std::size_t e = 0; e < E; ++e)
      if (ts[e]) S.c_off_[e] = ts[e]->array("codes").offset, S.s_off_[e] = ts[e]->array("scales").offset;
  } else {
    S.codes_ = mf.at(clo, chi - clo);
    S.scales_ = mf.at(slo, shi - slo);
    for (std::size_t e = 0; e < E; ++e) {
      if (!ts[e]) continue;
      eoff[2 * e] = ts[e]->array("codes").offset - clo;
      eoff[2 * e + 1] = ts[e]->array("scales").offset - slo;
    }
  }
  S.eoff_ = ctx.upload(eoff.data(), eoff.size() * 8);
  S.copied_bytes_ = eoff.size() * 8;
  // MLX's choice (quantized.cpp gather_qmv): the fast kernel when K % 512 == 0 and N % 8 == 0
  S.fast_ = S.in_ % 512 == 0 && S.out_ % 8 == 0;
  return S;
}

std::pair<std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>> Mxfp4Switch::expert_ranges(std::size_t e) const {
  if (!streamed_) throw std::logic_error("Mxfp4Switch::expert_ranges needs a streamed stack");
  if (e >= c_off_.size() || c_off_[e] == ~uint64_t(0)) throw std::logic_error("Mxfp4Switch::expert_ranges: empty slot");
  return {{c_off_[e], out_ * in_ / 2}, {s_off_[e], out_ * in_ / 32}};
}

void Mxfp4Switch::bind_expert(std::size_t e, MTL::Buffer* pool, uint64_t codes_at, uint64_t scales_at) {
  if (!streamed_) throw std::logic_error("Mxfp4Switch::bind_expert needs a streamed stack");
  if (e >= c_off_.size() || c_off_[e] == ~uint64_t(0)) throw std::logic_error("Mxfp4Switch::bind_expert: empty slot");
  if (codes_at % 4) throw std::logic_error("Mxfp4Switch::bind_expert: misaligned codes offset");   // read as uint32
  if (codes_.buffer != pool || scales_.buffer != pool) codes_ = scales_ = {pool, 0};
  auto* t = static_cast<uint64_t*>(eoff_->contents());
  t[2 * e] = codes_at;
  t[2 * e + 1] = scales_at;
}

void Mxfp4Switch::encode(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef eidx, BufRef y,
                         std::size_t N, Act act, const Tiles* tiles) const {
  if (!N) return;
  if (tiles && tiles->bt > 1) {
    if (N != tiles->count * tiles->bt) throw std::runtime_error("mxfp4 switch: tiled rows must be count * bt");
    const int bt = int(tiles->bt);
    static const int mm_min = [] { const char* e = std::getenv("SUC_MXFP4_MM_MIN"); return e ? std::atoi(e) : 8; }();
    static const bool runs_on = [] { const char* e = std::getenv("SUC_MXFP4_RUNS"); return !(e && e[0] == '0'); }();
    if (!tiles->exact && bt >= mm_min && bt <= 16) {   // batched prefill: simdgroup-matrix tiles (not row-exact)
      // runs of consecutive tiles of one expert (weights decoded once per run of <= 64 rows) when the host plan is known
      std::vector<std::pair<uint32_t, uint32_t>> runs;
      if (tiles->host_teidx && runs_on) runs = tile_runs(tiles->host_teidx, tiles->count, std::size_t(64 / bt));
      std::size_t mx = 1;
      for (const auto& r : runs) mx = std::max<std::size_t>(mx, r.second);
      const int rn = runs.empty() ? 0 : 1;
      const std::size_t rows = std::size_t(bt) * mx;
      const int bm = rows <= 8 ? 8 : rows <= 16 ? 16 : rows <= 32 ? 32 : 64;
      enc->setComputePipelineState(ctx.pipeline(PsoKey{2004, {int64_t(act), bm, bt, rn}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_mxfp4_mm_tile", std::string(act_msl(act)) + ", " +
                                                   std::to_string(bm) + ", " + std::to_string(bt) + ", " +
                                                   std::to_string(rn)};
      }));
      BufPtr rb = rn ? ctx.upload(runs.data(), runs.size() * 8) : BufPtr{};   // retained by the command buffer
      if (rn) enc->setBuffer(rb.get(), 0, 8);
      else set(enc, ctx.dummy(), 8);
      set(enc, codes_, 0);
      set(enc, scales_, 1);
      enc->setBuffer(eoff_.get(), 0, 2);
      set(enc, x, 3);
      set(enc, tiles->teidx, 4);
      set(enc, y, 5);
      const int in = int(in_), out = int(out_);
      enc->setBytes(&in, 4, 6);
      enc->setBytes(&out, 4, 7);
      enc->dispatchThreadgroups(MTL::Size(rn ? runs.size() : tiles->count, (out_ + 63) / 64, 1), MTL::Size(128, 1, 1));
      return;
    }
    enc->setComputePipelineState(ctx.pipeline(PsoKey{2003, {int64_t(act), fast_ ? 1 : 0, bt}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_mxfp4_qmv_tile",
                                                 std::string(act_msl(act)) + ", " + (fast_ ? "1" : "0") + ", " + std::to_string(bt)};
    }));
    set(enc, codes_, 0);
    set(enc, scales_, 1);
    enc->setBuffer(eoff_.get(), 0, 2);
    set(enc, x, 3);
    set(enc, tiles->teidx, 4);
    set(enc, y, 5);
    const int in = int(in_), out = int(out_);
    enc->setBytes(&in, 4, 6);
    enc->setBytes(&out, 4, 7);
    enc->dispatchThreadgroups(MTL::Size(1, (out_ + 7) / 8, tiles->count), MTL::Size(32, 2, 1));
    return;
  }
  enc->setComputePipelineState(ctx.pipeline(PsoKey{2001, {int64_t(act), fast_ ? 1 : 0}}, [&]() {
    return std::pair<std::string, std::string>{"ds4_mxfp4_qmv", std::string(act_msl(act)) + ", " + (fast_ ? "1" : "0")};
  }));
  set(enc, codes_, 0);
  set(enc, scales_, 1);
  enc->setBuffer(eoff_.get(), 0, 2);
  set(enc, x, 3);
  set(enc, eidx, 4);
  set(enc, y, 5);
  const int in = int(in_), out = int(out_);
  enc->setBytes(&in, 4, 6);
  enc->setBytes(&out, 4, 7);
  enc->dispatchThreadgroups(MTL::Size(1, (out_ + 7) / 8, N), MTL::Size(32, 2, 1));
}

// ------------------------------------------------------------------------------------------ ExpertStack

std::size_t ExpertScratch::bytes() const {
  std::size_t b = ysub_b;
  for (const Workspace& w : ws) b += w.xr_bytes + w.y32_bytes + w.y32b_bytes;
  return b;
}

ExpertStack ExpertStack::from_file(Context& ctx, const MappedFile& mf, const std::string& prefix,
                                   const std::string& proj, std::size_t num_experts, Rows rows, bool streamed) {
  const Suc1File& f = mf.file();
  const std::size_t E = num_experts ? num_experts : SwitchLinear::count_experts(f, prefix, proj);
  if (!E) throw FormatError(prefix + "." + proj + ": no expert tensors");
  ExpertStack S;
  S.E_ = E;
  S.rows_ = rows;
  S.streamed_ = streamed;
  S.nat_of_.assign(E, 0);
  S.grp_.assign(E, 0);
  // groups: native experts form one; vq / additive experts one per code layout (experts in order within each)
  std::vector<std::size_t> native;
  std::map<std::string, std::vector<std::size_t>> by_key;
  std::vector<std::string> key_order;
  for (std::size_t e = 0; e < E; ++e) {
    const std::string n = expert_name(prefix, e, proj);
    const TensorDesc* t = f.find_tensor(n);
    if (!t) throw FormatError(prefix + "." + proj + ": missing expert tensor " + n);
    if (t->representation_type == "mxfp4") {
      native.push_back(e);
      continue;
    }
    if (t->representation_type != "vq" && t->representation_type != "additive" && t->representation_type != "tcq")
      throw FormatError(n + ": expert representation '" + t->representation_type + "' has no expert-stack kernel");
    const std::string k = layout_key(f, *t);
    if (!by_key.count(k)) key_order.push_back(k);
    by_key[k].push_back(e);
  }
  std::vector<std::vector<std::size_t>> groups;
  for (const std::string& k : key_order) groups.push_back(by_key[k]);
  std::sort(groups.begin(), groups.end());   // as switch_from_encodings: sorted member lists
  S.native_ = native.size();
  const bool uniform = groups.size() + (native.empty() ? 0 : 1) == 1;
  auto slots_for = [&](const std::vector<std::size_t>& members) {
    std::vector<int64_t> s(E, rows == Rows::All ? int64_t(members[0]) : int64_t(-1));
    for (std::size_t e : members) s[e] = int64_t(e);
    return s;
  };
  if (streamed && rows == Rows::All && !uniform)
    throw std::invalid_argument(prefix + "." + proj + ": streamed mixed expert stacks need Rows::Routed (--mixed-experts "
                                "routed): Rows::All reads each sub-stack's first expert, which need not be resident");
  if (uniform && native.empty()) {
    S.sub_suc_.push_back(SwitchLinear::from_file(ctx, mf, prefix, proj, E,
                                                 streamed ? StackLayout::Streamed : StackLayout::Auto));
  } else {
    for (std::size_t g = 0; g < groups.size(); ++g) {
      S.sub_suc_.push_back(SwitchLinear::from_file_slots(ctx, mf, prefix, proj, slots_for(groups[g]), streamed));
      for (std::size_t e : groups[g]) S.grp_[e] = g;
    }
    if (!native.empty()) {
      S.sub_nat_.push_back(Mxfp4Switch::from_file(ctx, mf, prefix, proj, slots_for(native), streamed));
      for (std::size_t e : native) S.nat_of_[e] = 1, S.grp_[e] = 0;
    }
  }
  if (!S.sub_suc_.empty()) {
    S.in_ = S.sub_suc_[0].in_features();
    S.out_ = S.sub_suc_[0].out_features();
  } else {
    S.in_ = S.sub_nat_[0].in_features();
    S.out_ = S.sub_nat_[0].out_features();
  }
  for (const SwitchLinear& s : S.sub_suc_)
    if (s.in_features() != S.in_ || s.out_features() != S.out_) throw FormatError(prefix + "." + proj + ": expert shapes differ");
  for (const Mxfp4Switch& s : S.sub_nat_)
    if (s.in_features() != S.in_ || s.out_features() != S.out_) throw FormatError(prefix + "." + proj + ": expert shapes differ");
  if (S.mixed()) {
    std::vector<uint32_t> sel(E);
    for (std::size_t e = 0; e < E; ++e) sel[e] = uint32_t(S.nat_of_[e] ? S.sub_suc_.size() + S.grp_[e] : S.grp_[e]);
    S.sel_ = ctx.upload(sel.data(), sel.size() * 4);
  }
  return S;
}

std::pair<std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>> ExpertStack::expert_ranges(std::size_t e) const {
  if (e >= E_) throw std::out_of_range("ExpertStack::expert_ranges: expert id");
  return nat_of_[e] ? sub_nat_[grp_[e]].expert_ranges(e) : sub_suc_[grp_[e]].expert_ranges(e);
}

void ExpertStack::bind_expert(std::size_t e, MTL::Buffer* pool, uint64_t a0, uint64_t a1) {
  if (e >= E_) throw std::out_of_range("ExpertStack::bind_expert: expert id");
  if (nat_of_[e]) sub_nat_[grp_[e]].bind_expert(e, pool, a0, a1);
  else sub_suc_[grp_[e]].bind_expert(e, pool, a0, a1);
}

std::size_t ExpertStack::weight_bytes() const {
  std::size_t b = 0;
  for (const SwitchLinear& s : sub_suc_) b += s.weight_bytes();
  for (const Mxfp4Switch& s : sub_nat_) b += s.weight_bytes();
  return b;
}

std::size_t ExpertStack::copied_bytes() const {
  std::size_t b = sel_ ? E_ * 4 : 0;
  for (const SwitchLinear& s : sub_suc_) b += s.copied_bytes();
  for (const Mxfp4Switch& s : sub_nat_) b += s.copied_bytes();
  return b;
}

void ExpertStack::encode(Context& ctx, MTL::ComputeCommandEncoder* enc, BufRef x, BufRef eidx, BufRef y,
                         std::size_t N, Act act, ExpertScratch& s) const {
  if (!N) return;
  const std::size_t S = sub_stacks();
  if (s.ws.size() < S) s.ws.resize(S);
  if (!mixed()) {
    if (!sub_suc_.empty()) sub_suc_[0].encode(ctx, enc, x, eidx, y, N, act, s.ws[0]);
    else sub_nat_[0].encode(ctx, enc, x, eidx, y, N, act);
    return;
  }
  const std::size_t as = act_size(act);
  ensure(ctx, s.ysub, s.ysub_b, S * N * out_ * as);
  for (std::size_t i = 0; i < sub_suc_.size(); ++i)
    sub_suc_[i].encode(ctx, enc, x, eidx, {s.ysub.get(), i * N * out_ * as}, N, act, s.ws[i]);
  for (std::size_t i = 0; i < sub_nat_.size(); ++i)
    sub_nat_[i].encode(ctx, enc, x, eidx, {s.ysub.get(), (sub_suc_.size() + i) * N * out_ * as}, N, act);
  barrier(enc);
  enc->setComputePipelineState(ctx.pipeline(PsoKey{2002, {int64_t(act)}}, [&]() {
    return std::pair<std::string, std::string>{"ds4_mixed_select", act_msl(act)};
  }));
  enc->setBuffer(s.ysub.get(), 0, 0);
  set(enc, eidx, 1);
  enc->setBuffer(sel_.get(), 0, 2);
  set(enc, y, 3);
  const uint32_t n32 = uint32_t(N), o32 = uint32_t(out_);
  enc->setBytes(&n32, 4, 4);
  enc->setBytes(&o32, 4, 5);
  enc->dispatchThreads(MTL::Size(out_, N, 1), MTL::Size(std::min<std::size_t>(256, out_), 1, 1));
}

bool ExpertStack::routed_writes() const {
  if (!mixed()) return true;
  if (rows_ != Rows::Routed) return false;
  for (const SwitchLinear& s : sub_suc_)
    if (!s.routed_writes()) return false;
  return true;
}

std::size_t ExpertStack::max_tile_rows(const Context& ctx) const {
  std::size_t b = 16;
  for (const SwitchLinear& s : sub_suc_) b = std::min(b, s.max_tile_rows(ctx));
  return b;
}

std::size_t ExpertStack::fast_tile_rows() const {
  static const int env = [] { const char* e = std::getenv("SUC_FAST_TILE_ROWS"); return e ? std::atoi(e) : 0; }();
  return env >= 1 && env <= 16 ? std::size_t(env) : 16;
}

void ExpertStack::encode_group(Context& ctx, MTL::ComputeCommandEncoder* enc, const std::vector<Job>& jobs,
                               BufRef eidx, std::size_t N, Act act, const Tiles* tiles) {
  if (!N || jobs.empty()) return;
  struct Plan {
    const ExpertStack* st;
    BufRef x, y;
    ExpertScratch* s;
    bool select;
  };
  std::vector<Plan> plans;
  const std::size_t as = act_size(act);
  for (const Job& j : jobs) {
    const ExpertStack& S = *j.stack;
    const std::size_t n_sub = S.sub_stacks();
    if (j.scratch->ws.size() < n_sub) j.scratch->ws.resize(n_sub);
    const bool select = S.mixed() && !S.routed_writes();
    if (select) ensure(ctx, j.scratch->ysub, j.scratch->ysub_b, n_sub * N * S.out_ * as);
    for (std::size_t i = 0; i < S.sub_suc_.size(); ++i) S.sub_suc_[i].prepare(ctx, j.scratch->ws[i], N, act);
    plans.push_back({&S, j.x, j.y, j.scratch, select});
  }
  // output of sub-stack slot i (SUC sub-stacks first, then native) of a plan
  auto out = [&](const Plan& p, std::size_t i) -> BufRef {
    if (!p.st->mixed() || !p.select) return p.y;
    return {p.s->ysub.get(), i * N * p.st->out_ * as};
  };
  // Rows each sub-stack runs: all N rows, or (tiles with a host copy, mixed stack) the tile range [t0, t1) spanning
  // the tiles of its experts. Rows outside a sub-stack's own experts inside the range are handled exactly as before
  // (lookup / gather skip them, masked writes leave them, select reads them from their own sub-stack).
  static const bool all_tiles = [] { const char* e = std::getenv("SUC_MIXED_ALL_TILES"); return e && e[0] == '1'; }();
  struct Span {
    std::size_t r0 = 0, n = 0;   // rows
    Tiles t;
  };
  auto span = [&](const ExpertStack& S, std::size_t slot) -> Span {
    if (!tiles) return {0, N, {}};
    if (all_tiles || !tiles->host_teidx || !S.mixed()) return {0, N, *tiles};
    std::size_t t0 = tiles->count, t1 = 0;
    for (std::size_t t = 0; t < tiles->count; ++t)
      if (S.sub_slot(tiles->host_teidx[t]) == slot) t0 = std::min(t0, t), t1 = t + 1;
    if (t1 <= t0) return {0, 0, {}};
    Tiles sub = *tiles;
    sub.teidx = tiles->teidx.at(t0 * 4);
    sub.count = t1 - t0;
    sub.host_teidx = tiles->host_teidx + t0;
    return {t0 * tiles->bt, sub.count * tiles->bt, sub};
  };
  std::vector<std::vector<Span>> spans;
  for (const Plan& p : plans) {
    std::vector<Span> v;
    for (std::size_t i = 0; i < p.st->sub_stacks(); ++i) v.push_back(span(*p.st, i));
    spans.push_back(std::move(v));
  }
  for (int st = 0; st < SwitchLinear::kStages; ++st) {
    bool any = false;
    for (std::size_t pi = 0; pi < plans.size(); ++pi) {
      const Plan& p = plans[pi];
      const ExpertStack& S = *p.st;
      const bool masked = S.mixed() && !p.select;
      for (std::size_t i = 0; i < S.sub_stacks(); ++i) {
        const Span& sp = spans[pi][i];
        if (!sp.n) continue;
        const BufRef x = p.x.at(sp.r0 * S.in_ * as), e = eidx.at(sp.r0 * 4), y = out(p, i).at(sp.r0 * S.out_ * as);
        const Tiles* t = tiles ? &sp.t : nullptr;
        if (i < S.sub_suc_.size()) {
          any |= S.sub_suc_[i].encode_stage(st, ctx, enc, x, e, y, sp.n, act, p.s->ws[i], masked, t);
        } else if (st == 1) {
          S.sub_nat_[i - S.sub_suc_.size()].encode(ctx, enc, x, e, y, sp.n, act, t);
          any = true;
        }
      }
    }
    if (any) barrier(enc);
  }
  bool sel = false;
  for (const Plan& p : plans) {
    if (!p.select) continue;
    const ExpertStack& S = *p.st;
    enc->setComputePipelineState(ctx.pipeline(PsoKey{2002, {int64_t(act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_mixed_select", act_msl(act)};
    }));
    enc->setBuffer(p.s->ysub.get(), 0, 0);
    set(enc, eidx, 1);
    enc->setBuffer(S.sel_.get(), 0, 2);
    set(enc, p.y, 3);
    const uint32_t n32 = uint32_t(N), o32 = uint32_t(S.out_);
    enc->setBytes(&n32, 4, 4);
    enc->setBytes(&o32, 4, 5);
    enc->dispatchThreads(MTL::Size(S.out_, N, 1), MTL::Size(std::min<std::size_t>(256, S.out_), 1, 1));
    sel = true;
  }
  if (sel) barrier(enc);
}

}  // namespace suc::metal
