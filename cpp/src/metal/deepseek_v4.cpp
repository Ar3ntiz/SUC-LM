// deepseek_v4 model graph on the Metal backend (see suc/deepseek_v4.hpp). Kernels: cpp/metal/suc_ds4_kernels.metal (the
// deepseek_v4 ops, MXFP4 experts), cpp/metal/suc_model_kernels.metal (embedding gather, MLX-order gemv, RMSNorm,
// argmax, casts) and the SUC projection / expert-stack kernels of metal::Linear / metal::SwitchLinear.
#include "suc/deepseek_v4.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>

#include "suc/bits.hpp"
#include "suc/decode.hpp"
#include "suc/expert_stream.hpp"
#include "suc/json.hpp"

namespace suc::metal {

namespace {

struct Pool {
  NS::AutoreleasePool* p = NS::AutoreleasePool::alloc()->init();
  ~Pool() { p->release(); }
};

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("deepseek_v4: " + msg); }

std::string I(long v) { return std::to_string(v); }
std::string join(std::initializer_list<std::string> xs) {
  std::string s;
  for (const auto& x : xs) s += (s.empty() ? "" : ", ") + x;
  return s;
}
const char* msl_of(DType t) {
  switch (t) {
    case DType::F16: return "half";
    case DType::BF16: return "bfloat";
    case DType::F32: return "float";
    case DType::I32: return "int";
    case DType::I64: return "long";
    case DType::U32: return "uint";
    default: throw FormatError("deepseek_v4: unsupported dtype " + std::string(dtype_name(t)));
  }
}
DType dtype_of(Act a) { return a == Act::BF16 ? DType::BF16 : a == Act::F16 ? DType::F16 : DType::F32; }

void barrier(MTL::ComputeCommandEncoder* enc) { enc->memoryBarrier(MTL::BarrierScopeBuffers); }
// Simdgroups (row blocks) per threadgroup of the affine gemv when its columns are not split (BN = 1); bit-identical for
// any value. Default 4 (M1 Ultra, DeepSeek V1: 1 -> 49.2, 2 -> 46.8, 4 -> 45.6, 8 -> 46.4 ms/token;
// docs/decode_speed_vs_ds4.md §8); SUC_AFFINE_SG overrides it.
std::size_t affine_rows_sg() {
  static const std::size_t v = [] {
    const char* e = std::getenv("SUC_AFFINE_SG");
    const int n = e ? std::atoi(e) : 0;
    return std::size_t(n > 0 && n <= 32 ? n : 4);
  }();
  return v;
}
void set(MTL::ComputeCommandEncoder* enc, BufRef r, int i) { enc->setBuffer(r.buffer, r.offset, i); }
template <class V>
void bytes(MTL::ComputeCommandEncoder* enc, const V& v, int i) { enc->setBytes(&v, sizeof v, i); }

std::size_t get_size(const Json& a, const char* key) {
  const Json* v = a.find(key);
  if (!v || !v->is_int()) throw FormatError(std::string("deepseek_v4 architecture: missing integer '") + key + "'");
  return std::size_t(v->as_uint());
}
std::size_t get_size_or(const Json& a, const char* key, std::size_t dflt) {
  const Json* v = a.find(key);
  return v && v->is_int() ? std::size_t(v->as_uint()) : dflt;
}
double get_num_or(const Json& a, const char* key, double dflt) {
  const Json* v = a.find(key);
  return v && v->is_number() ? v->as_double() : dflt;
}
bool get_bool_or(const Json& a, const char* key, bool dflt) {
  const Json* v = a.find(key);
  return v && v->is_bool() ? v->as_bool() : dflt;
}

// mlx_deepseek_v4.rope_inv_freq: numpy float32 arithmetic (powf, one rounding per op), YaRN ramp when scaled.
std::vector<float> rope_inv_freq(std::size_t dim, double base, bool yarn, double factor, std::size_t orig,
                                 double beta_fast, double beta_slow) {
  std::vector<float> f(dim / 2);
  for (std::size_t i = 0; i < dim / 2; ++i) {
    const float ex = float(2 * i) / float(dim);
    const float p = std::pow(float(base), ex);
    f[i] = 1.0f / p;
  }
  if (yarn && orig > 0) {
    auto corr = [&](double rot) { return double(dim) * std::log(double(orig) / (rot * 2 * M_PI)) / (2 * std::log(base)); };
    const double low = std::max(std::floor(corr(beta_fast)), 0.0);
    double high = std::min(std::ceil(corr(beta_slow)), double(dim - 1));
    if (low == high) high += 0.001;
    for (std::size_t i = 0; i < dim / 2; ++i) {
      float ramp = (float(i) - float(low)) / float(high - low);
      ramp = std::min(std::max(ramp, 0.0f), 1.0f);
      const float smooth = 1.0f - ramp;
      const float a = f[i] / float(factor);
      const float b = 1.0f - smooth;
      const float c = a * b;
      const float d = f[i] * smooth;
      f[i] = c + d;
    }
  }
  return f;
}

// (out, in) matrix: SUC codes (metal::Linear), affine codes, or a dense matrix (in place when stored in the wanted
// dtype, else converted once at load as the Python loader casts it).
struct Mat {
  enum Kind { Suc, Affine, Dense } kind = Dense;
  std::string name;
  std::size_t out = 0, in = 0;
  std::unique_ptr<Linear> lin;
  BufRef packed, scales, biases;
  unsigned bits = 0;
  std::size_t group = 0, words = 0;
  DType sdtype = DType::F16;
  BufRef w;
  DType wdtype = DType::BF16;
  BufPtr owned;
};

// A raw vector read in place in its stored dtype, or converted at load to the wanted dtype.
struct Vec {
  BufRef ref;
  DType dtype = DType::BF16;
  BufPtr owned;
  const char* msl() const { return msl_of(dtype); }
};

// MLX's GEMM dispatch for a folded matmul (M, N, K) (matmul.cpp; = glm5.cpp gemm_plan): split-k when K >= max(M, N)
// and K / 16 >= 8 (P = min(max(2, next_pow2((K / 16) / (ceil(M / 32) · ceil(N / 32)))), 32) partitions of
// ((K / 16) / P) · 16 values), else the fused kernel whose unaligned K tail (K % BK) runs first; BK by device class
// (last letter of the GPU architecture).
struct GemmPlan {
  std::size_t parts = 1, psize = 0, tail_bk = 16;
};
GemmPlan gemm_plan(char devc, std::size_t M, std::size_t N, std::size_t K, bool nt, bool f32) {
  GemmPlan g;
  const std::size_t tk = K / 16;
  if (K >= std::max(M, N) && tk >= 8) {
    const std::size_t tm = (M + 31) / 32, tn = (N + 31) / 32, q = tk / (tm * tn);
    std::size_t p2 = 1;
    while (p2 < q) p2 <<= 1;
    g.parts = std::min<std::size_t>(std::max<std::size_t>(2, p2), 32);
    g.psize = (tk / g.parts) * 16;
    return g;
  }
  if (devc == 'g' || devc == 'p') {
    g.tail_bk = nt ? 32 : 16;
  } else if (devc == 'd') {
    if (M * N >= (std::size_t(1) << 20)) g.tail_bk = f32 ? 16 : (2 * std::max(M, N) > K ? 16 : nt ? 32 : 16);
    else g.tail_bk = f32 ? (nt ? 16 : 32) : (nt ? 32 : 16);
  }
  return g;
}

// MLX Device::get_architecture_gen: the number after "applegpu_g" ("applegpu_g15g" -> 15); 0 when unknown.
int architecture_gen(const std::string& arch) {
  const auto p = arch.find("_g");
  if (p == std::string::npos) return 0;
  int g = 0;
  for (std::size_t i = p + 2; i < arch.size() && arch[i] >= '0' && arch[i] <= '9'; ++i) g = g * 10 + (arch[i] - '0');
  return g;
}

}  // namespace

// ------------------------------------------------------------------------------------------ config

bool Ds4Config::is_ds4(const Json& arch) {
  const Json* mt = arch.find("model_type");
  return mt && mt->is_string() && mt->as_string() == "deepseek_v4";
}

Ds4Config Ds4Config::from_architecture(const Json& a) {
  if (!is_ds4(a)) throw FormatError("architecture: model_type is not deepseek_v4");
  Ds4Config c;
  c.hidden = get_size(a, "hidden_size");
  c.layers = get_size(a, "num_hidden_layers");
  c.heads = get_size(a, "num_attention_heads");
  c.head_dim = get_size(a, "head_dim");
  c.rope_dim = get_size(a, "qk_rope_head_dim");
  c.vocab = get_size(a, "vocab_size");
  c.q_lora = get_size(a, "q_lora_rank");
  c.o_groups = get_size(a, "o_groups");
  c.o_lora = get_size(a, "o_lora_rank");
  c.window = get_size(a, "sliding_window");
  c.ratios.assign(c.layers, 0);
  if (const Json* r = a.find("compress_ratios"); r && r->is_array()) {
    if (r->size() < c.layers) throw FormatError("deepseek_v4: compress_ratios shorter than num_hidden_layers");
    for (std::size_t i = 0; i < c.layers; ++i) c.ratios[i] = std::size_t(r->at(i).as_uint());   // MTP entries dropped
  }
  c.rope_theta = get_num_or(a, "rope_theta", 10000.0);
  c.compress_rope_theta = get_num_or(a, "compress_rope_theta", 160000.0);
  if (const Json* rs = a.find("rope_scaling"); rs && rs->is_object()) {
    c.yarn = true;
    c.yarn_factor = get_num_or(*rs, "factor", 1.0);
    c.yarn_orig = get_size_or(*rs, "original_max_position_embeddings", 0);
    c.yarn_beta_fast = get_num_or(*rs, "beta_fast", 32.0);
    c.yarn_beta_slow = get_num_or(*rs, "beta_slow", 1.0);
  }
  c.index_heads = get_size_or(a, "index_n_heads", 64);
  c.index_dim = get_size_or(a, "index_head_dim", 128);
  c.index_topk = get_size_or(a, "index_topk", 512);
  c.experts = get_size(a, "n_routed_experts");
  c.top_k = get_size(a, "num_experts_per_tok");
  c.moe_inter = get_size(a, "moe_intermediate_size");
  c.hash_layers = get_size_or(a, "num_hash_layers", 0);
  std::string sf = "sqrtsoftplus";
  if (const Json* v = a.find("scoring_func"); v && v->is_string()) sf = v->as_string();
  if (sf == "sqrtsoftplus") c.scoring = 0;
  else if (sf == "sigmoid") c.scoring = 1;
  else if (sf == "softmax") c.scoring = 2;
  else throw FormatError("deepseek_v4: unknown scoring_func '" + sf + "'");
  c.route_scale = get_num_or(a, "routed_scaling_factor", 1.5);
  c.swiglu_limit = get_num_or(a, "swiglu_limit", 10.0);
  c.hc = get_size_or(a, "hc_mult", 4);
  c.sinkhorn_iters = get_size_or(a, "hc_sinkhorn_iters", 20);
  c.hc_eps = get_num_or(a, "hc_eps", 1e-6);
  c.rms_eps = get_num_or(a, "rms_norm_eps", 1e-6);
  c.sim_kv_fp8 = get_bool_or(a, "simulate_kv_fp8", true);
  c.sim_index_fp4 = get_bool_or(a, "simulate_indexer_fp4", true);
  if (get_bool_or(a, "simulate_act_fp8", false)) throw FormatError("deepseek_v4: simulate_act_fp8 is not supported");
  if (get_bool_or(a, "tie_word_embeddings", false)) throw FormatError("deepseek_v4: tied embeddings are not supported");
  if (get_size_or(a, "n_shared_experts", 1) != 1) throw FormatError("deepseek_v4: n_shared_experts != 1");
  if (get_size_or(a, "num_key_value_heads", 1) != 1) throw FormatError("deepseek_v4: num_key_value_heads != 1");
  if (const Json* e = a.find("eos_token_id"); e && !e->is_null()) c.eos = uint32_t(e->is_array() ? e->at(0).as_uint() : e->as_uint());
  if (const Json* b = a.find("bos_token_id"); b && b->is_int()) c.bos = uint32_t(b->as_uint());
  if (const Json* v = a.find("torch_dtype"); v && v->is_string()) c.torch_dtype = v->as_string();
  // shape checks of the kernels
  const auto pow2 = [](std::size_t v) { return v && !(v & (v - 1)); };
  if (c.rope_dim % 2 || c.rope_dim > c.head_dim || c.rope_dim > c.index_dim) throw FormatError("deepseek_v4: rope dims");
  if ((c.head_dim - c.rope_dim) % 64) throw FormatError("deepseek_v4: (head_dim - qk_rope_head_dim) % 64 (FP8 blocks)");
  if (c.head_dim % 32 || c.head_dim > 1024) throw FormatError("deepseek_v4: head_dim must be a multiple of 32, <= 1024");
  if (!pow2(c.index_dim) || c.index_dim < 32 || c.index_dim > 4096) throw FormatError("deepseek_v4: index_head_dim");
  if ((c.heads * c.head_dim) % c.o_groups) throw FormatError("deepseek_v4: heads * head_dim % o_groups");
  if (c.hc != 4) throw FormatError("deepseek_v4: hc_mult must be 4");
  if (c.hash_layers && c.vocab > 64 * 64 * 64) throw FormatError("deepseek_v4: hash routing with vocab > 262144");
  if (c.experts > 1024 || c.top_k > 16) throw FormatError("deepseek_v4: more than 1024 experts or top_k > 16");
  for (std::size_t r : c.ratios)
    if (r && r != 4 && r > 512) throw FormatError("deepseek_v4: compress ratio " + std::to_string(r));
  return c;
}

// ------------------------------------------------------------------------------------------ model

struct Ds4Model::Impl {
  Context* ctx = nullptr;
  const MappedFile* mf = nullptr;
  Ds4Options opt;
  Ds4Config cfg;

  struct Hc {
    Vec fn;      // (mix, hc · D) float32
    Vec base, scale;
    bool head = false;
    std::size_t mix = 0;
  };
  struct Comp {
    std::size_t ratio = 0, d = 0, cd = 0;
    bool overlap = false, rotate = false;
    Mat wkv, wgate;   // dense (cd, D), float32 inputs
    Vec ape, norm;
    BufPtr kvbuf, scbuf;   // (2, ratio, cd) float32 (group parity)
    BufPtr ent;            // (cap, d) entries in T
  };
  struct Layer {
    Hc ha, hf;
    Vec attn_norm, ffn_norm, q_norm, kv_norm, sink;
    Mat wq_a, wq_b, wkv, wo_a, wo_b;
    std::size_t ratio = 0;
    bool idx = false, hash = false;
    Comp comp, icomp;
    Mat iwq_b, iwp;
    BufPtr win;   // (window, head_dim) ring
    Mat router;   // dense (E, D), float32 input
    Vec bias, tid2eid;
    Mat sh_g, sh_u, sh_d;
    std::unique_ptr<ExpertStack> eg, eu, ed;
    int stream_layer = -1;
  };
  Mat embed, head;
  Hc hc_head;
  Vec norm;
  std::vector<Layer> layers;
  Workspace ws[4];
  ExpertScratch es_g, es_u, es_d;
  BufPtr freq_base, freq_comp;

  // arena
  BufPtr tok, hs, xa, xf, ss, mixraw, coef_a, coef_f, y, xn, xnf;
  BufPtr qa, qr, q, kvr, kvn, pooled, iq, iw, iscore, sel, att, woy, z, f;
  BufPtr scores, probs, cpart;   // cpart: GEMM partials of the folded fallback attention (H > 1)
  BufPtr gates, eidx, ew, xk, gk, uk, hk, yk, shg, shu, shh, sh, mo;
  BufPtr pgates, peidx, pew;            // streamed + prefetch: the next layer's routing guess
  MTL::CommandBuffer* cur_cb = nullptr;  // streamed: replaced at every MoE layer (split_command_buffer)
  double split_gpu_s = 0.0;
  BufPtr logits, amax;
  std::size_t pos = 0, max_entries = 0, max_keys = 0;
  std::vector<std::size_t> nent;   // compressed entries per layer (main = indexer count)
  bool have_logits = false;
  uint32_t last_token = 0;
  double gpu_s = 0.0, encode_s = 0.0;
  // SUC_DS4_PROFILE=1: diagnostic stage profile. Every stage boundary commits the command buffer and waits, so each
  // stage's GPU time is measured alone (same kernels in the same order, same bits; slower: no overlap across stages).
  bool prof = false;
  const char* prof_label = nullptr;
  std::map<std::string, std::pair<double, std::size_t>> prof_acc;   // label -> (GPU seconds, command buffers)
  void mark(MTL::ComputeCommandEncoder*& enc, const char* next) {
    if (!prof) return;
    double g = 0.0;
    split_command_buffer(*ctx, cur_cb, enc, &g);
    prof_add(g);
    split_gpu_s += g;
    prof_label = next;
  }
  void prof_add(double g) {
    if (!prof || !prof_label) return;
    auto& a = prof_acc[prof_label];
    a.first += g;
    ++a.second;
  }
  uint64_t arena_bytes = 0, kv_bytes = 0, copied_bytes = 0;
  float eps = 0.0f, hc_eps = 0.0f;

  std::size_t as() const { return act_size(opt.act); }
  const char* T() const { return act_msl(opt.act); }
  BufPtr alloc(std::size_t bytes, uint64_t* counter) {
    if (counter) *counter += bytes;
    return ctx->new_buffer(std::max<std::size_t>(bytes, 16));
  }
  static BufRef R(const BufPtr& b, std::size_t off = 0) { return {b.get(), off}; }
  const TensorDesc& tensor(const std::string& name) const { return mf->file().tensor(name); }

  // a raw / bf16 tensor of n elements, in place when stored in `want` (or any float dtype when want is nullopt)
  Vec vec(const std::string& name, std::size_t n, std::optional<DType> want) {
    const TensorDesc& t = tensor(name);
    if (t.has_residual()) fail(name + ": residual on a kept tensor");
    Vec v;
    if ((t.representation_type == "raw" || t.representation_type == "bf16") && t.arrays.size() == 1) {
      const ArrayRecord& r = t.arrays[0];
      if (r.logical_numel() != n) fail(name + ": " + std::to_string(r.logical_numel()) + " elements, expected " + std::to_string(n));
      const bool is_float = r.dtype == DType::F32 || r.dtype == DType::BF16 || r.dtype == DType::F16;
      if ((!want && is_float) || (want && r.dtype == *want)) {
        v.ref = mf->at(r);
        v.dtype = r.dtype;
        return v;
      }
      if (want && (*want == DType::I32 || *want == DType::I64)) {   // integer table in another integer dtype
        std::vector<int32_t> iv(n);
        const auto b = mf->file().payload(r);
        for (std::size_t i = 0; i < n; ++i) iv[i] = int32_t(load_as_i64(b.data() + i * dtype_size(r.dtype), r.dtype));
        v.owned = ctx->upload(iv.data(), n * 4);
        v.ref = R(v.owned);
        v.dtype = DType::I32;
        copied_bytes += n * 4;
        return v;
      }
    }
    // other codecs: the reference decode (float32), cast to the wanted dtype (as load_suc1_model)
    const DecodedTensor d = decode_tensor(mf->file(), t);
    if (d.data.size() != n) fail(name + ": decoded size mismatch");
    const DType dt = want ? *want : DType::F32;
    if (dt == DType::I32 || dt == DType::I64) fail(name + ": integer table stored with a codec");
    const Act a = dt == DType::BF16 ? Act::BF16 : dt == DType::F16 ? Act::F16 : Act::F32;
    v.owned = ctx->new_buffer(n * act_size(a));
    f32_to_act(d.data.data(), v.owned->contents(), n, a);
    v.ref = R(v.owned);
    v.dtype = dt;
    copied_bytes += n * act_size(a);
    return v;
  }

  // dense matrix in dtype `want` (dense slots: compressor, router, hyper-connection mixers)
  Mat dense(const std::string& name, std::size_t out, std::size_t in, DType want) {
    const TensorDesc& t = tensor(name);
    if (t.shape.size() != 2 || t.shape[0] != out || t.shape[1] != in)
      fail(name + ": unexpected shape, expected (" + std::to_string(out) + ", " + std::to_string(in) + ")");
    Mat m;
    m.name = name;
    m.out = out;
    m.in = in;
    m.kind = Mat::Dense;
    Vec v = vec(name, out * in, want);
    m.w = v.ref;
    m.wdtype = v.dtype;
    m.owned = v.owned;
    return m;
  }

  // projection with T inputs: SUC codes, affine codes or a dense matrix in the model dtype
  Mat mat(const std::string& name, std::size_t out, std::size_t in, bool allow_suc = true) {
    const TensorDesc& t = tensor(name);
    if (t.shape.size() != 2 || t.shape[0] != out || t.shape[1] != in)
      fail(name + ": unexpected shape, expected (" + std::to_string(out) + ", " + std::to_string(in) + ")");
    const std::string& rep = t.representation_type;
    if (rep == "vq" || rep == "additive" || rep == "tcq") {
      if (!allow_suc) fail(name + ": SUC codes cannot serve an embedding gather");
      Mat m;
      m.name = name;
      m.out = out;
      m.in = in;
      m.kind = Mat::Suc;
      m.lin = std::make_unique<Linear>(Linear::from_file(*ctx, *mf, t));
      copied_bytes += m.lin->copied_bytes();
      return m;
    }
    if (rep == "affine" && !t.has_residual()) {
      const Json& p = *t.params;
      Mat m;
      m.name = name;
      m.out = out;
      m.in = in;
      m.kind = Mat::Affine;
      m.bits = unsigned(p["bits"].as_uint());
      m.group = std::size_t(p["group_size"].as_uint());
      const ArrayRecord& qi = t.array("indices");
      const ArrayRecord& sc = t.array("scales");
      const ArrayRecord& bi = t.array("biases");
      m.words = words_per_row(in, m.bits);
      if (qi.packed && qi.bits == m.bits && qi.shape == std::vector<uint64_t>{out, m.words} && in % m.group == 0 &&
          sc.dtype == bi.dtype && sc.shape == std::vector<uint64_t>{out, in / m.group} && sc.shape == bi.shape &&
          (sc.dtype == DType::F16 || sc.dtype == DType::BF16 || sc.dtype == DType::F32) &&
          (m.bits == 2 || m.bits == 4 || m.bits == 8)) {
        m.sdtype = sc.dtype;
        m.packed = mf->at(qi);
        m.scales = mf->at(sc);
        m.biases = mf->at(bi);
        return m;
      }
    }
    return dense(name, out, in, dtype_of(opt.act));
  }

  Hc hyper(const std::string& p, bool head) {
    const std::size_t W = cfg.stream(), hc = cfg.hc;
    Hc h;
    h.head = head;
    h.mix = head ? hc : (2 + hc) * hc;
    h.fn = vec(p + ".fn", h.mix * W, DType::F32);
    h.base = vec(p + ".base", h.mix, std::nullopt);
    h.scale = vec(p + ".scale", head ? 1 : 3, std::nullopt);
    if (h.base.dtype != h.scale.dtype) {   // one template type for both: convert the scale
      h.scale = vec(p + ".scale", head ? 1 : 3, h.base.dtype);
    }
    return h;
  }

  Comp compressor(const std::string& p, std::size_t ratio, std::size_t d, bool rotate) {
    Comp c;
    c.ratio = ratio;
    c.d = d;
    c.overlap = ratio == 4;
    c.cd = (c.overlap ? 2 : 1) * d;
    c.rotate = rotate;
    c.wkv = dense(p + ".wkv.weight", c.cd, cfg.hidden, dtype_of(opt.act));
    c.wgate = dense(p + ".wgate.weight", c.cd, cfg.hidden, dtype_of(opt.act));
    c.ape = vec(p + ".ape", ratio * c.cd, std::nullopt);
    c.norm = vec(p + ".norm.weight", d, dtype_of(opt.act));
    c.kvbuf = alloc(2 * ratio * c.cd * 4, &kv_bytes);
    c.scbuf = alloc(2 * ratio * c.cd * 4, &kv_bytes);
    c.ent = alloc((opt.max_context / ratio + 1) * d * as(), &kv_bytes);
    return c;
  }

  // ---- encoders
  // y = x Wᵀ for one row of T (SUC: metal::Linear with workspace ws[wsi]; affine / dense: MLX gemv order).
  void proj(MTL::ComputeCommandEncoder* enc, const Mat& m, BufRef x, BufRef y, int wsi = 0, std::size_t B = 1) {
    if (m.kind == Mat::Suc) {
      m.lin->encode(*ctx, enc, x, y, B, opt.act, ws[wsi], Path::Auto);
      return;
    }
    if (B != 1) fail(m.name + ": batched dense projection");
    const int sn = m.in <= 64 ? 4 : 32;
    const int bn = (m.in > 64 && m.in >= 16 * m.out) ? 8 : 1;
    const int sm = 32 / sn;
    if (m.kind == Mat::Affine) {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3001, {int64_t(m.in), int64_t(m.out), m.bits, int64_t(m.group), int64_t(m.sdtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{
            "suc_gemv_affine", join({T(), I(m.bits), I(long(m.group)), I(long(m.in)), I(long(m.out)), I(long(m.words)),
                                     msl_of(m.sdtype), I(sn), "4", I(bn)})};
      }));
      set(enc, x, 0);
      set(enc, m.packed, 1);
      set(enc, m.scales, 2);
      set(enc, m.biases, 3);
      set(enc, y, 4);
    } else {
      gemv_pso(enc, T(), m);
      set(enc, x, 0);
      set(enc, m.w, 1);
      set(enc, y, 2);
    }
    const std::size_t blocks = (m.out + sm - 1) / sm;
    if (m.kind == Mat::Affine && bn == 1) {   // affine, one simdgroup per row block: affine_rows_sg() of them per threadgroup
      const std::size_t g = affine_rows_sg();
      enc->dispatchThreadgroups(MTL::Size((blocks + g - 1) / g, 1, 1), MTL::Size(32 * g, 1, 1));
      return;
    }
    enc->dispatchThreadgroups(MTL::Size(blocks, 1, 1), MTL::Size(32 * bn, 1, 1));
  }
  void gemv_pso(MTL::ComputeCommandEncoder* enc, const char* tx, const Mat& m) {
    const int sn = m.in <= 64 ? 4 : 32;
    const int bn = (m.in > 64 && m.in >= 16 * m.out) ? 8 : 1;
    const bool f32 = std::string(tx) == "float";
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3002, {int64_t(m.in), int64_t(m.out), int64_t(m.wdtype), f32 ? 1 : int64_t(opt.act) + 10}}, [&]() {
      return std::pair<std::string, std::string>{
          "suc_moe_gemv", join({tx, msl_of(m.wdtype), I(long(m.in)), I(long(m.out)), I(sn), "4", I(bn)})};
    }));
  }
  // y (float32) = x (float32) Wᵀ (dense W in its dtype, MLX gemv order).
  void gemv_f32(MTL::ComputeCommandEncoder* enc, const Mat& m, BufRef x, BufRef y) {
    if (m.kind != Mat::Dense) fail(m.name + ": float32 projections must be dense");
    const int sn = m.in <= 64 ? 4 : 32;
    const int bn = (m.in > 64 && m.in >= 16 * m.out) ? 8 : 1;
    const int sm = 32 / sn;
    gemv_pso(enc, "float", m);
    set(enc, x, 0);
    set(enc, m.w, 1);
    set(enc, y, 2);
    enc->dispatchThreadgroups(MTL::Size((m.out + sm - 1) / sm, 1, 1), MTL::Size(32 * bn, 1, 1));
  }

  // y (float32) = float(x) Wᵀ for x in T (dense W, MLX gemv order): = gemv_f32 on the float32 copy of x, without
  // waiting for that copy (ds4_gemv_xf).
  void gemv_xf(MTL::ComputeCommandEncoder* enc, const Mat& m, BufRef x, BufRef y) {
    if (m.kind != Mat::Dense) fail(m.name + ": float32 projections must be dense");
    const int sn = m.in <= 64 ? 4 : 32;
    const int bn = (m.in > 64 && m.in >= 16 * m.out) ? 8 : 1;
    const int sm = 32 / sn;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3016, {int64_t(m.in), int64_t(m.out), int64_t(m.wdtype), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{
          "ds4_gemv_xf", join({T(), msl_of(m.wdtype), I(long(m.in)), I(long(m.out)), I(sn), "4", I(bn), "0"})};
    }));
    set(enc, x, 0);
    set(enc, m.w, 1);
    set(enc, y, 2);
    set(enc, R(ss), 3);
    const float z = 0.0f;
    bytes(enc, z, 4);
    bytes(enc, z, 5);
    enc->dispatchThreadgroups(MTL::Size((m.out + sm - 1) / sm, 1, 1), MTL::Size(32 * bn, 1, 1));
  }

  static std::size_t rms_tpg(std::size_t n) { return n <= 4096 ? 32 * (((n + 3) / 4 + 31) / 32) : 1024; }
  // MLX row-reduce threads of mean(x²) over n contiguous values (measured, MLX 0.32.3)
  static std::size_t red_tpg(std::size_t n) {
    if (n <= 512) return 32;
    if (n <= 1024) return 128;
    if (n <= 2048) return 512;
    return 1024;
  }

  void rms(MTL::ComputeCommandEncoder* enc, BufRef x, const Vec& w, BufRef y, std::size_t rows, std::size_t n) {
    const std::size_t tpg = rms_tpg(n);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3003, {int64_t(n), int64_t(tpg), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"suc_rms_norm", join({T(), I(long(n)), I(long(tpg))})};
    }));
    set(enc, x, 0);
    set(enc, w.ref, 1);
    set(enc, y, 2);
    bytes(enc, eps, 3);
    enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(tpg, 1, 1));
  }

  void cast_f32(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t n) {
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3004, {int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"suc_cast", join({T(), "float"})};
    }));
    set(enc, x, 0);
    set(enc, y, 1);
    const uint32_t n32 = uint32_t(n);
    bytes(enc, n32, 2);
    enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(std::min<std::size_t>(256, n), 1, 1));
  }

  // HyperConnection.pre / HCHead on the stream x: y (D) in T, coef (pre | post | comb) float32.
  // Three phases: {sum of squares + float copy, mixer gemv on x}, {coefficients (one simdgroup, element-parallel
  // Sinkhorn), collapse with its own pre coefficients}, then the caller's RMSNorm. Same values as the sequential
  // reduce -> gemv(xf) -> one-thread coef -> collapse chain (see the kernels).
  // nw / xn: the following RMSNorm (xn = rms_norm(y, nw)); fused into the collapse when D <= 4096 (y is then not
  // written), else run after it.
  void hc_pre(MTL::ComputeCommandEncoder*& enc, const Hc& h, BufRef x, BufRef coef, BufRef yout, const Vec& nw,
              BufRef xnout) {
    const std::size_t W = cfg.stream(), D = cfg.hidden, HC = cfg.hc;
    const std::size_t tpg = red_tpg(W);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3010, {int64_t(W), int64_t(tpg), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_hc_in", join({T(), I(long(W)), I(long(tpg))})};
    }));
    set(enc, x, 0);
    set(enc, R(xf), 1);
    set(enc, R(ss), 2);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(tpg, 1, 1));
    mark(enc, "hc_mix_gemv");
    {
      const int sn = W <= 64 ? 4 : 32;
      const int bn = (W > 64 && W >= 16 * h.mix) ? 8 : 1;
      const int sm = 32 / sn;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3014, {int64_t(W), int64_t(h.mix), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{
            "ds4_gemv_xf", join({T(), "float", I(long(W)), I(long(h.mix)), I(sn), "4", I(bn), "0"})};
      }));
      set(enc, x, 0);
      set(enc, h.fn.ref, 1);
      set(enc, R(mixraw), 2);
      set(enc, R(ss), 3);
      const float z = 0.0f;
      bytes(enc, z, 4);
      bytes(enc, z, 5);
      enc->dispatchThreadgroups(MTL::Size((h.mix + sm - 1) / sm, 1, 1), MTL::Size(32 * bn, 1, 1));
    }
    barrier(enc);
    mark(enc, "hc_coef_sinkhorn");
    const int head = h.head ? 1 : 0, iters = int(cfg.sinkhorn_iters);
    const float inv_w = float(1.0 / double(W)), ne = eps;
    if (HC * HC > 32) fail("deepseek_v4: hc_mult > 5 (the coefficient kernel holds hc^2 values in one simdgroup)");
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3011, {int64_t(h.base.dtype), int64_t(HC), iters, head}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_hc_coef_par", join({h.base.msl(), I(long(HC)), I(iters), I(head)})};
    }));
    set(enc, R(mixraw), 0);
    set(enc, R(ss), 1);
    set(enc, h.base.ref, 2);
    set(enc, h.scale.ref, 3);
    set(enc, coef, 4);
    bytes(enc, inv_w, 5);
    bytes(enc, ne, 6);
    bytes(enc, hc_eps, 7);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));
    mark(enc, "hc_collapse_norm");
    if (D <= 4096 && rms_tpg(D) <= 1024) {
      const std::size_t tpg = rms_tpg(D);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3015, {int64_t(D), int64_t(HC), int64_t(opt.act), int64_t(h.base.dtype), head, int64_t(tpg)}}, [&]() {
        return std::pair<std::string, std::string>{
            "ds4_hc_collapse_rms", join({T(), h.base.msl(), I(long(D)), I(long(HC)), I(head), I(long(tpg))})};
      }));
      set(enc, x, 0);
      set(enc, R(mixraw), 1);
      set(enc, R(ss), 2);
      set(enc, h.base.ref, 3);
      set(enc, h.scale.ref, 4);
      set(enc, nw.ref, 5);
      set(enc, xnout, 6);
      bytes(enc, inv_w, 7);
      bytes(enc, ne, 8);
      bytes(enc, hc_eps, 9);
      bytes(enc, eps, 10);
      enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(tpg, 1, 1));
      barrier(enc);
      return;
    }
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3012, {int64_t(D), int64_t(HC), int64_t(opt.act), int64_t(h.base.dtype), head}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_hc_collapse_pre", join({T(), h.base.msl(), I(long(D)), I(long(HC)), I(head)})};
    }));
    set(enc, x, 0);
    set(enc, R(mixraw), 1);
    set(enc, R(ss), 2);
    set(enc, h.base.ref, 3);
    set(enc, h.scale.ref, 4);
    set(enc, yout, 5);
    bytes(enc, inv_w, 6);
    bytes(enc, ne, 7);
    bytes(enc, hc_eps, 8);
    enc->dispatchThreads(MTL::Size(D, 1, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    barrier(enc);
    rms(enc, yout, nw, xnout, 1, D);
  }

  // hc_pre of n chunk tokens, stage-major: each stage's launches for all tokens back to back (per-token scratch in
  // xf_c / ss_c / mixraw_c / y_c), one barrier per stage instead of per stage and token. Same kernels and inputs per
  // token as hc_pre, so the same bits.
  // hc_pre of n contiguous stream rows x (n, W) in ONE launch per stage (fast batched prefill): Σ x² per row
  // (glm_hc_ss_rows: ds4_hc_in's sum order), the mixer gemv over the rows, the coefficients and the collapse + RMSNorm
  // with a row grid. Every row gets the bits of the per-row launches (same kernels, row offsets in the kernels).
  // False when the shape needs the per-row fallback.
  bool hc_pre_rows_batched(MTL::ComputeCommandEncoder* enc, const Hc& h, BufRef x, BufRef coef, const Vec& nw,
                           BufRef xnout, std::size_t n) {
    const std::size_t W = cfg.stream(), D = cfg.hidden, HC = cfg.hc;
    if (h.head || !(D <= 4096 && rms_tpg(D) <= 1024) || HC * HC > 32 || h.mix != (2 + HC) * HC) return false;
    const std::size_t tpg = red_tpg(W);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3090, {int64_t(W), int64_t(tpg), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"glm_hc_ss_rows", join({T(), I(long(W)), I(long(tpg))})};
    }));
    set(enc, x, 0);
    set(enc, R(ss_c), 1);
    enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(tpg, 1, 1));
    {
      const int sn = W <= 64 ? 4 : 32;
      const int bn = (W > 64 && W >= 16 * h.mix) ? 8 : 1;
      const int sm = 32 / sn;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3014, {int64_t(W), int64_t(h.mix), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{
            "ds4_gemv_xf", join({T(), "float", I(long(W)), I(long(h.mix)), I(sn), "4", I(bn), "0"})};
      }));
      const float z = 0.0f;
      set(enc, x, 0);
      set(enc, h.fn.ref, 1);
      set(enc, R(mixraw_c), 2);
      set(enc, R(ss_c), 3);
      bytes(enc, z, 4);
      bytes(enc, z, 5);
      enc->dispatchThreadgroups(MTL::Size((h.mix + sm - 1) / sm, n, 1), MTL::Size(32 * bn, 1, 1));
    }
    barrier(enc);
    const int iters = int(cfg.sinkhorn_iters);
    const float inv_w = float(1.0 / double(W)), ne = eps;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3011, {int64_t(h.base.dtype), int64_t(HC), iters, 0}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_hc_coef_par", join({h.base.msl(), I(long(HC)), I(iters), "0"})};
    }));
    set(enc, R(mixraw_c), 0);
    set(enc, R(ss_c), 1);
    set(enc, h.base.ref, 2);
    set(enc, h.scale.ref, 3);
    set(enc, coef, 4);
    bytes(enc, inv_w, 5);
    bytes(enc, ne, 6);
    bytes(enc, hc_eps, 7);
    enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(32, 1, 1));
    const std::size_t tpg2 = rms_tpg(D);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3015, {int64_t(D), int64_t(HC), int64_t(opt.act), int64_t(h.base.dtype), 0, int64_t(tpg2)}}, [&]() {
      return std::pair<std::string, std::string>{
          "ds4_hc_collapse_rms", join({T(), h.base.msl(), I(long(D)), I(long(HC)), "0", I(long(tpg2))})};
    }));
    set(enc, x, 0);
    set(enc, R(mixraw_c), 1);
    set(enc, R(ss_c), 2);
    set(enc, h.base.ref, 3);
    set(enc, h.scale.ref, 4);
    set(enc, nw.ref, 5);
    set(enc, xnout, 6);
    bytes(enc, inv_w, 7);
    bytes(enc, ne, 8);
    bytes(enc, hc_eps, 9);
    bytes(enc, eps, 10);
    enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(tpg2, 1, 1));
    barrier(enc);
    return true;
  }

  void hc_pre_rows(MTL::ComputeCommandEncoder* enc, const Hc& h, const std::function<BufRef(std::size_t)>& x,
                   BufRef coef, std::size_t coef_stride, const Vec& nw, BufRef xnout, std::size_t n) {
    const std::size_t W = cfg.stream(), D = cfg.hidden, HC = cfg.hc, a = as();
    const std::size_t tpg = red_tpg(W), ms = (2 + HC) * HC * 4;
    auto xf_i = [&](std::size_t i) { return R(xf_c, i * W * 4); };
    auto ss_i = [&](std::size_t i) { return R(ss_c, i * 16); };
    auto mr_i = [&](std::size_t i) { return R(mixraw_c, i * ms); };
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3010, {int64_t(W), int64_t(tpg), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_hc_in", join({T(), I(long(W)), I(long(tpg))})};
    }));
    for (std::size_t i = 0; i < n; ++i) {
      set(enc, x(i), 0);
      set(enc, xf_i(i), 1);
      set(enc, ss_i(i), 2);
      enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(tpg, 1, 1));
    }
    {
      const int sn = W <= 64 ? 4 : 32;
      const int bn = (W > 64 && W >= 16 * h.mix) ? 8 : 1;
      const int sm = 32 / sn;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3014, {int64_t(W), int64_t(h.mix), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{
            "ds4_gemv_xf", join({T(), "float", I(long(W)), I(long(h.mix)), I(sn), "4", I(bn), "0"})};
      }));
      const float z = 0.0f;
      for (std::size_t i = 0; i < n; ++i) {
        set(enc, x(i), 0);
        set(enc, h.fn.ref, 1);
        set(enc, mr_i(i), 2);
        set(enc, ss_i(i), 3);
        bytes(enc, z, 4);
        bytes(enc, z, 5);
        enc->dispatchThreadgroups(MTL::Size((h.mix + sm - 1) / sm, 1, 1), MTL::Size(32 * bn, 1, 1));
      }
    }
    barrier(enc);
    const int head = h.head ? 1 : 0, iters = int(cfg.sinkhorn_iters);
    const float inv_w = float(1.0 / double(W)), ne = eps;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3011, {int64_t(h.base.dtype), int64_t(HC), iters, head}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_hc_coef_par", join({h.base.msl(), I(long(HC)), I(iters), I(head)})};
    }));
    for (std::size_t i = 0; i < n; ++i) {
      set(enc, mr_i(i), 0);
      set(enc, ss_i(i), 1);
      set(enc, h.base.ref, 2);
      set(enc, h.scale.ref, 3);
      set(enc, coef.at(i * coef_stride), 4);
      bytes(enc, inv_w, 5);
      bytes(enc, ne, 6);
      bytes(enc, hc_eps, 7);
      enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));
    }
    if (D <= 4096 && rms_tpg(D) <= 1024) {
      const std::size_t tpg2 = rms_tpg(D);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3015, {int64_t(D), int64_t(HC), int64_t(opt.act), int64_t(h.base.dtype), head, int64_t(tpg2)}}, [&]() {
        return std::pair<std::string, std::string>{
            "ds4_hc_collapse_rms", join({T(), h.base.msl(), I(long(D)), I(long(HC)), I(head), I(long(tpg2))})};
      }));
      for (std::size_t i = 0; i < n; ++i) {
        set(enc, x(i), 0);
        set(enc, mr_i(i), 1);
        set(enc, ss_i(i), 2);
        set(enc, h.base.ref, 3);
        set(enc, h.scale.ref, 4);
        set(enc, nw.ref, 5);
        set(enc, xnout.at(i * D * a), 6);
        bytes(enc, inv_w, 7);
        bytes(enc, ne, 8);
        bytes(enc, hc_eps, 9);
        bytes(enc, eps, 10);
        enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(tpg2, 1, 1));
      }
      barrier(enc);
      return;
    }
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3012, {int64_t(D), int64_t(HC), int64_t(opt.act), int64_t(h.base.dtype), head}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_hc_collapse_pre", join({T(), h.base.msl(), I(long(D)), I(long(HC)), I(head)})};
    }));
    for (std::size_t i = 0; i < n; ++i) {
      set(enc, x(i), 0);
      set(enc, mr_i(i), 1);
      set(enc, ss_i(i), 2);
      set(enc, h.base.ref, 3);
      set(enc, h.scale.ref, 4);
      set(enc, R(y_c, i * D * a), 5);
      bytes(enc, inv_w, 6);
      bytes(enc, ne, 7);
      bytes(enc, hc_eps, 8);
      enc->dispatchThreads(MTL::Size(D, 1, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    }
    barrier(enc);
    rms(enc, R(y_c), nw, xnout, n, D);
    barrier(enc);
  }

  // rows > 1: rows contiguous rows of f (D), res / yout (HC · D) and coef ((2 + HC) · HC floats), one launch
  void hc_post(MTL::ComputeCommandEncoder* enc, BufRef fbuf, BufRef res, BufRef coef, BufRef yout, bool bar = true,
               std::size_t rows = 1) {
    const std::size_t D = cfg.hidden, HC = cfg.hc;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3013, {int64_t(D), int64_t(HC), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_hc_post", join({T(), I(long(D)), I(long(HC))})};
    }));
    set(enc, fbuf, 0);
    set(enc, res, 1);
    set(enc, coef, 2);
    set(enc, yout, 3);
    enc->dispatchThreads(MTL::Size(D, HC, rows), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    if (bar) barrier(enc);
  }

  void rope(MTL::ComputeCommandEncoder* enc, BufRef x, std::size_t rows, std::size_t D, const BufPtr& freq,
            uint32_t p, bool inverse) {
    const std::size_t RD = cfg.rope_dim;
    const int inv = inverse ? 1 : 0;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3020, {int64_t(D), int64_t(RD), inv, int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_rope", join({T(), I(long(D)), I(long(RD)), I(inv)})};
    }));
    set(enc, x, 0);
    set(enc, R(freq), 1);
    bytes(enc, p, 2);
    enc->dispatchThreads(MTL::Size(RD / 2, rows, 1), MTL::Size(std::min<std::size_t>(RD / 2, 256), 1, 1));
  }

  void kv_store(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t D) {
    const std::size_t NQ = D - cfg.rope_dim;
    const int sim = cfg.sim_kv_fp8 ? 1 : 0;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3021, {int64_t(D), int64_t(NQ), sim, int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_kv_store", join({T(), I(long(D)), I(long(NQ)), I(sim)})};
    }));
    set(enc, x, 0);
    set(enc, y, 1);
    enc->dispatchThreadgroups(MTL::Size((D + 63) / 64, 1, 1), MTL::Size(64, 1, 1));
  }

  void fp4(MTL::ComputeCommandEncoder* enc, BufRef x, std::size_t rows, std::size_t D) {
    if (!cfg.sim_index_fp4) return;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3022, {int64_t(D), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_fp4_sim", join({T(), I(long(D))})};
    }));
    set(enc, x, 0);
    enc->dispatchThreads(MTL::Size(D, rows, 1), MTL::Size(32, 1, 1));
  }

  void hadamard(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t rows, std::size_t N) {
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3023, {int64_t(N), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_hadamard", join({T(), I(long(N))})};
    }));
    set(enc, x, 0);
    set(enc, y, 1);
    const float scale = float(std::pow(double(N), -0.5));
    bytes(enc, scale, 2);
    enc->dispatchThreads(MTL::Size(N / 16, rows, 1), MTL::Size(N / 16, 1, 1));
  }

  // One compressor at position p: the current row (computed into its group slot by gemv) and, when the group is
  // complete, the new entry (pool -> norm -> RoPE -> Hadamard + FP4 | FP8) appended to c.ent.
  void comp_rows(MTL::ComputeCommandEncoder* enc, const Comp& c, std::size_t p, std::size_t = 0, BufRef x = {}) {
    const std::size_t g = p / c.ratio, slot = (g % 2) * c.ratio + p % c.ratio;
    const BufRef xi = x ? x : R(xn);
    gemv_xf(enc, c.wkv, xi, R(c.kvbuf, slot * c.cd * 4));
    gemv_xf(enc, c.wgate, xi, R(c.scbuf, slot * c.cd * 4));
  }
  void comp_emit(MTL::ComputeCommandEncoder* enc, const Comp& c, std::size_t p, const BufPtr& freq) {
    const std::size_t g = p / c.ratio, d = c.d;
    const std::size_t cur = (g % 2) * c.ratio * c.cd * 4, prev = ((g + 1) % 2) * c.ratio * c.cd * 4;
    const int ov = c.overlap ? 1 : 0;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3024, {int64_t(d), int64_t(c.ratio), ov, int64_t(c.ape.dtype), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_comp_pool", join({T(), c.ape.msl(), I(long(d)), I(long(c.ratio)), I(ov)})};
    }));
    set(enc, R(c.kvbuf, cur), 0);
    set(enc, R(c.scbuf, cur), 1);
    set(enc, R(c.kvbuf, prev), 2);
    set(enc, R(c.scbuf, prev), 3);
    set(enc, c.ape.ref, 4);
    set(enc, R(pooled), 5);
    const uint32_t has_prev = g > 0 ? 1 : 0;
    bytes(enc, has_prev, 6);
    enc->dispatchThreadgroups(MTL::Size(d, 1, 1), MTL::Size(32, 1, 1));
    barrier(enc);
    rms(enc, R(pooled), c.norm, R(pooled, d * as()), 1, d);
    barrier(enc);
    const BufRef e = R(pooled, d * as());
    rope(enc, e, 1, d, freq, uint32_t(g * c.ratio), false);
    barrier(enc);
    const BufRef dst = R(c.ent, g * d * as());
    if (c.rotate) {
      hadamard(enc, e, dst, 1, d);
      barrier(enc);
      fp4(enc, dst, 1, d);
    } else {
      kv_store(enc, e, dst, d);
    }
  }

  bool vector_attention() const {
    const std::size_t D = cfg.head_dim;
    return (D == 64 || D == 96 || D == 128 || D == 256) && cfg.heads <= 32;
  }

  // o (H, hd) = attention of q over [window keys | compressed entries (selected)], attention sinks
  void attention(MTL::ComputeCommandEncoder* enc, const Layer& L, std::size_t p, std::size_t n, bool use_sel,
                 std::size_t row = 0, BufRef qbuf = {}, BufRef attbuf = {}) {
    // row: the token's q / att rows in qbuf / attbuf (default: the one-token q / att)
    const std::size_t D = cfg.head_dim, H = cfg.heads, qo = row * H * D * as();
    const uint32_t wn = uint32_t(std::min(p + 1, cfg.window)), ws_ = uint32_t((p + 1 - wn) % cfg.window),
                   wsz = uint32_t(cfg.window), ne = uint32_t(n), nsel = use_sel ? 1u : 0u;
    const float scale = float(std::pow(double(D), -0.5));
    const BufRef ent = L.ratio ? R(L.comp.ent) : ctx->dummy();
    if (vector_attention()) {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3030, {int64_t(D), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_sdpa_vector", join({T(), I(long(D))})};
      }));
      set(enc, (qbuf ? qbuf : R(q)).at(qo), 0);
      set(enc, R(L.win), 1);
      set(enc, ent, 2);
      set(enc, R(sel), 3);
      set(enc, L.sink.ref, 4);
      set(enc, (attbuf ? attbuf : R(att)).at(qo), 5);
      bytes(enc, wn, 6);
      bytes(enc, ws_, 7);
      bytes(enc, wsz, 8);
      bytes(enc, ne, 9);
      bytes(enc, nsel, 10);
      bytes(enc, scale, 11);
      enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(1024, 1, 1));
      return;
    }
    // MLX's unfused fallback (fast.cpp): scores, softmax (precise), p[1:] @ V. One head: per-head gemv / gemv_t. H heads
    // (one KV head): MLX folds them into M = H rows of one matmul (see ds4_mma in suc_ds4_kernels.metal).
    const std::size_t N = wn + n, A = N + 1;
    const bool f32 = opt.act == Act::F32;
    const char devc = ctx->architecture().empty() ? 'g' : ctx->architecture().back();
    const uint32_t h32 = uint32_t(H);
    auto mma = [&](BufRef a, uint32_t lda, uint32_t aoff, std::size_t M, std::size_t Nn, std::size_t K, bool nt,
                   float ascale) {
      const GemmPlan g = gemm_plan(devc, M, Nn, K, nt, f32);
      const int ntf = nt ? 1 : 0;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3035, {int64_t(D), ntf, int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_mma", join({T(), I(long(D)), I(ntf)})};
      }));
      set(enc, a, 0);
      set(enc, R(L.win), 1);
      set(enc, ent, 2);
      set(enc, R(cpart), 3);
      const uint32_t m32 = uint32_t(M), n32 = uint32_t(Nn), k32 = uint32_t(K), ps = uint32_t(g.psize),
                     bk = uint32_t(g.tail_bk);
      bytes(enc, m32, 4);
      bytes(enc, n32, 5);
      bytes(enc, k32, 6);
      bytes(enc, lda, 7);
      bytes(enc, aoff, 8);
      bytes(enc, ps, 9);
      bytes(enc, bk, 10);
      bytes(enc, ascale, 11);
      bytes(enc, wn, 12);
      bytes(enc, ws_, 13);
      bytes(enc, wsz, 14);
      if (g.parts * M * Nn * 4 > cpart->length()) fail("GEMM partials exceed the scratch buffer");
      enc->dispatchThreadgroups(MTL::Size((Nn + 7) / 8, (M + 7) / 8, g.parts), MTL::Size(32, 1, 1));
      barrier(enc);
      const int sc = nt ? 1 : 0;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3036, {sc, int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_mma_out", join({T(), I(sc)})};
      }));
      set(enc, R(cpart), 0);
      set(enc, nt ? R(scores) : (attbuf ? attbuf : R(att)).at(qo), 1);
      set(enc, R(sel), 2);
      set(enc, L.sink.ref, 3);
      const uint32_t p32 = uint32_t(g.parts);
      bytes(enc, m32, 4);
      bytes(enc, n32, 5);
      bytes(enc, p32, 6);
      bytes(enc, wn, 7);
      bytes(enc, nsel, 8);
      enc->dispatchThreads(MTL::Size(Nn, M, 1), MTL::Size(std::min<std::size_t>(256, Nn), 1, 1));
    };
    // gemv_wide (matmul.cpp gemv_wide_config): generation >= 15, 2..15 rows (<= 3 passes of <= 5 vectors), N >= 2,
    // K % 4 == 0, bf16 / f16 output; 32 lanes per row with one pass or <= 64 keys, else 16.
    const bool wide = H > 1 && H <= 15 && N >= 2 && D % 4 == 0 && !f32 && architecture_gen(ctx->architecture()) >= 15;
    if (H > 1 && N >= 2 && !wide) {
      mma((qbuf ? qbuf : R(q)).at(qo), uint32_t(D), 0, H, N, D, true, scale);
    } else if (wide) {
      const int kl = ((H + 4) / 5 == 1 || N <= 64) ? 32 : 16;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3037, {int64_t(D), kl, int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_fb_scores_wide", join({T(), I(long(D)), I(kl)})};
      }));
      set(enc, (qbuf ? qbuf : R(q)).at(qo), 0);
      set(enc, R(L.win), 1);
      set(enc, ent, 2);
      set(enc, R(sel), 3);
      set(enc, L.sink.ref, 4);
      set(enc, R(scores), 5);
      bytes(enc, wn, 6);
      bytes(enc, ws_, 7);
      bytes(enc, wsz, 8);
      bytes(enc, ne, 9);
      bytes(enc, nsel, 10);
      bytes(enc, scale, 11);
      bytes(enc, h32, 12);
      enc->dispatchThreadgroups(MTL::Size((N + 3) / 4, 1, 1), MTL::Size(32 * (kl / 8), 1, 1));
    } else {
      // gemv: over the keys (one head; MLX out = keys) or over the H rows of q' (one key; MLX out = H)
      const std::size_t gout = H == 1 ? N : H;
      const int sn = D <= 64 ? 4 : 32, bn = (D > 64 && D >= 16 * gout) ? 8 : 1, sm = 32 / sn;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3031, {int64_t(D), sn, bn, int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_fb_scores", join({T(), I(long(D)), I(sn), "4", I(bn)})};
      }));
      set(enc, (qbuf ? qbuf : R(q)).at(qo), 0);
      set(enc, R(L.win), 1);
      set(enc, ent, 2);
      set(enc, R(sel), 3);
      set(enc, L.sink.ref, 4);
      set(enc, R(scores), 5);
      bytes(enc, wn, 6);
      bytes(enc, ws_, 7);
      bytes(enc, wsz, 8);
      bytes(enc, ne, 9);
      bytes(enc, nsel, 10);
      bytes(enc, scale, 11);
      enc->dispatchThreadgroups(MTL::Size((N + sm - 1) / sm, H, 1), MTL::Size(32 * bn, 1, 1));
    }
    barrier(enc);
    const int a32 = int(A);
    if (A <= 4096) {
      const std::size_t tpg = std::min<std::size_t>(1024, ((A + 3) / 4 + 31) / 32 * 32);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3032, {int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_softmax_row", T()};
      }));
      set(enc, R(scores), 0);
      set(enc, R(probs), 1);
      bytes(enc, a32, 2);
      enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(tpg, 1, 1));
    } else {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3033, {int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_softmax_looped", T()};
      }));
      set(enc, R(scores), 0);
      set(enc, R(probs), 1);
      bytes(enc, a32, 2);
      enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(1024, 1, 1));
    }
    barrier(enc);
    if (H > 1) {   // p[1:] (copied contiguous by MLX: same values) @ V as one (H, keys) x (keys, D) GEMM
      mma(R(probs), uint32_t(A), 1, H, D, N, false, 0.0f);
      return;
    }
    // MLX gemv_t tiling: SM 8 / SN 4 (SM 4 / SN 8 when in >= 8192 and out >= 2048), BN 16 / 4 / 2 by out size, TM = 4,
    // TN = 4 (1 when out < 4)
    int tsm = 8, tsn = 4;
    if (N >= 8192 && D >= 2048) tsm = 4, tsn = 8;
    const int tbn = D >= 2048 ? 16 : D >= 512 ? 4 : 2;
    const int ttn = D < 4 ? 1 : 4;
    const std::size_t blockN = std::size_t(tbn * tsn * ttn);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3034, {int64_t(D), tsm, tsn, tbn, ttn, int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_fb_pv", join({T(), I(long(D)), I(tsm), I(tsn), I(tbn), "4", I(ttn)})};
    }));
    set(enc, R(probs), 0);
    set(enc, R(L.win), 1);
    set(enc, ent, 2);
    set(enc, (attbuf ? attbuf : R(att)).at(qo), 3);
    bytes(enc, wn, 4);
    bytes(enc, ws_, 5);
    bytes(enc, wsz, 6);
    bytes(enc, ne, 7);
    enc->dispatchThreadgroups(MTL::Size((D + blockN - 1) / blockN, H, 1), MTL::Size(32 * tbn, 1, 1));
  }

  // The attention sub-layer on xn (D) at position p -> f (D)
  void attn(MTL::ComputeCommandEncoder*& enc, Layer& L, std::size_t l, std::size_t p) {
    const std::size_t H = cfg.heads, hd = cfg.head_dim, a = as();
    const BufPtr& freq = L.ratio ? freq_comp : freq_base;
    proj(enc, L.wq_a, R(xn), R(qa), 0);
    proj(enc, L.wkv, R(xn), R(kvr), 1);
    if (L.ratio) {   // compressor rows straight from xn, concurrent with wq_a / wkv
      comp_rows(enc, L.comp, p);
      if (L.idx) comp_rows(enc, L.icomp, p);
    }
    barrier(enc);
    mark(enc, "attn_q_kv_norm");
    rms(enc, R(qa), L.q_norm, R(qr), 1, cfg.q_lora);
    rms(enc, R(kvr), L.kv_norm, R(kvn), 1, hd);
    barrier(enc);
    mark(enc, "attn_wq_b_rope");
    proj(enc, L.wq_b, R(qr), R(q), 0);
    rope(enc, R(kvn), 1, hd, freq, uint32_t(p), false);
    barrier(enc);
    mark(enc, "attn_comp_emit_kv_qnorm");
    // compressed entries completed at this position (indexer compressor first, as Attention.__call__)
    const bool emit = L.ratio && (p + 1) % L.ratio == 0;
    std::size_t n = L.ratio ? nent[l] : 0;
    if (emit) {
      if (L.idx) {
        comp_emit(enc, L.icomp, p, freq);
        barrier(enc);
      }
      comp_emit(enc, L.comp, p, freq);
      ++n;
    }
    barrier(enc);
    kv_store(enc, R(kvn), R(L.win, (p % cfg.window) * hd * a), hd);
    {
      const std::size_t tpg = red_tpg(hd);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3040, {int64_t(hd), int64_t(cfg.rope_dim), int64_t(tpg), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_qnorm_rope", join({T(), I(long(hd)), I(long(cfg.rope_dim)), I(long(tpg))})};
      }));
      set(enc, R(q), 0);
      set(enc, R(freq), 1);
      const uint32_t p32 = uint32_t(p);
      bytes(enc, p32, 2);
      bytes(enc, eps, 3);
      enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(tpg, 1, 1));
    }
    // lightning indexer: top-k of the compressed entries once more than index_topk exist
    const bool use_sel = L.idx && n > cfg.index_topk;
    if (use_sel) {
      barrier(enc);
      mark(enc, "indexer");
      const std::size_t ih = cfg.index_heads, idd = cfg.index_dim;
      proj(enc, L.iwq_b, R(qr), R(iq), 2);
      proj(enc, L.iwp, R(xn), R(iw), 3);
      barrier(enc);
      rope(enc, R(iq), ih, idd, freq, uint32_t(p), false);
      barrier(enc);
      hadamard(enc, R(iq), R(iq), ih, idd);
      barrier(enc);
      fp4(enc, R(iq), ih, idd);
      barrier(enc);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3041, {int64_t(ih), int64_t(idd), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_index_scores", join({T(), I(long(ih)), I(long(idd))})};
      }));
      set(enc, R(iq), 0);
      set(enc, R(L.icomp.ent), 1);
      set(enc, R(iw), 2);
      set(enc, R(iscore), 3);
      const float cw = float(std::pow(double(idd), -0.5) * std::pow(double(ih), -0.5));
      bytes(enc, cw, 4);
      enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(32, 1, 1));
      barrier(enc);
      enc->setComputePipelineState(ctx->pipeline("q4_index_select", "1024"));
      set(enc, R(iscore), 0);
      set(enc, R(sel), 1);
      const uint32_t n32 = uint32_t(n), k32 = uint32_t(cfg.index_topk);
      bytes(enc, n32, 2);
      bytes(enc, k32, 3);
      enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
    }
    barrier(enc);
    mark(enc, "attn_core");
    attention(enc, L, p, n, use_sel);
    barrier(enc);
    rope(enc, R(att), H, hd, freq, uint32_t(p), true);
    barrier(enc);
    mark(enc, "attn_wo_a");
    // wo_a: row group g on input group g
    const std::size_t G = cfg.o_groups, Ro = cfg.o_lora, ing = H * hd / G;
    if (L.wo_a.kind == Mat::Suc) {
      proj(enc, L.wo_a, R(att), R(woy), 0, G);
      barrier(enc);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3042, {int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_diag", T()};
      }));
      set(enc, R(woy), 0);
      set(enc, R(z), 1);
      const uint32_t g32 = uint32_t(G), r32 = uint32_t(Ro);
      bytes(enc, g32, 2);
      bytes(enc, r32, 3);
      enc->dispatchThreads(MTL::Size(Ro, G, 1), MTL::Size(std::min<std::size_t>(256, Ro), 1, 1));
    } else if (L.wo_a.kind == Mat::Affine) {   // the codes in place: rows g·Ro .. (g+1)·Ro of the packed matrix
      Mat part;
      part.kind = Mat::Affine;
      part.name = L.wo_a.name;
      part.out = Ro;
      part.in = ing;
      part.bits = L.wo_a.bits;
      part.group = L.wo_a.group;
      part.words = L.wo_a.words;
      part.sdtype = L.wo_a.sdtype;
      const std::size_t sb = (ing / part.group) * dtype_size(part.sdtype);
      for (std::size_t g = 0; g < G; ++g) {
        part.packed = L.wo_a.packed.at(g * Ro * part.words * 4);
        part.scales = L.wo_a.scales.at(g * Ro * sb);
        part.biases = L.wo_a.biases.at(g * Ro * sb);
        proj(enc, part, R(att, g * ing * a), R(z, g * Ro * a), 0);
      }
    } else {
      Mat part;
      part.kind = Mat::Dense;
      part.name = L.wo_a.name;
      part.out = Ro;
      part.in = ing;
      part.wdtype = L.wo_a.wdtype;
      for (std::size_t g = 0; g < G; ++g) {
        part.w = L.wo_a.w.at(g * Ro * ing * dtype_size(L.wo_a.wdtype));
        proj(enc, part, R(att, g * ing * a), R(z, g * Ro * a), 0);
      }
    }
    barrier(enc);
    mark(enc, "attn_wo_b");
    proj(enc, L.wo_b, R(z), R(f), 0);
    barrier(enc);
    if (emit) nent[l] = n;
  }

  // Gate: float32 router logits in `gates` -> experts eidx (K) and weights ew (K) of layer L for the token in `slot`.
  // rows > 1: rows consecutive slots from `slot` (gates / eidx / ew rows contiguous), one launch
  void route(MTL::ComputeCommandEncoder* enc, const Layer& L, std::size_t slot, BufRef gates_, BufRef eidx_,
             BufRef ew_, std::size_t rows = 1) {
    const std::size_t E = cfg.experts, K = cfg.top_k;
    const int hash = L.hash ? 1 : 0;
    const DType ti = L.hash ? L.tid2eid.dtype : DType::I32;
    const DType tb = L.hash ? DType::F32 : L.bias.dtype;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3051, {int64_t(E), int64_t(K), cfg.scoring, hash, int64_t(ti), int64_t(tb)}}, [&]() {
      return std::pair<std::string, std::string>{
          "ds4_route", join({I(long(E)), I(long(K)), I(cfg.scoring), I(hash), msl_of(ti), msl_of(tb)})};
    }));
    set(enc, gates_, 0);
    set(enc, L.hash ? ctx->dummy() : L.bias.ref, 1);
    set(enc, L.hash ? L.tid2eid.ref : ctx->dummy(), 2);
    set(enc, R(tok, slot * cfg.hc * 4), 3);
    set(enc, eidx_, 4);
    set(enc, ew_, 5);
    const float rsc = float(cfg.route_scale);
    bytes(enc, rsc, 6);
    const uint32_t ts = uint32_t(cfg.hc);
    bytes(enc, ts, 7);
    const std::size_t tpg = (E + 31) / 32 * 32;
    enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(tpg, 1, 1));
  }

  // The MoE sub-layer on xn (D) -> mo (D). Streamed experts: enc (and cur_cb) are replaced by the host sync.
  void moe(MTL::ComputeCommandEncoder*& enc, const Layer& L, std::size_t l, std::size_t slot) {
    const std::size_t D = cfg.hidden, E = cfg.experts, K = cfg.top_k, F = cfg.moe_inter, a = as();
    mark(enc, "moe_shared_gate_up");
    proj(enc, L.sh_g, R(xn), R(shg), 1);
    proj(enc, L.sh_u, R(xn), R(shu), 2);
    {
      const uint32_t cols = uint32_t(D), rep = uint32_t(K);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3050, {int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"suc_repeat_rows", T()};
      }));
      set(enc, R(xn), 0);
      set(enc, R(xk), 1);
      bytes(enc, cols, 2);
      bytes(enc, rep, 3);
      enc->dispatchThreads(MTL::Size(D, K, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    }
    mark(enc, "moe_router_gemv_swiglu");
    gemv_xf(enc, L.router, R(xn), R(gates));   // router logits straight from xn, concurrent with the shared gate / up
    barrier(enc);
    mark(enc, "moe_route_topk");
    route(enc, L, slot, R(gates), R(eidx), R(ew));
    swiglu(enc, R(shg), R(shu), BufRef{}, R(shh), F, 1);
    barrier(enc);
    mark(enc, "moe_shared_down");
    proj(enc, L.sh_d, R(shh), R(sh), 3);   // concurrent with the routed experts' first stage (no barrier in between)
    if (opt.expert_stream) {   // host sync: fetch the routed experts into the slot pool (and prefetch a guess)
      ExpertStream& xs = *opt.expert_stream;
      const Layer* nxt = (l + 1 < layers.size() && xs.options().prefetch) ? &layers[l + 1] : nullptr;
      if (nxt) {   // pre-gating guess: the next layer's gate on this layer's MoE input
        gemv_xf(enc, nxt->router, R(xn), R(pgates));
        barrier(enc);
        route(enc, *nxt, slot, R(pgates), R(peidx), R(pew));
      }
      split_command_buffer(*ctx, cur_cb, enc, &split_gpu_s);
      xs.fetch(L.stream_layer, static_cast<const uint32_t*>(eidx->contents()), K);
      if (nxt) xs.prefetch(nxt->stream_layer, static_cast<const uint32_t*>(peidx->contents()), K);
    }
    mark(enc, "experts_gate_up");
    ExpertStack::encode_group(*ctx, enc, {{L.eu.get(), R(xk), R(uk), &es_u}, {L.eg.get(), R(xk), R(gk), &es_g}}, R(eidx), K,
                              opt.act);
    mark(enc, "experts_swiglu");
    swiglu(enc, R(gk), R(uk), R(ew), R(hk), F, K);
    barrier(enc);
    mark(enc, "experts_down");
    ExpertStack::encode_group(*ctx, enc, {{L.ed.get(), R(hk), R(yk), &es_d}}, R(eidx), K, opt.act);
    mark(enc, "moe_out_hc_post");
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3052, {int64_t(K), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_moe_out", join({T(), I(long(K))})};
    }));
    set(enc, R(yk), 0);
    set(enc, R(sh), 1);
    set(enc, R(mo), 2);
    const uint32_t d32 = uint32_t(D);
    bytes(enc, d32, 3);
    enc->dispatchThreads(MTL::Size(D, 1, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    barrier(enc);
    (void)a;
    (void)E;
  }

  void swiglu(MTL::ComputeCommandEncoder* enc, BufRef g, BufRef u, BufRef w, BufRef h, std::size_t F, std::size_t rows) {
    const int weight = w ? 1 : 0, lim = cfg.swiglu_limit > 0 ? 1 : 0;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3053, {weight, lim, int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_swiglu", join({T(), I(weight), I(lim)})};
    }));
    set(enc, g, 0);
    set(enc, u, 1);
    set(enc, w ? w : ctx->dummy(), 2);
    set(enc, h, 3);
    const uint32_t f32 = uint32_t(F);
    const float limit = float(cfg.swiglu_limit);
    bytes(enc, f32, 4);
    bytes(enc, limit, 5);
    enc->dispatchThreads(MTL::Size(F, rows, 1), MTL::Size(std::min<std::size_t>(256, F), 1, 1));
  }

  // One decode step for the token in slot `slot` at position p.
  void encode_step(MTL::ComputeCommandEncoder*& enc, std::size_t slot, std::size_t p, bool with_logits) {
    const std::size_t D = cfg.hidden, W = cfg.stream(), a = as();
    if (prof) prof_label = "embed";
    {   // embedding -> hc identical streams
      const Mat& m = embed;
      if (m.kind == Mat::Affine) {
        enc->setComputePipelineState(ctx->pipeline(PsoKey{3060, {int64_t(D), m.bits, int64_t(m.group), int64_t(m.sdtype), int64_t(opt.act)}}, [&]() {
          return std::pair<std::string, std::string>{
              "suc_embed_affine", join({T(), I(m.bits), I(long(m.group)), I(long(m.in)), I(long(m.words)), msl_of(m.sdtype)})};
        }));
        set(enc, R(tok, slot * cfg.hc * 4), 0);
        set(enc, m.packed, 1);
        set(enc, m.scales, 2);
        set(enc, m.biases, 3);
        set(enc, R(hs), 4);
      } else {
        enc->setComputePipelineState(ctx->pipeline(PsoKey{3061, {int64_t(D), int64_t(m.wdtype), int64_t(opt.act)}}, [&]() {
          return std::pair<std::string, std::string>{"suc_embed_dense", join({T(), I(long(m.in)), msl_of(m.wdtype)})};
        }));
        set(enc, R(tok, slot * cfg.hc * 4), 0);
        set(enc, m.w, 1);
        set(enc, R(hs), 4);
      }
      enc->dispatchThreads(MTL::Size(D, cfg.hc, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
      barrier(enc);
    }
    for (std::size_t l = 0; l < cfg.layers; ++l) {
      Layer& L = layers[l];
      const BufRef x = R(hs, l * W * a), xo = R(hs, (l + 1) * W * a);
      mark(enc, "hc_in_reduce");
      hc_pre(enc, L.ha, x, R(coef_a), R(y), L.attn_norm, R(xn));
      barrier(enc);
      mark(enc, "attn_wq_a_wkv_comp_rows");
      attn(enc, L, l, p);
      mark(enc, "hc_post_attn");
      hc_post(enc, R(f), x, R(coef_a), R(xa));
      barrier(enc);
      mark(enc, "hc_in_reduce");
      hc_pre(enc, L.hf, R(xa), R(coef_f), R(y), L.ffn_norm, R(xn));
      barrier(enc);
      moe(enc, L, l, slot);
      hc_post(enc, R(mo), R(xa), R(coef_f), xo);
    }
    if (with_logits) {
      barrier(enc);
      mark(enc, "hc_in_reduce");
      hc_pre(enc, hc_head, R(hs, cfg.layers * W * a), R(coef_a), R(y), norm, R(xn));
      barrier(enc);
      mark(enc, "head_argmax");
      proj(enc, head, R(xn), R(logits), 0);
      barrier(enc);
      enc->setComputePipelineState(ctx->pipeline("suc_argmax", join({T(), "1024"})));
      set(enc, R(logits), 0);
      set(enc, R(amax), 1);
      const uint32_t n = uint32_t(cfg.vocab);
      bytes(enc, n, 2);
      enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
    }
  }

  // ---------------------------------------------------------------------------------------- batched (layer-major) prefill
  // Chunk scratch (Ds4Options::layer_prefill), rows = chunk tokens unless noted.
  BufPtr ring, xa_c, coefa_c, coeff_c, xn_c, qa_c, kvr_c, qr_c, kvn_c, q_c, att_c, z_c, f_c, shg_c, shu_c, shh_c, sh_c,
      mo_c, eidx_c, ew_c, yk_c, xf_c, ss_c, mixraw_c, y_c, gates_c;
  BufPtr t_x, t_g, t_u, t_h, t_y;              // tile rows
  // fast (not row-exact) chunk attention: linear keys (window - 1 + chunk rows), compressor rows (chunk, cd) float32
  // of comp / icomp, pooled groups, per-row entry counts (per layer)
  BufPtr klin, ckv_c, csc_c, ickv_c, icsc_c, pooled_c, ne_c;
  // batched indexer (rows past index_topk entries): query rows (n, ih, idd), weights (n, ih), scores (n, isc_ps),
  // selected entries (n, index_topk) and counts (n); scores grown per chunk (isc_ps = entries at the chunk end)
  BufPtr iq_c, iw_c, isc_c, sidx_c, scnt_c;
  std::size_t isc_ps = 0;
  // forward_rows(): logits of every chunk row (chunk, vocab) in T, appended to rows_out after each command buffer
  BufPtr logits_rows;
  std::vector<float>* rows_out = nullptr;
  bool fast() const { return !opt.exact_prefill && !prefill_exact_env(); }
  // $SUC_FAST_OFF (diagnostics): letters of fast parts to keep row-exact: p projections, a attention, t expert tiles,
  // r router
  static bool fast_off(char c) {
    static const std::string v = [] { const char* e = std::getenv("SUC_FAST_OFF"); return std::string(e ? e : ""); }();
    return v.find(c) != std::string::npos;
  }
  bool fast(char c) const { return fast() && !fast_off(c); }
  BufPtr t_teidx, t_reid, t_src, t_dst, t_w;   // tile plan (host-written)
  std::size_t t_rows_cap = 0, t_tiles_cap = 0;
  uint64_t chunk_fetches = 0;
  // Rows per threadgroup of the row-batched affine / dense gemv (suc_gemv_affine_rows / suc_moe_gemv_rows): fewer rows
  // keep more threadgroups in flight (M1 Ultra, 512-token prefill GPU s, 2 / 4 / 8 rows: Qwen3.8 affine 6.05 / 5.63 /
  // 5.91, dense 5.92 / 5.91 / 6.43; DeepSeek V1 affine 12.87 / 12.05 / 12.64);
  // any value gives the same bits. $SUC_AFFINE_ROWS_NB / $SUC_GEMV_ROWS_NB override (1-8).
  static int affine_rows_nb() {
    static const int v = [] { const char* e = std::getenv("SUC_AFFINE_ROWS_NB"); const int x = e ? std::atoi(e) : 0; return x >= 1 && x <= 8 ? x : 4; }();
    return v;
  }
  static int gemv_rows_nb() {
    static const int v = [] { const char* e = std::getenv("SUC_GEMV_ROWS_NB"); const int x = e ? std::atoi(e) : 0; return x >= 1 && x <= 8 ? x : 4; }();
    return v;
  }


  // y rows r = x rows r · Wᵀ for r < n (row strides xs / ys in elements). Affine: one suc_gemv_affine_rows launch
  // (each row's arithmetic = the one-row gemv); SUC / dense: the one-row launch per row.
  // SUC (contiguous rows): Linear::encode_rows_exact; dense: suc_moe_gemv_rows; both row-exact (bits of the one-row
  // launch). $SUC_PREFILL_ROW_PROJ=1: one row at a time (the previous path, for A/B timing).
  void proj_rows(MTL::ComputeCommandEncoder* enc, const Mat& m, BufRef x, BufRef y, std::size_t n, std::size_t xs,
                 std::size_t ys, int wsi = 0) {
    const std::size_t a = as();
    static const bool row_proj = [] { const char* e = std::getenv("SUC_PREFILL_ROW_PROJ"); return e && e[0] == '1'; }();
    if (fast('p') && n >= kMmMinRows && !row_proj) {   // batched prefill, not row-exact: GEMMs over the n rows
      if (m.kind == Mat::Suc && xs == m.in && ys == m.out) {
        m.lin->encode(*ctx, enc, x, y, n, opt.act, ws[wsi], Path::Prefill);
        return;
      }
      if (m.kind == Mat::Affine && encode_mm_affine(*ctx, enc, x, m.packed, m.scales, m.biases, y, n, xs, ys, m.bits,
                                                    m.group, m.in, m.out, m.words, msl_of(m.sdtype), opt.act))
        return;
      if (m.kind == Mat::Dense && encode_mm_dense(*ctx, enc, x, m.w, y, n, xs, ys, m.in, m.out, msl_of(m.wdtype), opt.act))
        return;
    }
    if (n > 1 && m.kind == Mat::Suc && xs == m.in && ys == m.out && !row_proj) {
      m.lin->encode_rows_exact(*ctx, enc, x, y, n, opt.act, ws[wsi]);
      return;
    }
    if (n > 1 && m.kind == Mat::Dense && !row_proj) {
      const int sn = m.in <= 64 ? 4 : 32;
      const int bn = (m.in > 64 && m.in >= 16 * m.out) ? 8 : 1;
      const int sm = 32 / sn;
      static const int NB = gemv_rows_nb();
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3071, {int64_t(m.in), int64_t(m.out), int64_t(m.wdtype), int64_t(opt.act), NB}}, [&]() {
        return std::pair<std::string, std::string>{
            "suc_moe_gemv_rows", join({T(), msl_of(m.wdtype), I(long(m.in)), I(long(m.out)), I(sn), "4", I(bn), I(NB)})};
      }));
      set(enc, x, 0);
      set(enc, m.w, 1);
      set(enc, y, 2);
      const uint32_t n32 = uint32_t(n), xs32 = uint32_t(xs), ys32 = uint32_t(ys);
      bytes(enc, n32, 3);
      bytes(enc, xs32, 4);
      bytes(enc, ys32, 5);
      enc->dispatchThreadgroups(MTL::Size((m.out + sm - 1) / sm, (n + NB - 1) / NB, 1), MTL::Size(32 * bn, 1, 1));
      return;
    }
    if (m.kind != Mat::Affine || n == 1) {
      // one-row launches; SUC projections share the workspace ws[wsi] between rows, hence the barriers
      for (std::size_t r = 0; r < n; ++r) {
        proj(enc, m, x.at(r * xs * a), y.at(r * ys * a), wsi);
        if (m.kind == Mat::Suc && r + 1 < n) barrier(enc);
      }
      return;
    }
    const int sn = m.in <= 64 ? 4 : 32;
    const int bn = (m.in > 64 && m.in >= 16 * m.out) ? 8 : 1;
    const int sm = 32 / sn;
    static const int NB = affine_rows_nb();
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3070, {int64_t(m.in), int64_t(m.out), m.bits, int64_t(m.group), int64_t(m.sdtype), int64_t(opt.act), NB}}, [&]() {
      return std::pair<std::string, std::string>{
          "suc_gemv_affine_rows", join({T(), I(m.bits), I(long(m.group)), I(long(m.in)), I(long(m.out)), I(long(m.words)),
                                        msl_of(m.sdtype), I(sn), "4", I(bn), I(NB)})};
    }));
    set(enc, x, 0);
    set(enc, m.packed, 1);
    set(enc, m.scales, 2);
    set(enc, m.biases, 3);
    set(enc, y, 4);
    const uint32_t n32 = uint32_t(n), xs32 = uint32_t(xs), ys32 = uint32_t(ys);
    bytes(enc, n32, 5);
    bytes(enc, xs32, 6);
    bytes(enc, ys32, 7);
    const std::size_t blocks = (m.out + sm - 1) / sm, rb = (n + NB - 1) / NB;
    if (bn == 1) {
      const std::size_t g = affine_rows_sg();
      enc->dispatchThreadgroups(MTL::Size((blocks + g - 1) / g, rb, 1), MTL::Size(32 * g, 1, 1));
      return;
    }
    enc->dispatchThreadgroups(MTL::Size(blocks, rb, 1), MTL::Size(32 * bn, 1, 1));
  }

  // rows > 1: slots slot .. slot + rows - 1 into contiguous stream rows from dst (one launch)
  void encode_embed(MTL::ComputeCommandEncoder* enc, std::size_t slot, BufRef dst, std::size_t rows = 1) {
    const std::size_t D = cfg.hidden;
    const Mat& m = embed;
    if (m.kind == Mat::Affine) {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3060, {int64_t(D), m.bits, int64_t(m.group), int64_t(m.sdtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{
            "suc_embed_affine", join({T(), I(m.bits), I(long(m.group)), I(long(m.in)), I(long(m.words)), msl_of(m.sdtype)})};
      }));
      set(enc, R(tok, slot * cfg.hc * 4), 0);
      set(enc, m.packed, 1);
      set(enc, m.scales, 2);
      set(enc, m.biases, 3);
      set(enc, dst, 4);
    } else {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3061, {int64_t(D), int64_t(m.wdtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"suc_embed_dense", join({T(), I(long(m.in)), msl_of(m.wdtype)})};
      }));
      set(enc, R(tok, slot * cfg.hc * 4), 0);
      set(enc, m.w, 1);
      set(enc, dst, 4);
    }
    enc->dispatchThreads(MTL::Size(D, cfg.hc * rows, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    barrier(enc);
  }

  void encode_logits(MTL::ComputeCommandEncoder*& enc, BufRef x) {
    barrier(enc);
    hc_pre(enc, hc_head, x, R(coef_a), R(y), norm, R(xn));
    barrier(enc);
    proj(enc, head, R(xn), R(logits), 0);
    barrier(enc);
    enc->setComputePipelineState(ctx->pipeline("suc_argmax", join({T(), "1024"})));
    set(enc, R(logits), 0);
    set(enc, R(amax), 1);
    const uint32_t nv = uint32_t(cfg.vocab);
    bytes(enc, nv, 2);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
  }

  // The attention sub-layer of chunk rows 0 .. n-1 (positions p0 ..): xn_c -> f_c. Projections over the n rows, the
  // order-dependent parts (compressors, window, indexer, attention) token by token in position order: every kernel of
  // attn() with the same inputs, so the same bits per row.
  void attn_chunk(MTL::ComputeCommandEncoder*& enc, Layer& L, std::size_t l, std::size_t p0, std::size_t n) {
    const std::size_t H = cfg.heads, hd = cfg.head_dim, a = as(), D = cfg.hidden, ql = cfg.q_lora;
    const BufPtr& freq = L.ratio ? freq_comp : freq_base;
    static const long dbg_mask = [] { const char* e = std::getenv("SUC_DBG_ATTN_MASK"); return e ? std::atol(e) : -1L; }();
    if (fast('a') && n >= kMmMinRows && klin && ((dbg_mask >> l) & 1) &&
        !(L.idx && (p0 + n) / L.ratio > cfg.index_topk && fast_off('i'))) {
      attn_chunk_fast(enc, L, l, p0, n);
      return;
    }
    proj_rows(enc, L.wq_a, R(xn_c), R(qa_c), n, D, ql, 0);
    proj_rows(enc, L.wkv, R(xn_c), R(kvr_c), n, D, hd, 1);
    barrier(enc);
    rms(enc, R(qa_c), L.q_norm, R(qr_c), n, ql);
    rms(enc, R(kvr_c), L.kv_norm, R(kvn_c), n, hd);
    barrier(enc);
    proj_rows(enc, L.wq_b, R(qr_c), R(q_c), n, ql, H * hd, 0);
    barrier(enc);
    mark(enc, "chunk_attn_seq");
    // Token-independent per-token stages hoisted out of the sequential loop (same kernels and inputs, no barriers
    // between tokens): rope of every kv row and q norm + rope of every q row before it, the inverse rope of every
    // attention output row after it. $SUC_DS4_SEQ_TOKENS=1: everything inside the loop (the previous order, A/B).
    static const bool hoist = [] { const char* e = std::getenv("SUC_DS4_SEQ_TOKENS"); return !(e && e[0] == '1'); }();
    auto qnorm_rope = [&](BufRef qi, std::size_t p) {
      const std::size_t tpg = red_tpg(hd);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3040, {int64_t(hd), int64_t(cfg.rope_dim), int64_t(tpg), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_qnorm_rope", join({T(), I(long(hd)), I(long(cfg.rope_dim)), I(long(tpg))})};
      }));
      set(enc, qi, 0);
      set(enc, R(freq), 1);
      const uint32_t p32 = uint32_t(p);
      bytes(enc, p32, 2);
      bytes(enc, eps, 3);
      enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(tpg, 1, 1));
    };
    if (hoist) {
      for (std::size_t i = 0; i < n; ++i) {
        rope(enc, R(kvn_c, i * hd * a), 1, hd, freq, uint32_t(p0 + i), false);
        qnorm_rope(R(q_c, i * H * hd * a), p0 + i);
      }
      barrier(enc);
    }
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t p = p0 + i;
      const BufRef xni = R(xn_c, i * D * a), kvi = R(kvn_c, i * hd * a), qi = R(q_c, i * H * hd * a);
      if (L.ratio) {
        comp_rows(enc, L.comp, p, 0, xni);
        if (L.idx) comp_rows(enc, L.icomp, p, 0, xni);
      }
      const bool emit = L.ratio && (p + 1) % L.ratio == 0;
      if (!hoist) {
        rope(enc, kvi, 1, hd, freq, uint32_t(p), false);
        barrier(enc);
      } else if (emit) {
        barrier(enc);   // the pool reads this token's compressor rows
      }
      std::size_t ne = L.ratio ? nent[l] : 0;
      if (emit) {
        if (L.idx) {
          comp_emit(enc, L.icomp, p, freq);
          barrier(enc);
        }
        comp_emit(enc, L.comp, p, freq);
        ++ne;
      }
      barrier(enc);
      kv_store(enc, kvi, R(L.win, (p % cfg.window) * hd * a), hd);
      if (!hoist) qnorm_rope(qi, p);
      const bool use_sel = L.idx && ne > cfg.index_topk;
      if (use_sel) {
        barrier(enc);
        const std::size_t ih = cfg.index_heads, idd = cfg.index_dim;
        proj(enc, L.iwq_b, R(qr_c, i * ql * a), R(iq), 2);
        proj(enc, L.iwp, xni, R(iw), 3);
        barrier(enc);
        rope(enc, R(iq), ih, idd, freq, uint32_t(p), false);
        barrier(enc);
        hadamard(enc, R(iq), R(iq), ih, idd);
        barrier(enc);
        fp4(enc, R(iq), ih, idd);
        barrier(enc);
        enc->setComputePipelineState(ctx->pipeline(PsoKey{3041, {int64_t(ih), int64_t(idd), int64_t(opt.act)}}, [&]() {
          return std::pair<std::string, std::string>{"ds4_index_scores", join({T(), I(long(ih)), I(long(idd))})};
        }));
        set(enc, R(iq), 0);
        set(enc, R(L.icomp.ent), 1);
        set(enc, R(iw), 2);
        set(enc, R(iscore), 3);
        const float cw = float(std::pow(double(idd), -0.5) * std::pow(double(ih), -0.5));
        bytes(enc, cw, 4);
        enc->dispatchThreadgroups(MTL::Size(ne, 1, 1), MTL::Size(32, 1, 1));
        barrier(enc);
        enc->setComputePipelineState(ctx->pipeline("q4_index_select", "1024"));
        set(enc, R(iscore), 0);
        set(enc, R(sel), 1);
        const uint32_t n32 = uint32_t(ne), k32 = uint32_t(cfg.index_topk);
        bytes(enc, n32, 2);
        bytes(enc, k32, 3);
        enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
      }
      barrier(enc);
      attention(enc, L, p, ne, use_sel, i, R(q_c), R(att_c));
      barrier(enc);
      if (!hoist) {
        rope(enc, R(att_c, i * H * hd * a), H, hd, freq, uint32_t(p), true);
        barrier(enc);
      }
      if (emit) nent[l] = ne;
    }
    if (hoist) {
      for (std::size_t i = 0; i < n; ++i) rope(enc, R(att_c, i * H * hd * a), H, hd, freq, uint32_t(p0 + i), true);
      barrier(enc);
    }
    mark(enc, "chunk_attn_wo");
    // wo_a: row group g on input group g; wo_b
    const std::size_t G = cfg.o_groups, Ro = cfg.o_lora, ing = H * hd / G;
    if (L.wo_a.kind == Mat::Affine) {
      Mat part;
      part.kind = Mat::Affine;
      part.name = L.wo_a.name;
      part.out = Ro;
      part.in = ing;
      part.bits = L.wo_a.bits;
      part.group = L.wo_a.group;
      part.words = L.wo_a.words;
      part.sdtype = L.wo_a.sdtype;
      const std::size_t sb = (ing / part.group) * dtype_size(part.sdtype);
      for (std::size_t g = 0; g < G; ++g) {
        part.packed = L.wo_a.packed.at(g * Ro * part.words * 4);
        part.scales = L.wo_a.scales.at(g * Ro * sb);
        part.biases = L.wo_a.biases.at(g * Ro * sb);
        proj_rows(enc, part, R(att_c, g * ing * a), R(z_c, g * Ro * a), n, H * hd, G * Ro, 0);
      }
    } else {
      for (std::size_t i = 0; i < n; ++i) wo_a_row(enc, L, R(att_c, i * H * hd * a), R(z_c, i * G * Ro * a));
    }
    barrier(enc);
    proj_rows(enc, L.wo_b, R(z_c), R(f_c), n, G * Ro, D, 0);
    barrier(enc);
  }

  // ------------------------------------------------------------------ fast chunk attention (batched prefill, not exact)
  void rope_rows(MTL::ComputeCommandEncoder* enc, BufRef x, std::size_t rows, std::size_t D, const BufPtr& freq,
                 uint32_t p, uint32_t rpp, uint32_t pstep, bool inverse) {
    const std::size_t RD = cfg.rope_dim;
    const int inv = inverse ? 1 : 0;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3080, {int64_t(D), int64_t(RD), inv, int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_rope_rows", join({T(), I(long(D)), I(long(RD)), I(inv)})};
    }));
    set(enc, x, 0);
    set(enc, R(freq), 1);
    bytes(enc, p, 2);
    bytes(enc, rpp, 3);
    bytes(enc, pstep, 4);
    enc->dispatchThreads(MTL::Size(RD / 2, rows, 1), MTL::Size(std::min<std::size_t>(RD / 2, 256), 1, 1));
  }
  void kv_store_rows(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t rows, std::size_t D,
                     std::size_t ys) {
    const std::size_t NQ = D - cfg.rope_dim;
    const int sim = cfg.sim_kv_fp8 ? 1 : 0;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3081, {int64_t(D), int64_t(NQ), sim, int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_kv_store_rows", join({T(), I(long(D)), I(long(NQ)), I(sim)})};
    }));
    set(enc, x, 0);
    set(enc, y, 1);
    const uint32_t ys32 = uint32_t(ys);
    bytes(enc, ys32, 2);
    enc->dispatchThreadgroups(MTL::Size((D + 63) / 64, rows, 1), MTL::Size(64, 1, 1));
  }
  void ring_copy(MTL::ComputeCommandEncoder* enc, BufRef lin, BufRef ring, std::size_t rows, uint32_t start, int dir) {
    const uint32_t W = uint32_t(cfg.window), Dh = uint32_t(cfg.head_dim);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3082, {dir, int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_ring_copy", join({T(), I(dir)})};
    }));
    set(enc, lin, 0);
    set(enc, ring, 1);
    bytes(enc, start, 2);
    bytes(enc, W, 3);
    bytes(enc, Dh, 4);
    enc->dispatchThreads(MTL::Size(Dh, rows, 1), MTL::Size(std::min<uint32_t>(256, Dh), 1, 1));
  }
  void copy_f32_rows(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t rows, std::size_t cols) {
    if (rows) encode_copy_cols(*ctx, enc, x, y, rows, cols, cols, cols, Act::F32);
  }

  // Compressor c over the chunk rows (positions p0 .. p0 + n - 1): rows ckv / csc (n, cd) float32 by GEMM, then the
  // groups completed in the chunk: a group that began in the previous chunk through the group buffers (comp_emit), the
  // others pooled straight from the chunk rows in one launch (prev of each = the group before it), normed, roped,
  // stored (Hadamard + FP4 | FP8) as entries; the group buffers keep the last complete group and the incomplete tail.
  void comp_chunk(MTL::ComputeCommandEncoder* enc, const Comp& c, std::size_t p0, std::size_t n, const BufPtr& freq,
                  const BufPtr& ckv, const BufPtr& csc) {
    const std::size_t Rr = c.ratio, cd = c.cd, d = c.d, a = as(), D = cfg.hidden, pe = p0 + n;
    if (!encode_mm_dense(*ctx, enc, R(xn_c), c.wkv.w, R(ckv), n, D, cd, D, cd, msl_of(c.wkv.wdtype), opt.act, "float") ||
        !encode_mm_dense(*ctx, enc, R(xn_c), c.wgate.w, R(csc), n, D, cd, D, cd, msl_of(c.wgate.wdtype), opt.act, "float"))
      fail("compressor GEMM: unsupported shape");
    barrier(enc);
    auto slot = [&](std::size_t p) { return ((p / Rr) % 2) * Rr + p % Rr; };
    auto to_buf = [&](std::size_t q0, std::size_t q1) {   // positions [q0, q1) of one group -> group buffers
      if (q1 <= q0) return;
      copy_f32_rows(enc, R(ckv, (q0 - p0) * cd * 4), R(c.kvbuf, slot(q0) * cd * 4), q1 - q0, cd);
      copy_f32_rows(enc, R(csc, (q0 - p0) * cd * 4), R(c.scbuf, slot(q0) * cd * 4), q1 - q0, cd);
    };
    const std::size_t g0 = p0 / Rr;
    std::size_t ga = g0;   // first group starting inside the chunk
    if (p0 % Rr) {         // head group: began in the previous chunk
      ga = g0 + 1;
      const std::size_t e = std::min(ga * Rr, pe);
      to_buf(p0, e);
      if (e == ga * Rr) {
        barrier(enc);
        comp_emit(enc, c, e - 1, freq);
      }
    }
    const std::size_t gb = pe / Rr;   // groups [ga, gb) complete inside the chunk
    if (gb > ga) {
      const std::size_t G = gb - ga;
      const int ov = c.overlap ? 1 : 0;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3083, {int64_t(d), int64_t(Rr), ov, int64_t(c.ape.dtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_comp_pool_rows", join({T(), c.ape.msl(), I(long(d)), I(long(Rr)), I(ov)})};
      }));
      const std::size_t prev = ((ga + 1) % 2) * Rr * cd * 4;
      set(enc, R(ckv, (ga * Rr - p0) * cd * 4), 0);
      set(enc, R(csc, (ga * Rr - p0) * cd * 4), 1);
      set(enc, R(c.kvbuf, prev), 2);
      set(enc, R(c.scbuf, prev), 3);
      set(enc, c.ape.ref, 4);
      set(enc, R(pooled_c), 5);
      const uint32_t has_prev = ga > 0 ? 1 : 0;
      bytes(enc, has_prev, 6);
      enc->dispatchThreadgroups(MTL::Size(d, G, 1), MTL::Size(32, 1, 1));
      barrier(enc);
      const BufRef e = R(pooled_c, G * d * a);
      rms(enc, R(pooled_c), c.norm, e, G, d);
      barrier(enc);
      rope_rows(enc, e, G, d, freq, uint32_t(ga * Rr), 1, uint32_t(Rr), false);
      barrier(enc);
      const BufRef dst = R(c.ent, ga * d * a);
      if (c.rotate) {
        hadamard(enc, e, dst, G, d);
        barrier(enc);
        fp4(enc, dst, G, d);
      } else {
        kv_store_rows(enc, e, dst, G, d, d);
      }
      to_buf(std::max(p0, (gb - 1) * Rr), gb * Rr);   // the last complete group: prev of the next one
    }
    if (pe % Rr && gb * Rr >= p0) to_buf(gb * Rr, pe);   // incomplete tail group (a head group is already there)
  }

  // Attention sub-layer of chunk rows 0 .. n-1 (positions p0 ..), batched: projections as GEMMs, kv / q rope over all
  // rows, keys in a linear buffer (the window before the chunk gathered from the ring, then the chunk's FP8-simulated
  // keys), the compressors per chunk (comp_chunk), ONE ds4_attn_chunk launch, the ring updated with the chunk's last
  // window keys, inverse rope over all rows, wo_a / wo_b as GEMMs.
  void attn_chunk_fast(MTL::ComputeCommandEncoder*& enc, Layer& L, std::size_t l, std::size_t p0, std::size_t n) {
    const std::size_t H = cfg.heads, hd = cfg.head_dim, a = as(), D = cfg.hidden, ql = cfg.q_lora, Wn = cfg.window;
    const BufPtr& freq = L.ratio ? freq_comp : freq_base;
    mark(enc, "attn_proj_comp");
    proj_rows(enc, L.wq_a, R(xn_c), R(qa_c), n, D, ql, 0);
    proj_rows(enc, L.wkv, R(xn_c), R(kvr_c), n, D, hd, 1);
    if (L.ratio) {
      comp_chunk(enc, L.comp, p0, n, freq, ckv_c, csc_c);
      if (L.idx) comp_chunk(enc, L.icomp, p0, n, freq, ickv_c, icsc_c);
    }
    barrier(enc);
    rms(enc, R(qa_c), L.q_norm, R(qr_c), n, ql);
    rms(enc, R(kvr_c), L.kv_norm, R(kvn_c), n, hd);
    barrier(enc);
    proj_rows(enc, L.wq_b, R(qr_c), R(q_c), n, ql, H * hd, 0);
    rope_rows(enc, R(kvn_c), n, hd, freq, uint32_t(p0), 1, 1, false);
    const std::size_t WP = std::min(p0, Wn - 1);
    if (WP) ring_copy(enc, R(klin), R(L.win), WP, uint32_t((p0 - WP) % Wn), 0);
    barrier(enc);
    {
      const std::size_t tpg = red_tpg(hd);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3084, {int64_t(hd), int64_t(cfg.rope_dim), int64_t(tpg), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_qnorm_rope_rows", join({T(), I(long(hd)), I(long(cfg.rope_dim)), I(long(tpg))})};
      }));
      set(enc, R(q_c), 0);
      set(enc, R(freq), 1);
      const uint32_t p32 = uint32_t(p0), hpp = uint32_t(H);
      bytes(enc, p32, 2);
      bytes(enc, eps, 3);
      bytes(enc, hpp, 4);
      enc->dispatchThreadgroups(MTL::Size(n * H, 1, 1), MTL::Size(tpg, 1, 1));
    }
    kv_store_rows(enc, R(kvn_c), R(klin, WP * hd * a), n, hd, hd);
    // entries visible to row i: every group complete at its position (the entry of position p is emitted before p
    // attends)
    auto* ne = static_cast<uint32_t*>(ne_c->contents()) + l * opt.chunk;
    for (std::size_t i = 0; i < n; ++i) ne[i] = L.ratio ? uint32_t((p0 + i + 1) / L.ratio) : 0u;
    barrier(enc);
    // lightning indexer of the rows past index_topk entries (the decode step's selection, batched): query rows by GEMM,
    // rope / Hadamard / FP4 over all rows, scores of every row against its visible entries, top-k per row as an
    // ascending entry list
    const bool isel = L.idx && (p0 + n) / L.ratio > cfg.index_topk;
    if (isel) {
      mark(enc, "attn_indexer");
      const std::size_t ih = cfg.index_heads, idd = cfg.index_dim, K = cfg.index_topk, ps = (p0 + n) / L.ratio;
      if (ps > isc_ps || !iq_c) {
        if (!iq_c) {
          iq_c = alloc(opt.chunk * ih * idd * a, nullptr);
          iw_c = alloc(opt.chunk * ih * a, nullptr);
          sidx_c = alloc(opt.chunk * K * 4, nullptr);
          scnt_c = alloc(opt.chunk * 4, nullptr);
        }
        if (ps > isc_ps) {
          isc_ps = std::max(ps, isc_ps * 2);
          isc_c = alloc(opt.chunk * isc_ps * 4, nullptr);
        }
      }
      proj_rows(enc, L.iwq_b, R(qr_c), R(iq_c), n, ql, ih * idd, 2);
      proj_rows(enc, L.iwp, R(xn_c), R(iw_c), n, D, ih, 3);
      barrier(enc);
      rope_rows(enc, R(iq_c), n * ih, idd, freq, uint32_t(p0), uint32_t(ih), 1, false);
      barrier(enc);
      hadamard(enc, R(iq_c), R(iq_c), n * ih, idd);
      barrier(enc);
      fp4(enc, R(iq_c), n * ih, idd);
      barrier(enc);
      // scores: simdgroup-matrix tiles (ds4_index_scores_mm) when the shape allows ($SUC_IDX_MM=0: the per-entry
      // kernel with the decode step's arithmetic)
      static const bool idx_mm = [] { const char* e = std::getenv("SUC_IDX_MM"); return !(e && e[0] == '0'); }();
      const bool mm = idx_mm && ih % 16 == 0 && ih <= 64 && idd % 32 == 0;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3086, {int64_t(ih), int64_t(idd), int64_t(opt.act), mm ? 1 : 0}}, [&]() {
        return mm ? std::pair<std::string, std::string>{"ds4_index_scores_mm", join({T(), I(long(ih)), I(long(idd)), "0"})}
                  : std::pair<std::string, std::string>{"ds4_index_scores_rows", join({T(), I(long(ih)), I(long(idd))})};
      }));
      set(enc, R(iq_c), 0);
      set(enc, R(L.icomp.ent), 1);
      set(enc, R(iw_c), 2);
      set(enc, R(isc_c), 3);
      const float cw = float(std::pow(double(idd), -0.5) * std::pow(double(ih), -0.5));
      bytes(enc, cw, 4);
      set(enc, R(ne_c, l * opt.chunk * 4), 5);
      const uint32_t ps32 = uint32_t(isc_ps), k32 = uint32_t(K);
      bytes(enc, ps32, 6);
      if (mm) {
        const float one = 1.0f;
        bytes(enc, one, 7);
        enc->dispatchThreadgroups(MTL::Size((ps + 31) / 32, n, 1), MTL::Size(ih / 16 * 32, 1, 1));
      } else {
        enc->dispatchThreadgroups(MTL::Size(ps, n, 1), MTL::Size(32, 1, 1));
      }
      barrier(enc);
      enc->setComputePipelineState(ctx->pipeline("ds4_index_select_rows", "1024"));
      set(enc, R(isc_c), 0);
      set(enc, R(sidx_c), 1);
      set(enc, R(scnt_c), 2);
      set(enc, R(ne_c, l * opt.chunk * 4), 3);
      bytes(enc, k32, 4);
      bytes(enc, ps32, 5);
      enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(1024, 1, 1));
      barrier(enc);
    }
    // the roundings of the decode step's attention kernel, including the probabilities rounded to T (two passes; one
    // pass without that rounding, $SUC_DS4_ATTN_RND=2, gave KL 1.2e-2 vs 0 on the tiny wide file)
    mark(enc, "attn_kernel");
    static const int rnd_env = [] { const char* e = std::getenv("SUC_DS4_ATTN_RND"); return e ? std::atoi(e) : 1; }();
    const int rnd = vector_attention() ? 0 : rnd_env;
    // RND = 1: ds4_attn_chunk_h, HPS heads per simdgroup sharing each key load (same bits); $SUC_DS4_ATTN_HPS=1: one
    // head per simdgroup (ds4_attn_chunk)
    static const int hps_env = [] { const char* e = std::getenv("SUC_DS4_ATTN_HPS"); return e ? std::atoi(e) : 2; }();
    static const int keep_env = [] { const char* e = std::getenv("SUC_DS4_ATTN_KEEP"); return e ? std::atoi(e) : 1; }();
    const int hps = (rnd == 1 && (hps_env == 2 || hps_env == 4) && H % (4 * hps_env) == 0) ? hps_env : 1;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{3085, {int64_t(hd), rnd, int64_t(opt.act), hps, keep_env}}, [&]() {
      return hps > 1 ? std::pair<std::string, std::string>{"ds4_attn_chunk_h", join({T(), I(long(hd)), I(hps), I(keep_env ? 1 : 0)})}
                     : std::pair<std::string, std::string>{"ds4_attn_chunk", join({T(), I(long(hd)), I(rnd)})};
    }));
    set(enc, R(q_c), 0);
    set(enc, R(klin), 1);
    set(enc, L.ratio ? R(L.comp.ent) : ctx->dummy(), 2);
    set(enc, R(ne_c, l * opt.chunk * 4), 3);
    set(enc, L.sink.ref, 4);
    set(enc, R(att_c), 5);
    static const int dbg_w = [] { const char* e = std::getenv("SUC_DBG_WIN"); return e ? std::atoi(e) : 0; }();
    const uint32_t h32 = uint32_t(H), p32 = uint32_t(p0), wp32 = uint32_t(WP), w32 = uint32_t(int(Wn) + dbg_w);
    const float scale = float(std::pow(double(hd), -0.5));
    bytes(enc, h32, 6);
    bytes(enc, p32, 7);
    bytes(enc, wp32, 8);
    bytes(enc, w32, 9);
    bytes(enc, scale, 10);
    set(enc, isel ? R(sidx_c) : ctx->dummy(), 11);
    set(enc, isel ? R(scnt_c) : ctx->dummy(), 12);
    const uint32_t tk32 = isel ? uint32_t(cfg.index_topk) : 0u;
    bytes(enc, tk32, 13);
    enc->dispatchThreadgroups(MTL::Size((H + 4 * hps - 1) / (4 * hps), n, 1), MTL::Size(128, 1, 1));
    barrier(enc);
    {   // ring <- the last min(window, chunk) keys of the chunk
      const std::size_t k = std::min(Wn, n), q0 = p0 + n - k;
      ring_copy(enc, R(klin, (WP + (q0 - p0)) * hd * a), R(L.win), k, uint32_t(q0 % Wn), 1);
    }
    rope_rows(enc, R(att_c), n * H, hd, freq, uint32_t(p0), uint32_t(H), 1, true);
    if (L.ratio) nent[l] = (p0 + n) / L.ratio;
    barrier(enc);
    mark(enc, "attn_wo");
    const std::size_t G = cfg.o_groups, Ro = cfg.o_lora, ing = H * hd / G;
    if (L.wo_a.kind == Mat::Affine) {
      Mat part;
      part.kind = Mat::Affine;
      part.name = L.wo_a.name;
      part.out = Ro;
      part.in = ing;
      part.bits = L.wo_a.bits;
      part.group = L.wo_a.group;
      part.words = L.wo_a.words;
      part.sdtype = L.wo_a.sdtype;
      const std::size_t sb = (ing / part.group) * dtype_size(part.sdtype);
      for (std::size_t g = 0; g < G; ++g) {
        part.packed = L.wo_a.packed.at(g * Ro * part.words * 4);
        part.scales = L.wo_a.scales.at(g * Ro * sb);
        part.biases = L.wo_a.biases.at(g * Ro * sb);
        proj_rows(enc, part, R(att_c, g * ing * a), R(z_c, g * Ro * a), n, H * hd, G * Ro, 0);
      }
    } else if (L.wo_a.kind == Mat::Dense) {
      Mat part;
      part.kind = Mat::Dense;
      part.name = L.wo_a.name;
      part.out = Ro;
      part.in = ing;
      part.wdtype = L.wo_a.wdtype;
      for (std::size_t g = 0; g < G; ++g) {
        part.w = L.wo_a.w.at(g * Ro * ing * dtype_size(L.wo_a.wdtype));
        proj_rows(enc, part, R(att_c, g * ing * a), R(z_c, g * Ro * a), n, H * hd, G * Ro, 0);
      }
    } else {
      for (std::size_t i = 0; i < n; ++i) wo_a_row(enc, L, R(att_c, i * H * hd * a), R(z_c, i * G * Ro * a));
    }
    barrier(enc);
    proj_rows(enc, L.wo_b, R(z_c), R(f_c), n, G * Ro, D, 0);
    barrier(enc);
  }

  // wo_a of one token row (non-affine kinds, = attn()).
  void wo_a_row(MTL::ComputeCommandEncoder* enc, const Layer& L, BufRef att_, BufRef z_) {
    const std::size_t H = cfg.heads, hd = cfg.head_dim, a = as();
    const std::size_t G = cfg.o_groups, Ro = cfg.o_lora, ing = H * hd / G;
    if (L.wo_a.kind == Mat::Suc) {
      proj(enc, L.wo_a, att_, R(woy), 0, G);
      barrier(enc);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3042, {int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_diag", T()};
      }));
      set(enc, R(woy), 0);
      set(enc, z_, 1);
      const uint32_t g32 = uint32_t(G), r32 = uint32_t(Ro);
      bytes(enc, g32, 2);
      bytes(enc, r32, 3);
      enc->dispatchThreads(MTL::Size(Ro, G, 1), MTL::Size(std::min<std::size_t>(256, Ro), 1, 1));
      barrier(enc);
      return;
    }
    Mat part;
    part.kind = Mat::Dense;
    part.name = L.wo_a.name;
    part.out = Ro;
    part.in = ing;
    part.wdtype = L.wo_a.wdtype;
    for (std::size_t g = 0; g < G; ++g) {
      part.w = L.wo_a.w.at(g * Ro * ing * dtype_size(L.wo_a.wdtype));
      proj(enc, part, att_.at(g * ing * a), z_.at(g * Ro * a), 0);
    }
  }

  void ensure_tiles(std::size_t rows, std::size_t tiles) {
    const std::size_t D = cfg.hidden, F = cfg.moe_inter, a = as();
    if (rows > t_rows_cap) {
      t_x = alloc(rows * D * a, nullptr);
      t_g = alloc(rows * F * a, nullptr);
      t_u = alloc(rows * F * a, nullptr);
      t_h = alloc(rows * F * a, nullptr);
      t_y = alloc(rows * D * a, nullptr);
      t_reid = alloc(rows * 4, nullptr);
      t_src = alloc(rows * 4, nullptr);
      t_dst = alloc(rows * 4, nullptr);
      t_w = alloc(rows * 4, nullptr);
      t_rows_cap = rows;
    }
    if (tiles > t_tiles_cap) {
      t_teidx = alloc(tiles * 4, nullptr);
      t_tiles_cap = tiles;
    }
  }

  // Routed experts of the chunk tokens in `group` (routing in eidx_c / ew_c): tiles over the group's (token, k) rows,
  // gather xn rows, gate / up / clamped SwiGLU with the routing weight / down per tile, scatter to yk_c row
  // token · K + k. Same kernels per row as moe(), so the same bits.
  void experts_tiled(MTL::ComputeCommandEncoder* enc, const Layer& L, const std::vector<std::size_t>& group) {
    const std::size_t K = cfg.top_k, D = cfg.hidden, F = cfg.moe_inter, a = as();
    const auto* ids = static_cast<const uint32_t*>(eidx_c->contents());
    const auto* wts = static_cast<const float*>(ew_c->contents());
    std::vector<std::pair<uint32_t, uint32_t>> rows;
    for (std::size_t i : group)
      for (std::size_t k = 0; k < K; ++k) rows.push_back({uint32_t(i * K + k), ids[i * K + k]});
    const std::size_t mb = fast('t') ? std::min({L.eg->fast_tile_rows(), L.eu->fast_tile_rows(), L.ed->fast_tile_rows()})
                                  : std::min({L.eg->max_tile_rows(*ctx), L.eu->max_tile_rows(*ctx), L.ed->max_tile_rows(*ctx)});
    // experts ordered by (gate, up, down) sub-stack: each sub-stack of gate (and mostly up / down, which usually share
    // the gate's codec) then covers a contiguous tile range (ExpertStack::encode_group with host_teidx)
    std::vector<uint32_t> rank(cfg.experts);
    for (std::size_t e = 0; e < cfg.experts; ++e)
      rank[e] = uint32_t((L.eg->sub_slot(e) * 16 + L.eu->sub_slot(e)) * 16 + L.ed->sub_slot(e));
    const TilePlan P = plan_tiles(rows, mb, &rank, -1.0);
    ensure_tiles(P.rows(), P.count());
    auto* src = static_cast<uint32_t*>(t_src->contents());
    auto* dst = static_cast<uint32_t*>(t_dst->contents());
    auto* tw = static_cast<float*>(t_w->contents());
    for (std::size_t j = 0; j < P.rows(); ++j) {
      src[j] = P.src[j] / uint32_t(K);   // token row of xn_c
      dst[j] = P.dst[j];
      tw[j] = wts[P.src[j]];
    }
    std::memcpy(t_reid->contents(), P.eidx.data(), P.rows() * 4);
    std::memcpy(t_teidx->contents(), P.teidx.data(), P.count() * 4);
    Tiles tiles{R(t_teidx), P.count(), P.bt, P.teidx.data()};
    tiles.exact = !fast('t');
    const std::size_t N = P.rows();
    const uint32_t wD = uint32_t(D * a / 4);
    enc->setComputePipelineState(ctx->pipeline("suc_gather_words", "uint32_t"));
    set(enc, R(xn_c), 0);
    set(enc, R(t_x), 1);
    set(enc, R(t_src), 2);
    bytes(enc, wD, 3);
    enc->dispatchThreads(MTL::Size(wD, N, 1), MTL::Size(std::min<uint32_t>(256, wD), 1, 1));
    barrier(enc);
    ExpertStack::encode_group(*ctx, enc, {{L.eu.get(), R(t_x), R(t_u), &es_u}, {L.eg.get(), R(t_x), R(t_g), &es_g}},
                              R(t_reid), N, opt.act, &tiles);
    swiglu(enc, R(t_g), R(t_u), R(t_w), R(t_h), F, N);
    barrier(enc);
    ExpertStack::encode_group(*ctx, enc, {{L.ed.get(), R(t_h), R(t_y), &es_d}}, R(t_reid), N, opt.act, &tiles);
    enc->setComputePipelineState(ctx->pipeline("suc_scatter_words", "uint32_t"));
    set(enc, R(t_y), 0);
    set(enc, R(yk_c), 1);
    set(enc, R(t_dst), 2);
    bytes(enc, wD, 3);
    enc->dispatchThreads(MTL::Size(wD, N, 1), MTL::Size(std::min<uint32_t>(256, wD), 1, 1));
    barrier(enc);
  }

  // Layer-major chunk of n tokens in slots [0, n) at positions p0 ...; the last token's streams in hs.
  void encode_chunk(MTL::ComputeCommandEncoder*& enc, std::size_t n, std::size_t p0, bool with_logits) {
    const std::size_t W = cfg.stream(), D = cfg.hidden, K = cfg.top_k, F = cfg.moe_inter, a = as();
    const std::size_t cw = (2 + cfg.hc) * cfg.hc * 4;
    // fast: every row in the ring (contiguous rows, row-batched launches), the last row copied to hs per layer;
    // exact: the last row in hs as before
    const bool rows_b = fast('h');
    auto xin = [&](std::size_t l, std::size_t i) -> BufRef {
      return (i + 1 == n && !rows_b) ? R(hs, l * W * a) : R(ring, ((l % 2) * opt.chunk + i) * W * a);
    };
    auto to_hs = [&](std::size_t l) {   // hs[l] <- stream row n - 1 of xin(l)
      if (rows_b) encode_copy_cols(*ctx, enc, xin(l, n - 1), R(hs, l * W * a), 1, W, W, W, opt.act);
    };
    if (rows_b) {
      encode_embed(enc, 0, xin(0, 0), n);
      to_hs(0);
    } else {
      for (std::size_t i = 0; i < n; ++i) encode_embed(enc, i, xin(0, i));
    }
    for (std::size_t l = 0; l < cfg.layers; ++l) {
      Layer& L = layers[l];
      mark(enc, "chunk_hc_pre");
      if (!(rows_b && hc_pre_rows_batched(enc, L.ha, xin(l, 0), R(coefa_c), L.attn_norm, R(xn_c), n)))
        hc_pre_rows(enc, L.ha, [&](std::size_t i) { return xin(l, i); }, R(coefa_c), cw, L.attn_norm, R(xn_c), n);
      mark(enc, "chunk_attn");
      attn_chunk(enc, L, l, p0, n);
      mark(enc, "chunk_hc_post_pre");
      if (rows_b) hc_post(enc, R(f_c), xin(l, 0), R(coefa_c), R(xa_c), false, n);
      else
        for (std::size_t i = 0; i < n; ++i)
          hc_post(enc, R(f_c, i * D * a), xin(l, i), R(coefa_c, i * cw), R(xa_c, i * W * a), false);
      barrier(enc);
      if (!(rows_b && hc_pre_rows_batched(enc, L.hf, R(xa_c), R(coeff_c), L.ffn_norm, R(xn_c), n)))
        hc_pre_rows(enc, L.hf, [&](std::size_t i) { return R(xa_c, i * W * a); }, R(coeff_c), cw, L.ffn_norm, R(xn_c), n);
      mark(enc, "chunk_shared_router");
      // MoE: shared expert over the rows; router + routing per token (per-token gates rows, stage-major)
      proj_rows(enc, L.sh_g, R(xn_c), R(shg_c), n, D, F, 1);
      proj_rows(enc, L.sh_u, R(xn_c), R(shu_c), n, D, F, 2);
      if (!(fast('r') && n >= kMmMinRows &&
            encode_mm_dense(*ctx, enc, R(xn_c), L.router.w, R(gates_c), n, D, cfg.experts, L.router.in, L.router.out,
                            msl_of(L.router.wdtype), opt.act, "float")))
        for (std::size_t i = 0; i < n; ++i) gemv_xf(enc, L.router, R(xn_c, i * D * a), R(gates_c, i * cfg.experts * 4));
      barrier(enc);
      if (rows_b) route(enc, L, 0, R(gates_c), R(eidx_c), R(ew_c), n);
      else
        for (std::size_t i = 0; i < n; ++i)
          route(enc, L, i, R(gates_c, i * cfg.experts * 4), R(eidx_c, i * K * 4), R(ew_c, i * K * 4));
      barrier(enc);
      swiglu(enc, R(shg_c), R(shu_c), BufRef{}, R(shh_c), F, n);
      barrier(enc);
      proj_rows(enc, L.sh_d, R(shh_c), R(sh_c), n, F, D, 3);
      if (prof) mark(enc, "chunk_experts");   // (the profile's split is the host sync)
      else split_command_buffer(*ctx, cur_cb, enc, &split_gpu_s);   // the chunk's routing -> host
      std::vector<std::size_t> all(n);
      for (std::size_t i = 0; i < n; ++i) all[i] = i;
      if (!opt.expert_stream) {
        experts_tiled(enc, L, all);
      } else {
        ExpertStream& xs = *opt.expert_stream;
        const auto* ids = static_cast<const uint32_t*>(eidx_c->contents());
        const std::size_t cap = std::max<std::size_t>(xs.max_distinct(L.stream_layer), K);
        // consecutive token groups whose distinct experts fit the slots (any grouping gives the same bits)
        std::vector<char> in_uni(cfg.experts, 0);
        std::vector<std::size_t> group;
        std::vector<uint32_t> gids;
        std::size_t g = 0;
        for (std::size_t i0 = 0; i0 < n;) {
          std::fill(in_uni.begin(), in_uni.end(), 0);
          std::size_t usz = 0;
          group.clear();
          std::size_t i = i0;
          for (; i < n; ++i) {
            std::size_t nw = 0;
            for (std::size_t k = 0; k < K; ++k) nw += !in_uni[ids[i * K + k]];
            if (!group.empty() && usz + nw > cap) break;
            for (std::size_t k = 0; k < K; ++k)
              if (!in_uni[ids[i * K + k]]) in_uni[ids[i * K + k]] = 1, ++usz;
            group.push_back(i);
          }
          if (g++ > 0) split_command_buffer(*ctx, cur_cb, enc, &split_gpu_s);   // the previous group used the slots
          gids.clear();
          for (std::size_t t : group) gids.insert(gids.end(), ids + t * K, ids + (t + 1) * K);
          xs.fetch(L.stream_layer, gids.data(), gids.size());
          ++chunk_fetches;
          experts_tiled(enc, L, group);
          i0 = i;
        }
      }
      mark(enc, "chunk_moe_out_hc_post");
      enc->setComputePipelineState(ctx->pipeline(PsoKey{3052, {int64_t(K), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_moe_out", join({T(), I(long(K))})};
      }));
      for (std::size_t i = 0; i < (rows_b ? 1 : n); ++i) {   // row-batched: one launch, grid y = rows
        set(enc, R(yk_c, i * K * D * a), 0);
        set(enc, R(sh_c, i * D * a), 1);
        set(enc, R(mo_c, i * D * a), 2);
        const uint32_t d32 = uint32_t(D);
        bytes(enc, d32, 3);
        enc->dispatchThreads(MTL::Size(D, rows_b ? n : 1, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
      }
      barrier(enc);
      if (rows_b) hc_post(enc, R(mo_c), R(xa_c), R(coeff_c), xin(l + 1, 0), false, n);
      else
        for (std::size_t i = 0; i < n; ++i)
          hc_post(enc, R(mo_c, i * D * a), R(xa_c, i * W * a), R(coeff_c, i * cw), xin(l + 1, i), false);
      barrier(enc);
      to_hs(l + 1);
    }
    if (rows_out) {   // every row's logits (measurement runs): the head per row, then the last row as usual
      if (!logits_rows) logits_rows = alloc(opt.chunk * cfg.vocab * as(), nullptr);
      for (std::size_t i = 0; i < n; ++i) {
        barrier(enc);
        hc_pre(enc, hc_head, xin(cfg.layers, i), R(coef_a), R(y), norm, R(xn));
        barrier(enc);
        proj(enc, head, R(xn), R(logits_rows, i * cfg.vocab * as()), 0);
      }
    }
    if (with_logits) encode_logits(enc, xin(cfg.layers, n - 1));
  }

  void zero_states() {
    for (Layer& L : layers) {
      for (Comp* c : {&L.comp, &L.icomp}) {
        if (c->kvbuf) std::memset(c->kvbuf->contents(), 0, c->kvbuf->length());
        if (c->scbuf) std::memset(c->scbuf->contents(), 0, c->scbuf->length());
      }
    }
    std::fill(nent.begin(), nent.end(), 0);
    pos = 0;
    have_logits = false;
  }
};

// ------------------------------------------------------------------------------------------ Ds4Model

Ds4Model::Ds4Model(Context& ctx, const MappedFile& mf, const Ds4Options& opt) : p_(std::make_unique<Impl>()) {
  ctx.set_tcq_reuse_default(true);
  Pool pool;
  Impl& m = *p_;
  m.ctx = &ctx;
  m.mf = &mf;
  m.opt = opt;
  if (const char* e = std::getenv("SUC_DS4_PROFILE")) m.prof = e[0] && e[0] != '0';
  const Json* arch = mf.file().directory().find("architecture");
  if (!arch || !arch->is_object()) throw FormatError("directory has no architecture object");
  m.cfg = Ds4Config::from_architecture(*arch);
  const Ds4Config& c = m.cfg;
  if (opt.max_context == 0 || opt.chunk == 0) fail("max_context and chunk must be > 0");
  const std::size_t D = c.hidden, W = c.stream(), H = c.heads, hd = c.head_dim, a = m.as();
  const DType T = dtype_of(opt.act);
  m.eps = float(c.rms_eps);
  m.hc_eps = float(c.hc_eps);
  const auto fb = rope_inv_freq(c.rope_dim, c.rope_theta, false, 1.0, 0, 32.0, 1.0);
  const auto fc = rope_inv_freq(c.rope_dim, c.compress_rope_theta, c.yarn, c.yarn_factor, c.yarn_orig, c.yarn_beta_fast,
                                c.yarn_beta_slow);
  m.freq_base = ctx.upload(fb.data(), fb.size() * 4);
  m.freq_comp = ctx.upload(fc.data(), fc.size() * 4);

  m.embed = m.mat("model.embed_tokens.weight", c.vocab, D, false);
  m.head = m.mat("lm_head.weight", c.vocab, D, true);
  m.norm = m.vec("model.norm.weight", D, T);
  m.hc_head = m.hyper("model.hc_head", true);
  m.layers.resize(c.layers);
  m.nent.assign(c.layers, 0);
  m.max_entries = 1;
  for (std::size_t i = 0; i < c.layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i);
    Impl::Layer& L = m.layers[i];
    L.ratio = c.ratios[i];
    L.idx = L.ratio == 4;
    L.hash = i < c.hash_layers;
    L.ha = m.hyper(p + ".hc_attn", false);
    L.hf = m.hyper(p + ".hc_ffn", false);
    L.attn_norm = m.vec(p + ".attn_norm.weight", D, T);
    L.ffn_norm = m.vec(p + ".ffn_norm.weight", D, T);
    const std::string s = p + ".self_attn";
    L.wq_a = m.mat(s + ".wq_a.weight", c.q_lora, D);
    L.q_norm = m.vec(s + ".q_norm.weight", c.q_lora, T);
    L.wq_b = m.mat(s + ".wq_b.weight", H * hd, c.q_lora);
    L.wkv = m.mat(s + ".wkv.weight", hd, D);
    L.kv_norm = m.vec(s + ".kv_norm.weight", hd, T);
    L.wo_a = m.mat(s + ".wo_a.weight", c.o_groups * c.o_lora, H * hd / c.o_groups);
    // affine wo_a runs per group on the codes in place (suc_gemv_affine on row slices: the decoded weight's values,
    // the dense gemv's order), no decoded copy
    L.wo_b = m.mat(s + ".wo_b.weight", D, c.o_groups * c.o_lora);
    L.sink = m.vec(s + ".attn_sink", H, DType::F32);
    L.win = m.alloc(c.window * hd * a, &m.kv_bytes);
    if (L.ratio) {
      L.comp = m.compressor(s + ".compressor", L.ratio, hd, false);
      m.max_entries = std::max(m.max_entries, opt.max_context / L.ratio + 1);
      if (L.idx) {
        L.icomp = m.compressor(s + ".indexer.compressor", 4, c.index_dim, true);
        L.iwq_b = m.mat(s + ".indexer.wq_b.weight", c.index_heads * c.index_dim, c.q_lora);
        L.iwp = m.mat(s + ".indexer.weights_proj.weight", c.index_heads, D);
      }
    }
    const std::string f = p + ".mlp";
    L.router = m.dense(f + ".gate.weight", c.experts, D, T);
    if (L.hash) {
      const TensorDesc& t = m.tensor(f + ".gate.tid2eid");
      const DType st = t.arrays.at(0).dtype;
      L.tid2eid = m.vec(f + ".gate.tid2eid", c.vocab * c.top_k, (st == DType::I64 || st == DType::I32) ? st : DType::I32);
    } else {
      L.bias = m.vec(f + ".gate.e_score_correction_bias", c.experts, std::nullopt);
    }
    L.sh_g = m.mat(f + ".shared_experts.gate_proj.weight", c.moe_inter, D);
    L.sh_u = m.mat(f + ".shared_experts.up_proj.weight", c.moe_inter, D);
    L.sh_d = m.mat(f + ".shared_experts.down_proj.weight", D, c.moe_inter);
    const bool streamed = opt.expert_stream != nullptr;
    L.eg = std::make_unique<ExpertStack>(ExpertStack::from_file(ctx, mf, f, "gate_proj", c.experts, opt.mixed_rows, streamed));
    L.eu = std::make_unique<ExpertStack>(ExpertStack::from_file(ctx, mf, f, "up_proj", c.experts, opt.mixed_rows, streamed));
    L.ed = std::make_unique<ExpertStack>(ExpertStack::from_file(ctx, mf, f, "down_proj", c.experts, opt.mixed_rows, streamed));
    for (const ExpertStack* e : {L.eg.get(), L.eu.get()})
      if (e->in_features() != D || e->out_features() != c.moe_inter) fail(f + ": expert gate / up shape");
    if (L.ed->in_features() != c.moe_inter || L.ed->out_features() != D) fail(f + ": expert down shape");
    m.copied_bytes += L.eg->copied_bytes() + L.eu->copied_bytes() + L.ed->copied_bytes();
    if (streamed) {   // per expert: {gate, up, down} x {codes / packed indices, scales}
      std::vector<std::vector<ExpertStream::Range>> r(c.experts);
      for (std::size_t e = 0; e < c.experts; ++e)
        for (const ExpertStack* x : {L.eg.get(), L.eu.get(), L.ed.get()}) {
          const auto [a0, a1] = x->expert_ranges(e);
          r[e].push_back(a0);
          r[e].push_back(a1);
        }
      ExpertStack *g = L.eg.get(), *u = L.eu.get(), *d = L.ed.get();
      L.stream_layer = opt.expert_stream->add_layer(f, std::move(r), c.top_k,
                                                    [g, u, d](uint32_t e, MTL::Buffer* pool, const uint64_t* at) {
                                                      g->bind_expert(e, pool, at[0], at[1]);
                                                      u->bind_expert(e, pool, at[2], at[3]);
                                                      d->bind_expert(e, pool, at[4], at[5]);
                                                    });
    }
  }
  if (opt.expert_stream) opt.expert_stream->finalize();
  m.max_keys = c.window + m.max_entries;

  uint64_t* ab = &m.arena_bytes;
  const std::size_t K = c.top_k, F = c.moe_inter;
  m.tok = m.alloc(opt.chunk * c.hc * 4, ab);
  m.hs = m.alloc((c.layers + 1) * W * a, ab);
  m.xa = m.alloc(W * a, ab);
  m.xf = m.alloc(W * 4, ab);
  m.ss = m.alloc(4, ab);
  m.mixraw = m.alloc((2 + c.hc) * c.hc * 4, ab);
  m.coef_a = m.alloc((2 + c.hc) * c.hc * 4, ab);
  m.coef_f = m.alloc((2 + c.hc) * c.hc * 4, ab);
  m.y = m.alloc(D * a, ab);
  m.xn = m.alloc(D * a, ab);
  m.xnf = m.alloc(D * 4, ab);
  m.qa = m.alloc(c.q_lora * a, ab);
  m.qr = m.alloc(c.q_lora * a, ab);
  m.q = m.alloc(H * hd * a, ab);
  m.kvr = m.alloc(hd * a, ab);
  m.kvn = m.alloc(hd * a, ab);
  m.pooled = m.alloc(2 * std::max(hd, c.index_dim) * a, ab);
  m.iq = m.alloc(c.index_heads * c.index_dim * a, ab);
  m.iw = m.alloc(c.index_heads * a, ab);
  m.iscore = m.alloc(m.max_entries * 4, ab);
  m.sel = m.alloc(m.max_entries, ab);
  m.att = m.alloc(H * hd * a, ab);
  m.woy = m.alloc(c.o_groups * c.o_groups * c.o_lora * a, ab);
  m.z = m.alloc(c.o_groups * c.o_lora * a, ab);
  m.f = m.alloc(D * a, ab);
  if (!m.vector_attention()) {
    m.scores = m.alloc(H * (m.max_keys + 1) * a, ab);
    m.probs = m.alloc(H * (m.max_keys + 1) * a, ab);
    if (H > 1)   // <= 32 split-k partitions of (H, <= 512 keys) / (H, head_dim), or one of (H, keys)
      m.cpart = m.alloc(4 * H * std::max({32 * std::min<std::size_t>(m.max_keys, 512), m.max_keys, 32 * hd}), ab);
  }
  m.gates = m.alloc(c.experts * 4, ab);
  m.eidx = m.alloc(K * 4, ab);
  m.ew = m.alloc(K * 4, ab);
  if (opt.expert_stream && opt.expert_stream->options().prefetch) {
    m.pgates = m.alloc(c.experts * 4, ab);
    m.peidx = m.alloc(K * 4, ab);
    m.pew = m.alloc(K * 4, ab);
  }
  m.xk = m.alloc(K * D * a, ab);
  m.gk = m.alloc(K * F * a, ab);
  m.uk = m.alloc(K * F * a, ab);
  m.hk = m.alloc(K * F * a, ab);
  m.yk = m.alloc(K * D * a, ab);
  m.shg = m.alloc(F * a, ab);
  m.shu = m.alloc(F * a, ab);
  m.shh = m.alloc(F * a, ab);
  m.sh = m.alloc(D * a, ab);
  m.mo = m.alloc(D * a, ab);
  m.logits = m.alloc(c.vocab * a, ab);
  m.amax = m.alloc(4, ab);
  if (opt.layer_prefill && opt.chunk > 1) {   // batched prefill chunks (tile buffers grow on demand)
    const std::size_t n = opt.chunk, cw = (2 + c.hc) * c.hc * 4, ql = c.q_lora;
    m.ring = m.alloc(2 * n * W * a, ab);
    m.xa_c = m.alloc(n * W * a, ab);
    m.coefa_c = m.alloc(n * cw, ab);
    m.coeff_c = m.alloc(n * cw, ab);
    m.xn_c = m.alloc(n * D * a, ab);
    m.qa_c = m.alloc(n * ql * a, ab);
    m.qr_c = m.alloc(n * ql * a, ab);
    m.kvr_c = m.alloc(n * hd * a, ab);
    m.kvn_c = m.alloc(n * hd * a, ab);
    m.q_c = m.alloc(n * H * hd * a, ab);
    m.att_c = m.alloc(n * H * hd * a, ab);
    m.z_c = m.alloc(n * c.o_groups * c.o_lora * a, ab);
    m.f_c = m.alloc(n * D * a, ab);
    m.shg_c = m.alloc(n * F * a, ab);
    m.shu_c = m.alloc(n * F * a, ab);
    m.shh_c = m.alloc(n * F * a, ab);
    m.sh_c = m.alloc(n * D * a, ab);
    m.mo_c = m.alloc(n * D * a, ab);
    m.eidx_c = m.alloc(n * K * 4, ab);
    m.ew_c = m.alloc(n * K * 4, ab);
    m.yk_c = m.alloc(n * K * D * a, ab);
    m.xf_c = m.alloc(n * W * 4, ab);
    m.ss_c = m.alloc(n * 16, ab);
    m.mixraw_c = m.alloc(n * cw, ab);
    m.y_c = m.alloc(n * D * a, ab);
    m.gates_c = m.alloc(n * c.experts * 4, ab);
    if (m.fast()) {
      std::size_t cdm = 0, dm = 0;
      for (const auto& L : m.layers)
        for (const auto* cp : {&L.comp, &L.icomp})
          if (cp->ratio) cdm = std::max(cdm, cp->cd), dm = std::max(dm, cp->d);
      m.klin = m.alloc((c.window + n) * hd * a, ab);
      m.ne_c = m.alloc(c.layers * n * 4, ab);
      if (cdm) {
        m.ckv_c = m.alloc(n * cdm * 4, ab);
        m.csc_c = m.alloc(n * cdm * 4, ab);
        m.ickv_c = m.alloc(n * cdm * 4, ab);
        m.icsc_c = m.alloc(n * cdm * 4, ab);
        m.pooled_c = m.alloc(2 * (n / 2 + 2) * dm * a, ab);
      }
    }
  }
}

Ds4Model::~Ds4Model() = default;

const Ds4Config& Ds4Model::config() const { return p_->cfg; }
std::size_t Ds4Model::position() const { return p_->pos; }
void Ds4Model::reset() { p_->zero_states(); }
double Ds4Model::last_gpu_seconds() const { return p_->gpu_s; }
double Ds4Model::last_encode_seconds() const { return p_->encode_s; }
std::vector<Ds4ProfileRow> Ds4Model::profile() const {
  std::vector<Ds4ProfileRow> r;
  for (const auto& [k, v] : p_->prof_acc) r.push_back({k, v.first, v.second});
  return r;
}
void Ds4Model::reset_profile() { p_->prof_acc.clear(); }

uint32_t Ds4Model::forward(const std::vector<uint32_t>& tokens, bool with_logits) {
  Impl& m = *p_;
  const Ds4Config& c = m.cfg;
  if (tokens.empty()) fail("forward: no tokens");
  if (m.pos + tokens.size() > m.opt.max_context)
    fail("context full (max_context " + std::to_string(m.opt.max_context) + ")");
  for (uint32_t t : tokens)
    if (t >= c.vocab) fail("token id " + std::to_string(t) + " >= vocab " + std::to_string(c.vocab));
  uint32_t result = 0;
  m.encode_s = m.gpu_s = 0.0;
  for (std::size_t s0 = 0; s0 < tokens.size(); s0 += m.opt.chunk) {
    const std::size_t n = std::min(m.opt.chunk, tokens.size() - s0);
    const bool last_chunk = s0 + n == tokens.size();
    Pool pool;
    auto* tk = static_cast<uint32_t*>(m.tok->contents());
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t h = 0; h < c.hc; ++h) tk[i * c.hc + h] = tokens[s0 + i];
    const auto t_enc = std::chrono::steady_clock::now();
    MTL::CommandBuffer* cb = m.ctx->queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
    m.cur_cb = cb;
    m.split_gpu_s = 0.0;
    if (m.ring && n > 1) {
      m.encode_chunk(enc, n, m.pos, with_logits && last_chunk);
    } else {
      for (std::size_t i = 0; i < n; ++i) {
        const bool logits_here = with_logits && last_chunk && i + 1 == n;
        m.encode_step(enc, i, m.pos + i, logits_here);
        barrier(enc);
      }
    }
    cb = m.cur_cb;
    enc->endEncoding();
    m.encode_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_enc).count();
    cb->commit();
    cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted) {
      auto* e = cb->error();
      fail("command buffer failed: " + std::string(e ? e->localizedDescription()->utf8String() : "?"));
    }
    m.gpu_s += m.split_gpu_s + cb->GPUEndTime() - cb->GPUStartTime();
    m.prof_add(cb->GPUEndTime() - cb->GPUStartTime());
    m.prof_label = nullptr;
    if (m.rows_out) {
      const std::size_t V = c.vocab, o = m.rows_out->size();
      m.rows_out->resize(o + n * V);
      if (m.ring && n > 1) act_to_f32(m.logits_rows->contents(), m.rows_out->data() + o, n * V, m.opt.act);
      else if (n == 1) act_to_f32(m.logits->contents(), m.rows_out->data() + o, V, m.opt.act);
      else fail("forward_rows: chunks of decode steps have no per-row logits");
    }
    m.pos += n;
  }
  m.last_token = tokens.back();
  m.have_logits = with_logits;
  if (with_logits) result = *static_cast<const uint32_t*>(m.amax->contents());
  return result;
}

std::vector<float> Ds4Model::forward_rows(const std::vector<uint32_t>& tokens) {
  std::vector<float> out;
  p_->rows_out = &out;
  try {
    forward(tokens, true);
  } catch (...) {
    p_->rows_out = nullptr;
    throw;
  }
  p_->rows_out = nullptr;
  return out;
}

std::vector<float> Ds4Model::logits() const {
  if (!p_->have_logits) return {};
  std::vector<float> v(p_->cfg.vocab);
  act_to_f32(p_->logits->contents(), v.data(), v.size(), p_->opt.act);
  return v;
}

std::vector<float> Ds4Model::hidden_states() const {
  const Ds4Config& c = p_->cfg;
  const std::size_t W = c.stream(), X = c.token_channels(), L1 = c.layers + 1;
  std::vector<float> s(L1 * W);
  act_to_f32(p_->hs->contents(), s.data(), s.size(), p_->opt.act);
  if (!X) return s;
  std::vector<float> v(L1 * (W + X));
  const uint32_t t = p_->last_token;
  const float dg[3] = {float(t % 64), float((t / 64) % 64), float(t / 4096)};
  for (std::size_t l = 0; l < L1; ++l) {
    std::copy(s.begin() + std::ptrdiff_t(l * W), s.begin() + std::ptrdiff_t((l + 1) * W), v.begin() + std::ptrdiff_t(l * (W + X)));
    for (std::size_t k = 0; k < X; ++k) v[l * (W + X) + W + k] = dg[k];
  }
  return v;
}

Ds4Memory Ds4Model::memory() const {
  Ds4Memory r;
  r.mapped_file = p_->mf->mapped_bytes();
  r.copied_weights = p_->copied_bytes;
  r.kv_cache = p_->kv_bytes;
  r.arena = p_->arena_bytes;
  uint64_t ws = 0;
  for (const Workspace& w : p_->ws) ws += w.xr_bytes + w.y32_bytes + w.y32b_bytes;
  r.workspace = ws + p_->es_g.bytes() + p_->es_u.bytes() + p_->es_d.bytes();
  return r;
}

std::size_t Ds4Model::suc_projections() const {
  std::size_t n = 0;
  auto add = [&](const Mat& m) { n += m.kind == Mat::Suc; };
  for (const auto& L : p_->layers) {
    for (const Mat* m : {&L.wq_a, &L.wq_b, &L.wkv, &L.wo_a, &L.wo_b, &L.iwq_b, &L.iwp, &L.sh_g, &L.sh_u, &L.sh_d}) add(*m);
  }
  add(p_->head);
  return n;
}

uint64_t Ds4Model::chunk_fetches() const { return p_->chunk_fetches; }

Ds4StackStats Ds4Model::expert_stacks() const {
  Ds4StackStats s;
  for (const auto& L : p_->layers)
    for (const ExpertStack* e : {L.eg.get(), L.eu.get(), L.ed.get()}) {
      ++s.stacks;
      s.mixed += e->mixed();
      s.native_only += e->native_only();
      s.native_experts += e->native_experts();
      s.sub_stacks += e->sub_stacks();
    }
  return s;
}

}  // namespace suc::metal
