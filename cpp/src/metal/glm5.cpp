// glm5_next model graph on the Metal backend (see suc/glm5.hpp). Kernels: cpp/metal/suc_glm5_kernels.metal (the
// glm5_next ops), cpp/metal/suc_ds4_kernels.metal (mixer input reduce, stream collapse, k-pool pooling, clamped SwiGLU,
// softmax), cpp/metal/suc_qwen4_kernels.metal (top-k pool selection), cpp/metal/suc_model_kernels.metal (embedding
// gather, MLX-order gemv, RMSNorm, argmax, casts) and the SUC projection / expert-stack kernels of metal::Linear /
// metal::ExpertStack.
#include "suc/glm5.hpp"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <optional>
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

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("glm5_next: " + msg); }

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
    default: throw FormatError("glm5_next: unsupported dtype " + std::string(dtype_name(t)));
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
  if (!v || !v->is_int()) throw FormatError(std::string("glm5_next architecture: missing integer '") + key + "'");
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
std::string get_str_or(const Json& a, const char* key, const std::string& dflt) {
  const Json* v = a.find(key);
  return v && v->is_string() ? v->as_string() : dflt;
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

// MLX gemv tiling (non-transposed matrix): SN = 4 for in <= 64 else 32, BN = 8 when in >= 16 · out, TN = 4
int gemv_sn(std::size_t in) { return in <= 64 ? 4 : 32; }
int gemv_bn(std::size_t in, std::size_t out) { return (in > 64 && in >= 16 * out) ? 8 : 1; }

// MLX's GEMM dispatch for a folded matmul (M, N, K): split-k when K >= max(M, N) and K / 16 >= 8 (partitions
// min(max(2, next_pow2((K / 16) / (ceil(M / 32) · ceil(N / 32)))), 32) of ((K / 16) / P) · 16 values), else the fused
// kernel whose unaligned K tail (K % BK) runs first. BK follows MLX's GEMM_TPARAM_MACRO per device class (last letter
// of the GPU architecture: 'g' / 'p' small, 'd' large); measured on the M3 ('g') against MLX 0.32.3, the 'd' table as
// recalled (only float32 nn GEMMs with K % 16 != 0 depend on it).
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

}  // namespace

// ------------------------------------------------------------------------------------------ config

bool Glm5Config::is_glm5(const Json& arch) {
  const Json* mt = arch.find("model_type");
  return mt && mt->is_string() && (mt->as_string() == "glm5_next" || mt->as_string() == "glm5_next_text");
}

Glm5Config Glm5Config::from_architecture(const Json& root) {
  if (!is_glm5(root)) throw FormatError("architecture: model_type is not glm5_next");
  const Json* tc = root.find("text_config");
  const Json& a = tc && tc->is_object() ? *tc : root;
  Glm5Config c;
  c.hidden = get_size(a, "hidden_size");
  c.layers = get_size(a, "num_hidden_layers");
  c.vocab = get_size(a, "vocab_size");
  c.intermediate = get_size_or(a, "intermediate_size", 12288);
  const Json* la = a.find("linear_attn_config");
  const Json empty;
  const Json& lac = la && la->is_object() ? *la : empty;
  c.lin_heads = la ? get_size_or(lac, "num_heads", get_size_or(a, "linear_num_heads", 64)) : get_size_or(a, "linear_num_heads", 64);
  c.lin_dim = la ? get_size_or(lac, "head_dim", get_size_or(a, "linear_head_dim", 128)) : get_size_or(a, "linear_head_dim", 128);
  c.conv_kernel = la ? get_size_or(lac, "short_conv_kernel_size", get_size_or(a, "linear_conv_kernel_dim", 4))
                     : get_size_or(a, "linear_conv_kernel_dim", 4);
  {
    const Json* lb = la ? lac.find("gate_lower_bound") : nullptr;
    if (!lb) lb = a.find("linear_lower_bound");
    if (lb && lb->is_null()) throw FormatError("glm5_next: KDA without the safe gate lower bound");
    c.lower_bound = lb && lb->is_number() ? lb->as_double() : -5.0;
  }
  // layer kinds (TextArgs.__post_init__)
  c.dsa.assign(c.layers, false);
  const Json* lt = a.find("layer_types");
  if (lt && lt->is_array() && lt->size() > 0) {
    if (lt->size() < c.layers) throw FormatError("glm5_next: layer_types shorter than num_hidden_layers");
    for (std::size_t i = 0; i < c.layers; ++i) {
      const std::string t = lt->at(i).as_string();
      if (t == "deepseek_sparse_attention" || t == "indexed_attention" || t == "full_attention") c.dsa[i] = true;
      else if (t != "linear_attention") throw FormatError("glm5_next: unsupported layer type '" + t + "'");
    }
  } else if (const Json* kl = la ? lac.find("kda_layers") : nullptr; kl && kl->is_array()) {
    for (std::size_t i = 0; i < c.layers; ++i) c.dsa[i] = true;
    for (std::size_t j = 0; j < kl->size(); ++j) {
      const std::size_t i = std::size_t(kl->at(j).as_uint());
      if (i < c.layers) c.dsa[i] = false;
    }
  } else {
    for (std::size_t i = 0; i < c.layers; ++i) c.dsa[i] = i % 4 == 3;
  }
  c.sparse.assign(c.layers, false);
  const std::size_t fkd = get_size_or(a, "first_k_dense_replace", 3);
  const Json* mt = a.find("mlp_layer_types");
  if (mt && mt->is_array() && mt->size() > 0) {
    if (mt->size() < c.layers) throw FormatError("glm5_next: mlp_layer_types shorter than num_hidden_layers");
    for (std::size_t i = 0; i < c.layers; ++i) c.sparse[i] = mt->at(i).as_string() == "sparse";
  } else {
    for (std::size_t i = 0; i < c.layers; ++i) c.sparse[i] = i >= fkd;
  }
  if (const Json* it = a.find("indexer_types"); it && it->is_array())
    for (std::size_t i = 0; i < std::min(c.layers, it->size()); ++i)
      if (it->at(i).as_string() != "full") throw FormatError("glm5_next: shared (cross-layer) indexer top-k is not supported");
  c.heads = get_size(a, "num_attention_heads");
  c.q_lora = get_size(a, "q_lora_rank");
  c.kv_lora = get_size(a, "kv_lora_rank");
  c.nope = get_size(a, "qk_nope_head_dim");
  c.v_dim = get_size(a, "v_head_dim");
  if (get_size_or(a, "qk_rope_head_dim", 0) != 0) throw FormatError("glm5_next: RoPE (qk_rope_head_dim > 0) is not supported");
  c.index_heads = get_size_or(a, "index_n_heads", 32);
  c.index_dim = get_size_or(a, "index_head_dim", 128);
  c.index_topk = get_size_or(a, "index_topk", 2048);
  c.kpool = get_size_or(a, "index_kpool", 4);
  c.select_tail = get_bool_or(a, "index_kpool_always_select_tail", true);
  c.experts = get_size_or(a, "n_routed_experts", 0);
  c.top_k = get_size_or(a, "num_experts_per_tok", 0);
  c.moe_inter = get_size_or(a, "moe_intermediate_size", 0);
  c.norm_topk_prob = get_bool_or(a, "norm_topk_prob", true);
  c.route_scale = get_num_or(a, "routed_scaling_factor", 2.5);
  c.swiglu_limit = get_num_or(a, "swiglu_limit", 10.0);
  c.hc = get_size_or(a, "hc_mult", 4);
  c.sinkhorn_iters = get_size_or(a, "hc_sinkhorn_iters", 20);
  c.hc_eps = get_num_or(a, "hc_eps", 1e-6);
  c.rms_eps = get_num_or(a, "rms_norm_eps", 1e-5);
  if (get_str_or(a, "scoring_func", "sigmoid") != "sigmoid" || get_size_or(a, "n_group", 1) != 1 ||
      get_size_or(a, "topk_group", 1) != 1)
    throw FormatError("glm5_next: router other than sigmoid / one group");
  if (get_size_or(a, "n_shared_experts", 1) != 1) throw FormatError("glm5_next: n_shared_experts != 1");
  if (get_bool_or(a, "attention_bias", false)) throw FormatError("glm5_next: attention bias is not supported");
  if (get_bool_or(a, "tie_word_embeddings", false) || get_bool_or(root, "tie_word_embeddings", false))
    throw FormatError("glm5_next: tied embeddings are not supported");
  if (!c.norm_topk_prob) throw FormatError("glm5_next: norm_topk_prob = false is not supported");
  for (const Json* e : {a.find("eos_token_id"), root.find("eos_token_id")}) {
    if (!e || e->is_null() || !c.eos.empty()) continue;
    if (e->is_array())
      for (std::size_t i = 0; i < e->size(); ++i) c.eos.push_back(uint32_t(e->at(i).as_uint()));
    else if (e->is_int()) c.eos.push_back(uint32_t(e->as_uint()));
  }
  c.torch_dtype = get_str_or(root, "torch_dtype", get_str_or(a, "dtype", get_str_or(root, "dtype", "")));
  // shape checks of the kernels
  if (c.index_topk % c.kpool) throw FormatError("glm5_next: index_topk must be a multiple of index_kpool");
  if (c.hc != 4) throw FormatError("glm5_next: hc_mult must be 4");
  if (c.lin_dim % 32 || c.lin_dim > 1024) throw FormatError("glm5_next: linear head_dim must be a multiple of 32, <= 1024");
  if (c.conv_kernel < 2 || c.conv_kernel > 8) throw FormatError("glm5_next: short_conv_kernel_size");
  if (c.index_dim > 4096) throw FormatError("glm5_next: index_head_dim > 4096");
  if (c.index_heads < 2) throw FormatError("glm5_next: index_n_heads < 2 (MLX's gemv path) is not supported");
  if (c.kv_lora % 32) throw FormatError("glm5_next: kv_lora_rank must be a multiple of 32");
  bool any_sparse = false;
  for (bool s : c.sparse) any_sparse = any_sparse || s;
  if (any_sparse && (c.experts == 0 || c.top_k == 0 || c.experts > 1024 || c.top_k > 16))
    throw FormatError("glm5_next: MoE needs 1..1024 experts and top_k 1..16");
  return c;
}

// ------------------------------------------------------------------------------------------ model

struct Glm5Model::Impl {
  Context* ctx = nullptr;
  const MappedFile* mf = nullptr;
  Glm5Options opt;
  Glm5Config cfg;

  struct Hc {
    Vec fn;            // (mix, hc · D), float dtype as stored
    Vec base, scale;   // float32
  };
  struct Layer {
    bool dsa = false, sparse = false;
    Hc ha, hf;
    Vec in_norm, post_norm;
    // KDA
    Mat q, k, v, fa, fb, ga, gb, bproj, o;
    BufPtr convw;      // (3d, K) float32
    Vec dt_bias, a_log;
    BufPtr onorm;      // (Dk) float32 (the T weight cast up, as the Python model)
    BufPtr conv_state, rec_state;
    // MLA + DSA
    Mat q_a, q_b, kv_a, iwq_b, iwk, iwp, igate;
    Vec q_a_norm, kv_a_norm, ik_w, ik_b, ape;
    Vec kv_b;          // (H · (nope + v), rank) in T
    BufPtr latent, rawk, rawg, pools;
    // MLP / MoE
    Mat mg, mu, md;    // dense MLP or shared expert
    Mat router;        // dense (E, D) in T, float32 inputs
    Vec bias;
    std::unique_ptr<ExpertStack> eg, eu, ed;
    int stream_layer = -1;
  };
  Mat embed, head;
  Vec norm;
  std::vector<Layer> layers;
  Workspace ws[8];
  ExpertScratch es_g, es_u, es_d;

  // arena
  BufPtr tok, hs, xa, xf, ss, xs, mixraw, coef_a, coef_f, y, xn, xnf, f, mo;
  BufPtr qkv, conv, qkf, vf, fa, fb, ga, gb, bb, decay, beta, o32, on32, oy;
  BufPtr qa, qr, q, ikr, ik, ig, iw, iq, iscore, sel, qabs, att, ov, scores, probs, cpart;
  BufPtr gates, eidx, ew, pgates, peidx, pew, xk, gk, uk, hk, yk, mgb, mub, mhb, sh;
  BufPtr logits, amax;
  BufPtr ring, xa_c, coef_c, xk_c, sh_c, eidx_c, ew_c;   // layer-major prefill (streamed experts): per chunk token
  uint64_t chunk_fetches = 0;
  std::size_t pos = 0, max_pools = 0;
  std::vector<std::size_t> npools;   // complete pools per layer (DSA layers)
  bool have_logits = false;
  double gpu_s = 0.0, encode_s = 0.0, split_gpu_s = 0.0;
  MTL::CommandBuffer* cur_cb = nullptr;
  uint64_t arena_bytes = 0, kv_bytes = 0, state_bytes = 0, copied_bytes = 0;
  float eps = 0.0f, hc_eps = 0.0f;

  std::size_t as() const { return act_size(opt.act); }
  const char* T() const { return act_msl(opt.act); }
  BufPtr alloc(std::size_t bytes, uint64_t* counter) {
    if (counter) *counter += bytes;
    return ctx->new_buffer(std::max<std::size_t>(bytes, 16));
  }
  static BufRef R(const BufPtr& b, std::size_t off = 0) { return {b.get(), off}; }
  const TensorDesc& tensor(const std::string& name) const { return mf->file().tensor(name); }
  bool vector_attention() const {
    const std::size_t D = cfg.kv_lora;
    return (D == 64 || D == 96 || D == 128 || D == 256) && cfg.heads <= 32;
  }

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
    }
    // other dtypes / codecs: the reference decode (float32), cast to the wanted dtype (as load_suc1_model)
    const DecodedTensor d = decode_tensor(mf->file(), t);
    if (d.data.size() != n) fail(name + ": decoded size mismatch");
    const DType dt = want ? *want : DType::F32;
    const Act a = dt == DType::BF16 ? Act::BF16 : dt == DType::F16 ? Act::F16 : Act::F32;
    v.owned = ctx->new_buffer(n * act_size(a));
    f32_to_act(d.data.data(), v.owned->contents(), n, a);
    v.ref = R(v.owned);
    v.dtype = dt;
    copied_bytes += n * act_size(a);
    return v;
  }
  // host float32 copy of a tensor after the cast to `via` (the Python parameter dtype)
  std::vector<float> host(const std::string& name, std::size_t n, DType via) {
    const DecodedTensor d = decode_tensor(mf->file(), tensor(name));
    if (d.data.size() != n) fail(name + ": decoded size mismatch");
    std::vector<float> out(d.data.begin(), d.data.end());
    if (via != DType::F32) {
      const Act a = via == DType::BF16 ? Act::BF16 : Act::F16;
      std::vector<uint16_t> tmp(n);
      f32_to_act(out.data(), tmp.data(), n, a);
      act_to_f32(tmp.data(), out.data(), n, a);
    }
    return out;
  }
  BufPtr upload_f32(const std::vector<float>& v) {
    copied_bytes += v.size() * 4;
    return ctx->upload(v.data(), v.size() * 4);
  }

  Mat dense(const std::string& name, std::size_t out, std::size_t in, std::optional<DType> want) {
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

  Hc hyper(const std::string& p, const char* k) {
    const std::size_t W = cfg.stream(), mix = (2 + cfg.hc) * cfg.hc;
    Hc h;
    h.fn = vec(p + ".hc_" + k + "_fn", mix * W, std::nullopt);
    h.base = vec(p + ".hc_" + k + "_base", mix, DType::F32);
    h.scale = vec(p + ".hc_" + k + "_scale", 3, DType::F32);
    return h;
  }

  // ---- encoders
  // y = x Wᵀ for one row of T (SUC: metal::Linear with workspace ws[wsi]; affine / dense: MLX gemv order).
  void proj(MTL::ComputeCommandEncoder* enc, const Mat& m, BufRef x, BufRef y, int wsi = 0) {
    if (m.kind == Mat::Suc) {
      m.lin->encode(*ctx, enc, x, y, 1, opt.act, ws[wsi], Path::Auto);
      return;
    }
    const int sn = gemv_sn(m.in), bn = gemv_bn(m.in, m.out), sm = 32 / sn;
    if (m.kind == Mat::Affine) {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5001, {int64_t(m.in), int64_t(m.out), m.bits, int64_t(m.group), int64_t(m.sdtype), int64_t(opt.act)}}, [&]() {
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
    const int sn = gemv_sn(m.in), bn = gemv_bn(m.in, m.out);
    const bool f32 = std::string(tx) == "float";
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5002, {int64_t(m.in), int64_t(m.out), int64_t(m.wdtype), f32 ? 1 : int64_t(opt.act) + 10}}, [&]() {
      return std::pair<std::string, std::string>{
          "suc_moe_gemv", join({tx, msl_of(m.wdtype), I(long(m.in)), I(long(m.out)), I(sn), "4", I(bn)})};
    }));
  }
  // y (float32) = x (float32) Wᵀ (dense W in its dtype, MLX gemv order).
  void gemv_f32(MTL::ComputeCommandEncoder* enc, const Mat& m, BufRef x, BufRef y) {
    if (m.kind != Mat::Dense) fail(m.name + ": float32 projections must be dense");
    const int sn = gemv_sn(m.in), bn = gemv_bn(m.in, m.out), sm = 32 / sn;
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
    const int sn = gemv_sn(m.in), bn = gemv_bn(m.in, m.out), sm = 32 / sn;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5016, {int64_t(m.in), int64_t(m.out), int64_t(m.wdtype), int64_t(opt.act)}}, [&]() {
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
  // MLX row_reduce_simple threads (measured, MLX 0.32.3; rows > 64 values)
  static std::size_t red_tpg(std::size_t n) {
    if (n <= 512) return 32;
    if (n <= 1024) return 128;
    if (n <= 2048) return 512;
    return 1024;
  }

  // mx.fast.rms_norm rows of n in `tx` (T or float) with the weight in the same dtype
  void rms(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef w, BufRef y, std::size_t rows, std::size_t n,
           const char* tx = nullptr) {
    const std::size_t tpg = rms_tpg(n);
    const bool f32 = tx != nullptr;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5003, {int64_t(n), int64_t(tpg), f32 ? 99 : int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"suc_rms_norm", join({f32 ? tx : T(), I(long(n)), I(long(tpg))})};
    }));
    set(enc, x, 0);
    set(enc, w, 1);
    set(enc, y, 2);
    bytes(enc, eps, 3);
    enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(tpg, 1, 1));
  }

  void to_f32(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t n) {
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5004, {int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"glm_to_f32", T()};
    }));
    set(enc, x, 0);
    set(enc, y, 1);
    const uint32_t n32 = uint32_t(n);
    bytes(enc, n32, 2);
    enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(std::min<std::size_t>(256, n), 1, 1));
  }

  // mHC pre on the stream x: coef (pre | post | comb) float32, collapsed y (D) in T. Three phases: sum of squares (+ the
  // float copy) | mixer gemv on the normalized stream (the normalization folded into the gemv's input read) |
  // {coefficients on one simdgroup (element-parallel Sinkhorn), collapse with its own pre coefficients}. Same values as
  // the sequential reduce -> scale -> gemv -> one-thread coef -> collapse chain (see the kernels).
  // nw / xn: the following RMSNorm (xn = rms_norm(y, nw)); fused into the collapse when D <= 4096 (y is then not
  // written), else run after it.
  void hc_pre(MTL::ComputeCommandEncoder* enc, const Hc& h, BufRef x, BufRef coef, BufRef yout, const Vec& nw,
              BufRef xnout) {
    const std::size_t W = cfg.stream(), D = cfg.hidden, HC = cfg.hc, mix = (2 + HC) * HC;
    const std::size_t tpg = red_tpg(W);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5010, {int64_t(W), int64_t(tpg), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_hc_in", join({T(), I(long(W)), I(long(tpg))})};
    }));
    set(enc, x, 0);
    set(enc, R(xf), 1);
    set(enc, R(ss), 2);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(tpg, 1, 1));
    barrier(enc);
    const float inv_w = float(1.0 / double(W));
    {
      const int sn = gemv_sn(W), bn = gemv_bn(W, mix), sm = 32 / sn;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5014, {int64_t(W), int64_t(mix), int64_t(h.fn.dtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{
            "ds4_gemv_xf", join({T(), h.fn.msl(), I(long(W)), I(long(mix)), I(sn), "4", I(bn), "1"})};
      }));
      set(enc, x, 0);
      set(enc, h.fn.ref, 1);
      set(enc, R(mixraw), 2);
      set(enc, R(ss), 3);
      bytes(enc, inv_w, 4);
      bytes(enc, eps, 5);
      enc->dispatchThreadgroups(MTL::Size((mix + sm - 1) / sm, 1, 1), MTL::Size(32 * bn, 1, 1));
    }
    barrier(enc);
    const int iters = int(cfg.sinkhorn_iters);
    if (HC * HC > 32) fail("glm5_next: hc_mult > 5 (the coefficient kernel holds hc^2 values in one simdgroup)");
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5012, {int64_t(HC), iters}}, [&]() {
      return std::pair<std::string, std::string>{"glm_hc_coef_par", join({"float", I(long(HC)), I(iters)})};
    }));
    set(enc, R(mixraw), 0);
    set(enc, h.base.ref, 1);
    set(enc, h.scale.ref, 2);
    set(enc, coef, 3);
    bytes(enc, hc_eps, 4);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));
    if (D <= 4096) {
      const std::size_t tpg = rms_tpg(D);
      const float z = 0.0f;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5015, {int64_t(D), int64_t(HC), int64_t(opt.act), int64_t(tpg)}}, [&]() {
        return std::pair<std::string, std::string>{
            "ds4_hc_collapse_rms", join({T(), "float", I(long(D)), I(long(HC)), "2", I(long(tpg))})};
      }));
      set(enc, x, 0);
      set(enc, R(mixraw), 1);
      set(enc, R(ss), 2);
      set(enc, h.base.ref, 3);
      set(enc, h.scale.ref, 4);
      set(enc, nw.ref, 5);
      set(enc, xnout, 6);
      bytes(enc, inv_w, 7);
      bytes(enc, z, 8);
      bytes(enc, hc_eps, 9);
      bytes(enc, eps, 10);
      enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(tpg, 1, 1));
      barrier(enc);
      return;
    }
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5013, {int64_t(D), int64_t(HC), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"glm_hc_collapse_pre", join({T(), "float", I(long(D)), I(long(HC))})};
    }));
    set(enc, x, 0);
    set(enc, R(mixraw), 1);
    set(enc, h.base.ref, 2);
    set(enc, h.scale.ref, 3);
    set(enc, yout, 4);
    bytes(enc, hc_eps, 5);
    enc->dispatchThreads(MTL::Size(D, 1, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    barrier(enc);
    rms(enc, yout, nw.ref, xnout, 1, D);
  }

  void hc_post(MTL::ComputeCommandEncoder* enc, BufRef fbuf, BufRef res, BufRef coef, BufRef yout) {
    const std::size_t D = cfg.hidden, HC = cfg.hc;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5014, {int64_t(D), int64_t(HC), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"glm_hc_post", join({T(), I(long(D)), I(long(HC))})};
    }));
    set(enc, fbuf, 0);
    set(enc, res, 1);
    set(enc, coef, 2);
    set(enc, yout, 3);
    enc->dispatchThreads(MTL::Size(D, HC, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    barrier(enc);
  }

  // ---- KDA: xn (D) -> f (D)
  void kda(MTL::ComputeCommandEncoder* enc, Layer& L) {
    const std::size_t H = cfg.lin_heads, Dk = cfg.lin_dim, d = H * Dk, a = as(), K = cfg.conv_kernel;
    proj(enc, L.q, R(xn), R(qkv), 0);
    proj(enc, L.k, R(xn), R(qkv, d * a), 1);
    proj(enc, L.v, R(xn), R(qkv, 2 * d * a), 2);
    proj(enc, L.fa, R(xn), R(fa), 3);
    proj(enc, L.ga, R(xn), R(ga), 3);
    proj(enc, L.bproj, R(xn), R(bb), 3);
    barrier(enc);
    {
      const std::size_t C = 3 * d;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5020, {int64_t(C), int64_t(K), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_kda_conv", join({T(), I(long(C)), I(long(K))})};
      }));
      set(enc, R(qkv), 0);
      set(enc, R(L.conv_state), 1);
      set(enc, R(L.convw), 2);
      set(enc, R(conv), 3);
      enc->dispatchThreads(MTL::Size(C, 1, 1), MTL::Size(std::min<std::size_t>(256, C), 1, 1));
    }
    proj(enc, L.fb, R(fa), R(fb), 3);
    proj(enc, L.gb, R(ga), R(gb), 3);
    barrier(enc);
    {
      const std::size_t tpg = Dk <= 64 ? 32 : red_tpg(Dk);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5021, {int64_t(Dk), int64_t(tpg), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_l2norm", join({T(), I(long(Dk)), I(long(tpg))})};
      }));
      set(enc, R(conv), 0);
      set(enc, R(qkf), 1);
      const uint32_t qrows = uint32_t(H);
      const float qscale = float(std::pow(double(Dk), -0.5));
      bytes(enc, qrows, 2);
      bytes(enc, qscale, 3);
      enc->dispatchThreadgroups(MTL::Size(2 * H, 1, 1), MTL::Size(tpg, 1, 1));
    }
    to_f32(enc, R(conv, 2 * d * a), R(vf), d);
    {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5022, {int64_t(L.a_log.dtype), int64_t(L.dt_bias.dtype), int64_t(Dk), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_kda_gate", join({T(), L.a_log.msl(), L.dt_bias.msl(), I(long(Dk))})};
      }));
      set(enc, R(fb), 0);
      set(enc, L.dt_bias.ref, 1);
      set(enc, L.a_log.ref, 2);
      set(enc, R(bb), 3);
      set(enc, R(decay), 4);
      set(enc, R(beta), 5);
      const uint32_t n = uint32_t(d), heads = uint32_t(H);
      const float lb = float(cfg.lower_bound);
      bytes(enc, n, 6);
      bytes(enc, heads, 7);
      bytes(enc, lb, 8);
      enc->dispatchThreads(MTL::Size(d, 1, 1), MTL::Size(std::min<std::size_t>(256, d), 1, 1));
    }
    barrier(enc);
    {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5023, {int64_t(Dk), int64_t(H), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_gated_delta", join({T(), I(long(Dk)), I(long(Dk)), I(long(H)), I(long(H))})};
      }));
      set(enc, R(qkf), 0);
      set(enc, R(qkf, d * 4), 1);
      set(enc, R(vf), 2);
      set(enc, R(decay), 3);
      set(enc, R(beta), 4);
      set(enc, R(L.rec_state), 5);
      set(enc, R(o32), 6);
      enc->dispatchThreads(MTL::Size(32, Dk, H), MTL::Size(32, 4, 1));
    }
    barrier(enc);
    rms(enc, R(o32), R(L.onorm), R(on32), H, Dk, "float");
    barrier(enc);
    {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5024, {int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_sig_mul", T()};
      }));
      set(enc, R(on32), 0);
      set(enc, R(gb), 1);
      set(enc, R(oy), 2);
      const uint32_t n = uint32_t(d);
      bytes(enc, n, 3);
      enc->dispatchThreads(MTL::Size(d, 1, 1), MTL::Size(std::min<std::size_t>(256, d), 1, 1));
    }
    barrier(enc);
    proj(enc, L.o, R(oy), R(f), 0);
    barrier(enc);
  }

  // ---- MLA + DSA indexer at position p: xn (D) -> f (D)
  void mla(MTL::ComputeCommandEncoder* enc, Layer& L, std::size_t l, std::size_t p) {
    const std::size_t H = cfg.heads, nope = cfg.nope, vd = cfg.v_dim, rank = cfg.kv_lora, a = as();
    const std::size_t ih = cfg.index_heads, idd = cfg.index_dim, kp = cfg.kpool;
    proj(enc, L.q_a, R(xn), R(qa), 0);
    proj(enc, L.kv_a, R(xn), R(f), 1);   // f: scratch for the raw latent
    proj(enc, L.iwk, R(xn), R(ikr), 2);
    proj(enc, L.igate, R(xn), R(ig), 3);
    proj(enc, L.iwp, R(xn), R(iw), 3);
    barrier(enc);
    rms(enc, R(qa), L.q_a_norm.ref, R(qr), 1, cfg.q_lora);
    rms(enc, R(f), L.kv_a_norm.ref, R(L.latent, p * rank * a), 1, rank);
    {
      const std::size_t tpg = 32 * (((idd + 3) / 4 + 31) / 32);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5030, {int64_t(idd), int64_t(tpg), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_layer_norm", join({T(), I(long(idd)), I(long(tpg))})};
      }));
      set(enc, R(ikr), 0);
      set(enc, L.ik_w.ref, 1);
      set(enc, L.ik_b.ref, 2);
      set(enc, R(ik), 3);
      const float le = 1e-6f;
      bytes(enc, le, 4);
      enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(tpg, 1, 1));
    }
    to_f32(enc, R(ig), R(L.rawg, (p % kp) * idd * 4), idd);
    barrier(enc);
    proj(enc, L.q_b, R(qr), R(q), 0);
    to_f32(enc, R(ik), R(L.rawk, (p % kp) * idd * 4), idd);
    const bool emit = (p + 1) % kp == 0;
    std::size_t np = npools[l];
    if (emit) ++np;
    const bool use_sel = np > cfg.topk_pools();
    if (use_sel) proj(enc, L.iwq_b, R(qr), R(iq), 1);
    barrier(enc);
    if (emit) {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5031, {int64_t(idd), int64_t(kp), int64_t(L.ape.dtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_comp_pool", join({T(), L.ape.msl(), I(long(idd)), I(long(kp)), "0"})};
      }));
      set(enc, R(L.rawk), 0);
      set(enc, R(L.rawg), 1);
      set(enc, R(L.rawk), 2);
      set(enc, R(L.rawg), 3);
      set(enc, L.ape.ref, 4);
      set(enc, R(L.pools, (np - 1) * idd * a), 5);
      const uint32_t has_prev = 0;
      bytes(enc, has_prev, 6);
      enc->dispatchThreadgroups(MTL::Size(idd, 1, 1), MTL::Size(32, 1, 1));
      barrier(enc);
    }
    if (use_sel) {   // s = q (ih, idd) @ poolsᵀ (float32 operands: one MLX GEMM), relu, head-weighted sum
      const std::size_t parts = mma(enc, R(iq), dtype_of(opt.act), R(L.pools), dtype_of(opt.act), ih, np, idd, idd, idd,
                                    true, true, 0.0f);
      barrier(enc);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5032, {int64_t(ih), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_index_reduce", join({T(), I(long(ih))})};
      }));
      set(enc, R(cpart), 0);
      set(enc, R(iw), 1);
      set(enc, R(iscore), 2);
      const uint32_t p32 = uint32_t(np), parts32 = uint32_t(parts);
      const float dscale = float(std::pow(double(idd), -0.5)), hscale = float(std::pow(double(ih), -0.5));
      bytes(enc, p32, 3);
      bytes(enc, parts32, 4);
      bytes(enc, dscale, 5);
      bytes(enc, hscale, 6);
      enc->dispatchThreadgroups(MTL::Size(np, 1, 1), MTL::Size(32, 1, 1));
      barrier(enc);
      enc->setComputePipelineState(ctx->pipeline("q4_index_select", "1024"));
      set(enc, R(iscore), 0);
      set(enc, R(sel), 1);
      const uint32_t n32 = uint32_t(np), k32 = uint32_t(cfg.topk_pools());
      bytes(enc, n32, 2);
      bytes(enc, k32, 3);
      enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
    }
    // q'_h = q_h @ W_k,h (gemv_t: K = nope, N = rank)
    {
      const int tsm = (nope >= 8192 && rank >= 2048) ? 4 : 8, tsn = (nope >= 8192 && rank >= 2048) ? 8 : 4;
      const int tbn = rank >= 2048 ? 16 : rank >= 512 ? 4 : 2, ttn = rank < 4 ? 1 : 4;
      const std::size_t blockN = std::size_t(tbn * tsn * ttn);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5033, {int64_t(rank), tsm, tsn, tbn, ttn, int64_t(L.kv_b.dtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{
            "glm_gemv_t", join({T(), L.kv_b.msl(), I(long(rank)), I(tsm), I(tsn), I(tbn), "4", I(ttn)})};
      }));
      set(enc, R(q), 0);
      set(enc, L.kv_b.ref, 1);
      set(enc, R(qabs), 2);
      const uint32_t kk = uint32_t(nope), xhs = uint32_t(nope);
      const uint64_t whs = uint64_t((nope + vd) * rank);
      bytes(enc, kk, 3);
      bytes(enc, xhs, 4);
      bytes(enc, whs, 5);
      enc->dispatchThreadgroups(MTL::Size((rank + blockN - 1) / blockN, H, 1), MTL::Size(32 * tbn, 1, 1));
    }
    barrier(enc);
    attention(enc, L, p + 1, np, use_sel);
    barrier(enc);
    // o_h = W_v,h (P c) (gemv: in = rank, out = v)
    {
      const int sn = gemv_sn(rank), bn = gemv_bn(rank, vd), sm = 32 / sn;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5034, {int64_t(rank), int64_t(vd), sn, bn, int64_t(L.kv_b.dtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{
            "glm_gemv_b", join({T(), L.kv_b.msl(), I(long(rank)), I(long(vd)), I(sn), "4", I(bn)})};
      }));
      set(enc, R(att), 0);
      set(enc, L.kv_b.ref.at(nope * rank * dtype_size(L.kv_b.dtype)), 1);
      set(enc, R(ov), 2);
      const uint64_t whs = uint64_t((nope + vd) * rank);
      bytes(enc, whs, 3);
      enc->dispatchThreadgroups(MTL::Size((vd + sm - 1) / sm, H, 1), MTL::Size(32 * bn, 1, 1));
    }
    barrier(enc);
    proj(enc, L.o, R(ov), R(f), 0);
    barrier(enc);
    npools[l] = np;
  }

  // C_part = A (M, K) @ B ((N, K) when nt, else (K, N)) in MLX's GEMM order (glm_mma); returns the partitions.
  std::size_t mma(MTL::ComputeCommandEncoder* enc, BufRef A, DType ta, BufRef B, DType tb, std::size_t M, std::size_t N,
                  std::size_t K, std::size_t lda, std::size_t ldb, bool nt, bool f32_out, float ascale) {
    const GemmPlan g = gemm_plan(ctx->architecture().empty() ? 'g' : ctx->architecture().back(), M, N, K, nt, f32_out);
    const int ntf = nt ? 1 : 0, sc = ascale != 0.0f ? 1 : 0;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5070, {int64_t(ta), int64_t(tb), ntf, sc}}, [&]() {
      return std::pair<std::string, std::string>{"glm_mma", join({msl_of(ta), msl_of(tb), I(ntf), I(sc)})};
    }));
    set(enc, A, 0);
    set(enc, B, 1);
    set(enc, R(cpart), 2);
    const uint32_t m32 = uint32_t(M), n32 = uint32_t(N), k32 = uint32_t(K), la = uint32_t(lda), lb = uint32_t(ldb),
                   ps = uint32_t(g.psize), bk = uint32_t(g.tail_bk);
    bytes(enc, m32, 3);
    bytes(enc, n32, 4);
    bytes(enc, k32, 5);
    bytes(enc, la, 6);
    bytes(enc, lb, 7);
    bytes(enc, ps, 8);
    bytes(enc, bk, 9);
    bytes(enc, ascale, 10);
    if (g.parts * M * N * 4 > cpart->length()) fail("GEMM partials exceed the scratch buffer");
    enc->dispatchThreadgroups(MTL::Size((N + 7) / 8, (M + 7) / 8, g.parts), MTL::Size(32, 1, 1));
    return g.parts;
  }
  void mma_out(MTL::ComputeCommandEncoder* enc, BufRef out, std::size_t M, std::size_t N, std::size_t parts, bool mask,
               uint32_t nsel, uint32_t ncomp) {
    const int mk = mask ? 1 : 0;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5071, {mk, int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"glm_mma_out", join({T(), I(mk)})};
    }));
    set(enc, R(cpart), 0);
    set(enc, out, 1);
    set(enc, R(sel), 2);
    const uint32_t m32 = uint32_t(M), n32 = uint32_t(N), p32 = uint32_t(parts), kp = uint32_t(cfg.kpool);
    bytes(enc, m32, 3);
    bytes(enc, n32, 4);
    bytes(enc, p32, 5);
    bytes(enc, nsel, 6);
    bytes(enc, kp, 7);
    bytes(enc, ncomp, 8);
    enc->dispatchThreads(MTL::Size(N, M, 1), MTL::Size(std::min<std::size_t>(256, N), 1, 1));
  }

  // attention of the H absorbed queries over the N cached latents (selection when use_sel)
  void attention(MTL::ComputeCommandEncoder* enc, const Layer& L, std::size_t N, std::size_t ncomp, bool use_sel) {
    const std::size_t D = cfg.kv_lora, H = cfg.heads;
    const uint32_t nn = uint32_t(N), nsel = use_sel ? 1u : 0u, kp = uint32_t(cfg.kpool), nc = uint32_t(ncomp);
    const float scale = float(std::pow(double(cfg.nope), -0.5));
    if (!cfg.select_tail && use_sel) fail("index_kpool_always_select_tail = false is not supported");
    if (vector_attention()) {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5040, {int64_t(D), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_sdpa_vector", join({T(), I(long(D))})};
      }));
      set(enc, R(qabs), 0);
      set(enc, R(L.latent), 1);
      set(enc, R(sel), 2);
      set(enc, R(att), 3);
      bytes(enc, nn, 4);
      bytes(enc, nsel, 5);
      bytes(enc, kp, 6);
      bytes(enc, nc, 7);
      bytes(enc, scale, 8);
      enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(1024, 1, 1));
      return;
    }
    // MLX's unfused fallback (fast.cpp): the H query heads share the one latent KV head, so MLX folds them into one
    // GEMM per matmul (scores = T(T(scale) q') Kᵀ, out = P V; glm_mma), except a single key (N == 1: gemv) or a single
    // head (M == 1: gemv_t); softmax(precise) between them.
    const bool f32 = opt.act == Act::F32;
    const DType ta = dtype_of(opt.act);
    if (N == 1 || H == 1) {
      const int sn = gemv_sn(D), bn = N == 1 ? gemv_bn(D, H) : gemv_bn(D, N), sm = 32 / sn;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5041, {int64_t(D), sn, bn, int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_fb_scores", join({T(), I(long(D)), I(sn), "4", I(bn)})};
      }));
      set(enc, R(qabs), 0);
      set(enc, R(L.latent), 1);
      set(enc, R(sel), 2);
      set(enc, R(scores), 3);
      bytes(enc, nn, 4);
      bytes(enc, nsel, 5);
      bytes(enc, kp, 6);
      bytes(enc, nc, 7);
      bytes(enc, scale, 8);
      enc->dispatchThreadgroups(MTL::Size((N + sm - 1) / sm, H, 1), MTL::Size(32 * bn, 1, 1));
    } else {
      const std::size_t parts = mma(enc, R(qabs), ta, R(L.latent), ta, H, N, D, D, D, true, f32, scale);
      barrier(enc);
      mma_out(enc, R(scores), H, N, parts, true, nsel, nc);
    }
    barrier(enc);
    const int a32 = int(N);
    if (N <= 4096) {
      const std::size_t tpg = std::min<std::size_t>(1024, ((N + 3) / 4 + 31) / 32 * 32);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5042, {int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_softmax_row", T()};
      }));
      set(enc, R(scores), 0);
      set(enc, R(probs), 1);
      bytes(enc, a32, 2);
      enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(tpg, 1, 1));
    } else {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5043, {int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_softmax_looped", T()};
      }));
      set(enc, R(scores), 0);
      set(enc, R(probs), 1);
      bytes(enc, a32, 2);
      enc->dispatchThreadgroups(MTL::Size(H, 1, 1), MTL::Size(1024, 1, 1));
    }
    barrier(enc);
    if (H > 1) {
      const std::size_t parts = mma(enc, R(probs), ta, R(L.latent), ta, H, D, N, N, D, false, f32, 0.0f);
      barrier(enc);
      mma_out(enc, R(att), H, D, parts, false, 0, 0);
      return;
    }
    int tsm = 8, tsn = 4;
    if (N >= 8192 && D >= 2048) tsm = 4, tsn = 8;
    const int tbn = D >= 2048 ? 16 : D >= 512 ? 4 : 2;
    const int ttn = D < 4 ? 1 : 4;
    const std::size_t blockN = std::size_t(tbn * tsn * ttn);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5044, {int64_t(D), tsm, tsn, tbn, ttn, int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"glm_gemv_t", join({T(), T(), I(long(D)), I(tsm), I(tsn), I(tbn), "4", I(ttn)})};
    }));
    set(enc, R(probs), 0);
    set(enc, R(L.latent), 1);
    set(enc, R(att), 2);
    const uint32_t kk = uint32_t(N), xhs = uint32_t(N);
    const uint64_t whs = 0;
    bytes(enc, kk, 3);
    bytes(enc, xhs, 4);
    bytes(enc, whs, 5);
    enc->dispatchThreadgroups(MTL::Size((D + blockN - 1) / blockN, H, 1), MTL::Size(32 * tbn, 1, 1));
  }

  void swiglu(MTL::ComputeCommandEncoder* enc, BufRef g, BufRef u, BufRef h, std::size_t F, std::size_t rows) {
    const int lim = cfg.swiglu_limit > 0 ? 1 : 0;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5050, {lim, int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"ds4_swiglu", join({T(), "0", I(lim)})};
    }));
    set(enc, g, 0);
    set(enc, u, 1);
    set(enc, ctx->dummy(), 2);
    set(enc, h, 3);
    const uint32_t f32 = uint32_t(F);
    const float limit = float(cfg.swiglu_limit);
    bytes(enc, f32, 4);
    bytes(enc, limit, 5);
    enc->dispatchThreads(MTL::Size(F, rows, 1), MTL::Size(std::min<std::size_t>(256, F), 1, 1));
  }

  // dense MLP (layers < first_k_dense_replace) or the shared expert: xn -> out
  void mlp(MTL::ComputeCommandEncoder* enc, const Layer& L, BufRef out, std::size_t F) {
    proj(enc, L.mg, R(xn), R(mgb), 1);
    proj(enc, L.mu, R(xn), R(mub), 2);
    barrier(enc);
    swiglu(enc, R(mgb), R(mub), R(mhb), F, 1);
    barrier(enc);
    proj(enc, L.md, R(mhb), out, 3);
  }

  void route(MTL::ComputeCommandEncoder* enc, const Layer& L, BufRef g, BufRef e, BufRef w) {
    const std::size_t E = cfg.experts, K = cfg.top_k;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5051, {int64_t(E), int64_t(K), int64_t(L.bias.dtype)}}, [&]() {
      return std::pair<std::string, std::string>{"glm_route", join({I(long(E)), I(long(K)), L.bias.msl()})};
    }));
    set(enc, g, 0);
    set(enc, L.bias.ref, 1);
    set(enc, e, 2);
    set(enc, w, 3);
    const float rsc = float(cfg.route_scale);
    bytes(enc, rsc, 4);
    const std::size_t tpg = (E + 31) / 32 * 32;
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(tpg, 1, 1));
  }

  // MoE, first half on xn (D): the K repeated input rows (xk_), the shared expert (sh_) and the router (eidx_, ew_).
  void moe_pre(MTL::ComputeCommandEncoder* enc, const Layer& L, BufRef xk_, BufRef sh_, BufRef eidx_, BufRef ew_) {
    const std::size_t D = cfg.hidden, K = cfg.top_k, F = cfg.moe_inter;
    gemv_xf(enc, L.router, R(xn), R(gates));   // router logits straight from xn, concurrent with the shared expert
    {
      const uint32_t cols = uint32_t(D), rep = uint32_t(K);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5052, {int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"suc_repeat_rows", T()};
      }));
      set(enc, R(xn), 0);
      set(enc, xk_, 1);
      bytes(enc, cols, 2);
      bytes(enc, rep, 3);
      enc->dispatchThreads(MTL::Size(D, K, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    }
    // shared expert (= mlp()), the routing alongside its SwiGLU; its down projection overlaps the routed experts'
    // first stage (moe_post starts without a barrier)
    proj(enc, L.mg, R(xn), R(mgb), 1);
    proj(enc, L.mu, R(xn), R(mub), 2);
    barrier(enc);
    swiglu(enc, R(mgb), R(mub), R(mhb), F, 1);
    route(enc, L, R(gates), eidx_, ew_);
    barrier(enc);
    proj(enc, L.md, R(mhb), sh_, 3);
  }

  // MoE, second half: the routed experts of eidx_ (resident / fetched) weighted by ew_, plus sh_ -> mo (D).
  void moe_post(MTL::ComputeCommandEncoder* enc, const Layer& L, BufRef xk_, BufRef sh_, BufRef eidx_, BufRef ew_) {
    const std::size_t D = cfg.hidden, K = cfg.top_k, F = cfg.moe_inter;
    ExpertStack::encode_group(*ctx, enc, {{L.eu.get(), xk_, R(uk), &es_u}, {L.eg.get(), xk_, R(gk), &es_g}}, eidx_, K, opt.act);
    swiglu(enc, R(gk), R(uk), R(hk), F, K);
    barrier(enc);
    ExpertStack::encode_group(*ctx, enc, {{L.ed.get(), R(hk), R(yk), &es_d}}, eidx_, K, opt.act);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5053, {int64_t(K), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"glm_moe_out", join({T(), I(long(K))})};
    }));
    set(enc, R(yk), 0);
    set(enc, ew_, 1);
    set(enc, sh_, 2);
    set(enc, R(mo), 3);
    const uint32_t d32 = uint32_t(D);
    bytes(enc, d32, 4);
    enc->dispatchThreads(MTL::Size(D, 1, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    barrier(enc);
  }

  // MoE on xn (D) -> mo (D). Streamed experts: enc (and cur_cb) are replaced by the host sync.
  void moe(MTL::ComputeCommandEncoder*& enc, const Layer& L, std::size_t l) {
    const std::size_t K = cfg.top_k;
    moe_pre(enc, L, R(xk), R(sh), R(eidx), R(ew));
    if (opt.expert_stream) {   // host sync: fetch the routed experts into the slot pool (and prefetch a guess)
      ExpertStream& xs = *opt.expert_stream;
      const Layer* nxt = nullptr;
      if (xs.options().prefetch)
        for (std::size_t j = l + 1; j < layers.size() && !nxt; ++j)
          if (layers[j].sparse) nxt = &layers[j];
      if (nxt) {   // pre-gating guess: the next MoE layer's router on this layer's MoE input
        gemv_xf(enc, nxt->router, R(xn), R(pgates));
        barrier(enc);
        route(enc, *nxt, R(pgates), R(peidx), R(pew));
      }
      split_command_buffer(*ctx, cur_cb, enc, &split_gpu_s);
      xs.fetch(L.stream_layer, static_cast<const uint32_t*>(eidx->contents()), K);
      if (nxt) xs.prefetch(nxt->stream_layer, static_cast<const uint32_t*>(peidx->contents()), K);
    }
    moe_post(enc, L, R(xk), R(sh), R(eidx), R(ew));
  }

  // embedding of the token in slot `slot` -> hc identical streams at `out`
  void encode_embed(MTL::ComputeCommandEncoder* enc, std::size_t slot, BufRef out) {
    const std::size_t D = cfg.hidden;
    {
      const Mat& m = embed;
      if (m.kind == Mat::Affine) {
        enc->setComputePipelineState(ctx->pipeline(PsoKey{5060, {int64_t(D), m.bits, int64_t(m.group), int64_t(m.sdtype), int64_t(opt.act)}}, [&]() {
          return std::pair<std::string, std::string>{
              "suc_embed_affine", join({T(), I(m.bits), I(long(m.group)), I(long(m.in)), I(long(m.words)), msl_of(m.sdtype)})};
        }));
        set(enc, R(tok, slot * cfg.hc * 4), 0);
        set(enc, m.packed, 1);
        set(enc, m.scales, 2);
        set(enc, m.biases, 3);
        set(enc, out, 4);
      } else {
        enc->setComputePipelineState(ctx->pipeline(PsoKey{5061, {int64_t(D), int64_t(m.wdtype), int64_t(opt.act)}}, [&]() {
          return std::pair<std::string, std::string>{"suc_embed_dense", join({T(), I(long(m.in)), msl_of(m.wdtype)})};
        }));
        set(enc, R(tok, slot * cfg.hc * 4), 0);
        set(enc, m.w, 1);
        set(enc, out, 4);
      }
      enc->dispatchThreads(MTL::Size(D, cfg.hc, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
      barrier(enc);
    }
  }

  // logits (and argmax) of the final streams x
  void encode_logits(MTL::ComputeCommandEncoder* enc, BufRef x) {
    const std::size_t D = cfg.hidden;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5062, {int64_t(D), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"glm_stream_mean", join({T(), I(long(D)), I(long(cfg.hc))})};
    }));
    set(enc, x, 0);
    set(enc, R(y), 1);
    const float inv = float(1.0 / double(cfg.hc));
    bytes(enc, inv, 2);
    enc->dispatchThreads(MTL::Size(D, 1, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    barrier(enc);
    rms(enc, R(y), norm.ref, R(xn), 1, D);
    barrier(enc);
    proj(enc, head, R(xn), R(logits), 0);
    barrier(enc);
    enc->setComputePipelineState(ctx->pipeline("suc_argmax", join({T(), "1024"})));
    set(enc, R(logits), 0);
    set(enc, R(amax), 1);
    const uint32_t n = uint32_t(cfg.vocab);
    bytes(enc, n, 2);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
  }

  // layer l up to the MoE input / dense-MLP output of one token at position p: streams x -> xa_ (after the mixer),
  // coef_ (FFN mHC coefficients), xn (normed FFN input). Dense layers finish here (xo); MoE layers leave the MoE.
  void encode_layer_head(MTL::ComputeCommandEncoder* enc, std::size_t l, std::size_t p, BufRef x, BufRef xa_,
                         BufRef coef_) {
    Layer& L = layers[l];
    hc_pre(enc, L.ha, x, R(coef_a), R(y), L.in_norm, R(xn));
    barrier(enc);
    if (L.dsa) mla(enc, L, l, p);
    else kda(enc, L);
    hc_post(enc, R(f), x, R(coef_a), xa_);
    hc_pre(enc, L.hf, xa_, coef_, R(y), L.post_norm, R(xn));
    barrier(enc);
  }

  // One decode step for the token in slot `slot` at position p.
  void encode_step(MTL::ComputeCommandEncoder*& enc, std::size_t slot, std::size_t p, bool with_logits) {
    const std::size_t W = cfg.stream(), a = as();
    encode_embed(enc, slot, R(hs));
    for (std::size_t l = 0; l < cfg.layers; ++l) {
      Layer& L = layers[l];
      const BufRef x = R(hs, l * W * a), xo = R(hs, (l + 1) * W * a);
      encode_layer_head(enc, l, p, x, R(xa), R(coef_f));
      if (L.sparse) {
        moe(enc, L, l);
      } else {
        mlp(enc, L, R(mo), cfg.intermediate);
        barrier(enc);
      }
      hc_post(enc, R(mo), R(xa), R(coef_f), xo);
    }
    if (with_logits) encode_logits(enc, R(hs, cfg.layers * W * a));
  }

  // Layer-major prefill of the n tokens in slots [0, n) at positions p0, p0 + 1, ... (streamed experts, n > 1).
  // Every layer runs, token by token in position order, exactly the kernels of a decode step up to the router
  // (attention / KDA, mHC, shared expert, router; per-token MoE inputs kept in the chunk buffers); then ONE host sync
  // and one fetch of the union of the chunk's routed experts (in consecutive token groups whose union fits the
  // layer's unpinned slots), and the routed experts run token by token. Same bits as n decode steps (only buffer
  // addresses differ), but one host sync and one expert read per layer per chunk instead of per token. The streams of
  // the last token use hs (hidden_states()); the others alternate between two halves of `ring`.
  void encode_chunk_layer_major(MTL::ComputeCommandEncoder*& enc, std::size_t n, std::size_t p0, bool with_logits) {
    ExpertStream& xs = *opt.expert_stream;
    const std::size_t W = cfg.stream(), a = as(), K = cfg.top_k, D = cfg.hidden, cw = (2 + cfg.hc) * cfg.hc * 4;
    auto xin = [&](std::size_t l, std::size_t i) -> BufRef {
      return i + 1 == n ? R(hs, l * W * a) : R(ring, ((l % 2) * opt.chunk + i) * W * a);
    };
    auto xk_i = [&](std::size_t i) { return R(xk_c, i * K * D * a); };
    auto sh_i = [&](std::size_t i) { return R(sh_c, i * D * a); };
    auto ei_i = [&](std::size_t i) { return R(eidx_c, i * K * 4); };
    auto ew_i = [&](std::size_t i) { return R(ew_c, i * K * 4); };
    for (std::size_t i = 0; i < n; ++i) {
      encode_embed(enc, i, xin(0, i));
      barrier(enc);
    }
    for (std::size_t l = 0; l < cfg.layers; ++l) {
      const Layer& L = layers[l];
      for (std::size_t i = 0; i < n; ++i) {
        const BufRef xa_ = R(xa_c, i * W * a), cf = R(coef_c, i * cw);
        encode_layer_head(enc, l, p0 + i, xin(l, i), xa_, cf);
        if (L.sparse) {
          moe_pre(enc, L, xk_i(i), sh_i(i), ei_i(i), ew_i(i));
        } else {
          mlp(enc, L, R(mo), cfg.intermediate);
          barrier(enc);
          hc_post(enc, R(mo), xa_, cf, xin(l + 1, i));
        }
        barrier(enc);
      }
      if (!L.sparse) continue;
      split_command_buffer(*ctx, cur_cb, enc, &split_gpu_s);   // the chunk's routers -> host
      const auto* ids = static_cast<const uint32_t*>(eidx_c->contents());
      const std::size_t cap = std::max<std::size_t>(xs.max_distinct(L.stream_layer), K);
      // Token groups whose distinct experts fit the slots, built greedily: next the token adding the fewest experts
      // not yet in the group (ties: the most experts of the previous group, still resident). The routed experts of
      // a token do not depend on the other tokens, so any grouping / order gives the same bits.
      std::vector<char> done(n, 0), in_uni(cfg.experts, 0), in_prev(cfg.experts, 0);
      std::vector<std::size_t> group;
      std::vector<uint32_t> gids;
      for (std::size_t left = n, g = 0; left; ++g) {
        std::fill(in_uni.begin(), in_uni.end(), 0);
        std::size_t usz = 0;
        group.clear();
        for (;;) {
          std::size_t best = n, bnew = K + 1, bprev = 0;
          for (std::size_t i = 0; i < n; ++i) {
            if (done[i]) continue;
            std::size_t nw = 0, pv = 0;
            for (std::size_t k = 0; k < K; ++k) {
              const uint32_t e = ids[i * K + k];
              if (!in_uni[e]) ++nw, pv += std::size_t(in_prev[e]);
            }
            if (nw < bnew || (nw == bnew && pv > bprev)) best = i, bnew = nw, bprev = pv;
          }
          if (best == n || (!group.empty() && usz + bnew > cap)) break;
          done[best] = 1;
          group.push_back(best);
          for (std::size_t k = 0; k < K; ++k) {
            const uint32_t e = ids[best * K + k];
            if (!in_uni[e]) in_uni[e] = 1, ++usz;
          }
        }
        left -= group.size();
        std::sort(group.begin(), group.end());
        if (g > 0) split_command_buffer(*ctx, cur_cb, enc, &split_gpu_s);   // the previous group used the slots
        gids.clear();
        for (std::size_t i : group) gids.insert(gids.end(), ids + i * K, ids + (i + 1) * K);
        xs.fetch(L.stream_layer, gids.data(), gids.size());
        ++chunk_fetches;
        for (std::size_t i : group) {
          moe_post(enc, L, xk_i(i), sh_i(i), ei_i(i), ew_i(i));
          hc_post(enc, R(mo), R(xa_c, i * W * a), R(coef_c, i * cw), xin(l + 1, i));
          barrier(enc);
        }
        in_prev.swap(in_uni);
      }
    }
    if (with_logits) encode_logits(enc, xin(cfg.layers, n - 1));
  }

  // ---------------------------------------------------------------------------- batched prefill (not row-exact)
  // encode_chunk_fast: the n chunk rows' streams in ring half l % 2 ((n, W) rows), per-row scratch below; the last
  // row's streams are copied to hs (hidden_states()).
  BufPtr coefa_c, xn_c, ss_c, mix_c, f_c, mo_c, shg_c, gates_c, yk_c, scr, iscore_c, sel_c;
  BufPtr sidx_c, scnt_c, nep_c;   // per-row selected pool lists / counts / visible pools (fast chunk attention)
  std::size_t scr_bytes = 0;
  BufPtr t_x, t_g, t_u, t_teidx, t_reid, t_src, t_dst;   // expert tiles (t_y = t_x, t_h = t_g)
  std::size_t t_rows_cap = 0, t_tiles_cap = 0;
  std::vector<BufPtr> wkt;           // per DSA layer: W_k,h transposed (H, rank, nope) in T, built on first use
  BufPtr logits_rows;                // forward_rows(): (chunk, vocab) in T
  std::vector<float>* rows_out = nullptr;
  bool fast_ok = false;              // shapes supported by the batched chunks (set at load)
  bool fast() const { return fast_ok && opt.layer_prefill && !opt.exact_prefill && !prefill_exact_env(); }
  // $SUC_FAST_OFF (diagnostics): letters of batched parts to run per row instead: p projections (row gemvs / one-row
  // SUC launches), a MLA chunk attention (per-token decode kernels), t expert tiles (row-exact tiles), r router
  static bool fast_off(char c) {
    static const std::string v = [] { const char* e = std::getenv("SUC_FAST_OFF"); return std::string(e ? e : ""); }();
    return v.find(c) != std::string::npos;
  }
  // $SUC_GLM_PROFILE=1: every stage boundary of a batched chunk commits the command buffer and waits (same kernels and
  // bits, no overlap across stages), GPU seconds accumulated per stage label.
  bool prof = false;
  const char* prof_label = nullptr;
  std::map<std::string, double> prof_acc;
  void mark(MTL::ComputeCommandEncoder*& enc, const char* next) {
    if (!prof) return;
    double g = 0.0;
    split_command_buffer(*ctx, cur_cb, enc, &g);
    if (prof_label) prof_acc[prof_label] += g;
    split_gpu_s += g;
    prof_label = next;
  }

  // y rows r = x rows r · Wᵀ (r < n; row strides xs / ys elements): GEMMs over the rows (SUC: Linear::encode with
  // Path::Prefill on contiguous rows), else one-row launches.
  void proj_rows(MTL::ComputeCommandEncoder* enc, const Mat& m, BufRef x, BufRef y, std::size_t n, std::size_t xs,
                 std::size_t ys, int wsi) {
    const std::size_t a = as();
    if (m.kind == Mat::Suc && xs == m.in && ys == m.out) {
      if (fast_off('p')) m.lin->encode_rows_exact(*ctx, enc, x, y, n, opt.act, ws[wsi]);
      else m.lin->encode(*ctx, enc, x, y, n, opt.act, ws[wsi], Path::Prefill);
      return;
    }
    if (n >= kMmMinRows && !fast_off('p')) {
      if (m.kind == Mat::Affine && encode_mm_affine(*ctx, enc, x, m.packed, m.scales, m.biases, y, n, xs, ys, m.bits,
                                                    m.group, m.in, m.out, m.words, msl_of(m.sdtype), opt.act))
        return;
      if (m.kind == Mat::Dense && encode_mm_dense(*ctx, enc, x, m.w, y, n, xs, ys, m.in, m.out, msl_of(m.wdtype), opt.act))
        return;
    }
    for (std::size_t r = 0; r < n; ++r) {   // SUC rows share the workspace: barriers between them
      proj(enc, m, x.at(r * xs * a), y.at(r * ys * a), wsi);
      if (m.kind == Mat::Suc && r + 1 < n) barrier(enc);
    }
  }

  // mHC pre of the n stream rows x (n, W): coef (n, mix) float32, xnout (n, D) = rms_norm(collapse, nw).
  void hc_pre_rows(MTL::ComputeCommandEncoder* enc, const Hc& h, BufRef x, BufRef coef, const Vec& nw, BufRef xnout,
                   std::size_t n) {
    const std::size_t W = cfg.stream(), D = cfg.hidden, HC = cfg.hc, mix = (2 + HC) * HC, tpg = red_tpg(W);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5100, {int64_t(W), int64_t(tpg), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"glm_hc_ss_rows", join({T(), I(long(W)), I(long(tpg))})};
    }));
    set(enc, x, 0);
    set(enc, R(ss_c), 1);
    enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(tpg, 1, 1));
    barrier(enc);
    const float inv_w = float(1.0 / double(W));
    {
      const int sn = gemv_sn(W), bn = gemv_bn(W, mix), sm = 32 / sn;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5101, {int64_t(W), int64_t(mix), int64_t(h.fn.dtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{
            "glm_hc_mix_rows", join({T(), h.fn.msl(), I(long(W)), I(long(mix)), I(sn), "4", I(bn)})};
      }));
      set(enc, x, 0);
      set(enc, h.fn.ref, 1);
      set(enc, R(mix_c), 2);
      set(enc, R(ss_c), 3);
      bytes(enc, inv_w, 4);
      bytes(enc, eps, 5);
      enc->dispatchThreadgroups(MTL::Size((mix + sm - 1) / sm, n, 1), MTL::Size(32 * bn, 1, 1));
    }
    barrier(enc);
    const int iters = int(cfg.sinkhorn_iters);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5102, {int64_t(HC), iters}}, [&]() {
      return std::pair<std::string, std::string>{"glm_hc_coef_rows", join({"float", I(long(HC)), I(iters)})};
    }));
    set(enc, R(mix_c), 0);
    set(enc, h.base.ref, 1);
    set(enc, h.scale.ref, 2);
    set(enc, coef, 3);
    bytes(enc, hc_eps, 4);
    enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(32, 1, 1));
    const std::size_t tpg2 = rms_tpg(D);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5103, {int64_t(D), int64_t(HC), int64_t(opt.act), int64_t(tpg2)}}, [&]() {
      return std::pair<std::string, std::string>{
          "glm_hc_collapse_rms_rows", join({T(), "float", I(long(D)), I(long(HC)), I(long(tpg2))})};
    }));
    set(enc, x, 0);
    set(enc, R(mix_c), 1);
    set(enc, h.base.ref, 3);
    set(enc, h.scale.ref, 4);
    set(enc, nw.ref, 5);
    set(enc, xnout, 6);
    bytes(enc, hc_eps, 9);
    bytes(enc, eps, 10);
    enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(tpg2, 1, 1));
    barrier(enc);
  }

  // y (n, W) = mHC post of f (n, D) on the residual streams res (n, W) with coef (n, mix)
  void hc_post_rows(MTL::ComputeCommandEncoder* enc, BufRef fbuf, BufRef res, BufRef coef, BufRef yout, std::size_t n) {
    const std::size_t D = cfg.hidden, HC = cfg.hc;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5104, {int64_t(D), int64_t(HC), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"glm_hc_post_rows", join({T(), I(long(D)), I(long(HC))})};
    }));
    set(enc, fbuf, 0);
    set(enc, res, 1);
    set(enc, coef, 2);
    set(enc, yout, 3);
    enc->dispatchThreads(MTL::Size(D, HC, n), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    barrier(enc);
  }

  void copy_f32(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t n) {
    if (!n) return;
    enc->setComputePipelineState(ctx->pipeline("glm_copy_f32", "float"));
    set(enc, x, 0);
    set(enc, y, 1);
    const uint32_t n32 = uint32_t(n);
    bytes(enc, n32, 2);
    enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(std::min<std::size_t>(256, n), 1, 1));
  }
  void cast_rows(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t rows, std::size_t cols,
                 std::size_t xs, std::size_t ys) {
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5105, {int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"glm_cast_rows", T()};
    }));
    set(enc, x, 0);
    set(enc, y, 1);
    const uint32_t c32 = uint32_t(cols), x32 = uint32_t(xs), y32 = uint32_t(ys);
    bytes(enc, c32, 2);
    bytes(enc, x32, 3);
    bytes(enc, y32, 4);
    enc->dispatchThreads(MTL::Size(cols, rows, 1), MTL::Size(std::min<std::size_t>(256, cols), 1, 1));
  }

  // bump allocation in the chunk scratch `scr` (offsets aligned to 256 bytes); reset per sub-layer
  std::size_t scr_off = 0;
  BufRef take(std::size_t bytes) {
    const std::size_t o = scr_off;
    scr_off += (bytes + 255) / 256 * 256;
    if (scr_off > scr_bytes) fail("chunk scratch overflow");
    return R(scr, o);
  }

  // KDA of the n chunk rows: xn_c -> f_c. Projections as GEMMs, conv / l2norm / gates / recurrence / o_norm row-batched.
  void kda_chunk(MTL::ComputeCommandEncoder*& enc, Layer& L, std::size_t n) {
    const std::size_t H = cfg.lin_heads, Dk = cfg.lin_dim, d = H * Dk, a = as(), K = cfg.conv_kernel, D = cfg.hidden;
    scr_off = 0;
    const BufRef qkv_ = take(3 * n * d * a), conv_ = take(3 * n * d * a), qkf_ = take(2 * n * d * 4),
                 vf_ = take(n * d * 4), fa_ = take(n * Dk * a), ga_ = take(n * Dk * a), bb_ = take(n * H * a),
                 fb_ = take(n * d * a), gb_ = take(n * d * a), decay_ = take(n * d * 4), beta_ = take(n * H * 4);
    const BufRef o32_ = qkv_, on32_ = qkf_, oy_ = conv_;   // reused once their contents are consumed
    proj_rows(enc, L.q, R(xn_c), qkv_, n, D, d, 0);
    proj_rows(enc, L.k, R(xn_c), qkv_.at(n * d * a), n, D, d, 1);
    proj_rows(enc, L.v, R(xn_c), qkv_.at(2 * n * d * a), n, D, d, 2);
    proj_rows(enc, L.fa, R(xn_c), fa_, n, D, Dk, 3);
    proj_rows(enc, L.ga, R(xn_c), ga_, n, D, Dk, 4);
    proj_rows(enc, L.bproj, R(xn_c), bb_, n, D, H, 5);
    barrier(enc);
    mark(enc, "kda_conv_fb_gb");
    {
      const std::size_t C = 3 * d;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5110, {int64_t(C), int64_t(K), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_kda_conv_rows", join({T(), I(long(C)), I(long(K))})};
      }));
      set(enc, qkv_, 0);
      set(enc, R(L.conv_state), 1);
      set(enc, R(L.convw), 2);
      set(enc, conv_, 3);
      const uint32_t n32 = uint32_t(n);
      bytes(enc, n32, 4);
      enc->dispatchThreads(MTL::Size(C, 1, 1), MTL::Size(std::min<std::size_t>(256, C), 1, 1));
    }
    proj_rows(enc, L.fb, fa_, fb_, n, Dk, d, 3);
    proj_rows(enc, L.gb, ga_, gb_, n, Dk, d, 4);
    barrier(enc);
    mark(enc, "kda_norm_gate");
    {
      const std::size_t tpg = Dk <= 64 ? 32 : red_tpg(Dk);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5111, {int64_t(Dk), int64_t(tpg), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_l2norm_rows", join({T(), I(long(Dk)), I(long(tpg))})};
      }));
      set(enc, conv_, 0);
      set(enc, qkf_, 1);
      const uint32_t qrows = uint32_t(H);
      const float qscale = float(std::pow(double(Dk), -0.5));
      bytes(enc, qrows, 2);
      bytes(enc, qscale, 3);
      enc->dispatchThreadgroups(MTL::Size(2 * H, n, 1), MTL::Size(tpg, 1, 1));
    }
    cast_rows(enc, conv_.at(2 * d * a), vf_, n, d, 3 * d, d);
    {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5112, {int64_t(L.a_log.dtype), int64_t(L.dt_bias.dtype), int64_t(Dk), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_kda_gate_rows", join({T(), L.a_log.msl(), L.dt_bias.msl(), I(long(Dk))})};
      }));
      set(enc, fb_, 0);
      set(enc, L.dt_bias.ref, 1);
      set(enc, L.a_log.ref, 2);
      set(enc, bb_, 3);
      set(enc, decay_, 4);
      set(enc, beta_, 5);
      const uint32_t nd = uint32_t(d), heads = uint32_t(H);
      const float lb = float(cfg.lower_bound);
      bytes(enc, nd, 6);
      bytes(enc, heads, 7);
      bytes(enc, lb, 8);
      enc->dispatchThreads(MTL::Size(d, n, 1), MTL::Size(std::min<std::size_t>(256, d), 1, 1));
    }
    barrier(enc);
    mark(enc, "kda_recurrence");
    {
      static const int dvp = [] {   // $SUC_GLM_DELTA_DVP: state rows per simdgroup (1, 2, 4)
        const char* e = std::getenv("SUC_GLM_DELTA_DVP");
        const int x = e ? std::atoi(e) : 0;
        return x == 1 || x == 2 || x == 4 ? x : 2;
      }();
      const int dv = Dk % (4 * dvp) == 0 ? dvp : 1;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5113, {int64_t(Dk), int64_t(H), int64_t(opt.act), dv}}, [&]() {
        return std::pair<std::string, std::string>{
            "glm_gated_delta_rows", join({T(), I(long(Dk)), I(long(Dk)), I(long(H)), I(long(H)), I(dv)})};
      }));
      set(enc, qkf_, 0);
      set(enc, vf_, 2);
      set(enc, decay_, 3);
      set(enc, beta_, 4);
      set(enc, R(L.rec_state), 5);
      set(enc, o32_, 6);
      const uint32_t n32 = uint32_t(n);
      bytes(enc, n32, 7);
      enc->dispatchThreads(MTL::Size(32, Dk / dv, H), MTL::Size(32, 4, 1));
    }
    barrier(enc);
    mark(enc, "kda_onorm_o");
    rms(enc, o32_, R(L.onorm), on32_, n * H, Dk, "float");
    barrier(enc);
    {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5024, {int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_sig_mul", T()};
      }));
      set(enc, on32_, 0);
      set(enc, gb_, 1);
      set(enc, oy_, 2);
      const uint32_t nd = uint32_t(n * d);
      bytes(enc, nd, 3);
      enc->dispatchThreads(MTL::Size(n * d, 1, 1), MTL::Size(256, 1, 1));
    }
    barrier(enc);
    proj_rows(enc, L.o, oy_, R(f_c), n, d, D, 0);
    barrier(enc);
  }

  // W_k,h transposed per head ((H, rank, nope) in T) for the absorbed queries as GEMMs; built on first use.
  BufRef wk_t(std::size_t l) {
    if (wkt.empty()) wkt.resize(cfg.layers);
    if (!wkt[l]) {
      const Layer& L = layers[l];
      const std::size_t H = cfg.heads, nope = cfg.nope, vd = cfg.v_dim, rank = cfg.kv_lora;
      const std::size_t es = dtype_size(L.kv_b.dtype);
      wkt[l] = ctx->new_buffer(H * rank * nope * es);
      copied_bytes += H * rank * nope * es;
      const auto* src = static_cast<const uint8_t*>(L.kv_b.ref.buffer->contents()) + L.kv_b.ref.offset;
      auto* dst = static_cast<uint8_t*>(wkt[l]->contents());
      for (std::size_t h = 0; h < H; ++h)
        for (std::size_t c = 0; c < nope; ++c) {
          const uint8_t* row = src + ((h * (nope + vd) + c) * rank) * es;
          for (std::size_t r = 0; r < rank; ++r)
            std::memcpy(dst + ((h * rank + r) * nope + c) * es, row + r * es, es);
        }
    }
    return R(wkt[l]);
  }

  // MLA + DSA of the n chunk rows at positions p0 ..: xn_c -> f_c. Projections as GEMMs, latents / indexer keys /
  // k-pools row-batched; rows past index_topk / kpool complete pools: indexer pool scores of all rows in one launch, the
  // top-pool selection per row (q4_index_select); the absorbed queries and the per-head value projections as GEMMs, one
  // glm_attn_chunk launch. SUC_FAST_OFF=a: the decode kernels per token.
  void mla_chunk(MTL::ComputeCommandEncoder*& enc, Layer& L, std::size_t l, std::size_t p0, std::size_t n) {
    const std::size_t H = cfg.heads, nope = cfg.nope, vd = cfg.v_dim, rank = cfg.kv_lora, a = as(), D = cfg.hidden;
    const std::size_t idd = cfg.index_dim, kp = cfg.kpool, ql = cfg.q_lora;
    if (fast_off('a') || nope % 32 || rank % 32) {
      for (std::size_t i = 0; i < n; ++i) {
        encode_copy_cols(*ctx, enc, R(xn_c, i * D * a), R(xn), 1, D, D, D, opt.act);
        barrier(enc);
        mla(enc, L, l, p0 + i);
        encode_copy_cols(*ctx, enc, R(f), R(f_c, i * D * a), 1, D, D, D, opt.act);
        barrier(enc);
      }
      return;
    }
    scr_off = 0;
    const std::size_t r0 = p0 % kp, j0 = npools[l], j1 = (p0 + n) / kp, tail = (p0 + n) % kp;
    const std::size_t ih = cfg.index_heads, topk = cfg.topk_pools();
    const bool any_sel = j1 > topk;   // some row of the chunk has more than topk complete pools
    const BufRef qa_ = take(n * ql * a), qr_ = take(n * ql * a), kvr_ = take(n * rank * a), ikr_ = take(n * idd * a),
                 ik_ = take(n * idd * a), ig_ = take(n * idd * a), q_ = take(n * H * nope * a),
                 rk_ = take((kp + n) * idd * 4), rg_ = take((kp + n) * idd * 4), qabs_ = take(n * H * rank * a),
                 att_ = take(n * H * rank * a), ov_ = take(n * H * vd * a);
    const BufRef iq_ = any_sel ? take(n * ih * idd * a) : R(scr), iw_ = any_sel ? take(n * ih * a) : R(scr);
    proj_rows(enc, L.q_a, R(xn_c), qa_, n, D, ql, 0);
    if (any_sel) proj_rows(enc, L.iwp, R(xn_c), iw_, n, D, ih, 4);
    proj_rows(enc, L.kv_a, R(xn_c), kvr_, n, D, rank, 1);
    proj_rows(enc, L.iwk, R(xn_c), ikr_, n, D, idd, 2);
    proj_rows(enc, L.igate, R(xn_c), ig_, n, D, idd, 3);
    copy_f32(enc, R(L.rawk), rk_, r0 * idd);   // the incomplete pool's earlier tokens
    copy_f32(enc, R(L.rawg), rg_, r0 * idd);
    barrier(enc);
    rms(enc, qa_, L.q_a_norm.ref, qr_, n, ql);
    rms(enc, kvr_, L.kv_a_norm.ref, R(L.latent, p0 * rank * a), n, rank);
    {
      const std::size_t tpg = 32 * (((idd + 3) / 4 + 31) / 32);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5030, {int64_t(idd), int64_t(tpg), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_layer_norm", join({T(), I(long(idd)), I(long(tpg))})};
      }));
      set(enc, ikr_, 0);
      set(enc, L.ik_w.ref, 1);
      set(enc, L.ik_b.ref, 2);
      set(enc, ik_, 3);
      const float le = 1e-6f;
      bytes(enc, le, 4);
      enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(tpg, 1, 1));
    }
    cast_rows(enc, ig_, rg_.at(r0 * idd * 4), n, idd, idd, idd);
    barrier(enc);
    proj_rows(enc, L.q_b, qr_, q_, n, ql, H * nope, 0);
    if (any_sel) proj_rows(enc, L.iwq_b, qr_, iq_, n, ql, ih * idd, 1);
    cast_rows(enc, ik_, rk_.at(r0 * idd * 4), n, idd, idd, idd);
    barrier(enc);
    if (j1 > j0) {   // the k-pools completed in this chunk (decode: ds4_comp_pool at every kp-th token)
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5114, {int64_t(idd), int64_t(kp), int64_t(L.ape.dtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"ds4_comp_pool_rows", join({T(), L.ape.msl(), I(long(idd)), I(long(kp)), "0"})};
      }));
      set(enc, rk_, 0);
      set(enc, rg_, 1);
      set(enc, rk_, 2);
      set(enc, rg_, 3);
      set(enc, L.ape.ref, 4);
      set(enc, R(L.pools, j0 * idd * a), 5);
      const uint32_t has_prev = 0;
      bytes(enc, has_prev, 6);
      enc->dispatchThreadgroups(MTL::Size(idd, j1 - j0, 1), MTL::Size(32, 1, 1));
    }
    copy_f32(enc, rk_.at((j1 - j0) * kp * idd * 4), R(L.rawk), tail * idd);   // the new incomplete pool
    copy_f32(enc, rg_.at((j1 - j0) * kp * idd * 4), R(L.rawg), tail * idd);
    if (any_sel) {   // the indexer's pool selection of the rows past topk complete pools
      barrier(enc);
      mark(enc, "mla_index_select");
      const std::size_t ps = max_pools;
      if (!sidx_c) {
        sidx_c = alloc(opt.chunk * topk * 4, nullptr);
        scnt_c = alloc(opt.chunk * 4, nullptr);
        nep_c = alloc(opt.chunk * 4, nullptr);
      }
      auto* nep = static_cast<uint32_t*>(nep_c->contents());
      for (std::size_t i = 0; i < n; ++i) nep[i] = uint32_t((p0 + i + 1) / kp);   // same for every layer of the chunk
      const float dscale = float(std::pow(double(idd), -0.5)), hscale = float(std::pow(double(ih), -0.5));
      // scores: simdgroup-matrix tiles (ds4_index_scores_mm MODE 1) when the shape allows ($SUC_IDX_MM=0: the
      // per-pool kernel)
      static const bool idx_mm = [] { const char* e = std::getenv("SUC_IDX_MM"); return !(e && e[0] == '0'); }();
      if (idx_mm && ih % 16 == 0 && ih <= 64 && idd % 32 == 0) {
        enc->setComputePipelineState(ctx->pipeline(PsoKey{5117, {int64_t(ih), int64_t(idd), int64_t(opt.act)}}, [&]() {
          return std::pair<std::string, std::string>{"ds4_index_scores_mm", join({T(), I(long(ih)), I(long(idd)), "1"})};
        }));
        set(enc, iq_, 0);
        set(enc, R(L.pools), 1);
        set(enc, iw_, 2);
        set(enc, R(iscore_c), 3);
        bytes(enc, hscale, 4);
        set(enc, R(nep_c), 5);
        const uint32_t ps32 = uint32_t(ps);
        bytes(enc, ps32, 6);
        bytes(enc, dscale, 7);
        enc->dispatchThreadgroups(MTL::Size((j1 + 31) / 32, n, 1), MTL::Size(ih / 16 * 32, 1, 1));
      } else {
        enc->setComputePipelineState(ctx->pipeline(PsoKey{5116, {int64_t(ih), int64_t(idd), int64_t(opt.act)}}, [&]() {
          return std::pair<std::string, std::string>{"glm_index_scores_rows", join({T(), I(long(ih)), I(long(idd))})};
        }));
        set(enc, iq_, 0);
        set(enc, R(L.pools), 1);
        set(enc, iw_, 2);
        set(enc, R(iscore_c), 3);
        const uint32_t ps32 = uint32_t(ps), p32 = uint32_t(p0), kp32 = uint32_t(kp);
        bytes(enc, ps32, 4);
        bytes(enc, p32, 5);
        bytes(enc, kp32, 6);
        bytes(enc, dscale, 7);
        bytes(enc, hscale, 8);
        enc->dispatchThreadgroups(MTL::Size(j1, n, 1), MTL::Size(32, 1, 1));
      }
      barrier(enc);
      // top-k pools of every row in one launch, as ascending pool lists (q4_index_select's rule)
      enc->setComputePipelineState(ctx->pipeline("ds4_index_select_rows", "1024"));
      set(enc, R(iscore_c), 0);
      set(enc, R(sidx_c), 1);
      set(enc, R(scnt_c), 2);
      set(enc, R(nep_c), 3);
      const uint32_t k32 = uint32_t(topk), sps32 = uint32_t(ps);
      bytes(enc, k32, 4);
      bytes(enc, sps32, 5);
      enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(1024, 1, 1));
    }
    // q'_h = q_h W_k,h (GEMMs over the rows with W_k,hᵀ)
    mark(enc, "mla_absorb");
    const BufRef wt = wk_t(l);
    const std::size_t es = dtype_size(L.kv_b.dtype);
    for (std::size_t h = 0; h < H; ++h)
      if (!encode_mm_dense(*ctx, enc, q_.at(h * nope * a), wt.at(h * rank * nope * es), qabs_.at(h * rank * a), n,
                           H * nope, H * rank, nope, rank, L.kv_b.msl(), opt.act))
        fail("absorbed query GEMM");
    barrier(enc);
    mark(enc, "mla_attention");
    {
      const int rnd = vector_attention() ? 0 : 1;
      // RND = 1: glm_attn_chunk_h, 2 heads per simdgroup sharing each latent load (same bits); $SUC_GLM_ATTN_HPS=1:
      // one head per simdgroup (glm_attn_chunk)
      static const int hps_env = [] { const char* e = std::getenv("SUC_GLM_ATTN_HPS"); return e ? std::atoi(e) : 2; }();
      static const int keep_env = [] { const char* e = std::getenv("SUC_GLM_ATTN_KEEP"); return e ? std::atoi(e) : 0; }();
      const int hps = (rnd == 1 && hps_env == 2 && H % 4 == 0) ? 2 : 1;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5115, {int64_t(rank), rnd, int64_t(opt.act), hps, keep_env}}, [&]() {
        return hps > 1 ? std::pair<std::string, std::string>{"glm_attn_chunk_h", join({T(), I(long(rank)), "2", I(keep_env ? 1 : 0)})}
                       : std::pair<std::string, std::string>{"glm_attn_chunk", join({T(), I(long(rank)), I(rnd)})};
      }));
      set(enc, qabs_, 0);
      set(enc, R(L.latent), 1);
      set(enc, att_, 2);
      const uint32_t h32 = uint32_t(H), p32 = uint32_t(p0);
      const float scale = float(std::pow(double(nope), -0.5));
      bytes(enc, h32, 3);
      bytes(enc, p32, 4);
      bytes(enc, scale, 5);
      set(enc, any_sel ? R(sel_c) : R(scr), 6);
      const uint32_t ps32 = uint32_t(max_pools), kp32 = uint32_t(kp), tk32 = uint32_t(topk);
      bytes(enc, ps32, 7);
      bytes(enc, kp32, 8);
      bytes(enc, tk32, 9);
      set(enc, any_sel ? R(sidx_c) : R(scr), 10);
      set(enc, any_sel ? R(scnt_c) : R(scr), 11);
      if (hps > 1) enc->dispatchThreadgroups(MTL::Size((H + 3) / 4, n, 1), MTL::Size(64, 1, 1));
      else enc->dispatchThreadgroups(MTL::Size((H + 3) / 4, n, 1), MTL::Size(128, 1, 1));
    }
    barrier(enc);
    mark(enc, "mla_value_o");
    for (std::size_t h = 0; h < H; ++h)   // o_h = W_v,h (P c)
      if (!encode_mm_dense(*ctx, enc, att_.at(h * rank * a), L.kv_b.ref.at((h * (nope + vd) + nope) * rank * es),
                           ov_.at(h * vd * a), n, H * rank, H * vd, rank, vd, L.kv_b.msl(), opt.act))
        fail("value GEMM");
    barrier(enc);
    proj_rows(enc, L.o, ov_, R(f_c), n, H * vd, D, 0);
    barrier(enc);
    npools[l] = j1;
  }

  // tile rows per expert launch of the batched chunks ($SUC_GLM_EXPERT_PASS_ROWS, default 2048)
  static std::size_t expert_pass_rows() {
    static const std::size_t v = [] {
      const char* e = std::getenv("SUC_GLM_EXPERT_PASS_ROWS");
      const long x = e ? std::atol(e) : 0;
      return std::size_t(x > 0 ? x : 2048);
    }();
    return v;
  }
  void ensure_tiles(std::size_t rows, std::size_t tiles) {
    const std::size_t D = cfg.hidden, F = cfg.moe_inter, a = as();
    if (rows > t_rows_cap) {
      t_x = alloc(rows * D * a, nullptr);   // also the down outputs
      t_g = alloc(rows * F * a, nullptr);   // also the SwiGLU outputs
      t_u = alloc(rows * F * a, nullptr);
      t_reid = alloc(rows * 4, nullptr);
      t_src = alloc(rows * 4, nullptr);
      t_dst = alloc(rows * 4, nullptr);
      t_rows_cap = rows;
    }
    if (tiles > t_tiles_cap) {
      t_teidx = alloc(tiles * 4, nullptr);
      t_tiles_cap = tiles;
    }
  }

  // Routed experts of the chunk rows in `group` (routing in eidx_c): tiles over the (row, k) pairs, gather xn rows,
  // gate / up / clamped SwiGLU / down per tile, scatter to yk_c row i · K + k (the routing weight follows in
  // glm_moe_out, as in moe_post).
  void experts_tiled(MTL::ComputeCommandEncoder* enc, const Layer& L, const std::vector<std::size_t>& group) {
    const std::size_t K = cfg.top_k, D = cfg.hidden, F = cfg.moe_inter, a = as();
    const auto* ids = static_cast<const uint32_t*>(eidx_c->contents());
    std::vector<std::pair<uint32_t, uint32_t>> rows;
    rows.reserve(group.size() * K);
    for (std::size_t i : group)
      for (std::size_t k = 0; k < K; ++k) rows.push_back({uint32_t(i * K + k), ids[i * K + k]});
    const bool ft = !fast_off('t');
    const std::size_t mb = ft ? std::min({L.eg->fast_tile_rows(), L.eu->fast_tile_rows(), L.ed->fast_tile_rows()})
                              : std::min({L.eg->max_tile_rows(*ctx), L.eu->max_tile_rows(*ctx), L.ed->max_tile_rows(*ctx)});
    std::vector<uint32_t> rank(cfg.experts);
    for (std::size_t e = 0; e < cfg.experts; ++e)
      rank[e] = uint32_t((L.eg->sub_slot(e) * 16 + L.eu->sub_slot(e)) * 16 + L.ed->sub_slot(e));
    const TilePlan P = plan_tiles(rows, mb, &rank, -1.0);
    ensure_tiles(P.rows(), P.count());
    auto* src = static_cast<uint32_t*>(t_src->contents());
    for (std::size_t j = 0; j < P.rows(); ++j) src[j] = P.src[j] / uint32_t(K);
    std::memcpy(t_dst->contents(), P.dst.data(), P.rows() * 4);
    std::memcpy(t_reid->contents(), P.eidx.data(), P.rows() * 4);
    std::memcpy(t_teidx->contents(), P.teidx.data(), P.count() * 4);
    const std::size_t N = P.rows();
    const uint32_t wD = uint32_t(D * a / 4);
    enc->setComputePipelineState(ctx->pipeline("suc_gather_words", "uint32_t"));
    set(enc, R(xn_c), 0);
    set(enc, R(t_x), 1);
    set(enc, R(t_src), 2);
    bytes(enc, wD, 3);
    enc->dispatchThreads(MTL::Size(wD, N, 1), MTL::Size(std::min<uint32_t>(256, wD), 1, 1));
    barrier(enc);
    // passes of at most expert_pass_rows() tile rows: the expert stacks' scratch (rotations, float32 outputs, mixed
    // sub-stack outputs) scales with the rows of one launch
    const std::size_t pt = std::max<std::size_t>(1, expert_pass_rows() / P.bt);
    for (std::size_t t0 = 0; t0 < P.count(); t0 += pt) {
      const std::size_t tc = std::min(pt, P.count() - t0), np = tc * P.bt, r0 = t0 * P.bt;
      Tiles tiles{R(t_teidx, t0 * 4), tc, P.bt, P.teidx.data() + t0};
      tiles.exact = !ft;
      const BufRef xr = R(t_x, r0 * D * a), gr = R(t_g, r0 * F * a), ur = R(t_u, r0 * F * a), er = R(t_reid, r0 * 4);
      if (t0) barrier(enc);   // the previous pass used the stacks' scratch
      ExpertStack::encode_group(*ctx, enc, {{L.eu.get(), xr, ur, &es_u}, {L.eg.get(), xr, gr, &es_g}}, er, np, opt.act,
                                &tiles);
      swiglu(enc, gr, ur, gr, F, np);
      barrier(enc);
      ExpertStack::encode_group(*ctx, enc, {{L.ed.get(), gr, xr, &es_d}}, er, np, opt.act, &tiles);
    }
    barrier(enc);
    enc->setComputePipelineState(ctx->pipeline("suc_scatter_words", "uint32_t"));
    set(enc, R(t_x), 0);
    set(enc, R(yk_c), 1);
    set(enc, R(t_dst), 2);
    bytes(enc, wD, 3);
    enc->dispatchThreads(MTL::Size(wD, N, 1), MTL::Size(std::min<uint32_t>(256, wD), 1, 1));
    barrier(enc);
  }

  // MoE of the n chunk rows: xn_c -> mo_c (shared expert and router over the rows, one host sync for the routing,
  // routed experts in tiles; streamed: fetches of consecutive row groups whose experts fit the slots).
  void moe_chunk(MTL::ComputeCommandEncoder*& enc, const Layer& L, std::size_t n) {
    const std::size_t D = cfg.hidden, K = cfg.top_k, F = cfg.moe_inter, E = cfg.experts, a = as();
    scr_off = 0;
    const BufRef shu_ = take(n * F * a), shh_ = take(n * F * a);
    proj_rows(enc, L.mg, R(xn_c), R(shg_c), n, D, F, 1);
    proj_rows(enc, L.mu, R(xn_c), shu_, n, D, F, 2);
    if (fast_off('r') || n < kMmMinRows ||
        !encode_mm_dense(*ctx, enc, R(xn_c), L.router.w, R(gates_c), n, D, E, L.router.in, L.router.out,
                         msl_of(L.router.wdtype), opt.act, "float"))
      for (std::size_t i = 0; i < n; ++i) gemv_xf(enc, L.router, R(xn_c, i * D * a), R(gates_c, i * E * 4));
    barrier(enc);
    {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5051, {int64_t(E), int64_t(K), int64_t(L.bias.dtype)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_route", join({I(long(E)), I(long(K)), L.bias.msl()})};
      }));
      set(enc, R(gates_c), 0);
      set(enc, L.bias.ref, 1);
      set(enc, R(eidx_c), 2);
      set(enc, R(ew_c), 3);
      const float rsc = float(cfg.route_scale);
      bytes(enc, rsc, 4);
      enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size((E + 31) / 32 * 32, 1, 1));
    }
    swiglu(enc, R(shg_c), shu_, shh_, F, n);
    barrier(enc);
    proj_rows(enc, L.md, shh_, R(sh_c), n, F, D, 3);
    if (prof) mark(enc, "moe_routed_experts");
    else split_command_buffer(*ctx, cur_cb, enc, &split_gpu_s);   // the chunk's routing -> host
    std::vector<std::size_t> all(n);
    for (std::size_t i = 0; i < n; ++i) all[i] = i;
    if (!opt.expert_stream) {
      experts_tiled(enc, L, all);
    } else {
      ExpertStream& xs = *opt.expert_stream;
      const auto* ids = static_cast<const uint32_t*>(eidx_c->contents());
      const std::size_t cap = std::max<std::size_t>(xs.max_distinct(L.stream_layer), K);
      std::vector<char> in_uni(E, 0);
      std::vector<std::size_t> group;
      std::vector<uint32_t> gids;
      std::size_t g = 0;
      for (std::size_t i0 = 0; i0 < n;) {   // consecutive row groups whose distinct experts fit the slots
        std::fill(in_uni.begin(), in_uni.end(), 0);
        std::size_t usz = 0, i = i0;
        group.clear();
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
    if (prof) mark(enc, "moe_out_hc_post");
    enc->setComputePipelineState(ctx->pipeline(PsoKey{5053, {int64_t(K), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"glm_moe_out", join({T(), I(long(K))})};
    }));
    set(enc, R(yk_c), 0);
    set(enc, R(ew_c), 1);
    set(enc, R(sh_c), 2);
    set(enc, R(mo_c), 3);
    const uint32_t d32 = uint32_t(D);
    bytes(enc, d32, 4);
    enc->dispatchThreads(MTL::Size(D, n, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    barrier(enc);
  }

  // dense MLP of the n chunk rows: xn_c -> mo_c
  void mlp_chunk(MTL::ComputeCommandEncoder* enc, const Layer& L, std::size_t n) {
    const std::size_t D = cfg.hidden, F = cfg.intermediate, a = as();
    scr_off = 0;
    const BufRef g_ = take(n * F * a), u_ = take(n * F * a), h_ = take(n * F * a);
    proj_rows(enc, L.mg, R(xn_c), g_, n, D, F, 1);
    proj_rows(enc, L.mu, R(xn_c), u_, n, D, F, 2);
    barrier(enc);
    swiglu(enc, g_, u_, h_, F, n);
    barrier(enc);
    proj_rows(enc, L.md, h_, R(mo_c), n, F, D, 3);
    barrier(enc);
  }

  // Batched (not row-exact) layer-major chunk of the n tokens in slots [0, n) at positions p0 ...
  void encode_chunk_fast(MTL::ComputeCommandEncoder*& enc, std::size_t n, std::size_t p0, bool with_logits) {
    const std::size_t W = cfg.stream(), D = cfg.hidden, a = as();
    auto X = [&](std::size_t l) { return R(ring, (l % 2) * opt.chunk * W * a); };
    prof_label = "embed";
    {   // the n tokens' hc streams in one launch: tok holds hc copies per token, rows (n · hc, D) = (n, W)
      const Mat& m = embed;
      if (m.kind == Mat::Affine) {
        enc->setComputePipelineState(ctx->pipeline(PsoKey{5060, {int64_t(D), m.bits, int64_t(m.group), int64_t(m.sdtype), int64_t(opt.act)}}, [&]() {
          return std::pair<std::string, std::string>{
              "suc_embed_affine", join({T(), I(m.bits), I(long(m.group)), I(long(m.in)), I(long(m.words)), msl_of(m.sdtype)})};
        }));
        set(enc, R(tok), 0);
        set(enc, m.packed, 1);
        set(enc, m.scales, 2);
        set(enc, m.biases, 3);
        set(enc, X(0), 4);
      } else {
        enc->setComputePipelineState(ctx->pipeline(PsoKey{5061, {int64_t(D), int64_t(m.wdtype), int64_t(opt.act)}}, [&]() {
          return std::pair<std::string, std::string>{"suc_embed_dense", join({T(), I(long(m.in)), msl_of(m.wdtype)})};
        }));
        set(enc, R(tok), 0);
        set(enc, m.w, 1);
        set(enc, X(0), 4);
      }
      enc->dispatchThreads(MTL::Size(D, cfg.hc * n, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
      barrier(enc);
    }
    for (std::size_t l = 0; l < cfg.layers; ++l) {
      Layer& L = layers[l];
      encode_copy_cols(*ctx, enc, X(l).at((n - 1) * W * a), R(hs, l * W * a), 1, W, W, W, opt.act);
      mark(enc, "hc_pre");
      hc_pre_rows(enc, L.ha, X(l), R(coefa_c), L.in_norm, R(xn_c), n);
      mark(enc, L.dsa ? "mla_proj_pools" : "kda_proj");
      if (L.dsa) mla_chunk(enc, L, l, p0, n);
      else kda_chunk(enc, L, n);
      mark(enc, "hc_post_pre");
      hc_post_rows(enc, R(f_c), X(l), R(coefa_c), R(xa_c), n);
      hc_pre_rows(enc, L.hf, R(xa_c), R(coef_c), L.post_norm, R(xn_c), n);
      mark(enc, L.sparse ? "moe_shared_router" : "mlp");
      if (L.sparse) moe_chunk(enc, L, n);
      else mlp_chunk(enc, L, n);
      mark(enc, "hc_post");
      hc_post_rows(enc, R(mo_c), R(xa_c), R(coef_c), X(l + 1), n);
    }
    const BufRef xl = X(cfg.layers);
    encode_copy_cols(*ctx, enc, xl.at((n - 1) * W * a), R(hs, cfg.layers * W * a), 1, W, W, W, opt.act);
    mark(enc, "logits");
    if (rows_out) {   // every row's logits (measurement runs)
      if (!logits_rows) logits_rows = alloc(opt.chunk * cfg.vocab * a, nullptr);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{5062, {int64_t(D), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"glm_stream_mean", join({T(), I(long(D)), I(long(cfg.hc))})};
      }));
      set(enc, xl, 0);
      set(enc, R(f_c), 1);
      const float inv = float(1.0 / double(cfg.hc));
      bytes(enc, inv, 2);
      enc->dispatchThreads(MTL::Size(D, n, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
      barrier(enc);
      rms(enc, R(f_c), norm.ref, R(xn_c), n, D);
      barrier(enc);
      proj_rows(enc, head, R(xn_c), R(logits_rows), n, D, cfg.vocab, 0);
      barrier(enc);
    }
    if (with_logits) {
      barrier(enc);
      encode_logits(enc, R(hs, cfg.layers * W * a));
    }
  }

  void zero_states() {
    for (Layer& L : layers) {
      for (BufPtr* b : {&L.conv_state, &L.rec_state, &L.rawk, &L.rawg})
        if (*b) std::memset((*b)->contents(), 0, (*b)->length());
    }
    std::fill(npools.begin(), npools.end(), 0);
    pos = 0;
    have_logits = false;
  }
};

// ------------------------------------------------------------------------------------------ Glm5Model

Glm5Model::Glm5Model(Context& ctx, const MappedFile& mf, const Glm5Options& opt) : p_(std::make_unique<Impl>()) {
  Pool pool;
  Impl& m = *p_;
  m.ctx = &ctx;
  m.mf = &mf;
  m.opt = opt;
  const Json* arch = mf.file().directory().find("architecture");
  if (!arch || !arch->is_object()) throw FormatError("directory has no architecture object");
  m.cfg = Glm5Config::from_architecture(*arch);
  const Glm5Config& c = m.cfg;
  if (opt.max_context == 0 || opt.chunk == 0) fail("max_context and chunk must be > 0");
  const std::size_t D = c.hidden, W = c.stream(), a = m.as();
  const DType T = dtype_of(opt.act);
  m.eps = float(c.rms_eps);
  m.hc_eps = float(c.hc_eps);
  const std::size_t Hl = c.lin_heads, Dk = c.lin_dim, dl = Hl * Dk, Kc = c.conv_kernel;
  const std::size_t H = c.heads, rank = c.kv_lora, idd = c.index_dim, ih = c.index_heads;
  m.max_pools = opt.max_context / c.kpool + 1;

  m.embed = m.mat("model.embed_tokens.weight", c.vocab, D, false);
  m.head = m.mat("lm_head.weight", c.vocab, D, true);
  m.norm = m.vec("model.norm.weight", D, T);
  m.layers.resize(c.layers);
  m.npools.assign(c.layers, 0);
  std::size_t max_f = 1;
  bool any_dsa = false, any_kda = false, any_moe = false;
  for (std::size_t i = 0; i < c.layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i);
    Impl::Layer& L = m.layers[i];
    L.dsa = c.dsa[i];
    L.sparse = c.sparse[i];
    L.ha = m.hyper(p, "attn");
    L.hf = m.hyper(p, "ffn");
    L.in_norm = m.vec(p + ".input_layernorm.weight", D, T);
    L.post_norm = m.vec(p + ".post_attention_layernorm.weight", D, T);
    const std::string s = p + ".self_attn";
    if (!L.dsa) {
      any_kda = true;
      L.q = m.mat(s + ".q_proj.weight", dl, D);
      L.k = m.mat(s + ".k_proj.weight", dl, D);
      L.v = m.mat(s + ".v_proj.weight", dl, D);
      L.fa = m.mat(s + ".f_a_proj.weight", Dk, D);
      L.fb = m.mat(s + ".f_b_proj.weight", dl, Dk);
      L.ga = m.mat(s + ".g_a_proj.weight", Dk, D);
      L.gb = m.mat(s + ".g_b_proj.weight", dl, Dk);
      L.bproj = m.mat(s + ".b_proj.weight", Hl, D);
      L.o = m.mat(s + ".o_proj.weight", D, dl);
      std::vector<float> cw(3 * dl * Kc);
      std::size_t off = 0;
      for (const char* k : {"q", "k", "v"}) {   // float32 parameters (HF keeps the convs in float32)
        const auto w = m.host(s + "." + k + "_conv1d.weight", dl * Kc, DType::F32);
        std::copy(w.begin(), w.end(), cw.begin() + std::ptrdiff_t(off));
        off += w.size();
      }
      L.convw = m.upload_f32(cw);
      L.dt_bias = m.vec(s + ".dt_bias", dl, DType::F32);
      L.a_log = m.vec(s + ".A_log", Hl, DType::F32);
      L.onorm = m.upload_f32(m.host(s + ".o_norm.weight", Dk, T));
      L.conv_state = m.alloc((Kc - 1) * 3 * dl * a, &m.state_bytes);
      L.rec_state = m.alloc(Hl * Dk * Dk * 4, &m.state_bytes);
    } else {
      any_dsa = true;
      L.q_a = m.mat(s + ".q_a_proj.weight", c.q_lora, D);
      L.q_a_norm = m.vec(s + ".q_a_layernorm.weight", c.q_lora, T);
      L.q_b = m.mat(s + ".q_b_proj.weight", H * c.nope, c.q_lora);
      L.kv_a = m.mat(s + ".kv_a_proj_with_mqa.weight", rank, D);
      L.kv_a_norm = m.vec(s + ".kv_a_layernorm.weight", rank, T);
      L.kv_b = m.vec(s + ".kv_b_proj.weight", H * (c.nope + c.v_dim) * rank, T);
      L.o = m.mat(s + ".o_proj.weight", D, H * c.v_dim);
      const std::string x = s + ".indexer";
      L.iwq_b = m.mat(x + ".wq_b.weight", ih * idd, c.q_lora);
      L.iwk = m.mat(x + ".wk.weight", idd, D);
      L.ik_w = m.vec(x + ".k_norm.weight", idd, T);
      L.ik_b = m.vec(x + ".k_norm.bias", idd, T);
      L.iwp = m.mat(x + ".weights_proj.weight", ih, D);
      L.igate = m.dense(x + ".index_kpool_compress_gate", idd, D, T);
      L.ape = m.vec(x + ".index_kpool_compress_ape", c.kpool * idd, T);
      L.latent = m.alloc(opt.max_context * rank * a, &m.kv_bytes);
      L.rawk = m.alloc(c.kpool * idd * 4, &m.kv_bytes);
      L.rawg = m.alloc(c.kpool * idd * 4, &m.kv_bytes);
      L.pools = m.alloc(m.max_pools * idd * a, &m.kv_bytes);
    }
    const std::string f = p + ".mlp";
    if (!L.sparse) {
      L.mg = m.mat(f + ".gate_proj.weight", c.intermediate, D);
      L.mu = m.mat(f + ".up_proj.weight", c.intermediate, D);
      L.md = m.mat(f + ".down_proj.weight", D, c.intermediate);
      max_f = std::max(max_f, c.intermediate);
      continue;
    }
    any_moe = true;
    max_f = std::max(max_f, c.moe_inter);
    L.router = m.dense(f + ".gate.weight", c.experts, D, T);
    L.bias = m.vec(f + ".gate.e_score_correction_bias", c.experts, DType::F32);
    L.mg = m.mat(f + ".shared_experts.gate_proj.weight", c.moe_inter, D);
    L.mu = m.mat(f + ".shared_experts.up_proj.weight", c.moe_inter, D);
    L.md = m.mat(f + ".shared_experts.down_proj.weight", D, c.moe_inter);
    const bool streamed = opt.expert_stream != nullptr;
    L.eg = std::make_unique<ExpertStack>(ExpertStack::from_file(ctx, mf, f, "gate_proj", c.experts, opt.mixed_rows, streamed));
    L.eu = std::make_unique<ExpertStack>(ExpertStack::from_file(ctx, mf, f, "up_proj", c.experts, opt.mixed_rows, streamed));
    L.ed = std::make_unique<ExpertStack>(ExpertStack::from_file(ctx, mf, f, "down_proj", c.experts, opt.mixed_rows, streamed));
    for (const ExpertStack* e : {L.eg.get(), L.eu.get()})
      if (e->in_features() != D || e->out_features() != c.moe_inter) fail(f + ": expert gate / up shape");
    if (L.ed->in_features() != c.moe_inter || L.ed->out_features() != D) fail(f + ": expert down shape");
    m.copied_bytes += L.eg->copied_bytes() + L.eu->copied_bytes() + L.ed->copied_bytes();
    if (streamed) {   // per expert: {gate, up, down} x {packed indices, row scales}
      std::vector<std::vector<ExpertStream::Range>> r(c.experts);
      for (std::size_t e = 0; e < c.experts; ++e)
        for (const ExpertStack* xst : {L.eg.get(), L.eu.get(), L.ed.get()}) {
          const auto [a0, a1] = xst->expert_ranges(e);
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
  m.fast_ok = opt.chunk > 1 && D <= 4096 && Impl::rms_tpg(D) <= 1024;
  if (const char* e = std::getenv("SUC_GLM_PROFILE")) m.prof = e[0] && e[0] != '0';

  uint64_t* ab = &m.arena_bytes;
  const std::size_t K = std::max<std::size_t>(c.top_k, 1), F = c.moe_inter;
  m.tok = m.alloc(opt.chunk * c.hc * 4, ab);
  m.hs = m.alloc((c.layers + 1) * W * a, ab);
  m.xa = m.alloc(W * a, ab);
  m.xf = m.alloc(W * 4, ab);
  m.ss = m.alloc(4, ab);
  m.xs = m.alloc(W * 4, ab);
  m.mixraw = m.alloc((2 + c.hc) * c.hc * 4, ab);
  m.coef_a = m.alloc((2 + c.hc) * c.hc * 4, ab);
  m.coef_f = m.alloc((2 + c.hc) * c.hc * 4, ab);
  m.y = m.alloc(D * a, ab);
  m.xn = m.alloc(D * a, ab);
  m.xnf = m.alloc(D * 4, ab);
  m.f = m.alloc(std::max(D, rank) * a, ab);
  m.mo = m.alloc(D * a, ab);
  if (any_kda) {
    m.qkv = m.alloc(3 * dl * a, ab);
    m.conv = m.alloc(3 * dl * a, ab);
    m.qkf = m.alloc(2 * dl * 4, ab);
    m.vf = m.alloc(dl * 4, ab);
    m.fa = m.alloc(Dk * a, ab);
    m.fb = m.alloc(dl * a, ab);
    m.ga = m.alloc(Dk * a, ab);
    m.gb = m.alloc(dl * a, ab);
    m.bb = m.alloc(Hl * a, ab);
    m.decay = m.alloc(dl * 4, ab);
    m.beta = m.alloc(Hl * 4, ab);
    m.o32 = m.alloc(dl * 4, ab);
    m.on32 = m.alloc(dl * 4, ab);
    m.oy = m.alloc(dl * a, ab);
  }
  if (any_dsa) {
    m.qa = m.alloc(c.q_lora * a, ab);
    m.qr = m.alloc(c.q_lora * a, ab);
    m.q = m.alloc(H * c.nope * a, ab);
    m.ikr = m.alloc(idd * a, ab);
    m.ik = m.alloc(idd * a, ab);
    m.ig = m.alloc(idd * a, ab);
    m.iw = m.alloc(ih * a, ab);
    m.iq = m.alloc(ih * idd * a, ab);
    m.iscore = m.alloc(m.max_pools * 4, ab);
    m.sel = m.alloc(m.max_pools, ab);
    m.qabs = m.alloc(H * rank * a, ab);
    m.att = m.alloc(H * rank * a, ab);
    m.ov = m.alloc(H * c.v_dim * a, ab);
    if (!m.vector_attention()) {
      m.scores = m.alloc(H * opt.max_context * a, ab);
      m.probs = m.alloc(H * opt.max_context * a, ab);
    }
    // GEMM partials: split-k only when K >= max(M, N), so parts · N stays below ~2 · (K / 16) · 32 + N
    const std::size_t att_parts = m.vector_attention() ? 0 : std::max(H * std::max<std::size_t>(opt.max_context, 4096),
                                                                      32 * H * rank);
    m.cpart = m.alloc(4 * std::max(att_parts, ih * std::max<std::size_t>(m.max_pools, 32 * idd)), ab);
  }
  if (any_moe) {
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
    m.sh = m.alloc(D * a, ab);
    if (opt.expert_stream && opt.chunk > 1 && !m.fast()) {   // row-exact layer-major prefill chunks
      const std::size_t n = opt.chunk;
      m.ring = m.alloc(2 * n * W * a, ab);
      m.xa_c = m.alloc(n * W * a, ab);
      m.coef_c = m.alloc(n * (2 + c.hc) * c.hc * 4, ab);
      m.xk_c = m.alloc(n * K * D * a, ab);
      m.sh_c = m.alloc(n * D * a, ab);
      m.eidx_c = m.alloc(n * K * 4, ab);
      m.ew_c = m.alloc(n * K * 4, ab);
    }
  }
  if (m.fast()) {   // batched (not row-exact) layer-major chunks
    const std::size_t n = opt.chunk, cw = (2 + c.hc) * c.hc * 4, E = std::max<std::size_t>(c.experts, 1);
    m.ring = m.alloc(2 * n * W * a, ab);
    m.xa_c = m.alloc(n * W * a, ab);
    m.coef_c = m.alloc(n * cw, ab);
    m.coefa_c = m.alloc(n * cw, ab);
    m.mix_c = m.alloc(n * cw, ab);
    m.ss_c = m.alloc(n * 4, ab);
    m.xn_c = m.alloc(n * D * a, ab);
    m.f_c = m.alloc(n * D * a, ab);
    m.mo_c = m.alloc(n * D * a, ab);
    std::size_t sb = 0;
    auto al = [](std::size_t b) { return (b + 255) / 256 * 256; };
    if (any_kda) {
      std::size_t t = 0;
      for (std::size_t b : {3 * n * dl * a, 3 * n * dl * a, 2 * n * dl * 4, n * dl * 4, n * Dk * a, n * Dk * a,
                            n * Hl * a, n * dl * a, n * dl * a, n * dl * 4, n * Hl * 4})
        t += al(b);
      sb = std::max(sb, t);
    }
    if (any_dsa) {
      std::size_t t = 0;
      for (std::size_t b : {n * c.q_lora * a, n * c.q_lora * a, n * rank * a, n * idd * a, n * idd * a, n * idd * a,
                            n * H * c.nope * a, (c.kpool + n) * idd * 4, (c.kpool + n) * idd * 4, n * H * rank * a,
                            n * H * rank * a, n * H * c.v_dim * a, n * ih * idd * a, n * ih * a})
        t += al(b);
      sb = std::max(sb, t);
      m.iscore_c = m.alloc(n * m.max_pools * 4, ab);
      m.sel_c = m.alloc(n * m.max_pools, ab);
    }
    sb = std::max(sb, 3 * al(n * max_f * a));
    m.scr_bytes = sb;
    m.scr = m.alloc(sb, ab);
    if (any_moe) {
      m.sh_c = m.alloc(n * D * a, ab);
      m.shg_c = m.alloc(n * F * a, ab);
      m.gates_c = m.alloc(n * E * 4, ab);
      m.eidx_c = m.alloc(n * K * 4, ab);
      m.ew_c = m.alloc(n * K * 4, ab);
      m.yk_c = m.alloc(n * K * D * a, ab);
    }
  }
  m.mgb = m.alloc(max_f * a, ab);
  m.mub = m.alloc(max_f * a, ab);
  m.mhb = m.alloc(max_f * a, ab);
  m.logits = m.alloc(c.vocab * a, ab);
  m.amax = m.alloc(4, ab);
}

Glm5Model::~Glm5Model() = default;

const Glm5Config& Glm5Model::config() const { return p_->cfg; }
std::size_t Glm5Model::position() const { return p_->pos; }
uint64_t Glm5Model::chunk_fetches() const { return p_->chunk_fetches; }
void Glm5Model::reset() { p_->zero_states(); }
double Glm5Model::last_gpu_seconds() const { return p_->gpu_s; }
double Glm5Model::last_encode_seconds() const { return p_->encode_s; }

uint32_t Glm5Model::forward(const std::vector<uint32_t>& tokens, bool with_logits) {
  Impl& m = *p_;
  const Glm5Config& c = m.cfg;
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
    if (m.fast() && n > 1) {
      m.encode_chunk_fast(enc, n, m.pos, with_logits && last_chunk);
    } else if (m.opt.expert_stream && m.ring && n > 1) {
      m.encode_chunk_layer_major(enc, n, m.pos, with_logits && last_chunk);
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
    if (m.prof && m.prof_label) m.prof_acc[m.prof_label] += cb->GPUEndTime() - cb->GPUStartTime();
    m.prof_label = nullptr;
    if (m.rows_out) {
      const std::size_t V = c.vocab, o = m.rows_out->size();
      m.rows_out->resize(o + n * V);
      if (m.fast() && n > 1) act_to_f32(m.logits_rows->contents(), m.rows_out->data() + o, n * V, m.opt.act);
      else if (n == 1) act_to_f32(m.logits->contents(), m.rows_out->data() + o, V, m.opt.act);
      else fail("forward_rows: chunks of decode steps have no per-row logits");
    }
    m.pos += n;
  }
  m.have_logits = with_logits;
  if (with_logits) result = *static_cast<const uint32_t*>(m.amax->contents());
  return result;
}

std::vector<float> Glm5Model::forward_rows(const std::vector<uint32_t>& tokens) {
  std::vector<float> out;
  if (!p_->fast()) {   // decode steps, one per command buffer: each records its logits
    for (uint32_t t : tokens) {
      forward({t}, true);
      const auto l = logits();
      out.insert(out.end(), l.begin(), l.end());
    }
    return out;
  }
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

bool Glm5Model::batched_prefill() const { return p_->fast(); }

std::vector<std::pair<std::string, double>> Glm5Model::profile() const {
  return {p_->prof_acc.begin(), p_->prof_acc.end()};
}
void Glm5Model::reset_profile() { p_->prof_acc.clear(); }

std::vector<float> Glm5Model::logits() const {
  if (!p_->have_logits) return {};
  std::vector<float> v(p_->cfg.vocab);
  act_to_f32(p_->logits->contents(), v.data(), v.size(), p_->opt.act);
  return v;
}

std::vector<float> Glm5Model::hidden_states() const {
  const Glm5Config& c = p_->cfg;
  std::vector<float> s((c.layers + 1) * c.stream());
  act_to_f32(p_->hs->contents(), s.data(), s.size(), p_->opt.act);
  return s;
}

Glm5Memory Glm5Model::memory() const {
  Glm5Memory r;
  r.mapped_file = p_->mf->mapped_bytes();
  r.copied_weights = p_->copied_bytes;
  r.kv_cache = p_->kv_bytes;
  r.states = p_->state_bytes;
  r.arena = p_->arena_bytes;
  uint64_t ws = 0;
  for (const Workspace& w : p_->ws) ws += w.xr_bytes + w.y32_bytes + w.y32b_bytes;
  r.workspace = ws + p_->es_g.bytes() + p_->es_u.bytes() + p_->es_d.bytes();
  return r;
}

std::size_t Glm5Model::suc_projections() const {
  std::size_t n = 0;
  auto add = [&](const Mat& m) { n += m.kind == Mat::Suc; };
  for (const auto& L : p_->layers)
    for (const Mat* m : {&L.q, &L.k, &L.v, &L.o, &L.q_a, &L.q_b, &L.kv_a, &L.mg, &L.mu, &L.md}) add(*m);
  add(p_->head);
  return n;
}

Glm5StackStats Glm5Model::expert_stacks() const {
  Glm5StackStats s;
  for (const auto& L : p_->layers)
    for (const ExpertStack* e : {L.eg.get(), L.eu.get(), L.ed.get()}) {
      if (!e) continue;
      ++s.stacks;
      s.mixed += e->mixed();
      s.sub_stacks += e->sub_stacks();
    }
  return s;
}

}  // namespace suc::metal
