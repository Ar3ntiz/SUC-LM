// Qwen3 dense model graph on the Metal backend (see suc/model.hpp). Kernels: cpp/metal/suc_model_kernels.metal
// (MLX-order ports of the ops the Python runtime takes from MLX) and the SUC projection kernels of metal::Linear.
#include "suc/model.hpp"
#include "suc/expert_stream.hpp"
#include "suc/switch.hpp"

#include <mach/mach.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <stdexcept>
#include <variant>

#include "suc/bits.hpp"
#include "suc/hadamard.hpp"
#include "suc/json.hpp"

namespace suc::metal {

namespace {

struct Pool {
  NS::AutoreleasePool* p = NS::AutoreleasePool::alloc()->init();
  ~Pool() { p->release(); }
};

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("model: " + msg); }

std::string I(long v) { return std::to_string(v); }

constexpr int kTgFloats = 8192 - 64;   // rotation.TG_FLOATS (as in metal.cpp)

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
    default: throw FormatError("unsupported floating dtype " + std::string(dtype_name(t)));
  }
}

std::optional<Act> act_of(DType t) {
  if (t == DType::F16) return Act::F16;
  if (t == DType::BF16) return Act::BF16;
  if (t == DType::F32) return Act::F32;
  return std::nullopt;
}

std::size_t get_size(const Json& a, const char* key) {
  const Json* v = a.find(key);
  if (!v || !v->is_int()) throw FormatError(std::string("architecture: missing integer '") + key + "'");
  return std::size_t(v->as_uint());
}

void barrier(MTL::ComputeCommandEncoder* enc) { enc->memoryBarrier(MTL::BarrierScopeBuffers); }

void set(MTL::ComputeCommandEncoder* enc, BufRef r, int i) { enc->setBuffer(r.buffer, r.offset, i); }

}  // namespace

// ------------------------------------------------------------------------------------------ config / names

ModelConfig ModelConfig::from_architecture(const Json& a) {
  ModelConfig c;
  const Json* mt = a.find("model_type");
  if (!mt || !mt->is_string()) throw FormatError("architecture: no model_type");
  c.model_type = mt->as_string();
  if (c.model_type != "qwen3" && c.model_type != "qwen3_moe")
    throw FormatError("architecture: model_type '" + c.model_type + "' is not supported (qwen3, qwen3_moe)");
  if (c.model_type == "qwen3_moe") {
    c.num_experts = get_size(a, "num_experts");
    c.top_k = get_size(a, "num_experts_per_tok");
    c.moe_intermediate = get_size(a, "moe_intermediate_size");
    if (const Json* v = a.find("decoder_sparse_step"); v && !v->is_null()) c.decoder_sparse_step = std::size_t(v->as_uint());
    if (const Json* v = a.find("mlp_only_layers"); v && v->is_array())
      for (const Json& e : v->as_array()) c.mlp_only_layers.push_back(std::size_t(e.as_uint()));
    if (const Json* v = a.find("norm_topk_prob"); v && !v->is_null()) c.norm_topk_prob = v->as_bool();
    if (c.decoder_sparse_step == 0) throw FormatError("architecture: decoder_sparse_step must be > 0");
    if (c.num_experts && (c.top_k == 0 || c.top_k > c.num_experts)) throw FormatError("architecture: num_experts_per_tok");
  }
  c.hidden = get_size(a, "hidden_size");
  c.intermediate = get_size(a, "intermediate_size");
  c.layers = get_size(a, "num_hidden_layers");
  c.heads = get_size(a, "num_attention_heads");
  c.kv_heads = a.find("num_key_value_heads") ? get_size(a, "num_key_value_heads") : c.heads;
  c.head_dim = a.find("head_dim") && !a["head_dim"].is_null() ? get_size(a, "head_dim") : c.hidden / c.heads;
  c.vocab = get_size(a, "vocab_size");
  if (const Json* v = a.find("rms_norm_eps")) c.rms_norm_eps = v->as_double();
  if (const Json* v = a.find("rope_theta")) c.rope_theta = v->as_double();
  if (const Json* v = a.find("tie_word_embeddings"); v && !v->is_null()) c.tie_word_embeddings = v->as_bool();
  if (const Json* v = a.find("torch_dtype"); v && v->is_string()) c.torch_dtype = v->as_string();
  if (const Json* v = a.find("rope_scaling"); v && !v->is_null())
    throw FormatError("architecture: rope_scaling is not supported yet");
  if (const Json* v = a.find("attention_bias"); v && !v->is_null() && v->as_bool())
    throw FormatError("architecture: attention_bias is not supported");
  if (const Json* v = a.find("use_sliding_window"); v && !v->is_null() && v->as_bool())
    throw FormatError("architecture: sliding-window attention is not supported");
  if (const Json* v = a.find("hidden_act"); v && v->is_string() && v->as_string() != "silu")
    throw FormatError("architecture: hidden_act must be silu");
  if (c.heads % c.kv_heads) throw FormatError("architecture: heads not a multiple of kv heads");
  if (c.head_dim % 32 || c.head_dim > 256) throw FormatError("architecture: head_dim must be a multiple of 32, <= 256");
  return c;
}

bool ModelConfig::sparse_layer(std::size_t l) const {
  return num_experts > 0 && (l + 1) % decoder_sparse_step == 0 &&
         std::find(mlp_only_layers.begin(), mlp_only_layers.end(), l) == mlp_only_layers.end();
}

LayerNames NameTable::layer(std::size_t i) const {
  const std::string p = prefix + std::to_string(i) + ".";
  return {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight",
          p + "self_attn.o_proj.weight", p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
          p + "mlp.down_proj.weight", p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
          p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight"};
}

NameTable NameTable::for_family(const std::string& model_type) {
  if (model_type == "qwen3" || model_type == "qwen3_moe")
    return {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight", "model.layers."};
  throw FormatError("no tensor-name table for family '" + model_type + "'");
}

// ------------------------------------------------------------------------------------------ weights

namespace {

// A (out, in) matrix used as y = x Wᵀ, or as an embedding table (rows gathered).
struct Matrix {
  enum Kind { Suc, Affine, Dense } kind = Dense;
  std::string name;
  std::size_t out = 0, in = 0;
  std::unique_ptr<Linear> lin;            // Suc
  BufRef packed, scales, biases;          // Affine (zero-copy)
  unsigned bits = 0;
  std::size_t group = 0, words = 0;
  DType sdtype = DType::F16;
  BufRef w;                               // Dense (zero-copy)
  DType wdtype = DType::BF16;
};

Matrix load_matrix(Context& ctx, const MappedFile& mf, const TensorDesc& t, bool allow_suc) {
  Matrix m;
  m.name = t.name;
  if (t.shape.size() != 2) throw FormatError(t.name + ": expected a 2-D tensor");
  m.out = t.shape[0];
  m.in = t.shape[1];
  const std::string& rep = t.representation_type;
  if (rep == "vq" || rep == "additive" || rep == "tcq") {
    if (!allow_suc) throw FormatError(t.name + ": SUC codes cannot serve an embedding gather");
    m.kind = Matrix::Suc;
    m.lin = std::make_unique<Linear>(Linear::from_file(ctx, mf, t));
    return m;
  }
  if (t.has_residual()) throw FormatError(t.name + ": residual on a non-SUC base is not supported");
  if (rep == "affine") {
    const Json& p = *t.params;
    m.kind = Matrix::Affine;
    m.bits = unsigned(p["bits"].as_uint());
    m.group = std::size_t(p["group_size"].as_uint());
    const ArrayRecord& qi = t.array("indices");
    const ArrayRecord& sc = t.array("scales");
    const ArrayRecord& bi = t.array("biases");
    if (!qi.packed || qi.bits != m.bits) throw FormatError(t.name + ": affine indices must be bit-packed");
    m.words = words_per_row(m.in, m.bits);
    if (qi.shape != std::vector<uint64_t>{m.out, m.words}) throw FormatError(t.name + ": affine index layout");
    if (m.in % m.group || m.in % 8) throw FormatError(t.name + ": in_features must be a multiple of group_size and 8");
    if (sc.dtype != bi.dtype || sc.shape != std::vector<uint64_t>{m.out, m.in / m.group} || sc.shape != bi.shape)
      throw FormatError(t.name + ": affine scales / biases");
    m.sdtype = sc.dtype;
    msl_of(m.sdtype);
    m.packed = mf.at(qi);
    m.scales = mf.at(sc);
    m.biases = mf.at(bi);
    return m;
  }
  if (rep == "raw" || rep == "bf16") {
    const ArrayRecord& w = t.arrays.at(0);
    m.kind = Matrix::Dense;
    m.wdtype = w.dtype;
    msl_of(m.wdtype);
    if (m.in % 8) throw FormatError(t.name + ": in_features must be a multiple of 8");
    m.w = mf.at(w);
    return m;
  }
  throw FormatError(t.name + ": representation '" + rep + "' has no model-graph kernel (scalar is not supported yet)");
}

}  // namespace

// ------------------------------------------------------------------------------------------ Model

struct Model::Impl {
  Context* ctx = nullptr;
  const MappedFile* mf = nullptr;
  ModelOptions opt;
  ModelConfig cfg;
  NameTable names;

  // A fused projection group (= suc.runtime.compressed_model.SucInputGroup with fuse=True): members with equal
  // FusedLinear::key are row-concatenated into one FusedLinear (>= 2 members), the others run alone on the shared x̃.
  struct Group {
    struct Part {
      std::unique_ptr<FusedLinear> fused;   // null: a single member
      std::vector<std::size_t> members;     // member indices served by this part
    };
    std::vector<Part> parts;
    bool active() const { return !parts.empty(); }
  };
  struct Layer {
    Matrix q, k, v, o, gate, up, down;
    BufRef in_norm, post_norm, q_norm, k_norm;
    Group qkv, gu;
    std::unique_ptr<MoeBlock> moe;   // sparse (qwen3_moe) layers: router + experts instead of gate / up / down
  };
  // One output of a projection group: destination (may be redirected to a view of the fused buffer when view_ok).
  struct Out {
    const Matrix* m;
    BufRef* y;
    bool view_ok;
  };
  Matrix embed;
  const Matrix* head = nullptr;
  Matrix lm_head;            // untied only
  BufRef final_norm;
  std::vector<Layer> layers;
  std::vector<BufPtr> owned;   // converted norm weights
  uint64_t copied_norm_bytes = 0;

  // arena
  BufPtr tok, hs, hmid, xn, xr, q, k, att, ob, db, g, u, a, logits, amax, kcache, vcache, yfused;
  bool share_rotations = true, fuse_norm = true, fuse_projections = true, fuse_swiglu = false;
  uint64_t fused_copied_bytes = 0;
  // one-row copies of fused outputs whose destination is not read by the next stage (v -> V cache): encoded by the
  // caller together with that stage instead of after an extra barrier
  struct CopyOp {
    BufRef src, dst;
    std::size_t cols, src_stride;
  };
  bool defer_copies = false;
  std::vector<CopyOp> deferred;
  void flush_copies(MTL::ComputeCommandEncoder* enc) {
    for (const CopyOp& c : deferred) encode_copy_cols(*ctx, enc, c.src, c.dst, 1, c.cols, c.src_stride, c.cols, opt.act);
    deferred.clear();
  }
  MoeScratch moe_scratch;   // shared by the sparse layers (they run one after the other)
  // expert streaming: the command buffer being encoded (replaced at each streamed layer's host sync), GPU time of the
  // finished parts, next streamed layer per layer (-1), prediction scratch for prefetch
  MTL::CommandBuffer* cur_cb = nullptr;
  double split_gpu_s = 0.0;
  std::vector<int> next_moe;
  MoeScratch pred_scratch;
  BufPtr pred_eidx, pred_scores;
  std::size_t pred_b = 0;
  std::size_t fused_parts = 0;
  Workspace ws[3];               // scratch of the i-th projection of a group (members of one group interleave)
  std::size_t logits_rows_cap = 1;
  std::size_t pos = 0, rows = 0, logit_rows = 0;
  double gpu_s = 0.0, encode_s = 0.0;
  uint64_t arena_bytes = 0, kv_bytes = 0;
  float eps = 0.0f, rope_base_log2 = 0.0f, attn_scale = 0.0f;

  std::size_t as() const { return act_size(opt.act); }
  const char* T() const { return act_msl(opt.act); }

  BufPtr alloc(std::size_t bytes, uint64_t* counter) {
    if (counter) *counter += bytes;
    return ctx->new_buffer(bytes);
  }

  // A norm / raw vector in the activation dtype: zero-copy when stored in it, else converted once.
  BufRef vec(const std::string& name, std::size_t n) {
    const TensorDesc& t = mf->file().tensor(name);
    if (t.representation_type != "raw" && t.representation_type != "bf16")
      throw FormatError(name + ": norm weights must be stored raw");
    const ArrayRecord& r = t.arrays.at(0);
    if (r.logical_numel() != n) throw FormatError(name + ": unexpected size");
    if (act_of(r.dtype) == opt.act) return mf->at(r);
    std::vector<float> f(n);
    const auto bytes = mf->file().payload(r);
    for (std::size_t i = 0; i < n; ++i) {
      if (r.dtype == DType::BF16) f[i] = bf16_to_f32(reinterpret_cast<const uint16_t*>(bytes.data())[i]);
      else if (r.dtype == DType::F16) f[i] = f16_to_f32(reinterpret_cast<const uint16_t*>(bytes.data())[i]);
      else if (r.dtype == DType::F32) f[i] = reinterpret_cast<const float*>(bytes.data())[i];
      else throw FormatError(name + ": norm dtype");
    }
    BufPtr b = ctx->new_buffer(n * as());
    f32_to_act(f.data(), b->contents(), n, opt.act);
    owned.push_back(b);
    copied_norm_bytes += n * as();
    return {b.get(), 0};
  }

  Matrix mat(const std::string& name, std::size_t out, std::size_t in, bool suc = true) {
    Matrix m = load_matrix(*ctx, *mf, mf->file().tensor(name), suc);
    if (m.out != out || m.in != in)
      throw FormatError(name + ": shape (" + std::to_string(m.out) + ", " + std::to_string(m.in) + "), expected (" +
                        std::to_string(out) + ", " + std::to_string(in) + ")");
    return m;
  }

  // ---- encoders
  // y (n, out) = x (n, in) Wᵀ; the affine / dense kernels run one row per dispatch.
  void matvec(MTL::ComputeCommandEncoder* enc, const Matrix& m, BufRef x, BufRef y, std::size_t n = 1) {
    if (m.kind == Matrix::Suc) {
      m.lin->encode(*ctx, enc, x, y, n, opt.act, ws[0], Path::Auto);
      return;
    }
    for (std::size_t r = 0; r < n; ++r) matvec_row(enc, m, x.at(r * m.in * as()), y.at(r * m.out * as()));
  }

  void matvec_row(MTL::ComputeCommandEncoder* enc, const Matrix& m, BufRef x, BufRef y) {
    if (opt.mlx_gemv) {
      // MLX gemv (non-transposed matrix) tiling, as in MoeBlock's router: IN <= 64: SN = 4; IN >= 16·OUT: BN = 8
      const int sn = m.in <= 64 ? 4 : 32;
      const int bn = (m.in > 64 && m.in >= 16 * m.out) ? 8 : 1;
      const int sm = 32 / sn;
      if (m.kind == Matrix::Affine) {
        enc->setComputePipelineState(ctx->pipeline(
            "suc_gemv_affine", join({T(), I(m.bits), I(long(m.group)), I(long(m.in)), I(long(m.out)), I(long(m.words)),
                                     msl_of(m.sdtype), I(sn), "4", I(bn)})));
        set(enc, x, 0);
        set(enc, m.packed, 1);
        set(enc, m.scales, 2);
        set(enc, m.biases, 3);
        set(enc, y, 4);
      } else {
        enc->setComputePipelineState(ctx->pipeline(
            "suc_moe_gemv", join({T(), msl_of(m.wdtype), I(long(m.in)), I(long(m.out)), I(sn), "4", I(bn)})));
        set(enc, x, 0);
        set(enc, m.w, 1);
        set(enc, y, 2);
      }
      enc->dispatchThreadgroups(MTL::Size((m.out + sm - 1) / sm, 1, 1), MTL::Size(32 * bn, 1, 1));
      return;
    }
    constexpr int rps = 4;
    MTL::ComputePipelineState* ps;
    if (m.kind == Matrix::Affine) {
      ps = ctx->pipeline("suc_matvec_affine", join({T(), I(m.bits), I(long(m.group)), I(long(m.in)), I(long(m.out)),
                                                    I(long(m.words)), msl_of(m.sdtype), I(rps)}));
      enc->setComputePipelineState(ps);
      set(enc, x, 0);
      set(enc, m.packed, 1);
      set(enc, m.scales, 2);
      set(enc, m.biases, 3);
      set(enc, y, 4);
    } else {
      ps = ctx->pipeline("suc_matvec_dense", join({T(), I(long(m.in)), I(long(m.out)), msl_of(m.wdtype), I(rps)}));
      enc->setComputePipelineState(ps);
      set(enc, x, 0);
      set(enc, m.w, 1);
      set(enc, y, 4);
    }
    const std::size_t tgs = (m.out + 2 * rps - 1) / (2 * rps);
    enc->dispatchThreadgroups(MTL::Size(tgs, 1, 1), MTL::Size(64, 1, 1));
  }

  // Projections reading the same input x. SUC members with one common seed_in (q|k|v and gate|up of files written by
  // the compress pipeline) share ONE input rotation and run stage by stage (lookup kernels together, then the
  // inverse rotations, then the residuals), one barrier per stage; otherwise one after the other. No trailing barrier.
  static bool shareable(const std::vector<Out>& ms) {
    const Matrix* first = ms.front().m;
    bool ok = true;
    for (const Out& o : ms)
      ok = ok && o.m->kind == Matrix::Suc && o.m->in == first->in && o.m->lin->seed_in() == first->lin->seed_in();
    return ok;
  }

  // Clusters of fusable members in member order (= SucInputGroup.__init__ with fuse=True).
  Group make_group(const std::vector<const Matrix*>& ms) {
    Group g;
    for (const Matrix* m : ms)
      if (m->kind != Matrix::Suc || m->in != ms[0]->in || m->lin->seed_in() != ms[0]->lin->seed_in()) return g;
    std::vector<std::pair<std::string, std::vector<std::size_t>>> clusters;
    for (std::size_t i = 0; i < ms.size(); ++i) {
      const std::string k = FusedLinear::key(*ms[i]->lin);
      auto it = std::find_if(clusters.begin(), clusters.end(), [&](const auto& c) { return !k.empty() && c.first == k; });
      if (it == clusters.end()) clusters.push_back({k.empty() ? "single" + std::to_string(i) : k, {i}});
      else it->second.push_back(i);
    }
    for (auto& [k, idx] : clusters) {
      if (idx.size() >= 2) {
        std::vector<const Linear*> ls;
        for (std::size_t i : idx) ls.push_back(ms[i]->lin.get());
        Group::Part part;
        part.fused = std::make_unique<FusedLinear>(FusedLinear::build(*ctx, ls, opt.fused_in_place));
        part.members = idx;
        fused_copied_bytes += part.fused->copied_bytes();
        ++fused_parts;
        g.parts.push_back(std::move(part));
      } else {
        for (std::size_t i : idx) {
          Group::Part part;
          part.members = {i};
          g.parts.push_back(std::move(part));
        }
      }
    }
    return g;
  }

  // One launch of rotation._prologue (outn = True): PRO = 1 RMSNorm (x, w = weight, eps) / PRO = 2 SwiGLU (x = gate,
  // w = up); writes x̃ = V xn to xr and xn. The tuned layout of N: "blocks" (_pro_blocks_kernel) or "row" (_pro_kernel).
  void prologue(MTL::ComputeCommandEncoder* enc, int pro, BufRef x, BufRef w, BufRef signs, float e, BufRef xn,
                std::size_t N, std::size_t n) {
    const HadamardFactor hf = hadamard_factor(N);
    set(enc, x, 0);
    set(enc, w, 1);
    set(enc, signs, 2);
    if (ctx->rotation_row_layout(N)) {
      // rotation.threads_and_vec((p,), THREADS_MAX = 256, VEC = 4); BUF_FLOATS = 4096, HM_CONST
      const std::size_t t = std::min<std::size_t>(256, std::max<std::size_t>(32, hf.p / 4));
      const std::size_t c = std::min<std::size_t>(4, hf.p / t);
      enc->setComputePipelineState(ctx->pipeline(
          "suc_rot_prologue_row", join({I(long(t)), "4096", "1", I(long(hf.m)), I(long(hf.p)), I(long(c)), I(pro), T()})));
      set(enc, ctx->dummy(), 3);
      enc->setBytes(&e, sizeof e, 4);
      set(enc, {xr.get(), 0}, 5);
      set(enc, xn, 6);
      enc->dispatchThreads(MTL::Size(t, 1, n), MTL::Size(t, 1, 1));
      return;
    }
    const auto [qmax, sgmax] = ctx->rotation_block_config(N);   // = rotation.blocks_shape((p,), (m,), block_config)
    const int q = std::min<int>(qmax, int(hf.p));
    const int sg = std::min({sgmax, int(hf.m * hf.p) / q, kTgFloats / q});
    const int t = 32 * std::max(1, sg);
    const int bufn = std::max<int>(std::min<int>(int(N), 4096), (t / 32) * q);
    enc->setComputePipelineState(ctx->pipeline(
        "suc_rot_prologue_blocks", join({I(t), I(q), I(q / 32), I(bufn), I(long(hf.m)), I(long(hf.p)), I(pro), T()})));
    enc->setBytes(&e, sizeof e, 3);
    set(enc, {xr.get(), 0}, 4);
    set(enc, xn, 5);
    enc->dispatchThreads(MTL::Size(t, N / q, n), MTL::Size(t, 1, 1));
  }

  // xn = rms_norm(h, w); then the projections of xn. With fuse_norm and a shared input rotation of a blocks-layout
  // size: one prologue launch writes x̃ = V xn and xn (rotation._pro_blocks_kernel(outn=True)).
  void norm_project(MTL::ComputeCommandEncoder* enc, BufRef h, BufRef w, BufRef xn, std::size_t n,
                    const std::vector<Out>& ms, const Group* group = nullptr) {
    const Matrix* first = ms.front().m;
    const std::size_t N = first->in;
    const HadamardFactor hf = hadamard_factor(N);
    if (fuse_norm && share_rotations && shareable(ms) && first->lin->seed_in() && N <= 4096 && hf.m && hf.p >= 32) {
      prologue(enc, 1, h, w, first->lin->signs_in(), eps, xn, N, n);
      barrier(enc);
      project(enc, xn, n, ms, BufRef{xr.get(), 0}, group);
      return;
    }
    rms(enc, h, w, xn, n, N);
    barrier(enc);
    project(enc, xn, n, ms, {}, group);
  }

  // h = silu(g) · u and x̃ = V h for the down projection in one launch (rotation._pro_blocks_kernel, PRO = 2;
  // = suc.runtime.compressed_model.SucMLP with fuse_elementwise("swiglu")), then the down projection on x̃.
  bool swiglu_project(MTL::ComputeCommandEncoder* enc, BufRef g, BufRef u, BufRef h, std::size_t n, const Matrix& d,
                      BufRef y) {
    const std::size_t N = d.in;
    const HadamardFactor hf = hadamard_factor(N);
    if (!(fuse_swiglu && d.kind == Matrix::Suc && d.lin->seed_in() && hf.m && hf.p >= 32)) return false;
    prologue(enc, 2, g, u, d.lin->signs_in(), 0.0f, h, N, n);
    barrier(enc);
    BufRef yy = y;
    project(enc, h, n, {{&d, &yy, false}}, BufRef{xr.get(), 0});
    return true;
  }

  void project(MTL::ComputeCommandEncoder* enc, BufRef x, std::size_t n, const std::vector<Out>& ms,
               BufRef xr_ready = {}, const Group* group = nullptr) {
    const Matrix* first = ms.front().m;
    const bool shared = share_rotations && shareable(ms);
    if (!shared) {
      bool sep = false;
      for (const Out& o : ms) {
        if (sep) barrier(enc);
        matvec(enc, *o.m, x, *o.y, n);
        sep = true;
      }
      return;
    }
    BufRef xr = x;
    if (xr_ready) {
      xr = xr_ready;
    } else if (first->lin->seed_in()) {
      xr = {this->xr.get(), 0};
      first->lin->encode_rotate_in(*ctx, enc, x, xr, n, opt.act);
      barrier(enc);
    }
    if (ms.size() > 3) fail("project: at most 3 projections per group");
    if (group && group->active()) {
      project_group(enc, x, xr, n, ms, *group);
      return;
    }
    int stages = 0;
    for (const Out& o : ms) stages = std::max(stages, o.m->lin->stages());
    for (int st = 0; st < stages; ++st) {
      if (st) barrier(enc);
      std::size_t i = 0;
      for (const Out& o : ms) {
        if (st < o.m->lin->stages()) o.m->lin->encode_stage(*ctx, enc, st, x, xr, *o.y, n, opt.act, ws[i]);
        ++i;
      }
    }
  }

  // The parts of a fused group stage by stage on the shared x̃ (= SucInputGroup.outputs), then the fused outputs are
  // handed to the members: views into the fused buffer for one row (view_ok), else strided copies.
  void project_group(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef xr, std::size_t n, const std::vector<Out>& ms,
                     const Group& g) {
    const std::size_t a = as();
    int stages = 0;
    for (const auto& p : g.parts) stages = std::max(stages, p.fused ? p.fused->stages() : ms[p.members[0]].m->lin->stages());
    for (int st = 0; st < stages; ++st) {
      if (st) barrier(enc);
      for (std::size_t i = 0; i < g.parts.size(); ++i) {
        const auto& p = g.parts[i];
        if (p.fused) {
          if (st >= p.fused->stages()) continue;
          std::vector<BufRef> ys;
          for (std::size_t j : p.members) ys.push_back(*ms[j].y);
          p.fused->encode_stage(*ctx, enc, st, x, xr, ys, {yfused.get(), 0}, n, opt.act, ws[i]);
        } else {
          const Out& o = ms[p.members[0]];
          if (st < o.m->lin->stages()) o.m->lin->encode_stage(*ctx, enc, st, x, xr, *o.y, n, opt.act, ws[i]);
        }
      }
    }
    bool copies = false;
    for (const auto& p : g.parts) {
      if (!p.fused || !p.fused->fused_output()) continue;
      const std::size_t tot = p.fused->out_features();
      std::size_t off = 0;
      for (std::size_t j : p.members) {
        const Out& o = ms[j];
        const BufRef src{yfused.get(), off * a};
        if (n == 1 && o.view_ok) {
          *o.y = src;
        } else if (n == 1 && defer_copies) {   // the caller encodes it with its next stage (saves one barrier)
          deferred.push_back({src, *o.y, o.m->out, tot});
        } else {
          if (!copies) barrier(enc);
          copies = true;
          encode_copy_cols(*ctx, enc, src, *o.y, n, o.m->out, tot, o.m->out, opt.act);
        }
        off += o.m->out;
      }
    }
  }

  void gather(MTL::ComputeCommandEncoder* enc, BufRef tok_at, BufRef y, std::size_t n) {
    const Matrix& m = embed;
    if (m.kind == Matrix::Affine) {
      enc->setComputePipelineState(ctx->pipeline(
          "suc_embed_affine", join({T(), I(m.bits), I(long(m.group)), I(long(m.in)), I(long(m.words)), msl_of(m.sdtype)})));
      set(enc, tok_at, 0);
      set(enc, m.packed, 1);
      set(enc, m.scales, 2);
      set(enc, m.biases, 3);
      set(enc, y, 4);
    } else {
      enc->setComputePipelineState(ctx->pipeline("suc_embed_dense", join({T(), I(long(m.in)), msl_of(m.wdtype)})));
      set(enc, tok_at, 0);
      set(enc, m.w, 1);
      set(enc, y, 4);
    }
    enc->dispatchThreads(MTL::Size(m.in, n, 1), MTL::Size(std::min<std::size_t>(256, m.in), 1, 1));
  }

  void rms(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef w, BufRef y, std::size_t rows, std::size_t n) {
    const std::size_t tpg = n <= 4096 ? 32 * (((n + 3) / 4 + 31) / 32) : 1024;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{101, {int64_t(n), int64_t(tpg), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"suc_rms_norm", join({T(), I(long(n)), I(long(tpg))})};
    }));
    set(enc, x, 0);
    set(enc, w, 1);
    set(enc, y, 2);
    enc->setBytes(&eps, sizeof eps, 3);
    enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(tpg, 1, 1));
  }

  void rope(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t rows, int offset, uint32_t rows_per_pos) {
    const std::size_t D = cfg.head_dim;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{102, {int64_t(D), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"suc_rope", join({T(), I(long(D))})};
    }));
    set(enc, x, 0);
    set(enc, y, 1);
    const float scale = 1.0f;
    enc->setBytes(&offset, sizeof offset, 2);
    enc->setBytes(&rope_base_log2, sizeof rope_base_log2, 3);
    enc->setBytes(&scale, sizeof scale, 4);
    enc->setBytes(&rows_per_pos, sizeof rows_per_pos, 5);
    enc->dispatchThreads(MTL::Size(D / 2, rows, 1), MTL::Size(D / 2, 1, 1));
  }

  void elementwise(MTL::ComputeCommandEncoder* enc, const char* kernel, BufRef a, BufRef b, BufRef y, std::size_t n) {
    enc->setComputePipelineState(ctx->pipeline(PsoKey{103, {int64_t(reinterpret_cast<intptr_t>(kernel)), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{kernel, T()};
    }));
    set(enc, a, 0);
    set(enc, b, 1);
    set(enc, y, 2);
    const uint32_t nn = uint32_t(n);
    enc->setBytes(&nn, sizeof nn, 3);
    enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(std::min<std::size_t>(256, n), 1, 1));
  }

  // Two-pass attention choice for one decode row with n keys: {variant, blocks, hpt}; variant 0 = single pass,
  // 1 = sdpa_vector_2pass_1, 2 = sdpa_vector_2pass_1_gqa. $SUC_SDPA_2PASS overrides it for experiments:
  // "auto" (the rule) | "off" | "std:BLOCKS" | "gqa:HPT:BLOCKS" (applied from SUC_SDPA_2PASS_MIN keys, default 1).
  struct Sdpa2 {
    int variant = 0, blocks = 0, hpt = 0;
  };
  Sdpa2 sdpa_choice(int n) const {
    if (const char* e = std::getenv("SUC_SDPA_2PASS"); e && *e && std::string(e) != "auto") {
      const char* mn = std::getenv("SUC_SDPA_2PASS_MIN");
      if (n < (mn ? std::atoi(mn) : 1)) return {};
      std::string v = e;
      if (v.rfind("std:", 0) == 0) return {1, std::atoi(v.c_str() + 4), 0};
      if (v.rfind("gqa:", 0) == 0) {
        const auto c = v.find(':', 4);
        return {2, std::atoi(v.c_str() + c + 1), std::atoi(v.c_str() + 4)};
      }
      return {};
    }
    return sdpa2_rule(n);
  }
  // MLX's dispatch (scaled_dot_product_attention, one query row, no mask): two passes when the device architecture
  // ends in 'd' and n >= 1024, or with GQA (kv heads < q heads) and n >= 4096; blocks by device class and
  // n_simds = gqa factor. Verified bit for bit on the M3 ('g': 32 / 64 blocks, G = 1/2/4/8, up to 8400 keys); the
  // 'd' (M1 Ultra) and 's' branches follow MLX's host code as recalled and are NOT verified (run
  // benchmarks/cpp_sdpa_2pass_probe.py there). MLX 0.32.3 also has a GQA-specialised pass 1 (sdpa_vector_2pass_1_gqa,
  // ported) that the M3 never selected in these tests.
  Sdpa2 sdpa2_rule(int n) const {
    if (opt.sdpa_2pass == 0) return {};
    const std::string& arch = ctx->architecture();
    const char devc = arch.empty() ? ' ' : arch.back();
    const int ns = int(cfg.heads / cfg.kv_heads);   // n_simds = gqa_factor * q_seq_len (1)
    if (!((devc == 'd' && n >= 1024) || (cfg.kv_heads < cfg.heads && n >= 4096))) return {};
    int blocks;
    if (devc == 's') {
      blocks = 64;
      if (n > 1024 && ns > 4) blocks = n <= 8192 ? 128 : n <= 32768 ? 256 : n <= 65536 ? 512 : 1024;
    } else if (devc == 'd') {
      blocks = 128;
      if (ns <= 2 && n > 8192) blocks = 256;
      else if (ns >= 6) blocks = (n >= 16384 && n < 65536) ? 512 : n >= 65536 ? 1024 : 128;
    } else {
      blocks = ns >= 4 ? 64 : 32;
    }
    return {1, blocks, 0};
  }

  BufPtr sdpa_part, sdpa_sums, sdpa_maxs;
  std::size_t sdpa_part_b = 0, sdpa_sums_b = 0, sdpa_maxs_b = 0;

  void sdpa_2pass(MTL::ComputeCommandEncoder* enc, BufRef q, BufRef kc, BufRef vc, BufRef out, int n, const Sdpa2& c) {
    const std::size_t D = cfg.head_dim, Hq = cfg.heads, Hkv = cfg.kv_heads;
    const int G = int(Hq / Hkv);
    const std::size_t B = std::size_t(c.blocks);
    auto grow = [&](BufPtr& b, std::size_t& cap, std::size_t bytes) {
      if (cap < bytes) b = ctx->new_buffer(bytes), cap = bytes;
    };
    grow(sdpa_part, sdpa_part_b, Hq * B * D * as());
    grow(sdpa_sums, sdpa_sums_b, Hq * B * 4);
    grow(sdpa_maxs, sdpa_maxs_b, Hq * B * 4);
    if (c.variant == 1)
      enc->setComputePipelineState(ctx->pipeline("suc_sdpa_2pass_1", join({T(), I(long(D)), I(long(D)), I(c.blocks)})));
    else
      enc->setComputePipelineState(
          ctx->pipeline("suc_sdpa_2pass_1_gqa", join({T(), I(long(D)), I(long(D)), I(G), I(c.hpt)})));
    set(enc, q, 0);
    set(enc, kc, 1);
    set(enc, vc, 2);
    enc->setBuffer(sdpa_part.get(), 0, 3);
    enc->setBuffer(sdpa_sums.get(), 0, 4);
    enc->setBuffer(sdpa_maxs.get(), 0, 5);
    const std::size_t head_stride = D, seq_stride = Hkv * D;
    enc->setBytes(&n, sizeof n, 7);
    enc->setBytes(&head_stride, sizeof head_stride, 8);
    enc->setBytes(&seq_stride, sizeof seq_stride, 9);
    enc->setBytes(&head_stride, sizeof head_stride, 10);
    enc->setBytes(&seq_stride, sizeof seq_stride, 11);
    enc->setBytes(&attn_scale, sizeof attn_scale, 12);
    enc->dispatchThreadgroups(MTL::Size(Hkv, 1, B), MTL::Size(32, G, 1));
    barrier(enc);
    enc->setComputePipelineState(ctx->pipeline("suc_sdpa_2pass_2", join({T(), I(long(D))})));
    enc->setBuffer(sdpa_part.get(), 0, 0);
    enc->setBuffer(sdpa_sums.get(), 0, 1);
    enc->setBuffer(sdpa_maxs.get(), 0, 2);
    set(enc, out, 3);
    const int blocks = c.blocks;
    enc->setBytes(&blocks, sizeof blocks, 4);
    enc->dispatchThreadgroups(MTL::Size(Hq, 1, 1), MTL::Size(1024, 1, 1));
  }

  // queries (rows, Hq, D) at positions N - rows .. N - 1 (causal), keys / values 0 .. N - 1 in the cache
  void sdpa(MTL::ComputeCommandEncoder* enc, BufRef q, BufRef kc, BufRef vc, BufRef out, int n, std::size_t rows) {
    const std::size_t D = cfg.head_dim;
    if (rows == 1) {
      const Sdpa2 c = sdpa_choice(n);
      if (c.variant) {
        sdpa_2pass(enc, q, kc, vc, out, n, c);
        return;
      }
    }
    enc->setComputePipelineState(ctx->pipeline(PsoKey{104, {int64_t(D), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"suc_sdpa_vector", join({T(), I(long(D)), I(long(D))})};
    }));
    set(enc, q, 0);
    set(enc, kc, 1);
    set(enc, vc, 2);
    set(enc, out, 3);
    const int gqa = int(cfg.heads / cfg.kv_heads);
    const std::size_t head_stride = D, seq_stride = cfg.kv_heads * D;   // cache rows [position][kv head][D]
    enc->setBytes(&gqa, sizeof gqa, 4);
    enc->setBytes(&n, sizeof n, 5);
    enc->setBytes(&head_stride, sizeof head_stride, 6);
    enc->setBytes(&seq_stride, sizeof seq_stride, 7);
    enc->setBytes(&head_stride, sizeof head_stride, 8);
    enc->setBytes(&seq_stride, sizeof seq_stride, 9);
    enc->setBytes(&attn_scale, sizeof attn_scale, 10);
    enc->dispatchThreadgroups(MTL::Size(cfg.heads, rows, 1), MTL::Size(1024, 1, 1));
  }

  // argmax of logits row `row` into amax[row], or into `dst` when given (the next token slot of the token buffer)
  void argmax(MTL::ComputeCommandEncoder* enc, std::size_t row, BufRef dst = {}) {
    enc->setComputePipelineState(ctx->pipeline("suc_argmax", join({T(), "1024"})));
    set(enc, {logits.get(), row * cfg.vocab * as()}, 0);
    set(enc, dst ? dst : BufRef{amax.get(), row * 4}, 1);
    const uint32_t n = uint32_t(cfg.vocab);
    enc->setBytes(&n, sizeof n, 2);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
  }

  // One command buffer's work: n tokens at positions pos.. (read from the token buffer), logits / argmax of the last
  // lrows rows; next_tok: the argmax goes to tok[pos + n] (pipelined generation) instead of amax.
  // enc / cur_cb are replaced at every streamed sparse layer (split_command_buffer).
  void encode_chunk(MTL::ComputeCommandEncoder*& enc, std::size_t pos, std::size_t n, std::size_t lrows, bool next_tok);
};

void Model::Impl::encode_chunk(MTL::ComputeCommandEncoder*& enc, std::size_t pos, std::size_t n, std::size_t lrows,
                               bool next_tok) {
  Impl& m = *this;
  const ModelConfig& c = m.cfg;
  const std::size_t H = c.hidden, D = c.head_dim, KD = c.kv_heads * D, F = c.intermediate, a = m.as();
  auto B = [](const BufPtr& b, std::size_t off = 0) { return BufRef{b.get(), off}; };
  const std::size_t hl = n * H * a;   // one layer slot of hs (n rows)

  m.gather(enc, B(m.tok, pos * 4), B(m.hs), n);
  barrier(enc);
  const std::size_t kv_layer = m.opt.max_context * KD * a;
  for (std::size_t l = 0; l < c.layers; ++l) {
    const Impl::Layer& L = m.layers[l];
    const BufRef h = B(m.hs, l * hl), hout = B(m.hs, (l + 1) * hl);
    const BufRef kc = B(m.kcache, l * kv_layer), vc = B(m.vcache, l * kv_layer);
    const BufRef krow = kc.at(pos * KD * a), vrow = vc.at(pos * KD * a);   // cache rows [position][kv head][D]
    BufRef qb = B(m.q), kb = B(m.k), vb = vrow;   // v: straight into the V cache
    m.defer_copies = true;   // v's copy into the V cache runs alongside the q / k norms (read only by sdpa)
    m.norm_project(enc, h, L.in_norm, B(m.xn), n, {{&L.q, &qb, true}, {&L.k, &kb, true}, {&L.v, &vb, false}},
                   &L.qkv);
    m.defer_copies = false;
    barrier(enc);
    m.flush_copies(enc);
    m.rms(enc, qb, L.q_norm, B(m.q), n * c.heads, D);
    m.rms(enc, kb, L.k_norm, B(m.k), n * c.kv_heads, D);
    barrier(enc);
    m.rope(enc, B(m.q), B(m.q), n * c.heads, int(pos), uint32_t(c.heads));
    m.rope(enc, B(m.k), krow, n * c.kv_heads, int(pos), uint32_t(c.kv_heads));   // straight into the K cache
    barrier(enc);
    m.sdpa(enc, B(m.q), kc, vc, B(m.att), int(pos + n), n);
    barrier(enc);
    BufRef ob = B(m.ob);
    m.project(enc, B(m.att), n, {{&L.o, &ob, false}});
    barrier(enc);
    m.elementwise(enc, "suc_add", h, B(m.ob), B(m.hmid), n * H);
    barrier(enc);
    BufRef gb = B(m.g), ub = B(m.u), db = B(m.db);
    if (L.moe && L.moe->streamed()) {   // router, host sync, fetch the routed experts, experts
      m.rms(enc, B(m.hmid), L.post_norm, B(m.xn), n, H);
      barrier(enc);
      MoeScratch& s = m.moe_scratch;
      const std::size_t K = c.top_k;
      if (s.eidx_b < n * K * 4) s.eidx = m.ctx->new_buffer(s.eidx_b = n * K * 4);
      if (s.scores_b < n * K * a) s.scores = m.ctx->new_buffer(s.scores_b = n * K * a);
      L.moe->encode_route(*m.ctx, enc, B(m.xn), B(s.eidx), B(s.scores), n, s);
      const MoeBlock* nxt = (m.next_moe[l] >= 0 && L.moe->stream()->options().prefetch)
                                ? m.layers[std::size_t(m.next_moe[l])].moe.get() : nullptr;
      if (nxt) {   // pre-gating guess: the next sparse layer's router on this layer's input
        if (m.pred_b < n * K * 4) {
          m.pred_b = n * K * 4;
          m.pred_eidx = m.ctx->new_buffer(m.pred_b);
          m.pred_scores = m.ctx->new_buffer(n * K * a);
        }
        nxt->encode_route(*m.ctx, enc, B(m.xn), B(m.pred_eidx), B(m.pred_scores), n, m.pred_scratch);
      }
      split_command_buffer(*m.ctx, m.cur_cb, enc, &m.split_gpu_s);
      L.moe->stream_in(static_cast<const uint32_t*>(s.eidx->contents()), n * K);
      if (nxt) nxt->stream_prefetch(static_cast<const uint32_t*>(m.pred_eidx->contents()), n * K);
      L.moe->encode_experts(*m.ctx, enc, B(m.xn), B(s.eidx), B(s.scores), db, n, s);
    } else if (L.moe) {   // Qwen3MoeSparseMoeBlock: plain RMSNorm (no SUC group reads it), then router + experts
      m.rms(enc, B(m.hmid), L.post_norm, B(m.xn), n, H);
      barrier(enc);
      L.moe->encode(*m.ctx, enc, B(m.xn), db, n, m.moe_scratch);
    } else {
      m.norm_project(enc, B(m.hmid), L.post_norm, B(m.xn), n, {{&L.gate, &gb, true}, {&L.up, &ub, true}}, &L.gu);
      barrier(enc);
      if (!m.swiglu_project(enc, gb, ub, B(m.a), n, L.down, db)) {
        m.elementwise(enc, "suc_swiglu", gb, ub, B(m.a), n * F);
        barrier(enc);
        m.project(enc, B(m.a), n, {{&L.down, &db, false}});
      }
    }
    barrier(enc);
    m.elementwise(enc, "suc_add", B(m.hmid), B(m.db), hout, n * H);
    barrier(enc);
  }
  if (lrows) {
    const std::size_t first = n - lrows;   // the last lrows rows
    m.rms(enc, B(m.hs, c.layers * hl + first * H * a), m.final_norm, B(m.xn), lrows, H);
    barrier(enc);
    for (std::size_t r = 0; r < lrows; ++r) m.matvec_row(enc, *m.head, B(m.xn, r * H * a), B(m.logits, r * c.vocab * a));
    barrier(enc);
    for (std::size_t r = 0; r < lrows; ++r) m.argmax(enc, r, next_tok ? BufRef{m.tok.get(), (pos + n) * 4} : BufRef{});
  }
}

Model::Model(Context& ctx, const MappedFile& mf, const ModelOptions& opt) : p_(std::make_unique<Impl>()) {
  Pool pool;
  Impl& m = *p_;
  m.ctx = &ctx;
  m.mf = &mf;
  m.opt = opt;
  m.share_rotations = opt.share_rotations;
  m.fuse_norm = opt.fuse_norm < 0 ? ctx.fuse_elementwise("norm") : opt.fuse_norm > 0;
  m.fuse_projections = opt.fuse_projections && opt.share_rotations;
  // the Python runtime fuses SwiGLU + rotation only in its grouped (default) plan (SucMLP), per the tuned choice
  m.fuse_swiglu = opt.fuse_swiglu < 0 ? (m.fuse_projections && ctx.fuse_elementwise("swiglu")) : opt.fuse_swiglu > 0;
  const Suc1File& f = mf.file();
  const Json* arch = f.directory().find("architecture");
  if (!arch || !arch->is_object()) throw FormatError("directory has no architecture object");
  m.cfg = ModelConfig::from_architecture(*arch);
  m.names = NameTable::for_family(m.cfg.model_type);
  const ModelConfig& c = m.cfg;
  const std::size_t H = c.hidden, D = c.head_dim, QD = c.heads * D, KD = c.kv_heads * D, F = c.intermediate;
  if (opt.max_context == 0) fail("max_context must be > 0");

  m.embed = m.mat(m.names.embed, c.vocab, H, false);
  if (c.tie_word_embeddings) {
    m.head = &m.embed;
  } else {
    m.lm_head = m.mat(m.names.lm_head, c.vocab, H, true);
    m.head = &m.lm_head;
  }
  m.final_norm = m.vec(m.names.final_norm, H);
  m.layers.resize(c.layers);
  for (std::size_t i = 0; i < c.layers; ++i) {
    const LayerNames n = m.names.layer(i);
    Impl::Layer& L = m.layers[i];
    L.q = m.mat(n.q, QD, H);
    L.k = m.mat(n.k, KD, H);
    L.v = m.mat(n.v, KD, H);
    L.o = m.mat(n.o, H, QD);
    if (c.sparse_layer(i)) {
      const std::string lp = m.names.prefix + std::to_string(i);
      L.moe = std::make_unique<MoeBlock>(MoeBlock::from_file(
          ctx, mf, lp, MoeConfig{H, c.num_experts, c.top_k, c.moe_intermediate, c.norm_topk_prob}, opt.act,
          opt.expert_stream ? StackLayout::Streamed : StackLayout::Auto));
      if (opt.expert_stream) L.moe->attach_stream(*opt.expert_stream);
    } else {
      L.gate = m.mat(n.gate, F, H);
      L.up = m.mat(n.up, F, H);
      L.down = m.mat(n.down, H, F);
    }
    L.in_norm = m.vec(n.input_norm, H);
    L.post_norm = m.vec(n.post_norm, H);
    L.q_norm = m.vec(n.q_norm, D);
    L.k_norm = m.vec(n.k_norm, D);
    if (m.fuse_projections) {
      L.qkv = m.make_group({&L.q, &L.k, &L.v});
      if (!L.moe) L.gu = m.make_group({&L.gate, &L.up});
    }
  }

  m.next_moe.assign(c.layers, -1);
  if (opt.expert_stream) {
    for (std::size_t i = c.layers, nxt = SIZE_MAX; i-- > 0;) {
      m.next_moe[i] = nxt == SIZE_MAX ? -1 : int(nxt);
      if (m.layers[i].moe) nxt = i;
    }
    opt.expert_stream->finalize();
    if (opt.expert_stream->min_max_distinct() < c.top_k) fail("expert stream: fewer unpinned slots than top_k");
  }

  // MLX conventions: rms eps as float; rope base passed as std::log2(float base); attention scale head_dim ** -0.5
  m.eps = float(c.rms_norm_eps);
  m.rope_base_log2 = std::log2(float(c.rope_theta));
  m.attn_scale = float(std::pow(double(D), -0.5));

  if (opt.chunk == 0) fail("chunk must be > 0");
  const std::size_t a = m.as() * opt.chunk;   // per-row buffers hold `chunk` rows
  uint64_t* ab = &m.arena_bytes;
  m.tok = m.alloc((opt.max_context + 1) * 4, ab);
  m.hs = m.alloc((c.layers + 1) * H * a, ab);
  m.hmid = m.alloc(H * a, ab);
  m.xn = m.alloc(H * a, ab);
  m.xr = m.alloc(std::max({H, F, QD}) * a, ab);
  m.q = m.alloc(QD * a, ab);
  m.k = m.alloc(KD * a, ab);
  m.att = m.alloc(QD * a, ab);
  m.ob = m.alloc(H * a, ab);
  m.db = m.alloc(H * a, ab);
  m.g = m.alloc(F * a, ab);
  m.u = m.alloc(F * a, ab);
  m.a = m.alloc(F * a, ab);
  m.yfused = m.alloc(std::max(QD + 2 * KD, 2 * F) * a, ab);
  m.logits = m.alloc(c.vocab * m.as(), ab);   // one row; grown to `chunk` rows on the first Logits::All chunk
  m.amax = m.alloc(4 * opt.chunk, ab);
  const std::size_t kv = c.layers * opt.max_context * KD * m.as();
  m.kcache = m.alloc(kv, &m.kv_bytes);
  m.vcache = m.alloc(kv, &m.kv_bytes);
}

Model::~Model() = default;

const ModelConfig& Model::config() const { return p_->cfg; }
std::size_t Model::position() const { return p_->pos; }
void Model::reset() { p_->pos = 0; }
double Model::last_gpu_seconds() const { return p_->gpu_s; }
double Model::last_encode_seconds() const { return p_->encode_s; }

uint32_t Model::step(uint32_t token, bool with_logits) {
  const auto r = forward({token}, with_logits ? Logits::Last : Logits::None);
  return r.empty() ? 0 : r[0];
}

std::vector<uint32_t> Model::forward(const std::vector<uint32_t>& tokens, Logits which) {
  Impl& m = *p_;
  const ModelConfig& c = m.cfg;
  if (tokens.empty()) fail("forward: no tokens");
  if (m.pos + tokens.size() > m.opt.max_context)
    fail("KV cache full (max_context " + std::to_string(m.opt.max_context) + ")");
  for (uint32_t t : tokens)
    if (t >= c.vocab) fail("token id " + std::to_string(t) + " >= vocab " + std::to_string(c.vocab));
  std::vector<uint32_t> out;
  // streamed experts: one call's distinct experts must fit the unpinned slots of every layer
  const std::size_t chunk = m.opt.expert_stream && c.top_k
                                ? std::max<std::size_t>(1, std::min(m.opt.chunk, m.opt.expert_stream->min_max_distinct() / c.top_k))
                                : m.opt.chunk;
  for (std::size_t s0 = 0; s0 < tokens.size(); s0 += chunk) {
    const std::size_t n = std::min(chunk, tokens.size() - s0);
    const bool last_chunk = s0 + n == tokens.size();
    const std::size_t lrows = which == Logits::All ? n : (which == Logits::Last && last_chunk) ? 1 : 0;
    Pool pool;
    const std::size_t a = m.as();
    const std::size_t pos = m.pos;
    if (lrows > m.logits_rows_cap) {
      m.arena_bytes += (m.opt.chunk - m.logits_rows_cap) * c.vocab * a;
      m.logits = m.ctx->new_buffer(m.opt.chunk * c.vocab * a);
      m.logits_rows_cap = m.opt.chunk;
    }
    std::memcpy(static_cast<uint32_t*>(m.tok->contents()) + pos, tokens.data() + s0, n * 4);

    const auto t_enc = std::chrono::steady_clock::now();
    MTL::CommandBuffer* cb = m.ctx->queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
    m.cur_cb = cb;
    m.split_gpu_s = 0.0;
    m.encode_chunk(enc, pos, n, lrows, false);
    cb = m.cur_cb;
    enc->endEncoding();
    m.encode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_enc).count();
    cb->commit();
    cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted) {
      auto* e = cb->error();
      fail("command buffer failed: " + std::string(e ? e->localizedDescription()->utf8String() : "?"));
    }
    m.gpu_s = m.split_gpu_s + cb->GPUEndTime() - cb->GPUStartTime();
    m.pos += n;
    m.rows = n;
    m.logit_rows = lrows;
    const auto* am = static_cast<const uint32_t*>(m.amax->contents());
    out.insert(out.end(), am, am + lrows);
  }
  return out;
}

std::vector<uint32_t> Model::generate(uint32_t first, std::size_t steps, std::size_t depth, GenerateStats* stats,
                                      std::optional<uint32_t> stop,
                                      const std::function<bool(uint32_t)>& on_token) {
  Impl& m = *p_;
  const ModelConfig& c = m.cfg;
  if (first >= c.vocab) fail("token id " + std::to_string(first) + " >= vocab " + std::to_string(c.vocab));
  if (m.pos + steps > m.opt.max_context) fail("KV cache full (max_context " + std::to_string(m.opt.max_context) + ")");
  if (depth == 0) fail("generate: depth must be > 0");
  if (m.opt.expert_stream) depth = 1;   // a layer's slots may only change once its last reader has completed
  auto* tok = static_cast<uint32_t*>(m.tok->contents());
  tok[m.pos] = first;
  struct InFlight {
    MTL::CommandBuffer* cb;
    std::size_t pos;
  };
  std::vector<InFlight> q;
  std::vector<uint32_t> out;
  const std::size_t start = m.pos;
  std::size_t next_pos = start;
  bool stopped = false;
  std::exception_ptr callback_error;
  const auto t0 = std::chrono::steady_clock::now();
  auto t_prev = t0;
  auto retire = [&]() {
    InFlight f = q.front();
    q.erase(q.begin());
    f.cb->waitUntilCompleted();
    if (f.cb->status() != MTL::CommandBufferStatusCompleted) {
      auto* e = f.cb->error();
      fail("command buffer failed: " + std::string(e ? e->localizedDescription()->utf8String() : "?"));
    }
    const auto now = std::chrono::steady_clock::now();
    if (stats) {
      stats->gpu_ms.push_back(1e3 * (f.cb->GPUEndTime() - f.cb->GPUStartTime()));
      stats->step_ms.push_back(1e3 * std::chrono::duration<double>(now - t_prev).count());
    }
    t_prev = now;
    f.cb->release();
    if (!stopped) {
      const uint32_t t = tok[f.pos + 1];
      out.push_back(t);
      if (stop && t == *stop) stopped = true;
      if (on_token && !stopped) {
        try {
          if (!on_token(t)) stopped = true;
        } catch (...) {
          callback_error = std::current_exception();
          stopped = true;
        }
      }
    }
  };
  while ((next_pos < start + steps && !stopped) || !q.empty()) {
    if (next_pos < start + steps && !stopped && q.size() < depth) {
      Pool pool;
      const auto te = std::chrono::steady_clock::now();
      MTL::CommandBuffer* cb = m.ctx->queue()->commandBuffer();
      MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
      m.cur_cb = cb;
      m.encode_chunk(enc, next_pos, 1, 1, true);   // argmax -> tok[next_pos + 1]
      cb = m.cur_cb;   // the last part (streamed layers commit and wait for the parts before it)
      cb->retain();
      enc->endEncoding();
      cb->commit();
      if (stats) stats->encode_ms.push_back(1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - te).count());
      q.push_back({cb, next_pos});
      ++next_pos;
      continue;
    }
    retire();
  }
  // positions computed beyond a stop token stay in the cache; the position counts what was returned
  m.pos = start + out.size();
  m.rows = 1;
  m.logit_rows = 0;
  if (stats) stats->total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (callback_error) std::rethrow_exception(callback_error);
  return out;
}

std::size_t Model::last_rows() const { return p_->rows; }

std::vector<float> Model::logits() const {
  std::vector<float> v(p_->logit_rows * p_->cfg.vocab);
  act_to_f32(p_->logits->contents(), v.data(), v.size(), p_->opt.act);
  return v;
}

std::vector<float> Model::hidden_states() const {
  // hs holds (layers + 1) slots of (n, hidden); return (n, layers + 1, hidden)
  const std::size_t n = p_->rows, L1 = p_->cfg.layers + 1, H = p_->cfg.hidden;
  std::vector<float> flat(L1 * n * H), v(L1 * n * H);
  act_to_f32(p_->hs->contents(), flat.data(), flat.size(), p_->opt.act);
  for (std::size_t r = 0; r < n; ++r)
    for (std::size_t l = 0; l < L1; ++l)
      std::memcpy(v.data() + (r * L1 + l) * H, flat.data() + (l * n + r) * H, H * 4);
  return v;
}

MemoryReport Model::memory() const {
  MemoryReport r;
  r.mapped_file = p_->mf->mapped_bytes();
  uint64_t copied = p_->copied_norm_bytes;
  auto add = [&](const Matrix& m) {
    if (m.kind == Matrix::Suc) copied += m.lin->copied_bytes();
  };
  for (const auto& L : p_->layers)
  {
    for (const Matrix* m : {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}) add(*m);
    if (L.moe) copied += L.moe->copied_bytes();
  }
  if (!p_->cfg.tie_word_embeddings) add(p_->lm_head);
  r.copied_weights = copied + p_->fused_copied_bytes;
  r.kv_cache = p_->kv_bytes;
  r.arena = p_->arena_bytes;
  uint64_t ws = 0;
  for (const Workspace& w : p_->ws) ws += w.xr_bytes + w.y32_bytes + w.y32b_bytes;
  r.workspace = ws + p_->moe_scratch.bytes();
  return r;
}

std::size_t Model::suc_projections() const {
  std::size_t n = 0;
  for (const auto& L : p_->layers)
    for (const Matrix* m : {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}) n += m->kind == Matrix::Suc;
  return n;
}

bool Model::norm_fused() const { return p_->fuse_norm; }
bool Model::swiglu_fused() const { return p_->fuse_swiglu; }
std::size_t Model::fused_groups() const { return p_->fused_parts; }

std::size_t Model::other_projections() const {
  std::size_t n = 0;
  for (const auto& L : p_->layers) n += L.moe ? 4 : 7;   // sparse layers: q, k, v, o (experts: moe_layers())
  return n - suc_projections();
}
std::size_t Model::moe_layers() const {
  std::size_t n = 0;
  for (const auto& L : p_->layers) n += L.moe != nullptr;
  return n;
}

ProcessMemory process_memory() {
  ProcessMemory pm;
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
    pm.phys_footprint = info.phys_footprint;
    pm.resident = info.resident_size;
    pm.resident_peak = info.resident_size_peak;
  }
  return pm;
}

uint64_t resident_bytes(const void* addr, std::size_t len) {
  const std::size_t page = std::size_t(::getpagesize());
  const uintptr_t a = reinterpret_cast<uintptr_t>(addr) / page * page;
  const std::size_t n = (reinterpret_cast<uintptr_t>(addr) + len - a + page - 1) / page;
  std::vector<char> v(n);
  if (::mincore(reinterpret_cast<void*>(a), n * page, v.data()) != 0) return 0;
  uint64_t r = 0;
  for (char c : v) r += (c & MINCORE_INCORE) ? page : 0;
  return r;
}

}  // namespace suc::metal
