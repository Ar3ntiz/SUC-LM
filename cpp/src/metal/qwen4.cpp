// qwen4_exp model graph on the Metal backend (see suc/qwen4.hpp). Kernels: cpp/metal/suc_qwen4_kernels.metal (the
// qwen4_exp ops), cpp/metal/suc_model_kernels.metal (embedding gather, MLX-order gemv, argmax, sdpa_vector, SwiGLU,
// strided copy) and the SUC projection / expert-stack kernels of metal::Linear / metal::SwitchLinear.
#include "suc/qwen4.hpp"
#include "suc/expert_stream.hpp"
#include "suc/switch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>

#include <unistd.h>

#include "suc/bits.hpp"
#include "suc/json.hpp"
#include "suc/parallel.hpp"

namespace suc::metal {

namespace {

struct Pool {
  NS::AutoreleasePool* p = NS::AutoreleasePool::alloc()->init();
  ~Pool() { p->release(); }
};

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("qwen4: " + msg); }

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
    default: throw FormatError("unsupported floating dtype " + std::string(dtype_name(t)));
  }
}

void barrier(MTL::ComputeCommandEncoder* enc) { enc->memoryBarrier(MTL::BarrierScopeBuffers); }
void set(MTL::ComputeCommandEncoder* enc, BufRef r, int i) { enc->setBuffer(r.buffer, r.offset, i); }
template <class V>
void bytes(MTL::ComputeCommandEncoder* enc, const V& v, int i) { enc->setBytes(&v, sizeof v, i); }

const Json& text_config(const Json& arch) {
  const Json* t = arch.find("text_config");
  return t && t->is_object() ? *t : arch;
}

std::size_t get_size(const Json& a, const char* key) {
  const Json* v = a.find(key);
  if (!v || !v->is_int()) throw FormatError(std::string("qwen4_exp architecture: missing integer '") + key + "'");
  return std::size_t(v->as_uint());
}
std::size_t get_size_or(const Json& a, const char* key, std::size_t dflt) {
  const Json* v = a.find(key);
  return v && v->is_int() ? std::size_t(v->as_uint()) : dflt;
}

float round_to_act(float x, Act a) {
  if (a == Act::BF16) return bf16_to_f32(f32_to_bf16_bits(x));
  if (a == Act::F16) return f16_to_f32(f32_to_f16_bits(x));
  return x;
}

// splitmix64 / layer multipliers / primes of mlx_qwen4_exp (HF _build_layer_multipliers, _find_nth_prime_after)
uint64_t splitmix64(uint64_t v) {
  v += 0x9E3779B97F4A7C15ull;
  v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
  v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
  return v ^ (v >> 31);
}
bool is_prime(uint64_t v) {
  if (v < 2) return false;
  if (v % 2 == 0) return v == 2;
  for (uint64_t d = 3; d * d <= v; d += 2)
    if (v % d == 0) return false;
  return true;
}

}  // namespace

// ------------------------------------------------------------------------------------------ config

int Qwen4Config::ple_index(std::size_t layer) const {
  const auto it = std::find(ple_layer_ids.begin(), ple_layer_ids.end(), layer + 1);
  return it == ple_layer_ids.end() ? -1 : int(it - ple_layer_ids.begin());
}

bool Qwen4Config::is_qwen4(const Json& arch) {
  const Json* mt = arch.find("model_type");
  return mt && mt->is_string() && (mt->as_string() == "qwen4_exp" || mt->as_string() == "qwen3_5");
}

Qwen4Config Qwen4Config::from_architecture(const Json& arch) {
  if (!is_qwen4(arch)) throw FormatError("architecture: model_type is not qwen4_exp / qwen3_5");
  const Json& a = text_config(arch);
  Qwen4Config c;
  c.dense = arch.find("model_type")->as_string() == "qwen3_5";
  c.hidden = get_size(a, "hidden_size");
  c.layers = get_size(a, "num_hidden_layers");
  c.heads = get_size(a, "num_attention_heads");
  c.kv_heads = get_size(a, "num_key_value_heads");
  c.head_dim = get_size(a, "head_dim");
  c.vocab = get_size(a, "vocab_size");
  if (const Json* v = a.find("rms_norm_eps")) c.rms_norm_eps = v->as_double();
  // layer types (default: every full_attention_interval-th layer is full attention)
  const std::size_t interval = get_size_or(a, "full_attention_interval", 4);
  if (const Json* lt = a.find("layer_types"); lt && lt->is_array()) {
    if (lt->size() != c.layers) throw FormatError("qwen4_exp: layer_types length != num_hidden_layers");
    for (const Json& t : lt->as_array()) {
      const std::string s = t.as_string();
      if (s == "linear_attention") c.full_attention.push_back(false);
      else if (s == "full_attention" || s == "indexed_attention") c.full_attention.push_back(true);
      else throw FormatError("qwen4_exp: unsupported layer type '" + s + "'");
    }
  } else {
    for (std::size_t i = 0; i < c.layers; ++i) c.full_attention.push_back((i + 1) % interval == 0);
  }
  c.lin_v_heads = get_size(a, "linear_num_value_heads");
  c.lin_k_heads = get_size(a, "linear_num_key_heads");
  c.lin_k_dim = get_size(a, "linear_key_head_dim");
  c.lin_v_dim = get_size(a, "linear_value_head_dim");
  c.conv_kernel = get_size_or(a, "linear_conv_kernel_dim", 4);
  if (c.dense) {
    if (get_size_or(a, "num_experts", 0) != 0) throw FormatError("qwen3_5: MoE variants are not supported");
    if (const Json* v = a.find("attn_output_gate"); v && !v->is_null() && !v->as_bool())
      throw FormatError("qwen3_5: attn_output_gate false is not supported");
    c.intermediate = get_size(a, "intermediate_size");
    c.hc = 1;
  } else {
    c.num_experts = get_size(a, "num_experts");
    c.top_k = get_size(a, "num_experts_per_tok");
    c.moe_intermediate = get_size(a, "moe_intermediate_size");
    c.shared_intermediate = get_size(a, "shared_expert_intermediate_size");
    if (const Json* v = a.find("norm_topk_prob"); v && !v->is_null()) c.norm_topk_prob = v->as_bool();
    c.hc = get_size_or(a, "hc_count", 4);
    c.hc_lowrank = get_size(a, "hc_lowrank");
  }
  if (const Json* v = a.find("ple_layer_ids"); !c.dense && v && v->is_array())
    for (const Json& e : v->as_array()) c.ple_layer_ids.push_back(std::size_t(e.as_uint()));
  std::sort(c.ple_layer_ids.begin(), c.ple_layer_ids.end());
  c.ple_layer_ids.erase(std::unique(c.ple_layer_ids.begin(), c.ple_layer_ids.end()), c.ple_layer_ids.end());
  c.ple_dim = get_size_or(a, "ple_embed_dim", c.hidden);
  if (const Json* v = a.find("ple_embed_dim"); v && v->is_null()) c.ple_dim = c.hidden;
  c.ple_conv_kernel = get_size_or(a, "ple_conv_kernel_size", 4);
  c.ngram_size = get_size_or(a, "ngram_size", 3);
  c.heads_per_ngram = get_size_or(a, "heads_per_ngram", 8);
  c.split_ngram_parts = get_size_or(a, "split_ngram_parts", 512);
  c.seed = get_size_or(a, "seed", 1234);
  c.ngram_vocab_size_base = get_size_or(a, "ngram_vocab_size_base", 20000000);
  c.ngram_divisible_by = get_size_or(a, "make_ngram_vocab_size_divisible_by", 128);
  if (const Json* v = a.find("indexer_n_heads"); !c.dense && v && !v->is_null()) {
    c.indexer = true;
    c.idx_heads = get_size(a, "indexer_n_heads");
    c.idx_dim = get_size(a, "indexer_head_dim");
    c.idx_budget = get_size(a, "indexer_budget");
    c.idx_ratio = get_size(a, "indexer_compress_ratio");
    if (get_size(a, "indexer_kv_heads") != 1) throw FormatError("qwen4_exp: indexer_kv_heads must be 1");
    if (c.idx_ratio == 0 || c.idx_budget < c.idx_ratio) throw FormatError("qwen4_exp: indexer budget / ratio");
  }
  std::string act = "silu";
  if (const Json* v = a.find("hidden_act"); v && v->is_string()) act = v->as_string();
  if (act != "silu") throw FormatError("qwen4_exp: hidden_act must be silu");
  std::string gate = act;
  if (const Json* v = a.find("output_gate_type"); v && v->is_string()) gate = v->as_string();
  if (gate == "swish") gate = "silu";
  if (gate != "sigmoid" && gate != "silu") throw FormatError("qwen4_exp: unsupported output gate '" + gate + "'");
  c.gate_sigmoid = gate == "sigmoid";
  const Json* e = a.find("eos_token_id");
  if (!e || e->is_null()) e = arch.find("eos_token_id");
  if ((!e || e->is_null()) && !c.dense) throw FormatError("qwen4_exp: no eos_token_id (the n-gram hashing needs it)");
  if (e && !e->is_null()) c.eos = uint32_t(e->is_array() ? e->at(0).as_uint() : e->as_uint());
  double prf = 1.0;
  if (const Json* rp = a.find("rope_parameters"); rp && rp->is_object()) {
    if (const Json* v = rp->find("rope_theta")) c.rope_theta = v->as_double();
    if (const Json* v = rp->find("partial_rotary_factor")) prf = v->as_double();
  } else {
    if (const Json* v = a.find("rope_theta")) c.rope_theta = v->as_double();
    if (const Json* v = a.find("partial_rotary_factor")) prf = v->as_double();
  }
  c.rotary_dim = std::size_t(double(c.head_dim) * prf);
  if (const Json* v = a.find("attention_bias"); v && !v->is_null() && v->as_bool())
    throw FormatError("qwen4_exp: attention_bias is not supported");
  bool tie = false;
  if (const Json* v = arch.find("tie_word_embeddings"); v && !v->is_null()) tie = tie || v->as_bool();
  if (const Json* v = a.find("tie_word_embeddings"); v && !v->is_null()) tie = tie || v->as_bool();
  if (tie) throw FormatError("qwen4_exp: tied embeddings are not supported");
  if (const Json* v = arch.find("torch_dtype"); v && v->is_string()) c.torch_dtype = v->as_string();
  else if (const Json* w = a.find("torch_dtype"); w && w->is_string()) c.torch_dtype = w->as_string();
  for (std::size_t id : c.ple_layer_ids)
    if (id == 0 || id > c.layers || c.full_attention[id - 1]) throw FormatError("qwen4_exp: PLE only on linear-attention layers");
  if (c.heads % c.kv_heads) throw FormatError("qwen4_exp: heads not a multiple of kv heads");
  if (c.lin_v_heads % c.lin_k_heads) throw FormatError("qwen4_exp: linear value heads not a multiple of key heads");
  if (c.lin_k_dim % 32 || c.lin_v_dim % 4) throw FormatError("qwen4_exp: linear key dim % 32 / value dim % 4");
  if (c.rotary_dim % 2 || c.rotary_dim > c.head_dim || c.head_dim % 2) throw FormatError("qwen4_exp: rotary dims");
  if (c.indexer && (c.rotary_dim > c.idx_dim || c.idx_dim % 2 || c.idx_dim > 1024))
    throw FormatError("qwen4_exp: indexer head dim");
  if (c.ple_dim % c.ngram_heads()) throw FormatError("qwen4_exp: ple_embed_dim % n-gram heads");
  if (c.head_dim > 1024 || c.hidden > 4096 * 4) throw FormatError("qwen4_exp: head_dim / hidden too large");
  return c;
}

// ------------------------------------------------------------------------------------------ weights

namespace {

// (out, in) matrix: SUC codes (metal::Linear), affine codes, or a raw dense matrix (all read in place).
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
};

// A raw vector / tensor read in place (any float dtype; the kernels take its type as a template argument).
struct Vec {
  BufRef ref;
  DType dtype = DType::BF16;
  const char* msl() const { return msl_of(dtype); }
};

// One n-gram table shard (raw rows or row-affine records), as pointers into the mapping.
struct Shard {
  uint64_t rows = 0;
  bool affine = false;
  const uint8_t* raw = nullptr;   // raw: (rows, dim) of rdtype
  DType rdtype = DType::BF16;
  const uint8_t* words = nullptr; // affine: (rows, wpr) u32
  const uint8_t* scales = nullptr;
  const uint8_t* biases = nullptr;
  DType sdtype = DType::F16;
  unsigned bits = 0;
  std::size_t wpr = 0;
};

struct NgramLayer {
  std::size_t layer = 0, index = 0, dim = 0;
  std::vector<Shard> shards;
  std::vector<uint64_t> starts;   // cumulative rows
  std::vector<uint64_t> mult, sizes, offsets;
  std::vector<uint64_t> ctx;      // the previous ngram_size - 1 ids
  uint64_t table_bytes = 0;
  bool quantized = false;
};

}  // namespace

struct Qwen4Model::Impl {
  Context* ctx = nullptr;
  const MappedFile* mf = nullptr;
  Qwen4Options opt;
  Qwen4Config cfg;

  struct Hyper {
    Vec norm;
    Mat down, up, inject;
    bool combine = true;
  };
  struct LinAttn {
    Mat qkv, z, a, b, out;
    Vec conv, a_log, dt_bias, norm;
    BufPtr conv_state, rec_state;
  };
  struct Attn {
    Mat q, k, v, o, iqk;
    Vec q_norm, k_norm, iq_norm, ik_norm;
    BufPtr kcache, vcache, ikeys, bkeys;
  };
  struct Ple {
    Mat key, value;
    Vec norm_key, norm_query, norm_conv, conv;
    BufPtr state;
    int index = -1;
  };
  struct Layer {
    bool full = false;
    Hyper ahc, mhc;
    LinAttn lin;
    Attn att;
    std::unique_ptr<Ple> ple;
    Mat router, sh_gate, sh_up, sh_down, sh_router;
    std::unique_ptr<MoeBlock> moe;
    Vec in_norm, post_norm;   // dense (qwen3_5)
    Mat mlp_gate, mlp_up, mlp_down;
  };
  Mat embed, head;
  Hyper final_mixer;
  Vec final_norm;             // dense (qwen3_5)
  std::vector<Layer> layers;
  std::vector<NgramLayer> ngram;
  MoeScratch moe_scratch;
  Workspace ws[6];
  // expert streaming: the command buffer being encoded (replaced at each layer's host sync), GPU time of the
  // finished parts, prediction buffers for prefetch
  MTL::CommandBuffer* cur_cb = nullptr;
  double split_gpu_s = 0.0;
  // $SUC_Q4_PROFILE=1: batched-prefill chunks split into one command buffer per stage, GPU time per stage label
  // printed to stderr after forward() (the splits serialise the stages: for the breakdown, not for speed).
  bool prof = false;
  const char* prof_label = nullptr;
  std::map<std::string, std::pair<double, std::size_t>> prof_acc;
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
    auto& e = prof_acc[prof_label];
    e.first += g;
    ++e.second;
  }
  BufPtr pgates, peidx, pscores;

  // arena
  BufPtr tok, ple_e, hs, xa, xn, dbuf, sbuf, up, inj, u, f;
  BufPtr qkv, z, av, bv, conv_out, qn, kn, gdec, beta, yd, yn;
  BufPtr qg, q, kb, att, iqk, iq, scores, sel, sc;
  BufPtr gates, eidx, escores, ymoe, shg, shu, shh, sh, sgate;
  BufPtr kraw, kk, pv, qq, pg, gn;
  BufPtr logits, amax;
  std::size_t pos = 0, max_blocks = 0;
  bool have_logits = false;
  double gpu_s = 0.0, encode_s = 0.0, ngram_s = 0.0;
  uint64_t arena_bytes = 0, kv_bytes = 0, state_bytes = 0, copied_bytes = 0;
  float eps = 0.0f, rope_base_log2 = 0.0f, attn_scale = 0.0f, q_scale = 0.0f, k_scale = 0.0f, ple_rsh = 0.0f,
        ple_floor = 0.0f;

  std::size_t as() const { return act_size(opt.act); }
  const char* T() const { return act_msl(opt.act); }
  BufPtr alloc(std::size_t bytes, uint64_t* counter) {
    if (counter) *counter += bytes;
    return ctx->new_buffer(std::max<std::size_t>(bytes, 16));
  }
  static BufRef R(const BufPtr& b, std::size_t off = 0) { return {b.get(), off}; }

  const TensorDesc& tensor(const std::string& name) const { return mf->file().tensor(name); }

  Vec vec(const std::string& name, std::size_t n) {
    const TensorDesc& t = tensor(name);
    if (t.representation_type != "raw" && t.representation_type != "bf16")
      throw FormatError(name + ": expected a raw tensor");
    const ArrayRecord& r = t.arrays.at(0);
    if (r.logical_numel() != n) throw FormatError(name + ": " + std::to_string(r.logical_numel()) + " elements, expected " + std::to_string(n));
    Vec v{mf->at(r), r.dtype};
    v.msl();
    return v;
  }

  Mat mat(const std::string& name, std::size_t out, std::size_t in, bool allow_suc = true) {
    const TensorDesc& t = tensor(name);
    Mat m;
    m.name = name;
    if (t.shape.size() != 2 || t.shape[0] != out || t.shape[1] != in)
      throw FormatError(name + ": unexpected shape, expected (" + std::to_string(out) + ", " + std::to_string(in) + ")");
    m.out = out;
    m.in = in;
    const std::string& rep = t.representation_type;
    if (rep == "vq" || rep == "additive" || rep == "tcq") {
      if (!allow_suc) throw FormatError(name + ": SUC codes cannot serve an embedding gather");
      m.kind = Mat::Suc;
      m.lin = std::make_unique<Linear>(Linear::from_file(*ctx, *mf, t));
      copied_bytes += m.lin->copied_bytes();
      return m;
    }
    if (t.has_residual()) throw FormatError(name + ": residual on a non-SUC base is not supported");
    if (rep == "affine") {
      const Json& p = *t.params;
      m.kind = Mat::Affine;
      m.bits = unsigned(p["bits"].as_uint());
      m.group = std::size_t(p["group_size"].as_uint());
      const ArrayRecord& qi = t.array("indices");
      const ArrayRecord& sc = t.array("scales");
      const ArrayRecord& bi = t.array("biases");
      m.words = words_per_row(in, m.bits);
      if (!qi.packed || qi.bits != m.bits || qi.shape != std::vector<uint64_t>{out, m.words})
        throw FormatError(name + ": affine index layout");
      if (in % m.group) throw FormatError(name + ": in_features must be a multiple of group_size");
      if (sc.dtype != bi.dtype || sc.shape != std::vector<uint64_t>{out, in / m.group} || sc.shape != bi.shape)
        throw FormatError(name + ": affine scales / biases");
      m.sdtype = sc.dtype;
      msl_of(m.sdtype);
      m.packed = mf->at(qi);
      m.scales = mf->at(sc);
      m.biases = mf->at(bi);
      return m;
    }
    if (rep == "raw" || rep == "bf16") {
      const ArrayRecord& w = t.arrays.at(0);
      m.kind = Mat::Dense;
      m.wdtype = w.dtype;
      msl_of(m.wdtype);
      m.w = mf->at(w);
      return m;
    }
    throw FormatError(name + ": representation '" + rep + "' has no qwen4_exp kernel");
  }

  Hyper hyper(const std::string& p, bool combine) {
    const std::size_t W = cfg.stream(), Lr = cfg.hc_lowrank;
    Hyper h;
    h.combine = combine;
    h.norm = vec(p + ".hc_norm.weight", W);
    h.down = mat(p + ".input_mix_weight_down.weight", Lr, W);
    h.up = mat(p + ".input_mix_weight_up.weight", W, Lr);
    if (combine) h.inject = mat(p + ".block_inject_weight.weight", cfg.hc, W);
    return h;
  }

  // ---- encoders
  // y = x Wᵀ for one row (SUC: metal::Linear with workspace ws[wsi]; affine / dense: MLX gemv order).
  void proj(MTL::ComputeCommandEncoder* enc, const Mat& m, BufRef x, BufRef y, int wsi = 0) {
    if (m.kind == Mat::Suc) {
      m.lin->encode(*ctx, enc, x, y, 1, opt.act, ws[wsi], Path::Auto);
      return;
    }
    const int sn = m.in <= 64 ? 4 : 32;
    const int bn = (m.in > 64 && m.in >= 16 * m.out) ? 8 : 1;
    const int sm = 32 / sn;
    if (m.kind == Mat::Affine) {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{1001, {int64_t(m.in), int64_t(m.out), m.bits, int64_t(m.group), int64_t(m.sdtype), int64_t(opt.act)}}, [&]() {
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
      enc->setComputePipelineState(ctx->pipeline(PsoKey{1002, {int64_t(m.in), int64_t(m.out), int64_t(m.wdtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{
            "suc_moe_gemv", join({T(), msl_of(m.wdtype), I(long(m.in)), I(long(m.out)), I(sn), "4", I(bn)})};
      }));
      set(enc, x, 0);
      set(enc, m.w, 1);
      set(enc, y, 2);
    }
    enc->dispatchThreadgroups(MTL::Size((m.out + sm - 1) / sm, 1, 1), MTL::Size(32 * bn, 1, 1));
  }

  static std::size_t rms_tpg(std::size_t n) { return n <= 4096 ? 32 * (((n + 3) / 4 + 31) / 32) : 1024; }

  // rows of N: x + r·xs -> y + r·ys; weight + (r % wmod)·N. mode 0: (1 + w) norm; 1: weightless · scale; 2: gated.
  // rpt > 0: input row r at (r / rpt) · ts + (r % rpt) · xs (rows of several tokens in one launch)
  void rms(MTL::ComputeCommandEncoder* enc, int mode, BufRef x, const Vec* w, BufRef y, std::size_t rows, std::size_t N,
           std::size_t xs, std::size_t ys, std::size_t wmod, float scale = 1.0f, BufRef z = {}, float eps_override = -1.0f,
           std::size_t rpt = 0, std::size_t ts = 0) {
    const std::size_t tpg = rms_tpg(N);
    const DType wd = w ? w->dtype : DType::BF16;
    const int gate = cfg.gate_sigmoid ? 0 : 1;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1003, {int64_t(N), int64_t(tpg), mode, gate, int64_t(wd), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_rms", join({T(), msl_of(wd), I(long(N)), I(long(tpg)), I(mode), I(gate)})};
    }));
    set(enc, x, 0);
    set(enc, w ? w->ref : ctx->dummy(), 1);
    set(enc, y, 2);
    set(enc, z ? z : ctx->dummy(), 3);
    const float e = eps_override >= 0.0f ? eps_override : eps;
    bytes(enc, e, 4);
    bytes(enc, scale, 5);
    const uint32_t xs32 = uint32_t(xs), ys32 = uint32_t(ys), wm = uint32_t(std::max<std::size_t>(wmod, 1));
    bytes(enc, xs32, 6);
    bytes(enc, ys32, 7);
    bytes(enc, wm, 8);
    const uint32_t rpt32 = uint32_t(rpt), ts32 = uint32_t(ts);
    bytes(enc, rpt32, 9);
    bytes(enc, ts32, 10);
    enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(tpg, 1, 1));
  }

  // u (H) [and inj (hc)] of a gated residual on the stream x (W).
  void hyper_mix(MTL::ComputeCommandEncoder* enc, const Hyper& h, BufRef x) {
    const std::size_t H = cfg.hidden, W = cfg.stream(), HC = cfg.hc;
    rms(enc, 0, x, &h.norm, R(xn), HC, H, H, H, HC);
    barrier(enc);
    proj(enc, h.down, R(xn), R(dbuf));
    if (h.combine) proj(enc, h.inject, R(xn), R(inj));
    barrier(enc);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1004, {int64_t(HC), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_silu_div", join({T(), I(long(HC))})};
    }));
    set(enc, R(dbuf), 0);
    set(enc, R(sbuf), 1);
    const uint32_t n = uint32_t(cfg.hc_lowrank);
    bytes(enc, n, 2);
    enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(std::min<uint32_t>(256, n), 1, 1));
    barrier(enc);
    proj(enc, h.up, R(sbuf), R(up));
    barrier(enc);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1005, {int64_t(H), int64_t(HC), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_hc_mix", join({T(), I(long(H)), I(long(HC))})};
    }));
    set(enc, R(up), 0);
    set(enc, R(xn), 1);
    set(enc, R(u), 2);
    enc->dispatchThreads(MTL::Size(H, 1, 1), MTL::Size(std::min<std::size_t>(256, H), 1, 1));
    barrier(enc);
    (void)W;
  }

  void inject(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y) {
    const std::size_t H = cfg.hidden, HC = cfg.hc;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1006, {int64_t(H), int64_t(HC), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_inject", join({T(), I(long(H)), I(long(HC))})};
    }));
    set(enc, x, 0);
    set(enc, R(f), 1);
    set(enc, R(inj), 2);
    set(enc, y, 3);
    enc->dispatchThreads(MTL::Size(H, HC, 1), MTL::Size(std::min<std::size_t>(256, H), 1, 1));
    barrier(enc);
  }

  // x_out = x + PLE(x, e) (the conv kernel writes the sum)
  void ple(MTL::ComputeCommandEncoder* enc, const Ple& P, BufRef x, BufRef e, BufRef xout) {
    const std::size_t H = cfg.hidden, W = cfg.stream(), HC = cfg.hc;
    proj(enc, P.key, e, R(kraw), 0);
    proj(enc, P.value, e, R(pv), 1);
    rms(enc, 0, x, &P.norm_query, R(qq), HC, H, H, H, HC);
    barrier(enc);
    rms(enc, 0, R(kraw), &P.norm_key, R(kk), HC, H, H, H, HC);
    barrier(enc);
    // MLX's bf16 row-reduce threads for (hc, H) rows (measured, MLX 0.32): <= 512: one simdgroup; <= 1024: 128; else 1024
    const std::size_t tpg = H <= 512 ? 32 : H <= 1024 ? 128 : 1024;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1007, {int64_t(H), int64_t(tpg), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_ple_gate", join({T(), I(long(H)), I(long(tpg))})};
    }));
    set(enc, R(kk), 0);
    set(enc, R(qq), 1);
    set(enc, R(pv), 2);
    set(enc, R(pg), 3);
    bytes(enc, ple_rsh, 4);
    bytes(enc, ple_floor, 5);
    enc->dispatchThreadgroups(MTL::Size(HC, 1, 1), MTL::Size(tpg, 1, 1));
    barrier(enc);
    rms(enc, 0, R(pg), &P.norm_conv, R(gn), HC, H, H, H, HC);
    barrier(enc);
    conv(enc, 1, R(gn), R(P.state), P.conv, xout, W, cfg.ple_conv_kernel, cfg.ngram_size, R(pg), x);
    barrier(enc);
  }

  void conv(MTL::ComputeCommandEncoder* enc, int mode, BufRef xnew, BufRef state, const Vec& w, BufRef y, std::size_t C,
            std::size_t K, std::size_t dil, BufRef g = {}, BufRef xin = {}) {
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1008, {int64_t(K), int64_t(dil), mode, int64_t(w.dtype), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_conv", join({T(), w.msl(), I(long(K)), I(long(dil)), I(mode)})};
    }));
    set(enc, xnew, 0);
    set(enc, state, 1);
    set(enc, w.ref, 2);
    set(enc, y, 3);
    set(enc, g ? g : ctx->dummy(), 4);
    set(enc, xin ? xin : ctx->dummy(), 5);
    const uint32_t c32 = uint32_t(C);
    bytes(enc, c32, 6);
    enc->dispatchThreads(MTL::Size(C, 1, 1), MTL::Size(std::min<std::size_t>(256, C), 1, 1));
  }

  // f = Gated DeltaNet(u)
  void linear_attn(MTL::ComputeCommandEncoder* enc, const LinAttn& L) {
    const std::size_t Hk = cfg.lin_k_heads, Hv = cfg.lin_v_heads, Dk = cfg.lin_k_dim, Dv = cfg.lin_v_dim;
    const std::size_t key_dim = Hk * Dk, conv_dim = 2 * key_dim + Hv * Dv;
    proj(enc, L.qkv, R(u), R(qkv), 0);
    proj(enc, L.z, R(u), R(z), 1);
    proj(enc, L.a, R(u), R(av), 2);
    proj(enc, L.b, R(u), R(bv), 3);
    barrier(enc);
    linear_attn_core(enc, L);
    proj(enc, L.out, R(yn), R(f), 0);
    barrier(enc);
    (void)conv_dim;
  }

  // The Gated DeltaNet step after its input projections (qkv, z, av, bv) -> yn (gated RMSNorm output).
  void linear_attn_core(MTL::ComputeCommandEncoder* enc, const LinAttn& L) {
    const std::size_t Hk = cfg.lin_k_heads, Hv = cfg.lin_v_heads, Dk = cfg.lin_k_dim, Dv = cfg.lin_v_dim;
    const std::size_t key_dim = Hk * Dk, conv_dim = 2 * key_dim + Hv * Dv;
    conv(enc, 0, R(qkv), R(L.conv_state), L.conv, R(conv_out), conv_dim, cfg.conv_kernel, 1);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1009, {int64_t(L.a_log.dtype), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_delta_prep", join({T(), L.a_log.msl()})};
    }));
    if (L.a_log.dtype != L.dt_bias.dtype) fail("A_log / dt_bias dtypes differ");
    set(enc, R(av), 0);
    set(enc, R(bv), 1);
    set(enc, L.a_log.ref, 2);
    set(enc, L.dt_bias.ref, 3);
    set(enc, R(gdec), 4);
    set(enc, R(beta), 5);
    const uint32_t nh = uint32_t(Hv);
    bytes(enc, nh, 6);
    enc->dispatchThreads(MTL::Size(Hv, 1, 1), MTL::Size(std::min<std::size_t>(256, Hv), 1, 1));
    barrier(enc);
    // l2norm with eps 1e-6 on the sum of squares (HF / FLA): the kernel adds eps to the mean, hence 1e-6 / Dk.
    rms(enc, 1, R(conv_out), nullptr, R(qn), Hk, Dk, Dk, Dk, 1, q_scale, {}, 1e-6f / float(Dk));
    rms(enc, 1, R(conv_out, key_dim * as()), nullptr, R(kn), Hk, Dk, Dk, Dk, 1, k_scale, {}, 1e-6f / float(Dk));
    barrier(enc);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1010, {int64_t(Dk), int64_t(Dv), int64_t(Hk), int64_t(Hv), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_gated_delta", join({T(), I(long(Dk)), I(long(Dv)), I(long(Hk)), I(long(Hv))})};
    }));
    set(enc, R(qn), 0);
    set(enc, R(kn), 1);
    set(enc, R(conv_out, 2 * key_dim * as()), 2);
    set(enc, R(gdec), 3);
    set(enc, R(beta), 4);
    set(enc, R(L.rec_state), 5);
    set(enc, R(yd), 6);
    enc->dispatchThreads(MTL::Size(32, Dv, Hv), MTL::Size(32, 4, 1));
    barrier(enc);
    rms(enc, 2, R(yd), &L.norm, R(yn), Hv, Dv, Dv, Dv, 1, 1.0f, R(z));
    barrier(enc);
  }

  // rpp > 0: row r at position offset + r / rpp (chunk rows)
  void rope(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t rows, std::size_t D, std::size_t xs,
            std::size_t ys, int offset, uint32_t rpp = 0) {
    const std::size_t rot = cfg.rotary_dim;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1011, {int64_t(D), int64_t(rot), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_rope", join({T(), I(long(D)), I(long(rot))})};
    }));
    set(enc, x, 0);
    set(enc, y, 1);
    bytes(enc, offset, 2);
    bytes(enc, rope_base_log2, 3);
    const uint32_t xs32 = uint32_t(xs), ys32 = uint32_t(ys);
    bytes(enc, xs32, 4);
    bytes(enc, ys32, 5);
    bytes(enc, rpp, 6);
    enc->dispatchThreads(MTL::Size(D / 2, rows, 1), MTL::Size(D / 2, 1, 1));
  }

  // f = gated attention(u) at position p (the K / V caches hold rows 0 .. p - 1)
  void full_attn(MTL::ComputeCommandEncoder* enc, const Attn& A, std::size_t p) {
    const std::size_t KD = cfg.kv_heads * cfg.head_dim, a = as();
    if (cfg.indexer) proj(enc, A.iqk, R(u), R(iqk), 0);
    proj(enc, A.q, R(u), R(qg), 1);
    proj(enc, A.k, R(u), R(kb), 2);
    proj(enc, A.v, R(u), R(A.vcache, p * KD * a), 3);
    barrier(enc);
    full_attn_core(enc, A, p);
    proj(enc, A.o, R(att), R(f), 0);
    barrier(enc);
  }

  // Gated attention at position p after its input projections (iqk, qg, kb; v already in the cache) -> att (gated).
  void full_attn_core(MTL::ComputeCommandEncoder* enc, const Attn& A, std::size_t p) {
    const std::size_t Hq = cfg.heads, Hkv = cfg.kv_heads, D = cfg.head_dim, a = as();
    const std::size_t KD = Hkv * D;
    const std::size_t t_kv = p + 1;
    const std::size_t d = cfg.idx_dim, nh = cfg.idx_heads, ratio = cfg.idx_ratio;
    const std::size_t nblocks = cfg.indexer ? t_kv / ratio : 0;
    const bool sparse = cfg.indexer && nblocks > cfg.idx_budget / ratio;
    const bool block_done = cfg.indexer && t_kv % ratio == 0;
    rms(enc, 0, R(qg), &A.q_norm, R(q), Hq, D, 2 * D, D, 1);
    rms(enc, 0, R(kb), &A.k_norm, R(kn), Hkv, D, D, D, 1);
    if (cfg.indexer) {
      encode_copy_cols(*ctx, enc, R(iqk, nh * d * a), R(A.ikeys, p * d * a), 1, d, d, d, opt.act);
      if (sparse) rms(enc, 0, R(iqk), &A.iq_norm, R(iq), nh, d, d, d, 1);
    }
    barrier(enc);
    rope(enc, R(q), R(q), Hq, D, D, D, int(p));
    rope(enc, R(kn), R(A.kcache, p * KD * a), Hkv, D, D, D, int(p));
    if (sparse) rope(enc, R(iq), R(iq), nh, d, d, d, int(p));
    if (block_done) {
      const uint32_t b = uint32_t(t_kv / ratio - 1), r32 = uint32_t(ratio);
      const float base = float(cfg.rope_theta);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{1012, {int64_t(d), int64_t(cfg.rotary_dim), int64_t(A.ik_norm.dtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"q4_block_key", join({T(), A.ik_norm.msl(), I(long(d)), I(long(cfg.rotary_dim))})};
      }));
      set(enc, R(A.ikeys), 0);
      set(enc, A.ik_norm.ref, 1);
      set(enc, R(A.bkeys), 2);
      bytes(enc, b, 3);
      bytes(enc, r32, 4);
      bytes(enc, eps, 5);
      bytes(enc, base, 6);
      enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(d, 1, 1));
    }
    barrier(enc);
    uint32_t nb_mask = 0;
    if (sparse) {
      const uint32_t nb32 = uint32_t(nblocks), k32 = uint32_t(cfg.idx_budget / ratio);
      const float sqrt_d = float(std::sqrt(double(d)));
      enc->setComputePipelineState(ctx->pipeline(PsoKey{1013, {int64_t(d), int64_t(nh), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"q4_index_scores", join({T(), I(long(d)), I(long(nh))})};
      }));
      set(enc, R(iq), 0);
      set(enc, R(A.bkeys), 1);
      set(enc, R(scores), 2);
      bytes(enc, nb32, 3);
      bytes(enc, sqrt_d, 4);
      enc->dispatchThreads(MTL::Size(nblocks, 1, 1), MTL::Size(std::min<std::size_t>(256, nblocks), 1, 1));
      barrier(enc);
      enc->setComputePipelineState(ctx->pipeline("q4_index_select", "1024"));
      set(enc, R(scores), 0);
      set(enc, R(sel), 1);
      bytes(enc, nb32, 2);
      bytes(enc, k32, 3);
      enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
      barrier(enc);
      nb_mask = nb32;
    }
    attention(enc, A, int(t_kv), nb_mask);
    barrier(enc);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1014, {int64_t(D), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_sigmoid_mul", join({T(), I(long(D))})};
    }));
    set(enc, R(att), 0);
    set(enc, R(qg), 1);
    enc->dispatchThreads(MTL::Size(D, Hq, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    barrier(enc);
  }

  // MLX's dispatch of mx.fast.scaled_dot_product_attention for one query row: head dims 64 / 96 / 128 / 256 run
  // sdpa_vector (q4_sdpa_vector_sel = suc_sdpa_vector + MLX's bool-mask branch for the QSA selection) or, on long
  // contexts, the two-pass kernels (the rule of Model::Impl::sdpa2_rule: 'd' devices from 1024 keys, GQA from 4096 keys;
  // q4_sdpa_2pass_1_sel + suc_sdpa_2pass_2); other head dims run MLX's unfused fallback (q4_attn FB = 1).
  int sdpa2_blocks(int n) const {
    const std::string& arch = ctx->architecture();
    const char devc = arch.empty() ? ' ' : arch.back();
    const int ns = int(cfg.heads / cfg.kv_heads);
    if (!((devc == 'd' && n >= 1024) || (cfg.kv_heads < cfg.heads && n >= 4096))) return 0;
    if (devc == 's') {
      int blocks = 64;
      if (n > 1024 && ns > 4) blocks = n <= 8192 ? 128 : n <= 32768 ? 256 : n <= 65536 ? 512 : 1024;
      return blocks;
    }
    if (devc == 'd') {
      int blocks = 128;
      if (ns <= 2 && n > 8192) blocks = 256;
      else if (ns >= 6) blocks = (n >= 16384 && n < 65536) ? 512 : n >= 65536 ? 1024 : 128;
      return blocks;
    }
    return ns >= 4 ? 64 : 32;
  }
  BufPtr sdpa_part, sdpa_sums, sdpa_maxs;
  std::size_t sdpa_blocks_cap = 0;

  void attention(MTL::ComputeCommandEncoder* enc, const Attn& A, int n, uint32_t nb) {
    const std::size_t Hq = cfg.heads, Hkv = cfg.kv_heads, D = cfg.head_dim;
    const bool vec_dim = D == 64 || D == 96 || D == 128 || D == 256;
    const uint32_t r32 = uint32_t(std::max<std::size_t>(cfg.idx_ratio, 1));
    if (vec_dim) {
      const std::size_t head_stride = D, seq_stride = Hkv * D;
      const int blocks = sdpa2_blocks(n);
      if (blocks) {
        if (sdpa_blocks_cap < std::size_t(blocks)) {
          sdpa_part = ctx->new_buffer(Hq * blocks * D * as());
          sdpa_sums = ctx->new_buffer(Hq * blocks * 4);
          sdpa_maxs = ctx->new_buffer(Hq * blocks * 4);
          sdpa_blocks_cap = std::size_t(blocks);
        }
        enc->setComputePipelineState(ctx->pipeline(PsoKey{1021, {int64_t(D), blocks, int64_t(opt.act)}}, [&]() {
          return std::pair<std::string, std::string>{"q4_sdpa_2pass_1_sel", join({T(), I(long(D)), I(long(D)), I(blocks)})};
        }));
        set(enc, R(q), 0);
        set(enc, R(A.kcache), 1);
        set(enc, R(A.vcache), 2);
        set(enc, R(sdpa_part), 3);
        set(enc, R(sdpa_sums), 4);
        set(enc, R(sdpa_maxs), 5);
        bytes(enc, n, 7);
        bytes(enc, head_stride, 8);
        bytes(enc, seq_stride, 9);
        bytes(enc, head_stride, 10);
        bytes(enc, seq_stride, 11);
        bytes(enc, attn_scale, 12);
        set(enc, R(sel), 13);
        bytes(enc, nb, 14);
        bytes(enc, r32, 15);
        enc->dispatchThreadgroups(MTL::Size(Hkv, 1, blocks), MTL::Size(32, Hq / Hkv, 1));
        barrier(enc);
        enc->setComputePipelineState(ctx->pipeline("suc_sdpa_2pass_2", join({T(), I(long(D))})));
        set(enc, R(sdpa_part), 0);
        set(enc, R(sdpa_sums), 1);
        set(enc, R(sdpa_maxs), 2);
        set(enc, R(att), 3);
        bytes(enc, blocks, 4);
        enc->dispatchThreadgroups(MTL::Size(Hq, 1, 1), MTL::Size(1024, 1, 1));
        return;
      }
      enc->setComputePipelineState(ctx->pipeline(PsoKey{1015, {int64_t(D), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"q4_sdpa_vector_sel", join({T(), I(long(D)), I(long(D))})};
      }));
      set(enc, R(q), 0);
      set(enc, R(A.kcache), 1);
      set(enc, R(A.vcache), 2);
      set(enc, R(att), 3);
      const int gqa = int(Hq / Hkv);
      bytes(enc, gqa, 4);
      bytes(enc, n, 5);
      bytes(enc, head_stride, 6);
      bytes(enc, seq_stride, 7);
      bytes(enc, head_stride, 8);
      bytes(enc, seq_stride, 9);
      bytes(enc, attn_scale, 10);
      set(enc, R(sel), 11);
      bytes(enc, nb, 12);
      bytes(enc, r32, 13);
      const uint32_t zero = 0;
      bytes(enc, zero, 14);
      bytes(enc, zero, 15);
      set(enc, ctx->dummy(), 16);
      set(enc, ctx->dummy(), 17);
      enc->dispatchThreadgroups(MTL::Size(Hq, 1, 1), MTL::Size(1024, 1, 1));
      return;
    }
    const int fb = vec_dim ? 0 : 1;
    constexpr int tpg = 256;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1016, {int64_t(D), fb, int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_attn", join({T(), I(long(D)), I(tpg), I(fb)})};
    }));
    set(enc, R(q), 0);
    set(enc, R(A.kcache), 1);
    set(enc, R(A.vcache), 2);
    set(enc, R(att), 3);
    set(enc, R(sc), 4);
    set(enc, R(sel), 5);
    const uint32_t n32 = uint32_t(n), gqa = uint32_t(Hq / Hkv), ss = uint32_t(Hkv * D), scs = uint32_t(opt.max_context);
    bytes(enc, n32, 6);
    bytes(enc, gqa, 7);
    bytes(enc, ss, 8);
    bytes(enc, nb, 9);
    bytes(enc, r32, 10);
    bytes(enc, attn_scale, 11);
    bytes(enc, scs, 12);
    enc->dispatchThreadgroups(MTL::Size(Hq, 1, 1), MTL::Size(tpg, 1, 1));
  }

  // f = SparseMoeBlock(u)
  void moe(MTL::ComputeCommandEncoder*& enc, const Layer& L) {
    const std::size_t E = cfg.num_experts, K = cfg.top_k, H = cfg.hidden, Fs = cfg.shared_intermediate;
    proj(enc, L.router, R(u), R(gates), 0);
    proj(enc, L.sh_gate, R(u), R(shg), 1);
    proj(enc, L.sh_up, R(u), R(shu), 2);
    proj(enc, L.sh_router, R(u), R(sgate), 3);
    barrier(enc);
    const std::size_t tpg = 32 * (((E + 3) / 4 + 31) / 32);
    const int norm = cfg.norm_topk_prob ? 1 : 0;
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1017, {int64_t(E), int64_t(K), norm, int64_t(tpg), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_route", join({T(), I(long(E)), I(long(K)), I(norm), I(long(tpg))})};
    }));
    set(enc, R(gates), 0);
    set(enc, R(eidx), 1);
    set(enc, R(escores), 2);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(tpg, 1, 1));
    enc->setComputePipelineState(ctx->pipeline("suc_swiglu", T()));
    set(enc, R(shg), 0);
    set(enc, R(shu), 1);
    set(enc, R(shh), 2);
    const uint32_t nf = uint32_t(Fs);
    bytes(enc, nf, 3);
    enc->dispatchThreads(MTL::Size(Fs, 1, 1), MTL::Size(std::min<std::size_t>(256, Fs), 1, 1));
    barrier(enc);
    if (L.moe->streamed()) {   // host sync: fetch the routed experts into the slot pool (and prefetch a guess)
      const Layer* nxt = (&L != &layers.back() && L.moe->stream()->options().prefetch) ? &L + 1 : nullptr;
      if (nxt) {   // pre-gating guess: the next layer's router on this layer's MoE input
        proj(enc, nxt->router, R(u), R(pgates), 5);
        barrier(enc);
        enc->setComputePipelineState(ctx->pipeline(PsoKey{1017, {int64_t(E), int64_t(K), norm, int64_t(tpg), int64_t(opt.act)}}, [&]() {
          return std::pair<std::string, std::string>{"q4_route", join({T(), I(long(E)), I(long(K)), I(norm), I(long(tpg))})};
        }));
        set(enc, R(pgates), 0);
        set(enc, R(peidx), 1);
        set(enc, R(pscores), 2);
        enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(tpg, 1, 1));
      }
      split_command_buffer(*ctx, cur_cb, enc, &split_gpu_s);
      L.moe->stream_in(static_cast<const uint32_t*>(eidx->contents()), K);
      if (nxt) nxt->moe->stream_prefetch(static_cast<const uint32_t*>(peidx->contents()), K);
    }
    L.moe->encode_experts(*ctx, enc, R(u), R(eidx), R(escores), R(f), 1, moe_scratch);
    proj(enc, L.sh_down, R(shh), R(sh), 4);
    barrier(enc);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1018, {int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_shared_add", T()};
    }));
    set(enc, R(f), 0);
    set(enc, R(sgate), 1);
    set(enc, R(sh), 2);
    const uint32_t nh = uint32_t(H);
    bytes(enc, nh, 3);
    enc->dispatchThreads(MTL::Size(H, 1, 1), MTL::Size(std::min<std::size_t>(256, H), 1, 1));
    barrier(enc);
  }

  // One decode step for the token in slot `slot` (tok: hc copies; ple_e: n-gram rows) at position p.
  // enc / cur_cb are replaced at every streamed MoE layer (split_command_buffer).
  void encode_step(MTL::ComputeCommandEncoder*& enc, std::size_t slot, std::size_t p, bool with_logits) {
    const std::size_t H = cfg.hidden, W = cfg.stream(), a = as();
    // embedding -> hc identical streams (the gather of hc copies of the token id)
    {
      const Mat& m = embed;
      if (m.kind == Mat::Affine) {
        enc->setComputePipelineState(ctx->pipeline(PsoKey{1019, {int64_t(H), m.bits, int64_t(m.group), int64_t(m.sdtype), int64_t(opt.act)}}, [&]() {
          return std::pair<std::string, std::string>{
              "suc_embed_affine", join({T(), I(m.bits), I(long(m.group)), I(long(m.in)), I(long(m.words)), msl_of(m.sdtype)})};
        }));
        set(enc, R(tok, slot * cfg.hc * 4), 0);
        set(enc, m.packed, 1);
        set(enc, m.scales, 2);
        set(enc, m.biases, 3);
        set(enc, R(hs), 4);
      } else {
        enc->setComputePipelineState(ctx->pipeline(PsoKey{1020, {int64_t(H), int64_t(m.wdtype), int64_t(opt.act)}}, [&]() {
          return std::pair<std::string, std::string>{"suc_embed_dense", join({T(), I(long(m.in)), msl_of(m.wdtype)})};
        }));
        set(enc, R(tok, slot * cfg.hc * 4), 0);
        set(enc, m.w, 1);
        set(enc, R(hs), 4);
      }
      enc->dispatchThreads(MTL::Size(H, cfg.hc, 1), MTL::Size(std::min<std::size_t>(256, H), 1, 1));
      barrier(enc);
    }
    if (cfg.dense) {
      encode_dense_layers(enc, p, with_logits);
      return;
    }
    const std::size_t E = cfg.ple_dim, nple = cfg.ple_layer_ids.size();
    for (std::size_t l = 0; l < cfg.layers; ++l) {
      const Layer& L = layers[l];
      const BufRef x = R(hs, l * W * a), xo = R(hs, (l + 1) * W * a);
      BufRef cur = x;
      if (L.ple) {
        ple(enc, *L.ple, x, R(ple_e, (slot * nple + std::size_t(L.ple->index)) * E * a), R(xa));
        cur = R(xa);
      }
      hyper_mix(enc, L.ahc, cur);
      if (L.full) full_attn(enc, L.att, p);
      else linear_attn(enc, L.lin);
      inject(enc, cur, R(xa));
      hyper_mix(enc, L.mhc, R(xa));
      moe(enc, L);
      inject(enc, R(xa), xo);
    }
    if (with_logits) {
      hyper_mix(enc, final_mixer, R(hs, cfg.layers * W * a));
      proj(enc, head, R(u), R(logits), 0);
      barrier(enc);
      enc->setComputePipelineState(ctx->pipeline("suc_argmax", join({T(), "1024"})));
      set(enc, R(logits), 0);
      set(enc, R(amax), 1);
      const uint32_t n = uint32_t(cfg.vocab);
      bytes(enc, n, 2);
      enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
    }
  }

  // y = a + b (rows of n, residual add in T: suc_add)
  void add(MTL::ComputeCommandEncoder* enc, BufRef a, BufRef b, BufRef y, std::size_t n) {
    enc->setComputePipelineState(ctx->pipeline("suc_add", T()));
    set(enc, a, 0);
    set(enc, b, 1);
    set(enc, y, 2);
    const uint32_t n32 = uint32_t(n);
    bytes(enc, n32, 3);
    enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(std::min<std::size_t>(256, n), 1, 1));
    barrier(enc);
  }

  // Dense qwen3_5 blocks on hs (one stream of H), then the final (1 + w) norm and the LM head.
  void encode_dense_layers(MTL::ComputeCommandEncoder* enc, std::size_t p, bool with_logits) {
    const std::size_t H = cfg.hidden, F = cfg.intermediate, a = as();
    for (std::size_t l = 0; l < cfg.layers; ++l) {
      const Layer& L = layers[l];
      const BufRef x = R(hs, l * H * a), xo = R(hs, (l + 1) * H * a);
      rms(enc, 0, x, &L.in_norm, R(u), 1, H, H, H, 1);
      barrier(enc);
      if (L.full) full_attn(enc, L.att, p);
      else linear_attn(enc, L.lin);
      add(enc, x, R(f), R(xa), H);
      rms(enc, 0, R(xa), &L.post_norm, R(u), 1, H, H, H, 1);
      barrier(enc);
      proj(enc, L.mlp_gate, R(u), R(shg), 0);
      proj(enc, L.mlp_up, R(u), R(shu), 1);
      barrier(enc);
      enc->setComputePipelineState(ctx->pipeline("suc_swiglu", T()));
      set(enc, R(shg), 0);
      set(enc, R(shu), 1);
      set(enc, R(shh), 2);
      const uint32_t nf = uint32_t(F);
      bytes(enc, nf, 3);
      enc->dispatchThreads(MTL::Size(F, 1, 1), MTL::Size(std::min<std::size_t>(256, F), 1, 1));
      barrier(enc);
      proj(enc, L.mlp_down, R(shh), R(f), 0);
      barrier(enc);
      add(enc, R(xa), R(f), xo, H);
    }
    if (with_logits) {
      rms(enc, 0, R(hs, cfg.layers * H * a), &final_norm, R(u), 1, H, H, H, 1);
      barrier(enc);
      proj(enc, head, R(u), R(logits), 0);
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
  // Chunk scratch (Qwen4Options::layer_prefill), rows = chunk tokens.
  BufPtr ring, pl_c, xa_c, u_c, inj_c, inj2_c, qkv_c, z_c, av_c, bv_c, yn_c, qg_c, kb_c, iqk_c, att_c, f_c, gates_c,
      eidx_c, escores_c, shg_c, shu_c, shh_c, sh_c, sgate_c, hx_c, hd_c, hs_c, hu_c;
  BufPtr conv_c, qn_c, kn_c, gdec_c, beta_c, yd_c;   // Gated DeltaNet chunk core (lin_core_rows)
  BufPtr t_tok, t_dst, t_reid, t_teidx;
  BufPtr logits_rows;                   // forward_rows(): (chunk, vocab) logits of every chunk row
  std::vector<float>* rows_out = nullptr;
  bool fast() const { return !opt.exact_prefill && !prefill_exact_env(); }
  // $SUC_FAST_OFF (diagnostics): letters of fast parts to keep row-exact: p projections, t expert tiles
  static bool fast_off(char c) {
    static const std::string v = [] { const char* e = std::getenv("SUC_FAST_OFF"); return std::string(e ? e : ""); }();
    return v.find(c) != std::string::npos;
  }
  bool fast(char c) const { return fast() && !fast_off(c); }
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


  // y rows r = x rows r · Wᵀ (row strides xs / ys in elements), every row bit-identical to its one-row launch:
  // affine = one suc_gemv_affine_rows launch, dense = one suc_moe_gemv_rows launch, SUC = Linear::encode_rows_exact
  // (contiguous rows; else the one-row launch per row, barriers between rows: they share ws[wsi]).
  void proj_rows(MTL::ComputeCommandEncoder* enc, const Mat& m, BufRef x, BufRef y, std::size_t n, std::size_t xs,
                 std::size_t ys, int wsi = 0) {
    const std::size_t a = as();
    if (fast('p') && n >= kMmMinRows && !row_proj()) {   // batched prefill, not row-exact: GEMMs over the n rows
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
    if (n > 1 && m.kind == Mat::Suc && xs == m.in && ys == m.out && !row_proj()) {
      m.lin->encode_rows_exact(*ctx, enc, x, y, n, opt.act, ws[wsi]);
      return;
    }
    if (n > 1 && m.kind == Mat::Dense && !row_proj()) {
      gemv_rows(enc, m, x, y, n, xs, ys);
      return;
    }
    if (m.kind != Mat::Affine || n == 1) {
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
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1070, {int64_t(m.in), int64_t(m.out), m.bits, int64_t(m.group), int64_t(m.sdtype), int64_t(opt.act), NB}}, [&]() {
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
    enc->dispatchThreadgroups(MTL::Size((m.out + sm - 1) / sm, (n + NB - 1) / NB, 1), MTL::Size(32 * bn, 1, 1));
  }

  // $SUC_PREFILL_ROW_PROJ=1: SUC / dense projections of a batched-prefill chunk one row at a time (the previous path;
  // same bits, for A/B timing).
  static bool row_proj() {
    static const bool v = [] { const char* e = std::getenv("SUC_PREFILL_ROW_PROJ"); return e && e[0] == '1'; }();
    return v;
  }

  // Dense (raw) weight over n rows: suc_moe_gemv_rows (each row = the one-row suc_moe_gemv of proj()).
  void gemv_rows(MTL::ComputeCommandEncoder* enc, const Mat& m, BufRef x, BufRef y, std::size_t n, std::size_t xs,
                 std::size_t ys) {
    const int sn = m.in <= 64 ? 4 : 32;
    const int bn = (m.in > 64 && m.in >= 16 * m.out) ? 8 : 1;
    const int sm = 32 / sn;
    static const int NB = gemv_rows_nb();
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1071, {int64_t(m.in), int64_t(m.out), int64_t(m.wdtype), int64_t(opt.act), NB}}, [&]() {
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
  }

  void copy_row(MTL::ComputeCommandEncoder* enc, BufRef x, BufRef y, std::size_t cols) {
    encode_copy_cols(*ctx, enc, x, y, 1, cols, cols, cols, opt.act);
  }

  void encode_embed(MTL::ComputeCommandEncoder* enc, std::size_t slot, BufRef dst) {
    const std::size_t H = cfg.hidden;
    const Mat& m = embed;
    if (m.kind == Mat::Affine) {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{1019, {int64_t(H), m.bits, int64_t(m.group), int64_t(m.sdtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{
            "suc_embed_affine", join({T(), I(m.bits), I(long(m.group)), I(long(m.in)), I(long(m.words)), msl_of(m.sdtype)})};
      }));
      set(enc, R(tok, slot * cfg.hc * 4), 0);
      set(enc, m.packed, 1);
      set(enc, m.scales, 2);
      set(enc, m.biases, 3);
      set(enc, dst, 4);
    } else {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{1020, {int64_t(H), int64_t(m.wdtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"suc_embed_dense", join({T(), I(long(m.in)), msl_of(m.wdtype)})};
      }));
      set(enc, R(tok, slot * cfg.hc * 4), 0);
      set(enc, m.w, 1);
      set(enc, dst, 4);
    }
    enc->dispatchThreads(MTL::Size(H, cfg.hc, 1), MTL::Size(std::min<std::size_t>(256, H), 1, 1));
    barrier(enc);
  }

  void encode_logits(MTL::ComputeCommandEncoder* enc, BufRef x) {
    const std::size_t H = cfg.hidden;
    if (cfg.dense) {
      rms(enc, 0, x, &final_norm, R(u), 1, H, H, H, 1);
      barrier(enc);
    } else {
      hyper_mix(enc, final_mixer, x);
    }
    proj(enc, head, R(u), R(logits), 0);
    barrier(enc);
    enc->setComputePipelineState(ctx->pipeline("suc_argmax", join({T(), "1024"})));
    set(enc, R(logits), 0);
    set(enc, R(amax), 1);
    const uint32_t nv = uint32_t(cfg.vocab);
    bytes(enc, nv, 2);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
  }

  // hyper_mix of n chunk tokens, stage-major (per-token scratch hx_c / hd_c / hs_c / hu_c, outputs u_c rows and
  // inj_out rows): the kernels of hyper_mix per token, one barrier per stage. Same bits.
  // xb (fast batched prefill): the n stream rows contiguous from xb, every per-token launch row-batched (same bits)
  void hyper_mix_rows(MTL::ComputeCommandEncoder* enc, const Hyper& h, const std::function<BufRef(std::size_t)>& x,
                      BufRef inj_out, std::size_t n, BufRef xb = {}) {
    const std::size_t H = cfg.hidden, W = cfg.stream(), HC = cfg.hc, Lr = cfg.hc_lowrank, a = as();
    const bool rb = bool(xb);
    if (rb) rms(enc, 0, xb, &h.norm, R(hx_c), n * HC, H, H, H, HC);
    else
      for (std::size_t i = 0; i < n; ++i) rms(enc, 0, x(i), &h.norm, R(hx_c, i * W * a), HC, H, H, H, HC);
    barrier(enc);
    proj_rows(enc, h.down, R(hx_c), R(hd_c), n, W, Lr, 0);
    if (h.combine) proj_rows(enc, h.inject, R(hx_c), inj_out, n, W, HC, 1);
    barrier(enc);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1004, {int64_t(HC), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_silu_div", join({T(), I(long(HC))})};
    }));
    const uint32_t lr32 = uint32_t(rb ? n * Lr : Lr);
    for (std::size_t i = 0; i < (rb ? 1 : n); ++i) {
      set(enc, R(hd_c, i * Lr * a), 0);
      set(enc, R(hs_c, i * Lr * a), 1);
      bytes(enc, lr32, 2);
      enc->dispatchThreads(MTL::Size(lr32, 1, 1), MTL::Size(std::min<std::size_t>(256, Lr), 1, 1));
    }
    barrier(enc);
    proj_rows(enc, h.up, R(hs_c), R(hu_c), n, Lr, W, 0);
    barrier(enc);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1005, {int64_t(H), int64_t(HC), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_hc_mix", join({T(), I(long(H)), I(long(HC))})};
    }));
    for (std::size_t i = 0; i < (rb ? 1 : n); ++i) {
      set(enc, R(hu_c, i * W * a), 0);
      set(enc, R(hx_c, i * W * a), 1);
      set(enc, R(u_c, i * H * a), 2);
      enc->dispatchThreads(MTL::Size(H, rb ? n : 1, 1), MTL::Size(std::min<std::size_t>(256, H), 1, 1));
    }
    barrier(enc);
  }

  // inject of n chunk tokens: y(i) = x(i) + block output f_c row i, gated by the inj rows (q4_inject per token).
  // xb / yb (fast batched prefill): contiguous stream rows, one launch (grid z = rows)
  void inject_rows(MTL::ComputeCommandEncoder* enc, const std::function<BufRef(std::size_t)>& x, BufRef inj_rows,
                   const std::function<BufRef(std::size_t)>& y, std::size_t n, BufRef xb = {}, BufRef yb = {}) {
    const std::size_t H = cfg.hidden, HC = cfg.hc, a = as();
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1006, {int64_t(H), int64_t(HC), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_inject", join({T(), I(long(H)), I(long(HC))})};
    }));
    if (xb && yb) {
      set(enc, xb, 0);
      set(enc, R(f_c), 1);
      set(enc, inj_rows, 2);
      set(enc, yb, 3);
      enc->dispatchThreads(MTL::Size(H, HC, n), MTL::Size(std::min<std::size_t>(256, H), 1, 1));
      barrier(enc);
      return;
    }
    for (std::size_t i = 0; i < n; ++i) {
      set(enc, x(i), 0);
      set(enc, R(f_c, i * H * a), 1);
      set(enc, inj_rows.at(i * HC * a), 2);
      set(enc, y(i), 3);
      enc->dispatchThreads(MTL::Size(H, HC, 1), MTL::Size(std::min<std::size_t>(256, H), 1, 1));
    }
    barrier(enc);
  }

  // Gated full attention of chunk rows 0 .. n-1 (positions p0 ..; qg_c / kb_c / iqk_c / the v cache rows written) in
  // row-batched launches (fast batched prefill): q / k norms and RoPE over all rows (per-row positions), the indexer
  // keys stored, the blocks completed in the chunk pooled in one launch, the QSA block scores and top-k selection of
  // every row past the budget, ONE sdpa_vector launch over (heads, rows) with the causal key count and selection mask
  // of each row (always the single-pass kernel), the sigmoid gate over all rows. -> att_c.
  void full_attn_core_rows(MTL::ComputeCommandEncoder* enc, const Attn& A, std::size_t p0, std::size_t n) {
    const std::size_t Hq = cfg.heads, Hkv = cfg.kv_heads, D = cfg.head_dim, a = as(), KD = Hkv * D;
    const std::size_t d = cfg.idx_dim, nh = cfg.idx_heads, ratio = std::max<std::size_t>(cfg.idx_ratio, 1);
    const std::size_t iw = (nh + 1) * d, Kb = cfg.indexer ? cfg.idx_budget / ratio : 0;
    const std::size_t pe = p0 + n, nb_end = cfg.indexer ? pe / ratio : 0;
    const bool any_sparse = cfg.indexer && nb_end > Kb;
    if (!q_rc || q_rc_rows < n) {
      q_rc = alloc(n * Hq * D * a, nullptr);
      kn_rc = alloc(n * KD * a, nullptr);
      if (cfg.indexer) iq_rc = alloc(n * iw * a, nullptr);
      q_rc_rows = n;
    }
    if (any_sparse && (nb_end > isel_ps || isel_rows < n)) {
      isel_ps = std::max(nb_end, isel_ps * 2);
      isel_rows = std::max(isel_rows, n);
      iscore_rc = alloc(isel_rows * isel_ps * 4, nullptr);
    }
    rms(enc, 0, R(qg_c), &A.q_norm, R(q_rc), n * Hq, D, 2 * D, D, 1);
    rms(enc, 0, R(kb_c), &A.k_norm, R(kn_rc), n * Hkv, D, D, D, 1);
    if (cfg.indexer) {
      encode_copy_cols(*ctx, enc, R(iqk_c, nh * d * a), R(A.ikeys, p0 * d * a), n, d, iw, d, opt.act);
      if (any_sparse) rms(enc, 0, R(iqk_c), &A.iq_norm, R(iq_rc), n * (nh + 1), d, d, d, 1);
    }
    barrier(enc);
    rope(enc, R(q_rc), R(q_rc), n * Hq, D, D, D, int(p0), uint32_t(Hq));
    rope(enc, R(kn_rc), R(A.kcache, p0 * KD * a), n * Hkv, D, D, D, int(p0), uint32_t(Hkv));
    if (any_sparse) rope(enc, R(iq_rc), R(iq_rc), n * (nh + 1), d, d, d, int(p0), uint32_t(nh + 1));
    if (cfg.indexer && nb_end > p0 / ratio) {   // blocks completed inside the chunk (decode: at their last token)
      const uint32_t b0 = uint32_t(p0 / ratio), r32 = uint32_t(ratio);
      const float base = float(cfg.rope_theta);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{1012, {int64_t(d), int64_t(cfg.rotary_dim), int64_t(A.ik_norm.dtype), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"q4_block_key", join({T(), A.ik_norm.msl(), I(long(d)), I(long(cfg.rotary_dim))})};
      }));
      set(enc, R(A.ikeys), 0);
      set(enc, A.ik_norm.ref, 1);
      set(enc, R(A.bkeys), 2);
      bytes(enc, b0, 3);
      bytes(enc, r32, 4);
      bytes(enc, eps, 5);
      bytes(enc, base, 6);
      enc->dispatchThreadgroups(MTL::Size(nb_end - b0, 1, 1), MTL::Size(d, 1, 1));
    }
    barrier(enc);
    const uint32_t p32 = uint32_t(p0), r32 = uint32_t(ratio), k32 = uint32_t(Kb), ps32 = uint32_t(isel_ps);
    if (any_sparse) {
      const float sqrt_d = float(std::sqrt(double(d)));
      const uint32_t qs = uint32_t(iw);
      enc->setComputePipelineState(ctx->pipeline(PsoKey{1030, {int64_t(d), int64_t(nh), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"q4_index_scores_rows", join({T(), I(long(d)), I(long(nh))})};
      }));
      set(enc, R(iq_rc), 0);
      set(enc, R(A.bkeys), 1);
      set(enc, R(iscore_rc), 2);
      bytes(enc, p32, 3);
      bytes(enc, sqrt_d, 4);
      bytes(enc, r32, 5);
      bytes(enc, k32, 6);
      bytes(enc, qs, 7);
      bytes(enc, ps32, 8);
      enc->dispatchThreads(MTL::Size(nb_end, n, 1), MTL::Size(std::min<std::size_t>(256, nb_end), 1, 1));
      barrier(enc);
      // top-k blocks of every row as ascending block lists (q4_index_select's rule): the attention visits only them
      if (!sidx_rc || sidx_rows < n) {
        sidx_rc = alloc(n * Kb * 4, nullptr);
        scnt_rc = alloc(n * 4, nullptr);
        nb_rc = alloc(n * 4, nullptr);
        sidx_rows = n;
      }
      auto* nbv = static_cast<uint32_t*>(nb_rc->contents());   // same for every layer of the chunk
      for (std::size_t i = 0; i < n; ++i) nbv[i] = uint32_t((p0 + i + 1) / ratio);
      enc->setComputePipelineState(ctx->pipeline("ds4_index_select_rows", "1024"));
      set(enc, R(iscore_rc), 0);
      set(enc, R(sidx_rc), 1);
      set(enc, R(scnt_rc), 2);
      set(enc, R(nb_rc), 3);
      bytes(enc, k32, 4);
      bytes(enc, ps32, 5);
      enc->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(1024, 1, 1));
      barrier(enc);
    }
    {
      const std::size_t head_stride = D, seq_stride = KD;
      enc->setComputePipelineState(ctx->pipeline(PsoKey{1015, {int64_t(D), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"q4_sdpa_vector_sel", join({T(), I(long(D)), I(long(D))})};
      }));
      set(enc, R(q_rc), 0);
      set(enc, R(A.kcache), 1);
      set(enc, R(A.vcache), 2);
      set(enc, R(att_c), 3);
      const int gqa = int(Hq / Hkv), N = int(pe);
      bytes(enc, gqa, 4);
      bytes(enc, N, 5);
      bytes(enc, head_stride, 6);
      bytes(enc, seq_stride, 7);
      bytes(enc, head_stride, 8);
      bytes(enc, seq_stride, 9);
      bytes(enc, attn_scale, 10);
      set(enc, ctx->dummy(), 11);
      const uint32_t zero = 0, rr = any_sparse ? r32 : 1u, kk = any_sparse ? k32 : 0u;
      bytes(enc, zero, 12);
      bytes(enc, rr, 13);
      bytes(enc, zero, 14);   // list mode
      bytes(enc, kk, 15);
      set(enc, any_sparse ? R(sidx_rc) : ctx->dummy(), 16);
      set(enc, any_sparse ? R(scnt_rc) : ctx->dummy(), 17);
      enc->dispatchThreadgroups(MTL::Size(Hq, n, 1), MTL::Size(1024, 1, 1));
    }
    barrier(enc);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1014, {int64_t(D), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_sigmoid_mul", join({T(), I(long(D))})};
    }));
    set(enc, R(att_c), 0);
    set(enc, R(qg_c), 1);
    enc->dispatchThreads(MTL::Size(D, n * Hq, 1), MTL::Size(std::min<std::size_t>(256, D), 1, 1));
    barrier(enc);
  }
  BufPtr q_rc, kn_rc, iq_rc, iscore_rc, sidx_rc, scnt_rc, nb_rc;
  std::size_t q_rc_rows = 0, isel_ps = 0, isel_rows = 0, sidx_rows = 0;

  // Mixer of chunk rows (u_c -> f_c) at positions p0 ...: input projections over the rows, the recurrent / cached part
  // token by token on the one-token buffers (rows copied in and out), output projection over the rows.
  void mixer_chunk(MTL::ComputeCommandEncoder*& enc, const Layer& L, std::size_t p0, std::size_t n) {
    const std::size_t H = cfg.hidden, a = as();
    if (L.full) {
      const Attn& A = L.att;
      const std::size_t Hq = cfg.heads, D = cfg.head_dim, KD = cfg.kv_heads * D, iw = (cfg.idx_heads + 1) * cfg.idx_dim;
      mark(enc, "attn_in_proj");
      if (cfg.indexer) proj_rows(enc, A.iqk, R(u_c), R(iqk_c), n, H, iw, 0);
      proj_rows(enc, A.q, R(u_c), R(qg_c), n, H, 2 * Hq * D, 1);
      proj_rows(enc, A.k, R(u_c), R(kb_c), n, H, KD, 2);
      proj_rows(enc, A.v, R(u_c), R(A.vcache, p0 * KD * a), n, H, KD, 3);
      barrier(enc);
      mark(enc, "attn_core");
      if (fast('a') && n >= kMmMinRows && (D == 64 || D == 96 || D == 128 || D == 256)) {
        full_attn_core_rows(enc, A, p0, n);
        mark(enc, "attn_out_proj");
        proj_rows(enc, A.o, R(att_c), R(f_c), n, Hq * D, H, 0);
        barrier(enc);
        return;
      }
      for (std::size_t i = 0; i < n; ++i) {
        if (cfg.indexer) copy_row(enc, R(iqk_c, i * iw * a), R(iqk), iw);
        copy_row(enc, R(qg_c, i * 2 * Hq * D * a), R(qg), 2 * Hq * D);
        copy_row(enc, R(kb_c, i * KD * a), R(kb), KD);
        barrier(enc);
        full_attn_core(enc, A, p0 + i);
        copy_row(enc, R(att), R(att_c, i * Hq * D * a), Hq * D);
        barrier(enc);
      }
      mark(enc, "attn_out_proj");
      proj_rows(enc, A.o, R(att_c), R(f_c), n, Hq * D, H, 0);
      barrier(enc);
      return;
    }
    const LinAttn& La = L.lin;
    const std::size_t Hk = cfg.lin_k_heads, Hv = cfg.lin_v_heads, Dk = cfg.lin_k_dim, Dv = cfg.lin_v_dim;
    const std::size_t vd = Hv * Dv, cd = 2 * Hk * Dk + vd;
    mark(enc, "lin_in_proj");
    proj_rows(enc, La.qkv, R(u_c), R(qkv_c), n, H, cd, 0);
    proj_rows(enc, La.z, R(u_c), R(z_c), n, H, vd, 1);
    proj_rows(enc, La.a, R(u_c), R(av_c), n, H, Hv, 2);
    proj_rows(enc, La.b, R(u_c), R(bv_c), n, H, Hv, 3);
    barrier(enc);
    mark(enc, "lin_core");
    static const bool core_rows = [] { const char* e = std::getenv("SUC_PREFILL_LIN_TOKENS"); return !(e && e[0] == '1'); }();
    if (core_rows) lin_core_rows(enc, La, n);
    else for (std::size_t i = 0; i < n; ++i) {
      copy_row(enc, R(qkv_c, i * cd * a), R(qkv), cd);
      copy_row(enc, R(z_c, i * vd * a), R(z), vd);
      copy_row(enc, R(av_c, i * Hv * a), R(av), Hv);
      copy_row(enc, R(bv_c, i * Hv * a), R(bv), Hv);
      barrier(enc);
      linear_attn_core(enc, La);
      copy_row(enc, R(yn), R(yn_c, i * vd * a), vd);
      barrier(enc);
    }
    mark(enc, "lin_out_proj");
    proj_rows(enc, La.out, R(yn_c), R(f_c), n, vd, H, 0);
    barrier(enc);
  }

  // The Gated DeltaNet step (linear_attn_core) of n consecutive chunk tokens (qkv_c, z_c, av_c, bv_c -> yn_c): the
  // conv, the delta prep and the recurrence each in ONE launch over the tokens (q4_*_rows: per token the arithmetic of
  // the one-token kernels), the q / k l2 norms per token without barriers between tokens, the gated RMSNorm over all
  // n · Hv rows. Same bits as n linear_attn_core steps; 4 barriers instead of 6 per token.
  // $SUC_PREFILL_LIN_TOKENS=1: the token-by-token loop (A/B).
  void lin_core_rows(MTL::ComputeCommandEncoder* enc, const LinAttn& L, std::size_t n) {
    const std::size_t Hk = cfg.lin_k_heads, Hv = cfg.lin_v_heads, Dk = cfg.lin_k_dim, Dv = cfg.lin_v_dim, a = as();
    const std::size_t key_dim = Hk * Dk, vd = Hv * Dv, cd = 2 * key_dim + vd, K = cfg.conv_kernel;
    if (L.a_log.dtype != L.dt_bias.dtype) fail("A_log / dt_bias dtypes differ");
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1072, {int64_t(K), int64_t(L.conv.dtype), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_conv_rows", join({T(), L.conv.msl(), I(long(K)), "1"})};
    }));
    set(enc, R(qkv_c), 0);
    set(enc, R(L.conv_state), 1);
    set(enc, L.conv.ref, 2);
    set(enc, R(conv_c), 3);
    const uint32_t c32 = uint32_t(cd), n32 = uint32_t(n);
    bytes(enc, c32, 6);
    bytes(enc, n32, 7);
    enc->dispatchThreads(MTL::Size(cd, 1, 1), MTL::Size(std::min<std::size_t>(256, cd), 1, 1));
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1073, {int64_t(L.a_log.dtype), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_delta_prep_rows", join({T(), L.a_log.msl()})};
    }));
    set(enc, R(av_c), 0);
    set(enc, R(bv_c), 1);
    set(enc, L.a_log.ref, 2);
    set(enc, L.dt_bias.ref, 3);
    set(enc, R(gdec_c), 4);
    set(enc, R(beta_c), 5);
    const uint32_t nh = uint32_t(Hv);
    bytes(enc, nh, 6);
    enc->dispatchThreads(MTL::Size(Hv, n, 1), MTL::Size(std::min<std::size_t>(256, Hv), 1, 1));
    barrier(enc);
    if (fast('h')) {   // all tokens in one launch each (row (t, h) of the conv output; same bits)
      rms(enc, 1, R(conv_c), nullptr, R(qn_c), n * Hk, Dk, Dk, Dk, 1, q_scale, {}, 1e-6f / float(Dk), Hk, cd);
      rms(enc, 1, R(conv_c, key_dim * a), nullptr, R(kn_c), n * Hk, Dk, Dk, Dk, 1, k_scale, {}, 1e-6f / float(Dk), Hk, cd);
    } else {
      for (std::size_t t = 0; t < n; ++t) {
        rms(enc, 1, R(conv_c, t * cd * a), nullptr, R(qn_c, t * key_dim * a), Hk, Dk, Dk, Dk, 1, q_scale, {},
            1e-6f / float(Dk));
        rms(enc, 1, R(conv_c, (t * cd + key_dim) * a), nullptr, R(kn_c, t * key_dim * a), Hk, Dk, Dk, Dk, 1, k_scale, {},
            1e-6f / float(Dk));
      }
    }
    barrier(enc);
    enc->setComputePipelineState(ctx->pipeline(PsoKey{1074, {int64_t(Dk), int64_t(Dv), int64_t(Hk), int64_t(Hv), int64_t(opt.act)}}, [&]() {
      return std::pair<std::string, std::string>{"q4_gated_delta_rows", join({T(), I(long(Dk)), I(long(Dv)), I(long(Hk)), I(long(Hv))})};
    }));
    set(enc, R(qn_c), 0);
    set(enc, R(kn_c), 1);
    set(enc, R(conv_c, 2 * key_dim * a), 2);
    set(enc, R(gdec_c), 3);
    set(enc, R(beta_c), 4);
    set(enc, R(L.rec_state), 5);
    set(enc, R(yd_c), 6);
    const uint32_t vs32 = uint32_t(cd);
    bytes(enc, n32, 7);
    bytes(enc, vs32, 8);
    enc->dispatchThreads(MTL::Size(32, Dv, Hv), MTL::Size(32, 4, 1));
    barrier(enc);
    rms(enc, 2, R(yd_c), &L.norm, R(yn_c), n * Hv, Dv, Dv, Dv, 1, 1.0f, R(z_c));
    barrier(enc);
  }

  void ensure_tiles(std::size_t rows, std::size_t tiles) {
    if (rows > t_rows_cap) {
      t_tok = alloc(rows * 4, nullptr);
      t_dst = alloc(rows * 4, nullptr);
      t_reid = alloc(rows * 4, nullptr);
      t_rows_cap = rows;
    }
    if (tiles > t_tiles_cap) {
      t_teidx = alloc(tiles * 4, nullptr);
      t_tiles_cap = tiles;
    }
  }

  // Routed experts of the chunk tokens in `group` (routing in eidx_c / escores_c) in expert tiles; no combine.
  void experts_tiled(MTL::ComputeCommandEncoder* enc, const Layer& L, const std::vector<std::size_t>& group,
                     std::size_t n) {
    const std::size_t K = cfg.top_k;
    const auto* ids = static_cast<const uint32_t*>(eidx_c->contents());
    std::vector<std::pair<uint32_t, uint32_t>> rows;
    for (std::size_t i : group)
      for (std::size_t k = 0; k < K; ++k) rows.push_back({uint32_t(i * K + k), ids[i * K + k]});
    const TilePlan P = plan_tiles(rows, fast('t') ? L.moe->fast_tile_rows() : L.moe->max_tile_rows(*ctx));
    ensure_tiles(P.rows(), P.count());
    auto* tk = static_cast<uint32_t*>(t_tok->contents());
    for (std::size_t j = 0; j < P.rows(); ++j) tk[j] = P.src[j] / uint32_t(K);
    std::memcpy(t_dst->contents(), P.dst.data(), P.rows() * 4);
    std::memcpy(t_reid->contents(), P.eidx.data(), P.rows() * 4);
    std::memcpy(t_teidx->contents(), P.teidx.data(), P.count() * 4);
    Tiles tiles{R(t_teidx), P.count(), P.bt, P.teidx.data()};
    tiles.exact = !fast('t');
    L.moe->encode_experts_tiled(*ctx, enc, R(u_c), R(t_tok), R(t_dst), R(t_reid), tiles, R(escores_c), R(f_c), n,
                                moe_scratch, false);
  }

  // SparseMoeBlock of chunk rows u_c -> f_c (one host sync per layer per chunk; streamed: per fetch group).
  void moe_chunk(MTL::ComputeCommandEncoder*& enc, const Layer& L, std::size_t n) {
    const std::size_t E = cfg.num_experts, K = cfg.top_k, H = cfg.hidden, Fs = cfg.shared_intermediate, a = as();
    mark(enc, "moe_router_shared");
    proj_rows(enc, L.router, R(u_c), R(gates_c), n, H, E, 0);
    proj_rows(enc, L.sh_gate, R(u_c), R(shg_c), n, H, Fs, 1);
    proj_rows(enc, L.sh_up, R(u_c), R(shu_c), n, H, Fs, 2);
    proj_rows(enc, L.sh_router, R(u_c), R(sgate_c), n, H, 1, 3);
    barrier(enc);
    const std::size_t tpg = 32 * (((E + 3) / 4 + 31) / 32);
    const int norm = cfg.norm_topk_prob ? 1 : 0;
    const bool rb = fast('h');   // one launch over the rows (q4_route indexes rows by threadgroup; same bits)
    for (std::size_t i = 0; i < (rb ? 1 : n); ++i) {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{1017, {int64_t(E), int64_t(K), norm, int64_t(tpg), int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"q4_route", join({T(), I(long(E)), I(long(K)), I(norm), I(long(tpg))})};
      }));
      set(enc, R(gates_c, i * E * a), 0);
      set(enc, R(eidx_c, i * K * 4), 1);
      set(enc, R(escores_c, i * K * a), 2);
      enc->dispatchThreadgroups(MTL::Size(rb ? n : 1, 1, 1), MTL::Size(tpg, 1, 1));
    }
    enc->setComputePipelineState(ctx->pipeline("suc_swiglu", T()));
    set(enc, R(shg_c), 0);
    set(enc, R(shu_c), 1);
    set(enc, R(shh_c), 2);
    const uint32_t nf = uint32_t(n * Fs);
    bytes(enc, nf, 3);
    enc->dispatchThreads(MTL::Size(n * Fs, 1, 1), MTL::Size(std::min<std::size_t>(256, n * Fs), 1, 1));
    barrier(enc);
    if (prof) mark(enc, "moe_experts");   // (the profile's split is the host sync)
    else split_command_buffer(*ctx, cur_cb, enc, &split_gpu_s);   // the chunk's routing -> host
    std::vector<std::size_t> all(n);
    for (std::size_t i = 0; i < n; ++i) all[i] = i;
    if (!L.moe->streamed()) {
      experts_tiled(enc, L, all, n);
    } else {
      const ExpertStream& xs = *L.moe->stream();
      const auto* ids = static_cast<const uint32_t*>(eidx_c->contents());
      const std::size_t cap = std::max<std::size_t>(xs.max_distinct(L.moe->stream_layer()), K);
      std::vector<char> in_uni(E, 0);
      std::vector<std::size_t> group;
      std::vector<uint32_t> gids;
      std::size_t g = 0;
      for (std::size_t i0 = 0; i0 < n;) {
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
        L.moe->stream_in(gids.data(), gids.size());
        ++chunk_fetches;
        experts_tiled(enc, L, group, n);
        i0 = i;
      }
    }
    mark(enc, "moe_combine_shared");
    L.moe->encode_combine(*ctx, enc, R(escores_c), R(f_c), n, moe_scratch);
    proj_rows(enc, L.sh_down, R(shh_c), R(sh_c), n, Fs, H, 4);
    barrier(enc);
    for (std::size_t i = 0; i < (rb ? 1 : n); ++i) {
      enc->setComputePipelineState(ctx->pipeline(PsoKey{1018, {int64_t(opt.act)}}, [&]() {
        return std::pair<std::string, std::string>{"q4_shared_add", T()};
      }));
      set(enc, R(f_c, i * H * a), 0);
      set(enc, R(sgate_c, i * a), 1);
      set(enc, R(sh_c, i * H * a), 2);
      const uint32_t nh = uint32_t(H);
      bytes(enc, nh, 3);
      enc->dispatchThreads(MTL::Size(H, rb ? n : 1, 1), MTL::Size(std::min<std::size_t>(256, H), 1, 1));
    }
    barrier(enc);
  }

  // Layer-major chunk of n tokens in slots [0, n) at positions p0 ...; the last token's streams in hs.
  void encode_chunk(MTL::ComputeCommandEncoder*& enc, std::size_t n, std::size_t p0, bool with_logits) {
    const std::size_t H = cfg.hidden, W = cfg.dense ? cfg.hidden : cfg.stream(), a = as();
    // fast: every row in the ring (contiguous: row-batched launches), the last row copied to hs per layer
    const bool rb = fast('h');
    auto xin = [&](std::size_t l, std::size_t i) -> BufRef {
      return (i + 1 == n && !rb) ? R(hs, l * W * a) : R(ring, ((l % 2) * opt.chunk + i) * W * a);
    };
    auto to_hs = [&](std::size_t l) {
      if (rb) encode_copy_cols(*ctx, enc, xin(l, n - 1), R(hs, l * W * a), 1, W, W, W, opt.act);
    };
    for (std::size_t i = 0; i < n; ++i) encode_embed(enc, i, xin(0, i));
    to_hs(0);
    const std::size_t E = cfg.ple_dim, nple = cfg.ple_layer_ids.size();
    for (std::size_t l = 0; l < cfg.layers; ++l) {
      const Layer& L = layers[l];
      if (cfg.dense) {
        mark(enc, "dense_norm");
        if (rb) rms(enc, 0, xin(l, 0), &L.in_norm, R(u_c), n, H, H, H, 1);
        else
          for (std::size_t i = 0; i < n; ++i) rms(enc, 0, xin(l, i), &L.in_norm, R(u_c, i * H * a), 1, H, H, H, 1);
        barrier(enc);
        mixer_chunk(enc, L, p0, n);
        mark(enc, "dense_add_norm");
        if (rb) add(enc, xin(l, 0), R(f_c), R(xa_c), n * H);
        else
          for (std::size_t i = 0; i < n; ++i) add(enc, xin(l, i), R(f_c, i * H * a), R(xa_c, i * H * a), H);
        rms(enc, 0, R(xa_c), &L.post_norm, R(u_c), n, H, H, H, 1);
        barrier(enc);
        const std::size_t F = cfg.intermediate;
        mark(enc, "dense_mlp");
        proj_rows(enc, L.mlp_gate, R(u_c), R(shg_c), n, H, F, 0);
        proj_rows(enc, L.mlp_up, R(u_c), R(shu_c), n, H, F, 1);
        barrier(enc);
        enc->setComputePipelineState(ctx->pipeline("suc_swiglu", T()));
        set(enc, R(shg_c), 0);
        set(enc, R(shu_c), 1);
        set(enc, R(shh_c), 2);
        const uint32_t nf = uint32_t(n * F);
        bytes(enc, nf, 3);
        enc->dispatchThreads(MTL::Size(n * F, 1, 1), MTL::Size(std::min<std::size_t>(256, n * F), 1, 1));
        barrier(enc);
        proj_rows(enc, L.mlp_down, R(shh_c), R(f_c), n, F, H, 0);
        barrier(enc);
        mark(enc, "dense_add");
        if (rb) add(enc, R(xa_c), R(f_c), xin(l + 1, 0), n * H);
        else
          for (std::size_t i = 0; i < n; ++i) add(enc, R(xa_c, i * H * a), R(f_c, i * H * a), xin(l + 1, i), H);
        to_hs(l + 1);
        continue;
      }
      mark(enc, "ple");
      if (L.ple)   // the PLE conv state is sequential: token by token
        for (std::size_t i = 0; i < n; ++i)
          ple(enc, *L.ple, xin(l, i), R(ple_e, (i * nple + std::size_t(L.ple->index)) * E * a), R(pl_c, i * W * a));
      auto cur = [&](std::size_t i) { return L.ple ? R(pl_c, i * W * a) : xin(l, i); };
      auto xa_i = [&](std::size_t i) { return R(xa_c, i * W * a); };
      mark(enc, "hyper_attn");
      const BufRef curb = rb ? cur(0) : BufRef{}, xab = rb ? R(xa_c) : BufRef{}, outb = rb ? xin(l + 1, 0) : BufRef{};
      hyper_mix_rows(enc, L.ahc, cur, R(inj_c), n, curb);
      mixer_chunk(enc, L, p0, n);
      mark(enc, "inject_hyper_mlp");
      inject_rows(enc, cur, R(inj_c), xa_i, n, curb, xab);
      hyper_mix_rows(enc, L.mhc, xa_i, R(inj2_c), n, xab);
      moe_chunk(enc, L, n);
      mark(enc, "inject_mlp");
      inject_rows(enc, xa_i, R(inj2_c), [&](std::size_t i) { return xin(l + 1, i); }, n, xab, outb);
      to_hs(l + 1);
    }
    mark(enc, "logits");
    if (rows_out) {   // every row's logits (measurement runs): the head per row, then the last row as usual
      if (!logits_rows) logits_rows = alloc(opt.chunk * cfg.vocab * as(), nullptr);
      for (std::size_t i = 0; i < n; ++i) {
        barrier(enc);
        if (cfg.dense) {
          rms(enc, 0, xin(cfg.layers, i), &final_norm, R(u), 1, H, H, H, 1);
          barrier(enc);
        } else {
          hyper_mix(enc, final_mixer, xin(cfg.layers, i));
        }
        proj(enc, head, R(u), R(logits_rows, i * cfg.vocab * as()), 0);
      }
      barrier(enc);
    }
    if (with_logits) encode_logits(enc, xin(cfg.layers, n - 1));
  }

  // ---- n-gram tables (host)
  void load_ngram() {
    const Suc1File& f = mf->file();
    for (std::size_t l = 0; l < cfg.layers; ++l) {
      const int pi = cfg.ple_index(l);
      if (pi < 0) continue;
      NgramLayer g;
      g.layer = l;
      g.index = std::size_t(pi);
      const std::string pre = "model.layers." + std::to_string(l) + ".ple.ple_embedding.";
      // per-head prime sizes / offsets (HF: the consecutive primes >= ngram_vocab_size_base) and multipliers
      const std::size_t heads = cfg.ngram_heads();
      std::vector<uint64_t> primes;
      for (uint64_t v = cfg.ngram_vocab_size_base; primes.size() < (g.index + 1) * heads; ++v)
        if (is_prime(v)) primes.push_back(v);
      g.sizes.assign(primes.begin() + std::ptrdiff_t(g.index * heads), primes.end());
      uint64_t total = 0;
      for (uint64_t s : g.sizes) g.offsets.push_back(total), total += s;
      const uint64_t padded = (total + cfg.ngram_divisible_by - 1) / cfg.ngram_divisible_by * cfg.ngram_divisible_by;
      const uint64_t max_long = (uint64_t(1) << 63) - 1;
      const uint64_t half = std::max<uint64_t>(1, (max_long / std::max<uint64_t>(cfg.vocab, 1)) / 2);
      const uint64_t base_seed = cfg.seed + 10007ull * g.index;
      for (std::size_t i = 0; i < cfg.ngram_size; ++i)
        g.mult.push_back(2 * (splitmix64(base_seed + 0x9E3779B97F4A7C15ull * (i + 1)) % half) + 1);
      // the stored hash buffers (when present) must agree with the recomputation
      auto check = [&](const std::string& name, const std::vector<uint64_t>& want) {
        const TensorDesc* t = f.find_tensor(pre + name);
        if (!t) return;
        const ArrayRecord& r = t->arrays.at(0);
        if (r.logical_numel() != want.size()) fail(pre + name + ": size differs from the configuration");
        const auto b = f.payload(r);
        for (std::size_t i = 0; i < want.size(); ++i)
          if (uint64_t(load_as_i64(b.data() + i * dtype_size(r.dtype), r.dtype)) != want[i])
            fail(pre + name + ": stored value differs from the recomputed hash constant");
      };
      check("layer_multipliers", g.mult);
      check("ngram_heads_vocab_sizes", g.sizes);
      check("ngram_heads_offsets", g.offsets);
      g.dim = cfg.ple_dim / heads;
      uint64_t rows = 0;
      g.starts.push_back(0);
      for (std::size_t j = 0;; ++j) {
        const std::string n = pre + "ngram_embedding.shard_" + std::to_string(j) + ".weight";
        const TensorDesc* t = f.find_tensor(n);
        if (!t) break;
        if (t->shape.size() != 2 || t->shape[1] != g.dim) fail(n + ": shard shape");
        Shard s;
        s.rows = t->shape[0];
        if (t->representation_type == "raw" || t->representation_type == "bf16") {
          const ArrayRecord& r = t->arrays.at(0);
          s.rdtype = r.dtype;
          msl_of(r.dtype);
          s.raw = f.payload(r).data();
        } else if (t->representation_type == "affine" && !t->has_residual() &&
                   (*t->params)["group_size"].as_uint() == g.dim) {
          s.affine = true;
          s.bits = unsigned((*t->params)["bits"].as_uint());
          s.wpr = words_per_row(g.dim, s.bits);
          const ArrayRecord& qi = t->array("indices");
          const ArrayRecord& sc = t->array("scales");
          const ArrayRecord& bi = t->array("biases");
          if (!qi.packed || qi.bits != s.bits || qi.shape != std::vector<uint64_t>{s.rows, s.wpr}) fail(n + ": row-affine indices");
          if (sc.dtype != bi.dtype || sc.logical_numel() != s.rows || bi.logical_numel() != s.rows) fail(n + ": row-affine scales");
          if (sc.dtype != DType::F16 && sc.dtype != DType::BF16) fail(n + ": row-affine scale dtype");
          s.sdtype = sc.dtype;
          s.words = f.payload(qi).data();
          s.scales = f.payload(sc).data();
          s.biases = f.payload(bi).data();
        } else {
          fail(n + ": n-gram shard stored as " + t->representation_type + "; expected raw or row-wise affine");
        }
        if (j > 0 && s.affine != g.shards[0].affine) fail("mixed raw / row-affine n-gram shards");
        g.quantized = s.affine;
        g.table_bytes += t->compressed_size;
        rows += s.rows;
        g.starts.push_back(rows);
        g.shards.push_back(s);
      }
      if (g.shards.empty()) fail("layer " + std::to_string(l) + ": no n-gram table shards");
      if (rows != padded) fail("layer " + std::to_string(l) + ": n-gram table has " + std::to_string(rows) + " rows, expected " + std::to_string(padded));
      g.ctx.assign(cfg.ngram_size - 1, cfg.eos);
      ngram.push_back(std::move(g));
    }
  }

  // Row ids (ngram_heads) of `t` after the context (HF Qwen4ExpTextNGramEmbedding for one token), context advanced.
  std::vector<uint64_t> ngram_ids(NgramLayer& g, uint32_t t) const {
    const std::size_t N = cfg.ngram_size;
    std::vector<uint64_t> hist(g.ctx);
    hist.push_back(t);
    const std::size_t p = N - 1;
    long prev = -1;   // last EOS strictly before p
    for (std::size_t i = 0; i < p; ++i)
      if (hist[i] == cfg.eos) prev = long(i);
    const long in_seg = long(p) - (prev + 1);
    std::vector<uint64_t> sh(N);
    for (std::size_t s = 0; s < N; ++s) sh[s] = (in_seg >= long(s) && p >= s) ? hist[p - s] : uint64_t(cfg.eos);
    std::vector<uint64_t> ids;
    for (std::size_t n = 2; n <= N; ++n) {
      uint64_t mix = sh[0] * g.mult[0];
      for (std::size_t q = 1; q < n; ++q) mix ^= sh[q] * g.mult[q];
      for (std::size_t h = (n - 2) * cfg.heads_per_ngram; h < (n - 1) * cfg.heads_per_ngram; ++h)
        ids.push_back(mix % g.sizes[h] + g.offsets[h]);
    }
    return ids;
  }

  // Gather the rows of `ids` into dst (ngram_heads · dim activations), reading only those rows of the mapping.
  void gather_rows(const NgramLayer& g, const std::vector<uint64_t>& ids, uint8_t* dst) const {
    const std::size_t dim = g.dim, a = as();
    std::vector<float> row(dim);
    for (std::size_t h = 0; h < ids.size(); ++h) {
      const uint64_t id = ids[h];
      if (id >= g.starts.back()) fail("n-gram row id out of range");
      const std::size_t j = std::size_t(std::upper_bound(g.starts.begin(), g.starts.end(), id) - g.starts.begin()) - 1;
      const Shard& s = g.shards[j];
      const uint64_t r = id - g.starts[j];
      uint8_t* out = dst + h * dim * a;
      if (!s.affine) {
        const uint8_t* src = s.raw + r * dim * dtype_size(s.rdtype);
        if ((s.rdtype == DType::BF16 && opt.act == Act::BF16) || (s.rdtype == DType::F16 && opt.act == Act::F16) ||
            (s.rdtype == DType::F32 && opt.act == Act::F32)) {
          std::memcpy(out, src, dim * a);
          continue;
        }
        for (std::size_t i = 0; i < dim; ++i) row[i] = load_as_f32(src + i * dtype_size(s.rdtype), s.rdtype);
      } else {   // suc.quant.row_affine.dequantize_rows: q · s + b in float32 (two roundings)
        const uint8_t* w = s.words + r * s.wpr * 4;
        uint16_t sb, bb;
        std::memcpy(&sb, s.scales + r * 2, 2);
        std::memcpy(&bb, s.biases + r * 2, 2);
        const float sc = s.sdtype == DType::F16 ? f16_to_f32(sb) : bf16_to_f32(sb);
        const float bi = s.sdtype == DType::F16 ? f16_to_f32(bb) : bf16_to_f32(bb);
        for (std::size_t i = 0; i < dim; ++i) {
          const float q = float(unpack_one(w, s.bits, i));
          const float t = q * sc;
          row[i] = t + bi;
        }
      }
      f32_to_act(row.data(), out, dim, opt.act);
    }
  }

  void zero_states() {
    for (Layer& L : layers) {
      if (L.lin.conv_state) std::memset(L.lin.conv_state->contents(), 0, L.lin.conv_state->length());
      if (L.lin.rec_state) std::memset(L.lin.rec_state->contents(), 0, L.lin.rec_state->length());
      if (L.ple) std::memset(L.ple->state->contents(), 0, L.ple->state->length());
    }
    for (NgramLayer& g : ngram) g.ctx.assign(cfg.ngram_size - 1, cfg.eos);
    pos = 0;
    have_logits = false;
  }
};

// ------------------------------------------------------------------------------------------ Qwen4Model

std::vector<std::pair<uint64_t, uint64_t>> qwen4_table_ranges(const Suc1File& f) {
  std::vector<std::pair<uint64_t, uint64_t>> r;
  for (const TensorDesc& t : f.tensors())
    if (t.name.find(".ngram_embedding.shard_") != std::string::npos)
      for (const ArrayRecord& a : t.arrays) r.push_back({a.offset, a.offset + a.nbytes});
  return r;
}

Qwen4Model::Qwen4Model(Context& ctx, const MappedFile& mf, const Qwen4Options& opt) : p_(std::make_unique<Impl>()) {
  Pool pool;
  Impl& m = *p_;
  m.ctx = &ctx;
  m.mf = &mf;
  m.opt = opt;
  if (const char* e = std::getenv("SUC_Q4_PROFILE")) m.prof = e[0] && e[0] != '0';
  const Json* arch = mf.file().directory().find("architecture");
  if (!arch || !arch->is_object()) throw FormatError("directory has no architecture object");
  m.cfg = Qwen4Config::from_architecture(*arch);
  const Qwen4Config& c = m.cfg;
  if (opt.max_context == 0 || opt.chunk == 0) fail("max_context and chunk must be > 0");
  const std::size_t H = c.hidden, W = c.stream(), D = c.head_dim, Hq = c.heads, Hkv = c.kv_heads;
  const std::size_t Hk = c.lin_k_heads, Hv = c.lin_v_heads, Dk = c.lin_k_dim, Dv = c.lin_v_dim;
  const std::size_t key_dim = Hk * Dk, value_dim = Hv * Dv, conv_dim = 2 * key_dim + value_dim;
  const std::size_t a = m.as();

  m.embed = m.mat("model.embed_tokens.weight", c.vocab, H, false);
  m.head = m.mat("lm_head.weight", c.vocab, H, true);
  if (c.dense) m.final_norm = m.vec("model.norm.weight", H);
  else m.final_mixer = m.hyper("model.hyper_connection_mixer", false);
  m.layers.resize(c.layers);
  for (std::size_t i = 0; i < c.layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i);
    Impl::Layer& L = m.layers[i];
    L.full = c.full_attention[i];
    if (c.dense) {
      L.in_norm = m.vec(p + ".input_layernorm.weight", H);
      L.post_norm = m.vec(p + ".post_attention_layernorm.weight", H);
      L.mlp_gate = m.mat(p + ".mlp.gate_proj.weight", c.intermediate, H);
      L.mlp_up = m.mat(p + ".mlp.up_proj.weight", c.intermediate, H);
      L.mlp_down = m.mat(p + ".mlp.down_proj.weight", H, c.intermediate);
    } else {
      L.ahc = m.hyper(p + ".attn_hyper_connection", true);
      L.mhc = m.hyper(p + ".mlp_hyper_connection", true);
    }
    if (L.full) {
      Impl::Attn& A = L.att;
      A.q = m.mat(p + ".self_attn.q_proj.weight", 2 * Hq * D, H);
      A.k = m.mat(p + ".self_attn.k_proj.weight", Hkv * D, H);
      A.v = m.mat(p + ".self_attn.v_proj.weight", Hkv * D, H);
      A.o = m.mat(p + ".self_attn.o_proj.weight", H, Hq * D);
      A.q_norm = m.vec(p + ".self_attn.q_norm.weight", D);
      A.k_norm = m.vec(p + ".self_attn.k_norm.weight", D);
      const std::size_t kv = opt.max_context * Hkv * D * a;
      A.kcache = m.alloc(kv, &m.kv_bytes);
      A.vcache = m.alloc(kv, &m.kv_bytes);
      if (c.indexer) {
        A.iqk = m.mat(p + ".self_attn.indexer.index_qk_proj.weight", (c.idx_heads + 1) * c.idx_dim, H);
        A.iq_norm = m.vec(p + ".self_attn.indexer.q_layernorm.weight", c.idx_dim);
        A.ik_norm = m.vec(p + ".self_attn.indexer.k_layernorm.weight", c.idx_dim);
        A.ikeys = m.alloc(opt.max_context * c.idx_dim * a, &m.kv_bytes);
        A.bkeys = m.alloc((opt.max_context / c.idx_ratio + 1) * c.idx_dim * a, &m.kv_bytes);
      }
    } else {
      Impl::LinAttn& La = L.lin;
      La.qkv = m.mat(p + ".linear_attn.in_proj_qkv.weight", conv_dim, H);
      La.z = m.mat(p + ".linear_attn.in_proj_z.weight", value_dim, H);
      La.a = m.mat(p + ".linear_attn.in_proj_a.weight", Hv, H);
      La.b = m.mat(p + ".linear_attn.in_proj_b.weight", Hv, H);
      La.out = m.mat(p + ".linear_attn.out_proj.weight", H, value_dim);
      La.conv = m.vec(p + ".linear_attn.conv1d.weight", conv_dim * c.conv_kernel);
      La.a_log = m.vec(p + ".linear_attn.A_log", Hv);
      La.dt_bias = m.vec(p + ".linear_attn.dt_bias", Hv);
      La.norm = m.vec(p + ".linear_attn.norm.weight", Dv);
      La.conv_state = m.alloc((c.conv_kernel - 1) * conv_dim * a, &m.state_bytes);
      La.rec_state = m.alloc(Hv * Dv * Dk * 4, &m.state_bytes);
    }
    if (const int pi = c.ple_index(i); pi >= 0) {
      L.ple = std::make_unique<Impl::Ple>();
      Impl::Ple& P = *L.ple;
      P.index = pi;
      P.key = m.mat(p + ".ple.key_proj.weight", W, c.ple_dim);
      P.value = m.mat(p + ".ple.value_proj.weight", H, c.ple_dim);
      P.norm_key = m.vec(p + ".ple.norm_key.weight", W);
      P.norm_query = m.vec(p + ".ple.norm_query.weight", W);
      P.norm_conv = m.vec(p + ".ple.norm_conv.weight", W);
      P.conv = m.vec(p + ".ple.conv1d.weight", W * c.ple_conv_kernel);
      P.state = m.alloc((c.ple_conv_kernel - 1) * c.ngram_size * W * a, &m.state_bytes);
    }
    if (c.dense) continue;
    L.router = m.mat(p + ".mlp.gate.weight", c.num_experts, H);
    L.sh_gate = m.mat(p + ".mlp.shared_expert.gate_proj.weight", c.shared_intermediate, H);
    L.sh_up = m.mat(p + ".mlp.shared_expert.up_proj.weight", c.shared_intermediate, H);
    L.sh_down = m.mat(p + ".mlp.shared_expert.down_proj.weight", H, c.shared_intermediate);
    L.sh_router = m.mat(p + ".mlp.shared_expert_gate.weight", 1, H);
    L.moe = std::make_unique<MoeBlock>(MoeBlock::from_file(
        ctx, mf, p, MoeConfig{H, c.num_experts, c.top_k, c.moe_intermediate, c.norm_topk_prob}, opt.act,
        opt.expert_stream ? StackLayout::Streamed : StackLayout::Auto));
    if (opt.expert_stream) L.moe->attach_stream(*opt.expert_stream);
    m.copied_bytes += L.moe->copied_bytes();
  }
  if (opt.expert_stream) {
    opt.expert_stream->finalize();
    if (opt.expert_stream->min_max_distinct() < c.top_k) fail("expert stream: fewer unpinned slots than top_k");
  }
  m.load_ngram();
  // the GPU views must not cover the tables (only whole pages can be left out: a table's first / last page may be
  // shared with a neighbouring payload)
  {
    const std::size_t page = std::size_t(::getpagesize());
    for (auto [b, e] : qwen4_table_ranges(mf.file())) {
      const uint64_t pb = (b + page - 1) / page * page, pe = e / page * page;
      if (pb < pe && mf.overlaps(pb, pe))
        fail("the mapped GPU views cover the n-gram tables; construct the MappedFile with qwen4_table_ranges(file) "
             "excluded (Metal would make the whole table resident)");
    }
  }

  // MLX scalar conventions: Python float scalars meet T arrays as T values (weak typing)
  m.eps = float(c.rms_norm_eps);
  m.rope_base_log2 = std::log2(float(c.rope_theta));
  m.attn_scale = float(std::pow(double(D), -0.5));
  const double inv = std::pow(double(Dk), -0.5);
  m.q_scale = round_to_act(float(inv * inv), opt.act);
  m.k_scale = round_to_act(float(inv), opt.act);
  m.ple_rsh = round_to_act(float(std::sqrt(double(H))), opt.act);
  m.ple_floor = round_to_act(1e-6f, opt.act);

  uint64_t* ab = &m.arena_bytes;
  const std::size_t nple = c.ple_layer_ids.size();
  m.tok = m.alloc(opt.chunk * c.hc * 4, ab);
  m.ple_e = m.alloc(opt.chunk * std::max<std::size_t>(nple, 1) * c.ple_dim * a, ab);
  m.hs = m.alloc((c.layers + 1) * W * a, ab);
  m.xa = m.alloc(W * a, ab);
  m.xn = m.alloc(W * a, ab);
  m.dbuf = m.alloc(c.hc_lowrank * a, ab);
  m.sbuf = m.alloc(c.hc_lowrank * a, ab);
  m.up = m.alloc(W * a, ab);
  m.inj = m.alloc(c.hc * a, ab);
  m.u = m.alloc(H * a, ab);
  m.f = m.alloc(H * a, ab);
  m.qkv = m.alloc(conv_dim * a, ab);
  m.z = m.alloc(value_dim * a, ab);
  m.av = m.alloc(Hv * a, ab);
  m.bv = m.alloc(Hv * a, ab);
  m.conv_out = m.alloc(conv_dim * a, ab);
  m.qn = m.alloc(key_dim * a, ab);
  m.kn = m.alloc(std::max(key_dim, Hkv * D) * a, ab);
  m.gdec = m.alloc(Hv * 4, ab);
  m.beta = m.alloc(Hv * a, ab);
  m.yd = m.alloc(value_dim * a, ab);
  m.yn = m.alloc(value_dim * a, ab);
  m.qg = m.alloc(2 * Hq * D * a, ab);
  m.q = m.alloc(Hq * D * a, ab);
  m.kb = m.alloc(Hkv * D * a, ab);
  m.att = m.alloc(Hq * D * a, ab);
  m.max_blocks = c.indexer ? opt.max_context / c.idx_ratio + 1 : 1;
  m.iqk = m.alloc((c.idx_heads + 1) * c.idx_dim * a, ab);
  m.iq = m.alloc(c.idx_heads * c.idx_dim * a, ab);
  m.scores = m.alloc(m.max_blocks * 4, ab);
  m.sel = m.alloc(m.max_blocks, ab);
  m.sc = m.alloc(Hq * opt.max_context * 4, ab);
  m.gates = m.alloc(c.num_experts * a, ab);
  m.eidx = m.alloc(c.top_k * 4, ab);
  m.escores = m.alloc(c.top_k * a, ab);
  if (opt.expert_stream && opt.expert_stream->options().prefetch) {
    m.pgates = m.alloc(c.num_experts * a, ab);
    m.peidx = m.alloc(c.top_k * 4, ab);
    m.pscores = m.alloc(c.top_k * a, ab);
  }
  const std::size_t ffn = std::max(c.shared_intermediate, c.intermediate);
  m.shg = m.alloc(ffn * a, ab);
  m.shu = m.alloc(ffn * a, ab);
  m.shh = m.alloc(ffn * a, ab);
  m.sh = m.alloc(H * a, ab);
  m.sgate = m.alloc(a, ab);
  m.kraw = m.alloc(W * a, ab);
  m.kk = m.alloc(W * a, ab);
  m.pv = m.alloc(H * a, ab);
  m.qq = m.alloc(W * a, ab);
  m.pg = m.alloc(W * a, ab);
  m.gn = m.alloc(W * a, ab);
  m.logits = m.alloc(c.vocab * a, ab);
  m.amax = m.alloc(4, ab);
  if (opt.layer_prefill && opt.chunk > 1) {   // batched prefill chunks (tile plan buffers grow on demand)
    const std::size_t n = opt.chunk, Ws = c.dense ? H : W, iw = (c.idx_heads + 1) * c.idx_dim;
    m.ring = m.alloc(2 * n * Ws * a, ab);
    m.pl_c = m.alloc(n * Ws * a, ab);
    m.xa_c = m.alloc(n * Ws * a, ab);
    m.u_c = m.alloc(n * H * a, ab);
    m.inj_c = m.alloc(n * c.hc * a, ab);
    m.inj2_c = m.alloc(n * c.hc * a, ab);
    m.qkv_c = m.alloc(n * conv_dim * a, ab);
    m.z_c = m.alloc(n * value_dim * a, ab);
    m.av_c = m.alloc(n * Hv * a, ab);
    m.bv_c = m.alloc(n * Hv * a, ab);
    m.yn_c = m.alloc(n * value_dim * a, ab);
    m.conv_c = m.alloc(n * conv_dim * a, ab);
    m.qn_c = m.alloc(n * key_dim * a, ab);
    m.kn_c = m.alloc(n * key_dim * a, ab);
    m.gdec_c = m.alloc(n * Hv * 4, ab);
    m.beta_c = m.alloc(n * Hv * a, ab);
    m.yd_c = m.alloc(n * value_dim * a, ab);
    m.qg_c = m.alloc(n * 2 * Hq * D * a, ab);
    m.kb_c = m.alloc(n * Hkv * D * a, ab);
    m.iqk_c = m.alloc(n * iw * a, ab);
    m.att_c = m.alloc(n * Hq * D * a, ab);
    m.f_c = m.alloc(n * H * a, ab);
    m.gates_c = m.alloc(n * std::max<std::size_t>(c.num_experts, 1) * a, ab);
    m.eidx_c = m.alloc(n * std::max<std::size_t>(c.top_k, 1) * 4, ab);
    m.escores_c = m.alloc(n * std::max<std::size_t>(c.top_k, 1) * a, ab);
    m.shg_c = m.alloc(n * ffn * a, ab);
    m.shu_c = m.alloc(n * ffn * a, ab);
    m.shh_c = m.alloc(n * ffn * a, ab);
    m.sh_c = m.alloc(n * H * a, ab);
    m.sgate_c = m.alloc(n * a, ab);
    if (!c.dense) {
      m.hx_c = m.alloc(n * W * a, ab);
      m.hd_c = m.alloc(n * c.hc_lowrank * a, ab);
      m.hs_c = m.alloc(n * c.hc_lowrank * a, ab);
      m.hu_c = m.alloc(n * W * a, ab);
    }
  }
}

Qwen4Model::~Qwen4Model() = default;

const Qwen4Config& Qwen4Model::config() const { return p_->cfg; }
uint64_t Qwen4Model::chunk_fetches() const { return p_->chunk_fetches; }
std::size_t Qwen4Model::position() const { return p_->pos; }
void Qwen4Model::reset() { p_->zero_states(); }
double Qwen4Model::last_gpu_seconds() const { return p_->gpu_s; }
double Qwen4Model::last_encode_seconds() const { return p_->encode_s; }
double Qwen4Model::last_ngram_seconds() const { return p_->ngram_s; }

std::vector<float> Qwen4Model::forward_rows(const std::vector<uint32_t>& tokens) {
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

uint32_t Qwen4Model::forward(const std::vector<uint32_t>& tokens, bool with_logits) {
  Impl& m = *p_;
  const Qwen4Config& c = m.cfg;
  if (tokens.empty()) fail("forward: no tokens");
  if (m.pos + tokens.size() > m.opt.max_context)
    fail("context full (max_context " + std::to_string(m.opt.max_context) + ")");
  for (uint32_t t : tokens)
    if (t >= c.vocab) fail("token id " + std::to_string(t) + " >= vocab " + std::to_string(c.vocab));
  const std::size_t a = m.as(), nple = c.ple_layer_ids.size();
  uint32_t result = 0;
  m.encode_s = m.gpu_s = m.ngram_s = 0.0;
  for (std::size_t s0 = 0; s0 < tokens.size(); s0 += m.opt.chunk) {
    const std::size_t n = std::min(m.opt.chunk, tokens.size() - s0);
    const bool last_chunk = s0 + n == tokens.size();
    Pool pool;
    // host: token copies and n-gram rows of every step of the chunk
    const auto t_ng = std::chrono::steady_clock::now();
    auto* tk = static_cast<uint32_t*>(m.tok->contents());
    auto* pe = static_cast<uint8_t*>(m.ple_e->contents());
    // the row ids in token order (each token's n-gram context), then the row gathers of all tokens in parallel (pure
    // copies of mapped rows: on a cold page cache these are random SSD reads, which overlap across threads)
    std::vector<std::vector<uint64_t>> ng_ids(n * m.ngram.size());
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t h = 0; h < c.hc; ++h) tk[i * c.hc + h] = tokens[s0 + i];
      for (std::size_t gi = 0; gi < m.ngram.size(); ++gi) {
        NgramLayer& g = m.ngram[gi];
        ng_ids[i * m.ngram.size() + gi] = m.ngram_ids(g, tokens[s0 + i]);
        g.ctx.erase(g.ctx.begin());
        g.ctx.push_back(tokens[s0 + i]);
      }
    }
    parallel_for(n, 4, [&](std::size_t i) {
      for (std::size_t gi = 0; gi < m.ngram.size(); ++gi) {
        const NgramLayer& g = m.ngram[gi];
        m.gather_rows(g, ng_ids[i * m.ngram.size() + gi], pe + (i * nple + g.index) * c.ple_dim * a);
      }
    });
    m.ngram_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ng).count();
    const auto t_enc = std::chrono::steady_clock::now();
    MTL::CommandBuffer* cb = m.ctx->queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
    m.cur_cb = cb;
    m.split_gpu_s = 0.0;
    if (m.ring && n > 1) {
      if (m.prof) m.prof_label = "embed";
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
  if (m.prof && !m.prof_acc.empty()) {
    double tot = 0.0;
    for (const auto& [k, v] : m.prof_acc) tot += v.first;
    std::fprintf(stderr, "qwen4 prefill profile (GPU s, command buffers): total %.3f\n", tot);
    for (const auto& [k, v] : m.prof_acc)
      std::fprintf(stderr, "  %-22s %8.3f s %5.1f %%  %zu\n", k.c_str(), v.first, 100.0 * v.first / tot, v.second);
    m.prof_acc.clear();
  }
  m.have_logits = with_logits;
  if (with_logits) result = *static_cast<const uint32_t*>(m.amax->contents());
  return result;
}

std::vector<float> Qwen4Model::logits() const {
  if (!p_->have_logits) return {};
  std::vector<float> v(p_->cfg.vocab);
  act_to_f32(p_->logits->contents(), v.data(), v.size(), p_->opt.act);
  return v;
}

std::vector<float> Qwen4Model::hidden_states() const {
  std::vector<float> v((p_->cfg.layers + 1) * p_->cfg.stream());
  act_to_f32(p_->hs->contents(), v.data(), v.size(), p_->opt.act);
  return v;
}

Qwen4Memory Qwen4Model::memory() const {
  Qwen4Memory r;
  r.mapped_file = p_->mf->mapped_bytes();
  for (const NgramLayer& g : p_->ngram) r.ngram_tables += g.table_bytes;
  r.copied_weights = p_->copied_bytes;
  r.kv_cache = p_->kv_bytes;
  r.states = p_->state_bytes;
  r.arena = p_->arena_bytes;
  uint64_t ws = 0;
  for (const Workspace& w : p_->ws) ws += w.xr_bytes + w.y32_bytes + w.y32b_bytes;
  r.workspace = ws + p_->moe_scratch.bytes();
  return r;
}

std::size_t Qwen4Model::suc_projections() const {
  std::size_t n = 0;
  auto add = [&](const Mat& m) { n += m.kind == Mat::Suc; };
  for (const auto& L : p_->layers) {
    for (const Mat* m : {&L.lin.qkv, &L.lin.z, &L.lin.a, &L.lin.b, &L.lin.out, &L.att.q, &L.att.k, &L.att.v, &L.att.o,
                         &L.att.iqk, &L.sh_gate, &L.sh_up, &L.sh_down, &L.router, &L.sh_router, &L.mlp_gate,
                         &L.mlp_up, &L.mlp_down})
      add(*m);
    if (L.ple) add(L.ple->key), add(L.ple->value);
  }
  return n;
}

std::size_t Qwen4Model::expert_stacks() const { return p_->cfg.dense ? 0 : 3 * p_->layers.size(); }
bool Qwen4Model::ngram_quantized() const { return !p_->ngram.empty() && p_->ngram[0].quantized; }

}  // namespace suc::metal
