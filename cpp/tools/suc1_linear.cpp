// suc1-linear: run one or all SUC linear tensors of a SUC1 file on the GPU (Metal backend, zero-copy over the
// mapped file) and compare y = x Ŵᵀ with the CPU reference (decode_tensor -> Ŵ float32, matmul in float64).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "suc/decode.hpp"
#include "suc/metal.hpp"
#include "suc/npy.hpp"
#include "suc/parallel.hpp"

using namespace suc;
using namespace suc::metal;

static void usage() {
  std::fprintf(stderr,
               "usage: suc1-linear FILE (--all | --tensor NAME ...) [--batches 1,4,16] [--dtype f16|bf16|f32]\n"
               "                   [--path auto|vec|mma] [--seed N] [--no-verify] [--json] [--filter S]\n"
               "                   [--x X.npy --y-out Y.npy] [--codebook-in auto|device|threadgroup]\n"
               "                   [--mma-min-batch N] [--tuning-cache PATH|-] [--fast-math] [--max-view-bytes N] [--rows-exact]\n"
               "  Runs y = x W^T on the GPU for random x (normal, rounded to --dtype) at each batch size and compares\n"
               "  with the CPU reference decoder (W decoded to float32, matmul in float64): max abs, mean abs,\n"
               "  relative L2 error, max abs / max |ref|, and the error of rounding the reference itself to --dtype.\n"
               "  --x/--y-out: one tensor, x (B, in) float32 .npy instead of random; writes y (B, out) float32.\n"
               "  --json: one JSON object per (tensor, batch) line.\n"
               "  --max-view-bytes: map the file as overlapping buffers of at most N bytes (multi-view test).\n"
               "  --rows-exact: run Linear::encode_rows_exact (every row bit-identical to its one-row launch).\n");
}

static std::vector<std::size_t> parse_list(const char* s) {
  std::vector<std::size_t> v;
  for (const char* p = s; *p;) {
    v.push_back(std::strtoull(p, const_cast<char**>(&p), 10));
    if (*p == ',') ++p;
    else if (*p) throw std::invalid_argument("bad list");
  }
  return v;
}

static float round_act(float v, Act a) {
  float o = v;
  if (a == Act::F16) o = f16_to_f32(f32_to_f16_bits(v));
  if (a == Act::BF16) o = bf16_to_f32(f32_to_bf16_bits(v));
  return o;
}

struct Metrics {
  double max_abs = 0, mean_abs = 0, rel_l2 = 0, max_rel = 0, ref_round_rel_l2 = 0, ref_max = 0;
  bool finite = true;
};

static Metrics compare(const std::vector<float>& y, const std::vector<double>& ref, Act act) {
  Metrics m;
  double se = 0, sr = 0, sq = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = std::fabs(double(y[i]) - ref[i]);
    if (!std::isfinite(d)) m.finite = false;
    m.max_abs = std::max(m.max_abs, d);
    m.mean_abs += d;
    se += d * d;
    sr += ref[i] * ref[i];
    m.ref_max = std::max(m.ref_max, std::fabs(ref[i]));
    const double q = double(round_act(float(ref[i]), act)) - ref[i];
    sq += q * q;
  }
  m.mean_abs /= double(std::max<std::size_t>(1, ref.size()));
  m.rel_l2 = sr > 0 ? std::sqrt(se / sr) : std::sqrt(se);
  m.ref_round_rel_l2 = sr > 0 ? std::sqrt(sq / sr) : 0;
  m.max_rel = m.ref_max > 0 ? m.max_abs / m.ref_max : m.max_abs;
  return m;
}

static std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o;
}

int main(int argc, char** argv) {
  std::string path, filter, x_npy, y_out, path_name = "auto";
  std::vector<std::string> names;
  std::vector<std::size_t> batches = {1, 4, 16};
  bool all = false, json = false, rows_exact = false;
  Act act = Act::F16;
  uint64_t seed = 0, max_view = 0;
  OpenOptions opt;
  Tuning tu;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* flag) { return !std::strcmp(argv[i], flag) && i + 1 < argc; };
    if (!std::strcmp(argv[i], "--all")) all = true;
    else if (arg("--tensor")) names.push_back(argv[++i]);
    else if (arg("--batches")) batches = parse_list(argv[++i]);
    else if (arg("--dtype")) act = parse_act(argv[++i]);
    else if (arg("--path")) path_name = argv[++i];
    else if (arg("--seed")) seed = std::strtoull(argv[++i], nullptr, 10);
    else if (arg("--filter")) filter = argv[++i];
    else if (arg("--x")) x_npy = argv[++i];
    else if (arg("--y-out")) y_out = argv[++i];
    else if (arg("--codebook-in")) tu.codebook_in = argv[++i];
    else if (arg("--mma-min-batch")) tu.mma_min_batch = std::atoi(argv[++i]);
    else if (arg("--tuning-cache")) tu.tuning_cache = argv[++i];
    else if (arg("--max-view-bytes")) max_view = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--fast-math")) tu.fast_math = true;
    else if (!std::strcmp(argv[i], "--no-verify")) opt.verify_checksum = false;
    else if (!std::strcmp(argv[i], "--json")) json = true;
    else if (!std::strcmp(argv[i], "--rows-exact")) rows_exact = true;
    else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) { usage(); return 0; }
    else if (path.empty() && argv[i][0] != '-') path = argv[i];
    else { usage(); return 2; }
  }
  if (path.empty() || (all == !names.empty()) || (!x_npy.empty() && names.size() != 1)) { usage(); return 2; }
  const Path gpu_path = path_name == "vec" ? Path::Vec : path_name == "mma" ? Path::Mma : Path::Auto;
  if (path_name != "auto" && path_name != "vec" && path_name != "mma") { usage(); return 2; }
  try {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    Suc1File f = Suc1File::open(path, opt);
    Context ctx(tu);
    MappedFile mf(ctx, f, max_view);
    const double open_s = std::chrono::duration<double>(clk::now() - t0).count();
    if (!json)
      std::printf("# %s: %zu tensors, %llu bytes mapped zero-copy in %zu buffer(s); device %s (%s); open %.2f s; "
                  "dtype %s; tuning cache %s\n", path.c_str(), f.tensors().size(),
                  (unsigned long long)mf.mapped_bytes(), mf.views(), ctx.device_name().c_str(),
                  ctx.architecture().c_str(), open_s, std::string(act_name(act)).c_str(),
                  ctx.tuning_cache_found() ? "found" : "absent");
    std::vector<const TensorDesc*> todo;
    if (all) {
      for (const auto& t : f.tensors())
        if (filter.empty() || t.name.find(filter) != std::string::npos) todo.push_back(&t);
    } else {
      for (const auto& n : names) todo.push_back(&f.tensor(n));
    }
    if (!json)
      std::printf("%-44s %11s %3s %-9s %-4s %10s %10s %10s %10s %10s\n", "tensor", "shape", "B", "codec", "path",
                  "max_abs", "mean_abs", "rel_l2", "max_rel", "round_rel");
    std::size_t n_run = 0, n_skip = 0;
    double worst_rel = 0;
    for (const TensorDesc* t : todo) {
      std::string why;
      if (!Linear::supported(*t, &why)) {
        ++n_skip;
        if (!all) std::fprintf(stderr, "%s: not supported on the GPU: %s\n", t->name.c_str(), why.c_str());
        continue;
      }
      const Linear L = Linear::from_file(ctx, mf, *t);
      const DecodedTensor W = decode_tensor(f, *t);
      const std::size_t out = L.out_features(), in = L.in_features();
      std::vector<std::size_t> bs = batches;
      std::vector<float> x_given;
      if (!x_npy.empty()) {
        std::vector<uint64_t> shape;
        x_given = read_npy_f32(x_npy, shape);
        if (shape.size() != 2 || shape[1] != in) throw std::runtime_error("--x must be (B, " + std::to_string(in) + ")");
        bs = {std::size_t(shape[0])};
      }
      std::string codec = t->representation_type + (L.num_codebooks() > 1 ? std::to_string(L.num_codebooks()) : "");
      if (L.tcq()) {   // e.g. tcq_3inst_kv4_L10
        const Json& bp = t->has_residual() ? (*t->params)["base"] : *t->params;
        codec += "_" + bp["code"].as_string() + "_kv" + std::to_string(L.tcq_layout().KV) + "_L" +
                 std::to_string(L.tcq_layout().L);
      }
      codec += L.rotate_in() ? (L.rotate_out() ? "+rio" : "+ri") : (L.rotate_out() ? "+ro" : "");
      if (L.residual_entries()) codec += "+csr";
      if (L.residual_columns()) codec += "+cols";
      for (std::size_t B : bs) {
        std::vector<float> x(B * in);
        if (!x_given.empty()) {
          for (std::size_t i = 0; i < x.size(); ++i) x[i] = round_act(x_given[i], act);
        } else {
          std::mt19937_64 rng(seed * 1000003 + B * 7919 + std::hash<std::string>{}(t->name));
          std::normal_distribution<float> nd(0.0f, 1.0f);
          for (auto& v : x) v = round_act(nd(rng), act);
        }
        auto tg = clk::now();
        const std::vector<float> y = L.forward(ctx, x, B, act, gpu_path, rows_exact);
        const double gpu_s = std::chrono::duration<double>(clk::now() - tg).count();
        std::vector<double> ref(B * out);
        parallel_for(out, 16, [&](std::size_t r) {
          const float* w = W.data.data() + r * in;
          for (std::size_t b = 0; b < B; ++b) {
            const float* xb = x.data() + b * in;
            double acc = 0;
            for (std::size_t j = 0; j < in; ++j) acc += double(w[j]) * double(xb[j]);
            ref[b * out + r] = acc;
          }
        });
        const Metrics m = compare(y, ref, act);
        worst_rel = std::max(worst_rel, m.finite ? m.rel_l2 : INFINITY);
        ++n_run;
        const std::string p = L.path_for(ctx, B, gpu_path);
        if (json) {
          std::printf("{\"tensor\":\"%s\",\"out\":%zu,\"in\":%zu,\"batch\":%zu,\"dtype\":\"%s\",\"codec\":\"%s\","
                      "\"path\":\"%s\",\"max_abs\":%.9g,\"mean_abs\":%.9g,\"rel_l2\":%.9g,\"max_rel\":%.9g,"
                      "\"ref_round_rel_l2\":%.9g,\"ref_max\":%.9g,\"finite\":%s,\"gpu_s\":%.6f,"
                      "\"codebooks_zero_copy\":%s,\"zero_copy_bytes\":%zu,\"copied_bytes\":%zu,\"views\":%zu,"
                      "\"view_bytes\":%llu,\"mapped_bytes\":%llu}\n",
                      json_escape(t->name).c_str(), out, in, B, std::string(act_name(act)).c_str(), codec.c_str(),
                      p.c_str(), m.max_abs, m.mean_abs, m.rel_l2, m.max_rel, m.ref_round_rel_l2, m.ref_max,
                      m.finite ? "true" : "false", gpu_s, L.codebooks_zero_copy() ? "true" : "false",
                      L.zero_copy_bytes(), L.copied_bytes(), mf.views(), (unsigned long long)mf.view_bytes(),
                      (unsigned long long)mf.mapped_bytes());
        } else {
          char shape[32];
          std::snprintf(shape, sizeof shape, "%zux%zu", out, in);
          std::printf("%-44s %11s %3zu %-9s %-4s %10.3g %10.3g %10.3g %10.3g %10.3g\n", t->name.c_str(), shape, B,
                      codec.c_str(), p.c_str(), m.max_abs, m.mean_abs, m.rel_l2, m.max_rel, m.ref_round_rel_l2);
        }
        if (!y_out.empty()) write_npy_f32(y_out, {B, out}, y.data());
      }
    }
    if (!json)
      std::printf("# %zu runs, %zu tensors skipped (no GPU kernel); worst rel_l2 %.3g; %zu pipelines compiled in "
                  "%.2f s\n", n_run, n_skip, worst_rel, ctx.pipelines_compiled(), ctx.compile_seconds());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
