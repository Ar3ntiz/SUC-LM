// suc1-moe: run one MoE block (router + experts + combine) or one expert stack of a SUC1 file on the GPU, on given
// input rows; writes the output as float32 .npy (the values of the activation dtype). For the cross-checks with the
// Python runtime (tests/unit/test_cpp_moe.py).
//
//   suc1-moe FILE --layer L --x X.npy --y-out Y.npy [--dtype bf16|f16|f32] [--layout auto|stacked|table]
//            [--stack gate_proj|up_proj|down_proj --eidx E.npy]       one SwitchLinear, row n through expert E[n]
//            [--mixed routed|all]                                     with --stack: an ExpertStack (mixed codecs)
//            [--tiles BT]                                             with --stack: expert tiles of <= BT rows
//                                                                     (plan_tiles, the batched-prefill path)
//            [--route-out R.npy]                                      also write the routing: (n, 2k) = ids, scores
//            [--routing R.npy]                                        experts + combine with a given routing (n, 2k)
//            [--tuning-cache P|-] [--no-verify]
// X.npy: (N, in) float32 (rounded to the dtype). E.npy / R.npy: float32 (expert ids are exact small integers).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <algorithm>
#include <string>
#include <vector>

#include "suc/expert_mixed.hpp"
#include "suc/metal.hpp"
#include "suc/npy.hpp"
#include "suc/switch.hpp"

using namespace suc;
using namespace suc::metal;

namespace {

[[noreturn]] void usage(const char* msg) {
  std::fprintf(stderr, "suc1-moe: %s\nusage: suc1-moe FILE --layer L --x X.npy --y-out Y.npy [--dtype bf16|f16|f32] "
                       "[--layout auto|stacked|table] [--stack PROJ --eidx E.npy] [--route-out R.npy] "
                       "[--routing R.npy] [--mixed routed|all] [--tiles BT] [--tuning-cache P|-] [--no-verify]\n", msg);
  std::exit(2);
}

std::size_t arch_size(const Json& a, const char* key) {
  const Json* v = a.find(key);
  if (!v || !v->is_int()) throw FormatError(std::string("architecture: missing integer '") + key + "'");
  return std::size_t(v->as_uint());
}

void run(Context& ctx, const std::function<void(MTL::ComputeCommandEncoder*)>& body) {
  MTL::CommandBuffer* cb = ctx.queue()->commandBuffer();
  MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  body(enc);
  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();
  if (cb->status() != MTL::CommandBufferStatusCompleted) throw std::runtime_error("command buffer failed");
}

}  // namespace

int main(int argc, char** argv) {
  std::string file, xpath, ypath, stack, epath, route_out, routing, dtype = "bf16", layout_s = "auto", cache, mixed;
  long layer = -1, tiles_bt = 0;
  bool verify = true;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage(("missing value for " + a).c_str());
      return argv[++i];
    };
    if (a == "--layer") layer = std::stol(next());
    else if (a == "--x") xpath = next();
    else if (a == "--y-out") ypath = next();
    else if (a == "--stack") stack = next();
    else if (a == "--eidx") epath = next();
    else if (a == "--route-out") route_out = next();
    else if (a == "--routing") routing = next();
    else if (a == "--dtype") dtype = next();
    else if (a == "--layout") layout_s = next();
    else if (a == "--tuning-cache") cache = next();
    else if (a == "--mixed") mixed = next();
    else if (a == "--tiles") tiles_bt = std::stol(next());
    else if (a == "--no-verify") verify = false;
    else if (!a.empty() && a[0] == '-') usage(("unknown option " + a).c_str());
    else if (file.empty()) file = a;
    else usage("more than one FILE");
  }
  if (file.empty() || layer < 0 || xpath.empty() || ypath.empty()) usage("FILE, --layer, --x and --y-out are required");
  if (!stack.empty() && epath.empty()) usage("--stack needs --eidx");
  try {
    const StackLayout layout = layout_s == "auto" ? StackLayout::Auto
                               : layout_s == "stacked" ? StackLayout::Stacked
                               : layout_s == "table" ? StackLayout::Table
                                                     : (usage("--layout must be auto, stacked or table"), StackLayout::Auto);
    const Act act = parse_act(dtype);
    OpenOptions oo;
    oo.verify_checksum = verify;
    const Suc1File f = Suc1File::open(file, oo);
    Tuning tu;
    tu.tuning_cache = cache;
    Context ctx(tu);
    MappedFile mf(ctx, f);
    const std::string prefix = "model.layers." + std::to_string(layer);
    std::vector<uint64_t> xs;
    const std::vector<float> x = read_npy_f32(xpath, xs);
    if (xs.size() != 2) throw std::runtime_error("--x must be 2-D (rows, features)");
    const std::size_t n = xs[0];
    const std::size_t as = act_size(act);
    BufPtr xb = ctx.new_buffer(x.size() * as);
    f32_to_act(x.data(), xb->contents(), x.size(), act);

    if (!stack.empty() && (!mixed.empty() || tiles_bt > 0)) {
      // ExpertStack (--mixed) or SwitchLinear rows, optionally in expert tiles (plan_tiles: rows grouped by expert,
      // tile-major, written back through dst)
      std::vector<uint64_t> es;
      const std::vector<float> ef = read_npy_f32(epath, es);
      if (ef.size() != n) throw std::runtime_error("--eidx must hold one expert id per row of --x");
      std::vector<uint32_t> eidx(n);
      for (std::size_t i = 0; i < n; ++i) eidx[i] = uint32_t(ef[i]);
      std::unique_ptr<ExpertStack> es_;
      std::unique_ptr<SwitchLinear> sl_;
      std::size_t in = 0, out = 0, max_bt = 1;
      if (!mixed.empty()) {
        if (mixed != "routed" && mixed != "all") usage("--mixed must be routed or all");
        es_ = std::make_unique<ExpertStack>(ExpertStack::from_file(
            ctx, mf, prefix + ".mlp", stack, 0, mixed == "all" ? ExpertStack::Rows::All : ExpertStack::Rows::Routed));
        in = es_->in_features(), out = es_->out_features(), max_bt = es_->max_tile_rows(ctx);
      } else {
        sl_ = std::make_unique<SwitchLinear>(SwitchLinear::from_file(ctx, mf, prefix + ".mlp", stack, 0, layout));
        in = sl_->in_features(), out = sl_->out_features(), max_bt = sl_->max_tile_rows(ctx);
      }
      if (xs[1] != in) throw std::runtime_error("--x has the wrong number of columns");
      std::vector<uint32_t> src(n), dst(n), reid = eidx, teidx;
      for (std::size_t i = 0; i < n; ++i) src[i] = dst[i] = uint32_t(i);
      std::size_t N = n, bt = 1;
      if (tiles_bt > 0) {
        std::vector<std::pair<uint32_t, uint32_t>> rows(n);
        for (std::size_t i = 0; i < n; ++i) rows[i] = {uint32_t(i), eidx[i]};
        std::vector<uint32_t> rank;
        if (es_) {
          for (std::size_t e = 0; e < es_->num_experts(); ++e) rank.push_back(uint32_t(es_->sub_slot(e)));
        }
        const TilePlan P = plan_tiles(rows, std::min<std::size_t>(std::size_t(tiles_bt), max_bt), es_ ? &rank : nullptr);
        src = P.src, dst = P.dst, reid = P.eidx, teidx = P.teidx, N = P.rows(), bt = P.bt;
      }
      BufPtr xt = ctx.new_buffer(N * in * as), yt = ctx.new_buffer(N * out * as);
      for (std::size_t r = 0; r < N; ++r)
        std::memcpy(static_cast<uint8_t*>(xt->contents()) + r * in * as,
                    static_cast<const uint8_t*>(xb->contents()) + std::size_t(src[r]) * in * as, in * as);
      BufPtr eb = ctx.upload(reid.data(), N * 4);
      BufPtr tb = teidx.empty() ? BufPtr() : ctx.upload(teidx.data(), teidx.size() * 4);
      Tiles T;
      if (tb) T = Tiles{{tb.get(), 0}, teidx.size(), bt, teidx.data()};
      const Tiles* tp = tb ? &T : nullptr;
      Workspace ws;
      ExpertScratch scr;
      run(ctx, [&](MTL::ComputeCommandEncoder* enc) {
        if (es_ && tp) ExpertStack::encode_group(ctx, enc, {{es_.get(), {xt.get(), 0}, {yt.get(), 0}, &scr}},
                                                 {eb.get(), 0}, N, act, tp);
        else if (es_) es_->encode(ctx, enc, {xt.get(), 0}, {eb.get(), 0}, {yt.get(), 0}, N, act, scr);
        else sl_->encode(ctx, enc, {xt.get(), 0}, {eb.get(), 0}, {yt.get(), 0}, N, act, ws, tp);
      });
      std::vector<float> yall(N * out), y(n * out);
      act_to_f32(yt->contents(), yall.data(), yall.size(), act);
      for (std::size_t r = 0; r < N; ++r)
        if (dst[r] != ~0u) std::memcpy(y.data() + std::size_t(dst[r]) * out, yall.data() + r * out, out * 4);
      write_npy_f32(ypath, {n, out}, y.data());
      std::fprintf(stderr, "%s.mlp.%s: %s, tile rows %zu (max %zu)\n", prefix.c_str(), stack.c_str(),
                   es_ ? (es_->mixed() ? "mixed stack" : "uniform stack") : (sl_->tcq() ? "tcq stack" : "stack"), bt,
                   max_bt);
      return 0;
    }

    if (!stack.empty()) {
      const SwitchLinear sl = SwitchLinear::from_file(ctx, mf, prefix + ".mlp", stack, 0, layout);
      if (xs[1] != sl.in_features()) throw std::runtime_error("--x has the wrong number of columns");
      std::vector<uint64_t> es;
      const std::vector<float> ef = read_npy_f32(epath, es);
      if (ef.size() != n) throw std::runtime_error("--eidx must hold one expert id per row of --x");
      std::vector<uint32_t> eidx(n);
      for (std::size_t i = 0; i < n; ++i) eidx[i] = uint32_t(ef[i]);
      const std::vector<float> y = sl.forward(ctx, x, eidx, act);
      write_npy_f32(ypath, {n, sl.out_features()}, y.data());
      std::fprintf(stderr, "%s: E=%zu %zux%zu M=%zu bits=%u layout=%s codebooks=%s signs_in=%s signs_out=%s "
                           "residual=%zu columns=%zu weight=%zu B zero-copy=%zu B copied=%zu B\n",
                   sl.name().c_str(), sl.num_experts(), sl.out_features(), sl.in_features(), sl.num_codebooks(),
                   sl.bits(), sl.layout() == StackLayout::Table ? "table" : "stacked",
                   sl.shared_codebooks() ? "shared" : "per-expert",
                   !sl.rotate_in() ? "none" : sl.shared_signs_in() ? "shared" : "per-row",
                   !sl.rotate_out() ? "none" : sl.shared_signs_out() ? "shared" : "per-row", sl.residual_entries(),
                   sl.residual_columns(), sl.weight_bytes(), sl.zero_copy_bytes(), sl.copied_bytes());
      return 0;
    }

    const Json& a = f.directory()["architecture"];
    MoeConfig cfg;
    cfg.hidden = arch_size(a, "hidden_size");
    cfg.num_experts = arch_size(a, "num_experts");
    cfg.top_k = arch_size(a, "num_experts_per_tok");
    cfg.intermediate = arch_size(a, "moe_intermediate_size");
    if (const Json* v = a.find("norm_topk_prob"); v && !v->is_null()) cfg.norm_topk_prob = v->as_bool();
    if (xs[1] != cfg.hidden) throw std::runtime_error("--x must have hidden_size columns");
    const MoeBlock blk = MoeBlock::from_file(ctx, mf, prefix, cfg, act, layout);
    const std::size_t K = cfg.top_k;
    BufPtr yb = ctx.new_buffer(n * cfg.hidden * as);
    BufPtr eb = ctx.new_buffer(n * K * 4), sb = ctx.new_buffer(n * K * as);
    MoeScratch s;
    if (!routing.empty()) {
      std::vector<uint64_t> rs;
      const std::vector<float> r = read_npy_f32(routing, rs);
      if (r.size() != n * 2 * K) throw std::runtime_error("--routing must be (rows, 2 * top_k)");
      std::vector<float> sc(n * K);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < K; ++j) {
          static_cast<uint32_t*>(eb->contents())[i * K + j] = uint32_t(r[i * 2 * K + j]);
          sc[i * K + j] = r[i * 2 * K + K + j];
        }
      f32_to_act(sc.data(), sb->contents(), sc.size(), act);
      run(ctx, [&](MTL::ComputeCommandEncoder* enc) {
        blk.encode_experts(ctx, enc, {xb.get(), 0}, {eb.get(), 0}, {sb.get(), 0}, {yb.get(), 0}, n, s);
      });
    } else {
      run(ctx, [&](MTL::ComputeCommandEncoder* enc) {
        blk.encode_route(ctx, enc, {xb.get(), 0}, {eb.get(), 0}, {sb.get(), 0}, n, s);
        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        blk.encode_experts(ctx, enc, {xb.get(), 0}, {eb.get(), 0}, {sb.get(), 0}, {yb.get(), 0}, n, s);
      });
    }
    std::vector<float> y(n * cfg.hidden);
    act_to_f32(yb->contents(), y.data(), y.size(), act);
    write_npy_f32(ypath, {n, cfg.hidden}, y.data());
    if (!route_out.empty()) {
      std::vector<float> sc(n * K), r(n * 2 * K);
      act_to_f32(sb->contents(), sc.data(), sc.size(), act);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < K; ++j) {
          r[i * 2 * K + j] = float(static_cast<const uint32_t*>(eb->contents())[i * K + j]);
          r[i * 2 * K + K + j] = sc[i * K + j];
        }
      write_npy_f32(route_out, {n, 2 * K}, r.data());
    }
    std::fprintf(stderr, "%s.mlp: E=%zu top_k=%zu hidden=%zu moe_intermediate=%zu norm_topk_prob=%d layout=%s "
                         "weight=%zu B copied=%zu B scratch=%zu B\n",
                 prefix.c_str(), cfg.num_experts, K, cfg.hidden, cfg.intermediate, int(cfg.norm_topk_prob),
                 blk.gate_proj().layout() == StackLayout::Table ? "table" : "stacked", blk.weight_bytes(),
                 blk.copied_bytes(), s.bytes());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "suc1-moe: %s\n", e.what());
    return 1;
  }
  return 0;
}
