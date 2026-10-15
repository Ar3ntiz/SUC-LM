// suc1-generate: greedy generation from a SUC1 file with the C++/Metal model graph (milestone 3).
// Token ids in, greedy token ids out; optional logits / per-layer hidden states for validation.
#include <algorithm>
#include <cctype>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "suc/deepseek_v4.hpp"
#include "suc/expert_stream.hpp"
#include "suc/glm5.hpp"
#include "suc/model.hpp"
#include "suc/npy.hpp"
#include "suc/qwen4.hpp"
#include "suc/sampling.hpp"
#include "suc/sha256.hpp"
#include "suc/tokenizer.hpp"
#include "tool_common.hpp"

using namespace suc;
using namespace suc::metal;
using namespace suc::tools;

// Default chunk of the (not row-exact) batched prefill when --chunk is not given (M1 Ultra, DeepSeek-V4-Flash tcq, 786
// tokens: 8.9 ms/token at 512, 6.6 at 1024; 8,192 tokens: 134.5 / 142.8 / 147.9 tok/s at 1024 / 2048 / 4096, results
// prof_ds_new_8192*, docs/prefill_vs_ds4.md; larger chunks fill the expert tiles and amortise the weight reads).
// qwen4_exp / qwen3_5: Qwen3.8-Flash side8, 8,192 tokens, GPU ms/token 3.03 / 2.80 / 2.91 / 2.86 at chunks 512 / 1024 /
// 2048 / 4096 (prof_q38_new_8192*).
// Prompts beyond kLargeChunkMaxPrompt tokens use kDs4LongChunk: a 32,768-token prompt in chunks of 4096 exceeded the
// M1 Ultra's GPU memory with the 97 GB DeepSeek tcq file (chunks of 1024 measured fine there).
constexpr std::size_t kDs4FastChunk = 4096, kDs4LongChunk = 1024, kLargeChunkMaxPrompt = 8192, kQ4FastChunk = 1024;

static void usage() {
  std::fprintf(stderr,
               "usage: suc1-generate FILE (--tokens 1,2,3 | --tokens-file PATH) [--steps N] [--max-context N]\n"
               "                     [--dtype bf16|f16|f32] [--logits-out L.npy] [--hidden-out H.npy] [--nll-out N.npy]\n"
               "                     [--stop-eos] [--force 4,5,6] [--json OUT.json] [--verify] [--tuning-cache PATH|-] [--quiet]\n"
               "                     [--no-shared-rotations] [--fuse-norm auto|on|off] [--prefill batch|exact|token] [--chunk N]\n"
               "                     [--projections fused|separate] [--fuse-swiglu auto|on|off] [--sync] [--depth N]\n"
               "                     [--fused-layout inplace|copy] [--legacy-matvec] [--single-pass-attention]\n"
               "                     [--prompt TEXT | --prompt-file PATH] [--tokenizer tokenizer.json] [--text]\n"
               "                     [--chat [--system TEXT] [--no-think] [--reasoning-effort LEVEL]] [--print-prompt]\n"
               "                     [--temperature T] [--top-k K] [--top-p P] [--min-p P] [--seed S]\n"
               "                     [--mixed-experts routed|all] [--max-view-bytes N]\n"
               "                     [--expert-budget BYTES[K|M|G] [--hot-experts FILE|auto] [--expert-policy lru|slru]\n"
               "                      [--expert-io-threads N] [--expert-pin-fraction F] [--expert-prefetch] [--expert-cached-io]]\n"
               "  --expert-budget: SSD expert streaming (MoE files): the routed experts are read from the file on demand\n"
               "  (pread + F_NOCACHE, --expert-io-threads, default 8) into per-layer GPU slot pools of BYTES in total, kept\n"
               "  out of the GPU mapping; LRU / segmented LRU (default slru) plus a pinned hot set (--hot-experts: the\n"
               "  hot_experts.json of `suc compress`, auto = the one next to the file; up to --expert-pin-fraction of the\n"
               "  budget, default 0.5); --expert-prefetch reads the next layer's predicted experts ahead. Same tokens as\n"
               "  resident; one command buffer in flight. --expert-cached-io: reads through the page cache (A/B).\n"
               "  Sampling (temperature > 0): top-k, then top-p, then min-p on logits / T, a draw from std::mt19937_64(seed)\n"
               "  (reproducible, not mlx-lm's RNG); synchronous decode. Default: greedy.\n"
               "  --prompt / --prompt-file: text prompt, encoded with the Qwen BPE tokenizer (no chat template); the\n"
               "  tokenizer is --tokenizer or the HF-cache tokenizer.json pinned by the SUC1 tokenizer_id (sha256\n"
               "  checked). With a tokenizer (or --text) the output also has a line 'text: <decoded generation>'.\n"
               "  --chat: wrap the prompt in the Qwen3 chat template (one user turn, optional system turn, generation\n"
               "  prompt; --no-think = enable_thinking=False). Use --stop-eos to stop at <|im_end|>. qwen4_exp files use the\n"
               "  Qwen3.8-Flash-Next template (reasoning-effort system turn, '<think>\\n' generation prompt; --reasoning-effort,\n"
               "  default xhigh). --print-prompt: print 'prompt: ids' and exit (no model load).\n"
               "  Decode: pipelined by default (each step's argmax is written to the GPU token buffer and the next step\n"
               "  is encoded while the GPU runs, up to --depth command buffers in flight); --sync (and any per-step\n"
               "  output: --logits-out / --hidden-out / --nll-out / --force) waits for every step.\n"
               "  --projections: fused (default) = q|k|v and gate|up row-concatenated launches, the default Python\n"
               "                runtime (convert_to_compressed(group=True, fuse=True)); separate = one launch per\n"
               "                projection, the Python plan runtime (load_suc1_model(group=False)).\n"
               "  Prefill: the prompt in chunks of --chunk tokens per command buffer (batch, default; projections at\n"
               "  batch n, causal attention) or token by token (token: exactly the decode step, = the Python runtime fed\n"
               "  one token at a time), then N greedy decode steps (one command buffer per token).\n"
               "  deepseek_v4 / glm5_next / qwen4_exp / qwen3_5: batch = simdgroup-matrix GEMMs over the chunk rows (not\n"
               "  bit-identical to decode steps); exact = the row-exact batched prefill (decode-step bits). With --logits-out,\n"
               "  an explicit --prefill batch|exact computes every prompt row's logits in the batched chunks.\n"
               "  --force: teacher forcing; feed these tokens after the prompt instead of the argmax (validation).\n"
               "  --logits-out: (positions, vocab) float32 logits of every position (prompt + generated, the last\n"
               "                generated token excluded). --hidden-out: (positions, layers + 1, hidden) float32.\n"
               "  --nll-out: (prompt - 1,) float32 negative log-likelihood of each next prompt token (scoring; with\n"
               "                --steps 0 nothing is generated).\n"
               "  --tokens-file: ids separated by commas / whitespace. Output: one line 'generated: ids'.\n"
               "  --dtype: activation dtype (default: the checkpoint's torch_dtype, bf16 for Qwen3).\n"
               "  qwen4_exp / qwen3_5 files (Qwen3.8-Flash-Next) run the qwen4_exp graph (suc/qwen4.hpp): the prompt in the\n"
               "  fast batched layer-major prefill, chunks of --chunk tokens (default 1,024, at most the prompt); --prefill\n"
               "  exact = the row-exact layer-major prefill (chunks of --chunk, default 64); --prefill token = decode steps,\n"
               "  --chunk per command buffer (default 16); per-position outputs: one step per command buffer. --hidden-out\n"
               "  writes the (positions, layers + 1, hc * hidden) streams; the fused-plan / attention options do not apply.\n"
               "  deepseek_v4 files (DeepSeek-V4-Flash) run the deepseek_v4 graph (suc/deepseek_v4.hpp) the same way (default\n"
               "  chunk 4,096, 1,024 beyond 8,192-token prompts); --hidden-out writes the stream tensor (hc * hidden + 3 token-id digit\n"
               "  channels with hash-routed layers). --mixed-experts: rows of a mixed-codec expert stack only through the\n"
               "  sub-stack holding their expert (routed, default) or through every sub-stack (all, the Python semantics; A/B).\n"
               "  --chat on a deepseek_v4 file: BOS, [reasoning-effort text], [system text], <｜User｜>prompt<｜Assistant｜>,\n"
               "  then <think> (thinking) or </think> (--no-think), the single-turn form of encoding_dsv4.encode_messages\n"
               "  (--reasoning-effort low|high|max, default low = no text; only in thinking mode); --stop-eos stops at\n"
               "  eos_token_id. --expert-budget works on deepseek_v4 files too (mixed / native MXFP4 stacks; routed rows only).\n"
               "  glm5_next files (GLM-5.3-Flash) run the glm5_next graph (suc/glm5.hpp; the fast batched prefill with the\n"
               "  deepseek_v4 default chunks; --prefill exact = decode steps, --chunk per command buffer, default 16, or with\n"
               "  --expert-budget row-exact layer-major chunks of 64, one expert fetch per layer per chunk; --prefill token =\n"
               "  decode steps; --hidden-out: the (positions, layers + 1, hc * hidden) streams; --mixed-experts and\n"
               "  --expert-budget as for deepseek_v4). --chat: the GLM-5.3-Flash template, [gMASK]<sop><|system|>Reasoning Effort: Max\n"
               "  [<|system|>TEXT]<|user|>PROMPT<|assistant|><think> (--reasoning-effort low|high|max, default max; the\n"
               "  template ignores enable_thinking, so --no-think changes nothing); --stop-eos stops at any text\n"
               "  eos_token_id or the tokenizer's end tokens.\n");
}

static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  std::string path = argv[1], logits_out, hidden_out, nll_out, json_out;
  std::vector<uint32_t> prompt, forced;
  std::size_t steps = 16;
  bool stop_eos = false, quiet = false;
  RuntimeFlags rf;   // model / runtime / expert-streaming flags (tool_common.hpp)
  std::string &dtype = rf.dtype, &tuning_cache = rf.tuning_cache, &tokenizer_path = rf.tokenizer_path;
  std::size_t &max_ctx = rf.max_ctx, &depth = rf.depth, &chunk = rf.chunk;
  bool &verify = rf.verify, &share = rf.share, &batch_prefill = rf.batch_prefill, &sync_decode = rf.sync_decode;
  int &fuse_norm = rf.fuse_norm, &fuse_swiglu = rf.fuse_swiglu, &sdpa_2pass = rf.sdpa_2pass;
  bool &fuse_projections = rf.fuse_projections, &fused_in_place = rf.fused_in_place, &mlx_gemv = rf.mlx_gemv;
  ExpertStreamOptions& xo = rf.xo;
  ExpertStack::Rows& mixed_rows = rf.mixed_rows;
  std::optional<std::string> prompt_text;
  bool want_text = false, chat = false, think = true, print_prompt = false;
  std::string effort;   // "" = the model's default (qwen4_exp: xhigh, deepseek_v4: low)
  std::optional<std::string> system_text;
  SamplerConfig sc;
  try {
    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (parse_runtime_flag(a, next, rf)) continue;
      if (a == "--tokens") prompt = parse_ids(next());
      else if (a == "--prompt") prompt_text = next();
      else if (a == "--prompt-file") {
        std::ifstream in(next(), std::ios::binary);
        if (!in) throw std::invalid_argument("cannot read the prompt file");
        std::stringstream ss;
        ss << in.rdbuf();
        prompt_text = ss.str();
      }
      else if (a == "--text") want_text = true;
      else if (a == "--chat") chat = true;
      else if (a == "--temperature") sc.temperature = std::stod(next());
      else if (a == "--top-k") sc.top_k = std::stoul(next());
      else if (a == "--top-p") sc.top_p = std::stod(next());
      else if (a == "--min-p") sc.min_p = std::stod(next());
      else if (a == "--seed") sc.seed = std::stoull(next());
      else if (a == "--system") system_text = next();
      else if (a == "--no-think") think = false;
      else if (a == "--print-prompt") print_prompt = true;
      else if (a == "--reasoning-effort") {
        effort = next();
        if (!valid_effort(effort))
          throw std::invalid_argument("--reasoning-effort xhigh|medium|low (qwen4_exp) or low|high|max (deepseek_v4, glm5_next)");
      }
      else if (a == "--tokens-file") {
        std::ifstream in(next());
        if (!in) throw std::invalid_argument("cannot read the tokens file");
        std::stringstream ss;
        ss << in.rdbuf();
        prompt = parse_ids(ss.str());
      } else if (a == "--steps") steps = std::stoul(next());
      else if (a == "--logits-out") logits_out = next();
      else if (a == "--hidden-out") hidden_out = next();
      else if (a == "--nll-out") nll_out = next();
      else if (a == "--json") json_out = next();
      else if (a == "--force") forced = parse_ids(next());
      else if (a == "--stop-eos") stop_eos = true;
      else if (a == "--quiet") quiet = true;
      else if (a == "-h" || a == "--help") {
        usage();
        return 0;
      } else throw std::invalid_argument("unknown option " + a);
    }
    if (prompt.empty() && !prompt_text) throw std::invalid_argument("no prompt (--tokens / --tokens-file / --prompt / --prompt-file)");
    if (!prompt.empty() && prompt_text) throw std::invalid_argument("give either token ids or a text prompt");
    if ((chat || system_text || !think) && !prompt_text) throw std::invalid_argument("--chat / --system / --no-think need --prompt");
    if ((system_text || !think) && !chat) throw std::invalid_argument("--system / --no-think need --chat");
    if (!forced.empty()) steps = forced.size();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "suc1-generate: %s\n", e.what());
    usage();
    return 2;
  }

  try {
    const auto t0 = std::chrono::steady_clock::now();
    OpenOptions oo;
    oo.verify_checksum = verify;
    Suc1File f = Suc1File::open(path, oo);
    // tokenizer: --tokenizer PATH, else the tokenizer.json the SUC1 header pins (tokenizer_id, sha256-checked)
    std::optional<Tokenizer> tok;
    if (prompt_text || want_text || !tokenizer_path.empty()) {
      const std::string tp = tokenizer_path.empty() ? resolve_tokenizer_path(f.tokenizer_id()) : tokenizer_path;
      tok = Tokenizer::from_file(tp);
      if (prompt_text) {
        // --chat: the model family's chat template for one user turn (+ optional system turn) with the generation
        // prompt (tool_common.hpp chat_prompt_text)
        if (chat) {
          std::vector<ChatMessage> msgs;
          if (system_text) msgs.push_back({"system", *system_text});
          msgs.push_back({"user", *prompt_text});
          prompt_text = chat_prompt_text(f.directory()["architecture"], msgs, think, effort);
        }
        prompt = tok->encode(*prompt_text);
        if (prompt.empty()) throw std::invalid_argument("the prompt encodes to no tokens");
        if (!quiet) std::fprintf(stderr, "prompt: %zu tokens (%s)\n", prompt.size(), tp.c_str());
      }
    }
    if (print_prompt) {
      std::printf("prompt: %s\n", ids_str(prompt).c_str());
      return 0;
    }
    Tuning tu;
    tu.tuning_cache = tuning_cache;
    Context ctx(tu);
    // qwen4_exp: the n-gram tables stay out of the GPU views (read row by row on the CPU, never made resident);
    // expert streaming: so do the routed experts (read into the slot pools)
    const bool is_q4 = Qwen4Config::is_qwen4(f.directory()["architecture"]);
    std::vector<std::pair<uint64_t, uint64_t>> exclude;
    if (is_q4) exclude = qwen4_table_ranges(f);
    std::unique_ptr<ExpertStream> xs;
    if (xo.budget_bytes) {
      if (xo.hot_experts == "auto") {
        const std::string dir = path.find('/') == std::string::npos ? "." : path.substr(0, path.rfind('/'));
        const std::string hp = dir + "/hot_experts.json";
        xo.hot_experts = std::ifstream(hp).good() ? hp : "";
      }
      const auto r = routed_expert_ranges(f);
      if (r.empty()) throw std::invalid_argument("--expert-budget: the file has no routed experts");
      exclude.insert(exclude.end(), r.begin(), r.end());
      xs = std::make_unique<ExpertStream>(ctx, f, xo);
    }
    MappedFile mf(ctx, f, exclude, rf.max_view_bytes);
    // stderr / JSON report of the expert stream
    auto stream_report = [&](std::ostream* o) {
      if (o) *o << ", \"gpu_views\": " << mf.views() << ", \"gpu_view_bytes\": " << mf.view_bytes();
      if (!xs) return;
      const ExpertStreamStats& s = xs->stats();
      if (!quiet)
        std::fprintf(stderr,
                     "expert streaming: budget %.1f MB, pools %.1f MB (%zu layers, slot %.3f MB), pinned %.1f MB, all "
                     "experts %.1f MB, GPU views exclude %.1f MB; requests %llu, hit rate %.3f (prefetch hits %llu, "
                     "wasted %llu), misses %llu, evictions %llu, read %.1f MB in %llu calls, blocked %.1f ms\n",
                     xo.budget_bytes / 1e6, xs->pool_bytes() / 1e6, xs->layers(), xs->slot_bytes(0) / 1e6,
                     xs->pinned_bytes() / 1e6, xs->expert_bytes_total() / 1e6, mf.excluded_bytes() / 1e6,
                     (unsigned long long)s.requests, s.hit_rate(), (unsigned long long)s.prefetch_hits,
                     (unsigned long long)s.prefetch_wasted, (unsigned long long)s.misses,
                     (unsigned long long)s.evictions, s.bytes_read / 1e6, (unsigned long long)s.read_calls,
                     1e3 * s.wait_seconds);
      if (!quiet)
        std::fprintf(stderr, "residency set: %s, %.1f MB (%.1f MB left out: cap)\n", ctx.residency_set() ? "on" : "off",
                     ctx.residency_set_bytes() / 1e6, ctx.residency_skipped_bytes() / 1e6);
      if (o) *o << ", \"expert_streaming\": " << xs->stats_json() << ", \"excluded_from_gpu_views\": " << mf.excluded_bytes()
                << ", \"residency_set\": " << (ctx.residency_set() ? "true" : "false")
                << ", \"residency_set_bytes\": " << ctx.residency_set_bytes()
                << ", \"residency_skipped_bytes\": " << ctx.residency_skipped_bytes();
    };
    if (is_q4) {
      // ---- qwen4_exp: decode steps only (n-gram rows hashed / gathered on the host per token)
      Qwen4Options qo;
      const Qwen4Config qc = Qwen4Config::from_architecture(f.directory()["architecture"]);
      if (!dtype.empty()) qo.act = parse_act(dtype);
      else qo.act = qc.torch_dtype == "float16" ? Act::F16 : qc.torch_dtype == "float32" ? Act::F32 : Act::BF16;
      qo.max_context = max_ctx ? max_ctx : prompt.size() + steps + 1;
      const bool want_logits = !logits_out.empty(), want_hidden = !hidden_out.empty(), want_nll = !nll_out.empty();
      // --logits-out alone with an explicit --prefill batch|exact: every row's logits from the batched chunks
      const bool batch_rows = want_logits && !want_hidden && !want_nll && batch_prefill && rf.prefill_given;
      const bool rows_needed = (want_logits || want_hidden || want_nll) && !batch_rows;
      // batched (layer-major) prefill by default; --prefill token = decode steps, 16 per command buffer (the old path)
      qo.layer_prefill = batch_prefill && !rows_needed;
      qo.exact_prefill = rf.exact_prefill;
      // chunk: --chunk, else 64 (exact) / kQ4FastChunk (default batched prefill: GEMM chunks amortise weight reads)
      // the default chunk no larger than the prompt (the chunk scratch is allocated for --chunk rows)
      const std::size_t bchunk = chunk == 64 && !rf.exact_prefill && !prefill_exact_env()
                                     ? std::max<std::size_t>(2, std::min(kQ4FastChunk, prompt.size())) : chunk;
      qo.chunk = rows_needed ? 1 : qo.layer_prefill ? bchunk : (chunk == 64 ? 16 : chunk);
      qo.expert_stream = xs.get();
      Qwen4Model model(ctx, mf, qo);
      const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::vector<float> all_logits, all_hidden, nll;
      auto record = [&](std::size_t p) {
        if (want_logits || want_nll) {
          const auto l = model.logits();
          if (want_logits) all_logits.insert(all_logits.end(), l.begin(), l.end());
          if (want_nll && p + 1 < prompt.size()) {
            const double mx = *std::max_element(l.begin(), l.end());
            double se = 0.0;
            for (float v : l) se += std::exp(double(v) - mx);
            nll.push_back(float(mx + std::log(se) - double(l[prompt[p + 1]])));
          }
        }
        if (want_hidden) {
          const auto h = model.hidden_states();
          all_hidden.insert(all_hidden.end(), h.begin(), h.end());
        }
      };
      model.step(prompt[0], true);   // warm-up: compile the pipelines, then reset
      if (qo.layer_prefill && prompt.size() > 1) {   // and the batched-prefill pipelines (one chunk)
        model.reset();
        model.forward(std::vector<uint32_t>(prompt.begin(), prompt.begin() + std::min(prompt.size(), qo.chunk)), true);
      }
      model.reset();
      if (xs) xs->reset_stats();
      const double compile_s = ctx.compile_seconds();
      uint32_t next = 0;
      const auto p0 = std::chrono::steady_clock::now();
      double ngram_s = 0.0;
      if (rows_needed) {
        for (std::size_t i = 0; i < prompt.size(); ++i) {
          next = model.step(prompt[i], true);
          ngram_s += model.last_ngram_seconds();
          record(i);
        }
      } else if (batch_rows) {
        all_logits = model.forward_rows(prompt);
        ngram_s += model.last_ngram_seconds();
        const auto l = model.logits();
        next = uint32_t(std::max_element(l.begin(), l.end()) - l.begin());
      } else {
        next = model.forward(prompt, true);
        ngram_s += model.last_ngram_seconds();
      }
      const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
      const double prefill_compile_s = ctx.compile_seconds() - compile_s;
      const double prefill_gpu_s = model.last_gpu_seconds();
      std::string prefill_logits_sha, prefill_streams_sha;   // last prompt token: batch and token prefill must agree
      {
        const auto l = model.logits();
        const auto h = model.hidden_states();
        const auto dl = sha256(l.data(), l.size() * sizeof(float));
        const auto dh = sha256(h.data(), h.size() * sizeof(float));
        prefill_logits_sha = to_hex(dl.data(), dl.size());
        prefill_streams_sha = to_hex(dh.data(), dh.size());
        if (const char* e = std::getenv("SUC_PREFILL_DUMP")) {
          write_npy_f32(std::string(e) + ".streams.npy", {qc.layers + 1, h.size() / (qc.layers + 1)}, h.data());
          write_npy_f32(std::string(e) + ".logits.npy", {l.size()}, l.data());
        }
      }
      Sampler sampler(sc);
      auto draw = [&]() {
        const auto l = model.logits();
        return sampler.sample(l.data(), l.size());
      };
      if (!sc.greedy()) next = draw();
      std::vector<uint32_t> generated;
      std::vector<double> decode_ms, decode_gpu_ms, decode_ng_ms;
      std::vector<uint32_t> stop_ids;   // the text eos_token_id and the tokenizer's end tokens (<|im_end|>, ...)
      if (tok) stop_ids = tok->eos_ids();
      for (std::size_t s = 0; s < steps; ++s) {
        generated.push_back(next);
        if (stop_eos && (next == qc.eos || std::find(stop_ids.begin(), stop_ids.end(), next) != stop_ids.end())) break;
        if (s + 1 == steps) break;
        const uint32_t feed = forced.empty() ? next : forced[s];
        const auto s0 = std::chrono::steady_clock::now();
        next = model.step(feed, true);
        if (!sc.greedy()) next = draw();
        decode_ms.push_back(1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count());
        decode_gpu_ms.push_back(1e3 * model.last_gpu_seconds());
        decode_ng_ms.push_back(1e3 * model.last_ngram_seconds());
        if (rows_needed || batch_rows) record(model.position() - 1);
      }
      const std::size_t positions = model.position();
      const std::size_t L1 = qc.layers + 1, Wd = qc.stream();
      if (want_logits) write_npy_f32(logits_out, {all_logits.size() / qc.vocab, qc.vocab}, all_logits.data());
      if (want_nll) write_npy_f32(nll_out, {nll.size()}, nll.data());
      if (want_hidden) write_npy_f32(hidden_out, {all_hidden.size() / (L1 * Wd), L1, Wd}, all_hidden.data());
      const Qwen4Memory mr = model.memory();
      const ProcessMemory pm = process_memory();
      const uint64_t file_resident = resident_bytes(f.bytes(0, f.file_size()).data(), f.file_size());
      const double dmed = median(decode_ms), dgpu = median(decode_gpu_ms), dng = median(decode_ng_ms);
      if (prompt_text) std::printf("prompt: %s\n", ids_str(prompt).c_str());
      std::printf("generated: %s\n", ids_str(generated).c_str());
      if (tok) std::printf("text: %s\n", tok->decode(generated).c_str());
      if (!quiet)
        std::fprintf(stderr,
                     "model %s: %zu layers, hidden %zu x %zu streams, %zu experts top-%zu, %zu SUC projections + "
                     "%zu expert stacks, n-gram tables %s, activations %s, device %s\n"
                     "load %.3f s (pipelines compiled %zu in %.3f s)\n"
                     "prefill %zu tokens: %.2f ms/token (n-gram host %.3f ms/token); decode %zu tokens: median %.2f ms/token "
                     "(GPU %.2f ms, n-gram host %.3f ms), %.1f tokens/s\n"
                     "memory: GPU-mapped file %.1f MB (file %.1f MB resident in the page cache; n-gram tables %.1f MB outside the GPU "
                     "views, read row by row), "
                     "copied execution arrays %.2f MB, KV cache %.2f MB (%zu positions), states %.2f MB, arena %.2f MB, "
                     "workspace %.2f MB; process phys_footprint %.1f MB, resident %.1f MB (peak %.1f MB)\n",
                     qc.dense ? "qwen3_5" : "qwen4_exp", qc.layers, qc.hidden, qc.hc, qc.num_experts, qc.top_k, model.suc_projections(), model.expert_stacks(),
                     model.ngram_quantized() ? "row-affine" : "raw", std::string(act_name(qo.act)).c_str(),
                     ctx.device_name().c_str(), load_s, ctx.pipelines_compiled(), compile_s, prompt.size(),
                     1e3 * prefill_s / double(prompt.size()), 1e3 * ngram_s / double(prompt.size()), decode_ms.size(), dmed,
                     dgpu, dng, dmed > 0 ? 1e3 / dmed : 0.0, mf.mapped_bytes() / 1e6, file_resident / 1e6, mr.ngram_tables / 1e6,
                     mr.copied_weights / 1e6, mr.kv_cache / 1e6, qo.max_context, mr.states / 1e6, mr.arena / 1e6,
                     mr.workspace / 1e6, pm.phys_footprint / 1e6, pm.resident / 1e6, pm.resident_peak / 1e6);
      if (!json_out.empty()) {
        std::ofstream o(json_out);
        o << "{\"file\": \"" << path << "\", \"device\": \"" << ctx.device_name() << "\", \"architecture\": \""
          << ctx.architecture() << "\", \"model_type\": \"" << (qc.dense ? "qwen3_5" : "qwen4_exp")
          << "\", \"activation_dtype\": \"" << act_name(qo.act)
          << "\", \"prompt_tokens\": " << prompt.size() << ", \"generated\": [" << ids_str(generated)
          << "], \"teacher_forced\": " << (forced.empty() ? "false" : "true") << ", \"positions\": " << positions
          << ", \"load_s\": " << load_s << ", \"compile_s\": " << compile_s << ", \"chunk\": " << qo.chunk
          << ", \"prefill\": \"" << (qo.layer_prefill ? "batch" : "token") << "\", \"prefill_compile_s\": "
          << prefill_compile_s << ", \"prefill_gpu_s\": " << prefill_gpu_s << ", \"prefill_chunk_fetches\": "
          << model.chunk_fetches() << ", \"prefill_logits_sha256\": \"" << prefill_logits_sha
          << "\", \"prefill_streams_sha256\": \"" << prefill_streams_sha << "\""
          << ", \"prefill_s\": " << prefill_s << ", \"prefill_ngram_s\": " << ngram_s
          << ", \"decode_ms_per_token_median\": " << dmed
          << ", \"decode_gpu_ms_per_token_median\": " << dgpu << ", \"decode_ngram_ms_per_token_median\": " << dng
          << ", \"ngram_quantized\": " << (model.ngram_quantized() ? "true" : "false") << ", \"decode_ms_per_token\": [";
        for (std::size_t i = 0; i < decode_ms.size(); ++i) o << (i ? ", " : "") << decode_ms[i];
        o << "], \"memory\": {\"mapped_file\": " << mr.mapped_file << ", \"file_resident\": " << file_resident
          << ", \"ngram_tables\": " << mr.ngram_tables << ", \"excluded_from_gpu_views\": " << mf.excluded_bytes()
          << ", \"file_size\": " << f.file_size() << ", \"copied_execution_arrays\": " << mr.copied_weights
          << ", \"kv_cache\": " << mr.kv_cache << ", \"states\": " << mr.states << ", \"arena\": " << mr.arena
          << ", \"workspace\": " << mr.workspace << ", \"max_context\": " << qo.max_context
          << ", \"phys_footprint\": " << pm.phys_footprint << ", \"resident\": " << pm.resident
          << ", \"resident_peak\": " << pm.resident_peak << "}, \"suc_projections\": " << model.suc_projections();
        stream_report(&o);
        o << "}\n";
      } else {
        stream_report(nullptr);
      }
      return 0;
    }
    if (Glm5Config::is_glm5(f.directory()["architecture"]))   // ---- glm5_next (GLM-5.3-Flash): decode steps only
#include "suc1_generate_glm5.inc"
    if (Ds4Config::is_ds4(f.directory()["architecture"])) {
      // ---- deepseek_v4: decode steps only (the prompt as decode steps, --chunk per command buffer)
      if (xs && mixed_rows == ExpertStack::Rows::All)
        throw std::invalid_argument("--expert-budget with --mixed-experts all: streamed stacks run routed rows only");
      Ds4Options dopt;
      dopt.expert_stream = xs.get();
      const Ds4Config dc = Ds4Config::from_architecture(f.directory()["architecture"]);
      if (!dtype.empty()) dopt.act = parse_act(dtype);
      else dopt.act = dc.torch_dtype == "float16" ? Act::F16 : dc.torch_dtype == "float32" ? Act::F32 : Act::BF16;
      dopt.max_context = max_ctx ? max_ctx : prompt.size() + steps + 1;
      dopt.mixed_rows = mixed_rows;
      const bool want_logits = !logits_out.empty(), want_hidden = !hidden_out.empty(), want_nll = !nll_out.empty();
      // --logits-out alone with an explicit --prefill batch|exact: the batched chunks compute every row's logits
      // (forward_rows; scripts/prefill_batch_vs_token.py); without --prefill, --logits-out runs decode steps
      const bool batch_rows = want_logits && !want_hidden && !want_nll && batch_prefill && rf.prefill_given;
      const bool rows_needed = (want_logits || want_hidden || want_nll) && !batch_rows;
      // batched (layer-major) prefill by default; --prefill token = decode steps, 16 per command buffer (the old path)
      dopt.layer_prefill = batch_prefill && !rows_needed;
      dopt.exact_prefill = rf.exact_prefill;
      // the default chunk no larger than the prompt (the chunk scratch is allocated for --chunk rows)
      const std::size_t bchunk = chunk == 64 && !rf.exact_prefill && !prefill_exact_env()
                                     ? std::max<std::size_t>(2, std::min(prompt.size() > kLargeChunkMaxPrompt
                                                                             ? kDs4LongChunk : kDs4FastChunk,
                                                                         prompt.size()))
                                     : chunk;
      dopt.chunk = rows_needed ? 1 : dopt.layer_prefill ? bchunk : (chunk == 64 ? 16 : chunk);
      Ds4Model model(ctx, mf, dopt);
      const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::vector<float> all_logits, all_hidden, nll;
      auto record = [&](std::size_t p) {
        if (want_logits || want_nll) {
          const auto l = model.logits();
          if (want_logits) all_logits.insert(all_logits.end(), l.begin(), l.end());
          if (want_nll && p + 1 < prompt.size()) {
            const double mx = *std::max_element(l.begin(), l.end());
            double se = 0.0;
            for (float v : l) se += std::exp(double(v) - mx);
            nll.push_back(float(mx + std::log(se) - double(l[prompt[p + 1]])));
          }
        }
        if (want_hidden) {
          const auto h = model.hidden_states();
          all_hidden.insert(all_hidden.end(), h.begin(), h.end());
        }
      };
      model.step(prompt[0], true);   // warm-up: compile the pipelines, then reset
      if (dopt.layer_prefill && prompt.size() > 1) {   // and the batched-prefill pipelines (one chunk)
        model.reset();
        model.forward(std::vector<uint32_t>(prompt.begin(), prompt.begin() + std::min(prompt.size(), dopt.chunk)), true);
      }
      model.reset();
      model.reset_profile();   // SUC_DS4_PROFILE: the prompt's prefill only, not the warm-up chunk
      if (xs) xs->reset_stats();
      const double compile_s = ctx.compile_seconds();
      uint32_t next = 0;
      const auto p0 = std::chrono::steady_clock::now();
      if (rows_needed) {
        for (std::size_t i = 0; i < prompt.size(); ++i) {
          next = model.step(prompt[i], true);
          record(i);
        }
      } else if (batch_rows) {
        all_logits = model.forward_rows(prompt);
        const auto l = model.logits();
        next = uint32_t(std::max_element(l.begin(), l.end()) - l.begin());
      } else {
        next = model.forward(prompt, true);
      }
      const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
      const double prefill_compile_s = ctx.compile_seconds() - compile_s;   // pipelines first used by the prefill
      const double prefill_gpu_s = model.last_gpu_seconds();
      if (const auto prof = model.profile(); !prof.empty()) {   // SUC_DS4_PROFILE=1: the prefill's stages
        for (const auto& r : prof)
          std::fprintf(stderr, "prefill profile %-24s GPU %9.3f ms total, %8.1f command buffers\n", r.stage.c_str(),
                       1e3 * r.gpu_s, double(r.command_buffers));
        model.reset_profile();
      }
      // digests of the last prompt token's logits and streams: batch and token prefill must agree bit for bit
      std::string prefill_logits_sha, prefill_streams_sha;
      {
        const auto l = model.logits();
        const auto h = model.hidden_states();
        const auto dl = sha256(l.data(), l.size() * sizeof(float));
        const auto dh = sha256(h.data(), h.size() * sizeof(float));
        prefill_logits_sha = to_hex(dl.data(), dl.size());
        prefill_streams_sha = to_hex(dh.data(), dh.size());
        if (const char* e = std::getenv("SUC_PREFILL_DUMP")) {   // debugging: the last prompt token's streams
          write_npy_f32(std::string(e) + ".streams.npy", {dc.layers + 1, h.size() / (dc.layers + 1)}, h.data());
          write_npy_f32(std::string(e) + ".logits.npy", {l.size()}, l.data());
        }
      }
      Sampler sampler(sc);
      auto draw = [&]() {
        const auto l = model.logits();
        return sampler.sample(l.data(), l.size());
      };
      if (!sc.greedy()) next = draw();
      std::vector<uint32_t> generated;
      std::vector<double> decode_ms, decode_gpu_ms, decode_enc_ms;
      std::vector<uint32_t> stop_ids;
      if (tok) stop_ids = tok->eos_ids();
      model.reset_profile();
      for (std::size_t s = 0; s < steps; ++s) {
        generated.push_back(next);
        if (stop_eos && (next == dc.eos || std::find(stop_ids.begin(), stop_ids.end(), next) != stop_ids.end())) break;
        if (s + 1 == steps) break;
        const uint32_t feed = forced.empty() ? next : forced[s];
        const auto s0 = std::chrono::steady_clock::now();
        next = model.step(feed, true);
        if (!sc.greedy()) next = draw();
        decode_ms.push_back(1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count());
        decode_gpu_ms.push_back(1e3 * model.last_gpu_seconds());
        decode_enc_ms.push_back(1e3 * model.last_encode_seconds());
        if (rows_needed || batch_rows) record(model.position() - 1);
      }
      if (const auto prof = model.profile(); !prof.empty() && !decode_ms.empty()) {   // SUC_DS4_PROFILE=1
        double tot = 0.0;
        for (const auto& r : prof) tot += r.gpu_s;
        const double nt = double(decode_ms.size());
        for (const auto& r : prof)
          std::fprintf(stderr, "profile %-22s GPU %8.3f ms/token (%5.1f %%), %6.1f command buffers/token\n",
                       r.stage.c_str(), 1e3 * r.gpu_s / nt, 100.0 * r.gpu_s / tot, double(r.command_buffers) / nt);
        std::fprintf(stderr, "profile total GPU %.3f ms/token over %zu decode steps\n", 1e3 * tot / nt, decode_ms.size());
      }
      const std::size_t positions = model.position();
      const std::size_t L1 = dc.layers + 1, Wd = dc.stream() + dc.token_channels();
      if (want_logits) write_npy_f32(logits_out, {all_logits.size() / dc.vocab, dc.vocab}, all_logits.data());
      if (want_nll) write_npy_f32(nll_out, {nll.size()}, nll.data());
      if (want_hidden) write_npy_f32(hidden_out, {all_hidden.size() / (L1 * Wd), L1, Wd}, all_hidden.data());
      const Ds4Memory mr = model.memory();
      const Ds4StackStats st = model.expert_stacks();
      const ProcessMemory pm = process_memory();
      const uint64_t file_resident = resident_bytes(f.bytes(0, f.file_size()).data(), f.file_size());
      const double dmed = median(decode_ms), dgpu = median(decode_gpu_ms), denc = median(decode_enc_ms);
      if (prompt_text) std::printf("prompt: %s\n", ids_str(prompt).c_str());
      std::printf("generated: %s\n", ids_str(generated).c_str());
      if (tok) std::printf("text: %s\n", tok->decode(generated).c_str());
      if (!quiet)
        std::fprintf(stderr,
                     "model deepseek_v4: %zu layers, hidden %zu x %zu streams, %zu experts top-%zu, %zu SUC projections, "
                     "%zu expert stacks (%zu mixed, %zu native-only, %zu native experts, %zu sub-stacks, rows %s), "
                     "activations %s, device %s\n"
                     "load %.3f s (pipelines compiled %zu in %.3f s)\n"
                     "prefill %zu tokens: %.2f ms/token; decode %zu tokens: median %.2f ms/token (GPU %.2f ms, encode %.2f ms), "
                     "%.1f tokens/s\n"
                     "memory: GPU-mapped file %.1f MB (file %.1f MB resident in the page cache), copied execution arrays "
                     "%.2f MB, KV cache %.2f MB (%zu positions), arena %.2f MB, workspace %.2f MB; process phys_footprint "
                     "%.1f MB, resident %.1f MB (peak %.1f MB)\n",
                     dc.layers, dc.hidden, dc.hc, dc.experts, dc.top_k, model.suc_projections(), st.stacks, st.mixed,
                     st.native_only, st.native_experts, st.sub_stacks,
                     mixed_rows == ExpertStack::Rows::All ? "all" : "routed", std::string(act_name(dopt.act)).c_str(),
                     ctx.device_name().c_str(), load_s, ctx.pipelines_compiled(), compile_s, prompt.size(),
                     1e3 * prefill_s / double(prompt.size()), decode_ms.size(), dmed, dgpu, denc,
                     dmed > 0 ? 1e3 / dmed : 0.0, mf.mapped_bytes() / 1e6, file_resident / 1e6, mr.copied_weights / 1e6,
                     mr.kv_cache / 1e6, dopt.max_context, mr.arena / 1e6, mr.workspace / 1e6, pm.phys_footprint / 1e6,
                     pm.resident / 1e6, pm.resident_peak / 1e6);
      if (!json_out.empty()) {
        std::ofstream o(json_out);
        o << "{\"file\": \"" << path << "\", \"device\": \"" << ctx.device_name() << "\", \"architecture\": \""
          << ctx.architecture() << "\", \"model_type\": \"deepseek_v4\", \"activation_dtype\": \"" << act_name(dopt.act)
          << "\", \"prompt_tokens\": " << prompt.size() << ", \"generated\": [" << ids_str(generated)
          << "], \"teacher_forced\": " << (forced.empty() ? "false" : "true") << ", \"positions\": " << positions
          << ", \"load_s\": " << load_s << ", \"compile_s\": " << compile_s << ", \"chunk\": " << dopt.chunk
          << ", \"prefill\": \"" << (dopt.layer_prefill ? "batch" : "token") << "\", \"prefill_compile_s\": "
          << prefill_compile_s << ", \"prefill_gpu_s\": " << prefill_gpu_s << ", \"prefill_chunk_fetches\": "
          << model.chunk_fetches() << ", \"prefill_logits_sha256\": \"" << prefill_logits_sha
          << "\", \"prefill_streams_sha256\": \"" << prefill_streams_sha << "\""
          << ", \"prefill_s\": " << prefill_s << ", \"decode_ms_per_token_median\": " << dmed
          << ", \"decode_gpu_ms_per_token_median\": " << dgpu << ", \"decode_encode_ms_per_token_median\": " << denc
          << ", \"mixed_experts\": \"" << (mixed_rows == ExpertStack::Rows::All ? "all" : "routed")
          << "\", \"expert_stacks\": {\"stacks\": " << st.stacks << ", \"mixed\": " << st.mixed << ", \"native_only\": "
          << st.native_only << ", \"native_experts\": " << st.native_experts << ", \"sub_stacks\": " << st.sub_stacks
          << "}, \"decode_ms_per_token\": [";
        for (std::size_t i = 0; i < decode_ms.size(); ++i) o << (i ? ", " : "") << decode_ms[i];
        o << "], \"memory\": {\"mapped_file\": " << mr.mapped_file << ", \"file_resident\": " << file_resident
          << ", \"file_size\": " << f.file_size() << ", \"copied_execution_arrays\": " << mr.copied_weights
          << ", \"kv_cache\": " << mr.kv_cache << ", \"arena\": " << mr.arena << ", \"workspace\": " << mr.workspace
          << ", \"max_context\": " << dopt.max_context << ", \"phys_footprint\": " << pm.phys_footprint
          << ", \"resident\": " << pm.resident << ", \"resident_peak\": " << pm.resident_peak
          << "}, \"suc_projections\": " << model.suc_projections();
        stream_report(&o);
        o << "}\n";
      } else {
        stream_report(nullptr);
      }
      return 0;
    }
    const ModelConfig cfg = ModelConfig::from_architecture(f.directory()["architecture"]);
    ModelOptions mo;
    if (!dtype.empty()) mo.act = parse_act(dtype);
    else mo.act = cfg.torch_dtype == "float16" ? Act::F16 : cfg.torch_dtype == "float32" ? Act::F32 : Act::BF16;
    mo.max_context = max_ctx ? max_ctx : prompt.size() + steps + 1;
    mo.share_rotations = share;
    mo.fuse_norm = fuse_norm;
    mo.fuse_projections = fuse_projections;
    mo.fused_in_place = fused_in_place;
    mo.mlx_gemv = mlx_gemv;
    mo.sdpa_2pass = sdpa_2pass;
    mo.fuse_swiglu = fuse_swiglu;
    mo.chunk = chunk;
    mo.expert_stream = xs.get();
    Model model(ctx, mf, mo);
    // streamed experts: Model::forward runs at most this many tokens per command buffer (one call's distinct experts
    // must fit the unpinned slots); prefill in these chunks so that every chunk's logits rows are recorded
    if (xs && cfg.top_k) mo.chunk = std::max<std::size_t>(1, std::min(mo.chunk, xs->min_max_distinct() / cfg.top_k));
    for (std::size_t n : {cfg.hidden, cfg.intermediate, cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim})
      if (ctx.rotation_layout_overridden(n))
        std::fprintf(stderr, "warning: the tuning cache selects a rotation layout other than 'blocks' / 'row' for n = %zu; "
                     "this port runs 'blocks', so results are not bit-identical to the Python runtime\n", n);
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::size_t eos = 0;
    bool has_eos = false;
    if (const Json* e = f.directory()["architecture"].find("eos_token_id"); e && e->is_int()) {
      eos = std::size_t(e->as_uint());
      has_eos = true;
    }

    const bool want_logits = !logits_out.empty(), want_hidden = !hidden_out.empty(), want_nll = !nll_out.empty();
    const bool rows_needed = want_logits || want_hidden || want_nll;
    std::vector<float> all_logits, all_hidden, nll;
    std::size_t rows_seen = 0;
    std::vector<uint32_t> generated;
    std::vector<double> prefill_ms, decode_ms, decode_gpu_ms, decode_enc_ms;
    auto record = [&]() {
      if (want_logits || want_nll) {
        auto l = model.logits();
        const std::size_t rows = l.size() / cfg.vocab;
        if (want_logits) all_logits.insert(all_logits.end(), l.begin(), l.end());
        for (std::size_t r = 0; want_nll && r < rows; ++r) {   // -log softmax(logits)[next prompt token], float64
          const std::size_t p = rows_seen + r;
          if (p + 1 >= prompt.size()) break;
          const float* row = l.data() + r * cfg.vocab;
          const double mx = *std::max_element(row, row + cfg.vocab);
          double se = 0.0;
          for (std::size_t j = 0; j < cfg.vocab; ++j) se += std::exp(double(row[j]) - mx);
          nll.push_back(float(mx + std::log(se) - double(row[prompt[p + 1]])));
        }
        rows_seen += rows;
      }
      if (want_hidden) {
        auto h = model.hidden_states();
        all_hidden.insert(all_hidden.end(), h.begin(), h.end());
      }
    };
    // warm-up: compile every pipeline once (not timed), then reset the cache
    model.step(prompt[0], true);
    model.reset();
    if (batch_prefill && prompt.size() > 1) {
      model.forward(std::vector<uint32_t>(prompt.begin(), prompt.begin() + std::min(prompt.size(), mo.chunk)),
                    Model::Logits::Last);
      model.reset();
    }
    if (xs) xs->reset_stats();
    const double compile_s = ctx.compile_seconds();

    uint32_t next = 0;
    const auto p0 = std::chrono::steady_clock::now();
    if (batch_prefill) {
      const auto which = rows_needed ? Model::Logits::All : Model::Logits::Last;
      for (std::size_t i = 0; i < prompt.size(); i += mo.chunk) {
        const std::size_t n = std::min(mo.chunk, prompt.size() - i);
        const bool last = i + n == prompt.size();
        auto r = model.forward(std::vector<uint32_t>(prompt.begin() + i, prompt.begin() + i + n),
                               which == Model::Logits::All ? which : last ? Model::Logits::Last : Model::Logits::None);
        if (last) next = r.back();
        if (rows_needed) record();
      }
    } else {
      for (std::size_t i = 0; i < prompt.size(); ++i) {
        const bool last = i + 1 == prompt.size();
        const bool logits_here = last || rows_needed;
        const auto s0 = std::chrono::steady_clock::now();
        next = model.step(prompt[i], logits_here);
        prefill_ms.push_back(1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count());
        if (rows_needed) record();
      }
    }
    const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
    std::string prefill_logits_sha;   // last prompt token (batch vs token prefill: not bit-identical on this graph)
    if (!rows_needed) {
      const auto l = model.logits();
      if (l.size() >= cfg.vocab) {
        const auto dl = sha256(l.data() + (l.size() - cfg.vocab), cfg.vocab * sizeof(float));
        prefill_logits_sha = to_hex(dl.data(), dl.size());
        if (const char* e = std::getenv("SUC_PREFILL_DUMP"))
          write_npy_f32(std::string(e) + ".logits.npy", {cfg.vocab}, l.data() + (l.size() - cfg.vocab));
      }
    }
    // sampling (not greedy): the next token is drawn on the host from the last row of logits (synchronous decode)
    Sampler sampler(sc);
    auto draw = [&]() {
      const auto l = model.logits();
      return sampler.sample(l.data() + (l.size() / cfg.vocab - 1) * cfg.vocab, cfg.vocab);
    };
    if (!sc.greedy()) next = draw();
    // pipelined greedy decode (default when no per-step output is requested): argmax -> GPU token buffer
    const bool pipelined = !sync_decode && !rows_needed && forced.empty() && sc.greedy();
    if (pipelined && steps > 0) {
      Model::GenerateStats st;
      std::optional<uint32_t> stop;
      if (stop_eos && has_eos) stop = uint32_t(eos);
      generated.push_back(next);
      if (!(stop && next == *stop) && steps > 1) {
        const auto g = model.generate(next, steps - 1, depth, &st, stop);
        generated.insert(generated.end(), g.begin(), g.end());
      }
      decode_ms = st.step_ms;
      decode_gpu_ms = st.gpu_ms;
      decode_enc_ms = st.encode_ms;
      steps = 0;
    }
    for (std::size_t s = 0; s < steps; ++s) {
      generated.push_back(next);
      if (stop_eos && has_eos && next == eos) break;
      if (s + 1 == steps) break;
      const uint32_t feed = forced.empty() ? next : forced[s];
      const auto s0 = std::chrono::steady_clock::now();
      next = model.step(feed, true);
      if (!sc.greedy()) next = draw();
      decode_ms.push_back(1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count());
      decode_gpu_ms.push_back(1e3 * model.last_gpu_seconds());
      decode_enc_ms.push_back(1e3 * model.last_encode_seconds());
      if (rows_needed) record();
    }
    const std::size_t positions = model.position();
    const double prefill_ms_tok = 1e3 * prefill_s / double(prompt.size());
    if (want_logits) write_npy_f32(logits_out, {positions, cfg.vocab}, all_logits.data());
    if (want_nll) write_npy_f32(nll_out, {nll.size()}, nll.data());
    if (want_hidden) write_npy_f32(hidden_out, {positions, cfg.layers + 1, cfg.hidden}, all_hidden.data());

    const MemoryReport mr = model.memory();
    const ProcessMemory pm = process_memory();
    const uint64_t file_resident = resident_bytes(f.bytes(0, f.file_size()).data(), f.file_size());
    const double dmed = median(decode_ms), dgpu = median(decode_gpu_ms), denc = median(decode_enc_ms),
                 pmed = median(prefill_ms);
    if (prompt_text) std::printf("prompt: %s\n", ids_str(prompt).c_str());
    std::printf("generated: %s\n", ids_str(generated).c_str());
    if (tok) std::printf("text: %s\n", tok->decode(generated).c_str());
    if (!quiet) {
      std::fprintf(stderr,
                   "model %s: %zu layers, hidden %zu, %zu SUC projections + %zu other, activations %s, device %s, "
                   "shared rotations %s, fused norm %s, fused projection groups %zu, fused swiglu %s\n"
                   "load %.3f s (pipelines compiled %zu in %.3f s)\n"
                   "prefill %zu tokens (%s): %.2f ms/token; decode %zu tokens: median %.2f ms/token (GPU %.2f ms, encode %.2f ms), "
                   "%.1f tokens/s\n"
                   "memory: mapped file %.1f MB (%.1f MB resident in the page cache), copied execution arrays %.2f MB, KV cache %.2f MB (%zu positions), "
                   "arena %.2f MB, workspace %.2f MB; process phys_footprint %.1f MB, resident %.1f MB "
                   "(peak %.1f MB)\n",
                   cfg.model_type.c_str(), cfg.layers, cfg.hidden, model.suc_projections(), model.other_projections(),
                   std::string(act_name(mo.act)).c_str(), ctx.device_name().c_str(), share ? "on" : "off",
                   model.norm_fused() ? "on" : "off", model.fused_groups(), model.swiglu_fused() ? "on" : "off", load_s, ctx.pipelines_compiled(),
                   compile_s, prompt.size(), batch_prefill ? "batch" : "token", prefill_ms_tok, decode_ms.size(), dmed, dgpu, denc, dmed > 0 ? 1e3 / dmed : 0.0,
                   mr.mapped_file / 1e6, file_resident / 1e6, mr.copied_weights / 1e6, mr.kv_cache / 1e6, mo.max_context, mr.arena / 1e6,
                   mr.workspace / 1e6, pm.phys_footprint / 1e6, pm.resident / 1e6, pm.resident_peak / 1e6);
    }
    if (!json_out.empty()) {
      std::ofstream o(json_out);
      o << "{\"file\": \"" << path << "\", \"device\": \"" << ctx.device_name() << "\", \"architecture\": \""
        << ctx.architecture() << "\", \"model_type\": \"" << cfg.model_type << "\", \"activation_dtype\": \""
        << act_name(mo.act) << "\", \"prompt_tokens\": " << prompt.size() << ", \"generated\": [" << ids_str(generated)
        << "], \"teacher_forced\": " << (forced.empty() ? "false" : "true") << ", \"load_s\": " << load_s
        << ", \"compile_s\": " << compile_s << ", \"prefill\": \"" << (batch_prefill ? "batch" : "token")
        << "\", \"chunk\": " << mo.chunk << ", \"prefill_logits_sha256\": \"" << prefill_logits_sha << "\""
        << ", \"prefill_s\": " << prefill_s << ", \"prefill_ms_per_token\": " << prefill_ms_tok
        << ", \"prefill_ms_per_token_median\": " << pmed
        << ", \"decode_ms_per_token_median\": " << dmed << ", \"decode_gpu_ms_per_token_median\": " << dgpu
        << ", \"decode_encode_ms_per_token_median\": " << denc << ", \"shared_rotations\": " << (share ? "true" : "false") << ", \"fuse_norm\": " << (model.norm_fused() ? "true" : "false")
        << ", \"fused_projection_groups\": " << model.fused_groups() << ", \"fuse_swiglu\": " << (model.swiglu_fused() ? "true" : "false")
        << ", \"decode_ms_per_token\": [";
      for (std::size_t i = 0; i < decode_ms.size(); ++i) o << (i ? ", " : "") << decode_ms[i];
      o << "], \"memory\": {\"mapped_file\": " << mr.mapped_file << ", \"file_resident\": " << file_resident << ", \"copied_execution_arrays\": " << mr.copied_weights
        << ", \"kv_cache\": " << mr.kv_cache << ", \"arena\": " << mr.arena << ", \"workspace\": " << mr.workspace
        << ", \"max_context\": " << mo.max_context << ", \"phys_footprint\": " << pm.phys_footprint
        << ", \"resident\": " << pm.resident << ", \"resident_peak\": " << pm.resident_peak
        << "}, \"suc_projections\": " << model.suc_projections() << ", \"other_projections\": "
        << model.other_projections();
      stream_report(&o);
      o << "}\n";
    } else {
      stream_report(nullptr);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "suc1-generate: %s\n", e.what());
    return 1;
  }
  return 0;
}
