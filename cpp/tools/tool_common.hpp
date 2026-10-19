// Shared by suc1-generate and suc1-server: token-id lists, the runtime / expert-streaming command-line flags and the
// per-family chat templates. Header-only (the tools are single translation units).
#pragma once

#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "suc/deepseek_v4.hpp"
#include "suc/expert_stream.hpp"
#include "suc/glm5.hpp"
#include "suc/json.hpp"
#include "suc/prefill.hpp"
#include "suc/qwen4.hpp"
#include "suc/expert_mixed.hpp"

namespace suc::tools {

inline std::vector<uint32_t> parse_ids(const std::string& s) {
  std::vector<uint32_t> v;
  std::string cur;
  for (char ch : s + ",") {
    if (ch >= '0' && ch <= '9') {
      cur += ch;
    } else if (ch == ',' || ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r') {
      if (!cur.empty()) v.push_back(uint32_t(std::stoul(cur)));
      cur.clear();
    } else {
      throw std::invalid_argument(std::string("bad character in token list: '") + ch + "'");
    }
  }
  return v;
}

inline std::string ids_str(const std::vector<uint32_t>& v) {
  std::string s;
  for (std::size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
  return s;
}

// Model / runtime / expert-streaming options common to the tools (see suc1-generate's usage text).
struct RuntimeFlags {
  std::string dtype, tuning_cache, tokenizer_path;
  std::size_t max_ctx = 0;
  bool verify = false, share = true, batch_prefill = true, sync_decode = false;
  bool exact_prefill = false;   // --prefill exact: the row-exact batched prefill of the frontier graphs
  bool prefill_given = false;   // --prefill was passed (with --logits-out: batched rows, not decode steps)
  int fuse_norm = -1, fuse_swiglu = -1;
  std::size_t depth = 2;
  bool fuse_projections = true, fused_in_place = true, mlx_gemv = true;
  int sdpa_2pass = -1;
  std::size_t chunk = 64;
  bool chunk_given = false;
  uint64_t max_view_bytes = 0;   // --max-view-bytes: cap the GPU views of the mapped file (0 = maxBufferLength; tests)
  metal::ExpertStreamOptions xo;
  metal::ExpertStack::Rows mixed_rows = metal::ExpertStack::Rows::Routed;
};

// Parses one runtime flag `a` (its value, if any, from next()); false when `a` is not a runtime flag.
template <class Next>
bool parse_runtime_flag(const std::string& a, Next&& next, RuntimeFlags& r) {
  using metal::ExpertStack;
  using metal::ExpertStreamOptions;
  if (a == "--max-context") r.max_ctx = std::stoul(next());
  else if (a == "--dtype") r.dtype = next();
  else if (a == "--tokenizer") r.tokenizer_path = next();
  else if (a == "--verify") r.verify = true;
  else if (a == "--tuning-cache") r.tuning_cache = next();
  else if (a == "--no-shared-rotations") r.share = false;
  else if (a == "--fuse-norm") {
    const std::string v = next();
    if (v != "auto" && v != "on" && v != "off") throw std::invalid_argument("--fuse-norm auto|on|off");
    r.fuse_norm = v == "auto" ? -1 : v == "on" ? 1 : 0;
  }
  else if (a == "--sync") r.sync_decode = true;
  else if (a == "--legacy-matvec") r.mlx_gemv = false;
  else if (a == "--single-pass-attention") r.sdpa_2pass = 0;
  else if (a == "--fused-layout") {
    const std::string v = next();
    if (v != "inplace" && v != "copy") throw std::invalid_argument("--fused-layout inplace|copy");
    r.fused_in_place = v == "inplace";
  }
  else if (a == "--depth") r.depth = std::stoul(next());
  else if (a == "--projections") {
    const std::string v = next();
    if (v != "fused" && v != "separate") throw std::invalid_argument("--projections fused|separate");
    r.fuse_projections = v == "fused";
  }
  else if (a == "--fuse-swiglu") {
    const std::string v = next();
    if (v != "auto" && v != "on" && v != "off") throw std::invalid_argument("--fuse-swiglu auto|on|off");
    r.fuse_swiglu = v == "auto" ? -1 : v == "on" ? 1 : 0;
  }
  else if (a == "--prefill") {
    const std::string v = next();
    if (v != "batch" && v != "token" && v != "exact") throw std::invalid_argument("--prefill batch|exact|token");
    r.batch_prefill = v != "token";
    r.exact_prefill = v == "exact";
    r.prefill_given = true;
  }
  else if (a == "--chunk") { r.chunk = std::stoul(next()); r.chunk_given = true; if (!r.chunk) throw std::invalid_argument("--chunk must be positive"); }
  else if (a == "--max-view-bytes") r.max_view_bytes = std::stoull(next());
  else if (a == "--expert-budget") {
    std::string v = next();
    double mul = 1.0;
    const char u = v.empty() ? 0 : char(std::toupper(static_cast<unsigned char>(v.back())));
    if (u == 'K' || u == 'M' || u == 'G') {
      mul = u == 'K' ? 1024.0 : u == 'M' ? 1048576.0 : 1073741824.0;
      v.pop_back();
    }
    r.xo.budget_bytes = uint64_t(std::stod(v) * mul);
    if (!r.xo.budget_bytes) throw std::invalid_argument("--expert-budget must be > 0");
  }
  else if (a == "--hot-experts") r.xo.hot_experts = next();
  else if (a == "--expert-policy") {
    const std::string v = next();
    if (v != "lru" && v != "slru") throw std::invalid_argument("--expert-policy lru|slru");
    r.xo.policy = v == "lru" ? ExpertStreamOptions::Policy::LRU : ExpertStreamOptions::Policy::SLRU;
  }
  else if (a == "--expert-io-threads") r.xo.io_threads = unsigned(std::stoul(next()));
  else if (a == "--expert-pin-fraction") r.xo.pin_fraction = std::stod(next());
  else if (a == "--expert-prefetch") r.xo.prefetch = true;
  else if (a == "--expert-cached-io") r.xo.nocache = false;
  else if (a == "--mixed-experts") {
    const std::string v = next();
    if (v != "routed" && v != "all") throw std::invalid_argument("--mixed-experts routed|all");
    r.mixed_rows = v == "all" ? ExpertStack::Rows::All : ExpertStack::Rows::Routed;
  }
  else return false;
  return true;
}

inline bool valid_effort(const std::string& e) {
  return e == "xhigh" || e == "medium" || e == "low" || e == "high" || e == "max";
}

struct ChatMessage {
  std::string role;      // system | user | assistant
  std::string content;
};

// Chat prompt text of the model family of `arch` (the SUC1 directory's "architecture"), with the generation prompt.
// One user turn (optionally preceded by one system turn) renders exactly what suc1-generate --chat always rendered
// (the per-family templates, see its usage text). Further turns (multi-turn history, as suc1-server receives it) use
// the same per-turn markup: earlier assistant turns are rendered without their reasoning (a "<think>...</think>"
// prefix is dropped), as the Qwen3 template does. `effort` "" = the family's default.
inline std::string chat_prompt_text(const Json& arch, const std::vector<ChatMessage>& msgs, bool think,
                                    std::string effort) {
  auto trim = [](const std::string& x) {
    const char* ws = " \t\n\r\f\v";
    const auto b = x.find_first_not_of(ws);
    return b == std::string::npos ? std::string() : x.substr(b, x.find_last_not_of(ws) - b + 1);
  };
  auto strip_reasoning = [](const std::string& x) {
    const auto e = x.find("</think>");
    if (e == std::string::npos) return x;
    const auto b = x.find_first_not_of("\n", e + 8);
    return b == std::string::npos ? std::string() : x.substr(b);
  };
  std::string sys;
  bool has_sys = false;
  std::vector<ChatMessage> turns;
  for (const ChatMessage& m : msgs) {
    if (m.role == "system" || m.role == "developer") {
      sys += (has_sys ? "\n" : "") + m.content;
      has_sys = true;
    } else if (m.role == "user" || m.role == "assistant") {
      turns.push_back(m);
    } else {
      throw std::invalid_argument("unsupported chat role '" + m.role + "' (system, developer, user, assistant)");
    }
  }
  if (turns.empty() || turns.back().role != "user") throw std::invalid_argument("the last chat message must be a user turn");

  if (metal::Qwen4Config::is_qwen4(arch)) {
    // Qwen3.8-Flash-Next chat_template.jinja (no tools): with thinking a system turn carries the reasoning-effort
    // instructions (xhigh / low; medium has none), contents are trimmed, the generation prompt opens <think>
    if (effort.empty()) effort = "xhigh";
    if (effort == "high" || effort == "max")
      throw std::invalid_argument("--reasoning-effort " + effort + ": qwen4_exp takes xhigh|medium|low");
    std::string reasoning;
    if (think && effort == "xhigh")
      reasoning = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                  "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
                  "clarity in the final answer.";
    else if (think && effort == "low")
      reasoning = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                  "conclusion without unnecessary elaboration.";
    std::string t;
    const std::string s = has_sys ? trim(sys) : "";
    if (!s.empty()) t += "<|im_start|>system\n" + (reasoning.empty() ? "" : reasoning + "\n\n") + s + "<|im_end|>\n";
    else if (!reasoning.empty()) t += "<|im_start|>system\n" + reasoning + "<|im_end|>\n";
    for (const ChatMessage& m : turns)
      t += "<|im_start|>" + m.role + "\n" + trim(m.role == "assistant" ? strip_reasoning(m.content) : m.content) +
           "<|im_end|>\n";
    t += "<|im_start|>assistant\n";
    t += think ? "<think>\n" : "<think>\n\n</think>\n\n";
    return t;
  }
  if (metal::Glm5Config::is_glm5(arch)) {
    // GLM-5.3-Flash chat_template.jinja (zai-org/GLM-5.3-Flash rev eb9eb208, no tools), as transformers renders it:
    // "[gMASK]<sop>", always a "<|system|>Reasoning Effort: Max|Low|High" turn (reasoning_effort low / high, else max),
    // then the messages in order (system turns where they occur), earlier assistant turns as
    // "<|assistant|><think>REASONING</think>" (or "<think></think>") + the stripped answer, and the generation prompt
    // "<|assistant|><think>". The template ignores enable_thinking, so --no-think does not change the prompt.
    (void)think;
    if (effort.empty()) effort = "max";
    if (effort != "low" && effort != "high" && effort != "max")
      throw std::invalid_argument("--reasoning-effort " + effort + ": glm5_next takes low|high|max");
    std::string t = "[gMASK]<sop><|system|>Reasoning Effort: ";
    t += char(effort[0] - 'a' + 'A');
    t += effort.substr(1);
    for (const ChatMessage& m : msgs) {
      if (m.role == "system" || m.role == "developer") {
        if (m.role == "developer") throw std::invalid_argument("glm5_next: the chat template has no developer role");
        t += "<|system|>" + m.content;
      } else if (m.role == "user") {
        t += "<|user|>" + m.content;
      } else {
        std::string c = m.content, r;
        bool has_r = false;
        if (const auto e = c.find("</think>"); e != std::string::npos) {
          r = c.substr(0, e);
          if (const auto b = r.rfind("<think>"); b != std::string::npos) r = r.substr(b + 7);
          c = c.substr(c.rfind("</think>") + 8);
          has_r = true;
        }
        t += "<|assistant|>" + (has_r ? "<think>" + r + "</think>" : std::string("<think></think>")) + trim(c);
      }
    }
    t += "<|assistant|><think>";
    return t;
  }
  if (metal::Ds4Config::is_ds4(arch)) {
    // DeepSeek-V4 (encoding/encoding_dsv4.py encode_messages at deepseek-ai/DeepSeek-V4-Flash-0731 rev 7872f01b,
    // default BOS): BOS, in thinking mode the REASONING_EFFORT_PROMPTS text (low: none), the system text, the turns
    // (<｜User｜>text; earlier assistant turns </think>text<｜end▁of▁sentence｜>), <｜Assistant｜>, then <think> (thinking
    // mode) or </think> (chat mode)
    if (effort.empty()) effort = "low";
    if (effort != "low" && effort != "high" && effort != "max")
      throw std::invalid_argument("--reasoning-effort " + effort + ": deepseek_v4 takes low|high|max");
    std::string t = "<｜begin▁of▁sentence｜>";
    if (think && effort == "high")
      t += "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
           "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the "
           "root cause, rigorously stress-testing your logic against all potential paths, edge cases, and "
           "adversarial scenarios.\n"
           "Explicitly write out your entire deliberation process, documenting every intermediate step, considered "
           "alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
    else if (think && effort == "max")
      t += "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
           "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively "
           "decompose the problem into its most fundamental components, trace every causal chain to its root, and "
           "resolve the underlying cause rather than any surface symptom.\n"
           "Do not stop reasoning until you have independently verified the solution from multiple angles and are "
           "certain that no assumption remains unchecked and no error remains undiscovered.\n\n";
    if (has_sys) t += sys;
    for (const ChatMessage& m : turns)
      t += m.role == "user" ? "<｜User｜>" + m.content
                            : "<｜Assistant｜></think>" + strip_reasoning(m.content) + "<｜end▁of▁sentence｜>";
    t += "<｜Assistant｜>";
    t += think ? "<think>" : "</think>";
    return t;
  }
  // Qwen3 / Qwen3-MoE: transformers' apply_chat_template(..., add_generation_prompt=True, enable_thinking=think)
  std::string t;
  if (has_sys) t += "<|im_start|>system\n" + sys + "<|im_end|>\n";
  for (const ChatMessage& m : turns)
    t += "<|im_start|>" + m.role + "\n" + (m.role == "assistant" ? strip_reasoning(m.content) : m.content) +
         "<|im_end|>\n";
  t += "<|im_start|>assistant\n";
  if (!think) t += "<think>\n\n</think>\n\n";
  return t;
}

}  // namespace suc::tools
