// suc1-server: OpenAI-compatible HTTP server for one SUC1 file on the C++/Metal runtime.
//
// Endpoints: POST /v1/chat/completions, POST /v1/completions (both with "stream": true as server-sent events ending in
// "data: [DONE]"), GET /v1/models, GET /health. Minimal dependency-free HTTP/1.1 (POSIX sockets, keep-alive,
// Content-Length request bodies, chunked streaming responses). The model is loaded once; generations are serialized
// (batch 1, one at a time behind a mutex, the KV cache / recurrent state reset per request); each connection has its
// own thread, so /health and /v1/models answer during a generation. Decode as in suc1-generate: greedy qwen3 /
// qwen3_moe requests run its pipelined loop (Model::generate, --depth command buffers in flight; each token is
// streamed while the next one computes); sampling, --sync and the other families decode synchronously (one command
// buffer per token, sampling on the host), the same numerics.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "suc/deepseek_v4.hpp"
#include "suc/expert_stream.hpp"
#include "suc/glm5.hpp"
#include "suc/model.hpp"
#include "suc/live_prefix.hpp"
#include "suc/inference_engine.hpp"
#include "suc/qwen4.hpp"
#include "suc/request_params.hpp"
#include "suc/sampling.hpp"
#include "suc/tokenizer.hpp"
#include "tool_common.hpp"

using namespace suc;
using namespace suc::metal;
using namespace suc::tools;

static void usage() {
  std::fprintf(stderr,
               "usage: suc1-server FILE [--host ADDR] [--port N] [--model-name NAME] [--max-tokens N]\n"
               "                   [--temperature T] [--top-k K] [--top-p P] [--min-p P] [--no-think]\n"
               "                   [--reasoning-effort LEVEL] [--quiet] [runtime flags of suc1-generate]\n"
               "  OpenAI-compatible server: POST /v1/chat/completions, POST /v1/completions (stream: true = SSE chunks,\n"
               "  then 'data: [DONE]'), GET /v1/models, GET /health. --host default 127.0.0.1, --port default 8080\n"
               "  (0 = any free port); prints 'listening on http://HOST:PORT' on stdout once the model is loaded.\n"
               "  --model-name: the id /v1/models reports (default: the file name); requests may name any model.\n"
               "  Request parameters: max_tokens / max_completion_tokens (default --max-tokens, 256), temperature,\n"
               "  top_p, top_k, min_p, seed, stop (string or list), stream, stream_options.include_usage, n = 1;\n"
               "  chat: messages (system / developer / user / assistant, string or text-part content), reasoning_effort,\n"
               "  chat_template_kwargs.enable_thinking. Unset sampling parameters take the server defaults\n"
               "  (--temperature etc.; default greedy, = suc1-generate). Generation stops at the model's / tokenizer's\n"
               "  end tokens (finish_reason stop), a stop string (stop, excluded from the text) or max_tokens / the\n"
               "  context (--max-context, default 4096; finish_reason length). Chat prompts use the per-family template\n"
               "  of suc1-generate --chat (--no-think / --reasoning-effort = server defaults). Numeric fields are\n"
               "  validated before use (HTTP 400 naming the field): lengths are integers >= 1, temperature >= 0,\n"
               "  top_p / min_p in [0, 1], top_k an integer >= -1 (-1 / 0 = off), seed an integer >= -1 (-1 = random),\n"
               "  token-id prompts integers in [0, vocab).\n"
               "  Runtime flags (as suc1-generate): --tokenizer PATH --max-context N --dtype bf16|f16|f32 --verify\n"
               "  --tuning-cache PATH|- --no-shared-rotations --fuse-norm auto|on|off --projections fused|separate\n"
               "  --fuse-swiglu auto|on|off --fused-layout inplace|copy --legacy-matvec --single-pass-attention\n"
               "  --prefill batch|exact|token --chunk N --mixed-experts routed|all --expert-budget BYTES[K|M|G]\n"
               "  --hot-experts FILE|auto --expert-policy lru|slru --expert-io-threads N --expert-pin-fraction F\n"
               "  --expert-prefetch --expert-cached-io --sync --depth N. Decode: greedy qwen3 / qwen3_moe requests run\n"
               "  suc1-generate's pipelined loop (--depth command buffers in flight, default 2; streamed chunks are sent\n"
               "  while the next token computes); sampling, --sync and the other families: one command buffer per token.\n");
}

// ------------------------------------------------------------------------------------------ engines

// One loaded model graph behind a common interface (prefill, decode step, last logits row).
using Engine = suc::InferenceEngine;

class DenseEngine final : public Engine {   // qwen3 / qwen3_moe (suc/model.hpp)
 public:
  DenseEngine(Context& ctx, const MappedFile& mf, const ModelOptions& mo, bool batch_prefill, std::size_t chunk,
              std::size_t depth)
      : m_(ctx, mf, mo), batch_(batch_prefill), chunk_(chunk), depth_(depth) {}
  void reset() override { m_.reset(); }
  uint32_t prefill(const std::vector<uint32_t>& p) override {
    uint32_t next = 0;
    if (batch_) {
      for (std::size_t i = 0; i < p.size(); i += chunk_) {
        const std::size_t n = std::min(chunk_, p.size() - i);
        const bool last = i + n == p.size();
        auto r = m_.forward(std::vector<uint32_t>(p.begin() + i, p.begin() + i + n),
                            last ? Model::Logits::Last : Model::Logits::None);
        if (last) next = r.back();
      }
    } else {
      for (std::size_t i = 0; i < p.size(); ++i) next = m_.step(p[i], i + 1 == p.size());
    }
    return next;
  }
  uint32_t step(uint32_t t) override { return m_.step(t, true); }
  std::vector<float> last_logits() override {
    auto l = m_.logits();
    const std::size_t v = m_.config().vocab;
    return std::vector<float>(l.end() - std::ptrdiff_t(v), l.end());
  }
  bool generate(uint32_t first, std::size_t steps, const std::function<bool(uint32_t)>& on_token) override {
    if (depth_ == 0) return false;
    m_.generate(first, steps, depth_, nullptr, std::nullopt, on_token);
    return true;
  }
  Model& model() { return m_; }

 private:
  Model m_;
  bool batch_;
  std::size_t chunk_, depth_;   // depth 0: --sync
};

template <class M, class O>
class StepEngine final : public Engine {   // qwen4_exp / deepseek_v4 / glm5_next: the model's own prompt prefill (forward)
 public:
  StepEngine(Context& ctx, const MappedFile& mf, const O& o) : m_(ctx, mf, o) {}
  std::size_t checkpoint_stride() const override {
    if constexpr (requires { m_.checkpoint_stride(); }) return m_.checkpoint_stride();
    return 0;
  }
  bool save_checkpoint() override {
    if constexpr (requires { m_.save_checkpoint(); }) return m_.save_checkpoint();
    return false;
  }
  bool restore_checkpoint() override {
    if constexpr (requires { m_.restore_checkpoint(); }) return m_.restore_checkpoint();
    return false;
  }
  void clear_checkpoint() override {
    if constexpr (requires { m_.clear_checkpoint(); }) m_.clear_checkpoint();
  }
  void reset() override { m_.reset(); }
  uint32_t prefill(const std::vector<uint32_t>& p) override { return m_.forward(p, true); }
  uint32_t step(uint32_t t) override { return m_.step(t, true); }
  std::vector<float> last_logits() override { return m_.logits(); }

 private:
  M m_;
};

// ------------------------------------------------------------------------------------------ JSON output

static std::string jstr(const std::string& s) {   // JSON string literal (s is valid UTF-8)
  std::string o = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      case '\b': o += "\\b"; break;
      case '\f': o += "\\f"; break;
      default:
        if (c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof b, "\\u%04x", c);
          o += b;
        } else {
          o += char(c);
        }
    }
  }
  return o + "\"";
}

static std::string error_json(const std::string& msg, const std::string& type = "invalid_request_error") {
  return "{\"error\": {\"message\": " + jstr(msg) + ", \"type\": " + jstr(type) + ", \"param\": null, \"code\": null}}";
}

struct HttpError : std::runtime_error {
  int status;
  HttpError(int s, const std::string& m) : std::runtime_error(m), status(s) {}
};

// ------------------------------------------------------------------------------------------ HTTP

struct HttpRequest {
  std::string method, path, body;
  std::map<std::string, std::string> headers;   // lower-case names
  bool keep_alive = true;
};

static bool send_all(int fd, const std::string& d) {
  std::size_t off = 0;
  while (off < d.size()) {
    const ssize_t n = ::send(fd, d.data() + off, d.size() - off, 0);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      return false;
    }
    off += std::size_t(n);
  }
  return true;
}

static const char* reason(int s) {
  switch (s) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 411: return "Length Required";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    default: return "Error";
  }
}

static bool send_response(int fd, int status, const std::string& body, bool keep_alive,
                          const char* type = "application/json") {
  std::string h = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\nContent-Type: " + type +
                  "\r\nContent-Length: " + std::to_string(body.size()) +
                  (keep_alive ? "\r\nConnection: keep-alive\r\n\r\n" : "\r\nConnection: close\r\n\r\n");
  return send_all(fd, h + body);
}

// Reads one request (buf keeps bytes of the next pipelined request). false: connection closed / malformed.
static bool read_request(int fd, std::string& buf, HttpRequest& r) {
  constexpr std::size_t kMaxHeader = 64 * 1024, kMaxBody = 64u * 1024 * 1024;
  char tmp[16384];
  std::size_t hdr_end;
  while ((hdr_end = buf.find("\r\n\r\n")) == std::string::npos) {
    if (buf.size() > kMaxHeader) return false;
    const ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    buf.append(tmp, std::size_t(n));
  }
  std::istringstream hs(buf.substr(0, hdr_end));
  std::string line, version;
  std::getline(hs, line);
  if (!line.empty() && line.back() == '\r') line.pop_back();
  {
    std::istringstream ls(line);
    ls >> r.method >> r.path >> version;
  }
  if (r.method.empty() || r.path.empty()) return false;
  if (const auto q = r.path.find('?'); q != std::string::npos) r.path.resize(q);
  r.headers.clear();
  while (std::getline(hs, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto c = line.find(':');
    if (c == std::string::npos) continue;
    std::string k = line.substr(0, c), v = line.substr(c + 1);
    std::transform(k.begin(), k.end(), k.begin(), [](unsigned char ch) { return char(std::tolower(ch)); });
    v.erase(0, v.find_first_not_of(" \t"));
    r.headers[k] = v;
  }
  auto lower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return char(std::tolower(ch)); });
    return s;
  };
  const std::string conn = r.headers.count("connection") ? lower(r.headers["connection"]) : "";
  r.keep_alive = version == "HTTP/1.0" ? conn == "keep-alive" : conn != "close";
  buf.erase(0, hdr_end + 4);
  if (r.headers.count("transfer-encoding")) {
    send_response(fd, 411, error_json("chunked request bodies are not supported; send Content-Length"), false);
    return false;
  }
  std::size_t len = 0;
  if (auto it = r.headers.find("content-length"); it != r.headers.end()) {
    try {
      len = std::stoul(it->second);
    } catch (...) {
      return false;
    }
  }
  if (len > kMaxBody) {
    send_response(fd, 413, error_json("request body too large"), false);
    return false;
  }
  if (r.headers.count("expect") && lower(r.headers["expect"]) == "100-continue" && buf.size() < len)
    send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n");
  while (buf.size() < len) {
    const ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    buf.append(tmp, std::size_t(n));
  }
  r.body = buf.substr(0, len);
  buf.erase(0, len);
  return true;
}

// Streaming response: chunked transfer encoding of server-sent events.
class EventStream {
 public:
  explicit EventStream(int fd) : fd_(fd) {}
  bool begin() {
    return ok_ = send_all(fd_, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                               "Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n");
  }
  bool event(const std::string& data) {
    if (!ok_) return false;
    const std::string payload = "data: " + data + "\n\n";
    char h[32];
    std::snprintf(h, sizeof h, "%zx\r\n", payload.size());
    return ok_ = send_all(fd_, h + payload + "\r\n");
  }
  bool end() { return ok_ && (ok_ = send_all(fd_, "0\r\n\r\n")); }
  bool ok() const { return ok_; }

 private:
  int fd_;
  bool ok_ = false;
};

// ------------------------------------------------------------------------------------------ generation

struct GenParams {
  std::vector<uint32_t> prompt;
  std::size_t max_tokens = 256;
  SamplerConfig sc;
  std::vector<std::string> stop;
};

struct GenResult {
  std::string text, finish = "length";
  std::size_t completion_tokens = 0, cached_tokens = 0;
  double prefill_s = 0.0, decode_s = 0.0;   // decode_s: after the first token (the prefill's) to the last
  bool aborted = false;   // the client went away (streaming)
  double decode_ms_per_token() const {   // over the decode steps (the tokens after the first)
    return completion_tokens > 1 ? 1e3 * decode_s / double(completion_tokens - 1) : 0.0;
  }
};

struct Server {
  Engine* engine = nullptr;
  const Tokenizer* tok = nullptr;
  Json arch;
  std::string model_name;
  std::size_t max_ctx = 4096, default_max_tokens = 256, vocab = 0;
  SamplerConfig defaults;
  bool think = true, quiet = false;
  std::string effort;
  std::vector<uint32_t> stop_ids;
  std::time_t created = std::time(nullptr);
  std::mutex gen_mu;
  std::atomic<uint64_t> next_id{1};
};

// Length of the longest suffix of text that is a proper prefix of one of the stop strings.
static std::size_t partial_stop(const std::string& text, const std::vector<std::string>& stops) {
  std::size_t best = 0;
  for (const std::string& s : stops)
    for (std::size_t k = std::min(s.size() - 1, text.size()); k > best; --k)
      if (text.compare(text.size() - k, k, s, 0, k) == 0) {
        best = k;
        break;
      }
  return best;
}

// Text of generated ids; ids the tokenizer does not know (padding rows of the model's vocab) decode to nothing.
static std::string decode_text(const Tokenizer& tok, const std::vector<uint32_t>& ids) {
  try {
    return tok.decode(ids);
  } catch (const TokenizerError&) {
    std::vector<uint32_t> known;
    for (uint32_t t : ids) {
      try {
        tok.decode(std::vector<uint32_t>{t});
        known.push_back(t);
      } catch (const TokenizerError&) {
      }
    }
    return tok.decode(known);
  }
}

// Greedy / sampled generation; emit(delta) gets the text as it becomes final (held back: an incomplete UTF-8 tail,
// decoded as U+FFFD, and a possible start of a stop string). emit returning false aborts.
static GenResult generate(Server& S, const GenParams& g, const std::function<bool(const std::string&)>& emit) {
  GenResult res;
  const auto t0 = std::chrono::steady_clock::now();
  struct CacheGuard { Engine* e; bool success=false; ~CacheGuard(){if(!success)e->invalidate_prefix();} } guard{S.engine};
  uint32_t next = S.engine->prepare(g.prompt);
  res.cached_tokens = S.engine->cached_tokens();
  Sampler sampler(g.sc);
  if (!g.sc.greedy()) {
    const auto l = S.engine->last_logits();
    next = sampler.sample(l.data(), l.size());
  }
  const auto t1 = std::chrono::steady_clock::now();
  res.prefill_s = std::chrono::duration<double>(t1 - t0).count();
  const std::size_t room = S.max_ctx - g.prompt.size();   // fed tokens stay <= max_ctx
  const std::size_t limit = std::min(g.max_tokens, room);
  std::vector<uint32_t> ids;
  std::size_t sent = 0;
  std::string text, emitted;
  static const std::string kRepl = "\xEF\xBF\xBD";
  // takes one generated token; false ends the generation (end token, stop string, client gone, limit reached)
  auto consume = [&](uint32_t t) {
    ++res.completion_tokens;
    if (std::find(S.stop_ids.begin(), S.stop_ids.end(), t) != S.stop_ids.end()) {
      res.finish = "stop";
      return false;
    }
    ids.push_back(t);
    text = decode_text(*S.tok, ids);
    std::size_t hit = std::string::npos;
    for (const std::string& s : g.stop) {
      const std::size_t from = sent >= s.size() ? sent - s.size() + 1 : 0;
      const std::size_t p = text.find(s, from);
      if (p != std::string::npos && p < hit) hit = p;
    }
    if (hit != std::string::npos) {
      text.resize(std::max(hit, sent));
      res.finish = "stop";
      return false;
    }
    std::size_t safe = text.size();
    while (safe >= 3 && text.compare(safe - 3, 3, kRepl) == 0) safe -= 3;
    safe -= partial_stop(text.substr(0, safe), g.stop);
    if (safe > sent && text.compare(0, sent, emitted) == 0) {   // emitted text stays a prefix
      const std::string d = text.substr(sent, safe - sent);
      if (!emit(d)) {
        res.aborted = true;
        return false;
      }
      emitted += d;
      sent = safe;
    }
    return ids.size() < limit;
  };
  if (limit > 0 && consume(next) && limit > 1) {
    // greedy: the pipelined loop where the engine has one (consume runs while the next step is on the GPU)
    if (!(g.sc.greedy() && S.engine->generate(next, limit - 1, consume))) {
      do {
        next = S.engine->step(next);
        if (!g.sc.greedy()) {
          const auto l = S.engine->last_logits();
          next = sampler.sample(l.data(), l.size());
        }
      } while (consume(next));
    }
  }
  if (text.compare(0, sent, emitted) != 0) text = emitted + (text.size() > sent ? text.substr(sent) : "");
  if (!res.aborted && text.size() > sent && !emit(text.substr(sent))) res.aborted = true;
  res.text = text;
  res.decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  guard.success = !res.aborted;
  return res;
}

// ------------------------------------------------------------------------------------------ request handling

static const Json* field(const Json& o, const char* k) {
  const Json* v = o.find(k);
  return v && !v->is_null() ? v : nullptr;
}

static double num(const Json& v, const char* name) {
  if (!v.is_number()) throw HttpError(400, std::string("'") + name + "' must be a number");
  return v.is_int() ? double(v.as_int()) : v.as_double();
}

static std::string text_content(const Json& c) {   // string or [{"type": "text", "text": ...}]
  if (c.is_null()) return "";
  if (c.is_string()) return c.as_string();
  if (c.is_array()) {
    std::string t;
    for (const Json& p : c.as_array()) {
      const Json* ty = p.is_object() ? p.find("type") : nullptr;
      if (!ty || !ty->is_string() || ty->as_string() != "text" || !p.find("text") || !p["text"].is_string())
        throw HttpError(400, "only text content parts are supported");
      t += p["text"].as_string();
    }
    return t;
  }
  throw HttpError(400, "message content must be a string or a list of text parts");
}

// Common sampling / length / stop parameters.
static GenParams common_params(const Server& S, const Json& b) {
  GenParams g;
  g.sc = S.defaults;
  g.max_tokens = S.default_max_tokens;
  // numeric fields are validated before any narrowing conversion (suc/request_params.hpp; invalid -> HTTP 400)
  if (const Json* v = field(b, "max_completion_tokens")) g.max_tokens = req::length(*v, "max_completion_tokens");
  else if (const Json* v2 = field(b, "max_tokens")) g.max_tokens = req::length(*v2, "max_tokens");
  constexpr double kInf = std::numeric_limits<double>::max();
  if (const Json* v = field(b, "temperature")) g.sc.temperature = req::number(*v, "temperature", 0.0, kInf);
  if (const Json* v = field(b, "top_p")) g.sc.top_p = req::number(*v, "top_p", 0.0, 1.0);
  if (const Json* v = field(b, "top_k")) g.sc.top_k = req::top_k(*v);   // -1 / 0 = off (vLLM convention)
  if (const Json* v = field(b, "min_p")) g.sc.min_p = req::number(*v, "min_p", 0.0, 1.0);
  uint64_t seed = 0;
  if (const Json* v = field(b, "seed"); v && req::seed(*v, seed)) g.sc.seed = seed;
  else g.sc.seed = (uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();   // absent or -1: random
  if (const Json* v = field(b, "n"); v && num(*v, "n") != 1) throw HttpError(400, "only n = 1 is supported");
  if (const Json* v = field(b, "best_of"); v && num(*v, "best_of") != 1) throw HttpError(400, "only best_of = 1 is supported");
  for (const char* k : {"presence_penalty", "frequency_penalty"})
    if (const Json* v = field(b, k); v && num(*v, k) != 0) throw HttpError(400, std::string(k) + " is not supported");
  if (const Json* v = field(b, "repetition_penalty"); v && num(*v, "repetition_penalty") != 1)
    throw HttpError(400, "repetition_penalty is not supported");
  if (const Json* v = field(b, "logprobs"); v && !(v->is_bool() && !v->as_bool()))
    throw HttpError(400, "logprobs are not supported");
  if (const Json* v = field(b, "stop")) {
    if (v->is_string()) g.stop.push_back(v->as_string());
    else if (v->is_array())
      for (const Json& s : v->as_array()) {
        if (!s.is_string()) throw HttpError(400, "stop must be a string or a list of strings");
        g.stop.push_back(s.as_string());
      }
    else throw HttpError(400, "stop must be a string or a list of strings");
    g.stop.erase(std::remove(g.stop.begin(), g.stop.end(), std::string()), g.stop.end());
  }
  return g;
}

static void check_prompt(const Server& S, const std::vector<uint32_t>& p) {
  if (p.empty()) throw HttpError(400, "the prompt encodes to no tokens");
  if (p.size() >= S.max_ctx)
    throw HttpError(400, "the prompt has " + std::to_string(p.size()) + " tokens; the context is " +
                             std::to_string(S.max_ctx) + " (--max-context)");
  for (uint32_t t : p)
    if (t >= S.vocab)
      throw HttpError(400, "token id " + std::to_string(t) + " out of range (vocab " + std::to_string(S.vocab) + ")");
}

static std::string usage_json(std::size_t prompt, std::size_t completion, std::size_t cached = 0) {
  return "{\"prompt_tokens\": " + std::to_string(prompt) + ", \"completion_tokens\": " + std::to_string(completion) +
         ", \"total_tokens\": " + std::to_string(prompt + completion) + ",\"prompt_tokens_details\":{\"cached_tokens\":" + std::to_string(cached) + "}}";
}

// Handles one completion request (chat or text). Returns false when the connection must close.
static bool handle_completion(Server& S, int fd, const HttpRequest& rq, bool chat) {
  Json b;
  try {
    b = Json::parse(rq.body);
  } catch (const std::exception& e) {
    return send_response(fd, 400, error_json(std::string("invalid JSON body: ") + e.what()), rq.keep_alive) &&
           rq.keep_alive;
  }
  GenParams g;
  bool stream = false, include_usage = false;
  try {
    if (!b.is_object()) throw HttpError(400, "the body must be a JSON object");
    g = common_params(S, b);
    if (const Json* v = field(b, "stream")) stream = v->is_bool() && v->as_bool();
    if (const Json* so = field(b, "stream_options"); so && so->is_object())
      if (const Json* iu = field(*so, "include_usage")) include_usage = iu->is_bool() && iu->as_bool();
    if (chat) {
      const Json* m = field(b, "messages");
      if (!m || !m->is_array() || m->size() == 0) throw HttpError(400, "'messages' must be a non-empty list");
      std::vector<ChatMessage> msgs;
      for (const Json& x : m->as_array()) {
        if (!x.is_object() || !x.find("role") || !x["role"].is_string()) throw HttpError(400, "every message needs a role");
        msgs.push_back({x["role"].as_string(), text_content(x.find("content") ? x["content"] : Json())});
      }
      bool think = S.think;
      std::string effort = S.effort;
      if (const Json* k = field(b, "chat_template_kwargs"); k && k->is_object())
        if (const Json* t = field(*k, "enable_thinking"); t && t->is_bool()) think = t->as_bool();
      if (const Json* e = field(b, "reasoning_effort")) {
        if (!e->is_string() || !valid_effort(e->as_string()))
          throw HttpError(400, "reasoning_effort: xhigh|medium|low (qwen4_exp) or low|high|max (deepseek_v4, glm5_next)");
        effort = e->as_string();
      }
      std::string text;
      try {
        text = chat_prompt_text(S.arch, msgs, think, effort);
      } catch (const std::invalid_argument& e) {
        throw HttpError(400, e.what());
      }
      std::lock_guard<std::mutex> lk(S.gen_mu);   // the tokenizer's BPE cache is not thread-safe
      g.prompt = S.tok->encode(text);
    } else {
      const Json* p = field(b, "prompt");
      if (!p) throw HttpError(400, "'prompt' is required");
      if (field(b, "suffix")) throw HttpError(400, "suffix is not supported");
      if (const Json* e = field(b, "echo"); e && e->is_bool() && e->as_bool()) throw HttpError(400, "echo is not supported");
      const Json* one = p;
      if (p->is_array() && p->size() > 0 && (p->at(0).is_string() || p->at(0).is_array())) {
        if (p->size() != 1) throw HttpError(400, "only one prompt per request is supported");
        one = &p->at(0);
      }
      if (one->is_string()) {
        std::lock_guard<std::mutex> lk(S.gen_mu);
        g.prompt = S.tok->encode(one->as_string());
      } else if (one->is_array()) {
        for (const Json& t : one->as_array()) g.prompt.push_back(req::token_id(t, S.vocab));   // checked, then narrowed
      } else {
        throw HttpError(400, "prompt must be a string, a list of token ids, or a one-element list of either");
      }
    }
    check_prompt(S, g.prompt);
  } catch (const HttpError& e) {
    return send_response(fd, e.status, error_json(e.what()), rq.keep_alive) && rq.keep_alive;
  } catch (const std::exception& e) {
    return send_response(fd, 400, error_json(e.what()), rq.keep_alive) && rq.keep_alive;
  }

  const std::string id = (chat ? "chatcmpl-" : "cmpl-") + std::to_string(S.next_id++);
  const std::string created = std::to_string(std::time(nullptr));
  const std::string head = "{\"id\": " + jstr(id) + ", \"object\": " +
                           jstr(chat ? (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion") +
                           ", \"created\": " + created + ", \"model\": " + jstr(S.model_name) + ", ";
  auto choice = [&](const std::string& delta, const std::string* finish, bool first) {
    const std::string fr = finish ? jstr(*finish) : "null";
    if (chat) {
      std::string d = first ? "{\"role\": \"assistant\", \"content\": " + jstr(delta) + "}"
                            : finish ? "{}" : "{\"content\": " + jstr(delta) + "}";
      return "{\"index\": 0, \"delta\": " + d + ", \"logprobs\": null, \"finish_reason\": " + fr + "}";
    }
    return "{\"index\": 0, \"text\": " + jstr(delta) + ", \"logprobs\": null, \"finish_reason\": " + fr + "}";
  };

  GenResult res;
  std::string error;
  if (stream) {
    EventStream es(fd);
    if (!es.begin()) return false;
    {
      std::lock_guard<std::mutex> lk(S.gen_mu);
      if (chat) es.event(head + "\"choices\": [" + choice("", nullptr, true) + "]}");
      try {
        res = generate(S, g, [&](const std::string& d) {
          return es.event(head + "\"choices\": [" + choice(d, nullptr, false) + "]}");
        });
      } catch (const std::exception& e) {
        error = e.what();
      }
    }
    if (!error.empty()) {
      es.event(error_json(error, "server_error"));
    } else if (!res.aborted) {
      es.event(head + "\"choices\": [" + choice("", &res.finish, false) + "]}");
      if (include_usage)
        es.event(head + "\"choices\": [], \"usage\": " + usage_json(g.prompt.size(), res.completion_tokens, res.cached_tokens) + "}");
    }
    es.event("[DONE]");
    es.end();
    if (!S.quiet && error.empty())
      std::fprintf(stderr, "%s: prompt %zu tokens (%.1f ms), %zu generated (decode %.3f ms/token), finish %s%s\n",
                   id.c_str(), g.prompt.size(), 1e3 * res.prefill_s, res.completion_tokens, res.decode_ms_per_token(),
                   res.finish.c_str(),
                   res.aborted ? " (client gone)" : "");
    return es.ok() && rq.keep_alive;
  }
  {
    std::lock_guard<std::mutex> lk(S.gen_mu);
    try {
      res = generate(S, g, [](const std::string&) { return true; });
    } catch (const std::exception& e) {
      error = e.what();
    }
  }
  if (!error.empty()) return send_response(fd, 500, error_json(error, "server_error"), rq.keep_alive) && rq.keep_alive;
  if (!S.quiet)
    std::fprintf(stderr, "%s: prompt %zu tokens (%.1f ms), %zu generated (decode %.3f ms/token), finish %s\n",
                 id.c_str(), g.prompt.size(), 1e3 * res.prefill_s, res.completion_tokens, res.decode_ms_per_token(),
                 res.finish.c_str());
  std::string body = head + "\"choices\": [";
  if (chat)
    body += "{\"index\": 0, \"message\": {\"role\": \"assistant\", \"content\": " + jstr(res.text) +
            "}, \"logprobs\": null, \"finish_reason\": " + jstr(res.finish) + "}";
  else
    body += "{\"index\": 0, \"text\": " + jstr(res.text) + ", \"logprobs\": null, \"finish_reason\": " +
            jstr(res.finish) + "}";
  body += "], \"usage\": " + usage_json(g.prompt.size(), res.completion_tokens, res.cached_tokens) + "}";
  return send_response(fd, 200, body, rq.keep_alive) && rq.keep_alive;
}

static void serve_connection(Server& S, int fd) {
  std::string buf;
  HttpRequest rq;
  while (read_request(fd, buf, rq)) {
    bool keep = rq.keep_alive;
    try {
      if (rq.path == "/health" || rq.path == "/v1/health") {
        keep = send_response(fd, 200, "{\"status\": \"ok\", \"model\": " + jstr(S.model_name) + "}", keep) && keep;
      } else if (rq.path == "/v1/models" || rq.path == "/v1/models/" + S.model_name) {
        if (rq.method != "GET") {
          keep = send_response(fd, 405, error_json("use GET"), keep) && keep;
        } else {
          const std::string m = "{\"id\": " + jstr(S.model_name) + ", \"object\": \"model\", \"created\": " +
                                std::to_string(S.created) + ", \"owned_by\": \"suc\", \"max_model_len\": " +
                                std::to_string(S.max_ctx) + "}";
          keep = send_response(fd, 200, rq.path == "/v1/models" ? "{\"object\": \"list\", \"data\": [" + m + "]}" : m,
                               keep) && keep;
        }
      } else if (rq.path == "/v1/chat/completions" || rq.path == "/v1/completions") {
        if (rq.method != "POST") keep = send_response(fd, 405, error_json("use POST"), keep) && keep;
        else keep = handle_completion(S, fd, rq, rq.path == "/v1/chat/completions");
      } else {
        keep = send_response(fd, 404, error_json("no route " + rq.method + " " + rq.path), keep) && keep;
      }
    } catch (const std::exception& e) {
      send_response(fd, 500, error_json(e.what(), "server_error"), false);
      keep = false;
    }
    if (!keep) break;
  }
  ::close(fd);
}

// ------------------------------------------------------------------------------------------ main

int main(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
    usage();
    return argc < 2 ? 2 : 0;
  }
  std::string path = argv[1], host = "127.0.0.1", model_name;
  int port = 8080;
  RuntimeFlags rf;
  rf.max_ctx = 4096;
  Server S;
  try {
    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (parse_runtime_flag(a, next, rf)) continue;
      if (a == "--host") host = next();
      else if (a == "--port") port = std::stoi(next());
      else if (a == "--model-name") model_name = next();
      else if (a == "--max-tokens") S.default_max_tokens = std::stoul(next());
      else if (a == "--temperature") S.defaults.temperature = std::stod(next());
      else if (a == "--top-k") S.defaults.top_k = std::stoul(next());
      else if (a == "--top-p") S.defaults.top_p = std::stod(next());
      else if (a == "--min-p") S.defaults.min_p = std::stod(next());
      else if (a == "--no-think") S.think = false;
      else if (a == "--reasoning-effort") {
        S.effort = next();
        if (!valid_effort(S.effort))
          throw std::invalid_argument("--reasoning-effort xhigh|medium|low (qwen4_exp) or low|high|max (deepseek_v4, glm5_next)");
      }
      else if (a == "--quiet") S.quiet = true;
      else if (a == "-h" || a == "--help") {
        usage();
        return 0;
      } else throw std::invalid_argument("unknown option " + a);
    }
    if (rf.max_ctx < 2) throw std::invalid_argument("--max-context must be >= 2");
    if (port < 0 || port > 65535) throw std::invalid_argument("--port 0..65535");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "suc1-server: %s\n", e.what());
    usage();
    return 2;
  }
  if (model_name.empty()) model_name = path.substr(path.rfind('/') == std::string::npos ? 0 : path.rfind('/') + 1);
  S.model_name = model_name;
  S.max_ctx = rf.max_ctx;
  std::signal(SIGPIPE, SIG_IGN);

  try {
    const auto t0 = std::chrono::steady_clock::now();
    OpenOptions oo;
    oo.verify_checksum = rf.verify;
    Suc1File f = Suc1File::open(path, oo);
    S.arch = f.directory()["architecture"];
    const std::string tp = rf.tokenizer_path.empty() ? resolve_tokenizer_path(f.tokenizer_id()) : rf.tokenizer_path;
    Tokenizer tok = Tokenizer::from_file(tp);
    S.tok = &tok;
    Tuning tu;
    tu.tuning_cache = rf.tuning_cache;
    Context ctx(tu);
    const bool is_q4 = Qwen4Config::is_qwen4(S.arch), is_glm = Glm5Config::is_glm5(S.arch),
               is_ds4 = Ds4Config::is_ds4(S.arch);
    std::vector<std::pair<uint64_t, uint64_t>> exclude;
    if (is_q4) exclude = qwen4_table_ranges(f);
    std::unique_ptr<ExpertStream> xs;
    ExpertStreamOptions xo = rf.xo;
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
    auto act_of = [&](const std::string& torch_dtype) {
      if (!rf.dtype.empty()) return parse_act(rf.dtype);
      return torch_dtype == "float16" ? Act::F16 : torch_dtype == "float32" ? Act::F32 : Act::BF16;
    };

    std::unique_ptr<Engine> engine;
    std::string family;
    if (is_q4) {
      const Qwen4Config qc = Qwen4Config::from_architecture(S.arch);
      Qwen4Options qo;
      qo.act = act_of(qc.torch_dtype);
      qo.max_context = rf.max_ctx;
      // batched (layer-major) prompt prefill in chunks of --chunk (64): the fast GEMM path by default, the row-exact
      // one (decode-step bits) with --prefill exact; --prefill token: decode steps
      qo.layer_prefill = rf.batch_prefill;
      qo.exact_prefill = rf.exact_prefill;
      qo.chunk = suc::service_prefill_chunk(qc.dense ? "qwen3_5" : "qwen4_exp", rf.max_ctx, rf.chunk_given ? rf.chunk : 0, rf.exact_prefill || prefill_exact_env(), rf.batch_prefill, bool(xs));
      qo.expert_stream = xs.get();
      engine = std::make_unique<StepEngine<Qwen4Model, Qwen4Options>>(ctx, mf, qo);
      S.stop_ids.push_back(qc.eos);
      S.vocab = qc.vocab;
      family = "qwen4_exp";
    } else if (is_glm) {
      if (xs && rf.mixed_rows == ExpertStack::Rows::All)
        throw std::invalid_argument("--expert-budget with --mixed-experts all: streamed stacks run routed rows only");
      const Glm5Config gc = Glm5Config::from_architecture(S.arch);
      Glm5Options go;
      go.act = act_of(gc.torch_dtype);
      go.max_context = rf.max_ctx;
      // batched (not row-exact) layer-major prompt chunks, 512 rows unless --chunk is given; --prefill exact: decode
      // steps (streamed: row-exact layer-major chunks of 64), as before 2026-10-10
      go.layer_prefill = rf.batch_prefill;
      go.exact_prefill = rf.exact_prefill;
      go.chunk = suc::service_prefill_chunk("glm5_next", rf.max_ctx, rf.chunk_given ? rf.chunk : 0, rf.exact_prefill || prefill_exact_env(), rf.batch_prefill, bool(xs));
      go.mixed_rows = rf.mixed_rows;
      go.expert_stream = xs.get();
      engine = std::make_unique<StepEngine<Glm5Model, Glm5Options>>(ctx, mf, go);
      S.stop_ids = gc.eos;
      for (uint32_t t : tok.turn_end_ids())   // <|user|>, <|observation|> also when the file's config lacks them
        if (std::find(S.stop_ids.begin(), S.stop_ids.end(), t) == S.stop_ids.end()) S.stop_ids.push_back(t);
      S.vocab = gc.vocab;
      family = "glm5_next";
    } else if (is_ds4) {
      if (xs && rf.mixed_rows == ExpertStack::Rows::All)
        throw std::invalid_argument("--expert-budget with --mixed-experts all: streamed stacks run routed rows only");
      const Ds4Config dc = Ds4Config::from_architecture(S.arch);
      Ds4Options dopt;
      dopt.act = act_of(dc.torch_dtype);
      dopt.max_context = rf.max_ctx;
      // as qwen4_exp: fast batched chunks by default, row-exact with --prefill exact, decode steps with token
      dopt.layer_prefill = rf.batch_prefill;
      dopt.exact_prefill = rf.exact_prefill;
      dopt.chunk = suc::service_prefill_chunk("deepseek_v4", rf.max_ctx, rf.chunk_given ? rf.chunk : 0, rf.exact_prefill || prefill_exact_env(), rf.batch_prefill, bool(xs));
      dopt.mixed_rows = rf.mixed_rows;
      dopt.expert_stream = xs.get();
      engine = std::make_unique<StepEngine<Ds4Model, Ds4Options>>(ctx, mf, dopt);
      S.stop_ids.push_back(dc.eos);
      S.vocab = dc.vocab;
      family = "deepseek_v4";
    } else {
      const ModelConfig cfg = ModelConfig::from_architecture(S.arch);
      ModelOptions mo;
      mo.act = act_of(cfg.torch_dtype);
      mo.max_context = rf.max_ctx;
      mo.share_rotations = rf.share;
      mo.fuse_norm = rf.fuse_norm;
      mo.fuse_projections = rf.fuse_projections;
      mo.fused_in_place = rf.fused_in_place;
      mo.mlx_gemv = rf.mlx_gemv;
      mo.sdpa_2pass = rf.sdpa_2pass;
      mo.fuse_swiglu = rf.fuse_swiglu;
      mo.chunk = rf.chunk;
      mo.expert_stream = xs.get();
      std::size_t chunk = rf.chunk;   // streamed experts: prefill chunks whose distinct experts fit the slots
      if (xs && cfg.top_k) chunk = std::max<std::size_t>(1, std::min(chunk, xs->min_max_distinct() / cfg.top_k));
      engine = std::make_unique<DenseEngine>(ctx, mf, mo, rf.batch_prefill, chunk, rf.sync_decode ? 0 : rf.depth);
      if (const Json* e = S.arch.find("eos_token_id")) {
        if (e->is_int()) S.stop_ids.push_back(uint32_t(e->as_uint()));
        else if (e->is_array())
          for (const Json& x : e->as_array())
            if (x.is_int()) S.stop_ids.push_back(uint32_t(x.as_uint()));
      }
      family = cfg.model_type;
      S.vocab = cfg.vocab;
    }
    for (uint32_t t : tok.eos_ids())
      if (std::find(S.stop_ids.begin(), S.stop_ids.end(), t) == S.stop_ids.end()) S.stop_ids.push_back(t);
    engine = std::make_unique<suc::LivePrefixEngine<Engine>>(std::move(engine));
    S.engine = engine.get();
    S.engine->step(0);   // warm-up: compile the decode pipelines, then reset
    S.engine->reset();
    if (xs) xs->reset_stats();
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const int ls = ::socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) throw std::runtime_error("socket() failed");
    const int one = 1;
    ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    if (::inet_pton(AF_INET, host == "localhost" ? "127.0.0.1" : host.c_str(), &addr.sin_addr) != 1)
      throw std::invalid_argument("--host must be an IPv4 address (or localhost)");
    if (::bind(ls, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0)
      throw std::runtime_error("cannot bind " + host + ":" + std::to_string(port) + ": " + std::strerror(errno));
    if (::listen(ls, 64) < 0) throw std::runtime_error("listen() failed");
    socklen_t al = sizeof addr;
    ::getsockname(ls, reinterpret_cast<sockaddr*>(&addr), &al);
    if (!S.quiet)
      std::fprintf(stderr, "suc1-server: %s (%s, tokenizer %s), device %s, context %zu, loaded in %.2f s\n",
                   path.c_str(), family.c_str(), tp.c_str(), ctx.device_name().c_str(), S.max_ctx, load_s);
    std::printf("listening on http://%s:%d\n", host.c_str(), int(ntohs(addr.sin_port)));
    std::fflush(stdout);
    for (;;) {
      const int fd = ::accept(ls, nullptr, nullptr);
      if (fd < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error(std::string("accept() failed: ") + std::strerror(errno));
      }
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
      ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
      std::thread([&S, fd] { serve_connection(S, fd); }).detach();
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "suc1-server: %s\n", e.what());
    return 1;
  }
}
