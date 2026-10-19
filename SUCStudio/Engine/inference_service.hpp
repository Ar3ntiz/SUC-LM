// Extracted from the parent cpp/tools/suc1_server.cpp (app-specific integration); kernels and numeric paths unchanged.
#pragma once
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
#include "tool_common.hpp"  // resolved exclusively from ../cpp/tools by build scripts
#include "studio_support.hpp"
#include "memory_policy.hpp"
#include <sys/sysctl.h>
#include <sys/stat.h>

using namespace suc;
using namespace suc::metal;
using namespace suc::tools;

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
        if(studio_cancelled()) throw std::runtime_error("Request cancelled");
        const std::size_t n = std::min(chunk_, p.size() - i);
        const bool last = i + n == p.size();
        auto r = m_.forward(std::vector<uint32_t>(p.begin() + i, p.begin() + i + n),
                            last ? Model::Logits::Last : Model::Logits::None);
        if (last) next = r.back();
        suc::prefill_progress_advance(n);
      }
    } else {
      for (std::size_t i = 0; i < p.size(); ++i) { if(studio_cancelled()) throw std::runtime_error("Request cancelled"); next = m_.step(p[i], i + 1 == p.size()); suc::prefill_progress_advance(1); }
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
class StepEngine final : public Engine {   // qwen4_exp / deepseek_v4 / glm5_next: the prompt as decode steps
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

static std::string model_log_prefix(const std::string& text) {
  constexpr std::size_t limit=64*1024;
  if(text.size()<=limit)return text;
  std::size_t end=limit;
  while(end>0&&(static_cast<unsigned char>(text[end])&0xc0)==0x80)--end;
  return text.substr(0,end)+"\n[… input completo omesso dal registro …]";
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
    case 401: return "Unauthorized";
    case 429: return "Too Many Requests";
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
  std::string retry = status == 429 ? "Retry-After: 1\r\n" : "";
  std::string h = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\nContent-Type: " + type +
                  "\r\n" + retry + "Content-Length: " + std::to_string(body.size()) +
                  (keep_alive ? "\r\nConnection: keep-alive\r\n\r\n" : "\r\nConnection: close\r\n\r\n");
  return send_all(fd, h + body);
}

// Reads one request (buf keeps bytes of the next pipelined request). false: connection closed / malformed.
static bool read_request(int fd, std::string& buf, HttpRequest& r) {
  constexpr std::size_t kMaxHeader = 64 * 1024, kMaxBody = 2u * 1024 * 1024;
  char tmp[16384];
  std::size_t hdr_end;
  while ((hdr_end = buf.find("\r\n\r\n")) == std::string::npos) {
    if (buf.size() > kMaxHeader) return false;
    const ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    buf.append(tmp, std::size_t(n));
  }
  if (hdr_end > kMaxHeader) return false;
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
    if (r.headers.count(k)) return false;
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
  if (!studio_authorize(fd, r.path, r.headers)) return false;
  std::size_t len = 0;
  if (auto it = r.headers.find("content-length"); it != r.headers.end()) {
    try {
      std::size_t used = 0;
      if (it->second.empty() || it->second.find_first_not_of("0123456789") != std::string::npos) return false;
      len = std::stoul(it->second, &used);
      if (used != it->second.size()) return false;
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
                               "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n");
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
  ExpertStream* expert_stream = nullptr;
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

// Called only by the serialized generation owner, never by the status thread.
// Counters are snapshots for this request; cache contents persist across requests.
static void studio_expert_snapshot(Server& S, const char* phase) {
  if(!S.expert_stream || !current_request) return;
  const auto& stats=S.expert_stream->stats();
  event("expert_stats",current_request->id,"{\"phase\":"+jstr(phase)+
    ",\"bytes_read\":"+std::to_string(stats.bytes_read)+",\"hit_rate\":"+std::to_string(stats.hit_rate())+
    ",\"wait_s\":"+std::to_string(stats.wait_seconds)+",\"pool_bytes\":"+std::to_string(S.expert_stream->pool_bytes())+"}");
}

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
static GenResult generate(Server& S, const GenParams& g, const std::function<bool(const std::string&)>& emit,
                          const std::function<void(const std::string&)>& progress_emit = {},
                          const std::function<void(const std::string&)>& metrics_emit = {}) {
  GenResult res;
  const auto t0 = std::chrono::steady_clock::now();
  struct CacheGuard { Engine* e; bool success=false; ~CacheGuard(){if(!success)e->invalidate_prefix();} } guard{S.engine};
  if(S.expert_stream) S.expert_stream->reset_stats();
  uint32_t next;
  {
    suc::PrefillProgressScope progress([&](std::size_t processed,std::size_t total,std::size_t cached){
      const auto payload="{\"processed\":"+std::to_string(processed)+",\"total\":"+std::to_string(total)+",\"cache\":"+std::to_string(cached)+"}";
      event("prefill_progress",studio_request_id(),payload);
      if(progress_emit) progress_emit(payload);
      if(studio_cancelled()) throw std::runtime_error("Request cancelled");
    });
    next = S.engine->prepare(g.prompt);
  }
  res.cached_tokens = S.engine->cached_tokens();
  if (studio_cancelled()) throw std::runtime_error("Request cancelled");
  Sampler sampler(g.sc);
  if (!g.sc.greedy()) {
    const auto l = S.engine->last_logits();
    next = sampler.sample(l.data(), l.size());
  }
  const auto t1 = std::chrono::steady_clock::now();
  res.prefill_s = std::chrono::duration<double>(t1 - t0).count();
  studio_prefill(g.prompt.size()-res.cached_tokens, res.prefill_s);
  studio_expert_snapshot(S,"prefill");
  const std::size_t room = S.max_ctx - g.prompt.size();   // fed tokens stay <= max_ctx
  const std::size_t limit = std::min(g.max_tokens, room);
  std::vector<uint32_t> ids;
  std::size_t sent = 0;
  std::string text, emitted;
  static const std::string kRepl = "\xEF\xBF\xBD";
  // takes one generated token; false ends the generation (end token, stop string, client gone, limit reached)
  auto last_live_metrics = t1;
  auto consume = [&](uint32_t t) {
    if (studio_cancelled()) { res.aborted = true; return false; }
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
    const auto now=std::chrono::steady_clock::now();
    if(res.completion_tokens>1 && now-last_live_metrics>=std::chrono::milliseconds(250)) {
      last_live_metrics=now;
      const auto elapsed=std::chrono::duration<double>(now-t1).count();
      const auto payload="{\"prompt_tokens\":"+std::to_string(g.prompt.size())+
        ",\"cached_tokens\":"+std::to_string(res.cached_tokens)+
        ",\"prefill_tokens\":"+std::to_string(g.prompt.size()-res.cached_tokens)+
        ",\"completion_tokens\":"+std::to_string(res.completion_tokens)+
        ",\"prefill_s\":"+std::to_string(res.prefill_s)+",\"decode_s\":"+std::to_string(elapsed)+"}";
      event("metrics",studio_request_id(),payload);
      if(metrics_emit) metrics_emit(payload);
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
  studio_metrics(g.prompt.size()-res.cached_tokens, res.completion_tokens, res.prefill_s, res.decode_s);
  studio_expert_snapshot(S,"completed");
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

// Additive extension: timings measured by the same inference phases as local telemetry.
static std::string timing_json(const GenResult& result) {
  return "{\"prefill_s\":" + std::to_string(result.prefill_s) + ",\"decode_s\":" + std::to_string(result.decode_s) + "}";
}

static std::string usage_json(std::size_t prompt, std::size_t completion, std::size_t cached = 0) {
  return "{\"prompt_tokens\": " + std::to_string(prompt) + ", \"completion_tokens\": " + std::to_string(completion) +
         ", \"total_tokens\": " + std::to_string(prompt + completion) + ",\"prompt_tokens_details\":{\"cached_tokens\":" + std::to_string(cached) + "}}";
}

static GenParams parse_generation(Server& S, const Json& b, bool chat) {
  GenParams g;
    if (!b.is_object()) throw HttpError(400, "the body must be a JSON object");
    if (const Json* m = field(b, "model"); m && (!m->is_string() || m->as_string() != S.model_name)) throw HttpError(400, "Unknown model");
    g = common_params(S, b);
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
        g.prompt = S.tok->encode(one->as_string());
      } else if (one->is_array()) {
        for (const Json& t : one->as_array()) g.prompt.push_back(req::token_id(t, S.vocab));   // checked, then narrowed
      } else {
        throw HttpError(400, "prompt must be a string, a list of token ids, or a one-element list of either");
      }
    }
    check_prompt(S, g.prompt);
    if (g.max_tokens > S.max_ctx - g.prompt.size()) throw HttpError(400, "Context budget exceeded: reduce history or max_tokens");
  return g;
}

// Handles one completion request (chat or text). Returns false when the connection must close.
static bool handle_completion(Server& S, int fd, const HttpRequest& rq, bool chat) {
  StudioRequestContext request("api-" + std::to_string(S.next_id++), "api");
  StudioTurn turn;
  if (!turn.enter()) return send_response(fd, 429, error_json("Queue full or cancelled"), false) && false;
  studio_started();
  Json b;
  try {
    if (!bounded_json(rq.body)) throw std::invalid_argument("Invalid JSON nesting");
    b = Json::parse(rq.body);
  } catch (const std::exception& e) {
    return send_response(fd, 400, error_json(std::string("invalid JSON body: ") + e.what()), rq.keep_alive) &&
           rq.keep_alive;
  }
  GenParams g;
  bool stream = false, include_usage = false;
  try {
    g = parse_generation(S, b, chat);
    if (const Json* v = field(b, "stream")) stream = v->is_bool() && v->as_bool();
    if (const Json* so = field(b, "stream_options"); so && so->is_object())
      if (const Json* iu = field(*so, "include_usage")) include_usage = iu->is_bool() && iu->as_bool();
  } catch (const HttpError& e) {
    return send_response(fd, e.status, error_json(e.what()), rq.keep_alive) && rq.keep_alive;
  } catch (const std::exception& e) {
    return send_response(fd, 400, error_json(e.what()), rq.keep_alive) && rq.keep_alive;
  }

  const std::string id = studio_request_id();
  if(model_io_logging)event("model_input",id,"{\"text\":"+jstr(model_log_prefix(rq.body))+",\"origin\":\"api\"}");
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
      // StudioTurn serializes tokenizer and inference.
      if (chat) es.event(head + "\"choices\": [" + choice("", nullptr, true) + "]}");
      try {
        res = generate(S, g, [&](const std::string& d) {
          if(model_io_logging&&!d.empty())event("model_output",id,"{\"text\":"+jstr(d)+",\"origin\":\"api\"}");
          return es.event(head + "\"choices\": [" + choice(d, nullptr, false) + "]}");
        }, [&](const std::string& progress){
          if(!es.event(head + "\"choices\": [], \"prompt_progress\":" + progress + "}")) throw std::runtime_error("Client disconnected");
        }, [&](const std::string& metrics){
          if(!es.event(head + "\"choices\": [], \"suc_metrics\":" + metrics + "}")) throw std::runtime_error("Client disconnected");
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
        es.event(head + "\"choices\": [], \"usage\": " + usage_json(g.prompt.size(), res.completion_tokens, res.cached_tokens) + ", \"suc_metrics\": " + timing_json(res) + "}");
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
      res = generate(S, g, [&](const std::string& d) {if(model_io_logging&&!d.empty())event("model_output",id,"{\"text\":"+jstr(d)+",\"origin\":\"api\"}");return true;});
    } catch (const std::exception& e) {
      error = e.what();
    }
  }
  if (res.aborted || studio_cancelled()) return send_response(fd, 503, error_json("Request cancelled", "server_error"), false) && false;
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
  body += "], \"usage\": " + usage_json(g.prompt.size(), res.completion_tokens, res.cached_tokens) + ", \"suc_metrics\": " + timing_json(res) + "}";
  return send_response(fd, 200, body, rq.keep_alive) && rq.keep_alive;
}

static void serve_connection(Server& S, int fd) {
  std::string buf;
  HttpRequest rq;
  while (read_request(fd, buf, rq)) {
    rq.keep_alive = false;
    bool keep = false;
    try {
      if (rq.path == "/health" || rq.path == "/v1/health") {
        keep = send_response(fd, 200, "{\"status\": \"ok\"}", keep) && keep;
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


struct LoadedEngine {
  Server S;
  std::unique_ptr<Suc1File> file;
  std::unique_ptr<Tokenizer> tokenizer;
  std::unique_ptr<Context> context;
  std::unique_ptr<ExpertStream> xs;
  std::unique_ptr<MappedFile> mapping;
  std::unique_ptr<Engine> engine;
  std::string family;
  double load_s = 0;
  LoadedEngine(const std::string& path, const RuntimeFlags& rf, const std::string& memory_mode = "auto", const std::string& request_id = "") {
    S.model_name = path.substr(path.find_last_of('/') + 1);
    S.max_ctx = rf.max_ctx; S.think = false;
    const auto t0 = std::chrono::steady_clock::now();
    OpenOptions oo;
    oo.verify_checksum = rf.verify;
    event("load_progress",request_id,"{\"phase\":"+jstr("Verifica del file modello")+"}");
    file = std::make_unique<Suc1File>(Suc1File::open(path, oo));
    auto& f = *file;
    S.arch = f.directory()["architecture"];
    const std::string tp = rf.tokenizer_path.empty() ? resolve_tokenizer_path(f.tokenizer_id()) : rf.tokenizer_path;
    event("load_progress",request_id,"{\"phase\":"+jstr("Verifica del tokenizer")+"}");
    verify_tokenizer(f.tokenizer_id(), tp);
    tokenizer = std::make_unique<Tokenizer>(Tokenizer::from_file(tp));
    auto& tok = *tokenizer;
    S.tok = &tok;
    Tuning tu;
    tu.tuning_cache = rf.tuning_cache;
    event("load_progress",request_id,"{\"phase\":"+jstr("Preparazione Metal")+"}");
    context = std::make_unique<Context>(tu);
    auto& ctx = *context;
    const bool is_q4 = Qwen4Config::is_qwen4(S.arch), is_glm = Glm5Config::is_glm5(S.arch),
               is_ds4 = Ds4Config::is_ds4(S.arch);
    std::vector<std::pair<uint64_t, uint64_t>> exclude;
    if (is_q4) exclude = qwen4_table_ranges(f);

    // Evaluate before creating GPU file views or expert pools. Only full pages
    // entirely occupied by routed ranges are subtracted (boundary pages remain).
    struct stat st{};
    if (::stat(path.c_str(), &st) || st.st_size <= 0) throw std::runtime_error("Dimensione modello non disponibile");
    uint64_t physical=0; size_t physical_size=sizeof physical;
    if (sysctlbyname("hw.memsize", &physical, &physical_size, nullptr, 0)) throw std::runtime_error("Memoria fisica non disponibile");
    const uint64_t page=uint64_t(::getpagesize());
    uint64_t removable=0;
    for (const auto& [begin,end] : routed_expert_ranges(f)) {
      auto lo=(begin+page-1)/page*page, hi=end/page*page;
      if(hi>lo) removable+=hi-lo;
    }
    // Reuse the runtime's architecture parsers, including nested text_config.
    // Treat hybrid models as full attention for a deliberately cautious estimate.
    double kv=0;
    if(is_q4) {
      auto c=Qwen4Config::from_architecture(S.arch);
      kv=2.0*double(c.layers)*double(c.kv_heads)*double(c.head_dim)*double(rf.max_ctx)*4.0;
    } else if(is_glm) {
      // MLA caches latents (kv_lora + rope) and indexer keys, not per-head keys / values; ×2 and float32 stay
      // cautious (heads × max(nope, v) per layer overestimated the cache about 20× and alone forced SSD streaming)
      auto c=Glm5Config::from_architecture(S.arch);
      kv=2.0*double(c.layers)*double(c.kv_lora+64+c.index_dim)*double(rf.max_ctx)*4.0
        +2.0*double(c.heads)*double(rf.max_ctx)*4.0;
    } else if(is_ds4) {
      auto c=Ds4Config::from_architecture(S.arch);
      kv=2.0*double(c.layers)*double(c.head_dim)*double(rf.max_ctx)*4.0;
    } else {
      auto c=ModelConfig::from_architecture(S.arch);
      kv=2.0*double(c.layers)*double(c.kv_heads)*double(c.head_dim)*double(rf.max_ctx)*4.0;
    }
    // The model file already accounts for weights. Reserve the architecture-specific KV estimate plus 1 GiB for
    // transient work buffers; a percentage of the file double-counted mapped weights and capped useful context.
    double reserve_d=kv+1073741824.0;
    if(!std::isfinite(reserve_d)||reserve_d>double(UINT64_MAX/2)) throw std::runtime_error("Stima memoria fuori intervallo");
    auto plan=studio_memory_plan(physical,ctx.device()->recommendedMaxWorkingSetSize(),
      uint64_t(st.st_size),removable,uint64_t(reserve_d),memory_mode,rf.xo.budget_bytes);
    event("memory_plan", request_id, "{\"mode\":"+jstr(plan.streamed?"expert_ssd":"resident")+
      ",\"limit_bytes\":"+std::to_string(plan.limit)+",\"resident_bytes\":"+std::to_string(plan.resident)+
      ",\"reserve_bytes\":"+std::to_string(plan.reserve)+",\"expert_budget_bytes\":"+std::to_string(plan.expert_budget)+"}");
    ExpertStreamOptions xo = rf.xo;
    xo.budget_bytes=plan.expert_budget;
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
    event("load_progress",request_id,"{\"phase\":"+jstr("Mappatura dei pesi e allocazione memoria")+"}");
    mapping = std::make_unique<MappedFile>(ctx, f, exclude, rf.max_view_bytes);
    auto& mf = *mapping;
    auto act_of = [&](const std::string& torch_dtype) {
      if (!rf.dtype.empty()) return parse_act(rf.dtype);
      return torch_dtype == "float16" ? Act::F16 : torch_dtype == "float32" ? Act::F32 : Act::BF16;
    };



    event("load_progress",request_id,"{\"phase\":"+jstr("Preparazione del grafo del modello")+"}");
    if (is_q4) {
      const Qwen4Config qc = Qwen4Config::from_architecture(S.arch);
      Qwen4Options qo;
      qo.act = act_of(qc.torch_dtype);
      qo.max_context = rf.max_ctx;
      qo.layer_prefill = rf.batch_prefill;   // batched layer-major prefill; exact mode selects decode-step arithmetic
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
      dopt.layer_prefill = rf.batch_prefill;   // batched layer-major prefill; exact mode selects decode-step arithmetic
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
    // Actual file view + pool check before warm-up submits work to Metal.
    const auto weights=mapping->view_bytes()+(xs?xs->pool_bytes():0);
    if(plan.reserve>=plan.limit || weights>plan.limit-plan.reserve)
      throw std::runtime_error("Memoria pesi effettiva oltre il limite prudenziale; caricamento annullato prima del warm-up");
    S.expert_stream = xs.get();
    event("load_progress",request_id,"{\"phase\":"+jstr("Preparazione del motore")+"}");
    engine = std::make_unique<suc::LivePrefixEngine<Engine>>(std::move(engine));
    S.engine = engine.get();
    event("load_progress",request_id,"{\"phase\":"+jstr("Compilazione e riscaldamento Metal")+"}");
    S.engine->step(0);   // warm-up: compile the decode pipelines, then reset
    S.engine->reset();
    if (xs) xs->reset_stats();
    load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  }
};
