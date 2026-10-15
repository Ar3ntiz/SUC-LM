#include "suc/json.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace suc {

class JsonParser {
 public:
  explicit JsonParser(std::string_view t) : t_(t) {}

  Json run() {
    Json v = value(0);
    ws();
    if (p_ != t_.size()) fail("trailing characters");
    return v;
  }

 private:
  static constexpr int kMaxDepth = 256;
  std::string_view t_;
  std::size_t p_ = 0;

  [[noreturn]] void fail(const char* what) const {
    throw JsonError(std::string("JSON parse error at byte ") + std::to_string(p_) + ": " + what);
  }
  void ws() {
    while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\t' || t_[p_] == '\n' || t_[p_] == '\r')) ++p_;
  }
  char peek() {
    ws();
    if (p_ >= t_.size()) fail("unexpected end of input");
    return t_[p_];
  }
  void expect(std::string_view lit) {
    if (t_.substr(p_, lit.size()) != lit) fail("invalid literal");
    p_ += lit.size();
  }

  Json value(int depth) {
    if (depth > kMaxDepth) fail("nesting too deep");
    Json v;
    switch (peek()) {
      case '{': {
        ++p_;
        v.type_ = Json::Type::Object;
        if (peek() == '}') { ++p_; return v; }
        for (;;) {
          if (peek() != '"') fail("expected object key");
          std::string k = str();
          if (peek() != ':') fail("expected ':'");
          ++p_;
          v.o_.emplace_back(std::move(k), value(depth + 1));
          char c = peek();
          ++p_;
          if (c == '}') break;
          if (c != ',') fail("expected ',' or '}'");
        }
        return v;
      }
      case '[': {
        ++p_;
        v.type_ = Json::Type::Array;
        if (peek() == ']') { ++p_; return v; }
        for (;;) {
          v.a_.push_back(value(depth + 1));
          char c = peek();
          ++p_;
          if (c == ']') break;
          if (c != ',') fail("expected ',' or ']'");
        }
        return v;
      }
      case '"':
        v.type_ = Json::Type::String;
        v.s_ = str();
        return v;
      case 't': expect("true"); v.type_ = Json::Type::Bool; v.b_ = true; return v;
      case 'f': expect("false"); v.type_ = Json::Type::Bool; v.b_ = false; return v;
      case 'n': expect("null"); return v;
      default: return number();
    }
  }

  static void put_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out += char(cp);
    } else if (cp < 0x800) {
      out += char(0xC0 | (cp >> 6));
      out += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += char(0xE0 | (cp >> 12));
      out += char(0x80 | ((cp >> 6) & 0x3F));
      out += char(0x80 | (cp & 0x3F));
    } else {
      out += char(0xF0 | (cp >> 18));
      out += char(0x80 | ((cp >> 12) & 0x3F));
      out += char(0x80 | ((cp >> 6) & 0x3F));
      out += char(0x80 | (cp & 0x3F));
    }
  }

  uint32_t hex4() {
    if (p_ + 4 > t_.size()) fail("truncated \\u escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = t_[p_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
      else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
      else fail("bad hex digit in \\u escape");
    }
    return v;
  }

  std::string str() {
    ++p_;  // opening quote
    std::string out;
    for (;;) {
      if (p_ >= t_.size()) fail("unterminated string");
      char c = t_[p_++];
      if (c == '"') return out;
      if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
      if (c != '\\') { out += c; continue; }
      if (p_ >= t_.size()) fail("unterminated escape");
      char e = t_[p_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00) {   // surrogate pair
            if (t_.substr(p_, 2) != "\\u") fail("lone high surrogate");
            p_ += 2;
            uint32_t lo = hex4();
            if (lo < 0xDC00 || lo >= 0xE000) fail("bad low surrogate");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp < 0xE000) {
            fail("lone low surrogate");
          }
          put_utf8(out, cp);
          break;
        }
        default: fail("unknown escape");
      }
    }
  }

  Json number() {
    std::size_t s = p_;
    bool integer = true;
    if (p_ < t_.size() && t_[p_] == '-') ++p_;
    auto digits = [&] {
      std::size_t d = p_;
      while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
      if (p_ == d) fail("expected digit");
    };
    if (p_ < t_.size() && t_[p_] == '0') ++p_;
    else digits();
    if (p_ < t_.size() && t_[p_] == '.') { ++p_; digits(); integer = false; }
    if (p_ < t_.size() && (t_[p_] == 'e' || t_[p_] == 'E')) {
      ++p_;
      if (p_ < t_.size() && (t_[p_] == '+' || t_[p_] == '-')) ++p_;
      digits();
      integer = false;
    }
    std::string tok(t_.substr(s, p_ - s));
    Json v;
    if (integer) {
      errno = 0;
      char* end = nullptr;
      long long x = std::strtoll(tok.c_str(), &end, 10);
      if (errno == 0 && end == tok.c_str() + tok.size()) {
        v.type_ = Json::Type::Int;
        v.i_ = x;
        v.d_ = double(x);
        return v;
      }
    }
    v.type_ = Json::Type::Double;
    v.d_ = std::strtod(tok.c_str(), nullptr);
    return v;
  }
};

Json Json::parse(std::string_view text) { return JsonParser(text).run(); }

static const char* type_name(Json::Type t) {
  switch (t) {
    case Json::Type::Null: return "null";
    case Json::Type::Bool: return "bool";
    case Json::Type::Int: return "int";
    case Json::Type::Double: return "double";
    case Json::Type::String: return "string";
    case Json::Type::Array: return "array";
    case Json::Type::Object: return "object";
  }
  return "?";
}

[[noreturn]] static void type_fail(Json::Type got, const char* want) {
  throw JsonError(std::string("JSON type error: expected ") + want + ", got " + type_name(got));
}

bool Json::as_bool() const {
  if (type_ != Type::Bool) type_fail(type_, "bool");
  return b_;
}
int64_t Json::as_int() const {
  if (type_ == Type::Int) return i_;
  if (type_ == Type::Double && std::floor(d_) == d_ && std::fabs(d_) < 9.2e18) return int64_t(d_);
  type_fail(type_, "integer");
}
uint64_t Json::as_uint() const {
  int64_t v = as_int();
  if (v < 0) throw JsonError("JSON value error: expected a non-negative integer");
  return uint64_t(v);
}
double Json::as_double() const {
  if (!is_number()) type_fail(type_, "number");
  return d_;
}
const std::string& Json::as_string() const {
  if (type_ != Type::String) type_fail(type_, "string");
  return s_;
}
const Json::Array& Json::as_array() const {
  if (type_ != Type::Array) type_fail(type_, "array");
  return a_;
}
const Json::Object& Json::as_object() const {
  if (type_ != Type::Object) type_fail(type_, "object");
  return o_;
}
const Json* Json::find(std::string_view key) const {
  for (const auto& kv : as_object())
    if (kv.first == key) return &kv.second;
  return nullptr;
}
const Json& Json::operator[](std::string_view key) const {
  if (const Json* v = find(key)) return *v;
  throw JsonError("JSON key error: missing key '" + std::string(key) + "'");
}
std::size_t Json::size() const {
  if (type_ == Type::Array) return a_.size();
  if (type_ == Type::Object) return o_.size();
  type_fail(type_, "array or object");
}

static void dump_string(std::string& out, const std::string& s) {
  out += '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
    else if (c < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof buf, "\\u%04x", c);
      out += buf;
    } else out += char(c);
  }
  out += '"';
}

static void dump_into(std::string& out, const Json& v) {
  switch (v.type()) {
    case Json::Type::Null: out += "null"; break;
    case Json::Type::Bool: out += v.as_bool() ? "true" : "false"; break;
    case Json::Type::Int: out += std::to_string(v.as_int()); break;
    case Json::Type::Double: {
      char buf[32];
      std::snprintf(buf, sizeof buf, "%.17g", v.as_double());
      out += buf;
      break;
    }
    case Json::Type::String: dump_string(out, v.as_string()); break;
    case Json::Type::Array: {
      out += '[';
      bool first = true;
      for (const auto& x : v.as_array()) {
        if (!first) out += ',';
        first = false;
        dump_into(out, x);
      }
      out += ']';
      break;
    }
    case Json::Type::Object: {
      out += '{';
      bool first = true;
      for (const auto& kv : v.as_object()) {
        if (!first) out += ',';
        first = false;
        dump_string(out, kv.first);
        out += ':';
        dump_into(out, kv.second);
      }
      out += '}';
      break;
    }
  }
}

std::string Json::dump() const {
  std::string out;
  dump_into(out, *this);
  return out;
}

}  // namespace suc
