// Byte-level BPE tokenizer (see suc/tokenizer.hpp for the supported tokenizer.json subset).
#include "suc/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <unordered_map>

#include "suc/json.hpp"
#include "suc/sha256.hpp"

namespace suc {

namespace {

#include "unicode_tables.inc"

[[noreturn]] void fail(const std::string& what) { throw TokenizerError("tokenizer: " + what); }

// ---------------------------------------------------------------------------------------------------------------
// UTF-8

void put_utf8(std::string& out, uint32_t cp) {
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

// Length of the valid UTF-8 sequence starting at s[i], or 0 if s[i] does not start one. For an invalid sequence,
// *maximal is set to the length of its maximal valid prefix (>= 1), the unit Rust's from_utf8_lossy replaces by
// one U+FFFD (Unicode "substitution of maximal subparts").
std::size_t utf8_seq(std::string_view s, std::size_t i, uint32_t* cp, std::size_t* maximal) {
  const auto b = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
  unsigned char c = b(i);
  if (c < 0x80) { *cp = c; return 1; }
  std::size_t need;
  unsigned char lo = 0x80, hi = 0xBF;
  if (c >= 0xC2 && c <= 0xDF) need = 1;
  else if (c >= 0xE0 && c <= 0xEF) {
    need = 2;
    if (c == 0xE0) lo = 0xA0;
    if (c == 0xED) hi = 0x9F;
  } else if (c >= 0xF0 && c <= 0xF4) {
    need = 3;
    if (c == 0xF0) lo = 0x90;
    if (c == 0xF4) hi = 0x8F;
  } else {
    *maximal = 1;
    return 0;
  }
  uint32_t v = c & (need == 1 ? 0x1F : need == 2 ? 0x0F : 0x07);
  for (std::size_t k = 1; k <= need; ++k) {
    if (i + k >= s.size()) { *maximal = k; return 0; }
    unsigned char x = b(i + k);
    unsigned char l = k == 1 ? lo : 0x80, h = k == 1 ? hi : 0xBF;
    if (x < l || x > h) { *maximal = k; return 0; }
    v = (v << 6) | (x & 0x3F);
  }
  *cp = v;
  return need + 1;
}

// Strict decode: code points plus the byte offset of each (offs has one extra entry = s.size()).
void decode_utf8(std::string_view s, std::vector<uint32_t>& cps, std::vector<uint32_t>& offs) {
  cps.clear();
  offs.clear();
  for (std::size_t i = 0; i < s.size();) {
    uint32_t cp;
    std::size_t m = 0, n = utf8_seq(s, i, &cp, &m);
    if (!n) fail("input is not valid UTF-8 (byte offset " + std::to_string(i) + ")");
    cps.push_back(cp);
    offs.push_back(uint32_t(i));
    i += n;
  }
  offs.push_back(uint32_t(s.size()));
}

// ---------------------------------------------------------------------------------------------------------------
// Unicode classes (\p{L} = 1, \p{N} = 2, \s = 4, \p{M} = 8) and NFC

uint8_t class_slow(uint32_t cp) {
  std::size_t lo = 0, hi = std::size(kCatRanges);
  while (lo < hi) {
    std::size_t mid = (lo + hi) / 2;
    if (kCatRanges[mid].last < cp) lo = mid + 1;
    else hi = mid;
  }
  return lo < std::size(kCatRanges) && kCatRanges[lo].first <= cp ? kCatRanges[lo].cls : 0;
}

struct BmpClasses {
  std::vector<uint8_t> t;
  BmpClasses() : t(0x10000) {
    for (uint32_t c = 0; c < 0x10000; ++c) t[c] = class_slow(c);
  }
};

inline uint8_t cls(uint32_t cp) {
  static const BmpClasses bmp;
  return cp < 0x10000 ? bmp.t[cp] : class_slow(cp);
}
inline bool is_L(uint32_t c) { return cls(c) & 1; }
inline bool is_N(uint32_t c) { return cls(c) & 2; }
inline bool is_S(uint32_t c) { return cls(c) & 4; }
inline bool is_LM(uint32_t c) { return cls(c) & 9; }
inline bool is_PS(uint32_t c) { return cls(c) & 48; }   // \p{P} or \p{S}

uint8_t ccc(uint32_t cp) {
  if (cp < 0x300) return 0;
  auto it = std::lower_bound(std::begin(kCcc), std::end(kCcc), cp, [](const CccEntry& e, uint32_t c) { return e.cp < c; });
  return it != std::end(kCcc) && it->cp == cp ? it->ccc : 0;
}

constexpr uint32_t SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7;
constexpr uint32_t LCount = 19, VCount = 21, TCount = 28, NCount = VCount * TCount, SCount = LCount * NCount;

void decompose(uint32_t cp, std::vector<uint32_t>& out) {
  if (cp >= SBase && cp < SBase + SCount) {
    uint32_t s = cp - SBase;
    out.push_back(LBase + s / NCount);
    out.push_back(VBase + (s % NCount) / TCount);
    if (s % TCount) out.push_back(TBase + s % TCount);
    return;
  }
  if (cp >= 0xC0) {
    auto it = std::lower_bound(std::begin(kDecomp), std::end(kDecomp), cp,
                               [](const DecompEntry& e, uint32_t c) { return e.cp < c; });
    if (it != std::end(kDecomp) && it->cp == cp) {
      out.insert(out.end(), kDecompData + it->offset, kDecompData + it->offset + it->len);
      return;
    }
  }
  out.push_back(cp);
}

// Primary composite of (a, b), or 0.
uint32_t compose(uint32_t a, uint32_t b) {
  if (a >= LBase && a < LBase + LCount && b >= VBase && b < VBase + VCount)
    return SBase + ((a - LBase) * VCount + (b - VBase)) * TCount;
  if (a >= SBase && a < SBase + SCount && (a - SBase) % TCount == 0 && b > TBase && b < TBase + TCount)
    return a + (b - TBase);
  auto it = std::lower_bound(std::begin(kCompose), std::end(kCompose), std::make_pair(a, b),
                             [](const ComposeEntry& e, const std::pair<uint32_t, uint32_t>& k) {
                               return e.first != k.first ? e.first < k.first : e.second < k.second;
                             });
  return it != std::end(kCompose) && it->first == a && it->second == b ? it->composite : 0;
}

std::string nfc_impl(std::string_view s) {
  // Fast path: code points below U+0300 never decompose, never combine and have ccc 0, and their UTF-8 bytes are
  // all < 0xCC (U+0300 is CC 80).
  bool simple = true;
  for (unsigned char c : s)
    if (c >= 0xCC) { simple = false; break; }
  if (simple) return std::string(s);
  std::vector<uint32_t> cps, offs, d;
  decode_utf8(s, cps, offs);
  d.reserve(cps.size() + 8);
  for (uint32_t c : cps) decompose(c, d);
  // Canonical ordering: stable sort of every run of non-starters by combining class.
  for (std::size_t i = 0; i < d.size();) {
    if (!ccc(d[i])) { ++i; continue; }
    std::size_t j = i;
    while (j < d.size() && ccc(d[j])) ++j;
    std::stable_sort(d.begin() + i, d.begin() + j, [](uint32_t a, uint32_t b) { return ccc(a) < ccc(b); });
    i = j;
  }
  // Canonical composition.
  std::vector<uint32_t> r;
  r.reserve(d.size());
  long starter = -1;
  int last_cc = 0;
  for (uint32_t c : d) {
    int cc = ccc(c);
    if (starter >= 0) {
      bool adjacent = long(r.size()) - 1 == starter;
      if (adjacent || (last_cc != 0 && last_cc < cc)) {
        if (uint32_t comp = compose(r[starter], c)) {
          r[starter] = comp;
          continue;
        }
      }
    }
    if (cc == 0) starter = long(r.size());
    last_cc = cc;
    r.push_back(c);
  }
  std::string out;
  out.reserve(s.size());
  for (uint32_t c : r) put_utf8(out, c);
  return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Qwen2/Qwen3 Split regex, as a hand-written matcher with Oniguruma's leftmost-alternative semantics:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// and the variants of the same family, matched by two parameters:
//   marks   letter runs also take combining marks (\p{M}) -- Qwen3.5 / Qwen3.8 (qwen4_exp):
//           ...|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|...
//   digits  digit groups of up to `digits` (\p{N}{1,d}, greedy) instead of single digits -- GLM-5.x (d = 3)
//           ...|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|...
// The pattern of a tokenizer.json is accepted only if it is exactly one of these strings (qwen_regex).

struct QwenRx {
  bool marks = false;
  unsigned digits = 1;
};

std::string qwen_regex(QwenRx r) {
  const std::string num = r.digits == 1 ? "\\p{N}" : "\\p{N}{1," + std::to_string(r.digits) + "}";
  return std::string("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?") +
         (r.marks ? "[\\p{L}\\p{M}]+|" : "\\p{L}+|") + num +
         (r.marks ? "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|" : "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|") +
         "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
}

// The variant whose pattern is `rx`, if any (digit groups of 1..9).
std::optional<QwenRx> parse_qwen_regex(const std::string& rx) {
  for (bool marks : {false, true})
    for (unsigned d = 1; d <= 9; ++d)
      if (rx == qwen_regex({marks, d})) return QwenRx{marks, d};
  return std::nullopt;
}

inline bool is_crlf(uint32_t c) { return c == '\r' || c == '\n'; }
inline uint32_t lower_ascii(uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

// End (exclusive) of the match starting at i, or i when no alternative matches there. marks: the Qwen3.5 variant.
std::size_t qwen_match(const uint32_t* c, std::size_t n, std::size_t i, QwenRx rx) {
  const bool marks = rx.marks;
  // 1. contractions (ASCII case-insensitive: Oniguruma does not fold e.g. U+017F to 's' here, checked in tests)
  if (c[i] == '\'' && i + 1 < n) {
    uint32_t x = lower_ascii(c[i + 1]);
    if (x == 's' || x == 't' || x == 'm' || x == 'd') return i + 2;
    if (i + 2 < n) {
      uint32_t y = lower_ascii(c[i + 2]);
      if ((x == 'r' && y == 'e') || (x == 'v' && y == 'e') || (x == 'l' && y == 'l')) return i + 3;
    }
  }
  // 2. [^\r\n\p{L}\p{N}]?\p{L}+  (marks: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+; the prefix may itself be a mark, so
  //    without a run after it the alternative is retried without the prefix)
  {
    auto run = [&](uint32_t x) { return marks ? is_LM(x) : is_L(x); };
    std::size_t j = i;
    if (!is_L(c[j]) && !is_crlf(c[j]) && !is_N(c[j]) && j + 1 < n && run(c[j + 1])) ++j;
    if (run(c[j])) {
      while (j < n && run(c[j])) ++j;
      return j;
    }
  }
  // 3. \p{N}{1,digits} (greedy)
  if (is_N(c[i])) {
    std::size_t j = i + 1;
    while (j < n && j - i < rx.digits && is_N(c[j])) ++j;
    return j;
  }
  // 4. ' ?[^\s\p{L}\p{N}]+[\r\n]*'
  {
    std::size_t j = i + (c[i] == ' ' ? 1 : 0), k = j;
    const uint8_t excl = marks ? 15 : 7;
    while (k < n && !(cls(c[k]) & excl)) ++k;
    if (k > j) {
      while (k < n && is_crlf(c[k])) ++k;
      return k;
    }
  }
  if (!is_S(c[i])) return i;
  std::size_t k = i;
  while (k < n && is_S(c[k])) ++k;   // whitespace run [i, k)
  // 5. \s*[\r\n]+ : ends after the last CR/LF of the run
  for (std::size_t p = k; p > i; --p)
    if (is_crlf(c[p - 1])) return p;
  // 6. \s+(?!\S)
  if (k == n) return k;
  if (k - 1 > i) return k - 1;
  // 7. \s+
  return k;
}

// Byte ranges of the pre-tokenizer pieces (Split behavior "Isolated": matches and gaps both become pieces).
void qwen_split(std::string_view s, std::vector<std::pair<uint32_t, uint32_t>>& pieces, std::vector<uint32_t>& cps,
                std::vector<uint32_t>& offs, QwenRx rx = {}) {
  pieces.clear();
  decode_utf8(s, cps, offs);
  const std::size_t n = cps.size();
  std::size_t gap = 0;
  for (std::size_t i = 0; i < n;) {
    std::size_t e = qwen_match(cps.data(), n, i, rx);
    if (e == i) { ++i; continue; }
    if (gap < i) pieces.emplace_back(offs[gap], offs[i]);
    pieces.emplace_back(offs[i], offs[e]);
    i = gap = e;
  }
  if (gap < n) pieces.emplace_back(offs[gap], offs[n]);
}

// ---------------------------------------------------------------------------------------------------------------
// DeepSeek-V3 / V4 pre-tokenizer: Sequence[Split(\p{N}{1,3}), Split(CJK run), Split(R3)], each Isolated, applied to
// the pieces of the previous one (a regex runs on one piece: its end is the end of the string), with
//   R3 = [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|
//        ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+

const char* const kDsRegex1 = "\\p{N}{1,3}";
const char* const kDsRegex2 = "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+";
const char* const kDsRegex3 =
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| "
    "?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

inline bool is_ascii_punct(uint32_t c) {
  return (c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) || (c >= 0x5B && c <= 0x60) || (c >= 0x7B && c <= 0x7E);
}
inline bool is_ascii_alpha(uint32_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
inline bool is_ds_cjk(uint32_t c) {
  return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF);
}

std::size_t ds_match_num(const uint32_t* c, std::size_t n, std::size_t i) {
  std::size_t j = i;
  while (j < n && j - i < 3 && is_N(c[j])) ++j;
  return j;
}
std::size_t ds_match_cjk(const uint32_t* c, std::size_t n, std::size_t i) {
  std::size_t j = i;
  while (j < n && is_ds_cjk(c[j])) ++j;
  return j;
}
std::size_t ds_match_main(const uint32_t* c, std::size_t n, std::size_t i) {
  // 1. ASCII punctuation + ASCII letters
  if (is_ascii_punct(c[i]) && i + 1 < n && is_ascii_alpha(c[i + 1])) {
    std::size_t j = i + 1;
    while (j < n && is_ascii_alpha(c[j])) ++j;
    return j;
  }
  // 2. [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ (without a run after the prefix, retried without it)
  {
    std::size_t j = i;
    if (!is_crlf(c[j]) && !is_L(c[j]) && !is_PS(c[j]) && j + 1 < n && is_LM(c[j + 1])) ++j;
    if (is_LM(c[j])) {
      while (j < n && is_LM(c[j])) ++j;
      return j;
    }
  }
  // 3. ' ?[\p{P}\p{S}]+[\r\n]*'
  {
    std::size_t j = i + (c[i] == ' ' ? 1 : 0), k = j;
    while (k < n && is_PS(c[k])) ++k;
    if (k > j) {
      while (k < n && is_crlf(c[k])) ++k;
      return k;
    }
  }
  if (!is_S(c[i])) return i;
  std::size_t k = i;
  while (k < n && is_S(c[k])) ++k;   // whitespace run [i, k)
  // 4. \s*[\r\n]+
  for (std::size_t p = k; p > i; --p)
    if (is_crlf(c[p - 1])) return p;
  // 5. \s+(?!\S)
  if (k == n) return k;
  if (k - 1 > i) return k - 1;
  // 6. \s+
  return k;
}

// Isolated split of the code points [lo, hi) by `match` (end of the match at i, or i): matches and gaps as pieces.
template <class M>
void ds_split_range(const uint32_t* c, std::size_t lo, std::size_t hi, M match,
                    std::vector<std::pair<uint32_t, uint32_t>>& out) {
  std::size_t gap = lo;
  for (std::size_t i = lo; i < hi;) {
    const std::size_t e = lo + match(c + lo, hi - lo, i - lo);
    if (e == i) { ++i; continue; }
    if (gap < i) out.emplace_back(uint32_t(gap), uint32_t(i));
    out.emplace_back(uint32_t(i), uint32_t(e));
    i = gap = e;
  }
  if (gap < hi) out.emplace_back(uint32_t(gap), uint32_t(hi));
}

void ds_split(std::string_view s, std::vector<std::pair<uint32_t, uint32_t>>& pieces, std::vector<uint32_t>& cps,
              std::vector<uint32_t>& offs) {
  pieces.clear();
  decode_utf8(s, cps, offs);
  std::vector<std::pair<uint32_t, uint32_t>> a, b;
  ds_split_range(cps.data(), 0, cps.size(), ds_match_num, a);
  for (auto [lo, hi] : a) ds_split_range(cps.data(), lo, hi, ds_match_cjk, b);
  a.clear();
  for (auto [lo, hi] : b) ds_split_range(cps.data(), lo, hi, ds_match_main, a);
  for (auto [lo, hi] : a) pieces.emplace_back(offs[lo], offs[hi]);
}

// ---------------------------------------------------------------------------------------------------------------
// GPT-2 bytes_to_unicode

std::array<uint32_t, 256> make_byte_map() {
  std::array<uint32_t, 256> m{};
  std::array<bool, 256> direct{};
  for (int b = '!'; b <= '~'; ++b) direct[b] = true;
  for (int b = 0xA1; b <= 0xAC; ++b) direct[b] = true;
  for (int b = 0xAE; b <= 0xFF; ++b) direct[b] = true;
  uint32_t next = 256;
  for (int b = 0; b < 256; ++b) m[b] = direct[b] ? uint32_t(b) : next++;
  return m;
}
const std::array<uint32_t, 256> kByteMap = make_byte_map();

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fail("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string sha256_hex_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fail("cannot open " + path);
  Sha256 h;
  std::vector<char> buf(1 << 20);
  while (f) {
    f.read(buf.data(), std::streamsize(buf.size()));
    h.update(buf.data(), std::size_t(f.gcount()));
  }
  auto d = h.finish();
  return to_hex(d.data(), d.size());
}

// Literal matcher for added tokens: a byte trie giving the longest token starting at a position.
struct Trie {
  struct Node {
    std::unordered_map<unsigned char, uint32_t> next;
    int64_t token = -1;   // index into Impl::added
  };
  std::vector<Node> nodes{1};
  bool empty() const { return nodes.size() == 1; }
  void add(std::string_view s, std::size_t token) {
    uint32_t cur = 0;
    for (unsigned char c : s) {
      auto it = nodes[cur].next.find(c);
      if (it == nodes[cur].next.end()) {
        nodes.emplace_back();
        it = nodes[cur].next.emplace(c, uint32_t(nodes.size() - 1)).first;
      }
      cur = it->second;
    }
    nodes[cur].token = int64_t(token);
  }
  // (token index, length) of the longest match at s[i], or (-1, 0).
  std::pair<int64_t, std::size_t> longest(std::string_view s, std::size_t i) const {
    uint32_t cur = 0;
    std::pair<int64_t, std::size_t> best{-1, 0};
    for (std::size_t k = i; k < s.size(); ++k) {
      auto it = nodes[cur].next.find(static_cast<unsigned char>(s[k]));
      if (it == nodes[cur].next.end()) break;
      cur = it->second;
      if (nodes[cur].token >= 0) best = {nodes[cur].token, k + 1 - i};
    }
    return best;
  }
};

}  // namespace

// ---------------------------------------------------------------------------------------------------------------

struct Tokenizer::Impl {
  bool nfc = false, split_regex = false, ignore_merges = false, post_bytelevel = false;
  QwenRx split_rx;   // the Qwen-family Split variant
  bool split_ds = false;   // the DeepSeek-V3 / V4 three-Split sequence
  std::unordered_map<std::string, uint32_t> vocab;      // model vocab: byte-level string -> id
  std::vector<std::string> id_token;                    // id -> token string ("" when unused)
  std::vector<std::string> id_bytes;                    // id -> decoded bytes (model tokens)
  std::vector<int32_t> id_added;                        // id -> index into added, or -1
  std::vector<uint8_t> id_used;
  std::array<uint32_t, 256> byte_id{};                  // byte -> id of its byte-level character
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> merges;   // (a, b) -> (rank, new id)
  std::vector<AddedToken> added;
  std::unordered_map<std::string, uint32_t> added_by_content;
  Trie raw_trie, norm_trie;   // added tokens matched on the raw / normalized text
  mutable std::mutex cache_mu;
  mutable std::unordered_map<std::string, std::vector<uint32_t>> cache;
  static constexpr std::size_t kCacheMax = 1 << 18;

  static uint64_t key(uint32_t a, uint32_t b) { return (uint64_t(a) << 32) | b; }

  void bpe(std::string_view word, std::vector<uint32_t>& out) const;
  void encode_plain(std::string_view text, std::vector<uint32_t>& out) const;   // normalized text, no added tokens
  void split_added(std::string_view s, const Trie& trie,
                   std::vector<std::pair<int64_t, std::pair<std::size_t, std::size_t>>>& segs) const;
};

// HF tokenizers' Word::merge_all: a min-heap on (rank, position); an entry is stale when the pair now at its position
// no longer maps to the entry's new id.
void Tokenizer::Impl::bpe(std::string_view word, std::vector<uint32_t>& out) const {
  const std::size_t n = word.size();
  if (n == 1) { out.push_back(byte_id[static_cast<unsigned char>(word[0])]); return; }
  {
    std::lock_guard<std::mutex> lk(cache_mu);
    auto it = cache.find(std::string(word));
    if (it != cache.end()) {
      out.insert(out.end(), it->second.begin(), it->second.end());
      return;
    }
  }
  std::vector<uint32_t> result;
  bool done = false;
  if (ignore_merges) {
    std::string bl;
    for (unsigned char b : word) put_utf8(bl, kByteMap[b]);
    auto it = vocab.find(bl);
    if (it != vocab.end()) { result.push_back(it->second); done = true; }
  }
  if (!done) {
    struct Sym { uint32_t id; int32_t prev, next; bool alive; };
    std::vector<Sym> sym(n);
    for (std::size_t i = 0; i < n; ++i)
      sym[i] = {byte_id[static_cast<unsigned char>(word[i])], int32_t(i) - 1, i + 1 < n ? int32_t(i + 1) : -1, true};
    struct Merge { uint32_t rank, pos, new_id; };
    auto worse = [](const Merge& a, const Merge& b) { return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos; };
    std::priority_queue<Merge, std::vector<Merge>, decltype(worse)> q(worse);
    for (std::size_t i = 0; i + 1 < n; ++i) {
      auto it = merges.find(key(sym[i].id, sym[i + 1].id));
      if (it != merges.end()) q.push({it->second.first, uint32_t(i), it->second.second});
    }
    while (!q.empty()) {
      Merge top = q.top();
      q.pop();
      Sym& cur = sym[top.pos];
      if (!cur.alive || cur.next < 0) continue;
      Sym& right = sym[cur.next];
      auto it = merges.find(key(cur.id, right.id));
      if (it == merges.end() || it->second.second != top.new_id) continue;
      cur.id = top.new_id;
      right.alive = false;
      cur.next = right.next;
      if (cur.next >= 0) sym[cur.next].prev = int32_t(top.pos);
      if (cur.prev >= 0) {
        auto m = merges.find(key(sym[cur.prev].id, cur.id));
        if (m != merges.end()) q.push({m->second.first, uint32_t(cur.prev), m->second.second});
      }
      if (cur.next >= 0) {
        auto m = merges.find(key(cur.id, sym[cur.next].id));
        if (m != merges.end()) q.push({m->second.first, top.pos, m->second.second});
      }
    }
    for (int32_t i = 0; i >= 0; i = sym[i].next) result.push_back(sym[i].id);
  }
  out.insert(out.end(), result.begin(), result.end());
  std::lock_guard<std::mutex> lk(cache_mu);
  if (cache.size() >= kCacheMax) cache.clear();
  cache.emplace(std::string(word), std::move(result));
}

void Tokenizer::Impl::encode_plain(std::string_view text, std::vector<uint32_t>& out) const {
  if (text.empty()) return;
  if (!split_regex) {
    bpe(text, out);
    return;
  }
  thread_local std::vector<std::pair<uint32_t, uint32_t>> pieces;
  thread_local std::vector<uint32_t> cps, offs;
  if (split_ds) ds_split(text, pieces, cps, offs);
  else qwen_split(text, pieces, cps, offs, split_rx);
  for (auto [a, b] : pieces) bpe(text.substr(a, b - a), out);
}

namespace {
std::size_t ws_len_back(std::string_view s) {   // bytes of trailing White_Space
  std::size_t e = s.size();
  while (e > 0) {
    std::size_t b = e - 1;
    while (b > 0 && (static_cast<unsigned char>(s[b]) & 0xC0) == 0x80) --b;
    uint32_t cp;
    std::size_t m;
    if (!utf8_seq(s, b, &cp, &m) || !is_S(cp)) break;
    e = b;
  }
  return s.size() - e;
}
std::size_t ws_len_front(std::string_view s) {
  std::size_t i = 0;
  while (i < s.size()) {
    uint32_t cp;
    std::size_t m, n = utf8_seq(s, i, &cp, &m);
    if (!n || !is_S(cp)) break;
    i += n;
  }
  return i;
}
// Rust char::is_alphanumeric, approximated by \p{L} or \p{N} (only used for single_word tokens).
bool word_char(uint32_t cp) { return cls(cp) & 3; }
bool ends_with_word(std::string_view s) {
  if (s.empty()) return false;
  std::size_t b = s.size() - 1;
  while (b > 0 && (static_cast<unsigned char>(s[b]) & 0xC0) == 0x80) --b;
  uint32_t cp;
  std::size_t m;
  return utf8_seq(s, b, &cp, &m) && word_char(cp);
}
bool starts_with_word(std::string_view s) {
  uint32_t cp;
  std::size_t m;
  return !s.empty() && utf8_seq(s, 0, &cp, &m) && word_char(cp);
}
}  // namespace

// HF AddedVocabulary::find_matches: non-overlapping leftmost-longest literal matches, then single_word / lstrip /
// rstrip adjustments. segs gets (added index or -1, [start, stop)) covering s.
void Tokenizer::Impl::split_added(std::string_view s, const Trie& trie,
                                  std::vector<std::pair<int64_t, std::pair<std::size_t, std::size_t>>>& segs) const {
  segs.clear();
  std::size_t start_offset = 0;
  if (!trie.empty()) {
    for (std::size_t p = 0; p < s.size();) {
      auto [tok, len] = trie.longest(s, p);
      if (tok < 0) { ++p; continue; }
      std::size_t start = p, stop = p + len;
      p = stop;
      const AddedToken& t = added[std::size_t(tok)];
      if (t.single_word) {
        bool start_space = start == 0 || !ends_with_word(s.substr(0, start));
        bool stop_space = stop == s.size() || !starts_with_word(s.substr(stop));
        if (!start_space || !stop_space) continue;
      }
      if (t.lstrip) start = std::max(start - ws_len_back(s.substr(0, start)), start_offset);
      if (t.rstrip) stop += ws_len_front(s.substr(stop));
      if (start_offset < start) segs.push_back({-1, {start_offset, start}});
      segs.push_back({tok, {start, stop}});
      start_offset = stop;
    }
  }
  if (start_offset < s.size()) segs.push_back({-1, {start_offset, s.size()}});
}

Tokenizer::Tokenizer(std::unique_ptr<Impl> p) : p_(std::move(p)) {}
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;
Tokenizer::~Tokenizer() = default;

Tokenizer Tokenizer::from_file(const std::string& path) {
  try {
    return from_json(read_file(path));
  } catch (const TokenizerError& e) {
    throw TokenizerError(std::string(e.what()) + " [" + path + "]");
  } catch (const JsonError& e) {
    throw TokenizerError(std::string("tokenizer: ") + e.what() + " [" + path + "]");
  }
}

namespace {
bool str_or_null_empty(const Json* v) { return !v || v->is_null() || (v->is_string() && v->as_string().empty()); }
bool false_or_absent(const Json* v) { return !v || v->is_null() || (v->is_bool() && !v->as_bool()); }
bool bool_of(const Json& o, const char* k) {
  const Json* v = o.find(k);
  return v && !v->is_null() && v->as_bool();
}
void check_bytelevel(const Json& j, const char* where) {
  if (bool_of(j, "add_prefix_space")) fail(std::string(where) + ": ByteLevel add_prefix_space=true is not supported");
  if (bool_of(j, "use_regex"))
    fail(std::string(where) + ": ByteLevel use_regex=true (GPT-2 regex) is not supported");
}
}  // namespace

Tokenizer Tokenizer::from_json(std::string_view text) {
  Json j = Json::parse(text);
  auto p = std::make_unique<Impl>();
  if (const Json* t = j.find("truncation"); t && !t->is_null()) fail("truncation is not supported");
  if (const Json* t = j.find("padding"); t && !t->is_null()) fail("padding is not supported");

  // normalizer
  if (const Json* n = j.find("normalizer"); n && !n->is_null()) {
    std::vector<const Json*> norms;
    if ((*n)["type"].as_string() == "Sequence") {
      for (const Json& x : (*n)["normalizers"].as_array()) norms.push_back(&x);   // an empty Sequence = none
    } else {
      norms.push_back(n);
    }
    for (const Json* x : norms) {
      const std::string& ty = (*x)["type"].as_string();
      if (ty != "NFC" || p->nfc) fail("normalizer '" + ty + "' is not supported (only NFC or none)");
      p->nfc = true;
    }
  }
  // pre-tokenizer
  const Json& pre = j["pre_tokenizer"];
  if (pre.is_null()) fail("a null pre_tokenizer is not supported (expected ByteLevel)");
  std::vector<const Json*> steps;
  if (pre["type"].as_string() == "Sequence")
    for (const Json& s : pre["pretokenizers"].as_array()) steps.push_back(&s);
  else steps.push_back(&pre);
  bool bytelevel = false;
  // the DeepSeek-V3 / V4 sequence of three Splits
  {
    std::vector<std::string> rxs;
    for (const Json* s : steps)
      if ((*s)["type"].as_string() == "Split")
        if (const Json* rx = (*s)["pattern"].find("Regex")) rxs.push_back(rx->as_string());
    if (rxs == std::vector<std::string>{kDsRegex1, kDsRegex2, kDsRegex3}) {
      std::vector<const Json*> rest;
      for (const Json* s : steps) {
        if ((*s)["type"].as_string() != "Split") {
          rest.push_back(s);
          continue;
        }
        if (bytelevel || (*s)["behavior"].as_string() != "Isolated" || bool_of(*s, "invert"))
          fail("pre_tokenizer: the DeepSeek Splits must be Isolated, not inverted, before ByteLevel");
        (void)bytelevel;
      }
      p->split_regex = p->split_ds = true;
      steps = rest;
    }
  }
  for (const Json* s : steps) {
    const std::string& ty = (*s)["type"].as_string();
    if (ty == "Split") {
      if (bytelevel || p->split_regex) fail("pre_tokenizer: Split must come once, before ByteLevel");
      const Json* rx = (*s)["pattern"].find("Regex");
      const std::optional<QwenRx> v = rx && rx->is_string() ? parse_qwen_regex(rx->as_string()) : std::nullopt;
      if (!v)
        fail("pre_tokenizer: only the Qwen2/Qwen3-family Split regexes (Qwen3, Qwen3.5 marks, GLM-5 digit groups) and "
             "the DeepSeek-V3/V4 sequence are implemented, got " + (*s)["pattern"].dump());
      p->split_rx = *v;
      if ((*s)["behavior"].as_string() != "Isolated") fail("pre_tokenizer: Split behavior must be Isolated");
      if (bool_of(*s, "invert")) fail("pre_tokenizer: Split invert=true is not supported");
      p->split_regex = true;
    } else if (ty == "ByteLevel") {
      if (bytelevel) fail("pre_tokenizer: ByteLevel given twice");
      check_bytelevel(*s, "pre_tokenizer");
      bytelevel = true;
    } else {
      fail("pre_tokenizer '" + ty + "' is not supported");
    }
  }
  if (!bytelevel) fail("pre_tokenizer: ByteLevel is required");
  // post-processor and decoder
  if (const Json* pp = j.find("post_processor"); pp && !pp->is_null()) {
    if ((*pp)["type"].as_string() != "ByteLevel")
      fail("post_processor '" + (*pp)["type"].as_string() + "' is not supported (only ByteLevel or none)");
    p->post_bytelevel = true;
  }
  const Json& dec = j["decoder"];
  if (dec.is_null() || dec["type"].as_string() != "ByteLevel") fail("decoder must be ByteLevel");
  // (add_prefix_space / trim_offsets / use_regex of the ByteLevel decoder do not change the decoded bytes)

  // model
  const Json& m = j["model"];
  if (m["type"].as_string() != "BPE") fail("model '" + m["type"].as_string() + "' is not supported (only BPE)");
  if (const Json* d = m.find("dropout"); d && !d->is_null()) fail("BPE dropout is not supported");
  if (!str_or_null_empty(m.find("unk_token"))) fail("BPE unk_token is not supported");
  if (!str_or_null_empty(m.find("continuing_subword_prefix"))) fail("BPE continuing_subword_prefix is not supported");
  if (!str_or_null_empty(m.find("end_of_word_suffix"))) fail("BPE end_of_word_suffix is not supported");
  if (!false_or_absent(m.find("byte_fallback"))) fail("BPE byte_fallback=true is not supported");
  p->ignore_merges = bool_of(m, "ignore_merges");

  uint32_t max_id = 0;
  const auto& vocab = m["vocab"].as_object();
  p->vocab.reserve(vocab.size() * 2);
  for (const auto& [tok, id] : vocab) {
    uint64_t v = id.as_uint();
    if (v > 0x7FFFFFFF) fail("vocab id out of range");
    p->vocab[tok] = uint32_t(v);
    max_id = std::max(max_id, uint32_t(v));
  }
  const auto& added = j["added_tokens"].as_array();
  for (const Json& a : added) max_id = std::max(max_id, uint32_t(a["id"].as_uint()));
  const std::size_t n = std::size_t(max_id) + 1;
  p->id_token.assign(n, {});
  p->id_bytes.assign(n, {});
  p->id_added.assign(n, -1);
  p->id_used.assign(n, 0);
  std::array<int, 0x200> unbyte;
  unbyte.fill(-1);
  for (int b = 0; b < 256; ++b) unbyte[kByteMap[b]] = b;
  // ByteLevel decoder: map every char back to its byte; a token with any other char decodes to its UTF-8 bytes.
  auto bytelevel_bytes = [&](const std::string& tok) {
    std::string bytes;
    std::vector<uint32_t> cps, offs;
    decode_utf8(tok, cps, offs);
    for (uint32_t c : cps) {
      if (c >= unbyte.size() || unbyte[c] < 0) return tok;
      bytes += char(unbyte[c]);
    }
    return bytes;
  };
  for (const auto& [tok, idj] : p->vocab) {
    uint32_t id = idj;
    p->id_token[id] = tok;
    p->id_used[id] = 1;
    p->id_bytes[id] = bytelevel_bytes(tok);
  }
  for (int b = 0; b < 256; ++b) {
    std::string s;
    put_utf8(s, kByteMap[b]);
    auto it = p->vocab.find(s);
    if (it == p->vocab.end()) fail("vocab lacks the byte-level character for byte " + std::to_string(b));
    p->byte_id[b] = it->second;
  }
  const auto& merges = m["merges"].as_array();
  p->merges.reserve(merges.size() * 2);
  for (std::size_t r = 0; r < merges.size(); ++r) {
    std::string a, b;
    if (merges[r].is_array()) {
      if (merges[r].size() != 2) fail("merge " + std::to_string(r) + " is not a pair");
      a = merges[r].at(0).as_string();
      b = merges[r].at(1).as_string();
    } else {
      const std::string& s = merges[r].as_string();
      auto sp = s.find(' ');
      if (sp == std::string::npos || s.find(' ', sp + 1) != std::string::npos)
        fail("merge " + std::to_string(r) + " is not 'a b': " + s);
      a = s.substr(0, sp);
      b = s.substr(sp + 1);
    }
    auto ia = p->vocab.find(a), ib = p->vocab.find(b), ic = p->vocab.find(a + b);
    if (ia == p->vocab.end() || ib == p->vocab.end() || ic == p->vocab.end())
      fail("merge " + std::to_string(r) + " uses a token missing from the vocab: " + a + " " + b);
    p->merges[Impl::key(ia->second, ib->second)] = {uint32_t(r), ic->second};   // later duplicates win, as in HF
  }
  for (const Json& a : added) {
    AddedToken t;
    t.id = uint32_t(a["id"].as_uint());
    t.content = a["content"].as_string();
    t.special = bool_of(a, "special");
    t.lstrip = bool_of(a, "lstrip");
    t.rstrip = bool_of(a, "rstrip");
    t.normalized = bool_of(a, "normalized");
    t.single_word = bool_of(a, "single_word");
    if (t.content.empty()) fail("empty added token");
    std::vector<uint32_t> cps, offs;
    decode_utf8(t.content, cps, offs);
    std::size_t idx = p->added.size();
    p->id_token[t.id] = t.content;
    p->id_bytes[t.id] = bytelevel_bytes(t.content);
    p->id_used[t.id] = 1;
    p->id_added[t.id] = int32_t(idx);
    p->added_by_content[t.content] = t.id;
    (t.normalized ? p->norm_trie : p->raw_trie).add(t.normalized && p->nfc ? nfc_impl(t.content) : t.content, idx);
    p->added.push_back(std::move(t));
  }
  return Tokenizer(std::move(p));
}

std::vector<uint32_t> Tokenizer::encode(std::string_view utf8, bool /*add_special_tokens*/) const {
  // add_special_tokens only selects the post-processor, and the supported ones (ByteLevel or none) add no tokens.
  if (!tok_detail::utf8_valid(utf8)) fail("input is not valid UTF-8");
  std::vector<uint32_t> out;
  out.reserve(utf8.size() / 3 + 4);
  std::vector<std::pair<int64_t, std::pair<std::size_t, std::size_t>>> segs, sub;
  p_->split_added(utf8, p_->raw_trie, segs);
  for (const auto& [tok, r] : segs) {
    if (tok >= 0) { out.push_back(p_->added[std::size_t(tok)].id); continue; }
    std::string_view seg = utf8.substr(r.first, r.second - r.first);
    std::string norm = p_->nfc ? nfc_impl(seg) : std::string(seg);
    if (p_->norm_trie.empty()) { p_->encode_plain(norm, out); continue; }
    p_->split_added(norm, p_->norm_trie, sub);
    for (const auto& [t2, r2] : sub) {
      if (t2 >= 0) out.push_back(p_->added[std::size_t(t2)].id);
      else p_->encode_plain(std::string_view(norm).substr(r2.first, r2.second - r2.first), out);
    }
  }
  return out;
}

std::string Tokenizer::decode(std::span<const uint32_t> ids, bool skip_special) const {
  std::string out, buf;
  for (uint32_t id : ids) {
    if (id >= p_->id_used.size() || !p_->id_used[id]) fail("decode: unknown id " + std::to_string(id));
    int32_t a = p_->id_added[id];
    if (a >= 0) {
      const AddedToken& t = p_->added[std::size_t(a)];
      if (skip_special && t.special) continue;
      // HF decodes an added token through the decoder on its own, apart from the neighbouring model tokens; so
      // "<|im_end|>" is printed verbatim, while non-ASCII content goes through the byte-level map.
      out += tok_detail::utf8_lossy(buf);
      buf.clear();
      out += tok_detail::utf8_lossy(p_->id_bytes[id]);
    } else {
      buf += p_->id_bytes[id];
    }
  }
  out += tok_detail::utf8_lossy(buf);
  return out;
}

int64_t Tokenizer::token_id(std::string_view token) const {
  std::string k(token);
  if (auto it = p_->added_by_content.find(k); it != p_->added_by_content.end()) return it->second;
  if (auto it = p_->vocab.find(k); it != p_->vocab.end()) return it->second;
  return -1;
}

const std::string& Tokenizer::id_to_token(uint32_t id) const {
  if (id >= p_->id_used.size() || !p_->id_used[id]) fail("unknown id " + std::to_string(id));
  return p_->id_token[id];
}

std::size_t Tokenizer::vocab_size() const { return p_->id_used.size(); }
const std::vector<AddedToken>& Tokenizer::added_tokens() const { return p_->added; }
bool Tokenizer::is_special(uint32_t id) const {
  return id < p_->id_added.size() && p_->id_added[id] >= 0 && p_->added[std::size_t(p_->id_added[id])].special;
}

std::vector<uint32_t> Tokenizer::eos_ids() const {
  std::vector<uint32_t> r;
  for (const char* s : {"<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "</s>",
                        "<\uff5cend\u2581of\u2581sentence\uff5c>"}) {
    auto it = p_->added_by_content.find(s);
    if (it != p_->added_by_content.end() && is_special(it->second)) r.push_back(it->second);
  }
  return r;
}

std::vector<uint32_t> Tokenizer::turn_end_ids() const {
  std::vector<uint32_t> r;
  for (const char* s : {"<|user|>", "<|observation|>"}) {
    auto it = p_->added_by_content.find(s);
    if (it != p_->added_by_content.end() && is_special(it->second)) r.push_back(it->second);
  }
  return r;
}

std::size_t Tokenizer::cache_size() const {
  std::lock_guard<std::mutex> lk(p_->cache_mu);
  return p_->cache.size();
}

// ---------------------------------------------------------------------------------------------------------------

std::string resolve_tokenizer_path(std::string_view tokenizer_id) {
  namespace fs = std::filesystem;
  std::string id(tokenizer_id), sha;
  if (auto semi = id.find(';'); semi != std::string::npos) {
    std::string rest = id.substr(semi + 1);
    id.resize(semi);
    const std::string pre = "tokenizer.json:sha256:";
    if (rest.rfind(pre, 0) != 0) fail("tokenizer_id: unsupported suffix '" + rest + "' (expected " + pre + "<hex>)");
    sha = rest.substr(pre.size());
    if (sha.size() != 64 || sha.find_first_not_of("0123456789abcdef") != std::string::npos)
      fail("tokenizer_id: bad sha256 '" + sha + "'");
  }
  std::string model = id, rev;
  if (auto at = id.rfind('@'); at != std::string::npos) {
    model = id.substr(0, at);
    rev = id.substr(at + 1);
  }
  if (model.empty()) fail("tokenizer_id: empty model id in '" + std::string(tokenizer_id) + "'");
  std::string path;
  std::error_code ec;
  if (fs::is_directory(model, ec)) {
    path = (fs::path(model) / "tokenizer.json").string();
  } else {
    std::string hub;
    if (const char* h = std::getenv("HF_HUB_CACHE"); h && *h) hub = h;
    else if (const char* hh = std::getenv("HF_HOME"); hh && *hh) hub = std::string(hh) + "/hub";
    else if (const char* home = std::getenv("HOME"); home && *home) hub = std::string(home) + "/.cache/huggingface/hub";
    else fail("cannot locate the HF hub cache (HF_HUB_CACHE, HF_HOME and HOME unset)");
    std::string repo = "models--" + model;
    for (std::size_t k; (k = repo.find('/')) != std::string::npos;) repo.replace(k, 1, "--");
    fs::path base = fs::path(hub) / repo;
    if (rev.empty() || rev == "None") rev = "main";
    bool is_hash = rev.size() == 40 && rev.find_first_not_of("0123456789abcdef") == std::string::npos;
    if (!is_hash) {
      fs::path ref = base / "refs" / rev;
      std::ifstream f(ref);
      std::string h;
      if (!(f >> h)) fail("revision '" + rev + "' of " + model + " not found in the HF cache (" + ref.string() + ")");
      rev = h;
    }
    path = (base / "snapshots" / rev / "tokenizer.json").string();
  }
  if (!fs::exists(path, ec))
    fail("tokenizer.json for '" + std::string(tokenizer_id) + "' not found: " + path +
         " (download it, e.g. huggingface_hub.hf_hub_download, or set HF_HUB_CACHE)");
  if (!sha.empty()) {
    std::string got = sha256_hex_file(path);
    if (got != sha) fail("sha256 mismatch for " + path + ": expected " + sha + ", got " + got);
  }
  return path;
}

namespace tok_detail {

uint32_t byte_to_unicode(uint8_t b) { return kByteMap[b]; }

std::vector<std::string> pre_tokenize(std::string_view utf8) {
  std::vector<std::pair<uint32_t, uint32_t>> pieces;
  std::vector<uint32_t> cps, offs;
  qwen_split(utf8, pieces, cps, offs);
  std::vector<std::string> r;
  for (auto [a, b] : pieces) r.emplace_back(utf8.substr(a, b - a));
  return r;
}

std::string nfc(std::string_view utf8) { return nfc_impl(utf8); }

bool utf8_valid(std::string_view s) {
  for (std::size_t i = 0; i < s.size();) {
    if (static_cast<unsigned char>(s[i]) < 0x80) { ++i; continue; }
    uint32_t cp;
    std::size_t m = 0, n = utf8_seq(s, i, &cp, &m);
    if (!n) return false;
    i += n;
  }
  return true;
}

std::string utf8_lossy(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size();) {
    uint32_t cp;
    std::size_t m = 0, n = utf8_seq(s, i, &cp, &m);
    if (n) {
      out.append(s.substr(i, n));
      i += n;
    } else {
      out += "\xEF\xBF\xBD";
      i += m;
    }
  }
  return out;
}

}  // namespace tok_detail

}  // namespace suc
