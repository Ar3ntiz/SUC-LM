// suc-tokenize: encode / decode text with the C++ byte-level BPE tokenizer (suc/tokenizer.hpp).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "suc/json.hpp"
#include "suc/suc1.hpp"
#include "suc/tokenizer.hpp"

using namespace suc;

static void usage() {
  std::fprintf(stderr,
               "usage: suc-tokenize (--tokenizer tokenizer.json | --suc1 FILE) [--encode-lines FILE] [--decode IDS]\n"
               "                    [--encode-json FILE] [--decode-json FILE] [--text TEXT] [--skip-special]\n"
               "                    [--add-special-tokens] [--json]\n"
               "  --tokenizer PATH     HF tokenizer.json\n"
               "  --suc1 FILE          resolve the tokenizer from the SUC1 header tokenizer_id (HF cache, sha256 checked)\n"
               "  --encode-lines FILE  one output line of comma-separated ids per input line (the line without its\n"
               "                       '\\n'; a trailing '\\r' is kept); prints encode throughput on stderr\n"
               "  --encode-json FILE   FILE is a JSON array of strings (any content, e.g. embedded newlines); one output line\n"
               "                       of comma-separated ids per string; prints encode throughput on stderr\n"
               "  --decode-json FILE   FILE is a JSON array of id arrays; one JSON string per output line\n"
               "  --text TEXT          print the ids of TEXT (comma-separated)\n"
               "  --decode IDS         print the text of comma-separated IDS\n"
               "  --skip-special       decode: drop special tokens\n"
               "  --add-special-tokens encode with the post-processor (adds nothing for ByteLevel, i.e. Qwen3)\n"
               "  --json               JSON output: {\"ids\": [...]} / {\"text\": \"...\"} / one JSON array per line\n");
}

static std::string ids_str(const std::vector<uint32_t>& ids, bool json) {
  std::string s = json ? "[" : "";
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i) s += ',';
    s += std::to_string(ids[i]);
  }
  return json ? s + "]" : s;
}

static std::string json_str(const std::string& t) {
  std::string out = "\"";
  for (unsigned char c : t) {
    if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else if (c < 0x20) {
      char b[8];
      std::snprintf(b, sizeof b, "\\u%04x", c);
      out += b;
    } else out += char(c);
  }
  return out + "\"";
}

int main(int argc, char** argv) {
  std::string tok_path, suc1, lines_path, json_path, decode_json_path, text, decode_ids;
  bool have_text = false, have_decode = false, json = false, skip_special = false, add_special = false;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](std::string& dst) {
      if (i + 1 >= argc) { usage(); std::exit(2); }
      dst = argv[++i];
    };
    if (!std::strcmp(argv[i], "--tokenizer")) arg(tok_path);
    else if (!std::strcmp(argv[i], "--suc1")) arg(suc1);
    else if (!std::strcmp(argv[i], "--encode-lines")) arg(lines_path);
    else if (!std::strcmp(argv[i], "--encode-json")) arg(json_path);
    else if (!std::strcmp(argv[i], "--decode-json")) arg(decode_json_path);
    else if (!std::strcmp(argv[i], "--text")) { arg(text); have_text = true; }
    else if (!std::strcmp(argv[i], "--decode")) { arg(decode_ids); have_decode = true; }
    else if (!std::strcmp(argv[i], "--json")) json = true;
    else if (!std::strcmp(argv[i], "--skip-special")) skip_special = true;
    else if (!std::strcmp(argv[i], "--add-special-tokens")) add_special = true;
    else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) { usage(); return 0; }
    else { usage(); return 2; }
  }
  if (tok_path.empty() == suc1.empty()) { usage(); return 2; }
  try {
    auto t0 = std::chrono::steady_clock::now();
    if (!suc1.empty()) {
      OpenOptions opt;
      opt.verify_checksum = false;   // only the directory is needed here
      tok_path = resolve_tokenizer_path(Suc1File::open(suc1, opt).tokenizer_id());
    }
    Tokenizer tok = Tokenizer::from_file(tok_path);
    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "tokenizer: %s (%zu ids, loaded in %.2f s)\n", tok_path.c_str(), tok.vocab_size(), load_s);

    auto slurp = [](const std::string& path) {
      std::ifstream f(path, std::ios::binary);
      if (!f) throw TokenizerError("cannot open " + path);
      std::stringstream ss;
      ss << f.rdbuf();
      return ss.str();
    };
    std::string all;
    std::vector<std::string_view> lines;
    Json strings;
    if (!json_path.empty()) {
      strings = Json::parse(slurp(json_path));
      for (const Json& v : strings.as_array()) lines.emplace_back(v.as_string());
    } else if (!lines_path.empty()) {
      all = slurp(lines_path);
      for (std::size_t p = 0; p <= all.size();) {
        std::size_t e = all.find('\n', p);
        if (e == std::string::npos) {
          if (p < all.size()) lines.emplace_back(all.data() + p, all.size() - p);
          break;
        }
        lines.emplace_back(all.data() + p, e - p);
        p = e + 1;
      }
    }
    if (!json_path.empty() || !lines_path.empty()) {
      std::vector<std::vector<uint32_t>> out(lines.size());
      std::size_t bytes = 0, ntok = 0;
      auto e0 = std::chrono::steady_clock::now();
      for (std::size_t i = 0; i < lines.size(); ++i) {
        out[i] = tok.encode(lines[i], add_special);
        bytes += lines[i].size();
        ntok += out[i].size();
      }
      double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - e0).count();
      std::string o;
      for (const auto& ids : out) o += ids_str(ids, json) + "\n";
      std::fwrite(o.data(), 1, o.size(), stdout);
      std::fprintf(stderr, "encoded %zu lines, %zu bytes -> %zu tokens in %.3f s: %.2f MB/s, %.0f tokens/s (1 thread)\n",
                   lines.size(), bytes, ntok, s, bytes / 1e6 / (s > 0 ? s : 1e-9), ntok / (s > 0 ? s : 1e-9));
    }
    if (!decode_json_path.empty()) {
      Json arr = Json::parse(slurp(decode_json_path));
      std::string o;
      for (const Json& row : arr.as_array()) {
        std::vector<uint32_t> ids;
        for (const Json& v : row.as_array()) ids.push_back(uint32_t(v.as_uint()));
        o += json_str(tok.decode(ids, skip_special)) + "\n";
      }
      std::fwrite(o.data(), 1, o.size(), stdout);
    }
    if (have_text) {
      auto ids = tok.encode(text, add_special);
      std::printf(json ? "{\"ids\": %s}\n" : "%s\n", ids_str(ids, json).c_str());
    }
    if (have_decode) {
      std::vector<uint32_t> ids;
      for (std::size_t p = 0; p < decode_ids.size();) {
        std::size_t e = decode_ids.find(',', p);
        if (e == std::string::npos) e = decode_ids.size();
        std::string item = decode_ids.substr(p, e - p);
        if (!item.empty()) ids.push_back(uint32_t(std::stoul(item)));
        p = e + 1;
      }
      std::string s = tok.decode(ids, skip_special);
      if (json) std::printf("{\"text\": %s}\n", json_str(s).c_str());
      else std::printf("%s\n", s.c_str());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "suc-tokenize: %s\n", e.what());
    return 1;
  }
  return 0;
}
