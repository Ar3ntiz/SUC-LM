// Unit tests for libsuc (no framework): SHA-256 vectors, bit unpacking at widths 1..32, the numpy
// PCG64 sign stream, fp16/bf16 conversion, JSON parsing, Hadamard transforms and SUC1 header /
// footer / directory validation on synthetic files, and the tokenizer building blocks (byte-level map, UTF-8,
// Qwen regex pre-tokenizer, NFC, BPE on a synthetic tokenizer.json). Bit-exactness against the Python reference
// decoder is tested from Python (tests/unit/test_cpp_decoder.py).
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "suc/bits.hpp"
#include "suc/decode.hpp"
#include "suc/hadamard.hpp"
#include "suc/json.hpp"
#include "suc/npy.hpp"
#include "suc/numpy_random.hpp"
#include "suc/request_params.hpp"
#include "suc/sampling.hpp"
#include "suc/prefill.hpp"
#include "suc/prompt_lookup.hpp"
#include "suc/parallel.hpp"
#include "suc/sha256.hpp"
#include "suc/suc1.hpp"
#include "suc/tokenizer.hpp"

using namespace suc;

static int g_fail = 0, g_checks = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_fail;                                                              \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
    }                                                                        \
  } while (0)

template <typename F>
static bool throws(F&& f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

static std::string hex_of(const std::string& s, bool portable = false, std::size_t chunk = 0) {
  Sha256 h;
  h.set_portable(portable);
  if (!chunk) h.update(s.data(), s.size());
  else
    for (std::size_t i = 0; i < s.size(); i += chunk) h.update(s.data() + i, std::min(chunk, s.size() - i));
  auto d = h.finish();
  return to_hex(d.data(), d.size());
}

static void test_sha256() {
  std::string all256;
  for (int r = 0; r < 37; ++r)
    for (int i = 0; i < 256; ++i) all256 += char(i);
  const std::pair<std::string, const char*> vec[] = {
      {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
      {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
       "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
      {std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"},
      {all256, "480d8641d6c1055a8755af19a919c09c6b3f4f57ee7615d67da1f902e8e4c782"},  // hashlib
  };
  for (const auto& [msg, want] : vec) {
    CHECK(hex_of(msg) == want);
    CHECK(hex_of(msg, true) == want);
    CHECK(hex_of(msg, false, 7) == want);
    CHECK(hex_of(msg, true, 63) == want);
    CHECK(hex_of(msg, false, 65) == want);
  }
  // every length 0..200 (padding boundaries): hardware and portable paths agree
  std::string s;
  for (int n = 0; n <= 200; ++n) {
    CHECK(hex_of(s) == hex_of(s, true));
    s += char('a' + n % 26);
  }
  std::printf("sha256: hardware path %s\n", Sha256::hardware_available() ? "available" : "not available");
}

// Independent bit-by-bit packer of the rows_u32le_bitstream layout.
static std::vector<uint32_t> pack_ref(const std::vector<uint32_t>& v, unsigned bits) {
  std::vector<uint32_t> w(words_per_row(v.size(), bits), 0);
  for (std::size_t j = 0; j < v.size(); ++j)
    for (unsigned b = 0; b < bits; ++b)
      if ((v[j] >> b) & 1) {
        const uint64_t p = uint64_t(j) * bits + b;
        w[p / 32] |= 1u << (p % 32);
      }
  return w;
}

static void test_unpack() {
  std::mt19937_64 rng(123);
  for (unsigned bits = 1; bits <= 32; ++bits)
    for (std::size_t n : {1, 2, 3, 7, 31, 32, 33, 64, 100, 257}) {
      std::vector<uint32_t> v(n);
      const uint64_t lim = bits == 32 ? (1ull << 32) : (1ull << bits);
      for (auto& x : v) x = uint32_t(rng() % lim);
      v[0] = uint32_t(lim - 1);   // all-ones element
      auto w = pack_ref(v, bits);
      const auto* bytes = reinterpret_cast<const uint8_t*>(w.data());
      std::vector<uint32_t> got(n);
      unpack_row(bytes, bits, n, got.data());
      CHECK(got == v);
      bool one_ok = true;
      for (std::size_t j = 0; j < n; ++j) one_ok &= unpack_one(bytes, bits, j) == v[j];
      CHECK(one_ok);
    }
  // anchors produced by suc.quant.packing.pack_indices
  {
    const uint32_t w[] = {2139816925u, 0u};
    std::vector<uint32_t> got(11);
    unpack_row(reinterpret_cast<const uint8_t*>(w), 3, 11, got.data());
    CHECK((got == std::vector<uint32_t>{5, 3, 7, 1, 0, 6, 2, 4, 7, 7, 1}));
  }
  {
    const uint32_t w[] = {1610614783u, 19u};
    std::vector<uint32_t> got(4);
    unpack_row(reinterpret_cast<const uint8_t*>(w), 10, 4, got.data());
    CHECK((got == std::vector<uint32_t>{1023, 1, 512, 77}));
  }
  {
    const uint32_t w[] = {2863333133u, 3u};
    std::vector<uint32_t> want = {1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                                  1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1};
    std::vector<uint32_t> got(want.size());
    unpack_row(reinterpret_cast<const uint8_t*>(w), 1, want.size(), got.data());
    CHECK(got == want);
  }
  CHECK(throws([] { uint32_t w = 0, o; unpack_row(reinterpret_cast<const uint8_t*>(&w), 0, 1, &o); }));
}

static void test_numpy_random() {
  // np.random.default_rng(seed).integers(0, 2, size=80)
  const std::pair<uint64_t, const char*> vec[] = {
      {0u, "11100000011111111111011001101110010100000000011101100111011111101011110000110110"},
      {1u, "01110011001001001100111110010000100010100100000111010110100101000101011100111000"},
      {42u, "01100101001111111010100111011000011011010110010111100000101111010010001000100011"},
      {1824989360u, "11011000110010110101111000001001111011101011111111101011001000001111010010111101"},
      {3647390468u, "10101101100010000101111010101111101000000101100010001011010000101100011111011010"},
      {8589934599u, "00000111100000100001100011010001110111111000111110101000101011000100111110000001"},
      {18446744073709551615ull, "11110001100011000101010011111110000101110110110000110100101010001111110110010110"},
  };
  for (const auto& [seed, want] : vec) {
    auto b = numpy_random_bits(80, seed);
    std::string got;
    for (auto x : b) got += char('0' + x);
    CHECK(got == want);
    auto s = random_signs(80, seed);
    bool ok = true;
    for (std::size_t i = 0; i < 80; ++i) ok &= s[i] == (want[i] == '1' ? 1.0f : -1.0f);
    CHECK(ok);
  }
}

static void test_half() {
  CHECK(f16_to_f32(0x3C00) == 1.0f);
  CHECK(f16_to_f32(0xC000) == -2.0f);
  CHECK(f16_to_f32(0x7BFF) == 65504.0f);
  CHECK(f16_to_f32(0x0400) == std::ldexp(1.0f, -14));
  CHECK(f16_to_f32(0x0001) == std::ldexp(1.0f, -24));
  CHECK(f16_to_f32(0x03FF) == std::ldexp(1023.0f, -24));
  CHECK(f16_to_f32(0x3555) == 0.333251953125f);
  CHECK(std::signbit(f16_to_f32(0x8000)) && f16_to_f32(0x8000) == 0.0f);
  CHECK(std::isinf(f16_to_f32(0x7C00)) && f16_to_f32(0xFC00) < 0);
  CHECK(std::isnan(f16_to_f32(0x7E00)));
  CHECK(bf16_to_f32(0x3F80) == 1.0f);
  CHECK(bf16_to_f32(0xC0A0) == -5.0f);
  // exhaustive: every finite half equals mantissa * 2^exponent computed in double
  bool ok = true;
  for (uint32_t h = 0; h < 0x10000; ++h) {
    const uint32_t e = (h >> 10) & 31, m = h & 1023;
    if (e == 31) continue;
    double v = e ? std::ldexp(1024.0 + m, int(e) - 25) : std::ldexp(double(m), -24);
    if (h & 0x8000) v = -v;
    ok &= double(f16_to_f32(uint16_t(h))) == v;
  }
  CHECK(ok);
}

static void test_json() {
  Json j = Json::parse(R"({"a":[1,-2,3.5,1e3,true,false,null],"s":"x\"\\\/\n\u00e9\ud83d\ude00","big":18446744073709551615,"o":{}})");
  CHECK(j["a"].size() == 7);
  CHECK(j["a"].at(0).as_int() == 1 && j["a"].at(1).as_int() == -2);
  CHECK(j["a"].at(2).as_double() == 3.5 && j["a"].at(3).as_int() == 1000);
  CHECK(j["a"].at(4).as_bool() && !j["a"].at(5).as_bool() && j["a"].at(6).is_null());
  CHECK(j["s"].as_string() == "x\"\\/\n\xc3\xa9\xf0\x9f\x98\x80");
  CHECK(j["big"].is_number() && !j["big"].is_int());
  CHECK(j["o"].is_object() && j["o"].size() == 0);
  CHECK(j.find("missing") == nullptr);
  CHECK(throws([&] { (void)j["missing"]; }));
  CHECK(throws([&] { (void)j["s"].as_int(); }));
  CHECK(throws([&] { (void)j["a"].at(1).as_uint(); }));
  for (const char* bad : {"", "{", "[1,]", "{\"a\" 1}", "tru", "\"abc", "01x", "{} x", "\"\\ud800\"", "[1 2]"})
    CHECK(throws([&] { Json::parse(bad); }));
  CHECK(Json::parse(" [ 1 , {\"k\" : \"v\"} ] ").dump() == "[1,{\"k\":\"v\"}]");
}

static void test_hadamard() {
  // small integers: the FWHT of size 16 is exact, the scale 1/4 too -> compare with Sylvester H
  std::vector<float> x(16), scratch(32);
  for (int i = 0; i < 16; ++i) x[i] = float(i * i % 7 - 3);
  std::vector<float> ref(16, 0.0f);
  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < 16; ++k) ref[i] += (__builtin_popcount(i & k) & 1 ? -1.0f : 1.0f) * x[k] * 0.25f;
  auto y = x;
  hadamard_transform(y.data(), 16, false, scratch.data());
  CHECK(y == ref);
  // orthonormality / inverse on every supported family: H^T (H x) == x up to rounding
  std::mt19937 rng(7);
  std::normal_distribution<float> nd;
  for (std::size_t n : {2u, 64u, 1024u, 12u * 64u, 20u * 32u, 28u * 16u, 16384u, 12u * 16384u}) {
    std::vector<float> v(n), w, s(2 * n);
    for (auto& a : v) a = nd(rng);
    w = v;
    hadamard_transform(w.data(), n, false, s.data());
    double e2 = 0, n2 = 0;
    for (std::size_t i = 0; i < n; ++i) n2 += double(w[i]) * w[i];
    hadamard_transform(w.data(), n, true, s.data());
    for (std::size_t i = 0; i < n; ++i) e2 = std::max(e2, std::fabs(double(w[i]) - v[i]));
    double v2 = 0;
    for (float a : v) v2 += double(a) * a;
    CHECK(e2 < 1e-4);
    CHECK(std::fabs(n2 - v2) / v2 < 1e-5);
  }
  CHECK(!hadamard_supported(9728) && !hadamard_supported(0) && hadamard_supported(3072));
  // factor and H_m accessors (used by the Metal backend): H_m H_m^T = I
  CHECK(hadamard_factor(12288).m == 12 && hadamard_factor(12288).p == 1024 && hadamard_factor(448).m == 28);
  CHECK(hadamard_factor(4096).m == 1 && hadamard_factor(9728).m == 0);
  CHECK(hadamard_hm(1) == std::vector<float>{1.0f});
  for (std::size_t m : {12u, 20u, 28u}) {
    const auto h = hadamard_hm(m);
    double e = 0;
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t j = 0; j < m; ++j) {
        double d = 0;
        for (std::size_t k = 0; k < m; ++k) d += double(h[i * m + k]) * h[j * m + k];
        e = std::max(e, std::fabs(d - (i == j ? 1.0 : 0.0)));
      }
    CHECK(h.size() == m * m && e < 1e-6);
  }
  std::vector<float> bad(19 * 512), bs(2 * 19 * 512);
  CHECK(throws([&] { hadamard_transform(bad.data(), bad.size(), false, bs.data()); }));
}

static void test_npy() {
  const std::string p = "/tmp/suc_test_npy_" + std::to_string(::getpid()) + ".npy";
  const std::vector<float> v = {1.5f, -2.0f, 3.25f, 0.0f, 1e-30f, -7.0f};
  for (const std::vector<uint64_t>& shape : {std::vector<uint64_t>{2, 3}, {6}, {1, 1, 6}}) {
    write_npy_f32(p, shape, v.data());
    std::vector<uint64_t> got;
    CHECK(read_npy_f32(p, got) == v);
    CHECK(got == shape);
  }
  write_raw(p, "not an npy", 10);
  std::vector<uint64_t> sh;
  CHECK(throws([&] { read_npy_f32(p, sh); }));
  ::unlink(p.c_str());
}

// ------------------------------------------------------------------------------ synthetic SUC1 files

static void put64(std::string& b, std::size_t at, uint64_t v) { std::memcpy(&b[at], &v, 8); }
static void put32(std::string& b, std::size_t at, uint32_t v) { std::memcpy(&b[at], &v, 4); }
static uint64_t align64(uint64_t n) { return (n + 63) / 64 * 64; }

// One raw f32 tensor "w" of shape (2, 3); directory built with offsets resolved.
static std::string make_file(const std::function<void(std::string& dir)>& edit_dir = {},
                             const std::function<void(std::string& file)>& edit_bytes = {}, bool refresh_sha = true) {
  const float vals[6] = {1.5f, -2.0f, 0.25f, 3.0f, -0.0f, 7.0f};
  auto dir_for = [](uint64_t payload_off) {
    return std::string(R"({"alignment":64,"architecture":{"model_type":"qwen3"},"architecture_id":"qwen3",)"
                       R"("byte_order":"little","codebooks":[],"format":"SUC1","format_version":1,"global_flags":0,)"
                       R"("metadata":{},"sections":["codebooks","indices","scales","residual","raw"],)"
                       R"("tensors":[{"arrays":[{"dtype":"f32","key":"weight","nbytes":24,"offset":)") +
           std::to_string(payload_off) +
           R"(,"part":"raw","shape":[2,3]}],"block_size":null,"codebook_id":null,"codebook_ids":[],)"
           R"("codebook_key":null,"codec":null,"compressed_size":24,"data_offset":)" +
           std::to_string(payload_off) +
           R"(,"index_bits":null,"layout":"row_major","logical_size":6,"name":"w","params":{},)"
           R"("representation_type":"raw","residual_type":"none","rotation":"none","shape":[2,3]}],)"
           R"("tokenizer_id":"tok"})";
  };
  uint64_t payload_off = 64;
  std::string dir;
  for (;;) {
    dir = dir_for(payload_off);
    if (align64(64 + dir.size()) <= payload_off) break;
    payload_off = align64(64 + dir.size());
  }
  if (edit_dir) edit_dir(dir);
  const uint64_t footer_off = align64(payload_off + 24);
  std::string b(footer_off + 64, '\0');
  std::memcpy(&b[0], "SUC1", 4);
  put32(b, 4, 1);
  put64(b, 8, 0);
  put64(b, 16, 1);
  put64(b, 24, 0);
  put64(b, 32, 64);
  put64(b, 40, dir.size());
  put64(b, 48, payload_off);
  put64(b, 56, footer_off);
  std::memcpy(&b[64], dir.data(), dir.size());
  std::memcpy(&b[payload_off], vals, 24);
  std::memcpy(&b[footer_off], "SUCF", 4);
  put32(b, footer_off + 4, 1);
  put64(b, footer_off + 8, footer_off);
  if (edit_bytes) edit_bytes(b);
  if (refresh_sha) {
    auto d = sha256(b.data(), footer_off);
    std::memcpy(&b[footer_off + 16], d.data(), 32);
  }
  return b;
}

static std::string tmp_path(const char* name) {
  const char* t = std::getenv("TMPDIR");
  return std::string(t ? t : "/tmp") + "/suc_tests_" + std::to_string(getpid()) + "_" + name;
}

static std::string write_tmp(const std::string& bytes, const char* name) {
  const std::string p = tmp_path(name);
  std::FILE* fp = std::fopen(p.c_str(), "wb");
  std::fwrite(bytes.data(), 1, bytes.size(), fp);
  std::fclose(fp);
  return p;
}

static bool opens(const std::string& bytes, OpenOptions opt = {}) {
  const std::string p = write_tmp(bytes, "case.suc1");
  bool ok = !throws([&] { Suc1File::open(p, opt); });
  std::remove(p.c_str());
  return ok;
}

static void test_container() {
  const std::string good = make_file();
  {
    const std::string p = write_tmp(good, "good.suc1");
    Suc1File f = Suc1File::open(p);
    CHECK(f.checksum_verified());
    CHECK(f.header().format_version == 1 && f.header().number_of_tensors == 1);
    CHECK(f.header().payload_offset % 64 == 0 && f.file_size() == f.header().footer_offset + 64);
    CHECK(f.architecture_id() == "qwen3" && f.tokenizer_id() == "tok");
    CHECK(f.tensors().size() == 1 && f.tensor("w").shape == (std::vector<uint64_t>{2, 3}));
    CHECK(f.find_tensor("nope") == nullptr);
    auto d = decode_tensor(f, "w");
    CHECK((d.data == std::vector<float>{1.5f, -2.0f, 0.25f, 3.0f, -0.0f, 7.0f}));
    CHECK(std::signbit(d.data[4]));
    auto a = f.accounting();
    CHECK(a.header + a.directory + a.codebooks + a.indices + a.scales + a.residual + a.raw + a.padding + a.footer ==
          a.total);
    CHECK(a.raw == 24);
    Suc1File g = std::move(f);   // move keeps the mapping valid
    CHECK(decode_tensor(g, "w").data[5] == 7.0f);
    std::remove(p.c_str());
  }
  CHECK(opens(good));
  // fixed header
  CHECK(!opens(make_file({}, [](std::string& b) { b[0] = 'X'; })));                  // magic
  CHECK(!opens(make_file({}, [](std::string& b) { put32(b, 4, 2); })));              // version
  CHECK(!opens(make_file({}, [](std::string& b) { put64(b, 8, 8); })));              // unknown flag bit
  CHECK(!opens(make_file({}, [](std::string& b) { put64(b, 8, 1); })));              // flag not matching content
  CHECK(!opens(make_file({}, [](std::string& b) { put64(b, 16, 2); })));             // tensor count
  CHECK(!opens(make_file({}, [](std::string& b) { put64(b, 48, 65); })));            // misaligned payload_offset
  CHECK(!opens(make_file({}, [](std::string& b) { put64(b, 32, 128); })));           // directory_offset
  CHECK(!opens(good.substr(0, good.size() - 1)));                                     // truncated
  CHECK(!opens(good.substr(0, 100)));
  CHECK(!opens(good + std::string(64, '\0')));                                        // trailing bytes
  // footer
  const uint64_t fo = good.size() - 64;
  CHECK(!opens(make_file({}, [fo](std::string& b) { b[fo] = 'X'; })));               // footer magic
  CHECK(!opens(make_file({}, [fo](std::string& b) { put32(b, fo + 4, 2); })));       // checksum algorithm
  CHECK(!opens(make_file({}, [fo](std::string& b) { put64(b, fo + 8, 0); })));       // repeated offset
  // checksum: a flipped payload byte fails verification but opens with verify off
  std::string flipped = good;
  flipped[fo - 64] ^= 1;
  CHECK(!opens(flipped));
  OpenOptions nov;
  nov.verify_checksum = false;
  CHECK(opens(flipped, nov));
  // directory content
  auto repl = [](std::string from, std::string to) {
    return [from, to](std::string& d) {
      auto at = d.find(from);
      if (at != std::string::npos) d.replace(at, from.size(), to);
    };
  };
  CHECK(!opens(make_file(repl("\"format_version\":1", "\"format_version\":2"))));
  CHECK(!opens(make_file(repl("\"nbytes\":24", "\"nbytes\":28"))));                 // nbytes != shape*itemsize
  CHECK(!opens(make_file(repl("\"compressed_size\":24", "\"compressed_size\":25"))));
  CHECK(!opens(make_file(repl("\"dtype\":\"f32\"", "\"dtype\":\"f99\""))));
  CHECK(!opens(make_file(repl("\"alignment\":64", "\"alignment\":32"))));
  CHECK(!opens(make_file(repl("{\"alignment\"", "{{\"alignment\""))));              // corrupt JSON
  CHECK(!opens(make_file(repl("\"sections\":[\"codebooks\"", "\"sections\":[\"indices\""))));
  // checked arithmetic (audit observation "reader hardening"): a shape whose product wraps to 6 (11 * 1676976733973595602
  // = 2^64 + 6, so prod * 4 bytes == 24 == nbytes when computed modulo 2^64) and header offset sums that overflow
  CHECK(!opens(make_file(repl("\"part\":\"raw\",\"shape\":[2,3]", "\"part\":\"raw\",\"shape\":[1676976733973595602,11]"))));
  CHECK(!opens(make_file({}, [](std::string& b) { put64(b, 40, ~uint64_t(0) - 10); })));   // directory_size
  CHECK(!opens(make_file({}, [](std::string& b) { put64(b, 56, ~uint64_t(0) - 63); })));   // footer_offset
}

// Expected pieces / NFC outputs were produced with HF tokenizers 0.23 (Split with the Qwen3 regex, normalizers.NFC).
// ------------------------------------------------------------------------------ native MXFP4 decode (audit F05)

// One mxfp4 tensor "w" (out, inn): codes u8 [cr, cc], scales u8 [sr, sc] (given separately so that malformed shapes
// can be declared), group_size gs; arrays 64-byte aligned like the Python writer.
static std::string make_mxfp4_file(uint64_t out, uint64_t inn, const std::vector<uint8_t>& codes, uint64_t cr,
                                   uint64_t cc, const std::vector<uint8_t>& scales, uint64_t sr, uint64_t sc,
                                   uint64_t gs = 32) {
  auto dir_for = [&](uint64_t off) {
    const uint64_t soff = off + align64(codes.size());
    return std::string(R"({"alignment":64,"architecture":{},"architecture_id":"probe","byte_order":"little",)"
                       R"("codebooks":[],"format":"SUC1","format_version":1,"global_flags":0,"metadata":{},)"
                       R"("sections":["codebooks","indices","scales","residual","raw"],"tensors":[{"arrays":[)"
                       R"({"dtype":"u8","key":"codes","nbytes":)") + std::to_string(codes.size()) +
           R"(,"offset":)" + std::to_string(off) + R"(,"part":"indices","shape":[)" + std::to_string(cr) + "," +
           std::to_string(cc) + R"(]},{"dtype":"u8","key":"scales","nbytes":)" + std::to_string(scales.size()) +
           R"(,"offset":)" + std::to_string(soff) + R"(,"part":"scales","shape":[)" + std::to_string(sr) + "," +
           std::to_string(sc) + R"(]}],"block_size":32,"codebook_ids":[],"codebook_key":null,"codec":"mxfp4",)"
           R"("compressed_size":)" + std::to_string(codes.size() + scales.size()) + R"(,"data_offset":)" +
           std::to_string(off) + R"(,"index_bits":4,"layout":"mxfp4_u8_nibbles","logical_size":)" +
           std::to_string(out * inn) + R"(,"name":"w","params":{"element":"e2m1","format":"mxfp4","group_size":)" +
           std::to_string(gs) + R"(,"index_bits":4,"scale":"e8m0","source":"verbatim"},)"
           R"("representation_type":"mxfp4","residual_type":"none","rotation":"none","shape":[)" +
           std::to_string(out) + "," + std::to_string(inn) + R"(]}],"tokenizer_id":"probe"})";
  };
  uint64_t off = 64;
  std::string dir;
  for (;;) {
    dir = dir_for(off);
    if (align64(64 + dir.size()) <= off) break;
    off = align64(64 + dir.size());
  }
  const uint64_t soff = off + align64(codes.size());
  const uint64_t footer_off = align64(soff + scales.size());
  std::string b(footer_off + 64, '\0');
  std::memcpy(&b[0], "SUC1", 4);
  put32(b, 4, 1);
  put64(b, 16, 1);
  put64(b, 32, 64);
  put64(b, 40, dir.size());
  put64(b, 48, off);
  put64(b, 56, footer_off);
  std::memcpy(&b[64], dir.data(), dir.size());
  if (!codes.empty()) std::memcpy(&b[off], codes.data(), codes.size());
  if (!scales.empty()) std::memcpy(&b[soff], scales.data(), scales.size());
  std::memcpy(&b[footer_off], "SUCF", 4);
  put32(b, footer_off + 4, 1);
  put64(b, footer_off + 8, footer_off);
  auto d = sha256(b.data(), footer_off);
  std::memcpy(&b[footer_off + 16], d.data(), 32);
  return b;
}

// decode tensor "w" of a synthetic file; returns false when open or decode throws
static bool decode_mxfp4(const std::string& bytes, std::vector<float>* got = nullptr) {
  const std::string p = write_tmp(bytes, "mxfp4.suc1");
  bool ok = !throws([&] {
    Suc1File f = Suc1File::open(p);
    auto d = decode_tensor(f, f.tensor("w"));
    if (got) *got = d.data;
  });
  std::remove(p.c_str());
  return ok;
}

static void test_mxfp4_decode() {
  static const float e2m1[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6};
  std::vector<float> w;
  // the audit fixture: 1 x 32, every nibble 2 (= 1.0), scale 127 (= 2^0) -> 32 ones
  CHECK(decode_mxfp4(make_mxfp4_file(1, 32, std::vector<uint8_t>(16, 0x22), 1, 16, {127}, 1, 1), &w));
  CHECK(w == std::vector<float>(32, 1.0f));
  // all 16 codes in both nibble positions, 3 rows x 3 groups, extreme valid scales 0 (2^-127) and 254 (2^127)
  const uint64_t out = 3, inn = 96;
  std::vector<uint8_t> codes(out * inn / 2), scales(out * inn / 32);
  for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = uint8_t((i % 16) | (((i * 7 + 3) % 16) << 4));
  const uint8_t sv[9] = {127, 0, 254, 1, 126, 128, 200, 2, 251};
  for (std::size_t i = 0; i < scales.size(); ++i) scales[i] = sv[i];
  CHECK(decode_mxfp4(make_mxfp4_file(out, inn, codes, out, inn / 2, scales, out, inn / 32), &w));
  bool exact = w.size() == out * inn, negzero = false;
  for (std::size_t r = 0; r < out && exact; ++r)
    for (std::size_t k = 0; k < inn; ++k) {
      const uint8_t byte = codes[r * inn / 2 + k / 2];
      const int code = (k % 2) ? (byte >> 4) : (byte & 15);
      const float want = e2m1[code] * std::ldexp(1.0f, int(scales[r * 3 + k / 32]) - 127);
      exact = exact && w[r * inn + k] == want;
      negzero = negzero || (code == 8 && std::signbit(w[r * inn + k]));
    }
  CHECK(exact);
  CHECK(!negzero);   // code 8 (-0) decodes to +0 like suc.io.suc1._E2M1
  // malformed: scale 255 (E8M0 NaN), wrong code / scale shapes, width not divisible by 32, group size != 32
  CHECK(!decode_mxfp4(make_mxfp4_file(1, 32, std::vector<uint8_t>(16, 0x22), 1, 16, {255}, 1, 1)));
  CHECK(!decode_mxfp4(make_mxfp4_file(1, 32, std::vector<uint8_t>(32, 0x22), 1, 32, {127}, 1, 1)));
  CHECK(!decode_mxfp4(make_mxfp4_file(1, 32, std::vector<uint8_t>(16, 0x22), 1, 16, {127, 127}, 1, 2)));
  CHECK(!decode_mxfp4(make_mxfp4_file(1, 48, std::vector<uint8_t>(24, 0x22), 1, 24, {127}, 1, 1)));
  CHECK(!decode_mxfp4(make_mxfp4_file(1, 64, std::vector<uint8_t>(32, 0x22), 1, 32, {127}, 1, 1, 64)));
}

static void test_tokenizer() {
  using namespace suc::tok_detail;
  // GPT-2 bytes_to_unicode
  CHECK(byte_to_unicode('!') == '!' && byte_to_unicode('~') == '~' && byte_to_unicode(0xA1) == 0xA1);
  CHECK(byte_to_unicode(0) == 256 && byte_to_unicode(' ') == 0x120 && byte_to_unicode('\n') == 0x10A);
  CHECK(byte_to_unicode(0x7F) == 0x121 && byte_to_unicode(0xAD) == 0x143);
  // UTF-8 validity and Rust from_utf8_lossy (maximal subparts)
  CHECK(utf8_valid("") && utf8_valid("a\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80"));
  for (const char* bad : {"\xC0\x80", "\xED\xA0\x80", "\xF4\x90\x80\x80", "\xE4\xB8", "\x80", "\xFF"})
    CHECK(!utf8_valid(bad));
  const std::string R = "\xEF\xBF\xBD";
  CHECK(utf8_lossy("a\xE4\xB8") == "a" + R);
  CHECK(utf8_lossy("\xE4\xB8x\xB8\xAD") == R + "x" + R + R);
  CHECK(utf8_lossy("\xF0\x9F\x98") == R && utf8_lossy("\xED\xA0\x80") == R + R + R);
  CHECK(utf8_lossy("\xC3\xA9\xF5") == "\xC3\xA9" + R);
  // Qwen regex pre-tokenizer
  using V = std::vector<std::string>;
  CHECK(pre_tokenize("I'M they'LL x'S 've") == (V{"I", "'M", " they", "'LL", " x", "'S", " '", "ve"}));
  CHECK(pre_tokenize("a  \n\n  b") == (V{"a", "  \n\n", " ", " b"}));
  CHECK(pre_tokenize("a \t\n b  ") == (V{"a", " \t\n", " b", "  "}));
  CHECK(pre_tokenize("x\r\n\r\n  y") == (V{"x", "\r\n\r\n", " ", " y"}));
  CHECK(pre_tokenize("1234 ab12") == (V{"1", "2", "3", "4", " ab", "1", "2"}));
  CHECK(pre_tokenize("$%^ \n") == (V{"$%^", " \n"}));
  CHECK(pre_tokenize(" ?!\n\nz") == (V{" ?!\n\n", "z"}));
  CHECK(pre_tokenize("\xC2\xA0" "a caf\xC3\xA9") == (V{"\xC2\xA0" "a", " caf\xC3\xA9"}));
  CHECK(pre_tokenize("\xE4\xB8\xAD\xE6\x96\x87 abc") == (V{"\xE4\xB8\xAD\xE6\x96\x87", " abc"}));
  CHECK(pre_tokenize("").empty());
  // NFC
  CHECK(nfc("e\xCC\x81") == "\xC3\xA9");                                  // e + U+0301
  CHECK(nfc("a\xCC\xA3\xCC\x82") == "\xE1\xBA\xAD");                       // -> U+1EAD
  CHECK(nfc("s\xCC\x87\xCC\xA3") == "\xE1\xB9\xA9");                       // reordered -> U+1E69
  CHECK(nfc("\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8") == "\xEA\xB0\x81");    // Hangul L V T -> U+AC01
  CHECK(nfc("\xE2\x84\xAB") == "\xC3\x85");                                  // U+212B -> U+00C5
  CHECK(nfc("plain ascii") == "plain ascii");
  // BPE on a synthetic tokenizer.json: ids 0..255 are the byte-level characters of bytes 0..255.
  auto jstr = [](const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
      if (c == '"' || c == '\\') o += '\\';
      o += c;
    }
    return o + "\"";
  };
  auto bl = [](unsigned char b) {
    std::string s;
    uint32_t cp = byte_to_unicode(b);
    if (cp < 0x80) s += char(cp);
    else { s += char(0xC0 | (cp >> 6)); s += char(0x80 | (cp & 0x3F)); }
    return s;
  };
  auto make = [&](const std::string& normalizer, const std::string& model_type, const std::string& extra = "") {
    std::string v = "{";
    for (int b = 0; b < 256; ++b) v += (b ? "," : "") + jstr(bl(uint8_t(b))) + ":" + std::to_string(b);
    v += ",\"ab\":256,\"abc\":257," + jstr(bl(' ') + "a") + ":258}";
    return "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[{\"id\":259,"
           "\"content\":\"<|end|>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,"
           "\"special\":true}" + extra + "],\"normalizer\":" + normalizer +
           ",\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":"
           "\"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{N}]+"
           "[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+\"},\"behavior\":\"Isolated\",\"invert\":false},"
           "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}]},"
           "\"post_processor\":null,\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,"
           "\"trim_offsets\":false,\"use_regex\":false},\"model\":{\"type\":\"" + model_type +
           "\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":\"\",\"end_of_word_suffix\":\"\","
           "\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":false,\"vocab\":" + v +
           ",\"merges\":[[\"a\",\"b\"],\"ab c\",[" + jstr(bl(' ')) + ",\"a\"]]}}";
  };
  try {
    Tokenizer t = Tokenizer::from_json(make("{\"type\":\"NFC\"}", "BPE"));
    CHECK(t.vocab_size() == 260 && t.token_id("<|end|>") == 259 && t.token_id("ab") == 256 && t.is_special(259));
    // " ab": (a,b) has rank 0 < rank 2 of (G,a), so it becomes [G, ab]
    const std::vector<uint32_t> want = {257, 32, 256, 259, 97, 258};
    CHECK(t.encode("abc ab<|end|>a a") == want);
    CHECK(t.decode(want) == "abc ab<|end|>a a" && t.decode(want, true) == "abc aba a");
    CHECK(t.encode("") .empty() && t.eos_ids().empty());
    CHECK(throws([&] { t.encode("\xFF"); }));
    CHECK(throws([&] { t.decode(std::vector<uint32_t>{260}); }));
    CHECK(t.turn_end_ids().empty());
    // GLM chat-turn markers: only special added tokens count
    auto added = [](int id, const char* c, bool special) {
      return std::string(",{\"id\":") + std::to_string(id) + ",\"content\":\"" + c +
             "\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":" +
             (special ? "true" : "false") + "}";
    };
    Tokenizer g = Tokenizer::from_json(make("{\"type\":\"NFC\"}", "BPE",
                                            added(260, "<|user|>", true) + added(261, "<|observation|>", false)));
    CHECK(g.turn_end_ids() == std::vector<uint32_t>{260} && g.eos_ids().empty());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tokenizer test: %s\n", e.what());
    CHECK(false);
  }
  CHECK(throws([&] { Tokenizer::from_json(make("{\"type\":\"NFKC\"}", "BPE")); }));
  CHECK(throws([&] { Tokenizer::from_json(make("null", "WordPiece")); }));
  CHECK(throws([&] { resolve_tokenizer_path("Nobody/none@0123;tokenizer.json:sha256:xyz"); }));
}

// Sampler: greedy cases, filters, reproducibility, and empirical frequencies vs the softmax (chi-square-free bound).
// ------------------------------------------------------------------------------ request parameters (audit F06)

static void test_request_params() {
  auto J = [](const char* t) { return Json::parse(t); };
  // lengths: positive integers only, checked before narrowing
  CHECK(req::length(J("1"), "max_tokens") == 1);
  CHECK(req::length(J("256.0"), "max_tokens") == 256);
  CHECK(req::length(J("2147483647"), "max_tokens") == 2147483647u);
  for (const char* bad : {"-1", "0", "1.5", "-0.5", "2147483648", "1e30", "18446744073709551615", "\"8\"", "true"})
    CHECK(throws([&] { req::length(J(bad), "max_tokens"); }));
  bool named = false;
  try { req::length(J("-1"), "max_tokens"); } catch (const std::invalid_argument& e) {
    named = std::string(e.what()).find("max_tokens") != std::string::npos;
  }
  CHECK(named);
  // sampling fields: finite and in range; documented sentinels kept
  CHECK(req::number(J("0.7"), "temperature", 0.0, 1e300) == 0.7);
  CHECK(throws([&] { req::number(J("-0.1"), "temperature", 0.0, 1e300); }));
  CHECK(throws([&] { req::number(J("1e999"), "temperature", 0.0, 1e300); }));
  CHECK(throws([&] { req::number(J("1.5"), "top_p", 0.0, 1.0); }));
  CHECK(req::number(J("1"), "top_p", 0.0, 1.0) == 1.0);
  CHECK(req::top_k(J("-1")) == 0 && req::top_k(J("0")) == 0 && req::top_k(J("40")) == 40);
  CHECK(throws([&] { req::top_k(J("-2")); }) && throws([&] { req::top_k(J("2.5")); }));
  uint64_t sd = 7;
  CHECK(!req::seed(J("-1"), sd) && sd == 7);
  CHECK(req::seed(J("0"), sd) && sd == 0 && req::seed(J("123"), sd) && sd == 123);
  CHECK(throws([&] { req::seed(J("-2"), sd); }) && throws([&] { req::seed(J("0.5"), sd); }));
  // prompt token ids: no wrap-around (2^32 + 1 used to become 1), vocab bound inclusive-exclusive
  CHECK(req::token_id(J("0"), 100) == 0 && req::token_id(J("99"), 100) == 99);
  for (const char* bad : {"100", "-1", "4294967297", "4294967296", "1.0", "\"5\""})
    CHECK(throws([&] { req::token_id(J(bad), 100); }));
  CHECK(req::token_id(J("4294967295"), uint64_t(1) << 33) == 4294967295u);
  CHECK(throws([&] { req::token_id(J("4294967296"), uint64_t(1) << 33); }));
}

static void test_sampling() {
  const float lg[6] = {1.0f, 3.0f, 2.0f, 3.0f, -1.0f, 0.5f};
  suc::Sampler g({});   // temperature 0: first maximum
  CHECK(g.sample(lg, 6) == 1);
  suc::SamplerConfig c;
  c.temperature = 1.0;
  c.top_k = 1;
  CHECK(suc::Sampler(c).sample(lg, 6) == 1);
  // top_k = 2: only the two 3.0 logits (ids 1, 3) can be drawn
  c.top_k = 2;
  c.seed = 7;
  suc::Sampler s2(c);
  bool only = true;
  for (int i = 0; i < 2000; ++i) {
    const uint32_t t = s2.sample(lg, 6);
    only = only && (t == 1 || t == 3);
  }
  CHECK(only);
  // top_p = 0.5: the first of the descending order already reaches it? p(3.0) each ~0.38 -> ids {1, 3}
  c.top_k = 0;
  c.top_p = 0.5;
  suc::Sampler sp(c);
  only = true;
  for (int i = 0; i < 2000; ++i) {
    const uint32_t t = sp.sample(lg, 6);
    only = only && (t == 1 || t == 3);
  }
  CHECK(only);
  // full distribution: frequencies within 4 sigma of softmax(lg / 0.7)
  c.top_p = 1.0;
  c.temperature = 0.7;
  c.seed = 11;
  suc::Sampler sf(c);
  const int N = 200000;
  int cnt[6] = {0, 0, 0, 0, 0, 0};
  for (int i = 0; i < N; ++i) ++cnt[sf.sample(lg, 6)];
  double z = 0.0, p[6];
  for (int i = 0; i < 6; ++i) z += p[i] = std::exp(double(lg[i]) / 0.7);
  bool ok = true;
  for (int i = 0; i < 6; ++i) {
    const double q = p[i] / z, sd = std::sqrt(q * (1 - q) / N);
    ok = ok && std::fabs(double(cnt[i]) / N - q) < 4 * sd + 1e-9;
  }
  CHECK(ok);
  // same seed -> same draws
  suc::Sampler a(c), b(c);
  bool same = true;
  for (int i = 0; i < 100; ++i) same = same && a.sample(lg, 6) == b.sample(lg, 6);
  CHECK(same);
  c.legacy_order = true;
  suc::Sampler legacy_a(c), legacy_b(c);
  for (int i = 0; i < 100; ++i) CHECK(legacy_a.sample(lg, 6) == legacy_b.sample(lg, 6));
  c.legacy_order = false;
  // Extreme logits and a single-element vocabulary remain well-defined.
  const float extreme[] = {-10000, 10000, -10000};
  suc::Sampler extreme_sampler(c);
  CHECK(extreme_sampler.sample(extreme, 3) == 1);
  CHECK(extreme_sampler.sample(extreme, 1) == 0);
  // min_p = 0.5: tokens with p >= 0.5 * p_max: ids 1, 3 (equal max) and 2 (e^-1/0.7 = 0.24 < 0.5: dropped)
  c.min_p = 0.5;
  c.temperature = 1.0;
  suc::Sampler sm(c);
  only = true;
  for (int i = 0; i < 2000; ++i) {
    const uint32_t t = sm.sample(lg, 6);
    only = only && (t == 1 || t == 3);
  }
  CHECK(only);
}

int main() {
  test_sha256();
  test_unpack();
  test_numpy_random();
  test_half();
  test_json();
  test_hadamard();
  test_npy();
  test_container();
  test_mxfp4_decode();
  test_tokenizer();
  test_sampling();
  CHECK(suc::lookup_draft({1,2,3,4,1,2},4)==std::vector<uint32_t>({2,3,4,1}));
  CHECK(suc::lookup_draft({1,2,3},4)==std::vector<uint32_t>({3}));
  { suc::ParallelWorkers workers(4); std::vector<int> values(257);
    for(int pass=0;pass<20;++pass) { workers.run(values.size(),[&](std::size_t i){values[i]=int(i)+pass;}); CHECK(values[256]==256+pass); }
    bool caught=false;try { workers.run(17,[](std::size_t i){if(i==3)throw std::runtime_error("worker error");}); }catch(const std::runtime_error&){caught=true;}CHECK(caught);
    workers.run(values.size(),[&](std::size_t i){values[i]=42;});CHECK(values[256]==42);
  }
  CHECK(suc::prefill_chunk("qwen4_exp", 4096) == 64);
  CHECK(suc::prefill_chunk("qwen4_exp", 4096, 1024) == 1024);
  CHECK(suc::prefill_chunk("deepseek_v4", 8192) == 4096);
  CHECK(suc::prefill_chunk("deepseek_v4", 32768) == 1024);
  CHECK(suc::prefill_chunk("glm5_next", 512) == 512);
  CHECK(suc::prefill_chunk("qwen4_exp", 4096, 64) == 64);
  CHECK(suc::prefill_chunk("qwen4_exp", 13, 64) == 64);
  CHECK(suc::prefill_chunk("qwen4_exp", 4096, 0, true) == 16);
  CHECK(suc::prefill_chunk("qwen4_exp", 4096, 0, false, true, true) == 64);
  CHECK(suc::prefill_chunk("qwen4_exp", 4096, 0, true, false) == 1);
  CHECK(suc::service_prefill_chunk("deepseek_v4", 32768) == 64);
  CHECK(suc::service_prefill_chunk("glm5_next", 32768) == 512);
  CHECK(suc::service_prefill_chunk("glm5_next", 32768, 64) == 64);
  CHECK(suc::service_prefill_chunk("qwen4_exp", 32768) == 64);
  CHECK(suc::service_prefill_chunk("glm5_next", 32768, 0, true) == 16);
  CHECK(suc::service_prefill_chunk("glm5_next", 32768, 0, true, true, true) == 64);
  test_request_params();
  std::printf("%d checks, %d failures\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
