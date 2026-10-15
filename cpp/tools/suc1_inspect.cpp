// suc1-inspect: header, directory summary, per-tensor bits / bpw and checksum status of a SUC1 file.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <tuple>

#include "suc/sha256.hpp"
#include "suc/suc1.hpp"

using namespace suc;

static void usage() {
  std::fprintf(stderr,
               "usage: suc1-inspect FILE [--no-verify] [--codebook-hashes] [--tensors] [--filter SUBSTR]\n"
               "  --no-verify        skip the SHA-256 check of the whole file\n"
               "  --codebook-hashes  also recompute every codebook's dedup sha256\n"
               "  --tensors          print one line per tensor (bits, bpw, codec)\n"
               "  --filter SUBSTR    only list tensors whose name contains SUBSTR\n");
}

static std::string shape_str(const std::vector<uint64_t>& s) {
  std::string r = "(";
  for (std::size_t i = 0; i < s.size(); ++i) r += (i ? "," : "") + std::to_string(s[i]);
  return r + ")";
}

int main(int argc, char** argv) {
  std::string path, filter;
  OpenOptions opt;
  bool list = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--no-verify")) opt.verify_checksum = false;
    else if (!std::strcmp(argv[i], "--codebook-hashes")) opt.verify_codebook_hashes = true;
    else if (!std::strcmp(argv[i], "--tensors")) list = true;
    else if (!std::strcmp(argv[i], "--filter") && i + 1 < argc) { filter = argv[++i]; list = true; }
    else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) { usage(); return 0; }
    else if (path.empty() && argv[i][0] != '-') path = argv[i];
    else { usage(); return 2; }
  }
  if (path.empty()) { usage(); return 2; }
  try {
    auto t0 = std::chrono::steady_clock::now();
    Suc1File f = Suc1File::open(path, opt);
    double open_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const Header& h = f.header();
    std::printf("file             %s\n", path.c_str());
    std::printf("size             %llu bytes\n", (unsigned long long)f.file_size());
    std::printf("format_version   %u\n", h.format_version);
    std::printf("global_flags     0x%llx%s%s%s\n", (unsigned long long)h.global_flags,
                h.global_flags & kFlagCodebooks ? " CODEBOOKS" : "", h.global_flags & kFlagResidual ? " RESIDUAL" : "",
                h.global_flags & kFlagRotation ? " ROTATION" : "");
    std::printf("tensors          %llu\ncodebooks        %llu\n", (unsigned long long)h.number_of_tensors,
                (unsigned long long)h.number_of_codebooks);
    std::printf("directory        offset %llu, %llu bytes\n", (unsigned long long)h.directory_offset,
                (unsigned long long)h.directory_size);
    std::printf("payload_offset   %llu\nfooter_offset    %llu\n", (unsigned long long)h.payload_offset,
                (unsigned long long)h.footer_offset);
    std::printf("architecture_id  %s\ntokenizer_id     %s\n", f.architecture_id().c_str(), f.tokenizer_id().c_str());
    const auto& dg = f.footer().sha256;
    std::printf("sha256 (footer)  %s\n", to_hex(dg.data(), dg.size()).c_str());
    if (f.checksum_verified())
      std::printf("checksum         OK (verified in %.3f s)\n", f.checksum_seconds());
    else
      std::printf("checksum         not verified (--no-verify)\n");
    if (opt.verify_codebook_hashes) std::printf("codebook hashes  OK\n");
    std::printf("open time        %.3f s (incl. checksum)\n", open_s);

    // codebook ownership: each codebook's bytes are attributed to the first tensor (canonical order) using it
    std::vector<bool> owned(f.codebooks().size(), false);
    struct Row {
      uint64_t n = 0, params = 0, bytes = 0;
    };
    std::map<std::tuple<std::string, std::string, std::string>, Row> groups;
    uint64_t total_params = 0, codec_params = 0, codec_bytes = 0;
    if (list) std::printf("\n%-60s %-10s %-14s %-18s %5s %12s %8s %8s\n", "tensor", "repr", "residual/rot", "shape", "ibits",
                          "bytes", "bpw", "bpw+cb");
    for (const auto& t : f.tensors()) {
      uint64_t cb_bytes = 0;
      for (uint64_t id : t.codebook_ids)
        if (!owned[id]) {
          owned[id] = true;
          cb_bytes += f.codebooks()[id].nbytes;
        }
      const uint64_t bytes = t.compressed_size + cb_bytes;
      total_params += t.logical_size;
      if (t.representation_type != "raw") {
        codec_params += t.logical_size;
        codec_bytes += bytes;
      }
      Row& g = groups[{t.representation_type, t.residual_type, t.rotation}];
      ++g.n;
      g.params += t.logical_size;
      g.bytes += bytes;
      if (list && (filter.empty() || t.name.find(filter) != std::string::npos)) {
        const double n = double(t.logical_size ? t.logical_size : 1);
        std::printf("%-60s %-10s %-14s %-18s %5s %12llu %8.4f %8.4f\n", t.name.c_str(), t.representation_type.c_str(),
                    (t.residual_type + "/" + t.rotation).c_str(), shape_str(t.shape).c_str(),
                    t.index_bits ? std::to_string(*t.index_bits).c_str() : "-", (unsigned long long)bytes,
                    8.0 * double(t.compressed_size) / n, 8.0 * double(bytes) / n);
      }
    }
    std::printf("\n%-10s %-14s %-8s %6s %14s %14s %8s\n", "repr", "residual", "rotation", "count", "params", "bytes", "bpw");
    for (const auto& [k, g] : groups)
      std::printf("%-10s %-14s %-8s %6llu %14llu %14llu %8.4f\n", std::get<0>(k).c_str(), std::get<1>(k).c_str(),
                  std::get<2>(k).c_str(), (unsigned long long)g.n, (unsigned long long)g.params,
                  (unsigned long long)g.bytes, g.params ? 8.0 * double(g.bytes) / double(g.params) : 0.0);

    const ByteAccounting a = f.accounting();
    std::printf("\nbyte accounting (sums to the file size)\n");
    const std::pair<const char*, uint64_t> parts[] = {{"header", a.header},       {"directory", a.directory},
                                                      {"codebooks", a.codebooks}, {"indices", a.indices},
                                                      {"scales", a.scales},       {"residual", a.residual},
                                                      {"raw", a.raw},             {"padding", a.padding},
                                                      {"footer", a.footer}};
    uint64_t sum = 0;
    for (auto [name, v] : parts) {
      std::printf("  %-10s %14llu  %6.2f%%\n", name, (unsigned long long)v, 100.0 * double(v) / double(a.total));
      sum += v;
    }
    std::printf("  %-10s %14llu  (%s)\n", "total", (unsigned long long)a.total, sum == a.total ? "consistent" : "MISMATCH");
    std::printf("\nparams           %llu (codec tensors %llu)\n", (unsigned long long)total_params,
                (unsigned long long)codec_params);
    std::printf("file bpw         %.4f (8 * file size / all params)\n", 8.0 * double(a.total) / double(total_params));
    if (codec_params)
      std::printf("codec bpw        %.4f (codec tensors incl. owned codebooks, excl. raw tensors)\n",
                  8.0 * double(codec_bytes) / double(codec_params));
    return sum == a.total ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
