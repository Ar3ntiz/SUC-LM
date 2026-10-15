// suc1-decode: decode one, several or all tensors of a SUC1 file to float32 (.npy or raw little-endian).
#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "suc/decode.hpp"
#include "suc/npy.hpp"
#include "suc/parallel.hpp"
#include "suc/sha256.hpp"

using namespace suc;

static void usage() {
  std::fprintf(stderr,
               "usage: suc1-decode FILE (--all | --tensor NAME ...) [--out DIR] [--format npy|raw]\n"
               "                   [--no-verify] [--threads N] [--digest]\n"
               "  --out DIR      write DIR/<tensor name>.npy (or .f32 for raw); omit to only decode (timing)\n"
               "  --digest       print the sha256 of each decoded float32 buffer\n"
               "  --threads N    worker threads (default: SUC_THREADS or all cores); results do not depend on N\n");
}

int main(int argc, char** argv) {
  std::string path, out_dir, format = "npy";
  std::vector<std::string> names;
  bool all = false, digest = false;
  OpenOptions opt;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* flag) { return !std::strcmp(argv[i], flag) && i + 1 < argc; };
    if (!std::strcmp(argv[i], "--all")) all = true;
    else if (arg("--tensor")) names.push_back(argv[++i]);
    else if (arg("--out")) out_dir = argv[++i];
    else if (arg("--format")) format = argv[++i];
    else if (arg("--threads")) set_num_threads(unsigned(std::atoi(argv[++i])));
    else if (!std::strcmp(argv[i], "--no-verify")) opt.verify_checksum = false;
    else if (!std::strcmp(argv[i], "--digest")) digest = true;
    else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) { usage(); return 0; }
    else if (path.empty() && argv[i][0] != '-') path = argv[i];
    else { usage(); return 2; }
  }
  if (path.empty() || (all == !names.empty()) || (format != "npy" && format != "raw")) { usage(); return 2; }
  try {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    Suc1File f = Suc1File::open(path, opt);
    const double open_s = std::chrono::duration<double>(clk::now() - t0).count();
    if (!out_dir.empty()) ::mkdir(out_dir.c_str(), 0755);
    std::vector<const TensorDesc*> todo;
    if (all)
      for (const auto& t : f.tensors()) todo.push_back(&t);
    else
      for (const auto& n : names) todo.push_back(&f.tensor(n));
    double decode_s = 0, write_s = 0;
    uint64_t elems = 0;
    for (const TensorDesc* t : todo) {
      auto t1 = clk::now();
      DecodedTensor d = decode_tensor(f, *t);
      auto t2 = clk::now();
      decode_s += std::chrono::duration<double>(t2 - t1).count();
      elems += d.data.size();
      if (digest) {
        auto h = sha256(d.data.data(), d.data.size() * sizeof(float));
        std::printf("%s %s\n", to_hex(h.data(), h.size()).c_str(), t->name.c_str());
      }
      if (!out_dir.empty()) {
        if (format == "npy") write_npy_f32(out_dir + "/" + t->name + ".npy", d.shape, d.data.data());
        else write_raw(out_dir + "/" + t->name + ".f32", d.data.data(), d.data.size() * sizeof(float));
        write_s += std::chrono::duration<double>(clk::now() - t2).count();
      }
    }
    std::fprintf(stderr,
                 "opened %s in %.3f s (checksum %s, %.3f s); decoded %zu tensors, %llu elements in %.3f s"
                 " (%u threads)%s\n",
                 path.c_str(), open_s, f.checksum_verified() ? "verified" : "skipped", f.checksum_seconds(), todo.size(),
                 (unsigned long long)elems, decode_s, num_threads(),
                 out_dir.empty() ? "" : (", wrote in " + std::to_string(write_s) + " s").c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
