#include "suc/suc1.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <utility>

#include "suc/sha256.hpp"

namespace suc {
namespace {

[[noreturn]] void fail(const std::string& path, const std::string& msg) { throw FormatError(path + ": " + msg); }

uint64_t rd64(const uint8_t* p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}

// Checked unsigned arithmetic on header / directory values (a malformed file must fail, never wrap around).
uint64_t mul_checked(uint64_t a, uint64_t b) {
  uint64_t r;
  if (__builtin_mul_overflow(a, b, &r)) throw FormatError("SUC1 directory: size product overflows 64 bits");
  return r;
}

uint64_t add_checked(uint64_t a, uint64_t b) {
  uint64_t r;
  if (__builtin_add_overflow(a, b, &r)) throw FormatError("SUC1 header: offset sum overflows 64 bits");
  return r;
}

uint64_t prod(const std::vector<uint64_t>& s) {
  uint64_t n = 1;
  for (uint64_t d : s) n = mul_checked(n, d);
  return n;
}

std::vector<uint64_t> shape_of(const Json& j) {
  std::vector<uint64_t> s;
  for (const auto& d : j.as_array()) s.push_back(d.as_uint());
  return s;
}

std::string str_or_empty(const Json& j) { return j.is_null() ? std::string() : j.as_string(); }

std::optional<uint64_t> opt_uint(const Json* j) {
  if (!j || j->is_null()) return std::nullopt;
  return j->as_uint();
}

}  // namespace

uint64_t ArrayRecord::logical_numel() const { return packed ? prod(logical_shape) : prod(shape); }

const ArrayRecord* TensorDesc::find_array(std::string_view key) const {
  for (const auto& a : arrays)
    if (a.key == key) return &a;
  return nullptr;
}

const ArrayRecord& TensorDesc::array(std::string_view key) const {
  if (const ArrayRecord* a = find_array(key)) return *a;
  throw FormatError(name + ": no payload '" + std::string(key) + "'");
}

Suc1File::Suc1File(Suc1File&& o) noexcept { *this = std::move(o); }

Suc1File& Suc1File::operator=(Suc1File&& o) noexcept {
  if (this != &o) {
    if (map_) munmap(const_cast<uint8_t*>(map_), size_);
    path_ = std::move(o.path_);
    map_ = std::exchange(o.map_, nullptr);
    size_ = std::exchange(o.size_, 0);
    header_ = o.header_;
    footer_ = o.footer_;
    checksum_verified_ = o.checksum_verified_;
    codebook_hashes_verified_ = o.codebook_hashes_verified_;
    checksum_seconds_ = o.checksum_seconds_;
    dir_ = std::move(o.dir_);
    architecture_id_ = std::move(o.architecture_id_);
    tokenizer_id_ = std::move(o.tokenizer_id_);
    tensors_ = std::move(o.tensors_);
    codebooks_ = std::move(o.codebooks_);
    by_name_ = std::move(o.by_name_);
  }
  return *this;
}

Suc1File::~Suc1File() {
  if (map_) munmap(const_cast<uint8_t*>(map_), size_);
}

Suc1File Suc1File::open(const std::string& path, const OpenOptions& opt) {
  Suc1File f;
  f.path_ = path;
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error(path + ": cannot open: " + std::strerror(errno));
  struct stat st {};
  if (fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error(path + ": fstat failed: " + std::strerror(errno));
  }
  f.size_ = uint64_t(st.st_size);
  if (f.size_ < kHeaderSize + kFooterSize) {
    ::close(fd);
    fail(path, "file too small for a SUC1 header and footer");
  }
  void* m = mmap(nullptr, f.size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (m == MAP_FAILED) throw std::runtime_error(path + ": mmap failed: " + std::strerror(errno));
  f.map_ = static_cast<const uint8_t*>(m);

  // 1. fixed header (struct <4sIQQQQQQQ)
  const uint8_t* h = f.map_;
  if (std::memcmp(h, "SUC1", 4) != 0) fail(path, "bad magic (expected \"SUC1\")");
  std::memcpy(&f.header_.format_version, h + 4, 4);
  f.header_.global_flags = rd64(h + 8);
  f.header_.number_of_tensors = rd64(h + 16);
  f.header_.number_of_codebooks = rd64(h + 24);
  f.header_.directory_offset = rd64(h + 32);
  f.header_.directory_size = rd64(h + 40);
  f.header_.payload_offset = rd64(h + 48);
  f.header_.footer_offset = rd64(h + 56);
  const Header& H = f.header_;
  if (H.format_version != kFormatVersion)
    fail(path, "unsupported format_version " + std::to_string(H.format_version) + " (reader supports 1)");
  const uint64_t known = kFlagCodebooks | kFlagResidual | kFlagRotation;
  if (H.global_flags & ~known) fail(path, "unknown global_flags bits");
  if (H.directory_offset != kHeaderSize) fail(path, "directory_offset must be 64");
  if (H.footer_offset > f.size_ || H.directory_size > f.size_ ||
      f.size_ != add_checked(H.footer_offset, kFooterSize) || H.footer_offset % kAlignment ||
      H.payload_offset % kAlignment || H.payload_offset < add_checked(H.directory_offset, H.directory_size) ||
      H.footer_offset < H.payload_offset)
    fail(path, "inconsistent layout (size " + std::to_string(f.size_) + ", footer at " +
                   std::to_string(H.footer_offset) + "); truncated?");

  // 2. footer (struct <4sIQ32s16x)
  const uint8_t* ft = f.map_ + H.footer_offset;
  if (std::memcmp(ft, "SUCF", 4) != 0) fail(path, "bad footer magic");
  std::memcpy(&f.footer_.checksum_algorithm, ft + 4, 4);
  f.footer_.footer_offset = rd64(ft + 8);
  std::memcpy(f.footer_.sha256.data(), ft + 16, 32);
  if (f.footer_.checksum_algorithm != kChecksumSha256 || f.footer_.footer_offset != H.footer_offset)
    fail(path, "bad footer (checksum algorithm or repeated offset)");

  // 3. checksum
  if (opt.verify_checksum) {
    auto t0 = std::chrono::steady_clock::now();
    bool ok = f.compute_checksum_matches();
    f.checksum_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) fail(path, "checksum mismatch");
    f.checksum_verified_ = true;
  }

  // 4. directory + full layout validation
  f.parse_directory(opt);
  f.validate_layout();
  return f;
}

bool Suc1File::compute_checksum_matches() const {
  Sha256 s;
  constexpr uint64_t chunk = 64ull << 20;
  for (uint64_t off = 0; off < header_.footer_offset; off += chunk)
    s.update(map_ + off, std::min(chunk, header_.footer_offset - off));
  return s.finish() == footer_.sha256;
}

std::span<const uint8_t> Suc1File::bytes(uint64_t offset, uint64_t nbytes) const {
  if (offset > size_ || nbytes > size_ - offset) fail(path_, "payload view out of bounds");
  return {map_ + offset, std::size_t(nbytes)};
}

void Suc1File::parse_directory(const OpenOptions& opt) {
  const std::string_view text(reinterpret_cast<const char*>(map_ + header_.directory_offset),
                              std::size_t(header_.directory_size));
  try {
    dir_ = std::make_unique<Json>(Json::parse(text));
    const Json& d = *dir_;
    if (d["format"].as_string() != "SUC1") fail(path_, "directory format is not \"SUC1\"");
    if (d["format_version"].as_uint() != header_.format_version || d["tensors"].size() != header_.number_of_tensors ||
        d["codebooks"].size() != header_.number_of_codebooks)
      fail(path_, "directory disagrees with the fixed header");
    if (d["alignment"].as_uint() != kAlignment || d["byte_order"].as_string() != "little")
      fail(path_, "directory alignment/byte_order not 64/little");
    if (d["global_flags"].as_uint() != header_.global_flags) fail(path_, "directory global_flags != header");
    static const char* kSections[] = {"codebooks", "indices", "scales", "residual", "raw"};
    const auto& secs = d["sections"].as_array();
    if (secs.size() != 5) fail(path_, "unexpected sections list");
    for (std::size_t i = 0; i < 5; ++i)
      if (secs[i].as_string() != kSections[i]) fail(path_, "unexpected sections list");
    architecture_id_ = d["architecture_id"].as_string();
    tokenizer_id_ = d["tokenizer_id"].as_string();

    for (const auto& c : d["codebooks"].as_array()) {
      CodebookDesc cb;
      cb.id = c["id"].as_uint();
      cb.entries = c["entries"].as_uint();
      cb.vector_length = c["vector_length"].as_uint();
      cb.dtype = parse_dtype(c["dtype"].as_string());
      cb.offset = c["offset"].as_uint();
      cb.nbytes = c["nbytes"].as_uint();
      cb.sha256 = c["sha256"].as_string();
      codebooks_.push_back(std::move(cb));
    }
    for (const auto& t : d["tensors"].as_array()) {
      TensorDesc td;
      td.name = t["name"].as_string();
      td.shape = shape_of(t["shape"]);
      td.layout = t["layout"].as_string();
      td.representation_type = t["representation_type"].as_string();
      td.codec = str_or_empty(t["codec"]);
      td.params = &t["params"];
      td.block_size = opt_uint(t.find("block_size"));
      td.index_bits = opt_uint(t.find("index_bits"));
      for (const auto& i : t["codebook_ids"].as_array()) td.codebook_ids.push_back(i.as_uint());
      td.codebook_key = str_or_empty(t["codebook_key"]);
      td.residual_type = t["residual_type"].as_string();
      td.rotation = t["rotation"].as_string();
      td.data_offset = t["data_offset"].as_uint();
      td.compressed_size = t["compressed_size"].as_uint();
      td.logical_size = t["logical_size"].as_uint();
      for (const auto& a : t["arrays"].as_array()) {
        ArrayRecord r;
        r.key = a["key"].as_string();
        r.part = a["part"].as_string();
        r.dtype = parse_dtype(a["dtype"].as_string());
        r.shape = shape_of(a["shape"]);
        r.offset = a["offset"].as_uint();
        r.nbytes = a["nbytes"].as_uint();
        if (const Json* pk = a.find("packed")) {
          r.packed = true;
          r.bits = unsigned((*pk)["bits"].as_uint());
          r.logical_dtype = parse_dtype((*pk)["logical_dtype"].as_string());
          r.logical_shape = shape_of((*pk)["logical_shape"]);
        }
        td.arrays.push_back(std::move(r));
      }
      by_name_.emplace(td.name, tensors_.size());
      tensors_.push_back(std::move(td));
    }
  } catch (const JsonError& e) {
    fail(path_, std::string("corrupt directory: ") + e.what());
  } catch (const std::invalid_argument& e) {
    fail(path_, std::string("corrupt directory: ") + e.what());
  }

  if (opt.verify_codebook_hashes) {
    for (const auto& c : codebooks_) {
      std::string key = "[\"" + std::string(dtype_name(c.dtype)) + "\",[" + std::to_string(c.entries) + "," +
                        std::to_string(c.vector_length) + "]]";
      Sha256 s;
      s.update(key.data(), key.size());
      auto p = payload(c);
      s.update(p.data(), p.size());
      auto dg = s.finish();
      if (to_hex(dg.data(), dg.size()) != c.sha256) fail(path_, "codebook " + std::to_string(c.id) + " sha256 mismatch");
    }
    codebook_hashes_verified_ = true;
  }
}

void Suc1File::validate_layout() const {
  struct Seg {
    uint64_t off, n;
    std::string what;
  };
  std::vector<Seg> segs;
  auto check_seg = [&](uint64_t off, uint64_t n, const std::string& what) {
    if (off % kAlignment) fail(path_, what + ": offset not 64-byte aligned");
    if (off < header_.payload_offset || off > header_.footer_offset || n > header_.footer_offset - off)
      fail(path_, what + ": payload outside the payload region");
    if (n) segs.push_back({off, n, what});   // zero-length segments share the next aligned offset
  };
  for (std::size_t i = 0; i < codebooks_.size(); ++i) {
    const auto& c = codebooks_[i];
    const std::string what = "codebook " + std::to_string(i);
    if (c.id != i) fail(path_, what + ": id is not its position");
    if (c.nbytes != mul_checked(mul_checked(c.entries, c.vector_length), dtype_size(c.dtype)))
      fail(path_, what + ": nbytes != K*G*itemsize");
    check_seg(c.offset, c.nbytes, what);
  }
  uint64_t flags = 0;
  for (std::size_t i = 0; i < tensors_.size(); ++i) {
    const auto& t = tensors_[i];
    if (i && !(tensors_[i - 1].name < t.name)) fail(path_, "tensors not in canonical (sorted, unique) order at " + t.name);
    if (t.arrays.empty()) fail(path_, t.name + ": no payload arrays");
    uint64_t first = UINT64_MAX, total = 0;
    for (const auto& a : t.arrays) {
      const std::string what = t.name + "/" + a.key;
      if (a.packed) {
        if (a.bits < 1 || a.bits > 32) fail(path_, what + ": packed bits outside [1, 32]");
        if (a.dtype != DType::U32 || a.shape.size() != 2) fail(path_, what + ": packed array must be u32 [rows, words]");
        const auto& ls = a.logical_shape;
        const uint64_t n = ls.empty() ? 1 : ls.back();
        uint64_t rows = 1;
        if (ls.size() > 1)
          for (std::size_t k = 0; k + 1 < ls.size(); ++k) rows = mul_checked(rows, ls[k]);
        if (a.shape[0] != rows || a.shape[1] != words_per_row(n, a.bits))
          fail(path_, what + ": packed shape disagrees with logical shape and bits");
      }
      if (a.nbytes != mul_checked(prod(a.shape), dtype_size(a.dtype)))
        fail(path_, what + ": nbytes != prod(shape)*itemsize");
      check_seg(a.offset, a.nbytes, what);
      first = std::min(first, a.offset);
      total = add_checked(total, a.nbytes);
    }
    if (t.data_offset != first) fail(path_, t.name + ": data_offset != smallest array offset");
    if (t.compressed_size != total) fail(path_, t.name + ": compressed_size != sum of array nbytes");
    for (uint64_t id : t.codebook_ids)
      if (id >= codebooks_.size()) fail(path_, t.name + ": codebook id out of range");
    if (!t.codebook_ids.empty()) flags |= kFlagCodebooks;
    if (t.residual_type != "none") flags |= kFlagResidual;
    if (t.rotation != "none") flags |= kFlagRotation;
  }
  if (flags != header_.global_flags) fail(path_, "global_flags disagree with the tensor descriptors");
  std::sort(segs.begin(), segs.end(), [](const Seg& a, const Seg& b) { return a.off < b.off; });
  for (std::size_t i = 1; i < segs.size(); ++i)
    if (segs[i - 1].off + segs[i - 1].n > segs[i].off) fail(path_, "overlapping payloads: " + segs[i - 1].what + " / " + segs[i].what);
}

const TensorDesc* Suc1File::find_tensor(std::string_view name) const {
  auto it = by_name_.find(std::string(name));
  return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

const TensorDesc& Suc1File::tensor(std::string_view name) const {
  if (const TensorDesc* t = find_tensor(name)) return *t;
  throw std::out_of_range("unknown tensor '" + std::string(name) + "'");
}

ByteAccounting Suc1File::accounting() const {
  ByteAccounting a;
  a.header = kHeaderSize;
  a.directory = header_.directory_size;
  a.footer = kFooterSize;
  std::vector<std::pair<uint64_t, uint64_t>> segs{{kHeaderSize, header_.directory_size}};
  for (const auto& c : codebooks_) {
    a.codebooks += c.nbytes;
    segs.emplace_back(c.offset, c.nbytes);
  }
  for (const auto& t : tensors_)
    for (const auto& r : t.arrays) {
      if (r.part == "indices") a.indices += r.nbytes;
      else if (r.part == "scales") a.scales += r.nbytes;
      else if (r.part == "residual") a.residual += r.nbytes;
      else a.raw += r.nbytes;
      segs.emplace_back(r.offset, r.nbytes);
    }
  std::sort(segs.begin(), segs.end());
  uint64_t pad = 0, pos = kHeaderSize;
  for (auto [off, n] : segs) {
    pad += off - pos;
    pos = off + n;
  }
  a.padding = pad + header_.footer_offset - pos;
  a.total = size_;
  return a;
}

}  // namespace suc
