// SUC1 container reader (byte spec: docs/format.md; Python reference: suc/io/suc1.py).
//
// Suc1File memory-maps the file read-only, validates the 64-byte header, the footer and (unless
// disabled) the SHA-256 checksum, parses the canonical-JSON directory and exposes tensor/codebook
// descriptors plus zero-copy views of every payload segment.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "suc/bits.hpp"
#include "suc/json.hpp"

namespace suc {

class FormatError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline constexpr uint32_t kFormatVersion = 1;
inline constexpr uint64_t kAlignment = 64;
inline constexpr uint64_t kHeaderSize = 64, kFooterSize = 64;
inline constexpr uint64_t kFlagCodebooks = 1, kFlagResidual = 2, kFlagRotation = 4;
inline constexpr uint32_t kChecksumSha256 = 1;

struct Header {
  uint32_t format_version = 0;
  uint64_t global_flags = 0, number_of_tensors = 0, number_of_codebooks = 0;
  uint64_t directory_offset = 0, directory_size = 0, payload_offset = 0, footer_offset = 0;
};

struct Footer {
  uint32_t checksum_algorithm = 0;
  uint64_t footer_offset = 0;
  std::array<uint8_t, 32> sha256{};
};

struct ArrayRecord {
  std::string key, part;
  DType dtype = DType::U8;            // storage dtype (u32 for packed arrays)
  std::vector<uint64_t> shape;        // storage shape
  uint64_t offset = 0, nbytes = 0;
  bool packed = false;                // bit-packed (rows_u32le_bitstream)
  unsigned bits = 0;                  // packed: bits per element
  DType logical_dtype = DType::U8;    // packed: dtype of the unpacked values
  std::vector<uint64_t> logical_shape;

  uint64_t logical_numel() const;
};

struct CodebookDesc {
  uint64_t id = 0, entries = 0, vector_length = 0;
  DType dtype = DType::F16;
  uint64_t offset = 0, nbytes = 0;
  std::string sha256;
};

struct TensorDesc {
  std::string name;
  std::vector<uint64_t> shape;
  std::string layout, representation_type, codec, residual_type, rotation;   // codec "" when null
  const Json* params = nullptr;       // points into the owning Suc1File's directory
  std::optional<uint64_t> block_size, index_bits;
  std::vector<uint64_t> codebook_ids;
  std::string codebook_key;           // "" when null
  std::vector<ArrayRecord> arrays;
  uint64_t data_offset = 0, compressed_size = 0, logical_size = 0;

  const ArrayRecord* find_array(std::string_view key) const;
  const ArrayRecord& array(std::string_view key) const;   // throws FormatError when absent
  bool has_residual() const { return residual_type != "none"; }
};

struct OpenOptions {
  bool verify_checksum = true;        // stream every byte through SHA-256 (format.md §9 rule 3)
  bool verify_codebook_hashes = false; // also recompute each codebook's dedup sha256
};

// Byte accounting of the whole file (= SucModel.bits(), in bytes); the parts sum to the file size.
struct ByteAccounting {
  uint64_t header = 0, directory = 0, codebooks = 0, indices = 0, scales = 0, residual = 0, raw = 0,
           padding = 0, footer = 0, total = 0;
};

class Suc1File {
 public:
  static Suc1File open(const std::string& path, const OpenOptions& opt = {});

  Suc1File(Suc1File&&) noexcept;
  Suc1File& operator=(Suc1File&&) noexcept;
  Suc1File(const Suc1File&) = delete;
  Suc1File& operator=(const Suc1File&) = delete;
  ~Suc1File();

  const std::string& path() const { return path_; }
  uint64_t file_size() const { return size_; }
  const Header& header() const { return header_; }
  const Footer& footer() const { return footer_; }
  bool checksum_verified() const { return checksum_verified_; }
  bool codebook_hashes_verified() const { return codebook_hashes_verified_; }
  double checksum_seconds() const { return checksum_seconds_; }

  // Recomputes SHA-256 over [0, footer_offset) and compares with the footer.
  bool compute_checksum_matches() const;

  const Json& directory() const { return *dir_; }
  const std::string& architecture_id() const { return architecture_id_; }
  const std::string& tokenizer_id() const { return tokenizer_id_; }

  const std::vector<TensorDesc>& tensors() const { return tensors_; }
  const std::vector<CodebookDesc>& codebooks() const { return codebooks_; }
  const TensorDesc& tensor(std::string_view name) const;   // throws std::out_of_range
  const TensorDesc* find_tensor(std::string_view name) const;

  // Zero-copy views into the mapping.
  std::span<const uint8_t> bytes(uint64_t offset, uint64_t nbytes) const;
  std::span<const uint8_t> payload(const ArrayRecord& r) const { return bytes(r.offset, r.nbytes); }
  std::span<const uint8_t> payload(const CodebookDesc& c) const { return bytes(c.offset, c.nbytes); }

  ByteAccounting accounting() const;

 private:
  Suc1File() = default;
  void parse_directory(const OpenOptions& opt);
  void validate_layout() const;

  std::string path_;
  const uint8_t* map_ = nullptr;
  uint64_t size_ = 0;
  Header header_;
  Footer footer_;
  bool checksum_verified_ = false, codebook_hashes_verified_ = false;
  double checksum_seconds_ = 0.0;
  std::unique_ptr<Json> dir_;
  std::string architecture_id_, tokenizer_id_;
  std::vector<TensorDesc> tensors_;
  std::vector<CodebookDesc> codebooks_;
  std::unordered_map<std::string, std::size_t> by_name_;
};

}  // namespace suc
