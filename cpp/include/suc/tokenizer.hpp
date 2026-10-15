// Byte-level BPE tokenizer loaded from a Hugging Face `tokenizer.json` (CPU only, no dependencies).
//
// Implements exactly the pipeline HF `tokenizers` runs for the Qwen2/Qwen3 family, and fails loudly on any setting
// it does not implement:
//   1. added tokens are split out of the raw text first (leftmost-longest literal match; lstrip / rstrip /
//      single_word honoured; `normalized: true` tokens are matched on the normalized text instead);
//   2. normalizer: NFC (or none; also as a Sequence of at most one NFC);
//   3. pre-tokenizer: Sequence[Split(Regex, Isolated), ByteLevel(use_regex=false)], or a lone ByteLevel. The only
//      supported Split regexes are the Qwen2/Qwen3 one and its variants (Qwen3.5 / Qwen3.8: letter runs include
//      \p{M}; GLM-5.x: digit groups \p{N}{1,3} instead of single digits), implemented as one hand-written matcher,
//      and DeepSeek-V3 / V4's sequence of three Splits (digit groups of up to 3, CJK runs, then the punctuation /
//      letter / whitespace regex with \p{P} and \p{S});
//   4. model: BPE with merge ranks (HF's merge order, including its tie-breaking), ignore_merges supported,
//      no dropout / unk / byte_fallback / subword prefixes;
//   5. post-processor: ByteLevel (adds no tokens) or none;
//   6. decoder: ByteLevel. Runs of model tokens are mapped back to bytes and decoded as UTF-8 with the same
//      replacement rules as Rust's String::from_utf8_lossy; each added token is decoded on its own (so
//      "<|im_end|>" prints verbatim and splits the byte stream), as HF tokenizers does.
// Chat templates (tokenizer_config.json) are not implemented.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace suc {

class TokenizerError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct AddedToken {
  uint32_t id = 0;
  std::string content;
  bool special = false, lstrip = false, rstrip = false, normalized = false, single_word = false;
};

class Tokenizer {
 public:
  static Tokenizer from_file(const std::string& path);
  static Tokenizer from_json(std::string_view json_text);

  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;
  ~Tokenizer();

  // `utf8` must be valid UTF-8 (TokenizerError otherwise). add_special_tokens runs the post-processor, which for
  // ByteLevel adds nothing (Qwen3 has no BOS), so both settings give the same ids for Qwen3.
  std::vector<uint32_t> encode(std::string_view utf8, bool add_special_tokens = false) const;
  // Unknown ids throw TokenizerError. skip_special drops added tokens marked "special".
  std::string decode(std::span<const uint32_t> ids, bool skip_special = false) const;

  // Token string (added-token content, or the byte-level vocab string) -> id; -1 when absent.
  int64_t token_id(std::string_view token) const;
  // id -> token string as stored in tokenizer.json (byte-level form for model tokens).
  const std::string& id_to_token(uint32_t id) const;
  // Number of ids (max id + 1, added tokens included); ids without a token in between are rejected by decode.
  std::size_t vocab_size() const;
  const std::vector<AddedToken>& added_tokens() const;
  bool is_special(uint32_t id) const;
  // End-of-sequence ids: the special tokens <|endoftext|>, <|im_end|>, <|eot_id|>, </s>, <|end_of_text|>,
  // <｜end▁of▁sentence｜> that exist.
  std::vector<uint32_t> eos_ids() const;
  // GLM-style chat-turn markers that end an assistant turn (<|user|>, <|observation|>) when the tokenizer has them
  // as special tokens (GLM generation_config lists them as eos_token_id; SUC1 files without that list still stop).
  std::vector<uint32_t> turn_end_ids() const;

  // BPE cache statistics (words cached / lookups that hit).
  std::size_t cache_size() const;

  struct Impl;

 private:
  explicit Tokenizer(std::unique_ptr<Impl> p);
  std::unique_ptr<Impl> p_;
};

// Resolves a SUC1 `tokenizer_id` ("<hf id or local dir>@<revision>[;tokenizer.json:sha256:<hex>]") to the
// tokenizer.json path: a local directory id gives <dir>/tokenizer.json, otherwise the HF hub cache
// ($HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub) + models--<org>--<name>/snapshots/<rev>/
// (a non-hash revision such as "main" is looked up in refs/). When the id carries a sha256, the file's sha256 must
// match. Throws TokenizerError with the searched path on any failure.
std::string resolve_tokenizer_path(std::string_view tokenizer_id);

namespace tok_detail {
// Exposed for the unit tests.
uint32_t byte_to_unicode(uint8_t b);                       // GPT-2 bytes_to_unicode
std::vector<std::string> pre_tokenize(std::string_view utf8);   // Qwen regex split (no byte-level mapping)
std::string nfc(std::string_view utf8);                    // NFC with the tokenizer's Unicode tables
std::string utf8_lossy(std::string_view bytes);            // Rust String::from_utf8_lossy
bool utf8_valid(std::string_view bytes);
}  // namespace tok_detail

}  // namespace suc
