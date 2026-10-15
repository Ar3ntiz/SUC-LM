// Minimal JSON reader for the SUC1 directory (docs/format.md §5).
//
// The directory is canonical JSON written by Python's json.dumps(ensure_ascii=True, allow_nan=False),
// so this parser only needs strict RFC 8259 JSON. Integers that fit in int64 are kept exact (offsets,
// sizes, seeds); other numbers are parsed as double. Objects keep their key order.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace suc {

class JsonError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Json {
 public:
  enum class Type { Null, Bool, Int, Double, String, Array, Object };
  using Array = std::vector<Json>;
  using Object = std::vector<std::pair<std::string, Json>>;

  Json() = default;
  static Json parse(std::string_view text);

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_int() const { return type_ == Type::Int; }
  bool is_number() const { return type_ == Type::Int || type_ == Type::Double; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  bool as_bool() const;
  int64_t as_int() const;            // exact integer (throws on non-integers)
  uint64_t as_uint() const;          // exact non-negative integer
  double as_double() const;
  const std::string& as_string() const;
  const Array& as_array() const;
  const Object& as_object() const;

  // Object access: find() returns nullptr when absent, operator[] throws.
  const Json* find(std::string_view key) const;
  const Json& operator[](std::string_view key) const;
  const Json& at(std::size_t i) const { return as_array().at(i); }
  std::size_t size() const;

  // Compact (non-canonical) serialization, for diagnostics.
  std::string dump() const;

 private:
  friend class JsonParser;
  Type type_ = Type::Null;
  bool b_ = false;
  int64_t i_ = 0;
  double d_ = 0.0;
  std::string s_;
  Array a_;
  Object o_;
};

}  // namespace suc
