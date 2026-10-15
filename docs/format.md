# SUC1 file format — byte-level specification (format_version 1)

Reference implementation: `suc/io/suc1.py` (`write_suc1`, `SucModel`). Tests: `tests/unit/test_suc1.py`.
`write_suc1_streamed` writes the same bytes with one tensor resident (spec D8): because the directory with every
absolute offset precedes the section-grouped payloads, it appends each tensor's segments to one temporary file per
section at section-relative aligned offsets, then writes header + directory and copies the sections behind them
(transient disk ≈ payload size). Byte identity with `write_suc1` is tested.
Requirements source: spec §16 (container), §17 (reference decoder), README §10.

## 1. Global conventions

| Property | Rule |
|---|---|
| Byte order | Little-endian for every integer, float and packed word, in the header, footer and all payloads. |
| Offsets / sizes | Unsigned 64-bit, **absolute** byte offsets from the start of the file. |
| Alignment | `ALIGNMENT = 64` bytes. The payload region, every payload segment and the footer start at a multiple of 64. Gaps are filled with zero bytes. |
| Magic | `"SUC1"` (4 bytes) at offset 0 is a fixed identifier and never changes for format evolution. Compatibility is governed solely by `format_version` (a new magic only for a fully incompatible redesign; decision 2026-09-30). |
| Version | `format_version = 1`. A reader rejects any other value. |
| Checksum | SHA-256 over bytes `[0, footer_offset)`, stored in the footer. |
| Determinism | The file is a pure function of its content: canonical tensor order, canonical JSON, no timestamps, zero padding. Writing the same content twice yields byte-identical files, independently of the insertion order of the input dict. |
| Memory mapping | All payloads are fixed-size, aligned, uncompressed little-endian arrays at known offsets: a reader can `mmap` the file and view any payload (e.g. `numpy.memmap`, `mx.array` from a memoryview) without reading the rest. |

## 2. File layout

```text
offset 0                 fixed header (64 bytes)
offset 64                directory (canonical JSON, UTF-8/ASCII, directory_size bytes)
                         zero padding to payload_offset
payload_offset           section "codebooks"   one segment per unique codebook
                         section "indices"     packed index payloads
                         section "scales"      row scales, affine scales/biases
                         section "residual"    residual positions, values, scale
                         section "raw"         uncompressed tensors (bf16/f16/f32/int ...)
footer_offset            footer (64 bytes)
EOF = footer_offset + 64
```

Sections appear in exactly this order. Within each section, segments follow the canonical tensor order
(§4) and, within one tensor, the sorted order of its array keys. Every segment starts at
`align64(end of previous segment)`; a zero-length segment still gets an aligned offset. There is no
separate entropy-table section in version 1 (entropy coding is not part of the stored representation yet;
a future version may add one under a new `format_version`).

## 3. Fixed header (64 bytes, struct `<4sIQQQQQQQ`)

| Offset | Type | Field | Meaning |
|---:|---|---|---|
| 0 | `char[4]` | magic | `"SUC1"` |
| 4 | `u32` | format_version | `1` |
| 8 | `u64` | global_flags | bit 0 `HAS_CODEBOOKS`, bit 1 `HAS_RESIDUAL`, bit 2 `HAS_ROTATION`; other bits must be 0 (reader rejects unknown bits) |
| 16 | `u64` | number_of_tensors | must equal `len(directory.tensors)` |
| 24 | `u64` | number_of_codebooks | must equal `len(directory.codebooks)` |
| 32 | `u64` | directory_offset | always `64` |
| 40 | `u64` | directory_size | exact JSON byte length (no padding included) |
| 48 | `u64` | payload_offset | first payload byte, multiple of 64, `>= 64 + directory_size` |
| 56 | `u64` | footer_offset | multiple of 64; file size must equal `footer_offset + 64` |

`architecture_id` and `tokenizer_id` (spec §16 header fields) live in the directory (§5) because they are
variable-length strings; the fixed header holds every fixed-width field.

## 4. Canonical order

Tensors are sorted by name in Unicode code-point order (Python `sorted(str)`, identical to byte order of
the UTF-8 encoding). Codebook ids are assigned `0, 1, 2, ...` in order of first appearance while walking
tensors in canonical order and, inside a tensor, its codebooks in index order `m = 0..M-1`.

## 5. Directory (canonical JSON)

Encoding: `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)`.
Floats use the shortest round-trip representation (Python `repr`). Top-level object:

| Key | Type | Meaning |
|---|---|---|
| `format` | str | `"SUC1"` |
| `format_version` | int | must equal the header value |
| `alignment` | int | `64` |
| `byte_order` | str | `"little"` |
| `architecture_id` | str | `architecture.model_type` (e.g. `"qwen3"`) |
| `architecture` | object | full architecture description (the Hugging Face `config.json`) |
| `tokenizer_id` | str | identifier of the tokenizer (e.g. `"sha256:<tokenizer.json hash>"`) |
| `metadata` | object | free user metadata (source model hash, codec config, ...) |
| `global_flags` | int | same value as the header |
| `sections` | list[str] | `["codebooks","indices","scales","residual","raw"]` (order of §2) |
| `codebooks` | list | codebook directory (§5.1), index = codebook id |
| `tensors` | list | tensor directory (§5.2), canonical order |

### 5.1 Codebook descriptor

| Key | Meaning |
|---|---|
| `id` | codebook id (= position in the list) |
| `entries` | K (number of code vectors) |
| `vector_length` | G (1 for scalar codebooks) |
| `dtype` | storage dtype (§6), `f16` for every current codec |
| `offset`, `nbytes` | payload: row-major `(entries, vector_length)` array |
| `sha256` | SHA-256 of `canonical_json([dtype, [K, G]]) ‖ payload bytes` — the deduplication key |

Deduplication: two codebooks with identical dtype, shape and bytes (e.g. tied or repeated weights) are stored
once and referenced by the same id.

### 5.2 Tensor descriptor

| Key | Meaning |
|---|---|
| `name` | tensor name (checkpoint namespace) |
| `shape` | logical shape of the decoded tensor |
| `layout` | `"rows_u32le_bitstream"` for codec tensors (index layout, §7; also `tcq`, §5.5), `"mxfp4_u8_nibbles"` for `mxfp4` tensors (§5.4), `"row_major"` for raw tensors |
| `representation_type` | `raw`, `bf16`, `affine`, `scalar`, `vq`, `additive`, `mxfp4`, `tcq` (base representation; residual is separate) |
| `codec` | codec name of the `Encoded` (e.g. `"vq+res0.005_fp16_magnitude"`), `null` for raw |
| `params` | the codec's complete parameter dict (`Encoded.params`), including seeds, `index_bits`, rotation seeds; for residual tensors `params.base` holds the base codec's params. Experts of a shared expert stack (`suc.quant.stacks`) add `codebook_shared: true`, `codebook_owner` (true for expert 0 only: the shared codebook is counted in its accounting) and `codebook_stack` (the stacked mlx-lm name) to the base params; every expert references the same deduplicated codebook ids |
| `block_size` | G for `vq`/`additive`, group size for `affine`, 32 (scale group) for `mxfp4`, T (sequence length) for `tcq`, 1 for `scalar`, `null` for `bf16`/`raw` |
| `codebook_ids` | ids of the tensor's codebooks: M entries for `vq` (M=1) / `additive`, one per row group for `scalar`; `[]` otherwise |
| `codebook_id` | `codebook_ids[0]` or `null` (spec §16 name; `codebook_ids` is the general form) |
| `codebook_key` | array key under which the codebooks are reassembled (`"codebooks"` or `"base.codebooks"`), `null` if none |
| `index_bits` | bits per stored index (`ceil(log2 K)`, affine: its bit width, `mxfp4`: 4, `tcq`: kv = k·V bits per trellis step), `null` for `bf16`/`raw` |
| `residual_type` | `none`, `sparse_fp16`, `sparse_int8`; `columns_fp16`, `columns_int8` for column-structured residuals (`params.structure = "columns"`, optional sparse tail with `params.n > 0`) |
| `rotation` | `none`, `in` (input-side randomized Hadamard only), `out` (output-side only: `params.rotate_in = false`), `in_out` (both sides). Informative: decoders follow `params.rotate`, `rotate_in`, `rotate_out` (files written before `out` existed label output-only tensors `in_out`) |
| `arrays` | the tensor's own payload segments (§5.3) — shared codebooks are not included |
| `data_offset` | smallest offset among `arrays` (the tensor's first payload) |
| `compressed_size` | sum of `nbytes` of `arrays` (excludes shared codebooks and padding) |
| `logical_size` | number of logical elements (parameters) of the tensor — an element count, not bytes |

### 5.3 Array record

| Key | Meaning |
|---|---|
| `key` | the `Encoded.arrays` key (`indices`, `row_scales`, `scales`, `biases`, `base.<key>`, `res_pos`, `res_val`, `res_scale`, `res_cols`, `res_colval`, `res_colscale`, `codes`, `chunks`) or `weight` for raw / bf16 tensors |
| `part` | accounting part = payload section: `indices` (also `codes`, `chunks`), `scales`, `residual`, `raw` |
| `dtype`, `shape` | storage dtype and shape of the bytes at `offset` (packed arrays: `u32`, `[rows, words_per_row]`) |
| `offset`, `nbytes` | absolute offset (multiple of 64) and exact byte length |
| `packed` | present for bit-packed arrays: `{"bits", "logical_dtype", "logical_shape"}` |

Bit-packed arrays: `indices` / `base.indices` at `index_bits`; `chunks` / `base.chunks` (tcq) at `params.kv`; `res_pos` at `params.position_bits`
(`ceil(log2 numel)`); `res_cols` at `params.column_index_bits` (`ceil(log2 in)`). Logical shapes (research layout of `suc.quant.codecs`):

| Representation | Arrays |
|---|---|
| `bf16` | `weight` bf16 `(out, in)` |
| `affine` | `indices` u8 `(out, in)`, `scales` f16 `(out, in/gs)`, `biases` f16 `(out, in/gs)` |
| `affine`, row-wise table (`params.scheme = "row_minmax"`) | lookup-table rows (qwen4_exp n-gram shards, written by `suc repack` or by `suc compress` with `ngram_codec: row_affine`, `suc.io.repack`): `group_size` = row width (e.g. 160, not an MLX group size), `bits` 2/4/8, `scales` / `biases` f16 (or bf16) `(rows, 1)`; same decode as `affine` |
| `scalar` | codebooks `(n_groups, K, 1)`, `indices` u8/u16 `(out, in)`, `row_scales` f16 `(out,)` |
| `vq` / `additive` | codebooks `(M, K, G)`, `indices` u8/u16 `(M, out, in/G)`, `row_scales` f16 `(out,)` |
| `mxfp4` | `codes` u8 `(out, in/2)` (section `indices`, not bit-packed: the bytes ARE the layout of §5.4), `scales` u8 `(out, in/32)` E8M0 exponents (section `scales`) |
| `tcq` | `chunks` u8 `(out · in/T, T/V)` kv-bit step chunks, bit-packed per row at kv bits (§5.5: the packed rows are the tail-biting bitstreams), `row_scales` f16 `(out,)` |
| residual (any base) | base arrays prefixed `base.`, `res_pos` u32 `(n,)` sorted unique flat row-major positions, `res_val` f16 or i8 `(n,)`, `res_scale` f16 `()` (int8 only) |
| column residual (`columns_*`) | as above (the sparse tail, `n` may be 0) plus `res_cols` u32 `(c,)` sorted unique input-column ids, `res_colval` f16 or i8 `(c, out)` (column-major: row j is the residual of input column `res_cols[j]`), `res_colscale` f16 `(c,)` (int8 only, one scale per column) |

### 5.4 Native MXFP4 pass-through (`representation_type` `mxfp4`)

For tensors that are MXFP4 at source (DeepSeek-V4 routed experts; `suc.quant.codecs.MXFP4Native`, compress
candidate `{kind: mxfp4}`), the checkpoint's bytes are stored **verbatim**: `codes` = the source `I8 [out, in/2]`
weight (two E2M1 codes per byte, element `2c` in the LOW nibble, `2c+1` in the high nibble), `scales` = the source
`F8_E8M0 [out, in/32]` tensor (one exponent per 32 consecutive inputs of a row). Codec `"mxfp4"`, params
`{"format": "mxfp4", "group_size": 32, "index_bits": 4, "element": "e2m1", "scale": "e8m0", "source": "verbatim"}`
(`"requantized"` when written from a BF16 weight that is exactly representable). No codebooks, rotation or residual.
Size: 4 + 8/32 = 4.25 bits per weight (+ directory). Viewed as little-endian u32 words, `codes` is exactly MLX's
`mode="mxfp4"` quantized layout (8 codes per word, element `j` at bits `4j`), so a runtime runs it without repacking.
Decode: `W[i, k] = E2M1(code) · 2^(S[i, k div 32] − 127)` with E2M1 code → value
`[0, .5, 1, 1.5, 2, 3, 4, 6]` for codes 0..7 and the negatives for 8..15, code 8 (negative zero) decoding to `+0`
(as the source dequantization `suc.io.quantized_source.dequant_mxfp4`). For exponents 2..251 every product is an
exact, normal BF16 value, so the decode is bit-identical to the D8 reference of the checkpoint. Readers that predate
this representation reject it as an unknown `representation_type` (format_version stays 1: the addition is
backward compatible for files without `mxfp4` tensors). The C++ runtime (`cpp/`, milestone 6) runs it in place
(`metal::Mxfp4Switch`, a port of MLX's `fp_gather_qmv`), alone or inside a mixed-codec expert stack.

### 5.5 Trellis-coded quantization (`representation_type` `tcq`)

Bitshift-trellis quantizer of `suc.quant.trellis` (docs/low_tier_codec.md), compute codes only (`params.code` ∈
`1mad`, `3inst`, `pair3`; the hashed-LUT code `hyb` is not a SUC1 representation). Every row of the stored weight
W' (rotated basis, `params.rotate_in`: W' = W Vᵀ with the randomized Hadamard of `params.seed_in`, as `vq`) is cut
into `in / T` sequences of `T = params.T` weights; a sequence is coded by `n = T / V` trellis steps of
`kv = params.kv = k·V` bits (`k = params.k` bits per weight, `V = params.V` weights per step, `L = params.L`-bit
states). Array `chunks` holds the kv-bit chunk c_i of every step, logical shape `(out · in/T, T/V)` (sequence-major,
row r's sequences at rows `r · in/T …`), bit-packed at kv bits per element (§7): each packed row is the
**little-endian tail-biting bitstream** of one sequence (`NW = T·k/32` u32 words, chunk i at stream bits
`[i·kv, (i+1)·kv)`), so the payload viewed as `(out, in/T · NW)` u32 is exactly the kernels' execution layout.

State of step i (circular inside its sequence): the L-bit window of the stream ending at chunk i,
`s_i = (stream >> ((i+1)·kv − L)) & (2^L − 1)` with bit positions modulo `T·k` (equivalently
`s_i = Σ_t c_{i−t} << (L − kv − t·kv)`, negative shifts as right shifts, indices modulo n).
Raw code value of entry j < V of state s (all arithmetic mod 2^32, `a, b` the entry's LCG constants):

| code | (a, b) of entry 0 / entry 1 | raw_j |
|---|---|---|
| `1mad` | (34038481, 76625530) / (2654435761, 1013904223) | x = a·s + b; sum of the 4 bytes of x |
| `3inst` | (89226354, 64248484) / (2654435761, 1013904223) | x = ((a·s + b) & 0x8FFF8FFF) ^ 0x3B603B60; h = x as two fp16 (low half first); h₀ + h₁ |
| `pair3` (V = 2) | (89226354, 64248484) for both | x as `3inst` of ONE word; raw₀ = h₀ + h₁, raw₁ = h₀ − h₁ |

raw is exact in float64. Normalisation: `params.code_mu[j]`, `params.code_sd[j]` (float64, the mean / standard
deviation of raw_j over all 2^L states, stored so every decoder uses the same numbers), `params.gain`:
value_j(s) = float32((raw_j − code_mu[j]) / code_sd[j]) · float32(gain) (float64 division, then float32).
Decode: `W'[r, q·T + i·V + j] = value_j(s_i of sequence q of row r) · row_scales[r]` (float32), then the inverse
rotation as `vq` (`rotate_in`: `rht_inverse(W', seed_in)`; tcq is always encoded with `rotate_out = false`).
Other params: `T`, `L`, `V`, `k`, `kv`, `index_bits` (= kv), `gain`, `seed`, `seed_in`, `rotate`, `rotate_in`,
`rotate_out`, `stream_bits` (= k · out · in), optional `ldlq` (`{"damp": …}`: encoded with BlockLDLQ; informative).
Size: k bits per weight + 16 per row (+ residual, metadata). Kernels (`suc.kernels.tcq_matmul`, the C++ runtime)
read the streams in place; they use the affine form value = (raw − μ_j)·c_j with μ_j = float32(code_mu[j]),
c_j = float32(gain / code_sd[j]) (kernel_consts). Readers that predate `tcq` reject it as an unknown
representation_type (format_version stays 1).

## 6. Dtypes

`bf16` (stored as its u16 bit pattern), `f16`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`,
`u32`, `u64`, `bool` (1 byte, 0/1). All little-endian, row-major (C order), no strides.

## 7. Packed index layout (`rows_u32le_bitstream`, = `suc.quant.packing`)

A logical index array of shape `(..., n)` is viewed as `rows = prod(shape[:-1])` rows of `n` indices
(a 1-D array is one row). Each row is an independent little-endian bit stream: element `j` occupies stream
bits `[j·b, (j+1)·b)`; stream bit `p` is bit `p mod 32` (LSB first) of 32-bit word `p div 32`. Elements may
straddle a word boundary. Each row is zero-padded to `words_per_row = ceil(n·b / 32)` words, so row `r`
starts at word `r · words_per_row`. For `b ∈ {8, 16, 32}` with `n·b` a multiple of 32 the payload equals the
plain little-endian array of the indices at that width.

## 8. Footer (64 bytes at footer_offset, struct `<4sIQ32s16x`)

| Offset | Type | Field |
|---:|---|---|
| 0 | `char[4]` | footer magic `"SUCF"` |
| 4 | `u32` | checksum algorithm: `1` = SHA-256 |
| 8 | `u64` | footer_offset (repeated, must match the header) |
| 16 | `u8[32]` | SHA-256 of bytes `[0, footer_offset)` |
| 48 | `u8[16]` | reserved, zero |

## 9. Reader rules

1. Read the 64-byte header; reject a wrong magic, `format_version != 1`, unknown flag bits, a file size
   other than `footer_offset + 64`, or misaligned `payload_offset` / `footer_offset`.
2. Read the footer; reject a wrong footer magic, algorithm or repeated offset.
3. Unless explicitly disabled (`verify=False`), stream the file and compare the SHA-256 (this reads every byte).
4. Parse the directory; reject it if its version or tensor/codebook counts disagree with the header.
5. Payloads are accessed lazily by offset (mmap); nothing else needs to be read.

## 10. Reference decoder (spec §17)

`SucModel.decode_tensor(name)` returns float32, computed with numpy from the unpacked payloads plus the
shared Hadamard primitive (`suc.quant.hadamard`), independent of the codec decoders:

The Hadamard transform of size n = m·2^k is H_m ⊗ H_{2^k} in float32: MLX's Sylvester transform on the 2^k
factor, then the H_m factor (m ∈ {12, 20, 28}, MLX's ±1 tables scaled by `float32(1/sqrt(m))`) as a sequential
fma chain `o = 0; for k = 0..m-1: o = fma(H[i, k], y[k, c], o)`. The result is device-independent (MLX GPU, MLX
CPU and the C++ decoder agree bit for bit).

| Representation | Decode |
|---|---|
| raw / `bf16` | value cast to float32 (bf16 by bit shift) |
| `mxfp4` | `E2M1(code) · 2^(S − 127)` per group of 32 inputs (§5.4), float32 |
| `affine` | `q · scale_g + bias_g` per group of `group_size` input columns |
| `scalar` | `C_{r div row_group}[i] · s_r` |
| `vq` / `additive` | `x = (Σ_m C_m[i_m]) · s_r` (sum in order m = 0..M-1, float32), then `rotation = in_out`: `Uᵀ x V` (`unrotate_weight(seed_in, seed_out)`), `in`: `x V` (`rht_inverse(seed_in)`), `out`: `Uᵀ x` (`rht_inverse(xᵀ, seed_out)ᵀ`) |
| `tcq` | `x = value(s) · s_r` (§5.5, float32), then the rotation as `vq` |
| residual | base decode, then (column residuals) `w[:, cols[j]] += colval[j] · colscale[j]` (float32), then `w[pos] += val` (int8: `val · res_scale`) |

The tests require `decode_tensor(name)` to be bit-identical to `codec.decode(encoded)` for every codec kind.

## 11. Size accounting

`SucModel.bits()` splits the whole file into `header`, `directory`, `codebooks`, `indices`, `scales`,
`residual`, `raw`, `padding`, `footer` (bits); the parts sum exactly to `8 × file size` (`total`). Compared with
`Encoded.bits()`, the file additionally pays word padding of packed rows, 64-byte alignment gaps and the
directory/header/footer instead of the nominal 64-byte per-tensor metadata constant, and stores residual
positions at `position_bits` (as `Encoded.bits()` assumes) but word-padded.
