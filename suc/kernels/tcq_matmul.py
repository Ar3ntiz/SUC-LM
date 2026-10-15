"""Direct trellis-coded (TCQ) matmul y = x Ŵᵀ from the packed bitstreams (docs/low_tier_codec.md §3).

Representation (suc.quant.trellis, compute codes only): every row of Ŵ (in the rotated basis) is cut into sequences
of T weights; a sequence is a tail-biting little-endian bitstream of NW = T*k/32 uint32 words. Step i (V weights)
reads the L-bit window s_i = (stream >> ((i+1)*kv - L)) & (2^L - 1) (circular) and decodes it with the code:

  1mad   x = a_j*s + b_j (mod 2^32), v_j = (bytesum(x) - mu_j) * c_j       (one LCG per entry j)
  3inst  x = a_j*s + b_j, h = as_half2((x & 0x8fff8fff) ^ 0x3b603b60), v_j = (h.x + h.y - mu_j) * c_j
  pair3  x = a*s + b, h as for 3inst, (v_0, v_1) = ((h.x + h.y - mu_0) * c_0, (h.x - h.y - mu_1) * c_1)

c_j / mu_j fold the code's unit normalisation and gain; Ŵ[r] = s_r * v. The affine part is applied once per row:
y_r = s_r * Σ_j c_j (Σ_i x_i raw_ij - mu_j Σ_i x_i [i has entry j]), so the inner loop is raw code + one FMA.

Kernel "vec" (batch tiles of BT ≤ 8 rows): a simdgroup owns R output rows; per sequence its 32 lanes each take
T/32 consecutive weights (SPL = T/(32V) steps), read their x span once (shared by the R rows), load the few stream
words covering their windows (circular inside the sequence) and decode. No threadgroup memory, no LUT.
Ŵ is never materialised. Reference: tcq_matmul_reference (decode with the Python codec, then matmul).
"""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

from suc.quant.trellis import code_lcg, code_stats, compute_code

LANES = 32


def code_constants(code: str, L: int, V: int, gain: float, mu: list | None = None,
                   sd: list | None = None) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """(LCG (a, b) per entry, mu (V,), c (V,)) such that the codec value = (raw - mu) * c (float32): mu = float32(mean),
    c = float32(gain / std) from the code's float64 normalisation (mu / sd: the tensor's stored code_mu / code_sd,
    default suc.quant.trellis.code_stats)."""
    if mu is None or sd is None:
        mu, sd = code_stats(code, L, V)
    lcg = code_lcg(code, V)
    return lcg, np.array(mu, np.float64).astype(np.float32), (gain / np.array(sd, np.float64)).astype(np.float32)


def kernel_consts(p: dict) -> np.ndarray:
    """(4,) float32 [mu_0, c_0, mu_1, c_1] of a tcq tensor's params (the kernels' `cst` input; V <= 2)."""
    _, mu, c = code_constants(p["code"], p["L"], p["V"], float(p["gain"]), p.get("code_mu"), p.get("code_sd"))
    out = np.zeros(4, np.float32)
    for j in range(p["V"]):
        out[2 * j], out[2 * j + 1] = mu[j], c[j]
    return out


# Code constants of the kernels: the LCG (a, b) of entry j of code CODE (0 = 1mad, 1 = 3inst, 2 = pair3) as constexpr
# functions (folded at compile time), the raw value of a state, and the affine normalisation read from the `cst`
# input ([mu_0, c_0, mu_1, c_1] float32, kernel_consts): value = (raw - mu_j) * c_j.
_TCQ_HEADER = """
constexpr uint suc_tcq_a(int code, int j) { return j == 0 ? (code == 0 ? 34038481u : 89226354u) : 2654435761u; }
constexpr uint suc_tcq_b(int code, int j) { return j == 0 ? (code == 0 ? 76625530u : 64248484u) : 1013904223u; }
template <int CODE, int V>
inline void suc_tcq_raw(uint st, thread float* raw) {
    if (CODE == 0) {                        // 1mad: byte sum of a*s+b
        for (int j = 0; j < V; ++j) {
            uint z = suc_tcq_a(0, j) * st + suc_tcq_b(0, j);
            uint u2 = (z & 0x00FF00FFu) + ((z >> 8) & 0x00FF00FFu);
            raw[j] = float((u2 & 0xFFFFu) + (u2 >> 16));
        }
    } else if (CODE == 1) {                 // 3inst: h.x + h.y of the masked word
        for (int j = 0; j < V; ++j) {
            half2 h = as_type<half2>(((suc_tcq_a(1, j) * st + suc_tcq_b(1, j)) & 0x8FFF8FFFu) ^ 0x3B603B60u);
            raw[j] = float(h.x) + float(h.y);
        }
    } else {                                // pair3 (V = 2): (h.x + h.y, h.x - h.y) of ONE word
        half2 h = as_type<half2>(((suc_tcq_a(1, 0) * st + suc_tcq_b(1, 0)) & 0x8FFF8FFFu) ^ 0x3B603B60u);
        raw[0] = float(h.x) + float(h.y);
        raw[V - 1] = float(h.x) - float(h.y);
    }
}
"""

# Kernel "vec" (batch tiles BT < TCQ_MMA_MIN_BATCH, at most VEC_MAX_TILE rows): see the module docstring. The affine
# part is applied once per row: y = s_r * Σ_j c_j (Σ_i x_i raw_ij - mu_j Σ_{i: entry j} x_i).
_SRC = """
    constexpr uint MASKL = (1u << L) - 1u;
    constexpr uint N = NW * 32;                  // bits per sequence
    constexpr uint SPL = T / (V * 32);           // steps per lane
    constexpr uint WPL = T / 32;                 // weights per lane
    constexpr uint NSEQ = IN / T;
    // words a lane needs: its first window starts at bit p0 = (i0+1)*KV - L, its last one ends at (i0+SPL)*KV
    constexpr uint SPAN = (SPL - 1) * KV + L;
    constexpr uint NWL = (31 + SPAN + 31) / 32;  // words covering the lane's span at any bit offset

    uint lane = thread_index_in_simdgroup;
    uint row0 = (threadgroup_position_in_grid.x * SG + simdgroup_index_in_threadgroup) * R;
    if (row0 >= OUT) return;
    uint rows[R];
    for (int rr = 0; rr < R; ++rr) rows[rr] = min(row0 + rr, uint(OUT - 1));
    float acc[R][BT][V];
    for (int rr = 0; rr < R; ++rr) for (int b = 0; b < BT; ++b) for (int j = 0; j < V; ++j) acc[rr][b][j] = 0.0f;
    float sx[BT][V];
    for (int b = 0; b < BT; ++b) for (int j = 0; j < V; ++j) sx[b][j] = 0.0f;

    const uint i0 = lane * SPL;
    const int p0i = int((i0 + 1) * KV) - int(L);
    const uint p0 = uint(p0i + int(N)) % N;      // first window bit (circular)
    const uint w0 = p0 >> 5, off0 = p0 & 31;
    const uint w1 = w0 + 1 == NW ? 0u : w0 + 1;
    constexpr uint SPC = (32 - L) / KV + 1;      // steps whose windows fit one 32-bit register

    for (uint q = 0; q < NSEQ; ++q) {
        // x span of this lane (WPL consecutive inputs), shared by the R rows
        float xv[BT][WPL];
        for (int b = 0; b < BT; ++b) {
            for (uint e = 0; e < WPL; e += 4) {
                float4 v4 = suc_ld4(x + b * IN + q * T + lane * WPL + e);
                xv[b][e] = v4.x; xv[b][e + 1] = v4.y; xv[b][e + 2] = v4.z; xv[b][e + 3] = v4.w;
            }
            for (uint e = 0; e < WPL; ++e) sx[b][e % V] += xv[b][e];
        }
        for (int rr = 0; rr < R; ++rr) {
            const device uint* sw = words + (ulong(rows[rr]) * NSEQ + q) * NW;
            // The lane's windows are cut into chunks of SPC steps whose span fits one 32-bit register: one funnel
            // shift per chunk (row, sequence), then every window is a compile-time shift + mask of that register.
            uint wd[NWL];
            for (uint k2 = 0; k2 < NWL; ++k2) wd[k2] = sw[k2 == 0 ? w0 : (k2 == 1 ? w1 : (w0 + k2) % NW)];
            uint u = 0;
            #pragma clang loop unroll(full)
            for (uint t = 0; t < SPL; ++t) {
                if (t % SPC == 0) {
                    uint bit = off0 + t * KV;                // relative to word w0 (no wrap inside wd[])
                    uint wi = bit >> 5, o = bit & 31u;
                    uint hi = wd[min(wi + 1, NWL - 1)];
                    u = o == 0 ? wd[wi] : ((wd[wi] >> o) | (hi << (32u - o)));
                }
                uint st = (u >> ((t % SPC) * KV)) & MASKL;
                float raw[V];
                suc_tcq_raw<CODE, V>(st, raw);
                for (int b = 0; b < BT; ++b)
                    for (int j = 0; j < V; ++j) acc[rr][b][j] = fma(raw[j], xv[b][t * V + j], acc[rr][b][j]);
            }
        }
    }
    for (int b = 0; b < BT; ++b) for (int j = 0; j < V; ++j) sx[b][j] = simd_sum(sx[b][j]);
    for (int rr = 0; rr < R; ++rr) {
        uint row = row0 + rr;
        float s = float(scales[rows[rr]]);
        for (int b = 0; b < BT; ++b) {
            float tot = 0.0f;
            for (int j = 0; j < V; ++j) tot += cst[2 * j + 1] * (simd_sum(acc[rr][b][j]) - cst[2 * j] * sx[b][j]);
            if (lane == 0 && row < OUT) y[b * OUT + row] = static_cast<TO>(tot * s);
        }
    }
"""

# Kernel "mma" (batch tiles from TCQ_MMA_MIN_BATCH rows, at most MAX_BATCH_TILE): the register-fragment simdgroup-matrix
# layout of suc.kernels.vq_matmul._MMA_REG_SRC (Yᵀ = Ŵ · Xᵀ per simdgroup, RT 8-row tiles × NB 8-row batch tiles,
# lane ℓ owns fragment elements (fm, fn), (fm, fn + 1)). K is permuted inside blocks of 32 columns: lane group
# f = fn / 2 owns the SPAN = 8 consecutive weights f·8 .. f·8 + 7 of the block (8 / V trellis steps of one sequence,
# decoded with one funnel-shifted register like the vec kernel), k-step p uses its weights 2p, 2p + 1; the x fragment
# row fm reads physical column blk0 + (fm / 2)·8 + 2p + fm % 2, so the MMA sum is unchanged. Weights enter the MMA
# normalised, (raw - mu_j)·c_j rounded to TT; the row scale is applied at the store.
_MMA_SRC = """
    constexpr uint MASKL = (1u << L) - 1u;
    constexpr uint N = NW * 32;
    constexpr uint NSEQ = IN / T;
    constexpr uint SPAN = 8;                     // consecutive weights per lane and 32-column block
    constexpr uint SPS = SPAN / V;               // trellis steps per lane and block
    constexpr uint SPANB = (SPS - 1) * KV + L;   // bits covering those windows
    constexpr uint NWL = (31 + SPANB + 31) / 32;
    constexpr uint RW = 8 * RT;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint row0 = (threadgroup_position_in_grid.x * SG + sg) * RW;
    if (row0 >= OUT) return;
    uint qid = lane / 4;
    uint fm = (qid & 4) + (lane / 2) % 4;
    uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    uint rows[RT];
    bool rlive[RT];
    for (int rt = 0; rt < RT; ++rt) {
        uint r = row0 + rt * 8 + fm;
        rlive[rt] = r < OUT;
        rows[rt] = min(r, uint(OUT - 1));
    }
    float mu[V], cc[V];
    for (int j = 0; j < V; ++j) { mu[j] = cst[2 * j]; cc[j] = cst[2 * j + 1]; }
    simdgroup_matrix<float, 8, 8> acc[RT][NB];
    for (int rt = 0; rt < RT; ++rt)
        for (int nb = 0; nb < NB; ++nb) acc[rt][nb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint blk0 = 0; blk0 < IN; blk0 += 4 * SPAN) {
        uint col = blk0 + (fn / 2) * SPAN;       // first weight of this lane's span
        uint q = col / T, i0 = (col % T) / V;    // its sequence and first step
        int p0i = int((i0 + 1) * KV) - int(L);
        uint p0 = uint(p0i + int(N)) % N;
        uint w0 = p0 >> 5, off0 = p0 & 31u;
        float wv[RT][SPAN];
        for (int rt = 0; rt < RT; ++rt) {
            const device uint* sw = words + (ulong(rows[rt]) * NSEQ + q) * NW;
            uint wd[NWL + 1];
            for (uint k2 = 0; k2 < NWL; ++k2) wd[k2] = sw[(w0 + k2) % NW];
            wd[NWL] = 0u;
            #pragma clang loop unroll(full)
            for (uint t = 0; t < SPS; ++t) {
                uint bit = off0 + t * KV;
                uint wi = bit >> 5, o = bit & 31u;
                uint u = o == 0 ? wd[wi] : ((wd[wi] >> o) | (wd[wi + 1] << (32u - o)));
                float raw[V];
                suc_tcq_raw<CODE, V>(u & MASKL, raw);
                for (int j = 0; j < V; ++j) wv[rt][t * V + j] = (raw[j] - mu[j]) * cc[j];
            }
        }
        for (uint p = 0; p < SPAN / 2; ++p) {
            simdgroup_matrix<TT, 8, 8> bx[NB];
            uint pc = blk0 + (fm / 2) * SPAN + 2 * p + (fm % 2);
            for (int nb = 0; nb < NB; ++nb) {
                uint bb = nb * 8 + fn;
                bx[nb].thread_elements()[0] = bb < BT ? TT(x[bb * IN + pc]) : TT(0);
                bx[nb].thread_elements()[1] = bb + 1 < BT ? TT(x[(bb + 1) * IN + pc]) : TT(0);
            }
            for (int rt = 0; rt < RT; ++rt) {
                simdgroup_matrix<TT, 8, 8> aw;
                aw.thread_elements()[0] = TT(wv[rt][2 * p]);
                aw.thread_elements()[1] = TT(wv[rt][2 * p + 1]);
                for (int nb = 0; nb < NB; ++nb)
                    simdgroup_multiply_accumulate(acc[rt][nb], aw, bx[nb], acc[rt][nb]);
            }
        }
    }
    for (int rt = 0; rt < RT; ++rt) {
        if (!rlive[rt]) continue;
        uint r = rows[rt];
        float s = float(scales[r]);
        for (int nb = 0; nb < NB; ++nb) {
            uint b = nb * 8 + fn;
            float2 d = float2(acc[rt][nb].thread_elements()[0], acc[rt][nb].thread_elements()[1]);
            if (b < BT) y[b * OUT + r] = static_cast<TO>(d.x * s);
            if (b + 1 < BT) y[(b + 1) * OUT + r] = static_cast<TO>(d.y * s);
        }
    }
"""

# Expert stacks (suc.kernels.suc_switch): tile t = grid.y uses expert eidx[t] and rows x[t·BT : (t+1)·BT]; words
# (E, OUT, NSEQ·NW) and scales (E, OUT) are offset to the expert, the bodies above run unchanged.
_SWITCH_PROLOGUE = """
    {
        uint suc_t = threadgroup_position_in_grid.y;
        ulong suc_e = ulong(eidx[suc_t]);
        words += suc_e * ulong(OUT) * ulong((IN / T) * NW);
        scales += suc_e * ulong(OUT);
        x += ulong(suc_t) * ulong(BT * IN);
        y += ulong(suc_t) * ulong(BT * OUT);
    }
"""

LANES = 32
MAX_BATCH_TILE = 16          # batch rows per launch (as vq_matmul.MAX_BATCH_TILE)
TCQ_MMA_MIN_BATCH = 5        # tiles of at least this many rows run the mma kernel (C++: the same rule)
MMA_SG, MMA_RT = 4, 4        # simdgroups per threadgroup, 8-row tiles per simdgroup (128 rows per threadgroup)
VEC_SG, VEC_ROWS = 8, 4      # vec kernel: simdgroups, rows per simdgroup at BT <= 2 (VEC_ROWS / 2 above)


def _header() -> str:
    from suc.kernels.vq_matmul import _HEADER
    return _HEADER + _TCQ_HEADER


_IN = {"vec": ["x", "words", "scales", "cst"], "mma": ["x", "words", "scales", "cst"]}


@lru_cache(maxsize=None)
def _kernel(kind: str, switch: bool = False):
    body = (_SWITCH_PROLOGUE if switch else "") + (_SRC if kind == "vec" else _MMA_SRC)
    return mx.fast.metal_kernel(name=f"suc_tcq_{'switch_' if switch else ''}{kind}",
                                input_names=_IN[kind] + (["eidx"] if switch else []), output_names=["y"],
                                header=_header(), source=body)


_CODE_ID = {"1mad": 0, "3inst": 1, "pair3": 2}


def _consts(code: str, L: int, V: int, gain: float | None, cst) -> mx.array:
    if cst is not None:
        return mx.array(np.asarray(cst, np.float32)) if not isinstance(cst, mx.array) else cst.astype(mx.float32)
    return mx.array(kernel_consts({"code": code, "L": L, "V": V, "gain": 1.0 if gain is None else gain}))


def launch_plan(bt: int, path: str = "auto") -> tuple[str, list[tuple[str, int]], int, int]:
    """(kind, extra template args, simdgroups, rows per threadgroup) of one batch tile of bt rows (the C++ host,
    cpp/src/metal, applies the same rule)."""
    if path == "mma" or (path == "auto" and bt >= TCQ_MMA_MIN_BATCH):
        return "mma", [("SG", MMA_SG), ("RT", MMA_RT), ("NB", -(-bt // 8))], MMA_SG, 8 * MMA_RT
    r = VEC_ROWS if bt <= 2 else VEC_ROWS // 2
    return "vec", [("SG", VEC_SG), ("R", r)], VEC_SG, r


def _check(code: str, k: float, V: int, T: int, inn: int, out: int, words_shape) -> tuple[int, int]:
    if code not in _CODE_ID:
        raise ValueError(f"no kernel for code {code!r}")
    if V not in (1, 2):
        raise ValueError(f"tcq kernels support V in (1, 2), got {V}")
    kv = int(round(k * V))
    nw = int(round(T * k)) // 32
    if inn % T or T % (32 * V) or (T // 32) % 4 or tuple(words_shape) != (out, inn // T * nw):
        raise ValueError(f"bad shapes: in {inn}, words {tuple(words_shape)}, T={T}, V={V}, k={k}")
    return kv, nw


def tcq_matmul_metal(x: mx.array, words: mx.array, row_scales: mx.array, *, k: float, L: int, V: int, T: int,
                     code: str, gain: float | None = None, cst=None, path: str = "auto",
                     out_dtype: mx.Dtype | None = None) -> mx.array:
    """y (B, OUT) = x (B, IN) · Ŵᵀ with Ŵ given by packed streams words (OUT, IN/T * NW) uint32 (rotated basis:
    the caller rotates x), row_scales (OUT,) fp16 / float32. cst: kernel_consts of the tensor (default: from code,
    L, V, gain). Batch tiles of up to MAX_BATCH_TILE rows; vec kernel below TCQ_MMA_MIN_BATCH rows, mma from it."""
    if path not in ("auto", "vec", "mma"):
        raise ValueError(f"path must be 'auto', 'vec' or 'mma', got {path!r}")
    b, inn = x.shape
    out = row_scales.shape[0]
    kv, nw = _check(code, k, V, T, inn, out, words.shape)
    cst = _consts(code, L, V, gain, cst)
    out_dtype = out_dtype or x.dtype
    ys = []
    for s0 in range(0, b, MAX_BATCH_TILE):
        xb = x[s0:s0 + MAX_BATCH_TILE]
        bt = xb.shape[0]
        kind, extra, sgs, rows = launch_plan(bt, path)
        tmpl = [("L", L), ("V", V), ("KV", kv), ("T", T), ("NW", nw), ("IN", inn), ("OUT", out), ("BT", bt),
                ("CODE", _CODE_ID[code])] + extra + ([("TT", x.dtype)] if kind == "mma" else []) + [("TO", out_dtype)]
        groups = -(-out // (rows * sgs))
        ys.append(_kernel(kind)(inputs=[xb, words, row_scales, cst], template=tmpl,
                                grid=(groups * sgs * 32, 1, 1), threadgroup=(sgs * 32, 1, 1),
                                output_shapes=[(bt, out)], output_dtypes=[out_dtype])[0])
    return ys[0] if len(ys) == 1 else mx.concatenate(ys, axis=0)


def tcq_gather_matmul_metal(x: mx.array, words: mx.array, row_scales: mx.array, tile_experts: mx.array, *,
                            k: float, L: int, V: int, T: int, code: str, cst, tile: int = 1, path: str = "auto",
                            out_dtype: mx.Dtype | None = None) -> mx.array:
    """Expert-stack matmul: x (T·tile, in), tile t = rows [t·tile, (t+1)·tile) through expert tile_experts[t];
    words (E, OUT, IN/T·NW), row_scales (E, OUT) -> (T·tile, OUT). Same kernels as tcq_matmul_metal (+ the tile
    prologue), the kind chosen from `tile` as launch_plan."""
    if not 1 <= tile <= MAX_BATCH_TILE:
        raise ValueError(f"tile must be in 1..{MAX_BATCH_TILE}, got {tile}")
    if words.ndim != 3 or row_scales.ndim != 2 or words.shape[0] != row_scales.shape[0]:
        raise ValueError(f"expected words (E, OUT, W), scales (E, OUT); got {words.shape}, {row_scales.shape}")
    n_t = tile_experts.shape[0]
    if x.shape[0] != n_t * tile:
        raise ValueError(f"x has {x.shape[0]} rows, expected {n_t} tiles x {tile}")
    inn, out = x.shape[1], row_scales.shape[1]
    kv, nw = _check(code, k, V, T, inn, out, words.shape[1:])
    out_dtype = out_dtype or x.dtype
    kind, extra, sgs, rows = launch_plan(tile, path)
    tmpl = [("L", L), ("V", V), ("KV", kv), ("T", T), ("NW", nw), ("IN", inn), ("OUT", out), ("BT", tile),
            ("CODE", _CODE_ID[code])] + extra + ([("TT", x.dtype)] if kind == "mma" else []) + [("TO", out_dtype)]
    groups = -(-out // (rows * sgs))
    return _kernel(kind, True)(inputs=[x, words, row_scales, _consts(code, L, V, None, cst),
                                       tile_experts.astype(mx.uint32)],
                               template=tmpl, grid=(groups * sgs * 32, n_t, 1), threadgroup=(sgs * 32, 1, 1),
                               output_shapes=[(n_t * tile, out)], output_dtypes=[out_dtype])[0]


def pack_encoded(e) -> mx.array:
    """Execution layout of a tcq Encoded: (OUT, IN/T * NW) uint32 little-endian streams."""
    from suc.quant.trellis import pack_stream
    p = e.params
    ch = np.array(e.arrays["chunks"])
    return mx.array(pack_stream(ch, p["kv"]).reshape(e.shape[0], -1))


def tcq_matmul_reference(x: mx.array, e) -> mx.array:
    """x · Ŵᵀ in the ROTATED basis (Ŵ = row_scales * code values, no unrotation), float32."""
    from suc.quant.trellis import TrellisCodec, chunks_to_states
    p = e.params
    st = chunks_to_states(np.array(e.arrays["chunks"]), p["L"], p["kv"])
    vals = TrellisCodec._decode_values(p)
    w = mx.array(vals[st].reshape(e.shape)) * e.arrays["row_scales"].astype(mx.float32)[:, None]
    return x.astype(mx.float32) @ w.T
