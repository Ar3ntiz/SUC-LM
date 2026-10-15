"""Trellis-coded quantization (TCQ) with a bitshift trellis — the low tier candidate of docs/low_tier_codec.md.

QTIP-style (Tseng et al. 2024) quantizer: a tail-biting sequence of T weights is coded by a walk on a bitshift trellis
with L-bit states. Every step consumes kv = k*V new bits (k bits per weight, V weights per step) and the state is
the last L bits of the stream, so step i's state is bits [(i+1)kv - L, (i+1)kv) of a circular stream of T/V * kv
bits. The V reconstructed values of a step are a fixed function c(state) of the state ("code"):

  1mad   V=1, pure compute: x = a*s + b (mod 2^32), value = sum of x's 4 bytes (≈ Gaussian by the CLT)
  3inst  V=1, pure compute: x = a*s + b (mod 2^32), x = (x & 0x8fff8fff) ^ 0x3b603b60, value = sum of its two fp16
  hyb    V=2, hashed LUT: h = s*s + s (mod 2^32), value = lut[(h >> (15 - Q)) & (2^Q - 1)] (Q bits), with the sign of
         entry 1 flipped by bit 15 of h. The LUT (2^Q x 2 fp16, 2 KB at Q=9) is fitted (Lloyd over the trellis
         assignments) or the fixed Gaussian one.

Values are normalised to zero mean / unit variance over all 2^L states and multiplied by a gain. Encoding is exact
Viterbi (min squared error) over all 2^L states; tail-biting uses QTIP's two-pass heuristic (a rotated pass fixes
the L - kv overlap bits, a second pass is constrained to them), so a sequence costs exactly T*k bits.

Codec pipeline (as VectorCodebook): optional randomized Hadamard rotation (input side, optionally output side) ->
row-RMS normalisation (fp16 row scales) -> rows cut into sequences of T weights along the input dim -> Viterbi.
Research layout: "chunks" (n_seq, T/V) uint8/uint16 holds the kv-bit chunk of every step; params["stream_bits"] = T*k
bits per sequence (the packed stream) is what Encoded.bits() counts, never the uint8 container. The compute codes'
gain is a constant of the code (in the 64-byte descriptor); the hyb LUT is counted as codebook bits.
"""
from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import numpy as np

from suc.quant.codecs import Encoded, _check_2d, _check_decode, _derive_seeds, _row_normalize
from suc.quant.hadamard import rht, rht_inverse, rotate_weight, supported_size, unrotate_weight

MASK32 = (1 << 32) - 1
_LCG = {"1mad": (34038481, 76625530), "3inst": (89226354, 64248484)}


# ------------------------------------------------------------------------------------------------ codes

def _unit(v: np.ndarray, mu: float | None = None, sd: float | None = None) -> np.ndarray:
    """(v - mu) / sd in float64, rounded to float32 (mu / sd default to v's mean and std over all states)."""
    v = v.astype(np.float64)
    mu = v.mean() if mu is None else mu
    sd = v.std() if sd is None else sd
    return ((v - mu) / sd).astype(np.float32)


# extra LCG constants for entries j >= 1 of a V > 1 compute code (odd multipliers, arbitrary increments)
_LCG_EXTRA = [(2654435761, 1013904223), (2246822519, 3266489917), (3266489917, 668265263)]


def code_lcg(kind: str, V: int) -> list[tuple[int, int]]:
    """LCG (a, b) of every entry j of a compute code (pair3: one 3inst word for both entries)."""
    if kind == "pair3":
        return [_LCG["3inst"]]
    return [(_LCG[kind] if j == 0 else _LCG_EXTRA[j - 1]) for j in range(V)]


def code_raw(kind: str, L: int, V: int, s: np.ndarray | None = None) -> np.ndarray:
    """(n, V) float64 raw (un-normalised) values of a compute code for states s (default: all 2^L states):
    1mad the byte sum of a*s+b, 3inst h.x + h.y of the masked word, pair3 (h.x + h.y, h.x - h.y) of one word.
    Exact in float64 (integers / sums of two fp16 values)."""
    s = np.arange(1 << L, dtype=np.uint64) if s is None else np.asarray(s).astype(np.uint64)
    if kind == "pair3" and V != 2:
        raise ValueError("pair3 has V=2")
    cols = []
    for j in range(V):
        a, b = code_lcg(kind, V)[0 if kind == "pair3" else j]
        x = (s * np.uint64(a) + np.uint64(b)) & np.uint64(MASK32)
        if kind == "1mad":
            cols.append(sum(((x >> np.uint64(8 * i)) & np.uint64(0xFF)) for i in range(4)).astype(np.float64))
        else:
            x = ((x & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)).astype(np.uint32)
            h = x.view(np.float16).reshape(-1, 2).astype(np.float64)
            cols.append(h[:, 0] - h[:, 1] if (kind == "pair3" and j == 1) else h[:, 0] + h[:, 1])
    return np.stack(cols, axis=1)


@lru_cache(maxsize=None)
def code_stats(kind: str, L: int, V: int = 1) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(mean, std) per entry of the raw code over all 2^L states (float64): stored in the tensor params
    (code_mu / code_sd), so every decoder (Python, SUC1 reference, C++) normalises with the same doubles."""
    r = code_raw(kind, L, V)
    return tuple(float(v) for v in r.mean(axis=0)), tuple(float(v) for v in r.std(axis=0))


@lru_cache(maxsize=None)
def compute_code(kind: str, L: int, V: int = 1) -> np.ndarray:
    """(2^L, V) unit-variance values of a pure-compute code for every state (entry j uses its own LCG)."""
    r = code_raw(kind, L, V)
    mu, sd = code_stats(kind, L, V)
    return np.stack([_unit(r[:, j], mu[j], sd[j]) for j in range(V)], axis=1)


def hyb_index(L: int, Q: int) -> tuple[np.ndarray, np.ndarray]:
    """(lut index (2^L,), sign flip of entry 1 (2^L,) bool) of the hyb hash for every state."""
    s = np.arange(1 << L, dtype=np.uint64)
    h = (s * s + s) & np.uint64(MASK32)
    idx = ((h >> np.uint64(15 - Q)) & np.uint64((1 << Q) - 1)).astype(np.int64)
    flip = ((h >> np.uint64(15)) & np.uint64(1)).astype(bool)
    return idx, flip


def hyb_values(lut: np.ndarray, L: int, Q: int) -> np.ndarray:
    idx, flip = hyb_index(L, Q)
    v = lut[idx].astype(np.float32).copy()
    v[flip, 1] *= -1
    return v


@lru_cache(maxsize=None)
def gaussian_lut(Q: int, seed: int = 0) -> np.ndarray:
    """(2^Q, 2) k-means codebook of a 2-D unit Gaussian (fp16-rounded): the fixed hyb LUT."""
    from suc.quant.kmeans import kmeans
    x = mx.random.normal((1 << 20, 2), key=mx.random.key(seed))
    c = kmeans(x, 1 << Q, iters=30, seed=seed).codebook
    return np.array(c.astype(mx.float16).astype(mx.float32))


# ------------------------------------------------------------------------------------------------ viterbi

def _dist(x: mx.array, vals: mx.array) -> mx.array:
    """x (B, V), vals (S, V) -> (B, S) squared error."""
    d = None
    for j in range(vals.shape[1]):
        t = mx.square(x[:, j:j + 1] - vals[None, :, j])
        d = t if d is None else d + t
    return d


# Trellis convention (the kernel's): the stream of a sequence is little-endian (bit q of the stream = bit q % 32 of
# word q // 32), step i's kv-bit chunk c_i sits at stream bits [i*kv, (i+1)*kv), and step i's state is the L-bit
# window ending there, read LSB-first: s_i = (stream >> ((i+1)*kv - L)) & (2^L - 1), circular. So a new chunk
# enters at the TOP of the state: s_i = (s_{i-1} >> kv) | (c_i << (L - kv)), c_i = s_i >> (L - kv), and the
# predecessors of s' are the P = 2^kv states (s' & (R - 1)) * P + t, R = 2^(L - kv) (contiguous in memory).
# Fused Viterbi steps. M bitshift steps close on blocks of P^M states: after M steps a state depends on its ancestor
# only through the ancestor's top L - M*kv bits (the block key q), so one threadgroup runs M steps of one block
# entirely in threadgroup memory (one global read + write of the costs per M steps instead of per step). Inside a
# block, slot = (new chunks) << ((M-j)*kv) | (old low bits) is itself a bitshift trellis on M*kv bits:
# pred(slot') = ((slot' & (2^((M-1)kv) - 1)) << kv) | t, and the global state at sub-step j is
# s = (slot >> ((M-j)kv)) << (L - j*kv) | q << ((M-j)kv) | (slot & (2^((M-j)kv) - 1)).
_STEP_SOURCE = """
    constexpr uint NT = 1u << (M * KV);              // states per block
    constexpr uint R = S >> KV;
    uint lt = thread_position_in_threadgroup.x;
    uint tid = lt % NT;                               // slot inside the block
    uint base = lt - tid;                             // this block's offset in cs[]
    uint q = threadgroup_position_in_grid.x * (TPG / NT) + lt / NT;
    uint b = threadgroup_position_in_grid.y;
    threadgroup float cs[TPG];
    cs[lt] = cost[(ulong)b * S + ((q << (M * KV)) | tid)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 1; j <= M; ++j) {
        uint pbase = base + ((tid & ((1u << ((M - 1) * KV)) - 1u)) << KV);
        float m = cs[pbase];
        uint bt = 0;
        for (uint t = 1; t < (1u << KV); ++t) {
            float v = cs[pbase + t];
            if (v < m) { m = v; bt = t; }
        }
        uint sh = (M - j) * KV;
        uint s = ((tid >> sh) << (L - j * KV)) | (q << sh) | (tid & ((1u << sh) - 1u));
        float val[V];
        if (CODE == 0) {                                  // 1mad, computed (no table reads)
            for (uint v = 0; v < V; v++) {
                uint z = LA[v] * s + LB[v];
                uint u2 = (z & 0x00FF00FFu) + ((z >> 8) & 0x00FF00FFu);
                val[v] = (float((u2 & 0xFFFFu) + (u2 >> 16)) - MU[v]) * CC[v];
            }
        } else if (CODE == 1) {                           // 3inst
            for (uint v = 0; v < V; v++) {
                half2 h = as_type<half2>(((LA[v] * s + LB[v]) & 0x8FFF8FFFu) ^ 0x3B603B60u);
                val[v] = (float(h.x) + float(h.y) - MU[v]) * CC[v];
            }
        } else if (CODE == 2) {                           // pair3 (V = 2)
            half2 h = as_type<half2>(((LA[0] * s + LB[0]) & 0x8FFF8FFFu) ^ 0x3B603B60u);
            val[0] = (float(h.x) + float(h.y) - MU[0]) * CC[0];
            val[V - 1] = (float(h.x) - float(h.y) - MU[V - 1]) * CC[V - 1];
        } else {                                          // table (hyb)
            for (uint v = 0; v < V; v++) val[v] = vals[s * V + v];
        }
        float d = 0.0f;
        for (uint v = 0; v < V; v++) {
            float e = x[(b * M + (j - 1)) * V + v] - val[v];
            d = fma(e, e, d);
        }
        if (M > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
        cs[lt] = m + d;
        // one writer per predecessor group (tid < W); block-major layout q * W + tid keeps the stores contiguous
        // (bp_index() in Python maps a state back to it)
        if ((tid >> ((M - 1) * KV)) == 0) bp[((ulong)(j - 1) * NB + b) * R + q * (1u << ((M - 1) * KV)) + tid] = (uchar)bt;
        if (M > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // coalesced store: the block's slot s goes to state (s << (L - M*KV)) | q, so consecutive blocks (q) of this
    // threadgroup are contiguous for a fixed slot -> thread lt stores slot lt / NBT of block lt % NBT
    constexpr uint NBT = TPG / NT;
    if (M == 1) threadgroup_barrier(mem_flags::mem_threadgroup);
    uint qi = lt % NBT, sl = lt / NBT;
    out[(ulong)b * S + ((sl << (L - M * KV)) | (threadgroup_position_in_grid.x * NBT + qi))] = cs[qi * NT + sl];
"""


_REG_SOURCE = """
    // one thread per block of NT = 2^(M*KV) states, all M sub-steps in registers (no threadgroup memory, no
    // barriers); loads contiguous, stores coalesced across threads (consecutive q), back pointers block-major
    constexpr uint NT = 1u << (M * KV);
    constexpr uint P = 1u << KV;
    constexpr uint W = NT / P;                        // predecessor groups per sub-step
    constexpr uint R = S >> KV;
    constexpr uint NQ = S / NT;
    uint q = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    if (q >= NQ) return;
    float c[NT];
    // costs may be stored as half relative to state 0 of the sequence (COST_HALF): the survivors spread over a few
    // units only, so half keeps ~1e-3 resolution and halves the global traffic
    const device CT* src = cost + (ulong)b * S + ((ulong)q << (M * KV));
    float ref = float(cost[(ulong)b * S]);
    if (!isfinite(ref)) ref = 0.0f;
    #pragma clang loop unroll(full)
    for (uint t = 0; t < NT; ++t) c[t] = float(src[t]) - (HALF ? ref : 0.0f);
    #pragma clang loop unroll(full)
    for (uint j = 1; j <= M; ++j) {
        float mg[W];
        #pragma clang loop unroll(full)
        for (uint g = 0; g < W; ++g) {
            float m = c[g * P];
            uint bt = 0;
            #pragma clang loop unroll(full)
            for (uint t = 1; t < P; ++t) {
                float v = c[g * P + t];
                if (v < m) { m = v; bt = t; }
            }
            mg[g] = m;
            bp[((ulong)(j - 1) * NB + b) * R + q * W + g] = (uchar)bt;
        }
        uint sh = (M - j) * KV;
        #pragma clang loop unroll(full)
        for (uint cn = 0; cn < P; ++cn) {
            #pragma clang loop unroll(full)
            for (uint g = 0; g < W; ++g) {
                uint tid = (cn << ((M - 1) * KV)) | g;
                uint s = ((tid >> sh) << (L - j * KV)) | (q << sh) | (tid & ((1u << sh) - 1u));
        float val[V];
                if (CODE == 0) {                                  // 1mad, computed (no table reads)
                    for (uint v = 0; v < V; v++) {
                        uint z = LA[v] * s + LB[v];
                        uint u2 = (z & 0x00FF00FFu) + ((z >> 8) & 0x00FF00FFu);
                        val[v] = (float((u2 & 0xFFFFu) + (u2 >> 16)) - MU[v]) * CC[v];
                    }
                } else if (CODE == 1) {                           // 3inst
                    for (uint v = 0; v < V; v++) {
                        half2 h = as_type<half2>(((LA[v] * s + LB[v]) & 0x8FFF8FFFu) ^ 0x3B603B60u);
                        val[v] = (float(h.x) + float(h.y) - MU[v]) * CC[v];
                    }
                } else if (CODE == 2) {                           // pair3 (V = 2)
                    half2 h = as_type<half2>(((LA[0] * s + LB[0]) & 0x8FFF8FFFu) ^ 0x3B603B60u);
                    val[0] = (float(h.x) + float(h.y) - MU[0]) * CC[0];
                    val[V - 1] = (float(h.x) - float(h.y) - MU[V - 1]) * CC[V - 1];
                } else {                                          // table (hyb)
                    for (uint v = 0; v < V; v++) val[v] = vals[s * V + v];
                }
                float d = 0.0f;
                for (uint v = 0; v < V; v++) {
                    float e = x[(b * M + (j - 1)) * V + v] - val[v];
                    d = fma(e, e, d);
                }
                c[tid] = mg[g] + d;
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint t = 0; t < NT; ++t) out[(ulong)b * S + ((t << (L - M * KV)) | q)] = CT(c[t]);
"""


_CODE_IDS = {"1mad": 0, "3inst": 1, "pair3": 2}


@lru_cache(maxsize=None)
def _step_kernel(code: str | None, L: int, V: int, gain: float, kind: str = "tg"):
    """Step kernel; compute codes evaluate the state values in registers (the kernel's affine form of
    suc.kernels.tcq_matmul.code_constants), code None reads the vals table (hyb)."""
    if code is None:
        consts = f"constant uint LA[{V}] = {{0}};\nconstant uint LB[{V}] = {{0}};\n" \
                 f"constant float MU[{V}] = {{0}};\nconstant float CC[{V}] = {{0}};\n"
    else:
        from suc.kernels.tcq_matmul import code_constants
        lcg, mu, c = code_constants(code, L, V, gain)
        consts = (f"constant uint LA[{len(lcg)}] = {{{', '.join(f'{a}u' for a, _ in lcg)}}};\n"
                  f"constant uint LB[{len(lcg)}] = {{{', '.join(f'{b}u' for _, b in lcg)}}};\n"
                  f"constant float MU[{V}] = {{{', '.join(f'{float(v)!r}f' for v in mu)}}};\n"
                  f"constant float CC[{V}] = {{{', '.join(f'{float(v)!r}f' for v in c)}}};\n")
    tag = "table" if code is None else f"{code}_{abs(hash((L, V, gain))) % 10**8}"
    return mx.fast.metal_kernel(name=f"suc_tcq_viterbi_{kind}_{tag}", input_names=["cost", "x", "vals"],
                                output_names=["out", "bp"], header=consts,
                                source=_REG_SOURCE if kind == "reg" else _STEP_SOURCE)


FUSE_BITS = 6      # M*kv <= FUSE_BITS steps fused per launch (reg kernel, Studio: 6 best; 8 spills registers at kv=2)


def _fuse(L: int, kv: int, max_bits: int | None = None) -> int:
    """Steps fused per launch: the largest M with M*kv <= min(max_bits, L), at least 1."""
    return max(1, min(FUSE_BITS if max_bits is None else max_bits, L) // kv)


COST_HALF = True       # reg kernel: path metrics stored as half relative to state 0 between launches (Studio: -13..-25 % encode time, MSE equal to 4 digits)
STEP_KERNEL = "reg"    # "reg": thread per block of 2^(M*kv) states in registers (default; 1.4-1.7x faster than "tg"
                       # on the Studio, identical walks); "tg": thread per state, blocks in threadgroup memory


def _steps(cost: mx.array, x: mx.array, vals: mx.array, L: int, kv: int,
           code: tuple[str, float] | None = None) -> tuple[mx.array, mx.array]:
    """M = x.shape[1] fused steps: cost (B, S), x (B, M, V) -> new cost (B, S), back pointers (M, B, R) uint8.
    code = (name, gain) of a compute code: values computed in the kernel instead of read from vals."""
    b, m, v = x.shape
    S = 1 << L
    nt = 1 << (m * kv)
    tpg = max(nt, min(256, S))
    kind = STEP_KERNEL
    kern = _step_kernel(None, L, v, 0.0, kind) if code is None else _step_kernel(code[0], L, v, float(code[1]), kind)
    grid, tg = ((S, b, 1), (tpg, 1, 1)) if kind == "tg" else ((S // nt, b, 1), (min(256, S // nt), 1, 1))
    return kern(inputs=[cost, x, vals], template=[("S", S), ("L", L), ("KV", kv), ("M", m), ("V", v),
                                                 ("NB", b), ("TPG", tpg),
                                                 ("CODE", 3 if code is None else _CODE_IDS[code[0]]),
                                                 ("CT", cost.dtype), ("HALF", cost.dtype == mx.float16)],
                          grid=grid, threadgroup=tg,
                          output_shapes=[(b, S), (m, b, S >> kv)], output_dtypes=[cost.dtype, mx.uint8])


def bp_index(r: np.ndarray, L: int, kv: int, m: int, j: int) -> np.ndarray:
    """Position of the back pointer of new state r (= s' & (R-1)) at sub-step j of an m-step launch: the kernel stores
    it at q * W + w (block q, writer slot w = (c_part << sh) | old, W = 2^((m-1)kv), sh = (m-j)kv)."""
    sh = (m - j) * kv
    old = r & ((1 << sh) - 1)
    q = (r >> sh) & ((1 << (L - m * kv)) - 1)
    cpart = r >> (L - j * kv)
    return q * (1 << ((m - 1) * kv)) + ((cpart << sh) | old)


_BACK_SOURCE = """
    // one thread per sequence: walk the back pointers from the last state (block-major layout per launch, bp_index)
    uint b = thread_position_in_grid.x;
    if (b >= NB) return;
    constexpr uint R = 1u << (L - KV);
    uint s = last[b];
    st[b * N + (N - 1)] = s;
    for (int i = N - 1; i > 0; --i) {
        uint r = s & (R - 1u);
        uint m = smj[2 * (i - 1)], j = smj[2 * (i - 1) + 1];
        uint sh = (m - j) * KV;
        uint old = r & ((1u << sh) - 1u);
        uint q = (r >> sh) & ((1u << (L - m * KV)) - 1u);
        uint cp = r >> (L - j * KV);
        uint idx = q * (1u << ((m - 1) * KV)) + ((cp << sh) | old);
        uint t = bp[((ulong)(i - 1) * NB + b) * R + idx];
        s = r * (1u << KV) + t;
        st[b * N + (i - 1)] = s;
    }
"""


@lru_cache(maxsize=None)
def _back_kernel():
    return mx.fast.metal_kernel(name="suc_tcq_viterbi_back", input_names=["bp", "last", "smj"],
                                output_names=["st"], source=_BACK_SOURCE)


def _backtrack_gpu(bps: list, last: mx.array, L: int, kv: int, n: int) -> mx.array:
    """States (B, n) uint32 from the per-launch back pointers (m_c, B, R) on the GPU (no host transfer)."""
    B = int(last.shape[0])
    smj = []
    for bp in bps:
        m = int(bp.shape[0])
        smj += [v for j in range(1, m + 1) for v in (m, j)]
    bp_all = mx.concatenate(bps, axis=0) if bps else mx.zeros((1, B, 1 << (L - kv)), mx.uint8)
    return _back_kernel()(inputs=[bp_all, last.astype(mx.uint32), mx.array(np.array(smj or [1, 1], np.uint32))],
                          template=[("L", L), ("KV", kv), ("N", n), ("NB", B)],
                          grid=(B, 1, 1), threadgroup=(min(256, B), 1, 1),
                          output_shapes=[(B, n)], output_dtypes=[mx.uint32])[0]


BACKTRACK_GPU = True


def viterbi(x: mx.array, vals: mx.array, L: int, kv: int, init_r: np.ndarray | None = None,
            final_r: np.ndarray | None = None, code: tuple[str, float] | None = None) -> np.ndarray:
    """Min squared-error state walk. x (B, n, V) targets, vals (2^L, V). init_r / final_r (B,): constrain the first
    state's low L - kv bits / the last state's high L - kv bits to these values. Returns states (B, n) int64."""
    B, n, _ = x.shape
    S, P = 1 << L, 1 << kv
    R = S >> kv
    if P > 256:
        raise ValueError("k*V > 8 not supported")
    inf = mx.array(np.float32(np.inf))
    vals = vals.astype(mx.float32)
    cost = _dist(x[:, 0], vals)
    sidx = mx.arange(S, dtype=mx.uint32)
    if init_r is not None:
        cost = mx.where((sidx[None] & (R - 1)) == mx.array(init_r.astype(np.uint32))[:, None], cost, inf)
    bps = []
    mf = _fuse(L, kv)
    i = 1
    while i < n:
        m = min(mf, n - i)
        if COST_HALF and STEP_KERNEL == "reg" and cost.dtype != mx.float16:
            cost = (cost - mx.min(cost, axis=1, keepdims=True)).astype(mx.float16)   # relative (the kernel: to state 0)
        cost, bp = _steps(cost, x[:, i:i + m], vals, L, kv, code)
        bps.append(bp)
        i += m
        if len(bps) % 8 == 0:
            mx.eval(cost, bps[-8:])
    if final_r is not None:
        cost = mx.where((sidx[None] >> kv) == mx.array(final_r.astype(np.uint32))[:, None], cost.astype(mx.float32), inf)
    if BACKTRACK_GPU:
        st = _backtrack_gpu(bps, mx.argmin(cost, axis=1), L, kv, n)
        return np.array(st).astype(np.int64)
    last = np.array(mx.argmin(cost, axis=1)).astype(np.int64)
    states = np.empty((B, n), np.int64)
    states[:, -1] = last
    rows = np.arange(B)
    i = n - 1                     # backtrack launch by launch (bps[c] holds steps starts[c] .. starts[c] + m - 1)
    starts = np.cumsum([1] + [int(bp.shape[0]) for bp in bps[:-1]])
    for c in range(len(bps) - 1, -1, -1):
        bpc = np.array(bps[c])
        bps[c] = None
        m = bpc.shape[0]
        for j in range(m - 1, -1, -1):
            r = states[:, i] & (R - 1)
            states[:, i - 1] = r * P + bpc[j, rows, bp_index(r, L, kv, m, j + 1)].astype(np.int64)
            i -= 1
        assert i == starts[c] - 1
    return states


def viterbi_tailbiting(x: mx.array, vals: mx.array, L: int, kv: int, window: int = 64,
                       code: tuple[str, float] | None = None) -> np.ndarray:
    """Tail-biting walk (QTIP two-pass): states (B, n) with s_0 & (2^(L-kv) - 1) == s_{n-1} >> kv. The first pass
    only needs the state at the wrap point, so it runs on `window` steps on each side of it (Viterbi decisions
    settle within a few L/kv steps; window >= n/2 is the full rotated pass)."""
    n = x.shape[1]
    h = min(n // 2, window)
    rolled = mx.concatenate([x[:, n - h:], x[:, :h]], axis=1)      # original step n-1 sits at rolled step h-1
    s = viterbi(rolled, vals, L, kv, code=code)
    r = s[:, h - 1] >> kv
    return viterbi(x, vals, L, kv, init_r=r, final_r=r, code=code)


def states_to_chunks(states: np.ndarray, L: int, kv: int) -> np.ndarray:
    return (states >> (L - kv)).astype(np.uint8 if kv <= 8 else np.uint16)


def pack_stream(chunks: np.ndarray, kv: int) -> np.ndarray:
    """(n_seq, n) kv-bit chunks -> (n_seq, n*kv/32) uint32 little-endian streams (n*kv must be a multiple of 32)."""
    ns, n = chunks.shape
    if (n * kv) % 32:
        raise ValueError(f"sequence of {n * kv} bits is not a whole number of uint32 words")
    bits = ((chunks[:, :, None].astype(np.uint32) >> np.arange(kv, dtype=np.uint32)) & 1).reshape(ns, -1, 32)
    return (bits << np.arange(32, dtype=np.uint32)).sum(axis=2, dtype=np.uint64).astype(np.uint32)


def stream_states(words: np.ndarray, L: int, kv: int) -> np.ndarray:
    """Tail-biting state of every step read straight from the packed streams (n_seq, NW): the kernel's arithmetic
    (64-bit window from words (p >> 5, p >> 5 + 1) circular, p = (i+1)*kv - L mod N)."""
    ns, nw = words.shape
    N = nw * 32
    n = N // kv
    p = (np.arange(1, n + 1) * kv - L) % N
    w0, off = p >> 5, (p & 31).astype(np.uint64)
    lo = words[:, w0].astype(np.uint64)
    hi = words[:, (w0 + 1) % nw].astype(np.uint64)
    return (((lo | (hi << np.uint64(32))) >> off) & np.uint64((1 << L) - 1)).astype(np.int64)


def chunks_to_states(chunks: np.ndarray, L: int, kv: int) -> np.ndarray:
    """Tail-biting state of every step from the kv-bit chunks (n_seq, n): the L-bit window ending at chunk i."""
    c = chunks.astype(np.int64)
    s = np.zeros_like(c)
    for j in range(math.ceil(L / kv) + 1):
        sh = L - kv - j * kv
        cj = np.roll(c, j, axis=1)
        s |= (cj << sh) if sh >= 0 else (cj >> -sh)
    return s & ((1 << L) - 1)


# ------------------------------------------------------------------------------------------------ codec

_FACTOR_CACHE: dict = {}
_ROTATED_CACHE: list = []      # (hessian object, seed_in, damp, U) of the last few LDLQ factors


def _ldl_factor(h: np.ndarray, damp: float) -> mx.array:
    """Upper U with Hd^-1 = U^T U, Hd = H + damp*mean(diag) I, without an explicit inverse: with J the reversal,
    chol(J Hd J) = L gives Hd = Ut Ut^T (Ut = J L J upper), so U = Ut^-1 (a triangular inverse). float32 LAPACK on
    the CPU stream (12288^2: 5 s vs 30 s for float64 inv + chol, max rel. diff 6e-5); cached for the last few H
    (q/k/v and gate/up share their input second moment and rotation seed)."""
    key = (h.shape[0], damp, hash(h[:: max(1, h.shape[0] // 64), :: max(1, h.shape[0] // 64)].tobytes()),
           float(h.trace()))
    if key not in _FACTOR_CACHE:
        hd = h.astype(np.float32) + np.float32(damp * float(np.mean(np.diag(h)))) * np.eye(h.shape[0], dtype=np.float32)
        lt = mx.linalg.cholesky(mx.array(np.ascontiguousarray(hd[::-1, ::-1])), stream=mx.cpu)
        u = mx.linalg.tri_inv(lt[::-1, ::-1], upper=True, stream=mx.cpu)
        mx.eval(u)
        if not bool(mx.all(mx.isfinite(u)).item()):                         # float32 breakdown: float64 numpy
            u = mx.array(np.linalg.cholesky(np.linalg.inv(hd.astype(np.float64))).T.astype(np.float32))
        if len(_FACTOR_CACHE) >= 4:
            _FACTOR_CACHE.pop(next(iter(_FACTOR_CACHE)))
        _FACTOR_CACHE[key] = u
    return _FACTOR_CACHE[key]


class TrellisCodec:
    """Bitshift-trellis codec (module docstring). k bits per weight (k*V integer), sequences of T weights."""
    name = "tcq"

    def __init__(self, k: float = 2, L: int = 16, code: str = "hyb", V: int | None = None, Q: int = 9,
                 T: int = 256, rotate: bool = True, rotate_in: bool = True, rotate_out: bool = False,
                 lut: str = "fit", lut_iters: int = 3, fit_seqs: int = 2048, batch: int | None = None,
                 gain: float | None = None) -> None:
        self.V = V if V is not None else (2 if code == "hyb" else 1)
        kv = k * self.V
        if abs(kv - round(kv)) > 1e-9:
            raise ValueError(f"k*V must be an integer, got k={k} V={self.V}")
        if code not in ("1mad", "3inst", "pair3", "hyb"):
            raise ValueError(f"unknown trellis code {code!r}")
        if T % self.V or lut not in ("fit", "gaussian"):
            raise ValueError("bad T / lut")
        self.k, self.kv, self.L, self.code, self.Q, self.T = float(k), int(round(kv)), L, code, Q, T
        self.rotate, self.rotate_in, self.rotate_out = rotate, rotate_in, rotate_out
        # sequences per Viterbi batch: None = 2^24 / 2^L (256 at L=16; small trellises are launch-bound, so larger
        # batches; back pointers stay ~1 GB per batch)
        batch = batch if batch is not None else max(256, (1 << 24) >> L)
        self.lut, self.lut_iters, self.fit_seqs, self.batch = lut, lut_iters, fit_seqs, batch
        self.gain = gain if gain is not None else (1.0 if code == "hyb" else _DEFAULT_GAIN.get((code, self.kv), 0.95))

    # -- core: rows of prepared (rotated, row-normalised) values -> states
    def _encode_seqs(self, seqs: mx.array, vals: mx.array) -> np.ndarray:
        n = self.T // self.V
        out = []
        for i in range(0, seqs.shape[0], self.batch):
            x = seqs[i:i + self.batch].reshape(-1, n, self.V)
            out.append(viterbi_tailbiting(x, vals, self.L, self.kv,
                                          code=None if self.code == "hyb" else (self.code, self.gain)))
        return np.concatenate(out, axis=0)

    def _vals(self, lut: np.ndarray | None) -> np.ndarray:
        if self.code == "hyb":
            return hyb_values(lut, self.L, self.Q)
        return compute_code(self.code, self.L, self.V) * np.float32(self.gain)

    def fit_lut(self, seqs: mx.array, seed: int) -> np.ndarray | None:
        """Lloyd iterations of the hyb LUT on prepared sequences (N, T): assign by Viterbi, LUT entry = mean of the
        (sign-corrected) targets of the steps that use it. Returns the fp16-rounded LUT (2^Q, 2) float32."""
        if self.code != "hyb":
            return None
        lut = gaussian_lut(self.Q).copy()
        if self.lut == "gaussian":
            return lut
        rng = np.random.default_rng(seed)
        if seqs.shape[0] > self.fit_seqs:
            seqs = seqs[mx.array(np.sort(rng.choice(seqs.shape[0], self.fit_seqs, replace=False)))]
        idx, flip = hyb_index(self.L, self.Q)
        tgt = np.array(seqs).reshape(-1, 2)
        for _ in range(self.lut_iters):
            st = self._encode_seqs(seqs, mx.array(self._vals(lut))).reshape(-1)
            t = tgt.copy()
            t[flip[st], 1] *= -1
            q = idx[st]
            cnt = np.bincount(q, minlength=1 << self.Q)
            for j in range(2):
                s = np.bincount(q, weights=t[:, j], minlength=1 << self.Q)
                lut[:, j] = np.where(cnt > 0, s / np.maximum(cnt, 1), lut[:, j])
            lut = lut.astype(np.float16).astype(np.float32)
        return lut

    def _prepare(self, w: mx.array, seed: int, seed_in: int | None, seed_out: int | None):
        out, inn = _check_2d(w)
        if inn % self.T:
            raise ValueError(f"in_features={inn} not divisible by T={self.T}")
        x = w.astype(mx.float32)
        rin = self.rotate and self.rotate_in
        rout = self.rotate and self.rotate_out and supported_size(out)
        p = {"k": self.k, "kv": self.kv, "L": self.L, "code": self.code, "V": self.V, "Q": self.Q, "T": self.T,
             "gain": self.gain, "rotate": bool(rin or rout), "seed": seed, "rotate_in": bool(rin),
             "rotate_out": bool(rout), "index_bits": self.kv}
        if self.code != "hyb":     # normalisation of the compute code (float64): every decoder uses these numbers
            mu, sd = code_stats(self.code, self.L, self.V)
            p["code_mu"], p["code_sd"] = list(mu), list(sd)
        if rin or rout:
            d_in, d_out = _derive_seeds(seed, 2)
            p["seed_in"] = (d_in if seed_in is None else int(seed_in)) if rin else None
            p["seed_out"] = (d_out if seed_out is None else int(seed_out)) if rout else None
            if rin and rout:
                x = rotate_weight(x, p["seed_in"], p["seed_out"])
            elif rin:
                x = rht(x, p["seed_in"])
            else:
                x = rht(x.T, p["seed_out"]).T
        xn, s16 = _row_normalize(x)
        return (out, inn), p, xn.reshape(-1, self.T), s16

    def _ldlq_factor(self, hessian: mx.array, seed_in: int, damp: float) -> mx.array:
        """U of the rotated, damped hessian (_ldl_factor), cached per (hessian object, seed_in, damp): the experts of
        a stack share one hessian object and the stack's seed_in (gate / up of a MoE layer: 2 · E tensors)."""
        for h_ref, sd, dp, u in _ROTATED_CACHE:
            if h_ref is hessian and sd == seed_in and dp == damp:
                return u
        hr = rht(rht(hessian.astype(mx.float32), seed_in).T, seed_in)     # V H V^T
        u = _ldl_factor(np.array(hr), damp)
        if len(_ROTATED_CACHE) >= 4:
            _ROTATED_CACHE.pop(0)
        _ROTATED_CACHE.append((hessian, seed_in, damp, u))
        return u

    def _encode_ldlq(self, seqs: mx.array, out: int, vals: np.ndarray, u: mx.array) -> np.ndarray:
        """BlockLDLQ (QuIP# / QTIP; GPTQ's lazy block update with a jointly coded block): the column blocks of T
        inputs are coded in order, and the error of block b, E_b = (W_b - Q_b) U_bb^-1 with U the upper Cholesky
        factor of H^-1, is fed forward: W[:, later] -= E_b U_b,later. h is the input second moment in the ROTATED
        basis; the normalised rows only rescale W's rows, so the feedback is unchanged by them."""
        inn = u.shape[0]
        w = seqs.reshape(out, inn)
        T, n = self.T, self.T // self.V
        vm = mx.array(vals)
        states = []
        for b0 in range(0, inn, T):
            blk = w[:, b0:b0 + T]
            st = self._encode_seqs(blk, vm)                                  # (out, n)
            states.append(st)
            if b0 + T < inn:
                q = mx.array(vals[st].reshape(out, T))
                ub = u[b0:b0 + T, b0:b0 + T]
                e = mx.linalg.solve_triangular(ub.T, (blk - q).T, upper=False, stream=mx.cpu).T  # (W_b - Q_b) U_bb^-1
                w = mx.concatenate([w[:, :b0 + T], w[:, b0 + T:] - e @ u[b0:b0 + T, b0 + T:]], axis=1)
                mx.eval(w)
        return np.stack(states, axis=1).reshape(-1, n)

    def encode(self, w: mx.array, *, seed: int, col_weights: mx.array | None = None, seed_in: int | None = None,
               seed_out: int | None = None, lut: np.ndarray | None = None, hessian: mx.array | None = None,
               hessian_damp: float | None = None) -> Encoded:
        """hessian (in, in): E[x x^T] of the input in the ORIGINAL basis -> BlockLDLQ (input rotation required);
        hessian_damp default: self.hessian_damp (0.01; suc compress sets plan.ldlq_damp)."""
        if hessian_damp is None:
            hessian_damp = getattr(self, "hessian_damp", 0.01)
        shape, p, seqs, s16 = self._prepare(w, seed, seed_in, seed_out)
        if self.code == "hyb" and lut is None:
            lut = self.fit_lut(seqs, seed)
        if hessian is not None:
            if not p["rotate_in"] or p["rotate_out"]:
                raise ValueError("LDLQ needs the one-sided input rotation (rotate_in, not rotate_out)")
            u = self._ldlq_factor(hessian, p["seed_in"], float(hessian_damp))
            states = self._encode_ldlq(seqs, shape[0], self._vals(lut), u)
            p["ldlq"] = {"damp": hessian_damp}
        else:
            states = self._encode_seqs(seqs, mx.array(self._vals(lut)))
        arrays = {"chunks": mx.array(states_to_chunks(states, self.L, self.kv)), "row_scales": s16}
        if self.code == "hyb":
            arrays["lut"] = mx.array(lut).astype(mx.float16)
        p["stream_bits"] = int(round(self.k * shape[0] * shape[1]))       # T*k bits per tail-biting sequence
        return Encoded(self.name, shape, p, arrays)

    @staticmethod
    def _decode_values(p: dict, lut: mx.array | None = None) -> np.ndarray:
        """(2^L, V) float32 reconstruction values of every state: the stored code normalisation (code_mu / code_sd,
        float64) times the gain (float32), or the hyb LUT."""
        if p["code"] == "hyb":
            return hyb_values(np.array(lut.astype(mx.float32)), p["L"], p["Q"])
        if "code_mu" not in p:
            return compute_code(p["code"], p["L"], p["V"]) * np.float32(p["gain"])
        r = code_raw(p["code"], p["L"], p["V"])
        u = np.stack([_unit(r[:, j], p["code_mu"][j], p["code_sd"][j]) for j in range(p["V"])], axis=1)
        return u * np.float32(p["gain"])

    def bits_for(self, shape) -> int:
        """Encoded.bits()["total"] of an encoding of `shape` (out, in): data-independent (stream + fp16 row scales +
        hyb LUT + the metadata constant), so a candidate scored on a row sample gets the exact full-tensor bits."""
        from suc.quant.codecs import METADATA_BITS
        out, inn = int(shape[0]), int(shape[1])
        lut = (1 << self.Q) * 2 * 16 if self.code == "hyb" else 0
        return int(round(self.k * out * inn)) + 16 * out + lut + METADATA_BITS

    def decode(self, e: Encoded) -> mx.array:
        _check_decode(e, self.name)
        p = e.params
        out, inn = e.shape
        st = chunks_to_states(np.array(e.arrays["chunks"]), p["L"], p["kv"])
        vals = self._decode_values(p, e.arrays.get("lut"))
        x = mx.array(vals[st].reshape(out, inn)) * e.arrays["row_scales"].astype(mx.float32)[:, None]
        if not p["rotate"]:
            return x
        if p["rotate_in"] and p["rotate_out"]:
            return unrotate_weight(x, p["seed_in"], p["seed_out"])
        if p["rotate_in"]:
            return rht_inverse(x, p["seed_in"])
        return rht_inverse(x.T, p["seed_out"]).T


# unit-Gaussian MSE-optimal gains of the compute codes (scripts/low_tier_mse.py; 1.0 when not measured)
_DEFAULT_GAIN: dict[tuple[str, int], float] = {}
