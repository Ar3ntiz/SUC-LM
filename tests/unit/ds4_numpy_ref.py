"""Independent float64 numpy port of the DeepSeek-V4 reference PREFILL (inference/model.py at start_pos = 0, batch 1)
for unit tests of suc.models.mlx_deepseek_v4: the attention runs on the reference's explicit index matrices
(get_window_topk_idxs / get_compress_topk_idxs / Indexer top-k with -1 padding) through a port of the sparse_attn
kernel (sink in the denominator), the compressor uses the literal overlap_transform, the hyper-connections the literal
hc_split_sinkhorn order, the MoE the per-expert loop of MoE.forward. Weights come from the checkpoint dequantized in
float64 (suc.io.quantized_source numpy references). The QAT round trips are off (the MLX model is compared with
simulate_kv_fp8 / simulate_indexer_fp4 False): their own exactness is tested separately."""
from __future__ import annotations

import math

import numpy as np

from suc.io.quantized_source import np_dequant_fp8_block, np_dequant_mxfp4


def load_reference_weights(model_dir) -> dict[str, np.ndarray]:
    """{checkpoint name: float64 array}: FP8 / MXFP4 dequantized, scales dropped, mtp.* dropped."""
    from suc.io.safetensors_store import SafetensorsStore

    import mlx.core as mx

    st = SafetensorsStore.from_dir(model_dir)
    out = {}
    for n in st.names():
        if n.startswith("mtp.") or n.endswith(".scale"):
            continue
        dt = st.info(n).dtype
        a = st.load(n)
        if dt in ("F8_E4M3", "I8"):
            s = np.array(st.load(n[:-7] + ".scale"))
            f = np_dequant_fp8_block if dt == "F8_E4M3" else np_dequant_mxfp4
            out[n] = f(np.array(a), s)
        elif dt in ("BF16", "F16", "F32"):
            out[n] = np.array(a.astype(mx.float32)).astype(np.float64)
        else:
            out[n] = np.array(a)
    return out


def _rms(x, w, eps):
    return w * (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps))


def _freqs(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    def corr_dim(rot):
        return dim * math.log(original_seq_len / (rot * 2 * math.pi)) / (2 * math.log(base))
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    if original_seq_len > 0:
        low = max(math.floor(corr_dim(beta_fast)), 0)
        high = min(math.ceil(corr_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        smooth = 1 - np.clip((np.arange(dim // 2, dtype=np.float64) - low) / (high - low), 0, 1)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    ang = np.outer(np.arange(seqlen), freqs)
    return np.cos(ang) + 1j * np.sin(ang)


def _rope(x, fc, inverse=False):
    """apply_rotary_emb on x (..., S, [H,] d): pairs (2i, 2i+1); fc (S, d/2)."""
    z = x[..., 0::2] + 1j * x[..., 1::2]
    f = np.conj(fc) if inverse else fc
    f = f[:, None, :] if x.ndim == 3 else f
    z = z * f
    out = np.empty_like(x)
    out[..., 0::2], out[..., 1::2] = z.real, z.imag
    return out


def _softmax(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


class Ref:
    def __init__(self, cfg: dict, w: dict):
        self.c, self.w = cfg, w
        self.L = cfg["num_hidden_layers"]
        self.ratios = cfg["compress_ratios"][: self.L]

    def _fc(self, ratio, S):
        c = self.c
        rs = c.get("rope_scaling") or {}
        if ratio:
            return _freqs(c["qk_rope_head_dim"], S, rs.get("original_max_position_embeddings", 0),
                          c["compress_rope_theta"], rs.get("factor", 1), rs.get("beta_fast", 32), rs.get("beta_slow", 1))
        return _freqs(c["qk_rope_head_dim"], S, 0, c["rope_theta"], 1, 32, 1)

    # -- compressor (start_pos = 0) ----------------------------------------------------
    def compressor(self, p, x, ratio, d, rotate, fc):
        S = x.shape[0]
        w = self.w
        overlap = ratio == 4
        kv = x @ w[p + "wkv.weight"].T
        sc = x @ w[p + "wgate.weight"].T
        cutoff = S - S % ratio
        if cutoff < ratio:
            return np.zeros((0, d))
        kv = kv[:cutoff].reshape(-1, ratio, kv.shape[-1])
        sc = sc[:cutoff].reshape(-1, ratio, sc.shape[-1]) + w[p + "ape"]
        if overlap:
            def ot(t, v):
                n = t.shape[0]
                new = np.full((n, 2 * ratio, d), v)
                new[:, ratio:] = t[:, :, d:]
                new[1:, :ratio] = t[:-1, :, :d]
                return new
            kv, sc = ot(kv, 0.0), ot(sc, -np.inf)
        e = (kv * _softmax(sc, 1)).sum(1)
        e = _rms(e, w[p + "norm.weight"], self.c["rms_norm_eps"])
        rd = self.c["qk_rope_head_dim"]
        e[:, -rd:] = _rope(e[:, -rd:], fc[:cutoff:ratio])
        if rotate:
            n = e.shape[-1]
            h = np.array([[1.0]])
            while h.shape[0] < n:
                h = np.block([[h, h], [h, -h]])
            e = e @ h.T / math.sqrt(n)
        return e

    # -- attention -----------------------------------------------------------------------
    def attention(self, p, x, ratio, tokens_len):
        c, w = self.c, self.w
        S = x.shape[0]
        H, hd, rd, win, eps = c["num_attention_heads"], c["head_dim"], c["qk_rope_head_dim"], c["sliding_window"], \
            c["rms_norm_eps"]
        fc = self._fc(ratio, S)
        qr = _rms(x @ w[p + "wq_a.weight"].T, w[p + "q_norm.weight"], eps)
        q = (qr @ w[p + "wq_b.weight"].T).reshape(S, H, hd)
        q = q / np.sqrt((q * q).mean(-1, keepdims=True) + eps)
        q[..., -rd:] = _rope(q[..., -rd:], fc)
        kv = _rms(x @ w[p + "wkv.weight"].T, w[p + "kv_norm.weight"], eps)
        kv[:, -rd:] = _rope(kv[:, -rd:], fc)
        base = np.arange(S)[:, None]
        mw = np.clip(base - win + 1, 0, None) + np.arange(min(S, win))
        idx = np.where(mw > base, -1, mw)                                       # get_window_topk_idxs
        keys = kv
        if ratio:
            offset = S
            ent = self.compressor(p + "compressor.", x, ratio, hd, False, fc)
            n = S // ratio
            if ratio == 4:
                ci = self.indexer(p + "indexer.", x, qr, fc, S)
            else:
                m = np.tile(np.arange(n), (S, 1))
                ci = np.where(m >= np.arange(1, S + 1)[:, None] // ratio, -1, m + offset)   # get_compress_topk_idxs
            idx = np.concatenate([idx, ci], axis=1)
            keys = np.concatenate([kv, ent], axis=0)
        # sparse_attn: scores over the indexed keys, sink in the denominator
        o = np.zeros((S, H, hd))
        sink = w[p + "attn_sink"]
        scale = hd ** -0.5
        for t in range(S):
            ks = [j for j in idx[t] if j != -1]
            K = keys[ks]                                                            # (k, hd)
            s = (q[t] @ K.T) * scale                                                # (H, k)
            m = s.max(-1, keepdims=True)
            e = np.exp(s - m)
            o[t] = (e @ K) / (e.sum(-1, keepdims=True) + np.exp(sink[:, None] - m))
        o[..., -rd:] = _rope(o[..., -rd:], fc, inverse=True)
        G, R = c["o_groups"], c["o_lora_rank"]
        og = o.reshape(S, G, -1)
        wa = w[p + "wo_a.weight"].reshape(G, R, -1)
        u = np.einsum("sgd,grd->sgr", og, wa)
        return u.reshape(S, G * R) @ w[p + "wo_b.weight"].T

    def indexer(self, p, x, qr, fc, S):
        c, w = self.c, self.w
        ih, idd, rd, topk = c["index_n_heads"], c["index_head_dim"], c["qk_rope_head_dim"], c["index_topk"]
        q = (qr @ w[p + "wq_b.weight"].T).reshape(S, ih, idd)
        q[..., -rd:] = _rope(q[..., -rd:], fc)
        h = np.array([[1.0]])
        while h.shape[0] < idd:
            h = np.block([[h, h], [h, -h]])
        q = q @ h.T / math.sqrt(idd)
        k = self.compressor(p + "compressor.", x, 4, idd, True, fc)
        wts = (x @ w[p + "weights_proj.weight"].T) * (idd ** -0.5 * ih ** -0.5)
        sc = (np.maximum(np.einsum("shd,td->sht", q, k), 0) * wts[..., None]).sum(1)
        n = S // 4
        mask = np.tile(np.arange(n), (S, 1)) >= np.arange(1, S + 1)[:, None] // 4
        sc = np.where(mask, -np.inf, sc)
        kk = min(topk, n)
        top = np.argsort(-sc, axis=-1, kind="stable")[:, :kk]
        return np.where(top >= np.arange(1, S + 1)[:, None] // 4, -1, top + S)

    # -- MoE ------------------------------------------------------------------------------
    def moe(self, p, x, tokens, layer):
        c, w = self.c, self.w
        k, E, lim = c["num_experts_per_tok"], c["n_routed_experts"], c["swiglu_limit"]
        s = np.sqrt(np.logaddexp(x @ w[p + "gate.weight"].T, 0.0))
        if layer < c["num_hash_layers"]:
            inds = w[p + "gate.tid2eid"][tokens].astype(np.int64)
        else:
            inds = np.argsort(-(s + w[p + "gate.bias"]), axis=-1, kind="stable")[:, :k]
        wt = np.take_along_axis(s, inds, -1)
        wt = wt / wt.sum(-1, keepdims=True) * c["routed_scaling_factor"]

        def expert(q, h, wr=None):
            g, u = h @ w[q + "w1.weight"].T, h @ w[q + "w3.weight"].T
            u, g = np.clip(u, -lim, lim), np.minimum(g, lim)
            a = g / (1 + np.exp(-g)) * u
            if wr is not None:
                a = wr * a
            return a @ w[q + "w2.weight"].T

        y = np.zeros_like(x)
        for e in range(E):
            r, j = np.where(inds == e)
            if r.size:
                y[r] += expert(p + f"experts.{e}.", x[r], wt[r, j][:, None])
        return y + expert(p + "shared_experts.", x)

    # -- hyper-connections ---------------------------------------------------------------------
    def hc_pre(self, x, p):
        c, w = self.c, self.w
        hc, eps = c["hc_mult"], c["hc_eps"]
        S = x.shape[0]
        xf = x.reshape(S, -1)
        mixes = (xf @ w[p + "_fn"].T) / np.sqrt((xf * xf).mean(-1, keepdims=True) + c["rms_norm_eps"])
        sc, b = w[p + "_scale"], w[p + "_base"]
        pre = 1 / (1 + np.exp(-(mixes[:, :hc] * sc[0] + b[:hc]))) + eps
        post = 2 / (1 + np.exp(-(mixes[:, hc:2 * hc] * sc[1] + b[hc:2 * hc])))
        comb = (mixes[:, 2 * hc:] * sc[2] + b[2 * hc:]).reshape(S, hc, hc)
        comb = _softmax(comb, -1) + eps
        comb = comb / (comb.sum(-2, keepdims=True) + eps)
        for _ in range(c["hc_sinkhorn_iters"] - 1):
            comb = comb / (comb.sum(-1, keepdims=True) + eps)
            comb = comb / (comb.sum(-2, keepdims=True) + eps)
        return (pre[..., None] * x).sum(1), post, comb

    @staticmethod
    def hc_post(f, res, post, comb):
        return post[..., None] * f[:, None, :] + (comb[..., None] * res[:, :, None, :]).sum(1)

    def block(self, i, x, tokens):
        c, w = self.c, self.w
        p = f"layers.{i}."
        S = x.shape[0]
        res = x
        y, post, comb = self.hc_pre(x, p + "hc_attn")
        y = _rms(y, w[p + "attn_norm.weight"], c["rms_norm_eps"])
        x = self.hc_post(self.attention(p + "attn.", y, self.ratios[i], S), res, post, comb)
        res = x
        y, post, comb = self.hc_pre(x, p + "hc_ffn")
        y = _rms(y, w[p + "ffn_norm.weight"], c["rms_norm_eps"])
        return self.hc_post(self.moe(p + "ffn.", y, tokens, i), res, post, comb)

    def forward(self, tokens: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Block outputs (S, hc, D) and logits (S, V) for one token sequence."""
        c, w = self.c, self.w
        h = np.repeat(w["embed.weight"][tokens][:, None, :], c["hc_mult"], axis=1)
        outs = []
        for i in range(self.L):
            h = self.block(i, h, tokens)
            outs.append(h)
        S = h.shape[0]
        xf = h.reshape(S, -1)
        mixes = (xf @ w["hc_head_fn"].T) / np.sqrt((xf * xf).mean(-1, keepdims=True) + c["rms_norm_eps"])
        pre = 1 / (1 + np.exp(-(mixes * w["hc_head_scale"] + w["hc_head_base"]))) + c["hc_eps"]
        y = _rms((pre[..., None] * h).sum(1), w["norm.weight"], c["rms_norm_eps"])
        return outs, y @ w["head.weight"].T
