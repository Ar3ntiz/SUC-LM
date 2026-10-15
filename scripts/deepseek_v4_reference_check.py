"""Torch reference check of the MLX deepseek_v4 port (suc.models.mlx_deepseek_v4) on the REAL DeepSeek-V4-Flash blocks.

Two steps, two environments (the torch venv has no MLX, the project venv has no torch):

  1. torch  (e.g. ~/suc-local/torch-venv on the Mac Studio): imports the checkpoint's OFFICIAL reference implementation
     (<model_dir>/inference/model.py, or --ref-code DIR, or hf_hub_download at the pinned revision) with the CUDA-only
     tilelang kernels (inference/kernel.py) and fast_hadamard_transform replaced by plain-torch equivalents written
     from their source (act_quant / fp4_act_quant: power-of-two ue8m0 scales by IEEE bit rounding, E4M3 / E2M1
     round trips; fp8_gemm / fp4_gemm: dequantized float32 GEMMs; sparse_attn: gathered keys, sink in the denominator,
     P rounded to bf16 before P.V as the kernel does; hc_split_sinkhorn: the kernel's loop order). It builds the
     reference Block of layers 0..N-1 one at a time on the CPU (only that block's weights, read with safe_open), runs a
     fixed token sequence (seeded random ids) from the embedding through the blocks with start_pos = 0, and saves the
     tokens and every block output (float32) to an .npz; with --blocks 43 also hc_head + norm + head logits.
       --dtype float32   reference globals default_dtype = float32 (weights dequantized exactly, no FP8 GEMMs):
                         a port-correctness check (MLX float32 must agree to ~1e-5 relative; KV-FP8 / indexer-FP4
                         round trips stay on in both, so isolated rounding flips are possible)
       --dtype bfloat16  the same in bfloat16 (the dtype suc compress uses)
       --dtype fp8       the published inference path: FP8 / FP4 weights, activations FP8-quantized before every FP8
                         / FP4 GEMM (needs torch >= 2.8 for float8_e8m0fnu / float4_e2m1fn_x2); compared with MLX
                         bfloat16 + simulate_act_fp8 = True
  2. mlx    (project venv): runs the same tokens through suc.models.mlx_deepseek_v4 block by block (weights from the
     same checkpoint, dequantized by sanitize) in the matching configuration and writes a result JSON
     (experiments/results) with per-block max-abs / relative Frobenius errors of the 4 x 4096 streams (+ logits KL /
     top-1), plus, for --dtype fp8, the MLX-internal KL of turning the activation simulation off (the suc default).

Blocks 0..3 cover every layer type: 0-1 sliding window only + hash routing, 2 CSA (ratio 4, indexer) + hash routing,
3 HCA (ratio 128). --seq 2560 (default) makes 640 CSA entries, so the indexer top-512 selection is exercised
(torch.topk vs MLX argpartition may break exact score ties differently) and 20 HCA entries.

  M=~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062
  ~/suc-local/torch-venv/bin/python scripts/deepseek_v4_reference_check.py torch --model-dir $M --dtype float32 --out logs/ds4_ref_f32.npz
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/deepseek_v4_reference_check.py mlx --model-dir $M --ref logs/ds4_ref_f32.npz
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path

REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"


# ------------------------------------------------------------------------------------------- torch: fake kernels

def fake_kernel_modules():
    """(kernel, fast_hadamard_transform) modules in plain torch, semantics of inference/kernel.py."""
    import torch

    FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       dtype=torch.float32)

    def pow2_ceil(v: torch.Tensor) -> torch.Tensor:
        """fast_round_scale: 2^ceil(log2(v)) from the float32 bits (v > 0)."""
        b = v.float().contiguous().view(torch.int32)
        e = ((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0).to(torch.int32)
        return ((e.clamp(-126, 127) + 127) << 23).view(torch.float32)

    def e2m1_round(y: torch.Tensor) -> torch.Tensor:
        a = y.abs()
        code = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int() + (a > 2.5).int()
                + (a >= 3.5).int() + (a > 5.0).int())
        return FP4[:8][code.long()] * torch.where(y < 0, -1.0, 1.0)

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        N = x.size(-1)
        xf = x.float().unflatten(-1, (N // block_size, block_size))
        amax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-4)
        s = pow2_ceil(amax * torch.tensor(1 / 448.0, dtype=torch.float32)) if scale_fmt is not None else amax / 448.0
        q = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        if inplace:
            x.copy_((q.float() * s).flatten(-2).to(x.dtype))
            return x
        return q.flatten(-2), s.squeeze(-1).to(scale_dtype)

    def fp4_act_quant(x, block_size=32, inplace=False):
        N = x.size(-1)
        xf = x.float().unflatten(-1, (N // block_size, block_size))
        amax = xf.abs().amax(-1, keepdim=True).clamp(min=6 * 2.0 ** -126)
        s = pow2_ceil(amax * torch.tensor(1 / 6.0, dtype=torch.float32))
        q = e2m1_round((xf / s).clamp(-6.0, 6.0))
        if not inplace:
            raise NotImplementedError("fp4_act_quant is only used in place by model.py")
        x.copy_((q * s).flatten(-2).to(x.dtype))
        return x

    def _expand(s: torch.Tensor, rows: int, cols: int, br: int, bc: int) -> torch.Tensor:
        return s.float().repeat_interleave(br, 0)[:rows].repeat_interleave(bc, 1)[:, :cols]

    def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
        K = a.size(-1)
        af = a.float().unflatten(-1, (K // 128, 128)) * a_s.float().unsqueeze(-1)
        bf = b.float() * _expand(b_s, b.size(0), K, 128, 128)
        return (af.flatten(-2) @ bf.t()).to(torch.get_default_dtype())

    def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
        K = a.size(-1)
        u = b.view(torch.uint8)
        w = torch.stack([FP4[(u & 0x0F).long()], FP4[((u >> 4) & 0x0F).long()]], -1).flatten(-2)
        w = w * b_s.float().repeat_interleave(32, 1)
        af = a.float().unflatten(-1, (K // 128, 128)) * a_s.float().unsqueeze(-1)
        return (af.flatten(-2) @ w.t()).to(torch.get_default_dtype())

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, chunk=256):
        """Keys gathered by index (-1 = none), float32 scores, sink in the denominator; P rounded to q.dtype before
        P.V (the kernel rounds it to bf16, its only dtype). Query chunks bound the gathered keys' memory."""
        b, m, h, d = q.shape
        out = torch.empty_like(q)
        for bi in range(b):
            for s0 in range(0, m, chunk):
                idx = topk_idxs[bi, s0:s0 + chunk].long()
                valid = idx >= 0
                g = kv[bi][idx.clamp(min=0)].float()                                     # (c, k, d)
                s = torch.einsum("chd,ckd->chk", q[bi, s0:s0 + chunk].float(), g) * softmax_scale
                s = s.masked_fill(~valid[:, None, :], float("-inf"))
                mx_ = s.amax(-1, keepdim=True)
                e = torch.exp(s - mx_)
                den = e.sum(-1, keepdim=True) + torch.exp(attn_sink.float().view(1, h, 1) - mx_)
                o = torch.einsum("chk,ckd->chd", e.to(q.dtype).float(), g) / den
                out[bi, s0:s0 + chunk] = o.to(q.dtype)
        return out

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        hc = hc_mult
        pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
        comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
        comb = torch.softmax(comb, -1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    kernel = types.ModuleType("kernel")
    for f in (act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn):
        setattr(kernel, f.__name__, f)

    def hadamard_transform(x, scale=1.0):
        n = x.size(-1)
        H = torch.ones(1, 1, dtype=torch.float32)
        while H.size(0) < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        return ((x.float() @ H.t()) * scale).to(x.dtype)

    fht = types.ModuleType("fast_hadamard_transform")
    fht.hadamard_transform = hadamard_transform
    return kernel, fht


_HF_TO_REF = {"vocab_size": "vocab_size", "hidden_size": "dim", "moe_intermediate_size": "moe_inter_dim",
              "num_hidden_layers": "n_layers", "num_hash_layers": "n_hash_layers", "num_attention_heads": "n_heads",
              "n_routed_experts": "n_routed_experts", "n_shared_experts": "n_shared_experts",
              "num_experts_per_tok": "n_activated_experts", "scoring_func": "score_func",
              "routed_scaling_factor": "route_scale", "swiglu_limit": "swiglu_limit", "q_lora_rank": "q_lora_rank",
              "head_dim": "head_dim", "qk_rope_head_dim": "rope_head_dim", "rms_norm_eps": "norm_eps",
              "o_groups": "o_groups", "o_lora_rank": "o_lora_rank", "sliding_window": "window_size",
              "compress_ratios": "compress_ratios", "compress_rope_theta": "compress_rope_theta",
              "rope_theta": "rope_theta", "index_n_heads": "index_n_heads", "index_head_dim": "index_head_dim",
              "index_topk": "index_topk", "hc_mult": "hc_mult", "hc_sinkhorn_iters": "hc_sinkhorn_iters",
              "hc_eps": "hc_eps"}


def reference_args(hf: dict) -> dict:
    """inference/model.py ModelArgs fields from the HF config.json (the names of inference/config.json)."""
    out = {r: hf[k] for k, r in _HF_TO_REF.items() if k in hf}
    out["compress_ratios"] = tuple(hf["compress_ratios"])
    rs = hf.get("rope_scaling") or {}
    out.update(rope_factor=rs.get("factor", 1), original_seq_len=rs.get("original_max_position_embeddings", 0),
               beta_fast=rs.get("beta_fast", 32), beta_slow=rs.get("beta_slow", 1))
    return out


def _reference_code(a) -> Path:
    if a.ref_code:
        return Path(a.ref_code)
    local = Path(a.model_dir) / "inference"
    if (local / "model.py").exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, "inference/model.py", revision=REVISION)).parent


def torch_side(a) -> None:
    import importlib.util

    import numpy as np
    import torch
    from safetensors import safe_open

    kernel, fht = fake_kernel_modules()
    sys.modules["kernel"], sys.modules["fast_hadamard_transform"] = kernel, fht
    code = _reference_code(a)
    spec = importlib.util.spec_from_file_location("ds4_reference_model", code / "model.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)

    d = Path(a.model_dir)
    hf = json.loads((d / "config.json").read_text())
    fp8 = a.dtype == "fp8"
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "fp8": torch.bfloat16}[a.dtype]
    args = M.ModelArgs(**reference_args(hf))
    args.max_batch_size, args.max_seq_len = 1, max(a.seq, 128)
    args.dspark_block_size, args.n_mtp_layers = 0, 0
    if fp8 and not (hasattr(torch, "float4_e2m1fn_x2") and hasattr(torch, "float8_e8m0fnu")):
        raise SystemExit("--dtype fp8 needs torch >= 2.8 (float4_e2m1fn_x2, float8_e8m0fnu); use float32 / bfloat16")
    args.expert_dtype = "fp4" if fp8 else None
    args.dtype = "fp8" if fp8 else "bf16"
    # module globals normally set by Transformer.__init__ (only Blocks are built here)
    M.world_size, M.rank = 1, 0
    M.default_dtype = torch.float8_e4m3fn if fp8 else dt
    M.scale_fmt = "ue8m0"
    M.scale_dtype = torch.float8_e8m0fnu if fp8 else torch.float32
    torch.set_default_dtype(dt)
    if not fp8:                                      # rotate_activation asserts bf16 input
        M.rotate_activation = lambda x: fht.hadamard_transform(x, scale=x.size(-1) ** -0.5)

    wmap = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    handles: dict[str, object] = {}

    def load(name: str) -> torch.Tensor:
        f = wmap[name]
        if f not in handles:
            handles[f] = safe_open(str(d / f), framework="pt")
        return handles[f].get_tensor(name)

    def dequant(name: str) -> torch.Tensor:
        """Exact float32 value of a checkpoint weight (FP8 128x128 / MXFP4 per 32, E8M0 scales)."""
        w = load(name)
        sname = name[: -len(".weight")] + ".scale"
        if sname not in wmap:
            return w.float()
        s = torch.pow(2.0, load(sname).view(torch.uint8).float() - 127)
        if w.dtype == torch.int8:
            u = w.view(torch.uint8)
            tab = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                               dtype=torch.float32)
            v = torch.stack([tab[(u & 0x0F).long()], tab[((u >> 4) & 0x0F).long()]], -1).flatten(-2)
            return v * s.repeat_interleave(32, 1)
        return w.float() * s.repeat_interleave(128, 0)[: w.size(0)].repeat_interleave(128, 1)[:, : w.size(1)]

    rng = np.random.default_rng(a.seed)
    tokens = rng.integers(2, int(hf["vocab_size"]) - 1, (1, a.seq)).astype(np.int64)
    ids = torch.from_numpy(tokens)
    with torch.inference_mode():
        h = load("embed.weight").to(dt)[ids]
        h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
        outs = {}
        block = None
        for i in range(a.blocks):
            block = M.Block(i, args)
            if dt == torch.float32:          # the reference builds wo_a (and the gate) in bf16 explicitly
                block = block.float()
            for k, p in block.state_dict().items():
                name = f"layers.{i}.{k}"
                if name not in wmap:
                    if k.endswith(("kv_cache", "kv_state", "score_state", "freqs_cis")) or "kv_cache" in k:
                        continue
                    raise KeyError(f"{name}: not in the checkpoint")
                if p.dtype in (torch.float8_e4m3fn,) or str(p.dtype).startswith(("torch.float8_e8m0", "torch.float4")):
                    src = load(name)
                    p.copy_(src.view(p.dtype) if src.dtype != p.dtype else src)
                elif p.dtype == torch.int32 or p.dtype == torch.int64:
                    p.copy_(load(name))
                else:
                    p.copy_(dequant(name).to(p.dtype))
            h = block(h, 0, ids)
            outs[f"block_{i:02d}"] = h.float().numpy()
            print(f"[torch] block {i}: max |h| {h.float().abs().max().item():.3g}", flush=True)
            if i < a.blocks - 1:
                del block
        if a.blocks == int(hf["num_hidden_layers"]):
            fn, base, sc = (load(n).float() for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"))
            x = block.hc_head(h, fn, sc, base)
            normw = load("norm.weight").float()
            xf = x.float()
            xn = (normw * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + args.norm_eps))).to(dt)
            outs["logits"] = (xn.float() @ load("head.weight").float().t()).numpy()
    np.savez(a.out, tokens=tokens, dtype=a.dtype, blocks=a.blocks, seq=a.seq, seed=a.seed, **outs)
    print(f"[torch] wrote {a.out}")


# ------------------------------------------------------------------------------------------- mlx side

def mlx_block_outputs(model_dir, tokens, blocks: int, dtype, *, simulate_act: bool = False) -> dict:
    """Block outputs (streams only, (1, S, hc, D) float32) and, for every block, the logits; layer-streamed."""
    import mlx.core as mx
    import numpy as np

    from suc.models.streaming import _release, load_lazy

    model, _ = load_lazy(model_dir)
    if dtype == mx.float32:
        model.set_dtype(mx.float32)
    a = model.args
    for layer in model.layers:
        for m in layer.modules():
            if hasattr(m, "_sim_act"):
                m._sim_act = simulate_act
    h = model.embed_stream(mx.array(tokens))
    out = {}
    W = a.stream_width
    for i, layer in enumerate(model.layers[:blocks]):
        h = layer(h, None)
        mx.eval(h)
        out[f"block_{i:02d}"] = np.array(h[..., :W].astype(mx.float32)).reshape(*tokens.shape, a.hc_mult, a.hidden_size)
        _release(layer)
        mx.clear_cache()
    if blocks == a.num_hidden_layers:
        out["logits"] = np.array(model.head_stream(h).astype(mx.float32))
    return out


def _compare(ref: dict, got: dict) -> dict:
    import numpy as np

    rows = {}
    for k, r in ref.items():
        g = got[k].astype(np.float64)
        r = r.astype(np.float64)
        row = {"max_abs": float(np.abs(g - r).max()), "rel_fro": float(np.linalg.norm(g - r) / np.linalg.norm(r))}
        if k == "logits":
            lp = r - r.max(-1, keepdims=True)
            lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
            lq = g - g.max(-1, keepdims=True)
            lq -= np.log(np.exp(lq).sum(-1, keepdims=True))
            row["kl_mean"] = float((np.exp(lp) * (lp - lq)).sum(-1).mean())
            row["top1"] = float((r.argmax(-1) == g.argmax(-1)).mean())
        rows[k] = row
    return rows


def mlx_side(a) -> Path:
    import mlx.core as mx
    import numpy as np

    from suc.io.hashing import model_hash
    from suc.provenance import run_metadata, write_result

    z = np.load(a.ref)
    tokens, blocks, tdt = z["tokens"], int(z["blocks"]), str(z["dtype"])
    ref = {k: z[k] for k in z.files if k.startswith("block_") or k == "logits"}
    runs = {"float32": (mx.float32, False), "bfloat16": (mx.bfloat16, False), "fp8": (mx.bfloat16, True)}
    dt, sim = runs[tdt]
    got = mlx_block_outputs(a.model_dir, tokens, blocks, dt, simulate_act=sim)
    res = {"torch_dtype": tdt, "mlx_dtype": str(dt), "mlx_simulate_act_fp8": sim, "mlx_vs_torch": _compare(ref, got)}
    if tdt == "fp8":                     # cost of the suc default (activation simulation off) inside MLX
        off = mlx_block_outputs(a.model_dir, tokens, blocks, mx.bfloat16, simulate_act=False)
        res["act_sim_off_vs_on"] = _compare(got, off)
    for k, r in res["mlx_vs_torch"].items():
        print(f"{k}: max_abs {r['max_abs']:.3e} rel_fro {r['rel_fro']:.3e}"
              + (f" KL {r['kl_mean']:.3e} top1 {r['top1']:.3f}" if "kl_mean" in r else ""))
    md = run_metadata(model=REPO, model_revision=a.revision, model_hash=model_hash(a.model_dir),
                      dataset_hash=f"random-tokens:seed={int(z['seed'])}:seq={int(z['seq'])}", seed=int(z["seed"]),
                      config_hash=None)
    out = {"metadata": md, "name": f"deepseek_v4_reference_check_{tdt}", "blocks": blocks, "seq": int(z["seq"]),
           **res}
    return write_result(out, a.output_dir, out["name"])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="side", required=True)
    t = sub.add_parser("torch")
    t.add_argument("--model-dir", required=True)
    t.add_argument("--ref-code", default=None, help="directory with the reference model.py / config.json")
    t.add_argument("--out", required=True)
    t.add_argument("--blocks", type=int, default=4)
    t.add_argument("--seq", type=int, default=2560)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--dtype", choices=("float32", "bfloat16", "fp8"), default="float32")
    m = sub.add_parser("mlx")
    m.add_argument("--model-dir", required=True)
    m.add_argument("--ref", required=True)
    m.add_argument("--revision", default=REVISION)
    m.add_argument("--output-dir", default="experiments/results")
    a = p.parse_args(argv)
    if a.side == "torch":
        torch_side(a)
    else:
        print(mlx_side(a))


if __name__ == "__main__":
    main()
