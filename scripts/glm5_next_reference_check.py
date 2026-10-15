"""Torch reference check of the MLX glm5_next port (suc.models.mlx_glm5_next) against transformers' glm5_next.

Two steps, two environments (the torch venv has no MLX, the project venv has no torch):

  1. torch  (~/suc-local/torch-venv on the Mac Studio: torch 2.14, transformers 5.18 with models/glm5_next): builds
     transformers' Glm5NextTextDecoderLayer for layers 0..N-1 from the checkpoint's text_config, ONE AT A TIME on the
     CPU, loading only that layer's tensors with safe_open (FP8 weights dequantized as transformers does: float32
     q * weight_scale_inv, rounded to bfloat16; the per-expert tensors fused into gate_up_proj / down_proj, the three
     q / k / v short convolutions into conv1d, the forget-gate tensors under forget_gate, hc_{attn,ffn}_* into
     {attn,ffn}_hc). A fixed token sequence (seeded random ids) runs from the embedding (4 streams) through the blocks
     as one prefill without cache (attention_mask all ones, the indexer's own path; KDA in its chunked form), and every
     block output (float32) is saved to an .npz; with --blocks = num_hidden_layers also hc_head mean + norm + lm_head
     logits.
       --dtype float32   module dtype float32 (weights = the BF16 dequantization, then float32): port correctness
                         (MLX float32 must agree to ~1e-5 relative; the indexer top-k may flip on near-ties)
       --dtype bfloat16  the same in bfloat16 (the dtype suc compress uses)
     The shipped config names the DSA layers `deepseek_sparse_attention`, which transformers' mask mapping does not
     know (docs/arch/glm5_next.md §9 risk 6): they are mapped to `indexed_attention` here, as in the MLX port.
  2. mlx    (project venv): runs the same tokens through suc.models.mlx_glm5_next block by block (weights from the same
     checkpoint, dequantized by sanitize) and writes a result JSON (experiments/results) with per-block max-abs /
     relative Frobenius errors of the 4 x 4096 streams (+ logits KL / top-1).

Blocks 0..3 cover every layer type: 0-2 KDA + dense MLP, 3 MLA + DSA indexer + MoE. --seq 2560 (default) makes 640
complete k-pools, so the top-512 pool selection is active for the last ~500 queries.

  M=~/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
  ~/suc-local/torch-venv/bin/python scripts/glm5_next_reference_check.py torch --model-dir $M --dtype float32 --out logs/glm_ref_f32.npz
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/glm5_next_reference_check.py mlx --model-dir $M --ref logs/glm_ref_f32.npz
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
REPO = "zai-org/GLM-5.3-Flash"
_HF = "model.language_model."
_DSA = ("deepseek_sparse_attention", "full_attention")


def text_config_dict(hf: dict) -> dict:
    """The text_config with the DSA layer type renamed for transformers (indexed_attention)."""
    t = dict(hf.get("text_config") or hf)
    t["layer_types"] = ["indexed_attention" if x in _DSA else x for x in t["layer_types"]]
    return t


# ------------------------------------------------------------------------------------------- torch side

class _Tensors:
    """safe_open reader over the shards of a checkpoint (one tensor at a time); FP8 weights dequantized."""

    def __init__(self, model_dir: Path):
        from safetensors import safe_open

        idx = model_dir / "model.safetensors.index.json"
        if idx.exists():
            wmap = json.loads(idx.read_text())["weight_map"]
        else:
            wmap = {}
            for f in sorted(model_dir.glob("*.safetensors")):
                with safe_open(str(f), "pt") as h:
                    wmap.update(dict.fromkeys(h.keys(), f.name))
        self.dir, self.wmap, self._open, self._h = model_dir, wmap, safe_open, {}

    def raw(self, name: str):
        f = self.wmap[name]
        if f not in self._h:
            self._h[f] = self._open(str(self.dir / f), "pt")
        return self._h[f].get_tensor(name)

    def names(self, prefix: str) -> list[str]:
        return [n for n in self.wmap if n.startswith(prefix)]

    def get(self, name: str, dtype):
        """BF16 dequantization of FP8 weights (transformers Fp8Dequantize: float32 q * s, rounded), then `dtype`."""
        import torch

        w = self.raw(name)
        sname = name + "_scale_inv"
        if sname in self.wmap:
            s = self.raw(sname).float()
            r, c = w.shape
            br, bc = -(-r // s.shape[0]), -(-c // s.shape[1])
            full = s.repeat_interleave(br, 0)[:r].repeat_interleave(bc, 1)[:, :c]
            w = (w.float() * full).to(torch.bfloat16)
        return w.to(dtype)


def _layer_state(t: _Tensors, i: int, cfg, dtype) -> dict:
    """transformers parameter names of decoder layer i -> tensors (expert / conv fusion, renames)."""
    import torch

    p = f"{_HF}layers.{i}."
    sd: dict = {}
    experts: dict[int, dict[str, object]] = {}
    convs = {}
    for n in t.names(p):
        if n.endswith("_scale_inv"):
            continue
        k = n[len(p):]
        if (m := re.fullmatch(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight", k)) is not None:
            experts.setdefault(int(m.group(1)), {})[m.group(2)] = t.get(n, dtype)
            continue
        if (m := re.fullmatch(r"self_attn\.([qkv])_conv1d\.weight", k)) is not None:
            convs[m.group(1)] = t.get(n, torch.float32)
            continue
        if (m := re.fullmatch(r"hc_(attn|ffn)_(fn|base|scale)", k)) is not None:
            k = f"{m.group(1)}_hc.{m.group(2)}"
        elif (m := re.fullmatch(r"self_attn\.(f_a_proj\.weight|f_b_proj\.weight|dt_bias|A_log)", k)) is not None:
            k = "self_attn.forget_gate." + m.group(1)
        keep32 = k.endswith(("A_log", "dt_bias", "e_score_correction_bias")) or "_hc." in k
        sd[k] = t.get(n, torch.float32 if keep32 else dtype)
    if convs:
        sd["self_attn.conv1d.weight"] = torch.cat([convs["q"], convs["k"], convs["v"]], 0)
    if experts:
        E = len(experts)
        sd["mlp.experts.gate_up_proj"] = torch.stack([torch.cat([experts[e]["gate"], experts[e]["up"]], 0)
                                                      for e in range(E)])
        sd["mlp.experts.down_proj"] = torch.stack([experts[e]["down"] for e in range(E)])
    return sd


def torch_side(a) -> None:
    import numpy as np
    import torch
    from transformers.models.glm5_next import modeling_glm5_next as M
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    torch.manual_seed(0)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[a.dtype]
    model_dir = Path(a.model_dir).expanduser()
    hf = json.loads((model_dir / "config.json").read_text())
    cfg = Glm5NextTextConfig(**text_config_dict(hf))
    cfg._attn_implementation = "eager"
    if hasattr(cfg, "_experts_implementation"):
        cfg._experts_implementation = "eager"
    t = _Tensors(model_dir)
    rng = np.random.default_rng(a.seed)
    tokens = rng.integers(2, min(cfg.vocab_size, 150000), (1, a.seq)).astype(np.int64)
    ids = torch.from_numpy(tokens)
    emb = t.get(f"{_HF}embed_tokens.weight", dtype)
    h = emb[ids].unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
    mask = torch.ones(ids.shape, dtype=torch.bool)
    out = {"tokens": tokens, "dtype": a.dtype, "blocks": a.blocks, "seq": a.seq, "seed": a.seed}
    topk = None
    with torch.no_grad():
        for i in range(a.blocks):
            layer = M.Glm5NextTextDecoderLayer(cfg, i).to(dtype)
            for n, b in layer.named_buffers():
                if n.endswith("e_score_correction_bias"):
                    b.data = b.data.float()
            sd = _layer_state(t, i, cfg, dtype)
            missing, unexpected = layer.load_state_dict(sd, strict=False)
            missing = [m for m in missing if not m.endswith("e_score_correction_bias")] if \
                "mlp.gate.e_score_correction_bias" in sd else missing
            if missing or unexpected:
                raise RuntimeError(f"layer {i}: missing {missing[:6]}, unexpected {unexpected[:6]}")
            for n, b in layer.named_buffers():          # buffers are not loaded by strict=False on every version
                if n in sd:
                    b.data = sd[n].to(b.dtype)
            layer.eval()
            h, topk = layer(h, attention_mask=mask, position_ids=None, past_key_values=None,
                            position_embeddings=None, prev_topk_indices=topk)
            out[f"block_{i:02d}"] = h.float().numpy()
            print(f"block {i}: |h| {h.float().abs().max().item():.3f}", flush=True)
            del layer
        if a.blocks == cfg.num_hidden_layers:
            norm = M.Glm5NextTextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(dtype)
            norm.weight.data = t.get(f"{_HF}norm.weight", dtype)
            x = norm(h.mean(dim=2))
            out["logits"] = (x @ t.get("lm_head.weight", dtype).T).float().numpy()
    np.savez(a.out, **out)
    print(a.out)


# ------------------------------------------------------------------------------------------- mlx side

def mlx_block_outputs(model_dir, tokens, blocks: int, dtype) -> dict:
    """Block outputs (streams, (1, S, hc, D) float32) and, for every block, the logits; layer-streamed."""
    import mlx.core as mx
    import numpy as np

    from suc.models.streaming import _release, load_lazy

    model, _ = load_lazy(model_dir)
    if dtype == mx.float32:
        model.set_dtype(mx.float32)
    a = model.text
    h = model.embed_stream(mx.array(tokens))
    out = {}
    for i, layer in enumerate(model.layers[:blocks]):
        h = layer(h, None)
        mx.eval(h)
        out[f"block_{i:02d}"] = np.array(h.astype(mx.float32)).reshape(*tokens.shape, a.hc_mult, a.hidden_size)
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
    dt = {"float32": mx.float32, "bfloat16": mx.bfloat16}[tdt]
    got = mlx_block_outputs(a.model_dir, tokens, blocks, dt)
    res = {"torch_dtype": tdt, "mlx_dtype": str(dt), "mlx_vs_torch": _compare(ref, got)}
    for k, r in res["mlx_vs_torch"].items():
        print(f"{k}: max_abs {r['max_abs']:.3e} rel_fro {r['rel_fro']:.3e}"
              + (f" KL {r['kl_mean']:.3e} top1 {r['top1']:.3f}" if "kl_mean" in r else ""))
    md = run_metadata(model=REPO, model_revision=a.revision, model_hash=model_hash(a.model_dir),
                      dataset_hash=f"random-tokens:seed={int(z['seed'])}:seq={int(z['seq'])}", seed=int(z["seed"]),
                      config_hash=None)
    out = {"metadata": md, "name": f"glm5_next_reference_check_{tdt}", "blocks": blocks, "seq": int(z["seq"]), **res}
    return write_result(out, a.output_dir, out["name"])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="side", required=True)
    t = sub.add_parser("torch")
    t.add_argument("--model-dir", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--blocks", type=int, default=4)
    t.add_argument("--seq", type=int, default=2560)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
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
