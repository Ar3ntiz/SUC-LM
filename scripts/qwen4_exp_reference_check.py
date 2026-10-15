"""Torch reference check of the MLX qwen4_exp port (suc.models.mlx_qwen4_exp) on the REAL Qwen3.8-Flash-Next blocks.

Two steps, two environments (the torch venv has no MLX, the project venv has no torch):

  1. torch  (transformers with models/qwen4_exp, e.g. ~/suc-local/torch-venv on the Mac Studio): builds the HF
     Qwen4ExpTextDecoderLayer of blocks 0..N-1 one at a time (meta device, then only that block's weights; the PLE
     n-gram table is gathered row by row from the safetensors shards, never allocated), runs a fixed token sequence
     (seeded random ids with one EOS in the middle, so the n-gram context reset is exercised) from the embedding
     through the blocks, and saves the tokens and every block output (float32) to an .npz. With --blocks 48 it also
     applies the final hyper-connection mixer + LM head and saves the logits.
  2. mlx    (project venv): runs the same tokens through the MLX model block by block (suc.models.streaming), once with
     the checkpoint weights in float32 (port correctness: must agree with torch float32 to ~1e-5 relative) and once in
     bfloat16 (the dtype `suc compress` uses), and writes a result JSON (experiments/results) with per-block max-abs
     and relative Frobenius errors of the 4 x 2560 residual streams (and logits KL / top-1 with --blocks 48).

Blocks 0..3 cover both mixer types (Gated DeltaNet at 0-2, QSA full attention at 3) and the PLE layer (1). With
--seq <= 2051 the QSA indexer is exactly dense (docs/arch/qwen4_exp.md §2); --seq 4096 exercises the sparse selection
(torch.topk vs MLX argpartition may break exact score ties differently).

  ~/suc-local/torch-venv/bin/python scripts/qwen4_exp_reference_check.py torch --model-dir $M --out logs/q4_ref.npz
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/qwen4_exp_reference_check.py mlx --model-dir $M --ref logs/q4_ref.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"


# ------------------------------------------------------------------------------------------- torch side

def torch_side(a) -> None:
    import numpy as np
    import torch
    from safetensors import safe_open
    from torch import nn
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    d = Path(a.model_dir)
    cfg = json.loads((d / "config.json").read_text())
    tc = Qwen4ExpTextConfig.from_dict(cfg["text_config"])
    tc._attn_implementation = "sdpa"
    wmap = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    handles: dict[str, object] = {}

    def handle(name: str):
        f = wmap[name]
        if f not in handles:
            handles[f] = safe_open(str(d / f), framework="pt")
        return handles[f]

    def load(name: str) -> torch.Tensor:
        return handle(name).get_tensor(name)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[a.dtype]
    n_layers = tc.num_hidden_layers
    blocks = n_layers if a.blocks in ("all", str(n_layers)) else int(a.blocks)
    eos = tc.eos_token_id[0] if isinstance(tc.eos_token_id, list) else tc.eos_token_id
    rng = np.random.default_rng(a.seed)
    tokens = rng.integers(0, tc.vocab_size, (1, a.seq)).astype(np.int64)
    tokens[0, a.seq // 2] = eos
    ids = torch.from_numpy(tokens)

    class RowGather(nn.Module):
        """Stand-in for the 320 M x 160 n-gram nn.Embedding: rows read from the checkpoint shards on demand."""

        def __init__(self, prefix: str, parts: int):
            super().__init__()
            self.names = [f"{prefix}shard_{j}.weight" for j in range(parts)]
            self.rows = handle(self.names[0]).get_slice(self.names[0]).get_shape()[0]
            self.register_buffer("weight", torch.empty(0), persistent=False)   # HF reads weight.device

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            flat = idx.reshape(-1).tolist()
            out = []
            for r in flat:
                n = self.names[r // self.rows]
                out.append(handle(n).get_slice(n)[r % self.rows: r % self.rows + 1])
            return torch.cat(out).reshape(*idx.shape, -1).to(dtype)

    with torch.no_grad():
        h = nn.functional.embedding(ids, load("model.language_model.embed_tokens.weight").to(dtype))
        rotary = M.Qwen4ExpTextRotaryEmbedding(tc)
        pos = torch.arange(a.seq)[None, None].expand(3, 1, a.seq)
        cos, sin = rotary(h, pos)
        mask = torch.tril(torch.ones(a.seq, a.seq, dtype=torch.bool))[None, None]
        h = h.repeat(1, 1, tc.hc_count)
        outs = {}
        for i in range(blocks):
            prefix = f"model.language_model.layers.{i}."
            with torch.device("meta"):
                layer = M.Qwen4ExpTextDecoderLayer(tc, i)
            if layer.ple is not None:
                layer.ple.ple_embedding.ngram_embedding = RowGather(prefix + "ple.ple_embedding.ngram_embedding.",
                                                                    tc.split_ngram_parts)
            layer = layer.to_empty(device="cpu").to(dtype)
            sd = {k[len(prefix):]: load(k) for k in wmap if k.startswith(prefix) and ".ngram_embedding.shard_" not in k}
            missing, unexpected = layer.load_state_dict(sd, strict=False)
            missing = [m for m in missing if not m.endswith("ngram_embedding.weight")]
            if missing or unexpected:
                raise RuntimeError(f"block {i}: missing {missing[:5]}, unexpected {unexpected[:5]}")
            for k in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):   # keep int64
                if layer.ple is not None:
                    getattr(layer.ple.ple_embedding, k).copy_(load(prefix + "ple.ple_embedding." + k))
            layer.eval()
            h = layer(h, position_embeddings=(cos, sin), attention_mask=mask, conv_mask=None,
                      past_key_values=None, ple_input_ids=ids)
            outs[f"block_{i:02d}"] = h.float().numpy()
            print(f"torch block {i}: |h| max {h.abs().max().item():.3f}", flush=True)
            del layer
        if blocks == n_layers:
            mixer = M.Qwen4ExpTextGatedResidual(tc, use_combine=False).to(dtype)
            p = "model.language_model.hyper_connection_mixer."
            mixer.load_state_dict({k[len(p):]: load(k) for k in wmap if k.startswith(p)})
            u = mixer(h)
            outs["logits"] = (u @ load("lm_head.weight").to(dtype).T).float().numpy()
    np.savez(a.out, tokens=tokens, dtype=a.dtype, blocks=blocks, seq=a.seq, seed=a.seed, **outs)
    print(f"wrote {a.out}")


# ------------------------------------------------------------------------------------------- mlx side

def mlx_block_outputs(model_dir: str | Path, tokens, blocks: int, dtype) -> dict:
    """Stream tensors (float32 numpy, the hc*H stream channels only) after blocks 0..blocks-1, + logits if all."""
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_map

    from suc.models.streaming import _release, head, load_lazy, model_blocks

    model, _ = load_lazy(model_dir)
    cast = lambda m: m.update(tree_map(lambda p: p.astype(dtype), m.parameters()))   # noqa: E731
    cast(model.model.embed_tokens)
    h = model.embed_stream(mx.array(tokens)).astype(dtype)
    w = model.text.stream_width
    out = {}
    layers = model_blocks(model)
    for i in range(blocks):
        b = layers[i]
        cast(b)
        h = b(h)
        mx.eval(h)
        out[f"block_{i:02d}"] = np.array(h[..., :w].astype(mx.float32))
        _release(b)
        mx.clear_cache()
    if blocks == len(layers):
        cast(model.model.hyper_connection_mixer)
        cast(model.lm_head)
        out["logits"] = np.array(head(model, h).astype(mx.float32))
    return out


def compare(ref: dict, got: dict) -> dict:
    import numpy as np

    rows = {}
    for k in sorted(got):
        r, g = ref[k].astype(np.float64), got[k].astype(np.float64)
        rows[k] = {"max_abs": float(np.abs(g - r).max()), "rel_fro": float(np.linalg.norm(g - r) / np.linalg.norm(r)),
                   "ref_max_abs": float(np.abs(r).max())}
        if k == "logits":
            def logp(x):
                x = x - x.max(-1, keepdims=True)
                return x - np.log(np.exp(x).sum(-1, keepdims=True))
            lr, lg = logp(r[0]), logp(g[0])
            rows[k].update(kl_mean=float((np.exp(lr) * (lr - lg)).sum(-1).mean()),
                           top1=float((lr.argmax(-1) == lg.argmax(-1)).mean()))
    return rows


def mlx_side(a) -> Path:
    import mlx.core as mx
    import numpy as np

    from suc.io.hashing import model_hash
    from suc.provenance import run_metadata, write_result

    z = np.load(a.ref)
    blocks = int(z["blocks"])
    ref = {k: z[k] for k in z.files if k.startswith("block_") or k == "logits"}
    res = {}
    for name, dt in (("float32", mx.float32), ("bfloat16", mx.bfloat16)):
        got = mlx_block_outputs(a.model_dir, z["tokens"], blocks, dt)
        res[name] = compare(ref, got)
        worst = max(v["rel_fro"] for v in res[name].values())
        print(f"mlx {name} vs torch {z['dtype']}: worst block rel. Frobenius error {worst:.3e}", flush=True)
    tok_hash = __import__("hashlib").sha256(np.asarray(z["tokens"]).tobytes()).hexdigest()
    md = run_metadata(model="Qwen/Qwen3.8-Flash-Next", model_revision=a.revision, model_hash=model_hash(a.model_dir),
                      dataset_hash=f"random_tokens_seed{int(z['seed'])}_len{int(z['seq'])}:{tok_hash[:16]}",
                      seed=int(z["seed"]), config_hash="scripts/qwen4_exp_reference_check.py")
    result = {"metadata": md, "name": "qwen4_exp_reference_check", "torch_dtype": str(z["dtype"]), "blocks": blocks,
              "seq": int(z["seq"]), "mlx_vs_torch": res,
              "criterion": "float32 MLX vs float32 torch: port correctness; bfloat16: precision of the run dtype"}
    p = write_result(result, a.output_dir, "qwen4_exp_reference_check")
    print(f"wrote {p}")
    return p


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="side", required=True)
    t = sub.add_parser("torch")
    t.add_argument("--model-dir", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--blocks", default="4", help="number of blocks from 0, or 'all' (adds the logits)")
    t.add_argument("--seq", type=int, default=64)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    m = sub.add_parser("mlx")
    m.add_argument("--model-dir", required=True)
    m.add_argument("--ref", required=True)
    m.add_argument("--revision", default=REVISION)
    m.add_argument("--output-dir", default="experiments/results")
    a = ap.parse_args(argv)
    if a.side == "torch":
        torch_side(a)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        mlx_side(a)


if __name__ == "__main__":
    main()
