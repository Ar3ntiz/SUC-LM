"""MoE end-to-end check from SUC codes (target 2: Qwen3-30B-A3B): convert every eligible tensor of an mlx-lm MoE
model (attention + all routed experts; router, norms stay as loaded) to SUC modules (SucLinear / SucSwitchLinear),
then compare against the reconstructed-weights model (same encodings, decoded into the BF16 model) and time it.

Not a test. Usage (Mac Studio, sequentially, nothing else running):
  HF_HUB_OFFLINE=1 nohup .venv/bin/python benchmarks/moe_e2e_check.py --model Qwen/Qwen3-30B-A3B \
      --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 > logs/moe_e2e_check.log 2>&1 &
  Quick smoke run (first 2 blocks converted, the rest stays BF16): --layers 2 --batches 1 --repeats 1
Options: [--codec vq_g4_k256|add3_g8_k256] [--residual-fraction 0.005] [--kmeans-iters 6] [--fit-blocks 65536]
         [--layers N] [--check-seqs 4] [--check-len 64] [--batches 1,4] [--prompt-len 128] [--decode-tokens 32]
         [--repeats 3] [--embed q4|bf16] [--no-q4-baseline] [--no-rotation-groups] [--out-dir experiments/results]

Steps (one process; peak ≈ BF16 model + encodings + one stack being converted):
  1. load (evaluated); with --embed q4 the embedding / LM head become MLX affine 4-bit g64 (same in every variant).
  2. encode tensor by tensor from the loaded weights (cheap k-means: codes are real, quality is not the point;
     seed = SEED + crc32(name), q/k/v share seed_in with rotation groups), decoded rows written into the model once
     per expert stack → the RECONSTRUCTED model; reference logits on --check-seqs text windows.
  3. module check (float32, Python reference decoder): the first converted layer's expert stacks as
     SucSwitchLinear vs codec.decode row by row, on decode-sized (gather path) and prefill-sized (tiled path) rows:
     max / mean abs error, relative error.
  4. speed of the reconstructed model (BF16 weights: the BF16 reference speed).
  5. convert in place block by block (convert_to_compressed, dense stacks dropped), group_projections; resident
     weight bytes; logits → KL(reconstructed || compressed), top-1 agreement (BF16 both: includes BF16 rounding of
     the decoded weights and of the rotated activations, see benchmarks/e2e_generate.py); speed.
  6. strong low-bit baseline (unless --no-q4-baseline): reload, MLX affine 4-bit (router 8-bit, mlx-lm predicate),
     resident bytes and speed.
Result JSON (suc.provenance.write_result, standard metadata) in --out-dir: <ts>_moe_e2e_check_<model>.json.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import os
import re
import sys
import time
import zlib
from pathlib import Path

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suc.analysis.sensitivity import get_weight, set_weight  # noqa: E402
from suc.evaluation.baselines import apply_representation, logits_kl  # noqa: E402
from suc.evaluation.reconstruct import rotation_seeds  # noqa: E402
from suc.evaluation.speed import measure_generation  # noqa: E402
from suc.io.hashing import model_hash  # noqa: E402
from suc.kernels.suc_switch import SucSwitchLinear  # noqa: E402
from suc.models.adapters import get_adapter  # noqa: E402
from suc.provenance import run_metadata, seed_everything, write_result  # noqa: E402
from suc.quant.codecs import make_codec  # noqa: E402
from suc.quant.residual import ResidualCodec  # noqa: E402
from suc.runtime.compressed_model import (convert_to_compressed, expert_stack, group_projections,  # noqa: E402
                                          merge_reports, resident_weight_bytes)

CODECS = {"vq_g4_k256": {"kind": "vq", "k": 256, "block_size": 4, "rotate": True},
          "add3_g8_k256": {"kind": "additive", "k": 256, "block_size": 8, "num_codebooks": 3, "rotate": True}}
CHECK_TEXT = ("The history of the printing press begins in the fifteenth century, when movable metal type made it "
              "possible to reproduce books quickly and cheaply. def fib(n):\n    return n if n < 2 else "
              "fib(n - 1) + fib(n - 2)\nSolve for x: 3x + 7 = 22, so x = 5. Please summarise the main argument of the "
              "essay in two sentences, then list three counter-arguments a careful reader might raise. ")
SEED = 12345
_LAYER = re.compile(r"^model\.layers\.(\d+)\.")


def release() -> None:
    gc.collect()
    mx.clear_cache()


def resolve(model: str, revision: str | None) -> Path:
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model, revision=revision,
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]))


def load(model_dir: Path):
    from mlx_lm.utils import load_model, load_tokenizer
    model, config = load_model(model_dir)
    tok = None
    if any((model_dir / f).exists() for f in ("tokenizer.json", "tokenizer_config.json")):
        tok = load_tokenizer(model_dir, eos_token_ids=config.get("eos_token_id"))
    mx.eval(model.parameters())
    return model, tok


def quantize_embedding(model) -> None:
    import mlx.nn as nn
    nn.quantize(model, group_size=64, bits=4, mode="affine",
                class_predicate=lambda p, m: p in ("model.embed_tokens", "lm_head") and hasattr(m, "to_quantized"))
    mx.eval(model.parameters())


def eligible(model, layers: int | None) -> list[str]:
    adapter = get_adapter({"model_type": model.args.model_type, "num_experts": getattr(model.args, "num_experts", 0)})
    names = set(k for k, _ in tree_flatten(model.parameters()))
    # expand stacked expert weights back to checkpoint names (adapter roles are defined on HF names)
    for k, v in tree_flatten(model.parameters()):
        if (m := re.match(r"^(.*)\.switch_mlp\.(\w+)\.weight$", k)) is not None and v.ndim == 3:
            names.discard(k)
            names.update(f"{m.group(1)}.experts.{e}.{m.group(2)}.weight" for e in range(v.shape[0]))
    out = [s.name for s in adapter.linears(names) if s.eligible]
    if layers is not None:
        out = [n for n in out if (m := _LAYER.match(n)) and int(m.group(1)) < layers]
    return out


def check_tokens(tok, vocab: int, n: int, length: int) -> mx.array:
    if tok is None:
        return mx.array(np.random.default_rng(SEED).integers(0, vocab, size=(n, length)).astype(np.int32))
    ids = tok.encode(CHECK_TEXT * (1 + n * length // 40))
    return mx.array([ids[i * length:(i + 1) * length] for i in range(n)])


def by_layer(names: list[str]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for n in names:
        out.setdefault(int(_LAYER.match(n).group(1)), []).append(n)
    return out


def encode_and_reconstruct(model, names: list[str], codec, rot_groups: bool, verbose: bool) -> tuple[dict, dict]:
    """Encodings of `names` from the loaded weights; their decode written into the model (expert rows once per
    stack). Returns (encodings, info)."""
    enc, bits, params = {}, 0, 0
    t0 = time.perf_counter()
    for li, lnames in sorted(by_layer(names).items()):
        pending: dict[str, dict[int, mx.array]] = {}
        for n in lnames:
            w = get_weight(model, n)
            kw = rotation_seeds(n, SEED) if rot_groups else {}
            e = codec.encode(w.astype(mx.float32), seed=SEED + zlib.crc32(n.encode()), **kw)
            mx.eval(e.arrays)
            w_hat = codec.decode(e).astype(w.dtype)
            mx.eval(w_hat)
            if (st := expert_stack(n)) is not None:
                pending.setdefault(st[0], {})[st[1]] = w_hat
            else:
                set_weight(model, n, w_hat)
            bits += e.bits()["total"]
            params += w.size
            enc[n] = e
            del w
        for stack, rows in pending.items():
            cur = get_weight(model, stack)
            set_weight(model, stack, mx.stack([rows.get(r, cur[r]) for r in range(cur.shape[0])]))
            mx.eval(get_weight(model, stack))
        del pending
        release()
        if verbose:
            print(f"#   encoded layer {li} ({len(enc)} tensors, {time.perf_counter() - t0:.0f}s)", flush=True)
    return enc, {"encoded_tensors": len(enc), "encoded_params": params, "encoded_bits_storage": bits,
                 "encoded_bpw_storage": bits / max(params, 1), "encode_s": time.perf_counter() - t0}


def module_check(enc: dict, codec, names: list[str]) -> list[dict]:
    """SucSwitchLinear of the first converted layer's stacks vs the Python reference decoder (float32)."""
    stacks: dict[str, dict[int, str]] = {}
    for n in names:
        if (st := expert_stack(n)) is not None:
            stacks.setdefault(st[0], {})[st[1]] = n
    if not stacks:
        return []
    first = min(int(_LAYER.match(s).group(1)) for s in stacks)
    out = []
    rng = np.random.default_rng(SEED)
    for stack, rows in sorted(stacks.items()):
        if int(_LAYER.match(stack).group(1)) != first:
            continue
        encs = [enc[rows[e]] for e in range(len(rows))]
        m = SucSwitchLinear.from_encodings(encs)
        for n_rows, path in ((8, "gather"), (512, "tiled")):
            eidx = rng.integers(0, len(encs), size=n_rows).astype(np.uint32)
            x = mx.random.normal((n_rows, m.input_dims), key=mx.random.key(n_rows))
            y = m.rows(x, mx.array(eidx), path=path).astype(mx.float32)
            used = np.unique(eidx)
            dec = {int(e): codec.decode(encs[int(e)]).astype(mx.float32) for e in used}
            ref = mx.stack([x[i] @ dec[int(e)].T for i, e in enumerate(eidx)])
            d = (y - ref).reshape(-1)
            out.append({"stack": stack, "rows": n_rows, "path": path, "max_abs": float(mx.max(mx.abs(d))),
                        "mean_abs": float(mx.mean(mx.abs(d))),
                        "rel": float(mx.linalg.norm(d) / mx.linalg.norm(ref.reshape(-1)))})
            del dec, ref, y
        del m
        release()
    return out


def speed(model, args) -> list[dict]:
    out = []
    for b in args.batches:
        r = measure_generation(model, prompt_len=args.prompt_len, decode_tokens=args.decode_tokens, batch_size=b,
                               warmup=1, repeats=args.repeats, seed=1)
        r.pop("repeats", None)
        out.append(r)
        print(f"#   B={b}: prefill {r['prefill_tok_s']:.1f} tok/s, decode {r['decode_tok_s']:.2f} tok/s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision", default="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
    ap.add_argument("--codec", default="vq_g4_k256", choices=sorted(CODECS))
    ap.add_argument("--residual-fraction", type=float, default=0.0)
    ap.add_argument("--kmeans-iters", type=int, default=6)
    ap.add_argument("--fit-blocks", type=int, default=65536)
    ap.add_argument("--layers", type=int, default=None, help="convert only the first N blocks")
    ap.add_argument("--check-seqs", type=int, default=4)
    ap.add_argument("--check-len", type=int, default=64)
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--embed", default="q4", choices=["q4", "bf16"])
    ap.add_argument("--no-q4-baseline", action="store_true")
    ap.add_argument("--no-rotation-groups", action="store_true")
    ap.add_argument("--out-dir", default="experiments/results")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    args.batches = [int(b) for b in args.batches.split(",") if b]
    seed_everything(SEED)
    verbose = not args.quiet

    model_dir = resolve(args.model, args.revision)
    revision = model_dir.name if model_dir.parent.name == "snapshots" else args.revision
    t0 = time.perf_counter()
    model, tok = load(model_dir)
    load_s = time.perf_counter() - t0
    bf16_bytes = resident_weight_bytes(model)
    if args.embed == "q4":
        quantize_embedding(model)
    base = make_codec(dict(CODECS[args.codec], kmeans_iters=args.kmeans_iters, fit_blocks=args.fit_blocks))
    codec = ResidualCodec(base, args.residual_fraction, value_dtype="int8", selection="magnitude") \
        if args.residual_fraction > 0 else base
    names = eligible(model, args.layers)
    print(f"# {args.model}: {len(names)} tensors to encode ({codec.name})", flush=True)

    enc, enc_info = encode_and_reconstruct(model, names, codec, not args.no_rotation_groups, verbose)
    toks = check_tokens(tok, model.args.vocab_size, args.check_seqs, args.check_len)
    ref_logits = model(toks).astype(mx.float32)
    mx.eval(ref_logits)
    mod_check = module_check(enc, codec, names)
    for r in mod_check:
        print(f"#   module {r['stack']} rows={r['rows']} {r['path']}: rel {r['rel']:.2e} max {r['max_abs']:.2e}",
              flush=True)
    print("# speed: reconstructed (BF16 weights)", flush=True)
    speed_ref = speed(model, args)
    reconstructed_bytes = resident_weight_bytes(model)

    t0 = time.perf_counter()
    reports = []
    for li, lnames in sorted(by_layer(names).items()):
        reports.append(convert_to_compressed(model, {n: enc[n] for n in lnames}, group=False))
        for n in lnames:
            del enc[n]
        release()
    groups = group_projections(model)
    release()
    rep = merge_reports(reports)
    convert_s = time.perf_counter() - t0
    comp_logits = model(toks).astype(mx.float32)
    kl = logits_kl(ref_logits, comp_logits)
    print(f"# KL(reconstructed || compressed) = {kl['kl_mean']:.3e}, top-1 {kl['top1_agreement']:.4f}", flush=True)
    print("# speed: compressed (SUC codes)", flush=True)
    speed_comp = speed(model, args)
    comp_bytes = resident_weight_bytes(model)
    kinds: dict[str, int] = {}
    for r in rep["converted"]:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    del model, ref_logits, comp_logits
    release()

    q4 = None
    if not args.no_q4_baseline:
        model, _ = load(model_dir)
        apply_representation(model, "mlx_affine", bits=4)
        release()
        print("# speed: MLX affine 4-bit baseline", flush=True)
        q4 = {"resident_weight_bytes": resident_weight_bytes(model), "speed": speed(model, args)}
        del model
        release()

    tok_hash = hashlib.sha256(np.asarray(toks).astype(np.int32).tobytes()).hexdigest()
    cfg = {k: v for k, v in vars(args).items() if k not in ("out_dir", "quiet")}
    cfg_hash = hashlib.sha256(repr(sorted(cfg.items())).encode()).hexdigest()
    result = {
        "metadata": run_metadata(model=args.model, model_revision=revision, model_hash=model_hash(model_dir),
                                 dataset_hash=tok_hash, seed=SEED, config_hash=cfg_hash),
        "benchmark": "moe_e2e_check", "config": cfg, "codec": codec.name, "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__, "load_s": load_s,
        "encoding": enc_info, "module_check": mod_check,
        "conversion": {"seconds": convert_s, "kinds": kinds, "groups": len(groups),
                       "tensors_converted": rep["tensors_converted"], "bytes_before": rep["bytes_before"],
                       "bytes_after": rep["bytes_after"],
                       "converted_bpw_resident": rep["bytes_after"] * 8 / max(enc_info["encoded_params"], 1),
                       "n_residual": sum(r["n_residual"] for r in rep["converted"])},
        "resident_weight_bytes": {"bf16": bf16_bytes, "reconstructed": reconstructed_bytes, "compressed": comp_bytes,
                                  "mlx_q4": q4["resident_weight_bytes"] if q4 else None},
        "check": {"tokens": int(toks.size), "kl_reconstructed_vs_compressed": kl,
                  "note": "both models BF16; differences include BF16 rounding of decoded weights / activations"},
        "speed": {"reconstructed_bf16": speed_ref, "compressed": speed_comp, "mlx_q4": q4["speed"] if q4 else None},
    }
    name = f"moe_e2e_check_{Path(args.model).name}"
    path = write_result(result, args.out_dir, name)
    print(f"# wrote {path}", flush=True)


if __name__ == "__main__":
    main()
