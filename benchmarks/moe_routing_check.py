"""MoE routing check: does executing from SUC codes flip the router's top-k expert selection, and is that a SUC
issue or the model's intrinsic sensitivity to BF16 rounding?

Context: Qwen3-30B-A3B from codes (SucSwitchLinear) vs the same encodings decoded into the BF16 model gave
KL 0.032 / top-1 0.930 over 256 tokens (benchmarks/moe_e2e_check.py), ten times the dense 8B check, while the module
check in float32 is ~1e-7. Hypothesis: the compressed path is a different, equally valid BF16 rounding of the same
math (no BF16 rounding of the decoded weights, BF16 rounding of the rotated activations instead); the top-k router
is discontinuous, so tokens whose k-th / (k+1)-th router logits are nearly tied flip expert sets, and flips
compound over the layers.

Not a test. Usage (Mac Studio, sequentially, nothing else running; ≈ 55 min, peak ≈ 70 GB — see docs/benchmarks.md):
  HF_HUB_OFFLINE=1 nohup .venv/bin/python benchmarks/moe_routing_check.py --model Qwen/Qwen3-30B-A3B \
      --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 > logs/moe_routing_check.log 2>&1 &
  Local validation on a random tiny Qwen3-MoE (no download): --model tiny-moe [--tiny-layers 8] [--dtype float32]
Options: [--codec vq_g4_k256|add3_g8_k256] [--kmeans-iters 6] [--fit-blocks 65536] [--check-seqs 4]
         [--check-len 64] [--embed q4|bf16] [--controls repeat,sr] [--sr-fraction 1.0] [--trace-dir DIR]
         [--out-dir experiments/results]

Models compared (same token ids, same encodings — moe_e2e_check's encoder, seeds and check text, so the
reconstructed/compressed pair is the one behind the 0.032):
  A  "reconstructed": every eligible tensor's decode rounded to the model dtype (nearest) — the reference.
  A' "repeat"  (control): A run again — determinism floor (must be bit-identical).
  C  "sr"      (control): the same decodes rounded to BF16 *stochastically* (a --sr-fraction share of the weights;
     the rest nearest): another valid BF16 model of the same exact weights, no SUC code involved. Its flip rate is
     the model's intrinsic sensitivity to BF16-rounding-sized perturbations. Magnitude: full SR differs from
     nearest on 1/4 of the weights by one ulp (E δ² = ulp²/4); B's weights differ from A's by the nearest-rounding
     error (ulp²/12) plus B's BF16 rounding of the rotated activations — compare the teacher-forced local errors of
     B and C to see the perturbation sizes actually matched (--sr-fraction scales C's).
  B  "compressed": convert_to_compressed (SucLinear / SucSwitchLinear, as in moe_e2e_check).
One model at a time (D8): A's trace (per-layer inputs / outputs and router logits, float32 numpy, a few hundred MB
for 30B at 256 tokens) is kept; C is A's weights re-decoded in place; B is A converted in place. Peak memory is the
BF16 model + encodings, as in moe_e2e_check.

Per model X vs A, per layer:
  free run       — X on its own trajectory: hidden-state relative error at each layer input / output, router logit
                   difference, top-k set agreement (fraction of tokens with the identical expert set), mean set
                   overlap, per-slot agreement (experts ranked by router score), router margin (k-th minus
                   (k+1)-th router logit of A, float32) for flipped vs unflipped tokens, the share of flipped tokens
                   whose A-margin is below that token's max |router logit difference| (a flip explained by the
                   perturbation), and the share of A's tokens whose k-th / (k+1)-th BF16 router probabilities are
                   exactly tied;
  teacher-forced — every layer of X fed A's input of that layer: the error / flips each layer injects by itself
                   (separates injection from compounding);
  final          — KL(A || X) and top-1 agreement of the logits.
The traced forward re-implements the Qwen3-MoE model loop (embed → layers → norm → head) to read the router input
and logits; it is checked bit-exact against model(tokens) for A.
Result JSON (suc.provenance.write_result, standard metadata) in --out-dir: <ts>_moe_routing_check_<model>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
import time
import zlib
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from suc.analysis.sensitivity import get_weight, set_weight  # noqa: E402
from suc.evaluation.baselines import logits_kl  # noqa: E402
from suc.io.hashing import model_hash  # noqa: E402
from suc.provenance import run_metadata, seed_everything, write_result  # noqa: E402
from suc.quant.codecs import make_codec  # noqa: E402
from suc.runtime.compressed_model import (convert_to_compressed, expert_stack, group_projections,  # noqa: E402
                                          resident_weight_bytes)


def _load_e2e():
    spec = importlib.util.spec_from_file_location("moe_e2e_check", ROOT / "benchmarks" / "moe_e2e_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e2e = _load_e2e()
SEED = e2e.SEED
TINY = "tiny-moe"


# ------------------------------------------------------------------------------------------------ tracing

def _is_moe(layer) -> bool:
    return "mlp" in layer and "gate" in layer["mlp"] and hasattr(layer["mlp"], "top_k")


def router_topk(logits: mx.array, k: int) -> mx.array:
    """Expert indices exactly as mlx-lm's Qwen3MoeSparseMoeBlock selects them (precise softmax in the logits'
    dtype, argpartition), ranked by router score (descending)."""
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    order = mx.argsort(-scores.astype(mx.float32), axis=-1)
    return mx.take_along_axis(inds, order, axis=-1)


def traced_forward(model, toks: mx.array, teacher: list[np.ndarray] | None = None) -> dict:
    """Qwen3-MoE forward (no cache) recording every layer's input / output and, for MoE layers, the router logits
    (in the model dtype, as the block computes them) and top-k. teacher: layer l reads teacher[l] instead of the
    previous layer's output (teacher-forced). Returns float32 / int numpy arrays (lists over layers)."""
    from mlx_lm.models.base import create_attention_mask
    inner = model.model
    h = inner.embed_tokens(toks)
    mask = create_attention_mask(h, None)
    ins, outs, gl, tk = [], [], {}, {}
    for li, layer in enumerate(inner.layers):
        x = h if teacher is None else mx.array(teacher[li]).astype(h.dtype)
        r = layer.self_attn(layer.input_layernorm(x), mask, None)
        hh = x + r
        xn = layer.post_attention_layernorm(hh)
        if _is_moe(layer):
            g = layer.mlp.gate(xn)
            gl[li], tk[li] = g, router_topk(g, layer.mlp.top_k)
        h = hh + layer.mlp(xn)
        ins.append(x)
        outs.append(h)
        mx.eval(h, *(([gl[li], tk[li]]) if li in gl else []))
    out = inner.norm(h)
    logits = inner.embed_tokens.as_linear(out) if model.args.tie_word_embeddings else model.lm_head(out)
    mx.eval(logits)
    f32 = lambda a: np.asarray(a.astype(mx.float32))  # noqa: E731
    return {"logits": f32(logits), "inputs": [f32(a) for a in ins], "outputs": [f32(a) for a in outs],
            "router_logits": {li: f32(a) for li, a in gl.items()},
            # A's BF16 router probabilities at the k-th / (k+1)-th rank, to count exact ties
            "router_probs_dtype": {li: _boundary_probs(a, model.model.layers[li].mlp.top_k) for li, a in gl.items()},
            "topk": {li: np.asarray(a).astype(np.int64) for li, a in tk.items()}}


def _boundary_probs(logits: mx.array, k: int) -> np.ndarray:
    """(..., 2): the k-th and (k+1)-th largest router probabilities in the model dtype (as float32)."""
    s = mx.sort(mx.softmax(logits, axis=-1, precise=True).astype(mx.float32), axis=-1)
    return np.asarray(mx.stack([s[..., -k], s[..., -k - 1]], axis=-1))


# ------------------------------------------------------------------------------------------------ metrics

def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-30))


def _stats(v: np.ndarray) -> dict | None:
    if v.size == 0:
        return None
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.quantile(v, 0.1)), "p90": float(np.quantile(v, 0.9))}


def compare_layer(ref: dict, test: dict, li: int, k: int | None, *, local: bool) -> dict:
    """Metrics of layer li, test vs ref (test teacher-forced on ref's inputs when local)."""
    row = {"layer": li, "rel_err_out": _rel(test["outputs"][li], ref["outputs"][li])}
    if not local:
        row["rel_err_in"] = _rel(test["inputs"][li], ref["inputs"][li])
    if k is None or li not in ref["topk"]:
        return row
    ta, tt = ref["topk"][li].reshape(-1, k), test["topk"][li].reshape(-1, k)
    la = ref["router_logits"][li].reshape(ta.shape[0], -1)
    lt = test["router_logits"][li].reshape(ta.shape[0], -1)
    same_set = np.all(np.sort(ta, axis=1) == np.sort(tt, axis=1), axis=1)
    overlap = np.array([len(set(a) & set(b)) for a, b in zip(ta.tolist(), tt.tolist())]) / k
    srt = -np.sort(-la, axis=1)
    margin = srt[:, k - 1] - srt[:, k]
    dmax = np.abs(lt - la).max(axis=1)
    flipped = ~same_set
    pb = ref["router_probs_dtype"][li].reshape(-1, 2)
    row.update({
        "tokens": int(ta.shape[0]), "set_agreement": float(same_set.mean()), "flipped_tokens": int(flipped.sum()),
        "mean_overlap": float(overlap.mean()), "slot_agreement": [float(v) for v in (ta == tt).mean(axis=0)],
        "router_logit_absdiff_mean": float(np.abs(lt - la).mean()), "router_logit_absdiff_max": float(dmax.max()),
        "margin_flipped": _stats(margin[flipped]), "margin_unflipped": _stats(margin[~flipped]),
        "flips_explained_by_perturbation": float((margin[flipped] <= dmax[flipped]).mean()) if flipped.any() else None,
        "ref_boundary_ties": float((pb[:, 0] == pb[:, 1]).mean()),
        "ref_margin_below_test_perturbation": float((margin <= dmax).mean()),
    })
    return row


def compare(ref: dict, test: dict, model, *, local: bool) -> dict:
    layers = model.model.layers
    rows = [compare_layer(ref, test, li, layers[li].mlp.top_k if _is_moe(layers[li]) else None, local=local)
            for li in range(len(layers))]
    moe = [r for r in rows if "set_agreement" in r]
    summ = {"set_agreement_mean": float(np.mean([r["set_agreement"] for r in moe])) if moe else None,
            "set_agreement_min": float(np.min([r["set_agreement"] for r in moe])) if moe else None,
            "flipped_token_layers": int(sum(r["flipped_tokens"] for r in moe)),
            "token_layers": int(sum(r["tokens"] for r in moe)),
            "first_layer_with_flip": next((r["layer"] for r in moe if r["flipped_tokens"]), None),
            "rel_err_out_first": rows[0]["rel_err_out"], "rel_err_out_last": rows[-1]["rel_err_out"],
            "rel_err_out_mean": float(np.mean([r["rel_err_out"] for r in rows]))}
    fl = [r["flips_explained_by_perturbation"] for r in moe if r["flips_explained_by_perturbation"] is not None]
    summ["flips_explained_by_perturbation_mean"] = float(np.mean(fl)) if fl else None
    med = lambda key: [r[key]["median"] for r in moe if r[key]]  # noqa: E731
    summ["margin_flipped_median_of_layers"] = float(np.median(med("margin_flipped"))) if med("margin_flipped") \
        else None
    summ["margin_unflipped_median_of_layers"] = float(np.median(med("margin_unflipped"))) \
        if med("margin_unflipped") else None
    out = {"summary": summ, "layers": rows}
    if not local:
        out["final"] = logits_kl(mx.array(ref["logits"]), mx.array(test["logits"]))
        out["final_bit_identical"] = bool(np.array_equal(ref["logits"], test["logits"]))
    return out


def evaluate(name: str, model, toks, ref: dict | None, verbose: bool) -> tuple[dict, dict | None]:
    t0 = time.perf_counter()
    tr = traced_forward(model, toks)
    res = None
    if ref is not None:
        res = {"free": compare(ref, tr, model, local=False),
               "teacher_forced": compare(ref, traced_forward(model, toks, teacher=ref["inputs"]), model, local=True),
               "seconds": time.perf_counter() - t0}
        if verbose:
            report(name, res)
    return tr, res


def report(name: str, res: dict) -> None:
    f, t = res["free"], res["teacher_forced"]
    print(f"# {name}: KL {f['final']['kl_mean']:.3e} top-1 {f['final']['top1_agreement']:.4f} | free: set agree "
          f"mean {f['summary']['set_agreement_mean']} first flip L{f['summary']['first_layer_with_flip']} | "
          f"teacher-forced: set agree mean {t['summary']['set_agreement_mean']}, local rel err mean "
          f"{t['summary']['rel_err_out_mean']:.2e}", flush=True)
    print("#   layer  rel_in    rel_out   setagree  local_rel  local_set  margin_flip  margin_ok  ties", flush=True)
    for a, b in zip(f["layers"], t["layers"]):
        if "set_agreement" not in a:
            print(f"#   {a['layer']:5d}  {a['rel_err_in']:.2e}  {a['rel_err_out']:.2e}  (dense)  "
                  f"{b['rel_err_out']:.2e}", flush=True)
            continue
        mfl = a["margin_flipped"]["median"] if a["margin_flipped"] else float("nan")
        mok = a["margin_unflipped"]["median"] if a["margin_unflipped"] else float("nan")
        print(f"#   {a['layer']:5d}  {a['rel_err_in']:.2e}  {a['rel_err_out']:.2e}  {a['set_agreement']:.4f}    "
              f"{b['rel_err_out']:.2e}   {b['set_agreement']:.4f}     {mfl:.3e}    {mok:.3e}  "
              f"{a['ref_boundary_ties']:.3f}", flush=True)


# ------------------------------------------------------------------------------------------------ control

def stochastic_round_bf16(w: mx.array, key: mx.array, fraction: float = 1.0) -> mx.array:
    """BF16 stochastic rounding of float32 w (round magnitude up with probability = the dropped fraction of an
    ulp) on a `fraction` share of the elements (the rest round to nearest). Result is exactly representable."""
    w = w.astype(mx.float32)
    k1, k2 = mx.random.split(key)
    u = w.view(mx.uint32)
    r = mx.random.randint(0, 1 << 16, w.shape, dtype=mx.int32, key=k1).astype(mx.uint32)
    sr = ((u + r) & mx.array(0xFFFF0000, dtype=mx.uint32)).view(mx.float32).astype(mx.bfloat16)
    if fraction >= 1.0:
        return sr
    pick = mx.random.uniform(shape=w.shape, key=k2) < fraction
    return mx.where(pick, sr, w.astype(mx.bfloat16))


def redecode_stochastic(model, enc: dict, codec, fraction: float, verbose: bool) -> None:
    """Rewrite every encoded tensor with its decode rounded stochastically to BF16 (expert stacks once per layer)."""
    t0 = time.perf_counter()
    for li, lnames in sorted(e2e.by_layer(list(enc)).items()):
        pending: dict[str, dict[int, mx.array]] = {}
        for n in lnames:
            key = mx.random.key(SEED + 7 + zlib.crc32(n.encode()))
            w = stochastic_round_bf16(codec.decode(enc[n]), key, fraction)
            mx.eval(w)
            if (st := expert_stack(n)) is not None:
                pending.setdefault(st[0], {})[st[1]] = w
            else:
                set_weight(model, n, w)
        for stack, rows in pending.items():
            cur = get_weight(model, stack)
            set_weight(model, stack, mx.stack([rows.get(r, cur[r]) for r in range(cur.shape[0])]))
            mx.eval(get_weight(model, stack))
        del pending
        e2e.release()
    if verbose:
        print(f"#   stochastic re-decode: {time.perf_counter() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------ main

def build_tiny(layers: int, dtype: mx.Dtype):
    sys.path.insert(0, str(ROOT / "tests" / "unit"))
    from tiny_models import tiny_qwen3_moe
    m = tiny_qwen3_moe(seed=SEED % 1000, experts=16, top_k=4, layers=layers, vocab=256)
    m.set_dtype(dtype)
    mx.eval(m.parameters())
    return m


def tiny_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(tree_flatten(model.parameters()), key=lambda kv: kv[0]):
        h.update(k.encode())
        h.update(np.asarray(v.astype(mx.float32)).tobytes())
    return h.hexdigest()


def run(args) -> dict:
    seed_everything(SEED)
    verbose = not args.quiet
    dtype = getattr(mx, args.dtype)
    if args.model == TINY:
        model, tok, revision = build_tiny(args.tiny_layers, dtype), None, "random"
        mhash = tiny_hash(model)
    else:
        model_dir = e2e.resolve(args.model, args.revision)
        revision = model_dir.name if model_dir.parent.name == "snapshots" else args.revision
        model, tok = e2e.load(model_dir)
        if dtype != mx.bfloat16:
            model.set_dtype(dtype)
            mx.eval(model.parameters())
        mhash = model_hash(model_dir)
    if args.embed == "q4":
        e2e.quantize_embedding(model)
    codec = make_codec(dict(e2e.CODECS[args.codec], kmeans_iters=args.kmeans_iters, fit_blocks=args.fit_blocks))
    names = e2e.eligible(model, None)
    print(f"# {args.model}: {len(names)} tensors to encode ({codec.name}), dtype {args.dtype}", flush=True)
    enc, enc_info = e2e.encode_and_reconstruct(model, names, codec, not args.no_rotation_groups, verbose)
    toks = e2e.check_tokens(tok, model.args.vocab_size, args.check_seqs, args.check_len)

    results: dict[str, dict] = {}
    ref, _ = evaluate("reconstructed", model, toks, None, verbose)
    direct = np.asarray(model(toks).astype(mx.float32))
    traced_exact = bool(np.array_equal(direct, ref["logits"]))
    if not traced_exact:
        raise RuntimeError("traced forward does not reproduce model(tokens) bit-exactly; the trace is invalid")
    del direct
    if "repeat" in args.controls:
        _, results["repeat"] = evaluate("control: repeat", model, toks, ref, verbose)
    if "sr" in args.controls:
        if dtype != mx.bfloat16:
            print("# control sr skipped: only defined for bfloat16 models", flush=True)
        else:
            redecode_stochastic(model, enc, codec, args.sr_fraction, verbose)
            _, results["sr"] = evaluate(f"control: stochastic BF16 rounding (fraction {args.sr_fraction})",
                                        model, toks, ref, verbose)
    if args.trace_dir:
        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(args.trace_dir) / "reconstructed.npz", logits=ref["logits"],
                 inputs=np.stack(ref["inputs"]), outputs=np.stack(ref["outputs"]),
                 **{f"router_logits_{k}": v for k, v in ref["router_logits"].items()},
                 **{f"topk_{k}": v for k, v in ref["topk"].items()})

    t0 = time.perf_counter()
    for li, lnames in sorted(e2e.by_layer(names).items()):
        convert_to_compressed(model, {n: enc[n] for n in lnames}, group=False)
        for n in lnames:
            del enc[n]
        e2e.release()
    group_projections(model)
    e2e.release()
    convert_s = time.perf_counter() - t0
    _, results["compressed"] = evaluate("compressed (SUC codes)", model, toks, ref, verbose)
    comp_bytes = resident_weight_bytes(model)

    tok_hash = hashlib.sha256(np.asarray(toks).astype(np.int32).tobytes()).hexdigest()
    cfg = {k: v for k, v in vars(args).items() if k not in ("out_dir", "quiet", "trace_dir")}
    cfg_hash = hashlib.sha256(repr(sorted(cfg.items())).encode()).hexdigest()
    return {
        "metadata": run_metadata(model=args.model, model_revision=revision, model_hash=mhash, dataset_hash=tok_hash,
                                 seed=SEED, config_hash=cfg_hash, repo_dir=ROOT),
        "benchmark": "moe_routing_check", "config": cfg, "codec": codec.name, "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__, "encoding": enc_info, "convert_s": convert_s,
        "resident_weight_bytes_compressed": comp_bytes, "tokens": int(toks.size),
        "traced_forward_bit_exact": traced_exact,
        "reference": "reconstructed (decodes rounded to nearest in the model dtype)",
        "comparisons": results,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B", help=f"HF id / path, or '{TINY}' (random, local)")
    ap.add_argument("--revision", default="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
    ap.add_argument("--tiny-layers", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--codec", default="vq_g4_k256", choices=sorted(e2e.CODECS))
    ap.add_argument("--kmeans-iters", type=int, default=6)
    ap.add_argument("--fit-blocks", type=int, default=65536)
    ap.add_argument("--check-seqs", type=int, default=4)
    ap.add_argument("--check-len", type=int, default=64)
    ap.add_argument("--embed", default="q4", choices=["q4", "bf16"])
    ap.add_argument("--controls", default="repeat,sr")
    ap.add_argument("--sr-fraction", type=float, default=1.0)
    ap.add_argument("--no-rotation-groups", action="store_true")
    ap.add_argument("--trace-dir", default=None, help="also save the reference trace (npz) there")
    ap.add_argument("--out-dir", default="experiments/results")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    args.controls = tuple(c for c in args.controls.split(",") if c)
    unknown = set(args.controls) - {"repeat", "sr"}
    if unknown:
        ap.error(f"unknown controls {sorted(unknown)}")
    return args


def main(argv=None) -> Path:
    args = parse_args(argv)
    result = run(args)
    path = write_result(result, args.out_dir, f"moe_routing_check_{Path(args.model).name}")
    print(f"# wrote {path}", flush=True)
    return path


if __name__ == "__main__":
    main()
