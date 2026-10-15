"""WikiText-2 PPL and KL vs the BF16 reference for the inference systems of a `suc eval` config, measured with the
data and reference logits of a finished `suc compress` run directory.

`suc compress` reports PPL / KL of the reconstructed weights (layer-streamed); `suc eval` / `suc perf` never measure
quality on WikiText-2. This script closes the gap for whole-model systems that fit in memory: SUC1 files through the
Python runtime (suc1_python, e.g. raw vs `suc repack`ed n-gram tables) and the MLX affine baseline (mlx_affine,
loaded lazily and converted module by module, so a BF16 checkpoint larger than RAM works when its 4-bit form fits).
suc1_cpp systems are skipped (the C++ tool decodes token by token; its agreement with the Python runtime is checked by
`suc eval`). Same protocol as the compress `evaluate` stage: the reference stage's WikiText-2 split and windows
(token sha256 checked against reference/reference.json), the first `kl_sequences` rows of the held-out validation
partition (calibrate/data.safetensors), KL / top-1 / delta-CE vs reference/ref_logits.safetensors (BF16 logits,
read row slice by row slice), token-weighted as suc.evaluation.reconstruct._kl. By default the loaded model is run
block-sequentially over all windows (suc.evaluation.streamed.streamed_quality with release=False, as the compress
evaluate stage; bounded temporaries); --whole-model runs model(window) instead.

  HF_HUB_OFFLINE=1 .venv/bin/python scripts/system_quality.py --config experiments/configs/eval_X.yaml \\
      --run-dir RUN_DIR [--systems a,b] [--max-tokens N]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from suc.analysis.sensitivity import logit_divergence  # noqa: E402
from suc.config import config_hash, config_to_dict, load_eval_config  # noqa: E402
from suc.evaluation import runners as rn  # noqa: E402
from suc.evaluation.data import load_wikitext2, tokenize  # noqa: E402
from suc.evaluation.perplexity import perplexity  # noqa: E402
from suc.evaluation.streamed import RefLogits, streamed_quality  # noqa: E402
from suc.provenance import run_metadata, seed_everything, write_result  # noqa: E402

_KL_KEYS = ("kl_mean", "kl_p95", "top1_agreement", "ce_ref", "ce_test")


def kl_vs_reference(model, kl_tokens: mx.array, ref: RefLogits, chunk: int) -> dict:
    """suc.evaluation.reconstruct._kl with the reference logits read `chunk` rows at a time."""
    acc, n_tok = dict.fromkeys(_KL_KEYS, 0.0), 0
    for s in range(0, kl_tokens.shape[0], chunk):
        tok = kl_tokens[s:s + chunk]
        r = ref.rows(s, s + tok.shape[0])
        test = model(tok).astype(r.dtype)
        d = logit_divergence(r[:, :-1], test[:, :-1], tok[:, 1:])
        n = tok.shape[0] * (tok.shape[1] - 1)
        for k in _KL_KEYS:
            acc[k] += d[k] * n
        n_tok += n
        del test, r
    out = {k: v / n_tok for k, v in acc.items()}
    out["delta_ce"] = out["ce_test"] - out["ce_ref"]
    out["tokens"] = n_tok
    return out


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="suc eval config (model + systems)")
    ap.add_argument("--run-dir", required=True, help="suc compress run directory (calibrate/, reference/)")
    ap.add_argument("--systems", default=None, help="comma-separated subset of system names")
    ap.add_argument("--max-tokens", type=int, default=None, help="truncate WikiText-2 (smoke tests; not reported)")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--output-dir", default="experiments/results")
    ap.add_argument("--whole-model", action="store_true", help="model(window) per window instead of block-sequential")
    ap.add_argument("--spill-dir", default="~/suc-local/quality_spill", help="hidden states of the sweep (~8 GB)")
    a = ap.parse_args(argv)
    cfg = load_eval_config(a.config)
    seed_everything(cfg.seed)
    run = Path(a.run_dir).expanduser()
    ref_rec = json.loads((run / "reference" / "reference.json").read_text())
    pick = set(a.systems.split(",")) if a.systems else None
    systems = [s for s in cfg.systems if (pick is None or s.name in pick) and s.kind != "suc1_cpp"]
    need_weights = any(s.kind in ("bf16", "mlx_affine") for s in systems)
    model_dir = rn.resolve_model_dir(cfg.model, need_weights)
    from mlx_lm.utils import load_tokenizer

    text, wiki_hash = load_wikitext2(ref_rec["wikitext_split"])
    ppl_tokens = tokenize(load_tokenizer(model_dir), text)
    tok_sha = hashlib.sha256(np.asarray(ppl_tokens, dtype=np.int32).astype("<i4").tobytes()).hexdigest()
    if tok_sha != ref_rec["ppl_tokens_sha256"]:
        raise RuntimeError("WikiText-2 tokens differ from those of the compress reference stage")
    with mx.stream(mx.cpu):
        kl_tokens = mx.load(str(run / "calibrate" / "data.safetensors"))["validation"][: ref_rec["kl_sequences"]]
        mx.eval(kl_tokens)
    ref_logits = RefLogits(run / "reference" / "ref_logits.safetensors")
    seq_len = int(ref_rec["quality"]["perplexity"]["seq_len"])
    out: dict[str, dict] = {}
    for sc in systems:
        print(f"[quality] system {sc.name} ({sc.kind})", flush=True)
        runner = rn.open_system(sc, model_dir)
        t0 = time.perf_counter()
        if a.whole_model:
            ppl = perplexity(runner.model, ppl_tokens, seq_len, a.max_tokens)
            kl = kl_vs_reference(runner.model, kl_tokens, ref_logits, a.chunk)
            streaming = None
        else:   # block-sequential over every window (the compress evaluate harness, == model(window)): one block's
            #     temporaries at a time (a whole 2048-token forward of the 180 B Python runtime needs ~40 GB extra)
            r = streamed_quality(runner.model, ppl_tokens=ppl_tokens, seq_len=seq_len, max_tokens=a.max_tokens,
                                 kl_tokens=kl_tokens, chunk=a.chunk, ref_logits=ref_logits, release=False,
                                 spill_dir=Path(a.spill_dir).expanduser() / sc.name if a.spill_dir else None)
            ppl, kl, streaming = r["perplexity"], r["kl"], r["streaming"]
        print(f"  PPL {ppl['ppl']:.4f}  KL {kl['kl_mean']:.4f}  top-1 {kl['top1_agreement']:.4f} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
        out[sc.name] = {"info": dict(runner.info), "perplexity": ppl, "kl": kl, "streaming": streaming,
                        "seconds": time.perf_counter() - t0, "peak_memory_bytes": int(mx.get_peak_memory())}
        runner.close()
        del runner
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
    md = run_metadata(model=cfg.model.id, model_revision=cfg.model.revision,
                      model_hash=rn.reference_model_hash(model_dir if need_weights else None, cfg.systems),
                      dataset_hash=f"wikitext:{wiki_hash};tokens:{tok_sha}", seed=cfg.seed, config_hash=config_hash(cfg))
    res = {"metadata": md, "kind": "system_quality", "config": config_to_dict(cfg), "config_path": a.config,
           "run_dir": str(run), "max_tokens": a.max_tokens,
           "reference": {"bf16_ppl": ref_rec["quality"]["perplexity"], "kl_sequences": ref_rec["kl_sequences"],
                         "kl_seq_len": ref_rec["kl_seq_len"], "wikitext_split": ref_rec["wikitext_split"]},
           "systems": out}
    p = write_result(res, a.output_dir, f"quality_{cfg.name}")
    print(f"wrote {p}")
    return p


if __name__ == "__main__":
    main()
