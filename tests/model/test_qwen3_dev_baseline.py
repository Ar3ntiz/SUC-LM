"""Dev-model Phase 0 baseline: Qwen3-0.6B, BF16 vs MLX affine 4-bit on WikiText-2 test (final-test set)."""
import json
import math
import time

import mlx.core as mx
import pytest

from suc.evaluation.baselines import apply_representation, logits_kl, size_report
from suc.evaluation.data import load_wikitext2, tokenize
from suc.evaluation.perplexity import perplexity
from suc.evaluation.speed import measure_generation, timed_load

MODEL_ID = "Qwen/Qwen3-0.6B"
SEQ_LEN, MAX_TOKENS = 1024, 8192


@pytest.mark.model
def test_qwen3_dev_bf16_vs_affine4():
    t_start = time.perf_counter()
    model, tok, load_s = timed_load(MODEL_ID)
    text, data_sha = load_wikitext2("test")
    tokens = tokenize(tok, text)[: MAX_TOKENS]
    window = tokens[: SEQ_LEN][None]

    apply_representation(model, "bf16")
    size_bf16 = size_report(model)
    ppl_bf16 = perplexity(model, tokens, seq_len=SEQ_LEN, max_tokens=MAX_TOKENS)
    ref_logits = model(window)
    mx.eval(ref_logits)
    speed_bf16 = measure_generation(model, prompt_len=128, decode_tokens=32, repeats=3)

    apply_representation(model, "mlx_affine", bits=4, group_size=64)
    size_q4 = size_report(model)
    ppl_q4 = perplexity(model, tokens, seq_len=SEQ_LEN, max_tokens=MAX_TOKENS)
    kl = logits_kl(ref_logits, model(window))
    speed_q4 = measure_generation(model, prompt_len=128, decode_tokens=32, repeats=3)

    summary = {
        "model": MODEL_ID, "dataset_sha256": data_sha, "load_s": load_s,
        "ppl_bf16": ppl_bf16["ppl"], "ppl_q4": ppl_q4["ppl"], "windows": ppl_bf16["windows"],
        "bpp_bf16": size_bf16["bpp"], "bpp_q4": size_q4["bpp"],
        "params": size_bf16["parameter_count"], "bytes_bf16": size_bf16["bytes_total"],
        "bytes_q4": size_q4["bytes_total"], "kl": kl,
        "bf16_speed": {k: speed_bf16[k] for k in ("ttft_s", "prefill_tok_s", "decode_tok_s", "peak_memory_bytes")},
        "q4_speed": {k: speed_q4[k] for k in ("ttft_s", "prefill_tok_s", "decode_tok_s", "peak_memory_bytes")},
        "wall_s": time.perf_counter() - t_start,
    }
    print(json.dumps(summary, indent=2))

    assert size_q4["parameter_count"] == size_bf16["parameter_count"]
    assert size_bf16["bpp"] == 16.0 and size_q4["bpp"] < 6.0
    assert math.isfinite(ppl_bf16["ppl"]) and math.isfinite(ppl_q4["ppl"])
    assert ppl_q4["ppl"] < 1.5 * ppl_bf16["ppl"]
    assert kl["kl_mean"] > 0
