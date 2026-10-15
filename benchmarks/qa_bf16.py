"""Answer the QA prompts (experiments/qa/prompts.json) with the BF16 model via mlx-lm: greedy, Qwen3 chat template
(enable_thinking per prompt), max tokens per prompt — the reference for the SUC side-by-side check.
Usage: python benchmarks/qa_bf16.py --model Qwen/Qwen3-8B --revision <sha> --out <file.json>"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--prompts", default="experiments/qa/prompts.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, tok = load(a.model, revision=a.revision)
    res = []
    for p in json.loads(Path(a.prompts).read_text()):
        prompt = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=p["think"])
        t0 = time.time()
        text = generate(model, tok, prompt=prompt, max_tokens=p["steps"], sampler=make_sampler(temp=0.0), verbose=False)
        res.append({**p, "text": text, "seconds": round(time.time() - t0, 1)})
        print(p["type"], len(text), flush=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
