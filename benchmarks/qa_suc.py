"""Answer the QA prompts (experiments/qa/prompts.json) with the C++ SUC runtime (greedy, Qwen3 chat template). Usage: python benchmarks/qa_suc.py MODEL.suc1 tokenizer.json OUT.json"""
import json, re, subprocess, sys, time
from pathlib import Path
d = Path.home() / "suc-local"  # C++ build at d / "cpp-build"; prompts from experiments/qa/prompts.json
model, tok, out = sys.argv[1], sys.argv[2], sys.argv[3]
res = []
for p in json.loads(Path("experiments/qa/prompts.json").read_text()):
    cmd = [str(d / "cpp-build/suc1-generate"), model, "--tokenizer", tok, "--chat", "--stop-eos",
           "--steps", str(p["steps"]), "--prompt", p["prompt"]] + ([] if p["think"] else ["--no-think"])
    t0 = time.time(); o = subprocess.run(cmd, capture_output=True, text=True).stdout
    text = o.split("\ntext: ", 1)[1].rstrip() if "\ntext: " in o else "<no output>"
    m = re.search(r"decode (\d+) tokens: median ([\d.]+) ms/token", o)
    res.append({**p, "text": text, "decode_tokens": int(m[1]) if m else None, "ms_per_token": float(m[2]) if m else None,
                "seconds": round(time.time() - t0, 1)})
    print(p["type"], res[-1]["decode_tokens"], res[-1]["ms_per_token"], flush=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
