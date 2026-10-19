#!/usr/bin/env python3
"""Summarize qualified paired runs; never mix obsolete candidate binaries or warmups."""
import argparse,json,statistics
from pathlib import Path

EXPECTED={('ngram','q38'):3,('sampling','q8'):3,('chunks','q38'):2,('attention','q8'):2,
          ('tcq','ds'):2,('tcq','glm'):2,('lookup','q8'):2,('coverage','q27'):1,('coverage','q30'):1}
def med(values):return statistics.median(values)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);ap.add_argument('--results',type=Path,default=Path('experiments/results'));ap.add_argument('--output',type=Path);ap.add_argument('--require-complete',action='store_true');a=ap.parse_args()
 pairs={};caches={}
 for path in sorted(a.results.glob('*eff_*.json')):
  d=json.loads(path.read_text())
  if d.get('metadata',{}).get('git_commit')!=a.commit:continue
  kind=d.get('kind','');cfg=d.get('config',{})
  if kind.startswith('eff_comparison_') and not cfg.get('warmup'):
   pairs[(cfg['case'],cfg['model'],cfg['repeat'])]=d
  elif kind.startswith('eff_live_cache_'):caches[kind.removeprefix('eff_live_cache_')]=d
 lines=['# M1 inference efficiency results','',f'Candidate binary commit: `{a.commit}`. Warm-up pairs and obsolete candidate builds excluded.','',
 '| Experiment | Model | Measured pairs | Prefill tok/s A → B | Decode tok/s A → B | Token identity | Last-position KL max |',
 '|---|---|---:|---:|---:|---|---:|']
 missing=[]
 for (case,model),minimum in EXPECTED.items():
  rows=[d for (c,m,_),d in pairs.items() if (c,m)==(case,model)]
  if len(rows)<minimum:missing.append(f'{case}/{model}: {len(rows)}/{minimum}')
  if not rows:continue
  pa=med([d['a_summary']['prefill_tok_s'] for d in rows]);pb=med([d['b_summary']['prefill_tok_s'] for d in rows])
  da=med([d['a_summary']['decode_tok_s_median'] for d in rows]);db=med([d['b_summary']['decode_tok_s_median'] for d in rows])
  identity=all(d['quality']['generated_identical'] for d in rows)
  kl=max([d['quality'].get('kl',0) for d in rows])
  lines.append(f'| {case} | {model} | {len(rows)} | {pa:.1f} → {pb:.1f} | {da:.1f} → {db:.1f} | {identity} | {kl:.6g} |')
  if case=='ngram':
   lines += [f'<!-- ngram: B workers={[d["b_run"].get("ngram_cpu_workers") for d in rows]}, tasks={[d["b_run"].get("ngram_head_tasks") for d in rows]} -->']
 lines += ['','## Prefix reuse','', '| Model | New-request tokens | Cached tokens | Request wall seconds off → on | Speedup | First-token identity |','|---|---:|---:|---:|---:|---|']
 for model in ['q8','q38']:
  d=caches.get(model)
  if not d:missing.append('cache/'+model);continue
  left=[r for r in d['records']['0'] if not r['warmup']];right=[r for r in d['records']['1'] if not r['warmup']]
  x=med([r['wall_s'] for r in left]);y=med([r['wall_s'] for r in right])
  lines.append(f'| {model} | 16 | 2048 | {x:.4f} → {y:.4f} | {x/y:.2f}× | {d["choices_identical"]} |')
 lines += ['','## Interpretation and limits','',
 '- A/B order alternates; each experiment excludes its first warm-up pair. No forced OS cache purge or exclusive machine reservation was performed.',
 '- Sampling compares historical sorted and linear categorical CDF order. Seeded strings may differ while the probability distribution is preserved; greedy sampling is unchanged.',
 '- Chunk comparisons reproduce the old application chunk of 64 versus the shared policy. Fast batched arithmetic can differ across chunk boundaries.',
 '- Query tiling and TCQ reuse remain opt-in. Half-operand TCQ failed the synthetic quality budget and is rejected.',
 '- Lookup verification is a greedy prompt-lookup prototype, not MTP. Amortized generation timing is not per-token latency. Hybrid graphs/SSD experts/sampled speculative generation are explicitly unsupported.',
 '- Cache results cover an exact, chunk-aligned live prefix. Divergent history, equal/shorter prompts, cancellation and reload invalidate or replay safely. This is not an LRU snapshot cache for every chat.',
 '- Last-position KL and short continuations do not replace a full held-out quality/hard-suite qualification. No new model compression or production/main deployment was performed.',
 '- Real MTP remains a separate pending milestone; see MTP_PLAN.md.']
 if missing:lines += ['','Pending results: '+', '.join(missing)]
 output='\n'.join(lines)+'\n'
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(output)
 print(output)
 return int(bool(missing) and a.require_complete)
if __name__=='__main__':raise SystemExit(main())
