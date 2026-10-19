#!/usr/bin/env python3
"""Consolidate final qualification evidence without substituting speed for KL."""
import argparse
import json
from pathlib import Path
import sys

MODELS=['q8','q27','q30','q38','ds','glm']
ROOT=Path(__file__).resolve().parents[1]

def latest(name):
    paths=sorted((ROOT/'experiments/results').glob('*_'+name+'.json'))
    return (json.loads(paths[-1].read_text()),paths[-1].name) if paths else (None,None)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--require-complete',action='store_true')
    a=ap.parse_args();missing=[]
    lines=['# Engine 1.4.7 final qualification','',
           'All tests use only the selected complete local files on M1 Ultra. Long-context checks are one bounded A/B pair per model/length, not repeated throughput or full-context BF16 evaluation. The repeated 2k/sampling matrix is in COMBINED_RESULTS.md.','',
           '## Context and memory checks','',
           '| Model | Tokens | Same final logits / continuation | KV MiB | States MiB | Arena + workspace MiB | Process footprint MiB |',
           '|---|---:|---|---:|---:|---:|---:|']
    references={}
    for m in MODELS:
        for n in [8192,32768]:
            d,name=latest(f'eff_closeout_long_{m}_{n}')
            if not d or not d.get('passed'):
                missing.append(f'long {m}/{n}');continue
            mem=d['candidate']['run']['memory'];v=lambda k:mem.get(k,0)/2**20
            old=d['baseline']['run']['memory']
            allocated=['mapped_file','copied_execution_arrays','kv_cache','states','arena','workspace']
            if any(mem.get(k,0)>old.get(k,0) for k in allocated):
                missing.append(f'review accounted allocation growth {m}/{n}')
            lines.append(f'| {m} | {n} | {d["last_logits_identical"]} / {d["continuation_identical"]} | {v("kv_cache"):.1f} | {v("states"):.1f} | {v("arena")+v("workspace"):.1f} | {v("phys_footprint"):.1f} |')
    lines+=['','Process footprint is not total unified/model memory. GPU-mapped file views, CPU n-gram tables, page cache, driver residency and shared OS memory have different accounting; do not add this column to file size or compare it to a whole-system RAM measurement. All raw runs retain detailed memory fields. No memory-policy or compressed-payload change was made.','',
            '## Real bundled-engine sessions','',
            '| Model | Cache-on/off outputs | Maximum replay-footprint growth MiB | Cancellation / reload |',
            '|---|---|---:|---|']
    for m in ['q8','q38']:
        d,name=latest('eff_session_closeout_'+m)
        if not d or not d.get('passed'):
            missing.append('session '+m);continue
        growth=max(max(x['footprints_after_replay'])-x['footprint_before_replay'] for x in d['results'].values())/2**20
        lines.append(f'| {m} | {d["choices_identical"]} | {growth:.2f} | passed |')
    lines+=['','Each session experiment covers strict aligned-prefix extension, divergent/equal/shorter histories, cancellation after streamed output, eight replay cycles and model unload/reload on the actual bundled engine. Cache-off is the oracle for the same requests. This is neither a visual GUI test nor a proof of zero leaks under every workload.','',
            '## Complete reference KL','',
            '| Model | Chunk | Scored positions | Mean KL before / after | Global p95 before / after | Raw logits identical |',
            '|---|---:|---:|---:|---:|---|']
    for m in MODELS:
        d,name=latest('eff_reference_kl_'+m)
        if not d or d['config'].get('smoke_subset') or not d.get('all_rows_bit_identical'):
            missing.append('full reference '+m);continue
        left=d['systems']['baseline']['kl'];right=d['systems']['candidate']['kl']
        references[m]=(d,name)
        lines.append(f'| {m} | {d["config"]["chunk"]} | {right["tokens"]} | {left["kl_mean"]:.8f} / {right["kl_mean"]:.8f} | {left["kl_p95"]:.8f} / {right["kl_p95"]:.8f} | True |')
    lines+=['','The same existing BF16 references, validation rows, BF16 logit casts and canonical float32 divergence/global-p95 aggregation are used within every pair. These runtime figures must not replace reconstructed-weight compression metrics or be advertised as improved compression. Seeded sampled strings are a separate CDF-order issue.','',
            '## Service profiles and final artifacts','']
    for m in ['ds','glm']:
        d,name=latest('eff_service_profile_'+m)
        if not d or not d.get('passed'):missing.append('actual service profile '+m)
        else:lines.append(f'- {m}: standalone and bundled-app defaults match the explicit historical-kernel CLI oracle at chunk {d["config"]["historical_service_chunk"]}.')
    d,name=latest('eff_reference_kl_service64_ds')
    if not d or not d.get('all_rows_bit_identical') or d['config'].get('smoke_subset'):
        missing.append('full DS service-64 reference')
    else:
        x=d['systems']['candidate']['kl'];lines.append(f'- DS service-64 arithmetic: all {x["tokens"]} validation positions match the historical TCQ path; mean BF16-reference KL {x["kl_mean"]:.8f}, with identical raw logits.')
    smoke=ROOT/'docs/optimization/RELEASE_SMOKE_1.4.7.json'
    continuity=ROOT/'docs/optimization/CLI_BINARY_CONTINUITY.json'
    distribution=ROOT/'SUCStudio/dist/CLI_SERVER_BUILD.json'
    if not smoke.exists() or not distribution.exists():missing.append('final artifact manifests/smoke')
    else:
        s=json.loads(smoke.read_text());p=json.loads(distribution.read_text())
        if not s.get('passed') or s['binary_commit']!=p['binary_commit']:missing.append('matching final artifact smoke')
        else:lines.append('- Final seven-check shipped-artifact smoke and standalone distribution refer to compiled commit `'+p['binary_commit']+'`. Shared C++/Metal input hash: `'+p['engine_input_sha256']+'`.')
    if not continuity.exists():missing.append('CLI continuity')
    else:
        c=json.loads(continuity.read_text());same=c['suc1-generate']['identical']
        lines.append(f'- The final generation CLI is byte-for-byte identical to the binary used for the six-model matrix and long-context qualification: {same}.')
        if not same:missing.append('resolve final CLI binary continuity before applying prior timing evidence')
    lines+=['','Manual GUI acceptance of build 28 on physical Intel/other Apple Silicon machines is not claimed. The unchanged UI already passed native and simulated remote-only Swift suites; final app/helpers retain Universal slices and the local engine remains ARM-only. Intel uses remote inference. MTP, new compression, online model-card publication, installation and main-branch merge are outside this closeout.']
    lines+=['','Status: '+('PENDING — '+', '.join(missing) if missing else 'QUALIFIED within the scope and limits above.')]
    if len(references)==6:
        rlines=['# Full same-reference runtime KL — engine 1.4.7','',
                'Same stored BF16 reference and validation rows within every pair; 16x512 tokens, final position excluded (8176 scored positions). Raw float logits must match; scoring casts to BF16 then uses canonical float32 divergence and global p95. These are C++ runtime qualification values, separate from historical reconstructed-weight compression metrics.','',
                '| Model | Effective chunk | Mean KL before / after | Global p95 before / after | Raw logits identical |',
                '|---|---:|---:|---:|---|']
        for m in MODELS:
            d,name=references[m];x=d['systems']['baseline']['kl'];y=d['systems']['candidate']['kl']
            rlines.append(f'| {m} | {d["config"]["chunk"]} | {x["kl_mean"]:.8f} / {y["kl_mean"]:.8f} | {x["kl_p95"]:.8f} / {y["kl_p95"]:.8f} | True |')
        rlines+=['','The first Qwen chunk-64 command-line comparison used mismatched effective chunks and remains diagnostic only. It is excluded here. DeepSeek service chunk 64 is a separate full-profile check in FINAL_QUALIFICATION_1.4.7.md. CLI/service profiles must not be silently mixed.']
        for m,(d,name) in references.items():rlines.append(f'- {m}: `{name}`; compiled candidate `{d["metadata"]["git_commit"]}`.')
        (ROOT/'docs/optimization/REFERENCE_KL.md').write_text('\n'.join(rlines)+'\n')
    out=ROOT/'docs/optimization/FINAL_QUALIFICATION_1.4.7.md';out.write_text('\n'.join(lines)+'\n')
    print(out);print('PENDING' if missing else 'ALL_GATES_PASSED',missing,flush=True)
    return int(bool(missing) and a.require_complete)

if __name__=='__main__':sys.exit(main())
