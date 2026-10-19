#!/usr/bin/env python3
"""Bounded 8k/32k qualification of final binaries, not repeated speed claims.

Use only the six selected local files. Compare final logits, greedy continuation,
memory accounting and effective prefill chunks against the unchanged baseline.
Complete same-reference KL is a separate stored validation experiment.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import frontier_measure as fm
from efficiency_compare_m1 import measure, local_model

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', required=True, type=Path)
    ap.add_argument('--candidate', required=True, type=Path)
    ap.add_argument('--models', nargs='+', default=['q8','q27','q30','q38','ds','glm'])
    a=ap.parse_args(); old=a.baseline.expanduser(); new=a.candidate.expanduser()
    commit=(new/'GIT_COMMIT').read_text().strip()
    for model in a.models:
        local_model(model)
        for length in [8192,32768]:
            name=f'eff_closeout_long_{model}_{length}'
            done=[]
            for p in (fm.REPO/'experiments/results').glob('*_'+name+'.json'):
                d=json.loads(p.read_text())
                if d['metadata']['git_commit']==commit and d.get('passed'): done.append(p)
            if done:
                print('RESUME',name,flush=True); continue
            pair={}
            for label,build in ([('baseline',old),('candidate',new)] if length==8192 else [('candidate',new),('baseline',old)]):
                print('LONG_START',model,length,label,flush=True)
                with tempfile.TemporaryDirectory() as tmp:
                    os.environ['SUC_PREFILL_DUMP']=str(Path(tmp)/'last')
                    try:
                        record=measure(name+'_'+label,model,build,length,{},[],steps=8)
                        logits=np.load(Path(tmp)/'last.logits.npy')
                    finally: os.environ.pop('SUC_PREFILL_DUMP',None)
                pair[label]=(record,logits)
            left,right=pair['baseline'][0],pair['candidate'][0]
            identical=bool(np.array_equal(pair['baseline'][1],pair['candidate'][1]))
            tokens=left['run']['generated']==right['run']['generated']
            same_input=left['metadata']['dataset_hash']==right['metadata']['dataset_hash']
            good=all(r['summary']['returncode']==0 and not r['summary']['timed_out'] for r in [left,right])
            good=good and all(r['run']['prompt_tokens']==length for r in [left,right])
            good=good and same_input and identical and tokens
            x=pair['baseline'][1].astype(np.float64);y=pair['candidate'][1].astype(np.float64)
            def lp(z):
                z=z-z.max();return z-np.log(np.exp(z).sum())
            lx,ly=lp(x),lp(y)
            os.environ['SUC_BENCH_BUILD_COMMIT']=commit
            fm._write(name,str(local_model(model)),left['metadata']['dataset_hash'],
                      {'model':model,'prompt_tokens':length,'generation_steps':8,
                       'protocol':'one functional/memory pair, not repeated throughput',
                       'baseline_commit':(old/'GIT_COMMIT').read_text().strip()},
                      {'passed':good,'last_logits_identical':identical,'continuation_identical':tokens,
                       'last_position_kl':float((np.exp(lx)*(lx-ly)).sum()),
                       'baseline':left,'candidate':right})
            print('LONG_RESULT',model,length,'passed',good,flush=True)
            if not good: raise RuntimeError(f'Long qualification failed: {model}/{length}')
    print('ALLDONE_LONG_CLOSEOUT',flush=True)

if __name__=='__main__': main()
