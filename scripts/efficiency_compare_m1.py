#!/usr/bin/env python3
"""Interleaved M1 A/B tests using only selected local, complete SUC1 models."""
import argparse, hashlib, json, os, subprocess, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import frontier_measure as fm

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path.home()/"Projects/SUC-LM/experiments/artifacts/compress"
FILES={
 "q8":"compress_qwen3_8b_final/container/model.suc1",
 "q38":"compress_qwen3.8_flash_next_side8/container/qwen3.8_flash_next_side8/qwen3.8_flash_next_side8.suc1",
 "ds":"compress_deepseek_v4_flash_tcq/container/DeepSeek-V4-Flash_tcq/DeepSeek-V4-Flash_tcq.suc1",
 "glm":"compress_glm5.3_flash_tcq/container/glm5.3_flash_tcq/glm5.3_flash_tcq.suc1",
 "q27":"compress_qwen3.8_27b_side8/container/Qwen3.8_27b_side8/Qwen3.8_27B_side8.suc1",
 "q30":"compress_qwen3_30b_a3b_final/container/Qwen3_30b_a3b_final/Qwen3-30B-A3B.suc1"}
FLAGS=("SUC_NGRAM_THREADS","SUC_PREFIX_CACHE","SUC_PREFILL_QUERY_TILE","SUC_TCQ_PREFILL_HALF","SUC_SAMPLING_LEGACY","SUC_TCQ_REUSE")

def local_model(name):
 p=SOURCE/FILES[name]
 assert p.is_file() and not p.is_symlink() and p.stat().st_size>100_000_000,p
 assert p.open('rb').read(4)==b'SUC1',p
 fs=subprocess.check_output(['df','-P',str(p.resolve())],text=True).splitlines()[-1].split()[0]
 assert fs.startswith('/dev/'),f'Network filesystem excluded: {fs}'
 return p

def measure(name,model,build,length,flags,extra,steps=64):
 subprocess.run(['bash',str(ROOT/'scripts/studio_wait_idle.sh'),'--check'],check=True)
 for key in FLAGS:os.environ.pop(key,None)
 os.environ.update(flags)
 os.environ['SUC_BENCH_BUILD_COMMIT']=(build/'GIT_COMMIT').read_text().strip()
 a=SimpleNamespace(name=name,suc1=str(local_model(model)),build=str(build),long_prompt=length,
   prompt='The history of the printing press begins in the fifteenth century',steps=steps,timeout=0)
 before=set((ROOT/'experiments/results').glob('*_'+name+'.json'))
 rc=fm.cmd_generate(a,extra)
 if rc:raise RuntimeError(f'{name} failed: {rc}')
 files=set((ROOT/'experiments/results').glob('*_'+name+'.json'))-before
 assert len(files)==1
 result=json.loads(files.pop().read_text())
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True)
 ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--case',choices=['all','combined','ngram','sampling','chunks','attention','tcq','coverage','lookup'],default='all')
 a=ap.parse_args();baseline=a.baseline.expanduser();candidate=a.candidate.expanduser()
 cases={
  'combined':[(m,0 if m=='q8' else 2048,{},['--temperature','.7','--seed','42'] if m=='q8' else [],baseline,
               {},['--temperature','.7','--seed','42'] if m=='q8' else [],candidate)
              for m in ['q8','q38','ds','glm','q27','q30']],
  'ngram':[('q38',0,{},[],baseline,{'SUC_NGRAM_THREADS':'4'},[],candidate)],
  'sampling':[('q8',0,{'SUC_SAMPLING_LEGACY':'1'},['--temperature','.7','--seed','42'],candidate,{},['--temperature','.7','--seed','42'],candidate)],
  'chunks':[('q38',2048,{},['--chunk','64'],candidate,{},[],candidate)],
  'attention':[('q8',8192,{},[],candidate,{'SUC_PREFILL_QUERY_TILE':'1'},[],candidate)],
  'tcq':[(m,2048,{},[],candidate,{'SUC_TCQ_REUSE':'1'},[],candidate) for m in ['ds','glm']],
  'coverage':[(m,2048,{},[],baseline,{},[],candidate) for m in ['q27','q30']],
  'lookup':[('q8',2048,{},[],candidate,{},['--lookup-window','4'],candidate)]}
 for case,entries in cases.items():
  if a.case not in ['all',case]:continue
  for model,length,env_a,args_a,build_a,env_b,args_b,build_b in entries:
   for repeat in range(a.repeats+1):
    pair={}
    for variant in (['a','b'] if repeat%2==0 else ['b','a']):
     env,extra,build=(env_a,args_a,build_a) if variant=='a' else (env_b,args_b,build_b)
     name=f'eff_pair_{case}_{model}_{variant}_r{repeat}'
     with tempfile.TemporaryDirectory() as tmp:
      os.environ['SUC_PREFILL_DUMP']=str(Path(tmp)/'last')
      result=measure(name,model,build,length,env,extra)
      result['config']['experiment_environment']={key:os.environ.get(key) for key in FLAGS}
      logits=Path(tmp)/'last.logits.npy'
      pair[variant]=(result,np.load(logits).astype(np.float64) if logits.exists() else None)
     os.environ.pop('SUC_PREFILL_DUMP',None)
    left,right=pair['a'][0],pair['b'][0]
    quality={'generated_identical':left['run']['generated']==right['run']['generated']}
    if pair['a'][1] is not None and pair['b'][1] is not None:
     x,y=pair['a'][1],pair['b'][1]
     def lp(z):z=z-z.max();return z-np.log(np.exp(z).sum())
     lx,ly=lp(x),lp(y);quality.update(kl=float((np.exp(lx)*(lx-ly)).sum()),top1_equal=bool(x.argmax()==y.argmax()),bit_identical=bool(np.array_equal(x,y)))
    os.environ['SUC_BENCH_BUILD_COMMIT']=(candidate/'GIT_COMMIT').read_text().strip()
    fm._write(f'eff_comparison_{case}_{model}_r{repeat}',str(local_model(model)),left['metadata']['dataset_hash'],
      {'case':case,'model':model,'repeat':repeat,'warmup':repeat==0,'a_environment':env_a,'b_environment':env_b,'a_args':args_a,'b_args':args_b},
      {'a_summary':left['summary'],'b_summary':right['summary'],'a_run':left['run'],'b_run':right['run'],'quality':quality})
    print('PAIR',case,model,repeat,json.dumps(quality),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
