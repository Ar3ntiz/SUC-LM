#!/usr/bin/env python3
"""Measure exact live-prefix hits on selected local real models, sequentially."""
import argparse, hashlib, json, os, subprocess, tempfile, time, urllib.request
from pathlib import Path
import frontier_measure as fm
from efficiency_compare_m1 import local_model

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--build',type=Path,required=True);ap.add_argument('--model',choices=['q8','q38'],default='q8');ap.add_argument('--repeats',type=int,default=3);a=ap.parse_args()
 build=a.build.expanduser();model=local_model(a.model);prompt=fm._long_prompt_ids(build,str(model),2048);extended=prompt+[17,29,31,47]*4
 os.environ['SUC_BENCH_BUILD_COMMIT']=(build/'GIT_COMMIT').read_text().strip()
 records={}
 for enabled in ['0','1']:
  subprocess.run(['bash',str(fm.REPO/'scripts/studio_wait_idle.sh'),'--check'],check=True)
  with tempfile.TemporaryDirectory() as tmp:
   with open(Path(tmp)/'server.log','w') as log:
    process=subprocess.Popen([str(build/'suc1-server'),str(model),'--port','0','--max-context','4096','--quiet'],stdout=subprocess.PIPE,stderr=log,text=True,env={**os.environ,'SUC_PREFIX_CACHE':enabled})
    try:
     line=process.stdout.readline();assert line.startswith('listening on '),(line,(Path(tmp)/'server.log').read_text())
     url=line.split()[-1]+'/v1/completions'
     def ask(tokens):
      request=urllib.request.Request(url,data=json.dumps({'prompt':tokens,'max_tokens':1,'temperature':0}).encode(),headers={'Content-Type':'application/json'})
      start=time.perf_counter()
      with urllib.request.urlopen(request) as response:body=json.load(response)
      return {'wall_s':time.perf_counter()-start,'response':body}
     records[enabled]=[]
     for repeat in range(a.repeats+1):
      ask(prompt);result=ask(extended);result['warmup']=repeat==0;records[enabled].append(result)
      cached=result['response']['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)
      assert cached==(2048 if enabled=='1' else 0),(enabled,cached)
      print('CACHE',a.model,enabled,repeat,result['wall_s'],'cached_tokens',cached,flush=True)
    finally:
     process.terminate()
     try:process.wait(timeout=20)
     except subprocess.TimeoutExpired:process.kill();process.wait()
 summary={'records':records,'choices_identical':all(x['response']['choices']==y['response']['choices'] for x,y in zip(records['0'],records['1']))}
 fm._write('eff_live_cache_'+a.model,str(model),hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
   {'build':str(build),'prompt_tokens':2048,'new_tokens':16,'max_tokens':1,'repeats':a.repeats,'mode':'aligned exact live prefix'},summary)
 print('CACHE RESULT',a.model,'choices_identical',summary['choices_identical'],flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
