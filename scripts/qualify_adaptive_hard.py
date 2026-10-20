#!/usr/bin/env python3
"""Fixed six-case hard smoke: first item per source, original renderer/token ids/grader."""
import argparse,hashlib,json,queue,secrets,socket,subprocess,sys,tempfile,threading,time,urllib.request
from pathlib import Path
ap=argparse.ArgumentParser()
for n in ['helper','model','tokenizer','suite','output']:ap.add_argument('--'+n,type=Path,required=True)
a=ap.parse_args();sys.path.insert(0,str(Path.cwd()))
from suc.evaluation.runners import load_tokenizer
from suc.evaluation.suite import load_suite,render_prompt_ids
from suc.evaluation.answers import grade
suite=load_suite(a.suite);tok=load_tokenizer(a.tokenizer.parent);seen=set();cases=[]
for c in suite['cases']:
 if c['source'] not in seen:seen.add(c['source']);cases.append((c,render_prompt_ids(tok,c,suite.get('system'),c['think'])))
assert len(cases)==6 and all(not c['think'] for c,_ in cases)
rows=[]
def save():
 a.output.parent.mkdir(parents=True,exist_ok=True)
 tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(dict(scope='Six-case hard smoke only, not full 120-case score or KL qualification',selection='First case per source in original suite order',suite_sha256=suite['sha256'],helper_sha256=hashlib.sha256(a.helper.read_bytes()).hexdigest(),records=rows),indent=2)+'\n');tmp.replace(a.output)
for profile in ['off','auto']:
 with tempfile.TemporaryFile(mode='w+') as log:
  p=subprocess.Popen([str(a.helper)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True);events=queue.Queue()
  def reader(proc=p,out=events):
   for line in proc.stdout:out.put(json.loads(line))
   out.put(dict(event='eof',payload={}))
  threading.Thread(target=reader,daemon=True).start()
  def send(c):p.stdin.write(json.dumps(c)+'\n');p.stdin.flush()
  def until(name,rid=None):
   while True:
    e=events.get(timeout=600)
    if e['event'] in ['error','eof']:raise RuntimeError(e)
    if e['event']==name and (rid is None or e['id']==rid):return e['payload']
  try:
   until('ready');send(dict(command='load',id='load',path=str(a.model),tokenizer=str(a.tokenizer),context=16384,memory_mode='resident',runtime_args=['--prefill','batch','--history-prefill',profile]));until('loaded','load')
   with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
   key=secrets.token_hex(32);send(dict(command='start_server',host='127.0.0.1',port=port,key=key));until('server')
   for c,ids in cases:
    assert len(ids)+c['max_tokens']<=16384
    body=dict(prompt=ids,max_tokens=c['max_tokens'],temperature=0,stream=False)
    req=urllib.request.Request('http://127.0.0.1:%d/v1/completions'%port,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key,'Connection':'close'})
    start=time.monotonic()
    with urllib.request.urlopen(req,timeout=600) as response:value=json.load(response)
    text=value['choices'][0]['text'];picked,passed=grade(c,text)
    assert value['usage']['prompt_tokens']==len(ids),value['usage']
    assert value['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)==0
    rows.append(dict(profile=profile,id=c['id'],source=c['source'],prompt_tokens=len(ids),prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),budget=c['max_tokens'],picked=picked,passed=passed,text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest(),wall_s=time.monotonic()-start,usage=value['usage'],metrics=value.get('suc_metrics')))
    save();print(json.dumps({k:v for k,v in rows[-1].items() if k not in ['text','usage','metrics']}),flush=True)
   send(dict(command='stop_server'));until('server')
  except Exception:
   log.seek(0);print(log.read()[-4000:],flush=True);raise
  finally:
   if p.poll() is None:send(dict(command='shutdown'));p.wait(timeout=60)
value=json.loads(a.output.read_text());value['completed']=True;value['scores']={s:sum(r['passed'] for r in rows if r['profile']==s) for s in ['off','auto']};a.output.write_text(json.dumps(value,indent=2)+'\n');print('ALLDONE_ADAPTIVE_HARD_SMOKE',flush=True)
