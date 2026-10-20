#!/usr/bin/env python3
"""Sequential real-model timing and load telemetry checks; not a KL/hard qualification."""
import argparse,hashlib,json,queue,subprocess,threading,time,tempfile
from pathlib import Path
ap=argparse.ArgumentParser()
for n in ['helper','model','tokenizer','output']:ap.add_argument('--'+n,type=Path,required=True)
ap.add_argument('--manifest',type=Path)
a=ap.parse_args();records=[];loads=[]
def save():
 a.output.parent.mkdir(parents=True,exist_ok=True)
 value=dict(engine_manifest=json.loads(a.manifest.read_text()) if a.manifest else None,model_bytes=a.model.stat().st_size,scope='Actual helper/model synthetic cold and cached chat; not KL or hard-suite quality',helper_sha256=hashlib.sha256(a.helper.read_bytes()).hexdigest(),model=str(a.model),tokenizer_sha256=hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),loads=loads,records=records)
 tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(a.output)
for profile in ['off','auto']:
 with tempfile.TemporaryFile(mode='w+') as err:
  p=subprocess.Popen([str(a.helper)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True)
  q=queue.Queue()
  def reader(proc=p,out=q):
   for line in proc.stdout:
    try:out.put(json.loads(line))
    except ValueError:out.put(dict(event='error',payload=dict(message='Invalid helper JSON')))
   out.put(dict(event='eof',payload={}))
  threading.Thread(target=reader,daemon=True).start()
  def send(value):p.stdin.write(json.dumps(value)+'\n');p.stdin.flush()
  def until(name,rid=None):
   seen=[]
   while True:
    e=q.get(timeout=600);seen.append(e)
    if e['event'] in ['error','eof']:raise RuntimeError(e)
    if e['event']==name and (rid is None or e['id']==rid):return seen
  def idle():
   for _ in range(100):
    send(dict(command='status',id='status'));s=until('status','status')[-1]['payload']
    if s['requests']==0:return
    time.sleep(.05)
   raise RuntimeError('Helper did not idle')
  def generate(name,body):
   t=time.monotonic();send(dict(command='generate',id=name,body=body));events=[];first=None
   while True:
    e=q.get(timeout=600)
    if e['event'] in ['error','eof']:raise RuntimeError(e)
    if e['id']!=name:continue
    events.append(e)
    if e['event']=='delta' and first is None:first=time.monotonic()-t
    if e['event']=='finished':break
   elapsed=time.monotonic()-t;idle()
   progress=[e['payload'] for e in events if e['event']=='prefill_progress']
   assert progress and progress[-1]['processed']==progress[-1]['total']
   assert all(x['processed']<=y['processed'] for x,y in zip(progress,progress[1:]))
   metrics=[e['payload'] for e in events if e['event']=='metrics'][-1]
   text=''.join(e['payload']['text'] for e in events if e['event']=='delta')
   row=dict(profile=profile,case=name,ttft_s=first,wall_s=elapsed,metrics=metrics,progress_start=progress[0],progress_end=progress[-1],text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest())
   records.append(row);save();print(json.dumps({k:v for k,v in row.items() if k not in ['text','text_sha256']}),flush=True)
   return text
  try:
   until('ready');t=time.monotonic()
   send(dict(command='load',id='load',path=str(a.model),tokenizer=str(a.tokenizer),context=16384,memory_mode='resident',runtime_args=['--prefill','batch','--history-prefill',profile]))
   events=until('loaded','load');samples=[e['payload'] for e in events if e['event']=='load_progress' and e['payload'].get('resident_bytes') is not None]
   assert samples and all(0<=s['resident_bytes']<=s['target_bytes'] for s in samples)
   assert all('metal_allocated_bytes' in s and 'initial_metal_bytes' in s for s in samples)
   plan=next(e['payload'] for e in events if e['event']=='memory_plan');assert plan['mode']=='resident'
   loads.append(dict(memory_plan=plan,profile=profile,load_s=time.monotonic()-t,samples=samples,phases=[e['payload']['phase'] for e in events if e['event']=='load_progress' and 'phase' in e['payload']]))
   save();print('LOADED '+profile,flush=True)
   for count,label in [(500,'2k'),(2000,'8k')]:
    messages=[dict(role='system',content='Timing probe '+label+'. Return only READY.'),dict(role='user',content='alpha beta gamma delta '*count+'\nReturn only READY.')]
    body=dict(messages=messages,max_tokens=16,temperature=0,chat_template_kwargs=dict(enable_thinking=False))
    text=generate(label+'-cold',body);assert records[-1]['progress_start']['cache']==0
    generate(label+'-repeat',body);assert records[-1]['progress_start']['cache']>0
    follow={**body,'messages':messages+[dict(role='assistant',content=text),dict(role='user',content='Return READY again.')]}
    generate(label+'-follow',follow);assert records[-1]['progress_start']['cache']>0
   send(dict(command='unload',id='unload'));until('unloaded','unload')
  except Exception:
   err.seek(0);print(err.read()[-4000:],flush=True);raise
  finally:
   if p.poll() is None:
    send(dict(command='shutdown'));p.wait(timeout=60)
value=json.loads(a.output.read_text());value['passed']=True;a.output.write_text(json.dumps(value,indent=2)+'\n');print('ALLDONE_ADAPTIVE_PREFILL_HELPER',flush=True)
