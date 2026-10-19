#!/usr/bin/env python3
"""Real bundled-engine session isolation, cancellation, reload and memory checks."""
import argparse
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import tempfile
import threading
import time
import httpx
import frontier_measure as fm
from efficiency_compare_m1 import local_model

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build',required=True,type=Path)
    ap.add_argument('--app',type=Path,default=Path('SUCStudio/dist/SUC Studio.app'))
    ap.add_argument('--model',choices=['q8','q38'],default='q8')
    a=ap.parse_args();build=a.build.expanduser();model=local_model(a.model)
    repo,rev=('models--Qwen--Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218') if a.model=='q8' else ('models--Qwen--Qwen3.8-Flash-Next','de4b8e4d43b917e7706784d8bb445c9af86a3540')
    tok=Path.home()/'.cache/huggingface/hub'/repo/'snapshots'/rev/'tokenizer.json'
    assert tok.is_file()
    base=fm._long_prompt_ids(build,str(model),768)
    other=[77]+base[1:]
    prompts=[base[:256],base[:320],base[:384],other[:384],base[:384],base[:448],base[:256],base[:320],base[:320],base[:384]]
    expected=[0,256,320,0,0,384,0,256,0,320]
    results={}
    for enabled in ['0','1']:
        subprocess.run(['bash',str(fm.REPO/'scripts/studio_wait_idle.sh'),'--check'],check=True)
        with tempfile.TemporaryDirectory() as tmp, open(Path(tmp)/'engine.log','w') as err:
            engine=subprocess.Popen([str(a.app/'Contents/Helpers/suc-studio-engine')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,env={**os.environ,'SUC_PREFIX_CACHE':enabled})
            events=queue.Queue()
            def read_events(stream=engine.stdout,mailbox=events):
                try:
                    for line in stream:mailbox.put(json.loads(line))
                except Exception as e:mailbox.put({'event':'error','payload':{'reader':str(e)}})
            threading.Thread(target=read_events,daemon=True).start()
            def send(body):engine.stdin.write(json.dumps(body)+'\n');engine.stdin.flush()
            def wait(name,identity=None):
                end=time.monotonic()+180
                while True:
                    e=events.get(timeout=max(.1,end-time.monotonic()))
                    assert e['event']!='error',e
                    if e['event']==name and (identity is None or e.get('id')==identity):return e['payload']
            def status():
                send({'command':'status','id':'status'});return wait('status','status')
            def idle():
                for _ in range(200):
                    s=status()
                    if s['requests']==0 and s['connections']==0:return
                    time.sleep(.05)
                raise RuntimeError('Engine did not become idle')
            def load():
                send({'command':'load','id':'load','path':str(model),'tokenizer':str(tok),'context':4096,'memory_mode':'auto'})
                return wait('loaded','load')
            def server():
                with socket.socket() as sock:sock.bind(('127.0.0.1',0));p=sock.getsockname()[1]
                send({'command':'start_server','host':'127.0.0.1','port':p,'key':'session-check-'+'x'*32});wait('server')
                return f'http://127.0.0.1:{p}/v1/completions'
            try:
                wait('ready');load();url=server();headers={'Authorization':'Bearer '+'session-check-'+'x'*32}
                def ask(tokens):
                    start=time.monotonic()
                    with httpx.Client(timeout=180) as client:
                        r=client.post(url,headers=headers,json={'prompt':tokens,'max_tokens':1,'temperature':0})
                        r.raise_for_status();body=r.json()
                    return {'wall_s':time.monotonic()-start,'cached_tokens':body['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0),'choices':body['choices']}
                turns=[]
                for i,prompt in enumerate(prompts):
                    r=ask(prompt);assert r['cached_tokens']==(expected[i] if enabled=='1' else 0),(enabled,i,r)
                    turns.append(r);print('SESSION',a.model,enabled,i,r['cached_tokens'],flush=True)
                # Invalidate while inference has actually emitted text, not just queued.
                send({'command':'generate','id':'cancel','body':{'messages':[{'role':'user','content':'Count from one to one thousand, explaining every number.'}],'max_tokens':1024,'temperature':0}})
                wait('delta','cancel');send({'command':'cancel','id':'cancel'});wait('cancelled','cancel');idle()
                after_cancel=ask(base[:256]);assert after_cancel['cached_tokens']==0
                warm=ask(base[:320]);assert warm['cached_tokens']==(256 if enabled=='1' else 0)
                idle();before=status()['footprint_bytes'];footprints=[]
                for _ in range(8):
                    ask(base[:256]);ask(base[:320]);idle();footprints.append(status()['footprint_bytes'])
                if before is not None and all(v is not None for v in footprints):
                    assert max(footprints)-before<=256*1024**2,(before,footprints)
                send({'command':'stop_server'});wait('server');idle()
                send({'command':'unload','id':'unload'});wait('unloaded','unload');load();url=server()
                reloaded=ask(base[:256]);assert reloaded['cached_tokens']==0
                send({'command':'stop_server'});wait('server');idle()
                results[enabled]={'turns':turns,'after_cancel':after_cancel,'reloaded':reloaded,
                                  'footprint_before_replay':before,'footprints_after_replay':footprints}
            finally:
                if engine.poll() is None:
                    send({'command':'shutdown'})
                    try:engine.wait(timeout=30)
                    except subprocess.TimeoutExpired:engine.terminate();engine.wait(timeout=15)
    same=all(x['choices']==y['choices'] for x,y in zip(results['0']['turns'],results['1']['turns']))
    same=same and results['0']['after_cancel']['choices']==results['1']['after_cancel']['choices']
    same=same and results['0']['reloaded']['choices']==results['1']['reloaded']['choices']
    os.environ['SUC_BENCH_BUILD_COMMIT']=(build/'GIT_COMMIT').read_text().strip()
    fm._write('eff_session_closeout_'+a.model,str(model),__import__('hashlib').sha256(json.dumps(base).encode()).hexdigest(),
              {'aligned_chunk':64,'context':4096,'model':a.model,'replay_cycles':8,
               'memory_growth_guard_bytes':256*1024**2,'scope':'real bundled engine; raw-token API session isolation, not visual GUI'},
              {'passed':same,'choices_identical':same,'results':results})
    assert same,'Cached and uncached session choices differ'
    print('ALLDONE_SESSION_CLOSEOUT',a.model,flush=True)

if __name__=='__main__':main()
