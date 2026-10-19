#!/usr/bin/env python3
"""Compare default services with a matched historical-kernel CLI oracle."""
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

def free_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build',required=True,type=Path)
    ap.add_argument('--model',required=True,choices=['ds','glm'])
    ap.add_argument('--app',type=Path,default=Path('SUCStudio/dist/SUC Studio.app'))
    a=ap.parse_args();b=a.build.expanduser();model=local_model(a.model)
    prompt=fm._long_prompt_ids(b,str(model),512);chunk=64 if a.model=='ds' else 512
    body={'prompt':prompt,'max_tokens':8,'temperature':0}
    results={}
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);ids=root/'ids';ids.write_text(','.join(map(str,prompt)))
        subprocess.run([str(b/'suc1-generate'),str(model),'--tokens-file',str(ids),'--prefill','batch',
                        '--chunk',str(chunk),'--max-context','4096','--steps','8','--stop-eos',
                        '--json',str(root/'oracle.json'),'--quiet'],check=True,
                       env={**os.environ,'SUC_TCQ_REUSE':'0'},capture_output=True,text=True)
        oracle=json.loads((root/'oracle.json').read_text())
        text=json.loads(subprocess.check_output([str(b/'suc-tokenize'),'--suc1',str(model),'--decode',
                        ','.join(map(str,oracle['generated'])),'--skip-special','--json'],text=True))['text']
        def completion(url,headers=None):
            r=httpx.post(url+'/v1/completions',json=body,headers=headers,timeout=240)
            r.raise_for_status();x=r.json();assert x['choices'][0]['text']==text,(a.model,x['choices'],text)
            return x
        with (root/'server.log').open('w') as err:
            p=free_port();url=f'http://127.0.0.1:{p}'
            proc=subprocess.Popen([str(b/'suc1-server'),str(model),'--port',str(p),'--max-context','4096'],stdout=err,stderr=err)
            try:
                for _ in range(600):
                    assert proc.poll() is None
                    try:
                        if httpx.get(url+'/v1/models',timeout=5).status_code==200:break
                    except httpx.TransportError:pass
                    time.sleep(.2)
                else:raise RuntimeError('service startup deadline')
                results['standalone']=completion(url)
            finally:proc.terminate();proc.wait(timeout=30)
        with (root/'app.log').open('w') as err:
            proc=subprocess.Popen([str(a.app/'Contents/Helpers/suc-studio-engine')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True)
            mailbox=queue.Queue()
            def reader():
                for line in proc.stdout:mailbox.put(json.loads(line))
            threading.Thread(target=reader,daemon=True).start()
            def send(x):proc.stdin.write(json.dumps(x)+'\n');proc.stdin.flush()
            def wait(event):
                end=time.monotonic()+240
                while True:
                    x=mailbox.get(timeout=max(.1,end-time.monotonic()));assert x['event']!='error',x
                    if x['event']==event:return x
            try:
                wait('ready');send({'command':'load','id':'load','path':str(model),'tokenizer':'','context':4096,'memory_mode':'auto'});wait('loaded')
                p=free_port();key='profile-test-'+'x'*32
                send({'command':'start_server','host':'127.0.0.1','port':p,'key':key});wait('server')
                results['bundled_app']=completion(f'http://127.0.0.1:{p}',{'Authorization':'Bearer '+key})
                send({'command':'stop_server'});wait('server')
            finally:
                if proc.poll() is None:send({'command':'shutdown'});proc.wait(timeout=30)
    os.environ['SUC_BENCH_BUILD_COMMIT']=(b/'GIT_COMMIT').read_text().strip()
    fm._write('eff_service_profile_'+a.model,str(model),__import__('hashlib').sha256(json.dumps(prompt).encode()).hexdigest(),
              {'model':a.model,'historical_service_chunk':chunk,'context':4096,
               'oracle':'same binary, historical TCQ=0, explicit matched chunk'},
              {'passed':True,'oracle':oracle,'services':results})
    print('SERVICE_PROFILE_PASS',a.model,chunk,flush=True)

if __name__=='__main__':main()
