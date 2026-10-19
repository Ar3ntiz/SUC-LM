#!/usr/bin/env python3
"""Exercise shipped CLI/server and bundled app engine on the selected local 8B."""
import argparse
import json
from pathlib import Path
import queue
import socket
import subprocess
import tempfile
import threading
import time
import httpx
from efficiency_compare_m1 import local_model

def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); return s.getsockname()[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--version', default='1.4.7')
    ap.add_argument('--app', type=Path, default=Path('SUCStudio/dist/SUC Studio.app'))
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); model = local_model('q8')
    tok = Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json'
    assert tok.is_file(); build = a.build.expanduser(); checks = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        subprocess.run([str(build/'suc1-generate'), str(model), '--tokenizer', str(tok),
                        '--tokens', '5,17,300', '--steps', '8', '--json', str(root/'cli.json'), '--quiet'], check=True)
        assert len(json.loads((root/'cli.json').read_text())['generated']) == 8
        checks.append('CLI raw-token generation: 8 tokens')
        with (root/'server.log').open('w') as err:
            p = port(); url = f'http://127.0.0.1:{p}'
            server = subprocess.Popen([str(build/'suc1-server'), str(model), '--tokenizer', str(tok), '--port', str(p)], stdout=err, stderr=err)
            try:
                for _ in range(240):
                    assert server.poll() is None
                    try:
                        if httpx.get(url+'/v1/models').status_code == 200: break
                    except httpx.TransportError: pass
                    time.sleep(.25)
                else: raise RuntimeError('server startup deadline')
                r = httpx.post(url+'/v1/completions', json={'prompt':[5,17,300], 'max_tokens':8, 'temperature':0}, timeout=120)
                r.raise_for_status(); assert r.json()['usage']['completion_tokens'] == 8
                checks.append('Standalone server completion and usage')
                with httpx.stream('POST', url+'/v1/completions', json={'prompt':[5,17,300], 'max_tokens':8, 'temperature':0, 'stream':True}, timeout=120) as r:
                    r.raise_for_status(); lines=list(r.iter_lines())
                assert 'data: [DONE]' in lines
                checks.append('Standalone server SSE stream completed')
            finally:
                server.terminate(); server.wait(timeout=30)
        with (root/'engine.log').open('w') as err:
            engine = subprocess.Popen([str(a.app/'Contents/Helpers/suc-studio-engine')], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True)
            events = queue.Queue()
            def reader():
                for line in engine.stdout:
                    events.put(json.loads(line))
            threading.Thread(target=reader, daemon=True).start()
            def send(command):
                engine.stdin.write(json.dumps(command)+'\n'); engine.stdin.flush()
            def until(name, request=None):
                deadline=time.monotonic()+120
                while True:
                    e=events.get(timeout=max(.1,deadline-time.monotonic()))
                    assert e['event'] != 'error', e
                    if e['event']==name and (request is None or e['id']==request): return e
            try:
                ready=until('ready'); assert ready['payload']['version']=='1.4.7'
                send({'command':'load','id':'load','path':str(model),'tokenizer':str(tok),'context':4096,'memory_mode':'auto'})
                until('loaded','load'); checks.append('Bundled app engine loads selected local model')
                body={'messages':[{'role':'user','content':'Say hello.'}],'max_tokens':16,'temperature':0}
                send({'command':'generate','id':'chat','body':body}); until('delta','chat'); until('finished','chat')
                checks.append('Bundled app chat streams and completes')
                p=port(); key='suc-test-key-'+'x'*32; url=f'http://127.0.0.1:{p}'
                send({'command':'start_server','host':'127.0.0.1','port':p,'key':key}); until('server')
                assert httpx.get(url+'/v1/models').status_code==401
                r=httpx.post(url+'/v1/chat/completions', headers={'Authorization':'Bearer '+key}, json=body, timeout=120)
                r.raise_for_status(); assert r.json()['choices']
                checks.append('Bundled app server rejects missing key and completes authenticated chat')
                send({'command':'stop_server'}); until('server')
                send({'command':'generate','id':'cancel','body':{**body,'max_tokens':512}})
                send({'command':'cancel','id':'cancel'}); until('cancelled','cancel')
                checks.append('Bundled app engine cancels chat request')
            finally:
                if engine.poll() is None:
                    send({'command':'shutdown'}); engine.wait(timeout=30)
    a.output.write_text(json.dumps({'version':a.version,'checks':checks,'passed':True,
                                   'binary_commit':(build/'GIT_COMMIT').read_text().strip()},indent=2)+'\n')
    print('RELEASE_SMOKE_PASS',len(checks),flush=True)

if __name__=='__main__': main()
