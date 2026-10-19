#!/usr/bin/env python3
"""Exercise the real helper protocol and check that explicit unload releases model mappings."""
import argparse,hashlib,json,queue,socket,subprocess,tempfile,threading,time,urllib.request
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--helper',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--expect-retained',action='store_true')
    a=ap.parse_args();events=queue.Queue();checks=[]
    with tempfile.TemporaryDirectory() as tmp,open(Path(tmp)/'helper.log','w') as err:
        proc=subprocess.Popen([str(a.helper)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True)
        def reader():
            for line in proc.stdout:events.put(json.loads(line))
        threading.Thread(target=reader,daemon=True).start()
        def send(body):proc.stdin.write(json.dumps(body)+'\n');proc.stdin.flush()
        def until(name,request=None):
            start=time.monotonic()
            while True:
                e=events.get(timeout=max(.1,180-(time.monotonic()-start)))
                if e['event']=='error':raise RuntimeError(e)
                if e['event']==name and (request is None or e['id']==request):return e
        def mapped():
            out=subprocess.run(['/usr/sbin/lsof','-p',str(proc.pid),'-Fn'],capture_output=True,text=True).stdout
            return 'n'+str(a.model.resolve()) in out.splitlines()
        try:
            until('ready');send({'command':'load','id':'load','path':str(a.model),'tokenizer':str(a.tokenizer),'context':512,'memory_mode':'auto'})
            until('loaded','load');assert mapped(),'model not mapped after load';checks.append('model loaded and mapped')
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            key='suc-checkpoint-test-'+'x'*32
            send({'command':'start_server','host':'127.0.0.1','port':port,'key':key});until('server')
            request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',data=json.dumps({'prompt':[5,17,3],'max_tokens':4,'temperature':0}).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
            with urllib.request.urlopen(request,timeout=180) as response:value=json.load(response)
            assert value['usage']['completion_tokens']==4;checks.append('bundled server raw completion')
            send({'command':'stop_server'});until('server')
            for i in range(60):
                send({'command':'status','id':'status'});status=until('status','status')['payload']
                if status['connections']==0 and status['requests']==0:break
                time.sleep(.05)
            else:raise RuntimeError('helper did not become idle')
            send({'command':'unload','id':'unload'});until('unloaded','unload')
            retained=mapped()
            assert retained==a.expect_retained,{'retained_model_mapping':retained,'expected':a.expect_retained}
            checks.append('retained mapping reproduced' if retained else 'unload released model mapping after completed API job')
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps({'passed':True,'helper':str(a.helper),'model':str(a.model),'checks':checks,'retained_after_unload':retained},indent=2)+'\n')
            print('HELPER_UNLOAD_PASS',json.dumps(checks),flush=True)
        finally:
            if proc.poll() is None:
                send({'command':'shutdown'})
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
if __name__=='__main__':main()
