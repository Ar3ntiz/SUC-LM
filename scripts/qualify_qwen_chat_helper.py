#!/usr/bin/env python3
"""Real Qwen chat/cancel/server/unload smoke of the new app helper. No installed app is touched."""
import argparse,hashlib,json,queue,socket,subprocess,tempfile,threading,time,urllib.request
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['helper','model','tokenizer','output','prompt-probe']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();events=queue.Queue();records=[]
    with tempfile.TemporaryDirectory() as tmp,open(Path(tmp)/'helper.log','w') as err:
        p=subprocess.Popen([str(a.helper)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True)
        def reader():
            for line in p.stdout:events.put(json.loads(line))
        threading.Thread(target=reader,daemon=True).start()
        def send(c):p.stdin.write(json.dumps(c)+'\n');p.stdin.flush()
        def until(name,rid=None):
            start=time.monotonic()
            while True:
                e=events.get(timeout=max(.1,300-(time.monotonic()-start)))
                if e['event']=='error':raise RuntimeError(e)
                if e['event']==name and (rid is None or e['id']==rid):return e['payload']
        def status():send({'command':'status','id':'state'});return until('status','state')
        def idle():
            for i in range(60):
                if status()['requests']==0:return
                time.sleep(.05)
            raise RuntimeError('requests did not finish')
        def generate(rid,msgs,cancel=False):
            body={'messages':msgs,'max_tokens':256 if cancel else 16,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},'reasoning_effort':'medium'}
            t=time.monotonic();send({'command':'generate','id':rid,'body':body});text='';metrics={};first=None;sent=False;prefill_event_wall=None
            while True:
                e=events.get(timeout=300)
                if e['event']=='error':raise RuntimeError(e)
                if e['id']!=rid:continue
                if e['event']=='delta':
                    text+=e['payload']['text']
                    if first is None:first=time.monotonic()-t
                    if cancel and not sent:send({'command':'cancel','id':rid});sent=True
                if e['event']=='prefill':prefill_event_wall=time.monotonic()-t
                if e['event']=='metrics':metrics=e['payload']
                if e['event'] in ['finished','cancelled']:
                    assert e['event']==('cancelled' if cancel else 'finished'),e
                    break
            idle();record={'case':rid,'ttft_s':first,'wall_s':time.monotonic()-t,'metrics':metrics,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'status':status()}
            seconds=metrics['prefill_s'];work=metrics['prompt_tokens']
            assert prefill_event_wall is not None and abs(prefill_event_wall-seconds)<max(.25,seconds*.15),record
            record.update(prefill_event_wall_s=prefill_event_wall,prefill_tps_server=work/seconds,prefill_tps_wall=work/prefill_event_wall)
            records.append(record);print(json.dumps(record),flush=True);return record
        try:
            until('ready');send({'command':'load','id':'load','path':str(a.model),'tokenizer':str(a.tokenizer),'context':4096,'memory_mode':'auto'})
            until('loaded','load');loaded_status=status()
            context='\n'.join(f'Voce {i}: il valore di prova è {i%97}; conserva questa informazione nel contesto.' for i in range(96))
            base=[{'role':'system','content':'Rispondi brevemente in italiano, senza ragionamento esplicito.'},{'role':'user','content':context+'\nRiporta soltanto il valore della voce 10.'}]
            follow=base+[{'role':'assistant','content':'<think>Calcolo interno.</think>\n\nIl valore è 10.'},{'role':'user','content':'Ora riporta soltanto il valore della voce 11.'}]
            # The Rust HF tokenizer independently encodes the exact shared C++ server rendering.
            # Jinja is checked separately: historical assistant formatting differs in the inherited SUC renderer.
            from tokenizers import Tokenizer
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            config=json.loads(a.tokenizer.with_name('tokenizer_config.json').read_text())
            source=a.tokenizer.with_name('chat_template.jinja')
            template=source.read_text() if source.exists() else config['chat_template']
            env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
            hf=Tokenizer.from_file(str(a.tokenizer));official_counts={}
            def token_count(msgs,label):
                request=Path(tmp)/(label+'.json')
                request.write_text(json.dumps({'messages':msgs,'chat_template_kwargs':{'enable_thinking':False},'reasoning_effort':'medium'}))
                actual=subprocess.check_output([str(a.prompt_probe),str(a.model),str(request)],text=True)
                official=env.from_string(template).render(messages=msgs,add_generation_prompt=True,enable_thinking=False,reasoning_effort='medium',tools=None)
                official_counts[label]=len(hf.encode(official,add_special_tokens=False).ids)
                return len(hf.encode(actual,add_special_tokens=False).ids)
            independent_counts={'initial':token_count(base,'initial'),'followup':token_count(follow,'followup')}
            initial=generate('initial',base);cached=generate('followup',follow)
            assert initial['metrics']['prompt_tokens']==independent_counts['initial']
            assert cached['metrics']['prompt_tokens']<initial['metrics']['prompt_tokens']/2
            generate('cancel',follow,True);replayed=generate('after_cancel',follow)
            assert replayed['metrics']['prompt_tokens']>initial['metrics']['prompt_tokens']
            assert replayed['text_sha256']==cached['text_sha256'],'cancellation changed replay'
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            key='suc-checkpoint-test-'+'x'*32
            send({'command':'start_server','host':'127.0.0.1','port':port,'key':key});until('server')
            body={'messages':follow,'max_tokens':16,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},'reasoning_effort':'medium'}
            req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
            with urllib.request.urlopen(req,timeout=300) as response:value=json.load(response)
            assert value['usage']['prompt_tokens_details']['cached_tokens']>0
            server_usage=value['usage'];server_metrics=value['suc_metrics']
            assert server_usage['prompt_tokens']==independent_counts['followup']
            assert server_usage['prompt_tokens']-server_usage['prompt_tokens_details']['cached_tokens']==cached['metrics']['prompt_tokens']
            send({'command':'stop_server'});until('server');idle()
            for i in range(60):
                if status()['connections']==0:break
                time.sleep(.05)
            send({'command':'unload','id':'unload'});until('unloaded','unload')
            mapped=subprocess.run(['/usr/sbin/lsof','-p',str(p.pid),'-Fn'],capture_output=True,text=True).stdout
            assert 'n'+str(a.model.resolve()) not in mapped.splitlines(),'model retained after unload'
            result={'passed':True,'records':records,'loaded_status':loaded_status,'server_usage':server_usage,'server_metrics':server_metrics,'independent_hf_token_counts':independent_counts,'official_jinja_token_counts':official_counts,'token_count_protocol':'Rust HF tokenizer on exact shared C++ prompt rendering; official Jinja history formatting counted separately, not asserted equivalent','unload_mapping_released':True}
            a.output.write_text(json.dumps(result,indent=2)+'\n');print('QWEN_CHAT_HELPER_PASS',flush=True)
        finally:
            if p.poll() is None:
                send({'command':'shutdown'})
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=15)
if __name__=='__main__':main()
