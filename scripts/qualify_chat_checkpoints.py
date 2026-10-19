#!/usr/bin/env python3
"""Paired natural-Qwen chat tests on a selected LOCAL model. Own sequential loopback servers only."""
import argparse, hashlib, json, os, queue, subprocess, tempfile, threading, time, urllib.request
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build',type=Path,required=True);ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--items',type=int,nargs='+',default=[96,512]);a=ap.parse_args()
    model=a.model.expanduser();build=a.build.expanduser();tok=a.tokenizer.expanduser()
    assert model.is_file() and not model.is_symlink() and model.open('rb').read(4)==b'SUC1'
    assert subprocess.check_output(['df','-P',str(model.resolve())],text=True).splitlines()[-1].split()[0].startswith('/dev/')
    root=Path(__file__).resolve().parents[1]
    results={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
             'model':str(model),'items':a.items,'records':[],
             'protocol':'Same fixed conversation and official template; checkpoint off/on, prefill batch chunk64, greedy16; one process at a time; fixed baseline/candidate order (warm-up recorded separately). No private data.'}
    for variant in ['0','1']:
        env={**os.environ,'SUC_PREFIX_CHECKPOINTS':variant,'SUC_PREFIX_CACHE':'1','HF_HUB_OFFLINE':'1'}
        with tempfile.TemporaryDirectory() as tmp,open(Path(tmp)/'server.log','w') as err:
            server=subprocess.Popen([str(build/'suc1-server'),str(model),'--tokenizer',str(tok),'--port','0',
                                     '--max-context','20000','--prefill','batch','--chunk','64','--quiet'],
                                    env=env,stdout=subprocess.PIPE,stderr=err,text=True)
            try:
                messages=queue.Queue()
                def reader():
                    for line in server.stdout:messages.put(line)
                    messages.put(None)
                threading.Thread(target=reader,daemon=True).start()
                start=time.monotonic()
                while True:
                    line=messages.get(timeout=max(.1,180-(time.monotonic()-start)))
                    if line is None:raise RuntimeError('server startup failed: '+Path(tmp,'server.log').read_text()[-4000:])
                    if line.startswith('listening on '):url=line.split()[-1]+'/v1/chat/completions';break
                def ask(label,msgs,think=False):
                    body={'stream':True,'stream_options':{'include_usage':True},'messages':msgs,'max_tokens':16,'temperature':0,'chat_template_kwargs':{'enable_thinking':think},'reasoning_effort':'medium'}
                    t=time.monotonic()
                    content='';first=None;finish=None;u={}
                    with urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=1200) as response:
                        done=False
                        for raw in response:
                            if not raw.startswith(b'data:'):continue
                            data=raw[5:].strip()
                            if data==b'[DONE]':done=True;break
                            value=json.loads(data)
                            if 'error' in value:raise RuntimeError(value['error'])
                            u.update(value.get('usage') or {})
                            for choice in value.get('choices',[]):
                                delta=choice.get('delta',{}).get('content') or ''
                                if delta and first is None:first=time.monotonic()-t
                                content+=delta
                                if choice.get('finish_reason'):finish=choice['finish_reason']
                        assert done and u,'stream incomplete'
                    choices={'text':content,'finish_reason':finish}
                    record={'variant':variant,'case':label,'wall_s':time.monotonic()-t,'ttft_s':first,'usage':u,'suc_metrics':value.get('suc_metrics',{}),
                            'choices_sha256':hashlib.sha256(json.dumps(choices,sort_keys=True).encode()).hexdigest()}
                    results['records'].append(record);a.output.parent.mkdir(parents=True,exist_ok=True)
                    a.output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(record),flush=True)
                    return content
                # Warm kernels on the same fixed text; excluded from performance comparisons.
                ask('warmup',[{'role':'user','content':'Say hello.'}])
                for count in a.items:
                    context='\n'.join(f'Voce {i}: il valore di prova è {i%97}; conserva questa informazione nel contesto.' for i in range(count))
                    base=[{'role':'system','content':'Rispondi brevemente in italiano, senza ragionamento esplicito.'},
                          {'role':'user','content':context+'\nRiporta soltanto il valore della voce 10.'}]
                    ask(f'{count}_initial',base)
                    # Fixed historical answer makes both variants receive IDENTICAL tokens even if generation diverges.
                    follow=base+[{'role':'assistant','content':'<think>Calcolo interno.</think>\n\nIl valore è 10.'},
                                 {'role':'user','content':'Ora riporta soltanto il valore della voce 11.'}]
                    ask(f'{count}_followup',follow)
                    if count==min(a.items):
                        follow[-1]['content']='Ora riporta soltanto il valore della voce 12.'
                        ask(f'{count}_edit',follow);ask(f'{count}_reload',follow)
                        follow[0]['content']='Rispondi brevemente in inglese.';ask(f'{count}_system_change',follow)
                        ask(f'{count}_thinking_initial',base,True);ask(f'{count}_thinking_followup',base+follow[2:],True)
                # Verify normal shutdown; never signal another process or the installed app.
            finally:
                server.terminate()
                try:server.wait(timeout=30)
                except subprocess.TimeoutExpired:server.kill();server.wait()
    indexed={(r['variant'],r['case']):r for r in results['records']}
    results['choices_identical']=all(r['choices_sha256']==indexed['0',r['case']]['choices_sha256'] for r in results['records'] if r['variant']=='1')
    assert results['choices_identical'],'cache changed greedy responses'
    for r in results['records']:
        cached=r['usage']['prompt_tokens_details']['cached_tokens']
        if r['case'].endswith(('_followup','_edit','_reload')):assert (cached>0)==(r['variant']=='1'),r
        if r['case'].endswith('_system_change'):assert cached==0,r
    results['passed']=True;a.output.write_text(json.dumps(results,indent=2)+'\n');print('CHAT_CHECKPOINT_QUALIFICATION_PASS',flush=True)

if __name__=='__main__':main()
