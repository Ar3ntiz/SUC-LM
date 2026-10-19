#!/usr/bin/env python3
"""Time an already-running authenticated OpenAI-compatible server; synthetic input only."""
import argparse,getpass,hashlib,json,time,urllib.request,urllib.error,uuid
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-url',required=True);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--max-items',type=int,default=1024)
    a=ap.parse_args();base=a.base_url.rstrip('/');key=getpass.getpass('API key (not saved): ')
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept':'text/event-stream'}
    with urllib.request.urlopen(urllib.request.Request(base+'/models',headers=headers),timeout=10) as r:
        model=json.load(r)['data'][0]['id']
    results=[];run=uuid.uuid4().hex
    def measure(label,messages):
        body={'model':model,'messages':messages,'stream':True,'stream_options':{'include_usage':True},'temperature':0,'max_tokens':32,'chat_template_kwargs':{'enable_thinking':False}}
        encoded=json.dumps(body).encode();start=time.monotonic();first=None;last=None;done=None;text='';thought='';usage={};metrics={};events=0
        req=urllib.request.Request(base+'/chat/completions',data=encoded,headers=headers,method='POST')
        with urllib.request.urlopen(req,timeout=1200) as resp:
            for raw in resp:
                if not raw.startswith(b'data:'):continue
                data=raw[5:].strip()
                if data==b'[DONE]':done=time.monotonic()-start;break
                if not data:continue
                item=json.loads(data);events+=1
                if 'error' in item:raise RuntimeError('Server returned error: '+str(item['error'].get('type','unknown')))
                usage.update(item.get('usage') or {});metrics.update(item.get('suc_metrics') or {})
                for choice in item.get('choices',[]):
                    d=choice.get('delta',{});content=d.get('content') or '';reason=d.get('reasoning_content') or d.get('reasoning') or ''
                    if content or reason:
                        now=time.monotonic()-start
                        if first is None:first=now
                        last=now;text+=content;thought+=reason
        end=time.monotonic()-start
        result={'case':label,'request_bytes':len(encoded),'request_sha256':hashlib.sha256(encoded).hexdigest(),'ttft_http_s':first,'last_text_http_s':last,'done_http_s':done,'wall_http_s':end,'after_last_text_s':None if last is None else end-last,'usage':usage,'suc_metrics':metrics,'events':events,'text_bytes':len(text.encode()),'reasoning_bytes':len(thought.encode()),'text_sha256':hashlib.sha256(text.encode()).hexdigest()}
        if done is None:raise RuntimeError('Stream ended without DONE')
        results.append(result);a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps({'kind':'live_server_synthetic_chat_latency','model':model,'requests':results,'scope':'Raw HTTP reader; no private conversation or API key stored; currently loaded model reused'},indent=2)+'\n')
        print(json.dumps(result,sort_keys=True),flush=True)
        return text
    system={'role':'system','content':'Diagnostic '+run+'. Rispondi brevemente in italiano, senza ragionamento esplicito.'}
    for count in [0,96,512,1024]:
        if count>a.max_items:continue
        context='\n'.join(f'Voce {i}: il valore di prova è {i%97}; conserva questa informazione nel contesto.' for i in range(count))
        user={'role':'user','content':context+'\nRiporta soltanto il valore della voce 10, oppure saluta se la voce manca.'}
        messages=[system,user]
        answer=measure(f'items_{count}',messages)
        if count:
            measure(f'items_{count}_followup',messages+[{'role':'assistant','content':answer},{'role':'user','content':'Ora riporta il valore della voce 11, in una frase breve.'}])
    print('ALLDONE_LIVE_CHAT_DIAGNOSTICS',flush=True)
if __name__=='__main__':main()
