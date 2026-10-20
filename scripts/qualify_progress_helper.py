#!/usr/bin/env python3
"""Actual bundled helper + loopback SSE progress qualification, synthetic conversations only."""
import argparse, hashlib, json, queue, socket, subprocess, tempfile, threading, time, urllib.request
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['helper','model','tokenizer','output']: ap.add_argument('--'+n,required=True,type=Path)
    ap.add_argument('--baseline-helper',type=Path)
    a=ap.parse_args()
    body={'messages':[{'role':'user','content':'alpha beta gamma delta '*120+'\nPrint the integers from 1 to 500, in order. Do not stop early.'}], 'max_tokens':64,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
    results={}
    def run(helper,qualification):
        with tempfile.TemporaryFile() as log:
            process=subprocess.Popen([str(helper)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True)
            q=queue.Queue()
            def reader():
                for line in process.stdout:
                    try:q.put(json.loads(line))
                    except ValueError:q.put({'event':'error','payload':{'message':'Invalid helper JSON'}})
            threading.Thread(target=reader,daemon=True).start()
            def send(c): process.stdin.write(json.dumps(c)+'\n');process.stdin.flush()
            def until(event,id=None):
                observed=[];deadline=time.monotonic()+180
                while True:
                    e=q.get(timeout=max(.01,deadline-time.monotonic()));observed.append(e)
                    assert e['event']!='error',e
                    if e['event']==event and (id is None or e.get('id')==id):return observed
            try:
                until('ready');send({'command':'load','id':'load','path':str(a.model),'tokenizer':str(a.tokenizer),'context':4096,'memory_mode':'auto'})
                load=until('loaded','load')
                if qualification: assert any(e['event']=='load_progress' for e in load)
                send({'command':'generate','id':'chat','body':body});events=until('finished','chat')
                text=''.join(e['payload']['text'] for e in events if e['event']=='delta')
                result={'text_sha256':hashlib.sha256(text.encode()).hexdigest()}
                if not qualification:return result
                progress=[e['payload'] for e in events if e['event']=='prefill_progress']
                assert len(progress)>=2 and progress[0]['processed']==0 and progress[0]['cache']==0
                assert progress[-1]['processed']==progress[-1]['total']
                assert all(x['processed']<=y['processed'] for x,y in zip(progress,progress[1:]))
                live=[e['payload'] for e in events if e['event']=='metrics' and 'prefill_tokens' in e['payload']]
                assert live and all(m['completion_tokens']>1 and m['decode_s']>0 for m in live)
                with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
                key='local-progress-test-'+'x'*32
                send({'command':'start_server','host':'127.0.0.1','port':port,'key':key});until('server')
                remote_body={**body,'messages':[{'role':'system','content':'Synthetic SSE verification.'}]+body['messages'],'stream':True,'stream_options':{'include_usage':True}}
                req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(remote_body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
                with urllib.request.urlopen(req,timeout=180) as response:
                    frames=[json.loads(line[6:]) for line in response if line.startswith(b'data: ') and line.strip()!=b'data: [DONE]']
                remote_progress=[f['prompt_progress'] for f in frames if 'prompt_progress' in f]
                remote_live=[f['suc_metrics'] for f in frames if 'completion_tokens' in f.get('suc_metrics',{})]
                assert len(remote_progress)>=2 and remote_live
                assert remote_progress[-1]['processed']==remote_progress[-1]['total']
                assert all(f.get('choices')==[] for f in frames if 'prompt_progress' in f or 'suc_metrics' in f)
                final=next(f for f in reversed(frames) if 'usage' in f)
                assert final['usage']['completion_tokens']>=remote_live[-1]['completion_tokens']
                send({'command':'stop_server'});until('server')
                # Report the actual continuation policy; reconstructed text may invalidate a strict-only cache.
                follow={**body,'messages':body['messages']+[{'role':'assistant','content':text},{'role':'user','content':'Continue.'}],'max_tokens':16}
                send({'command':'generate','id':'base','body':body});until('finished','base')
                send({'command':'generate','id':'follow','body':follow});follow_events=until('finished','follow')
                reused=[e['payload'] for e in follow_events if e['event']=='prefill_progress']
                assert reused and 0<=reused[0]['cache']<=reused[0]['processed']<=reused[0]['total']
                # Cancellation after actual prefill begins must not leave a stale observer/cache.
                cancel_body={**body,'messages':[{'role':'system','content':'Cancellation probe.'}]+body['messages']}
                send({'command':'generate','id':'cancel','body':cancel_body});until('prefill_progress','cancel')
                send({'command':'cancel','id':'cancel'});until('cancelled','cancel')
                send({'command':'generate','id':'replay','body':body});replay=until('finished','replay')
                replay_start=next(e['payload'] for e in replay if e['event']=='prefill_progress')
                assert replay_start['cache']==0
                replay_text=''.join(e['payload']['text'] for e in replay if e['event']=='delta')
                assert replay_text==text
                result.update({'loading_phases':[e['payload']['phase'] for e in load if e['event']=='load_progress' and 'phase' in e['payload']], 'ipc_progress_events':len(progress),'ipc_live_metric_events':len(live),'sse_progress_events':len(remote_progress),'sse_live_metric_events':len(remote_live),'followup_cached_tokens':reused[0]['cache'],'cancel_replay_cache_zero':True,'cancel_replay_text_identical':True})
                return result
            finally:
                if process.poll() is None:
                    send({'command':'shutdown'});process.wait(timeout=30)
    if a.baseline_helper:results['baseline']=run(a.baseline_helper,False)
    results['candidate']=run(a.helper,True)
    if a.baseline_helper:assert results['baseline']['text_sha256']==results['candidate']['text_sha256']
    a.output.write_text(json.dumps({'passed':True,'version':'1.4.7.04','scope':'Synthetic actual helper and SSE checks; not a throughput or hard/KL benchmark','baseline_text_identical':bool(a.baseline_helper),'results':results},indent=2)+'\n')
    print('Actual helper progress, live server counts, cache and cancel qualification: PASS')
if __name__=='__main__':main()
