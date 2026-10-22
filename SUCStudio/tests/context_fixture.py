"""Offline context/compaction/plugin fixture, never contacts external endpoints."""
import json,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
attempts={}
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        if self.path=="/stats":
            body=json.dumps({k:v for k,v in attempts.items() if isinstance(v,int)}).encode();self.send_response(200);self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body);return
        body=json.dumps({'data':[{'id':'context-fixture','max_model_len':4096}]}).encode()
        self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_POST(self):
        value=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages=value['messages'];system=messages[0]['content'] if messages[0]['role']=='system' else ''
        summary=system.startswith('Riassumi fedelmente')
        assert value.get('think') is False and value.get('chat_template_kwargs',{}).get('enable_thinking') is False
        finish='stop';reasoning=''

        if summary and value['model']=='summary-error':self.send_error(503);return
        if summary:
            assert len(messages)==2 and 'suc_action' not in system
            window=40000 if value['model']=='summary-large-memory' else (32768 if value['model'].startswith(('summary-retry','summary-truncated','summary-only-thinking','summary-adaptive','summary-persistent')) else 4096)
            estimated=lambda text: (len(text.encode())+2)//3 if text else 0
            assert estimated(system)+estimated(messages[-1]['content'])+128+value['max_tokens']<=window
            if value['model'] in ('summary-large-memory','summary-shrunk'):
                # Accumulate sentinels only when actually received, including later chunks.
                seen=attempts.setdefault(value['model']+'-markers',set())
                for marker in ('OLD_MEMORY_START','OLD_MEMORY_END','NEW_HISTORY_END'):
                    if marker in messages[-1]['content']:seen.add(marker)
            if value['model']=='summary-slow':time.sleep(.5)
            text='MEMORY: obiettivo preservato; file queued.swift; decisioni e attività pendenti.'
            model=value['model'];attempts[model]=attempts.get(model,0)+1
            if model in ('summary-large-memory','summary-shrunk'):
                text+=' '+' '.join(sorted(attempts[model+'-markers']))
            if model=='summary-router':assert value.get('reasoning_effort')=='none'
            if model in ('summary-adaptive','summary-persistent'):
                material=messages[-1]['content'].split('Nuovi dati storici:\n',1)[1]
                if model=='summary-persistent' or len(material.encode())>6500:
                    text='BAD_PARTIAL_SUMMARY';reasoning='Budget consumed by reasoning';finish='length'
                else:
                    seen=attempts.setdefault(model+'-markers',set())
                    for marker in ('ADAPT_START','ADAPT_END'):
                        if marker in material:seen.add(marker)
                    text+=' '+' '.join(sorted(seen))
            if model=='summary-only-thinking' or (model=='summary-retry' and attempts[model]==1):text='';reasoning='Thinking without answer';finish='length'
            if model=='summary-truncated' and attempts[model]==1:text='MEMORY: truncated unusable';finish='length'
            if model in ('summary-retry','summary-truncated','summary-only-thinking'):
                assert value['max_tokens']==(2048 if attempts[model]==1 else 4096)

        else:
            latest=messages[-1]['content']
            if value['model']=='thinking-only':
                text='';reasoning='Still thinking';finish='length'
            elif latest=='routine-prompt':
                assert value['max_tokens']>=4000
                text=json.dumps({'suc_action':{'name':'load_plugin','arguments':{'id':'plug.test'}}})
            elif latest.startswith('Risultato reale dello strumento load_plugin') and any(m.get('content')=='routine-prompt' for m in messages):
                assert value['max_tokens']>=4000;text='ROUTINE_RESULT_OK'
            elif latest=='after-compact':
                assert 'MEMORY:' in system
                assert 'RAW_HISTORICAL_MARKER' not in str(messages)
                text='AFTER_COMPACT_OK'
            elif latest.startswith('@plugin:'):
                assert 'PLUGIN_BODY_SECRET' in system;text='EXPLICIT_PLUGIN_OK'
            elif latest=='auto-plugin':
                assert 'plug.test' in system and 'PLUGIN_BODY_SECRET' not in system
                text=json.dumps({'suc_action':{'name':'load_plugin','arguments':{'id':'plug.test'}}})
            elif latest.startswith('Risultato reale dello strumento load_plugin'):
                assert 'PLUGIN_BODY_SECRET' in latest;text='AUTO_PLUGIN_OK'
            else:text='CONTEXT_OK'
        try:
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
            if reasoning:self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'reasoning_content':reasoning}}]})+'\n\n').encode())
            for i in range(0,len(text),16):
                self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':text[i:i+16]}}]})+'\n\n').encode());self.wfile.flush();time.sleep(.005)
            usage={'prompt_tokens':4080 if value['model']=='fill-context' and not summary else 100,'completion_tokens':8}
            self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{},'finish_reason':finish}],'usage':usage})+'\n\ndata: [DONE]\n\n').encode());self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError):pass
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
print(server.server_port,flush=True);server.serve_forever()
