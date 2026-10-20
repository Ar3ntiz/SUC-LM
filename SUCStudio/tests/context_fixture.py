"""Offline context/compaction/plugin fixture, never contacts external endpoints."""
import json,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        body=json.dumps({'data':[{'id':'context-fixture','max_model_len':4096}]}).encode()
        self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_POST(self):
        value=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages=value['messages'];system=messages[0]['content'] if messages[0]['role']=='system' else ''
        summary=system.startswith('Riassumi fedelmente')
        if summary and value['model']=='summary-error':self.send_error(503);return
        if summary:
            assert len(messages)==2 and 'suc_action' not in system
            assert len(system.encode())+len(messages[-1]['content'].encode())+value['max_tokens']<4096
            if value['model']=='summary-slow':time.sleep(.5)
            text='MEMORY: obiettivo preservato; file queued.swift; decisioni e attività pendenti.'
        else:
            latest=messages[-1]['content']
            if latest=='after-compact':
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
            for i in range(0,len(text),16):
                self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':text[i:i+16]}}]})+'\n\n').encode());self.wfile.flush();time.sleep(.005)
            usage={'prompt_tokens':4080 if value['model']=='fill-context' and not summary else 100,'completion_tokens':8}
            self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{},'finish_reason':'stop'}],'usage':usage})+'\n\ndata: [DONE]\n\n').encode());self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError):pass
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
print(server.server_port,flush=True);server.serve_forever()
