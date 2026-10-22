"""Synthetic streaming model used to validate user updates between operations."""
import json,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_GET(self):
        body=b'{"status":"ok"}';self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_POST(self):
        value=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages=value['messages'];latest=messages[-1]['content'];model=value['model']
        def action(name,**args): return json.dumps({'suc_action':{'name':name,'arguments':args}})
        if latest=='START':
            if model=='during': answer=action('run_command',command='sleep 1; printf TOOL_SENTINEL',wait_seconds=2)
            else: answer=action('write_file',path='wrong.txt',content='must-not-run')
            chunks=['Preparing.\n',answer] if model=='before' else [answer]
        elif latest.startswith('CORRECTION'):
            assert 'DRAFT_NOT_SENT' not in str(messages)
            assert messages[-1]['role']=='user'
            if model=='during':
                assert 'TOOL_SENTINEL' in messages[-2]['content']
                assert messages[-2]['role']=='user'
                assert sum('Risultato reale dello strumento run_command' in str(m['content']) for m in messages)==1
                chunks=['CORRECTED_AFTER_TOOL']
            else:
                assert any('NOT executed' in str(m['content']) for m in messages if m['role']=='assistant')
                assert not any('Risultato reale dello strumento write_file' in str(m['content']) for m in messages)
                assert 'ATTACHMENT_SENTINEL' in latest
                chunks=[action('write_file',path='correct.txt',content='CORRECTED_CONTENT')]
        else:
            assert latest.startswith('Risultato reale dello strumento write_file')
            chunks=['CORRECTED_FINAL']
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
        try:
            for chunk in chunks:
                self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':chunk}}]})+'\n\n').encode());self.wfile.flush();time.sleep(.5)
            self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n');self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError): pass
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);print(server.server_port,flush=True);server.serve_forever()
