"""Synthetic model responses. Commands touch only the test's temporary workspace."""
import json,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        b=b'{"status":"ok"}';self.send_response(200);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        latest=body['messages'][-1]['content'];model=body['model']
        def action(name,**args):return json.dumps({'suc_action':{'name':name,'arguments':args}})
        if model in ['loop','progress']:
            if model=='loop':command="printf x >> runs.txt; printf LOOP_SENTINEL"
            else:command="n=$(cat count.txt 2>/dev/null || printf 0); n=$((n+1)); printf '%s' \"$n\" > count.txt; printf 'COUNT:%s' \"$n\""
            if model=='progress' and latest.startswith('Risultato reale dello strumento run_command') and json.loads(latest.split('\n',1)[1]).get('output')=='COUNT:5':text='PROGRESS_FINAL'
            else:text=action('run_command',command=command,wait_seconds=60)
        elif latest=='UPDATE':text='STEERED_FINAL'
        elif latest.startswith('Risultato reale dello strumento read_file'):
            assert 'FIXTURE_READ_SENTINEL' in latest;text='RECOVERED_FINAL'
        elif latest.startswith('Feedback harness') and model!='always-inline':
            assert 'JSON canonico' in latest
            if model=='stop-repair':time.sleep(.6)
            text=action('read_file',path='fixture.txt')
        elif model=='inline-suc':text='Riprovo.<tool_call>suc_action({"name":"read_file","arguments":{"path":"fixture.txt"}})'
        elif model=='inline-json':text='Verifico.<tool_call>'+action('read_file',path='fixture.txt')+'</tool_call>'
        else:text='Verifico lo strumento.<tool_call>run_command("printf fixture", 30, 60)'
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
        try:
            for i in range(0,len(text),7):
                self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':text[i:i+7]}}]})+'\n\n').encode());self.wfile.flush();time.sleep(.001)
            reason='length' if model=='length-inline' else 'stop'
            self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{},'finish_reason':reason}]})+'\n\ndata: [DONE]\n\n').encode());self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError):pass
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);print(server.server_port,flush=True);server.serve_forever()
