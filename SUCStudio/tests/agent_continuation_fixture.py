"""Offline streaming fixture. Never contacts models or executes user commands."""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        latest=body['messages'][-1]['content']
        model=body['model']
        if latest == 'next queued turn':
            text='Queued turn completed.'
        elif latest.startswith('Risultato reale dello strumento write_file'):
            text='Markdown file saved and published.'
        elif model in ['broken-write','always-invalid','truncated-invalid','mixed-repair','missing-outer']:
            wire=json.dumps({'suc_action':{'name':'write_file','arguments':{'path':'summary.md','content':'# Fixture summary\n\n```text\n"quoted" \\ path à🙂\n```\n'}}})
            if model=='mixed-repair' and not latest.startswith('Feedback harness'):
                text='Procedo con la sintesi.'
            else:
                text=wire if model=='broken-write' and latest.startswith('Feedback harness') else (wire[:-1] if model in ['missing-outer','truncated-invalid'] else wire[:-1]+'INVALID}')
        elif latest.startswith('Risultato reale dello strumento read_file'):
            assert 'VERIFIED_FIXTURE_CONTENT' in latest
            text='Verified fixture file read successfully.'
        elif latest.startswith('Feedback harness') and model != 'always-promise':
            text=json.dumps({'suc_action':{'name':'read_file','arguments':{'path':'fixture.txt'}}})
        elif model == 'question':
            text='Vuoi che proceda con la ricerca?'
        elif model == 'final':
            text='Il lavoro è completato.'
        else:
            text='Hai ragione, procedo subito con le ricerche.'
        time.sleep(.1)
        self.send_response(200)
        self.send_header('Content-Type','text/event-stream')
        self.end_headers()
        try:
            for delta in [text[:10],text[10:]]:
                self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':delta}}]})+'\n\n').encode());self.wfile.flush()
            reason='length' if model=='truncated-invalid' else 'stop'
            self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{},'finish_reason':reason}]})+'\n\ndata: [DONE]\n\n').encode());self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError):
            pass

server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
print(server.server_port,flush=True)
server.serve_forever()
