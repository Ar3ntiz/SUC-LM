#!/usr/bin/env python3
"""Localhost-only OpenAI fixture, never contacts a real model."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json,time,sys
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_GET(self):
        if self.path=='/challenge':
            body=("""<html><head><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><h1>Verifica di prova</h1><form id="captcha-form"><p>CAPTCHA simulato per il test locale, senza servizi esterni.</p><button id="solve" type="button" onclick="document.getElementById('captcha-form').remove();document.getElementById('result').textContent='Fixture page: verifica completata. '+('Contenuto leggibile della pagina dopo la verifica. ').repeat(10)">Completa verifica di prova</button></form><p id="result"></p></body></html>""").encode()
            self.send_response(200);self.send_header('Content-Type','text/html');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
        if self.path=='/page':
            body=('<html><body><h1>Fixture page</h1><p>'+'Contenuto pubblico di prova per verificare la lettura reale della pagina web. '*10+'</p></body></html>').encode(); self.send_response(200);self.send_header('Content-Type','text/html');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
        if self.path=='/v1/models':
            body=json.dumps({'data':[{'id':'fixture-chat'}]}).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
        self.send_error(404)
    def do_POST(self):
        request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.headers.get('Authorization')!='Bearer fixture-key':self.send_error(401);return
        messages=request['messages'];last=messages[-1]['content']
        if isinstance(last,list):last=json.dumps(last)
        if 'Risultato publish_artifact' in last:
            text='Artifact creato.'
        elif 'artifact test' in last:
            text=json.dumps({'suc_action':{'name':'publish_artifact','arguments':{'name':'hello.py','content':'print("hello")\n'}}})
        elif 'Risultato visit_page' in last:
            assert 'Fixture page' in last
            text='Web verificato.\n\n```python\nprint("hello")\n```'
        elif 'captcha test' in last:
            text=json.dumps({'suc_action':{'name':'visit_page','arguments':{'url':'http://127.0.0.1:9876/challenge'}}})
        elif 'tool test' in last:
            text='Procedo con la lettura.\n{"suc_action":{"name":"visit_page","arguments":{"url":"http://127.0.0.1:9876/page"}}}'
        elif 'long test' in last:text='Risposta lunga terminata.'
        else:text='Risposta di prova **completata**.'
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
        try:
            for item in [{'choices':[{'delta':{'reasoning_content':'Analizzo la richiesta.'}}]}]+[{'choices':[{'delta':{'content':text[i:i+4]}}]} for i in range(0,len(text),4)]+[{'choices':[{'delta':{},'finish_reason':'stop'}],'usage':{'prompt_tokens':100,'completion_tokens':20},'timings':{'prompt_per_second':120,'predicted_per_second':35}}]:
                self.wfile.write(('data: '+json.dumps(item)+'\n\n').encode());self.wfile.flush();time.sleep(.08 if 'long test' not in last else 1)
            self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError):pass
if __name__=='__main__':
    server=ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1]) if len(sys.argv)>1 else 9876),Handler);print(server.server_port,flush=True);server.serve_forever()
