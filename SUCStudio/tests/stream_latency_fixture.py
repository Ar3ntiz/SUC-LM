#!/usr/bin/env python3
"""Loopback-only burst SSE source; no inference, private chat or credentials."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json,time,urllib.parse
stats={}
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        run=urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get('run',[''])[0]
        body=json.dumps(stats.get(run,{})).encode()
        self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        count=int(body.get('diagnostic_chunks',4000));assert 1<=count<=64000
        run=str(body.get('diagnostic_run',''));piece='riga **test**\n'
        event=('data: '+json.dumps({'choices':[{'delta':{'content':piece}}]})+'\n\n').encode()
        data=event*count+b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(data)));self.end_headers()
        start=time.time();self.wfile.write(data);self.wfile.flush();end=time.time()
        stats[run]={'send_start_unix':start,'send_end_unix':end,'send_wall_s':end-start,'sse_bytes':len(data),'expected_text_bytes':len(piece.encode())*count}
server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
print(server.server_port,flush=True);server.serve_forever()
