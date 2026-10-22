"""Local-only MCP lifecycle fixture; records counts, never runs user tools."""
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
lock=threading.Lock()
counts={'calls':0,'cancelled':0}
def reply(value):
    method=value.get('method'); params=value.get('params',{})
    if method=='notifications/cancelled':
        with lock: counts['cancelled']+=1
    if 'id' not in value: return None
    if method=='initialize': result={'protocolVersion':'2025-11-25','capabilities':{'tools':{}}}
    elif method=='tools/list': result={'tools':[{'name':n,'inputSchema':{'type':'object'}} for n in ['echo','stats']]}
    elif method=='tools/call':
        if params['name']=='stats':
            with lock: text=json.dumps(counts)
        else:
            with lock: counts['calls']+=1
            args=params.get('arguments',{})
            time.sleep(min(35,max(0,float(args.get('delay',0)))))
            text=args.get('text','done')
        result={'content':[{'type':'text','text':text}]}
    else: return {'jsonrpc':'2.0','id':value['id'],'error':{'code':-32601,'message':'unknown'}}
    return {'jsonrpc':'2.0','id':value['id'],'result':result}
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_POST(self):
        result=reply(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
        body=b'' if result is None else b'data: '+json.dumps(result).encode()+b'\n\n'
        self.send_response(202 if result is None else 200)
        self.send_header('Content-Type','text/event-stream')
        self.send_header('Content-Length',str(len(body)));self.end_headers()
        try: self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError): pass
if '--http' in sys.argv:
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    print(server.server_port,flush=True);server.serve_forever()
else:
    output_lock=threading.Lock()
    def consume(value):
        result=reply(value)
        if result is not None:
            with output_lock: print(json.dumps(result),flush=True)
    for line in sys.stdin:
        threading.Thread(target=consume,args=(json.loads(line),),daemon=True).start()
