import json,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def authorized(self):
  if self.headers.get('Authorization')=='Bearer test': return True
  self.send_response(401);self.end_headers();return False
 def do_GET(self):
  if not self.authorized():return
  data=json.dumps({'status':'ok'} if self.path.endswith('/health') else {'data':[{'id':'fixture'},{'id':'slow'}]}).encode();self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def do_POST(self):
  if not self.authorized():return
  value=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  assert value['stream'] and 'top_k' not in value and 'min_p' not in value and 'seed' not in value
  if value['model'] in ('eof','unicode','json','oversized-line','oversized-total','oversized-json'):
   mode=value['model'];kind='text/event-stream'
   if mode=='eof':data=b'data: {"choices":[{"delta":{"content":"EOF"},"finish_reason":"stop"}]}\n\ndata: [DONE]'
   elif mode=='unicode':data=('data: '+json.dumps({'choices':[{'delta':{'content':'Città 🌍'},'finish_reason':'stop'}]},ensure_ascii=False)+'\r\n\r\ndata: [DONE]\r\n\r\n').encode()
   elif mode=='json':kind='application/json';data=b'{"choices":[{"message":{"content":"JSON response"}}],"usage":{"prompt_tokens":1000,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":800}},"suc_metrics":{"prefill_s":0.5}}'
   elif mode=='oversized-line':data=b'data: '+b'x'*(4*1024*1024+1)+b'\n'
   elif mode=='oversized-json':kind='application/json';data=b' '*(4*1024*1024+1)
   else:data=(b':'+b'x'*100000+b'\n')*170
   self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.end_headers()
   try:self.wfile.write(data);self.wfile.flush()
   except (BrokenPipeError,ConnectionResetError):pass
   return
  if value['model']=='progress-live':
   events=[{'choices':[],'prompt_progress':{'processed':2048,'total':2165,'cache':2048}},
           {'choices':[],'prompt_progress':{'processed':2165,'total':2165,'cache':2048}},
           {'choices':[{'delta':{'content':'A'}}]},
           {'choices':[],'suc_metrics':{'prompt_tokens':2165,'cached_tokens':2048,'prefill_tokens':117,'prefill_s':1.0,'completion_tokens':41,'decode_s':2.0}},
           {'choices':[{'delta':{'content':'B'},'finish_reason':'stop'}]},
           {'choices':[],'usage':{'prompt_tokens':2165,'completion_tokens':81,'prompt_tokens_details':{'cached_tokens':2048}},'suc_metrics':{'prefill_s':1.0,'decode_s':4.0}}]
   self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
   for event in events:
    self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode());self.wfile.flush();time.sleep(.1)
   self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush();return
  if value['model']=='vision':
   content=value['messages'][0]['content']
   assert content[0]['type']=='text' and content[1]['type']=='image_url'
   assert content[1]['image_url']['url'].startswith('data:image/jpeg;base64,')
  if value['model'] in ('ds4','llama','lmstudio','custom'):
   assert value['stream_options']['include_usage']
   if value['model']=='llama': assert value['timings_per_token'] and value['return_progress']
   events=[]
   if value['model']=='llama':events.append({'prompt_progress':{'processed':200,'cache':100,'total':300,'time_ms':500}})
   events += [{'choices':[{'delta':{'reasoning_content':'Analizzo.'}}]}, {'choices':[{'delta':{'content':'Risposta'},'finish_reason':'stop'}]}]
   events.append({'usage':{'prompt_tokens':100,'completion_tokens':11},'choices':[]})
   if value['model']=='lmstudio':events.append({'stats':{'tokens_per_second':42.5,'time_to_first_token':10}})
   if value['model']=='custom':events.append({'rates':{'p':123.0,'d':45.0}})
   self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
   for event in events:
    self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode());self.wfile.flush();time.sleep(.1)
   self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush();return
  self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
  try:
   self.wfile.write(b'data: {"choices":[{"delta":{"content":"Hello "}}]}\r\n\r\n');self.wfile.flush()
   time.sleep(2 if value['model']=='slow' else .05)
   self.wfile.write(b'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}\n\n');self.wfile.flush()
   if value['model']=='timed': self.wfile.write(b'data: {"suc_metrics":{"prefill_s":2.0,"decode_s":0.5}}\n\n')
   self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
  except (BrokenPipeError,ConnectionResetError):pass
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);print(server.server_port,flush=True);server.serve_forever()
