"""No model required: checks the actual helper framing, error routing and EOF shutdown."""
import json, subprocess, sys, time, threading, queue
p=subprocess.Popen([sys.argv[1]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
q=queue.Queue()
def reader():
 for line in p.stdout:q.put(json.loads(line))
threading.Thread(target=reader,daemon=True).start()
def event(name,id=None):
 deadline=time.monotonic()+10
 while time.monotonic()<deadline:
  e=q.get(timeout=10)
  if e['event']==name and (id is None or e['id']==id):return e
 raise AssertionError(name)
def send(c):p.stdin.write(json.dumps(c).encode()+b'\n');p.stdin.flush()
event('ready')
p.stdin.write(b'{"command":"hel');p.stdin.flush();p.stdin.write(b'lo","id":"split"}\n');p.stdin.flush();assert event('ready','split')['payload']['protocol']==1
send({'command':'generate','id':'no-model','body':{}});assert 'Load a model' in event('error','no-model')['payload']['message']
send({'command':'load','id':'bad-file','path':'/nonexistent/model.suc1','tokenizer':'/nonexistent/tokenizer.json','context':512});event('error','bad-file')
send({'command':'hello','id':'after-failure'});event('ready','after-failure')
p.stdin.close();assert p.wait(timeout=10)==0
print('protocol_test: split framing, errors, failed load recovery and EOF shutdown passed')
