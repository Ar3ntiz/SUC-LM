"""Local HTTP fixture for WebToolsTests: static pages on 127.0.0.1, random port printed on stdout."""
import sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HITS = []
FILLER = ''.join(f'<p>Filler paragraph {i} with enough text to make the page tall.</p>' for i in range(80))
PAGES = {
    '/results': '''<html><head><title>Results</title></head><body>
<div><a href="https://www.google.com/preferences">Impostazioni Google</a></div>
<div><a href="https://fonts.gstatic.com/asset">Static asset</a></div>
<div><a href="https://example.com/hidden" style="display:none">Hidden link</a></div>
<div><a href="https://example.com/one">First result</a></div>
<div><a href="https://example.com/one">First result again</a></div>
<div><a href="/url?q=https://example.org/target&amp;sa=U">Redirected result</a></div>
<div><a href="https://example.net/short">ab</a></div>
<p>Snippet text.</p>
</body></html>''',
    '/consent': '''<html><head><title>Consent</title></head><body><div id="box"><p>Prima di continuare</p>
<button onclick="setTimeout(()=>{document.getElementById('box').innerHTML='<p>Contenuto dopo il consenso</p>'},300)">Accetta tutto</button></div></body></html>''',
    '/lazy': '''<html><head><title>Lazy</title></head><body>''' + FILLER + '''<script>
let batches=0;
window.addEventListener('scroll',()=>{if(batches>=2)return;if(scrollY+innerHeight<document.body.scrollHeight-1200)return;batches++;
for(let i=0;i<20;i++){const p=document.createElement('p');p.textContent='Lazy block '+(batches*100+i);document.body.appendChild(p);}});
</script></body></html>''',
    '/long': '<html><head><title>Long</title></head><body>' + ''.join(f'<p>Line {i:03d}: ' + 'x' * 70 + '</p>' for i in range(150)) + '</body></html>',
    '/short': '''<html><head><title>Short page</title></head><body><h1>Titolo</h1>
<p>Un paragrafo con <a href="https://example.com/a">un link</a> e <code>codice</code>.</p>
<ul><li>voce uno</li></ul></body></html>''',
    # 1.4.5_Final hidden-window checks: content fetched after load, IntersectionObserver lazy loading, timer/rAF/visibility probe.
    '/xhr': '''<html><head><title>Fetch</title></head><body><p id="status">In attesa dei dati</p><script>
addEventListener('load',()=>setTimeout(()=>fetch('/data').then(r=>r.text()).then(t=>{const p=document.createElement('p');p.textContent=t;document.body.appendChild(p);
const x=new XMLHttpRequest();x.open('GET','/data?xhr=1');x.onload=()=>{const q=document.createElement('p');q.textContent='XHR '+x.responseText;document.body.appendChild(q);};x.send();}),400));
</script></body></html>''',
    '/iolazy': '''<html><head><title>Observer</title></head><body>''' + FILLER + '''<div id="sentinel" class="lazy-sentinel">...</div><script>
let loaded=0;new IntersectionObserver(es=>{for(const e of es){if(!e.isIntersecting||loaded>=2)continue;loaded++;
for(let i=0;i<20;i++){const p=document.createElement('p');p.textContent='Observed block '+(loaded*100+i);document.getElementById('sentinel').before(p);}}},{rootMargin:'0px 0px 1500px 0px'}).observe(document.getElementById('sentinel'));
</script></body></html>''',
    '/timers': '<html><head><title>Timers</title></head><body>' + FILLER + '</body></html>',
    '/sorry/index': '''<html><head><title>Sorry</title></head><body><p>Our systems have detected unusual traffic from your computer network.</p>
<button onclick="fetch('/clicked')">I agree</button></body></html>''',
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split('?')[0]
        HITS.append(path)
        if path == '/slow':
            time.sleep(15)
        if path == '/hits':
            body = '\n'.join(HITS).encode()
        elif path in PAGES:
            body = PAGES[path].encode()
        elif path == '/data':
            body = b'Dati caricati dopo il load'
        elif path == '/slow':
            body = b'<html><body><p>slow</p></body></html>'
        else:
            body = b''
        try:
            self.send_response(200 if body or path in ('/clicked', '/hits') else 404)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
server.daemon_threads = True
print(server.server_address[1], flush=True)
threading.Thread(target=lambda: (sys.stdin.read(), server.shutdown()), daemon=True).start()
server.serve_forever()
