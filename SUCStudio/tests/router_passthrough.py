"""Read-only compatibility test of an external LLM_Router checkout with a fake backend."""
import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import httpx
from llmrouter.app import create_app
from llmrouter.config import RouterConfig, ResidentBackend, ModelRoute

received = []
events = [
    {'choices': [{'delta': {'reasoning_content': 'Analizzo i dati.'}}]},
    {'choices': [{'delta': {'content': 'Risposta'}, 'finish_reason': 'stop'}]},
    {'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 11},
     'suc_metrics': {'prefill_s': 2.0, 'decode_s': 1.0},
     'timings': {'prompt_per_second': 50, 'predicted_per_second': 10}},
]
wire = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events) + 'data: [DONE]\n\n'
class Backend(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(b'{"data":[{"id":"real-model"}]}')
    def do_POST(self):
        received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        self.wfile.write(wire.encode())
server = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
threading.Thread(target=server.serve_forever, daemon=True).start()
async def main():
    cfg = RouterConfig(residents={'fixture': ResidentBackend('fixture', f'http://127.0.0.1:{server.server_port}/v1')},
                       exclusives={}, models={'alias': ModelRoute('alias', 'fixture', 'real-model')})
    app = create_app(cfg)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://router-test') as client:
            response = await client.post('/v1/chat/completions', json={
                'model': 'alias', 'messages': [{'role': 'user', 'content': 'test'}],
                'stream': True, 'stream_options': {'include_usage': True},
                'timings_per_token': True, 'return_progress': True})
            assert response.status_code == 200, response.text
            assert response.text == wire, response.text
            assert received[0]['model'] == 'real-model'
            assert received[0]['stream_options']['include_usage'] and received[0]['timings_per_token']
            print('LLM_Router actual ASGI app passed: alias routing, unchanged reasoning/usage/timings SSE and forwarded telemetry options')
try:
    asyncio.run(main())
finally:
    server.shutdown(); server.server_close()
