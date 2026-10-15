"""Deterministic MCP test server: no external services, no user data."""
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def reply(value):
    method, params = value.get('method'), value.get('params', {})
    if 'id' not in value: return None
    if method == 'initialize':
        result = {'protocolVersion': '2025-11-25', 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'fixture', 'version': '1'}}
    elif method == 'tools/list':
        if params.get('cursor') == 'page2': result = {'tools': [{'name': 'fail', 'description': 'Failure', 'inputSchema': {'type': 'object'}}]}
        else: result = {'tools': [{'name': 'echo', 'description': 'Echo', 'inputSchema': {'type': 'object', 'properties': {'text': {'type': 'string'}}}}], 'nextCursor': 'page2'}
    elif method == 'tools/call':
        if params.get('name') == 'fail': result = {'isError': True, 'content': [{'type': 'text', 'text': 'expected failure'}]}
        else:
            if params.get('arguments', {}).get('delay'): time.sleep(.2)
            result = {'content': [{'type': 'text', 'text': params.get('arguments', {}).get('text', 'echo')}], 'structuredContent': {'ok': True}}
    else: return {'jsonrpc': '2.0', 'id': value['id'], 'error': {'code': -32601, 'message': 'unknown'}}
    return {'jsonrpc': '2.0', 'id': value['id'], 'result': result}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_POST(self):
        value = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/redirect':
            self.send_response(307); self.send_header('Location', '/mcp'); self.end_headers(); return
        if self.headers.get('Authorization') != 'Bearer fixture' or (value.get('method') != 'initialize' and self.headers.get('MCP-Session-Id') != 'fixture-session') or self.headers.get('MCP-Protocol-Version') != '2025-11-25':
            self.send_response(401); self.end_headers(); return
        result = reply(value)
        if result is None:
            self.send_response(202); self.end_headers(); return
        sse = value.get('method') == 'tools/call'
        body = json.dumps(result).encode()
        if sse: body = b': heartbeat\r\n\r\ndata: ' + body + b'\r\n\r\n'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if sse else 'application/json')
        self.send_header('MCP-Session-Id', 'fixture-session')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)

if '--http' in sys.argv:
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    print(server.server_port, flush=True)
    server.serve_forever()
else:
    for line in sys.stdin:
        result = reply(json.loads(line))
        if result is not None: print(json.dumps(result), flush=True)
