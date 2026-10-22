"""Offline model + search fixture for the scheduled web/skill integration tests."""
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith('/results'):
            body = b'<html><head><title>Fixture search</title></head><body><div><a href="https://example.com/one">First result</a></div><p>Verified fixture source snippet.</p></body></html>'
            content_type = 'text/html; charset=utf-8'
        else:
            body = json.dumps({'status': 'ok', 'data': [{'id': 'routine-web'}, {'id': 'skill'}]}).encode()
            content_type = 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        value = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = value['messages']
        latest = messages[-1]['content']
        system = messages[0]['content'] if messages[0]['role'] == 'system' else ''
        assert value['max_tokens'] == 4000
        if value['model'].startswith('queue'):
            prompt = latest.split('\n\nDocumenti allegati')[0]
            if value['model'] == 'queue-error' and prompt == 'queue-first':
                self.send_error(503, 'fixture failure'); return
            if prompt == 'queue-first':
                assert not any('queue-second' in str(m['content']) for m in messages)
            if value['model'] == 'queue-tool' and prompt == 'queue-first':
                text = json.dumps({'suc_action': {'name': 'load_skill', 'arguments': {'name': 'QueueSkill'}}})
            elif latest.startswith('Risultato reale dello strumento load_skill'):
                assert 'QUEUE_SKILL_BODY' in latest
                assert not any('queue-second' in str(m['content']) for m in messages)
                text = 'ACK:queue-first'
            else:
                text = 'ACK:' + prompt
                if prompt == 'queue-second' and value['model'] == 'queue-tool':
                    assert any('NOT executed' in str(m['content']) for m in messages if m['role'] == 'assistant')
            time.sleep(.25)
        elif value['model'] == 'explicit-mcp':
            assert 'CORPO_SKILL_RISERVATO' in system
            assert len(system.encode()) < 8192
            assert 'HUGE_SCHEMA_MARKER' not in system
            if latest.startswith('Risultato reale dello strumento mcp_call'):
                assert latest.endswith('x' * (4 * 1024 * 1024))
                assert len(latest.encode()) > 4 * 1024 * 1024
                text = 'Skill e MCP verificati.'
            elif latest.startswith('Risultato reale dello strumento mcp_schema'):
                text = json.dumps({'suc_action': {'name': 'mcp_call', 'arguments': {'server': 'fixture', 'tool': 'echo', 'input': json.dumps({'result_bytes': 4 * 1024 * 1024})}}})
            elif latest.startswith('Risultato reale dello strumento mcp_list'):
                text = json.dumps({'suc_action': {'name': 'mcp_schema', 'arguments': {'server': 'fixture', 'tool': 'echo'}}})
            else:
                text = json.dumps({'suc_action': {'name': 'mcp_list', 'arguments': {'server': 'fixture', 'query': 'echo'}}})
        elif value['model'] == 'skill':
            assert 'Descrizione per verificare documenti' in system
            assert 'CORPO_SKILL_RISERVATO' not in system
            if latest.startswith('Risultato reale dello strumento load_skill'):
                assert 'CORPO_SKILL_RISERVATO' in latest
                text = 'Skill applicata correttamente.'
            else:
                skill_id = re.search(r'\[id: ([A-Fa-f0-9-]{36})\]', system).group(1)
                text = json.dumps({'suc_action': {'name': 'load_skill', 'arguments': {'id': skill_id}}})
        elif latest.startswith('Risultato reale dello strumento web_search'):
            assert 'https://example.com/one' in latest, latest
            text = 'Risultato web verificato dalla routine.'
        elif latest.startswith('Errore dello strumento'):
            text = 'Errore verificabile: ' + latest
        else:
            wire = json.dumps({'suc_action': {'name': 'web_search', 'arguments': {'query': 'prezzi GPL Marche MIMIT ottobre 2026'}}})
            if value['model'].startswith('routine-commentary'):
                intro = 'Ho capito, vuoi che io verifichi ora i dati più recenti su questi portali per darti un quadro aggiornato e preciso. Procedo con le ricerche mirate.\n\n'
                text = intro + ('```json\n' + wire + '\n```' if value['model'].endswith('fenced') else wire)
            else:
                text = '```json\n' + wire + '\n```'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        for index in range(0, len(text), 7):
            event = {'choices': [{'delta': {'content': text[index:index+7]}}]}
            self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
            self.wfile.flush()
            time.sleep(0.08 if value["model"].startswith("queue") else 0.005)
        self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        self.wfile.flush()


server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
server.daemon_threads = True
print(server.server_port, flush=True)
server.serve_forever()
