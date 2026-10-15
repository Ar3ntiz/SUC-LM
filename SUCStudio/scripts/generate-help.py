#!/usr/bin/env python3
"""Generate an offline, escaped HTML guide from the bundled Markdown source."""
import html
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent

def inline(text):
    text = html.escape(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    def link(match):
        label, url = match.groups()
        if url.startswith(('https://', 'http://', '#')):
            return f'<a href="{url}">{label}</a>'
        return f'{label} <code>{url}</code>'
    return re.sub(r'\[([^]]+)\]\(([^)]+)\)', link, text)

parts, toc, paragraph, code, listing = [], [], [], None, None
table = False

def flush():
    if paragraph:
        parts.append('<p>' + inline(' '.join(paragraph)) + '</p>')
        paragraph.clear()

def close_list():
    global listing
    if listing:
        parts.append(f'</{listing}>')
        listing = None

for line in (root / 'USAGE.md').read_text().splitlines():
    if line.startswith('|'):
        flush(); close_list()
        cells = line.strip('|').split('|')
        if all(re.fullmatch(r'\s*:?-+:?\s*', cell) for cell in cells): continue
        tag = 'td' if table else 'th'
        if not table: parts.append('<div class="table"><table>'); table = True
        parts.append('<tr>' + ''.join(f'<{tag}>' + inline(cell.strip()) + f'</{tag}>' for cell in cells) + '</tr>')
        continue
    if table: parts.append('</table></div>'); table = False
    if line.startswith('```'):
        flush(); close_list()
        if code is None:
            code = []
        else:
            parts.append('<pre><code>' + html.escape('\n'.join(code)) + '</code></pre>'); code = None
    elif code is not None:
        code.append(line)
    elif not line.strip():
        flush(); close_list()
    elif (m := re.match(r'^(#{1,6}) (.*)', line)):
        flush(); close_list()
        level, title = len(m[1]), m[2]
        anchor = f'section-{len(toc)}'
        parts.append(f'<h{level} id="{anchor}">{inline(title)}</h{level}>')
        if level <= 2: toc.append((anchor, title))
    elif (m := re.match(r'^\s*(?:([-*])|\d+\.) (.*)', line)):
        flush()
        kind = 'ul' if m[1] else 'ol'
        if listing != kind:
            close_list(); listing = kind; parts.append(f'<{kind}>')
        parts.append('<li>' + inline(m[2]) + '</li>')
    else:
        close_list(); paragraph.append(line)
flush(); close_list()
if table: parts.append('</table></div>')
if code is not None: raise SystemExit('Unclosed Markdown code block')
css = ''' :root {color-scheme:light dark;--bg:#f6f7fb;--card:#fff;--fg:#232438;--muted:#60657b;--accent:#5554ce}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.7 -apple-system,BlinkMacSystemFont,sans-serif}nav{position:fixed;width:260px;height:100vh;overflow:auto;padding:30px 22px;border-right:1px solid #8883}nav strong{display:block;font-size:23px;margin-bottom:22px}nav a{display:block;font-size:13px;margin:10px 0;color:var(--muted);text-decoration:none}main{margin-left:260px;padding:42px 6vw 100px;max-width:1200px}h1{font-size:36px;line-height:1.2;letter-spacing:-1px}h2{font-size:25px;margin-top:48px;scroll-margin-top:25px}h3{margin-top:30px}a{color:var(--accent)}pre{background:var(--card);border:1px solid #8883;border-radius:12px;padding:20px;overflow:auto;font-size:13px}code{font-family:ui-monospace,monospace;font-size:.88em;overflow-wrap:anywhere}li{margin:8px 0}.table{overflow:auto}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:12px;border-bottom:1px solid #8883}p{max-width:850px}@media(prefers-color-scheme:dark){:root{--bg:#171820;--card:#22232d;--fg:#ececf3;--muted:#b1b3c7;--accent:#b3acff}}@media(max-width:800px){nav{position:static;width:auto;height:auto;border-right:0;border-bottom:1px solid #8883}nav a{display:inline-block;margin:4px 12px 4px 0}main{margin:0;padding:25px}h1{font-size:30px}}'''
nav = ''.join(f'<a href="#{anchor}">{html.escape(title)}</a>' for anchor,title in toc[1:])
page = '<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SUC Studio · Setup e utilizzo</title><style>' + css + '</style><body><nav><strong>SUC Studio</strong>' + nav + '</nav><main>' + '\n'.join(parts) + '</main></body></html>\n'
(root / 'Resources/USAGE.html').write_text(page)
print('Offline HTML guide generated')
