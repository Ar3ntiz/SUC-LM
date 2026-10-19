const fs=require('fs'),path=require('path'),assert=require('node:assert/strict'),{JSDOM}=require('jsdom');
const script=fs.readFileSync(path.join(__dirname,'../app/src/main/java/local/sucstudio/android/WebPagePolicy.kt'),'utf8').match(/val extraction = """([\s\S]*?)"""/)[1];
function page(html,url='https://example.org/'){const d=new JSDOM(html,{url,runScripts:'outside-only'});Object.defineProperty(d.window.HTMLElement.prototype,'innerText',{get(){return this.textContent}});d.window.HTMLElement.prototype.getBoundingClientRect=()=>({width:200,height:100});return d}
let d=page('<form id="captcha-form"><button>Verifica</button></form><p>Attesa</p>');assert(d.window.eval(script).challenge);d.window.document.querySelector('form').remove();d.window.document.querySelector('p').textContent='Risultato leggibile dopo verifica manuale';assert(!d.window.eval(script).challenge);d.window.close();
d=page('<h1>Come funzionano CAPTCHA e robot</h1><p>Articolo informativo: not a robot</p>');assert(!d.window.eval(script).challenge);d.window.close();
d=page('<p>Controllo accesso</p>','https://www.google.com/sorry/index?continue=search');assert(d.window.eval(script).challenge);d.window.close();
d=page('<iframe src="https://example.org/recaptcha" style="display:none"></iframe><a href="https://example.org/result">Risultato</a>');assert(!d.window.eval(script).challenge);assert.equal(d.window.eval(script).links.length,1);d.window.close();
console.log('Web page policy: visible challenge, dynamic completion, Google verification URL, ordinary articles and hidden widgets PASS');
