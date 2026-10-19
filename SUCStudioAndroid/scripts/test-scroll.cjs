const fs=require('fs'),path=require('path'),assert=require('node:assert/strict'),{JSDOM}=require('jsdom');
const html=fs.readFileSync(path.join(__dirname,'../app/src/main/assets/index.html'),'utf8');
let frames=new Map(),frameId=0;
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true,beforeParse(w){
 w.requestAnimationFrame=fn=>{frames.set(++frameId,fn);return frameId};w.cancelAnimationFrame=id=>frames.delete(id);
 w.Studio={command(){}};w.matchMedia=()=>({matches:false});w.HTMLElement.prototype.scrollIntoView=function(){};
}});
const w=dom.window,t=w.document.getElementById('transcript');let height=2000,viewport=500,top=0;
Object.defineProperties(t,{scrollHeight:{get:()=>height},clientHeight:{get:()=>viewport},scrollTop:{get:()=>top,set:v=>{top=Math.max(0,Math.min(v,height-viewport))}}});
t.getBoundingClientRect=()=>({top:0,bottom:viewport});
const flush=()=>{const q=[...frames.values()];frames.clear();q.forEach(f=>f())};
const scroll=v=>{t.scrollTop=v;t.dispatchEvent(new w.Event('scroll'))};
const wheel=v=>t.dispatchEvent(new w.WheelEvent('wheel',{deltaY:v}));
const touch=(type,y)=>{const e=new w.Event(type);Object.defineProperty(e,'touches',{value:y===undefined?[]:[{clientY:y}]});t.dispatchEvent(e)};
const snap=(id='a',generating=true)=>w.nativeEvent('state',{sessions:[{id:'s',title:'Chat'}],selected:'s',messages:[{id,role:'assistant',text:'<think>Pensiero</think>Risposta',generating}],settings:{server:{},skills:[],plugins:[],mcp:[]},attachments:[]});
snap();flush();assert.equal(top,1500);
height=2100;w.scrollIfNeeded(); // Queue auto-scroll, then start reading before the frame executes.
touch('touchstart',100);touch('touchmove',150);scroll(1450);touch('touchend');flush();assert.equal(top,1450);
height=2400;w.scrollIfNeeded();flush();assert.equal(top,1450,'streaming must leave the reader in place');
// A programmatic scroll at the bottom must never reattach on its own.
scroll(1900);height=2500;w.scrollIfNeeded();flush();assert.equal(top,1900);
// User scrolling down to the bottom reattaches; growth then follows.
wheel(100);scroll(2000);height=2700;w.scrollIfNeeded();flush();assert.equal(top,2200);
// Even a small upward move within the bottom tolerance must detach.
wheel(-2);scroll(2198);height=2800;w.scrollIfNeeded();flush();assert.equal(top,2198);
w.resumeScroll();flush();assert.equal(top,2300);
// A tap does not disable following; frames wait until touch ends.
touch('touchstart',100);height=2900;w.scrollIfNeeded();flush();assert.equal(top,2300);touch('touchend');flush();assert.equal(top,2400);
// Preserve a visible message anchor when DOM replacement temporarily clamps scrollTop.
wheel(-100);scroll(1000);const rect=w.HTMLElement.prototype.getBoundingClientRect;
w.HTMLElement.prototype.getBoundingClientRect=function(){return this.classList.contains('message')?{top:800-top,bottom:1800-top}:rect.call(this)};
w.preserveTranscript(()=>{height=700;t.scrollTop=top;height=3000});assert.equal(top,1000);
// Final snapshot retains reading position and disclosure state, without enabling follow.
snap('a',false);flush();assert.equal(top,1000);assert(w.document.querySelector('.reasoning').open);
height=3200;w.scrollIfNeeded();flush();assert.equal(top,1000);
w.resumeScroll();flush();assert.equal(top,2700);
// A newly selected chat deliberately starts at its end.
w.nativeEvent('state',{sessions:[{id:'other',title:'Altra'}],selected:'other',messages:[],settings:{server:{}},attachments:[]});flush();assert.equal(top,2700);
(async()=>{
 snap();flush();wheel(-100);scroll(1000);height=3400;
 w.nativeEvent('delta',{id:'a',text:'<think>Pensiero aggiornato</think>Risposta in streaming',generating:true});
 await new Promise(resolve=>setTimeout(resolve,160));flush();assert.equal(top,1000);assert(t.textContent.includes('Risposta in streaming'));
 w.dispatchEvent(new w.Event('resize'));flush();assert.equal(top,1000);
 touch('touchstart',200);touch('touchmove',100);scroll(2900);touch('touchend');height=3500;w.scrollIfNeeded();flush();assert.equal(top,3000);
 console.log('Scroll: stale frames, touch/tap, wheel, detached streaming, programmatic events, bottom reattach, DOM anchor, final snapshot, real delta handler, resize and session reset PASS');dom.window.close();
})().catch(error=>{dom.window.close();console.error(error);process.exitCode=1});
