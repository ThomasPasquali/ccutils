#!/usr/bin/env python3
"""logviz - follow a structured log file and explore it as a live, interactive web page.

Usage:  python logviz.py run.log [-p 8000] [--host 127.0.0.1] [--no-follow] [--no-browser]
                         [--lenient | --patterns my_patterns.json]
Structure (sections / MPI prints / ranks / JSON blocks) comes from ccutils_parser.py; if the default
CCUTILS markers are not found, any ===/---/### banner is used as a section instead.

No third-party packages. Format-agnostic: it auto-detects
  * section banners   (=+=+= name :: title =+=+=  ...  =+=+= name END =+=+=, --- x ---, ### x ###)
  * blocks            ([[Rank 0]] ... [[END Rank 0]])
  * key=value pairs   (also "Key: value", "[TAG] 12.3 MB") -> metrics
  * JSON lines        -> collapsible tree + numeric leaves become metrics
  * ANSI colour codes (also when the ESC byte was lost, e.g. "[96m")
"""
import argparse, html, json, math, os, re, sys, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import ccutils_parser  # structure detection (sections, MPI prints, ranks); keep next to this file

def parse(text, patterns=None):
    # ccutils_parser annotates each line (section / MPI print / rank). Metrics are derived client-side
    # from the JSON; timers come from the parser's CCUTILS_TIMER records.
    doc = ccutils_parser.MPIOutputParser(patterns).parse_document(text)
    if patterns is None and not doc.marker_count:      # unknown banner style: generic fallback
        doc = ccutils_parser.MPIOutputParser(ccutils_parser.ParserPatterns.lenient()).parse_document(text)
    recs, secs, seen = [], {}, {}
    for sec in doc.all_sections():
        secs[sec.path] = sec.title
        for name in sec.mpi_all_prints:
            secs[f'{sec.path}/{name}'] = 'MPI all-print'
    for ln in doc.lines:
        s, d = ln.text, ln.data
        r = {'i': ln.no - 1, 'k': 'text', 't': s, 's': ln.section + ('/' + ln.mpi if ln.mpi else ''),
             'b': f'Rank {ln.rank}' if ln.rank is not None else ''}
        if ln.kind == 'marker':
            r['k'] = 'mark'
        elif ln.kind == 'json_body':
            r['k'] = 'jbody'
        elif ln.kind == 'timer' and d is not None:
            f = {k: (v if not isinstance(v, float) or math.isfinite(v) else str(v)) for k, v in d.fields.items()}
            r.update(k='timer', tm={'type': d.record_type, 'f': f})
        elif s.strip() in ('...', '\u2026'):
            r['k'] = 'gap'
        if r['k'] != 'timer' and d is not None and ln.kind in ('json', 'marker'):
            key = (r['s'], r['b'])                      # occurrence number: repeated sections = separate runs
            r['o'] = seen.get(key, 0)
            seen[key] = r['o'] + 1
            r.update(k='json', j=d, t=s[:400])
        recs.append(r)
    return {'recs': recs, 'secs': secs, 'warnings': doc.warnings}


class App:
    def __init__(self, path, follow, patterns=None):
        self.path, self.follow, self.sig, self.ver = path, follow, None, 0
        self.patterns = patterns
        self.data, self.lock = {'recs': [], 'secs': {}}, threading.Lock()

    def get(self):
        with self.lock:
            try:
                st = os.stat(self.path)
                sig = (st.st_mtime_ns, st.st_size)
            except OSError:
                sig = ('missing',)
            if sig != self.sig and (self.follow or self.sig is None):
                self.sig = sig
                try:
                    with open(self.path, encoding='utf-8', errors='replace') as f:
                        text = f.read()
                except OSError:
                    text = ''
                self.data, self.ver = parse(text, self.patterns), self.ver + 1
            return self.ver, self.data


def handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype):
            self.send_response(200)
            self.send_header('Content-Type', ctype + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == '/':
                page = PAGE.replace('__FILE__', html.escape(os.path.basename(app.path)))
                self.send(page.encode(), 'text/html')
            elif u.path == '/api/data':
                ver, data = app.get()
                if parse_qs(u.query).get('v', [''])[0] == str(ver):
                    out = {'ver': ver, 'same': True}
                else:
                    out = {'ver': ver, 'same': False, 'follow': app.follow, **data}
                self.send(json.dumps(out, ensure_ascii=False, separators=(',', ':'),
                                     allow_nan=False).encode(), 'application/json')
            else:
                self.send_error(404)
    return H


def main():
    ap = argparse.ArgumentParser(description='Interactive web view for structured log files.')
    ap.add_argument('file')
    ap.add_argument('-p', '--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--no-follow', action='store_true', help='parse once, do not watch for changes')
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--lenient', action='store_true', help='treat any ===/---/### banner as a nested section')
    ap.add_argument('--patterns', metavar='JSON', help='JSON file overriding ccutils_parser marker regexes')
    a = ap.parse_args()
    if not os.path.exists(a.file):
        print(f'warning: {a.file} does not exist yet; waiting for it to appear', file=sys.stderr)
    pats = (ccutils_parser.ParserPatterns.from_json_file(a.patterns) if a.patterns
            else ccutils_parser.ParserPatterns.lenient() if a.lenient else None)
    app = App(a.file, not a.no_follow, pats)
    for port in range(a.port, a.port + 20):
        try:
            srv = ThreadingHTTPServer((a.host, port), handler(app))
            break
        except OSError:
            continue
    else:
        sys.exit('no free port found')
    url = f'http://{a.host}:{port}/'
    print(f'Serving {a.file} at {url}  (Ctrl+C to stop)')
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__FILE__ - logviz</title>
<style>
:root{--bg:#f6f7f9;--pn:#fff;--fg:#1d2330;--mu:#677084;--bd:#dde1e8;--ac:#2b6cb0;--ac2:#e7f0fb;--ok:#2f9e5e;--mk:#fff3cd;color-scheme:light}
@media(prefers-color-scheme:dark){:root{--bg:#14171d;--pn:#1c2029;--fg:#e4e8ef;--mu:#8b94a7;--bd:#2c3240;--ac:#6aa6e8;--ac2:#22304a;--ok:#5ccb8b;--mk:#5a4a14;color-scheme:dark}}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{display:grid;grid-template:48px 1fr/250px 1fr;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{grid-column:1/3;display:flex;align-items:center;gap:14px;padding:0 14px;background:var(--pn);border-bottom:1px solid var(--bd)}
header h1{font-size:15px;margin:0;font-weight:650}
#q{flex:1;max-width:420px;padding:6px 10px;border:1px solid var(--bd);border-radius:6px;background:var(--bg);color:var(--fg)}
.dot{width:9px;height:9px;border-radius:50%;background:var(--mu);display:inline-block}.dot.on{background:var(--ok)}
.mu{color:var(--mu)}label{display:flex;gap:5px;align-items:center;cursor:pointer}
aside{overflow:auto;padding:10px;border-right:1px solid var(--bd);background:var(--pn)}
.item{display:flex;justify-content:space-between;gap:8px;padding:4px 10px;border-radius:5px;cursor:pointer;white-space:nowrap}
.item:hover{background:var(--bg)}.item.on{background:var(--ac2);color:var(--ac);font-weight:600}.n{color:var(--mu);font-size:12px}
.cap{margin:14px 6px 4px;color:var(--mu);font-size:12px}
.chip{margin:2px;padding:2px 9px;border:1px solid var(--bd);border-radius:12px;background:none;color:var(--fg);cursor:pointer;font:inherit;font-size:12px}
.chip.on{background:var(--ac);border-color:var(--ac);color:#fff}
section{display:flex;flex-direction:column;min-height:0}
nav{display:flex;gap:2px;padding:8px 14px 0;border-bottom:1px solid var(--bd);background:var(--pn)}
.tab{padding:7px 14px;border:0;border-bottom:2px solid transparent;background:none;color:var(--mu);cursor:pointer;font:inherit}
.tab.on{color:var(--ac);border-color:var(--ac);font-weight:600}
#main{flex:1;overflow:auto;padding:16px}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px}
.stat{padding:8px 14px;background:var(--pn);border:1px solid var(--bd);border-radius:8px;min-width:96px}
.stat b{display:block;font-size:20px}
h3{margin:18px 0 8px;font-size:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}
.card{background:var(--pn);border:1px solid var(--bd);border-radius:8px;padding:10px 12px;min-width:0}
.card h4{margin:0 0 6px;font-size:13px;overflow-wrap:anywhere}
table{border-collapse:collapse;width:100%;background:var(--pn);border:1px solid var(--bd)}
th,td{text-align:left;padding:5px 9px;border-bottom:1px solid var(--bd);vertical-align:top}
th{position:sticky;top:-16px;background:var(--pn);font-weight:600;color:var(--mu)}
.kv{display:inline-block;margin:1px 8px 1px 0;white-space:nowrap}.kv b{color:var(--mu);font-weight:500;margin-right:4px}
.rng{position:relative;height:6px;margin:6px 0 2px;background:var(--bd);border-radius:3px;min-width:120px}
.rng i{position:absolute;top:0;bottom:0;background:var(--ac);opacity:.45;border-radius:3px}
.rng u{position:absolute;top:-2px;width:3px;height:10px;background:var(--ac)}
.charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}
svg{width:100%;height:auto;font-size:10px;fill:var(--mu)}svg .gr{stroke:var(--bd);stroke-width:1}
.leg{font-size:12px;color:var(--mu);display:flex;flex-wrap:wrap;gap:2px 10px}.leg i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px}
details{margin:2px 0}summary{cursor:pointer}.ind{margin-left:18px;border-left:1px solid var(--bd);padding-left:10px}
.leaf{margin-left:14px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px}
.number{color:var(--ac)}.string{color:var(--ok)}.boolean,.object{color:var(--mu)}
.raw{font:12.5px/1.5 ui-monospace,Menlo,Consolas,monospace}
.ln{display:flex;gap:12px;white-space:pre-wrap;overflow-wrap:anywhere}.ln .no{flex:none;width:44px;text-align:right;color:var(--mu);user-select:none}
.ln.mark{font-weight:700;background:var(--ac2)}.ln.gap{color:var(--mu)}.ln .k{color:var(--ac)}.ln .vv{color:var(--ok)}
mark{background:var(--mk);color:inherit}.note,.empty{color:var(--mu);padding:10px 0}
@media(max-width:800px){body{grid-template:48px auto 1fr/1fr}header{grid-column:1}aside{max-height:28vh;border-right:0;border-bottom:1px solid var(--bd)}}
.bar{display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center;margin-bottom:12px}
select{padding:4px 8px;border:1px solid var(--bd);border-radius:6px;background:var(--pn);color:var(--fg);font:inherit;max-width:100%}
.mini{margin-left:8px;padding:0 6px;border:1px solid var(--bd);border-radius:4px;background:none;color:var(--mu);font-size:11px;cursor:pointer;visibility:hidden}
summary:hover .mini{visibility:visible}.pv{color:var(--mu);font-size:12px;margin-left:8px}details[open]>summary .pv{display:none}
.leaf{cursor:copy}.leaf:hover{background:var(--ac2)}
tr.clk{cursor:pointer}tr.clk:hover td{background:var(--ac2)}tr.mu td{color:var(--mu)}td.df{background:var(--mk)}th.s{cursor:pointer}
.tb{position:relative;height:12px;min-width:180px;background:var(--bg);border-radius:3px}.tb i{position:absolute;left:0;top:0;bottom:0;background:var(--ac);opacity:.55;border-radius:3px}.tb u{position:absolute;top:5px;height:2px;background:var(--fg)}
#toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);padding:6px 14px;background:var(--fg);color:var(--bg);border-radius:6px;opacity:0;pointer-events:none;transition:opacity .2s;z-index:9}#toast.show{opacity:1}
</style></head><body>
<header><h1>__FILE__</h1><input id="q" type="search" placeholder="Filter lines (text, key, value)">
<span class="mu" id="stat"></span><span style="flex:1"></span>
<label><input type="checkbox" id="fol" checked> Follow</label><span class="dot" id="dot" title="connection"></span></header>
<aside><div id="secs"></div><div id="blks"></div></aside>
<section><nav id="tabs"></nav><div id="main"></div></section>
<div id="toast"></div>
<script>
const $=s=>document.querySelector(s);
const h=(t,a={},...c)=>{const e=document.createElement(t);for(const k in a){if(k==='class')e.className=a[k];else if(k==='html')e.innerHTML=a[k];else if(k.startsWith('on'))e.addEventListener(k.slice(2),a[k]);else e.setAttribute(k,a[k])}c.flat().forEach(x=>x!=null&&e.append(x.nodeType?x:document.createTextNode(x)));return e};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt=x=>typeof x!=='number'?String(x):(x!==0&&(Math.abs(x)>=1e6||Math.abs(x)<1e-3))?x.toExponential(2):String(+x.toFixed(4));
const nat=(a,b)=>a.localeCompare(b,undefined,{numeric:true});
const C=['#2b6cb0','#d9534f','#2f9e5e','#e0a100','#8e5bd0','#14a3a3','#e07a3f','#7a8599'];
const S={recs:[],secs:{},ver:-1,sec:'',blk:'',q:'',tab:'overview',follow:true,more:30,split:false,ts:{k:'',d:-1},jg:'',jd:1,jmode:'tree',jdiff:true,warn:[]};
const OWNQ=['metrics','charts','json'];
const inSec=r=>!S.sec||r.s===S.sec||r.s.startsWith(S.sec+'/');
function vis(){const q=OWNQ.includes(S.tab)?'':S.q.toLowerCase();return S.recs.filter(r=>inSec(r)&&(!S.blk||r.b===S.blk)&&(!q||r.t.toLowerCase().includes(q)))}

function side(){
 const cnt=new Map();
 for(const r of S.recs){let p='';for(const n of(r.s?r.s.split('/'):[])){p=p?p+'/'+n:n;cnt.set(p,(cnt.get(p)||0)+1)}}
 const box=$('#secs');
 box.replaceChildren(h('div',{class:'item'+(S.sec?'':' on'),onclick:()=>{S.sec='';S.blk='';render()}},'All lines',h('span',{class:'n'},S.recs.length)));
 for(const[p,n]of cnt){const d=p.split('/').length-1;
  box.append(h('div',{class:'item'+(S.sec===p?' on':''),style:`padding-left:${10+d*14}px`,title:S.secs[p]||p,onclick:()=>{S.sec=S.sec===p?'':p;S.blk='';render()}},p.split('/').pop(),h('span',{class:'n'},n)))}
 const bs=[...new Set(S.recs.filter(inSec).map(r=>r.b).filter(Boolean))].sort(nat);
 $('#blks').replaceChildren(...(bs.length?[h('div',{class:'cap'},'Blocks')]:[]),...bs.map(b=>h('button',{class:'chip'+(S.blk===b?' on':''),onclick:()=>{S.blk=S.blk===b?'':b;render()}},b)));
}
const toast=m=>{const t=$('#toast');t.textContent=m;t.className='show';clearTimeout(toast.t);toast.t=setTimeout(()=>t.className='',1500)};
const copy=(s,m)=>navigator.clipboard.writeText(s).then(()=>toast(m||'Copied'),()=>toast('Copy failed'));
const mk=(t,q)=>{t=esc(t);return q?t.replace(new RegExp(esc(q).replace(/[.*+?^${}()|[\]\\]/g,'\\$&'),'gi'),m=>'<mark>'+m+'</mark>'):t};
const chips=a=>a.map(([k,x])=>h('span',{class:'kv'},h('b',{},k),fmt(x)));
const empty=m=>h('div',{class:'empty'},m);
const gk=r=>r.s+(r.o?' #'+(r.o+1):'');
function facts(j){const out=[];(function w(o,p){if(out.length>=40)return;
 if(Array.isArray(o)&&o.length>6)out.push([p,'['+o.length+' items]']);
 else if(o&&typeof o==='object')for(const k in o)w(o[k],p?p+'.'+k:k);else out.push([p,o])})(j,'');return out}
/* metrics = numeric leaves of the JSON. Array indices become [] (or [event=frontier] when items carry a discriminator field) */
const DK=['event','name','type','kind','label','phase','op'];
function dkey(a){if(!a.length||a.some(e=>!e||typeof e!=='object'))return'';
 return DK.find(k=>a.every(e=>typeof e[k]==='string')&&new Set(a.map(e=>e[k])).size<=24)||''}
function jm(j){const out=[];
 (function w(o,p,ix){if(out.length>4000)return;
  if(typeof o==='number'&&isFinite(o))out.push({p,v:o,x:ix.join('.')});
  else if(Array.isArray(o)){const d=dkey(o);o.forEach((e,i)=>w(e,p+'['+(d?d+'='+e[d]:'')+']',ix.concat(i)))}
  else if(o&&typeof o==='object')for(const k in o)w(o[k],p?p+'.'+k:k,ix)})(j,'',[]);return out}
function mgroups(rs,split){const g=new Map();
 for(const r of rs){if(r.j===undefined)continue;const m=r._m||(r._m=jm(r.j)),sn=(r.b||'all')+(split&&r.o?' \u00b7 run '+(r.o+1):'');
  for(const q of m){const id=r.s+'\u203a'+q.p;let e=g.get(id);
   if(!e)g.set(id,e={id:(r.s?r.s+' \u203a ':'')+q.p,sec:r.s,path:q.p,s:new Map(),n:0});
   let a=e.s.get(sn);if(!a)e.s.set(sn,a=[]);a.push({y:q.v,i:r.i,x:q.x});e.n++}}
 const out=[...g.values()];for(const e of out)e.vary=new Set([...e.s.values()].flat().map(p=>p.y)).size>1;return out}
/* timers */
function tmain(f){const ks=Object.keys(f).filter(k=>typeof f[k]==='number');
 return ks.find(k=>/^(avg|mean|value|total|time|elapsed|seconds)/.test(k))||ks.find(k=>!/^(n|count)$/.test(k))||''}
function timers(rs,top){
 let ts=rs.filter(r=>r.tm);
 if(!ts.length)return empty('No timer records in this selection.');
 const mkey=tmain(ts[0].tm.f),sfx=mkey.replace(/^(avg|mean|value)/,''),F=(r,k)=>r.tm.f[k],T=S.ts,hasB=ts.some(r=>r.b);
 const cols=[];ts.forEach(r=>{for(const k in r.tm.f)if(typeof r.tm.f[k]==='number'&&!cols.includes(k))cols.push(k)});
 const mx=ts.reduce((a,r)=>Math.max(a,F(r,mkey)||0,F(r,'max'+sfx)||0),1e-12);
 const nm=r=>String(r.tm.f.name??r.tm.f.id??r.tm.type);
 if(top)ts=[...ts].sort((a,b)=>(F(b,mkey)||0)-(F(a,mkey)||0)).slice(0,top);
 else if(T.k)ts=[...ts].sort((a,b)=>T.d*(T.k==='name'?nm(a).localeCompare(nm(b)):T.k==='b'?nat(a.b,b.b):(F(a,T.k)||0)-(F(b,T.k)||0)));
 const p=x=>(x/mx*100).toFixed(1)+'%';
 const bar=r=>{const v=F(r,mkey);if(typeof v!=='number')return'';const lo=F(r,'min'+sfx),hi=F(r,'max'+sfx),rg=typeof lo==='number'&&typeof hi==='number';
  return h('div',{class:'tb',title:mkey+': '+fmt(v)+(rg?' (min '+fmt(lo)+', max '+fmt(hi)+')':'')},h('i',{style:'width:'+p(v)}),rg?h('u',{style:'left:'+p(lo)+';width:'+Math.max(.5,(hi-lo)/mx*100)+'%'}):null)};
 const hd=[['name','Timer'],['','Group'],...(hasB?[['b','Block']]:[]),['','Duration ('+mkey+')'],...cols.map(c=>[c,c])];
 return h('table',{},h('tr',{},hd.map(([k,l])=>h('th',{class:top||!k?'':'s',onclick:top||!k?null:()=>{S.ts={k,d:T.k===k?-T.d:-1};render()}},l+(k&&T.k===k&&!top?(T.d<0?' \u25bc':' \u25b2'):'')))),
  ts.map(r=>h('tr',{title:r.t},h('td',{},nm(r)),h('td',{class:'mu'},r.tm.f.group??''),...(hasB?[h('td',{},r.b)]:[]),h('td',{},bar(r)),...cols.map(c=>h('td',{},F(r,c)==null?'':fmt(F(r,c)))))))
}
function overview(rs){
 const js=rs.filter(r=>r.j!==undefined),tm=rs.filter(r=>r.tm);
 const st=[['Lines',rs.length],['Sections',new Set(rs.map(r=>r.s).filter(Boolean)).size],['Blocks',new Set(rs.map(r=>r.b).filter(Boolean)).size],['JSON records',js.length],['Metrics',mgroups(rs).length],['Timers',tm.length]];
 const by=new Map();for(const r of rs){const e=by.get(r.s)||{n:0,j:0,t:0};e.n++;if(r.j!==undefined)e.j++;if(r.tm)e.t++;by.set(r.s,e)}
 let gj=js.filter(r=>!r.b);if(!gj.length){const seen=new Set();gj=js.filter(r=>!seen.has(gk(r))&&seen.add(gk(r)))}
 const out=rs.filter(r=>!r.s&&r.k==='text'&&r.t.trim()),os=out.length>14?[...out.slice(0,10),...out.slice(-4)]:out;
 return h('div',{},h('div',{class:'stats'},st.map(([a,b])=>h('div',{class:'stat'},h('b',{},b),a))),
  (S.warn||[]).length?h('div',{class:'card',style:'margin-bottom:12px'},h('h4',{},'Parser notes'),S.warn.slice(0,15).map(w=>h('div',{class:'mu'},w))):null,
  tm.length?[h('h3',{},'Slowest timers'),h('div',{class:'card'},timers(rs,6))]:null,
  gj.length?[h('h3',{},'Key values from JSON'),h('div',{class:'cards'},gj.slice(0,12).map(r=>h('div',{class:'card'},h('h4',{},(r.s||'top level')+(r.b?' \u00b7 '+r.b:'')+' \u00b7 line '+(r.i+1)),chips(facts(r.j)))))]:null,
  h('h3',{},'Sections'),
  h('table',{},h('tr',{},['Section','Description','Lines','JSON','Timers'].map(x=>h('th',{},x))),
   [...by].map(([p,e])=>h('tr',{},h('td',{},p||'(top level)'),h('td',{class:'mu'},S.secs[p]||''),h('td',{},e.n),h('td',{},e.j),h('td',{},e.t)))),
  os.length?[h('h3',{},'Text outside sections'),h('div',{class:'card raw'},os.map(r=>h('div',{class:'ln'},h('span',{class:'no'},r.i+1),h('span',{},r.t))))]:null)
}
function chart(g){
 const W=380,H=190,L=48,R=10,T=10,B=26,ser=[...g.s];
 const ys=ser.flatMap(([,p])=>p.map(q=>q.y));let lo=Math.min(...ys),hi=Math.max(...ys);
 const bars=ser.length>1&&ser.every(([,p])=>p.length===1);if(bars)lo=Math.min(0,lo);if(lo===hi){lo-=1;hi+=1}
 const n=bars?ser.length:Math.max(...ser.map(([,p])=>p.length));
 const X=i=>L+(bars?(i+.5)/n:(n>1?i/(n-1):.5))*(W-L-R),Y=v=>T+(1-(v-lo)/(hi-lo))*(H-T-B);
 let s=`<svg viewBox="0 0 ${W} ${H}">`;
 for(let k=0;k<=3;k++){const v=lo+(hi-lo)*k/3,y=Y(v);s+=`<line class="gr" x1="${L}" x2="${W-R}" y1="${y}" y2="${y}"/><text x="${L-4}" y="${y+3}" text-anchor="end">${fmt(v)}</text>`}
 if(bars)ser.forEach(([nm,p],i)=>{const x=X(i),w=(W-L-R)/n*.6,y=Y(p[0].y),y0=Y(Math.max(lo,0));
  s+=`<rect x="${x-w/2}" y="${Math.min(y,y0)}" width="${w}" height="${Math.max(1,Math.abs(y0-y))}" fill="${C[i%8]}"><title>${esc(nm||'all')}: ${fmt(p[0].y)} (line ${p[0].i+1})</title></rect>`;
  if(n<=12)s+=`<text x="${x}" y="${H-10}" text-anchor="middle">${esc(nm.replace(/^\D+/,'')||nm||'\u00b7')}</text>`});
 else{ser.forEach(([nm,p],i)=>{const c=C[i%8];
  s+=`<path d="${p.map((q,j)=>(j?'L':'M')+X(j).toFixed(1)+' '+Y(q.y).toFixed(1)).join('')}" fill="none" stroke="${c}" stroke-width="1.5"/>`;
  if(p.length<=80)p.forEach((q,j)=>{s+=`<circle cx="${X(j)}" cy="${Y(q.y)}" r="2.5" fill="${c}"><title>${esc(nm||'all')} #${j}${q.x?' ['+q.x+']':''}: ${fmt(q.y)} (line ${q.i+1})</title></circle>`})});
  s+=`<text x="${L}" y="${H-10}">0</text><text x="${W-R}" y="${H-10}" text-anchor="end">${n-1}</text>`}
 s+='</svg>';
 const leg=!bars&&ser.length>1?h('div',{class:'leg'},ser.slice(0,12).map(([nm],i)=>h('span',{},h('i',{style:`background:${C[i%8]}`}),nm||'all'))):null;
 return h('div',{class:'card'},h('h4',{},g.id),h('div',{html:s}),leg)
}
const mfilter=gs=>{const q=S.q.trim().toLowerCase();return q?gs.filter(g=>g.id.toLowerCase().includes(q)):gs};
function charts(rs){
 const gs=mfilter(mgroups(rs,S.split)).filter(g=>g.n>1&&g.vary);
 const cb=h('input',{type:'checkbox',onchange:e=>{S.split=e.target.checked;render()}});cb.checked=S.split;
 return h('div',{},h('div',{class:'bar'},h('label',{},cb,'Split repeated runs'),h('span',{class:'mu'},gs.length+' varying metrics from JSON. Lines: one series per rank, x = position in the JSON; bars: one bar per rank.')),
  gs.length?[h('div',{class:'charts'},gs.slice(0,S.more).map(chart)),gs.length>S.more?h('p',{},h('button',{class:'chip',onclick:()=>{S.more+=30;render()}},'Show more ('+(gs.length-S.more)+' left)')):null]:empty('No varying numeric JSON values in this selection.'))
}
function metrics(rs){
 const gs=mfilter(mgroups(rs,S.split));
 if(!gs.length)return empty('No numeric JSON values in this selection.');
 const tb=h('table',{},h('tr',{},['Metric (click for chart)','Series','Points','Min','Mean','Max'].map(x=>h('th',{},x))));
 gs.slice(0,400).forEach(g=>{let n=0,mn=Infinity,mx=-Infinity,sum=0;for(const a of g.s.values())for(const p of a){n++;sum+=p.y;mn=Math.min(mn,p.y);mx=Math.max(mx,p.y)}
  const tr=h('tr',{class:'clk'+(g.vary?'':' mu')},h('td',{},h('span',{class:'mu'},g.sec?g.sec+' \u203a ':''),g.path),h('td',{},g.s.size),h('td',{},n),h('td',{},fmt(mn)),h('td',{},fmt(sum/n)),h('td',{},fmt(mx)));
  let ex=null;tr.onclick=()=>{if(ex){ex.remove();ex=null}else{ex=h('tr',{},h('td',{colspan:6},chart(g)));tr.after(ex)}};tb.append(tr)});
 return h('div',{},h('div',{class:'note'},gs.length+' metrics derived from JSON numeric values (greyed = constant).'+(gs.length>400?' Showing first 400.':'')),tb)
}
const jmatch=(o,key,q)=>String(key).toLowerCase().includes(q)||(o&&typeof o==='object'?Object.keys(o).some(k=>jmatch(o[k],k,q)):String(o).toLowerCase().includes(q));
const pvw=o=>{const ks=Object.keys(o),sv=v=>{if(v&&typeof v==='object')return Array.isArray(v)?'[\u2026]':'{\u2026}';const s=JSON.stringify(v);return s.length>24?s.slice(0,23)+'\u2026':s};
 return (Array.isArray(o)?ks.slice(0,5).map(k=>sv(o[k])):ks.slice(0,5)).join(', ')+(ks.length>5?', \u2026':'')};
function atable(a){
 const cols=[];a.slice(0,200).forEach(e=>{for(const k in e)if(!cols.includes(k)&&cols.length<14)cols.push(k)});
 const cell=v=>v&&typeof v==='object'?(Array.isArray(v)?'['+v.length+']':'{\u2026}'):fmt(v);
 return h('div',{style:'overflow:auto;max-height:420px'},h('table',{},h('tr',{},cols.map(c=>h('th',{},c))),a.slice(0,200).map(e=>h('tr',{},cols.map(c=>h('td',{},c in e?cell(e[c]):''))))))}
let jbud=0;
function jt(o,key,path,depth,q){
 if(q&&!jmatch(o,key,q))return null;
 const isc=o&&typeof o==='object',ks=isc?Object.keys(o):[];
 if(!ks.length||(Array.isArray(o)&&o.length<=16&&o.every(x=>x===null||typeof x!=='object')))
  return h('div',{class:'leaf',title:(path||key)+' \u2014 click to copy path',onclick:()=>copy(path||key,'Copied path: '+(path||key)),html:'<b>'+mk(key,q)+'</b>: <span class="'+(o===null?'object':typeof o)+'">'+mk(JSON.stringify(o),q)+'</span>'});
 const arr=Array.isArray(o),d=h('details',{}),body=h('div',{class:'ind'}),stop=f=>e=>{e.preventDefault();e.stopPropagation();f()};
 const sm=h('summary',{html:'<b>'+mk(key,q)+'</b> <i class="mu">'+(arr?'['+ks.length+']':'{'+ks.length+'}')+'</i><span class="pv">'+esc(pvw(o))+'</span>'});
 sm.append(h('button',{class:'mini',title:'Copy this subtree as JSON',onclick:stop(()=>copy(JSON.stringify(o,null,2),'Copied JSON'))},'copy'));
 let tv=false,done=false;
 const draw=()=>{done=true;body.replaceChildren(...(tv?[atable(o)]:ks.map(k=>jt(o[k],k,arr?path+'['+k+']':path?path+'.'+k:k,depth+1,q)).filter(Boolean)))};
 if(arr&&o.length>1&&o.every(x=>x&&typeof x==='object'&&!Array.isArray(x)))sm.append(h('button',{class:'mini',title:'Toggle table view of this array',onclick:stop(()=>{tv=!tv;d.open=true;draw()})},'table'));
 d.append(sm,body);
 if(q||(depth<S.jd&&jbud-->0)){d.open=true;draw()}else d.addEventListener('toggle',()=>d.open&&!done&&draw());
 return d}
function cmp(recs,q){
 const rk=recs.filter(r=>r.b),flat=j=>{const m=new Map();(function w(o,p){if(m.size>20000)return;if(o&&typeof o==='object'&&Object.keys(o).length)for(const k in o)w(o[k],Array.isArray(o)?p+'['+k+']':p?p+'.'+k:k);else m.set(p,o)})(j,'');return m};
 const maps=rk.map(r=>flat(r.j)),paths=[...new Set(maps.flatMap(m=>[...m.keys()]))];
 const all=paths.map(p=>{const v=maps.map(m=>m.has(p)?m.get(p):undefined);return{p,v,same:v.every(x=>JSON.stringify(x)===JSON.stringify(v[0]))}});
 const rows=all.filter(x=>(!S.jdiff||!x.same)&&(!q||x.p.toLowerCase().includes(q)||x.v.some(y=>String(y).toLowerCase().includes(q))));
 return h('div',{},h('div',{class:'note'},all.filter(x=>!x.same).length+' of '+all.length+' values differ across '+rk.length+' ranks'+(rows.length>1500?' (showing first 1500)':'')),
  h('table',{},h('tr',{},h('th',{},'Path'),rk.map(r=>h('th',{},r.b))),rows.slice(0,1500).map(x=>h('tr',{},h('td',{},x.p),x.v.map(y=>h('td',{class:x.same?'':'df'},y===undefined?'\u2014':fmt(y)))))))}
function json(rs){
 const js=rs.filter(r=>r.j!==undefined);
 if(!js.length)return empty('No JSON records in this selection.');
 const g=new Map();js.forEach(r=>{const k=gk(r);if(!g.has(k))g.set(k,[]);g.get(k).push(r)});
 if(!g.has(S.jg))S.jg=[...g.keys()][0];
 const recs=g.get(S.jg),q=S.q.trim().toLowerCase(),canCmp=recs.filter(r=>r.b).length>1,mode=canCmp?S.jmode:'tree';
 const sel=h('select',{onchange:e=>{S.jg=e.target.value;render()}},[...g].map(([k,v])=>h('option',{value:k},(k||'(top level)')+' \u2014 '+v.length+' record'+(v.length>1?'s':''))));sel.value=S.jg;
 const b=(t,f,title,on)=>h('button',{class:'chip'+(on?' on':''),title:title||'',onclick:f},t);
 const dc=h('input',{type:'checkbox',onchange:e=>{S.jdiff=e.target.checked;render()}});dc.checked=S.jdiff;
 const bar=h('div',{class:'bar'},sel,
  canCmp?[b('Tree',()=>{S.jmode='tree';render()},'',mode==='tree'),b('Compare ranks',()=>{S.jmode='cmp';render()},'Values side by side, one column per rank',mode==='cmp')]:null,
  mode==='tree'?[b('Collapse',()=>{S.jd=0;render()}),b('Level 1',()=>{S.jd=1;render()}),b('Level 2',()=>{S.jd=2;render()}),b('Expand all',()=>{S.jd=99;render()})]:h('label',{},dc,'Differences only'),
  b('Copy JSON',()=>copy(JSON.stringify(recs.length>1?Object.fromEntries(recs.map(r=>[r.b||'line '+(r.i+1),r.j])):recs[0].j,null,2),'Copied JSON')),
  h('span',{class:'mu'},'Click a value to copy its path; hover a node for copy / table.'));
 jbud=3000;
 if(mode==='cmp')return h('div',{},bar,cmp(recs,q));
 const trees=recs.slice(0,200).map(r=>jt(r.j,(r.b||'JSON')+' \u00b7 line '+(r.i+1),'',0,q)).filter(Boolean);
 return h('div',{},bar,trees.length?trees:empty('No keys or values match "'+S.q+'".'))
}
const hl=(t,q)=>{let s=esc(t).replace(/([A-Za-z_](?:[\w.\-\/]|&gt;)*)=([^\s,;]+)/g,'<span class="k">$1</span>=<span class="vv">$2</span>');
 if(q)s=s.replace(new RegExp('(?![^<]*>)('+esc(q).replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')','gi'),'<mark>$1</mark>');return s};
function raw(rs){
 const cap=3000,sl=rs.length>cap?rs.slice(-cap):rs;
 return h('div',{class:'raw'},rs.length>cap?h('div',{class:'note'},`Showing last ${cap} of ${rs.length} lines. Narrow with the filters.`):null,
  sl.map(r=>h('div',{class:'ln '+r.k},h('span',{class:'no'},r.i+1),h('span',{html:hl(r.t,S.q)}))))
}
const TABS={overview,timers,metrics,charts,json,raw};
function render(){
 const m=$('#main'),st=m.scrollTop,bot=m.scrollHeight-st-m.clientHeight<80;
 side();
 const rs=vis();$('#q').placeholder=({metrics:'Filter metrics by name',charts:'Filter metrics by name',json:'Search keys and values in JSON'})[S.tab]||'Filter lines (text, key, value)';$('#stat').textContent=`${rs.length} / ${S.recs.length} lines`;
 $('#tabs').replaceChildren(...Object.keys(TABS).map(t=>h('button',{class:'tab'+(S.tab===t?' on':''),onclick:()=>{S.tab=t;render()}},t[0].toUpperCase()+t.slice(1))));
 m.replaceChildren(TABS[S.tab](rs));
 m.scrollTop=(S.tab==='raw'&&S.follow&&bot)?m.scrollHeight:st;
}
async function poll(){
 try{if(S.follow||S.ver<0){const d=await(await fetch('/api/data?v='+S.ver)).json();
   if(!d.same){S.recs=d.recs;S.secs=d.secs;S.warn=d.warnings||[];S.ver=d.ver;if(d.follow===false){S.follow=false;$('#fol').checked=false;$('#fol').disabled=true}render()}}
  $('#dot').className='dot on'}catch(e){$('#dot').className='dot'}
 setTimeout(poll,1000)
}
let tm;$('#q').addEventListener('input',e=>{clearTimeout(tm);tm=setTimeout(()=>{S.q=e.target.value.trim();render()},150)});
$('#fol').addEventListener('change',e=>{S.follow=e.target.checked;render()});
render();poll();
</script></body></html>
"""

if __name__ == '__main__':
    main()