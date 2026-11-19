"use strict";
const LANG="id";
const LOCALE=LANG==="en"?"en-US":"id-ID";
let D, atlasCtx, CLS, SUMMARY=null;
const DSC = {};               // dataset key -> {D, atlasCtx}
const WCACHE = {};            // "<dataset>/<model>-<slug>" -> weights
/* One selection drives the whole page: dataset (S.ds) -> model (S.model) -> optimizer (S.opt).
   MLP and CNN run in the browser; EfficientNetV2B0 results come from exported predictions. */
const S = {ds:"cifar10", model:"cnn", opt:"adam", inv:true, cur:null, chMode:"top", sel:null, status:"all", agree:"all", cls:-1, pred:null, sort:"hi", shown:24, uploads:[], cmpAll:false, cmpB:""};
const wk = (slug,ds=S.ds,model=S.model)=>ds+"/"+model+"-"+slug;
const $ = s=>document.querySelector(s);
const nf = (v,d=0)=>v.toLocaleString(LOCALE,{minimumFractionDigits:d,maximumFractionDigits:d});
const pct = (v,d=1)=>nf(v*100,d)+"%";
const css = n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const hex = h=>{h=h.replace("#","");return [0,2,4].map(i=>parseInt(h.slice(i,i+2),16));};
const MNAME={mlp:"MLP",cnn:"CNN",effnet:"EfficientNetV2B0"};
const MODELS=["mlp","cnn","effnet"];
const EFF_PARAMS=5932122;     // model.count_params() in experiments/runs/budget/*/effnet-*/seed-*/run.json
const OPTS = ()=>D.models[S.model].opts;    // optimizers that run in the browser (MLP or CNN)
const MODEL = ()=>MNAME[S.model];
const optOf = ()=>curE();

function f16(b64){const s=atob(b64),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);
  const h=new Uint16Array(u.buffer),o=new Float32Array(h.length);
  for(let i=0;i<h.length;i++){const x=h[i],sg=x&0x8000?-1:1,e=(x>>10)&31,f=x&1023;
    o[i]=e===0?sg*f*Math.pow(2,-24):e===31?(f?NaN:sg*Infinity):sg*(1+f/1024)*Math.pow(2,e-15);}
  return o;}
async function loadW(slug,ds=S.ds,model=S.model){
  const key=wk(slug,ds,model);if(WCACHE[key])return WCACHE[key];
  const r=await fetch(`data/${ds}/w-${model}-${slug}.json`);if(!r.ok)throw new Error(r.status);const j=await r.json();
  const L=j.layers.map(l=>({kind:l.kind,w:f16(l.w),b:f16(l.b),shape:l.shape}));
  const W=model==="cnn"?{model,conv1:L[0],conv2:L[1],dense:L[2],output:L[3]}:{model,h1:L[0],h2:L[1],output:L[2]};
  W.ref=j.ref;WCACHE[key]=W;return W;}
async function loadDS(ds){
  if(DSC[ds])return DSC[ds];
  const r=await fetch(`data/${ds}/data.json`);if(!r.ok)throw new Error(r.status);const d=await r.json();
  const img=new Image();await new Promise((ok,no)=>{img.onload=ok;img.onerror=no;img.src=d.atlas;});
  const c=document.createElement("canvas");c.width=img.width;c.height=img.height;const ctx=c.getContext("2d",{willReadFrequently:true});ctx.drawImage(img,0,0);
  DSC[ds]={D:d,atlasCtx:ctx};return DSC[ds];}
/* input geometry of the current dataset: 32 → 30 → 15 → 13 → 6 for CIFAR-10, 28 → 26 → 13 → 11 → 5 for Fashion-MNIST */
function G(){const H=D.shape[0],C=D.shape[2],c1=H-2,p1=Math.floor(c1/2),c2=p1-2,p2=Math.floor(c2/2);return {H,C,c1,p1,c2,p2,flat:p2*p2*64};}
const CHN=()=>G().C===3?["merah","hijau","biru"]:["abu-abu"];
const CHN_CAP=()=>G().C===3?["Merah","Hijau","Biru"]:["Abu-abu"];

/* ---------- trained configurations ("entries") of a dataset ----------
   MLP and CNN: the optimizer sweep (10 epochs, seed 42, full training split), 8 optimizers each.
   EfficientNetV2B0: the fine-tuning branches on the small subset, lowest finished seed per optimizer.
   Only finished runs become entries; planned runs are never shown. */
const EXPC={}, RUNC={}, ENTC={};
async function loadExp(ds=S.ds){if(EXPC[ds])return EXPC[ds];
  const r=await fetch(`data/${ds}/experiments.json`);if(!r.ok)throw new Error(r.status);const x=await r.json();
  x.byKey={};x.runs.forEach(r=>x.byKey[r.key]=r);EXPC[ds]=x;return x;}
function u8(b64){const s=atob(b64),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}
async function loadRun(key,ds=S.ds){const id=ds+"/"+key;if(RUNC[id])return RUNC[id];
  const r=await fetch(`data/${ds}/runs/${key}.json`);if(!r.ok)throw new Error(r.status);const j=await r.json();
  if(j.key!==key)throw new Error("key mismatch");
  j.p=u8(j.preds);j.c=u8(j.conf);RUNC[id]=j;return j;}
const EFF_OPTS={adam:"Adam",adamw:"AdamW",nadam:"Nadam"};
function effRun(X,opt){return X.runs.filter(r=>r.study==="budget"&&r.model==="effnet"&&r.opt===opt&&r.status==="complete").sort((a,b)=>a.seed-b.seed)[0];}
function buildEntries(ds){const d=DSC[ds].D,X=EXPC[ds],out=[];
  for(const m of ["mlp","cnn"])for(const o of d.models[m].opts){const key=`historical__${m}-${o.slug}__s42`,r=X.byKey[key];if(!r)continue;
    out.push({id:m+"-"+o.slug,model:m,opt:o.slug,name:o.name,acc:r.acc,key,seed:42,hist:o.hist,warm:0,best:null,params:d.models[m].params,browser:true});}
  for(const o of Object.keys(EFF_OPTS)){const r=effRun(X,o);if(!r)continue;
    out.push({id:"effnet-"+o,model:"effnet",opt:o,name:EFF_OPTS[o],acc:r.acc,key:r.key,seed:r.seed,hist:null,warm:r.warmup_epochs||2,best:r.best_epoch,params:EFF_PARAMS,browser:false});}
  const sorted=[...out].sort((a,b)=>b.acc-a.acc);
  out.forEach(e=>{e.rank=1+out.filter(x=>x.acc>e.acc).length;});   // ties share a rank
  ENTC[ds]={all:out,sorted};return ENTC[ds];}
const E=()=>ENTC[S.ds].all;
const curE=()=>E().find(e=>e.model===S.model&&e.opt===S.opt);
const ofModel=m=>E().filter(e=>e.model===m);
const label=e=>`${MNAME[e.model]} · ${e.name}`;
/* EfficientNet curves live in the run files; MLP/CNN curves are already in data.json */
async function ensureHist(e,ds=S.ds){if(e.hist)return e.hist;const R=await loadRun(e.key,ds);
  const h=R.history;e.hist={accuracy:h.map(x=>x.accuracy),val_accuracy:h.map(x=>x.val_accuracy),loss:h.map(x=>x.loss),val_loss:h.map(x=>x.val_loss)};return e.hist;}

/* ---------- forward pass (same math as Keras, channels-last) ---------- */
function conv(x,H,Wd,C,L){const k=L.shape,O2=k[3],oh=H-2,ow=Wd-2,y=new Float32Array(oh*ow*O2),w=L.w,b=L.b;
  for(let i=0;i<oh;i++)for(let j=0;j<ow;j++){const o=(i*ow+j)*O2;for(let f=0;f<O2;f++)y[o+f]=b[f];
    for(let di=0;di<3;di++)for(let dj=0;dj<3;dj++){const xo=((i+di)*Wd+(j+dj))*C,wo=(di*3+dj)*C*O2;
      for(let c=0;c<C;c++){const v=x[xo+c];if(v===0)continue;const wc=wo+c*O2;for(let f=0;f<O2;f++)y[o+f]+=v*w[wc+f];}}
    for(let f=0;f<O2;f++)if(y[o+f]<0)y[o+f]=0;}
  return {d:y,h:oh,w:ow,c:O2};}
function pool(t){const oh=Math.floor(t.h/2),ow=Math.floor(t.w/2),C=t.c,y=new Float32Array(oh*ow*C);
  for(let i=0;i<oh;i++)for(let j=0;j<ow;j++)for(let c=0;c<C;c++){let m=-Infinity;
    for(let a=0;a<2;a++)for(let b=0;b<2;b++){const v=t.d[((2*i+a)*t.w+(2*j+b))*C+c];if(v>m)m=v;}y[(i*ow+j)*C+c]=m;}
  return {d:y,h:oh,w:ow,c:C};}
function dense(x,L,relu){const [I,O2]=L.shape,y=new Float32Array(O2);for(let o=0;o<O2;o++)y[o]=L.b[o];
  for(let i=0;i<I;i++){const v=x[i];if(v===0)continue;const r=i*O2;for(let o=0;o<O2;o++)y[o]+=v*L.w[r+o];}
  if(relu)for(let o=0;o<O2;o++)if(y[o]<0)y[o]=0;return y;}
function forward(px,W,full=true){
  const g=G(),inp={d:px,h:g.H,w:g.H,c:g.C};let z,out;
  if(W.model==="mlp"){   // Flatten keeps the channels-last order, so the pixel array is already the flattened vector
    const h1=dense(px,W.h1,true),h2=dense(h1,W.h2,true);z=dense(h2,W.output,false);out={model:"mlp",inp,flat:px,h1,h2};}
  else{const c1=conv(px,g.H,g.H,g.C,W.conv1),p1=pool(c1),c2=conv(p1.d,p1.h,p1.w,32,W.conv2),p2=pool(c2);
    const h=dense(p2.d,W.dense,true);z=dense(h,W.output,false);out={model:"cnn",inp,c1,p1,c2,p2,h};}
  const m=Math.max(...z),e=z.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);
  const prob=Array.from(e,v=>v/s);
  return full?{...out,prob}:{prob};}

/* ---------- drawing ---------- */
function ramp(t){const a=hex(css("--map0")),b=hex(css("--map1")),c=hex(css("--map2"));
  const [p,q,u]=t<.5?[a,b,t*2]:[b,c,(t-.5)*2];return [0,1,2].map(i=>Math.round(p[i]+(q[i]-p[i])*u));}
function chanOf(t,c){const n=t.h*t.w,o=new Float32Array(n);for(let i=0;i<n;i++)o[i]=t.d[i*t.c+c];return o;}
function paint(cv,vals,w,h,max,mode){cv.width=w;cv.height=h;const ctx=cv.getContext("2d"),im=ctx.createImageData(w,h);
  for(let i=0;i<w*h;i++){let rgb;const v=vals[i];
    if(mode==="gray"){const g=Math.round(v*255);rgb=[g,g,g];}else rgb=ramp(max>0?Math.min(1,v/max):0);
    im.data.set([...rgb,255],i*4);}ctx.putImageData(im,0,0);}
function paintDiv(cv,vals,w,h,max){cv.width=w;cv.height=h;const ctx=cv.getContext("2d"),im=ctx.createImageData(w,h);
  const neg=hex(css("--bad")),pos=hex(css("--accent")),z=hex(css("--surface"));
  for(let i=0;i<w*h;i++){const t=Math.max(-1,Math.min(1,vals[i]/max)),e=t<0?neg:pos,u=Math.abs(t);
    im.data.set([0,1,2].map(k=>Math.round(z[k]+(e[k]-z[k])*u)).concat(255),i*4);}ctx.putImageData(im,0,0);}
function tilePixels(k){const {H:s,C}=G(),c=k%D.cols,r=Math.floor(k/D.cols),im=atlasCtx.getImageData(c*s,r*s,s,s).data,px=new Float32Array(s*s*C);
  for(let i=0;i<s*s;i++)for(let ch=0;ch<C;ch++)px[i*C+ch]=im[i*4+ch]/255;return px;}
function drawTile(cv,k){const s=G().H;cv.width=s;cv.height=s;const c=k%D.cols,r=Math.floor(k/D.cols);cv.getContext("2d").drawImage(atlasCtx.canvas,c*s,r*s,s,s,0,0,s,s);}
function drawPx(cv,px){const {H:s,C}=G();cv.width=s;cv.height=s;const ctx=cv.getContext("2d"),im=ctx.createImageData(s,s);
  for(let i=0;i<s*s;i++){const r=px[i*C],g=px[i*C+(C>1?1:0)],b=px[i*C+(C>1?2:0)];im.data.set([r*255,g*255,b*255,255],i*4);}ctx.putImageData(im,0,0);}
/* an upload keeps its 96 × 96 center crop; the model input is derived from it for the current dataset */
function uploadPx(u){const {H:s,C}=G(),cv=document.createElement("canvas");cv.width=s;cv.height=s;const ctx=cv.getContext("2d");
  ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality="high";ctx.drawImage(u.src,0,0,s,s);
  const d=ctx.getImageData(0,0,s,s).data,px=new Float32Array(s*s*C);
  for(let i=0;i<s*s;i++){if(C===3){for(let c=0;c<3;c++)px[i*3+c]=d[i*4+c]/255;}
    else{const y=(0.299*d[i*4]+0.587*d[i*4+1]+0.114*d[i*4+2])/255;px[i]=S.inv?1-y:y;}}
  return px;}
function curPx(){return S.cur.kind==="g"?tilePixels(S.cur.k):uploadPx(S.uploads[S.cur.i]);}

/* ---------- strip ---------- */
function buildStrip(){const st=$("#strip");st.innerHTML="";
  const items=[...D.demo.map(k=>({kind:"g",k})),...S.uploads.map((u,i)=>({kind:"u",i}))];
  items.forEach(it=>{const b=document.createElement("button");b.type="button";b.className="thumb";
    const cv=document.createElement("canvas"),sp=document.createElement("span");
    if(it.kind==="g"){drawTile(cv,it.k);sp.textContent=CLS[D.pool[it.k].t];b.setAttribute("aria-label","Periksa contoh "+CLS[D.pool[it.k].t]);}
    else{drawPx(cv,uploadPx(S.uploads[it.i]));sp.textContent="Unggahan "+(it.i+1);b.setAttribute("aria-label","Periksa gambar unggahan "+(it.i+1));}
    const on=S.cur&&S.cur.kind===it.kind&&(it.kind==="g"?S.cur.k===it.k:S.cur.i===it.i);
    b.setAttribute("aria-pressed",on?"true":"false");b.append(cv,sp);b.onclick=()=>select(it);st.append(b);});
  const up=document.createElement("button");up.type="button";up.className="upload";
  up.innerHTML='<svg width="22" height="22" viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M11 4v14M4 11h14"/></svg>Unggah';
  up.setAttribute("aria-label","Unggah gambar sendiri");up.onclick=()=>$("#file").click();
  st.append(up);}
function msg(t,err){const m=$("#stripMsg");m.textContent=t;m.classList.toggle("err",!!err);}
async function handleFile(f){
  if(!f.type.startsWith("image/")){msg("Berkas ini bukan gambar. Pilih JPG, PNG, atau WEBP.",true);return;}
  msg("Membaca gambar…");
  try{const url=URL.createObjectURL(f),img=new Image();
    await new Promise((ok,no)=>{img.onload=ok;img.onerror=no;img.src=url;});
    const s=Math.min(img.naturalWidth,img.naturalHeight);if(!s)throw 0;
    const cv=document.createElement("canvas");cv.width=96;cv.height=96;const ctx=cv.getContext("2d");
    ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality="high";
    ctx.drawImage(img,(img.naturalWidth-s)/2,(img.naturalHeight-s)/2,s,s,0,0,96,96);URL.revokeObjectURL(url);
    S.uploads.push({src:cv,name:f.name});select({kind:"u",i:S.uploads.length-1});
    msg(`"${f.name}" diperkecil ke ${G().H} × ${G().H} dan diprediksi. Label aslinya tidak diketahui, jadi tidak ada status cocok.`);
  }catch(e){msg("Gambar tidak bisa dibaca. Coba berkas lain atau simpan ulang sebagai JPG atau PNG.",true);}
  $("#file").value="";}

/* ---------- flow ---------- */
let flowState=null;
function select(it){if(S.model==="effnet")return;const W=WCACHE[wk(S.opt)];if(!W)return;
  const truth=it.kind==="g"?D.pool[it.k].t:null;
  S.cur={...it,truth};S.cur.res=forward(curPx(),W);S.sel=null;$("#detail").hidden=true;buildStrip();renderFlow();renderCompare();}
function pickCh(t,n=8){if(S.chMode==="first")return [...Array(n).keys()];
  const m=[];for(let c=0;c<t.c;c++){let s=0;for(let i=c;i<t.d.length;i+=t.c)s+=t.d[i];m.push([s,c]);}
  return m.sort((a,b)=>b[0]-a[0]).slice(0,n).map(x=>x[1]).sort((a,b)=>a-b);}
function maxOf(t,chs){let m=0;for(const c of chs)for(let i=c;i<t.d.length;i+=t.c)if(t.d[i]>m)m=t.d[i];return m;}
const dims=t=>`${t.h} × ${t.w} × ${t.c}`;
function renderFlow(){if(!S.cur)return;const r=S.cur.res,fl=$("#flow");fl.innerHTML="";
  const cnn=r.model==="cnn";$("#chSeg").hidden=!cnn;fl.classList.toggle("mlp",!cnn);
  const cols=[{key:"inp",t:r.inp,name:"input",sub:dims(r.inp),chs:[...Array(r.inp.c).keys()],gray:true}];
  if(cnn){const ch1=pickCh(r.c1),ch2=pickCh(r.c2);cols.push(
    {key:"c1",t:r.c1,name:"conv1 + ReLU",sub:dims(r.c1),chs:ch1},
    {key:"p1",t:r.p1,name:"max_pool1",sub:dims(r.p1),chs:ch1},
    {key:"c2",t:r.c2,name:"conv2 + ReLU",sub:dims(r.c2),chs:ch2},
    {key:"p2",t:r.p2,name:"max_pool2",sub:dims(r.p2),chs:ch2});}
  const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");svg.setAttribute("class","links");svg.setAttribute("aria-hidden","true");fl.append(svg);
  const colEls=[];const dark=document.documentElement.dataset.theme==="dark";
  cols.forEach(col=>{const el=document.createElement("div");el.className="col";
    el.innerHTML=`<div class="col-h"><b>${col.name}</b><small>${col.sub}</small></div>`;
    const maps=document.createElement("div");maps.className="maps";const mx=col.gray?1:maxOf(col.t,col.chs);
    col.chs.forEach(c=>{const b=document.createElement("button");b.type="button";b.className="map"+(col.gray?" in":"");
      b.dataset.col=col.key;b.dataset.ch=c;const lbl=col.gray?CHN()[c]:"#"+c;
      b.setAttribute("aria-label",`${col.name}, channel ${lbl}. Buka detail`);b.setAttribute("aria-pressed","false");
      const cv=document.createElement("canvas");paint(cv,chanOf(col.t,c),col.t.w,col.t.h,mx,col.gray?"gray":"ramp");b.append(cv);
      if(!col.gray){const s=document.createElement("span");s.className="idx";s.textContent="#"+c;b.append(s);}
      b.onclick=()=>openDetail(col,c,mx);
      b.onmouseenter=b.onfocus=()=>hot(b,true);b.onmouseleave=b.onblur=()=>hot(b,false);maps.append(b);
      if(col.gray){const t=document.createElement("div");t.className="chan";
        t.style.color=G().C===1?"var(--muted)":(dark?["#FF8A7A","#4ADE80","#6EA0FF"]:["#C0392B","#1E8449","#2563EB"])[c];t.textContent=CHN_CAP()[c];maps.append(t);}});
    el.append(maps);fl.append(el);colEls.push(el);});
  if(cnn){const de=document.createElement("div");de.className="col dense";
  de.innerHTML=`<div class="col-h"><b>dense + ReLU</b><small>128 unit</small></div>`;
  const dm=document.createElement("div");dm.className="maps";const db=document.createElement("button");db.type="button";db.className="map";db.dataset.col="dense";
  db.setAttribute("aria-label","Dense 128 unit. Buka detail");const dc=document.createElement("canvas");
  const dmax=Math.max(...r.h);paint(dc,r.h,8,16,dmax,"ramp");db.append(dc);db.onclick=()=>openDense(dmax);
  db.onmouseenter=db.onfocus=()=>hot(db,true);db.onmouseleave=db.onblur=()=>hot(db,false);
  dm.append(db);de.append(dm);fl.append(de);colEls.push(de);}
  else{const n=r.flat.length,H=G().H;
    colEls.push(vecCol(fl,"flat","flatten",`${nf(n)} nilai`,r.flat,H,n/H,"Flatten. Buka detail"));
    colEls.push(vecCol(fl,"h1","dense1 + ReLU","256 unit",r.h1,16,16,"Dense1 256 unit. Buka detail"));
    colEls.push(vecCol(fl,"h2","dense2 + ReLU","128 unit",r.h2,8,16,"Dense2 128 unit. Buka detail"));}
  const oe=document.createElement("div");oe.className="col";
  oe.innerHTML=`<div class="col-h"><b>output (softmax)</b><small>10 kelas</small></div>`;
  const om=document.createElement("div");om.className="out";
  const top=r.prob.indexOf(Math.max(...r.prob));
  r.prob.forEach((p,i)=>{const row=document.createElement("div");row.className="out-row"+(i===top?" top":"")+(i===S.cur.truth?" truth":"");
    row.innerHTML=`<span>${CLS[i]}</span><span class="bar"><i style="width:${(p*100).toFixed(1)}%"></i></span><em>${pct(p)}</em>`;om.append(row);});
  oe.append(om);fl.append(oe);colEls.push(oe);
  const v=$("#verdict"),on=MODEL()+" + "+optOf().name;
  if(S.cur.truth===null)v.innerHTML=`<span class="tab unk">Label asli tidak diketahui</span><strong>Prediksi: ${CLS[top]}</strong><span style="color:var(--muted)">keyakinan ${pct(r.prob[top])}, ${on}</span>`;
  else{const ok=top===S.cur.truth;v.innerHTML=`<span class="tab ${ok?"ok":"bad"}">${ok?"Cocok":"Tidak cocok"}</span><strong>Prediksi: ${CLS[top]}</strong><span style="color:var(--muted)">asli ${CLS[S.cur.truth]}, keyakinan ${pct(r.prob[top])}, ${on}</span>`;}
  flowState={colEls,cnn};requestAnimationFrame(drawLinks);}
/* one column holding a single vector (MLP layers, flatten), drawn as a small grid */
function vecCol(fl,key,name,sub,vals,gw,gh,label){const el=document.createElement("div");el.className="col dense";
  el.innerHTML=`<div class="col-h"><b>${name}</b><small>${sub}</small></div>`;
  const m=document.createElement("div");m.className="maps";const b=document.createElement("button");b.type="button";b.className="map";b.dataset.col=key;
  b.setAttribute("aria-label",label);const cv=document.createElement("canvas");let mx=0;for(const v of vals)if(v>mx)mx=v;
  paint(cv,vals,gw,gh,mx,key==="flat"?"gray":"ramp");cv.style.width="52px";cv.style.height=Math.round(52*gh/gw)+"px";b.append(cv);
  b.onclick=()=>openVec(key,mx);b.onmouseenter=b.onfocus=()=>hot(b,true);b.onmouseleave=b.onblur=()=>hot(b,false);
  m.append(b);el.append(m);fl.append(el);return el;}
function drawLinks(){if(!flowState)return;const fl=$("#flow"),svg=fl.querySelector("svg.links");if(!svg)return;const fr=fl.getBoundingClientRect();
  svg.innerHTML="";const pts=el=>[...el.querySelectorAll(".maps > .map, .out")].map(m=>{const b=m.getBoundingClientRect();return {el:m,l:b.left-fr.left,r:b.right-fr.left,y:b.top-fr.top+b.height/2};});
  const C=flowState.colEls.map(pts);
  for(let k=0;k<C.length-1;k++){const A=C[k],B=C[k+1],one=flowState.cnn&&(k===1||k===3);
    A.forEach((a,ai)=>B.forEach((b,bi)=>{if(one&&ai!==bi)return;
      const p=document.createElementNS("http://www.w3.org/2000/svg","path"),mx=(a.r+b.l)/2;
      p.setAttribute("d",`M${a.r} ${a.y}C${mx} ${a.y} ${mx} ${b.y} ${b.l} ${b.y}`);p._a=a.el;p._b=b.el;svg.append(p);}));}}
function hot(el,on){$("#flow").querySelectorAll("svg.links path").forEach(p=>{if(p._a===el||p._b===el)p.classList.toggle("hot",on);});}

const DESC={
  inp:c=>`Channel ${CHN()[c]} dari gambar input, nilai piksel 0 sampai 1. `+(G().C===3?"Ketiga channel inilah yang membuat C_in conv1 bernilai 3.":"Fashion-MNIST hanya punya satu channel, jadi C_in conv1 bernilai 1."),
  c1:c=>`Feature map dari kernel #${c} conv1 setelah ReLU. Kernel 3 × 3 × ${G().C} di bawah digeser ke seluruh gambar (stride 1, tanpa padding), jadi ukurannya ${G().H} − 3 + 1 = ${G().c1}. Bentuk kernel ini berbeda untuk tiap optimizer karena hasil latihannya berbeda.`,
  p1:c=>`Max pooling 2 × 2 dari feature map conv1 #${c}. Setiap piksel adalah nilai terbesar dari kotak 2 × 2, sehingga ${G().c1} × ${G().c1} menjadi ${G().p1} × ${G().p1}. Tidak ada parameter.`,
  c2:c=>`Feature map dari kernel #${c} conv2 setelah ReLU. Kernel ini membaca ke-32 channel max_pool1 sekaligus (3 × 3 × 32 bobot + 1 bias), lalu ${G().p1} − 3 + 1 = ${G().c2}.`,
  p2:c=>`Max pooling 2 × 2 dari conv2 #${c}: ${G().c2} × ${G().c2} menjadi ${G().p2} × ${G().p2} (baris dan kolom terakhir terbuang). Hasil ${G().p2} × ${G().p2} × 64 di-flatten menjadi ${nf(G().flat)} nilai untuk layer dense.`,
};
const closeSvg='<svg width="16" height="16" viewBox="0 0 16 16" stroke="currentColor" stroke-width="1.8"><path d="M3 3l10 10M13 3L3 13"/></svg>';
function openDetail(col,c,mx){S.sel={col:col.key,ch:c};const W=WCACHE[wk(S.opt)];
  document.querySelectorAll(".map[aria-pressed]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.col===col.key&&+b.dataset.ch===c?"true":"false"));
  const vals=chanOf(col.t,c);let mn=Infinity,mxv=-Infinity,s=0;for(const v of vals){mn=Math.min(mn,v);mxv=Math.max(mxv,v);s+=v;}
  const d=$("#detail");d.hidden=false;
  const title=col.gray?`input, channel ${CHN()[c]}`:`${col.name}, channel #${c} (${optOf().name})`;
  d.innerHTML=`<div class="detail-big"><canvas></canvas></div><div>
    <div class="d-head"><div><h3>${title}</h3><p>${DESC[col.key](c)}</p></div>
    <button class="icon-btn" type="button" id="dClose" aria-label="Tutup detail">${closeSvg}</button></div>
    <div class="stats"><div>Ukuran<b>${col.t.w} × ${col.t.h}</b></div><div>Minimum<b>${nf(mn,3)}</b></div><div>Maksimum<b>${nf(mxv,3)}</b></div><div>Rata-rata<b>${nf(s/vals.length,3)}</b></div></div>
    <div id="kwrap"></div></div>`;
  paint(d.querySelector(".detail-big canvas"),vals,col.t.w,col.t.h,col.gray?1:mx,col.gray?"gray":"ramp");
  const kw=d.querySelector("#kwrap");
  if(col.key==="c1"||col.key==="c2"){const L=col.key==="c1"?W.conv1:W.conv2,[,,C,O2]=L.shape;let km=0;
    for(let i=0;i<9*C;i++)km=Math.max(km,Math.abs(L.w[i*O2+c]));
    kw.innerHTML=`<p style="margin:0 0 4px;font-size:13px">Bobot kernel #${c} (${3*3*C+1} parameter termasuk bias ${nf(L.b[c],3)}). Merah negatif, biru positif.</p><div class="kernels"></div>`;
    const kk=kw.querySelector(".kernels");
    for(let ci=0;ci<C;ci++){const v=new Float32Array(9);for(let p=0;p<9;p++)v[p]=L.w[(p*C+ci)*O2+c];
      const k=document.createElement("div");k.className="kern";const cv=document.createElement("canvas");paintDiv(cv,v,3,3,km);
      const lab=document.createElement("span");lab.textContent=C===3?["R","G","B"][ci]:C===1?"gray":"in "+ci;k.append(cv,lab);kk.append(k);}}
  d.querySelector("#dClose").onclick=closeDetail;}
function openDense(mx){const r=S.cur.res,d=$("#detail");d.hidden=false;S.sel={col:"dense"};
  const act=r.h.filter(v=>v>0).length;
  d.innerHTML=`<div class="detail-big"><canvas style="width:104px;height:208px"></canvas></div><div>
   <div class="d-head"><div><h3>dense + ReLU, 128 unit (${optOf().name})</h3><p>Setiap unit menerima ke-${nf(G().flat)} nilai hasil flatten (${nf(G().flat)} × 128 + 128 = ${nf(G().flat*128+128)} parameter, layer terbesar di CNN ini). Ditampilkan sebagai grid 8 × 16. Dropout 0,3 hanya aktif saat training, jadi tidak terlihat di sini.</p></div>
   <button class="icon-btn" type="button" id="dClose" aria-label="Tutup detail">${closeSvg}</button></div>
   <div class="stats"><div>Unit aktif (&gt; 0)<b>${act} dari 128</b></div><div>Maksimum<b>${nf(mx,3)}</b></div></div></div>`;
  paint(d.querySelector("canvas"),r.h,8,16,mx,"ramp");d.querySelector("#dClose").onclick=closeDetail;}
function openVec(key,mx){const r=S.cur.res,W=WCACHE[wk(S.opt)],d=$("#detail");d.hidden=false;S.sel={col:key};
  document.querySelectorAll(".map[aria-pressed]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.col===key?"true":"false"));
  const g=G(),n=r.flat.length,vals=key==="flat"?r.flat:key==="h1"?r.h1:r.h2,[gw,gh]=key==="flat"?[g.H,n/g.H]:key==="h1"?[16,16]:[8,16];
  const act=vals.filter(v=>v>0).length,run=MODEL()+" + "+optOf().name;
  const T={flat:[`flatten, ${nf(n)} nilai`,`Gambar ${g.H} × ${g.H} × ${g.C} diratakan menjadi satu vektor berisi ${nf(n)} angka. Mulai di sini posisi piksel tidak lagi diperhatikan: piksel yang bersebelahan diperlakukan sama seperti piksel yang berjauhan. Inilah perbedaan utama MLP dengan CNN.`],
    h1:[`dense1 + ReLU, 256 unit (${run})`,`Setiap unit menerima ke-${nf(n)} nilai input (${nf(n)} × 256 + 256 = ${nf(n*256+256)} parameter, layer terbesar di MLP ini). Ditampilkan sebagai grid 16 × 16.`],
    h2:[`dense2 + ReLU, 128 unit (${run})`,`Setiap unit menerima 256 nilai dari dense1 (256 × 128 + 128 = 32.896 parameter). Ditampilkan sebagai grid 8 × 16.`]}[key];
  const big=Math.round(208*gh/gw)>320?[Math.round(320*gw/gh),320]:[208,Math.round(208*gh/gw)];
  d.innerHTML=`<div class="detail-big"><canvas style="width:${big[0]}px;height:${big[1]}px"></canvas></div><div>
   <div class="d-head"><div><h3>${T[0]}</h3><p>${T[1]}</p></div>
   <button class="icon-btn" type="button" id="dClose" aria-label="Tutup detail">${closeSvg}</button></div>
   <div class="stats">${key==="flat"?"":`<div>Unit aktif (&gt; 0)<b>${act} dari ${vals.length}</b></div>`}<div>Maksimum<b>${nf(mx,3)}</b></div></div><div id="kwrap"></div></div>`;
  paint(d.querySelector("canvas"),vals,gw,gh,mx,key==="flat"?"gray":"ramp");
  if(key==="h1"){let u=0;for(let i=1;i<vals.length;i++)if(vals[i]>vals[u])u=i;
    const kw=d.querySelector("#kwrap"),C=g.C,H=g.H;let km=0;for(let i=0;i<n;i++)km=Math.max(km,Math.abs(W.h1.w[i*256+u]));
    kw.innerHTML=`<p style="margin:0 0 6px;font-size:13px">Bobot masuk unit #${u}, unit paling aktif untuk gambar ini, disusun kembali ke bentuk gambar. Merah negatif, biru positif: ini pola yang "dicari" unit tersebut di seluruh gambar sekaligus.</p><div class="kernels"></div>`;
    const kk=kw.querySelector(".kernels");
    for(let c=0;c<C;c++){const v=new Float32Array(H*H);for(let p=0;p<H*H;p++)v[p]=W.h1.w[(p*C+c)*256+u];
      const k=document.createElement("div");k.className="kern";const cv=document.createElement("canvas");paintDiv(cv,v,H,H,km);cv.style.width="96px";cv.style.height="96px";
      const lab=document.createElement("span");lab.textContent=CHN()[c];k.append(cv,lab);kk.append(k);}}
  d.querySelector("#dClose").onclick=closeDetail;}
function closeDetail(){$("#detail").hidden=true;S.sel=null;document.querySelectorAll(".map[aria-pressed]").forEach(b=>b.setAttribute("aria-pressed","false"));}
document.addEventListener("keydown",e=>{if(e.key==="Escape"&&!$("#detail").hidden)closeDetail();});

/* ---------- compare all optimizers of the current model for the current image ---------- */
function resetCompare(){S.cmpAll=false;const b=$("#cmpBtn");b.hidden=false;b.disabled=false;b.textContent="Bandingkan semua optimizer";$("#cmpNote").textContent="";$("#crows").innerHTML="";}
async function loadAll(){const b=$("#cmpBtn");b.disabled=true;b.textContent="Memuat bobot…";$("#cmpNote").textContent="Memuat bobot kedelapan optimizer…";
  const ds=S.ds,md=S.model,res=await Promise.allSettled(OPTS().map(o=>loadW(o.slug,ds,md)));if(ds!==S.ds||md!==S.model){b.disabled=false;return;}const fail=res.filter(r=>r.status==="rejected").length;
  S.cmpAll=true;b.disabled=false;
  if(fail){b.textContent="Coba muat ulang";$("#cmpNote").textContent=`${fail} optimizer gagal dimuat. Tekan tombol untuk mencoba lagi.`;}
  else{b.hidden=true;$("#cmpNote").textContent="Klik salah satu kartu untuk memakai optimizer itu di seluruh dashboard.";}
  renderCompare();}
function renderCompare(){const box=$("#crows");if(!S.cmpAll||!S.cur||S.model==="effnet"){box.innerHTML="";return;}
  const px=curPx();box.innerHTML="";
  OPTS().forEach(o=>{const W=WCACHE[wk(o.slug)];const b=document.createElement("button");b.type="button";b.className="crow";
    b.setAttribute("aria-pressed",o.slug===S.opt?"true":"false");
    if(!W){b.disabled=true;b.innerHTML=`<div class="r1"><b>${o.name}</b></div><span class="p">Bobot belum termuat</span>`;box.append(b);return;}
    const p=forward(px,W,false).prob,top=p.indexOf(Math.max(...p));
    const tab=S.cur.truth===null?`<span class="tab unk">Tanpa label</span>`:`<span class="tab ${top===S.cur.truth?"ok":"bad"}">${top===S.cur.truth?"Cocok":"Tidak cocok"}</span>`;
    b.innerHTML=`<div class="r1"><b>${o.name}</b>${tab}</div><span class="p">Prediksi <strong>${CLS[top]}</strong>, keyakinan ${pct(p[top])}</span>
      <span class="conf"><span class="bar"><i style="width:${(p[top]*100).toFixed(1)}%"></i></span></span>`;
    b.setAttribute("aria-label",`${o.name}: prediksi ${CLS[top]}, keyakinan ${pct(p[top])}. Pakai optimizer ini`);
    b.onclick=()=>setPick(S.model,o.slug);box.append(b);});}

/* ---------- selection: dataset -> model -> optimizer ---------- */
function btn(cls,pressed,html,aria,on){const b=document.createElement("button");b.type="button";b.className=cls;
  b.setAttribute("aria-pressed",pressed?"true":"false");b.innerHTML=html;if(aria)b.setAttribute("aria-label",aria);b.onclick=on;return b;}
function renderPick(){
  document.querySelectorAll("[data-ds]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.ds===S.ds?"true":"false"));
  $("#pageTitle").textContent=L.pageTitle(D.title);
  const ml=$("#modelList");ml.innerHTML="";
  MODELS.forEach(m=>{const es=ofModel(m);if(!es.length)return;const top=Math.max(...es.map(e=>e.acc));
    ml.append(btn("optbtn",m===S.model,`<span>${MNAME[m]}</span><small>${L.best(pct(top))}</small>`,L.modelAria(MNAME[m],pct(top)),()=>setPick(m,S.opt)));});
  const ol=$("#optList");ol.innerHTML="";
  ofModel(S.model).forEach(e=>ol.append(btn("optbtn",e.opt===S.opt,`<span>${e.name}</span><small>${pct(e.acc)}</small>`,L.optAria(e.name,pct(e.acc,2)),()=>setPick(S.model,e.opt))));
  const e=curE(),n=E().length;
  $("#score").innerHTML=`<span class="who">${label(e)}</span><b class="num">${pct(e.acc,2)}</b><span class="cap">${L.accCap}</span><span class="rk">${L.rankLine(e.rank,n)}</span>`;}
function renderRank(){const box=$("#rank"),{sorted}=ENTC[S.ds],mx=sorted[0].acc;box.innerHTML="";
  sorted.forEach(e=>{const sel=e.model===S.model&&e.opt===S.opt;
    const b=btn(e.model===S.model&&!sel?"same":"",sel,`<span class="n">${e.rank}</span><span class="who"><b>${MNAME[e.model]}</b>${e.name}</span><span class="bar"><i style="width:${(e.acc/mx*100).toFixed(2)}%"></i></span><em>${pct(e.acc,2)}</em>`,
      L.rankAria(label(e),pct(e.acc,2),e.rank),()=>setPick(e.model,e.opt));box.append(b);});
  const eff=E().find(e=>e.model==="effnet");$("#rankNote").textContent=eff?L.rankNote(eff.seed):"";}
function syncFlowMode(){const eff=S.model==="effnet";$("#flowArea").hidden=eff;$("#effMsg").hidden=!eff;$("#effMsgText").textContent=L.effMsg;
  const g=G();$("#heroTitle").textContent=`Bagaimana ${MODEL()} membaca gambar ${g.H} × ${g.H}`;}
let pickSeq=0;
async function setPick(model,opt){
  const has=(m,o)=>E().some(e=>e.model===m&&e.opt===o);if(!ofModel(model).length)return;
  if(!has(model,opt))opt=has(model,S.opt)?S.opt:has(model,"adam")?"adam":ofModel(model)[0].opt;
  const seq=++pickSeq,modelChanged=model!==S.model;S.model=model;S.opt=opt;
  try{localStorage.setItem("cnn-model",model);localStorage.setItem("cnn-opt",opt);}catch(e){}
  if(S.cmpB===curE().id)S.cmpB="";
  if(modelChanged)resetCompare();
  renderPick();renderRank();syncFlowMode();syncCmpSelect();S.shown=24;renderGallery();renderTraining();
  const m=$("#optMsg");m.classList.remove("err");m.textContent="";
  if(model==="effnet")return;
  if(!WCACHE[wk(opt)]){m.textContent=L.loadingW(label(curE()));$("#flowWrap").classList.add("busy");}
  try{await loadW(opt);}catch(e){if(seq!==pickSeq)return;$("#flowWrap").classList.remove("busy");m.classList.add("err");m.textContent=L.wErr(label(curE()));return;}
  if(seq!==pickSeq)return;
  $("#flowWrap").classList.remove("busy");m.textContent="";
  if(S.cur){S.cur.res=forward(curPx(),WCACHE[wk(opt)]);closeDetail();buildStrip();renderFlow();renderCompare();}
  else select({kind:"g",k:D.demo[9]});}

/* ---------- gallery: the selected model, optionally against one other trained model ---------- */
function syncCmpSelect(){const sel=$("#selB"),cur=curE();sel.innerHTML="";
  const none=document.createElement("option");none.value="";none.textContent=L.noB;sel.append(none);
  MODELS.forEach(m=>{const es=ofModel(m).filter(e=>e.id!==cur.id);if(!es.length)return;const g=document.createElement("optgroup");g.label=MNAME[m];
    es.forEach(e=>{const o=document.createElement("option");o.value=e.id;o.textContent=`${label(e)} (${pct(e.acc,2)})`;g.append(o);});sel.append(g);});
  sel.value=S.cmpB;$("#agreeRow").hidden=!S.cmpB;if(!S.cmpB)S.agree="all";}
function buildClassChips(){const fc=$("#fClass");fc.innerHTML="";
  [-1,...CLS.keys()].forEach(i=>{const b=document.createElement("button");b.type="button";b.className="chip";b.dataset.v=i;
    b.textContent=i<0?L.allCls:CLS[i];b.onclick=()=>{S.cls=i;S.pred=null;S.shown=24;syncFilters();renderGallery();};fc.append(b);});}
function buildFilters(){buildClassChips();
  $("#fStatus").querySelectorAll("button").forEach(b=>b.onclick=()=>{S.status=b.dataset.v;S.shown=24;syncFilters();renderGallery();});
  $("#fAgree").querySelectorAll("button").forEach(b=>b.onclick=()=>{S.agree=b.dataset.v;S.shown=24;syncFilters();renderGallery();});
  $("#fSort").onchange=e=>{S.sort=e.target.value;renderGallery();};
  $("#selB").onchange=e=>{S.cmpB=e.target.value;S.shown=24;syncCmpSelect();syncFilters();renderGallery();};
  syncFilters();}
function syncFilters(){$("#fClass").querySelectorAll("button").forEach(b=>b.setAttribute("aria-pressed",+b.dataset.v===S.cls?"true":"false"));
  $("#fStatus").querySelectorAll("button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.v===S.status?"true":"false"));
  $("#fAgree").querySelectorAll("button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.v===S.agree?"true":"false"));}
const agreeOf=(a,b)=>a&&b?"both":a?"onlyA":b?"onlyB":"none";
/* ground truth for all 10,000 test images by official index (the gallery pool only covers 1,000) */
function yTrue(){const X=EXPC[S.ds];return X.y||(X.y=u8(X.labels));}
let galSeq=0;
async function renderGallery(){const sum=$("#summary"),g=$("#grid");
  const seq=++galSeq,ds=S.ds,a=curE(),b=S.cmpB?E().find(e=>e.id===S.cmpB):null;
  let A,B=null;
  if(!RUNC[ds+"/"+a.key]||(b&&!RUNC[ds+"/"+b.key]))sum.textContent=L.loading;
  try{A=await loadRun(a.key,ds);if(b)B=await loadRun(b.key,ds);}
  catch(e){if(seq!==galSeq)return;sum.innerHTML=`<span style="color:var(--bad)">${L.runErr}</span>`;g.innerHTML="";$("#more").innerHTML="";return;}
  if(seq!==galSeq||ds!==S.ds)return;
  const ids=EXPC[ds].pool_ids;
  let rows=D.pool.map((it,k)=>{const id=ids[k],p=A.p[id],q=B?B.p[id]:null;return {...it,k,id,p,c:A.c[id]/255,q};})
    .filter(it=>(S.cls<0||it.t===S.cls)&&(S.status==="all"||(S.status==="ok")===(it.t===it.p))&&(S.pred==null||it.p===S.pred)
      &&(!B||S.agree==="all"||agreeOf(it.t===it.p,it.q===it.t)===S.agree));
  rows.sort(S.sort==="hi"?(x,y)=>y.c-x.c:S.sort==="lo"?(x,y)=>x.c-y.c:(x,y)=>x.id-y.id);
  let txt=L.aLine(label(a),pct(A.acc,2));
  if(S.cls>=0){const r=A.cm[S.cls];txt+=L.classLine(CLS[S.cls],nf(r[S.cls]),nf(r.reduce((x,y)=>x+y,0)));}
  if(B){txt+=L.bLine(label(b),pct(B.acc,2));
    const n=[0,0,0,0];const y=yTrue();for(let i=0;i<10000;i++){const ok1=A.p[i]===y[i],ok2=B.p[i]===y[i];n[ok1&&ok2?0:ok1?1:ok2?2:3]++;}
    txt+=L.agreeLine(...n.map(v=>nf(v)));}
  txt+=L.galLine(nf(Math.min(S.shown,rows.length)),nf(rows.length));
  if(S.pred!=null)txt+=L.predOnly(CLS[S.pred])+` <button class="chip" type="button" id="clrPred">${L.clrPred}</button>`;
  sum.innerHTML=txt;const cp=$("#clrPred");if(cp)cp.onclick=()=>{S.pred=null;renderGallery();};
  g.innerHTML="";
  if(!rows.length){g.innerHTML=`<div class="empty" style="grid-column:1/-1">${L.empty}</div>`;$("#more").innerHTML="";return;}
  const live=a.browser;   // only MLP/CNN can be opened in the network flow
  rows.slice(0,S.shown).forEach(it=>{const ok=it.t===it.p,el=document.createElement(live?"button":"div");if(live)el.type="button";el.className="card";
    let lab=L.cardLab(ok,CLS[it.t],CLS[it.p],pct(it.c));
    el.innerHTML=`<span class="tab ${ok?"ok":"bad"}">${ok?L.ok:L.bad}</span>`;
    const cv=document.createElement("canvas");drawTile(cv,it.k);el.append(cv);
    el.insertAdjacentHTML("beforeend",`<dl><dt>${L.trueL}</dt><dd>${CLS[it.t]}</dd><dt>${L.predL}</dt><dd>${CLS[it.p]}</dd></dl>
      <div class="conf"><span class="bar"><i style="width:${(it.c*100).toFixed(1)}%"></i></span><span>${pct(it.c)}</span></div>`);
    if(B){const ok2=it.q===it.t;
      el.insertAdjacentHTML("beforeend",`<div class="vs"><span>${L.cmpL}:</span><strong style="color:var(--ink);font-weight:500">${CLS[it.q]}</strong><span class="tab ${ok2?"ok":"bad"}">${ok2?L.okShort:L.badShort}</span></div>`);
      lab+=L.cardLabB(CLS[it.q]);}
    el.insertAdjacentHTML("beforeend",`<span class="sid">${L.id(it.id)}</span>`);
    if(live){el.setAttribute("aria-label",lab+". "+L.id(it.id)+L.cardOpen);el.onclick=()=>{select({kind:"g",k:it.k});document.getElementById("alur").scrollIntoView();};}
    g.append(el);});
  $("#more").innerHTML=rows.length>S.shown?`<button class="btn" type="button" id="moreBtn">${L.more(Math.min(24,rows.length-S.shown))}</button>`:"";
  const mb=$("#moreBtn");if(mb)mb.onclick=()=>{S.shown+=24;renderGallery();};}
function showInGallery(cls,pred){S.cls=cls;S.pred=pred;S.status=pred==null?"all":"bad";S.shown=24;syncFilters();renderGallery();document.getElementById("hasil").scrollIntoView();}

/* ---------- strings for the selection, ranking, gallery and detail sections (the build swaps this block for strings_en.js) ---------- */
/*L:ID*/const L={
  pageTitle:t=>`Akurasi uji pada ${t}`, best:v=>`terbaik ${v}`, accCap:"akurasi uji, 10.000 citra",
  rankLine:(r,n)=>`Peringkat <b>${r}</b> dari ${n}`, rankAria:(lab,acc,r)=>`${lab}, akurasi uji ${acc}, peringkat ${r}. Pilih model ini`,
  rankNote:s=>`MLP dan CNN: 10 epoch dengan seluruh data latih, seed 42. EfficientNetV2B0: bobot ImageNet, fine-tuning dengan 3.000 citra latih pada input 96 × 96, seed ${s}. Semua akurasi dihitung pada 10.000 citra uji yang sama.`,
  optAria:(n,acc)=>`${n}, akurasi uji ${acc}`, modelAria:(n,acc)=>`${n}, akurasi terbaik ${acc}`,
  loadingW:n=>`Memuat bobot ${n}…`, wErr:n=>`Bobot ${n} gagal dimuat. Pilih lagi untuk mencoba ulang.`,
  loadingDs:t=>`Memuat ${t}…`, dsErr:t=>`${t} gagal dimuat. Pilih dataset itu sekali lagi untuk mencoba ulang.`,
  effMsg:"EfficientNetV2B0 terlalu besar untuk dijalankan di browser, jadi alur jaringan hanya tersedia untuk MLP dan CNN.",
  loading:"Memuat…", runErr:"Hasil model ini gagal dimuat. Pilih model itu sekali lagi untuk mencoba ulang.",
  expErr:"Daftar hasil training gagal dimuat. Muat ulang halaman untuk mencoba lagi.",
  noB:"Tidak ada",
  aLine:(lab,acc)=>`<b>${lab}</b>: akurasi uji <b>${acc}</b>.`,
  bLine:(lab,acc)=>` Pembanding <b>${lab}</b>: <b>${acc}</b>.`,
  agreeLine:(a,b,c,d)=>` Dari 10.000 citra uji: keduanya benar <b>${a}</b>, hanya terpilih benar <b>${b}</b>, hanya pembanding benar <b>${c}</b>, keduanya salah <b>${d}</b>.`,
  classLine:(cls,ok,tot)=>` Kelas <b>${cls}</b>: ${ok} benar dari ${tot}.`,
  galLine:(n,m)=>` Galeri menampilkan <b>${n}</b> dari ${m} contoh.`,
  predOnly:c=>` Hanya yang diprediksi sebagai <b>${c}</b>.`, clrPred:"Hapus filter prediksi",
  empty:"Tidak ada contoh untuk kombinasi filter ini. Ganti status, kelas, atau pembanding.",
  trueL:"Asli", predL:"Prediksi", cmpL:"Pembanding", ok:"Cocok", bad:"Tidak cocok", okShort:"Benar", badShort:"Salah",
  cardLab:(ok,t,p,c)=>`${ok?"Benar":"Salah"}. Asli ${t}, prediksi ${p}, keyakinan ${c}`, cardLabB:p=>`. Pembanding memprediksi ${p}`,
  cardOpen:". Buka di alur jaringan", more:n=>`Tampilkan ${n} contoh lagi`, id:i=>`ID uji ${i}`, allCls:"Semua kelas",
  detT:lab=>`Detail model: ${lab}`,
  warm:"warm-up", best2:e=>`dipakai: epoch ${e}`, trainL:"train", valL:"validasi", accT:"Akurasi", lossT:"Loss", epoch:"epoch",
  valAccT:"Akurasi validasi", valLossT:"Loss validasi",
  pcAria:(c,a)=>`${c} akurasi ${a}. Tampilkan di galeri`, cmAria:(v,t,p)=>`${v} citra ${t} diprediksi ${p}. Tampilkan contoh`,
  optT:m=>`Ringkasan per optimizer: ${m}`, optHead:["Optimizer","Akurasi uji","Peringkat","Val terbaik","Train − val"],
  rankOf:(r,n)=>`${r} dari ${n}`, ep:e=>`epoch ${e}`, pts:v=>`${v} poin`,
  optNoteHist:"Peringkat dihitung dari semua model. Train − val diukur di epoch terakhir.",
  optNoteEff:"Peringkat dihitung dari semua model. Train − val diukur di epoch yang dipakai untuk uji.",
  adamwSame:" AdamW sama persis dengan Adam: weight decay 1e-5 pada learning rate 6e-5 terlalu kecil untuk mengubah bobot float32.",
  curvesT:m=>`Kurva validasi per epoch: ${m}`,
  mvcNote:o=>`Akurasi uji dengan optimizer ${o}.`, mvcNoEff:o=>` EfficientNetV2B0 hanya dilatih dengan Adam, AdamW, dan Nadam.`,
  mvcHead:["Model","Parameter","Akurasi uji"],
};/*L:END*/

/* ---------- charts ---------- */
function lineChart(series,title,key,w=420,h=210,opt={}){
  const n=opt.n||Math.max(...series.map(s=>s.v.length)),all=series.flatMap(s=>s.v).filter(v=>v!=null&&isFinite(v));
  const lo=Math.min(...all),hi=Math.max(...all),pad=(hi-lo)*.08||.05,y0=lo>=0?Math.max(0,lo-pad):lo-pad,y1=lo>=0&&hi<=1?Math.min(1,hi+pad):hi+pad;
  const l=38,b=22,X=i=>l+(n>1?i*(w-l-10)/(n-1):(w-l-10)/2),Y=v=>14+(h-b-14)*(1-(v-y0)/(y1-y0));
  let s=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${title}"><text x="${l}" y="10" font-weight="600" style="fill:var(--ink);font-size:11px">${title}</text>`;
  if(opt.warm){const x2=(X(opt.warm-1)+X(opt.warm))/2;s+=`<rect x="${l}" y="14" width="${Math.max(0,x2-l)}" height="${h-b-14}" fill="var(--sunken)"/><line x1="${x2}" x2="${x2}" y1="14" y2="${h-b}" stroke="var(--muted)" stroke-width="1"/><text x="${l+3}" y="${h-b-4}" style="font-size:10px">${L.warm}</text>`;}
  for(let g=0;g<=3;g++){const v=y0+(y1-y0)*g/3;s+=`<line x1="${l}" x2="${w-10}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${l-4}" y="${Y(v)+3}" text-anchor="end">${v.toFixed(2)}</text>`;}
  const ticks=[...new Set([1,Math.round((n+1)/2),n])].filter(e=>e>=1&&e<=n);
  ticks.forEach(e=>s+=`<text x="${X(e-1)}" y="${h-6}" text-anchor="${e===1?"start":e===n?"end":"middle"}">${e===n?L.epoch+" "+e:e}</text>`);
  if(opt.best){const x=X(opt.best-1),r=x>w*.65;s+=`<line x1="${x}" x2="${x}" y1="14" y2="${h-b}" stroke="var(--ink)" stroke-dasharray="3 3" stroke-width="1"/><text x="${r?x-4:x+4}" y="26" text-anchor="${r?"end":"start"}" style="font-size:10px;fill:var(--ink);paint-order:stroke;stroke:var(--surface);stroke-width:4px">${L.best2(opt.best)}</text>`;}
  series.filter(x=>!x.sel).concat(series.filter(x=>x.sel)).forEach(x=>{const pts=x.v.map((v,i)=>v==null?null:X(i)+","+Y(v)).filter(Boolean);
    s+=`<polyline fill="none" stroke="${x.color}" stroke-width="${x.sel?2.6:1.3}" ${x.dash?'stroke-dasharray="4 3"':""} opacity="${x.sel||x.solid?1:.75}" points="${pts.join(" ")}"/>`;
    if(opt.best&&x.dot&&x.v[opt.best-1]!=null)s+=`<circle cx="${X(opt.best-1)}" cy="${Y(x.v[opt.best-1])}" r="3.5" fill="${x.color}"/>`;
    if(x.label){const li=x.v.length-1,lv=x.v[li];s+=`<text x="${Math.min(w-12,X(li)+4)}" y="${Y(lv)-5}" text-anchor="end" style="fill:${x.color};font-size:11px;font-weight:600">${x.label}</text>`;}});
  return s+"</svg>";}

/* ---------- model detail: curves, per-class accuracy, confusion matrix, optimizer summary, model comparison ---------- */
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
function bestVal(h){const v=h.val_accuracy,m=Math.max(...v);return [m,v.indexOf(m)+1];}
/* the epoch whose weights were tested: the last one for the sweep, the chosen checkpoint for EfficientNet */
const usedEpoch=e=>e.best||e.hist.accuracy.length;
let detSeq=0;
async function renderDetail(){const e=curE(),seq=++detSeq,ds=S.ds,lab=label(e);
  $("#detTitle").textContent=L.detT(lab);
  $("#optTitle").textContent=L.optT(MNAME[e.model]);$("#curvesTitle").textContent=L.curvesT(MNAME[e.model]);
  const es=ofModel(e.model);
  $("#runCurves").innerHTML=`<p class="note">${L.loading}</p>`;
  let R;try{[R]=await Promise.all([loadRun(e.key,ds),...es.map(x=>ensureHist(x,ds))]);}
  catch(err){if(seq!==detSeq)return;$("#runCurves").innerHTML=`<p class="note" style="color:var(--bad)">${L.runErr}</p>`;return;}
  if(seq!==detSeq||ds!==S.ds)return;
  const h=e.hist,acc=css("--accent"),o={best:e.best,warm:e.warm,n:h.accuracy.length};
  const pair=(a,b)=>[{v:h[a],color:acc,solid:true,sel:true,label:L.trainL},{v:h[b],color:acc,dash:true,solid:true,dot:true,label:L.valL}];
  $("#runCurves").innerHTML=lineChart(pair("accuracy","val_accuracy"),L.accT,"",420,210,o)+lineChart(pair("loss","val_loss"),L.lossT,"",420,210,o);
  renderClassPanels(R);renderOptSummary(es);renderCurves(es);renderMvc();}
function renderClassPanels(R){const pc=$("#pcls");pc.innerHTML="";
  R.cm.forEach((row,i)=>{const a=row[i]/row.reduce((x,y)=>x+y,0),b=document.createElement("button");b.type="button";
    b.setAttribute("aria-label",L.pcAria(CLS[i],pct(a)));
    b.innerHTML=`<span>${CLS[i]}</span><span class="bar"><i style="width:${a*100}%"></i></span><em>${pct(a)}</em>`;
    b.onclick=()=>showInGallery(i,null);pc.append(b);});
  const acc=css("--accent"),cmx=Math.max(...R.cm.map((x,i)=>x[i]));let ct=`<table class="cm"><thead><tr><th></th>${CLS.map(c=>`<th scope="col">${c}</th>`).join("")}</tr></thead><tbody>`;
  const a1=hex(acc),z=hex(css("--surface")),bad=hex(css("--bad"));
  R.cm.forEach((row,i)=>{ct+=`<tr><th scope="row" style="text-align:right">${CLS[i]}</th>`;
    row.forEach((v,j)=>{const u=i===j?v/cmx:Math.min(1,v/150),e=i===j?a1:bad,c=[0,1,2].map(k=>Math.round(z[k]+(e[k]-z[k])*u));
      const dk=(0.299*c[0]+0.587*c[1]+0.114*c[2])<140,fg=dk?"#fff":"inherit";
      ct+=`<td style="background:rgb(${c})">${i!==j&&v>0?`<button type="button" data-t="${i}" data-p="${j}" aria-label="${L.cmAria(v,CLS[i],CLS[j])}" style="all:unset;cursor:pointer;display:block;width:100%;height:100%;color:${fg}">${v}</button>`:`<span style="color:${fg}">${v}</span>`}</td>`;});ct+="</tr>";});
  $("#cm").innerHTML=ct+"</tbody></table>";
  $("#cm").querySelectorAll("button").forEach(b=>{b.addEventListener("focus",()=>b.style.outline="2px solid var(--accent)");b.addEventListener("blur",()=>b.style.outline="");
    b.onclick=()=>showInGallery(+b.dataset.t,+b.dataset.p);});}
const sameHist=(a,b)=>JSON.stringify([a.hist.loss,a.hist.val_accuracy])===JSON.stringify([b.hist.loss,b.hist.val_accuracy]);
function renderOptSummary(es){const n=E().length,sorted=[...es].sort((a,b)=>b.acc-a.acc);
  let t=`<thead><tr>${L.optHead.map(h=>`<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>`;
  sorted.forEach(x=>{const [bv,be]=bestVal(x.hist),u=usedEpoch(x)-1,gap=x.hist.accuracy[u]-x.hist.val_accuracy[u],sel=x.opt===S.opt;
    t+=`<tr class="click${sel?" sel":""}" data-opt="${x.opt}"><th scope="row"><button class="optrow-btn" type="button" aria-pressed="${sel}">${x.name}</button></th>
      <td>${pct(x.acc,2)}</td><td>${L.rankOf(x.rank,n)}</td><td>${pct(bv)} <span style="color:var(--muted)">(${L.ep(be)})</span></td><td>${gap>=0?"+":"−"}${L.pts(nf(Math.abs(gap)*100,1))}</td></tr>`;});
  $("#optTable").innerHTML=t+"</tbody>";
  $("#optTable").querySelectorAll("tr.click").forEach(r=>r.onclick=()=>setPick(S.model,r.dataset.opt));
  const eff=es[0].model==="effnet",a=es.find(x=>x.opt==="adam"),w=es.find(x=>x.opt==="adamw");
  $("#optNote").textContent=(eff?L.optNoteEff:L.optNoteHist)+(eff&&a&&w&&sameHist(a,w)?L.adamwSame:"");}
function renderCurves(es){const acc=css("--accent"),grey=css("--link"),warm=es[0].warm,n=Math.max(...es.map(x=>x.hist.accuracy.length));
  const ser=k=>es.map(x=>({v:x.hist[k],sel:x.opt===S.opt,color:x.opt===S.opt?acc:grey,dash:x.model==="effnet"&&x.opt==="adamw",label:x.opt===S.opt?x.name:""}));
  $("#curves").innerHTML=lineChart(ser("val_accuracy"),L.valAccT,"",420,210,{warm,n})+lineChart(ser("val_loss"),L.valLossT,"",420,210,{warm,n});
  const ok=$("#okey");ok.innerHTML="";
  es.forEach(x=>{const b=document.createElement("button");b.type="button";b.setAttribute("aria-pressed",x.opt===S.opt?"true":"false");
    b.innerHTML=`<i></i>${x.name}`;b.onclick=()=>setPick(S.model,x.opt);ok.append(b);});}
/* MLP vs CNN vs EfficientNetV2B0 on both datasets, with the selected optimizer */
let mvcSeq=0;
async function renderMvc(){const seq=++mvcSeq,o=S.opt,e=curE();
  await Promise.allSettled(DSLIST.map(d=>loadExp(d.key)));if(seq!==mvcSeq)return;
  const effHas=Object.keys(EFF_OPTS).includes(o);
  $("#mvcNote").textContent=L.mvcNote(e.name)+(effHas?"":L.mvcNoEff(e.name));
  const row=(m,p,a,ds)=>`<tr${m===S.model&&ds===S.ds?' class="sel"':""}><th scope="row">${MNAME[m]}</th><td>${nf(p)}</td><td>${pct(a,2)}</td></tr>`;
  let body="";
  for(const ds of DSLIST){const s=SUMMARY&&SUMMARY[ds.key];if(!s||!s.mlp||!s.mlp.acc)continue;
    let rows=row("mlp",s.mlp.params,s.mlp.acc[o],ds.key)+row("cnn",s.cnn.params,s.cnn.acc[o],ds.key);
    const X=EXPC[ds.key],r=X&&effHas?effRun(X,o):null;if(r)rows+=row("effnet",EFF_PARAMS,r.acc,ds.key);
    body+=`<tr class="grp${ds.key===S.ds?" sel":""}"><td colspan="3">${s.title} (${s.shape.join(" × ")})</td></tr>${rows}`;}
  $("#cmpTable").innerHTML=`<thead><tr>${L.mvcHead.map(h=>`<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${body}</tbody>`;}
function renderTraining(){if(detailSeen)renderDetail();}
/* curves and per-run files load only once the detail section is near the screen */
let detailSeen=false;
new IntersectionObserver((es,ob)=>{if(es.some(e=>e.isIntersecting)&&!detailSeen){detailSeen=true;ob.disconnect();if(D&&ENTC[S.ds])renderDetail();}},{rootMargin:"400px 0px"}).observe(document.getElementById("training"));

/* ---------- dataset control ---------- */
const DSLIST=[{key:"cifar10"},{key:"fashion"}];
function syncDS(){
  const g=G();
  $("#invWrap").hidden=g.C!==1;
  $("#stripMsg").classList.remove("err");
  $("#stripMsg").textContent=`Unggah JPG atau PNG, atau tarik gambar ke mana saja di halaman ini. Gambar dipotong persegi di tengah lalu diperkecil ke ${g.H} × ${g.H}`+(g.C===1?", diubah ke grayscale, dan (jika dicentang) dibalik warnanya.":".");
}
let dsSeq=0;
async function setDataset(ds){if(ds===S.ds)return;const seq=++dsSeq,m=$("#optMsg");m.classList.remove("err");
  const title=ds==="fashion"?"Fashion-MNIST":"CIFAR-10";
  document.querySelectorAll("[data-ds]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.ds===ds?"true":"false"));
  m.textContent=L.loadingDs(title);$("#flowWrap").classList.add("busy");
  let c;try{[c]=await Promise.all([loadDS(ds),loadExp(ds),S.model==="effnet"?null:loadW(S.opt,ds)]);}catch(e){if(seq!==dsSeq)return;$("#flowWrap").classList.remove("busy");
    m.classList.add("err");m.textContent=L.dsErr(title);document.querySelectorAll("[data-ds]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.ds===S.ds?"true":"false"));return;}
  if(seq!==dsSeq)return;
  S.ds=ds;D=c.D;atlasCtx=c.atlasCtx;CLS=D.classes;try{localStorage.setItem("cnn-dataset",ds);}catch(e){}
  if(!ENTC[ds])buildEntries(ds);
  S.cls=-1;S.pred=null;S.shown=24;S.sel=null;$("#detail").hidden=true;
  if(S.cmpB&&!E().some(e=>e.id===S.cmpB))S.cmpB="";
  $("#flowWrap").classList.remove("busy");m.textContent="";
  buildClassChips();syncFilters();syncDS();
  if(S.cmpAll){S.cmpAll=false;resetCompare();}
  const keep=S.cur&&S.cur.kind==="u"?S.cur:{kind:"g",k:D.demo[9]};
  if(S.model==="effnet"){S.cur=null;}
  await setPick(S.model,S.opt);
  if(S.model!=="effnet")select(keep);}
document.querySelectorAll("[data-ds]").forEach(b=>b.onclick=()=>setDataset(b.dataset.ds));
$("#fInv").addEventListener("change",e=>{S.inv=e.target.checked;if(S.cur&&S.cur.kind==="u")select(S.cur);else buildStrip();});

/* ---------- sidebar: open by default on desktop, a drawer on mobile ---------- */
const app=document.querySelector(".app"),MQ=matchMedia("(max-width:860px)");
const sideOpen=()=>MQ.matches?app.classList.contains("drawer-open"):!app.classList.contains("rail-closed");
function setSidebar(open,{focus=true,persist=true}={}){
  const mob=MQ.matches,side=$("#sidebar"),btn=$("#menuBtn");
  if(mob){app.classList.toggle("drawer-open",open);app.classList.remove("rail-closed");$("#scrim").hidden=!open;
    $("#main").inert=open;$("#topbar").inert=open;side.inert=!open;
    if(open)side.setAttribute("role","dialog"),side.setAttribute("aria-modal","true");else side.removeAttribute("role"),side.removeAttribute("aria-modal");}
  else{app.classList.toggle("rail-closed",!open);app.classList.remove("drawer-open");$("#scrim").hidden=true;
    $("#main").inert=false;$("#topbar").inert=false;side.inert=!open;side.removeAttribute("role");side.removeAttribute("aria-modal");
    if(persist)try{localStorage.setItem("cnn-rail",open?"0":"1");}catch(e){}}
  btn.setAttribute("aria-expanded",open?"true":"false");
  const t=open?"Tutup menu samping":"Buka menu samping";btn.setAttribute("aria-label",t);btn.title=t;
  if(focus){if(open&&mob)$("#railClose").focus();else if(!open&&(side.contains(document.activeElement)||document.activeElement===document.body))btn.focus();}}
$("#menuBtn").onclick=()=>setSidebar(!sideOpen());
$("#railClose").onclick=()=>setSidebar(false);
$("#scrim").onclick=()=>setSidebar(false);
document.addEventListener("keydown",e=>{if(e.key==="Escape"&&MQ.matches&&sideOpen()){e.preventDefault();setSidebar(false);}});
document.querySelectorAll("#railNav a").forEach(a=>a.addEventListener("click",()=>{if(MQ.matches)setSidebar(false,{focus:false});}));
function initSidebar(){let closed=false;try{closed=localStorage.getItem("cnn-rail")==="1";}catch(e){}
  setSidebar(MQ.matches?false:!closed,{focus:false,persist:false});}
MQ.addEventListener("change",initSidebar);

/* ---------- theme (header, right) + nav ---------- */
function setRamp(){$("#ramp").style.background=`linear-gradient(90deg,${css("--map0")},${css("--map1")},${css("--map2")})`;}
function setTheme(t){document.documentElement.dataset.theme=t;$("#themeBtn").setAttribute("aria-pressed",t==="dark"?"true":"false");
  try{localStorage.setItem("cnn-theme",t);}catch(e){}setRamp();
  if(D&&ENTC[S.ds]){buildStrip();renderFlow();renderTraining();renderGallery();renderCompare();}}
$("#themeBtn").onclick=()=>setTheme(document.documentElement.dataset.theme==="dark"?"light":"dark");
document.querySelectorAll(".flow-bar .seg button").forEach(b=>b.onclick=()=>{S.chMode=b.dataset.ch;
  document.querySelectorAll(".flow-bar .seg button").forEach(x=>x.setAttribute("aria-pressed",x===b?"true":"false"));closeDetail();renderFlow();});
$("#file").addEventListener("change",e=>{const f=e.target.files[0];if(f)handleFile(f);});
$("#cmpBtn").onclick=loadAll;
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)document.querySelectorAll(".nav a").forEach(a=>a.classList.toggle("on",a.getAttribute("href")==="#"+e.target.id));}),{rootMargin:"-40% 0px -55% 0px"});
/* the flow links are drawn from element positions, so redraw whenever the canvas changes width (window, sidebar) */
let rz;new ResizeObserver(()=>{cancelAnimationFrame(rz);rz=requestAnimationFrame(drawLinks);}).observe($("#flowWrap"));

/* drag and drop an image anywhere on the page: it opens in the network flow of the selected model */
let dragDepth=0;
const hasFiles=ev=>ev.dataTransfer&&[...ev.dataTransfer.types].includes("Files");
function setDropping(on){document.body.classList.toggle("dropping",on);$("#dropZone").hidden=!on;}
document.addEventListener("dragenter",ev=>{if(!hasFiles(ev))return;ev.preventDefault();dragDepth++;setDropping(true);});
document.addEventListener("dragover",ev=>{if(!hasFiles(ev))return;ev.preventDefault();ev.dataTransfer.dropEffect="copy";});
document.addEventListener("dragleave",ev=>{if(!hasFiles(ev))return;dragDepth=Math.max(0,dragDepth-1);if(!dragDepth)setDropping(false);});
document.addEventListener("drop",ev=>{if(!hasFiles(ev))return;ev.preventDefault();dragDepth=0;setDropping(false);
  const f=ev.dataTransfer.files[0];if(!f||!D)return;
  const go=()=>{handleFile(f);document.getElementById("alur").scrollIntoView();};
  if(S.model==="effnet")setPick("cnn","adam").then(go);else go();});

/* ---------- boot ---------- */
(async()=>{
  initSidebar();
  let t="light";try{t=localStorage.getItem("cnn-theme")||t;}catch(e){}
  document.documentElement.dataset.theme=t;$("#themeBtn").setAttribute("aria-pressed",t==="dark"?"true":"false");setRamp();
  try{let d="cifar10";try{d=localStorage.getItem("cnn-dataset")||d;}catch(e){}if(!DSLIST.some(x=>x.key===d))d="cifar10";
    try{const mm=localStorage.getItem("cnn-model"),oo=localStorage.getItem("cnn-opt");if(MODELS.includes(mm))S.model=mm;if(oo)S.opt=oo;}catch(e){}
    S.ds=d;const [c]=await Promise.all([loadDS(d),loadExp(d)]);D=c.D;atlasCtx=c.atlasCtx;CLS=D.classes;buildEntries(d);
    if(!curE()){S.model="cnn";S.opt="adam";}
    if(S.model!=="effnet")await loadW(S.opt,d);
    try{const r=await fetch("data/summary.json");if(r.ok)SUMMARY=await r.json();}catch(e){}
  }catch(e){const b=$("#boot");b.classList.add("err");b.textContent="Data model gagal dimuat. Muat ulang halaman; kalau masih gagal, berkas data tidak ikut terunggah.";return;}
  $("#boot").remove();["akurasi","alur","hasil","training"].forEach(id=>{$("#"+id).hidden=false;io.observe($("#"+id));});
  buildFilters();syncDS();
  await setPick(S.model,S.opt);
  window.__ready=true;
})();
