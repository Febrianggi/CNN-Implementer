"use strict";
const LANG="id";
const LOCALE=LANG==="en"?"en-US":"id-ID";
const tx=(o,k)=>o[LANG==="en"?k+"_en":k];
let D, atlasCtx, CLS, SUMMARY=null;
const DSC = {};               // dataset key -> {D, atlasCtx}
const WCACHE = {};            // "<dataset>/<model>-<slug>" -> weights
const wk = (slug,ds=S.ds,model="cnn")=>ds+"/"+model+"-"+slug;
/* Network Flow always runs the historical CNN in the browser (S.opt picks its optimizer).
   Test Results compare two runs (S.runA, S.runB); the comparison section has its own selection (S.detail). */
const S = {ds:"cifar10", inv:true, opt:"adam", cur:null, chMode:"top", sel:null, status:"all", agree:"all", cls:-1, pred:null, sort:"hi", shown:24, uploads:[], cmpAll:false,
  runA:"historical__cnn-adam__s42", runB:"", study:"full", metric:"acc", detail:"", hm:"cnn", hopt:"adam"};
const $ = s=>document.querySelector(s);
const nf = (v,d=0)=>v.toLocaleString(LOCALE,{minimumFractionDigits:d,maximumFractionDigits:d});
const pct = (v,d=1)=>nf(v*100,d)+"%";
const css = n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const hex = h=>{h=h.replace("#","");return [0,2,4].map(i=>parseInt(h.slice(i,i+2),16));};
const OPTS = ()=>D.models.cnn.opts;
const MODEL = ()=>"CNN";
const O = slug=>{const o=OPTS().find(o=>o.slug===slug);return o&&{...D.meta[slug],...o};};
const optOf = ()=>O(S.opt);

function f16(b64){const s=atob(b64),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);
  const h=new Uint16Array(u.buffer),o=new Float32Array(h.length);
  for(let i=0;i<h.length;i++){const x=h[i],sg=x&0x8000?-1:1,e=(x>>10)&31,f=x&1023;
    o[i]=e===0?sg*f*Math.pow(2,-24):e===31?(f?NaN:sg*Infinity):sg*(1+f/1024)*Math.pow(2,e-15);}
  return o;}
async function loadW(slug,ds=S.ds,model="cnn"){
  const key=wk(slug,ds,model);if(WCACHE[key])return WCACHE[key];
  const r=await fetch(`data/${ds}/w-${model}-${slug}.json`);if(!r.ok)throw new Error(r.status);const j=await r.json();
  const L=j.layers.map(l=>({kind:l.kind,w:f16(l.w),b:f16(l.b),shape:l.shape}));
  const W={model,conv1:L[0],conv2:L[1],dense:L[2],output:L[3]};
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
  const c1=conv(px,g.H,g.H,g.C,W.conv1),p1=pool(c1),c2=conv(p1.d,p1.h,p1.w,32,W.conv2),p2=pool(c2);
  const h=dense(p2.d,W.dense,true);z=dense(h,W.output,false);out={model:"cnn",inp,c1,p1,c2,p2,h};
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

/* ---------- optimizer control ---------- */
function bestVal(o){const v=o.hist.val_accuracy,m=Math.max(...v);return [m,v.indexOf(m)+1];}
function buildOptList(){const L=$("#optList");L.innerHTML="";
  OPTS().forEach(o=>{const b=document.createElement("button");b.type="button";b.className="optbtn";b.dataset.slug=o.slug;
    b.setAttribute("aria-pressed",o.slug===S.opt?"true":"false");
    b.setAttribute("aria-label",`${o.name}, akurasi uji ${pct(o.acc)}`);
    b.innerHTML=`<span>${o.name}</span><small>${pct(o.acc)}</small>`;b.onclick=()=>setOpt(o.slug);L.append(b);});}
function renderOptInfo(){const o=optOf(),[bv,be]=bestVal(o),gap=o.hist.accuracy.at(-1)-o.hist.val_accuracy.at(-1);
  const rank=[...OPTS()].sort((a,b)=>b.acc-a.acc).findIndex(x=>x.slug===o.slug)+1;
  $("#optInfo").innerHTML=`<h3>${o.name}</h3><p class="cfg">${tx(o,"cfg")}</p>
   <dl><dt>Karakteristik</dt><dd>${tx(o,"ch")}</dd><dt>Kapan digunakan</dt><dd>${tx(o,"use")}</dd></dl>
   <div class="optstats"><div>Akurasi uji<b>${pct(o.acc,2)}</b></div><div>Peringkat<b>${rank} dari ${OPTS().length}</b></div>
   <div>Val terbaik<b>${pct(bv)} <span style="font-size:13px;color:var(--muted);font-weight:400">epoch ${be}</span></b></div>
   <div>Selisih train−val<b>${gap>=0?"+":"−"}${nf(Math.abs(gap)*100,1)} poin</b></div><div>Waktu latih<b>${nf(o.time,0)} dtk</b></div></div>`;}
let optSeq=0;
async function setOpt(slug){
  const seq=++optSeq;S.opt=slug;
  document.querySelectorAll("#optList .optbtn").forEach(b=>b.setAttribute("aria-pressed",b.dataset.slug===slug?"true":"false"));
  renderOptInfo();renderFacts();renderCompare();
  const m=$("#optMsg");m.classList.remove("err");
  if(!WCACHE[wk(slug)]){m.textContent=`Memuat bobot ${optOf().name}…`;$("#flowWrap").classList.add("busy");}
  try{await loadW(slug);}catch(e){if(seq!==optSeq)return;$("#flowWrap").classList.remove("busy");
    m.classList.add("err");m.textContent=`Bobot ${optOf().name} gagal dimuat. Pilih optimizer itu sekali lagi untuk mencoba ulang.`;return;}
  if(seq!==optSeq)return;
  $("#flowWrap").classList.remove("busy");m.textContent="";
  if(S.cur){S.cur.res=forward(curPx(),WCACHE[wk(slug)]);if(S.sel)closeDetail();renderFlow();renderCompare();}
}
function renderFacts(){const o=optOf();
  $("#facts").innerHTML=`<div><dt>Parameter ${MODEL()}</dt><dd>${nf(D.models.cnn.params)}</dd></div><div><dt>Akurasi uji (${MODEL()} + ${o.name})</dt><dd>${pct(o.acc,2)}</dd></div><div><dt>Data uji</dt><dd>10.000 citra</dd></div>`;}

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
  ["dragenter","dragover"].forEach(e=>up.addEventListener(e,ev=>{ev.preventDefault();up.classList.add("drag");}));
  ["dragleave","drop"].forEach(e=>up.addEventListener(e,()=>up.classList.remove("drag")));
  up.addEventListener("drop",ev=>{ev.preventDefault();const f=ev.dataTransfer.files[0];if(f)handleFile(f);});
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
function select(it){const W=WCACHE[wk(S.opt)];if(!W)return;
  const truth=it.kind==="g"?D.pool[it.k].t:null;
  S.cur={...it,truth};S.cur.res=forward(curPx(),W);S.sel=null;$("#detail").hidden=true;buildStrip();renderFlow();renderCompare();}
function pickCh(t,n=8){if(S.chMode==="first")return [...Array(n).keys()];
  const m=[];for(let c=0;c<t.c;c++){let s=0;for(let i=c;i<t.d.length;i+=t.c)s+=t.d[i];m.push([s,c]);}
  return m.sort((a,b)=>b[0]-a[0]).slice(0,n).map(x=>x[1]).sort((a,b)=>a-b);}
function maxOf(t,chs){let m=0;for(const c of chs)for(let i=c;i<t.d.length;i+=t.c)if(t.d[i]>m)m=t.d[i];return m;}
const dims=t=>`${t.h} × ${t.w} × ${t.c}`;
function renderFlow(){if(!S.cur)return;const r=S.cur.res,fl=$("#flow");fl.innerHTML="";const cnn=true;
  const cols=[{key:"inp",t:r.inp,name:"input",sub:dims(r.inp),chs:[...Array(r.inp.c).keys()],gray:true}];
  {const ch1=pickCh(r.c1),ch2=pickCh(r.c2);cols.push(
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
  {const de=document.createElement("div");de.className="col dense";
  de.innerHTML=`<div class="col-h"><b>dense + ReLU</b><small>128 unit</small></div>`;
  const dm=document.createElement("div");dm.className="maps";const db=document.createElement("button");db.type="button";db.className="map";db.dataset.col="dense";
  db.setAttribute("aria-label","Dense 128 unit. Buka detail");const dc=document.createElement("canvas");
  const dmax=Math.max(...r.h);paint(dc,r.h,8,16,dmax,"ramp");db.append(dc);db.onclick=()=>openDense(dmax);
  db.onmouseenter=db.onfocus=()=>hot(db,true);db.onmouseleave=db.onblur=()=>hot(db,false);
  dm.append(db);de.append(dm);fl.append(de);colEls.push(de);}
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
function closeDetail(){$("#detail").hidden=true;S.sel=null;document.querySelectorAll(".map[aria-pressed]").forEach(b=>b.setAttribute("aria-pressed","false"));}
document.addEventListener("keydown",e=>{if(e.key==="Escape"&&!$("#detail").hidden)closeDetail();});

/* ---------- compare all optimizers for current image ---------- */
async function loadAll(){const b=$("#cmpBtn");b.disabled=true;b.textContent="Memuat bobot…";$("#cmpNote").textContent="Memuat bobot kedelapan optimizer…";
  const ds=S.ds,res=await Promise.allSettled(OPTS().map(o=>loadW(o.slug,ds)));if(ds!==S.ds){b.disabled=false;return;}const fail=res.filter(r=>r.status==="rejected").length;
  S.cmpAll=true;b.disabled=false;
  if(fail){b.textContent="Coba muat ulang";$("#cmpNote").textContent=`${fail} optimizer gagal dimuat. Tekan tombol untuk mencoba lagi.`;}
  else{b.hidden=true;$("#cmpNote").textContent="Klik salah satu kartu untuk memakai optimizer itu di seluruh dashboard.";}
  renderCompare();}
function renderCompare(){const box=$("#crows");if(!S.cmpAll||!S.cur){box.innerHTML="";return;}
  const px=curPx();box.innerHTML="";
  OPTS().forEach(o=>{const W=WCACHE[wk(o.slug)];const b=document.createElement("button");b.type="button";b.className="crow";
    b.setAttribute("aria-pressed",o.slug===S.opt?"true":"false");
    if(!W){b.disabled=true;b.innerHTML=`<div class="r1"><b>${o.name}</b></div><span class="p">Bobot belum termuat</span>`;box.append(b);return;}
    const p=forward(px,W,false).prob,top=p.indexOf(Math.max(...p));
    const tab=S.cur.truth===null?`<span class="tab unk">Tanpa label</span>`:`<span class="tab ${top===S.cur.truth?"ok":"bad"}">${top===S.cur.truth?"Cocok":"Tidak cocok"}</span>`;
    b.innerHTML=`<div class="r1"><b>${o.name}</b>${tab}</div><span class="p">Prediksi <strong>${CLS[top]}</strong>, keyakinan ${pct(p[top])}</span>
      <span class="conf"><span class="bar"><i style="width:${(p[top]*100).toFixed(1)}%"></i></span></span>`;
    b.setAttribute("aria-label",`${o.name}: prediksi ${CLS[top]}, keyakinan ${pct(p[top])}. Pakai optimizer ini`);
    b.onclick=()=>setOpt(o.slug);box.append(b);});}

/* ---------- strings for the run-based sections (the build swaps this block for strings_en.js) ---------- */
/*L:ID*/const L={
  mHist:"Historis", mFull:"Terkontrol, split penuh", mBudget:"Terkontrol, subset kecil",
  stShort:{historical:"historis 10 epoch",full:"split penuh",budget:"subset kecil"},
  ft:o=>`fine-tune ${o} (warm-up Adam)`, fixed:(o,e)=>`${o}, berhenti di epoch ${e}`, es:o=>`${o}, early stopping`,
  seed:s=>`seed ${s}`, notRun:"belum dilatih", noB:"Tanpa pembanding",
  where:{browser:"dijalankan di browser dari bobot 16-bit", export:"prediksi hasil ekspor (dihitung di Python, dimuat dari berkas)"},
  loadingRuns:"Memuat daftar run…", runErr:k=>`Artefak run ${k} gagal dimuat. Tidak ada data pengganti yang ditampilkan; pilih run itu sekali lagi untuk mencoba ulang.`,
  expErr:"Daftar run gagal dimuat, jadi hasil dan perbandingan model tidak bisa ditampilkan. Muat ulang halaman untuk mencoba lagi.",
  aLine:(lab,acc,f1)=>`<b>A</b>: ${lab}. Akurasi uji <b>${acc}</b>, macro-F1 <b>${f1}</b> pada 10.000 citra uji.`,
  bLine:(lab,acc,f1)=>` <b>B</b>: ${lab}. Akurasi uji <b>${acc}</b>, macro-F1 <b>${f1}</b>.`,
  agreeLine:(a,b,c,d)=>` Pada 10.000 citra uji: keduanya benar <b>${a}</b>, hanya A benar <b>${b}</b>, hanya B benar <b>${c}</b>, keduanya salah <b>${d}</b>.`,
  classLine:(cls,ok,tot)=>` Kelas <b>${cls}</b> menurut A: ${ok} benar dari ${tot}.`,
  galLine:(n,m)=>` Galeri memakai subset tetap 1.000 citra uji (100 per kelas): menampilkan <b>${n}</b> dari ${m} yang sesuai filter.`,
  predOnly:c=>` Hanya yang diprediksi A sebagai <b>${c}</b>.`, clrPred:"Hapus filter prediksi",
  same:" Pembanding sama dengan A, jadi tidak ditampilkan.",
  empty:"Tidak ada contoh untuk kombinasi filter ini. Ganti status, kelas, atau pembanding.",
  cardA:"Prediksi A", ok:"Cocok", bad:"Tidak cocok", okShort:"Benar", badShort:"Salah",
  cardLab:(ok,t,p,c)=>`A ${ok?"benar":"salah"}. Asli ${t}, prediksi A ${p}, keyakinan ${c}`, cardLabB:p=>`. B memprediksi ${p}`,
  cardOpen:". Buka di alur jaringan (CNN historis)", more:n=>`Tampilkan ${n} contoh lagi`, id:i=>`ID uji ${i}`,
  metAcc:"Akurasi uji", metF1:"Macro-F1",
  cfgHead:["Konfigurasi","Input","Data latih","Seed","Akurasi uji","Macro-F1","Δ vs CNN awal","Epoch terbaik / jalan"],
  seedsOf:(n,m)=>n===m?`${n}`:`${n} dari ${m}`, nRun:"belum ada", nr:"belum dilatih",
  cfgNote:{full:"Split stratifikasi tetap 90/10 dari data latih resmi, sama untuk semua model dan seed. Checkpoint dipilih dengan macro-F1 validasi; test set hanya dipakai untuk evaluasi akhir. CNN awal = CNN–Adam yang dihentikan di epoch 10 pada split dan seed yang sama (pembanding berpasangan).",
    budget:"Subset stratifikasi tetap: 300 citra latih dan 150 citra validasi per kelas, diambil dari split yang sama. Dipakai agar EfficientNetV2B0 bisa dilatih di CPU. Test set resmi penuh (10.000 citra). EfficientNet memakai input 96 × 96, bobot ImageNet, dan augmentasi flip + rotasi seperti repo referensi; MLP dan CNN memakai resolusi asli tanpa augmentasi, jadi perbandingan arsitektur ini tidak hanya mengubah satu faktor."},
  delta:(m,s,n)=>n>1?`${m} ± ${s}`:m, deltaNone:"tidak ada pasangan seed",
  chartLab:(st,met)=>`${met} per konfigurasi, ${st}. Titik = satu seed, garis tebal = rata-rata.`,
  curveT:lab=>`Kurva training: ${lab}`, warm:"warm-up", best:e=>`terbaik: epoch ${e}`, last:"tanpa pemilihan checkpoint (epoch terakhir dipakai)",
  trainL:"train", valL:"validasi", accT:"Akurasi", lossT:"Loss", epoch:"epoch",
  seedPick:"Seed", fitGap:"Train − validasi di epoch terpilih", fitMin:"Loss validasi minimum", fitAfter:(e,n)=>`epoch ${e}, lalu ${n} epoch sesudahnya`,
  fitStop:"Epoch dijalankan", fitSel:"Checkpoint", fitSelBest:e=>`epoch ${e} (macro-F1 validasi tertinggi)`, fitSelLast:e=>`epoch ${e} (terakhir)`, fitSelFixed:e=>`epoch ${e} (tetap)`,
  points:v=>`${v} poin`,
  identical:n=>`${n.join(" dan ")} menghasilkan kurva dan bobot yang identik di seed ini: dengan learning rate 6e-5, weight decay 1e-5 hanya mengurangi bobot sebesar 6e-10 kali nilainya per langkah, lebih kecil dari presisi float32, jadi AdamW berperilaku persis seperti Adam.`,
  branchT:s=>`Cabang fine-tuning EfficientNet, seed ${s}: akurasi validasi`, branchN:"Epoch 1 sampai 2 adalah warm-up bersama (Adam 0,001, backbone beku) dari checkpoint yang sama. Garis berhenti di epoch yang berbeda karena early stopping.",
  trainNote:"Akurasi train dihitung selama epoch dengan dropout dan augmentasi aktif; validasi dihitung di akhir epoch. Garis putus-putus vertikal = checkpoint terpilih.",
  pclsT:lab=>`Akurasi per kelas: ${lab}`, cmT:lab=>`Confusion matrix: ${lab}`,
  effElsewhere:"Hasil EfficientNet 96 px ada di tab Subset kecil", effShort:"lihat tab Subset kecil",
  pickRun:"Pilih konfigurasi di tabel untuk melihat kurvanya.", noneDone:"Belum ada run yang selesai untuk eksperimen ini. Tabel di bawah menunjukkan run yang direncanakan.", loading:"Memuat…",
  histT:m=>`Sapuan optimizer historis: ${m}, 10 epoch, seed 42`,
  aWhere:"Model A", bWhere:"Model B",
};/*L:END*/

const MNAME={mlp:"MLP",cnn:"CNN",effnet:"EfficientNetV2B0"};
const EXPC={}, RUNC={};
const fmtF1=v=>nf(v,3);
function optName(slug){const o=D.models.cnn.opts.find(o=>o.slug===slug);return o?o.name:{adam:"Adam",adamw:"AdamW",nadam:"Nadam"}[slug]||slug;}
function optPart(r){const m=/^(.*)-e(\d+)$/.exec(r.opt);
  if(r.model==="effnet")return L.ft(optName(r.opt));
  if(m)return L.fixed(optName(m[1]),+m[2]);
  return r.study==="historical"?optName(r.opt):L.es(optName(r.opt));}
const cfgLabel=r=>`${MNAME[r.model]} · ${optPart(r)}`;
const runLabel=r=>`${cfgLabel(r)} · ${L.stShort[r.study]} · ${L.seed(r.seed)}`;
const runWhere=r=>r.browser?L.where.browser:L.where.export;
async function loadExp(ds=S.ds){if(EXPC[ds])return EXPC[ds];
  const r=await fetch(`data/${ds}/experiments.json`);if(!r.ok)throw new Error(r.status);const x=await r.json();
  x.byKey={};x.runs.forEach(r=>x.byKey[r.key]=r);EXPC[ds]=x;return x;}
function u8(b64){const s=atob(b64),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}
async function loadRun(key,ds=S.ds){const id=ds+"/"+key;if(RUNC[id])return RUNC[id];
  const r=await fetch(`data/${ds}/runs/${key}.json`);if(!r.ok)throw new Error(r.status);const j=await r.json();
  if(j.key!==key)throw new Error("key mismatch");
  j.p=u8(j.preds);j.c=u8(j.conf);RUNC[id]=j;return j;}
const EX=()=>EXPC[S.ds];

/* ---------- run pickers (Test Results) ---------- */
function fillRunSelect(sel,value,withNone){const X=EX();sel.innerHTML="";
  if(withNone){const o=document.createElement("option");o.value="";o.textContent=L.noB;sel.append(o);}
  for(const [st,name] of [["historical",L.mHist],["full",L.mFull],["budget",L.mBudget]]){
    const rows=[...X.runs.filter(r=>r.study===st).map(r=>({r,done:true})),...X.planned.filter(p=>p.study===st).map(r=>({r,done:false}))];
    if(!rows.length)continue;
    rows.sort((a,b)=>cfgOrder(a.r)-cfgOrder(b.r)||(a.r.seed-b.r.seed));
    const g=document.createElement("optgroup");g.label=name;
    rows.forEach(({r,done})=>{const o=document.createElement("option");o.value=done?r.key:"";o.disabled=!done;
      o.textContent=`${cfgLabel(r)} · ${L.seed(r.seed)}`+(done?"":` (${L.notRun})`);g.append(o);});
    sel.append(g);}
  sel.value=value;}
const ORDER=["mlp-adam-e10","mlp-adam","cnn-adam-e10","cnn-adam","effnet-adam","effnet-adamw","effnet-nadam"];
const HORDER=["sgd","rmsprop","adam","adamw","adagrad","nadam","adadelta","sgd-momentum"];
function cfgOrder(r){const k=r.model+"-"+r.opt;const i=ORDER.indexOf(k);
  return i>=0?i:(r.model==="cnn"?0:20)+HORDER.indexOf(r.opt);}
function syncRunPickers(){if(!EX())return;fillRunSelect($("#selA"),S.runA,false);fillRunSelect($("#selB"),S.runB,true);
  const a=EX().byKey[S.runA],b=EX().byKey[S.runB];
  $("#whereA").textContent=a?runWhere(a):"";$("#whereB").textContent=b?runWhere(b):"";
  $("#agreeRow").hidden=!b||S.runB===S.runA;}

/* ---------- gallery ---------- */
function buildClassChips(){const fc=$("#fClass");fc.innerHTML="";
  [-1,...CLS.keys()].forEach(i=>{const b=document.createElement("button");b.type="button";b.className="chip";b.dataset.v=i;
    b.textContent=i<0?"Semua kelas":CLS[i];b.onclick=()=>{S.cls=i;S.pred=null;S.shown=24;syncFilters();renderGallery();};fc.append(b);});}
function buildFilters(){buildClassChips();
  $("#fStatus").querySelectorAll("button").forEach(b=>b.onclick=()=>{S.status=b.dataset.v;S.shown=24;syncFilters();renderGallery();});
  $("#fAgree").querySelectorAll("button").forEach(b=>b.onclick=()=>{S.agree=b.dataset.v;S.shown=24;syncFilters();renderGallery();});
  $("#fSort").onchange=e=>{S.sort=e.target.value;renderGallery();};
  $("#selA").onchange=e=>{S.runA=e.target.value;S.shown=24;syncRunPickers();renderGallery();};
  $("#selB").onchange=e=>{S.runB=e.target.value;if(!S.runB)S.agree="all";S.shown=24;syncRunPickers();syncFilters();renderGallery();};
  syncFilters();}
function syncFilters(){$("#fClass").querySelectorAll("button").forEach(b=>b.setAttribute("aria-pressed",+b.dataset.v===S.cls?"true":"false"));
  $("#fStatus").querySelectorAll("button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.v===S.status?"true":"false"));
  $("#fAgree").querySelectorAll("button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.v===S.agree?"true":"false"));}
let galSeq=0;
async function renderGallery(){const X=EX(),sum=$("#summary"),g=$("#grid");if(!X){return;}
  const seq=++galSeq,ds=S.ds,a=X.byKey[S.runA],b=S.runB&&S.runB!==S.runA?X.byKey[S.runB]:null;
  if(!a){sum.innerHTML=L.runErr(S.runA);g.innerHTML="";$("#more").innerHTML="";return;}
  let A,B=null;
  if(!RUNC[ds+"/"+a.key]||(b&&!RUNC[ds+"/"+b.key]))sum.textContent=L.loading;
  try{A=await loadRun(a.key,ds);if(b)B=await loadRun(b.key,ds);}
  catch(e){if(seq!==galSeq)return;const bad=!RUNC[ds+"/"+a.key]?a.key:b.key;sum.innerHTML=`<span style="color:var(--bad)">${L.runErr(bad)}</span>`;g.innerHTML="";$("#more").innerHTML="";return;}
  if(seq!==galSeq||ds!==S.ds)return;
  const ids=X.pool_ids;
  let rows=D.pool.map((it,k)=>{const id=ids[k],p=A.p[id],q=B?B.p[id]:null;return {...it,k,id,p,c:A.c[id]/255,q};})
    .filter(it=>(S.cls<0||it.t===S.cls)&&(S.status==="all"||(S.status==="ok")===(it.t===it.p))&&(S.pred==null||it.p===S.pred)
      &&(!B||S.agree==="all"||agreeOf(it.t===it.p,it.q===it.t)===S.agree));
  rows.sort(S.sort==="hi"?(x,y)=>y.c-x.c:S.sort==="lo"?(x,y)=>x.c-y.c:(x,y)=>x.id-y.id);
  let txt=L.aLine(`<b>${runLabel(a)}</b>`,pct(A.acc,2),fmtF1(A.f1));
  if(S.cls>=0){const r=A.cm[S.cls];txt+=L.classLine(CLS[S.cls],nf(r[S.cls]),nf(r.reduce((x,y)=>x+y,0)));}
  if(B){txt+=L.bLine(`<b>${runLabel(b)}</b>`,pct(B.acc,2),fmtF1(B.f1));
    const n=[0,0,0,0];const y=yTrue(A);for(let i=0;i<10000;i++){const ok1=A.p[i]===y[i],ok2=B.p[i]===y[i];n[ok1&&ok2?0:ok1?1:ok2?2:3]++;}
    txt+=L.agreeLine(...n.map(v=>nf(v)));}
  else if(S.runB&&S.runB===S.runA)txt+=L.same;
  txt+=L.galLine(Math.min(S.shown,rows.length),rows.length);
  if(S.pred!=null)txt+=L.predOnly(CLS[S.pred])+` <button class="chip" type="button" id="clrPred">${L.clrPred}</button>`;
  sum.innerHTML=txt;const cp=$("#clrPred");if(cp)cp.onclick=()=>{S.pred=null;renderGallery();};
  g.innerHTML="";
  if(!rows.length){g.innerHTML=`<div class="empty" style="grid-column:1/-1">${L.empty}</div>`;$("#more").innerHTML="";return;}
  rows.slice(0,S.shown).forEach(it=>{const ok=it.t===it.p,el=document.createElement("button");el.type="button";el.className="card";
    let lab=L.cardLab(ok,CLS[it.t],CLS[it.p],pct(it.c));
    el.innerHTML=`<span class="tab ${ok?"ok":"bad"}">${ok?L.ok:L.bad}</span>`;
    const cv=document.createElement("canvas");drawTile(cv,it.k);el.append(cv);
    el.insertAdjacentHTML("beforeend",`<dl><dt>Asli</dt><dd>${CLS[it.t]}</dd><dt>${L.cardA}</dt><dd>${CLS[it.p]}</dd></dl>
      <div class="conf"><span class="bar"><i style="width:${(it.c*100).toFixed(1)}%"></i></span><span>${pct(it.c)}</span></div>`);
    if(B){const ok2=it.q===it.t;
      el.insertAdjacentHTML("beforeend",`<div class="vs"><span>B:</span><strong style="color:var(--ink);font-weight:500">${CLS[it.q]}</strong><span class="tab ${ok2?"ok":"bad"}">${ok2?L.okShort:L.badShort}</span></div>`);
      lab+=L.cardLabB(CLS[it.q]);}
    el.insertAdjacentHTML("beforeend",`<span class="sid">${L.id(it.id)}</span>`);
    el.setAttribute("aria-label",lab+". "+L.id(it.id)+L.cardOpen);
    el.onclick=()=>{select({kind:"g",k:it.k});document.getElementById("alur").scrollIntoView();};g.append(el);});
  $("#more").innerHTML=rows.length>S.shown?`<button class="btn" type="button" id="moreBtn">${L.more(Math.min(24,rows.length-S.shown))}</button>`:"";
  const mb=$("#moreBtn");if(mb)mb.onclick=()=>{S.shown+=24;renderGallery();};}
const agreeOf=(a,b)=>a&&b?"both":a?"onlyA":b?"onlyB":"none";
/* ground truth for all 10,000 test images by official index (the gallery pool only covers 1,000) */
function yTrue(){return EX().y||(EX().y=u8(EX().labels));}

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
  if(opt.best){const x=X(opt.best-1),r=x>w*.65;s+=`<line x1="${x}" x2="${x}" y1="14" y2="${h-b}" stroke="var(--ink)" stroke-dasharray="3 3" stroke-width="1"/><text x="${r?x-4:x+4}" y="26" text-anchor="${r?"end":"start"}" style="font-size:10px;fill:var(--ink)">${L.best(opt.best)}</text>`;}
  series.filter(x=>!x.sel).concat(series.filter(x=>x.sel)).forEach(x=>{const pts=x.v.map((v,i)=>v==null?null:X(i)+","+Y(v)).filter(Boolean);
    s+=`<polyline fill="none" stroke="${x.color}" stroke-width="${x.sel?2.6:1.3}" ${x.dash?'stroke-dasharray="4 3"':""} opacity="${x.sel||x.solid?1:.75}" points="${pts.join(" ")}"/>`;
    if(opt.best&&x.dot&&x.v[opt.best-1]!=null)s+=`<circle cx="${X(opt.best-1)}" cy="${Y(x.v[opt.best-1])}" r="3.5" fill="${x.color}"/>`;
    if(x.label){const li=x.v.length-1,lv=x.v[li];s+=`<text x="${Math.min(w-12,X(li)+4)}" y="${Y(lv)-5}" text-anchor="end" style="fill:${x.color};font-size:11px;font-weight:600">${x.label}</text>`;}});
  return s+"</svg>";}

/* ---------- Model & Optimizer Comparison ---------- */
function configs(study){const X=EX(),m={};
  X.runs.filter(r=>r.study===study).forEach(r=>{const k=r.model+"-"+r.opt;(m[k]=m[k]||{k,model:r.model,opt:r.opt,study,runs:[],planned:[]}).runs.push(r);});
  X.planned.filter(p=>p.study===study).forEach(p=>{const k=p.model+"-"+p.opt;(m[k]=m[k]||{k,model:p.model,opt:p.opt,study,runs:[],planned:[]}).planned.push(p);});
  return Object.values(m).sort((a,b)=>cfgOrder(a)-cfgOrder(b));}
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
const sd=a=>a.length>1?Math.sqrt(a.reduce((s,x)=>s+(x-mean(a))**2,0)/(a.length-1)):0;
function renderCfg(){const X=EX();if(!X)return;const met=S.metric==="acc"?"acc":"f1",C=configs(S.study);
  document.querySelectorAll("#studySeg button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.v===S.study?"true":"false"));
  document.querySelectorAll("#metricSeg button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.v===S.metric?"true":"false"));
  $("#cfgNote").textContent=L.cfgNote[S.study];
  const base=C.find(c=>c.k==="cnn-adam-e10"),bs={};if(base)base.runs.forEach(r=>bs[r.seed]=r[met]);
  const fmt=v=>met==="acc"?pct(v,2):fmtF1(v);
  // chart: one row per configuration, a dot per seed and a bar up to the mean (HTML, so it reflows on small screens)
  const done=C.filter(c=>c.runs.length),vals=done.flatMap(c=>c.runs.map(r=>r[met]));
  const box=$("#cfgChart");
  if(!vals.length)box.innerHTML=`<p class="note">${L.noneDone}</p>`;
  else{let lo=Math.min(...vals),hi=Math.max(...vals);const pd=(hi-lo)*.12||.02;lo=Math.max(0,lo-pd);hi=Math.min(1,hi+pd);
    const P=v=>(100*(v-lo)/(hi-lo)).toFixed(2);
    let h=`<div class="dots" role="img" aria-label="${L.chartLab(L.stShort[S.study],S.metric==="acc"?L.metAcc:L.metF1)}">`;
    C.forEach(c=>{const sel=c.runs.some(r=>r.key===S.detail),v=c.runs.map(r=>r[met]);
      h+=`<div class="drow${sel?" sel":""}" aria-hidden="true"><span class="dl">${MNAME[c.model]} · ${optPart(c)}</span><span class="dt">`
        +(v.length?`<i style="width:${P(mean(v))}%"></i>`+v.map(x=>`<b style="left:${P(x)}%"></b>`).join(""):`<span class="nr">${L.nr}${c.model==="effnet"&&S.study==="full"&&EX().runs.some(r=>r.study==="budget"&&r.model==="effnet")?" · "+L.effShort:""}</span>`)
        +`</span><span class="dn">${v.length?fmt(mean(v))+` <small>n = ${v.length}</small>`:""}</span></div>`;});
    h+=`<div class="drow axis" aria-hidden="true"><span class="dl"></span><span class="dt">${[0,1,2,3,4].map(g=>{const v=lo+(hi-lo)*g/4;return `<em style="left:${g*25}%">${met==="acc"?nf(v*100,0)+"%":v.toFixed(2)}</em>`;}).join("")}</span><span class="dn"></span></div>`;
    box.innerHTML=h+"</div>";}
  // table
  let t=`<thead><tr>${L.cfgHead.map((h,i)=>`<th scope="col"${i?"":' style="text-align:left"'}>${h}</th>`).join("")}</tr></thead><tbody>`;
  C.forEach(c=>{const R=c.runs,total=R.length+c.planned.length,sel=R.some(r=>r.key===S.detail);
    const first=R[0]||c.planned[0];
    if(!R.length){t+=`<tr><th scope="row">${MNAME[c.model]} · ${optPart(c)}</th><td colspan="7" style="text-align:left;color:var(--muted)">${L.nr} (0 / ${total} seed)${c.model==="effnet"&&S.study==="full"&&EX().runs.some(r=>r.study==="budget"&&r.model==="effnet")?". "+L.effElsewhere:""}</td></tr>`;return;}
    const a=R.map(r=>r.acc),f=R.map(r=>r.f1),mv=R.map(r=>r[met]);
    const pairs=R.filter(r=>bs[r.seed]!=null).map(r=>(r[met]-bs[r.seed])*100),md=pairs.length?mean(pairs):0;
    const dtxt=c.k==="cnn-adam-e10"?"0":pairs.length?L.delta((md>=0?"+":"−")+nf(Math.abs(md),2),nf(sd(pairs),2),pairs.length):L.deltaNone;
    const ep=R.map(r=>`${r.best_epoch??r.epochs_run}/${r.epochs_run}`).join(", ");
    t+=`<tr class="click${sel?" sel":""}" data-k="${c.k}"><th scope="row"><button class="optrow-btn" type="button" data-k="${c.k}" aria-pressed="${sel}">${MNAME[c.model]} · ${optPart(c)}</button></th>
      <td>${first.input??"?"} px</td><td>${nf(first.n_train??0)}</td><td>${L.seedsOf(R.length,total)}</td>
      <td>${pct(mean(a),2)}${R.length>1?` <span style="color:var(--muted)">± ${nf(sd(a)*100,2)}</span>`:""}</td>
      <td>${fmtF1(mean(f))}${R.length>1?` <span style="color:var(--muted)">± ${nf(sd(f),3)}</span>`:""}</td>
      <td>${dtxt}</td><td>${ep}</td></tr>`;});
  $("#cfgTable").innerHTML=t+"</tbody>";
  $("#cfgTable").querySelectorAll("tr.click").forEach(r=>r.onclick=()=>{const c=C.find(c=>c.k===r.dataset.k),cur=EX().byKey[S.detail];
    const pick=c.runs.find(x=>cur&&x.seed===cur.seed)||c.runs[0];setDetail(pick.key);});
  $("#seedTable").innerHTML=seedTable(C,met,fmt);}
function seedTable(C,met,fmt){const seeds=[...new Set(C.flatMap(c=>[...c.runs,...c.planned].map(r=>r.seed)))].sort((a,b)=>a-b);
  let t=`<thead><tr><th scope="col" style="text-align:left">${L.cfgHead[0]}</th>${seeds.map(s=>`<th scope="col">${L.seed(s)}</th>`).join("")}<th scope="col">${S.metric==="acc"?L.metAcc:L.metF1}</th></tr></thead><tbody>`;
  C.forEach(c=>{const by={};c.runs.forEach(r=>by[r.seed]=r);
    t+=`<tr><th scope="row">${MNAME[c.model]} · ${optPart(c)}</th>${seeds.map(s=>by[s]?`<td>${fmt(by[s][met])}</td>`:`<td style="color:var(--muted)">${c.planned.some(p=>p.seed===s)?L.nr:""}</td>`).join("")}<td><b>${c.runs.length?fmt(mean(c.runs.map(r=>r[met]))):""}</b></td></tr>`;});
  return t+"</tbody>";}

let detSeq=0;
async function setDetail(key){S.detail=key;renderCfg();renderHist();await renderDetail();}
async function renderDetail(){const X=EX();if(!X)return;const r=X.byKey[S.detail];const box=$("#runCurves");
  if(!r){$("#runTitle").textContent=L.pickRun;box.innerHTML="";$("#seedSeg").innerHTML="";$("#fitStats").innerHTML="";$("#branchWrap").hidden=true;$("#pcls").innerHTML="";$("#cm").innerHTML="";return;}
  const seq=++detSeq,ds=S.ds;$("#runTitle").textContent=L.curveT(runLabel(r));
  // seed buttons for this configuration
  const same=[...X.runs,...X.planned].filter(x=>x.study===r.study&&x.model===r.model&&x.opt===r.opt).sort((a,b)=>a.seed-b.seed);
  const ss=$("#seedSeg");ss.innerHTML="";
  same.forEach(x=>{const b=document.createElement("button");b.type="button";b.textContent=L.seed(x.seed)+(x.key?"":` (${L.nr})`);
    b.disabled=!x.key;b.setAttribute("aria-pressed",x.key===r.key?"true":"false");if(x.key)b.onclick=()=>setDetail(x.key);ss.append(b);});
  box.innerHTML=`<p class="note">${L.loading}</p>`;
  let R;try{R=await loadRun(r.key,ds);}catch(e){if(seq!==detSeq)return;box.innerHTML=`<p class="note" style="color:var(--bad)">${L.runErr(r.key)}</p>`;return;}
  if(seq!==detSeq||ds!==S.ds)return;
  const h=R.history,acc=css("--accent"),warm=r.model==="effnet"?(r.warmup_epochs||2):0;
  const o={best:r.best_epoch||null,warm,n:h.length};
  box.innerHTML=lineChart([{v:h.map(e=>e.accuracy),color:acc,solid:true,sel:true,label:L.trainL},{v:h.map(e=>e.val_accuracy),color:acc,dash:true,solid:true,dot:true,label:L.valL}],L.accT,"",420,210,o)
    +lineChart([{v:h.map(e=>e.loss),color:acc,solid:true,sel:true,label:L.trainL},{v:h.map(e=>e.val_loss),color:acc,dash:true,solid:true,dot:true,label:L.valL}],L.lossT,"",420,210,o);
  // fit indicators read off the curves
  const sel=(r.best_epoch||h.length)-1,gap=h[sel].accuracy-h[sel].val_accuracy;
  const vl=h.map(e=>e.val_loss),mi=vl.indexOf(Math.min(...vl));
  const selTxt=r.study==="historical"?L.fitSelLast(h.length):/-e\d+$/.test(r.opt)?L.fitSelFixed(r.best_epoch):L.fitSelBest(r.best_epoch);
  $("#fitStats").innerHTML=`<div>${L.fitSel}<b>${selTxt}</b></div><div>${L.fitGap}<b>${gap>=0?"+":"−"}${L.points(nf(Math.abs(gap)*100,1))}</b></div>
    <div>${L.fitMin}<b>${L.fitAfter(mi+1,h.length-mi-1)}</b></div><div>${L.fitStop}<b>${h.length}</b></div>`;
  // the three EfficientNet branches of this seed
  const bw=$("#branchWrap");
  if(r.model==="effnet"){bw.hidden=false;$("#branchTitle").textContent=L.branchT(r.seed);$("#branchNote").textContent=L.branchN;
    const sib=["adam","adamw","nadam"].map(o=>X.runs.find(x=>x.study===r.study&&x.model==="effnet"&&x.opt===o&&x.seed===r.seed)).filter(Boolean);
    let S2;try{S2=await Promise.all(sib.map(x=>loadRun(x.key,ds)));}catch(e){S2=null;}
    if(seq!==detSeq)return;
    if(!S2){$("#branchCurves").innerHTML=`<p class="note" style="color:var(--bad)">${L.runErr("EfficientNet")}</p>`;}
    else{const grey=css("--muted"),n=Math.max(...S2.map(x=>x.history.length));
      // branches whose curves are identical share one label (AdamW with a tiny weight decay can equal Adam exactly)
      const sig=x=>JSON.stringify(x.history.map(e=>[e.loss,e.val_accuracy])),groups={};S2.forEach((x,i)=>(groups[sig(x)]=groups[sig(x)]||[]).push(i));
      const labelOf=i=>{const g=groups[sig(S2[i])];return g[0]===i?g.map(j=>optName(sib[j].opt)).join(" = "):"";};
      $("#branchCurves").innerHTML=lineChart(S2.map((x,i)=>({v:x.history.map(e=>e.val_accuracy),sel:sib[i].key===r.key,color:sib[i].key===r.key?acc:grey,dash:sib[i].opt==="adamw",solid:true,label:labelOf(i)})),L.accT,"",1000,260,{warm,n});
      const same=Object.values(groups).filter(g=>g.length>1);
      $("#branchNote").textContent=L.branchN+(same.length?" "+same.map(g=>L.identical(g.map(j=>optName(sib[j].opt)))).join(" "):"");
      const k=$("#branchKey");k.innerHTML="";sib.forEach(x=>{const b=document.createElement("button");b.type="button";b.setAttribute("aria-pressed",x.key===r.key?"true":"false");
        b.innerHTML=`<i></i>${optName(x.opt)} · ${pct(x.acc,2)}`;b.onclick=()=>setDetail(x.key);k.append(b);});}}
  else bw.hidden=true;
  renderClassPanels(r,R);}
function renderClassPanels(r,R){const lab=cfgLabel(r)+" · "+L.seed(r.seed);
  $("#pclsTitle").textContent=L.pclsT(lab);const pc=$("#pcls");pc.innerHTML="";
  R.cm.forEach((row,i)=>{const a=row[i]/row.reduce((x,y)=>x+y,0),b=document.createElement("button");b.type="button";
    b.setAttribute("aria-label",`${CLS[i]} akurasi ${pct(a)}. Tampilkan di galeri`);
    b.innerHTML=`<span>${CLS[i]}</span><span class="bar"><i style="width:${a*100}%"></i></span><em>${pct(a)}</em>`;
    b.onclick=()=>{S.runA=r.key;S.cls=i;S.status="all";S.pred=null;S.shown=24;syncRunPickers();syncFilters();renderGallery();document.getElementById("hasil").scrollIntoView();};pc.append(b);});
  $("#cmTitle").textContent=L.cmT(lab);
  const acc=css("--accent"),cmx=Math.max(...R.cm.map((x,i)=>x[i]));let ct=`<table class="cm"><thead><tr><th></th>${CLS.map(c=>`<th scope="col">${c}</th>`).join("")}</tr></thead><tbody>`;
  const a1=hex(acc),z=hex(css("--surface")),bad=hex(css("--bad"));
  R.cm.forEach((row,i)=>{ct+=`<tr><th scope="row" style="text-align:right">${CLS[i]}</th>`;
    row.forEach((v,j)=>{const u=i===j?v/cmx:Math.min(1,v/150),e=i===j?a1:bad,c=[0,1,2].map(k=>Math.round(z[k]+(e[k]-z[k])*u));
      const dk=(0.299*c[0]+0.587*c[1]+0.114*c[2])<140,fg=dk?"#fff":"inherit";
      ct+=`<td style="background:rgb(${c})">${i!==j&&v>0?`<button type="button" data-t="${i}" data-p="${j}" aria-label="${v} citra ${CLS[i]} diprediksi ${CLS[j]}. Tampilkan contoh" style="all:unset;cursor:pointer;display:block;width:100%;height:100%;color:${fg}">${v}</button>`:`<span style="color:${fg}">${v}</span>`}</td>`;});ct+="</tr>";});
  $("#cm").innerHTML=ct+"</tbody></table>";
  $("#cm").querySelectorAll("button").forEach(b=>{b.addEventListener("focus",()=>b.style.outline="2px solid var(--accent)");b.addEventListener("blur",()=>b.style.outline="");
    b.onclick=()=>{S.runA=r.key;S.cls=+b.dataset.t;S.pred=+b.dataset.p;S.status="bad";S.shown=24;syncRunPickers();syncFilters();renderGallery();document.getElementById("hasil").scrollIntoView();};});}

/* ---------- historical optimizer sweep (10 epochs, seed 42) ---------- */
const HOPTS=()=>D.models[S.hm].opts;
const hOpt=()=>HOPTS().find(o=>o.slug===S.hopt);
function pickHist(slug){S.hopt=slug;setDetail(`historical__${S.hm}-${slug}__s42`);}
function renderHist(){const o=hOpt(),sorted=[...HOPTS()].sort((a,b)=>b.acc-a.acc),HM=S.hm.toUpperCase();
  document.querySelectorAll("#hmSeg button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.v===S.hm?"true":"false"));
  $("#histTitle").textContent=L.histT(HM);
  const onDetail=x=>S.detail===`historical__${S.hm}-${x.slug}__s42`;
  let t=`<thead><tr><th scope="col">Optimizer</th><th scope="col">Akurasi uji</th><th scope="col">Loss uji</th><th scope="col">Val terbaik</th><th scope="col">Train−val</th><th scope="col">Waktu</th></tr></thead><tbody>`;
  sorted.forEach(x=>{const [bv,be]=bestVal(x),gap=x.hist.accuracy.at(-1)-x.hist.val_accuracy.at(-1);
    t+=`<tr class="click${x.slug===S.hopt?" sel":""}" data-slug="${x.slug}"><th scope="row"><button class="optrow-btn" type="button" data-slug="${x.slug}" aria-pressed="${x.slug===S.hopt}">${x.name}</button></th>
      <td>${pct(x.acc,2)}</td><td>${nf(x.loss,3)}</td><td>${pct(bv)} <span style="color:var(--muted)">(ep ${be})</span></td><td>${gap>=0?"+":"−"}${nf(Math.abs(gap)*100,1)}</td><td>${nf(x.time,0)} dtk</td></tr>`;});
  $("#optTable").innerHTML=t+"</tbody>";
  $("#optTable").querySelectorAll("tr.click").forEach(r=>r.onclick=()=>pickHist(r.dataset.slug));
  const other=S.hm==="cnn"?"mlp":"cnn",oth=s=>D.models[other].opts.find(x=>x.slug===s).acc;
  $("#accNote").textContent=`Garis putus-putus: ${other.toUpperCase()} dengan optimizer yang sama.`;
  $("#accTitle").textContent=`Akurasi uji ${HM}`;
  const bx=$("#accBars");bx.innerHTML="";const mx=Math.max(...HOPTS().map(x=>x.acc),...D.models[other].opts.map(x=>x.acc));
  sorted.forEach(x=>{const b=document.createElement("button");b.type="button";b.setAttribute("aria-pressed",x.slug===S.hopt?"true":"false");
    b.setAttribute("aria-label",`${x.name} akurasi uji ${pct(x.acc)}. Pilih optimizer ini`);
    b.innerHTML=`<span>${x.name}</span><span class="bar" style="position:relative"><i style="width:${x.acc/mx*100}%"></i><span style="position:absolute;top:-3px;bottom:-3px;left:${Math.min(100,oth(x.slug)/mx*100)}%;border-left:2px dashed var(--muted)"></span></span><em>${pct(x.acc)}</em>`;
    b.onclick=()=>pickHist(x.slug);bx.append(b);});
  const acc=css("--accent"),grey=css("--link");
  const ser=k=>HOPTS().map(x=>({v:x.hist[k],sel:x.slug===S.hopt,color:x.slug===S.hopt?acc:grey,label:x.slug===S.hopt?x.name:""}));
  $("#curves").innerHTML=lineChart(ser("val_accuracy"),"Akurasi validasi")+lineChart(ser("val_loss"),"Loss validasi");
  const ok=$("#okey");ok.innerHTML="";
  HOPTS().forEach(x=>{const b=document.createElement("button");b.type="button";b.setAttribute("aria-pressed",x.slug===S.hopt?"true":"false");
    b.innerHTML=`<i></i>${x.name}`;b.onclick=()=>pickHist(x.slug);ok.append(b);});
  // MLP vs CNN
  const sl=S.hopt;$("#mvcNote").textContent=`Keduanya dilatih dengan ${o.name}, optimizer yang sedang dipilih. Dataset yang sedang dipilih ditandai biru.`;
  const row=(n,p,a)=>`<tr${n===HM?' class="sel"':""}><th scope="row">${n}</th><td>${nf(p)}</td><td>${pct(a,2)}</td></tr>`;
  const gap=(a,b)=>`<tr><th scope="row">Selisih CNN − MLP</th><td></td><td class="${b>=a?"pos":""}">${b>=a?"+":"−"}${nf(Math.abs(b-a)*100,2)} poin</td></tr>`;
  let body="";
  for(const ds of DSLIST){const s=SUMMARY&&SUMMARY[ds.key];if(!s||!s.mlp||!s.mlp.acc)continue;
    body+=`<tr class="grp${ds.key===S.ds?" sel":""}"><td colspan="3">${s.title} (${s.shape.join(" × ")})</td></tr>${row("MLP",s.mlp.params,s.mlp.acc[sl])}${row("CNN",s.cnn.params,s.cnn.acc[sl])}${gap(s.mlp.acc[sl],s.cnn.acc[sl])}`;}
  $("#cmpTable").innerHTML=`<thead><tr><th scope="col">Model</th><th scope="col">Parameter</th><th scope="col">Akurasi uji</th></tr></thead><tbody>${body}</tbody>`;}
function renderTraining(){if(!EX())return;
  if(!EX().byKey[S.detail]){const C=configs(S.study),c=C.find(c=>c.runs.length&&c.model==="effnet")||C.find(c=>c.runs.length&&c.k==="cnn-adam")||C.find(c=>c.runs.length);S.detail=c?c.runs[0].key:"";}
  renderCfg();renderHist();if(detailSeen)renderDetail();}
/* curves and per-run files load only once the comparison section is near the screen */
let detailSeen=false;
new IntersectionObserver((es,ob)=>{if(es.some(e=>e.isIntersecting)&&!detailSeen){detailSeen=true;ob.disconnect();if(EX())renderDetail();}},{rootMargin:"400px 0px"}).observe(document.getElementById("training"));

/* ---------- dataset control ---------- */
const DSLIST=[{key:"cifar10"},{key:"fashion"}];
function syncDS(){
  document.querySelectorAll("[data-ds]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.ds===S.ds?"true":"false"));
  const g=G();$("#heroTitle").textContent=`Bagaimana ${MODEL()} membaca gambar ${g.H} × ${g.H}`;
  $("#heroLead").textContent=`${D.title}: ${nf(D.pool.length)} contoh uji di galeri, ${CLS.length} kelas, input ${g.H} × ${g.H} × ${g.C}. Alur ini menjalankan CNN historis di browser, dengan delapan optimizer dari bobot awal yang sama. Hasil MLP, CNN, dan EfficientNetV2B0 ada di bagian hasil dan perbandingan.`;
  $("#invWrap").hidden=g.C!==1;
  $("#stripMsg").classList.remove("err");
  $("#stripMsg").textContent=`Unggah JPG atau PNG, atau tarik gambar ke kotak +. Gambar dipotong persegi di tengah lalu diperkecil ke ${g.H} × ${g.H}`+(g.C===1?", diubah ke grayscale, dan (jika dicentang) dibalik warnanya.":".");
}
let dsSeq=0;
async function setDataset(ds){if(ds===S.ds)return;const seq=++dsSeq,m=$("#dsMsg");m.classList.remove("err");
  const title=ds==="fashion"?"Fashion-MNIST":"CIFAR-10";
  document.querySelectorAll("[data-ds]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.ds===ds?"true":"false"));
  m.textContent=`Memuat ${title}…`;$("#flowWrap").classList.add("busy");
  let c;try{[c]=await Promise.all([loadDS(ds),loadW(S.opt,ds)]);}catch(e){if(seq!==dsSeq)return;$("#flowWrap").classList.remove("busy");
    m.classList.add("err");m.textContent=`${title} gagal dimuat. Pilih dataset itu sekali lagi untuk mencoba ulang.`;syncDSButtons();return;}
  if(seq!==dsSeq)return;
  S.ds=ds;D=c.D;atlasCtx=c.atlasCtx;CLS=D.classes;try{localStorage.setItem("cnn-dataset",ds);}catch(e){}
  S.cls=-1;S.pred=null;S.shown=24;S.sel=null;$("#detail").hidden=true;
  $("#flowWrap").classList.remove("busy");m.textContent="";
  buildClassChips();syncFilters();syncDS();buildOptList();renderOptInfo();renderFacts();
  const keep=S.cur&&S.cur.kind==="u"?S.cur:{kind:"g",k:D.demo[9]};select(keep);
  if(S.cmpAll){S.cmpAll=false;loadAll();}else renderCompare();
  $("#grid").innerHTML="";$("#more").innerHTML="";$("#summary").textContent=L.loadingRuns;
  let X;try{X=await loadExp(ds);}catch(e){if(seq!==dsSeq)return;$("#summary").innerHTML=`<span style="color:var(--bad)">${L.expErr}</span>`;$("#cfgNote").textContent=L.expErr;return;}
  if(seq!==dsSeq)return;
  // keep the same run identities when the other dataset has them; never substitute another run
  if(!X.byKey[S.runA])S.runA="historical__cnn-adam__s42";if(S.runB&&!X.byKey[S.runB])S.runB="";
  if(!X.byKey[S.detail])S.detail="";
  syncRunPickers();renderGallery();renderTraining();}
function syncDSButtons(){document.querySelectorAll("[data-ds]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.ds===S.ds?"true":"false"));}
document.querySelectorAll("[data-ds]").forEach(b=>b.onclick=()=>setDataset(b.dataset.ds));
$("#fInv").addEventListener("change",e=>{S.inv=e.target.checked;if(S.cur&&S.cur.kind==="u")select(S.cur);else buildStrip();});
document.querySelectorAll("#studySeg button").forEach(b=>b.onclick=()=>{S.study=b.dataset.v;const c=configs(S.study),cur=EX().byKey[S.detail];
  const same=cur&&c.find(x=>x.model===cur.model&&x.opt===cur.opt&&x.runs.length);const pick=same||c.find(x=>x.model==="effnet"&&x.runs.length)||c.find(x=>x.runs.length);
  if(pick)setDetail((pick.runs.find(r=>cur&&r.seed===cur.seed)||pick.runs[0]).key);else renderCfg();});
document.querySelectorAll("#metricSeg button").forEach(b=>b.onclick=()=>{S.metric=b.dataset.v;renderCfg();});
document.querySelectorAll("#hmSeg button").forEach(b=>b.onclick=()=>{S.hm=b.dataset.v;try{localStorage.setItem("cnn-model",S.hm);}catch(e){}pickHist(S.hopt);});

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
  if(D){buildStrip();renderFlow();renderTraining();renderGallery();renderCompare();}}
$("#themeBtn").onclick=()=>setTheme(document.documentElement.dataset.theme==="dark"?"light":"dark");
document.querySelectorAll(".flow-bar .seg button").forEach(b=>b.onclick=()=>{S.chMode=b.dataset.ch;
  document.querySelectorAll(".flow-bar .seg button").forEach(x=>x.setAttribute("aria-pressed",x===b?"true":"false"));closeDetail();renderFlow();});
$("#file").addEventListener("change",e=>{const f=e.target.files[0];if(f)handleFile(f);});
$("#cmpBtn").onclick=loadAll;
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)document.querySelectorAll(".nav a").forEach(a=>a.classList.toggle("on",a.getAttribute("href")==="#"+e.target.id));}),{rootMargin:"-40% 0px -55% 0px"});
/* the flow links are drawn from element positions, so redraw whenever the canvas changes width (window, sidebar) */
let rz;new ResizeObserver(()=>{cancelAnimationFrame(rz);rz=requestAnimationFrame(drawLinks);}).observe($("#flowWrap"));

/* ---------- boot ---------- */
(async()=>{
  initSidebar();
  let t="light";try{t=localStorage.getItem("cnn-theme")||t;}catch(e){}
  document.documentElement.dataset.theme=t;$("#themeBtn").setAttribute("aria-pressed",t==="dark"?"true":"false");setRamp();
  try{let d="cifar10";try{d=localStorage.getItem("cnn-dataset")||d;}catch(e){}if(!DSLIST.some(x=>x.key===d))d="cifar10";
    try{const mm=localStorage.getItem("cnn-model");if(mm==="mlp"||mm==="cnn")S.hm=mm;}catch(e){}
    S.ds=d;const [c]=await Promise.all([loadDS(d),loadW(S.opt,d)]);D=c.D;atlasCtx=c.atlasCtx;CLS=D.classes;
    try{const r=await fetch("data/summary.json");if(r.ok)SUMMARY=await r.json();}catch(e){}
  }catch(e){const b=$("#boot");b.classList.add("err");b.textContent="Data model gagal dimuat. Muat ulang halaman; kalau masih gagal, berkas data tidak ikut terunggah.";return;}
  $("#boot").remove();["alur","hasil","training"].forEach(id=>{$("#"+id).hidden=false;io.observe($("#"+id));});
  buildOptList();renderOptInfo();renderFacts();buildFilters();syncDS();
  select({kind:"g",k:D.demo[9]});
  // run index: small, loaded after the flow is on screen; predictions and curves load per run on demand
  $("#summary").textContent=L.loadingRuns;
  try{await loadExp(S.ds);}catch(e){$("#summary").innerHTML=`<span style="color:var(--bad)">${L.expErr}</span>`;$("#cfgNote").textContent=L.expErr;window.__ready=true;return;}
  // open on the experiment that has EfficientNet results (the full 224 px recipe has not been run yet)
  const effIn=st=>EX().runs.some(r=>r.study===st&&r.model==="effnet");
  if(!effIn(S.study)&&effIn("budget"))S.study="budget";
  syncRunPickers();renderGallery();renderTraining();
  window.__ready=true;
})();
