
/* ======================================================================================
   Input panel of the network flow (drawing, file, imported image), imported datasets and live evaluation.
   Everything here runs on the models the page already has: forward() for MLP and CNN
   (16-bit weights from w-<model>-<opt>.json) and the ONNX session for EfficientNetV2B0.
   Imported files never leave the browser. Metrics on imported files are computed here and are
   always labelled as sample metrics, never mixed with the official 10,000-image test results.
   ====================================================================================== */
/*T:ID*/const T={
  srcDraw:"Gambar", srcFile:"Unggah gambar",
  brush:"Kuas", color:"Warna", clear:"Hapus kanvas", undo:"Urungkan",
  drawHintGray:"Gambar objek terang di atas latar hitam, seperti citra Fashion-MNIST.",
  drawHintRgb:"Gambar dengan warna di atas latar putih. Sketsa berada di luar distribusi data latih CIFAR-10, jadi prediksinya hanya untuk eksperimen.",
  pickFile:"Pilih gambar", dropHere:"atau tarik gambar ke kotak ini", fileTypes:"PNG, JPG, atau WEBP, maksimal 10 MB",
  bg:"Latar", bgAuto:"Otomatis dari tepi gambar", bgDark:"Objek terang, latar gelap", bgLight:"Objek gelap, latar terang (dibalik)",
  crop:"Potong margin kosong dan pusatkan objek", exact:H=>`Pakai piksel apa adanya bila gambar sudah ${H} × ${H} (seperti data uji)`,
  original:"Gambar asli", modelIn:H=>`Masukan model, ${H} × ${H}`, emptySrc:"Belum ada gambar",
  infoExact:H=>`Ukuran sudah ${H} × ${H}: piksel dipakai apa adanya, tanpa inversi atau crop.`,
  infoInv:"Warna dibalik (tepi gambar terang).", infoNoInv:"Warna tidak dibalik.",
  infoCrop:(w,h)=>`Objek dipotong (${w} × ${h} px) lalu dipusatkan.`, infoNoCrop:"Gambar utuh, tanpa crop.",
  infoGray:"Diubah ke grayscale.",
  effFail:"EfficientNetV2B0 tidak bisa dimuat. Periksa koneksi internet, atau pilih MLP atau CNN.",
  effOffline:"EfficientNetV2B0 tidak tersedia di versi offline. Pilih MLP atau CNN.", saveJson:"Unduh prediksi (.json)",
  srcGroup:"Sumber gambar", srcEx:"Contoh bawaan", stripAria:"Pilih gambar uji untuk diperiksa", exHint:"Sepuluh citra dari data uji, satu per kelas.",
  exNote:"Citra uji dipakai apa adanya, tanpa prapemrosesan. Pilih Gambar atau Unggah gambar untuk mencoba gambar lain.",
  waitSrc:"Belum ada gambar. Alur jaringan di bawah masih menampilkan gambar sebelumnya.", badFile:"Pilih berkas PNG, JPG, atau WEBP yang tidak lebih dari 10 MB.",
  dsTitle:"Evaluasi data sendiri", dsLead:"Impor gambar berlabel milikmu, jelajahi, dan ukur kinerja model langsung di browser.",
  impZip:"Impor ZIP", impFolder:"Impor folder", impFiles:"Impor berkas", impClear:"Kosongkan",
  impFormat:(cls)=>`Label dibaca dari labels.csv (kolom filename dan label, label berupa 0 sampai 9 atau nama kelas) atau dari nama folder, misalnya 0_${cls[0].replace(/[^A-Za-z0-9]+/g,"_")}/ atau ${cls[1].replace(/[^A-Za-z0-9]+/g,"_")}/. Maksimal 3.000 gambar, 5 MB per gambar, 100 MB total. Berkas tetap di perangkatmu.`,
  impEmpty:"Belum ada gambar yang diimpor untuk dataset ini.",
  impReading:"Membaca berkas…", impUnzip:"Mengekstrak ZIP di browser…",
  impDone:(n,l)=>`${n} gambar dimuat, ${l} berlabel.`,
  impNoImg:"Tidak ada gambar PNG, JPG, atau WEBP yang bisa dipakai.",
  zipBad:"Berkas ini bukan ZIP yang valid.", zip64:"ZIP64 tidak didukung. Kompres ulang dengan ZIP biasa atau impor sebagai folder.",
  zipMethod:m=>`Metode kompresi ZIP ${m} tidak didukung (hanya stored dan deflate).`,
  zipNoInflate:"Browser ini tidak bisa mengekstrak ZIP. Impor sebagai folder.",
  zipBig:"ZIP lebih dari 200 MB.", tooMany:"Lebih dari 3.000 gambar; hanya 3.000 pertama yang dimuat.",
  repTitle:"Pemeriksaan impor",
  repLabeled:n=>`${n} berlabel`, repUnl:n=>`${n} tanpa label (ditampilkan, tidak ikut evaluasi)`,
  repBadLbl:(n,ex)=>`${n} label tidak dikenal untuk dataset ini, misalnya ${ex}`,
  repSkipType:n=>`${n} berkas dilewati karena bukan gambar`, repSkipBig:n=>`${n} gambar dilewati karena lebih dari 5 MB`,
  repSkipTotal:"Sisanya dilewati karena total melebihi 100 MB",
  repCsv:(f,n,u)=>`${f}: ${n} baris dibaca, ${u} tidak cocok dengan berkas mana pun`, repCsvCols:f=>`${f}: kolom filename atau label tidak ditemukan`,
  repOther:t=>`Banyak label cocok dengan kelas ${t}. Ganti dataset di atas bila gambarnya dari ${t}.`,
  repSize:(n,H)=>`${n} gambar berukuran ${H} × ${H}`,
  galEmpty:"Tidak ada gambar untuk filter ini.", allCls:"Semua kelas", unl:"Tanpa label", stAll:"Semua", stOk:"Benar", stBad:"Salah",
  cellFilter:(t,p)=>`Filter: label ${t}, prediksi ${p}`, clrFilter:"Hapus filter",
  galCount:(a,b,n)=>`${a}–${b} dari ${n}`, prev:"Sebelumnya", next:"Berikutnya", openLab:"Lihat di Alur jaringan",
  evTitle:"Evaluasi langsung", modelA:"Model", modelB:"Bandingkan dengan", none:"Tidak ada",
  prepMode:"Prapemrosesan", prepBench:"Seperti data uji: hanya diubah ukurannya", prepLab:"Ikuti pengaturan panel masukan",
  run:"Jalankan evaluasi", cancel:"Batalkan", needLabels:"Impor gambar berlabel dulu untuk menjalankan evaluasi.",
  progress:(a,b)=>`Memproses ${a} dari ${b} gambar…`, cancelled:"Evaluasi dibatalkan.",
  evFail:"Evaluasi gagal: ", decodeFail:n=>`${n} gambar tidak bisa dibaca dan dilewati.`,
  sampleBadge:n=>`Hasil pada ${n} gambar impor. Ini bukan hasil benchmark resmi.`,
  legA:"(angka besar)", legB:"(angka kecil)", kAcc:"Akurasi", kP:"Presisi makro", kR:"Recall makro", kF:"F1 makro", kN:"Gambar dievaluasi",
  benchRef:a=>`Benchmark resmi model ini: ${a} pada 10.000 citra uji.`,
  macroNote:"Rata-rata makro dihitung dari kelas yang ada di label impor. Kelas yang tidak pernah diprediksi mendapat presisi 0.",
  perCls:"Per kelas", clsHead:["Kelas","Presisi","Recall","F1","Jumlah"],
  cmTitle:"Confusion matrix", cmNote:"Baris label, kolom prediksi. Klik sel untuk menampilkan gambarnya di galeri.",
  agree:"Kesepakatan kedua model", agBoth:"Keduanya benar", agA:"Hanya model pertama benar", agB:"Hanya pembanding benar", agNone:"Keduanya salah",
  csvRows:"Unduh prediksi per gambar (.csv)", csvMetrics:"Unduh metrik (.csv)",
  stale:"Data atau model berubah sejak evaluasi terakhir. Jalankan lagi untuk memperbarui.",
  fromEval:"Gambar dari Evaluasi data sendiri", truthLbl:"Label", backGal:"Kembali ke galeri",
  ready:n=>`${n} gambar berlabel siap dievaluasi`, adv:"Opsi lanjutan", galTitle:"Gambar impor",
};/*T:END*/

const LAB={mode:"ex",brush:18,color:"#1D4ED8",inv:"auto",crop:true,exact:true,src:null,srcW:0,srcH:0,srcName:"",truth:null,pre:null,lastG:null,strokes:[]};
const IMP={};        // dataset key -> {items:[{path,name,blob,url,label,bytes}], report}
const IMPJOB={};     // dataset key -> latest import/clear sequence number
const EVAL={};       // dataset key -> last evaluation result
const LIM={files:3000,img:5*1024*1024,total:100*1024*1024,zip:200*1024*1024,dim:4096,upload:10*1024*1024};
const GV={cls:-1,st:"all",cell:null,page:0,per:40};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt=(v,d=1)=>pct(v,d);
const entById=id=>E().find(e=>e.id===id);
const isOffline=()=>!!window.CNN_OFFLINE;

/* ---------- preprocessing ----------
   Benchmark images (already H × H) keep their pixels. Everything else: grayscale and optional
   inversion for Fashion-MNIST (light object on black, as in the training data), optional crop to
   the object, then resize. Returns the model input in [0, 1], channels last. */
function borderMean(v,w,h,ch=1,c=0){const t=Math.max(1,Math.floor(Math.min(w,h)*.07));let s=0,n=0;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(x<t||y<t||x>=w-t||y>=h-t){s+=v[(y*w+x)*ch+c];n++;}return s/Math.max(1,n);}
function prep(src,sw,sh,o){const {H,C}=G();
  if(o.exact&&sw===H&&sh===H){const cv=document.createElement("canvas");cv.width=H;cv.height=H;const x=cv.getContext("2d",{willReadFrequently:true});
    x.imageSmoothingEnabled=false;x.drawImage(src,0,0);const d=x.getImageData(0,0,H,H).data,px=new Float32Array(H*H*C);
    for(let i=0;i<H*H;i++){if(C===3){for(let c=0;c<3;c++)px[i*3+c]=d[i*4+c]/255;}
      else{const r=d[i*4],g=d[i*4+1],b=d[i*4+2];px[i]=(r===g&&g===b?r:0.299*r+0.587*g+0.114*b)/255;}}
    return {px,exact:true,invert:false,box:null,sw,sh};}
  const sc=Math.min(1,256/Math.max(sw,sh)),w=Math.max(1,Math.round(sw*sc)),h=Math.max(1,Math.round(sh*sc));
  const wc=document.createElement("canvas");wc.width=w;wc.height=h;const wx=wc.getContext("2d",{willReadFrequently:true});
  wx.fillStyle=o.fill||"#fff";wx.fillRect(0,0,w,h);wx.imageSmoothingEnabled=true;wx.imageSmoothingQuality="high";wx.drawImage(src,0,0,w,h);
  const im=wx.getImageData(0,0,w,h),d=im.data,n=w*h;let invert=false,mask=new Uint8Array(n),fillCol="#000";
  if(C===1){const g=new Float32Array(n);for(let i=0;i<n;i++)g[i]=0.299*d[i*4]+0.587*d[i*4+1]+0.114*d[i*4+2];
    invert=o.inv==="light"||(o.inv==="auto"&&borderMean(g,w,h)>127);
    for(let i=0;i<n;i++){const v=invert?255-g[i]:g[i];d[i*4]=d[i*4+1]=d[i*4+2]=v;d[i*4+3]=255;mask[i]=v>35?1:0;}
    wx.putImageData(im,0,0);}
  else{const rgb=new Float32Array(n*3);for(let i=0;i<n;i++)for(let c=0;c<3;c++)rgb[i*3+c]=d[i*4+c];
    const bm=[0,1,2].map(c=>borderMean(rgb,w,h,3,c));fillCol=`rgb(${bm.map(Math.round)})`;
    for(let i=0;i<n;i++){const dd=Math.abs(d[i*4]-bm[0])+Math.abs(d[i*4+1]-bm[1])+Math.abs(d[i*4+2]-bm[2]);mask[i]=dd>60?1:0;}}
  let box={x:0,y:0,w,h},cropped=false;
  if(o.crop){let x0=w,y0=h,x1=-1,y1=-1,cnt=0;for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(mask[y*w+x]){cnt++;if(x<x0)x0=x;if(x>x1)x1=x;if(y<y0)y0=y;if(y>y1)y1=y;}
    if(cnt>0&&cnt<n*.92){const p=Math.max(1,Math.round(.07*Math.max(x1-x0+1,y1-y0+1)));
      const l=Math.max(0,x0-p),t=Math.max(0,y0-p),r=Math.min(w,x1+1+p),b=Math.min(h,y1+1+p);box={x:l,y:t,w:r-l,h:b-t};cropped=true;}}
  const oc=document.createElement("canvas");oc.width=H;oc.height=H;const ox=oc.getContext("2d",{willReadFrequently:true});
  ox.imageSmoothingEnabled=true;ox.imageSmoothingQuality="high";
  if(C===1){ox.fillStyle="#000";ox.fillRect(0,0,H,H);const room=cropped?H-4:H,f=Math.min(room/box.w,room/box.h),dw=box.w*f,dh=box.h*f;
    ox.drawImage(wc,box.x,box.y,box.w,box.h,(H-dw)/2,(H-dh)/2,dw,dh);}
  else{let side,cx,cy;if(cropped){side=Math.max(box.w,box.h);cx=box.x+box.w/2;cy=box.y+box.h/2;}else{side=Math.min(w,h);cx=w/2;cy=h/2;}
    ox.fillStyle=fillCol;ox.fillRect(0,0,H,H);const f=H/side;ox.drawImage(wc,0,0,w,h,(0-(cx-side/2))*f,(0-(cy-side/2))*f,w*f,h*f);
    box={x:Math.max(0,cx-side/2),y:Math.max(0,cy-side/2),w:Math.min(side,w),h:Math.min(side,h)};}
  const od=ox.getImageData(0,0,H,H).data,px=new Float32Array(H*H*C);
  for(let i=0;i<H*H;i++){if(C===3){for(let c=0;c<3;c++)px[i*3+c]=od[i*4+c]/255;}else px[i]=od[i*4]/255;}
  return {px,exact:false,invert,cropped,box:{x:box.x/w,y:box.y/h,w:box.w/w,h:box.h/h},bw:Math.round(box.w/sc),bh:Math.round(box.h/sc),gray:C===1,sw,sh};}

/* ---------- inference on any trained entry, in batches ---------- */
async function predictBatch(e,pxs,ds=S.ds){
  if(e.browser){const W=await loadW(e.opt,ds,e.model);return pxs.map(p=>forward(p,W,false).prob);}
  if(isOffline())throw new Error("offline");
  const sess=await effSession(ds,e.opt),{H,C}=G(),n=pxs.length,sz=H*H*C,x=new Float32Array(n*sz);
  pxs.forEach((p,i)=>{for(let j=0;j<sz;j++)x[i*sz+j]=p[j]*255;});
  const out=await sess.run({[sess.inputNames[0]]:new ort.Tensor("float32",x,[n,H,H,C])}),d=out[sess.outputNames[0]].data;
  return [...Array(n)].map((_,i)=>Array.from(d.slice(i*10,i*10+10)));}
const argmax=a=>{let k=0;for(let i=1;i<a.length;i++)if(a[i]>a[k])k=i;return k;};
function modelOptions(sel,withNone){sel.innerHTML="";if(withNone){const o=document.createElement("option");o.value="";o.textContent=T.none;sel.append(o);}
  MODELS.forEach(m=>{const es=ofModel(m);if(!es.length)return;const g=document.createElement("optgroup");g.label=MNAME[m];
    es.forEach(e=>{const o=document.createElement("option");o.value=e.id;o.textContent=`${label(e)} (${pct(e.acc,2)})`;g.append(o);});sel.append(g);});}

/* ---------- input panel markup (top of the network flow) and the imported-dataset section ---------- */
function labMount(){
  $("#inPanel").innerHTML=`<div class="in-grid">
    <div class="in-src">
      <div class="seg" role="group" aria-label="${T.srcGroup}" id="labSrc">
        <button type="button" data-m="ex">${T.srcEx}</button><button type="button" data-m="draw">${T.srcDraw}</button><button type="button" data-m="file">${T.srcFile}</button></div>
      <div id="labEx" class="lab-src"><div class="strip" id="strip" role="group" aria-label="${T.stripAria}"></div><p class="hint">${T.exHint}</p></div>
      <div id="labDraw" class="lab-src" hidden>
        <div class="draw-box"><canvas id="labCanvas" width="280" height="280" aria-label="${T.srcDraw}"></canvas></div>
        <div class="draw-tools">
          <label class="rng">${T.brush} <input type="range" id="labBrush" min="4" max="40" value="${LAB.brush}"><b id="labBrushV">${LAB.brush} px</b></label>
          <label class="rng" id="labColorWrap">${T.color} <input type="color" id="labColor" value="${LAB.color}"></label>
          <button type="button" class="btn sm" id="labUndo">${T.undo}</button><button type="button" class="btn sm" id="labClear">${T.clear}</button></div>
        <p class="hint" id="labDrawHint"></p></div>
      <div id="labFile" class="lab-src" hidden>
        <div class="lab-drop" id="labDrop"><button type="button" class="btn" id="labPick">${T.pickFile}</button><span>${T.dropHere}</span><small>${T.fileTypes}</small><small id="labFileName" class="fname"></small></div>
        <input type="file" id="labFileIn" accept="image/png,image/jpeg,image/webp" class="sr" tabindex="-1"></div>
      <div id="labImport" class="lab-src" hidden><div class="from-ev"><span class="hint">${T.fromEval}</span>
        <p class="from-ev-nm"><b id="labImpName"></b> <span id="labImpSize"></span></p><p id="labImpLbl"></p>
        <button type="button" class="btn sm" id="labBack">${T.backGal}</button></div></div>
    </div>
    <div class="in-prep">
      <p class="hint" id="labExNote">${T.exNote}</p>
      <div id="labPrep">
        <div class="prep">
          <label class="ctl" id="labBgWrap"><span>${T.bg}</span><select id="labBg"><option value="auto">${T.bgAuto}</option><option value="dark">${T.bgDark}</option><option value="light">${T.bgLight}</option></select></label>
          <label class="check"><input type="checkbox" id="labCrop"> ${T.crop}</label>
          <label class="check"><input type="checkbox" id="labExact"> <span id="labExactT"></span></label></div>
        <div class="prev2"><figure><figcaption>${T.original}</figcaption><div class="pv"><canvas id="labOrig"></canvas><span class="pv-empty" id="labOrigE">${T.emptySrc}</span></div></figure>
          <figure><figcaption id="labInT"></figcaption><div class="pv"><canvas id="labInput" class="pix"></canvas></div></figure></div>
        <p class="hint" id="labInfo" aria-live="polite"></p>
      </div>
      <div class="lab-acts"><button type="button" class="btn sm" id="labJson">${T.saveJson}</button></div>
    </div>
  </div>`;
  $("#evaluasi").innerHTML=`<div class="sec-head"><div><h2>${T.dsTitle}</h2><p>${T.dsLead}</p></div></div>
  <div class="panel imp">
    <div class="imp-bar"><div class="lab-acts"><button type="button" class="btn sm primary" id="impZip">${T.impZip}</button><button type="button" class="btn sm" id="impFolder">${T.impFolder}</button>
      <button type="button" class="btn sm" id="impFiles">${T.impFiles}</button><button type="button" class="btn sm" id="impClear">${T.impClear}</button></div>
      <input type="file" id="impZipIn" accept=".zip,application/zip" class="sr" tabindex="-1"><input type="file" id="impFolderIn" webkitdirectory directory multiple class="sr" tabindex="-1">
      <input type="file" id="impFilesIn" multiple accept="image/png,image/jpeg,image/webp,.csv,text/csv" class="sr" tabindex="-1"></div>
    <p class="hint" id="impFormat"></p>
    <p class="hint" id="impMsg" aria-live="polite"></p>
    <div id="impReport"></div>
  </div>
  <div class="panel ev" id="evPanel" hidden>
    <h3>${T.evTitle}</h3>
    <p class="ev-ready" id="evReady"></p>
    <div class="ev-ctl">
      <label class="ctl"><span>${T.modelA}</span><select id="evA"></select></label>
      <div class="lab-acts"><button type="button" class="btn primary" id="evRun">${T.run}</button><button type="button" class="btn" id="evCancel" hidden>${T.cancel}</button></div></div>
    <details class="adv"><summary>${T.adv}</summary><div class="ev-ctl">
      <label class="ctl"><span>${T.modelB}</span><select id="evB"></select></label>
      <label class="ctl"><span>${T.prepMode}</span><select id="evPrep"><option value="bench">${T.prepBench}</option><option value="lab">${T.prepLab}</option></select></label></div></details>
    <div class="evprog" id="evProg" hidden><span class="bar"><i id="evBar"></i></span><span id="evProgT"></span></div>
    <p class="hint" id="evMsg" aria-live="polite"></p>
    <div id="evOut"></div>
  </div>
  <div class="panel gal" id="galPanel" hidden>
    <h3>${T.galTitle}</h3>
    <div class="chips" id="galCls" role="group" aria-label="${T.galTitle}"></div>
    <div class="ctrl-line"><div class="seg" id="galSt" role="group" hidden><button type="button" data-v="all">${T.stAll}</button><button type="button" data-v="ok">${T.stOk}</button><button type="button" data-v="bad">${T.stBad}</button></div><span id="galCell"></span></div>
    <div class="igrid" id="galGrid"></div>
    <div class="pager" id="galPager"></div>
  </div>`;
  labWire();}

/* ---------- input panel behaviour: every source ends in select({kind:"x", px}) so the flow below shows it ---------- */
let drawCtx=null,drawing=false,lastPt=null;
function labCanvasReset(){const cv=$("#labCanvas");drawCtx=cv.getContext("2d");const {C}=G();
  drawCtx.fillStyle=C===1?"#000":"#fff";drawCtx.fillRect(0,0,cv.width,cv.height);LAB.strokes=[];}
function labWire(){
  $("#labSrc").querySelectorAll("button").forEach(b=>b.onclick=()=>labMode(b.dataset.m));
  const cv=$("#labCanvas");
  const pt=e=>{const r=cv.getBoundingClientRect();return [(e.clientX-r.left)*cv.width/r.width,(e.clientY-r.top)*cv.height/r.height];};
  const stroke=(a,b)=>{const {C}=G();drawCtx.strokeStyle=C===1?"#fff":LAB.color;drawCtx.lineWidth=LAB.brush;drawCtx.lineCap="round";drawCtx.lineJoin="round";
    drawCtx.beginPath();drawCtx.moveTo(a[0],a[1]);drawCtx.lineTo(b[0],b[1]);drawCtx.stroke();};
  cv.addEventListener("pointerdown",e=>{e.preventDefault();cv.setPointerCapture(e.pointerId);drawing=true;
    LAB.strokes.push(drawCtx.getImageData(0,0,cv.width,cv.height));if(LAB.strokes.length>20)LAB.strokes.shift();lastPt=pt(e);stroke(lastPt,lastPt);labDrawn(false);});
  cv.addEventListener("pointermove",e=>{if(!drawing)return;const p=pt(e);stroke(lastPt,p);lastPt=p;labDrawn(false);});
  const end=()=>{if(!drawing)return;drawing=false;labDrawn(true);};
  cv.addEventListener("pointerup",end);cv.addEventListener("pointercancel",end);
  $("#labBrush").oninput=e=>{LAB.brush=+e.target.value;$("#labBrushV").textContent=LAB.brush+" px";};
  $("#labColor").oninput=e=>{LAB.color=e.target.value;};
  $("#labClear").onclick=()=>{labCanvasReset();LAB.src=null;labRun();};
  $("#labUndo").onclick=()=>{const s=LAB.strokes.pop();if(s){drawCtx.putImageData(s,0,0);if(!LAB.strokes.length)LAB.src=null;labRun();}};
  $("#labPick").onclick=()=>$("#labFileIn").click();
  $("#labFileIn").onchange=e=>{const f=e.target.files[0];if(f)labFile(f);e.target.value="";};
  const dz=$("#labDrop");["dragenter","dragover"].forEach(t=>dz.addEventListener(t,ev=>{ev.preventDefault();dz.classList.add("over");}));
  ["dragleave","drop"].forEach(t=>dz.addEventListener(t,()=>dz.classList.remove("over")));
  $("#labBg").onchange=e=>{LAB.inv=e.target.value;labRun();};
  $("#labCrop").onchange=e=>{LAB.crop=e.target.checked;labRun();};
  $("#labExact").onchange=e=>{LAB.exact=e.target.checked;labRun();};
  $("#labJson").onclick=labJson;$("#labBack").onclick=labBack;
  $("#impZip").onclick=()=>$("#impZipIn").click();$("#impFolder").onclick=()=>$("#impFolderIn").click();$("#impFiles").onclick=()=>$("#impFilesIn").click();
  $("#impZipIn").onchange=async e=>{const f=e.target.files[0];e.target.value="";if(f)await impZip(f);};
  $("#impFolderIn").onchange=async e=>{const fs=[...e.target.files];e.target.value="";if(fs.length)await impList(fs.map(f=>({path:f.webkitRelativePath||f.name,file:f})));};
  $("#impFilesIn").onchange=async e=>{const fs=[...e.target.files];e.target.value="";if(fs.length)await impList(fs.map(f=>({path:f.name,file:f})));};
  $("#impClear").onclick=()=>{IMPJOB[S.ds]=(IMPJOB[S.ds]||0)+1;impDrop(S.ds);$("#impMsg").textContent="";$("#impMsg").classList.remove("err");$("#evMsg").textContent="";$("#evMsg").classList.remove("err");renderImport();};
  $("#galSt").querySelectorAll("button").forEach(b=>b.onclick=()=>{GV.st=b.dataset.v;GV.page=0;renderGal();});
  $("#evRun").onclick=runEval;$("#evCancel").onclick=()=>{EVJOB.cancel=true;};
  ["evA","evB","evPrep"].forEach(id=>$("#"+id).onchange=()=>renderEval());}
let drawT=0;
function labDrawn(final){if(LAB.mode!=="draw")return;LAB.srcName="sketch";LAB.truth=null;
  LAB.src=$("#labCanvas");LAB.srcW=280;LAB.srcH=280;clearTimeout(drawT);drawT=setTimeout(labRun,final?0:120);}
function labShow(m){LAB.mode=m;$("#labSrc").querySelectorAll("button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.m===m?"true":"false"));
  $("#labEx").hidden=m!=="ex";$("#labDraw").hidden=m!=="draw";$("#labFile").hidden=m!=="file";$("#labImport").hidden=m!=="import";
  $("#labPrep").hidden=m==="ex";$("#labExNote").hidden=m!=="ex";}
function labMode(m){labShow(m);
  if(m==="ex"){LAB.src=null;LAB.pre=null;select(S.cur&&S.cur.kind==="g"?S.cur:{kind:"g",k:LAB.lastG??D.demo[9]});return;}
  if(m==="draw"){LAB.src=LAB.strokes.length?$("#labCanvas"):null;LAB.srcW=LAB.srcH=280;LAB.srcName="sketch";LAB.truth=null;}
  else if(m==="file"){LAB.src=LAB.file?LAB.file.img:null;if(LAB.file){LAB.srcW=LAB.file.w;LAB.srcH=LAB.file.h;LAB.srcName=LAB.file.name;}LAB.truth=null;}
  else{LAB.src=LAB.imp?LAB.imp.img:null;if(LAB.imp){LAB.srcW=LAB.imp.w;LAB.srcH=LAB.imp.h;LAB.srcName=LAB.imp.path;LAB.truth=LAB.imp.label;}}
  labRun();}
async function decodeImage(blob){if(blob.size>LIM.upload*20)throw new Error("size");
  const bm=await createImageBitmap(blob);if(!bm.width||!bm.height||bm.width>LIM.dim||bm.height>LIM.dim){bm.close&&bm.close();throw new Error("dim");}return bm;}
async function labFile(f){if(LAB.mode!=="file")labShow("file");const m=$("#labFileName");
  if(!/^image\/(png|jpeg|webp)$/.test(f.type)||f.size>LIM.upload){m.textContent=T.badFile;m.classList.add("err");return;}
  try{const bm=await decodeImage(f);LAB.file={img:bm,w:bm.width,h:bm.height,name:f.name};m.classList.remove("err");m.textContent=`${f.name} (${bm.width} × ${bm.height})`;labMode("file");}
  catch(e){m.textContent=T.badFile;m.classList.add("err");}}
/* an imported image opened from the gallery: its name and label stay visible above the flow, with a way back to the same gallery view */
async function labFromImport(it,i){try{const bm=await decodeImage(it.blob);LAB.imp&&LAB.imp.img&&LAB.imp.img.close&&LAB.imp.img.close();
    LAB.imp={img:bm,w:bm.width,h:bm.height,path:it.path,label:it.label,i};
    $("#labImpName").textContent=it.name;$("#labImpName").title=it.path;$("#labImpSize").textContent=`${bm.width} × ${bm.height}`;
    $("#labImpLbl").innerHTML=`${T.truthLbl}: ${it.label==null?`<i>${T.unl}</i>`:`<b>${esc(CLS[it.label])}</b>`}`;
    labMode("import");document.getElementById("alur").scrollIntoView();$("#labBack").focus({preventScroll:true});}
  catch(e){$("#impMsg").textContent=T.decodeFail(1);}}
/* the gallery filters (GV) are not touched by the flow, so going back shows the same page and filter */
function labBack(){const c=LAB.imp&&$("#galGrid").querySelector(`[data-i="${LAB.imp.i}"]`);
  if(c){c.scrollIntoView({block:"center"});c.focus({preventScroll:true});}else $("#galPanel").scrollIntoView();}
function labOpts(){const {C}=G();return {inv:C===1?LAB.inv:"none",crop:LAB.crop,exact:LAB.exact,fill:LAB.mode==="draw"?(C===1?"#000":"#fff"):"#fff"};}
function labRun(){if(LAB.mode==="ex")return;const {H,C}=G(),oc=$("#labOrig"),ic=$("#labInput");
  $("#labExactT").textContent=T.exact(H);$("#labInT").textContent=T.modelIn(H);$("#labBgWrap").hidden=C!==1;$("#labColorWrap").hidden=C===1;
  $("#labDrawHint").textContent=C===1?T.drawHintGray:T.drawHintRgb;
  if(!LAB.src){$("#labOrigE").hidden=false;oc.width=oc.height=1;ic.width=ic.height=H;ic.getContext("2d").clearRect(0,0,H,H);$("#labInfo").textContent=T.waitSrc;LAB.pre=null;return;}
  $("#labOrigE").hidden=true;
  const p=prep(LAB.src,LAB.srcW,LAB.srcH,labOpts());LAB.pre=p;
  const s=Math.min(1,320/Math.max(LAB.srcW,LAB.srcH));oc.width=Math.max(1,Math.round(LAB.srcW*s));oc.height=Math.max(1,Math.round(LAB.srcH*s));
  const ox=oc.getContext("2d");ox.imageSmoothingEnabled=LAB.srcW>oc.width;ox.drawImage(LAB.src,0,0,oc.width,oc.height);oc.classList.toggle("pix",LAB.srcW<=64);
  if(p.box&&p.cropped){ox.strokeStyle=css("--accent");ox.lineWidth=2;ox.setLineDash([6,4]);ox.strokeRect(p.box.x*oc.width+1,p.box.y*oc.height+1,p.box.w*oc.width-2,p.box.h*oc.height-2);}
  drawPx(ic,p.px);
  const info=p.exact?[T.infoExact(H)]:[...(C===1?[T.infoGray,p.invert?T.infoInv:T.infoNoInv]:[]),p.cropped?T.infoCrop(p.bw,p.bh):T.infoNoCrop];
  $("#labInfo").textContent=info.join(" ");
  select({kind:"x",px:p.px,truth:LAB.truth,name:LAB.srcName});}
function download(name,text,type){const a=document.createElement("a"),u=URL.createObjectURL(new Blob([text],{type}));a.href=u;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),2000);}
const stamp=()=>new Date().toISOString().replace(/[:.]/g,"-").slice(0,19);
/* the prediction of the image shown in the flow, with the model that is selected at the top of the page */
async function labJson(){if(!S.cur)return;const {H,C}=G(),e=curE(),ds=S.ds,b=$("#labJson");b.disabled=true;
  let px,prob;try{px=S.model==="effnet"?await effPx(ds):curPx();[prob]=await predictBatch(e,[px],ds);}catch(err){b.disabled=false;return;}
  b.disabled=false;const top=argmax(prob),g=S.cur.kind==="g",p=g?null:LAB.pre,o=labOpts();
  const input=g?{source:"test example",test_id:EXPC[ds].pool_ids[S.cur.k],original_size:[H,H],label:CLS[S.cur.truth],preprocessing:{exact_pixels:true}}
    :{source:LAB.mode,name:LAB.srcName,original_size:p?[p.sw,p.sh]:null,label:S.cur.truth==null?null:CLS[S.cur.truth],
      preprocessing:!p||p.exact?{exact_pixels:true}:{grayscale:C===1,background:o.inv,inverted:p.invert,crop_to_object:o.crop,cropped:!!p.cropped,crop_box_fraction:p.cropped?p.box:null}};
  input.model_input_0_255=Array.from(px,v=>Math.round(v*255));
  const out={app:"CNN Explorer",created:new Date().toISOString(),dataset:{key:ds,title:D.title,classes:CLS,input_shape:[H,H,C]},
    model:{id:e.id,architecture:MNAME[e.model],optimizer:e.name,parameters:e.params,seed:e.seed,official_test_accuracy:e.acc,
      engine:e.browser?"JavaScript forward pass, 16-bit weights":"ONNX Runtime Web, 16-bit weights"},input,
    prediction:{class_id:top,class:CLS[top],confidence:prob[top],correct:S.cur.truth==null?null:top===S.cur.truth,probabilities:Object.fromEntries(CLS.map((c,i)=>[c,prob[i]]))}};
  download(`prediction-${ds}-${e.id}-${stamp()}.json`,JSON.stringify(out,null,1),"application/json");}

/* ---------- import: ZIP (stored and deflate, read locally), folders and loose files ---------- */
async function unzip(file){if(file.size>LIM.zip)throw new Error(T.zipBig);
  const buf=await file.arrayBuffer(),dv=new DataView(buf),b=new Uint8Array(buf);let end=-1;
  for(let i=b.length-22;i>=Math.max(0,b.length-65557);i--)if(dv.getUint32(i,true)===0x06054b50){end=i;break;}
  if(end<0)throw new Error(T.zipBad);
  const count=dv.getUint16(end+10,true),start=dv.getUint32(end+16,true);if(count===0xffff||start===0xffffffff)throw new Error(T.zip64);
  const dec=new TextDecoder("utf-8"),out=[];let p=start;
  for(let i=0;i<count;i++){if(p+46>b.length||dv.getUint32(p,true)!==0x02014b50)throw new Error(T.zipBad);
    const method=dv.getUint16(p+10,true),csize=dv.getUint32(p+20,true),size=dv.getUint32(p+24,true),nl=dv.getUint16(p+28,true),xl=dv.getUint16(p+30,true),cl=dv.getUint16(p+32,true),off=dv.getUint32(p+42,true);
    const name=dec.decode(b.subarray(p+46,p+46+nl));p+=46+nl+xl+cl;
    if(name.endsWith("/"))continue;
    out.push({path:name,size,read:async()=>{if(off+30>b.length||dv.getUint32(off,true)!==0x04034b50)throw new Error(T.zipBad);
      const s=off+30+dv.getUint16(off+26,true)+dv.getUint16(off+28,true);if(s+csize>b.length)throw new Error(T.zipBad);
      const comp=b.slice(s,s+csize);if(method===0)return comp;
      if(method!==8)throw new Error(T.zipMethod(method));if(typeof DecompressionStream==="undefined")throw new Error(T.zipNoInflate);
      const raw=new Uint8Array(await new Response(new Blob([comp]).stream().pipeThrough(new DecompressionStream("deflate-raw"))).arrayBuffer());
      if(raw.length!==size)throw new Error(T.zipBad);return raw;}});}
  return out;}
const IMG_RE=/\.(png|jpe?g|webp)$/i,MIME={png:"image/png",jpg:"image/jpeg",jpeg:"image/jpeg",webp:"image/webp"};
const hidden=p=>p.split("/").some(s=>s.startsWith(".")||s==="__MACOSX");
const norm=s=>String(s).toLowerCase().replace(/[^a-z0-9]/g,"");
function classIndex(ds){const cls=DSC[ds].D.classes,m=new Map();cls.forEach((c,i)=>{m.set(norm(c),i);});
  if(ds==="fashion"){[["tshirt",0],["top",0],["tshirttop",0],["boot",9],["ankleboots",9]].forEach(([k,v])=>m.set(k,v));}
  else{[["plane",0],["car",1],["auto",1]].forEach(([k,v])=>m.set(k,v));}return m;}
function parseLabel(v,idx){const s=String(v).trim();if(/^\d+$/.test(s)){const n=+s;return n>=0&&n<10?n:undefined;}const n=idx.get(norm(s));return n===undefined?undefined:n;}
function folderLabel(path,idx){const seg=path.split("/").slice(0,-1).reverse();
  for(const s of seg){const m=s.match(/^(\d{1,2})(?:[ _\-.](.*))?$/);if(m){const n=+m[1];if(n<10)return n;}const k=idx.get(norm(s));if(k!==undefined)return k;}return null;}
function readCsv(txt){txt=txt.replace(/^﻿/,"");const first=txt.split(/\r?\n/)[0]||"",dl=[",",";","\t"].sort((a,b)=>first.split(b).length-first.split(a).length)[0];
  const rows=[];let row=[],cur="",q=false;
  for(let i=0;i<txt.length;i++){const ch=txt[i];
    if(q){if(ch==='"'){if(txt[i+1]==='"'){cur+='"';i++;}else q=false;}else cur+=ch;}
    else if(ch==='"')q=true;else if(ch===dl){row.push(cur);cur="";}else if(ch==="\n"||ch==="\r"){if(ch==="\r"&&txt[i+1]==="\n")i++;row.push(cur);rows.push(row);row=[];cur="";}else cur+=ch;}
  if(cur||row.length){row.push(cur);rows.push(row);}return rows.filter(r=>r.some(c=>c.trim()!==""));}
function csvMap(name,txt,idx,rep){const rows=readCsv(txt);if(!rows.length)return null;const h=rows[0].map(c=>norm(c));
  const fi=["filename","file","path","image","img","name","filepath","imagepath"].map(k=>h.indexOf(k)).find(i=>i>=0),
        li=["labelid","label","classid","class","classname","target","y","category"].map(k=>h.indexOf(k)).find(i=>i>=0);
  if(fi===undefined||li===undefined){rep.csv.push(T.repCsvCols(name));return null;}
  const base=name.includes("/")?name.slice(0,name.lastIndexOf("/")+1):"",m=new Map();
  for(const r of rows.slice(1)){if(r.length<=Math.max(fi,li))continue;m.set(r[fi].trim().replace(/\\/g,"/").replace(/^\.\//,""),{v:r[li],used:false,base});}
  return {name,m};}
function impDrop(ds){if(EVJOB.busy&&ds===S.ds)EVJOB.cancel=true;const st=IMP[ds];if(st)st.items.forEach(it=>URL.revokeObjectURL(it.url));
  delete IMP[ds];delete EVAL[ds];
  if(ds===S.ds){GV.cls=-1;GV.st="all";GV.cell=null;GV.page=0;
    if(LAB.imp){LAB.imp.img&&LAB.imp.img.close&&LAB.imp.img.close();LAB.imp=null;if(LAB.mode==="import")labMode("ex");}}}
/* entries: [{path, size, read() -> Uint8Array|Blob}] */
async function impEntries(entries,ds=S.ds,tok){if(tok==null)tok=IMPJOB[ds]=(IMPJOB[ds]||0)+1;
  const idx=classIndex(ds),other=ds==="fashion"?"cifar10":"fashion",oidx=DSC[other]?classIndex(other):null;
  const rep={skipType:0,skipBig:0,skipTotal:false,csv:[],bad:[],other:0,many:false,exact:0};
  entries=entries.map(e=>({...e,path:e.path.replace(/\\/g,"/")})).filter(e=>!hidden(e.path));
  const csvs=entries.filter(e=>/\.csv$/i.test(e.path)),maps=[];
  for(const c of csvs.sort((a,b)=>a.path.split("/").length-b.path.split("/").length)){try{const raw=await c.read();if(tok!==IMPJOB[ds])return;
    const txt=typeof raw==="string"?raw:new TextDecoder().decode(raw instanceof Blob?new Uint8Array(await raw.arrayBuffer()):raw);
    const mm=csvMap(c.path,txt,idx,rep);if(mm)maps.push(mm);}catch(e){rep.csv.push(c.path+": "+e.message);}}
  if(tok!==IMPJOB[ds])return;
  let imgs=entries.filter(e=>IMG_RE.test(e.path));rep.skipType=entries.length-imgs.length-csvs.length;
  imgs.sort((a,b)=>a.path.localeCompare(b.path,undefined,{numeric:true}));
  if(imgs.length>LIM.files){rep.many=true;imgs=imgs.slice(0,LIM.files);}
  const items=[];let total=0;
  for(const e of imgs){if(e.size>LIM.img){rep.skipBig++;continue;}if(total+e.size>LIM.total){rep.skipTotal=true;break;}
    let raw;try{raw=await e.read();}catch(err){rep.csv.push(e.path+": "+err.message);continue;}
    if(tok!==IMPJOB[ds]){items.forEach(it=>URL.revokeObjectURL(it.url));return;}
    const ext=e.path.split(".").pop().toLowerCase(),blob=raw instanceof Blob?raw:new Blob([raw],{type:MIME[ext]});total+=blob.size;
    let lab=null,src=null;
    for(const mp of maps){const rel=mp.name.includes("/")&&e.path.startsWith(mp.name.slice(0,mp.name.lastIndexOf("/")+1))?e.path.slice(mp.name.lastIndexOf("/")+1):e.path;
      const hit=mp.m.get(rel)||mp.m.get(e.path)||mp.m.get(e.path.split("/").pop());if(hit){hit.used=true;src=hit.v;break;}}
    if(src!=null){const v=parseLabel(src,idx);if(v===undefined){rep.bad.push(String(src));if(oidx&&parseLabel(src,oidx)!==undefined&&!/^\d+$/.test(String(src).trim()))rep.other++;}else lab=v;}
    else{lab=folderLabel(e.path,idx);if(lab==null&&oidx&&folderLabel(e.path,oidx)!=null&&!/(^|\/)\d/.test(e.path))rep.other++;}
    items.push({path:e.path,name:e.path.split("/").pop(),blob,url:URL.createObjectURL(blob),label:lab,bytes:blob.size});}
  if(tok!==IMPJOB[ds]){items.forEach(it=>URL.revokeObjectURL(it.url));return;}
  impDrop(ds);
  rep.csvRows=maps.map(mp=>[mp.name,mp.m.size,[...mp.m.values()].filter(v=>!v.used).length]);
  IMP[ds]={items,report:rep};
  // read sizes in the background, to count benchmark-sized images
  const H=DSC[ds].D.shape[0];Promise.all(items.slice(0,600).map(async it=>{try{const b=await createImageBitmap(it.blob);it.w=b.width;it.h=b.height;b.close&&b.close();}catch(e){it.broken=true;}}))
    .then(()=>{if(IMP[ds]&&IMP[ds].items===items){rep.exact=items.filter(it=>it.w===H&&it.h===H).length;rep.sized=Math.min(items.length,600);if(ds===S.ds)renderReport();}});
  if(ds===S.ds){renderImport();$("#impMsg").classList.remove("err");$("#impMsg").textContent=items.length?T.impDone(nf(items.length),nf(items.filter(i=>i.label!=null).length)):T.impNoImg;}}
async function impZip(f){const ds=S.ds,tok=IMPJOB[ds]=(IMPJOB[ds]||0)+1;$("#impMsg").classList.remove("err");$("#impMsg").textContent=T.impUnzip;
  let entries;try{entries=await unzip(f);}catch(e){if(ds===S.ds&&tok===IMPJOB[ds]){$("#impMsg").textContent=e.message||T.zipBad;$("#impMsg").classList.add("err");}return;}
  if(tok!==IMPJOB[ds])return;await impEntries(entries,ds,tok);}
async function impList(list){const ds=S.ds,tok=IMPJOB[ds]=(IMPJOB[ds]||0)+1;$("#impMsg").classList.remove("err");$("#impMsg").textContent=T.impReading;
  await impEntries(list.map(({path,file})=>({path,size:file.size,read:async()=>/\.csv$/i.test(path)?await file.text():file})),ds,tok);}

/* ---------- gallery ---------- */
function renderImport(){$("#impFormat").textContent=T.impFormat(CLS);renderReport();renderGal();renderEval();}
function renderReport(){const st=IMP[S.ds],box=$("#impReport");if(!st){box.innerHTML=`<p class="hint">${T.impEmpty}</p>`;return;}const r=st.report,it=st.items,{H}=G();
  const nl=it.filter(i=>i.label!=null).length,li=[`<li class="ok">${T.repLabeled(nf(nl))}</li>`];
  if(it.length-nl)li.push(`<li>${T.repUnl(nf(it.length-nl))}</li>`);
  if(r.bad.length)li.push(`<li class="warn">${T.repBadLbl(nf(r.bad.length),esc([...new Set(r.bad)].slice(0,3).join(", ")))}</li>`);
  if(r.other>Math.max(3,it.length*.2))li.push(`<li class="warn">${T.repOther(S.ds==="fashion"?"CIFAR-10":"Fashion-MNIST")}</li>`);
  r.csvRows&&r.csvRows.forEach(([f,n,u])=>li.push(`<li${u?' class="warn"':""}>${esc(T.repCsv(f,nf(n),nf(u)))}</li>`));
  r.csv.forEach(c=>li.push(`<li class="warn">${esc(c)}</li>`));
  if(r.skipType)li.push(`<li>${T.repSkipType(nf(r.skipType))}</li>`);if(r.skipBig)li.push(`<li class="warn">${T.repSkipBig(nf(r.skipBig))}</li>`);
  if(r.skipTotal)li.push(`<li class="warn">${T.repSkipTotal}</li>`);if(r.many)li.push(`<li class="warn">${T.tooMany}</li>`);
  if(r.sized)li.push(`<li>${T.repSize(nf(r.exact),H)}${r.sized<it.length?` (${nf(r.sized)})`:""}</li>`);
  box.innerHTML=`<div class="rep"><b>${T.repTitle}</b><ul>${li.join("")}</ul></div>`;}
function galRows(){const st=IMP[S.ds];if(!st)return [];const ev=EVAL[S.ds],byPath=ev&&!ev.stale?ev.byIdx:null;
  return st.items.map((it,i)=>({it,i,r:byPath?byPath[i]:null})).filter(({it,r})=>{
    if(GV.cls===-2&&it.label!=null)return false;if(GV.cls>=0&&it.label!==GV.cls)return false;
    if(GV.cell){if(!r||it.label!==GV.cell[0]||r.pa!==GV.cell[1])return false;}
    if(GV.st!=="all"){if(!r)return false;if((GV.st==="ok")!==(r.pa===it.label))return false;}return true;});}
function renderGal(){const st=IMP[S.ds],cls=$("#galCls"),grid=$("#galGrid"),pg=$("#galPager");
  $("#galPanel").hidden=!(st&&st.items.length);
  if(!st||!st.items.length){cls.innerHTML="";grid.innerHTML="";pg.innerHTML="";$("#galSt").hidden=true;$("#galCell").innerHTML="";return;}
  const cnt=new Array(10).fill(0);let unl=0;st.items.forEach(it=>it.label==null?unl++:cnt[it.label]++);
  const chip=(v,txt)=>`<button type="button" class="chip" data-v="${v}" aria-pressed="${GV.cls===v}">${txt}</button>`;
  cls.innerHTML=chip(-1,`${T.allCls} <small>${nf(st.items.length)}</small>`)+CLS.map((c,i)=>cnt[i]?chip(i,`${esc(c)} <small>${nf(cnt[i])}</small>`):"").join("")+(unl?chip(-2,`${T.unl} <small>${nf(unl)}</small>`):"");
  cls.querySelectorAll("button").forEach(b=>b.onclick=()=>{GV.cls=+b.dataset.v;GV.cell=null;GV.page=0;renderGal();});
  const ev=EVAL[S.ds];$("#galSt").hidden=!(ev&&!ev.stale);$("#galSt").querySelectorAll("button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.v===GV.st?"true":"false"));
  $("#galCell").innerHTML=GV.cell?`${T.cellFilter(esc(CLS[GV.cell[0]]),esc(CLS[GV.cell[1]]))} <button type="button" class="chip" id="galCellX">${T.clrFilter}</button>`:"";
  const x=$("#galCellX");if(x)x.onclick=()=>{GV.cell=null;renderGal();};
  const rows=galRows(),pages=Math.max(1,Math.ceil(rows.length/GV.per));GV.page=Math.min(GV.page,pages-1);const a=GV.page*GV.per,sl=rows.slice(a,a+GV.per);
  grid.innerHTML=sl.length?"":`<div class="empty">${T.galEmpty}</div>`;
  sl.forEach(({it,i,r})=>{const b=document.createElement("button");b.type="button";b.className="icard";b.dataset.i=i;
    const small=it.w&&it.w<=64;b.innerHTML=`<img src="${it.url}" alt="" loading="lazy"${small?' class="pix"':""}>
      <span class="nm" title="${esc(it.path)}">${esc(it.name)}</span><span class="lb">${it.label==null?`<i>${T.unl}</i>`:esc(CLS[it.label])}</span>
      ${r?`<span class="tab ${r.pa===it.label?"ok":"bad"}">${esc(CLS[r.pa])}</span>`:""}<span class="act">${T.openLab}</span>`;
    b.setAttribute("aria-label",`${it.name}, ${it.label==null?T.unl:CLS[it.label]}${r?`, ${CLS[r.pa]}`:""}. ${T.openLab}`);
    b.onclick=()=>labFromImport(it,i);grid.append(b);});
  pg.innerHTML=rows.length>GV.per?`<button type="button" class="btn sm" id="pgP"${GV.page?"":" disabled"}>${T.prev}</button><span>${T.galCount(nf(a+1),nf(Math.min(a+GV.per,rows.length)),nf(rows.length))}</span><button type="button" class="btn sm" id="pgN"${GV.page<pages-1?"":" disabled"}>${T.next}</button>`:(rows.length?`<span>${T.galCount(nf(1),nf(rows.length),nf(rows.length))}</span>`:"");
  const pp=$("#pgP"),pn=$("#pgN");if(pp)pp.onclick=()=>{GV.page--;renderGal();};if(pn)pn.onclick=()=>{GV.page++;renderGal();};}

/* ---------- live evaluation ---------- */
const EVJOB={cancel:false,busy:false};
function metricsOf(rows,key){const K=10,cm=[...Array(K)].map(()=>new Array(K).fill(0));let ok=0;
  rows.forEach(r=>{cm[r.label][r[key]]++;if(r.label===r[key])ok++;});
  const per=cm.map((row,k)=>{const tp=row[k],sup=row.reduce((a,b)=>a+b,0),pr=cm.reduce((a,rr)=>a+rr[k],0),p=pr?tp/pr:0,rc=sup?tp/sup:0;return {p,r:rc,f:p+rc?2*p*rc/(p+rc):0,sup,pred:pr};});
  const pres=per.filter(x=>x.sup>0),m=k=>pres.reduce((a,x)=>a+x[k],0)/Math.max(1,pres.length);
  return {n:rows.length,acc:rows.length?ok/rows.length:0,mp:m("p"),mr:m("r"),mf:m("f"),per,cm};}
function evPrepOpts(){return $("#evPrep").value==="lab"?{...labOpts(),fill:"#fff"}:{inv:"none",crop:false,exact:true,fill:"#fff"};}
async function runEval(){const st=IMP[S.ds],msgEl=$("#evMsg");msgEl.classList.remove("err");
  if(!st||!st.items.some(i=>i.label!=null)){msgEl.textContent=T.needLabels;return;}
  if(EVJOB.busy)return;const ds=S.ds,A=entById($("#evA").value),B=entById($("#evB").value)||null,prepMode=$("#evPrep").value,opts=evPrepOpts();
  const items=st.items.map((it,i)=>({it,i})).filter(x=>x.it.label!=null);
  EVJOB.busy=true;EVJOB.cancel=false;$("#evRun").disabled=true;$("#evCancel").hidden=false;$("#evProg").hidden=false;msgEl.textContent="";
  if((!A.browser||(B&&!B.browser))&&isOffline()){msgEl.textContent=T.effOffline;msgEl.classList.add("err");EVJOB.busy=false;$("#evRun").disabled=false;$("#evCancel").hidden=true;$("#evProg").hidden=true;return;}
  const rows=[];let broken=0;const bs=A.browser&&(!B||B.browser)?64:24;
  try{for(let s=0;s<items.length;s+=bs){if(EVJOB.cancel||ds!==S.ds)break;
      $("#evBar").style.width=(s/items.length*100).toFixed(1)+"%";$("#evProgT").textContent=T.progress(nf(s),nf(items.length));
      const chunk=items.slice(s,s+bs),ok=[],pxs=[];
      for(const x of chunk){try{const bm=await decodeImage(x.it.blob);x.it.w=bm.width;x.it.h=bm.height;pxs.push(prep(bm,bm.width,bm.height,opts).px);ok.push(x);bm.close&&bm.close();}catch(e){broken++;x.it.broken=true;}}
      if(!ok.length)continue;
      const pa=await predictBatch(A,pxs,ds),pb=B?await predictBatch(B,pxs,ds):null;
      ok.forEach((x,j)=>{const ta=argmax(pa[j]),r={i:x.i,path:x.it.path,label:x.it.label,pa:ta,ca:pa[j][ta],probA:pa[j]};
        if(pb){const tb=argmax(pb[j]);r.pb=tb;r.cb=pb[j][tb];}rows.push(r);});
      await new Promise(r=>setTimeout(r,0));}}
  catch(err){msgEl.textContent=T.evFail+(err&&err.message==="offline"?T.effOffline:(!A.browser||(B&&!B.browser))?T.effFail:(err.message||err));msgEl.classList.add("err");}
  EVJOB.busy=false;$("#evRun").disabled=false;$("#evCancel").hidden=true;$("#evProg").hidden=true;
  if(EVJOB.cancel){msgEl.textContent=T.cancelled;return;}if(ds!==S.ds||!rows.length){renderEval();return;}
  const byIdx={};rows.forEach(r=>byIdx[r.i]=r);
  EVAL[ds]={A,B,prepMode,rows,byIdx,mA:metricsOf(rows,"pa"),mB:B?metricsOf(rows,"pb"):null,broken,n:items.length,stale:false};
  if(broken)msgEl.textContent=T.decodeFail(nf(broken));
  GV.st="all";GV.cell=null;renderEval();renderGal();}
function cmTable(m,clickable){const acc=hex(css("--accent")),bad=hex(css("--bad")),z=hex(css("--surface")),mx=Math.max(1,...m.cm.map((r,i)=>r[i])),mo=Math.max(1,...m.cm.flatMap((r,i)=>r.filter((_,j)=>j!==i)));
  let t=`<table class="cm"><thead><tr><th></th>${CLS.map(c=>`<th scope="col">${esc(c)}</th>`).join("")}</tr></thead><tbody>`;
  m.cm.forEach((row,i)=>{t+=`<tr><th scope="row" style="text-align:right">${esc(CLS[i])}</th>`;row.forEach((v,j)=>{const u=v?(i===j?v/mx:.15+.85*v/mo):0,e=i===j?acc:bad,c=[0,1,2].map(k=>Math.round(z[k]+(e[k]-z[k])*u)),fg=cmFg(c);
    t+=`<td style="background:rgb(${c});color:${fg}">${clickable&&v?`<button type="button" class="cmb" data-t="${i}" data-p="${j}" aria-label="${v}: ${esc(CLS[i])} → ${esc(CLS[j])}">${v}</button>`:v}</td>`;});t+="</tr>";});
  return t+"</tbody></table>";}
function renderEval(){const out=$("#evOut");if(!out)return;const ev=EVAL[S.ds],st=IMP[S.ds];
  ["evA","evB"].forEach((id,k)=>{const sel=$("#"+id),v=sel.value;modelOptions(sel,k===1);sel.value=entById(v)?v:(k?"":(ev?ev.A.id:curE().id));});
  if(ev&&!ev.stale&&($("#evA").value!==ev.A.id||($("#evB").value||null)!==(ev.B?ev.B.id:null)||$("#evPrep").value!==ev.prepMode))ev.dirty=true;else if(ev)ev.dirty=false;
  const nl=st?st.items.filter(i=>i.label!=null).length:0;$("#evPanel").hidden=!(st&&st.items.length);
  $("#evReady").textContent=nl?T.ready(nf(nl)):T.needLabels;
  $("#evRun").disabled=EVJOB.busy||!nl;
  if(!ev){out.innerHTML="";return;}
  const {mA,mB,A,B}=ev,k=(lab,a,b)=>`<div class="kpi"><span>${lab}</span><b>${a}</b>${b!=null?`<em>${b}</em>`:""}</div>`;
  const hdr=B?`<p class="kpi-leg"><b>${esc(label(A))}</b> <span>${T.legA}</span> · <em>${esc(label(B))}</em> <span>${T.legB}</span></p>`:"";
  let ag="";if(B){const n=[0,0,0,0];ev.rows.forEach(r=>{const a=r.pa===r.label,b=r.pb===r.label;n[a&&b?0:a?1:b?2:3]++;});
    ag=`<div class="agree"><h4>${T.agree}</h4><ul><li>${T.agBoth} <b>${nf(n[0])}</b></li><li>${T.agA} <b>${nf(n[1])}</b></li><li>${T.agB} <b>${nf(n[2])}</b></li><li>${T.agNone} <b>${nf(n[3])}</b></li></ul></div>`;}
  const clsRows=mA.per.map((x,i)=>x.sup||x.pred?`<tr><th scope="row">${esc(CLS[i])}</th><td>${fmt(x.p)}${mB?` <small>${fmt(mB.per[i].p)}</small>`:""}</td><td>${fmt(x.r)}${mB?` <small>${fmt(mB.per[i].r)}</small>`:""}</td><td>${fmt(x.f)}${mB?` <small>${fmt(mB.per[i].f)}</small>`:""}</td><td>${nf(x.sup)}</td></tr>`:"").join("");
  out.innerHTML=`${ev.dirty?`<p class="hint warn">${T.stale}</p>`:""}<p class="badge-sample">${T.sampleBadge(nf(mA.n))}</p>
    ${hdr}<div class="kpis">${k(T.kAcc,fmt(mA.acc,2),mB&&fmt(mB.acc,2))}${k(T.kP,fmt(mA.mp,2),mB&&fmt(mB.mp,2))}${k(T.kR,fmt(mA.mr,2),mB&&fmt(mB.mr,2))}${k(T.kF,fmt(mA.mf,2),mB&&fmt(mB.mf,2))}${k(T.kN,nf(mA.n),null)}</div>
    <p class="hint">${T.benchRef(pct(A.acc,2))}${B?" "+esc(label(B))+": "+pct(B.acc,2)+".":""} ${T.macroNote}</p>
    <div class="ev-grid"><div><h4>${T.perCls}${B?` <small>(${esc(label(A))} / ${esc(label(B))})</small>`:""}</h4><div class="tblw"><table class="tbl"><thead><tr>${T.clsHead.map(h=>`<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${clsRows}</tbody></table></div>${ag}</div>
      <div><h4>${T.cmTitle}: ${esc(label(A))}</h4><p class="hint">${T.cmNote}</p><div class="cm-scroll">${cmTable(mA,true)}</div></div></div>
    <div class="lab-acts"><button type="button" class="btn sm" id="evCsv">${T.csvRows}</button><button type="button" class="btn sm" id="evCsvM">${T.csvMetrics}</button></div>`;
  out.querySelectorAll(".cmb").forEach(b=>b.onclick=()=>{GV.cell=[+b.dataset.t,+b.dataset.p];GV.cls=-1;GV.st="all";GV.page=0;renderGal();$("#galGrid").scrollIntoView({block:"nearest"});});
  $("#evCsv").onclick=()=>evCsvRows(ev);$("#evCsvM").onclick=()=>evCsvMetrics(ev);}
const csvCell=v=>{const s=v==null?"":String(v);return /[",\n;]/.test(s)?`"${s.replace(/"/g,'""')}"`:s;};
const csvText=rows=>rows.map(r=>r.map(csvCell).join(",")).join("\n")+"\n";
function evCsvRows(ev){const h=["file","label_id","label",`pred_id_${ev.A.id}`,`pred_${ev.A.id}`,`confidence_${ev.A.id}`,`correct_${ev.A.id}`];
  if(ev.B)h.push(`pred_id_${ev.B.id}`,`pred_${ev.B.id}`,`confidence_${ev.B.id}`,`correct_${ev.B.id}`);CLS.forEach(c=>h.push(`prob_${ev.A.id}_${c}`));
  const rows=ev.rows.map(r=>{const o=[r.path,r.label,CLS[r.label],r.pa,CLS[r.pa],r.ca.toFixed(6),r.pa===r.label?1:0];
    if(ev.B)o.push(r.pb,CLS[r.pb],r.cb.toFixed(6),r.pb===r.label?1:0);r.probA.forEach(p=>o.push(p.toFixed(6)));return o;});
  download(`evaluation-${S.ds}-${ev.A.id}${ev.B?"-vs-"+ev.B.id:""}-${stamp()}.csv`,csvText([h,...rows]),"text/csv");}
function evCsvMetrics(ev){const rows=[["scope","model","dataset","preprocessing","n_images","class","precision","recall","f1","support","accuracy","official_test_accuracy"]];
  for(const [e,m] of [[ev.A,ev.mA],...(ev.B?[[ev.B,ev.mB]]:[])]){
    rows.push(["imported_sample_macro",e.id,S.ds,ev.prepMode,m.n,"",m.mp.toFixed(6),m.mr.toFixed(6),m.mf.toFixed(6),m.n,m.acc.toFixed(6),e.acc]);
    m.per.forEach((x,i)=>{if(x.sup||x.pred)rows.push(["imported_sample_class",e.id,S.ds,ev.prepMode,m.n,CLS[i],x.p.toFixed(6),x.r.toFixed(6),x.f.toFixed(6),x.sup,"",""]);});}
  download(`metrics-${S.ds}-${ev.A.id}${ev.B?"-vs-"+ev.B.id:""}-${stamp()}.csv`,csvText(rows),"text/csv");}

/* ---------- hooks used by the rest of the page ---------- */
function labInit(){labMount();labCanvasReset();labShow("ex");$("#labBg").value=LAB.inv;$("#labCrop").checked=LAB.crop;$("#labExact").checked=LAB.exact;renderImport();}
/* after a dataset change the flow shows a test example again; drawings and files belong to the previous input size */
function labOnDataset(){LAB.file&&LAB.file.img&&LAB.file.img.close&&LAB.file.img.close();LAB.imp&&LAB.imp.img&&LAB.imp.img.close&&LAB.imp.img.close();
  LAB.file=null;LAB.imp=null;LAB.pre=null;LAB.src=null;LAB.lastG=null;$("#labFileName").textContent="";
  $("#evMsg").textContent="";$("#evMsg").classList.remove("err");
  const st=IMP[S.ds];$("#impMsg").classList.remove("err");
  $("#impMsg").textContent=st&&st.items.length?T.impDone(nf(st.items.length),nf(st.items.filter(i=>i.label!=null).length)):"";
  labCanvasReset();labShow("ex");GV.cls=-1;GV.cell=null;GV.st="all";GV.page=0;renderImport();}
function labOnPick(){if($("#evA"))renderEval();}
function labOnTheme(){if($("#evA"))renderEval();}
function labOnSelect(it){if(it.kind==="g"){LAB.lastG=it.k;if(LAB.mode!=="ex"&&$("#labSrc"))labShow("ex");}}
