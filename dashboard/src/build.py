"""Build the dashboard pages from these sources.

  head.html + base.css + body.html + app.js  ->  ../index.id.html (Indonesian source text)
  the same page with the EN pairs below      ->  ../index.html    (English, default)

Run from anywhere:  python dashboard/src/build.py
Optional single-file offline copy (not used by GitHub Pages, written to dashboard/dist/, which git ignores):
    python dashboard/src/build.py --offline [--dataset fashion|cifar10|all]
It embeds the dataset files, the MLP/CNN weights and the run files into one HTML page, so it opens from disk
without a web server or internet. EfficientNetV2B0 needs ONNX Runtime Web from a CDN and is left out of it.
Every Indonesian string in app.js / body.html that the visitor can see needs an (ID, EN) pair here;
the build stops if a pair no longer matches the source, and prints Indonesian-looking words left on
the English page.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent

P = [
('<span class="lbl">Evaluasi data sendiri</span>', '<span class="lbl">Evaluate your own data</span>'),
('<p>Lepaskan gambar di mana saja untuk diperiksa</p>', '<p>Drop the image anywhere to inspect it</p>'),
('aria-label="Tutup menu samping" title="Tutup menu samping"', 'aria-label="Close the side menu" title="Close the side menu"'),
('"Tutup menu samping":"Buka menu samping"', '"Close the side menu":"Open the side menu"'),
('aria-label="Menu samping"', 'aria-label="Side menu"'),
('>Mode gelap<', '>Dark mode<'),
('>Keduanya benar<', '>Both right<'),
('>Keduanya salah<', '>Both wrong<'),
('<h3 id="runTitle">Kurva training</h3>', '<h3 id="runTitle">Training curves</h3>'),
('<html lang="id">', '<html lang="en">'),
('<span>Analisis Data Citra</span>', '<span>Image Data Analysis</span>'),
('aria-label="Bagian halaman"', 'aria-label="Page sections"'),
('href="index.html" lang="en" hreflang="en"', 'href="index.id.html" lang="id" hreflang="id"'),
('<span>English</span><b class="lang-short">EN</b>', '<span>Bahasa Indonesia</span><b class="lang-short">ID</b>'),
('Memuat data uji dan hasil training…', 'Loading test data and training results…'),
('Bagaimana CNN membaca gambar 32 × 32', 'How a CNN reads a 32 × 32 image'),
('aria-label="Channel yang ditampilkan"', 'aria-label="Channels shown"'),
('>8 paling aktif<', '>8 most active<'),
('>Channel #0 sampai #7<', '>Channels #0 to #7<'),
('<span>Aktivasi rendah</span>', '<span>Low activation</span>'),
('tinggi (skala per kolom). Klik peta untuk melihat detailnya.', 'high (scaled per column). Click a map to see its details.'),
('Gambar ini menurut kedelapan optimizer', 'This image according to all eight optimizers'),
('>Bandingkan semua optimizer<', '>Compare all optimizers<'),
('<h2>Hasil pada data uji</h2>', '<h2>Results on test data</h2>'),
('aria-pressed="true">Semua</button>', 'aria-pressed="true">All</button>'),
('aria-pressed="false">Cocok</button>', 'aria-pressed="false">Match</button>'),
('aria-pressed="false">Tidak cocok</button>', 'aria-pressed="false">No match</button>'),
('>Urutkan</label>', '>Sort by</label>'),
('>Keyakinan tertinggi<', '>Highest confidence<'),
('>Keyakinan terendah<', '>Lowest confidence<'),
('>Indeks citra<', '>Image index<'),
('id="lblKelas">Kelas asli<', 'id="lblKelas">True class<'),
('aria-label="Pilih optimizer di grafik"', 'aria-label="Select an optimizer in the chart"'),
('id="pclsTitle">Akurasi per kelas<', 'id="pclsTitle">Per-class accuracy<'),
('Baris kelas asli, kolom kelas prediksi. Klik sel di luar diagonal untuk melihat contohnya.', 'Rows are true classes, columns are predicted classes. Click an off-diagonal cell to see examples.'),
('content:" (asli)"', 'content:" (true)"'),
('const LANG="id";', 'const LANG="en";'),
('"Periksa contoh "+', '"Inspect example "+'),
('"Unggahan "+(it.i+1)', '"Upload "+(it.i+1)'),
('"Periksa gambar unggahan "+', '"Inspect uploaded image "+'),
("</svg>Unggah';", "</svg>Upload';"),
('"Unggah gambar sendiri"', '"Upload your own image"'),
('["merah","hijau","biru"]', '["red","green","blue"]'),
('["Merah","Hijau","Biru"]', '["Red","Green","Blue"]'),
('. Buka detail`', '. Open details`'),
('"Dense 128 unit. Buka detail"', '"Dense, 128 units. Open details"'),
('<small>128 unit</small>', '<small>128 units</small>'),
('<small>10 kelas</small>', '<small>10 classes</small>'),
('<div>Ukuran<b>', '<div>Size<b>'),
('<div>Maksimum<b>${nf(mxv,3)}</b></div><div>Rata-rata<b>', '<div>Maximum<b>${nf(mxv,3)}</b></div><div>Mean<b>'),
('Bobot kernel #${c} (${3*3*C+1} parameter termasuk bias ${nf(L.b[c],3)}). Merah negatif, biru positif.', 'Kernel #${c} weights (${3*3*C+1} parameters including bias ${nf(L.b[c],3)}). Red is negative, blue is positive.'),
('aria-label="Tutup detail"', 'aria-label="Close details"'),
('<div>Unit aktif (&gt; 0)<b>${act} dari 128</b></div><div>Maksimum<b>', '<div>Active units (&gt; 0)<b>${act} of 128</b></div><div>Maximum<b>'),
('b.textContent="Memuat bobot…";$("#cmpNote").textContent="Memuat bobot kedelapan optimizer…";', 'b.textContent="Loading weights…";$("#cmpNote").textContent="Loading weights for all eight optimizers…";'),
('b.textContent="Coba muat ulang";$("#cmpNote").textContent=`${fail} optimizer gagal dimuat. Tekan tombol untuk mencoba lagi.`;', 'b.textContent="Retry loading";$("#cmpNote").textContent=`${fail} optimizer(s) failed to load. Press the button to try again.`;'),
('"Klik salah satu kartu untuk memakai optimizer itu di seluruh dashboard."', '"Click a card to use that optimizer across the whole dashboard."'),
('Bobot belum termuat', 'Weights not loaded yet'),
('<span class="tab unk">Tanpa label</span>', '<span class="tab unk">No label</span>'),
('<span class="p">Prediksi <strong>${CLS[top]}</strong>, keyakinan ${pct(p[top])}</span>', '<span class="p">Prediction <strong>${CLS[top]}</strong>, confidence ${pct(p[top])}</span>'),
('`${o.name}: prediksi ${CLS[top]}, keyakinan ${pct(p[top])}. Pakai optimizer ini`', '`${o.name}: prediction ${CLS[top]}, confidence ${pct(p[top])}. Use this optimizer`'),
('Tidak ada contoh untuk kombinasi filter ini. Ganti status, kelas, atau pembanding.', 'No examples match this combination of filters. Change the status, class, or comparison.'),
('"Tidak cocok"', '"No match"'),
('"Cocok"', '"Match"'),
('"Akurasi validasi"', '"Validation accuracy"'),
('"Loss validasi"', '"Validation loss"'),
('"Data model gagal dimuat. Muat ulang halaman; kalau masih gagal, berkas data tidak ikut terunggah."', '"Model data could not be loaded. Reload the page. If it still fails, the data files are missing next to this page, or the page was opened as a local file instead of through a web server."'),
('<title>CNN Explorer: dataset dan optimizer</title>', '<title>CNN Explorer: datasets and optimizers</title>'),
('["abu-abu"]', '["gray"]'),
('["Abu-abu"]', '["Gray"]'),
('`Channel ${CHN()[c]} dari gambar input, nilai piksel 0 sampai 1. `', '`The ${CHN()[c]} channel of the input image, pixel values from 0 to 1. `'),
('"Ketiga channel inilah yang membuat C_in conv1 bernilai 3."', '"These three channels are why C_in for conv1 is 3."'),
('"Fashion-MNIST hanya punya satu channel, jadi C_in conv1 bernilai 1."', '"Fashion-MNIST has a single channel, so C_in for conv1 is 1."'),
('`Feature map dari kernel #${c} conv1 setelah ReLU. Kernel 3 × 3 × ${G().C} di bawah digeser ke seluruh gambar (stride 1, tanpa padding), jadi ukurannya ${G().H} − 3 + 1 = ${G().c1}. Bentuk kernel ini berbeda untuk tiap optimizer karena hasil latihannya berbeda.`', '`Feature map from conv1 kernel #${c} after ReLU. The 3 × 3 × ${G().C} kernel below slides across the whole image (stride 1, no padding), so the size is ${G().H} − 3 + 1 = ${G().c1}. The kernel looks different for each optimizer because each training run ends differently.`'),
('`Max pooling 2 × 2 dari feature map conv1 #${c}. Setiap piksel adalah nilai terbesar dari kotak 2 × 2, sehingga ${G().c1} × ${G().c1} menjadi ${G().p1} × ${G().p1}. Tidak ada parameter.`', '`2 × 2 max pooling of conv1 feature map #${c}. Each pixel is the largest value in a 2 × 2 window, so ${G().c1} × ${G().c1} becomes ${G().p1} × ${G().p1}. No parameters.`'),
('`Feature map dari kernel #${c} conv2 setelah ReLU. Kernel ini membaca ke-32 channel max_pool1 sekaligus (3 × 3 × 32 bobot + 1 bias), lalu ${G().p1} − 3 + 1 = ${G().c2}.`', '`Feature map from conv2 kernel #${c} after ReLU. This kernel reads all 32 max_pool1 channels at once (3 × 3 × 32 weights + 1 bias), and ${G().p1} − 3 + 1 = ${G().c2}.`'),
('`Max pooling 2 × 2 dari conv2 #${c}: ${G().c2} × ${G().c2} menjadi ${G().p2} × ${G().p2} (baris dan kolom terakhir terbuang). Hasil ${G().p2} × ${G().p2} × 64 di-flatten menjadi ${nf(G().flat)} nilai untuk layer dense.`', '`2 × 2 max pooling of conv2 #${c}: ${G().c2} × ${G().c2} becomes ${G().p2} × ${G().p2} (the last row and column are dropped). The ${G().p2} × ${G().p2} × 64 output is flattened into ${nf(G().flat)} values for the dense layer.`'),
('<h3>dense + ReLU, 128 unit (${optOf().name})</h3><p>Setiap unit menerima ke-${nf(G().flat)} nilai hasil flatten (${nf(G().flat)} × 128 + 128 = ${nf(G().flat*128+128)} parameter, layer terbesar di CNN ini). Ditampilkan sebagai grid 8 × 16. Dropout 0,3 hanya aktif saat training, jadi tidak terlihat di sini.</p>', '<h3>dense + ReLU, 128 units (${optOf().name})</h3><p>Each unit receives all ${nf(G().flat)} flattened values (${nf(G().flat)} × 128 + 128 = ${nf(G().flat*128+128)} parameters, the largest layer in this CNN). Shown as an 8 × 16 grid. Dropout 0.3 is only active during training, so it does not appear here.</p>'),
('</svg><span class="lbl">Alur jaringan</span></a>', '</svg><span class="lbl">Network flow</span></a>'),
('</svg><span class="lbl">Hasil uji resmi</span></a>', '</svg><span class="lbl">Official test results</span></a>'),
('Label asli tidak diketahui</span><strong>Prediksi: ${CLS[top]}</strong><span style="color:var(--muted)">keyakinan ${pct(r.prob[top])}, ${on}', 'True label unknown</span><strong>Prediction: ${CLS[top]}</strong><span style="color:var(--muted)">confidence ${pct(r.prob[top])}, ${on}'),
('<strong>Prediksi: ${CLS[top]}</strong><span style="color:var(--muted)">asli ${CLS[S.cur.truth]}, keyakinan ${pct(r.prob[top])}, ${on}', '<strong>Prediction: ${CLS[top]}</strong><span style="color:var(--muted)">true ${CLS[S.cur.truth]}, confidence ${pct(r.prob[top])}, ${on}'),
('`Bagaimana ${MODEL()} membaca gambar ${g.H} × ${g.H}`', '`How ${S.model==="mlp"?"an":"a"} ${MODEL()} reads a ${g.H} × ${g.H} image`'),
('<span class="lbl">Akurasi</span>', '<span class="lbl">Accuracy</span>'),
('<span class="lbl">Detail model</span>', '<span class="lbl">Model details</span>'),
('<small>32 × 32 berwarna</small>', '<small>32 × 32 color</small>'),
('<h2>Peringkat akurasi uji</h2>', '<h2>Test accuracy ranking</h2>'),
('aria-label="Peringkat akurasi uji"', 'aria-label="Test accuracy ranking"'),
('>Bandingkan dengan<', '>Compare with<'),
('id="lblStatus">Status<', 'id="lblStatus">Status<'),
('id="lblAgree">Kedua model<', 'id="lblAgree">Both models<'),
('>Hanya terpilih benar<', '>Only selected right<'),
('>Hanya pembanding benar<', '>Only comparison right<'),
('<h2 id="detTitle">Detail model</h2>', '<h2 id="detTitle">Model details</h2>'),
('<h3 id="optTitle">Ringkasan per optimizer</h3>', '<h3 id="optTitle">Summary per optimizer</h3>'),
('<h3 id="curvesTitle">Kurva validasi per epoch</h3>', '<h3 id="curvesTitle">Validation curves per epoch</h3>'),
('b.textContent="Bandingkan semua optimizer"', 'b.textContent="Compare all optimizers"'),
('`flatten, ${nf(n)} nilai`,`Gambar ${g.H} × ${g.H} × ${g.C} diratakan menjadi satu vektor berisi ${nf(n)} angka. Mulai di sini posisi piksel tidak lagi diperhatikan: piksel yang bersebelahan diperlakukan sama seperti piksel yang berjauhan. Inilah perbedaan utama MLP dengan CNN.`', '`flatten, ${nf(n)} values`,`The ${g.H} × ${g.H} × ${g.C} image is flattened into one vector of ${nf(n)} numbers. From here on, pixel position no longer matters: neighbouring pixels are treated the same as pixels far apart. This is the main difference between an MLP and a CNN.`'),
('`dense1 + ReLU, 256 unit (${run})`,`Setiap unit menerima ke-${nf(n)} nilai input (${nf(n)} × 256 + 256 = ${nf(n*256+256)} parameter, layer terbesar di MLP ini). Ditampilkan sebagai grid 16 × 16.`', '`dense1 + ReLU, 256 units (${run})`,`Each unit receives all ${nf(n)} input values (${nf(n)} × 256 + 256 = ${nf(n*256+256)} parameters, the largest layer in this MLP). Shown as a 16 × 16 grid.`'),
('`dense2 + ReLU, 128 unit (${run})`,`Setiap unit menerima 256 nilai dari dense1 (256 × 128 + 128 = 32.896 parameter). Ditampilkan sebagai grid 8 × 16.`', '`dense2 + ReLU, 128 units (${run})`,`Each unit receives the 256 values from dense1 (256 × 128 + 128 = 32,896 parameters). Shown as an 8 × 16 grid.`'),
('<div>Unit aktif (&gt; 0)<b>${act} dari ${vals.length}</b></div>', '<div>Active units (&gt; 0)<b>${act} of ${vals.length}</b></div>'),
('Bobot masuk unit #${u}, unit paling aktif untuk gambar ini, disusun kembali ke bentuk gambar. Merah negatif, biru positif: ini pola yang "dicari" unit tersebut di seluruh gambar sekaligus.', 'Incoming weights of unit #${u}, the most active unit for this image, arranged back into the image shape. Red is negative, blue is positive: this is the pattern the unit "looks for" across the whole image at once.'),
('`${nf(n)} nilai`,r.flat', '`${nf(n)} values`,r.flat'),
('"256 unit",r.h1', '"256 units",r.h1'),
('"128 unit",r.h2', '"128 units",r.h2'),
('"Flatten. Buka detail"', '"Flatten. Open details"'),
('"Dense1 256 unit. Buka detail"', '"Dense1, 256 units. Open details"'),
('"Dense2 128 unit. Buka detail"', '"Dense2, 128 units. Open details"'),
('<div>Maksimum<b>${nf(mx,3)}</b></div></div><div id="kwrap">', '<div>Maximum<b>${nf(mx,3)}</b></div></div><div id="kwrap">'),
]


def build():
    src = lambda n: (HERE / n).read_text(encoding="utf-8")
    ID = (src("head.html") + "<style>" + src("base.css") + src("lab.css") + "</style>\n"
          + src("body.html") + "\n<script>\n" + src("app.js") + src("lab.js") + "</script>\n</body>\n</html>\n")
    ID = ID.replace("/*L:ID*/", "").replace("/*L:END*/", "") if False else ID
    missing = [old for old, new in P if old not in ID]
    assert not missing, "not found:\n" + "\n".join(m[:90] for m in missing)
    # strings of the run-based sections live in one object; swap it whole for the English one
    start, end = ID.index("/*L:ID*/"), ID.index("/*L:END*/") + len("/*L:END*/")
    en_obj = (HERE / "strings_en.js").read_text(encoding="utf-8").strip()
    keys = lambda js: re.findall(r"^\s*([A-Za-z]\w*):", js, flags=re.M)
    assert keys(ID[start:end]) == keys(en_obj), "strings_en.js keys differ from the L object in app.js"
    EN = ID[:start] + en_obj + ID[end:]
    # the image lab and live evaluation keep their strings in a second object (lab.js / strings_lab_en.js)
    start, end = EN.index("/*T:ID*/"), EN.index("/*T:END*/") + len("/*T:END*/")
    en_lab = (HERE / "strings_lab_en.js").read_text(encoding="utf-8").strip()
    assert keys(EN[start:end]) == keys(en_lab), "strings_lab_en.js keys differ from the T object in lab.js"
    EN = EN[:start] + en_lab + EN[end:]
    for old, new in sorted(P, key=lambda p: -len(p[0])):
        EN = EN.replace(old, new)
    # ID page: clearer error for opening as a local file, same as EN
    ID = ID.replace('"Data model gagal dimuat. Muat ulang halaman; kalau masih gagal, berkas data tidak ikut terunggah."',
                    '"Data model gagal dimuat. Muat ulang halaman. Kalau masih gagal, berkas data tidak ada di samping halaman ini, atau halaman dibuka langsung sebagai berkas lokal, bukan lewat web server."')
    words = r"\b(dan|yang|dari|untuk|gambar|citra|cocok|bobot|kelas|memuat|klik|tidak|dengan|hasil|pilih|akurasi|keyakinan|asli|prediksi|unggah|detail\b)"
    left = sorted({m.group(0) for m in re.finditer(words, EN, flags=re.I)})
    print("Indonesian-looking words left on the EN page:", left)
    assert "\u2014" not in EN and "\u2014" not in ID, "em dash in page text"
    (OUT / "index.html").write_text(encoding="utf-8", newline="\n", data=EN)
    (OUT / "index.id.html").write_text(encoding="utf-8", newline="\n", data=ID)
    print("built", OUT / "index.html", len(EN), "and", OUT / "index.id.html", len(ID))


def offline(datasets):
    """One self-contained HTML file per language: the page's fetch() calls for data/... are answered from
    JSON embedded in the page. Nothing about the GitHub Pages build changes."""
    import json
    data = OUT / "data"
    files = {"data/summary.json": (data / "summary.json").read_text(encoding="utf-8")}
    for ds in datasets:
        for f in sorted((data / ds).glob("*.json")) + sorted((data / ds / "runs").glob("*.json")):
            files[f.relative_to(OUT).as_posix()] = f.read_text(encoding="utf-8")
    blob = json.dumps(files, separators=(",", ":")).replace("</", "<\\/")
    shim = ('<script id="cnn-offline-data" type="application/json">' + blob + "</script>\n<script>window.CNN_OFFLINE=true;(function(){"
            "var F=JSON.parse(document.getElementById('cnn-offline-data').textContent),f=window.fetch.bind(window);"
            "window.fetch=function(u,o){var k=String(u).replace(/^\\.\\//,'');if(Object.prototype.hasOwnProperty.call(F,k))"
            "return Promise.resolve(new Response(F[k],{status:200,headers:{'Content-Type':'application/json'}}));"
            "if(k.indexOf('data/')===0)return Promise.resolve(new Response('',{status:404}));return f(u,o);};})();</script>\n")
    dist = OUT / "dist"
    dist.mkdir(exist_ok=True)
    for name, out in (("index.html", "cnn-explorer-offline.html"), ("index.id.html", "cnn-explorer-offline.id.html")):
        page = (OUT / name).read_text(encoding="utf-8")
        page = re.sub(r'<link rel="preconnect"[^>]*>\n|<link href="https://fonts.googleapis.com[^>]*>\n', "", page)
        page = page.replace('href="index.html"', 'href="cnn-explorer-offline.html"').replace('href="index.id.html"', 'href="cnn-explorer-offline.id.html"')
        i = page.index("\n<script>\n")
        (dist / out).write_text(page[:i] + "\n" + shim + page[i + 1:], encoding="utf-8", newline="\n")
        print("offline:", dist / out, f"{(dist / out).stat().st_size / 1e6:.1f} MB", "datasets:", ", ".join(datasets))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="also write a single-file offline copy to dashboard/dist/")
    ap.add_argument("--dataset", choices=["all", "fashion", "cifar10"], default="all")
    a = ap.parse_args()
    build()
    if a.offline:
        offline(["cifar10", "fashion"] if a.dataset == "all" else [a.dataset])
