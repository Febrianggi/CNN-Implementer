"""Build the dashboard pages from these sources.

  head.html + base.css + body.html + app.js  ->  ../index.id.html (Indonesian source text)
  the same page with the EN pairs below      ->  ../index.html    (English, default)

Run from anywhere:  python dashboard/src/build.py
Every Indonesian string in app.js / body.html that the visitor can see needs an (ID, EN) pair here;
the build stops if a pair no longer matches the source, and prints Indonesian-looking words left on
the English page.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent

P = [
('contoh uji di galeri, ${CLS.length} kelas, input ${g.H}', 'test examples in the gallery, ${CLS.length} classes, input ${g.H}'),
('aria-label="Tutup menu samping" title="Tutup menu samping"', 'aria-label="Close the side menu" title="Close the side menu"'),
('"Tutup menu samping":"Buka menu samping"', '"Close the side menu":"Open the side menu"'),
('aria-label="Menu samping"', 'aria-label="Side menu"'),
('<span class="lbl">Perbandingan model &amp; optimizer</span>', '<span class="lbl">Model &amp; optimizer comparison</span>'),
('>Mode gelap<', '>Dark mode<'),
('>Optimizer CNN<', '>CNN optimizer<'),
('Alur ini selalu menjalankan CNN historis (10 epoch, seed 42) di browser. Pilih model lain di bagian hasil dan perbandingan.', 'This flow always runs the historical CNN (10 epochs, seed 42) in your browser. Choose other models in the results and comparison sections.'),
('Pilih model A, dan bila perlu model B, untuk melihat prediksi keduanya pada citra uji yang sama. Galeri memakai kumpulan tetap 1.000 citra uji (100 per kelas, dipilih acak, dengan ID uji resmi); akurasi dan macro-F1 selalu dihitung dari 10.000 citra uji.', 'Choose model A, and optionally model B, to see both predictions on the same test images. The gallery uses a fixed set of 1,000 test images (100 per class, chosen at random, with official test IDs); accuracy and macro-F1 are always computed on all 10,000 test images.'),
('>Status A<', '>A status<'),
('>A dan B<', '>A and B<'),
('>Keduanya benar<', '>Both right<'),
('>Hanya A benar<', '>Only A right<'),
('>Hanya B benar<', '>Only B right<'),
('>Keduanya salah<', '>Both wrong<'),
('<h2>Perbandingan model &amp; optimizer</h2>', '<h2>Model &amp; optimizer comparison</h2>'),
('MLP, CNN, dan EfficientNetV2B0 dari eksperimen terkontrol, lalu sapuan optimizer historis. Setiap angka menyebut model, optimizer, seed, dan anggaran data yang menghasilkannya.', 'MLP, CNN, and EfficientNetV2B0 from the controlled experiments, then the historical optimizer sweep. Every number names the model, optimizer, seed, and data budget that produced it.'),
('<h3>Model dan optimizer, per seed</h3>', '<h3>Models and optimizers, per seed</h3>'),
('aria-label="Eksperimen"', 'aria-label="Experiment"'),
('aria-label="Metrik"', 'aria-label="Metric"'),
('>Split penuh<', '>Full split<'),
('>Subset kecil<', '>Small subset<'),
('<button type="button" data-v="acc" aria-pressed="true">Akurasi uji</button>', '<button type="button" data-v="acc" aria-pressed="true">Test accuracy</button>'),
('Klik baris untuk melihat kurva, akurasi per kelas, dan confusion matrix run tersebut. Δ adalah selisih berpasangan per seed (rata-rata ± SD) dalam poin persentase.', "Click a row to see that run's curves, per-class accuracy, and confusion matrix. Δ is the paired per-seed difference (mean ± SD) in percentage points."),
('>Hasil per seed<', '>Results per seed<'),
('<h3 id="runTitle">Kurva training</h3>', '<h3 id="runTitle">Training curves</h3>'),
('Akurasi train dihitung selama epoch dengan dropout dan augmentasi aktif; validasi dihitung di akhir epoch. Garis putus-putus vertikal menandai checkpoint yang dipakai untuk uji. Area abu-abu adalah warm-up.', 'Training accuracy is measured during the epoch with dropout and augmentation on; validation is measured at the end of the epoch. The dashed vertical line marks the checkpoint used for testing. The grey area is the warm-up.'),
('aria-label="Cabang fine-tuning"', 'aria-label="Fine-tuning branches"'),
('Dari 1.000 citra uji per kelas. Klik kelas untuk membuka galeri dengan run ini sebagai model A.', 'Out of 1,000 test images per class. Click a class to open the gallery with this run as model A.'),
('<h3 id="histTitle">Sapuan optimizer historis</h3>', '<h3 id="histTitle">Historical optimizer sweep</h3>'),
('aria-label="Model historis"', 'aria-label="Historical model"'),
('Eksperimen awal: delapan optimizer, bobot awal yang sama (seed 42), 10 epoch, batch 128, validasi 10% terakhir, tanpa pemilihan checkpoint. Satu seed, jadi selisih kecil antar optimizer belum bisa dibedakan dari variasi acak. Klik optimizer untuk membuka kurvanya di atas.', 'The first experiment: eight optimizers, the same initial weights (seed 42), 10 epochs, batch 128, the last 10% held out for validation, no checkpoint selection. One seed, so small gaps between optimizers cannot be told apart from random variation. Click an optimizer to open its curves above.'),
('Sapuan historis dilatih di CPU (seed 42); CIFAR-10 historis memakai salinan berkas JPG, dan prediksinya dipetakan ke ID uji resmi. Eksperimen terkontrol memakai berkas resmi CIFAR-10 dan Fashion-MNIST. CNN di alur jaringan memakai bobot 16-bit; angka sapuan historis dihitung ulang dari bobot 16-bit tersebut, sedangkan run terkontrol dievaluasi dari checkpoint 32-bit di Python.', 'The historical sweep was trained on a CPU (seed 42); the historical CIFAR-10 runs used a JPG copy of the files, and their predictions are mapped to official test IDs. The controlled experiments use the official CIFAR-10 and Fashion-MNIST files. The CNN in the network flow uses 16-bit weights; the historical sweep numbers are recomputed from those 16-bit weights, while controlled runs are evaluated from 32-bit checkpoints in Python.'),
('<dl><dt>Asli</dt><dd>${CLS[it.t]}</dd><dt>${L.cardA}</dt>', '<dl><dt>True</dt><dd>${CLS[it.t]}</dd><dt>${L.cardA}</dt>'),
('`Akurasi uji ${HM}`', '`Test accuracy ${HM}`'),
('Alur ini menjalankan CNN historis di browser, dengan delapan optimizer dari bobot awal yang sama. Hasil MLP, CNN, dan EfficientNetV2B0 ada di bagian hasil dan perbandingan.', 'This flow runs the historical CNN in your browser, with eight optimizers from the same initial weights. MLP, CNN, and EfficientNetV2B0 results are in the results and comparison sections.'),
('<div>Peringkat<b>${rank} dari ${OPTS().length}</b>', '<div>Rank<b>${rank} of ${OPTS().length}</b>'),
('<html lang="id">', '<html lang="en">'),
('<span>Analisis Data Citra</span>', '<span>Image Data Analysis</span>'),
('aria-label="Bagian halaman"', 'aria-label="Page sections"'),
('href="index.html" lang="en" hreflang="en"', 'href="index.id.html" lang="id" hreflang="id"'),
('<span>English</span><b class="lang-short">EN</b>', '<span>Bahasa Indonesia</span><b class="lang-short">ID</b>'),
('Model berjalan di browser. Gambar yang kamu unggah tidak dikirim ke mana pun.', 'The model runs in your browser. Images you upload are not sent anywhere.'),
('Memuat data uji dan hasil training…', 'Loading test data and training results…'),
('Bagaimana CNN membaca gambar 32 × 32', 'How a CNN reads a 32 × 32 image'),
('aria-label="Pilih gambar untuk diperiksa"', 'aria-label="Choose an image to inspect"'),
('Unggah JPG atau PNG, atau tarik gambar ke kotak +. Gambar dipotong persegi di tengah lalu diperkecil ke 32 × 32.', 'Upload a JPG or PNG, or drag an image onto the + box. It is center-cropped to a square and resized to 32 × 32.'),
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
('<h3>Ringkasan per optimizer</h3>', '<h3>Summary per optimizer</h3>'),
('Diurutkan dari akurasi uji tertinggi. Akurasi dan loss dihitung pada 10.000 citra uji.', 'Sorted by test accuracy, highest first. Accuracy and loss are computed on 10,000 test images.'),
('<h3>Kurva validasi per epoch</h3>', '<h3>Validation curves per epoch</h3>'),
('Optimizer terpilih berwarna biru, sisanya abu-abu.', 'The selected optimizer is blue, the others are grey.'),
('aria-label="Pilih optimizer di grafik"', 'aria-label="Select an optimizer in the chart"'),
('id="pclsTitle">Akurasi per kelas<', 'id="pclsTitle">Per-class accuracy<'),
('Baris kelas asli, kolom kelas prediksi. Klik sel di luar diagonal untuk melihat contohnya.', 'Rows are true classes, columns are predicted classes. Click an off-diagonal cell to see examples.'),
('content:" (asli)"', 'content:" (true)"'),
('const LANG="id";', 'const LANG="en";'),
('`Memuat bobot ${optOf().name}…`', '`Loading ${optOf().name} weights…`'),
('`Bobot ${optOf().name} gagal dimuat. Pilih optimizer itu sekali lagi untuk mencoba ulang.`', '`Could not load the ${optOf().name} weights. Select that optimizer again to retry.`'),
('`${o.name}, akurasi uji ${pct(o.acc)}`', '`${o.name}, test accuracy ${pct(o.acc)}`'),
('<dt>Karakteristik</dt>', '<dt>Characteristics</dt>'),
('<dt>Kapan digunakan</dt>', '<dt>When to use</dt>'),
('<div>Akurasi uji<b>', '<div>Test accuracy<b>'),
('<div>Val terbaik<b>', '<div>Best val<b>'),
('<div>Selisih train−val<b>${gap>=0?"+":"−"}${nf(Math.abs(gap)*100,1)} poin</b></div><div>Waktu latih<b>${nf(o.time,0)} dtk</b>', '<div>Train−val gap<b>${gap>=0?"+":"−"}${nf(Math.abs(gap)*100,1)} pts</b></div><div>Training time<b>${nf(o.time,0)} s</b>'),
('<dt>Data uji</dt><dd>10.000 citra</dd>', '<dt>Test data</dt><dd>10,000 images</dd>'),
('"Periksa contoh "+', '"Inspect example "+'),
('"Unggahan "+(it.i+1)', '"Upload "+(it.i+1)'),
('"Periksa gambar unggahan "+', '"Inspect uploaded image "+'),
("</svg>Unggah';", "</svg>Upload';"),
('"Unggah gambar sendiri"', '"Upload your own image"'),
('"Berkas ini bukan gambar. Pilih JPG, PNG, atau WEBP."', '"This file is not an image. Choose a JPG, PNG, or WEBP file."'),
('msg("Membaca gambar…")', 'msg("Reading image…")'),
('"Gambar tidak bisa dibaca. Coba berkas lain atau simpan ulang sebagai JPG atau PNG."', '"The image could not be read. Try another file or re-save it as JPG or PNG."'),
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
('i<0?"Semua kelas":CLS[i]', 'i<0?"All classes":CLS[i]'),
('Tidak ada contoh untuk kombinasi filter ini. Ganti status, kelas, atau pembanding.', 'No examples match this combination of filters. Change the status, class, or comparison.'),
('"Tidak cocok"', '"No match"'),
('"Cocok"', '"Match"'),
('<th scope="col">Akurasi uji</th><th scope="col">Loss uji</th><th scope="col">Val terbaik</th><th scope="col">Train−val</th><th scope="col">Waktu</th>', '<th scope="col">Test accuracy</th><th scope="col">Test loss</th><th scope="col">Best val</th><th scope="col">Train−val</th><th scope="col">Time</th>'),
('<td>${nf(x.time,0)} dtk</td>', '<td>${nf(x.time,0)} s</td>'),
('`${x.name} akurasi uji ${pct(x.acc)}. Pilih optimizer ini`', '`${x.name} test accuracy ${pct(x.acc)}. Select this optimizer`'),
('"Akurasi validasi"', '"Validation accuracy"'),
('"Loss validasi"', '"Validation loss"'),
('`${CLS[i]} akurasi ${pct(a)}. Tampilkan di galeri`', '`${CLS[i]} accuracy ${pct(a)}. Show in gallery`'),
('${v} citra ${CLS[i]} diprediksi ${CLS[j]}. Tampilkan contoh', '${v} ${CLS[i]} images predicted as ${CLS[j]}. Show examples'),
('<th scope="col">Model</th><th scope="col">Parameter</th><th scope="col">Akurasi uji</th>', '<th scope="col">Model</th><th scope="col">Parameters</th><th scope="col">Test accuracy</th>'),
('"Data model gagal dimuat. Muat ulang halaman; kalau masih gagal, berkas data tidak ikut terunggah."', '"Model data could not be loaded. Reload the page. If it still fails, the data files are missing next to this page, or the page was opened as a local file instead of through a web server."'),
('<title>CNN Explorer: dataset dan optimizer</title>', '<title>CNN Explorer: datasets and optimizers</title>'),
('Pilih dataset dan optimizer, lalu lihat bagaimana bobot hasil latihannya membaca gambar yang sama.', 'Pick a dataset and an optimizer, then see how the trained weights read the same image.'),
('`Unggah JPG atau PNG, atau tarik gambar ke kotak +. Gambar dipotong persegi di tengah lalu diperkecil ke ${g.H} × ${g.H}`+(g.C===1?", diubah ke grayscale, dan (jika dicentang) dibalik warnanya.":".")', '`Upload a JPG or PNG, or drag an image onto the + box. It is center-cropped to a square and resized to ${g.H} × ${g.H}`+(g.C===1?", converted to grayscale, and (if checked) color-inverted.":".")'),
('> Balik warna gambar unggahan (Fashion-MNIST memakai benda terang di latar hitam)<', '> Invert uploaded images (Fashion-MNIST shows light items on a black background)<'),
('<small>32 × 32 berwarna, 10 objek</small>', '<small>32 × 32 color, 10 objects</small>'),
('<small>28 × 28 grayscale, 10 pakaian</small>', '<small>28 × 28 grayscale, 10 clothing items</small>'),
('Bobot optimizer lain dimuat saat kamu memintanya (sekitar 0,5 sampai 2 MB per optimizer).', 'Weights for the other optimizers load when you ask for them (about 0.5 to 2 MB each).'),
('`"${f.name}" diperkecil ke ${G().H} × ${G().H} dan diprediksi. Label aslinya tidak diketahui, jadi tidak ada status cocok.`', '`"${f.name}" was resized to ${G().H} × ${G().H} and classified. Its true label is unknown, so it has no match status.`'),
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
('m.textContent=`Memuat ${title}…`', 'm.textContent=`Loading ${title}…`'),
('`${title} gagal dimuat. Pilih dataset itu sekali lagi untuk mencoba ulang.`', '`${title} could not be loaded. Select that dataset again to retry.`'),
('</svg><span class="lbl">Alur jaringan</span></a>', '</svg><span class="lbl">Network flow</span></a>'),
('</svg><span class="lbl">Hasil uji</span></a>', '</svg><span class="lbl">Test results</span></a>'),
('<div><dt>Parameter ${MODEL()}</dt>', '<div><dt>Parameters ${MODEL()}</dt>'),
('<dt>Akurasi uji (${MODEL()} + ${o.name})</dt>', '<dt>Test accuracy (${MODEL()} + ${o.name})</dt>'),
('Label asli tidak diketahui</span><strong>Prediksi: ${CLS[top]}</strong><span style="color:var(--muted)">keyakinan ${pct(r.prob[top])}, ${on}', 'True label unknown</span><strong>Prediction: ${CLS[top]}</strong><span style="color:var(--muted)">confidence ${pct(r.prob[top])}, ${on}'),
('<strong>Prediksi: ${CLS[top]}</strong><span style="color:var(--muted)">asli ${CLS[S.cur.truth]}, keyakinan ${pct(r.prob[top])}, ${on}', '<strong>Prediction: ${CLS[top]}</strong><span style="color:var(--muted)">true ${CLS[S.cur.truth]}, confidence ${pct(r.prob[top])}, ${on}'),
('`Garis putus-putus: ${other.toUpperCase()} dengan optimizer yang sama.`', '`Dashed line: ${other.toUpperCase()} with the same optimizer.`'),
('<h3 id="accTitle">Akurasi uji</h3>', '<h3 id="accTitle">Test accuracy</h3>'),
('`Keduanya dilatih dengan ${o.name}, optimizer yang sedang dipilih. Dataset yang sedang dipilih ditandai biru.`', '`Both trained with ${o.name}, the selected optimizer. The selected dataset is marked in blue.`'),
('<th scope="row">Selisih CNN − MLP</th><td></td><td class="${b>=a?"pos":""}">${b>=a?"+":"−"}${nf(Math.abs(b-a)*100,2)} poin</td>', '<th scope="row">CNN − MLP</th><td></td><td class="${b>=a?"pos":""}">${b>=a?"+":"−"}${nf(Math.abs(b-a)*100,2)} pts</td>'),
('`Bagaimana ${MODEL()} membaca gambar ${g.H} × ${g.H}`', '`How ${S.model==="mlp"?"an":"a"} ${MODEL()} reads a ${g.H} × ${g.H} image`'),
]


def build():
    ID = ((HERE / "head.html").read_text(encoding="utf-8") + "<style>" + (HERE / "base.css").read_text(encoding="utf-8") + "</style>\n"
          + (HERE / "body.html").read_text(encoding="utf-8") + "\n<script>\n" + (HERE / "app.js").read_text(encoding="utf-8") + "</script>\n</body>\n</html>\n")
    ID = ID.replace("/*L:ID*/", "").replace("/*L:END*/", "") if False else ID
    missing = [old for old, new in P if old not in ID]
    assert not missing, "not found:\n" + "\n".join(m[:90] for m in missing)
    # strings of the run-based sections live in one object; swap it whole for the English one
    start, end = ID.index("/*L:ID*/"), ID.index("/*L:END*/") + len("/*L:END*/")
    en_obj = (HERE / "strings_en.js").read_text(encoding="utf-8").strip()
    keys = lambda js: re.findall(r"^\s*([A-Za-z]\w*):", js, flags=re.M)
    assert keys(ID[start:end]) == keys(en_obj), "strings_en.js keys differ from the L object in app.js"
    EN = ID[:start] + en_obj + ID[end:]
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


if __name__ == "__main__":
    build()
