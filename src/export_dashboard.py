"""
Export trained models into the static files the dashboard reads.

Reads   models/<dataset>/mlp.history.json and models/<dataset>/cnn-<optimizer>.keras (+ .history.json)
Writes  dashboard/data/<dataset>/data.json           gallery images, predictions, confusion matrices, histories
        dashboard/data/<dataset>/w-<optimizer>.json  CNN weights in float16, base64-encoded
        dashboard/data/summary.json                  MLP vs CNN numbers for every exported dataset

Weights are rounded to float16 to keep each file under 1 MB. All numbers in data.json are
recomputed from the rounded weights, so the browser and this script agree exactly.

Usage:
  python src/export_dashboard.py --dataset cifar10
  python src/export_dashboard.py --dataset fashion --source files --data-dir data/fashion
"""
import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import DATASETS, OPTIMIZERS, ROOT, load_data, slug  # noqa: E402

# Text shown in the optimizer card. The Indonesian rows follow the course slide.
META = {
    "SGD": {
        "cfg": "learning rate 0,01 (bawaan), tanpa momentum", "cfg_en": "learning rate 0.01 (default), no momentum",
        "ch": "Pembaruan sederhana berdasarkan gradien", "ch_en": "Simple updates based on the gradient",
        "use": "Memahami dasar optimisasi dan membuat baseline", "use_en": "Understanding optimization basics and building a baseline"},
    "RMSprop": {
        "cfg": "learning rate 0,001 (bawaan), rho 0,9", "cfg_en": "learning rate 0.001 (default), rho 0.9",
        "ch": "Menyesuaikan learning rate berdasarkan riwayat kuadrat gradien", "ch_en": "Adapts the learning rate using a running average of squared gradients",
        "use": "Alternatif untuk pelatihan jaringan saraf", "use_en": "An alternative for training neural networks"},
    "Adam": {
        "cfg": "learning rate 0,001 (bawaan), beta 0,9 dan 0,999", "cfg_en": "learning rate 0.001 (default), beta 0.9 and 0.999",
        "ch": "Menggabungkan momentum dan penskalaan adaptif dari gradien", "ch_en": "Combines momentum with adaptive scaling of the gradient",
        "use": "Pilihan awal yang praktis untuk CNN", "use_en": "A practical first choice for CNNs"},
    "AdamW": {
        "cfg": "learning rate 0,001 (bawaan), weight decay 0,004", "cfg_en": "learning rate 0.001 (default), weight decay 0.004",
        "ch": "Adam dengan decoupled weight decay", "ch_en": "Adam with decoupled weight decay",
        "use": "Ketika ingin menambahkan regularisasi bobot", "use_en": "When you want to add weight regularization"},
    "Adagrad": {
        "cfg": "learning rate 0,001 (bawaan)", "cfg_en": "learning rate 0.001 (default)",
        "ch": "Menyesuaikan langkah berdasarkan akumulasi kuadrat gradien", "ch_en": "Adapts the step size using accumulated squared gradients",
        "use": "Masalah dengan fitur atau gradien yang jarang muncul", "use_en": "Problems with features or gradients that appear rarely"},
    "Nadam": {
        "cfg": "learning rate 0,001 (bawaan)", "cfg_en": "learning rate 0.001 (default)",
        "ch": "Adam dengan momentum bergaya Nesterov", "ch_en": "Adam with Nesterov-style momentum",
        "use": "Alternatif Adam untuk eksperimen", "use_en": "An Adam alternative for experiments"},
    "Adadelta": {
        "cfg": "learning rate 0,001 (bawaan), rho 0,95", "cfg_en": "learning rate 0.001 (default), rho 0.95",
        "ch": "Metode adaptif yang membatasi akumulasi gradien Adagrad", "ch_en": "An adaptive method that limits Adagrad's gradient accumulation",
        "use": "Eksperimen optimisasi adaptif", "use_en": "Experiments with adaptive optimization"},
    "SGD momentum": {
        "cfg": "learning rate 0,05, momentum 0,9, Nesterov", "cfg_en": "learning rate 0.05, momentum 0.9, Nesterov",
        "ch": "SGD yang membawa arah pembaruan sebelumnya (tambahan, di luar tabel materi)",
        "ch_en": "SGD that carries over the previous update direction (extra, not in the course table)",
        "use": "Pembanding untuk melihat apakah SGD bisa menyusul jika diberi momentum",
        "use_en": "A comparison to see whether SGD can catch up when given momentum"},
}


def b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), default="cifar10")
    ap.add_argument("--source", choices=["keras", "files"], default="keras")
    ap.add_argument("--data-dir", help="folder for --source files")
    ap.add_argument("--per-class", type=int, default=100, help="test images per class in the gallery")
    args = ap.parse_args()

    info = DATASETS[args.dataset]
    models_dir = ROOT / "models" / args.dataset
    out = ROOT / "dashboard" / "data" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    _, _, xte, yte = load_data(args.dataset, args.source, args.data_dir)
    xf = (xte / 255.0).astype("float32")

    # Fixed gallery pool: the same test images for every optimizer.
    rng = np.random.default_rng(42)
    pool = np.concatenate([np.sort(rng.choice(np.where(yte == c)[0], args.per_class, replace=False))
                           for c in range(10)])
    size, cols = info["shape"][0], 40
    rows = int(np.ceil(len(pool) / cols))
    atlas = np.zeros((rows * size, cols * size, 3), np.uint8)
    for k, i in enumerate(pool):
        r, c = divmod(k, cols)
        atlas[r * size:(r + 1) * size, c * size:(c + 1) * size] = xte[i]  # grayscale broadcasts to RGB
    buf = io.BytesIO()
    Image.fromarray(atlas).save(buf, "PNG", optimize=True)

    opts, pool_ok, params = [], np.zeros(len(pool)), 0
    for name in OPTIMIZERS:
        file = f"cnn-{slug(name)}"
        model = keras.models.load_model(models_dir / f"{file}.keras")
        hist = json.load(open(models_dir / f"{file}.history.json"))
        params = model.count_params()
        weights = {}
        for layer in ["conv1", "conv2", "dense", "output"]:
            w, b = model.get_layer(layer).get_weights()
            w16, b16 = w.astype("float16"), b.astype("float16")
            model.get_layer(layer).set_weights([w16.astype("float32"), b16.astype("float32")])
            weights[layer] = {"w": b64(w16, "<f2"), "b": b64(b16, "<f2"), "shape": list(w.shape)}

        prob = model.predict(xf, verbose=0)
        pred = prob.argmax(1)
        cm = np.zeros((10, 10), int)
        for t, p in zip(yte, pred):
            cm[t, p] += 1
        loss = float(-np.mean(np.log(np.clip(prob[np.arange(len(yte)), yte], 1e-7, 1))))
        pp = prob[pool]
        pool_ok += pp.argmax(1) == yte[pool]

        json.dump({"weights": weights, "ref": pp[:4].round(6).tolist()},
                  open(out / f"w-{slug(name)}.json", "w"), separators=(",", ":"))
        opts.append({"name": name, "slug": slug(name), **META[name],
                     "acc": round(float((pred == yte).mean()), 4), "loss": round(loss, 4),
                     "time": hist["time"], "hist": hist["hist"], "cm": cm.tolist(),
                     "pp": pp.argmax(1).tolist(), "pc": [round(float(x), 3) for x in pp.max(1)]})
        print(f"{name:13s} test accuracy {opts[-1]['acc']:.4f}")

    # Strip images: per class, an image most optimizers get right, preferring Adam's most confident one.
    adam = next(o for o in opts if o["name"] == "Adam")
    demo = []
    for c in range(10):
        ks = [k for k in range(len(pool)) if yte[pool[k]] == c]
        demo.append(max(ks, key=lambda k: (pool_ok[k], adam["pc"][k] if adam["pp"][k] == c else 0)))

    mlp = json.load(open(models_dir / "mlp.history.json"))
    data = {"key": args.dataset, "title": info["title"], "shape": list(info["shape"]), "classes": info["classes"],
            "atlas": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), "cols": cols,
            "pool": [{"i": int(i), "t": int(yte[i])} for i in pool], "demo": demo, "opts": opts, "params": params,
            "mlp": {"params": mlp["params"], "acc": mlp["test_accuracy"], "time": mlp["time"]}}
    json.dump(data, open(out / "data.json", "w"), separators=(",", ":"))

    summary_path = ROOT / "dashboard" / "data" / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    summary[args.dataset] = {"title": info["title"], "shape": list(info["shape"]),
                             "mlp": data["mlp"], "cnn": {"params": params, "acc": adam["acc"]}}
    json.dump(dict(sorted(summary.items())), open(summary_path, "w"), indent=1)
    print(f"wrote {out} and {summary_path}")


if __name__ == "__main__":
    main()
