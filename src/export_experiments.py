"""
Export every run for the dashboard's Test Results and Model & Optimizer Comparison sections.

Reads   dashboard/data/<dataset>/data.json            historical sweep (written by export_dashboard.py)
        models/<dataset>/<model>-<optimizer>.keras    historical models, for full-test predictions
        experiments/runs/<study>/<dataset>/...         controlled runs (src/experiments.py)
Writes  dashboard/data/<dataset>/experiments.json     small index: every run, metrics, status (no curves, no predictions)
        dashboard/data/<dataset>/runs/<run>.json      one file per run, loaded only when that run is opened:
                                                      full test-set predictions and confidences keyed by official
                                                      test index, confusion matrix, per-epoch history
        results/controlled_summary.csv                mean, standard deviation and seeds per configuration

Stable sample IDs: the official test-set index (0 to 9999, keras.datasets / official file order).
The historical CIFAR-10 sweep was trained and evaluated on a JPEG copy of CIFAR-10 stored in a different
order; its predictions are mapped to official indices with experiments/splits/cifar10_jpeg_to_official.json
(each JPEG matched to its nearest official image, 10,000 of 10,000 unique). Historical predictions come from
the float16-rounded weights the browser runs, exactly as in data.json.

Usage:
  python src/export_experiments.py --cifar-dir data/cifar-10-batches-py --fashion-dir data/fashion \
         --jpeg-dir data/CIFAR-10-images
  (without --cifar-dir / --fashion-dir the official files are downloaded through keras.datasets)
"""
import argparse
import base64
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import EXP, EFFNET, SMALL, STUDIES, confusion, load_official, macro_f1  # noqa: E402
from train import DATASETS, ROOT, load_data  # noqa: E402

PLANNED = {
    # study: [(model, optimizer, seeds)]
    "full": [("mlp", "adam", [1, 2, 3]), ("cnn", "adam", [1, 2, 3]), ("cnn", "adam-e10", [1, 2, 3]),
             ("effnet", "adam", [1, 2, 3]), ("effnet", "adamw", [1, 2, 3]), ("effnet", "nadam", [1, 2, 3])],
    "budget": [("mlp", "adam", [1, 2, 3]), ("cnn", "adam", [1, 2, 3]),
               ("effnet", "adam", [1, 2, 3]), ("effnet", "adamw", [1, 2, 3]), ("effnet", "nadam", [1, 2, 3])],
}


def b64u8(a):
    return base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode()


def jpeg_mapping(data_dir, jpeg_dir):
    """Official test index for each image of the JPEG copy (class folders, sorted file names)."""
    path = EXP / "splits" / "cifar10_jpeg_to_official.json"
    if path.exists():
        return np.array(json.loads(path.read_text())["official_index"])
    from PIL import Image
    _, _, xte, yte = load_official("cifar10", "files" if data_dir else "keras", data_dir)
    x = xte.reshape(len(xte), -1).astype(np.float32)
    out = []
    for c, cls in enumerate(DATASETS["cifar10"]["classes"]):
        cand = np.where(yte == c)[0]
        for f in sorted((Path(jpeg_dir) / "test" / cls).iterdir()):
            im = np.asarray(Image.open(f).convert("RGB"), np.float32).ravel()
            out.append(int(cand[np.abs(x[cand] - im).mean(1).argmin()]))
    assert len(set(out)) == len(out) == 10000, "JPEG to official mapping is not one-to-one"
    path.write_text(json.dumps({"note": "position k = k-th image of github.com/YoongiKim/CIFAR-10-images test/ "
                                        "(class folders in label order, files sorted); value = official test index",
                                "official_index": out}))
    return np.array(out)


def historical(ds, args, D, yte_official, out_dir):
    """Full-test predictions of the 16 historical models, from float16-rounded weights."""
    if ds == "cifar10":
        _, _, xte, yte = load_data(ds, "files", args.jpeg_dir)
        to_off = jpeg_mapping(args.cifar_dir, args.jpeg_dir)
    else:
        _, _, xte, yte = load_data(ds, "files" if args.fashion_dir else "keras", args.fashion_dir)
        to_off = np.arange(len(yte))
    assert np.array_equal(yte_official[to_off], yte), "labels disagree after mapping"
    xf = (xte / 255.0).astype("float32")
    runs = []
    for mk in ("cnn", "mlp"):
        for o in D["models"][mk]["opts"]:
            model = keras.models.load_model(ROOT / "models" / ds / f"{mk}-{o['slug']}.keras")
            for layer in model.layers:
                if layer.weights:
                    w, b = layer.get_weights()
                    layer.set_weights([w.astype("float16").astype("float32"), b.astype("float16").astype("float32")])
            prob = model.predict(xf, verbose=0)
            pred = np.empty(10000, np.int64)
            conf = np.empty(10000)
            pred[to_off], conf[to_off] = prob.argmax(1), prob.max(1)
            acc = float((pred == yte_official).mean())
            cm = confusion(yte_official, pred)
            assert abs(acc - o["acc"]) < 6e-5 and cm.tolist() == o["cm"], f"{mk} {o['slug']} differs from data.json"
            pool_pred = [int(pred[i]) for i in D["pool_ids"]]
            assert pool_pred == o["pp"], f"{mk} {o['slug']} pool predictions differ from data.json"
            f1, per = macro_f1(yte_official, pred)
            key = f"historical__{mk}-{o['slug']}__s42"
            h = o["hist"]
            history = [{"epoch": e + 1, "phase": "train", "loss": h["loss"][e], "accuracy": h["accuracy"][e],
                        "val_loss": h["val_loss"][e], "val_accuracy": h["val_accuracy"][e]} for e in range(len(h["loss"]))]
            write_run(out_dir, key, pred, conf, cm, acc, f1, per, history)
            runs.append({"key": key, "study": "historical", "model": mk, "opt": o["slug"], "opt_name": o["name"],
                         "seed": 42, "status": "complete", "acc": round(acc, 6), "f1": round(f1, 6),
                         "epochs_run": len(history), "best_epoch": None, "n_train": 45000 if ds == "cifar10" else 54000,
                         "input": D["shape"][0], "selection": "last epoch (no checkpoint selection)",
                         "browser": mk in ("cnn", "mlp")})
            print(f"historical {mk} {o['slug']}: acc {acc:.4f} macro-F1 {f1:.4f}")
    return runs


def write_run(out_dir, key, pred, conf, cm, acc, f1, per, history):
    assert cm.sum() == 10000 and len(pred) == 10000
    data = {"key": key, "preds": b64u8(pred), "conf": b64u8(np.round(np.asarray(conf) * 255)),
            "acc": round(acc, 6), "f1": round(f1, 6), "per_class_f1": [round(float(v), 4) for v in per],
            "cm": cm.tolist(), "history": history}
    (out_dir / f"{key}.json").write_text(json.dumps(data, separators=(",", ":")))


def controlled(ds, yte, out_dir):
    runs = []
    for meta in sorted((EXP / "runs").glob(f"*/{ds}/*/seed-*/run.json")):
        info = json.loads(meta.read_text())
        if info["phase"] == "warmup" or info.get("smoke") or info["status"] != "evaluated":
            continue
        hist = json.loads((meta.parent / "history.json").read_text())
        variants = [("", info["optimizer"], hist, info.get("best_epoch"))]
        if info.get("snapshot_epoch"):
            se = info["snapshot_epoch"]
            variants.append((f"epoch{se}", f"{info['optimizer']}-e{se}", hist[:se], se))
        for tag, opt, history, best in variants:
            z = np.load(meta.parent / tag / "test_predictions.npz")
            m = json.loads((meta.parent / tag / "metrics.json").read_text())
            assert np.array_equal(z["ids"], np.arange(10000)) and np.array_equal(z["y_true"], yte)
            pred = z["y_pred"].astype(np.int64)
            prob = z["prob"].astype(np.float32)
            # probabilities are stored as float16, so an argmax can land on an equal (tied) value; the stored
            # prediction must still be one of the largest stored probabilities
            assert (prob[np.arange(10000), pred] >= prob.max(1)).all()
            acc = float((pred == yte).mean())
            f1, per = macro_f1(yte, pred)
            cm = confusion(yte, pred)
            assert abs(acc - m["test_accuracy"]) < 1e-6 and abs(f1 - m["test_macro_f1"]) < 1e-6 and cm.tolist() == m["confusion"]
            key = f"{info['study']}__{info['model']}-{opt}__s{info['seed']}"
            write_run(out_dir, key, pred, prob[np.arange(10000), pred], cm, acc, f1, per, history)
            r = info["recipe"]
            runs.append({"key": key, "study": info["study"], "model": info["model"], "opt": opt,
                         "seed": info["seed"], "status": "complete", "acc": round(acc, 6), "f1": round(f1, 6),
                         # epochs are counted globally: for EfficientNet the two warm-up epochs come first
                         "epochs_run": len(history), "best_epoch": best,
                         "best_val_f1": None if tag else info.get("best_val_macro_f1"),
                         "n_train": info["n_train"], "n_val": info["n_val"],
                         "input": r.get("input_size", DATASETS[ds]["shape"][0]),
                         "selection": f"fixed epoch {best}" if tag else "best validation macro-F1",
                         "warmup_epochs": info.get("warmup_epochs"), "train_seconds": info["train_seconds"],
                         "browser": False})
    return runs


def planned(runs):
    have = {(r["study"], r["model"], r["opt"], r["seed"]) for r in runs}
    out = []
    for study, items in PLANNED.items():
        for model, opt, seeds in items:
            for s in seeds:
                if (study, model, opt, s) not in have:
                    out.append({"study": study, "model": model, "opt": opt, "seed": s, "status": "not_run"})
    return out


def summarize(ds, runs, rows):
    groups = {}
    for r in runs:
        groups.setdefault((r["study"], r["model"], r["opt"]), []).append(r)
    for (study, model, opt), rs in sorted(groups.items()):
        a, f = np.array([r["acc"] for r in rs]), np.array([r["f1"] for r in rs])
        rows.append({"dataset": ds, "study": study, "model": model, "optimizer": opt, "seeds": len(rs),
                     "acc_mean": round(a.mean(), 4), "acc_sd": round(a.std(ddof=1), 4) if len(rs) > 1 else "",
                     "macro_f1_mean": round(f.mean(), 4), "macro_f1_sd": round(f.std(ddof=1), 4) if len(rs) > 1 else "",
                     "per_seed_acc": " ".join(f"s{r['seed']}={r['acc']:.4f}" for r in sorted(rs, key=lambda r: r["seed"]))})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), nargs="+", default=list(DATASETS))
    ap.add_argument("--cifar-dir", help="official cifar-10-batches-py folder (default: keras.datasets download)")
    ap.add_argument("--fashion-dir", help="official Fashion-MNIST .gz folder (default: keras.datasets download)")
    ap.add_argument("--jpeg-dir", default="data/CIFAR-10-images", help="JPEG copy used by the historical CIFAR-10 sweep")
    args = ap.parse_args()
    rows = []
    for ds in args.dataset:
        out_dir = ROOT / "dashboard" / "data" / ds / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        D = json.loads((ROOT / "dashboard" / "data" / ds / "data.json").read_text())
        ddir = args.cifar_dir if ds == "cifar10" else args.fashion_dir
        _, _, _, yte = load_official(ds, "files" if ddir else "keras", ddir)
        to_off = jpeg_mapping(args.cifar_dir, args.jpeg_dir) if ds == "cifar10" else np.arange(10000)
        D["pool_ids"] = [int(to_off[p["i"]]) for p in D["pool"]]
        assert all(yte[i] == p["t"] for i, p in zip(D["pool_ids"], D["pool"]))
        runs = historical(ds, args, D, yte, out_dir) + controlled(ds, yte, out_dir)
        keep = {r["key"] for r in runs}
        for f in out_dir.glob("*.json"):
            if f.stem not in keep:
                f.unlink()
        warm = {}
        for meta in sorted((EXP / "runs").glob(f"*/{ds}/effnet-warmup/seed-*/run.json")):
            w = json.loads(meta.read_text())
            if not w.get("smoke") and w["status"] == "trained":
                warm[f"{w['study']}:{w['seed']}"] = {"sha256": w["warmup_weights_sha256"][:16],
                                                     "input_range": w.get("backbone_input_range")}
        index = {
            "dataset": ds, "sample_id": "official test-set index (0-9999)", "n_test": 10000,
            "pool_ids": D["pool_ids"], "labels": b64u8(yte),
            "studies": {
                "historical": {"epochs": 10, "seeds": [42], "split": "last 10% of a shuffled training set",
                               "test": "JPEG copy of the test set" if ds == "cifar10" else "official test set"},
                "full": {"split": "fixed stratified 90/10 split", "max_epochs_small": SMALL["max_epochs"],
                         "patience": SMALL["patience"], "effnet_input": STUDIES["full"]["effnet_size"],
                         "test": "official test set"},
                "budget": {"split": "fixed stratified subset: 300 train + 150 validation images per class",
                           "effnet_input": STUDIES["budget"]["effnet_size"], "test": "official test set"},
            },
            "effnet": {k: EFFNET[k] for k in ("warm_epochs", "warm_lr", "fine_epochs", "learning_rate", "weight_decay",
                                              "dropout", "patience", "batch_size")},
            "warmups": warm,
            "runs": runs, "planned": planned(runs),
        }
        (ROOT / "dashboard" / "data" / ds / "experiments.json").write_text(json.dumps(index, separators=(",", ":")))
        summarize(ds, [r for r in runs], rows)
        print(f"{ds}: {len(runs)} runs exported, {len(index['planned'])} planned runs without artifacts")
    with open(ROOT / "results" / "controlled_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
