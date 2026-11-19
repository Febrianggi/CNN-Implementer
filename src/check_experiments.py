"""
Independent checks of the controlled experiments and of what the dashboard shows.

  1. Splits: train and validation are disjoint, stratified, and the budget subset sits inside them.
  2. Run identity: every run's recorded split hash matches the split file; every EfficientNet branch of a
     (study, dataset, seed) starts from the same warm-up checkpoint (weights hash and file hash), with the same
     split; BatchNorm statistics did not move; the backbone saw pixels in [0, 255].
  3. Metrics: accuracy, macro-F1 and the confusion matrix are recomputed from the saved predictions and compared
     with metrics.json; 10,000 unique sample IDs; the confusion matrix sums to 10,000.
  4. Dashboard files: every exported run decodes to the same predictions and metrics; the gallery pool has
     1,000 unique official IDs whose labels match; historical runs agree with data.json.

Usage:  python src/check_experiments.py [--cifar-dir ...] [--fashion-dir ...]
Exits with status 1 if any check fails.
"""
import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import EXP, STUDIES, _sha_list, confusion, fingerprint, load_official, macro_f1  # noqa: E402
from train import DATASETS, ROOT  # noqa: E402

FAIL = []


def check(ok, msg):
    if not ok:
        FAIL.append(msg)
    return ok


def u8(s):
    return np.frombuffer(base64.b64decode(s), np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar-dir")
    ap.add_argument("--fashion-dir")
    args = ap.parse_args()
    for ds in DATASETS:
        ddir = args.cifar_dir if ds == "cifar10" else args.fashion_dir
        xtr, ytr, xte, yte = load_official(ds, "files" if ddir else "keras", ddir)
        split = json.loads((EXP / "splits" / f"{ds}.json").read_text(encoding="utf-8"))
        val = np.array(split["val"])
        train = np.setdiff1d(np.arange(len(ytr)), val)
        check(split["train_fingerprint"] == fingerprint(xtr, ytr), f"{ds}: split made from other data")
        check(len(np.intersect1d(train, val)) == 0, f"{ds}: train/val overlap")
        cv = np.bincount(ytr[val], minlength=10)
        check(cv.max() - cv.min() <= 1, f"{ds}: validation not stratified {cv}")
        bt, bv = np.array(split["budget_train"]), np.array(split["budget_val"])
        check(np.isin(bt, train).all() and np.isin(bv, val).all(), f"{ds}: budget subset leaves the split")
        check((np.bincount(ytr[bt], minlength=10) == STUDIES["budget"]["train_per_class"]).all(), f"{ds}: budget train counts")
        check((np.bincount(ytr[bv], minlength=10) == STUDIES["budget"]["val_per_class"]).all(), f"{ds}: budget val counts")
        want = {"full": (_sha_list(train), _sha_list(val)), "budget": (_sha_list(bt), _sha_list(bv))}
        print(f"{ds}: split train {len(train)} / val {len(val)} (val per class {cv.min()}-{cv.max()}), "
              f"budget {len(bt)} / {len(bv)}, test {len(yte)}")

        runs = []
        for meta in sorted((EXP / "runs").glob(f"*/{ds}/*/seed-*/run.json")):
            info = json.loads(meta.read_text(encoding="utf-8"))
            if info.get("smoke") or info["status"] not in ("trained", "evaluated"):
                continue
            runs.append((meta.parent, info))
            check((info["split_sha256"]["train"], info["split_sha256"]["val"]) == want[info["study"]],
                  f"{meta.parent}: split hash differs from the split file")
            check(info["test_fingerprint"] == fingerprint(xte, yte), f"{meta.parent}: test data differs")
            check(info["class_order"] == DATASETS[ds]["classes"], f"{meta.parent}: class order")

        # EfficientNet: branches share the warm-up checkpoint
        groups = {}
        for path, info in runs:
            if info["model"] == "effnet":
                groups.setdefault((info["study"], info["seed"]), []).append((path, info))
        for (study, seed), items in sorted(groups.items()):
            warm = [i for p, i in items if i["phase"] == "warmup"]
            fine = [(p, i) for p, i in items if i["phase"] == "finetune"]
            if not check(len(warm) == 1, f"{ds} {study} seed {seed}: no warm-up"):
                continue
            w = warm[0]
            wfile = EXP / "runs" / study / ds / "effnet-warmup" / f"seed-{seed}" / "warmup.weights.h5"
            file_ok = (not wfile.exists()) or hashlib.sha256(wfile.read_bytes()).hexdigest() == w["warmup_file_sha256"]
            check(file_ok, f"{ds} {study} seed {seed}: warm-up file changed after training")
            rng = w.get("backbone_input_range") or [None, None]
            check(rng[0] is not None and rng[0] >= 0 and rng[1] <= 255, f"{ds} {study} seed {seed}: backbone input range {rng}")
            starts = {i["start_weights_sha256"] for p, i in fine}
            splits = {(i["split_sha256"]["train"], i["split_sha256"]["val"]) for p, i in fine}
            check(starts == {w["warmup_weights_sha256"]}, f"{ds} {study} seed {seed}: branches start from different weights")
            check(len(splits) == 1, f"{ds} {study} seed {seed}: branches use different splits")
            for p, i in fine:
                check(i.get("batchnorm_stats_unchanged") is True, f"{p}: BatchNorm statistics changed")
            print(f"  effnet {study} seed {seed}: {len(fine)} branches ({', '.join(sorted(i['optimizer'] for _, i in fine))}) "
                  f"start from warm-up {w['warmup_weights_sha256'][:12]}, input range {rng}, "
                  f"BatchNorm frozen: {all(i.get('batchnorm_stats_unchanged') for _, i in fine)}")

        # metrics recomputed from saved predictions, and the dashboard copy of them
        index = json.loads((ROOT / "dashboard" / "data" / ds / "experiments.json").read_text(encoding="utf-8"))
        by_key = {r["key"]: r for r in index["runs"]}
        n_checked = 0
        for path, info in runs:
            if info["phase"] == "warmup" or info["status"] != "evaluated":
                continue
            for tag, opt in [("", info["optimizer"])] + ([(f"epoch{info['snapshot_epoch']}", f"{info['optimizer']}-e{info['snapshot_epoch']}")] if info.get("snapshot_epoch") else []):
                z = np.load(path / tag / "test_predictions.npz")
                m = json.loads((path / tag / "metrics.json").read_text(encoding="utf-8"))
                ids, pred = z["ids"], z["y_pred"].astype(int)
                check(len(np.unique(ids)) == 10000 and np.array_equal(z["y_true"], yte[ids]), f"{path}/{tag}: ids or labels")
                acc, (f1, _) = float((pred == yte[ids]).mean()), macro_f1(yte[ids], pred)
                cm = confusion(yte[ids], pred)
                check(abs(acc - m["test_accuracy"]) < 1e-6 and abs(f1 - m["test_macro_f1"]) < 1e-6, f"{path}/{tag}: metrics")
                check(cm.sum() == 10000 and cm.tolist() == m["confusion"], f"{path}/{tag}: confusion")
                key = f"{info['study']}__{info['model']}-{opt}__s{info['seed']}"
                r = by_key.get(key)
                if check(r is not None, f"{key}: not in experiments.json"):
                    check(abs(r["acc"] - acc) < 1e-6 and abs(r["f1"] - f1) < 1e-6, f"{key}: dashboard metrics")
                    f = ROOT / "dashboard" / "data" / ds / "runs" / f"{key}.json"
                    # EfficientNet seeds the dashboard does not show have no predictions file
                    if check(f.exists() == r.get("file", True), f"{key}: dashboard file present only when the dashboard uses it") and f.exists():
                        j = json.loads(f.read_text(encoding="utf-8"))
                        check(np.array_equal(u8(j["preds"]), pred) and j["cm"] == cm.tolist(), f"{key}: dashboard predictions")
                n_checked += 1
        # historical runs in the dashboard
        D = json.loads((ROOT / "dashboard" / "data" / ds / "data.json").read_text(encoding="utf-8"))
        pool = np.array(index["pool_ids"])
        check(len(set(pool.tolist())) == 1000 and all(yte[i] == p["t"] for i, p in zip(pool, D["pool"])), f"{ds}: gallery pool ids")
        check(np.array_equal(u8(index["labels"]), yte), f"{ds}: labels in experiments.json")
        for mk in ("cnn", "mlp"):
            for o in D["models"][mk]["opts"]:
                j = json.loads((ROOT / "dashboard" / "data" / ds / "runs" / f"historical__{mk}-{o['slug']}__s42.json").read_text(encoding="utf-8"))
                p = u8(j["preds"])
                check(p[pool].tolist() == o["pp"] and j["cm"] == o["cm"] and abs((p == yte).mean() - o["acc"]) < 6e-5,
                      f"{ds} historical {mk} {o['slug']}: differs from data.json")
        print(f"  {n_checked} evaluated checkpoints recomputed from saved predictions; 16 historical runs match data.json")
    print("FAILED:\n  " + "\n  ".join(FAIL) if FAIL else "all checks passed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
