"""
Controlled experiments: MLP, CNN and EfficientNetV2B0 on CIFAR-10 and Fashion-MNIST.

This is separate from src/train.py (the historical 10-epoch optimizer sweep, which stays as it is).
Every run here uses:
  * the official test set (10,000 images), only for the final evaluation;
  * one fixed, stratified train/validation split per dataset (experiments/splits/<dataset>.json),
    shared by every model, optimizer and seed;
  * paired seeds (seed s gives every model the same shuffling and dropout/augmentation streams);
  * checkpoint selection on validation macro-F1 (EarlyStopping + ModelCheckpoint on val_macro_f1).

Studies
  full    the whole training split (45,000 CIFAR-10 / 54,000 Fashion-MNIST images, 10% held out)
  budget  a fixed stratified subset of that split (300 train + 150 validation images per class),
          used to run EfficientNet on a CPU. Same test set.

Models
  mlp     Flatten > Dense 256 > Dense 128 > Dense 10          (same as src/train.py)
  cnn     Conv32 > pool > Conv64 > pool > Dense 128 > Dropout 0.3 > Dense 10   (same as src/train.py)
  effnet  EfficientNetV2B0, ImageNet weights, recipe from rayyanif/COVID-CT-Scans-Classification:
          resize > RandomFlip > RandomRotation(0.03) > backbone(training=False) > Dropout 0.4 > Dense 10.
          Warm-up: backbone frozen, Adam 1e-3, 2 epochs. Then three independent fine-tuning branches
          (Adam, AdamW wd 1e-5, Nadam; lr 6e-5, at most 23 epochs, patience 5) that all start from the
          same warm-up checkpoint with a fresh optimizer. BatchNorm statistics stay frozen because the
          backbone is always called with training=False.

Run layout (one folder per run identity; finished runs are never overwritten)
  experiments/runs/<study>/<dataset>/<model>-<optimizer>/seed-<seed>/
      run.json              identity, config, split hash, warm-up hash, status, timings
      history.json          per-epoch train/validation loss, accuracy, validation macro-F1, phase
      metrics.json          test accuracy, macro-F1, confusion matrix (written by `evaluate`)
      test_predictions.npz  ids, y_true, y_pred, probabilities (float16)
      best.weights.h5       selected checkpoint (not committed)

Usage
  python src/experiments.py run --dataset cifar10 --study full --models mlp cnn --seeds 1 2 3
  python src/experiments.py run --dataset fashion --study budget --models effnet --seeds 1 2 3
  python src/experiments.py evaluate --dataset cifar10 --study full     # test set, after training
  python src/experiments.py report                                      # tables in results/

  Data sources: --source keras (default, downloads the official files through keras.datasets) or
  --source files --data-dir <dir>, where <dir> holds the official cifar-10-batches-py files or the
  four Fashion-MNIST .gz files. Both are checked against their published MD5 sums.
  EfficientNet weights: --effnet-weights imagenet (default, keras.applications download) or a path to
  an .h5 file with the same layer names.
"""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras  # noqa: E402
from keras import layers  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import DATASETS, build_cnn, build_mlp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"

SPLIT_SEED = 2026
VAL_FRACTION = 0.1

STUDIES = {
    "full": {"train_per_class": None, "val_per_class": None, "effnet_size": 224},
    "budget": {"train_per_class": 300, "val_per_class": 150, "effnet_size": 96},
}

# Recipe for the small models: Adam with default settings, early stopping on validation macro-F1.
SMALL = {"batch_size": 128, "max_epochs": 40, "patience": 5, "learning_rate": 1e-3, "snapshot_epoch": 10}

# Recipe from the reference notebook (covid_ct_scans.ipynb, cells 4, 10, 12, 14).
EFFNET = {
    "batch_size": 16, "warm_epochs": 2, "warm_lr": 1e-3, "fine_epochs": 23, "learning_rate": 6e-5,
    "weight_decay": 1e-5, "dropout": 0.4, "rotation": 0.03, "horizontal_flip": True, "patience": 5,
}
FINE_OPTIMIZERS = {
    "adam": lambda: keras.optimizers.Adam(learning_rate=EFFNET["learning_rate"]),
    "adamw": lambda: keras.optimizers.AdamW(learning_rate=EFFNET["learning_rate"], weight_decay=EFFNET["weight_decay"]),
    "nadam": lambda: keras.optimizers.Nadam(learning_rate=EFFNET["learning_rate"]),
}
OPT_NAMES = {"adam": "Adam", "adamw": "AdamW", "nadam": "Nadam", "warmup": "Adam (warm-up)"}

MD5 = {
    "cifar10": {
        "data_batch_1": "c99cafc152244af753f735de768cd75f", "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
        "data_batch_3": "54ebc095f3ab1f0389bbae665268c751", "data_batch_4": "634d18415352ddfa80567beed471001a",
        "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb", "test_batch": "40351d587109b95175f43aff81a1287e",
    },
    "fashion": {
        "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
        "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
        "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
    },
}


# ----------------------------------------------------------------------------- data

def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_files(dataset, folder):
    for name, want in MD5[dataset].items():
        got = _md5(Path(folder) / name)
        if got != want:
            raise SystemExit(f"{folder}/{name}: MD5 {got}, expected {want}. Use the official files.")


def load_official(dataset, source="keras", data_dir=None):
    """uint8 images (N, H, W, C) and int labels, official train and test order."""
    if source == "keras":
        loader = keras.datasets.cifar10 if dataset == "cifar10" else keras.datasets.fashion_mnist
        (xtr, ytr), (xte, yte) = loader.load_data()
    elif dataset == "cifar10":
        _check_files(dataset, data_dir)  # verified before unpickling

        def batch(name):
            with open(Path(data_dir) / name, "rb") as f:
                d = pickle.load(f, encoding="bytes")
            return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[b"labels"])

        parts = [batch(f"data_batch_{i}") for i in range(1, 6)]
        xtr, ytr = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        xte, yte = batch("test_batch")
    else:
        _check_files(dataset, data_dir)

        def read(name, offset, shape):
            with gzip.open(Path(data_dir) / name, "rb") as f:
                return np.frombuffer(f.read(), np.uint8, offset=offset).reshape(shape)

        xtr = read("train-images-idx3-ubyte.gz", 16, (-1, 28, 28))
        ytr = read("train-labels-idx1-ubyte.gz", 8, (-1,))
        xte = read("t10k-images-idx3-ubyte.gz", 16, (-1, 28, 28))
        yte = read("t10k-labels-idx1-ubyte.gz", 8, (-1,))
    if dataset == "fashion":
        xtr, xte = xtr[..., None], xte[..., None]
    xtr, xte = np.ascontiguousarray(xtr, np.uint8), np.ascontiguousarray(xte, np.uint8)
    return xtr, np.asarray(ytr).ravel().astype(np.int64), xte, np.asarray(yte).ravel().astype(np.int64)


def fingerprint(x, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x).tobytes())
    h.update(np.ascontiguousarray(y, np.int64).tobytes())
    return h.hexdigest()


def _sha_list(ids):
    return hashlib.sha256(np.asarray(sorted(ids), np.int64).tobytes()).hexdigest()


def make_split(dataset, xtr, ytr):
    """Stratified validation split of the official training set, plus the fixed budget subset."""
    rng = np.random.default_rng(SPLIT_SEED)
    val, train = [], []
    for c in range(10):
        idx = np.where(ytr == c)[0]
        idx = idx[rng.permutation(len(idx))]
        n_val = int(round(len(idx) * VAL_FRACTION))
        val += idx[:n_val].tolist()
        train += idx[n_val:].tolist()
    budget = {"train": [], "val": []}
    for part, pool in (("train", np.array(train)), ("val", np.array(val))):
        per = STUDIES["budget"][f"{part}_per_class"]
        for c in range(10):
            idx = np.sort(pool[ytr[pool] == c])
            budget[part] += np.sort(rng.choice(idx, per, replace=False)).tolist()
    split = {
        "dataset": dataset, "split_seed": SPLIT_SEED, "val_fraction": VAL_FRACTION,
        "indices_refer_to": "official training set order (keras.datasets / official files)",
        "train_fingerprint": fingerprint(xtr, ytr),
        "n_train": len(train), "n_val": len(val),
        "val": sorted(val),
        "budget_train": sorted(budget["train"]), "budget_val": sorted(budget["val"]),
    }
    split["sha256"] = {"train": _sha_list(train), "val": _sha_list(val),
                       "budget_train": _sha_list(budget["train"]), "budget_val": _sha_list(budget["val"])}
    return split


def get_split(dataset, xtr, ytr):
    path = EXP / "splits" / f"{dataset}.json"
    if path.exists():
        split = json.loads(path.read_text())
        if split["train_fingerprint"] != fingerprint(xtr, ytr):
            raise SystemExit(f"{path} was made from different training data; refusing to continue.")
    else:
        split = make_split(dataset, xtr, ytr)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(split))
    val = np.array(split["val"])
    train = np.setdiff1d(np.arange(len(ytr)), val)
    assert _sha_list(train) == split["sha256"]["train"] and _sha_list(val) == split["sha256"]["val"]
    assert len(np.intersect1d(train, val)) == 0
    return split, train, val


def study_indices(study, split, train, val):
    if study == "full":
        return train, val
    return np.array(split["budget_train"]), np.array(split["budget_val"])


# ----------------------------------------------------------------------------- models

def load_h5_by_name(model, path):
    """Load an EfficientNetV2 weight file whose groups use keras.applications layer names."""
    import h5py
    n = 0
    with h5py.File(path, "r") as h:
        for layer in model.layers:
            if not layer.get_weights():
                continue
            g = h[layer.name]
            names = [x.decode() if isinstance(x, bytes) else x for x in g.attrs["weight_names"]]
            arrays = [np.array(g[x]) for x in names]
            if [a.shape for a in arrays] != [w.shape for w in layer.get_weights()]:
                raise SystemExit(f"{path}: shape mismatch in {layer.name}")
            layer.set_weights(arrays)
            n += 1
    return n


def force_inference(base):
    """Run the backbone's BatchNorm and Dropout layers in inference mode, always.

    The reference notebook (Keras 2.10) calls base(x, training=False), which keeps BatchNorm statistics
    and stochastic depth in inference mode while gamma/beta and conv weights still train. Keras 3 does
    not pass that nested training=False through model.fit (tested with Keras 3.15: moving_mean changed),
    so the flag is fixed per layer here. Weights stay trainable; only the mode is pinned.
    """
    n = 0
    for layer in base.layers:
        if isinstance(layer, (layers.BatchNormalization, layers.Dropout)):
            call = layer.call
            if isinstance(layer, layers.BatchNormalization):
                layer.call = lambda inputs, training=None, mask=None, _c=call: _c(inputs, training=False, mask=mask)
            else:
                layer.call = lambda inputs, training=None, _c=call: _c(inputs, training=False)
            n += 1
    return n


def build_effnet(input_shape, size, seed, weights="imagenet"):
    inp = keras.Input(input_shape, name="pixels_0_255")
    x = inp
    if input_shape[-1] == 1:  # Fashion-MNIST: grey to RGB by repeating the channel
        x = layers.Concatenate(name="gray_to_rgb")([x, x, x])
    x = layers.Resizing(size, size, interpolation="bilinear", name="resize")(x)
    if EFFNET["horizontal_flip"]:
        x = layers.RandomFlip("horizontal", seed=seed, name="flip")(x)
    x = layers.RandomRotation(EFFNET["rotation"], seed=seed, name="rotate")(x)
    base = keras.applications.EfficientNetV2B0(
        weights="imagenet" if weights == "imagenet" else None, include_top=False,
        include_preprocessing=True, input_shape=(size, size, 3), pooling="avg")
    if weights != "imagenet":
        load_h5_by_name(base, weights)
    force_inference(base)
    base.trainable = False
    x = base(x, training=False)  # plus force_inference: BatchNorm statistics never update
    x = layers.Dropout(EFFNET["dropout"], name="head_dropout")(x)
    out = layers.Dense(10, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="effnet"), base


def weights_sha(model):
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(np.ascontiguousarray(w, np.float32).tobytes())
    return h.hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def effnet_weight_source(weights):
    if weights == "imagenet":
        path = Path(keras.utils.get_file(
            "efficientnetv2-b0_notop.h5",
            "https://storage.googleapis.com/tensorflow/keras-applications/efficientnet_v2/efficientnetv2-b0_notop.h5",
            cache_subdir="models"))
        return {"weights": "imagenet (keras.applications)", "file": path.name, "sha256": file_sha(path)}
    return {"weights": "imagenet (file)", "file": Path(weights).name, "sha256": file_sha(weights)}


# ----------------------------------------------------------------------------- training helpers

def macro_f1(y, p, k=10):
    cm = confusion(y, p, k)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return float(f1.mean()), f1


def confusion(y, p, k=10):
    cm = np.zeros((k, k), np.int64)
    np.add.at(cm, (np.asarray(y), np.asarray(p)), 1)
    return cm


class ValidationF1(keras.callbacks.Callback):
    """Adds val_macro_f1 to the logs; must run before the checkpoint and early-stopping callbacks."""

    def __init__(self, xv, yv, batch_size):
        super().__init__()
        self.xv, self.yv, self.bs = xv, yv, batch_size

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.xv, batch_size=max(self.bs, 64), verbose=0).argmax(1)
        logs["val_macro_f1"] = macro_f1(self.yv, p)[0]


class Snapshot(keras.callbacks.Callback):
    def __init__(self, epoch, path):
        super().__init__()
        self.epoch, self.path = epoch, path

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.epoch:
            self.model.save_weights(self.path)


class Progress(keras.callbacks.Callback):
    """Writes history.json after every epoch so an interrupted run leaves a readable trace."""

    def __init__(self, run_dir, phase, offset, prior):
        super().__init__()
        self.run_dir, self.phase, self.offset, self.rows = run_dir, phase, offset, list(prior)
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": self.offset + epoch + 1, "phase": self.phase, "seconds": round(time.time() - self.t0, 1)}
        for k in ("loss", "accuracy", "val_loss", "val_accuracy", "val_macro_f1"):
            if k in logs:
                row[k] = round(float(logs[k]), 5)
        self.rows.append(row)
        self.t0 = time.time()
        (self.run_dir / "history.json").write_text(json.dumps(self.rows, indent=0))
        print(f"  epoch {row['epoch']:>2} [{self.phase}] loss {row['loss']:.4f} acc {row['accuracy']:.4f} "
              f"val_acc {row.get('val_accuracy', 0):.4f} val_f1 {row.get('val_macro_f1', 0):.4f} "
              f"({row['seconds']:.0f}s)", flush=True)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def run_dir(root, study, dataset, model, opt, seed):
    return root / study / dataset / f"{model}-{opt}" / f"seed-{seed}"


def claim(path, identity, force=False):
    """Create a fresh run folder. Finished runs are skipped; an unfinished one is kept aside."""
    meta = path / "run.json"
    if meta.exists():
        status = json.loads(meta.read_text()).get("status")
        if status in ("trained", "evaluated") and not force:
            print(f"skip {path.relative_to(ROOT)} ({status})")
            return None
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        aside = path.with_name(f"{path.name}.attempt-{stamp}")
        shutil.move(str(path), str(aside))
        print(f"kept previous attempt as {aside.name}")
    path.mkdir(parents=True)
    info = dict(identity, status="running", started=now())
    meta.write_text(json.dumps(info, indent=1))
    return info


def finish(path, info, **extra):
    info.update(extra, status="trained", finished=now())
    (path / "run.json").write_text(json.dumps(info, indent=1))


# ----------------------------------------------------------------------------- runs

def run_small(kind, ds, study, seed, data, ctx, root, force):
    xtr, ytr, xv, yv = data
    path = run_dir(root, study, ds, kind, "adam", seed)
    identity = dict(ctx, model=kind, optimizer="adam", seed=seed, phase="train",
                    recipe=dict(SMALL, input="pixels / 255", augmentation="none"))
    info = claim(path, identity, force)
    if info is None:
        return
    keras.utils.set_random_seed(seed)
    shape = DATASETS[ds]["shape"]
    model = (build_mlp if kind == "mlp" else build_cnn)(shape)
    info["init_weights_sha256"] = weights_sha(model)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=SMALL["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    x, v = (xtr / 255.0).astype("float32"), (xv / 255.0).astype("float32")
    best = path / "best.weights.h5"
    cbs = [ValidationF1(v, yv, SMALL["batch_size"]),
           keras.callbacks.ModelCheckpoint(str(best), monitor="val_macro_f1", mode="max",
                                           save_best_only=True, save_weights_only=True),
           keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=SMALL["patience"]),
           Snapshot(SMALL["snapshot_epoch"], path / f"epoch{SMALL['snapshot_epoch']}.weights.h5"),
           Progress(path, "train", 0, [])]
    t0 = time.time()
    model.fit(x, ytr, validation_data=(v, yv), epochs=SMALL["max_epochs"], batch_size=SMALL["batch_size"],
              shuffle=True, callbacks=cbs, verbose=0)
    hist = json.loads((path / "history.json").read_text())
    f1s = [r["val_macro_f1"] for r in hist]
    finish(path, info, params=model.count_params(), train_seconds=round(time.time() - t0, 1),
           epochs_run=len(hist), best_epoch=int(np.argmax(f1s)) + 1, best_val_macro_f1=round(max(f1s), 5),
           snapshot_epoch=SMALL["snapshot_epoch"] if len(hist) >= SMALL["snapshot_epoch"] else None)


def _effnet_model(ds, size, seed, weights):
    keras.utils.set_random_seed(seed)
    return build_effnet(DATASETS[ds]["shape"], size, seed, weights)


def run_effnet(ds, study, seed, data, ctx, root, force, weights, branches):
    xtr, ytr, xv, yv = data
    size = STUDIES[study]["effnet_size"]
    src = effnet_weight_source(weights)
    recipe = dict(EFFNET, input_size=size, resize="bilinear, inside the model",
                  channels="Fashion-MNIST grey repeated to 3 channels" if ds == "fashion" else "RGB",
                  backbone_input="float pixels in [0, 255]; keras include_preprocessing does /255 and ImageNet mean/std",
                  pretraining=src, batchnorm="frozen statistics (backbone BatchNorm and stochastic-depth Dropout pinned to inference mode; gamma/beta train)",
                  loss="sparse_categorical_crossentropy", class_weight="none (classes are balanced)")
    xtr_f, xv_f = xtr.astype("float32"), xv.astype("float32")
    bs = EFFNET["batch_size"]

    # Warm-up: one per dataset and seed, shared by every fine-tuning branch.
    wpath = run_dir(root, study, ds, "effnet", "warmup", seed)
    winfo = claim(wpath, dict(ctx, model="effnet", optimizer="warmup", seed=seed, phase="warmup", recipe=recipe), force)
    if winfo is not None:
        model, base = _effnet_model(ds, size, seed, weights)
        probe = keras.Model(model.input, model.get_layer("rotate").output)
        sample = probe(xtr_f[:64], training=True).numpy()
        winfo["backbone_input_range"] = [round(float(sample.min()), 3), round(float(sample.max()), 3)]
        winfo["init_weights_sha256"] = weights_sha(model)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=EFFNET["warm_lr"]),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        t0 = time.time()
        model.fit(xtr_f, ytr, validation_data=(xv_f, yv), epochs=EFFNET["warm_epochs"], batch_size=bs,
                  shuffle=True, verbose=0,
                  callbacks=[ValidationF1(xv_f, yv, bs), Progress(wpath, "warmup", 0, [])])
        model.save_weights(wpath / "warmup.weights.h5")
        finish(wpath, winfo, params=model.count_params(), trainable_params=int(sum(np.prod(w.shape) for w in model.trainable_weights)),
               train_seconds=round(time.time() - t0, 1), epochs_run=EFFNET["warm_epochs"],
               warmup_weights_sha256=weights_sha(model), warmup_file_sha256=file_sha(wpath / "warmup.weights.h5"))
    winfo = json.loads((wpath / "run.json").read_text())
    if winfo.get("status") not in ("trained", "evaluated"):
        raise SystemExit(f"warm-up {wpath} did not finish")
    warm_hist = json.loads((wpath / "history.json").read_text())

    for opt in branches:
        path = run_dir(root, study, ds, "effnet", opt, seed)
        info = claim(path, dict(ctx, model="effnet", optimizer=opt, seed=seed, phase="finetune", recipe=recipe,
                                warmup_run=str(wpath.relative_to(root)),
                                warmup_optimizer="Adam", warmup_epochs=EFFNET["warm_epochs"]), force)
        if info is None:
            continue
        model, base = _effnet_model(ds, size, seed, weights)
        model.load_weights(wpath / "warmup.weights.h5")
        start = weights_sha(model)
        if start != winfo["warmup_weights_sha256"] or file_sha(wpath / "warmup.weights.h5") != winfo["warmup_file_sha256"]:
            raise SystemExit("fine-tuning would not start from the recorded warm-up checkpoint")
        info["start_weights_sha256"] = start
        bn = [w for w in base.non_trainable_weights if "moving" in w.path]
        bn_before = hashlib.sha256(b"".join(np.asarray(w).tobytes() for w in bn)).hexdigest()
        base.trainable = True
        keras.utils.set_random_seed(seed + 1000)  # same stream for every branch of this seed
        model.compile(optimizer=FINE_OPTIMIZERS[opt](), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        best = path / "best.weights.h5"
        t0 = time.time()
        model.fit(xtr_f, ytr, validation_data=(xv_f, yv), epochs=EFFNET["fine_epochs"], batch_size=bs,
                  shuffle=True, verbose=0,
                  callbacks=[ValidationF1(xv_f, yv, bs),
                             keras.callbacks.ModelCheckpoint(str(best), monitor="val_macro_f1", mode="max",
                                                             save_best_only=True, save_weights_only=True),
                             keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=EFFNET["patience"]),
                             Progress(path, "finetune", EFFNET["warm_epochs"], warm_hist)])
        bn_after = hashlib.sha256(b"".join(np.asarray(w).tobytes() for w in bn)).hexdigest()
        hist = json.loads((path / "history.json").read_text())
        fine = [r for r in hist if r["phase"] == "finetune"]
        f1s = [r["val_macro_f1"] for r in fine]
        finish(path, info, params=model.count_params(),
               trainable_params=int(sum(np.prod(w.shape) for w in model.trainable_weights)),
               train_seconds=round(time.time() - t0, 1), epochs_run=len(fine),
               best_epoch=fine[int(np.argmax(f1s))]["epoch"], best_val_macro_f1=round(max(f1s), 5),
               batchnorm_stats_unchanged=bn_before == bn_after)


def cmd_run(args):
    xtr, ytr, xte, yte = load_official(args.dataset, args.source, args.data_dir)
    split, train, val = get_split(args.dataset, xtr, ytr)
    tr, va = study_indices(args.study, split, train, val)
    if args.limit:  # smoke tests only
        tr, va = tr[:args.limit], va[:max(args.limit // 4, 50)]
    data = (xtr[tr], ytr[tr], xtr[va], ytr[va])
    ctx = {"study": args.study, "dataset": args.dataset, "smoke": bool(args.limit),
           "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(yte)),
           "split_file": f"experiments/splits/{args.dataset}.json",
           "split_sha256": {"train": _sha_list(tr), "val": _sha_list(va)},
           "test_fingerprint": fingerprint(xte, yte), "class_order": DATASETS[args.dataset]["classes"],
           "tuning_budget": "1 configuration per model (initial config, no search)",
           "keras": keras.__version__}
    root = Path(args.out) if args.out else EXP / "runs"
    if args.fine_epochs:
        EFFNET["fine_epochs"] = args.fine_epochs
    if args.max_epochs:
        SMALL["max_epochs"] = args.max_epochs
    print(f"{args.dataset} {args.study}: train {len(tr)}, val {len(va)}, test {len(yte)} -> {root}")
    for seed in args.seeds:
        for kind in args.models:
            print(f"== {kind} seed {seed}", flush=True)
            if kind == "effnet":
                run_effnet(args.dataset, args.study, seed, data, ctx, root, args.force, args.effnet_weights, args.branches)
            else:
                run_small(kind, args.dataset, args.study, seed, data, ctx, root, args.force)


def _rebuild(info, ds, weights):
    if info["model"] == "effnet":
        return _effnet_model(ds, info["recipe"]["input_size"], info["seed"], weights)[0]
    return (build_mlp if info["model"] == "mlp" else build_cnn)(DATASETS[ds]["shape"])


def cmd_evaluate(args):
    """Test-set evaluation of every trained run of one dataset and study (after training is done)."""
    _, _, xte, yte = load_official(args.dataset, args.source, args.data_dir)
    root = Path(args.out) if args.out else EXP / "runs"
    base = root / args.study / args.dataset
    for meta in sorted(base.glob("*/seed-*/run.json")):
        info = json.loads(meta.read_text())
        if info["phase"] == "warmup" or info["status"] not in ("trained", "evaluated"):
            continue
        if info["status"] == "evaluated" and not args.force:
            continue
        if info["test_fingerprint"] != fingerprint(xte, yte):
            raise SystemExit("test data differs from the data used at training time")
        path = meta.parent
        targets = [("", "best.weights.h5")]
        if info.get("snapshot_epoch"):
            targets.append((f"epoch{info['snapshot_epoch']}", f"epoch{info['snapshot_epoch']}.weights.h5"))
        for tag, wfile in targets:
            model = _rebuild(info, args.dataset, args.effnet_weights)
            model.load_weights(path / wfile)
            x = xte.astype("float32") if info["model"] == "effnet" else (xte / 255.0).astype("float32")
            prob = model.predict(x, batch_size=256, verbose=0)
            pred = prob.argmax(1)
            f1, per = macro_f1(yte, pred)
            out = path / tag if tag else path
            out.mkdir(exist_ok=True)
            np.savez_compressed(out / "test_predictions.npz", ids=np.arange(len(yte), dtype=np.int32),
                                y_true=yte.astype(np.int8), y_pred=pred.astype(np.int8), prob=prob.astype(np.float16))
            m = {"test_accuracy": round(float((pred == yte).mean()), 6), "test_macro_f1": round(f1, 6),
                 "per_class_f1": [round(float(v), 5) for v in per], "confusion": confusion(yte, pred).tolist(),
                 "n_test": int(len(yte)), "checkpoint": wfile, "evaluated": now()}
            (out / "metrics.json").write_text(json.dumps(m, indent=1))
            print(f"{path.relative_to(root)} {tag or 'best'}: acc {m['test_accuracy']:.4f} macro-F1 {f1:.4f}")
        info["status"] = "evaluated"
        meta.write_text(json.dumps(info, indent=1))


def cmd_report(args):
    root = Path(args.out) if args.out else EXP / "runs"
    rows = []
    for meta in sorted(root.glob("*/*/*/seed-*/run.json")):
        info = json.loads(meta.read_text())
        if info["phase"] == "warmup" or info["status"] != "evaluated" or info.get("smoke"):
            continue
        for tag in ["", f"epoch{info.get('snapshot_epoch')}" if info.get("snapshot_epoch") else None]:
            if tag is None:
                continue
            m = json.loads((meta.parent / tag / "metrics.json").read_text())
            rows.append({"study": info["study"], "dataset": info["dataset"], "model": info["model"],
                         "optimizer": info["optimizer"], "checkpoint": tag or "best-val-macro-f1",
                         "seed": info["seed"], "n_train": info["n_train"],
                         "input_size": info["recipe"].get("input_size", DATASETS[info["dataset"]]["shape"][0]),
                         # global epoch numbers (EfficientNet: 2 warm-up epochs, then fine-tuning)
                         "epochs_total": len(json.loads((meta.parent / "history.json").read_text())) if not tag else info["snapshot_epoch"],
                         "best_epoch": info.get("best_epoch") if not tag else info.get("snapshot_epoch"),
                         "best_val_macro_f1": info.get("best_val_macro_f1"),
                         "test_accuracy": m["test_accuracy"], "test_macro_f1": m["test_macro_f1"],
                         "train_seconds": info["train_seconds"]})
    if not rows:
        print("no evaluated runs")
        return
    out = ROOT / "results" / "controlled_runs.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.relative_to(ROOT)} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "evaluate", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", help="run root (default experiments/runs); use another folder for smoke tests")
        if name == "report":
            continue
        p.add_argument("--dataset", choices=list(DATASETS), required=True)
        p.add_argument("--study", choices=list(STUDIES), required=True)
        p.add_argument("--source", choices=["keras", "files"], default="keras")
        p.add_argument("--data-dir")
        p.add_argument("--effnet-weights", default="imagenet")
        p.add_argument("--force", action="store_true", help="redo finished runs (old folder is kept aside)")
        if name == "run":
            p.add_argument("--models", nargs="+", choices=["mlp", "cnn", "effnet"], default=["mlp", "cnn", "effnet"])
            p.add_argument("--branches", nargs="+", choices=list(FINE_OPTIMIZERS), default=list(FINE_OPTIMIZERS))
            p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
            p.add_argument("--limit", type=int, default=0, help="smoke test: use only N training images")
            p.add_argument("--fine-epochs", type=int, default=0, help="smoke test: fewer fine-tuning epochs")
            p.add_argument("--max-epochs", type=int, default=0, help="smoke test: fewer epochs for MLP/CNN")
    args = ap.parse_args()
    {"run": cmd_run, "evaluate": cmd_evaluate, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
