"""
Train the CIFAR-10 models used by the dashboard.

What it trains (same setup for every run: seed 42, 10 epochs, batch 128, 10% validation split):
  * one MLP baseline with Adam
  * the same CNN eight times, once per optimizer:
    SGD, RMSprop, Adam, AdamW, Adagrad, Nadam, Adadelta, and SGD with Nesterov momentum

Outputs:
  models/<name>.keras            trained model
  models/<name>.history.json     per-epoch history and training time
  results/optimizer_results.csv  test accuracy and loss for every run

Usage:
  python src/train.py                       # official CIFAR-10 via keras.datasets
  python src/train.py --source jpg --jpg-dir path/to/CIFAR-10-images
  python src/train.py --epochs 1 --limit 2000   # quick smoke test
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras  # noqa: E402
from keras import layers  # noqa: E402

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
ROOT = Path(__file__).resolve().parents[1]

OPTIMIZERS = {
    "SGD": lambda: keras.optimizers.SGD(),
    "RMSprop": lambda: keras.optimizers.RMSprop(),
    "Adam": lambda: keras.optimizers.Adam(),
    "AdamW": lambda: keras.optimizers.AdamW(),
    "Adagrad": lambda: keras.optimizers.Adagrad(),
    "Nadam": lambda: keras.optimizers.Nadam(),
    "Adadelta": lambda: keras.optimizers.Adadelta(),
    "SGD momentum": lambda: keras.optimizers.SGD(learning_rate=0.05, momentum=0.9, nesterov=True),
}


def slug(name):
    return name.lower().replace(" ", "-")


def load_jpg(jpg_dir):
    """Read the PNG/JPG mirror (github.com/YoongiKim/CIFAR-10-images): <dir>/{train,test}/<class>/*.jpg"""
    from PIL import Image

    def split(name):
        xs, ys = [], []
        for label, cls in enumerate(CLASSES):
            folder = Path(jpg_dir) / name / cls
            for f in sorted(folder.iterdir()):
                xs.append(np.asarray(Image.open(f).convert("RGB")))
                ys.append(label)
        return np.stack(xs), np.array(ys)

    return split("train"), split("test")


def load_data(source, jpg_dir):
    if source == "keras":
        (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
        ytr, yte = ytr.ravel(), yte.ravel()
    else:
        (xtr, ytr), (xte, yte) = load_jpg(jpg_dir)
    return xtr, ytr, xte, yte


def build_cnn():
    return keras.Sequential([
        layers.Input((32, 32, 3)),
        layers.Conv2D(32, 3, activation="relu", name="conv1"),
        layers.MaxPooling2D(2, name="pool1"),
        layers.Conv2D(64, 3, activation="relu", name="conv2"),
        layers.MaxPooling2D(2, name="pool2"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense"),
        layers.Dropout(0.3, name="dropout"),
        layers.Dense(10, activation="softmax", name="output"),
    ], name="cnn_cifar")


def build_mlp():
    return keras.Sequential([
        layers.Input((32, 32, 3)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ], name="mlp_cifar")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["keras", "jpg"], default="keras")
    ap.add_argument("--jpg-dir", default="data/CIFAR-10-images")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training images (smoke test)")
    ap.add_argument("--only", nargs="*", help="train only these runs, e.g. --only MLP Adam")
    args = ap.parse_args()

    models_dir = ROOT / "models"
    results_dir = ROOT / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    xtr, ytr, xte, yte = load_data(args.source, args.jpg_dir)
    # Shuffle once with a fixed seed: the JPG mirror is ordered by class and
    # validation_split always takes the last 10% of the array.
    keras.utils.set_random_seed(42)
    order = np.random.permutation(len(xtr))
    xtr, ytr = xtr[order], ytr[order]
    if args.limit:
        xtr, ytr = xtr[:args.limit], ytr[:args.limit]
    xtr = (xtr / 255.0).astype("float32")
    xte_f = (xte / 255.0).astype("float32")
    print(f"train {xtr.shape}, test {xte_f.shape}, source={args.source}")

    runs = {"MLP": (build_mlp, OPTIMIZERS["Adam"])}
    runs.update({name: (build_cnn, opt) for name, opt in OPTIMIZERS.items()})
    if args.only:
        runs = {k: v for k, v in runs.items() if k in args.only}

    rows = []
    for name, (builder, make_opt) in runs.items():
        keras.utils.set_random_seed(42)
        model = builder()
        model.compile(optimizer=make_opt(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        t0 = time.time()
        hist = model.fit(xtr, ytr, epochs=args.epochs, batch_size=128, validation_split=0.1, verbose=2)
        seconds = time.time() - t0
        test_loss, test_acc = model.evaluate(xte_f, yte, verbose=0)

        file = "mlp" if name == "MLP" else f"cnn-{slug(name)}"
        model.save(models_dir / f"{file}.keras")
        json.dump({"name": name, "time": round(seconds, 1),
                   "hist": {k: [round(float(v), 4) for v in vs] for k, vs in hist.history.items()}},
                  open(models_dir / f"{file}.history.json", "w"), indent=1)
        rows.append({"run": name, "model": "MLP" if name == "MLP" else "CNN",
                     "params": model.count_params(), "test_accuracy": round(float(test_acc), 4),
                     "test_loss": round(float(test_loss), 4), "train_seconds": round(seconds, 1)})
        print(f"{name:13s} test accuracy {test_acc:.4f}  ({seconds:.0f} s)", flush=True)

    with open(results_dir / "optimizer_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved models/ and results/optimizer_results.csv")


if __name__ == "__main__":
    main()
