"""
Train the models used by the dashboard, for CIFAR-10 or Fashion-MNIST.

What it trains (same setup for every run: seed 42, 10 epochs, batch 128, 10% validation split):
  two architectures, an MLP and a CNN, each trained eight times, once per optimizer:
  SGD, RMSprop, Adam, AdamW, Adagrad, Nadam, Adadelta, and SGD with Nesterov momentum

Outputs (per dataset):
  models/<dataset>/<model>-<optimizer>.keras         trained model (weights only, no optimizer state)
  models/<dataset>/<model>-<optimizer>.history.json  per-epoch history, test accuracy, training time
  results/<dataset>_results.csv                      test accuracy and loss for every run

Usage:
  python src/train.py --dataset cifar10                    # downloads through keras.datasets
  python src/train.py --dataset fashion
  python src/train.py --dataset cifar10 --source files --data-dir data/CIFAR-10-images
  python src/train.py --dataset fashion --source files --data-dir data/fashion
  python src/train.py --dataset cifar10 --models mlp                # only the MLP runs
  python src/train.py --dataset cifar10 --epochs 1 --limit 2000   # quick smoke test
"""
import argparse
import csv
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras  # noqa: E402
from keras import layers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "cifar10": {
        "title": "CIFAR-10",
        "classes": ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
        "shape": (32, 32, 3),
    },
    "fashion": {
        "title": "Fashion-MNIST",
        "classes": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
        "shape": (28, 28, 1),
    },
}

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


def _cifar_files(folder):
    """Image-file copy of CIFAR-10 (github.com/YoongiKim/CIFAR-10-images): <dir>/{train,test}/<class>/*.jpg"""
    from PIL import Image

    def split(name):
        xs, ys = [], []
        for label, cls in enumerate(DATASETS["cifar10"]["classes"]):
            for f in sorted((Path(folder) / name / cls).iterdir()):
                xs.append(np.asarray(Image.open(f).convert("RGB")))
                ys.append(label)
        return np.stack(xs), np.array(ys)

    return split("train"), split("test")


def _fashion_files(folder):
    """The four .gz files from github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion"""
    def read(name, offset, shape):
        with gzip.open(Path(folder) / name, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=offset).reshape(shape)

    xtr = read("train-images-idx3-ubyte.gz", 16, (-1, 28, 28))
    ytr = read("train-labels-idx1-ubyte.gz", 8, (-1,))
    xte = read("t10k-images-idx3-ubyte.gz", 16, (-1, 28, 28))
    yte = read("t10k-labels-idx1-ubyte.gz", 8, (-1,))
    return (xtr, ytr), (xte, yte)


def load_data(dataset, source="keras", data_dir=None):
    """Returns uint8 images with a channel axis, (N, H, W, C), and integer labels."""
    if dataset == "cifar10":
        if source == "keras":
            (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
        else:
            (xtr, ytr), (xte, yte) = _cifar_files(data_dir or "data/CIFAR-10-images")
    else:
        if source == "keras":
            (xtr, ytr), (xte, yte) = keras.datasets.fashion_mnist.load_data()
        else:
            (xtr, ytr), (xte, yte) = _fashion_files(data_dir or "data/fashion")
        xtr, xte = xtr[..., None], xte[..., None]
    return xtr, np.asarray(ytr).ravel(), xte, np.asarray(yte).ravel()


def build_cnn(input_shape):
    return keras.Sequential([
        layers.Input(input_shape),
        layers.Conv2D(32, 3, activation="relu", name="conv1"),
        layers.MaxPooling2D(2, name="pool1"),
        layers.Conv2D(64, 3, activation="relu", name="conv2"),
        layers.MaxPooling2D(2, name="pool2"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense"),
        layers.Dropout(0.3, name="dropout"),
        layers.Dense(10, activation="softmax", name="output"),
    ], name="cnn")


def build_mlp(input_shape):
    return keras.Sequential([
        layers.Input(input_shape),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ], name="mlp")


def save_weights_only(trained, fresh, path):
    """Save the trained weights in an uncompiled copy, so the file carries no optimizer state
    (an Adam checkpoint would otherwise be about three times larger)."""
    fresh.set_weights(trained.get_weights())
    fresh.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), default="cifar10")
    ap.add_argument("--source", choices=["keras", "files"], default="keras")
    ap.add_argument("--data-dir", help="folder for --source files")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training images (smoke test)")
    ap.add_argument("--models", nargs="*", choices=["mlp", "cnn"], default=["mlp", "cnn"])
    ap.add_argument("--only", nargs="*", help="train only these optimizers, e.g. --only Adam SGD")
    args = ap.parse_args()

    shape = DATASETS[args.dataset]["shape"]
    models_dir = ROOT / "models" / args.dataset
    results_dir = ROOT / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    xtr, ytr, xte, yte = load_data(args.dataset, args.source, args.data_dir)
    # Shuffle once with a fixed seed: the image-file copies are ordered by class
    # and validation_split always takes the last 10% of the array.
    keras.utils.set_random_seed(42)
    order = np.random.permutation(len(xtr))
    xtr, ytr = xtr[order], ytr[order]
    if args.limit:
        xtr, ytr = xtr[:args.limit], ytr[:args.limit]
    xtr = (xtr / 255.0).astype("float32")
    xte_f = (xte / 255.0).astype("float32")
    print(f"{args.dataset}: train {xtr.shape}, test {xte_f.shape}, source={args.source}")

    builders = {"mlp": build_mlp, "cnn": build_cnn}
    rows = []
    for model_key in args.models:
        for name, make_opt in OPTIMIZERS.items():
            if args.only and name not in args.only:
                continue
            keras.utils.set_random_seed(42)
            model = builders[model_key](shape)
            model.compile(optimizer=make_opt(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            t0 = time.time()
            hist = model.fit(xtr, ytr, epochs=args.epochs, batch_size=128, validation_split=0.1, verbose=2)
            seconds = time.time() - t0
            test_loss, test_acc = model.evaluate(xte_f, yte, verbose=0)

            file = f"{model_key}-{slug(name)}"
            save_weights_only(model, builders[model_key](shape), models_dir / f"{file}.keras")
            json.dump({"model": model_key, "name": name, "params": model.count_params(), "time": round(seconds, 1),
                       "test_accuracy": round(float(test_acc), 4), "test_loss": round(float(test_loss), 4),
                       "hist": {k: [round(float(v), 4) for v in vs] for k, vs in hist.history.items()}},
                      open(models_dir / f"{file}.history.json", "w"), indent=1)
            rows.append({"dataset": DATASETS[args.dataset]["title"], "model": model_key.upper(),
                         "optimizer": name, "params": model.count_params(),
                         "test_accuracy": round(float(test_acc), 4), "test_loss": round(float(test_loss), 4),
                         "train_seconds": round(seconds, 1)})
            print(f"{model_key.upper()} {name:13s} test accuracy {test_acc:.4f}  ({seconds:.0f} s)", flush=True)

    out = results_dir / f"{args.dataset}_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {models_dir} and {out}")


if __name__ == "__main__":
    main()
