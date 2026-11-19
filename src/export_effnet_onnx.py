"""Export the EfficientNetV2B0 checkpoints to ONNX so the dashboard can classify uploaded images.

For every finished EfficientNet run of the small-subset study (seed 1) this rebuilds the inference
graph (gray to RGB for Fashion-MNIST, resize to 96 x 96, backbone, dense head; the flip and rotation
augmentation is inactive at inference), loads best.weights.h5, converts it to ONNX with weights
stored as 16-bit floats (computed in float32), and writes dashboard/data/<dataset>/effnet-<optimizer>.onnx. Input: pixels 0..255 as
float32, shape (n, H, W, C) at the dataset's own size. Output: 10 softmax probabilities.

AdamW produced bit-identical weights to Adam (weight decay 1e-5 at learning rate 6e-5 is below
float32 precision), so when the weights match, no separate AdamW file is written and the dashboard
uses the Adam file for it.

Before writing, the ONNX model is run on all 10,000 official test images and compared with the
predictions saved by `experiments.py evaluate`; the export stops if the predicted class differs on
more than 0.1% of the images. The agreement is written to dashboard/data/effnet_onnx.json.

    python src/export_effnet_onnx.py --source files --cifar-dir <dir> --fashion-dir <dir> --effnet-weights <imagenet .h5>
Needs: pip install tf2onnx onnx onnxruntime
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments as E  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dashboard" / "data"
MAX_DISAGREE = 0.001


def official_atlas(ds, xte):
    """The gallery atlas in data.json holds the images the historical sweep was trained on; for CIFAR-10
    that was a JPG copy whose pixels differ from the official files. EfficientNet was trained and
    evaluated on the official files, so it gets an atlas of the same 1,000 test images from those
    files. Returns the file name, or None when the existing atlas already equals the official pixels."""
    import base64
    import io
    from PIL import Image
    d = json.loads((OUT / ds / "data.json").read_text(encoding="utf-8"))
    ids = json.loads((OUT / ds / "experiments.json").read_text(encoding="utf-8"))["pool_ids"]
    cols, size = d["cols"], xte.shape[1]
    old = np.array(Image.open(io.BytesIO(base64.b64decode(d["atlas"].split(",", 1)[1]))).convert("RGB"))
    new = np.zeros_like(old)
    for k, i in enumerate(ids):
        r, c = divmod(k, cols)
        new[r * size:(r + 1) * size, c * size:(c + 1) * size] = xte[i]  # grayscale broadcasts to RGB
    diff = float(np.abs(old.astype(int) - new.astype(int)).mean())
    print(f"{ds}: gallery atlas vs official pixels, mean absolute difference {diff:.3f}")
    if diff == 0:
        return None
    name = "atlas-official.png"
    Image.fromarray(new).save(OUT / ds / name, "PNG", optimize=True)
    return name


def half_weights(proto, min_size=256):
    """Store the large float32 weights as float16 and cast them back to float32 inside the graph.
    The file is half the size, and the computation itself stays in float32."""
    import onnx
    from onnx import helper, numpy_helper
    g = proto.graph
    keep, casts = [], []
    for init in g.initializer:
        if init.data_type == onnx.TensorProto.FLOAT and np.prod(init.dims) >= min_size:
            arr = numpy_helper.to_array(init).astype(np.float16)
            h = numpy_helper.from_array(arr, init.name + "__f16")
            keep.append(h)
            casts.append(helper.make_node("Cast", [h.name], [init.name], to=onnx.TensorProto.FLOAT, name=init.name + "__cast"))
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = casts + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    return proto


def export(ds, opt, seed, xte, weights_src):
    import onnx
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx

    path = E.EXP / "runs" / "budget" / ds / f"effnet-{opt}" / f"seed-{seed}"
    info = json.loads((path / "run.json").read_text(encoding="utf-8"))
    size = info["recipe"]["input_size"]
    model, _ = E.build_effnet(E.DATASETS[ds]["shape"], size, seed, weights_src)
    model.load_weights(path / "best.weights.h5")
    h, w, c = E.DATASETS[ds]["shape"]
    spec = (tf.TensorSpec((None, h, w, c), tf.float32, name="pixels"),)
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    # freeze every variable (including the backbone's input normalization) into constants first
    cf = convert_variables_to_constants_v2(tf.function(lambda x: model(x, training=False)).get_concrete_function(*spec))
    gd = cf.graph.as_graph_def()
    proto, _ = tf2onnx.convert.from_graph_def(gd, input_names=[t.name for t in cf.inputs], output_names=[t.name for t in cf.outputs], opset=17)
    proto = half_weights(proto)
    onnx.checker.check_model(proto)
    sess = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    prob = np.concatenate([sess.run(None, {name: xte[i:i + 250].astype("float32")})[0] for i in range(0, len(xte), 250)])
    saved = np.load(path / "test_predictions.npz")
    agree = float((prob.argmax(1) == saved["y_pred"]).mean())
    acc = float((prob.argmax(1) == saved["y_true"]).mean())
    gap = float(np.abs(prob - saved["prob"].astype("float32")).max())
    saved_acc = json.loads((path / "metrics.json").read_text(encoding="utf-8"))["test_accuracy"]
    print(f"{ds} {opt}: class agreement {agree:.4%}, ONNX accuracy {acc:.4%} (Keras {saved_acc:.4%}), max prob diff {gap:.4f}")
    if 1 - agree > MAX_DISAGREE:
        raise SystemExit(f"{ds} {opt}: the ONNX model disagrees with the saved predictions on {1 - agree:.2%} of the test images")
    return proto, {"agreement": round(agree, 6), "accuracy": round(acc, 6), "max_prob_diff": round(gap, 5),
                   "input_size": size, "seed": seed, "checkpoint": f"experiments/runs/budget/{ds}/effnet-{opt}/seed-{seed}/best.weights.h5"}, model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["keras", "files"], default="keras")
    ap.add_argument("--cifar-dir", help="folder with the official cifar-10-batches-py files (with --source files)")
    ap.add_argument("--fashion-dir", help="folder with the official Fashion-MNIST .gz files (with --source files)")
    ap.add_argument("--effnet-weights", default="imagenet")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    report = {}
    for ds in ["cifar10", "fashion"]:
        _, _, xte, _ = E.load_official(ds, a.source, a.cifar_dir if ds == "cifar10" else a.fashion_dir)
        report[ds] = {}
        atlas = official_atlas(ds, xte)
        if atlas:
            report[ds]["atlas"] = atlas
        adam_w = None
        for opt in ["adam", "adamw", "nadam"]:
            path = E.EXP / "runs" / "budget" / ds / f"effnet-{opt}" / f"seed-{a.seed}" / "best.weights.h5"
            if not path.exists():
                continue
            if opt == "adamw" and adam_w is not None:
                m, _ = E.build_effnet(E.DATASETS[ds]["shape"], 96, a.seed, a.effnet_weights)
                m.load_weights(path)
                if E.weights_sha(m) == adam_w:
                    report[ds]["adamw"] = {"same_as": "adam"}
                    print(f"{ds} adamw: weights identical to adam, using effnet-adam.onnx")
                    continue
            proto, rep, model = export(ds, opt, a.seed, xte, a.effnet_weights)
            if opt == "adam":
                adam_w = E.weights_sha(model)
            out = OUT / ds / f"effnet-{opt}.onnx"
            out.write_bytes(proto.SerializeToString())
            rep["file"] = out.name
            rep["bytes"] = out.stat().st_size
            report[ds][opt] = rep
    (OUT / "effnet_onnx.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
