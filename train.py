"""Reproduce the AM3d defect classification experiments using frozen ImageNet features."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("KERAS_HOME", str(ROOT / ".keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

SEED = 42
AM = ROOT / "data/AMdataset-52fe75dec70c1e8046493b4e3256f8f752a4d39d"
# Dataset, batch size, epochs, expected per-class counts (lexical order)
STUDIES = {
    "one": (AM / "dataset", 8, 30, (136, 135)),
    "two": (AM / "Augmented_data", 50, 50, (947, 2070)),
    "three": (ROOT / "data/study3", 30, 30, (759, 798)),
}
OPTIMIZERS = {"Adam": keras.optimizers.Adam, "SGD": keras.optimizers.SGD,
              "RMSprop": keras.optimizers.RMSprop}
MODELS = {"InceptionResNetV2": keras.applications.InceptionResNetV2,
          "EfficientNetV2M": keras.applications.EfficientNetV2M}
RESULTS = ROOT / "results"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def inventory(study, output=RESULTS):
    root, _, _, counts = STUDIES[study]
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".bmp"} and not p.name.startswith("."))
    classes = sorted({p.parent.name for p in files})
    y = np.array([classes.index(p.parent.name) for p in files], dtype=np.int32)
    if len(classes) != 2 or tuple(np.bincount(y)) != counts:
        raise ValueError(f"{root}: expected {counts}; found {classes}, {np.bincount(y)}")
    train, hold = train_test_split(np.arange(len(files)), test_size=0.30, random_state=SEED)
    val, test = train_test_split(hold, test_size=0.66, random_state=SEED)
    splits = {"train": train, "validation": val, "test": test}
    names = [p.relative_to(root).as_posix() for p in files]
    rows = [dict(path=names[i], label=classes[y[i]], class_id=y[i], split=s)
            for s, indices in splits.items() for i in indices]
    write_csv(output / study / "split.csv", rows)
    return files, names, classes, y, splits


def load_images(files, *, divide_by_255=True):
    batch = []
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot decode {path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        batch.append(cv2.resize(rgb, (224, 224)))
    images = np.asarray(batch, dtype=np.float32)
    return images / 255.0 if divide_by_255 else images


def make_head(n_features):
    return keras.Sequential([
        keras.Input(shape=(n_features,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.5, seed=SEED),
        keras.layers.Dense(2, activation="softmax"),
    ])


def extract_features(encoder, files):
    @tf.function(input_signature=[tf.TensorSpec((None, 224, 224, 3), tf.float32)])
    def encode(images):
        return encoder(images, training=False)

    batches = []
    for start in range(0, len(files), 16):
        images = load_images(files[start:start + 16])
        batches.append(encode(images).numpy())
        if start == 0:
            # Compare cached predictions against a single full image-to-softmax graph.
            probe = make_head(batches[0].shape[1])
            full = keras.Model(encoder.input, probe(encoder.output))
            cached = probe(batches[0], training=False).numpy()
            direct = full(images, training=False).numpy()
            np.testing.assert_allclose(cached, direct, atol=1e-5, rtol=1e-5)
            error = float(np.max(np.abs(cached - direct)))
        if start % 256 == 0:
            print(f"Encoded {start + len(images)}/{len(files)}", flush=True)
    return np.concatenate(batches), error


def run(study, optimizers, encoder, model_name="InceptionResNetV2", output=RESULTS):
    _, batch, epochs, _ = STUDIES[study]
    files, names, classes, y, splits = inventory(study, output)
    features, error = extract_features(encoder, files)
    labels = keras.utils.to_categorical(y, num_classes=2)
    train, val, test = (splits[key] for key in ("train", "validation", "test"))
    for name in optimizers:
        keras.utils.set_random_seed(SEED)
        head = make_head(features.shape[1])
        init_hash = hashlib.sha256(b"".join(w.tobytes() for w in head.get_weights())).hexdigest()
        head.compile(optimizer=OPTIMIZERS[name](learning_rate=0.001),
                     loss="binary_crossentropy", metrics=["accuracy"])
        dest = output / study / name
        dest.mkdir(parents=True, exist_ok=True)
        # The completion record is published after all outputs are saved.
        (dest / "run.json").unlink(missing_ok=True)
        history = head.fit(features[train], labels[train], batch_size=batch, epochs=epochs,
                           validation_data=(features[val], labels[val]), verbose=0)
        write_csv(dest / "history.csv", [
            dict(epoch=i + 1, **{k: v[i] for k, v in history.history.items()})
            for i in range(epochs)])
        summaries = {}
        for split, indices in (("validation", val), ("test", test)):
            probs = head.predict(features[indices], batch_size=batch, verbose=0)
            pred = probs.argmax(axis=1)
            write_csv(dest / f"{split}_predictions.csv", [
                dict(path=names[i], label=classes[y[i]], prediction=classes[j],
                     probability_0=float(p[0]), probability_1=float(p[1]))
                for i, j, p in zip(indices, pred, probs)])
            summaries[split] = {"n": len(indices), "correct": int(np.sum(pred == y[indices]))}
        head.save_weights(dest / "head.weights.h5")
        record = dict(model=model_name, study=study, optimizer=type(head.optimizer).__name__,
                      learning_rate=float(head.optimizer.learning_rate.numpy()),
                      epochs=len(history.epoch), batch_size=batch, seed=SEED,
                      class_order=classes, initial_weights_sha256=init_hash,
                      cache_prediction_max_abs_error=error, **summaries)
        temp = dest / "run.tmp"
        temp.write_text(json.dumps(record, indent=2) + "\n")
        temp.replace(dest / "run.json")
        print(f"{study} {name}: {summaries}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="InceptionResNetV2")
    parser.add_argument("--study", choices=["all", *STUDIES], default="all")
    parser.add_argument("--optimizer", choices=["all", *OPTIMIZERS], default="all")
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    output = RESULTS if args.model == "InceptionResNetV2" else ROOT / "results_efficientnetv2m"
    weight_name = ("inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
                   if args.model == "InceptionResNetV2" else "efficientnetv2-m_notop.h5")
    weight_path = ROOT / ".keras" / "models" / weight_name
    if not weight_path.is_file():
        raise FileNotFoundError(f"Missing {weight_path}; run python main.py prepare first")
    base = MODELS[args.model](include_top=False, weights=str(weight_path), input_shape=(224, 224, 3))
    base.trainable = False
    pooled = keras.layers.AveragePooling2D((4, 4))(base.output)
    encoder = keras.Model(base.input, keras.layers.Flatten()(pooled))
    for study in STUDIES if args.study == "all" else [args.study]:
        run(study, OPTIMIZERS if args.optimizer == "all" else [args.optimizer], encoder, args.model, output)
