"""Reproducible EfficientNetV2M multi-scale adaptive fusion experiment."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train import ROOT, STUDIES, OPTIMIZERS, load_images, write_csv
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

OUT = HERE / "results"
SEEDS = (42, 43, 44)
VARIANTS = ("B", "C", "D", "E", "F")
LAYERS = ("block3e_add", "block5n_add", "top_activation")
CHANNELS = (80, 176, 1280)
WIDTH = CHANNELS[-1] + 2 * sum(CHANNELS)
VERSION = "msaf-1"


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def manifest(study):
    original = ROOT / "results_efficientnetv2m" / study / "split.csv"
    rows = read_csv(original)
    paths = [STUDIES[study][0] / r["path"] for r in rows]
    y = np.array([int(r["class_id"]) for r in rows], np.int32)
    assert len(paths) == len(set(paths)) == sum(STUDIES[study][3])
    assert tuple(np.bincount(y)) == STUDIES[study][3]
    assert all(p.is_file() for p in paths)
    splits = {s: np.flatnonzero([r["split"] == s for r in rows]) for s in ("train", "validation", "test")}
    assert sum(map(len, splits.values())) == len(rows)
    return rows, paths, y, splits


def pixel_batch(paths):
    return np.rint(load_images(paths) * 255.).astype(np.float32)


def make_encoder():
    weight_file = ROOT / ".keras" / "models" / "efficientnetv2-m_notop.h5"
    if not weight_file.is_file():
        raise FileNotFoundError(f"Missing pretrained EfficientNetV2M weights: {weight_file}")
    with weight_file.open("rb") as file:
        assert hashlib.file_digest(file, "md5").hexdigest() == "1b8dc055df72dde80d614482840fe342"
    base = keras.applications.EfficientNetV2M(include_top=False, weights=str(weight_file), input_shape=(224, 224, 3))
    base.trainable = False
    maps = keras.Model(base.input, [base.get_layer(name).output for name in LAYERS], name="multiscale_backbone")
    assert [tuple(x.shape[1:]) for x in maps.outputs] == [(28, 28, 80), (14, 14, 176), (7, 7, 1280)]
    return maps, base.count_params()


def pooled(maps):
    """Width 4352: legacy pool, then each scale's GAP and GMP."""
    legacy = tf.reshape(tf.nn.avg_pool2d(maps[-1], ksize=4, strides=4, padding="VALID"), (-1, CHANNELS[-1]))
    return tf.concat([legacy] + [stat for m in maps for stat in (tf.reduce_mean(m, (1, 2)), tf.reduce_max(m, (1, 2)))], axis=1)


def cache_features(studies):
    encoder = None
    for study in studies:
        rows, paths, y, splits = manifest(study)
        root = OUT / study
        root.mkdir(parents=True, exist_ok=True)
        # Identify the manifest and source by path+content, and audit byte-identical duplicates.
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        identity = hashlib.sha256(json.dumps([VERSION, rows, hashes]).encode()).hexdigest()
        duplicates = {}
        for row, digest in zip(rows, hashes):
            duplicates.setdefault(digest, []).append(row)
        cross = [group for group in duplicates.values() if len({r["split"] for r in group}) > 1]
        audit = dict(identity=identity, images=len(rows), cross_split_exact_duplicate_groups=len(cross),
                     images_in_cross_split_exact_duplicate_groups=sum(map(len, cross)), duplicate_groups=cross)
        save_json(root / "data_audit.json", audit)
        write_csv(root / "split.csv", rows)
        features_file = root / "features.npy"
        meta_file = root / "features.json"
        if features_file.exists() and meta_file.exists():
            meta = json.loads(meta_file.read_text())
            if meta["identity"] == identity and meta["version"] == VERSION:
                assert np.load(features_file, mmap_mode="r").shape == (len(rows), WIDTH)
                print(f"{study}: reused feature cache", flush=True)
                continue
        if encoder is None:
            encoder, parameters = make_encoder()

            @tf.function(input_signature=[tf.TensorSpec((None, 224, 224, 3), tf.float32)])
            def encode(x):
                return pooled(encoder(x, training=False))

        features = []
        for start in range(0, len(paths), 16):
            batch = pixel_batch(paths[start:start + 16])
            x = encode(batch).numpy()
            assert x.shape == (len(batch), WIDTH) and np.isfinite(x).all()
            features.append(x)
            if start == 0 or start % 256 == 0 or start + len(batch) == len(paths):
                print(f"{study}: encoded {start + len(batch)}/{len(paths)} images", flush=True)
        temp = root / "features.tmp.npy"
        np.save(temp, np.concatenate(features))
        temp.replace(features_file)
        save_json(meta_file, dict(version=VERSION, identity=identity, backbone_parameters=parameters,
                                  feature_layers=list(LAYERS), feature_channels=list(CHANNELS), input_range=[0, 255]))
    del encoder
    keras.backend.clear_session()


def normalize(x, name):
    return keras.layers.LayerNormalization(axis=-1, center=False, scale=False, epsilon=1e-5, name=name)(x)


def make_head(variant, seed):
    """B=legacy; C=deep GAP; D=multi GAP; E=fixed mix; F=learned mix."""
    x = keras.Input((WIDTH,), name="pooled_features")
    gates = []
    if variant == "B":
        combined = x[:, :CHANNELS[-1]]
    else:
        offset = CHANNELS[-1]
        vectors = []
        for i, ch in enumerate(CHANNELS):
            avg_raw = x[:, offset:offset + ch]
            max_raw = x[:, offset + ch:offset + 2 * ch]
            offset += 2 * ch
            if variant == "C" and i < 2:
                continue
            avg = normalize(avg_raw, f"avg_norm_{i}")
            if variant in ("C", "D"):
                vector = avg
            else:
                maximum = normalize(max_raw, f"max_norm_{i}")
                if variant == "E":
                    vector = keras.layers.Average(name=f"fixed_mix_{i}")([avg, maximum])
                else:
                    gate = keras.layers.Dense(1, activation="sigmoid", kernel_initializer="zeros",
                                               bias_initializer="zeros", name=f"max_gate_{i}")(
                        keras.layers.Concatenate()([avg, maximum]))
                    # gate denotes the learned proportion of peak information.
                    vector = avg + gate * (maximum - avg)
                    gates.append(gate)
            vectors.append(vector)
        combined = vectors[0] if len(vectors) == 1 else keras.layers.Concatenate(name="multi_scale_fusion")(vectors)
    hidden = keras.layers.Dense(64, activation="relu", kernel_initializer=keras.initializers.GlorotUniform(seed), name="head_dense")(combined)
    hidden = keras.layers.Dropout(0.5, seed=seed, name="dropout")(hidden)
    logits = keras.layers.Dense(2, kernel_initializer=keras.initializers.GlorotUniform(seed + 1000), name="logits")(hidden)
    probs = keras.layers.Softmax(name="class_probability")(logits)
    model = keras.Model(x, probs, name=f"variant_{variant}")
    return model


def metrics(y, p):
    pred = p.argmax(axis=1)
    tp, fn, fp, tn = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    divide = lambda a, b: float(a / b) if b else 0.0
    return dict(n=len(y), correct=int(tp + tn), tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn),
                accuracy=divide(tp + tn, len(y)), precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                f1=divide(2 * tp, 2 * tp + fp + fn), specificity=divide(tn, tn + fp),
                roc_auc=float(roc_auc_score(y == 0, p[:, 0])), average_precision=float(average_precision_score(y == 0, p[:, 0])))


def save_predictions(dest, split, rows, y, p):
    assert p.shape == (len(rows), 2) and np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1, atol=1e-5)
    write_csv(dest / f"{split}_predictions.csv", [dict(path=r["path"], true_class=int(label), predicted_class=int(prob.argmax()),
              defect_probability=float(prob[0]), normal_probability=float(prob[1])) for r, label, prob in zip(rows, y, p, strict=True)])
    return metrics(y, p)


def data_for(study):
    rows, paths, y, split = manifest(study)
    x = np.load(OUT / study / "features.npy")
    assert x.shape == (len(rows), WIDTH)
    return rows, y, split, x


def fit_all(studies):
    for study in studies:
        rows, y, split, x = data_for(study)
        train, val = split["train"], split["validation"]
        _, batch_size, epochs, _ = STUDIES[study]
        identity = json.loads((OUT / study / "features.json").read_text())["identity"]
        for variant in VARIANTS:
            for optimizer in OPTIMIZERS:
                for seed in SEEDS:
                    dest = OUT / study / variant / optimizer / str(seed)
                    config = dict(version=VERSION, identity=identity, study=study, variant=variant, optimizer=optimizer,
                                  seed=seed, batch_size=batch_size, epochs=epochs, learning_rate=0.001)
                    record = dest / "run.json"
                    if record.exists() and (dest / "head.weights.h5").exists() and (dest / "validation_predictions.csv").exists():
                        if json.loads(record.read_text())["config"] == config:
                            continue
                    dest.mkdir(parents=True, exist_ok=True)
                    keras.backend.clear_session()
                    keras.utils.set_random_seed(seed)
                    model = make_head(variant, seed)
                    if variant == "F":
                        gate_model = keras.Model(model.input, [model.get_layer(f"max_gate_{i}").output for i in range(3)])
                        np.testing.assert_allclose(np.asarray(gate_model(x[train[:2]])), 0.5)
                        same = make_head("E", seed)
                        np.testing.assert_allclose(model(x[train[:2]]).numpy(), same(x[train[:2]]).numpy(), atol=1e-6)
                    model.compile(optimizer=OPTIMIZERS[optimizer](learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
                    labels = keras.utils.to_categorical(y[train], 2)
                    options = tf.data.Options()
                    options.threading.private_threadpool_size = 1
                    ds = tf.data.Dataset.from_tensor_slices((x[train], labels)).shuffle(len(train), seed=seed, reshuffle_each_iteration=True).batch(batch_size).with_options(options)
                    before = time.perf_counter()
                    history = model.fit(ds, epochs=epochs, verbose=0)
                    elapsed = time.perf_counter() - before
                    write_csv(dest / "history.csv", [dict(epoch=i + 1, **{k: float(v[i]) for k, v in history.history.items()}) for i in range(epochs)])
                    p = model(x[val], training=False).numpy()
                    scores = save_predictions(dest, "validation", [rows[i] for i in val], y[val], p)
                    model.save_weights(dest / "head.weights.h5")
                    save_json(record, dict(config=config, validation=scores, head_parameters=model.count_params(), seconds=elapsed,
                                           updates=int(model.optimizer.iterations.numpy())))
                    print(f"{study} {variant} {optimizer} {seed}: val acc {scores['accuracy']:.4f}, F1 {scores['f1']:.4f} ({elapsed:.1f}s)", flush=True)


def load_head(study, variant, optimizer, seed):
    dest = OUT / study / variant / optimizer / str(seed)
    head = make_head(variant, seed)
    head.load_weights(dest / "head.weights.h5")
    return head, dest


def evaluate_all(studies):
    validation, test, cross, gates = [], [], [], []
    selected = {}
    for study in studies:
        for variant in VARIANTS:
            for optimizer in OPTIMIZERS:
                for seed in SEEDS:
                    r = json.loads((OUT / study / variant / optimizer / str(seed) / "run.json").read_text())
                    validation.append(dict(study=study, variant=variant, optimizer=optimizer, seed=seed, **r["validation"],
                                           head_parameters=r["head_parameters"], train_seconds=r["seconds"]))
            matching = [r for r in validation if r["study"] == study and r["variant"] == variant]
            rank = sorted(OPTIMIZERS, key=lambda opt: (-np.mean([r["f1"] for r in matching if r["optimizer"] == opt]),
                                                       -np.mean([r["recall"] for r in matching if r["optimizer"] == opt]), list(OPTIMIZERS).index(opt)))
            selected[f"{study}:{variant}"] = rank[0]
    write_csv(OUT / "validation.csv", validation)
    save_json(OUT / "selection.json", dict(rule="highest mean validation defect F1 across 3 seeds; recall then fixed optimizer order on ties", optimizers=selected))
    for study in studies:
        rows, y, split, x = data_for(study)
        ids = split["test"]
        for variant in VARIANTS:
            for optimizer in OPTIMIZERS:
                for seed in SEEDS:
                    keras.backend.clear_session()
                    head, dest = load_head(study, variant, optimizer, seed)
                    p = head(x[ids], training=False).numpy()
                    m = save_predictions(dest, "test", [rows[i] for i in ids], y[ids], p)
                    test.append(dict(study=study, variant=variant, optimizer=optimizer, seed=seed, **m))
                    if variant == "F" and optimizer == selected[f"{study}:F"]:
                        gm = keras.Model(head.input, [head.get_layer(f"max_gate_{i}").output for i in range(3)])
                        weights = np.concatenate([g.numpy() for g in gm(x[ids], training=False)], axis=1)
                        for i, row in enumerate(ids):
                            gates.append(dict(study=study, seed=seed, path=rows[row]["path"], true_class=int(y[row]),
                                              shallow_max=float(weights[i, 0]), middle_max=float(weights[i, 1]), deep_max=float(weights[i, 2])))
        print(f"{study}: test evaluations complete", flush=True)
    write_csv(OUT / "test.csv", test)
    write_csv(OUT / "gate_weights.csv", gates)
    if {"one", "three"}.issubset(studies):
        for source, target in (("one", "three"), ("three", "one")):
            rows, y, _, x = data_for(target)
            for variant in ("B", "F"):
                opt = selected[f"{source}:{variant}"]
                for seed in SEEDS:
                    head, _ = load_head(source, variant, opt, seed)
                    p = head(x, training=False).numpy()
                    dest = OUT / "cross_dataset" / f"{source}_to_{target}" / variant / str(seed)
                    m = save_predictions(dest, "target", rows, y, p)
                    cross.append(dict(source=source, target=target, variant=variant, optimizer=opt, seed=seed, **m))
        write_csv(OUT / "cross_dataset.csv", cross)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "cache", "train", "evaluate"), default="all")
    parser.add_argument("--study", choices=("all", *STUDIES), default="all")
    args = parser.parse_args()
    studies = list(STUDIES) if args.study == "all" else [args.study]
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    if args.stage in ("all", "cache"):
        cache_features(studies)
    if args.stage in ("all", "train"):
        fit_all(studies)
    if args.stage in ("all", "evaluate"):
        evaluate_all(studies)


if __name__ == "__main__":
    main()
