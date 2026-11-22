"""Controlled EfficientNetV2M augmentation experiment. Run from the existing project venv."""
import csv
import hashlib
import json
from pathlib import Path

from train import ROOT, SEED, STUDIES, OPTIMIZERS, inventory, load_images, make_head, write_csv
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

OUT = ROOT / "augmentation_study"
SCALES = ("original", "keras")
POLICIES = ("none", "geometry", "geometry_light")


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def transform(images, image_ids):
    geometry, lighting = [], []
    for image, index in zip(images, image_ids):
        rng = np.random.default_rng([SEED, int(index)])
        matrix = cv2.getRotationMatrix2D((111.5, 111.5), rng.uniform(-8, 8), 1)
        matrix[:, 2] += rng.uniform(-0.03, 0.03, 2) * 224
        view = cv2.warpAffine(image, matrix, (224, 224), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
        geometry.append(view)
        gain = rng.uniform(0.90, 1.10)
        offset = rng.uniform(-12, 12)
        lighting.append(np.clip((view - 127.5) * gain + 127.5 + offset, 0, 255))
    return np.asarray(geometry), np.asarray(lighting)


def prepare_features(study, files, splits, encode):
    """Clean features for all images, one cached transformed view per training image/policy."""
    all_ids = np.arange(len(files))
    banks = {scale: {} for scale in SCALES}
    for augmented, ids in [(False, all_ids), (True, splits["train"])]:
        chunks = {(scale, policy): [] for scale in SCALES
                  for policy in (POLICIES[1:] if augmented else ("none",))}
        for start in range(0, len(ids), 16):
            batch = ids[start:start + 16]
            images = np.rint(load_images([files[i] for i in batch]) * 255.0)
            views = dict(zip(POLICIES[1:], transform(images, batch))) if augmented else {"none": images}
            for (scale, policy), parts in chunks.items():
                pixels = views[policy]
                parts.append(encode(pixels / 255.0 if scale == "original" else pixels).numpy())
            if start % 512 == 0:
                print(f"{study} {'training variants' if augmented else 'original images'} {start + len(batch)}/{len(ids)}", flush=True)
        for (scale, policy), parts in chunks.items():
            banks[scale][policy] = np.concatenate(parts)
    return banks


def fit_head(clean, variant, y, batch_size, epochs, optimizer, dest):
    keras.utils.set_random_seed(SEED)
    model = make_head(clean.shape[1])
    initial = hashlib.sha256(b"".join(w.tobytes() for w in model.get_weights())).hexdigest()
    model.compile(optimizer=OPTIMIZERS[optimizer](learning_rate=0.001),
                  loss="binary_crossentropy", metrics=["accuracy"])
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        use_variant = rng.random(len(y)) < 0.5
        model.reset_metrics()
        for start in range(0, len(y), batch_size):
            idx = order[start:start + batch_size]
            x = clean[idx]
            if variant is not None:
                x = np.where(use_variant[idx, None], variant[idx], x)
            metrics = model.train_on_batch(x, y[idx], return_dict=True)
        history.append(dict(epoch=epoch + 1, **{k: float(v) for k, v in metrics.items()}))
    write_csv(dest / "history.csv", history)
    model.save_weights(dest / "head.weights.h5")
    record = dict(initial_sha256=initial, optimizer=type(model.optimizer).__name__,
                  epochs=epochs, batch_size=batch_size, updates=int(model.optimizer.iterations.numpy()),
                  learning_rate=float(model.optimizer.learning_rate.numpy()), seed=SEED)
    (dest / "run.json").write_text(json.dumps(record, indent=2))
    return model


def evaluate(model, features, manifest, directory, split):
    probs = model(features, training=False).numpy()
    if not np.isfinite(probs).all():
        raise ValueError("Non-finite class probabilities")
    predictions = probs.argmax(axis=1)
    rows = [dict(path=r["path"], true_class=int(r["class_id"]), predicted_class=int(p),
                 defect_probability=float(v[0]), normal_probability=float(v[1]))
            for r, p, v in zip(manifest, predictions, probs, strict=True)]
    write_csv(directory / f"{split}_predictions.csv", rows)
    # Recount from disk, checking class order, sample identity and probability threshold.
    saved = read_rows(directory / f"{split}_predictions.csv")
    correct = tp = positives = predicted_defects = 0
    for row, source in zip(saved, manifest, strict=True):
        label = int(source["class_id"])
        pred = int(float(row["normal_probability"]) > float(row["defect_probability"]))
        assert row["path"] == source["path"] and label == int(row["true_class"])
        assert pred == int(row["predicted_class"])
        correct += int(pred == label)
        tp += int(label == pred == 0)
        positives += int(label == 0)
        predicted_defects += int(pred == 0)
    return dict(correct=correct, total=len(saved), accuracy_pct=100 * correct / len(saved),
                defect_detected=tp, defect_total=positives, recall_pct=100 * tp / positives,
                predicted_defects=predicted_defects)


def train_study(study, encode):
    root = OUT / study
    files, _, classes, labels, splits = inventory(study, OUT)
    baseline_split = ROOT / "results_efficientnetv2m" / study / "split.csv"
    if baseline_split.read_bytes() != (root / "split.csv").read_bytes():
        raise ValueError("Split differs from reproduction")
    assert classes[0] in {"badcylinder", "Badaug", "defected"}
    manifest = read_rows(root / "split.csv")
    val_rows = [r for r in manifest if r["split"] == "validation"]
    banks = prepare_features(study, files, splits, encode)
    train, val, test = [splits[s] for s in ("train", "validation", "test")]
    _, batch, epochs, _ = STUDIES[study]
    y = keras.utils.to_categorical(labels[train], 2)
    rows = []
    for scale, bank in banks.items():
        clean = bank["none"]
        # The frozen backbone has no fitted statistics from the held-out images.
        np.save(root / f"test_features_{scale}.npy", clean[test])
        for policy in POLICIES:
            for optimizer in OPTIMIZERS:
                dest = root / scale / policy / optimizer
                model = fit_head(clean[train], None if policy == "none" else bank[policy],
                                 y, batch, epochs, optimizer, dest)
                metrics = evaluate(model, clean[val], val_rows, dest, "validation")
                rows.append(dict(study=study, scale=scale, policy=policy, optimizer=optimizer, **metrics))
        print(f"{study} {scale} completed", flush=True)
    write_csv(root / "validation.csv", rows)
    return rows


def choose(validation):
    # Equal weight per study and optimizer; prefer recall on accuracy ties, then simplicity.
    ranked = []
    for policy in POLICIES:
        group = [r for r in validation if r["scale"] == "keras" and r["policy"] == policy]
        assert len(group) == len(STUDIES) * len(OPTIMIZERS)
        ranked.append(dict(policy=policy, validation_accuracy_pct=np.mean([r["accuracy_pct"] for r in group]),
                           validation_recall_pct=np.mean([r["recall_pct"] for r in group])))
    ordered = sorted(ranked, key=lambda r: (-r["validation_accuracy_pct"], -r["validation_recall_pct"]))
    choice = dict(selected=ordered[0]["policy"], best_augmented=next(r["policy"] for r in ordered if r["policy"] != "none"),
                  validation_ranking=ordered)
    (OUT / "selection.json").write_text(json.dumps(choice, indent=2))
    return choice


def test_models():
    # Evaluate all prespecified contrasts only after the validation choice has been recorded.
    rows = []
    for study in STUDIES:
        root = OUT / study
        manifest = [r for r in read_rows(root / "split.csv") if r["split"] == "test"]
        for scale in SCALES:
            x = np.load(root / f"test_features_{scale}.npy")
            for policy in POLICIES:
                for optimizer in OPTIMIZERS:
                    dest = root / scale / policy / optimizer
                    model = make_head(x.shape[1])
                    model.load_weights(dest / "head.weights.h5")
                    rows.append(dict(study=study, scale=scale, policy=policy, optimizer=optimizer,
                                     **evaluate(model, x, manifest, dest, "test")))
    write_csv(OUT / "test.csv", rows)


if __name__ == "__main__":
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    weight_path = ROOT / ".keras" / "models" / "efficientnetv2-m_notop.h5"
    if not weight_path.is_file():
        raise FileNotFoundError(f"Missing {weight_path}; run python main.py prepare first")
    base = keras.applications.EfficientNetV2M(include_top=False, weights=str(weight_path), input_shape=(224, 224, 3))
    base.trainable = False
    encoder = keras.Model(base.input, keras.layers.Flatten()(keras.layers.AveragePooling2D((4, 4))(base.output)))

    @tf.function(input_signature=[tf.TensorSpec((None, 224, 224, 3), tf.float32)])
    def encode(x):
        return encoder(x, training=False)

    OUT.mkdir(exist_ok=True)
    validation = []
    for study in STUDIES:
        validation.extend(train_study(study, encode))
    write_csv(OUT / "validation.csv", validation)
    print(choose(validation), flush=True)
    test_models()
