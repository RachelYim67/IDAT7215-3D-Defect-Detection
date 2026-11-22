"""Benchmark end-to-end inference and produce matched Grad-CAM comparisons."""
import json
import time
from collections import Counter

import cv2
import numpy as np
from run import (OUT, ROOT, STUDIES, VARIANTS, read_csv, write_csv, data_for, load_head, make_encoder,
                 pixel_batch, pooled, manifest, save_json)
import tensorflow as tf
from tensorflow import keras


def gradcam(maps, logits_model):
    # Intermediate tensors are watched explicitly, so the frozen backbone needs no gradient history.
    # For F, each map's heatmap represents that scale's direct branch contribution.
    with tf.GradientTape() as tape:
        tape.watch(maps)
        logits = logits_model(pooled(maps), training=False)
        score = logits[:, 0]
    grads = tape.gradient(score, maps, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    heatmaps = []
    for activation, grad in zip(maps, grads):
        heat = tf.nn.relu(tf.reduce_sum(activation * tf.reduce_mean(grad, axis=(1, 2), keepdims=True), axis=-1))[0].numpy()
        heatmaps.append(heat / max(float(heat.max()), 1e-10))
    return heatmaps


def overlay(rgb, heat):
    color = cv2.applyColorMap(np.uint8(np.clip(cv2.resize(heat, (224, 224)), 0, 1) * 255), cv2.COLORMAP_TURBO)
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(bgr, 0.60, color, 0.40, 0) if heat.max() > 0 else bgr


def label(img, text, x, y, size=0.51, color=(40, 45, 50)):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def explain(encoder, study, choices):
    rows, paths, y, split = manifest(study)
    x = np.load(OUT / study / "features.npy")
    test = split["test"]
    heads = {v: load_head(study, v, choices[f"{study}:{v}"], 42)[0] for v in ("B", "F")}
    probs = {v: head(x[test], training=False).numpy() for v, head in heads.items()}
    pred = {v: p.argmax(axis=1) for v, p in probs.items()}
    # Deterministic non-cherry-picked selection: first test case in each outcome stratum.
    masks = [((y[test] == 0) & (pred["B"] == 1) & (pred["F"] == 0), "Recovered defect"),
             ((y[test] == 0) & (pred["B"] == 0) & (pred["F"] == 1), "New missed defect"),
             ((y[test] == 0) & (pred["B"] == 1) & (pred["F"] == 1), "Both missed defect"),
             ((y[test] == 0) & (pred["B"] == 0) & (pred["F"] == 0), "Both detected defect"),
             ((y[test] == 1) & ((pred["B"] == 0) | (pred["F"] == 0)), "False alarm"),
             ((y[test] == 1) & (pred["B"] == 1) & (pred["F"] == 1), "Both normal")]
    picks = [(int(np.flatnonzero(mask)[0]), title) for mask, title in masks if mask.any()]
    cw, rh = 248, 285
    canvas = np.full((118 + rh * len(picks), cw * 5 + 24, 3), 255, np.uint8)
    label(canvas, f"Study {study.title()}: defect-class Grad-CAM (seed 42)", 16, 30, 0.8)
    label(canvas, f"B: {choices[f'{study}:B']} | F: {choices[f'{study}:F']} | heatmaps are relative within each branch, not defect masks", 16, 58, 0.53)
    for j, title in enumerate(("Input", "B: deep", "F: shallow", "F: middle", "F: deep")):
        label(canvas, title, 16 + cw*j, 94, 0.64)
    logits = {v: keras.Model(head.input, head.get_layer("logits").output) for v, head in heads.items()}
    output = []
    for j, (pos, title) in enumerate(picks):
        idx = test[pos]
        batch = pixel_batch([paths[idx]])
        maps = encoder(batch, training=False)
        features = pooled(maps)
        np.testing.assert_allclose(features.numpy(), x[idx:idx+1], rtol=2e-3, atol=2e-3)
        for v, head in heads.items():
            np.testing.assert_allclose(head(features, training=False).numpy()[0], probs[v][pos], atol=2e-3, rtol=2e-3)
        b = gradcam(maps, logits["B"])[-1]
        f = gradcam(maps, logits["F"])
        pics = [cv2.cvtColor(batch[0].astype(np.uint8), cv2.COLOR_RGB2BGR), overlay(batch[0], b), *[overlay(batch[0], m) for m in f]]
        sy = 118 + j*rh
        for k, pic in enumerate(pics):
            sx = 16 + k*cw
            canvas[sy:sy+224, sx:sx+224] = pic
        label(canvas, f"{title}: {rows[idx]['path']}", 16, sy+244, 0.55)
        label(canvas, f"Defect probability: B {probs['B'][pos,0]:.3f} | F {probs['F'][pos,0]:.3f}", 16, sy+265, 0.5)
        output.append(dict(path=rows[idx]["path"], outcome=title, true_class=int(y[idx]), b_probability=float(probs["B"][pos,0]), f_probability=float(probs["F"][pos,0])))
    assert cv2.imwrite(str(OUT / f"gradcam_{study}.png"), canvas)
    write_csv(OUT / f"gradcam_{study}_cases.csv", output)


def main():
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(42)
    chosen = json.loads((OUT / "selection.json").read_text())["optimizers"]
    encoder, params = make_encoder()
    bench = []
    # Use the very same images for every configuration; record CPU single-image latency incl. backbone.
    rows, paths, _, split = manifest("three")
    inputs = [pixel_batch([paths[i]]) for i in split["test"][:12]]
    for variant in VARIANTS:
        model, _ = load_head("three", variant, chosen[f"three:{variant}"], 42)

        @tf.function(input_signature=[tf.TensorSpec((1,224,224,3), tf.float32)])
        def full(inp):
            return model(pooled(encoder(inp, training=False)), training=False)

        for x in inputs[:3]:
            full(x).numpy()
        times = []
        for x in inputs:
            start = time.perf_counter()
            full(x).numpy()
            times.append((time.perf_counter()-start)*1000)
        # Correct A and B have identical architecture; timing benchmark is for corrected models.
        bench.append(dict(variant=variant, optimizer=chosen[f"three:{variant}"], head_parameters=model.count_params(),
                          backbone_parameters=params, total_parameters=params+model.count_params(),
                          mean_ms=float(np.mean(times)), median_ms=float(np.median(times)), p90_ms=float(np.percentile(times, 90)), n=len(times)))
        print(f"{variant}: {bench[-1]['median_ms']:.1f} ms median", flush=True)
    write_csv(OUT / "inference_benchmark.csv", bench)
    for study in STUDIES:
        explain(encoder, study, chosen)
        print(f"{study}: matched Grad-CAM saved", flush=True)


if __name__ == "__main__":
    main()
