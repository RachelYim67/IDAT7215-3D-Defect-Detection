"""Evaluate the final saved weights on the training splits for paper-style tables.

No fitting or test-set selection takes place. Historical validation/test predictions
are retained; the training predictions are computed with dropout disabled.
"""
import argparse
import json
import sys
from pathlib import Path
import train
import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = train.ROOT
sys.path.insert(0, str(ROOT / "innovation"))
import run as msaf


def baseline(model_name):
    directory = ROOT / ("results" if model_name == "InceptionResNetV2" else "results_efficientnetv2m")
    weight_name = ("inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5" if model_name == "InceptionResNetV2" else "efficientnetv2-m_notop.h5")
    weight_path = ROOT / ".keras" / "models" / weight_name
    base = train.MODELS[model_name](include_top=False, weights=str(weight_path), input_shape=(224,224,3))
    base.trainable = False
    encoder = keras.Model(base.input, keras.layers.Flatten()(keras.layers.AveragePooling2D((4,4))(base.output)))
    @tf.function(input_signature=[tf.TensorSpec((None,224,224,3),tf.float32)])
    def encode(x):
        return encoder(x, training=False)
    for study in train.STUDIES:
        rows = msaf.read_csv(directory / study / "split.csv")
        train_rows = [r for r in rows if r["split"] == "train"]
        paths = [train.STUDIES[study][0] / r["path"] for r in train_rows]
        features = []
        for i in range(0,len(paths),16):
            features.append(encode(train.load_images(paths[i:i+16])).numpy())
            if i == 0 or (i+16 >= len(paths)) or i % 512 == 0:
                print(f"{model_name} {study}: {min(i+16,len(paths))}/{len(paths)} train images",flush=True)
        x = np.concatenate(features)
        labels = [r["label"] for r in train_rows]
        for optimizer in train.OPTIMIZERS:
            dest = directory / study / optimizer
            record = json.loads((dest / "run.json").read_text())
            head = train.make_head(x.shape[1])
            head.load_weights(dest / "head.weights.h5")
            probs = np.concatenate([head(x[i:i+128], training=False).numpy() for i in range(0,len(x),128)])
            classes = record["class_order"]
            train.write_csv(dest / "train_predictions.csv", [dict(path=r["path"], label=label, prediction=classes[p.argmax()],
                probability_0=float(p[0]), probability_1=float(p[1])) for r,label,p in zip(train_rows,labels,probs,strict=True)])
    del encoder,base
    keras.backend.clear_session()


def innovation():
    for study in train.STUDIES:
        rows,y,splits,x = msaf.data_for(study)
        ids = splits["train"]
        for variant in msaf.VARIANTS:
            for opt in train.OPTIMIZERS:
                for seed in msaf.SEEDS:
                    head,dest = msaf.load_head(study,variant,opt,seed)
                    p = np.concatenate([head(x[ids[i:i+128]],training=False).numpy() for i in range(0,len(ids),128)])
                    msaf.save_predictions(dest,"train",[rows[i] for i in ids],y[ids],p)
        print(f"MSAF {study}: evaluated 45 final heads on train split",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=("all","InceptionResNetV2","EfficientNetV2M","innovation"),default="all")
    model=parser.parse_args().model
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    if model in ("all","innovation"): innovation()
    for m in ("EfficientNetV2M","InceptionResNetV2"):
        if model in (m,"all"): baseline(m)
