"""Recount predictions, cross-check all nine runs and generate the selected model's summary CSV."""
import argparse
import csv
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results"
OPTIMIZERS = ("Adam", "SGD", "RMSprop")
# Table 2 through Table 10, percent, rounded as in the publication.
PAPER = {"one": (98, 100, 100), "two": (100, 99, 100), "three": (98, 83, 93)}
# Train/validation/test sizes follow the notebook's two train_test_split calls.
EXPECTED = {"one": (189, 27, 55, 8, 30), "two": (2111, 308, 598, 50, 50),
            "three": (1089, 159, 309, 30, 30)}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def recount(directory, split, labels, classes):
    predictions = read_csv(directory / f"{split}_predictions.csv")
    check(len(predictions) == len(labels) and {r["path"] for r in predictions} == labels.keys(),
          f"{directory}: {split} file list differs from split manifest")
    correct = 0
    for r in predictions:
        path = r["path"]
        probs = [float(r[f"probability_{i}"]) for i in (0, 1)]
        check(all(0 <= p <= 1 for p in probs) and math.isclose(sum(probs), 1, abs_tol=1e-5),
              f"Invalid probabilities: {path}")
        prediction = classes[probs.index(max(probs))]
        check(r["label"] == labels[path] and prediction == r["prediction"], path)
        correct += int(prediction == labels[path])
    return correct, len(labels)


def main(model="InceptionResNetV2"):
    efficient = model == "EfficientNetV2M"
    output = OUT.with_name("results_efficientnetv2m") if efficient else OUT
    paper = {"one": (49, 49, 49), "two": (67, 67, 67), "three": (50, 50, 50)} if efficient else PAPER
    rows = []
    for study, references in paper.items():
        manifest = read_csv(output / study / "split.csv")
        train, val, test, batch, epochs = EXPECTED[study]
        check(len(set(r["path"] for r in manifest)) == len(manifest) == train + val + test,
              f"{study}: duplicate or missing path")
        groups = {s: {r["path"]: r["label"] for r in manifest if r["split"] == s}
                  for s in ("train", "validation", "test")}
        check([len(g) for g in groups.values()] == [train, val, test], study)
        init_hashes = set()
        for name, reference in zip(OPTIMIZERS, references):
            directory = output / study / name
            run = json.loads((directory / "run.json").read_text())
            check(run.get("model", "InceptionResNetV2") == model, str(directory))
            history = read_csv(directory / "history.csv")
            check(run["optimizer"] == name and run["study"] == study and run["batch_size"] == batch
                  and run["epochs"] == epochs and run["seed"] == 42
                  and math.isclose(run["learning_rate"], 0.001, rel_tol=1e-6), str(directory))
            check([int(r["epoch"]) for r in history] == list(range(1, epochs + 1)), str(directory))
            init_hashes.add(run["initial_weights_sha256"])
            measurements = {}
            for split in ("validation", "test"):
                c, n = recount(directory, split, groups[split], run["class_order"])
                check(run[split] == {"correct": c, "n": n}, f"{directory}: {split}")
                measurements[split] = (c, n)
            val_correct, val_total = measurements["validation"]
            val_fraction = val_correct / val_total
            check(math.isclose(float(history[-1]["val_accuracy"]), val_fraction,
                               abs_tol=1e-6), str(directory))
            test_correct, test_total = measurements["test"]
            measured = 100 * test_correct / test_total
            rows.append(dict(study=study, optimizer=name,
                             validation_accuracy_pct=round(100 * val_fraction, 4),
                             test_correct=test_correct, test_total=test_total, test_accuracy_pct=round(measured, 4),
                             paper_test_accuracy_pct=reference,
                             difference_pp=round(measured - reference, 4)))
        check(len(init_hashes) == 1, f"{study}: initial weights differ")
    temp = output / "summary.tmp"
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(output / "summary.csv")
    print((output / "summary.csv").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["InceptionResNetV2", "EfficientNetV2M"], default="InceptionResNetV2")
    main(parser.parse_args().model)
