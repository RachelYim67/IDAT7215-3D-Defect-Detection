"""Audit the augmentation experiment and write paired comparisons for the selected augmentation."""
import csv
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "augmentation_study"


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    test = read(OUT / "test.csv")
    validation = read(OUT / "validation.csv")
    selection = json.loads((OUT / "selection.json").read_text())
    policy = selection["best_augmented"]
    studies = ("one", "two", "three")
    optimizers = ("Adam", "SGD", "RMSprop")
    for split, summaries in (("test", test), ("validation", validation)):
        ids = {(r["study"], r["optimizer"], r["scale"], r["policy"]) for r in summaries}
        assert len(ids) == len(summaries) == 54
        for r in summaries:
            root = OUT / r["study"]
            run = root / r["scale"] / r["policy"] / r["optimizer"]
            manifest = [row for row in read(root / "split.csv") if row["split"] == split]
            pred = read(run / f"{split}_predictions.csv")
            correct = detected = defects = 0
            for m, p in zip(manifest, pred, strict=True):
                y = int(m["class_id"])
                probs = [float(p[k]) for k in ("defect_probability", "normal_probability")]
                assert math.isclose(sum(probs), 1.0, abs_tol=1e-5)
                guess = probs.index(max(probs))
                assert p["path"] == m["path"] and int(p["true_class"]) == y and guess == int(p["predicted_class"])
                correct += guess == y
                defects += y == 0
                detected += y == guess == 0
            assert (correct, len(manifest), detected, defects) == tuple(int(r[k]) for k in
                    ("correct", "total", "defect_detected", "defect_total"))
    ranks = []
    for name in ("none", "geometry", "geometry_light"):
        subset = [r for r in validation if r["scale"] == "keras" and r["policy"] == name]
        ranks.append((name, sum(float(r["accuracy_pct"]) for r in subset) / 9,
                      sum(float(r["recall_pct"]) for r in subset) / 9))
    ranks.sort(key=lambda item: (-item[1], -item[2]))
    assert selection["selected"] == ranks[0][0]
    assert policy == next(name for name, _, _ in ranks if name != "none")
    result = []
    for study in studies:
        study_root = OUT / study
        records = [json.loads(p.read_text()) for p in study_root.glob("*/*/*/run.json")]
        assert len(records) == 18
        assert len({(r["initial_sha256"], r["updates"], r["epochs"], r["seed"]) for r in records}) == 1
        for opt in optimizers:
            for scale in ("original", "keras"):
                groups = {r["policy"]: r for r in test if (r["study"], r["optimizer"], r["scale"]) ==
                          (study, opt, scale)}
                base, aug = groups["none"], groups[policy]
                folder = study_root / scale
                a, b = (read(folder / p / opt / "test_predictions.csv") for p in ("none", policy))
                gained = lost = 0
                for before, after in zip(a, b, strict=True):
                    assert before["path"] == after["path"] and before["true_class"] == after["true_class"]
                    was = before["true_class"] == before["predicted_class"]
                    now = after["true_class"] == after["predicted_class"]
                    gained += now and not was
                    lost += was and not now
                assert gained - lost == int(aug["correct"]) - int(base["correct"])
                result.append(dict(study=study, optimizer=opt, scale=scale, policy=policy,
                                   total=base["total"], correct_before=base["correct"], correct_after=aug["correct"],
                                   accuracy_before_pct=base["accuracy_pct"], accuracy_after_pct=aug["accuracy_pct"],
                                   accuracy_delta_pp=round(float(aug["accuracy_pct"])-float(base["accuracy_pct"]), 4),
                                   recall_before_pct=base["recall_pct"], recall_after_pct=aug["recall_pct"],
                                   gained=gained, lost=lost))
    with (OUT / "paired_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print("Verified 54 validation and 54 test results.")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
