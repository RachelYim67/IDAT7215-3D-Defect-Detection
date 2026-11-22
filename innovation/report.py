"""Verify per-image predictions, aggregate the experiments, and draw report figures."""
from collections import defaultdict
from html import escape
from statistics import mean, stdev

from run import OUT, ROOT, SEEDS, STUDIES, OPTIMIZERS, VARIANTS, read_csv, write_csv, metrics, manifest
import json
import hashlib
import numpy as np
from run import save_json

SCORES = ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "average_precision")
TITLES = {"A": "Original", "B": "Corrected", "C": "Deep GAP", "D": "Multi GAP", "E": "Fixed dual", "F": "Adaptive dual"}


def audit_prediction(path, expected, original=False):
    rows = read_csv(path)
    assert len(rows) == len(expected)
    assert [r["path"] for r in rows] == [r["path"] for r in expected]
    y = np.array([int(r["class_id"]) for r in expected])
    if original:
        assert [r["label"] for r in rows] == [r["label"] for r in expected]
        p = np.array([[float(r["probability_0"]), float(r["probability_1"])] for r in rows])
        classes = {int(r["class_id"]): r["label"] for r in expected}
        assert [r["prediction"] for r in rows] == [classes[i] for i in p.argmax(axis=1)]
    else:
        p = np.array([[float(r["defect_probability"]), float(r["normal_probability"])] for r in rows])
        np.testing.assert_array_equal(y, [int(r["true_class"]) for r in rows])
        np.testing.assert_array_equal(p.argmax(axis=1), [int(r["predicted_class"]) for r in rows])
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-5)
    return metrics(y, p), p.argmax(axis=1) == y


def aggregate(rows, keys, numeric=SCORES):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, group in groups.items():
        result.append(dict(zip(keys, key)) | dict(runs=len(group)) |
                      {f"{m}_{stat}": func([float(r[m]) for r in group]) if len(group) > 1 or stat == "mean" else ""
                       for m in numeric for stat, func in (("mean", mean), ("sd", stdev))})
    return result


def svg_chart(rows):
    """Publication-friendly vector chart: same optimizer (Adam) for all designs."""
    w, h = 1160, 690
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#263343}.axis{font-size:12px}.title{font-size:23px;font-weight:bold}.sub{font-size:13px;fill:#566575}</style>',
           '<text x="32" y="37" class="title">EfficientNetV2M: architecture ablation (Adam)</text>',
           '<text x="32" y="61" class="sub">Original: one reproduction run; B–F: means of three seeds, error bars = sample SD</text>']
    colors = ["#a5acb5", "#64748b", "#779ac3", "#497db4", "#28a3a2", "#126c56"]
    for row_idx, metric in enumerate(("accuracy", "recall")):
        top = 112 + row_idx * 270
        for col, study in enumerate(STUDIES):
            left = 65 + col * 370
            svg.append(f'<text x="{left}" y="{top-16}" font-size="15" font-weight="bold">Study {study.title()} · {metric.title()}</text>')
            for tick in (0, 25, 50, 75, 100):
                yy = top + 180 - 1.8 * tick
                svg.append(f'<line x1="{left}" y1="{yy}" x2="{left+310}" y2="{yy}" stroke="#dbe3ea"/>')
                svg.append(f'<text x="{left-9}" y="{yy+4}" text-anchor="end" class="axis">{tick}</text>')
            for i, v in enumerate(TITLES):
                r = next(r for r in rows if r["study"] == study and r["variant"] == v and r["optimizer"] == "Adam")
                value = 100 * float(r[f"{metric}_mean"])
                sd = 100 * float(r[f"{metric}_sd"] or 0)
                x, y = left + 10 + i * 51, top + 180 - 1.8 * value
                svg.append(f'<rect x="{x}" y="{y}" width="36" height="{max(0.8, value*1.8)}" fill="{colors[i]}"/>')
                if sd:
                    high, low = top + 180 - 1.8 * min(100, value + sd), top + 180 - 1.8 * max(0, value - sd)
                    svg.append(f'<path d="M{x+12} {high}h12 M{x+18} {high}V{low} M{x+12} {low}h12" stroke="#152b38" stroke-width="1.4"/>')
                svg.append(f'<text x="{x+18}" y="{max(top+9, y-5)}" class="axis" text-anchor="middle">{value:.1f}</text>')
                svg.append(f'<text x="{x+18}" y="{top+200}" class="axis" text-anchor="middle">{v}</text>')
    for i, (v, name) in enumerate(TITLES.items()):
        x = 32 + i * 186
        svg.extend([f'<rect x="{x}" y="650" width="14" height="14" fill="{colors[i]}"/>', f'<text x="{x+20}" y="662" class="axis">{v} · {escape(name)}</text>'])
    svg.append('</svg>')
    (OUT / "ablation.svg").write_text("\n".join(svg))


def confusion_chart(chosen, test):
    w, h = 1000, 420
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#263343}</style><text x="30" y="30" font-size="20" font-weight="bold">Confusion matrices · seed 42 · validation-selected optimizers</text>',
           '<text x="30" y="53" font-size="12">Rows: actual defect / normal; columns: predicted defect / normal. Counts, not percentages.</text>']
    for j, study in enumerate(STUDIES):
        for k, variant in enumerate(("B", "F")):
            r = next(r for r in test if r["study"] == study and r["variant"] == variant and r["seed"] == 42 and r["optimizer"] == chosen[f"{study}:{variant}"])
            x, y = 28 + j * 325 + k * 154, 110
            svg.append(f'<text x="{x}" y="{y-28}" font-size="14" font-weight="bold">{study.title()} · {variant}</text>')
            svg.append(f'<text x="{x}" y="{y-10}" font-size="11">{r["optimizer"]}</text>')
            counts = (r["tp"], r["fn"], r["fp"], r["tn"])
            for i, count in enumerate(counts):
                xx, yy = x + (i%2)*65, y + (i//2)*65
                fill = "#d0ebdc" if i in (0,3) else "#f8ddc9"
                svg.append(f'<rect x="{xx}" y="{yy}" width="63" height="63" rx="4" fill="{fill}"/>')
                svg.append(f'<text x="{xx+31}" y="{yy+36}" text-anchor="middle" font-size="20">{count}</text>')
            svg.append(f'<text x="{x}" y="{y+152}" font-size="12">Recall: {100*r["recall"]:.1f}%</text>')
            svg.append(f'<text x="{x}" y="{y+170}" font-size="12">F1: {100*r["f1"]:.1f}%</text>')
    svg.append('</svg>')
    (OUT / "confusion_matrices.svg").write_text("\n".join(svg))


def main():
    chosen = json.loads((OUT / "selection.json").read_text())["optimizers"]
    recorded_validation = {(r['study'], r['variant'], r['optimizer'], int(r['seed'])): r for r in read_csv(OUT / "validation.csv")}
    recorded_test = {(r['study'], r['variant'], r['optimizer'], int(r['seed'])): r for r in read_csv(OUT / "test.csv")}
    assert len(recorded_validation) == len(recorded_test) == 135
    test, pairs, checks = [], [], []
    for study in STUDIES:
        rows, _, _, _ = manifest(study)
        expected = {s: [r for r in rows if r["split"] == s] for s in ("validation", "test")}
        correctness = {}
        for variant in ("A", *VARIANTS):
            for opt in OPTIMIZERS:
                for seed in ((42,) if variant == "A" else SEEDS):
                    original = variant == "A"
                    dest = ROOT / "results_efficientnetv2m" / study / opt if original else OUT / study / variant / opt / str(seed)
                    rec = json.loads((dest / "run.json").read_text())
                    if not original:
                        _, batch, epochs, _ = STUDIES[study]
                        n_train = sum(r['split'] == 'train' for r in rows)
                        assert rec['config']['optimizer'] == opt and rec['config']['seed'] == seed
                        assert len(read_csv(dest / 'history.csv')) == epochs
                        assert rec['updates'] == ((n_train + batch - 1) // batch) * epochs
                    for split in expected:
                        m, correct = audit_prediction(dest / f"{split}_predictions.csv", expected[split], original)
                        if not original:
                            recorded = (recorded_validation if split == 'validation' else recorded_test)[(study,variant,opt,seed)]
                            for name, value in m.items():
                                np.testing.assert_allclose(value, float(recorded[name]), atol=1e-8)
                        checks.append(dict(study=study, variant=variant, optimizer=opt, seed=seed, split=split, **m))
                        if split == "test":
                            test.append(dict(study=study, variant=variant, optimizer=opt, seed=seed, **m))
                            correctness[(variant, opt, seed)] = correct
        for opt in OPTIMIZERS:
            for seed in SEEDS:
                for before, after in (("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("B", "F")):
                    a, b = correctness[(before, opt, seed)], correctness[(after, opt, seed)]
                    pairs.append(dict(study=study, optimizer=opt, seed=seed, before=before, after=after,
                                      gained=int(np.sum(~a & b)), lost=int(np.sum(a & ~b)), difference_pp=100*(b.mean()-a.mean())))
    assert len(test) == 144 and len(checks) == 288
    write_csv(OUT / "all_test_runs.csv", test)
    write_csv(OUT / "verified_predictions.csv", checks)
    write_csv(OUT / "paired_ablation.csv", pairs)
    summary = aggregate(test, ("study", "variant", "optimizer"))
    write_csv(OUT / "summary.csv", summary)
    selected = [r for r in summary if r["variant"] != "A" and r["optimizer"] == chosen[f'{r["study"]}:{r["variant"]}']]
    write_csv(OUT / "selected_summary.csv", selected)
    cross = read_csv(OUT / "cross_dataset.csv")
    hashes = {}
    for study in ("one", "three"):
        _, paths, _, _ = manifest(study)
        hashes[study] = {hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    shared = hashes["one"] & hashes["three"]
    save_json(OUT / "cross_data_audit.json", dict(study_one_unique_hashes=len(hashes['one']), study_three_unique_hashes=len(hashes['three']), cross_dataset_identical_hashes=len(shared)))
    assert not shared, "Overlapping image bytes in cross-domain study; disjoint evaluation required"
    for row in cross:
        target_rows, _, _, _ = manifest(row["target"])
        dest = OUT / "cross_dataset" / f'{row["source"]}_to_{row["target"]}' / row["variant"] / row["seed"]
        m, _ = audit_prediction(dest / "target_predictions.csv", target_rows)
        for key in SCORES:
            np.testing.assert_allclose(m[key], float(row[key]))
    write_csv(OUT / "cross_summary.csv", aggregate(cross, ("source", "target", "variant", "optimizer")))
    svg_chart(summary)
    confusion_chart(chosen, test)
    print("Verified 144 test runs / 288 prediction files and 12 cross-dataset runs.")
    print("Validation-selected results (mean ± SD, %):")
    for r in selected:
        print(r["study"], r["variant"], r["optimizer"], *(f'{100*r[m+"_mean"]:.2f} ± {100*r[m+"_sd"]:.2f}' for m in ("accuracy", "recall", "f1")))


if __name__ == "__main__":
    main()
