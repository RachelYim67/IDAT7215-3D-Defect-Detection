"""Audit saved predictions and print paper-style tables, using only Python stdlib.

Accuracy is calculated from final-weight image predictions, with the original
paper's Eq. (11) normal 95% CI half-width. Other metrics are point estimates
with defect=positive and macro AP across two classes, consistently for all studies.
"""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_results"
STUDIES = ("one", "two", "three")
OPTIMIZERS = ("Adam", "SGD", "RMSprop")
SPLITS = ("train", "validation", "test")
VARIANTS = {"B":"Corrected EfficientNetV2M", "C":"Deep GAP", "D":"Multi-scale GAP", "E":"Fixed dual pooling", "F":"MSAF-EfficientNetV2M"}
# Accuracy as printed (train, validation, test), not computed from our images.
PAPER = {
    "one": {"InceptionResNetV2": (("0.99 ± 0.01","0.99 ± 0.04","0.98 ± 0.04"),("0.95 ± 0.03","1.00 ± 0.00","1.00 ± 0.00"),("1.00 ± 0.00","1.00 ± 0.00","1.00 ± 0.00")),
            "EfficientNetV2M": (("0.52 ± 0.07","0.41 ± 0.19","0.49 ± 0.13"),)*3},
    "two": {"InceptionResNetV2": (("1.00 ± 0.00","1.00 ± 0.00","1.00 ± 0.00"),("0.99 ± 0.00","1.00 ± 0.00","0.99 ± 0.01"),("1.00 ± 0.00","1.00 ± 0.00","1.00 ± 0.00")),
            "EfficientNetV2M": (("0.69 ± 0.02","0.71 ± 0.05","0.67 ± 0.04"),)*3},
    "three": {"InceptionResNetV2": (("0.99 ± 0.01","0.99 ± 0.02","0.98 ± 0.01"),("0.84 ± 0.02","0.84 ± 0.06","0.83 ± 0.02"),("0.92 ± 0.02","0.92 ± 0.04","0.93 ± 0.01")),
              "EfficientNetV2M": (("0.52 ± 0.03","0.51 ± 0.08","0.50 ± 0.02"),)*3},
}
METRICS = ("accuracy","precision","recall","f1","mcc","kappa","auc","map")


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def ranked(y, score):
    """AUC with ties (Mann-Whitney), AP via threshold groups."""
    pairs=sorted(zip(score,y),reverse=True)
    pos=sum(y); neg=len(y)-pos
    assert pos and neg
    tp=fp=0; auc=ap=0.0; i=0
    while i<len(pairs):
        j=i+1
        while j<len(pairs) and pairs[j][0]==pairs[i][0]: j+=1
        p=sum(v for _,v in pairs[i:j]); n=j-i-p
        auc += n*(tp+0.5*p)
        tp+=p; fp+=n
        ap += (p/pos)*tp/(tp+fp)
        i=j
    return auc/(pos*neg),ap


def calculate(rows, manifest, original):
    expected={r['path']:r for r in manifest}
    assert len(rows)==len(expected) and {r['path'] for r in rows}==set(expected)
    y=[]; p=[]; pred=[]
    classes={int(r['class_id']):r['label'] for r in manifest}
    for r in rows:
        truth=int(expected[r['path']]['class_id'])
        a,b=(float(r[k]) for k in (('probability_0','probability_1') if original else ('defect_probability','normal_probability')))
        assert all(0<=v<=1 for v in (a,b)) and math.isclose(a+b,1,abs_tol=1e-5)
        chosen=0 if a>=b else 1
        if original:
            assert r['label']==classes[truth] and r['prediction']==classes[chosen]
        else:
            assert int(r['true_class'])==truth and int(r['predicted_class'])==chosen
        y.append(int(truth==0)); p.append(a); pred.append(int(chosen==0))
    tp=sum(a==b==1 for a,b in zip(y,pred)); tn=sum(a==b==0 for a,b in zip(y,pred))
    fp=sum(a==0 and b==1 for a,b in zip(y,pred)); fn=len(y)-tp-tn-fp
    n=len(y); acc=(tp+tn)/n
    d=lambda a,b:a/b if b else 0.0
    pe=((tp+fp)*(tp+fn)+(tn+fn)*(tn+fp))/n**2
    auc,ap=ranked(y,p)
    # Score for normal class is 1-p up to float rounding; read stored normal scores.
    q=[float(r['probability_1' if original else 'normal_probability']) for r in rows]
    _,ap_normal=ranked([1-a for a in y],q)
    return dict(n=n,correct=tp+tn,tp=tp,fp=fp,fn=fn,tn=tn,accuracy=acc,accuracy_pct=100*acc,
                ci95_halfwidth=1.96*math.sqrt(acc*(1-acc)/n),precision=d(tp,tp+fp),recall=d(tp,tp+fn),
                f1=d(2*tp,2*tp+fp+fn),mcc=d(tp*tn-fp*fn,math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),
                kappa=d(acc-pe,1-pe),auc=auc,map=(ap+ap_normal)/2)


def build():
    results=[]
    for study in STUDIES:
        for opt in OPTIMIZERS:
            for model,directory,variant in (("InceptionResNetV2","results","reference"),("EfficientNetV2M","results_efficientnetv2m","A")):
                root=ROOT/directory/study
                manifest=read(root/'split.csv')
                for split in SPLITS:
                    rows=read(root/opt/f'{split}_predictions.csv')
                    m=calculate(rows,[r for r in manifest if r['split']==split],True)
                    results.append(dict(study=study,optimizer=opt,algorithm=model,variant=variant,seed=42,split=split,
                      paper_accuracy_reference=PAPER[study][model][OPTIMIZERS.index(opt)][SPLITS.index(split)],**m))
            root=ROOT/'innovation'/'results'/study
            manifest=read(root/'split.csv')
            for variant,model in VARIANTS.items():
                for seed in (42,43,44):
                    for split in SPLITS:
                        rows=read(root/variant/opt/str(seed)/f'{split}_predictions.csv')
                        m=calculate(rows,[r for r in manifest if r['split']==split],False)
                        results.append(dict(study=study,optimizer=opt,algorithm=model,variant=variant,seed=seed,split=split,paper_accuracy_reference='',**m))
    assert len(results)==459
    return results


def tables(results):
    lines=["# 论文格式实测结果", "", "每张 Study × 优化器结果表使用种子 42、最终模型权重，按原论文表 2–10 展示 Train / Validation / Test。Accuracy 为比例 ± 正态近似 95% 置信区间半宽（论文式 11），另给出两位小数百分比和正确张数。Precision、Recall、F1、MCC、Kappa、AUC 和 mAP 为实测点估计，以缺陷为正类，mAP 为两个类别的 AP 平均值。Paper accuracy 为论文原始印刷值，单独列示；本次区间按实际样本数重新计算。正态区间在极端准确率下可能不可靠（100% 时半宽为零），且不是三种子标准差。三种子结果见末表和 CSV。",""]
    for study in STUDIES:
        for opt in OPTIMIZERS:
            table=2+3*STUDIES.index(study)+OPTIMIZERS.index(opt)
            lines.extend([f"## Study {study.title()} · {opt} (reference Table {table})", "", "| Algorithm | Dataset | Accuracy (95% CI half-width) | Accuracy (%) | Precision | Recall | F1-score | MCC | Kappa | AUC | mAP | Correct / n | Paper accuracy |","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
            for r in results:
                if r['study']!=study or r['optimizer']!=opt or r['seed']!=42: continue
                line=[r['algorithm'],r['split'].title(),f"{r['accuracy']:.2f} ± {r['ci95_halfwidth']:.2f}",f"{r['accuracy_pct']:.2f}",*[f"{r[k]:.2f}" for k in METRICS[1:]],f"{r['correct']}/{r['n']}",r['paper_accuracy_reference'] or '—']
                lines.append('| '+' | '.join(line)+' |')
            lines.append('')
    groups=defaultdict(list)
    for r in results:
        if r['variant'] in VARIANTS and r['split']=='test':
            groups[(r['study'],r['variant'],r['optimizer'])].append(r)
    summary=[]
    for (study,variant,opt),group in groups.items():
        summary.append(dict(study=study,variant=variant,optimizer=opt,runs=len(group),**{f'{m}_{k}':fn([r[m] for r in group]) for m in METRICS for k,fn in [('mean',mean),('sd',stdev)]}))
    write(OUT/'three_seed_test_summary.csv',summary)
    lines.extend(['## 三个种子测试集对比（准确率均值 ± 样本标准差）','', '| Study | Variant | Adam | SGD | RMSprop |','|---|---|---:|---:|---:|'])
    for study in STUDIES:
        for variant in VARIANTS:
            row=[study.title(),variant]
            for opt in OPTIMIZERS:
                s=next(r for r in summary if (r['study'],r['variant'],r['optimizer'])==(study,variant,opt))
                row.append(f"{100*s['accuracy_mean']:.2f}% ± {100*s['accuracy_sd']:.2f}%")
            lines.append('| '+' | '.join(row)+' |')
    (OUT/'paper_style_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    rows=build()
    write(OUT/'all_metrics.csv',rows)
    tables(rows)
    print(f"Verified {len(rows)} split-level prediction files. Results: {OUT}")
    for r in rows:
        if r['seed']==42 and r['split']=='test' and r['variant'] in ('A','B','F'):
            print(f"Study {r['study'].title():5} {r['optimizer']:7} {r['variant']} | Accuracy {r['accuracy']:.2f} ± {r['ci95_halfwidth']:.2f} ({100*r['accuracy']:.2f}%, {r['correct']}/{r['n']})")


if __name__=='__main__': main()
