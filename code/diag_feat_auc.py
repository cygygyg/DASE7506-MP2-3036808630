"""AUC / rule accuracy of handcrafted features at group and candidate level."""
import sys, io
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect
from mp2features import build


def auc(score, label):
    s = np.asarray(score, float); y = np.asarray(label, int)
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = y.sum(); n0 = len(y)-n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    return (ranks[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)


def group_dataset(split):
    d = load_split(split); f = build(split, d); nc = ncorrect(d)
    rows = []          # (qi, gi, label, feats)
    for qi, rec in enumerate(f['records']):
        gids = rec['gids']; gf = f['group'][qi]
        for gi in range(rec['G']):
            if not rec['valid'][gi]:
                continue
            members = np.where(gids == gi)[0]
            lab = int(d['labels'][qi, members].max())
            rows.append((qi, gi, lab, gf[gi]))
    return d, f, nc, rows


def eval_feature(rows, idx, nc, name):
    # AUC overall
    y = np.array([r[2] for r in rows]); s = np.array([r[3][idx] for r in rows])
    a = auc(s, y)
    # selection accuracy per question
    byq = {}
    for qi, gi, lab, feat in rows:
        byq.setdefault(qi, []).append((feat[idx], lab, gi))
    correct = {1: [0,0], 2: [0,0], 3: [0,0]}
    for qi, items in byq.items():
        best = max(items, key=lambda x: x[0])
        k = nc[qi]
        if k in correct:
            correct[k][0] += best[1]; correct[k][1] += 1
    accs = {k: 100*v[0]/v[1] for k, v in correct.items() if v[1]}
    tl = sum(151/467*accs.get(1,0) + 145/467*accs.get(2,0) + 171/467*accs.get(3,0) for _ in [0])
    return a, accs, tl


for split in ('train', 'validation', 'test'):
    d, f, nc, rows = group_dataset(split)
    print(f'===== {split} =====')
    res = []
    for i, name in enumerate(f['group_names']):
        a, accs, tl = eval_feature(rows, i, nc, name)
        res.append((abs(a-0.5), a, name, accs, tl))
    res.sort(reverse=True)
    for dev, a, name, accs, tl in res[:25]:
        print(f'  {name:28s} AUC={a:.3f} testlike={tl:5.1f} #1={accs.get(1,0):5.1f} #2={accs.get(2,0):5.1f} #3={accs.get(3,0):5.1f}')
