"""Analyse baseline score gaps for ambiguous group-size patterns (3,1), (2,2), etc."""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt
from mp2features import build

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
Xtr = (tr['emb'] - tr['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', random_state=0).fit(sc.transform(Xtr), tr['labels'].reshape(-1))


def scores(d):
    X = (d['emb'] - d['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
    p = clf.predict_proba(sc.transform(X))[:, 1].reshape(-1, 4)
    # mask unparseable
    p = np.where(d['labels'].reshape(-1, 4) >= 0, p, p)  # no-op
    for i in range(len(p)):
        for c in range(4):
            if d['answers'][i, c] is None:
                p[i, c] = -np.inf
    return p


def qinfo(d, i, p):
    f = build(split, d) if False else None
    ans = d['answers'][i]
    groups = {}
    for c, a in enumerate(ans):
        key = ('a', a) if a is not None else ('none', c)
        groups.setdefault(key, []).append(c)
    keys = list(groups)
    gsize = np.array([len(groups[k]) for k in keys])
    gscore = np.array([max(p[i, c] for c in groups[k]) for k in keys])
    lab = np.array([int(any(d['labels'][i, c] for c in groups[k])) for k in keys])
    return gsize, gscore, lab


for name, d in [('validation', va), ('test', te)]:
    p = scores(d); nc = ncorrect(d)
    print(f'===== {name} =====')
    buckets = defaultdict(list)
    for i in range(len(d['bank'])):
        gs, gsc, lab = qinfo(d, i, p)
        order = np.argsort(-gs)
        pattern = tuple(sorted(gs.tolist(), reverse=True))
        # gap between best-support group and best-score group, etc.
        imax = int(np.argmax(gs))
        # sorted groups
        best_support = gsc[imax]
        other = np.delete(gsc, imax)
        gap = best_support - other.max() if len(other) else np.inf
        # whether support group is correct
        buckets[pattern].append((gap, lab[imax], nc[i]))
    for pat, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        if len(items) < 5: continue
        gaps = np.array([x[0] for x in items])
        labs = np.array([x[1] for x in items])
        print(f'  pattern={pat} n={len(items)} support_correct={labs.mean()*100:5.1f}% gap: '
              f'min={gaps.min():6.2f} p10={np.percentile(gaps,10):6.2f} med={np.median(gaps):6.2f} p90={np.percentile(gaps,90):6.2f}')
        # best threshold on gap: choose support group if gap>t else best-score group
        # only compute for ambiguous patterns
        best = (0, 0)
        for t in np.unique(np.round(gaps, 2)):
            acc = ((gaps > t) == (labs == 1)).mean()
            if acc > best[0]:
                best = (acc, t)
        print(f'      threshold rule acc={best[0]*100:5.1f}% at t={best[1]:.2f}')
