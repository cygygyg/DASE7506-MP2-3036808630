"""Experiment 1: reproduce baseline, compare group pooling, training subsets/weights."""
import sys, io
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from mp2lib import load_split, ncorrect, eval_choice, test_like_metric, fmt, group_index

tr = load_split('train'); va = load_split('validation'); te = load_split('test')


def centre(emb):
    return emb - emb.mean(axis=1, keepdims=True)


Xtr = centre(tr['emb']).reshape(-1, 1024)
Xva = centre(va['emb']).reshape(-1, 1024)
Xte = centre(te['emb']).reshape(-1, 1024)
ytr = tr['labels'].reshape(-1)
yva = va['labels'].reshape(-1)
yce = te['labels'].reshape(-1)
nc_tr = ncorrect(tr); nc_va = ncorrect(va); nc_te = ncorrect(te)

# group ids per question
gva = [group_index(va['answers'][i])[0] for i in range(len(va['bank']))]
gte = [group_index(te['answers'][i])[0] for i in range(len(te['bank']))]


def agg_choice(scores, gids, mode, size_pen=0.0):
    N = len(gids)
    out = np.zeros(N, dtype=np.int64)
    for qi in range(N):
        g = gids[qi]
        G = g.max() + 1
        aggs = []
        for k in range(G):
            s = scores[qi][g == k]
            if mode == 'max': v = s.max()
            elif mode == 'mean': v = s.mean()
            elif mode == 'sum': v = s.sum()
            elif mode == 'logsumexp':
                m = s.max(); v = m + np.log(np.exp(s - m).sum())
            elif mode == 'top2': v = np.sort(s)[-2:].mean() if len(s) > 1 else s[0]
            elif mode == 'mean+size': v = s.mean() + size_pen * len(s)
            else: raise ValueError(mode)
            aggs.append(v)
        best = int(np.argmax(aggs))
        out[qi] = int(np.where(g == best)[0][0])
    return out


print('--- baseline reproduction (all train, C=1) ---')
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', random_state=0).fit(sc.transform(Xtr), ytr)
for name, d, X, nc, gids in [('val', va, Xva, nc_va, gva), ('test', te, Xte, nc_te, gte)]:
    p = clf.predict_proba(sc.transform(X))[:, 1].reshape(-1, 4)
    for mode in ('max', 'mean', 'sum', 'logsumexp', 'top2'):
        ch = agg_choice(p, gids, mode)
        r = eval_choice(ch, d, nc)
        extra = f' testlike={test_like_metric(r):5.2f}' if name == 'val' else ''
        print(fmt(r, f'{name} {mode}') + extra)

print('\n--- training on hard-like subsets (max pooling) ---')
for subset in ('all', 'le2', 'eq1', 'eq2', 'nonagree'):
    if subset == 'all':
        m = np.ones(len(nc_tr), bool)
    elif subset == 'le2':
        m = nc_tr <= 2
    elif subset == 'eq1':
        m = nc_tr == 1
    elif subset == 'eq2':
        m = nc_tr == 2
    else:
        # exclude questions where all parseable candidates agree (trivial)
        m = np.array([len(set(a for a in tr['answers'][i] if a)) >= 2 for i in range(len(nc_tr))])
    mm = np.repeat(m, 4)
    for C in (0.1, 0.3, 1.0):
        sc2 = StandardScaler().fit(Xtr[mm])
        clf2 = LogisticRegression(C=C, max_iter=5000, solver='lbfgs', random_state=0).fit(sc2.transform(Xtr[mm]), ytr[mm])
        pv = clf2.predict_proba(sc2.transform(Xva))[:, 1].reshape(-1, 4)
        rv = eval_choice(agg_choice(pv, gva, 'mean'), va, nc_va)
        print(fmt(rv, f'train={subset} C={C}') + f' testlike={test_like_metric(rv):5.2f}')
