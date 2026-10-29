"""Per-pattern breakdown: baseline vs support vs best-singleton."""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, fmt

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
Xtr = (tr['emb'] - tr['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', random_state=0).fit(sc.transform(Xtr), tr['labels'].reshape(-1))


def scores(d):
    X = (d['emb'] - d['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
    p = clf.predict_proba(sc.transform(X))[:, 1].reshape(-1, 4)
    for i in range(len(p)):
        for c in range(4):
            if d['answers'][i, c] is None: p[i, c] = -1e9
    return p


for name, d in (('validation', va), ('test', te)):
    p = scores(d); nc = ncorrect(d)
    print(f'===== {name} =====')
    tab = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # pattern -> method -> [ok,n,nc1]
    for i in range(len(d['bank'])):
        ans = d['answers'][i]; groups = {}
        for c, a in enumerate(ans):
            key = ('a', a) if a is not None else ('none', c)
            groups.setdefault(key, []).append(c)
        keys = list(groups); sizes = np.array([len(groups[k]) for k in keys])
        gs = np.array([max(p[i, c] for c in groups[k]) for k in keys])
        labs = np.array([int(any(d['labels'][i, c] for c in groups[k])) for k in keys])
        pat = tuple(sorted(sizes.tolist(), reverse=True))
        methods = {
            'base': int(np.argmax(gs)),
            'support': int(np.argmax(sizes)),
            'best_single': int(max([g for g in range(len(keys)) if sizes[g] == 1], key=lambda g: gs[g], default=int(np.argmax(gs)))),
        }
        for m, g in methods.items():
            tab[pat][m][0] += labs[g]; tab[pat][m][1] += 1
        tab[pat]['nc'][0] += int(nc[i] == 1); tab[pat]['nc'][1] += 1
    print(f'{"pattern":18s} {"n":>4s} {"#1%":>6s} {"base":>6s} {"support":>8s} {"best_single":>11s}')
    for pat, methods in sorted(tab.items(), key=lambda kv: -kv[1]['nc'][1]):
        n = methods['nc'][1]
        print(f'{str(pat):18s} {n:4d} {100*methods["nc"][0]/n:6.1f} '
              f'{100*methods["base"][0]/n:6.1f} {100*methods["support"][0]/n:8.1f} {100*methods["best_single"][0]/n:11.1f}')
