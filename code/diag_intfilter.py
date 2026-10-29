"""Effect of integer-answer filtering on baseline and pattern structure."""
import sys, io
from fractions import Fraction
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
Xtr = (tr['emb'] - tr['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', random_state=0).fit(sc.transform(Xtr), tr['labels'].reshape(-1))


def is_int(a):
    if a is None: return False
    return Fraction(a).denominator == 1


def scores(d, integer_filter):
    X = (d['emb'] - d['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
    p = clf.predict_proba(sc.transform(X))[:, 1].reshape(-1, 4)
    for i in range(len(p)):
        for c in range(4):
            a = d['answers'][i, c]
            if a is None or (integer_filter and not is_int(a)):
                p[i, c] = -1e9
    return p


for split, d in (('validation', va), ('test', te)):
    print(f'===== {split} =====')
    for filt in (False, True):
        p = scores(d, filt)
        r = eval_choice(p.argmax(1), d)
        print(fmt(r, f'integer_filter={filt}'))
        # pattern table
        tab = defaultdict(lambda: [0, 0])
        for i in range(len(d['bank'])):
            ans = d['answers'][i]; groups = {}
            for c, a in enumerate(ans):
                if a is None or (filt and not is_int(a)): continue
                groups.setdefault(a, []).append(c)
            keys = list(groups); sizes = np.array([len(groups[k]) for k in keys])
            if len(keys) == 0: continue
            gs = np.array([max(p[i, c] for c in groups[k]) for k in keys])
            pick = int(np.argmax(gs))
            pat = tuple(sorted(sizes.tolist(), reverse=True))
            tab[pat][0] += int(d['labels'][i, groups[keys[pick]][0]]); tab[pat][1] += 1
        for pat, v in sorted(tab.items(), key=lambda kv: -kv[1][1])[:6]:
            print(f'    {pat} {v[0]}/{v[1]}={100*v[0]/v[1]:.1f}%')
