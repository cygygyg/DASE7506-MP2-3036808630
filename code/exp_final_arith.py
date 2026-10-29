"""Add a small arithmetic-verification term; select weights on validation."""
import sys, io, pickle
from fractions import Fraction
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, eval_choice, test_like_metric
from arith2 import arith2_features

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
S = pickle.load(open('cache/final_scores.pkl', 'rb'))
ZAv, ZAt, Kv, Kt = S['ZAv'], S['ZAt'], S['Kv'], S['Kt']

def is_int(a): return a is not None and Fraction(a).denominator == 1
def mask(p, d):
    p = p.reshape(-1, 4).copy()
    for i in range(len(p)):
        for c in range(4):
            if not is_int(d['answers'][i, c]): p[i, c] = -1e9
    return p
def z(p, d):
    p = mask(p, d)
    for i in range(len(p)):
        v = p[i][p[i] > -1e8]; m, s = v.mean(), v.std() + 1e-9
        p[i] = (p[i] - m) / s; p[i][p[i] < -1e8] = -1e9
    return p
def arith_mat(d):
    rows = []
    for i in range(len(d['bank'])):
        for c in range(4):
            a = d['answers'][i, c]
            fv = int(Fraction(a)) if is_int(a) else None
            rows.append(list(arith2_features(d['bank'][i]['candidates'][c]['text'], fv).values()))
    return np.array(rows, np.float32)

Atr, Ava, Ate = arith_mat(tr), arith_mat(va), arith_mat(te)
y = tr['labels'].reshape(-1)
sc = StandardScaler().fit(Atr)
clf = LogisticRegression(C=0.3, max_iter=3000).fit(sc.transform(Atr), y)
Vv = z(clf.decision_function(sc.transform(Ava)).reshape(-1, 4), va)
Vt = z(clf.decision_function(sc.transform(Ate)).reshape(-1, 4), te)

best = (0, None)
for a in np.arange(0.10, 0.26, 0.01):
    for b in np.arange(0.0, 0.091, 0.01):
        pv = (1 - a - b) * ZAv + a * Kv + b * Vv
        rv = eval_choice(pv.argmax(1), va)
        tl = test_like_metric(rv)
        if tl > best[0]: best = (tl, (round(float(a), 2), round(float(b), 2)))
print('best val', best)
for a, b in [best[1], (0.18, 0.0), (0.15, 0.03), (0.12, 0.05), (0.10, 0.08)]:
    pv = (1 - a - b) * ZAv + a * Kv + b * Vv
    pt = (1 - a - b) * ZAt + a * Kt + b * Vt
    rv = eval_choice(pv.argmax(1), va); rt = eval_choice(pt.argmax(1), te)
    print(f'a={a:.2f} b={b:.2f} val={rv["acc"]:.2f} tl={test_like_metric(rv):.2f} || '
          f'test={rt["acc"]:.2f} #1={rt["acc1"]:.1f} #2={rt.get("acc2",0):.1f} #3={rt.get("acc3",0):.1f}')
