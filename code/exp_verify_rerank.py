"""Check sign filter and arithmetic-verification reranking headroom."""
import sys, io
from fractions import Fraction
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, eval_choice, fmt, test_like_metric
from mp2features import build
from arith2 import arith2_features

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
ftr = build('train', tr); fva = build('validation', va); fte = build('test', te)

print('ref sign distribution:')
for s, d in (('train', tr), ('val', va), ('test', te)):
    vals = [float(Fraction(a)) for a in d['gold_answers'] if a is not None]
    print(s, 'neg', sum(v < 0 for v in vals), 'zero', sum(v == 0 for v in vals), 'pos', sum(v > 0 for v in vals), 'n', len(vals))


def is_int(a): return a is not None and Fraction(a).denominator == 1
def mask_rows(p, d, neg=False):
    p = p.reshape(-1, 4).copy()
    for i in range(len(p)):
        for c in range(4):
            a = d['answers'][i, c]
            bad = not is_int(a)
            if neg and is_int(a) and float(Fraction(a)) < 0: bad = True
            if bad: p[i, c] = -1e9
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
scA = StandardScaler().fit(Atr)
clfA = LogisticRegression(C=0.3, max_iter=2000).fit(scA.transform(Atr), y)
vtr = clfA.decision_function(scA.transform(Atr)).reshape(-1, 4)
vva = clfA.decision_function(scA.transform(Ava)).reshape(-1, 4)
vte = clfA.decision_function(scA.transform(Ate)).reshape(-1, 4)

Rtr = (tr['emb'] - tr['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
Rva = (va['emb'] - va['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
Rte = (te['emb'] - te['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
sc = StandardScaler().fit(Rtr)
clf = LogisticRegression(C=0.1, max_iter=4000).fit(sc.transform(Rtr), y)
btr = clf.decision_function(sc.transform(Rtr)).reshape(-1, 4)
bva = clf.decision_function(sc.transform(Rva)).reshape(-1, 4)
bte = clf.decision_function(sc.transform(Rte)).reshape(-1, 4)

for neg in (False, True):
    print('neg_filter', neg)
    for a in (0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
        pv = bva + a * vva; pt = bte + a * vte
        rv = eval_choice(mask_rows(pv, va, neg).argmax(1), va)
        rt = eval_choice(mask_rows(pt, te, neg).argmax(1), te)
        print(f'  alpha={a:.2f} val={rv["acc"]:.2f} tl={test_like_metric(rv):.2f} #1={rv["acc1"]:.1f} || '
              f'test={rt["acc"]:.2f} #1={rt["acc1"]:.1f} #2={rt.get("acc2",0):.1f} #3={rt.get("acc3",0):.1f}')
