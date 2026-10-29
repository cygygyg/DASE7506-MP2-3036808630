"""Fusion: centred embedding + magnitude/arith/text features (no support features)."""
import sys, io, re
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
NUM = re.compile(r'\d+(?:\.\d+)?')


def is_int(a): return a is not None and Fraction(a).denominator == 1
def mask_rows(p, d):
    p = p.reshape(-1, 4).copy()
    for i in range(len(p)):
        for c in range(4):
            if not is_int(d['answers'][i, c]): p[i, c] = -1e9
    return p


def qstats(q):
    v = [float(x) for x in NUM.findall(q)]
    if not v: return 1.0, 1.0, 1.0, 1
    return min(v), max(v), sum(v), len(v)


def mag_features(d):
    rows = []
    for i in range(len(d['bank'])):
        mn, mx, sm, qn = qstats(d['bank'][i]['question'])
        mx = max(1e-9, mx); mn = max(1e-9, mn); sm = max(1e-9, sm)
        for c in range(4):
            a = d['answers'][i, c]
            if not is_int(a):
                rows.append([0.0] * 8); continue
            v = float(Fraction(a))
            rows.append([v / mx, v / mn, v / sm, (v - mx) / (mx + 1e-9),
                         float(0 <= v <= mx * max(1, qn)), float(v == mx), float(v == sm),
                         np.log1p(abs(v)) - np.log1p(mx)])
    return np.array(rows, np.float32)


def arith_mat(d):
    rows = []
    for i in range(len(d['bank'])):
        for c in range(4):
            a = d['answers'][i, c]
            fv = int(Fraction(a)) if is_int(a) else None
            rows.append(list(arith2_features(d['bank'][i]['candidates'][c]['text'], fv).values()))
    return np.array(rows, np.float32)


Mtr, Mva, Mte = mag_features(tr), mag_features(va), mag_features(te)
Atr, Ava, Ate = arith_mat(tr), arith_mat(va), arith_mat(te)
Rtr = (tr['emb'] - tr['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
Rva = (va['emb'] - va['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
Rte = (te['emb'] - te['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
Str = ftr['cand'].reshape(-1, ftr['cand'].shape[-1]); Sva = fva['cand'].reshape(-1, fva['cand'].shape[-1]); Ste = fte['cand'].reshape(-1, fte['cand'].shape[-1])
y = tr['labels'].reshape(-1)


def run(name, blocks_tr, blocks_va, blocks_te, C):
    A = np.hstack(blocks_tr); B = np.hstack(blocks_va); D = np.hstack(blocks_te)
    sc = StandardScaler().fit(A)
    clf = LogisticRegression(C=C, max_iter=4000, solver='lbfgs', random_state=0).fit(sc.transform(A), y)
    rv = eval_choice(mask_rows(clf.predict_proba(sc.transform(B))[:, 1], va).argmax(1), va)
    rt = eval_choice(mask_rows(clf.predict_proba(sc.transform(D))[:, 1], te).argmax(1), te)
    print(fmt(rv, f'{name} C={C}') + f' tl={test_like_metric(rv):5.2f} || ' + fmt(rt, 'TEST'), flush=True)


for C in (0.05, 0.1, 0.3, 1.0):
    run('emb', [Rtr], [Rva], [Rte], C)
for C in (0.05, 0.1, 0.3, 1.0):
    run('emb+mag', [Rtr, Mtr], [Rva, Mva], [Rte, Mte], C)
for C in (0.05, 0.1, 0.3, 1.0):
    run('emb+mag+arith', [Rtr, Mtr, Atr], [Rva, Mva, Ava], [Rte, Mte, Ate], C)
for C in (0.05, 0.1, 0.3, 1.0):
    run('emb+mag+arith+text', [Rtr, Mtr, Atr, Str], [Rva, Mva, Ava, Sva], [Rte, Mte, Ate, Ste], C)
