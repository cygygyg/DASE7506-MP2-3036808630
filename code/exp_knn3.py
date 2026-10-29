"""kNN variant check: raw vs centred vs answer-debiased candidate residuals."""
import sys, io, pickle
from fractions import Fraction
import numpy as np
from sklearn.preprocessing import normalize
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, eval_choice, test_like_metric

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
S = pickle.load(open('cache/final_scores.pkl', 'rb'))
ZAv, ZAt = S['ZAv'], S['ZAt']

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

qtr = tr['emb'].mean(1); qva = va['emb'].mean(1); qte = te['emb'].mean(1)
g = qtr.mean(0)
Qtr = normalize(qtr - g); Qva = normalize(qva - g); Qte = normalize(qte - g)
ytr = tr['labels']

variants = {}
variants['raw'] = (
    normalize(tr['emb'].reshape(-1, 1024)),
    normalize(va['emb'].reshape(-1, 1024)),
    normalize(te['emb'].reshape(-1, 1024)),
)
for nm, d in (('center', tr), ('debiased', None)):
    pass
Rtr = (tr['emb'] - tr['emb'].mean(1, keepdims=True)).reshape(-1, 1024)
Rva = (va['emb'] - va['emb'].mean(1, keepdims=True)).reshape(-1, 1024)
Rte = (te['emb'] - te['emb'].mean(1, keepdims=True)).reshape(-1, 1024)
variants['center'] = (normalize(Rtr), normalize(Rva), normalize(Rte))
# answer-debiased (recompute B quickly using cached? skip: approximate with center)

def knn(Qtr_, Q, R, d, Rtr_, k=80, kernel=True):
    Sm = np.asarray(Q @ Qtr_.T)
    out = np.zeros((len(d['bank']), 4))
    for i in range(len(d['bank'])):
        idx = np.argsort(-Sm[i])[:k]
        w = np.maximum(Sm[i, idx], 0); w = w / (w.sum() + 1e-12)
        pos, neg = [], []
        for j, wj in zip(idx, w):
            for c in range(4):
                if not is_int(tr['answers'][j, c]): continue
                (pos if ytr[j, c] else neg).append((Rtr_[j * 4 + c], wj))
        if not pos or not neg: continue
        P = np.stack([x for x, _ in pos], 0); Wp = np.array([w for _, w in pos])
        N = np.stack([x for x, _ in neg], 0); Wn = np.array([w for _, w in neg])
        sp = R[i * 4:(i + 1) * 4] @ P.T; sn = R[i * 4:(i + 1) * 4] @ N.T
        out[i] = (sp * Wp).sum(1) / Wp.sum() - (sn * Wn).sum(1) / Wn.sum()
    return out

for name, (Rtr_, Rva_, Rte_) in variants.items():
    Kv = z(knn(Qtr, Qva, Rva_, va, Rtr_), va); Kt = z(knn(Qtr, Qte, Rte_, te, Rtr_), te)
    best = (0, 0)
    for a in np.arange(0.01, 0.31, 0.01):
        rv = eval_choice(((1 - a) * ZAv + a * Kv).argmax(1), va)
        if test_like_metric(rv) > best[0]: best = (test_like_metric(rv), a)
    tl, a = best
    rv = eval_choice(((1 - a) * ZAv + a * Kv).argmax(1), va)
    rt = eval_choice(((1 - a) * ZAt + a * Kt).argmax(1), te)
    print(f'{name:8s} a={a:.2f} val={rv["acc"]:.2f} tl={tl:.2f} || test={rt["acc"]:.2f} '
          f'#1={rt["acc1"]:.1f} #2={rt.get("acc2",0):.1f} #3={rt.get("acc3",0):.1f}')
