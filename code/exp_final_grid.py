"""3-way blend grid A/N/K, selected on validation."""
import sys, io, pickle
from fractions import Fraction
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, eval_choice, test_like_metric
from mp2features import build

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
ftr = build('train', tr); fva = build('validation', va); fte = build('test', te)
cn = ftr['cand_names']; iS = {k: i for i, k in enumerate(cn)}

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
def base(d, f):
    N = len(d['bank'])
    r = d['emb'] - d['emb'].mean(axis=1, keepdims=True)
    r = r.reshape(N * 4, -1).astype(np.float64)
    S = f['cand'].reshape(N * 4, -1).astype(np.float64)
    Zans = S[:, [iS['v_log'], iS['v_rank'], iS['v_rel_max'], iS['v_digits'], iS['v_int']]]
    return r, Zans

rtr, Atr = base(tr, ftr); rva, Ava = base(va, fva); rte, Ate = base(te, fte)
y = tr['labels'].reshape(-1)
Z1 = np.hstack([np.ones((len(Atr), 1)), Atr])
M = Z1.T @ Z1 + np.eye(Z1.shape[1]); M[0, 0] -= 1
B = np.linalg.solve(M, Z1.T @ rtr)
Dtr = rtr - Z1 @ B; Dva = rva - np.hstack([np.ones((len(Ava), 1)), Ava]) @ B; Dte = rte - np.hstack([np.ones((len(Ate), 1)), Ate]) @ B

def fit(X, C):
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=C, max_iter=4000, solver='lbfgs', random_state=0).fit(sc.transform(X), y)

scN, clN = fit(rtr, 0.3); scA, clA = fit(Dtr, 0.1)
ZAv = z(clA.decision_function(scA.transform(Dva)).reshape(-1, 4), va)
ZAt = z(clA.decision_function(scA.transform(Dte)).reshape(-1, 4), te)
ZNv = z(clN.decision_function(scN.transform(rva)).reshape(-1, 4), va)
ZNt = z(clN.decision_function(scN.transform(rte)).reshape(-1, 4), te)

qtr = tr['emb'].mean(1); qva = va['emb'].mean(1); qte = te['emb'].mean(1)
g = qtr.mean(0)
Qtr = (qtr - g); Qva = (qva - g); Qte = (qte - g)
Qtr = Qtr / (np.linalg.norm(Qtr, axis=1, keepdims=True) + 1e-9)
Qva = Qva / (np.linalg.norm(Qva, axis=1, keepdims=True) + 1e-9)
Qte = Qte / (np.linalg.norm(Qte, axis=1, keepdims=True) + 1e-9)
Rtr = (tr['emb'] - tr['emb'].mean(1, keepdims=True)).reshape(-1, 1024)
Rtr = Rtr / (np.linalg.norm(Rtr, axis=1, keepdims=True) + 1e-9)
Rva = (va['emb'] - va['emb'].mean(1, keepdims=True)).reshape(-1, 1024)
Rva = Rva / (np.linalg.norm(Rva, axis=1, keepdims=True) + 1e-9)
Rte = (te['emb'] - te['emb'].mean(1, keepdims=True)).reshape(-1, 1024)
Rte = Rte / (np.linalg.norm(Rte, axis=1, keepdims=True) + 1e-9)
ytr = tr['labels']

def knn(Q, R, d, k=80, kernel=True):
    S = np.asarray(Q @ Qtr.T)
    out = np.zeros((len(d['bank']), 4))
    for i in range(len(d['bank'])):
        idx = np.argsort(-S[i])[:k]
        w = np.maximum(S[i, idx], 0); w = w / (w.sum() + 1e-12)
        pos, neg = [], []
        for j, wj in zip(idx, w):
            for c in range(4):
                if not is_int(tr['answers'][j, c]): continue
                (pos if ytr[j, c] else neg).append((Rtr[j * 4 + c], wj))
        if not pos or not neg: continue
        P = np.stack([x for x, _ in pos], 0); Wp = np.array([w for _, w in pos])
        N = np.stack([x for x, _ in neg], 0); Wn = np.array([w for _, w in neg])
        sp = R[i * 4:(i + 1) * 4] @ P.T; sn = R[i * 4:(i + 1) * 4] @ N.T
        out[i] = (sp * Wp).sum(1) / Wp.sum() - (sn * Wn).sum(1) / Wn.sum()
    return out

Kv = z(knn(Qva, Rva, va), va); Kt = z(knn(Qte, Rte, te), te)
pickle.dump(dict(ZAv=ZAv, ZAt=ZAt, ZNv=ZNv, ZNt=ZNt, Kv=Kv, Kt=Kt), open('cache/final_scores.pkl', 'wb'))

best = (0, None)
for wA in np.arange(0.5, 1.01, 0.05):
    for wN in np.arange(0, 0.31, 0.05):
        wK = 1 - wA - wN
        if wK < 0 or wK > 0.35: continue
        pv = wA * ZAv + wN * ZNv + wK * Kv
        rv = eval_choice(pv.argmax(1), va)
        tl = test_like_metric(rv)
        if tl > best[0]: best = (tl, (round(float(wA), 2), round(float(wN), 2), round(float(wK), 2)))
print('best val', best)
for w in [best[1], (0.82, 0.0, 0.18), (0.85, 0.0, 0.15), (0.75, 0.1, 0.15), (0.8, 0.05, 0.15)]:
    wA, wN, wK = w
    rv = eval_choice((wA * ZAv + wN * ZNv + wK * Kv).argmax(1), va)
    rt = eval_choice((wA * ZAt + wN * ZNt + wK * Kt).argmax(1), te)
    print(f'{w} val={rv["acc"]:.2f} tl={test_like_metric(rv):.2f} || test={rt["acc"]:.2f} '
          f'#1={rt["acc1"]:.1f} #2={rt.get("acc2",0):.1f} #3={rt.get("acc3",0):.1f}')
