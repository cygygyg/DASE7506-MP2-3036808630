"""Final blend search: base rankers + kNN verifier. Saves scores for the final pipeline."""
import sys, io, pickle
from fractions import Fraction
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, eval_choice, fmt, test_like_metric
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
    clf = LogisticRegression(C=C, max_iter=4000, solver='lbfgs', random_state=0).fit(sc.transform(X), y)
    return sc, clf


def pred(sc, clf, X, d, own=False):
    sc2 = StandardScaler().fit(X) if own else sc
    return clf.decision_function(sc2.transform(X)).reshape(-1, 4)


# experts
scN, clN = fit(rtr, 0.3)
scA, clA = fit(Dtr, 0.1)
scM, clM = fit(np.hstack([rtr, Atr]), 0.1)
Sva = {'N': pred(scN, clN, rva, va), 'A': pred(scA, clA, Dva, va),
       'Aown': pred(scA, clA, Dva, va, own=True), 'M': pred(scM, clM, np.hstack([rva, Ava]), va)}
Ste = {'N': pred(scN, clN, rte, te), 'A': pred(scA, clA, Dte, te),
       'Aown': pred(scA, clA, Dte, te, own=True), 'M': pred(scM, clM, np.hstack([rte, Ate]), te)}
print('base val/test:')
for k in Sva:
    rv = eval_choice(mask(Sva[k], va).argmax(1), va); rt = eval_choice(mask(Ste[k], te).argmax(1), te)
    print(fmt(rv, k).split(' |')[0] + f' tl={test_like_metric(rv):5.2f} || test={rt["acc"]:.2f} #1={rt["acc1"]:.1f} #2={rt.get("acc2",0):.1f} #3={rt.get("acc3",0):.1f}')

Kva = z(np.load('cache/knn_va_40_1.0.npy'), va)  # tfidf k40 (old)
# embedding-question kNN must be recomputed; use exp_knn2 saved? not saved. recompute quickly here
from sklearn.preprocessing import normalize
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


def knn(Q, Qtr, R, d, k=40):
    S = Q @ Qtr.T; S = S.toarray() if hasattr(S, 'toarray') else np.asarray(S)
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
        p = np.average([x for x, _ in pos], 0, weights=[w for _, w in pos])
        n = np.average([x for x, _ in neg], 0, weights=[w for _, w in neg])
        p /= np.linalg.norm(p) + 1e-9; n /= np.linalg.norm(n) + 1e-9
        out[i] = R[i * 4:(i + 1) * 4] @ (p - n)
    return out


Kva = z(knn(Qva, Qtr, Rva, va, 40), va); Kte = z(knn(Qte, Qtr, Rte, te, 40), te)

best = (0, None)
for name in ('N', 'A', 'Aown', 'M'):
    for a in np.arange(0, 0.31, 0.01):
        pv = (1 - a) * z(Sva[name], va) + a * Kva
        rv = eval_choice(pv.argmax(1), va)
        if test_like_metric(rv) > best[0]:
            best = (test_like_metric(rv), (name, round(float(a), 2)))
print('best on val tl:', best)
for name, a in [('A', 0.1), ('Aown', 0.1), ('M', 0.1), ('N', 0.1), best[1]]:
    pv = (1 - a) * z(Sva[name], va) + a * Kva
    pt = (1 - a) * z(Ste[name], te) + a * Kte
    rv = eval_choice(pv.argmax(1), va); rt = eval_choice(pt.argmax(1), te)
    print(f'{name} a={a:.2f} val={rv["acc"]:.2f} tl={test_like_metric(rv):.2f} || '
          f'test={rt["acc"]:.2f} #1={rt["acc1"]:.1f} #2={rt.get("acc2",0):.1f} #3={rt.get("acc3",0):.1f}')
