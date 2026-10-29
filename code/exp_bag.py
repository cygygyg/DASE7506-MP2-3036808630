"""Bagged answer-debiased ranker + kNN: robustness check (select on validation)."""
import sys, io
from fractions import Fraction
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from common import load_inputs
from mp2lib import load_split, eval_choice, test_like_metric
import submission as sub

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
btr, atr = load_inputs('train'); bva, ava = load_inputs('validation'); bte, ate = load_inputs('test')
ctr, ftr, vtr, _, _ = sub._candidate_matrix(btr, atr['embeddings'])
cva, fva, vva, _, _ = sub._candidate_matrix(bva, ava['embeddings'])
cte, fte, vte, _, _ = sub._candidate_matrix(bte, ate['embeddings'])


def is_int(a): return a is not None and Fraction(a).denominator == 1
def maskz(p, d):
    p = p.reshape(-1, 4).copy()
    for i in range(len(p)):
        valid = np.array([is_int(d['answers'][i, c]) for c in range(4)])
        if not valid.any(): p[i] = 0; continue
        v = p[i][valid]; m, s = v.mean(), v.std() + 1e-9
        p[i] = (p[i] - m) / s; p[i][~valid] = -1e9
    return p


y = tr['labels'].reshape(-1)
# residualisation matrix from train
z1 = np.concatenate([np.ones((len(ctr) * 4, 1)), ftr.reshape(-1, 5)], axis=1)
M = z1.T @ z1 + np.eye(6); M[0, 0] -= 1
B = np.linalg.solve(M, z1.T @ ctr.reshape(-1, 1024))
Dtr = ctr.reshape(-1, 1024) - z1 @ B
Dva = cva.reshape(-1, 1024) - np.concatenate([np.ones((len(cva) * 4, 1)), fva.reshape(-1, 5)], 1) @ B
Dte = cte.reshape(-1, 1024) - np.concatenate([np.ones((len(cte) * 4, 1)), fte.reshape(-1, 5)], 1) @ B
sc = StandardScaler().fit(Dtr)
Ztr, Zva, Zte = sc.transform(Dtr), sc.transform(Dva), sc.transform(Dte)
nq = len(btr)


def fit_coef(idx):
    m = (np.asarray(idx)[:, None] * 4 + np.arange(4)).ravel()
    clf = LogisticRegression(C=0.1, max_iter=3000, solver='lbfgs', random_state=0).fit(Ztr[m], y[m])
    return clf.coef_[0], clf.intercept_[0]


w_full, b_full = fit_coef(np.arange(nq))
w_list, b_list = [], []
rng = np.random.default_rng(0)
for k in range(5):
    idx = rng.choice(nq, size=int(0.8 * nq), replace=False)
    w, b = fit_coef(idx)
    w_list.append(w); b_list.append(b)
w_bag = np.mean(w_list, 0); b_bag = np.mean(b_list)

Ava = (Zva @ w_full + b_full).reshape(-1, 4)
Ate = (Zte @ w_full + b_full).reshape(-1, 4)
Bva = np.mean([Zva @ w + b for w, b in zip(w_list, b_list)], 0).reshape(-1, 4)
Bte = np.mean([Zte @ w + b for w, b in zip(w_list, b_list)], 0).reshape(-1, 4)
Cva = 0.5 * maskz(Ava, va) + 0.5 * maskz(Bva, va)
Cte = 0.5 * maskz(Ate, te) + 0.5 * maskz(Bte, te)

# kNN
ck = dict(np.load('runs/final_bundle/model.npz'))
train = sub._load_train_bank(sub._find_train_data())
Kva = maskz(sub._knn_scores(bva, cva, np.asarray(ava['embeddings']).mean(1), vva, train, ck['qmean'], k=80), va)
Kte = maskz(sub._knn_scores(bte, cte, np.asarray(ate['embeddings']).mean(1), vte, train, ck['qmean'], k=80), te)

for name, Av, At in (('full', maskz(Ava, va), maskz(Ate, te)),
                     ('bag', maskz(Bva, va), maskz(Bte, te)),
                     ('avg', Cva, Cte)):
    best = (0, 0)
    for a in np.arange(0.0, 0.31, 0.01):
        rv = eval_choice(((1 - a) * Av + a * Kva).argmax(1), va)
        tl = test_like_metric(rv)
        if tl > best[0]: best = (tl, a)
    tl, a = best
    rv = eval_choice(((1 - a) * Av + a * Kva).argmax(1), va)
    rt = eval_choice(((1 - a) * At + a * Kte).argmax(1), te)
    print(f'{name:5s} best_a={a:.2f} val={rv["acc"]:.2f} tl={tl:.2f} || test={rt["acc"]:.2f} '
          f'#1={rt["acc1"]:.1f} #2={rt.get("acc2",0):.1f} #3={rt.get("acc3",0):.1f}')
