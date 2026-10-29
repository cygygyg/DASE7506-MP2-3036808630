"""Per-bank standardisation and #1-specialised experts."""
import sys, io
from fractions import Fraction
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric
from mp2features import build

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
ftr = build('train', tr); fva = build('validation', va); fte = build('test', te)
cn = ftr['cand_names']; iS = {k: i for i, k in enumerate(cn)}


def is_int(a): return a is not None and Fraction(a).denominator == 1
def mask_rows(p, d):
    p = p.reshape(-1, 4).copy()
    for i in range(len(p)):
        for c in range(4):
            if not is_int(d['answers'][i, c]): p[i, c] = -1e9
    return p


def base(d, f):
    N = len(d['bank'])
    r = d['emb'] - d['emb'].mean(axis=1, keepdims=True)
    r = r.reshape(N * 4, -1).astype(np.float64)
    S = f['cand'].reshape(N * 4, -1).astype(np.float64)
    G = np.zeros((N * 4, 6))
    for i, rec in enumerate(f['records']):
        gids = rec['gids']; sizes = rec['sizes'].astype(float)
        for c in range(4):
            g = gids[c]
            G[i * 4 + c] = [sizes[g], float(sizes[g] == sizes.max()), float(sizes[g] == 1),
                            float(np.sum(sizes > sizes[g])), float(len(sizes)), sizes.max()]
    Zans = S[:, [iS['v_log'], iS['v_rank'], iS['v_rel_max'], iS['v_digits'], iS['v_int']]]
    Zsup = np.hstack([G, S[:, [iS['v_rank'], iS['v_rel_max'], iS['v_int']]]])
    return r, Zans, Zsup


rtr, Atr, Str = base(tr, ftr); rva, Ava, Sva = base(va, fva); rte, Ate, Ste = base(te, fte)


def resid(r, Z):
    Z1 = np.hstack([np.ones((len(Z), 1)), Z])
    M = Z1.T @ Z1 + np.eye(Z1.shape[1]); M[0, 0] -= 1
    B = np.linalg.solve(M, Z1.T @ r)
    return r - Z1 @ B, B


def apply_resid(r, Z, B):
    return r - np.hstack([np.ones((len(Z), 1)), Z]) @ B


nc_tr = ncorrect(tr); ytr = tr['labels'].reshape(-1)


def run(name, Xtr, Xva, Xte, mask_tr=None, C=0.3, own_scaler=False):
    m = np.repeat(np.ones(len(nc_tr), bool) if mask_tr is None else mask_tr, 4)
    sc = StandardScaler().fit(Xtr[m])
    clf = LogisticRegression(C=C, max_iter=4000, solver='lbfgs', random_state=0).fit(sc.transform(Xtr[m]), ytr[m])
    out = []
    for d, X in ((va, Xva), (te, Xte)):
        if own_scaler:
            sc2 = StandardScaler().fit(X)
            Z = sc2.transform(X)
        else:
            Z = sc.transform(X)
        p = mask_rows(clf.predict_proba(Z)[:, 1], d)
        out.append(eval_choice(p.argmax(1), d))
    print(fmt(out[0], name) + f' tl={test_like_metric(out[0]):5.2f} || ' + fmt(out[1], 'TEST'), flush=True)


print('--- per-bank scaler ---')
run('center train-scaler', rtr, rva, rte, C=0.3, own_scaler=False)
run('center own-scaler', rtr, rva, rte, C=0.3, own_scaler=True)
Dtr, Ba = resid(rtr, Atr)
run('answer train-scaler', Dtr, apply_resid(rva, Ava, Ba), apply_resid(rte, Ate, Ba), C=0.1, own_scaler=False)
run('answer own-scaler', Dtr, apply_resid(rva, Ava, Ba), apply_resid(rte, Ate, Ba), C=0.1, own_scaler=True)

print('--- #1-specialised ---')
run('plain #1', rtr, rva, rte, nc_tr == 1, C=0.1, own_scaler=False)
run('plain #1+#2', rtr, rva, rte, nc_tr <= 2, C=0.1, own_scaler=False)
run('answer #1', Dtr, apply_resid(rva, Ava, Ba), apply_resid(rte, Ate, Ba), nc_tr == 1, C=0.1, own_scaler=False)
