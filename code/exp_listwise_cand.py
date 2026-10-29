"""Candidate-level listwise / pairwise ranking with L2 toward the old logistic model."""
import sys, io
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, eval_choice, fmt, test_like_metric

tr = load_split('train'); va = load_split('validation'); te = load_split('test')


def is_int(a):
    from fractions import Fraction
    return a is not None and Fraction(a).denominator == 1


def build(d):
    N = len(d['bank'])
    r = d['emb'] - d['emb'].mean(axis=1, keepdims=True)
    r = r.reshape(N * 4, -1).astype(np.float64)
    valid = np.array([[is_int(d['answers'][i, c]) for c in range(4)] for i in range(N)])
    return r, valid, d['labels']


Rtr, Vtr, Ltr = build(tr); Rva, Vva, Lva = build(va); Rte, Vte, Lte = build(te)
sc = StandardScaler().fit(Rtr)
Ztr, Zva, Zte = sc.transform(Rtr), sc.transform(Rva), sc.transform(Rte)
y = Ltr.reshape(-1)
w0 = LogisticRegression(C=1.0, max_iter=4000, solver='lbfgs', random_state=0).fit(Ztr, y).coef_[0]
b0 = LogisticRegression(C=1.0, max_iter=4000, solver='lbfgs', random_state=0).fit(Ztr, y).intercept_[0]


def obj(theta, Z, V, L, lam, kind):
    w = theta[:-1]; b = theta[-1]
    N = Z.shape[0] // 4
    s = (Z @ w + b).reshape(N, 4)
    loss = 0.5 * lam * np.sum((w - w0) ** 2)
    gw = lam * (w - w0); gb = 0.0
    for i in range(N):
        v = V[i]; yv = L[i]
        pos = np.where((yv == 1) & v)[0]
        neg = np.where((yv == 0) & v)[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        if kind == 'list':
            idx = np.where(v)[0]
            sv = s[i, idx]; mm = sv.max(); ee = np.exp(sv - mm)
            posmask = (yv[idx] == 1)
            Zp = ee[posmask].sum()
            loss += (mm + np.log(ee.sum())) - (mm + np.log(Zp))
            d = ee / ee.sum(); d[posmask] -= ee[posmask] / Zp
            gw += (Z[i * 4 + idx].T @ d)
            gb += d.sum()
        else:
            # pairwise logistic
            ps = s[i, pos]; ns = s[i, neg]
            diff = ps[:, None] - ns[None, :]
            loss += np.log1p(np.exp(-diff)).sum() / (len(pos) * len(neg))
            p = 1 / (1 + np.exp(diff))
            coef = -(1 - p) / (len(pos) * len(neg))
            gps = coef.sum(axis=1)
            gns = -coef.sum(axis=0)
            gw += gps @ Z[i * 4 + pos] + gns @ Z[i * 4 + neg]
            gb += gps.sum() + gns.sum()
    return loss / N, np.concatenate([gw / N, [gb / N]])


def fit(lam, kind, maxiter=300):
    theta0 = np.concatenate([w0, [b0]])
    res = minimize(obj, theta0, args=(Ztr, Vtr, Ltr, lam, kind), method='L-BFGS-B', jac=True,
                   options=dict(maxiter=maxiter, maxfun=maxiter * 2, ftol=1e-9))
    return res.x, res


def pred(theta, Z, V, d):
    w = theta[:-1]; b = theta[-1]
    s = (Z @ w + b).reshape(-1, 4)
    for i in range(len(s)):
        for c in range(4):
            if not V[i, c]: s[i, c] = -1e9
    return eval_choice(s.argmax(1), d)


for kind in ('list', 'pair'):
    for lam in (0.1, 0.3, 1.0, 3.0, 10.0):
        theta, res = fit(lam, kind, 200)
        rv = pred(theta, Zva, Vva, va); rt = pred(theta, Zte, Vte, te)
        print(f'{kind} lam={lam:5.2f} nit={res.nit:3d} ' + fmt(rv, 'val') + f' tl={test_like_metric(rv):5.2f} || ' + fmt(rt, 'TEST'), flush=True)
