"""Exact 5180-weight group-level joint ranker.

Five 1024-d candidate embedding views + 60 group/question features:
  view 1: centred residual r
  view 2: L2-normalised r
  view 3: r * normalised question vector (question-evidence interaction)
  view 4: r * answer log-magnitude
  view 5: r * log group size
Group score: max over members + v . h_g (60 features).
Loss: listwise softmax over answer groups + lambda/2 ||theta-theta_old||^2.
"""
import sys, io
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric
from mp2features import build

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
ftr = build('train', tr); fva = build('validation', va); fte = build('test', te)


def is_int(a):
    return a is not None and Fraction(a).denominator == 1


def candidate_views(d):
    emb = d['emb'].astype(np.float64)
    n = len(emb)
    qbar = emb.mean(1)
    r = emb - qbar[:, None, :]
    rnorm = np.linalg.norm(r, axis=2, keepdims=True) + 1e-9
    qhat = qbar / (np.linalg.norm(qbar, axis=1, keepdims=True) + 1e-9)
    views = np.zeros((n, 4, 5, 1024), dtype=np.float64)
    views[:, :, 0] = r
    views[:, :, 1] = r / rnorm
    views[:, :, 2] = r * qhat[:, None, :]
    # answer log magnitude and group size scalars
    for i in range(n):
        vals = []
        for c in range(4):
            a = d['answers'][i, c]
            vals.append(a)
        gn = len(set(a for a in vals if is_int(a)))
        # group sizes from features
        sizes = None
        for c in range(4):
            a = d['answers'][i, c]
            if not is_int(a):
                continue
            v = float(Fraction(a))
            gs = sum(1 for x in vals if x == a)
            views[i, c, 3] = r[i, c] * (np.sign(v) * np.log1p(abs(v)) / 5.0)
            views[i, c, 4] = r[i, c] * (np.log1p(gs) / 2.0)
    return views


def group_data(d, feats, views, labels):
    n = len(d['bank'])
    out = []
    for i in range(n):
        rec = feats['records'][i]
        gids = rec['gids']; valid = rec['valid']
        G = len(rec['valid'])
        h = np.concatenate([feats['group'][i], np.tile(feats['q'][i][:9], (G, 1))], axis=1)
        y = np.array([int(any(labels[i, gids == g])) for g in range(rec['G'])])
        out.append(dict(i=i, views=views[i], gids=gids, valid=valid, h=h, y=y,
                        groups=rec['groups'] if 'groups' in rec else None,
                        keys=rec.get('keys')))
    return out


# attach group member indices
def attach_groups(d, feats, qs):
    for q in qs:
        i = q['i']
        groups = {}
        for c, a in enumerate(d['answers'][i]):
            if not is_int(a):
                continue
            groups.setdefault(a, []).append(c)
        q['groups'] = groups
        q['keys'] = list(groups.keys())
    return qs


Vtr = candidate_views(tr); Vva = candidate_views(va); Vte = candidate_views(te)
qtr = attach_groups(tr, ftr, group_data(tr, ftr, Vtr, tr['labels']))
qva = attach_groups(va, fva, group_data(va, fva, Vva, va['labels']))
qte = attach_groups(te, fte, group_data(te, fte, Vte, te['labels']))

# standardise each view and group features on train
vm = np.zeros((5, 1024)); vs = np.zeros((5, 1024))
for v in range(5):
    x = Vtr[:, :, v, :].reshape(-1, 1024)
    vm[v] = x.mean(0); vs[v] = x.std(0) + 1e-6
H = np.concatenate([q['h'] for q in qtr], 0)
hm = H.mean(0); hs = H.std(0) + 1e-6
for qs in (qtr, qva, qte):
    for q in qs:
        q['zv'] = (q['views'] - vm[None, :, :]) / vs[None, :, :]
        q['zh'] = (q['h'] - hm) / hs

train_qs = [q for q in qtr if q['valid'].sum() >= 2]
print('train questions >=2 groups:', len(train_qs))

# old model on view 1

X1 = np.stack([q['zv'][:, 0, :] for q in train_qs]).reshape(-1, 1024)
y1 = np.stack([tr['labels'][q['i']] for q in train_qs]).reshape(-1)
old = LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs', random_state=0).fit(X1, y1)
w0 = old.coef_[0].astype(np.float64)


def objective(theta, qs, lam, nh=60, nv=5):
    W = theta[:nv * 1024].reshape(nv, 1024)
    b = theta[nv * 1024]
    v = theta[nv * 1024 + 1:]
    loss = 0.5 * lam * np.sum((W[0] - w0) ** 2) + 0.5 * lam * np.sum(W[1:] ** 2) + 0.5 * 1e-3 * np.sum(v ** 2)
    gW = np.zeros_like(W); gW[0] = lam * (W[0] - w0); gW[1:] = lam * W[1:]
    gb = 0.0; gv = 1e-3 * v
    for q in qs:
        s = np.einsum('cvf,vf->c', q['zv'], W) + b
        g = q['zh'] @ v
        valid = q['valid']; y = q['y']; gids = q['gids']
        G = len(valid)
        S = np.full(G, -1e9); am = np.zeros(G, dtype=int)
        for gi in range(G):
            if not valid[gi]:
                continue
            mem = np.where(gids == gi)[0]
            j = mem[np.argmax(s[mem])]
            am[gi] = j
            S[gi] = s[j] + g[gi]
        m = S.max(); e = np.exp(S - m); Zs = e.sum()
        pos = np.where((y == 1) & valid)[0]
        if len(pos) == 0:
            continue
        Zp = e[pos].sum()
        loss += (m + np.log(Zs)) - (m + np.log(Zp))
        dS = e / Zs; dS[pos] -= e[pos] / Zp
        for gi in range(G):
            if dS[gi] == 0 or not valid[gi]:
                continue
            gW += dS[gi] * q['zv'][am[gi]]
            gb += dS[gi]
            gv += dS[gi] * q['zh'][gi]
    return loss / len(qs), np.concatenate([gW.ravel(), [gb], gv]) / len(qs)


def fit(lam, maxiter=120):
    theta0 = np.concatenate([np.zeros(5 * 1024), [0.0], np.zeros(60)])
    theta0[:1024] = w0
    res = minimize(objective, theta0, args=(train_qs, lam), method='L-BFGS-B', jac=True,
                   options=dict(maxiter=maxiter, maxfun=maxiter * 2, ftol=1e-9))
    return res.x, res


def predict(qs, theta):
    W = theta[:5 * 1024].reshape(5, 1024); b = theta[5 * 1024]; v = theta[5 * 1024 + 1:]
    ch = np.zeros(len(qs), dtype=int)
    for k, q in enumerate(qs):
        s = np.einsum('cvf,vf->c', q['zv'], W) + b
        g = q['zh'] @ v
        G = len(q['valid']); S = np.full(G, -1e9)
        for gi in range(G):
            if not q['valid'][gi]:
                continue
            mem = np.where(q['gids'] == gi)[0]
            S[gi] = s[mem].max() + g[gi]
        gi = int(np.argmax(S))
        ch[k] = int(np.where(q['gids'] == gi)[0][0])
    return ch


for lam in (1.0, 3.0):
    theta, res = fit(lam)
    print(f'lambda={lam} nit={res.nit} success={res.success}')
    for name, qs, d in (('val', qva, va), ('test', qte, te)):
        ch = predict(qs, theta)
        r = eval_choice(ch, d)
        print(fmt(r, f'{name} joint5180') + (f' tl={test_like_metric(r):.2f}' if name == 'val' else ''))

import pickle
with open('cache/joint5180.pkl', 'wb') as fh:
    pickle.dump(dict(vm=vm, vs=vs, hm=hm, hs=hs, w0=w0), fh)
