"""Embedding view sweep: centring, normalisation, question interaction, PCA."""
import sys, io
from fractions import Fraction
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, eval_choice, fmt, test_like_metric

tr = load_split('train'); va = load_split('validation'); te = load_split('test')


def is_int(a): return a is not None and Fraction(a).denominator == 1
def mask(d, p):
    p = p.reshape(-1, 4).copy()
    for i in range(len(p)):
        for c in range(4):
            if not is_int(d['answers'][i, c]): p[i, c] = -1e9
    return p


def views(d, qbar_mode='mean'):
    x = d['emb'].reshape(-1, 4, 1024).astype(np.float64)
    qbar = x.mean(1)                       # (N,1024)
    r = x - qbar[:, None, :]
    out = {}
    out['center'] = r
    out['center_l2'] = r / (np.linalg.norm(r, axis=2, keepdims=True) + 1e-9)
    xn = x / (np.linalg.norm(x, axis=2, keepdims=True) + 1e-9)
    out['norm_center'] = xn - xn.mean(1, keepdims=True)
    std = r.std(1, keepdims=True) + 1e-6
    out['group_std'] = r / std
    qn = qbar / (np.linalg.norm(qbar, axis=1, keepdims=True) + 1e-9)
    out['inter1'] = np.concatenate([r, r * qn[:, None, :]], axis=2)
    out['inter2'] = np.concatenate([r, r * qn[:, None, :], r * (qn ** 2)[:, None, :]], axis=2)
    out['raw_center'] = np.concatenate([x, r], axis=2)
    out['center_cos'] = np.concatenate([r, out['center_l2']], axis=2)
    return out


Vtr = views(tr); Vva = views(va); Vte = views(te)


def run(name, C=1.0, pca=None):
    A = Vtr[name]; B = Vva[name]; D = Vte[name]
    nf = A.shape[2]
    A = A.reshape(-1, nf); B = B.reshape(-1, nf); D = D.reshape(-1, nf)
    if pca:
        p = PCA(n_components=pca, random_state=0).fit(A)
        A = p.transform(A); B = p.transform(B); D = p.transform(D)
    sc = StandardScaler().fit(A)
    clf = LogisticRegression(C=C, max_iter=5000, solver='lbfgs', random_state=0).fit(sc.transform(A), tr['labels'].reshape(-1))
    rv = eval_choice(mask(va, clf.predict_proba(sc.transform(B))[:, 1]).argmax(1), va)
    rt = eval_choice(mask(te, clf.predict_proba(sc.transform(D))[:, 1]).argmax(1), te)
    print(fmt(rv, f'{name} C={C}' + (f' pca={pca}' if pca else '')) + f' tl={test_like_metric(rv):5.2f} || ' + fmt(rt, 'TEST'))


for name in ('center', 'center_l2', 'norm_center', 'group_std', 'inter1', 'inter2', 'raw_center', 'center_cos'):
    for C in (0.1, 0.3, 1.0):
        run(name, C)

print('--- PCA on center ---')
for pca in (64, 128, 256, 512):
    for C in (0.1, 1.0):
        run('center', C, pca)
