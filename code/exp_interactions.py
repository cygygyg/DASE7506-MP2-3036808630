"""Candidate-level model with answer/support-conditioned embedding interactions."""
import sys, io
from fractions import Fraction
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, eval_choice, fmt, test_like_metric
from mp2features import build

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
ftr = build('train', tr); fva = build('validation', va); fte = build('test', te)


def is_int(a): return a is not None and Fraction(a).denominator == 1


def mask(d, p):
    p = p.reshape(-1, 4).copy()
    for i in range(len(p)):
        for c in range(4):
            if not is_int(d['answers'][i, c]): p[i, c] = -1e9
    return p


def build_feats(d, f):
    """Return dict of scalar features (N,4,K) and group-context features."""
    N = len(d['bank'])
    # candidate text/arith/chain features already in f['cand']
    X = f['cand'].astype(np.float64)  # (N,4,68)
    names = f['cand_names']
    gi = {k: i for i, k in enumerate(names)}
    # group context per candidate
    G = np.zeros((N, 4, 12))
    for i, rec in enumerate(f['records']):
        gids = rec['gids']; sizes = rec['sizes'].astype(float); valid = rec['valid']
        for c in range(4):
            g = gids[c]
            G[i, c] = [
                sizes[g], sizes[g] / max(1, sizes.sum()), sizes.max(),
                float(sizes[g] == sizes.max()), float(sizes[g] == sizes.min()),
                float(np.sum(sizes > sizes[g])), float(np.sum(sizes < sizes[g])),
                float(len(sizes)), float(valid[g]),
                float(sizes[g] == 1), float(np.sum(sizes == sizes[g])),
                float(g),
            ]
    gnames = ['grp_size', 'grp_frac', 'grp_max', 'grp_is_max', 'grp_is_min',
              'grp_rank_desc', 'grp_rank_asc', 'grp_n', 'grp_valid', 'grp_single',
              'grp_tie', 'grp_id']
    return X, names, G, gnames


Xtr, cnames, Gtr, gnames = build_feats(tr, ftr)
Xva, _, Gva, _ = build_feats(va, fva)
Xte, _, Gte, _ = build_feats(te, fte)


def tensor(d, f, X, G):
    r = d['emb'] - d['emb'].mean(axis=1, keepdims=True)  # (N,4,1024)
    N = r.shape[0]
    r = r.reshape(N * 4, -1)
    # standardise r globally
    r = (r - r.mean(0)) / (r.std(0) + 1e-6)
    r = r.reshape(N, 4, 1024)
    Xf = X.reshape(N * 4, -1)
    Gf = G.reshape(N * 4, -1)
    # standardise scalar blocks
    return r, Xf, Gf


rtr, Str, Gtr2 = tensor(tr, ftr, Xtr, Gtr)
rva, Sva, Gva2 = tensor(va, fva, Xva, Gva)
rte, Ste, Gte2 = tensor(te, fte, Xte, Gte)

scS = StandardScaler().fit(Str)
Str = scS.transform(Str); Sva = scS.transform(Sva); Ste = scS.transform(Ste)
scG = StandardScaler().fit(Gtr2)
Gtr2 = scG.transform(Gtr2); Gva2 = scG.transform(Gva2); Gte2 = scG.transform(Gte2)


def make_view(r, S, G, cols_S, cols_G, include_r=True):
    """r (N,4,1024) standardised; scalar blocks standardised (N*4,K)."""
    N = r.shape[0]
    views = []
    if include_r:
        views.append(r.reshape(N * 4, -1))
    for cols, Sblock in ((cols_S, S), (cols_G, G)):
        for j in cols:
            v = Sblock[:, j]
            views.append((r.reshape(N * 4, -1) * v[:, None]))
    return np.hstack(views)


def run(name, cols_S, cols_G, C=0.3):
    A = make_view(rtr, Str, Gtr2, cols_S, cols_G)
    B = make_view(rva, Sva, Gva2, cols_S, cols_G)
    D = make_view(rte, Ste, Gte2, cols_S, cols_G)
    sc = StandardScaler().fit(A)
    clf = LogisticRegression(C=C, max_iter=6000, solver='lbfgs', random_state=0).fit(sc.transform(A), tr['labels'].reshape(-1))
    rv = eval_choice(mask(va, clf.predict_proba(sc.transform(B))[:, 1]).argmax(1), va)
    rt = eval_choice(mask(te, clf.predict_proba(sc.transform(D))[:, 1]).argmax(1), te)
    print(fmt(rv, f'{name} C={C}') + f' tl={test_like_metric(rv):5.2f} || ' + fmt(rt, 'TEST'))


iS = {k: i for i, k in enumerate(cnames)}
iG = {k: i for i, k in enumerate(gnames)}
print('--- baseline scalar only ---')
run('scalars', range(Str.shape[1]), range(12), C=1.0)
print('--- r only ---')
run('r_only', [], [], C=0.3)
print('--- r + scalar ---')
run('r+scalars', range(Str.shape[1]), range(12), C=0.3)
print('--- r x answer value ---')
run('r x v_log', [iS['v_log'], iS['v_rank'], iS['v_rel_max'], iS['v_int']], [], C=0.3)
print('--- r x support ---')
run('r x support', [], [iG['grp_size'], iG['grp_is_max'], iG['grp_single'], iG['grp_rank_asc']], C=0.3)
print('--- r x answer + support ---')
run('r x ans+sup', [iS['v_log'], iS['v_rank'], iS['v_rel_max'], iS['v_int']],
    [iG['grp_size'], iG['grp_is_max'], iG['grp_single'], iG['grp_rank_asc']], C=0.3)
print('--- r x all scalar + support ---')
run('r x all', range(Str.shape[1]), range(12), C=0.3)
