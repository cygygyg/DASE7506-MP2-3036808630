"""Focused interaction experiment (few views, fast)."""
import sys, io, time
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


def assemble(d, f):
    N = len(d['bank'])
    r = d['emb'] - d['emb'].mean(axis=1, keepdims=True)
    r = r.reshape(N * 4, -1)
    r = (r - r.mean(0)) / (r.std(0) + 1e-6)
    S = f['cand'].reshape(N * 4, -1).astype(np.float64)
    G = np.zeros((N * 4, 10))
    for i, rec in enumerate(f['records']):
        gids = rec['gids']; sizes = rec['sizes'].astype(float)
        for c in range(4):
            g = gids[c]
            G[i * 4 + c] = [sizes[g], sizes[g] / max(1, sizes.sum()), sizes.max(),
                            float(sizes[g] == sizes.max()), float(sizes[g] == sizes.min()),
                            float(np.sum(sizes > sizes[g])), float(np.sum(sizes < sizes[g])),
                            float(len(sizes)), float(sizes[g] == 1), float(np.sum(sizes == sizes[g]))]
    return r.reshape(N, 4, -1), S, G


rtr, Str, Gtr = assemble(tr, ftr); rva, Sva, Gva = assemble(va, fva); rte, Ste, Gte = assemble(te, fte)
scS = StandardScaler().fit(Str); Str, Sva, Ste = scS.transform(Str), scS.transform(Sva), scS.transform(Ste)
scG = StandardScaler().fit(Gtr); Gtr, Gva, Gte = scG.transform(Gtr), scG.transform(Gva), scG.transform(Gte)
cnames = ftr['cand_names']; iS = {k: i for i, k in enumerate(cnames)}


def build(spec):
    """spec: list of ('r',), ('s',), ('g',), ('rs', col), ('rg', col)."""
    outs = []
    for sp in spec:
        kind = sp[0]
        if kind == 'r':
            outs.append(rtr.reshape(-1, 1024))
        elif kind == 's':
            outs.append(Str)
        elif kind == 'g':
            outs.append(Gtr)
        elif kind == 'rs':
            outs.append(rtr.reshape(-1, 1024) * Str[:, sp[1]][:, None])
        elif kind == 'rg':
            outs.append(rtr.reshape(-1, 1024) * Gtr[:, sp[1]][:, None])
    return np.hstack(outs)


def build_eval(spec, d, r, S, G):
    outs = []
    for sp in spec:
        kind = sp[0]
        if kind == 'r': outs.append(r.reshape(-1, 1024))
        elif kind == 's': outs.append(S)
        elif kind == 'g': outs.append(G)
        elif kind == 'rs': outs.append(r.reshape(-1, 1024) * S[:, sp[1]][:, None])
        elif kind == 'rg': outs.append(r.reshape(-1, 1024) * G[:, sp[1]][:, None])
    return np.hstack(outs)


def run(name, spec, C=0.3):
    t0 = time.time()
    A = build(spec); sc = StandardScaler().fit(A)
    clf = LogisticRegression(C=C, max_iter=2000, solver='lbfgs', random_state=0).fit(sc.transform(A), tr['labels'].reshape(-1))
    out = []
    for d, r, S, G in ((va, rva, Sva, Gva), (te, rte, Ste, Gte)):
        B = build_eval(spec, d, r, S, G)
        p = clf.predict_proba(sc.transform(B))[:, 1]
        out.append(eval_choice(mask(d, p).argmax(1), d))
    print(fmt(out[0], f'{name} C={C}') + f' tl={test_like_metric(out[0]):5.2f} || ' + fmt(out[1], 'TEST') + f' [{time.time()-t0:.0f}s]', flush=True)


run('scalars', [('s',), ('g',)], 1.0)
run('r_only', [('r',)], 0.3)
run('r+scalars', [('r',), ('s',), ('g',)], 0.3)
run('r x v_log', [('r',), ('rs', iS['v_log']), ('rs', iS['v_rank']), ('rs', iS['v_int']), ('s',), ('g',)], 0.3)
run('r x support', [('r',), ('rg', 0), ('rg', 3), ('rg', 8), ('s',), ('g',)], 0.3)
run('r x ans+sup', [('r',), ('rs', iS['v_log']), ('rs', iS['v_rank']), ('rg', 0), ('rg', 3), ('rg', 8), ('s',), ('g',)], 0.3)
