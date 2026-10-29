"""Sweep partial residualisation strength alpha for nuisance directions."""
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
    return r, S, G


rtr, Str, Gtr = base(tr, ftr); rva, Sva, Gva = base(va, fva); rte, Ste, Gte = base(te, fte)
sets = {
    'size': Gtr[:, [0]],
    'plural': Gtr[:, [1]],
    'rank': Gtr[:, [3]],
    'support': Gtr,
    'answer': Str[:, [iS['v_log'], iS['v_rank'], iS['v_rel_max'], iS['v_digits']]],
    'answer_support': np.hstack([Gtr, Str[:, [iS['v_log'], iS['v_rank'], iS['v_int']]]]),
    'style': Str[:, [iS['t_len'], iS['t_lines'], iS['t_equals'], iS['a_n_eq'], iS['t_boxed'], iS['t_hash']]],
}
sets_va = {
    'size': Gva[:, [0]], 'plural': Gva[:, [1]], 'rank': Gva[:, [3]], 'support': Gva,
    'answer': Sva[:, [iS['v_log'], iS['v_rank'], iS['v_rel_max'], iS['v_digits']]],
    'answer_support': np.hstack([Gva, Sva[:, [iS['v_log'], iS['v_rank'], iS['v_int']]]]),
    'style': Sva[:, [iS['t_len'], iS['t_lines'], iS['t_equals'], iS['a_n_eq'], iS['t_boxed'], iS['t_hash']]],
}
sets_te = {
    'size': Gte[:, [0]], 'plural': Gte[:, [1]], 'rank': Gte[:, [3]], 'support': Gte,
    'answer': Ste[:, [iS['v_log'], iS['v_rank'], iS['v_rel_max'], iS['v_digits']]],
    'answer_support': np.hstack([Gte, Ste[:, [iS['v_log'], iS['v_rank'], iS['v_int']]]]),
    'style': Ste[:, [iS['t_len'], iS['t_lines'], iS['t_equals'], iS['a_n_eq'], iS['t_boxed'], iS['t_hash']]],
}
y = tr['labels'].reshape(-1)


def fit_B(Z, r):
    Z1 = np.hstack([np.ones((len(Z), 1)), Z])
    M = Z1.T @ Z1 + np.eye(Z1.shape[1]); M[0, 0] -= 1
    return np.linalg.solve(M, Z1.T @ r)


for name in sets:
    B = fit_B(sets[name], rtr)
    print(f'===== nuisance={name} =====')
    for alpha in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        def adj(r, Z):
            return r - alpha * (np.hstack([np.ones((len(Z), 1)), Z]) @ B)
        Atr = adj(rtr, sets[name]); Ava = adj(rva, sets_va[name]); Ate = adj(rte, sets_te[name])
        for C in (0.1, 0.3):
            sc = StandardScaler().fit(Atr)
            clf = LogisticRegression(C=C, max_iter=3000, solver='lbfgs', random_state=0).fit(sc.transform(Atr), y)
            rv = eval_choice(mask_rows(clf.predict_proba(sc.transform(Ava))[:, 1], va).argmax(1), va)
            rt = eval_choice(mask_rows(clf.predict_proba(sc.transform(Ate))[:, 1], te).argmax(1), te)
            print(f'  a={alpha:.1f} C={C:.1f} val={rv["acc"]:.2f} tl={test_like_metric(rv):.2f} #1={rv["acc1"]:.1f} || '
                  f'test={rt["acc"]:.2f} #1={rt["acc1"]:.1f} #2={rt.get("acc2",0):.1f} #3={rt.get("acc3",0):.1f}')
