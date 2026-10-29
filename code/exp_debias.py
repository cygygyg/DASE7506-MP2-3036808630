"""Remove support/style/answer nuisance directions from centred embeddings, then rank."""
import sys, io
from fractions import Fraction
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
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
cn = ftr['cand_names']; iS = {k: i for i, k in enumerate(cn)}
sets = {
    'none': np.zeros((Str.shape[0], 0)),
    'support': np.hstack([Gtr, Str[:, [iS['v_rank'], iS['v_rel_max'], iS['v_int']]]]),
    'style': Str[:, [iS['t_len'], iS['t_words'], iS['t_lines'], iS['t_equals'], iS['t_boxed'],
                     iS['t_hash'], iS['t_latex'], iS['t_frac'], iS['a_n_eq'], iS['a_n_ops']]],
    'answer': Str[:, [iS['v_log'], iS['v_rank'], iS['v_rel_max'], iS['v_digits']]],
    'all': np.hstack([Gtr, Str]),
}


def apply_resid(Ztr, Rtr, Zother, alpha=1.0):
    # ridge R ~ Z, with intercept
    Z1 = np.hstack([np.ones((len(Ztr), 1)), Ztr])
    A = Z1.T @ Z1 + alpha * np.eye(Z1.shape[1]); A[0, 0] -= alpha
    B = np.linalg.solve(A, Z1.T @ Rtr)
    out = []
    for Z in Zother:
        Z1o = np.hstack([np.ones((len(Z), 1)), Z])
        out.append(Rtr - Z1o @ B if Z is Rtr else None)
    return B


for sname, Ztr in sets.items():
    if sname == 'none':
        rr_tr, rr_va, rr_te = rtr, rva, rte
    else:
        Z1 = np.hstack([np.ones((len(Ztr), 1)), Ztr])
        A = Z1.T @ Z1 + 1.0 * np.eye(Z1.shape[1]); A[0, 0] -= 1.0
        B = np.linalg.solve(A, Z1.T @ rtr)
        def resid(r, Z):
            return r - np.hstack([np.ones((len(Z), 1)), Z]) @ B
        rr_tr = resid(rtr, Ztr)
        rr_va = resid(rva, np.hstack([Gva, Sva[:, [iS['v_rank'], iS['v_rel_max'], iS['v_int']]]]) if sname == 'support'
                      else Sva[:, [iS['v_log'], iS['v_rank'], iS['v_rel_max'], iS['v_digits']]] if sname == 'answer'
                      else Sva[:, [iS['t_len'], iS['t_words'], iS['t_lines'], iS['t_equals'], iS['t_boxed'],
                                   iS['t_hash'], iS['t_latex'], iS['t_frac'], iS['a_n_eq'], iS['a_n_ops']]] if sname == 'style'
                      else np.hstack([Gva, Sva]))
        rr_te = resid(rte, np.hstack([Gte, Ste[:, [iS['v_rank'], iS['v_rel_max'], iS['v_int']]]]) if sname == 'support'
                      else Ste[:, [iS['v_log'], iS['v_rank'], iS['v_rel_max'], iS['v_digits']]] if sname == 'answer'
                      else Ste[:, [iS['t_len'], iS['t_words'], iS['t_lines'], iS['t_equals'], iS['t_boxed'],
                                   iS['t_hash'], iS['t_latex'], iS['t_frac'], iS['a_n_eq'], iS['a_n_ops']]] if sname == 'style'
                      else np.hstack([Gte, Ste]))
    for C in (0.1, 0.3, 1.0):
        sc = StandardScaler().fit(rr_tr)
        clf = LogisticRegression(C=C, max_iter=3000, solver='lbfgs', random_state=0).fit(sc.transform(rr_tr), tr['labels'].reshape(-1))
        rv = eval_choice(mask(va, clf.predict_proba(sc.transform(rr_va))[:, 1]).argmax(1), va)
        rt = eval_choice(mask(te, clf.predict_proba(sc.transform(rr_te))[:, 1]).argmax(1), te)
        print(fmt(rv, f'resid={sname} C={C}') + f' tl={test_like_metric(rv):5.2f} || ' + fmt(rt, 'TEST'))
