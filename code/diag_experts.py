"""Two experts: normal centred ranker (N) and support-debiased ranker (D). Oracle + routing headroom."""
import sys, io
from fractions import Fraction
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric
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
Zsup_tr = np.hstack([Gtr, Str[:, [iS['v_rank'], iS['v_rel_max'], iS['v_int']]]])
Zsup_va = np.hstack([Gva, Sva[:, [iS['v_rank'], iS['v_rel_max'], iS['v_int']]]])
Zsup_te = np.hstack([Gte, Ste[:, [iS['v_rank'], iS['v_rel_max'], iS['v_int']]]])

# nuisance regression
Z1 = np.hstack([np.ones((len(Zsup_tr), 1)), Zsup_tr])
A = Z1.T @ Z1 + np.eye(Z1.shape[1]); A[0, 0] -= 1
B = np.linalg.solve(A, Z1.T @ rtr)
Dtr = rtr - Z1 @ B
Dva = rva - np.hstack([np.ones((len(Zsup_va), 1)), Zsup_va]) @ B
Dte = rte - np.hstack([np.ones((len(Zsup_te), 1)), Zsup_te]) @ B


def fit(X, C=1.0):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=3000, solver='lbfgs', random_state=0).fit(sc.transform(X), tr['labels'].reshape(-1))
    return sc, clf


scN, clN = fit(rtr, 0.3)
scD, clD = fit(Dtr, 0.3)


def scores(sc, clf, X, d):
    return mask(d, clf.predict_proba(sc.transform(X))[:, 1])


Nva, Nte = scores(scN, clN, rva, va), scores(scN, clN, rte, te)
Dva, Dte = scores(scD, clD, Dva, va), scores(scD, clD, Dte, te)


def report(name, d, N, D):
    nc = ncorrect(d)
    chN, chD = N.argmax(1), D.argmax(1)
    okN = d['labels'][np.arange(len(d['bank'])), chN]
    okD = d['labels'][np.arange(len(d['bank'])), chD]
    both = okN | okD
    print(f'===== {name} =====')
    print(f'  N {okN.mean()*100:.2f}  D {okD.mean()*100:.2f}  oracle {both.mean()*100:.2f}  agree {np.mean(chN==chD)*100:.1f}%')
    for k in (1,2,3):
        m = nc==k
        print(f'  #{k}: n={m.sum()} N={100*okN[m].mean():.1f} D={100*okD[m].mean():.1f} oracle={100*both[m].mean():.1f} '
              f'D_wins={int((~okN[m] & okD[m]).sum())} N_wins={int((okN[m] & ~okD[m]).sum())}')
    # by pattern
    pats = defaultdict(lambda: np.zeros(4))
    for i in range(len(d['bank'])):
        ans = d['answers'][i]; groups = {}
        for c, a in enumerate(ans):
            if not is_int(a): continue
            groups.setdefault(a, []).append(c)
        sizes = tuple(sorted((len(v) for v in groups.values()), reverse=True))
        if len(sizes) < 2: continue
        pats[sizes] += [1, okN[i], okD[i], okN[i] | okD[i]]
    print('  pattern: n  N% D% oracle%')
    for p, v in sorted(pats.items(), key=lambda kv: -kv[1][0])[:8]:
        n = v[0]
        print(f'    {p} n={int(n)} N={100*v[1]/n:.1f} D={100*v[2]/n:.1f} O={100*v[3]/n:.1f}')


report('validation', va, Nva, Dva)
report('test', te, Nte, Dte)
