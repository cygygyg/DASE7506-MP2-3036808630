"""Three experts + OOF stack: N (plain), A (answer-debiased), S (support-debiased).

Saves OOF scores and evaluates oracle / top-K routing.
"""
import sys, io, pickle
from fractions import Fraction
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric
from mp2features import build

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
ftr = build('train', tr); fva = build('validation', va); fte = build('test', te)
cn = ftr['cand_names']; iS = {k: i for i, k in enumerate(cn)}


def is_int(a): return a is not None and Fraction(a).denominator == 1
def mask_rows(p, ans):
    p = p.copy()
    for i in range(len(p)):
        for c in range(4):
            if not is_int(ans[i, c]): p[i, c] = -1e9
    return p


def base(d, f):
    N = len(d['bank'])
    r = d['emb'] - d['emb'].mean(axis=1, keepdims=True)
    r = r.reshape(N * 4, -1).astype(np.float64)
    S = f['cand'].reshape(N * 4, -1).astype(np.float64)
    G = np.zeros((N * 4, 6))
    vs = []
    for i, rec in enumerate(f['records']):
        gids = rec['gids']; sizes = rec['sizes'].astype(float)
        vs.append(sizes[rec['valid']])
        for c in range(4):
            g = gids[c]
            G[i * 4 + c] = [sizes[g], float(sizes[g] == sizes.max()), float(sizes[g] == 1),
                            float(np.sum(sizes > sizes[g])), float(len(sizes)), sizes.max()]
    Zans = S[:, [iS['v_log'], iS['v_rank'], iS['v_rel_max'], iS['v_digits'], iS['v_int']]]
    Zsup = np.hstack([G, S[:, [iS['v_rank'], iS['v_rel_max'], iS['v_int']]]])
    return r, Zans, Zsup, vs


rtr, Atr, Str, gtr = base(tr, ftr); rva, Ava, Sva, gva = base(va, fva); rte, Ate, Ste, gte = base(te, fte)


def resid_fit(r, Z, alpha=1.0):
    Z1 = np.hstack([np.ones((len(Z), 1)), Z])
    M = Z1.T @ Z1 + alpha * np.eye(Z1.shape[1]); M[0, 0] -= alpha
    return np.linalg.solve(M, Z1.T @ r)


def apply_B(r, Z, B):
    return r - np.hstack([np.ones((len(Z), 1)), Z]) @ B


def fit_expert(r, y, C):
    sc = StandardScaler().fit(r)
    clf = LogisticRegression(C=C, max_iter=3000, solver='lbfgs', random_state=0).fit(sc.transform(r), y)
    return sc, clf


def fit_all(r, Za, Zs, y, C=0.3):
    scN, clN = fit_expert(r, y, C)
    Ba = resid_fit(r, Za); scA, clA = fit_expert(apply_B(r, Za, Ba), y, 0.1)
    Bs = resid_fit(r, Zs); scS, clS = fit_expert(apply_B(r, Zs, Bs), y, C)
    return dict(scN=scN, clN=clN, Ba=Ba, scA=scA, clA=clA, Bs=Bs, scS=scS, clS=clS)


def predict_all(m, r, Za, Zs, ans):
    pN = m['clN'].predict_proba(m['scN'].transform(r))[:, 1].reshape(-1, 4)
    pA = m['clA'].predict_proba(m['scA'].transform(apply_B(r, Za, m['Ba'])))[:, 1].reshape(-1, 4)
    pS = m['clS'].predict_proba(m['scS'].transform(apply_B(r, Zs, m['Bs'])))[:, 1].reshape(-1, 4)
    return tuple(mask_rows(p, ans) for p in (pN, pA, pS))


yall = tr['labels'].reshape(-1)
oof = {k: np.full((len(tr['bank']), 4), -1e9) for k in 'NAS'}
for itr, iva in KFold(5, shuffle=True, random_state=0).split(np.arange(len(tr['bank']))):
    m4 = (np.asarray(itr)[:, None] * 4 + np.arange(4)).ravel()
    v4 = (np.asarray(iva)[:, None] * 4 + np.arange(4)).ravel()
    q = np.asarray(iva)
    m = fit_all(rtr[m4], Atr[m4], Str[m4], yall[m4])
    pN, pA, pS = predict_all(m, rtr[v4], Atr[v4], Str[v4], tr['answers'][q])
    oof['N'][q], oof['A'][q], oof['S'][q] = pN, pA, pS

mfull = fit_all(rtr, Atr, Str, yall)
Eva = predict_all(mfull, rva, Ava, Sva, va['answers'])
Ete = predict_all(mfull, rte, Ate, Ste, te['answers'])
print('train OOF acc', {k: float(tr['labels'][np.arange(len(oof[k])), oof[k].argmax(1)].mean()) for k in oof})
print('val acc', {k: float(va['labels'][np.arange(len(va['bank'])), Eva[i].argmax(1)].mean()) for i, k in enumerate('NAS')})
print('test acc', {k: float(te['labels'][np.arange(len(te['bank'])), Ete[i].argmax(1)].mean()) for i, k in enumerate('NAS')})

nc_va, nc_te = ncorrect(va), ncorrect(te)
for name, d, E in (('val', va, Eva), ('test', te, Ete)):
    print(f'===== {name} =====')
    for i, k in enumerate('NAS'):
        r = eval_choice(E[i].argmax(1), d)
        print(fmt(r, k) + (f' tl={test_like_metric(r):5.2f}' if name == 'val' else ''))
    # oracle any expert correct
    ok = np.stack([d['labels'][np.arange(len(d['bank'])), e.argmax(1)] for e in E])
    print('oracle any:', 100 * ok.max(0).mean(), ' N or A:', 100 * ok[:2].max(0).mean(), ' N/A/S:', 100 * ok.max(0).mean())
    for kk in (1, 2, 3):
        m = nc_va == kk if name == 'val' else nc_te == kk
        print(f'  #{kk} oracle {100*ok[:, m].max(0).mean():.1f}  N {100*ok[0,m].mean():.1f} A {100*ok[1,m].mean():.1f} S {100*ok[2,m].mean():.1f}')

with Path('cache/oof_experts.pkl').open('wb') as fh:
    pickle.dump(dict(oof=oof, Eva=Eva, Ete=Ete, gtr=gtr, gva=gva, gte=gte), fh, protocol=4)
