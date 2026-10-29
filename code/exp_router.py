"""Learn a router between the normal ranker (N) and the support-debiased ranker (D).

All router training uses out-of-fold expert scores on train.
"""
import sys, io, pickle
from fractions import Fraction
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric
from mp2features import build

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
ftr = build('train', tr); fva = build('validation', va); fte = build('test', te)
cn = ftr['cand_names']; iS = {k: i for i, k in enumerate(cn)}


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
    gsizes = []
    for i, rec in enumerate(f['records']):
        gids = rec['gids']; sizes = rec['sizes'].astype(float)
        gsizes.append(sizes[rec['valid']])
        for c in range(4):
            g = gids[c]
            G[i * 4 + c] = [sizes[g], float(sizes[g] == sizes.max()), float(sizes[g] == 1),
                            float(np.sum(sizes > sizes[g])), float(len(sizes)), sizes.max()]
    Zsup = np.hstack([G, S[:, [iS['v_rank'], iS['v_rel_max'], iS['v_int']]]])
    return r, Zsup, gsizes


rtr, Ztr, gtr = base(tr, ftr); rva, Zva, gva = base(va, fva); rte, Zte, gte = base(te, fte)


def fit_models(r, Z, y, C=0.3):
    scN = StandardScaler().fit(r)
    clN = LogisticRegression(C=C, max_iter=3000, solver='lbfgs', random_state=0).fit(scN.transform(r), y)
    # nuisance regression on this fit set
    Z1 = np.hstack([np.ones((len(Z), 1)), Z])
    A = Z1.T @ Z1 + np.eye(Z1.shape[1]); A[0, 0] -= 1
    B = np.linalg.solve(A, Z1.T @ r)
    Dr = r - Z1 @ B
    scD = StandardScaler().fit(Dr)
    clD = LogisticRegression(C=C, max_iter=3000, solver='lbfgs', random_state=0).fit(scD.transform(Dr), y)
    return dict(scN=scN, clN=clN, B=B, scD=scD, clD=clD)


def apply_models(m, r, Z):
    pN = m['clN'].predict_proba(m['scN'].transform(r))[:, 1].reshape(-1, 4)
    Dr = r - np.hstack([np.ones((len(Z), 1)), Z]) @ m['B']
    pD = m['clD'].predict_proba(m['scD'].transform(Dr))[:, 1].reshape(-1, 4)
    return pN, pD


def mask_rows(p, ans):
    p = p.copy()
    for i in range(len(p)):
        for c in range(4):
            if not is_int(ans[i, c]):
                p[i, c] = -1e9
    return p


# OOF expert scores
oofN = np.zeros((len(tr['bank']), 4)); oofD = np.zeros_like(oofN)
yall = tr['labels'].reshape(-1)
for itr, iva in KFold(5, shuffle=True, random_state=0).split(np.arange(len(tr['bank']))):
    m4 = (np.asarray(itr)[:, None] * 4 + np.arange(4)).ravel()
    v4 = (np.asarray(iva)[:, None] * 4 + np.arange(4)).ravel()
    m = fit_models(rtr[m4], Ztr[m4], yall[m4])
    pN, pD = apply_models(m, rtr[v4], Ztr[v4])
    qiva = np.asarray(iva)
    oofN[qiva] = mask_rows(pN, tr['answers'][qiva])
    oofD[qiva] = mask_rows(pD, tr['answers'][qiva])
# note: pN was already masked against full tr by index; reconstruct by masking again
for i in range(len(oofN)):
    for c in range(4):
        if not is_int(tr['answers'][i, c]):
            oofN[i, c] = -1e9; oofD[i, c] = -1e9

mfull = fit_models(rtr, Ztr, yall)
Nva, Dva = apply_models(mfull, rva, Zva)
Nte, Dte = apply_models(mfull, rte, Zte)
Nva, Dva = mask_rows(Nva, va['answers']), mask_rows(Dva, va['answers'])
Nte, Dte = mask_rows(Nte, te['answers']), mask_rows(Dte, te['answers'])


def expert_acc(d, p):
    return d['labels'][np.arange(len(d['bank'])), p.argmax(1)].astype(int)


okN_oof = expert_acc(tr, oofN); okD_oof = expert_acc(tr, oofD)
okN_va, okD_va = expert_acc(va, Nva), expert_acc(va, Dva)
okN_te, okD_te = expert_acc(te, Nte), expert_acc(te, Dte)
nc_tr = ncorrect(tr); nc_va = ncorrect(va); nc_te = ncorrect(te)
print('train OOF N/D', okN_oof.mean(), okD_oof.mean())
print('val N/D', okN_va.mean(), okD_va.mean(), 'test N/D', okN_te.mean(), okD_te.mean())


def qfeat(p, sizes):
    s = np.sort(p)[::-1]
    mx = s[0]; second = s[1]
    e = np.exp(s - mx); prob = e / e.sum()
    ent = -(prob * np.log(prob + 1e-12)).sum()
    return [mx, second, mx - second, s.std(), ent, prob[0]]


def router_features(d, pN, pD, feats, sizes_list):
    X = []
    for i in range(len(d['bank'])):
        sizes = np.array(sizes_list[i], float)
        ss = np.sort(sizes)[::-1]
        pad = np.zeros(4); pad[:min(4, len(ss))] = ss[:4]
        fN = qfeat(pN[i], sizes); fD = qfeat(pD[i], sizes)
        aN = int(np.argmax(pN[i])); aD = int(np.argmax(pD[i]))
        cross = [pN[i, aD] - fN[0], pD[i, aN] - fD[0], float(aN == aD)]
        pat = [len(sizes), float(np.sum(sizes == 1)), float(np.sum(sizes >= 2)),
               float(np.sum(sizes >= 3)), float(len(set(sizes.tolist())) == 1),
               float(ss[0] >= 3 and len(ss) == 2), float(ss[0] == 2 and len(ss) == 2)]
        X.append(np.concatenate([pad, fN, fD, cross, pat, feats['q'][i]]))
    return np.array(X, float)


Xr_tr = router_features(tr, oofN, oofD, ftr, gtr)
Xr_va = router_features(va, Nva, Dva, fva, gva)
Xr_te = router_features(te, Nte, Dte, fte, gte)

# --- router A: predict #1 ---
y1_tr = (nc_tr == 1).astype(int)
sc = StandardScaler().fit(Xr_tr)
for C in (0.1, 1.0):
    clf = LogisticRegression(C=C, max_iter=3000, class_weight='balanced', random_state=0).fit(sc.transform(Xr_tr), y1_tr)
    pva1 = clf.predict_proba(sc.transform(Xr_va))[:, 1]
    pte1 = clf.predict_proba(sc.transform(Xr_te))[:, 1]
    print(f'routerA C={C} AUC val={roc_auc_score((nc_va==1).astype(int),pva1):.3f} '
          f'test={roc_auc_score((nc_te==1).astype(int),pte1):.3f} meanP val={pva1.mean():.3f} test={pte1.mean():.3f}')
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        ch = np.where(pva1 > t, Dva.argmax(1), Nva.argmax(1))
        rv = eval_choice(ch, va)
        print(f'  t={t}: val {rv["acc"]:.2f} #1={rv.get("acc1",0):.1f}', end='')
        ch = np.where(pte1 > t, Dte.argmax(1), Nte.argmax(1))
        rt = eval_choice(ch, te)
        print(f' || test {rt["acc"]:.2f} #1={rt.get("acc1",0):.1f} #2={rt.get("acc2",0):.1f} #3={rt.get("acc3",0):.1f}')

# --- router B: among disagreements, which expert is right ---
dis = (oofN.argmax(1) != oofD.argmax(1))
mask_dis = dis & (okN_oof != okD_oof)
print('disagreements', dis.sum(), 'decisive', mask_dis.sum(), 'D wins', int((mask_dis & (okD_oof==1)).sum()))
yD = okD_oof[mask_dis].astype(int)
scB = StandardScaler().fit(Xr_tr[dis])
clfB = LogisticRegression(C=0.3, max_iter=3000, class_weight='balanced', random_state=0).fit(scB.transform(Xr_tr[dis]), yD)
print('routerB AUC on decisive train (in-sample)', roc_auc_score(yD, clfB.predict_proba(scB.transform(Xr_tr[dis]))[:, 1]))
for name, X, N, D, okN, okD in (('val', Xr_va, Nva, Dva, okN_va, okD_va), ('test', Xr_te, Nte, Dte, okN_te, okD_te)):
    p = clfB.predict_proba(scB.transform(X))[:, 1]
    d_ = (N.argmax(1) != D.argmax(1))
    ch = N.argmax(1).copy()
    ch[d_ & (p > 0.5)] = D.argmax(1)[d_ & (p > 0.5)]
    r = eval_choice(ch, va if name == 'val' else te)
    print(f'routerB {name}: {r["acc"]:.2f} #1={r.get("acc1",0):.1f} #2={r.get("acc2",0):.1f} #3={r.get("acc3",0):.1f}')
