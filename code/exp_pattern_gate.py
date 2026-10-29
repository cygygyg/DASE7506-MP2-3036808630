"""Per-pattern gates: decide between the support group and the best-evidence alternative."""
import sys, io
from fractions import Fraction
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingClassifier
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric, auc
from mp2features import build

tr = load_split('train'); va = load_split('validation'); te = load_split('test')


def centred(d): return (d['emb'] - d['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
def is_int(a): return a is not None and Fraction(a).denominator == 1


def fit_base(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', random_state=0).fit(sc.transform(X), y)
    return sc, clf


Xtr = centred(tr); ytr = tr['labels'].reshape(-1)
scf, bf = fit_base(Xtr, ytr)
oof = np.zeros((len(tr['bank']), 4))
for itr, iva in KFold(5, shuffle=True, random_state=0).split(np.arange(len(tr['bank']))):
    m = (np.asarray(itr)[:, None] * 4 + np.arange(4)).ravel()
    mv = (np.asarray(iva)[:, None] * 4 + np.arange(4)).ravel()
    sc, clf = fit_base(Xtr[m], ytr[m])
    oof[iva] = clf.predict_proba(sc.transform(Xtr[mv]))[:, 1].reshape(-1, 4)


def mask(d, p):
    for i in range(len(p)):
        for c in range(4):
            if not is_int(d['answers'][i, c]): p[i, c] = -1e9
    return p


oof = mask(tr, oof)
pva = mask(va, bf.predict_proba(scf.transform(centred(va)))[:, 1].reshape(-1, 4))
pte = mask(te, bf.predict_proba(scf.transform(centred(te)))[:, 1].reshape(-1, 4))
print('base OOF acc', eval_choice(oof.argmax(1), tr)['acc'])


def groups_of(d, i):
    ans = d['answers'][i]; g = {}
    for c, a in enumerate(ans):
        if not is_int(a): continue
        g.setdefault(a, []).append(c)
    keys = list(g); sizes = np.array([len(g[k]) for k in keys])
    gs = np.array([max(1e-9 if False else 0, 0) for _ in keys], float)
    return keys, g, sizes


def make(d, feats, scores):
    """One decision example per question with >=2 valid groups."""
    rows = []
    for i in range(len(d['bank'])):
        ans = d['answers'][i]; groups = {}
        for c, a in enumerate(ans):
            if not is_int(a): continue
            groups.setdefault(a, []).append(c)
        keys = list(groups)
        if len(keys) < 2: continue
        sizes = np.array([len(groups[k]) for k in keys]) if False else np.array([len(groups[k]) for k in keys])
        gs = np.array([max(scores[i, c] for c in groups[k]) for k in keys])
        labs = np.array([int(any(d['labels'][i, c] for c in groups[k])) for k in keys])
        pat = tuple(sorted(sizes.tolist(), reverse=True))
        imax = int(np.argmax(sizes))
        others = [g for g in range(len(keys)) if g != imax]
        # is there a unique largest group?
        has_unique_max = bool(others) and sizes[imax] > max(sizes[g] for g in others)
        best_other = max(others, key=lambda g: gs[g]) if others else imax
        # group-level feature aggregates of support vs alternative
        def gf(g):
            arr = feats['group'][i][g]
            names = feats['group_names']
            return arr
        fsup = gf(imax); falt = gf(best_other)
        # selected candidate features
        csup = int(np.argmax([scores[i, c] for c in groups[keys[imax]]]))
        csup = groups[keys[imax]][csup]
        calt = groups[keys[best_other]][int(np.argmax([scores[i, c] for c in groups[keys[best_other]]]))]
        cf = feats['cand'][i]
        cnames = feats['cand_names']
        def cv(c, k): return cf[c, cnames.index(k)]
        sf = [
            gs[imax] - gs[best_other],
            gs[imax], gs[best_other], gs.std(),
            sizes[imax], len(keys), float(has_unique_max),
            sizes[imax] - sizes[best_other],
            float(labs[imax]) if False else 0.0,
            cv(csup, 'a_bad_ratio') - cv(calt, 'a_bad_ratio'),
            cv(csup, 'a_n_eq') - cv(calt, 'a_n_eq'),
            cv(csup, 'a_last_ok') - cv(calt, 'a_last_ok'),
            cv(csup, 'a_final_eq_last_rhs') - cv(calt, 'a_final_eq_last_rhs'),
            cv(csup, 'c_chain_link_ratio') - cv(calt, 'c_chain_link_ratio'),
            cv(csup, 'c_last_operand_support') - cv(calt, 'c_last_operand_support'),
            cv(csup, 'c_final_from_rhs') - cv(calt, 'c_final_from_rhs'),
            cv(csup, 'c_unknown_ratio') - cv(calt, 'c_unknown_ratio'),
            cv(csup, 'v_log') - cv(calt, 'v_log'),
            cv(csup, 'v_rank') - cv(calt, 'v_rank'),
            cv(csup, 't_len') - cv(calt, 't_len'),
            float(csup == int(np.argmax(scores[i]))),
            float(np.sum(scores[i] > scores[i, csup])),
            gs[imax] - gs.mean(),
        ]
        qf = list(feats['q'][i])
        rows.append(dict(i=i, pat=pat, imax=imax, best_other=best_other, keys=keys,
                         groups=groups, sizes=sizes, gs=gs, labs=labs,
                         x=np.array(sf + qf, float), unique_max=has_unique_max))
    return rows


ftr = build('train', tr); fva = build('validation', va); fte = build('test', te)
rtr = make(tr, ftr, oof); rva = make(va, fva, pva); rte = make(te, fte, pte)
print('decision rows', len(rtr), len(rva), len(rte))


def fit_gate(rows, kind='logit', C=1.0):
    X = np.array([r['x'] for r in rows]); y = np.array([int(r['labs'][r['imax']]) for r in rows])
    if kind == 'logit':
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=C, max_iter=5000, solver='lbfgs', random_state=0).fit(sc.transform(X), y)
        return lambda Z: clf.predict_proba(sc.transform(Z))[:, 1]
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                         min_samples_leaf=20, l2_regularization=1.0, random_state=0)
    clf.fit(X, y)
    return lambda Z: clf.predict_proba(Z)[:, 1]


def apply(d, rows, proba, use_gate_patterns=('all',)):
    """For each row, choose support if p>=0.5 else best_other; for non-unique-max patterns use baseline."""
    N = len(d['bank']); choice = np.zeros(N, dtype=int)
    for r in rows:
        i = r['i']
        pat = r['pat']
        base_g = max(range(len(r['gs'])), key=lambda g: r['gs'][g])
        if r['unique_max']:
            p = proba(r['x'][None, :])[0]
            g = r['imax'] if p >= 0.5 else r['best_other']
        else:
            g = base_g
        choice[i] = r['groups'][r['keys'][g]][0]
    # questions with 1 valid group -> any valid candidate
    for i in range(N):
        if not any(True for _ in []): pass
    covered = set(r['i'] for r in rows)
    for i in range(N):
        if i in covered: continue
        valid = [c for c in range(4) if d['answers'][i, c] is not None and is_int(d['answers'][i, c])]
        choice[i] = valid[0] if valid else 0
    return eval_choice(choice, d)


for kind in ('logit', 'hgb'):
    for C in ((0.1, 1.0) if kind == 'logit' else (0,)):
        proba = fit_gate(rtr, kind, C)
        print(f'--- gate {kind} C={C} ---')
        for name, d, rows in (('val', va, rva), ('test', te, rte)):
            r = apply(d, rows, proba)
            print(fmt(r, f'{name} gate') + (f' testlike={test_like_metric(r):5.2f}' if name == 'val' else ''))
