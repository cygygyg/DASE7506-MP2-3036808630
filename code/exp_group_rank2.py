"""Stage 2: group ranker with OOF baseline scores + interaction features."""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric
from mp2features import build

tr = load_split('train'); va = load_split('validation'); te = load_split('test')


def centred(d):
    return (d['emb'] - d['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)


def fit_base(X, y, C=1.0):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000, solver='lbfgs', random_state=0).fit(sc.transform(X), y)
    return sc, clf


Xtr = centred(tr); ytr = tr['labels'].reshape(-1)
sc_full, base_full = fit_base(Xtr, ytr)
Xva = centred(va); Xte = centred(te)


def predict_base(sc, clf, X, d):
    p = clf.predict_proba(sc.transform(X))[:, 1].reshape(-1, 4)
    for i in range(len(p)):
        for c in range(4):
            if d['answers'][i, c] is None:
                p[i, c] = -1e9
    return p


# OOF scores on train
oof = np.zeros((len(tr['bank']), 4))
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for fold, (itr, iva) in enumerate(kf.split(np.arange(len(tr['bank'])))):
    m = (np.asarray(itr)[:, None] * 4 + np.arange(4)).ravel()
    mv = (np.asarray(iva)[:, None] * 4 + np.arange(4)).ravel()
    sc, clf = fit_base(Xtr[m], ytr[m])
    oof[iva] = clf.predict_proba(sc.transform(Xtr[mv]))[:, 1].reshape(-1, 4)
for i in range(len(oof)):
    for c in range(4):
        if tr['answers'][i, c] is None:
            oof[i, c] = -1e9
print('OOF computed')

pva = predict_base(sc_full, base_full, Xva, va)
pte = predict_base(sc_full, base_full, Xte, te)


def group_rows(d, feats, scores):
    rows = []
    for qi, rec in enumerate(feats['records']):
        gids = rec['gids']; G = rec['G']; sizes = rec['sizes']; valid = rec['valid']
        if G < 2:
            continue
        gs = []
        for gi in range(G):
            members = np.where(gids == gi)[0]
            s = scores[qi, members]
            gs.append((s.max(), s.mean(), s.min(), s[0] if len(s) else -1e9))
        gs = np.array(gs)
        imax_size = int(np.argmax(sizes))
        gmax_score = gs[:, 0].max()
        for gi in range(G):
            if not valid[gi]:
                continue
            y = int(d['labels'][qi, gids == gi].max())
            other = np.delete(gs[:, 0], gi)
            best_other = other.max() if len(other) else -1e9
            s = gs[gi]
            gap = s[0] - best_other
            sf = [
                s[0], s[1], s[2], gap,
                s[0] - gmax_score, s[0] - gs[imax_size, 0],
                float(np.sum(gs[:, 0] > s[0])), float(np.sum(gs[:, 0] < s[0])),
                sizes[gi], sizes[gi] - sizes.max(),
                float(gi == imax_size), float(sizes[gi] == sizes.max()),
                float(np.sum(sizes > sizes[gi])), float(np.sum(sizes < sizes[gi])),
                G,
                # interactions
                gap * sizes[gi], gap * (sizes[gi] - sizes.max()),
                gap * float(gi == imax_size), gap * G,
                s[0] * sizes[gi], gap * float(np.sum(sizes > sizes[gi])),
                (s[0] - gmax_score) * float(gi == imax_size),
                gap * float(sizes[gi] == sizes.max()),
            ]
            x = np.concatenate([sf, feats['group'][qi][gi], feats['q'][qi]])
            rows.append((qi, gi, y, x))
    return rows


ftr = build('train', tr); fva = build('validation', va); fte = build('test', te)
rows_tr = group_rows(tr, ftr, oof)
rows_va = group_rows(va, fva, pva)
rows_te = group_rows(te, fte, pte)


def fit(rows_tr, rows_va, rows_te, C=0.1, weight_mode='question'):
    X = np.array([r[3] for r in rows_tr], float); y = np.array([r[2] for r in rows_tr])
    byq = defaultdict(list)
    for k, r in enumerate(rows_tr):
        byq[r[0]].append(k)
    w = np.ones(len(rows_tr))
    for qi, ks in byq.items():
        pos = [k for k in ks if y[k] == 1]; neg = [k for k in ks if y[k] == 0]
        for k in pos: w[k] = 1.0 / max(1, len(pos))
        for k in neg: w[k] = 1.0 / max(1, len(neg))
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=8000, solver='lbfgs', random_state=0).fit(sc.transform(X), y, sample_weight=w)
    out = {}
    for name, d, rows in [('val', va, rows_va), ('test', te, rows_te)]:
        best = {}
        for qi, gi, yy, x in rows:
            s = clf.decision_function(sc.transform(x[None, :]))[0]
            if qi not in best or s > best[qi][0]:
                best[qi] = (s, gi)
        choice = np.zeros(len(d['bank']), dtype=int)
        feats = fva if name == 'val' else fte
        for qi, (s, gi) in best.items():
            gids = feats['records'][qi]['gids']
            choice[qi] = int(np.where(gids == gi)[0][0])
        out[name] = eval_choice(choice, d)
    return out


for C in (0.003, 0.01, 0.03, 0.1, 0.3):
    o = fit(rows_tr, rows_va, rows_te, C)
    print(fmt(o['val'], f'C={C} val') + f' testlike={test_like_metric(o["val"]):5.2f} || ' + fmt(o['test'], 'test'))
