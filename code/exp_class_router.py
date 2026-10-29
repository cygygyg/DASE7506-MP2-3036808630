"""3-class difficulty router from OOF expert scores + group structure."""
import sys, io, pickle
from fractions import Fraction
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric

with open('cache/oof_experts.pkl', 'rb') as f:
    C = pickle.load(f)
tr = load_split('train'); va = load_split('validation'); te = load_split('test')
Eva, Ete = C['Eva'], C['Ete']
gtr, gva, gte = C['gtr'], C['gva'], C['gte']
nc_tr, nc_va, nc_te = ncorrect(tr), ncorrect(va), ncorrect(te)


def is_int(a): return a is not None and Fraction(a).denominator == 1


def qfeat(p):
    s = np.sort(p)[::-1]
    e = np.exp(s - s[0]); pr = e / e.sum()
    ent = -(pr * np.log(pr + 1e-12)).sum()
    return [s[0], s[1], s[0] - s[1], s.std(), ent, pr[0], pr[1]]


def features(d, E, gs):
    X = []
    for i in range(len(d['bank'])):
        sizes = np.sort(np.array(gs[i], float))[::-1]
        pad = np.zeros(5); pad[:min(5, len(sizes))] = sizes[:5]
        fN, fA, fS = qfeat(E[0][i]), qfeat(E[1][i]), qfeat(E[2][i])
        aN, aA, aS = E[0][i].argmax(), E[1][i].argmax(), E[2][i].argmax()
        agree = [float(aN == aA), float(aN == aS), float(aA == aS)]
        pat = [len(sizes), float(np.sum(sizes == 1)), float(np.sum(sizes >= 2)), float(np.sum(sizes >= 3)),
               float(len(set(sizes.tolist())) == 1), float(len(sizes) == 2 and sizes[0] >= 3),
               float(len(sizes) == 2 and sizes[0] == 2 and sizes[1] == 2)]
        # score of each expert's pick under the other experts
        cross = [E[0][i, aA] - fN[0], E[1][i, aN] - fA[0], E[0][i, aS] - fN[0],
                 E[2][i, aN] - fS[0], E[1][i, aS] - fA[0], E[2][i, aA] - fS[0]]
        X.append(np.concatenate([pad, fN, fA, fS, agree, pat, cross]))
    return np.array(X, float)


Xtr = features(tr, [C['oof']['N'], C['oof']['A'], C['oof']['S']], gtr)
Xva = features(va, Eva, gva)
Xte = features(te, Ete, gte)
print('features', Xtr.shape)

# estimate test class prior from observable patterns using train conditional
bypat = defaultdict(lambda: np.zeros(4))
for i, p in enumerate(gtr):
    p = tuple(sorted(np.asarray(p).tolist(), reverse=True))
    if len(p) >= 2:
        bypat[p][min(3, int(nc_tr[i]))] += 1
prior = np.zeros(4)
for p, cnt in bypat.items():
    n = cnt.sum()
    prob = cnt / n
    # count test questions with this pattern
    nt = sum(1 for q in gte if len(q) >= 2 and tuple(sorted(np.asarray(q).tolist(), reverse=True)) == p)
    prior += nt * prob
print('estimated test prior from patterns', prior[:4] / prior[:4].sum(), 'actual', [(nc_te == k).mean() for k in (1, 2, 3)])

for kind in ('logit', 'hgb'):
    if kind == 'logit':
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.3, max_iter=4000, random_state=0).fit(sc.transform(Xtr), nc_tr)
        pv = clf.predict_proba(sc.transform(Xva)); pt = clf.predict_proba(sc.transform(Xte))
    else:
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                             min_samples_leaf=30, l2_regularization=1.0, random_state=0).fit(Xtr, nc_tr)
        pv = clf.predict_proba(Xva); pt = clf.predict_proba(Xte)
    print(f'--- {kind} ---')
    print('classes', clf.classes_)
    pv_pred = clf.predict(sc.transform(Xva)) if kind == 'logit' else clf.predict(Xva)
    print('val 3-class acc', 100 * (pv_pred == nc_va).mean())
    for name, d, P, nc in (('val', va, pv, nc_va), ('test', te, pt, nc_te)):
        pred = clf.classes_[P.argmax(1)]
        # routing: #1 -> S, #2 -> A, #3 -> N (and optionally support)
        E = Eva if name == 'val' else Ete
        ch = np.zeros(len(d['bank']), dtype=int)
        for i in range(len(d['bank'])):
            k = int(pred[i])
            e = 2 if k == 1 else 1 if k == 2 else 0
            ch[i] = E[e][i].argmax()
        r = eval_choice(ch, d)
        print(fmt(r, f'{name} route') + (f' tl={test_like_metric(r):.2f}' if name == 'val' else ''))
        print('  confusion (true rows 1..3):')
        print(confusion_matrix(nc, pred, labels=[1, 2, 3]))
