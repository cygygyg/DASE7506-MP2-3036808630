"""Soft routing: P(difficulty class) weights experts N, A, S, support."""
import sys, io, pickle, itertools
from fractions import Fraction
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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


def support(d):
    N = len(d['bank']); out = np.full((N, 4), -1e9)
    for i in range(N):
        groups = {}
        for c, a in enumerate(d['answers'][i]):
            if is_int(a): groups.setdefault(a, []).append(c)
        for v in groups.values():
            for c in v: out[i, c] = len(v)
    return out


def zs(p, d):
    p = p.copy()
    for i in range(len(p)):
        valid = np.array([is_int(d['answers'][i, c]) for c in range(4)])
        if valid.sum() == 0: p[i] = 0; continue
        v = p[i][valid]; m, s = v.mean(), v.std() + 1e-9
        p[i] = (p[i] - m) / s; p[i][~valid] = -1e9
    return p


Gtr, Gva, Gte = support(tr), support(va), support(te)
Ztr = [zs(C['oof'][k], tr) for k in 'NAS'] + [zs(Gtr, tr)]
Zva = [zs(Eva[i], va) for i in range(3)] + [zs(Gva, va)]
Zte = [zs(Ete[i], te) for i in range(3)] + [zs(Gte, te)]


def features(d, Z, gs):
    X = []
    for i in range(len(d['bank'])):
        sizes = np.sort(np.array(gs[i], float))[::-1]
        pad = np.zeros(5); pad[:min(5, len(sizes))] = sizes[:5]
        f = [qfeat(Z[0][i]), qfeat(Z[1][i]), qfeat(Z[2][i])]
        a = [Z[k][i].argmax() for k in range(3)]
        agree = [float(a[0] == a[1]), float(a[0] == a[2]), float(a[1] == a[2])]
        pat = [len(sizes), float(np.sum(sizes == 1)), float(np.sum(sizes >= 2)), float(np.sum(sizes >= 3)),
               float(len(set(sizes.tolist())) == 1), float(len(sizes) == 2 and sizes[0] >= 3),
               float(len(sizes) == 2 and sizes[0] == 2 and sizes[1] == 2)]
        cross = [Z[0][i, a[1]] - f[0][0], Z[1][i, a[0]] - f[1][0], Z[0][i, a[2]] - f[0][0],
                 Z[2][i, a[0]] - f[2][0], Z[1][i, a[2]] - f[1][0], Z[2][i, a[1]] - f[2][0]]
        X.append(np.concatenate([pad, *f, agree, pat, cross]))
    return np.array(X, float)


Xtr = features(tr, Ztr, gtr); Xva = features(va, Zva, gva); Xte = features(te, Zte, gte)
ytr = np.minimum(nc_tr, 3); yva = np.minimum(nc_va, 3); yte = np.minimum(nc_te, 3)
sc = StandardScaler().fit(Xtr)
for Creg in (0.1, 0.3, 1.0):
    clf = LogisticRegression(C=Creg, max_iter=4000, random_state=0).fit(sc.transform(Xtr), ytr)
    Pva = clf.predict_proba(sc.transform(Xva)); Pte = clf.predict_proba(sc.transform(Xte))
    print(f'C={Creg} classes={clf.classes_} val acc={100*(clf.classes_[Pva.argmax(1)]==yva).mean():.2f} '
          f'test acc={100*(clf.classes_[Pte.argmax(1)]==yte).mean():.2f}')
    # search mapping class->expert on validation
    best = (0, None, None)
    for temp in (0.5, 1.0, 2.0):
        Pv = np.power(Pva, 1 / temp); Pv /= Pv.sum(1, keepdims=True)
        Pt = np.power(Pte, 1 / temp); Pt /= Pt.sum(1, keepdims=True)
        for mapping in itertools.product(range(4), repeat=3):
            # expert per class 1,2,3
            ch = []
            for i in range(len(va['bank'])):
                w = np.zeros(4)
                for k in range(3):
                    w[mapping[k]] += Pv[i, k]
                s = sum(w[e] * Zva[e][i] for e in range(4))
                ch.append(int(np.argmax(s)))
            rv = eval_choice(ch, va)
            if rv['acc'] > best[0]:
                best = (rv['acc'], (temp, mapping), rv)
    temp, mapping = best[1]
    print(f'  best val temp={temp} mapping(classes1,2,3)->{mapping} acc={best[0]:.2f}')
    Pv = np.power(Pva, 1 / temp); Pv /= Pv.sum(1, keepdims=True)
    Pt = np.power(Pte, 1 / temp); Pt /= Pt.sum(1, keepdims=True)
    for name, d, P, Z in (('val', va, Pv, Zva), ('test', te, Pt, Zte)):
        ch = []
        for i in range(len(d['bank'])):
            w = np.zeros(4)
            for k in range(3): w[mapping[k]] += P[i, k]
            s = sum(w[e] * Z[e][i] for e in range(4))
            ch.append(int(np.argmax(s)))
        r = eval_choice(ch, d)
        print('   ' + fmt(r, name) + (f' tl={test_like_metric(r):.2f}' if name == 'val' else ''))
