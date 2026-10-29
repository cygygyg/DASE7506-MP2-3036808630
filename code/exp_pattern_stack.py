"""Pattern-specific stacking of N, A, S experts plus a group-support expert."""
import sys, io, pickle
from fractions import Fraction
from collections import defaultdict
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric

with open('cache/oof_experts.pkl', 'rb') as f:
    C = pickle.load(f)
tr = load_split('train'); va = load_split('validation'); te = load_split('test')
Eva, Ete = C['Eva'], C['Ete']
nc_va, nc_te = ncorrect(va), ncorrect(te)


def is_int(a): return a is not None and Fraction(a).denominator == 1


def support_score(d):
    """Score = size of the candidate's valid-answer group; invalid = -inf."""
    N = len(d['bank']); out = np.full((N, 4), -1e9)
    pats = []
    for i in range(N):
        groups = {}
        for c, a in enumerate(d['answers'][i]):
            if is_int(a): groups.setdefault(a, []).append(c)
        sizes = tuple(sorted((len(v) for v in groups.values()), reverse=True))
        pats.append(sizes)
        for v in groups.values():
            for c in v: out[i, c] = len(v)
    return out, pats


def zscore(p, d):
    p = p.copy()
    for i in range(len(p)):
        valid = np.array([is_int(d['answers'][i, c]) for c in range(4)])
        if valid.sum() == 0:
            p[i] = 0; continue
        v = p[i][valid]; m, s = v.mean(), v.std() + 1e-9
        p[i] = (p[i] - m) / s
        p[i][~valid] = -1e9
    return p


Gtr, ptr = support_score(tr); Gva, pva = support_score(va); Gte, pte = support_score(te)
Ztr = [zscore(C['oof'][k], tr) for k in 'NAS'] + [zscore(Gtr, tr)]
Zva = [zscore(Eva[i], va) for i in range(3)] + [zscore(Gva, va)]
Zte = [zscore(Ete[i], te) for i in range(3)] + [zscore(Gte, te)]
ytr = tr['labels']; yva = va['labels']; yte = te['labels']


def acc(Z, y, w, idx):
    p = sum(w[k] * Z[k][idx] for k in range(4))
    return float(y[idx, p.argmax(1)].mean())


def best_w(idx, seed=0):
    rng = np.random.default_rng(seed)
    best = (acc(Ztr, ytr, np.ones(4) / 4, idx), np.ones(4) / 4)
    # random search + local refine
    for it in range(400):
        w = rng.dirichlet(np.ones(4) * 0.8)
        a = acc(Ztr, ytr, w, idx)
        if a > best[0]: best = (a, w)
    w = best[1]
    for step in (0.15, 0.07, 0.03):
        improved = True
        while improved:
            improved = False
            for k in range(4):
                for s in (-step, step):
                    w2 = w.copy(); w2[k] = max(0, w2[k] + s); w2 = w2 / (w2.sum() + 1e-12)
                    a = acc(Ztr, ytr, w2, idx)
                    if a > best[0] + 1e-12:
                        best = (a, w2); w = w2; improved = True
    return best


# group patterns: use exact tuple when enough data, else collapse
train_by_pat = defaultdict(list)
for i, p in enumerate(ptr):
    if len(p) >= 2:
        train_by_pat[p].append(i)
val_by_pat = defaultdict(list)
for i, p in enumerate(pva):
    if len(p) >= 2: val_by_pat[p].append(i)
te_by_pat = defaultdict(list)
for i, p in enumerate(pte):
    if len(p) >= 2: te_by_pat[p].append(i)

MIN_N = 40
weights = {}
for p, idx in sorted(train_by_pat.items(), key=lambda kv: -len(kv[1])):
    idx = np.array(idx)
    if len(idx) >= MIN_N:
        a, w = best_w(idx, seed=len(p) + sum(p))
        weights[p] = w
        print(f'pattern {p} n={len(idx)} train={100*a:.1f}% w={np.round(w,2)}')

# global fallback
allidx = np.array([i for i, p in enumerate(ptr) if len(p) >= 2])
ga, gw = best_w(allidx, seed=7)
print('global', np.round(gw, 3), ga)


def apply(d, Z, pats, y, name):
    N = len(d['bank']); choice = np.zeros(N, dtype=int)
    for i in range(N):
        p = pats[i]
        w = weights.get(p, gw)
        if len(p) < 2:
            valid = [c for c in range(4) if is_int(d['answers'][i, c])]
            choice[i] = valid[0] if valid else 0
            continue
        s = sum(w[k] * Z[k][i] for k in range(4))
        choice[i] = int(np.argmax(s))
    r = eval_choice(choice, d)
    print(fmt(r, name) + (f' tl={test_like_metric(r):5.2f}' if len(d['bank']) == 798 else ''))
    return r, choice


rv, cv = apply(va, Zva, pva, yva, 'val pattern-stack')
rt, ct = apply(te, Zte, pte, yte, 'test pattern-stack')

for name, d, pats, ch in (('val', va, pva, cv), ('test', te, pte, ct)):
    tab = defaultdict(lambda: [0, 0])
    for i in range(len(d['bank'])):
        if len(pats[i]) < 2: continue
        tab[pats[i]][0] += int(d['labels'][i, ch[i]]); tab[pats[i]][1] += 1
    print(name)
    for p, v in sorted(tab.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f'  {p} {v[0]}/{v[1]}={100*v[0]/v[1]:.1f}%')
