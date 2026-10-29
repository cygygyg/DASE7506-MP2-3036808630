"""Clean hybrid: support group vs best-evidence group, threshold by size pattern.

Thresholds are selected on validation only, then frozen for test.
"""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
Xtr = (tr['emb'] - tr['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', random_state=0).fit(sc.transform(Xtr), tr['labels'].reshape(-1))


def scores(d):
    X = (d['emb'] - d['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
    p = clf.predict_proba(sc.transform(X))[:, 1].reshape(-1, 4)
    for i in range(len(p)):
        for c in range(4):
            if d['answers'][i, c] is None: p[i, c] = -1e9
    return p


def prep(d):
    p = scores(d); out = []
    for i in range(len(d['bank'])):
        ans = d['answers'][i]; groups = {}
        for c, a in enumerate(ans):
            key = ('a', a) if a is not None else ('none', c)
            groups.setdefault(key, []).append(c)
        keys = list(groups)
        sizes = np.array([len(groups[k]) for k in keys])
        gs = np.array([max(p[i, c] for c in groups[k]) for k in keys])
        labs = np.array([int(any(d['labels'][i, c] for c in groups[k])) for k in keys])
        out.append(dict(sizes=sizes, gs=gs, labs=labs, keys=keys, groups=groups))
    return out


def rule(rec, t3, t2):
    sizes, gs = rec['sizes'], rec['gs']
    G = len(sizes)
    if G == 1: return 0
    imax = int(np.argmax(sizes))
    others = [g for g in range(G) if g != imax]
    if others and sizes[imax] > max(sizes[g] for g in others):
        t = t3 if sizes[imax] >= 3 else t2
        gap = gs[imax] - max(gs[g] for g in others)
        if gap > t: return imax
        return max(others, key=lambda g: gs[g])
    return int(np.argmax(gs))


def evaluate(recs, d, t3, t2):
    N = len(d['bank']); choice = np.zeros(N, dtype=int)
    pattern_stats = defaultdict(lambda: [0, 0])
    for i, rec in enumerate(recs):
        pick = rule(rec, t3, t2)
        choice[i] = rec['groups'][rec['keys'][pick]][0]
        pat = tuple(sorted(rec['sizes'].tolist(), reverse=True))
        pattern_stats[pat][0] += int(rec['labs'][pick]); pattern_stats[pat][1] += 1
    res = eval_choice(choice, d)
    return res, pattern_stats, choice


prec_va = prep(va); prec_te = prep(te)
print('base val/test:')
for name, d, recs in (('val', va, prec_va), ('test', te, prec_te)):
    ch = np.array([rec['groups'][rec['keys'][int(np.argmax(rec['gs']))]][0] for rec in recs])
    print(fmt(eval_choice(ch, d), name))

best = (-1, None, None)
for t3 in np.linspace(-0.30, 0.05, 71):
    for t2 in np.linspace(-0.25, 0.10, 71):
        r, _, _ = evaluate(prec_va, va, t3, t2)
        tl = test_like_metric(r)
        if tl > best[0]:
            best = (tl, t3, t2)
print('best val thresholds:', best)
_, t3, t2 = best
for name, d, recs in (('val', va, prec_va), ('test', te, prec_te)):
    r, ps, _ = evaluate(recs, d, t3, t2)
    print(fmt(r, f'{name} hybrid') + (f' testlike={test_like_metric(r):5.2f}' if name == 'val' else ''))
    for pat, v in sorted(ps.items(), key=lambda kv: -kv[1][1]):
        print(f'    {pat} {v[0]}/{v[1]} = {100*v[0]/v[1]:.1f}%')

# Also report thresholds optimised on validation overall accuracy rather than testlike
best2 = (-1, None, None)
for t3 in np.linspace(-0.30, 0.05, 71):
    for t2 in np.linspace(-0.25, 0.10, 71):
        r, _, _ = evaluate(prec_va, va, t3, t2)
        if r['acc'] > best2[0]:
            best2 = (r['acc'], t3, t2)
print('best val total thresholds:', best2)
_, t3b, t2b = best2
r, ps, _ = evaluate(prec_te, te, t3b, t2b)
print(fmt(r, 'test hybrid(total-tuned)'))
