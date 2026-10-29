"""Blend cached expert scores N and A; evaluate simple ensembles and per-pattern routing."""
import sys, io, pickle
from fractions import Fraction
import numpy as np
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric

with open('cache/oof_experts.pkl', 'rb') as f:
    C = pickle.load(f)
tr = load_split('train'); va = load_split('validation'); te = load_split('test')
Eva, Ete = C['Eva'], C['Ete']
Nva, Ava, Sva = Eva; Nte, Ate, Ste = Ete
nc_va, nc_te = ncorrect(va), ncorrect(te)


def is_int(a): return a is not None and Fraction(a).denominator == 1
def zmask(p, d):
    p = p.copy()
    for i in range(len(p)):
        valid = np.array([is_int(d['answers'][i, c]) for c in range(4)])
        if valid.sum() == 0:
            p[i] = 0; continue
        v = p[i][valid]
        m, s = v.mean(), v.std() + 1e-9
        p[i] = (p[i] - m) / s
        p[i][~valid] = -1e9
    return p


def ev(p, d, name):
    r = eval_choice(np.array([int(np.argmax(p[i])) for i in range(len(d['bank']))]), d)
    print(fmt(r, name) + (f' tl={test_like_metric(r):5.2f}' if len(d['bank']) == 798 else ''))
    return r


Zva = [zmask(p, va) for p in (Nva, Ava, Sva)]
Zte = [zmask(p, te) for p in (Nte, Ate, Ste)]

print('--- score blends N/A (z-scored) ---')
for a in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    pv = (1 - a) * Zva[0] + a * Zva[1]
    pt = (1 - a) * Zte[0] + a * Zte[1]
    print(f'alpha={a:.1f}', end=' ')
    rv = eval_choice([int(np.argmax(pv[i])) for i in range(len(va['bank']))], va)
    rt = eval_choice([int(np.argmax(pt[i])) for i in range(len(te['bank']))], te)
    print(f'val={rv["acc"]:.2f} tl={test_like_metric(rv):.2f} #1={rv["acc1"]:.1f} || test={rt["acc"]:.2f} #1={rt["acc1"]:.1f} #2={rt.get("acc2",0):.1f} #3={rt.get("acc3",0):.1f}')

print('--- 3-way blend ---')
for a in (0.2, 0.3, 0.4, 0.5):
    for s in (0.0, 0.1, 0.2):
        pv = (1 - a - s) * Zva[0] + a * Zva[1] + s * Zva[2]
        pt = (1 - a - s) * Zte[0] + a * Zte[1] + s * Zte[2]
        rv = eval_choice([int(np.argmax(pv[i])) for i in range(len(va['bank']))], va)
        rt = eval_choice([int(np.argmax(pt[i])) for i in range(len(te['bank']))], te)
        print(f'a={a:.1f} s={s:.1f} val={rv["acc"]:.2f} tl={test_like_metric(rv):.2f} || test={rt["acc"]:.2f} #1={rt["acc1"]:.1f} #2={rt.get("acc2",0):.1f} #3={rt.get("acc3",0):.1f}')

print('--- per-pattern choice among experts (oracle diagnostics) ---')
for name, d, E, nc in (('val', va, (Nva, Ava, Sva), nc_va), ('test', te, (Nte, Ate, Ste), nc_te)):
    pats = defaultdict(lambda: np.zeros(6))
    for i in range(len(d['bank'])):
        ans = d['answers'][i]; groups = {}
        for c, a in enumerate(ans):
            if not is_int(a): continue
            groups.setdefault(a, []).append(c)
        sizes = tuple(sorted((len(v) for v in groups.values()), reverse=True))
        if len(sizes) < 2: continue
        oks = [int(d['labels'][i, p[i].argmax()]) for p in E]
        pats[sizes] += [1, *oks, max(oks)]
    print(name)
    for p, v in sorted(pats.items(), key=lambda kv: -kv[1][0])[:10]:
        n = v[0]
        print(f'  {p} n={int(n)} N={100*v[1]/n:.1f} A={100*v[2]/n:.1f} S={100*v[3]/n:.1f} O={100*v[4]/n:.1f}')
