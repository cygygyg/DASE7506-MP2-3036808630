"""Diagnostics for improved arithmetic chains + candidate model fusion."""
import sys, io
import numpy as np
from fractions import Fraction
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, auc, test_like_metric
from arith2 import arith2_features, extract_chains

# sanity examples
for t in ['3 + 5 + 7 + 9 + 11 = 36', '\\frac{1}{4} \\times 8 + 7 = 2 + 7 = 9',
          '20 - 12 = 8', 'x = 282 - 184 = 98', '24t = 15t + 0.6']:
    print(t, '->', extract_chains(t))

for split in ('validation', 'test'):
    d = load_split(split); nc = ncorrect(d)
    rows = []
    for i in range(len(d['bank'])):
        for c in range(4):
            a = d['answers'][i, c]
            f = arith2_features(d['bank'][i]['candidates'][c]['text'],
                                int(Fraction(a)) if a is not None and Fraction(a).denominator == 1 else None)
            rows.append((i, c, d['labels'][i, c], f))
    print(f'===== {split} =====')
    for k in (1, 2, 3):
        sub = [r for r in rows if nc[r[0]] == k]
        y = np.array([r[2] for r in sub])
        print(f'-- #correct={k} pos={y.mean():.2f}')
        for name in sub[0][3]:
            s = np.array([r[3][name] for r in sub])
            print(f'   {name:20s} AUC={auc(s,y):.3f}', end='')
        print()
