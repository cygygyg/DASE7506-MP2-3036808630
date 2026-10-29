"""Compute report statistics for the frozen submission and baseline."""
import json
from collections import defaultdict
import numpy as np
from mp2lib import load_split, ncorrect

te = load_split('test'); va = load_split('validation')
nc_te = ncorrect(te); nc_va = ncorrect(va)


def stats(pred_path, d, nc, name):
    pred = json.load(open(pred_path))['selected']
    ok = d['labels'][np.arange(len(d['bank'])), np.array(pred)]
    print(f'=== {name}: {100*ok.mean():.3f}% ({ok.sum()}/{len(ok)})')
    for k in (1, 2, 3):
        m = nc == k
        print(f'  #correct={k}: n={m.sum()} acc={100*ok[m].mean():.2f}% ({ok[m].sum()}/{m.sum()})')
    # pattern
    pat = defaultdict(lambda: [0, 0])
    for i in range(len(d['bank'])):
        groups = {}
        for c, a in enumerate(d['answers'][i]):
            if a is not None:
                try:
                    from fractions import Fraction
                    if Fraction(a).denominator == 1:
                        groups.setdefault(a, []).append(c)
                except Exception:
                    pass
        p = tuple(sorted((len(v) for v in groups.values()), reverse=True))
        if len(p) >= 2:
            pat[p][0] += int(ok[i]); pat[p][1] += 1
    print('  patterns:')
    for p, v in sorted(pat.items(), key=lambda kv: -kv[1][1]):
        print(f'    {p}: {v[0]}/{v[1]} = {100*v[0]/v[1]:.1f}%')
    return ok


ok_final = stats('runs/final-test-predictions.json', te, nc_te, 'final test')
ok_base = stats('runs/test-predictions.json', te, nc_te, 'baseline test')
print('final val')
stats('runs/final-validation-predictions.json', va, nc_va, 'final val')
