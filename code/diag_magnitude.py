"""Answer-magnitude vs question-number diagnostics."""
import sys, io, re
from fractions import Fraction
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, auc

NUM = re.compile(r'\d+(?:\.\d+)?')


def qstats(q):
    v = [float(x) for x in NUM.findall(q)]
    if not v: return dict(qmin=0, qmax=0, qsum=0, qprod=1, qn=0)
    return dict(qmin=min(v), qmax=max(v), qsum=sum(v), qprod=float(np.prod([x for x in v if x > 0]) or 1), qn=len(v))


for split in ('validation', 'test'):
    d = load_split(split); nc = ncorrect(d)
    rows = []
    for i in range(len(d['bank'])):
        qs = qstats(d['bank'][i]['question'])
        for c in range(4):
            a = d['answers'][i, c]
            f = {}
            if a is None or Fraction(a).denominator != 1:
                f = {k: 0.0 for k in ('r_max', 'r_min', 'r_sum', 'd_max', 'log_ratio', 'in_range',
                                       'mult2', 'mult10', 'mult60', 'mult100', 'eq_qmax', 'eq_sum')}
            else:
                v = float(Fraction(a)); mx = max(1e-9, qs['qmax']); mn = max(1e-9, qs['qmin'])
                f['r_max'] = v / mx; f['r_min'] = v / mn; f['r_sum'] = v / max(1e-9, qs['qsum'])
                f['d_max'] = (v - mx) / (mx + 1e-9)
                f['log_ratio'] = np.log1p(abs(v)) - np.log1p(mx)
                f['in_range'] = float(0 <= v <= mx * max(1, qs['qn']))
                f['mult2'] = float(any(abs(v - 2 * q) < 1e-6 for q in [qs['qmax'], qs['qmin']]))
                f['mult10'] = float(any(abs(v - 10 * q) < 1e-6 for q in [qs['qmax'], qs['qmin']]))
                f['mult60'] = float(any(abs(v - 60 * q) < 1e-6 for q in [qs['qmax'], qs['qmin']]))
                f['mult100'] = float(any(abs(v - 100 * q) < 1e-6 for q in [qs['qmax'], qs['qmin']]))
                f['eq_qmax'] = float(abs(v - qs['qmax']) < 1e-6)
                f['eq_sum'] = float(abs(v - qs['qsum']) < 1e-6)
            rows.append((i, c, d['labels'][i, c], f))
    print(f'===== {split} =====')
    for k in (1, 2, 3):
        sub = [r for r in rows if nc[r[0]] == k]
        y = np.array([r[2] for r in sub])
        print(f'-- #{k} pos={y.mean():.2f}')
        for name in sub[0][3]:
            s = np.array([r[3][name] for r in sub])
            print(f'   {name:10s} AUC={auc(s,y):.3f}', end='')
        print()
