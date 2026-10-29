"""Does kNN help broadly or only on near-duplicate questions?"""
import sys, io, pickle
from fractions import Fraction
import numpy as np
from sklearn.preprocessing import normalize
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect
from mp2features import build

tr = load_split('train'); te = load_split('test')
qtr = tr['emb'].mean(1); qte = te['emb'].mean(1)
g = qtr.mean(0)
Qtr = normalize(qtr - g); Qte = normalize(qte - g)
S = Qte @ Qtr.T
mx = S.max(1); top80 = np.sort(S, axis=1)[:, -80:].mean(1)
print('top1 similarity percentiles', np.round(np.percentile(mx, [0, 10, 25, 50, 75, 90, 100]), 3))
print('mean top80 percentiles', np.round(np.percentile(top80, [0, 10, 25, 50, 75, 90, 100]), 3))

Scores = pickle.load(open('cache/final_scores.pkl', 'rb'))
A = Scores['ZAt']; K = Scores['Kt']
fin = 0.82 * A + 0.18 * K
chA, chK, chF = A.argmax(1), K.argmax(1), fin.argmax(1)
okA = te['labels'][np.arange(len(te['bank'])), chA].astype(bool)
okK = te['labels'][np.arange(len(te['bank'])), chK].astype(bool)
okF = te['labels'][np.arange(len(te['bank'])), chF].astype(bool)
print('A', okA.mean(), 'K', okK.mean(), 'F', okF.mean())
print('kNN fixes A errors:', int((~okA & okF).sum()), 'breaks A corrects:', int((okA & ~okF).sum()))
print('both correct', int((okA & okK).sum()), 'A only', int((okA & ~okK).sum()),
      'K only', int((~okA & okK).sum()), 'neither', int((~okA & ~okK).sum()))

for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
    a, b = np.percentile(mx, [100 * lo, 100 * hi])
    m = (mx >= a) & (mx <= b if hi == 1 else mx < b)
    print(f'top1 sim [{a:.3f},{b:.3f}] n={m.sum()} A={100*okA[m].mean():.1f} K={100*okK[m].mean():.1f} F={100*okF[m].mean():.1f}')

# near-duplicate threshold
for t in (0.6, 0.7, 0.8, 0.9, 0.95):
    m = mx >= t
    print(f'sim>={t}: n={m.sum()} A={100*okA[m].mean() if m.any() else 0:.1f} K={100*okK[m].mean() if m.any() else 0:.1f} F={100*okF[m].mean() if m.any() else 0:.1f}')
