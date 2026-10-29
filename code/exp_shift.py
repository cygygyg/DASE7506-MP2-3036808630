"""Candidate-level experiments evaluated on validation AND test (distribution shift check)."""
import sys, io
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
ce = lambda d: (d['emb'] - d['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
Xtr, Xva, Xte = ce(tr), ce(va), ce(te)
ytr, yva, yte = tr['labels'].reshape(-1), va['labels'].reshape(-1), te['labels'].reshape(-1)
nc_tr, nc_va, nc_te = ncorrect(tr), ncorrect(va), ncorrect(te)
# distinct answer groups >=2
multi_tr = np.array([len(set(a for a in tr['answers'][i] if a)) >= 2 for i in range(len(nc_tr))])


def eval_scores(p, d, nc):
    p = p.reshape(-1, 4).copy()
    for i in range(len(p)):
        for c in range(4):
            if d['answers'][i, c] is None: p[i, c] = -1e9
    return eval_choice(p.argmax(1), d, nc)


def run(mask, weights=None, C=1.0, penalty='l2', name=''):
    mm = np.repeat(mask, 4)
    sc = StandardScaler().fit(Xtr[mm])
    Z = sc.transform(Xtr[mm])
    if weights is None:
        w = None
    else:
        w = np.repeat(weights, 4)[mm]
    clf = LogisticRegression(C=C, max_iter=5000, solver='lbfgs', random_state=0, penalty=penalty)
    clf.fit(Z, ytr[mm], sample_weight=w)
    rv = eval_scores(clf.predict_proba(sc.transform(Xva))[:, 1], va, nc_va)
    rt = eval_scores(clf.predict_proba(sc.transform(Xte))[:, 1], te, nc_te)
    print(fmt(rv, name + ' VAL') + f' tl={test_like_metric(rv):5.2f} || ' + fmt(rt, 'TEST'))


print('--- subset / weight experiments (C=1) ---')
run(np.ones(len(nc_tr), bool), name='all')
run(multi_tr, name='G>=2')
run(nc_tr <= 2, name='nc<=2')
run(nc_tr == 1, name='nc=1')
run(nc_tr >= 2, name='nc>=2')
# importance weights matching test composition among G>=2
w = np.ones(len(nc_tr))
tgt = {1: 151, 2: 145, 3: 171}
cnt = {k: int(((nc_tr == k) & multi_tr).sum()) for k in (1, 2, 3)}
for k in (1, 2, 3):
    w[nc_tr == k] = tgt[k] / max(1, cnt[k])
w /= w[multi_tr].mean()
run(multi_tr, weights=w, name='G>=2 + test-prior')
# equal weight per ncorrect class
w2 = np.ones(len(nc_tr))
for k in (1, 2, 3):
    w2[nc_tr == k] = 1.0 / max(1, int(((nc_tr == k) & multi_tr).sum()))
w2[~multi_tr] = 0
w2 /= w2[multi_tr].mean()
run(multi_tr, weights=w2, name='G>=2 + balanced')

print('--- C grid on G>=2 ---')
for C in (0.03, 0.1, 0.3, 1.0, 3.0):
    run(multi_tr, C=C, name=f'G>=2 C={C}')
