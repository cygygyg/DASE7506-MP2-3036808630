"""最后微调：ZCA白化 / L1正则 / C精细扫描。"""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl, load_inputs


def load(split):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    _, arrays = load_inputs(split)
    return arrays['embeddings'], bank, gold


def Y(bank, gold):
    y = np.zeros(len(bank)*4, dtype=np.int8)
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        for ci, c in enumerate(row['candidates']):
            a = parse_answer(c['text'])
            y[qi*4+ci] = int(a is not None and a == ga)
    return y


def n_correct(bank, gold):
    return [sum(1 for c in row['candidates'] if parse_answer(c['text']) == parse_answer(g['answer']))
            for g, row in zip(gold, bank)]


emb_tr, bank_tr, gold_tr = load('train')
emb_va, bank_va, gold_va = load('validation')
y_tr = Y(bank_tr, gold_tr)
nc_va = n_correct(bank_va, gold_va)
ce_tr = (emb_tr - emb_tr.mean(axis=1, keepdims=True)).reshape(-1, 1024)
ce_va = (emb_va - emb_va.mean(axis=1, keepdims=True)).reshape(-1, 1024)


def report(sco, name):
    sco = sco.reshape(len(bank_va), 4)
    for qi, row in enumerate(bank_va):
        for ci, c in enumerate(row['candidates']):
            if parse_answer(c['text']) is None:
                sco[qi, ci] = -np.inf
    stats = defaultdict(lambda: [0,0]); tot=[0,0]
    for qi in range(len(bank_va)):
        ga = parse_answer(gold_va[qi]['answer'])
        ch = int(np.argmax(sco[qi]))
        a = parse_answer(bank_va[qi]['candidates'][ch]['text'])
        ok = int(a is not None and a == ga)
        tot[0]+=ok; tot[1]+=1; stats[nc_va[qi]][0]+=ok; stats[nc_va[qi]][1]+=1
    one, two, three = stats[1], stats[2], stats[3]
    w = np.array([151,145,171])/467
    tl = one[0]/max(1,one[1])*w[0]+two[0]/max(1,two[1])*w[1]+three[0]/max(1,three[1])*w[2]
    print(f'{name:26s}: total={100*tot[0]/tot[1]:.2f}% | testlike={100*tl:.2f}% | #1={one[0]}/{one[1]}')


# ZCA whitening
U, s, Vt = np.linalg.svd(ce_tr - ce_tr.mean(axis=0), full_matrices=False)
print('ZCA whitening...')
W = Vt.T @ np.diag(1.0/(s + 1e-5)) @ Vt  # 白化矩阵 (近似)
def whiten(X, mean, W):
    return (X - mean) @ W
mean_tr = ce_tr.mean(axis=0)
Xtr_w = whiten(ce_tr, mean_tr, W)
Xva_w = whiten(ce_va, mean_tr, W)

for name, Xtr, Xva in [('center+StandardScaler', ce_tr, ce_va), ('center+ZCA', Xtr_w, Xva_w)]:
    if name.endswith('StandardScaler'):
        sc = StandardScaler().fit(Xtr)
        Xtr2, Xva2 = sc.transform(Xtr), sc.transform(Xva)
    else:
        Xtr2, Xva2 = Xtr, Xva
    clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(Xtr2, y_tr)
    report(clf.predict_proba(Xva2)[:, 1], name)

# L1 正则扫描
print('--- L1 正则 (liblinear) ---')
sc = StandardScaler().fit(ce_tr)
Xtr_s = sc.transform(ce_tr); Xva_s = sc.transform(ce_va)
for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
    clf = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=5000).fit(Xtr_s, y_tr)
    nz = int((clf.coef_ != 0).sum())
    report(clf.predict_proba(Xva_s)[:, 1], f'L1 C={C} (nonzero={nz})')

# L2 C 精细扫描 (lbfgs)
print('--- L2 C 精细 (lbfgs) ---')
for C in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    clf = LogisticRegression(C=C, max_iter=5000, solver='lbfgs').fit(Xtr_s, y_tr)
    report(clf.predict_proba(Xva_s)[:, 1], f'L2 C={C}')
