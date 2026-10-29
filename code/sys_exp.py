"""系统实验：训练子集 + 特征 + 分类器。"""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl, load_inputs


def load(split):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    _, arrays = load_inputs(split)
    return arrays['embeddings'], bank, gold


def n_correct(bank, gold):
    return [sum(1 for c in row['candidates'] if parse_answer(c['text']) == parse_answer(g['answer']))
            for g, row in zip(gold, bank)]


emb_tr, bank_tr, gold_tr = load('train')
emb_va, bank_va, gold_va = load('validation')
nc_tr = np.array(n_correct(bank_tr, gold_tr))
nc_va = np.array(n_correct(bank_va, gold_va))

def center(emb): return (emb - emb.mean(axis=1, keepdims=True))

ce_tr = center(emb_tr).reshape(-1, 1024)
ce_va = center(emb_va).reshape(-1, 1024)

def Y(bank, gold):
    y = np.zeros(len(bank)*4, dtype=np.int8)
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        for ci, c in enumerate(row['candidates']):
            a = parse_answer(c['text'])
            y[qi*4+ci] = int(a is not None and a == ga)
    return y
y_tr = Y(bank_tr, gold_tr)
y_va = Y(bank_va, gold_va)

# 额外无害特征：残差norm + 文本长度
def extra_feats(emb, bank):
    N = emb.shape[0]
    ce = center(emb)
    rn = np.linalg.norm(ce, axis=2).reshape(-1, 1)  # 残差norm
    tl = np.array([[len(c['text'])] for row in bank for c in row['candidates']], dtype=np.float32)
    return np.hstack([rn, tl])

ex_tr = extra_feats(emb_tr, bank_tr)
ex_va = extra_feats(emb_va, bank_va)

def evaluate(clf, Xtr, ytr, Xva, mask=None):
    if mask is not None:
        Xtr = Xtr[mask]; ytr = ytr[mask]
    sc = StandardScaler().fit(Xtr)
    clf.fit(sc.transform(Xtr), ytr)
    if hasattr(clf, 'predict_proba'):
        sco = clf.predict_proba(sc.transform(Xva))[:, 1]
    else:
        sco = clf.decision_function(sc.transform(Xva))
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
    one = stats[1]
    return tot[0]/tot[1]*100, one[0]/max(1,one[1])*100

# 构建训练 mask
def qmask(cond):
    m = np.zeros(len(nc_tr)*4, dtype=bool)
    for qi in range(len(nc_tr)):
        if cond(nc_tr[qi]):
            m[qi*4:qi*4+4] = True
    return m

print('=== C 调参 (center, 全训练集) ===')
for C in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
    t, o = evaluate(LogisticRegression(C=C, max_iter=5000, solver='lbfgs'), ce_tr, y_tr, ce_va)
    print(f'  C={C:5.2f}: total={t:.2f}%  #1correct={o:.1f}%')

print('=== 训练子集选择 (center, C=0.3) ===')
for name, cond in [('nc<=2', lambda n: n<=2), ('nc<=3', lambda n: n<=3),
                   ('nc==1', lambda n: n==1), ('nc in {1,2}', lambda n: n in (1,2)),
                   ('all', lambda n: True)]:
    t, o = evaluate(LogisticRegression(C=0.3, max_iter=5000, solver='lbfgs'), ce_tr, y_tr, ce_va, qmask(cond))
    print(f'  {name:12s}: total={t:.2f}%  #1correct={o:.1f}%')

print('=== 特征组合 (C=0.3) ===')
for name, Xtr, Xva in [
    ('center', ce_tr, ce_va),
    ('center+resnorm+len', np.hstack([ce_tr, ex_tr]), np.hstack([ce_va, ex_va])),
]:
    for clfname, clf in [('LogReg', LogisticRegression(C=0.3, max_iter=5000, solver='lbfgs')),
                         ('LinearSVC', LinearSVC(C=0.3, max_iter=5000))]:
        t, o = evaluate(clf, Xtr, y_tr, Xva)
        print(f'  {name} {clfname}: total={t:.2f}%  #1correct={o:.1f}%')
