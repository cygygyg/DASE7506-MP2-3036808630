"""新方法：答案级聚合验证器 + 正确原型距离。"""
import sys, io
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
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
nc_va = n_correct(bank_va, gold_va)


def center(emb): return emb - emb.mean(axis=1, keepdims=True)
ce_tr = center(emb_tr)
ce_va = center(emb_va)


def report(sco, bank, gold, nc, name):
    sco = sco.reshape(len(bank), 4)
    for qi, row in enumerate(bank):
        for ci, c in enumerate(row['candidates']):
            if parse_answer(c['text']) is None:
                sco[qi, ci] = -np.inf
    stats = defaultdict(lambda: [0,0]); tot=[0,0]
    for qi in range(len(bank)):
        ga = parse_answer(gold[qi]['answer'])
        ch = int(np.argmax(sco[qi]))
        a = parse_answer(bank[qi]['candidates'][ch]['text'])
        ok = int(a is not None and a == ga)
        tot[0]+=ok; tot[1]+=1; stats[nc[qi]][0]+=ok; stats[nc[qi]][1]+=1
    one, two, three = stats[1], stats[2], stats[3]
    w = np.array([151,145,171])/467
    tl = one[0]/max(1,one[1])*w[0]+two[0]/max(1,two[1])*w[1]+three[0]/max(1,three[1])*w[2]
    print(f'{name:30s}: total={100*tot[0]/tot[1]:.2f}% | testlike={100*tl:.2f}% | #1={one[0]}/{one[1]}')


# ---- 1. 答案级聚合 ----
def answer_level(emb, bank, gold):
    ce = center(emb)
    X, y = [], []
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        ans = [parse_answer(c['text']) for c in row['candidates']]
        groups = {}
        for ci, a in enumerate(ans):
            if a is None:
                continue
            groups.setdefault(a, []).append(ci)
        for a, idxs in groups.items():
            X.append(ce[qi, idxs].mean(axis=0))
            y.append(int(a == ga))
    return np.array(X), np.array(y, dtype=np.int8)


Xa_tr, ya_tr = answer_level(emb_tr, bank_tr, gold_tr)
print('答案级样本数:', Xa_tr.shape, '正样本比例:', ya_tr.mean())
sc_a = StandardScaler().fit(Xa_tr)
clf_a = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc_a.transform(Xa_tr), ya_tr)

# 答案级推理
ce_va_flat = ce_va.reshape(-1, 1024)
sco_a = np.full((len(bank_va), 4), -np.inf)
for qi, row in enumerate(bank_va):
    ans = [parse_answer(c['text']) for c in row['candidates']]
    groups = {}
    for ci, a in enumerate(ans):
        if a is None:
            continue
        groups.setdefault(a, []).append(ci)
    for a, idxs in groups.items():
        agg = ce_va[qi, idxs].mean(axis=0)
        s = clf_a.predict_proba(sc_a.transform(agg[None, :]))[0, 1]
        for ci in idxs:
            sco_a[qi, ci] = s
report(sco_a, bank_va, gold_va, nc_va, 'answer-level agg')

# ---- 2. 正确原型距离 ----
ytr = np.zeros(len(bank_tr)*4, dtype=np.int8)
for qi, (g, row) in enumerate(zip(gold_tr, bank_tr)):
    ga = parse_answer(g['answer'])
    for ci, c in enumerate(row['candidates']):
        a = parse_answer(c['text'])
        ytr[qi*4+ci] = int(a is not None and a == ga)
ce_tr_flat = ce_tr.reshape(-1, 1024)
proto_corr = ce_tr_flat[ytr == 1].mean(axis=0)
proto_wrong = ce_tr_flat[ytr == 0].mean(axis=0)
print('正确原型 norm:', np.linalg.norm(proto_corr), '错误原型 norm:', np.linalg.norm(proto_wrong))

def proto_scores(ce_flat):
    return ce_flat @ proto_corr / (np.linalg.norm(ce_flat, axis=1) * np.linalg.norm(proto_corr) + 1e-9) - \
           ce_flat @ proto_wrong / (np.linalg.norm(ce_flat, axis=1) * np.linalg.norm(proto_wrong) + 1e-9)
sco_p = proto_scores(ce_va_flat).reshape(len(bank_va), 4)
report(sco_p, bank_va, gold_va, nc_va, 'prototype-cos-diff')

# ---- 3. 原型距离 + LogReg 融合（作为额外特征）----
def proto_feat(ce_flat):
    p = ce_flat @ proto_corr / (np.linalg.norm(ce_flat, axis=1) * np.linalg.norm(proto_corr) + 1e-9)
    w = ce_flat @ proto_wrong / (np.linalg.norm(ce_flat, axis=1) * np.linalg.norm(proto_wrong) + 1e-9)
    return np.stack([p, w, p - w], axis=1)

Xtr_c = np.hstack([ce_tr_flat, proto_feat(ce_tr_flat)])
Xva_c = np.hstack([ce_va_flat, proto_feat(ce_va_flat)])
sc_c = StandardScaler().fit(Xtr_c)
clf_c = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc_c.transform(Xtr_c), ytr)
sco_c = clf_c.predict_proba(sc_c.transform(Xva_c))[:, 1].reshape(len(bank_va), 4)
report(sco_c, bank_va, gold_va, nc_va, 'center+proto-feat LogReg')
