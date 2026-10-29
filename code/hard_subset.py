"""困难题专用模型 vs 全数据模型，在 #1 题上的对比。"""
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
emb_te, bank_te, gold_te = load('test')
y_tr = Y(bank_tr, gold_tr)
nc_tr = np.array(n_correct(bank_tr, gold_tr))
nc_va = n_correct(bank_va, gold_va)
nc_te = n_correct(bank_te, gold_te)

ce_tr = (emb_tr - emb_tr.mean(axis=1, keepdims=True)).reshape(-1, 1024)
ce_va = (emb_va - emb_va.mean(axis=1, keepdims=True)).reshape(-1, 1024)
ce_te = (emb_te - emb_te.mean(axis=1, keepdims=True)).reshape(-1, 1024)


def report_te(sco, bank, gold, nc, name):
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
    print(f'{name:24s}: total={100*tot[0]/tot[1]:.2f}% | #1={one[0]}/{one[1]} | #2={two[0]}/{two[1]} | #3={three[0]}/{three[1]}')


# 训练子集选择
def qmask(cond):
    m = np.zeros(len(nc_tr)*4, dtype=bool)
    for qi in range(len(nc_tr)):
        if cond(nc_tr[qi]):
            m[qi*4:qi*4+4] = True
    return m

configs = [
    ('all', None),
    ('nc==1', qmask(lambda n: n == 1)),
    ('nc<=2', qmask(lambda n: n <= 2)),
    ('nc==1 only (few)', qmask(lambda n: n == 1)),
]
for name, mask in configs:
    Xtr = ce_tr if mask is None else ce_tr[mask]
    ytr = y_tr if mask is None else y_tr[mask]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc.transform(Xtr), ytr)
    sco = clf.predict_proba(sc.transform(ce_te))[:, 1]
    report_te(sco, bank_te, gold_te, nc_te, name)
