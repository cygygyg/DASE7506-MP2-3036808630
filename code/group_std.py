"""组内标准化 + 答案值相对特征（最后冲刺）。"""
import sys, io, re
import numpy as np
from collections import defaultdict
from fractions import Fraction
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


def transform(emb, mode):
    ce = emb - emb.mean(axis=1, keepdims=True)  # (N,4,1024)
    if mode == 'center':
        return ce.reshape(-1, 1024)
    if mode == 'group_std':
        # 除以每题4候选的逐维标准差
        std = ce.std(axis=1, keepdims=True) + 1e-6  # (N,1,1024)
        return (ce / std).reshape(-1, 1024)
    if mode == 'group_std_then_global':
        std = ce.std(axis=1, keepdims=True) + 1e-6
        g = ce / std
        return g.reshape(-1, 1024)
    if mode == 'per_cand_l2':
        g = ce.reshape(-1, 1024)
        return g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)


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
    print(f'{name:30s}: total={100*tot[0]/tot[1]:.2f}% | testlike={100*tl:.2f}% | #1={one[0]}/{one[1]}')


for mode in ['center', 'group_std', 'group_std_then_global', 'per_cand_l2']:
    Xtr = transform(emb_tr, mode); Xva = transform(emb_va, mode)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc.transform(Xtr), y_tr)
    sco = clf.predict_proba(sc.transform(Xva))[:, 1]
    report(sco, mode)
