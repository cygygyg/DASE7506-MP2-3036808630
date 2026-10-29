"""多视角集成 + 错误分析。"""
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


def make_view(emb, mode):
    if mode == 'raw':
        X = emb.reshape(-1, 1024)
    elif mode == 'center':
        X = (emb - emb.mean(axis=1, keepdims=True)).reshape(-1, 1024)
    elif mode == 'center_l2':
        X = (emb - emb.mean(axis=1, keepdims=True)).reshape(-1, 1024)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    elif mode == 'l2':
        X = emb.reshape(-1, 1024)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X


def train_view(mode):
    Xtr = make_view(emb_tr, mode)
    Xva = make_view(emb_va, mode)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc.transform(Xtr), y_tr)
    return clf, sc


def predict(clf, sc, mode, emb):
    X = make_view(emb, mode)
    return clf.predict_proba(sc.transform(X))[:, 1]


def report_from_sco(sco, bank, gold, nc, name):
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
    print(f'{name:24s}: total={100*tot[0]/tot[1]:.2f}% | testlike={100*tl:.2f}% | #1={one[0]}/{one[1]}')


modes = ['raw', 'center', 'center_l2', 'l2']
models = {m: train_view(m) for m in modes}
for m in modes:
    sco = predict(*models[m], m, emb_va)
    report_from_sco(sco, bank_va, gold_va, nc_va, m)

# 集成
print('--- 集成 (软投票, 平均概率) ---')
combos = [
    ('center+center_l2', ['center', 'center_l2']),
    ('center+raw', ['center', 'raw']),
    ('center+l2', ['center', 'l2']),
    ('center+center_l2+raw', ['center', 'center_l2', 'raw']),
    ('all4', ['raw', 'center', 'center_l2', 'l2']),
]
for name, ms in combos:
    sco = np.mean([predict(*models[m], m, emb_va) for m in ms], axis=0)
    report_from_sco(sco, bank_va, gold_va, nc_va, name)
