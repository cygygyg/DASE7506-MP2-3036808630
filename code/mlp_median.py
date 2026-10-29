"""median center + MLP调参 + test参考。"""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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


def Y(bank, gold):
    y = np.zeros(len(bank)*4, dtype=np.int8)
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        for ci, c in enumerate(row['candidates']):
            a = parse_answer(c['text'])
            y[qi*4+ci] = int(a is not None and a == ga)
    return y


emb_tr, bank_tr, gold_tr = load('train')
emb_va, bank_va, gold_va = load('validation')
emb_te, bank_te, gold_te = load('test')
y_tr = Y(bank_tr, gold_tr)
y_va = Y(bank_va, gold_va)
nc_va = n_correct(bank_va, gold_va)
nc_te = n_correct(bank_te, gold_te)


def prep(emb, mode):
    if mode == 'mean':
        return (emb - emb.mean(axis=1, keepdims=True)).reshape(-1, 1024)
    if mode == 'median':
        return (emb - np.median(emb, axis=1, keepdims=True)).reshape(-1, 1024)


def score(sco, bank, gold, nc):
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
    return tot, stats


# 训练集用 mean center
Xtr = prep(emb_tr, 'mean')
sc = StandardScaler().fit(Xtr)
Xtr_s = sc.transform(Xtr)

print('=== center mode (LogReg C=1) ===')
for mode in ['mean', 'median']:
    Xva = prep(emb_va, mode)
    clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(Xtr_s, y_tr)
    sco = clf.predict_proba(sc.transform(Xva))[:, 1]
    tot, stats = score(sco, bank_va, gold_va, nc_va)
    print(f'  {mode:8s}: total={100*tot[0]/tot[1]:.2f}%  #1correct={stats[1][0]}/{stats[1][1]}')

print('=== MLP 调参 (mean center) ===')
Xva = prep(emb_va, 'mean')
for hid, alpha in [((256,), 1e-3), ((512,), 1e-3), ((512,256), 1e-4), ((1024,), 1e-4), ((256,), 1e-5)]:
    clf = MLPClassifier(hidden_layer_sizes=hid, alpha=alpha, max_iter=500, early_stopping=True,
                        validation_fraction=0.1, random_state=0, learning_rate_init=1e-3)
    clf.fit(Xtr_s, y_tr)
    sco = clf.predict_proba(sc.transform(Xva))[:, 1]
    tot, stats = score(sco, bank_va, gold_va, nc_va)
    print(f'  MLP{hid} alpha={alpha}: total={100*tot[0]/tot[1]:.2f}%  #1correct={stats[1][0]}/{stats[1][1]}')

print('=== test 参考 (mean center LogReg C=1) ===')
Xte = prep(emb_te, 'mean')
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(Xtr_s, y_tr)
sco = clf.predict_proba(sc.transform(Xte))[:, 1]
tot, stats = score(sco, bank_te, gold_te, nc_te)
print(f'  test total={100*tot[0]/tot[1]:.2f}%  #1correct={stats[1][0]}/{stats[1][1]}  #2={stats[2][0]}/{stats[2][1]}  #3={stats[3][0]}/{stats[3][1]}')
