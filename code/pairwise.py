"""pairwise ranking 实验。"""
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


def center(emb): return emb - emb.mean(axis=1, keepdims=True)


def build_pairs(emb, bank, gold):
    ce = center(emb)
    X, y = [], []
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        correct = [ci for ci, c in enumerate(row['candidates']) if parse_answer(c['text']) == ga]
        wrong = [ci for ci, c in enumerate(row['candidates']) if parse_answer(c['text']) != ga]
        for i in correct:
            for j in wrong:
                X.append(ce[qi, i] - ce[qi, j]); y.append(1)   # i 优于 j
                X.append(ce[qi, j] - ce[qi, i]); y.append(0)   # j 不优于 i
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8)


def n_correct(bank, gold):
    return [sum(1 for c in row['candidates'] if parse_answer(c['text']) == parse_answer(g['answer']))
            for g, row in zip(gold, bank)]


emb_tr, bank_tr, gold_tr = load('train')
emb_va, bank_va, gold_va = load('validation')
nc_va = n_correct(bank_va, gold_va)
Xp, yp = build_pairs(emb_tr, bank_tr, gold_tr)
print('pair 样本数:', Xp.shape)
sc = StandardScaler().fit(Xp)
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc.transform(Xp), yp)

# 推理：Borda count
ce_va = center(emb_va)
N = len(bank_va)
scores = np.zeros((N, 4))
for qi in range(N):
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            d = ce_va[qi, i] - ce_va[qi, j]
            scores[qi, i] += clf.predict_proba(sc.transform(d[None, :]))[0, 1]

for qi, row in enumerate(bank_va):
    for ci, c in enumerate(row['candidates']):
        if parse_answer(c['text']) is None:
            scores[qi, ci] = -np.inf

stats = defaultdict(lambda: [0,0]); tot=[0,0]
for qi in range(N):
    ga = parse_answer(gold_va[qi]['answer'])
    ch = int(np.argmax(scores[qi]))
    a = parse_answer(bank_va[qi]['candidates'][ch]['text'])
    ok = int(a is not None and a == ga)
    tot[0]+=ok; tot[1]+=1; stats[nc_va[qi]][0]+=ok; stats[nc_va[qi]][1]+=1
one, two, three = stats[1], stats[2], stats[3]
w = np.array([151,145,171])/467
tl = one[0]/max(1,one[1])*w[0]+two[0]/max(1,two[1])*w[1]+three[0]/max(1,three[1])*w[2]
print(f'pairwise Borda: total={100*tot[0]/tot[1]:.2f}% | testlike={100*tl:.2f}% | #1={one[0]}/{one[1]}')
