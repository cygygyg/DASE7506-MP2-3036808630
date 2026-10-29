"""确定性规则：非整数答案必错（GSM8K 答案全为整数）。"""
import sys, io
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


def is_integer_answer(text):
    a = parse_answer(text)
    if a is None:
        return False
    try:
        return Fraction(a).denominator == 1
    except Exception:
        return False


emb_tr, bank_tr, gold_tr = load('train')
emb_va, bank_va, gold_va = load('validation')
emb_te, bank_te, gold_te = load('test')
y_tr = Y(bank_tr, gold_tr)
nc_va = n_correct(bank_va, gold_va)
nc_te = n_correct(bank_te, gold_te)
ce_tr = (emb_tr - emb_tr.mean(axis=1, keepdims=True)).reshape(-1, 1024)
sc = StandardScaler().fit(ce_tr)
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc.transform(ce_tr), y_tr)


def scores(emb, bank, exclude_nonint):
    ce = (emb - emb.mean(axis=1, keepdims=True)).reshape(-1, 1024)
    sco = clf.predict_proba(sc.transform(ce))[:, 1].reshape(len(bank), 4)
    for qi, row in enumerate(bank):
        for ci, c in enumerate(row['candidates']):
            if parse_answer(c['text']) is None:
                sco[qi, ci] = -np.inf
            elif exclude_nonint and not is_integer_answer(c['text']):
                sco[qi, ci] = -np.inf
    return sco


def report(sco, bank, gold, nc, name):
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
    print(f'{name:26s}: total={100*tot[0]/tot[1]:.2f}% | testlike={100*tl:.2f}% | #1={one[0]}/{one[1]}')


for excl in [False, True]:
    sva = scores(emb_va, bank_va, excl)
    report(sva, bank_va, gold_va, nc_va, f'val exclude_nonint={excl}')
print()
for excl in [False, True]:
    ste = scores(emb_te, bank_te, excl)
    report(ste, bank_te, gold_te, nc_te, f'test exclude_nonint={excl}')
