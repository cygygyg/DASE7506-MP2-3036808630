"""class_weight / RobustScaler / 不同 solver 最后测试。"""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
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
    print(f'{name:30s}: total={100*tot[0]/tot[1]:.2f}% | testlike={100*tl:.2f}% | #1={one[0]}/{one[1]}')


sc = StandardScaler().fit(ce_tr)
Xtr = sc.transform(ce_tr); Xva = sc.transform(ce_va)

for cw in [None, 'balanced']:
    for solver in ['lbfgs', 'liblinear', 'newton-cg', 'saga']:
        try:
            clf = LogisticRegression(C=1.0, max_iter=5000, solver=solver, class_weight=cw).fit(Xtr, y_tr)
            report(clf.predict_proba(Xva)[:, 1], f'LogReg solver={solver} cw={cw}')
        except Exception as e:
            print(f'LogReg solver={solver} cw={cw} FAILED: {e}')

# RobustScaler
rs = RobustScaler().fit(ce_tr)
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(rs.transform(ce_tr), y_tr)
report(clf.predict_proba(rs.transform(ce_va))[:, 1], 'RobustScaler LogReg')

# Normalizer (L2 per-sample)
nm = Normalizer().fit(ce_tr)
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(nm.transform(ce_tr), y_tr)
report(clf.predict_proba(nm.transform(ce_va))[:, 1], 'Normalizer LogReg')
