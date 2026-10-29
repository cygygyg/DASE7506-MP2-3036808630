"""最后尝试：二次特征 / 题目上下文 / 非线性核近似。"""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler, Nystroem
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

def center(emb): return emb - emb.mean(axis=1, keepdims=True)
ce_tr = center(emb_tr); ce_va = center(emb_va)
ce_tr_f = ce_tr.reshape(-1, 1024); ce_va_f = ce_va.reshape(-1, 1024)
q_tr = emb_tr.mean(axis=1)  # 题目成分
q_va = emb_va.mean(axis=1)


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


def fit_report(Xtr, Xva, name):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc.transform(Xtr), y_tr)
    report(clf.predict_proba(sc.transform(Xva))[:, 1], name)


# baseline
fit_report(ce_tr_f, ce_va_f, 'center (baseline)')
# 二次特征
fit_report(np.hstack([ce_tr_f, ce_tr_f**2]), np.hstack([ce_va_f, ce_va_f**2]), 'center+square')
# 题目上下文
q_tr_rep = np.repeat(q_tr, 4, axis=0); q_va_rep = np.repeat(q_va, 4, axis=0)
fit_report(np.hstack([ce_tr_f, q_tr_rep]), np.hstack([ce_va_f, q_va_rep]), 'center+question')
# 二次 + 题目
fit_report(np.hstack([ce_tr_f, ce_tr_f**2, q_tr_rep]), np.hstack([ce_va_f, ce_va_f**2, q_va_rep]), 'center+sq+question')

# RBF 核近似 (Nystroem)
for k in [256, 512]:
    nys = Nystroem(kernel='rbf', gamma=0.01, n_components=k, random_state=0).fit(ce_tr_f)
    Xtr_k = nys.transform(ce_tr_f); Xva_k = nys.transform(ce_va_f)
    fit_report(Xtr_k, Xva_k, f'RBF-Nystroem({k})')
