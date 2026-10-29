"""center + 困难加权/PCA/特征选择 系统优化。"""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
y_tr = Y(bank_tr, gold_tr)
nc_tr = np.array(n_correct(bank_tr, gold_tr))
nc_va = np.array(n_correct(bank_va, gold_va))

def center(emb): return (emb - emb.mean(axis=1, keepdims=True)).reshape(-1, 1024)
ce_tr = center(emb_tr); ce_va = center(emb_va)


def testlike_weight(nc):
    # 按 test 比例给难度加权: #1,#2,#3 权重高, #4 权重低
    w = np.ones(len(nc))
    for i, n in enumerate(nc):
        if n == 1: w[i] = 3.0
        elif n == 2: w[i] = 2.0
        elif n == 3: w[i] = 1.0
        elif n == 4: w[i] = 0.3
    return w


def sample_weight_vec(w_per_q):
    w = np.zeros(len(nc_tr)*4)
    for qi in range(len(nc_tr)):
        w[qi*4:qi*4+4] = w_per_q[qi]
    return w


def evaluate(clf, Xtr, ytr, Xva, w=None):
    sc = StandardScaler().fit(Xtr)
    clf.fit(sc.transform(Xtr), ytr, sample_weight=w)
    sco = clf.predict_proba(sc.transform(Xva))[:, 1].reshape(len(bank_va), 4)
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
    return tot, stats


def report(name, tot, stats):
    one, two, three = stats[1], stats[2], stats[3]
    # test-like 加权分 (151:145:171 比例 -> 权重)
    w = np.array([151, 145, 171]) / 467
    tl = (one[0]/max(1,one[1]))*w[0] + (two[0]/max(1,two[1]))*w[1] + (three[0]/max(1,three[1]))*w[2]
    print(f'{name:26s}: total={100*tot[0]/tot[1]:.2f}% | testlike={100*tl:.2f}% | '
          f'#1={one[0]}/{one[1]} #2={two[0]}/{two[1]} #3={three[0]}/{three[1]}')


def logreg(C=1.0): return LogisticRegression(C=C, max_iter=5000, solver='lbfgs')

# baseline
tot, stats = evaluate(logreg(), ce_tr, y_tr, ce_va)
report('center (baseline)', tot, stats)

# 困难加权
print('--- 困难加权 ---')
for wname, wq in [('hard5x', sample_weight_vec(np.where(nc_tr==1, 5.0, 1.0))),
                  ('testlike', sample_weight_vec(testlike_weight(nc_tr))),
                  ('hard3x', sample_weight_vec(np.where(nc_tr==1, 3.0, 1.0)))]:
    tot, stats = evaluate(logreg(), ce_tr, y_tr, ce_va, wq)
    report('center+'+wname, tot, stats)

# PCA 降维
print('--- PCA 降维 ---')
for k in [64, 128, 256, 512]:
    pca = PCA(n_components=k, random_state=0).fit(ce_tr)
    Xtr_p = pca.transform(ce_tr); Xva_p = pca.transform(ce_va)
    tot, stats = evaluate(logreg(), Xtr_p, y_tr, Xva_p)
    report(f'center+PCA({k})', tot, stats)

# 方差过滤
print('--- 方差过滤 ---')
var = ce_tr.var(axis=0)
for frac in [0.5, 0.75, 0.9]:
    th = np.quantile(var, 1-frac)
    mask = var >= th
    tot, stats = evaluate(logreg(), ce_tr[:, mask], y_tr, ce_va[:, mask])
    report(f'center+varfilter({frac})', tot, stats)
