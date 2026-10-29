"""软融合：center_score + alpha * majority_bonus 扫描。"""
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
nc_va = n_correct(bank_va, gold_va)
nc_te = n_correct(bank_te, gold_te)

ce_tr = (emb_tr - emb_tr.mean(axis=1, keepdims=True)).reshape(-1, 1024)
sc = StandardScaler().fit(ce_tr)
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc.transform(ce_tr), y_tr)

def center_scores(emb, bank):
    ce = (emb - emb.mean(axis=1, keepdims=True)).reshape(-1, 1024)
    return clf.predict_proba(sc.transform(ce))[:, 1].reshape(len(bank), 4)

def majority_bonus(bank):
    N = len(bank)
    B = np.zeros((N, 4))
    for qi, row in enumerate(bank):
        ans = [parse_answer(c['text']) for c in row['candidates']]
        cnt = Counter(a for a in ans if a is not None)
        if not cnt:
            continue
        best, mx = cnt.most_common(1)[0]
        for ci, a in enumerate(ans):
            if a == best:
                B[qi, ci] = mx  # 票数作为 bonus
    return B

sco_va = center_scores(emb_va, bank_va)
sco_te = center_scores(emb_te, bank_te)
B_va = majority_bonus(bank_va)
B_te = majority_bonus(bank_te)

def eval_alpha(sco, B, bank, gold, nc, alpha):
    s = sco + alpha * B
    for qi, row in enumerate(bank):
        for ci, c in enumerate(row['candidates']):
            if parse_answer(c['text']) is None:
                s[qi, ci] = -np.inf
    stats = defaultdict(lambda: [0,0]); tot=[0,0]
    for qi in range(len(bank)):
        ga = parse_answer(gold[qi]['answer'])
        ch = int(np.argmax(s[qi]))
        a = parse_answer(bank[qi]['candidates'][ch]['text'])
        ok = int(a is not None and a == ga)
        tot[0]+=ok; tot[1]+=1; stats[nc[qi]][0]+=ok; stats[nc[qi]][1]+=1
    one, two, three = stats[1], stats[2], stats[3]
    w = np.array([151,145,171])/467
    tl = one[0]/max(1,one[1])*w[0]+two[0]/max(1,two[1])*w[1]+three[0]/max(1,three[1])*w[2]
    return tot[0]/tot[1]*100, tl*100, (one, two, three)

print('alpha  val_total  val_testlike  (val #1/#2/#3)')
for alpha in [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    t, tl, stats = eval_alpha(sco_va, B_va, bank_va, gold_va, nc_va, alpha)
    print(f'{alpha:5.2f}  {t:6.2f}%  {tl:6.2f}%  ({stats[0][0]}/{stats[0][1]},{stats[1][0]}/{stats[1][1]},{stats[2][0]}/{stats[2][1]})')

print('\n=== test 参考 (alpha 0 和 0.3) ===')
for alpha in [0, 0.1, 0.3, 0.5]:
    t, tl, stats = eval_alpha(sco_te, B_te, bank_te, gold_te, nc_te, alpha)
    print(f'alpha={alpha:4.2f}: test total={t:.2f}% | #1={stats[0][0]}/{stats[0][1]} #2={stats[1][0]}/{stats[1][1]} #3={stats[2][0]}/{stats[2][1]}')
