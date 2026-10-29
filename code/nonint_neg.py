"""答案值确定性规则综合测试：非整数/负数排除。"""
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


def ans_val(text):
    a = parse_answer(text)
    if a is None:
        return None
    try:
        fr = Fraction(a)
        return fr
    except Exception:
        return None


# 统计 train/test 里 负数、非整数 正确答案 的比例
for split in ('train', 'validation', 'test'):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    neg_gold = 0; total_gold = 0
    for g in gold:
        fr = ans_val(g['answer'])
        if fr is not None:
            total_gold += 1
            if fr.numerator * fr.denominator < 0:
                neg_gold += 1
    print(f'{split}: gold 答案中负数比例 = {neg_gold}/{total_gold} = {100*neg_gold/total_gold:.2f}%')


emb_tr, bank_tr, gold_tr = load('train')
emb_va, bank_va, gold_va = load('validation')
emb_te, bank_te, gold_te = load('test')
y_tr = Y(bank_tr, gold_tr)
nc_va = n_correct(bank_va, gold_va)
nc_te = n_correct(bank_te, gold_te)
ce_tr = (emb_tr - emb_tr.mean(axis=1, keepdims=True)).reshape(-1, 1024)
sc = StandardScaler().fit(ce_tr)
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc.transform(ce_tr), y_tr)


def scores(emb, bank, mode):
    ce = (emb - emb.mean(axis=1, keepdims=True)).reshape(-1, 1024)
    sco = clf.predict_proba(sc.transform(ce))[:, 1].reshape(len(bank), 4)
    for qi, row in enumerate(bank):
        for ci, c in enumerate(row['candidates']):
            fr = ans_val(c['text'])
            if fr is None:
                sco[qi, ci] = -np.inf
                continue
            is_int = fr.denominator == 1
            is_neg = fr.numerator * fr.denominator < 0
            if mode == 'nonint' and not is_int:
                sco[qi, ci] = -np.inf
            elif mode == 'nonint_neg' and (not is_int or is_neg):
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
    print(f'{name:24s}: total={100*tot[0]/tot[1]:.2f}% | testlike={100*tl:.2f}% | #1={one[0]}/{one[1]}')


for mode in ['none', 'nonint', 'nonint_neg']:
    sva = scores(emb_va, bank_va, mode)
    report(sva, bank_va, gold_va, nc_va, f'val {mode}')
print()
for mode in ['none', 'nonint', 'nonint_neg']:
    ste = scores(emb_te, bank_te, mode)
    report(ste, bank_te, gold_te, nc_te, f'test {mode}')
