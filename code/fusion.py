"""多数投票分层 + 与 center LogReg 的融合。"""
import sys, io
import numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
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
nc_va = n_correct(bank_va, gold_va)
nc_te = n_correct(bank_te, gold_te)

Xtr = (emb_tr - emb_tr.mean(axis=1, keepdims=True)).reshape(-1, 1024)
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc.transform(Xtr), y_tr)


def logreg_scores(emb, bank):
    X = (emb - emb.mean(axis=1, keepdims=True)).reshape(-1, 1024)
    sco = clf.predict_proba(sc.transform(X))[:, 1].reshape(len(bank), 4)
    for qi, row in enumerate(bank):
        for ci, c in enumerate(row['candidates']):
            if parse_answer(c['text']) is None:
                sco[qi, ci] = -np.inf
    return sco


def majority_choice(row):
    ans = [parse_answer(c['text']) for c in row['candidates']]
    cnt = Counter(a for a in ans if a is not None)
    if not cnt:
        return None
    best, mx = cnt.most_common(1)[0]
    for i, a in enumerate(ans):
        if a == best:
            return i


def evaluate(chooser, emb, bank, gold, nc):
    stats = defaultdict(lambda: [0,0]); tot=[0,0]
    for qi, row in enumerate(bank):
        ga = parse_answer(gold[qi]['answer'])
        ch = chooser(qi, row)
        a = parse_answer(bank[qi]['candidates'][ch]['text'])
        ok = int(a is not None and a == ga)
        tot[0]+=ok; tot[1]+=1; stats[nc[qi]][0]+=ok; stats[nc[qi]][1]+=1
    return tot, stats


def report(name, tot, stats):
    def p(k): return f'{stats[k][0]}/{stats[k][1]}'
    print(f'{name:28s}: total={100*tot[0]/tot[1]:.2f}%  #1={p(1)} #2={p(2)} #3={p(3)} #4={p(4)}')


# 多数投票分层 (test)
tot, stats = evaluate(lambda qi,row: majority_choice(row) or 0, emb_te, bank_te, gold_te, nc_te)
report('majority (test)', tot, stats)

# center LogReg (test)
sco_te = logreg_scores(emb_te, bank_te)
tot, stats = evaluate(lambda qi,row: int(np.argmax(sco_te[qi])), emb_te, bank_te, gold_te, nc_te)
report('center-LogReg (test)', tot, stats)


# 融合策略：众数票数>=threshold 用众数，否则 LogReg
def make_fusion(sco, threshold):
    def f(qi, row):
        ans = [parse_answer(c['text']) for c in row['candidates']]
        cnt = Counter(a for a in ans if a is not None)
        if cnt:
            best, mx = cnt.most_common(1)[0]
            if mx >= threshold:
                for i, a in enumerate(ans):
                    if a == best:
                        return i
        return int(np.argmax(sco[qi]))
    return f

sco_va = logreg_scores(emb_va, bank_va)
print('\n=== 融合 (validation) ===')
for th in [2, 3]:
    tot, stats = evaluate(make_fusion(sco_va, th), emb_va, bank_va, gold_va, nc_va)
    report(f'fusion th={th} (val)', tot, stats)

print('\n=== 融合 (test, 参考) ===')
for th in [2, 3]:
    tot, stats = evaluate(make_fusion(sco_te, th), emb_te, bank_te, gold_te, nc_te)
    report(f'fusion th={th} (test)', tot, stats)
