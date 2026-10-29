"""诊断：embedding 主要编码什么？答案值回归 vs 推理判别。"""
import sys, io
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl, load_inputs
from fractions import Fraction


def load(split):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    _, arrays = load_inputs(split)
    return arrays['embeddings'], bank, gold


emb_tr, bank_tr, gold_tr = load('train')
emb_va, bank_va, gold_va = load('validation')

def ans_value(text):
    a = parse_answer(text)
    if a is None:
        return None
    try:
        fr = Fraction(a)
        return fr.numerator / fr.denominator
    except Exception:
        return None

# 构建 答案值 标签
def build(split_emb, bank):
    X = split_emb.reshape(-1, 1024)
    vals = []
    for row in bank:
        for c in row['candidates']:
            v = ans_value(c['text'])
            vals.append(v if v is not None else np.nan)
    vals = np.array(vals)
    mask = ~np.isnan(vals)
    return X[mask], vals[mask]

Xtr, ytr = build(emb_tr, bank_tr)
Xva, yva = build(emb_va, bank_va)
print(f'答案值回归样本: train={len(ytr)}, val={len(yva)}')

sc = StandardScaler().fit(Xtr)
reg = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
pred = reg.predict(sc.transform(Xva))
ss_res = np.sum((yva - pred)**2)
ss_tot = np.sum((yva - yva.mean())**2)
r2 = 1 - ss_res/ss_tot
print(f'embedding -> 答案值 回归 R² = {r2:.3f}')

# 对比：center embedding 的判别 AUC
def Y(bank, gold):
    y = np.zeros(len(bank)*4, dtype=np.int8)
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        for ci, c in enumerate(row['candidates']):
            a = parse_answer(c['text'])
            y[qi*4+ci] = int(a is not None and a == ga)
    return y
from sklearn.metrics import roc_auc_score
ytr2 = Y(bank_tr, gold_tr); yva2 = Y(bank_va, gold_va)
ce_tr = (emb_tr - emb_tr.mean(axis=1, keepdims=True)).reshape(-1, 1024)
ce_va = (emb_va - emb_va.mean(axis=1, keepdims=True)).reshape(-1, 1024)
sc2 = StandardScaler().fit(ce_tr)
clf = LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs').fit(sc2.transform(ce_tr), ytr2)
auc = roc_auc_score(yva2, clf.predict_proba(sc2.transform(ce_va))[:, 1])
print(f'center embedding 判别 正确/错误 AUC = {auc:.3f}')
