"""验证离群度假设 + center 变体 + 分类器对比。"""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
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


# 先验证离群度假设
emb_va, bank_va, gold_va = load('validation')
nc_va = n_correct(bank_va, gold_va)
res_norm_corr, res_norm_err = [], []
for qi in range(len(bank_va)):
    if nc_va[qi] != 1:
        continue
    ga = parse_answer(gold_va[qi]['answer'])
    e = emb_va[qi]
    m = e.mean(axis=0)
    for ci in range(4):
        r = e[ci] - m
        rn = np.linalg.norm(r)
        a = parse_answer(bank_va[qi]['candidates'][ci]['text'])
        if a == ga:
            res_norm_corr.append(rn)
        else:
            res_norm_err.append(rn)
print('=== 离群度假设验证 (validation, 仅"1个正确"题) ===')
print(f'  正确候选 残差norm: mean={np.mean(res_norm_corr):.2f}')
print(f'  错误候选 残差norm: mean={np.mean(res_norm_err):.2f}')
print(f'  (正确残差 / 错误残差 比值: {np.mean(res_norm_corr)/np.mean(res_norm_err):.2f})')
