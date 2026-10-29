"""诊断：困难题(#1)里正确候选的分数排名分布。"""
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


emb_tr, bank_tr, gold_tr = load('train')
emb_te, bank_te, gold_te = load('test')
y_tr = Y(bank_tr, gold_tr)
ce_tr = (emb_tr - emb_tr.mean(axis=1, keepdims=True)).reshape(-1, 1024)
ce_te = (emb_te - emb_te.mean(axis=1, keepdims=True)).reshape(-1, 1024)
sc = StandardScaler().fit(ce_tr)
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc.transform(ce_tr), y_tr)
sco = clf.predict_proba(sc.transform(ce_te))[:, 1].reshape(len(bank_te), 4)

# 在 #1 题里，看正确候选的排名
rank_dist = Counter()
margin_stats = []  # 正确候选分数 - 最高错误候选分数
for qi, (g, row) in enumerate(zip(gold_te, bank_te)):
    ga = parse_answer(g['answer'])
    correct_idx = [ci for ci, c in enumerate(row['candidates']) if parse_answer(c['text']) == ga]
    if len(correct_idx) != 1:
        continue
    ci = correct_idx[0]
    s = sco[qi].copy()
    s_rank = np.argsort(-s)  # 降序
    rank = int(np.where(s_rank == ci)[0][0]) + 1
    rank_dist[rank] += 1
    wrong = [j for j in range(4) if j != ci]
    margin = s[ci] - max(s[j] for j in wrong)
    margin_stats.append(margin)

print('test #1题 正确候选排名分布:', dict(sorted(rank_dist.items())))
print('test #1题 正确候选 vs 最高错误候选 分数差(margin):')
m = np.array(margin_stats)
print(f'  margin 均值={m.mean():.4f}, 中位数={np.median(m):.4f}')
print(f'  margin>0 比例(选对)={np.mean(m>0):.3f}')
print(f'  margin 分位数: 25%={np.quantile(m,0.25):.4f}, 75%={np.quantile(m,0.75):.4f}')

# 正确候选排前2 的比例（"可恢复"空间）
top2 = sum(rank_dist[k] for k in (1, 2))
print(f'  正确候选排第1或第2的比例: {top2}/{sum(rank_dist.values())} = {100*top2/sum(rank_dist.values()):.1f}%')
