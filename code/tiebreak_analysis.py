"""分析'正确候选排第2'的题：找 tie-break 信号。"""
import sys, io
import numpy as np
from collections import Counter
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

# 收集 正确候选排第2 的题
cases = []
for qi, (g, row) in enumerate(zip(gold_te, bank_te)):
    ga = parse_answer(g['answer'])
    ans = [parse_answer(c['text']) for c in row['candidates']]
    correct_idx = [ci for ci, c in enumerate(row['candidates']) if parse_answer(c['text']) == ga]
    if len(correct_idx) != 1:
        continue
    ci = correct_idx[0]
    s = sco[qi]
    rank = int(np.argsort(-s).tolist().index(ci)) + 1
    if rank == 2:
        top1 = int(np.argsort(-s)[0])  # 分数最高的（错误）候选
        # 众数
        cnt = Counter(a for a in ans if a is not None)
        majority = cnt.most_common(1)[0][0]
        cases.append({
            'correct_ans': ans[ci],
            'top1_ans': ans[top1],
            'top1_is_majority': ans[top1] == majority,
            'correct_is_majority': ans[ci] == majority,
            'n_correct': sum(1 for a in ans if a == ga),
            'ans_count_top1': cnt[ans[top1]],
            'ans_count_correct': cnt[ans[ci]],
            'margin': s[ci] - s[top1],
        })

print(f'正确候选排第2的题数: {len(cases)}')
print(f'  其中 排第1的错误候选是众数: {sum(1 for c in cases if c["top1_is_majority"])}/{len(cases)}')
print(f'  其中 正确候选是众数: {sum(1 for c in cases if c["correct_is_majority"])}/{len(cases)}')
print(f'  其中 正确候选是非众数(唯一答案): {sum(1 for c in cases if c["ans_count_correct"]==1)}/{len(cases)}')

# 如果"选非众数里分数最高的"，能纠正多少？
# 规则：在 center 分数 top2 里，如果一个是众数一个不是，选非众数
corrected = 0
for qi, (g, row) in enumerate(zip(gold_te, bank_te)):
    ga = parse_answer(g['answer'])
    ans = [parse_answer(c['text']) for c in row['candidates']]
    correct_idx = [ci for ci, c in enumerate(row['candidates']) if parse_answer(c['text']) == ga]
    if len(correct_idx) != 1:
        continue
    ci = correct_idx[0]
    s = sco[qi]
    cnt = Counter(a for a in ans if a is not None)
    majority = cnt.most_common(1)[0][0]
    # top2
    order = np.argsort(-s)
    # 如果正确候选在top2，且它不是众数，且top1是众数 -> 规则可纠正
    rank = int(np.argsort(-s).tolist().index(ci)) + 1
    if rank == 2:
        top1 = int(order[0])
        if ans[top1] == majority and ans[ci] != majority:
            corrected += 1
print(f'\n规则"top2中弃众数选非众数"可纠正: {corrected} 题 (在36题中)')

# 反之，如果正确候选是众数（错误候选不是），选众数
corrected2 = 0
for qi, (g, row) in enumerate(zip(gold_te, bank_te)):
    ga = parse_answer(g['answer'])
    ans = [parse_answer(c['text']) for c in row['candidates']]
    correct_idx = [ci for ci, c in enumerate(row['candidates']) if parse_answer(c['text']) == ga]
    if len(correct_idx) != 1:
        continue
    ci = correct_idx[0]
    s = sco[qi]
    cnt = Counter(a for a in ans if a is not None)
    majority = cnt.most_common(1)[0][0]
    rank = int(np.argsort(-s).tolist().index(ci)) + 1
    if rank == 2:
        top1 = int(np.argsort(-s)[0])
        if ans[ci] == majority and ans[top1] != majority:
            corrected2 += 1
print(f'规则"top2中选众数弃非众数"可纠正: {corrected2} 题')
