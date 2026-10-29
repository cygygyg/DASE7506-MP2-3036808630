"""分析 pairwise 相似度特征在'1个正确'题上的判别力。"""
import sys, io
import numpy as np
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl, load_inputs


def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb + 1e-9))


def analyze(split):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    _, arrays = load_inputs(split)
    emb = arrays['embeddings']
    # 统计：正确候选 vs 错误候选 的 pairwise 特征
    corr_sim_other = []   # 正确候选与同题其他候选的平均 cos
    err_sim_other = []
    corr_sim_same = []    # 正确候选与同答案候选的 cos
    corr_sim_diff = []    # 正确候选与不同答案候选的 cos
    err_sim_same = []
    err_sim_diff = []
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        ans = [parse_answer(c['text']) for c in row['candidates']]
        e = emb[qi]
        for i in range(4):
            others = [j for j in range(4) if j != i]
            sim_other = np.mean([cos(e[i], e[j]) for j in others])
            same = [j for j in others if ans[j] == ans[i]]
            diff = [j for j in others if ans[j] != ans[i] and ans[j] is not None]
            sim_same = np.mean([cos(e[i], e[j]) for j in same]) if same else np.nan
            sim_diff = np.mean([cos(e[i], e[j]) for j in diff]) if diff else np.nan
            if ans[i] == ga:
                corr_sim_other.append(sim_other)
                corr_sim_same.append(sim_same); corr_sim_diff.append(sim_diff)
            else:
                err_sim_other.append(sim_other)
                err_sim_same.append(sim_same); err_sim_diff.append(sim_diff)
    def stats(v):
        v = np.array([x for x in v if not np.isnan(x)])
        return f'mean={v.mean():.3f} std={v.std():.3f}'
    print(f'=== {split} ===')
    print('  正确候选: sim_other', stats(corr_sim_other), '| sim_same', stats(corr_sim_same), '| sim_diff', stats(corr_sim_diff))
    print('  错误候选: sim_other', stats(err_sim_other), '| sim_same', stats(err_sim_same), '| sim_diff', stats(err_sim_diff))


for s in ('train', 'validation', 'test'):
    analyze(s)
