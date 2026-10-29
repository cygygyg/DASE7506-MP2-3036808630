"""字符级文本相似度：正确候选是否是'异类'。"""
import sys, io, difflib
import numpy as np
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl


def char_sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def analyze(split):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    corr_sim = []; err_sim = []
    for g, row in zip(gold, bank):
        ga = parse_answer(g['answer'])
        texts = [c['text'] for c in row['candidates']]
        ans = [parse_answer(t) for t in texts]
        # 只关注 #1 题（唯一正确）
        correct_idx = [i for i, a in enumerate(ans) if a == ga]
        if len(correct_idx) != 1:
            continue
        ci = correct_idx[0]
        for i in range(4):
            others = [j for j in range(4) if j != i]
            sim_other = np.mean([char_sim(texts[i], texts[j]) for j in others])
            # 与同答案候选的相似度
            same = [j for j in others if ans[j] == ans[i]]
            sim_same = np.mean([char_sim(texts[i], texts[j]) for j in same]) if same else np.nan
            if i == ci:
                corr_sim.append(sim_other)
            else:
                err_sim.append(sim_other)
    print(f'=== {split} (#1题) ===')
    print(f'  正确候选 与其他候选字符相似度: mean={np.mean(corr_sim):.4f}')
    print(f'  错误候选 与其他候选字符相似度: mean={np.mean(err_sim):.4f}')
    print(f'  正确候选 相似度 < 错误候选相似度 的比例: {np.mean(np.array(corr_sim) < np.mean(err_sim)):.3f}')


for s in ('train', 'validation', 'test'):
    analyze(s)
