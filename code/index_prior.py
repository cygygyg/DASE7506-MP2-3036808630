"""检查候选 index 位置先验 + embedding norm 与正确性。"""
import sys, io
import numpy as np
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl, load_inputs

for split in ('train', 'validation', 'test'):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    _, arrays = load_inputs(split)
    emb = arrays['embeddings']
    idx_correct = Counter()
    idx_total = Counter()
    norm_corr = []; norm_err = []
    for g, row in zip(gold, bank):
        ga = parse_answer(g['answer'])
        for c in row['candidates']:
            a = parse_answer(c['text'])
            ok = a == ga
            idx_total[c['index']] += 1
            if ok:
                idx_correct[c['index']] += 1
            e = emb[0]  # placeholder
    # norm 分析
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        for ci, c in enumerate(row['candidates']):
            a = parse_answer(c['text'])
            n = np.linalg.norm(emb[qi, ci])
            if a == ga:
                norm_corr.append(n)
            else:
                norm_err.append(n)
    print(f'=== {split} ===')
    for i in range(4):
        c = idx_correct[i]; t = idx_total[i]
        print(f'  index {i}: {c}/{t} = {100*c/t:.1f}%')
    print(f'  raw norm: 正确={np.mean(norm_corr):.2f} 错误={np.mean(norm_err):.2f}')
