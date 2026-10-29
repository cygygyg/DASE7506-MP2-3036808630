import json, sys, io
import numpy as np
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl, load_inputs

for split in ('train', 'validation', 'test'):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    n = len(bank)
    n_correct_per_q = []
    n_parse_none = 0
    distinct_ans = []
    for g, row in zip(gold, bank):
        gold_ans = parse_answer(g['answer'])
        cands = []
        for c in row['candidates']:
            a = parse_answer(c['text'])
            if a is None:
                n_parse_none += 1
            cands.append(a)
        n_correct = sum(1 for a in cands if a is not None and a == gold_ans)
        n_correct_per_q.append(n_correct)
        distinct = len(set(a for a in cands if a is not None))
        distinct_ans.append(distinct)
    cnt = Counter(n_correct_per_q)
    print(f'=== {split} (n={n}) ===')
    print('  #correct candidates per question:', dict(sorted(cnt.items())))
    print('  mean correct per q:', np.mean(n_correct_per_q))
    print('  candidates with parse_answer None:', n_parse_none)
    print('  distinct parsed answers per q (mean):', np.mean(distinct_ans))
    print()

# 检查 embedding 基本信息
bank, arrays = load_inputs('train')
emb = arrays['embeddings']
print('embedding shape:', emb.shape, emb.dtype)
print('embedding stats: mean=%.4f std=%.4f min=%.3f max=%.3f' % (
    emb.mean(), emb.std(), emb.min(), emb.max()))
print('L2 norm mean:', np.linalg.norm(emb, axis=-1).mean())
