"""分析"只有1个正确候选"的题：失败模式。"""
import sys, io
import numpy as np
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl


def show_cases(split, max_cases=4):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    shown = 0
    n_one_correct = 0
    n_majority_wrong = 0
    for g, row in zip(gold, bank):
        ga = parse_answer(g['answer'])
        ans = [parse_answer(c['text']) for c in row['candidates']]
        correct_idx = [i for i, a in enumerate(ans) if a == ga]
        if len(correct_idx) != 1:
            continue
        n_one_correct += 1
        cnt = Counter(a for a in ans if a is not None)
        majority = cnt.most_common(1)[0][0]
        if majority != ga:
            n_majority_wrong += 1
            if shown < max_cases:
                shown += 1
                print(f'===== 案例 ({split}) 正确答案={ga} ====')
                print('问题:', g['question'])
                for c in row['candidates']:
                    a = parse_answer(c['text'])
                    tag = '★正确' if a == ga else ('无答案' if a is None else f'错误ans={a}')
                    print(f'  [候选{c["index"]}] {tag}')
                    print(f'      {c["text"][:200]}')
                print()
    print(f'[{split}] "1个正确"题数={n_one_correct}, 其中众数=错误答案的题数={n_majority_wrong}')


show_cases('validation', max_cases=3)
