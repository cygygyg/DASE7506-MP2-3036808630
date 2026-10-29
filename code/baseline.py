import json, sys, io, random
import numpy as np
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl


def eval_split(split, method):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    correct = 0
    for g, row in zip(gold, bank):
        choice = method(row)
        a = parse_answer(row['candidates'][choice]['text'])
        if a is not None and a == parse_answer(g['answer']):
            correct += 1
    return correct, len(bank)


def method_random(row):
    return random.randint(0, 3)


def method_majority(row):
    """众数投票：解析最终答案，选出现次数最多的答案（平局选第一个出现的）。"""
    answers = [parse_answer(c['text']) for c in row['candidates']]
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return random.randint(0, 3)
    top = counts.most_common()
    best_ans = top[0][0]
    # 选第一个给出 best_ans 的候选
    for i, a in enumerate(answers):
        if a == best_ans:
            return i
    return 0


def method_majority_tiebreak_len(row):
    """众数投票，平局时选更短(更简洁)的候选。"""
    answers = [parse_answer(c['text']) for c in row['candidates']]
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return random.randint(0, 3)
    # 找最高频
    mx = max(counts.values())
    top_ans = [a for a, v in counts.items() if v == mx]
    # 在所有属于 top 答案的候选中，选文本最短的
    best = None
    best_len = 1e18
    for i, c in enumerate(row['candidates']):
        a = answers[i]
        if a in top_ans:
            if len(c['text']) < best_len:
                best_len = len(c['text'])
                best = i
    return best if best is not None else 0


if __name__ == '__main__':
    random.seed(0)
    for split in ('validation', 'test'):
        print(f'=== {split} ===')
        for name, m in [('random', method_random), ('majority', method_majority),
                        ('majority_tiebreak_len', method_majority_tiebreak_len)]:
            random.seed(0)
            c, n = eval_split(split, m)
            print(f'  {name:24s}: {c}/{n} = {100*c/n:.2f}%')
