"""自洽性验证：推理最后一步结果 vs boxed 最终答案是否一致。"""
import sys, io, re
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl
from fractions import Fraction


def last_calc_result(text):
    """提取文本里最后一个 'X = N' 等式的结果 N。"""
    pat = re.compile(r'=\s*(\d+(?:\.\d+)?(?:/\d+)?)')
    matches = pat.findall(text)
    if not matches:
        return None
    # 最后一个等式的结果
    last = matches[-1]
    try:
        return float(Fraction(last.replace(' ', '')))
    except Exception:
        return None


def boxed_value(text):
    a = parse_answer(text)
    if a is None:
        return None
    try:
        return float(Fraction(a))
    except Exception:
        return None


def analyze(split):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    corr_inconsist = []; err_inconsist = []
    for g, row in zip(gold, bank):
        ga = parse_answer(g['answer'])
        for c in row['candidates']:
            a = parse_answer(c['text'])
            last = last_calc_result(c['text'])
            boxed = boxed_value(c['text'])
            if last is not None and boxed is not None:
                incons = abs(last - boxed) > 1e-9
                if a == ga:
                    corr_inconsist.append(float(incons))
                else:
                    err_inconsist.append(float(incons))
    print(f'=== {split} ===')
    print(f'  正确候选 推理与boxed不一致比例: n={len(corr_inconsist)}, 比例={np.mean(corr_inconsist) if corr_inconsist else 0:.3f}')
    print(f'  错误候选 推理与boxed不一致比例: n={len(err_inconsist)}, 比例={np.mean(err_inconsist) if err_inconsist else 0:.3f}')


for s in ('train', 'validation', 'test'):
    analyze(s)
