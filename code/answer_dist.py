"""分析答案值分布：正确答案 vs 错误答案的数值特征。"""
import sys, io
import numpy as np
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl
from fractions import Fraction


def load(split):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    return bank, gold


def val_features(split):
    bank, gold = load(split)
    corr = []; err = []
    for g, row in zip(gold, bank):
        ga = parse_answer(g['answer'])
        for c in row['candidates']:
            a = parse_answer(c['text'])
            if a is None:
                continue
            try:
                fr = Fraction(a)
                fv = float(fr.numerator) / fr.denominator
            except Exception:
                continue
            feat = {
                'value': fv,
                'is_int': float(fr.denominator == 1),
                'positive': float(fv > 0),
                'ndigits': len(str(abs(fr.numerator))) if fr.denominator == 1 else -1,
                'denom': fr.denominator,
            }
            if a == ga:
                corr.append(feat)
            else:
                err.append(feat)
    return corr, err


for split in ('train', 'validation', 'test'):
    corr, err = val_features(split)
    print(f'=== {split} ===')
    print(f'  正确答案: n={len(corr)}, 整数比例={np.mean([f["is_int"] for f in corr]):.3f}, '
          f'正值比例={np.mean([f["positive"] for f in corr]):.3f}, 值中位数={np.median([f["value"] for f in corr]):.1f}')
    print(f'  错误答案: n={len(err)}, 整数比例={np.mean([f["is_int"] for f in err]):.3f}, '
          f'正值比例={np.mean([f["positive"] for f in err]):.3f}, 值中位数={np.median([f["value"] for f in err]):.1f}')
    # 值域
    cv = [f['value'] for f in corr]; ev = [f['value'] for f in err]
    print(f'  正确答案值范围: [{min(cv):.1f}, {max(cv):.1f}]  错误答案值范围: [{min(ev):.1f}, {max(ev):.1f}]')
