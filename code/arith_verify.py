"""算术等式验证：提取 'expr = result' 并验证，看错误候选是否更多算错。"""
import sys, io, re
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl
from fractions import Fraction


def extract_equations(text):
    """提取 '数字(运算数字)* = 数字' 等式，返回 [(lhs_str, rhs_value)]."""
    eqs = []
    # 匹配 数字 (op 数字)+ = 数字
    pat = re.compile(r'(\d+(?:\.\d+)?(?:\s*[+\-*/]\s*\d+(?:\.\d+)?)+)\s*=\s*(\d+(?:\.\d+)?)')
    for m in pat.finditer(text):
        lhs = m.group(1)
        rhs = m.group(2)
        eqs.append((lhs, rhs))
    return eqs


def eval_lhs(lhs):
    """安全求值 '3 + 4 * 5' 形式的表达式（无括号，只含 + - * / 数字）。"""
    # tokenize
    toks = re.findall(r'\d+(?:\.\d+)?|[+\-*/]', lhs)
    if not toks:
        return None
    try:
        # 先处理 * 和 /
        i = 0
        while i < len(toks):
            if toks[i] in '*/':
                a = Fraction(toks[i-1]); b = Fraction(toks[i+1])
                v = a * b if toks[i] == '*' else a / b
                toks[i-1:i+2] = [str(v.numerator / v.denominator)]
                i -= 1
            else:
                i += 1
        # 再处理 + 和 -
        result = Fraction(toks[0])
        i = 1
        while i < len(toks):
            op = toks[i]; b = Fraction(toks[i+1])
            result = result + b if op == '+' else result - b
            i += 2
        return result.numerator / result.denominator
    except Exception:
        return None


def verify(text):
    eqs = extract_equations(text)
    wrong = 0; total = 0
    for lhs, rhs in eqs:
        v = eval_lhs(lhs)
        if v is None:
            continue
        total += 1
        if abs(v - float(Fraction(rhs))) > 1e-9:
            wrong += 1
    return wrong, total


def analyze(split):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    corr_wrong = []; err_wrong = []
    for g, row in zip(gold, bank):
        ga = parse_answer(g['answer'])
        for c in row['candidates']:
            a = parse_answer(c['text'])
            w, t = verify(c['text'])
            if t > 0:
                ratio = w / t
                if a == ga:
                    corr_wrong.append(ratio)
                else:
                    err_wrong.append(ratio)
    print(f'=== {split} ===')
    print(f'  正确候选 含错误等式比例: n={len(corr_wrong)}, 平均={np.mean(corr_wrong) if corr_wrong else 0:.3f}')
    print(f'  错误候选 含错误等式比例: n={len(err_wrong)}, 平均={np.mean(err_wrong) if err_wrong else 0:.3f}')


for s in ('train', 'validation', 'test'):
    analyze(s)
