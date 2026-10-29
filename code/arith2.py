"""Improved chained-equality arithmetic parser and feature diagnostics."""
from __future__ import annotations
import re, math
from fractions import Fraction
import numpy as np

NUM = r'\d+(?:\.\d+)?'


def normalize_text(t: str) -> str:
    t = str(t)
    t = t.replace('\\left', ' ').replace('\\right', ' ')
    for _ in range(4):
        t = re.sub(r'\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'((\1)/(\2))', t)
    t = re.sub(r'\\(?:text|mathrm|mathbf|operatorname|boxed)\s*\{([^{}]*)\}', r' \1 ', t)
    t = t.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/')
    t = t.replace('\\%', '/100').replace('%', '/100')
    t = t.replace('×', '*').replace('÷', '/')
    t = t.replace('−', '-').replace('–', '-').replace('—', '-')
    t = t.replace('**', '*').replace('$', ' ').replace('\\', ' ')
    t = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', t)
    return t


def safe_eval(expr: str):
    expr = expr.strip().strip('()') if expr.count('(') == expr.count(')') and expr.startswith('(') and expr.endswith(')') else expr.strip()
    expr = expr.strip()
    if not expr or len(expr) > 160:
        return None
    if not re.fullmatch(r'[0-9eE.+\-*/() ]+', expr):
        return None
    if expr.count('(') != expr.count(')'):
        return None
    # avoid things like '..' or trailing operators
    if re.search(r'[.][.]|[+\-*/]\s*$', expr):
        return None
    # reject implicit multiplication such as "12(3)" or "(2)(3)"
    if re.search(r'\d\s*\(', expr) or re.search(r'\)\s*\d', expr) or re.search(r'\)\s*\(', expr):
        return None
    try:
        v = eval(expr, {'__builtins__': {}}, {})
    except Exception:
        return None
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def extract_chains(text: str):
    """Return list of chains. Each chain is (values, ok, raw_segments).

    A chain is a maximal run of numeric expressions joined by '=' (possibly
    chained, e.g. '1/4*8+7 = 2+7 = 9').  ok is True when all numeric segments
    agree.
    """
    t = normalize_text(text)
    # keep only characters that can occur in numeric expressions / equality
    t = re.sub(r'[^0-9eE.+\-*/()= \n]', '|', t)
    chains = []
    for chunk in re.split(r'[|\n]+', t):
        if '=' not in chunk:
            continue
        parts = [p.strip() for p in chunk.split('=')]
        vals = []
        segs = []
        okparse = True
        for p in parts:
            if not p:
                continue
            v = safe_eval(p)
            if v is None:
                # a segment with letters/variables: stop the chain here
                okparse = False
                continue
            vals.append(v); segs.append(p)
        if len(vals) >= 2:
            ok = all(abs(v - vals[0]) <= 1e-6 * max(1.0, abs(vals[0])) for v in vals[1:])
            chains.append((vals, ok, segs))
    return chains


def arith2_features(text, final):
    chains = extract_chains(text)
    n = len(chains)
    bad = sum(1 for c in chains if not c[1])
    f = {
        'b_n_chain': n,
        'b_n_bad': bad,
        'b_bad_ratio': bad / n if n else 0.0,
        'b_has_chain': float(n > 0),
        'b_last_ok': float(chains[-1][1]) if n else 0.0,
        'b_last_missing': float(n == 0),
    }
    all_vals = [v for c in chains for v in c[0]]
    good_vals = [v for c in chains if c[1] for v in c[0]]
    if final is None:
        f.update({'b_final_is_rhs': 0.0, 'b_final_eq_last': 0.0, 'b_final_in_last': 0.0,
                  'b_final_supported': 0.0, 'b_final_minus_last': 0.0})
    else:
        last = chains[-1][0][-1] if n else None
        f['b_final_is_rhs'] = float(any(abs(v - final) <= 1e-6 * max(1, abs(final)) for v in good_vals))
        f['b_final_eq_last'] = float(last is not None and abs(last - final) <= 1e-6 * max(1, abs(final)))
        f['b_final_in_last'] = float(n > 0 and any(abs(v - final) <= 1e-6 * max(1, abs(final)) for v in chains[-1][0]))
        f['b_final_supported'] = float(f['b_final_is_rhs'] or f['b_final_eq_last'])
        f['b_final_minus_last'] = float(abs(final - last)) if last is not None else 0.0
    f['b_n_values'] = len(all_vals)
    f['b_n_good_values'] = len(good_vals)
    f['b_n_equals'] = text.count('=')
    f['b_n_ops'] = sum(text.count(op) for op in '+-*/')
    return f
