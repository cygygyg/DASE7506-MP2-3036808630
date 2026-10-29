"""Feature extraction for the group-level joint ranker.

Four evidence blocks:
  1. group / answer structure + candidate text statistics
  2. local arithmetic verification (equations inside the trace)
  3. question-evidence interaction (question numbers, question type x operations)
  4. final computation chain (does the last step support the final answer)
"""
from __future__ import annotations

import math
import pickle
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

from common import parse_answer

CACHE = Path('cache')
CACHE.mkdir(exist_ok=True)

NUM = r'\d+(?:\.\d+)?'
ATOM = rf'\(?\s*{NUM}\s*\)?'
EXPR = rf'{ATOM}(?:\s*[+\-*/]\s*{ATOM})+'
EQ_RE = re.compile(rf'({EXPR})\s*=\s*({NUM})')
NUM_RE = re.compile(r'\d+(?:\.\d+)?')


def normalize_text(t: str) -> str:
    t = str(t)
    t = t.replace('\\left', ' ').replace('\\right', ' ')
    for _ in range(3):
        t = re.sub(r'\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'((\1)/(\2))', t)
    t = re.sub(r'\\(?:text|mathrm|mathbf|operatorname)\s*\{[^{}]*\}', ' ', t)
    t = t.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/')
    t = t.replace('\\%', '/100').replace('%', '/100')
    t = t.replace('×', '*').replace('÷', '/')
    t = t.replace('−', '-').replace('–', '-').replace('—', '-')
    t = t.replace('**', '*').replace('$', ' ').replace('\\', ' ')
    t = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', t)
    return t


def safe_eval(expr: str):
    expr = expr.strip()
    if not expr or len(expr) > 120:
        return None
    if not re.fullmatch(r'[0-9eE.+\-*/() ]+', expr):
        return None
    if expr.count('(') != expr.count(')'):
        return None
    try:
        v = eval(expr, {'__builtins__': {}}, {})
    except Exception:
        return None
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def extract_equations(text: str):
    """Return list of (lhs, rhs, ok, rhs_value) in text order (normalized text)."""
    t = normalize_text(text)
    out = []
    for m in EQ_RE.finditer(t):
        lhs, rhs = m.group(1), m.group(2)
        v = safe_eval(lhs)
        rv = safe_eval(rhs)
        ok = bool(v is not None and rv is not None and abs(v - rv) <= 1e-6 * max(1.0, abs(rv)))
        out.append((lhs, rhs, ok, rv))
    return out


def _nums(text):
    return [float(x) for x in NUM_RE.findall(normalize_text(text))]


def arith_features(text: str, final: float | None):
    t = normalize_text(text)
    eqs = extract_equations(text)
    n = len(eqs)
    bad = sum(1 for e in eqs if not e[2])
    rhs_vals = [e[3] for e in eqs if e[3] is not None]
    last = eqs[-1] if eqs else None
    f = {}
    f['a_n_eq'] = n
    f['a_n_bad'] = bad
    f['a_bad_ratio'] = bad / n if n else 0.0
    f['a_has_eq'] = float(n > 0)
    f['a_last_ok'] = float(last[2]) if last else 0.0
    f['a_last_missing'] = float(last is None)
    # final answer consistency with arithmetic results
    if final is None:
        f['a_final_is_rhs'] = 0.0
        f['a_final_in_any_rhs'] = 0.0
        f['a_final_in_last_eq'] = 0.0
        f['a_final_eq_last_rhs'] = 0.0
        f['a_final_minus_last'] = 0.0
    else:
        f['a_final_is_rhs'] = float(any(r is not None and abs(r - final) <= 1e-6 * max(1, abs(final)) for r in rhs_vals))
        f['a_final_in_any_rhs'] = f['a_final_is_rhs']
        last_rhs = last[3] if last and last[3] is not None else None
        f['a_final_eq_last_rhs'] = float(last_rhs is not None and abs(last_rhs - final) <= 1e-6 * max(1, abs(final)))
        f['a_final_in_last_eq'] = float(final in [abs(x) for x in _nums(last[0] + ' ' + last[1])]) if last else 0.0
        f['a_final_minus_last'] = float(abs(final - last_rhs)) if last_rhs is not None else 0.0
    f['a_n_rhs'] = len(set(round(r, 8) for r in rhs_vals))
    f['a_n_equals'] = t.count('=')
    f['a_n_ops'] = sum(t.count(op) for op in '+-*/')
    f['a_n_numbers'] = len(_nums(t))
    return f


STOP = set('a an the and or of to in on for with is are was were be been it its this that as at by from '
           'how many much what which who when where why if then than there their his her he she they them '
           'we you i do does did can will would should could total each per every'.split())


def chain_features(text: str, final: float | None, qnums: set[float]):
    """Final computation chain: linkage, provenance, and last-step support."""
    t = normalize_text(text)
    eqs = extract_equations(text)
    nums = _nums(t)
    f = {}
    f['c_n_eq'] = len(eqs)
    if eqs:
        # forward linkage: RHS of an earlier equation reused later
        links = 0
        for i, (lhs, rhs, ok, rv) in enumerate(eqs):
            if rv is None:
                continue
            later = ' '.join(e[0] + ' ' + e[1] for e in eqs[i + 1:])
            if any(abs(rv - x) <= 1e-6 * max(1, abs(rv)) for x in _nums(later)):
                links += 1
        f['c_chain_links'] = links
        f['c_chain_link_ratio'] = links / max(1, len(eqs))
        # provenance of operands in the last equation
        lhs, rhs, ok, rv = eqs[-1]
        ops = _nums(lhs)
        prior_rhs = [e[3] for e in eqs[:-1] if e[3] is not None]
        supported = 0
        for x in ops:
            if any(abs(x - q) <= 1e-6 * max(1, abs(q)) for q in qnums) or \
               any(abs(x - p) <= 1e-6 * max(1, abs(p)) for p in prior_rhs):
                supported += 1
        f['c_last_operand_support'] = supported / max(1, len(ops))
        f['c_last_operand_n'] = len(ops)
        f['c_last_unsupported'] = len(ops) - supported
    else:
        f['c_chain_links'] = 0
        f['c_chain_link_ratio'] = 0.0
        f['c_last_operand_support'] = 0.0
        f['c_last_operand_n'] = 0
        f['c_last_unsupported'] = 0
    # final answer provenance: question number, computed RHS, or unexplained
    if final is None:
        f['c_final_from_question'] = 0.0
        f['c_final_from_rhs'] = 0.0
        f['c_final_unexplained'] = 1.0
    else:
        fq = any(abs(final - q) <= 1e-6 * max(1, abs(q)) for q in qnums)
        fr = any(r is not None and abs(final - r) <= 1e-6 * max(1, abs(final))
                 for r in [e[3] for e in eqs])
        f['c_final_from_question'] = float(fq)
        f['c_final_from_rhs'] = float(fr)
        f['c_final_unexplained'] = float(not (fq or fr))
    # unsupported numbers in the whole trace
    known = set(round(x, 8) for x in qnums)
    known |= set(round(e[3], 8) for e in eqs if e[3] is not None)
    unk = [x for x in nums if round(x, 8) not in known]
    f['c_n_numbers'] = len(nums)
    f['c_unknown_numbers'] = len(unk)
    f['c_unknown_ratio'] = len(unk) / max(1, len(nums))
    # textual steps / concluding markers
    f['c_n_steps'] = len(re.findall(r'(?m)^\s*(?:\d+[.)]|[-*])\s', t))
    f['c_has_steps'] = float(f['c_n_steps'] > 0)
    return f


QKW = {
    'total': ('total', 'altogether', 'in all', 'sum of', 'combined'),
    'left': ('left', 'remain', 'remaining', 'how many more', 'fewer', 'lost'),
    'times': ('times', 'each', 'per ', 'every', 'rate', 'speed', 'hour', 'minute', 'day', 'week'),
    'average': ('average', 'mean'),
    'percent': ('percent', '%'),
    'ratio': ('ratio', 'fraction', 'half', 'third', 'twice', 'double', 'quarter'),
    'more': ('more than', 'less than', 'fewer than'),
    'howmany': ('how many', 'how much'),
}


def question_features(question: str):
    q = question.lower()
    f = {}
    for k, words in QKW.items():
        f['q_' + k] = float(any(w in q for w in words))
    qn = _nums(question)
    f['q_n_numbers'] = len(qn)
    f['q_n_unique_numbers'] = len(set(round(x, 8) for x in qn))
    f['q_len'] = len(question)
    f['q_words'] = len(question.split())
    return f, set(round(x, 8) for x in qn)


def value_feats(final: float | None, all_vals, qnums):
    f = {}
    if final is None:
        for k in ('v_valid', 'v_log', 'v_abs_log', 'v_sign', 'v_int', 'v_digits',
                  'v_rank', 'v_rel_max', 'v_rel_min', 'v_in_question', 'v_big'):
            f[k] = 0.0
        return f
    f['v_valid'] = 1.0
    a = abs(final)
    f['v_log'] = math.copysign(math.log1p(a), final) if final != 0 else 0.0
    f['v_abs_log'] = math.log1p(a)
    f['v_sign'] = float(final > 0) - float(final < 0)
    fr = Fraction(final).limit_denominator(10 ** 6)
    f['v_int'] = float(abs(final - round(final)) < 1e-9)
    f['v_digits'] = len(str(int(round(a)))) if f['v_int'] else len(str(int(a)))
    vals = [v for v in all_vals if v is not None]
    f['v_rank'] = float(sum(1 for v in vals if v < final)) / max(1, len(vals) - 1)
    mx = max(abs(v) for v in vals) if vals else a
    mn = min(abs(v) for v in vals) if vals else a
    f['v_rel_max'] = a / mx if mx else 0.0
    f['v_rel_min'] = a / mn if mn else 0.0
    f['v_in_question'] = float(any(abs(final - q) <= 1e-6 * max(1, abs(q)) for q in qnums))
    f['v_big'] = float(a >= 100)
    return f


def group_feats(group_sizes, valid, gi, final, all_vals, member_cand):
    """Structure + aggregates over candidate-level features for one group."""
    sizes = np.asarray(group_sizes, float)
    G = len(sizes)
    s = sizes[gi]
    f = {}
    f['s_size'] = s
    f['s_frac'] = s / max(1.0, sizes.sum())
    f['s_max'] = sizes.max()
    f['s_min'] = sizes.min()
    f['s_is_max'] = float(s >= sizes.max() - 1e-9)
    f['s_is_min'] = float(s <= sizes.min() + 1e-9)
    others = np.delete(sizes, gi) if G > 1 else sizes
    f['s_is_unique_max'] = float(G > 1 and s > others.max())
    f['s_is_unique_min'] = float(G > 1 and s < others.min())
    f['s_n_groups'] = G
    f['s_n_valid_groups'] = int(valid.sum())
    f['s_gap_max'] = s - sizes.max()
    f['s_rank_desc'] = float(np.sum(sizes > s))
    f['s_rank_asc'] = float(np.sum(sizes < s))
    f['s_is_plural'] = float(s >= sizes.max() - 1e-9)
    f['s_tie_max'] = float(np.sum(sizes >= sizes.max() - 1e-9))
    # aggregates over member candidate features
    for k in ('a_n_eq', 'a_n_bad', 'a_bad_ratio', 'a_last_ok', 'a_final_eq_last_rhs',
              'a_final_in_last_eq', 'c_chain_link_ratio', 'c_last_operand_support',
              'c_final_from_rhs', 'c_final_unexplained', 'c_unknown_ratio'):
        vals = np.array([m[k] for m in member_cand], float)
        f['g_' + k + '_mean'] = float(vals.mean())
        f['g_' + k + '_max'] = float(vals.max())
        f['g_' + k + '_min'] = float(vals.min())
    # group internal disagreement (reasoning diversity)
    for k in ('a_n_eq', 'a_bad_ratio', 'c_last_operand_support'):
        vals = np.array([m[k] for m in member_cand], float)
        f['g_' + k + '_std'] = float(vals.std())
    return f


def cand_text_feats(text):
    t = str(text)
    f = {}
    f['t_len'] = len(t)
    f['t_words'] = len(t.split())
    f['t_lines'] = t.count('\n')
    f['t_equals'] = t.count('=')
    f['t_boxed'] = float('boxed' in t)
    f['t_hash'] = float('####' in t)
    f['t_latex'] = float('$' in t)
    f['t_frac'] = t.count('\\frac') + t.count('\\dfrac')
    f['t_therefore'] = float(bool(re.search(r'(?i)(therefore|thus|hence|so the answer|final answer)', t)))
    f['t_lets'] = float(bool(re.search(r'(?i)(let |denote|assume)', t)))
    f['t_bullets'] = float(bool(re.search(r'(?m)^\s*(?:\d+[.)]|[-*])\s', t)))
    return f


def build(split, data):
    path = CACHE / f'feat_{split}.pkl'
    if path.exists():
        with path.open('rb') as fh:
            return pickle.load(fh)

    bank = data['bank']
    N = len(bank)
    q_feat_list = []
    cand_rows = []          # (N,4,F) candidate text/arith/chain/value features
    group_records = []      # list over questions: dict(gids, G, sizes, valid, feats (G,F), cand_feats (4,F))
    cand_names = None
    group_names = None

    for qi, row in enumerate(bank):
        q = row['question']
        texts = [c['text'] for c in row['candidates']]
        finals = [parse_answer(t) for t in texts]
        fvals = []
        for a in finals:
            if a is None:
                fvals.append(None)
            else:
                fr = Fraction(a)
                # the reference answers are all integers: a fractional final
                # answer can never be correct, so treat it as invalid
                if fr.denominator != 1:
                    fvals.append(None)
                else:
                    fvals.append(float(fr.numerator))
        qf, qnums = question_features(q)
        q_feat_list.append(qf)
        cf = []
        for ci, t in enumerate(texts):
            f = {}
            f.update(cand_text_feats(t))
            f.update(arith_features(t, fvals[ci]))
            f.update(chain_features(t, fvals[ci], qnums))
            f.update(value_feats(fvals[ci], fvals, qnums))
            # question x candidate interactions
            for k in QKW:
                f['x_' + k + '_ops'] = qf['q_' + k] * f['a_n_ops']
                f['x_' + k + '_eq'] = qf['q_' + k] * f['a_n_eq']
            f['x_qnums_used'] = len(set(round(x, 8) for x in _nums(t)) & qnums) / max(1, len(qnums))
            cf.append(f)
        # group structure
        groups = {}
        gids = []
        for ci, a in enumerate(fvals):
            key = ('a', a) if a is not None else ('none', ci)
            groups.setdefault(key, []).append(ci)
            gids.append(key)
        keys = list(groups.keys())
        key_to_g = {k: i for i, k in enumerate(keys)}
        gids = np.array([key_to_g[k] for k in gids], dtype=int)
        G = len(keys)
        sizes = np.array([len(groups[k]) for k in keys], dtype=int)
        valid = np.array([keys[g][0] == 'a' for g in range(G)], dtype=bool)
        gf = []
        for g in range(G):
            members = groups[keys[g]]
            final = fvals[members[0]] if keys[g][0] == 'a' else None
            gf.append(group_feats(sizes, valid, g, final, fvals, [cf[m] for m in members]))
        cand_rows.append(cf)
        group_records.append(dict(gids=gids, G=G, sizes=sizes, valid=valid, feats=gf, finals=fvals))
        if cand_names is None:
            cand_names = sorted(cf[0].keys())
            group_names = sorted(gf[0].keys())

    Xc = np.array([[ [c[k] for k in cand_names] for c in row] for row in cand_rows], dtype=np.float32)
    Gf = []
    for rec in group_records:
        Gf.append(np.array([[f[k] for k in group_names] for f in rec['feats']], dtype=np.float32))
    qX = np.array([[f[k] for k in sorted(q_feat_list[0])] for f in q_feat_list], dtype=np.float32)
    out = dict(cand=Xc, cand_names=cand_names, group=Gf, group_names=group_names,
               q=qX, q_names=sorted(q_feat_list[0]), records=group_records,
               finals=np.array([rec['finals'] for rec in group_records], dtype=object))
    with path.open('wb') as fh:
        pickle.dump(out, fh, protocol=4)
    return out


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    from mp2lib import load_split
    for s in ('train', 'validation', 'test'):
        d = load_split(s)
        f = build(s, d)
        print(s, 'cand', f['cand'].shape, 'group feats', len(f['group_names']), 'q', f['q'].shape)
        print('  cand names:', f['cand_names'])
        print('  group names:', f['group_names'])
