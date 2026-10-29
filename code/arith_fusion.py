"""算术验证特征 + center embedding 融合。"""
import sys, io, re
import numpy as np
from collections import defaultdict
from fractions import Fraction
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl, load_inputs


def load(split):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    _, arrays = load_inputs(split)
    return arrays['embeddings'], bank, gold


def Y(bank, gold):
    y = np.zeros(len(bank)*4, dtype=np.int8)
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        for ci, c in enumerate(row['candidates']):
            a = parse_answer(c['text'])
            y[qi*4+ci] = int(a is not None and a == ga)
    return y


def n_correct(bank, gold):
    return [sum(1 for c in row['candidates'] if parse_answer(c['text']) == parse_answer(g['answer']))
            for g, row in zip(gold, bank)]


def extract_equations(text):
    eqs = []
    pat = re.compile(r'(\d+(?:\.\d+)?(?:\s*[+\-*/]\s*\d+(?:\.\d+)?)+)\s*=\s*(\d+(?:\.\d+)?)')
    for m in pat.finditer(text):
        eqs.append((m.group(1), m.group(2)))
    return eqs


def eval_lhs(lhs):
    toks = re.findall(r'\d+(?:\.\d+)?|[+\-*/]', lhs)
    if not toks:
        return None
    try:
        i = 0
        while i < len(toks):
            if toks[i] in '*/':
                a = Fraction(toks[i-1]); b = Fraction(toks[i+1])
                v = a * b if toks[i] == '*' else a / b
                toks[i-1:i+2] = [str(v.numerator / v.denominator)]
                i -= 1
            else:
                i += 1
        result = Fraction(toks[0]); i = 1
        while i < len(toks):
            op = toks[i]; b = Fraction(toks[i+1])
            result = result + b if op == '+' else result - b
            i += 2
        return result.numerator / result.denominator
    except Exception:
        return None


def arith_feats(bank):
    """每个候选返回 [等式总数, 错误等式数, 错误比例]."""
    F = []
    for row in bank:
        for c in row['candidates']:
            eqs = extract_equations(c['text'])
            wrong = 0
            for lhs, rhs in eqs:
                v = eval_lhs(lhs)
                if v is not None and abs(v - float(Fraction(rhs))) > 1e-9:
                    wrong += 1
            total = len(eqs)
            ratio = wrong / total if total > 0 else 0.0
            F.append([float(total), float(wrong), float(ratio)])
    return np.array(F, dtype=np.float32)


emb_tr, bank_tr, gold_tr = load('train')
emb_va, bank_va, gold_va = load('validation')
y_tr = Y(bank_tr, gold_tr)
nc_va = n_correct(bank_va, gold_va)
ce_tr = (emb_tr - emb_tr.mean(axis=1, keepdims=True)).reshape(-1, 1024)
ce_va = (emb_va - emb_va.mean(axis=1, keepdims=True)).reshape(-1, 1024)
af_tr = arith_feats(bank_tr)
af_va = arith_feats(bank_va)


def report(sco, name):
    sco = sco.reshape(len(bank_va), 4)
    for qi, row in enumerate(bank_va):
        for ci, c in enumerate(row['candidates']):
            if parse_answer(c['text']) is None:
                sco[qi, ci] = -np.inf
    stats = defaultdict(lambda: [0,0]); tot=[0,0]
    for qi in range(len(bank_va)):
        ga = parse_answer(gold_va[qi]['answer'])
        ch = int(np.argmax(sco[qi]))
        a = parse_answer(bank_va[qi]['candidates'][ch]['text'])
        ok = int(a is not None and a == ga)
        tot[0]+=ok; tot[1]+=1; stats[nc_va[qi]][0]+=ok; stats[nc_va[qi]][1]+=1
    one, two, three = stats[1], stats[2], stats[3]
    w = np.array([151,145,171])/467
    tl = one[0]/max(1,one[1])*w[0]+two[0]/max(1,two[1])*w[1]+three[0]/max(1,three[1])*w[2]
    print(f'{name:32s}: total={100*tot[0]/tot[1]:.2f}% | testlike={100*tl:.2f}% | #1={one[0]}/{one[1]}')


def fit_report(Xtr, Xva, name):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc.transform(Xtr), y_tr)
    report(clf.predict_proba(sc.transform(Xva))[:, 1], name)


fit_report(ce_tr, ce_va, 'center (baseline)')
fit_report(af_tr, af_va, 'arith only')
fit_report(np.hstack([ce_tr, af_tr]), np.hstack([ce_va, af_va]), 'center+arith')
