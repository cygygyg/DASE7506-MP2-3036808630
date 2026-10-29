"""综合硬规则 + 中心化变体。"""
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


def has_arith_error(text):
    pat = re.compile(r'(\d+(?:\.\d+)?(?:\s*[+\-*/]\s*\d+(?:\.\d+)?)+)\s*=\s*(\d+(?:\.\d+)?)')
    for m in pat.finditer(text):
        lhs = m.group(1); rhs = m.group(2)
        toks = re.findall(r'\d+(?:\.\d+)?|[+\-*/]', lhs)
        try:
            i = 0
            while i < len(toks):
                if toks[i] in '*/':
                    a = Fraction(toks[i-1]); b = Fraction(toks[i+1])
                    v = a * b if toks[i] == '*' else a / b
                    toks[i-1:i+2] = [str(v.numerator/v.denominator)]
                    i -= 1
                else:
                    i += 1
            r = Fraction(toks[0]); i = 1
            while i < len(toks):
                op = toks[i]; b = Fraction(toks[i+1])
                r = r + b if op == '+' else r - b
                i += 2
            if abs(r.numerator/r.denominator - float(Fraction(rhs))) > 1e-9:
                return True
        except Exception:
            pass
    return False


emb_tr, bank_tr, gold_tr = load('train')
emb_va, bank_va, gold_va = load('validation')
emb_te, bank_te, gold_te = load('test')
y_tr = Y(bank_tr, gold_tr)
nc_va = n_correct(bank_va, gold_va)
nc_te = n_correct(bank_te, gold_te)


def center_mode(emb, mode):
    if mode == 'mean':
        return emb - emb.mean(axis=1, keepdims=True)
    if mode == 'shortest':
        # 减最短候选
        N = emb.shape[0]
        lengths = np.array([[len(c['text']) for c in row['candidates']]] for row in bank_tr)  # placeholder
        # 需要 bank，改为在函数外处理
        return emb
    if mode == 'first':
        return emb - emb[:, 0:1, :]


def make_center(emb, bank, mode):
    if mode == 'mean':
        return emb - emb.mean(axis=1, keepdims=True)
    if mode == 'shortest':
        N = emb.shape[0]
        ref = np.zeros_like(emb)
        for qi in range(N):
            lens = [len(c['text']) for c in bank[qi]['candidates']]
            si = int(np.argmin(lens))
            ref[qi] = emb[qi, si]
        return emb - ref
    if mode == 'first':
        return emb - emb[:, 0:1, :]


def report(sco, bank, gold, nc, name):
    stats = defaultdict(lambda: [0,0]); tot=[0,0]
    for qi in range(len(bank)):
        ga = parse_answer(gold[qi]['answer'])
        ch = int(np.argmax(sco[qi]))
        a = parse_answer(bank[qi]['candidates'][ch]['text'])
        ok = int(a is not None and a == ga)
        tot[0]+=ok; tot[1]+=1; stats[nc[qi]][0]+=ok; stats[nc[qi]][1]+=1
    one, two, three = stats[1], stats[2], stats[3]
    w = np.array([151,145,171])/467
    tl = one[0]/max(1,one[1])*w[0]+two[0]/max(1,two[1])*w[1]+three[0]/max(1,three[1])*w[2]
    print(f'{name:30s}: total={100*tot[0]/tot[1]:.2f}% | testlike={100*tl:.2f}% | #1={one[0]}/{one[1]}')


for cmode in ['mean', 'shortest', 'first']:
    ce_tr = make_center(emb_tr, bank_tr, cmode).reshape(-1, 1024)
    ce_va = make_center(emb_va, bank_va, cmode).reshape(-1, 1024)
    ce_te = make_center(emb_te, bank_te, cmode).reshape(-1, 1024)
    sc = StandardScaler().fit(ce_tr)
    clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc.transform(ce_tr), y_tr)

    for rules in ['none', 'nonint', 'nonint+arith']:
        def sco_fn(emb, bank, cmode2):
            ce = make_center(emb, bank, cmode2).reshape(-1, 1024)
            s = clf.predict_proba(sc.transform(ce))[:, 1].reshape(len(bank), 4)
            for qi, row in enumerate(bank):
                for ci, c in enumerate(row['candidates']):
                    a = parse_answer(c['text'])
                    if a is None:
                        s[qi, ci] = -np.inf; continue
                    fr = Fraction(a)
                    if 'nonint' in rules and fr.denominator != 1:
                        s[qi, ci] = -np.inf
                    if 'arith' in rules and has_arith_error(c['text']):
                        s[qi, ci] = -np.inf
            return s
        sva = sco_fn(emb_va, bank_va, cmode)
        report(sva, bank_va, gold_va, nc_va, f'{cmode}+{rules} (val)')
    print()
