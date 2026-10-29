"""文本特征实验：推理质量特征是否有独立信息。"""
import sys, io, re
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl, load_inputs


def load(split):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    _, arrays = load_inputs(split)
    return arrays['embeddings'], bank, gold


def text_feats(bank):
    rows = []
    for row in bank:
        for c in row['candidates']:
            t = c['text']
            rows.append([
                len(t),                          # 总长度
                t.count('\n'),                   # 换行
                len(re.findall(r'\d+', t)),      # 数字个数
                t.count('='),                    # 等号
                t.count('+') + t.count('-') + t.count('*') + t.count('/'),  # 运算符
                t.count('$'),                    # LaTeX
                t.count('<<'),                   # GSM8K 计算标记
                float('####' in t),              # 结论标记 ####
                float('boxed' in t),             # 结论标记 boxed
                float(bool(re.search(r'(?i)(therefore|so |thus|hence|answer|finally|in total)', t))),
                len(t.split()),                  # 词数
                float(t.strip().endswith(('.', '!', '?'))),
            ])
    return np.array(rows, dtype=np.float32)


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


emb_tr, bank_tr, gold_tr = load('train')
emb_va, bank_va, gold_va = load('validation')
y_tr = Y(bank_tr, gold_tr)
nc_va = n_correct(bank_va, gold_va)
tf_tr = text_feats(bank_tr)
tf_va = text_feats(bank_va)
ce_tr = (emb_tr - emb_tr.mean(axis=1, keepdims=True)).reshape(-1, 1024)
ce_va = (emb_va - emb_va.mean(axis=1, keepdims=True)).reshape(-1, 1024)


def evaluate(Xtr, Xva, name):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs').fit(sc.transform(Xtr), y_tr)
    sco = clf.predict_proba(sc.transform(Xva))[:, 1].reshape(len(bank_va), 4)
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
    tl = one[0]/max(1,one[1])*w[0] + two[0]/max(1,two[1])*w[1] + three[0]/max(1,three[1])*w[2]
    print(f'{name:28s}: total={100*tot[0]/tot[1]:.2f}% | testlike={100*tl:.2f}% | #1={one[0]}/{one[1]}')


evaluate(ce_tr, ce_va, 'center (emb only)')
evaluate(tf_tr, tf_va, 'text only')
evaluate(np.hstack([ce_tr, tf_tr]), np.hstack([ce_va, tf_va]), 'center+text')
