"""测试 embedding 预处理：中心化(减题目成分) / L2归一化。"""
import sys, io
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


def prep(emb, mode):
    N = emb.shape[0]
    X = emb.reshape(N*4, -1)
    if mode == 'raw':
        pass
    elif mode == 'l2':
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    elif mode == 'center':
        m = emb.mean(axis=1, keepdims=True)  # 题目成分
        X = (emb - m).reshape(N*4, -1)
    elif mode == 'center_l2':
        m = emb.mean(axis=1, keepdims=True)
        X = (emb - m).reshape(N*4, -1)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X


def build_y(bank, gold):
    y = np.zeros(len(bank)*4, dtype=np.int8)
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        for ci, c in enumerate(row['candidates']):
            a = parse_answer(c['text'])
            y[qi*4+ci] = int(a is not None and a == ga)
    return y


def main():
    emb_tr, bank_tr, gold_tr = load('train')
    emb_va, bank_va, gold_va = load('validation')
    y_tr = build_y(bank_tr, gold_tr)
    y_va = build_y(bank_va, gold_va)
    nc_va = [sum(1 for c in row['candidates'] if parse_answer(c['text']) == parse_answer(g['answer']))
             for g, row in zip(gold_va, bank_va)]

    for mode in ['raw', 'l2', 'center', 'center_l2']:
        Xtr = prep(emb_tr, mode); Xva = prep(emb_va, mode)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs').fit(sc.transform(Xtr), y_tr)
        sco = clf.predict_proba(sc.transform(Xva))[:, 1].reshape(len(bank_va), 4)
        for qi, row in enumerate(bank_va):
            for ci, c in enumerate(row['candidates']):
                if parse_answer(c['text']) is None:
                    sco[qi, ci] = -np.inf
        stats = defaultdict(lambda: [0, 0]); tot = [0, 0]
        for qi in range(len(bank_va)):
            ga = parse_answer(gold_va[qi]['answer'])
            ch = int(np.argmax(sco[qi]))
            a = parse_answer(bank_va[qi]['candidates'][ch]['text'])
            ok = int(a is not None and a == ga)
            tot[0] += ok; tot[1] += 1
            stats[nc_va[qi]][0] += ok; stats[nc_va[qi]][1] += 1
        one = stats[1]
        print(f'{mode:10s}: total={100*tot[0]/tot[1]:.2f}% | #1correct={one[0]}/{one[1]}={100*one[0]/one[1]:.1f}%')


if __name__ == '__main__':
    main()
