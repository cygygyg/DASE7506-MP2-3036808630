import json, sys, io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl, load_inputs


def build_dataset(split):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    _, arrays = load_inputs(split)
    emb = arrays['embeddings']  # (N,4,1024)
    N = len(bank)
    X = emb.reshape(N*4, -1)
    y = np.zeros(N*4, dtype=np.int8)
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        for ci, c in enumerate(row['candidates']):
            a = parse_answer(c['text'])
            y[qi*4+ci] = int(a is not None and a == ga)
    return X, y, bank, gold


def selection_accuracy(scores, gold, bank):
    """scores: (N,4) 分数，选 argmax，看是否正确。"""
    N = len(bank)
    correct = 0
    for qi in range(N):
        ga = parse_answer(gold[qi]['answer'])
        choice = int(np.argmax(scores[qi]))
        a = parse_answer(bank[qi]['candidates'][choice]['text'])
        correct += int(a is not None and a == ga)
    return correct, N


def main():
    Xtr, ytr, bank_tr, gold_tr = build_dataset('train')
    Xva, yva, bank_va, gold_va = build_dataset('validation')

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)

    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=2000, solver='lbfgs')
        clf.fit(Xtr_s, ytr)
        # 二分类 AUC
        auc = roc_auc_score(yva, clf.predict_proba(Xva_s)[:, 1])
        # 选择准确率
        sco_va = clf.predict_proba(Xva_s)[:, 1].reshape(len(bank_va), 4)
        c, n = selection_accuracy(sco_va, gold_va, bank_va)
        print(f'C={C:5.2f}  AUC={auc:.4f}  val_selection={100*c/n:.2f}% ({c}/{n})')


if __name__ == '__main__':
    main()
