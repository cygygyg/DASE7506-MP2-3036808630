"""按难度分层诊断 + test 参考分数。"""
import sys, io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl
from features import to_arrays


def get_scores(X_emb_tr, X_meta_tr, y_tr, X_emb_va, X_meta_va, use_meta=True):
    if use_meta:
        Xtr = np.hstack([X_emb_tr, X_meta_tr]); Xva = np.hstack([X_emb_va, X_meta_va])
    else:
        Xtr = X_emb_tr; Xva = X_emb_va
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs').fit(sc.transform(Xtr), y_tr)
    return clf.predict_proba(sc.transform(Xva))[:, 1].reshape(-1, 4)


def apply_none_penalty(sco, bank):
    for qi, row in enumerate(bank):
        for ci, c in enumerate(row['candidates']):
            if parse_answer(c['text']) is None:
                sco[qi, ci] = -np.inf
    return sco


def stratified_eval(sco, bank, gold):
    # 按正确候选数分层
    from collections import defaultdict
    stats = defaultdict(lambda: [0, 0])
    for qi, row in enumerate(bank):
        ga = parse_answer(gold[qi]['answer'])
        n_correct = sum(1 for c in row['candidates'] if parse_answer(c['text']) == ga)
        choice = int(np.argmax(sco[qi]))
        a = parse_answer(bank[qi]['candidates'][choice]['text'])
        ok = int(a is not None and a == ga)
        stats[n_correct][0] += ok
        stats[n_correct][1] += 1
    for k in sorted(stats):
        c, n = stats[k]
        print(f'  #{k} correct candidates: {c}/{n} = {100*c/n:.1f}%')


if __name__ == '__main__':
    X_emb_tr, X_meta_tr, y_tr, _, _, _ = to_arrays('train')
    for split in ('validation', 'test'):
        X_emb, X_meta, y, bank, gold, _ = to_arrays(split)
        sco = get_scores(X_emb_tr, X_meta_tr, y_tr, X_emb, X_meta)
        sco = apply_none_penalty(sco, bank)
        correct = 0
        for qi in range(len(bank)):
            ga = parse_answer(gold[qi]['answer'])
            choice = int(np.argmax(sco[qi]))
            a = parse_answer(bank[qi]['candidates'][choice]['text'])
            correct += int(a is not None and a == ga)
        print(f'=== {split}: LogReg-emb+meta selection = {100*correct/len(bank):.2f}% ({correct}/{len(bank)}) ===')
        stratified_eval(sco, bank, gold)
        print()
