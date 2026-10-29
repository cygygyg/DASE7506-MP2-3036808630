"""困难样本加权 + 模型对比，重点关注 '1个正确' 题。"""
import sys, io
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl
from features import to_arrays


def n_correct_per_q(bank, gold):
    out = []
    for g, row in zip(gold, bank):
        ga = parse_answer(g['answer'])
        out.append(sum(1 for c in row['candidates'] if parse_answer(c['text']) == ga))
    return np.array(out)


def run(clf_factory, use_meta, weight_hard, seed=0):
    X_emb_tr, X_meta_tr, y_tr, bank_tr, gold_tr, _ = to_arrays('train')
    X_emb_va, X_meta_va, y_va, bank_va, gold_va, _ = to_arrays('validation')
    if use_meta:
        Xtr = np.hstack([X_emb_tr, X_meta_tr]); Xva = np.hstack([X_emb_va, X_meta_va])
    else:
        Xtr = X_emb_tr; Xva = X_emb_va
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr); Xva = sc.transform(Xva)

    # sample weight: 按题难度
    nc = n_correct_per_q(bank_tr, gold_tr)
    w = np.ones(len(y_tr))
    for qi in range(len(nc)):
        if weight_hard and nc[qi] == 1:
            w[qi*4:qi*4+4] = 5.0   # 困难题5倍权重
    clf = clf_factory(seed)
    if hasattr(clf, 'sample_weight') is False and isinstance(clf, (LogisticRegression, MLPClassifier)):
        clf.fit(Xtr, y_tr, sample_weight=w)
    else:
        clf.fit(Xtr, y_tr)

    if hasattr(clf, 'predict_proba'):
        sco = clf.predict_proba(Xva)[:, 1]
    else:
        sco = clf.decision_function(Xva)
    sco = sco.reshape(len(bank_va), 4)
    for qi, row in enumerate(bank_va):
        for ci, c in enumerate(row['candidates']):
            if parse_answer(c['text']) is None:
                sco[qi, ci] = -np.inf

    nc_va = n_correct_per_q(bank_va, gold_va)
    from collections import defaultdict
    stats = defaultdict(lambda: [0, 0])
    for qi in range(len(bank_va)):
        ga = parse_answer(gold_va[qi]['answer'])
        choice = int(np.argmax(sco[qi]))
        a = parse_answer(bank_va[qi]['candidates'][choice]['text'])
        ok = int(a is not None and a == ga)
        stats[nc_va[qi]][0] += ok; stats[nc_va[qi]][1] += 1
    tot_c = sum(v[0] for v in stats.values()); tot_n = len(bank_va)
    return tot_c, tot_n, stats


if __name__ == '__main__':
    def logreg(s): return LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs', random_state=s)
    def mlp(s): return MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=300,
                                     early_stopping=True, random_state=s, validation_fraction=0.1)
    for use_meta in (True,):
        for hard in (False, True):
            for name, fac in [('LogReg', logreg), ('MLP', mlp)]:
                c, n, stats = run(fac, use_meta, hard)
                one = stats.get(1, [0, 0])
                print(f'meta={use_meta} hard_w={int(hard)} {name:8s}: total={100*c/n:.2f}% '
                      f'| #1correct={one[0]}/{one[1]}={100*one[0]/max(1,one[1]):.1f}%')
