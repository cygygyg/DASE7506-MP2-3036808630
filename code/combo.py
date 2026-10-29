"""中心化 embedding + meta + 困难加权 的组合探索。"""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl, load_inputs
from features import extract_features


def center(emb):
    return (emb - emb.mean(axis=1, keepdims=True))


def n_correct_per_q(bank, gold):
    return [sum(1 for c in row['candidates'] if parse_answer(c['text']) == parse_answer(g['answer']))
            for g, row in zip(gold, bank)]


def main():
    # train
    _, feats_tr, bank_tr, gold_tr = extract_features('train')
    emb_tr = load_inputs('train')[1]['embeddings']
    # validation
    _, feats_va, bank_va, gold_va = extract_features('validation')
    emb_va = load_inputs('validation')[1]['embeddings']

    keys = sorted(feats_tr[0].keys())
    meta_tr = np.array([[f[k] for k in keys] for f in feats_tr], dtype=np.float32)
    meta_va = np.array([[f[k] for k in keys] for f in feats_va], dtype=np.float32)

    def Y(bank, gold):
        y = np.zeros(len(bank)*4, dtype=np.int8)
        for qi, (g, row) in enumerate(zip(gold, bank)):
            ga = parse_answer(g['answer'])
            for ci, c in enumerate(row['candidates']):
                a = parse_answer(c['text'])
                y[qi*4+ci] = int(a is not None and a == ga)
        return y
    y_tr = Y(bank_tr, gold_tr); y_va = Y(bank_va, gold_va)

    # 中心化 embedding
    ce_tr = center(emb_tr).reshape(-1, 1024)
    ce_va = center(emb_va).reshape(-1, 1024)

    nc_tr = n_correct_per_q(bank_tr, gold_tr)
    nc_va = n_correct_per_q(bank_va, gold_va)

    def build(use_meta, use_center, center_mode):
        if use_center:
            Xtr_e = center(emb_tr).reshape(-1, 1024)
            Xva_e = center(emb_va).reshape(-1, 1024)
        else:
            Xtr_e = emb_tr.reshape(-1, 1024)
            Xva_e = emb_va.reshape(-1, 1024)
        if center_mode == 'l2':
            Xtr_e = Xtr_e / (np.linalg.norm(Xtr_e, axis=1, keepdims=True)+1e-9)
            Xva_e = Xva_e / (np.linalg.norm(Xva_e, axis=1, keepdims=True)+1e-9)
        parts = [Xtr_e]; partv = [Xva_e]
        if use_meta:
            parts.append(meta_tr); partv.append(meta_va)
        return np.hstack(parts), np.hstack(partv)

    def evalfit(clf, Xtr, Xva, hard_w):
        sc = StandardScaler().fit(Xtr)
        w = np.ones(len(y_tr))
        if hard_w:
            for qi in range(len(nc_tr)):
                if nc_tr[qi] == 1:
                    w[qi*4:qi*4+4] = 5.0
        clf.fit(sc.transform(Xtr), y_tr, sample_weight=w)
        sco = clf.predict_proba(sc.transform(Xva))[:, 1].reshape(len(bank_va), 4)
        for qi, row in enumerate(bank_va):
            for ci, c in enumerate(row['candidates']):
                if parse_answer(c['text']) is None:
                    sco[qi, ci] = -np.inf
        stats = defaultdict(lambda: [0, 0]); tot=[0,0]
        for qi in range(len(bank_va)):
            ga = parse_answer(gold_va[qi]['answer'])
            ch = int(np.argmax(sco[qi]))
            a = parse_answer(bank_va[qi]['candidates'][ch]['text'])
            ok = int(a is not None and a == ga)
            tot[0]+=ok; tot[1]+=1; stats[nc_va[qi]][0]+=ok; stats[nc_va[qi]][1]+=1
        one = stats[1]
        return tot[0], tot[1], one[0], one[1]

    def logreg(): return LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs')
    def mlp(): return MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=300,
                                    early_stopping=True, validation_fraction=0.1)

    configs = [
        ('center', True, False, 'raw'),
        ('center+meta', True, True, 'raw'),
        ('center+meta+hardw', True, True, 'raw'),
        ('center_l2+meta', True, True, 'l2'),
        ('center_l2+meta+hardw', True, True, 'l2'),
    ]
    for name, use_center, use_meta, cmode in configs:
        Xtr, Xva = build(use_meta, use_center, cmode)
        for clfname, fac in [('LogReg', logreg), ('MLP', mlp)]:
            hard_w = 'hardw' in name
            c, n, o1, n1 = evalfit(fac(), Xtr, Xva, hard_w)
            print(f'{name:24s} {clfname:8s}: total={100*c/n:.2f}% | #1correct={o1}/{n1}={100*o1/n1:.1f}%')


if __name__ == '__main__':
    main()
