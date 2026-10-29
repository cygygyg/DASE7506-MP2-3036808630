"""模型 + 特征组对比，validation 评估。"""
import json, sys, io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer
from features import to_arrays


def run_model(name, clf_factory, use_meta, normalize=True, seed=0):
    X_emb_tr, X_meta_tr, y_tr, _, _, _ = to_arrays('train')
    X_emb_va, X_meta_va, y_va, bank_va, gold_va, _ = to_arrays('validation')

    if use_meta:
        Xtr = np.hstack([X_emb_tr, X_meta_tr])
        Xva = np.hstack([X_emb_va, X_meta_va])
    else:
        Xtr = X_emb_tr
        Xva = X_emb_va

    if normalize:
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xva = scaler.transform(Xva)

    clf = clf_factory(seed)
    clf.fit(Xtr, y_tr)

    # AUC（只看可比较的二分类）
    if hasattr(clf, 'predict_proba'):
        auc = roc_auc_score(y_va, clf.predict_proba(Xva)[:, 1])
    else:
        auc = roc_auc_score(y_va, clf.decision_function(Xva))
    if hasattr(clf, 'predict_proba'):
        sco = clf.predict_proba(Xva)[:, 1]
    else:
        sco = clf.decision_function(Xva)
    sco = sco.reshape(len(bank_va), 4)

    # 对无法解析的候选强制 -inf
    for qi, row in enumerate(bank_va):
        for ci, c in enumerate(row['candidates']):
            if parse_answer(c['text']) is None:
                sco[qi, ci] = -np.inf

    correct = 0
    for qi in range(len(bank_va)):
        ga = parse_answer(gold_va[qi]['answer'])
        choice = int(np.argmax(sco[qi]))
        a = parse_answer(bank_va[qi]['candidates'][choice]['text'])
        correct += int(a is not None and a == ga)
    acc = 100*correct/len(bank_va)
    print(f'{name:40s} AUC={auc:.4f}  val_selection={acc:.2f}% ({correct}/{len(bank_va)})')
    return acc


def logreg(seed):
    return LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs', random_state=seed)

def mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=200,
                         early_stopping=True, random_state=seed, validation_fraction=0.1)

def hgb(seed):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                          max_leaf_nodes=31, random_state=seed,
                                          early_stopping=True, validation_fraction=0.1)


if __name__ == '__main__':
    print('=== embedding only ===')
    run_model('LogReg-emb', logreg, False)
    run_model('MLP-emb', mlp, False)
    run_model('HGB-emb', hgb, False)
    print('=== embedding + meta ===')
    run_model('LogReg-emb+meta', logreg, True)
    run_model('MLP-emb+meta', mlp, True)
    run_model('HGB-emb+meta', hgb, True)
