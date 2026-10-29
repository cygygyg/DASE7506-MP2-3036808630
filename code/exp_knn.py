"""k-NN reasoning verifier: similar training questions' correct/wrong residuals."""
import sys, io
from fractions import Fraction
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, eval_choice, fmt, test_like_metric

tr = load_split('train'); va = load_split('validation'); te = load_split('test')


def is_int(a): return a is not None and Fraction(a).denominator == 1
def mask_rows(p, d):
    p = p.reshape(-1, 4).copy()
    for i in range(len(p)):
        for c in range(4):
            if not is_int(d['answers'][i, c]): p[i, c] = -1e9
    return p


def resid(d):
    r = d['emb'] - d['emb'].mean(axis=1, keepdims=True)
    r = r.reshape(-1, 1024)
    r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-9)
    return r


qv = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                     stop_words='english').fit([b['question'] for b in tr['bank']])
Qtr = normalize(qv.transform([b['question'] for b in tr['bank']]))
Qva = normalize(qv.transform([b['question'] for b in va['bank']]))
Qte = normalize(qv.transform([b['question'] for b in te['bank']]))
Rtr = resid(tr); Rva = resid(va); Rte = resid(te)
ytr = tr['labels']


def knn_scores(Q, R, d, k=20, power=3.0):
    S = (Q @ Qtr.T).toarray()  # (N, Ntr)
    out = np.zeros((len(d['bank']), 4))
    for i in range(len(d['bank'])):
        idx = np.argsort(-S[i])[:k]
        w = S[i, idx] ** power
        w = w / (w.sum() + 1e-12)
        pos = []; neg = []
        for j, wj in zip(idx, w):
            for c in range(4):
                if not is_int(tr['answers'][j, c]): continue
                if ytr[j, c]: pos.append((Rtr[j * 4 + c], wj))
                else: neg.append((Rtr[j * 4 + c], wj))
        if not pos or not neg:
            continue
        p = np.average([x for x, _ in pos], axis=0, weights=[w for _, w in pos])
        n = np.average([x for x, _ in neg], axis=0, weights=[w for _, w in neg])
        p = p / (np.linalg.norm(p) + 1e-9); n = n / (np.linalg.norm(n) + 1e-9)
        out[i] = R[i * 4:(i + 1) * 4] @ (p - n)
    return out


for k in (10, 20, 40):
    for power in (1.0, 3.0):
        sv = knn_scores(Qva, Rva, va, k, power)
        st = knn_scores(Qte, Rte, te, k, power)
        rv = eval_choice(mask_rows(sv, va).argmax(1), va)
        rt = eval_choice(mask_rows(st, te).argmax(1), te)
        print(f'k={k} p={power} ' + fmt(rv, 'val knn') + f' tl={test_like_metric(rv):5.2f} || ' + fmt(rt, 'TEST'), flush=True)
        np.save(f'cache/knn_va_{k}_{power}.npy', sv); np.save(f'cache/knn_te_{k}_{power}.npy', st)
