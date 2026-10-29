"""Tune the k-NN reasoning verifier: question similarity, k, pooling, and blend with expert A."""
import sys, io, pickle
from fractions import Fraction
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, eval_choice, fmt, test_like_metric

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
C = pickle.load(open('cache/oof_experts.pkl', 'rb'))
Eva, Ete = C['Eva'], C['Ete']


def is_int(a): return a is not None and Fraction(a).denominator == 1


def mask(p, d):
    p = p.copy()
    for i in range(len(p)):
        for c in range(4):
            if not is_int(d['answers'][i, c]): p[i, c] = -1e9
    return p


def z(p, d):
    p = mask(p, d)
    for i in range(len(p)):
        v = p[i][p[i] > -1e8]; m, s = v.mean(), v.std() + 1e-9
        p[i] = (p[i] - m) / s; p[i][p[i] < -1e8] = -1e9
    return p


Ava, Ate = z(Eva[1], va), z(Ete[1], te)


def resid(d):
    r = d['emb'] - d['emb'].mean(axis=1, keepdims=True)
    r = r.reshape(-1, 1024)
    return r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-9)


Rtr, Rva, Rte = resid(tr), resid(va), resid(te)
ytr = tr['labels']

# question representations
qv = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                     stop_words='english').fit([b['question'] for b in tr['bank']])
Qtr_tf = normalize(qv.transform([b['question'] for b in tr['bank']]))
Qva_tf = normalize(qv.transform([b['question'] for b in va['bank']]))
Qte_tf = normalize(qv.transform([b['question'] for b in te['bank']]))

# embedding question representation
qtr = tr['emb'].mean(1); qva = va['emb'].mean(1); qte = te['emb'].mean(1)
gmean = qtr.mean(0)
qtr = qtr - gmean; qva = qva - gmean; qte = qte - gmean
Qtr_em = qtr / (np.linalg.norm(qtr, axis=1, keepdims=True) + 1e-9)
Qva_em = qva / (np.linalg.norm(qva, axis=1, keepdims=True) + 1e-9)
Qte_em = qte / (np.linalg.norm(qte, axis=1, keepdims=True) + 1e-9)


def knn(Q, Qtr, R, d, k=40, power=1.0, pool='mean', kernel=False):
    S = Q @ Qtr.T
    S = S.toarray() if hasattr(S, 'toarray') else np.asarray(S)
    out = np.zeros((len(d['bank']), 4))
    for i in range(len(d['bank'])):
        idx = np.argsort(-S[i])[:k]
        w = np.maximum(S[i, idx], 0) ** power
        if w.sum() <= 0: w = np.ones_like(w)
        w = w / w.sum()
        posv, negv = [], []
        for j, wj in zip(idx, w):
            for c in range(4):
                if not is_int(tr['answers'][j, c]): continue
                v = Rtr[j * 4 + c]
                if ytr[j, c]: posv.append((v, wj))
                else: negv.append((v, wj))
        if not posv or not negv:
            continue
        if pool == 'mean':
            p = np.average([x for x, _ in posv], axis=0, weights=[w for _, w in posv])
            n = np.average([x for x, _ in negv], axis=0, weights=[w for _, w in negv])
        elif pool == 'max':
            p = max(posv, key=lambda t: np.dot(t[0], R[i * 4:(i + 1) * 4].T @ np.ones(4) / 4))[0] if False else None
            # use max similarity to any positive/negative
            sims = R[i * 4:(i + 1) * 4] @ np.stack([x for x, _ in posv], 1)
            p = np.stack([x for x, _ in posv], 0)[np.argmax(sims.max(1))]
            sims2 = R[i * 4:(i + 1) * 4] @ np.stack([x for x, _ in negv], 1)
            n = np.stack([x for x, _ in negv], 0)[np.argmax(sims2.max(1))]
        p = p / (np.linalg.norm(p) + 1e-9); n = n / (np.linalg.norm(n) + 1e-9)
        if kernel:
            # weighted softmax over individual candidate similarities
            P = np.stack([x for x, _ in posv], 0); Wp = np.array([w for _, w in posv])
            N = np.stack([x for x, _ in negv], 0); Wn = np.array([w for _, w in negv])
            sp = R[i * 4:(i + 1) * 4] @ P.T; sn = R[i * 4:(i + 1) * 4] @ N.T
            out[i] = (sp * Wp).sum(1) / Wp.sum() - (sn * Wn).sum(1) / Wn.sum()
        else:
            out[i] = R[i * 4:(i + 1) * 4] @ (p - n)
    return out


def ev(sv, st, name, alpha=0.1):
    kva, kte = z(sv, va), z(st, te)
    rv0 = eval_choice(kva.argmax(1), va); rt0 = eval_choice(kte.argmax(1), te)
    pv = (1 - alpha) * Ava + alpha * kva; pt = (1 - alpha) * Ate + alpha * kte
    rv = eval_choice(pv.argmax(1), va); rt = eval_choice(pt.argmax(1), te)
    print(f'{name:38s} knn: val={rv0["acc"]:5.2f} test={rt0["acc"]:5.2f} || '
          f'blend a={alpha}: val={rv["acc"]:5.2f} tl={test_like_metric(rv):5.2f} test={rt["acc"]:5.2f} '
          f'#1={rt["acc1"]:.1f} #2={rt.get("acc2",0):.1f} #3={rt.get("acc3",0):.1f}', flush=True)
    return kva, kte


best = {}
for qname, Qtr_, Qva_, Qte_ in (('tf', Qtr_tf, Qva_tf, Qte_tf), ('emb', Qtr_em, Qva_em, Qte_em)):
    for k in (20, 40, 80):
        sv = knn(Qva_, Qtr_, Rva, va, k=k)
        st = knn(Qte_, Qtr_, Rte, te, k=k)
        best[(qname, k)] = ev(sv, st, f'{qname} k={k} mean')
        sv = knn(Qva_, Qtr_, Rva, va, k=k, kernel=True)
        st = knn(Qte_, Qtr_, Rte, te, k=k, kernel=True)
        best[(qname, k, 'ker')] = ev(sv, st, f'{qname} k={k} kernel')
