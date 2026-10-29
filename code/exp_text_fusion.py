"""Text TF-IDF fusion with centred embeddings (candidate level)."""
import sys, io
from fractions import Fraction
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric
from arith2 import arith2_features

tr = load_split('train'); va = load_split('validation'); te = load_split('test')


def centred(d): return (d['emb'] - d['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
def is_int(a): return a is not None and Fraction(a).denominator == 1
def texts(d): return [c['text'] for row in d['bank'] for c in row['candidates']]
def mask(d, p):
    p = p.reshape(-1, 4).copy()
    for i in range(len(p)):
        for c in range(4):
            if not is_int(d['answers'][i, c]): p[i, c] = -1e9
    return p


def ev(p, d, name):
    r = eval_choice(mask(d, p).argmax(1), d)
    print(fmt(r, name))
    return r


Xtr, Xva, Xte = centred(tr), centred(va), centred(te)
ytr = tr['labels'].reshape(-1)
Ttr, Tva, Tte = texts(tr), texts(va), texts(te)

print('--- TF-IDF alone ---')
for ngram in ((1, 1), (1, 2)):
    for mf in (3000, 10000):
        vec = TfidfVectorizer(max_features=mf, ngram_range=ngram, min_df=3, sublinear_tf=True,
                              strip_accents='unicode', lowercase=True)
        A = vec.fit_transform(Ttr); B = vec.transform(Tva); C = vec.transform(Tte)
        for Creg in (0.3, 1.0, 3.0):
            clf = LogisticRegression(C=Creg, max_iter=3000, solver='liblinear', random_state=0).fit(A, ytr)
            print(f'ngram={ngram} mf={mf} C={Creg}')
            ev(clf.predict_proba(B)[:, 1], va, '  val tfidf')
            ev(clf.predict_proba(C)[:, 1], te, '  test tfidf')

print('--- arith2 only ---')
def arith_mat(d):
    rows = []
    for i in range(len(d['bank'])):
        for c in range(4):
            a = d['answers'][i, c]
            fv = int(Fraction(a)) if is_int(a) else None
            rows.append(list(arith2_features(d['bank'][i]['candidates'][c]['text'], fv).values()))
    return np.array(rows, np.float32)
Atr, Ava, Ate = arith_mat(tr), arith_mat(va), arith_mat(te)
sc = StandardScaler().fit(Atr)
clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Atr), ytr)
ev(clf.predict_proba(sc.transform(Ava))[:, 1], va, 'val arith2-only')
ev(clf.predict_proba(sc.transform(Ate))[:, 1], te, 'test arith2-only')

print('--- embedding + arith2 ---')
Xtr2 = np.hstack([Xtr, Ava[:0] if False else Atr]); Xva2 = np.hstack([Xva, Ava]); Xte2 = np.hstack([Xte, Ate])
for C in (0.1, 0.3, 1.0):
    sc = StandardScaler().fit(Xtr2)
    clf = LogisticRegression(C=C, max_iter=5000, solver='lbfgs', random_state=0).fit(sc.transform(Xtr2), ytr)
    print('C', C)
    ev(clf.predict_proba(sc.transform(Xva2))[:, 1], va, '  val emb+arith2')
    ev(clf.predict_proba(sc.transform(Xte2))[:, 1], te, '  test emb+arith2')
