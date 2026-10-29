"""Nonlinear text model on TF-IDF/SVD; evaluate alone and fused with embeddings."""
import sys, io, time
from fractions import Fraction
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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


def texts(d): return [c['text'] for row in d['bank'] for c in row['candidates']]
Ttr, Tva, Tte = texts(tr), texts(va), texts(te)
y = tr['labels'].reshape(-1)

vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                      strip_accents='unicode', lowercase=True)
Xtr = vec.fit_transform(Ttr); Xva = vec.transform(Tva); Xte = vec.transform(Tte)
svd = TruncatedSVD(n_components=256, random_state=0).fit(Xtr)
Str = svd.transform(Xtr); Sva = svd.transform(Xva); Ste = svd.transform(Xte)
sc = StandardScaler().fit(Str)
Str, Sva, Ste = sc.transform(Str), sc.transform(Sva), sc.transform(Ste)
print('SVD done', Str.shape)

models = {
    'mlp256': MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, learning_rate_init=1e-3,
                            max_iter=60, early_stopping=True, n_iter_no_change=8, random_state=0),
    'mlp64': MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, learning_rate_init=1e-3,
                           max_iter=80, early_stopping=True, n_iter_no_change=8, random_state=0),
    'hgb': HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05,
                                          min_samples_leaf=30, l2_regularization=1.0, random_state=0),
}
for name, clf in models.items():
    t0 = time.time()
    clf.fit(Str, y)
    pv = clf.predict_proba(Sva)[:, 1]; pt = clf.predict_proba(Ste)[:, 1]
    rv = eval_choice(mask_rows(pv, va).argmax(1), va); rt = eval_choice(mask_rows(pt, te).argmax(1), te)
    print(fmt(rv, f'{name} val') + f' tl={test_like_metric(rv):5.2f} || ' + fmt(rt, 'TEST') + f' [{time.time()-t0:.0f}s]', flush=True)
    np.save(f'cache/text_{name}_va.npy', pv); np.save(f'cache/text_{name}_te.npy', pt)
