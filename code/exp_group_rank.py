"""Stage 1: group-level ranker on top of baseline embedding scores + handcrafted features."""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, group_index
from mp2features import build

tr = load_split('train'); va = load_split('validation'); te = load_split('test')
Xtr = (tr['emb'] - tr['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
scaler = StandardScaler().fit(Xtr)
base = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', random_state=0).fit(scaler.transform(Xtr), tr['labels'].reshape(-1))


def get_scores(d):
    X = (d['emb'] - d['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
    p = base.predict_proba(scaler.transform(X))[:, 1].reshape(-1, 4)
    for i in range(len(p)):
        for c in range(4):
            if d['answers'][i, c] is None:
                p[i, c] = -1e9
    return p


def make_rows(d, scores, feats, use_numeric=True):
    rows = []
    for qi, rec in enumerate(feats['records']):
        gids = rec['gids']; G = rec['G']
        if G < 2:
            continue
        sizes = rec['sizes']; valid = rec['valid']
        gs = []
        for gi in range(G):
            members = np.where(gids == gi)[0]
            s = scores[qi, members]
            gs.append((s.max(), s.mean(), s.min()))
        gs = np.array(gs)
        # best other group score
        for gi in range(G):
            if not valid[gi]:
                continue
            y = int(d['labels'][qi, gids == gi].max())
            other = np.delete(gs[:, 0], gi)
            best_other = other.max() if len(other) else -1e9
            sf = [gs[gi, 0], gs[gi, 1], gs[gi, 2], gs[gi, 0] - best_other,
                  (gs[gi, 0] - best_other) * sizes[gi],
                  gs[gi, 0] * sizes[gi], sizes[gi],
                  float(np.sum(gs[:, 0] > gs[gi, 0])), float(np.sum(gs[:, 0] < gs[gi, 0]))]
            x = np.concatenate([sf, feats['group'][qi][gi] if use_numeric else np.zeros(0)])
            rows.append((qi, gi, y, x))
    return rows


def fit_eval(C=1.0, use_numeric=True, weight_mode='question'):
    rows_tr = make_rows(tr, get_scores(tr), build('train', tr), use_numeric)
    X = np.array([r[3] for r in rows_tr], dtype=np.float64)
    y = np.array([r[2] for r in rows_tr])
    # per-question normalization weights
    byq = defaultdict(list)
    for k, r in enumerate(rows_tr):
        byq[r[0]].append(k)
    w = np.ones(len(rows_tr))
    for qi, ks in byq.items():
        pos = [k for k in ks if y[k] == 1]
        neg = [k for k in ks if y[k] == 0]
        for k in pos: w[k] = 1.0 / max(1, len(pos))
        for k in neg: w[k] = 1.0 / max(1, len(neg))
        if weight_mode == 'balanced':
            for k in pos: w[k] *= 0.5
            for k in neg: w[k] *= 0.5
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000, solver='lbfgs', random_state=0).fit(sc.transform(X), y, sample_weight=w)
    out = {}
    for name, d in [('val', va), ('test', te)]:
        fe = build('validation' if name == 'val' else 'test', d)
        rows = make_rows(d, get_scores(d), fe, use_numeric)
        rows.sort(key=lambda r: (r[0], r[1]))
        # choose best group per question
        best = {}
        for qi, gi, yy, x in rows:
            s = clf.decision_function(sc.transform(x[None, :]))[0]
            if qi not in best or s > best[qi][0]:
                best[qi] = (s, gi)
        choice = np.zeros(len(d['bank']), dtype=int)
        for qi, (s, gi) in best.items():
            gids = fe['records'][qi]['gids']
            choice[qi] = int(np.where(gids == gi)[0][0])
        r = eval_choice(choice, d)
        out[name] = r
    return out


print('--- group ranker: baseline score features only ---')
for C in (0.03, 0.1, 0.3, 1.0):
    o = fit_eval(C, use_numeric=False)
    print(fmt(o['val'], f'C={C} val') + f' testlike={__import__("mp2lib").test_like_metric(o["val"]):5.2f} || ' + fmt(o['test'], 'test'))

print('--- + handcrafted group features ---')
for C in (0.01, 0.03, 0.1, 0.3, 1.0):
    o = fit_eval(C, use_numeric=True)
    print(fmt(o['val'], f'C={C} val') + f' testlike={__import__("mp2lib").test_like_metric(o["val"]):5.2f} || ' + fmt(o['test'], 'test'))
