"""Does the support-vs-embedding threshold transfer from train OOF to val/test?"""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric
from mp2features import build

tr = load_split('train'); va = load_split('validation'); te = load_split('test')


def centred(d):
    return (d['emb'] - d['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)


def fit_base(X, y, C=1.0):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000, solver='lbfgs', random_state=0).fit(sc.transform(X), y)
    return sc, clf


Xtr = centred(tr); ytr = tr['labels'].reshape(-1)
sc_full, base_full = fit_base(Xtr, ytr)
oof = np.zeros((len(tr['bank']), 4))
for itr, iva in KFold(5, shuffle=True, random_state=0).split(np.arange(len(tr['bank']))):
    m = (np.asarray(itr)[:, None] * 4 + np.arange(4)).ravel()
    mv = (np.asarray(iva)[:, None] * 4 + np.arange(4)).ravel()
    sc, clf = fit_base(Xtr[m], ytr[m])
    oof[iva] = clf.predict_proba(sc.transform(Xtr[mv]))[:, 1].reshape(-1, 4)
for i in range(len(oof)):
    for c in range(4):
        if tr['answers'][i, c] is None: oof[i, c] = -1e9
pva = base_full.predict_proba(sc_full.transform(centred(va)))[:, 1].reshape(-1, 4)
pte = base_full.predict_proba(sc_full.transform(centred(te)))[:, 1].reshape(-1, 4)
for d, p in ((va, pva), (te, pte)):
    for i in range(len(p)):
        for c in range(4):
            if d['answers'][i, c] is None: p[i, c] = -1e9


def analyse(d, scores, name, mode='raw'):
    rows = {}
    for i in range(len(d['bank'])):
        ans = d['answers'][i]
        groups = {}
        for c, a in enumerate(ans):
            key = ('a', a) if a is not None else ('none', c)
            groups.setdefault(key, []).append(c)
        keys = list(groups)
        sizes = np.array([len(groups[k]) for k in keys])
        gs = np.array([max(scores[i, c] for c in groups[k]) for k in keys])
        labs = np.array([int(any(d['labels'][i, c] for c in groups[k])) for k in keys])
        rows[i] = (sizes, gs, labs, keys, groups)
    return rows


def apply_rule(d, rows, t31, t211, transform=None):
    N = len(d['bank']); choice = np.zeros(N, dtype=int)
    for i in range(N):
        sizes, gs, labs, keys, groups = rows[i]
        G = len(sizes)
        if G == 1:
            choice[i] = groups[keys[0]][0]; continue
        sc = gs.copy()
        if transform == 'z':
            sc = (gs - gs.mean()) / (gs.std() + 1e-9)
        elif transform == 'softmax':
            e = np.exp((gs - gs.max()) / 0.2); sc = e / e.sum()
        pattern = tuple(sorted(sizes.tolist(), reverse=True))
        imax = int(np.argmax(sizes))
        others = [g for g in range(G) if g != imax]
        if sizes[imax] > max(sizes[g] for g in others):
            t = t31 if sizes[imax] >= 3 else t211
            gap = sc[imax] - max(sc[g] for g in others)
            if gap > t:
                pick = imax
            else:
                pick = max(others, key=lambda g: sc[g])
        else:
            pick = int(np.argmax(sc))
        choice[i] = groups[keys[pick]][0]
    return eval_choice(choice, d), choice


for mode in ('raw', 'z', 'softmax'):
    rtr = analyse(tr, oof, 'train')
    rva = analyse(va, pva, 'val')
    rte = analyse(te, pte, 'test')
    print(f'===== transform={mode} =====')
    # grid search thresholds on train for each pattern size
    grid = np.linspace(-1.0, 1.0, 401) if mode != 'softmax' else np.linspace(0, 0.6, 301)
    best = {}
    for k, tname in ((3, 't31'), (2, 't211')):
        bestacc, bestt = -1, 0
        for t in grid:
            # evaluate rule on train with both thresholds set; only pattern-specific? use same t for simplicity
            pass
        # direct optimisation: for each pattern subset compute accuracy
        for t in grid:
            ok = 0; n = 0
            for i, (sizes, gs, labs, keys, groups) in rtr.items():
                G = len(sizes)
                if G == 1: continue
                imax = int(np.argmax(sizes)); others = [g for g in range(G) if g != imax]
                if not others or not (sizes[imax] > max(sizes[g] for g in others)): continue
                if sizes[imax] != k and not (k == 2 and sizes[imax] == 2): continue
                sc = gs.copy()
                if mode == 'z': sc = (gs-gs.mean())/(gs.std()+1e-9)
                elif mode == 'softmax':
                    e=np.exp((gs-gs.max())/0.2); sc=e/e.sum()
                gap = sc[imax] - max(sc[g] for g in others)
                pick = imax if gap > t else max(others, key=lambda g: sc[g])
                ok += labs[pick]; n += 1
            if n and ok/n > bestacc:
                bestacc, bestt = ok/n, t
        best[k] = bestt
        print(f'  train best t for largest size {k}: {bestt:.3f} acc={bestacc*100:.2f}%')
    t31 = best[3]; t211 = best[2]
    for name, d, rows in (('val', va, rva), ('test', te, rte)):
        r, ch = apply_rule(d, rows, t31, t211, mode)
        print(fmt(r, f'{name} rule') + (f' testlike={test_like_metric(r):5.2f}' if name == 'val' else ''))
