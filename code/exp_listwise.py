"""Joint group-listwise ranker.

Group candidates by final numeric answer, score each group with
  S_g = max_{i in g} (w . z_i + b) + v . h_g
and train with a listwise softmax loss over groups plus L2 toward the old
centred-embedding logistic model.  lambda is chosen by 3-fold CV on train.
"""
import sys, io, json
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, test_like_metric
from mp2features import build


def is_int(a):
    return a is not None and Fraction(a).denominator == 1


def groups_of(d, i):
    """Valid integer-answer groups; each unparseable/non-integer candidate is its own group."""
    ans = d['answers'][i]; groups = {}
    for c, a in enumerate(ans):
        if not is_int(a):
            groups[('bad', c)] = [c]
        else:
            groups.setdefault(('a', a), []).append(c)
    keys = list(groups)
    gids = np.zeros(4, dtype=int)
    for gi, k in enumerate(keys):
        for c in groups[k]:
            gids[c] = gi
    sizes = np.array([len(groups[k]) for k in keys])
    valid = np.array([k[0] == 'a' for k in keys])
    return keys, groups, gids, sizes, valid


def dataset(d, feats):
    """Return list of questions: dict(z (4,1024), h (G,F), gids, valid, labels)."""
    N = len(d['bank'])
    Z = (d['emb'] - d['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
    out = []
    for i in range(N):
        keys, groups, gids, sizes, valid = groups_of(d, i)
        # only valid integer groups participate; bad groups get -inf and are ignored
        H = feats['group'][i]
        out.append(dict(i=i, z=Z[i * 4:(i + 1) * 4], gids=gids, sizes=sizes, valid=valid,
                        labels=np.array([int(any(d['labels'][i, c] for c in groups[k])) for k in keys]),
                        keys=keys, groups=groups, h=H))
    return out


def pack(qs, mean_z, scale_z, mean_h, scale_h):
    Z = np.zeros((len(qs), 4, 1024), np.float32)
    for k, q in enumerate(qs):
        Z[k] = (q['z'] - mean_z) / scale_z
    Hs = []
    for q in qs:
        Hs.append((q['h'] - mean_h) / scale_h)
    return Z, Hs


def group_scores(theta, Z, Hs, nz, nh):
    w = theta[:nz]; b = theta[nz]; v = theta[nz + 1:nz + 1 + nh]
    S = []
    arg = []
    for k in range(len(Z)):
        s = Z[k] @ w + b
        g = Hs[k] @ v
        gids = Z_gids[k]; valid = Z_valid[k]
        G = len(valid)
        row = np.full(G, -1e9)
        am = np.zeros(G, dtype=int)
        for gi in range(G):
            if not valid[gi]:
                continue
            mem = np.where(gids == gi)[0]
            j = mem[np.argmax(s[mem])]
            am[gi] = j
            row[gi] = s[j] + g[gi]
        S.append(row); arg.append(am)
    return S, arg


def objective(theta, Z, Hs, Z_gids, Z_valid, labels, nz, nh, w0, lam, lam_v=1e-3):
    w = theta[:nz]; b = theta[nz]; v = theta[nz + 1:nz + 1 + nh]
    loss = 0.5 * lam * np.sum((w - w0) ** 2) + 0.5 * lam_v * np.sum(v ** 2)
    gw = lam * (w - w0); gv = lam_v * v; gb = 0.0
    for k in range(len(Z)):
        s = Z[k] @ w + b
        g = Hs[k] @ v
        gids = Z_gids[k]; valid = Z_valid[k]; y = labels[k]
        G = len(valid)
        S = np.full(G, -1e9); am = np.zeros(G, dtype=int)
        for gi in range(G):
            if not valid[gi]:
                continue
            mem = np.where(gids == gi)[0]
            j = mem[np.argmax(s[mem])]
            am[gi] = j
            S[gi] = s[j] + g[gi]
        m = S.max()
        e = np.exp(S - m)
        Zs = e.sum()
        # positive groups (there is exactly one valid correct answer group)
        pos = np.where((y == 1) & valid)[0]
        if len(pos) == 0:
            continue
        Zp = e[pos].sum()
        loss += (m + np.log(Zs)) - (m + np.log(Zp))
        dS = e / Zs
        dS[pos] -= e[pos] / Zp
        for gi in range(G):
            if dS[gi] == 0 or not valid[gi]:
                continue
            gw += dS[gi] * Z[k, am[gi]]
            gb += dS[gi]
            gv += dS[gi] * Hs[k][gi]
    grad = np.concatenate([gw, [gb], gv])
    # normalise by number of questions
    return loss / len(Z), grad / len(Z)


def fit(train_qs, lam, w0, nz, nh, mean_z, scale_z, mean_h, scale_h, maxiter=300):
    Z, Hs = pack(train_qs, mean_z, scale_z, mean_h, scale_h)
    gids = [q['gids'] for q in train_qs]
    valid = [q['valid'] for q in train_qs]
    labels = [q['labels'] for q in train_qs]
    theta0 = np.concatenate([w0, [0.0], np.zeros(nh)])
    res = minimize(objective, theta0, args=(Z, Hs, gids, valid, labels, nz, nh, w0, lam),
                   method='L-BFGS-B', jac=True, options=dict(maxiter=maxiter, maxfun=maxiter * 2, ftol=1e-9, gtol=1e-6))
    return res.x, res


def predict(qs, theta, nz, nh, mean_z, scale_z, mean_h, scale_h):
    """Return one candidate index per question."""
    Z, Hs = pack(qs, mean_z, scale_z, mean_h, scale_h)
    w = theta[:nz]; b = theta[nz]; v = theta[nz + 1:nz + 1 + nh]
    out = np.zeros(len(qs), dtype=int)
    for k, q in enumerate(qs):
        s = Z[k] @ w + b
        g = Hs[k] @ v
        G = len(q['valid'])
        S = np.full(G, -1e9)
        for gi in range(G):
            if not q['valid'][gi]:
                continue
            mem = np.where(q['gids'] == gi)[0]
            S[gi] = s[mem].max() + g[gi]
        gi = int(np.argmax(S))
        out[k] = q['groups'][q['keys'][gi]][0]
    return out


if __name__ == '__main__':
    tr = load_split('train'); va = load_split('validation'); te = load_split('test')
    ftr = build('train', tr); fva = build('validation', va); fte = build('test', te)
    qtr = dataset(tr, ftr); qva = dataset(va, fva); qte = dataset(te, fte)

    # only questions with >=2 valid groups carry ranking signal
    train_qs = [q for q in qtr if q['valid'].sum() >= 2]
    print('train questions with >=2 groups:', len(train_qs))

    # standardisation statistics from train
    Zraw = np.concatenate([q['z'] for q in train_qs], axis=0)
    mean_z = Zraw.mean(0); scale_z = Zraw.std(0) + 1e-6
    Hraw = np.concatenate([q['h'] for q in train_qs], axis=0)
    mean_h = Hraw.mean(0); scale_h = Hraw.std(0) + 1e-6
    nz, nh = 1024, Hraw.shape[1]

    # old model on the same standardised view
    Xall = np.concatenate([q['z'] for q in train_qs], axis=0)
    yall = np.concatenate([tr['labels'][q['i']] for q in train_qs])
    Zstd = (Xall - mean_z) / scale_z
    old = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', random_state=0).fit(Zstd, yall)
    w0 = old.coef_[0].astype(np.float64)

    # 3-fold CV over training questions to choose lambda
    print('\n--- lambda CV (group selection accuracy) ---')
    for lam in (0.1, 0.3, 1.0, 3.0, 10.0):
        accs = []
        for itr, iva in KFold(3, shuffle=True, random_state=0).split(np.arange(len(train_qs))):
            sub = [train_qs[k] for k in itr]; valq = [train_qs[k] for k in iva]
            theta, res = fit(sub, lam, w0, nz, nh, mean_z, scale_z, mean_h, scale_h, maxiter=150)
            ch = predict(valq, theta, nz, nh, mean_z, scale_z, mean_h, scale_h)
            # labels for valq
            ok = 0
            for q, c in zip(valq, ch):
                ok += int(tr['labels'][q['i'], c])
            accs.append(ok / len(valq))
        print(f'  lambda={lam:5.2f}: CV acc={100*np.mean(accs):.2f}%')

    print('\n--- full train, evaluate ---')
    for lam in (0.3, 1.0, 3.0):
        theta, res = fit(train_qs, lam, w0, nz, nh, mean_z, scale_z, mean_h, scale_h, maxiter=400)
        print('lam', lam, 'success', res.success, 'nit', res.nit)
        for name, qs, d in (('val', qva, va), ('test', qte, te)):
            ch = predict(qs, theta, nz, nh, mean_z, scale_z, mean_h, scale_h)
            r = eval_choice(ch, d)
            print(fmt(r, f'{name} listwise lam={lam}') + (f' testlike={test_like_metric(r):5.2f}' if name == 'val' else ''))
