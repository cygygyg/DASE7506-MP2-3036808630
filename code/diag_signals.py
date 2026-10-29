"""Diagnostics: candidate-level signal AUC / selection accuracy on hard (#1) questions."""
import sys, io, re
from collections import Counter
from fractions import Fraction
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect

def auc(score, label):
    """Rank AUC, higher score -> label 1."""
    s = np.asarray(score, float); y = np.asarray(label, int)
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    # average ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = y.sum(); n0 = len(y)-n1
    if n1 == 0 or n0 == 0: return float('nan')
    return (ranks[y==1].sum() - n1*(n1+1)/2) / (n1*n0)


def candidate_feats(data):
    emb = data['emb']; N = len(emb)
    res = emb - emb.mean(axis=1, keepdims=True)
    rn = np.linalg.norm(res, axis=2)
    # cosine of residual to mean direction of other candidates (outlierness)
    out = np.zeros((N,4)); pair = np.zeros((N,4)); qsim = np.zeros((N,4))
    rflat = res.reshape(-1, 1024)
    rnorm = np.linalg.norm(rflat, axis=1) + 1e-9
    for i in range(N):
        for c in range(4):
            others = np.delete(res[i], c, axis=0)
            m = others.mean(axis=0)
            out[i,c] = float(res[i,c] @ m / (rnorm[i*4+c]*np.linalg.norm(m)+1e-9))
            sims = []
            for d in range(4):
                if d==c: continue
                sims.append(float(res[i,c] @ res[i,d]/(rnorm[i*4+c]*rnorm[i*4+d]+1e-9)))
            pair[i,c] = np.mean(sims)
        q = emb[i].mean(axis=0)
        for c in range(4):
            qsim[i,c] = float(emb[i,c] @ q/(np.linalg.norm(emb[i,c])*np.linalg.norm(q)+1e-9))
    feats = {'res_norm': rn, 'outlier_cos': -out, 'pair_cos': -pair, 'qsim': qsim}
    # text/answer numeric
    val = np.zeros((N,4)); grp = np.zeros((N,4)); uniq = np.zeros((N,4)); plural = np.zeros((N,4))
    length = np.zeros((N,4)); neq = np.zeros((N,4)); lines = np.zeros((N,4))
    for i,row in enumerate(data['bank']):
        ans = data['answers'][i]
        cnt = Counter(a for a in ans if a is not None)
        G = len(cnt)
        for c,txt in enumerate([x['text'] for x in row['candidates']]):
            a = ans[c]
            if a is not None:
                f = Fraction(a); val[i,c] = f.numerator/f.denominator
                grp[i,c] = cnt[a]; uniq[i,c] = cnt[a]==1; plural[i,c] = cnt[a]==max(cnt.values())
            length[i,c] = len(txt); neq[i,c] = txt.count('='); lines[i,c] = txt.count('\n')
    feats['log_value'] = np.sign(val)*np.log1p(np.abs(val))
    feats['group_size'] = grp; feats['is_unique'] = uniq; feats['is_plural'] = plural
    feats['length'] = length; feats['n_equals'] = neq; feats['n_lines'] = lines
    return feats


for split in ('validation','test'):
    d = load_split(split); nc = ncorrect(d); y = d['labels']
    f = candidate_feats(d)
    print(f'===== {split} =====')
    for subset, mask in [('all', np.ones(len(nc),bool)), ('#1', nc==1), ('#2', nc==2), ('#3', nc==3)]:
        mm = np.repeat(mask,4); yy = y.reshape(-1)[mm]
        if yy.sum()==0 or (1-yy).sum()==0: continue
        print(f'-- {subset} (pos={yy.mean():.2f})')
        for k,v in f.items():
            a = auc(v.reshape(-1)[mm], yy)
            print(f'   {k:12s} AUC={a:.3f}', end='')
        print()
