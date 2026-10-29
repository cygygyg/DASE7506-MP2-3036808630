"""Compare support-gap separability on train OOF vs val/test."""
import sys, io
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect
from mp2lib import auc

tr = load_split('train'); va = load_split('validation'); te = load_split('test')

def centred(d): return (d['emb']-d['emb'].mean(axis=1,keepdims=True)).reshape(-1,1024)
def fit_base(X,y):
    sc=StandardScaler().fit(X); clf=LogisticRegression(C=1.0,max_iter=5000).fit(sc.transform(X),y); return sc,clf

Xtr=centred(tr); ytr=tr['labels'].reshape(-1)
scf,bf=fit_base(Xtr,ytr)
oof=np.zeros((len(tr['bank']),4)); ins=np.zeros_like(oof)
for itr,iva in KFold(5,shuffle=True,random_state=0).split(np.arange(len(tr['bank']))):
    m=(np.asarray(itr)[:,None]*4+np.arange(4)).ravel(); mv=(np.asarray(iva)[:,None]*4+np.arange(4)).ravel()
    sc,clf=fit_base(Xtr[m],ytr[m])
    oof[iva]=clf.predict_proba(sc.transform(Xtr[mv]))[:,1].reshape(-1,4)
ins=bf.predict_proba(scf.transform(Xtr))[:,1].reshape(-1,4)
pva=bf.predict_proba(scf.transform(centred(va)))[:,1].reshape(-1,4)
pte=bf.predict_proba(scf.transform(centred(te)))[:,1].reshape(-1,4)
for d,p in ((tr,oof),(tr,ins),(va,pva),(te,pte)):
    for i in range(len(p)):
        for c in range(4):
            if d['answers'][i,c] is None: p[i,c]=-1e9

def stats(d,scores,name):
    print('=====',name)
    buckets=defaultdict(list)
    for i in range(len(d['bank'])):
        ans=d['answers'][i]; groups={}
        for c,a in enumerate(ans):
            key=('a',a) if a is not None else ('none',c); groups.setdefault(key,[]).append(c)
        keys=list(groups); sizes=np.array([len(groups[k]) for k in keys])
        gs=np.array([max(scores[i,c] for c in groups[k]) for k in keys])
        labs=np.array([int(any(d['labels'][i,c] for c in groups[k])) for k in keys])
        pat=tuple(sorted(sizes.tolist(),reverse=True))
        imax=int(np.argmax(sizes)); others=[g for g in range(len(keys)) if g!=imax]
        if not others or sizes[imax]<=max(sizes[g] for g in others): continue
        gap=gs[imax]-max(gs[g] for g in others)
        buckets[pat].append((gap,labs[imax]))
    for pat,items in sorted(buckets.items(),key=lambda kv:-len(kv[1])):
        if len(items)<10: continue
        g=np.array([x[0] for x in items]); y=np.array([x[1] for x in items])
        print(f'  {pat} n={len(items)} support={y.mean()*100:5.1f}% gapAUC={auc(g,y):.3f} '
              f'gap med(correct)={np.median(g[y==1]) if y.any() else 0:.2f} med(wrong)={np.median(g[y==0]) if (~y.astype(bool)).any() else 0:.2f}')
stats(tr,oof,'train OOF')
stats(tr,ins,'train in-sample')
stats(va,pva,'validation')
stats(te,pte,'test')
