"""Show baseline errors on hard (#1/#2) validation questions with feature summaries."""
import sys, io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect
from mp2features import build

d = load_split('validation'); f = build('validation', d)
tr = load_split('train')
Xtr = (tr['emb'] - tr['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', random_state=0).fit(sc.transform(Xtr), tr['labels'].reshape(-1))
X = (d['emb'] - d['emb'].mean(axis=1, keepdims=True)).reshape(-1, 1024)
p = clf.predict_proba(sc.transform(X))[:, 1].reshape(-1, 4)
for i in range(len(p)):
    for c in range(4):
        if d['answers'][i, c] is None: p[i, c] = -1e9
nc = ncorrect(d)
shown = 0
for i in range(len(d['bank'])):
    if nc[i] != 1: continue
    ch = int(np.argmax(p[i]))
    if d['labels'][i, ch]: continue
    correct = int(np.where(d['labels'][i] == 1)[0][0])
    shown += 1
    if shown > 6: break
    rec = f['records'][i]
    print('=' * 110)
    print(f'Q{i}: {d["bank"][i]["question"]}')
    print(f'parsed answers: {list(d["answers"][i])}  sizes={rec["sizes"].tolist()}  scores={np.round(p[i],3).tolist()}')
    for c in [ch, correct]:
        fe = f['cand'][i, c]
        names = f['cand_names']
        def g(k): return fe[names.index(k)]
        print(f'--- cand {c} {"PICKED(wrong)" if c == ch else "CORRECT"} ans={d["answers"][i,c]} score={p[i,c]:.3f}')
        print(f'    eq={g("a_n_eq"):.0f} bad={g("a_n_bad"):.0f} last_ok={g("a_last_ok"):.0f} '
              f'final_eq_last={g("a_final_eq_last_rhs"):.0f} final_from_rhs={g("c_final_from_rhs"):.0f} '
              f'chain={g("c_chain_link_ratio"):.2f} last_support={g("c_last_operand_support"):.2f} '
              f'unknown={g("c_unknown_ratio"):.2f} len={g("t_len"):.0f} logv={g("v_log"):.2f}')
        txt = d['bank'][i]['candidates'][c]['text']
        print('    ' + txt[:700].replace('\n', ' | '))
