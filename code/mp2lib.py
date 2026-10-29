"""Shared data loading / evaluation helpers for MP2 experiments.

Caches parsed splits as pickle for fast iteration.
"""
from __future__ import annotations
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from common import parse_answer, read_jsonl, load_inputs

CACHE = Path('cache')
CACHE.mkdir(exist_ok=True)


def load_split(split):
    """Return dict with emb (N,4,1024), bank, gold, answers (N,4) object, labels (N,4)."""
    path = CACHE / f'{split}.pkl'
    if path.exists():
        with path.open('rb') as f:
            return pickle.load(f)
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    _, arrays = load_inputs(split)
    emb = arrays['embeddings'].astype(np.float32)
    N = len(bank)
    answers = np.empty((N, 4), dtype=object)
    labels = np.zeros((N, 4), dtype=np.int8)
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        for ci, c in enumerate(row['candidates']):
            a = parse_answer(c['text'])
            answers[qi, ci] = a
            labels[qi, ci] = int(a is not None and a == ga)
    data = dict(bank=bank, gold=gold, emb=emb, answers=answers, labels=labels,
                gold_answers=np.array([parse_answer(g['answer']) for g in gold], dtype=object))
    with path.open('wb') as f:
        pickle.dump(data, f, protocol=4)
    return data


def group_index(answers_row):
    """Map each candidate to a group id (first occurrence order); None -> own singleton."""
    groups = {}
    out = []
    for a in answers_row:
        key = ('a', a) if a is not None else ('none', len(out))
        if key not in groups:
            groups[key] = len(groups)
        out.append(groups[key])
    return np.array(out), len(groups)


def ncorrect(data):
    return data['labels'].sum(axis=1)


def eval_choice(choice, data, ncorr=None):
    """choice: (N,) candidate index. Returns dict accuracy + breakdown."""
    labels = data['labels']
    N = len(choice)
    ok = labels[np.arange(N), choice]
    if ncorr is None:
        ncorr = ncorrect(data)
    out = {'acc': 100 * ok.mean(), 'correct': int(ok.sum()), 'n': N}
    for k in sorted(set(ncorr.tolist())):
        m = ncorr == k
        out[f'acc{k}'] = 100 * ok[m].mean()
        out[f'n{k}'] = int(m.sum())
    return out


def test_like_metric(res):
    """Reweight validation by the test split's #correct composition (151/145/171)."""
    w = {1: 151 / 467, 2: 145 / 467, 3: 171 / 467}
    return sum(w[k] * res[f'acc{k}'] for k in (1, 2, 3) if f'acc{k}' in res)


def fmt(res, name=''):
    s = f'{name:34s} acc={res["acc"]:6.2f}'
    if 'acc1' in res:
        s += f' | #1={res["acc1"]:5.1f} #2={res.get("acc2", 0):5.1f} #3={res.get("acc3", 0):5.1f}'
    if 'acc4' in res:
        s += f' #4={res["acc4"]:5.1f}'
    return s


def auc(score, label):
    """Rank AUC, higher score -> label 1, ties averaged."""
    s = np.asarray(score, float); y = np.asarray(label, int)
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def group_scores_to_choice(group_score, group_ids):
    """group_score: (N,G) or list of arrays; pick candidate in best group (first member)."""
    N = len(group_ids)
    choice = np.zeros(N, dtype=np.int64)
    for qi in range(N):
        gids = group_ids[qi]
        sc = group_score[qi]
        best = int(np.argmax(sc))
        choice[qi] = int(np.where(gids == best)[0][0])
    return choice
