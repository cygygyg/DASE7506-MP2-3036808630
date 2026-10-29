"""Final MP2 method: answer-debiased embedding ranker + k-NN reasoning verifier.

predict(bank, arrays, checkpoint) -> one candidate index in 0..3 per question.

The method has two frozen components:

1. A linear ranker on per-question-centred candidate embeddings, after
   removing directions explained by the candidate's numeric answer.  This
   makes the ranker rely on reasoning rather than answer-magnitude shortcuts.
2. A k-NN verifier over the supplied *training* split.  For a new question we
   retrieve semantically similar training questions, build a prototype of
   correct and incorrect reasoning residuals, and score the new candidates by
   their cosine similarity to those prototypes.

The two score vectors are standardised within each question and blended.
Only training references are used; test references are never opened.
"""
from __future__ import annotations

import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

# the runner already inserts the method directory, but keep direct imports robust
sys.path.insert(0, str(Path(__file__).resolve().parent))


# --------------------------------------------------------------------------- #
# Fixed numeric parser (identical to common.parse_answer)
# --------------------------------------------------------------------------- #
def parse_answer(text):
    text = str(text)
    tail = text.rsplit('####', 1)[1] if '####' in text else None
    if tail is None:
        boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
        tail = boxed[-1] if boxed else ''
    match = re.search(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', tail)
    if not match:
        return None
    try:
        return str(Fraction(match.group().replace(',', '')))
    except (ValueError, ZeroDivisionError):
        return None


def _is_int_answer(a):
    return a is not None and Fraction(a).denominator == 1


def _answer_features(final, vals):
    """Five numeric-answer features, matching the training script.

    Invalid (unparseable or non-integer) answers are all zeros.  Reference
    answers in this benchmark are integers, so a fractional final answer can
    never be correct.
    """
    if final is None:
        return np.zeros(5, dtype=np.float64)
    a = abs(final)
    v_log = np.copysign(np.log1p(a), final) if final != 0 else 0.0
    v_rank = sum(1 for v in vals if v < final) / max(1, len(vals) - 1)
    mx = max((abs(v) for v in vals), default=1.0) or 1.0
    v_rel_max = a / mx
    v_int = 1.0
    v_digits = float(len(str(int(round(a)))))
    return np.array([v_log, v_rank, v_rel_max, v_digits, v_int], dtype=np.float64)


def _question_answer_values(answers_row):
    vals = []
    for a in answers_row:
        if _is_int_answer(a):
            vals.append(float(Fraction(a)))
    return vals


def _candidate_matrix(bank, emb):
    """Return centred embeddings, answer features, validity mask, raw question means."""
    n = len(bank)
    emb = np.asarray(emb, dtype=np.float64)
    qraw = emb.mean(axis=1)
    centred = emb - qraw[:, None, :]
    feats = np.zeros((n, 4, 5), dtype=np.float64)
    valid = np.zeros((n, 4), dtype=bool)
    finals = np.empty((n, 4), dtype=object)
    for i, row in enumerate(bank):
        answers = [parse_answer(c['text']) for c in row['candidates']]
        finals[i] = answers
        vals = _question_answer_values(answers)
        for c, a in enumerate(answers):
            if _is_int_answer(a):
                valid[i, c] = True
                feats[i, c] = _answer_features(float(Fraction(a)), vals)
    return centred, feats, valid, finals, qraw


# --------------------------------------------------------------------------- #
# Training-split retrieval model (cached in-process)
# --------------------------------------------------------------------------- #
_TRAIN_CACHE = {}


def _find_train_data():
    here = Path(__file__).resolve().parent
    candidates = [here / 'data', Path.cwd() / 'data']
    for d in candidates:
        if (d / 'candidates' / 'gsm8k_train.npz').exists() and (d / 'gsm8k_train.jsonl').exists():
            return d
    return None


def _load_train_bank(data_dir):
    key = str(data_dir)
    if key in _TRAIN_CACHE:
        return _TRAIN_CACHE[key]
    from common import read_jsonl
    bank = read_jsonl(data_dir / 'candidates' / 'gsm8k_train.jsonl')
    gold = read_jsonl(data_dir / 'gsm8k_train.jsonl')
    with np.load(data_dir / 'candidates' / 'gsm8k_train.npz', allow_pickle=False) as f:
        emb = f['embeddings'].copy().reshape(-1, 4, 1024)
    emb = emb.astype(np.float64)
    n = len(bank)
    resid = emb - emb.mean(axis=1, keepdims=True)
    rnorm = np.linalg.norm(resid.reshape(-1, 1024), axis=1, keepdims=True) + 1e-9
    rflat = (resid.reshape(-1, 1024) / rnorm)
    qmean = emb.mean(axis=1)
    labels = np.zeros((n, 4), dtype=bool)
    for i, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        for c, cand in enumerate(row['candidates']):
            labels[i, c] = parse_answer(cand['text']) == ga
    _TRAIN_CACHE[key] = dict(rflat=rflat, qmean=qmean, labels=labels)
    return _TRAIN_CACHE[key]


def _knn_scores(bank, centred, qraw, valid, train, qmean_global, k=80):
    """Cosine-prototype k-NN score for each candidate."""
    n = len(bank)
    q = qraw - qmean_global
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    qtr = train['qmean'] - qmean_global
    qtr = qtr / (np.linalg.norm(qtr, axis=1, keepdims=True) + 1e-9)
    sim = q @ qtr.T  # (n, n_train)

    rflat = centred.reshape(-1, 1024)
    rflat = rflat / (np.linalg.norm(rflat, axis=1, keepdims=True) + 1e-9)
    rtr = train['rflat']
    labels = train['labels']
    pos_idx = np.where(labels.reshape(-1))[0]
    neg_idx = np.where(~labels.reshape(-1))[0]
    pos_set = set(pos_idx.tolist())
    neg_set = set(neg_idx.tolist())

    scores = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        idx = np.argsort(-sim[i])[:k]
        w = np.maximum(sim[i, idx], 0.0)
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()
        p_ids, p_w, n_ids, n_w = [], [], [], []
        for j, wj in zip(idx, w):
            for c in range(4):
                flat = int(j * 4 + c)
                if flat in pos_set:
                    p_ids.append(flat); p_w.append(wj)
                elif flat in neg_set:
                    n_ids.append(flat); n_w.append(wj)
        if not p_ids or not n_ids:
            continue
        sp = rflat[i * 4:(i + 1) * 4] @ rtr[p_ids].T
        sn = rflat[i * 4:(i + 1) * 4] @ rtr[n_ids].T
        p_w = np.asarray(p_w); n_w = np.asarray(n_w)
        scores[i] = (sp * p_w).sum(axis=1) / p_w.sum() - (sn * n_w).sum(axis=1) / n_w.sum()
    return scores


# --------------------------------------------------------------------------- #
# Post-processing: mask invalid answers, per-question standardisation, blend
# --------------------------------------------------------------------------- #
def _mask_and_z(scores, valid):
    out = scores.copy()
    for i in range(scores.shape[0]):
        vi = valid[i]
        if not vi.any():
            out[i] = 0.0
            continue
        v = out[i][vi]
        m, s = v.mean(), v.std() + 1e-9
        out[i] = (out[i] - m) / s
        out[i][~vi] = -1e9
    return out


def _linear_scores(centred, feats, ckpt):
    n = len(centred)
    z1 = np.concatenate([np.ones((n * 4, 1)), feats.reshape(-1, 5)], axis=1)
    d = centred.reshape(-1, 1024) - z1 @ ckpt['B']
    d = (d - ckpt['scaler_mean']) / ckpt['scaler_scale']
    return (d @ ckpt['coef'] + ckpt['intercept']).reshape(n, 4)


def _arith_scores(bank, valid, ckpt):
    """Low-weight local-arithmetic verification score."""
    if 'arith_coef' not in ckpt:
        return None
    from fractions import Fraction
    from arith2 import arith2_features
    n = len(bank)
    rows = []
    for i, row in enumerate(bank):
        for c, cand in enumerate(row['candidates']):
            a = parse_answer(cand['text'])
            fv = int(Fraction(a)) if (a is not None and Fraction(a).denominator == 1) else None
            rows.append(list(arith2_features(cand['text'], fv).values()))
    ar = np.asarray(rows, dtype=np.float64)
    ar = (ar - ckpt['arith_mean']) / ckpt['arith_scale']
    return (ar @ ckpt['arith_coef'] + ckpt['arith_intercept']).reshape(n, 4)


def predict(bank, arrays, checkpoint):
    emb = arrays['embeddings']
    centred, feats, valid, _, qraw = _candidate_matrix(bank, emb)
    n = len(bank)

    # training-free fallback if no checkpoint is available
    if checkpoint is None:
        from collections import Counter
        out = []
        for row in bank:
            ans = [parse_answer(c['text']) for c in row['candidates']]
            cnt = Counter(a for a in ans if a is not None)
            if not cnt:
                out.append(0)
            else:
                best = cnt.most_common(1)[0][0]
                out.append(next(i for i, a in enumerate(ans) if a == best))
        return out

    path = Path(checkpoint)
    if path.is_dir():
        path = path / 'model.npz'
    with np.load(path, allow_pickle=False) as f:
        ckpt = {k: f[k] for k in f.files}
    alpha = float(ckpt.get('alpha', 0.18))
    beta = float(ckpt.get('beta', 0.0))
    k = int(ckpt.get('k', 80))

    linear = _linear_scores(centred, feats, ckpt)
    linear = _mask_and_z(linear, valid)

    arith = _arith_scores(bank, valid, ckpt) if beta > 0 else None
    if arith is not None:
        arith = _mask_and_z(arith, valid)

    knn = None
    data_dir = _find_train_data()
    if data_dir is not None:
        train = _load_train_bank(data_dir)
        knn = _knn_scores(bank, centred, qraw, valid, train, ckpt['qmean'], k=k)
        knn = _mask_and_z(knn, valid)

    if knn is not None:
        final_scores = (1.0 - alpha - beta) * linear + alpha * knn
        if arith is not None:
            final_scores = final_scores + beta * arith
    elif arith is not None:
        final_scores = (1.0 - beta) * linear + beta * arith
    else:
        final_scores = linear

    choices = []
    for i in range(n):
        if valid[i].any():
            choices.append(int(np.argmax(final_scores[i])))
        else:
            # no integer-answer candidate: choose the first parseable one
            fallback = next((j for j, c in enumerate(bank[i]['candidates'])
                             if parse_answer(c['text']) is not None), 0)
            choices.append(int(fallback))
    return choices
