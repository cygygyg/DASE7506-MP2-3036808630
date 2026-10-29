"""MP2 selection method: centred-embedding logistic regression.

predict(bank, arrays, checkpoint) returns one integer in 0..3 per question.

The model is trained by train_model.py.  Inference is pure NumPy: it centres
each question's four candidate embeddings (subtracting the shared question
component), applies a frozen standardisation + linear classifier, and picks the
highest-scoring candidate.  Candidates whose final numeric answer cannot be
parsed (parse_answer -> None) can never be correct, so they are forced to -inf.
"""
from __future__ import annotations

import re
from fractions import Fraction

import numpy as np


# --- Local copy of the fixed answer parser (identical to common.parse_answer) ---
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


def _majority_fallback(bank):
    """Training-free fallback used only when no checkpoint is supplied."""
    from collections import Counter
    out = []
    for row in bank:
        ans = [parse_answer(c['text']) for c in row['candidates']]
        cnt = Counter(a for a in ans if a is not None)
        if not cnt:
            out.append(0)
            continue
        best = cnt.most_common(1)[0][0]
        out.append(next(i for i, a in enumerate(ans) if a == best))
    return out


def predict(bank, arrays, checkpoint):
    emb = arrays['embeddings']  # (N, 4, 1024) float32

    if checkpoint is None:
        return _majority_fallback(bank)

    ckpt = np.load(str(checkpoint) + '/model.npz')
    mean = ckpt['scaler_mean'].astype(np.float64)
    scale = ckpt['scaler_scale'].astype(np.float64)
    coef = ckpt['coef'].astype(np.float64)
    intercept = float(ckpt['intercept'])

    centred = emb - emb.mean(axis=1, keepdims=True)
    X = centred.reshape(-1, emb.shape[-1]).astype(np.float64)
    X = (X - mean) / scale
    logit = X @ coef + intercept
    scores = logit.reshape(len(bank), 4)

    for qi, row in enumerate(bank):
        for ci, c in enumerate(row['candidates']):
            a = parse_answer(c['text'])
            if a is None:
                scores[qi, ci] = -np.inf
                continue
            # GSM8K reference answers are all integers: a non-integer final
            # answer can never equal the reference, so it is excluded.
            if Fraction(a).denominator != 1:
                scores[qi, ci] = -np.inf

    return [int(np.argmax(scores[qi])) for qi in range(len(bank))]
