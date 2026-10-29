"""Train the final MP2 model and write a small inference checkpoint.

Run from code/:
    python train_final.py --output runs/final_bundle

The checkpoint contains only the linear ranker (answer-residualisation matrix,
standardiser, 1024-d coefficients) and the k-NN configuration.  The retrieval
component reads the supplied training split at inference time, so it adds no
new inference asset.  No test references are read here or at inference.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import load_inputs, parse_answer, read_jsonl
from submission import _candidate_matrix
from arith2 import arith2_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('runs/final_bundle'))
    ap.add_argument('--C', type=float, default=0.1)
    ap.add_argument('--alpha', type=float, default=0.18)
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--k', type=int, default=80)
    args = ap.parse_args()

    bank, arrays = load_inputs('train')
    emb = arrays['embeddings']
    centred, feats, valid, finals, qraw = _candidate_matrix(bank, emb)
    n = len(bank)
    r = centred.reshape(-1, 1024)
    a = feats.reshape(-1, 5)

    # remove answer-explained directions from the centred embedding
    z1 = np.concatenate([np.ones((n * 4, 1)), a], axis=1)
    m = z1.T @ z1 + np.eye(z1.shape[1])
    m[0, 0] -= 1.0
    b = np.linalg.solve(m, z1.T @ r)
    d = r - z1 @ b

    gold = read_jsonl('data/gsm8k_train.jsonl')
    y = np.zeros(n * 4, dtype=np.int8)
    for i, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        for c, cand in enumerate(row['candidates']):
            y[i * 4 + c] = int(parse_answer(cand['text']) == ga)

    scaler = StandardScaler().fit(d)
    clf = LogisticRegression(C=args.C, max_iter=4000, solver='lbfgs', random_state=0)
    clf.fit(scaler.transform(d), y)

    # small arithmetic-verification model (used only as a low-weight tie-breaker)
    from fractions import Fraction
    rows = []
    for i, row in enumerate(bank):
        for c, cand in enumerate(row['candidates']):
            a = finals[i][c]
            fv = int(Fraction(a)) if (a is not None and Fraction(a).denominator == 1) else None
            rows.append(list(arith2_features(cand['text'], fv).values()))
    ar = np.asarray(rows, dtype=np.float64)
    ar_scaler = StandardScaler().fit(ar)
    ar_clf = LogisticRegression(C=0.3, max_iter=3000, solver='lbfgs', random_state=0)
    ar_clf.fit(ar_scaler.transform(ar), y)

    # global mean of question means, used to centre retrieval question vectors
    qmean = emb.astype(np.float64).mean(axis=1).mean(axis=0)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / 'model.npz',
        B=b.astype(np.float32),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        coef=clf.coef_.astype(np.float32)[0],
        intercept=np.float32(clf.intercept_[0]),
        qmean=qmean.astype(np.float32),
        alpha=np.float32(args.alpha),
        beta=np.float32(args.beta),
        k=np.int32(args.k),
        C=np.float32(args.C),
        arith_mean=ar_scaler.mean_.astype(np.float32),
        arith_scale=ar_scaler.scale_.astype(np.float32),
        arith_coef=ar_clf.coef_.astype(np.float32)[0],
        arith_intercept=np.float32(ar_clf.intercept_[0]),
    )
    meta = {
        'method': 'answer-debiased centred-embedding ranker + kNN reasoning verifier',
        'train_questions': n,
        'train_candidates': int(n * 4),
        'train_pos_ratio': float(y.mean()),
        'embedding_dim': 1024,
        'answer_features': 5,
        'C': args.C,
        'alpha': args.alpha,
        'beta': args.beta,
        'knn_k': args.k,
        'knn_question_representation': 'mean of four candidate embeddings, train-global centred, L2 normalised',
        'knn_candidate_representation': 'per-question-centred embedding, L2 normalised',
        'uses_test_references': False,
        'uses_training_references_at_inference': True,
    }
    (out / 'config.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    size = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(json.dumps({'saved': str(out), 'asset_bytes': size, **meta}, indent=2))


if __name__ == '__main__':
    main()
