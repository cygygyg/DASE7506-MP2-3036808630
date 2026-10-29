"""Train the MP2 selection model and save a minimal checkpoint.

Method: per-question centring of candidate embeddings (removing the shared
question component), then logistic regression on the centred 1024-d vectors.

Run from code/:  python train_model.py --output runs/model_bundle
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import parse_answer, read_jsonl, load_inputs


def build(split):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    _, arrays = load_inputs(split)
    emb = arrays['embeddings']  # (N,4,1024)
    return emb, bank, gold


def centre(emb):
    return emb - emb.mean(axis=1, keepdims=True)


def labels(bank, gold):
    y = np.zeros(len(bank) * 4, dtype=np.int8)
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        for ci, c in enumerate(row['candidates']):
            a = parse_answer(c['text'])
            y[qi * 4 + ci] = int(a is not None and a == ga)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('runs/model_bundle'))
    ap.add_argument('--C', type=float, default=1.0)
    args = ap.parse_args()

    emb_tr, bank_tr, gold_tr = build('train')
    Xtr = centre(emb_tr).reshape(-1, 1024)
    ytr = labels(bank_tr, gold_tr)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)

    clf = LogisticRegression(C=args.C, max_iter=5000, solver='lbfgs', random_state=0)
    clf.fit(Xtr_s, ytr)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / 'model.npz',
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        coef=clf.coef_.astype(np.float32)[0],
        intercept=clf.intercept_.astype(np.float32)[0],
        C=np.float32(args.C),
    )
    meta = {
        'method': 'centred-embedding-logreg',
        'train_samples': int(len(ytr)),
        'train_pos_ratio': float(ytr.mean()),
        'n_features': int(Xtr.shape[1]),
        'C': args.C,
    }
    (out / 'config.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')

    # report asset sizes
    size = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(json.dumps({'saved': str(out), 'asset_bytes': size, **meta}, indent=2))


if __name__ == '__main__':
    main()
