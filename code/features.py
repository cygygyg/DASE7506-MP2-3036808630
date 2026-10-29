"""完整特征工程 + 多模型对比。"""
import json, re
import numpy as np
from collections import Counter

from common import parse_answer, read_jsonl, load_inputs


def extract_features(split):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    _, arrays = load_inputs(split)
    emb = arrays['embeddings']  # (N,4,1024)
    N = len(bank)
    feats = []
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        texts = [c['text'] for c in row['candidates']]
        ans = [parse_answer(t) for t in texts]
        # 众数答案（多数投票）
        cnt = Counter(a for a in ans if a is not None)
        majority = cnt.most_common(1)[0][0] if cnt else None
        # 每个候选的特征
        for ci in range(4):
            t = texts[ci]
            a = ans[ci]
            f = {
                'is_parseable': float(a is not None),
                'ans_eq_majority': float(a is not None and a == majority),
                'ans_count': float(cnt.get(a, 0)) if a is not None else 0.0,
                'ans_is_unique': float(a is not None and cnt.get(a, 0) == 1),
                'text_len': float(len(t)),
                'num_digits': float(sum(ch.isdigit() for ch in t)),
                'num_equals': float(t.count('=')),
                'num_ops': float(sum(t.count(op) for op in '+-*/')),
                'num_lines': float(t.count('\n')),
                'has_marker': float(('####' in t) or ('boxed' in t)),
                'num_numbers': float(len(re.findall(r'\d+', t))),
            }
            feats.append(f)
    return emb, feats, bank, gold


def to_arrays(split):
    emb, feats, bank, gold = extract_features(split)
    N = len(bank)
    X_emb = emb.reshape(N*4, -1)
    keys = sorted(feats[0].keys())
    X_meta = np.array([[f[k] for k in keys] for f in feats], dtype=np.float32)
    y = np.zeros(N*4, dtype=np.int8)
    for qi, (g, row) in enumerate(zip(gold, bank)):
        ga = parse_answer(g['answer'])
        for ci, c in enumerate(row['candidates']):
            a = parse_answer(c['text'])
            y[qi*4+ci] = int(a is not None and a == ga)
    return X_emb, X_meta, y, bank, gold, keys


def selection_acc(scores, bank, gold):
    N = len(bank)
    correct = 0
    for qi in range(N):
        ga = parse_answer(gold[qi]['answer'])
        choice = int(np.argmax(scores[qi]))
        a = parse_answer(bank[qi]['candidates'][choice]['text'])
        correct += int(a is not None and a == ga)
    return correct, N


if __name__ == '__main__':
    X_emb_tr, X_meta_tr, y_tr, _, _, keys = to_arrays('train')
    X_emb_va, X_meta_va, y_va, bank_va, gold_va, _ = to_arrays('validation')
    print('meta feature keys:', keys)
    print('train/val shapes:', X_emb_tr.shape, X_emb_va.shape)
    print('train pos ratio: %.3f  val pos ratio: %.3f' % (y_tr.mean(), y_va.mean()))
