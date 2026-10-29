"""文本长度等简单信号单独判别力。"""
import sys, io
import numpy as np
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import parse_answer, read_jsonl

for split in ('train', 'validation', 'test'):
    gold = read_jsonl(f'data/gsm8k_{split}.jsonl')
    bank = read_jsonl(f'data/candidates/gsm8k_{split}.jsonl')
    lens_corr = []; lens_err = []
    for g, row in zip(gold, bank):
        ga = parse_answer(g['answer'])
        for c in row['candidates']:
            a = parse_answer(c['text'])
            if a == ga:
                lens_corr.append(len(c['text']))
            else:
                lens_err.append(len(c['text']))
    print(f'{split}: 正确候选文本长度 mean={np.mean(lens_corr):.0f}, 错误候选 mean={np.mean(lens_err):.0f}, '
          f'正确/错误比={np.mean(lens_corr)/np.mean(lens_err):.2f}')
    # AUC of length
    y = np.array([1]*len(lens_corr) + [0]*len(lens_err))
    x = np.array(lens_corr + lens_err)
    # 正确候选是否更长
    auc = roc_auc_score(y, x)
    print(f'  长度 AUC(正确更长)={auc:.3f}')
