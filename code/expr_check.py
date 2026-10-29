"""检查候选文本里的计算表达式格式。"""
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from common import read_jsonl

bank = read_jsonl('data/candidates/gsm8k_train.jsonl')
# 统计各种表达式模式
pat_gsm = re.compile(r'<<([^>]+)>>')   # GSM8K <<expr=result>>
pat_eq = re.compile(r'(\d[\d,.\s*/+\-]*\d)\s*=\s*([\d,.\s]+)')  # a=b
pat_box = re.compile(r'\\boxed\{([^{}]+)\}')
pat_frac = re.compile(r'\\frac\{([^{}]+)\}\{([^{}]+)\}')

n_gsm = n_box = n_frac = 0
samples_gsm = []
for row in bank[:200]:
    for c in row['candidates']:
        t = c['text']
        g = pat_gsm.findall(t)
        b = pat_box.findall(t)
        f = pat_frac.findall(t)
        n_gsm += len(g); n_box += len(b); n_frac += len(f)
        if g and len(samples_gsm) < 5:
            samples_gsm.append(g[:3])

print(f'前200题×4候选里: <<>>={n_gsm}, boxed={n_box}, frac={n_frac}')
print('<<>> 样本:', samples_gsm[:5])

# 看一个完整候选
row = bank[0]
for c in row['candidates'][:2]:
    print('\n--- 候选', c['index'], '---')
    print(c['text'][:600])
