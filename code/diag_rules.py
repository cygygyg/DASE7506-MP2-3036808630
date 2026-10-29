"""Group-level rule diagnostics: support, uniqueness, and combinations."""
import sys, io
from collections import Counter
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from mp2lib import load_split, ncorrect, eval_choice, fmt, group_index


def group_info(data, i):
    ans = data['answers'][i]
    gids, G = group_index(ans)
    sizes = np.bincount(gids, minlength=G)
    # groups of unparseable candidates: each singleton, exclude from support rules
    valid = np.array([ans[gids == g][0] is not None for g in range(G)])
    return gids, G, sizes, valid


def choose_by(data, key):
    N = len(data['bank'])
    out = np.zeros(N, dtype=int)
    for i in range(N):
        gids, G, sizes, valid = group_info(data, i)
        sc = np.full(G, -1e9)
        for g in range(G):
            if not valid[g]:
                continue
            sc[g] = key(sizes, valid, g)
        best = int(np.argmax(sc))
        out[i] = int(np.where(gids == best)[0][0])
    return out


rules = {
    'largest_group': lambda s, v, g: s[g],
    'smallest_group': lambda s, v, g: -s[g],
    'largest_group_tie_first': lambda s, v, g: s[g] - 1e-3 * g,
    'unique_if_any_else_largest': None,
}

for split in ('validation', 'test'):
    d = load_split(split); nc = ncorrect(d)
    print(f'===== {split} =====')
    for name in ('largest_group', 'smallest_group'):
        ch = choose_by(d, rules[name])
        r = eval_choice(ch, d, nc)
        print(fmt(r, name))
    # unique singleton if exactly one, else largest
    N = len(d['bank']); out = np.zeros(N, dtype=int)
    for i in range(N):
        gids, G, sizes, valid = group_info(d, i)
        uniq = [g for g in range(G) if valid[g] and sizes[g] == 1]
        if len(uniq) == 1:
            best = uniq[0]
        else:
            sc = np.where(valid, sizes, -1)
            best = int(np.argmax(sc))
        out[i] = int(np.where(gids == best)[0][0])
    r = eval_choice(out, d, nc)
    print(fmt(r, 'unique_else_largest'))

    # oracle: choose largest, but if #3 use largest (same)
    # distribution of group size patterns per ncorrect
    for k in (1, 2, 3):
        sub = [i for i in range(N) if nc[i] == k]
        pats = Counter()
        for i in sub:
            gids, G, sizes, valid = group_info(d, i)
            pats[tuple(sorted(sizes[valid].tolist(), reverse=True))] += 1
        print(f'  #correct={k}: size patterns {pats.most_common(6)}')
    print()
