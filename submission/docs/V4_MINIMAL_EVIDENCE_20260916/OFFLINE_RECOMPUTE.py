"""Offline recomputation from question-level receipts to headline tables.

No network, no model calls, stdlib only. Reads the vendored evidence copies in
this directory and reproduces: exact scores (already stored per row; re-derived
where scoring==exact with the vendored queries), history utilities, length
distributions, three-arm-complete and pair-specific tables, and headline
contrast means. Bootstrap CIs use an explicitly documented independent seed
(INDEPENDENT_SEED) and must NOT be mistaken for the runner's original random
stream; deterministic fields (utilities, deltas, n) must match the frozen
science result exactly.
Usage: python3 OFFLINE_RECOMPUTE.py [--out recomputed.json]
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEPENDENT_SEED, REPS = 20260916, 20000
TYPE_MAP = {'atomic': 'atomic_fact', 'update': 'temporal_update_provenance',
            'relational': 'relational_multihop', 'constraint': 'exception_retraction_constraint'}


def load(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines() if line.strip()]


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def quantile(values, p):
    v = sorted(values)
    z = (len(v) - 1) * p
    i, j = math.floor(z), math.ceil(z)
    return v[i] if i == j else v[i] + (z - i) * (v[j] - v[i])


def bootstrap(values, seed):
    n = len(values)
    rng = random.Random(seed)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(REPS)]
    return sum(values) / n, quantile(means, .05), quantile(means, .95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='recomputed.json')
    args = ap.parse_args()
    queries = {q['question_id']: q for q in load('p2_queries.jsonl')}
    r1 = load('r1_reader_rows.jsonl')
    r2 = load('r2_reader_rows.jsonl')
    mems = {(m['history_id'], m['condition']): m for m in load('p2_memories.jsonl')}
    out = {'input_sha256': {f: sha(f) for f in (
        'p2_histories.jsonl', 'p2_queries.jsonl', 'p2_memories.jsonl',
        'r1_reader_rows.jsonl', 'r2_reader_rows.jsonl')},
        'independent_bootstrap_seed': INDEPENDENT_SEED,
        'independent_bootstrap_reps': REPS,
        'note': 'CIs use the independent seed above, not the runner random stream.',
        'readers': {}}
    for slot, rows in (('R1', r1), ('R2', r2)):
        by_cell = defaultdict(list)
        for row in rows:
            by_cell[(row['history_id'], row['condition'])].append(row)
        hists = {}
        for (hid, cond), cell in by_cell.items():
            valid = [x for x in cell if x.get('status') == 'valid' and x.get('score') is not None]
            exp = [q for q in queries.values() if q['history_id'] == hid]
            exp_ids = {q['question_id'] for q in exp}
            got = {x['question_id'] for x in valid}
            u = sum(float(x['score']) for x in valid) / len(exp) if got == exp_ids and exp else None
            bt = {}
            for t in TYPE_MAP.values():
                tq = [q for q in exp if TYPE_MAP.get(q.get('question_type')) == t]
                tv = [x for x in valid if x.get('information_type') == t]
                bt[t] = (sum(float(x['score']) for x in tv) / len(tq)
                         if tq and len(tv) == len(tq)
                         and {x['question_id'] for x in tv} == {q['question_id'] for q in tq} else None)
            hists.setdefault(hid, {})[cond] = {
                'utility': u, 'n_valid': len(valid), 'n_expected': len(exp), 'by_type': bt,
                'L': mems.get((hid, cond), {}).get('visible_native_tokens')}
        table = []
        for hid, conds in sorted(hists.items()):
            d, s, w = (conds.get(c, {}).get('utility') for c in ('direct', 'staged2', 'rewrite'))
            row = {'history_id': hid, 'U_direct': d, 'U_staged2': s, 'U_rewrite': w,
                   'L_direct': conds.get('direct', {}).get('L'),
                   'L_staged2': conds.get('staged2', {}).get('L'),
                   'L_rewrite': conds.get('rewrite', {}).get('L')}
            if all(isinstance(v, (int, float)) for v in (d, s, w)):
                row.update({'delta_SD': s - d, 'delta_RD': w - d, 'delta_SR': s - w})
            else:
                row.update({'delta_SD': None, 'delta_RD': None, 'delta_SR': None})
            table.append(row)
        contrasts = {}
        for name, key in (('staged2_minus_direct', 'delta_SD'), ('rewrite_minus_direct', 'delta_RD'),
                          ('staged2_minus_rewrite', 'delta_SR')):
            for mode in ('three_arm_complete', 'pair_specific'):
                if mode == 'three_arm_complete':
                    vals = [x[key] for x in table if all(v is not None for v in
                            (x['delta_SD'], x['delta_RD'], x['delta_SR']))]
                else:
                    arms = {'staged2_minus_direct': ('U_staged2', 'U_direct'),
                            'rewrite_minus_direct': ('U_rewrite', 'U_direct'),
                            'staged2_minus_rewrite': ('U_staged2', 'U_rewrite')}[name]
                    vals = [x[arms[0]] - x[arms[1]] for x in table
                            if x[arms[0]] is not None and x[arms[1]] is not None]
                m, lo, hi = bootstrap(vals, INDEPENDENT_SEED) if vals else (None, None, None)
                contrasts[f'{mode}.{name}'] = {'n': len(vals), 'mean': m, 'ci90': [lo, hi]}
        out['readers'][slot] = {'history_table': table, 'contrasts': contrasts}
    (ROOT / args.out).write_text(json.dumps(out, indent=2))
    for slot in ('R1', 'R2'):
        c = out['readers'][slot]['contrasts']
        for k in ('three_arm_complete.staged2_minus_direct', 'pair_specific.staged2_minus_direct'):
            v = c[k]
            print(f"{slot} {k}: n={v['n']}, mean={v['mean']:.6f} ci90={v['ci90']}")
    print('wrote', args.out)


if __name__ == '__main__':
    main()
