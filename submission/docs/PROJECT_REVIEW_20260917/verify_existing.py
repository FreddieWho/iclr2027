"""Read-only, offline artifact review. No provider calls or upstream writes."""
import csv
import hashlib
import json
import random
import sqlite3
import statistics
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('direct', 'staged2', 'rewrite')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':')).encode()).hexdigest()


def norm(x):
    return ' '.join(str(x).strip().casefold().split())


def main():
    out = {'scope': 'local existing artifacts; no API or upstream writes', 'issues': []}
    checks = []

    def check(label, actual, expected):
        checks.append({'check': label, 'pass': actual == expected})
        if actual != expected:
            out['issues'].append({'check': label, 'actual': actual, 'expected': expected})

    # Check every existing V4 reader manifest, including P0 calibration and P1.
    data_paths = list(ROOT.glob('work/v4*/**/data/*.jsonl'))
    data_by_sha = {sha(p): p for p in data_paths}
    phase_readers = []
    for manifest_path in sorted(ROOT.glob('work/v4*/**/reader_manifest.json')):
        m = json.loads(manifest_path.read_text())
        rr = rows(manifest_path.parent / 'reader_rows.jsonl')
        hh = rows(data_by_sha[m['histories_sha256']])
        qq = rows(data_by_sha[m['queries_sha256']])
        qmap = {q['question_id']: q for q in qq}
        label = str(manifest_path.relative_to(ROOT))
        check(label + ': reader hash', sha(manifest_path.parent / 'reader_rows.jsonl'), m['reader_rows_sha256'])
        expected = {(h['history_id'], q['question_id'], c) for h in hh
                    for q in qq if q['history_id'] == h['history_id'] for c in m['conditions']}
        keys = [(r['history_id'], r['question_id'], r['condition']) for r in rr]
        check(label + ': key coverage', set(keys) == expected and len(keys) == len(expected), True)
        score_errors = []
        for r in rr:
            q = qmap[r['question_id']]
            if r['status'] == 'valid':
                score = float(norm(r['answer']) in {norm(a) for a in q['answers']})
                if score != r['score']:
                    score_errors.append(r['score_id'])
            elif r['score'] is not None:
                score_errors.append(r['score_id'])
            if r['question_hash'] != digest(q['question']):
                score_errors.append(r['score_id'] + ': question hash')
        check(label + ': exact scores and question hashes', score_errors, [])
        per_condition = {}
        for c in m['conditions']:
            v = [r['score'] for r in rr if r['condition'] == c and r['status'] == 'valid']
            per_condition[c] = {'n_valid': len(v), 'mean': statistics.mean(v) if v else None}
        phase_readers.append({'manifest': label, 'rows': len(rr),
                             'status_counts': dict(Counter(r['status'] for r in rr)),
                             'per_condition': per_condition})
    out['reader_audits'] = phase_readers

    p2 = ROOT / 'work/v4_fast_decision_20260914_compact_v2/p2'
    memories = rows(p2 / 'live/memories.jsonl')
    cm = json.loads((p2 / 'live/compression_manifest.json').read_text())
    check('P2 memory file hash', sha(p2 / 'live/memories.jsonl'), cm['memory_rows_sha256'])
    check('P2 runtime hash', sha(ROOT / 'configs/v4_compact_v2_continuation_runtime_20260914.json'), cm['runtime_config_sha256'])
    check('P2 prompt hash', sha(ROOT / cm['compressor_prompt_path']), cm['compressor_prompt_sha256'])
    check('P2 memory keys', len({(m['history_id'], m['condition']) for m in memories}), 72)
    check('P2 memory body hashes', all(digest(m['text']) == m['memory_hash'] for m in memories), True)
    mmap = {(m['history_id'], m['condition']): m for m in memories}
    published = {(r['reader_slot'], r['history_id']): r for r in rows(ROOT / 'artifacts/v4_confirmation_rows_compact_v2_20260914.jsonl') if r['record_type'] == 'p2_history_utility'}
    out['p2_lengths'] = {c: {'natural_stop': sum(m['status'] == 'natural_stop' for m in memories if m['condition'] == c),
                               'median': statistics.median(m['visible_native_tokens'] for m in memories if m['condition'] == c and m['status'] == 'natural_stop')}
                         for c in ARMS}
    out['p2_target_overshoots'] = sum(m['status'] == 'natural_stop' and m['visible_native_tokens'] > 8192 for m in memories)
    comparisons = 0
    contrasts = []
    for reader in ('R1', 'R2'):
        rr = rows(p2 / (reader.lower() + '_low') / 'reader_rows.jsonl')
        groups = defaultdict(list)
        for r in rr:
            if r['condition'] in ARMS:
                groups[(r['history_id'], r['condition'])].append(r)
                check(f"{reader}/{r['score_id']}: memory hash", r['memory_hash'], mmap[(r['history_id'], r['condition'])]['memory_hash'])
        utilities = {}
        for (reader_id, hid), published_row in published.items():
            if reader_id != reader:
                continue
            uu = []
            for i, c in enumerate(ARMS):
                group = groups[(hid, c)]
                valid = [r for r in group if r['status'] == 'valid']
                u = sum(r['score'] for r in valid) / 8 if len(valid) == 8 else None
                uu.append(u)
                for field, actual in [('U', u), ('L_native', mmap[(hid, c)]['visible_native_tokens']), ('n_valid', len(valid))]:
                    check(f'{reader}/{hid}/{c}/{field}', actual, published_row[field][i])
                    comparisons += 1
            utilities[hid] = uu
        for mode in ('three_arm_complete', 'pair_specific'):
            for name, a, b in [('SD', 1, 0), ('RD', 2, 0), ('SR', 1, 2)]:
                vals = [u[a] - u[b] for u in utilities.values() if all(u[j] is not None for j in (range(3) if mode == 'three_arm_complete' else [a, b]))]
                contrasts.append({'reader': reader, 'mode': mode, 'contrast': name, 'n': len(vals), 'mean_pp': statistics.mean(vals) * 100})
    out['p2_history_fields_compared'] = comparisons
    out['contrasts'] = contrasts

    # Run the delivered reanalysis in a temporary matching tree, preserving originals.
    package = ROOT / 'docs/ICLR_MemoryV4_Independent_Reanalysis_20260914'
    with tempfile.TemporaryDirectory(prefix='memory-review-') as tmp:
        t = Path(tmp); dest = t / 'docs' / package.name; dest.mkdir(parents=True)
        for p in package.iterdir():
            if p.is_file(): (dest / p.name).write_bytes(p.read_bytes())
        (t / 'artifacts').mkdir()
        src = ROOT / 'artifacts/v4_confirmation_rows_compact_v2_20260914.jsonl'
        (t / 'artifacts' / src.name).write_bytes(src.read_bytes())
        run = subprocess.run(['python3', str(dest / 'recompute.py')], capture_output=True, text=True)
        check('reanalysis exit', run.returncode, 0)
        check('reanalysis output bytes', sha(dest / 'paired_reanalysis.csv'), sha(package / 'paired_reanalysis.csv'))
        out['reanalysis_stdout'] = run.stdout.strip()
        # Simulate copying the directory alone.
        (t / 'artifacts' / src.name).unlink()
        alone = subprocess.run(['python3', str(dest / 'recompute.py')], capture_output=True, text=True)
        out['standalone_directory'] = {'exit': alone.returncode, 'error': alone.stderr.splitlines()[-1] if alone.stderr else ''}

    db_path = ROOT / 'work/v4_fast_decision_20260913/cost_ledger.sqlite'
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as db:
        out['ledger'] = {'states': dict(db.execute('select state,count(*) from requests group by state')),
                         'total_accounted_usd': db.execute('select sum(accounted_usd) from requests').fetchone()[0],
                         'reserved_included_in_total_usd': db.execute("select sum(accounted_usd) from requests where state='reserved'").fetchone()[0],
                         'model_prior': list(db.execute('select model,usd from model_prior'))}
    checksum = {'ok': 0, 'mismatch': [], 'missing': []}
    for line in (ROOT / 'SHA256SUMS.txt').read_text().splitlines():
        h, f = line.split(maxsplit=1); p = ROOT / f.lstrip('*')
        if not p.exists(): checksum['missing'].append(f)
        elif sha(p) != h: checksum['mismatch'].append(f)
        else: checksum['ok'] += 1
    out['root_checksums'] = checksum
    with zipfile.ZipFile(package.with_suffix('.zip')) as z:
        out['reanalysis_zip'] = {'crc_error': z.testzip(), 'entries': z.namelist(),
                                 'different_from_directory': [n for n in z.namelist() if (ROOT / 'docs' / n).is_file() and z.read(n) != (ROOT / 'docs' / n).read_bytes()]}
    out['checks_count'] = len(checks)
    out['checks_failed'] = sum(not c['pass'] for c in checks)
    out['inputs'] = {str(p.relative_to(ROOT)): sha(p) for p in [
        ROOT / 'scripts/run_v4_fast_pilot.py', ROOT / 'scripts/core.py',
        ROOT / 'artifacts/v4_confirmation_rows_compact_v2_20260914.jsonl',
        ROOT / 'reports/V4_ROUTE_DECISION.json', package / 'recompute.py']}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
