"""Offline reproductions against the current runner, using only fake providers."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_v4_fast_pilot as runner


class FakeTokenizer:
    def __init__(self, name):
        self.name = name

    def count(self, text):
        return len(text.split())


class FakeProvider:
    def __init__(self, config, *args, **kwargs):
        self.config = config


class FakeLedger:
    def summary(self):
        return {'offline_fake': True}


def main():
    findings = {}
    with tempfile.TemporaryDirectory(prefix='v4-review-repro-') as tmp:
        p = Path(tmp)
        hp, qp, mp = p / 'histories.jsonl', p / 'queries.jsonl', p / 'memories.jsonl'
        hp.write_text(json.dumps({'history_id': 'audit-h', 'text': 'original'}) + '\n')
        qs = [{'history_id': 'audit-h', 'question_id': f'q{i}', 'question': f'Q{i}',
               'question_type': 'atomic', 'answers': ['A'], 'scoring': 'exact'} for i in range(8)]
        qp.write_text(''.join(json.dumps(q) + '\n' for q in qs))
        memory = {'history_id': 'audit-h', 'condition': 'direct', 'status': 'natural_stop',
                  'text': 'original memory', 'memory_hash': runner.digest('original memory')}
        mp.write_text(json.dumps(memory) + '\n')
        cm = {'status': 'COMPLETE', 'no_future_query_visible_to_compressor': True,
              'memory_rows_sha256': runner.file_sha256(mp), 'histories_sha256': runner.file_sha256(hp),
              'history_ids': ['audit-h'], 'memory_rows': 1}
        (p / 'compression_manifest.json').write_text(json.dumps(cm))
        memory['text'] = 'MODIFIED after freeze'
        mp.write_text(json.dumps(memory) + '\n')
        seen = []

        def fake_attempt(provider, ledger, runtime, phase, hid, condition, memory_text,
                         group, mode, retry_index, effort, out):
            seen.append(memory_text)
            return {'text': json.dumps({'answers': [{'question_id': q['question_id'],
                    'answer': 'A', 'evidence_ids': []} for q in group]}), 'finish_reason': 'stop'}

        with patch.object(runner, 'Tokenizer', FakeTokenizer), \
             patch.object(runner, 'Provider', FakeProvider), \
             patch.object(runner, 'init_accounting', return_value=FakeLedger()), \
             patch.object(runner, '_reader_attempt', side_effect=fake_attempt):
            kwargs = dict(histories_path=hp, queries_path=qp, memories_path=mp,
                          out_dir=p / 'reader', phase='p2', conditions=['direct'],
                          runtime_path=ROOT / 'configs/v4_compact_v2_continuation_runtime_20260914.json')
            first = runner.score_phase(**kwargs)
            findings['tampered_memory_accepted'] = {'status': first['status'],
                    'fake_attempts': len(seen), 'memory_received': seen[0],
                    'file_hash_matches_frozen': runner.file_sha256(mp) == cm['memory_rows_sha256']}
            # Emulate a crash after call receipts but before question rows are durable.
            (p / 'reader/reader_rows.jsonl').write_text('')
            resumed = runner.score_phase(**kwargs, resume=True)
            findings['resume_after_call_receipt_before_scores'] = {
                'status': resumed['status'], 'reader_rows': resumed['reader_rows'],
                'expected_reader_rows': resumed['expected_reader_rows'], 'fake_attempts_total': len(seen)}
    runtime = runner.load_runtime()
    findings['default_prompt_config_mismatch'] = {
        'default_runtime': str(runner.RUNTIME_DEFAULT.relative_to(ROOT)),
        'configured_prompt': runtime['compressor_C1'].get('prompt_path', runtime['compressor_C1'].get('prompt_version')),
        'actual_prompt': str(runner.PROMPT_COMPRESS.relative_to(ROOT))}
    partial = [{'history_id': 'partial-h', 'condition': c, 'reader_slot': 'R1',
                'question_id': 'q0', 'information_type': 'atomic_fact',
                'status': 'valid', 'score': 1.0} for c in runner.PATHS]
    agg = runner.aggregate_history_utilities(partial, [], 'p2')[0]
    findings['partial_history_accepted_by_aggregator'] = {
        k: agg[k] for k in ('U_direct', 'U_staged2', 'U_rewrite',
                           'n_questions_direct', 'delta_SD')}
    print(json.dumps(findings, indent=2))


if __name__ == '__main__':
    main()
