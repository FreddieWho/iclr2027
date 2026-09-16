"""Offline regression tests for the V4 repair (F3/F4/F5/F8).

No provider calls, no network, no credentials. Provider, tokenizer and
accounting are faked; responses are deterministic fixtures.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_v4_fast_pilot as runner

CONT_RUNTIME = ROOT / 'configs/v4_compact_v2_continuation_runtime_20260914.json'
OLD_RUNTIME = ROOT / 'configs/v4_frozen_runtime.json'
QTYPES = ['atomic', 'update', 'relational', 'constraint',
          'atomic', 'update', 'relational', 'constraint']


class FakeTokenizer:
    def __init__(self, name):
        self.name = name

    def count(self, text):
        return len((text or '').split())


class FakeProvider:
    def __init__(self, config, *args, **kwargs):
        self.config = config


class FakeLedger:
    def summary(self):
        return {'offline_fake': True}


def make_fixture(tmp: Path, n_histories: int = 1):
    hp = tmp / 'histories.jsonl'
    qp = tmp / 'queries.jsonl'
    mp = tmp / 'memories.jsonl'
    hids = [f'h{i}' for i in range(n_histories)]
    hp.write_text(''.join(json.dumps({'history_id': h, 'text': f'history text {h}'}) + '\n'
                          for h in hids))
    qs = []
    for h in hids:
        for i in range(8):
            qs.append({'history_id': h, 'question_id': f'{h}-q{i}',
                       'question': f'Question {i} for {h}?',
                       'question_type': QTYPES[i], 'answers': ['YES'], 'scoring': 'exact'})
    qp.write_text(''.join(json.dumps(q) + '\n' for q in qs))
    mems = []
    for h in hids:
        for cond in ('direct', 'staged2', 'rewrite'):
            mems.append({'history_id': h, 'condition': cond, 'status': 'natural_stop',
                         'text': f'memory {h} {cond} text here',
                         'memory_hash': runner.digest(f'memory {h} {cond} text here')})
    mp.write_text(''.join(json.dumps(m) + '\n' for m in mems))
    cm = {'status': 'COMPLETE', 'no_future_query_visible_to_compressor': True,
          'memory_rows_sha256': runner.file_sha256(mp),
          'histories_sha256': runner.file_sha256(hp),
          'history_ids': hids, 'memory_rows': len(mems)}
    (tmp / 'compression_manifest.json').write_text(json.dumps(cm))
    return hp, qp, mp


def patch_offline(testcase):
    p1 = patch.object(runner, 'Tokenizer', FakeTokenizer)
    p2 = patch.object(runner, 'Provider', FakeProvider)
    p3 = patch.object(runner, 'init_accounting', return_value=FakeLedger())
    testcase.addCleanup(p1.stop); testcase.addCleanup(p2.stop); testcase.addCleanup(p3.stop)
    p1.start(); p2.start(); p3.start()


def counting_attempt_factory(calls):
    def fake_attempt(provider, ledger, runtime, phase, hid, condition, memory_text,
                     group, mode, retry_index, effort, out):
        calls.append((hid, condition, tuple(q['question_id'] for q in group), retry_index))
        return {'text': json.dumps({'answers': [{'question_id': q['question_id'],
                'answer': 'YES', 'evidence_ids': []} for q in group]}),
                'finish_reason': 'stop', 'cache_hit': True}
    return fake_attempt


def base_kwargs(hp, qp, mp, tmp, **over):
    kw = dict(histories_path=hp, queries_path=qp, memories_path=mp,
              out_dir=tmp / 'reader', phase='p2', conditions=['direct'],
              runtime_path=CONT_RUNTIME)
    kw.update(over)
    return kw


class FrozenInputTests(unittest.TestCase):
    def test_tampered_memory_text_rejected_before_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            hp, qp, mp = make_fixture(p)
            rows = [json.loads(line) for line in mp.read_text().splitlines()]
            rows[0]['text'] = 'MODIFIED after freeze'
            mp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            patch_offline(self)
            calls = []
            with patch.object(runner, '_reader_attempt',
                              side_effect=counting_attempt_factory(calls)):
                with self.assertRaisesRegex(ValueError, 'hash mismatch|SHA does not match'):
                    runner.score_phase(**base_kwargs(hp, qp, mp, p))
            self.assertEqual(calls, [])

    def test_memory_file_sha_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            hp, qp, mp = make_fixture(p)
            with open(mp, 'a') as f:
                f.write(json.dumps({'history_id': 'hX', 'condition': 'direct',
                                    'status': 'natural_stop', 'text': 'extra',
                                    'memory_hash': runner.digest('extra')}) + '\n')
            patch_offline(self)
            calls = []
            with patch.object(runner, '_reader_attempt',
                              side_effect=counting_attempt_factory(calls)):
                with self.assertRaisesRegex(ValueError, 'SHA does not match'):
                    runner.score_phase(**base_kwargs(hp, qp, mp, p))
            self.assertEqual(calls, [])

    def test_history_id_set_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            hp, qp, mp = make_fixture(p)
            with open(hp, 'a') as f:
                f.write(json.dumps({'history_id': 'intruder', 'text': 'x'}) + '\n')
            patch_offline(self)
            with patch.object(runner, '_reader_attempt',
                              side_effect=counting_attempt_factory([])):
                # histories file changed -> sha mismatch fires first; both are fail-closed
                with self.assertRaisesRegex(ValueError, 'SHA does not match|History IDs do not match'):
                    runner.score_phase(**base_kwargs(hp, qp, mp, p))

    def test_complete_fixture_scores_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            hp, qp, mp = make_fixture(p)
            patch_offline(self)
            calls = []
            with patch.object(runner, '_reader_attempt',
                              side_effect=counting_attempt_factory(calls)):
                first = runner.score_phase(**base_kwargs(hp, qp, mp, p))
                self.assertEqual(first['status'], 'COMPLETE')
                self.assertEqual(first['reader_rows'], 8)
                n_calls = len(calls)
                second = runner.score_phase(**base_kwargs(hp, qp, mp, p, resume=True))
                self.assertEqual(second['status'], 'COMPLETE')
            self.assertEqual(len(calls), n_calls)


class ResumeRecoveryTests(unittest.TestCase):
    def test_call_receipt_without_scores_recovers_exact_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            hp, qp, mp = make_fixture(p)
            patch_offline(self)
            calls = []
            with patch.object(runner, '_reader_attempt',
                              side_effect=counting_attempt_factory(calls)):
                first = runner.score_phase(**base_kwargs(hp, qp, mp, p))
                self.assertEqual(first['status'], 'COMPLETE')
                before = sorted(json.loads(line)['score_id']
                                for line in (p / 'reader/reader_rows.jsonl').read_text().splitlines())
                # Simulate a crash after call receipts but before any score row was durable.
                (p / 'reader/reader_rows.jsonl').write_text('')
                resumed = runner.score_phase(**base_kwargs(hp, qp, mp, p, resume=True))
                self.assertEqual(resumed['status'], 'COMPLETE')
                self.assertEqual(resumed['reader_rows'], 8)
                after = sorted(json.loads(line)['score_id']
                               for line in (p / 'reader/reader_rows.jsonl').read_text().splitlines())
                self.assertEqual(before, after)
                # A further resume is a strict no-op.
                n_calls = len(calls)
                third = runner.score_phase(**base_kwargs(hp, qp, mp, p, resume=True))
                self.assertEqual(third['status'], 'COMPLETE')
            self.assertEqual(len(calls), n_calls)

    def test_partial_scores_recovers_only_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            hp, qp, mp = make_fixture(p)
            patch_offline(self)
            calls = []
            with patch.object(runner, '_reader_attempt',
                              side_effect=counting_attempt_factory(calls)):
                runner.score_phase(**base_kwargs(hp, qp, mp, p))
                lines = (p / 'reader/reader_rows.jsonl').read_text().splitlines()
                (p / 'reader/reader_rows.jsonl').write_text('\n'.join(lines[:3]) + '\n')
                resumed = runner.score_phase(**base_kwargs(hp, qp, mp, p, resume=True))
                self.assertEqual(resumed['status'], 'COMPLETE')
                self.assertEqual(resumed['reader_rows'], 8)

    def test_unrecoverable_cell_blocks_without_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            hp, qp, mp = make_fixture(p)
            patch_offline(self)

            def failing_attempt(*args, **kwargs):
                raise RuntimeError('CACHE_RECEIPT_BLOCK: completed request cache is missing')

            with patch.object(runner, '_reader_attempt', side_effect=failing_attempt):
                result = runner.score_phase(**base_kwargs(hp, qp, mp, p))
                self.assertEqual(result['status'], 'INCOMPLETE')
                self.assertEqual(len(result['blocked_cells']), 1)
                self.assertIn('BLOCKED_MISSING_RESPONSE_RECEIPT',
                              result['blocked_cells'][0]['error'])

    def test_duplicate_and_unknown_keys_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            hp, qp, mp = make_fixture(p)
            patch_offline(self)
            calls = []
            with patch.object(runner, '_reader_attempt',
                              side_effect=counting_attempt_factory(calls)):
                runner.score_phase(**base_kwargs(hp, qp, mp, p))
                with open(p / 'reader/reader_rows.jsonl', 'a') as f:
                    f.write(json.dumps({'score_id': 'bogus-unknown-key'}) + '\n')
                with self.assertRaisesRegex(ValueError, 'Unknown score keys'):
                    runner.score_phase(**base_kwargs(hp, qp, mp, p, resume=True))


class AggregationContractTests(unittest.TestCase):
    def test_partial_arm_stays_missing_not_shrunk(self):
        partial = [{'history_id': 'partial-h', 'condition': c, 'reader_slot': 'R1',
                    'question_id': 'q0', 'information_type': 'atomic_fact',
                    'status': 'valid', 'score': 1.0} for c in runner.PATHS]
        agg = runner.aggregate_history_utilities(partial, [], 'p2')[0]
        self.assertIsNone(agg['U_direct'])
        self.assertIsNone(agg['U_staged2'])
        self.assertIsNone(agg['delta_SD'])

    def test_full_cell_scores_over_contract_denominator(self):
        rows = []
        for c in runner.PATHS:
            for i in range(8):
                rows.append({'history_id': 'full-h', 'condition': c, 'reader_slot': 'R1',
                             'question_id': f'q{i}',
                             'information_type': 'atomic_fact' if i % 2 == 0 else 'temporal_update_provenance',
                             'status': 'valid', 'score': 1.0})
        agg = runner.aggregate_history_utilities(rows, [], 'p2')[0]
        self.assertEqual(agg['U_direct'], 1.0)
        self.assertEqual(agg['delta_SD'], 0.0)


class PromptBindingTests(unittest.TestCase):
    def test_distinct_runtimes_resolve_distinct_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            pa = p / 'prompt_a.txt'
            pb = p / 'prompt_b.txt'
            pa.write_text('TEMPLATE_A {target_role} {target_tokens} {tokenizer}')
            pb.write_text('TEMPLATE_B {target_role} {target_tokens} {tokenizer}')
            ra = {'compressor_C1': {'prompt_path': str(pa)}}
            rb = {'compressor_C1': {'prompt_path': str(pb)}}
            path_a, sha_a, res_a = runner.resolve_compressor_prompt(ra, CONT_RUNTIME)
            path_b, sha_b, res_b = runner.resolve_compressor_prompt(rb, CONT_RUNTIME)
            self.assertEqual(res_a, 'runtime_explicit')
            self.assertNotEqual(sha_a, sha_b)
            self.assertIn('TEMPLATE_A', runner.c1_system_prompt(8, 'final', 'tok',
                                                               template_text=path_a.read_text()))
            self.assertIn('TEMPLATE_B', runner.c1_system_prompt(8, 'final', 'tok',
                                                               template_text=path_b.read_text()))

    def test_legacy_runtime_fails_closed_without_flag(self):
        with self.assertRaisesRegex(ValueError, 'prompt_path'):
            runner.resolve_compressor_prompt(runner.load_runtime(OLD_RUNTIME), OLD_RUNTIME)
        path, _, res = runner.resolve_compressor_prompt(
            runner.load_runtime(OLD_RUNTIME), OLD_RUNTIME, allow_legacy=True)
        self.assertEqual(res, 'legacy_global_fallback')

    def test_modified_template_rejects_frozen_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            hp, qp, mp = make_fixture(p)
            prompt = p / 'prompt.txt'
            prompt.write_text((ROOT / 'prompts/compress_v4_route_selection_compact_v2.txt')
                              .read_text())
            rt = dict(runner.load_runtime(CONT_RUNTIME))
            import copy
            rt = copy.deepcopy(rt)
            rt['compressor_C1']['prompt_path'] = str(prompt)
            rt_path = p / 'runtime.json'
            rt_path.write_text(json.dumps(rt))
            patch_offline(self)
            calls = []
            with patch.object(runner, '_reader_attempt',
                              side_effect=counting_attempt_factory(calls)):
                kw = base_kwargs(hp, qp, mp, p, runtime_path=rt_path)
                first = runner.score_phase(**kw)
                self.assertEqual(first['status'], 'COMPLETE')
                prompt.write_text('TAMPERED TEMPLATE')
                with self.assertRaisesRegex(RuntimeError, 'manifest mismatch'):
                    runner.score_phase(**{**kw, 'resume': True})


if __name__ == '__main__':
    unittest.main()
