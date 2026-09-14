"""Regression tests for real failure modes; no scientific results or network."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from core import Ledger, Provider, final_visible_body


class CountingTokenizer:
    is_demo = False
    name = 'fixed-test-tokenizer'

    def count(self, value):
        return len(value)


def config():
    return {'live_authorized': True, 'usd_cap': 1, 'max_calls': 10,
            'max_input_tokens': 10000, 'max_output_tokens': 10000,
            'provider_name': 'test', 'compressor': {
                'model': 'test', 'base_url': 'https://example.invalid/v1',
                'local_no_key': True, 'context_tokens': 1000,
                'max_visible_output_tokens': 100, 'reservation_output_tokens': 1,
                'input_usd_per_million': .15, 'output_usd_per_million': .5}}


class TransportRegressionTests(unittest.TestCase):
    def test_final_visible_body_excludes_explicit_thinking(self):
        body, removed = final_visible_body('<think>PRIVATE_THOUGHT</think>VISIBLE_MEMORY')
        self.assertEqual(body, 'VISIBLE_MEMORY')
        self.assertTrue(removed)
        body, removed = final_visible_body('<think>UNFINISHED_THOUGHT')
        self.assertEqual(body, '')
        self.assertTrue(removed)

    def test_stage_budgets_follow_reasoning_v2_contract(self):
        from run_pilot import completion_budget_for, validate_compression_target
        cfg = {'provider_output_budget_tokens': 24000,
               'final_memory_calls': {'target_memory_tokens': [1024, 2048], 'max_completion_tokens': 12000},
               'intermediate_memory_calls': {'max_memory_tokens': 8192, 'max_completion_tokens': 24000}}
        self.assertEqual(completion_budget_for(cfg, 4096, 1, 2), 24000)
        self.assertEqual(completion_budget_for(cfg, 1024, 2, 2), 12000)
        validate_compression_target(cfg, 4096, 1, 2)
        with self.assertRaises(RuntimeError):
            validate_compression_target(cfg, 512, 1, 1)

    def test_hf_json_native_memory_excludes_special_tokens(self):
        from core import Tokenizer
        tok = Tokenizer.__new__(Tokenizer)
        tok.name = 'hf_json:test'
        tok.is_demo = False
        tok.encoder = Mock()
        tok.encoder.encode.return_value.ids = [7, 8]
        self.assertEqual(tok.count('memory'), 2)
        tok.encoder.encode.assert_called_once_with('memory', add_special_tokens=False)

    def test_sessions_stable_within_path_and_isolated_across_paths(self):
        with tempfile.TemporaryDirectory() as td:
            provider = Provider(config(), CountingTokenizer(), td, mock=True)
            a = provider.call('compressor','s','a',20,0,
                request_metadata={'experimental_path': ['h', 'direct', 1024, 0]})
            b = provider.call('compressor','s','b',20,0,
                request_metadata={'experimental_path': ['h', 'direct', 1024, 0]})
            c = provider.call('compressor','s','a',20,0,
                request_metadata={'experimental_path': ['h', 'staged3', 1024, 0]})
            self.assertEqual(a['opencode_session_id'], b['opencode_session_id'])
            self.assertNotEqual(a['opencode_session_id'], c['opencode_session_id'])
            self.assertNotEqual(a['request_key'], c['request_key'])

    def test_different_output_roots_share_configured_spending_limit(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = config()
            cfg['ledger_path'] = str(Path(td) / 'shared.sqlite')
            a = Provider(cfg, CountingTokenizer(), Path(td) / 'a', mock=True)
            b = Provider(cfg, CountingTokenizer(), Path(td) / 'b', mock=True)
            a.ledger.reserve('earlier_request', .99999, 10, 10)
            with self.assertRaises(RuntimeError):
                b.ledger.reserve('new_request', .001, 1, 1)

    def test_reader_reasoning_only_not_scored_or_truncated(self):
        from run_pilot import validate_reader_response
        obj, error, answer_tokens, visible = validate_reader_response(
            {'text': '', 'finish_reason': 'length'}, CountingTokenizer(), 512)
        self.assertIsNone(obj)
        self.assertEqual(error, 'TECHNICAL_INVALID')
        self.assertIsNone(answer_tokens)
        self.assertEqual(visible, 0)

    def test_reasoning_only_is_unscorable_but_usage_and_shape_persist(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            'model': 'test', 'choices': [{'finish_reason': 'length', 'message': {
                'content': None, 'reasoning_content': 'PRIVATE_THOUGHT'}}],
            'usage': {'prompt_tokens': 9, 'completion_tokens': 100}}
        with tempfile.TemporaryDirectory() as td, patch('requests.post', return_value=response):
            provider = Provider(config(), CountingTokenizer(), td)
            result = provider.call('compressor', 's', 'u', 100, 0)
            self.assertEqual(result['text'], '')
            self.assertEqual(result['finish_reason'], 'length')
            self.assertEqual(result['visible_response_tokens'], 0)
            self.assertEqual(provider.ledger.summary()['output_tokens'], 100)
            saved = list((Path(td) / 'cache').glob('*.json'))
            self.assertEqual(len(saved), 1)
            self.assertNotIn('PRIVATE_THOUGHT', saved[0].read_text())

    def test_think_block_and_separate_reasoning_are_never_saved(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            'model': 'test', 'choices': [{'finish_reason': 'stop', 'message': {
                'content': '<think>PRIVATE_THOUGHT</think>VISIBLE_MEMORY',
                'reasoning_content': 'SEPARATE_PRIVATE_THOUGHT'}}],
            'usage': {'prompt_tokens': 9, 'completion_tokens': 7}}
        with tempfile.TemporaryDirectory() as td, patch('requests.post', return_value=response):
            result = Provider(config(), CountingTokenizer(), td).call('compressor', 's', 'u', 100, 0)
            self.assertEqual(result['text'], 'VISIBLE_MEMORY')
            self.assertTrue(result['think_blocks_removed'])
            self.assertTrue(result['reasoning_content_present'])
            saved = list((Path(td) / 'cache').glob('*.json'))
            self.assertNotIn('PRIVATE_THOUGHT', saved[0].read_text())
            self.assertNotIn('SEPARATE_PRIVATE_THOUGHT', saved[0].read_text())

    def test_entire_provider_output_reserved_before_network(self):
        with tempfile.TemporaryDirectory() as td, patch('requests.post', side_effect=ValueError('intercept')):
            provider = Provider(config(), CountingTokenizer(), td)
            with self.assertRaises(ValueError):
                provider.call('compressor', 's', 'u', 100, 0)
            self.assertEqual(provider.ledger.summary()['output_tokens'], 100)

    def test_actual_overflow_is_committed_even_when_exception_raised(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / 'ledger.sqlite', config())
            rid = ledger.reserve('request', .1, 1, 1)
            with self.assertRaises(RuntimeError):
                ledger.settle(rid, 1.1, 3, 4)
            self.assertEqual(ledger.summary()['usd_actual_plus_uncertain_reservations'], 1.1)
            with sqlite3.connect(ledger.path) as db:
                self.assertEqual(db.execute('select state from calls').fetchone()[0], 'completed_over_cap')


if __name__ == '__main__':
    unittest.main()
