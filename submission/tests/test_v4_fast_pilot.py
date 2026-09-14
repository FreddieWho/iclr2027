import json
import tempfile
import unittest
from pathlib import Path

from run_v4_fast_pilot import (
    V4CostLedger, c1_system_prompt, classify_memory_output, compression_user,
    parse_batch_response, path_targets,
    _request_cost_bound,
)


class V4ProtocolTests(unittest.TestCase):
    def test_only_predeclared_two_stage_paths_are_built(self):
        self.assertEqual(path_targets(4096), {
            'direct':[4096], 'staged2':[8192,4096], 'rewrite':[4096,4096]
        })

    def test_compressor_input_is_only_the_history_text(self):
        user = compression_user('history-only')
        self.assertIn('history-only', user)
        self.assertNotIn('future-question', user)
        self.assertNotIn('gold-answer', user)

    def test_compact_compressor_prompt_is_shared_template_with_soft_target(self):
        prompt = c1_system_prompt(8192, 'final', 'deepseek-test-tokenizer')
        self.assertIn('high-density factual ledger', prompt)
        self.assertIn('one terse record per entity', prompt)
        self.assertIn('soft reference, not a hard ceiling', prompt)
        self.assertNotIn('{target_role}', prompt)
        self.assertNotIn('{target_tokens}', prompt)
        self.assertNotIn('{tokenizer}', prompt)

    def test_batch_parser_requires_exact_question_ids_and_schema(self):
        payload = {'answers':[
            {'question_id':'q1','answer':'A-1','evidence_ids':['E1']},
            {'question_id':'q2','answer':'UNKNOWN','evidence_ids':[]},
        ]}
        parsed,error=parse_batch_response(json.dumps(payload),['q1','q2'])
        self.assertIsNone(error)
        self.assertEqual(parsed['q1']['answer'],'A-1')
        missing,error=parse_batch_response(json.dumps({'answers':payload['answers'][:1]}),['q1','q2'])
        self.assertIsNone(missing)
        self.assertEqual(error,'missing_or_unexpected_question_ids')

    def test_memory_target_is_not_used_as_a_posthoc_length_cap(self):
        # A naturally stopped body remains usable when it is longer than B.
        self.assertEqual(classify_memory_output('x' * 5000,'stop'),'natural_stop')
        self.assertEqual(classify_memory_output('partial','length'),'TECHNICAL_INVALID_LENGTH')

    def test_cost_reservation_uses_peak_rates_and_enforces_both_caps(self):
        cfg={'context_tokens':100000,'input_reservation_overhead_tokens':256,
             'input_usd_per_million':.3,'output_usd_per_million':1.2}
        reserved_input,reserved_usd=_request_cost_bound(cfg,1000,2000)
        self.assertEqual(reserved_input,1256)
        self.assertAlmostEqual(reserved_usd,(1256*.3+2000*1.2)/1_000_000)

        with tempfile.TemporaryDirectory() as tmp:
            ledger=V4CostLedger(Path(tmp)/'ledger.sqlite',{
                'project_total':.20,
                'model_caps':{'deepseek-v4.1-flash':.10,'glm-5.3-flash':.15},
                'prior_project_actual_plus_uncertain':.05,
            })
            state=ledger.reserve('ds1',phase='p0',history_id='h1',condition='direct',
                step=1,role='compressor',model='deepseek-v4.1-flash',usd=.04,
                input_tokens=10,output_tokens=10,metadata={})
            self.assertEqual(state,'new')
            ledger.settle('ds1',usd=.02,input_tokens=8,output_tokens=4,
                          cache_read_tokens=0,provider_request_key='provider-1',finish_reason='stop')
            state=ledger.reserve('glm1',phase='p0',history_id='h1',condition='raw',
                step=0,role='reader',model='glm-5.3-flash',usd=.10,
                input_tokens=10,output_tokens=10,metadata={})
            self.assertEqual(state,'new')
            ledger.settle('glm1',usd=.08,input_tokens=8,output_tokens=4,
                          cache_read_tokens=0,provider_request_key='provider-2',finish_reason='stop')
            with self.assertRaisesRegex(RuntimeError,'cap'):
                ledger.reserve('ds2',phase='p0',history_id='h2',condition='direct',
                    step=1,role='compressor',model='deepseek-v4.1-flash',usd=.04,
                    input_tokens=10,output_tokens=10,metadata={})
            summary=ledger.summary()
            self.assertAlmostEqual(summary['project_actual_plus_uncertain_usd'],.15)
            self.assertAlmostEqual(summary['model_caps']['deepseek-v4.1-flash']['remaining_usd'],.03)

    def test_unresolved_request_is_quarantined_by_logical_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=V4CostLedger(Path(tmp)/'ledger.sqlite',{
                'project_total':1.0,
                'model_caps':{'deepseek-v4.1-flash':1.0},
                'prior_project_actual_plus_uncertain':0.0,
            })
            ledger.reserve('uncertain-key',phase='p0',history_id='h1',condition='direct',
                step=1,role='compressor',model='deepseek-v4.1-flash',usd=.04,
                input_tokens=10,output_tokens=10,
                metadata={'full_request_native_tokens':123,'prompt_provider_budget':12288})
            prior=ledger.unresolved_request(phase='p0',history_id='h1',condition='direct',
                step=1,role='compressor',model='deepseek-v4.1-flash',provider_budget=12288)
            self.assertEqual(prior['request_key'],'uncertain-key')
            self.assertEqual(prior['metadata']['full_request_native_tokens'],123)
            # The predeclared B=8192 adjustment uses a different completion
            # budget, so the old request remains reserved without blocking it.
            self.assertIsNone(ledger.unresolved_request(phase='p0',history_id='h1',condition='direct',
                step=1,role='compressor',model='deepseek-v4.1-flash',provider_budget=16384))
            self.assertIsNone(ledger.unresolved_request(phase='p0',history_id='h1',condition='rewrite',
                step=1,role='compressor',model='deepseek-v4.1-flash',provider_budget=12288))
            ledger.settle('uncertain-key',usd=.02,input_tokens=8,output_tokens=4,
                          cache_read_tokens=0,provider_request_key='provider-1',finish_reason='stop')
            self.assertIsNone(ledger.unresolved_request(phase='p0',history_id='h1',condition='direct',
                step=1,role='compressor',model='deepseek-v4.1-flash',provider_budget=12288))


if __name__ == '__main__':
    unittest.main()
