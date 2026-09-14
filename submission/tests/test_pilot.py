"""Offline unit tests. Test fixtures are not research results."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from core import Tokenizer,path_targets,public_text,parse_answer,exact_score,Ledger,Provider,write_json,load_json,response_schema_summary
from make_synthetic import make_history
from build_plan import build
from import_longmemeval import convert
from analyze import bootstrap,paired_values
from decide import decide,REQUIRED

class CoreTests(unittest.TestCase):
    def test_demo_mark(self):self.assertTrue(Tokenizer('demo').is_demo)
    def test_bad_tokenizer(self):
        with self.assertRaises(ValueError):Tokenizer('words_as_real_tokens')
    def test_hf_json_tokenizer(self):
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'tokenizer.json'
            raw=HFTokenizer(WordLevel({'[UNK]':0,'alpha':1,'beta':2},unk_token='[UNK]'))
            raw.pre_tokenizer=Whitespace();path.write_text(raw.to_str())
            tok=Tokenizer('hf_json:'+str(path))
            self.assertEqual(tok.count('alpha beta'),2)
    def test_cap(self):
        t=Tokenizer('demo');self.assertEqual(t.count(t.cap('one two three four',2)),2)
    def test_positive_budget(self):
        with self.assertRaises(ValueError):Tokenizer('demo').cap('text',0)
    def test_targets(self):self.assertEqual(path_targets('staged3',100,1000),[400,200,100])
    def test_schedule_control(self):self.assertEqual(len(path_targets('staged3',100,1000)),len(path_targets('wait3',100,1000)))
    def test_too_large_budget(self):
        with self.assertRaises(ValueError):path_targets('staged3',300,1000)
    def test_unknown_arm(self):
        with self.assertRaises(ValueError):path_targets('unknown',100,1000)
    def test_query_allowlist(self):self.assertEqual(public_text({'text':'safe','question':'secret','answer':'hidden'}),'safe')
    def test_reader_parse(self):self.assertEqual(parse_answer('{"answer":"X"}')[0]['answer'],'X')
    def test_parse_failure_not_score(self):self.assertIsNone(parse_answer('not json')[0])
    def test_exact(self):self.assertEqual(exact_score('  ABC ',['abc']),1.)
    def test_not_substring(self):self.assertEqual(exact_score('ABC and wrong detail',['ABC']),0.)
    def test_raw_once(self):
        p=build({'budgets':[10,20],'arms':['raw','direct'],'replicates':[0,1],'history_tokens':100,'n_histories':2})
        self.assertEqual(sum(c['arm']=='raw' for c in p['cells']),1)
    def test_synthetic_determinism(self):
        a,b=make_history(1,9,Tokenizer('demo'),3000,8);c,d=make_history(1,9,Tokenizer('demo'),3000,8)
        self.assertEqual(a,c);self.assertEqual(b,d)
    def test_synthetic_gold_is_present(self):
        h,qs=make_history(1,9,Tokenizer('demo'),3000,8)
        self.assertEqual(len(qs),8)
        for q in qs:
            for e in q['gold_evidence']:self.assertIn(e,h['text'])
        self.assertNotIn('answers',h)
    def test_synthetic_new_seed(self):
        a,_=make_history(1,9,Tokenizer('demo'),3000,8);b,_=make_history(1,10,Tokenizer('demo'),3000,8)
        self.assertNotEqual(a['history_id'],b['history_id']);self.assertNotEqual(a['text'],b['text'])
    def test_import_no_label_leak(self):
        item={'question_id':'1','question_type':'knowledge-update','question':'SECRET_QUESTION','answer':'SECRET_GOLD','haystack_sessions':[[{'role':'user','content':'my site is A','has_answer':True}]],'haystack_dates':['2024-01-01'],'haystack_session_ids':['s1'],'answer_session_ids':['s1']}
        h,q=convert(item)
        for s in ['has_answer','SECRET_QUESTION','SECRET_GOLD','answer_session_ids']:self.assertNotIn(s,h['text'])
        self.assertEqual(q['scoring'],'semantic_required');self.assertTrue(q['gold_evidence'])
        item2=dict(item,question_id='2',question='different question')
        h2,_=convert(item2);self.assertEqual(h['group_id'],h2['group_id'])
    def test_schema_mismatch(self):
        with self.assertRaises(ValueError):convert({'haystack_sessions':[[]],'haystack_dates':[],'haystack_session_ids':[]})
    def test_live_forbidden(self):
        with tempfile.TemporaryDirectory() as td:
            cfg={'live_authorized':False,'usd_cap':10,'max_calls':10,'max_input_tokens':10000,'max_output_tokens':1000}
            with self.assertRaises(RuntimeError):Provider(cfg,Tokenizer('demo'),td,mock=False)
    def test_cost_reservation(self):
        with tempfile.TemporaryDirectory() as td:
            l=Ledger(Path(td)/'x.sqlite',{'usd_cap':1,'max_calls':2,'max_input_tokens':1000,'max_output_tokens':1000})
            a=l.reserve('a',.8,100,100)
            with self.assertRaises(RuntimeError):l.reserve('b',.3,100,100)
            l.settle(a,.1,20,20);l.reserve('b',.3,100,100)
            with self.assertRaises(RuntimeError):l.reserve('c',.1,10,10)
            self.assertEqual(l.summary()['attempts'],2)

    def test_live_metadata_endpoint_and_cache_accounting(self):
        import requests
        cfg={'live_authorized':True,'usd_cap':1,'max_calls':10,'max_input_tokens':10000,'max_output_tokens':10000,'models_endpoint_sha256':'models-sha','opencode_session_id':'test-session','compressor':{'model':'glm-5.3-flash','base_url':'https://example/v1','chat_completions_url':'https://example/v1/chat/completions','api_key_env':'PILOT_TEST_KEY','context_tokens':1000,'max_visible_output_tokens':100,'output_parameter':'max_tokens','supports_seed':False,'extra_parameters':{'temperature':0.2},'input_usd_per_million':.15,'output_usd_per_million':.5,'cache_read_usd_per_million':.03,'input_reservation_overhead_tokens':16,'full_HF_tokenizer_revision':'hf-sha','user_agent':'iclr-memory-pilot/0.1'}}
        response=Mock(status_code=200)
        response.json.return_value={'model':'glm-5.3-flash','choices':[{'message':{'content':'memory'},'finish_reason':'stop'}],'usage':{'prompt_tokens':7,'completion_tokens':2,'prompt_tokens_details':{'cached_tokens':3},'completion_tokens_details':{'reasoning_tokens':1}}}
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ,{'PILOT_TEST_KEY':'secret'}), patch.object(requests,'post',return_value=response) as post:
            result=Provider(cfg,Tokenizer('tiktoken:cl100k_base'),td).call('compressor','system','user',5,0,request_metadata={'native_memory_tokens':4},provider_output_budget=20,visible_output_limit=5)
        self.assertEqual(post.call_args.args[0],'https://example/v1/chat/completions')
        self.assertEqual(post.call_args.kwargs['headers']['User-Agent'],'iclr-memory-pilot/0.1')
        self.assertEqual(post.call_args.kwargs['headers']['x-opencode-session'],'test-session')
        self.assertEqual(post.call_args.kwargs['json']['max_tokens'],20)
        self.assertTrue(all(set(message) == {'role','content'} for message in post.call_args.kwargs['json']['messages']))
        self.assertEqual(result['provider_input_tokens'],7);self.assertEqual(result['provider_output_tokens'],2);self.assertEqual(result['cache_read_tokens'],3)
        self.assertEqual(result['reasoning_tokens'],1)
        self.assertEqual(result['full_HF_tokenizer_revision'],'hf-sha');self.assertEqual(result['GET_v1_models_sha256'],'models-sha');self.assertEqual(result['native_memory_tokens'],4)
        self.assertEqual(result['provider_output_budget_tokens'],20);self.assertEqual(result['max_completion_tokens'],20);self.assertEqual(result['visible_output_limit_tokens'],5)

    def test_reader_visible_limit_never_truncates(self):
        from run_pilot import validate_reader_response
        result={'text':'{"answer":"one two three","evidence_ids":[]}','finish_reason':'stop'}
        obj,error,answer_tokens,visible_tokens=validate_reader_response(result,Tokenizer('demo'),2)
        self.assertEqual(obj['answer'],'one two three');self.assertEqual(error,'protocol_violation_answer_over_limit');self.assertEqual(answer_tokens,3);self.assertGreaterEqual(visible_tokens,1)

    def test_response_schema_summary_excludes_generated_text(self):
        body={'model':'glm-5.3-flash','choices':[{'finish_reason':'length','message':{'content':None,'reasoning_content':'SECRET_REASONING','role':'assistant'}}],'usage':{'prompt_tokens':10,'completion_tokens':20}}
        summary=response_schema_summary(body)
        encoded=json.dumps(summary)
        self.assertEqual(summary['content_type'],'NoneType');self.assertEqual(summary['completion_tokens'],20)
        self.assertNotIn('SECRET_REASONING',encoded)

    def test_ledger_rejects_measured_overflow(self):
        with tempfile.TemporaryDirectory() as td:
            l=Ledger(Path(td)/'x.sqlite',{'usd_cap':1,'max_calls':2,'max_input_tokens':1000,'max_output_tokens':1000})
            l.reserve('a',.2,10,10);rid=l.reserve('b',.2,10,10)
            with self.assertRaises(RuntimeError):l.settle(rid,.9,10,10)

    def test_rewrite_identical_text_still_new_step(self):
        with tempfile.TemporaryDirectory() as td:
            cfg={'usd_cap':1,'max_calls':10,'max_input_tokens':10000,'max_output_tokens':10000,'compressor':{'model':'MOCK','base_url':'mock://','api_key_env':'UNUSED'}}
            p=Provider(cfg,Tokenizer('demo'),td,mock=True)
            a=p.call('compressor','s','same',100,0,mock_text='same',namespace=[100])
            b=p.call('compressor','s','same',100,0,mock_text='same',namespace=[100,100])
            c=p.call('compressor','s','same',100,0,mock_text='same',namespace=[100,100])
            self.assertNotEqual(a['request_key'],b['request_key']);self.assertFalse(b['cache_hit']);self.assertTrue(c['cache_hit'])
    def test_oracle_is_read_only_arm(self):
        self.assertEqual(path_targets('oracle',100,1000),[])
        self.assertEqual(path_targets('no_memory',100,1000),[])

    def test_synthetic_is_chronological(self):
        h,_=make_history(3,12,Tokenizer('demo'),3000,8)
        import re
        days=[int(x) for x in re.findall(r'2025-03-(\d+):',h['text'])]
        self.assertEqual(days,sorted(days))
    def test_mock_end_to_end_and_resume(self):
        from run_pilot import run
        from core import append_jsonl,read_jsonl,load_json
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            for i in range(2):
                h,qs=make_history(i,31,Tokenizer('demo'),3000,8)
                append_jsonl(td/'h.jsonl',h)
                for q in qs:append_jsonl(td/'q.jsonl',q)
            profile={'budgets':[256],'arms':['direct','staged3','rewrite3','oracle','no_memory'],'replicates':[0],'history_tokens':3000,'n_histories':2,'tokenizer':'demo'}
            write_json(td/'plan.json',build(profile))
            cfg=load_json(Path(__file__).resolve().parents[1]/'configs/models.example.json')
            write_json(td/'models.json',cfg)
            args=(td/'h.jsonl',td/'q.jsonl',td/'plan.json',td/'models.json',td/'out')
            run(*args,mock=True)
            before=len(read_jsonl(td/'out/scores.jsonl'))
            run(*args,mock=True)
            after=len(read_jsonl(td/'out/scores.jsonl'))
            self.assertEqual(before,80);self.assertEqual(before,after)
            self.assertTrue(all(x['mock'] for x in read_jsonl(td/'out/scores.jsonl')))
            self.assertIn('code_hash',load_json(td/'out/run_manifest.json'))

    def test_compressor_respects_provider_reasoning_floor(self):
        from run_pilot import compression_maxout
        models={'compressor':{'max_visible_output_tokens':10240,'provider_min_output_tokens':2048}}
        self.assertEqual(compression_maxout(1024,models),2048)

    def test_reasoning_budget_v2_contract_is_frozen(self):
        cfg=load_json(Path(__file__).resolve().parents[1]/'configs/models_opencode_go_p0_reasoning_v2.json')
        self.assertEqual(cfg['contract_revision'],'REASONING_BUDGET_AMENDMENT_V2')
        self.assertEqual(cfg['compressor']['model'],'deepseek-v4.1-flash')
        self.assertEqual(cfg['compressor']['reasoning_effort'],'low')
        self.assertEqual(cfg['compressor']['final_memory_calls']['max_completion_tokens'],12000)
        self.assertEqual(cfg['compressor']['intermediate_memory_calls']['max_completion_tokens'],24000)
        self.assertEqual(cfg['reader']['model'],'glm-5.3-flash')
        self.assertEqual(cfg['reader']['reasoning_effort'],'low')
        self.assertEqual(cfg['reader']['max_completion_tokens'],12000)
        self.assertEqual(cfg['reader']['max_visible_output_tokens'],512)
        self.assertFalse(cfg['max_effort']['use_in_primary_pilot'])
        self.assertFalse(cfg['truncation']['posthoc_memory_truncation'])

    def test_primary_max_effort_is_blocked_by_v2_policy(self):
        from run_pilot import validate_primary_reasoning_policy
        with self.assertRaises(RuntimeError):
            validate_primary_reasoning_policy({'max_effort': {'use_in_primary_pilot': False},
                                               'compressor': {'reasoning_effort': 'max'}})

    def test_visible_budget_v3_config_is_inherited_and_fail_closed(self):
        from core import load_model_config
        from run_pilot import validate_visible_budget_protocol
        root=Path(__file__).resolve().parents[1]
        cfg=load_model_config(root/'configs/models_opencode_go_p0_visible_budget_v3.json')
        self.assertEqual(cfg['compressor']['model'],'deepseek-v4.1-flash')
        self.assertEqual(cfg['reader']['model'],'glm-5.3-flash')
        self.assertEqual(cfg['visible_budget_protocol_v3']['preflight_budget_feasibility']['candidate_budgets'],[1024,1536,2048,3072,4096])
        self.assertEqual(cfg['final_memory_budgets_native_tokens'],[])
        with self.assertRaisesRegex(RuntimeError,'preflight is not complete'):
            validate_visible_budget_protocol(cfg)

    def test_visible_budget_v3_classifies_overflow_separately_from_memory_quality(self):
        from run_pilot import compressor_budget_compliance,reasoning_tokens_or_estimate,actual_compression_rate
        models={'visible_budget_protocol_v3':{'hard_facts':{}},'truncation':{'accepted_finish_reasons':['stop']}}
        natural={'text':'memory','finish_reason':'stop','provider_output_tokens':11,'reasoning_tokens':7}
        self.assertEqual(compressor_budget_compliance(natural,11,10,models),'BUDGET_NONCOMPLIANT')
        self.assertEqual(compressor_budget_compliance(natural,10,10,models),'COMPLIANT')
        self.assertEqual(reasoning_tokens_or_estimate(natural,4),(7,'PROVIDER_REPORTED'))
        self.assertEqual(reasoning_tokens_or_estimate({'provider_output_tokens':11},4),(7,'ESTIMATED_COMPLETION_MINUS_VISIBLE_NATIVE'))
        self.assertEqual(actual_compression_rate(100,25),0.75)

    def test_visible_budget_overflow_is_unscored_and_reader_is_skipped(self):
        from run_pilot import run
        from core import append_jsonl,read_jsonl
        class FakeProvider:
            def __init__(self,*args,**kwargs):self.ledger=type('LedgerStub',(),{'summary':lambda self:{'attempts':1}})()
            def call(self,role,system,user,max_tokens,replicate,**kwargs):
                if role=='reader':raise AssertionError('R1 must not receive a budget-noncompliant memory')
                body=' '.join(['x']*(int(kwargs['visible_output_limit'])+1))
                return {'text':body,'finish_reason':'stop','model':'TEST','provider_model_id':'TEST','returned_model_field':'TEST','provider_name':'TEST','reasoning_effort':'low','provider_output_budget_tokens':100,'max_completion_tokens':100,'completion_budget_class':'final','visible_output_limit_tokens':kwargs['visible_output_limit'],'provider_input_tokens':20,'provider_output_tokens':14,'reasoning_tokens':3,'cache_read_tokens':0,'full_HF_tokenizer_revision':'test-rev','GET_v1_models_sha256':'test-catalog','run_start_utc':'test-time','opencode_session_id':'test-session','native_memory_tokens':20,'think_blocks_removed':False,'reasoning_content_present':True,'usage':{'prompt_tokens':20,'completion_tokens':14},'request_key':'test-request','cache_hit':False,'cost_usd_or_conservative':0}
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            history,questions=make_history(0,41,Tokenizer('demo'),1500,8)
            append_jsonl(td/'histories.jsonl',history);append_jsonl(td/'queries.jsonl',questions[0])
            write_json(td/'plan.json',build({'budgets':[10],'arms':['direct'],'replicates':[0],'history_tokens':1500,'n_histories':1,'tokenizer':'demo'}))
            cfg=load_json(Path(__file__).resolve().parents[1]/'configs/models.example.json')
            cfg['visible_budget_protocol_v3']={'hard_facts':{}}
            cfg['compressor']['final_memory_calls']={'target_memory_tokens':[10],'max_completion_tokens':100}
            cfg['compressor']['provider_output_budget_tokens']=100
            cfg['compressor']['max_visible_output_tokens']=100
            write_json(td/'models.json',cfg)
            with patch('run_pilot.Provider',FakeProvider),patch('run_pilot.validate_visible_budget_protocol'):
                run(td/'histories.jsonl',td/'queries.jsonl',td/'plan.json',td/'models.json',td/'out',mock=False)
            score=read_jsonl(td/'out/scores.jsonl')[0]
            self.assertEqual(score['status'],'BUDGET_NONCOMPLIANT')
            self.assertIsNone(score['score'])
            self.assertEqual(score['budget_compliance'],'BUDGET_NONCOMPLIANT')
            self.assertGreater(score['visible_native_tokens'],score['budget'])

class StatisticsTests(unittest.TestCase):
    def test_bootstrap_direction(self):
        r=bootstrap([.1,.2,.15,.12],n_boot=1000);self.assertGreater(r['ci_low'],0)
    def test_zero_ci_is_not_failure(self):
        r=bootstrap([0.,0.,0.,0.],n_boot=1000);self.assertEqual(r['ci_high'],0)
    def test_seed_not_independent_history(self):
        rows=[]
        for rep in (0,1,2):
            for q in ('q1','q2'):
                for arm,score in [('direct',1.),('staged3',0.)]:rows.append({'group_id':'same','question_id':q,'replicate':rep,'reader':'r','arm':arm,'budget':100,'question_type':'atomic','status':'valid','score':score,'mock':False})
        v,qc=paired_values(rows,'direct','staged3',100);self.assertEqual(len(v),1);self.assertEqual(qc['paired_question_replicates'],6)
    def test_technical_failure_not_zero(self):
        rows=[{'group_id':'h','question_id':'q','replicate':0,'reader':'r','arm':'direct','budget':100,'question_type':'atomic','status':'valid','score':1,'mock':False},{'group_id':'h','question_id':'q','replicate':0,'reader':'r','arm':'staged3','budget':100,'question_type':'atomic','status':'timeout','score':None,'mock':False}]
        v,qc=paired_values(rows,'direct','staged3',100);self.assertFalse(v);self.assertEqual(qc['valid_pair_fraction'],0)
    def test_duplicate_rejected(self):
        r={'group_id':'h','question_id':'q','replicate':0,'arm':'direct','budget':1,'question_type':'atomic','status':'valid','score':1}
        with self.assertRaises(ValueError):paired_values([r,r],'direct','staged3',1)

class DecisionTests(unittest.TestCase):
    def fixture(self,td):
        Path(td,'evidence.md').write_text('UNIT TEST FIXTURE, NOT REAL RESULTS')
        return {'contains_mock':False,'audits':{k:{'pass':True,'evidence':['evidence.md']} for k in REQUIRED},'candidates':[{'route_id':'G1_ROBUST_PATH','estimate':.1,'ci_low':.05,'ci_high':.15,'n_histories':96,'confirmatory':True,'evidence':['evidence.md']}]}
    def test_go_fixture(self):
        with tempfile.TemporaryDirectory() as td:self.assertEqual(decide(self.fixture(td),td)['decision'],'GO')
    def test_mock_never_go(self):
        with tempfile.TemporaryDirectory() as td:
            x=self.fixture(td);x['contains_mock']=True;self.assertEqual(decide(x,td)['decision'],'NO_GO')
    def test_missing_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            x=self.fixture(td);x['audits']['novelty']['evidence']=['absent'];self.assertEqual(decide(x,td)['reason'],'NOVELTY')
    def test_non_sig_not_go(self):
        with tempfile.TemporaryDirectory() as td:
            x=self.fixture(td);x['candidates'][0]['ci_low']=-.05;self.assertEqual(decide(x,td)['decision'],'NO_GO')
    def test_negative_effect_can_go(self):
        with tempfile.TemporaryDirectory() as td:
            x=self.fixture(td);x['candidates'][0].update(estimate=-.1,ci_low=-.15,ci_high=-.05);self.assertEqual(decide(x,td)['decision'],'GO')
    def test_small_n_no_go(self):
        with tempfile.TemporaryDirectory() as td:
            x=self.fixture(td);x['candidates'][0]['n_histories']=8;self.assertEqual(decide(x,td)['decision'],'NO_GO')
    def test_small_effect_needs_precision(self):
        with tempfile.TemporaryDirectory() as td:
            x=self.fixture(td);x['candidates']=[];x['reason_if_none']='SMALL_EFFECT';self.assertEqual(decide(x,td)['reason'],'EVIDENCE')
    def test_invalid_ci_not_go(self):
        with tempfile.TemporaryDirectory() as td:
            x=self.fixture(td);x['candidates'][0].update(ci_low=.2,ci_high=.3);self.assertEqual(decide(x,td)['decision'],'NO_GO')

if __name__=='__main__':unittest.main()
