"""Reference static compaction runner: run all memories first, then open questions."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from core import *


def completion_budget_for(compressor_cfg, target, step_index, step_count):
    """Resolve the provider budget from the V2 stage contract, never from target size."""
    is_final = step_index == step_count
    section_name = 'final_memory_calls' if is_final else 'intermediate_memory_calls'
    section = compressor_cfg.get(section_name)
    if section is None:
        return int(compressor_cfg.get('provider_output_budget_tokens', target))
    key = 'max_completion_tokens'
    if key not in section or int(section[key]) <= 0:
        raise RuntimeError(f'CONFIG_BLOCK: {section_name}.{key} must be positive')
    return int(section[key])


def validate_compression_target(compressor_cfg, target, step_index, step_count):
    """Validate V2 memory targets before a provider call; never clip them."""
    if step_count <= 0 or not 1 <= step_index <= step_count:
        raise ValueError('invalid compression stage index')
    if step_index == step_count:
        allowed = compressor_cfg.get('final_memory_calls', {}).get('target_memory_tokens')
        if allowed is not None and int(target) not in {int(x) for x in allowed}:
            raise RuntimeError('CONFIG_BLOCK: final memory target is outside the frozen target set')
    else:
        maximum = compressor_cfg.get('intermediate_memory_calls', {}).get('max_memory_tokens')
        if maximum is not None and int(target) > int(maximum):
            raise RuntimeError('CONFIG_BLOCK: intermediate memory target exceeds the frozen maximum')


def reader_reasoning_contract(reader_cfg):
    """Return the fixed R1 effort and calibration state, rejecting drift."""
    extra = reader_cfg.get('extra_parameters', {})
    effort = reader_cfg.get('reasoning_effort', extra.get('reasoning_effort'))
    calibration = reader_cfg.get('calibration', {})
    initial = calibration.get('initial_reasoning_effort', reader_cfg.get('initial_reasoning_effort'))
    status = calibration.get('status', 'UNSPECIFIED')
    if initial is None:
        return effort, status
    promoted = status in ('PROMOTED_HIGH', 'HIGH')
    expected = 'high' if promoted else initial
    if effort != expected:
        raise RuntimeError('CONFIG_BLOCK: R1 effort does not match its frozen calibration state')
    if effort not in ('low', 'high'):
        raise RuntimeError('CONFIG_BLOCK: R1 formal effort must be low or high')
    return effort, status


def validate_primary_reasoning_policy(models):
    """Prevent an accidental max-effort primary run under the V2 contract."""
    if models.get('max_effort', {}).get('use_in_primary_pilot') is not False:
        return
    for role in ('compressor', 'reader'):
        cfg = models.get(role, {})
        effort = cfg.get('reasoning_effort', cfg.get('extra_parameters', {}).get('reasoning_effort'))
        if effort == 'max':
            raise RuntimeError('CONFIG_BLOCK: max effort is reserved for post-GO robustness')

def validate_visible_budget_protocol(models, plan=None):
    """Fail closed until V3 feasibility is measured and budgets are frozen."""
    protocol = models.get('visible_budget_protocol_v3')
    if not protocol:
        return
    feasibility = protocol.get('preflight_budget_feasibility', {})
    if feasibility.get('preflight_status') != 'PASS':
        raise RuntimeError('CONFIG_BLOCK: VISIBLE_BUDGET_PROTOCOL_V3 preflight is not complete')
    if (feasibility.get('sample_design_status') != 'FROZEN' or
            not isinstance(feasibility.get('sample_size_per_candidate'), int) or
            feasibility.get('sample_size_per_candidate', 0) <= 0 or
            not feasibility.get('preflight_histories_path')):
        raise RuntimeError('CONFIG_BLOCK: VISIBLE_BUDGET_PROTOCOL_V3 preflight sample design is not frozen')
    if not feasibility.get('formal_budgets_frozen'):
        raise RuntimeError('CONFIG_BLOCK: VISIBLE_BUDGET_PROTOCOL_V3 formal budgets are not frozen')
    selected = feasibility.get('selected_formal_budgets_native_tokens')
    if not isinstance(selected, list) or len(selected) < 2:
        raise RuntimeError('CONFIG_BLOCK: at least two qualified formal budgets must be frozen')
    selected = [int(x) for x in selected]
    min_rate = float(feasibility.get('minimum_natural_compliance_rate', 0.90))
    rates = feasibility.get('observed_natural_compliance_rates', {})
    if any(float(rates.get(str(budget), -1)) < min_rate for budget in selected):
        raise RuntimeError('CONFIG_BLOCK: a frozen budget does not meet the natural-compliance threshold')
    top_level = [int(x) for x in models.get('final_memory_budgets_native_tokens', [])]
    role_level = [int(x) for x in models.get('compressor', {}).get('final_memory_calls', {}).get('target_memory_tokens', [])]
    metadata_level = [int(x) for x in models.get('reasoning_budget_amendment', {}).get('C1_compressor', {}).get('final_memory_calls', {}).get('target_memory_tokens', [])]
    if top_level != selected or role_level != selected or metadata_level != selected:
        raise RuntimeError('CONFIG_BLOCK: frozen V3 budgets disagree across configuration fields')
    if plan is not None:
        planned = [int(x) for x in plan.get('profile', {}).get('budgets', [])]
        if sorted(planned) != sorted(selected):
            raise RuntimeError('CONFIG_BLOCK: P0 plan budgets do not match the frozen V3 budgets')
    artifact = feasibility.get('preflight_artifact')
    if not artifact or not (ROOT / artifact).is_file():
        raise RuntimeError('CONFIG_BLOCK: preflight evidence artifact is missing')
    artifact_hash = feasibility.get('preflight_artifact_sha256')
    if not artifact_hash or hashlib.sha256((ROOT / artifact).read_bytes()).hexdigest() != artifact_hash:
        raise RuntimeError('CONFIG_BLOCK: preflight evidence artifact hash is missing or mismatched')

def compressor_budget_compliance(result, visible_tokens, target, models):
    """Classify the final visible body independently from memory utility."""
    if visible_tokens > int(target):
        return 'BUDGET_NONCOMPLIANT' if models.get('visible_budget_protocol_v3') else 'OVER_BUDGET'
    accepted = accepted_finish_reasons(models)
    if result.get('finish_reason') not in accepted or not result.get('text', '').strip():
        return 'TECHNICAL_INVALID'
    return 'COMPLIANT'

def reasoning_tokens_or_estimate(result, visible_native_tokens):
    if result.get('mock'):
        return None, 'NOT_APPLICABLE_MOCK'
    reported = result.get('reasoning_tokens')
    if isinstance(reported, (int, float)) and reported >= 0:
        return int(reported), 'PROVIDER_REPORTED'
    completion = result.get('provider_output_tokens')
    if isinstance(completion, (int, float)) and completion >= 0:
        return max(0, int(completion) - int(visible_native_tokens)), 'ESTIMATED_COMPLETION_MINUS_VISIBLE_NATIVE'
    return None, 'UNAVAILABLE'

def actual_compression_rate(original_native_tokens, visible_native_tokens):
    if not original_native_tokens:
        return None
    return 1.0 - (int(visible_native_tokens) / int(original_native_tokens))


def accepted_finish_reasons(models):
    configured = models.get('truncation', {}).get('accepted_finish_reasons')
    return tuple(configured or ('stop', 'eos'))


def compression_maxout(target, models):
    desired=int(target*1.25)+32
    floor=int(models['compressor'].get('provider_min_output_tokens',0))
    return min(max(desired,floor),models['compressor']['max_visible_output_tokens'])

def validate_reader_response(result, tok, visible_limit, accepted_reasons=('stop', 'eos'), finish_reason_length='TECHNICAL_INVALID'):
    obj, error = parse_answer(result['text'])
    if result.get('finish_reason') not in accepted_reasons:
        error=finish_reason_length if result.get('finish_reason') == 'length' else 'reader_non_natural_stop'
    answer_tokens = tok.count(obj['answer']) if obj else None
    visible_response_tokens = tok.count(result['text'])
    if error is None and answer_tokens > visible_limit:
        error='protocol_violation_answer_over_limit'
    if error is None and (not isinstance(obj.get('evidence_ids'), list) or
                          any(not isinstance(item, str) for item in obj['evidence_ids'])):
        error='invalid_reader_evidence_schema'
    return obj, error, answer_tokens, visible_response_tokens

def reader_retry_prompt(reader, tokenizer_name, visible_limit):
    return reader + (f'\nPROTOCOL RETRY: return exactly one JSON object and no markdown or explanation. '
                     f'The answer string must be at most {visible_limit} tokens under {tokenizer_name}; '
                     'keep it concise, and do not include reasoning in the answer field.')

def run(histories_path,queries_path,plan_path,models_path,out,mock=False,limit=None):
    plan=load_json(plan_path);models=load_model_config(models_path)
    validate_primary_reasoning_policy(models)
    validate_visible_budget_protocol(models,plan)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    memory_tok=Tokenizer(models.get('compressor', {}).get('tokenizer', plan['profile']['tokenizer']))
    reader_tok=Tokenizer(models.get('reader', {}).get('tokenizer', memory_tok.name))
    reader_effort, reader_calibration_status = reader_reasoning_contract(models.get('reader', {}))
    accepted_reasons=accepted_finish_reasons(models)
    histories=read_jsonl(histories_path)
    if limit is not None:histories=histories[:limit]
    manifest={'plan_hash':digest(plan),'model_config_hash':digest(models),'model_config_source':str(Path(models_path).resolve()),'model_config_source_sha256':hashlib.sha256(Path(models_path).read_bytes()).hexdigest(),'history_hash':digest(histories),'query_file_hash':hashlib.sha256(Path(queries_path).read_bytes()).hexdigest(),'mock':mock,'tokenizer':memory_tok.name,'reader_tokenizer':reader_tok.name,'r1_reasoning_effort':reader_effort,'r1_calibration_status':reader_calibration_status,'accepted_finish_reasons':list(accepted_reasons),'memory_definition':models.get('memory_definition'),'reasoning_budget_amendment':models.get('reasoning_budget_amendment'),'visible_budget_protocol_v3':models.get('visible_budget_protocol_v3'),'staged_calls':models.get('staged_calls'),'opencode_session_id':models.get('opencode_session_id') or f'iclr-memory-pilot/{out.name}','code_hash':digest({p.name:p.read_text() for p in sorted((ROOT/'scripts').glob('*.py'))}),'prompt_hashes':{p.name:digest(p.read_text()) for p in sorted((ROOT/'prompts').glob('*.txt'))}}
    # Query bytes are hashed for provenance, not parsed or inserted into writing calls.
    if (out/'run_manifest.json').exists() and load_json(out/'run_manifest.json')!=manifest:
        raise RuntimeError('Output directory belongs to a different experiment; use a new directory')
    write_json(out/'run_manifest.json',manifest)
    compressor_provider=Provider(models,memory_tok,out,mock=mock)
    reader_provider=Provider(models,reader_tok,out,mock=mock)
    comp=(ROOT/models.get('compressor_prompt','prompts/compress_neutral.txt')).read_text()
    reader=(ROOT/'prompts/reader.txt').read_text()
    memories=[]
    steps_seen={digest([r['history_id'],r['arm'],r['budget'],r['replicate'],r['step']]) for r in read_jsonl(out/'steps.jsonl')} if (out/'steps.jsonl').exists() else set()
    # No questions are parsed until ALL writing is complete.
    for h in histories:
        ht=public_text(h)
        original_history_native_tokens=memory_tok.count(ht)
        for cell in plan['cells']:
            session_scope=[h['history_id'],cell['arm'],cell['budget'],cell['replicate']]
            current='' if cell['arm'] in ('no_memory','oracle') else ht;valid=True;reason=None
            last_compression_metrics={'reasoning_tokens_or_native_estimate':None,'reasoning_token_measurement':'NOT_APPLICABLE_NO_COMPRESSOR_CALL','visible_native_tokens':None,'total_completion_tokens':None,'finish_reason':None,'budget_compliance':'NOT_APPLICABLE','actual_compression_rate':None,'actual_compression_rate_definition':'1 - visible_native_tokens / original_history_native_tokens','original_history_native_tokens':original_history_native_tokens}
            stage_count=len(cell['targets'])
            for j,target in enumerate(cell['targets'],1):
                compressor_cfg=models['compressor']
                validate_compression_target(compressor_cfg,target,j,stage_count)
                if j==1 and target>=memory_tok.count(current):
                    valid=False;reason='intermediate_budget_not_compression';break
                user='BEGIN_MEMORY_DATA\n'+current+'\nEND_MEMORY_DATA'
                maxout=compression_maxout(target,models)
                if maxout<target and not mock:
                    valid=False;reason='output_cap_too_small';break
                try:
                    provider_budget=completion_budget_for(compressor_cfg,target,j,stage_count)
                    result=compressor_provider.call('compressor',comp.format(budget=target,tokenizer=memory_tok.name),user,maxout,cell['replicate'],mock_text=memory_tok.cap(current,target),namespace=cell['targets'][:j],request_metadata={'native_memory_tokens':memory_tok.count(current),'phase':plan['profile'].get('name'),'completion_budget_class':'final' if j==stage_count else 'intermediate','max_completion_tokens':provider_budget,'reasoning_effort':compressor_cfg.get('reasoning_effort',compressor_cfg.get('extra_parameters',{}).get('reasoning_effort')),'memory_definition':'final_memory_body_only','include_reasoning_tokens':False,'include_think_blocks':False,'fresh_request_each_stage':True,'reasoning_forwarded_to_next_stage':False,'experimental_path':session_scope+['compressor']},provider_output_budget=provider_budget,visible_output_limit=target)
                except Exception as e:
                    failure={'history_id':h['history_id'],**cell,'step':j,'error':str(e)}
                    failure.update(getattr(e,'metadata',{}))
                    failure.setdefault('reasoning_tokens_or_native_estimate',failure.get('reasoning_tokens'))
                    failure.setdefault('reasoning_token_measurement','PROVIDER_REPORTED' if failure.get('reasoning_tokens') is not None else 'UNAVAILABLE')
                    failure.setdefault('visible_native_tokens',None)
                    failure.setdefault('total_completion_tokens',failure.get('provider_output_tokens'))
                    failure.setdefault('finish_reason',failure.get('finish_reason'))
                    failure.setdefault('budget_compliance','TECHNICAL_INVALID')
                    failure.setdefault('actual_compression_rate',None)
                    failure.setdefault('actual_compression_rate_definition','1 - visible_native_tokens / original_history_native_tokens')
                    failure.setdefault('original_history_native_tokens',original_history_native_tokens)
                    append_jsonl(out/'failures.jsonl',failure)
                    # Stop on budget/authorization/network failures instead of repeated full-grid retries.
                    write_json(out/'cost_summary.json',compressor_provider.ledger.summary())
                    raise
                raw=result['text'];raw_tokens=memory_tok.count(raw);capped=raw
                bad_finish=result.get('finish_reason') not in accepted_reasons
                budget_compliance=compressor_budget_compliance(result,raw_tokens,target,models)
                reasoning_metric,reasoning_source=reasoning_tokens_or_estimate(result,raw_tokens)
                compression_rate=actual_compression_rate(original_history_native_tokens,raw_tokens)
                last_compression_metrics={'reasoning_tokens_or_native_estimate':reasoning_metric,'reasoning_token_measurement':reasoning_source,'visible_native_tokens':raw_tokens,'total_completion_tokens':result.get('total_completion_tokens',result.get('provider_output_tokens')),'finish_reason':result.get('finish_reason'),'budget_compliance':budget_compliance,'actual_compression_rate':compression_rate,'actual_compression_rate_definition':'1 - visible_native_tokens / original_history_native_tokens','original_history_native_tokens':original_history_native_tokens}
                key=digest([h['history_id'],cell['arm'],cell['budget'],cell['replicate'],j])
                row={'history_id':h['history_id'],'group_id':h.get('group_id',h['history_id']),'source':h['source'],**cell,'step':j,'target':target,'input_tokens':memory_tok.count(current),'raw_output_tokens':raw_tokens,'memory_tokens':memory_tok.count(capped),'native_memory_tokens':memory_tok.count(capped),'full_request_native_tokens':result.get('full_request_native_tokens'),'provider_input_tokens':result.get('provider_input_tokens'),'provider_output_tokens':result.get('provider_output_tokens'),'cache_read_tokens':result.get('cache_read_tokens',0),'full_HF_tokenizer_revision':result.get('full_HF_tokenizer_revision'),'GET_v1_models_sha256':result.get('GET_v1_models_sha256'),'run_start_utc':result.get('run_start_utc'),'opencode_session_id':result.get('opencode_session_id'),'provider_model_id':result.get('provider_model_id'),'provider_name':result.get('provider_name'),'reasoning_effort':result.get('reasoning_effort'),'provider_output_budget_tokens':result.get('provider_output_budget_tokens',provider_budget),'max_completion_tokens':result.get('max_completion_tokens',provider_budget),'completion_budget_class':result.get('completion_budget_class'),'visible_output_limit_tokens':result.get('visible_output_limit_tokens'),'returned_model_field':result.get('returned_model_field'),'think_blocks_removed':result.get('think_blocks_removed',False),'reasoning_content_present':result.get('reasoning_content_present',False),'memory_definition':'final_memory_body_only','include_reasoning_tokens':False,'include_think_blocks':False,'fresh_request_each_stage':True,'reasoning_forwarded_to_next_stage':False,'protocol_violation':'TECHNICAL_INVALID' if bad_finish else None,'clip_tokens':0,'input_hash':digest(current),'memory_hash':digest(capped),'raw_output':raw,'memory_text':capped,'model':result['model'],'request_key':result['request_key'],'finish_reason':result['finish_reason'],'mock':mock,'cache_hit':result['cache_hit'],'usage':result['usage']}
                row.update(last_compression_metrics)
                if budget_compliance=='BUDGET_NONCOMPLIANT':row['protocol_violation']='BUDGET_NONCOMPLIANT'
                if key not in steps_seen:append_jsonl(out/'steps.jsonl',row);steps_seen.add(key)
                if budget_compliance!='COMPLIANT':
                    valid=False
                    if models.get('visible_budget_protocol_v3'):
                        reason=budget_compliance if budget_compliance=='BUDGET_NONCOMPLIANT' else ('TECHNICAL_INVALID' if bad_finish else 'empty_visible_memory')
                    else:
                        reason='TECHNICAL_INVALID' if bad_finish else ('empty_visible_memory' if not capped.strip() else 'compressor_output_over_limit')
                    current=raw
                    if models.get('stop_on_compression_protocol_failure',False) and reason!='BUDGET_NONCOMPLIANT':
                        write_json(out/'cost_summary.json',compressor_provider.ledger.summary())
                        append_jsonl(out/'failures.jsonl',{'history_id':h['history_id'],**cell,'step':j,'error':reason,'budget_compliance':budget_compliance,'request_key':result['request_key']})
                        raise RuntimeError('COMPRESSION_PROTOCOL_BLOCK: '+reason)
                    if reason=='BUDGET_NONCOMPLIANT':
                        append_jsonl(out/'failures.jsonl',{'history_id':h['history_id'],**cell,'step':j,'error':reason,'budget_compliance':budget_compliance,'visible_native_tokens':raw_tokens,'target_memory_tokens':int(target),'request_key':result['request_key']})
                    break
                current=capped
            memories.append({'history_id':h['history_id'],'group_id':h.get('group_id',h['history_id']),'source':h['source'],**cell,'text':current,'n_tokens':memory_tok.count(current),'memory_hash':digest(current),'valid':valid,'invalid_reason':reason,'mock':mock,'memory_definition':'final_memory_body_only','include_reasoning_tokens':False,'include_think_blocks':False,**last_compression_metrics})
    write_json(out/'memories.json',memories)
    # Reading begins only after memory outputs are finalized.
    queries=read_jsonl(queries_path);by_h={}
    for q in queries:by_h.setdefault(q['history_id'],[]).append(q)
    scores_seen={r['score_id'] for r in read_jsonl(out/'scores.jsonl')} if (out/'scores.jsonl').exists() else set()
    for m in memories:
        for q in by_h.get(m['history_id'],[]):
            sid=digest([m['history_id'],m['arm'],m['budget'],m['replicate'],q['question_id'],models['reader']['model']])
            if sid in scores_seen:continue
            read_text='\n'.join(q.get('gold_evidence',[])) if m['arm']=='oracle' else m['text']
            meta={k:m[k] for k in ['history_id','group_id','source','arm','budget','replicate','memory_hash','mock']}
            meta.update({k:m.get(k) for k in ['reasoning_tokens_or_native_estimate','reasoning_token_measurement','visible_native_tokens','total_completion_tokens','finish_reason','budget_compliance','actual_compression_rate','actual_compression_rate_definition','original_history_native_tokens']})
            meta['memory_hash']=digest(read_text)
            base={**meta,'question_id':q['question_id'],'question_type':q['question_type'],'score_id':sid,'reader':models['reader']['model'],'reader_tokenizer':reader_tok.name,'r1_reasoning_effort':reader_effort,'r1_calibration_status':reader_calibration_status,'n_tokens':reader_tok.count(read_text),'memory_definition':'final_memory_body_only','include_reasoning_tokens':False,'include_think_blocks':False}
            if not m['valid']:
                append_jsonl(out/'scores.jsonl',{**base,'status':m['invalid_reason'],'score':None});continue
            date_hint=('Question date: '+str(q['question_date'])+'\n') if q.get('question_date') else ''
            user='MEMORY:\n'+read_text+'\n\nQUESTION:\n'+date_hint+q['question']
            reader_cfg=models['reader']
            visible_limit=int(reader_cfg.get('visible_output_limit_tokens',reader_cfg['max_visible_output_tokens']))
            provider_budget=int(reader_cfg.get('max_completion_tokens',reader_cfg.get('provider_output_budget_tokens',visible_limit)))
            retry_limit=int(reader_cfg.get('protocol_retry_limit',1))
            if retry_limit > 1:
                raise RuntimeError('CONFIG_BLOCK: protocol_retry_limit cannot exceed one')
            attempts=[]
            for retry_index in range(retry_limit+1):
                system_prompt=reader if retry_index == 0 else reader_retry_prompt(reader,reader_tok.name,visible_limit)
                result=reader_provider.call('reader',system_prompt,user,visible_limit,m['replicate'],namespace=[sid,'reader_retry',retry_index],request_metadata={'native_memory_tokens':m['n_tokens'],'phase':plan['profile'].get('name'),'retry_index':retry_index,'max_completion_tokens':provider_budget,'reasoning_effort':reader_effort,'memory_definition':'final_memory_body_only','include_reasoning_tokens':False,'include_think_blocks':False,'fresh_request_each_stage':True,'reasoning_forwarded_to_next_stage':False,'experimental_path':[m['history_id'],m['arm'],m['budget'],m['replicate'],'reader',q['question_id']]},provider_output_budget=provider_budget,visible_output_limit=visible_limit)
                obj,error,answer_tokens,visible_response_tokens=validate_reader_response(result,reader_tok,visible_limit,accepted_reasons,models.get('truncation',{}).get('finish_reason_length','TECHNICAL_INVALID'))
                attempt={'result':result,'obj':obj,'error':error,'answer_tokens':answer_tokens,'visible_response_tokens':visible_response_tokens,'retry_index':retry_index}
                attempts.append(attempt)
                attempt_base={**base,'native_memory_tokens':result.get('native_memory_tokens'),'full_request_native_tokens':result.get('full_request_native_tokens'),'provider_input_tokens':result.get('provider_input_tokens'),'provider_output_tokens':result.get('provider_output_tokens'),'cache_read_tokens':result.get('cache_read_tokens',0),'full_HF_tokenizer_revision':result.get('full_HF_tokenizer_revision'),'GET_v1_models_sha256':result.get('GET_v1_models_sha256'),'run_start_utc':result.get('run_start_utc'),'opencode_session_id':result.get('opencode_session_id'),'provider_model_id':result.get('provider_model_id'),'provider_name':result.get('provider_name'),'reasoning_effort':result.get('reasoning_effort'),'provider_output_budget_tokens':result.get('provider_output_budget_tokens'),'max_completion_tokens':result.get('max_completion_tokens',provider_budget),'completion_budget_class':result.get('completion_budget_class'),'visible_output_limit_tokens':result.get('visible_output_limit_tokens'),'returned_model_field':result.get('returned_model_field'),'think_blocks_removed':result.get('think_blocks_removed',False),'reasoning_content_present':result.get('reasoning_content_present',False),'final_visible_answer_tokens':answer_tokens,'visible_response_tokens':visible_response_tokens,'retry_index':retry_index,'protocol_error':error}
                append_jsonl(out/'predictions.jsonl',{**attempt_base,'question':q['question'],'response':result,'parsed':obj})
                if error is None:
                    break
            final=attempts[-1];result=final['result'];obj=final['obj'];error=final['error'];answer_tokens=final['answer_tokens'];visible_response_tokens=final['visible_response_tokens']
            score=None
            if not error and q.get('scoring')=='exact':score=exact_score(obj['answer'],q['answers'])
            status=error or ('valid' if score is not None else 'semantic_grading_required')
            base.update({'native_memory_tokens':result.get('native_memory_tokens'),'full_request_native_tokens':result.get('full_request_native_tokens'),'provider_input_tokens':result.get('provider_input_tokens'),'provider_output_tokens':result.get('provider_output_tokens'),'cache_read_tokens':result.get('cache_read_tokens',0),'full_HF_tokenizer_revision':result.get('full_HF_tokenizer_revision'),'GET_v1_models_sha256':result.get('GET_v1_models_sha256'),'run_start_utc':result.get('run_start_utc'),'opencode_session_id':result.get('opencode_session_id'),'provider_model_id':result.get('provider_model_id'),'returned_model_field':result.get('returned_model_field')})
            base.update({'provider_name':result.get('provider_name'),'reasoning_effort':result.get('reasoning_effort'),'provider_output_budget_tokens':result.get('provider_output_budget_tokens'),'max_completion_tokens':result.get('max_completion_tokens',provider_budget),'completion_budget_class':result.get('completion_budget_class'),'visible_output_limit_tokens':result.get('visible_output_limit_tokens'),'think_blocks_removed':result.get('think_blocks_removed',False),'reasoning_content_present':result.get('reasoning_content_present',False),'final_visible_answer_tokens':answer_tokens,'visible_response_tokens':visible_response_tokens,'protocol_retry_count':len(attempts)-1,'protocol_attempt_errors':[a['error'] for a in attempts]})
            append_jsonl(out/'scores.jsonl',{**base,'status':status,'score':score,'answer':obj['answer'] if obj else None,'scoring':q.get('scoring')})
            scores_seen.add(sid)
    write_json(out/'cost_summary.json',reader_provider.ledger.summary())
    print(f'Completed reference run: {out}; mock={mock}. Evidence/length/novelty audits are still required.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--histories',required=True);p.add_argument('--queries',required=True);p.add_argument('--plan',required=True);p.add_argument('--models',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--mock',action='store_true');group.add_argument('--live',action='store_true')
    a=p.parse_args();run(a.histories,a.queries,a.plan,a.models,a.out,mock=a.mock,limit=a.limit)
