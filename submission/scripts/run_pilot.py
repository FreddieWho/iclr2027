"""Reference static compaction runner: run all memories first, then open questions."""
from __future__ import annotations
import argparse
from pathlib import Path
from core import *

def compression_maxout(target, models):
    desired=int(target*1.25)+32
    floor=int(models['compressor'].get('provider_min_output_tokens',0))
    return min(max(desired,floor),models['compressor']['max_visible_output_tokens'])

def validate_reader_response(result, tok, visible_limit):
    obj, error = parse_answer(result['text'])
    if result.get('finish_reason') not in ('stop','eos'):
        error='reader_non_natural_stop'
    answer_tokens = tok.count(obj['answer']) if obj else None
    visible_response_tokens = tok.count(result['text'])
    if error is None and answer_tokens > visible_limit:
        error='protocol_violation_answer_over_limit'
    return obj, error, answer_tokens, visible_response_tokens

def reader_retry_prompt(reader, tokenizer_name, visible_limit):
    return reader + (f'\nPROTOCOL RETRY: return exactly one JSON object and no markdown or explanation. '
                     f'The answer string must be at most {visible_limit} tokens under {tokenizer_name}; '
                     'keep it concise, and do not include reasoning in the answer field.')

def run(histories_path,queries_path,plan_path,models_path,out,mock=False,limit=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plan=load_json(plan_path);models=load_json(models_path);tok=Tokenizer(plan['profile']['tokenizer'])
    histories=read_jsonl(histories_path)
    if limit is not None:histories=histories[:limit]
    manifest={'plan_hash':digest(plan),'model_config_hash':digest(models),'history_hash':digest(histories),'query_file_hash':__import__('hashlib').sha256(Path(queries_path).read_bytes()).hexdigest(),'mock':mock,'tokenizer':tok.name,'opencode_session_id':models.get('opencode_session_id') or f'iclr-memory-pilot/{out.name}','code_hash':digest({p.name:p.read_text() for p in sorted((ROOT/'scripts').glob('*.py'))}),'prompt_hashes':{p.name:digest(p.read_text()) for p in sorted((ROOT/'prompts').glob('*.txt'))}}
    # Query bytes are hashed for provenance, not parsed or inserted into writing calls.
    if (out/'run_manifest.json').exists() and load_json(out/'run_manifest.json')!=manifest:
        raise RuntimeError('Output directory belongs to a different experiment; use a new directory')
    write_json(out/'run_manifest.json',manifest)
    provider=Provider(models,tok,out,mock=mock)
    comp=(ROOT/'prompts/compress_neutral.txt').read_text()
    reader=(ROOT/'prompts/reader.txt').read_text()
    memories=[]
    steps_seen={digest([r['history_id'],r['arm'],r['budget'],r['replicate'],r['step']]) for r in read_jsonl(out/'steps.jsonl')} if (out/'steps.jsonl').exists() else set()
    # No questions are parsed until ALL writing is complete.
    for h in histories:
        ht=public_text(h)
        for cell in plan['cells']:
            current='' if cell['arm'] in ('no_memory','oracle') else ht;valid=True;reason=None
            for j,target in enumerate(cell['targets'],1):
                if j==1 and target>=tok.count(current):
                    valid=False;reason='intermediate_budget_not_compression';break
                user='BEGIN_MEMORY_DATA\n'+current+'\nEND_MEMORY_DATA'
                maxout=compression_maxout(target,models)
                if maxout<target and not mock:
                    valid=False;reason='output_cap_too_small';break
                try:
                    compressor_cfg=models['compressor']
                    provider_budget=int(compressor_cfg.get('provider_output_budget_tokens',maxout))
                    result=provider.call('compressor',comp.format(budget=target,tokenizer=tok.name),user,maxout,cell['replicate'],mock_text=tok.cap(current,target),namespace=cell['targets'][:j],request_metadata={'native_memory_tokens':tok.count(current),'phase':plan['profile'].get('name')},provider_output_budget=provider_budget,visible_output_limit=target)
                except Exception as e:
                    failure={'history_id':h['history_id'],**cell,'step':j,'error':str(e)}
                    failure.update(getattr(e,'metadata',{}))
                    append_jsonl(out/'failures.jsonl',failure)
                    # Stop on budget/authorization/network failures instead of repeated full-grid retries.
                    write_json(out/'cost_summary.json',provider.ledger.summary())
                    raise
                raw=result['text'];raw_tokens=tok.count(raw);capped=raw
                bad_finish=result.get('finish_reason') not in ('stop','eos')
                key=digest([h['history_id'],cell['arm'],cell['budget'],cell['replicate'],j])
                row={'history_id':h['history_id'],'group_id':h.get('group_id',h['history_id']),'source':h['source'],**cell,'step':j,'target':target,'input_tokens':tok.count(current),'raw_output_tokens':raw_tokens,'memory_tokens':tok.count(capped),'native_memory_tokens':tok.count(capped),'full_request_native_tokens':result.get('full_request_native_tokens'),'provider_input_tokens':result.get('provider_input_tokens'),'provider_output_tokens':result.get('provider_output_tokens'),'cache_read_tokens':result.get('cache_read_tokens',0),'full_HF_tokenizer_revision':result.get('full_HF_tokenizer_revision'),'GET_v1_models_sha256':result.get('GET_v1_models_sha256'),'run_start_utc':result.get('run_start_utc'),'opencode_session_id':result.get('opencode_session_id'),'provider_model_id':result.get('provider_model_id'),'provider_name':result.get('provider_name'),'reasoning_effort':result.get('reasoning_effort'),'provider_output_budget_tokens':result.get('provider_output_budget_tokens'),'visible_output_limit_tokens':result.get('visible_output_limit_tokens'),'returned_model_field':result.get('returned_model_field'),'clip_tokens':0,'input_hash':digest(current),'memory_hash':digest(capped),'raw_output':raw,'memory_text':capped,'model':result['model'],'request_key':result['request_key'],'finish_reason':result['finish_reason'],'mock':mock,'cache_hit':result['cache_hit'],'usage':result['usage']}
                if key not in steps_seen:append_jsonl(out/'steps.jsonl',row);steps_seen.add(key)
                if bad_finish or not capped.strip() or raw_tokens>target:
                    valid=False;reason=('truncated_or_empty_output' if bad_finish or not capped.strip() else 'compressor_output_over_limit');break
                current=capped
            memories.append({'history_id':h['history_id'],'group_id':h.get('group_id',h['history_id']),'source':h['source'],**cell,'text':current,'n_tokens':tok.count(current),'memory_hash':digest(current),'valid':valid,'invalid_reason':reason,'mock':mock})
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
            meta['memory_hash']=digest(read_text)
            base={**meta,'question_id':q['question_id'],'question_type':q['question_type'],'score_id':sid,'reader':models['reader']['model'],'n_tokens':tok.count(read_text)}
            if not m['valid']:
                append_jsonl(out/'scores.jsonl',{**base,'status':m['invalid_reason'],'score':None});continue
            date_hint=('Question date: '+str(q['question_date'])+'\n') if q.get('question_date') else ''
            user='MEMORY:\n'+read_text+'\n\nQUESTION:\n'+date_hint+q['question']
            reader_cfg=models['reader']
            visible_limit=int(reader_cfg.get('visible_output_limit_tokens',reader_cfg['max_visible_output_tokens']))
            provider_budget=int(reader_cfg.get('provider_output_budget_tokens',visible_limit))
            retry_limit=int(reader_cfg.get('protocol_retry_limit',1))
            if retry_limit > 1:
                raise RuntimeError('CONFIG_BLOCK: protocol_retry_limit cannot exceed one')
            attempts=[]
            for retry_index in range(retry_limit+1):
                system_prompt=reader if retry_index == 0 else reader_retry_prompt(reader,tok.name,visible_limit)
                result=provider.call('reader',system_prompt,user,visible_limit,m['replicate'],namespace=[sid,'reader_retry',retry_index],request_metadata={'native_memory_tokens':tok.count(read_text),'phase':plan['profile'].get('name'),'retry_index':retry_index},provider_output_budget=provider_budget,visible_output_limit=visible_limit)
                obj,error,answer_tokens,visible_response_tokens=validate_reader_response(result,tok,visible_limit)
                attempt={'result':result,'obj':obj,'error':error,'answer_tokens':answer_tokens,'visible_response_tokens':visible_response_tokens,'retry_index':retry_index}
                attempts.append(attempt)
                attempt_base={**base,'native_memory_tokens':result.get('native_memory_tokens'),'full_request_native_tokens':result.get('full_request_native_tokens'),'provider_input_tokens':result.get('provider_input_tokens'),'provider_output_tokens':result.get('provider_output_tokens'),'cache_read_tokens':result.get('cache_read_tokens',0),'full_HF_tokenizer_revision':result.get('full_HF_tokenizer_revision'),'GET_v1_models_sha256':result.get('GET_v1_models_sha256'),'run_start_utc':result.get('run_start_utc'),'opencode_session_id':result.get('opencode_session_id'),'provider_model_id':result.get('provider_model_id'),'provider_name':result.get('provider_name'),'reasoning_effort':result.get('reasoning_effort'),'provider_output_budget_tokens':result.get('provider_output_budget_tokens'),'visible_output_limit_tokens':result.get('visible_output_limit_tokens'),'returned_model_field':result.get('returned_model_field'),'final_visible_answer_tokens':answer_tokens,'visible_response_tokens':visible_response_tokens,'retry_index':retry_index,'protocol_error':error}
                append_jsonl(out/'predictions.jsonl',{**attempt_base,'question':q['question'],'response':result,'parsed':obj})
                if error is None:
                    break
            final=attempts[-1];result=final['result'];obj=final['obj'];error=final['error'];answer_tokens=final['answer_tokens'];visible_response_tokens=final['visible_response_tokens']
            score=None
            if not error and q.get('scoring')=='exact':score=exact_score(obj['answer'],q['answers'])
            status=error or ('valid' if score is not None else 'semantic_grading_required')
            base.update({'native_memory_tokens':result.get('native_memory_tokens'),'full_request_native_tokens':result.get('full_request_native_tokens'),'provider_input_tokens':result.get('provider_input_tokens'),'provider_output_tokens':result.get('provider_output_tokens'),'cache_read_tokens':result.get('cache_read_tokens',0),'full_HF_tokenizer_revision':result.get('full_HF_tokenizer_revision'),'GET_v1_models_sha256':result.get('GET_v1_models_sha256'),'run_start_utc':result.get('run_start_utc'),'opencode_session_id':result.get('opencode_session_id'),'provider_model_id':result.get('provider_model_id'),'returned_model_field':result.get('returned_model_field')})
            base.update({'provider_name':result.get('provider_name'),'reasoning_effort':result.get('reasoning_effort'),'provider_output_budget_tokens':result.get('provider_output_budget_tokens'),'visible_output_limit_tokens':result.get('visible_output_limit_tokens'),'final_visible_answer_tokens':answer_tokens,'visible_response_tokens':visible_response_tokens,'protocol_retry_count':len(attempts)-1,'protocol_attempt_errors':[a['error'] for a in attempts]})
            append_jsonl(out/'scores.jsonl',{**base,'status':status,'score':score,'answer':obj['answer'] if obj else None,'scoring':q.get('scoring')})
            scores_seen.add(sid)
    write_json(out/'cost_summary.json',provider.ledger.summary())
    print(f'Completed reference run: {out}; mock={mock}. Evidence/length/novelty audits are still required.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--histories',required=True);p.add_argument('--queries',required=True);p.add_argument('--plan',required=True);p.add_argument('--models',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--mock',action='store_true');group.add_argument('--live',action='store_true')
    a=p.parse_args();run(a.histories,a.queries,a.plan,a.models,a.out,mock=a.mock,limit=a.limit)
