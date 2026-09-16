"""V4 route-selection pilot. C1 receives histories only; readers run after freeze.

All live requests go through the repository's OpenCode Go Provider adapter. A
single V4 ledger guards project and per-model caps; the Provider sidecar ledger
is retained as an independent usage receipt. Hidden reasoning text is never
written to disk.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from core import (
    ROOT, Provider, Tokenizer, append_jsonl, digest, exact_score,
    final_visible_body, load_json, read_jsonl, write_json,
)

RUNTIME_DEFAULT = ROOT / 'configs/v4_frozen_runtime.json'
V4_ROOT = ROOT / 'work/v4_fast_decision_20260913'
PROMPT_COMPRESS = ROOT / 'prompts/compress_v4_route_selection_compact_v2.txt'
PROMPT_READER_BATCH = ROOT / 'prompts/reader_batch_v4.txt'
PROMPT_READER_BATCH_RETRY = ROOT / 'prompts/reader_batch_retry_v4.txt'
PROMPT_READER_SINGLE = ROOT / 'prompts/reader.txt'
READER_PROMPT_PATHS = {
    'batch': PROMPT_READER_BATCH,
    'batch_retry': PROMPT_READER_BATCH_RETRY,
    'single': PROMPT_READER_SINGLE,
}
NATURAL_FINISH = 'stop'


def runner_code_sha256() -> str:
    return file_sha256(Path(__file__).resolve())


def reader_prompt_hashes() -> dict[str, str]:
    return {name: file_sha256(path) for name, path in READER_PROMPT_PATHS.items()}


def resolve_compressor_prompt(runtime: dict, runtime_path: str | Path = RUNTIME_DEFAULT,
                              allow_legacy: bool = False) -> tuple[Path, str, str]:
    """Resolve the actual compressor template from the runtime (F8).

    Returns (prompt_path, sha256, resolution) where resolution is
    'runtime_explicit' or 'legacy_global_fallback'. Runtimes without an
    explicit compressor_C1.prompt_path fail closed unless allow_legacy is set,
    so a historical runtime can never silently run under the newest prompt.
    """
    declared = runtime.get('compressor_C1', {}).get('prompt_path')
    if declared:
        cand = Path(declared)
        if not cand.is_absolute():
            cand = ROOT / cand
        if not cand.is_file():
            raise ValueError(f'compressor prompt_path missing on disk: {declared}')
        return cand, file_sha256(cand), 'runtime_explicit'
    if allow_legacy:
        return PROMPT_COMPRESS, file_sha256(PROMPT_COMPRESS), 'legacy_global_fallback'
    raise ValueError(
        'Runtime lacks compressor_C1.prompt_path; pass the continuation runtime via '
        '--runtime (e.g. configs/v4_compact_v2_continuation_runtime_20260914.json) '
        'or opt into historical reads with --allow-legacy-prompt')

PATHS = ('direct', 'staged2', 'rewrite')
TYPE_MAP = {
    'atomic': 'atomic_fact',
    'update': 'temporal_update_provenance',
    'relational': 'relational_multihop',
    'constraint': 'exception_retraction_constraint',
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_runtime(path: str | Path = RUNTIME_DEFAULT) -> dict:
    runtime = load_json(path)
    if runtime.get('protocol_id') != 'memory_path_v4_fast_route_decision_20260913':
        raise ValueError('Unexpected V4 protocol id')
    if runtime.get('compressor_C1', {}).get('reasoning_effort') is not None:
        raise ValueError('V4 C1 must not send reasoning_effort')
    if runtime.get('compressor_C1', {}).get('thinking') != 'disabled':
        raise ValueError('V4 C1 thinking must be disabled')
    if runtime.get('sampling', {}).get('posthoc_truncation') is not False:
        raise ValueError('Post-hoc memory truncation is forbidden')
    return runtime


def path_targets(final_target: int) -> dict[str, list[int]]:
    if final_target <= 0:
        raise ValueError('final target must be positive')
    return {
        'direct': [final_target],
        'staged2': [2 * final_target, final_target],
        'rewrite': [final_target, final_target],
    }


def read_history_rows(path: str | Path) -> list[dict]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError('No histories found')
    seen = set()
    for row in rows:
        if not isinstance(row.get('history_id'), str) or not isinstance(row.get('text'), str):
            raise ValueError('Every history requires history_id and text')
        if row['history_id'] in seen:
            raise ValueError('Duplicate history_id')
        seen.add(row['history_id'])
    return rows


def public_history_text(history: dict) -> str:
    """Explicit allowlist: only public history text enters C1 requests."""
    text = history.get('text')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('History requires nonempty public text')
    return text


def c1_system_prompt(target: int, role: str, tokenizer_name: str,
                    template_text: str | None = None) -> str:
    template = template_text if template_text is not None else PROMPT_COMPRESS.read_text(encoding='utf-8')
    return template.format(target_role=role, target_tokens=target, tokenizer=tokenizer_name)


def expected_query_contract(queries: list[dict], questions_per_history: int = 8) -> dict:
    """Derive the frozen query contract: expected question ids and per-type counts."""
    by_history: dict[str, list[dict]] = {}
    for q in queries:
        by_history.setdefault(q['history_id'], []).append(q)
    expected_ids: dict[str, set[str]] = {}
    expected_types: dict[str, dict[str, int]] = {}
    for hid, qs in by_history.items():
        if len(qs) != questions_per_history:
            raise ValueError(f'Query contract violation for {hid}: expected '
                             f'{questions_per_history} questions, got {len(qs)}')
        qids = [q['question_id'] for q in qs]
        if len(set(qids)) != len(qids):
            raise ValueError(f'Query contract violation for {hid}: duplicate question_id')
        expected_ids[hid] = set(qids)
        counts: dict[str, int] = {}
        for q in qs:
            counts[q.get('question_type', '')] = counts.get(q.get('question_type', ''), 0) + 1
        expected_types[hid] = counts
    return {'expected_ids': expected_ids, 'expected_types': expected_types,
            'questions_per_history': questions_per_history}


def verify_frozen_score_inputs(*, compression_manifest: dict, compression_manifest_path: Path,
                               histories_path: str | Path, memories_path: str | Path) -> dict:
    """Verify frozen inputs BEFORE queries are read or any provider is built (F3).

    Checks manifest status, actual histories/memory file SHAs against the frozen
    manifest, history-id coverage, per-row memory text digests and (history,
    condition) key uniqueness. Raises on any mismatch; never trusts boolean
    flags or self-reported hashes alone.
    """
    if compression_manifest.get('status') != 'COMPLETE':
        raise RuntimeError('Refusing to score: compression manifest is not COMPLETE')
    if not compression_manifest.get('no_future_query_visible_to_compressor'):
        raise RuntimeError('Refusing to score: query-blindness flag missing in compression manifest')
    histories_path = Path(histories_path)
    memories_path = Path(memories_path)
    actual_hist_sha = file_sha256(histories_path)
    if actual_hist_sha != compression_manifest.get('histories_sha256'):
        raise ValueError('Histories file SHA does not match the frozen compression manifest')
    actual_mem_sha = file_sha256(memories_path)
    if actual_mem_sha != compression_manifest.get('memory_rows_sha256'):
        raise ValueError('Memory file SHA does not match the frozen compression manifest')
    histories = read_history_rows(histories_path)
    hist_ids = [h['history_id'] for h in histories]
    if set(hist_ids) != set(compression_manifest.get('history_ids', [])):
        raise ValueError('History IDs do not match the frozen compression manifest')
    memory_rows = read_jsonl(memories_path)
    seen_keys = set()
    for m in memory_rows:
        key = (m.get('history_id'), m.get('condition'))
        if key in seen_keys:
            raise ValueError(f'Duplicate frozen memory key: {key}')
        seen_keys.add(key)
        if digest(m.get('text', '')) != m.get('memory_hash'):
            raise ValueError(f'Frozen memory text hash mismatch for {key}: '
                             'file content differs from its recorded hash')
    return {'histories': histories, 'memory_rows': memory_rows,
            'histories_sha256': actual_hist_sha, 'memory_rows_sha256': actual_mem_sha}


def compression_user(history_text: str) -> str:
    return 'BEGIN_MEMORY_DATA\n' + history_text + '\nEND_MEMORY_DATA'


def classify_memory_output(body: str, finish_reason: str) -> str:
    """Natural-stop length is observed, not rejected against the requested target."""
    if finish_reason == 'length':
        return 'TECHNICAL_INVALID_LENGTH'
    if finish_reason != NATURAL_FINISH:
        return 'TECHNICAL_INVALID'
    return 'natural_stop' if body.strip() else 'TECHNICAL_INVALID_EMPTY'


def _uncertain_step_row(phase: str, history_id: str, condition: str, step: int,
                        step_count: int, target: int, completion_class: str,
                        provider_budget: int, input_tokens: int, prior: dict,
                        runtime: dict) -> dict:
    """Record an earlier reserved request without replaying or scoring it."""
    body = ''
    return {
        'phase':phase,'history_id':history_id,'condition':condition,
        'step':step,'step_count':step_count,'requested_target_tokens':target,
        'requested_target_is_hard_cap':False,'completion_budget_class':completion_class,
        'status':'TECHNICAL_INVALID_UNCERTAIN','memory_body':body,
        'memory_hash':digest(body),'visible_native_tokens':0,
        'reasoning_tokens_or_native_estimate':None,
        'reasoning_token_measurement':'UNAVAILABLE',
        'total_completion_tokens':None,
        'full_request_native_tokens':prior['metadata'].get('full_request_native_tokens'),
        'provider_input_tokens':None,'provider_output_tokens':None,
        'cache_read_tokens':None,'finish_reason':None,
        'provider_name':runtime['provider']['name'],
        'provider_model_id':runtime['compressor_C1']['provider_model_id'],
        'returned_model_field':None,'reasoning_effort':'NOT_SENT','thinking':'disabled',
        'provider_output_budget_tokens':provider_budget,
        'run_start_utc':prior['created_utc'],
        'full_HF_tokenizer_revision':runtime['compressor_C1']['full_HF_tokenizer_revision'],
        'GET_v1_models_sha256':runtime['provider']['models_catalog_sha256'],
        'opencode_session_id':None,'request_key':prior['request_key'],
        'cache_hit':None,'cost_usd_or_conservative':prior['accounted_usd'],
        'reasoning_content_present':None,'think_blocks_removed':False,
        'query_file_opened':False,'input_native_tokens':input_tokens,
        'unresolved_reserved_usd':prior['accounted_usd'],
        'failure_class':'UNRESOLVED_PRIOR_RESERVATION_NOT_REPLAYED',
        'failure_detail':'No response/usage receipt or provider cache; excluded from scoring.',
    }


def _core_role_config(runtime: dict, role: str, r1_effort: str | None = None) -> dict:
    provider = runtime['provider']
    source = runtime['compressor_C1'] if role == 'compressor' else (
        runtime['reader_R1'] if role == 'reader_R1' else runtime['reader_R2'])
    source = copy.deepcopy(source)
    if role == 'reader_R1':
        effort = r1_effort or source['reasoning_effort']
        source['reasoning_effort'] = effort
        extra = {'temperature': float(source.get('temperature', 0.0)), 'reasoning_effort': effort}
        output_limit = int(source['provider_output_budget_tokens'])
        tokenizer = source['visible_limit_tokenizer']
    elif role == 'reader_R2':
        effort = r1_effort or runtime['reader_R1']['reasoning_effort']
        if effort == 'MATCH_FINAL_R1_SETTING':
            raise ValueError('Resolve R2 reasoning effort after R1 calibration')
        source['reasoning_effort'] = effort
        extra = {'temperature': 0.0, 'reasoning_effort': effort}
        output_limit = int(source['provider_output_budget_tokens'])
        tokenizer = source['tokenizer']
    else:
        effort = None
        extra = {
            'thinking': {'type': 'disabled'},
            'temperature': float(source['temperature']),
            'top_p': float(source['top_p']),
        }
        output_limit = int(source['intermediate_provider_output_budget_tokens'])
        tokenizer = source['tokenizer']

    core = {
        'base_url': provider['chat_completions_url'].rsplit('/chat/completions', 1)[0],
        'chat_completions_url': provider['chat_completions_url'],
        'model': source['provider_model_id'],
        'family': source['provider_model_id'],
        'api_key_env': provider['api_key_env'],
        'tokenizer': tokenizer,
        'full_HF_tokenizer_revision': source['full_HF_tokenizer_revision'],
        'tokenizer_add_special_tokens': False,
        'models_endpoint_sha256': provider['models_catalog_sha256'],
        'context_tokens': int(source['context_tokens']),
        'max_visible_output_tokens': output_limit,
        'provider_output_budget_tokens': output_limit,
        'max_completion_tokens': output_limit,
        'output_parameter': 'max_tokens',
        'supports_seed': False,
        'reasoning_effort': effort,
        'extra_parameters': extra,
        'input_usd_per_million': float(source['input_usd_per_million']),
        'output_usd_per_million': float(source['output_usd_per_million']),
        'cache_read_usd_per_million': float(source['cache_read_usd_per_million']),
        'user_agent': provider['user_agent'],
        'input_reservation_overhead_tokens': 256,
    }
    if role == 'compressor':
        core['provider_output_budget_tokens'] = int(source['intermediate_provider_output_budget_tokens'])
        core['max_completion_tokens'] = core['provider_output_budget_tokens']
        core['max_visible_output_tokens'] = core['provider_output_budget_tokens']
    elif role == 'reader_R1':
        core['max_visible_output_tokens'] = int(source['visible_answer_limit_tokens'])
        core['visible_output_limit_tokens'] = int(source['visible_answer_limit_tokens'])
    else:
        core['max_visible_output_tokens'] = int(source['visible_answer_limit_tokens'])
        core['visible_output_limit_tokens'] = int(source['visible_answer_limit_tokens'])
    return core


def make_provider_config(runtime: dict, role: str, session_label: str,
                         r1_effort: str | None = None) -> dict:
    base = {
        'provider_name': runtime['provider']['name'],
        'models_endpoint_sha256': runtime['provider']['models_catalog_sha256'],
        'live_authorized': True,
        'ledger_path': str(V4_ROOT / 'provider_sidecar_ledger.sqlite'),
        'usd_cap': float(runtime['resource_caps_usd']['project_total']),
        'max_calls': int(runtime['execution']['max_calls']),
        'max_input_tokens': int(runtime['execution']['max_input_tokens']),
        'max_output_tokens': int(runtime['execution']['max_output_tokens']),
        'transport_attempts': int(runtime['execution']['max_transport_attempts']),
        'read_timeout_seconds': int(runtime['execution']['read_timeout_seconds']),
        'opencode_session_id': f'iclr-memory-pilot/v4-fast-decision/{session_label}',
        'run_start_utc': utc_now(),
    }
    base['compressor'] = _core_role_config(runtime, 'compressor', r1_effort)
    if role == 'reader_R1':
        base['reader'] = _core_role_config(runtime, 'reader_R1', r1_effort)
    elif role == 'reader_R2':
        base['reader'] = _core_role_config(runtime, 'reader_R2', r1_effort)
    else:
        base['reader'] = _core_role_config(runtime, 'reader_R1', r1_effort)
    return base


class V4CostLedger:
    """One authoritative project ledger with project and per-model guards."""
    def __init__(self, path: str | Path, caps: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.caps = caps
        with sqlite3.connect(self.path) as db:
            db.execute('''CREATE TABLE IF NOT EXISTS requests (
                request_key TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                history_id TEXT,
                condition TEXT,
                step INTEGER,
                role TEXT NOT NULL,
                model TEXT,
                accounted_usd REAL NOT NULL,
                reserved_usd REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cache_read_tokens INTEGER,
                state TEXT NOT NULL,
                provider_request_key TEXT,
                finish_reason TEXT,
                metadata_json TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                settled_utc TEXT
            )''')
            db.execute('''CREATE TABLE IF NOT EXISTS model_prior (
                model TEXT PRIMARY KEY,
                usd REAL NOT NULL,
                note TEXT NOT NULL
            )''')
            prior = float(caps['prior_project_actual_plus_uncertain'])
            for model in caps['model_caps']:
                db.execute('INSERT OR IGNORE INTO model_prior(model,usd,note) VALUES(?,?,?)',
                           (model, prior, 'conservatively assigns full historical project spend to each model cap independently'))
            exists = db.execute('SELECT 1 FROM requests WHERE request_key=?',
                                ('v3_actual_plus_uncertain_carry_forward',)).fetchone()
            if not exists:
                db.execute('''INSERT INTO requests
                    (request_key,phase,role,model,accounted_usd,reserved_usd,state,metadata_json,created_utc)
                    VALUES(?,?,?,?,?,?,?,?,?)''',
                    ('v3_actual_plus_uncertain_carry_forward','prior','accounting',None,prior,prior,
                     'prior_actual_plus_uncertain',
                     json.dumps({'source':'artifacts/visible_budget_preflight_v3_partial_20260913.json',
                                 'includes_v3_completed_and_interrupted_reservation':True}, sort_keys=True),
                     utc_now()))

    def existing(self, key: str):
        with sqlite3.connect(self.path) as db:
            return db.execute('SELECT state,accounted_usd FROM requests WHERE request_key=?', (key,)).fetchone()

    def unresolved_request(self, *, phase: str, history_id: str, condition: str,
                           step: int, role: str, model: str,
                           provider_budget: int) -> dict | None:
        """Find an unresolved reservation for this cell at the same provider budget."""
        with sqlite3.connect(self.path) as db:
            rows = db.execute('''SELECT request_key,accounted_usd,created_utc,metadata_json
                FROM requests WHERE phase=? AND history_id=? AND condition=? AND step=?
                  AND role=? AND model=? AND state='reserved'
                ORDER BY created_utc''',
                (phase,history_id,condition,step,role,model)).fetchall()
            if rows is None:
                return None
            # A changed, predeclared completion budget is a distinct request. Keep
            # its old reservation in the ledger, but do not let it suppress the
            # authorized technical adjustment. Missing legacy metadata stays
            # conservative and continues to block the cell.
            for row in rows:
                metadata=json.loads(row[3])
                prior_budget=metadata.get('prompt_provider_budget')
                if prior_budget is not None and int(prior_budget)!=int(provider_budget):
                    continue
                return {'request_key':row[0],'accounted_usd':float(row[1]),
                        'created_utc':row[2],'metadata':metadata}
        return None

    def _spent(self, db, model: str | None = None) -> float:
        if model is None:
            return float(db.execute('SELECT COALESCE(SUM(accounted_usd),0) FROM requests').fetchone()[0])
        prior = db.execute('SELECT COALESCE(usd,0) FROM model_prior WHERE model=?', (model,)).fetchone()
        current = db.execute('SELECT COALESCE(SUM(accounted_usd),0) FROM requests WHERE model=?', (model,)).fetchone()
        return float((prior[0] if prior else 0) + current[0])

    def reserve(self, key: str, *, phase: str, history_id: str | None, condition: str | None,
                step: int | None, role: str, model: str, usd: float,
                input_tokens: int, output_tokens: int, metadata: dict) -> str:
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT state FROM requests WHERE request_key=?', (key,)).fetchone()
            if row:
                if row[0] == 'completed':
                    return 'completed'
                if row[0] == 'reserved':
                    return 'reserved'
                raise RuntimeError(f'Ledger request state is not reusable: {row[0]}')
            project_spent = self._spent(db)
            model_spent = self._spent(db, model)
            project_cap = float(self.caps['project_total'])
            model_cap = float(self.caps['model_caps'][model])
            if project_spent + usd > project_cap + 1e-12:
                raise RuntimeError(f'BUDGET_BLOCK: project cap {project_cap:.6f} would be exceeded')
            if model_spent + usd > model_cap + 1e-12:
                raise RuntimeError(f'BUDGET_BLOCK: {model} cap {model_cap:.6f} would be exceeded')
            db.execute('''INSERT INTO requests
                (request_key,phase,history_id,condition,step,role,model,accounted_usd,reserved_usd,
                 input_tokens,output_tokens,state,metadata_json,created_utc)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (key,phase,history_id,condition,step,role,model,usd,usd,input_tokens,output_tokens,
                 'reserved',json.dumps(metadata,sort_keys=True),utc_now()))
            return 'new'

    def settle(self, key: str, *, usd: float, input_tokens: int | None,
               output_tokens: int | None, cache_read_tokens: int | None,
               provider_request_key: str | None, finish_reason: str | None) -> None:
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT model,state FROM requests WHERE request_key=?', (key,)).fetchone()
            if not row or row[1] not in ('reserved', 'completed'):
                raise RuntimeError('Cannot settle a request without a valid reservation')
            model = row[0]
            other_project = float(db.execute('SELECT COALESCE(SUM(accounted_usd),0) FROM requests WHERE request_key<>?', (key,)).fetchone()[0])
            other_model = self._spent(db, model) - float(db.execute('SELECT COALESCE(accounted_usd,0) FROM requests WHERE request_key=?', (key,)).fetchone()[0])
            over = other_project + usd > float(self.caps['project_total']) + 1e-12
            if model:
                over = over or other_model + usd > float(self.caps['model_caps'][model]) + 1e-12
            state = 'completed_over_cap' if over else 'completed'
            db.execute('''UPDATE requests SET accounted_usd=?,input_tokens=?,output_tokens=?,
                cache_read_tokens=?,state=?,provider_request_key=?,finish_reason=?,settled_utc=?
                WHERE request_key=?''',
                (usd,input_tokens,output_tokens,cache_read_tokens,state,provider_request_key,
                 finish_reason,utc_now(),key))
            if over:
                raise RuntimeError('BUDGET_OVERFLOW: measured usage exceeded the authorized cap')

    def summary(self) -> dict:
        with sqlite3.connect(self.path) as db:
            total, reserved = db.execute('SELECT COALESCE(SUM(accounted_usd),0), COALESCE(SUM(CASE WHEN state="reserved" THEN accounted_usd ELSE 0 END),0) FROM requests').fetchone()
            by_model = {}
            for model, cap in self.caps['model_caps'].items():
                prior = db.execute('SELECT COALESCE(usd,0) FROM model_prior WHERE model=?', (model,)).fetchone()
                current = db.execute('SELECT COALESCE(SUM(accounted_usd),0) FROM requests WHERE model=?', (model,)).fetchone()
                spent = float((prior[0] if prior else 0) + current[0])
                by_model[model] = {'actual_plus_uncertain_usd': spent, 'cap_usd': cap,
                                   'remaining_usd': float(cap) - spent}
            counts = dict(db.execute('SELECT state,COUNT(*) FROM requests GROUP BY state').fetchall())
        return {'project_actual_plus_uncertain_usd': float(total),
                'project_reserved_usd': float(reserved),
                'project_cap_usd': float(self.caps['project_total']),
                'project_remaining_usd': float(self.caps['project_total']) - float(total),
                'model_caps': by_model, 'request_states': counts}


def _provider_cache_path(provider: Provider, role: str, system: str, user: str,
                         max_tokens: int, provider_budget: int, visible_limit: int,
                         replicate: int, namespace, request_metadata: dict) -> Path:
    cfg = provider.config[role]
    safe_cfg = {k: v for k, v in cfg.items() if 'key' not in k.lower()}
    cache_key = digest({'role':role,'model_config':safe_cfg,'system':system,'user':user,
                        'limit':max_tokens,'provider_output_budget':provider_budget,
                        'visible_output_limit':visible_limit,'replicate':replicate,
                        'namespace':namespace,'mock':False,'tokenizer':provider.tok.name,
                        'request_metadata':request_metadata})
    return provider.cache / (cache_key + '.json')


def _request_cost_bound(cfg: dict, full_input_tokens: int, output_budget: int) -> tuple[int, float]:
    reserved_input = min(int(cfg['context_tokens']),
                         int(full_input_tokens) + int(cfg.get('input_reservation_overhead_tokens', 256)))
    usd = (reserved_input * float(cfg['input_usd_per_million']) +
           int(output_budget) * float(cfg['output_usd_per_million'])) / 1_000_000
    return reserved_input, usd


def budgeted_call(provider: Provider, ledger: V4CostLedger, role: str,
                  system: str, user: str, *, phase: str, history_id: str | None,
                  condition: str | None, step: int | None, max_tokens: int,
                  provider_budget: int, visible_limit: int, namespace,
                  request_metadata: dict):
    cfg = provider.config[role]
    full_input = provider.tok.count(system + user)
    reserved_input, reserve_usd = _request_cost_bound(cfg, full_input, provider_budget)
    model = cfg['model']
    ledger_key = digest({'protocol_id':'memory_path_v4_fast_route_decision_20260913',
                         'phase':phase,'history_id':history_id,'condition':condition,
                         'step':step,'role':role,'model':model,'system_sha256':digest(system),
                         'user_sha256':digest(user),'provider_budget':provider_budget,
                         'visible_limit':visible_limit,'namespace':namespace,
                         'runtime_parameters':{k:v for k,v in cfg.items() if 'key' not in k.lower()}})
    state = ledger.reserve(ledger_key, phase=phase, history_id=history_id,
                           condition=condition, step=step, role=role, model=model,
                           usd=reserve_usd, input_tokens=reserved_input,
                           output_tokens=provider_budget,
                           metadata={'system_sha256':digest(system),'user_sha256':digest(user),
                                     'prompt_provider_budget':provider_budget,
                                     'full_request_native_tokens':full_input})
    cache_path = _provider_cache_path(provider, role, system, user, max_tokens,
                                      provider_budget, visible_limit, 0, namespace,
                                      request_metadata)
    if state == 'reserved' and not cache_path.is_file():
        raise RuntimeError('UNCERTAIN_REQUEST_BLOCK: prior request is not resent')
    if state == 'completed' and not cache_path.is_file():
        raise RuntimeError('CACHE_RECEIPT_BLOCK: completed request cache is missing')
    try:
        result = provider.call(role, system, user, max_tokens, 0,
                               namespace=namespace, request_metadata=request_metadata,
                               provider_output_budget=provider_budget,
                               visible_output_limit=visible_limit)
    except Exception as exc:
        metadata = getattr(exc, 'metadata', {}) or {}
        inp = metadata.get('provider_input_tokens')
        out = metadata.get('provider_output_tokens')
        if inp is not None and out is not None:
            cache_read = int(metadata.get('cache_read_tokens') or 0)
            actual = ((int(inp)-cache_read)*float(cfg['input_usd_per_million']) +
                      cache_read*float(cfg.get('cache_read_usd_per_million',cfg['input_usd_per_million'])) +
                      int(out)*float(cfg['output_usd_per_million'])) / 1_000_000
            ledger.settle(ledger_key,usd=actual,input_tokens=int(inp),output_tokens=int(out),
                          cache_read_tokens=cache_read,provider_request_key=metadata.get('request_key'),
                          finish_reason=metadata.get('finish_reason'))
        raise
    if state == 'completed' and not result.get('cache_hit'):
        raise RuntimeError('IDEMPOTENCY_BLOCK: completed request unexpectedly reached provider')
    actual = float(result.get('cost_usd_or_conservative', reserve_usd))
    if state != 'completed':
        ledger.settle(ledger_key,usd=actual,
                      input_tokens=result.get('provider_input_tokens'),
                      output_tokens=result.get('provider_output_tokens'),
                      cache_read_tokens=result.get('cache_read_tokens',0),
                      provider_request_key=result.get('request_key'),
                      finish_reason=result.get('finish_reason'))
    return result


def init_accounting(runtime: dict) -> V4CostLedger:
    load_dotenv = None
    try:
        from dotenv import load_dotenv as _load_dotenv
        load_dotenv = _load_dotenv
    except ImportError:
        pass
    env_path = ROOT / '.env'
    if load_dotenv and env_path.is_file():
        load_dotenv(env_path, override=False)
    key_name = runtime['provider']['api_key_env']
    if not os.environ.get(key_name):
        raise RuntimeError(f'Missing authorized credential environment variable: {key_name}')

    V4_ROOT.mkdir(parents=True, exist_ok=True)
    sidecar = V4_ROOT / 'provider_sidecar_ledger.sqlite'
    limits = {'usd_cap':float(runtime['resource_caps_usd']['project_total']),
              'max_calls':int(runtime['execution']['max_calls']),
              'max_input_tokens':int(runtime['execution']['max_input_tokens']),
              'max_output_tokens':int(runtime['execution']['max_output_tokens'])}
    with sqlite3.connect(sidecar) as db:
        db.execute('CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, key TEXT, usd REAL, input_tokens INTEGER, output_tokens INTEGER, state TEXT)')
        prior_key = 'v4_prior_project_actual_plus_uncertain_carry_forward'
        if not db.execute('SELECT 1 FROM calls WHERE key=?',(prior_key,)).fetchone():
            db.execute('INSERT INTO calls(key,usd,input_tokens,output_tokens,state) VALUES(?,?,?,?,?)',
                       (prior_key,float(runtime['resource_caps_usd']['prior_project_actual_plus_uncertain']),0,0,'reserved'))
    caps = {'project_total':float(runtime['resource_caps_usd']['project_total']),
            'model_caps':{
                'deepseek-v4.1-flash':float(runtime['resource_caps_usd']['deepseek_v4_1_flash']),
                'glm-5.3-flash':float(runtime['resource_caps_usd']['glm_5_3_flash']),
            },
            'prior_project_actual_plus_uncertain':float(runtime['resource_caps_usd']['prior_project_actual_plus_uncertain'])}
    return V4CostLedger(V4_ROOT / 'cost_ledger.sqlite',caps)


def _write_or_verify_manifest(path: Path, manifest: dict, resume: bool) -> None:
    if path.exists():
        old = load_json(path)
        for key, value in manifest.items():
            if key in ('status','updated_utc','created_utc'):
                continue
            if old.get(key) != value:
                raise RuntimeError(f'Frozen run manifest mismatch for {key}; use a new run directory')
        if not resume:
            raise RuntimeError(f'Run directory already exists: {path.parent}; pass --resume only to continue receipts')
    else:
        write_json(path,manifest)


def compress_phase(histories_path: str | Path, out_dir: str | Path, phase: str,
                   runtime_path: str | Path = RUNTIME_DEFAULT, resume: bool = False,
                   r1_effort: str | None = None, allow_legacy_prompt: bool = False) -> dict:
    """Generate direct/staged2/rewrite memories without opening query data."""
    runtime = load_runtime(runtime_path)
    prompt_path, prompt_hash, prompt_resolution = resolve_compressor_prompt(
        runtime, runtime_path, allow_legacy=allow_legacy_prompt)
    prompt_template = prompt_path.read_text(encoding='utf-8')
    histories = read_history_rows(histories_path)
    out = Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    config_hash = file_sha256(runtime_path)
    tok = Tokenizer(runtime['compressor_C1']['tokenizer'])
    manifest_path = out / 'compression_manifest.json'
    manifest = {
        'protocol_id':runtime['protocol_id'],'phase':phase,'status':'RUNNING',
        'runtime_config_path':str(Path(runtime_path).resolve()),'runtime_config_sha256':config_hash,
        'compressor_prompt_path':str(prompt_path.relative_to(ROOT)) if prompt_path.is_relative_to(ROOT) else str(prompt_path),
        'compressor_prompt_sha256':prompt_hash,
        'compressor_prompt_resolution':prompt_resolution,
        'runner_code_sha256':runner_code_sha256(),
        'histories_path':str(Path(histories_path).resolve()),'histories_sha256':file_sha256(histories_path),
        'history_ids':[h['history_id'] for h in histories],
        'history_count':len(histories),'paths':list(PATHS),
        'requested_final_target_tokens':int(runtime['compressor_C1']['final_requested_target_tokens']),
        'requested_intermediate_target_tokens':int(runtime['compressor_C1']['intermediate_requested_target_tokens']),
        'provider_model_id':runtime['compressor_C1']['provider_model_id'],
        'provider_name':runtime['provider']['name'],
        'thinking':'disabled','reasoning_effort':'NOT_SENT',
        'full_HF_tokenizer_revision':runtime['compressor_C1']['full_HF_tokenizer_revision'],
        'GET_v1_models_sha256':runtime['provider']['models_catalog_sha256'],
        'query_file_opened':False,'no_future_query_visible_to_compressor':True,
        'posthoc_truncation':False,'finish_reason_required':NATURAL_FINISH,
        'fresh_request_each_stage':True,'reasoning_forwarded_to_next_stage':False,
        'created_utc':utc_now(),
    }
    _write_or_verify_manifest(manifest_path,manifest,resume)
    if not os.environ.get(runtime['provider']['api_key_env']):
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / '.env',override=False)
        except ImportError:
            pass
    ledger = init_accounting(runtime)
    models = make_provider_config(runtime,'compressor','C1')
    provider = Provider(models,tok,V4_ROOT,mock=False)
    steps_path = out / 'memory_steps.jsonl'
    memories_path = out / 'memories.jsonl'
    done_memory = {(r['history_id'],r['condition']) for r in read_jsonl(memories_path)} if memories_path.exists() else set()
    step_rows = read_jsonl(steps_path) if steps_path.exists() else []
    done_steps = {(r['history_id'],r['condition'],r['step']) for r in step_rows}
    c1 = runtime['compressor_C1']
    final_target = int(c1['final_requested_target_tokens'])
    targets_by_path = path_targets(final_target)

    for history in histories:
        history_text = public_history_text(history)
        history_id = history['history_id']
        history_len = tok.count(history_text)
        for condition, targets in targets_by_path.items():
            if (history_id,condition) in done_memory:
                continue
            current = history_text
            path_status = 'natural_stop'
            final_result = None
            for step_index,target in enumerate(targets,1):
                if (history_id,condition,step_index) in done_steps:
                    existing = next(r for r in step_rows if r['history_id']==history_id and
                                    r['condition']==condition and r['step']==step_index)
                    final_result = existing
                    if existing.get('status') != 'natural_stop':
                        path_status = existing.get('status','TECHNICAL_INVALID')
                        break
                    current = existing['memory_body']
                    continue
                is_final = step_index == len(targets)
                completion_class = 'final' if is_final else 'intermediate'
                budget = int(c1['final_provider_output_budget_tokens'] if is_final else
                             c1['intermediate_provider_output_budget_tokens'])
                unresolved = ledger.unresolved_request(
                    phase=phase,history_id=history_id,condition=condition,step=step_index,
                    role='compressor',model=c1['provider_model_id'],provider_budget=budget)
                if unresolved:
                    row = _uncertain_step_row(
                        phase,history_id,condition,step_index,len(targets),target,
                        completion_class,budget,tok.count(current),unresolved,runtime)
                    append_jsonl(steps_path,row);step_rows.append(row)
                    done_steps.add((history_id,condition,step_index))
                    final_result=row;path_status=row['status']
                    break
                system = c1_system_prompt(target,'final' if is_final else 'intermediate',tok.name,
                                          template_text=prompt_template)
                user = compression_user(current)
                meta = {
                    'native_memory_tokens':tok.count(current),
                    'phase':phase,
                    'completion_budget_class':completion_class,
                    'max_completion_tokens':budget,
                    'reasoning_effort':None,
                    'memory_definition':'final_memory_body_only',
                    'include_reasoning_tokens':False,
                    'include_think_blocks':False,
                    'fresh_request_each_stage':True,
                    'reasoning_forwarded_to_next_stage':False,
                    'experimental_path':[phase,history_id,condition,final_target],
                }
                namespace=targets[:step_index]
                result=budgeted_call(provider,ledger,'compressor',system,user,
                    phase=phase,history_id=history_id,condition=condition,step=step_index,
                    max_tokens=budget,provider_budget=budget,visible_limit=budget,
                    namespace=namespace,request_metadata=meta)
                body,removed=final_visible_body(result.get('text',''))
                visible=tok.count(body)
                reported=result.get('reasoning_tokens')
                if reported is not None:
                    reasoning_value=int(reported); reasoning_source='PROVIDER_REPORTED'
                elif result.get('provider_output_tokens') is not None:
                    reasoning_value=max(0,int(result['provider_output_tokens'])-visible)
                    reasoning_source='ESTIMATED_COMPLETION_MINUS_VISIBLE_NATIVE'
                else:
                    reasoning_value=None; reasoning_source='UNAVAILABLE'
                status=classify_memory_output(body,result.get('finish_reason'))
                row={
                    'phase':phase,'history_id':history_id,'condition':condition,
                    'step':step_index,'step_count':len(targets),'requested_target_tokens':target,
                    'requested_target_is_hard_cap':False,'completion_budget_class':completion_class,
                    'status':status,'memory_body':body,'memory_hash':digest(body),
                    'visible_native_tokens':visible,'reasoning_tokens_or_native_estimate':reasoning_value,
                    'reasoning_token_measurement':reasoning_source,
                    'total_completion_tokens':result.get('total_completion_tokens',result.get('provider_output_tokens')),
                    'full_request_native_tokens':result.get('full_request_native_tokens'),
                    'provider_input_tokens':result.get('provider_input_tokens'),
                    'provider_output_tokens':result.get('provider_output_tokens'),
                    'cache_read_tokens':result.get('cache_read_tokens',0),
                    'finish_reason':result.get('finish_reason'),
                    'provider_name':result.get('provider_name'),
                    'provider_model_id':result.get('provider_model_id'),
                    'returned_model_field':result.get('returned_model_field'),
                    'reasoning_effort':result.get('reasoning_effort'),
                    'thinking':'disabled','provider_output_budget_tokens':result.get('provider_output_budget_tokens'),
                    'run_start_utc':result.get('run_start_utc'),
                    'full_HF_tokenizer_revision':result.get('full_HF_tokenizer_revision'),
                    'GET_v1_models_sha256':result.get('GET_v1_models_sha256'),
                    'opencode_session_id':result.get('opencode_session_id'),
                    'request_key':result.get('request_key'),'cache_hit':result.get('cache_hit'),
                    'cost_usd_or_conservative':result.get('cost_usd_or_conservative'),
                    'reasoning_content_present':result.get('reasoning_content_present',False),
                    'think_blocks_removed':removed,'query_file_opened':False,
                    'input_native_tokens':tok.count(current),
                }
                append_jsonl(steps_path,row);step_rows.append(row);done_steps.add((history_id,condition,step_index))
                final_result=row
                if status!='natural_stop':
                    path_status=status
                    break
                current=body
            if path_status != 'natural_stop':
                current = ''
            memory_tokens=tok.count(current)
            final_row={
                'phase':phase,'history_id':history_id,'group_id':history.get('group_id',history_id),
                'condition':condition,'status':path_status,'text':current,
                'visible_native_tokens':memory_tokens,'memory_hash':digest(current),
                'original_history_native_tokens':history_len,
                'actual_compression_rate':1.0-memory_tokens/history_len if history_len and path_status=='natural_stop' else None,
                'requested_final_target_tokens':final_target,
                'requested_target_is_hard_cap':False,
                'reasoning_tokens_or_native_estimate':(final_result or {}).get('reasoning_tokens_or_native_estimate'),
                'reasoning_token_measurement':(final_result or {}).get('reasoning_token_measurement'),
                'total_completion_tokens':(final_result or {}).get('total_completion_tokens'),
                'finish_reason':(final_result or {}).get('finish_reason'),
                'provider_model_id':c1['provider_model_id'],'provider_name':runtime['provider']['name'],
                'reasoning_effort':'NOT_SENT','thinking':'disabled',
                'provider_output_budget_tokens':(final_result or {}).get('provider_output_budget_tokens'),
                'full_HF_tokenizer_revision':c1['full_HF_tokenizer_revision'],
                'GET_v1_models_sha256':runtime['provider']['models_catalog_sha256'],
                'query_file_opened':False,'no_future_query_visible_to_compressor':True,
                'posthoc_truncation':False,
            }
            append_jsonl(memories_path,final_row);done_memory.add((history_id,condition))

    final_memories=read_jsonl(memories_path)
    expected=len(histories)*len(PATHS)
    status='COMPLETE' if len(final_memories)==expected else 'INCOMPLETE'
    manifest.update({'status':status,'memory_rows':len(final_memories),
                     'expected_memory_rows':expected,'updated_utc':utc_now(),
                     'memory_rows_sha256':file_sha256(memories_path) if memories_path.exists() else None,
                     'unresolved_requests_quarantined':[{
                         'history_id':r['history_id'],'condition':r['condition'],'step':r['step'],
                         'request_key':r['request_key'],'reserved_usd':r['unresolved_reserved_usd']}
                         for r in step_rows if r.get('status')=='TECHNICAL_INVALID_UNCERTAIN'],
                     'query_file_opened':False,'no_future_query_visible_to_compressor':True,
                     'cost_ledger_summary':ledger.summary()})
    write_json(manifest_path,manifest)
    return manifest


def _oracle_text(questions: list[dict]) -> str:
    seen=set(); pieces=[]
    for question in questions:
        for evidence in question.get('gold_evidence',[]):
            if evidence not in seen:
                seen.add(evidence);pieces.append(evidence)
    return '\n'.join(pieces)


def _question_payload(questions: list[dict]) -> list[dict]:
    return [{'question_id':q['question_id'],'question_type':q.get('question_type'),
             'question':q['question']} for q in questions]


def _reader_single_user(memory: str, question: dict) -> str:
    date_hint=('Question date: '+str(question['question_date'])+'\n') if question.get('question_date') else ''
    return 'MEMORY:\n'+memory+'\n\nQUESTION:\n'+date_hint+question['question']


def parse_batch_response(text: str, expected_ids: list[str]) -> tuple[dict[str,dict] | None,str | None]:
    try:
        obj=json.loads(text.strip())
    except (ValueError,TypeError):
        return None,'invalid_reader_json'
    if not isinstance(obj,dict) or not isinstance(obj.get('answers'),list):
        return None,'invalid_reader_batch_schema'
    parsed={}
    for row in obj['answers']:
        if not isinstance(row,dict) or not isinstance(row.get('question_id'),str) or not isinstance(row.get('answer'),str):
            return None,'invalid_reader_batch_schema'
        if not isinstance(row.get('evidence_ids'),list) or any(not isinstance(x,str) for x in row['evidence_ids']):
            return None,'invalid_reader_batch_schema'
        qid=row['question_id']
        if qid in parsed:
            return None,'duplicate_question_id'
        parsed[qid]=row
    if set(parsed)!=set(expected_ids):
        return None,'missing_or_unexpected_question_ids'
    return parsed,None


def _reader_attempt(provider: Provider, ledger: V4CostLedger, runtime: dict,
                    phase: str, history_id: str, condition: str, memory: str,
                    questions: list[dict], mode: str, retry_index: int,
                    reader_effort: str, out_dir: Path):
    reader_cfg=provider.config['reader']
    tok=provider.tok
    visible_limit=int(reader_cfg['visible_output_limit_tokens'])
    provider_budget=int(reader_cfg['provider_output_budget_tokens'])
    qpayload=_question_payload(questions)
    if mode=='batch':
        prompt_path=PROMPT_READER_BATCH if retry_index==0 else PROMPT_READER_BATCH_RETRY
        system=prompt_path.read_text(encoding='utf-8').format(visible_limit=visible_limit,tokenizer=tok.name)
        user=json.dumps({'memory':memory,'questions':qpayload},ensure_ascii=False,separators=(',',':'))
    elif mode=='single':
        system=PROMPT_READER_SINGLE.read_text(encoding='utf-8')
        if retry_index:
            system += (f'\nPROTOCOL RETRY: return exactly one JSON object, no markdown. '
                       f'The answer must be at most {visible_limit} tokens under {tok.name}; '
                       'use a concise phrase and do not include reasoning.')
        if len(questions)!=1:
            raise ValueError('single mode accepts one question per call')
        user=_reader_single_user(memory,questions[0])
    else:
        raise ValueError('reader mode must be batch or single')
    metadata={
        'native_memory_tokens':tok.count(memory),
        'phase':phase,'completion_budget_class':'reader',
        'max_completion_tokens':provider_budget,'reasoning_effort':reader_effort,
        'memory_definition':'final_memory_body_only','include_reasoning_tokens':False,
        'include_think_blocks':False,'fresh_request_each_stage':True,
        'reasoning_forwarded_to_next_stage':False,
        'experimental_path':[phase,history_id,condition,'reader'],
    }
    namespace=[condition,mode,'retry',retry_index]
    return budgeted_call(provider,ledger,'reader',system,user,phase=phase,
        history_id=history_id,condition=condition,step=retry_index,
        max_tokens=provider_budget,provider_budget=provider_budget,
        visible_limit=visible_limit,namespace=namespace,request_metadata=metadata)


def _score_cell(provider, ledger, runtime, tok, reader_runtime, reader_slot, effort, mode,
                phase, hid, condition, memory, memory_status, memory_tokens, memory_hash,
                qs, call_id, call_path, score_path, out, done_call_ids, done_score_ids,
                done_call_attempts, replaying_from_receipt) -> None:
    """Score one (history, condition) cell; append call receipts and score rows.

    Recovery (F4) is deterministic: replayed provider calls resolve through the
    confirmed response cache (budgeted_call enforces cache-hit for completed
    ledger keys), so no new model spend occurs. Missing cache raises
    BLOCKED_MISSING_RESPONSE_RECEIPT instead of refetching or fabricating.
    Already-durable rows are never duplicated.
    """
    parsed_answers={};question_errors={};question_results={};retry_count=0;attempt_metadata=[]
    if memory_status not in ('natural_stop','raw_history','calibration_oracle','calibration_no_memory'):
        question_errors={q['question_id']:'compressor_memory_invalid' for q in qs}
    else:
        qgroups=[[q] for q in qs] if mode=='single' else [qs]
        for group in qgroups:
            group_key=digest([q['question_id'] for q in group])
            group_error=None;group_result=None;parsed=None
            for retry_index in range(2):
                try:
                    result=_reader_attempt(provider,ledger,runtime,phase,hid,condition,
                        memory,group,mode,retry_index,effort,out)
                except RuntimeError as exc:
                    msg=str(exc)
                    if 'CACHE_RECEIPT_BLOCK' in msg or 'UNCERTAIN_REQUEST_BLOCK' in msg:
                        raise RuntimeError(f'BLOCKED_MISSING_RESPONSE_RECEIPT: {hid}/{condition}: {msg}') from exc
                    raise
                if replaying_from_receipt and result.get('cache_hit') is False:
                    raise RuntimeError(f'IDEMPOTENCY_BLOCK: replay of {hid}/{condition} reached provider')
                response_text=result.get('text','')
                visible_tokens=tok.count(response_text)
                if result.get('finish_reason')!=NATURAL_FINISH:
                    parsed=None;error=('TECHNICAL_INVALID_LENGTH' if result.get('finish_reason')=='length'
                                       else 'reader_non_natural_stop')
                elif visible_tokens>int(reader_runtime['visible_answer_limit_tokens']):
                    parsed=None;error='protocol_violation_answer_over_limit'
                elif mode=='batch':
                    parsed,error=parse_batch_response(response_text,[q['question_id'] for q in group])
                else:
                    obj,error=parse_single_response(response_text)
                    parsed={group[0]['question_id']:obj} if obj else None
                retryable=error in ('invalid_reader_json','invalid_reader_schema',
                            'invalid_reader_batch_schema','duplicate_question_id',
                            'missing_or_unexpected_question_ids','protocol_violation_answer_over_limit')
                attempt_metadata.append({'result':result,'error':error,'retry_index':retry_index,
                    'visible_tokens':visible_tokens,'group_key':group_key,
                    'question_ids':[q['question_id'] for q in group],
                    'will_retry':retry_index==0 and retryable})
                group_error=error;group_result=result
                if error is None:
                    break
                if retry_index==0 and retryable:
                    retry_count+=1
                    continue
                break
            for q in group:
                qid=q['question_id'];question_results[qid]=group_result
                if parsed and qid in parsed:
                    parsed_answers[qid]=parsed[qid]
                    question_errors[qid]=None
                else:
                    question_errors[qid]=group_error or 'invalid_reader_response'
    # Every question gets a row. Protocol failures are missing, not silently truncated/scored.
    final_retry_by_group={}
    for entry in attempt_metadata:
        final_retry_by_group[entry['group_key']]=entry['retry_index']
    for entry in attempt_metadata:
        attempt_key=(call_id,tuple(entry['question_ids']),entry['retry_index'])
        if attempt_key in done_call_attempts:
            continue
        result=entry['result']
        append_jsonl(call_path,{
            'call_id':call_id,'phase':phase,'history_id':hid,'condition':condition,
            'question_ids':entry['question_ids'],'reader_slot':reader_slot,
            'reader_mode':mode,'reasoning_effort':effort,'attempt_index':entry['retry_index'],
            'final_attempt_index':final_retry_by_group[entry['group_key']],
            'protocol_error':entry['error'],'response_visible_tokens':entry['visible_tokens'],
            'finish_reason':result.get('finish_reason'),
            'provider_name':result.get('provider_name'),'provider_model_id':result.get('provider_model_id'),
            'returned_model_field':result.get('returned_model_field'),
            'provider_output_budget_tokens':result.get('provider_output_budget_tokens'),
            'provider_input_tokens':result.get('provider_input_tokens'),
            'provider_output_tokens':result.get('provider_output_tokens'),
            'cache_read_tokens':result.get('cache_read_tokens',0),
            'full_request_native_tokens':result.get('full_request_native_tokens'),
            'run_start_utc':result.get('run_start_utc'),
            'full_HF_tokenizer_revision':result.get('full_HF_tokenizer_revision'),
            'GET_v1_models_sha256':result.get('GET_v1_models_sha256'),
            'opencode_session_id':result.get('opencode_session_id'),
            'request_key':result.get('request_key'),
            'cost_usd_or_conservative':result.get('cost_usd_or_conservative'),
            'response_hash':digest(result.get('text','')),
        })
        done_call_attempts.add(attempt_key)
    all_parsed=len(parsed_answers)==len(qs)
    condition_errors=[e for e in question_errors.values() if e]
    call_row={
        'call_id':call_id,'phase':phase,'history_id':hid,'condition':condition,
        'reader_slot':reader_slot,'reader_mode':mode,'reasoning_effort':effort,
        'status':'valid' if all_parsed else (condition_errors[0] if condition_errors else 'invalid_reader_response'),
        'protocol_retry_count':retry_count,'n_attempts':len(attempt_metadata),
        'memory_hash':memory_hash,'memory_native_tokens':memory_tokens,
        'reader_output_budget_tokens':int(reader_runtime['provider_output_budget_tokens']),
        'final_visible_tokens':max((e['visible_tokens'] for e in attempt_metadata),default=None),
    }
    if call_id not in done_call_ids:
        append_jsonl(call_path,{'call_id':call_id,'history_id':hid,'condition':condition,
                                 'reader_slot':reader_slot,'reader_mode':mode,
                                 'attempt_index':-1,'final_attempt_index':-1,**call_row})
    done_call_ids.add(call_id)
    for q in qs:
        qid=q['question_id'];obj=parsed_answers.get(qid)
        score=None;status=question_errors.get(qid) or 'invalid_reader_response';answer=None;evidence=[]
        if obj is not None:
            answer=obj.get('answer');evidence=obj.get('evidence_ids',[])
            score=exact_score(answer,q['answers']) if q.get('scoring')=='exact' else None
            status='valid' if score is not None else 'semantic_grading_required'
        score_id=digest([call_id,qid])
        if score_id in done_score_ids:
            continue
        result=question_results.get(qid) or {}
        group_attempts=[e for e in attempt_metadata if qid in e['question_ids']]
        final_attempt=next((e for e in reversed(group_attempts) if e['retry_index']==final_retry_by_group.get(e['group_key'])),None)
        row={
            'score_id':score_id,'phase':phase,'history_id':hid,
            'question_id':qid,'question_type':q.get('question_type'),
            'information_type':TYPE_MAP.get(q.get('question_type'),q.get('question_type')),
            'condition':condition,'reader_slot':reader_slot,'reader_model':reader_runtime['provider_model_id'],
            'provider_name':runtime['provider']['name'],'reasoning_effort':effort,
            'provider_output_budget_tokens':int(reader_runtime['provider_output_budget_tokens']),
            'visible_answer_limit_tokens':int(reader_runtime['visible_answer_limit_tokens']),
            'final_visible_tokens':final_attempt['visible_tokens'] if final_attempt else None,
            'final_answer_tokens':tok.count(answer) if isinstance(answer,str) else None,
            'provider_input_tokens':result.get('provider_input_tokens'),
            'provider_output_tokens':result.get('provider_output_tokens'),
            'cache_read_tokens':result.get('cache_read_tokens',0),
            'full_request_native_tokens':result.get('full_request_native_tokens'),
            'finish_reason':result.get('finish_reason'),
            'returned_model_field':result.get('returned_model_field'),
            'run_start_utc':result.get('run_start_utc'),
            'full_HF_tokenizer_revision':reader_runtime.get('full_HF_tokenizer_revision'),
            'GET_v1_models_sha256':runtime['provider']['models_catalog_sha256'],
            'memory_hash':memory_hash,'memory_native_tokens':memory_tokens,
            'answer':answer,'evidence_ids':evidence,'score':score,'status':status,
            'protocol_retry_count':sum(e['retry_index']==1 for e in group_attempts),
            'question_hash':digest(q['question']),
        }
        append_jsonl(score_path,row);done_score_ids.add(score_id)


def score_phase(histories_path: str | Path, queries_path: str | Path,
                memories_path: str | Path, out_dir: str | Path, phase: str,
                mode: str = 'batch', reader_slot: str = 'R1',
                conditions: list[str] | None = None,
                runtime_path: str | Path = RUNTIME_DEFAULT,
                r1_effort: str | None = None, resume: bool = False,
                allow_legacy_prompt: bool = False) -> dict:
    runtime=load_runtime(runtime_path)
    prompt_path, prompt_hash, prompt_resolution = resolve_compressor_prompt(
        runtime, runtime_path, allow_legacy=allow_legacy_prompt)
    _ = prompt_path, prompt_hash, prompt_resolution  # compressor prompt is bound, not re-read here
    compression_manifest_path=Path(memories_path).parent/'compression_manifest.json'
    if not compression_manifest_path.is_file():
        raise RuntimeError('Frozen compression manifest missing')
    compression_manifest=load_json(compression_manifest_path)
    # F3: verify actual frozen files BEFORE queries are read or any provider exists.
    frozen=verify_frozen_score_inputs(
        compression_manifest=compression_manifest,
        compression_manifest_path=compression_manifest_path,
        histories_path=histories_path, memories_path=memories_path)
    # Only now are query/gold rows opened.
    histories=frozen['histories']
    queries=read_jsonl(queries_path)
    by_history={}
    for q in queries:
        by_history.setdefault(q['history_id'],[]).append(q)
    if set(by_history)!=set(h['history_id'] for h in histories):
        raise ValueError('History/question ID sets differ')
    memory_rows=frozen['memory_rows']
    by_memory={(m['history_id'],m['condition']):m for m in memory_rows}
    if len(by_memory)!=len(memory_rows):
        raise ValueError('Duplicate (history_id, condition) keys in frozen memories')
    selected=conditions or (['oracle','no_memory','raw','direct','staged2','rewrite'] if phase=='p0'
                            else ['raw','direct','staged2','rewrite'])
    allowed={'oracle','no_memory','raw',*PATHS}
    if set(selected)-allowed:
        raise ValueError('Unknown reader condition')
    if phase!='p0' and {'oracle','no_memory'} & set(selected):
        raise ValueError('oracle/no_memory are calibration-only')
    effort=(r1_effort or runtime['reader_R1']['reasoning_effort']) if reader_slot=='R1' else None
    if reader_slot=='R2':
        configured_effort=runtime['reader_R2']['reasoning_effort']
        if configured_effort=='MATCH_FINAL_R1_SETTING':
            if r1_effort not in ('low','high'):
                raise ValueError('Pass the calibrated R1 effort to resolve R2')
            effort=r1_effort
        else:
            if r1_effort is not None and r1_effort!=configured_effort:
                raise ValueError('Explicit R2 effort conflicts with the frozen runtime')
            effort=configured_effort
    if reader_slot not in ('R1','R2'):
        raise ValueError('reader_slot must be R1 or R2')

    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/'reader_manifest.json'
    reader_runtime=runtime['reader_R1'] if reader_slot=='R1' else runtime['reader_R2']
    tok=Tokenizer(reader_runtime['visible_limit_tokenizer'] if reader_slot=='R1' else reader_runtime['tokenizer'])
    role_name='reader_R1' if reader_slot=='R1' else 'reader_R2'
    models=make_provider_config(runtime,role_name,reader_slot,r1_effort=effort)
    provider=Provider(models,tok,V4_ROOT,mock=False)
    ledger=init_accounting(runtime)
    manifest={
        'protocol_id':runtime['protocol_id'],'phase':phase,'reader_slot':reader_slot,
        'reader_mode':mode,'reader_model':reader_runtime['provider_model_id'],
        'reasoning_effort':effort,'provider_output_budget_tokens':reader_cfg_budget(provider),
        'visible_answer_limit_tokens':int(reader_runtime['visible_answer_limit_tokens']),
        'runtime_config_sha256':file_sha256(runtime_path),
        'compression_manifest_sha256':file_sha256(compression_manifest_path),
        'memory_rows_sha256':frozen['memory_rows_sha256'],
        'histories_sha256':frozen['histories_sha256'],
        'queries_sha256':file_sha256(queries_path),
        'reader_prompt_sha256':reader_prompt_hashes(),
        'compressor_prompt_sha256':prompt_hash,
        'compressor_prompt_resolution':prompt_resolution,
        'runner_code_sha256':runner_code_sha256(),
        'conditions':selected,'created_utc':utc_now(),'status':'RUNNING',
        'full_HF_tokenizer_revision':reader_runtime.get('full_HF_tokenizer_revision'),
        'GET_v1_models_sha256':runtime['provider']['models_catalog_sha256'],
        'unit_of_scoring':'question; aggregate by history',
        'protocol_retry_limit':1,
    }
    _write_or_verify_manifest(manifest_path,manifest,resume)
    call_path=out/'reader_calls.jsonl';score_path=out/'reader_rows.jsonl'
    call_rows=read_jsonl(call_path) if call_path.exists() else []
    score_rows=read_jsonl(score_path) if score_path.exists() else []
    done_call_ids={r['call_id'] for r in call_rows if r.get('attempt_index')==r.get('final_attempt_index')}
    done_score_ids={r['score_id'] for r in score_rows}
    done_call_attempts={(r['call_id'],tuple(r.get('question_ids',[])),r.get('attempt_index'))
                          for r in call_rows if 'call_id' in r}
    # F4/F5: the expected (reader, history, condition, question_id) key set is the
    # contract. A cell is done only when all its score keys exist; call receipts
    # alone never mark a cell complete.
    contract=expected_query_contract(queries,int(runtime['data_design']['questions_per_history']))
    qph=int(runtime['data_design']['questions_per_history'])
    expected_score_ids: set[str] = set()
    call_id_by_cell: dict[tuple[str, str], str] = {}
    for history in histories:
        hid=history['history_id']
        for condition in selected:
            cid=digest([phase,hid,condition,reader_slot,effort,mode,manifest['runtime_config_sha256']])
            call_id_by_cell[(hid,condition)]=cid
            for qid in contract['expected_ids'][hid]:
                expected_score_ids.add(digest([cid,qid]))
    blocked_cells: list[dict] = []

    for history in histories:
        hid=history['history_id']
        qs=by_history[hid]
        if len(qs)!=qph:
            raise ValueError(f'Expected {qph} questions for {hid}, got {len(qs)}')
        for condition in selected:
            if condition=='oracle':
                memory=_oracle_text(qs)
                memory_status='calibration_oracle'
                memory_tokens=tok.count(memory)
                memory_hash=digest(memory)
            elif condition=='no_memory':
                memory='';memory_status='calibration_no_memory';memory_tokens=0;memory_hash=digest('')
            elif condition=='raw':
                memory=public_history_text(history);memory_status='raw_history'
                memory_tokens=tok.count(memory);memory_hash=digest(memory)
            else:
                m=by_memory.get((hid,condition))
                if m is None:
                    raise ValueError(f'Missing frozen memory: {hid}/{condition}')
                memory=m['text'];memory_status=m['status']
                memory_tokens=tok.count(memory);memory_hash=m['memory_hash']
            call_id=call_id_by_cell[(hid,condition)]
            # F4: skip only when every expected score key for this cell is durable.
            # A completed call receipt with missing score rows triggers deterministic
            # recovery from the confirmed response cache (no new provider spend);
            # without a recoverable response the cell stops BLOCKED, never COMPLETE.
            missing_before=[qid for qid in contract['expected_ids'][hid]
                            if digest([call_id,qid]) not in done_score_ids]
            if call_id in done_call_ids and not missing_before:
                continue
            replaying_from_receipt=call_id in done_call_ids
            try:
                _score_cell(
                    provider,ledger,runtime,tok,reader_runtime,reader_slot,effort,mode,
                    phase,hid,condition,memory,memory_status,memory_tokens,memory_hash,
                    qs,call_id,call_path,score_path,out,done_call_ids,done_score_ids,
                    done_call_attempts,replaying_from_receipt)
            except RuntimeError as exc:
                msg = str(exc)
                if ('BLOCKED_MISSING_RESPONSE_RECEIPT' in msg or 'CACHE_RECEIPT_BLOCK' in msg
                        or 'UNCERTAIN_REQUEST_BLOCK' in msg or 'IDEMPOTENCY_BLOCK' in msg):
                    blocked_cells.append({'history_id': hid, 'condition': condition,
                                          'error': msg[:300]})
                    continue
                raise

    rows=read_jsonl(score_path) if score_path.exists() else []
    # F5: COMPLETE requires exact coverage of the expected key set. Duplicates,
    # cross-phase admixture and unknown keys are hard errors, never silent.
    actual_ids=[r.get('score_id') for r in rows]
    if len(set(actual_ids))!=len(actual_ids):
        dupes=sorted({x for x in actual_ids if actual_ids.count(x)>1})
        raise ValueError(f'Duplicate score keys in {score_path}: {dupes[:5]}')
    unknown=sorted(set(actual_ids)-expected_score_ids)
    if unknown:
        raise ValueError(f'Unknown score keys (cross-phase admixture?) in {score_path}: {unknown[:5]}')
    missing=sorted(expected_score_ids-set(actual_ids))
    expected=len(histories)*len(selected)*qph
    complete=not missing and not blocked_cells
    manifest.update({'status':'COMPLETE' if complete else 'INCOMPLETE',
                     'reader_rows':len(rows),'expected_reader_rows':expected,
                     'missing_score_ids':missing,'missing_score_count':len(missing),
                     'blocked_cells':blocked_cells,
                     'reader_rows_sha256':file_sha256(score_path) if score_path.exists() else None,
                     'updated_utc':utc_now(),'cost_ledger_summary':ledger.summary()})
    write_json(manifest_path,manifest)
    return manifest


def reader_cfg_budget(provider: Provider) -> int:
    return int(provider.config['reader']['provider_output_budget_tokens'])


def parse_single_response(text: str) -> tuple[dict | None,str | None]:
    try:
        obj=json.loads(text.strip())
    except (ValueError,TypeError):
        return None,'invalid_reader_json'
    if not isinstance(obj,dict) or not isinstance(obj.get('answer'),str):
        return None,'invalid_reader_schema'
    if not isinstance(obj.get('evidence_ids'),list) or any(not isinstance(x,str) for x in obj['evidence_ids']):
        return None,'invalid_reader_schema'
    return obj,None


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered=sorted(values)
    position=(len(ordered)-1)*probability
    low=math.floor(position);high=math.ceil(position)
    if low==high:
        return float(ordered[low])
    return float(ordered[low]+(ordered[high]-ordered[low])*(position-low))


def _bootstrap_mean_ci(values: list[float], seed: int, reps: int = 20000) -> dict:
    if not values:
        return {'mean':None,'ci90':[None,None],'n':0}
    rng=random.Random(seed)
    n=len(values);means=[]
    for _ in range(reps):
        means.append(sum(values[rng.randrange(n)] for _ in range(n))/n)
    return {'mean':sum(values)/n,'ci90':[_quantile(means,.05),_quantile(means,.95)],'n':n}


def _length_regression(rows: list[dict], seed: int, reps: int = 20000) -> dict:
    usable=[(float(r['delta_log_length']),float(r['delta_utility'])) for r in rows
            if r.get('delta_log_length') is not None and r.get('delta_utility') is not None]
    if not usable:
        return {'equal_length_intercept':None,'slope':None,'intercept_ci90':[None,None],'n':0}
    def fit(sample):
        xs=[x for x,y in sample];ys=[y for x,y in sample]
        mx=sum(xs)/len(xs);my=sum(ys)/len(ys)
        den=sum((x-mx)**2 for x in xs)
        beta=sum((x-mx)*(y-my) for x,y in sample)/den if den>1e-15 else 0.0
        return my-beta*mx,beta
    alpha,beta=fit(usable)
    rng=random.Random(seed);boot=[];n=len(usable)
    for _ in range(reps):
        boot.append(fit([usable[rng.randrange(n)] for _ in range(n)])[0])
    return {'equal_length_intercept':alpha,'slope':beta,
            'intercept_ci90':[_quantile(boot,.05),_quantile(boot,.95)],'n':n}


def aggregate_history_utilities(reader_rows: list[dict], memory_rows: list[dict], phase: str,
                                queries: list[dict] | None = None,
                                questions_per_history: int = 8) -> list[dict]:
    """Aggregate question rows to history utilities with frozen-contract denominators (F5).

    A (history, reader, condition) cell is scorable only when its valid rows
    cover the contract's expected question set exactly. Missing or duplicate
    questions leave utility None (technical missing) instead of shrinking the
    denominator. With explicit queries the per-type denominators come from the
    contract; otherwise each of the four information types expects
    questions_per_history/4 questions.
    """
    if queries is not None:
        contract=expected_query_contract(queries,questions_per_history)
        contract_source='explicit_queries'
    else:
        contract=None
        contract_source='default_8q_2pertype'
    by_hist_condition={}
    seen_qids: dict[tuple, set] = {}
    duplicate_cells: set = set()
    for row in reader_rows:
        key=(row['history_id'],row['condition'],row.get('reader_slot','R1'))
        by_hist_condition.setdefault(key,[]).append(row)
        qkey=(key,row.get('question_id'))
        if qkey in seen_qids:
            duplicate_cells.add(key)
        seen_qids[qkey]=seen_qids.get(qkey,set())|{id(row)}
    lengths={(m['history_id'],m['condition']):m.get('visible_native_tokens') for m in memory_rows}
    groups={}
    for (hid,condition,reader),rows in by_hist_condition.items():
        valid=[r for r in rows if r.get('status')=='valid' and r.get('score') is not None]
        qtypes={r['question_id']:r.get('information_type') for r in rows}
        if contract is not None:
            expected_qids=contract['expected_ids'].get(hid,set())
        else:
            expected_qids=None  # resolved per-cell below from observed ids
        observed_qids={r['question_id'] for r in rows}
        valid_qids={r['question_id'] for r in valid}
        if expected_qids is None:
            expected_qids=observed_qids
        cell_complete=(valid_qids==expected_qids and len(observed_qids)==len(rows)
                       and (hid,condition,reader) not in duplicate_cells
                       and len(expected_qids)==questions_per_history)
        expected=len(expected_qids)
        utility=(sum(float(r['score']) for r in valid)/expected) if cell_complete and expected else None
        by_type={}
        for typ in sorted(set(qtypes.values())):
            tr=[r for r in valid if r.get('information_type')==typ]
            if contract is not None:
                type_key={v:k for k,v in TYPE_MAP.items()}.get(typ,typ)
                n_expected=sum(1 for q in queries if q['history_id']==hid
                               and TYPE_MAP.get(q.get('question_type'),q.get('question_type'))==typ) \
                    if any(q['history_id']==hid for q in queries) else 0
            else:
                n_expected=questions_per_history//len(TYPE_MAP)
            type_complete=(cell_complete and len(tr)==n_expected and n_expected
                           and len({r['question_id'] for r in tr})==n_expected)
            by_type[typ]=(sum(float(r['score']) for r in tr)/n_expected) if type_complete else None
        groups[(hid,reader,condition)]={'history_id':hid,'reader_slot':reader,'condition':condition,
                                        'utility':utility,'n_questions':expected,'n_valid':len(valid),
                                        'contract_source':contract_source,
                                        'contract_complete':cell_complete,
                                        'by_type':by_type,'visible_native_tokens':lengths.get((hid,condition))}
    output=[]
    by_h_readers={}
    for row in groups.values():
        by_h_readers.setdefault((row['history_id'],row['reader_slot']),{})[row['condition']]=row
    for (hid,reader),conds in sorted(by_h_readers.items()):
        d=conds.get('direct',{});s=conds.get('staged2',{});r=conds.get('rewrite',{})
        row={'phase':phase,'history_id':hid,'reader_slot':reader,
             'U_direct':d.get('utility'),'U_staged2':s.get('utility'),'U_rewrite':r.get('utility'),
             'L_direct':d.get('visible_native_tokens'),'L_staged2':s.get('visible_native_tokens'),
             'L_rewrite':r.get('visible_native_tokens'),
             'U_raw':conds.get('raw',{}).get('utility'),'U_no_memory':conds.get('no_memory',{}).get('utility'),
             'U_oracle':conds.get('oracle',{}).get('utility'),
             'n_questions_direct':d.get('n_valid',0),'n_questions_staged2':s.get('n_valid',0),
             'n_questions_rewrite':r.get('n_valid',0)}
        if all(isinstance(row.get(k),(int,float)) for k in ('U_direct','U_staged2','U_rewrite')):
            row.update({'delta_SD':row['U_staged2']-row['U_direct'],
                        'delta_RD':row['U_rewrite']-row['U_direct'],
                        'delta_SR':row['U_staged2']-row['U_rewrite']})
        else:
            row.update({'delta_SD':None,'delta_RD':None,'delta_SR':None})
        if row.get('L_direct') and row.get('L_staged2'):
            row['delta_log_length']=math.log(row['L_staged2']/row['L_direct'])
        else:
            row['delta_log_length']=None
        row['by_type']={}
        for typ in TYPE_MAP.values():
            ud=d.get('by_type',{}).get(typ);us=s.get('by_type',{}).get(typ);ur=r.get('by_type',{}).get(typ)
            row['by_type'][typ]={'U_direct':ud,'U_staged2':us,'U_rewrite':ur,
                                 'delta_SD':us-ud if us is not None and ud is not None else None,
                                 'delta_RD':ur-ud if ur is not None and ud is not None else None,
                                 'delta_SR':us-ur if us is not None and ur is not None else None}
        output.append(row)
    return output


def analyze_phase(reader_rows: list[dict], memory_rows: list[dict], phase: str,
                  seed: int = 20260913, reps: int = 20000,
                  queries: list[dict] | None = None) -> dict:
    histories=aggregate_history_utilities(reader_rows,memory_rows,phase,queries=queries)
    contrasts={}
    for name,key in [('staged2_minus_direct','delta_SD'),('rewrite_minus_direct','delta_RD'),
                     ('staged2_minus_rewrite','delta_SR')]:
        vals=[float(r[key]) for r in histories if r.get(key) is not None]
        result=_bootstrap_mean_ci(vals,seed+len(contrasts),reps)
        result['win_rate']=sum(v>0 for v in vals)/len(vals) if vals else None
        result['loss_rate']=sum(v<0 for v in vals)/len(vals) if vals else None
        contrasts[name]=result
    staged=[r for r in histories if r.get('delta_SD') is not None and r.get('L_direct') and r.get('L_staged2')]
    balanced=[r for r in staged if max(r['L_direct'],r['L_staged2'])/min(r['L_direct'],r['L_staged2'])<=1.25]
    balanced_effect=_bootstrap_mean_ci([r['delta_SD'] for r in balanced],seed+100,reps)
    adjusted_rows=[{'delta_utility':r['delta_SD'],'delta_log_length':r['delta_log_length']} for r in staged]
    length_adjusted=_length_regression(adjusted_rows,seed+101,reps)
    type_effects={}
    for index,typ in enumerate(TYPE_MAP.values()):
        vals=[r['by_type'][typ]['delta_SD'] for r in histories if r['by_type'][typ]['delta_SD'] is not None]
        type_effects[typ]=_bootstrap_mean_ci(vals,seed+200+index,reps)
    return {'phase':phase,'statistics_unit':'history','n_histories':len(histories),
            'history_utilities':histories,'contrasts':contrasts,
            'length_balanced_staged2_vs_direct':{'n':len(balanced),**balanced_effect},
            'length_adjusted_staged2_vs_direct':length_adjusted,
            'type_effects_staged2_vs_direct':type_effects,
            'length_distributions':length_distributions(memory_rows),
            'bootstrap_replicates':reps,'bootstrap_seed':seed}


def length_distributions(memory_rows: list[dict]) -> dict:
    result={}
    for condition in PATHS:
        vals=[int(r['visible_native_tokens']) for r in memory_rows
              if r.get('condition')==condition and r.get('status')=='natural_stop']
        if not vals:
            result[condition]={'n':0,'median':None,'iqr':[None,None],'min':None,'max':None}
        else:
            result[condition]={'n':len(vals),'median':median(vals),
                               'iqr':[_quantile(vals,.25),_quantile(vals,.75)],
                               'min':min(vals),'max':max(vals)}
    return result


def p0_check(memories: list[dict], reader_rows: list[dict], runtime: dict) -> dict:
    finals=[m for m in memories if m.get('condition') in PATHS]
    target=int(runtime['compressor_C1']['final_requested_target_tokens'])
    natural=sum(m.get('status')=='natural_stop' for m in finals)
    in_band=sum(m.get('status')=='natural_stop' and .5*target<=int(m.get('visible_native_tokens',-1))<=2*target for m in finals)
    by_condition={c:[m for m in finals if m.get('condition')==c] for c in PATHS}
    values={}
    for condition in ('oracle','raw','no_memory'):
        rs=[r for r in reader_rows if r.get('condition')==condition]
        vals=[float(r['score']) for r in rs if r.get('status')=='valid' and r.get('score') is not None]
        values[condition]={'accuracy':sum(vals)/len(vals) if vals else None,'n_valid':len(vals),'n_total':len(rs)}
    oracle=values['oracle']['accuracy']
    raw=values['raw']['accuracy'];no_mem=values['no_memory']['accuracy']
    p0={
        'decision':'PASS' if natural>=5 and in_band>=4 and oracle is not None and oracle>=.90 and raw is not None and no_mem is not None and raw-no_mem>=.20 else 'FAIL_REVIEW_REQUIRED',
        'natural_stop_final_count':natural,'natural_stop_final_total':len(finals),
        'final_lengths_in_half_to_double_target':in_band,'final_length_target_interval':[.5*target,2*target],
        'reader_calibration':values,
        'raw_minus_no_memory_accuracy':raw-no_mem if raw is not None and no_mem is not None else None,
        'query_blind_verified':all(m.get('no_future_query_visible_to_compressor') for m in memories),
        'path_status_counts':{c:{s:sum(m.get('status')==s for m in by_condition[c]) for s in sorted({m.get('status') for m in by_condition[c]})} for c in PATHS},
        'note':'If raw minus no-memory is below 0.20, manually review the predeclared eight P0 questions before classifying the instrument.'
    }
    return p0


def estimate_preflight_costs(histories_path: str | Path, queries_path: str | Path,
                            runtime_path: str | Path = RUNTIME_DEFAULT) -> dict:
    """Peak-rate reservation forecast; opens query data locally, never sends it to C1."""
    runtime=load_runtime(runtime_path)
    histories=read_history_rows(histories_path)
    if not histories:
        raise ValueError('Preflight needs at least one representative history')
    queries=read_jsonl(queries_path)
    by_history={}
    for q in queries:
        by_history.setdefault(q['history_id'],[]).append(q)
    history=histories[0]
    qs=by_history.get(history['history_id'],[])
    if len(qs)!=int(runtime['data_design']['questions_per_history']):
        raise ValueError('Preflight representative history must have the frozen question count')
    c1=runtime['compressor_C1'];r1=runtime['reader_R1'];r2=runtime['reader_R2']
    ds_tok=Tokenizer(c1['tokenizer']);glm_tok=Tokenizer(r1['visible_limit_tokenizer'])
    htext=public_history_text(history)
    h_ds=ds_tok.count(htext)
    c1_final=int(c1['final_provider_output_budget_tokens'])
    c1_mid=int(c1['intermediate_provider_output_budget_tokens'])
    target=int(c1['final_requested_target_tokens'])

    def c1_input_bound(target_tokens: int, stage_role: str, body_text: str | None,
                       body_token_bound: int | None) -> int:
        system=c1_system_prompt(target_tokens,stage_role,ds_tok.name)
        prefix='BEGIN_MEMORY_DATA\n';suffix='\nEND_MEMORY_DATA'
        if body_text is not None:
            full=ds_tok.count(system+prefix+body_text+suffix)
        else:
            full=ds_tok.count(system+prefix)+int(body_token_bound or 0)+ds_tok.count(suffix)+64
        return full

    c1_cfg=_core_role_config(runtime,'compressor')
    hist_full=[
        c1_input_bound(target,'final',htext,None),
        c1_input_bound(2*target,'intermediate',htext,None),
        c1_input_bound(target,'final',None,c1_mid),
        c1_input_bound(target,'intermediate',htext,None),
        c1_input_bound(target,'final',None,c1_final),
    ]
    c1_reserved_inputs=[_request_cost_bound(c1_cfg,x,
        c1_final if index in (0,2,4) else c1_mid)[0] for index,x in enumerate(hist_full)]
    c1_cost_per_history=sum((inp*float(c1_cfg['input_usd_per_million'])+
        (c1_final if index in (0,2,4) else c1_mid)*float(c1_cfg['output_usd_per_million']))/1e6
        for index,inp in enumerate(c1_reserved_inputs))
    n_hist=int(runtime['data_design']['p0_histories'])+int(runtime['data_design']['p1_exploration_histories'])+int(runtime['data_design']['p2_confirmation_histories'])
    c1_call_count=n_hist*5
    c1_max_cost=c1_cost_per_history*n_hist
    c1_target_output_tokens=(3*target+3*target)*n_hist
    c1_max_output_tokens=(3*c1_final+2*c1_mid)*n_hist

    batch_systems=[path.read_text(encoding='utf-8').format(
        visible_limit=int(r1['visible_answer_limit_tokens']),tokenizer=glm_tok.name)
        for path in (PROMPT_READER_BATCH,PROMPT_READER_BATCH_RETRY)]
    raw_payload=json.dumps({'memory':htext,'questions':_question_payload(qs)},
                           ensure_ascii=False,separators=(',',':'))
    oracle_payload=json.dumps({'memory':_oracle_text(qs),'questions':_question_payload(qs)},
                              ensure_ascii=False,separators=(',',':'))
    empty_payload=json.dumps({'memory':'','questions':_question_payload(qs)},
                             ensure_ascii=False,separators=(',',':'))
    r1_batch_full=max(
        max(glm_tok.count(system+payload) for system in batch_systems
            for payload in (raw_payload,oracle_payload)),
        max(glm_tok.count(system+empty_payload)+2*c1_final+512
            for system in batch_systems))
    single_system=PROMPT_READER_SINGLE.read_text(encoding='utf-8')
    single_retry_system=single_system+(
        f'\nPROTOCOL RETRY: return exactly one JSON object, no markdown. '
        f'The answer must be at most {int(r1["visible_answer_limit_tokens"])} tokens under {glm_tok.name}; '
        'use a concise phrase and do not include reasoning.')
    single_full=[]
    for q in qs:
        raw_user=_reader_single_user(htext,q)
        oracle_user=_reader_single_user(_oracle_text([q]),q)
        empty_user=_reader_single_user('',q)
        single_full.append(max(
            max(glm_tok.count(system+user) for system in (single_system,single_retry_system)
                for user in (raw_user,oracle_user)),
            max(glm_tok.count(system+empty_user)+2*c1_final+512
                for system in (single_system,single_retry_system))))
    r1_batch_cfg=_core_role_config(runtime,'reader_R1','high')
    r1_budget=int(r1['provider_output_budget_tokens'])
    r1_batch_input=_request_cost_bound(r1_batch_cfg,r1_batch_full,r1_budget)[0]
    r1_single_input=max(_request_cost_bound(r1_batch_cfg,x,r1_budget)[0] for x in single_full)
    r1_batch_calls=(2*6)+(int(runtime['data_design']['p1_exploration_histories'])*4)+(
        int(runtime['data_design']['p2_confirmation_histories'])*4)+2
    r1_single_calls=(2*6*8)+(int(runtime['data_design']['p1_exploration_histories'])*4*8)+(
        int(runtime['data_design']['p2_confirmation_histories'])*4*8)+14
    r1_batch_cost=(r1_batch_calls*(r1_batch_input*float(r1['input_usd_per_million'])+
                    r1_budget*float(r1['output_usd_per_million'])))/1e6
    r1_single_cost=(r1_single_calls*(r1_single_input*float(r1['input_usd_per_million'])+
                     r1_budget*float(r1['output_usd_per_million'])))/1e6

    ds_reader_tok=Tokenizer(r2['tokenizer'])
    ds_questions=_question_payload(qs)
    ds_batch_systems=[path.read_text(encoding='utf-8').format(
        visible_limit=int(r2['visible_answer_limit_tokens']),tokenizer=ds_reader_tok.name)
        for path in (PROMPT_READER_BATCH,PROMPT_READER_BATCH_RETRY)]
    ds_empty=json.dumps({'memory':'','questions':ds_questions},ensure_ascii=False,separators=(',',':'))
    # R2 never receives raw histories: bound its memory input by C1's final
    # provider completion cap, then add the frozen question payload.
    ds_memory_batch=max(ds_reader_tok.count(system+ds_empty) for system in ds_batch_systems)+c1_final+256
    ds_single_system=PROMPT_READER_SINGLE.read_text(encoding='utf-8')
    ds_single_retry_system=ds_single_system+(
        f'\nPROTOCOL RETRY: return exactly one JSON object, no markdown. '
        f'The answer must be at most {int(r2["visible_answer_limit_tokens"])} tokens under {ds_reader_tok.name}; '
        'use a concise phrase and do not include reasoning.')
    ds_single_full=[]
    for q in qs:
        ds_single_full.append(max(ds_reader_tok.count(system+_reader_single_user('',q))
            for system in (ds_single_system,ds_single_retry_system))+c1_final+256)
    r2_cfg=_core_role_config(runtime,'reader_R2','low')
    r2_budget=int(r2['provider_output_budget_tokens'])
    r2_batch_input=_request_cost_bound(r2_cfg,ds_memory_batch,r2_budget)[0]
    r2_single_input=max(_request_cost_bound(r2_cfg,x,r2_budget)[0] for x in ds_single_full)
    r2_batch_calls=int(runtime['data_design']['p2_confirmation_histories'])*3
    r2_single_calls=r2_batch_calls*8
    r2_batch_cost=(r2_batch_calls*(r2_batch_input*float(r2['input_usd_per_million'])+
                    r2_budget*float(r2['output_usd_per_million'])))/1e6
    r2_single_cost=(r2_single_calls*(r2_single_input*float(r2['input_usd_per_million'])+
                     r2_budget*float(r2['output_usd_per_million'])))/1e6

    input_token_stress={
        'C1':sum(c1_reserved_inputs)*n_hist,
        'R1_batch_with_one_retry_per_call':2*r1_batch_calls*r1_batch_input,
        'R1_low_P0_calibration_before_high_promotion':4*6*r1_batch_input,
        'R1_single_fallback_with_one_retry_per_call':2*r1_single_calls*r1_single_input,
        'R2_batch_with_one_retry_per_call':2*r2_batch_calls*r2_batch_input,
    }
    output_token_stress={
        'C1':c1_max_output_tokens,
        'R1_batch_with_one_retry_per_call':2*r1_batch_calls*r1_budget,
        'R1_low_P0_calibration_before_high_promotion':4*6*r1_budget,
        'R1_single_fallback_with_one_retry_per_call':2*r1_single_calls*r1_budget,
        'R2_batch_with_one_retry_per_call':2*r2_batch_calls*r2_budget,
    }
    call_stress={
        'C1':c1_call_count,
        'R1_batch_with_one_retry_per_call':2*r1_batch_calls,
        'R1_low_P0_calibration_before_high_promotion':4*6,
        'R1_single_fallback_with_one_retry_per_call':2*r1_single_calls,
        'R2_batch_with_one_retry_per_call':2*r2_batch_calls,
    }
    r1_low_p0_cost=24*(r1_batch_input*float(r1['input_usd_per_million'])+
                       r1_budget*float(r1['output_usd_per_million']))/1e6
    stress_costs={
        'C1_future_max_usd':c1_max_cost,
        'R1_batch_with_all_retries_usd':2*r1_batch_cost,
        'R1_low_P0_calibration_before_high_promotion_usd':r1_low_p0_cost,
        'R1_single_fallback_with_all_retries_usd':2*r1_single_cost,
        'R2_batch_with_all_retries_usd':2*r2_batch_cost,
        'DeepSeek_future_max_usd':c1_max_cost+2*r2_batch_cost,
        'GLM_future_max_usd':2*r1_batch_cost+r1_low_p0_cost+2*r1_single_cost,
        'project_future_max_usd':c1_max_cost+2*r2_batch_cost+
            2*r1_batch_cost+r1_low_p0_cost+2*r1_single_cost,
    }
    stress_reservations={
        'calls_by_role':call_stress,
        'total_future_calls':sum(call_stress.values()),
        'input_tokens_by_role':input_token_stress,
        'total_future_input_tokens':sum(input_token_stress.values()),
        'output_tokens_by_role':output_token_stress,
        'total_future_output_tokens':sum(output_token_stress.values()),
        'future_cost_stress_usd':stress_costs,
    }

    prior=float(runtime['resource_caps_usd']['prior_project_actual_plus_uncertain'])
    ds_cap=float(runtime['resource_caps_usd']['deepseek_v4_1_flash'])
    glm_cap=float(runtime['resource_caps_usd']['glm_5_3_flash'])
    project_cap=float(runtime['resource_caps_usd']['project_total'])
    ds_batch=c1_max_cost+r2_batch_cost
    planned_glm=r1_batch_cost
    planned_project=prior+ds_batch+planned_glm
    single_ds=c1_max_cost+r2_single_cost
    single_glm=r1_single_cost
    single_project=prior+single_ds+single_glm
    c1_one_input=max(c1_reserved_inputs)
    c1_one_budget=max(c1_final,c1_mid)
    c1_one_usd=(c1_one_input*float(c1_cfg['input_usd_per_million'])+
                c1_one_budget*float(c1_cfg['output_usd_per_million']))/1e6
    r1_one_usd=(r1_batch_input*float(r1['input_usd_per_million'])+
                r1_budget*float(r1['output_usd_per_million']))/1e6
    fits=(prior+ds_batch<=ds_cap and prior+planned_glm<=glm_cap and planned_project<=project_cap)
    return {
        'protocol_id':runtime['protocol_id'],'created_utc':utc_now(),
        'runtime_config_sha256':file_sha256(runtime_path),
        'pricing_basis':'peak rates, no cache discount assumed in reservations',
        'pricing_source':'https://dev.opencode.ai/docs/go/','pricing_verified_date':'2026-09-13',
        'representative_history_id':history['history_id'],'representative_history_tokens_deepseek':h_ds,
        'planned_histories':n_hist,'prior_actual_plus_uncertain_usd':prior,
        'requested_target_tokens':target,'provider_output_budgets':{'C1_final':c1_final,'C1_intermediate':c1_mid,
            'R1':r1_budget,'R2':r2_budget},
        'planned_batch_max_completion_forecast':{
            'C1_calls':c1_call_count,'C1_max_output_tokens':c1_max_output_tokens,
            'C1_requested_target_output_tokens':c1_target_output_tokens,
            'C1_peak_rate_max_cost_usd':c1_max_cost,
            'R1_calls_including_high_promotion_calibration':r1_batch_calls,
            'R1_peak_rate_max_cost_usd':r1_batch_cost,
            'R2_calls':r2_batch_calls,'R2_peak_rate_max_cost_usd':r2_batch_cost,
            'DeepSeek_future_max_cost_usd':ds_batch,
            'DeepSeek_with_conservative_prior_usd':prior+ds_batch,
            'DeepSeek_cap_usd':ds_cap,
            'GLM_future_max_cost_usd':planned_glm,
            'GLM_with_conservative_prior_usd':prior+planned_glm,
            'GLM_cap_usd':glm_cap,
            'project_with_prior_max_cost_usd':planned_project,
            'project_cap_usd':project_cap,'fits_all_caps':fits,
            'largest_single_request_reservations_usd':{'C1':c1_one_usd,'R1':r1_one_usd},
        },
        'r1_per_question_fallback_max_completion_forecast':{
            'R1_calls_upper':r1_single_calls,'R1_future_max_cost_usd':single_glm,
            'R2_reader_mode':'batch, independent of the R1 parser-mode switch',
            'R2_calls_upper':r2_batch_calls,'R2_future_max_cost_usd':r2_batch_cost,
            'DeepSeek_with_C1_and_R2_usd':prior+ds_batch,'DeepSeek_cap_usd':ds_cap,
            'GLM_with_conservative_prior_usd':prior+single_glm,'GLM_cap_usd':glm_cap,
            'project_with_prior_max_cost_usd':prior+ds_batch+single_glm,'project_cap_usd':project_cap,
            'requires_reforecast_after_P0_observed_usage':True,
        },
        'r2_independent_per_question_sensitivity':{
            'R2_calls_upper':r2_single_calls,'R2_future_max_cost_usd':r2_single_cost,
            'DeepSeek_with_C1_and_R2_usd':prior+single_ds,'DeepSeek_cap_usd':ds_cap,
            'project_with_prior_max_cost_usd':single_project,'project_cap_usd':project_cap,
            'requires_reforecast_before_any_R2_mode_change':True,
        },
        'context_gate':{
            'C1_largest_input_plus_output_tokens':max(hist_full)+c1_mid+128,
            'C1_context_tokens':int(c1['context_tokens']),
            'R1_raw_batch_input_plus_output_tokens':r1_batch_full+r1_budget+128,
            'R1_context_tokens':int(r1['context_tokens']),
            'R2_largest_input_plus_output_tokens':ds_memory_batch+r2_budget+128,
            'R2_context_tokens':int(r2['context_tokens']),
        },
        'execution_cap_stress_forecast':stress_reservations,
        'no_live_requests_made':True,
    }


def _cli() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',default=str(RUNTIME_DEFAULT))
    sub=parser.add_subparsers(dest='command',required=True)
    f=sub.add_parser('preflight');f.add_argument('--histories',required=True);f.add_argument('--queries',required=True);f.add_argument('--out',required=True)
    c=sub.add_parser('compress');c.add_argument('--phase',required=True,choices=['p0','p1','p2']);c.add_argument('--histories',required=True);c.add_argument('--out',required=True);c.add_argument('--resume',action='store_true');c.add_argument('--allow-legacy-prompt',action='store_true',help='Allow a runtime without compressor_C1.prompt_path to run under the built-in template (historical reads only; recorded in manifest)')
    s=sub.add_parser('score');s.add_argument('--phase',required=True,choices=['p0','p1','p2']);s.add_argument('--histories',required=True);s.add_argument('--queries',required=True);s.add_argument('--memories',required=True);s.add_argument('--out',required=True);s.add_argument('--mode',choices=['batch','single'],default='batch');s.add_argument('--reader-slot',choices=['R1','R2'],default='R1');s.add_argument('--effort',choices=['low','high']);s.add_argument('--conditions');s.add_argument('--resume',action='store_true');s.add_argument('--allow-legacy-prompt',action='store_true',help='Allow a runtime without compressor_C1.prompt_path for historical reads (recorded in manifest)')
    a=parser.parse_args()
    if a.command=='preflight':
        result=estimate_preflight_costs(a.histories,a.queries,a.runtime)
        write_json(a.out,result)
    elif a.command=='compress':
        result=compress_phase(a.histories,a.out,a.phase,a.runtime,a.resume,
                              allow_legacy_prompt=a.allow_legacy_prompt)
    else:
        conditions=[x for x in a.conditions.split(',') if x] if a.conditions else None
        result=score_phase(a.histories,a.queries,a.memories,a.out,a.phase,a.mode,a.reader_slot,
                           conditions,a.runtime,a.effort,a.resume,
                           allow_legacy_prompt=a.allow_legacy_prompt)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    _cli()
