"""Reference pilot utilities. No live requests are made by importing this module."""
from __future__ import annotations
import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

def digest(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(b).hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def _deep_merge_config(base, overlay):
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_model_config(path, _seen=None):
    """Load a standalone model config or a local overlay that extends one."""
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f'cyclic model config inheritance: {path}')
    seen.add(path)
    overlay = load_json(path)
    parent = overlay.pop('extends_config', None)
    expected_parent_sha256 = overlay.pop('extends_config_sha256', None)
    if not parent:
        return overlay
    parent_path = (path.parent / parent).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f'extended model config not found: {parent_path}')
    if expected_parent_sha256:
        actual_parent_sha256 = hashlib.sha256(parent_path.read_bytes()).hexdigest()
        if actual_parent_sha256 != expected_parent_sha256:
            raise ValueError(f'extended model config hash mismatch: {parent_path}')
    base = load_model_config(parent_path, seen)
    return _deep_merge_config(base, overlay)

def reported_reasoning_tokens(usage):
    """Read a provider-reported hidden reasoning token count without retaining it."""
    if not isinstance(usage, dict):
        return None
    details = usage.get('completion_tokens_details') or {}
    value = usage.get('reasoning_tokens')
    if value is None and isinstance(details, dict):
        value = details.get('reasoning_tokens')
    return int(value) if isinstance(value, (int, float)) and value >= 0 else None

def write_json(path, data):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(p)

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def append_jsonl(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def response_schema_summary(body):
    """Return response shape metadata without retaining provider-generated text."""
    if not isinstance(body, dict):
        return {'body_type': type(body).__name__}
    summary = {'body_keys': sorted(body.keys())}
    choices = body.get('choices')
    if not isinstance(choices, list):
        summary['choices_type'] = type(choices).__name__
        return summary
    summary['choice_count'] = len(choices)
    if not choices or not isinstance(choices[0], dict):
        summary['first_choice_type'] = type(choices[0]).__name__ if choices else None
        return summary
    choice = choices[0]
    summary['choice_keys'] = sorted(choice.keys())
    summary['finish_reason'] = choice.get('finish_reason')
    usage = body.get('usage')
    if isinstance(usage, dict):
        summary['usage_keys'] = sorted(usage.keys())
        for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            if name in usage and isinstance(usage[name], (int, float)):
                summary[name] = usage[name]
    message = choice.get('message')
    if not isinstance(message, dict):
        summary['message_type'] = type(message).__name__
        return summary
    summary['message_keys'] = sorted(message.keys())
    for name, value in message.items():
        if name == 'content':
            if isinstance(value, list):
                summary['content_type'] = 'list'
                summary['content_blocks'] = []
                for block in value[:16]:
                    if isinstance(block, dict):
                        block_summary = {'keys': sorted(block.keys()), 'type': block.get('type')}
                        for field in ('text', 'content', 'reasoning_content'):
                            if field in block:
                                block_summary[field + '_type'] = type(block[field]).__name__
                                if isinstance(block[field], str):
                                    block_summary[field + '_length'] = len(block[field])
                        summary['content_blocks'].append(block_summary)
                    else:
                        summary['content_blocks'].append({'type': type(block).__name__})
                if len(value) > 16:
                    summary['content_blocks_truncated'] = True
            else:
                summary['content_type'] = type(value).__name__
                if isinstance(value, str):
                    summary['content_length'] = len(value)
        elif isinstance(value, str):
            summary[name + '_type'] = 'str'
            summary[name + '_length'] = len(value)
        else:
            summary[name + '_type'] = type(value).__name__
    return summary


_THINK_BLOCK_RE = re.compile(r'<think\b[^>]*>.*?</think\s*>', re.IGNORECASE | re.DOTALL)


def final_visible_body(text: str) -> tuple[str, bool]:
    """Return only provider content outside explicit hidden-thought blocks.

    Reasoning returned in a separate ``reasoning_content`` field is never read
    into this function.  An unclosed think block is treated as having no
    visible body instead of being copied into a memory or answer artifact.
    """
    if not isinstance(text, str):
        return '', False
    opening = re.search(r'<think\b[^>]*>', text, re.IGNORECASE)
    closing = re.search(r'</think\s*>', text, re.IGNORECASE)
    if opening and not closing:
        return text[:opening.start()].strip(), True
    cleaned = _THINK_BLOCK_RE.sub('', text)
    cleaned = re.sub(r'</?think\b[^>]*>', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip(), cleaned != text

class Tokenizer:
    """Demo is NOT a real token budget. Formal runs must specify a fixed tokenizer."""
    def __init__(self, name: str):
        self.name = name
        self.is_demo = name == 'demo'
        self.encoder = None
        if name.startswith('tiktoken:'):
            try:
                import tiktoken
            except ImportError as e:
                raise RuntimeError('Install tiktoken or provide a local HF tokenizer. No silent fallback.') from e
            self.encoder = tiktoken.get_encoding(name.split(':', 1)[1])
        elif name.startswith('hf:'):
            from transformers import AutoTokenizer
            self.encoder = AutoTokenizer.from_pretrained(name[3:], local_files_only=True, trust_remote_code=False)
        elif name.startswith('hf_json:'):
            from tokenizers import Tokenizer as HFTokenizer
            path = Path(name.split(':', 1)[1])
            if path.is_dir():
                path = path / 'tokenizer.json'
            if not path.is_file():
                raise ValueError(f'HF tokenizer.json not found: {path}')
            self.encoder = HFTokenizer.from_file(str(path))
        elif not self.is_demo:
            raise ValueError('tokenizer must be demo, tiktoken:ENCODING, hf:LOCAL_PATH, or hf_json:LOCAL_PATH')
    def encode(self, text: str):
        if self.is_demo:
            return re.findall(r'\S+\s*', text)
        if self.name.startswith('tiktoken:'):
            return self.encoder.encode(text, disallowed_special=())
        if self.name.startswith('hf_json:'):
            return self.encoder.encode(text, add_special_tokens=False).ids
        return self.encoder.encode(text, add_special_tokens=False)
    def decode(self, tokens):
        if self.is_demo:
            return ''.join(tokens)
        if self.name.startswith('hf_json:'):
            return self.encoder.decode(tokens)
        return self.encoder.decode(tokens)
    def count(self, text: str) -> int:
        return len(self.encode(text))
    def cap(self, text: str, budget: int) -> str:
        if budget <= 0:
            raise ValueError('positive budget required')
        tokens = self.encode(text)
        out = self.decode(tokens[:budget])
        # Re-encoding a prefix can change tokenization at the boundary.
        while self.count(out) > budget:
            tokens = tokens[:budget - 1]; budget -= 1
            out = self.decode(tokens)
        return out

def path_targets(arm: str, budget: int, history_tokens: int):
    if budget <= 0:
        raise ValueError('budget must be positive')
    paths = {'raw': [], 'no_memory': [], 'oracle': [], 'direct': [budget], 'staged2_wide': [4*budget, budget],
             'staged2_tight': [2*budget, budget], 'staged3': [4*budget, 2*budget, budget],
             'rewrite3': [budget, budget, budget], 'wait3': [4*budget, 4*budget, budget]}
    if arm not in paths:
        raise ValueError(f'Unknown arm: {arm}')
    seq = paths[arm]
    if seq and max(seq) >= history_tokens:
        raise ValueError('Intermediate target must be below history length; choose budgets in calibration.')
    return seq

def public_text(history: dict) -> str:
    # Deliberate allowlist: never serialize an entire dataset row into a request.
    if not isinstance(history.get('text'), str):
        raise ValueError('History requires text: str')
    return history['text']

def parse_answer(raw: str):
    try:
        obj = json.loads(raw.strip())
        if isinstance(obj, dict) and isinstance(obj.get('answer'), str):
            return obj, None
    except (ValueError, TypeError):
        pass
    return None, 'invalid_reader_json'

def normalize_answer(s):
    return ' '.join(str(s).strip().casefold().split())

def exact_score(answer: str, aliases: list[str]) -> float:
    return float(normalize_answer(answer) in {normalize_answer(a) for a in aliases})

class Ledger:
    """Transactional reservations; uncertain failed requests retain their reservation."""
    def __init__(self, path, limits):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        with sqlite3.connect(self.path) as c:
            c.execute('CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, key TEXT, usd REAL, input_tokens INTEGER, output_tokens INTEGER, state TEXT)')
    def reserve(self, key, usd, inp, out):
        with sqlite3.connect(self.path, timeout=30) as c:
            c.execute('BEGIN IMMEDIATE')
            n, money, ni, no = c.execute('SELECT COUNT(*), COALESCE(SUM(usd),0), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) FROM calls').fetchone()
            cap = self.limits.get('usd_cap')
            if cap is None or cap <= 0:
                raise RuntimeError('BUDGET_BLOCK: a positive authorized USD cap is required')
            if n + 1 > self.limits['max_calls'] or money + usd > cap or ni + inp > self.limits['max_input_tokens'] or no + out > self.limits['max_output_tokens']:
                raise RuntimeError('BUDGET_BLOCK: request would exceed a configured limit')
            row = c.execute('INSERT INTO calls (key,usd,input_tokens,output_tokens,state) VALUES (?,?,?,?,?)', (key,usd,inp,out,'reserved'))
            return row.lastrowid
    def settle(self, rid, usd, inp, out):
        overflow = False
        with sqlite3.connect(self.path) as c:
            c.execute('BEGIN IMMEDIATE')
            other_cost = c.execute('SELECT COALESCE(SUM(usd),0) FROM calls WHERE id<>?', (rid,)).fetchone()[0]
            cap = self.limits.get('usd_cap')
            if cap is not None and other_cost + usd > cap:
                c.execute('UPDATE calls SET usd=?,input_tokens=?,output_tokens=?,state=? WHERE id=?', (usd,inp,out,'completed_over_cap',rid))
                overflow = True
            else:
                c.execute('UPDATE calls SET usd=?,input_tokens=?,output_tokens=?,state=? WHERE id=?', (usd,inp,out,'completed',rid))
        if overflow:
            raise RuntimeError('BUDGET_OVERFLOW: measured provider usage exceeds the authorized USD cap; no score assigned')
    def summary(self):
        with sqlite3.connect(self.path) as c:
            n, cost, inp, out = c.execute('SELECT COUNT(*),COALESCE(SUM(usd),0),COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0) FROM calls').fetchone()
            pending = c.execute("SELECT COUNT(*) FROM calls WHERE state='reserved'").fetchone()[0]
        return {'attempts':n,'usd_actual_plus_uncertain_reservations':cost,'input_tokens':inp,'output_tokens':out,'uncertain_attempts':pending}

class Provider:
    """Chat-Completions-compatible reference adapter. Verify each vendor before live use."""
    def __init__(self, config, tokenizer, out, mock=False):
        self.config = config; self.tok = tokenizer; self.mock = mock
        self.run_start_utc = config.get('run_start_utc') or datetime.now(timezone.utc).isoformat()
        self.opencode_session_id = config.get('opencode_session_id') or f'iclr-memory-pilot/{Path(out).name}'
        self.cache = Path(out)/'cache'; self.cache.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(config.get('ledger_path', Path(out)/'cost_ledger.sqlite'), config)
        if not mock and (not config.get('live_authorized') or tokenizer.is_demo):
            raise RuntimeError('Live mode requires explicit authorization and a real tokenizer')
    def call(self, role, system, user, max_tokens, replicate, mock_text=None, namespace=None, request_metadata=None, provider_output_budget=None, visible_output_limit=None):
        cfg = self.config[role]
        request_metadata = request_metadata or {}
        models_sha256 = cfg.get('models_endpoint_sha256', self.config.get('models_endpoint_sha256'))
        scope = request_metadata.get('experimental_path')
        session_id = self.opencode_session_id if scope is None else self.opencode_session_id + '/' + digest(scope)[:24]
        configured_effort = cfg.get('reasoning_effort', cfg.get('extra_parameters', {}).get('reasoning_effort'))
        requested_effort = request_metadata.get('reasoning_effort')
        if requested_effort is not None and requested_effort != configured_effort:
            raise RuntimeError('CONFIG_BLOCK: reasoning effort cannot vary by experimental path')
        provider_budget = int(provider_output_budget if provider_output_budget is not None else max_tokens)
        visible_limit = int(visible_output_limit if visible_output_limit is not None else cfg.get('max_visible_output_tokens', provider_budget))
        configured_provider_budget = int(cfg.get('provider_output_budget_tokens', cfg.get('max_visible_output_tokens', provider_budget)))
        if provider_budget <= 0 or provider_budget > configured_provider_budget:
            raise RuntimeError('OUTPUT_CAP_BLOCK: provider output budget exceeds the authorized role budget')
        safe_cfg = {k:v for k,v in cfg.items() if 'key' not in k.lower()}
        full_request_native_tokens = self.tok.count(system + user)
        key = digest({'role':role,'model_config':safe_cfg,'system':system,'user':user,'limit':max_tokens,'provider_output_budget':provider_budget,'visible_output_limit':visible_limit,'replicate':replicate,'namespace':namespace,'mock':self.mock,'tokenizer':self.tok.name,'request_metadata':request_metadata})
        cp = self.cache/(key+'.json')
        if cp.exists():
            obj = load_json(cp); obj['cache_hit'] = True
            return obj
        if self.mock:
            body, think_blocks_removed = final_visible_body(mock_text if mock_text is not None else '{"answer":"UNKNOWN","evidence_ids":[]}')
            obj = {'text':body, 'mock':True, 'cache_hit':False, 'model':'MOCK', 'returned_model_field':'MOCK', 'provider_model_id':'MOCK', 'provider_name':self.config.get('provider_name'), 'reasoning_effort':configured_effort, 'provider_output_budget_tokens':provider_budget, 'max_completion_tokens':provider_budget, 'completion_budget_class':request_metadata.get('completion_budget_class'), 'visible_output_limit_tokens':visible_limit, 'usage':{'prompt_tokens':full_request_native_tokens,'completion_tokens':provider_budget}, 'provider_input_tokens':full_request_native_tokens, 'provider_output_tokens':provider_budget, 'cache_read_tokens':0, 'usage_is_measured':False, 'run_start_utc':self.run_start_utc, 'opencode_session_id':self.opencode_session_id, 'full_HF_tokenizer_revision':cfg.get('full_HF_tokenizer_revision'), 'GET_v1_models_sha256':models_sha256, 'full_request_native_tokens':full_request_native_tokens, 'native_memory_tokens':request_metadata.get('native_memory_tokens'), 'finish_reason':'stop','request_key':key,'latency_seconds':0, 'think_blocks_removed':think_blocks_removed, 'reasoning_content_present':False}
            obj['total_completion_tokens'] = provider_budget
            obj['reasoning_tokens'] = None
            obj['opencode_session_id'] = session_id
            write_json(cp,obj); return obj
        if 'REPLACE' in cfg['model'] or 'REPLACE' in cfg['base_url']:
            raise RuntimeError('Configure a real, exact model and endpoint before live requests')
        if cfg.get('input_usd_per_million') is None or cfg.get('output_usd_per_million') is None:
            raise RuntimeError('Price metadata required before live calls')
        if full_request_native_tokens + provider_budget + 128 > cfg['context_tokens']:
            raise RuntimeError('CONTEXT_BLOCK: do not silently truncate history')
        api_key = os.environ.get(cfg.get('api_key_env',''), '')
        if not api_key and not cfg.get('local_no_key', False):
            raise RuntimeError(f'Missing authorized credential environment variable: {cfg.get("api_key_env")}')
        import requests
        payload = {'model':cfg['model'],'messages':[{'role':'system','content':system},{'role':'user','content':user}],cfg.get('output_parameter','max_tokens'):provider_budget}
        extra = cfg.get('extra_parameters', {})
        reserved_names = {'model','messages','max_tokens','max_completion_tokens','seed'}
        if reserved_names.intersection(extra):
            raise ValueError('extra_parameters cannot replace model/messages/budget/seed')
        payload.update(extra)
        if cfg.get('supports_seed'):
            payload['seed'] = replicate
        url = cfg.get('chat_completions_url') or cfg['base_url'].rstrip('/')+'/chat/completions'
        headers = {'Content-Type':'application/json'}
        if api_key: headers['Authorization']='Bearer '+api_key
        if cfg.get('user_agent'): headers['User-Agent'] = cfg['user_agent']
        headers['x-opencode-session'] = session_id
        p_in = float(cfg['input_usd_per_million']); p_out = float(cfg['output_usd_per_million'])
        p_cache = float(cfg.get('cache_read_usd_per_million', p_in))
        # Reserve the measured native request plus a protocol margin. The context
        # gate above still protects the full authorized window.
        reserve_in = min(int(cfg['context_tokens']), full_request_native_tokens + int(cfg.get('input_reservation_overhead_tokens', 256)))
        reserve_out = provider_budget  # Hidden reasoning is part of paid output.
        transport_attempts = int(self.config.get('transport_attempts', 3))
        if not 1 <= transport_attempts <= 3:
            raise ValueError('transport_attempts must be between one and three')
        for attempt in range(transport_attempts):
            rid = self.ledger.reserve(key, (reserve_in*p_in+reserve_out*p_out)/1e6, reserve_in, reserve_out)
            start = time.monotonic()
            try:
                response = requests.post(url, json=payload, headers=headers,
                                         timeout=(15,int(self.config.get('read_timeout_seconds',180))))
            except requests.RequestException:
                if attempt == transport_attempts-1: raise RuntimeError('NETWORK_FAILURE: uncertain charges retained; no score assigned')
                time.sleep(2**(attempt+1)); continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == transport_attempts-1: raise RuntimeError(f'HTTP_RETRY_EXHAUSTED {response.status_code}')
                time.sleep(2**(attempt+1)); continue
            if response.status_code >= 400:
                try:
                    detail = json.dumps(response.json(), ensure_ascii=False, separators=(',', ':'))
                except (ValueError, TypeError):
                    detail = response.text
                detail = re.sub(r'(?i)(bearer\s+)[^\s,}\"]+', r'\1[REDACTED]', str(detail))[:1200]
                raise RuntimeError(f'HTTP_{response.status_code}: adapter/request invalid; provider_detail={detail}; no automatic model switch')
            body = None; choice = None
            try:
                body = response.json(); choice=body['choices'][0]
                message = choice['message']
                reasoning_content_present = isinstance(message, dict) and bool(message.get('reasoning_content'))
                text=message['content']
                if text is None:
                    # A reasoning-only/empty completion is a protocol failure,
                    # not an unknown wire format. Preserve usage and finish reason.
                    text = ''
                if not isinstance(text,str): raise ValueError('non-text content')
                text, think_blocks_removed = final_visible_body(text)
            except (KeyError,IndexError,TypeError,ValueError) as e:
                shape = response_schema_summary(body)
                detail = json.dumps(shape, ensure_ascii=False, separators=(',', ':'))[:2400]
                usage = body.get('usage') if isinstance(body, dict) else None
                measured = isinstance(usage, dict) and 'prompt_tokens' in usage and 'completion_tokens' in usage
                inp = int(usage['prompt_tokens']) if measured else None
                output = int(usage['completion_tokens']) if measured else None
                cache_read = 0
                if measured:
                    prompt_details = usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
                    cache_read = int(usage.get('cache_read_tokens', usage.get('cache_read_input_tokens', prompt_details.get('cached_tokens', 0) if isinstance(prompt_details, dict) else 0)) or 0)
                    cache_read = max(0, min(cache_read, inp))
                    cost = ((inp-cache_read)*p_in + cache_read*p_cache + output*p_out) / 1e6
                    self.ledger.settle(rid, cost, inp, output)
                metadata = {'run_start_utc':self.run_start_utc,'provider_name':self.config.get('provider_name'),'provider_model_id':cfg['model'],'returned_model_field':body.get('model',cfg['model']) if isinstance(body,dict) else cfg['model'],'reasoning_effort':configured_effort,'provider_output_budget_tokens':provider_budget,'max_completion_tokens':provider_budget,'completion_budget_class':request_metadata.get('completion_budget_class'),'visible_output_limit_tokens':visible_limit,'full_HF_tokenizer_revision':cfg.get('full_HF_tokenizer_revision'),'GET_v1_models_sha256':models_sha256,'full_request_native_tokens':full_request_native_tokens,'native_memory_tokens':request_metadata.get('native_memory_tokens'),'provider_input_tokens':inp,'provider_output_tokens':output,'cache_read_tokens':cache_read,'usage_is_measured':measured,'finish_reason':choice.get('finish_reason') if isinstance(choice,dict) else None,'opencode_session_id':self.opencode_session_id,'reasoning_content_present':locals().get('reasoning_content_present',False)}
                metadata['reasoning_tokens'] = reported_reasoning_tokens(usage)
                metadata['total_completion_tokens'] = output
                error = RuntimeError(f'RESPONSE_SCHEMA_ERROR: requires vendor adaptation; response_shape={detail}')
                metadata['opencode_session_id'] = session_id
                error.metadata = metadata
                raise error from e
            usage=body.get('usage') or {}
            known='prompt_tokens' in usage and 'completion_tokens' in usage
            inp=int(usage.get('prompt_tokens',reserve_in)); output=int(usage.get('completion_tokens',provider_budget))
            reasoning_tokens=reported_reasoning_tokens(usage)
            prompt_details=usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
            cache_read=int(usage.get('cache_read_tokens', usage.get('cache_read_input_tokens', prompt_details.get('cached_tokens', 0) if isinstance(prompt_details, dict) else 0)) or 0)
            cache_read=max(0,min(cache_read,inp))
            cost=((inp-cache_read)*p_in+cache_read*p_cache+output*p_out)/1e6
            self.ledger.settle(rid,cost,inp,output)
            returned_model=body.get('model',cfg['model'])
            obj={'text':text,'mock':False,'cache_hit':False,'model':returned_model,'returned_model_field':returned_model,'provider_model_id':cfg['model'],'provider_name':self.config.get('provider_name'),'reasoning_effort':configured_effort,'provider_output_budget_tokens':provider_budget,'max_completion_tokens':provider_budget,'completion_budget_class':request_metadata.get('completion_budget_class'),'visible_output_limit_tokens':visible_limit,'fingerprint':body.get('system_fingerprint'), 'usage':usage,'usage_is_measured':known,'provider_input_tokens':inp,'provider_output_tokens':output,'cache_read_tokens':cache_read,'cost_usd_or_conservative':cost,'run_start_utc':self.run_start_utc,'opencode_session_id':self.opencode_session_id,'full_HF_tokenizer_revision':cfg.get('full_HF_tokenizer_revision'),'GET_v1_models_sha256':models_sha256,'full_request_native_tokens':full_request_native_tokens,'native_memory_tokens':request_metadata.get('native_memory_tokens'),'finish_reason':choice.get('finish_reason'),'request_key':key,'latency_seconds':time.monotonic()-start,'think_blocks_removed':think_blocks_removed,'reasoning_content_present':reasoning_content_present}
            obj['reasoning_tokens'] = reasoning_tokens
            obj['total_completion_tokens'] = output
            obj.update({'response_shape': response_schema_summary(body),
                        'visible_response_tokens': self.tok.count(text),
                        'opencode_session_id': session_id,
                        'request_system_hash': digest(system), 'request_user_hash': digest(user)})
            write_json(cp,obj)
            return obj
        raise RuntimeError('Unreachable')
