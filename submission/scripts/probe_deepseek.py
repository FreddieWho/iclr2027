"""Bounded, query-blind C1 replacement calibration. No automatic grid or retries."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from core import ROOT, Tokenizer, digest, load_json, public_text, read_jsonl, write_json
from repair_calibration import credential, probe

BASE = ROOT / 'work/p0_deepseek_switch_20260912_v1'
AUTH = ROOT / 'configs/deepseek_switch_authorization_20260912.json'


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def ledger_state(auth):
    with sqlite3.connect((ROOT / auth['shared_ledger']).as_uri() + '?mode=ro', uri=True) as db:
        return db.execute('select count(*),coalesce(sum(usd),0) from calls').fetchone()


def prepare(model):
    """GET metadata and download only public tokenizer files; no completions."""
    from huggingface_hub import HfApi, hf_hub_download
    import requests
    auth = load_json(AUTH)
    location = BASE / model
    require(not location.exists(), 'Preparation directory already exists; preserve its frozen files')
    location.mkdir(parents=True)
    repo = 'deepseek-ai/DeepSeek-V4.1-Flash' if model == auth['preferred_model'] else 'deepseek-ai/DeepSeek-V4-Flash'
    revision = auth['tokenizer_revision_preferred'] if model == auth['preferred_model'] else HfApi(token=False).model_info(repo).sha
    meta = HfApi(token=False).model_info(repo, revision=revision)
    require(meta.sha == revision and len(revision) == 40, 'Tokenizer revision not fully resolved')
    token_dir = location / 'tokenizer'
    hashes = {}
    files = ['tokenizer.json', 'tokenizer_config.json']
    if model == auth['preferred_model']:
        files.append('encoding/README.md')
    for filename in files:
        path = hf_hub_download(repo, filename, revision=revision, token=False,
                               local_dir=token_dir, cache_dir=BASE / 'hf-cache')
        hashes[filename] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    cfg = load_json(ROOT / 'work/p0_repair_accounting_v1/models.json')
    credential(cfg['compressor']['api_key_env'])
    response = requests.get('https://opencode.ai/zen/go/v1/models', headers={
        'Authorization': 'Bearer ' + os.environ[cfg['compressor']['api_key_env']],
        'User-Agent': 'iclr-memory-pilot/0.1'}, timeout=(15,60))
    require(response.status_code == 200, 'Models metadata GET failed: HTTP ' + str(response.status_code))
    body = response.json()
    require(any(row.get('id') == model for row in body.get('data', [])), 'Exact model absent from Go model list')
    write_json(location / 'models_response.json', body)
    tok = Tokenizer('hf_json:' + str(token_dir))
    history = public_text(read_jsonl(ROOT / auth['history'])[0])
    require(digest(history) == auth['history_input_hash'], 'Original history changed')
    require(tok.count('') == 0, 'Tokenizer includes unwanted special tokens')
    info = {'model': model, 'repo': repo, 'full_HF_tokenizer_revision': revision,
            'tokenizer': tok.name, 'tokenizer_file_hashes': hashes,
            'GET_v1_models_sha256': hashlib.sha256(response.content).hexdigest(),
            'models_response_canonical_digest': digest(body),
            'observed_utc': datetime.now(timezone.utc).isoformat(),
            'history_native_tokens': tok.count(history), 'history_input_hash': digest(history),
            'native_budget_add_special_tokens': False, 'model_weights_downloaded': False,
            'hf_cli_fallback_reason': 'Installed CLI Typer/Click conflict; existing Hub SDK used',
            'reasoning_effort_requested': 'max',
            'upstream_effort_mapping': 100 if model == auth['preferred_model'] else 'max',
            'gateway_effort_mapping': 'NOT_EXPOSED; must not equate accepted parameter with verified internal effort',
            'gateway_exact_context_limit': None}
    write_json(location / 'preflight.json', info)
    print(json.dumps(info), flush=True)


def run(model, target, preflight_only):
    auth = load_json(AUTH)
    info = load_json(BASE / model / 'preflight.json')
    n, cost = ledger_state(auth)
    require(auth['expected_prior_ledger_rows'] <= n < auth['expected_prior_ledger_rows'] + 3,
            'Bounded request allowance consumed')
    require(cost >= auth['expected_prior_conservative_usd'] - 1e-9, 'Prior accounting disappeared')
    out = BASE / model / ('b' + str(target))
    require(not out.exists(), 'Output exists; no repeat or overwrite')
    if model == auth['fallback_model']:
        prior = load_json(BASE / auth['preferred_model'] / 'b1024/result.json')
        require(prior['status'] != 'TECHNICAL_PASS', 'Fallback forbidden after preferred model passes')
    if target != 1024:
        require(load_json(BASE / model / 'b1024/result.json')['status'] == 'TECHNICAL_PASS',
                'Larger target requires a valid 1024 probe')
    prompt = ROOT / auth['compressor_prompt']
    require(hashlib.sha256(prompt.read_bytes()).hexdigest() == auth['prompt_sha256'], 'Frozen prompt changed')
    history = public_text(read_jsonl(ROOT / auth['history'])[0])
    require(digest(history) == auth['history_input_hash'], 'Frozen history changed')
    cfg = copy.deepcopy(load_json(ROOT / 'work/p0_repair_accounting_v1/models.json'))
    cfg.update({'phase': 'P0_DEEPSEEK_CAPABILITY_ONLY', 'live_authorized': True,
        'opencode_session_id': 'iclr-memory-pilot/deepseek-switch/' + model + '/b' + str(target),
        'authorization_note': auth['authorization_source'], 'authorization_hash': digest(auth),
        'ledger_path': str(ROOT / auth['shared_ledger']), 'transport_attempts': 1,
        'max_calls': auth['expected_prior_ledger_rows'] + auth['max_new_compressor_requests'],
        'usd_cap': min(.5, auth['expected_prior_conservative_usd'] + auth['max_new_conservative_usd']),
        'compressor_prompt': auth['compressor_prompt'],
        'models_endpoint_sha256': info['GET_v1_models_sha256'],
        'models_endpoint_observed_utc': info['observed_utc']})
    cfg['compressor'].update({'model': model, 'family': 'deepseek-v4',
        'tokenizer': info['tokenizer'], 'full_HF_tokenizer_revision': info['full_HF_tokenizer_revision'],
        'provider_output_budget_tokens': 24576, 'reservation_output_tokens': 24576,
        'input_reservation_overhead_tokens': 65536,
        'extra_parameters': {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'max', 'top_p': .95},
        'input_usd_per_million': .30, 'output_usd_per_million': 1.20, 'cache_read_usd_per_million': .006,
        'price_basis': 'Conservative Go peak-rate ceiling; usage is measured, dollar estimate is an upper bound',
        'price_source_url': 'https://opencode.ai/docs/go/', 'price_verified_date': '2026-09-12',
        'temperature_policy': 'Provider default; official DeepSeek thinking API ignores temperature',
        'gateway_effort_mapping': info['gateway_effort_mapping'],
        'upstream_effort_mapping': info['upstream_effort_mapping']})
    tok = Tokenizer(info['tokenizer'])
    require(hashlib.sha256((Path(info['tokenizer'].split(':',1)[1])/'tokenizer.json').read_bytes()).hexdigest() ==
            info['tokenizer_file_hashes']['tokenizer.json'], 'Frozen tokenizer changed')
    full = tok.count(prompt.read_text().format(budget=target) + 'BEGIN_MEMORY_DATA\n' + history + '\nEND_MEMORY_DATA')
    require(full + 24576 + 128 <= 65536, 'Conservative context gate; do not truncate history')
    reserve = (65536*.30 + 24576*1.20)/1e6
    require(cost + reserve <= cfg['usd_cap'], 'Insufficient bounded budget')
    print(json.dumps({'preflight': 'PASS', 'model': model, 'target': target,
                      'history_native_tokens': tok.count(history), 'full_request_native_tokens': full,
                      'max_request_reservation_usd': reserve, 'network_called': False}), flush=True)
    if preflight_only:
        return
    args = SimpleNamespace(accounting=str(ROOT/'work/p0_repair_accounting_v1'), out=str(out),
                           reader=False, histories=str(ROOT/auth['history']), history_index=0,
                           target=target, read_timeout=600)
    try:
        probe(args, models_override=cfg)
    finally:
        report()


def report():
    auth = load_json(AUTH)
    n, cost = ledger_state(auth)
    runs = []
    for path in sorted(BASE.glob('deepseek-*/b*/result.json')):
        result = load_json(path)
        runs.append({'output_root': str(path.parent.relative_to(ROOT)),
                     'manifest': load_json(path.parent/'run_manifest.json'), 'result': result,
                     'provider_output_budget_overrun_tokens': max(0, result.get('provider_output_tokens', 0) -
                         result.get('provider_output_budget_tokens', 0))})
    data = {'status': 'CAPABILITY_DIAGNOSTICS_ONLY', 'authorization': str(AUTH.relative_to(ROOT)),
            'authorization_hash': digest(auth), 'runs': runs,
            'preflights': [load_json(p) for p in sorted(BASE.glob('deepseek-*/preflight.json'))],
            'new_attempts': n-auth['expected_prior_ledger_rows'],
            'new_usage_cost_upper_usd': cost-auth['expected_prior_conservative_usd'],
            'P0_cumulative_conservative_usd': cost, 'P0_remaining_usd': .5-cost,
            'dollar_basis': 'Conservative peak rates; not a provider billing receipt',
            'formal_P0_completed': False, 'scientific_scores': 0, 'R1_changed': False,
            'scientific_result_selection': False, 'full_grid_restarted': False,
            'valid_memory_count': sum(r['result']['status'] == 'TECHNICAL_PASS' for r in runs),
            'larger_memory_probe_targets_executed': sorted({r['manifest']['target'] for r in runs if r['manifest']['target'] > 1024})}
    write_json(ROOT/'artifacts/deepseek_switch_result.json', data)
    print(json.dumps({k:v for k,v in data.items() if k not in ('runs','preflights')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare','probe','report'])
    p.add_argument('--model', choices=['deepseek-v4.1-flash','deepseek-v4-flash'], default='deepseek-v4.1-flash')
    p.add_argument('--target', type=int, choices=[1024,2048,8192], default=1024)
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.model)
    elif a.action == 'report':
        report()
    else:
        run(a.model, a.target, a.preflight_only)
