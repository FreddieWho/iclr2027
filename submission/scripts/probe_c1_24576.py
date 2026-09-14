"""One explicitly authorized C1 probe, using the existing shared P0 ledger."""
import argparse
import hashlib
import json
import sqlite3
from types import SimpleNamespace

from core import ROOT, Tokenizer, digest, load_json, public_text, read_jsonl, write_json
from repair_calibration import probe


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main(preflight_only=False):
    auth_path = ROOT / 'configs/c1_output_budget_authorization_24576.json'
    auth = load_json(auth_path)
    proposal_path = ROOT / auth['proposal']
    require(auth['live_authorized'] and auth['max_probe_calls'] == 1 and
            auth['automatic_transport_retries'] == 0, 'Authorization mismatch')
    require(hashlib.sha256(proposal_path.read_bytes()).hexdigest() == auth['proposal_sha256'],
            'Approved proposal changed')
    proposal = load_json(proposal_path)
    accounting = ROOT / auth['accounting']
    models = load_json(accounting / 'models.json')
    ledger_path = accounting / 'cost_ledger.sqlite'
    with sqlite3.connect(ledger_path.as_uri() + '?mode=ro', uri=True) as db:
        count, prior_cost = db.execute('select count(*),sum(usd) from calls').fetchone()
    require(count == auth['expected_prior_ledger_rows'] and
            abs(prior_cost - auth['expected_prior_actual_or_conservative_usd']) < 1e-9,
            'Ledger changed or one-shot authorization already consumed')
    require(models['ledger_path'] == str(ledger_path) and models['usd_cap'] == .5,
            'Shared P0 ledger/cap mismatch')
    require(models['provider_name'] == proposal['provider'] and
            models['compressor']['model'] == proposal['provider_model_id'] and
            models['compressor']['chat_completions_url'] == proposal['endpoint'] and
            models['compressor']['extra_parameters'] ==
            {'reasoning_effort': 'max', 'temperature': .2, 'top_p': .95},
            'Original C1 generation contract changed')
    require(models['reader']['provider_output_budget_tokens'] == 2048 and
            models['reader']['visible_output_limit_tokens'] == 512 and
            models['reader']['extra_parameters']['reasoning_effort'] == 'max',
            'R1 contract changed')
    history = public_text(read_jsonl(ROOT / proposal['history'])[proposal['history_index']])
    prompt_path = ROOT / proposal['compressor_prompt']
    require(digest(history) == auth['expected_memory_input_hash'] and
            hashlib.sha256(prompt_path.read_bytes()).hexdigest() == auth['original_prompt_sha256'],
            'Original history/prompt mismatch')
    out = ROOT / auth['output_root']
    require(not out.exists(), 'One-shot output directory already exists; do not retry')
    budget = proposal['C1_proposed_provider_output_tokens']
    require(budget == 24576 and proposal['memory_target_native_tokens'] == 1024,
            'Probe scope mismatch')
    require(prior_cost + proposal['probe_cost_upper_usd_with_full_65536_input_ceiling'] <= .5,
            'Insufficient remaining P0 authorization')
    models.update({'phase': 'P0_AUTHORIZED_SINGLE_C1_24576', 'transport_attempts': 1,
                   'max_calls': count + 1, 'compressor_prompt': proposal['compressor_prompt'],
                   'authorization_note': auth['approval_source'],
                   'authorization_record': str(auth_path.relative_to(ROOT)),
                   'authorization_hash': digest(auth)})
    models['compressor']['provider_output_budget_tokens'] = budget
    models['compressor']['reservation_output_tokens'] = budget
    tok = Tokenizer('hf:' + str(ROOT / 'work/tokenizers/glm-5.3-flash-eb9eb208'))
    system = prompt_path.read_text().format(budget=1024, tokenizer=tok.name)
    user = 'BEGIN_MEMORY_DATA\n' + history + '\nEND_MEMORY_DATA'
    request_tokens = tok.count(system + user)
    require(tok.count(history) == 32747 and request_tokens + budget + 128 <= 65536,
            'Original input or conservative context check failed')
    print(json.dumps({'preflight': 'PASS', 'network_called': False,
                      'full_request_native_tokens': request_tokens, 'provider_output_budget': budget,
                      'input_hash': digest(history), 'system_prompt_hash': digest(system),
                      'prior_cumulative_conservative_usd': prior_cost, 'max_new_calls': 1}), flush=True)
    if preflight_only:
        return
    args = SimpleNamespace(accounting=str(accounting), out=str(out), reader=False,
                           histories=str(ROOT / proposal['history']), history_index=0,
                           target=1024, read_timeout=auth['read_timeout_seconds'])
    try:
        probe(args, models_override=models)
    finally:
        if (out / 'result.json').exists():
            result = load_json(out / 'result.json')
            costs = load_json(out / 'cost_summary.json')
            write_json(ROOT / 'artifacts/c1_24576_probe_result.json', {
                'authorization': str(auth_path.relative_to(ROOT)), 'authorization_hash': digest(auth),
                'output_root': str(out.relative_to(ROOT)), 'manifest': load_json(out / 'run_manifest.json'),
                'result': result, 'prior_cumulative_conservative_usd': prior_cost,
                'cumulative_actual_or_conservative_usd': costs['usd_actual_plus_uncertain_reservations'],
                'new_attempts': costs['attempts'] - count, 'cost_summary': costs,
                'formal_P0_completed': False, 'scientific_scores': 0,
                'full_grid_restarted': False, 'posthoc_truncation': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    main(parser.parse_args().preflight_only)
