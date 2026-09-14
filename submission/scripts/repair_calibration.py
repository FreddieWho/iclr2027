"""Bounded P0 transport/length diagnostics, never scientific effect selection.

Loads only the explicitly named credential, never evaluates shell contents.
All new diagnostics share one ledger including a conservative prior charge.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3

from core import ROOT, Ledger, Provider, Tokenizer, digest, load_json, public_text, read_jsonl, write_json
from run_pilot import validate_reader_response


def credential(name):
    if os.environ.get(name):
        return
    for line in (ROOT / '.env').read_text().splitlines():
        key, sep, value = line.removeprefix('export ').partition('=')
        if sep and key.strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ[name] = value
            return
    raise RuntimeError('Authorized credential is missing')


def initialize(accounting):
    accounting.mkdir(parents=True, exist_ok=False)
    models = load_json(ROOT / 'configs/models_opencode_go_p0_r1_2048.json')
    prior = load_json(ROOT / 'artifacts/provider_probe_manifest.json')
    audit = []
    total = 0.0
    for probe in prior['probes']:
        location = probe.get('output_root')
        if location is None:
            cost = float(probe.get('estimated_actual_usd', 0))
            audit.append({'name': probe['name'], 'status': 'unledgered_usage_control', 'usd': cost})
            total += cost
            continue
        ledger_file = ROOT / location / 'cost_ledger.sqlite'
        with sqlite3.connect(ledger_file.as_uri() + '?mode=ro', uri=True) as db:
            rows = db.execute('select id,usd,input_tokens,output_tokens,state from calls').fetchall()
        for row_id, usd, inp, out, state in rows:
            # Old code reserved less than the request's full output budget.
            # No response usage survives for uncertain attempts: reserve the
            # entire authorized input window and known requested output ceiling.
            ceiling = int(probe.get('provider_output_budget_tokens', out))
            upper = max(usd, (65536 * .15 + max(out, ceiling) * .5) / 1e6) if state == 'reserved' else usd
            total += upper
            audit.append({'source': location, 'row_id': row_id, 'state': state,
                          'recorded_usd': usd, 'carried_upper_usd': upper,
                          'basis': 'full_65536_input_and_requested_output_if_uncertain'})
    models.update({'phase': 'P0_REPAIR_CALIBRATION', 'usd_cap': .50,
                   'phase_usd_cap': .50, 'ledger_path': str(accounting / 'cost_ledger.sqlite'),
                   'transport_attempts': 1, 'gateway_exact_limit': None,
                   'gateway_exact_limit_source': 'NOT_EXPOSED; 65536 is only the authorized input ceiling',
                   'prior_probe_actual_or_uncertain_usd': total,
                   'formal_ledger_usd_cap': None,
                   'compressor_prompt': 'prompts/compress_neutral_calibration.txt',
                   'stop_on_compression_protocol_failure': True,
                   'session_policy': 'isolated_per_diagnostic_output_root'})
    for role in ('compressor', 'reader'):
        models[role]['reservation_output_tokens'] = models[role]['provider_output_budget_tokens']
        models[role]['input_reservation_overhead_tokens'] = 1024
    ledger = Ledger(models['ledger_path'], models)
    rid = ledger.reserve('prior_P0_conservative_carry_forward', total, 0, 0)
    ledger.settle(rid, total, 0, 0)
    write_json(accounting / 'prior_accounting.json', {'cost_basis': 'USD-equivalent Go usage',
        'audit': audit, 'carried_upper_usd': total, 'original_ledgers_modified': False})
    write_json(accounting / 'models.json', models)
    print(json.dumps({'status': 'accounting_initialized', 'prior_upper_usd': total,
                      'p0_remaining_usd': .5-total}), flush=True)


def probe(args, models_override=None):
    accounting = Path(args.accounting).resolve()
    models = models_override if models_override is not None else load_json(accounting / 'models.json')
    models['read_timeout_seconds'] = args.read_timeout
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'request_config.json', models)
    role_config = models['reader' if args.reader else 'compressor']
    tok = Tokenizer(role_config.get('tokenizer', 'hf:' + str(ROOT / 'work/tokenizers/glm-5.3-flash-eb9eb208')))
    if args.reader:
        # Synthetic no-data control, independent of all experimental queries.
        memory = '[S1] Calibration marker: M4Q7.'
        system = (ROOT / 'prompts/reader.txt').read_text()
        user = 'MEMORY:\n' + memory + '\n\nQUESTION:\nWhat is the calibration marker?'
        role, target = 'reader', 512
        history_id = 'transport_sentinel'
    else:
        h = read_jsonl(args.histories)[args.history_index]
        memory = public_text(h)
        target = args.target
        role, history_id = 'compressor', h['history_id']
        system = (ROOT / models['compressor_prompt']).read_text().format(budget=target, tokenizer=tok.name)
        user = 'BEGIN_MEMORY_DATA\n' + memory + '\nEND_MEMORY_DATA'
    provider_budget = models[role]['provider_output_budget_tokens']
    write_json(out / 'run_manifest.json', {
        'purpose': 'P0_technical_diagnostic_only', 'history_id': history_id,
        'model_config_hash': digest(models), 'input_hash': digest(memory),
        'system_prompt_hash': digest(system), 'native_input_tokens': tok.count(memory),
        'target': target, 'role': role, 'provider_output_budget_tokens': provider_budget,
        'code_hash': digest({p.name: p.read_text() for p in sorted((ROOT / 'scripts').glob('*.py'))}),
        'selection_uses_scores': False, 'queries_loaded_by_compressor': False,
        'run_start_utc': datetime.now(timezone.utc).isoformat()})
    credential(models[role]['api_key_env'])
    provider = Provider(models, tok, out)
    try:
        result = provider.call(role, system, user, target, 0,
            request_metadata={'native_memory_tokens': tok.count(memory), 'phase': models['phase']},
            namespace=[history_id, target], provider_output_budget=provider_budget, visible_output_limit=target)
        visible = tok.count(result['text'])
        error = None
        if role == 'reader':
            obj, error, answer_tokens, visible = validate_reader_response(result, tok, 512)
        else:
            answer_tokens = None
            if result['finish_reason'] not in ('stop', 'eos'):
                error = 'compressor_non_natural_stop'
            elif not result['text'].strip():
                error = 'compressor_empty_output'
            elif visible > target:
                error = 'compressor_output_over_limit'
        summary = {k:v for k,v in result.items() if k != 'text'}
        summary.update({'status': error or 'TECHNICAL_PASS', 'history_id': history_id,
                        'target': target, 'final_visible_answer_tokens': answer_tokens,
                        'visible_response_tokens': visible, 'scientific_score': None})
        write_json(out / 'result.json', summary)
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        # Do not expose HTTP library internals or credentials.
        error = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        summary = {'status': 'BLOCKED', 'error': error, 'scientific_score': None}
        write_json(out / 'result.json', summary)
        print(json.dumps(summary), flush=True)
        raise SystemExit(1)
    finally:
        write_json(out / 'cost_summary.json', provider.ledger.summary())


def report(args):
    accounting = Path(args.accounting).resolve()
    cfg = load_json(accounting / 'models.json')
    prior = load_json(accounting / 'prior_accounting.json')
    results = []
    for path in sorted((ROOT / 'work').glob('p0_repair_*/result.json')):
        item = load_json(path)
        manifest = load_json(path.parent / 'run_manifest.json')
        results.append({'output_root': str(path.parent.relative_to(ROOT)),
                        'manifest': manifest, 'result': item})
    with sqlite3.connect(Path(cfg['ledger_path']).as_uri() + '?mode=ro', uri=True) as db:
        rows = db.execute('select key,usd,input_tokens,output_tokens,state from calls').fetchall()
    calls = [row for row in rows if row[0] != 'prior_P0_conservative_carry_forward']
    new_cost = sum(row[1] for row in calls)
    data = {'status': 'P0_REPAIR_DIAGNOSTICS_ONLY', 'provider': cfg['provider_name'],
            'provider_model_id': cfg['compressor']['model'], 'reasoning_effort': 'max',
            'C1_provider_output_budget_tokens': 10240, 'R1_provider_output_budget_tokens': 2048,
            'R1_visible_answer_limit_tokens': 512, 'P0_cap_usd': .5,
            'prior_actual_or_conservative_upper_usd': prior['carried_upper_usd'],
            'new_attempts': len(calls), 'new_actual_or_reserved_usd': new_cost,
            'cumulative_actual_or_conservative_upper_usd': prior['carried_upper_usd']+new_cost,
            'new_uncertain_attempts': sum(row[4] == 'reserved' for row in calls),
            'formal_P0_completed': False, 'scientific_scores': 0,
            'calibration_only_not_independent_confirmation': True,
            'original_artifacts_preserved': True, 'runs': results,
            'repair_scope': ['classify_empty_content_without_losing_usage',
                'reserve_full_provider_output', 'commit_measured_cost_on_overflow',
                'share_P0_ledger_between_new_roots', 'isolate_experimental_sessions',
                'uniform_neutral_prompt_length_calibration'],
            'prior_accounting_evidence': str((accounting/'prior_accounting.json').relative_to(ROOT))}
    write_json(ROOT / 'artifacts/p0_repair_result.json', data)
    print(json.dumps({k:v for k,v in data.items() if k != 'runs'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--accounting', required=True)
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--out')
    parser.add_argument('--histories', default=str(ROOT / 'work/p0_calibration_data_glm_20260911/histories.jsonl'))
    parser.add_argument('--history-index', type=int, default=0)
    parser.add_argument('--target', type=int, default=1024)
    parser.add_argument('--reader', action='store_true')
    parser.add_argument('--read-timeout', type=int, default=180)
    args = parser.parse_args()
    if args.initialize:
        initialize(Path(args.accounting).resolve())
    elif args.report:
        report(args)
    else:
        probe(args)
