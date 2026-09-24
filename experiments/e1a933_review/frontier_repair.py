"""Complete dev policy enumeration, fixed-budget extrapolation and all-scene
deployment estimation on immutable inputs. Two modes:

  repair    : archived dev-policy frontier on the oracle-feasible split.
  fullscene : apply the dev-selected budget policies to every scene with no
              oracle-feasibility filter at deployment time, plus a clearly
              separated diagnostic oracle-label frontier.

Threshold selection never uses eval labels. The diagnostic frontier does use
eval labels and is reported apart from the deployment estimate.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/f095_campaign'))
from d10_frontier import (load, split_scenes, reachable_ladder, policy_outcomes,
                          fit_isotonic, apply_isotonic, paired_own_quality_ci,
                          summarize)
from d10_frontier_enc import logits_on, arm_checkpoint
OUT = ROOT / 'artifacts/e1a933_review/frontier'
N_BOOT = 2000
BUDGETS = (0.25, 0.5, 0.75, 1.0)
METRICS = ('coverage', 'own_quality', 'success_rate_per_scene',
           'net_benefit_per_scene')


def frontier(scores, scene_of, costs, feas, scenes):
    rows = np.flatnonzero(np.isin(scene_of, scenes))
    thresholds = reachable_ladder(scores[rows])
    selected = np.full((len(thresholds), len(scenes)), -1, dtype=int)
    for col, s in enumerate(scenes):
        ix = np.flatnonzero(scene_of == s)
        # Original index settles equal costs, independently of labels.
        ix = ix[np.argsort(costs[ix], kind='stable')]
        valid = scores[ix][None, :] >= thresholds[:, None]
        has = valid.any(1)
        selected[has, col] = ix[valid[has].argmax(1)]
    # Complete enumeration retained separately; dedup by actions, not coverage.
    _, reverse_keep = np.unique(selected[::-1], axis=0, return_index=True)
    keep = np.sort(len(thresholds) - 1 - reverse_keep)
    sel = selected[keep]
    thresholds = thresholds[keep]
    acted = sel >= 0
    success = acted & feas[np.maximum(sel, 0)]
    paid = np.where(acted, costs[np.maximum(sel, 0)], 0.)
    return thresholds, sel, acted, success, paid


def logit_sources(d, dev):
    """Label-free score arrays per encoder/seed/model.

    Models are `clean`, `flipmine` (negative raw logits of the frozen arms) and
    `isotonic_clean` (PAVA map fitted on dev rows only). Re-encoded arms record
    their checkpoint hash.
    """
    so = d['scene_of']
    feas = d['feas'].astype(bool)
    devmask = np.isin(so, dev)
    out = {}
    for enc in ['raw', 'rel12', 'sixdist']:
        for seed in [11, 23, 47]:
            ck = {}
            if enc == 'raw':
                scores = {m: -np.asarray(d[f'lg_s{seed}_{m}'], float)
                          for m in ['clean', 'flipmine']}
            else:
                paths = ({'clean': arm_checkpoint('relfeat', seed),
                          'flipmine': arm_checkpoint('relflip', seed)} if enc == 'rel12'
                         else {m: ROOT / f'artifacts/f095_campaign/U02/sixdist_w64_s{seed}/{m}'
                               for m in ['clean', 'flipmine']})
                scores = {m: -logits_on(d['frames'], p).astype(float)
                          for m, p in paths.items()}
                ck = {m: hashlib.sha256((p / 'model.pt').read_bytes()).hexdigest()
                      for m, p in paths.items()}
            bx, by = fit_isotonic(scores['clean'][devmask], feas[devmask])
            scores['isotonic_clean'] = apply_isotonic(bx, by, scores['clean'])
            out[f'{enc}_s{seed}'] = {'scores': scores, 'checkpoints': ck}
    return out


def scene_row_matrix(scene_of):
    """Group row indices by scene once; every scene must share the candidate count."""
    scenes, counts = np.unique(scene_of, return_counts=True)
    if counts.min() != counts.max():
        raise ValueError('uneven candidate count across scenes')
    order = np.argsort(scene_of, kind='stable')
    return scenes, order.reshape(len(scenes), counts[0])


def rows_for(scenes, scenes_all, rows):
    """Row-index matrix for `scenes` (ascending, all present) in that order."""
    scenes = np.asarray(scenes)
    if len(scenes) > 1 and not np.all(np.diff(scenes) > 0):
        raise ValueError('scenes must be ascending')
    return rows[np.searchsorted(scenes_all, scenes)]


def grouped_outcomes(score, rows, act_cost, feas, tau):
    """Vectorised `policy_outcomes` over precomputed scene groups.

    Same action rule: act iff some candidate has score>=tau, pick the cheapest
    such candidate, ties by original row order. No feasibility filter is applied.
    """
    valid = score[rows] >= tau
    j = np.where(valid, act_cost[rows], np.inf).argmin(1)
    acted = valid.any(1)
    sel = np.where(acted, rows[np.arange(len(rows)), j], -1)
    pick = np.maximum(sel, 0)
    return acted, acted & feas[pick], np.where(acted, act_cost[pick], 0.), sel


def deployment_metrics(acted, succ, cost, feas_scene):
    """All-scene deployment summary; infeasible scenes stay in the denominator."""
    acted = np.asarray(acted)
    succ = np.asarray(succ)
    cost = np.asarray(cost)
    feas_scene = np.asarray(feas_scene).astype(bool)
    n, n_acted, n_succ, total = len(acted), int(acted.sum()), int(succ.sum()), float(cost.sum())
    inf = ~feas_scene

    def block(mask):
        a, s, c = acted[mask], succ[mask], cost[mask]
        na, ns, nsc = int(a.sum()), int(s.sum()), int(mask.sum())
        return {'n_scenes': nsc, 'n_acted': na,
                'coverage': round(na / nsc, 4) if nsc else None,
                'n_success': ns,
                'own_quality': round(ns / na, 4) if na else None,
                'total_cost': round(float(c.sum()), 4),
                'net_benefit_per_scene': round((ns - float(c.sum())) / nsc, 4) if nsc else None}

    infeas = block(inf)
    infeas['cost_share_of_total'] = round(float(cost[inf].sum()) / total, 4) if total > 0 else None
    return {'n_scenes': n, 'n_acted': n_acted,
            'coverage': round(n_acted / n, 4) if n else None,
            'n_success': n_succ,
            'success_rate_per_scene': round(n_succ / n, 4) if n else None,
            'own_quality': round(n_succ / n_acted, 4) if n_acted else None,
            'total_cost': round(total, 4),
            'mean_cost_per_acted': round(total / n_acted, 4) if n_acted else None,
            'mean_cost_per_scene': round(total / n, 4) if n else None,
            'net_benefit_per_scene': round((n_succ - total) / n, 4) if n else None,
            'oracle_feasible_scenes': block(~inf),
            'oracle_infeasible_scenes': infeas}


def budget_threshold(dev_acted, thresholds, budget):
    """Dev-only budget rule: coverage closest to target, tie -> largest tau."""
    gaps = np.abs(dev_acted.mean(1) - budget)
    k = np.flatnonzero(gaps == gaps.min())[-1]
    return int(k), float(thresholds[k]), float(dev_acted[k].mean())


def _ratio_stats(acted, succ, cost, draw):
    """Coverage / own quality / per-scene success / per-scene net benefit."""
    a, s, c = acted[draw], succ[draw], cost[draw]
    n, na = len(draw), int(a.sum())
    return np.array([na / n, int(s.sum()) / na if na else np.nan,
                     int(s.sum()) / n, (int(s.sum()) - float(c.sum())) / n])


def deployment_bootstrap_ci(acted, succ, cost, rng, n_boot=N_BOOT):
    """Same-scene resampling; every replicate recomputes each full ratio."""
    n = len(acted)
    v = np.array([_ratio_stats(acted, succ, cost, rng.choice(n, size=n, replace=True))
                  for _ in range(n_boot)])
    return {m: [round(float(np.nanquantile(v[:, i], 0.025)), 4),
                round(float(np.nanquantile(v[:, i], 0.975)), 4)]
            for i, m in enumerate(METRICS)}


def paired_deployment_ci(a1, s1, c1, a2, s2, c2, rng, n_boot=N_BOOT):
    """Paired same-scene bootstrap of the difference; each policy recomputed."""
    n = len(a1)
    v = np.array([_ratio_stats(a1, s1, c1, d) - _ratio_stats(a2, s2, c2, d)
                  for d in (rng.choice(n, size=n, replace=True) for _ in range(n_boot))])
    return {m: [round(float(np.nanquantile(v[:, i], 0.025)), 4),
                round(float(np.nanquantile(v[:, i], 0.975)), 4)]
            for i, m in enumerate(METRICS)}


def pareto_envelope(curve):
    """Non-dominated diagnostic points: no other point is at least as good on
    own quality, per-scene net benefit and total cost with one strictly better.
    """
    def q(r):
        return r[2] if r[2] is not None else 0.0
    keep = []
    for r in curve:
        dom = any((q(o) >= q(r) and o[5] >= r[5] and o[4] <= r[4])
                  and (q(o) > q(r) or o[5] > r[5] or o[4] < r[4]) for o in curve)
        if not dom:
            keep.append(r)
    return keep


def oracle_label_curve(score, rmat, costs, feas, fvec):
    """DIAGNOSTIC: all deployment-universe candidate breakpoints, deduped by the
    full selected-action vector (largest threshold kept as representative).
    Action vectors depend on scores and costs only; the label use happens when a
    point is picked as 'best', not here.
    """
    thresholds = reachable_ladder(score[rmat])
    n_raw = int(len(thresholds))
    sel = np.empty((n_raw, len(rmat)), dtype=int)
    for i, t in enumerate(thresholds):
        sel[i] = grouped_outcomes(score, rmat, costs, feas, float(t))[3]
    _, rev = np.unique(sel[::-1], axis=0, return_index=True)
    keep = np.sort(n_raw - 1 - rev)
    curve = []
    for i in keep:
        t = float(thresholds[i])
        a, s, c, _ = grouped_outcomes(score, rmat, costs, feas, t)
        mm = deployment_metrics(a, s, c, fvec)
        curve.append([round(t, 6), mm['coverage'], mm['own_quality'],
                      mm['success_rate_per_scene'], mm['total_cost'],
                      mm['net_benefit_per_scene'], mm['oracle_infeasible_scenes']['n_acted']])
    return thresholds[keep], curve, n_raw


def rng_for(*tags):
    """Deterministic, platform-independent bootstrap seed per comparison."""
    key = '|'.join(str(t) for t in tags).encode()
    return np.random.default_rng(int(hashlib.sha256(key).hexdigest()[:8], 16))


def source_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def run_repair(out):
    out = Path(out)
    if (out / 'FRONTIER_REPAIR.json').exists():
        raise FileExistsError('Immutable completed/partial run exists; use --out NEW_DIRECTORY')
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    d = load()
    feas = d['feas'].astype(bool)
    so = d['scene_of']
    costs = d['costs'][d['act_idx']]
    dev, ev = split_scenes(d, feas)
    result = {'split': 'archived first-86 feasible-scene dev; remaining feasible eval',
              'dev_n': len(dev), 'eval_n': len(ev),
              'scope': 'conditional on oracle-feasible scenes; not all-scene deployment',
              'tie_rule': 'lowest cost then original row index, no labels; largest threshold representative per dev action vector',
              'net_benefit_units': 'unit successful action reward minus one times archived action cost; illustrative utility only',
              'interpretation': 'dev budget selected policy extrapolation; eval coverage is not forced equal',
              'table_sha256': hashlib.sha256((ROOT / 'artifacts/p123_upgrade/p2/p2_table.npz').read_bytes()).hexdigest(),
              'pairs': {}}
    sources = logit_sources(d, dev)
    for key, entry in sources.items():
        enc, seed = key.rsplit('_s', 1)
        seed = int(seed)
        scores = entry['scores']
        curves = {}
        R = {'policies': {}, 'budget_comparisons': []}
        for m, score in scores.items():
            ts, sel, act, succ, paid = frontier(score, so, costs, feas, dev)
            curves[m] = (ts, act, succ, paid)
            rows = [dict(tau=float(t), **summarize(a, u, c), net_benefit=float((u - c).mean()))
                    for t, a, u, c in zip(ts, act, succ, paid)]
            eval_rows = []
            for t in ts:
                ea, eu, ec = policy_outcomes(score, so, costs, feas, ev, t)
                eval_rows.append(dict(tau=float(t), **summarize(ea, eu, ec),
                                      net_benefit=float((eu - ec).mean())))
            R['policies'][m] = {'n_action_policies': len(ts), 'frontier_dev': rows,
                                'frontier_eval_dev_selected_thresholds': eval_rows}
            np.savez_compressed(out / f'{enc}_s{seed}_{m}_dev.npz', thresholds=ts,
                                selected_rows=sel, scenes=dev)
        for budget in BUDGETS:
            row = {'target_dev_coverage': budget, 'models': {}}
            outcomes = {}
            for m, (ts, ac, su, co) in curves.items():
                # Coverage proximity then maximum threshold; no dev/eval labels.
                gaps = np.abs(ac.mean(1) - budget)
                k = np.flatnonzero(gaps == gaps.min())[-1]
                oa, ou, oc = policy_outcomes(scores[m], so, costs, feas, ev, ts[k])
                outcomes[m] = (oa, ou, oc)
                row['models'][m] = {'tau': float(ts[k]), 'dev_coverage': float(ac[k].mean()),
                                    'eval': dict(**summarize(oa, ou, oc),
                                                 net_benefit=float((ou - oc).mean()))}
            a, u, _ = outcomes['flipmine']
            b, v, _ = outcomes['clean']
            if a.any() and b.any():
                row['quality_difference_CI'] = paired_own_quality_ci(
                    u, a, v, b, ev, np.random.default_rng(260924 + seed))
            R['budget_comparisons'].append(row)
        # Cost budgets fixed as fractions of all-execute zero-threshold-independent max candidate cost sum.
        max_budget = sum(costs[so == s].max() for s in dev)
        for fraction in [0.0, 0.25, 0.5, 1.0]:
            row = {'target_dev_total_cost': float(fraction * max_budget), 'models': {}}
            outcomes = {}
            for m, (ts, ac, su, co) in curves.items():
                gaps = np.abs(co.sum(1) - fraction * max_budget)
                k = np.flatnonzero(gaps == gaps.min())[-1]
                oa, ou, oc = policy_outcomes(scores[m], so, costs, feas, ev, ts[k])
                outcomes[m] = (oa, ou, oc)
                row['models'][m] = {'tau': float(ts[k]), 'dev_total_cost': float(co[k].sum()),
                                    'eval': dict(**summarize(oa, ou, oc),
                                                 net_benefit=float((ou - oc).mean()))}
            a, u, _ = outcomes['flipmine']
            b, v, _ = outcomes['clean']
            if a.any() and b.any():
                row['quality_difference_CI'] = paired_own_quality_ci(
                    u, a, v, b, ev, np.random.default_rng(260925 + seed))
            R['budget_comparisons'].append(row)
        result['pairs'][key] = R
        print(enc, seed, {m: v['n_action_policies'] for m, v in R['policies'].items()}, flush=True)
    result['wall_seconds'] = time.monotonic() - start
    (out / 'FRONTIER_REPAIR.json').write_text(json.dumps(result, indent=2))


def run_fullscene(out):
    """Deployment over every scene with no oracle-feasibility filter."""
    out = Path(out)
    if (out / 'FRONTIER_FULLSCENE.json').exists():
        raise FileExistsError('Immutable completed run exists; use --out NEW_DIRECTORY')
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    d = load()
    feas = d['feas'].astype(bool)
    so = d['scene_of']
    costs = d['costs'][d['act_idx']]
    scenes_all, rows = scene_row_matrix(so)
    feas_scene_all = feas[rows].any(1)
    dev, _ = split_scenes(d, feas)                      # archived 43 feasible scenes
    dev_all = np.array(sorted(set(so))[:86])            # sensitivity: same 86 scenes, unfiltered
    ev_full = scenes_all[~np.isin(scenes_all, dev)]     # 213 scenes, no feasibility filter
    ev_rest = scenes_all[~np.isin(scenes_all, dev_all)]  # 170 scenes, unfiltered dev
    sub = {name: (rows_for(sc, scenes_all, rows),
                  feas_scene_all[np.searchsorted(scenes_all, sc)])
           for name, sc in (('dev', dev), ('eval_full', ev_full),
                            ('all256', scenes_all), ('dev_all', dev_all),
                            ('eval_rest', ev_rest))}
    n_inf = int((~feas_scene_all).sum())
    result = {
        'mode': 'all-scene deployment, no oracle-feasibility filter',
        'source_commit': source_commit(),
        'source_commit_note': 'git HEAD at run time; the lane edits to this script are uncommitted',
        'script_sha256': script_sha256(),
        'command': ' '.join(sys.argv),
        'table_sha256': hashlib.sha256((ROOT / 'artifacts/p123_upgrade/p2/p2_table.npz').read_bytes()).hexdigest(),
        'torch_threads': 4,
        'scene_universe': {
            'n_scenes': int(len(scenes_all)),
            'n_oracle_feasible': int(len(scenes_all) - n_inf),
            'n_oracle_infeasible': n_inf,
            'oracle_infeasible_share': round(n_inf / len(scenes_all), 4),
            'n_candidates_per_scene': int(rows.shape[1]),
            'dev_selection_scenes': int(len(dev)),
            'eval_full_scenes': int(len(ev_full)),
            'eval_full_oracle_infeasible': int((~sub['eval_full'][1]).sum()),
            'eval_full_oracle_feasible': int(sub['eval_full'][1].sum()),
            'sensitivity_dev_all_scenes': int(len(dev_all)),
            'sensitivity_eval_rest_scenes': int(len(ev_rest))},
        'protocol': {
            'dev_scene_selection': 'archived split: sorted first-86 scenes intersected with oracle-feasible = 43; used for threshold selection only',
            'dev_oracle_filter_disclosure': 'the dev scene set itself was defined with dev labels (candidate feasibility); deployment evaluation applies no filter',
            'deployment_universe': 'eval_full = all 213 scenes outside dev, including the 117 oracle-infeasible ones; all256 adds the in-sample dev scenes',
            'budget_rule': 'dev coverage closest to target, tie -> largest threshold',
            'interpretation': 'dev-matched-policy extrapolation; eval coverage is not forced equal to dev coverage',
            'threshold_correlation': 'the ~350-400 dev ladder thresholds per model, and the 25k-31k diagnostic breakpoints collapsing to ~1.7-1.9k action vectors, are nested correlated policies rather than independent replicates; no multiplicity correction is applied and none should be read in',
            'net_benefit_units': 'one unit reward per successful scene minus one times archived action cost; illustrative utility, not football/business value',
            'bootstrap': {'n': N_BOOT, 'unit': 'scene', 'paired': True,
                          'recomputation': 'each replicate recomputes each policy full ratio'},
            'per_scene_csv': 'FRONTIER_FULLSCENE_PERSCENE.csv holds the primary protocol only; scene_set dev43/eval213 partition all 256 scenes; acted=False rows carry selected_row=-1, action_cost 0 and net_benefit 0; score_used is the raw negative logit, or the calibrated score for isotonic_clean',
            'sensitivity': 'dev_all = same first-86 scenes without feasibility filter; thresholds selected on it, evaluated on the remaining 170 scenes'},
        'deployment': {},
        'diagnostic_oracle_frontier': {
            'warning': 'DIAGNOSTIC ONLY: thresholds enumerated and selected with eval labels; not deployable, no valid confidence interval',
            'universe': 'eval_full (213 scenes, no feasibility filter)',
            'columns': ['tau', 'coverage', 'own_quality', 'success_rate_per_scene',
                        'total_cost', 'net_benefit_per_scene', 'n_acted_infeasible'],
            'envelope_rule': 'non-dominated over (own_quality, net_benefit_per_scene, -total_cost); dominated thresholds are omitted from the record',
            'pairs': {}},
        'wall_seconds': None}
    csv_path = out / 'FRONTIER_FULLSCENE_PERSCENE.csv'
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['pair', 'encoder', 'seed', 'model', 'target_dev_coverage', 'tau',
                    'dev_coverage_selected', 'scene_set', 'scene', 'scene_oracle_feasible',
                    'acted', 'success', 'selected_row', 'selected_candidate',
                    'score_used', 'action_cost', 'net_benefit'])
        sources = logit_sources(d, dev)
        for key, entry in sources.items():
            enc, seed = key.rsplit('_s', 1)
            seed = int(seed)
            scores = entry['scores']
            R = {'checkpoints': entry['checkpoints'], 'models': {}}
            per_model = {}
            for m, score in scores.items():
                ts, _, act, _, _ = frontier(score, so, costs, feas, dev)
                per_model[m] = {'score': score, 'ladder': ts, 'dev_acted': act}
            for budget in BUDGETS:
                for m, pm in per_model.items():
                    k, tau, devcov = budget_threshold(pm['dev_acted'], pm['ladder'], budget)
                    dep = {}
                    arrays = {}
                    for name in ('eval_full', 'all256', 'dev'):
                        rmat, fvec = sub[name]
                        a, s, c, srows = grouped_outcomes(pm['score'], rmat, costs, feas, tau)
                        dep[name] = deployment_metrics(a, s, c, fvec)
                        dep[name]['ci95_scene_bootstrap'] = deployment_bootstrap_ci(
                            a, s, c, rng_for(key, m, budget, name))
                        if name == 'eval_full':
                            arrays = {'a': a, 's': s, 'c': c, 'sel': srows, 'feas': fvec}
                    # sensitivity: dev side unfiltered as well
                    ts2, _, act2, _, _ = frontier(pm['score'], so, costs, feas, dev_all)
                    k2, tau2, devcov2 = budget_threshold(act2, ts2, budget)
                    rmat, fvec = sub['eval_rest']
                    a2, s2, c2, _ = grouped_outcomes(pm['score'], rmat, costs, feas, tau2)
                    sens = deployment_metrics(a2, s2, c2, fvec)
                    sens['ci95_scene_bootstrap'] = deployment_bootstrap_ci(
                        a2, s2, c2, rng_for(key, m, budget, 'eval_rest'))
                    R['models'].setdefault(m, {'n_dev_action_policies': int(len(pm['ladder'])),
                                               'budget_deployment': []})
                    R['models'][m]['budget_deployment'].append({
                        'target_dev_coverage': budget, 'tau': tau, 'dev_coverage_selected': devcov,
                        'deployment_eval_full_213': dep['eval_full'],
                        'deployment_all256_insample': dep['all256'],
                        'deployment_dev43_insample': dep['dev'],
                        'sensitivity_unfiltered_dev_eval_rest_170': dict(
                            sens, tau=tau2, dev_coverage_selected=devcov2,
                            note='dev_all = first-86 scenes without feasibility filter'),
                        'paired_vs_clean': {}})
                    per_model[m].setdefault('store', {})[budget] = arrays
                    # per-scene CSV rows (primary protocol only)
                    for name in ('dev', 'eval_full'):
                        rmat2, fvec2 = sub[name]
                        a3, s3, c3, srows3 = grouped_outcomes(pm['score'], rmat2, costs, feas, tau)
                        sc_list = dev if name == 'dev' else ev_full
                        for i, sc in enumerate(sc_list):
                            rr = int(srows3[i]) if a3[i] else -1
                            w.writerow([key, enc, seed, m, budget, round(tau, 6),
                                        round(devcov, 4),
                                        'dev43' if name == 'dev' else 'eval213', int(sc),
                                        bool(fvec2[i]), bool(a3[i]), bool(s3[i]), rr,
                                        int(d['act_idx'][rr]) if rr >= 0 else -1,
                                        round(float(pm['score'][rr]), 6) if rr >= 0 else '',
                                        round(float(costs[rr]), 6) if rr >= 0 else 0.0,
                                        round(float(s3[i]) - float(c3[i]), 6)])
                # paired comparisons on the deployment universe (same scenes)
                for m1, m2 in (('flipmine', 'clean'), ('isotonic_clean', 'clean')):
                    A = per_model[m1]['store'][budget]
                    B = per_model[m2]['store'][budget]
                    ci = paired_deployment_ci(A['a'], A['s'], A['c'], B['a'], B['s'], B['c'],
                                              rng_for(key, m1, m2, budget, 'paired'))
                    R['models'][m1]['budget_deployment'][-1]['paired_vs_clean'] = {
                        'other_model': m2, 'difference_ci95_scene_bootstrap': ci,
                        'note': 'positive means %s minus %s on the same 213 scenes' % (m1, m2)}
            # diagnostic oracle-label frontier on the deployment universe
            orc = {}
            for m, pm in per_model.items():
                rmat, fvec = sub['eval_full']
                ts_or, curve, n_raw = oracle_label_curve(pm['score'], rmat, costs, feas, fvec)
                ok = [row for row in curve if row[1] and row[1] > 0]
                best_net = max(ok, key=lambda r: r[5])
                best_q = max(ok, key=lambda r: (r[2], r[1]))
                dep_cov = R['models'][m]['budget_deployment'][1]['deployment_eval_full_213']['coverage']
                matched = min(ok, key=lambda r: (abs(r[1] - dep_cov), -r[1]))
                envelope = pareto_envelope(curve)
                orc[m] = {'n_thresholds_enumerated': n_raw,
                          'n_action_vectors': int(len(ts_or)),
                          'n_envelope_points': int(len(envelope)),
                          'envelope': envelope,
                          'oracle_best_net_benefit_point': best_net,
                          'oracle_best_own_quality_point': best_q,
                          'coverage_matched_to_deployed_0.5': matched,
                          'deployed_0.5_coverage': dep_cov,
                          'selection_bias_note': 'points selected on eval labels; optimistic upper bound, no valid CI'}
            result['diagnostic_oracle_frontier']['pairs'][key] = orc
            result['deployment'][key] = R
            print(key, {m: v['n_dev_action_policies'] for m, v in R['models'].items()}, flush=True)
    result['wall_seconds'] = time.monotonic() - start
    (out / 'FRONTIER_FULLSCENE.json').write_text(json.dumps(result, indent=2))
    print('wrote', out / 'FRONTIER_FULLSCENE.json', 'and', csv_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--mode', choices=('repair', 'fullscene', 'both'), default='both')
    args = ap.parse_args()
    if args.mode in ('repair', 'both'):
        run_repair(args.out)
    if args.mode in ('fullscene', 'both'):
        run_fullscene(args.out)


if __name__ == '__main__':
    main()
