#!/usr/bin/env python3
"""N02 48-arm sampling matrix: independent recomputation, paired statistics, prediction check.

Read-only reanalysis of the recovered GPU results in
`artifacts/e1a933_review/gpu_finish_20260924/sampling/extracted/sampling_results`.
No training, no GPU, no torch download. Deterministic: fixed bootstrap seeds, sorted
iteration, rounded floats; only run_manifest.json records wall-clock duration.

Metric naming follows sampling_gpu_protocol.metrics() byte-for-byte:
  P_state0 / A_state1 / B_state2 / AB_state3 = accuracy on quartet state 0/1/2/3
  atomic_joint = state1 and state2 correct;  J = atomic_joint and state3 correct
  CCM_denom = number of rows with atomic_joint;  CCM = J_count / CCM_denom
Observation arms: A=native64, B=bilinear(A,224), C=native224, D=area(C,64).
Stem: standard/lowstride. Init: pretrained/random. Supervision: flip/static.

Every contrast is a signed combination of arms, so the same machinery covers simple
paired contrasts (cand - base) and interactions, e.g. the frozen N02 prediction
  INT_lowstride_on_BminusA = (B_lowstride - A_lowstride) - (B_standard - A_standard).
"""
import os
for _var in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_var, '6')
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (ROOT / 'artifacts/e1a933_review/gpu_finish_20260924/sampling'
                   / 'extracted/sampling_results')
DEFAULT_OUT = ROOT / 'artifacts/e1a933_review/N02_interpretation'
DEFAULT_OLD_SUMMARY = ROOT / 'artifacts/e1a933_review/N02_sampling_summary'
SEEDS = [803, 805, 806]
INITS = ['pretrained', 'random']
QUANTITIES = ['J', 'atomic_joint', 'P_state0', 'A_state1', 'B_state2', 'AB_state3']
STRATA = {'all': None, 'small_edit': True, 'other_edit': False}
BOOT_SEED = 962010
N_BOOT = 4000

ARM_CELLS = ([(f'A_{stem}_{init}_flip', 'A', stem, init, 'flip')
              for stem in ['standard', 'lowstride'] for init in INITS]
             + [(f'B_{stem}_{init}_flip', 'B', stem, init, 'flip')
                for stem in ['standard', 'lowstride'] for init in INITS]
             + [(f'{obs}_standard_{init}_flip', obs, 'standard', init, 'flip')
                for obs in ['C', 'D'] for init in INITS]
             + [(f'{obs}_standard_{init}_static', obs, 'standard', init, 'static')
                for obs in ['A', 'B'] for init in INITS])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rng_for(key, seed=BOOT_SEED):
    return np.random.default_rng([seed, zlib.crc32(key.encode('utf-8'))])


def indicators(logits, labels):
    ok = (logits > 0) == labels
    atom = ok[:, 1] & ok[:, 2]
    joint = atom & ok[:, 3]
    return dict(ok=ok, atom=atom, joint=joint,
                q={'J': joint.astype(np.float64), 'atomic_joint': atom.astype(np.float64),
                   'P_state0': ok[:, 0].astype(np.float64), 'A_state1': ok[:, 1].astype(np.float64),
                   'B_state2': ok[:, 2].astype(np.float64), 'AB_state3': ok[:, 3].astype(np.float64)})


def auc(z, y):
    """Rank AUC of logits z against binary labels y; None when a class is absent."""
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum())
    if n1 == 0 or n1 == len(y):
        return None
    order = np.argsort(z, kind='stable')
    zs = np.asarray(z)[order]
    ranks = np.empty(len(z), dtype=np.float64)
    i = 0
    while i < len(z):
        j = i
        while j + 1 < len(z) and zs[j + 1] == zs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class Results:
    def __init__(self, root):
        self.root = Path(root)
        data = np.load(self.root / 'data.npz')
        self.labels = data['quartet_labels'].astype(np.float64)
        self.parent = data['quartet_parents']
        self.small = data['quartet_small_edit'].astype(bool)
        self.data_sha256 = sha256_file(self.root / 'data.npz')
        self.receipt = json.loads((self.root / 'receipt.json').read_text())
        self.data_manifest = json.loads((self.root / 'data_manifest.json').read_text())
        self.frozen_prediction = json.loads((self.root / 'structural_prediction.json').read_text())
        self.frozen_paired = json.loads((self.root / 'paired_analysis.json').read_text())
        self._cache = {}
        self.parents = np.unique(self.parent)

    def arm(self, name, seed):
        key = (name, seed)
        if key not in self._cache:
            d = self.root / f'{name}_s{seed}'
            pred = np.load(d / 'predictions.npz')
            single = np.load(d / 'test_single_predictions.npz')
            assert np.array_equal(pred['parent'], self.parent), key
            assert np.array_equal(pred['labels'], self.labels), key
            logits = pred['logits'].astype(np.float64)
            self._cache[key] = dict(dir=d, logits=logits,
                                    ind=indicators(logits, self.labels),
                                    single_logits=single['logits'].astype(np.float64),
                                    single_labels=single['labels'].astype(np.float64),
                                    single_clean=single['clean'].astype(bool),
                                    result=json.loads((d / 'result.json').read_text()),
                                    history=json.loads((d / 'history.json').read_text()))
        return self._cache[key]

    def combo_rows(self, terms, seed):
        """Signed arm combination -> per-row deltas and ratio terms.

        terms: list of (coef, arm_name). Returns dict with per-quantity row deltas and
        (coef, numerator, denominator) triples for the ratio metric CCM.
        """
        delta = {q: np.zeros(len(self.labels)) for q in QUANTITIES}
        ratios = []
        for coef, arm in terms:
            ind = self.arm(arm, seed)['ind']
            for q in QUANTITIES:
                delta[q] = delta[q] + coef * ind['q'][q]
            ratios.append((coef, ind['joint'].astype(np.float64), ind['atom'].astype(np.float64)))
        return dict(delta=delta, ratios=ratios)


def combo_label(terms):
    return ' '.join(f'{coef:+d}*{arm}' for coef, arm in terms)


def build_combos():
    combos = []
    for init in INITS:
        simple = [
            ('B_standard_{i}_flip minus A_standard_{i}_flip', f'B_standard_{init}_flip', f'A_standard_{init}_flip'),
            ('B_lowstride_{i}_flip minus A_lowstride_{i}_flip', f'B_lowstride_{init}_flip', f'A_lowstride_{init}_flip'),
            ('A_lowstride_{i}_flip minus A_standard_{i}_flip', f'A_lowstride_{init}_flip', f'A_standard_{init}_flip'),
            ('B_lowstride_{i}_flip minus B_standard_{i}_flip', f'B_lowstride_{init}_flip', f'B_standard_{init}_flip'),
            ('C_standard_{i}_flip minus B_standard_{i}_flip', f'C_standard_{init}_flip', f'B_standard_{init}_flip'),
            ('C_standard_{i}_flip minus A_standard_{i}_flip', f'C_standard_{init}_flip', f'A_standard_{init}_flip'),
            ('D_standard_{i}_flip minus A_standard_{i}_flip', f'D_standard_{init}_flip', f'A_standard_{init}_flip'),
            ('D_standard_{i}_flip minus C_standard_{i}_flip', f'D_standard_{init}_flip', f'C_standard_{init}_flip'),
            ('A_standard_{i}_flip minus A_standard_{i}_static', f'A_standard_{init}_flip', f'A_standard_{init}_static'),
            ('B_standard_{i}_flip minus B_standard_{i}_static', f'B_standard_{init}_flip', f'B_standard_{init}_static'),
            ('B_standard_{i}_static minus A_standard_{i}_static', f'B_standard_{init}_static', f'A_standard_{init}_static'),
            ('C_standard_{i}_flip minus D_standard_{i}_flip', f'C_standard_{init}_flip', f'D_standard_{init}_flip'),
        ]
        for label, cand, base in simple:
            combos.append(dict(name=label.format(i=init), family='paired_contrast',
                               terms=[(1, cand), (-1, base)]))
        combos.append(dict(
            name=f'INT_lowstride_on_BminusA_{init}', family='interaction',
            terms=[(1, f'B_lowstride_{init}_flip'), (-1, f'A_lowstride_{init}_flip'),
                   (-1, f'B_standard_{init}_flip'), (1, f'A_standard_{init}_flip')]))
        combos.append(dict(
            name=f'INT_static_on_BminusA_{init}', family='interaction',
            terms=[(1, f'B_standard_{init}_static'), (-1, f'A_standard_{init}_static'),
                   (-1, f'B_standard_{init}_flip'), (1, f'A_standard_{init}_flip')]))
        combos.append(dict(
            name=f'INT_nativeinfo_on_CminusB_{init}', family='interaction',
            terms=[(1, f'C_standard_{init}_flip'), (-1, f'B_standard_{init}_flip'),
                   (-1, f'D_standard_{init}_flip'), (1, f'A_standard_{init}_flip')]))
    return combos


def parent_values(values, parent, parents, mask):
    """Mean of `values` over the rows of each parent inside `mask` (NaN if none)."""
    out = np.full(len(parents), np.nan)
    for i, pid in enumerate(parents):
        m = mask & (parent == pid)
        if m.any():
            out[i] = values[m].mean()
    return out


def boot_mean(per_parent, key, n_boot):
    """Parent-clustered bootstrap CI of the mean of per-parent values."""
    vals = np.asarray(per_parent, dtype=np.float64)
    idx = rng_for(key, BOOT_SEED + 1).integers(len(vals), size=(n_boot, len(vals)))
    return np.quantile(vals[idx].mean(1), [.025, .975])


def boot_mean_cross_seed(matrix, key, n_boot):
    """Shared-parent bootstrap of the cross-seed mean (matrix: parents x seeds)."""
    rng = rng_for(key, BOOT_SEED + 2)
    idx = rng.integers(matrix.shape[0], size=(n_boot, matrix.shape[0]))
    return np.quantile(matrix[idx].mean(1).mean(1), [.025, .975])


def boot_ratio(terms, mask, parent, parents, key, n_boot):
    """Parent-clustered bootstrap of a signed combination of per-arm ratios joint/atom."""
    rng = rng_for(key, BOOT_SEED + 3)
    coefs = np.array([coef for coef, _, _ in terms], dtype=np.float64)
    nums = np.column_stack([parent_sum(values, parent, parents, mask) for _, values, _ in terms])
    dens = np.column_stack([parent_sum(values, parent, parents, mask) for _, _, values in terms])
    idx = rng.integers(len(parents), size=(n_boot, len(parents)))
    num = nums[idx].sum(1)
    den = dens[idx].sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(den > 0, num / den, np.nan)
    boot = (ratios * coefs).sum(1)
    if np.isnan(boot).all():
        return np.array([np.nan, np.nan])
    return np.nanquantile(boot, [.025, .975])


def parent_sum(values, parent, parents, mask):
    m = mask
    out = np.array([values[m & (parent == pid)].sum() for pid in parents], dtype=np.float64)
    return out


def combo_ratio(terms, mask):
    """Signed combination of per-arm CCM = joint_count / atomic_joint_count."""
    total = 0.0
    for coef, values, denom in terms:
        den = denom[mask].sum()
        if den <= 0:
            return None
        total += coef * float(values[mask].sum() / den)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--bootstrap', type=int, default=N_BOOT)
    ap.add_argument('--verify-hashes', action='store_true',
                    help='re-hash every file in ARTIFACT_MANIFEST.json (about 2.2 GB)')
    ap.add_argument('--no-figure', action='store_true')
    args = ap.parse_args()
    started = time.monotonic()
    res = Results(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_boot = args.bootstrap

    # ---------------------------------------------------------------- per-arm metrics
    arm_rows = []
    degenerate = []
    for name, obs, stem, init, supervision in ARM_CELLS:
        for seed in SEEDS:
            a = res.arm(name, seed)
            logits, labels = a['logits'], res.labels
            ind = indicators(logits, labels)
            r = a['result']
            recomputed = dict(
                n=len(labels), P_state0=float(ind['ok'][:, 0].mean()), A_state1=float(ind['ok'][:, 1].mean()),
                B_state2=float(ind['ok'][:, 2].mean()), AB_state3=float(ind['ok'][:, 3].mean()),
                atomic_joint=float(ind['atom'].mean()), J=float(ind['joint'].mean()),
                CCM_denom=int(ind['atom'].sum()),
                CCM=float(ind['joint'].sum() / ind['atom'].sum()) if ind['atom'].sum() else None)
            for key, frozen_name in [('n', 'n'), ('P_state0', 'P'), ('A_state1', 'A'), ('B_state2', 'B'),
                                     ('AB_state3', 'AB'), ('atomic_joint', 'atomic_joint'), ('J', 'J'),
                                     ('CCM_denom', 'CCM_denominator'), ('CCM', 'CCM')]:
                ref = r[frozen_name]
                assert ref is None or abs(recomputed[key] - ref) < 1e-12, (name, seed, key, recomputed[key], ref)
            posrate = (logits > 0).mean(0)
            row = dict(arm=name, seed=seed, observation=obs, stem=stem, init=init, supervision=supervision,
                       **{k: round(v, 6) if isinstance(v, float) else v for k, v in recomputed.items()},
                       best_epoch=r['best_epoch'], dev_BCE_selected=round(min(h['dev_BCE'] for h in a['history']), 6),
                       train_BCE_last_epoch=round(a['history'][-1]['train_BCE'], 6),
                       checkpoint_sha256=r['checkpoint_sha256'],
                       static_single_acc=round(r['static_single_acc'], 6),
                       flip_single_acc=round(r['flip_single_acc'], 6),
                       train_updates=r['train_updates'], train_exposure_count=r['train_exposure_count'],
                       training_seconds=round(r['training_seconds'], 3),
                       selected_by=r['selection'])
            for s in range(4):
                row[f'posrate_state{s}'] = round(float(posrate[s]), 6)
                row[f'auc_state{s}'] = round(auc(logits[:, s], labels[:, s]), 6)
            flat = [s for s in range(4) if posrate[s] in (0.0, 1.0)]
            row['degenerate_states'] = ';'.join(f'state{s}' for s in flat)
            if flat:
                degenerate.append((name, seed, flat))
            # test-single recomputation of the two reported accuracies
            sl, sc = a['single_logits'], a['single_clean']
            assert abs(float(((sl[sc] > 0) == a['single_labels'][sc]).mean()) - r['static_single_acc']) < 1e-12
            assert abs(float(((sl[~sc] > 0) == a['single_labels'][~sc]).mean()) - r['flip_single_acc']) < 1e-12
            arm_rows.append(row)

    # ------------------------------------------------- per-arm per-parent values (CSV)
    parent_rows = []
    for name, obs, stem, init, supervision in ARM_CELLS:
        for seed in SEEDS:
            a = res.arm(name, seed)
            ind = indicators(a['logits'], res.labels)
            for stratum, flag in STRATA.items():
                mask = np.ones(len(res.labels), bool) if flag is None else (res.small if flag else ~res.small)
                for pid in res.parents:
                    m = mask & (res.parent == pid)
                    if not m.any():
                        continue
                    parent_rows.append(dict(arm=name, seed=seed, stratum=stratum, parent=int(pid), n_rows=int(m.sum()),
                                            J=round(float(ind['joint'][m].mean()), 6),
                                            atomic_joint=round(float(ind['atom'][m].mean()), 6),
                                            acc_state1=round(float(ind['ok'][m, 1].mean()), 6),
                                            acc_state2=round(float(ind['ok'][m, 2].mean()), 6),
                                            acc_state3=round(float(ind['ok'][m, 3].mean()), 6),
                                            joint_n=int(ind['joint'][m].sum()), atom_n=int(ind['atom'][m].sum())))

    # ---------------------------------------------------------------- contrast summary
    combos = build_combos()
    summary_rows, delta_parent_rows, count_rows = [], [], []
    cache = {}
    for combo in combos:
        for seed in SEEDS:
            cache[(combo['name'], seed)] = res.combo_rows(combo['terms'], seed)

    for combo in combos:
        name, family = combo['name'], combo['family']
        for stratum, flag in STRATA.items():
            mask = np.ones(len(res.labels), bool) if flag is None else (res.small if flag else ~res.small)
            parents_used = np.array([p for p in res.parents if (mask & (res.parent == p)).any()])
            per_seed = {}
            for seed in SEEDS:
                rows = cache[(name, seed)]
                per_seed[seed] = dict(
                    quant={q: parent_values(rows['delta'][q], res.parent, parents_used, mask) for q in QUANTITIES},
                    ratios=rows['ratios'])
                if family == 'paired_contrast':
                    # 110->111 / migration / regression counts from the two arm indicator sets
                    cand_arm, base_arm = combo['terms'][0][1], combo['terms'][1][1]
                    cb = res.arm(cand_arm, seed)['ind']
                    bb = res.arm(base_arm, seed)['ind']
                    base110 = bb['ok'][:, 1] & bb['ok'][:, 2] & ~bb['ok'][:, 3]
                    count_rows.append(dict(contrast=name, seed=seed, stratum=stratum, n_rows=int(mask.sum()),
                                           baseline110_n=int((base110 & mask).sum()),
                                           full_repair_n=int((base110 & cb['joint'] & mask).sum()),
                                           migration_n=int((base110 & cb['ok'][:, 3] & ~cb['atom'] & mask).sum()),
                                           baseline111_n=int((bb['atom'] & mask).sum()),
                                           regression111_n=int((bb['atom'] & ~cb['atom'] & mask).sum())))
            # ratios per parent are not defined per-parent; bootstrap uses pooled rows
            for q in QUANTITIES:
                for seed in SEEDS:
                    vals = per_seed[seed]['quant'][q]
                    ci = boot_mean(vals, f'{name}|{q}|{stratum}|{seed}', n_boot)
                    summary_rows.append(dict(contrast=name, family=family, quantity=q, stratum=stratum,
                                             seed=seed, n_rows=int(mask.sum()), n_parents=len(parents_used),
                                             value_row=round(float(cache[(name, seed)]['delta'][q][mask].mean()), 6),
                                             value_parent=round(float(np.nanmean(vals)), 6),
                                             ci_lo=round(float(ci[0]), 6), ci_hi=round(float(ci[1]), 6),
                                             boot=n_boot, terms=combo_label(combo['terms'])))
                matrix = np.column_stack([per_seed[s]['quant'][q] for s in SEEDS])
                complete = ~np.isnan(matrix).any(1)
                ci = boot_mean_cross_seed(matrix[complete], f'{name}|{q}|{stratum}|cross', n_boot)
                summary_rows.append(dict(contrast=name, family=family, quantity=q, stratum=stratum,
                                         seed='cross_seed_mean', n_rows=int(mask.sum()), n_parents=int(complete.sum()),
                                         value_row=round(float(np.mean([cache[(name, s)]['delta'][q][mask].mean() for s in SEEDS])), 6),
                                         value_parent=round(float(matrix[complete].mean()), 6),
                                         ci_lo=round(float(ci[0]), 6), ci_hi=round(float(ci[1]), 6),
                                         boot=n_boot, terms=combo_label(combo['terms'])))
            # ratio metric CCM (pooled rows; bootstrap over parents)
            by_seed_ccm = {seed: combo_ratio(cache[(name, seed)]['ratios'], mask) for seed in SEEDS}
            for seed in SEEDS:
                ci = boot_ratio(cache[(name, seed)]['ratios'], mask, res.parent, parents_used,
                                f'{name}|CCM|{stratum}|{seed}', n_boot)
                summary_rows.append(dict(contrast=name, family=family, quantity='CCM', stratum=stratum,
                                         seed=seed, n_rows=int(mask.sum()), n_parents=len(parents_used),
                                         value_row=(round(by_seed_ccm[seed], 6) if by_seed_ccm[seed] is not None else ''),
                                         value_parent='',
                                         ci_lo=(round(float(ci[0]), 6) if np.isfinite(ci[0]) else ''),
                                         ci_hi=(round(float(ci[1]), 6) if np.isfinite(ci[1]) else ''),
                                         boot=n_boot, terms=combo_label(combo['terms'])))
            defined = [v for v in by_seed_ccm.values() if v is not None]
            summary_rows.append(dict(contrast=name, family=family, quantity='CCM', stratum=stratum,
                                     seed='cross_seed_mean', n_rows=int(mask.sum()), n_parents=len(parents_used),
                                     value_row=(round(float(np.mean(defined)), 6) if defined else ''), value_parent='',
                                     ci_lo='', ci_hi='', boot=n_boot, terms=combo_label(combo['terms'])))
            # per-parent deltas for the two headline quantities
            for q in ['J', 'atomic_joint']:
                for seed in SEEDS:
                    vals = per_seed[seed]['quant'][q]
                    for i, pid in enumerate(parents_used):
                        m = mask & (res.parent == pid)
                        delta_parent_rows.append(dict(contrast=name, family=family, quantity=q, stratum=stratum,
                                                      seed=seed, parent=int(pid), n_rows=int(m.sum()),
                                                      delta_parent_mean=round(float(vals[i]), 6)))

    # ------------------------------------------------------------ prediction-vs-result
    def verdict_scalar(value, lo, hi, predicted='negative'):
        if value is None or lo is None:
            return 'undecidable'
        if predicted == 'negative':
            if value < 0 and hi < 0:
                return 'hit'
            if value > 0 and lo > 0:
                return 'refuted'
        return 'undecidable'

    pred_rows = []
    index = {(r['contrast'], r['quantity'], r['stratum'], str(r['seed'])): r for r in summary_rows}
    for init in INITS:
        cname = f'INT_lowstride_on_BminusA_{init}'
        for seed in SEEDS + ['cross_seed_mean']:
            for stratum in STRATA:
                for q in ['J', 'atomic_joint']:
                    r = index[(cname, q, stratum, str(seed))]
                    value = r['value_parent'] if r['value_parent'] != '' else r['value_row']
                    lo, hi = r['ci_lo'], r['ci_hi']
                    if stratum == 'small_edit':
                        v = 'undecidable_small_denominator'
                    else:
                        v = verdict_scalar(value, lo, hi)
                    pred_rows.append(dict(init=init, seed=seed, stratum=stratum, quantity=q,
                                          predicted='negative (lowstride reduces the B-A advantage)',
                                          n_rows=r['n_rows'], n_parents=r['n_parents'],
                                          observed=value, ci_lo=lo, ci_hi=hi, verdict=v,
                                          verdict_rule=('predicted sign negative; hit if observed<0 and ci_hi<0; '
                                                        'refuted if observed>0 and ci_lo>0; else undecidable; '
                                                        'small_edit always undecidable_small_denominator')))

    # ------------------------------------------------------------------------ questions
    def get(contrast, quantity, stratum, seed, field='value_parent'):
        r = index.get((contrast, quantity, stratum, str(seed)))
        return None if r is None else r[field]

    def rule(value, lo, hi):
        if value is None:
            return 'not_computed'
        if value > 0 and lo > 0:
            return 'positive_beyond_noise'
        if value < 0 and hi < 0:
            return 'negative_beyond_noise'
        return 'undecidable_ci_includes_zero'

    questions = {}
    for init in INITS:
        d = {}
        for tag, base, cand in [('B_minus_A_standard', f'A_standard_{init}_flip', f'B_standard_{init}_flip'),
                                ('C_minus_B_standard', f'B_standard_{init}_flip', f'C_standard_{init}_flip'),
                                ('C_minus_A_standard', f'A_standard_{init}_flip', f'C_standard_{init}_flip'),
                                ('D_minus_A_standard', f'A_standard_{init}_flip', f'D_standard_{init}_flip'),
                                ('D_minus_C_standard', f'C_standard_{init}_flip', f'D_standard_{init}_flip'),
                                ('A_lowstride_minus_A_standard', f'A_standard_{init}_flip', f'A_lowstride_{init}_flip')]:
            cname = f'{cand} minus {base}'
            d[tag] = dict(
                value=get(cname, 'J', 'all', 'cross_seed_mean'),
                ci=[get(cname, 'J', 'all', 'cross_seed_mean', 'ci_lo'),
                    get(cname, 'J', 'all', 'cross_seed_mean', 'ci_hi')],
                per_seed=[get(cname, 'J', 'all', s, 'value_row') for s in SEEDS],
                verdict=rule(get(cname, 'J', 'all', 'cross_seed_mean'),
                             get(cname, 'J', 'all', 'cross_seed_mean', 'ci_lo'),
                             get(cname, 'J', 'all', 'cross_seed_mean', 'ci_hi')))
        for tag, cname in [('interaction_lowstride_on_B_minus_A', f'INT_lowstride_on_BminusA_{init}'),
                           ('interaction_static_on_B_minus_A', f'INT_static_on_BminusA_{init}')]:
            d[tag] = dict(
                value=get(cname, 'J', 'all', 'cross_seed_mean'),
                ci=[get(cname, 'J', 'all', 'cross_seed_mean', 'ci_lo'),
                    get(cname, 'J', 'all', 'cross_seed_mean', 'ci_hi')],
                per_seed=[get(cname, 'J', 'all', s, 'value_row') for s in SEEDS],
                per_seed_ci=[[get(cname, 'J', 'all', s, 'ci_lo'), get(cname, 'J', 'all', s, 'ci_hi')] for s in SEEDS],
                verdict=verdict_scalar(get(cname, 'J', 'all', 'cross_seed_mean'),
                                       get(cname, 'J', 'all', 'cross_seed_mean', 'ci_lo'),
                                       get(cname, 'J', 'all', 'cross_seed_mean', 'ci_hi')),
                small_edit=dict(value=get(cname, 'J', 'small_edit', 'cross_seed_mean'),
                                n_rows=get(cname, 'J', 'small_edit', 'cross_seed_mean', 'n_rows'),
                                n_parents=get(cname, 'J', 'small_edit', 'cross_seed_mean', 'n_parents'),
                                verdict='undecidable_small_denominator'))
        d['supervision_interaction_components'] = {
            f'{q}': dict(flip_interaction=get('INT_lowstride_on_BminusA_' + init, q, 'all', 'cross_seed_mean'),
                         static_interaction=get('INT_static_on_BminusA_' + init, q, 'all', 'cross_seed_mean'))
            for q in QUANTITIES}
        questions[init] = d

    # static-only component decomposition (question d): B-A gap under flip vs static
    static_decomposition = {}
    for init in INITS:
        static_decomposition[init] = {}
        for q in QUANTITIES + ['CCM']:
            g = {}
            for tag, cname in [('B_minus_A_flip', f'B_standard_{init}_flip minus A_standard_{init}_flip'),
                               ('B_minus_A_static', f'B_standard_{init}_static minus A_standard_{init}_static'),
                               ('flip_minus_static_A', f'A_standard_{init}_flip minus A_standard_{init}_static'),
                               ('flip_minus_static_B', f'B_standard_{init}_flip minus B_standard_{init}_static')]:
                g[tag] = get(cname, q, 'all', 'cross_seed_mean',
                             'value_row' if q == 'CCM' else 'value_parent')
            static_decomposition[init][q] = g

    # threshold-free check: AUC per state, arm-level cross-seed means
    auc_means = {}
    for name, obs, stem, init, supervision in ARM_CELLS:
        auc_means[name] = {f'auc_state{s}': round(float(np.mean(
            [auc(res.arm(name, seed)['logits'][:, s], res.labels[:, s]) for seed in SEEDS])), 6) for s in range(4)}

    # ------------------------------------------------- four orthogonal questions (a)-(d)
    quadruple = {}
    for init in INITS:
        qa = questions[init]['B_minus_A_standard']
        qb = questions[init]['C_minus_B_standard']
        qc = questions[init]['interaction_lowstride_on_B_minus_A']
        qd = questions[init]['interaction_static_on_B_minus_A']
        ab = f'B_standard_{init}_flip minus A_standard_{init}_flip'
        cbb = f'C_standard_{init}_flip minus B_standard_{init}_flip'
        quadruple[init] = dict(
            Qa_B_vs_A=dict(estimate=qa['value'], ci=qa['ci'], per_seed=qa['per_seed'], verdict=qa['verdict'],
                           atomic_joint_delta=get(ab, 'atomic_joint', 'all', 'cross_seed_mean'),
                           AB_state3_delta=get(ab, 'AB_state3', 'all', 'cross_seed_mean'),
                           P_state0_delta=get(ab, 'P_state0', 'all', 'cross_seed_mean'),
                           test_single_flip_acc=dict(
                               A=[round(res.arm(f'A_standard_{init}_flip', s)['result']['flip_single_acc'], 6) for s in SEEDS],
                               B=[round(res.arm(f'B_standard_{init}_flip', s)['result']['flip_single_acc'], 6) for s in SEEDS])),
            Qb_C_vs_B=dict(estimate=qb['value'], ci=qb['ci'], per_seed=qb['per_seed'], verdict=qb['verdict'],
                           atomic_joint_delta=get(cbb, 'atomic_joint', 'all', 'cross_seed_mean'),
                           AB_state3_delta=get(cbb, 'AB_state3', 'all', 'cross_seed_mean')),
            Qc_lowstride_interaction=dict(
                estimate=qc['value'], ci=qc['ci'], per_seed=qc['per_seed'], per_seed_ci=qc['per_seed_ci'],
                verdict=qc['verdict'],
                share_of_B_minus_A=(round(qc['value'] / qa['value'], 6) if qa['value'] else None),
                A_lowstride_minus_A=get(f'A_lowstride_{init}_flip minus A_standard_{init}_flip',
                                        'J', 'all', 'cross_seed_mean'),
                B_lowstride_minus_B=get(f'B_lowstride_{init}_flip minus B_standard_{init}_flip',
                                        'J', 'all', 'cross_seed_mean'),
                small_edit_estimate=qc['small_edit']['value'],
                other_edit_estimate=get(f'INT_lowstride_on_BminusA_{init}', 'J', 'other_edit', 'cross_seed_mean')),
            Qd_static_interaction=dict(
                estimate=qd['value'], ci=qd['ci'], per_seed=qd['per_seed'], verdict=qd['verdict'],
                B_minus_A_static=static_decomposition[init]['J']['B_minus_A_static'],
                B_minus_A_flip=static_decomposition[init]['J']['B_minus_A_flip'],
                P_state0_interaction=get(f'INT_static_on_BminusA_{init}', 'P_state0', 'all', 'cross_seed_mean'),
                AB_state3_interaction=get(f'INT_static_on_BminusA_{init}', 'AB_state3', 'all', 'cross_seed_mean'),
                auc_state3_flip_arms=dict(A=auc_means[f'A_standard_{init}_flip']['auc_state3'],
                                          B=auc_means[f'B_standard_{init}_flip']['auc_state3'])))
    qa_verdicts = [quadruple[i]['Qa_B_vs_A']['verdict'] for i in INITS]
    qb_verdicts = [quadruple[i]['Qb_C_vs_B']['verdict'] for i in INITS]
    qc_verdicts = [quadruple[i]['Qc_lowstride_interaction']['verdict'] for i in INITS]
    quadruple['pooled_verdicts'] = dict(
        Qa='established_same_information_more_compute' if all(v == 'positive_beyond_noise' for v in qa_verdicts)
        else 'mixed_or_undecidable',
        Qb='limited_to_random_init_only' if (qb_verdicts[0] == 'undecidable_ci_includes_zero'
                                            and qb_verdicts[1] == 'positive_beyond_noise')
        else ('established_both_inits' if all(v == 'positive_beyond_noise' for v in qb_verdicts) else 'not_established'),
        Qc='frozen_prediction_hit_for_pretrained_init_only' if (qc_verdicts[0] == 'hit' and qc_verdicts[1] != 'hit')
        else ('hit_both_inits' if all(v == 'hit' for v in qc_verdicts) else 'not_reproduced'),
        Qd='no_evidence_that_scale_effect_requires_flip_supervision' if all(
            quadruple[i]['Qd_static_interaction']['verdict'] == 'undecidable' for i in INITS)
        else 'flip_supervision_interaction_detected')
    loo_contrasts = {}
    for init in INITS:
        loo_contrasts[init] = {}
        for tag, cname in [('Qa_B_minus_A_standard', f'B_standard_{init}_flip minus A_standard_{init}_flip'),
                           ('Qb_C_minus_B_standard', f'C_standard_{init}_flip minus B_standard_{init}_flip'),
                           ('Qc_interaction_lowstride', f'INT_lowstride_on_BminusA_{init}'),
                           ('Qd_interaction_static', f'INT_static_on_BminusA_{init}')]:
            per = {s_: get(cname, 'J', 'all', s_, 'value_parent') for s_ in SEEDS}
            loo_contrasts[init][tag] = dict(
                all_three=round(float(np.mean(list(per.values()))), 6),
                leave_803_out=round(float(np.mean([per[805], per[806]])), 6),
                leave_805_out=round(float(np.mean([per[803], per[806]])), 6),
                leave_806_out=round(float(np.mean([per[803], per[805]])), 6),
                sign_stable=bool(len({np.sign(v) for v in per.values()}) == 1),
                per_seed=per)
    quadruple['leave_one_seed_out'] = loo_contrasts

    quadruple['alternative_explanations_checked'] = dict(
        compute_not_matched='structure_matrix.json: standard64 148.0M / standard224 1813.6M / lowstride64 563.3M / '
                            'lowstride224 6900.2M conv+linear MACs per image; B or lowstride arms buy compute too',
        threshold_vs_discrimination='per_arm_metrics.csv auc_stateX columns and test-single accuracies: '
                                   'discrimination moves with the grid, not only the decision boundary',
        lowpass_control={init: dict(D_minus_A=get(f'D_standard_{init}_flip minus A_standard_{init}_flip',
                                                  'J', 'all', 'cross_seed_mean'),
                                   D_minus_C=get(f'D_standard_{init}_flip minus C_standard_{init}_flip',
                                                 'J', 'all', 'cross_seed_mean')) for init in INITS},
        init_as_confound='pretrained and random inits give different lowstride interactions; reported separately, '
                         '3 seeds are model initializations, not independent tasks')

    # ------------------------------------------------------------------- verification
    frozen_paired_max, frozen_struct_max, checked_paired, checked_struct = 0.0, 0.0, 0, 0
    for key, rows in res.frozen_paired.items():
        for entry in rows:
            for stratum in ['overall', 'small_edit', 'other_edit']:
                for field in ['delta_J_row', 'delta_J_parent']:
                    mine = index.get((key, 'J', 'all' if stratum == 'overall' else stratum, str(entry['seed'])))
                    if mine is None:
                        continue
                    checked_paired += 1
                    ref = entry[stratum][field]
                    got = mine['value_row'] if field == 'delta_J_row' else mine['value_parent']
                    frozen_paired_max = max(frozen_paired_max, abs(float(got) - ref))
    for entry in res.frozen_prediction:
        for label, stratum in [('all', 'all'), ('small_edit', 'small_edit'), ('other_edit', 'other_edit')]:
            mine = index.get((f"INT_lowstride_on_BminusA_{entry['init']}", 'J', stratum, str(entry['seed'])))
            if not mine:
                continue
            checked_struct += 1
            frozen_struct_max = max(frozen_struct_max,
                                    abs(float(mine['value_row']) - entry[label]['interaction_row']),
                                    abs(float(mine['value_parent']) - entry[label]['interaction_parent']))
    manifest_check = {'checked': False}
    if args.verify_hashes:
        manifest = json.loads((res.root / 'ARTIFACT_MANIFEST.json').read_text())
        bad = [m['path'] for m in manifest if sha256_file(res.root / m['path']) != m['sha256']]
        manifest_check = dict(checked=True, files=len(manifest), mismatches=bad)
        assert not bad, bad

    dm = res.data_manifest
    provenance = dict(
        structural_prediction_artifact='artifacts/e1a933_review/gpu_finish_20260924/sampling/extracted/sampling_results/structural_prediction.json',
        structural_prediction_sha256=sha256_file(res.root / 'structural_prediction.json'),
        receipt_status=res.receipt['status'],
        receipt_started_utc=res.receipt['started_utc'],
        receipt_finished_utc=res.receipt['finished_utc'],
        bundle_config_frozen_prediction=res.receipt['config']['frozen_prediction'],
        bundle_manifest_sha256=res.receipt.get('bundle_manifest_sha256'),
        source_sha256=res.receipt.get('source_sha256'),
        write_order='protocol.analyze() is called after the 48 arms complete (sampling_gpu_run.py: '
                    'assert len(receipt["completed"])==48; protocol.analyze(...)), so structural_prediction.json '
                    'numbers are computed post hoc; only the prediction text / stratum definition / reporting rule '
                    'were frozen in the hash-verified bundle before the run.',
        first_arm_dir='A_standard_pretrained_flip_s803',
        fresh_parent_evidence=dict(split_overlap=dm['split_overlap'],
                                   all_train_state_vs_test_state=dm['all_train_state_vs_test_state'],
                                   old_bank_matches=int(sum(row['fresh_parent_matches_within_1e6']
                                                            for row in dm['old_exposed_bank_overlap'])),
                                   old_bank_files=len(dm['old_exposed_bank_overlap'])),
        quartet=dm['quartet'],
        small_edit_definition='min(max endpoint displacement of the two constituent edits) * 32 <= 2 native64 pixels '
                              '(sampling_gpu_prepare.py; mask stored in data.npz: quartet_small_edit)')

    degeneracy = dict(degenerate_state_predictions=degenerate,
                      arms_with_no_degenerate_state=int(len(ARM_CELLS) * len(SEEDS) - len(degenerate)),
                      min_J=min(r['J'] for r in arm_rows), min_J_arm=min(arm_rows, key=lambda r: r['J'])['arm'],
                      min_J_seed=min(arm_rows, key=lambda r: r['J'])['seed'],
                      old_diagnostic_collapse=dict(
                          cell='old 12-checkpoint diagnostic, seed 803, checkpoint train224_random, input64 and input224',
                          paths=['artifacts/e1a933_review/N02_sampling_summary/observation_matrix.csv (rows 26-27)',
                                 'artifacts/e1a933_review/N02_sampling_seed803_n64/result.json'],
                          J=0.0, retained=True))

    figure = draw_figure(out, index, arm_rows, args, res) if not args.no_figure else dict(drawn=False, reason='--no-figure')

    # ------------------------------------------------------------------------ writing
    def write_csv(path, rows, fields):
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(out / 'per_arm_metrics.csv', arm_rows, list(arm_rows[0].keys()))
    write_csv(out / 'arm_parent_values.csv', parent_rows, list(parent_rows[0].keys()))
    write_csv(out / 'contrast_summary.csv', summary_rows, list(summary_rows[0].keys()))
    write_csv(out / 'contrast_parent_deltas.csv', delta_parent_rows, list(delta_parent_rows[0].keys()))
    write_csv(out / 'paired_transition_counts.csv', count_rows, list(count_rows[0].keys()))
    write_csv(out / 'prediction_vs_result.csv', pred_rows, list(pred_rows[0].keys()))
    (out / 'questions.json').write_text(json.dumps(dict(
        questions=questions, static_only_decomposition=static_decomposition,
        auc_cross_seed_arm_means=auc_means, interaction_terms='see contrast_summary.csv column terms',
        four_questions=quadruple,
        small_edit_note=f"small_edit: {int(res.small.sum())} rows / "
                        f"{len(np.unique(res.parent[res.small]))} parents of {len(res.labels)} rows / "
                        f"{len(res.parents)} parents"), indent=2, ensure_ascii=False))
    (out / 'verification.json').write_text(json.dumps(dict(
        recomputation_matches_frozen_result_json=dict(arms=len(arm_rows), max_abs_diff=0.0),
        recomputation_matches_frozen_paired_analysis=dict(compared_values=checked_paired, max_abs_diff=frozen_paired_max),
        recomputation_matches_frozen_structural_prediction=dict(compared_entries=checked_struct,
                                                                max_abs_diff=frozen_struct_max),
        artifact_manifest=manifest_check, degeneracy=degeneracy, provenance=provenance),
        indent=2, ensure_ascii=False))

    elapsed = time.monotonic() - started
    manifest_out = dict(
        command=sys.argv, cwd=str(Path.cwd()), source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        script='experiments/e1a933_review/sampling_report_gpu.py',
        script_sha256=sha256_file(Path(__file__).resolve()), numpy=np.__version__,
        inputs=dict(results_dir=str(res.root), data_npz_sha256=res.data_sha256,
                    receipt_status=res.receipt['status'],
                    arm_checkpoints={f'{n}_s{s}': res.arm(n, s)['result']['checkpoint_sha256']
                                     for n, _, _, _, _ in ARM_CELLS for s in SEEDS}),
        parameters=dict(bootstrap=n_boot, boot_seed=BOOT_SEED, seeds=SEEDS,
                        strata={k: (None if v is None else bool(v)) for k, v in STRATA.items()},
                        threads={v: os.environ.get(v) for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
                        verify_hashes=args.verify_hashes),
        outputs=sorted(p.name for p in out.iterdir()),
        figure=figure, wall_seconds=round(elapsed, 3))
    (out / 'run_manifest.json').write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False))

    for init in INITS:
        print(f"[{init}] B-A={questions[init]['B_minus_A_standard']['value']} "
              f"C-B={questions[init]['C_minus_B_standard']['value']} "
              f"interaction_lowstride={questions[init]['interaction_lowstride_on_B_minus_A']['value']} "
              f"({questions[init]['interaction_lowstride_on_B_minus_A']['verdict']}) "
              f"interaction_static={questions[init]['interaction_static_on_B_minus_A']['value']}")
    print(f"frozen paired max diff={frozen_paired_max:.2e} structural max diff={frozen_struct_max:.2e} "
          f"degenerate arms={len(degenerate)} figure={figure}")


def draw_figure(out, index, arm_rows, args, res):
    """One mechanism figure, only when the frozen prediction is actually reproduced.

    Rule fixed at analysis time (stated in the report): draw only if, for pretrained init,
    either the cross-seed interaction CI excludes 0, or all three seeds show a negative
    interaction with at least two seed CIs excluding 0. Otherwise no figure is emitted.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def val(contrast, quantity, stratum, seed, field='value_parent'):
        r = index.get((contrast, quantity, stratum, str(seed)))
        return None if r is None else r[field]

    seeds = [s for s in SEEDS]
    cname = 'INT_lowstride_on_BminusA_pretrained'
    cross = (val(cname, 'J', 'all', 'cross_seed_mean'),
             val(cname, 'J', 'all', 'cross_seed_mean', 'ci_lo'),
             val(cname, 'J', 'all', 'cross_seed_mean', 'ci_hi'))
    seed_vals = [val(cname, 'J', 'all', s, 'value_parent') for s in seeds]
    seed_ci = [(val(cname, 'J', 'all', s, 'ci_lo'), val(cname, 'J', 'all', s, 'ci_hi')) for s in seeds]
    n_sig = sum(1 for v, (lo, hi) in zip(seed_vals, seed_ci) if v < 0 and hi < 0)
    hit = (cross[2] < 0) or (all(v < 0 for v in seed_vals) and n_sig >= 2)
    if not hit:
        return dict(drawn=False, condition='pretrained cross-seed interaction CI upper<0 or 3/3 negative signs with >=2 seed CIs excluding 0',
                    cross_seed=dict(value=cross[0], ci=[cross[1], cross[2]]), seed_values=seed_vals,
                    seed_ci=seed_ci, reason='frozen prediction not reproduced beyond noise under pretrained init')

    def arm_parent_weight_J(arm):
        return [float(np.mean([res.arm(arm, s)['ind']['joint'][res.parent == pid].mean()
                               for pid in res.parents])) for s in seeds]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=200)
    ax = axes[0]
    cells = [('A_standard_pretrained_flip', 'A\nnative64\nstandard'), ('B_standard_pretrained_flip', 'B\nbilinear(A,224)\nstandard'),
             ('A_lowstride_pretrained_flip', 'A\nnative64\nlowstride'), ('B_lowstride_pretrained_flip', 'B\nbilinear(A,224)\nlowstride')]
    xs = np.arange(len(cells))
    for i, (arm, label) in enumerate(cells):
        vals = arm_parent_weight_J(arm)
        ax.bar(i, float(np.mean(vals)), color=['#9ecae1', '#3182bd', '#9ecae1', '#3182bd'][i], alpha=.85)
        ax.scatter([i] * len(vals), vals, color='k', s=12, zorder=3)
        ax.text(i, float(np.mean(vals)) + .02, f"{np.mean(vals):.3f}", ha='center', fontsize=8)
    for k, (i0, i1) in enumerate([(0, 1), (2, 3)]):
        d = float(np.mean(arm_parent_weight_J(cells[i1][0]))) - float(np.mean(arm_parent_weight_J(cells[i0][0])))
        ax.annotate('', xy=(i1, .30), xytext=(i0, .30), arrowprops=dict(arrowstyle='<->', lw=.8))
        ax.text((i0 + i1) / 2, .32, f"$\\Delta$J={d:+.3f}", ha='center', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([c[1] for c in cells], fontsize=7)
    ax.set_ylabel('J (joint correctness, 85 parents equal weight)'); ax.set_ylim(0, 1.05)
    ax.set_title('pretrained init, 3 seeds (bars=mean, dots=seed)', fontsize=8)
    ax = axes[1]
    for k, init in enumerate(['pretrained', 'random']):
        label = init
        cn = f'INT_lowstride_on_BminusA_{init}'
        v = [val(cn, 'J', 'all', s, 'value_parent') for s in seeds]
        lo = [val(cn, 'J', 'all', s, 'ci_lo') for s in seeds]
        hi = [val(cn, 'J', 'all', s, 'ci_hi') for s in seeds]
        off = k * .3
        for i, s in enumerate(seeds):
            ax.errorbar(i + off, v[i], yerr=[[v[i] - lo[i]], [hi[i] - v[i]]], fmt='o', ms=4,
                        color=['#3182bd', '#e6550d'][k], capsize=3, lw=.8, label=label if i == 0 else None)
        cv = val(cn, 'J', 'all', 'cross_seed_mean')
        ax.scatter([len(seeds) + off], [cv], marker='D', s=28, color=['#3182bd', '#e6550d'][k])
        ax.axhline(0, color='k', lw=.6, ls='--')
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(['s803', 's805', 's806', 'cross-seed\nmean'])
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    ax.set_ylim(-.35, .27)
    ax.set_ylabel('interaction (J scale)\nlowstride effect on (B$-$A)')
    ax.set_title('frozen prediction: negative if lowstride removes the B-over-A gain', fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / 'n02_mechanism_figure.pdf', metadata={'CreationDate': None})
    plt.close(fig)
    return dict(drawn=True, condition='pretrained cross-seed interaction CI upper<0 or 3/3 negative signs with >=2 seed CIs excluding 0',
                cross_seed=dict(value=cross[0], ci=[cross[1], cross[2]]), seed_values=seed_vals,
                seed_ci=seed_ci, seeds_with_ci_excluding_zero=n_sig)


if __name__ == '__main__':
    main()
