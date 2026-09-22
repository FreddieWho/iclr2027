#!/usr/bin/env python3
"""Validate package kernels on synthetic matrices, never project outcomes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.perturbations import (balanced_sign_patterns, antithetic_fields,
                               weighted_sign_law, worst_case_weights,
                               directional_gram, expected_quadratic)
from core.metric_edit import fit_metric_initializer
from core.relations import make_relational_scenes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must be a new directory')
    rng = np.random.default_rng(17)
    patterns = balanced_sign_patterns([4,4], max_patterns=16, seed=17)
    fields = antithetic_fields(patterns, .2, np.array([1.,.5]))
    uniform = weighted_sign_law(patterns, np.ones(len(patterns)))
    weighted = weighted_sign_law(patterns, worst_case_weights(np.arange(len(patterns)), 2.))
    j = rng.normal(size=(5,8,2))
    gram = directional_gram(j, np.array([1.,.5]))
    predicted = expected_quadratic(gram, weighted['covariance'], .2/np.sqrt(8))
    # Compare quadratic formula with its literal finite pair distribution.
    transformed = np.einsum('lnd,bqnd->bql', j, fields)
    direct = float(weighted['weights'] @ (.5 * np.sum(transformed**2, axis=-1).mean(axis=1)))
    semantic, nuisance = rng.normal(size=(32,8)), rng.normal(size=(32,8))
    metric = fit_metric_initializer(semantic, nuisance)['metric']
    x, y, margin = make_relational_scenes(32, seed=17)
    checks = {
        'marginal_plus_probability_difference_max':float(np.max(np.abs(uniform['plus_probability']-weighted['plus_probability']))),
        'arm_energy_error_max':float(np.max(np.abs(np.sum(fields**2,axis=(-1,-2))-.2**2))),
        'group1_displacement_sum_max':float(np.abs(fields[:,:,:4].sum(axis=2)).max()),
        'group2_displacement_sum_max':float(np.abs(fields[:,:,4:].sum(axis=2)).max()),
        'quadratic_identity_absolute_error':abs(predicted-direct),
        'metric_min_eigenvalue':float(np.linalg.eigvalsh(metric).min()),
        'oracle_class_counts':np.bincount(y, minlength=2).tolist(),
    }
    assert checks['marginal_plus_probability_difference_max'] < 1e-12
    assert checks['arm_energy_error_max'] < 1e-12
    assert checks['group1_displacement_sum_max'] < 1e-12
    assert checks['group2_displacement_sum_max'] < 1e-12
    assert checks['quadratic_identity_absolute_error'] < 1e-12
    assert checks['metric_min_eigenvalue'] >= 1-1e-10
    result = {'evidence_level':'engineering_only','scientific_claims':[],
              'description':'Synthetic matrix/oracle identities only; no trained model or project data loaded.',
              'checks':checks}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'kernel_smoke.json').write_text(json.dumps(result, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
