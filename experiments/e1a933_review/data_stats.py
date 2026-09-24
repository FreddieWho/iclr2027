"""Paired cluster bootstrap: resample parents, retain every state/arm together."""
import numpy as np


def parent_bootstrap(values, parents, n_boot=2000, seed=933):
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    unique, inv = np.unique(parents, return_inverse=True)
    sums = np.zeros((len(unique), values.shape[1]))
    np.add.at(sums, inv, values)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(unique), size=(n_boot, len(unique)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)[:, None]
    return {"estimate": values.mean(0).tolist(),
            "ci95": np.quantile(boot, [.025, .975], axis=0).T.tolist(),
            "n_parents": len(unique), "n_rows": len(values),
            "bootstrap_unit": "parent", "n_boot": n_boot, "seed": seed}
